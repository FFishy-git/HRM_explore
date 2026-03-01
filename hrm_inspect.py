"""Eval harness for inspecting HRM hidden states via V2 hooks.

Preserves all evaluate.py functionality (full ACT eval with metrics) and adds
hook-based residual stream inspection.

Usage:
    from hrm_inspect import load_model, load_sudoku_batch, collect_residual_streams

    model = load_model("/nemo-workspace/inf-evolve/hrm/checkpoints/step_166310")
    batches = load_sudoku_batch("/nemo-workspace/inf-evolve/hrm/data/sudoku-hard-full")

    hook, storage = collect_residual_streams()
    remove = model.register_hook_H(hook)

    batch = next(batches)
    batch = {k: v.cuda() for k, v in batch.items()}
    results = run_act_loop(model, batch)
    remove()
"""
import os
import json
import yaml
from typing import Dict, List, Optional, Tuple, Iterator

import torch
import torch.nn.functional as F

from models.hrm.hrm_act_v2 import (
    HierarchicalReasoningModel_ACTV1Config,
    HierarchicalReasoningModel_ACTV1_Inner,
    HierarchicalReasoningModel_ACTV1,
    HookContext,
    HookCallback,
)
from models.losses import ACTLossHead, IGNORE_LABEL_ID
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata


# ──────────────────────────── Model loading ─────────────────────────────────

def _resolve_data_path(checkpoint_path: str, data_path: Optional[str]) -> str:
    """Resolve data_path from checkpoint config if not provided."""
    if data_path is not None:
        return data_path
    ckpt_dir = os.path.dirname(checkpoint_path)
    with open(os.path.join(ckpt_dir, "all_config.yaml"), "r") as f:
        return yaml.safe_load(f)["data_path"]


def _load_config_and_state(
    checkpoint_path: str,
    data_path: str,
    device: str,
) -> Tuple[dict, dict, dict]:
    """Load arch config, model config dict, and state dict."""
    ckpt_dir = os.path.dirname(checkpoint_path)
    with open(os.path.join(ckpt_dir, "all_config.yaml"), "r") as f:
        pretrain_config = yaml.safe_load(f)

    arch_config = pretrain_config["arch"]

    # Read metadata from test set
    with open(os.path.join(data_path, "test", "dataset.json"), "r") as f:
        metadata = json.load(f)

    model_cfg = {
        k: v for k, v in arch_config.items()
        if k not in ("name", "loss")
    }
    model_cfg.update(
        batch_size=1,  # doesn't matter in eval mode
        vocab_size=metadata["vocab_size"],
        seq_len=metadata["seq_len"],
        num_puzzle_identifiers=metadata["num_puzzle_identifiers"],
    )

    # Load + strip torch.compile prefix
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cleaned = {}
    for k, v in state_dict.items():
        key = k.removeprefix("_orig_mod.")
        cleaned[key] = v

    return arch_config, model_cfg, cleaned


def load_model(
    checkpoint_path: str,
    data_path: Optional[str] = None,
    device: str = "cuda",
) -> HierarchicalReasoningModel_ACTV1_Inner:
    """Load the Inner model from a checkpoint (for direct ACT loop with hooks).

    Args:
        checkpoint_path: Path to checkpoint file (e.g. .../step_166310)
        data_path: Path to dataset root. If None, read from config.
        device: Device to load onto.

    Returns:
        The _Inner model with hook support.
    """
    data_path = _resolve_data_path(checkpoint_path, data_path)
    _, model_cfg, state_dict = _load_config_and_state(checkpoint_path, data_path, device)

    with torch.device(device):
        inner = HierarchicalReasoningModel_ACTV1_Inner(
            HierarchicalReasoningModel_ACTV1Config(**model_cfg)
        )

    # Strip ACTLossHead.model.inner. prefix
    inner_state = {}
    prefix = "model.inner."
    for k, v in state_dict.items():
        if k.startswith(prefix):
            inner_state[k[len(prefix):]] = v

    inner.load_state_dict(inner_state, strict=False)
    inner.eval()
    return inner


def load_full_model(
    checkpoint_path: str,
    data_path: Optional[str] = None,
    device: str = "cuda",
) -> ACTLossHead:
    """Load full ACTLossHead-wrapped model (for evaluate() compatibility).

    Args:
        checkpoint_path: Path to checkpoint file.
        data_path: Path to dataset root. If None, read from config.
        device: Device to load onto.

    Returns:
        ACTLossHead wrapping HierarchicalReasoningModel_ACTV1 with V2 hooks.
    """
    data_path = _resolve_data_path(checkpoint_path, data_path)
    arch_config, model_cfg, state_dict = _load_config_and_state(checkpoint_path, data_path, device)

    loss_cfg = arch_config.get("loss", {})
    loss_type = loss_cfg.get("loss_type", "softmax_cross_entropy")

    with torch.device(device):
        act_model = HierarchicalReasoningModel_ACTV1(model_cfg)
        model = ACTLossHead(act_model, loss_type=loss_type)
        model.load_state_dict(state_dict, strict=False)
        model.eval()

    return model


# ──────────────────────────── Data loading ──────────────────────────────────

def load_sudoku_batch(
    data_path: str,
    batch_size: int = 32,
    split: str = "test",
) -> Iterator[Dict[str, torch.Tensor]]:
    """Yield batches from a sudoku dataset split.

    Yields:
        Dict with keys: inputs, labels, puzzle_identifiers (torch tensors)
    """
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=0,
        dataset_path=data_path,
        global_batch_size=batch_size,
        test_set_mode=(split == "test"),
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
    ), split=split)

    for _set_name, batch, _global_batch_size in dataset:
        yield batch


def create_dataloader(
    data_path: str,
    split: str,
    batch_size: int,
    test_set_mode: bool = True,
) -> Tuple[torch.utils.data.DataLoader, PuzzleDatasetMetadata]:
    """Create a DataLoader mirroring pretrain.py's create_dataloader for single GPU."""
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=0,
        dataset_path=data_path,
        global_batch_size=batch_size,
        test_set_mode=test_set_mode,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
    ), split=split)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True,
    )
    return dataloader, dataset.metadata


# ──────────────────────────── Evaluation (mirrors evaluate.py) ──────────────

def evaluate(
    model: ACTLossHead,
    data_path: str,
    batch_size: int = 2304,
    save_outputs: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Run full evaluation mirroring evaluate.py functionality.

    Args:
        model: Full ACTLossHead-wrapped model (from load_full_model).
        data_path: Path to dataset root.
        batch_size: Global batch size.
        save_outputs: List of output keys to save. Default: logits, q_halt_logits, etc.
        save_path: Directory to save prediction tensors. None = don't save.

    Returns:
        Dict mapping set_name -> {metric_name: value}.
    """
    if save_outputs is None:
        save_outputs = ["inputs", "labels", "puzzle_identifiers", "logits",
                        "q_halt_logits", "q_continue_logits"]

    eval_loader, eval_metadata = create_dataloader(
        data_path, "test", batch_size=batch_size, test_set_mode=True,
    )

    set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}
    all_preds: Dict[str, List[torch.Tensor]] = {}

    metric_keys: List[str] = []
    metric_values = None
    metric_global_batch_size = [0] * len(set_ids)

    with torch.inference_mode():
        for set_name, batch, global_batch_size in eval_loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = model.initial_carry(batch)

            while True:
                carry, _, metrics, preds, all_finish = model(
                    carry=carry, batch=batch, return_keys=save_outputs,
                )
                if all_finish:
                    break

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in save_outputs:
                        all_preds.setdefault(k, [])
                        all_preds[k].append(v.cpu())

            del carry, preds, batch, all_finish

            set_id = set_ids[set_name]
            if metric_values is None:
                metric_keys = list(sorted(metrics.keys()))
                metric_values = torch.zeros(
                    (len(set_ids), len(metric_keys)), dtype=torch.float32, device="cuda",
                )
            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])
            metric_global_batch_size[set_id] += global_batch_size

    # Save predictions
    if save_path and all_preds:
        os.makedirs(save_path, exist_ok=True)
        concatenated = {k: torch.cat(v, dim=0) for k, v in all_preds.items()}
        torch.save(concatenated, os.path.join(save_path, "all_preds.pt"))

    # Reduce metrics
    if metric_values is not None:
        reduced = metric_values.cpu().numpy()
        result = {}
        for set_name, set_id in set_ids.items():
            metrics_dict = {
                metric_keys[i]: reduced[set_id, i] for i in range(len(metric_keys))
            }
            count = max(metrics_dict.pop("count", 1), 1)
            result[set_name] = {k: v / count for k, v in metrics_dict.items()}
        return result

    return {}


# ──────────────────────────── Pre-built hooks ───────────────────────────────

def collect_residual_streams(
    clone: bool = True,
) -> Tuple[HookCallback, List[HookContext]]:
    """Collect all HookContext objects (containing z_H and z_L).

    Args:
        clone: If True, clone z_H/z_L tensors to prevent aliasing.

    Returns:
        (callback, storage_list) — register the callback, read storage after run.
    """
    storage: List[HookContext] = []

    def hook(ctx: HookContext):
        if clone:
            ctx = HookContext(
                act_step=ctx.act_step,
                h_cycle=ctx.h_cycle,
                l_cycle=ctx.l_cycle,
                is_grad_step=ctx.is_grad_step,
                z_H=ctx.z_H.clone(),
                z_L=ctx.z_L.clone(),
            )
        storage.append(ctx)

    return hook, storage


def track_prediction_evolution(
    model: HierarchicalReasoningModel_ACTV1_Inner,
    labels: torch.Tensor,
) -> Tuple[HookCallback, List[Dict]]:
    """Track accuracy and entropy at each ACT step (on H_level grad-step hooks).

    Args:
        model: The _Inner model (needed for lm_head access).
        labels: Ground truth labels tensor [batch, seq_len].

    Returns:
        (callback, results_list) — register as H hook, read results after run.
    """
    results: List[Dict] = []
    puzzle_emb_len = model.puzzle_emb_len

    def hook(ctx: HookContext):
        if not ctx.is_grad_step:
            return
        with torch.no_grad():
            logits = model.lm_head(ctx.z_H)[:, puzzle_emb_len:]
            preds = logits.argmax(dim=-1)
            mask = labels != IGNORE_LABEL_ID
            correct = (preds == labels) & mask
            count = mask.sum().clamp_min(1)
            accuracy = correct.sum().float() / count.float()
            exact_accuracy = (correct.sum(-1) == mask.sum(-1)).float().mean()

            probs = F.softmax(logits.float(), dim=-1)
            entropy = -(probs * probs.clamp_min(1e-10).log()).sum(-1).mean()

            results.append({
                "act_step": ctx.act_step,
                "accuracy": accuracy.item(),
                "exact_accuracy": exact_accuracy.item(),
                "entropy": entropy.item(),
                "predictions": preds.cpu(),
            })

    return hook, results


def compute_residual_norms() -> Tuple[HookCallback, List[Dict]]:
    """Track per-position L2 norms of z_H and z_L with full statistics.

    Records mean/std/min/max of the per-position norms across the batch,
    plus cosine similarity between z_H and z_L.

    Returns:
        (callback, norms_list) — register as both H and L hook.
    """
    norms: List[Dict] = []

    def hook(ctx: HookContext):
        # Per-position norms: [batch, seq_len]
        z_H_norms = ctx.z_H.float().norm(dim=-1)
        z_L_norms = ctx.z_L.float().norm(dim=-1)

        # Per-position cosine sim: [batch, seq_len]
        cos_sim = F.cosine_similarity(ctx.z_H.float(), ctx.z_L.float(), dim=-1)

        norms.append({
            "act_step": ctx.act_step,
            "h_cycle": ctx.h_cycle,
            "l_cycle": ctx.l_cycle,
            "is_grad_step": ctx.is_grad_step,
            # z_H norm stats
            "z_H_norm_mean": z_H_norms.mean().item(),
            "z_H_norm_std": z_H_norms.std().item(),
            "z_H_norm_min": z_H_norms.min().item(),
            "z_H_norm_max": z_H_norms.max().item(),
            # z_L norm stats
            "z_L_norm_mean": z_L_norms.mean().item(),
            "z_L_norm_std": z_L_norms.std().item(),
            "z_L_norm_min": z_L_norms.min().item(),
            "z_L_norm_max": z_L_norms.max().item(),
            # cosine similarity stats
            "cosine_sim_mean": cos_sim.mean().item(),
            "cosine_sim_std": cos_sim.std().item(),
            "cosine_sim_min": cos_sim.min().item(),
            "cosine_sim_max": cos_sim.max().item(),
        })

    return hook, norms


# ──────────────────────────── ACT inference loop ────────────────────────────

def run_act_loop(
    model: HierarchicalReasoningModel_ACTV1_Inner,
    batch: Dict[str, torch.Tensor],
    num_steps: Optional[int] = None,
) -> List[Dict[str, torch.Tensor]]:
    """Run the ACT loop manually on the _Inner model.

    This bypasses the ACT wrapper and loss head, giving direct control
    over the number of reasoning steps. Hooks registered on the model fire.

    Args:
        model: The _Inner model with hooks registered.
        batch: Dict with inputs, labels, puzzle_identifiers on device.
        num_steps: Number of ACT steps to run. Defaults to halt_max_steps.

    Returns:
        List of per-step dicts with 'logits', 'predictions', q logits.
    """
    if num_steps is None:
        num_steps = model.config.halt_max_steps

    batch_size = batch["inputs"].shape[0]
    device = batch["inputs"].device
    results = []

    with torch.no_grad(), torch.device(device):
        carry = model.empty_carry(batch_size)
        carry = model.reset_carry(
            torch.ones(batch_size, dtype=torch.bool, device=device),
            carry,
        )

        for step in range(num_steps):
            carry, logits, (q_halt, q_continue) = model(carry, batch, act_step=step)
            preds = logits.argmax(dim=-1)
            results.append({
                "logits": logits,
                "predictions": preds,
                "q_halt_logits": q_halt,
                "q_continue_logits": q_continue,
            })

    return results


# ──────────────────────────── CLI entry point ───────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HRM Inspection & Evaluation Tool")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file (e.g. .../step_166310)")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to dataset root. Reads from config if not provided.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=None,
                        help="Number of ACT steps (for inspect mode).")
    parser.add_argument("--mode", choices=["inspect", "evaluate"], default="inspect",
                        help="'inspect': hook-based analysis on one batch. "
                             "'evaluate': full test set evaluation mirroring evaluate.py.")
    parser.add_argument("--save-path", type=str, default=None,
                        help="Directory to save predictions (evaluate mode).")
    args = parser.parse_args()

    data_path = _resolve_data_path(args.checkpoint, args.data_path)

    if args.mode == "evaluate":
        print("Loading full model (ACTLossHead)...")
        full_model = load_full_model(args.checkpoint, data_path=data_path)

        # Register norm hooks on the inner model for benchmarking
        inner: HierarchicalReasoningModel_ACTV1_Inner = full_model.model.inner  # type: ignore[attr-defined]
        norm_hook, norm_results = compute_residual_norms()
        inner.register_hook_H(norm_hook)
        inner.register_hook_L(norm_hook)

        print(f"Evaluating on {data_path} with batch_size={args.batch_size}...")
        metrics = evaluate(
            full_model, data_path, batch_size=args.batch_size,
            save_path=args.save_path,
        )

        print("\n=== Evaluation Metrics ===")
        for set_name, set_metrics in metrics.items():
            print(f"\n  [{set_name}]")
            for k, v in set_metrics.items():
                print(f"    {k}: {v:.6f}")

        # Print residual norm summary per ACT step (grad-step H hooks only)
        print("\n=== Residual Norm Benchmark (per ACT step, H grad-step) ===")
        print(f"  {'step':>4}  {'z_H mean':>9} {'z_H std':>8} {'z_H min':>8} {'z_H max':>8}"
              f"  {'z_L mean':>9} {'z_L std':>8} {'z_L min':>8} {'z_L max':>8}"
              f"  {'cos mean':>9} {'cos std':>8} {'cos min':>8} {'cos max':>8}")
        for n in norm_results:
            if n["is_grad_step"] and n["h_cycle"] > 0:  # final H hook per ACT step
                print(f"  {n['act_step']:>4}"
                      f"  {n['z_H_norm_mean']:>9.4f} {n['z_H_norm_std']:>8.4f} {n['z_H_norm_min']:>8.4f} {n['z_H_norm_max']:>8.4f}"
                      f"  {n['z_L_norm_mean']:>9.4f} {n['z_L_norm_std']:>8.4f} {n['z_L_norm_min']:>8.4f} {n['z_L_norm_max']:>8.4f}"
                      f"  {n['cosine_sim_mean']:>9.4f} {n['cosine_sim_std']:>8.4f} {n['cosine_sim_min']:>8.4f} {n['cosine_sim_max']:>8.4f}")

    else:  # inspect mode
        print("Loading model (inner)...")
        model = load_model(args.checkpoint, data_path=data_path)

        print(f"Loading data from {data_path}...")
        batches = load_sudoku_batch(data_path, batch_size=args.batch_size)
        batch = next(batches)
        batch = {k: v.cuda() for k, v in batch.items()}

        # Register hooks
        pred_hook, pred_results = track_prediction_evolution(model, batch["labels"])
        norm_hook, norm_results = compute_residual_norms()

        model.register_hook_H(pred_hook)
        model.register_hook_H(norm_hook)
        model.register_hook_L(norm_hook)

        num_steps = args.num_steps or model.config.halt_max_steps
        print(f"Running {num_steps} ACT steps on batch of {batch['inputs'].shape[0]} puzzles...")
        results = run_act_loop(model, batch, num_steps=num_steps)

        print("\n=== Prediction Evolution ===")
        for r in pred_results:
            print(f"  ACT step {r['act_step']}: accuracy={r['accuracy']:.4f}, "
                  f"exact={r['exact_accuracy']:.4f}, entropy={r['entropy']:.4f}")

        print("\n=== Residual Norm Benchmark (all hooks) ===")
        print(f"  {'step':>4} {'h_cyc':>5} {'l_cyc':>5} {'grad':>4}"
              f"  {'z_H mean':>9} {'z_H std':>8} {'z_H min':>8} {'z_H max':>8}"
              f"  {'z_L mean':>9} {'z_L std':>8} {'z_L min':>8} {'z_L max':>8}"
              f"  {'cos mean':>9} {'cos std':>8} {'cos min':>8} {'cos max':>8}")
        for n in norm_results:
            print(f"  {n['act_step']:>4} {n['h_cycle']:>5} {n['l_cycle']:>5} {'Y' if n['is_grad_step'] else 'N':>4}"
                  f"  {n['z_H_norm_mean']:>9.4f} {n['z_H_norm_std']:>8.4f} {n['z_H_norm_min']:>8.4f} {n['z_H_norm_max']:>8.4f}"
                  f"  {n['z_L_norm_mean']:>9.4f} {n['z_L_norm_std']:>8.4f} {n['z_L_norm_min']:>8.4f} {n['z_L_norm_max']:>8.4f}"
                  f"  {n['cosine_sim_mean']:>9.4f} {n['cosine_sim_std']:>8.4f} {n['cosine_sim_min']:>8.4f} {n['cosine_sim_max']:>8.4f}")
