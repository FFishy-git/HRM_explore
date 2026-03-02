"""Residual stream probing: train a probing MLP to measure H-level information gain.

Extracts z_H (before H-level) and z_H* (after H-level) residual streams via hooks,
then trains a lightweight MLP probe to decode digit predictions from concatenated
residual streams. Compares probe accuracy on z_H vs z_H* to quantify information gain.

Usage (hooks only):
    from probe_train import create_probe_hooks, ProbeData

    hooks, storage = create_probe_hooks(skip_steps=2, to_cpu=True)
    remove_l = model.register_hook_L(hooks['L'])
    remove_h = model.register_hook_H(hooks['H'])

Usage (training):
    python probe_train.py --checkpoint <path> --data <path> --mode train
"""
from typing import Dict, List, Optional
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.hrm.hrm_act_v2 import HookContext, HookCallback


@dataclass
class ProbeData:
    """Collected residual stream data for a single ACT step."""
    act_step: int
    z_H_before: torch.Tensor   # [batch, seq+puzzle_emb_len, hidden_size]
    z_H_after: torch.Tensor    # [batch, seq+puzzle_emb_len, hidden_size]
    z_L: torch.Tensor          # [batch, seq+puzzle_emb_len, hidden_size]
    input_embeddings: torch.Tensor  # [batch, seq+puzzle_emb_len, hidden_size]


def create_probe_hooks(
    skip_steps: int = 2,
    to_cpu: bool = True,
) -> tuple[Dict[str, HookCallback], List[ProbeData]]:
    """Create L and H hook callbacks for residual stream extraction.

    The L-hook captures z_H_before (carry-in z_H before H-level) and z_L
    at the grad-step (last L/H cycle). The H-hook captures z_H_after
    (z_H* after H-level) at the grad-step.

    Args:
        skip_steps: Skip ACT steps < skip_steps (early steps are unstable).
        to_cpu: If True, detach and move tensors to CPU to avoid OOM.

    Returns:
        (hooks_dict, storage) where hooks_dict has 'L' and 'H' keys,
        and storage accumulates ProbeData per qualifying ACT step.
    """
    storage: List[ProbeData] = []
    # L-hook fires before H-hook within the same ACT step; buffer partial data
    _pending: Dict[int, Dict[str, torch.Tensor]] = {}

    def _detach(t: torch.Tensor) -> torch.Tensor:
        d = t.detach()
        return d.cpu() if to_cpu else d

    def l_hook(ctx: HookContext) -> None:
        """Capture z_H_before and z_L at the grad-step L-hook."""
        if not ctx.is_grad_step:
            return
        if ctx.act_step < skip_steps:
            return
        # ctx.z_H here is z_H BEFORE H-level (the carry-in)
        # ctx.z_L here is z_L AFTER the final L-level call
        _pending[ctx.act_step] = {
            'z_H_before': _detach(ctx.z_H),
            'z_L': _detach(ctx.z_L),
            'input_embeddings': _detach(ctx.input_embeddings) if ctx.input_embeddings is not None else torch.zeros_like(ctx.z_H.detach()),
        }

    def h_hook(ctx: HookContext) -> None:
        """Capture z_H_after at the grad-step H-hook and assemble ProbeData."""
        if not ctx.is_grad_step:
            return
        if ctx.act_step < skip_steps:
            return
        # ctx.z_H here is z_H AFTER H-level (z_H*)
        pending = _pending.pop(ctx.act_step, None)
        if pending is None:
            return
        storage.append(ProbeData(
            act_step=ctx.act_step,
            z_H_before=pending['z_H_before'],
            z_H_after=_detach(ctx.z_H),
            z_L=pending['z_L'],
            input_embeddings=pending['input_embeddings'],
        ))

    hooks: Dict[str, HookCallback] = {'L': l_hook, 'H': h_hook}
    return hooks, storage


# ──────────────────────── Training helpers ────────────────────────────────


def _run_act_loop_with_hooks(
    model: "HierarchicalReasoningModel_ACTV1_Inner",
    batch: Dict[str, torch.Tensor],
    hooks_dict: Dict[str, HookCallback],
    storage: List[ProbeData],
    num_steps: int,
) -> None:
    """Run the ACT loop on the frozen Inner model with hooks registered.

    Populates *storage* with ProbeData entries (one per qualifying ACT step).
    """
    storage.clear()
    device = batch["inputs"].device
    batch_size = batch["inputs"].shape[0]

    remove_l = model.register_hook_L(hooks_dict['L'])
    remove_h = model.register_hook_H(hooks_dict['H'])
    try:
        with torch.no_grad():
            carry = model.empty_carry(batch_size)
            carry = model.reset_carry(
                torch.ones(batch_size, dtype=torch.bool, device=device),
                carry,
            )
            for step in range(num_steps):
                carry, _logits, _q = model(carry, batch, act_step=step)
    finally:
        remove_l()
        remove_h()


def _form_probe_inputs(
    data: ProbeData,
    puzzle_emb_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Form probe inputs by concatenating residual streams and stripping puzzle prefix.

    Returns:
        (before_input, after_input) each of shape [batch, seq_len, 3*hidden_size]
    """
    # Strip puzzle prefix positions
    z_H = data.z_H_before[:, puzzle_emb_len:].to(device)
    z_H_star = data.z_H_after[:, puzzle_emb_len:].to(device)
    z_L = data.z_L[:, puzzle_emb_len:].to(device)
    inp_emb = data.input_embeddings[:, puzzle_emb_len:].to(device)

    before_input = torch.cat([z_H, z_L, inp_emb], dim=-1)
    after_input = torch.cat([z_H_star, z_L, inp_emb], dim=-1)
    return before_input, after_input


def _compute_probe_loss(
    probe: "ProbingMLP",
    inputs: torch.Tensor,
    labels: torch.Tensor,
    blank_mask: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Compute cross-entropy loss and accuracy on blank cells only.

    Args:
        probe: The probing MLP.
        inputs: [batch, seq, d_in] concatenated residual streams.
        labels: [batch, seq] ground-truth token IDs (digits are tokens 2-10).
        blank_mask: [batch, seq] boolean mask where inputs == 1 (blank cells).

    Returns:
        (loss, accuracy) where loss is a scalar tensor and accuracy is a float.
    """
    logits = probe(inputs.float())  # [batch, seq, 9]
    # Remap label tokens 2-10 -> classes 0-8
    target = labels - 2

    logits_flat = logits[blank_mask]   # [num_blank, 9]
    target_flat = target[blank_mask]   # [num_blank]

    if logits_flat.numel() == 0:
        return torch.tensor(0.0, device=inputs.device, requires_grad=True), 0.0

    loss = F.cross_entropy(logits_flat, target_flat.long())

    with torch.no_grad():
        preds = logits_flat.argmax(dim=-1)
        acc = (preds == target_flat).float().mean().item()

    return loss, acc


# ──────────────────────── Training loop ───────────────────────────────────


def train_probe(args) -> None:
    """Main probe training loop."""
    from hrm_inspect import load_model, load_sudoku_batch
    from models.probing import ProbingMLP

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load frozen HRM ---
    print(f"Loading HRM checkpoint from {args.checkpoint} ...")
    model = load_model(args.checkpoint, data_path=args.data, device=str(device))
    model.eval()
    model.requires_grad_(False)
    puzzle_emb_len = model.puzzle_emb_len
    hidden_size = model.config.hidden_size
    print(f"  hidden_size={hidden_size}, puzzle_emb_len={puzzle_emb_len}")

    # --- Create probe ---
    probe = ProbingMLP(hidden_size=hidden_size).to(device)
    if args.probe_checkpoint:
        print(f"Resuming probe from {args.probe_checkpoint}")
        ckpt = torch.load(args.probe_checkpoint, map_location=device, weights_only=True)
        probe.load_state_dict(ckpt["state_dict"])

    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # --- Optional wandb ---
    use_wandb = args.wandb_project is not None
    if use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=vars(args),
            settings=wandb.Settings(_disable_stats=True),
        )

    # --- Create hooks (keep tensors on GPU, detached) ---
    hooks_dict, storage = create_probe_hooks(skip_steps=args.skip_act_steps, to_cpu=False)

    # --- Training ---
    num_steps = args.act_steps if args.act_steps else model.config.halt_max_steps
    print(f"Training for {args.epochs} epochs, act_steps={num_steps}, skip_steps={args.skip_act_steps}")

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_acc_before = 0.0
        epoch_acc_after = 0.0
        num_batches = 0

        for batch in load_sudoku_batch(args.data, batch_size=args.batch_size, split="train"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Run ACT loop with hooks
            _run_act_loop_with_hooks(model, batch, hooks_dict, storage, num_steps)

            if not storage:
                continue

            # Labels and blank mask (on the original seq_len, no puzzle prefix)
            labels = batch["labels"]     # [batch, seq_len]
            blank_mask = batch["inputs"] == 1  # [batch, seq_len]

            # Accumulate loss across all collected ACT steps
            total_loss = torch.tensor(0.0, device=device)
            step_count = 0
            batch_acc_before = 0.0
            batch_acc_after = 0.0

            for probe_data in storage:
                before_input, after_input = _form_probe_inputs(probe_data, puzzle_emb_len, device)

                loss_before, acc_before = _compute_probe_loss(probe, before_input, labels, blank_mask)
                loss_after, acc_after = _compute_probe_loss(probe, after_input, labels, blank_mask)

                total_loss = total_loss + loss_before + loss_after
                batch_acc_before += acc_before
                batch_acc_after += acc_after
                step_count += 1

            if step_count > 0:
                avg_loss = total_loss / step_count
                optimizer.zero_grad()
                avg_loss.backward()
                optimizer.step()

                epoch_loss += avg_loss.item()
                epoch_acc_before += batch_acc_before / step_count
                epoch_acc_after += batch_acc_after / step_count
                num_batches += 1

        # Epoch summary
        if num_batches > 0:
            avg_epoch_loss = epoch_loss / num_batches
            avg_epoch_acc_before = epoch_acc_before / num_batches
            avg_epoch_acc_after = epoch_acc_after / num_batches
            print(
                f"Epoch {epoch+1}/{args.epochs}  "
                f"loss={avg_epoch_loss:.4f}  "
                f"acc(z_H)={avg_epoch_acc_before:.4f}  "
                f"acc(z_H*)={avg_epoch_acc_after:.4f}  "
                f"delta={avg_epoch_acc_after - avg_epoch_acc_before:.4f}"
            )
            if use_wandb:
                import wandb
                wandb.log({
                    "train/loss": avg_epoch_loss,
                    "train/acc_z_H": avg_epoch_acc_before,
                    "train/acc_z_H_star": avg_epoch_acc_after,
                    "train/acc_delta": avg_epoch_acc_after - avg_epoch_acc_before,
                }, step=epoch)
        else:
            print(f"Epoch {epoch+1}/{args.epochs}  no data collected")

    # Save final probe checkpoint
    _save_probe(probe, args.output if hasattr(args, 'output') and args.output else "probe_checkpoint.pt")

    if use_wandb:
        import wandb
        wandb.finish()


def _save_probe(probe: "ProbingMLP", path: str) -> None:
    """Save probe state_dict and config."""
    torch.save({
        "state_dict": probe.state_dict(),
        "config": {
            "hidden_size": probe.hidden_size,
            "hidden_mult": probe.hidden_mult,
            "num_classes": probe.num_classes,
        },
    }, path)
    print(f"Saved probe checkpoint to {path}")


# ──────────────────────── CLI ─────────────────────────────────────────────


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Residual Stream Probe Training")

    # Required
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to frozen HRM checkpoint")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to dataset root")

    # Mode
    parser.add_argument("--mode", choices=["train", "eval"], default="train",
                        help="'train': train probe. 'eval': evaluate probe.")

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=1e-2,
                        help="Weight decay (default: 1e-2)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")

    # ACT configuration
    parser.add_argument("--act-steps", type=int, default=None,
                        help="Number of ACT steps (default: model's halt_max_steps)")
    parser.add_argument("--skip-act-steps", type=int, default=2,
                        help="Skip ACT steps < this value (default: 2)")

    # Probe checkpoint
    parser.add_argument("--probe-checkpoint", type=str, default=None,
                        help="Path to saved probe checkpoint (resume training or eval)")
    parser.add_argument("--output", type=str, default="probe_checkpoint.pt",
                        help="Output path for probe checkpoint (default: probe_checkpoint.pt)")

    # Wandb
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="Wandb project name (enables wandb logging)")
    parser.add_argument("--wandb-entity", type=str, default="LoopTF-4-CSPs",
                        help="Wandb entity")
    parser.add_argument("--wandb-name", type=str, default=None,
                        help="Wandb run name")

    args = parser.parse_args()

    if args.mode == "train":
        train_probe(args)
    elif args.mode == "eval":
        print("Eval mode not yet implemented (see US-005)")
        raise SystemExit(1)
