---
id: 2
title: Evaluation with z_H Projection to lm_head Subspace
filename: 002_eval_zh_projection_to_lm_head_subspace.md
created: 2026-03-02T09:39:36Z
depends_on: [1]
all_ancestors: [1]
---

# Evaluation with z_H Projection to lm_head Subspace

## Summary

Plan for evaluating HRM performance when z_H is projected onto the lm_head row space after every H_level call. Finding #1 showed z_H degenerates into this 11-dimensional subspace for actual token positions — this experiment tests whether the ~2% of dimensions (11/512) that lm_head can read already carry all the useful information, or whether the remaining ~98% of dimensions contribute to computation across ACT steps via attention and input injection.

## Details

### Motivation

From #1, we observed that `proj/z_H/lm_head/seq/ratio_mean` is high — z_H's token positions live mostly in the lm_head row space. The question is: **if we force z_H to stay in this subspace after each H loop, does performance degrade?**

- If performance is preserved: the model's H-level computation is already effectively operating within the lm_head subspace, and the extra dimensions are unused/noise.
- If performance degrades: the model uses the full 512-dim space for intermediate H-level computation (e.g., attention reads from orthogonal directions), and only projects down to lm_head subspace at the final step. This would indicate the degeneration is a natural convergence behavior, not a structural limitation.

### Approach: New Script `hrm_eval_projected.py`

Create a standalone evaluation script that modifies the ACT inner loop to project z_H after each H_level call. This avoids modifying the core model code.

**Why not use hooks?** The V2 hook system is observation-only — hooks receive `HookContext` but cannot modify `z_H` or `z_L`. We need to intervene in the forward pass.

**Why not modify the model code?** We want to keep the original model untouched for reproducibility. A separate eval script that reimplements the inner loop with projection is cleaner.

### Implementation Plan

#### Step 1: Projection utility

Reuse `_build_basis()` from `hrm_inspect.py` to compute the orthonormal basis for lm_head.weight row space:

```python
from hrm_inspect import _build_basis

def build_projector(model):
    """Build projection matrix P = U @ U^T for lm_head row space."""
    U_basis, rank = _build_basis(model.lm_head.weight)  # [hidden_size, rank]
    U_basis = U_basis.to(model.lm_head.weight.device)
    return U_basis, rank

def project_to_lm_head(z, U_basis, positions="seq", puzzle_emb_len=1):
    """Project z onto lm_head subspace.

    Args:
        z: [batch, seq_total, hidden_size]
        U_basis: [hidden_size, rank] orthonormal basis
        positions: "all" projects all positions, "seq" only projects token positions
                   (puzzle_emb_len:), "seq_and_pos0" projects token + pos0

    Returns:
        z with specified positions projected onto lm_head subspace
    """
    z_out = z.clone()
    if positions == "all":
        # z_proj = z @ U @ U^T
        z_out = (z @ U_basis) @ U_basis.T
    elif positions == "seq":
        z_seq = z[:, puzzle_emb_len:]
        z_out[:, puzzle_emb_len:] = (z_seq @ U_basis) @ U_basis.T
    elif positions == "seq_and_pos0":
        z_seq = z[:, puzzle_emb_len:]
        z_out[:, puzzle_emb_len:] = (z_seq @ U_basis) @ U_basis.T
        z_p0 = z[:, 0:1]
        z_out[:, 0:1] = (z_p0 @ U_basis) @ U_basis.T
    return z_out
```

#### Step 2: Modified ACT inner loop

Reimplement `HierarchicalReasoningModel_ACTV1_Inner.forward()` with projection injected after each H_level call. This mirrors `hrm_act_v2.py:243-287` but adds the projection step:

```python
def forward_with_projection(model, carry, batch, act_step, U_basis, project_positions="seq"):
    """Modified inner forward that projects z_H after each H_level call.

    Mirrors hrm_act_v2.py:243-287 but adds:
        z_H = project_to_lm_head(z_H, U_basis, project_positions)
    after every H_level call (both no-grad and grad steps).
    """
    seq_info = dict(
        cos_sin=model.rotary_emb() if hasattr(model, "rotary_emb") else None,
    )
    input_embeddings = model._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
    puzzle_emb_len = model.puzzle_emb_len

    with torch.no_grad():
        z_H, z_L = carry.z_H, carry.z_L

        for _H_step in range(model.config.H_cycles):
            for _L_step in range(model.config.L_cycles):
                if not ((_H_step == model.config.H_cycles - 1) and (_L_step == model.config.L_cycles - 1)):
                    z_L = model.L_level(z_L, z_H + input_embeddings, **seq_info)
                    model._fire_L(HookContext(act_step, _H_step, _L_step, False, z_H, z_L))

            if not (_H_step == model.config.H_cycles - 1):
                z_H = model.H_level(z_H, z_L, **seq_info)
                # >>> PROJECTION INTERVENTION <<<
                z_H = project_to_lm_head(z_H, U_basis, project_positions, puzzle_emb_len)
                model._fire_H(HookContext(act_step, _H_step, model.config.L_cycles - 1, False, z_H, z_L))

    # 1-step grad (no-grad in eval, but keep structure identical)
    z_L = model.L_level(z_L, z_H + input_embeddings, **seq_info)
    model._fire_L(HookContext(act_step, model.config.H_cycles - 1, model.config.L_cycles - 1, True, z_H, z_L))

    z_H = model.H_level(z_H, z_L, **seq_info)
    # >>> PROJECTION INTERVENTION <<<
    z_H = project_to_lm_head(z_H, U_basis, project_positions, puzzle_emb_len)
    model._fire_H(HookContext(act_step, model.config.H_cycles - 1, model.config.L_cycles - 1, True, z_H, z_L))

    # Output (same as original)
    new_carry = HierarchicalReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
    output = model.lm_head(z_H)[:, puzzle_emb_len:]
    q_logits = model.q_head(z_H[:, 0]).to(torch.float32)

    return new_carry, output, (q_logits[..., 0], q_logits[..., 1])
```

**Important**: The carry stores the projected z_H, so subsequent ACT steps start from the projected state. This means the projection compounds across ACT steps.

#### Step 3: Modified ACT evaluation loop

```python
def run_projected_act_loop(model, batch, U_basis, num_steps=None, project_positions="seq"):
    """Run ACT loop with z_H projection after each H_level call."""
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
            carry, logits, (q_halt, q_continue) = forward_with_projection(
                model, carry, batch, act_step=step,
                U_basis=U_basis, project_positions=project_positions,
            )
            preds = logits.argmax(dim=-1)
            results.append({
                "logits": logits,
                "predictions": preds,
                "q_halt_logits": q_halt,
                "q_continue_logits": q_continue,
            })

    return results
```

#### Step 4: Full evaluation with ACTLossHead

Also implement a full evaluation mode that wraps the modified forward into the ACTLossHead evaluation loop. This mirrors `hrm_inspect.py:evaluate()` but uses the projected forward. The cleanest approach:

```python
def evaluate_projected(model_full, data_path, U_basis, batch_size=2304,
                       project_positions="seq"):
    """Full evaluation with z_H projection, mirroring evaluate() from hrm_inspect.py.

    Monkey-patches model_full.model.inner.forward to use the projected version,
    then runs the standard evaluation loop.
    """
    inner = model_full.model.inner
    original_forward = inner.forward

    def patched_forward(carry, batch, act_step=0):
        return forward_with_projection(
            inner, carry, batch, act_step, U_basis, project_positions
        )

    inner.forward = patched_forward
    try:
        from hrm_inspect import evaluate
        return evaluate(model_full, data_path, batch_size=batch_size)
    finally:
        inner.forward = original_forward
```

#### Step 5: Comparison framework

The script runs three conditions and compares:

1. **Baseline** (no projection): Standard evaluation — establishes reference performance
2. **Project seq only**: Project z_H at token positions (puzzle_emb_len:) — tests if token position info is lm_head-degenerate
3. **Project all**: Project z_H at all positions including pos0 — tests if even the CLS position has collapsed

For each condition, report:
- Per-puzzle-set metrics: accuracy, exact_accuracy, steps
- Per-ACT-step: accuracy evolution (via hooks)
- Projection statistics (via `compute_unembed_projection` hook)

#### Step 6: CLI interface

```bash
python hrm_eval_projected.py \
    --checkpoint /path/to/step_166310 \
    --mode evaluate \
    --project-positions seq \
    --wandb-project hrm-inspect \
    --wandb-name projected-eval-step_166310
```

Arguments:
- `--checkpoint`: Path to checkpoint
- `--data-path`: Optional dataset path (read from config if not given)
- `--mode`: `inspect` (single batch, detailed hooks) or `evaluate` (full test set)
- `--project-positions`: `seq` (default), `all`, `seq_and_pos0`, `none` (baseline)
- `--batch-size`: Batch size for evaluation
- `--num-steps`: Number of ACT steps (inspect mode)
- `--wandb-project`, `--wandb-entity`, `--wandb-name`: Wandb logging options

### Expected Outcomes

| Scenario | Interpretation |
|:---|:---|
| Projected performance ≈ baseline | z_H is already effectively low-rank in lm_head subspace; the 501 orthogonal dimensions carry negligible information for output |
| Projected performance < baseline, but still reasonable | Some useful computation happens outside the lm_head subspace, but the model is partially robust to projection |
| Projected performance << baseline | The model critically depends on the full 512-dim space for intermediate H-level computation; degeneration happens only at the output step |

### Key Design Decisions

1. **Project after H_level only, not L_level**: z_L is never decoded through lm_head. Projecting z_L would artificially constrain it. The injection `z_L → H_level(z_H, z_L)` naturally allows z_L to contribute information from any direction.

2. **Carry stores projected z_H**: The next ACT step sees the projected z_H. This is by design — we want to test whether the model can function entirely within the lm_head subspace across all ACT steps.

3. **Preserve pos0 by default**: The `seq` mode only projects token positions, leaving pos0 (the CLS token for q_head) intact. This isolates the lm_head degeneration question from the q_head question. The `all` mode tests both.

## Key Takeaways

- Create a standalone `hrm_eval_projected.py` that reimplements the inner forward loop with z_H projection after each H_level call
- Reuse `_build_basis()` from `hrm_inspect.py` for SVD-based basis computation
- Support three projection modes: `seq` (token positions only), `all` (all positions), `none` (baseline)
- Compare projected vs. baseline performance to determine if degeneration is structural or convergent
- Use both `inspect` mode (single batch with detailed hooks) and `evaluate` mode (full test set metrics)
- Results disambiguate whether the model's H-level computation uses the full hidden space or has already collapsed

## Related Files

- `/home/developer/HRM_explore/hrm_inspect.py` — Base inspection harness with `_build_basis()`, `load_model()`, `load_full_model()`, `evaluate()`, `run_act_loop()`
- `/home/developer/HRM_explore/models/hrm/hrm_act_v2.py` — HRM Inner model forward (lines 243-287) that we reimplement with projection
- `/home/developer/HRM_explore/models/losses.py` — ACTLossHead wrapping model for full evaluation
- `/home/developer/HRM_explore/pretrain.py` — Reference training/eval loop (lines 266-330)
- `/home/developer/HRM_explore/.claude/findings/001_unembedding_subspace_degeneration_analysis.md` — Finding #1 showing z_H degeneration into lm_head subspace
