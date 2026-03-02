---
id: 3
title: Finetuning with Fixed Embedding and z_H Projection
filename: 003_finetune_with_fixed_embedding_and_zh_projection.md
created: 2026-03-02T09:39:36Z
depends_on: [1, 2]
all_ancestors: [1, 2]
---

# Finetuning with Fixed Embedding and z_H Projection

## Summary

Plan for finetuning the HRM model with two key modifications: (1) freeze lm_head weights to fix the unembedding subspace, and (2) project z_H onto this fixed subspace after every H_level call. This forces the model's H-level reasoning to operate entirely within the 11-dimensional lm_head row space, testing whether the model can learn effective Sudoku-solving strategies within this constraint. This builds on Finding #1 (degeneration observed) and Finding #2 (evaluation-only projection).

## Details

### Motivation

Finding #1 showed z_H degenerates into the lm_head subspace. Finding #2 measures the effect of projection at evaluation time (no training). This finding goes further: **can the model learn to reason effectively when structurally constrained to the lm_head subspace?**

The hypothesis is:
- If the model already degenerates into this subspace, explicitly constraining it shouldn't hurt — and might even help by removing noise from orthogonal directions.
- The fixed embedding ensures the target subspace doesn't shift during training.
- If the model can learn effectively under this constraint, it confirms the lm_head subspace is sufficient for the task.
- If it cannot, it suggests the model needs the full hidden space for intermediate computation during H-level reasoning, and degeneration is a natural output-stage phenomenon.

### Approach: Modified Training Script `finetune_projected.py`

Create a new training script that extends `pretrain.py` with:
1. A modified model that projects z_H after each H_level call (differentiable)
2. Frozen lm_head weights
3. Optional: frozen embed_tokens (since lm_head and embed_tokens may share structure)
4. Loading from a pretrained checkpoint

### Implementation Plan

#### Step 1: Modified Inner Model with Projection

Create a subclass of `HierarchicalReasoningModel_ACTV1_Inner` that overrides `forward()` to add the projection. The projection is differentiable — gradients flow through `z_H @ U_basis @ U_basis.T` back to the parameters that produced z_H (H_level weights).

```python
class ProjectedInner(HierarchicalReasoningModel_ACTV1_Inner):
    """Inner model that projects z_H onto lm_head subspace after each H_level call.

    The projection is P(z) = z @ U @ U^T where U is the orthonormal basis of
    lm_head.weight row space. Since lm_head is frozen, U is constant and the
    projection is a fixed linear operation — differentiable w.r.t. z but with
    no parameters of its own.
    """

    def __init__(self, config, project_positions="seq"):
        super().__init__(config)
        self.project_positions = project_positions
        # U_basis will be set after loading weights (need lm_head to be loaded first)
        self._projection_basis = None

    def setup_projection(self):
        """Compute and cache the projection basis from current lm_head weights.

        Call this AFTER loading the checkpoint and BEFORE training.
        The basis is registered as a buffer (non-parameter, moves with model).
        """
        with torch.no_grad():
            W = self.lm_head.weight.float()  # [vocab_size, hidden_size]
            U, S, _ = torch.linalg.svd(W.T, full_matrices=False)
            rank = (S > 1e-6).sum().item()
            U_basis = U[:, :rank]  # [hidden_size, rank]
        # Register as buffer so it moves with .cuda() and .to()
        self.register_buffer('_projection_basis', U_basis)

    def _project_z_H(self, z_H):
        """Project z_H onto lm_head subspace (differentiable w.r.t. z_H).

        z_H: [batch, seq_total, hidden_size]
        Returns: z_H with specified positions projected
        """
        U = self._projection_basis  # [hidden_size, rank]
        if self.project_positions == "all":
            return (z_H @ U) @ U.T
        elif self.project_positions == "seq":
            z_out = z_H.clone() if not z_H.requires_grad else torch.cat([
                z_H[:, :self.puzzle_emb_len],
                (z_H[:, self.puzzle_emb_len:] @ U) @ U.T
            ], dim=1)
            return z_out
        else:
            return z_H

    def forward(self, carry, batch, act_step=0):
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # No-grad iterations (identical to base except projection added)
        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L

            for _H_step in range(self.config.H_cycles):
                for _L_step in range(self.config.L_cycles):
                    if not ((_H_step == self.config.H_cycles - 1) and
                            (_L_step == self.config.L_cycles - 1)):
                        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                        self._fire_L(HookContext(act_step, _H_step, _L_step, False, z_H, z_L))

                if not (_H_step == self.config.H_cycles - 1):
                    z_H = self.H_level(z_H, z_L, **seq_info)
                    z_H = self._project_z_H(z_H)  # <<< PROJECTION
                    self._fire_H(HookContext(act_step, _H_step,
                                             self.config.L_cycles - 1, False, z_H, z_L))

        assert not z_H.requires_grad and not z_L.requires_grad

        # 1-step grad (gradients flow through projection)
        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        self._fire_L(HookContext(act_step, self.config.H_cycles - 1,
                                 self.config.L_cycles - 1, True, z_H, z_L))

        z_H = self.H_level(z_H, z_L, **seq_info)
        z_H = self._project_z_H(z_H)  # <<< PROJECTION (differentiable)
        self._fire_H(HookContext(act_step, self.config.H_cycles - 1,
                                 self.config.L_cycles - 1, True, z_H, z_L))

        # Output
        new_carry = HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=z_H.detach(), z_L=z_L.detach()
        )
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])
```

**Key detail on gradient flow**: In the 1-step grad section, `z_H = self._project_z_H(z_H)` is differentiable. The Jacobian of the projection `P(z) = z @ U @ U^T` with respect to z is simply `U @ U^T` — a constant matrix (since lm_head is frozen). Gradients from `lm_head(z_H)` flow back through the projection to `H_level` parameters. Since `lm_head(P(z)) = (z @ U @ U^T) @ W^T`, and W's rows span the same space as U's columns, lm_head perfectly reads the projected z_H. The gradient signal is "compressed" but not lost — it just can't push z_H outside the subspace.

#### Step 2: Modified ACT Wrapper

The ACT wrapper (`HierarchicalReasoningModel_ACTV1`) needs to use `ProjectedInner` instead of the base inner model. The cleanest approach is to swap the inner model after construction:

```python
def create_projected_model(checkpoint_path, data_path, project_positions="seq"):
    """Load a pretrained model and convert its inner model to ProjectedInner."""
    from hrm_inspect import _load_config_and_state, _resolve_data_path

    data_path = _resolve_data_path(checkpoint_path, data_path)
    arch_config, model_cfg, state_dict = _load_config_and_state(
        checkpoint_path, data_path, "cuda"
    )

    # Create the projected inner model
    config = HierarchicalReasoningModel_ACTV1Config(**model_cfg)
    with torch.device("cuda"):
        projected_inner = ProjectedInner(config, project_positions=project_positions)

    # Load the pretrained weights into projected_inner
    inner_state = {}
    prefix = "model.inner."
    for k, v in state_dict.items():
        if k.startswith(prefix):
            inner_state[k[len(prefix):]] = v
    projected_inner.load_state_dict(inner_state, strict=False)

    # Setup projection basis from loaded lm_head weights
    projected_inner.setup_projection()

    # Freeze lm_head
    for param in projected_inner.lm_head.parameters():
        param.requires_grad = False

    # Create full model with the projected inner
    act_model = HierarchicalReasoningModel_ACTV1(model_cfg)
    act_model.inner = projected_inner

    loss_cfg = arch_config.get("loss", {})
    loss_type = loss_cfg.get("loss_type", "softmax_cross_entropy")
    model = ACTLossHead(act_model, loss_type=loss_type)

    return model, data_path
```

#### Step 3: Freezing Strategy

Which parameters to freeze:

| Parameter | Freeze? | Rationale |
|:---|:---:|:---|
| `lm_head.weight` | **Yes** | Defines the projection subspace — must stay fixed |
| `q_head.weight`, `q_head.bias` | No | Q-head reads from pos0, which may not be projected. Even if projected, q_head should adapt |
| `embed_tokens.weight` | **Optional** | Input embeddings feed into L_level via `z_H + input_embeddings`. If frozen, the "input signal" is fixed. If unfrozen, the model can adapt how it injects information |
| `puzzle_emb` | No | Puzzle embeddings should adapt to the new constraint |
| `H_level.*` | No | Must learn to reason within the projected subspace |
| `L_level.*` | No | Must adapt to receiving projected z_H as injection |
| `H_init`, `L_init` | **Optional** | Initial states — freezing prevents the model from learning a different starting point |

**Recommended default**: Freeze only `lm_head.weight`. Let everything else adapt.

**Variant**: Also freeze `embed_tokens.weight` to test pure architectural constraint (fixed I/O subspace).

#### Step 4: Training Script Structure

```python
# finetune_projected.py

@hydra.main(config_path="config", config_name="cfg_finetune_projected", version_base=None)
def launch(hydra_config):
    config = FinetuneProjectedConfig(**hydra_config)

    # Load pretrained model and convert to projected version
    model, data_path = create_projected_model(
        config.checkpoint_path,
        config.data_path,
        project_positions=config.project_positions,
    )

    # Freeze specified parameters
    freeze_params(model, config.freeze_list)

    # Create optimizers (skip frozen params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizers = create_optimizers(model, config, trainable_params)

    # Standard training loop (reuse from pretrain.py)
    # ...same train_batch / evaluate / checkpoint pattern...
```

#### Step 5: Config

Create `config/cfg_finetune_projected.yaml` and `config/experiment/sdk_finetune_projected.yaml`:

```yaml
# config/cfg_finetune_projected.yaml
defaults:
  - cfg_pretrain  # Inherit base pretrain config

# Finetuning overrides
checkpoint_path_pretrained: null  # REQUIRED: path to pretrained checkpoint
project_positions: "seq"          # "seq", "all", or "none"
freeze_lm_head: true
freeze_embed_tokens: false

# Lower LR for finetuning
lr: 1e-5
lr_warmup_steps: 500
```

```yaml
# config/experiment/sdk_finetune_projected.yaml
checkpoint_path_pretrained: "checkpoints/.../step_166310"
data_path: data/sudoku-hard-full
global_batch_size: 2304
epochs: 20
eval_interval: 5
lr: 1e-5
lr_min_ratio: 0.1
lr_warmup_steps: 200
project_positions: "seq"
freeze_lm_head: true
freeze_embed_tokens: false
```

#### Step 6: Gradient Flow Analysis

The gradient path under projection:

```
Loss = CE(lm_head(P(H_level(z_H_prev, z_L))), labels)

∂Loss/∂θ_H = ∂Loss/∂z_H_proj × ∂z_H_proj/∂z_H × ∂z_H/∂θ_H

where:
  z_H_proj = P(z_H) = z_H @ U @ U^T
  ∂z_H_proj/∂z_H = U @ U^T   (constant projection matrix)

So: ∂Loss/∂θ_H = (∂Loss/∂z_H_proj @ U @ U^T) × ∂z_H/∂θ_H
```

This means:
- Gradients for H_level parameters are "filtered" through the projection — only the component of the gradient that lies in the lm_head subspace propagates.
- H_level cannot learn to produce z_H components outside the subspace (they get projected away in forward AND the gradient for those components is zero).
- L_level receives projected z_H as input injection (via `H_level(z_H, z_L)` in the next cycle), so it sees only the projected version.

**No gradient through the projection basis**: Since `U_basis` comes from the frozen `lm_head.weight` and is registered as a buffer (not a parameter), it has no gradient. The projection is a fixed linear transform.

#### Step 7: Evaluation and Metrics

During finetuning, track:

1. **Standard training metrics**: lm_loss, accuracy, exact_accuracy, q_halt_accuracy, steps
2. **Projection metrics** (via hooks from `hrm_inspect.py`):
   - `proj/z_H/lm_head/seq/ratio_mean` — should be exactly 1.0 after projection
   - `proj/z_L/lm_head/seq/ratio_mean` — interesting: does z_L also collapse?
3. **Comparison checkpoints**: Save checkpoints and evaluate with the standard (non-projected) model to see if the learned weights generalize

#### Step 8: Experimental Conditions

Run a grid of experiments:

| Condition | Project | Freeze lm_head | Freeze embed_tokens | Purpose |
|:---|:---:|:---:|:---:|:---|
| A: Baseline | none | No | No | Standard finetuning from checkpoint (control) |
| B: Project-only | seq | Yes | No | Core experiment: projected finetuning |
| C: Project+freeze-embed | seq | Yes | Yes | Fixed I/O subspace |
| D: Project-all | all | Yes | No | Project all positions including pos0 |

### Architectural Considerations

#### Why this might work
- z_H is already close to the lm_head subspace (Finding #1)
- The model might be "wasting" capacity on orthogonal directions
- Constraining to the subspace could act as regularization
- For Sudoku (vocab_size=11), the task is relatively low-dimensional — 11 directions might suffice for representing digit distributions

#### Why this might not work
- Attention mechanisms in H_level and L_level operate in the full 512-dim space. Key/query/value projections might need the orthogonal dimensions for meaningful attention patterns
- RMS normalization operates on the full vector — projection changes the effective norm and can disrupt the learned scale
- The H_init buffer is 512-dimensional — projecting it to 11 dimensions loses most of the initial state

#### Mitigation: Norm preservation
After projection, the norm of z_H drops (since we remove components). Consider:
```python
def _project_z_H_norm_preserve(self, z_H):
    """Project and rescale to preserve original norm."""
    orig_norm = z_H.norm(dim=-1, keepdim=True)
    z_proj = (z_H @ self._projection_basis) @ self._projection_basis.T
    proj_norm = z_proj.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return z_proj * (orig_norm / proj_norm)
```
This is an optional variant to test. The norm-preserving version might work better with RMS norm layers.

### File Structure

```
HRM_explore/
├── finetune_projected.py          # New training script
├── hrm_eval_projected.py          # New evaluation script (Finding #2)
├── models/hrm/
│   ├── hrm_act_v2.py             # Unchanged
│   └── hrm_act_v2_projected.py   # ProjectedInner subclass
├── config/
│   ├── cfg_finetune_projected.yaml
│   └── experiment/
│       └── sdk_finetune_projected.yaml
```

## Key Takeaways

- Freeze lm_head to fix the projection subspace, then finetune all other parameters with z_H projected after each H_level call
- The projection is differentiable — gradients flow through `U @ U^T` (constant matrix) back to H_level parameters
- Gradient signal is "filtered" to only the lm_head-subspace component, preventing H_level from learning to use orthogonal directions
- Create `ProjectedInner` as a subclass of `HierarchicalReasoningModel_ACTV1_Inner` with overridden `forward()`
- Consider norm-preserving projection as a variant (rescale projected z_H to match original norm)
- Experimental grid: project-none (baseline), project-seq, project-all, with/without frozen embed_tokens
- If the model can learn effectively under this constraint, it confirms the lm_head subspace is sufficient; if not, the full hidden space is needed for intermediate computation

## Related Files

- `/home/developer/HRM_explore/pretrain.py` — Base training script (lines 108-159: model creation, 209-263: train_batch, 266-330: evaluate)
- `/home/developer/HRM_explore/models/hrm/hrm_act_v2.py` — HRM Inner model forward (lines 243-287) that ProjectedInner overrides
- `/home/developer/HRM_explore/models/losses.py` — ACTLossHead (lines 40-101) wrapping model for loss computation
- `/home/developer/HRM_explore/hrm_inspect.py` — `_build_basis()` for SVD-based basis computation, evaluation utilities
- `/home/developer/HRM_explore/config/arch/hrm_v1.yaml` — Architecture config (hidden_size=512, vocab_size=11)
- `/home/developer/HRM_explore/config/experiment/sdk_full_lr3e4_Lc8.yaml` — Reference experiment config for Sudoku full
- `/home/developer/HRM_explore/.claude/findings/001_unembedding_subspace_degeneration_analysis.md` — Finding #1: degeneration observation
- `/home/developer/HRM_explore/.claude/findings/002_eval_zh_projection_to_lm_head_subspace.md` — Finding #2: evaluation-only projection
