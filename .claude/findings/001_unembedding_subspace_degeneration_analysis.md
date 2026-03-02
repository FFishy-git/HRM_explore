---
id: 1
title: Unembedding Subspace Degeneration Analysis for HRM
filename: 001_unembedding_subspace_degeneration_analysis.md
created: 2026-03-02T05:50:39Z
depends_on: [0]
all_ancestors: [0]
---

# Unembedding Subspace Degeneration Analysis for HRM

## Summary

Analysis of what unembedding matrices (subspaces) exist in the HRM architecture and how to check whether the output residual streams z_H and z_L have "degenerated" — i.e., collapsed into the subspace spanned by the unembedding matrix rows. Includes a concrete hook implementation using the existing V2 hook infrastructure in `hrm_inspect.py`.

## Details

### Background: Sequence Layout

The residual streams z_H and z_L have shape `[batch, puzzle_emb_len + seq_len, hidden_size]`. This is because `_input_embeddings` (`hrm_act_v2.py:209-229`) **prepends** learned puzzle embeddings as virtual prefix positions before the actual token embeddings:

```python
embedding = torch.cat(
    (puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding),
    dim=-2
)
```

The resulting sequence dimension layout:

```
index:    0       1       ...  L-1       L       L+1     ...  L+seq_len-1
content:  puz_0   puz_1   ...  puz_{L-1} tok_0   tok_1   ...  tok_{seq_len-1}
          ├── puzzle prefix ──┤ ├──────── actual tokens ────────────────────┤
```

where `L = puzzle_emb_len = ceil(puzzle_emb_ndim / hidden_size)`.

The puzzle prefix positions are learned per-puzzle context vectors (via `CastedSparseEmbedding`), reshaped to fill `puzzle_emb_len` positions of width `hidden_size`. They are **not tokens to predict** — the **labels** tensor from the dataset has shape `[batch, seq_len]` covering only the token positions.

This layout means:
- **Positions `0..L-1`** (puzzle prefix): participate in bidirectional self-attention (causal=False) with all positions, serving as global context. Position 0 is used as a CLS-like summary token by `q_head`.
- **Positions `L..L+seq_len-1`** (token positions): these correspond 1:1 with the labels tensor and are where `lm_head` reads predictions from.

### Architecture: What Gets Decoded From Where

In the HRM Inner model forward pass (`hrm_act_v2.py:280-287`), after all H/L cycles complete:

```python
output = self.lm_head(z_H)[:, self.puzzle_emb_len:]    # label logits — [batch, seq_len, vocab_size]
q_logits = self.q_head(z_H[:, 0]).to(torch.float32)    # ACT halt/continue — [batch, 2]
```

The `[:, puzzle_emb_len:]` slice strips the puzzle prefix so the logits align with the `[batch, seq_len]` labels tensor. `q_head` reads position 0 (first puzzle prefix position) as a CLS token for the halt/continue Q-value.

**Only z_H is decoded.** z_L is never directly projected through an unembedding matrix. z_L influences z_H indirectly via the H_level reasoning module: `z_H = H_level(z_H, z_L)`, where z_L is the `input_injection` that gets added in `ReasoningModule.forward` (line 123-125):

```python
def forward(self, hidden_states, input_injection, **kwargs):
    hidden_states = hidden_states + input_injection  # z_L injected here
    for layer in self.layers:
        hidden_states = layer(hidden_states=hidden_states, **kwargs)
    return hidden_states
```

Similarly, z_H feeds into L_level as input_injection: `z_L = L_level(z_L, z_H + input_embeddings)`.

### The Two Unembedding Matrices

| Head | Module | Weight Shape | What it decodes | From which positions |
|------|--------|-------------|-----------------|---------------------|
| **lm_head** | `CastedLinear(hidden_size, vocab_size, bias=False)` | `[vocab_size, hidden_size]` | Label token logits | `z_H[:, puzzle_emb_len:]` (token positions) |
| **q_head** | `CastedLinear(hidden_size, 2, bias=True)` | `[2, hidden_size]` | Halt/Continue Q-values | `z_H[:, 0]` (first puzzle prefix position) |

Both are standard `CastedLinear` layers (`models/layers.py:43-59`) using `F.linear(input, weight, bias)`, so the weight matrix rows define the directions in hidden_size space that the model reads from.

### Sudoku Defaults

From `config/arch/hrm_v1.yaml` and `dataset/build_sudoku_dataset.py`:

| Parameter | Value | Source |
|-----------|-------|--------|
| **vocab_size** | **11** | `build_sudoku_dataset.py:133` — PAD(0) + digits "0".."9" (tokens 1-10) |
| **seq_len** | **81** | 9x9 grid flattened |
| **hidden_size** | **512** | `config/arch/hrm_v1.yaml:15` |
| **puzzle_emb_ndim** | **512** (= hidden_size) | `hrm_v1.yaml:19` → `puzzle_emb_len = ceil(512/512) = 1` |
| **H_cycles** | **2** | `hrm_v1.yaml:9` (overridden to other values in experiments) |
| **L_cycles** | **2** | `hrm_v1.yaml:10` (overridden to 8 in `sdk_full_lr3e4_Lc8.yaml`) |
| **halt_max_steps** | **16** | `hrm_v1.yaml:8` (overridden to 8 in some experiments) |

The unembedding subspace is **11-dimensional** (lm_head only) or **13-dimensional** (lm_head + q_head) in a **512-dimensional** hidden space — roughly 2% of the dimensions. The random baseline projection ratio would be `sqrt(11/512) ≈ 0.147`, giving a large gap between "random" (~0.15) and "degenerated" (~1.0). This makes the check very discriminating.

### Which Subspace to Consider

**For sequence positions (puzzle_emb_len onward):** The `lm_head.weight` row space (vocab_size dimensions) is the primary subspace. These positions produce label predictions.

**For position 0:** The `q_head.weight` row space (2 dimensions) is directly relevant since `q_head` reads from position 0. But position 0 of z_H also participates in attention with all other positions, so it may carry information for both heads.

**Combined subspace:** For a comprehensive check, concatenate both weight matrices:
```python
W = torch.cat([lm_head.weight, q_head.weight], dim=0)  # [vocab_size+2, hidden_size]
```
This gives the full subspace that the model's output heads can "see."

### How the Forward Loop Works (for hook context)

Per ACT step (`hrm_act_v2.py:254-278`):

1. **No-grad iterations** (warmup cycles):
   - For `_H_step` in `0..H_cycles-1`:
     - For `_L_step` in `0..L_cycles-1` (skip last-last):
       - `z_L = L_level(z_L, z_H + input_embeddings)` → fires L hook
     - If not last H_step:
       - `z_H = H_level(z_H, z_L)` → fires H hook

2. **1-step grad** (final cycle):
   - `z_L = L_level(z_L, z_H + input_embeddings)` → fires L hook with `is_grad_step=True`
   - `z_H = H_level(z_H, z_L)` → fires H hook with `is_grad_step=True`

Hooks receive a `HookContext` with `(act_step, h_cycle, l_cycle, is_grad_step, z_H, z_L)`.

### Degeneration Check: Method

The check projects a residual vector onto the row space of an unembedding matrix and measures what fraction of the norm is preserved:

```
ratio = ||P(z)|| / ||z||
```

where P is the orthogonal projector onto the row space of W.

**Steps:**
1. Get W (either `lm_head.weight` or `q_head.weight`)
2. Compute SVD of W^T: `U, S, V^T = SVD(W^T)` where U has shape `[hidden_size, rank]`
3. Keep columns of U corresponding to non-negligible singular values → `U_basis`
4. For each residual vector z: `z_proj = z @ U_basis @ U_basis^T`
5. Compute `ratio = ||z_proj|| / ||z||`

If ratio ≈ 1.0, the residual has degenerated into that subspace.

### What We Track

We track **two subspaces** applied to **two position groups**, for **both streams** (z_H and z_L):

**Subspaces:**
- **lm_head subspace**: row space of `lm_head.weight` — rank 11 for sudoku (vocab_size=11), random baseline ratio ≈ `sqrt(11/512) ≈ 0.147`
- **q_head subspace**: row space of `q_head.weight` — rank 2, random baseline ratio ≈ `sqrt(2/512) ≈ 0.063`

**Position groups:**
- **pos0**: position 0 only (the CLS-like puzzle prefix token read by q_head) — yields one vector per batch element → stats over batch
- **seq**: positions `puzzle_emb_len:` onward (token positions read by lm_head) — per-position norms are averaged over the sequence dimension first, yielding one value per batch element → stats over batch

**Metrics per combination** (mean/std/min/max over batch):
- `orig_norm`: `||z||` — original L2 norm
- `proj_norm`: `||P(z)||` — projected L2 norm
- `ratio`: `||P(z)|| / ||z||` — fraction of norm in the subspace

This gives a 2×2 grid of measurements per hook firing:

```
                    lm_head subspace (rank 11)     q_head subspace (rank 2)
                    ─────────────────────────      ─────────────────────────
  pos0 (z_H/z_L):  orig/proj/ratio stats          orig/proj/ratio stats
  seq  (z_H/z_L):  orig/proj/ratio stats          orig/proj/ratio stats
```

### Implementation

The hook `compute_unembed_projection` and helpers `_build_basis`/`_proj_stats` are implemented directly in `hrm_inspect.py` (not a separate file). The wandb logging function `log_inspect_to_wandb` is also in `hrm_inspect.py`.

Key functions added to `hrm_inspect.py`:
- `_build_basis(W)` — SVD to get orthonormal row-space basis
- `_proj_stats(z, U_basis)` — compute orig_norm/proj_norm/ratio with mean/std/min/max over batch
- `compute_unembed_projection(model)` — returns `(hook, results_list)`, register as both H and L hook
- `log_inspect_to_wandb(pred_results, norm_results, proj_results)` — logs all metrics to wandb keyed by ACT step

### Usage

**CLI with wandb logging:**

```bash
python hrm_inspect.py \
    --checkpoint /path/to/step_166310 \
    --mode inspect \
    --wandb-project hrm-inspect \
    --wandb-name unembed-projection-step_166310
```

If `--wandb-name` is omitted, it defaults to `unembed-projection-{checkpoint_basename}`.

**Programmatic:**

```python
from hrm_inspect import (
    load_model, load_sudoku_batch, run_act_loop,
    track_prediction_evolution, compute_residual_norms,
    compute_unembed_projection, log_inspect_to_wandb,
)

model = load_model("/path/to/checkpoint")
batches = load_sudoku_batch("/path/to/data")
batch = next(batches)
batch = {k: v.cuda() for k, v in batch.items()}

pred_hook, pred_results = track_prediction_evolution(model, batch["labels"])
norm_hook, norm_results = compute_residual_norms()
proj_hook, proj_results = compute_unembed_projection(model)

model.register_hook_H(pred_hook)
model.register_hook_H(norm_hook)
model.register_hook_L(norm_hook)
model.register_hook_H(proj_hook)
model.register_hook_L(proj_hook)

results = run_act_loop(model, batch)

# Optional: log to wandb
import wandb
wandb.init(project="hrm-inspect", name="unembed-projection-step_166310")
log_inspect_to_wandb(pred_results, norm_results, proj_results)
wandb.finish()
```

### Wandb Metric Namespace

All metrics are logged with ACT step as the x-axis (`wandb.log(..., step=act_step)`). Only grad-step results are logged (the final H/L state per ACT step).

```
pred/accuracy                                    # token-level accuracy
pred/exact_accuracy                              # puzzle-level exact match
pred/entropy                                     # output distribution entropy

norms/z_H_norm_{mean,std,min,max}                # z_H L2 norm stats
norms/z_L_norm_{mean,std,min,max}                # z_L L2 norm stats
norms/cosine_sim_{mean,std,min,max}              # z_H·z_L cosine similarity

proj/{stream}/{subspace}/{pos_group}/{metric}_{stat}
  e.g.:
  proj/z_H/lm_head/pos0/ratio_mean              # z_H pos0 onto lm_head subspace
  proj/z_H/lm_head/seq/ratio_mean               # z_H seq positions onto lm_head
  proj/z_H/q_head/pos0/orig_norm_mean            # z_H pos0 original norm
  proj/z_L/lm_head/seq/proj_norm_std             # z_L seq projected norm std
  ...
```

Wandb **run config** (logged once at init, not per-step) includes full model architecture plus subspace ranks:

```
checkpoint, data_path, batch_size, num_steps, mode,
hidden_size, vocab_size, seq_len, puzzle_emb_ndim, puzzle_emb_len,
H_cycles, L_cycles, H_layers, L_layers, halt_max_steps,
lm_head_rank, q_head_rank
```

Suggested wandb settings:
- **Project**: `hrm-inspect`
- **Run name**: `unembed-projection-{checkpoint_step}` (e.g., `unembed-projection-step_166310`, auto-derived if `--wandb-name` omitted)

### Interpreting Results

| `ratio_mean` | Interpretation |
|:---:|:---|
| ≈ 1.0 | **Degenerated** — residual lives almost entirely in that subspace. |
| ≈ `sqrt(rank/hidden_size)` | **Random baseline** — no special alignment. For sudoku: lm_head ≈ 0.147, q_head ≈ 0.063. |
| Between | **Partial alignment** — some dimensions used for output, others for internal computation. |

**Cross-referencing the grid:**
- **z_H + lm_head + seq**: The most direct check — this is exactly the path that produces label predictions. High ratio expected for a working model, but ≈1.0 means degeneration.
- **z_H + q_head + pos0**: The path that produces halt/continue. A 2-dim subspace in 512-dim space, so high ratio here would be very notable.
- **z_L + lm_head + seq**: If high, z_L carries the same kind of information as z_H (redundant streams).
- **z_L + q_head + pos0**: If high, z_L's position 0 has collapsed to the halt/continue directions.
- **lm_head on pos0 / q_head on seq**: Cross-checks — do positions carry information for heads that don't directly read them? Measures information leakage across position roles via attention.

### Notes on z_L

Since z_L is never directly decoded, "degeneration" for z_L has a different meaning. If z_L also collapses into the lm_head row space, it suggests z_L is carrying the same kind of information as z_H rather than complementary/auxiliary information. This could indicate:
- The two streams are redundant (not utilizing the full capacity)
- z_L has been "pulled" toward the output subspace by gradient flow through H_level

A healthy model might show z_H with moderate-to-high projection ratio (it needs to produce outputs) but z_L with a lower ratio (it should carry complementary information that z_H can't represent alone).

### Tracking Across ACT Steps

If the ratio starts low and rises toward 1.0 over ACT steps, it suggests the model progressively "focuses" its residual into the output subspace as it converges — which could be normal iterative refinement behavior. If the ratio is already ~1.0 from step 0, that's more concerning and suggests structural degeneration.

## Key Takeaways

- Only **z_H** is decoded through unembedding matrices; z_L influences z_H indirectly via input injection in H_level
- Two unembedding matrices exist: **lm_head.weight** `[vocab_size, hidden_size]` (rank 11 for sudoku) and **q_head.weight** `[2, hidden_size]` (rank 2)
- We track both subspaces independently across both position groups (pos0 and seq), for both streams (z_H and z_L) — a 2×2×2 grid
- For each cell: original norm, projected norm, and ratio, with mean/std/min/max over batch
- For sequence positions, per-position norms are averaged over the sequence dimension before computing batch statistics
- Random baselines for sudoku (hidden_size=512): lm_head ratio ≈ 0.147, q_head ratio ≈ 0.063
- The existing V2 hook system (`register_hook_H`, `register_hook_L`) provides exactly the access needed
- Wandb logging is integrated into `hrm_inspect.py` via `--wandb-project hrm-inspect` CLI flag, or programmatically via `log_inspect_to_wandb()`
- Metrics are namespaced as `proj/{stream}/{subspace}/{pos_group}/{metric}_{stat}` with ACT step as x-axis

## Related Files

- `/home/developer/HRM_explore/models/hrm/hrm_act_v2.py` — HRM model with hooks (Inner model forward: lines 243-287, hook firing: lines 262-278)
- `/home/developer/HRM_explore/hrm_inspect.py` — Inspection harness with pre-built hooks (collect_residual_streams, track_prediction_evolution, compute_residual_norms, compute_unembed_projection) and wandb logging (log_inspect_to_wandb)
- `/home/developer/HRM_explore/models/layers.py` — CastedLinear definition (lm_head and q_head are instances, lines 43-59)
- `/home/developer/HRM_explore/models/losses.py` — ACTLossHead wrapper (lines 40-101)
- `/home/developer/HRM_explore/config/arch/hrm_v1.yaml` — Default architecture config (hidden_size=512, H/L cycles, halt_max_steps)
- `/home/developer/HRM_explore/dataset/build_sudoku_dataset.py` — Sudoku dataset builder (vocab_size=11, seq_len=81)
