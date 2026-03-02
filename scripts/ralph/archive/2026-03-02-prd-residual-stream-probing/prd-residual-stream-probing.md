# PRD: Residual Stream Probing Layer for H-Level Information Gain

## Introduction

Implement a probing layer to measure how much information the H-level model adds during each higher-loop iteration. Given:

- **z_H**: H model residual stream output from the *previous* higher loop
- **z_H\***: H model residual stream output from the *current* higher loop (i.e., z_H\* = f_H(z_L, z_H))
- **z_L**: L model residual stream output when the last lower loop finishes within each higher loop
- **input_embedding**: The original input embedding

We train a single probing MLP that takes `concat(z_H_or_zHstar, z_L, input_embedding)` and outputs logits over digits {1..9}. The same probe is used for both z_H and z_H\* inputs, so any difference in probing accuracy is attributable to the information gained by the H-level computation, not to probe bias.

This builds on findings:
- `000_hrm_architecture_and_residual_stream_layout.md`: z_H\* = f_H(z_L, z_H)
- `002_eval_zh_projection_to_lm_head_subspace.md`: z_H and z_H\* lie in the same vector space

## Goals

- Train a shared probing MLP on `concat(z_H, z_L, input_emb)` and `concat(z_H*, z_L, input_emb)` inputs with cross-entropy loss on blank-cell digit prediction
- Extract z_H (before) and z_H\* (after) at ACT steps ≥ 2 (skipping the first two steps where z_H and z_L may not be stable) to measure per-step information gain
- Produce accuracy metrics that quantify how much the H-level improves cell-level digit prediction at each ACT step
- Provide a reusable, well-structured training script that follows existing codebase patterns

## User Stories

### US-001: Implement the probing MLP module
**Description:** As a researcher, I want a probing MLP class so that I can decode digit predictions from concatenated residual streams.

**Acceptance Criteria:**
- [ ] New file `models/probing.py` with a `ProbingMLP` class
- [ ] Architecture: `Linear(d_in, 4*d_in)` → `ReLU` → `Linear(4*d_in, 9)` where `d_in = 3 * hidden_size` (default 1536)
- [ ] Output is logits over 9 classes (digits 1-9)
- [ ] Hidden dim multiplier is configurable (default 4)
- [ ] Class follows existing codebase conventions (e.g., `trunc_normal_init_` for weight init)
- [ ] Typecheck passes

### US-002: Extend HookContext with input_embeddings
**Description:** As a researcher, I need access to `input_embeddings` inside hook callbacks so that probing (and future hooks) can use them without re-computing.

**Acceptance Criteria:**
- [ ] Add `input_embeddings: torch.Tensor` field to the `HookContext` dataclass in `hrm_act_v2.py`
- [ ] Pass `input_embeddings` (computed at the start of `forward()`) to every `HookContext` construction site (4 sites: 2 no-grad L/H, 2 grad-step L/H)
- [ ] Existing hooks in `hrm_inspect.py` continue to work (they simply ignore the new field)
- [ ] Typecheck passes

### US-003: Implement residual stream extraction hook callbacks
**Description:** As a researcher, I need hook callbacks that capture z_H (before H-level) and z_H\* (after H-level) plus z_L and input_embedding at each ACT step, so that I can feed them to the probe.

**Acceptance Criteria:**
- [ ] Hook callback functions in `probe_train.py` that collect, at each ACT step:
  - `z_H_before`: captured from the L-hook at the last l_cycle (where `ctx.z_H` is the carry-in z_H before the upcoming H-level call)
  - `z_H_after`: captured from the H-hook at the last h_cycle (where `ctx.z_H` is z_H\* after the H-level call)
  - `z_L`: from the L-hook context at the last l_cycle
  - `input_embeddings`: from `ctx.input_embeddings` (added in US-002)
- [ ] Register via the existing `register_hook_H` / `register_hook_L` mechanism
- [ ] Data is collected for ACT steps ≥ 2 (skipping steps 0 and 1 where z_H/z_L are not yet stable)
- [ ] Extracted tensors are detached and optionally moved to CPU to avoid OOM
- [ ] Typecheck passes

### US-004: Implement the probe training loop
**Description:** As a researcher, I want a training script that loads a frozen HRM checkpoint, extracts residual streams, and trains the probing MLP so that I can measure H-level information gain.

**Acceptance Criteria:**
- [ ] New file `probe_train.py` with CLI interface (argparse or Hydra)
- [ ] Loads a frozen HRM checkpoint using the existing `load_model()` from `hrm_inspect.py`
- [ ] HRM model is set to `eval()` mode with all parameters frozen (`requires_grad_(False)`)
- [ ] For each batch:
  1. Runs the HRM forward with hooks to extract z_H, z_H\*, z_L, input_embedding at each ACT step
  2. Forms probe inputs: `concat(z_H, z_L, input_emb)` and `concat(z_H*, z_L, input_emb)` along hidden dim → shape `[batch, 81, 1536]`
  3. Passes both through the *same* probe → logits `[batch, 81, 9]`
  4. Computes cross-entropy loss on blank cells only (where input != given digit)
  5. Combined loss = loss(z_H input) + loss(z_H\* input) (both contribute gradients to the same probe)
  6. Backprop and optimizer step on probe parameters only
- [ ] Target labels: remapped from token encoding (2-10) to class indices (0-8) for `nn.CrossEntropyLoss`
- [ ] Blank cell mask: cells where `inputs == 1` (token for digit "0" / blank)
- [ ] Uses AdamW optimizer with configurable learning rate (default 1e-3) and weight decay (default 1e-2)
- [ ] Supports configurable number of ACT steps to run the HRM for
- [ ] Skips the first N ACT steps (default N=2) — only uses data from step N onward for probe training, as early steps have unstable z_H/z_L
- [ ] Logs training loss and accuracy per epoch to stdout and optionally to Wandb
- [ ] Typecheck passes

### US-005: Implement probe evaluation and metrics reporting
**Description:** As a researcher, I want to evaluate the trained probe and see per-ACT-step accuracy for z_H vs z_H\*, so I can quantify the H-level information gain.

**Acceptance Criteria:**
- [ ] Evaluation mode in `probe_train.py` (e.g., `--mode eval`)
- [ ] Reports per-ACT-step metrics separately for z_H and z_H\* (for steps ≥ 2):
  - Per-cell accuracy (on blank cells only)
  - Cross-entropy loss
  - Accuracy delta: `acc(z_H*) - acc(z_H)` at each ACT step
- [ ] Prints a summary table (ACT step × {z_H acc, z_H\* acc, delta}) starting from step 2
- [ ] Optionally logs metrics to Wandb with step-indexed keys
- [ ] Supports loading a saved probe checkpoint
- [ ] Typecheck passes

### US-006: Add probe checkpoint saving/loading
**Description:** As a researcher, I want to save and load trained probe weights so I can reuse them without retraining.

**Acceptance Criteria:**
- [ ] Saves probe state dict + config (hidden_size, hidden_mult, etc.) to a checkpoint file
- [ ] Saves at configurable interval and at end of training
- [ ] Loading reconstructs the probe architecture from saved config
- [ ] CLI flag `--probe-checkpoint` to resume training or run eval from saved probe
- [ ] Typecheck passes

### US-007: Add unit tests for probe module and data pipeline
**Description:** As a developer, I want tests for the probing MLP and the data remapping logic so that correctness is verified.

**Acceptance Criteria:**
- [ ] Test file at `tests/test_probing.py`
- [ ] Tests that `ProbingMLP` output shape is `[batch, seq, 9]` for input shape `[batch, seq, 1536]`
- [ ] Tests that target label remapping converts tokens 2-10 to classes 0-8 correctly
- [ ] Tests that blank cell mask correctly identifies cells where `inputs == 1`
- [ ] Tests that loss is computed only on masked (blank) cells — given cells do not contribute to loss
- [ ] Tests that probe gradients flow only to probe parameters, not HRM parameters
- [ ] `pytest tests/test_probing.py` passes
- [ ] Typecheck passes

## Functional Requirements

- **FR-1:** The probing MLP must accept input of shape `[batch, seq_len, 3 * hidden_size]` and output `[batch, seq_len, 9]`
- **FR-2:** The same probe instance must be used for both z_H and z_H\* inputs within each training step — no separate probes
- **FR-3:** Cross-entropy loss must be computed only on blank cells (where `inputs == 1`, i.e., the cell was not a given digit)
- **FR-4:** The HRM model must be completely frozen during probe training — no gradient flow to HRM parameters
- **FR-5:** Residual stream extraction must work at ACT steps ≥ 2 (skip the first two steps where z_H/z_L are not yet stable), configurable via `--skip-act-steps` (default 2)
- **FR-6:** Target labels must be remapped from token encoding (2-10) to zero-indexed class indices (0-8) for cross-entropy
- **FR-7:** The probe training must support both single-GPU execution and batch size configuration
- **FR-8:** Input embeddings must be extracted once per batch (they don't change across ACT steps) and reused
- **FR-9:** The script must be runnable as `python probe_train.py --checkpoint <path> --data <path> [options]`

## Non-Goals

- No distributed/multi-GPU training for the probe (single GPU is sufficient)
- No modification to the HRM model architecture or training code
- No hyperparameter search or AutoML for probe architecture
- No probing of individual H-level or L-level layers (only final outputs)
- No visualization or plotting code (downstream analysis notebooks are separate)
- No integration with the existing `pretrain.py` training loop

## Technical Considerations

- **Memory management:** Running the full HRM forward with hooks at every ACT step produces many intermediate tensors. Detach and optionally move to CPU between extraction and probe training to avoid OOM. Consider processing one ACT step at a time if memory is tight.
- **Hook mechanism:** Use the V2 hook API (`register_hook_H`, `register_hook_L`) from `hrm_act_v2.py`. The `HookContext` is extended with `input_embeddings` (US-002). The H-hook fires *after* each H-level call, so `ctx.z_H` in the H-hook is z_H\*. The L-hook at the last l_cycle provides `ctx.z_H` which is the z_H *before* the upcoming H-level call. This gives us both z_H and z_H\* without any model surgery.
- **Existing utilities:** Reuse `load_model()`, `load_sudoku_batch()`, `run_act_loop()` from `hrm_inspect.py` where possible.
- **Tensor shapes:** All of z_H, z_L, input_embedding are `[batch, puzzle_emb_len + seq_len, hidden_size]` = `[batch, 82, 512]`. For probing, strip the puzzle prefix: use positions `[:, 1:, :]` → `[batch, 81, 512]`. Concatenation yields `[batch, 81, 1536]`.
- **Number of training examples:** The probe should converge quickly (it's a simple MLP). A few epochs over the training set should suffice. Include early stopping or fixed epoch count.

## Success Metrics

- Probe achieves non-trivial accuracy (above random 11.1% baseline) on blank cells, confirming the residual streams contain digit information
- Accuracy on z_H\* inputs is measurably higher than on z_H inputs, quantifying H-level information gain
- Per-ACT-step accuracy curves show monotonic (or near-monotonic) improvement, consistent with iterative refinement
- Training completes in under 30 minutes on a single GPU for the full training set

## Open Questions

- Should we also probe z_L alone (without z_H) as an additional baseline to measure L-level contribution independently?
- Should the probe be trained on data from a single ACT step or across all ACT steps jointly? (Current design: jointly, with data from all steps contributing to loss)
- Is there value in probing at intermediate h_cycles within each ACT step, or is the final h_cycle sufficient?
