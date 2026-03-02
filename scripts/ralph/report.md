# Residual Stream Probing — Implementation Report

**Branch:** `ralph/residual-stream-probing`
**Date:** 2026-03-02
**User Stories:** US-001 through US-007 — all passing

---

## 1. Files Created / Modified

| File | Status | Description |
|------|--------|-------------|
| `models/probing.py` | New | `ProbingMLP` module — two-layer MLP (Linear→ReLU→Linear) decoding 9-class digit predictions from concatenated residual streams |
| `models/hrm/hrm_act_v2.py` | Modified | Added `input_embeddings` field to `HookContext` dataclass; wired to all 4 hook construction sites |
| `probe_train.py` | New | Training + evaluation script — hook-based residual stream extraction, ACT-loop runner, training loop, per-step eval, checkpoint save/load, CLI |
| `tests/conftest.py` | New | Mocks GPU-only `flash_attn` modules so tests run on CPU |
| `tests/test_probing.py` | New | 14 unit tests covering probe shape, label remapping, blank-cell masking, loss correctness, gradient isolation |

---

## 2. Key Architectural Decisions

- **Single shared probe for before/after.** One `ProbingMLP` processes both `concat(z_H, z_L, input_emb)` and `concat(z_H*, z_L, input_emb)`. Combined loss trains a single probe to decode from both representations, enabling direct comparison.
- **Hook-based extraction.** Reuses existing `register_hook_L`/`register_hook_H` mechanism. L-hook fires before H-level (captures `z_H_before`); H-hook fires after (captures `z_H*`). A `_pending` buffer synchronizes the two per ACT step.
- **Frozen HRM.** Checkpoint loaded via `load_model()`, set to `eval()` + `requires_grad_(False)`. Only probe parameters receive gradients.
- **Skip early ACT steps.** Steps < `skip_steps` (default 2) are discarded since `z_H`/`z_L` are not yet stable.
- **Blank-cell-only loss.** Cross-entropy masked to blank cells (`inputs == 1`). Labels remapped from tokens 2–10 to classes 0–8.
- **Codebase-native patterns.** Uses `F.linear` + `nn.Parameter` (not `nn.Linear`), `trunc_normal_init_` for weights, zero-init biases — matching existing conventions.

---

## 3. Codebase Patterns Discovered

- Weight init: `trunc_normal_init_(tensor, std=1.0 / (in_features ** 0.5))` from `models/common.py`
- Forward passes use `F.linear(input, weight, bias)` with `nn.Parameter` weights, not `nn.Linear`
- `HookContext` fields must default to `None` for backward compatibility (`hrm_inspect.py` uses keyword args)
- `input_embeddings` is computed once at the top of `_Inner.forward()` and available at all hook sites
- Manual ACT loop pattern: `empty_carry()` → `reset_carry()` → loop `model(carry, batch, act_step=step)`
- Residual streams include a puzzle-embedding prefix that must be stripped to align with label shape
- `flash_attn` is GPU-only; mock via `sys.modules` in `conftest.py` for CPU-based pytest
- No formal typecheck config exists; `python -m py_compile` is the local quality gate

---

## 4. Test Results

**14 tests — all passing** (`python -m pytest tests/ -v`)

| Test Class | Tests | Coverage |
|------------|------:|----------|
| `TestProbingMLPShape` | 4 | Output shape, custom hidden_size, single position, unbounded logits |
| `TestLabelRemapping` | 1 | Tokens 2–10 → classes 0–8 |
| `TestBlankCellMask` | 3 | Blank detection, no blanks, all blanks |
| `TestLossOnBlankCells` | 3 | Loss restricted to blanks, given cells excluded, empty mask → zero loss |
| `TestGradientIsolation` | 2 | Gradients flow to probe only; detached inputs block upstream grads |

All files pass `python -m py_compile` checks.

---

## 5. Open Items & Recommendations

- **GPU integration test.** All tests run on CPU with mocked flash_attn. An end-to-end run on GPU with a real HRM checkpoint is needed to validate the full pipeline.
- **First training run.** No training run has been executed yet. Suggested command: `python probe_train.py --checkpoint <path> --data <path> --mode train --epochs 10`
- **Wandb logging.** Integration is wired up but optional — pass `--wandb` flags to enable experiment tracking.
- **Hyperparameter tuning.** Defaults (lr=1e-3, weight_decay=1e-2, hidden_mult=4, skip_steps=2) are untested starting points.
- **Multi-GPU.** Script assumes single-GPU execution; no distributed training support.
