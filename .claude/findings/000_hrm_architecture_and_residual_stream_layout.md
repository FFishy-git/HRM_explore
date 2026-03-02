---
id: 0
title: HRM Architecture and Residual Stream Layout
filename: 000_hrm_architecture_and_residual_stream_layout.md
created: 2026-03-02T08:00:00Z
depends_on: []
all_ancestors: []
---

# HRM Architecture and Residual Stream Layout

## Summary

Documents how the HRM (Hierarchical Reasoning Model) residual streams z_H and z_L are initialized, laid out in memory, and decoded. This serves as the foundational reference for all downstream analysis findings.

## Details

### z_H and z_L Initialization

Both z_H and z_L are initialized from **learned buffers** defined in `hrm_act_v2.py:170-171`:

```python
self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
```

These are 1-D vectors of shape `[hidden_size]` that get broadcast across all sequence positions.

**Carry lifecycle:**

1. **`empty_carry`** (`hrm_act_v2.py:231-235`) — allocates uninitialized tensors of shape `[batch, seq+puzzle_emb_len, hidden_size]` via `torch.empty`.
2. **`reset_carry`** (`hrm_act_v2.py:237-241`) — conditionally initializes: where `reset_flag` is true, sets `z_H = H_init` and `z_L = L_init` (broadcast across seq positions). Where false, keeps the previous carry.
3. **At forward entry** (`hrm_act_v2.py:255`) — unpacked from the carry: `z_H, z_L = carry.z_H, carry.z_L`.

For the **very first ACT step** (reset_flag=true), both streams start as the learned init vectors broadcast across all positions. For subsequent ACT steps, they are the **detached outputs** from the previous step (`hrm_act_v2.py:281`).

### Sequence Layout

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

The puzzle prefix is a **learned task-type embedding**, not a per-instance encoding. The flow:

1. Each example carries a `puzzle_identifiers` integer (shape `[batch]`) indicating its puzzle type.
2. `CastedSparseEmbedding` (`models/sparse_embedding.py:11-38`) maintains a weight table of shape `[num_puzzle_identifiers, puzzle_emb_ndim]` and looks up `weights[puzzle_identifiers]` → shape `[batch, puzzle_emb_ndim]`.
3. The result is reshaped to `[batch, puzzle_emb_len, hidden_size]` (zero-padded if `puzzle_emb_ndim` is not a multiple of `hidden_size`) and prepended to the token embeddings.

**For sudoku**, `num_puzzle_identifiers = 1` and every example has `puzzle_identifiers = 0` (`build_sudoku_dataset.py:109`). The embedding table is a single row `[1, 512]`, so every sudoku instance in every batch receives the **exact same** prefix vector — effectively a learned "I am a sudoku" context token. In contrast, multi-task datasets like ARC assign distinct identifiers per puzzle type (`build_arc_dataset.py:240`), giving each type its own learned prefix.

The puzzle prefix positions are **not tokens to predict** — the **labels** tensor from the dataset has shape `[batch, seq_len]` covering only the token positions.

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

**Only z_H is decoded.** z_L is never directly projected through an unembedding matrix. Both levels use the same `ReasoningModule.forward` (line 123-125), which adds `input_injection` to `hidden_states` before passing through transformer layers:

```python
def forward(self, hidden_states, input_injection, **kwargs):
    hidden_states = hidden_states + input_injection
    for layer in self.layers:
        hidden_states = layer(hidden_states=hidden_states, **kwargs)
    return hidden_states
```

Tracing through the call sites:

- **H_level** (`z_H = H_level(z_H, z_L)`): `hidden_states=z_H`, `input_injection=z_L` → transformer layers receive **z_H + z_L**
- **L_level** (`z_L = L_level(z_L, z_H + input_embeddings)`): `hidden_states=z_L`, `input_injection=z_H + input_embeddings` → transformer layers receive **z_L + z_H + input_embeddings**

| Level | Call signature | Transformer layers actually see |
|-------|---------------|-------------------------------|
| **H_level** | `H_level(z_H, z_L)` | **z_H + z_L** |
| **L_level** | `L_level(z_L, z_H + input_embeddings)` | **z_H + z_L + input_embeddings** |

**Key asymmetry:** `input_embeddings` (token + puzzle embeddings) is only injected into L_level, and it is re-injected on every L_level call. This means L_level is re-anchored to the raw input at every cycle, while H_level never directly sees the token embeddings — it only accesses them indirectly through whatever z_L has absorbed. H_level is a purely "abstract" reasoning stream operating over the two residual states; L_level is the token-grounded stream.

### Embedding and Unembedding Are Position-Agnostic

Both the embedding and unembedding are single shared matrices applied uniformly across all sequence positions — there is no per-cell specialization:

- **Embedding** (`embed_tokens`): a single `CastedEmbedding(vocab_size, hidden_size)` with weight shape `[vocab_size, hidden_size]`. Every cell's token indexes into the same table.
- **Unembedding** (`lm_head`): a single `CastedLinear(hidden_size, vocab_size, bias=False)` with weight shape `[vocab_size, hidden_size]`. Applied via `F.linear` broadcast across all token positions with the same weights.

For sudoku, this means the same `[11, 512]` matrix embeds all 81 cells, and the same `[11, 512]` matrix decodes all 81 cells. The model differentiates cells solely through **positional encodings** (learned `embed_pos` or RoPE) and **attention**, which route cell-specific information through the residual streams.

### The Two Unembedding Matrices

| Head | Module | Weight Shape | What it decodes | From which positions |
|------|--------|-------------|-----------------|---------------------|
| **lm_head** | `CastedLinear(hidden_size, vocab_size, bias=False)` | `[vocab_size, hidden_size]` | Label token logits | `z_H[:, puzzle_emb_len:]` (token positions) |
| **q_head** | `CastedLinear(hidden_size, 2, bias=True)` | `[2, hidden_size]` | Halt/Continue Q-values | `z_H[:, 0]` (first puzzle prefix position) |

Both are standard `CastedLinear` layers (`models/layers.py:43-59`) using `F.linear(input, weight, bias)`, so the weight matrix rows define the directions in hidden_size space that the model reads from.

### How the Forward Loop Works

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

## Key Takeaways

- z_H and z_L are initialized from **learned buffers** (`H_init`, `L_init`) of shape `[hidden_size]`, broadcast across all sequence positions
- On subsequent ACT steps, they carry forward as **detached** outputs from the previous step
- The sequence dimension has a **puzzle prefix** (positions `0..L-1`) prepended before **token positions** (`L..L+seq_len-1`)
- Only **z_H** is decoded: `lm_head` reads token positions, `q_head` reads position 0
- H_level transformer layers receive **z_H + z_L**; L_level layers receive **z_H + z_L + input_embeddings**
- **Asymmetry**: `input_embeddings` is re-injected into L_level every cycle (token-grounded), while H_level never directly sees token embeddings (abstract reasoning over residual states only)
- For sudoku: hidden_size=512, vocab_size=11, seq_len=81, puzzle_emb_len=1

## Related Files

- `/home/developer/HRM_explore/models/hrm/hrm_act_v2.py` — HRM model with hooks (initialization: 170-171, carry: 231-241, forward: 243-287)
- `/home/developer/HRM_explore/models/layers.py` — CastedLinear definition (lm_head and q_head are instances, lines 43-59)
- `/home/developer/HRM_explore/models/losses.py` — ACTLossHead wrapper (lines 40-101)
- `/home/developer/HRM_explore/config/arch/hrm_v1.yaml` — Default architecture config
- `/home/developer/HRM_explore/dataset/build_sudoku_dataset.py` — Sudoku dataset builder (vocab_size=11, seq_len=81)
