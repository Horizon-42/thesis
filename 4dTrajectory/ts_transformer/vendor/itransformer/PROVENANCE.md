# iTransformer — vendoring provenance

| | |
|---|---|
| Upstream | https://github.com/thuml/iTransformer |
| Commit | `c2426e68ca13f74aaec08045c5c724d8ad328124` (2025-07-17) |
| License | MIT — see `LICENSE` (Copyright (c) 2022 THUML @ Tsinghua University) |
| Paper | *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting*, ICLR 2024 — https://arxiv.org/abs/2310.06625 |
| Vendored | 2026-07-19 |

## Why vendored rather than pip-installed or submoduled

Upstream is a research codebase, not a library: it has no `setup.py`, its modules import
each other by top-level path (`from layers.Embed import ...`), and those names collide
with the sibling PatchTST vendoring (both ship a `layers/Embed.py` and a
`layers/SelfAttention_Family.py`, with *different* contents). Vendoring the reachable
subset under a package prefix is what lets both models coexist in one process.

## What was copied and what changed

| This file | Upstream file | Change |
|---|---|---|
| `model.py` | `model/iTransformer.py` | imports rewritten to relative (`from .encoder import ...`); body untouched |
| `encoder.py` | `layers/Transformer_EncDec.py` | `DecoderLayer` + `Decoder` removed (iTransformer is encoder-only); `ConvLayer`, `EncoderLayer`, `Encoder` verbatim |
| `attention.py` | `layers/SelfAttention_Family.py` + `utils/masking.py` | **extracted** `FullAttention`, `AttentionLayer`, `TriangularCausalMask` verbatim; see below |
| `embed.py` | `layers/Embed.py` | **extracted** `DataEmbedding_inverted` verbatim; the six unused embeddings dropped |

Nothing else was altered — no reformatting, no renames, no behavioural edits. Every class
body is byte-identical to upstream so that a future `git diff` against a newer commit is
readable.

## Dropped upstream dependencies

`layers/SelfAttention_Family.py` does `from reformer_pytorch import LSHSelfAttention` and
`from einops import rearrange` **at module scope**, to support the `iReformer` /
`iFlowformer` / `iFlashformer` / `iInformer` variants in `model/`. Those variants are not
used here, so both imports — and the pip dependencies behind them — were dropped along
with the classes that need them (`FlowAttention`, `FlashAttention`, `ProbAttention`,
`ReformerLayer`). If you ever want one of those variants, re-vendor the full file and add
`reformer_pytorch` + `einops` to `requirements.txt`.

## Config contract

`Model.__init__` takes a single `configs` object and reads these attributes:

```
seq_len  pred_len  output_attention  use_norm  embed  freq  dropout
class_strategy  d_model  n_heads  d_ff  e_layers  activation  factor
```

`ts_transformer/config.py` supplies all of them. Note `embed`, `freq`, `factor` and
`class_strategy` are read but never affect this code path (`DataEmbedding_inverted`
ignores `embed`/`freq`; `factor` only matters to `ProbAttention`, which is dropped;
`class_strategy` is stored and unused upstream too) — they are kept so the object stays a
drop-in for upstream's `run.py` argparse namespace.

## Forward signature

Upstream keeps the four-argument Autoformer-family signature even though the inverted
encoder-only design uses only the first two:

```python
model(x_enc, x_mark_enc, x_dec, x_mark_dec)   # -> [B, pred_len, N]
```

`x_dec` / `x_mark_dec` are accepted and ignored; `x_mark_enc` may be `None`, which is what
this project passes (no calendar covariates — time enters as a channel instead).
