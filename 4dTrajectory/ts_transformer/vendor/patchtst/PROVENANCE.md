# PatchTST — vendoring provenance

| | |
|---|---|
| Upstream | https://github.com/yuqinie98/PatchTST |
| Commit | `204c21efe0b39603ad6e2ca640ef5896646ab1a9` (2023-08-11) |
| License | Apache-2.0 — see `LICENSE` |
| Paper | *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*, ICLR 2023 — https://arxiv.org/abs/2211.14730 |
| Vendored | 2026-07-19 |

Taken from the **`PatchTST_supervised/`** tree. The repo also ships
`PatchTST_self_supervised/` (masked-patch pretraining, a separate `src/` layout with its
own trainer and callbacks); that is not vendored — this project trains supervised from
scratch. If pretraining on unlabelled ADS-B ever becomes interesting, that tree is the
thing to vendor next, and it is a genuinely separate integration.

## What was copied and what changed

| This file | Upstream file | Change |
|---|---|---|
| `model.py` | `models/PatchTST.py` | imports rewritten to relative; body untouched |
| `backbone.py` | `layers/PatchTST_backbone.py` | imports rewritten to relative; body untouched |
| `layers.py` | `layers/PatchTST_layers.py` | copied whole, unmodified |
| `revin.py` | `layers/RevIN.py` | copied whole, unmodified |

All four are complete files — unlike the iTransformer vendoring, nothing needed extracting,
because this dependency closure is self-contained (torch + numpy only) and pulls no
research-variant baggage.

## Config contract

`Model.__init__` takes `configs` plus a long tail of keyword arguments that upstream
defaults and never overrides from the CLI. Attributes read off `configs`:

```
enc_in  seq_len  pred_len  e_layers  n_heads  d_model  d_ff  dropout
fc_dropout  head_dropout  individual  patch_len  stride  padding_patch
revin  affine  subtract_last  decomposition  kernel_size
```

`ts_transformer/config.py` supplies all of them.

## Forward signature

```python
model(x)    # x: [B, seq_len, n_channels]  ->  [B, pred_len, n_channels]
```

Note this differs from iTransformer's four-argument call. `ts_transformer/models.py`
normalises the two behind one `build_model(config)` / `model(x)` interface, so training
code never branches on architecture.

## Two behaviours worth knowing before reading results

**Channel independence.** The backbone reshapes `[B, C, L]` to `[B*C, L]` and runs every
channel through the *same* weights with no cross-channel attention (`TSTiEncoder` — the
`i` is for "channel-independent"). For trajectory data this means the model cannot, by
construction, learn that a left turn couples east and north displacement — each channel is
forecast in isolation. That is the documented PatchTST design and a large part of why it
generalises, but it is exactly the property iTransformer inverts (its attention is *across*
variates). The contrast is the reason both are worth having side by side here.

**RevIN.** With `revin=True` (the default) each input window is instance-normalised on the
way in and denormalised on the way out, so the model sees only the window's own shape, not
its absolute position. Combined with an ENU frame anchored at the runway threshold, this
means absolute position information reaches the model only through the channels that
survive normalisation — worth remembering when a prediction looks shape-correct but
offset.
