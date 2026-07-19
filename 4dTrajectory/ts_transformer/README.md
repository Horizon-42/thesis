# ts_transformer — learned 4D trajectory prediction

Two transformer forecasters, **iTransformer** (ICLR 2024) and **PatchTST** (ICLR 2023),
integrated separately behind one data plane, one training harness, and one export seam.

The sibling `4dTrajectory/optimization` answers *what trajectory should this aircraft fly?*
(direct collocation, a dynamics model, hard procedure constraints). This package answers a
different question — *what trajectory will it fly?* — learned from observed ADS-B arrivals
with no dynamics model at all. Both emit the **same evaluation records**, so
`python -m evaluation --input <dir>` grades either one against the identical regulatory gates
(lateral ≤ 106.75 m, vertical ∈ [−3.05, +6.10] m).

```
                observed arrival tracks (trajectory_data_process)
                                  │
        ┌─────────────────────────┴─────────────────────────┐
        │                                                   │
  flight_scenarios                                    flight_scenarios
        │                                                   │
  optimization/  (casadi, IPOPT)              ts_transformer/  (torch)
  "what SHOULD it fly"                        "what WILL it fly"
        │                                                   │
        └──────────────► evaluation/ ◄──────────────────────┘
                    same records, same gates
```

## Status

Trained and evaluated on **real harvested ADS-B** (KRDU, 995 arrivals, 2026-07-19) — see
[First real-data results](#first-real-data-results-krdu). Earlier synthetic numbers are kept
where they are labelled as such, because two of the design decisions below were made on
synthetic data and the real run either confirmed or corrected them.

**Scope:** this is a purely kinematic, single-aircraft baseline. No aerodynamic or dynamics
model is connected — by design, not by omission. See
[Inputs, outputs, and the deliberate absence of dynamics](#inputs-outputs-and-the-deliberate-absence-of-dynamics)
before changing anything here.

## Layout

| File | What it is |
|---|---|
| `config.py` | `TSConfig` — the one namespace both vendored models read, serialised into every checkpoint |
| `channels.py` | the feature contract: geodetic states ⇄ threshold-anchored ENU channels |
| `dataset.py` | track loading, uniform resampling, windowing + masking, the by-flight split, normalisation |
| `models.py` | one `model(x) -> y` interface over the two vendored architectures |
| `train.py` | masked-MSE training loop, early stopping, self-contained checkpoints |
| `forecast.py` | one-pass and chained (`recursive_forecast`) prediction, threshold truncation |
| `metrics.py` | ADE / FDE plus the along-track / cross-track / altitude decomposition |
| `export.py` | evaluation records + `summary.json` manifest, via the optimizer's own record emitters |
| `synthetic.py` | synthetic arrivals, so the pipeline is runnable before real data lands |
| `vendor/` | upstream model code, byte-identical, with `LICENSE` + `PROVENANCE.md` each |

## Running it

Environment is conda **`aeroviz`** (Python 3.12) — the single thesis env: data acquisition
(`traffic`, `pyopensky`), CIFP parsing (`cifparse`, `arinc424`), `casadi`, `openap`, the
geospatial stack, and now `torch`. The package code stays casadi-free by design, but it lives
here so one env runs everything.

```bash
conda activate aeroviz
pip install -r 4dTrajectory/ts_transformer/requirements.txt

TS=4dTrajectory/ts_transformer/__main__.py

# train
python $TS train --data trajectory_data_process/outputs/landings/KRDU --airport KRDU \
    --model itransformer --horizon-mode window \
    --output-dir 4dTrajectory/outputs/KRDU/ts_itr_window

# predict the held-out split, then grade it exactly like an optimizer batch
python $TS predict --checkpoint 4dTrajectory/outputs/KRDU/ts_itr_window/checkpoint.pt \
    --data trajectory_data_process/outputs/landings/KRDU --airport KRDU \
    --output-dir 4dTrajectory/outputs/KRDU/ts_pred
python -m evaluation --input 4dTrajectory/outputs/KRDU/ts_pred
```

`--data` takes an arrivals file, a czml-input file, or a directory of either.
`predict` defaults to `--split test` — the only flights the model never saw.

⚠️ **Directory mode is selective, and says so.** A harvest directory holds five overlapping
views of the same flights:

| file | what it is |
|---|---|
| `<ICAO>_<RWY>_arrivals.json` | truncated at the 25 km ring — **what training uses** |
| `<ICAO>_<RWY>_landings.json` | the same flights, untruncated — duplicates |
| `<ICAO>_combined_czml_input.json` | all runways merged — duplicates again |
| `<ICAO>_<RWY>_heading_rejected.json` | flights the harvester **threw out** (bad heading) |
| `<ICAO>_local_rejected.json` | local circuits, not arrivals |

`select_flight_files` takes the first matching pattern only (`*_arrivals.json` →
`*_czml_input*.json` → `*_landings.json`), never mixes them, always excludes `*_rejected*`,
and prints what it skipped. A naive `glob("*.json")` loaded every flight three times over
plus the known-bad ones — which a loss curve cannot show you. Passing an explicit **file**
path bypasses all filtering: you chose it.

## Two horizon modes

Both models are fixed lookback→horizon (`L → H`); `L` and `H` are baked into the layer
shapes, so changing either means retraining.

**`--horizon-mode window`** — short `H` (default 30 steps = 60 s). To cover a whole approach,
`recursive_forecast` chains passes: predict 30, append them to the history, slide, predict
again. From the second pass on the model is reading its own output, so error compounds.

**`--horizon-mode full`** — `H` covers the whole approach (default 300 steps = 10 min) in one
pass. No compounding, because every predicted step came from real observed history. Approaches
shorter than `H` are padded and **masked out of the loss** — without that mask the model
learns to reproduce its own zero padding and every forecast tail collapses.

Having both is the point: the gap between them measures what chaining actually costs.

⚠️ The two modes' headline ADE/FDE are **not** directly comparable — window mode averages over
a 60 s horizon, full mode over horizons up to 10 min (and over anchors whose remainder is
mostly padding). The honest comparison is `metrics.error_by_horizon`, which reports error
against *lead time*, so a chained window forecast and a one-pass full forecast can be read at
the same number of seconds ahead.

> Naming: `4dTrajectory/optimization` already uses *rollout* for forward-integrating
> optimizer controls through the true dynamics. That is a different operation, so the ML
> chaining here is called `recursive_forecast` and never a rollout.

## Sizing (why the defaults are what they are)

Measured from the **3747 harvested arrivals** (5 airports, truncated at the 25 km entry ring):

| p5 | p25 | p50 | p75 | p90 | p95 | p99 |
|---:|---:|---:|---:|---:|---:|---:|
| 235 s | 271 s | 328 s | 533 s | 607 s | 651 s | 920 s |

Defaults: `dt = 2 s`, `L = 60` (120 s), `H = 30` (window) / `300` (full, 600 s).

`L = 60, H_full = 300` covers the complete remaining approach for **97.8%** of flights.
`H_full = 150` would have covered only **57.6%**.

Two wrong guesses got corrected here, both by measuring rather than reasoning:

- The first draft used `L = 60 @ 4 s` (4 min of lookback) and **skipped 5 of every 6 flights**
  as "shorter than one window".
- The second used the straight-line estimate "25 km at 120 m/s ⇒ 3.5–5 min" and sized
  `H_full = 150`. Real arrivals are **vectored** — downwind legs, base turns, the occasional
  hold — so the flown path is far longer than the straight-line distance to the ring, and the
  real median is 328 s with a tail past 900 s. That guess covered barely half an approach.

## First real-data results (KRDU)

995 arrivals across 6 runways, split by flight into 697 train / 149 val / 149 test
(96k training windows). Both models, both horizon modes, 120-epoch cap with patience 15,
`lr=5e-4`, on an RTX 4060. Every prediction batch was graded by `python -m evaluation`.

**Displacement error at matched lead times.** This is the only axis on which the two
horizon modes can be compared — the headline ADE/FDE cannot, because they average over
different horizon-length distributions.

| model | mode | 10 s | 30 s | 60 s | 120 s | 300 s | 600 s |
|---|---|---:|---:|---:|---:|---:|---:|
| iTransformer | window | **259 m** | **390 m** | **687 m** | — | — | — |
| iTransformer | full | 587 m | 611 m | 840 m | **1502 m** | **3227 m** | **6135 m** |
| PatchTST | window | **158 m** | 409 m | 995 m | — | — | — |
| PatchTST | full | 253 m | 399 m | 830 m | 1914 m | 3983 m | 7142 m |

**Whole-approach prediction, graded at the threshold** (149 test flights). Directly
comparable across all four — every row predicts the complete remaining approach.

| model | mode | lateral mean | lateral p95 | path deviation | gate pass |
|---|---|---:|---:|---:|---:|
| iTransformer | **full** | **1070 m** | **3136 m** | **1895 m** | 0/149 |
| iTransformer | window (chained ×10) | 1594 m | 6898 m | 1918 m | 0/149 |
| PatchTST | full | 1804 m | 4666 m | 2488 m | 0/149 |
| PatchTST | window (chained ×10) | 2815 m | 7354 m | 3152 m | 0/149 |

### What the numbers say

**Short lead → window; long lead or whole approach → full.** Window mode wins inside the
60 s it was trained for (687 vs 840 m), but chaining it out to a whole approach degrades
badly. The cost of compounding lands in the **tail, not the mean**: lateral mean is only
1.5× worse, but p95 is 2.2× worse (6898 vs 3136 m). Once a chained pass goes wrong, the
next nine extrapolate from a wrong history. Training directly for the long horizon beats
chaining a short one.

**iTransformer beats PatchTST at long lead, for a structural reason.** PatchTST is
channel-**independent** (`TSTiEncoder` — every channel forecast in isolation by shared
weights), while iTransformer's attention runs *across* variates. For a turning aircraft
east and north are strongly coupled, and PatchTST cannot represent that by construction.
Note the reversal at 10 s, where PatchTST is *better* (158 vs 259 m): at that scale the
aircraft is nearly straight and channel independence costs nothing. **The coupling only
starts paying once the turn develops** — which is exactly what the two architectures'
designs predict.

**Zero gate passes, in all four runs, is the honest result — not a failure.** The 106.75 m
lateral limit is FAA containment for a *planned or flown* approach; this is a *forecast*
extrapolating 5–10 minutes from 120 s of history. The number quantifies the distance
between a statistical prediction and a certifiable trajectory — the flyability gap the
survey in `4dTrajectory/docs` is about, now measured rather than asserted.

**Real data is much harder than synthetic**, as it should be: iTransformer window went from
286 m (synthetic) to 423 m ADE. Synthetic approaches are straight-in, so the model only has
to extrapolate a line. Real arrivals are vectored, and *when* the turn onto final happens is
a controller's decision — information a single-aircraft model with no traffic context and no
ATC intent input structurally cannot have. That is the survey's central open problem, and
this baseline is where it gets measured from.

## Channels

Six channels in a local ENU frame **anchored at the runway threshold**:

```
e, n, u      metres from the threshold (u is height above it)
ve, vn, vu   m/s
```

The evaluation state `(lat, lon, alt, V, psi, gamma, m)` is a bad regression target directly:
`lat`/`lon` waste float range on the airport's absolute position, `psi` wraps at ±π (a model
regressing it averages 179° and −179° to 0°, pointing the aircraft backwards, right where the
turn onto final happens), and `m` is not observable from ADS-B at all.

Predicting velocity *components* makes the reconstruction exact and the convention automatic:
`psi = atan2(vn, ve)` **is** the modeling layer's math-ENU heading, so there is no remaining
place to substitute a compass bearing by accident. `m` is carried, never predicted.

## Inputs, outputs, and the deliberate absence of dynamics

> **Read this before "improving" anything here.** This package is a *kinematic baseline*.
> No aerodynamic or dynamics model is connected, and that is a scope decision, not an
> oversight or an unfinished TODO.

Both models consume and produce the same tensor shape — channels in, channels out:

```
input   x : [B,  60, 6]         seq_len=60 (120 s of lookback @ dt=2 s), 6 channels
output  y : [B,  30, 6]         window mode (60 s ahead)
        y : [B, 300, 6]         full mode  (600 s ahead)
```

Same six channels on both sides (`e, n, u, ve, vn, vu`). The two architectures differ only
in how they read that tensor: iTransformer makes each **channel** a token (attention *across*
variates), PatchTST patches each channel along **time** and runs them independently (no
cross-channel coupling at all).

**Not in the input:** timestamps (`x_mark = None` — the grid is uniform, so `t` is implicit),
mass, aircraft type, aerodynamic parameters, wind/weather, other traffic, runway or procedure
geometry (present only implicitly, as the frame origin).

**Not in the output:** controls (thrust, bank, load factor) — which is exactly why the exported
records carry `controls == []` — and any notion of uncertainty or multimodality.

### What "no dynamics" concretely means

The only symbol this package imports from `aerodynamic_model` is `GeodeticState`
(`channels.py`) — a plain dataclass holding seven floats. It contains no equations. Compare
what the optimizer imports from the same package: `CasadiSimulator` (the point-mass dynamics)
and `rollout_piecewise_constant` (true-dynamics integration).

```
optimization/     initial state ─► NLP (point-mass dynamics + hard constraints)
                                ─► controls ─► true-dynamics integration ─► states
                                   ▲ every step is bound by the equations of motion

ts_transformer/   observed channels ─► Transformer ─► predicted channels
                                       ▲ curve fitting; no equations anywhere
```

So **nothing here guarantees** a predicted trajectory is flyable: speeds may leave the
envelope, turn rates may imply impossible bank angles, the implied thrust may exceed the
engines, the implied lift coefficient may exceed `Cl_max` (stall). This is the
"statistically plausible but physically unflyable" / *flyability* problem the survey in
`4dTrajectory/docs` names explicitly.

That is the correct shape for a baseline. The trajectory-prediction literature reports
ADE/FDE on exactly this kind of purely kinematic, data-driven model, and mixing dynamics in
now would make it impossible to say what the learned component contributes on its own.

### If dynamics is added later, these are the four routes

Listed in increasing order of intrusiveness, so a future change can pick deliberately rather
than drift into one:

1. **Post-hoc flyability check** — invert the point-mass equations on the predicted
   trajectory to recover the required load factor / bank / thrust, and report what fraction
   sits inside the envelope. Does not touch training, and needs **no casadi** (the inversion
   is algebra), so it can live in this package and needs no second environment. Highest value
   per unit of work: it is what makes the head-to-head against the optimizer meaningful —
   *"the learned model tracks the observed path closely but X% of its output is unflyable;
   the optimizer is 100% flyable but is not what the aircraft actually flew."*
2. **Post-hoc dynamics projection** — treat the prediction as a reference and solve for the
   nearest flyable trajectory with `CasadiSimulator`. Pulls casadi back in, so it has to run
   as a second stage in the `aeroviz` env.
3. **Soft physical constraints in the loss** — penalise out-of-envelope acceleration / turn
   rate during training. This is the "constrained LSTM" line (Shi, IEEE T-ITS) in the survey.
4. **Predict controls and integrate them** — structurally guarantees flyability, but needs a
   *differentiable* dynamics model; casadi cannot backpropagate into torch, so the point-mass
   model would have to be reimplemented in torch. Largest effort.

## Instance normalisation is OFF by default

Both architectures ship a per-window instance normalisation — iTransformer's `use_norm`,
PatchTST's `RevIN` — on by default upstream, and a large part of why they win on generic
forecasting benchmarks. **Here it hurts**, measurably:

| model | mode | instance norm | epochs | ADE | FDE | cross-track p95 | alt p95 |
|---|---|---|---:|---:|---:|---:|---:|
| iTransformer | window | on (upstream default) | 54 | 771 m | 1641 m | 2595 m | 55.8 m |
| iTransformer | window | **off** | 90 | **286 m** | **341 m** | **438 m** | **28.2 m** |
| iTransformer | full | on (upstream default) | 56 | 1972 m | 3283 m | 5432 m | 77.2 m |
| iTransformer | full | **off** | 74 | **303 m** | **312 m** | **468 m** | **30.0 m** |
| PatchTST | window | on (upstream default) | 49 | 910 m | 2021 m | 2766 m | 55.5 m |
| PatchTST | window | **off** | 88 | **672 m** | **1178 m** | **1842 m** | **40.0 m** |
| PatchTST | full | on (upstream default) | 52 | 2268 m | 3696 m | 5413 m | 75.2 m |
| PatchTST | full | **off** | 79 | **701 m** | **687 m** | **1707 m** | **45.6 m** |

*(synthetic KRDU arrivals, 120 flights, trained to early stop at patience 25. Relative sizes
are the point; the absolute values are synthetic and mean nothing on their own.)*

Off wins in all four cells, on both models and in both horizon modes, and by 2.4–6.5× on ADE.
Note also that instance norm *converges sooner* (49–56 epochs vs 74–90) — it is not
undertrained, it has hit a worse optimum.

The reason is structural. Instance norm exists to remove distribution shift, on the assumption
that a window's absolute level is nuisance and its shape is signal. In a threshold-anchored ENU
frame that assumption is inverted: **absolute position is the signal.** Where the aircraft is
determines where the turn onto final happens, when the descent starts, and where the approach
ends. Normalising it away leaves the model guessing at the geometry it most needs.

Turn it back on with `--instance-norm` — it is worth re-ablating on real data, where the
distribution shift the technique was designed for (mixed aircraft types, runway configurations,
wind) is genuinely present and may pay for what it costs.

## What gets written

```
<output-dir>/
  <flight>_states.json                     predicted + observed, side by side
  <flight>_eval.json                       the evaluation record
  references/<flight>_reference_eval.json  the observed track, same contract
  summary.json                             the manifest — load_records reads ONLY this
```

Records are **reference-shaped**: `controls == []`. That is the contract, not a shortcut — a
learned predictor emits no control schedule, and `evaluation.records` reads an empty control
list as exactly that. Records are built by
`4dTrajectory/optimization/evaluation_export.py` rather than hand-rolled here, so there is one
definition of the record shape. (That module is casadi-free, which is why it imports into the
torch env.)

`final_time_s` is always taken from the last predicted sample, never from `pred_len * dt` —
threshold truncation makes those differ, and `evaluation.records` rejects a mismatch above 1e-6.

## Vendored code

`vendor/itransformer/` (MIT, commit `c2426e6`) and `vendor/patchtst/` (Apache-2.0, commit
`204c21e`) are copied from upstream **byte-identical**, with only import paths rewritten, so a
future `git diff` against a newer upstream stays readable. Each carries its `LICENSE` and a
`PROVENANCE.md` recording exactly what was copied, what was dropped, and why. Adapting a model
to this project belongs in `models.py`, never in a vendored file.

Dropped from iTransformer: the Reformer/Flowformer/Flashformer/Informer attention variants,
and with them the `reformer_pytorch` and `einops` dependencies.

## Testing

```bash
python -m pytest 4dTrajectory/ts_transformer/tests -q --import-mode=importlib
```

Picked up automatically by `run_all_tests.sh`, which already lists `4dTrajectory` — since
`aeroviz` now carries torch, one invocation runs the whole repo.

The tests pin contracts, not quality: the heading convention in both directions, the padding
mask, the by-flight split, threshold truncation, and — the important one — that an exported
record satisfies the real `evaluation.records.record_from_dict` validator and that a written
batch is loadable by the real manifest-only `load_records`.

## Deliberate scope — NOT bugs, do not "fix" without deciding to

These are what make this a baseline. Each is a defensible starting point that later work may
choose to extend, but none is an accident:

- **No dynamics / aerodynamics.** Purely kinematic. See
  [the dynamics section](#inputs-outputs-and-the-deliberate-absence-of-dynamics) for what that
  means and the four routes if it is ever added.
- **Single aircraft, no interaction.** One track in, one track out — no traffic context. The
  survey in `4dTrajectory/docs` identifies multi-aircraft interaction and ATC intent as the
  central open problem in terminal-airspace prediction; this is deliberately the single-aircraft
  baseline that work would build on.
- **Deterministic point prediction.** No uncertainty, no multimodality. The literature has moved
  toward generative/probabilistic terminal models (CVAE, diffusion) precisely because runway
  configuration and vectoring make the future genuinely multimodal; these two models cannot
  represent that, and a point prediction is the honest baseline against which they are measured.

## Known gaps — actual unfinished work

- **Only KRDU so far.** 3747 arrivals are harvested across 5 airports (KMSY, KRDU, KSJC,
  KSMF, KSTL); only KRDU has been trained. Cross-airport generalisation is untested, and the
  ENU frame is per-runway-threshold, so a pooled model is a real design question, not a
  bigger `--data` glob.
- **`--instance-norm` not re-ablated on real data.** The OFF default was decided on synthetic
  approaches; real data carries the distribution shift the technique was built for.
- **No comparison CZML.** Predictions are not yet fed to
  `aeroviz-4d/python/build_scenario_comparison_czml.py`, which expects
  `optimizer_states` / `simulator_states` keys. The states file here writes
  `predicted_states` / `observed_states`.
- **Aircraft type is `"UNK"` for all 3747 harvested flights** (`czml_export` hardcodes it), so
  every flight is built with the `--aircraft-type` fallback (default `A320`). Mass only sets a
  carried constant, but Vref and the threshold-crossing height set the target state the gates
  measure against — so the whole batch is judged against one assumed aircraft. Extracting real
  types (ICAO24 → registry) would make the target per-flight.
