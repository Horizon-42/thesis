# ts_transformer — learned 4D trajectory prediction

Two transformer forecasters, **iTransformer** (ICLR 2024) and **PatchTST** (ICLR 2023),
integrated separately behind one data plane, one training harness, and one export seam.

The sibling `4dTrajectory/optimization` computes trajectories by **optimization** — direct
collocation over a point-mass dynamics model, with hard procedure constraints. This package
computes them by **data-driven learning** — transformers trained on observed ADS-B
arrivals, with no dynamics model at all. Both emit the **same evaluation records**, so
`python -m evaluation --input <dir>` grades either one against the identical regulatory
gates (lateral ≤ 106.75 m, vertical ∈ [−3.05, +6.10] m).

```
                observed arrival tracks (trajectory_data_process)
                                  │
        ┌─────────────────────────┴─────────────────────────┐
        │                                                   │
  flight_scenarios                                    flight_scenarios
        │                                                   │
  optimization/  (casadi, IPOPT)              ts_transformer/  (torch)
        │                                                   │
        └──────────────► evaluation/ ◄──────────────────────┘
                    same records, same gates
```

**Contents** —
[Status](#status) ·
[Glossary](#glossary) ·
[Layout](#layout) ·
[Running it](#running-it) ·
[Data selection & flight identity](#data-selection--flight-identity) ·
[Design](#design) ·
[Results on real KRDU data](#results-on-real-krdu-data) ·
[Flyability](#flyability--measuring-the-gap-this-baseline-deliberately-leaves-open) ·
[Instance normalisation](#instance-normalisation-is-off-by-default) ·
[Artifacts & contracts](#what-gets-written) ·
[Vendored code](#vendored-code) ·
[Testing](#testing) ·
[Scope & gaps](#deliberate-scope--not-bugs-do-not-fix-without-deciding-to)

## Status

Trained and evaluated on **real harvested ADS-B** (KRDU, 995 arrivals). Checkpoints were
retrained 2026-07-20 on the reproducible `flight_key` split (702 train / 141 val / 152 test);
prediction artifacts were regenerated the same day, **test split only**, after the repo-wide
flight-identity unification (see
[Data selection & flight identity](#data-selection--flight-identity)). Every number in this
README's result tables is read from those on-disk artifacts
(`4dTrajectory/outputs/KRDU/ts_pred_*/`). Earlier synthetic numbers are kept where they are
labelled as such, because two design decisions were made on synthetic data and the real run
either confirmed or corrected them.

**Scope:** this is a purely kinematic, single-aircraft baseline. No aerodynamic or dynamics
model is connected — by design, not by omission. See
[the dynamics section](#inputs-outputs-and-the-deliberate-absence-of-dynamics)
before changing anything here.

## Glossary

The abbreviations and terms of art this README (and `metrics.py` / the summary JSONs) use:

| term | meaning |
|---|---|
| **ADE** | **Average Displacement Error** — 3D distance between the predicted and observed position, averaged over every valid forecast step of a flight, then over flights. The standard headline metric of the trajectory-prediction literature. |
| **FDE** | **Final Displacement Error** — the same 3D distance, taken only at the **last** valid step (for a full approach: where it ended). Measures endpoint placement rather than the whole path. |
| **p95** | 95th percentile over the batch — the tail, where compounding error shows up before it moves the mean. |
| **lead time** | How many seconds ahead of the anchor a predicted step lies. Error *by lead time* is the only axis on which the two horizon modes compare fairly. |
| **anchor** | The last observed sample the model was conditioned on; records rebase time so the anchor is `t = 0`. |
| **`L` / `H` / `dt`** | Lookback steps / horizon steps / resample step in seconds. Defaults `L=60, dt=2 s` (120 s of history); `H=30` (window) or `H=300` (full). Baked into layer shapes — changing them means retraining. |
| **ENU** | Local **E**ast/**N**orth/**U**p Cartesian frame; here anchored at the runway threshold, so `(0,0,0)` is where an approach should end. |
| **cross-track / along-track** | Horizontal error decomposed across / along the observed track's own course — "beside the path" vs "ahead/behind on it". |
| **gates** | The evaluation thresholds every record is graded against: final lateral ≤ 106.75 m, vertical ∈ [−3.05, +6.10] m (FAA 8260.58D / 8260.3F derived; see `evaluation/thresholds.py`). |
| **`flight_key`** | The repo-wide flight identity `id_runway_icao24_landingTime` — the record filename stem, the train/val/test split key, and the observed CZML entity id. |
| **pp** | Percentage points (a difference of rates, e.g. 69.7% − 63.2% = +6.6 pp). |
| **chained forecast** | `recursive_forecast`: predict `H` steps, append them to the history, slide, repeat — the model reads its own output from pass 2 on. |
| **instance norm / RevIN** | Per-window normalisation that strips a window's absolute level before the model sees it (iTransformer `use_norm`, PatchTST `RevIN`). OFF here by default — see [the ablation](#instance-normalisation-is-off-by-default). |
| **horizon-capped** | A full-mode forecast the fixed `H` ended *before* the predicted path reached the threshold; its final state (what the gates judge) is a cap artifact, flagged `horizonCapped` in the record. |
| **ADS-B** | Automatic Dependent Surveillance–Broadcast — the aircraft-broadcast position reports the observed tracks come from (via the OpenSky history DB). |

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
| `flyability.py` | closed-form control inversion — what a predicted path would have required, vs the envelope |
| `synthetic.py` | synthetic arrivals, so the pipeline is runnable before real data lands |
| `vendor/` | upstream model code, byte-identical, with `LICENSE` + `PROVENANCE.md` each |

## Running it

Environment is conda **`aeroviz`** (Python 3.12) — the single thesis env: data acquisition
(`traffic`, `pyopensky`), CIFP parsing (`cifparse`, `arinc424`), `casadi`, `openap`, the
geospatial stack, and `torch`. The package code stays casadi-free by design, but it lives
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
    --output-dir 4dTrajectory/outputs/KRDU/ts_pred \
    --split test
python -m evaluation --input 4dTrajectory/outputs/KRDU/ts_pred
```

`--data` takes an arrivals file, a czml-input file, or a directory of either.
`predict` defaults to `--split test` — the only flights the model never saw. The split is
recorded in the checkpoint and keyed by `flight_key`, so re-predicting later selects exactly
the same flights (no leakage on a re-run).

## Data selection & flight identity

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

**One flight = one `flight_key`** (`id_runway_icao24_landingTime`,
`flight_scenarios.identity`). The raw data has no unique flight id — `id` is a copy of the
callsign and the same callsign flies daily — so uniqueness comes from `icao24` + landing
time. The key names everything about a flight: this package's record stems and its
train/val/test split, the optimizer's record stems, the comparison-CZML group, and (since
2026-07-20) the observed layer's CZML entity ids. Keying anything on the bare callsign is
how a split leaks and how namesake flights swap each other's data.

## Design

### Channels

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

### Two horizon modes

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
a 60 s horizon per pass, full mode over horizons up to 10 min. The honest comparison is error
against **lead time** (same seconds-ahead for both), which the results section reports.

> Naming: `4dTrajectory/optimization` already uses *rollout* for forward-integrating
> optimizer controls through the true dynamics. That is a different operation, so the ML
> chaining here is called `recursive_forecast` and never a rollout.

### Sizing (why the defaults are what they are)

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

### Inputs, outputs, and the deliberate absence of dynamics

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

**What "no dynamics" concretely means.** The only symbol this package imports from
`aerodynamic_model` is `GeodeticState` (`channels.py`) — a plain dataclass holding seven
floats, no equations. Compare what the optimizer imports from the same package:
`CasadiSimulator` (the point-mass dynamics) and `rollout_piecewise_constant` (true-dynamics
integration).

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

#### If dynamics is added later, these are the four routes

Listed in increasing order of intrusiveness, so a future change can pick deliberately rather
than drift into one:

1. **Post-hoc flyability check** — ✅ **DONE**, see
   [Flyability](#flyability--measuring-the-gap-this-baseline-deliberately-leaves-open)
   (`flyability.py`). Inverts the point-mass equations on the predicted trajectory to recover
   the required load factor / bank / thrust and reports what fraction sits inside the
   envelope. Does not touch training, and needs **no casadi** (the inversion is algebra), so
   it lives in this package and needs no second environment. Note that it measures the gap;
   it does not close it — routes 2–4 are still the ways to actually make predictions flyable.
2. **Post-hoc dynamics projection** — treat the prediction as a reference and solve for the
   nearest flyable trajectory with `CasadiSimulator`. Pulls casadi back in, so it has to run
   as a second stage in the `aeroviz` env.
3. **Soft physical constraints in the loss** — penalise out-of-envelope acceleration / turn
   rate during training. This is the "constrained LSTM" line (Shi, IEEE T-ITS) in the survey.
4. **Predict controls and integrate them** — structurally guarantees flyability, but needs a
   *differentiable* dynamics model; casadi cannot backpropagate into torch, so the point-mass
   model would have to be reimplemented in torch. Largest effort.

## Results on real KRDU data

995 arrivals across 6 runways, split **by flight** (`flight_key`) into 702 train / 141 val /
152 test. Both models, both horizon modes, 120-epoch cap with patience 15, `lr=5e-4`, on an
RTX 4060. Every prediction batch is graded by `python -m evaluation`; all numbers below are
read from the regenerated 2026-07-20 artifacts in `4dTrajectory/outputs/KRDU/ts_pred_*/`.

> These numbers replace an earlier set trained under the pre-`flight_key` split, which
> `hash(flight_key)` reproduces for only 552/995 flights. That partition was clean
> (train/val/test verified disjoint) but is not reproducible from current code, so it was
> retrained rather than quoted. Two conclusions did not survive the change of split; they are
> marked below. **Treat any single-split margin under ~1.5× as provisional** — this is one
> seed on one split, and that is the size of effect it turned out to move.

> **Run-to-run jitter, measured.** Re-running `predict` on the same checkpoints and data
> (CUDA, no retraining) reproduced the one-pass full-mode aggregate ADE to <0.1 m, while the
> chained-window cells moved 2–4% and one borderline flight crossed the lateral gate
> (3→4 passes for iTransformer full). Chaining amplifies floating-point nondeterminism the
> way it amplifies everything else. This is another reason for the provisional-margin rule
> above.

**Whole-approach prediction, graded at the threshold** (152 test flights). Directly
comparable across all four — every row predicts the complete remaining approach.

| model | mode | ADE mean/p95 | FDE mean/p95 | lateral mean/p95 | path deviation | flyable (obs. floor 63.2%) | gate pass |
|---|---|---:|---:|---:|---:|---:|---:|
| iTransformer | **full** | **1746 / 4539 m** | **2230 / 6021 m** | **767 / 2434 m** | 1608 m | 46.7% (−16.4 pp) | **4/152** |
| iTransformer | window (chained ×10) | 1930 / 6429 m | 2522 / 7436 m | 1130 / 3979 m | **1590 m** | **69.7% (+6.6 pp)** | 1/152 |
| PatchTST | full | 1912 / 4996 m | 3110 / 7474 m | 2105 / 6001 m | 2105 m | 28.9% (−34.2 pp) | 0/152 |
| PatchTST | window (chained ×10) | 2476 / 6414 m | 3873 / 12568 m | 3183 / 8470 m | 4194 m | 32.9% (−30.3 pp) | 0/152 |

**Displacement error at matched lead times** — the axis on which the two horizon modes can be
compared fairly. Recomputed directly from the exported records: 3D distance between the
predicted and observed position at the same `t`, mean over the flights whose record reaches
that lead (`n`; records end at the threshold, so `n` falls with lead — long leads survive
only for long approaches, and 600 s has n=1, too few to quote).

| model | mode | 10 s | 30 s | 60 s | 120 s | 300 s |
|---|---|---:|---:|---:|---:|---:|
| iTransformer | window (chained) | 294 m | **380 m** | **711 m** | **1354 m** | 4765 m |
| iTransformer | full | 943 m | 886 m | 964 m | 1323 m | **3802 m** |
| PatchTST | window (chained) | 663 m | 812 m | 1252 m | 2518 m | 6972 m |
| PatchTST | full | **231 m** | 392 m | 749 m | 1519 m | 3852 m |
| *n (of 152)* | | *152* | *152* | *152* | *119–146* | *49–80* |

### What the numbers say

**Whole approach → full, and this is the robust result.** One-pass full mode beats chained
window on lateral error at the threshold for both architectures, on both splits — ≈1.5× on
the mean (iTransformer 1130/767, PatchTST 3183/2105) and 1.4–1.6× on p95. Once a chained
pass goes wrong the next nine extrapolate from a wrong history. Training directly for the
long horizon beats chaining a short one.

**Did NOT survive the split change: "the compounding cost lands in the tail, not the mean".**
On the old split the lateral mean was 1.5× worse while p95 was 2.2× worse. Here the two
ratios are the same to within noise. The tail effect is still visible in *final*
displacement — PatchTST FDE p95 12568 m chained vs 7474 m one-pass (1.68×) against a 1.25×
mean — but that is a claim about FDE, not about lateral error at the threshold, and it is
not the clean 1.5-vs-2.2 story originally written here.

**Short lead → PatchTST; long lead → iTransformer, now with a caveat.** Within full mode,
PatchTST clearly leads at 10 s (231 vs 943 m) and the two are within 2% by 300 s (3852 vs
3802 m) — on this run the long-lead reversal is inside the provisional band at 300 s; the
earlier run's raw-tensor curves (which see past threshold truncation) had iTransformer
clearly ahead at 600 s (5407 vs 6962 m). The mechanism is architectural either way:
PatchTST is channel-**independent** (`TSTiEncoder` — every channel forecast in isolation by
shared weights) while iTransformer's attention runs *across* variates; for a turning
aircraft east and north are strongly coupled and PatchTST cannot represent that by
construction. Near-straight flight costs it nothing, so **the coupling only starts paying
once the turn develops.**

**Did NOT survive the split change: "zero gate passes in all four runs".** iTransformer now
passes 4/152 in full mode and 1/152 chained. The point stands in substance — ~3% is not a
usable approach predictor — but "zero, always" was a property of that split, not of the
method. The 106.75 m lateral limit is FAA containment for a *planned or flown* approach,
while this is a *forecast* extrapolating 5–10 minutes from 120 s of history; the number
quantifies the distance between a statistical prediction and a certifiable trajectory.

**Flyability and accuracy disagree, deliberately.** iTransformer window is the *most* flyable
run (69.7%, above the observed tracks' 63.2% floor) while losing to full mode on every error
metric; PatchTST full is the least flyable (28.9%) at a comparable ADE. Chaining short
windows produces smooth, conservative paths — easy to fly, not where the aircraft went.
Neither metric substitutes for the other.

**Real data is much harder than synthetic**, as it should be. Synthetic approaches are
straight-in, so the model only has to extrapolate a line. Real arrivals are vectored, and
*when* the turn onto final happens is a controller's decision — information a single-aircraft
model with no traffic context and no ATC intent input structurally cannot have. That is the
survey's central open problem, and this baseline is where it gets measured from.

## Flyability — measuring the gap this baseline deliberately leaves open

`predict` writes a `flyability_report.json` beside its records and prints a one-line
summary. It answers: *what controls would this trajectory have required, and do they fit
inside the airframe's envelope?*

The load-factor point-mass model inverts in **closed form** — no solver, no casadi. Its two
rotational equations rearrange directly:

```
A = psi_dot   * V * cos(gamma) / g   = n sin(mu)      n  = hypot(A, B)      -> load factor
B = gamma_dot * V / g + cos(gamma)   = n cos(mu)      mu = atan2(A, B)      -> bank
```

`n` fixes the required lift coefficient (→ stall check), lift fixes drag, and the along-track
equation `T = m (V_dot + g sin(gamma)) + D` gives thrust explicitly. Earth-frame transport
terms are subtracted first, or the inversion bills the aircraft for a bank that the rotating
tangent plane produced rather than the pilot.

**Read the delta, not the absolute rate.** Run against *real flown tracks*, the check first
scored 0/149 fully flyable. Those trajectories were flown by real aircraft, so the check was
wrong, not the flights. The cause: `thrust_negative`. Median required thrust on a real
arrival is 0.43 kN — essentially idle — and a negative requirement simply means the aircraft
needed **more drag than a clean airframe has**: speedbrake, flaps, gear. Every approach does
this; a single clean-configuration drag polar cannot represent it. So `thrust_negative` is a
**soft** violation (reported, not counted as unflyable), and the report leads with the
comparison against the observed tracks measured by the identical code, because both sides
carry the same polar bias:

```
flyability: 46.7% of predictions fully flyable vs 63.2% of the observed tracks (-16.4 pp)
```

The observed baseline is the floor, not 100%. `HARD_VIOLATIONS` (stall, bank, load factor,
thrust over max) vs `SOFT_VIOLATIONS` (thrust below idle) is where that judgement lives.

Each flight is judged against its **own airframe** (`report_for_records` takes one
`Aircraft` per flight; the report carries the `fleet` and per-type `envelopes`) — the first
version shared one A320 envelope and mis-graded ~44% of a mixed batch. `Cl_max` comes from
`aero_params_for_aircraft` (2.7 for an A320), **not** from `LoadFactorSimulator`'s hardcoded
1.5 — the two disagree by 80%, and `aero_params.py` documents itself as the source of truth
for the stall model.

**Flyability is not a quality metric on its own.** A straight line is perfectly flyable and
completely wrong; see the instance-norm table below, where the *worse* predictor scores
*higher* on flyability by being blander. Pair it with the error metrics or it misleads.

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

### Re-ablated on real data — the synthetic result holds

The obvious objection to the synthetic table was that instance norm exists for distribution
shift, and synthetic straight-ins have none. Real KRDU arrivals have plenty (mixed types,
6 runways, vectored downwinds, wind). Re-run there, all 8 cells, same hyperparameters
(`ep=120 lr=5e-4 patience=15 seed=1337`), each graded on its own checkpoint's held-out test
split — artifacts in `4dTrajectory/outputs/KRDU/_ablation_norm/`:

| model | mode | norm | epochs | val loss | ADE | ADE p95 | FDE | lateral p95 | flyable |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| iTransformer | window | on | 20 | 0.0970 | 2508 m | 9051 m | 4942 m | 14291 m | 81.5% |
| iTransformer | window | **off** | 33 | **0.0511** | **2090 m** | **5585 m** | **2871 m** | **6224 m** | 71.7% |
| iTransformer | full | on | 27 | 0.3439 | 2400 m | 8186 m | 4365 m | 14280 m | 34.9% |
| iTransformer | full | **off** | 34 | **0.0577** | **1756 m** | **4434 m** | **2190 m** | **2560 m** | **46.1%** |
| PatchTST | window | on | 67 | 0.0845 | **2571 m** | 7172 m | 5683 m | 14498 m | 89.3% |
| PatchTST | window | **off** | 61 | **0.0648** | 2580 m | **6474 m** | **3987 m** | **8465 m** | 29.6% |
| PatchTST | full | on | 32 | 0.4016 | 5105 m | 10852 m | 7299 m | 14480 m | 64.5% |
| PatchTST | full | **off** | 57 | **0.0903** | **1903 m** | **5221 m** | **2942 m** | **5675 m** | 27.0% |

*(flyable = fraction of predictions fully inside the envelope, each flight judged against its
own airframe; the observed tracks score 63.2% measured the same way. This table is from the
ablation's own training runs — its "off" cells are separate checkpoints from the headline
results above, which is why the numbers differ slightly.)*

**Off wins 19 of the 20 accuracy comparisons, with one tie.** Every cell on validation loss,
every cell on FDE, every cell on ADE p95, every cell on lateral error at the threshold; on
mean ADE it is 3 wins and one dead heat (PatchTST window, 2580 vs 2571 m — a 0.4% difference
in the cell where off wins the other four metrics). A sweep that lopsided is not run-to-run
variance, which matters because the individual gaps here are smaller than on synthetic data
and a single metric on a single cell would not have settled it.

**The tell is the lateral p95 column.** All four instance-norm-on cells land at
14.28–14.50 km — a near-constant number across two architectures and both horizon modes.
That is the signature of a model that cannot place the endpoint at all: strip the absolute
level and the prediction ends at a distance set by the frame, not by the flight. Off spans
2.6–8.5 km, i.e. it varies with how hard the flight was.

**Flyability moves the opposite way, and that is the point of having both metrics.**
Instance norm scores *better* on flyability in 3 of the 4 cells — PatchTST window 89.3% vs
29.6%, i.e. the configuration that is 2.2× worse at the threshold looks three times more
flyable. It is not flying better; it is predicting smoother, blander paths that are easy to
fly and far from the truth. A straight line is perfectly flyable and completely wrong.
(iTransformer full is the exception, 46.1% off vs 34.9% on — there instance norm is bad
enough that even the smoothing does not save it.) Flyability bounds whether a prediction
*could* be flown; only the error metrics say whether it is the right trajectory, and neither
substitutes for the other.

So `use_norm` / `revin` stay **off** by default, now on real-data evidence rather than
synthetic. `--instance-norm` still turns them on.

## What gets written

```
<output-dir>/
  <flight_key>_states.json                     predicted + observed, side by side
  <flight_key>_eval.json                       the evaluation record
  references/<flight_key>_reference_eval.json  the observed track, same contract
  summary.json                                 the manifest — load_records reads ONLY this
```

The filename stem IS the flight identity (`flight_key`), shared with the optimizer's record
filenames and the comparison-CZML group key, so learned and optimized records for one flight
always share a stem. Summary rows carry the full identity too (`id`, `icao24`, `runway`,
`landing_time_utc`).

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
- **No flyability *fix*, only a measurement.** `flyability.py` reports how far outside the
  envelope a prediction sits; nothing projects it back inside. That is routes 2–4 above.
  The check also judges against one clean-configuration drag polar and one `Cl_max`, which
  is why it is calibrated against the observed tracks rather than read absolutely — a
  configuration-aware polar (flap/gear schedule) would make the absolute rate meaningful.
- **The declared aircraft type is `"UNK"` for all 3747 harvested flights** (`czml_export`
  hardcodes it), but `_resolve_aircraft` recovers the real airframe from `icao24` via the
  OpenAP lookup — 20 distinct types across 400 KRDU arrivals (A320 224, B738 38, E75L 25,
  B737 25, CRJ9 23, …). Only genuinely unresolvable flights use the `--aircraft-type`
  fallback. **A batch is a real fleet, so nothing may assume one airframe for it** — the
  flyability check first shipped doing exactly that and graded ~44% of flights against an
  A320's `Cl_max` and max thrust. What is still missing is coverage checking: how often the
  fallback is actually hit is not reported per batch.
