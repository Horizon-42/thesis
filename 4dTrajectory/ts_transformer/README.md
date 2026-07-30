# ts_transformer — learned 4D trajectory prediction

Two transformer forecasters, **iTransformer** (ICLR 2024) and **PatchTST** (ICLR 2023),
integrated separately behind one data plane, one training harness, and one export seam.

The sibling `4dTrajectory/optimization` computes trajectories by **optimization** — direct
collocation over a point-mass dynamics model, with hard procedure constraints. This package
computes them by **data-driven learning** — transformers trained on observed ADS-B arrivals.
Its default state-output path is the original kinematic baseline; its opt-in control-output
path integrates learned controls through a differentiable Torch twin of the same point-mass
dynamics. Both emit the **same evaluation records**, so
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

The current code keeps three explicit prediction contracts: `normalized`, `full`, and
`window`. They share channels, output heads, losses, split policy, and anchor policy, while
their target clocks and inference strategies are dispatched independently. Checkpoints record
the selected mode and output length; changing modes requires retraining.

Trained and evaluated on **real harvested ADS-B** (KRDU, 995 arrivals). Checkpoints were
retrained twice on 2026-07-20, both times on the reproducible `flight_key` split
(702 train / 141 val / 152 test): first after the repo-wide flight-identity unification
(see [Data selection & flight identity](#data-selection--flight-identity)), then again
under the **transport-consistent channels + physical-velocity fit** (B3.1 — see
[Channels](#channels); the B3 rename `ve/vn/vu → edot/ndot/udot` makes `load_checkpoint`
refuse every earlier checkpoint). Every number in this README's result tables is read from
the current on-disk artifacts (`4dTrajectory/outputs/KRDU/ts_pred_*/`); the intermediate
generation is parked in `outputs/KRDU/_pre_b3_transport/`. Earlier synthetic numbers are
kept where they are labelled as such, because two design decisions were made on synthetic
data and the real run either confirmed or corrected them.

**Scope:** `prediction_output=state` remains the purely kinematic, single-aircraft baseline
used by the recorded experiments below. `prediction_output=control` is a separate,
checkpointed architecture; it adds per-flight aircraft conditioning and a differentiable
dynamics rollout. See [the output architecture](#state-baseline-and-dynamics-constrained-control-output).

## Glossary

The abbreviations and terms of art this README (and `metrics.py` / the summary JSONs) use:

| term | meaning |
|---|---|
| **ADE** | **Average Displacement Error** — 3D distance between the predicted and observed position, averaged over every valid forecast step of a flight, then over flights. The standard headline metric of the trajectory-prediction literature. |
| **FDE** | **Final Displacement Error** — the same 3D distance, taken only at the **last** valid step (for a full approach: where it ended). Measures endpoint placement rather than the whole path. |
| **p95** | 95th percentile over the batch — the tail, where compounding error shows up before it moves the mean. |
| **normalized progress `tau`** | Position in the predicted remainder, from anchor `tau=0` to endpoint `tau=1`, independent of physical duration. |
| **anchor** | The last observed sample the model was conditioned on; records rebase time so the anchor is `t = 0`. |
| **`L` / `N` / `H` / `dt`** | Observed lookback / normalized output segments / fixed-time horizon steps / input resample step. Defaults: `L=60`, model-specific `N`, `H_window=30`, `H_full=300`, `dt=2 s`. |
| **ENU** | Local **E**ast/**N**orth/**U**p Cartesian frame; here anchored at the runway threshold, so `(0,0,0)` is where an approach should end. |
| **cross-track / along-track** | Horizontal error decomposed across / along the observed track's own course — "beside the path" vs "ahead/behind on it". |
| **gates** | The evaluation thresholds every record is graded against: final lateral ≤ 106.75 m, vertical ∈ [−3.05, +6.10] m (FAA 8260.58D / 8260.3F derived; see `evaluation/thresholds.py`). |
| **`flight_key`** | The repo-wide flight identity `id_runway_icao24_landingTime` — the record filename stem, the train/val/test split key, and the observed CZML entity id. |
| **pp** | Percentage points (a difference of rates, e.g. 73.7% − 63.2% = +10.5 pp). |
| **`final_time_s`** | Learned physical time from the observed anchor to the predicted endpoint. |
| **instance norm / RevIN** | Per-window normalisation that strips a window's absolute level before the model sees it (iTransformer `use_norm`, PatchTST `RevIN`). OFF here by default — see [the ablation](#instance-normalisation-is-off-by-default). |
| **ADS-B** | Automatic Dependent Surveillance–Broadcast — the aircraft-broadcast position reports the observed tracks come from (via the OpenSky history DB). |

## Layout

| File | What it is |
|---|---|
| `config.py` | `TSConfig` — the one namespace both vendored models read, serialised into every checkpoint |
| `channels.py` | the feature contract: geodetic states ⇄ threshold-anchored ENU channels |
| `dataset.py` | track loading, input resampling, mode-dispatched targets, by-flight split, normalisation |
| `models.py` | model construction over the two vendored encoders and selected state/control head |
| `prediction_outputs.py` | typed state/control outputs, final-time head, bounded controls and duration partition |
| `train.py` | state loss or differentiable control-rollout loss, early stopping, checkpoints |
| `forecast.py` | independent normalized, one-pass full, and recursive-window inference strategies |
| `metrics.py` | ADE / FDE plus the along-track / cross-track / altitude decomposition |
| `export.py` | evaluation records + `summary.json` manifest, via the optimizer's own record emitters |
| `flyability.py` | closed-form control inversion — what a predicted path would have required, vs the envelope |
| `synthetic.py` | synthetic arrivals, so the pipeline is runnable before real data lands |
| `vendor/` | upstream model code, byte-identical, with `LICENSE` + `PROVENANCE.md` each |

## Running it

Environment is conda **`aeroviz`** — the thesis env: data acquisition
(`traffic`, `pyopensky`), CIFP parsing (`cifparse`, `arinc424`), `casadi`, `openap`, the
geospatial stack, and `torch`. The package code stays casadi-free by design, but it lives
here so one env runs everything.

```bash
conda activate aeroviz
pip install -r 4dTrajectory/ts_transformer/requirements.txt

TS=4dTrajectory/ts_transformer/__main__.py

# train
python $TS train \
    --data trajectory_data_process/outputs/harvest/KRDU/arrivals/manifest.json \
    --airport KRDU \
    --model itransformer --n-segments 128 \
    --output-dir 4dTrajectory/outputs/KRDU/ts_itr_normalized_time

# opt-in control output: bounded schedule -> differentiable CasADi-equivalent rollout
python $TS train \
    --data trajectory_data_process/outputs/harvest/KRDU/arrivals/manifest.json \
    --airport KRDU --model itransformer \
    --prediction-output control --horizon-mode normalized --n-segments 64 \
    --batch-size auto \
    --output-dir 4dTrajectory/outputs/KRDU/ts_itr_control

# independently replay the retained best checkpoint on fixed-anchor train + validation
python $TS evaluate-fit \
    --checkpoint 4dTrajectory/outputs/KRDU/ts_itr_normalized_time/checkpoint.pt \
    --data trajectory_data_process/outputs/harvest/KRDU/arrivals/manifest.json

# predict the held-out split, then grade it exactly like an optimizer batch
python $TS predict --checkpoint 4dTrajectory/outputs/KRDU/ts_itr_normalized_time/checkpoint.pt \
    --data trajectory_data_process/outputs/harvest/KRDU/arrivals/manifest.json \
    --airport KRDU \
    --output-dir 4dTrajectory/outputs/KRDU/ts_pred \
    --split test
python -m evaluation --input 4dTrajectory/outputs/KRDU/ts_pred
```

`--data` takes an airport harvest directory, its `arrivals/` directory, or the explicit
`arrivals/manifest.json`; repeat the flag to train one model over multiple airports. Legacy
flight arrays and arbitrary JSON directories are rejected.
`predict` defaults to `--split val`; routine development never opens outer-test. The split is
recorded in the checkpoint and keyed by `flight_key`. The checkpoint also carries the exact
arrival-manifest SHA-256 and every flight's canonical source SHA-256; prediction rejects a
rebuilt or changed manifest instead of silently intersecting its roster with an old split.
Retrain after rebuilding arrivals.

Every completed `train` call automatically performs the same deterministic fit replay on
both outer-train and outer-validation: the retained best checkpoint runs under
`model.eval()`, dropout is disabled, every flight uses anchor `L-1`, batches are sequential,
and no shuffle is involved. `evaluate-fit` exposes that exact operation independently for an
existing checkpoint; it never evaluates outer-test.

`run_ts_pipeline.py` namespaces prediction output and frontend categories by split
(`..._<horizon-mode>_<split>`). It defaults to `--split development`, which publishes separate
train and validation artifacts. `--split test` is rejected unless the same command also carries
`--release-test`. That release creates `test_release.json` beside the checkpoint before loading
any test trajectory, binds it to the checkpoint/data/split hashes, and records each claimed
flight. Started or completed flights cannot be evaluated again. The frontend manifest and
comparison index also carry an explicit `datasetSplit` field.

### Pooled training and cross-validation

The top-level runner now has two explicit training scopes. `per-airport` retains one model per
airport; `pooled` runs one CV search and one final fit over all selected manifests, then uses
that checkpoint to publish each airport separately:

```bash
# All discovered K-airports -> one checkpoint per model, then train/validation outputs.
conda run -n aeroviz python run_ts_pipeline.py \
  --training-mode pooled \
  --models itransformer \
  --coordinate-frame runway-aligned

# Inspect every resolved command without training.
conda run -n aeroviz python run_ts_pipeline.py \
  --training-mode pooled --models itransformer --dry-run
```

Each development cell executes `cross-validate -> final train -> predict train/val ->
evaluate/publish`.
The split boundary is deliberately nested:

1. lock airport-qualified outer train/validation/test flights;
2. construct K folds from **outer-train only** and select hyperparameters by mean
   airport-macro fold loss;
3. fit the selected configuration on outer-train with outer-validation early stopping;
4. after every analysis and model decision is permanently frozen, explicitly release test once:

   ```bash
   conda run -n aeroviz python run_ts_pipeline.py \
     --skip-train --models itransformer --split test --release-test
   ```

The release ledger is an experiment artifact and must not be deleted or reset. If development
continues after seeing test, that partition becomes a development test and a new temporal
holdout is required for the final claim.

`cross_validation/cv_results.json` records split digests, every fold score and explicit
`outer_validation_used: false` / `outer_test_used: false` guards. `best_config.json` contains
only the selected `TSConfig` overrides. `--skip-cv` reuses those artifacts only if every
arrival-manifest digest still matches; otherwise final training uses the base configuration.

Pooled mode defaults to one full-trajectory example per flight: the fixed training and
validation anchor is `L-1`, immediately after the first complete lookback. Every eligible
training flight appears exactly once per epoch and the complete flight order is reshuffled
for the next epoch. The loss weights each flight by the inverse flight count of its airport,
normalized to mean one, so every airport has the same total epoch weight without oversampling
smaller airports. `--random-train-anchor` switches to a separate rolling/replanning dataset:
it still uses every flight once, but independently selects one uniformly random valid anchor
for that flight on each epoch. Random anchors require at least 60 s of remaining supervision
by default (`--random-train-anchor-min-future-s`), and their per-flight choice is derived from
the training seed, epoch and stable flight identity rather than roster order. The train-only
normalizer uses the same airport-then-flight weighting rather than duration-weighted moments.
For control-output models, the independent anchor-eligibility policy also requires observed
speed at the anchor to be at least `1.10 ×` the aircraft's sea-level stall speed. This keeps
stopped/ground reports outside the airborne point-mass ODE; state-output random anchors retain
the temporal-only policy. Per-epoch audit rows record the policy and excluded-candidate count.
Controlled fixed-vs-random comparisons can additionally set
`--training-cohort-min-future-s 60` on both arms. This independent floor is applied only to
the outer-training roster at fixed anchor `L-1`; validation remains complete and fixed-anchor.
The checkpoint, metadata, summary and artifact path record the filtered cohort separately from
the random-anchor eligibility rule.
Random-anchor training, CV, predictions and
frontend categories receive a `_random_anchor` path suffix, so they cannot overwrite the
fixed-anchor baseline. `--cv-folds`, `--cv-parameters`, and
`--cv-epochs` expose the search dimensions and budgets. CV exhaustively evaluates the fixed
grids; the default `n_segments,learning_rate,d_model` grid has 27 candidates and no random
candidate sampling.
The default CV budget is 36 epochs with patience 6.

Run pooled CV only, followed by automatic result plotting, with:

```bash
conda run -n aeroviz python run_ts_cv.py
```

Use `--batch-size 2048` to force a larger batch, or leave the default
`--batch-size auto` to run the CUDA training-step probe.

Plots and flat CSV tables are kept beside the source artifacts under the same run directory:

```text
ts_<model>_normalized_time/
  cross_validation/
    cv_results.json
    best_config.json
  history.json                 # present after final training
  plots/
    index.md                   # one-page chart index
    plot_manifest.json
    cv_candidate_scores.{png,svg}
    cv_hyperparameter_effects.{png,svg}
    cv_airport_heatmap.{png,svg}
    cv_candidates.csv
    cv_airport_scores.csv
    training_curves.{png,svg}
    training_airport_loss.{png,svg}
    training_epochs.csv
```

After final training, refresh the same directory with
`conda run -n aeroviz python plot_ts_results.py <run-directory>`.

Run the controlled history-length ablation with:

```bash
conda run -n aeroviz python run_ts_history_ablation.py
```

The default candidates are `L=30,60,90`. All candidates use the same flight roster, outer-
train folds, batch size, and anchor population. The common minimum anchor is fixed at
`max(L)-1`, so changing L only changes how many samples before that anchor enter the model;
the predicted remainder is identical. Results are written under
`4dTrajectory/outputs/POOLED/ts_<model>_normalized_time_history_length_ablation/` as JSON,
flat CSV tables, PNG/SVG plots, and `plots/index.md`. Use `--dry-run` to inspect the protocol
and `--config-overrides <best_config.json>` to hold a previously selected architecture fixed.

The runner defaults to `--batch-size auto`. It probes actual FP32 forward/backward/Adam steps
for the selected architecture and normalized output grid; an explicit integer disables
probing. State output uses the largest successful power of two. Control output phase-shifts
the duration partition across probe rows, exercising the per-segment maxima that determine
batched rollout graph depth, then backs off one successful power of two as a safety margin
for sharper partitions later in training. The result remains runtime-specific rather than a
hard-coded GPU-name table.

For a throughput-based batch measurement, run the standalone outer-train-only benchmark while
the GPU is otherwise idle:

```bash
conda run --no-capture-output -n aeroviz \
  python -u detect_ts_best_batch.py \
  --model itransformer \
  --n-segments 128 \
  --coordinate-frame enu
```

Unlike `--batch-size auto`, this script doubles candidates through the configured maximum,
executes repeated real in-memory batch construction + FP32 forward/backward/Adam steps, and selects the highest
median samples/second rather than the largest allocation that fits. Split membership is
computed from manifest `flight_key` values before track files are opened: only outer-train
source tracks are loaded, and validation/test counts enter the audit JSON without their
trajectory values being read. The default result is written under
`4dTrajectory/outputs/POOLED/batch_benchmarks/`; the final terminal line is
`BEST_BATCH_SIZE=<integer>`. Pass a CV `best_config.json` through `--config-overrides` to
benchmark an already selected architecture, and use the resulting integer with
`run_ts_coordinate_ablation.py --batch-size <integer>`.

`--coordinate-frame enu` is the unchanged baseline. `runway-aligned` rotates horizontal
position and chart-velocity channels into along-runway/cross-runway axes while retaining the
same six-channel tensor contract; use separate runs as an ablation. Neither mode edits the
vendored architectures.

Run the paired pooled coordinate-frame ablation with:

```bash
conda run -n aeroviz python run_ts_coordinate_ablation.py \
  --model itransformer \
  --outputs eval
```

The script discovers the same fixed K-airport roster once, runs ENU and runway-aligned CV
with the same seed, candidates, folds, sampling budget, and batch size, and rejects the
comparison if their recorded split or fold digests differ. It selects the frame and that
frame's hyperparameters using outer-train airport-macro CV loss, trains only the winner with
outer-validation early stopping, and leaves outer-test sealed by default. Pass `--release-test`
only after the experiment is permanently frozen to perform the one-shot test prediction.
The losing frame never sees outer-validation or outer-test. The auditable decision is written
under `4dTrajectory/outputs/POOLED/ts_<model>_normalized_time_coordinate_frame_ablation/` as
`coordinate_frame_ablation.json`. Use `--reuse-cv` only to reuse two
artifacts that pass the complete paired-run contract check. Immediately before the first
test prediction, the decision file is marked `outer_test_evaluation_started`; the script
then refuses to evaluate test again from the same experiment directory, including after a
partially failed test stage.

## Data selection & flight identity

Training reads only the `records` roster in `arrivals/manifest.json`. The manifest is built
from `assigned` tracks with a published CIFP TCH/glidepath, cropped from the final 25 km
entry to the measured landing anchor; local circuits remain in the audit counts but cannot
enter training. An orphan JSON beside `records/` is ignored, and a duplicate `flight_key`
in the roster raises. There is no pattern priority or legacy fallback.

**One flight = one `flight_key`** (`id_runway_icao24_landingTime`,
`flight_scenarios.identity`). The raw data has no unique flight id — `id` is a copy of the
callsign and the same callsign flies daily — so uniqueness comes from `icao24` + landing
time. The key names everything about a flight: this package's record stems and its
train/val/test split, the optimizer's record stems, the comparison-CZML group, and (since
2026-07-20) the observed layer's CZML entity ids. Keying anything on the bare callsign is
how a split leaks and how namesake flights swap each other's data.

## Design

### Channels

Six channels in a local ENU chart **anchored at the runway threshold**:

```
e, n, u           metres from the threshold (u is height above it)
edot, ndot, udot  d(e)/dt, d(n)/dt, d(u)/dt — m/s in the chart
```

The evaluation state `(lat, lon, alt, V, psi, gamma, m)` is a bad regression target directly:
`lat`/`lon` waste float range on the airport's absolute position, `psi` wraps at ±π (a model
regressing it averages 179° and −179° to 0°, pointing the aircraft backwards, right where the
turn onto final happens), and `m` is not observable from ADS-B at all.

Predicting velocity *components* makes the reconstruction exact and the convention automatic:
after undoing the transport factors, `psi = atan2(V_north, V_east)` **is** the modeling
layer's math-ENU heading, so there is no remaining place to substitute a compass bearing by
accident. `m` is carried, never predicted.

**Transport-consistent since 2026-07-20 (B3.1).** The velocity channels are the exact time
derivatives of the position channels, not the raw physical ENU components: the chart
coordinates are scaled angles (`n = Δlat_rad·a`, `e = Δlon_rad·a·cos lat₀`), so a physical
velocity picks up the full-transport Jacobian the optimizer's geodetic RHS encodes —
`ndot = V_north·a/(R_M+h)`, `edot = V_east·a·cos(lat₀)/((R_N+h)·cos lat)` with the exact
WGS84 curvature radii (`geokit.wgs84_curvature_radii`). The factors are 1 ± ~0.3% over a
TMA. Before the fix the channels mixed chart positions with raw physical velocities, so
integrating the velocity channels did not reproduce the position channels; the upstream
least-squares velocity fit (`flight_scenarios.state_samples_from_track`) carried the
mirror-image bias (chart scales where physical was meant), fixed at the same time. Measured
on all 995 KRDU arrivals (median whole-track drift of ∫v dt against the position channels,
m/min): east 3.5 → 2.4, north 2.7 → 2.7, up 0.45 → 0.45 — what remains is unbiased
least-squares smoothing, no longer a systematic. (Fixing only the channels and not the fit
would have *added* a +0.33% north systematic — 8.6 m/min — which is why both seams moved
together.) The rename `ve/vn/vu → edot/ndot/udot` is deliberate: the channel tuple is
serialised into every checkpoint and `load_checkpoint` refuses a mismatch, so pre-change
checkpoints fail loudly instead of silently mis-scaling every velocity.

### Three horizon modes

Select one contract with `--horizon-mode normalized|full|window`. The modes coexist; none is
an alias or compatibility branch.

| mode | training target | inference |
|---|---|---|
| `normalized` (default) | Complete remainder at `tau_i=i/N`; no padding | One pass; `final_time_s` reconstructs physical timestamps; no geometric truncation or fixed cap |
| `full` | `H_full=300` physical `dt=2 s` nodes; short remainders padded and loss-masked | One pass; fixed-dt timestamps; cut at closest threshold approach; cap at 600 s |
| `window` | One complete `H_window=30` physical `dt=2 s` target per anchor | Recursively append each 60 s prediction, slide the history, and continue to `H_full`; then apply the same threshold/cap rule as `full` |

The state and `final_time_s` heads remain shared. In `normalized`, the duration head defines
the state-node clock. In `full/window`, state nodes retain their fixed `dt` clock and the head
is an auxiliary remaining-time estimate. Target-grid construction lives in `time_grids.py`;
mode-specific inference is dispatched in `forecast.py`; fixed/random anchor policy remains a
separate dataset choice.

```bash
# normalized complete remainder
conda run -n aeroviz python run_ts_pipeline.py --horizon-mode normalized

# one-pass 600 s full horizon
conda run -n aeroviz python run_ts_pipeline.py \
  --horizon-mode full --full-horizon-steps 300

# recursive 60 s windows, chained to the same 600 s cap
conda run -n aeroviz python run_ts_pipeline.py \
  --horizon-mode window --window-horizon-steps 30 --full-horizon-steps 300
```

The default anchor is fixed at `L-1`, so every flight contributes its earliest full-trajectory
forecast once per epoch; flight order is reshuffled between epochs. Pass
`--random-train-anchor` only when training a rolling predictor that must start from later
approach phases as well. That mode selects one valid anchor per flight and epoch. Validation
and exported prediction remain fixed at `L-1`. For control experiments whose deployment
metric is physical-time accuracy, `--checkpoint-selection-metric
fixed-anchor-common-grid-ade` also makes LR scheduling and early stopping use deterministic
fixed-anchor common-grid validation ADE; the native model-clock loss remains a diagnostic.
When fixed and random arms must use an identical 60-second-capable training roster, pass
`--training-cohort-min-future-s 60` to both arms; it never filters validation.
The frozen train/validation protocol is recorded in
[`docs/2026-07-30_random_anchor_experiment_plan.zh.md`](docs/2026-07-30_random_anchor_experiment_plan.zh.md).

`N` controls output resolution, not forecast seconds. It is serialized in checkpoints and
included in the default cross-validation grid (`64, 128, 256`). The objective combines
normalized state MSE, scaled final-time MSE, position/velocity displacement consistency,
and an explicit terminal-position term. Displacement consistency is normalized by position
scale so its gradient does not grow with N. Held-out physics/accuracy ablations selected
iTransformer `N=64` and PatchTST `N=256`; both use kinematic weight `3.0` and terminal
weight `0.02`. The full pooled iTransformer CUDA validation curve selected epoch 161;
training therefore uses a 180-epoch cap with patience 20 and always reloads the best epoch.
Validation and CV select on the same joint loss. See
[`docs/normalized_time_and_control_output.zh.md`](docs/normalized_time_and_control_output.zh.md)
for the exact state and control contracts.

Validation history, CV folds and exported prediction summaries also persist raw-node
position/velocity RMSE, heading-consistency p95, turn-rate p95, acceleration p95 and jerk
p95. Prediction batches retain per-flight values plus fleet `median/mean/p95/max`; comparisons
and the ablation selector use fleet p95 against the observed-track baseline. These metrics
always consume the model's untouched nodes and explicit segment durations, so they are also
valid for the non-uniform control rollout. The complete experiment protocol and
validation/test tables are in
[`docs/2026-07-27_kinematic_weight_epoch_ablation.zh.md`](docs/2026-07-27_kinematic_weight_epoch_ablation.zh.md).

PatchTST remains enabled as a comparison model, but its selected `N=256` state-head run is
not physically usable (outer-test flyability 0.0% and raw jerk fleet p95 about 2204 m/s³).
`N=256` is only its best held-out ADE among the tested grids, not a claim of flyability.

The validation-only comparison, future-dispersion analysis and deterministic/multi-candidate
coverage report are generated by `run_ts_predictability_report.py`; the current illustrated
HTML is at
`4dTrajectory/outputs/POOLED/ts_time_parameterization_predictability_report/report.html`.

### Full versus recursive window

Both models are fixed lookback→horizon (`L → H`); `L` and `H` are baked into the layer
shapes, so changing either means retraining.

**`--horizon-mode window`** — short `H` (default 30 steps = 60 s). To cover a whole approach,
the window forecaster chains passes: predict 30, append them to the history, slide, predict
again. From the second pass on the model is reading its own output, so error compounds.

**`--horizon-mode full`** — `H` covers the whole approach (default 300 steps = 10 min) in one
pass. No compounding, because every predicted step came from real observed history. Approaches
shorter than `H` are padded and **masked out of the loss** — without that mask the model
learns to reproduce its own zero padding and every forecast tail collapses.

Having both is the point: the gap between them measures what chaining actually costs.

Native training/validation ADE is not directly comparable across the two fixed-time modes:
window loss covers one 60 s pass, while full loss covers up to 600 s. Exported whole-approach
predictions are comparable because both are evaluated after window recursion on the same
physical timestamps; the report also publishes error against lead time.

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

**Why not `dt = 1 s`?** Considered for the 2026-07-20 retrain (B3.2) and deliberately not
taken. Keeping the same time coverage at `dt = 1 s` needs `L = 120, H = 600` — roughly 2×
the training cost — for very little information: the source reports at ≤ 1 Hz with ragged
gaps (a finer grid mostly interpolates), and the velocity channels come from a **15 s**
least-squares window fit, so their bandwidth is unchanged by a denser grid. `--dt` remains
a CLI knob if a future dataset (e.g. 1 Hz radar) justifies it; resizing `L`/`H` with it is
mandatory, per the coverage math above.

### State baseline and dynamics-constrained control output

The output strategy is explicit and serialized in every checkpoint:

```text
prediction_output=state (default)
  observed channels [B,L,6] -> Transformer -> states [B,N,6] + final_time_s [B]

prediction_output=control
  observed channels [B,L,6] -> Transformer encoder features ┐
  per-flight mass/aero data -> condition encoder            ├-> bounded controls [B,N,3]
                                                            └-> non-uniform durations [B,N]
  anchor geodetic state + controls + durations -> Torch dynamics -> state endpoints [B,N,7]
```

The six observed channels remain `(e, n, u, edot, ndot, udot)`. iTransformer makes each
channel a token; PatchTST patches each channel along time. In control mode the state forecast
projector is removed. Time/patches are pooled within each channel, but the ordered channel
axis is retained and flattened as `[B,C*d_model]`; an unlabeled mean across channels is
forbidden because both vendored encoders lack channel-index identity. A small MLP fuses this
ordered representation with eight scaled per-flight quantities: mass, maximum thrust, wing
area, `Cl_max`, `Cd0`, induced-drag `k`, stall threshold, and stall-drag coefficient.

The control contract is:

```text
controls[..., 0] = thrust_N       bounded by [0, aircraft max thrust]
controls[..., 1] = bank_rad       bounded by [-pi/4, pi/4]
controls[..., 2] = load_factor    bounded by [0.5, 2.0]

segment_durations_i = softmax(duration_logits)_i * final_time_s
sum(segment_durations) = final_time_s
```

Control mode currently requires `--horizon-mode normalized`. It needs no inverse-control
labels. Uniform truth nodes are interpolated onto `cumsum(segment_durations)` before the
rollout-state loss is evaluated, so every predicted endpoint and target share one physical
timestamp; fit metrics use the same alignment. Its loss contains rollout-state, final-time, terminal,
dimensionless control-effort, and control-smoothness terms. Position/velocity consistency is
structural because every output state comes from one dynamics integration.

`aerodynamic_model/torch_dynamics.py` is deliberately a discrete twin of
`CasadiSimulator`, not a cheaper training surrogate. Each substep creates the same moving
local ENU frame, evaluates the same ISA density, drag polar, `Cl_max`/stall transition and
realized-load limit, advances the same explicit RK4 equations, and converts position and
velocity through WGS84 ECEF into the next local frame. Physical rollout tensors use float64
because subtracting local offsets from Earth-scale ECEF coordinates in float32 loses
sub-metre information. `control_rollout_integrator_dt_s` defaults to the replay value 0.5 s.

This makes control-output trajectories dynamically generated, but does not reproduce the
optimizer's hard procedure/path constraints; learned controls can still leave the intended
operating envelope. The state-output path remains unchanged so its historical results remain
an independently measurable kinematic baseline.

#### Four physics integration routes

The repository now contains routes 1 and 4; routes 2 and 3 remain distinct alternatives:

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
4. **Predict controls and integrate them** — ✅ **DONE as an opt-in output strategy**. The
   differentiable Torch rollout is numerically contract-tested against CasADi for normal and
   stalled steps, non-uniform multi-segment endpoints, and gradients through controls and
   durations.

## Historical results on real KRDU data (pre-normalized-time architecture)

995 arrivals across 6 runways, split **by flight** (`flight_key`) into 702 train / 141 val /
152 test. Both models, both horizon modes, 120-epoch cap with patience 15, `lr=5e-4`, on an
RTX 4060. Every prediction batch is graded by `python -m evaluation`; all numbers below are
read from the on-disk artifacts in `4dTrajectory/outputs/KRDU/ts_pred_*/` — the
**2026-07-20 B3 retrain** under the transport-consistent channels and the physical-velocity
fit (a ≤ 0.3% rescale of the velocity channels; see [Channels](#channels)).

> This is the third training generation. The first (pre-`flight_key` split) is not
> reproducible from current code and is never quoted; the second (identity retrain, same
> day) lives in `4dTrajectory/outputs/KRDU/_pre_b3_transport/`. Same split, same seed, same
> recipe across generations two and three — yet sub-1.5× margins still moved (e.g. gate
> passes 4→0, chained lateral mean ±15%). **Treat any margin under ~1.5× as provisional**;
> only conclusions that held across generations are stated as findings below.

> **Run-to-run jitter, measured** (on the pre-B3 checkpoints). Re-running `predict` on the
> same checkpoints and data (CUDA, no retraining) reproduced the one-pass full-mode
> aggregate ADE to <0.1 m, while the chained-window cells moved 2–4% and one borderline
> flight crossed the lateral gate. Chaining amplifies floating-point nondeterminism the way
> it amplifies everything else. This is another reason for the provisional-margin rule
> above.

**Whole-approach prediction, graded at the threshold** (152 test flights). Directly
comparable across all four — every row predicts the complete remaining approach. `capped` =
flights whose forecast was still short of the threshold when `H` ran out (their gate
verdicts are cap artifacts; the count is model behaviour, it moves between retrains).

| model | mode | ADE mean/p95 | FDE mean/p95 | lateral mean/p95 | path deviation | flyable (obs. floor 63.2%) | gate pass | capped |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| iTransformer | **full** | **1756 / 4656 m** | **2082 / 6002 m** | **868 / 2750 m** | 1754 m | 46.1% (−17.1 pp) | 0/152 | 14 |
| iTransformer | window (chained ×10) | 1772 / 5261 m | 2486 / 7679 m | 1302 / 6576 m | **1728 m** | **73.7% (+10.5 pp)** | 0/152 | 4 |
| PatchTST | full | 1961 / 5075 m | 2945 / 6618 m | 2016 / 5598 m | 2267 m | 25.0% (−38.2 pp) | 0/152 | 30 |
| PatchTST | window (chained ×10) | 1947 / 5427 m | 2864 / 7145 m | 3433 / 8993 m | 4769 m | 40.8% (−22.4 pp) | 0/152 | 2 |

**Displacement error at matched lead times** — the axis on which the two horizon modes can
be compared fairly. Two accountings, deliberately both (B3.3): they answer different
questions and their numbers must not be mixed across tables.

*Record accounting* — one forecast per flight (earliest anchor), truncated at the
threshold, 3D displacement at the same `t`, mean over the flights whose record reaches
that lead. Cross-mode comparable (chained cells included), but `n` falls with lead because
records end at the threshold — 600 s survives for at most one flight and cannot be quoted:

| model | mode | 10 s | 30 s | 60 s | 120 s | 300 s |
|---|---|---:|---:|---:|---:|---:|
| iTransformer | window (chained) | **284 m** | **365 m** | **656 m** | **1247 m** | 4382 m |
| iTransformer | full | 1084 m | 934 m | 980 m | 1358 m | **3803 m** |
| PatchTST | window (chained) | 547 m | 729 m | 1289 m | 2177 m | 6849 m |
| PatchTST | full | 603 m | 624 m | 854 m | 1594 m | 4037 m |
| *n (of 152)* | | *152* | *149–152* | *149–152* | *106–146* | *33–79* |

*Raw-tensor accounting* — `history.json` `metrics.test.by_horizon`: every test window
(21k–25k windows, all anchors), truth past the threshold included, mean per lead. This is
the accounting that can see 600 s (n = 893 windows); window-mode models only reach their
own 60 s horizon here (chaining is a forecast-time construction, not a tensor):

| model | mode | 10 s | 30 s | 60 s | 120 s | 300 s | 600 s |
|---|---|---:|---:|---:|---:|---:|---:|
| iTransformer | window | **300 m** | **395 m** | **752 m** | — | — | — |
| iTransformer | full | 615 m | 628 m | 922 m | **1785 m** | **4122 m** | **5438 m** |
| PatchTST | window | 503 m | 654 m | 1183 m | — | — | — |
| PatchTST | full | 478 m | 549 m | 937 m | 2074 m | 4316 m | 7384 m |
| *n (windows)* | | *21k–25k* | *21k–23k* | *21069* | *16543* | *7380* | *893* |

### What the numbers say

**Whole approach → full, and this is the robust result.** One-pass full mode beats chained
window on lateral error at the threshold for both architectures, in every training
generation — this one: 1.5× on the mean for iTransformer (1302/868) and 1.7× for PatchTST
(3433/2016), 1.6–2.4× on p95. Once a chained pass goes wrong the next nine extrapolate
from a wrong history. Training directly for the long horizon beats chaining a short one.

**"The compounding cost lands in the tail" is architecture-dependent, not a law.** The
claim was withdrawn after the split change, and the B3 generation shows why: iTransformer
again fits it (mean ratio 1.5× vs p95 ratio 2.4×) while PatchTST does not (1.7× vs 1.6×).
Chained tails blow up for the model that was otherwise placing the endpoint well; do not
quote the tail story without naming the architecture.

**Short lead → PatchTST; long lead → iTransformer — now stated on the restored raw-tensor
column (B3.3).** Within full mode, PatchTST leads at 10–30 s (478/549 vs 615/628 m raw;
603 vs 1084 m record) and iTransformer leads at 600 s: **5438 vs 7384 m** over n = 893
windows. The 600 s direction has held in all three training generations (6135/7142,
5407/6962, 5438/7384) at margins of 1.16–1.36× — consistent, but each individual margin
sits under the provisional band, which is why the per-generation history is listed. The
mechanism is architectural: PatchTST is channel-**independent** (`TSTiEncoder` — every
channel forecast in isolation by shared weights) while iTransformer's attention runs
*across* variates; for a turning aircraft east and north are strongly coupled and PatchTST
cannot represent that by construction. Near-straight flight costs it nothing, so **the
coupling only starts paying once the turn develops.**

**Gate passes: an exact count is noise; the conclusion is not.** Across the three
generations the four cells produced 0–4 passes per 152 (0–2.6%), and this generation is
0/152 everywhere — the pre-B3 "4/152 full, 1/152 chained" did not survive a retrain whose
data moved by ≤ 0.3%. The stable statement: the 106.75 m lateral limit is FAA containment
for a *planned or flown* approach, while this is a *forecast* extrapolating 5–10 minutes
from 120 s of history; whether a borderline flight lands inside it is jitter.

**Flyability and accuracy disagree, deliberately.** iTransformer window is the *most*
flyable run (73.7%, ten points above the observed tracks' 63.2% floor) while losing to
full mode on threshold lateral error; PatchTST full is the least flyable (25.0%) at a
comparable ADE. Chaining short windows produces smooth, conservative paths — easy to fly,
not where the aircraft went. Neither metric substitutes for the other.

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
flyability: 46.1% of predictions fully flyable vs 63.2% of the observed tracks (-17.1 pp)
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

> Dating note: this ablation was measured under the pre-B3.1 `ve/vn/vu` channels and was
> not re-run after the transport-consistency change — the change rescales the velocity
> channels by ≤ 0.3%, which is far below the 1.2–2.7× margins here, and the argument for
> "off" is structural (absolute position is the signal), untouched by the channel rescale.

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

Training writes:

```
<training-dir>/
  cross_validation/
    cv_results.json          outer-train-only fold scores + split audit digests
    best_config.json         selected TSConfig overrides
  checkpoint.pt             weights, config, normalizer, split, and arrival-data provenance
  checkpoint_metadata.json  checkpoint/manifest/split hashes and anchor policy for audit/reuse
  history.json              training history, deterministic train+validation metrics, provenance
  fit_evaluation.json       best-checkpoint fixed-anchor train/validation replay + fit gaps
```

`fit_evaluation.json` is checkpoint-SHA-bound. Its native-grid ADE/FDE and TTA blocks compare
train with validation under identical inference conditions. For recursive `window`, these
native metrics describe one short prediction pass; whole-trajectory compounding remains a
separate forecast/export metric.

Prediction writes one split-specific batch:

```
<output-dir>/
  <flight_key>_states.json                     canonical predicted + observed state arrays
  <flight_key>_eval.json                       evaluation metadata + predicted states_ref
  references/<flight_key>_reference_eval.json  observed metadata + states_ref into states file
  summary.json                                 the manifest — load_records reads ONLY this
```

`summary.json`, its result rows, and every predicted source record explicitly name the
selected split. The filename stem IS the flight identity (`flight_key`), shared with the optimizer's record
filenames and the comparison-CZML group key, so learned and optimized records for one flight
always share a stem. Summary rows carry the full identity too (`id`, `icao24`, `runway`,
`landing_time_utc`).

State-output records are **reference-shaped** (`controls == []`). Control-output records use
the optimizer shape: the anchor and every segment endpoint have a 1:1 aligned active control.
The two evaluation JSONs do not copy their state arrays: `states_ref` selects the appropriate
key (and observed anchor slice) from the single states file. Records are built by
`4dTrajectory/optimization/evaluation_export.py` rather than hand-rolled here, so there is one
definition of the record shape. (That module is casadi-free, which is why it imports into the
torch env.)

`final_time_s` is predicted explicitly. The N sample timestamps are reconstructed from it,
so the last predicted state's `t` equals `final_time_s`; N and `dt` never determine duration.

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

The tests pin contracts, not quality: the heading convention in both directions, the
transport-consistency of the channels (the factor closed form against pinned WGS84 values,
and that a state sequence generated by the geodetic kinematics integrates its velocity
channels back into its position channels exactly), normalized-progress interpolation,
final-time loss, non-uniform control-duration invariants, the by-flight split, and — the important one — that an exported record satisfies the real
`evaluation.records.record_from_dict` validator and that a written batch is loadable by the
real manifest-only `load_records`. `aerodynamic_model/tests/test_torch_dynamics.py` separately
pins the differentiable rollout against `CasadiSimulator`, including its stalled branch and
non-uniform segment endpoints, and checks gradients to both controls and durations.

## Deliberate scope — NOT bugs, do not "fix" without deciding to

These are what make the default state-output path a baseline. Each is a defensible starting
point that later work may choose to extend, but none is an accident:

- **State output has no dynamics / aerodynamics.** It remains purely kinematic. Select the
  separate `control` output architecture when dynamics-generated states are the experiment.
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
