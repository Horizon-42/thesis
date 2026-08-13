# Pipeline Regeneration and Optimization Log

This log records measured regeneration costs and concrete optimization opportunities.
It is evidence from real pipeline runs, not a replacement for the canonical commands in
the component READMEs. Timings use `/usr/bin/time` wall time and maximum resident memory.

## 2026-08-12 — New final-approach evaluation regeneration

Scope: retained ADS-B source tracks for KMSY, KRDU, KSJC, KSMF, and KSTL. No OpenSky
download and no outer-test ML release were requested or performed.

### Commands and intentional reuse

The stored threshold events had to be regenerated against the current runway/CIFP frame:

```bash
conda run -n aeroviz python -m trajectory_data_process.harvest \
  --airport <ICAO> --reclassify-existing
```

After that single observed rebuild per airport, scenario preparation reused the new
arrival manifest instead of invoking the default observed step again:

```bash
conda run -n aeroviz python prepare_scenario_inputs.py \
  --airport <ICAO> --skip-observed
```

The optimizer uses the canonical three-mode runner:

```bash
conda run -n aeroviz python run_scenario_optimization.py \
  --airport <ICAO> --jobs <N>
```

### Measured completed stages

| Airport | Stored tracks | Assigned | Model-ready | Reclassify wall | Peak RSS | Prepare wall | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| KMSY | 8,903 | 1,870 | 1,870 | 52.50 s | 0.61 GiB | 8.34 s | 0.67 GiB |
| KRDU | 28,981 | 7,265 | 6,552 | 227.79 s | 2.67 GiB | 18.86 s | 1.55 GiB |
| KSJC | 47,373 | 4,919 | 4,916 | 188.30 s | 1.29 GiB | 11.68 s | 0.93 GiB |
| KSMF | 11,036 | 1,942 | 1,942 | 63.84 s | 0.62 GiB | 8.67 s | 0.68 GiB |
| KSTL | 24,768 | 4,462 | 4,462 | 151.64 s | 1.41 GiB | 12.13 s | 0.98 GiB |

The five preparations produced 19,742 fitted-threshold scenarios and 19,742 runway-
threshold scenarios.

### Confirmed optimization opportunities

1. **Avoid a second observed rebuild after explicit reclassification.** The current
   `prepare_scenario_inputs.py` default calls harvest `--evaluate-only`. The regeneration
   therefore must use `--skip-observed` after `--reclassify-existing`, or it repeats
   arrival/report/CZML work. A future runner mode could express “reclassify, then prepare”
   as one supported operation.
2. **Incremental reclassification.** `--reclassify-existing` rereads and classifies every
   stored track, including large non-landing populations. KSJC processed 47,373 tracks to
   publish 4,916 model-ready arrivals; KRDU used 2.67 GiB peak RSS. Store enough
   dependency fingerprints to refit only tracks whose samples or runway/CIFP frame
   changed, while preserving a full-rebuild verification mode.
3. **Share arrival materialization between target builders.** Fitted-ADS-B and runway
   scenario creation independently materialize the same arrival samples into two large
   JSON documents. Preparation reached 1.55 GiB peak RSS for KRDU. A canonical shared
   scenario base plus small target overlays, or a single pass that writes both targets,
   would reduce parsing, memory, and duplicated storage without changing optimizer input
   semantics.
4. **Expose structured timing.** The high-level runners print human progress but do not
   persist stage timing, input counts, peak memory, or cache/reuse decisions. Add a
   machine-readable regeneration manifest so performance regressions can be compared
   without wrapper commands.

### In-progress measurements

The five airport optimizer sweeps were started with bounded aggregate concurrency (KMSY
at six workers; the other airports at four workers each), then intentionally interrupted
when the additional `trajectory_data_process/outputs/harvest-may-2026` source harvest was
identified. Partial optimizer/reference output and the now-stale preparation/observed
views were removed through the allow-listed cleaner: 40,125 files, 4.2 GiB. No source
track, experiment, test ledger, or mixed experiment comparison publication was removed.

### Capacity finding before the merged regeneration

The Linux filesystem had 3.8 GiB free before partial-output cleanup and 8.0 GiB after it.
The canonical and May source harvests occupy 2.1 GiB and 4.5 GiB respectively after the
canonical derived views were cleaned. The five acquisition windows do not overlap by
`flight_key`; May ends on 2026-06-01 and the later windows begin in late June or July.

Eight GiB is not sufficient for a safe merge/reclassification plus the combined observed,
scenario, optimizer, and development-prediction publications. A conservative prerequisite
is at least 25 GiB free on a Linux-writable filesystem, retaining roughly 10 GiB final
headroom. The unmounted 930 GiB NTFS device was inspected read-only and rejected as
overflow storage: it is the Windows system volume, 95% used, and contains a current
hibernation file. It was not mounted read-write or modified.

This gap was addressed before the merged regeneration described below. The supported
`--merge-source ROOT` mode validates every manifest record, preserves acquisition
provenance, stages same-filesystem hard links, reclassifies against the current
runway/CIFP frame, and rejects collisions under the resulting current `flight_key` before
atomically replacing canonical tracks and invalidating derived views.

## 2026-08-12 — Merged harvest and data-driven development rerun

Scope: merge the canonical and `harvest-may-2026` stored ADS-B harvests for five US
airports, rebuild observed evaluation against the current FAA runway/CIFP cycle, and rerun
the pooled iTransformer and PatchTST development pipelines. No OpenSky query, scenario
optimization, hyperparameter search, or outer-test release was performed.

### Supported merge command

Each airport used the new pipeline mode rather than an ad-hoc copy:

```bash
conda run -n aeroviz python -m trajectory_data_process.harvest \
  --airport <ICAO> \
  --merge-source trajectory_data_process/outputs/harvest-may-2026
```

Preflight found no duplicate `flight_key` or destination-path collision. Every source
manifest and rostered track was validated and reclassified in the staged tree before the
canonical destination changed. Reclassification rewrote current derived records, so the
temporary hard links did not leave old runway assignments or threshold events in the
canonical harvest. Both acquisition provenances and manifest hashes remain in the merged
manifest; `network_access` is false.

### Merged observed results

| Airport | Tracks | Assigned | Model-ready arrivals | Merge/reclassify wall | Peak RSS | Observed verdicts (fail / indeterminate) |
|---|---:|---:|---:|---:|---:|---:|
| KMSY | 19,786 | 4,299 | 4,299 | 103.23 s | 1.26 GiB | 11 / 4,288 |
| KRDU | 67,528 | 16,530 | 14,961 | 535.90 s | 5.99 GiB | 51 / 14,916 |
| KSJC | 106,468 | 10,945 | 10,938 | 448.81 s | 2.81 GiB | 31 / 10,914 |
| KSMF | 25,378 | 4,683 | 4,411 | 146.69 s | 1.32 GiB | 4 / 4,408 |
| KSTL | 50,492 | 9,328 | 9,328 | 311.96 s | 2.91 GiB | 19 / 9,309 |
| **Total** | **269,652** | **45,785** | **43,937** | **1,546.59 s** | — | **116 / 43,835** |

Observed event availability uses the complete arrival-candidate denominator, including
ambiguous and unassignable tracks. Model-ready exclusions remain explicit: KRDU excluded
1,563 arrivals with no published TCH and 6 local circuits; KSMF excluded 271 with no
published TCH and 1 local circuit; KSJC excluded 7 local circuits. The observed-verdict
column includes those 14 local circuits, while the model-ready column correctly excludes
them, which explains the different totals.

### Development-model cohort and commands

The canonical runner was used once per architecture:

```bash
conda run -n aeroviz python run_ts_pipeline.py \
  --training-mode pooled --models <itransformer|patchtst> --split development
```

The merged manifests contain 43,937 source arrivals. The development loader selected
37,361 train/validation identities and built 37,352 series: 30,659 train and 6,693
validation. Nine tracks were shorter than the 120-second history window. The 6,576
outer-test identities stayed closed: their tracks were not loaded, no test prediction was
created, and no `test_release.json` was written or modified.

Both models used the existing base recipe because cross-validation was intentionally
skipped and no reusable CV artifact matched the new manifest digests. This is a faithful
regeneration, not a new tuning experiment.

| Model | Output grid | Best epoch / epochs run | Best val objective | Median epoch | Pipeline wall | Peak process RSS |
|---|---:|---:|---:|---:|---:|---:|
| iTransformer | 64 | 150 / 170 | 0.225841 | 5.00 s | 1,246.60 s initial + 484.26 s resume | 11.93 GiB initial |
| PatchTST | 256 | 105 / 125 | 0.377014 | 9.60 s | 2,083.29 s | 13.77 GiB |

The iTransformer initial command trained successfully but stopped during KRDU publication
when a stationary observed tail produced mathematically undefined heading and turn-rate
diagnostics. The summary writer had emitted Python `NaN`. The persistence boundary now
maps only unavailable optional diagnostics to JSON `null`, uses `allow_nan=False`, and
continues to reject non-finite required states and metrics. The supported `--skip-train`
resume path verified all five manifest digests, reused the completed checkpoint, and
republished train/validation outputs. The 484.26-second resume time is therefore separate
from the initial command rather than additional training time.

### Validation results

These are development validation results and may guide future work. They are not final-test
claims.

| Model | Airport | Flights | ADE | FDE | Verdicts (pass / fail / indeterminate) |
|---|---|---:|---:|---:|---:|
| iTransformer | KMSY | 663 | 2,003.7 m | 2,691.8 m | 0 / 645 / 18 |
| iTransformer | KRDU | 2,254 | 1,828.9 m | 2,106.7 m | 0 / 2,220 / 34 |
| iTransformer | KSJC | 1,684 | 974.4 m | 1,406.3 m | 0 / 1,558 / 126 |
| iTransformer | KSMF | 677 | 1,857.8 m | 2,807.6 m | 0 / 674 / 3 |
| iTransformer | KSTL | 1,415 | 1,522.5 m | 2,085.7 m | 0 / 1,365 / 50 |
| PatchTST | KMSY | 663 | 4,389.1 m | 4,470.3 m | 0 / 663 / 0 |
| PatchTST | KRDU | 2,254 | 4,337.3 m | 3,443.1 m | 0 / 2,252 / 2 |
| PatchTST | KSJC | 1,684 | 2,489.4 m | 2,421.5 m | 0 / 1,684 / 0 |
| PatchTST | KSMF | 677 | 4,987.3 m | 5,746.3 m | 0 / 677 / 0 |
| PatchTST | KSTL | 1,415 | 4,286.6 m | 3,558.5 m | 0 / 1,411 / 4 |

iTransformer is consistently better than the base PatchTST recipe on every airport.
PatchTST validation flyability is 0.0–0.4% and its raw derivative diagnostics are very
rough, so this checkpoint should be treated as a documented negative development result,
not a release candidate. Zero approach passes are model outcomes under the new threshold-
event verdict, not publication failures.

### Final integrity and capacity

- All 20 model/split summary and evaluation-report pairs load as strict JSON and have
  matching split names and trajectory counts.
- Focused tests: 15 harvest merge/reclassification tests and 4 TS strict-export/evaluation
  tests passed; changed Python modules also passed `py_compile`.
- The earlier full harvest/data-process run passed 164 tests and exposed 2 unrelated
  pre-existing CV recipe-reuse failures (`control_horizon_curriculum_s` mismatch); those
  interfaces were not changed in this task.
- Final sizes: `trajectory_data_process/outputs` 13 GiB, `4dTrajectory/outputs` 12 GiB,
  and frontend airport data 15 GiB. The filesystem retained 43 GiB free.

### New optimization backlog from measured evidence

1. **Cache development series by manifest and build recipe.** Each architecture spent
   roughly six minutes decoding the same 43,937 manifest-rostered JSON tracks and reached
   about 8 GiB loader RSS before training. A digest-keyed, strict, regenerable cache could
   remove this repeated CPU/RAM stage without changing the split or model recipe.
2. **Add digest-safe per-airport publication resume.** A failure at KRDU forced the pooled
   runner to republish KMSY and rerun KRDU inference even though the checkpoint was valid.
   Completion markers must bind checkpoint digest, arrival-manifest digest, split,
   evaluation schema, and publication command before a stage can be skipped safely.
3. **Reduce dense prediction duplication.** The PatchTST KRDU train record directory alone
   is about 1.0 GiB because every flight stores a 256-point predicted state payload, then
   the frontend publishes another representation. Keep one canonical strict state payload
   and derive bounded visualization views without duplicating full-resolution arrays.
4. **Stream or batch reclassification inputs.** KRDU reclassification peaked near 6 GiB
   and KSJC processed 106,468 tracks to retain 10,938 model-ready arrivals. The current
   transactional safety is correct, but record processing can be streamed into the staged
   tree while retaining manifest validation and an atomic commit.
5. **Persist runner stage timings automatically.** This log still required `/usr/bin/time`
   wrappers and manual aggregation. The high-level runners should publish wall time, peak
   RSS, input/output counts, cache decisions, checkpoint digest, and completion status in
   a machine-readable regeneration manifest.
6. **Audit the native training raw-kinematics aggregate.** iTransformer printed an
   astronomically large position/velocity RMSE during final fit evaluation while its
   per-airport published summaries remained finite and plausible. This diagnostic should
   not be used for model selection until the duration/outlier contribution is traced and
   reported robustly.
