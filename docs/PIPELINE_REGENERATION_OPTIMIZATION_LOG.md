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

The repository has no manifest-aware harvest merge command. The merge must become a
supported pipeline operation rather than an ad-hoc copy: validate every manifest record,
reject identity/path collisions, preserve both acquisition provenances, use same-filesystem
hard links or another low-space transactional commit, invalidate derived views, and only
then reclassify against the current runway/CIFP frame.
