# trajectory_data_process

This package has one trajectory-acquisition pipeline: `trajectory_data_process.harvest`.
It downloads OpenSky historical state vectors, reconstructs contiguous flights, assigns
each landing to at most one runway, and publishes explicit manifests for audit and modeling.

## Canonical data flow

```text
OpenSky history DB
  → harvest.tracks        reconstruct one contiguous track per flight (HAE)
  → harvest.classify      assigned | ambiguous | unassignable | not_landing
  → tracks/manifest.json  authoritative roster of every harvested outcome
       ├─ observed fit/evaluation + frontend CZML
       └─ arrivals/manifest.json
            assigned + published CIFP TCH/glidepath
            + final 25 km entry → measured landing anchor
            - local circuits
                 ├─ flight_scenarios / optimizer reference
                 └─ ts_transformer training and prediction
```

Consumers never discover trajectories with a JSON glob. A manifest is the data contract,
so stale/orphan files from an earlier run cannot silently enter counts, scenarios, or a
train/validation/test split.

The cross-pipeline canonical-file and migration contract is documented in
[`docs/data-pipeline-canonical-storage.zh.md`](../docs/data-pipeline-canonical-storage.zh.md).

## Commands

Single airport:

```bash
conda run -n aeroviz python -m trajectory_data_process.harvest \
  --airport KRDU --count 200
```

Multiple airports (`download_landings.py` is now only a thin
multi-airport wrapper around the same harvest implementation):

```bash
conda run -n aeroviz python trajectory_data_process/download_landings.py \
  --airports KRDU KSJC --count 200

# omit --airports to process every airport in runway_thresholds.json
conda run -n aeroviz python trajectory_data_process/download_landings.py --count 200
```

Rebuild all derived outputs from an existing track harvest without downloading:

```bash
conda run -n aeroviz python -m trajectory_data_process.harvest \
  --airport KRDU --evaluate-only
```

`--evaluate-only` requires
`trajectory_data_process/outputs/harvest/KRDU/tracks/manifest.json`. That file is created
only by a successful full harvest; it is intentionally not reconstructed by globbing old
track files.

Procedure assets are a separate static-data pipeline:

```bash
./generate_aeroviz_airport_procedure_data.sh KRDU
```

## On-disk layout

```text
outputs/harvest/<ICAO>/
  tracks/
    manifest.json
    assigned/<RUNWAY>/<flight_key>.json
    ambiguous/<flight_key>.json
    unassignable/<flight_key>.json
    not_landing/<flight_key>.json
  arrivals/
    manifest.json
  approach/
    records/<flight_key>_eval.json
    summary.json
    evaluation_report.json
```

`tracks/manifest.json` represents the complete harvest and retains all four categories:

- `assigned`: one runway wins the global all-threshold assignment.
- `ambiguous`: more than one runway remains plausible.
- `unassignable`: coverage/geometry is insufficient for a reliable runway.
- `not_landing`: the track does not satisfy the landing screen.

`arrivals/manifest.json` is narrower by design. It contains only supervised/modeling-ready
arrivals: assigned tracks with a published CIFP Path Point (TCH and glidepath), cropped at
the final entry into the terminal ring and stopped at `landing_sample_index`. Exclusions
such as `local_circuit`, `no_published_tch`, and `no_published_glidepath` remain counted in
the manifest. The v3 manifest stores only the source track file and slice indices; consumers
materialize the crop from that canonical track instead of keeping a second JSON copy. Each
view row also pins the source file's SHA-256 so a changed track cannot be combined silently
with stale slice metadata.

## Responsibilities

```text
harvest/runner.py       backward time-window scan and stopping policy
harvest/tracks.py       row → contiguous measured Track
harvest/classify.py     one-track/one-runway assignment and landing anchor
harvest/store.py        complete measured buckets + tracks manifest
harvest/arrivals.py     model-ready crop/filter + arrivals manifest
harvest/observed.py     inferred final-approach fit and evaluation records
harvest/czml.py         one canonical observed CZML + runway selector metadata
download_landings.py    airport-list expansion only; no acquisition logic
arrival_segment.py      final terminal-entry crop and local-circuit rule
```

There is no second `Trajectory` model, event-extraction downloader, partitioned JSONL
training store, or per-runway landing-file pipeline. Those implementations were removed;
`download_trajectories.py` is no longer part of the project.

## Downstream inputs

Scenario/optimization:

```bash
conda run -n aeroviz python -m flight_scenarios \
  --input trajectory_data_process/outputs/harvest/KRDU/arrivals/manifest.json \
  --target-from-threshold
```

TS training:

```bash
conda run -n aeroviz python 4dTrajectory/ts_transformer/__main__.py train \
  --data trajectory_data_process/outputs/harvest/KRDU/arrivals/manifest.json \
  --airport KRDU --model itransformer --horizon-mode window \
  --output-dir 4dTrajectory/outputs/KRDU/ts_itransformer_window
```

Both consumers therefore see the same flights, the same final-entry boundary, the same
runway assignment, and the same published runway target.

## Vertical datum

OpenSky `geoaltitude` is height above the WGS84 ellipsoid (HAE). Harvest records keep HAE
unchanged because Cesium expects ellipsoidal height. Modeling targets and regulatory gates
are MSL, so HAE → MSL conversion occurs exactly once at `flight_scenarios.datum` (and in
the TS dataset builder before it reads bare waypoints). Moving the conversion into harvest
would make the visualization wrong by the same geoid offset it fixes for modeling.

## Reference data and access

- `config/runway_thresholds.json` supplies airport centers and every displaced landing
  threshold. Regenerate it with `build_runway_config.py`.
- FAA CIFP Path Point records supply landing-threshold position, published TCH, and
  glidepath. A runway without that record stays in the complete harvest but is excluded
  from model-ready arrivals.
- OpenSky historical DB access is a separate account entitlement. Credentials are read by
  `pyopensky` (`OPENSKY_USERNAME` / `OPENSKY_PASSWORD` or its settings file).
- The historical DB lags real time. Use a fixed `--start` sufficiently in the past when
  reproducibility/cache reuse matters.
- `--no-cache` bypasses the history query cache. Interrupt handlers cancel the in-flight
  query so a stopped harvest does not keep a Trino quota slot occupied.

## Tests

```bash
conda run -n aeroviz pytest -o "pythonpath=. geokit/src" \
  trajectory_data_process/harvest/tests trajectory_data_process/tests -q
```

The design documents under `docs/01`–`docs/08` describe earlier experiments as well as
current domain rationale; this README is authoritative for the executable download and
dataset architecture.
