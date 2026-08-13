# trajectory_data_process

This package has one trajectory-acquisition pipeline: `trajectory_data_process.harvest`.
It downloads OpenSky historical state vectors, reconstructs contiguous flights, assigns
each landing to at most one runway, and publishes explicit manifests for audit and modeling.

## Canonical data flow

```text
OpenSky history DB
  → harvest.tracks        reconstruct one contiguous track per flight (HAE)
  → harvest.classify      runway assignment + one final-segment fit
                          assigned | ambiguous | unassignable | not_landing
                          + policy-free observed threshold event when estimated
  → tracks/manifest.json  authoritative roster of every harvested outcome
       ├─ observed event evaluation + frontend CZML
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

Rebuild downstream views from unchanged assignment/event data without downloading:

```bash
conda run -n aeroviz python -m trajectory_data_process.harvest \
  --airport KRDU --evaluate-only
```

`--evaluate-only` requires
`trajectory_data_process/outputs/harvest/KRDU/tracks/manifest.json`. That file is created
by a successful harvest or local reclassification; it is intentionally not reconstructed
by globbing old track files. This mode does not refit trajectories. It rejects a legacy
or stale threshold event whose runway-data fingerprint differs from the active runway
configuration or CIFP cycle.

Re-run runway assignment and final-segment fitting from the stored HAE samples, then
rebuild arrivals, observed evaluation, CZML, and publication—still without downloading:

```bash
conda run -n aeroviz python -m trajectory_data_process.harvest \
  --airport KRDU --reclassify-existing
```

Use `--reclassify-existing` after changing the runway/CIFP data cycle or the assignment
or fitting implementation. It validates every rostered source record, writes the complete
new classification into a staging directory, and replaces `tracks/` only after every
record succeeds. The manifest records `network_access: false`, the source-manifest hash,
and the new per-runway fingerprints.

Merge a second stored harvest root into the canonical root, then reclassify and rebuild
all derived views without downloading:

```bash
conda run -n aeroviz python -m trajectory_data_process.harvest \
  --airport KRDU \
  --merge-source trajectory_data_process/outputs/harvest-may-2026
```

`--merge-source` accepts a harvest root containing `<ICAO>/tracks/manifest.json`, not an
individual airport directory, and may be repeated. The destination is the airport under
`--output` (the canonical harvest root by default). Before changing it, the merge validates
every rostered file and manifest count, rejects unsafe paths, and stages a
same-filesystem hard-link tree. It then performs the complete reclassification in that
staging tree and rejects collisions under the newly derived `flight_key`. Only after
parsing, classification, uniqueness checks, and serialization all succeed does it
atomically replace canonical `tracks/` and invalidate the derived `arrivals/` and
`approach/` views. A failure leaves all three canonical trees unchanged. The manifest
records source hashes and acquisition provenance. This mode never queries OpenSky.

Download ADS-B history again only when the stored source samples themselves must be
replaced:

```bash
conda run -n aeroviz python -m trajectory_data_process.harvest \
  --airport KRDU --full-redownload
```

The four modes are mutually exclusive:

- `--evaluate-only`: reuse assignment and threshold events; rebuild downstream views.
- `--reclassify-existing`: reuse samples; rebuild assignment, events, and downstream views.
- `--merge-source ROOT`: merge stored samples, then rebuild assignment, events, and views.
- `--full-redownload`: acquire the source samples again and rebuild everything.

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

Every roster row carries `event_status`. Each assigned track with a valid winning fit
stores one `observed_threshold_event`: the estimated threshold crossing, uncertainty,
source sample range, fit diagnostics, and extrapolation distance. It contains physical
fit results only—not an LPV/LNAV-VNAV benchmark, limit, or verdict. Its runway-data
fingerprint binds it to the exact threshold position, course, vertical datum, width,
procedure facts, and FAA runway/CIFP cycles used by assignment.

Observed event availability is calculated from the source manifest before filtering:
assigned, ambiguous, and unassignable tracks form the arrival-candidate denominator;
known `not_landing` tracks are reported separately as excluded.

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
harvest/classify.py     assignment, one final-segment fit, landing anchor, threshold event
harvest/store.py        complete measured buckets + tracks manifest
harvest/merge.py        transactional manifest merge + source provenance, no download
harvest/reclassify.py   no-download reassignment/refitting from stored HAE samples
harvest/arrivals.py     model-ready crop/filter + arrivals manifest
harvest/observed.py     datum conversion + evaluation records from the stored event
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
- Stored threshold events are cycle-bound derived data. If the active FAA runway or CIFP
  facts change, `--evaluate-only` refuses to mix cycles; run `--reclassify-existing`.
- OpenSky historical DB access is a separate account entitlement. Credentials are read by
  `pyopensky` (`OPENSKY_USERNAME` / `OPENSKY_PASSWORD` or its settings file).
- The historical DB lags real time. Use a fixed `--start` sufficiently in the past when
  reproducibility/cache reuse matters.
- `--no-cache` bypasses the history query cache. Interrupt handlers cancel the in-flight
  query so a stopped harvest does not keep a Trino quota slot occupied.

## Safe derived-data cleanup

Always preview a single-airport cleanup before deleting anything:

```bash
conda run -n aeroviz python clean_pipeline_data.py --airport KRDU --dry-run
conda run -n aeroviz python clean_pipeline_data.py --airport KRDU
```

Use repeated `--airport ICAO` arguments for several airports. `--all-airports` is an
explicit broader scope; omitting the scope is an error. The cleaner uses a producer-owned
allowlist and removes only regenerable arrival/approach views, scenario files, canonical
optimizer outputs, confirmed validation predictions, and eligible frontend publications.

It never selects downloaded `tracks/`, checkpoint/history or `test_release.json`, formal
experiments, final-test predictions, parked/manual/unknown outputs, static airport data,
git-tracked files, or archives. A comparison tree containing an experiment or final-test
publication is preserved as a unit. Missing or malformed prediction/comparison metadata
also fails closed. On an approved clean, targets are validated and staged on the same
filesystem before deletion; a staging failure restores every file already moved.

## Tests

```bash
conda run -n aeroviz pytest -o "pythonpath=. geokit/src" \
  trajectory_data_process/harvest/tests trajectory_data_process/tests -q
```

The design documents under `docs/01`–`docs/08` describe earlier experiments as well as
current domain rationale; this README is authoritative for the executable download and
dataset architecture.
