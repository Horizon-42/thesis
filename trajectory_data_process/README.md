# trajectory_data_process

This package has one trajectory-acquisition pipeline: `trajectory_data_process.harvest`.
It downloads OpenSky historical state vectors, reconstructs contiguous flights, assigns
each landing to at most one runway, and publishes explicit manifests for audit and modeling.

## Canonical data flow

```text
OpenSky history DB
  → harvest.tracks        freshness gate + held-state collapse
                          position clock = lastposupdate; altitude remains HAE
                          final continuous position-update block only
  → harvest.classify      runway assignment + threshold-event estimation
                          assigned | ambiguous | unassignable | not_landing
                          + bracket-anchored robust 3D threshold event
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
It also requires the `harvest-tracks-v2-source-timing` contract; a pre-fix manifest is
rejected instead of silently rebuilding arrivals from held OpenSky state vectors.

The canonical five-airport dataset has already completed the source-timing migration
(see `docs/10-source-timed-canonical-promotion-audit.md`). To migrate another legacy
harvest safely with the backfilled ADS-B timing sidecars, use distinct legacy and
staging roots:

```bash
conda run -n aeroviz python -m trajectory_data_process.harvest \
  --airport KRDU \
  --rebuild-fresh-from /path/to/legacy-harvest \
  --output /path/to/new-source-timed-staging
```

This is the recommended migration mode. The source and destination airport roots must
be different and non-nested, and the destination airport must not already exist. The
mode resolves sidecar metadata in batches (default 512 tracks), writes to a temporary
directory on the destination filesystem, and verifies the source manifest plus every
rostered file's size/mtime fingerprint before and after processing. It then builds
arrivals and observed evaluation inside the new output root. Frontend CZML/publication
are deliberately skipped until the staging result has been reviewed.

Before writing airport data, the mode requires free space for three times the rostered
source-track bytes plus a 2 GiB reserve. This deliberately conservative preflight covers
the additional reported-speed array, evaluation records, and temporary serialization;
it is repeated independently for every airport, so a multi-airport rebuild stops before
starting the next airport if accumulated outputs have consumed the reserve.

Horizontal samples are timestamped by `lastposupdate`. A sample is retained only when
both `state time - lastcontact <= 15 s` and
`state time - lastposupdate <= 15 s`. Repeated snapshots with the same position time
collapse to the snapshot nearest that update. If `geoaltitude` changes while the
horizontal position time is held, the later height is not treated as a synchronized 3D
point; the group is counted in `source_integrity.geoaltitude_async_groups`. Position
gaps over 15 seconds are never bridged: only the final continuous block remains.

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

The five modes are mutually exclusive:

- `--evaluate-only`: reuse assignment and threshold events; rebuild downstream views.
- `--reclassify-existing`: reuse samples; rebuild assignment, events, and downstream views.
- `--merge-source ROOT`: merge stored samples, then rebuild assignment, events, and views.
- `--rebuild-fresh-from ROOT`: read old tracks and timing sidecars into a new safe staging root.
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

The tracks manifest uses `harvest-tracks-v2-source-timing` and records whether source
integrity is complete. Every track record carries its own cleanup counts and a parallel
`reported_ground_speeds_m_s` array; the four-column geometry sample contract remains
unchanged, and `start_time_utc` retains millisecond precision. Every roster
row also carries `event_status`. Each assigned track with a valid winning fit stores one
version-6 `observed_threshold_event`. A source-valid threshold bracket selects the runway
and physical inbound pass after its displacement over `lastposupdate` time agrees with
the two ADS-B reported ground-speed values; it does not independently supply the final
lateral or vertical estimate. Both components come from the same robust 3D final-segment
fit, using the preferred 3 km window or audited 4 km/5 km availability candidates.
Rejected brackets remain audited. The event records source ranges, window sensitivity,
uncertainty, method diagnostics, and extrapolation distance. It contains physical
estimator results only—not an LPV/LNAV-VNAV benchmark, limit, or verdict. Its runway-data
fingerprint binds it to the exact threshold position, course, vertical datum, width,
procedure facts, and FAA runway/CIFP cycles used by assignment.

The uncertainty fields describe estimator/data quality; evaluation publishes
them as diagnostics but does not use them to shrink a component bound or turn
an available point estimate into `indeterminate`.

The estimator design, metadata experiments, position-jump audit, uncertainty
construction, and version-4
contract are documented in
[`final_approach/FIT_MODEL_OPTIMIZATION.md`](../final_approach/FIT_MODEL_OPTIMIZATION.md).

Observed event availability is calculated from the source manifest before filtering:
assigned, ambiguous, and unassignable tracks form the arrival-candidate denominator;
known `not_landing` tracks are reported separately as excluded. Arrival candidates that
cannot yield a two-point fresh final block remain unavailable in this denominator via
the staging manifest's source-integrity exclusion roster.

`arrivals/manifest.json` is narrower by design. It contains only supervised/modeling-ready
arrivals: assigned tracks with a published CIFP Path Point (TCH and glidepath), cropped at
the final entry into the terminal ring and stopped at `landing_sample_index`. Exclusions
such as `local_circuit`, `no_published_tch`, and `no_published_glidepath` remain counted in
the manifest. The v4 source-timed manifest stores only the source track file and slice indices; consumers
materialize the crop from that canonical track instead of keeping a second JSON copy. Each
view row also pins the source file's SHA-256 so a changed track cannot be combined silently
with stale slice metadata.

## Responsibilities

```text
harvest/runner.py       backward time-window scan and stopping policy
harvest/tracks.py       row → contiguous measured Track
harvest/freshness_rebuild.py batched, source-preserving staging migration
harvest/classify.py     assignment, landing anchor, threshold-event orchestration
harvest/threshold_event.py structural bracket selection + one robust 3D event
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

## Altitude outliers

A handful of state vectors report an altitude the aircraft cannot have been at — measured
extremes reach 20 147 m between neighbours at 724 m, which renders as a needle-shaped
vertical peak and drags any fit through it. `harvest/altitude_filter.py` replaces those
altitudes **as a stored track is read into a derived view**, and only there: the observed
CZML, the model-ready arrival slices, and the evaluation records. `tracks/` keeps the
broadcast value, so the arrival roster's SHA-256 pins, `--reclassify-existing`, and
`source_integrity` all still describe the bytes the receiver produced.

The criterion is a deviation from the median of the ±2-sample window that exceeds both
100 m and what `25 m/s` could fly in the tighter adjacent gap. Over the five harvested
airports that is 561 samples in 451 of 44 622 assigned tracks (0.0027 %). Only
`samples[i][3]` changes — count, times and horizontal positions are untouched, because
`landing_sample_index`, the arrival slice bounds and the threshold event's
`source_sample_range` all index this array.

```bash
# what would be replaced, per airport (read-only)
conda run -n aeroviz python -m trajectory_data_process.altitude_outliers --airport KRDU
# republish public/data/<ICAO>/trajectories.czml through the filter
conda run -n aeroviz python -m trajectory_data_process.altitude_outliers --airport KRDU \
  --rerender-czml
```

Stored `observed_threshold_event`s were fitted during assignment, before this filter; the
audit reports outliers landing inside an event's source range so those cases can be
judged, and `--reclassify-existing` is what re-derives them.

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
