# 4-Airport ADS-B Training Data Collection Plan

> **Archived / pre-harvest:** retained as planning history. Current collection and TS
> inputs are documented in `trajectory_data_process/README.md`.

Status: development plan, not yet fully implemented.

Date: 2026-05-22.

Scope: prepare training ADS-B data for 4 airports. Minimum target per airport:

- landing trajectories: at least 1,000 usable records
- departure trajectories: at least 500 usable records

Recommended production target should include buffer for quarantine and train/val/test
split:

- landing trajectories: 1,200 training-ready records per airport
- departure trajectories: 600 training-ready records per airport
- total for 4 airports: 4,800 landing + 2,400 departure = 7,200 usable records

Use `ADS-B` in code/docs. `asd-b` in notes should be treated as a typo.

## Decision

Use OpenSky historical database access through `traffic` / Trino as the primary
source. Do not use OpenSky REST `/tracks/all` as the production data source for
this training set.

Recommended initial production airport set:

```text
KRDU, KSJC, KSMF, KSTL
```

Rationale:

- Keep the first production set U.S.-only so FAA CIFP/DOF/chart data, OpenSky
  airport metadata, and procedure validation are all in the same regulatory/data
  environment.
- `KRDU` and `KSJC` are already in the current AeroViz airport catalog.
- Replace the previous Canadian candidates with `KSMF` and `KSTL`. Both are
  U.S. medium/large commercial airports with passenger volume in the same broad
  band as `KRDU`/`KSJC`, so they should provide enough operations without the
  extreme traffic density of the largest hub airports.
- `KSMF` and `KSTL` may need AeroViz airport/procedure assets generated before
  visualization, but this document changes the data-collection plan only. Do
  not delete existing Canadian data.

Scale check used for this recommendation:

| Airport | Reason for inclusion |
| --- | --- |
| `KRDU` | Existing target; airport publishes passenger, cargo, flight, and aircraft operation reports. |
| `KSJC` | Existing target; airport publishes monthly activity reports. |
| `KSMF` | Sacramento reported 13,912,718 passengers in 2025, close to the current medium/large target band. |
| `KSTL` | St. Louis Lambert states it serves about 15.6 million passengers annually, also close to the target band. |

The first production run should start with 60 days per airport, inspect realized
usable counts, then expand to 90/180/365 days only where needed.

## External Source Facts

OpenSky REST API:

- REST now uses OAuth2 client credentials for authenticated programmatic access.
- Airport arrivals/departures are inferred in nightly batch jobs, so current-day
  operation metadata is not available as final historical flight data.
- `/tracks` is experimental, sparse by design, and exposes only selected
  waypoints, not all state-vector samples.
- `/tracks` cannot access tracks older than 30 days.
- `/flights/*` and `/tracks/all` are rate-limited.

Implication: REST is useful for smoke tests, recent debugging, and visualization,
but not for building a multi-airport training corpus.

OpenSky historical DB / Trino:

- OpenSky provides historical database access for university-affiliated
  researchers, government organizations, and aviation authorities.
- `traffic.data.opensky.history` can query by time, bounds, departure airport,
  arrival airport, or airport, and can return pandas/Traffic objects.
- The `traffic` wrapper splits requests and caches intermediary results, but it
  still issues Trino queries underneath.
- Trino usage must follow partition filters and small chunking. OpenSky warns
  that repeated failure to follow performance rules can suspend the account.
- Limit concurrency: OpenSky docs state two concurrent and two queued queries
  per user as a guideline.

Implication: use daily/weekly chunks and airport-scoped queries, never global
full-day scans.

ADSB.lol:

- ADSB.lol historical data is ODbL 1.0.
- Daily global history is published as GitHub release assets.
- The released data contains one JSON GZIP file per aircraft per day.

Implication: ADSB.lol is a viable backup or cross-check source if OpenSky DB
access is blocked, but the current repo only downloads/extracts daily releases.
It does not yet parse per-aircraft files, derive airport operation labels, or
join runway/airport metadata. It should not be the first production path.

References:

- OpenSky REST API: https://openskynetwork.github.io/opensky-api/rest.html
- OpenSky Trino / historical data: https://openskynetwork.github.io/opensky-api/trino.html
- OpenSky data access overview: https://opensky-network.org/data/
- OpenSky FAQ: https://opensky-network.org/about/faq
- `traffic` OpenSky history DB guide: https://traffic-viz.github.io/data_sources/opensky_db.html
- ADSB.lol historical data: https://www.adsb.lol/docs/open-data/historical/
- RDU statistics: https://www.rdu.com/airport-authority/statistics/
- SJC airport activity: https://www.flysanjose.com/airport-activity
- SMF 2025 passenger record: https://flysmf.gov/articles/sacramento-international-airport-sets-new-records-in-2025-accelerates-growth-and-expansion
- STL 2025 fact sheet: https://www.flystl.com/newsroom/2025-fact-sheet/

## Existing Subpackage Map

`trajectory_data_process/` currently has the right high-level split:

- `acquisition/fetch_cylw_opensky.py`
  - CLI entry point for CZML and training fetches.
  - REST client and historical training orchestration.
  - Has both `--training-source history-db` and `--training-source rest`.
- `acquisition/opensky_history_db.py`
  - Adapter around `traffic.data.opensky.history`.
  - Requests both `baroaltitude` and `geoaltitude`.
  - Writes returned history rows without sorting/renaming/deduping.
- `acquisition/download_adsblol_history.py`
  - Downloads one ADSB.lol daily split-tar release.
  - Can stream-extract release assets.
  - Does not yet parse/filter per-aircraft history.
- `processing/trajectory_events.py`
  - Builds immutable analysis waypoints from raw tracks.
  - Extracts complete airport-radius entry/exit events.
  - Labels `landing`, `depart`, `pass`, `unknown`, `ambiguous`.
- `processing/history_training.py`
  - Converts history DB rows into track-like records.
  - Preserves dual altitude points.
  - Segments by aircraft, time gap, and metadata changes.
- `processing/altitude_matching.py`
  - REST fallback helper to join `/states/all` `geo_altitude` onto
    `/tracks/all` points.
- `datasets/dataset_store.py`
  - Partitioned JSONL/source-response storage.
- `datasets/training_dataset.py`
  - Raw-track record assembly and training event/quarantine assembly.

The current architecture direction is sound. The production gap is not the
folder split; it is event semantics, quota control, and batch orchestration.

## Problems in the Current Scheme

### 1. REST `/tracks/all` cannot meet production requirements

The old CZML path and REST training fallback depend on OpenSky `/tracks/all`.
That endpoint is sparse and only available for recent tracks. A 4-airport
training set needs historical depth, repeatable collection windows, and many
state-vector samples. REST should remain smoke-test-only.

### 2. Existing CZML normalization is not training data preparation

`processing/trajectory_normalization.py` trims/filters tracks for AeroViz
visualization. It keeps an approach window before landing or closest approach
and optionally applies display-oriented altitude bias.

For model training, raw evidence and derived training episodes must stay
separate. Do not train directly from `*_czml_input_*.json`.

### 3. Current 5 NM complete entry-exit event model undercounts landings and departures

`extract_complete_airport_events()` currently emits an event only when a track
crosses from outside to inside a radius and later inside to outside the same
radius.

This works for pass-through aircraft. It is a bad primary event rule for the
requested labels:

- A landing trajectory normally enters the airport radius and ends on/near the
  airport. It may not exit the 5 NM radius in the available training window.
- A departure trajectory normally starts on/near the airport and exits the
  radius. It may not have a prior outside-to-inside entry.

Result: the current extractor can produce many `pass` events while missing the
exact `landing` and `depart` records needed for the 1,000/500 target.

### 4. `max_tracks` is not a training quota

The CLI default `--max-tracks 80` is a safety limit, not a production quota. It
counts track-like segments, not usable landing/departure events. A production
collector needs label-aware quotas:

```text
per airport:
  landing_ready >= 1200
  departure_ready >= 600
```

Counting must happen after quality gates and quarantine, not at candidate fetch
time.

### 5. `airport_ops` row fetch lacks an operation manifest

The current history DB path fetches rows with `airport=<ICAO>` and derives
tracks from rows. That is useful, but production collection also needs an
operation-level manifest:

- which OpenSky inferred flight operation was requested
- whether it was arrival or departure
- requested window
- raw rows returned
- derived episode emitted or quarantined
- label and quality reason

Without this manifest, it is hard to prove why a target count was or was not
met.

### 6. `terminal_all` can weaken label evidence

`terminal_all` concatenates:

- airport operation rows with flight metadata
- terminal-area bounded rows without flight metadata

This is acceptable for pass-through discovery, but it should not be the primary
way to count landing/departure data. For landing/departure quotas, airport
operation metadata should drive candidate selection first; terminal-area rows
should only enrich context or find pass-through samples.

### 7. There is no multi-airport resumable batch planner

The requested dataset requires 4 airports and minimum per-label counts. Running
manual single-window commands is error-prone. The package needs a batch planner
that:

- reads airport list and target quotas
- splits calendar windows into small chunks
- records per-airport progress
- resumes from the last incomplete chunk
- stops only when usable label counts are met

## Recommended Data Model

Keep the current immutable/raw principle, but add operation-level production
records.

Storage layers:

```text
trajectory_data_process/outputs/
  history_rows/v2/
    airport=<ICAO>/year=YYYY/month=MM/day=DD/hour=HH/*.jsonl

  operations/v1/
    airport=<ICAO>/year=YYYY/month=MM/day=DD/*.jsonl

  raw_tracks/v2/
    airport=<ICAO>/year=YYYY/month=MM/day=DD/hour=HH/tracks.jsonl

  airport_events/v2/
    airport=<ICAO>/year=YYYY/month=MM/day=DD/hour=HH/events.jsonl

  quarantine/v2/
    airport=<ICAO>/year=YYYY/month=MM/day=DD/hour=HH/incomplete_events.jsonl

  manifests/v2/
    airport=<ICAO>/year=YYYY/month=MM/day=DD/fetch_manifest.json

  quality_reports/v1/
    airport=<ICAO>/run=<RUN_ID>/summary.json
```

Operation manifest record:

```json
{
  "schema_version": "opensky-airport-operation-v1",
  "operation_id": "KRDU_arrival_a1b2c3_1776257000_1776259400",
  "airport": "KRDU",
  "label_target": "landing",
  "icao24": "a1b2c3",
  "callsign": "AAL123",
  "estDepartureAirport": "KCLT",
  "estArrivalAirport": "KRDU",
  "firstSeen": 1776257000,
  "lastSeen": 1776259400,
  "source": "opensky-history-db",
  "requested_window": {
    "start": "2026-04-19T09:30:00Z",
    "stop": "2026-04-19T10:20:00Z"
  },
  "derived_event_id": null,
  "status": "pending"
}
```

Training event records should continue to include `baro_altitude_m` and
`geo_altitude_m` as separate fields. Missing geometric altitude must be
quarantined for the default training dataset. Do not fill missing
`geo_altitude_m` with barometric altitude, zero, airport elevation, or a
pressure-corrected estimate.

## Correct Event Semantics for Landing and Departure

The extractor should support three operation-oriented event types.

### Landing event

Anchor:

- source metadata indicates arrival at target airport, or label target is
  `landing`
- aircraft has low-altitude or on-ground evidence near the airport

Event shape:

```text
terminal_entry -> terminal_end
```

Recommended default terminal radii:

- `event_radius_nm = 30` for model training horizon
- also derive `core_radius_nm = 5` for runway/airport evidence

`terminal_entry`:

- first outside-to-inside crossing of the training radius before the terminal
  endpoint
- if missing but enough inside points exist, quarantine as
  `missing_landing_entry`

`terminal_end`:

- first reliable on-ground point inside the core radius, or
- lowest AGL point inside the core radius when on-ground is unavailable, or
- last low-altitude point inside the core radius

Do not require a radius exit for landing.

### Departure event

Anchor:

- source metadata indicates departure from target airport, or label target is
  `departure`
- aircraft has on-ground or low-altitude evidence near the airport before
  climbing out

Event shape:

```text
terminal_start -> terminal_exit
```

`terminal_start`:

- last reliable on-ground point inside the core radius before climb-out, or
- first low-altitude airborne point inside the core radius when on-ground is
  unavailable

`terminal_exit`:

- first inside-to-outside crossing of the training radius after the terminal
  start

Do not require a prior radius entry for departure.

### Pass-through event

Anchor:

- no target-airport arrival/departure evidence
- no low-altitude/on-ground airport evidence

Event shape:

```text
terminal_entry -> terminal_exit
```

Pass-through still requires complete entry and exit crossings.

## Best Collection Workflow

### Phase 0: Access and smoke test

Goal: prove the runtime can access OpenSky history DB and write outputs.

Use current command only for connectivity and small data-shape inspection:

```bash
/Users/liudongxu/opt/miniconda3/envs/aviation/bin/python \
  trajectory_data_process/acquisition/fetch_cylw_opensky.py \
  --mode historical \
  --dataset-mode training \
  --training-source history-db \
  --fetch-profile airport_ops \
  --airport KRDU \
  --begin 2026-04-19T00:00:00Z \
  --end 2026-04-20T00:00:00Z \
  --max-tracks 200
```

Expected validation:

- `history_rows/v2/.../airport_ops.jsonl` exists
- returned rows contain `baroaltitude` and `geoaltitude`
- rows contain or can be joined to estimated departure/arrival airport fields
- manifest reports nonzero history rows

Do not interpret current `airport_events` counts as production label counts
until landing/departure event semantics are fixed.

### Phase 1: Operation list first

Add a collector/planner that first enumerates airport operations, not state rows.

Preferred logical steps:

1. For each airport and date chunk, query arrival operation list.
2. For each airport and date chunk, query departure operation list.
3. Store one `operations/v1` record per candidate operation.
4. Build target windows around each operation:
   - landing: from `lastSeen - 60 min` to `lastSeen + 10 min`
   - departure: from `firstSeen - 10 min` to `firstSeen + 60 min`
5. Fetch history state-vector rows for each window in chunks.
6. Derive operation-oriented landing/departure events.
7. Apply quality gates and update operation status.

The exact traffic API can use `flightlist()` where available for enumeration,
and `history()` for state-vector rows. If `flightlist()` is unavailable in the
runtime, derive an operation list from history DB flight metadata but still
write explicit `operations/v1` records before event extraction.

### Phase 2: Label-aware quota loop

Implement a resumable command similar to:

```bash
/Users/liudongxu/opt/miniconda3/envs/aviation/bin/python \
  -m trajectory_data_process.acquisition.collect_training_set \
  --airports KRDU KSJC KSMF KSTL \
  --begin 2026-01-01T00:00:00Z \
  --end 2026-03-01T00:00:00Z \
  --target-landing 1200 \
  --target-departure 600 \
  --event-radius-nm 30 \
  --core-radius-nm 5 \
  --chunk-days 7 \
  --max-concurrent-queries 2
```

Stop condition per airport:

```text
quality.accepted.landing >= target_landing
quality.accepted.departure >= target_departure
```

Non-stop conditions:

- candidate fetched
- raw track/row written
- ambiguous event written
- quarantined record written

These are useful evidence, but they do not satisfy the requested training
dataset target.

### Phase 3: Terminal context enrichment

After landing/departure quotas are healthy, run `terminal_all`-style bounded
state-vector queries to collect:

- pass-through aircraft
- nearby traffic context
- multi-aircraft interaction context

These records should not be allowed to dilute or relabel the operation-driven
landing/departure corpus.

### Phase 4: Optional ADSB.lol fallback

Use ADSB.lol only if OpenSky DB access is unavailable or if a second source is
needed for comparison.

Missing implementation before ADSB.lol can be used for this target:

- parse extracted per-aircraft JSON GZIP files
- build airport bounding-box/radius filters
- infer landing/departure without OpenSky airport metadata
- map fields into the same raw-track and event schema
- handle ODbL attribution/license requirements in dataset manifests

## Quality Gates

Apply gates after event extraction. Quarantine failures with exact reasons.

Required for accepted landing/departure records:

- label is exactly `landing` or `departure`
- operation metadata agrees with label when available
- `geo_altitude_m` exists for every selected training point
- `baro_altitude_m` exists for every selected training point
- timestamps are strictly increasing
- minimum selected point count:
  - landing/departure: at least 30 points for a 30 NM horizon, or tune after
    inspecting actual sampling cadence
  - 5 NM core-only debug events can be smaller but should not be the main
    training horizon
- maximum internal time gap:
  - initial gate: 120 seconds
  - stricter model-ready gate may use 60 seconds
- landing has terminal endpoint inside core radius and below
  `airport_elev_m + 600 m`, or on-ground evidence
- departure has terminal start inside core radius and below
  `airport_elev_m + 600 m`, or on-ground evidence
- event duration is plausible:
  - landing 30 NM horizon: roughly 5-45 minutes
  - departure 30 NM horizon: roughly 5-45 minutes
- no impossible coordinate jump or speed outlier after resampling diagnostics

Recommended quarantine reasons:

```text
missing_geo_altitude
missing_baro_altitude
missing_operation_metadata
missing_landing_entry
missing_landing_terminal_anchor
missing_departure_start_anchor
missing_departure_exit
too_few_points
max_gap_too_large
ambiguous_label
duplicate_operation
implausible_speed
outside_airport_context
```

## Output Metrics Required Before Training

Every production run should produce one quality summary per airport:

```json
{
  "airport": "KRDU",
  "date_range": {
    "begin": "2026-01-01T00:00:00Z",
    "end": "2026-03-01T00:00:00Z"
  },
  "targets": {
    "landing": 1200,
    "departure": 600
  },
  "accepted": {
    "landing": 1248,
    "departure": 634
  },
  "quarantine": {
    "missing_geo_altitude": 31,
    "too_few_points": 44,
    "ambiguous_label": 12
  },
  "raw": {
    "history_rows": 1234567,
    "operations": 2201
  }
}
```

Training should begin only after all four airports meet accepted target counts
and label balance has been reviewed.

## Implementation Tasks

1. Add operation-oriented event extraction.
   - Do not replace current pass-through entry/exit extraction.
   - Add landing and departure extractors that do not require both crossings.
   - Add tests for landing-without-exit and departure-without-entry.

2. Add operation manifest storage.
   - New `operations/v1` writer.
   - Stable operation ID.
   - Status updates or append-only status records.

3. Add label-aware quota accounting.
   - Count accepted landing/departure events after quality gates.
   - Produce per-airport `quality_reports/v1`.

4. Add a multi-airport batch planner.
   - Inputs: airports, date range, targets, chunk size.
   - Resume from manifests.
   - Enforce at most two concurrent OpenSky history DB queries.

5. Add production CLI.
   - Either new module `acquisition/collect_training_set.py`, or a separate
     planner wrapper that calls existing fetch functions.
   - Prefer a new module to keep `fetch_cylw_opensky.py` from becoming larger.

6. Keep REST fallback as debug-only.
   - Document this in CLI help.
   - Do not let REST fallback silently satisfy production targets.

7. Decide whether to add ADSB.lol parser.
   - Only needed if OpenSky DB access is blocked or if external validation is
     required.

## Immediate Next Step

Before any large download, implement and test the event extractor fix:

- landing event from outside radius to terminal low/on-ground anchor
- departure event from terminal low/on-ground anchor to outside radius
- pass event remains complete entry-exit

Then run a 1-day smoke fetch for `KRDU` and manually inspect:

- raw history rows
- operation metadata availability
- accepted vs quarantined landing/departure event counts
- representative trajectories in AeroViz or a quick notebook plot

Only after that should the 4-airport 60-day collection start.
