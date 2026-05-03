# OpenSky Training Data Fetch and Store Redesign

## Status
Implementation in progress.

Implemented core v2 pieces:

- partitioned source-response/raw-track/event/quarantine/manifest storage
- derived 5 nm entry/exit airport event extraction with interpolated boundary metadata
- explicit `landing`, `depart`, `pass`, `unknown`, and `ambiguous` labels
- REST fallback with source response text preservation, cache reuse, request budgets, and rate-limit stop behavior
- preferred OpenSky history DB ingestion through `traffic.data.opensky.history`
- history DB raw row storage under `history_rows/v2`
- derived training points with both `baro_altitude_m` and `geo_altitude_m`

Still required before large production collection:

- install and configure `traffic`/OpenSky DB access in the runtime environment
- run a real KRDU integration fetch against the history DB
- inspect label balance and quarantined records before expanding the date range

## Objective
Redesign `opensky_data_query` from an airport-centered visualization downloader into a data collection pipeline suitable for training 4D trajectory prediction models.

The redesigned pipeline must:

- collect more data than the current airport-arrival-focused workflow
- store data by airport and by time partition
- preserve the original fetched response text exactly for audit and comparison
- never modify fetched raw data in place
- preserve complete trajectory evidence for aircraft near the airport
- include aircraft that only pass through the airport area
- avoid hard clipping tracks at arbitrary sample points
- extract complete airport-area episodes from the 5 nm airport boundary entry to the 5 nm boundary exit as derived records
- keep the existing CZML visualization export as a derived output, not as the primary training dataset

## Training Data Volume Targets

The planning unit should be usable airport episodes, not downloaded raw tracks. A downloaded raw track may produce zero, one, or multiple airport-area episodes depending on whether it crosses the airport radius and whether the episode is complete.

Recommended single-airport targets:

| Use case | Usable complete episodes |
| --- | ---: |
| Pipeline validation, visual inspection, label rule debugging | 100-500 |
| Initial baseline model, such as interpolation, Kalman, simple LSTM/GRU | 1,000-5,000 |
| Practical single-airport model experiments | 10,000-50,000 |
| Better coverage across runway direction, weather, day/night, traffic mix, and season | 50,000-200,000 |
| Larger Transformer/generative trajectory models | 100,000+ |

Recommended first production target per airport:

```text
total complete episodes: 20,000+
landing: 8,000-12,000
depart: 8,000-12,000
pass: 2,000-5,000
unknown/ambiguous: store, but exclude from default training split
```

For lower-traffic airports, a smaller first milestone is acceptable:

```text
minimum usable dataset: 3,000-5,000 complete episodes
formal experiment dataset: 20,000+ complete episodes
stronger generalization dataset: 50,000+ complete episodes
```

Collection window guidance:

- start with 30-90 days of historical data per airport
- inspect the realized episode count and label balance
- expand to 6-12 months if the airport has low traffic or if seasonal/weather coverage is needed

The 5 nm event radius should be treated as a core airport-area label, not necessarily the only training horizon. For 4D trajectory prediction, preserve the full raw track and consider derived event sets at multiple radii:

- 5 nm core airport event
- 10 nm / 20 nm terminal event
- 50 nm extended terminal event
- full available OpenSky track

## Current Implementation Summary
The package has two flows:

1. Existing CZML visualization flow.
2. New training-dataset flow.

The main implemented modules are:

- `fetch_cylw_opensky.py`
- `trajectory_normalization.py`
- `dataset_store.py`
- `trajectory_events.py`
- `training_dataset.py`
- `altitude_matching.py`
- `opensky_history_db.py`
- `history_training.py`

Current CZML behavior:

- Live mode uses `/flights/all` and primarily filters flights whose estimated arrival airport matches the target airport.
- Historical mode uses `/flights/arrival` only.
- Track data is fetched through `/tracks/all`.
- Raw OpenSky payloads are written as `*_raw_*.json`.
- Normalized visualization input is written as `*_czml_input_*.json`.
- CZML normalization keeps only a final approach window before touchdown or closest approach.

Current training behavior:

- default training source is `--training-source history-db`
- history DB mode uses `traffic.data.opensky.history`, not REST `/tracks/all`
- `airport_ops` fetches flights associated with the airport through OpenSky DB airport filtering
- `terminal_all` additionally fetches state-vector rows in an airport-centered bounding box to collect pass-through candidates
- raw history DB rows are written before derived processing and are not sorted, renamed, filtered, or deduplicated in storage
- derived processing normalizes a copy of the history rows for event extraction
- REST `/tracks/all` remains available only with `--training-source rest` for debugging/fallback

## Problems to Fix

### Arrival Bias
The old historical CZML mode only fetched arrivals. Training mode now supports departures and pass-through aircraft:

- history DB `airport_ops` covers airport arrivals/departures
- history DB `terminal_all` adds airport-area state-vector rows for pass-through discovery
- REST fallback supports arrival, departure, and sampled area candidates, but it is not the preferred training path because of `/tracks/all` limitations

### Visualization-Oriented Filtering
`trajectory_normalization.py` filters and trims trajectories for AeroViz display. In particular, it keeps a bounded approach window before landing or closest approach. That creates short visualization-friendly tracks, not training-ready samples.

### Hard Clipping
The current logic may keep or discard waypoints based on time windows and sampled closest points. For training, an airport-area sample should have a physically meaningful boundary:

- start when the aircraft enters the 5 nm airport radius
- end when the aircraft exits the 5 nm airport radius

However, this boundary must not be implemented by modifying the raw OpenSky waypoint list. Boundary crossing times and positions are derived annotations only. The original fetched response and parsed raw track must remain unchanged.

### Raw Data Mutability Risk
The current `*_raw_*.json` output is already parsed and re-serialized. That is useful, but it is not the exact original HTTP response text. For training data collection, the pipeline needs an intermediate source layer that stores the original response body exactly as received, so later analysis can compare:

- original OpenSky response text
- parsed raw track record
- derived airport event record
- downstream model-ready features

### Missing Transit Aircraft
Aircraft that pass through the airport area without landing or departing can be useful for model training and traffic-context learning. These should be included and labeled rather than filtered out.

## Proposed Architecture

Split the package into separate pipeline concerns:

1. Fetch OpenSky responses and save the original response text.
2. Parse raw track payloads without changing their values or waypoint order.
3. Extract airport-area training episodes as derived records.
4. Store source responses, raw tracks, extracted episodes, and manifests in partitioned dataset folders.
5. Keep CZML export as a separate derived product for visualization.

Suggested modules:

- `opensky_client.py`
  - OAuth token handling
  - REST API GET requests
  - return both parsed JSON and original response text
  - retries and rate-limit aware backoff in a later iteration

- `airport_profile.py`
  - airport coordinate and elevation resolution
  - radius and bounding-box helpers

- `trajectory_events.py`
  - OpenSky waypoint parsing
  - distance-to-airport computation
  - 5 nm boundary crossing estimation on a derived copy
  - airport episode extraction
  - relation labeling

- `dataset_store.py`
  - source response text writer
  - JSONL partition writer
  - manifest writer
  - deterministic event IDs
  - deduplication keys

- `trajectory_normalization.py`
  - keep existing CZML-oriented conversion logic
  - do not use this as the training dataset transformation

- `opensky_history_db.py`
  - adapter around `traffic.data.opensky.history`
  - request state-vector columns with both `baroaltitude` and `geoaltitude`
  - request joined flight-table airport metadata with `FlightsData4.estdepartureairport` and `FlightsData4.estarrivalairport`
  - write raw returned history rows as JSONL without sorting or renaming

- `history_training.py`
  - build track-like derived records from history DB rows
  - split aircraft history rows into track segments by time gap and metadata changes
  - attach same-row `baro_altitude_m` and `geo_altitude_m`
  - reuse the same airport-event extraction and labeling logic as REST-derived tracks

## Fetch Design

### Preferred Training Source: OpenSky History DB

For training data, the preferred source is the OpenSky history database via the `traffic` package:

```python
from traffic.data import opensky

opensky.history(...)
```

Reasons:

- avoids REST `/tracks/all` 429 behavior during training collection
- avoids REST `/tracks/all` 30-day lookback limitation
- returns state-vector rows that can include both `baroaltitude` and `geoaltitude`
- can query by airport for airport operations and by geographic bounds for pass-through candidates

Runtime prerequisite:

- `traffic` must be installed
- OpenSky DB access must be configured according to the [`traffic` OpenSky DB documentation](https://traffic-viz.github.io/data_sources/opensky_db.html)

The code should fail fast with a clear error if `traffic` is unavailable.

### Historical Mode
Historical training mode has two source implementations.

#### `--training-source history-db`

This is the default and recommended path.

`airport_ops`:

- fetch OpenSky history rows with `airport=<ICAO>`
- request state-vector columns plus joined flight metadata:
  - `baroaltitude`
  - `geoaltitude`
  - `FlightsData4.estdepartureairport`
  - `FlightsData4.estarrivalairport`
- derive `landing` and `depart` labels from metadata plus airport-area geometry

`terminal_all`:

- run the `airport_ops` query
- additionally fetch state-vector rows in an airport-centered bounding box
- use the bounded state-vector rows to discover pass-through aircraft
- store the bounded result separately as raw history rows
- label complete high-altitude entry/exit episodes as `pass` only when landing/departure evidence is absent

Important limitation:

- a bounded history query is a spatial raw data query, not a mutation of the stored track
- the query bounds should be larger than the 5 nm event radius so the derived event extractor has samples before entry and after exit

#### `--training-source rest`

REST fallback remains available for debugging and short recent windows.

It supports three candidate sources:

1. Landing candidates
   - OpenSky `/flights/arrival`
   - target airport is the estimated arrival airport

2. Depart candidates
   - OpenSky `/flights/departure`
   - target airport is the estimated departure airport

3. Pass candidates
   - query states in an airport-centered spatial area over the requested time range
   - fetch `/tracks/all` for aircraft observed in or near the airport radius
   - include aircraft even when estimated departure and arrival airports do not match the target airport

The implementation can introduce two fetch profiles:

- `airport_ops`
  - arrivals and departures only
  - lower API usage
  - useful for initial validation

- `terminal_all`
  - landing candidates, depart candidates, and pass candidates
  - higher API usage
  - preferred for training data expansion

Historical labels must be explicit:

- `landing`
- `depart`
- `pass`
- `unknown`
- `ambiguous`

The label is stored as a derived field and must never overwrite original OpenSky fields such as `estArrivalAirport`, `estDepartureAirport`, `firstSeen`, or `lastSeen`.

### Live Mode
Live mode should remain available for quick inspection, but it should not be treated as the main training ingestion path.

Recommended behavior:

- query current states in an airport-centered bbox
- fetch full `/tracks/all` for nearby aircraft
- classify and store complete episodes when possible
- mark incomplete episodes clearly because live data may not yet contain the future exit from the 5 nm radius

### Track Fetching
Do not use one track-fetching rule for every source.

History DB source:

- do not call REST `/tracks/all`
- write returned DB rows under `history_rows/v2`
- build a derived track-like record from the DB rows for compatibility with the existing event extractor
- preserve both `baroaltitude` and `geoaltitude` from the same state-vector row

REST source:

- candidate flights or aircraft are converted into `/tracks/all` requests when the timestamp is within the REST endpoint lookback window
- the original `/tracks/all` response body is saved before JSON parsing
- `/tracks/all` is not used to obtain `geo_altitude` because the endpoint does not contain it

Deduplication key:

```text
(icao24, startTime, endTime)
```

When `startTime` or `endTime` is missing, fall back to:

```text
(icao24, first_waypoint_time, last_waypoint_time)
```

The deduplication process must not delete the original source response text. If two candidate sources produce the same track, keep one parsed raw track record but keep source references that show every candidate source that found it.

## Source Response Text Layer

Every OpenSky HTTP response used by the REST pipeline should be saved before any parsing, filtering, sorting, or normalization.

This layer is the audit source of truth.

Recommended layout:

```text
opensky_data_query/outputs/
  source_responses/v2/
    airport=CYLW/year=2026/month=04/day=15/hour=13/
      20260415T130001Z_flights_arrival_CYLW_1776211200_1776214800.body.txt
      20260415T130002Z_tracks_all_c07b0a_1776258422.body.txt
      source_index.jsonl
```

`*.body.txt` should contain the response body text exactly as returned by OpenSky. It should not be pretty-printed, sorted, filtered, or re-serialized.

`source_index.jsonl` should store request and integrity metadata:

```json
{
  "schema_version": "opensky-source-response-v2",
  "source_id": "sha256:...",
  "fetched_at_utc": "2026-04-15T13:00:02Z",
  "endpoint": "/tracks/all",
  "params": {
    "icao24": "c07b0a",
    "time": 1776258422
  },
  "http_status": 200,
  "body_path": "20260415T130002Z_tracks_all_c07b0a_1776258422.body.txt",
  "body_sha256": "...",
  "body_bytes": 12345
}
```

Implementation rule:

- first write the response body text and source index
- then parse JSON from that stored text
- never derive training records directly from an unsaved response

For OpenSky history DB mode, there is no REST response body text exposed by `traffic`. The equivalent audit layer is:

```text
opensky_data_query/outputs/
  history_rows/v2/
    airport=KRDU/year=2026/month=04/day=19/hour=10/
      airport_ops.jsonl
      terminal_area.jsonl
```

History row storage rule:

- write the dataframe returned by `traffic` before derived analysis
- do not sort rows in this stored layer
- do not rename columns in this stored layer
- do not filter rows in this stored layer
- do not deduplicate rows in this stored layer
- only perform JSON-safe serialization, for example timestamp to ISO text

## Airport Event Extraction

### Radius
Use a default airport event radius of 5 nautical miles:

```text
5 nm = 9.260 km
```

CLI option:

```text
--airport-event-radius-nm 5.0
```

### Input Waypoint Format
OpenSky `/tracks/all` path entries are expected as:

```text
[time, latitude, longitude, baro_altitude, true_track, on_ground]
```

The altitude in `/tracks/all` is `baro_altitude`, not GPS/WAAS/geometric altitude. The raw field should be named and documented as barometric altitude in meters.

The internal representation should preserve:

- absolute UTC time
- latitude
- longitude
- barometric altitude in meters
- true track in degrees
- on-ground flag
- distance to airport in nautical miles

Important: this internal representation is a derived analysis view. It must not replace or mutate the parsed raw `track.path`.

### Altitude Source Semantics

OpenSky exposes altitude differently depending on endpoint:

- `/tracks/all`
  - waypoint field index 3 is `baro_altitude`
  - this is barometric altitude in meters
  - track waypoints do not include `geo_altitude`

- `/states/all`
  - state vector field index 7 is `baro_altitude`
  - state vector field index 13 is `geo_altitude`
  - `geo_altitude` is geometric altitude in meters when available

- OpenSky history DB through `traffic.data.opensky.history`
  - state-vector column `baroaltitude` is barometric altitude in meters
  - state-vector column `geoaltitude` is geometric altitude in meters when available
  - these two values should be preserved as separate fields in derived training points

Design implication:

- raw track storage must preserve OpenSky track altitude exactly as `baro_altitude_m`
- do not label `/tracks/all` altitude as GPS altitude, WAAS altitude, geometric altitude, or height above ground
- training trajectory/event storage must include both `baro_altitude_m` and `geo_altitude_m`
- because `/tracks/all` does not contain `geo_altitude`, `geo_altitude_m` must be joined from an additional source such as `/states/all` state vectors
- in history DB mode, no REST altitude join is needed because both altitude fields can come from the same history row
- any corrected altitude, QNH-adjusted altitude, AGL estimate, or geometric-altitude join must be a derived field in a later layer
- if future collection uses `/states/all` snapshots for pass candidates, store `baro_altitude_m` and `geo_altitude_m` separately and preserve nullability

Recommended raw field names:

```text
track.path[*][3]                  # original OpenSky baro_altitude
raw_altitude_source = "opensky_tracks_all_baro_altitude_m"
baro_altitude_m                   # derived parsed field name, same value as raw path index 3
geo_altitude_m                    # only from state vectors when available, not from tracks/all
altitude_correction_mode          # derived layer only
```

### Required Dual-Altitude Training Records

For model training records, every stored trajectory point should have:

```text
baro_altitude_m
geo_altitude_m
```

This is a requirement for the derived training dataset, not a mutation of the OpenSky raw track. The pipeline should satisfy it by joining geometric altitude from state-vector data.

Recommended history DB process:

1. Fetch history DB state-vector rows with both `baroaltitude` and `geoaltitude`.
2. Save the raw returned rows under `history_rows/v2`.
3. Build a derived analysis dataframe copy.
4. Segment rows by aircraft, time gap, and flight metadata changes.
5. Build a track-like derived record for event extraction.
6. Build training points with:
   - `baro_altitude_m` from `baroaltitude`
   - `geo_altitude_m` from `geoaltitude`
7. Store a derived training event only when required altitude fields are available.

REST fallback process:

1. Fetch `/tracks/all` for the complete aircraft track.
2. Save the original `/tracks/all` response text.
3. Parse the raw track without modifying `path`.
4. Fetch state-vector samples for the same `icao24` and time range when available.
5. Save the original state-vector response text.
6. Time-match state-vector `geo_altitude` onto derived trajectory points.
7. Store a derived training point only when both altitude fields are available.

Matching policy:

- exact timestamp match is preferred
- otherwise use nearest state-vector sample within a configurable tolerance
- recommended initial tolerance: 15 seconds
- do not interpolate `geo_altitude_m` in the first implementation unless explicitly enabled
- if multiple samples are equally close, choose the one with valid `geo_altitude`

CLI options:

```text
--require-geo-altitude
--geo-altitude-source states-all-time-match
--geo-altitude-max-age-sec 15
--allow-geo-altitude-interpolation
```

Default training behavior:

- `--require-geo-altitude` should be enabled for training dataset generation
- records missing either `baro_altitude_m` or `geo_altitude_m` should be written to quarantine
- missing geometric altitude must not be filled with barometric altitude
- missing geometric altitude must not be silently set to `0`, airport elevation, or a corrected barometric estimate

History DB derived point schema:

```json
{
  "time": 1776257000,
  "lat": 49.98,
  "lon": -119.31,
  "baro_altitude_m": 1500.0,
  "geo_altitude_m": 1468.2,
  "altitude_sources": {
    "baro_altitude_m": "opensky_history_db_baroaltitude_m",
    "geo_altitude_m": "opensky_history_db_geoaltitude_m"
  },
  "geo_altitude_match": {
    "method": "same_history_row",
    "source_time": 1776257000,
    "delta_t_sec": 0,
    "source_response_id": "opensky_history_db"
  }
}
```

REST fallback derived point schema:

```json
{
  "time": 1776257000,
  "lat": 49.98,
  "lon": -119.31,
  "baro_altitude_m": 1500.0,
  "geo_altitude_m": 1468.2,
  "altitude_sources": {
    "baro_altitude_m": "opensky_tracks_all_baro_altitude_m",
    "geo_altitude_m": "opensky_states_all_geo_altitude_m"
  },
  "geo_altitude_match": {
    "method": "nearest_state_vector",
    "source_time": 1776257004,
    "delta_t_sec": 4,
    "source_response_id": "sha256:..."
  }
}
```

Quality counters should include:

```text
points_total
points_with_baro_altitude
points_with_geo_altitude
points_with_both_altitudes
points_quarantined_missing_geo_altitude
```

### Crossing Detection
For each full track:

1. Keep the raw OpenSky `path` unchanged in raw storage.
2. Build a derived analysis list from valid waypoints.
3. Sort only the derived analysis list by time if needed for event detection.
4. Compute distance from each derived waypoint to airport center.
5. Identify outside-to-inside transitions as 5 nm entry crossings.
6. Identify inside-to-outside transitions as 5 nm exit crossings.
7. Build one airport episode for each complete entry-exit pair.

### Boundary Crossing Estimate
If a crossing occurs between two sampled waypoints, estimate the crossing time and position as derived metadata.

For consecutive points `A` and `B`, where distance crosses the radius:

```text
alpha = (radius_nm - distance_a_nm) / (distance_b_nm - distance_a_nm)
```

Estimate:

- time
- latitude
- longitude
- altitude
- true track, with heading wrap-around handling when implemented

Do not insert this estimated point into the raw waypoint sequence.

Instead, store it under a derived field:

```json
{
  "entry_crossing": {
    "kind": "estimated_boundary_crossing",
    "radius_nm": 5.0,
    "time": 1776257000.4,
    "lat": 49.98,
    "lon": -119.31,
    "source_segment": {
      "before_raw_index": 41,
      "after_raw_index": 42
    }
  }
}
```

The same structure applies to `exit_crossing`.

The airport event should store raw waypoint references separately:

```json
{
  "raw_waypoint_range": {
    "start_raw_index": 42,
    "end_raw_index": 57
  }
}
```

This avoids hard clipping while still preserving the original OpenSky data without modification.

### Completeness Policy
Default policy:

- keep only complete airport episodes that have both 5 nm entry and 5 nm exit
- discard or quarantine tracks that begin inside the radius or end inside the radius

Optional CLI:

```text
--allow-incomplete-events
```

When enabled, incomplete episodes may be stored with:

```json
{
  "complete_radius_crossing": false,
  "incomplete_reason": "track_started_inside_radius"
}
```

The recommended default for training is to exclude incomplete episodes from the main dataset and write them to a diagnostics/quarantine output.

## Event Classification

Each extracted airport episode should be labeled. Labels are metadata, not hard filters.

For historical data, the required labels are:

- `landing`
- `depart`
- `pass`
- `unknown`
- `ambiguous`

The label must be derived carefully from both OpenSky historical endpoint evidence and track geometry. A label must include its evidence, because OpenSky estimated airports and track samples can be incomplete or wrong.

### Label Evidence Inputs

Use these inputs when available:

- candidate source:
  - `/flights/arrival`
  - `/flights/departure`
  - area/state candidate
- OpenSky flight metadata:
  - `estDepartureAirport`
  - `estArrivalAirport`
  - `firstSeen`
  - `lastSeen`
- track geometry:
  - complete 5 nm entry
  - complete 5 nm exit
  - closest distance to airport
  - first and last distance relative to airport
- vertical and ground evidence:
  - `on_ground` points near airport
  - low altitude above airport elevation
  - descent before closest approach
  - climb after airport area

### Landing Label

Assign `landing` only when there is strong evidence that the aircraft landed at the target airport.

Primary evidence:

- the candidate came from `/flights/arrival` for the target airport
- or `estArrivalAirport` equals the target airport

Geometry consistency checks:

- the episode has a complete 5 nm entry
- the track reaches near-airport low altitude or on-ground state
- the closest point is near the airport center
- the trajectory is not simply high-altitude pass-through traffic

Recommended derived evidence fields:

```json
{
  "label": "landing",
  "label_evidence": {
    "historical_endpoint": "arrival",
    "estArrivalAirport_matches": true,
    "has_complete_entry": true,
    "has_complete_exit": true,
    "has_near_airport_on_ground": true,
    "min_agl_m": 32.0,
    "min_distance_nm": 0.4
  }
}
```

If the endpoint says arrival but the track never gets near the airport or remains high, label as `ambiguous`, not `landing`.

### Depart Label

Assign `depart` only when there is strong evidence that the aircraft departed from the target airport.

Primary evidence:

- the candidate came from `/flights/departure` for the target airport
- or `estDepartureAirport` equals the target airport

Geometry consistency checks:

- the episode has a complete 5 nm exit
- the track starts near the airport, starts on ground, or starts at low altitude near the airport
- altitude or distance trend is consistent with departure
- the trajectory is not just a high-altitude pass through the 5 nm circle

Recommended derived evidence fields:

```json
{
  "label": "depart",
  "label_evidence": {
    "historical_endpoint": "departure",
    "estDepartureAirport_matches": true,
    "has_complete_entry": true,
    "has_complete_exit": true,
    "has_near_airport_on_ground": true,
    "initial_agl_m": 45.0,
    "min_distance_nm": 0.5
  }
}
```

If the endpoint says departure but the track does not contain enough airport-area evidence, label as `ambiguous`.

### Pass Label

Assign `pass` when the aircraft passes through the 5 nm airport area without evidence of landing or departing at the target airport.

Required evidence:

- complete 5 nm entry and complete 5 nm exit
- candidate is not from target-airport `/flights/arrival`
- candidate is not from target-airport `/flights/departure`
- `estArrivalAirport` does not equal the target airport
- `estDepartureAirport` does not equal the target airport
- no near-airport on-ground evidence
- no convincing low-altitude landing or departure evidence

Recommended derived evidence fields:

```json
{
  "label": "pass",
  "label_evidence": {
    "historical_endpoint": "area",
    "estArrivalAirport_matches": false,
    "estDepartureAirport_matches": false,
    "has_complete_entry": true,
    "has_complete_exit": true,
    "has_near_airport_on_ground": false,
    "min_agl_m": 1800.0,
    "min_distance_nm": 2.1
  }
}
```

Do not call a sample `pass` just because there is no arrival/departure metadata. The track must have complete entry and exit through the 5 nm radius, and it must lack landing/departure evidence.

### Unknown and Ambiguous Labels

Use `unknown` when the track has a usable airport-area episode but insufficient evidence for `landing`, `depart`, or `pass`.

Use `ambiguous` when evidence conflicts, for example:

- endpoint says arrival but geometry looks like high-altitude pass
- endpoint says departure but the track starts far outside the airport area
- both arrival and departure evidence are present
- track starts inside or ends inside the 5 nm radius and `--allow-incomplete-events` is enabled

Ambiguous records should be stored, but they should not be included in the default training split.

Suggested CLI thresholds:

```text
--terminal-altitude-agl-m 3000
--low-altitude-agl-m 600
```

The first implementation can use conservative rules and store the supporting evidence so classification can be refined offline.

## Storage Layout

Training data should be stored separately from CZML output.

Recommended layout:

```text
opensky_data_query/outputs/
  history_rows/v2/
    airport=CYLW/year=2026/month=04/day=15/hour=13/*.jsonl
  source_responses/v2/
    airport=CYLW/year=2026/month=04/day=15/hour=13/
      *.body.txt
      source_index.jsonl
  raw_tracks/v2/
    airport=CYLW/year=2026/month=04/day=15/hour=13/tracks.jsonl
  airport_events/v2/
    airport=CYLW/year=2026/month=04/day=15/hour=13/events.jsonl
  manifests/v2/
    airport=CYLW/year=2026/month=04/day=15/fetch_manifest.json
  quarantine/v2/
    airport=CYLW/year=2026/month=04/day=15/hour=13/incomplete_events.jsonl
  czml/
    airport=CYLW/
```

Partitioning:

- history rows: partition by DB fetch time
- source responses: partition by response fetch time
- raw tracks: partition by track start time
- airport events: partition by event entry time
- manifests: partition by fetch date or requested begin date

Default partition granularity:

```text
hour
```

The current implementation uses fixed hour partitions for row/event outputs and day partitions for manifests. A public `--partition-granularity` option is not implemented yet.

## Raw Track JSONL Schema

One line per fetched or derived source track.

For REST source, this is parsed from the stored source response text, but it must preserve OpenSky values exactly. It must not sort, filter, interpolate, or rewrite `track.path`.

For history DB source, this is a derived track-like compatibility record built from stored history rows. The immutable source layer is `history_rows/v2`; this record exists so the same airport-event extractor can run on REST and history DB data.

```json
{
  "schema_version": "opensky-raw-track-v2",
  "airport": "CYLW",
  "fetch_profile": "terminal_all",
  "source": {
    "api": "opensky",
    "endpoint": "/tracks/all",
    "candidate_sources": ["arrival"],
    "source_response_ids": ["sha256:..."]
  },
  "flight_metadata": {
    "icao24": "c07b0a",
    "callsign": "ACA327",
    "estDepartureAirport": "CYUL",
    "estArrivalAirport": "CYYC",
    "firstSeen": 1776214861,
    "lastSeen": 1776258422
  },
  "track": {
    "icao24": "c07b0a",
    "callsign": "ACA327",
    "startTime": 1776214861,
    "endTime": 1776258422,
    "path": []
  }
}
```

The full OpenSky `path` should be preserved in raw storage exactly as parsed from the source response.

History DB raw-track source example:

```json
{
  "source": {
    "api": "opensky-history-db",
    "endpoint": "traffic.data.opensky.history",
    "candidate_sources": ["arrival"],
    "source_response_ids": ["history_rows:airport_ops.jsonl"]
  }
}
```

Do not:

- insert boundary points
- drop invalid points
- sort path entries
- convert relative times
- apply altitude correction
- deduplicate waypoints inside the raw track

## Airport Event JSONL Schema

One line per complete airport-area episode. This is a derived record. It may include computed fields, label evidence, and boundary crossing estimates, but it must keep references back to the raw source.

```json
{
  "schema_version": "opensky-airport-event-v2",
  "event_id": "CYLW_20260415T132233Z_c07b0a_1776214861_5nm",
  "airport": "CYLW",
  "radius_nm": 5.0,
  "label": "landing",
  "complete_radius_crossing": true,
  "source": {
    "raw_track_id": "sha256:...",
    "source_response_ids": ["sha256:..."]
  },
  "flight": {
    "icao24": "c07b0a",
    "callsign": "ACA327",
    "track_start_time": 1776214861,
    "track_end_time": 1776258422,
    "estDepartureAirport": "CYUL",
    "estArrivalAirport": "CYYC"
  },
  "event_time": {
    "entry_time": 1776257000,
    "exit_time": 1776257600,
    "closest_time": 1776257300
  },
  "quality": {
    "num_points": 18,
    "max_gap_s": 35,
    "min_distance_nm": 0.46,
    "complete": true,
    "points_total": 18,
    "points_with_baro_altitude": 18,
    "points_with_geo_altitude": 18,
    "points_with_both_altitudes": 18
  },
  "label_evidence": {
    "historical_endpoint": "arrival",
    "estArrivalAirport_matches": true,
    "estDepartureAirport_matches": false,
    "has_complete_entry": true,
    "has_complete_exit": true,
    "has_near_airport_on_ground": true,
    "min_agl_m": 32.0
  },
  "boundary_crossings": {
    "entry": {
      "kind": "estimated_boundary_crossing",
      "radius_nm": 5.0,
      "time": 1776257000.4,
      "source_segment": {
        "before_raw_index": 41,
        "after_raw_index": 42
      }
    },
    "exit": {
      "kind": "estimated_boundary_crossing",
      "radius_nm": 5.0,
      "time": 1776257600.8,
      "source_segment": {
        "before_raw_index": 56,
        "after_raw_index": 57
      }
    }
  },
  "raw_waypoint_range": {
    "start_raw_index": 42,
    "end_raw_index": 57
  }
}
```

Notes:

- airport events are derived records, not raw data
- raw latitude, longitude, altitude, and time values remain in the raw track
- event records refer to raw waypoint indices instead of copying and mutating the raw path
- model-ready fixed-window tensors should be generated in a later feature extraction stage
- derived local coordinates can be added later without replacing raw coordinates.

## Manifest Schema

Each fetch run should write a manifest:

```json
{
  "schema_version": "opensky-fetch-manifest-v2",
  "airport": "CYLW",
  "fetch_profile": "terminal_all",
  "begin": 1776211200,
  "end": 1776297600,
  "created_at_utc": "2026-04-15T13:00:00Z",
  "radius_nm": 5.0,
  "candidate_counts": {
    "landing_candidates": 120,
    "depart_candidates": 95,
    "pass_candidates": 44
  },
  "track_counts": {
    "requested": 259,
    "downloaded": 231,
    "deduplicated": 218
  },
  "event_counts": {
    "complete": 172,
    "incomplete": 19,
    "landing": 83,
    "depart": 71,
    "pass": 14,
    "unknown": 0,
    "ambiguous": 4
  },
  "outputs": {
    "source_responses": [],
    "raw_tracks": [],
    "airport_events": [],
    "quarantine": []
  }
}
```

## CLI Proposal

Recommended training dataset fetch:

```bash
python opensky_data_query/fetch_cylw_opensky.py \
  --mode historical \
  --dataset-mode training \
  --training-source history-db \
  --airport CYLW \
  --begin 2026-04-01T00:00:00Z \
  --end 2026-04-02T00:00:00Z \
  --fetch-profile terminal_all
```

Lower-cost airport operations only:

```bash
python opensky_data_query/fetch_cylw_opensky.py \
  --mode historical \
  --dataset-mode training \
  --training-source history-db \
  --airport CYLW \
  --begin 2026-04-01T00:00:00Z \
  --end 2026-04-02T00:00:00Z \
  --fetch-profile airport_ops
```

Small KRDU smoke command:

```bash
python opensky_data_query/fetch_cylw_opensky.py \
  --mode historical \
  --dataset-mode training \
  --training-source history-db \
  --airport KRDU \
  --begin 2026-04-19T10:00:00Z \
  --end 2026-04-19T10:15:00Z \
  --fetch-profile terminal_all \
  --max-tracks 10
```

REST fallback command for recent data only:

```bash
python opensky_data_query/fetch_cylw_opensky.py \
  --mode historical \
  --dataset-mode training \
  --training-source rest \
  --airport KRDU \
  --begin 2026-04-19T10:00:00Z \
  --end 2026-04-19T10:15:00Z \
  --fetch-profile airport_ops \
  --max-tracks 10
```

Existing visualization flow:

```bash
python opensky_data_query/fetch_cylw_opensky.py \
  --mode historical \
  --airport CYLW \
  --dataset-mode czml
```

Suggested new arguments:

```text
--dataset-mode czml|training
--training-source history-db|rest
--fetch-profile airport_ops|terminal_all
--airport-event-radius-nm 5.0
--allow-incomplete-events
--terminal-altitude-agl-m 3000
--low-altitude-agl-m 600
--require-geo-altitude
--geo-altitude-source states-all-time-match
--geo-altitude-max-age-sec 15
--allow-geo-altitude-interpolation
--write-source-responses
--write-raw-tracks
--write-events
--write-quarantine
```

Current implementation intentionally hides several low-level REST tuning arguments from help output. The training default should stay usable without manually choosing sleep and retry knobs.

## Backward Compatibility

Keep these existing output types working:

- `*_raw_*.json`
- `*_czml_input_*.json`

Do not change the existing CZML input schema consumed by `aeroviz-4d/python/generate_czml.py`.

The new training dataset should be additive and should not break:

- `run_asd-b_fetch_and_generate.py`
- `generate_czml.py`
- existing `--input-raw-json` reuse flow

## Implementation Plan

1. Add `trajectory_events.py`. Completed.
   - Parse OpenSky waypoints.
   - Compute distance to airport.
   - Detect entry and exit crossings.
   - Estimate 5 nm boundary crossings as derived metadata.
   - Emit complete airport episodes.

2. Add `dataset_store.py`. Completed.
   - Write original source response body text before parsing.
   - Write JSONL records into airport/time partitions.
   - Create deterministic event IDs.
   - Write manifests.

3. Refactor REST fetch code lightly. Completed.
   - Keep `OpenSkyClient` behavior initially.
   - Add departures to historical fetch.
   - Add candidate source metadata.
   - Preserve existing CZML flow.

4. Add `--dataset-mode training`. Completed.
   - Training mode writes raw tracks and airport events.
   - CZML mode keeps current behavior.

5. Add dual-altitude derived training points. Completed.
   - REST mode can time-match `/states/all` geo altitude when explicitly enabled.
   - history DB mode keeps same-row `baroaltitude` and `geoaltitude`.
   - missing required `geo_altitude_m` records are quarantined.

6. Add OpenSky history DB adapter. Completed.
   - Use `traffic.data.opensky.history`.
   - Request `baroaltitude` and `geoaltitude`.
   - Request joined flight metadata for airport operations.
   - Write raw returned rows before derived analysis.

7. Add history DB training assembly. Completed.
   - Build track-like derived records from history rows.
   - Split aircraft rows into segments by time gap and metadata changes.
   - Use the same event extraction/labeling path as REST.

8. Add tests. Completed locally.
   - Source response text is saved before parsing.
   - Raw track path remains value-equivalent to parsed source response.
   - Entry boundary estimate.
   - Exit boundary estimate.
   - Landing label evidence.
   - Depart label evidence.
   - Pass label evidence.
   - Derived training points include both `baro_altitude_m` and `geo_altitude_m`.
   - Missing `geo_altitude_m` records are quarantined when `--require-geo-altitude` is enabled.
   - Complete pass episode.
   - Track starts inside radius.
   - Track ends inside radius.
   - Multiple entry-exit pairs in one track.
   - Event ID stability.
   - Partition path generation.
   - History DB raw row storage preserves returned order.
   - History DB column aliases are normalized only in the derived view.
   - History DB `raw_index` aligns with the derived track path.

9. Remaining integration step.
   - Install/configure `traffic` and OpenSky DB access.
   - Run KRDU smoke fetch with `--training-source history-db`.
   - Inspect generated `history_rows`, `raw_tracks`, `airport_events`, `quarantine`, and `manifest` outputs.

## Validation Plan

Validate data quality with summary metrics:

- number of complete 5 nm episodes per airport and hour
- label distribution for `landing`, `depart`, `pass`, `unknown`, and `ambiguous`
- max sample gap per episode
- min distance to airport
- entry and exit boundary estimates should reference the raw source segment that crosses 5 nm
- percentage of incomplete/quarantined episodes
- duplicate event count
- raw response SHA256 verification success count
- points with both `baro_altitude_m` and `geo_altitude_m`
- records quarantined because geometric altitude is missing

Manual inspection:

- compare source response text against parsed raw track for several records
- sample several `landing` records
- sample several `depart` records
- sample several `pass` records
- compare extracted episodes against full raw tracks

## Open Questions for Review

1. Should incomplete 5 nm episodes be fully excluded by default, or stored in the main dataset with a quality flag?
2. For `pass`, should high-altitude overflight remain a subtype of `pass`, or should it become a separate label later?
3. Should the default partition granularity be `hour` or `day` for your training workflow?
4. Should model-ready features include a configurable context window before entry and after exit while keeping raw events unchanged?
5. Should the first implementation keep JSONL only, or also produce Parquet/Arrow for faster model training?
