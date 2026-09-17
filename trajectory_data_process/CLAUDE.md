# trajectory_data_process — acquisition, harvest, datasets

`harvest/` is the download pipeline: fetch → reconstruct → assign (one runway per track) →
`tracks/` + `approach/`. CLI: `python -m trajectory_data_process.harvest --airport KRDU`.
Geometry lives in `final_approach/` (see its `CLAUDE.md`); this tree is fetch, reconstruct,
store, roster.

**This file is an index**: each line ends in the ID of its full text in
`docs/06-harvest-reference.md` (moved there verbatim 2026-09-16, with dated corrections). A new
fact gets a new ID there and one line here.

## Which rebuild — read before running anything

| need | flag | what it touches |
|---|---|---|
| an observed-record contract change (a new `source` field, a resolver fix, records without `crossing_span`) | `--observed-only` | `approach/` only (records, `summary.json`, report, publication); `arrivals/` and `lateral_pass_eligibility.json` untouched, no CZML (TD3) |
| an arrival-schema bump or evaluation-policy change | `--evaluate-only` | re-rosters `arrivals/` from stored `tracks/` and **DELETES `lateral_pass_eligibility.json`**, which nothing rebuilds for you — run `lateral_eligibility.ensure_lateral_pass_roster(<arrivals>/manifest.json)` right after (TD16, TD17) |
| a changed runway-data cycle or assignment method | `--reclassify-existing` | re-derives runway assignment — NOT the rebuild command (TD18) |

Never under a running ts campaign: a rebuilt roster changes every ts dataset split (2026-08-21: a
rebuild under a running campaign cost an arm). A harvest killed with SIGTERM can still
finish its write — confirm the process is gone (`kill -0`) before rebuilding downstream (TD19).
(`evaluation/arrival.py`'s stale-record error still names `--evaluate-only`; `--observed-only` is
the lighter cure — `docs/code-health-followups.md`.)

## Contracts

- **Harvest is manifest-only.** `tracks/manifest.json` rosters all four measured outcomes;
  `arrivals/manifest.json` rosters only assigned, CIFP-targeted, final-entry-cropped model
  inputs and records every exclusion. Scenario, optimizer-reference, and TS loaders follow that
  roster and never glob, so an orphan/rejected/stale JSON cannot enter a model split.
- The threshold event's `crossing_ground_speed_m_s` is GROUND speed (direct: interpolated at the
  bracket; censored: OLS over the same kept samples as the position fit), optional on read;
  evaluation judges the observed baseline on it as a STATED proxy (TD1).
- Observed evaluation records carry a `crossing_span` and write `source.aircraft_type` whenever
  the identity resolves, dynamics or not (`source.mass_source` says where the mass came from);
  `final_time_s` stays on the last MEASURED row (TD2).
- `unassignable` (the receiver lost it) and `not_established` (not stabilised) must stay
  distinct outcomes — both counted, neither dropped (TD4).
- **Archived history rows are in FEET; `fetch_history_dataframe` output is in METRES** —
  `harvest.reconstruct_tracks` takes a REQUIRED `altitude_units`, no default, no sniffing (TD5).
- An arrival segment must not BEGIN on the ground: `arrival_segment` returns `"takeoff"`,
  rostered as `takeoff_in_segment` (75 flights, 64 at KSJC). The test is ALTITUDE ONLY and applies
  to the segment the ring cut produced, not the raw track (TD6).
- A landing ends at a sustained ON-GROUND run and a spatial crop keeps only the final contiguous
  run; `_complete()` requires BOTH dep and arr airports (TD7).

## Runway thresholds & TCH (static data feeding the harvest)

- `runway_thresholds.json` holds LANDING (displaced) thresholds, not pavement ends
  (`runway-thresholds-v2`). **Fix the GENERATOR, never the JSON** —
  `acquisition/runways.landing_thresholds_from_row` is the one computation (TD8).
- **Per-runway TCH is published in the CIFP and is NOT 15 m** (15.27–18.11 m): LPV Path Point
  records, else the RNAV approach's runway leg where LNAV/VNAV minima are published (KRDU 32,
  KSMF 35R; `tch_source == "faa_cifp_approach_leg"`); only a runway with no RNAV procedure keeps
  `None` (KRDU 14). The PG runway record's TCH is the ILS/VGSI figure — not read (TD9).

## Altitude outlier repair

- **Repaired in the VIEW at read time, never in `tracks/`** — editing the store breaks the
  roster's per-record SHA-256, `--reclassify-existing` and `source_integrity.retained_rows` (TD10).
- Criterion: deviation from the ±2-sample median exceeding BOTH 100 m AND `25 m/s × min(adjacent
  gap)` — both halves are load-bearing (TD11); incidence ~0.0027 % of samples, only
  `samples[i][3]` changes, a row is never dropped (TD12).
- Audit/republish: `python -m trajectory_data_process.altitude_outliers [--rerender-czml]`; stored
  `observed_threshold_event`s were fitted from raw samples — only `--reclassify-existing`
  re-derives those (TD13).

## Constants

- Outlier filter (`harvest/altitude_filter.py`, single source; `AltitudePolicy`):
  `half_window = 2`, `min_deviation_m = 100.0`, `max_vertical_rate_m_s = 25.0` (TD14).
- Arrival truncation (`arrival_segment.py`): `ENTRY_RADIUS_KM = 25`, `ENTRY_HYSTERESIS_SAMPLES = 3`,
  `LOCAL_START_RADIUS_KM = 5`, `GROUND_START_AGL_M = 100` (in an empty band of the fleet);
  `field_elevation_m` is REQUIRED — the waypoints are HAE (TD15).
- Arrival manifest schema: the writer is `harvest-arrivals-v6-published-vertical-path`, loaders
  accept v5 and v6 (`READABLE_SCHEMA_VERSIONS`), v4 fails loudly; the five manifests on disk are
  still v5 (read 2026-09-16) (TD16).
