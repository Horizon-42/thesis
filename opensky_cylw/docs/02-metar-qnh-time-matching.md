# METAR QNH-Time Matching for Altitude Correction (Design Document)

## Status
Design proposal. Not implemented in code yet.

## Objective
Replace or augment constant per-flight bias with a time-aware pressure correction based on METAR QNH observations, so waypoint altitude is corrected with local pressure at the corresponding timestamp.

## Motivation
OpenSky baro_altitude is pressure-based. A single constant offset can improve alignment, but it cannot capture time-varying pressure changes. METAR QNH-time matching can provide physically grounded correction per waypoint.

## Inputs
1. Flight waypoint stream (from OpenSky tracks/all):
- time_utc
- latitude
- longitude
- baro_altitude_m

2. Airport profile:
- airport ICAO
- airport elevation

3. METAR sequence near airport:
- observation_time_utc
- QNH or altimeter setting

## Data Source Options
- Aviation Weather Center (AWC) METAR APIs
- National weather service feeds
- other METAR archives used in your environment

The implementation should normalize all sources to one internal format:
- time_utc
- qnh_hpa
- source_id

## Core Model
### Pressure Correction Approximation
Using FAA rule-of-thumb relation:
- 1 inHg setting error corresponds to about 1000 ft altitude error

Equivalent hPa form:
- 1 inHg = 33.8639 hPa
- scale is about 29.53 ft per hPa

Given OpenSky barometric altitude h_baro and matched QNH value qnh_hpa:

h_corr_ft = h_baro_ft + (qnh_hpa - 1013.25) * 29.53
h_corr_m  = h_corr_ft * 0.3048

This is a first-order correction model and should be documented as approximate.

## Time Matching Strategy
For each waypoint timestamp t_wp:

1. Find nearest METAR report in time.
2. Require report freshness:
- abs(t_wp - t_metar) <= qnh_max_age_min
3. If both surrounding reports exist and gap is acceptable, allow linear interpolation:
- qnh(t_wp) = linear interpolation between bracketing reports
4. If no valid match:
- mark waypoint as unmatched
- apply fallback mode (auto-bias or raw)

Recommended defaults:
- qnh_max_age_min = 60
- max interpolation gap = 120 minutes

## Processing Flow
1. Fetch and parse METAR records for time range [track_start - margin, track_end + margin].
2. Convert all QNH values to hPa.
3. Build time-indexed QNH lookup.
4. For each waypoint:
- match QNH by time
- compute corrected altitude
5. Apply quality controls.
6. Export metadata for traceability.

## Quality Controls
- Reject impossible QNH values (for example outside 950 to 1050 hPa).
- Cap per-waypoint correction magnitude (for example <= 500 m).
- Optional smoothing of corrected altitude (small median or low-pass window) to avoid step artifacts at METAR boundary times.
- Log unmatched waypoint ratio.

## Fallback Policy
Use a deterministic fallback chain:
1. metar-qnh-time-match (if sufficient matched waypoints)
2. auto-bias (touchdown then approach)
3. raw

A minimum matched ratio threshold is recommended (for example >= 70%).

## Proposed CLI Extension
Add future options such as:
- --altitude-mode metar-qnh
- --metar-source awc
- --qnh-max-age-min 60
- --qnh-interpolate
- --qnh-min-match-ratio 0.7
- --qnh-max-correction-m 500
- --qnh-fallback-mode auto-bias

## Proposed Output Metadata
Per flight:
- altitude_correction_mode: metar-qnh-time-match
- qnh_match_ratio
- qnh_source
- qnh_time_window
- qnh_correction_stats (min/median/max)
- altitude_fallback_mode_used

Optional per-waypoint diagnostics can be exported in debug mode.

## Validation Plan
1. Ground consistency:
- compare corrected touchdown/near-runway altitudes with airport elevation.

2. Temporal consistency:
- verify no unrealistic jumps between adjacent corrected waypoints.

3. Comparative benchmark:
- compare metar-qnh mode against auto-bias on the same tracks.

4. Outlier checks:
- inspect flights with large residual offset after correction.

## Risks and Limitations
- METAR is airport-local and may not represent pressure away from terminal area.
- Temperature error is not corrected by QNH alone.
- Sparse METAR cadence can introduce staircase behavior without interpolation/smoothing.
- OpenSky track sampling is simplified and may miss key approach states.

## References
- OpenSky REST API, Track by Aircraft:
  https://openskynetwork.github.io/opensky-api/rest.html#track-by-aircraft
- FAA AIM, Section 7-2, altimeter setting error relation:
  https://www.faa.gov/air_traffic/publications/atpubs/aim_html/chap7_section_2.html
