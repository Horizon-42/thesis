# Current Altitude Bias Correction in trajectory_normalization.py

## Scope
This document describes the altitude correction logic implemented in opensky_cylw/trajectory_normalization.py, inside function track_to_czml_flight.

The fetch CLI (opensky_cylw/fetch_cylw_opensky.py) imports this module for normalization.

## Goal
OpenSky track waypoints provide barometric altitude. In practice, this can be offset from airport elevation during terminal operations. The current implementation applies an optional per-flight constant altitude bias to improve visual alignment in trajectory playback.

The method is intentionally simple:
- keep trajectory shape unchanged
- estimate one constant bias per flight
- use robust statistics (median) for sample aggregation
- reject extreme bias values

## Data Inputs Used
For each waypoint in OpenSky tracks/all path:
- time
- latitude
- longitude
- baro_altitude (meters)
- on_ground flag

Airport profile inputs:
- airport latitude and longitude
- airport elevation in meters

## Correction Modes
The CLI option --altitude-mode supports four modes:

1. raw
- No correction.
- altitude output equals OpenSky baro_altitude.

2. touchdown-bias
- Select near-runway on-ground samples.
- Sample condition:
  - on_ground is true
  - distance_to_airport <= landing_radius_km
- Compute candidate bias:
  - bias_m = airport_elev_m - median(touchdown_altitudes)
- Apply only when:
  - sample_count >= min_ground_samples
  - abs(bias_m) <= max_altitude_bias_m

3. approach-bias
- Fallback-friendly mode for cases where on_ground is missing.
- Select near-airport low-altitude samples.
- Sample condition:
  - distance_to_airport <= landing_radius_km
  - altitude <= airport_elev_m + approach_alt_buffer_m
- Compute candidate bias:
  - bias_m = airport_elev_m - median(near_low_altitudes)
- Apply only when:
  - sample_count >= min_ground_samples
  - abs(bias_m) <= max_altitude_bias_m

4. auto-bias
- Try touchdown-bias first.
- If not applied, try approach-bias.

## Formula
For a chosen sample set S = {h1, h2, ..., hn}:

bias_m = airport_elev_m - median(S)

For each waypoint altitude h:

h_corrected = h + bias_m

## Why Median
Median is more robust than mean under outliers and sparse noise spikes in ADS-B derived tracks. This reduces instability when a few sample points are bad.

## Guard Rails
The implementation includes practical safety checks:
- min_ground_samples: minimum support needed to trust the estimate
- max_altitude_bias_m: hard cap to reject unrealistic corrections
- no correction when conditions are not met

## Output Metadata in CZML Input JSON
Each exported flight includes:
- altitude_source
- altitude_correction_mode
- altitude_bias_m
- altitude_bias_applied
- altitude_bias_source
- altitude_ground_samples
- altitude_approach_samples

These fields make the correction transparent and auditable.

## Relevant Parameters and Defaults
- --altitude-mode: raw | touchdown-bias | approach-bias | auto-bias (default raw)
- --landing-radius-km: 15.0
- --min-ground-samples: 2
- --max-altitude-bias-m: 400.0
- --approach-alt-buffer-m: 450.0

## Suggested Usage
For best vertical alignment in live CYYC data:
- use --altitude-mode auto-bias
- keep default guard rails initially
- tighten max-altitude-bias-m if overly large corrections appear in your area

Example:

/Users/liudongxu/opt/miniconda3/envs/aviation/bin/python \
  /Users/liudongxu/Desktop/studys/thesis/run_fetch_and_generate.py \
  --mode live \
  --airport CYYC \
  --altitude-mode auto-bias

## Limitations
- This is a per-flight constant offset model, not a full atmospheric correction model.
- It does not account for spatial pressure gradients or temperature profile errors.
- It is intended for trajectory visualization alignment, not operational navigation or safety use.

## References
- OpenSky REST API, Track by Aircraft:
  https://openskynetwork.github.io/opensky-api/rest.html#track-by-aircraft
- FAA AIM, Barometric Altimeter Errors and Setting Procedures:
  https://www.faa.gov/air_traffic/publications/atpubs/aim_html/chap7_section_2.html
