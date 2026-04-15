# OpenSky Airport Downloader

This folder contains a standalone downloader module (outside `aeroviz-4d`) that fetches real OpenSky data for airport-centered traffic (for example CYYC or CYLW) and converts it into the JSON format required by:

- `aeroviz-4d/python/generate_czml.py`

The two stages are intentionally decoupled:

1. `opensky_cylw/fetch_cylw_opensky.py` fetches + normalizes track data and writes `*_czml_input_*.json`
2. `aeroviz-4d/python/generate_czml.py` converts that JSON into `trajectories.czml`

## File

- `fetch_cylw_opensky.py`

## Pipeline Script

- `../run_fetch_and_generate.py` (repo root)

## Documentation

- `docs/01-current-bias-correction.md`
- `docs/02-metar-qnh-time-matching.md`

## Quick start

### 1) Live mode (no credentials, default airport CYYC)

```bash
/Users/liudongxu/opt/miniconda3/envs/aviation/bin/python \
  /Users/liudongxu/Desktop/studys/thesis/opensky_cylw/fetch_cylw_opensky.py \
  --mode live
```

This will:

1. fetch recent CYYC-related tracks from OpenSky
2. create CZML-input JSON under `opensky_cylw/outputs/`
3. print the generated input file path for the next stage

To run both stages in one command, use the root-level pipeline script:

```bash
/Users/liudongxu/opt/miniconda3/envs/aviation/bin/python \
  /Users/liudongxu/Desktop/studys/thesis/run_fetch_and_generate.py \
  --mode live \
  --airport CYYC \
  --altitude-mode auto-bias
```

Altitude handling (research note):

- Default is `--altitude-mode raw`, which preserves OpenSky track altitude as-is.
- `--altitude-mode touchdown-bias` applies a per-flight constant offset estimated from near-runway on-ground samples.
- `--altitude-mode approach-bias` applies a per-flight constant offset estimated from near-runway low-altitude samples (works even when on_ground is sparse).
- `--altitude-mode auto-bias` tries touchdown-bias first, then approach-bias.
- All bias modes preserve trajectory shape and only shift altitude.

By default, only trajectories that include touchdown near the airport are kept,
so landing process is preserved.

To switch airport (example CYLW):

```bash
/Users/liudongxu/opt/miniconda3/envs/aviation/bin/python \
  /Users/liudongxu/Desktop/studys/thesis/opensky_cylw/fetch_cylw_opensky.py \
  --airport CYLW \
  --mode live

# Keep raw altitude (default)
/Users/liudongxu/opt/miniconda3/envs/aviation/bin/python \
  /Users/liudongxu/Desktop/studys/thesis/opensky_cylw/fetch_cylw_opensky.py \
  --mode live \
  --airport CYYC \
  --altitude-mode raw

# Apply touchdown-bias altitude correction (optional)
/Users/liudongxu/opt/miniconda3/envs/aviation/bin/python \
  /Users/liudongxu/Desktop/studys/thesis/opensky_cylw/fetch_cylw_opensky.py \
  --mode live \
  --airport CYYC \
  --altitude-mode touchdown-bias \
  --min-ground-samples 2 \
  --max-altitude-bias-m 400

# Recommended automatic altitude correction (touchdown first, approach fallback)
/Users/liudongxu/opt/miniconda3/envs/aviation/bin/python \
  /Users/liudongxu/Desktop/studys/thesis/opensky_cylw/fetch_cylw_opensky.py \
  --mode live \
  --airport CYYC \
  --altitude-mode auto-bias \
  --min-ground-samples 2 \
  --max-altitude-bias-m 400 \
  --approach-alt-buffer-m 450
```

### 2) Historical mode (recommended, requires OpenSky OAuth credentials)

Set credentials:

```bash
export OPENSKY_CLIENT_ID="..."
export OPENSKY_CLIENT_SECRET="..."
```

Run:

```bash
/Users/liudongxu/opt/miniconda3/envs/aviation/bin/python \
  /Users/liudongxu/Desktop/studys/thesis/opensky_cylw/fetch_cylw_opensky.py \
  --mode historical \
  --airport CYYC \
  --begin "2026-04-05T00:00:00Z" \
  --end "2026-04-06T00:00:00Z"
```

## Notes

- OpenSky historical arrivals/departures require authenticated OAuth access.
- `--mode auto` will choose `historical` if credentials exist, otherwise `live`.
- Output files are written to `opensky_cylw/outputs/` with UTC timestamp suffixes.
