# OpenSky Airport Downloader

This folder contains a standalone downloader module (outside `aeroviz-4d`) that fetches real OpenSky data for airport-centered traffic (for example CYYC or CYLW) and converts it into the JSON format required by:

- `aeroviz-4d/python/generate_czml.py`

The two stages are intentionally decoupled:

1. `opensky_data_query/fetch_cylw_opensky.py` fetches + normalizes track data and writes `*_czml_input_*.json`
2. `aeroviz-4d/python/generate_czml.py` converts that JSON into `trajectories.czml`

## File

- `fetch_cylw_opensky.py`

## Pipeline Script

- `../run_fetch_and_generate.py` (repo root)

## Documentation

- `docs/01-current-bias-correction.md`
- `docs/02-metar-qnh-time-matching.md`
- `docs/03-training-data-fetch-store-redesign.md`

## Quick start

### Training data mode (recommended source: OpenSky history DB)

Training ingestion defaults to OpenSky history DB mode through `traffic.data.opensky.history`, not REST `/tracks/all`.

Prerequisites:

- install `traffic`
- configure OpenSky DB access for `traffic` using the [traffic OpenSky DB guide](https://traffic-viz.github.io/data_sources/opensky_db.html)

KRDU smoke command:

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

Outputs are written under:

- `outputs/history_rows/v2/`
- `outputs/raw_tracks/v2/`
- `outputs/airport_events/v2/`
- `outputs/quarantine/v2/`
- `outputs/manifests/v2/`

Use `--fetch-profile airport_ops` for a smaller arrival/departure-only run. Use `terminal_all` when pass-through tracks are required.

### ADSB.lol global history download

ADSB.lol `globe_history_2026` is published as daily global split-tar releases. The downloader below fetches one date; airport filtering should happen locally after download.

Dry-run release discovery:

```bash
python opensky_data_query/download_adsblol_history.py \
  --date 2026-04-19 \
  --dry-run
```

Download one day:

```bash
python opensky_data_query/download_adsblol_history.py \
  --date 2026-04-19
```

Download and stream-extract the split tar:

```bash
python opensky_data_query/download_adsblol_history.py \
  --date 2026-04-19 \
  --extract
```

Default output:

```text
opensky_data_query/outputs/adsblol_globe_history/YYYY.MM.DD/<release-tag>/
```

### 1) Live mode (no credentials, default airport CYYC)

```bash
/Users/liudongxu/opt/miniconda3/envs/aviation/bin/python \
  /Users/liudongxu/Desktop/studys/thesis/opensky_data_query/fetch_cylw_opensky.py \
  --mode live
```

This will:

1. fetch recent CYYC-related tracks from OpenSky
2. create CZML-input JSON under `opensky_data_query/outputs/`
3. print the generated input file path for the next stage

To run both stages in one command, use the root-level pipeline script:

```bash
/Users/liudongxu/opt/miniconda3/envs/aviation/bin/python \
  /Users/liudongxu/Desktop/studys/thesis/run_fetch_and_generate.py \
  --mode live \
  --airport CYYC \
  --altitude-mode auto-bias

# Reuse existing JSON and bypass live fetch:
# 1) existing *_raw_*.json -> run normalization/conversion + generate
# 2) existing *_czml_input_*.json -> run generate directly
/Users/liudongxu/opt/miniconda3/envs/aviation/bin/python \
  /Users/liudongxu/Desktop/studys/thesis/run_fetch_and_generate.py \
  --input-json /Users/liudongxu/Desktop/studys/thesis/opensky_data_query/outputs/cyyc_raw_20260415T152417Z.json
```

Altitude handling (research note):

- Default is `--altitude-mode raw`, which preserves OpenSky track altitude as-is.
- `--altitude-mode touchdown-bias` applies a per-flight constant offset estimated from near-runway on-ground samples.
- `--altitude-mode approach-bias` applies a per-flight constant offset estimated from near-runway low-altitude samples (works even when on_ground is sparse).
- `--altitude-mode auto-bias` tries touchdown-bias first, then approach-bias.
- All bias modes preserve trajectory shape and only shift altitude.

Normalization switch (debug note):

- `--disable-normalization` bypasses airport/landing filtering and bias correction, and exports raw OpenSky `tracks/all` waypoints directly into CZML-input schema.
- Use this mode when you want to verify whether empty or distorted output comes from normalization logic vs source track data quality.

By default, only trajectories that include touchdown near the airport are kept,
so landing process is preserved.

To switch airport (example CYLW):

```bash
/Users/liudongxu/opt/miniconda3/envs/aviation/bin/python \
  /Users/liudongxu/Desktop/studys/thesis/opensky_data_query/fetch_cylw_opensky.py \
  --airport CYLW \
  --mode live

# Keep raw altitude (default)
/Users/liudongxu/opt/miniconda3/envs/aviation/bin/python \
  /Users/liudongxu/Desktop/studys/thesis/opensky_data_query/fetch_cylw_opensky.py \
  --mode live \
  --airport CYYC \
  --altitude-mode raw

# Apply touchdown-bias altitude correction (optional)
/Users/liudongxu/opt/miniconda3/envs/aviation/bin/python \
  /Users/liudongxu/Desktop/studys/thesis/opensky_data_query/fetch_cylw_opensky.py \
  --mode live \
  --airport CYYC \
  --altitude-mode touchdown-bias \
  --min-ground-samples 2 \
  --max-altitude-bias-m 400

# Recommended automatic altitude correction (touchdown first, approach fallback)
/Users/liudongxu/opt/miniconda3/envs/aviation/bin/python \
  /Users/liudongxu/Desktop/studys/thesis/opensky_data_query/fetch_cylw_opensky.py \
  --mode live \
  --airport CYYC \
  --altitude-mode auto-bias \
  --min-ground-samples 2 \
  --max-altitude-bias-m 400 \
  --approach-alt-buffer-m 450

# Disable normalization to export raw tracks for debugging
/Users/liudongxu/opt/miniconda3/envs/aviation/bin/python \
  /Users/liudongxu/Desktop/studys/thesis/opensky_data_query/fetch_cylw_opensky.py \
  --mode live \
  --airport CYYC \
  --disable-normalization

# Same via one-command pipeline wrapper
/Users/liudongxu/opt/miniconda3/envs/aviation/bin/python \
  /Users/liudongxu/Desktop/studys/thesis/run_fetch_and_generate.py \
  --mode live \
  --airport CYYC \
  --disable-normalization
```

### 2) CZML historical mode (REST, requires OpenSky OAuth credentials)

Set credentials:

```bash
export OPENSKY_CLIENT_ID="..."
export OPENSKY_CLIENT_SECRET="..."
```

Run:

```bash
/Users/liudongxu/opt/miniconda3/envs/aviation/bin/python \
  /Users/liudongxu/Desktop/studys/thesis/opensky_data_query/fetch_cylw_opensky.py \
  --mode historical \
  --airport CYYC \
  --begin "2026-04-05T00:00:00Z" \
  --end "2026-04-06T00:00:00Z"
```

## Notes

- REST historical arrivals/departures require authenticated OAuth access.
- `--mode auto` will choose REST `historical` if credentials exist for CZML mode, otherwise `live`.
- Output files are written to `opensky_data_query/outputs/` with UTC timestamp suffixes.
