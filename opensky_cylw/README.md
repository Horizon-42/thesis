# OpenSky Airport Downloader

This folder contains a standalone downloader module (outside `aeroviz-4d`) that fetches real OpenSky data for CYLW and converts it into the JSON format required by:

- `aeroviz-4d/python/generate_czml.py`

## File

- `fetch_cylw_opensky.py`

## Quick start

### 1) Live mode (no credentials, default airport CYYC)

```bash
/Users/liudongxu/opt/miniconda3/envs/aviation/bin/python \
  /Users/liudongxu/Desktop/studys/thesis/opensky_cylw/fetch_cylw_opensky.py \
  --mode live \
  --to-czml
```

This will:

1. fetch recent CYYC-related tracks from OpenSky
2. create CZML-input JSON under `opensky_cylw/outputs/`
3. update `aeroviz-4d/public/data/trajectories.czml`

By default, only trajectories that include touchdown near the airport are kept,
so landing process is preserved.

To switch airport (example CYLW):

```bash
/Users/liudongxu/opt/miniconda3/envs/aviation/bin/python \
  /Users/liudongxu/Desktop/studys/thesis/opensky_cylw/fetch_cylw_opensky.py \
  --airport CYLW \
  --mode live \
  --to-czml
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
  --end "2026-04-06T00:00:00Z" \
  --to-czml
```

## Notes

- OpenSky historical arrivals/departures require authenticated OAuth access.
- `--mode auto` will choose `historical` if credentials exist, otherwise `live`.
- Output files are written to `opensky_cylw/outputs/` with UTC timestamp suffixes.
