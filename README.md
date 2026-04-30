# Thesis Workspace

## AeroViz RNAV Chart Data

The AeroViz procedure details page can link each generated RNAV procedure to a
local FAA chart PDF. Chart PDFs are stored outside the frontend source first,
then copied into the browser-served public data directory when procedure data is
generated.

### Download FAA RNAV Charts

Download both RNAV(GPS) and RNAV(RNP) approach charts for one airport:

```bash
python aeroviz-4d/python/download_faa_rnav_charts.py KRDU
```

By default this discovers the current FAA d-TPP cycle and saves PDFs under:

```text
data/RNAV_CHARTS/<ICAO>/
```

Use a fixed FAA d-TPP cycle when you need reproducible data:

```bash
python aeroviz-4d/python/download_faa_rnav_charts.py KRDU --cycle 2604
```

Preview matching chart URLs without downloading:

```bash
python aeroviz-4d/python/download_faa_rnav_charts.py KRDU --cycle 2604 --dry-run
```

Download only one chart type:

```bash
python aeroviz-4d/python/download_faa_rnav_charts.py KRDU --modes RNAV_GPS
python aeroviz-4d/python/download_faa_rnav_charts.py KRDU --modes RNAV_RNP
```

If you explicitly want to write directly into the frontend public assets, use:

```bash
python aeroviz-4d/python/download_faa_rnav_charts.py KRDU --public
```

The preferred workflow is to keep source PDFs in `data/RNAV_CHARTS/<ICAO>/` and
let the procedure generation step publish browser-ready copies and
`charts/index.json`.

### Regenerate Procedure Data

After downloading charts, regenerate the airport procedure data:

```bash
./generate_aeroviz_airport_procedure_data.sh KRDU
```

This writes generated browser data under:

```text
aeroviz-4d/public/data/airports/<ICAO>/
```

The chart manifest generator recognizes FAA RNAV(GPS) filenames such as
`00516RY5L.PDF` and RNAV(RNP) filenames such as `00516RRZ23L.PDF`, then maps
them to the corresponding generated procedure reference.

### Validate CIFP Parser Packages

The production AeroViz parser lives in:

```text
aeroviz-4d/python/cifp_parser.py
```

To compare it against third-party parser packages, run:

```bash
/Users/liudongxu/opt/miniconda3/envs/aviation/bin/python \
  aeroviz-4d/python/validate_cifp_parser_packages.py --airport KRDU
```

The validator compares:

- local fixed-width parser output
- `arinc424` record-level decoding
- `cifparse` structured CIFP objects

Current KRDU result: `cifparse` is the best candidate for a future primary
parser because it matches procedure legs and required fix coordinates while
providing higher-level procedure, runway, terminal-waypoint, and enroute-waypoint
objects. Keep `arinc424` as an audit/cross-check decoder because it is closer to
raw ARINC 424 records.
