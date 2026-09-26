#!/usr/bin/env bash
# Re-download the FAA Minimum Vectoring Altitude (MVA) charts of the four terminal facilities that
# cover the thesis airports into data/MVA/<date>/ (git-ignored), then print the sha256 of each file.
#
#   bash docs/literature/minimum_vectoring_altitude/download.sh              # -> data/MVA/2026-09-26/
#   bash docs/literature/minimum_vectoring_altitude/download.sh 2026-12-01   # -> data/MVA/2026-12-01/
#
# Facilities: MSY (KMSY), RDU (KRDU), NCT (KSJC and KSMF), T75 (KSTL). The FAA MVA page lists two
# current charts per facility, FUS3 and FUS5 (README.md section 2), and both are fetched: the AIXM 5.1
# XML (for a parser) and the PDF (for reading by eye). Nothing else is downloaded.
#
# The FAA re-uses the same file name when a chart is revised, so a later download can hold different
# bytes under the same name. Compare the printed sha256 with the table in README.md.
# The script refuses to write into a date directory that already exists: pass a new date instead.
#
# Run from the main checkout. From a git worktree, set MVA_DATA to the main tree's data/MVA.
set -euo pipefail

DATE="${1:-2026-09-26}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT="${MVA_DATA:-$ROOT/data/MVA}/$DATE"

if [ -e "$OUT" ]; then
    echo "refusing to overwrite: $OUT already exists (pass another date)" >&2
    exit 1
fi
mkdir -p "$OUT"

# The page these links come from (lists every facility; returns 403 without a browser user agent):
#   https://www.faa.gov/air_traffic/flight_info/aeronav/digital_products/mva_mia/mva/
C=(curl -fsSL -A 'Mozilla/5.0' --retry 3 --retry-delay 5)

"${C[@]}" -o "$OUT/MSY_MVA_FUS3.xml" https://aeronav.faa.gov/MVA_Charts/aixm/MSY_MVA_FUS3.xml
"${C[@]}" -o "$OUT/MSY_MVA_FUS3.pdf" https://aeronav.faa.gov/MVA_Charts/pdf/MSY_MVA_FUS3.pdf
"${C[@]}" -o "$OUT/MSY_MVA_FUS5.xml" https://aeronav.faa.gov/MVA_Charts/aixm/MSY_MVA_FUS5.xml
"${C[@]}" -o "$OUT/MSY_MVA_FUS5.pdf" https://aeronav.faa.gov/MVA_Charts/pdf/MSY_MVA_FUS5.pdf

"${C[@]}" -o "$OUT/RDU_MVA_FUS3.xml" https://aeronav.faa.gov/MVA_Charts/aixm/RDU_MVA_FUS3.xml
"${C[@]}" -o "$OUT/RDU_MVA_FUS3.pdf" https://aeronav.faa.gov/MVA_Charts/pdf/RDU_MVA_FUS3.pdf
"${C[@]}" -o "$OUT/RDU_MVA_FUS5.xml" https://aeronav.faa.gov/MVA_Charts/aixm/RDU_MVA_FUS5.xml
"${C[@]}" -o "$OUT/RDU_MVA_FUS5.pdf" https://aeronav.faa.gov/MVA_Charts/pdf/RDU_MVA_FUS5.pdf

"${C[@]}" -o "$OUT/NCT_MVA_FUS3.xml" https://aeronav.faa.gov/MVA_Charts/aixm/NCT_MVA_FUS3.xml
"${C[@]}" -o "$OUT/NCT_MVA_FUS3.pdf" https://aeronav.faa.gov/MVA_Charts/pdf/NCT_MVA_FUS3.pdf
"${C[@]}" -o "$OUT/NCT_MVA_FUS5.xml" https://aeronav.faa.gov/MVA_Charts/aixm/NCT_MVA_FUS5.xml
"${C[@]}" -o "$OUT/NCT_MVA_FUS5.pdf" https://aeronav.faa.gov/MVA_Charts/pdf/NCT_MVA_FUS5.pdf

"${C[@]}" -o "$OUT/T75_MVA_FUS3.xml" https://aeronav.faa.gov/MVA_Charts/aixm/T75_MVA_FUS3.xml
"${C[@]}" -o "$OUT/T75_MVA_FUS3.pdf" https://aeronav.faa.gov/MVA_Charts/pdf/T75_MVA_FUS3.pdf
"${C[@]}" -o "$OUT/T75_MVA_FUS5.xml" https://aeronav.faa.gov/MVA_Charts/aixm/T75_MVA_FUS5.xml
"${C[@]}" -o "$OUT/T75_MVA_FUS5.pdf" https://aeronav.faa.gov/MVA_Charts/pdf/T75_MVA_FUS5.pdf

(cd "$OUT" && sha256sum -- *.xml *.pdf)
