#!/usr/bin/env bash
# Re-fetches the GAO report into papers/ (PDFs are not tracked in git).
set -euo pipefail
dest="$(cd "$(dirname "$0")" && pwd)/papers"
mkdir -p "$dest"
curl -fL --retry 3 -o "$dest/GAO-24-105254_ATC_Modernization_Program_Management.pdf" "https://www.gao.gov/assets/gao-24-105254.pdf"
