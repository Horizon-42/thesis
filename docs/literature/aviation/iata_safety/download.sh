#!/usr/bin/env bash
# Re-fetches the IATA safety sources into papers/ (PDFs are not tracked in git).
set -euo pipefail
dest="$(cd "$(dirname "$0")" && pwd)/papers"
mkdir -p "$dest"
get() { curl -fL --max-time 120 --retry 3 -o "$dest/$1" "$2"; }
get IATA_Safety_Report_2025_Executive_Summary.pdf "https://www.iata.org/contentassets/a8e49941e8824a058fee3f5ae0c005d9/safety-report-executive-summary-and-safety-overview_2025.pdf"
get IATA_Safety_Report_2024_Executive_Summary.pdf "https://www.iata.org/contentassets/a8e49941e8824a058fee3f5ae0c005d9/safety-report-executive-summary-and-safety-overview-2024_final.pdf"
get IATA_White_Paper_Flight_Path_Management_2026.pdf "https://www.iata.org/contentassets/c0f61fc821dc4f62bb6441d7abedb076/flight-path-management-recommendations-white-paper.pdf"
