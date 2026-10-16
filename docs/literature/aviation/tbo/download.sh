#!/usr/bin/env bash
# Re-fetch the TBO sources. faa.gov refuses requests without a browser User-Agent.
# icao.int returns HTTP 403 to curl (2026-10-16); fetch those two papers by hand (see README.md).
cd "$(dirname "$0")"
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124 Safari/537.36"
dl(){ curl -fsSL -A "$UA" --max-time 120 -o "$1" "$2" && echo "ok $1"; }
dl FAA_TBO_handout_external.pdf "https://www.faa.gov/sites/faa.gov/files/air_traffic/technology/tbo/documents/TBO%20Handout%20external.pdf"
dl FAA_TBO_letter_2021-06.pdf "https://www.faa.gov/sites/faa.gov/files/air_traffic/technology/tbo/documents/TBO-letter-June2021-v4%5B90%5D.pdf"
dl FAA_NextGen_Final_Report_2025.pdf "https://www.faa.gov/nextgen/NextGen-Final-Report-2025.pdf"
dl FAA_TBO_REDAC_2018-03.pdf "https://www.faa.gov/sites/faa.gov/files/media/nasOps/2018/mar/508.05%20-%20TBO%20REDAC%20-%20Brief%20for%2003.18%20V1.0.pdf"
