#!/usr/bin/env bash
# Re-fetches the ADS-B sources into papers/ (PDFs are not tracked in git).
set -euo pipefail
dest="$(cd "$(dirname "$0")" && pwd)/papers"
mkdir -p "$dest"
get() { curl -fL --retry 3 -o "$dest/$1" "$2"; }
get Sun2021_1090_Megahertz_Riddle_2nd_ed.pdf "https://books.open.tudelft.nl/home/catalog/download/11/14/69?inline=1"
get Schafer2014_Bringing_Up_OpenSky_IPSN.pdf "https://www.cs.ox.ac.uk/files/7238/ipsn2014.pdf"

# official documents (FAA, ICAO)
odest="$(cd "$(dirname "$0")" && pwd)/official"
mkdir -p "$odest"
oget() { curl -fL --retry 3 -o "$odest/$1" "$2"; }
oget FAA_AC_90-114B_ADS-B_Operations.pdf "https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_90-114B.pdf"
oget FAA_AC_20-165B_ADS-B_Out_Airworthiness.pdf "https://www.faa.gov/documentlibrary/media/advisory_circular/ac_20-165b.pdf"
oget ICAO_APAC_2024_SP02_Annexes_and_Documents.pdf "https://www.icao.int/sites/default/files/APAC/Meetings/2024/2024%20ADS-B%20Workshop/5-Presentations/SP02-ICAO-ANNEXES-AND-DOCUMENTS.pdf"
