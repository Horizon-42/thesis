#!/usr/bin/env bash
# Re-fetches the SESAR sources into papers/ (PDFs are not tracked in git).
set -euo pipefail
dest="$(cd "$(dirname "$0")" && pwd)/papers"
mkdir -p "$dest"
get() { curl -fL --retry 3 -o "$dest/$1" "$2"; }
get SESAR_Deployment_Programme_2024.pdf "https://assets.ctfassets.net/krj50g99u3hm/4lu3tiExvyepltjL8cU7uq/9df2bd1333710236fad264b3ac37a9cf/20240930_SESAR_Deployment_Programme_2024.pdf"
get SESAR_3_JU_Annual_Activity_Report_2024.pdf "https://www.sesarju.eu/sites/default/files/documents/reports/SESAR%20Annual%20Activity%20Report%202024.pdf"
