#!/usr/bin/env bash
# The go-around folder cites the FAA PDFs kept in ../runway_assignment/official/ (7110.65BB Chg 3, AIM Chg 3);
# that folder's script fetches them.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here/../runway_assignment"
exec ./download.sh "$@"
