#!/bin/bash
# The data the two-tier training chain reads (train_only/data_paths.txt, and the track files of
# train_only/track_paths.py), measured, copied or packed.
#
#   train_only/pack_data.sh size                   the size of each part and the total
#   train_only/pack_data.sh list                   every path, one a line (directories as given)
#   train_only/pack_data.sh rsync <dest>           rsync to <dest>: user@host:/path/to/the/clone, or a local directory
#   train_only/pack_data.sh tar <file.tar>         one tar file (no compression: the npz files are compressed)
#
# Run from a checkout whose data trees hold the data (the main checkout, or a worktree whose data trees link to it).
# The paths are read through the links; the copy holds real directories and files. Nothing at the source is written.
set -euo pipefail
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
cd "$ROOT"
ARTEFACT=4dTrajectory/outputs/POOLED/instruction_language/v19_20261014
mapfile -t ENTRIES < <(grep -v '^\s*#' train_only/data_paths.txt | sed '/^\s*$/d')
for entry in "${ENTRIES[@]}"; do
    test -e "$entry" || { echo "missing: $entry" >&2; exit 1; }
done
TRACKS=$(mktemp)
trap 'rm -f "$TRACKS"' EXIT
python3 train_only/track_paths.py $ARTEFACT > "$TRACKS"
all() { printf '%s\n' "${ENTRIES[@]}"; cat "$TRACKS"; }
case "${1:-}" in
    size)
        du -shL "${ENTRIES[@]}"
        echo "$(wc -l < "$TRACKS") track files: $(tr '\n' '\0' < "$TRACKS" | du -chL --files0-from=- | tail -1 | cut -f1)"
        echo "total: $(all | tr '\n' '\0' | du -chL --files0-from=- | tail -1 | cut -f1)"
        ;;
    list)
        all
        ;;
    rsync)
        DEST=${2:?the destination}
        all | rsync -a -k -R -r --info=progress2 --files-from=- . "$DEST/"
        ;;
    tar)
        OUT=${2:?the tar file}
        all | tar -chf "$OUT" -T -
        ls -l "$OUT"
        ;;
    *)
        sed -n '2,9p' "$0"; exit 2
        ;;
esac
