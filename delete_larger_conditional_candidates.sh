#!/usr/bin/env bash

set -euo pipefail

mount_is_at_or_inside_target() {
  local mountpoint="$1"
  local target="$2"
  [[ "${mountpoint}" == "${target}" || "${mountpoint}" == "${target}/"* ]]
}

# Permit the pure mount-boundary predicate to be sourced by its regression test
# without running any validation or deletion logic.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  return 0
fi

usage() {
  cat <<'EOF'
Usage:
  ./delete_larger_conditional_candidates.sh             # dry run (default)
  ./delete_larger_conditional_candidates.sh --dry-run   # dry run
  ./delete_larger_conditional_candidates.sh --delete    # validate, confirm, delete

This script deletes only the concrete paths identified in the storage audit.
It deliberately does not delete any Conda environment.
EOF
}

if (( $# > 1 )); then
  usage >&2
  exit 2
fi

mode="${1:---dry-run}"
case "${mode}" in
  --dry-run | --delete)
    ;;
  -h | --help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

if [[ "$(id -un)" != "supercomputing" ]]; then
  echo "Refusing: run this script as supercomputing without sudo." >&2
  exit 1
fi

directory_targets=(
  /home/supercomputing/works/build_mesh2/.deps
  /home/supercomputing/.local/share/ov/data/exts
  /home/supercomputing/.android/avd/mb_x86_64_api34.avd
  /home/supercomputing/.config/Code/User/workspaceStorage/a941234ac9792e99432fb030ce389b8d
)

file_targets=(
  /home/supercomputing/.android/avd/mb_x86_64_api34.ini
  /home/supercomputing/Downloads/FullIJCNN2013.zip
  /home/supercomputing/Downloads/google-chrome-stable_current_amd64.deb
  /home/supercomputing/Downloads/commandlinetools-linux-14742923_latest.zip
  /home/supercomputing/Downloads/code_1.132.0-1785860022_amd64.deb
)

validate_directory() {
  local target="$1"

  if [[ ! -d "${target}" || -L "${target}" ]]; then
    echo "Refusing: expected a real directory: ${target}" >&2
    exit 1
  fi
  if [[ "$(realpath -e -- "${target}")" != "${target}" ]]; then
    echo "Refusing: directory resolves somewhere unexpected: ${target}" >&2
    exit 1
  fi
  if [[ "$(stat -c '%U' -- "${target}")" != "supercomputing" ]]; then
    echo "Refusing: directory has an unexpected owner: ${target}" >&2
    exit 1
  fi
}

validate_file() {
  local target="$1"

  if [[ ! -f "${target}" || -L "${target}" ]]; then
    echo "Refusing: expected a real file: ${target}" >&2
    exit 1
  fi
  if [[ "$(realpath -e -- "${target}")" != "${target}" ]]; then
    echo "Refusing: file resolves somewhere unexpected: ${target}" >&2
    exit 1
  fi
  if [[ "$(stat -c '%U' -- "${target}")" != "supercomputing" ]]; then
    echo "Refusing: file has an unexpected owner: ${target}" >&2
    exit 1
  fi
}

echo "Validating exact deletion targets..."
for target in "${directory_targets[@]}"; do
  validate_directory "${target}"
done
for target in "${file_targets[@]}"; do
  validate_file "${target}"
done

if ! git -C /home/supercomputing/works/build_mesh2 check-ignore -q -- .deps; then
  echo "Refusing: build_mesh2/.deps is no longer ignored by Git." >&2
  exit 1
fi
if [[ -n "$(git -C /home/supercomputing/works/build_mesh2 ls-files -- .deps)" ]]; then
  echo "Refusing: build_mesh2/.deps unexpectedly contains tracked files." >&2
  exit 1
fi

workspace_file=/home/supercomputing/.config/Code/User/workspaceStorage/a941234ac9792e99432fb030ce389b8d/workspace.json
if ! grep -Fq 'file:///home/supercomputing/studys/ExplainableAI_Medical' "${workspace_file}"; then
  echo "Refusing: the VS Code workspace identity has changed." >&2
  exit 1
fi

if [[ ! -d /home/supercomputing/Downloads/FullIJCNN2013 ]]; then
  echo "Refusing: the extracted FullIJCNN2013 dataset is missing." >&2
  exit 1
fi

while IFS= read -r mountpoint; do
  for target in "${directory_targets[@]}"; do
    if mount_is_at_or_inside_target "${mountpoint}" "${target}"; then
      echo "Refusing: a filesystem is mounted at or inside ${target}: ${mountpoint}" >&2
      exit 1
    fi
  done
done < <(findmnt -rn -o TARGET)

active_processes="$(
  ps -eo pid=,comm=,args= |
    awk '$2 ~ /^(code|code-insiders|emulator|qemu-system.*|kit|omni.*)$/ {print}'
)"
if [[ -n "${active_processes}" ]]; then
  echo "Refusing: close these related applications first:" >&2
  echo "${active_processes}" >&2
  exit 1
fi

echo
echo "Validated targets:"
du -sch -- "${directory_targets[@]}" "${file_targets[@]}"
echo
echo "Effects:"
echo "  - build_mesh2 dependencies will need to be restored before rebuilding."
echo "  - Omniverse extensions will need to be downloaded again when needed."
echo "  - mb_x86_64_api34 and all of its emulator user data will be removed."
echo "  - saved VS Code state for ExplainableAI_Medical will be removed."
echo "  - the extracted FullIJCNN2013 directory will be preserved."
echo "  - no Conda environments are included."

if [[ "${mode}" == "--dry-run" ]]; then
  echo
  echo "DRY RUN COMPLETE: nothing was deleted."
  echo "Run '$0 --delete' to perform the deletion."
  exit 0
fi

if [[ ! -t 0 ]]; then
  echo "Refusing: --delete requires an interactive terminal." >&2
  exit 1
fi

echo
read -r -p 'Type DELETE-CONDITIONAL-CANDIDATES to continue: ' confirmation
if [[ "${confirmation}" != "DELETE-CONDITIONAL-CANDIDATES" ]]; then
  echo "Cancelled. Nothing was deleted."
  exit 1
fi

rm -rf --one-file-system -- \
  /home/supercomputing/works/build_mesh2/.deps \
  /home/supercomputing/.local/share/ov/data/exts \
  /home/supercomputing/.android/avd/mb_x86_64_api34.avd \
  /home/supercomputing/.config/Code/User/workspaceStorage/a941234ac9792e99432fb030ce389b8d

rm -f -- \
  /home/supercomputing/.android/avd/mb_x86_64_api34.ini \
  /home/supercomputing/Downloads/FullIJCNN2013.zip \
  /home/supercomputing/Downloads/google-chrome-stable_current_amd64.deb \
  /home/supercomputing/Downloads/commandlinetools-linux-14742923_latest.zip \
  /home/supercomputing/Downloads/code_1.132.0-1785860022_amd64.deb

echo
echo "Deletion completed. Current root-filesystem space:"
df -h /
