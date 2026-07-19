# Activate the thesis conda env. SOURCE this file, then call aeroviz_activate_env.
#
# It must be sourced (not executed): activation has to mutate the caller's shell so the
# env's activate.d hooks run — zz-libstdcxx.sh prepends $CONDA_PREFIX/lib to
# LD_LIBRARY_PATH, and invoking envs/<env>/bin/python directly bypasses it (see CLAUDE.md
# "Operational Gotchas": that bypass is how the torch/matplotlib CXXABI clash comes back).
#
# Resolution:
#   AEROVIZ_CONDA_ENV set  -> that env and ONLY that env. A typo fails loudly instead of
#                             silently falling back to a different env.
#   otherwise              -> first of (aeroviz, aviation) whose python can import casadi.
#                             The NAME alone is not enough: on this machine `aviation` is
#                             a DIFFERENT project's env (py3.11, no casadi) while on other
#                             machines it IS the thesis env — the casadi probe is what
#                             tells them apart. An already-active qualifying env is kept.
#
# aeroviz_activate_env returns 0 with the env active, non-zero with a message on stderr.
# The CALLER decides whether failure is fatal (the test runner warns and continues, the
# fullstack launcher aborts).

_aeroviz_env_qualifies() {  # <path-to-env-python>
  [ -x "$1" ] && "$1" -c "import casadi" >/dev/null 2>&1
}

aeroviz_activate_env() {
  local candidates
  if [ -n "${AEROVIZ_CONDA_ENV:-}" ]; then
    candidates=("$AEROVIZ_CONDA_ENV")
  else
    candidates=(aeroviz aviation)
  fi

  local conda_base
  conda_base="$(conda info --base 2>/dev/null)"
  if [ -z "$conda_base" ]; then
    echo "error: conda is not on PATH — cannot locate a thesis env (tried: ${candidates[*]})" >&2
    return 1
  fi

  # Already inside a qualifying candidate env? Keep it (its activate.d hooks already ran).
  local name
  for name in "${candidates[@]}"; do
    if [ "${CONDA_DEFAULT_ENV:-}" = "$name" ] \
        && _aeroviz_env_qualifies "$conda_base/envs/$name/bin/python"; then
      return 0
    fi
  done

  if [ ! -f "$conda_base/etc/profile.d/conda.sh" ]; then
    echo "error: $conda_base/etc/profile.d/conda.sh not found — cannot activate a conda env" >&2
    return 1
  fi
  # shellcheck disable=SC1091
  source "$conda_base/etc/profile.d/conda.sh"

  for name in "${candidates[@]}"; do
    if _aeroviz_env_qualifies "$conda_base/envs/$name/bin/python"; then
      conda activate "$name"
      return 0
    fi
  done

  echo "error: no thesis conda env under $conda_base/envs — tried: ${candidates[*]}" >&2
  echo "       (an env qualifies when its python can 'import casadi'; regenerate from" >&2
  echo "       .env-backup/, or pin one explicitly with AEROVIZ_CONDA_ENV=<name>)" >&2
  return 1
}
