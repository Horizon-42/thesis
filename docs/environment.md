# Environment — the thesis conda env, the resolver, and the machine (reference)

Full text behind the index lines of `CLAUDE.md`, moved here VERBATIM on 2026-09-16 so that
file stays a short index (it is injected into every session that touches the tree). Only the
`### <ID>` headings and the dated **Correction / Note** paragraphs are new; every heading has
exactly one index line in that CLAUDE.md ending in its ID (`grep -n '^### # ·' docs/environment.md`).

**Maintenance: when a fact here changes, edit it HERE and keep its index line true; a new fact
gets a new ID here and ONE new line in the index.**

## Environment

### E1 · `aeroviz` (Python 3.12) is the thesis env on the Linux box

- **`aeroviz` (Python 3.12) is THE thesis env on this machine** — data acquisition (`traffic`,
  `pyopensky`), CIFP parsing (`cifparse`, `arinc424`), `casadi` + IPOPT, `openap`, the
  conda-forge geospatial stack, editable `geokit`, and `torch`. One env runs everything;
  `run_all_tests.sh` picks it and its `4dTrajectory` entry covers the ts_transformer suite.

### E2 · machine-dependent: on the Mac the thesis env is `aviation`

- **Machine-dependent (READ THIS FIRST on a new machine):** on THIS Mac there is no `aeroviz` env
  — `aviation` (py3.13, casadi 3.7.2) IS the thesis env, and `scripts/activate_aeroviz_env.sh`
  resolves to it correctly by probing for casadi. The warning below is written from the Linux
  compute box's perspective and misleads when read here; trust the resolver, which selects by
  CONTENT not name.

### E3 · `aviation` on the Linux box is NOT the thesis env; the resolver

- **`aviation` on the LINUX box is NOT the thesis env** — it belongs to
  `/home/supercomputing/studys/AivationTransformer` (a different project; pure-pip, py3.11).
  The name collides because on another machine the thesis env IS called `aviation`.
  **Do not install thesis packages into it and do not delete it.** `run_all_tests.sh` and
  `start_aeroviz_fullstack.sh` both resolve the env via `scripts/activate_aeroviz_env.sh`,
  which probes candidates with `import casadi` (so the wrong-project `aviation` here is
  skipped by content, not trusted by name), keeps a qualifying already-active env,
  ACTIVATES (never direct-execs `envs/<env>/bin/python` — activate.d hooks must run), and
  treats an explicit `AEROVIZ_CONDA_ENV` as the only candidate (a typo fails loudly).

### E4 · a py3.11 consolidation is blocked (`cifparse`)

- **Consolidating the thesis into a py3.11 env is BLOCKED**, tested: `cifparse` >= 2.0.4
  (aeroviz has 2.0.9) uses PEP 701 f-strings — nested same-type quotes — which is Python
  3.12+ syntax; every version from 2.0.4 up fails `compileall` on 3.11. Only 2.0.0 and
  earlier import there, i.e. a 9-patch regression in the ARINC 424 parser that feeds
  `approach_constraints`. Its PyPI metadata claims `>=3.10` and is simply wrong.

### E5 · `import torch` before `import traffic` and the activate.d fix

- **`import torch` BEFORE `import traffic` used to break matplotlib** — pip's manylinux torch
  wheel resolves `libstdc++.so.6` from `/lib/x86_64-linux-gnu` (CXXABI ≤ 1.3.13); once that
  SONAME is loaded, conda-forge matplotlib's `_c_internal_utils.so` (needs CXXABI_1.3.15) fails.
  The reverse import order worked, and `run_all_tests.sh` runs both suites in ONE pytest process
  with `4dTrajectory` (torch) ahead of `trajectory_data_process` (traffic) — i.e. exactly the
  failing order. Fixed by `$CONDA_PREFIX/etc/conda/activate.d/zz-libstdcxx.sh`, which prepends
  `$CONDA_PREFIX/lib` to `LD_LIBRARY_PATH` (with a matching `deactivate.d`). **This only applies
  under `conda activate`** — invoking `envs/aeroviz/bin/python` directly bypasses it and the old
  failure returns.

### E6 · why the env line used to say `aviation`

- Python env is conda **`aeroviz`**. This line used to say `aviation`, which is a DIFFERENT
  project's env on this machine, and that caused a near-miss deletion — hence the warning above.

### E7 · a nested worktree under `.claude/worktrees/` dirties the main tree

- **A nested git worktree under `.claude/worktrees/` shows up as untracked in the main tree**,
  and a formal ts run (`--campaign-id`/`--experiment-id`) refuses to start on a dirty tree
  (`experiment_index.begin_run`) — so creating one mid-campaign would abort the next arm. On this
  machine `.git/info/exclude` carries `.claude/worktrees/` (local, not committed); re-add it
  after a fresh clone before working in a worktree beside a running campaign.

### E8 · recovering a killed formal ts run

- **A killed formal run leaves a `running` `experiment_manifest.json` that makes the relaunch
  refuse the arm directory as occupied** (`begin_run` writes the manifest before training
  starts). Recovery: if the directory holds only `config.json` + the manifest, move it aside
  as `<arm>.aborted-<UTC>` (evidence, never deleted) and rerun the SAME campaign command —
  `run_ts.py frame_ablation` has no `--resume`; it skips every step whose artifact exists.

### E9 · env spec backups

- Env spec backups (regenerate `aeroviz` if ever needed): `.env-backup/aeroviz-pip-freeze.txt`,
  `aeroviz-conda-explicit.txt`, `aeroviz-environment.yml`.

### E10 · GPU

- GPU: RTX 4060, 8 GB (compute capability 8.9), cu128 wheels.

### E11 · RAM and swap

- This machine: 16 GB RAM, frequently swap-bound — memory pressure (Cesium + casadi + IDE +
  browser) causes UI lag independent of code changes.

### E12 · frontend build config

- Frontend build config (Cesium Ion token, vite-plugin-cesium, TS strict, jsdom):
  `aeroviz-4d/CLAUDE.md`.
