# Faster IPOPT via the HSL linear solvers (MA57 / MA27)

IPOPT's per-iteration cost is dominated by factorizing the sparse symmetric-indefinite
KKT system. This project's casadi wheel ships IPOPT built with **MUMPS** (open source,
no license). On these small–medium NLPs (~800 vars) the HSL solvers **MA57 / MA27** are
typically **2–4× faster** and change nothing about the solution.

**No rebuild of IPOPT or CasADi is needed.** The casadi 3.7 wheel's IPOPT is compiled with
the HSL *runtime loader* (the `hsllib` option is accepted — verified on this machine). We
only need to compile a **CoinHSL** shared library and point IPOPT at it via two environment
variables (already wired in `collocation/components.py`).

---

## 1. Get a CoinHSL source (two free routes)

Both routes require creating an STFC account and completing a **£0 order**, then downloading
from *My account / Downloads* — there is no anonymous direct-download link.

| Route | Contains | Wait | Terms |
|---|---|---|---|
| **Coin-HSL Archive** <https://licences.stfc.ac.uk/product/coin-hsl-archive> | **MA27** (+ older solvers) | instant | free, **personal use only, no redistribution** |
| **Coin-HSL (full)** <https://licences.stfc.ac.uk/product/coin-hsl> | MA27, MA57, MA77, MA86, MA97 | ~1 business day (academic approval, institutional email) | HSL Academic Licence, free |

**Recommended for this project: the Archive / MA27.** For these ~800-var NLPs MA27 is often
the *fastest* HSL solver anyway (single-threaded, low overhead), it **needs no METIS** (the
one macOS build dependency that tends to fight back), and the Archive is instant with no
academic-approval wait. A thesis is personal academic research, so the Archive's personal-use
terms are fine (we never redistribute the library — it stays on your machine, git-ignored).

Download the archive tarball (e.g. `coinhsl-archive-2024.05.15.tar.gz`); the full package is
`coinhsl-2023.11.17.tar.gz`. The build below is the same for either.

> MA27 needs no METIS; MA57/MA86/MA97 use METIS for fill-reducing ordering. With the Archive
> (MA27-only) you can skip every METIS step below.

## 2. Get `libcoinhsl.dylib`

### 2a. Prebuilt binary (recommended — no compilation)

The STFC Archive downloads include **prebuilt Mac binaries**. On Apple Silicon grab
`CoinHSL-archive.v2023.11.17.aarch64-apple-darwin-libgfortran5.tar.gz` (the
`aarch64-apple-darwin` triplet = arm64 macOS; `libgfortran5` = its Fortran runtime dep). This
project's machine is arm64 with an arm64 casadi/IPOPT and `libgfortran.5.dylib` already in the
conda env, so it is compatible out of the box — **no Meson / gfortran / METIS build needed.**

```bash
mkdir -p ~/coinhsl && tar xf CoinHSL-archive.v2023.11.17.aarch64-apple-darwin-libgfortran5.tar.gz -C ~/coinhsl
ls ~/coinhsl/lib/                       # libcoinhsl.dylib (+ bundled deps)
otool -L ~/coinhsl/lib/libcoinhsl.dylib # verify every dependency resolves (see note below)
```

The BinaryBuilder tarball bundles its dependencies (OpenBLAS, libgfortran) with
`@rpath`/`@loader_path`, so siblings in the same `lib/` resolve automatically — usually
zero fixup. If `otool -L` shows an unresolved absolute path, add the env's lib to the rpath:
`install_name_tool -add_rpath "$CONDA_PREFIX/lib" ~/coinhsl/lib/libcoinhsl.dylib`.

Then skip to step 3 with `AEROVIZ_IPOPT_HSLLIB=~/coinhsl/lib/libcoinhsl.dylib`.

### 2b. Build from source (only if no prebuilt binary fits — CoinHSL 2023+ uses Meson)

```bash
conda activate aviation
# build deps into the env (gfortran + BLAS/LAPACK + METIS + meson/ninja)
conda install -c conda-forge meson ninja "gfortran>=13" openblas metis pkg-config

tar xf coinhsl-2023.11.17.tar.gz
cd coinhsl-2023.11.17

# METIS from conda-forge is METIS 5 (64-bit idx off); CoinHSL expects METIS 4/5 — point it at the env
meson setup builddir --buildtype=release \
  -Dlibblas=openblas -Dliblapack=openblas \
  -Dlibmetis_path="$CONDA_PREFIX/lib" -Dlibmetis_include="$CONDA_PREFIX/include"
ninja -C builddir
```

Result: `builddir/libcoinhsl.dylib`. Install it somewhere stable, e.g.:

```bash
mkdir -p "$CONDA_PREFIX/hsl" && cp builddir/libcoinhsl.dylib "$CONDA_PREFIX/hsl/"
# make its runtime deps (openblas, metis, gfortran) findable — they are already in the env:
install_name_tool -add_rpath "$CONDA_PREFIX/lib" "$CONDA_PREFIX/hsl/libcoinhsl.dylib"
```

**If METIS is painful — MA27-only fallback:** configure with `-Dlibmetis_path=` empty (skip
METIS); the build still produces MA27. Then use `AEROVIZ_IPOPT_LINSOL=ma27` below.

## 3. Point the optimizer at it (already wired — just set env vars)

`collocation/components.py` reads two environment variables (default = MUMPS, unchanged):

```bash
export AEROVIZ_IPOPT_LINSOL=ma57                       # or ma27
export AEROVIZ_IPOPT_HSLLIB="$CONDA_PREFIX/hsl/libcoinhsl.dylib"
```

Put these in the shell (or the backend launcher `start_aeroviz_fullstack.sh`) before running
any solve — the batch (`run_scenario_pipeline.py` / `scenario_optimization.py`) and the live
backend both go through `_make_nlp_solver`, so both pick it up. Unset them → back to MUMPS.

## 4. Verify it loaded and measure the speedup

```bash
AEROVIZ_IPOPT_LINSOL=ma57 AEROVIZ_IPOPT_HSLLIB="$CONDA_PREFIX/hsl/libcoinhsl.dylib" \
PYTHONPATH=4dTrajectory/optimization python - <<'PY'
import json, sys, time
sys.path.insert(0, ".")
from flight_scenarios import FlightScenario
sys.path.insert(0, "4dTrajectory/optimization")
import scenario_optimization as so
import casadi as ca

scen = next(FlightScenario.from_dict(d) for d in json.loads(open(
    "flight_scenarios/outputs/KRDU_combined_czml_input_threshold_scenarios.json").read())
    if d["source"]["id"] == "DAL1407")
# verbose solve prints the linear-solver banner — confirm it says MA57, not MUMPS
opt_states = so.optimize_scenario(scen)
t0 = time.time(); so.optimize_scenario(scen); print("MA57 solve %.2f s (was ~0.9 s on MUMPS)" % (time.time()-t0))
PY
```

To confirm the solver actually loaded (not silently falling back), run one solve with
`verbose=True` on the optimizer and read IPOPT's banner line: it should say
`running with linear solver MA57`. If the library failed to load you'll instead see an
IPOPT error / `nan` result — check the `libcoinhsl.dylib` rpath (step 2) with
`otool -L "$CONDA_PREFIX/hsl/libcoinhsl.dylib"` (every dependency must resolve).

## Notes

- **Which HSL solver:** for these ~800-var NLPs, **MA27** (single-threaded, low overhead) is
  often the fastest and needs no METIS; **MA57** is close and scales better. The parallel ones
  (MA86/MA97/Pardiso) only pay off on much larger problems — not worth it here.
- **Reproducibility:** MA27/MA57 are deterministic single-threaded; results match MUMPS to
  solver tolerance (the *solution* is unchanged — only the Newton-step factorization differs).
- **Portability:** the repo default stays MUMPS, so anyone without HSL can still build/run.
  These env vars are a per-machine opt-in.
- **macOS SIP + DYLD:** we use IPOPT's `hsllib` option (an absolute path) rather than
  `DYLD_LIBRARY_PATH`, so System Integrity Protection's env stripping is not an issue.
