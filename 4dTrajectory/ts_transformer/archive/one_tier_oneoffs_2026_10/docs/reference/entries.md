# Runner entries cut from `docs/reference/runners.md` (2026-10-04)

Verbatim, as at `dev-two-tier-v4` `4efbf5d3`.

### R4 · `run_ts.py eta_error_readout` — B0

- `run_ts.py eta_error_readout` — **B0**: |`final_time_error_s`| p50/p80/p90 and the SIGNED
  p10/p50/p90 (same for `fde_m`) per stratum, straight out of existing `summary.json` files.
  A row is used only if it carries every metric AND every `STRATA_COVARIATES` field — a
  present-but-null `established_at_anchor` would otherwise read as False and change stratum.
  Measured 2026-09-07 on KRDU val: vectored |Δt| p80 **65.8–72.5 s** against straight-in
  **12.0–20.3 s**, so one pooled ETA interval cannot serve both strata.


### R5 · `run_ts.py latent_probe` — L2.f

**The latent line's own runner** (`docs/history/2026-09_latent_anytime/2026-09-07_latent_intent_design.zh.md` §六 L2.f):

- `run_ts.py latent_probe` — **L2.f**: the training-side densities of one or more latent
  checkpoints on a split (`--checkpoint LABEL=PATH`, repeatable), through the SAME cohort
  rebuild as A0 (`load_arm` / `cohort_series`, so the roster rule has one owner): prior and
  posterior per-dimension spread, the posterior mean's displacement in prior sigmas, the
  per-dimension KL and its mean/variance split, the prior's total std against N(0, I).
  Refuses a non-latent checkpoint (no posterior) and, through the shared loader, a
  `cta=given` / `intent=truth-…` one (that loader takes an `instrument` name so each runner
  refuses in its own voice). `--limit N` is a PREFIX of the split, not a sample — the
  displacement median moved 25 % between 100 and 200 KRDU flights, so a limited table is a
  smoke test and the artifact says so. **The posterior reads the future: never a prediction
  result.**


### R6 · `run_ts.py latent_fan_readout` — 4(a)

- `run_ts.py latent_fan_readout` — **4(a)**: the sample fan read by the B line's gate 3.4-3
  protocol, so the latent fan and the quantile fan are one deliverable measured one way. The
  chamfer-to-nearest-leaf logic has ONE implementation (`experiments.quantile_fan_readout.leaf_rows`
  / `leaf_geometry` / `geometry_cell`, imported); this runner supplies the leaves. `--arm
  <pred_dir>` of a `predict --latent-samples K --latent-random K` run, refused without its
  `modes/` leaves. Per stratum: the truth's chamfer to the top-1 decode, to the nearest of the
  `modes/` leaves and to the nearest of the `random/` leaves, each with the share of flights
  the nearest leaf beats top-1 on; minADE_K and top-1 ADE off the same records (the SAME
  definition as `run_ts.py latent_readout`, so the two artifacts cross-check); and the fan's
  lateral spread (p50 of the widest pairwise endpoint gap). **The random fan is the reading,
  not a footnote** — a fan always contains something nearer the truth than its own mean, so a
  nearest-leaf number alone measures nothing; the prior fan is informative only where it beats
  the N(0, I) control. KRDU val, 1404 flights, both L2.g seeds and L2.z: prior 124 m at
  0.92–0.93 against random 166–201 m at 0.34 (2026-09-08).

