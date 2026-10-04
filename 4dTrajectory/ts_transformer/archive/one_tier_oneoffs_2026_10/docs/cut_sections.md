# Passages cut from live documents when this campaign was archived (2026-10-04)

Each block is the text as it stood at `dev-two-tier` `47ac23a4`, cut verbatim; the live document keeps a one-line pointer here.

## `4dTrajectory/ts_transformer/README.md` (the paragraph before "Data selection & flight identity")

```
The four `run_ts_*.py` scripts above (repo root, alongside this package) are the general
ones. The rest — kinematic-loss/overfit diagnostics for the state path, and the control
path's own drivers (`run_ts.py control_basis_oracle`, `run_ts.py control_capacity_ceiling`,
…) — are indexed with dates and one-line purposes in
```

## `4dTrajectory/ts_transformer/docs/reference/contracts.md` (C29, the users list)

```
Users: `experiments/support.checkpoint_manifests`, `anytime_curve.load_arm` (and through it
`latent_probe`, `chain_sensitivity`), `control_capacity_ceiling`, `clock_attribution`,
`predictability_report`, `control_basis_oracle` (teacher fit), `runway_hypotheses`.
```

## `4dTrajectory/ts_transformer/docs/reference/ENGINEERING_NOTES.md` (the corridor-bounded checkpoint note)

```
  (`batch_benchmark`, `experiments.overfit_diagnostic`, `experiments.predictability_report`) cannot run a
```

## Left as written (not edited)

- `outputs/control/forecast.py` docstring names `clock_attribution` as an example caller: live code, a comment only; it
  changes with that file's next real edit.
- `docs/code-health-followups.md` entry 6 cites `experiments/runway_intent_r0.py:180`: a verified finding with its line
  at the time; the file is now under `archive/one_tier_oneoffs_2026_10/experiments/`.
