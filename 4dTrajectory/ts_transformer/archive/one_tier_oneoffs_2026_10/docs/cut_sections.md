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

## `4dTrajectory/ts_transformer/CLAUDE.md` (the Runners paragraph and C8; second pass, the readouts)

```
`eta_error_readout` — B0; vectored and straight-in errors differ several-fold,
so one pooled ETA interval cannot serve both (R4). `latent_probe` — L2.f;
`--limit N` is a prefix (a smoke test); the posterior reads the future (R5). `latent_fan_readout`
— 4(a); the RANDOM fan is the reading, not a footnote (R6).
```

```
`mean_displacement_sigma`, `active_units_0p05` (`run_ts.py latent_readout` / `latent_probe`);
```

## `4dTrajectory/ts_transformer/docs/reference/contracts.md` (C8 and C29, second pass)

```
a checkpoint with `run_ts.py latent_probe`; the gate sentence is
```

```
`latent_probe`, `chain_sensitivity`),
```

## Left as written (not edited)

- `outputs/control/forecast.py` docstring names `clock_attribution` as an example caller: live code, a comment only; it
  changes with that file's next real edit.
- `docs/code-health-followups.md` entry 6 cites `experiments/runway_intent_r0.py:180`: a verified finding with its line
  at the time; the file is now under `archive/one_tier_oneoffs_2026_10/experiments/`.
- Comments naming the archived readouts: `experiments/anytime_curve.py` (the instrument name its loader speaks in),
  `outputs/control/latent.py` (which readouts print the displacement), `cli/predict.py` (the message that points at
  `run_ts.py eta_calibration`, which is NOT archived), and the arms files' `_comment` fields in `docs/experiments/`
  (`l2f_mean_information_arms.json` names `latent_probe`, `b1b_two_head_arms.json` names `eta_error_readout`).
