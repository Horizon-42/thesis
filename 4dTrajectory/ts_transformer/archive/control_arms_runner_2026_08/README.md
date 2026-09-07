# Archived: `run_ts_control_arms.py`, the first arm-campaign runner (2026-08-16 → 2026-09-07)

A **superseded** runner, kept in the repository as the record behind the campaigns it drove
and taken off the import path. Nothing here runs against the current package.

## What it was

The train → predict → evaluate → publish driver for a set of control arms declared in a
JSON file (`{"base_recipe": …, "arms": [{"key", "label", "overrides"}]}`). It drove the
bank-wiggle investigation (`docs/2026-08-19_control_bank_wiggle_diagnosis.zh.md`) and the
KSJC imitation-weight ladder (`docs/experiments/imitation_ksjc_v5_*.json`,
`docs/experiments/RESUME_ksjc_v5_ladder.md`).

## Why it is archived

`docs/2026-09-07_package_audit_plan.zh.md` T4-27 scheduled it: `run_ts_frame_ablation.py`
does the same job, its `main` having grown from this one, and it has since gained everything
this lacked — resume (it skips steps whose output already exists, which is why the KSJC
ladder needed a hand-written RESUME document), a free-disk pre-check, per-arm run names and
slugs, predict-only arms, and `--campaign-id` / `--experiment-id` provenance.

Two behaviours made keeping it live a hazard rather than a convenience, and both were found
by using it on 2026-09-07:

1. **it reads only `base_recipe` and silently ignores an arm file's `base` block.** Every
   arm declaration written since 2026-09-03 carries one (`a0_random_arms.json`,
   `l1b_full_arms.json`, `l2*_arms.json`, `closure_p1c_arms.json`, …), so pointing this
   runner at one of them would have trained the bare recipe under the arm's name — a
   silently wrong campaign, not a failure;
2. **its `--dry-run` wrote `config.json` for every arm.** Aimed at
   `4dTrajectory/outputs`, which is a read-only symlink in a development worktree, a dry
   run left a campaign directory behind in the shared tree. `run_ts_frame_ablation.arm_config`
   now takes `write=not --dry-run` and constructs the config either way, so an unrunnable
   arm still fails during the dry run.

Use `run_ts_frame_ablation.py` for any new campaign; it accepts `base_recipe`, `base` or
both (`base_recipe` first, `base` layered on top).
