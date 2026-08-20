# Code-health follow-ups (deferred)

Findings noticed while working elsewhere, recorded rather than fixed on the spot so the
change that surfaced them stays reviewable. Nothing here is a live bug unless it says so.

Each entry states what was **verified** versus what is **judgement**, so a later reader can
tell how much re-checking it needs. Delete an entry when it is fixed or dismissed.

Opened 2026-08-17, during the `final_approach` / `evaluation` design pass.

---

## 1. `harvest/classify.py` and `harvest/store.py` define a byte-identical `_iso`

**Verified.** Both are:

```python
def _iso(time_s: float) -> str:
    return datetime.fromtimestamp(time_s, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
```

`classify.py:268` and `store.py:287`, same package. `store.py` additionally has
`_iso_precise` (millisecond `isoformat`). Timestamp formatting is identity-bearing in this
project — `landing_time_utc` is part of `flight_key` — so two copies of the format string
is exactly the kind of duplication that produced the `INBOUND_TOLERANCE_M` drift risk.

**Suggested:** one `_iso` / `_iso_precise` pair in `store.py`, imported by `classify.py`.
Low risk, no behaviour change. Check first whether the second-precision truncation is
deliberate in one of the two call sites before unifying.

## 2. `evaluation_export.summary_row` writes explicit JSON nulls

**Verified** (`4dTrajectory/optimization/evaluation_export.py:152` and the surrounding
block). Every field is built as `source.get(name)`, so a scenario that does not carry
`target_source`, `callsign`, `icao24`, … produces a roster row with the key **present and
null** rather than absent.

Why it is worth recording: `evaluation.arrival` was just fixed for precisely this Python
hazard — `dict.get(key, DEFAULT)` returns `None`, not `DEFAULT`, for a key present with a
null value, so an explicit null silently took a "this record aims elsewhere" branch. The
roster row is a different payload and no consumer currently keys a decision off its
`target_source`, so this is **latent, not live**. But the two shapes travel together, and
the next consumer to read the roster inherits the trap.

**Suggested:** decide per field whether it is required (index it) or optional (omit the key
when absent). Audited on the shipped artifacts: all 144,764 `*_eval.json` records carry
`target_source == "runway_threshold"` explicitly, so nothing on disk is affected today.

## 3. `harvest/observed.py::source_event_availability` re-validates its own manifest

**Judgement, not a defect.** The function interleaves deriving the availability block with
per-row `isinstance` validation of `tracks/manifest.json` — record shape, outcome
vocabulary, `event_status` presence, and every `source_integrity.excluded` entry.

This is the same shape as `evaluation.metrics._validated_observed_availability`, which was
deleted in the 2026-08-17 pass. The difference that matters: this one reads a **file**, so
validating is legitimate — a manifest can be stale or hand-edited. What is worth revisiting
is that the validation is spread through a derivation rather than done once at the read
boundary, so the manifest contract has no single place a reader can find it.

**Suggested:** if `tracks/manifest.json` gains a typed loader, move these checks into it and
leave this function doing arithmetic only. Not worth doing on its own.

## 4. Stale schema fixture in the comparison-CZML tests

**Verified.** `aeroviz-4d/python/tests/test_build_scenario_comparison_czml.py:539` builds a
report fixture pinned to `terminal-approach-evaluation-v2` — two schema generations behind
the current `v5`. It does not fail because `build_scenario_comparison_czml` copies
`schema_version` through into the comparison index without validating it, which is the
correct behaviour for a pass-through.

So this is a **documentation defect**: the fixture asserts a contract no producer has
written since v2, and a reader checking "what does the builder expect?" gets a wrong answer.
The same class of staleness is what let the v4 → v5 bump ship green past the frontend suite.

**Suggested:** either regenerate the fixture from the current producer, or add a one-line
comment saying the version is deliberately arbitrary because the builder does not read it.

## 5. `trajectory_data_process/tests` has 12 pre-existing failures

**Verified pre-existing** (reproduced with the working tree stashed, twice, on 2026-08-17).
Not a code-health item — recorded so nobody re-diagnoses it as fallout from an unrelated
change.

- `test_ts_pipeline.py` — 11 failures, all downstream of the same cause:
  `TrainingPlan.cv_reuse_error()` returns `"one or more lateral-pass eligibility rosters are
  missing"` where the tests expect `None`. The fixtures do not create the eligibility
  rosters that `run_ts_pipeline` now requires.
- `test_download_landings.py::test_download_reuses_interrupted_checkpoint_start_for_cache_keys` — 1.

Separately, `4dTrajectory/optimization/collocation/tests/test_optimizer.py::
test_fixed_time_objective_weights_control_effort_at_one` tracks the numpy 2.x scalar-
conversion change and is already documented in `CLAUDE.md`.

**Suggested:** fix the fixtures (build the roster, or let the plan treat a missing roster as
"no eligibility filter") so the suite is green and a real regression is visible again. A
suite with 13 known-red tests cannot report a 14th.

**Checked 2026-08-17:** rebuilding the five real `lateral_pass_eligibility.json` rosters did
NOT fix these — the tests build their own `tmp_path` trees, so the cause is in the fixtures,
not in missing production artifacts. Still 12 failed / 30 passed afterwards.

## 6. `write_arrival_records` clears its output directory, taking co-located artifacts

**Verified.** `harvest/arrivals.py` starts with `_clear(root)` on `arrivals/`, and
`lateral_pass_eligibility.json` lives in that same directory. So any `--evaluate-only` or
`--reclassify-existing` run silently deletes the eligibility roster, which the ts pipeline
then reports as "one or more lateral-pass eligibility rosters are missing" — a message that
points at the roster rather than at what removed it.

Observed directly: a KRDU `--evaluate-only` run left KRDU as the only airport of five
without a roster.

**Suggested:** either write the roster somewhere the arrivals rebuild does not own, or have
the clear step preserve known derived siblings, or have the harvest rebuild the roster after
writing arrivals. Low risk either way; the roster is cheap to regenerate
(`ensure_lateral_pass_roster`).

## `run_all_tests.sh` has 13 pre-existing failures, and they hide new ones

**Verified** (2026-08-20, clean `git worktree` at `84725d6`, i.e. the commit this session
started from — so none of them come from the `simple-v3` work):

- `trajectory_data_process/tests/test_ts_pipeline.py` — **11 failures**. The runner's printed
  summary no longer matches what the tests assert: they expect lines like
  `loss      : final_time=1, kinematic=3, terminal=0.02` and a `config    : TSConfig defaults`
  block, while the runner now prints a different banner. Also affected: the CV default grid,
  fixed/random anchor artifact paths, and five `--skip-train` checkpoint-rejection cases.
  Most likely stale since `dd6191d` / `13d14a6` changed the config and CLI surface — the
  tests were not updated with them.
- `trajectory_data_process/tests/test_download_landings.py::test_download_reuses_interrupted_checkpoint_start_for_cache_keys`
  — 1 failure, not investigated.
- `4dTrajectory/optimization/collocation/tests/test_optimizer.py::test_fixed_time_objective_weights_control_effort_at_one`
  — 1 failure, already documented in `4dTrajectory/CLAUDE.md` as a numpy 2.x regression.

**Judgement**: the eleven are assertion drift rather than broken behaviour — the pipeline
itself runs — but that is inferred from the assertion text, not from exercising the runner.

**Why this matters more than the count suggests**: `run_all_tests.sh` exits 1 either way, so
its exit code currently carries no information, and a *new* failure in the modeling suite is
invisible unless someone diffs the failure list by hand. The per-subsystem suites are still
clean (`ts_transformer` 366 passed, `aeroviz-4d/python` 154 passed), so the working practice
is to run those directly and treat the aggregate script as advisory until this is fixed.
