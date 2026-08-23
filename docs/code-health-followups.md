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

---

Opened 2026-08-23, during the `evaluation` review that accompanied the threshold speed
gate (schema v6).

## 7. `evaluation.records.roster_context_keys` leaks a raw FileNotFoundError

**Verified** (reproduced): `python -m evaluation --input <dir-without-summary.json>`
raises `FileNotFoundError: .../summary.json` with a bare traceback. The crafted message
for exactly this case lives in `record_files` ("has no summary.json manifest; pass a
record file for a loose record"), but the CLI calls `contexts_from_roster` →
`roster_context_keys` FIRST, and that function reads `p / "summary.json"` without an
existence check (`records.py:227`); `_load_json` only converts `ValueError`.

**Suggested:** in `roster_context_keys`, return `None` (the documented "roster cannot
name them" outcome) when `summary.json` does not exist — the CLI then falls through to
`record_files`, which raises the intended message. Error-path quality only; no verdict
can change.

## 8. `record_from_dict` does not enforce strictly-increasing `t`

**Verified**: a record whose `states` carry `t = [0, 10, 5]` and `final_time_s = 5.0`
passes `record_from_dict`. The contract says `t` is "the one field with hard contracts —
`final_time_s == states[-1]['t']` to 1e-6, and strictly increasing offsets"
(`evaluation/CLAUDE.md`, `evaluation_export.py`), but only the first half is validated
at the boundary; producers enforce ordering on the write side only. A non-monotonic
record would corrupt `flight_time_delta_s` and the interpolated crossing's `t` silently.

**Judgement:** add `t[i] > t[i-1]` to the `_state` loop in `record_from_dict` (one pass,
already iterating). Cheap, and it makes the documented invariant structural on the read
side too.

## 9. `evaluation/arrival.py` restates `STATE_KEYS` as a literal tuple

**Verified**: the crossing-interpolation dict comprehension iterates
`("t", "lat", "lon", "alt", "V", "psi", "gamma", "m")` (`arrival.py`,
`_computed_arrival`) — a mirror of `records.STATE_KEYS` restated without a mirror
comment. The project rule is "a schema literal in a consumer is a mirror — import it"
(`CLAUDE.md`). Adding a state key would silently skip it in interpolated crossings.

**Suggested:** `("t", *STATE_KEYS)` imported from `evaluation.records`.

## 10. `evaluate_batch` on an EMPTY iterable reports `subject: "mixed"`

**Verified**: with zero records, `sorted(subjects)[0] if len(subjects) == 1 else
"mixed"` takes the else branch, so an empty batch serializes `subject: "mixed"`,
`solve_rate: 0.0`. Not reachable through the CLI (`record_files` raises on an empty
roster), only via direct library calls.

**Judgement:** label it `"empty"` or raise; cosmetic until someone streams a filtered
generator that comes up empty and reads "mixed" as two subjects having been present.

## 11. Observed rows publish last-sample kinematics under event-flavoured names

**Judgement** (mechanism verified, impact assessed): for observed records the
deviation's `speed_ms`, `heading_rad`, and `final_time_s` are LAST-SAMPLE quantities
(median 325 m before the threshold), while `cross_track_m`/`vertical_m` in the same row
are event-based estimates AT the threshold. `_row` publishes all five flat with no
distinction, so a consumer can read `speed_ms`/`final_time_s` as crossing quantities.
The v6 speed gate sidesteps this (observed `crossing_speed_ms` is `None`), but the
pre-existing three fields keep the ambiguity.

**Suggested:** either rename on the row (`final_sample_*`) for observed subjects or
document the split in `methodology.event`; renaming touches report consumers, so it
should ride the next schema bump.

## 12. `_reference_aggregate` reports an unweighted mean of per-flight means

**Judgement**: `path_lateral_m.mean` in the batch reference aggregate is
`fmean(per-flight means)` — every flight weighs equally regardless of its resample
count (fixed at 101, so today the two definitions coincide; the p95 is dropped
entirely at the aggregate level). Worth a one-line comment stating the weighting so a
future variable-N resample does not silently change the metric's meaning.

## 13. Eleven `test_ts_pipeline.py` reuse-guard tests fail at HEAD: fixtures predate the roster requirement

**Verified** (2026-08-24, while running the fleet reclassify). `TrainingPlan.cv_reuse_error`
and `checkpoint_reuse_error` in `run_ts_pipeline.py` require every airport's
`lateral_pass_eligibility.json` beside its arrival manifest, but the hermetic tests
(`monkeypatch HARVEST_ROOT` → tmp dir) build their harvest fixture with `_manifest(...)`
only — no roster — so 11 tests fail with "one or more lateral-pass eligibility rosters
are missing" on a clean tree, independent of disk state. (Three OTHER tests in the same
file read the REAL harvest root and were failing for the opposite reason — the rosters
really were missing after the 2026-08-21 re-roster; rebuilding them via
`ensure_lateral_pass_roster` fixed exactly those three, confirming the split.) A few of
the 11 fail on assertion text that has drifted for other reasons (candidate counts, loss
line format), so this is fixture rot, not one missing file.

**Suggested:** give `_manifest` a companion that also writes a minimal roster (or have
the fixture call `ensure_lateral_pass_roster` against a stub approach report), and
re-derive the drifted assertion strings. Owned by whoever is actively working the ts
pipeline; not fixed here to avoid colliding with in-flight changes.
