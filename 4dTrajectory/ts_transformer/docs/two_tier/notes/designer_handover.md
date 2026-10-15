# Handover (two-tier model)

Written 2026-10-15 ~07:45 UTC by Claude (the session that designed and built the two-tier work since 2026-10-12) for
the agent that takes over. It replaces the designer's handover of 2026-10-09 (in git history). Paths are relative to
`4dTrajectory/ts_transformer/` unless they start with `4dTrajectory/`, `/` or name the repository root.

Read in this order: this note → `docs/two_tier/design/outline.md` §0, §3.2 (decision index), §4 (plan), §5 (rules) →
the §0 of `prior.md`, `post_training.md`, `multi_control.md` → the memory index
`~/.claude/projects/-home-supercomputing-studys-thesis/memory/MEMORY.md` (loaded automatically in this project; the
long state log is `eval-landing-branch.md`) → root `CLAUDE.md` ("Coding Conventions", "Code review").

## 1 The role now

One agent does it all: the design documents (`docs/two_tier/design/`), the code, the runs, the reports. The stage
implementers and the notes `notes/stage_{a,b,c,d}.md`, `notes/fronter.md` and the `requests_from_*` files are no longer
in use (history). Write code yourself; a review agent (`opus-code-reviewer`) reviews it. Next free numbers: **D210,
O27** (outline §3.2 keeps the index; update it with every new number).

## 2 The user's rules that cost the most when broken

- **Branch.** `two-tier2` is the only branch the user maintains: "two-tier2 维护这个就行 其他的都是临时分支。注意这很重要
  如果你没明白我的意思 就直接问我". Never move `dev-two-tier` or any other branch (the main checkout
  `/home/supercomputing/studys/thesis` is on `dev-two-tier` `ae486e66`: do not touch it). Work in the worktree
  `.claude/worktrees/eval-landing` (on `two-tier2`). A larger change goes on a temporary branch there, then
  `git update-ref refs/heads/two-tier2 <new> <old>` (fast-forward only), `git checkout two-tier2`, delete the
  temporary branch. **Push `two-tier2` after each finished, reviewed change without asking** (the user, 10-14); never
  force-push, never push another branch.
- **Frontend and backend** (`aeroviz-4d/`, `aeroviz_backend/`): no change without the user's permission; they are the
  last stage. The frontend-fixture rewrite: "先不动". Keep the `generate` helper signature in
  `tests/test_prior_free_generation.py`.
- **Data:** "旧的数据不要动" — never overwrite or delete an artefact or run directory without permission. Finished runs are
  sealed: `SHA256SUMS` + `chmod -R a-w`.
- **兼容 is forbidden** (root `CLAUDE.md`): no fallbacks, no schema branches; each exception is the user's.
- **Splits:** val may be read in labelling; a campaign's base reads val once (claimed by its readers); test days and
  KAUS stay sealed until the end.
- **Change only what is ordered.** Deleting a D-numbered element needs asking. A choice the rules already allow is
  yours: do it and report (no choiceless questions).
- **Decisions:** explain the context plainly in Chinese FIRST, then ask with AskUserQuestion, the recommended option
  first. Replies: Chinese, plain words, concise, no LaTeX.
- **Reports:** you write them yourself, concise, Chinese, in `docs/two_tier/readouts/` (no agent writes a report).
- **Git:** never `git add -A` (check `git diff --cached --stat`); never `pkill -f`; commit messages end with the
  Co-Authored-By and Claude-Session lines the harness gives.
- **Review** (root `CLAUDE.md` "Code review"): your code → `opus-code-reviewer`, scope = the diff; a small change
  (≤ ~30 lines in one place, tests only, one constant) needs no agent ("small change, no agent review").
- **Runs:** formal runs start from a detached run worktree at a committed `two-tier2` commit
  (`git worktree add --detach .claude/worktrees/run-<name> <commit>`, then symlink `4dTrajectory/outputs` and `data`
  to the main checkout's), inside `systemd-run --user --unit <name>`; watch with a Monitor (30 min each, re-arm). Beside
  a running campaign run only the changed modules' tests, few processes. **Never edit a shell script a running unit
  executes** (bash reads it by offset).
- **A new cross-validation runs on every change the user has decided** (labels and inputs together); never ask which.

## 3 State at 2026-10-15 07:45 UTC

**two-tier2 = `5f72d159`, pushed.** Recent decisions, all built, reviewed and documented: vocabulary D203 (v19 labels),
prior D204 (lateral input /100), D205 (stop: 15 epochs without a better select loss, at most 150), D206 (the prior reads
no landings), post-training D207 (the reward ignores the landing runway's direction), D208 (the loops keep no landings),
prior D63/D209 (a prior's checkpoint opens on the spec sha, day split, candidates, Δ and the selection rule only).

**B21, the prior: DONE** (`4dTrajectory/outputs/POOLED/prior/prior_b21_20261014`, read-only, from `run-b21` at
`f1f2a21a`). Configuration A; the folds landed 88–92 % at their held-out airports; **the base
`prior_b21_20261014/base/run` lands 97.2 % of the val days** (B19's base 93.0 %). Its val read is spent. Report:
`docs/two_tier/readouts/2026-10-15_b21_prior.zh.md`. This base is the start of every stage C and D run from now on.

**D200, the comparison of three branch-training methods: RUNNING** — unit `multi-compare-d200`, script
`4dTrajectory/outputs/POOLED/multi/compare_20261014.run/chain.sh fccc9d11` (from run worktree `run-d200` at
`fccc9d11`), log `chain.log` there. Inputs: B21's base, `v19_20261014`, `v24_20261014`, windows `windows_20261013`;
4 rounds × (500 real + 500 compressed windows), seed 2035; selection readout each round, **no val read**.
- The four profiles: done (01:28–01:46 UTC).
- Arm (a) `multi_compare_grid120_20261014` (grid every 120 s, credit to the event): done 06:16 UTC. Select readout,
  landed share of the commanded aircraft over the five airports: rounds 0–3 = 80.4, 80.5, 80.8, 80.8 %; lost
  separation ~17 % every round (the main failure; other failures ~2 %).
- Arm (b) `multi_compare_seg60_20261014` (grid every 60 s, credit to the next branch point): running since 06:16 UTC;
  round 0 took 84 min (3,474 groups; (a) 62 min, 2,031 groups) and read 80.5 % landed, 17.5 % lost separation; its
  end about 11:50 UTC.
- Arm (c) `multi_compare_majority_20261014` (D192): after (b). Expected end of the chain: about 16:00–17:00 UTC 10-15.
- Watch: `compare_20261014.run/watch.sh` (prints each step's "done", any STOP/Traceback, and the unit's end; use it as
  a Monitor command and re-arm every 30 min). Compare: `python3 compare_20261014.run/compare.py <arm dir> ...` (per
  round: landed, lost separation, other outcomes, windows with a loss between two commanded aircraft, per-airport).
- A resume after a failure: run the same `chain.sh fccc9d11` again in the same unit (each step skips what is done;
  `.aborted-<UTC>` handling is in the script). Do not edit `chain.sh` while it runs.

## 4 What to do next

1. Watch D200 to its end ("every arm done" in `chain.log`). Then report to the user in Chinese: one table of the three
   arms by round (`compare.py` on the three arm directories), what differs (landed, lost separation, cost per round),
   and a recommendation. **The user chooses one method for stages C and D** (multi_control D200). Record the choice as
   the user's in multi_control (D200 row, §0.3, §0.4) and outline §4; seal the three arm directories and the profiles.
2. Then the outline §4 plan: version 2 of the multi-aircraft control (multi_control D153, MC8–MC10), from the chosen
   method; the test days at the end (labelling check D198, KAUS in its three parts D158).
3. Frontend publication of B21 / D200 results: last, and only with the user's permission.

## 5 Waiting for the user (do not do these unasked)

- The frontend-fixture rewrite ("先不动"). The tests that fail only because of it: `test_prior_training_export`
  (fixtures), the post export's listed set and fixtures, the multi export's stage D set, backend `test_prior_segment`.
  In the last runs (the changed modules of D63/D209, 10-14) nothing else failed; the full suite last ran before B21.
- Removing old run worktrees (`run-v8`, `run-lateral`, `run-b15`, `run-b16`, `run-b19`, `run-b20`, `run-d-compare`,
  `trial-no-landings`, …); the branch `trial-no-landings` is kept. `run-b21` and `run-d200` are needed (B21's record,
  D200 running).
- Prior O25 (wind): a METAR download needs the user's permission.
- The user asked how much faster an A40 would be (answered: about 2–2.5× at FP32, not measured; TF32/BF16 or several
  folds at once would need code changes and the user's decision).

## 6 Where things are

- Design: `docs/two_tier/design/{outline,vocabulary,prior,post_training,multi_control,frontend}.md`; each §0 is the
  status. `superseded_decisions.md` keeps the replaced passages word for word for the user's review; it is generated
  by `~/.claude/projects/-home-supercomputing-studys-thesis/handover/superseded/write_index.py` from `superseded.json`
  there (154 records; add a record per rewritten passage, run the script from the `eval-landing` worktree root).
- Readouts (reports): `docs/two_tier/readouts/`; the last ones `2026-10-14_b19_prior.zh.md`,
  `2026-10-15_b21_prior.zh.md`.
- Runs: `4dTrajectory/outputs/POOLED/prior/` (B-campaigns; each `<run>.run/` holds launch/watch/summary scripts;
  B21's `fold_summary.py`, `summary.py`, `base_summary.py`), `4dTrajectory/outputs/POOLED/multi/` (stage D).
- Runners manual: `docs/reference/runners.md` (R56–R61 the prior, R60 `prior_campaign`, R65 `post_train`, D's
  `multi_train`/`multi_profile`).
- Tests: `conda activate aeroviz`; `python -m pytest 4dTrajectory/ts_transformer/tests/<file> -q
  --import-mode=importlib`; the full suite `-n 8 --dist worksteal` with `OMP_NUM_THREADS=1` (~3 min), only when no
  campaign runs.
- CHANGELOG: repo `docs/CHANGELOG.md`, newest first.
