# manoeuvre_codes_2026_09 — the INTENT-CODE second layer (two-tier v3 stage B, 2026-09-18 … 09-19)

The second layer as a small **discrete intent space**: a learned encoder + FSQ compressed 20–60 s
of the truth into one code `z`, the control-path executor held that tokenizer as a submodule and
flew the segment under the code, and a causal transformer over code sequences was the prior that
said the next code. Design: `docs/2026-09-18_manoeuvre_token_plan.zh.md` (the intent-token plan)
and `docs/2026-09-18_two_tier_plan_v3.zh.md` §5.2 **as it read before 2026-09-20**.

**Archived 2026-09-20.** Plan v3 §10's audit (written after the user stopped the stage B queue)
found the defect the readings were taken on top of: **both layers train on the truth and are only
ever evaluated closed-loop** — the executor never sees a history it flew itself, and the truth
codes are indexed by TIME, not by where the executor actually is. Stage B was rewritten the same
day (§5.2, §6.2) around an **instruction vocabulary**: absolute control instructions read back off
the track (heading target relative to the final approach course, height above the threshold, ground
speed, intercept), one position every τ seconds, with CAT-K-style closed-loop post-training for
both layers and goal-directed decoding. The codes are not part of that design, so their code is
archived rather than kept "just in case" — `CLAUDE.md`'s no-compatibility rule: a stored config
naming this layer is refused at load by name (`config.PLAN_CONDITIONINGS_RETIRED`), with a pointer
to this directory.

Off the import path on purpose (`tests/test_architecture.py`): nothing live imports it, and
nothing here is importable — there is no `__init__.py` on the way in.

## What moved (`git mv`, byte for byte)

| file here | was | what it was |
|---|---|---|
| `tokenizer.py` | `manoeuvre/tokenizer.py` | the learned encoder + FSQ, the 63-word command-vocabulary baseline, the hashed codebook artefact (C28) |
| `sequences.py` | `manoeuvre/sequences.py` | a flight as its `[code, state]` sequence from the anchor |
| `prior.py` | `manoeuvre/prior.py` | the causal transformer over codes (discrete and continuous), its loss, its decoding |
| `readout.py` | `manoeuvre/readout.py` | gate T's readout (the L1 gain, code usage, the paired no-token twin) and the code atlas |
| `plan_token.py` | `outputs/control/plan_token.py` | the manoeuvre-code token: `z`, `token_phase`, `token_span_start_s`, `training_plan_context`, the batch-size probe's context |
| `manoeuvre_codebook.py`, `manoeuvre_readout.py`, `manoeuvre_prior.py`, `manoeuvre_prior_readout.py`, `manoeuvre_gates.py`, `manoeuvre_code_atlas.py`, `two_tier_b_queue.py` | `experiments/` | the intent-code chain's runners and stage B's queue (R7, R9) |

`gates_manoeuvre.py` is **not** a move: `manoeuvre/gates.py` stayed live for `gate_grid`
(stage A1) and `gate_relative` (stage A3 / B), so gates **T / X / P / E / S** — their
pre-registered constants, `gate_x`, `gate_p_open_loop`, `gate_p_discrete_vs_continuous`,
`gate_e`, `gate_s`, `gate_t` — were **copied verbatim** into that file and deleted from the live
one. The three helpers it uses (`_two_seeds`, `_require_protocol`, `_cell`) are copies of the
live ones, so the file reads on its own.

Their tests are archived beside them, unmodified (`tests/`); they do not run.
`tests/test_manoeuvre_lockstep.py` and `tests/test_manoeuvre_gates.py` are the ORIGINALS of two
files that are partly still live: the live ones were rewritten to the stripped API and keep only
the no-token tests (protocol none, `--execute-s`, `from_remaining_path`, `from_row`, the budget,
`cohort_keys`) and the grid-gate tests.

## What stayed live, stripped

- `manoeuvre/lockstep.py` — the **no-token closed loop only**: protocol `none`, one round =
  `executed_step_s`, the budget rule, `from_remaining_path` / `from_row`, the reference verdicts.
  Protocols C / A / A-truth, the codebook, the prior, the held token, the landing machinery and
  every code column are gone; a no-token row has no code columns.
- `experiments/manoeuvre_lockstep.py` — the same runner without `--codebook`, `--protocol`,
  `--prior`, `--prior-landing-ends-flight`. Its payload schema is **`ts-manoeuvre-lockstep-v4`**
  (v3 minus the token / prior keys and `e_plan`); an older schema is refused by name, with no
  compatibility path.
- `manoeuvre/segments.py` (the segment-start frame — the rewritten stage B's instruction labeller
  reads a manoeuvre in the same frame), `manoeuvre/context.py` (the type vocabulary — the
  instruction prior's context), `manoeuvre/failure_modes.py`, `experiments/executor_*`,
  `two_tier_grid_queue.py`, `plan_cohort.py`, `frame_ablation.py`.
- `config.plan_conditioning` — the axis, with **`off` alone**. `manoeuvre-code` is in
  `PLAN_CONDITIONINGS_RETIRED` (refused at load by name, pointing here), and the token's own
  fields (`manoeuvre_tokenizer`, `manoeuvre_fsq_levels`, `manoeuvre_codebook`,
  `manoeuvre_token_s`, `manoeuvre_token_step_s`, `FSQ_MINIMUM_LEVELS`, `token_span_s` /
  `token_step_s` / `token_hold`) are gone, together with their run-naming abbreviations and the
  `control_recipe` identity's `manoeuvre` block. The rewritten plan's `instruction` value is NOT
  added here — it arrives with its own code.

## Where the numbers and the artefacts are

Nothing under `4dTrajectory/outputs/` was touched. The readings this layer produced stay where
they were written and stay citable **with their configuration** (open-loop-trained executors —
read them through plan v3 §10 item 1):

- `4dTrajectory/outputs/KRDU/experiments/two_tier_v3_b_20260919/` — stage B's campaign: the
  baselines, S20_K16 (gate B FAIL), S60h60_K16 (gate B1 FAIL), S60held_K16 (gate B1 and row B
  PASS on the truth-token reading) and its seed-1337 prior. Stopped 2026-09-19T23:24Z.
- `4dTrajectory/outputs/KRDU/experiments/manoeuvre_tok_20260918/` — the P1.4 token campaign
  (stopped the day it started; its executor baseline was unsettled).
- `4dTrajectory/outputs/codebooks/` — the exported codebook artefacts.
- `docs/2026-09-18_manoeuvre_token_results.zh.md` §9–§11 — the readouts themselves, and
  `docs/2026-09-18_two_tier_plan_v3.zh.md` §5.2 the evidence paragraph that quotes them.
- The published picker categories carrying the `manoeuvre_readout` / `manoeuvre_lockstep` record
  blocks stay published: `publish_ts_experiment_trajectories.VARIANT_RECORD_BLOCKS` keeps both
  names (the publisher cannot import torch, so `manoeuvre_readout` is a mirror of
  `readout.RECORDS_BLOCK` here).
