# instruction_vocabulary_2026_09 — the INSTRUCTION VOCABULARY second layer (two-tier v3 stage B, 2026-09-20 … 09-22)

The second layer as controller-like WORDS read back off the track: a labeller turned each arrival
into an event sequence of heading / vertical / speed / runway / duration / terminal words
(`segment-v14`, then the box vocabulary `box-v3`), a causal prior predicted the next word, the
control-path executor was conditioned on the words in force over its segment
(`plan_conditioning='instruction'`), and the closed loop could hand it the truth's words by flown
position (protocol `truth-instruction`). The B′ queue was to train and gate both layers.

**Archived 2026-09-23, on the user's word** ("对于词表设计相关的，把当前所有代码，都archive；我要重新写
一版"): the vocabulary is being redesigned from scratch, and the live tree is to read as if this
design had never been there. No instruction executor was ever trained (no stored config carries
`plan_conditioning='instruction'` or `instruction_vocabulary`, checked over all 392 `history.json`
under `4dTrajectory/outputs/`), and no lockstep payload of the schema this layer introduced
(`ts-manoeuvre-lockstep-v5`) exists on disk, so nothing stored stops loading.

Off the import path on purpose (`tests/test_architecture.py`): nothing live imports it, and
nothing here is importable — there is no `__init__.py` on the way in.

## What moved (`git mv`, byte for byte)

| here | was | what it was |
|---|---|---|
| `manoeuvre/instructions.py` | `manoeuvre/instructions.py` | the vocabulary spec + sha, the labeller (`read_instructions`, `segment-v14`), the course frame, `Reading`, the artefact's write / load |
| `manoeuvre/instruction_kinematics.py` | same | flying a sentence back (the replay gate's kinematics) |
| `manoeuvre/box_vocabulary.py` | same | the box vocabulary (`box-v3`) the frontend's Training view reads |
| `manoeuvre/instruction_sequences.py`, `manoeuvre/instruction_prior.py` | same | the sentence with state tokens, and the causal prior over it |
| `manoeuvre/context.py` | same | the prior's context tokens (type vocabulary, runway token) — kept live at the intent-code archive for this layer |
| `manoeuvre/segments.py` | same | the segment-start frame — kept live at the intent-code archive "for the instruction labeller"; nothing live read it any more |
| `control/instruction_token.py` | `outputs/control/instruction_token.py` | the executor's token (words in force at the Δ / τ positions), `nearest_truth_time_s` |
| `experiments/instruction_vocabulary.py`, `instruction_replay.py`, `box_vocabulary.py`, `instruction_sample_export.py`, `instruction_prior.py`, `two_tier_bprime_queue.py` | `experiments/` | the vocabulary chain's runners and the stage B′ queue |
| `experiments/approach_events.py` | `experiments/` | B0′′: runway changes and go-arounds, measured to decide two words |
| `experiments/measure_instruction_distribution.py`, `summarise_instruction_distribution.py`, `turn_change_distribution.py`, `absolute_heading_distribution.py` | `experiments/` | the 2026-09-22 signal-distribution measurements for the next vocabulary |
| `tests/*.py` | `tests/` | their tests, unmodified; they do not run |
| `docs/*.md` | `docs/` | the vocabulary's own documents (design, plan, method, rebuild notes, prior results, distribution report, the English specification) and plan v3's stage B document `2026-09-18_two_tier_plan_v3_B.zh.md` |
| `docs/assets/instruction_distribution_20260922/` | `docs/assets/` | the distribution report's figures and tables |
| `docs/experiments/two_tier_v3_bprime_arms.json` | `docs/experiments/` | the B′ campaign's arm declaration |

**Never committed anywhere else**: `experiments/final_vertical_check.py` — the D61 vertical check
(where a sentence's last event sits against the published path), left untracked in the
`dev-freegen-readout` worktree and copied here byte for byte on 2026-09-23 when that worktree was
removed. That branch's nine commits (the prior's free generation and the D61 landing gate, based
on 09-20 code) were never merged; the tag `archive/freegen-readout` (`208ab00f`) keeps them.

**Cut verbatim out of live files** (not a `git mv`; the text is unchanged):

- `docs/reference/entries.md` — contracts C30 and C29 (`docs/reference/contracts.md`) and the
  measurements H5–H12 (`docs/reference/defaults.md`).
- `docs/cut_sections.md` — every passage cut out of the mixed live documents (the ts `CLAUDE.md`
  index lines, plan v3's overview and results documents, the stage A notes, the repo's
  `docs/open-items.md` / `docs/code-health-followups.md`, the harvest reference), each under the file
  and place it came from.
- `docs/experiments/intents_entries.json` — the two campaigns `two_tier_v3_bprime_20260920` /
  `two_tier_v3_bprime_20260921` from `docs/experiments/intents.json` (nothing of them was
  published to the picker).

## What stayed live, stripped

- `config.py` — `plan_conditioning` is `off` alone again; the `instruction` value and the
  `instruction_vocabulary` field are gone (no stored config carries either).
- `outputs/control/{heads,strategy,forecast}.py`, `run_naming.py`,
  `experiments/predictability_report.py`, `experiments/support.py`,
  `experiments/executor_relative_gate.py` — back to their content before the layer (commit
  `79f871f1`), plus a comment in `heads.py` and the docstrings of `predictability_report.py` and
  `executor_relative_gate.py` that still pointed at the vocabulary or described the pre-D57 row B1.
- `outputs/base.py`, `training/train.py` — the `checkpoint_payload_extras` /
  `verify_checkpoint_payload` hooks are gone (the vocabulary sha was their only use).
- `manoeuvre/lockstep.py`, `experiments/manoeuvre_lockstep.py` — protocol `none` only again,
  payload schema `ts-manoeuvre-lockstep-v4`.
- `manoeuvre/gates.py` — row B1 stays with its stricter rule: it is the upper bound for any second
  layer fed the truth. The rule was decided as D57 in the stage B document archived here; it answers
  plan v3 §10 audit item 5 (a gate rule, not a vocabulary rule), which is what the live code cites.
- `outputs/guidance/controller.py` — `CLIMB_MAX_RAD` back to 2.0° (raised to 4.0° in `81d3d5ac`
  only so the vocabulary's climb mode could be flown). Its only live reader, `PlanGuidance`, has had
  no live consumer since 2026-09-18, so no guidance result was produced at 4.0°. The vocabulary's own
  replays (`two_tier_v3_bprime_20260921/replay_*`, 09-21) WERE flown at 4.0°:
  `manoeuvre/instruction_kinematics.py` here imports the constant from the live controller and would
  read 2.0° now.
- `data/approach_difficulty.course_frame_rows` stays: it is the one course-frame definition the
  anchor covariates and the failure modes share.
- The harvest's published minima (`runway_thresholds.json` schema v3) stay; only the test pinning
  them to "altitude word 0" went.
- The frontend's Training view (`aeroviz-4d`, which reads `instruction_sample_export`'s output) was
  NOT touched (the user's choice, 2026-09-23); it has no producer in the live tree now.

## On disk (untouched, not in git)

`4dTrajectory/outputs/{KRDU,POOLED}/experiments/two_tier_v3_bprime_2026092{0,1}/` hold the
vocabulary artefacts, replays and the two priors; `aeroviz-4d/public/data/airports/*/training/` the
exported Training samples. Only `vocabulary_v15_nomerge_noposition` still loaded under the code
archived here — every `segment-v14` artefact (including `2b8bf25c2a36`) had been refused since
`b067ca95` added two spec fields without bumping `READING_RULE`.

## Defects found in this code before it was archived (review, 2026-09-23)

Recorded for the rewrite; none was fixed here.

1. `two_tier_bprime_queue` imported `repo_layout.TS_SCRIPT as RUN_TS` — `__main__.py`, not
   `run_ts.py` — so every step would have failed with "invalid choice".
2. Adding spec fields (`use_established_word`, `min_instruction_s`) without bumping
   `READING_RULE` moved every sha; `Vocabulary.from_dict` filled missing keys with defaults, so
   the refusal read as a corrupt file rather than "another rule".
3. The replay gate flew the artefact's `sentences_*.json`, but the executor (training, `predict`,
   the closed loop) and the prior re-read the words from the tracks at run time — the gate did not
   guard what the models consumed.
4. `nearest_truth_time_s` matched the flown position to ANY truth row: on stored no-token flown
   paths (L120_D20, two seeds) the matched truth time leapt > 80 s ahead (p50 ≈ 145 s) between
   rounds in 5.1 % / 8.5 % of flights and went back in ~2 %.
5. The prior's runway head could subtract its answer: runway-relative heading words beside an
   absolute state course give the landed runway's course (104 / 104 KRDU val flights at event 0).
6. `box_vocabulary.heading_edges` split the reciprocal (±180°) into two words — 24.5 % of the
   five-airport train flights carried fake 0↔64 events; `pick` used a different half-width than
   `contains` for negative heights.
7. `_fit_residual_m` read an uninitialised array when float rounding left the last point outside
   the last span (~1 % of breakpoint sets); `tile_segments`' departure pass re-created segments
   shorter than `min_instruction_s`; `instruction_replay`'s "vectored" stratum differed from
   `strata_masks`.
