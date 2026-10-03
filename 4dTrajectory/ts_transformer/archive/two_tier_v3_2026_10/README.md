# two_tier_v3_2026_10 — the code of the instruction-v3 vocabulary that stage A of v4 does not rewrite

Archived 2026-10-03, at the start of the next two-tier version (`docs/two_tier/two_tier_design.md` §14.2 A0, the user's
decision D20: a new version on its own branch `dev-two-tier-v4`; artefacts `v1`–`v6`, the executor specs and every prior
are superseded and are not kept readable). The last commit before the move is `dev-two-tier` `a9dc3c00`. Every file here
is moved unmodified (`git mv`), with its tests. Off the import path on purpose (`tests/test_architecture.py`): nothing
live imports it, and its tests do not run.

Stage A rewrites the vocabulary, the labeller, the artefact, the executor, the judge and the replay in place; everything
below reads the old format (six columns, absolute heading classes, the approach column, "descend to land", the executor's
capture and glidepath laws) and comes back, rewritten, in the stage named.

| moved | what it was | comes back in |
|---|---|---|
| `prior/` (whole package) | the prior: data, scenes, model, training, the speaker, the procedure masks, augmentation, the window speaker | stage B (§14.3), rewritten to §6 |
| `experiments/prior_*.py` | R15–R17, R19–R22, R35, R38 and the prior's Training exports | stages B and C |
| `experiments/traffic_*.py` | multi-aircraft M0–M4 and step 7–9 runners and their shared loop modules (`traffic_loop`, `traffic_rounds`, `traffic_scene_data`, `traffic_speaking`, `traffic_window`, `traffic_window_augment`, `traffic_window_tuner`) | stage C (§14.4) |
| `experiments/window_training_export.py` | R36 | stage D |
| `experiments/code_version.py` (moved in a second commit) | a window readout's code version, read only by the archived R34 / R41 / R44 | stage C |
| `inference/separation_masks.py`, `inference/scene_edges.py` | the loop's separation masks; the scene edge features | stage C (O6; §9.2 #6) |
| `experiments/instruction_training_export.py`, `experiments/executor_training_export.py`, `experiments/training_attitude.py`, `instructions/training_files.py`, `instructions/display.py` | the frontend's Training exports (R11, R13) and their shared files | stage D (§14.5) |
| `experiments/go_around_census.py` (R40) | the go-around census | its detection rule moves into `instructions/labeller/go_around.py` (A2); the census itself is not rewritten |
| `experiments/instruction_word_frames.py` (R48), `experiments/instruction_final_approach.py` (R49) | readouts of the old artefacts for the design's §11 evidence | not planned (their outputs stay in `4dTrajectory/outputs/POOLED/analyses/`) |
| `experiments/heading_lead_ablation.py` (R23) | the heading-lead ablation on executor v11 | not planned |
| `experiments/executor_sensitivity.py` (R12's middle runner; moved in A4–A5) | one executor v11 parameter moved at a time on train | not planned (its test, in `tests/test_autopilot.py`, went with the old executor's tests) |

Their tests are under `tests/`, unmodified. `tests/test_autopilot.py` is the old executor's test file as it was (moved in A4–A5: the executor's laws, judge and replay were rewritten in place, and the live `tests/test_autopilot.py` is new; the tests that still applied were carried over, adapted). `tests/test_mirrors_cut_from_live_tests.py` holds test functions cut
verbatim from live test files because their other side moved here: the executor's glidepath edge against
`prior.procedure` (the edge itself goes in A4), the publisher's generation-record names against R35 / R38, and the prior's
layout rules from `tests/test_architecture.py` (stage B puts them back with the prior).

**Not runnable until stage D.** The backend's live executor (`aeroviz_backend/autopilot_segment/`) imports `prior/`,
`instructions/training_files.py`, `instructions/display.py` and `experiments/training_attitude.py`; it and its tests
(`aeroviz_backend/tests/test_single_executor.py`, `test_autopilot_segment.py`, and `test_http_server.py` where it calls
that route) fail on this branch until stage D rewrites them. The branch is not merged before then (§14.1 rule 11).

**Documents.** The runner manual entries of the moved runners (`docs/reference/runners.md` R11, R13, R15–R17, R19–R30, R34–R41, R43–R45,
R48, R49; R31–R33 were archived before, R42 `executor_conformance` stays live) and the ts `CLAUDE.md` index lines that name them still describe the old code; they are cut or rewritten when
the stage that brings each part back lands. The published results these runners made stay where they are.

**Live, with archived users only until stage C/D:** `inference/separation.py` (the separation judge, kept for §8 and
tested by `tests/test_separation.py`), `tests/support.py` `signal_attitudes` (the Training exports' fixture). The
executor's `GLIDEPATH_BELOW_M` mirror lost its pin with the cut test; the constant itself goes in A4 (D9).
`tests/test_mirrors_cut_from_live_tests.py` is a verbatim cut: whoever restores it adds the names it uses
(`publisher`, `TS_DIR`, `_imported_names`, `_module_files`).
