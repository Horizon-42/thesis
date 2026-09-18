# two_tier_v2_2026_09 — the two-tier plan v2 (L1 short-horizon control + L2 segment-plan layer)

Archived 2026-09-18 when the manoeuvre-token plan
(`docs/2026-09-18_manoeuvre_token_plan.zh.md`, §5.1) replaced it. Design and every measurement:
`docs/2026-09-17_two_tier_plan_v2.zh.md` (superseded; its §10–§12 numbers are what the new plan
cites). Off the import path on purpose (`tests/test_architecture.py`): nothing live imports it.

| file | was | replaced by |
|---|---|---|
| `segment_plan/` | `outputs/segment_plan/` — the L2 layer (27 segment features in runway axes, one learned query, M = 10 segment ends decoded in one shot) | the manoeuvre-token prior (`manoeuvre/prior.py`, plan §2.5); the descriptor maths is rewritten in the segment-start frame (`manoeuvre/segments.py`) |
| `plan_token_v2.py` | `outputs/control/plan_token.py` — the `truth-next` and `waypoints` decoder tokens | the `manoeuvre-code` token (plan §2.6, P3.1) |
| `tracker_lockstep.py`, `two_tier_gates.py`, `short_horizon_readout.py`, `segment_plan_readout.py` | `experiments/` — the protocol-A/C lockstep, gates L1 / L2 / E2E, the 60 s and the segment-plan readouts | `manoeuvre/lockstep.py`, `manoeuvre/readout.py`, `manoeuvre/gates.py` |
| `two_tier_*_arms.json` | `docs/experiments/` — the L1 / L1b / L1c / L2 / L2c / L2d arm declarations | `docs/experiments/manoeuvre_*_arms.json` |

Their tests are archived beside them, unmodified (`tests/`); they do not run. The campaign trees
(checkpoints, gate and readout JSON/TXT; record trees deleted) are under
`4dTrajectory/outputs/archive/KRDU/two_tier_2026_09/`; the picker categories were withdrawn.
The `PREDICTION_SEGMENT_PLAN` name survives only in `config.PREDICTION_OUTPUTS_RETIRED`, and a
stored config carrying it is refused at load. `control_horizon_s` (fixed rollout horizon) and
`inference/receding.py` stayed live: the new executor uses both.
