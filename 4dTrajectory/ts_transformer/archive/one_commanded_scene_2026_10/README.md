# one_commanded_scene_2026_10 — the one-aircraft scene loop of M3 and M4's first pass

Archived 2026-10-02 (multi-aircraft design `docs/two_tier/multi_aircraft_design.zh.md` §6.6 step 9, "9.4 的代码"; the
user's decision §9 item 37: R37 takes a one-commanded draw, and the one-aircraft mode code this made redundant is
removed). The setting "一架由模型指挥" (design §2.2) — one aircraft spoken to, every other one replayed along its
record — is now a window with one commanded aircraft in the window loop: `experiments/traffic_window.py`
`draw_windows(commanded="one")`, read by R34 `traffic_window_generation --commanded one` and trained by R37
`traffic_window_reward --commanded one`. Before the move, tests pinned that a one-commanded window says, flies, ends and
trains as this scene loop did (`tests/test_traffic_window.py`, `tests/test_traffic_window_tuner.py`,
`tests/test_prior_window_speaker.py` at `dev-two-tier` `6a4514da`). Off the import path on purpose
(`tests/test_architecture.py`): nothing live imports it.

| file | was | replaced by |
|---|---|---|
| `experiments/traffic_free_generation.py` | R31: M3 — the start speaking to one aircraft of each scene, read beside the model alone, the labelled words and the record | R34 `--commanded one` |
| `experiments/traffic_reward.py` | R32: M4's first runner — the traffic post-training on one-aircraft scenes, round by round | R37 `--commanded one`; its round protocol lives on unchanged in `experiments/traffic_rounds.py` |
| `experiments/traffic_reward_readout.py` | R33: an R32 run read round by round | R39 `traffic_window_reward_readout` |
| `experiments/traffic_reward_val.py` | an R32 round read on the val days | R34 `--split val` + R41 `traffic_window_pair` |
| `experiments/traffic_tuner.py` | `SceneRewardTuner`: each sentence scored in its scene | `experiments/traffic_window_tuner.WindowRewardTuner` (the shared parts — parameter groups, the data term, `sweeps`, `part_cost`, `SCORE_BUDGET` — moved there unchanged) |
| `experiments/traffic_augment.py` | augmented scenes: the leader moved (D), the start moved (B), a flight inserted (A) | `experiments/traffic_window_augment.py` — D and the replayed A added for one-commanded draws (`KINDS_OF`); `moved`, `SHIFT_S`, `GAP_RANGE`, `TRIES` moved there |
| `prior/scene_speaker.py` | `SceneSpeaker`: the prior speaking to one aircraft of each scene | `prior/window_speaker.py` (`BLOCK_STEPS` moved there) |
| `experiments/traffic_speaking.py` | this copy is the module as it was; the live one keeps the scene view, masks, edge rows and judge the window loop uses | its scene loop (`SceneLoop`, `scene_of`, `speaking_edges`, `other_node`) → `traffic_window.WindowLoop` |

Their tests are archived beside them, unmodified (`tests/`; `tests/test_traffic_speaking.py` is the file as it was —
the live one keeps the tests of the live functions); they do not run. The published results these runners made stay where
they are: `outputs/POOLED/traffic/free_generation_*`, `outputs/POOLED/prior/m4_traffic_20260929/`,
`outputs/POOLED/prior/m4_passes_20260929/` and their readouts (`docs/two_tier/readouts/2026-09-28_m3_free_generation.zh.md`,
`2026-09-29_m4_traffic.zh.md`, `2026-09-30_m4_passes.zh.md`). Their traffic priors (traffic r5 among them) already refuse
to load on the code that put every row on the UTC steps (the edge features' source hash changed, 2026-10-02).

**Documents**: the runner manual entries R31–R33 are cut here verbatim (`docs/reference/entries.md`).
