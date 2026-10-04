# 两层模型 v4 阶段 A：§14.6 核对报告（2026-10-04）

设计文档 `docs/two_tier/two_tier_design.md` §14.6 列了阶段 A 结束时 Claude 要核对的五项。本报告逐项给结果。
核对对象：分支 `dev-two-tier-v4` 的 `ab295b18`（A0–A14 全部完成，没有合并）；冒烟构建 `smoke_v4/data/a14/`（闭环部分）和
`smoke_v4/data/a12/`（开环回放、A13 读数）。A14 和它审核后的修改见 §6。

## 1 结论

- 五项里，第 2、4、5 项完全通过；第 1、3 项大体通过，有几处要你决定或要补（§3）。
- **最要紧的一处（§3 第 1 条）**：先验需要“每行之前 2 s 内的运动”，在 Δ = 4、8 s 时闭环产物里没有存这个数。正式产物建之前要补，否则阶段 B 读不到。

## 2 逐项结果

### 第 1 项：每个决定对代码

每个决定（D1–D22、D25 的 Δ 值、D26–D28、D32、D33、D38、D42–D47）都找了承载它的模块和测试，逐条读代码核对（表在附录）。
- 完全承载并有测试：大多数。
- 不属于阶段 A（属于阶段 B 的先验）：D5、D13、D16、D17，以及 D1、D3、D10、D32 中关于先验的那一半。
- 部分承载或需要你决定的，见 §3。

### 第 2 项：测试

- 定向测试（闭环、标注器、A13、架构）：116 + 6 个通过。
- 完整 ts 测试集（`ab295b18`）：1,524 个通过（2026-10-04 10:34 跑完，约 10 分钟；日志在 worktree 的 `smoke_v4/data/logs/suite14.log`，被 git 忽略），比 A12/A13 时的 1,523 多 1 个，就是 A14 新加的测试。

### 第 3 项：冒烟构建

| 要求 | 结果 |
|---|---|
| 标注器、执行器、闭环三项一致性核对通过 | 通过（`ab295b18` 的代码重新核对：标注器 252 架读出相同；执行器 250 架，相差 4e-9 m 以内；闭环 150 架，相差 0 m） |
| Δ = 2、4 s 的回放跑完 | 通过（开环 2、4 s；闭环 2、4、8 s） |
| 标注的复飞按复飞飞（G、从高度词爬升、新跑道词、之后着陆） | 通过：select 两架（KRDU、KSTL）、val 一架（KSTL）的闭环句子里都有复飞词、之后有跑道词，复飞后 60 s 内爬升 200–300 m；select 两架回放后着陆（val 的回放要等你同意才跑） |
| 复飞那一行，“不改平”和“未指定速度”按进近是否已到而生效（D26） | 通过：KSTL 进近已截获，两者都生效；KRDU 进近没有截获，两者都不生效 |
| 闭环句子回放能回到存下的状态 | 通过（每架都核对，超过 1e-6 m 会报错） |
| 飞出的航迹在容差内，除了 §4.9 不允许修正的地方（D32） | **字面上不成立**，见 §3 第 4 条 |
| 观测的词跟着匹配点，在最近的 Δ 行说出（D42、D45） | 通过：三个划分、三个 Δ，所有行都对，0 处不符 |
| 变速按档说，执行器每档按 a_max（D43） | 通过：相邻两个速度词 99.9 % 只差一档（每个划分 1–2 处差两档）；a_max 由测试核对 |
| 观测航迹末端之后不修正（D44） | 通过：修正标记只出现在末端后第一行（结束原来的修正）；末端之后生效的角度档和观测的一致 |
| 跨跑道词的航向词按听到时的跑道换算（D46） | 冒烟里没有出现这种情况（0 次），只由测试核对 |
| A13 的读数存在 | 有（`a12/turns_train`、`turns_select`） |

### 第 4 项：没有写真实数据目录

`4dTrajectory/outputs`、`trajectory_data_process/outputs`、`data` 三处，从分支开始（2026-10-03 21:56）以来没有任何文件或目录被写过（按修改时间查，0 个）。

### 第 5 项：归档没有被改

`archive/two_tier_v3_2026_10/` 里 107 个模块都是原样移动（git 认定 100 % 相同）；另有一份旧测试 `tests/test_autopilot.py` 的原样拷贝（和移动前逐字节相同）、一份从现有测试里剪出的段落、和 README。移动之后只有 README 随着后来又移进的模块改过；其他归档目录没有任何改动。

## 3 需要你决定或要补的

1. **先验的运动输入在 Δ = 4、8 s 时没有存（要补）。** 设计 §4.8、§6.1：每行的运动输入取“这一行之前 2 s 内的位移”，不随 Δ 变；闭环里由执行器的飞行状态给出。可是闭环产物只存了每个 Δ 行上的状态，Δ = 4、8 s 时，行前 2 s 的位置不在文件里。阶段 B 要么再飞一遍闭环句子，要么产物里要多存这个数。**我的建议：** 在闭环产物里，每个飞出的行再存“2 s 前的位置”（或直接存这 2 s 的位移），格式名升一版；作为阶段 A 的一个新里程碑（A15），在正式产物之前做。观测的那几行（第一个预测步之前）数据里本来就有。
2. **2 s 标注仍会拒绝一种句子（D46 是否也管 2 s 标注）。** 2 s 标注里，换跑道之后如果新的航向词和生效的档位数字相同，整架被拒绝（§4.3 这么写的，代码也这么做）。D46 写的是 Δ 网格和闭环，那里已经改成“照说”。两边说法不一样：你在第 4 项里同意了 Δ 网格和闭环照说；2 s 标注要不要也照说，还是保留拒绝？**我的建议：** 也照说，规则统一；冒烟里这种情况很少（需要的话我可以先数一下被拒绝了几架）。
3. **Δ 网格没有自己的一致性参照（D21）。** 标注器的一致性核对只重读 2 s 的词；把句子放到 Δ 网格的代码（`on_interval`，D45、D46 在开环里就靠它）改了也会照样通过。闭环那边有自己的参照，开环 Δ 网格没有。**我的建议：** 在标注器的参照样本里，同时存 Δ = 4、8 s 的网格，核对时一起比；也放进 A15。
4. **“飞出的航迹在容差内”这句话字面上不成立（D32）。** 在允许修正的行上，仍有 10–25 % 的行横向偏离超过 30 m（Δ 越大越多：2 s 约 10–12 %，8 s 约 20–25 %），最大几百米。原因是修正一次只偏一档（5°），把偏差拉回来要一段时间，§11.11 已经量过。到最后一行，横向偏差 p90 约 24 m，在容差内。**我的读法：** §14.6 这句应理解为“修正会把飞机拉回容差内”，不是“每一行都在容差内”；请你确认，或者改这句的写法。
5. **高度档位容差的两处读法（D22）。** 每段最高的那一档，容差比 §3.4 写的宽：1,260 m 用 70 m（§3.4 是 40 m），2,700 m 用 235 m（§3.4 是 70 m），因为按 §3.4 的写法，保持在 1,301–1,320 m 或 2,771–2,925 m 的高度都会被拒绝；“不改平”用 40 m。这是 A1 时我的读法，代码和测试里都写明了，这里再请你确认一次。
6. **两处只有行为核对、没有钉住数值（小事）。** D9：没有测试禁止执行器的规律模块去读跑道数据（现在没有读，只是没有东西拦住）；D32：Y = 30 m、H = 15 m 只在一处定义，测试读的是规格里的值，没有钉住 30 和 15。**我的建议：** 各加一条测试，也放进 A15。

另外两处不属于阶段 A：D14 里“复飞时取消爬升不低于进入高度的屏蔽”是先验说话时的屏蔽（阶段 B 的 B4）；Δ = 16 s 能被接受（它能整除 16 s），而消融只用 2、4、8 s，不影响。

## 4 下一步

你决定 §3 的第 1–5 项之后：我把决定写进设计文档，把要补的写成 A15（闭环产物存行前 2 s 的位置、Δ 网格的一致性参照、两条钉住数值的测试，以及第 2 项如果改 2 s 标注），实现、审核、重建冒烟，然后就可以等你选 D15 的拟合值、建正式产物。

## 5 A14

A14（闭环里正好在中间的词也放到后一行，D45）：`66daefa2`；审核后把“放哪一行”的规则只定义一次（`labeller.interval.last_heard_row`，Δ 网格和闭环共用），并加了一个测试：按观测时间飞的飞机，在闭环里听到的词和 Δ 网格说的完全一样（Δ = 2、4、8 s）：`ab295b18`。冒烟 `smoke_v4/data/a14/` 的闭环部分重建：Δ = 2 s 和以前完全一样，Δ = 4、8 s 只有少数词挪了一行，回放结果和 A12 几乎一样（train Δ8 引导偏离 > 300 m 4/40，A12 时 2/40）。

## 附录：决定与代码对照表

（英文，由一个子代理逐条读代码整理，我抽查了 D22、D25、D46 三条；行号以 `ab295b18` 为准。）

| Decision | What it says (≤ 15 words) | Code (path:function, path:line) | Test (file::test name) | Verdict | Note |
|---|---|---|---|---|---|
| D1 | Runway column says the runway and "go-around"; no approach column or "cleared" | `instructions/words.py:20` `COLUMNS`, `:24` `RUNWAY_GO_AROUND`; `instructions/grammar.py:58` `apply` (rule 1: a candidate at the first step); `autopilot/closed_loop.py:259` `Corrector.row` (first predicted step says every column) | test_instruction_vocabulary.py::test_the_columns_are_five_and_the_class_counts_are_the_designs; test_instruction_labeller.py::test_every_row_of_the_runway_table; test_closed_loop.py::test_the_first_predicted_step_says_every_column_in_force_where_the_matched_point_is | carried | "The model says the runway at the first predicted step" as a decoding mask is stage B (§6.2, B2/B4) |
| D2 | Heading words fly the aircraft to the runway; no executor turn onto the final | `autopilot/lateral.py:52` `word_rate`, `:154` `Lateral.rate` (heading words only) | test_autopilot.py::test_a_downwind_base_final_sentence_is_flown_onto_the_final_by_its_words_and_lands | carried | |
| D3 | On the final the executor flies the words; LPV minimums are a judgement only | `autopilot/vertical.py:88` `Vertical.rate` (no aim, no floor); `autopilot/judge.py:136` `decision_check`; `autopilot/runway_data.py:33` (judge or replay only) | test_autopilot.py::test_class_0_holds_the_course_and_keeps_the_offset_the_turn_left; ::test_no_level_off_flies_the_class_angle_without_levelling_off; ::test_the_decision_altitude_check_passes_and_fails | carried | "The model must learn to land or go around" is stage B/C |
| D4 | "Unspecified" starts at the labeller's capture row | `instructions/labeller/speed.py:95` `read_speed` (`:101-115`) | test_instruction_labeller.py::test_the_speed_is_unspecified_from_the_capture_row_unless_a_hold_ends_far_enough_out | carried | Both exceptions of §4.5 are in place |
| D5 | Prior inputs in the frame of R; no airport embedding or absolute position | (stage B) `instructions/airport.py:162` `relative_to_runway` exists | — | not stage A | §6.1, §14.3 B1 |
| D6 | Grids: heading 5°, four descent classes | `instructions/measure.py:50` (`heading_step_deg` 5.0), `:99` `DESCENT_CLASSES = 4`; `instructions/words.py:29` `Words` | test_instruction_vocabulary.py::test_the_columns_are_five_and_the_class_counts_are_the_designs (72 headings; angle 6 = level + 4 + climb) | carried | |
| D7 | No test or measurement criteria in the design; the user sets them | `experiments/executor_replay.py`, `experiments/executor_turns.py` (readouts, no criterion) | test_autopilot.py::test_a_replayed_flight_becomes_a_control_record_and_the_readout_reads_no_criterion | carried | Carried by absence |
| D8 | Heading grid relative to the course of the runway in force | `instructions/words.py:80` `heading_class`, `:84` `heading_track_deg`; `instructions/labeller/lateral.py:71` `per_step_words`; `autopilot/lateral.py:127` `Lateral.word_error` | test_instruction_vocabulary.py::test_heading_classes_are_relative_to_the_course_both_ways; test_instruction_labeller.py::test_downwind_base_final_reads_heading_words_relative_to_the_course_to_the_end; test_autopilot.py::test_a_heading_word_is_converted_with_the_course_of_r_when_heard_and_a_runway_change_does_not_turn | carried | |
| D9 | Executor has none of the §5.7 laws, glidepath floor included | `autopilot/lateral.py` (whole module), `autopilot/vertical.py:88` (the §5.5 modes only), `autopilot/executor.py:58` `MODES` | test_autopilot.py::test_a_downwind_base_final_sentence_is_flown_onto_the_final_by_its_words_and_lands (asserts `MODES == {go_around, level_captured}`); ::test_no_level_off_flies_the_class_angle_without_levelling_off; ::test_class_0_holds_the_course_and_keeps_the_offset_the_turn_left | carried | No lock, capture, intercept, centreline law or floor in `autopilot/`. Checked only by behaviour: no import test stops a law module from reading `runway_data` |
| D10 | Go-around is an event; R unchanged; a runway word ends it | `instructions/grammar.py:77-92` (G set, R kept; a candidate ends G); `autopilot/sentence.py:89` `_filled` (R = last candidate); `autopilot/executor.py:184-190` (+900 s) | test_instruction_labeller.py::test_the_runway_word_ends_the_go_around_and_keeps_r_through_it; test_autopilot.py::test_the_runway_word_after_a_go_around_ends_it_and_a_level_word_holds | carried | Mask effects are stage B (B4) |
| D11 | Labeller stage includes an ablation of the row interval Δ | `instructions/labeller/interval.py:90` `on_interval`; `autopilot/replay.py:77` `sentence_on_interval`; `experiments/executor_replay.py` `--row-interval-s`, `closed_loop_columns` (D34); `experiments/instruction_closed_loop.py` `--row-interval-s` | test_instruction_labeller.py::test_four_seconds_keeps_the_last_word_of_each_interval; test_autopilot.py::test_a_sentence_on_a_coarser_interval_is_flown_and_judged_on_its_own_rows; test_closed_loop.py::test_the_rows_without_a_correction_are_marked_for_the_ablation | carried | |
| D12 | No runway lock; judge reads R at the crossing | `instructions/grammar.py:87-92` (only k ≠ R while G is false); `autopilot/judge.py:159` `_outcome` (`runway_row` per cycle) | test_instruction_labeller.py::test_every_row_of_the_runway_table; test_autopilot.py::test_each_outcome_on_a_hand_built_track (R = 09L case) | carried | |
| D13 | Prior gets the height above R's published glidepath | (stage B) `instructions/airport.py:183` `glidepath_height_m` (the judge uses it too) | — | not stage A | §6.1, B1 |
| D14 | Under G: no "no level-off" (rule 5); entry-height climb mask lifted | `instructions/grammar.py:95-96` (rule 5) | test_instruction_labeller.py::test_rule_5_no_level_off_waits_for_the_runway_word | partly | Rule 5 is carried. The procedure mask half is stage B (§3.7, B4) |
| D15 | Fitted values given with rounder candidates and their fit; the user chooses | `instructions/measure.py:318` `rounding_candidates`, `:335` `climb_distribution`; `experiments/instruction_spec.py:195-198` | test_instruction_vocabulary.py::test_every_fitted_angle_comes_with_rounder_candidates_and_the_fit_each_leaves | carried | The user's choice is still open (§0.4 item 3); the spec keeps the fitted values |
| D16 | No row position embedding; RoPE on row time in seconds | — | — | not stage A | §6.1, B2 |
| D17 | Every column has "time since its word in force", in seconds | — | — | not stage A | §6.1, B1 |
| D18 | Labeller reads real go-arounds and says "go-around" | `instructions/labeller/go_around.py:62` `low_passes`, `:109` `go_arounds`; `instructions/labeller/read.py:168` `flight_go_arounds`, `:208` `read_flight` | test_instruction_labeller.py::test_a_go_around_and_a_second_approach; ::test_a_go_around_needs_a_held_level_before_and_after; ::test_a_touch_and_go_is_refused | carried | Said at the go-around row (the first climb row, D26), not at the lowest point |
| D19 | Runway word ending G: first level-off after the climb, ≤ next "no level-off" | `instructions/labeller/read.py:185` `runway_again_rows` | test_instruction_labeller.py::test_the_runway_word_after_a_go_around_waits_for_the_level_off_after_the_climb | carried | The level-off counts only if it is ≥ 150 m above the low point (`go_around_min_climb_m`) |
| D20 | New version; reads no artefact, spec or prior from v1–v6 | `instructions/spec.py:27-28`; `instructions/artefact.py:43`, `:216`; `autopilot/spec.py:37` (new names, each refused by name) | test_instruction_vocabulary.py::test_the_artefact_round_trips_and_refuses_overwrites_and_other_specs; ::test_from_dict_refuses_a_missing_or_an_extra_key_and_another_reading_rule | carried | |
| D21 | Identities bind format and data rules; code checked by behaviour, data by flights | `instructions/spec.py:227` `sha256`; `instructions/conformance.py:59`, `:150` `check`, `:204`; `autopilot/spec.py:179`; `autopilot/closed_loop.py:575` `check`, `:647`; `autopilot/flights.py:79` `require_same_flight` | test_instruction_conformance.py::test_the_conformance_fails_when_a_labeller_rule_changes; test_executor_conformance.py::test_a_changed_law_is_found; test_closed_loop.py::test_the_conformance_check_passes_the_same_reading_and_finds_every_change; test_autopilot.py::test_a_flight_is_rebuilt_without_reading_its_manifests_bytes | partly | The open-loop Δ grid (`interval.on_interval`) is named in the labeller's code sha, but no reference reads it again. The labeller reference compares 2 s words only (`instructions/conformance.py:150-205`) |
| D22 | Altitude grid: 60 m to 1,260, 120 m to 2,700, 450 m to 5,400 | `instructions/measure.py:61-62`; `instructions/words.py:33-48` `Words.__init__`, `:89` `altitude_index` | test_instruction_vocabulary.py::test_altitude_words_are_the_40_levels_of_three_segments | partly | The grid is exact. ε departs from §3.4 at the two top levels: 1,260 m has 70 m (§3.4: 40) and 2,700 m has 235 m (§3.4: 70). `words.py:41-45` labels this Claude's reading. "No level-off" takes ε = 40 m (`words.py:102-107`, Claude's choice) |
| D25 (Δ = 2, 4, 8 s) | Δ divides the 16 s observation; the first predicted step is a Δ row | `instructions/labeller/interval.py:47` `interval_rows`, `:59` `first_interval_row`; `autopilot/closed_loop.py:111` `start_row` | test_instruction_labeller.py::test_the_row_interval_divides_the_observation; ::test_six_seconds_is_refused_and_a_go_around_at_the_first_row_too; test_autopilot.py::test_a_sentence_on_a_coarser_interval_starts_on_its_utc_grid | carried (values) | `interval_rows` also accepts Δ = 16 s. The "motion from the 2 s before the row" half is stage B. The closed-loop artefact stores states on Δ rows only (`closed_loop.py:392-393`, `:443`; `artefact.py:229`), so at Δ = 4 and 8 s the position 2 s before a flown row is not on disk |
| D26 | Approach by approach; "no level-off" and "unspecified" per approach; go-around row = climb word | `instructions/labeller/read.py:208` `read_flight` (approaches; speed per approach `:233-245`); `instructions/labeller/vertical.py:142` (last descent before a go-around), `:202-209` (go-around rows); `instructions/labeller/lateral.py:53` `capture_row` | test_instruction_labeller.py::test_two_go_arounds_are_three_approaches_each_with_its_own_words; ::test_an_approach_that_ends_at_a_go_around_off_the_corridor_has_no_capture_row; ::test_a_go_around_on_one_runway_and_a_landing_on_another; ::test_a_go_around_without_a_climb_word_is_refused | carried | |
| D27 | Go-around changes no other target; rule 6; heading kept; "unspecified" holds airspeed | `instructions/grammar.py:80-85` (rule 6); `autopilot/lateral.py` (reads no G); `autopilot/speed.py:73` `hear_go_around`, `:88` | test_instruction_labeller.py::test_rule_6_a_go_around_on_a_final_descent_says_a_level_above_and_the_climb; test_autopilot.py::test_a_go_around_climbs_at_the_go_around_angle_keeps_the_heading_word_and_the_airspeed; ::test_a_go_around_on_the_base_keeps_flying_the_base; ::test_go_around_alone_starts_no_climb | carried | Rule 6 reads "above" as more than the level's band above the aircraft (Claude's reading, `grammar.py:18-19`) |
| D28 | One climb word; G true: thrust-limited 1.885°–3°; G false: nominal climb | `autopilot/vertical.py:49-61` `go_around_angle_rad`, `:106-109`; `autopilot/single.py:424-429` | test_autopilot.py::test_the_go_around_angle_is_the_thrust_limited_climb_within_its_limits; ::test_the_runway_word_after_a_go_around_ends_it_and_a_level_word_holds (nominal climb, `:763`) | carried | |
| D32 | Closed-loop reading with corrections; tolerances 30 m lateral, 15 m vertical | `autopilot/closed_loop.py:228` `Corrector`, `:259` `row`, `:312` `_heading`, `:371` `read`; `instructions/measure.py:93-94` (Y, H) | test_closed_loop.py::test_a_heading_correction_is_one_class_toward_the_path_and_ends_under_half_the_tolerance_or_at_a_sign_change; ::test_an_angle_correction_on_a_final_descent_and_none_on_a_level_a_climb_or_beyond_the_classes; ::test_a_flight_whose_words_leave_an_offset_is_brought_back_by_heading_corrections; ::test_a_closed_loop_sentence_flown_again_gives_its_states | carried | Training on flown states is stage B. No test pins Y = 30 m or H = 15 m (tests read `spec.closed_loop_*`) |
| D33 | While G is true, no threshold crossing is an event | `autopilot/judge.py:184-185`; `autopilot/executor.py:237-239` (`~force.go_around`) | test_autopilot.py::test_a_crossing_under_a_go_around_is_not_an_event | carried | Covers both R and another candidate |
| D38 | DA check: ±22 m of evaluation; FAS cone at DA distance; no parameter | `autopilot/judge.py:55` (import `RNAV_TERMINAL_VERTICAL_BOUND_M`), `:136` `decision_check` | test_autopilot.py::test_the_decision_altitude_check_passes_and_fails | carried | Tests 21 m / 23 m above and below, and inside or outside the cone |
| D42 | Observed words said at the place (nearest Δ row to the matched point) | `autopilot/closed_loop.py:171` `ObservedPath.match`, `:259` `Corrector.row`, `:218` `last_words`, `:416-457` (ends when done or at the time limit) | test_closed_loop.py::test_the_observed_words_wait_while_the_flown_aircraft_is_behind; ::test_a_flown_aircraft_ahead_passes_two_observed_rows_in_one_and_hears_the_last_word_of_each_column; ::test_a_flown_aircraft_behind_hears_the_turn_where_the_observed_one_did_and_flies_past_the_observed_time; ::test_a_flown_aircraft_ahead_hears_the_go_around_before_the_threshold_where_the_observed_one_went_around | carried | |
| D43 | Speed words in 5 m/s steps; executor makes each at a_max | `instructions/labeller/speed.py:158` `run_steps`; `autopilot/speed.py:78` `Speed.rate`; `autopilot/single.py:470-475` | test_instruction_labeller.py::test_a_deceleration_of_40_mps_from_hold_to_hold_says_its_eight_steps_where_the_speed_comes_nearer_to_each; ::test_a_run_that_turns_without_a_hold_steps_to_the_grid_value_nearest_the_turn_and_noise_says_nothing_back; test_autopilot.py::test_a_speed_word_is_flown_at_a_max_both_ways_and_unspecified_at_its_own_pace; ::test_a_speed_step_is_made_in_seconds_and_the_thrust_limit_binds_where_the_aircraft_cannot_slow_at_a_max | carried | |
| D44 | Past the observed path's end: no correction; lateral readout only; no vertical error | `autopilot/closed_loop.py:183-189` (`past_end`, e_h NaN), `:279-292` (no correction, one in force ends), `:192` `uncorrected_m` | test_closed_loop.py::test_past_the_end_of_the_observed_path_its_last_segment_goes_on_without_a_height; ::test_past_the_end_of_the_observed_path_no_correction_starts_and_one_in_force_ends; ::test_a_flight_whose_observed_path_ends_early_gets_no_correction_past_its_end | carried | |
| D45 | A word goes to the nearest row; a tie goes to the later row, on both grids | `instructions/labeller/interval.py:68` `last_heard_row`, `:101` (Δ grid); `autopilot/closed_loop.py:268`, `:360` (closed loop, first step, refusal) | test_instruction_labeller.py::test_a_word_goes_to_the_nearest_interval_row_and_a_tie_to_the_later; test_closed_loop.py::test_a_flight_on_schedule_hears_in_closed_loop_the_words_the_interval_grid_says; ::test_the_observed_heading_words_of_a_turn_are_said_at_the_nearest_row_on_average; ::test_a_go_around_at_the_first_predicted_step_and_a_short_sentence_are_refused | carried | One rule for both grids since `ab295b18` |
| D46 | Heading word in the frame where heard; said when its track differs; none refused | `instructions/labeller/interval.py:108-122` (Δ grid); `autopilot/closed_loop.py:312` `Corrector._heading` | test_instruction_labeller.py::test_a_heading_word_moved_across_a_runway_word_is_said_in_the_frame_where_it_is_heard; ::test_a_heading_word_of_the_class_in_force_under_another_course_is_said_on_the_interval; test_closed_loop.py::test_a_heading_word_passed_with_a_runway_word_is_said_in_the_frame_where_it_is_heard; ::test_a_change_of_runway_alone_says_no_heading_word_and_a_correction_is_in_the_frame_where_it_is_heard | partly | Carried on the Δ grid and in the closed loop. The 2 s open-loop reading still refuses a new word whose class equals the class in force after a change of R (`instructions/labeller/lateral.py:117-122`; tested by test_instruction_labeller.py::test_a_heading_word_of_the_class_in_force_after_a_runway_change_is_refused). §4.3 prescribes this, but D46 read literally ("also when its class is the class in force said under another course"; "no sentence is refused") does not |
| D47 | Executor turn law stays; turn readout is its own part, with Δe_y beside it | `autopilot/lateral.py` unchanged since A13 (`git log 47865c8e..HEAD -- autopilot/lateral.py` is empty); `experiments/executor_turns.py` (`own_m`, `change_m`; ways a/b/c and the control) | test_executor_turns.py::test_three_ways_and_the_control_fly_at_the_observed_speed_and_read_the_offset_across_each_turn; ::test_the_readout_reads_each_way_by_stratum_and_speed_band | carried | |

