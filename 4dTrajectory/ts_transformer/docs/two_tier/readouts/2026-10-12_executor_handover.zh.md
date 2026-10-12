# 执行器与词表提前量交接（2026-10-12 18:40）

接手的 agent 先读本文件，再读设计文档 `docs/two_tier/design/executor_first_principles.zh.md`（草案 5，§15 是决定，§17
是词表改动），再读 `docs/two_tier/design/vocabulary.md` §4.4。

## 0. 现在的状态

- **没有任何活动在跑。** Mac 上的完整测试已由用户在 18:35 喊停。
- **开发转到 SP-AI**（用户 2026-10-12："我会转移到sp-ai上开发"）。
- 本分支 `two-tier2` 的全部改动都已提交（见第 1 节）。分支还没有推送，也没有合并。
- 正式重建（新的 instruction 产物、执行器规格、闭环、训练）**还没有开始，要用户明确同意**。

## 1. 代码与分支

- 分支 `two-tier2`，worktree `.claude/worktrees/two-tier2`，起点 `6c3cd8f6`（instruction-v7）。
- 今天的提交（从旧到新）：

| 提交 | 内容 | 审核 | 测试（Mac） |
|---|---|---|---|
| `223f1e30` 执行器 B | 第一性原理执行器，设计草案 5：一阶滞后的 plant、作动器观测器加精确反求、滚转率 p = 15°/s、三个作动器时间常数（假设值）；规格 `ts-executor-spec-v11`，回放 `ts-executor-replay-v9`，后端回答 `aeroviz-autopilot-segment-v10`（推力、坡度、载荷系数是作动器实际到达的值，不再是指令） | Opus 两轮，第二轮无问题 | 见第 4 节 |
| `477df46a` instruction-v8 | 标注器：最后一个下降段的最后一个角度词，用最后进近直线段的路径角来读（D10）；`final_approach.fit_projected_segment` 从 `fit_final_segment` 拆出（结果不变，有测试） | Opus 两轮：第一轮 1 个 S2（复飞前下降段的规则没有测试），已补；第二轮无问题 | 标注器 + `final_approach` 133 passed |
| 本提交 lead_ablation | runner R71（`experiments/lead_ablation.py`，V12：一次改一列提前量，重新标注、闭环读取）、它的测试、`runners.md` R71、`CLAUDE.md` 索引行、本交接文档 | **没有审核**（用户要求随交接一起提交） | `test_lead_ablation.py` 在相关套件里通过 |

## 2. 用户今天的决定

| 编号 | 决定 |
|---|---|
| D9 | 三列提前量都取 4 s：L = L_γ = L_v = 4 s |
| D10 | 方案 B：最后一个下降段的最后一个角度词，用最后进近直线段（入口前 5000–300 m，与 harvest 外推观测入口高度的拟合相同）的路径角来读；没有这条线的航迹被拒绝；复飞前的下降段不变。读法版本升到 instruction-v8 |

"unchanged" 的含义（用户问过，代码已是这样）：这一行没有新词，上一个词连同它的到达时刻（听到 + 4 s）仍然有效。它不是
"保持当前状态"。新词替换旧词，重新计时。这一条还没写进词表文档（Q3）。

## 3. 证据（本机实验）

实验脚本已移入仓库：`4dTrajectory/ts_transformer/archive/executor_first_principles_2026_10/`（README 有每个脚本的用途和
路径的改写）。实验结果在 Mac 上：`4dTrajectory/outputs/POOLED/spai_copy_20261011/plant_comparison/`。**结果在 git 忽略的
outputs 下，不随代码过去**；需要时从 Mac 用 rsync 取。数字也已写进设计文档 D9、D10。

- `lead_eval_1000.py`、`lead_eval_1000_combo.py`：V12，train 999 架（每机场 200 架，种子 1337），Δ 2 s 与 4 s。4/4/4 对基准
  4/0/0：Δ 2 s 845 对 818，Δ 4 s 836 对 818。代价：垂直失败 22 → 34、20 → 30。
- `lead_eval_tail.py`、`compare_tail.py`：末端规则原型。B：垂直失败 34 → 11（Δ 2 s）、30 → 9（Δ 4 s），通过 +20（两个
  间隔都是），逐架变好 23–24、变差 3–4。A（入口前 300 m 内不读新角度词）只到 27 / 25。
  注意：第一版原型只用"到入口前的距离"选行，三边腿也会被选中，结果错误，已删除。
- `v8_refusals.py`：全部 train 信号 44,703 架，v8 新拒绝 37 架（0.08%，"no final straight-in line"；KSJC 25、
  KSMF 10），最后一个角度词换类别 5,727 架（12.9%）。

## 4. 测试状态（Mac，未完成）

- Mac 上没有 xdist，ts 全套串行要约 2 小时。
- 完整测试的第一遍：ts 40 failed / 2137 passed；后端 + `final_approach` + evaluation 40 failed / 367 passed。
- 后端那组和干净 HEAD 对比过：只多出 2 个，都是前端测试数据比较
  （`test_autopilot_segment.py::FixtureTest::test_the_frontend_fixture_of_an_answer_is_what_the_service_answers`、
  `test_window_segment.py::test_the_frontend_fixtures_of_the_live_answers_are_what_the_service_answers`），原因是执行器 B
  改了回答。其余失败在 HEAD 上也有（`test_traffic_jobs` 读 /proc，只能在 Linux 上跑）。
- **ts 那组的 HEAD 对比没有做完**（被喊停）。已知：
  - 3 个前端测试数据比较（stage A/B/C 的 `test_the_frontend_fixtures_are_what_the_export_writes`）在 Mac 上一直失败，
    测试数据要在 Linux 上写。
  - `test_post_training_export.py::test_a_listed_set_flies_each_window_as_its_selection_readout_did` 和
    `test_multi_training_export.py::test_stage_d_s_set_holds_the_answered_losses_and_every_other_one_dashed`
    在 HEAD 上也失败。
  - `test_multi_training_export.py::test_a_silent_aircraft_the_executor_finished_ends_where_the_judge_ended_it`：
    执行器 B 让它失败（静默飞机遇到第二次冲突，窗口提前结束），已把复制飞机加入的延迟从 4 步改为 5 步，通过。
  - 其余失败（`test_post_train.py`、`test_post_profile.py`、`test_post_value.py`、`test_prior_train.py` 等，多是
    worker 进程、内存采样、冒烟训练）**还没有和 HEAD 对比**，不能断定和改动无关。

## 5. 在 SP-AI 上的下一步（按顺序）

1. 取分支：`git fetch` 后检出 `two-tier2`（要先从 Mac 推送；本文件写成时还没推送）。
2. **重新生成测试数据**：在 Linux 上用 `AEROVIZ_WRITE_FIXTURES=1` 跑这些测试，再看 `git diff` 是否只有预期的变化：
   - `4dTrajectory/ts_transformer/tests/test_training_export.py`、`test_prior_training_export.py`、
     `test_post_training_export.py`、`test_multi_training_export.py`（stage A/B/C/D 的前端测试数据）
   - `aeroviz_backend/tests/test_autopilot_segment.py`、`test_window_segment.py`、`test_prior_segment.py`
3. **跑全套**：`./run_all_tests.sh`（xdist，约 3 分钟）。把失败清单和 `6c3cd8f6` 的对比；新增失败逐个查原因。
4. **审核 lead_ablation**（Opus 子代理，范围是它的提交）。`runners.md` R71 最后一句还写着 "Needs an `instruction-v7`
   artefact"，应改为 instruction-v8。
5. **正式重建**（要用户同意）：
   - instruction 产物：读法 instruction-v8，提前量 4/4/4（`instruction_spec --angle-lead-s 4 --speed-lead-s 4`，航向
     提前量沿用规格里的 4 s）
   - 执行器规格 `ts-executor-spec-v11`（`executor_spec`）
   - 闭环句子（`instruction_closed_loop`）
   - 训练，约 26 小时
6. 其他未完成：
   - 设计文档 §14 第 13 条：教程 `docs/two_tier/tutorial/index.html` 还是旧执行器，`make_vectors.py` 导入就失败。
   - Q3：§17 的词表改动要写进词表文档并编号（词表文档的 D 编号由用户确认）；"unchanged" 的含义也要写进去。
   - 设计文档 §10.2 的验证项 V1–V14 多数没有做。

## 6. 用户的规则（今天仍然有效）

- 回答用中文，ASD-STE100 风格：短句，一句一个意思，先给答案再给理由，给小数字例子。
- 执行器是训练工具，不只是模拟：不跟踪中心线，不看观测路径，不看未来的词。
- 限制只用物理极限或已有的明确值，不要自己加限制。
- 审核代码用 Opus 子代理；审核者不是作者；文档不送审核。
- 非平凡的修法先让用户同意（诊断可以自己做）。
- 不读 val。
- 在 Mac 上时，用户要求远程 SP-AI 只读、不占用太多资源。现在开发转到 SP-AI，这条规则怎么变，先问用户。
- 不要在没有用户明确命令时启动长时间的运行。
