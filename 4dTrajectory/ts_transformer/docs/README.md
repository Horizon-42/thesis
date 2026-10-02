# ts_transformer 文档

先读哪一份，取决于要做什么；包的规矩（契约、默认值、陷阱、布局）的索引是上一级的 `CLAUDE.md`。

## 现在读什么

| 要做的事 | 读 |
|---|---|
| 接手两层模型（现行研究线） | [`two_tier/two_tier_stage_notes.zh.md`](two_tier/two_tier_stage_notes.zh.md)：每个阶段做到哪、现行产物、用户的决定、下一步；阶段小结与按优先级的待办：[`two_tier/readouts/2026-09-27_progress_report.zh.md`](two_tier/readouts/2026-09-27_progress_report.zh.md) |
| 两层模型的设计 | [`two_tier/two_tier_framework.zh.md`](two_tier/two_tier_framework.zh.md)（分层、包、产物、门）→ [`instruction_vocabulary_design`](two_tier/instruction_vocabulary_design.zh.md)（词、包络、标注器、取值）→ [`executor_design`](two_tier/executor_design.zh.md) → [`prior_design`](two_tier/prior_design.zh.md) → [`post_training_design`](two_tier/post_training_design.zh.md) → [`multi_aircraft_design`](two_tier/multi_aircraft_design.zh.md)（多机：情境、扩充、间隔、排序；草稿）；对照实验：[`continuous_segment_ablation_design`](two_tier/continuous_segment_ablation_design.zh.md)（连续片段代替指令词，不用标注器；草稿，决定项待确认） |
| 两层模型的读数 | [`two_tier/readouts/2026-09-24_prior_readouts.zh.md`](two_tier/readouts/2026-09-24_prior_readouts.zh.md)（先验、后训练、重建后的重读）；词表第一版的读数 `two_tier/readouts/2026-09-23_instruction_labels_readout.zh.md`（记录）；多机：平行跑道与交叉跑道上的间隔（英文，附航迹实例）[`two_tier/readouts/2026-09-27_parallel_runway_separation.md`](two_tier/readouts/2026-09-27_parallel_runway_separation.md)；新颖性评估与投稿建议（与自动驾驶的异同）[`two_tier/readouts/2026-09-28_novelty_assessment.zh.md`](two_tier/readouts/2026-09-28_novelty_assessment.zh.md) |
| 包的契约、默认值、布局、runner、陷阱的全文 | [`reference/`](reference/)，按编号查（`grep -n '^### C7 ·' reference/*.md`）；单模型路径 2026-09-10 以前的证据在 [`reference/ENGINEERING_NOTES.md`](reference/ENGINEERING_NOTES.md) |
| 发布一次实验 | [`experiments/intents.json`](experiments/intents.json)（没有条目，发布脚本拒绝）和代码仍在的实验的配置（`experiments/*_arms.json`） |
| 两个骨干网络怎么工作 | [`tutorials/`](tutorials/)：iTransformer / PatchTST 教程（机制准确，文中的项目路径是 2026-07 的）、条件化综述、多机终止调研、`einsum` 交互教程（`einsum_tutorial.zh.html`，读 `prior/model.py` 的注意力时用） |
| 查一次已经做完的实验 | [`history/README.md`](history/README.md)（按研究线，一份一行：问的什么、结论、现在在哪） |

## 目录

```text
docs/
  README.md            本文件
  two_tier/            现行设计：文件名不带日期，只写最终设计，原地改（历史交给 git 和 docs/CHANGELOG.md）
    readouts/          带日期的读数，只追加
    figures/           设计文档的示意图；每张由一个 runner 画（`python run_ts.py <名字>`，脚本在 experiments/，测试核对图与脚本一致），不手改
  reference/           CLAUDE.md 索引的全文（P C D S G H L R T W 编号）+ ENGINEERING_NOTES.md
  experiments/         intents.json + 代码仍在的实验的配置（代码已归档的实验，配置跟着进 archive/<名>/docs/experiments/）
  tutorials/           教程与综述；arch_*.svg 由 `python run_ts.py trace_architecture` 生成
  figures/             2026-08-19 坡度摆动诊断的插图
  history/             做完的研究线（代码仍在的），按线分子目录，文件名不改
```

- **代码已归档的研究线**，文档跟代码一起放在 `../archive/<名>/docs/`（oracle 教师、场景编码、closure、计划头、两层 v2、
  意图编码、第一版指令词表、闭环监督微调、程序符合度普查）。
- **`docs/` 下不放代码**（`tests/test_architecture.py::test_docs_holds_no_python`，布局规则 L20）：测量代码是
  `experiments/` 下的 runner，一次性脚本进 `archive/`。

## 以前留在根目录的 10 份记录

它们被参与执行器源码指纹的代码按路径引用，一直等到执行器 v11（2026-09-27）：v11 起指纹只算代码逻辑（去掉文档字符串的语法树，
注释不算），改注释不再作废规格，于是移进 `history/` 和对应的 `archive/`，引用它们路径的代码与文档一起改了（仓库的
`docs/CHANGELOG.md` 是历史，没改；按文件名总能找到）。

| 文件 | 现在在哪 |
|---|---|
| `2026-08-19_control_bank_wiggle_diagnosis.zh.md` | `history/2026-08_control_path/` |
| `2026-09-03_airport_frame_ablation_plan.md`、`2026-09-03_krdu_nw_endpoint_bias.md` | `history/2026-09_frames/` |
| `2026-09-04_procedure_constraints_design.zh.md`、`2026-09-06_control_hooks_results.zh.md` | `history/2026-09_constraints/` |
| `2026-09-07_anytime_prediction_and_calibrated_eta_design.zh.md`、`2026-09-07_latent_intent_design.zh.md` | `history/2026-09_latent_anytime/` |
| `2026-09-14_specific_force_control_design.md` | `history/2026-09_specific_force/` |
| `2026-09-17_two_tier_plan_v2.zh.md` | `../archive/two_tier_v2_2026_09/docs/` |
| `2026-09-18_manoeuvre_token_plan.zh.md` | `../archive/manoeuvre_codes_2026_09/docs/` |
