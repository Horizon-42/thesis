# ts_transformer 文档

先读哪一份，取决于要做什么；包的规矩（契约、默认值、陷阱、布局）的索引是上一级的 `CLAUDE.md`。

## 现在读什么

| 要做的事 | 读 |
|---|---|
| 接手两层模型（现行研究线） | [`two_tier/two_tier_stage_notes.zh.md`](two_tier/two_tier_stage_notes.zh.md)：每个阶段做到哪、现行产物、用户的决定、下一步 |
| 两层模型的设计 | [`two_tier/two_tier_framework.zh.md`](two_tier/two_tier_framework.zh.md)（分层、包、产物、门）→ [`instruction_vocabulary_design`](two_tier/instruction_vocabulary_design.zh.md)（词、包络、标注器、取值）→ [`executor_design`](two_tier/executor_design.zh.md) → [`prior_design`](two_tier/prior_design.zh.md) → [`post_training_design`](two_tier/post_training_design.zh.md) |
| 两层模型的读数 | [`two_tier/readouts/2026-09-24_prior_readouts.zh.md`](two_tier/readouts/2026-09-24_prior_readouts.zh.md)（先验、后训练、重建后的重读）；词表第一版的读数 `two_tier/readouts/2026-09-23_instruction_labels_readout.zh.md`（记录） |
| 包的契约、默认值、布局、runner、陷阱的全文 | [`reference/`](reference/)，按编号查（`grep -n '^### C7 ·' reference/*.md`）；单模型路径 2026-09-10 以前的证据在 [`reference/ENGINEERING_NOTES.md`](reference/ENGINEERING_NOTES.md) |
| 发布一次实验 | [`experiments/intents.json`](experiments/intents.json)（没有条目，发布脚本拒绝）和代码仍在的实验的配置（`experiments/*_arms.json`） |
| 两个骨干网络怎么工作 | [`tutorials/`](tutorials/)：iTransformer / PatchTST 教程（机制准确，文中的项目路径是 2026-07 的）、条件化综述、多机终止调研 |
| 查一次已经做完的实验 | [`history/README.md`](history/README.md)（按研究线，一份一行：问的什么、结论、现在在哪） |

## 目录

```text
docs/
  README.md            本文件
  two_tier/            现行设计：文件名不带日期，只写最终设计，原地改（历史交给 git 和 docs/CHANGELOG.md）
    readouts/          带日期的读数，只追加
  reference/           CLAUDE.md 索引的全文（P C D S G H L R T W 编号）+ ENGINEERING_NOTES.md
  experiments/         intents.json + 代码仍在的实验的配置（代码已归档的实验，配置跟着进 archive/<名>/docs/experiments/）
  tutorials/           教程与综述；arch_*.svg 由 `python run_ts.py trace_architecture` 生成
  figures/             2026-08-19 坡度摆动诊断的插图
  history/             做完的研究线（代码仍在的），按线分子目录，文件名不改
  2026-…（10 份）       留在根目录的历史记录，见下
```

- **代码已归档的研究线**，文档跟代码一起放在 `../archive/<名>/docs/`（oracle 教师、场景编码、closure、计划头、两层 v2、
  意图编码、第一版指令词表、闭环监督微调、程序符合度普查）。
- **`docs/` 下不放代码**（`tests/test_architecture.py::test_docs_holds_no_python`，布局规则 L20）：测量代码是
  `experiments/` 下的 runner，一次性脚本进 `archive/`。

## 留在根目录的 10 份历史记录

它们都已是记录（结论已进 `reference/` 的默认值或陷阱），但被**参与执行器源码指纹的代码**按路径引用（`config.py`、
`outputs/envelope.py`、`outputs/constraints/speed_floor.py`；契约 C33）：改那几行注释会让现行代码拒绝执行器规格 v9。
所以等下一次执行器规格重建时，再和那些注释一起移进 `history/`（或对应的 `archive/`）。

| 文件 | 是什么 | 以后去哪 |
|---|---|---|
| `2026-08-19_control_bank_wiggle_diagnosis.zh.md` | 控制输出坡度摆动的诊断 → `simple-v3`、T6–T8 | `history/2026-08_control_path/` |
| `2026-09-03_airport_frame_ablation_plan.md` | 机场坐标系消融的计划（结果在 `history/2026-09_frames/`）→ D1 | `history/2026-09_frames/` |
| `2026-09-03_krdu_nw_endpoint_bias.md` | 状态模型 KRDU 终点偏西北 → T13 | `history/2026-09_frames/` |
| `2026-09-04_procedure_constraints_design.zh.md` | 程序约束的量测与设计 → D5 | `history/2026-09_constraints/` |
| `2026-09-06_control_hooks_results.zh.md` | 指令钩子（barrier / trombone）读数 → D6、H1–H4 | `history/2026-09_constraints/` |
| `2026-09-07_anytime_prediction_and_calibrated_eta_design.zh.md` | 任意时刻预测与 ETA 校准（A、B 线）→ R1–R4 | `history/2026-09_latent_anytime/` |
| `2026-09-07_latent_intent_design.zh.md` | 潜在意图 L0–L5 → D11–D13、R5–R6 | `history/2026-09_latent_anytime/` |
| `2026-09-14_specific_force_control_design.md` | 比力控制参数化 N3–N7′ → C27、D25–D27 | `history/2026-09_specific_force/` |
| `2026-09-17_two_tier_plan_v2.zh.md` | 两层 v2 计划（代码已归档）；§10–§12 的数可引用 | `archive/two_tier_v2_2026_09/docs/` |
| `2026-09-18_manoeuvre_token_plan.zh.md` | 意图编码计划（代码已归档） | `archive/manoeuvre_codes_2026_09/docs/` |
