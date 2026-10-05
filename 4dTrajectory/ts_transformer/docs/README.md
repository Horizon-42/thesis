# ts_transformer 文档

先读哪一份，取决于要做什么；包的规矩（契约、默认值、陷阱、布局）的索引是上一级的 `CLAUDE.md`。

## 现在读什么

| 要做的事 | 读 |
|---|---|
| 接手两层模型（现行研究线） | [`two_tier/design/outline.md`](two_tier/design/outline.md)：设计由哪几份文档组成、文档之间只经公共接口往来、共同的原则与规则、各阶段计划；每个阶段做到哪在各文档的 §0 |
| 两层模型的设计（英文） | [`outline`](two_tier/design/outline.md) → [`vocabulary`](two_tier/design/vocabulary.md)（阶段 A：词表、标注器、执行器、判定；公共接口 §6）→ [`prior`](two_tier/design/prior.md)（阶段 B；公共接口 §7）→ [`post_training`](two_tier/design/post_training.md)（阶段 C：后训练与多机，大纲） |
| 两层模型被取代的设计 | `instruction-v3` 及以前的中文设计（框架、词表、执行器、先验、后训练、多机）、阶段笔记和对照实验草稿（连续片段代替指令词）：[`../archive/two_tier_v3_2026_10/docs/`](../archive/two_tier_v3_2026_10/docs/README.md)；拆分以前的单份设计 `two_tier_design.md` 和高度网格提案：[`history/2026-10_two_tier_design/`](history/README.md) |
| 审核两层模型的一个阶段 | [`two_tier/review_guide.md`](two_tier/review_guide.md)（英文，ASD-STE100）：审核原则（每个输入追到来源和时间、输入与目标/读数分开、由数据决定的集合也是通道、只按行为检查）、步骤、阶段 B 和 C 的检查清单、阶段 A 审核的发现作为例子 |
| 给各阶段实现者的当前命令 | [`two_tier/notes/`](two_tier/notes/)：`stage_a.md`、`stage_b.md`、`stage_c.md`，每个阶段一份，只放最新一条命令，新命令整份覆盖（不是日志；过程在各阶段的实现日志里） |
| 两层模型的读数 | [`two_tier/readouts/2026-09-24_prior_readouts.zh.md`](two_tier/readouts/2026-09-24_prior_readouts.zh.md)（先验、后训练、重建后的重读）；词表第一版的读数 `two_tier/readouts/2026-09-23_instruction_labels_readout.zh.md`（记录）；多机：平行跑道与交叉跑道上的间隔（英文，附航迹实例）[`two_tier/readouts/2026-09-27_parallel_runway_separation.md`](two_tier/readouts/2026-09-27_parallel_runway_separation.md)；新颖性评估与投稿建议（与自动驾驶的异同）[`two_tier/readouts/2026-09-28_novelty_assessment.zh.md`](two_tier/readouts/2026-09-28_novelty_assessment.zh.md) |
| 包的契约、默认值、布局、runner、陷阱的全文 | [`reference/`](reference/)，按编号查（`grep -n '^### C7 ·' reference/*.md`）；单模型路径 2026-09-10 以前的证据在 [`reference/ENGINEERING_NOTES.md`](reference/ENGINEERING_NOTES.md) |
| 发布一次实验 | [`experiments/intents.json`](experiments/intents.json)（没有条目，发布脚本拒绝）和代码仍在的实验的配置（`experiments/*_arms.json`） |
| 两个骨干网络怎么工作 | [`tutorials/`](tutorials/)：iTransformer / PatchTST 教程（机制准确，文中的项目路径是 2026-07 的）、条件化综述、多机终止调研、`einsum` 交互教程（`einsum_tutorial.zh.html`，读 `prior/model.py` 的注意力时用） |
| 查一次已经做完的实验 | [`history/README.md`](history/README.md)（按研究线，一份一行：问的什么、结论、现在在哪） |
| 整理测试套件（哪些必须留、哪些已归档、哪些待判断） | [`test_suite_audit_2026-10-04.zh.md`](test_suite_audit_2026-10-04.zh.md) |

## 目录

```text
docs/
  README.md            本文件
  two_tier/            两层模型
    design/            现行设计（英文）：大纲 + 词表 + 先验 + 后训练，文件名不带日期，只写最终设计，原地改（历史交给 git 和 docs/CHANGELOG.md）
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
