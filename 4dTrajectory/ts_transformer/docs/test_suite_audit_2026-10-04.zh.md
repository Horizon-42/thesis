# ts_transformer 测试套件审查（2026-10-04）

问题：`4dTrajectory/ts_transformer/tests/` 里哪些测试必须留、哪些多余、设置合不合理。
这份文档记录结论、依据、已经做了什么、还剩什么，换一个会话也能接着做。

**基线换过一次。** 审查是在 `dev-two-tier`（`47ac23a4`）上做的；之后用户要求不在 `dev-two-tier` 上开发，把
`dev-two-tier-v4` 合进自己的分支再整理。v4 把 instruction-v3 的先验和多机测试连同代码一起归档了
（`archive/two_tier_v3_2026_10/`），所以下面的数字和分类都以**合并 v4 之后**为准；审查时的旧数字只在 §1 列出对照。

## 0. 状态

| 项 | 状态 | 在哪 |
|---|---|---|
| A 类（已收尾的一次性实验）搬进 archive | 已做 | 分支 `dev-archive-one-tier-oneoffs`，提交 `f0e527e6`；归档目录 `archive/one_tier_oneoffs_2026_10/` |
| 把 `dev-two-tier-v4` 合进该分支 | 已做 | 合并提交 `045439a7`，无冲突 |
| 一次性实验的复核（v4 基线） | 已做 | §3 |
| 重复夹具合并、测试之间互相导入的清理（冷文件） | 已做，28 个测试文件 + 新模块 `support_prediction.py` | §4 |
| B 类中 `eta_error_readout`、`latent_probe`、`latent_fan_readout` 搬进 archive（用户：「B 类的也搬进去」） | 已做 | 同一归档目录；`test_two_head_duration` 里属于前者的 2 个测试摘出，见 §3 |
| `chain_sensitivity`（用户：「搬进 archive」）、B 类中的 `eta_calibration` | **没搬，等用户**：搬了会让活代码失去测试或失去输入的产生者 | §3 |
| C 类其余、`runway_hypotheses` | 按用户决定先不动 | §3 |
| 热文件里测试互相导入的清理 | 没做，等 v4 开发告一段落 | §4 |
| 各测试文件的运行时间、分层标记 | **没测，没做** | §5 |
| 金丝雀测试（旧 checkpoint 仍能加载）去留 | 等用户决定 | §4 |

## 1. 规模与方法

| | 审查时（`dev-two-tier`） | 合并 v4 后（A 类已搬走） |
|---|---|---|
| 测试文件 | 147 | 100 |
| 行数 | 47,388 | 36,038 |
| 测试函数（按 `def test_`） | 约 1,737 | 1,364 |

- 全套约 55 分钟（出自项目记忆，是旧基线上的数）。没有 `pytest.ini`，没有 slow 之类的标记；只有 `test_end_to_end.py` 的两个 `skipif`。
- 方法是**静态判断**，没有跑全套，也没有测耗时：另一个会话一直在跑 pytest，占着 CPU。
- 判断一个 runner 是否还活着，用四个信号：`docs/reference/runners.md` 是否收录、有没有活代码导入、`docs/experiments/intents.json` 与两层设计文档是否提到、模块自己的说明。
- 不能用的两个信号：
  - 「`run_ts.py --list` 里有」——`experiments/__main__.py` 用 `pkgutil` 自动发现 `experiments/` 下所有模块，文件在就会列出。
  - git 最后修改日期——09-26、09-27 是一次统一整理，不是使用记录。

## 2. 必须保留的

这些守着现行契约，删了就没人把关。以合并 v4 后的文件为准：

- 两层模型阶段 A（v4）：`closed_loop`、`start`、`final_descent_tolerance`、`instruction_*`（`vocabulary`、`labeller`、`conformance`）、`autopilot`、`executor_*`、`training_export`。这几组正在被 v4 的各分支频繁修改。
- 发布与身份：`publish_ts_experiment_trajectories`（60 个测试，是通往前端的闸门）、`eligible_set_identity`、`day_split`、`checkpoint_data_generation`、`lateral_eligibility`。
- 物理与契约：`control_constraints`、`control_inverse_dynamics`、`state_objective`、`channel_contract`、`supervision_terms`、`config_contract`、`run_naming`、`final_approach_geometry`。
- 结构守卫：`architecture`、`import_boundaries`、`frontend_mirrors`、`guidance_skeleton_mirrors`。几乎不花时间；`frontend_mirrors` 防的是新输出类型先发布、前端镜像没更新，导致每个机场的选择器整个变空（发生过两次）。

审查时列在这里的 `prior`、`traffic_*`、`window_*` 测试，v4 已随 v3 一起归档，不再在 `tests/` 里。

## 3. 一次性实验的分类（以合并 v4 后为准）

### A 类：已搬（`archive/one_tier_oneoffs_2026_10/`）

判据：`runners.md` 没收录、没有活代码导入、设计文档和 `intents.json` 没提到。

| 模块 | 是什么 | 测试 |
|---|---|---|
| `kinematic_ablation` | 状态路径：动力学损失权重的筛选 | `test_ts_kinematic_ablation` |
| `overfit_diagnostic` | 状态路径：小样本能否过拟合 | `test_ts_overfit_diagnostic` |
| `clock_attribution` | 控制路径：误差按时长、几何、时钟拆开 | `test_ts_clock_attribution` |
| `control_capacity_ceiling` | 控制路径：逐航班的能力上限 | `test_ts_control_capacity_ceiling` |
| `runway_intent_r0` + `_readout` | 跑道意图 R0 | `test_runway_intent_r0` |
| `runway_intent_r11` + `_readout` | 跑道意图 R1.1 | `test_runway_intent_r11` |

两个读数器必须跟着搬：它们读对应 runner 的输出文件。

### 审查时先列入、核对后收回的：三个图 runner

`approach_clock_figure`、`approach_legs_figure`、`scene_sample_figure` 画的是已提交的 `docs/two_tier/figures/*.svg`，测试负责让 SVG 和代码保持一致。**它们是活的，不搬。** 早先说「文档里没有引用」是错的——当时只按模块名搜，漏了图文件本身。v4 基线上这三张图仍在。

### B 类：有 `runners.md` 条目、没有活代码导入的四个（用户要求搬）

| runner | 结果 |
|---|---|
| `latent_fan_readout`（R6） | 已搬。干净 |
| `latent_probe`（R5） | 已搬。活文档里提到它的几处（`CLAUDE.md` 的 C8 一行、`contracts.md` 两处）改成指向归档，注释里的提法保留 |
| `eta_error_readout`（R4） | 已搬。它的 2 个测试原在 `test_two_head_duration.py` 里，和活代码的测试混在一起：从活文件摘出（其余 20 个测试逐个比对未变），连同头部导入和它们用到的辅助函数，作为**摘录**放进归档的 `tests/test_two_head_duration_eta_readout.py` |
| `eta_calibration`（R2） | **没搬。** 审查时「没有导入者」只查了 Python 导入，漏了三处：(1) 它是 `predict --cta-from-quantiles` 读的那张共形校准表的**唯一产生者**，`cli/predict.py` 的提示信息还叫用户去运行它；(2) `test_two_head_duration` 里 `test_the_calibration_runner_takes_a_two_head_checkpoint` 用它；(3) `test_cta_from_quantiles` 用它的测试夹具。搬走后活的预测功能没法再得到校准表 |

`runners.md` 里这三个条目的全文搬进归档的 `docs/reference/entries.md`，活文件只留一行标题（沿用 R18 的先例）；`CLAUDE.md` 索引里的相应句子剪走，索引里不留指针。

### C 类：被耦合住，不能单独搬

| 模块 | 状态 |
|---|---|
| `runway_intent_r1`（含 `_readout`） | `run_naming.py` 导入 r1；R1.1 的 runner 已归档但读它们 |
| `anytime_curve` | `eta_calibration`、`chain_sensitivity` 共用；`runners.md` R1 收录。`latent_probe` 搬走后少了一个用户 |
| `lead_time_error` | `data/anchor_grid.py` 导入 |
| `control_basis_oracle` | **已核实是活的**：`cli/common.py` 和 `config.py` 的报错信息让用户去跑它，`outputs/control/basis_fit.py` 读它的输出（拟合教师表，C23） |
| `chain_sensitivity` | 用户要求搬，**没搬**。它没有活代码导入、`runners.md` 没收录，按判据符合；但 (1) `test_chain_sensitivity.py` 的 15 个测试是 `inference/receding.py`（`rolled_series`、`cut_at_lead`、`displacement_at`）**仅有**的测试，而 `receding.py` 被活的无令牌闭环 `manoeuvre/lockstep.py` 导入；搬走测试，这些活代码就没有测试了；(2) `test_publish_ts_experiment_trajectories.py` 的 `test_the_publishers_variant_blocks_mirror_the_runners` 导入它的 `RECORDS_BLOCK`，要像 `manoeuvre_readout` 那样改成镜像常量 |

`frame_ablation` 是活的（`runners.md` R8、`intents.json` 6 处、两层格子队列在调），不是候选。

### 另一个悬着的：`runway_hypotheses`

R0b 的 runner。`runners.md` 没收录；唯一导入它的是已搬走的 `runway_intent_r0_readout`；`data/runway_context.py` 里只有一句注释提到名字。不在批准名单里，没动。

## 4. 设置上的问题

**已做（冷文件，即 v4 各分支不在改的）：**

- 新增 `tests/support_prediction.py`，放单层预测测试共用的夹具。`support.py` 仍是两层线的夹具；分开是因为 v4 的几个分支（`dev-two-tier-v4-prior`、`dev-step9-one-commanded`）都在改 `support.py`，往里追加会和它们冲突。
- 搬进去的只有两类：几个文件里**逐字相同**的（`_series` 14 处、`_series(config, n)` 4 处、`_config` 3 处、`_identity_normalizer` 3 处，先用 AST 逐一核对相同再搬），以及**被别的测试文件导入**的（`quantile_config`、`latent_config`、`given_cta_config`、ETA 校准的 `_cohort` 等一组）。名字相同但内容不同的（`_config` 其余 16 种，`_state`、`_row` 等）不是同一个东西，没合并。
- 原文件用原来的名字导入（`series as _series`），调用处一个字没改。涉及 28 个测试文件，371 个测试函数的语法树和 HEAD 比对：25 个文件完全相同；另 3 个（`test_auto_batch`、`test_ts_predictability_report`、`test_two_head_duration`）只差函数里的一行导入语句，是预期的引用改动。
- 因搬走而不再用到的导入已清掉。`test_evaluation_protocol` 里的 `_series` 本来就没人用，是死代码，只去掉了导入。
- 测试文件之间互相导入是个隐患：`--import-mode=importlib` 下，一个测试模块被另一个测试模块导入，会被加载两次。

**没做：**

1. **热文件里的测试互相导入**：`test_closed_loop` ← `test_instruction_labeller`（`GO_AROUND_LEGS`）、`test_instruction_conformance`（`_artefact`）；`test_executor_turns`、`test_training_export` ← `test_closed_loop`（`_batch`、`_params`）；`test_executor_conformance` ← `test_autopilot`；`test_instruction_vocabulary` ← `test_instruction_labeller`。这些文件近三天每个都有多次提交（`test_closed_loop` 23 次、`test_autopilot` 17 次），搬走会和 v4 各分支冲突。等阶段 A 告一段落再做。
2. **没有分层**：想只跑「快的契约测试」做不到。慢的候选：`ts_pipeline`（3 折、12 轮、带 subprocess）、`end_to_end`、`anchor_grid_selection`（10 轮训练）、`latent_control`（1046 行）、`publish_*`、`two_tier_v3_grid`。拆分前必须先量耗时（§5）。
3. **`sys.path` 前导**：审查时数到 4 个文件，核对后其中三个是各有用处的（`final_approach_geometry`、`guidance_skeleton_mirrors` 要加别的包路径，`import_boundaries` 是子进程里的），`test_architecture` 那处与 `conftest.py` 重复，但同一个测试文件在检查「没人把包目录放进 `sys.path`」，留着不影响。**没有可删的。**
4. **两个「金丝雀」测试**（`test_end_to_end.py` 的 `skipif` 两个）：要求带已退役字段的旧 checkpoint 仍能加载，文件不在本机时静默跳过。这和项目规则「兼容是禁用词、按名字拒绝」冲突，也可能是用户特许保留的。**要不要留，用户决定。**
5. **价值偏低的小文件：** `test_arm_readout`（1 个测试，断言整段表格输出的精确字符串）、`test_experiment_index`（测 legacy 运行目录）、`test_review_2026_09_09`（文件名是日期，内容是 review 的回归点）、`test_batch_benchmark`、`test_metrics_spread`。它们测的是仍在用的代码，归档会让这些代码没有测试，所以不搬；要不要把 review 回归点并进各模块的测试文件，等用户说。
6. **没有按子系统分目录。** 100 个文件平铺，但文档和别的会话的命令里到处写着 `tests/test_xxx.py` 的路径，分目录要改一大批引用，又会和 v4 各分支的测试改动冲突；没做，需要的话再议。

## 5. 没测过的：耗时

所有「慢」的判断都来自代码形状（训练轮数、subprocess），不是测量。要拆分前：

- 在两层队列和 GPU 都空闲时，后台跑一次 `pytest ... --durations=0`，输出写进文件；按「长时间运行脱离会话」的做法（`nohup setsid`，脚本里 `echo $$ > pid`，用 Monitor 按 PID 监视）。
- 跑之前先看内存、GPU 和有没有实验在跑。

## 6. 没动的地方

- `outputs/control/forecast.py` 的一句注释举了 `clock_attribution` 当例子：活代码里的注释，等该文件下次真正修改时一起改。
- `docs/code-health-followups.md` 第 6 条引用 `experiments/runway_intent_r0.py:180`：当时核实过的发现，保留。
- 以上两处也记在 `archive/one_tier_oneoffs_2026_10/docs/cut_sections.md` 末尾。

## 7. 接下来（按优先级）

1. 把本分支报告给用户合并（用户决定后 worktree 与分支再清理）。
2. 用户决定：`eta_calibration` 与 `chain_sensitivity` 怎么办（见 §3：要么保留，要么先为 `receding.py` 单独补一份针对活接口的测试、为校准表另找产生者，再搬）；`runway_hypotheses` 搬不搬。
3. 两层队列空闲时测一次耗时，再据此打分层标记。
4. 用户决定金丝雀测试的去留。
5. v4 阶段 A 告一段落后，清理热文件里测试互相导入的那一批（§4 没做的第 1 条）。
