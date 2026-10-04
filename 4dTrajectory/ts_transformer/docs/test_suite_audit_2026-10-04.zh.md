# ts_transformer 测试套件审查（2026-10-04）

问题：`4dTrajectory/ts_transformer/tests/` 里哪些测试必须留、哪些多余、设置合不合理。
这份文档记录结论、依据、已经做了什么、还剩什么，换一个会话也能接着做。

## 0. 状态

| 项 | 状态 | 在哪 |
|---|---|---|
| A 类（已收尾的一次性实验）搬进 archive | 已做，未合并 | 分支 `dev-archive-one-tier-oneoffs`，提交 `f0e527e6`，worktree `.claude/worktrees/archive-oneoffs`；`dev-two-tier` 已前进到 `c47fcf2a`，合并前要先变基 |
| B 类（有文档、无活代码依赖，共 5 个 runner） | 等用户判断，先不动 | §3 |
| C 类（被别的代码导入，不能单独搬） | 先不动（用户决定） | §3 |
| `runway_hypotheses`（R0b runner） | 等用户决定 | §3 |
| 各测试文件的运行时间 | **没测过** | §5 |
| 夹具重复、导入方式、分层标记 | 只提了建议，没动 | §4 |
| `two_tier_v3_2026_10` 缺顶层 README，`test_architecture` 一项失败 | 已有的问题，不是这次造成的，没修 | §6 |

## 1. 规模与方法

- 147 个文件，47,388 行，约 1,737 个测试函数（按 `def test_` 数，参数化展开后更多）；全套约 55 分钟（出自项目记忆）。
- 没有 `pytest.ini`，没有 slow 之类的标记；只有 `test_end_to_end.py` 的两个 `skipif`。
- 方法是**静态判断**，没有跑套件，也没有测耗时：另一个会话当时正在跑 pytest，占着 CPU。
- 判断一个 runner 是否还活着，用四个信号：`docs/reference/runners.md` 是否收录、有没有活代码导入、`docs/experiments/intents.json` 与两层设计文档是否提到、模块自己的说明。
- 不能用的两个信号：
  - 「`run_ts.py --list` 里有」——`experiments/__main__.py` 用 `pkgutil` 自动发现 `experiments/` 下所有模块，文件在就会列出。
  - git 最后修改日期——09-26、09-27 是一次统一整理，不是使用记录。

## 2. 必须保留的

这些守着现行契约，删了就没人把关：

- 两层模型主线：`prior`、`instructions`、`autopilot`、`executor_conformance`、各个 `traffic_*`，以及 `window/prior/instruction_training_export`、`training_overlays`。
- 发布与身份：`publish_ts_experiment_trajectories`（60 个测试，是通往前端的闸门）、`eligible_set_identity`、`day_split`（封存测试日）、`checkpoint_data_generation`、`lateral_eligibility`。
- 物理与契约：`control_constraints`、`control_inverse_dynamics`、`state_objective`、`channel_contract`、`supervision_terms`、`config_contract`、`run_naming`、`final_approach_geometry`。
- 结构守卫：`architecture`、`import_boundaries`、`frontend_mirrors`、`guidance_skeleton_mirrors`。几乎不花时间；`frontend_mirrors` 防的是新输出类型先发布、前端镜像没更新，导致每个机场的选择器整个变空（发生过两次）。

## 3. 一次性实验的分类

### A 类：已搬（`archive/one_tier_oneoffs_2026_10/`）

判据：`runners.md` 没收录、没有活代码导入、设计文档和 `intents.json` 没提到。

| 模块 | 是什么 | 测试 |
|---|---|---|
| `kinematic_ablation` | 状态路径：动力学损失权重的筛选 | `test_ts_kinematic_ablation` |
| `overfit_diagnostic` | 状态路径：小样本能否过拟合 | `test_ts_overfit_diagnostic` |
| `clock_attribution` | 控制路径：误差按时长、几何、时钟拆开 | `test_ts_clock_attribution` |
| `control_capacity_ceiling` | 控制路径：逐航班的能力上限 | `test_ts_control_capacity_ceiling` |
| `runway_intent_r0` + `_readout` | 跑道意图 R0：只靠因果上下文能否说出落哪条跑道 | `test_runway_intent_r0` |
| `runway_intent_r11` + `_readout` | 跑道意图 R1.1：对称候选的列表式打分头 | `test_runway_intent_r11` |

两个读数器必须跟着搬：它们读对应 runner 的输出文件，留下就成了读不到数据的 runner。

### 审查时先列入、核对后收回的：三个图 runner

`approach_clock_figure`、`approach_legs_figure`、`scene_sample_figure` 画的是已提交的 `docs/two_tier/figures/*.svg`，测试负责让 SVG 和代码保持一致（`test_prior_scene` 还用到其中一个）。**它们是活的，不搬。** 早先说「文档里没有引用」是错的——当时只按模块名搜，漏了图文件本身。

### B 类：有文档、没有活代码导入，等用户判断

| runner | `runners.md` | 备注 |
|---|---|---|
| `heading_lead_ablation` | R23 | 对应 executor v11，已合并 |
| `eta_error_readout` | R4 | |
| `latent_fan_readout` | R6 | |
| `latent_probe` | R5 | `intents.json` 提到 1 次；依赖 `anytime_curve` |
| `eta_calibration` | R2 | `intents.json` 提到 2 次；依赖 `anytime_curve` |

### C 类：被耦合住，不能单独搬

| 模块 | 被谁导入 |
|---|---|
| `runway_intent_r1`（含 `_readout`） | `run_naming.py`；R1.1 的 runner |
| `chain_sensitivity` | `publish_ts_experiment_trajectories.py` |
| `anytime_curve` | `eta_calibration`、`chain_sensitivity`、`latent_probe` |
| `lead_time_error` | `data/anchor_grid.py` |
| `control_basis_oracle` | 两次检查对「有没有导入者」结果不同，**没核实清楚**；`intents.json` 提到 1 次 |

`frame_ablation` 是活的（`runners.md` R8、`intents.json` 6 处、两层格子队列在调），不是候选。

### 另一个悬着的：`runway_hypotheses`

R0b 的 runner。唯一导入它的是已搬走的 `runway_intent_r0_readout`，现在没有活代码读它；只有 `data/runway_context.py` 的一句注释提到名字。不在批准名单里，没动。

## 4. 设置上的问题（只提了建议，没动）

1. **没有分层。** 想只跑「快的契约测试」做不到。慢的候选：`ts_pipeline`（3 折、12 轮、带 subprocess）、`end_to_end`、`anchor_grid_selection`（10 轮训练）、`latent_control`（1046 行）、`publish_*`、`two_tier_v3_grid`。拆分前必须先量耗时（§5）。
2. **夹具重复。** `tests/support.py` 已存在，但 `_series` 在 25 个文件里各定义一遍，`_config` 19 次，`_row` 11 次。同名不一定同内容，合并前要逐个核对。
3. **导入方式脆弱。** `tests/` 没有 `__init__.py`，靠命名空间包被当作 `ts_transformer.tests` 导入，共 64 处；`test_two_head_duration` 还借用另一个测试文件的配置；4 个文件仍自带 `sys.path` 前导，而 `conftest.py` 已经做了。
4. **两个「金丝雀」测试**（`test_end_to_end.py` 的 `skipif` 两个）：要求带已退役字段的旧 checkpoint 仍能加载，文件不在本机时静默跳过。这和项目规则「兼容是禁用词、按名字拒绝」冲突，也可能是用户特许保留的。**要不要留，用户决定。**
5. **价值偏低的小文件：** `test_arm_readout`（1 个测试，断言整段表格输出的精确字符串）、`test_experiment_index`（测 legacy 运行目录）、`test_review_2026_09_09`（文件名是日期，内容是 review 的回归点，应归到各自模块的测试里）、`test_batch_benchmark`、`test_metrics_spread`。

## 5. 没测过的：耗时

所有「慢」的判断都来自代码形状（训练轮数、subprocess），不是测量。要拆分前：

- 在两层队列和 GPU 都空闲时，后台跑一次 `pytest ... --durations=0`，输出写进文件；按「长时间运行脱离会话」的做法（`nohup setsid`，脚本里 `echo $$ > pid`，用 Monitor 按 PID 监视）。
- 跑之前先看内存、GPU 和有没有实验在跑。

## 6. 已有的失败

`tests/test_architecture.py::test_nothing_live_imports_the_archive` 在没改动的 `dev-two-tier`（`47ac23a4`）上就失败：`archive/two_tier_v3_2026_10/` 只有 `docs/README.md`，没有顶层 `README.md`，而测试要求每个归档目录都有。这是提交 `a534a2d9` 把设计文档归档时留下的，与这次搬迁无关，没修。

## 7. 没动的地方

- `outputs/control/forecast.py` 的一句注释举了 `clock_attribution` 当例子：活代码里的注释，等该文件下次真正修改时一起改。
- `docs/code-health-followups.md` 第 6 条引用 `experiments/runway_intent_r0.py:180`：当时核实过的发现，保留。
- 以上两处也记在 `archive/one_tier_oneoffs_2026_10/docs/cut_sections.md` 末尾。

## 8. 接下来（按优先级）

1. 把 `dev-archive-one-tier-oneoffs` 变基到最新 `dev-two-tier`，报告给用户合并（用户决定后 worktree 与分支再清理）。
2. 用户判断 B 类、`runway_hypotheses` 与 `control_basis_oracle`；决定后同样按「模块连同旧测试原样搬」处理。
3. 两层队列空闲时测一次耗时，再据此打分层标记。
4. 用户决定金丝雀测试的去留。
5. 夹具合并与导入清理（机械活，逐个核对同名夹具是否真的相同）。
