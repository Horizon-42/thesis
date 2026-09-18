# 两层计划 v3 · 阶段 A 开发笔记（工作文档，开发中随时维护）

目的：压缩 context 后从这里接着做。计划本身在 `2026-09-18_two_tier_plan_v3.zh.md`（§5–§8 是阶段 A 的详细计划，用户已看过）。
本文只放开发时要用的事实、状态和约定；结论和读数不在这里。

## 0. 硬约束（用户，2026-09-18）

- **签字前不启动任何实验**（M-A0′）。代码写完、测试过、review 过之后停下，等用户说"跑"。
- 代码要简洁，不写补丁式的守卫；契约清楚，一处定义。杂活（下载、归档、长测试、bulk 改动）和 review 交给 opus agent；设计与数字留在主 context。
- 每个开发项完成 → opus review（只 review 代码）→ 修 → 提交（`git diff --cached --stat` 先看，只 stage 明确路径）。
- 不拿早先实验的结论当设计；引用要带配置。计划里定死的参数都是 §6 的编号决定项。
- 不删除、不覆盖任何产物；`manoeuvre_tok_20260918/` 原样保留。

## 1. 阶段 A 的开发项与状态

| # | 开发 | 状态 | 提交 |
|---|---|---|---|
| A-dev1 | 网格臂文件：声明 (L, Δ) 网格；每臂 `seq_len = L/2` 行、`control_horizon_s = Δ`、`n_segments = Δ/10`；不设 `anchor_floor_index` | 未开始 | |
| A-dev2 | 新 cohort：KRDU 全部进场按航班一次划分（split seed 1337）；每格可用集 = 记录 ≥ L + Δ；`rebuild_cohort` 出处校验更新；旧 cohort 文件保留 | 未开始 | |
| A-dev3 | 首问 = L−1 行（不设 floor 即是）；测试钉住 | 未开始 | |
| A-dev4 | `manoeuvre_lockstep --anchor-remaining-km 12|8|6`：每架在剩余路程首次 ≤ X km 的行起问 | 未开始 | |
| A-dev5 | 训练锚点全随机：`random_train_anchor_l1_share = 0`（09-18 是 0.5） | 未开始 | |
| A-dev6 | 新 runner `executor_failure_modes`：读 lockstep 每架记录，算停在哪 / 对准过没有 / 转弯早晚，归类计数，写记录供发布 | 未开始 | |
| A-dev7 | 选格 gate：读 40 份 lockstep 产物，按 §5.1 规则输出胜出格与整表；seed 线由本网格 seed 对读出 | 未开始 | |
| A-dev8 | 队列脚本：按格串行（Δ 升序、L 升序），每格两 seed 训完立即读两种读数并通知；PID 文件；PushNotification | 未开始 | |

顺序建议：dev2 → dev1 → dev3 → dev5（训练侧，一起冒烟）→ dev4 → dev7 → dev6 → dev8。

## 2. 代码事实（写开发项时查到的）

- 训练 runner：`experiments/frame_ablation.py`，吃臂文件 JSON（09-18 的在 `docs/experiments/manoeuvre_tok_20260918_s{20,30,60,90,120}_arms.json`）；
  每臂目录 `<campaign>/<arm>/{config.json, checkpoint.pt, history.json, experiment_manifest.json, data_selection.json}`；训练完成的标志 `history.json`（`TRAIN_COMPLETE_ARTIFACT`）。
- 首问行：`config.default_anchor(config)` = `seq_len − 1`，除非 `anchor_floor_index` 设了更晚的行；`data/dataset.fixed_anchor_index` 是唯一定义（C17）。
  09-18 campaign 用 floor 59（"120 s 契约"），v3 不设。
- 训练锚点：`random_train_anchor = True`、`random_train_anchor_sampling = remaining-path-uniform`；`random_train_anchor_l1_share` 是钉在固定锚点的份额（09-18 为 0.5）。
- cohort：`<campaign>/development_cohort.json`（schema `ts-development-cohort-v1`，`selection` + `splits`）；09-18 的选法是"锁定划分（split_seed None，openap-direct）去掉
  120 s 随机锚点契约覆盖不到的 train 航班"（`random_train_anchor_min_future_s = 120`）。`experiments/support.rebuild_cohort(payload, config, keys)` 按 checkpoint 的出处重建（C25）。
- 09-18 训练超参（D8 继承）：`epochs 180, patience 180, learning_rate 3e-5, batch_size 512, checkpoint_selection_metric fixed-anchor-common-grid-ade`；
  契约 `control_thrust_parameterization specific-force+path-angle`、`control_dynamics_model first-order-lag`、`control_duration_parameterization uniform`。
- 距离 bin：`data/anchor_strata.DEFAULT_ANCHOR_GRID_KM = (20, 16, 12, 8, 6, 4, 2)`；剩余路程 `approach_difficulty(item, anchor).remaining_path_m`；anchor grid 机制在 `data/anchor_grid.py`、runner `anytime_curve`。
- lockstep：`manoeuvre/lockstep.py`（协议 C / A / A-truth / none；`fly()`、`flight_row()`；预算 `closing_horizon_s`），runner `experiments/manoeuvre_lockstep.py`
  （`--protocol none` 需要一个同段长的 codebook 只作标注；`--split`、`--limit`、`--write-records`）。阶段 A 只用 `none`。
  **待定**：`none` 下是否还要求 codebook——阶段 A 没有 codebook 可给（无 token 臂），要么允许省略（truth_codes / flown_codes 列为空），要么导出一个占位 codebook。倾向前者（简洁）。
- 每臂约 45 MB；GPU RTX 4060 8 GB；每臂 12–15 min。
- 发布：`publish_ts_experiment_trajectories.py --reuse-prediction-dir … --category-variant …`，intents 变体 key 大小写不敏感（57ae87e）；`VARIANT_RECORD_BLOCKS` 含 `manoeuvre_lockstep`。
- readout 的臂发现按 `S<seg>_(K\\d+|cv|nt)_s<seed>` 形状（01e5286）；新网格的臂名要定（建议 `L<L>_D<Δ>_s<seed>`，并让发现规则认识它，或阶段 A 不用 `manoeuvre_readout`）。

## 3. 测试与环境

- 单文件测试：`timeout 600 conda run -n aeroviz --no-capture-output python -m pytest 4dTrajectory/ts_transformer/tests/<file> -q --import-mode=importlib`
- 全套 ~7 min，前台 600 s 或 detached；`test_architecture.py` 管 import 方向。测试不得触及 live 根目录（写根全进 tmp）。
- 永远 `conda run -n aeroviz --no-capture-output python …`。
- 主树自由（没有 campaign 在跑）；实验从 detached worktree 跑（`.claude/worktrees/manoeuvre-runs`，launch 前 `git -C … checkout --detach <sha>`）。

## 4. 本文维护约定

每完成一个开发项：更新 §1 状态与提交号；新查到的代码事实进 §2；待定项解决后从 §2 删掉并在 §1 注明。
