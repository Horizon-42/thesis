# 两层计划 v3 · 阶段 A 开发笔记（工作文档，开发中随时维护）

目的：压缩 context 后从这里接着做。计划本身在 `2026-09-18_two_tier_plan_v3.zh.md`（§5–§8 是阶段 A 的详细计划，用户已看过）。
本文只放开发时要用的事实、状态和约定；结论和读数不在这里。

## 现状（2026-09-19 深夜更新；接手的 agent 先读这一节和 §7，再读计划 v3 的 §5.2 / §6.2 / §7.2 / §8.2 / §9.2）

**2026-09-20 更新**：意图码版本的阶段 B 作废（读数留在结果文档 §9–§11）；计划 v3 的 §5.2 / §6.2 / §7.2 / §8.2 / §9.2 已按指令词表重写（提交 1fe0660）：词 = 绝对目标的管制指令（航向相对最后进近航道、跑道入口以上高度、地速、切入、保持，约 58 词，τ = 5–10 s），执行器吃生效指令，两层 CAT-K 式闭环再训（D49、D55），K 候选滚出按目标打分（D56），程序作解码掩码（D59），RL 本阶段不做（D58），多机层在阶段 C 之后（D60）；文献 `docs/literature/trajectory_as_language/`。**未签字、未开发**；下一步是用户审 §5.2 / §6.2，然后 B0′（标注器 + 300 架人工核）。§7 以下是意图码版本的开发记录。

**哪里停下**：阶段 A 全部完成（M-A0 … M-A3）。阶段 B：用户 2026-09-19 晚答复——指令词表对照臂做（12 臂）、B0 可以先跑、"最大程度利用时间，不干等实验"；
B-dev0…7 全部写完、测试通过（全套 1288）、三轮 opus review（见 §7.1），**已提交 4cd11ea（M-B0 代码就绪）；未签字（M-B0′）、队列未跑**；runs worktree 已移到 4cd11ea，从它 dry-run 通过（GROUP baselines + 6 组，122 步全 todo）。
提前跑的 B0（19:53Z，半成品代码，payload v2）已按用户指示删除；基线改由队列第 0 步自己飞（§7.2），门只认本版代码的 payload——用户由此立了"兼容是禁词、结构先敲定再实验"的原则（根 CLAUDE.md 编码约定、记忆 `no-compat-settle-structures-before-experiments`）。
Q3（S60-held 的训练锚点契约）用户已按建议定（D48：≥ Δ = 20 s，token 段放不下时用上一段的码）。
什么都没在跑：GPU 空闲；runs worktree `.claude/worktrees/manoeuvre-runs` 停在 5208707（launch 前 `checkout --detach` 到含阶段 B 代码的提交）。磁盘约 10.5 GB 空余（B0 用了 0.5 GB）。

**阶段 A 的结论（细节与每个数在 `2026-09-18_two_tier_v3_results.zh.md`，节号见下）**
- 网格 20 格 × 2 seed（L ∈ {30,60,90,120} s × Δ ∈ {20,30,60,90,120} s）跑完（§1 读数 (a) 从各自 L−1 起、§2 读数 (b) 从 12/8/6 km 起、§7 读数 (c) 从第 59 行起）。
- 判定（§3，读数 (a)）：L120_D20 胜出，decisive；seed 线 p75：established 全部 0.078、雷达 0.184、雷达 ADE 150 m；读数 (c) 上 seed 线 0.080 / 0.195 / 124 m。
- 读数 (c) 改写了 L 轴的解释：同一起点下 L ≥ 60 的格差在 seed 线内，L = 120 在 (a) 里的领先是起点效应（少飞 60 s）。**用户定阶段 B 的执行器 = L60_D20**（(a) 0.748 / 0.750，(c) 0.786 / 0.785）。
- 条件（§5）：L = 30 在 Δ ≤ 30 选模落在第 1 epoch（固定 L−1 锚点的 ADE 在初始网络最好），读的是初始网络；八格（L ≥ 60、Δ ≥ 30）选模贴 180 epoch 预算；Δ = 20/30 好于 Δ ≥ 60。
- A2（§6，胜出格 L120_D20）：雷达引导航班未越线的 278 / 183 架里 passed-abeam（没转基边）0.36 / 0.61、established-short（对准晚约 20 s、预算用完）0.26 / 0.21、overshoot + parallel-offset（末段形状错）0.37 / 0.18。假设 H1（转弯时机是意图信息 → 阶段 B）、H2（转晚 → A3-a）、H3（形状错 → A3-b）。
- A3（§8）：A3-a = 网格的 L60_D60 两臂只飞前 20 s（`--execute-s 20`，不重训）——雷达组两读数两 seed 都好 +0.07–0.14、未超线，直线组变差（seed 2024 从第 29 行起 0.940 → 0.682）；A3-b = N₁ = 4 新训两臂——无收益，雷达 ADE 变差。两者 §3.3 A3 行都 FAIL；执行器不换。
- 发布：L120_D20 两 seed 的闭环记录与 9 个失败方式子集共 11 类进了 picker（`categories.json` 171 → 182）；正在跑的 Vite dev server 对新目录返回 SPA HTML，重启前端才能加载；`experiments/index.json` 早于本战役，发布用了临时索引，未重建（重建会覆盖，等用户）。
- 文献：`docs/literature/prediction_horizons/`（14 篇核过原文：终端区历史 11 s–3 min、视界 90 s–5 min，采样 1/6/10 s；没有人闭环飞到跑道入口、没有人扫历史长度）。

**产物（`4dTrajectory/outputs/KRDU/experiments/`）**
- `two_tier_v3_grid_20260918/`：40 个臂目录；`cohorts/<cell>/development_cohort.json`；`lockstep/<arm>/{L-1,12km,8km,6km,row59}/`（Δ = 120 的格没有 6km）；`lockstep_exec20/L60_D60_s*/{L-1,row59}/`（A3-a）；
  `lockstep_records/L120_D20_s*/L-1/`（带记录，218 / 213 MB）；`failure_modes/L120_D20_s*/`；`gate/after_<cell>/`、`gate/final_{12km,row59}/`、`gate/relative_a3a_{L-1,row59}/`；`two_tier_grid_queue.log`。
- `two_tier_v3_a3b_20260919/`：`L60_D20_N4_s*/`、`lockstep/…/{L-1,row59}/`、`gate/after_L60_D20/`、`gate/relative_a3b_{L-1,row59}/`。
- 没有码本、没有先验、没有任何带 token 的臂（阶段 B 一个都没建）。

**今天加的代码（都已 review、测试、提交）**
- `manoeuvre_lockstep --first-prediction-row N`（`lockstep.from_row`；队列读数名 `row<N>`；payload `first_prediction.common_row`）——读数 (c)。
- `manoeuvre_lockstep --execute-s N`（`lockstep.fly(execute_s)`、`lockstep.executed_step_s`；payload `executed_s`；门表与失败方式脚本带出该字段）——只允许无 token 读法；N 要能被 dt 与积分步长 0.5 s 整除。
- `executor_relative_gate --baseline SEED=DIR SEED=DIR --candidate … (--seed-line-from grid_gate.json | --seed-line a b c) --out DIR`（`gates.gate_relative`）——共有航班上重算三项、给 §3.3 A3 行与 B 行两个判定；就是计划里的 B-dev4（D45），已可用。
- 未做的：`executor_grid_gate --reading 8km|6km`（`gates.cell_reading` 拒绝空的雷达层）；B-dev1/2/3/3′/5/6/7 全部未做。

**用户今天定的规则（都已写进记忆文件，接手的 agent 照做）**
- 说人话：不造词（"题 / 首问 / 上限 / 回灌 / 持有 / 相位 / 码图谱"都改掉了）；没解释过的概念第一次出现就说明（FSQ、直通估计、teacher forcing、top-1、cohort……）；代码里的代号不当名字用（协议 C / A / A-truth / none 写成真值 token 读法 / 先验 token 读法 / 先验读真值历史的读法 / 无 token 读法）。
- 文档结构：§5 / §6 / §7 / §8 / §9 各一个大块，阶段是块内的小节（5.1 A、5.2 B …），不另起 §5B 这类平行节。
- 每一步先写"为什么有这一步 → 做什么 → 怎么判 → 不过怎么办"，配置表要写清每个配置怎么来的；不把阶段 A 的做法按惯性搬到 B（第 59 行读数在 B 里没用处，已删）。
- 用户说"代码就位就跑"时可以让实验跑；没说之前不建臂、不跑。计划的每个参数是编号决定，用户可能改。

**等用户决定的**：签阶段 B（§5.2 等）；D36 要不要指令词表对照（+6 臂）；D35 的 K64 条件步骤；A4 采样间隔消融（§5.1.5 / D29，未排期）；L = 30 的选模锚点、Δ ≥ 30 的 epoch 预算要不要改；前端重启与 `index.json` 重建。

**阶段 B 签字后的开发顺序（§5.2.5）**：B-dev4 已有 → B-dev7 臂文件与 intents（S20 两臂先）→ B-dev2 / B-dev3′（S60-h60）→ B-dev1 / B-dev3（S60-held）→ B-dev5（先验落地不停飞）→ B-dev6 队列；每项 opus review、测试、提交；B cohort（L60 划分、记录 ≥ 120 s）用 `plan_cohort --arms` 写出；worktree 移到该提交；dry-run；等用户说"跑"。

## 0. 硬约束（用户，2026-09-18）

- **签字前不启动任何实验**（M-A0′）。代码写完、测试过、review 过之后停下，等用户说"跑"。
- 代码要简洁，不写补丁式的守卫；契约清楚，一处定义。杂活（下载、归档、长测试、bulk 改动）和 review 交给 opus agent；设计与数字留在主 context。
- 每个开发项完成 → opus review（只 review 代码）→ 修 → 提交（`git diff --cached --stat` 先看，只 stage 明确路径）。
- 不拿早先实验的结论当设计；引用要带配置。计划里定死的参数都是 §6 的编号决定项。
- 不删除、不覆盖任何产物；`manoeuvre_tok_20260918/` 原样保留。

## 1. 阶段 A 的开发项与状态

| # | 开发 | 状态 | 提交 |
|---|---|---|---|
| A-dev1 | 网格臂文件：`docs/experiments/two_tier_v3_grid_arms.json`，40 臂，键 `L<L>_D<Δ>_s<seed>`，每臂自己的 cohort 路径；`frame_ablation --only KEY…` 只跑指定臂；测试钉住 D1/D2/D4–D8/D10 | 完成（review 1 通过并修） | fa0426f（文档 4dd2082） |
| A-dev2 | cohort：`plan_cohort --arms <臂文件>` 一次加载、按格写 20 份 `cohorts/L<L>_D<Δ>/development_cohort.json`（已写出，n 表见 §2）；`frame_ablation` 每臂用自己的 cohort；`rebuild_cohort` 不用改（读 checkpoint 自带的 split） | 完成（review 1 通过并修；cohort 已写出） | fa0426f（文档 4dd2082） |
| A-dev3 | 第一次预测在第 L−1 行：臂文件 `anchor_floor_index = 0`，测试钉住 `default_anchor == seq_len − 1` | 完成（含在 dev1 测试里） | fa0426f（文档 4dd2082） |
| A-dev4 | `manoeuvre_lockstep --anchor-remaining-km 12|8|6`：`lockstep.from_remaining_path` 把每架从 bin 行（`anchor_grid.bin_anchor`：剩余路程最接近 X km 的行）切起，使该行成为固定锚点；协议 `none` 不再要 codebook；payload schema v2（`first_prediction` 块、每行 `first_prediction_row`；轮次记录叫 `rounds`，次数叫 `predictions`） | 完成（review 通过并修，冒烟通过） | fa0426f（文档 4dd2082） |
| A-dev5 | 训练锚点全随机：臂文件 `random_train_anchor_l1_share = 0.0`，测试钉住 | 完成（含在 dev1 测试里） | fa0426f（文档 4dd2082） |
| A-dev6 | `executor_failure_modes --lockstep <dir> --out <dir>`（`manoeuvre/failure_modes.py`）：course 坐标系、六类失败方式、每类记录子集 `records_<mode>/`（`export.copy_record_subset`） | 完成（review 通过并修，冒烟通过） | fa0426f（文档 4dd2082） |
| A-dev7 | `executor_grid_gate --campaign --arms --out [--reading L-1]`（`gates.gate_grid`）：整表 + 胜出格；seed 线 = 本网格 seed 差的 p75 | 完成（review 1 通过并修） | fa0426f（文档 4dd2082） |
| A-dev8 | `two_tier_grid_queue --arms --campaign --airport KRDU [--readings …] [--cells …] [--dry-run]`：每格 train（`frame_ablation --only`）→ 8 份 lockstep 读数 → 选格表；PID 文件 `<campaign>/two_tier_grid_queue.pid` | 完成（review 2 通过并修，dry-run 通过） | fa0426f（文档 4dd2082） |

顺序建议：dev2 → dev1 → dev3 → dev5（训练侧，一起冒烟）→ dev4 → dev7 → dev6 → dev8。

**M-A0 达成（2026-09-18 晚）**：全部八项提交在主树（fa0426f 代码、4dd2082 文档、5079bb4 把 lockstep 里的 "ask" 改名为 prediction），全套 ts 测试 1273 通过，两轮 opus review 的发现都已修。**M-A0′ 用户 2026-09-19 说"开始实验"，队列按 §5 启动并跑完（见 §6 与开头的现状）。**

## 2. 代码事实（写开发项时查到的）

- 训练 runner：`experiments/frame_ablation.py`，吃臂文件 JSON（09-18 的在 `docs/experiments/manoeuvre_tok_20260918_s{20,30,60,90,120}_arms.json`）；
  每臂目录 `<campaign>/<arm>/{config.json, checkpoint.pt, history.json, experiment_manifest.json, data_selection.json}`；训练完成的标志 `history.json`（`TRAIN_COMPLETE_ARTIFACT`）。
- 开始预测的行：`config.default_anchor(config)` = `seq_len − 1`，除非 `anchor_floor_index` 设了更晚的行；`data/dataset.fixed_anchor_index` 是唯一定义（C17）。
  09-18 campaign 用 floor 59（"120 s 契约"），v3 不设。
- 训练锚点：`random_train_anchor = True`、`random_train_anchor_sampling = remaining-path-uniform`；`random_train_anchor_l1_share` 是钉在固定锚点的份额（09-18 为 0.5）。
- cohort：`<campaign>/development_cohort.json`（schema `ts-development-cohort-v1`，`selection` + `splits`）；09-18 的选法是"锁定划分（split_seed None，openap-direct）去掉
  120 s 随机锚点契约覆盖不到的 train 航班"（`random_train_anchor_min_future_s = 120`）。`experiments/support.rebuild_cohort(payload, config, keys)` 按 checkpoint 的出处重建（C25）。
- 09-18 训练超参（D8 继承）：`epochs 180, patience 180, learning_rate 3e-5, batch_size 512, checkpoint_selection_metric fixed-anchor-common-grid-ade`；
  契约 `control_thrust_parameterization specific-force+path-angle`、`control_dynamics_model first-order-lag`、`control_duration_parameterization uniform`。
- 距离 bin：`data/anchor_strata.DEFAULT_ANCHOR_GRID_KM = (20, 16, 12, 8, 6, 4, 2)`；剩余路程 `approach_difficulty(item, anchor).remaining_path_m`；anchor grid 机制在 `data/anchor_grid.py`、runner `anytime_curve`。
- lockstep：`manoeuvre/lockstep.py`（协议 C / A / A-truth / none；`fly()`、`flight_row()`；预算 `closing_horizon_s`），runner `experiments/manoeuvre_lockstep.py`
  （`--split`、`--limit`、`--write-records`、`--anchor-remaining-km`）。阶段 A 只用 `none`：不带 codebook，行里没有码列。
  旧的 09-18 `lockstep_none_*` 产物是旧 schema（v1：带标注 codebook，字段叫 `asks` / `asks_e` / `first_ask`），原样保留，没有活代码再读它们。
- 用词（2026-09-19 用户定）：中文文档一律"预测 / 开始预测 / 每预测一次"，不用"问 / 首问 / 起问"；代码里 `predictions` / `held_predictions` / `rounds` /
  `first_prediction` / `first_prediction_row` / `predictions_p50`，不用 ask。`chain_sensitivity` 里的 "re-ask" 是更早的仪器，没动。
- 每臂约 45 MB；GPU RTX 4060 8 GB；每臂 12–15 min。
- **各格 cohort（已写出，`two_tier_v3_grid_20260918/cohorts.json`）**：锁定划分 12218 架里 3956 架被机型过滤（openap-direct）拒掉，与格无关；
  可用集 train 6798–6857 / val 1392–1405（L30_D20 最多，L120_D120 最少；随机锚点覆盖不到的 train 航班 0–3 架）。各格 n 差 ≤ 65 架。
- **从固定距离处开始预测的读数（冒烟，09-18 的 S60_nt 臂，24 架）**：12 km 处 3 架没有可用行；在 12 km 行重读分组时几乎全是 straight-in（曲折度 < 1.05），
  雷达引导组会很薄——读数 (b) 主要看"全部"与 straight-in，8 / 6 km 更甚；写结果时要报每组 n。
- `series_from_row(series, r)`（`data/dataset.py`）：航班从第 r 行起看，时钟不变、真值同样切；lockstep 的 bin 读数靠它把 bin 行变成固定锚点。
- 发布：`publish_ts_experiment_trajectories.py --reuse-prediction-dir … --category-variant …`，intents 变体 key 大小写不敏感（57ae87e）；`VARIANT_RECORD_BLOCKS` 含 `manoeuvre_lockstep`。
- readout 的臂发现按 `S<seg>_(K\\d+|cv|nt)_s<seed>` 形状（01e5286）；网格的臂名定为 `L<L>_D<Δ>_s<seed>`，阶段 A 没有用 `manoeuvre_readout`（已解决）。

- **产物布局（dev7/dev8 定）**：`<campaign>/<arm>/`（训练）、`<campaign>/lockstep/<arm>/<reading>/manoeuvre_lockstep.json`（reading ∈ L-1、12km、8km、6km）、
  `<campaign>/gate/after_<cell>/grid_gate.json`（每格完成后的整表；40 臂齐后才有 verdict）、`<campaign>/cohorts/<cell>/development_cohort.json`。
- `frame_ablation` 的磁盘估算：predict 臂 400 MB、train-only 臂 100 MB（09-18 实测 45 MB）；整文件 40 臂 dry-run 现在报需 3.9 GiB，空余 13.5 GiB。
- opus review 1（dev1–5）已做完并修：cohort 加 build 的长度门（`dataset.minimum_build_samples`）；`plan_cohort --arms` 要求各格在 `GRID_AXES`
  之外完全一致、拒绝任何 CLI 配置 flag、cohort 目录名唯一；`frame_ablation --only` 仍校验全部臂、`null` cohort 继承文件级；lockstep 拒绝 `control_horizon_s = 0`。
  未改的提醒：距离读数 (b) 的覆盖随 Δ 变（写进 D3）。
- opus review 2（dev6–8）已做完并修：记录子集自带重算的 `accuracy`（`export.metrics_from_row`；`write_batch` 的行多了 `cross_track_p95_m` / `altitude_p95_m`
  两列，旧记录目录做不了子集）；`gate_grid` 的 decisive 看两项主指标、只认恰好两 seed；`cell_reading` 单独成函数；队列的格名来自配置（与 gate 表一致）、PID 文件退出即删、
  有活着的 PID 拒绝再起；失败方式：航向用 chart track、只数跑道入口前的中线穿越、加 `first_aligned_s` 与开始预测时的 to-go、`FAILURE_MODES_BLOCK` 登记进 publisher。
- 冒烟（09-18 S60_nt 臂 24 架、从第 59 行起）：10 架雷达引导没穿越 → established-short 1、overshoot 1、passed-abeam 8；子集 summary 能过 picker 的 accuracy 读取。
- intents 登记：`two_tier_v3_grid_20260918`（40 runs + 每 run 4 个 reading 变体 `@lockstep-none[-12km|-8km|-6km]`）；失败方式子集发布时再加 `@failure-<mode>` 变体。

## 3. 测试与环境

- 单文件测试：`timeout 600 conda run -n aeroviz --no-capture-output python -m pytest 4dTrajectory/ts_transformer/tests/<file> -q --import-mode=importlib`
- 全套 ~7 min，前台 600 s 或 detached；`test_architecture.py` 管 import 方向。测试不得触及 live 根目录（写根全进 tmp）。
- 永远 `conda run -n aeroviz --no-capture-output python …`。
- 主树自由（没有 campaign 在跑）；实验从 detached worktree 跑（`.claude/worktrees/manoeuvre-runs`，launch 前 `git -C … checkout --detach <sha>`）。

## 4. 本文维护约定

每完成一个开发项：更新 §1 状态与提交号；新查到的代码事实进 §2；待定项解决后从 §2 删掉并在 §1 注明。

## 5. 启动清单（阶段 A 的，已按此执行完；留作阶段 B 队列的样板——B 的命令、目录、读数名都不同，照 §5.2.5 / §8.2 另写）

**状态（2026-09-18 晚写，已过时）**：主树干净，最新提交 5079bb4；runs worktree `.claude/worktrees/manoeuvre-runs` 在 5079bb4（launch 前 `git -C .claude/worktrees/manoeuvre-runs status --short` 必须为空，
`git -C … log --oneline -1` 必须是主树上最新的、含阶段 A 代码的提交；不是就 `checkout --detach <sha>`）。20 份 cohort 已在
`4dTrajectory/outputs/KRDU/experiments/two_tier_v3_grid_20260918/cohorts/`。intents 已登记。磁盘 2026-09-18 晚 13.5 GiB 空余（需 ≥ 3 GiB，40 臂约 2 GiB + 读数）。

**启动前检查**（一条一条做，任何一条不满足就停下问用户）：
1. `git status --short` 为空；`git -C .claude/worktrees/manoeuvre-runs status --short` 为空且在最新提交。
2. `nvidia-smi --query-compute-apps=pid --format=csv,noheader` 没有别的训练在跑；`ps -p $(cat …/two_tier_grid_queue.pid)` 不存在或已死（文件正常情况下不存在）。
3. dry-run 无误：`conda run -n aeroviz --no-capture-output python .claude/worktrees/manoeuvre-runs/run_ts.py two_tier_grid_queue --arms
   4dTrajectory/ts_transformer/docs/experiments/two_tier_v3_grid_arms.json --campaign 4dTrajectory/outputs/KRDU/experiments/two_tier_v3_grid_20260918 --airport KRDU --dry-run`
   （从 worktree 的 run_ts.py 跑；`--arms` 相对 worktree 根解析；campaign 目录是绝对的主树路径，两处一样）。应列 20 个 CELL、每格 1 train + 8 lockstep + 1 gate，全是 todo。

**启动命令**（在主树目录执行，程序来自 worktree；`nohup setsid`，日志与 PID 放 campaign 目录）：
```
cd /home/supercomputing/studys/thesis
W=.claude/worktrees/manoeuvre-runs
C=4dTrajectory/outputs/KRDU/experiments/two_tier_v3_grid_20260918
nohup setsid conda run -n aeroviz --no-capture-output python $W/run_ts.py two_tier_grid_queue \
  --arms 4dTrajectory/ts_transformer/docs/experiments/two_tier_v3_grid_arms.json \
  --campaign $C --airport KRDU > $C/two_tier_grid_queue.log 2>&1 &
```
队列自己写 `$C/two_tier_grid_queue.pid`（退出即删）。记下 `$!` 与开始时间。

**启动后立刻**：
- Monitor 盯 `$C/two_tier_grid_queue.log`，匹配 `CELL .* complete|STOP:|queue done`；到期续（每 30 min）。
- 每格完成：读 `$C/gate/after_<cell>/grid_gate.txt`（该格两 seed 的 n、established 全部 / 雷达 / 直线、flyable、雷达 ADE、FDE）和
  `$C/lockstep/<arm>/L-1/manoeuvre_lockstep.txt` 的 `e_track p50 by round` 一行，发 PushNotification（格名 + 两 seed 的 established 全部/雷达 + n）。
  读数写进 `docs/2026-09-18_two_tier_v3_results.zh.md`（新建，结构按 v3 §7 M-A1 的交付：20 格 × 2 seed 表，两种读数各一张；每格 n）。
- `STOP:` 出现：不重跑（D11），看日志最后 40 行，发通知说明哪一步、rc 多少，等用户。一个"目录已存在"的失败多半是上一步创建了目录才死：按 E8 移成 `<dir>.aborted-<UTC>`，报告后等用户说再起。
- 可选：一个 opus queue agent 只做"看日志、报数字"，不做决定、不改任何东西。
- 预计 40 臂 × 12–15 min ≈ 8–10 h GPU；每格另加 8 份读数各 1–2 min；顺序 Δ 升序、L 升序（20 s 的四格最先出）。

**链结束后（M-A1）**：
1. `$C/gate/after_L120_D120/grid_gate.txt` 的 verdict 就是 A1 的判定（40 臂齐才有；seed 线、eligible、leaders、winners、selected、decisive 都在 json 里）。
   另跑 `executor_grid_gate --reading 12km|8km|6km --out $C/gate/final_<reading>` 得读数 (b) 的三张表（这些只是信息，判定只看 L-1；比 (b) 时注意各格覆盖数不同，D3）。
2. 胜出格两 seed 带记录重飞：`manoeuvre_lockstep --executor $C/<arm>/checkpoint.pt --protocol none --write-records --out $C/lockstep_records/<arm>/L-1`
   （约 40 MB / seed）；用 opus agent 发布到 picker：`--reuse-prediction-dir` + `--category-variant lockstep-none`（intents 变体已登记）。
3. A2：`executor_failure_modes --lockstep $C/lockstep_records/<arm>/L-1 --out $C/failure_modes/<arm>`，两 seed；表 + 至多三条假设写进结果文档；
   发布每类子集前先在 intents 里加 `<arm>@failure-<mode>` 变体。然后停下，等用户选 A3。
4. 结果文档里每个数都带配置（L、Δ、seed、读数 (a)/(b)、n）；不引用 09-18 的数当基线。

**不做的事**：不删不覆盖任何产物；不动 `manoeuvre_tok_20260918/`；队列跑着时不改 worktree、不在主树里编辑臂文件或 intents；失败不自动重跑；不跑任何带第二层的臂。

**用户可能要改的两个参数**（改了要重新 review 并提交后再跑）：读数 (b) 要求开始行后有 Δ 秒真值（`lockstep.from_remaining_path` 里 `min_future_s=effective_min_future_s(config)`）；
seed 线取 p75（`gates.GRID_SEED_LINE_QUANTILE`）。

## 6. 阶段 A 跑完之后（2026-09-19）

- 队列 2026-09-18T22:49:33Z 起、2026-09-19T08:44:21Z 止（9 h 55 min），中间在 L30_D120_s1337 的 6 km 读数处 STOP 一次（Δ = 120 s 时 6 km 行之后没有 120 s 真值，整个 cohort 无一行；D3 的结构性空覆盖），Δ = 120 的四格去掉 6 km 读数后续跑（`--readings L-1 12km 8km --cells …`），没有重跑任何一步。
- 读数、判定、条件、A2 都在 `docs/2026-09-18_two_tier_v3_results.zh.md`（提交 00ec6e8）：**M-A1 判定 L120_D20 胜出（decisive）**；L = 30 在 Δ ≤ 30 的格选模落在第 1 epoch（初始网络），八格（L ≥ 60、Δ ≥ 30）选模贴 180 epoch 预算；胜出格两 seed 带记录重飞在 `<campaign>/lockstep_records/<arm>/L-1/`，A2 在 `<campaign>/failure_modes/<arm>/`。
- 没做成的：`executor_grid_gate --reading 8km|6km`（`gates.cell_reading` 要求雷达引导层非空，这两个距离点上所有格都是空的）——若要这两张表，`cell_reading` 得允许雷达层为空（返回 None），改代码要 review。
- 下一步由用户选 A3（v3 §8）。两个可能要改的计划参数在 §5 末尾；再加两个由结果引出的：L = 30 的选模锚点，Δ ≥ 30 各格的 epoch 预算。
- 读数 (c)（2026-09-19 11:08–11:40Z，提交 1e0c6aa，D28）：所有格从第 59 行开始预测，40 份 `lockstep/<arm>/row59/`，门表 `gate/final_row59/`。结论（结果文档 §7）：同一起点下 L ≥ 60 各格差在 seed 线内，
  L = 120 在 (a) 里的领先是起点效应；L ≤ 60 里最好的是 L60_D20（0.786 / 0.785）；各 Δ 行内四个 L 格的航班集合完全一致。论文里的 L 结论按 (c) 写。
- A4 采样间隔消融（2026-09-19 列入计划 §5.5 / D29，未排期，未建臂）：dt ∈ {2, 5, 10} s × 两 seed，其余同 L120_D20；每个 dt 一份声明与 cohort（GRID_AXES 不含 dt_s）；读数 (a) 与 (c)（t* = 120 s，行号 60 / 24 / 12）。用户说"跑"之后才建臂。

- 2026-09-19：计划 v3 补了阶段 A 的结论与索引（§5.1 A1 判定与读数 (c) 的改写、§5.2 A2 答案与 H1–H3、§5.3 A3 = A3-a receding / A3-b N₁=4，未排期；§7 M-A0…M-A2 标完成），并写了阶段 B（§5.2、§6.2、§7.2、§8.2、§9.2：执行器 L60_D20，token 跨度 S ∈ {20, 60} s 消融，K16 学习码本主线、指令词表可选，先读真值 token 上限（门 B1）再端到端相对门（门 B），B-dev1…7，决定 D30–D44）。**未签字、未开发、未跑**：用户审 §5.2、§6.2、§7.2、§8.2、§9.2 后才进 M-B0。
- A3 读完（2026-09-19 18:43Z，结果文档 §8）：A3-a（L60_D60 两臂只飞 20 s，`--execute-s 20`，提交 f89be15）雷达组一致变好但未超线、直线组变差；A3-b（N₁ = 4，campaign `two_tier_v3_a3b_20260919`）无收益。两者 A3 行都 FAIL，B 的执行器仍是 L60_D20。
  相对门 runner `executor_relative_gate`（D45）已写并用于这两个比较（产物 `gate/relative_a3a_*`、`two_tier_v3_a3b_20260919/gate/relative_a3b_*`）。
- 2026-09-19 晚（用户审阶段 B 时）：计划重排成每节一个大块、阶段为小节（6515cbf）；四种读法改用名字（真值 token / 先验 token / 先验读真值历史 / 无 token）；5.2.0 加了第二层两部分的说明、先验一段重写（4af760f）；B0–B3 按"为什么 → 做什么 → 怎么判 → 不过怎么办"重写、术语行逐词解释（55ae671）；B 里的第 59 行读数删掉（961025f）。
  相对门 runner review 修完提交（01f29dc）；A3 结果与 A3-a 代码提交（f89be15、d76ed64）。记忆新增 `plain-wording.md`、`doc-structure.md`。

## 7. 阶段 B 开发（2026-09-19 晚起；用户答复：对照臂做、B0 先跑、Q3 待答；"最大程度利用时间，不干等实验"）

### 7.1 开发项与状态

| # | 开发 | 状态 | 提交 |
|---|---|---|---|
| B-dev0 | `manoeuvre_lockstep --cohort <development_cohort.json>`（`cohort_keys`：只飞该 cohort 在这个 split 里的航班，按 checkpoint 的顺序；cohort 里有 checkpoint 没有的航班就拒绝；payload 记 `cohort`） | 完成 | 4cd11ea |
| B-dev1 | 字段 `manoeuvre_token_s`（S，0 = 视界）与 `manoeuvre_token_step_s`（闭环里每次飞多长，0 = 视界）；`token_span_s / token_step_s / token_hold`；校验（S ≥ 视界、步长 ≤ 视界、都是 dt 的整数倍、步长是积分步的整数倍、S 是步长的整数倍）；码本 `segment_s` = S；`plan_token`、码本导出、先验、lockstep、码图谱都改读 S；命名 `tok-s=` / `tok-step=` | 完成（S = 0 时与 09-18 行为逐位相同，测试钉住） | 4cd11ea |
| B-dev2 | 训练时 token 段内位置 φ 的抽样（+ Q3 若采纳：段放不下时用上一段的码） | 完成（φ 抽样 + D48 的上一段规则；训练集建窗口时整体核一遍） | 4cd11ea |
| B-dev3 / 3′ / 5 | lockstep：同一 token 连用 S/步长 轮、带 token 读法下每轮只飞步长秒、先验落地只记时刻不停飞 | 完成（`round_step_s`、连用、按段编码、`--prior-landing-ends-flight`） | 4cd11ea |
| B-dev4 | `executor_relative_gate`（已有）+ **门 B1 一行**（`verdicts.b1`：任一项两 seed 都超线 + fully flyable，不要求其余项不差） | 完成 | 4cd11ea |
| B-dev6 | `two_tier_b_queue --arms --campaign --airport [--groups] [--dry-run]`：按组（配置 × 词表，两 seed）串行——train → 每臂码本导出、真值 token 读数（带记录）、失败方式、码图谱 → 门 B1 → 过则每臂先验、先验 token（带记录）/ 先验读真值历史读数、失败方式 → 门 B；先验步骤在运行时读 `gate/b1_<组>/relative_gate.json` 决定跑不跑；PID `<campaign>/two_tier_b_queue.pid` | 完成（第 0 步自己飞基线；真实臂文件 dry-run 通过） | 4cd11ea |
| B-dev7 | 臂文件 `docs/experiments/two_tier_v3_b_arms.json`（12 臂：S20 / S60h60 / S60held × K16 / cv × 2 seed，`stage_b` 块给队列：各配置的基线读数模板、seed 线来源、码本目录模板）；intents：campaign `two_tier_v3_b_20260919` 12 run + 36 读法变体，网格加 `L60_D20_s*@lockstep-none-b` | 完成 | 4cd11ea |
| B0 | 基线重读 + 失败方式表 | 19:53Z 提前跑的那次用了没写完的代码（payload v2），已删；改为队列第 0 步（`baseline_steps`：L60_D20 与 L60_D60 只飞 20 s，各两 seed，带记录 + 失败方式） | — |

### 7.2 代码事实

- 臂名 `<配置>_<词表>_s<seed>`；组 = 去掉 `_s<seed>`；配置名与 `stage_b.configurations` 的键对应，每臂的 `configuration` 字段写明；overrides 重述该配置的数字（测试 `test_two_tier_v3_b.py` 钉住 D30–D36、D38、D41、D43）。
- 码本目录 `4dTrajectory/outputs/codebooks/two_tier_v3_b_20260919_<臂>/`（09-18 的位置约定）；读数 `<campaign>/lockstep/<臂>/{C,A,A-truth}/`；失败方式 `<campaign>/failure_modes/<臂>_{C,A}/`；码图谱 `<campaign>/atlas/<臂>/`；先验 `<campaign>/priors/<臂>/prior.pt`；门 `<campaign>/gate/{b1,b}_<组>/`。
- 队列的基线由队列自己飞（第 0 步，臂文件 `stage_b.baselines`，只飞所选组要比的那些）：`L60_D20`（S20 与 S60held 的对照）→ `<campaign>/baseline/L60_D20_s{seed}/L-1`；`L60_D60_exec20`（`--execute-s 20`，S60h60 的对照）→ `<campaign>/baseline/L60_D60_exec20_s{seed}/L-1`；失败方式 `failure_modes/<基线>_s{seed}_none/`。seed 线 → 网格 `gate/after_L120_D120/grid_gate.json`（0.078 / 0.184 / 150 m，数值来源，D39）。启动前检查 checkpoint、cohort 文件、seed 线三项指标的 p75 都在。
- 相对门 `executor_relative_gate` 与失败方式脚本只认本版代码的 `LOCKSTEP_SCHEMA`（v3），别的 schema 按名字拒绝；分层表与 `gates.cell_reading` 严格读 v3 的列（`executed_s` 早先的 `.get` 兼容一并去掉）。阶段 A 的门与失败方式产物是当时的代码写的，不重跑；`executor_grid_gate` 因此不能再在阶段 A 的 v2 读数上重跑（用户 2026-09-19 的原则）。
- B cohort 文件 = 网格 `cohorts/L60_D60/development_cohort.json`（D47）；每臂不另写 `development_cohort`，文件级一条。
- 一份基线读数约 1.5 min（1404 架，19:53Z 那次量的），记录约 250 MB / seed；四份基线 + 失败方式约 1 GB。
- 相对门的 `_pb` / `_pc` 未用变量是 pyflakes 早先就有的提示，不是本轮引入。

### 7.3 待用户定 / 待办

- ~~Q3~~ 用户 2026-09-19 晚定：按建议（D48）——锚点后 ≥ Δ、token 段放不下时用上一段的码；臂文件 `random_train_anchor_min_future_s = 20`，代码 `plan_token.token_span_start_s`。
- B-dev0/1 的 opus review 结果落地后修、提交；再做 B-dev2、B-dev3/3′/5；每项 review、提交；最后 worktree 移到该提交，dry-run，等用户说"跑"。
- 19:53Z 那次 B0 的两份记录已发布到 picker（`experiment_l60_d20_s<seed>_…_lockstep-none-b_val`，182 → 184 类，来源目录已删，轨迹与重飞的相同）；队列重飞后的基线与失败方式子集等阶段 B 读数一起发（变体 `@lockstep-none-b`、`@lockstep-none-exec20-b` 已登记；子集发布时加 `@fail-<mode>`）。

### 7.4 阶段 B 队列的启动清单（M-B0′ 之后）

**已启动 2026-09-19T21:33:10Z、用户 2026-09-19T23:24:54Z 叫停**（`kill -TERM` 进程组，当时在 S60held_K16_s1337 的先验 token 读法第 1 轮，无残留目录；跑完的：基线、S20_K16 整组、S60h60_K16（门 B1 FAIL）、S60held_K16 的训练 / 真值 token 读数 / 门 B1（PASS）/ 先验 s1337；未跑：S60held 的先验 token 读法与门 B、三组指令词表）。停下的原因：用户要全面审计划——执行器只在真值历史上训练、从未见过自己飞出的历史（09-18 计划 §2.7 第四步(2) 有此步，v3 漏掉）。

**已启动 2026-09-19T21:33:10Z**（用户："跑"）：PID 181477，程序来自 worktree @ 4cd11ea，日志 `<campaign>/two_tier_b_queue.log`，PID 文件 `<campaign>/two_tier_b_queue.pid`（退出即删）；7 个 GROUP（baselines + 6 臂组）。主树可继续改文档（队列不从主树跑）。

1. `git status --short` 为空；`git -C .claude/worktrees/manoeuvre-runs status --short` 为空且 `checkout --detach` 到含阶段 B 代码的最新提交。
2. `nvidia-smi --query-compute-apps=pid --format=csv,noheader` 没有别的训练；`<campaign>/two_tier_b_queue.pid` 不存在；空余 ≥ 3 GB（12 臂约 0.6 GB + 记录约 2.5 GB）。
3. dry-run：`conda run -n aeroviz --no-capture-output python .claude/worktrees/manoeuvre-runs/run_ts.py two_tier_b_queue --arms 4dTrajectory/ts_transformer/docs/experiments/two_tier_v3_b_arms.json --campaign 4dTrajectory/outputs/KRDU/experiments/two_tier_v3_b_20260919 --airport KRDU --dry-run`，应列 `GROUP baselines`（4 份 lockstep none + 4 份失败方式）再 6 个臂组、每组 1 train + 8 读数步 + 门 B1 + 9 个 `[if gate b1 passes]` 步，全是 todo。
4. 启动（主树目录，程序来自 worktree）：
```
cd /home/supercomputing/studys/thesis
W=.claude/worktrees/manoeuvre-runs
C=4dTrajectory/outputs/KRDU/experiments/two_tier_v3_b_20260919
nohup setsid conda run -n aeroviz --no-capture-output python $W/run_ts.py two_tier_b_queue \
  --arms 4dTrajectory/ts_transformer/docs/experiments/two_tier_v3_b_arms.json \
  --campaign $C --airport KRDU > $C/two_tier_b_queue.log 2>&1 &
```
5. Monitor 盯 `$C/two_tier_b_queue.log`，匹配 `GROUP .* complete|GATE B1|STOP:|queue done`；每组完成读 `gate/b1_<组>/relative_gate.txt`，写进结果文档 §10；`STOP:` 不重跑，按 E8 处理后等用户。
6. 预计：第 0 步 4 份基线约 8 min；训练 12 臂 × 12–15 min ≈ 3 h；每臂真值 token 读数 + 失败方式 + 图谱约 3 min；过门的组再加先验 3–5 min + 两读数 4 min / 臂。
