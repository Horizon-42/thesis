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
| A-dev1 | 网格臂文件：`docs/experiments/two_tier_v3_grid_arms.json`，40 臂，键 `L<L>_D<Δ>_s<seed>`，每臂自己的 cohort 路径；`frame_ablation --only KEY…` 只跑指定臂；测试钉住 D1/D2/D4–D8/D10 | 完成（review 1 通过并修） | fa0426f（文档 4dd2082） |
| A-dev2 | cohort：`plan_cohort --arms <臂文件>` 一次加载、按格写 20 份 `cohorts/L<L>_D<Δ>/development_cohort.json`（已写出，n 表见 §2）；`frame_ablation` 每臂用自己的 cohort；`rebuild_cohort` 不用改（读 checkpoint 自带的 split） | 完成（review 1 通过并修；cohort 已写出） | fa0426f（文档 4dd2082） |
| A-dev3 | 第一次预测在第 L−1 行：臂文件 `anchor_floor_index = 0`，测试钉住 `default_anchor == seq_len − 1` | 完成（含在 dev1 测试里） | fa0426f（文档 4dd2082） |
| A-dev4 | `manoeuvre_lockstep --anchor-remaining-km 12|8|6`：`lockstep.from_remaining_path` 把每架从 bin 行（`anchor_grid.bin_anchor`：剩余路程最接近 X km 的行）切起，使该行成为固定锚点；协议 `none` 不再要 codebook；payload schema v2（`first_prediction` 块、每行 `first_prediction_row`；轮次记录叫 `rounds`，次数叫 `predictions`） | 完成（review 通过并修，冒烟通过） | fa0426f（文档 4dd2082） |
| A-dev5 | 训练锚点全随机：臂文件 `random_train_anchor_l1_share = 0.0`，测试钉住 | 完成（含在 dev1 测试里） | fa0426f（文档 4dd2082） |
| A-dev6 | `executor_failure_modes --lockstep <dir> --out <dir>`（`manoeuvre/failure_modes.py`）：course 坐标系、六类失败方式、每类记录子集 `records_<mode>/`（`export.copy_record_subset`） | 完成（review 通过并修，冒烟通过） | fa0426f（文档 4dd2082） |
| A-dev7 | `executor_grid_gate --campaign --arms --out [--reading L-1]`（`gates.gate_grid`）：整表 + 胜出格；seed 线 = 本网格 seed 差的 p75 | 完成（review 1 通过并修） | fa0426f（文档 4dd2082） |
| A-dev8 | `two_tier_grid_queue --arms --campaign --airport KRDU [--readings …] [--cells …] [--dry-run]`：每格 train（`frame_ablation --only`）→ 8 份 lockstep 读数 → 选格表；PID 文件 `<campaign>/two_tier_grid_queue.pid` | 完成（review 2 通过并修，dry-run 通过） | fa0426f（文档 4dd2082） |

顺序建议：dev2 → dev1 → dev3 → dev5（训练侧，一起冒烟）→ dev4 → dev7 → dev6 → dev8。

**M-A0 达成（2026-09-18 晚）**：全部八项提交在主树（fa0426f 代码、4dd2082 文档、5079bb4 把 lockstep 里的 "ask" 改名为 prediction），全套 ts 测试 1273 通过，两轮 opus review 的发现都已修；runs worktree `.claude/worktrees/manoeuvre-runs` 已移到 5079bb4。**等用户说"跑"（M-A0′）再启动；启动清单在 §5。**

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
- readout 的臂发现按 `S<seg>_(K\\d+|cv|nt)_s<seed>` 形状（01e5286）；新网格的臂名要定（建议 `L<L>_D<Δ>_s<seed>`，并让发现规则认识它，或阶段 A 不用 `manoeuvre_readout`）。

- **产物布局（dev7/dev8 定）**：`<campaign>/<arm>/`（训练）、`<campaign>/lockstep/<arm>/<reading>/manoeuvre_lockstep.json`（reading ∈ L-1、12km、8km、6km）、
  `<campaign>/gate/after_<cell>/grid_gate.json`（每格完成后的整表；40 臂齐后才有 verdict）、`<campaign>/cohorts/<cell>/development_cohort.json`。
- `frame_ablation` 的磁盘估算：predict 臂 400 MB、train-only 臂 100 MB（09-18 实测 45 MB）；整文件 40 臂 dry-run 现在报需 3.9 GiB，空余 13.5 GiB。
- opus review 1（dev1–5）已做完并修：cohort 加 build 的长度门（`dataset.minimum_build_samples`）；`plan_cohort --arms` 要求各格在 `GRID_AXES`
  之外完全一致、拒绝任何 CLI 配置 flag、cohort 目录名唯一；`frame_ablation --only` 仍校验全部臂、`null` cohort 继承文件级；lockstep 拒绝 `control_horizon_s = 0`。
  未改的提醒：距离读数 (b) 的覆盖随 Δ 变（写进 D3）。
- opus review 2（dev6–8）已做完并修：记录子集自带重算的 `accuracy`（`export.metrics_from_row`；`write_batch` 的行多了 `cross_track_p95_m` / `altitude_p95_m`
  两列，旧记录目录做不了子集）；`gate_grid` 的 decisive 看两项主指标、只认恰好两 seed；`cell_reading` 单独成函数；队列的格名来自配置（与 gate 表一致）、PID 文件退出即删、
  有活着的 PID 拒绝再起；失败方式：航向用 chart track、只数入口前的中线穿越、加 `first_aligned_s` 与开始预测时的 to-go、`FAILURE_MODES_BLOCK` 登记进 publisher。
- 冒烟（09-18 S60_nt 臂 24 架、从第 59 行起）：10 架雷达引导没穿越 → established-short 1、overshoot 1、passed-abeam 8；子集 summary 能过 picker 的 accuracy 读取。
- intents 登记：`two_tier_v3_grid_20260918`（40 runs + 每 run 4 个 reading 变体 `@lockstep-none[-12km|-8km|-6km]`）；失败方式子集发布时再加 `@failure-<mode>` 变体。

## 3. 测试与环境

- 单文件测试：`timeout 600 conda run -n aeroviz --no-capture-output python -m pytest 4dTrajectory/ts_transformer/tests/<file> -q --import-mode=importlib`
- 全套 ~7 min，前台 600 s 或 detached；`test_architecture.py` 管 import 方向。测试不得触及 live 根目录（写根全进 tmp）。
- 永远 `conda run -n aeroviz --no-capture-output python …`。
- 主树自由（没有 campaign 在跑）；实验从 detached worktree 跑（`.claude/worktrees/manoeuvre-runs`，launch 前 `git -C … checkout --detach <sha>`）。

## 4. 本文维护约定

每完成一个开发项：更新 §1 状态与提交号；新查到的代码事实进 §2；待定项解决后从 §2 删掉并在 §1 注明。

## 5. 启动清单（用户说"跑"之后照此做；之前一步都不做）

**状态**：主树干净，最新提交 5079bb4；runs worktree `.claude/worktrees/manoeuvre-runs` 在 5079bb4（launch 前 `git -C .claude/worktrees/manoeuvre-runs status --short` 必须为空，
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
