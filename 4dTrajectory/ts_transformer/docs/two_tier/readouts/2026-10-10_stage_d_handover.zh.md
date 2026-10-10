# Stage D 交接（2026-10-10 16:40）

接手的 agent 先读本文件，再读 `notes/stage_d.md`（设计方的命令）和
`readouts/2026-10-06_stage_d_implementation_log.md`（stage D 的日志，最新是 §21）。本文件之后的日志 §22 还没写，见第 8 节。

## 0. 现在的状态（一句话）

**没有任何活动在跑。** MC7 由用户在 16:31 喊停（第 5 轮做了一半），学习率对照已跑完，梯度一致性读取因内存被我停了。
所有决定都等用户：MC7 是否续跑、是否先改奖励/信用分配设计、是否读梯度一致性。**没有用户明确命令，不要续跑或启动任何活动**
（用户 16:30："谁让你开始mc7的？"——我把"拍在那个空档"当成了续跑许可，这是错的）。

## 1. 代码与分支

| 位置 | 内容 |
|---|---|
| 代码分支 `dev-multi-control`（worktree `.claude/worktrees/multi-control`，干净） | HEAD `a2f51add`；`dev-two-tier`（`ca00c334`）可快进到它，合并由用户做 |
| 运行 worktree `.claude/worktrees/run-mc7` | detached `c7c35b67`，MC7 和学习率对照都从这里跑；里面有数据软链（`data`、`4dTrajectory/outputs`、`trajectory_data_process/outputs`、`aeroviz-4d/public/data/airports`），删除 worktree 前先 `unlink` 软链 |

今天 `dev-multi-control` 上的提交（`ca00c334` 日志只记到 `e2b69736`）：

| 提交 | 内容 | 测试 / 审核 |
|---|---|---|
| `ada6a79d` `7893d787` `82dd7695` | D190：联合循环的缓存余量；共享前缀，`front_attention` copied/merged | 已记在日志 §20 |
| `5b4cd739` | D191：GPU + CPU 两种 speaking worker，规则 `cpu_batches`/`cpu_period` | 日志 §21 |
| `e2b69736` | D191：GPU 批超出份额就留到最后、以更少并发重说（`Speakers.deferred`） | 日志 §21 |
| `c7c35b67` | 续跑可以改 CPU 规则（`post_train.RESUME_MAY_CHANGE`）；读数只在 speak device 上（`Speakers.read`）；`post_ceiling`/`post_diagnose` 去掉 `--cpu-workers` | 全套 2,115 passed；审核无 S1/S2 |
| `35a6e072` | R69 `multi_compare`：起点读数（`--read-start`，同次重读最后一轮做行为核对）+ 各轮对比表（降落率、每架奖励、失去间隔、KL、数据项） | 6 个测试；审核 2 个 S2 已修，第二轮无 |
| `a23a49d7` | R70 `pass_gradients` + `post.loss.update_gradients`：一次更新的梯度按奖励项/拉回项/数据项拆开 | 审核 1 个 S2 已修，第二轮无 |
| `a2f51add` | R70 v2：各次更新之间梯度方向的一致性（`Consistency`：两两余弦、奇偶两半余弦、合成比） | 审核 1 个 S2 已修，第二轮无；**还没在真实数据上跑过** |

`runners.md` R69、R70 已在代码分支里。全套测试最后一次是 `c7c35b67` 时跑的（2,115 passed, 1 skipped）；之后三个提交只跑了改动模块的测试。

## 2. MC7（`4dTrajectory/outputs/POOLED/multi/multi_train_20261010`）

设置：起点 stage C `post_seg60_20261007` 第 5 轮，种子 2035，8 轮，backward 分支，`front_attention merged`，学习率
prior 1e-5 / traffic 1e-4（全程不变，用户 09:27 决定"先不改"），AdamW，weight decay 0.01，无梯度裁剪。

| 轮 | 说话配置 | 批数 | 说话用时 | 整轮 | select 降落率 | 每架奖励 | 失去间隔 | KL |
|---|---|---|---|---|---|---|---|---|
| 起点（一次性读，未核对） | – | – | – | – | 3660/4492 = 81.5% | 0.8089 | 16.4% | – |
| 0 | 5 GPU | 224 | 131.0 min | 145.3 | 3720/4492 = 82.8% | 0.8221 | 15.2% | 0.0456 |
| 1 | 5 GPU | 214 | 139.0 | 153.7 | 82.1% | 0.8140 | 15.9% | 0.0491 |
| 2 | 3G+3C，6/13 | 218 | 66.1 | 82.6 | 82.9% | 0.8217 | 14.6% | 0.0485 |
| 3 | 3G+3C，5/9 | 224 | 62.1 | 78.2 | 82.5% | 0.8175 | 15.0% | 0.0487 |
| 4 | 3G+4C，7/11 | 212 | 57.2 | 74.2 | 82.8% | 0.8224 | 14.7% | 0.0530 |
| 5 | 3G+3C，5/9 | 217 | **停在 92/217** | – | – | – | – | – |

- 读数来自各轮 `round_*/round.json` 的 `selection_readout`，用 `multi_compare` 输出（`handover_20261010/mc7/compare_r01234.json`）。
  **日志行里的 `readout` 是每架奖励 `reward_mean`，不是降落率**（我曾经报错过）。
- 训练窗口第一遍说话的降落率（`round.json` 的 `speaking.outcomes`）5 轮都在 80.9–81.6%，失去间隔约 16.5%：**训练窗口上没有进步**。
- 第 5 轮：92 批的 `record_*.json` 和 `groups_*.pt` 在盘上；续跑会读回它们（脚本 `handover_20261010/mc7/resume_mixed.sh`，
  3G+3C 5/9，不带 `--restart-round`）。**续跑要用户明确同意。**
- 各次续跑留下的 `round_N.aborted-*` 是换规则时移开的少量批（或空），不要删前先确认。

## 3. 学习率对照（`4dTrajectory/outputs/POOLED/multi/multi_train_lr3_20261010`，已完成）

用户 13:27 下令。MC7 同起点、同种子、同窗口，只把学习率乘 3（3e-5 / 3e-4），跑 2 轮，3G+3C 5/9。

| | 降落率 | 每架奖励 | 失去间隔 | KL | 被裁剪的词 |
|---|---|---|---|---|---|
| 第 0 轮 1×（MC7） / 3× | 82.8% / 82.0% | 0.8221 / 0.8148 | 15.2% / 15.6% | 0.0456 / 0.0523 | 0.83% / 2.21% |
| 第 1 轮 1×（MC7） / 3× | 82.1% / **80.4%** | 0.8140 / **0.7985** | 15.9% / 16.2% | 0.0491 / 0.0638 | 0.67% / 2.40% |

训练窗口第一遍：第 0 轮两边 5440 / 5441 落地（证明样本相同）；第 1 轮 81.6% / 81.3%。
**结论：步子太小不是原因，3 倍更差，两轮后低于起点。** 对照表 `handover_20261010/lr3/compare_r01.json`。

## 4. 梯度诊断

- **梯度构成**（R70 v1，`handover_20261010/grad/mc7_r5.json`）：MC7 第 5 轮前 46 批（只有 300 s 和 600 s 窗口），第 4 轮模型，
  29 个更新。先验参数：奖励项占更新方向 55%（中间一半 41–69%），数据项 45%，拉回 ≈0，奖励与数据项余弦 0.005；交通参数：奖励项 ≈100%。
  **奖励信号没有被别的项盖过。**
- **一致性**（R70 v2）：**没读成**。16:22 开始，16:29 我停掉——它一个进程占 7.1 GB（我估 3 GB），挤 MC7 内存，而且和 MC7 抢 CPU 很慢。
  现在机器空，第 5 轮 92 批分支组在盘上，是读它的好时机（要用户同意）：
  `python run_ts.py pass_gradients --campaign 4dTrajectory/outputs/POOLED/multi/multi_train_20261010 --round 5 --out <新文件>.json --every 2 --threads 8`
  （从 `multi-control` worktree，干净树；约 7 GB 主机内存）。**MC7 第 5 轮一旦跑完，它的分支组会被删。**

## 5. 速度（同一批：MC7 第 0 轮第 86 批，1200 s，4 窗口 24 架，起点模型，机器上无其他任务）

| | 整批 | speaker | 窗口循环 | 执行器 | 其他 |
|---|---|---|---|---|---|
| CPU 单线程 | 66.7 s | 32.1（48%） | 16.2（24%） | 12.8（19%） | 5.5 |
| GPU 单进程 | 63.6 s | 17.1（27%） | 20.8（33%） | 21.1（33%） | 4.6 |

文件 `outputs/POOLED/multi/speed_mc7_r0_b86_{cpu,cuda}_20261010/speed.json`；10-09 的旧测量（MC6，grid 分支）
`speed_cpu_b86_d186b_20261009`（130.2 s）、`speed_mc6_r0_b86_20261009`（166.5 s）。backward 比 grid 快约一半。
执行器和窗口循环在 GPU 上比 CPU 慢（小运算，启动开销大）。可提速方向（未做，需用户/设计方定）：GPU worker 的执行器和窗口循环放 CPU；
窗口循环里逐架的 Python 循环（交通 token、速度掩码、间隔判定）改成整批计算。

## 6. 资源与 worker 的经验（都已实测）

- GPU 已满（99%）：5 个 GPU worker 不比 3 个快，份额小反而多出"超份额重说"的浪费（第 0/1 轮 30/40 批）。
- 3G+3C 混合比纯 GPU 快约一倍；规则按两边实测速度调（5/9 时两边同时说完）。
- **4 个 CPU worker 不可行**：第 4 轮 12:40 一个 CPU 批让 worker 超过 4.7 GB，swap 被挤满（7.8 GB）。3 个 CPU worker 在对照活动的第 0 轮也挤满过一次 swap（14:00–14:08）。
  CPU 批没有"超份额重说"兜底，单批主机内存没有上限。
- swap 满时 `MemAvailable` 反而变高，**要同时看 swap**（内存记录脚本 `handover_20261010/mc7/memlog.sh`，监控阈值：可用 < 2.5 GiB 或 swap > 7 GB）。
  轮与轮之间重启进程能清空 swap。
- 用户同意后我关掉了 10-04 起闲置的后端（端口 8781）和 vite（5181），都是 job 90ff3246 的，swap 底数少了约 1.4 GB。

## 7. 等用户决定的事

1. MC7 续跑剩下的第 5–7 轮，还是先改设计再开新活动。
2. 是否现在读梯度一致性（第 4 节）。读出来若奖励梯度各次更新几乎不相关，就该改奖励/信用分配设计（分支组大小、优势摊到哪些行、基线）。
3. 起点的正式读数（`multi_compare --read-start`，约 40 分钟 GPU，含重读最后一轮的核对）：表里的起点 81.5% 是一次性读的，未核对。
4. 速度方向的两个提案（第 5 节）。
5. requests_from_d 里还开着的 86（前缀分两块存）和 89–94（今天凌晨的读数）。

## 8. 还没写的文档（交给接手者）

- 日志 `2026-10-06_stage_d_implementation_log.md` §22：`c7c35b67`、`35a6e072`、`a23a49d7`、`a2f51add`；MC7 各轮；学习率对照；梯度构成；速度；资源经验（本文件第 2–6 节就是素材）。
- `multi_control.md` §0.3、`post_training.md` §0.3 的行；`requests_from_d_to_designer.md` 重写（加今天的读数和提案）。
- `docs/experiments/intents.json`：MC7 和学习率对照发布前要有条目。

## 9. 文件索引

- 活动：`4dTrajectory/outputs/POOLED/multi/multi_train_20261010`（MC7）、`multi_train_lr3_20261010`（学习率对照）。
- Profile：`outputs/POOLED/multi/profile_d190_merged_20261010`（GPU worker）、`profile_d191_cpu_20261010`（CPU worker）。
- 交接目录 `4dTrajectory/outputs/POOLED/multi/handover_20261010/`（/tmp 里的东西都复制在这里）：
  `mc7/`（各次启动脚本 `run.sh`、`resume_mixed.sh`，日志 `run.log`、`run_mixed*.log`，对比表 `compare_r*.json`，`memlog.sh`）、
  `lr3/`（脚本、日志、对比表）、`grad/`（`mc7_r5.json` 梯度构成，`run2.log` 停掉的一致性读取）、
  `start_read/start_readout.json`（起点一次性读数）、`gap/`（空档任务链）、`memlog_journal.txt`（全天每分钟内存和 swap）、
  `mixed_measure.json`（D191 混合实测）、一次性脚本 `fifo_check.py`（按完成时间倒推超份额批）、`start_readout.py`、`fit_cpu.py`、`d191_mixed.py`。
- 记忆：`~/.claude/projects/-home-supercomputing-studys-thesis/memory/stage-d-multi-control-branch.md`（按时间记了今天的每一步）。
