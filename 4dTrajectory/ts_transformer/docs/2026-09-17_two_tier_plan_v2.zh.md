# 两重 Transformer 开发计划 v2（2026-09-17）

本文件取代 `2026-09-16_two_tier_transformer_feasibility.zh.md` 的设计与开发方案（那份文档改写了需求：
把低层做成整段预测、用一个不相干的测试否掉了高层；其 §3、§10.2、§10.4、§10.6 的测量数字仍可引用，
§10.9 是错的设计下的结果，只作记录；设计不再沿用）。

## 0. 需求（用户原文，2026-09-16 / 09-17）

> 短时间内用 iTransformer 预测控制参数（预测时间短，应该能大幅降低 ADE）；大尺度上把每个时间段作为 token，预测长期飞行计划；
> 大尺度流程再引入 graph 网络，用多机的预测段预测多机交互（优先级靠后）。
>
> 底层的航空动力学模型要能应用多种动力学，比如预测 load factor、thrust 和 bank 的，以及预测航迹角、thrust 和 bank 的（已有相关实验，可以改进）。
> 高层段 token 重写设计，不要被之前的 plan 实现影响；整体的航迹本身、起始点和结束点、整段的大体方向都可以作为 channel 计算注意力。

三层，从下到上：**L1 短时控制层** → **L2 段 token 计划层** → **L3 多机图层**（排后）。

## 1. 状态表（压缩 context 后从这里继续）

| 项 | 状态 |
|---|---|
| 分支 / 目录 | 开发分支 `dev-two-tier`，worktree `.claude/worktrees/two-tier`（数据目录软链到主树）；主树 `dev-two-tier-feasibility` 与它同在 15fb8f3，**用户合并**；runs worktree `two-tier-runs`（队列 agent，训练前移到已提交的 commit），pub worktree `two-tier-pub` |
| 保留的代码 | 控制契约注册表（77e1372，四种动力学一行一个、可直接切换）、航迹角下的 hook（87f46cc）、`tracker_lockstep` + `two_tier_gates`（c660600，结束规则与规则制导同时限）、`inference/receding.py`、`plan_conditioning` 的 token 机制、`anchor_floor_index`、`lead_time_error`、`chain_sensitivity` |
| 废弃的设计 | 旧文档 §4 的"改前提"论证；T1a 的整段跟踪器；T0(c) 作为高层的门；旧 plan 头作为高层；`control_horizon_curriculum_s`（已退役字段，与本计划的 `control_horizon_s` 无关） |
| 旧实验 | T0(b)/T0(c) 已发布（picker）；T1a/T2 留在 `4dTrajectory/outputs/KRDU/experiments/two_tier_t1a_20260916/`：**用户已批准删除**两个 lockstep 记录树（`lockstep_30s/records` 2.2 GB、`lockstep_30s_head/records` 2.4 GB）与三个 `*_pred_val`（各 222 MB），留 3 个 checkpoint、`gates*`、`tracker_lockstep.{json,txt}`；沙盒拒绝 `rm -rf`，命令见 §9，由用户执行 |
| 决定 | §8 的八项已由用户拍板（2026-09-17，"按建议来"）；开发中需要选择的按直觉选、次日汇报，不中断（§8 末"我替你选的"） |
| S0 文档 + 磁盘 | 本文件（1625dc3）；删除待用户执行 |
| S1.1 `control_horizon_s` | **完成** 29646ab + review 修正 1604b65（opus 6 项确认：anchor-grid 与 anytime 的未来下限跟 Δ、每 epoch 报告的终端速度/弧长几何读 Δ 内的真值、排除原因写明 Δ、审计里的下限是生效值、`intent=truth-join-duration` 拒绝、命名改 `ctrl-horizon=`、时长项为零写明） |
| S1.2 `plan_conditioning=waypoints` | **完成** ddfbf82（token 11 维、训练行/forecast 行/probe 一处分派、`PLAN_WAYPOINT_SEGMENT_S=30`） |
| S1.3 读数工具 | **完成** bea33af：`tracker_lockstep` 航路点来源 + 逐次 e + `--command-hook`（schema v3）；`two_tier_gates` 门 L1 + 漂移读数（schema v2，cohort 可为基线子集并计数）；`short_horizon_readout` 新 runner；`frame_ablation` `"predict": false`。全套 1318 测试通过 |
| S1.4 臂 + intents | **已写**：`docs/experiments/two_tier_l1_arms.json`（8 臂，dry run 通过）、intents `two_tier_l1_20260917`；campaign 待队列 agent 启动（runs worktree 先移到已提交的 commit） |
| S2 L2 段 token 计划头 | 未开建 |
| S3 联合 lockstep + 门 | 未开建 |

## 2. 架构

```
观测历史 ──┬─► L2 段 token 计划层 ──► 粗计划：未来 M 段的终点 (Δe,Δn,Δu)、到达段 ──┐
           │        每 30 s 在滚动历史上再问                                     │
           └─► L1 短时控制层 ◄──── 计划 token = 下 2 个粗航路点（相对位置、剩余时间）◄─┘
                    │ 短历史 H + 计划 token → 下 Δ 内 N₁ 段控制（契约可切换）
                    ▼
               动力学 rollout（Δ）── 只执行前 30 s ── 飞出的行接到历史 ── 再问
```

接口只有一个：**粗计划 = 未来若干段的终点序列**。L1 训练时用真值的粗计划（协议 C），部署时用 L2 的（协议 A）。

**坐标系（接口约定）**：L1 的通道与 token 都在阈值锚定的 ENU chart（`coordinate_frame=enu`，config 对 `plan_conditioning` 只允许它）；
token 的 (Δe,Δn,Δu) = 航路点的 chart 坐标减去**当前位置**（训练时是锚点行，lockstep 里是飞到的行）。
L2 的段特征按跑道航向旋转是特征工程，它的输出在接口处转回 chart 再给 L1。

## 3. L1：短时控制层

**输入**
- 历史：最近 H 秒的状态序列（现有 6 通道 e,n,u,ė,ṅ,u̇，2 s 采样）。H = 60 s 为主臂（`seq_len=30`），30 s 与 118 s（`seq_len` 15 / 60）为消融。
- 计划 token（`plan_conditioning` 的新取值 `waypoints`）：粗计划的下 2 个航路点，各 (Δe, Δn, Δu, 到达该点的剩余时间 τ)，
  按尺度归一（水平 5000 m、垂直 300 m、时间 60 s）+ 每点 1 位有效位 + 1 位在场位，共 11 维；缺席 token 全零。
  航路点在锚点后 k·Δ₂（Δ₂ = 30 s，`PLAN_SEGMENT_S`，L2 同一常量），k = 1..Δ/Δ₂；真值不到该时刻则该点无效。
  同步询问时 τ 恒为 30 / 60 s，字段留给不同步的部署（L1 在两次 L2 询问之间被问、调度器改时）。
- 到达时间不单独给：它由粗计划的段数决定，L1 只管接下来的 Δ。

**输出**：接下来 Δ = 60 s 的 N₁ = 6 段控制（每段 10 s，`uniform` 时长），rollout 恰好 Δ 长；lockstep 每次只执行前 30 s。

**动力学契约（可切换，现有注册表一行一个）**
| 取值 | 三列 | 说明 |
|---|---|---|
| `thrust-fraction` | δ=T/T_max, 坡度, 过载 | 历史默认，native32 的契约 |
| `specific-force` | n_x=(T−D)/W, 坡度, 过载 | 纵向闭环（N3） |
| `specific-force+path-angle` | n_x, 坡度, 航迹角目标 γ* | 纵向+垂直闭环（N7′，可飞 97.7 %） |

三种都跑（用户 2026-09-17 定）：过载族用 δ 与 n_x 两行，航迹角族一行；全部 `first-order-lag`。**改进（S5，条件性）**：横向也做成闭环——坡度列换成航向变化率目标 ψ̇*，RHS 按 `n sinφ = V cosγ ψ̇*/g` 解坡度；
只在 L1 结果显示横向误差主导时做。

**训练（新配置轴 `control_horizon_s = Δ`，默认 0 = 整段；命名 `ctrl-horizon=`）**
- 窗口规则：锚点之后的真值 ≥ Δ（`effective_min_future_s` 一处定义：`window_anchors`、anchor-grid 的档、anytime 曲线都取 `max(下限, Δ)`；随机锚点的 `random_train_anchor_min_future_s` 不得小于 Δ；不满足的航班不入 cohort，排除通知写明 Δ，审计里记生效的下限）。
- 锚点：所有臂 `anchor_floor_index = 59`，固定锚点与 native32（seq_len 60 的 L−1）和规则制导基线（anchor 59）对齐，H 消融也在同一锚点判；L1 的 val cohort 可能是基线的子集（Δ 后真值不足的被排除），门逐航班在 L1 的集合上配对并报缺的数目。
- 目标网格：N₁ 个节点在 (k+1)·Δ/N₁，真值按 2 s 采样插值；`final_time_s = Δ`；heading-rate 项的端点同一网格。
- 头：无时长头（不建、不训）；`cta_conditioning` 必须 off；`duration_head` 必须 point；`final_time_loss_weight` 必须 0（项恒为零）；模仿项未建（拒绝非零）。
- 损失只算 [0, Δ]：位置项、velocity 0.003、heading-rate 8、bank TV 1（A2b 配方）；`state_endpoint_loss_weight = 0`（臂文件置零，不是硬规则）。
- 选择指标 `fixed-anchor-common-grid-ade`：共同网格的真值跨度也是 Δ（`common_truth_at_anchors` 读同一个 `target_horizon_s`），即固定锚点处的 ADE[0,Δ]；每 epoch 的报告块（终端速度、弧长几何）对 Δ 内截断的真值（`series_within_horizon`）读。
- 记录：读数 runner 写的 predict 形状记录，其 summary 的 ade/fde 仍是 export 的整段记账（预测被 hold 到真值结束），只有 `coverage_ratio` 说明它；读数自己的 [0,Δ] 数字才是 L1 的数。
- 随机锚点（remaining-path-uniform，l1-share 0.5）；计划 token 按 50 % 遮掉；180 epoch 不早停；hook 关。
- 臂：3 契约 × 2 种子 = 6，加 H 消融 2（航迹角契约、种子 1337、H = 30 / 118 s），KRDU。每臂预计 < 1 h。campaign 只训练（arm 文件 `"predict": false`），记录由读数 runner 自己写。

**评估**
1. 短时读数（`run_ts.py short_horizon_readout`）：锚点集 = L−1 + 剩余路程 12 / 8 / 6 km 档（`anchor_grid`，未来 ≥ Δ），分层固定在 L−1。
   每个 L1 臂两列：**带 token**（真值航路点，协议 C，= e_track）与**遮 token**（协议 B）；对照 native32 在同一批锚点上的 rollout 切到 60 s。
   读 e(30)、e(60)、ADE[0,60]（1 s 网格，`lead_time_error` 口径），配对差。
2. lockstep 自身历史（协议 C，真值航路点按**绝对时间**读、相对飞到的位置）：`tracker_lockstep`，每架记录逐次询问的 e(30 s)（每条腿末端对真值），
   整段 ADE / 完全可飞 / 建立（`two_tier_gates` 现有读法，基线规则制导 1847 / 283 m，建立 0.879 / 0.998）。变体：receding、receding-no-plan、one-shot（对固定 Δ 头 = 整段 Δ 都执行）。
   门读数 hook 关；另跑一列 `barrier+speed-floor` soft 作交付形态旁读。

**门 L1（两种子，同一契约）**：
- 带 token 的 ADE[0,60] 雷达引导 < 125 m、直线 < 60 m（e_track）；遮 token 的 ADE[0,60] 不差于 native32 同锚点读数（短时预测能力没丢）；
- lockstep 整段雷达引导 < 1500 m、直线 < 250 m（好于规则制导）；完全可飞 ≥ 95 %；建立 ≥ 规则制导的 88 %；
- 漂移读数（不计入过否，触发 S5）：逐次 e(30 s) 的 p50 按询问序号排列，第 3 次及以后的 p50 ≤ 1.5 × 前 3 次的 p50，按分层；超过即"上升"。

## 4. L2：段 token 计划层

**段**：固定 Δ₂ = 30 s 一段（15 个间隔）。输入取锚点前最近 K_in = 4 段（60 个间隔 = 61 个采样，L2 自己的 `seq_len`，与 L1 的 H 无关）。
**K_in = 8 的消融删除**：旧文档 §10.2（切片序列时长 p50 296 s，240 s 历史加 60 s 未来把 cohort 砍到偏长航班）与 §10.6（238 s 对 118 s 回看两种子都无信息）两条测量都反对它。

**每段的 channel**（用户点名的三类 + 补充），全部相对锚点位置、按跑道系旋转，与航班无关：
| 组 | channel |
|---|---|
| 起点 / 终点 | 起点 (e,n,u)、终点 (e,n,u) |
| 大体方向 | 起点→终点方位的 (cos, sin)、下滑比 Δu/水平长 |
| 航迹本身 | 段内 5 个等距子采样点相对段起点的 (Δe,Δn)，路径长、平均地速、净航向变化 |
| 全局 | 到入口的距离与方位、跑道航向（现有 final-approach 上下文） |

**注意力**：两个臂共享同一套段特征。
- **A（主）**：iTransformer 反转——每个 channel 在 K_in 段上的序列是一个 token，注意力在 channel 之间（用户的"作为 channel 计算注意力"）。复用现有骨干，只换 `channels` 与 `seq_len`。
- **B**：PatchTST 式——每段一个 token，由该段的 channel 嵌入，注意力在段之间。

**输出**：未来 M = 10 段（300 s）的终点 (Δe, Δn, Δu) 相对锚点、在 chart 里，每段一个"已到达"位（到达段内的分数给到达时间）。一次直接输出 M 段（非自回归）。单模态先做；K 混合（多模态）为 S5。

**标签**：真值轨迹在锚点后 t = 30k s 的位置；到达 = 在五边上过阈值（`threshold_crossing_index`，包内唯一规则）所在段及其分数；到达之后的段无效（掩码）。300 s 处只有长雷达引导航班还在，读数带 n。

**损失**：终点的掩码 L1（按尺度归一）+ 到达位的 BCE + 到达分数的 L1。

**评估**：航路点误差 p50 在 60 / 120 / 180 / 300 s，分层。对照：(i) 匀速直线外推；(ii) native32 自身 rollout 在同一提前量的位移（`lead_time_error` 已有：雷达引导 632 / 1884 / 3659 m at 60 / 120 / 300 s）；
(iii) **state 路径**（现有 `airport_frame_20260903/A_threshold_enu` checkpoint，与 L2 按共同航班配对；旧文档 §3 表里它在 120 / 180 s 是 1109 / 1673 m，已经好于 native32）。
**门 L2（两种子）**：120 s 与 180 s 的雷达引导航路点 p50 比 native32 与 state 臂中**更强的一个**好 ≥ 125 m（种子线），直线不差；A 与 B 谁过谁用。

## 5. 联合评估与门

lockstep：L2 每 30 s 在滚动历史的段上再问，给 L1 下 2 个航路点；L1 飞 30 s；结束规则与规则制导同一时限（c660600）。读数：整段 ADE / 完全可飞 / 建立 + 逐次 e(30 s)。
**门 E2E（两种子）**：雷达引导 ADE ≤ 2745 m 且直线 ≤ 415 m（native32 一次成形 2870 / 445 减种子线）；完全可飞 ≥ 95 %；建立 ≥ 94 %。
L2 − 真值粗计划的差就是 e_plan，L1 在真值计划下的误差就是 e_track，两者分别报。hook 同 §3：门关，旁读开。

## 6. 现有代码：保留 / 改造 / 废弃

| 代码 | 处置 |
|---|---|
| `outputs/envelope.py` 契约注册表、`aerodynamic_model` 定律协议、hook 的 `VerticalChannel` | 保留（L1 的可切换动力学） |
| `config.py` | 新轴 `control_horizon_s`（ControlOutput 视图；命名 `horizon=`）；`plan_conditioning` 加 `waypoints` |
| `data/dataset.py` | `target_horizon_s(series, anchor, config)`：目标跨度的唯一定义（Δ 或整段），`_sample_arrays`、控制上下文、`common_truth_at_anchors` 都读它；`window_anchors` 的未来下限取 Δ |
| `outputs/control/heads.py` | 固定 Δ 下不建时长头，`duration()` 返回 Δ |
| `outputs/control/plan_token.py` | 新 token `waypoint_token` + `truth_waypoints`；宽度按取值 |
| `experiments/tracker_lockstep.py`、`two_tier_gates.py` | 改造：计划来源按 checkpoint 的 `plan_conditioning`（`TruthWaypoints` 新增，S3 加 L2 头）；每架记录逐次 e(30 s)；`--command-hook` 旁读；门 L1 按 §3，E2E 按 §5 |
| `experiments/short_horizon_readout.py` | 新：§3 评估 1（`--checkpoint L=ckpt` 多臂 + `--reference native32=ckpt`；固定锚点 + 12/8/6 km 档；`truth-plan` / `no-plan` 两列；配对；`--write-records`） |
| `inference/receding.py` | 加 `mean_displacement_to`（ADE[0,Δ]，`lead_time_error` 口径） |
| `experiments/frame_ablation.py` | arm 文件顶层 `"predict": false`：只训练 |
| `publish_ts_experiment_trajectories.py` | S4 待做：识别 `short_horizon` 记录块，hooked lockstep 记录的 variant id 加 hook 后缀 |
| `anchor_floor_index`、`lead_time_error`、`chain_sensitivity` | 保留（读数工具） |
| T1a 臂文件、intents `two_tier_t1a_20260916` | 废弃（不发布，intents 条目留作记录） |
| 旧 plan 头（`outputs/plan/*`） | 不作为高层；`plan_guidance` 线的产物不动；`truth-next` token 保留可加载 |

## 7. 开发步骤（每步：写代码 → opus review → 修 → 测试 → 提交；实验交队列 agent）

| 步 | 内容 | 产物 / 门 |
|---|---|---|
| S0 | 本文档；磁盘（T1a 5 GB）由用户执行删除 | 提交 |
| S1.1 | `control_horizon_s`：config 规则、`target_horizon_s`、窗口下限、无时长头、命名、CLI、测试 | **完成** 29646ab + review 修正 |
| S1.2 | `plan_conditioning=waypoints`：token、训练行、forecast 行、probe、测试 | **完成** |
| S1.3 | `tracker_lockstep` 航路点来源 + 逐次 e(30) + hook 旁读；`two_tier_gates` 门 L1；`short_horizon_readout`；`frame_ablation` 只训练；测试 | **完成** |
| S1.4 | 臂文件 `docs/experiments/two_tier_l1_arms.json` + intents `two_tier_l1_20260917`（**launch 前提交**）→ 队列 agent：6 + 2 臂训练 → `short_horizon_readout`（8 臂 + native32 参照）→ `tracker_lockstep`（hook 关；另一次 `--command-hook barrier+speed-floor`）→ `two_tier_gates --baseline plan_guidance_20260910/step3d_lockstep_l1` | 门 L1 |
| S2 | L2：段特征 + 标签 + 新输出 `segment-plan`（A、B 两臂）+ 读数 runner（含 state 对照）；2 臂 × 2 种子 | 门 L2 |
| S3 | lockstep 接 L2；联合读数 | 门 E2E |
| S4 | 发布 S1–S3 结果 | — |
| S5（条件性） | 横向闭环定律；L2 的 K 混合；L1 在自身状态上训练（若 e(30 s) 沿航班上升） | 各自的门 |
| S6 | L3 多机图层：以 L2 的多机预测段为节点；先做前机真值段的 oracle 前置测量 | 后续设计 |

S1 与 S2 代码互不依赖，可并行开发；GPU 串行，S1 先训。

## 8. 已决定（用户 2026-09-17）

| 项 | 决定 |
|---|---|
| 1. Δ / 执行 / N₁ | 60 s / 30 s / 6 |
| 2. L1 历史 H | 60 s 主臂，30 / 118 s 消融（单种子、航迹角契约） |
| 3. Δ₂ / K_in / M | 30 s / 4 / 10；K_in = 8 删除 |
| 4. 计划 token | 下 2 个航路点 + 剩余时间，坐标系见 §2 |
| 5. 门 | §3 门 L1（两句 + 漂移读数）、§4 门 L2（加 state 对照）、§5 门 E2E |
| 6. T1a 产物 | 删两个 lockstep 记录树与三个 pred_val（约 5.3 GB），留 checkpoint、gates、摘要 |
| 7. 过载族的 thrust | 三契约都跑：δ、n_x、n_x+γ*，各两种子 |
| 8. L1 lockstep 的 hook | 门读数关；`barrier+speed-floor` soft 旁读一列 |

**我替你选的（开发中按直觉定，次日汇报）**：
- 协议 B 的锚点集用 `anchor_grid` 的 L−1 + 12 / 8 / 6 km 档，而不是逐航班随机抽：档是包内配对与分层的既有规则，随机抽是可加的第五档。
- 漂移"上升"的判据：逐次 e(30 s) 的 p50，第 3 次及以后对前 3 次之比 > 1.5（§3）。它只触发 S5，不计入门。
- campaign 只训练，60 s 的 predict 记录不写（对整段进近的评估报告没有意义），记录由两个读数 runner 写。
- `random_train_anchor_min_future_s` 保持 60 s（= Δ）；T1a 的 20 s 在固定 Δ 下不合法。
- one-shot 变体在固定 Δ 头下的含义是"整段 Δ 都执行"（询问周期 60 s 对 30 s），名字不改，runner 文档写明。
- 所有 L1 臂 `anchor_floor_index = 59`：不然 seq_len 30 的固定锚点是 29，lockstep 无法与 anchor 59 的规则制导基线逐航班配对，H 消融也不在一个锚点上。代价为零（前 60 s 只作历史）。
- lockstep 里真值航路点按**时间**索引（飞到的时刻 + 30k s 的真值位置，相对飞到的位置），不按最近真值行的位姿：粗计划是带时刻的日程，落后的跟踪器被告知真值此刻在哪。
- 门的 cohort 允许 L1 是基线的子集（配对在 L1 的航班上，缺的数目写进判定），因为固定 Δ 会排除锚点后真值不足 60 s 的航班。
- 命名 `ctrl-horizon=`（review 指出 `horizon=` 与 `horizon_mode` 的词撞）。
- S4 发布前 publisher 要认 `short_horizon` 记录块；hooked lockstep 的记录先不注册 intents 变体。

## 9. 磁盘

现在剩 13 GB（98 %）。用户批准的删除（沙盒拒绝执行）：

```bash
D=/home/supercomputing/studys/thesis/4dTrajectory/outputs/KRDU/experiments/two_tier_t1a_20260916
rm -rf $D/lockstep_30s/records $D/lockstep_30s_head/records $D/T1a_plan_p0_s1337_pred_val $D/T1a_plan_p50_s1337_pred_val $D/T1a_plan_p50_s2024_pred_val
```

S1 的 8 个 checkpoint 约 0.4 GB；两个读数 runner 的记录（1404 架 × 变体）按 T1a 的 lockstep 记录树估计每臂 2 GB 量级，只给门臂写。
