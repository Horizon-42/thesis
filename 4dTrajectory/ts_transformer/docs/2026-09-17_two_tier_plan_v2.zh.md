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
| S0 文档 + 磁盘 | 本文件（1625dc3）；T1a 的 5 个产物树已删（2026-09-17 晚，用户再次确认后执行：5.4 GB → 170 MB，磁盘 5.7 → 11 GB） |
| S1.1 `control_horizon_s` | **完成** 29646ab + review 修正 1604b65（opus 6 项确认：anchor-grid 与 anytime 的未来下限跟 Δ、每 epoch 报告的终端速度/弧长几何读 Δ 内的真值、排除原因写明 Δ、审计里的下限是生效值、`intent=truth-join-duration` 拒绝、命名改 `ctrl-horizon=`、时长项为零写明） |
| S1.2 `plan_conditioning=waypoints` | **完成** ddfbf82（token 11 维、训练行/forecast 行/probe 一处分派、`PLAN_WAYPOINT_SEGMENT_S=30`） |
| S1.3 读数工具 | **完成** bea33af：`tracker_lockstep` 航路点来源 + 逐次 e + `--command-hook`（schema v3）；`two_tier_gates` 门 L1 + 漂移读数（schema v2，cohort 可为基线子集并计数）；`short_horizon_readout` 新 runner；`frame_ablation` `"predict": false`。全套 1318 测试通过 |
| S1.4 臂 + intents | **完成**（训练 8 臂 + 2a–2d 读数，结果 §10.1–10.4；门 L1 三契约全 FAIL）：`docs/experiments/two_tier_l1_arms.json`（8 臂）、intents `two_tier_l1_20260917`；首次 launch 死在 `prepare_session`（60 s 未来契约覆盖 6848/6849 训练航班，`KRDU:FFT1168_05L_a8d27e_20260501T121308Z` 无可用锚点）→ 82294f3：`frame_ablation` 把声明里的 `"development_cohort"` 传给每个 train 步，臂文件声明 `<campaign>/development_cohort.json`（队列 agent 用 `run_ts.py plan_cohort` 写出：6848 / 1401，h30 臂交叉核对相同）；队列 agent 于 03:47 UTC 重启（PID 1967516），~5.6 s/epoch，臂 1 `L1_tf_s1337` 完成（选择 ADE 168.2 m），全部 8 臂约 2.5–3 h，之后自动跑 2a–2d（`short_horizon_readout` → lockstep hook 关 / 开 → `two_tier_gates` ×3 契约）并汇报 |
| S2.1 `segment-plan` 输出 | **完成** 9d9afc8（opus review 12 项已处理、全套测试通过）；本文件 §4 已按实现改：输出在**跑道系**而非 chart；到达 = 真值终点 `truth_duration_s`；每 epoch 的 `segment_plan_validation` 读数块 |
| S2.2 L2 读数 runner | **代码完成**：`run_ts.py segment_plan_readout`（每个 checkpoint 在共同固定锚点——各自固定锚点中最晚的——与 12/8/6 km bin 上预测，60/120/180/300 s 的位移分层；forecast 结束后**保持末行**（到达的计划停在入口，读的是它的到达判断），只有真值落地才缺席；每个 checkpoint 在**自己的 split** 上读、按共同航班配对——native32 是 openap-direct 队列（val 1404），state 臂与 L2 臂是全机型（val 2104，前者 ⊂ 后者，已核）；匀速外推内置参照；segment-plan 臂带计划块）+ `two_tier_gates --segment-readout`（门 L2：120 与 180 s 雷达引导 p50 比**每个**整段参照都低 ≥ 125 m、直线不高于，两种子）。opus review 11 项已处理（固定锚点集要求该锚点存在且其后 ≥ 60 s 观测轨迹、held 计数进配对格与门判据行、多个 readout 参照/锚点/提前量不一致拒绝、segment-plan 臂一次前向、`window` 参照拒绝、`strata_fixed_at_anchor`）。测试 `test_segment_plan_readout.py`（8）、`test_two_tier_gates.py` L2 段（5）。**待用户定**：直线"不差"= 差 ≤ 0（零容差；包内直线种子线是 30 m，review 建议考虑）；参照的记录也写出（S4 需为 native32/state 注册 `@readout-*` 变体或跳过） |
| S2.3 L2 臂 + intents | **campaign 运行中**（首次启动死在第 1 轮的打印行——`segment_plan_validation` 的到达时间误差为 None，3512fdc 修；重启 @3512fdc，PID 2140320，第 1 轮的损失与中断那次逐位相同；3.6 s/epoch，4 臂 ≈ 45 min，然后读数带记录、门 L2 ×2）：`docs/experiments/two_tier_l2_arms.json`（A/B × 2 种子，seq_len 61，M = 10，全机型 cohort，随机锚点 remaining-path-uniform、未来 ≥ 30 s、半数留在固定锚点 60，180 epoch，按目标函数选），dry run 通过；intents `two_tier_l2_20260917`（4 run + `@readout-<set>` 变体）；cohort 已写 `two_tier_l2_20260917/development_cohort.json`（10102 / 2104，丢 3 个训练航班）。等 L1 campaign 跑完后交队列 agent：训练 4 臂 → `segment_plan_readout`（参照 native32 + `airport_frame_20260903/A_threshold_enu`，`--write-records`）→ `two_tier_gates --segment-readout`（A 两种子、B 两种子各判一次） |
| **首批读数（2026-09-17 晚，2a）** | **L1 不用它的计划 token**：六个主臂带真值航路点与不带的 ADE[0,60] 只差 2–4 m（见 §10）；不带 token 时优于 native32 切到 60 s。诊断臂 `docs/experiments/two_tier_l1b_arms.json`（4 臂、单种子、pa 契约：遮蔽率 0 / 不减 lr / 两者 / 关平滑项）+ intents `two_tier_l1b_20260917`，排在 L2 之后、E2E 之前 |
| S4 发布 | **L1 短时读数已发布**（2026-09-17 晚，opus agent，pub worktree @cbdf6ca）：6 个主臂 × {truth-plan, no-plan} × {fixed, 8km} = 24 个类目 `experiment_<arm>_<token>_short-<variant>-<set>_val`，intents 标题「L1 · 短时控制层…」与每 run/变体行已盖章；`npm run check-publication -- --airport KRDU --server http://localhost:5173` → 195 类目 0 错 0 警；前端重启后在 app 里打开确认渲染。发布命令形状（publisher 默认根指向 POOLED，需显式 `--experiment-index …/KRDU/experiments/index.json --output-root …/KRDU/experiment_predictions`；索引先用 `python -m ts_transformer.training.experiment_index --root outputs/KRDU/experiments` 重建）在 scratchpad `publish_brief_two_tier.md`。未发布：12km/6km 档、两个 H 消融臂、native32 的记录、lockstep/L2/E2E（未就绪）。磁盘 5.9 GB（99 %） |
| S3 联合 lockstep + 门 E2E | **完成** a1e58aa：`tracker_lockstep --plan-head <segment-plan ckpt>` 的来源 `SegmentHeadWaypoints`（协议 A：L2 在滚动历史上的下 K 个航路点相对飞到的行、自己的到达时间作首次询问的时限；每次询问旁记 e_plan = L2 航路点对同一时刻真值航路点的误差，逐航路点与合并）；`--anchor-floor-index 60`（L2 seq_len 61 的回看放不进 L1 的锚点 59，把所有 tracker 读在 60；拒绝早于其自身锚点）；门 E2E = G3 的判据（§5 = 旧 §6 的数）；`two_tier_gates` 按 plan source `segment-head` 归到 E2E。opus review 7 项已处理（e_plan 的 ask 用 lockstep 的序号、首问没画出到达的航班按"时限来自计划跨度"逐航班计数、`--batch-size` 传到头、逐航路点 n、record 块写 floor、schema v4 而门仍读 v3）。等 L1 门与 L2 门的结果后由队列 agent 跑（种子配对：L1 s1337 + L2 s1337 …；brief 第 4 步） |

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

**输出**（按实现，2026-09-17 S2.1）：未来 M = 10 段（300 s）的终点 (along, across, up) 相对锚点、**在跑道系里**（`labels.runway_deltas`；接 L1 的 token 时用 `chart_deltas` 转回 chart），每段一个"已到达"位（到达段内的分数给到达时间）。一次直接输出 M 段（非自回归）。单模态先做；K 混合（多模态）为 S5。两臂的 token 经**同一个学习查询**（`TokenPool`）读出为 d_model 再进同一个头——review 指出展平读出让 A 臂的头是 27·d_model、B 臂 K·d_model，门 L2 的 A/B 比较会变成容量比较。

**标签**：真值轨迹（监督行：观测轨迹按过阈值点——观测到的或拟合的——闭合到入口）在锚点后 t = 30k s 的位置；**到达 = 真值终点 `truth_duration_s`**（包内每条路径的时长头都训练的那一个定义；未拟合到五边的航班在轨迹末尾结束，与其它路径一致）所在段及其分数；真值没到的段无位置目标（掩码），到达位每段都监督。300 s 处只有长雷达引导航班还在，读数带 n。

**损失**：终点的掩码 L1（按尺度归一）+ 到达位的 BCE + 到达分数的 L1。

**解码与记录**：第一个到达概率 ≥ 0.5 的段为到达段；forecast 行 = 之前各段的航路点 + 到达时刻的入口（`truncatedAtThreshold`），没到达则 M 个航路点（`horizonCapped`）；记录写 `segmentPlanArrivalSegment` / `segmentPlanArrivalProbability`。验证回放把计划补到 M 行（入口按 30 s 保持、零速度）并用 `Replay.row_valid` 告诉运动学读数哪些行是补的。**每 epoch 读数块 `segment_plan_validation`**（`readout.py`）：[0, min(T_真值, M·30)] 上的 ADE、各段末的 e(30k) 及 n、到达混淆（计划/真值在 300 s 内到达）、带号到达时间误差——common-grid 报告块把最后一个航路点平推到落地、cap rate 结构性为 0，不能引用。

**评估**：航路点误差 p50 在 60 / 120 / 180 / 300 s，分层。对照：(i) 匀速直线外推；(ii) native32 自身 rollout 在同一提前量的位移（`lead_time_error` 已有：雷达引导 632 / 1884 / 3659 m at 60 / 120 / 300 s）；
(iii) **state 路径**（现有 `airport_frame_20260903/A_threshold_enu` checkpoint，与 L2 按共同航班配对；旧文档 §3 表里它在 120 / 180 s 是 1109 / 1673 m，已经好于 native32）。
**门 L2（两种子）**：120 s 与 180 s 的雷达引导航路点 p50 比 native32 与 state 臂中**更强的一个**好 ≥ 125 m（种子线），直线不差；A 与 B 谁过谁用。实现（S2.2，`two_tier_gates --segment-readout`）："更强的一个" = 对每个整段参照都成立；p50 在双方共同持有的航班上各算一次再相减（`delta_of_p50_m`），直线"不差" = 差 ≤ 0；匀速外推只旁读不判。读数在 forecast 结束后保持末行——到达的计划停在入口，120 s 处读到的是它的到达判断，S1 读数的"缺席不计"会让早到逃掉；只有真值落地才缺席。每个 checkpoint 在自己的 split 上读（native32 openap-direct 1404 航班，state 臂与 L2 全机型 2104，前者是后者的子集），配对按共同航班，n 写在每格。

## 5. 联合评估与门

lockstep：L2 每 30 s 在滚动历史的段上再问，给 L1 下 2 个航路点；L1 飞 30 s；结束规则与规则制导同一时限（c660600）。读数：整段 ADE / 完全可飞 / 建立 + 逐次 e(30 s)。
**门 E2E（两种子）**：雷达引导 ADE ≤ 2745 m 且直线 ≤ 415 m（native32 一次成形 2870 / 445 减种子线）；完全可飞 ≥ 95 %；建立 ≥ 94 %。
L2 − 真值粗计划的差就是 e_plan，L1 在真值计划下的误差就是 e_track，两者分别报。hook 同 §3：门关，旁读开。
实现（S3）：`tracker_lockstep --plan-head <segment-plan ckpt> --anchor-floor-index 60`——L2 的 61 采样回看放不进 L1 臂的锚点 59，联合 lockstep 把 L1 读在 60（比它训练的固定锚点晚一个采样；产物写明）；L2 的到达时间是首次询问的时限（没画出到达时取计划跨度 300 s，即 T₀+max(30, 30)=330 s——头没看到落地就只有这个预算，一个 > 330 s 的航班会在时限被切、算未建立：这是头自己的判断，门读的就是它）；e_plan 每次询问旁记（`asks_plan_e_m`，逐航路点），`two_tier_gates` 里 `plan_e_by_ask_p50_m` 逐层旁读。门 E2E 的数与 G3 相同（`judge_g3`）。

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
| S2.1 | L2：段特征 + 标签 + 新输出 `segment-plan`（A、B 两臂，共用读出头）+ 每 epoch 读数块 | **完成**（review 12 项：到达定义写明、回放行掩码回到 `raw_kinematic_metrics`、cap 读数块、容量配平、`use_norm` 拒绝、编码栈一处构造、记录字段独立、特征从 `target_chart` 量、子采样退化拒绝；测试 15） |
| S2.2 | L2 读数 runner（航路点 p50 at 60/120/180/300 s 分层；对照匀速外推、native32、state 臂，共同航班配对）+ 门 L2 判定 | 门 L2 |
| S2.3 | 臂文件 + intents（A / B × 2 种子）→ 队列 agent | 门 L2 的数 |
| S3 | lockstep 接 L2（`SegmentHeadWaypoints`、`--anchor-floor-index`、e_plan 旁读）；门 E2E 归入 `two_tier_gates` | **代码完成**；实验待 L1/L2 门 |
| S4 | 发布 S1–S3 结果 | — |
| S5（条件性） | 横向闭环定律；L2 的 K 混合；L1 在自身状态上训练（若 e(30 s) 沿航班上升）；**L1b：L1 为什么不用 token（4 个诊断臂，已写，见 §10）** | 各自的门 |
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
- S4 发布前 publisher 要认 `short_horizon` 记录块；hooked lockstep 的记录先不注册 intents 变体。（已做：publisher 的 `VARIANT_RECORD_BLOCKS` 加了 `short_horizon` 与 `segment_plan`，发布时用 `--category-variant readout-<set>` / `<variant>`，intents 里变体 id 同名。）
- **S2（2026-09-17 下午）**：L2 的输出在**跑道系**（along/across/up 相对锚点）而非 chart——同一基线转弯在 05L 与 23R 是同一个标签，接 L1 时 `chart_deltas` 转回；到达 = 真值终点 `truth_duration_s`（包内每条路径时长头的同一定义），不另做 `threshold_crossing_index` 标志；两臂共用一个学习查询读出（`TokenPool`），否则 A 臂的头是 27·d_model、B 臂 K·d_model，门 L2 的 A/B 比较会变成容量比较；`use_norm` 在 segment-plan 下拒绝（编码栈读在 vendored 实例归一化之下，写进 config 却不生效）；验证回放补到 M 行时入口按 30 s 保持、`Replay.row_valid` 告诉运动学读数哪些是补的（review C-10 删掉的掩码回来了——这是第一条会补行的活路径）；每 epoch 的 `segment_plan_validation` 块是 L2 的可引用数（common-grid 报告把末航路点平推到落地、cap rate 结构性为 0）。
- **L2 的 cohort 是全机型**（`aircraft_filter=all`，train 10102 / val 2104），比 native32 的 openap-direct（6851 / 1404）大：计划层不含动力学，不需要 openap 机型；读数在每个 checkpoint 自己的 split 上读、按共同航班配对（native32 的 val 是 L2 val 的子集，已核），state 臂 `A_threshold_enu` 本就是全机型。
- 读数的共同锚点取各 checkpoint 固定锚点中**最晚的**（L2 seq_len 61 → 60；native32 与 state 臂 59，晚一个采样读它们），不加 `anchor_floor_index`。
- 读数在 forecast 结束后**保持末行**而不是缺席：到达的计划停在入口，120 s 处读到的是它的到达判断；S1 的"缺席不计"会让早到逃掉。
- 门 L2 的"更强的一个" = 对每个整段参照都成立；p50 在共同航班上各算再相减；直线"不差"= 差 ≤ 0；匀速外推旁读不判。
- 随机锚点未来下限 30 s（第一段总有监督）、半数留在固定锚点 60（读数与门在那里）；180 epoch 不早停，与 L1 同。

## 9. 磁盘

现在剩 13 GB（98 %）。用户批准的删除（沙盒拒绝执行）：

```bash
D=/home/supercomputing/studys/thesis/4dTrajectory/outputs/KRDU/experiments/two_tier_t1a_20260916
rm -rf $D/lockstep_30s/records $D/lockstep_30s_head/records $D/T1a_plan_p0_s1337_pred_val $D/T1a_plan_p50_s1337_pred_val $D/T1a_plan_p50_s2024_pred_val
```

S1 的 8 个 checkpoint 约 0.4 GB；两个读数 runner 的记录（1404 架 × 变体）按 T1a 的 lockstep 记录树估计每臂 2 GB 量级，只给门臂写。


## 10. 首批读数（2026-09-17 晚）

### 10.1 L1 短时读数（2a，`two_tier_l1_20260917/short_horizon`，KRDU val，固定锚点 59，horizon 60 s）

ADE[0,60] 均值 m（全部 n=1401 / 雷达引导 n=497），fixed 与 8 km 两档；native32 = 整段头切到 60 s：

| 臂 | truth-plan fixed 全部/雷达 | no-plan fixed 全部/雷达 | truth-plan 8 km 全部/雷达 | no-plan 8 km 全部/雷达 |
|---|---|---|---|---|
| native32（参照） | — | 163 / 250 | — | 136 / 143 |
| L1_tf_s1337 | 165 / 218 | 165 / 215 | 106 / 113 | 105 / 112 |
| L1_tf_s2024 | 161 / 217 | 163 / 218 | 103 / 110 | 103 / 111 |
| L1_sf_s1337 | 152 / 214 | 151 / 210 | 82 / 88 | 82 / 87 |
| L1_sf_s2024 | 149 / 212 | 149 / 209 | 82 / 89 | 83 / 89 |
| L1_pa_s1337 | 134 / 181 | 136 / 183 | 77 / 84 | 77 / 84 |
| L1_pa_s2024 | 135 / 185 | （待 artifact） | 80 / 88 | （待 artifact） |

读法：
- **门 L1 第一句不过**：truth-plan 雷达引导 181–218 m ≫ 125 m；直线约 108 m ≫ 60 m（由全部与雷达引导反推）。
- **第二句过**：no-plan 优于 native32（固定锚点 136 对 163；8 km 处 77 对 136）——固定 60 s 时域的控制头是更好的短时预测器，航迹角契约最好，推力分数最差（与 N7′ 的次序一致）。
- **关键发现：truth-plan ≈ no-plan**。头没有用 token。探针（`scratchpad/token_probe.py`，L1_pa_s1337，val 前 48 航班，CPU）：把 token 的航路点向北平移 2 km，60 s 终点只动 67 m（p50）；向上 2 km 动 161 m；带 token 对不带，终点差 40 m。token 确实进了 `feature_fusion`（`heads.py`，一个 Linear+GELU 并进 6·d_model 的编码特征旁），训练行与读数行用同一个 `training_plan_token`，所以这是**优化侧没学会用**，不是接线错误。
- 三个可疑的优化因素：token 遮蔽率 0.5（半数行没有 token）；lr 从 3e-5 起、按 objective plateau（patience 3）减半，到 126 轮已是 1e-7（大半个 campaign 没在训练）；平滑项（航向率 8、坡度 TV 1）主导目标。L1b 四臂各隔离一个（§7 S5）。若都不用 token → 需要把 token 直接接到控制头（架构改动，用户定）。
- 对 S3/E2E 的含义：在 L1 不用 token 之前，端到端 lockstep 里 L2 的计划不影响 L1，E2E 读数等于协议 B 的 lockstep；E2E 排在 L1b 之后（brief 第 4 步）。

**分层与配对（artifact `short_horizon_readout.txt`）**，固定锚点，ADE[0,60] 均值 m（straight-in n=901 / vectored n=497），e(60) 为 p50：

| 臂 | truth-plan 直线 / 雷达 | no-plan 直线 / 雷达 | e(60) p50 雷达（truth / no） | no-plan − native32 雷达 fixed（均值，arm-better） | no-plan − native32 直线 8 km | truth-plan − no-plan 雷达 fixed |
|---|---|---|---|---|---|---|
| native32 | — | 115 / 250 | — / 632 | — | — | — |
| L1_tf_s1337 | 136 / 218 | 138 / 215 | 399 / 385 | −35（0.62） | −30 | +3（0.38） |
| L1_tf_s2024 | 130 / 217 | 133 / 218 | 374 / 362 | −32（0.65） | −34 | −1（0.50） |
| L1_sf_s1337 | 118 / 214 | 119 / 210 | 361 / 336 | −40（0.68） | −53 | +4（0.38） |
| L1_sf_s2024 | 115 / 212 | 116 / 209 | 353 / 344 | −41（0.69） | −54 | +3（0.41） |
| L1_pa_s1337 | 109 / 181 | 110 / 183 | 299 / 290 | −67（0.75） | −60 | −1（0.44） |
| L1_pa_s2024 | 107 / 185 | 110 / 186 | 281 / 279 | −64（0.77） | −57 | −2（0.54） |

历史消融（pa 契约，种子 1337，同锚点 59）：truth-plan 固定锚点 直线 / 雷达 ADE[0,60]：h30（seq_len 15）**106 / 178**，H 60（主臂）109 / 181，h118（seq_len 60）110 / 179；8 km 处 72 / 80、73 / 84、77 / 89。**30 s 历史已经够，60 与 118 s 不再多给**（与旧文档 §10.6 的整段读数一致）。训练：8 臂都跑满 180 epoch；选择指标（固定锚点 common-grid ADE）最优 tf 168.2 / 164.1、sf 156.7 / 155.2、pa 139.8 / 139.5、h30 138.1、h118 139.3 m。no-plan 对 native32 的直线格：tf **差** +22 / +18 m（arm-better 0.32 / 0.35），sf 持平，pa −5。

三种契约的次序（pa < sf < tf）与 N7′/N3 的整段读数一致；no-plan 优于 native32 在雷达引导上是 0.62–0.77 的逐航班多数，直线在固定锚点持平（±5 m）、8 km 处 −30 到 −60 m。**truth-plan − no-plan 在每格都是 0 ± 4 m、arm-better 0.36–0.55：token 没有被使用。**

### 10.2 L1 lockstep（2b，hook 关，`two_tier_l1_20260917/lockstep_30s`，a0 = 59，每 30 s 再问，真值航路点）

receding 变体，ADE 均值 m（直线 n=901 / 雷达引导 n=497），完全可飞 / 建立为雷达引导层的份额；括号里是 receding-no-plan 的雷达引导 ADE：

| 臂 | 直线 ADE | 雷达 ADE（no-plan） | 雷达 可飞 / 建立 | 直线 可飞 / 建立 |
|---|---|---|---|---|
| L1_tf_s1337 | 534 | 6943（7235） | 0.000 / 0.125 | 0.838 / 0.912 |
| L1_tf_s2024 | 474 | 4230（5037） | 0.000 / 0.205 | 0.653 / 0.900 |
| L1_sf_s1337 | 360 | 8061（5558） | 0.002 / 0.193 | 0.972 / 0.942 |
| L1_sf_s2024 | 359 | 3964（4660） | 0.002 / 0.249 | 0.967 / 0.891 |
| L1_pa_s1337 | 359 | 2295（2790） | 0.358 / 0.125 | 0.990 / 0.903 |
| L1_pa_s2024 | 379 | 2405（2537） | 0.465 / 0.225 | 0.966 / 0.922 |
| L1_pa_h30_s1337 | 341 | 2818（3203） | 0.101 / 0.022 | 0.991 / 0.928 |
| L1_pa_h118_s1337 | 380 | 2133（2671） | 0.322 / 0.113 | 0.978 / 0.900 |

**门 L1 的 lockstep 句全部不过**：雷达引导 ADE 2133–8061 m（门 < 1500），直线 341–534 m（门 < 250），雷达引导完全可飞 0–0.47（门 ≥ 0.95），雷达引导建立 0.02–0.25（门 ≥ 0.88 × 规则制导 0.879 = 0.77）；只有直线建立（0.89–0.94 对 0.88）过。雷达引导航班几乎都在时限被切（435/497 `horizon`），即跟踪器沿着真值航路点也没飞到五边。receding 比 no-plan 好 130–800 m（tf/sf/pa），说明 token 在多次询问下有一点作用，但离"跟着计划飞"很远——与 §10.1 的 token 盲一致。

### 10.3 门 L1 判定（2d，`gates_{pa,tf,sf}`，基线 `plan_guidance_20260910/step3d_lockstep_l1`，同航班规则制导建立份额直线 0.998 / 雷达 0.879）

**三种契约、六个臂全部 FAIL**；每个臂只有"直线建立 ≥ 0.878"一条过（0.891–0.942）。其余四条（直线 ADE < 250、雷达 ADE < 1500、完全可飞 ≥ 0.95、雷达建立 ≥ 0.774）无一过：整臂完全可飞 tf 0.54 / 0.42、sf 0.63 / 0.63、pa 0.77 / 0.79；雷达建立 0.13–0.25。每个门文件都写明基线里有 3 个航班不在跟踪器 cohort 里（cohort 规则允许子集）。

**漂移读数：每个臂、每层都 RISING**（晚/早 p50 之比，阈值 1.5）：全部层 8.6–13.8，直线 3.3–4.1，雷达 8.3–11.9。例：`L1_pa_h118_s1337` 雷达逐次 e(30) p50：k0 123 → k3 889 → k6 2140 → k9 2851 → k12 3016 m。按 §3 的规则这**触发 S5**（在自身状态上训练）——但上游是 token 盲：跟踪器不看计划，所以在自己飞出的行上再问只能把误差滚大；先修 token（L1b），再决定 S5。

receding − 规则制导（`L1_pa_h118_s1337`）：全部 +163 m 均值 / +104 p50（receding 更好 0.355），雷达 +286 / +754（0.334）：现在的 L1 沿真值航路点跟踪，比规则制导沿真值指令飞还差。

### 10.4 hook 开的 lockstep（2c，`lockstep_30s_hooked`，`barrier+speed-floor` soft，六个主臂，旁读不计门）

hook 关 → 开，receding：

| 臂 | 直线 ADE | 直线 可飞 / 建立 | 雷达 ADE | 雷达 可飞 / 建立 |
|---|---|---|---|---|
| L1_tf_s1337 | 534 → 488 | 0.84 → 0.99 / 0.91 → 0.97 | 6943 → 7487 | 0.00 → 0.00 / 0.13 → 0.14 |
| L1_tf_s2024 | 474 → 438 | 0.65 → 0.99 / 0.90 → 0.97 | 4230 → 8197 | 0.00 → 0.00 / 0.21 → 0.28 |
| L1_sf_s1337 | 360 → 343 | 0.97 → 1.00 / 0.94 → 0.99 | 8061 → 6525 | 0.00 → 0.05 / 0.19 → 0.17 |
| L1_sf_s2024 | 359 → 331 | 0.97 → 1.00 / 0.89 → 0.98 | 3964 → 5262 | 0.00 → 0.06 / 0.25 → 0.10 |
| L1_pa_s1337 | 359 → **319** | 0.99 → 1.00 / 0.90 → **1.00** | 2295 → **2175** | 0.36 → **1.00** / 0.13 → **0.40** |
| L1_pa_s2024 | 379 → **341** | 0.97 → 1.00 / 0.92 → **1.00** | 2405 → **2271** | 0.47 → **1.00** / 0.23 → **0.42** |

读法：交付形态下航迹角契约的雷达引导航班全部可飞、建立翻倍到 0.40，直线建立 1.00；推力分数与比力契约的雷达引导 ADE 在 hook 下反而变差（hook 在没跟上计划的轨迹上拉回五边，越拉越远）。L1 campaign 至此全部完成（2026-09-17 深夜）；2b/2c 因磁盘未写记录，发布 lockstep 变体需另跑一次带记录的 lockstep（约 6 GB），由用户定。
