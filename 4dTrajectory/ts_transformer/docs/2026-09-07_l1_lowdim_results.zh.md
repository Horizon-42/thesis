# L1 结果：32 段控制头免费，但轨迹误差损失单独不够（KRDU，2026-09-07）

承接 `2026-09-07_latent_intent_design.zh.md` §六 L1 与 L0 的结论（N\* = 32）。campaign
`4dTrajectory/outputs/KRDU/experiments/l1_lowdim_20260907/`（`readout.txt/.json`、`readout_bank.txt`；
臂 `docs/experiments/l1_lowdim_arms.json`），对照 `control_procedure_20260905/A_control_v3`（simple-v3，
N=64）。KRDU 验证集 1404 架，同队列、同种子、同划分，逐航班配对。

## 一、臂

| 臂 | 宽度 | 监督 | 教师 | 训练 |
|---|---:|---|---|---|
| A_control_v3（对照） | 64 | 段端点按真值时钟对齐（native）+ 速度项 | 逆动力学模仿 64.0 | — |
| L1_dense32 | 32 | 2 s 真值网格稠密 MSE（fixed-dt） | **无** | 156/180 轮早停，55 min |
| L1_dense64 | 64 | 同上 | **无** | 85/180 轮早停，30 min |
| L1_native32 | 32 | 同 A | 同 A | **180/180 轮未早停**，63 min |

设计上 fixed-dt 网格下 `velocity` / `imitation` 两个分量根本不注册（`train.loss_component_names` 只在
`true-time-position` 目标下注册它们），所以 dense 臂天然没有教师——这正是"轨迹误差损失够不够"的干净实验。

## 二、结果（ADE 均值 / FDE p50 / chamfer p50 / Fréchet p50，m）

| 臂 | 全体（1404） | 直线进近（904） | 雷达引导（497） |
|---|---|---|---|
| A_control_v3 | 1333 / 908 / 256 / 1426 | 469 / 703 / 134 / 890 | 2858 / 1971 / 942 / 3100 |
| L1_dense32 | 2515 / 3333 / 869 / 4264 | 1455 / 2863 / 371 / 3624 | 4393 / 5138 / 1842 / 5922 |
| L1_dense64 | 2603 / 3293 / 731 / 4391 | 1364 / 2576 / 404 / 3305 | 4811 / 6312 / 1950 / 7068 |
| **L1_native32** | **1322 / 864 / 224 / 1357** | **445 / 671 / 109 / 820** | 2870 / 1982 / 901 / 3069 |

配对胜率（全体，ADE 优于 A 的航班份额）：dense32 6.8 %、dense64 7.0 %、**native32 52.6 %**（中位 Δ −7 m）。

**Bank skill**（`score_control_arms.py`，随机航班地板 0.170 / 同跑道孪生天花板 0.699）：

| 臂 | bank skill | 共同剖面份额（观测 1.8 %） | 直线参考 bank RMS（观测 0.41°） | 反转（观测 0.0） | 路径长度比（观测 1.01） |
|---|---:|---:|---:|---:|---:|
| L1_dense32 | 0.360 | 10.6 % | 0.94° | **1.0** | 1.25 |
| L1_dense64 | 0.324 | 12.0 % | 1.15° | 0.0 | 1.23 |
| A_control_v3 | 0.728 | 3.8 % | 0.39° | 0.0 | — |
| L1_native32 | 0.726 | 2.6 % | 0.34° | 0.0 | — |

（读法按 `CLAUDE.md`：对地板与天花板读，不对 1.0 读。`GENUINELY_STRAIGHT_TORTUOSITY = 1.02` 的直线参考
是比读数分层更严的一组航班，见 T0-5。）

其他：垂直窗口违反行（观测地板 5.7 %）A 46.6 / dense32 34.2 / dense64 45.7 / native32 46.4 %；横向违反行
native32 50.6 < A 54.8 < dense32 68.8 < dense64 71.6 %；训练日志 jerk 预测/观测：dense 0.05 / 2.84，
native32 0.31 / 2.84 m/s³。

## 三、读法

1. **预注册否决在两个 dense 臂上触发**：直线进近 FDE p50 703 → 2863 / 2576，远超种子噪声。
2. **"轨迹误差损失单独够不够？"——不够。** 同宽度、同参数量（10.52 M）、同种子，dense 2515 m 对
   native+教师 1322 m，1.9 倍；FDE 2.7 倍。而且 2026-08-19 的 **bank wiggle 回来了**：dense 臂把 6 倍于观测
   的坡度能量放进一个共同剖面、直线参考上 bank 大 2.3–2.8 倍、dense32 出现符号反转、路径长度比 1.25——
   位置是 0 阶、速度 1 阶、坡度 2 阶，没有教师就没有任何项给坡度命名。这是设计文档 §六 L1 预注册的
   失效模式，逐字兑现。
3. **宽度几乎免费，网格决定一切。** native32 与 A（N=64）打平略胜（1322 vs 1333，52.6 % 配对胜率，
   直线进近 445 vs 469，chamfer 224 vs 256），bank skill 0.726 vs 0.728。**N=32、96 个操作参数取代 257，
   精度无损**——L0 的表示上界（203 m）在预测侧兑现。宽度与网格只弱交互：两种监督下 32 ≥ 64。
4. **注意**：native32 跑满 180 轮未早停（dense 臂在 156 / 85 轮早停），1322 是预算受限的上界；
   `best_val_loss`（0.67 vs 0.20/0.23）跨监督不可比，只有 ADE 选择指标可比。
5. dense 的垂直违反行反而少（34 %）——更平淡的预测器更"可飞"，正是 `CLAUDE.md` 警告的陷阱；单看可飞性
   不是质量指标。

## 四、对后续的决定

- **L2 的 base = native32**（simple-v3 的监督 + 逆动力学教师，N=32；教师剂量 64.0 在 N=32 上未重标即打平，
  保留）。`l2_latent_arms.json` 已改。
- **拟合教师（L0 的 `basis_fit.json`）升为真正的候选臂**：现在的教师照单飞出来离真值 2.5–7.8 km 而拟合
  教师只差 88–433 m（L0 §三.5），且 L1 证明教师不可缺——它是下一个值得花 15 h 训练集拟合的地方，进
  L5 与 `imitation-target=fitted` 臂并列。
- 设计文档 §四 废除清单里"复审文档 P0/P1（垂直定价、事件锚定、闭环教师）"维持废除：教师需要，但教师
  的**改进方向**是拟合教师，不是闭环教师。

## 五、复现

```bash
python run_ts_frame_ablation.py --arms 4dTrajectory/ts_transformer/docs/experiments/l1_lowdim_arms.json \
    --campaign 4dTrajectory/outputs/KRDU/experiments/l1_lowdim_20260907 --airport KRDU
K=4dTrajectory/outputs/KRDU/experiments
python 4dTrajectory/ts_transformer/docs/compare_constraint_arms.py A_control_v3=$K/control_procedure_20260905/A_control_v3_pred_val \
    L1_dense32=$K/l1_lowdim_20260907/L1_dense32_pred_val L1_dense64=$K/l1_lowdim_20260907/L1_dense64_pred_val \
    L1_native32=$K/l1_lowdim_20260907/L1_native32_pred_val --json $K/l1_lowdim_20260907/readout.json
python 4dTrajectory/ts_transformer/docs/score_control_arms.py $K/control_procedure_20260905 $K/l1_lowdim_20260907
```
（campaign 期间工作树曾被并行提交弄脏，`begin_run` 的干净树检查只在 `train` 步触发；开工前提交。）
