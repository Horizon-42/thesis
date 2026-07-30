# Direct-control oracle：N=64 实验记录

日期：2026-07-30

## 目的

绕过 Transformer，直接把一条已知未来航迹的 64 组 control logits 和 duration logits
作为可训练参数，通过现有可微 RK4 与固定 2 秒六状态损失反向传播。该实验用于隔离“预测网络”
与“控制/动力学可表达性”，不是可部署预测模型。

## 数据边界

- 航迹：`KSJC:ASA956_30L_a1f7fb_20260719T014743Z`
- 只打开该 outer-train 航迹的数值。
- validation/test 仅使用 manifest 中的锁定身份集合判断该航迹属于 outer-train；没有打开其航迹值。
- 没有运行或检查 outer-test 预测。

## 实现边界

- `control_oracle.py`：无网络的控制参数容器、固定 2 秒 oracle objective、通用 Adam 循环。
- `control_oracle_curriculum.py`：短到长 horizon 的独立裁剪与阶段契约。
- `train_only_diagnostics.py`：只允许选择一个 outer-train 航迹的共享审计入口。
- `run_ts_control_oracle.py`：实验 CLI、指标与结果持久化。
- 正式 `train.py` 不包含 oracle 分支。
- 本轮没有调用 CasADi。

oracle 直接优化：

```text
control logits [64,3] + duration logits [64]
    -> control bounds + softmax(true total duration)
    -> differentiable RK4
    -> fixed-2s state loss + 0.02 * terminal position loss
```

## 正式运行

```bash
conda run -n aeroviz python run_ts_control_oracle.py \
  --airport KSJC \
  --flight-id ASA956_30L_a1f7fb_20260719T014743Z \
  --n-segments 64 \
  --duration-mode learned \
  --steps 500 \
  --restarts 1 \
  --control-learning-rate 0.0001 \
  --duration-learning-rate 0.000025 \
  --gradient-clip-norm 20 \
  --output-dir 4dTrajectory/outputs/KSJC/ts_control_oracle_N64_learned_500
```

此前 train-only 短程校准发现 control LR `0.02` 会在一步下降后立即反弹；`0.002` 在第 9 步
附近出现相同现象；`0.0005` 的前 25 步单调下降。正式运行采用更保守的 `0.0001`。

## 结果

| 指标 | 结果 |
|---|---:|
| 初始 objective | 6.423609 |
| 最佳 objective | 4.099839（step 500） |
| 最佳 state loss | 3.808319 |
| 最佳 terminal contribution | 0.291520 |
| fixed-2s ADE | 17,392.06 m |
| 最后完整 2 秒点误差 | 41,937.26 m |
| 真实终点 3D 误差 | 42,030.87 m |
| duration 最小/最大 | 6.015 / 6.081 s |
| duration entropy | 4.158875（接近 `ln(64)`） |
| control 上下界 1% 内占比 | 0 / 0 |

结果文件：

- `4dTrajectory/outputs/KSJC/ts_control_oracle_N64_learned_500/oracle_result.json`
- `4dTrajectory/outputs/KSJC/ts_control_oracle_N64_learned_500/best_control_schedule.npz`

## 首轮结论

本轮 **数值 oracle 求解失败**，不能把 17.4 km 当作动力学可表达性的下界。原因是该参数空间
至少能够直接表示此前 Transformer 给出的 64 段控制与 observed-clock duration 分配，而此前
Transformer 在同一航迹上的 fixed-2s ADE 为 1.85 km；direct oracle 比它更差，说明 Adam
尚未找到连已知可达基线都能复现的参数点。

运行日志还显示 step 150 以后梯度范数频繁达到 `1e2`--`1e3`，loss 在下降趋势上剧烈摆动；
与此同时 control 和 duration 仍接近均匀初始化。这更符合长时域 single-shooting 的优化/条件数
问题，而不是已经测得了动力学表达上限。

因此首轮停止，先报告该失败；没有自动转向 CasADi。

## 继续实验：future-aware warm-start 与 objective 消融

用户批准继续 PyTorch oracle 后，增加了两个仍与正式训练解耦的能力：

1. `inverse-dynamics` 使用完整未来参考的 `V/psi/gamma` 及其导数，代数反演 point-mass RHS，
   再在 64 个控制段中点采样并裁剪到控制边界。它是 oracle 特权初始化，不可用于部署预测。
2. `schedule` 从上一阶段的 `best_control_schedule.npz` 严格恢复 controls 与 durations；duration
   必须全为正且总和等于真实时长，并在结果中记录输入文件 SHA-256。

本航迹的 inverse-dynamics 初值中，推力有 42.2% 的段落低于物理下界而被裁剪，bank 与 load
factor 没有越界。该现象反映参考减速度与“推力不可为负 + 当前阻力模型”之间存在张力。

### 消融结果

| 阶段 | N | 选优 objective | 新增步数 | fixed-2s ADE | 终点 3D 误差 |
|---|---:|---|---:|---:|---:|
| neutral 初值 | 64 | 六状态 normalized MSE | 500 | 17,392.06 m | 42,030.87 m |
| inverse-dynamics warm-start | 64 | 六状态 normalized MSE | 500 | 1,016.29 m | 4,311.09 m |
| schedule resume | 64 | 六状态 normalized MSE | 1,000 | 853.62 m | 3,459.18 m |
| normalized position-only | 64 | 三位置 normalized MSE | 500 | 1,021.77 m | 3,717.00 m |
| physical-ADE | 64 | 米制 3D 距离 | 500 | 377.81 m | 448.13 m |
| physical-ADE 小 LR 精修 | 64 | 米制 3D 距离 | 1,000 | **332.95 m** | **320.12 m** |

normalized position-only 虽然将自己的 objective 从 0.03193 降到 0.01847，却让 ADE 从
853.62 m 增至 1,021.77 m。它把终点垂直误差从 268.26 m 降至 9.50 m，同时允许水平误差
从 3,448.77 m 增至 3,716.99 m：即使去掉速度通道，各位置通道的不同标准差仍会使训练目标
偏离米制 ADE。

physical-ADE 模式直接优化每个 2 秒点的三维米制距离（除以 1000 只用于数值缩放），终点项也
使用米制 3D 距离。因此其 state objective 数值就是 `ADE / 1000`。它把 ADE 降到 377.81 m；
再从该最佳 schedule 以 control LR `2e-5`、duration LR `5e-6` 精修 1,000 步，得到当前最佳
332.95 m，终点 320.12 m。

另做了一个只运行 1 步的 N=192 inverse-dynamics 探针。其初始 ADE 为 3,404.33 m，一步后
为 3,338.01 m，未显示相对 N=64 初值的即时优势。该探针只能说明把噪声较强的瞬时反演控制
直接提高到约 2 秒分辨率并非良好初值，不能作为 N=192 表达能力的最终结论。

### 此前停止点

当前最佳 332.95 m 仍高于预先声明的 100 m representability 阈值，而且即使 control LR 已降到
`2e-5`，梯度仍偶发达到 `1.4e3`。因此这依然是一个数值近似 oracle，尚不能把 333 m 声称为
动力学的严格误差下界。

按照约定，到此再次停止并报告：所有 continuation 都是 PyTorch direct shooting；没有调用
CasADi，没有读取 validation/test 航迹值。若继续，需在以下方向中另行选择：平滑后的 N=192
inverse warm-start、分阶段/多重 shooting、重新训练并保存 Transformer control schedule 作为
已知可达 warm-start，或明确批准 collocation tracking oracle。

## 继续实验：短到长 horizon 的 single-shooting curriculum

用户选择先做分阶段 horizon，不做 multiple shooting。实现仍然只有一条从真实 anchor 初始状态
连续积分的 single-shooting 轨迹，没有中间状态变量、状态重置或窗口拼接：

```text
60 s（250 步，冻结 duration）
  -> 120 s（250 步，冻结 duration）
  -> 240 s（250 步，冻结 duration）
  -> 386.685 s full（1,000 步，解冻 duration）
```

早期阶段不是只 mask loss。它同时裁掉 horizon 之后的 control、segment duration 和 fixed-2s
监督列，并把最后一个活动 control 段截到精确阶段边界，因此 RK4 的正向与反向链确实在阶段
horizon 结束。60/120/240 秒均与 2 秒 reference grid 精确对齐；只有 full 阶段联合优化 controls
与 duration logits。

正式 recipe：N=64、inverse-dynamics 初始化、physical-ADE、control LR `1e-4`、duration LR
`2.5e-5`、gradient clip 20、seed/split seed 均为 1337。仍只打开同一条 outer-train 航迹。

### 阶段结果

| 阶段 | 本阶段 prefix ADE | prefix 终点 3D 误差 | 阶段结束后 full-horizon ADE |
|---|---:|---:|---:|
| 60 s | 16.82 m | 20.76 m | 3,602.50 m |
| 120 s | 23.21 m | 30.00 m | 3,206.16 m |
| 240 s | 96.96 m | 358.26 m | 966.08 m |
| full（best step 949） | **239.67 m** | **273.65 m** | **239.67 m** |

最终 full-horizon 指标：

- fixed-2s ADE：`239.6698 m`；最后完整 2 秒点误差：`284.4412 m`。
- 真实终点 3D 误差：`273.6464 m`，其中水平 `68.6699 m`、垂直 `264.8902 m`。
- duration 最小/最大：`6.0132 / 6.0572 s`，entropy `4.158881`，仍非常接近均匀分配
  `ln(64)`。
- 相比此前最佳 ADE `332.95 m`，下降约 `28.0%`；终点误差相比 `320.12 m` 下降约
  `14.5%`。

结果位于：

- `4dTrajectory/outputs/KSJC/ts_control_oracle_staged_60_120_240_full_N64_seed1337_run1/oracle_result.json`
- `4dTrajectory/outputs/KSJC/ts_control_oracle_staged_60_120_240_full_N64_seed1337_run1/best_control_schedule.npz`

### 当前停止点

curriculum 明显改善了长时域 direct shooting 的初值与最终结果，尤其 240 秒阶段结束时已把
full-horizon ADE 降到 966.08 m；它不是无效方向。但最终 `239.67 m / 273.65 m` 仍未同时满足
预先固定的 `ADE <= 100 m` 与 `terminal <= 100 m` representability 条件，且 full 阶段仍有
明显梯度尖峰，duration 也几乎没有脱离均匀分配。因此本轮结论仍是
`oracle-fit-above-capacity-threshold`，不能把 239.67 m 当作严格动力学表达下界。

按照事先约定，此处停止并先报告：没有运行 multiple shooting、CasADi、validation 或 test。
