# `control-simple-v1` 开发设计

日期：2026-08-16
状态：设计已冻结并实现；尚未启动新的 development 精度实验
数据边界：只允许 outer-train 与 outer-validation；本文不授权读取、预测或评估 outer-test

## 1. 背景与问题

现有 control-output 路径已经证明“历史航迹 → 分段控制 → 可微动力学 rollout → 4D
轨迹”可以训练，但研究接口同时保留了历史探索产生的多种 duration、clock、监督网格、
tracking objective、terminal objective、动力学后端、正则项、curriculum 和 backbone。
这些模式对复现实验历史有价值，却使正式实验难以回答一个清晰问题：

> 在相同航迹、历史输入、数据 split 和真实物理时间评价口径下，动力学约束的 control
> 输出是否比 direct-state 输出更准确，并同时给出可执行的有界控制？

本设计新增一个命名且冻结的 `control-simple-v1` recipe。它不删除、不改写现有实验模式，
只提供一条参数数量更少、训练与选模语义清楚、可以和 direct-state 公平比较的新路径。

## 2. 非目标与兼容边界

本次不做以下工作：

- 不删除 `factorized`、`direct` duration 或任何历史 control objective；
- 不修改已有 checkpoint，旧派生产物继续由其原有 target contract 读取；
- 不重新生成 arrival manifest、eligibility roster 或 raw/downloaded 航迹；
- 不运行新的精度实验，不据 validation 临时改变冻结参数；
- 不创建或消费 `test_release.json`，不读取 outer-test 轨迹值或指标；
- 不把 control bounds 等同于适航、安全着陆或动力学认证。

`control-simple-v1` 是新增 recipe，不是旧模式的兼容别名。其 checkpoint 必须记录独立的
recipe 名称、uniform-duration target contract、完整配置和数据身份。

## 3. 冻结模型合同

### 3.1 输入与输出

输入保持现有六通道历史窗口：

```text
(e, n, u, edot, ndot, udot), L=60, dt=2 s
```

同时保留逐航班 OpenAP dynamics condition。模型输出：

```text
final_time_s: [B]
controls:     [B, 64, 3]
               └─ thrust_N, bank_rad, load_factor
```

control 继续通过逐航班上下界映射，不改变物理单位和输出记录格式。

### 3.2 固定均匀 duration

删除该 recipe 内的 duration-fraction 自由度：

\[
\Delta t_{i,n}=\frac{\hat T_i}{N},\qquad N=64.
\]

训练 state rollout 继续使用已验证较稳定的 observed-clock 分离：

\[
\Delta t^{train}_{i,n}=\frac{T_i}{N}.
\]

`final_time_s` 始终由独立 time loss 训练；部署和 validation replay 始终使用
`T_hat/N`。这样把当前“均匀 teacher + detached duration gradient”的有效行为写成显式
合同，并彻底移除未受有效梯度训练的 duration projection。

observed-clock 是当前 development 证据下的稳定选择，但它会形成明确的 train/inference
clock 差异。该差异必须在论文 limitation 中说明；本轮不再并行保留 dual-clock 候选。

### 3.3 Backbone 与动力学

冻结如下实现参数：

| 项目 | 值 |
|---|---|
| backbone | iTransformer |
| `d_model / d_ff / e_layers / n_heads` | `512 / 1024 / 4 / 8` |
| dropout | `0.1` |
| coordinate frame | ENU |
| reference velocity | `track-fit` |
| aircraft filter | `openap-direct` |
| control value | absolute bounded control |
| dynamics | `scaled-transport-chart-velocity` |
| RK4 最大步长 | `0.5 s` |

不把 PatchTST、mixture head、trim residual 或其他 dynamics backend 纳入新 recipe。

## 4. 最小训练目标

### 4.1 路径位置

因为 `N=Q=64` 且训练 duration 为 `T/N`，第 `q` 个 rollout endpoint 和真实物理时间
`q*T/64` 一一对应。只比较物理米制三维位置：

\[
L_{p,i}=\frac{1}{Q s_p^2}\sum_{q=1}^{Q}
\left\|\hat{\mathbf p}_{i,q}-\mathbf p_{i,q}\right\|_2^2,
\quad Q=64,\quad s_p=10000\;m.
\]

输入仍按通道标准化，但 loss 必须先恢复为物理米，避免垂直轴因训练集标准差较小而获得
数百倍隐式权重。

### 4.2 输出端点

终点是该航班真实的 threshold crossing 位置，不是硬编码跑道中心或 TCH：

\[
L_{E,i}=\frac{\left\|\hat{\mathbf p}_{i,Q}-
\mathbf p_{i,Q}\right\|_2^2}{s_p^2}.
\]

端点使用和路径相同的物理尺度，固定权重 `0.25`。

### 4.3 总时长

\[
L_{T,i}=\left(\frac{\hat T_i-T_i}{600\;s}\right)^2.
\]

### 4.4 总目标

按现有 flight/airport weighting 做 batch 聚合：

\[
\boxed{L=L_p+0.25L_E+L_T}
\]

本 recipe 明确设置：

```text
kinematic_consistency_weight = 0
legacy_terminal_weight = 0
control_effort_weight = 0
control_smoothness_weight = 0
dense/geometry/arc/terminal-velocity compatibility weights = 0
```

未来速度、kinematic consistency、arc-length、local velocity、terminal velocity、跑道
分量 `1/3/5`、末段 `end-weight=4`、acceleration 和 jerk 均不进入训练目标。速度和运动学
一致性由同一次 dynamics rollout 产生；控制饱和与跳变只作为离线 QA。

## 5. 训练与选模合同

| 项目 | 冻结值 |
|---|---|
| batch | `512` |
| epochs | `180` |
| learning rate | `3e-5` |
| ReduceLROnPlateau | factor `0.5`，patience `8` |
| early stopping patience | `20` |
| gradient clipping | global L2 `20` |
| random train anchor | off；固定 `L-1` |
| validation grid | `Q=64` common true physical time |
| scheduler / early stop / checkpoint selector | airport-macro 3D ADE |

不再生成第二个加权 selection score。若正式候选平均 ADE 差异小于 2%，结论为没有可靠
差异，并选择更简单的模型。

## 6. Teacher 初始化

Teacher 是训练初始化，不是推理输入。正式 control 候选使用现有、固定哈希的 32 条
outer-train schedule：

```text
teacher_schedules.npz
SHA-256 60e40e00e90b12880a76b68e1eaf28bdce74110ffc60c7d017d73ef54b973adc
```

CLI 必须显式接收 schedule 路径，训练前验证其中每个 flight 都属于当前 outer-train，
control shape 为 `[32,64,3]`、duration shape 为 `[32,64]`，且 duration 总和等于 train
真值总时长。uniform recipe 下 teacher duration fraction 必须为均匀分布；否则拒绝，而不是
静默忽略。

初始化固定为 1000 steps、LR `1e-4`、global clip `20`。checkpoint metadata 保留 teacher
文件哈希、flight IDs 和 imitation history。无 teacher 的单 seed run 只用于解释初始化贡献。

## 7. 最小实验矩阵

正式核心矩阵固定为六个 run：

| 输出 | seed 1337 | seed 2027 | seed 4242 |
|---|---:|---:|---:|
| direct-state iTransformer | yes | yes | yes |
| `control-simple-v1` + teacher | yes | yes | yes |

另允许一个解释性开发 run：

```text
control-simple-v1, seed=1337, no teacher
```

训练 seed 只能影响初始化和 batch shuffle；`split_seed=1337` 始终固定。正式比较必须使用
相同 manifest、eligibility、aircraft filter、history、anchor、validation flight 顺序和
Q=64 evaluator。

本轮不再运行 backbone、duration、N、clock、loss weight、regularization、dropout、batch
或 LR 网格。如果未来改变其中任何一项，应建立新的预注册 campaign，不能继续沿用
`control-simple-v1` 名称。

## 8. 报告合同

正式主指标：

- `Q=64` common true-time airport-macro 3D ADE，三种子 mean ± sample SD。

正式次指标：

- per-airport 3D ADE；
- true-arrival-time FDE；
- predicted-arrival endpoint error；
- final-time MAE 与 signed bias；
- horizontal、along-track、cross-track、vertical 的 p50/p95；
- invalid/failed rollout 数量和推理耗时。

control-only QA：

- thrust、bank、load factor 分位数和上下界饱和率；
- 相邻 control 的归一化变化量；
- rollout 成功率。

QA 不进入 scheduler、early stopping 或 checkpoint selection。最终进近 lateral/vertical
verdict 继续由独立 evaluation 模块生成，也不参与选模。

## 9. CLI 与产物设计

新增训练入口：

```bash
conda run -n aeroviz python 4dTrajectory/ts_transformer/__main__.py train \
  --data trajectory_data_process/outputs/harvest/KSJC/arrivals/manifest.json \
  --eligibility-roster trajectory_data_process/outputs/harvest/KSJC/arrivals/lateral_pass_eligibility.json \
  --airport KSJC \
  --control-recipe simple-v1 \
  --control-teacher-schedules \
    4dTrajectory/outputs/KSJC/experiments/oracle_teacher_20260816_current_manifest/optimized_arc24_32/teacher_schedules.npz \
  --seed 1337 \
  --split-seed 1337 \
  --output-dir 4dTrajectory/outputs/KSJC/experiments/control_simple_v1/seed1337
```

允许 recipe 外覆盖的字段只有运行身份，不改变科学定义：

- `seed`；
- `split_seed`，但正式矩阵要求为 1337；
- `device`；
- output/campaign/experiment 路径与标识；
- teacher schedule 路径，以及 no-teacher 解释性 run。

对 recipe 内冻结字段给出冲突值时 CLI 必须失败并指出冲突，不能静默覆盖。checkpoint 的
config 与 metadata 至少记录：

```text
control_recipe_name = simple-v1
duration_parameterization = uniform
state_objective = true-time-position
state_supervision_clock = observed
checkpoint_selection_metric = fixed-anchor-common-grid-ade
validation_common_grid_points = 64
teacher audit（若启用）
```

## 10. 实现步骤

1. 在 `config.py` 注册 `simple-v1`、`uniform` duration 和
   `true-time-position` objective，并验证冻结字段。
2. 新增 uniform-duration control head，只保留 control projection 和 final-time head。
3. 在 control rollout loss registry 中加入物理三维 path MSE 与共享 `0.25` endpoint MSE。
4. 继续复用现有 common-grid ADE validation selector，不创建新 selector。
5. 在 CLI 中实现命名 recipe、冲突检查和 cached teacher schedule 入口。
6. 更新 checkpoint target contract 与 `control_recipe` metadata。
7. 添加配置、head、loss、CLI、teacher 和 checkpoint round-trip 定向测试。

## 11. 验收条件

实现完成必须同时满足：

- `TSConfig` 能序列化/恢复 `simple-v1`，且冻结字段不允许漂移；
- uniform head 不包含 `duration_projection` 参数；
- 对所有 batch，`segment_durations == final_time_s / 64`、全正且和为 final time；
- train observed-clock rollout 使用 `T/64`，inference 使用 `T_hat/64`；
- path loss 只使用物理米制 position，端点权重固定为 0.25，time loss 独立存在；
- control effort/smoothness、future velocity 和 arc/terminal velocity 不影响该 recipe loss；
- checkpoint selector 为 Q=64 common true-time ADE；
- teacher schedule 非 train ID、非均匀 duration 或 shape 不符时失败；
- 旧 control/state 模式的既有定向测试保持通过；
- 未运行或读取 outer-test。

## 12. 实现落点

实现严格按上述顺序在本文档冻结之后完成：

- `config.py`：注册 recipe、uniform duration、minimal objective 及冻结字段检查；
- `uniform_duration_control.py`：实现不含 duration projection 的单一路径 output head；
- `control_loss_components.py`、`train.py`：实现 observed true-time 的物理三维位置与端点损失；
- `__main__.py`：提供 recipe 冲突检查和 outer-train cached teacher 初始化入口；
- `oracle_teacher/pretraining.py`：验证 train membership、shape、总时长及 uniform duration；
- `tests/test_ts_transformer.py`、`tests/test_oracle_teacher.py`：覆盖配置、模型、loss、teacher、
  checkpoint round-trip 和真实可微动力学反向传播。

实现不会自动启动六个正式 run。新的 validation 精度结果应按第 7 节矩阵单独执行并记录；
在冻结实验和显式请求最终发布之前，outer-test 继续保持封存。

实现验证命令：

```bash
conda run -n aeroviz python -m pytest \
  aerodynamic_model/tests/test_torch_dynamics.py \
  4dTrajectory/ts_transformer/tests/ -q
```

2026-08-16 验证结果：`384 passed, 6 skipped`。本次只运行单元/集成合同测试，没有启动
训练 campaign，没有读取或预测 outer-test。
