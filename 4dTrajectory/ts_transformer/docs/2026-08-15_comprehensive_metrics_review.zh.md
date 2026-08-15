# 4D 轨迹模型 metrics 全面审核与最小化设计

日期：2026-08-15  
状态：审核与小规模 development 验证完成；正式实现已更新，全量 CV 尚未启动
数据隔离：只读取 train/validation 与既有 development 产物；**未读取 outer-test 轨迹或预测**

## 1. 最终结论

本次审核确认，问题不是“指标太少”，而是过去同一模型曾被三套不等价口径描述：

1. 训练 loss 曾在归一化进度上比较六维 state，并叠加时间、运动学和弱终点项；
2. CV/fit report 曾同时保留 native、common-grid、terminal、arc-length 等 selector；
3. prediction `summary.json` 曾只比较预测与观测的时间重叠部分。

这三套数值不能同时叫“模型准确率”。建议收缩为以下单一合同：

- **训练**：真实物理时间对齐的各向同性三维路径位置 MSE + `0.25` 倍真实输出端点
  位置 MSE + 独立到达时间 MSE；机场宏平均。
- **CV、学习率调度、早停、checkpoint、正式 validation**：全部使用同一个
  `Q=64` 真实物理时间公共网格上的 **airport-macro 3D ADE**。
- **正式报告**：主指标之外，分别报告水平、沿航迹、横航迹、垂直、真实时刻 FDE、
  预测到达端点误差和到达时间误差，不再把它们任意加权成第二个总分。
- **最终进近 evaluation**：继续判断跑道阈值处的横向/垂直通过状态，但它是终端运行
  验收，不参与模型训练或选模。
- **模型输出**：历史输入可以保留位置和速度；未来输出应只学习位置曲线与总时长，
  未来速度由曲线和时钟求导得到。模型必须预测实际跑道阈值穿越位置，不能把终点强制
  固定在跑道中心/TCH；真实 lateral-pass 航迹在阈值处仍有非零横向与垂直偏差。

因此，CV **必须使用 common grid**。候选的输出节点数 `N=16/32/64/128/256` 可以不同，
但正式比较网格 `Q` 必须相同；否则节点更多的候选会改变“考题”本身。

## 2. 审核对象与非目标

### 2.1 模型任务

对航班 `i`，在固定预测锚点 `t=0` 已知历史状态后，模型输出：

\[
\hat{\mathbf p}_i(t)\in\mathbb R^3,\qquad
0\le t\le \hat T_i,
\]

其中 `p=(e,n,u)` 是同一跑道局部坐标系内的位置，`T_i` 是真实剩余时长，
`T_hat_i` 是预测剩余时长。当前正式模型是确定性预测器，因此本审核不把 oracle
`minADE_K/minFDE_K` 当作在线可实现的准确率。

### 2.2 不在本指标合同内的事项

- ADS-B 下载、航迹重建和 threshold-event fitting 的质量审计；
- LPV/LNAV-VNAV 门限本身的再设计；
- 障碍物净空、机体动力学认证或一次飞行是否“安全着陆”；
- outer-test 上的任何开发选择。

这些模块可以向模型提供输入或在模型输出后作独立验收，但不能与预测误差混成一个 loss。

## 3. 当前实现的完整盘点

| 层 | 当前核心口径 | 主要问题 |
|---|---|---|
| direct-state 训练 | 真实时间三维路径位置 + 输出端点位置 + final-time；未来速度由位置导出 | 三项分别对应整条曲线、到达几何和 4D 时钟；不再监督冗余 future velocity |
| checkpoint/CV | 正式默认固定为 `Q=64` airport-macro common true-time ADE | scheduler、early stop、checkpoint 和 CV 候选使用同一标量；其他 selector 只属显式实验模式 |
| fit evaluation | 同一 immutable replay 产生 common-time 主指标、分量、端点、ETA 和 QA | 不再用 native normalized loss 充当模型准确率；完整诊断只在训练结束后计算 |
| prediction summary | 固定真实时长上的 `Q=64` common-time 比较 | 提前结束保持其末点并继续受罚；另分开报告预测到达端点误差 |
| approach evaluation | 单个跑道阈值事件的 lateral/vertical/overall 三态 verdict | 这是终端验收，不是整条预测轨迹的模型准确率；不应进入 CV |

关键实现位置：

- `train.py` 的 `state_prediction_loss_components`：当前 direct-state 三任务 loss；
- `train.py` 的 validation selection：默认 airport-macro common-grid ADE selector；
- `train.py` 的 `evaluate_split`：训练结束后的完整 common-grid 诊断；
- `fixed_anchor_validation.py:356–445`：公共物理时间评估；
- `export.py` 的 `observed_series_metrics` / `accuracy_block`：prediction summary；
- `metrics.py` 的 `common_physical_time_flight_metrics`：独立于模型节点数的单航班核。

## 4. 已发现的实质问题

### 4.1 逐轴标准化不是中性的

当前已训练 iTransformer checkpoint 的位置标准差是：

\[
(\sigma_e,\sigma_n,\sigma_u)
=(9108.30,9379.10,577.61)\ \mathrm m.
\]

逐轴 normalized MSE 对同样一米误差的平方惩罚与 `1/sigma^2` 成正比。因此一米垂直
误差分别约等于一米 east/north 误差的 `248.7/263.6` 倍。这不是航空标准给出的权重，
只是训练样本分布产生的隐式权重。

位置应先还原为米，再用一个各向同性三维距离计算。输入特征仍可逐轴标准化；问题只在于
不能把输入 normalization 直接当作输出物理误差的价值函数。

### 4.2 复杂 loss 实际贡献很小

现有 pooled iTransformer 的最佳 epoch 150：

| 已加权 component | validation 值 | 占总 loss |
|---|---:|---:|
| state | 0.181687 | 80.45% |
| final time | 0.043173 | 19.12% |
| kinematic consistency | 0.000731 | 0.32% |
| terminal | 0.000249 | 0.11% |
| total | 0.225841 | 100% |

运动学和终点项合计不足 `0.5%`，却引入未来速度标签、额外计算、多个参数和大量诊断代码。
既有 ablation 还表明，提高运动学权重可使曲线更平滑，但会明显恶化 ADE；终点重权重则
多次得到“终点很准、中段很差”的轨迹。

### 4.3 native progress、true time 和 overlap 不是同一个问题

定义：

- normalized progress：比较 `p_hat(q*T_hat/Q)` 与 `p(q*T/Q)`；
- true physical time：比较 `p_hat(q*T/Q)` 与 `p(q*T/Q)`；
- overlap only：只比较 `0 <= t <= min(T,T_hat)`。

在 2167 条既有 validation 预测上，airport-macro ADE 为：

| 对齐口径 | ADE |
|---|---:|
| true physical time | 1532.1 m |
| normalized progress | 1254.9 m |
| overlap only | 1561.2 m |

逐航班排序的相关性：

- true-time vs normalized-progress：`0.7625`；
- true-time vs overlap：`0.9984`。

normalized-progress 对单航班 true-time ADE 的相对差异中位数为 `-17.7%`，且 `67.7%`
航班的绝对相对差异超过 `25%`。它测量的是“路径形状相似”，会把一部分到达时间错误
重新参数化掉。

overlap 在这一个模型内看似接近 true-time，但在候选模型之间会改变排序。例如同一旧
validation cohort 上：

| 候选 | overlap ADE | true-time ADE |
|---|---:|---:|
| control, lr=`1e-4` | **1583.5 m** | 3357.2 m |
| control, `N=16` | 2160.7 m | **2373.9 m** |

只看 overlap 会选择第一项，而真实时间预测明显选择第二项。原因是第一项平均到达时间
误差更大，错误尾部被 overlap 截掉了。

### 4.4 airport-macro 与 flight-micro 不应混用

现有训练通过航班权重使每个机场贡献相同；CV selector 也按机场先平均。但部分 fit report
把所有航班直接合并。以上同一 validation 预测在 Q=64 下：

\[
ADE_{micro}=1444.2\ \mathrm m,\qquad
ADE_{airport\text{-}macro}=1533.4\ \mathrm m.
\]

差约 `6.2%`，因为 KRDU 等机场样本更多且机场难度不同。正式主指标必须与训练人口政策
一致，采用 airport-macro；micro 只能作为描述性补充。

### 4.5 FDE 与预测到达端点误差回答不同问题

既有 validation-only predictability report 显示：训练库近邻 oracle 的平均未来离散度为
`1666.1 m`，但终点离散度只有 `19.3 m`，因为所有目标都在同一跑道阈值平面附近结束。
其 `K=20 minADE=453.5 m`，而 `minFDE=4.85 m`。

这不等于终点坐标已知。对五个机场各抽取 5 条当前 lateral-pass 航迹后，实际末端局部
坐标仍出现约 `-13.1…+7.5 m` 的水平分量和 `-8.9…+17.5 m` 的垂直分量。阈值平面和
目标跑道已知，**实际穿越偏差未知且正是 downstream verdict 要评价的量**。因此不能使用
endpoint-constrained curve 把终点写死；否则所有预测都会人为命中目标，终端 evaluation
失效。

但“不能写死终点”不等于“终点无需监督”。后续 development 审计发现，仅用 `Q=64`
路径平均时，最后一个查询点只占路径任务的 `1/64`；`N=16` baseline 的 validation
预测到达端点误差仍约 `1.9 km`。因此必须区分：

- `FDE_true`：在真实到达时间 `T_i` 查询预测曲线，既受路径误差也受时钟误差；
- `E_end`：在模型自己的预测到达时间 `T_hat_i` 查询预测曲线，再与该航班真实终点比较，
  主要描述终点几何。

最终设计保留一个**软的、对真实航班终点的监督项**，不使用跑道中心或 TCH 作为硬编码
答案。它不参与 checkpoint selector；正式 selector 仍只用 whole-path ADE。

## 5. 正式 validation 的严格数学定义

### 5.1 公共时间网格

令机场集合为 `A`，机场 `a` 的 validation 航班集合为 `V_a`。对每条航班使用同一个
`Q=64`：

\[
t_{iq}=\frac{q}{Q}T_i,\qquad q=1,\ldots,Q.
\]

真实轨迹和预测轨迹都按各自有序物理时钟作线性插值。预测定义域外采用以下确定性规则：

\[
\hat{\mathbf p}_i(t)=
\begin{cases}
\text{interpolate predicted nodes at }t,&0\le t\le\hat T_i,\\
\hat{\mathbf p}_i(\hat T_i),&t>\hat T_i.
\end{cases}
\]

这样，提前结束不会通过删掉尾部逃避处罚；预测较晚时，在真实终点 `T_i` 仍可看见飞机
位于何处。任何非有限坐标、非有限时长、非正时长或非单调预测时钟都使该航班预测无效，
不能生成 NaN 后继续参与平均。

### 5.2 主指标

三维误差向量和距离为：

\[
\mathbf e_{iq}=\hat{\mathbf p}_i(t_{iq})-\mathbf p_i(t_{iq}),\qquad
d_{iq}=\lVert\mathbf e_{iq}\rVert_2.
\]

先对每条航班平均：

\[
ADE_i=\frac1Q\sum_{q=1}^{Q}d_{iq}.
\]

再对每个机场平均，最后等权平均机场：

\[
ADE_a=\frac1{|V_a|}\sum_{i\in V_a}ADE_i,
\qquad
\boxed{S=\frac1{|A|}\sum_{a\in A}ADE_a}.
\]

`S` 是唯一的开发选模标量，单位为米。以下四处必须使用同一 `S`：

1. CV 候选排序；
2. `ReduceLROnPlateau`；
3. early stopping；
4. retained checkpoint。

CV 的候选分数是各 outer-train fold 的 `S` 的算术平均，取最小者；若浮点值完全相同，
按预先生成的候选顺序决定，不能临时查看 FDE、pass rate 或另一诊断指标后改变赢家。

### 5.3 必须报告、但不混入 `S` 的指标

令 `e=(e_e,e_n,e_u)`，则：

\[
h_{iq}=\sqrt{e_e^2+e_n^2},\qquad v_{iq}=e_u.
\]

从真实位置公共网格的中心差分得到真实水平切向单位向量
`t_iq=(t_e,t_n)`，不要再次依赖可能带噪的未来 velocity label：

\[
a_{iq}=e_e t_e+e_n t_n,
\qquad
c_{iq}=-e_e t_n+e_n t_e.
\]

正式报告至少包含：

| 指标 | 聚合和单位 | 用途 |
|---|---|---|
| 3D ADE | per-flight mean/p50/p95；per-airport；airport-macro，m | 主准确率 |
| horizontal ADE | 同上，m | 水平整体误差 |
| signed along-track | mean bias、MAE、p95，m | 时间/速度造成的沿程偏差 |
| signed cross-track | mean bias、MAE、p95，m | 路径形状/横向偏差 |
| signed vertical | mean bias、MAE、p95，m | 垂直剖面偏差 |
| final-time error | signed mean、MAE、p95，s | CTA/到达时间能力 |
| FDE at `t=T_i` | mean/p95，m | 真实到达时刻的位置误差，不选模 |
| arrival endpoint error | mean/p50/p95，m | `p_hat(T_hat_i)` 对真实终点的几何误差，不选模 |
| invalid/coverage | count/rate | 防止丢弃困难样本后虚假变好 |
| inference runtime | median/p95，ms/flight | 工程部署能力 |

还应按 `q/Q` 或剩余时间分箱画误差曲线，但图上的每个点仍先按航班、再按机场聚合。

### 5.4 为什么 Q 固定为 64

在 2167 条既有 validation 预测上，把同一预测重采样到不同公共网格：

| Q | airport-macro ADE | 相对 Q=256 |
|---:|---:|---:|
| 8 | 1586.06 m | +3.96% |
| 16 | 1556.51 m | +2.02% |
| 32 | 1541.65 m | +1.05% |
| 64 | 1533.44 m | +0.51% |
| 128 | 1528.31 m | +0.17% |
| 256 | 1525.65 m | baseline |

Q=64 与 Q=256 的逐航班排序相关性为 `0.99998`。因此 Q=64 已达到良好的精度/成本
平衡。`N` 是模型输出分辨率，`Q` 是考核分辨率；二者不应绑定。

## 6. 推荐的最小训练 loss

### 6.1 输出合同

历史输入仍可为：

\[
(e,n,u,\dot e,\dot n,\dot u)_{t-L+1:t}.
\]

未来优化目标只使用：

\[
\hat{\mathbf p}_{i1:N},\qquad \hat T_i,
\]

并在 loss 内把这 `N` 个节点可微地采样到固定的 `Q` 个真实时间查询点。`N` 属于模型，
`Q` 属于考核/训练对齐合同。

速度、航向和下滑角由预测位置曲线与预测时钟求导：

\[
\hat{\mathbf v}_i(t)=\frac{d\hat{\mathbf p}_i(t)}{dt}.
\]

这不会丢失下游需要的速度；它只删除“模型同时输出一条位置曲线和一条可能与其矛盾的
速度曲线”的自由度。

跑道阈值定义了参考平面和期望中心/TCH，但没有给出该航班将实际穿越平面的横向、高度
偏差。因此输出曲线的末端必须保持可学习；不能采用 `p_hat(1)=p_target` 的硬端点约束。
输出端点保持可学习，并以该航班的真实监督端点作为软目标；它从不被替换为跑道中心/TCH。

### 6.2 loss 定义

训练位置误差使用与 validation 相同的真实时间查询点。令：

\[
\hat\tau_{iq}=\operatorname{clip}\left(\frac{t_{iq}}{\hat T_i},0,1\right).
\]

各向同性位置项：

\[
L_{p,i}=\frac{1}{Q s_p^2}
\sum_{q=1}^{Q}
\left\lVert
\hat{\mathbf p}_i(\hat\tau_{iq})-\mathbf p_i(t_{iq})
\right\rVert_2^2.
\]

到达时间项：

\[
L_{T,i}=\left(\frac{\hat T_i-T_i}{s_T}\right)^2.
\]

令 `q_i` 为该样本最后一个有位置监督的输出节点（固定长度模式可在其后带 padding），
端点项为：

\[
L_{E,i}=\frac{\left\lVert
\hat{\mathbf p}_{i,q_i}-\mathbf p_{i,q_i}
\right\rVert_2^2}{s_p^2}.
\]

机场宏平均总 loss：

\[
\boxed{
L=\frac1{|A|}\sum_{a\in A}\frac1{|B_a|}
\sum_{i\in B_a}(L_{p,i}+0.25L_{E,i}+L_{T,i})
}.
\]

最小 baseline 使用 `s_p=10,000 m`、`s_T=600 s`。端点与路径使用同一个 `s_p`，避免
引入第二套物理单位；`0.25` 是在冻结 development cohort 上按 Pareto 结果预注册的任务
权重。`s_p`、`s_T` 是训练数值尺度和位置/时间
任务权衡，**不是航空通过门限**。它们必须写入 checkpoint 与报告，并在全量 CV 前冻结；
不能根据 outer-validation/test 临时调节。

`L_T` 不能因 true-time 路径项已包含时钟影响而直接删除。冻结 cohort 的两种子消融显示，
删除显式时间项后 ADE 可维持在 `1312.9–1389.5 m`，但 ETA MAE 恶化到
`553.4–723.7 s`。模型可以用一条很长的预测时钟，只让曲线前缀覆盖真实时域；路径位置
仍然接近，但模型自己声明的到达时刻已经失效。因此位置曲线与总时长是两个不可合并的
输出任务，`L_T` 是必要项。

训练用平方误差是为了稳定优化；正式模型比较仍使用物理单位 ADE、误差分布和时间 MAE。
这一做法也与 iTransformer/PatchTST 原论文使用 MSE 训练、MAE/MSE 类物理误差评估的
基本范式一致，但这里去掉了逐轴物理权重和冗余未来速度目标。

### 6.3 明确删除的 loss 项

| 当前项 | 处理 | 原因 |
|---|---|---|
| 六维 future-state MSE | 改为三维物理位置 | 未来速度冗余且会与位置不一致；逐轴标准化隐藏物理权重 |
| kinematic consistency | 删除 | 速度由位置曲线求导后，一致性是结构性质，不再需要惩罚 |
| 旧 terminal position | 替换为 `0.25 L_E` | 旧项过弱且语义混杂；新项只监督真实输出端点，不强制命中跑道中心 |
| arc-length/path-length scalar | 不进入正式 loss | 同一路径长度可以对应完全不同的曲线 |
| acceleration/jerk/smoothness | 只保留离线 QA | 没有项目级可接受上限时，不应主导准确率选模 |
| approach pass rate | 不进入训练/CV | 单个终端事件，离散且大量候选并列；不能衡量中段 |

Kendall–Gal–Cipolla §3 说明多任务性能会强烈依赖手工权重，并给出可学习同方差不确定性
权重。当前设计暂不引入它：位置和时间只有两个明确任务，固定尺度更容易审计。只有在
train/validation 上预注册的 `s_T` 敏感性实验显示结论不稳定时，才考虑两个 learned
log-variance；不应一开始再增加新机制。

### 6.4 control-output loss 的审核边界

`prediction_output=control` 不是当前 production baseline，而是显式动力学消融。它最后
仍生成一条位置—时间曲线，因此**训练完成后的正式 evaluation 完全相同**：同一个 Q=64
common-time ADE、FDE、arrival endpoint、ETA 和阈值 verdict，不允许为 control 另造更有利
的 headline metric。

其现有训练代码提供 normalized-MSE、physical-criteria、terminal-state 和 arc-length 多个
recipe。这些 recipe 的用途是研究“怎样训练控制量”，不能同时被解释成四套正式模型质量
定义；尤其默认 normalized-channel MSE 仍有第 4.1 节的隐式轴权重问题。若未来要把
control-output 晋升为正式候选，最小目标应复用：

\[
L_{control}=L_p+0.25L_E+L_T
+\lambda_u L_{effort}+\lambda_{\Delta u}L_{smooth},
\]

前三项必须与 state-output 使用相同物理定义；后两项只负责避免不可执行或高频抖动控制，
且使用按飞机控制包线归一化的现有定义。是否需要以及如何冻结 `lambda_u`、
`lambda_Delta_u` 必须另做 control 专用 development ablation。当前证据不足以把 state 模型
的小实验结果直接移植到昂贵 RK4 control 训练，因此本轮不改它的实验 recipe，也不把它
混入接下来的默认 state-model CV。这是明确的适用边界，不是遗漏。

## 7. 小规模开发实验

### 7.1 协议

- 当前 lateral-pass eligibility 数据；五个美国机场；
- 每机场 100 条 outer-train、50 条 outer-validation；候选选择前先验证可构建性，最终
  精确重建 500 train、250 validation；
- outer split seed 固定为 1337；模型初始化 seed 为 1337、2027；
- 轻量 iTransformer：`d_model=64`、2 层、`N=64`、batch 64、最多 100 epochs；
- 所有组均用 `Q=64 airport-macro true-time ADE` 调度、早停与选 checkpoint；
- 没有创建正式 checkpoint，没有读取 outer-test。

为了只审核 loss，实验仍使用现有六通道 tensor 合同，但未来速度不接收监督梯度，发布前
由预测位置和时钟统一导出；因此结果证明“删掉冗余监督”有价值，不声称改变 backbone。

### 7.2 两个种子的结果

| 训练目标 | seed 1337 | seed 2027 | 两次均值 |
|---|---:|---:|---:|
| 当前六维复合 loss | 3579.7 m | 3605.0 m | 3592.4 m |
| normalized-progress 位置 + 时间 | 2584.3 m | 2351.0 m | 2467.6 m |
| **true-time 位置 + 时间** | **2094.4 m** | **2054.6 m** | **2074.5 m** |

true-time 简化 loss 相对当前复合 loss 的两次 improvement 分别为 `41.5%` 和 `43.0%`；
相对 progress 简化项还改善 `19.0%` 和 `12.6%`。这与第 4.3 节的对齐审计一致：时钟
属于 4D 轨迹，不能在训练时被归一化进度完全消掉。

另测试了把时间误差乘以每条真实轨迹平均速度、转换为“等效沿程米数”的无手工单位方案。
在 seed 1337 下：

| 时间项 | ADE | time MAE |
|---|---:|---:|
| fixed `s_T=600 s` | **2094.4 m** | 61.1 s |
| speed-equivalent distance | 2313.4 m | **47.9 s** |

它说明空间准确率和 ETA 准确率确实存在 trade-off；不存在一个由航空规范自动给出的万能
加权总分。由于本项目主问题是整条 4D 位置轨迹，baseline 选择 ADE 更好的固定尺度，
同时把 time MAE 作为独立正式指标。若研究目标改成 CTA 优先，必须显式改变任务要求，
不能暗中换权重。

### 7.3 输出节点数 `N` 与 validation 网格 `Q` 的消融

保持 `Q=64`、模型和 cohort 不变，仅改变模型原生输出节点数：

| N | validation ADE | FDE at true time | time MAE | raw acceleration p95 | raw jerk p95 |
|---:|---:|---:|---:|---:|---:|
| 16 | **1414.2 m** | **2026.3 m** | 31.4 s | **10.10 m/s²** | **1.81 m/s³** |
| 32 | 1457.1 m | 2208.4 m | 35.6 s | 44.74 m/s² | 14.48 m/s³ |
| 64 | 1417.7 m | 2098.6 m | **28.4 s** | 209.5 m/s² | 169.2 m/s³ |

同一 `Q=64` 观测 baseline 的 acceleration/jerk p95 约为 `3.96 m/s²` / `2.43 m/s³`；
按同为 16 节点的公平网格则约为 `2.12 m/s²` / `0.10 m/s³`。这说明：

1. `Q` 只负责公平考核，不能随候选 `N` 改变；
2. `N` 是实质模型容量/平滑度超参数；旧 CV 只测 `64/128/256` 会漏掉有效区域；
3. 先把 `16/32` 加入 CV，比增加一个缺乏阈值依据的 acceleration loss 更简约。

因此正式 CV 保留原候选并扩成 `N={16,32,64,128,256}`。运动学只报告 prediction 与
同分辨率 observed baseline，不进入 selector。

### 7.4 输出端点 loss 的两种子消融

用 `N=16` 比较无端点项和候选权重。下表的 endpoint 是
`||p_hat(T_hat)-p(T)||`，不是在真实时间查询的 FDE：

| seed | endpoint 权重 | ADE | FDE true-time | endpoint error | time MAE | accel p95 |
|---:|---:|---:|---:|---:|---:|---:|
| 1337 | 0 | 1414.2 m | 2026.3 m | 1942.9 m | 31.4 s | 10.10 m/s² |
| 1337 | 0.25 | **1372.9 m** | **1803.1 m** | **1289.2 m** | **28.1 s** | 10.57 m/s² |
| 1337 | 0.5 | 1400.3 m | 1839.4 m | 1293.2 m | 28.4 s | 12.09 m/s² |
| 1337 | 1.0 | 1426.5 m | 1792.7 m | 1255.9 m | 27.8 s | 13.07 m/s² |
| 2027 | 0 | **1400.1 m** | 2032.0 m | 1900.1 m | 30.6 s | **10.36 m/s²** |
| 2027 | 0.25 | 1417.5 m | **1877.0 m** | **1371.5 m** | **28.0 s** | 14.00 m/s² |

两种子均值中，`0.25` 相对 `0`：ADE 从约 `1407.1` 降至 `1395.2 m`，endpoint 从约
`1921.5` 降至 `1330.4 m`，FDE 从约 `2029.1` 降至 `1840.1 m`，time MAE 从约 `31.0`
降至 `28.1 s`。它是唯一在两种子均值上同时改善整条路径、终点和 ETA 的最小候选；更大
权重只换来很少的 endpoint 收益，并恶化 ADE/运动学。因此冻结为 `0.25`，不把权重继续
扩成正式 CV 维度。

### 7.5 时间任务尺度的敏感性

`s_T` 不是航空法规门限，而是空间与时间两项之间必须显式声明的研究取舍。在同一
`N=16`、endpoint weight `0.25`、两个种子的 cohort 上：

| `s_T` | 两种子平均 ADE | 两种子平均 FDE | 平均 endpoint error | 平均 ETA MAE |
|---:|---:|---:|---:|---:|
| 300 s | 1421.1 m | **1833.8 m** | 1343.6 m | **24.0 s** |
| 600 s | **1395.2 m** | 1840.1 m | **1330.4 m** | 28.1 s |

两者不存在全指标上的绝对支配：300 s 更重视 ETA，600 s 更重视本项目预注册的主指标
whole-path ADE，并略改善预测到达端点。由于当前论文问题以完整 4D 位置曲线为主、ETA
作为独立 secondary metric，正式 baseline 冻结 `s_T=600 s`。如果研究问题改成 CTA
优先，应新建并预注册一个 CTA 实验，而不是在看到正式 validation/test 后换尺度。

### 7.6 低 N 训练真值的近似边界

正式 validation 始终直接从原始监督时间轴采样 Q=64 真值。当前训练张量为了保持每个
候选自己的 `N` 节点表示，会先在 N 个节点上取真值，再把该分段线性表示查询到 Q=64。
它不改变正式评分，但意味着低 N 模型的训练目标是“该模型分辨率可表达的真值近似”，
而不是保留原始轨迹的所有局部折点。

在冻结的 750 条 development 航迹上，相对直接 Q=64 真值的三维重建误差为：

| N | 所有查询点 mean | 查询点 p95 | per-flight mean p95 |
|---:|---:|---:|---:|
| 16 | 18.3 m | 84.6 m | 65.4 m |
| 32 | 5.2 m | 18.6 m | 14.7 m |
| 64 | 约 0 m | 约 0 m | 约 0 m |

相对于当前约 1.4 km 的 validation ADE，这一近似不足以证明值得为 training batch 再维护
一套独立 Q64 真值张量；而且 N=16 的分段线性输出本身也无法表达这些高频折点。最简设计
因此保留当前训练表示，并把**正式 validation 的原始 Q64 真值**作为最终裁判。若全量
validation 的 ADE 已降到百米量级，再重新审计这一近似；不要现在提前增加数据接口。

## 8. 正式训练后 evaluation

### 8.1 development 阶段

每个冻结 checkpoint 对 train 和 validation 各做一次 immutable replay，随后所有指标
都从同一 replay 计算，不能为每个图表重新推理。正式 JSON 的最小结构应为：

```text
contract
  split identities + hashes
  checkpoint hash
  anchor policy
  Q=64 and interpolation/extrapolation rule
  airport-macro aggregation

primary
  airport_macro_3d_ade_m

secondary
  per-airport 3D ADE
  per-flight p50/p95
  horizontal / along / cross / vertical
  final-time error
  FDE at true arrival time
  predicted-arrival endpoint error
  invalid and coverage
  inference runtime

quality_assurance
  raw turn / acceleration / jerk
  same-resolution observed baseline

terminal_evaluation
  reference to the separate evaluation report and criteria
```

不再把完整 per-node 数组、数千个 arc/control scalar 或重复的 train/val 图表塞进 headline
report。需要调试时可单独生成 diagnostics artifact。

### 8.2 final release

在模型结构、loss、CV grid、训练预算和报告 schema 全部冻结后，才允许执行一次
`--split test --release-test`。final test 使用完全相同的 `Q=64` 和聚合合同，并保留
checkpoint 邻接的 `test_release.json`。任何 test 结果暴露后的修改都会使该分区变为
development test，不能靠改 split seed 恢复盲性。

### 8.3 概率模型的未来扩展

当前确定性 baseline 用 ADE/FDE 合理。若以后输出多条候选或概率分布：

- 必须增加 proper scoring rule（例如轨迹 likelihood/NLL）和 calibration/coverage；
- `minADE_K/minFDE_K` 只能作为候选集合 coverage；
- 不能把知道 ground truth 后选出的 oracle 最佳轨迹写成在线模型准确率。

Trajectron §4 明确指出 ADE/FDE 适合确定性回归，但不能评价分布的方差和多模态性；其
best-of-N 也需要事后知道哪条样本最接近真值。

## 9. 与最终进近 verdict 的边界

`evaluation/FINAL_APPROACH_VERDICT_STANDARD.md` 回答的是：

> 在跑道阈值平面处，轨迹参考点是否在选定跑道和发布 TCH 的允许范围内？

模型 validation 回答的是：

> 从固定锚点到真实阈值时间，预测的整条 4D 位置曲线与真实曲线相差多少？

两者关系如下：

```text
model prediction
  ├─ whole-path validation: common-grid ADE + decomposed errors + ETA
  └─ terminal acceptance: threshold lateral/vertical/overall verdict
```

阈值 verdict 继续使用当前横向规则与 `±22 m` 垂直规则，保持 pass/fail/indeterminate。
它作为独立 downstream evaluation 发布，不能代替 ADE，也不能因一个阈值事件通过就宣称
整条预测正确。

## 10. 实施状态与后续运行顺序

当前已经完成且不会改写 raw/downloaded 航迹：

1. common-time 纯 metric kernel；提前结束的尾部不再被截掉；
2. 默认 selector 统一为 `Q=64` airport-macro ADE，供 CV、scheduler、early stop 和
   checkpoint 共用；
3. direct-state loss 改为第 6.2 节的路径 + 端点 + 时间三项；future velocity 不再监督，
   而由发布曲线导出；
4. fit evaluation 与 prediction summary 使用同一 true-time 定义；FDE 与预测到达端点
   error 分开；
5. raw kinematics 增加同分辨率 observed baseline；它属于 QA，不选模；
6. CV 的 `N` 候选加入 `16/32`，progress checkpoint 原子写入，可断点续跑。

后续只按以下顺序继续：全量单元测试 → development CV → 冻结最优候选 → outer-train +
validation checkpoint → train/validation 报告。仍不打开 outer-test；只有用户明确宣布最终
release 后才能执行一次 test。

## 11. 正式合同与实验模式的边界

为避免“多个指标都像正式答案”，以下内容不属于默认 production contract：

- `fixed-anchor-objective` selector；
- `fixed-anchor-common-grid-criteria` 的 ADE/FDE smooth-max；
- terminal-state selector；
- arc-length-geometry selector；
- prediction summary 的 overlap-only accuracy；
- future velocity 监督和 kinematic consistency；
- 旧的 control/oracle terminal loss（direct-state 的 `0.25 L_E` 是新的明确输出端点任务）；
- 把 raw acceleration/jerk 或 approach pass rate用于 checkpoint selection。

现有 control、oracle 和其他 selector 仍作为显式研究消融保留，避免破坏既有实验功能；
`run_ts_pipeline.py` 默认路径不会选用它们。旧的可再生 checkpoint/summary schema 不做
双读迁移，必须用新合同重新生成。

## 12. 依据与适用边界

### 12.1 官方航空资料

1. [EUROCONTROL FDPS Guidance Edition 1.0 (2024)](../../docs/metrics_review/EUROCONTROL-GUID-199-FDPS-Ed1.0-2024.pdf)：
   §5.13，PDF pp. 50–52 分别给出 longitudinal、vertical、lateral accuracy；Annex A
   §§3.2–3.4，PDF pp. 108–114 要求在固定预测时刻比较，报告 peak、signed mean/bias 和
   standard deviation。它支撑“物理时间对齐 + 分量分开报告”，但面向中期冲突探测，
   其数值不能直接当最终进近或 ML 模型门限。
2. [EUROCONTROL Trajectory Prediction Specification Edition 2.0 (2017)](../../docs/metrics_review/EUROCONTROL-SPEC-0143-Trajectory-Prediction-Ed2.0.pdf)：
   2024 guidance 的正式前身；当前官方 specification 页面仍列 Version 2.0。
3. [NASA/FAA trajectory-predictor validation methodology](../../docs/metrics_review/NASA-20110003565-Validation-Methodology-for-Aircraft-Trajectory-Predictors.pdf)：
   §II.A，PDF pp. 2–3 要求指标显式刻画“有定义准确率限制”的方面，并从预期用途比较
   predicted 与 actual behavior；§VI.B，PDF pp. 17–19 分开使用 time-coincident
   horizontal、spatially coincident cross-track/along-track 诊断误差来源。它反对只靠一个
   不可诊断的 pass/fail 总分。

### 12.2 原始 ML/trajectory 文献

1. [Kendall, Gal, Cipolla, CVPR 2018](../../docs/metrics_review/Kendall-Gal-Cipolla-2018-Multitask-Uncertainty-Loss.pdf)：
   §3、Eq. (1)–(7) 说明手工加权多任务 loss 对量纲和权重敏感，并给出 learned
   homoscedastic uncertainty 方案。本项目用它说明风险，而不是无条件增加该机制。
2. [Ivanovic, Pavone, ICCV 2019](../../docs/metrics_review/Ivanovic-Pavone-2019-Trajectron.pdf)：
   §4 定义 ADE/FDE，并说明概率分布不能只靠 ADE/FDE，best-of-N 不是在线选择器。
3. 本仓库的 iTransformer、PatchTST 原论文副本：两者均采用简单 MSE/MAE 类回归口径，
   没有为航空位置、速度、终点和飞行力学给出通用复合权重。

完整文件版本、精读章节和 SHA-256 见
[metrics_review/README.md](../../docs/metrics_review/README.md)。

## 13. 证据强度与尚待验证事项

高置信度、可以直接实施：

- CV 必须使用统一 Q=64 common physical-time grid；
- primary selector 为 airport-macro 3D ADE；
- overlap-only 与 normalized-progress 不再作正式准确率；
- 分量和 ETA 分开报告；threshold verdict 与模型准确率分层；
- 每 epoch 只算 lean selector，完整 diagnostics 只在训练结束后算。

已由单元测试和两种子小实验支持：

- true-time position + time loss 优于当前六维复合 loss；
- future velocity、kinematic 可从正式 direct-state 训练目标中移除；
- 一个小权重的真实输出端点任务是必要的，`0.25` 比 `0/0.5/1.0` 的综合 Pareto 更好；
- 独立时间项不能删除；删除后 ETA MAE 恶化到约 9–12 分钟；
- `s_T=300/600 s` 存在可解释的 ETA/ADE trade-off，当前 whole-path 主目标选择 600 s；
- `N=16/32` 必须进入 CV，而 `Q=64` 保持固定。

仍需由完整 development CV 回答：

- 两个 backbone 在新增低 `N` 候选下各自的最优超参数；
- 小 cohort 的 `N=16` 优势能否在全部 outer-train folds 保持；
- 完整 validation 上 endpoint、ETA 与 raw-kinematic baseline 的分布。

`s_T=600 s` 与 endpoint weight `0.25` 已在进入正式 CV 前冻结，不再扩成 loss-weight 搜索；
否则 CV 会同时承担模型容量、空间/时间价值判断和终点价值判断，既昂贵又无法解释。
