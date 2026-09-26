# 多机轨迹预测中的可变长度与终止条件

> 调研日期：2026-07-26  
> 适用模块：`4dTrajectory/ts_transformer`  
> 主题：归一化轨迹进度、剩余时间预测、多机交互和轨迹终止

## 1. 结论

如果只能在“归一化进度”和“预测剩余时间”中选择一个，**预测剩余时间（Time-to-Threshold / Time-to-Land）对多架飞机智能交互更友好**。

原因是多机交互依赖共享的真实时间轴。冲突检测、尾流间隔、进场排序、CTA 和跑道占用都需要回答：

> 在同一个绝对时刻，每架飞机分别位于哪里？

归一化进度适合学习不同长度轨迹的空间形状，但不同飞机处于相同进度并不表示它们处于同一时刻。因此，工程上更合适的设计不是二选一，而是：

1. 用归一化进度描述空间路径；
2. 用剩余时间或单调时间映射恢复真实时间；
3. 在共享时间轴上建模多机交互；
4. 用跑道入口穿越事件确定最终终点。

推荐的总体形式是：

```text
多机历史轨迹（共享真实时间轴）
        ↓
scene / agent interaction encoder
        ↓
多模态空间路径 x_i(s) + 单调时间映射 t_i(s)
        ↓
重新采样到共享未来时刻 x_i(t)
        ↓
冲突、排序、间隔和 CTA 计算
        ↓
threshold-plane crossing 精确终止
```

## 2. 当前实现及其限制

当前实现并不是在完整目标状态满足后停止，而是：

1. `full` 模式一次输出固定的 `pred_len`，默认 300 步；
2. `window` 模式递归预测，累计到至少 300 步；
3. 预测完成后计算每个点到跑道入口的水平距离；
4. 取全局水平距离最小的点作为轨迹终点。

等价于：

\[
k^*=\arg\min_k\sqrt{e_k^2+n_k^2}.
\]

它没有直接检查：

- 是否真正穿越跑道入口平面；
- 高度是否接近目标 TCH；
- 航向是否与跑道方向一致；
- 飞机是否仍在下降；
- 速度和下降角是否合理；
- 该点是否只是机场附近振荡产生的偶然最近点。

此外，这是一种事后截断：`full` 模式仍然计算全部 300 步；当前 `window` 模式也会先递归到固定上限，再统一截断。

## 3. 归一化进度与剩余时间的比较

### 3.1 归一化进度

将每条轨迹重新参数化为：

\[
s\in[0,1],
\]

其中 `s=0` 是预测起点，`s=1` 是跑道入口。

优点：

- 不同持续时间的轨迹可以表示成相同点数；
- 减少短轨迹 padding 和长轨迹截断；
- 更容易学习转弯、下滑和对准跑道等空间阶段；
- 适合路径聚类、轨迹模式学习和空间形状生成。

限制：

- `s` 不是时间；
- 两架飞机的 `s=0.7` 可能相差数分钟；
- 无法仅凭进度计算同时刻的水平/垂直间隔；
- 无法直接支持 CTA、排序和跑道时间窗；
- 若简单令 `t=sT`，会忽略等待、雷达引导、减速和非均匀飞行阶段。

因此，归一化进度不能单独作为多机交互的时间坐标。

### 3.2 剩余时间

为每架飞机预测：

\[
T_i=t_{i,\mathrm{threshold}}-t_{\mathrm{now}}.
\]

优点：

- 所有飞机共享同一个当前时刻；
- 可以直接计算预测到达顺序；
- 可以计算跑道入口时间间隔；
- 适合到达管理、CTA、排序和资源分配；
- 可以输出概率分布而不是单一点估计。

例如，可以计算：

\[
P(T_i<T_j)
\]

表示飞机 `i` 先于飞机 `j` 到达的概率，以及：

\[
P(|T_i-T_j|<S_{\min})
\]

表示两架飞机到达间隔不足的概率。

限制是单一的 `T_i` 只描述终点时间，无法说明中间航段在什么时候经过。因此剩余时间必须与完整的时变轨迹配合，才能用于中途冲突预测。

### 3.3 推荐：路径与时间解耦后再组合

对飞机 `i` 预测固定数量的归一化空间点：

\[
\mathbf{x}_i(s_k),\qquad s_k\in[0,1].
\]

同时预测每段的正时间增量：

\[
\Delta t_{i,k}=\operatorname{softplus}(z_{i,k}),
\]

并累加得到单调时间映射：

\[
t_i(s_k)=t_{\mathrm{now}}+\sum_{j=1}^{k}\Delta t_{i,j}.
\]

于是：

- `x_i(s)` 描述路径形状；
- `t_i(s)` 描述该路径如何随时间执行；
- `T_i=t_i(1)-t_now` 是剩余到达时间；
- `x_i(t)` 可以通过插值恢复到共享时间网格。

这种表示比只预测一个总时长更适合存在等待、转弯和速度调整的终端区轨迹。

## 4. 建议的多机模型结构

### 4.1 输入

场景输入应保持共享绝对时间：

```text
X: [batch, agents, history_steps, channels]
agent_mask: [batch, agents]
time_mask:  [batch, agents, history_steps]
```

建议补充的上下文包括：

- 跑道和进场程序；
- 飞机类型或尾流等级；
- 当前进场序列；
- 相对位置、相对速度和预计交汇时间；
- 天气、跑道使用方向和流量状态；
- 可用时，加入 ATC 指令或 intent 信息。

### 4.2 交互编码

可以采用 agent attention 或动态图结构：

- 每架飞机是一个 node/token；
- 边由距离、预计交汇、同跑道、同进场流等条件确定；
- agent attention 学习谁对谁产生影响；
- temporal encoder 学习每架飞机自身的历史运动。

不建议仅按空间最近邻建立交互，因为真正影响进场顺序的飞机未必是当前欧氏距离最近的飞机。

### 4.3 输出头

推荐至少包含三个输出头：

```text
trajectory head  → 多模态空间路径或共享时间轨迹
timing head      → TTA 分布或单调时间映射
mode head        → 每种路径/时间联合假设的概率
```

时间输出最好是概率形式，例如：

- Gaussian 的 `mu + log_sigma`；
- quantile：P10 / P50 / P90；
- Gaussian mixture；
- 离散时间 hazard / survival distribution。

若需要表达多架飞机到达时间的相关性，不能只预测互相独立的 marginal Gaussian；应进一步预测联合模式、arrival-order distribution 或带相关结构的时间分布。

## 5. 轨迹终止方案

### 5.1 跑道入口平面穿越

对当前项目，最适合的无需重训方案是检测飞机第一次穿越跑道入口平面。

在 runway-aligned frame 中，第一个水平轴是沿跑道方向，第二个是横向方向。对于从跑道前方飞向入口的轨迹，可检测：

```text
along_prev < 0
along_curr >= 0
along_velocity > 0
abs(cross_track) < broad_terminal_corridor
```

穿越位于两个采样点之间时，使用线性插值：

\[
\alpha=\frac{-a_{k-1}}{a_k-a_{k-1}},
\]

\[
\mathbf{x}_{\mathrm{cross}}
=\mathbf{x}_{k-1}+\alpha(\mathbf{x}_k-\mathbf{x}_{k-1}).
\]

这样可以避免 2 秒采样间隔引起的终点量化误差。

终止逻辑建议记录明确原因：

```text
threshold_crossing
closest_approach_fallback
predicted_tta_fallback
horizon_cap
```

### 5.2 TTA 辅助的终点选择

时间预测可以将几何终点搜索限制在合理窗口内：

```text
predicted TTA ± uncertainty window
        ↓
在窗口内寻找第一次有效 threshold crossing
        ↓
若无穿越，再使用局部最近点 fallback
```

这种方式比在全部 300 步上寻找全局最近点更不容易选择偶然的机场附近轨迹点。

### 5.3 学习式终止

可以给每个未来 step 预测离散 hazard：

\[
h_k=P(K_{\mathrm{end}}=k\mid K_{\mathrm{end}}\geq k,X).
\]

它比单个 EOS 标签更适合当前数据：

- 300 个 step 中通常只有一个终点，类别极度不平衡；
- 超过 300 步的航班可以作为 right-censored 样本；
- 可以得到终止时间分布和置信区间；
- 可用 NLL、Brier score 和 calibration error 评估。

不过，学习式终止不应完全替代几何事件检查。更稳妥的做法是让模型预测“什么时候到”，由跑道入口平面定义“是否真的到”。

## 6. 现有方法的主流设计

截至 2026 年，轨迹预测领域没有统一的动态 EOS SOTA。不同数据集规定不同预测时长，排行榜结果不能直接跨数据集比较。

### 6.1 固定共享预测时域

多数自动驾驶和航空预测方法仍在固定真实时间网格上输出未来轨迹。这样便于计算同时刻的 agent 间距离和联合指标。

[FlightBERT++](https://ojs.aaai.org/index.php/AAAI/article/view/27763) 采用 non-autoregressive multi-horizon 设计，一次预测多个未来 horizon，以减少自回归误差累积和推理开销。它代表了航空短中期预测中常见的“固定 horizon、直接多步输出”路线，而不是学习 EOS。

### 6.2 目标或意图条件预测

成熟的通用轨迹预测结构通常是：

```text
scene encoder
    → goal / intention candidates
    → goal-conditioned trajectory completion
    → multimodal scoring and selection
```

代表工作：

- [TNT: Target-driven Trajectory Prediction](https://proceedings.mlr.press/v155/zhao21b.html)：先预测未来目标状态，再生成目标条件轨迹；
- [DenseTNT](https://openaccess.thecvf.com/content/ICCV2021/html/Gu_DenseTNT_End-to-End_Trajectory_Prediction_From_Dense_Goal_Sets_ICCV_2021_paper.html)：从密集目标候选生成多模态轨迹；
- [MTR](https://papers.neurips.cc/paper_files/paper/2022/hash/2ab47c960bfee4f86dfc362f26ad066a-Abstract-Conference.html)：以 motion queries 表示全局意图，并进行局部运动细化；
- [PECNet](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/4423_ECCV_2020_paper.php)：先推断远期 endpoint，再生成社会一致的轨迹；
- [Y-Net](https://openaccess.thecvf.com/content/ICCV2021/html/Mangalam_From_Goals_Waypoints__Paths_to_Long_Term_Human_Trajectory_ICCV_2021_paper.html)：将 goal、waypoint 和 path 分层建模。

这些方法中的 endpoint 通常是固定预测时刻的位置，不一定是真实任务终点。对于已知跑道入口的 AeroViz 场景，空间终点的不确定性相对较低，更重要的是到达时间、路径模式和多机相互影响。

### 6.3 联合多智能体预测

代表性设计包括：

- [FJMP](https://openaccess.thecvf.com/content/CVPR2023/html/Rowe_FJMP_Factorized_Joint_Multi-Agent_Motion_Prediction_Over_Learned_Directed_Acyclic_CVPR_2023_paper.html)：通过有向交互图分解联合多机预测；
- [M2I](https://openaccess.thecvf.com/content/CVPR2022/html/Sun_M2I_From_Factored_Marginal_Trajectory_Prediction_to_Interactive_Prediction_CVPR_2022_paper.html)：将交互对象分解为 influencer 和 reactor；
- [MotionLM](https://openaccess.thecvf.com/content/ICCV2023/html/Seff_MotionLM_Multi-Agent_Motion_Forecasting_as_Language_Modeling_ICCV_2023_paper.html)：将连续运动离散为 token，并联合自回归生成多个 agent 的未来；
- [MAIFormer](https://arxiv.org/abs/2509.21004)：同时使用个体时空 attention 和 agent attention 预测多机航空轨迹；
- [FPG-SLSTM](https://research.polyu.edu.hk/en/publications/joint-prediction-of-multi-aircraft-trajectories-in-terminal-airsp/)：把进场模式作为 intent 先验，并通过 social pooling 建模多机影响。

这些研究的共同方向是：不能让每架飞机完全独立预测后再简单拼接；路径模式、到达时间和其他飞机的状态应共同进入 scene-level 表示。

### 6.4 概率式落地时间

[Probabilistic Multi-Agent Aircraft Landing Time Prediction](https://arxiv.org/abs/2512.08281) 与本问题最接近：模型联合编码多架飞机的历史轨迹和相互影响，为每架飞机输出剩余落地时间的均值和不确定性，并评估到达序列的一致性。

它说明多机航空交互中，时间预测本身可以作为独立而重要的任务，而不必只从预测轨迹的最后一个点间接推导。

### 6.5 可变 horizon

[VH-Diffuser](https://arxiv.org/abs/2509.11930) 先用 Length Predictor 估计实例特定 horizon，再使用随机长度轨迹片段训练 diffusion planner。它是可变长度设计的直接参考，但属于目标条件轨迹规划，而不是多机航空轨迹预测，因此不应直接称为航空预测 SOTA。

## 7. 推荐的渐进实施路线

### 阶段 A：不改变模型

1. 保留 300 步作为安全上限；
2. 将全局水平最近点改为入口平面首次穿越；
3. 对穿越点做时间和状态插值；
4. 增加局部最近点 fallback；
5. 输出结构化的 `termination_reason`。

该阶段改善终点语义，但不会减少 `full` 模式的前向计算。

### 阶段 B：增加概率 TTA head

在现有 trajectory encoder 上增加：

```text
time head → TTA distribution
```

当前数据中已有 fitted final-approach tail 和 terminal supervision，可由：

```text
terminal_supervision_time - anchor_time
```

构造剩余时间标签。

可首先比较：

- point regression：Huber loss on `log1p(TTA)`；
- Gaussian NLL：预测 `mu, log_sigma`；
- quantile loss：预测 P10/P50/P90；
- survival/hazard loss：显式处理超过 horizon 的样本。

### 阶段 C：多机交互编码

将单航班样本重构为同时刻的 traffic scene：

```text
scene timestamp
    ├── aircraft A history
    ├── aircraft B history
    ├── aircraft C history
    └── masks / runway / procedure context
```

然后对 trajectory、TTA 和 arrival order 进行联合训练。

### 阶段 D：归一化路径 + 单调时间映射

只有在固定时间轨迹的 padding、截断或长尾问题被实验确认是主要瓶颈后，再引入完整的 progress/time-warp decoder。该阶段会改变数据表示、loss、指标和下游输出契约，应作为独立消融实验，而不是与其他修改同时进行。

## 8. 建议的实验和指标

建议至少比较以下实验单元：

| 实验 | 空间表示 | 终止方式 | 时间输出 |
|---|---|---|---|
| A | 固定 2 s 网格 | 全局最近点 | 无 |
| B | 固定 2 s 网格 | threshold crossing | 无 |
| C | 固定 2 s 网格 | TTA 窗口 + crossing | point TTA |
| D | 固定 2 s 网格 | TTA 窗口 + crossing | probabilistic TTA |
| E | normalized progress | crossing | total TTA |
| F | normalized progress | crossing | monotonic time warp |

单机精度：

- ADE / FDE；
- along-track / cross-track / altitude error；
- threshold crossing position error；
- TTA MAE / RMSE；
- premature-stop rate；
- missed-stop rate；
- horizon-cap rate。

概率质量：

- negative log-likelihood；
- CRPS；
- interval coverage；
- calibration error。

多机交互：

- arrival-order Kendall tau / Spearman correlation；
- pairwise spacing error；
- minimum-separation violation rate；
- joint ADE / joint FDE；
- predicted conflict precision / recall；
- 不同 traffic-density 分组下的性能。

不能只比较轨迹 ADE，因为一个空间误差较小但到达时间错位的模型，仍可能给出错误的冲突和排序结论。

## 9. 综述和推荐阅读

### 航空轨迹预测

1. [The Evolution and Taxonomy of Deep Learning Models for Aircraft Trajectory Prediction: A Review of Performance and Future Directions](https://www.mdpi.com/2076-3417/15/19/10739)（2025）  
   较新的航空深度学习轨迹预测综述，覆盖 RNN、attention、生成模型、图模型和混合模型。

2. [Aircraft 4D Trajectory Prediction in Civil Aviation: A Review](https://www.mdpi.com/2226-4310/9/2/91)（2022）  
   覆盖状态估计、动力学和机器学习方法，并讨论航空轨迹预测的完整处理流程。

3. [Aircraft trajectory prediction and synchronization for air traffic management applications](https://www.sciencedirect.com/science/article/pii/S037604212030052X)（2020）  
   与共享时间轴、TBO、轨迹同步和多参与方决策最相关。

### 多智能体和多模态轨迹预测

4. [Recent Advances in Multi-Agent Human Trajectory Prediction: A Comprehensive Review](https://arxiv.org/abs/2506.14831)（2025）  
   总结多智能体交互编码、Transformer、生成模型及社会交互建模。

5. [Multimodal Trajectory Prediction: A Survey](https://arxiv.org/abs/2302.10463)（2023）  
   总结 goal-conditioned、CVAE、GAN、normalizing flow 和 occupancy-based 预测路线。

目前尚未发现一篇被广泛采用、专门以“航空物理轨迹的 EOS/终止条件”为主题的综述。相关工作分散在以下领域：

- ETA / time-to-event prediction；
- goal-conditioned trajectory prediction；
- multi-agent joint forecasting；
- survival / hazard modeling；
- variable-horizon planning；
- autoregressive sequence generation。

## 10. 最终建议

对当前 `ts_transformer`，建议采用以下优先级：

```text
第一优先：threshold-plane crossing 替代全局最近点
第二优先：增加概率 TTA head
第三优先：构建共享时间轴上的 multi-agent encoder
第四优先：研究 normalized path + monotonic time warp
```

核心原则是：

> 进度负责描述路径形状，时间负责多机交互与调度，几何事件负责定义真正的轨迹终点。
