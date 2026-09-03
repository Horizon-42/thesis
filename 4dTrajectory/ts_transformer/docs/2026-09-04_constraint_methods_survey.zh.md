# 让学习模型满足约束的方法：综述、落点与优先级（dev 文档，2026-09-04）

配套 `2026-09-04_procedure_constraints_design.zh.md`（约束是什么、数据支持哪些、损失方案细节）。本文回答"除了构造损失函数还有什么办法"，每种方法给出机制、保证强度、代价、在本仓库的落点和进一步阅读，最后给出优先实现的顺序。

## 0. 前提

- **约束对象**：五边段的 LPV 角走廊 `|xt| ≤ k·hw(d)` 和下滑窗口 `u ∈ [u_GP(d) − 60, u_GP(d) + 120]`，在入口锚定的 `enu` 图里表达（`d` 沿航道离入口的距离，`xt` 横向，`u_GP(d) = d·tan GPA`）。前一文档量过：观测数据在建立之后 87–99 % 满足，程序前段（IAF 过渡、FAF 上游汇入窗口）0 % 被飞过，所以不在约束范围内。
- **评价一种方法的五个维度**：保证强度（硬 / 软 / 统计）、是否需要重新训练、是否有需要标定的权重、是否保持可飞性（无折角）、实现成本与现有代码的接口。
- **已有实验给的先验**：目标几何作为输入通道对轨迹无效（H3）；损失在 10 km 尺度下看不见 300 m 横向偏差；`absolute` 与 `anchor-relative` 两种输出各带一个先验，分别对直线进近和雷达引导有利；种子噪声 5–22 m 以下的差异不算数。

## 1. 一览

| # | 方法 | 保证 | 重训 | 权重标定 | 可飞性 | 落点 |
|---|---|---|---|---|---|---|
| 2.1 | 损失罚项（soft penalty） | 软 | 是 | 是 | 不变 | `train.state_prediction_loss_components` |
| 2.2 | 输出参数化：名义路径 + 有界残差 | 硬 | 是 | 否 | 不变 | `prediction_outputs.StateOutputLayer` |
| 2.3 | 可微投影层 / 安全过滤器 | 硬 | 是 | 否 | 视投影而定 | 同上（区间约束时与 2.2 重合） |
| 2.4 | 约束式训练：原始对偶、增广拉格朗日、对数障碍 | 软，ε 可控 | 是 | 自动 | 不变 | `train.fit_model` 的 epoch 循环 |
| 2.5 | 采样 + 筛选、引导采样、受约束解码 | 视筛选而定 | 需多模态头 | 否 | 不变 | `forecast`，`run_ts_runway_hypotheses.py` |
| 2.6 | 两阶段：预测决策 + 优化器 | 硬且可飞 | 预测器不必 | 否 | 保证 | `collocation/optimizer.py` 新目标项 |
| 2.7 | 数据侧：名册、加权、反向约束学习 | 统计 | 是 | 否 | 不变 | `lateral_eligibility.py` |
| 2.8 | 推理期后处理投影 | 硬 | 否 | 否 | 有折角 | `forecast._POSTPROCESSORS` |

## 2. 各方法

### 2.1 损失罚项（soft penalty）

**机制。** 把违反量 `g(z) > 0` 的 hinge 或 hinge² 加进目标：`L = L_pred + λ·E[relu(g)²]`。PINN 一族把物理方程残差当罚项就是这个形式；航空里 Shi 等人的"constrained LSTM"把高度、速度的可行区间做成罚项。

**保证。** 软：训练分布上平均违反小，单条轨迹不保证。λ 决定精度与主任务的折中，且要在收敛点重新标定（本仓库的 imitation 权重就是这样定的：先量未加权项的量级，再按位置项倍数排 1/4/16/64 阶梯）。

**坑。** (a) 尺度：违反量要用约束自己的尺度（跑道尺度 100 m），不能沿用 10 km 的位置尺度，否则永远看不见。(b) 门控：只在真值已建立的行上开，见前一文档。(c) 过约束的表现和"平淡陷阱"一模一样，即满足率超过观测底线的同时精度下降，读数必须并列看。

**落点。** 新分量 `procedure` 加进 `STATE_LOSS_COMPONENT_NAMES`；几何从 `approach_constraints` 走 torch 分派复用。细节见前一文档 §3.2。

**阅读。**
- Raissi, Perdikaris, Karniadakis, *Physics-informed neural networks*, J. Comput. Phys. 2019：罚项式物理约束的原型。
- Karniadakis et al., *Physics-informed machine learning*, Nature Reviews Physics 2021：软约束与硬约束的全景，先读这篇。
- Shi, Xu, Pan, *4-D Flight Trajectory Prediction With Constrained LSTM Network*, IEEE T-ITS 2021：本领域的软约束实例，也是仓库综述里引的那条线。
- Kendall, Gal, Cipolla, *Multi-task learning using uncertainty to weigh losses*, CVPR 2018；Chen et al., *GradNorm*, ICML 2018：多项损失权重的自动化，若不用 2.4 的对偶法可以参考。

### 2.2 输出参数化：构造上满足（名义路径 + 有界残差）

**机制。** 让网络的值域就是可行集。区间约束用有界激活：`xt = k·hw(d)·tanh(z₁)`，`u = u_GP(d) + (−60 + 180·sigmoid(z₂))`；网络在跑道轴坐标 `(d, z₁, z₂)` 里输出，输出层旋转回 `(e, n, u)`。推广是**名义路径 + 有界残差**：名义路径取中线 + 下滑道（或按航线簇取的观测中位路径），网络只出偏离量，偏离量限幅。样条基是另一种表达：用 Bézier/B-spline 的控制点表示路径，凸包性质保证控制点在走廊内则整条曲线在走廊内，约束从"逐点"变成"逐控制点"的线性约束。

**保证。** 硬：零违反、可微、没有权重。它同时显式写进了"终点在原点、沿下滑道"（名义路径）和"从飞机所在处出发"（残差在锚点为零）两个先验，这正是 state-v2 没能兼顾的。

**坑。** (a) **门控是核心设计问题**：约束从哪里开始生效。训练时可用真值的建立距离，推理时没有；可选固定 `d_gate`（如 12 km，会把 15–62 % 在 FAF 内建立的航班提前压进走廊）、网络预测一个进度权重做软过渡、或由 2.5 的多假设决定。(b) tanh 在边界附近梯度消失，网络可能"贴墙"；用较宽的软边界或 `k=1` 的走廊配 `k=0.5` 的罚项组合。(c) 速度通道与位置通道要保持是同一条曲线的导数，否则破坏通道契约（`channels.py` 的速度是位置的精确图导数）。

**落点。** `TSConfig.state_position_reference` 加第三个值（如 `final-corridor`），`StateOutputLayer` 里实现旋转 + 限幅 + 门控；`run_naming.py` 会自动进 meta。

**阅读。**
- Beucler et al., *Enforcing Analytic Constraints in Neural Networks Emulating Physical Systems*, Phys. Rev. Lett. 2021：硬约束输出层与罚项的直接对比，结论是构造上满足更稳。
- Márquez-Neila, Salzmann, Fua, *Imposing Hard Constraints on Deep Networks: Promises and Limitations*, arXiv 2017：硬约束进网络的早期系统讨论和失败模式。
- Werling et al., *Optimal trajectory generation for dynamic street scenarios in a Frenét frame*, ICRA 2010：沿路径 / 横向的坐标分解，本文的 `(d, xt)` 就是这个框架。
- Gao et al., *Online Safe Trajectory Generation for Quadrotors Using Fast Marching Method and Bernstein Basis Polynomial*, ICRA 2018；Zhou et al., *EGO-Planner*, IEEE RA-L 2021：用凸包性质把走廊约束变成控制点约束。
- Song et al., *Learning to Predict Vehicle Trajectories with Model-based Planning* (PRIME), CoRL 2021：模型生成可行候选、网络评分，"可行性由构造保证"的另一种分工。

### 2.3 可微投影层与安全过滤器

**机制。** 在输出后接一个投影算子 `Π_C`，训练时梯度穿过投影。区间约束的投影就是 clamp（与 2.2 的限幅只差激活函数的选择）；凸约束用可微 QP 层（OptNet、cvxpylayers）；带等式的用 DC3 的"补全 + 梯度修正"。控制论里的对应物是安全过滤器：每步解一个小 QP，最小改动命令使其满足控制屏障函数（CBF）条件；RL 里叫 shielding。

**保证。** 硬。与 2.8 的区别只在梯度是否回传：回传后主干学会输出"投影之后好"的东西，而不是被事后改掉。

**坑。** 对本文这种逐行区间约束，QP 层没有必要，clamp 的零梯度区反而不如 tanh；QP 层的价值在约束**耦合**多行时（转弯率、加速度界、动力学可行），那时它就是 2.6 的可微版本。cvxpylayers 每批解一次 QP，300 行 × batch 的规模要先测吞吐。

**落点。** 区间约束与 2.2 同一处；耦合约束走控制输出路径（rollout 已是可微动力学）。

**阅读。**
- Amos, Kolter, *OptNet: Differentiable Optimization as a Layer in Neural Networks*, ICML 2017。
- Agrawal et al., *Differentiable Convex Optimization Layers*, NeurIPS 2019（cvxpylayers）。
- Donti, Rolnick, Kolter, *DC3: A learning method for optimization with hard constraints*, ICLR 2021。
- Ames et al., *Control Barrier Functions: Theory and Applications*, ECC 2019；Wabersich, Zeilinger, *A predictive safety filter for learning-based control of constrained nonlinear dynamical systems*, Automatica 2021；Alshiekh et al., *Safe Reinforcement Learning via Shielding*, AAAI 2018。

### 2.4 约束式训练：原始对偶、增广拉格朗日、对数障碍

**机制。** 把训练写成约束经验风险最小化 `min L_pred  s.t.  E[relu(g)] ≤ ε`，用对偶上升更新乘子：`λ ← max(0, λ + η·(E[relu(g)] − ε))`，每个 epoch 一次。增广拉格朗日再加二次项以改善条件数；对数障碍法把不等式换成 `−μ·log(−g)`，从内部逼近。

**保证。** 软，但违反水平由 ε 直接指定，λ 由算法找到，不再靠阶梯扫描；Chamon 等人证明了非凸情形下的对偶间隙界。

**坑。** ε 要用观测底线来定（比如"违反率不高于观测的 1–3 %"），否则又回到调参；λ 的更新步长与主优化器的学习率要分开；早停判据要看主任务的验证指标，不能看带 λ 的总损失。

**落点。** `fit_model` 的 epoch 循环里维护 λ 并写进 `loss_components` 的日志；配 2.1 的 `procedure` 分量。改动很小。

**阅读。**
- Chamon, Ribeiro, *Probably Approximately Correct Constrained Learning*, NeurIPS 2020；Chamon, Paternain, Calvo-Fullana, Ribeiro, *Constrained Learning with Non-Convex Losses*, IEEE Trans. Inf. Theory 2023：理论基础，先读 2020 那篇。
- Fioretto et al., *Lagrangian Duality for Constrained Deep Learning*, ECML-PKDD 2020：工程化的对偶上升配方。
- Kervadec et al., *Constrained deep networks: Lagrangian optimization via log-barrier extensions*, ICPR 2022：对数障碍替代罚项。
- Nocedal, Wright, *Numerical Optimization*, 2nd ed., ch. 17：罚函数与增广拉格朗日的经典推导。

### 2.5 采样 + 筛选、引导采样、受约束解码

**机制。** 生成 K 条候选，按约束满足度拒绝或排序（MultiPath / TNT 一类多模态头就是"K 个模式 + 打分"）；扩散模型可以在采样时把约束梯度当引导项（classifier guidance 的形式），把违反量当能量往下推（Diffuser、MotionDiffuser）；自回归或分段链式预测可以每段末尾投影一次，即受约束解码。本仓库的按跑道假设展开就是 K 条候选 + 选择器，只是候选来自跑道假设而不是随机采样。

**保证。** 取决于筛选：拒绝式是硬的（只要有一条满足），引导式是软的。

**坑。** 现在的模型是确定性的，只有一条候选；要先有多模态或随机头（混合密度、CVAE、扩散），那是综述里点名的开放问题。K 条候选的"最好一条"读数有运气成分（镜像假兄弟实验测过：真兄弟 oracle 79 m，假兄弟 32 m），必须配噪声对照。

**落点。** `forecast` 加采样接口；`run_ts_runway_hypotheses.py` 的选择器框架可复用；`window` 模式的逐段链式预测天然适合每段投影。

**阅读。**
- Chai et al., *MultiPath*, CoRL 2019；Zhao et al., *TNT: Target-driveN Trajectories*, CoRL 2020：多模态候选 + 打分的两种标准形式。
- Salzmann, Ivanovic, Chakravarty, Pavone, *Trajectron++*, ECCV 2020：多模态 + 动力学积分保证可行，与本仓库的控制输出路径同构。
- Janner et al., *Planning with Diffusion for Flexible Behavior Synthesis*, ICML 2022；Jiang et al., *MotionDiffuser*, CVPR 2023：约束作为采样期引导；Dhariwal, Nichol, *Diffusion Models Beat GANs on Image Synthesis*, NeurIPS 2021 是引导的出处。
- Pang, Xu, Liu, *Data-driven trajectory prediction with weather uncertainties: A Bayesian deep learning approach*, Transportation Research Part C 2021：航空轨迹的随机预测实例。

### 2.6 两阶段：预测决策 + 优化器

**机制。** 预测器不出路径，出**决策量或初值**：总时长、汇入距离、跑道、以及作为初值和跟踪参考的粗路径；优化器在硬约束下生成最终轨迹（目标 = 跟踪参考 + 控制代价，程序约束照常作为不等式行）。变体：预测作 warm start（amortized optimization）、predict-then-optimize（按下游决策质量训练预测器）、可微 MPC（穿过优化器求梯度）。

**保证。** 硬且可飞，由已经验证过的 casadi 行保证；同时是部署时"预测器 + 调度层"的自然分工。

**坑。** 当前 `collocation/optimizer.py` 只有 min-time 和 fixed-time 控制代价两种目标，没有跟踪参考的项，要新加（参考点取求解器自己的节点时间，见 `dense_node_times` 的教训）；每架航班一次 IPOPT（约 4 s，失败 56 s），2,000 架验证集是小时级；casadi 只能在隔离子进程里跑；可微 MPC 在这个规模和线程限制下不划算。

**落点。** 优化器新目标项 + `run_scenario_optimization` 风格的批处理；预测器提供 `final_time_s`（作 fixed duration）和路径（作 x0）。

**阅读。**
- Amos, *Tutorial on amortized optimization*, Foundations and Trends in ML 2023：把"学习给优化器初值 / 解"整个谱系讲清楚，先读这篇。
- Sambharya, Hall, Amos, Stellato, *End-to-End Learning to Warm-Start for Real-Time Quadratic Optimization*, L4DC 2023。
- Elmachtoub, Grigas, *Smart "Predict, then Optimize"*, Management Science 2022。
- Amos et al., *Differentiable MPC for End-to-end Planning and Control*, NeurIPS 2018。
- Andersson et al., *CasADi: a software framework for nonlinear optimization and optimal control*, Math. Prog. Comp. 2019。

### 2.7 数据侧：名册、加权、反向约束学习

**机制。** 只用满足约束的样本训练（本仓库的横向通过名册 `lateral_eligibility.py` 就是），或按满足度加权；反过来也可以从数据里学约束是什么（inverse constraint learning），本文用测量代替了它。

**保证。** 统计意义上的：模型学到"数据里没有违反"，不保证输出没有。

**坑。** 这里数据本身 85–99 % 满足，模型的违反是模型的（NW 平移、终点尾部），不是数据的；再筛数据帮助不大，还会引入选择偏差。

**阅读。**
- Scobee, Sastry, *Maximum Likelihood Constraint Inference for Inverse Reinforcement Learning*, ICLR 2020。

### 2.8 推理期后处理投影

**机制。** 预测完成后，从路径最后一次进入走廊的行起，把 `xt`、`u` 夹进边界，终点钉到目标。

**保证。** 硬，但有折角，flyability 会变差；不改变模型学到的东西。

**价值。** 它是"约束能收回多少终点误差"的天花板，也是部署兜底；零成本，所以任何一轮实验都应带这一臂。

**落点。** `forecast._POSTPROCESSORS` 在 `truncate_at_threshold` 之后。

## 3. 优先实现的建议

先做共同前置，再按顺序推进；每一步都有可写进论文的读数。

| 顺序 | 内容 | 依赖 | 预期读数 | 预注册否决 |
|---|---|---|---|---|
| P0 | 共享几何：`_lpv_spec` 桥下移到 `approach_constraints`，`mathx` 加 torch 分派，ts 侧每航班一行程序常量进 batch 的 `dynamics` 位 | 无 | 单元测试：三处几何逐比特一致 | |
| P0 | 2.8 后处理投影臂（不训练） | P0 几何 | 终点误差可收回上限、门通过率上限、flyability 代价 | |
| P1 | 2.2 名义路径 + 有界残差（`state_position_reference` 第三值），门控先用固定 `d_gate` 和"真值建立距离"两档 | P0 | 直线进近 FDE、第一步跳变、走廊满足率对观测底线、雷达引导层是否保持 | 雷达引导层两个种子退化 |
| P1 | 2.1 + 2.4 罚项 + 原始对偶（ε 取观测底线），作为 P1 的配对对照臂 | P0 | 同上，另加 λ 轨迹 | 同上 |
| P2 | 2.6 优化器跟踪目标 + 批处理，预测给时长和初值 | P0；优化器新目标项 | 硬约束 + 可飞的最终轨迹；求解率、耗时 | |
| P3 | 2.5 多模态头 + 筛选 / 引导 | 多模态输出（未做） | 跑道左右选择、约束满足的候选率 | 必须配镜像噪声对照 |
| 不做 | 目标 / 程序几何作输入通道；IAF 段与汇入窗口作约束；±22 m 硬窗口；再筛数据 | | | |

**为什么 P1 是参数化而不是罚项。** 参数化没有权重、保证硬、并且直接回答 state-v2 留下的问题（两个先验能否同时保留）；罚项作为同一轮的对照臂跑，用对偶法免去阶梯扫描。两臂共用 P0 的几何模块，所以增量成本很低。

**为什么 P2 放在 P1 后。** 它是部署路径，不是精度实验；它需要优化器侧的新目标项，而且 P1 的结果会决定预测器给优化器的是路径还是只有决策量。

**规模。** P0 + P1 是两机场 × 两种子 × 3 臂 = 12 次训练，与机场帧消融同量级（那次 14 次约一天）。

## 4. 阅读顺序

1. Karniadakis et al. 2021（全景）→ Beucler et al. 2021（硬 vs 软的直接对比）。
2. Chamon & Ribeiro 2020 → Fioretto et al. 2020（对偶法怎么落地）。
3. Amos & Kolter 2017 → Donti et al. 2021（投影层与耦合约束）。
4. Amos 2023 tutorial（两阶段与 warm start 的谱系）。
5. Janner et al. 2022 → Jiang et al. 2023（等模型有了多模态输出再读）。
6. Shi et al. 2021、Pang et al. 2021（本领域的两个实例，写相关工作时用）。
