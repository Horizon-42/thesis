# 跑道阈值事件：第一性原理开发档案

状态：**完整设计档案；已由单独的简化方案实现。本文保留推导与约束，不是当前 runtime 合同。**

日期：2026-08-15

范围：观测航迹的跑道关联、跑道阈值事件估计、统一评价入口，以及由此产生的可再生成派生数据

不在范围内：重新下载 ADS-B、修改原始样本、改变 LPV 横向标准、筛选训练集、查看或使用 outer-test

## 1. 文档目的与当前结论

本文是下一轮重构的完整开发档案。即使脱离此前对话，也应能仅凭本文恢复问题、数学目标、数据契约、实验方法、实施顺序和不能破坏的边界。

当前实现与验证记录见
[`threshold-event-simplified-implementation.zh.md`](threshold-event-simplified-implementation.zh.md)。

本文冻结时的实现是 `observed-threshold-event-v7`：它找到一个跨越跑道阈值平面的 bracket，却只用 bracket 选择跑道和物理进近段；最终发布的横向、垂直阈值坐标仍由阈值前直线拟合外推得到。这造成一个根本冲突：**已经在观测支持域内的阈值交点，仍被当成域外点外推**。

目标设计必须把三个问题分开：

1. **跑道与 inbound pass 关联**：这条航迹对应哪条跑道、哪一次进近？
2. **阈值事件估计**：该 pass 的轨迹与 LTP 平面在哪里相交？这个交点是域内插值还是域外推断？
3. **标准评价**：交点相对跑道中心线和发布 TCH 路径的误差是否满足 LPV 或明确选择的 fallback 标准？

核心决策是：

- 有 source-valid bracket：对同一对三维样本做阈值平面内插，得到一个完整的三维事件；
- 航迹在阈值前终止：把它标为 right-censored，再使用经过人工截断实验验证的阈值前预测器；
- bracket 存在但源完整性失败：不能伪装成 right-censored，也不能静默回退；事件不可用并保留原因；
- evaluation 只消费事件，不选择样本、不拟合、不外推；
- TCH、`±7.5 m`、跑道宽度和最终 pass rate 不得参与估计器训练或选择。

## 2. 第一性原理目标

### 2.1 被估计的物理量

设某次已选定的 final inbound pass 在连续时间上的导航参考点轨迹为

$$
\mathbf{x}(t)=
\begin{bmatrix}
\varphi(t) & \lambda(t) & H^{\mathrm{HAE}}(t)
\end{bmatrix}^{\mathsf T},
\qquad t\in I_p,
$$

其中 $I_p=[t_p^-,t_p^+]$ 是该物理 pass 的时间区间。评价对象不是整个 LPV cone，也不是最后若干离散点，而是这条连续轨迹与所选跑道 LTP 平面的一次 inbound 交点。

在跑道坐标系中写成

$$
\mathbf{r}(t)=
\begin{bmatrix}
a(t) & c(t) & u(t)
\end{bmatrix}^{\mathsf T},
$$

其中：

- $a$：沿跑道着陆方向的 along-track 坐标，阈值前为负；
- $c$：cross-track 坐标，着陆方向右侧为正；
- $u$：相对 LTP 高程的高度，与 LTP 使用相同垂直基准。

目标事件时间严格定义为

$$
t^*=\inf\left\{t\in I_p:\ a(t)=0\ \land\ \dot a(t)>0\right\},
$$

目标事件为

$$
E^*=\left(t^*,\,a(t^*),\,c(t^*),\,u(t^*)\right),
\qquad a(t^*)=0.
$$

因此，fitting 的根本目的只是在 $E^*$ 不处于观测支持域时预测它；它不是为了“制造一条更像标准进近的航迹”。

### 2.2 不可破坏的不变量

以下约束优先于实现便利：

1. 原始/重建航迹只存源观测与源完整性事实，不存 LPV/LNAV-VNAV 政策或 verdict。
2. 一个 event 的 $c$ 与 $u$ 必须来自同一个事件时间、同一个 pass、同一种估计路径；禁止“bracket 横向 + 拟合垂直”等混合事件。
3. 观测支持域内只能称为 interpolation；支持域外必须称为 inferred/extrapolated。
4. 跑道关联不得使用 evaluation pass/fail 条件，否则会把评价率预先筛成 100%。
5. evaluation 不得对 observed event 重新拟合。
6. HAE、MSL 和 geoid offset 必须显式；不得依赖数值看起来合理来猜 datum。
7. 每个派生 event 必须绑定精确的物理 LTP frame fingerprint；evaluation context 另有包含 TCH/FSD/width 的 policy fingerprint。任一所需 cycle 不匹配时必须重生成对应派生物。
8. `flight_key` 仍是稳定航班身份；event 不创建第二套身份。
9. 非有限数不得参与比较或写入 JSON。
10. 估计器只能由源数据误差指标选择；不能用 TCH gate、pass rate 或 outer-test 调参。

## 3. 跑道坐标系的严格定义

### 3.1 坐标变换

设 LTP 经纬度为 $(\varphi_0,\lambda_0)$，HAE 高程为 $H_0^{\mathrm{HAE}}$，跑道着陆方向的真航向/罗盘方位为 $\psi$，其中北为 $0$、顺时针为正。令

$$
N=(\varphi-\varphi_0)m_{\mathrm{lat}},
\qquad
E=(\lambda-\lambda_0)m_{\mathrm{lon}}(\varphi_0),
$$

则当前项目的局部平面投影严格为

$$
\begin{aligned}
a &= E\sin\psi+N\cos\psi,\\
c &= E\cos\psi-N\sin\psi,\\
u &= H^{\mathrm{HAE}}-H_0^{\mathrm{HAE}}.
\end{aligned}
$$

这里 $m_{\mathrm{lat}}$ 与 $m_{\mathrm{lon}}(\varphi_0)$ 必须复用 `geokit`；不得另写近似常数。该定义与 [`final_approach/frame.py`](../final_approach/frame.py) 一致。

其逆变换为

$$
\begin{aligned}
E &= a\sin\psi+c\cos\psi,\\
N &= a\cos\psi-c\sin\psi,\\
\varphi &= \varphi_0+N/m_{\mathrm{lat}},\\
\lambda &= \lambda_0+E/m_{\mathrm{lon}}(\varphi_0),\\
H^{\mathrm{HAE}} &= H_0^{\mathrm{HAE}}+u.
\end{aligned}
$$

### 3.2 HAE 到 MSL

令同一 LTP 的 geoid offset 为

$$
G_0=H_0^{\mathrm{HAE}}-H_0^{\mathrm{MSL}}.
$$

在当前机场局部范围内，event 的 MSL 高度按同一已审计 offset 转换：

$$
\widehat H_*^{\mathrm{MSL}}=widehat H_*^{\mathrm{HAE}}-G_0.
$$

等价地，由于 $\widehat u_*=\widehat H_*^{\mathrm{HAE}}-H_0^{\mathrm{HAE}}$，有

$$
\widehat H_*^{\mathrm{MSL}}=H_0^{\mathrm{MSL}}+\widehat u_*.
$$

物理 event snapshot 定义为

$$
S_{\mathrm{frame}}=
\{\text{airport, runway, LTP lat/lon, }H_0^{\mathrm{HAE}},
H_0^{\mathrm{MSL}},G_0,\psi,\text{ physical source ids/cycles}\}.
$$

其 fingerprint 为 canonical JSON 的 SHA-256：

$$
f_{\mathrm{frame}}=
\operatorname{SHA256}(\operatorname{canonicalJSON}(S_{\mathrm{frame}})).
$$

实现必须验证 event 与 evaluation context 使用同一个 $f_{\mathrm{frame}}$，不能拿旧 cycle 的 $\widehat u_*$ 与新 cycle 的 LTP frame 组合。TCH、FSD、runway width 和 approach method 不属于 $S_{\mathrm{frame}}$；它们进入 evaluation-owned context：

$$
S_{\mathrm{eval}}=
\{f_{\mathrm{frame}},H_{\mathrm{TCH}},F_L,W,
\text{approach id, standard id, policy source ids/cycles}\},
$$

并独立生成

$$
f_{\mathrm{eval}}=
\operatorname{SHA256}(\operatorname{canonicalJSON}(S_{\mathrm{eval}})).
$$

这样既能审计 cycle，又不会把 approach profile 塞进 trajectory/event schema。

## 4. 源观测与时间语义

### 4.1 观测模型

重建后第 $i$ 个位置样本写为

$$
Y_i=(\tau_i,\varphi_i,\lambda_i,H_i^{\mathrm{HAE}},v_i),
$$

其中：

- $\tau_i=\texttt{lastposupdate}_i$ 是位置物理时间；
- $v_i$ 是 ADS-B 报告的 ground speed；
- OpenSky 没有给 `geoaltitude` 单独的测量时间，因此 $H_i^{\mathrm{HAE}}$ 是与该位置更新时间最接近的可用 state snapshot 中的高度，而不是“已证明与位置同步”的高度。

可以把采样误差写成

$$
\mathbf y_i=\mathbf r(\tau_i)+\boldsymbol\varepsilon_i,
$$

但 $\boldsymbol\varepsilon_i$ 的高度分量同时包含量化、接收/状态聚合和未知的高度更新时间偏差。不能仅凭重建完成就令 $\varepsilon_{u,i}=0$。

### 4.2 当前源清理契约

对 state-row 时间 $s_i$、`lastcontact` 时间 $\ell_i$、位置时间 $\tau_i$，可用行必须满足

$$
-1\ \mathrm{s}\le s_i-\tau_i\le15\ \mathrm{s},
\qquad
-1\ \mathrm{s}\le s_i-\ell_i\le15\ \mathrm{s}.
$$

对相同 $(\texttt{icao24},\tau)$ 的组

$$
\mathcal G_\tau=\{Y_i:\tau_i=\tau\},
$$

先要求组内经纬度一致，再选择

$$
i^*=\arg\min_{i\in\mathcal G_\tau}
\left(|s_i-\tau|,\,s_i\right)
$$

对应的 state snapshot。若组内高度极差大于 $0.05\ \mathrm m$，记一次 `geoaltitude_async_groups`，但仍保留上述最接近位置时间的唯一行。

规范序列还必须满足

$$
0<\tau_{i+1}-\tau_i\le15\ \mathrm{s}.
$$

不满足时切分 coverage block，仅保留机场裁剪后的最终连续 block。该过程见 [`trajectory_data_process/harvest/tracks.py`](../trajectory_data_process/harvest/tracks.py)。

### 4.3 对高度可信度的正确表述

现有高度不是“整条航迹不可用”，也不是“已经完全同步”。正确结论是：

- 每个重建样本都使用当前数据能提供的最佳位置时间；
- 高度没有独立 source timestamp，因此单个阈值邻域样本可能仍有额外时间对齐误差；
- 这类风险应通过 bracket 邻域审计、人工降采样/留点实验和方法间稳定性量化；
- 不能因为有误差就放弃垂直评价，也不能因为完成重建就把误差假设为零。

## 5. 跑道与 physical pass 关联

### 5.1 几何 bracket

对某跑道 $r$，将按时间排序的同一条完整轨迹投影为

$$
R_i^{(r)}=(a_i^{(r)},c_i^{(r)},u_i^{(r)}).
$$

相邻样本 $(i,i+1)$ 是 inbound 几何 bracket，当且仅当

$$
a_i^{(r)}<0\le a_{i+1}^{(r)},
\qquad
a_{i+1}^{(r)}-a_i^{(r)}>0.
$$

定义

$$
\alpha_i^{(r)}=
\frac{-a_i^{(r)}}{a_{i+1}^{(r)}-a_i^{(r)}}.
$$

由 bracket 定义立即得到 $0<\alpha_i^{(r)}\le1$。

### 5.2 源完整性谓词

对 bracket 定义

$$
\Delta\tau_i=\tau_{i+1}-\tau_i,
$$

球面水平距离为 $d_i$，位置导出速度和报告速度均值为

$$
v_i^{\mathrm{pos}}=\frac{d_i}{\Delta\tau_i},
\qquad
\bar v_i^{\mathrm{rep}}=\frac{v_i+v_{i+1}}{2}.
$$

当前实现使用的 source-integrity 条件为

$$
\begin{aligned}
0 &<\Delta\tau_i\le30\ \mathrm{s},\\
0 &<v_i,v_{i+1}\le200\ \mathrm{m/s},\\
|v_i^{\mathrm{pos}}-\bar v_i^{\mathrm{rep}}| &\le25\ \mathrm{m/s},\\
0.5&\le\frac{v_i^{\mathrm{pos}}}{\bar v_i^{\mathrm{rep}}}\le1.5.
\end{aligned}
$$

这些是源/关联检查，不是 LPV 性能门限。后续简化审查可以删除与上游 $15\ \mathrm s$ 连续块契约重复的 $30\ \mathrm s$ 分支，但不能删除“真实位移必须与报告速度相容”这一物理检查。

### 5.3 结构约束与跑道选择

先计算 bracket 的线性阈值交点

$$
\widetilde c_i=(1-\alpha_i)c_i+\alpha_i c_{i+1},
\qquad
\widetilde u_i=(1-\alpha_i)u_i+\alpha_i u_{i+1}.
$$

只允许满足

$$
|\widetilde c_i|\le C_r,
\qquad
|\widetilde u_i|\le U_{\mathrm{struct}}
$$

的结构候选。$U_{\mathrm{struct}}$ 当前为 $100\ \mathrm m$；它只用于排除不可能在此跑道着陆的高空穿越，不是垂直 verdict gate。

对平行跑道，若航向差不超过 $5^\circ$，令两中心线横向间距为 $D_{rj}$，则

$$
C_r=\min\left(C_{\mathrm{broad}},\ \min_j\frac{D_{rj}}2\right).
$$

每条跑道内部选择 $|\widetilde c_i|$ 最小的有效 bracket；若相同，选择时间上更晚者。跑道分数为

$$
S_r=|\widetilde c_r|,
\qquad
r^*=\arg\min_r S_r.
$$

令第一、第二名分数为 $S_{(1)},S_{(2)}$。当前相对歧义规则是

$$
S_{(2)}-S_{(1)}<50\ \mathrm m
\quad\Longrightarrow\quad
\texttt{ambiguous}.
$$

这个分数只回答“更像哪条跑道”，不回答“是否满足 LPV”。

### 5.4 selected inbound pass

一旦选定 bracket $(i,i+1)$，其前方连续 inbound run 定义为包含 $i$ 的最大连续索引区间

$$
P=[k,i+1]
$$

使得区间内时间严格递增、没有 coverage gap，并且沿跑道方向不存在超过允许噪声的真实反向跳跃。任何用于该事件的局部估计或 censored 实验都只能取自 $P$；不得把早一次进近、复飞或晚一次落地的样本混入。

## 6. 可观测性分类

阈值 event 的证据状态必须先于估计方法确定：

### 6.1 `within_observed_support`

存在通过源完整性和结构检查的 selected bracket：

$$
\exists(i,i+1):\quad a_i<0\le a_{i+1}.
$$

此时 $a=0$ 位于两个观测点的凸包内，event 是插值问题。

### 6.2 `right_censored`

已关联一个 final inbound pass，但其最后一个可信样本仍在阈值前：

$$
\max_{i\in P}a_i<0.
$$

此时 $E^*$ 不在观测支持域，任何阈值值都是模型推断。必须报告外推距离

$$
d_{\mathrm{ext}}=-\max_{i\in P}a_i>0.
$$

### 6.3 `invalid_support`

已关联的 winning runway 上几何存在 $a_i<0\le a_{i+1}$，但时间、速度、位移、结构或数值完整性失败。它不是 right-censored，禁止通过阈值前外推覆盖掉失败原因；其他候选跑道的失败不得压制 winning runway 的可信 censored event。

### 6.4 `unavailable`

无法唯一关联跑道/pass，或没有足够信息定义域内交点或可信的域外预测。evaluation 应返回 `indeterminate`，并保留结构化原因。

## 7. 域内三维阈值事件估计

### 7.1 基准估计器：piecewise-linear bracket

对 selected bracket，唯一共享的插值系数为

$$
\alpha=\frac{-a_i}{a_{i+1}-a_i}.
$$

完整事件估计为

$$
\begin{aligned}
\widehat t_* &= (1-\alpha)\tau_i+\alpha\tau_{i+1},\\
\widehat a_* &= 0,\\
\widehat c_* &= (1-\alpha)c_i+\alpha c_{i+1},\\
\widehat u_* &= (1-\alpha)u_i+\alpha u_{i+1}.
\end{aligned}
$$

经纬度与 HAE 高度必须由 $\operatorname{unproject}(0,\widehat c_*,\widehat u_*)$ 一次性得到。这保证 lateral、vertical、位置与事件时间属于同一物理交点。

该估计器是目标默认值，因为它只作 bracket 内的局部线性假设，不引入 TCH、glidepath、远端窗口或外推模型。它并不宣称零误差；其证据类型是 `interpolated`，而不是 `measured_exact`。

### 7.2 只能通过实验晋级的候选方法

如果两点高度量化或异步导致明显不稳定，可比较局部鲁棒模型。设 selected pass 中、阈值附近的支持集合为

$$
\mathcal L=\{j\in P:-D_-\le a_j\le D_+\}.
$$

对 $y_j\in\{c_j,u_j\}$，$q$ 阶局部模型为

$$
f_q(a;\boldsymbol\beta)=\sum_{k=0}^{q}\beta_k a^k,
\qquad q\in\{1,2\},
$$

并通过

$$
\widehat{\boldsymbol\beta}
=\arg\min_{\boldsymbol\beta}
\sum_{j\in\mathcal L}
w(a_j)\,
\rho\!\left(
\frac{y_j-f_q(a_j;\boldsymbol\beta)}{s}
\right)
$$

求解，其中 $w$ 是预先固定的局部权重、$\rho$ 是预先固定的鲁棒损失、$s$ 是只由源残差估计的尺度。阈值估计为

$$
\widehat y_* = f_q(0;\widehat{\boldsymbol\beta})=\widehat\beta_0.
$$

局部模型不能因为让更多轨迹落入 `±7.5 m` 而被采用；只有第 11 节的 source-only 验证证明它稳定优于两点插值时才可替换默认方法。

### 7.3 不允许的 hybrid

以下定义被明确禁止：

$$
\widehat c_*=\widehat c_*^{\mathrm{bracket}},
\qquad
\widehat u_*=\widehat u_*^{\mathrm{fit}}.
$$

它把两个不同的隐含事件模型拼成一个不存在的三维状态，无法定义统一的 $\widehat t_*$，也无法给出正确的协方差。

## 8. 域外 censored event 预测

### 8.1 问题定义

对 right-censored pass，观测支持域为

$$
\mathcal A_P=[a_{\min},a_{\max}],
\qquad a_{\max}<0.
$$

目标 $a=0\notin\mathcal A_P$。因此任何估计器

$$
\widehat E_*=g(\{Y_i:i\in P\})
$$

都是预测器，不得在字段或 UI 中称为 direct/measured crossing。

### 8.2 基准预测器：单一鲁棒阈值前直线

在固定的、预先声明的阈值前窗口

$$
\mathcal F=\{i\in P:A_-\le a_i\le A_+<0\}
$$

中，分别对 $c$ 和 $u$ 拟合

$$
y_i=\beta_0+\beta_1a_i+\epsilon_i.
$$

先用鲁棒 seed 识别 gross outlier，再对保留样本求

$$
(\widehat\beta_0,\widehat\beta_1)
=\arg\min_{\beta_0,\beta_1}
\sum_{i\in\mathcal F_{\mathrm{keep}}}
(y_i-\beta_0-\beta_1a_i)^2.
$$

阈值预测为

$$
\widehat c_*=\widehat\beta_{0,c},
\qquad
\widehat u_*=\widehat\beta_{0,u}.
$$

该模型的物理依据仅限于稳定 final approach 在阈值前近似直线。它不能覆盖 flare 后数据，也不能跨越 selected pass 下界。最终窗口不能由 pass rate 选择。

### 8.3 预测器必须通过人工截断验证

令 $\mathcal D_B$ 是 source-valid bracket 航迹集合。对每条 $j\in\mathcal D_B$，先用域内方法得到参考事件

$$
E_j^{\mathrm{ref}}.
$$

从真实 censored 航迹的末端距离经验分布 $F_D$ 中取 $d>0$，构造人工截断航迹

$$
P_j^{(d)}=\{i\in P_j:a_i\le-d\}.
$$

候选预测器 $g_m$ 的误差为

$$
\mathbf e_{j,m}(d)
=g_m(P_j^{(d)})-E_j^{\mathrm{ref}}
=\begin{bmatrix}e_{c,j,m}(d)&e_{u,j,m}(d)\end{bmatrix}^{\mathsf T}.
$$

训练/校准和比较只使用 development airports，并采用 leave-one-airport-out：每一折用 $K-1$ 个机场确定参数，在剩余机场计算误差。不能查看 outer-test。

### 8.4 无法验证时的结果

若某个 $d_{\mathrm{ext}}$ 超出开发验证支持域，或预测器不满足最低样本/跨度条件，则

$$
\texttt{event.status}=\texttt{unavailable},
$$

而不是无限外推。是否允许某个最大外推距离，必须由第 11 节误差曲线预注册确定，不能由 `7.5 m` gate 倒推。

## 9. 不确定性与 evidence 的分离

### 9.1 事件协方差

令输入观测向量为 $\mathbf y$，事件估计器为 $g$，其一阶 Jacobian 为

$$
J_g=\left.\frac{\partial g}{\partial\mathbf y}\right|_{\mathbf y}.
$$

事件协方差的一般形式为

$$
\Sigma_E=J_g\Sigma_YJ_g^{\mathsf T}+\Sigma_{\mathrm{model}}.
$$

对把每个高度量化误差近似成独立均匀分布

$$
q_i\sim\mathcal U(-Q/2,Q/2),
\qquad Q=25\ \mathrm{ft}=7.62\ \mathrm m,
$$

的两点插值，仅量化项为

$$
\sigma_{u,q}^2=
\left((1-\alpha)^2+\alpha^2\right)\frac{Q^2}{12}.
$$

但实际相邻 ADS-B 高度误差可能相关，且高度没有独立 source timestamp，因此该式只能作为分析下界，不能直接冒充完整不确定性。最终 $\Sigma_E$ 或区间必须由 source-only 留点、降采样、人工截断和跨机场覆盖率校准。

### 9.2 verdict 与 evidence

事件必须分别保存：

$$
\text{point estimate},\quad
\text{observability},\quad
\text{method},\quad
\text{uncertainty status/value},\quad
\text{source diagnostics}.
$$

评价使用 point estimate。若有标准差 $\sigma$，可报告诊断区间

$$
[\widehat e-1.96\sigma,\ \widehat e+1.96\sigma],
$$

但它不改变标准 gate，也不把 point pass/fail 改成 `indeterminate`。`indeterminate` 只表示 event 或适用标准本身不可用。

## 10. 统一事件契约

### 10.1 领域对象

新实现只维护一个 `RunwayThresholdEvent` 领域对象。其最小语义为：

```text
schema_version
status                    estimated | unavailable | not_reached
runway
threshold_frame_snapshot
threshold_frame_fingerprint
observability             within_observed_support | right_censored | invalid_support | unavailable
method                    direct_linear_bracket | censored_robust_line | none
event_time_s              nullable only when genuinely not identifiable
threshold_crossing_lat
threshold_crossing_lon
threshold_crossing_altitude_m
altitude_datum
signed_cross_track_m
source_sample_range       exact inclusive original indices used by the estimator
interpolation_fraction    direct method only
extrapolation_distance_m  censored method only
uncertainty
source_integrity
diagnostics
unavailable_reason
```

严格字段约束为：

$$
\texttt{status}=\texttt{estimated}
\Longrightarrow
\widehat t_*,\widehat\varphi_*,\widehat\lambda_*,
\widehat H_*,\widehat c_*\in\mathbb R
$$

且全部有限。对于 direct event：

$$
\texttt{observability}=\texttt{within\_observed\_support},
\quad
0<\alpha\le1,
\quad
d_{\mathrm{ext}}=0.
$$

对于 censored event：

$$
\texttt{observability}=\texttt{right\_censored},
\quad
d_{\mathrm{ext}}>0,
\quad
\alpha=\varnothing.
$$

event 中不得存 approach type、TCH、FSD、runway width、LPV/LNAV-VNAV bound 或 verdict。`threshold_frame_snapshot` 只含第 3.2 节的物理坐标、datum、来源与 cycle；evaluation-owned facts 只写入 evaluation context/report。

### 10.2 单一版本策略

新 schema 替换 `observed-threshold-event-v7`。由于 event 是可再生成派生数据，不增加 v7/vNext 双读、legacy fallback 或两套可执行估计器。旧格式被严格拒绝，并提示运行 `--reclassify-existing`。

历史算法及其原因只保留在本文和 git 历史中，不保留在生产代码分支中。

### 10.3 三类轨迹的统一入口

- observed：读取 harvest 已生成的 event，evaluation 禁止 refit；
- optimized/predicted：若最终离散段 bracket 阈值，则用相同平面交点定义；若目标状态明确就是 LTP event，则直接映射为 event；
- computed trajectory 若实质性未到达阈值，返回 `not_reached`，不得外推制造 pass。

三者最终都交给同一个纯函数

$$
V=\operatorname{evaluate}(E,\,C),
$$

其中 $E$ 是 policy-free event，$C$ 是 evaluation-owned approach/runway context。

## 11. 估计器实验与选择规则

### 11.1 禁止的数据泄漏

候选方法选择时不得读取：

$$
H_{\mathrm{TCH}},\quad B_L,\quad B_V,\quad
\texttt{component verdict},\quad\texttt{overall verdict},\quad
\texttt{pass rate}.
$$

outer-test 仍是一次性最终发布集；本设计验证只能使用 development split 或明确标记的开发机场数据。

### 11.2 direct 方法验证

对有密集连续样本的局部 inbound 片段，选择内部样本 $k$，删除 $Y_k$，以平面 $a=a_k$ 为 pseudo-threshold。候选方法 $m$ 的留点误差为

$$
\mathbf e^{\mathrm{LOO}}_{k,m}
=\widehat{\mathbf r}_{k,m}-
\begin{bmatrix}c_k&u_k\end{bmatrix}^{\mathsf T}.
$$

另定义降采样稳定性。对固定保留规则 $S_b$，完整数据与子采样数据的 event 差为

$$
\Delta\mathbf e_{j,m}^{(b)}
=\widehat E_{j,m}(S_b)-\widehat E_{j,m}(S_{\mathrm{full}}).
$$

报告必须按以下维度分层：机场、bracket 时间间隔、是否出现 `geoaltitude_async_groups`、高度变化/保持模式、$\alpha$ 分位区间。

### 11.3 censored 方法验证

使用第 8.3 节的人工截断误差。对方法 $m$、机场 $k$、分量 $q\in\{c,u\}$ 定义

$$
\begin{aligned}
b_{m,k,q}&=\operatorname{median}(e_{m,k,q}),\\
M_{m,k,q}&=\operatorname{median}(|e_{m,k,q}|),\\
Q_{m,k,q}^{95}&=\operatorname{quantile}_{0.95}(|e_{m,k,q}|).
\end{aligned}
$$

若方法给出名义 $95\%$ 区间 $I_{j,m,q}$，覆盖率为

$$
C_{m,k,q}^{95}
=\frac1{n_k}\sum_{j=1}^{n_k}
\mathbf1\{E_{j,q}^{\mathrm{ref}}\in I_{j,m,q}\}.
$$

### 11.4 严格选择顺序

每个方法先必须满足：

1. 所有输入和输出有限；
2. 不跨 pass；
3. 不使用 policy 字段；
4. 在所有 leave-one-airport-out 折中可执行；
5. 若声称区间，则给出按机场和截断距离分层的覆盖率。

对满足条件的方法，定义最坏机场风险向量

$$
\mathcal R_m=
\left(
\max_{k,q}Q_{m,k,q}^{95},\
\max_{k,q}|b_{m,k,q}|,\
\max_{k,q}M_{m,k,q},\
\operatorname{Complexity}(m)
\right).
$$

在同一 evidence coverage 下按上述向量字典序最小化；若复杂方法不能改善排在它之前的 source-error 项，就选择更简单的方法。这样无需用 `±7.5 m` 或人为 pass-rate 目标决定估计器。

实验开始前必须固定：候选方法、窗口、截断距离分箱、缺失处理、机场折和报告字段；之后不得根据结果临时增加“更好看”的分支。

## 12. Evaluation 的严格数学定义

### 12.1 目标点与误差

对已选 procedure/runway context，发布的阈值路径高度为

$$
H_{\mathrm{ref}}^{\mathrm{MSL}}
=H_0^{\mathrm{MSL}}+H_{\mathrm{TCH}}.
$$

event 的 lateral 与 vertical signed errors 为

$$
e_L=\widehat c_*,
\qquad
e_V=\widehat H_*^{\mathrm{MSL}}-
H_{\mathrm{ref}}^{\mathrm{MSL}}.
$$

由于 $\widehat H_*^{\mathrm{MSL}}=H_0^{\mathrm{MSL}}+\widehat u_*$，垂直误差也可写成

$$
e_V=\widehat u_*-H_{\mathrm{TCH}}.
$$

### 12.2 LPV lateral

令 $F_L$ 为 authoritative FAS 在 LTP 的 one-sided lateral full-scale magnitude，跑道宽度为 $W$。有效 lateral bound 是

$$
B_L=\min\left(0.5F_L,\ 0.5W\right).
$$

判决为

$$
V_L=
\begin{cases}
\texttt{pass},&|e_L|\le B_L,\\
\texttt{fail},&|e_L|>B_L,\\
\texttt{indeterminate},&e_L\text{ 或 }B_L\text{ 不可用}.
\end{cases}
$$

本轮不改变该横向标准。`0.5W` 是导航参考点处于跑道边界内的项目几何 guard；它不声称机翼或起落架完全位于道面内。

### 12.3 LPV vertical

依据 [`evaluation/FINAL_APPROACH_VERDICT_STANDARD.md`](../evaluation/FINAL_APPROACH_VERDICT_STANDARD.md) 中已审计的标准链：DO-229 的 LTP 附近 LPV minimum linear vertical full-scale magnitude 为

$$
F_V=15\ \mathrm m,
$$

ICAO Doc 9613 Fifth Edition, Volume II, Part C, Chapter 5, Section B, §5.3.3.1.1.1(b) 的正常运行比例为

$$
\gamma_V=0.5.
$$

因此

$$
B_V=\gamma_VF_V=0.5\times15=7.5\ \mathrm m.
$$

垂直判决为

$$
V_V=
\begin{cases}
\texttt{pass},&|e_V|\le7.5\ \mathrm m,\\
\texttt{fail},&|e_V|>7.5\ \mathrm m,\\
\texttt{indeterminate},&e_V\text{ 或适用 vertical context 不可用}.
\end{cases}
$$

### 12.4 Composite verdict

对有效 LPV event：

$$
V=
\begin{cases}
\texttt{fail},&V_L=\texttt{fail}\ \lor\ V_V=\texttt{fail},\\
\texttt{pass},&V_L=V_V=\texttt{pass},\\
\texttt{indeterminate},&\text{其他情况}.
\end{cases}
$$

对 computed/predicted trajectory 的 `not_reached`，按现有标准为 `fail`；对 observed 数据没有可信 event，则为 `indeterminate`。这一区别来自 source semantics，不来自不同的 LPV gate。

## 13. 当前证据与问题规模

当前五机场开发数据中，有有效 estimated event 与 bracket 的 28,090 条航迹里，v7 fit 与 bracket 的垂直 pass/fail 分类有 8,753 条不一致：

$$
\frac{8753}{28090}=31.16\%.
$$

列联表为：

| v7 fit | bracket | 数量 |
|---|---|---:|
| pass | pass | 16,147 |
| pass | fail | 6,000 |
| fail | pass | 2,753 |
| fail | fail | 3,190 |

该矩阵不能证明 bracket 一定正确，也不能证明整条高度无效；它证明两种估计路径在目标位置上存在系统性差异，必须通过 source-only 实验解决，而不能靠 verdict gate 选边。

差异随 bracket gap 明显增大：

| `lastposupdate` gap | 数量 | 分类不一致率 |
|---|---:|---:|
| $\le2$ s | 19,564 | 20.97% |
| $(2,5]$ s | 5,266 | 48.65% |
| $(5,10]$ s | 2,280 | 62.54% |
| $>10$ s | 980 | 67.65% |

按整条 track 是否出现 async group 分层：

| track audit | 数量 | 分类不一致率 |
|---|---:|---:|
| `geoaltitude_async_groups = 0` | 10,605 | 19.25% |
| `geoaltitude_async_groups > 0` | 17,485 | 38.39% |

所以 `geoaltitude` 时间对齐风险是重要解释变量，但不是唯一原因；即使没有记录到 async group，也不能让远端拟合代替域内交点而不验证。

此前“direct bracket 不可用”的实验发生在 source-timing 重建之前，使用了错误/混合的 state-row 时间与 `lastposupdate` 时间语义。重建之后没有按同一预注册方案重新比较，因此旧结论不能继续作为 v7 拟合优先的依据。

## 14. Pipeline 边界与迁移

目标数据流为：

```text
OpenSky source rows
  -> source-timed contiguous Track (HAE; raw observations preserved)
  -> airport landing screen
  -> runway/pass association
  -> RunwayThresholdEvent
       direct bracket interpolation, or
       validated censored predictor, or
       unavailable
  -> observed record (datum conversion only; no refit)
  -> LPV/LNAV-VNAV evaluation
  -> report / CZML / frontend
```

实施后再生成顺序必须是：

1. 在不下载 ADS-B 的前提下，对存储的 HAE samples 运行 `--reclassify-existing`；
2. 写出新 event schema 和新的 track SHA；
3. 重建 arrivals/manifest 等可再生成派生视图；
4. 重跑 evaluation 报告；
5. 更新 CZML/前端展示，使 direct 与 inferred evidence 可区分；
6. 先一个 development airport 审计，再跑其余开发机场。

`--evaluate-only` 只能消费当前 schema，不能升级 event。旧 event 必须因 schema 不匹配而失败并提示 reclassify，不能静默读取。

## 15. 测试与验收矩阵

### 15.1 单元测试

至少覆盖：

1. 坐标投影/逆投影 round trip；
2. $a_i<0<a_{i+1}$ 的 $alpha,t,c,u$ 精确结果；
3. $a_{i+1}=0$ 的唯一 bracket，不重复选择下一对；
4. 同一 $alpha$ 同时用于三维事件；
5. bracket 横跨非递增时间、coverage gap、速度不相容时拒绝；
6. 几何 bracket 但 integrity 失败时为 `invalid_support`，不回退为 censored fit；
7. multipass 只能使用 selected inbound pass；
8. right-censored event 的 source range 不越过 pass 边界；
9. threshold-frame fingerprint mismatch 被严格拒绝；
10. HAE/MSL 等价公式；
11. evaluation 不导入或调用 `fit_final_segment()`；
12. event 中没有 TCH、FSD、approach type 或 verdict；
13. 所有 NaN/Inf 在读取、计算或严格 JSON 写出前失败；
14. LPV `±7.5 m` 和 lateral inclusive boundary；
15. observed unavailable 与 computed not_reached 的不同 composite semantics。

### 15.2 回归与性质测试

对任意合法 bracket，应满足

$$
0<\alpha\le1,
\quad
\widehat a_*=0,
\quad
\widehat t_*\in(\tau_i,\tau_{i+1}],
$$

且

$$
\widehat c_*\in[\min(c_i,c_{i+1}),\max(c_i,c_{i+1})],
$$

$$
\widehat u_*\in[\min(u_i,u_{i+1}),\max(u_i,u_{i+1})].
$$

更换 evaluation approach context 不得改变同一个 physical event：

$$
E(\text{LPV context})=E(\text{LNAV/VNAV context}).
$$

更换会改变 LTP frame 的 runway/FAS cycle 必须导致 $f_{\mathrm{frame}}$ 不同，从而拒绝旧 event：

$$
f_{\mathrm{frame,old}}\ne f_{\mathrm{frame,new}}
\Longrightarrow
\operatorname{consume}(E_{\mathrm{old}},C_{\mathrm{new}})=\texttt{error}.
$$

若只改变 TCH/FSD/width 等评价事实而 LTP frame 不变，则 physical event 仍可复用，但 $f_{\mathrm{eval}}$ 必须变化并重算 verdict/report：

$$
f_{\mathrm{frame,old}}=f_{\mathrm{frame,new}},\quad
f_{\mathrm{eval,old}}\ne f_{\mathrm{eval,new}}
\Longrightarrow
E_{\mathrm{old}}\text{ 可复用，}V_{\mathrm{old}}\text{ 不可复用}.
$$

### 15.3 数据级验收

数据级报告必须至少给出：

- 每机场 `within_observed_support / right_censored / invalid_support / unavailable` 数量；
- direct bracket gap、$alpha$、高度量化模式、async-group 分层；
- censored 外推距离分布；
- artificial-censoring 的 bias、median absolute error、P95 和区间覆盖率；
- 新旧 event 差异，但不得用 pass rate 选择方法；
- runtime、峰值内存和每条 track 的处理复杂度。

## 16. 性能设计约束

设轨迹点数为 $N$，机场候选跑道数为 $R$。每条航迹允许的目标复杂度是

$$
T(N,R)=O(NR),
\qquad
M(N,R)=O(NR)\ \text{或更小}.
$$

具体要求：

1. 每个 track/runway 只投影一次，关联、event 和诊断复用投影数组；
2. bracket 扫描为一次线性 pass；
3. direct event 不运行 final-segment fit；
4. censored event 只运行最终被采用的一种 fit，不在生产路径维护三窗口 ensemble；
5. robust seed 若使用 pairwise slope，样本数必须有固定上限，使成本受控；
6. 批量写出使用原子替换，但不得复制一份完整原始数据作为临时备份；
7. 所有实验产物写到明确的 development/audit 路径，并可独立清理，不覆盖 raw source。

## 17. 实施阶段与停止条件

### 阶段 A：设计冻结与实验工具

- 冻结本文中的数学对象、候选方法、指标和数据切分；
- 写纯函数单元测试与只读实验入口；
- 不改正式 event schema，不重跑全量数据。

### 阶段 B：小规模 source-only 实验

- 使用一个 development airport 和预注册子集验证 direct 方法；
- 用 bracket 航迹做人工截断，验证 censored predictor；
- 若基础方法已按第 11.4 节胜出，不增加更复杂模型。

停止条件：若 source-time 或 datum 审计发现输入契约被破坏，先修复输入；不得继续用更复杂拟合掩盖输入问题。

### 阶段 C：单一实现

- 删除生产路径中的 v7 event 估计分支；
- 实现一个 event domain object 和两个互斥 evidence path；
- evaluation 只消费新 event；
- focused Python tests 全部通过。

### 阶段 D：一个机场重生成与审计

- `--reclassify-existing`，不下载；
- 重建 observed/evaluation/CZML 派生数据；
- 检查 event counts、identity、runway association、datum、runtime 和报告解释；
- 不根据 pass rate 临时改 estimator。

### 阶段 E：其余 development airports

- 空间检查后完成全量派生数据重生成；
- 发布方法分层统计；
- outer-test 保持未查看，直到用户明确冻结实验并请求最终发布。

## 18. 参考资料与代码事实来源

### 18.1 项目内代码与标准档案

- [`final_approach/frame.py`](../final_approach/frame.py)：当前跑道局部坐标定义。
- [`trajectory_data_process/harvest/tracks.py`](../trajectory_data_process/harvest/tracks.py)：`lastposupdate`、source freshness、重复 snapshot 合并与连续块定义。
- [`trajectory_data_process/harvest/threshold_event.py`](../trajectory_data_process/harvest/threshold_event.py)：当前 bracket 与 v7 event 生产逻辑。
- [`final_approach/fit.py`](../final_approach/fit.py)：当前 robust final-segment fit。
- [`evaluation/arrival.py`](../evaluation/arrival.py)：当前 observed/optimized/predicted event 入口。
- [`evaluation/metrics.py`](../evaluation/metrics.py)：当前 verdict 与 report criteria。
- [`evaluation/FINAL_APPROACH_VERDICT_STANDARD.md`](../evaluation/FINAL_APPROACH_VERDICT_STANDARD.md)：LPV/LNAV-VNAV 标准、章节索引与官方文档审计。
- [`final_approach/FIT_MODEL_OPTIMIZATION.md`](../final_approach/FIT_MODEL_OPTIMIZATION.md)：v7 历史决策与实验记录；仅作历史，不作为新实现规范。

### 18.2 本地文献

- Olive et al. (2020), *Detecting Events in Aircraft Trajectories: Rule-Based and Data-Driven Approaches*，§§3.3、3.5 与 Figures 2、4：支持先建立跑道/航段上下文再识别事件，并明确列出 successive runway alignments、go-around 与 circle-to-land 的歧义。见 [本地 PDF](literature_review/threshold_event_estimation/Olive_et_al_2020_Trajectory_Event_Detection.pdf)。
- Olive et al. (2024), *Filtering Techniques for ADS-B Trajectory Preprocessing*，§§2.1–2.3、3：支持显式处理 receiver timestamp、state-vector 高度聚合、25/100/500 ft 量化和一致性/鲁棒滤波，并警示未知的上游滤波会妨碍可靠清理。见 [本地 PDF](literature_review/threshold_event_estimation/Olive_et_al_2025_ADSB_Filtering.pdf)。
- Waltert & Figuet (2023), *Using ADS-B Trajectories to Measure How Rapid Exit Taxiways Affect Airport Capacity*，**Estimation of landing time** 小节：用 fully covered landings 建立 train/validation/test 数据，再预测覆盖不足航迹的剩余时间。它支持“用完整覆盖样本验证缺失终点预测”的一般范式；本文的 artificial-censoring 方案是针对阈值高度/横向交点进一步作出的项目方法，不冒充论文原结论。见 [本地 PDF](literature_review/threshold_event_estimation/Waltert_Figuet_2024_ADSB_Missing_Landing_Time.pdf)。
- NASA/TM-20220019263, *Assessing Several Non-Traditional Data Sources for Value in Aviation Safety*，**Crowdsourced ADS-B Data to Estimate Stability of Flight Approach and Landing** 小节：其处理顺序先确定 landing runway、再计算 threshold distance/localizer/glideslope deviations，支持关联先于偏差解释。见 [本地 PDF](literature_review/threshold_event_estimation/NASA_TM_20220019263.pdf)。

上述论文用于事件检测与验证方法，不提供 LPV verdict bound。LPV 标准仍以 `FINAL_APPROACH_VERDICT_STANDARD.md` 中逐章审计的官方来源为准。

## 19. 尚未通过实验决定的问题

以下问题保持显式开放，不能在实现时偷偷决定：

1. direct 两点插值是否需要由局部鲁棒一次模型替代；
2. bracket gap 是否需要比上游连续块更严格的最大值；
3. `geoaltitude_async_groups` 应只作诊断，还是应触发 event-level evidence 降级；
4. censored predictor 的固定窗口与最大允许外推距离；
5. 是否能校准可信的 direct/censored uncertainty，或只能报告 `uncertainty.status = unavailable`；
6. KRDU 等 bracket coverage 低的机场是否暴露接收覆盖差异，还是关联逻辑仍有缺口。

这些问题必须由第 11 节的 source-only 开发实验回答。任何答案都不能由“让成功落地航迹更容易 pass”推出。

## 20. 在冻结基线上的简化审查

本节是在第 1–19 节完整方案已经落盘后进行的第二步。简化目标不是“减少字段数量看起来更短”，而是用更少的生产分支表达同一个物理问题。

### 20.1 简化判据

设候选架构为 $A$，其生产状态数、事件估计器数、对同一数据的重复投影/拟合次数和 public contract 字段数分别为

$$
n_s(A),\quad n_e(A),\quad n_r(A),\quad n_f(A).
$$

定义只用于架构比较的复杂度序组

$$
\mathcal C(A)=
\left(n_s(A),\ n_e(A),\ n_r(A),\ n_f(A)\right).
$$

可接受的简化 $A'\prec A$ 必须同时满足：

$$
\mathcal C(A')<_{\mathrm{lex}}\mathcal C(A)
$$

以及所有第 2.2 节不变量、所有第 15 节测试性质仍成立。换言之，少一个分支但失去 `observability`、frame fingerprint 或 evaluation-context fingerprint 不是简化，而是丢失问题定义。

### 20.2 推荐的最小生产架构

最终生产数据流可以收敛为三个纯边界：

```text
normalize_source(rows) -> Track
resolve_final_approach(Track, AirportGeometry) -> FinalApproachResolution
evaluate_threshold_event(Event, EvaluationContext) -> Verdict
```

其中 `FinalApproachResolution` 只组合两个概念上独立的结果：

```text
RunwayAssociation
RunwayThresholdEvent
```

`resolve_final_approach` 是一个编排函数，而不是把关联和评价混成一个算法。其严格决策树为

$$
\operatorname{resolve}(T,R)=
\begin{cases}
\operatorname{direct}(B^*),
&\exists\text{ 唯一 source-valid structural bracket }B^*,\\
\operatorname{unavailable}(\texttt{invalid\_support}),
&\exists\text{ plausible geometric bracket 但 integrity 失败},\\
\operatorname{censored}(F^*),
&\nexists\text{ geometric bracket 且存在唯一可信 final fit }F^*,\\
\operatorname{unavailable}(q),
&\text{其他情况}.
\end{cases}
$$

这里 direct 与 censored 两个方法不是 legacy 版本，也不是可切换实验模式；它们分别对应两个互斥的数学问题：插值与外推。每个可观测性类别在生产中只有一个实现。

### 20.3 可以直接删除的重复复杂度

#### A. direct path 的 final-segment fit

当 valid bracket 已存在时，当前 v7 仍运行多个阈值前窗口并发布 fit intercept。新设计中：

$$
\texttt{within\_observed\_support}
\Longrightarrow
\widehat E_*=\operatorname{direct\_linear\_bracket}(B^*).
$$

因此 direct path 不需要：

- assignment 后再次 event fit；
- `[-3000,-300]`、`[-4000,-300]`、`[-5000,-300]` 三窗口 ensemble；
- window-sensitivity 选择；
- `candidate_fits` public payload；
- “bracket 只作 anchor”的角色字段。

#### B. censored path 的第二次拟合

无 valid bracket 时，跑道关联已经需要对候选跑道拟合。令候选集合为 $\{F_r\}$，获胜跑道为

$$
r^*=\arg\min_r\operatorname{median}_{i\in\mathcal F_r}|c_i^{(r)}|.
$$

获胜的 $F_{r^*}$ 已包含 $(\widehat\beta_{0,c},\widehat\beta_{0,u})$，所以 event 直接复用它：

$$
\widehat E_*=\operatorname{event}(F_{r^*}).
$$

禁止 `assign_runway()` 拟合一次、`build_event()` 再拟合一次。

#### C. 重复时间查找

新 schema 只接受已完成 `opensky-source-timing-v1` 重建的 track。对这种 track：

$$
\tau_i=\texttt{sample.time}_i=\texttt{lastposupdate}_i.
$$

且 `reported_ground_speeds_m_s` 已随 stored track 保存。因此 reclassification/event producer 不需要再为每个 bracket 向 sidecar 查询同一条 `lastposupdate` 和 velocity；sidecar 只属于旧 raw-to-source-timed 重建阶段。

#### D. 重复 gap gate

上游已经保证连续 block 内

$$
0<\Delta\tau_i\le15\ \mathrm s.
$$

因此 canonical event producer 中的独立 `\Delta\tau\le30\ \mathrm s` 条件没有额外接受域：

$$
\{\Delta\tau:0<\Delta\tau\le15\}
\subset
\{\Delta\tau:0<\Delta\tau\le30\}.
$$

新 event producer 应断言 source-timing schema，而不是维护第二套 gap policy。位移/报告速度一致性检查仍保留，因为它验证的是不同物理事实。

### 20.4 public event 的最小字段集合

第 10 节给出语义全集。实现时可把字段进一步归并为五个不可再删的块：

```text
identity:
  schema_version, runway,
  threshold_frame_snapshot, threshold_frame_fingerprint

state:
  status, event_time_s, lat, lon, altitude_m, altitude_datum,
  signed_cross_track_m

evidence:
  observability, method, source_sample_range,
  interpolation_fraction | extrapolation_distance_m

quality:
  uncertainty_status, uncertainty, source_integrity_summary

failure:
  unavailable_reason
```

只有 `status = estimated` 时才允许 `state` 数值存在；只有 direct 才允许 `interpolation_fraction`；只有 censored 才允许 `extrapolation_distance_m`。这可以写成 discriminated union，而不是一个允许任意字段组合的大字典。

生产 payload 不再发布所有 candidate fit。需要复现实验的候选细节写入独立 development audit artifact；正式 event 只保留最终采用方法的必要残差/样本跨度诊断。

### 20.5 uncertainty 的最小可靠方案

当前 evaluation 明确不使用 uncertainty 改变 point verdict，因此在尚未校准完整 $\Sigma_E$ 时，最可靠的最小契约是

$$
\texttt{uncertainty.status}=\texttt{uncalibrated},
\qquad
\texttt{uncertainty.value}=\varnothing.
$$

这比把 regression standard error、窗口敏感度和高度量化下界相加后称作完整 `95%` 更诚实，也能删除三窗口 ensemble。只有第 11 节验证出按机场/距离具有可接受覆盖率的区间后，才发布

$$
\texttt{uncertainty.status}=\texttt{calibrated},
\qquad
\texttt{uncertainty.covariance}=\widehat\Sigma_E.
$$

该简化不会削弱 verdict，因为 verdict 从来不依赖 uncertainty；它只避免伪精度。

### 20.6 实验代码与生产代码的隔离

为了满足“只维护一个正式版本”，候选 direct/censored 方法的生命周期应为：

1. 在只读 development experiment 中实现为局部函数；
2. 按第 11 节生成固定审计结果；
3. 选出一个 direct 方法和一个 censored 方法；
4. 仅把胜出者移入生产模块；
5. 删除失败候选的可执行代码，保留公式、commit 和结果表。

生产 CLI 不增加 `--event-method=v7|v8|linear|quadratic`。可再生成 artifact 只接受一个当前 schema。

### 20.7 不应简化掉的内容

以下内容看似增加字段或分支，实际表达了不可合并的物理状态：

| 必须保留 | 原因 |
|---|---|
| selected pass 下界 | 防止 multipass 把不同进近拼成一个 event |
| `within_observed_support` 与 `right_censored` | 区分插值和外推，二者证据强度不同 |
| `invalid_support` | 防止 integrity 失败被静默伪装成正常缺失 |
| HAE/MSL datum、frame fingerprint 与 evaluation-context fingerprint | 防止数十米级 datum/cycle 错配，同时不把 approach profile 塞进 event |
| source sample range | 使 event 可复核且保证没有跨 pass |
| runway ambiguity | 防止平行跑道被强行猜测 |
| strict finite JSON | 防止 NaN 假 pass 和非标准 JSON |
| evaluation no-refit | 保证所有消费者评价同一个物理事件 |
| method/evidence label | 防止 inferred event 混入 direct observed 统计 |

### 20.8 对 identity 和 ML pipeline 的最小影响

本重构不把新 event time 改成新的航班身份时间。保持

$$
\texttt{flight\_key}
=f(\texttt{callsign},\texttt{runway},\texttt{icao24},
\texttt{existing landing\_time\_utc semantics}).
$$

否则即使轨迹样本完全没变，也会让 train/validation joins、文件名和预测引用整体失配。若未来要改变 landing-time identity，必须作为独立迁移设计，不能夹带在 fitting 重构中。

arrivals 的数值样本仍来自原始重建 Track；event 只供评价和可视化阈值点使用。因此本重构本身不加入“只训练 pass 航迹”的筛选，也不改变 TS split。任何训练筛选必须由独立、显式的实验设计决定。

### 20.9 推荐的最终模块责任

```text
trajectory_data_process/harvest/tracks.py
  唯一 source timing / contiguity normalization

final_approach/frame.py
  唯一 runway-frame projection

final_approach/resolve.py
  project once
  -> associate runway/pass
  -> direct event OR censored event OR unavailable

final_approach/fit.py
  仅保留 censored/无 bracket 路径所需的一个 robust line implementation

trajectory_data_process/harvest/classify.py
  landing screen + 调用 resolver + stable identity/storage orchestration

evaluation/arrival.py
  把 observed/optimized/predicted 输入映射为同一个 Event；observed 只读取

evaluation/metrics.py
  纯 policy function：Event × Context -> Verdict
```

不必为了模块数量少而把 `frame.py`、source normalization 或 evaluation 合进 resolver；那会破坏可测试边界。真正应删除的是重复估计，而不是清晰责任。

### 20.10 推荐的最小算法（实验前默认）

在尚未得到新实验结果前，最小且可证伪的候选是：

$$
\boxed{
\begin{aligned}
\text{direct:}\quad
&\widehat E_*=\operatorname{linear\_interpolate}(Y_i,Y_{i+1};a=0),\\
\text{censored:}\quad
&\widehat E_*=\operatorname{robust\_line\_extrapolate}
(\{Y_j:A_-\le a_j\le A_+<0\};a=0).
\end{aligned}}
$$

其中 direct 没有窗口、没有远端 glidepath 假设；censored 只有一个固定窗口、一个获胜 fit、一个方法版本。若 source-only 实验不能证明更复杂方法降低最坏机场误差，生产实现就停在这里。

### 20.11 简化后的代码量与运行路径预期

不能在实现前承诺具体加速倍数，但可以严格比较调用次数。设有 bracket 的航迹数为 $N_B$，right-censored 航迹数为 $N_C$，候选跑道数均值为 $\bar R$。当前 v7 的 event 阶段还对已关联航迹运行最多三个窗口；简化后 event 额外 fit 次数为

$$
N_{\mathrm{event\ fit,new}}=0.
$$

censored 关联所需 fit 已计入 runway assignment 并直接复用。direct 航迹的 event 计算从多次 robust fit 降为常数次线性插值。总投影仍受跑道候选扫描约束：

$$
T_{\mathrm{new}}=O((N_B+N_C)\bar R\bar N),
$$

但移除了同一 winning runway 的重复投影/多窗口拟合常数项。实际 wall time 必须由一个机场的相同输入 A/B benchmark 报告。

### 20.12 简化方案的实施门槛

这份简化建议现在仍是设计，不授权直接改生产代码。进入实施前只需完成两个小规模、只读实验：

1. direct：两点线性与局部鲁棒一次模型的留点/降采样稳定性比较；
2. censored：复用当前单窗口 robust fit，在真实缺失距离分布上的 artificial-censoring 跨机场误差曲线。

若结果支持最小算法，后续实现应一次替换 v7，而不是先加入第三套模式。若结果不支持，则只增加由误差证据明确要求的最小复杂度，并回写本文第 19 节对应 decision record。
