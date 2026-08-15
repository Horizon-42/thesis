# 跑道阈值事件简化实现开发文档

状态：**已实现；source-only 实验、focused tests 与五机场全量迁移完成**

日期：2026-08-15

> 标准更新说明：本文的事件估计器设计保持有效；垂直 verdict 已由
> `terminal-approach-evaluation-v4` 统一改为 `±22 m` RNAV 终端几何界限。
> 第 17.3–17.4 节的旧 pass/fail 数量是标准更新前的历史运行记录，不是当前结果。

上位规范：[跑道阈值事件：第一性原理开发档案](threshold-event-first-principles-development.zh.md)

## 1. 目标

用最少的生产算法估计同一个物理量：已选 final inbound pass 的导航参考点轨迹与 landing threshold point（LTP）平面的一次 inbound 交点。

最终生产路径只允许：

$$
\boxed{
\begin{aligned}
\text{域内：}\quad
&\widehat E_*=\operatorname{direct\_linear\_bracket}(Y_i,Y_{i+1}),\\
\text{域外：}\quad
&\widehat E_*=\operatorname{censored\_robust\_line}(F_{r^*}),\\
\text{无有效证据：}\quad
&E_*=\texttt{unavailable}.
\end{aligned}}
$$

这里的两个算法不是版本或用户可选模式。它们分别对应插值和外推两个互斥数学问题。生产 CLI 不增加 estimator 开关，也不保留 `observed-threshold-event-v7` 双读。

## 2. 明确不做的事情

- 不重新下载 ADS-B；
- 不修改 `samples` 的数值、顺序、datum 或时间语义；
- 不修改 raw samples、`icao24`、`start_time_utc` 等源身份锚点；
- `--reclassify-existing` 会在 runway/outcome 变化时重建包含这些派生字段的
  `flight_key`，因此必须重建下游 roster/split，不能把旧 checkpoint 视为与新
  manifest 同一数据版本；
- 不按 evaluation pass/fail 筛选轨迹；
- 不让 LPV lateral 或 vertical 标准参与事件估计器选择；
- 不查看或使用 outer-test；
- 不把 TCH、FSD、runway width、approach type 或 verdict 写进 event；
- 不在 evaluation 中 refit observed trajectory；
- 不保留三窗口 ensemble、候选 fit payload 或 v7 runtime fallback。

## 3. 数学对象

### 3.1 跑道坐标

对 LTP $(\varphi_0,\lambda_0,H_0^{\mathrm{HAE}})$ 和着陆方位 $\psi$，令

$$
N=(\varphi-\varphi_0)m_{\mathrm{lat}},
\qquad
E=(\lambda-\lambda_0)m_{\mathrm{lon}}(\varphi_0),
$$

$$
\begin{aligned}
a &= E\sin\psi+N\cos\psi,\\
c &= E\cos\psi-N\sin\psi,\\
u &= H^{\mathrm{HAE}}-H_0^{\mathrm{HAE}}.
\end{aligned}
$$

$a<0$ 表示阈值前，$c>0$ 表示着陆方向右侧。

### 3.2 目标事件

对已选 pass $I_p$，事件时间与状态为

$$
t^*=\inf\{t\in I_p:a(t)=0\land\dot a(t)>0\},
$$

$$
E^*=(t^*,0,c(t^*),u(t^*)).
$$

evaluation 只评价 $E^*$，不评价整个 LPV cone，也不评价最后若干样本的最大误差。

## 4. 单一 resolver 决策树

输入是已经满足 `opensky-source-timing-v1` 的一个 `Track` 和机场的所有 landing thresholds。输出是 `RunwayAssociation` 与 `RunwayThresholdEvent`。

$$
\operatorname{resolve}(T,R)=
\begin{cases}
\operatorname{direct}(B^*),
&B^*\text{ 是唯一且无歧义的 source-valid bracket},\\
\operatorname{unavailable}(\texttt{invalid\_support}),
&\text{winning runway 上存在 plausible bracket，但其 source integrity 失败},\\
\operatorname{censored}(F_{r^*}),
&\max_{i\in P_{r^*}}a_i<0\text{ 且存在唯一可信 winning fit},\\
\operatorname{unavailable}(q),
&\text{其他情况}.
\end{cases}
$$

概念边界仍然分开：

1. bracket/fit 选择 runway 与 physical pass；
2. event estimator 只估计交点；
3. evaluation 只应用标准。

它们可以由一个编排函数顺序调用，但不得共享 verdict policy。

## 5. Direct event

### 5.1 bracket

相邻样本 $(i,i+1)$ 构成 inbound bracket，当且仅当

$$
a_i<0\le a_{i+1},
\qquad
a_{i+1}>a_i.
$$

定义唯一插值系数

$$
\alpha=\frac{-a_i}{a_{i+1}-a_i},
\qquad 0<\alpha\le1.
$$

### 5.2 完整三维插值

$$
\begin{aligned}
\widehat t_*&=(1-\alpha)\tau_i+\alpha\tau_{i+1},\\
\widehat a_*&=0,\\
\widehat c_*&=(1-\alpha)c_i+\alpha c_{i+1},\\
\widehat u_*&=(1-\alpha)u_i+\alpha u_{i+1}.
\end{aligned}
$$

经纬度和 HAE 高度通过同一个 inverse projection 产生：

$$
(\widehat\varphi_*,\widehat\lambda_*,\widehat H_*^{\mathrm{HAE}})
=\operatorname{unproject}(0,\widehat c_*,\widehat u_*).
$$

禁止使用

$$
(\widehat c_*^{\mathrm{bracket}},\widehat u_*^{\mathrm{fit}})
$$

这样的 hybrid。

### 5.3 source-integrity 条件

上游连续块已经保证

$$
0<\Delta\tau\le15\ \mathrm{s}.
$$

event producer 不再维护重复的 $30\ \mathrm s$ gap policy，但保留 reported-speed 一致性：

$$
v^{\mathrm{pos}}=\frac{d(Y_i,Y_{i+1})}{\Delta\tau},
\qquad
\bar v^{\mathrm{rep}}=\frac{v_i+v_{i+1}}2,
$$

$$
|v^{\mathrm{pos}}-\bar v^{\mathrm{rep}}|\le25\ \mathrm{m/s},
\qquad
0.5\le\frac{v^{\mathrm{pos}}}{\bar v^{\mathrm{rep}}}\le1.5,
$$

$$
0<v_i,v_{i+1}\le200\ \mathrm{m/s}.
$$

速度来自 stored track 的 `reported_ground_speeds_m_s`；source-timed track 的 `sample.time_s` 已经等于 `lastposupdate`，event resolver 不再逐 bracket 查询 metadata sidecar。

## 6. Censored event

### 6.1 适用条件

只有 winning inbound pass 的全部可信支持点都在阈值前时才可进入 censored path：

$$
a_{\max}=\max_{i\in P}a_i<0.
$$

外推距离为

$$
d_{\mathrm{ext}}=-a_{\max}>0.
$$

几何 bracket 已存在但 integrity 失败时，禁止使用 censored path。

### 6.2 单一鲁棒直线

对固定阈值前窗口

$$
\mathcal F=\{i\in P:-5000\le a_i\le-300\ \mathrm m\},
$$

分别拟合

$$
y_i=\beta_0+\beta_1a_i+\epsilon_i,
\qquad y_i\in\{c_i,u_i\}.
$$

现有 `fit_final_segment()` 的 robust seed、gross-outlier removal 和 winning assignment fit 保留。winner $F_{r^*}$ 已含

$$
\widehat c_*=\widehat\beta_{0,c},
\qquad
\widehat u_*=\widehat\beta_{0,u}.
$$

event producer 必须直接复用 $F_{r^*}$，不能再运行第二次 fit，也不再计算三个窗口的 ensemble。

## 7. 跑道与 pass 关联

### 7.1 bracket path

每条跑道选择绝对 cross-track 最小的 source-valid structural bracket：

$$
S_r=|\widehat c_{*,r}|,
\qquad
r^*=\arg\min_r S_r.
$$

第一、第二名差值小于 $50\ \mathrm m$ 时为 ambiguous：

$$
S_{(2)}-S_{(1)}<50\ \mathrm m.
$$

selected bracket 同时定义 physical pass anchor；任何局部样本不得越过该 pass 下界。

### 7.2 censored path

没有 plausible geometric bracket 时，使用现有 candidate fit 的相对 cross-track score：

$$
S_r^{\mathrm{fit}}
=\operatorname{median}_{i\in\mathcal F_r}|c_i^{(r)}|.
$$

winning fit 同时供 event 使用。assignment 与 event 不允许各自选择不同窗口或不同 pass。

## 8. 物理 frame 与 evaluation policy 分离

event 绑定物理 threshold frame snapshot：

$$
S_{\mathrm{frame}}=
\{\text{airport, runway, LTP lat/lon, }H_0^{\mathrm{HAE}},
H_0^{\mathrm{MSL}},G_0,\psi,\text{ physical source ids/cycles}\}.
$$

$$
f_{\mathrm{frame}}
=\operatorname{SHA256}(\operatorname{canonicalJSON}(S_{\mathrm{frame}})).
$$

evaluation context 独立保存

$$
S_{\mathrm{eval}}=
\{f_{\mathrm{frame}},H_{\mathrm{TCH}},F_L,W,
\text{approach id, standard id, source ids/cycles}\},
$$

$$
f_{\mathrm{eval}}
=\operatorname{SHA256}(\operatorname{canonicalJSON}(S_{\mathrm{eval}})).
$$

因此 event 不含 TCH/FSD/width，却仍能拒绝旧 LTP frame。只改变 evaluation policy 时可以复用 physical event，但必须重算 verdict/report。

## 9. 新 event schema

新 schema 名为：

```text
runway-threshold-event-v1
```

只有三个互斥 payload 形态。

### 9.1 Direct

```text
schema_version: runway-threshold-event-v1
status: estimated
runway
threshold_frame_snapshot
threshold_frame_fingerprint
observability: within_observed_support
method: direct_linear_bracket
event_time_s
threshold_crossing_lat/lon/altitude_m
altitude_datum: hae
signed_cross_track_m
source_sample_range: [i, i+1]
interpolation_fraction
extrapolation_distance_m: 0
uncertainty: {status: uncalibrated}
source_integrity
diagnostics
```

### 9.2 Censored

```text
schema_version: runway-threshold-event-v1
status: estimated
runway
threshold_frame_snapshot
threshold_frame_fingerprint
observability: right_censored
method: censored_robust_line
event_time_s: null
threshold_crossing_lat/lon/altitude_m
altitude_datum: hae
signed_cross_track_m
source_sample_range
interpolation_fraction: null
extrapolation_distance_m > 0
uncertainty: {status: uncalibrated}
source_integrity
diagnostics
```

### 9.3 Unavailable

```text
schema_version: runway-threshold-event-v1
status: unavailable
runway: optional
threshold_frame_snapshot/fingerprint: present when runway is known
observability: invalid_support | unavailable
method: none
unavailable_reason
diagnostics
```

所有数值必须有限。旧 schema 被严格拒绝并提示 `--reclassify-existing`；不提供 v7 fallback。

## 10. Evaluation

物理 event 的 MSL 高度为

$$
\widehat H_*^{\mathrm{MSL}}
=H_0^{\mathrm{MSL}}+\widehat u_*.
$$

LPV 误差为

$$
e_L=\widehat c_*,
\qquad
e_V=\widehat H_*^{\mathrm{MSL}}
-(H_0^{\mathrm{MSL}}+H_{\mathrm{TCH}})
=\widehat u_*-H_{\mathrm{TCH}}.
$$

横向 bound 不变：

$$
B_L=\min(0.5F_L,0.5W).
$$

垂直 bound 由 evaluation v4 统一定义：

$$
B_V=22\ \mathrm m.
$$

其依据和适用范围见
[`evaluation/FINAL_APPROACH_VERDICT_STANDARD.md`](../evaluation/FINAL_APPROACH_VERDICT_STANDARD.md)。
它不进入 direct/censored resolver，也不改变物理 event。

$$
V_q=
\begin{cases}
\texttt{pass},&|e_q|\le B_q,\\
\texttt{fail},&|e_q|>B_q,\\
\texttt{indeterminate},&e_q\text{ 或 }B_q\text{ 不可用}.
\end{cases}
$$

uncertainty 在校准前写为 `uncalibrated`，不制造伪 `95%` 值，也不改变 point verdict。

## 11. 实验门槛

### 11.1 Direct

在 development 数据上比较：

- 两点 linear bracket；
- 局部 robust linear challenger。

对内部样本 $k$ 做 leave-one-out pseudo-threshold：

$$
\mathbf e_{k,m}^{\mathrm{LOO}}
=\widehat{\mathbf r}_{k,m}-[c_k,u_k]^{\mathsf T}.
$$

另比较降采样 event 稳定性：

$$
\Delta\mathbf e_{j,m}^{(b)}
=\widehat E_{j,m}(S_b)-\widehat E_{j,m}(S_{\mathrm{full}}).
$$

不得读取 TCH、gate 或 pass rate。复杂方法只有在最坏机场 P95、bias、median absolute error 的字典序风险上优于 linear bracket 时才能进入生产。

### 11.2 Censored

从真实 censored endpoint distance 分布取 $d$，对完整 bracket 航迹构造

$$
P_j^{(d)}=\{i:a_i\le-d\},
$$

并比较

$$
\mathbf e_{j}(d)=g(P_j^{(d)})-E_j^{\mathrm{direct}}.
$$

必须报告机场、$d$ 分箱、bias、median absolute error、P95 与可用率。若当前单窗口 fit 在某距离外失去支持，则 event 变成 unavailable，不能无限外推。

## 12. 实现文件边界

### 12.1 保留

- `trajectory_data_process/harvest/tracks.py`：唯一 source timing/contiguity normalization；
- `final_approach/frame.py`：唯一 runway-frame transform；
- `final_approach/fit.py`：只服务无 bracket 的 runway assignment/censored event；
- `trajectory_data_process/harvest/threshold_event.py`：OpenSky/FAA-aware resolver 与 event serialization；
- `evaluation/arrival.py`：严格消费 observed event，并把 optimized/predicted state 映射为相同的 evaluation event；
- `evaluation/metrics.py`：纯标准判决。

### 12.2 删除的生产逻辑

- `_FitEnsemble`；
- `EXTRAPOLATION_WINDOWS_M`；
- `_fit_ensemble()`；
- direct bracket 后的 `fit_final_segment()`；
- event producer 内任何二次 fit；
- `candidate_fits` 与 window sensitivity；
- `threshold_bracket.role = runway_and_pass_anchor_not_event_estimator`；
- v7 schema/method acceptance。

### 12.3 新增/调整

- `threshold_frame_snapshot()` 与 `threshold_frame_fingerprint()`；
- strict `require_matching_threshold_frame()`；
- `final_approach/event_contract.py` 中 producer/evaluator 共享的 v1
  schema/method 判别常量，以及两端各自的严格语义校验；
- direct/censored discriminated event builders；
- CZML 按 `observability` 判断 direct 与 inferred tail；
- evaluation report 保存 evaluation-context facts/fingerprint；
- tests 先于实现修改。

## 13. 测试先行矩阵

实现前必须先写出会在 v7 上失败的测试：

1. valid bracket 的 event 高度等于同一 bracket 的线性三维交点；
2. direct event 不调用 `fit_final_segment()`；
3. direct event 的时间、横向、垂直共享同一 $\alpha$；
4. direct payload 不含 `candidate_fits`、fit window 或 TCH/FSD；
5. winning runway 的 plausible bracket integrity 失败时不得回退为 censored fit；
6. 无 bracket 时 event 直接复用 `Assignment.fit`；
7. censored source range 不跨 selected inbound pass；
8. physical frame 变化拒绝旧 event；
9. 仅 TCH/FSD/width 变化不改变 frame fingerprint，但改变 evaluation fingerprint；
10. runtime integration test 证明 evaluation 不调用 `fit_final_segment()`；
11. observed/optimized/predicted 使用相同误差符号与 gate；
12. CZML direct event 不添加 inferred tail，censored event 才添加；
13. 非有限 event/criteria 被严格拒绝；
14. v7 artifact 被严格拒绝并要求 reclassify。

## 14. 性能约束

对 $N$ 个样本、$R$ 条候选跑道：

$$
T(N,R)=O(NR),
\qquad M(N,R)=O(NR)\text{ 或更小}.
$$

必须做到：

- 每个 track/runway 的 bracket scan 只投影一次，复杂度为 $O(N)$；
- 仅在 direct 不可用时进入 assignment fitter；fitter 可以维护自己的投影，
  但 event producer 不再为 event 二次投影整条拟合窗口或二次拟合；
- direct event fit 次数为零；
- censored event 的额外 fit 次数为零，因为复用 winning assignment fit；
- 不复制 raw data；
- 全量前先运行一个 development airport，并记录 wall time 与 event 分层数量。

## 15. 数据迁移

新 event 是可再生成派生数据，不做 schema migration branch。顺序为：

1. 运行 focused tests；
2. 对一个 development airport 执行 `--reclassify-existing`，不下载；
3. 重建 arrivals、observed records、evaluation 与 CZML；
4. 审计 `within_observed_support/right_censored/invalid_support/unavailable` 数量；
5. 通过事务式 reclassify regression 锁定样本序列和 `flight_key` 算法；全机场
   批量迁移前另保存逐机场 pre-run hash 以做 artifact 级前后审计；
6. 用户确认后按机场串行执行全量重跑，并逐机场核验 source-only hash、event
   schema、evaluation report 与 frontend publication。

`--evaluate-only` 不升级旧 event；遇到 v7 必须失败并提示 reclassify。

## 16. 完成定义

只有同时满足以下条件才可把本文状态改为 implemented：

- 第 11 节小规模实验完成且不使用 evaluation gate；
- 第 13 节 focused tests 通过；
- 生产代码只剩 direct 与 censored 两个物理分支；
- evaluation 不 refit；
- 一个 development airport 无下载重生成成功；
- raw samples 与源身份锚点未改变；派生 `flight_key` 的变化被明确审计，下游
  roster/split 不得静默复用；
- event/report 中的 frame 与 evaluation policy provenance 可分别审计；
- 未触碰本任务外的现有 TS 工作树改动。

## 17. 实施记录

### 17.1 Source-only 实验（2026-08-15）

使用五个 development airports，每机场最多 100 条 source-valid bracket 航迹；未读取 TCH、FSD、verdict 或 pass rate。KRDU 也取得 100 条 bracket 航迹。第一轮 local challenger 因没有限制 selected pass 而出现公里级误差，该轮结果作废；修正为 contiguous inbound pass 后重新运行，以下只记录修正后的结果。

Direct leave-one-out 的最坏机场 source error：

| 方法 | evidence coverage | lateral abs P95 最坏值 | vertical abs P95 最坏值 |
|---|---|---:|---:|
| 两点 bracket | 所有可用 pseudo-events；每机场 388–400 个 | 3.68 m | 11.10 m |
| 300 m local robust line | 仅每机场 210–398 个 | 3.39 m | 10.12 m |

局部鲁棒方法在部分样本上降低了误差，但不能保持同等 coverage。二倍降采样时，两点方法每机场仍有 138–200 个结果；local challenger 在 KSMF、KMSY、KSTL 为 0，KRDU 为 2，KSJC 为 47。按照第 11 节“先 evidence coverage、再最坏机场误差、最后 complexity”的顺序，生产 direct estimator 决定为：

$$
\boxed{\operatorname{direct\_linear\_bracket}}
$$

不把 local challenger 加入生产代码。

Censored 人工截断以 direct event 为 source reference，固定复用 `[-5000,-300] m` winning robust fit。主要垂直结果为：

| 截断距离 | 最坏机场 median bias | 最坏机场 abs P95 | 备注 |
|---:|---:|---:|---|
| 300 m | −11.37 m（KRDU） | 35.93 m（KRDU） | 五机场均 100/100 可拟合 |
| 800 m | −12.37 m（KRDU） | 41.06 m（KRDU） | KSJC 99/100，其余 100/100 |
| 2,000 m | −12.23 m（KRDU） | 40.97 m（KRDU） | KSJC 98/100，其余 100/100 |

这些数值不参与 LPV gate 选择。它们说明 censored point 是明显弱于 direct 的模型推断，必须以 `right_censored/censored_robust_line` 标记且 uncertainty 保持 `uncalibrated`；但没有证据支持在生产路径加入另一个复杂 predictor。

实施前 v7 baseline artifact 的 censored 外推距离分布只从 event evidence 读取：

| 机场 | n | P50 | P95 | max |
|---|---:|---:|---:|---:|
| KSMF | 81 | 334.63 m | 574.24 m | 954.41 m |
| KMSY | 64 | 359.40 m | 562.53 m | 1,026.28 m |
| KRDU | 13,718 | 382.37 m | 762.77 m | 1,980.34 m |
| KSTL | 891 | 375.07 m | 639.97 m | 1,298.66 m |
| KSJC | 48 | 325.02 m | 443.65 m | 613.73 m |

人工截断验证已覆盖到 2,000 m，因此覆盖当前数据的最大实际外推距离。生产实现不新增数据集调出的硬距离 gate；如果未来数据超出 2,000 m，必须先扩展 source-only 验证，不能静默声称已有支持。

### 17.2 生产实现（2026-08-15）

实现收敛为一个严格 schema 和两个估计方法：

```text
runway-threshold-event-v1
  within_observed_support -> direct_linear_bracket
  right_censored          -> censored_robust_line
  invalid/unavailable     -> none
```

主要修改：

- `harvest/threshold_event.py` 删除 `_FitEnsemble`、三窗口候选、event-side refit、
  v7 method/version 和 metadata sidecar bracket lookup；
- `harvest/classify.py` 对 direct event 不再调用 fitter；winning runway 的
  integrity-failed bracket 不允许降级为 censored；
- `harvest/airports.py` 新增只包含物理 LTP frame 的 snapshot/fingerprint；
- `evaluation/thresholds.py` 独立生成包含 TCH/FSD/width 的 evaluation-context
  fingerprint；
- `evaluation/arrival.py` 严格消费新 schema，不导入或调用 fitter；
- CZML 对 direct 不添加 inferred tail，只对 right-censored event 添加；
- observed uncertainty 明确为 `uncalibrated`，report 不再输出零宽或伪 95% 区间；
- v7 只保留在历史设计文档中，不存在 runtime dual reader。

Focused 验证命令：

```bash
conda run -n aeroviz python -m pytest \
  final_approach/tests trajectory_data_process/harvest/tests evaluation/tests -q
```

结果：`173 passed in 1.32s`。其中新测试明确覆盖 direct 共享 $\alpha$、direct
零 fit、invalid support 不外推、censored 复用 `Assignment.fit`、物理与 policy
fingerprint 分离，以及 CZML/evaluation 不 refit。

### 17.3 KSMF 真实流水线验证（2026-08-15）

在剩余约 12 GB 空间、KSMF harvest 约 859 MB 的条件下运行：

```bash
conda run -n aeroviz python -m trajectory_data_process.harvest \
  --airport KSMF --reclassify-existing
```

命令未联网、未下载、未修改源 ADS-B；事务式重分类和 arrivals、observed
evaluation、CZML、frontend publication 全部成功，耗时约 1 分 48 秒。

新 event 分层（24,984 条 track）：

| observability | 数量 |
|---|---:|
| `within_observed_support` | 4,484 |
| `right_censored` | 4 |
| `invalid_support` | 79 |
| `unavailable` | 20,417 |

全部 24,984 条均为 `runway-threshold-event-v1`。评价的 4,231 条 LPV 记录中：

| verdict | 数量 |
|---|---:|
| pass | 3,837 |
| fail | 392 |
| indeterminate | 2 |

report 的 assessment contexts 全部具有 `evaluation_context_fingerprint`，event
methodology 标明 direct/censored 分支且无 evaluation refit；4,490 条 assigned
航迹成功生成 canonical CZML。

### 17.4 五机场全量迁移（2026-08-15）

在用户确认后，对剩余机场逐个串行运行同一命令，避免并行生成大型 CZML
耗尽磁盘：

```bash
conda run -n aeroviz python -m trajectory_data_process.harvest \
  --airport <AIRPORT> --reclassify-existing
```

全量结果如下。`direct`、`censored`、`invalid` 与 `unavailable` 四列覆盖全部
267,194 条 track；verdict 只统计进入 observed evaluation 的 42,746 条记录。

| 机场 | tracks | direct | censored | invalid | unavailable | pass | fail | indeterminate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| KSMF | 24,984 | 4,484 | 4 | 79 | 20,417 | 3,837 | 392 | 2 |
| KMSY | 19,581 | 4,097 | 52 | 12 | 15,420 | 1,418 | 2,731 | 1 |
| KSTL | 49,437 | 7,925 | 840 | 51 | 40,621 | 4,467 | 4,298 | 4 |
| KRDU | 66,942 | 760 | 15,295 | 15 | 50,872 | 11,648 | 2,790 | 1 |
| KSJC | 106,250 | 11,148 | 3 | 45 | 95,054 | 9,558 | 1,593 | 6 |
| **合计** | **267,194** | **28,414** | **16,194** | **202** | **222,384** | **30,928** | **11,804** | **14** |

全局结构审计确认：

- 267,194/267,194 条 event 均为 `runway-threshold-event-v1`；
- direct 只使用 `direct_linear_bracket`，censored 只使用
  `censored_robust_line`，其余记录的 method 为 `none`；
- 五份 harvest report 与 frontend `comparison/observed/evaluation_report.json`
  逐对象相同；
- 所有 assessment context 都有 `evaluation_context_fingerprint`；
- methodology 明确写出 observed event 由序列化 v1 event 提供且
  `no evaluation refit`；
- 五个机场的 CZML 均已重新发布。

为验证重分类没有修改 source payload，对运行前后每条记录的下列字段生成
canonical JSON hash，再对排序后的逐记录 hash 集合生成机场级 SHA-256：

```text
icao24, callsign, start_time_utc, duration_s, max_sample_gap_s,
altitude_source, altitude_datum, source_integrity,
reported_ground_speeds_m_s, samples
```

四个在全量运行前已保存基线的机场，前后 hash 完全一致：

| 机场 | source-only SHA-256（pre = post） |
|---|---|
| KMSY | `39ec6a638f0b272c013f60877d71106f2f214e5e1130e1735b399b3c82518ad9` |
| KSTL | `4620f9bce78c87e251c7ceccec3d036cb55b9a48873dc70f8fedfb76a8233a6e` |
| KRDU | `de80c665bf72f764adaf9babffbe6e7561009660955d62f739a3ff1fdc443980` |
| KSJC | `6486a6570d796b9104def820de50f8cd9342c073ba7e7165d0210684dd48d456` |

KSMF 在先行验证前没有保存同口径 artifact hash，不能事后补造 pre/post
相等结论；它通过了同一事务式 reclassify regression 和全 event schema 审计。

这里必须区分 source identity 与当前仓库的派生 `flight_key`：后者编码
runway/outcome，因此重分类结果变化时其值也会变化。KMSY、KSTL、KRDU、KSJC
运行前后的 flight-key 集合 hash 均发生变化；这不表示 raw track 被修改，但
意味着基于旧 arrivals manifest 的训练 roster、split、checkpoint 或 prediction
不得静默复用。后续数据驱动实验应从新 manifest 重新建立 development split；
本次没有运行或查看 outer-test。

迁移完成后重新运行 focused suite，结果为：`173 passed in 1.27s`。

### 17.5 Evaluation v4 policy-only 重算（2026-08-15）

事件估计器和 tracks 不变，仅用 `--evaluate-only --no-czml` 重新应用
`±22 m` RNAV terminal geometry 规则并发布 schema
`terminal-approach-evaluation-v4`：

| 机场 | total | pass | fail | indeterminate | pass rate |
|---|---:|---:|---:|---:|---:|
| KMSY | 4,150 | 3,892 | 257 | 1 | 93.78% |
| KRDU | 14,439 | 14,168 | 270 | 1 | 98.12% |
| KSJC | 11,157 | 11,144 | 7 | 6 | 99.88% |
| KSMF | 4,231 | 4,221 | 8 | 2 | 99.76% |
| KSTL | 8,769 | 8,485 | 280 | 4 | 96.76% |
| **合计** | **42,746** | **41,910** | **822** | **14** | **98.04%** |

KMSY 的 257 条 fail 全部来自 `direct_linear_bracket`，而不是 censored fit；
KRDU 的 270 条 fail 中有 195 条来自 censored fit。该分层说明 v4 标准修正与
event estimator 诊断是两个独立问题，不能再用一个机场的总 fail rate 推断
fitting 全局失效。

### 17.6 尚存边界

- censored 的 source-only 截断误差已经报告，但 numeric uncertainty 尚未校准，
  因此继续保持 `uncalibrated`；
- 五机场 event artifact 已全部迁移；未来新增机场或旧备份仍必须显式运行
  `--reclassify-existing`，strict reader 不提供旧 schema fallback；
- 本次没有读取或使用 outer-test，也没有运行训练、改变 `flight_key` 算法或
  evaluation gate；原始样本 schema 和 source-only payload 未被重写。
