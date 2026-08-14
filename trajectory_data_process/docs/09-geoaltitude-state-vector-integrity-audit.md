# ADS-B 几何高度与 OpenSky state-vector 完整性审计

审计日期：2026-08-14  
审计范围：KMSY、KRDU、KSJC、KSMF、KSTL 当前 harvest、ADS-B metadata sidecar、arrival manifest 和 threshold event  
审计性质：只读；没有删除、覆盖、重分类、重新评估或重新训练任何现有数据

## 1. 结论

本项目不需要因为使用 `geoaltitude` 而推翻重做。`geoaltitude` 是 ADS-B/GNSS 提供的 WGS-84 椭球高（HAE），正是本项目保存三维几何轨迹时应使用的高度基准。当前 HAE 到 MSL 的显式转换边界也是正确的。

但是，本次审计所针对的既有派生数据存在一个独立且真实的完整性问题：OpenSky 会在离开覆盖后继续保留 state vector 最多 300 秒，而旧 harvest 没有执行 OpenSky 官方建议的 `time - lastcontact <= 15` 过滤，也没有在重分类时恢复并使用 `lastposupdate`/`lastcontact`。因此，同一个最后已知位置和高度会被当成多个新的逐秒观测参与最终段拟合和模型输入。

五机场全量审计得到：

- 46,212 个已有 threshold event 中，8,865 个（19.183%）的拟合范围包含 stale state-vector；其中 8,670 个来自 KRDU。
- 44,439 条 model-ready arrival 中，8,148 条（18.335%）包含 stale 行。
- arrival 输入共 19,144,167 行，其中 1,828,475 行（9.551%）为本审计判定的 stale 行；KRDU 单独为 22.184%。
- stale event 行几乎全部是前一行完全相同的三维坐标：例如 KRDU 为 2,159,464/2,159,552。这是 held state，不是飞机真实悬停或高度测量噪声。
- 仅在内存中排除 stale 行后，46,212/46,212 个 event 的当前拟合范围仍满足现行最低点数（8）和空间跨度（500 m）。这证明真实最终下降数据仍然存在，不能把问题解释为“整条航迹没有可用垂直高度”。
- 更严格的 KRDU 连续性检查表明，清理会暴露少量真实接收空洞：按 15 秒断开时，16,638 个 event 中 16,631 个的最后连续块仍可拟合，16,637 个至少有一个可拟合连续块。其余少数应标记 unavailable，而不是跨空洞强行拟合。

因此，必须作废并重建的是受 stale/held 行影响的派生 arrivals、observed threshold events、observed evaluation，以及用旧 arrivals 训练出的开发模型。无需删除或重新下载现有 HAE 轨迹和 ADS-B metadata sidecar；它们保留了完成安全重建所需的信息。

该修复现已实现为 `--rebuild-fresh-from SOURCE --output STAGING`：源目录只读，目标机场必须不存在；输出采用 `harvest-tracks-v2-source-timing` 和 `harvest-arrivals-v4-source-timed-track-slices`。五机场迁移、差异审计和 canonical 切换已经完成，结果见 [10-source-timed-canonical-promotion-audit.md](10-source-timed-canonical-promotion-audit.md)。本文件前述统计描述的是修复前数据，不能误读为当前 canonical 结果。

## 2. 必须区分的三个问题

### 2.1 高度基准是否正确

正确。

FAA AC 20-165B §3.3.3.8 明确要求 ADS-B geometric altitude 基于 Height Above Ellipsoid（HAE），而不是 Height Above Geoid 或 MSL。OpenSky 当前 API 1.4.0 的 `state_vectors_data4` schema 也将 `geoaltitude` 定义为 geometric (GNSS) altitude，单位为米。

项目当前的数据边界为：

```text
ADS-B geoaltitude（HAE）
    -> harvest 原样保存 HAE
    -> runway/CIFP 提供同点 HAE-MSL offset
    -> modeling/evaluation 显式转换到 MSL
```

这条 datum 路径不应改为 barometric altitude。压力高度对应 29.92 inHg 标准基准，不能直接与跑道 MSL 高程或 CIFP TCH 相减；若要使用，必须另行引入随时间和地点匹配的 QNH/大气转换及其误差模型。

### 2.2 几何高度测量本身是否毫无意义

不是。

FAA 的 `Summary of the Investigation into the Use of Automatic Dependent Surveillance-Broadcast Data for Monitoring Aircraft Altimetry System Error` 使用独立、后处理的 GPS truth 对 ADS-B geometric height 做了实测比较：

- §2.1.2–2.1.3：UAT 和 1090ES 几何高度均以 25 ft（7.62 m）量化；
- Attachment A §3.3.7：再次确认两种 ADS-B 来源均为 25 ft 量化；
- Attachment A Table 2（§3.4.1–3.4.3）：8 个 WAAS level-flight segment 中，1090ES 对 truth 的平均差为 2.580 ft，UAT 为 4.303 ft；三组均值在 95% 置信水平下没有显著差异；
- Table 3（§3.4.6–3.4.7）：非 WAAS 1090ES 各段平均差为 1.560–14.134 ft，标准差为 6.579–9.758 ft；样本仍有限；
- §4.1–4.3：FAA 判断 UAT 足以用于 ASE 估计，1090ES 也可能足够，但非 WAAS 情况需要更多试飞。

这份研究说明 ADS-B 几何高度不是“非物理数据”或随机数。不过它是在 FL280/FL410 的专用试飞、level segment 和独立 truth 时间匹配条件下完成，不能直接证明商业航班在跑道入口处每个 OpenSky 样本都满足 ±7.5 m。它支持“数据源有物理意义”，不替代本项目的近地面完整性控制。

当前五机场数据也呈现清晰的物理信号。在先前固定 2,500 条航迹检查中，最终段拟合得到：

- glidepath 中位数 3.053°，P05/P95 为 2.748°/3.487°；
- 垂直残差 RMS 中位数 3.362 m，P95 为 5.017 m；
- 相邻高度相同率 55.09%。

前两项与稳定进近的物理几何一致。第三项主要反映 25 ft 量化，不能单独作为坏数据证据。

### 2.3 当前 OpenSky state-vector 时间序列是否可以原样当作逐秒观测

不可以。

OpenSky API 1.4.0 的 `State Vectors -> Timestamps & Data Retention` 明确说明：离开覆盖后 state vector 最多保留 300 秒，并给出排除 stale vector 的过滤条件：

```sql
AND time - lastcontact <= 15
```

同一文档定义：

- `time`：state vector 有效时刻；
- `lastposupdate`：最近一次记录到的位置更新时间；
- `lastcontact`：OpenSky 最近一次收到该飞机任何信号的时间；
- state vector 是从原始 Mode S/ADS-B 消息汇总出的 position、velocity 和 status 摘要，不是每列在 `time` 时刻都刚刚获得的新测量。

REST API 的 state-vector contract 还说明：如果过去 15 秒没有 position report，`time_position` 可以为空。历史 Trino 数据仍会保留 state vector，因此研究查询必须主动执行 freshness 过滤。

修复前的代码在构造 `Sample` 时保存了 `lastposupdate` 和 `lastcontact`，但只检查经纬度和 `geoaltitude` 非空，没有 freshness gate。重分类从存储 JSON 恢复时又只恢复 `[time, lon, lat, alt]`，把两个 freshness 字段留空；metadata lookup 只保护 threshold bracket，没有保护整个拟合或 arrival slice。最终段拟合因而会把 held rows 当作额外权重，arrival 则会把 held tail 当作真实 4D 运动时间。

## 3. 只读审计方法

### 3.1 数据源

每个机场只读取：

- `trajectory_data_process/outputs/harvest/<AIRPORT>/tracks/manifest.json`；
- manifest roster 明确引用的 assigned track JSON；
- `trajectory_data_process/outputs/harvest/<AIRPORT>/arrivals/manifest.json`；
- `trajectory_data_process/outputs/adsb-metadata/<AIRPORT>/manifest.json` 及其只读 Parquet partitions。

metadata 通过现有 `SidecarStateMetadata` 以精确 `(icao24, state-row time)` 键查找。没有 glob 未列入 manifest 的 track，也没有发起 OpenSky 网络查询。

### 3.2 本次 freshness 判定

一个轨迹行只有同时满足下列条件才计为 fresh：

```text
metadata exact match exists
time - lastcontact <= 15 s
time - lastposupdate <= 15 s
```

第一个 15 秒条件是 OpenSky 历史 API 明示的 stale-vector 过滤。第二个条件对位置轨迹更保守：即使收到了新的非位置消息，也不能把超过 15 秒的旧经纬度当成当前位置。

本次审计只把不满足条件的行从内存统计集合中排除，没有从磁盘删除它们。

### 3.3 event 可拟合检查

对每个当前 `status=estimated` event：

1. 读取其当前 `source_sample_range`；
2. 在内存中排除 stale 行；
3. 投影到当前已记录的 runway HAE frame；
4. 检查至少 8 个 fresh 点且 along-track span 至少 500 m。

该检查回答“真实数据是否仍足以支撑一个空间拟合”，不是重新发布的 event，也没有重新计算 verdict。

KRDU 另做时间连续性检查：fresh 点相邻 state-row time 相差超过 15 秒或 30 秒时分块，再分别检查最后块与任一块是否满足 8 点/500 m。该检查揭示必须显式处理的接收空洞。

### 3.4 小规模重分类实验

为判断 freshness 过滤是否会整体摧毁跑道分配，另取每个机场按 landing time 排序后居中的连续 100 条 assigned 航迹；仅在内存中保留 fresh sample，并调用当前 classifier：

- 500/500 条仍为 assigned；
- 500/500 条仍分配到原跑道；
- KRDU 新旧 threshold altitude 差的 P05/中位数/P95 为 -6.306/0.000/+3.362 m，最大绝对差 18.397 m；
- 其他四机场大部分完全不变，最大差仅 KMSY 0.085 m。

该实验说明问题不是“删掉 stale 后没有轨迹”，同时证明 held rows 确实能显著移动个别 KRDU 拟合。它不是全量重分类结果，不能替代安全 staging rebuild。

### 3.5 `lastposupdate` 与 `geoaltitude` 同步审计

为关闭原设计中的 open question，另取每机场按时间连续的 100 条轨迹，共
213,127 个已精确匹配 sidecar 的 state rows，比较相邻 state snapshot：

- 33,566 对相邻 snapshot 具有相同 `lastposupdate`；
- 经 freshness 双门限后，仍有 1,260 对在经纬度和 `lastposupdate` 不变时改变了
  `geoaltitude`；
- 这些变化全部伴随 `lastcontact` 前进；典型绝对高度步进为 7.6 m（25 ft）；
- 因此 `lastposupdate` 是水平位置的正确主时间，但不是可证明的独立
  `geoaltitude` 更新时间。OpenSky state-vector schema 不提供
  `lastgeoaltitudeupdate`。

在每机场 100 条 assigned 轨迹上，又比较两种明确的高度对齐选择：

| Airport | 可比 events | “held 组最后快照”减“最接近 lastposupdate 快照”的 threshold altitude P05/中位数/P95/max abs (m) | runway/assignment 改变 |
|---|---:|---:|---:|
| KMSY | 98 | -2.469 / 0.000 / +0.590 / 5.516 | 0 |
| KRDU | 96 | -5.891 / -0.743 / +0.540 / 10.675 | 0 |
| KSJC | 99 | 0.000 / 0.000 / 0.000 / 0.398 | 0 |
| KSMF | 88 | 0.000 / 0.000 / 0.000 / 1.027 | 0 |
| KSTL | 98 | -0.898 / 0.000 / 0.000 / 2.408 | 0 |

这个差异不能靠 verdict 标准消除；尤其 KRDU，随意选 held 组的最后高度可移动
拟合结果超过 7.5 m 垂直门限。因此新实现采用以下可审计规则：

1. 以 `lastposupdate` 作为水平位置样本时间；
2. 同一 `(icao24, lastposupdate)` 只产生一个三维样本；
3. 选择 state-row time 最接近该 `lastposupdate` 的 snapshot，避免把之后才到达的
   异步高度错误回填到较早水平位置；
4. 同组发生高度变化时增加 `geoaltitude_async_groups`，不把后续高度当作新的空间点；
5. 剩余的近地面几何高度测量/对齐误差继续作为 estimator uncertainty，而不是
   伪装成已知为零。

该规则不声称恢复了不存在的 message-level 高度时间戳；它选择与水平测量时刻
距离最近的可观测 snapshot，并把无法完全消除的不确定性保留下来。

## 4. 五机场全量结果

### 4.1 Threshold-event 拟合范围

| Airport | Estimated events | 含 stale 的 events | 比例 | Stale rows | 与前一行三维坐标完全相同 | Fresh 后满足 8 点/500 m |
|---|---:|---:|---:|---:|---:|---:|
| KMSY | 4,292 | 29 | 0.676% | 770 | 768 | 4,292 |
| KRDU | 16,638 | 8,670 | 52.110% | 2,159,552 | 2,159,464 | 16,638 |
| KSJC | 11,237 | 7 | 0.062% | 418 | 418 | 11,237 |
| KSMF | 4,706 | 11 | 0.234% | 194 | 194 | 4,706 |
| KSTL | 9,339 | 148 | 1.585% | 15,688 | 15,680 | 9,339 |
| **Total** | **46,212** | **8,865** | **19.183%** | **2,176,622** | **2,176,524** | **46,212** |

KRDU 占所有 stale event 的 97.80%。这不是所有机场共同的 `geoaltitude` 系统误差，而是 coverage/held-state 处理缺陷在 KRDU 数据中的集中暴露。

### 4.2 Model-ready arrivals

| Airport | Arrivals | 含 stale 的 arrivals | 比例 | Arrival rows | Stale rows | Stale-row 比例 |
|---|---:|---:|---:|---:|---:|---:|
| KMSY | 4,306 | 110 | 2.555% | 1,650,553 | 3,606 | 0.218% |
| KRDU | 15,009 | 7,611 | 50.710% | 8,113,720 | 1,799,951 | 22.184% |
| KSJC | 11,324 | 52 | 0.459% | 3,772,461 | 3,317 | 0.088% |
| KSMF | 4,455 | 36 | 0.808% | 1,829,288 | 1,763 | 0.096% |
| KSTL | 9,345 | 339 | 3.628% | 3,778,145 | 19,838 | 0.525% |
| **Total** | **44,439** | **8,148** | **18.335%** | **19,144,167** | **1,828,475** | **9.551%** |

KRDU arrival 的单航迹 stale 比例 P50/P95/max 为 0.587%/51.155%/74.026%，stale tail 长度 P50/P95/max 为 2/288/578 个 state rows。大量长尾会扭曲 flight duration、速度、时间归一化和序列损失；它不能只在 evaluation 端修补。

### 4.3 KRDU 清理后暴露的真实时间空洞

| 检查 | 数量 |
|---|---:|
| Events | 16,638 |
| Fresh event 范围内存在 >15 s gap | 9 |
| Fresh event 范围内存在 >30 s gap | 5 |
| 15 s 分块后最后连续块可拟合 | 16,631 |
| 15 s 分块后至少一个连续块可拟合 | 16,637 |
| 30 s 分块后最后连续块可拟合 | 16,633 |
| 30 s 分块后至少一个连续块可拟合 | 16,637 |
| Arrivals | 15,009 |
| Fresh arrival 内存在 >15 s gap | 53 |
| Fresh arrival 内存在 >30 s gap | 24 |
| Fresh arrival 内存在 >60 s gap | 7 |
| Fresh arrival 最大 gap P50/P95/max | 1/2/1,111 s |

这说明大多数 KRDU 问题是 terminal held tail；排除它之后不会在航迹中间留下大洞。但少数航迹存在真实 coverage discontinuity，必须分段或标记 unavailable，不能用普通插值跨越。

## 5. 对 bracket、fit 和 verdict 的含义

### 5.1 Bracket 仍然有用，但不是垂直 truth

通过 `lastposupdate`、reported ground speed 和跑道结构验证的 threshold bracket 可以可靠地回答：

- 飞机是否在该物理 pass 穿过 threshold plane；
- 哪条 runway 与这次 crossing 匹配；
- 哪两个源位置点夹住 crossing。

它不应单独提供最终垂直 verdict 的精确真值，原因是：

- 几何高度以 25 ft 量化；
- OpenSky state vector 没有独立的 `lastgeoaltitudeupdate` 字段；
- 两个 bracket 端点可以移动了位置但仍报告同一量化高度；
- state vector 是字段聚合，不保证所有字段同一时刻更新。

正确边界是：bracket 选择 runway/pass；fresh、连续的最终段估计 threshold crossing altitude；evaluation 只应用标准，不重新拟合。

### 5.2 不能把 GVA 类别直接当作实际误差

FAA AC 20-165B §3.3.3.9 规定 GVA 应来自位置源设计数据或合格 VFOM；没有合格 accuracy metric 时应广播 GVA=0。GVA 是声明的 95% accuracy category，不是当前样本的实际误差。

当前 operational-status sidecar 的可用 GVA 绝大多数解码为 `<=45 m` 类别。这个类别太粗，不能单独证明某个 crossing 满足 ±7.5 m；但也不能推导“实际误差就是 45 m”。因此：

- GVA 可作为来源资格与 uncertainty 上界的 metadata；
- 不应以 `GVA=45 m` 直接把全部轨迹判为不可用；
- 也不应在 GVA 不可用时假装每个观测都具有精确 ±7.5 m accuracy。

### 5.3 Ideal/predicted trajectory 不受 ADS-B source uncertainty 限制

evaluation 标准仍然可以对任意 ideal、optimized 或 predicted trajectory 的 threshold-crossing state 使用同一跑道/TCH/垂直 gate。ADS-B 的量化、coverage 和 freshness 只影响 observed reference 的估计质量，不能被写进通用 verdict 标准本身。

## 6. 对项目有效性的判断

### 6.1 保留且仍成立

- 使用 `geoaltitude`/HAE 表示三维几何轨迹；
- 使用 CIFP 同点数据完成 HAE/MSL datum conversion；
- runway frame、TCH benchmark 和理想/预测轨迹的 evaluation 定义；
- 已下载的 track HAE 数值和 ADS-B metadata sidecar；
- 横向几何和大多数非时间敏感的可视化能力。

### 6.2 当前不能作为最终论文结果

- 当前 observed threshold event 与 observed pass/fail 统计；
- 当前 arrival duration 和含 held tail 的 4D 序列；
- 使用当前 arrivals 训练、选参或比较得到的开发模型结果；
- 尤其是包含 KRDU 数据的模型结论。

这些结果应标记 provisional。修复后需要重新生成派生数据并重新训练开发模型。根据仓库的 ML Experiment Isolation 规则，不得因为本次修复去查看或复用 outer-test 结果；只能使用 train/validation/CV 开发，直到用户明确冻结并请求新的 final release。

### 6.3 不需要做的事

- 不需要删除现有 outputs；
- 不需要因为这一问题重新下载全部 OpenSky history；
- 不需要把 canonical altitude 改成 barometric altitude；
- 不需要让 evaluation 自己重新拟合 ADS-B；
- 不应把现有数据原地覆盖后假装与旧实验同一版本。

## 7. 已实施的安全修复边界

建议的数据流为：

```text
现有 track HAE + exact metadata sidecar（均只读）
    -> 恢复每个 sample 的 lastcontact / lastposupdate / reported velocity
    -> freshness gate
    -> held-state 去重或源时间重建
    -> 显式 coverage-gap 分段
    -> runway assignment + 一次 final-segment fit
    -> 新版本 tracks/arrivals/observed events
    -> 新 observed evaluation
    -> train/validation 模型重训
```

实现满足以下安全要求：

1. 对 source roots 在运行前后做 roster、size、mtime 或 SHA-256 审计，保证零修改。
2. 输出只能进入新的 sibling staging/version directory。
3. 所有 metadata join 必须是 exact `(icao24, state-row time)`；冲突或缺失不得猜测。
4. 明确区分 state-row time、`lastcontact` 和 `lastposupdate`，不得用 held row 的新 `time` 制造低速移动。
5. 对 freshness 过滤后暴露的 gap 设置独立的 segment-integrity policy；超过阈值不得插值或跨 pass 拟合。
6. fit 只能消费一个 fresh、连续的 inbound pass；evaluation 只消费序列化 event。
7. 记录每个排除原因和每个机场的 denominator，不能只报告 surviving tracks。
8. 在发布前比较旧/新 runway assignment、arrival roster、fit 参数和 verdict confusion matrix。
9. 在用户明确批准前，不删除、移动或覆盖任何现有 track、sidecar、arrival、evaluation 或模型文件。

迁移命令：

```bash
conda run -n aeroviz python -m trajectory_data_process.harvest \
  --airport KRDU \
  --rebuild-fresh-from /path/to/legacy-harvest \
  --output /path/to/new-source-timed-staging
```

当前 canonical 五机场数据已经迁移；以上命令只用于其他 legacy harvest，不能把
已经 source-timed 的 canonical root 再作为 migration source。

实现细节：

- sidecar 以默认 512 条轨迹为批次做向量化 exact join，避免给 6 小时分区内每个
  airport state row 建 Python 对象字典；
- KMSY 512 条真实轨迹、224,269 个查询键的小实验中，读取轨迹 0.426 s、sidecar
  join 0.808 s、清理 0.545 s；峰值 RSS 约 435 MB；
- 该批次保留 504 条，8 条因最终 fresh block 少于两个点进入显式 exclusion；
  14,983 个 held rows 被移除，2,277 个异步高度组被审计；
- 源 manifest 和所有 rostered record 的 relative path、size、mtime_ns 被散列，
  处理前后不一致时 staging 不提交；
- 写入前要求可用空间至少为 rostered source-track bytes 的 3 倍再加 2 GiB 保留量；
- 新 staging 模式自动跳过 frontend CZML/publication，避免未经审核的数据替换当前展示；
- evaluation 不做 refit，只消费 classification 阶段从 clean track 产生的序列化 event。

## 8. 官方参考资料与本地快照

### 8.1 OpenSky Network API 1.4.0

- 在线：<https://openskynetwork.github.io/opensky-api/trino.html>
- 关键章节：`State Vectors -> Schema`、`Timestamps & Data Retention`、`The Other Tables`
- 本地快照：[OpenSky_API_1.4.0_Trino_Historical_Data.html](references/OpenSky_API_1.4.0_Trino_Historical_Data.html)
- SHA-256：`cf13016c8c9f7d607dcae4bee958de5fed5133491b3e0f25bd2f5d14fe9f958c`

### 8.2 FAA AC 20-165B

- 标题：*Airworthiness Approval of Automatic Dependent Surveillance-Broadcast OUT Systems*
- 在线：<https://www.faa.gov/documentlibrary/media/advisory_circular/ac_20-165b.pdf>
- 关键章节：§3.3.3.8 `Geometric Altitude`；§3.3.3.9 `Geometric Vertical Accuracy (GVA)`；Appendix B §B.4.9 `Geometric Altitude`
- 本地副本：[FAA_AC_20-165B.pdf](references/FAA_AC_20-165B.pdf)
- SHA-256：`6a79b03244f4a38f0334250b52178aab63eab6981bc9c76f118a3938cfc5b34e`
- 版本核验：FAA `ADS-B Quick Links` 页面在 2026-07-08 更新后仍将 AC 20-165B 列为 ADS-B Out airworthiness approval 文档：<https://www.faa.gov/air_traffic/technology/adsb/quicklinks>

### 8.3 FAA ISPACG/23 WP-02

- 标题：*Summary of the Investigation into the Use of Automatic Dependent Surveillance-Broadcast Data for Monitoring Aircraft Altimetry System Error*
- 发布/会议：US FAA，ISPACG/23，2009-03-26/27
- 在线：<https://www.faa.gov/media/100441>
- 关键章节：§2.1–2.2；Attachment A §3.2.2、§3.3.7、§3.4/Table 2、Table 3、§4 Conclusion
- 本地副本：[FAA_ADSB_Altimetry_System_Error_Investigation_ISPACG23_WP02.pdf](references/FAA_ADSB_Altimetry_System_Error_Investigation_ISPACG23_WP02.pdf)
- SHA-256：`3887e8d032763e1f5fe55dc30982cbdff718c445d2f24cdf281b91a084d5ebfe`
- 限制：这是一份 FAA 官方试飞研究，不是现行设备标准；它支持数据源可用性判断，但不能替代近地面 threshold-event 的数据质量控制。

## 9. 本地代码证据

- `trajectory_data_process/harvest/tracks.py::source_timed_final_block`：执行 freshness、held 去重、异步高度审计和 15 s coverage 分块。
- `trajectory_data_process/harvest/adsb_metadata.py::lookup_many`：按分区批量 exact join，并拒绝冲突 duplicate。
- `trajectory_data_process/harvest/freshness_rebuild.py::rebuild_fresh_tracks`：只读源、批量处理、源指纹复核和新目录提交。
- `trajectory_data_process/harvest/reclassify.py::_source_timed_track`：旧的原子 reclassification 模式也必须先恢复 source timing。
- `trajectory_data_process/harvest/arrivals.py::write_arrival_records`：只接受完整的 `harvest-tracks-v2-source-timing` manifest。
- `evaluation/arrival.py`：继续只消费序列化 threshold event，不调用 final-segment fitter。

这进一步说明修复必须发生在轨迹派生/分类边界，而不是在 evaluation 中再次拟合或事后猜测。
