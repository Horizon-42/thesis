# 数据下载、生成与处理流程审计

本文记录 2026-07 对 ADS-B 下载、观察轨迹生成、场景构建、优化、预测、评估和
前端发布链路的完整审计，以及当前采用的“同一事实只保留一个物理锚点”契约。
它描述当前可执行代码；早期实验文档中的 `*_czml_input_*.json`、逐跑道 CZML
和旧版 `arrivals/records/` 不再是数据入口。

## 1. 当前端到端数据血缘

```text
OpenSky history
  │  分时段下载；中断时仅用 checkpoint/history.sqlite 暂存
  ▼
tracks/manifest.json + tracks/<outcome>/<flight_key>.json
  │  唯一的实测 HAE 样本锚点；manifest 是唯一 roster
  ├──────────────────────────────────────────────────────┐
  │                                                      │
  ▼                                                      ▼
arrivals/manifest.json                         approach/records/*_eval.json
  只保存 source_file + 样本切片索引               HAE→MSL、速度拟合、阈值外推
  不再复制截断后的 waypoint 数组                   属于新的派生事实，不等同于原样本
  │                                                      │
  ├─► flight_scenarios/outputs/*_scenarios.json           ├─► evaluation_report.json
  │       两套准备数据：fitted_adsb / runway              └─► trajectories.czml
  │                                                           全机场唯一观察 CZML
  ├─► optimizer
  │     4dTrajectory/outputs/<ICAO>/
  │       shared_references/fitted_adsb/
  │       shared_references/runway/     ← runway 与 runway_cons 共用
  │       <category>/*_states.json      ← optimizer + simulator 状态唯一锚点
  │       <category>/*_eval.json        ← states_ref 指向上述 simulator_states
  │
  └─► ts_transformer
        ts_pred_*/*_states.json         ← predicted + observed 状态唯一锚点
        ts_pred_*/*_eval.json           ← states_ref: predicted_states
        ts_pred_*/references/*.json     ← states_ref: observed_states 的锚点切片

summary / evaluation report
  └─► comparison/<category>/*.czml
        只存 optimizer/simulator/prediction 路径
        referenceSource=canonicalObserved
        CZML/report 使用不可变 generation 文件名，index 最后原子提交
        前端从同一 trajectories.czml 选择观察参考，不再复制
```

## 2. 物理文件的唯一职责

| 数据事实 | 唯一物理锚点 | 其他消费者如何引用 |
|---|---|---|
| OpenSky 实测 HAE 样本 | `tracks/.../<flight_key>.json` | `tracks/manifest.json` roster；arrival 用文件名和样本索引 |
| 模型可用 arrival 集合 | `arrivals/manifest.json` | schema `harvest-arrivals-v3-track-slices`；文件名、SHA-256 和样本索引锚定 track 切片 |
| 跑道 TCH、下滑角和 datum | arrival manifest 顶层 `runway_targets` | 每条记录只保存 runway key |
| 观察评估状态 | `approach/records/*_eval.json` | 流式进入 `evaluate_batch`；这是包含 MSL 和拟合速度的新派生量 |
| 前端观察轨迹 | `<airport>/trajectories.czml` | `landings/index.json` 的各 runway 均指向该文件，前端内存过滤 |
| 优化参考轨迹 | `shared_references/<target>/` | 类别 eval 的 `reference_file`；v2 manifest 用输入签名、flight identity 和逐文件 SHA-256 决定是否复用 |
| 优化计划与仿真回放 | 每航班一个 `*_states.json` | `*_eval.json.states_ref` 指向 `simulator_states` |
| TS 预测与观察序列 | 每航班一个 `*_states.json` | prediction/reference eval 分别引用不同 key 和 slice |
| 比较视图的观察 reference | 同一 `trajectories.czml` | index 的 `referenceSource: canonicalObserved` 和 flight key；旧 observed schema 明确拒绝并要求重新 preparation |
| 前端 comparison 批次 | generation-suffixed CZML/report | `comparison_index.json` 是唯一提交点，提交后才清理旧 generation |

“唯一锚点”不表示所有派生格式都消失。HAE 实测点、MSL 拟合状态、优化器节点、
动力学仿真状态和 CZML entity 是不同语义或不同消费格式，不能用硬链接假装为同一
数据。当前删除的是内容与语义都相同的整数组复制。

## 3. 审计中发现的问题与处理

### 3.1 重复存储

改造前对本机现有数据抽样：

- 整个 `outputs/harvest` 约 7.0 GB。
- KRDU：`tracks` 739 MB，`arrivals` 217 MB，`approach` 约 1.5 GB。
- KRDU 的 `approach/_czml_input` 约 661 MB，只是渲染中间输入。
- KRDU 的全机场 `trajectories.czml` 约 1.1 GB；`landings/` 中逐跑道文件的并集
  又约 1.1 GB，内容是同一批 entity 的物理重复。
- KMSY 三个优化类别过去各复制约 166 MB reference；`runway` 与
  `runway_cons` 的 reference 哈希完全相同。
- KMSY 各 comparison 类别再次嵌入观察 reference，comparison 目录合计约
  946 MB。
- optimizer/TS 的 eval JSON 过去还会复制 states JSON 中已经存在的状态数组。

处理结果：

1. arrival v3 只保留 track slice，不写 `arrivals/records/`。
   每条 slice 同时记录 canonical track 的 SHA-256；源文件被替换后会要求重建，
   而不是把旧索引悄悄套到新样本上。
2. CZML 生成使用临时 JSONL 流，且每机场只写一次 `trajectories.czml`。
3. `landings/index.json` 只充当 runway 选择索引。
4. `runway` 与 `runway_cons` 共用 `shared_references/runway`，并用源文件哈希缓存。
5. eval/reference records 使用 `states_ref`，不再复制状态数组。
6. comparison 类别不再嵌入观察路径，前端复用 canonical observed datasource。
7. 大型逐航班 JSON 改用紧凑编码；manifest/report 继续保持可读缩进。
8. canonical observed 和每个 comparison CZML 先写同目录临时文件，再原子替换，
   所以不会发布半截 JSON。Observed 失败时保留上一份 canonical 文件；comparison
   目前仍是逐文件发布而非整个 category 的原子事务，批次中断风险列在 review 指南。

### 3.2 重复计算和峰值内存

原 `state_samples_from_track` 对每个样本重新扫描整个窗口，复杂度接近
`O(n²)`。在 KRDU 最长的 2734 点轨迹上，本机基准从 0.154 s 降至 0.022 s，
约 7 倍；结果的速度值逐点一致。

观察重建还做过以下调整：

- 评估 records 逐文件 yield，`evaluate_batch` 单次遍历；不再一次性持有 KRDU
  数百 MB 的 records。
- CZML 逐 entity 写出，并由 harvest 直接传入已知最大时间偏移；不再构造一个
  完整航班数组，也不再为求 clock 范围重复解析临时 JSONL。
- comparison batch 不再读取和解析约 1.1 GB 的观察 CZML；只有浏览器显示时加载
  那一个 canonical 文件。
- CIFP airport load 在一次 parser pass 中同时收集 Path Point 与 runway geometry，
  不再为一个机场读取两遍整份分发文件。

### 3.3 并行策略

- 优化器是 CPU/求解器密集型，保留 `--jobs` 进程并行，并把每个 worker 的
  BLAS/OpenMP 线程限制为 1，避免 `N 个进程 × N 个线程` 过度订阅。
- OpenSky 下载受远端 Trino 配额、查询缓存和 checkpoint 顺序约束。当前按机场、
  时间块串行最安全；并行下载容易增加限流和恢复状态复杂度，因此没有盲目并行。
- 观察重建优化后主要是许多 JSON 文件的顺序读写。对同一机械盘/SSD 启动更多
  进程可能只会争用 I/O；当前先通过线性算法、流式读写和取消重复生成提速。
- 多机场可由上层独立调度，但应根据磁盘吞吐控制并发，不在单机场内部重复解析
  同一文件。

## 4. 下载与 checkpoint 的合理性

下载器按固定 UTC 时间块向过去扫描。每个块写入
`checkpoint/history.sqlite`，并只重算该块影响到的 aircraft；中断后由
`checkpoint/state.json` 和 SQLite 继续。完整 harvest 成功写出
`tracks/manifest.json` 后 checkpoint 被删除。

这意味着：

- checkpoint 是可恢复的临时下载数据库，不是第二份长期数据源；
- `tracks/manifest.json` 是成功批次的提交点；
- `--evaluate-only` 永远不访问 OpenSky，只从该 manifest 重建派生物；
- 不用 JSON glob 猜测批次内容，旧 orphan 文件不会进入 roster；
- `--no-cache` 只应在明确需要绕过 pyopensky 查询缓存时使用。

下载循环目前仍会在新时间块补入同一 aircraft 后重新分类该 aircraft，并在最终
提交前完整分类一次。这保证跨 chunk 的连续航班不会被错误截断，是有意的正确性
成本；如果将来继续优化，应基于增量轨迹边界做 profile，再决定是否值得增加状态
复杂度。

## 5. 无 LPV Path Point 的跑道

跑道 geometry 和 LPV vertical target 是两件事。KRDU 14/32、KSMF 35R 等跑道
可能存在可用阈值几何，却没有符合筛选条件的 FAA CIFP LPV Path Point。

当前契约是：

- geometry fallback 保留，以便实测轨迹仍能分配到正确跑道；
- 这些航班留在完整 `tracks` harvest；
- 因为没有 published TCH/glidepath，它们以 `no_published_tch` 等原因从
  model-ready arrivals 排除；
- 不伪造 TCH，也不让整个机场因为一个非 LPV 跑道而失败。

## 6. 重建、迁移与旧数据

代码不会在安装或测试时主动删除现有研究结果。旧目录中的重复文件会一直存在，
直到对应机场/类别被重新生成。

需要完整清理派生链时先预览：

```bash
conda run -n aeroviz python clean_pipeline_data.py --airport KRDU --dry-run
```

机场范围是强制参数：重复使用 `--airport ICAO` 可选多个机场；只有确实要清理
所有机场时才使用 `--all-airports`。清理器只选择 producer 明确拥有且可重建的
输出：`harvest/*/{arrivals,approach}`、scenario JSON、三个 optimizer 类别及其
shared references、带有效 `summary.json` 且明确为 `split: "val"` 的 standalone
TS prediction、canonical observed CZML/landings，以及不含 experiment/test 发布物的
comparison tree。

清理器不会提供删除 source 或研究结果的扩大范围选项。下载得到的 canonical
`harvest/*/tracks`、checkpoint/history、`test_release.json`、formal experiment、
parked/manual/未知输出、混有 experiment 或 final-test 的 comparison tree、git tracked
文件、静态机场数据和 `data/archive` 一律保留。实际删除前，所有目标先移动到同一
文件系统的 staging 目录；如果 staging 过程失败，已移动文件会回滚。

先重建观察与场景输入：

```bash
conda run -n aeroviz python prepare_scenario_inputs.py --airport KRDU
```

这会：

- 从现有 `tracks/manifest.json` 生成 arrival v3；
- 删除旧 `arrivals/records/`；
- 流式重建 approach evaluation；
- 写一个 canonical `trajectories.czml`；
- 删除旧 `landings/*.czml` 和 `approach/_czml_input`；
- 生成 fitted-ADS-B 与 runway 两套 scenario JSON。

再运行优化和发布：

```bash
conda run -n aeroviz python run_scenario_optimization.py \
  --airport KRDU --jobs 14 --fitting-type trapezoidal
```

首次新格式运行会创建 shared references、states-ref eval records 和不含观察副本的
comparison 文件。`runway` 与 `runway_cons` 在源签名相同时复用同一 reference
集合。若已有旧 optimization，不能用 `--skip-optimize` 期待它自动转成新格式；
需要正常重跑相应类别。

TS 流程：

```bash
conda run -n aeroviz python run_ts_pipeline.py --airport KRDU
```

已有 checkpoint 可继续复用；要把旧 prediction 输出迁移为 states-ref 格式，至少
重新执行 predict（可用 `--skip-train`），再发布 evaluation/comparison。

## 7. 仍然存在但合理的存储成本

- `tracks/not_landing` 可能很大，但它维持完整 harvest 的分母和可审计性；不能在
  不改变研究样本定义的情况下静默删除。若只需模型结果，可用归档工具压缩整个批次。
- `approach/records` 包含 MSL、拟合速度和外推所需状态，是可复现 observed verdict
  的派生证据；它不是 HAE track 的字节重复。
- `trajectories.czml` 对大型机场仍可能超过 1 GB。这是 Cesium 当前消费契约的成本。
  现在只有一份；未来若浏览器首屏加载成为瓶颈，可设计按时间/空间分页或二进制
  传输，但不应重新引入互相重叠的逐跑道完整副本。
- scenario JSON 只有每航班的边界条件和 aircraft/aero 参数，不复制完整 track；
  fitted-ADS-B 和 runway 的 target 不同，因此两份 scenario 是不同实验输入。

## 8. 验证不变量

相关测试固定以下契约：

- arrival manifest 必须引用存在于 source manifest 的同一 `flight_key` 和文件；
- slice 索引必须落在 canonical track 范围内；
- observed CZML 只有一个物理文件，runway 属性可供前端过滤；
- 切换 runway 不触发第二次 CZML fetch/load；
- comparison 的 logical `ref-*` 可以没有物理 reference packet；
- `states_ref` 只能引用合法文件、key 和 slice，加载后仍经过完整 record 校验；
- shared reference 的源签名未变时不重写；
- 流式 `evaluate_batch` 保持输入顺序和原报告 schema。

项目 Python 命令和测试统一优先使用 conda `aeroviz`；该约定也已写入仓库根目录
`AGENTS.md`。
