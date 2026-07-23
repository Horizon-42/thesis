# 当前数据与场景流水线修改 Review 指南

本文用于 review 当前工作区中围绕 ADS-B harvest、arrival 场景生成、优化/预测评估、
CZML 发布和脚本拆分的一组关联修改。建议按本文顺序 review，不要按 `git diff` 的文件
字母顺序逐个阅读；这些修改跨越多个模块，先确认数据契约，再看实现细节更容易发现
真正的错误。

配套的最终数据契约见
[`data-pipeline-canonical-storage.zh.md`](data-pipeline-canonical-storage.zh.md)。

## 1. Review 目标

本次修改应同时满足以下要求：

1. 无 LPV Path Point 的跑道仍保留 threshold geometry，能够参与实测轨迹跑道分配；
   但因为没有 published TCH/glidepath，不进入 model-ready arrivals。
2. fitted-ADS-B target 的位置和 `V/psi/gamma` 都来自同一条 final-approach 拟合，
   不再用 threshold 处的零速度或不相关的最后样本速度。
3. 原 `run_scenario_pipeline.py` 被拆成“准备输入”和“运行优化”两个明确阶段，且
   旧脚本被删除、调用路径和文档得到更新。
4. preparation 有包含阶段名称和 elapsed time 的可见进度输出。
5. 同一实测或计算数组只有一个物理锚点；manifest、切片和 `states_ref` 不得指向
   模糊或可能被替换的数据。
6. 取消重复计算和重复文件时，不能改变轨迹数值、跑道归属、flight identity、
   datum 或评估报告语义。
7. 可重新生成的旧派生格式默认不兼容：loader 必须明确拒绝并要求重建，不能加入
   dual-read、旧文件 fallback 或“尽力读取”。只有 OpenSky 下载结果等不可再生成的
   source data 才允许有最小范围的数据保护。
8. 所有 Python review 命令优先使用 conda `aeroviz`。

如果其中任一项不成立，应视为阻塞性问题。

本指南中的“迁移”统一指通过当前 producer 重新生成派生数据，不表示新版 consumer
需要读取旧 schema。遇到兼容性、数据损失、架构或迁移风险，reviewer 应要求先由用户
决定，不能自行增加兼容层。

### 当前工作区状态（2026-07-23）

- legacy observed manifest 已改为严格拒绝，前端不再回退旧分跑道 CZML；
- arrival v1/v2、reference cache v1 和不完整 optimization batch 均通过重建解决，
  不作为兼容输入；
- comparison generation 原子发布、fitted datum、`states_ref` 边界、source identity
  和 reference SHA-256 修复已有测试覆盖；
- comparison 前端只接受完整 `comparison-v2-generation` index；embedded `ref-*`
  和固定名 report fallback 已删除，batch producer 也拒绝缺少 evaluation report
  的发布；
- 尚未执行第 12 节的真实 KRDU 大型派生数据重建。

### 30 分钟快速路径

时间有限时，按下面顺序即可覆盖最高风险部分：

1. `git diff --check`，确认没有 generated artifacts 进入 diff。
2. 阅读两个新 runner，确认 preparation 与 optimization 边界。
3. 阅读 `airports.py`、`cifp.py`、`arrivals.py`，确认无 LPV runway 的处理。
4. 阅读 `fitted_approach.py`、`build.py`，确认 fitted position/velocity 和 datum。
5. 阅读 `evaluation/records.py`、optimizer/TS export，确认 `states_ref`。
6. 阅读 observed/comparison CZML builder 与两个前端 hooks，确认只加载一个
   canonical observed 文件。
7. 验证 comparison batch 原子提交、`--skip-optimize` 完整校验和
   shared-reference cache SHA-256 校验已经落实。
8. 搜索 derived-data 的 `legacy`/`fallback` 分支，确认旧 observed、arrival、
   comparison 和 reference schema 都是明确拒绝而非兼容读取。
9. 运行第 11 节测试；真实 KRDU 派生数据重建放到代码 review 通过之后。

## 2. 开始前：确认 diff 边界

先在仓库根目录执行：

```bash
git status --short
git diff --name-status
git diff --stat
git diff --check
```

关注点：

- `run_scenario_pipeline.py` 应显示为删除；
- `prepare_scenario_inputs.py`、`run_scenario_optimization.py`、本指南以及新增测试
  应显示为新文件；
- `git diff --check` 应无输出；
- 不应把 `trajectory_data_process/outputs/`、`4dTrajectory/outputs/`、
  `aeroviz-4d/dist/` 或机场大型 CZML 加入源代码 diff。

当前工作区是一组连续需求的组合修改，不适合只看总 diff。建议用下面各节提供的
分组命令。

注意：未跟踪的新文件不会出现在普通 `git diff -- <path>` 中。Review 两个新 runner
时应直接打开完整文件，或分别执行：

```bash
git diff --no-index /dev/null prepare_scenario_inputs.py
git diff --no-index /dev/null run_scenario_optimization.py
```

这两条命令在显示差异时退出码为 1，是 `--no-index` 的正常语义。

## 3. 第一轮：脚本边界与 CLI 行为

先 review：

```bash
git diff -- \
  run_scenario_pipeline.py \
  prepare_scenario_inputs.py \
  run_scenario_optimization.py \
  run_ts_pipeline.py \
  trajectory_data_process/tests/test_scenario_pipeline.py
```

### 3.1 `prepare_scenario_inputs.py`

预期职责仅有：

```text
stored tracks
  → harvest --evaluate-only
  → arrivals manifest + observed evaluation/CZML
  → fitted_adsb scenario JSON
  → runway scenario JSON
```

检查：

- 不导入或调用 optimizer。
- 省略 `--target-type` 时只准备两套数据：
  `fitted-adsb` 和 `runway`。
- `runway_cons` 不单独生成第三份 scenario，因为它与 `runway` 共用 threshold
  target 数据。
- `--skip-observed` 只跳过 observed rebuild，不跳过 scenario build。
- 无 `tracks/manifest.json` 时明确跳过/报错，不能通过 glob 猜输入。
- `_run_command_with_progress` 正确传播子进程退出码和 Ctrl-C。
- 进度输出包含具体阶段和 elapsed time。

只查看命令解析，不运行：

```bash
conda run -n aeroviz python prepare_scenario_inputs.py \
  --airport KRDU --dry-run
```

预期能看到一次 observed rebuild 和两次 scenario build。

### 3.2 `run_scenario_optimization.py`

预期职责从“已经准备好的 scenario JSON”开始：

```text
scenario JSON
  → optimizer
  → evaluation JSON/HTML
  → comparison CZML
```

检查：

- 不调用 harvest 或 `flight_scenarios`。
- 未指定 `--target-type` 时运行：
  `fitted_adsb`、`runway`、`runway_cons`。
- `--with-constraint` 只能与明确的 `--target-type` 一起使用。
- `--jobs`、`--fitting-type`、mesh 参数正确透传给 optimizer。
- `runway` 和 `runway_cons` 都传入
  `../shared_references/runway`；fitted 使用独立的
  `../shared_references/fitted_adsb`。
- `--skip-optimize` 只能复用当前契约下完整且一致的结果。
  `optimization_reuse_error()` 应验证
  summary 计数和 roster、每条 eval、solved states + `states_ref`、所有 reference
  文件及其 v2 manifest identity/SHA-256；准备输入仍在时还要核对输入签名。仅有一个
  遗留 summary、不完整批次或 v1 reference cache 都必须回到重新优化，不能用兼容
  分支修补后复用。
- 同一 category 用另一种 fitting 重跑会覆盖前一批，CLI 和文档都明确说明这一点。

Dry run：

```bash
conda run -n aeroviz python run_scenario_optimization.py \
  --airport KRDU --jobs 14 --fitting-type trapezoidal --dry-run
```

预期显示三个 mode；命令中不应再出现 `run_scenario_pipeline.py`。

## 4. 第二轮：CIFP、跑道 fallback 和 model-ready 排除

Review：

```bash
git diff -- \
  trajectory_data_process/harvest/cifp.py \
  trajectory_data_process/harvest/airports.py \
  trajectory_data_process/harvest/arrivals.py \
  trajectory_data_process/harvest/tests/test_cifp.py \
  trajectory_data_process/harvest/tests/test_arrivals.py \
  trajectory_data_process/harvest/tests/test_classify.py
```

### 4.1 CIFP parser

检查：

- 一个 airport load 只读取/拆分 CIFP 一次。
- Path Point `001` 有效但缺匹配 `002` continuation 时，结构错误必须传播；
  broad parse-error handler 不能将其吞掉。
- 单条无法解析但非结构性的数据是否跳过，应由现有容错契约决定。
- Path Point 和 runway-primary geometry 从同一 pass 得到，不要重新读约 51 MB 文件。

### 4.2 跑道构造

必须区分两类数据：

- runway geometry：位置、航向、elevation，可来自 runway-primary fallback；
- LPV vertical target：TCH 和 glidepath，只能来自合格 Path Point。

KRDU 14/32、KSMF 35R 这类无 LPV Path Point 的 configured runway：

- `load_airport` 不应失败；
- 应能参与 `classify_track`；
- `threshold_crossing_height_m` 和 `published_glidepath_deg` 应为 `None`；
- `write_arrival_records` 应将相应航班计入 `no_published_tch` 或
  `no_published_glidepath`，而不是伪造值。

不要接受以下“修复”：

- 为无 LPV 跑道填默认 15 m TCH 或 3° glidepath；
- 直接从 airport runway roster 删除这些跑道；
- 因一个跑道无 LPV 而让整个机场 load 失败；
- 让无 TCH 航班进入 optimizer/TS。

## 5. 第三轮：fitted target 的位置与速度语义

Review：

```bash
git diff -- \
  flight_scenarios/build.py \
  flight_scenarios/fitted_approach.py \
  flight_scenarios/start_state.py \
  flight_scenarios/datum.py \
  flight_scenarios/tests/test_build.py \
  flight_scenarios/tests/test_fitted_approach.py \
  flight_scenarios/tests/test_start_state.py
```

### 5.1 选项错误顺序

`target_from_threshold=True` 与 `target_from_fitted_adsb=True` 的互斥检查必须发生在：

- trajectory fitting 之前；
- datum conversion 之前；
- aircraft lookup 之前。

无效调用应稳定抛出 documented option error，不能先出现 fitting/datum 异常。

### 5.2 拟合 target

检查 fitted-ADS-B target：

- lat/lon/alt 是 final-approach fit 外推到 threshold crossing 的结果；
- `V/psi/gamma` 来自同一个 fitted segment 的运动学拟合；
- HAE→MSL 只应用一次；
- target mass 与 scenario initial mass 一致；
- 没有可用 final approach 时抛出带 flight/runway 的明确错误；
- 不允许通过 `V=0` 绕过问题。

这里最重要的数值 review 是确认“位置拟合”和“速度拟合”使用一致的样本范围，而不是
只确认字段非零。

### 5.3 `O(n)` velocity window

`state_samples_from_track` 的双指针窗口必须与旧 `_window_around` 完全等价：

- 左边界是 `t - window/2`，排除更早点；
- 右边界包含 `t + window/2`；
- 不足两点时使用相邻两点 fallback；
- 输入顺序和输出时间基准不变。

现有最长 KRDU 轨迹（2734 点）的本机对比约为 0.154 s → 0.022 s，约 7 倍；
review 重点是数值一致，而不是只看速度。

## 6. 第四轮：canonical tracks 与 arrival v3

Review：

```bash
git diff -- \
  trajectory_data_process/harvest/store.py \
  trajectory_data_process/harvest/arrivals.py \
  4dTrajectory/ts_transformer/dataset.py \
  trajectory_data_process/harvest/tests/test_arrivals.py \
  4dTrajectory/ts_transformer/tests/test_ts_transformer.py
```

核心 schema：

```text
harvest-arrivals-v3-track-slices
```

每条 arrival row 应包含：

- `flight_key`
- `source_file`
- `source_sha256`
- inclusive `first_sample_index` / `last_sample_index`
- runway 和必要的 arrival metadata

顶层 `runway_targets` 每条跑道只保存一次 target。

Loader 必须验证：

1. schema 是 v3；
2. `source_manifest` 存在；
3. `flight_key` 在 source manifest 中存在且不重复；
4. `source_file` 与 source manifest 一致；
5. 文件 SHA-256 与 arrival row 一致；
6. slice 不越界；
7. runway target 字段完整且 HAE/MSL/geoid 恒等式成立；
8. slice 后重新以第一点为 `t=0`；
9. flight identity round-trip 不变。

阻塞性错误：

- 继续写 `arrivals/records/*.json`；
- source track 改变后 loader 仍接受旧 slice；
- 旧 v1/v2 manifest 被“尽力兼容”并悄悄进入训练；
- 对 source records 使用 glob fallback。

旧 manifest 被拒绝是有意迁移边界，正确处理方式是重新运行 preparation，而不是
在 loader 中加入模糊 fallback。

## 7. 第五轮：observed evaluation 与 CZML

Review：

```bash
git diff -- \
  trajectory_data_process/harvest/observed.py \
  trajectory_data_process/harvest/__main__.py \
  evaluation/metrics.py \
  aeroviz-4d/python/generate_czml.py \
  trajectory_data_process/harvest/czml.py \
  trajectory_data_process/harvest/tests/test_czml.py \
  aeroviz-4d/python/tests/test_generate_czml.py
```

### 7.1 流式 evaluation

检查：

- `iter_observed_records` 每次只 materialize 一个 record；
- `evaluate_batch` 接受任意 `Iterable`，只消费一次；
- report row 顺序仍与 roster 一致；
- reference comparison、subject、solved/measured/successful 统计没有改变；
- generator 输入不能被第二次迭代。

### 7.2 唯一 observed CZML

一个机场只应发布：

```text
airports/<ICAO>/trajectories.czml
airports/<ICAO>/landings/index.json
```

检查：

- 每个 CZML entity 有 `properties.runway`；
- `landings/index.json` 的所有 runway entry 都指向 `trajectories.czml`；
- 不再生成 `landings/<ICAO>_<RWY>.czml`；
- 不再永久保存 `approach/_czml_input`；
- JSONL 是临时文件，退出后自动删除；
- max time offset 由 harvest 已知值传给 generator，不再次解析 JSONL；
- duplicate entity id 会失败；
- 最终 CZML 通过同目录临时文件原子替换，失败时旧文件保持完整；
- 大型 CZML 使用 compact JSON，不再使用 `indent=2`。

注意：`approach/records` 不是 track 的重复副本。它增加了 MSL datum、拟合速度和
threshold extrapolation，是新的派生语义，应保留以复现 observed verdict。

## 8. 第六轮：optimizer/TS states 锚定

Review：

```bash
git diff -- \
  evaluation/records.py \
  evaluation/tests/test_evaluation.py \
  4dTrajectory/optimization/scenario_optimization.py \
  4dTrajectory/optimization/tests/test_scenario_optimization.py \
  4dTrajectory/ts_transformer/export.py \
  4dTrajectory/ts_transformer/tests/test_ts_transformer.py
```

### 8.1 `states_ref`

Optimizer solved record：

```json
{
  "states": [],
  "states_ref": {
    "file": "<flight>_states.json",
    "key": "simulator_states"
  }
}
```

TS prediction/reference 应分别引用：

- `predicted_states`
- `observed_states`，并用 `start_index` 对齐 forecast anchor

检查：

- `load_record` 先解析 `states_ref`，再走原来的完整 record validator；
- 非法 file/key/slice 明确失败；
- `final_time_s`、initial state、controls alignment 等验证没有绕过；
- archive/restore 后相对路径仍成立；
- states JSON 是唯一数组锚点，eval JSON 不再复制同一数组。

### 8.2 shared optimizer references

检查：

- `fitted_adsb` 和 `runway` target 不同，因此保留两套 reference record metadata；
- `runway` 与 `runway_cons` 输入完全相同，应共享
  `shared_references/runway`；
- cache signature 至少锚定 scenario 文件和 arrival manifest；
- arrival manifest 中的每航班 SHA-256 又锚定具体 canonical track；
- roster、signature、identity、逐文件 SHA-256 或任一预期文件变化时必须重写；
- 从 per-category reference 迁移后只删除已知旧生成文件，不递归删除不明目录。

完整性负向测试应把 reference 替换为另一份结构合法 JSON，确认 cache miss 并重写；
`evaluation.load_reference` 还应独立核对优化记录与 reference 的
`id/icao24/landing_time_utc/runway`，防止绕过 cache 的错误指针进入指标。

## 9. 第七轮：comparison 与前端 datasource 复用

Review：

```bash
git diff -- \
  aeroviz-4d/python/build_scenario_comparison_czml.py \
  aeroviz-4d/python/tests/test_build_scenario_comparison_czml.py \
  aeroviz-4d/src/App.tsx \
  aeroviz-4d/src/data/airportData.ts \
  aeroviz-4d/src/data/observedTracks.ts \
  aeroviz-4d/src/data/observedTracks.test.ts \
  aeroviz-4d/src/hooks/useCzmlLoader.ts \
  aeroviz-4d/src/hooks/useComparisonTrajectoryLayer.ts \
  aeroviz-4d/src/hooks/useLandingsManifest.ts \
  aeroviz-4d/src/hooks/useObservedVerdictColors.ts \
  aeroviz-4d/src/hooks/useFlightOptimizerData.ts \
  aeroviz-4d/src/hooks/__tests__/useLandingsManifest.test.ts \
  aeroviz-4d/src/components/OptimizationSummary.tsx
```

### 9.1 Builder

新 batch index 必须包含：

```json
{
  "schemaVersion": "comparison-v2-generation",
  "generation": "<immutable generation id>",
  "evaluationReport": "evaluation_report_<generation>.json",
  "referenceSource": "canonicalObserved"
}
```

检查：

- category CZML 只包含 `opt-`、`sim-`、`pred-`、`look-` 等结果 entity；
- index 中仍保留 logical `ref-<flight_key>`，使 failed-only group 仍是可选择的完整
  roster 项；它不是可加载的 comparison entity；
- CZML 物理文件中不应再有 `ref-*` packet；
- failed-only group 即使没有结果 entity，也仍可通过 canonical observed reference 显示；
- 旧 index 缺少 `schemaVersion`、`generation`、`evaluationReport` 或
  `referenceSource` 时必须明确拒绝并要求重新发布；不得退回 embedded reference
  或固定名 `evaluation_report.json`；
- comparison CZML 使用紧凑 JSON，并且单个文件通过临时文件原子写出；
- CZML 与 report 文件名含同一个 generation；index 的 `generation`、
  `evaluationReport` 和各 group 的 `czml` 必须只指向该代文件；
- `comparison_index.json` 是批次唯一提交点：所有代际文件先完成，index 最后原子替换；
- 任一 runway 中途失败时，旧 index、旧 CZML、旧 report 均仍存在且内容不变；
- 只有新 index 提交成功后，才清理不再被它引用的旧代文件。

建议用 `test_batch_failure_preserves_the_previous_committed_generation` 注入第二条跑道
写入失败，确认失败清理只影响尚未发布的新 generation。单文件原子性仍需保留，但
不能替代上述批次提交协议。

### 9.2 Frontend

检查：

- `isLandingsManifest` 只接受 `observed-landings-v2-canonical`，并验证每条 runway
  entry 都指向 manifest 的同一个 `combined` 文件；
- manifest 仍在 loading、缺失、无效或属于旧 schema 时不猜测 CZML 文件名；
- legacy observed manifest 显示明确重建命令，且不读取旧分跑道文件；
- Observe mode 对每机场只 fetch/load 一次 `trajectories.czml`；
- 切换 runway 只更新 entity visibility 和 flight list，不重新下载约 1.1 GB 文件；
- comparison mode 使用同一个 `trajectoryDataSource` 显示 reference；
- comparison index 必须通过当前 schema 校验，observed verdict report 只读取 index
  指向的 immutable `evaluationReport`；不得回退固定文件名；
- solved/off-target/failed reference 颜色保持白/暗黄/红；
- reference checkbox 可即时隐藏 canonical observed reference；
- 切换 category/runway/mode 时旧 selection 被清理；
- 退出 comparison 后普通 observed filter、颜色、label 和 model 状态恢复；
- canonical observed reference 仍可 hover/click 显示 label；
- comparison clock 同时覆盖 observed 和结果轨迹的可用时间。

重点查看 hook 的 effect dependency 和 cleanup。这里最容易出现“第一次正确，切机场或
第二次打开后错误”的生命周期 bug。

按当前项目原则，以下代码形态本身就是阻塞项，即使现有旧数据能因此继续显示：

- 可选的 comparison `schemaVersion` / `referenceSource` / `evaluationReport`；
- `index.referenceSource !== "canonicalObserved"` 时改读 embedded `ref-*`；
- comparison index 404 后尝试固定名 `evaluation_report.json`；
- observed manifest 无 schema 时改读 `landings/<ICAO>_<RWY>.czml`。

## 10. 第八轮：清理、归档、文档与环境

Review：

```bash
git diff -- \
  clean_pipeline_data.py \
  archive_pipeline_data.py \
  trajectory_data_process/README.md \
  4dTrajectory/ts_transformer/README.md \
  docs/data-pipeline-canonical-storage.zh.md \
  AGENTS.md
```

检查：

- clean script 将 `trajectories.czml` 描述为唯一 observed 文件；
- default clean 删除 preparation 派生的 `harvest/*/{arrivals,approach}`，但保留
  下载的 `harvest/*/tracks`；`--include-downloads` 只扩展到 measured tracks；
- archive 覆盖 `4dTrajectory/outputs`，因此 sibling `shared_references` 不会漏掉；
- 文档不再宣称当前流程生成逐跑道 observed CZML 或 arrival record copies；
- 根目录 `AGENTS.md` 明确优先使用 conda `aeroviz`；
- 根目录 `AGENTS.md` 明确派生数据 backward compatibility 为 opt-in：默认淘汰旧
  schema 并重建，只有不可再生成 source data 才考虑最小保护；
- 有实际兼容、数据损失、架构或迁移风险时，agent 必须先询问用户；
- 命令示例使用 `conda run -n aeroviz ...` 或该环境的明确解释器路径；
- 历史 dev log 中出现旧 `aviation` 或旧 layout 不等于当前契约，当前 README 和
  根 `AGENTS.md` 才是执行依据。

## 11. 自动化验证

### 11.1 后端完整相关回归

在仓库根目录执行：

```bash
conda run -n aeroviz python -m pytest \
  flight_scenarios/tests \
  trajectory_data_process/harvest/tests \
  trajectory_data_process/tests \
  evaluation/tests \
  4dTrajectory/optimization/tests/test_scenario_optimization.py \
  4dTrajectory/ts_transformer/tests/test_ts_transformer.py \
  aeroviz-4d/python/tests/test_generate_czml.py \
  aeroviz-4d/python/tests/test_build_scenario_comparison_czml.py \
  aeroviz-4d/python/tests/test_run_asd_b_pipeline.py \
  -q
```

2026-07-23 当前验证结果：`328 passed, 1 warning`。warning 是已有的 PFAF
intercept-angle warning；不应有失败。

### 11.2 前端

```bash
cd aeroviz-4d
npx vitest run
npm run build
```

当前预期：

- 76 test files / 478 tests passed；
- production build 成功；
- 允许已有的 React `act(...)` test warning、严格拒绝 legacy manifest 的测试错误
  日志，以及 Vite chunk-size warning。

这些 warning 不是本次数据改造引入的阻塞项，但如果数量或位置改变，应重新确认。

## 12. 可选：真实 KRDU 派生数据重建验收

这一节会替换大型可重新生成的派生文件，只有确认当前没有
preparation/optimization 进程同时运行时才执行。它不重新下载或修改 canonical
OpenSky tracks。

### 12.1 重建 observed/scenario 输入

```bash
conda run -n aeroviz python prepare_scenario_inputs.py --airport KRDU
```

完成后检查：

```bash
find trajectory_data_process/outputs/harvest/KRDU/arrivals \
  -maxdepth 2 -type f -print

find aeroviz-4d/public/data/airports/KRDU/landings \
  -maxdepth 1 -name '*.czml' -print

test ! -d trajectory_data_process/outputs/harvest/KRDU/approach/_czml_input

jq '{schema_version, counts, first: .records[0]}' \
  trajectory_data_process/outputs/harvest/KRDU/arrivals/manifest.json

jq '{schemaVersion, combined, runways}' \
  aeroviz-4d/public/data/airports/KRDU/landings/index.json
```

预期：

- arrivals 下只有 manifest，不再有 `records/`；
- `landings/` 下没有 `.czml`；
- `_czml_input` 不存在；
- schema 为 `harvest-arrivals-v3-track-slices`；
- runway entries 全部指向 `trajectories.czml`；
- KRDU 的 `no_published_tch` 数量仍明确存在，而不是变成 load failure。

### 12.2 重建 optimizer/comparison

完整优化可能耗时很长。可以先选择一个 mode：

```bash
conda run -n aeroviz python run_scenario_optimization.py \
  --airport KRDU \
  --target-type runway \
  --jobs 14 \
  --fitting-type trapezoidal
```

完成后抽查：

```bash
find 4dTrajectory/outputs/KRDU/shared_references -maxdepth 2 -type f | head

jq '{states_length: (.states | length), states_ref, reference_file}' \
  "$(find 4dTrajectory/outputs/KRDU/runway -maxdepth 1 -name '*_eval.json' | head -1)"

jq '.referenceSource' \
  aeroviz-4d/public/data/airports/KRDU/comparison/runway/comparison_index.json

rg -l '"id":"ref-' \
  aeroviz-4d/public/data/airports/KRDU/comparison/runway/*.czml
```

预期：

- solved eval 的 `states_length` 为 0，`states_ref.key` 为 `simulator_states`；
- reference path 指向 sibling shared directory；
- index 输出 `canonicalObserved`；
- 最后一条 `rg` 无输出，因为物理 comparison CZML 不再嵌入 reference packet。

### 12.3 存储对比

重建前后分别记录：

```bash
du -sh trajectory_data_process/outputs/harvest/KRDU/{tracks,arrivals,approach}
du -sh aeroviz-4d/public/data/airports/KRDU/{trajectories.czml,landings,comparison}
du -sh 4dTrajectory/outputs/KRDU
```

旧数据抽样中，KRDU 明确可消除的重复约为：

- arrival copies：217 MB；
- persistent `_czml_input`：661 MB；
- per-runway observed CZML：约 1.1 GB；
- 另有 compact JSON、states_ref 和 comparison reference 去重带来的额外节省。

Review 时不要仅以 `du` 变小作为正确性证据；必须同时核对 flight count、runway
count、manifest roster、evaluation totals 和 frontend selection。

## 13. 建议的验收矩阵

| 场景 | 应通过 | 应明确失败/排除 |
|---|---|---|
| 有 LPV Path Point 的跑道 | assignment、arrival、scenario、evaluation | — |
| 有 geometry、无 LPV 的跑道 | assignment、observed CZML | model-ready arrival |
| Path Point 001 缺 002 continuation | — | airport/CIFP load，结构错误可见 |
| 同 callsign、同 runway、不同日期 | 两个不同 `flight_key` | entity merge |
| source track 在 arrival manifest 后被修改 | — | SHA-256 mismatch |
| threshold 与 fitted target 同时开启 | — | 在 fitting/datum 前 option error |
| canonical-v2 切换 observed runway | visibility/list 更新 | 第二次 fetch/load |
| legacy observed manifest | — | 前端明确报错并要求重新运行 preparation；不得回退到旧分跑道文件 |
| legacy comparison index / fixed report | — | 前端明确拒绝并要求重新发布；不得读取 embedded reference 或固定 report |
| comparison failed flight | canonical reference 红色可见 | 空 group 消失 |
| CZML duplicate identity | 保留上一份有效文件 | 发布半截 JSON |
| comparison batch 中途失败 | 旧完整 category 或明确不可用状态 | 旧 index 指向已删除/半成品文件 |
| 旧 arrival v1/v2 | 通过 preparation 重建 | loader 隐式兼容 |
| 旧 reference cache v1 / 无 hash | 重新优化生成 v2 cache | `--skip-optimize` 复用 |
| canonical downloaded tracks | 原文件复用且不被改写 | 为迁移派生 schema 而重新下载或覆盖 source data |

## 14. 最终 Review 结论模板

可以按下面格式记录结论：

```text
范围：
- reviewed script split / CIFP fallback / fitted kinematics /
  canonical storage / optimizer+TS refs / frontend reuse

阻塞问题：
- 无，或逐条列出文件、行号、复现条件和错误后果

非阻塞建议：
- 只列不改变本次数据契约的后续优化

验证：
- Python: 328 passed, 1 expected warning
- Frontend: 478 passed
- Build: passed
- KRDU derived-data rebuild: not run / passed（附 counts 与 du）

格式替换：
- existing tracks: reusable
- old derived schemas: rejected, no compatibility reader
- arrivals: must rebuild
- optimization eval/comparison: must rerun to obtain corrected fitted datum,
  states_ref、v2 hashed references 和 generation publication
```

只有在自动化测试通过，并且第 13 节中与本次修改相关的负向场景也确实失败时，才建议
批准当前修改。
