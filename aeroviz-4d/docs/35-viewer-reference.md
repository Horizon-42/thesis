# aeroviz-4d — viewer contracts and gotchas (reference)

Full text behind the index lines of `aeroviz-4d/CLAUDE.md`, moved here VERBATIM on 2026-09-16 so that
file stays a short index (it is injected into every session that touches the tree). Only the
`### <ID>` headings and the dated **Correction / Note** paragraphs are new; every heading has
exactly one index line in that CLAUDE.md ending in its ID (`grep -n '^### # ·' aeroviz-4d/docs/35-viewer-reference.md`).

**Maintenance: when a fact here changes, edit it HERE and keep its index line true; a new fact
gets a new ID here and ONE new line in the index.**

## Utility modules

### AV1 · `EVALUATION_REPORT_SCHEMA_VERSION` mirror and the older report classes

- `src/data/evaluationReport.ts` exports `EVALUATION_REPORT_SCHEMA_VERSION` as a declared
  MUST-match mirror of `evaluation.metrics.REPORT_SCHEMA_VERSION`; fixtures import it, never
  restate it (see `evaluation/CLAUDE.md`). The reader ALSO accepts the enumerated
  `LEGACY_EVALUATION_REPORT_SCHEMA_VERSIONS` (currently v5, pre-speed-gate): the published
  reports on disk are v5 and their optimizer record batches are gone until the batch rerun,
  so the report window shows them behind an explicit legacy banner rather than refusing —
  a bare version bump therefore no longer blanks Details, but v6-only fields must stay
  optional in the TS types until every published report is regenerated.

**Correction (2026-09-16):** "the published reports on disk are v5" is out of date. Reading the schema version of the first 400 `*report*.json` files (≤ 4 levels) under `public/data/airports` gave v5 67, v6 7, v7 74, v9 124, and the reader has TWO older classes: `LEGACY_EVALUATION_REPORT_SCHEMA_VERSIONS` (v5, the pre-speed-gate banner, `isLegacyEvaluationReport`) and `PRIOR_SPEED_GATE_REPORT_SCHEMA_VERSIONS` (v6–v8, `isPriorSpeedGateReport`). The optimizer record batches are back on disk (root `CLAUDE.md` Open Items: 70,267 records whose reports still need regenerating).

### AV2 · `categoryResultSource` is the one result-source classifier

- `utils/trajectoryResultSources.ts` → `categoryResultSource` is the ONE classifier
  splitting comparison categories into `optimization | prediction | experiment` (Observe's
  "Result source" selector and `EvaluationSummary`'s presentation both key off it).
  Optimizer publishes never stamp `resultSource` — absent field + non-`ts_` key ⇒
  optimization; the `ts_` prefix is the legacy marker for pre-`resultSource` data-driven
  publishes. Don't re-derive this split locally.

### AV3 · `ExperimentPicker` + `ExperimentDetails`

- **Experiments picker = `ExperimentPicker`** (a trigger in the panel that opens a portalled
  two-pane browser: campaigns as collapsible headings WITH their question, runs WITH their
  intent, a filter over names/intent/parameters, the hovered run previewed) **+
  `ExperimentDetails`** (intent + every parameter as a named row by section; also the panel's
  compact card). They read the publisher-stamped `experiment.runName / variantLabel / intent /
  parameters` — optional in the types (an unstamped publish falls back to the flat `label` and
  says "No intent recorded"), but SHAPE-checked when present, so a malformed one empties the
  airport's picker like any other field; `npm run check-publication` names
  `experiment.intent` / `experiment.parameters[i]`. Grouping/filtering:
  `trajectoryResultSources.experimentGroups` / `experimentMatches`.

## Gotchas (recurring, verified)

### AV4 · Vite must never watch `public/data`

- **Vite must never watch `public/data`** (`vite.config.ts` → `server.watch.ignored`). chokidar
  takes ONE inotify watch per file and that tree is ~40k files (39,307 local-terrain `.f32`
  heightmap tiles), so a single dev server ate 41,260 of the system's 65,536
  `fs.inotify.max_user_watches`. Two at once — e.g. a forgotten `nohup npm run dev` from an
  earlier session still holding the port — blew the limit and vite died on boot with
  `ENOSPC: System limit for number of file watchers reached`, which the supervisor then
  restart-looped (frontend dying at ~2 s, backend healthy the whole time). With the ignore rule:
  **363 watches**, 113× less. Nothing under `data/` is a build input (git-ignored generated
  output, fetched over HTTP at runtime), so watching it only ever bought the crash.
  Symptom to recognise: `Port 5173 is in use, trying another one…` plus a restart streak means a
  stale dev server is alive — `ss -ltnp | grep 5173`, and count a pid's watches via
  `/proc/<pid>/fdinfo/<fd>`.

### AV5 · a RUNNING dev server never sees a newly published category

- **…and the price of that ignore: a RUNNING dev server never sees a newly published category.**
  Vite caches the public-directory file list and refreshes it from the watcher — exactly what
  `server.watch.ignored` switches off for this tree — so a comparison directory created after the
  server booted 404s and the SPA fallback returns `index.html`. The frontend reports it as
  *"Expected JSON from …/comparison_index.json, but received HTML. The airport data file is
  probably missing"* — about a file that is on disk and readable. `categories.json` keeps working
  (it existed at boot; only its CONTENT changed), so the picker lists a category it cannot load,
  and that combination IS the signature. Fix: restart the frontend dev server after publishing
  NEW categories; `curl -o /dev/null -w '%{content_type}\n' <index url>` says which side you are
  on (`application/json` = served, `text/html` = fallback). Overwriting files that already
  existed at boot needs no restart. Verified 2026-09-07 on the anytime publication.
  **Restart = kill the `vite` NODE process, not the `npm run dev` wrapper**: under
  `start_aeroviz_fullstack.sh` killing npm alone leaves the node grandchild holding 5173 and
  still serving the stale list, the supervisor relaunches npm, and the new vite silently binds
  5175 — the app looks restarted while the old process still answers on 5173 (2026-09-07,
  three attempts). `ss -ltnp | grep 517` shows which pid owns which port.

### AV6 · an EMPTY picker on every airport is the manifest validator

- **An EMPTY picker on every airport after a publication is the manifest validator, not the
  server.** `airportData.ts::EXPERIMENT_PREDICTION_OUTPUTS` mirrors the package's
  `config.PREDICTION_OUTPUTS`; the publisher writes `experiment.predictionOutput` straight from
  the run's config; `isComparisonCategoriesManifest` is `.every(isComparisonCategory)`, so ONE
  category with an unlisted output empties the whole airport's list. Signature: every file
  answers 200 `application/json`, a restart changes nothing, and the console says
  `comparison categories for <ICAO> is not a valid manifest`. It happened with `closure`
  (2026-09-07) and again with `plan` (2026-09-12). The mirror is now pinned by
  `4dTrajectory/ts_transformer/tests/test_frontend_mirrors.py` — adding an output to the package
  fails the ts suite until this list learns it — and `npm run check-publication` (above) is the
  check to run at a milestone or when results are explicitly published: it answers "picker
  loads" / "picker BROKEN: [category] field value" instead of a bare HTTP 200.

### AV7 · `.flight-ops-panel` has `backdrop-filter`

- **`.flight-ops-panel` has `backdrop-filter`** → it becomes the containing block for
  `position:fixed` descendants AND clips overflow; floating windows must render via React portal
  into `document.body`.

### AV8 · aircraft CZML uses `forwardExtrapolationType: HOLD`

- **All aircraft CZML sets `forwardExtrapolationType:"HOLD"`** — `position.getValue` returns the
  frozen final position forever forward; any outward time-walk must stop when the position stops
  changing, not only on null.

### AV9 · Cesium `Clock.tick()` LOOP_STOP preserves overshoot

- **Cesium `Clock.tick()` LOOP_STOP wrap preserves overshoot**
  (`currentTime = startTime + (currentTime − stopTime)`); use `clock.onStop` (fires at the stop
  time for both CLAMPED and LOOP_STOP) for exact end-of-playback emits — never elapsed-based
  heuristics.

### AV10 · CZML clock intervals use `iso_ms`

- **CZML document clock intervals must use `iso_ms`** — second-precision `iso()` truncation made
  the clock stop up to 1 s before the last sample (~25–75 m phantom position error).

### AV11 · comparison-overlay entities are time-windowed

- **Comparison-overlay entities are TIME-WINDOWED — an empty scene is not a broken overlay.**
  Each group only shows inside its own availability interval, so at a clock time outside it the
  map is legitimately blank. Pause inside a window before diagnosing (≈08:09 UTC on the KRDU data).

### AV12 · the comparison reference uses the `arrival` track window

- **The comparison reference must be requested on the `arrival` track window**, not the full
  track — see `aeroviz_backend/CLAUDE.md` (median 5055 m of apparent "model error" otherwise).

## Comparison CZML colour contract

### AV13 · group status and the verdict-colour repaint skip

Group status lives on entity `properties.status` ∈ solved/offTarget/failed. Reference: white /
dark-red (failed) / dark-amber `OFF_TARGET_REF_COLOR` (off-target); simulator/result path bakes
bright yellow `OFF_TARGET_COLOR` (255,205,40) + "(off target)" name; optimizer plan keeps legend
orange/cyan. **The frontend repaint skip is keyed on "a verdict colour was baked", NOT on
`status` alone** — reference always, plus off-target optimizer/simulator paths.

### AV14 · `states_schema` dispatch and the `look-` entity

`build_scenario_comparison_czml.states_schema` dispatches on the record keys
(`optimizer_states`/`simulator_states` → `opt-` + `sim-` entities; `predicted_states` +
`observed_states` → `pred-` (purple `PREDICTION_COLOR`, kind `predicted`) **plus `look-`**
(same RGB at alpha 85 `LOOKBACK_COLOR`, kind `lookback`) — see the anchor-shift gotcha in
`4dTrajectory/ts_transformer/CLAUDE.md`).

### AV15 · predictions never get the off-target bake

**Predictions never get the off-target bake** (`mark_off_target = off_target and schema ==
"optimizer"`): a forecast essentially always misses the 106.75 m gate, so marking it repainted
27/27 groups yellow and the kind colour was never visible. Their `status` stays accurate and they
ARE repainted from the legend — so `PREDICTION_COLOR` and the TS legend entry are not required to
agree.

### AV16 · `look-` takes its forecast's verdict colour, faded

**`look-` takes its forecast's verdict colour, faded — never a hue of its own.** The frontend
paints BOTH prediction halves from the group status (pass green / fail red / indeterminate gray);
the input window is separated from the forecast by `COMPARISON_KIND_ALPHA.lookback` (85/255)
alone. The purple `COMPARISON_KIND_COLORS.predicted`/`.lookback` is only the no-verdict fallback.
A distinct input hue reads as a third kind of result rather than as the first half of one track,
which is why this is a contract and not a preference. The builder still bakes purple into both —
that divergence is a known open item (see the README's "Future Improvements").

## Comparison CZML is split by group count, not by runway alone

### AV17 · one CZML per runway stops working at thesis scale

- **One CZML per runway stops working at thesis scale.** At 38–54 KB of CZML per flight a
  2,000-flight runway is a single ~100 MB file, and the viewer `JSON.parse`s a whole file to
  show even one sampled group (the existing KRDU 23R prediction CZML is already 153 MB).
  `build_scenario_comparison_czml.py --max-groups-per-czml N` splits each runway into
  `comparison_<ICAO>_<RW>_pNNN_<generation>.czml`.

### AV18 · the split is transparent to the frontend

- **The split is transparent to the frontend by construction**: every
  `comparison_index.json` group record carries its own `czml` field and
  `selectComparisonGroups` derives the file list from those (`[...new Set(groups.map(g =>
  g.czml))]`). `prune_unreferenced_outputs` keeps files by the same set, so chunking needs no
  change on either side.

### AV19 · Training 只读一份词表：`instruction-v3`

（2026-09-24 改写：词表从 `instruction-v2` 换成 `instruction-v3`，航向词按步读。）

`src/data/trainingSample.ts` 钉住并逐项核对：样本格式名 `aeroviz-training-sample-v7`（`instruction-v3` 的形状：
航向词是它被判的那几行上的航向带和逐行判定，截获转弯是它的起止行和判定；没有转弯区、平行四边形、保持漏斗、拆分的
几份和补出来的切入航向）、读法 `TRAINING_READING_RULE = "instruction-v3"`、规格 sha 全文 `TRAINING_SPEC_SHA256`、
六列的名字和顺序 `TRAINING_COLUMNS`（跑道、进近、航向、高度、下降角、速度，与 `instructions/words.COLUMNS` 相同，
列是按位置读的）、"不变" `TRAINING_UNCHANGED = -1`、标注器会写的发令原因 `TRAINING_WORD_KINDS`（`initial`、
`per-step`、`clear`、`target`、`step`、`angle`、`unspecified`；v2 的 `turn`、`turn-split`、`intercept` 按名字拒读）。
对不上就整份拒读并报出是哪一项，不做兼容分支。Python 一侧的对应常量（`instructions/training_files.py` 的
`SAMPLE_SCHEMA`、`INDEX_SCHEMA`、`KIND_READBACK`、`WORD_KINDS`，以及 `spec.READING_RULE`、`words.COLUMNS`、
`words.UNCHANGED`）由 `tests/test_instruction_training_export.py` 与 TypeScript 源码逐字比对。
（2026-09-25 起，Training 文件的结构、各导出器与后端共用的检查都在 `instructions/training_files.py`：不是 runner、不依赖 torch、
只抛 `ValueError`；四个导出器和后端的实时执行器都从这里导入。索引与叠加清单在写入前逐个机场确认"仍是本次运行开头读到的样子"，
任何一个被别的导出改过就一个都不写。）

**规格 sha 是 `145d6911e75b`**（2026-09-25）：`instruction-v3` 按运行日划分后在新训练集上重新测量的规格
（`instruction_language/v4_20260924`），现在的执行器代码只飞这份产物。按航班划分的 `v3_20260924`（`0b4ea75be36d`）上
导出的 `instruction_v3` 集合和它上面的叠加层因为规格 sha 不对被按名字拒读（`check-publication` 报出）；现行集合是
`instruction_v3_day_split`。测试的样本通过 import 跟着走。

格式名跟着文件的形状走（用户 2026-09-24 的规定）：样本或索引的字段一有增、删、改名，两边的格式名在同一次改动里
一起换新名字，盘上的集合重新导出；不为旧文件还能读、或"现在没人读错"而保留旧名字。样本格式名不对就整份拒读，
不看文件里是哪个词表——v6（`instruction-v2` 的转弯与保持段）以及更早的都按名字拒读。

`training/index.json` 的格式（`aeroviz-training-index-v1`）不随词表变，所有集合都列在里面，每一条写着自己的
`readingRule` 和 `vocabularySha256`。所以**面板不下载任何样本就知道哪个集合能读**（`trainingSetRefusal`），
别的词表的集合（`instruction_v1`、`instruction_v2` 等）列着、标 "refused"、选中时按名字说出原因，永远不会被下载。
`npm run check-publication` 把这类集合记为警告而不是错误：拒读是有意的，删不删是用户的决定。

词表换版本（新读法或新 sha）时界面会拒读新产物，这是读者在工作：同一次改动里更新钉住值、测试和
`docs/36-2026-09-20-training-module.zh.md`。

### AV20 · Training 的界面不算包络

航向词的航向带与逐行判定、截获转弯、走廊、高度管子、速度的过渡与速度带，全部由
`python run_ts.py instruction_training_export` 在 Python 里算好写进文件：来自
`4dTrajectory/ts_transformer/instructions/display.py`，它只用 `envelope.py` 与 `labeller/*` 的公开函数
（航向词被判的行是 `envelope.heading_word_rows`，每一行在不在带里是 `envelope.heading_words_inside` 一行一行地问，
合计必须等于标注器的判定，否则导出停下；管子就是 `labeller.vertical.tube_bounds`），不在标注器 sha
（`artefact.LABELLER_MODULES`）的范围里。每一个判定都是标注器的 `Reading.checks`。

前端只检查"账"：数组长度与行号、每步生效的词等于发令事件逐行往后填、每个包络对应它那一列的一个词、判定里的
计数等于它自己逐行标记的计数、航向带的第一行是这个词的行加上提前量、带的末行不超过下一个词的第一行和许可行、
带是这个词的目标 ± 词表的容差、截获转弯从许可行到截获行（`headingBandProblem`）。带到底在哪一行结束不在前端重算。
它不再算一遍包络——那会在一个屏幕上给出两个答案。导出器也会把每一架读过的航班用 `read_flight` 重读并与产物里存的
句子逐格比对，不同就停。

### AV21 · Training 的游标与高亮

游标（航班内的秒数，`useTrainingCursor`，AV28）由句子条、读数核对窗口和三维图层共用。游标和实时执行器的选择（`trainingPick`）
属于屏幕上这架航班——`trainingSelectionKey`：机场 / 集合 / 航班（航班键在同一机场的不同集合里会重复）——换航班、换集合都归零 /
清空；在换之前拿到的设置函数写不进新航班（2026-09-25，整模块审查：旧的选择会在换回来时被重新飞）。离开 Training 再回来则都
保留（AV28）。

**高亮的是选中的一个词，不是一步**（用户 2026-09-24）。`AppContext.trainingColumn` 是选中的词类（六列之一，或
没有），高亮的是这一列在游标处生效的那个词（`trainingWordAt`），别的列在同一步生效的词一概不亮——它们的起止
各不相同，一起亮就把没对齐的几段画成了"同一时刻"。

- 句子条：点一条带 = 选中它的列并把游标移到它的发令步；再点已选中的带 = 取消；点顶上的步号只移游标，列不变。
  选中的列名变黄。选中的词类在换航班时保留，游标不保留。
- 读数窗口：鼠标移过图只移游标；在图上点一下 = 选中这张图的列（航向图→航向，高度图→高度，地速图→速度）。
- 每类词亮什么（一个词对应它那一列的第几个包络，解析器核对过一一对应）：航向词→航向图上它自己的航向带，三维里
  它被判的那一段地面投影（截获之后也是它自己的，走廊属于许可）；进近的"许可"词→截获转弯、走廊及其中线、截获后的
  航道带；高度词→它的管子；跑道词→所指的跑道和延长中线；下降角词和速度词在三维里没有位置上的包络，只画生效段——
  读数窗口里下降角词亮它在高度图上的竖线，速度词亮它的过渡和速度带。
- 另外，这个词**生效的那几行**在航迹上画成黄色：三维、平面图，以及画它那个信号的那张图（航向→航向图，
  高度 / 下降角→高度图，速度→地速图）；三维在发令处加一个黄点和词名标签。航向词生效的几行（从它说出到下一个词）
  和它被判的几行（晚一个提前量）不是同一段，两者都画。
- 选中的包络：线（航向词的地面段、截获转弯）变黄；面（走廊、管子）保留本色加深、边线变黄。
- **其余的淡化**：选中一个词时，别的词的包络在三维里颜色透明度乘 0.3（`trainingEnvelopeEntities` 列出全部包络及其
  边线），在读数窗口里整体透明度 0.3；红色的出界行不淡化。
- 游标在同一个词里移动，三维什么都不重画（以该词的发令行为键）；换词只改材质和线宽、加减两个小实体，不重建
  其余几何，不动共享的 `viewer.clock`。游标从不移动相机（选航班时取景一次，见 AV22）。

### AV22 · Training 的三维与图表各用什么坐标

- 三维：航迹用 `signals.altitudeHaeM`（椭球高）；高度管子是一面墙，沿飞机自己的地面航迹，上下沿为导出的
  `lowerHaeM` / `upperHaeM`；航向词被判的那几行、截获转弯的那几行、走廊、候选跑道和延长中线**贴地画**——它们只关于
  平面，词没有给它们高度，飘在任何高度都是编出来的。文件里其余高度都是几何 MSL，椭球高只在这三处，由导出器
  一次换好：MSL 加航班所落跑道的 `hae_minus_msl_m`（数据面减掉的同一个数，`training_files.runway_hae_minus_msl_m`；
  执行器回放、生成层和后端现飞的轨迹也加这个数）。2026-09-26 之前用 EGM96，没重导的旧集仍是旧值（差 −1.1 到 +2.8 m）。
- 航迹下面贴地画一条**地面投影**——贴地的东西只能对着它读，斜着看时空中的航迹和地面上的东西有视差；走廊另画一条
  **贴地边线**；每根管子另画**上下沿两条线**——±25 m 的墙在任何正常距离都缩成航迹下面的一条缝；**选中一架航班时
  相机取景一次**（`frameTrajectoryCamera`，框住航迹，留 1.5 倍余量，因为句子条和左栏遮住了画面的一部分），之后相机
  归用户。
- **开关**（`TrainingLayers`）：`headingBands`（航向词，默认开：航向图上的航向带，三维里被判的地面段，平面图 / 三维 /
  航向图上的出界行）、`corridor`（截获，默认开：截获转弯、走廊、截获后的航道带）、`vertical`（管子，速度图上的带也
  跟着它）、`candidates`（所有候选跑道）。三维右下角的图例（`TrainingLegend`）只列打开的开关。
- 读数窗口的高度图横轴是**水平飞过的距离**（平滑地速积分），因为管子就定义在这根轴上；游标经由行号在时间和
  距离之间换算。航向图画的是展开（不回绕）的航迹，航向带也在同一分支上（`targetOnTrackDeg`，导出器按带的第一行
  的航迹选分支）。
- 平面图按航迹、所指跑道入口、走廊和执行器的航迹取景并裁剪。

### AV23 · Training 的航向词：逐行判定的航向带

（2026-09-24 起，`instruction-v3`；原来的转弯区、平行四边形和保持漏斗随保持段读法一起删掉。）

- **一个航向词不约束位置**：它说的是"提前量（4 s）之后航迹在这个 5° 格子里"，所以它的包络是**一段时间上的一条航向带**：
  从它说出的那一行加提前量，到下一个航向词的那一行加提前量，每一行的平滑航迹都要在目标 ± 航向容差（4.5°）以内（按
  圆周差算）；不超过许可那一行（之后是截获转弯的事）。提前量把最后一两个词的行推到许可行或之后时，这个词**没有自己的行**，
  不画、不判（句子条头部写"no row of their own"）。
- **画法**：航向图上每个词一块矩形，横跨它被判的那几步，纵向是目标 ± 4.5°；有行在带外时边线是红的，那几行在平滑航迹上
  画红——航向图、平面图的航迹上、三维的地面投影上都画。观测航迹按构造全在带里（标注器就是按它读的），红色主要出现在执行器
  的回放上。
- **三维**：一个航向词没有平面上的区域，所以画它被判的那一段地面投影（航向带的颜色，每个词一个实体，选中变黄、其余淡化），
  出界的行再用红线盖在上面；执行器的回放打开时，它自己出界的行红色画在它的地面投影上。航向带本身（目标 ± 容差随时间）只在
  读数窗口的航向图上。
- **截获转弯**：从许可那一行到截获那一行，朝航道转；判定是单调、平均和最大转弯率、最大坡度（`checks["capture_turn"]`）。
  航向图上画成一段竖条（起止）和航道那条虚线，平面图和三维画成航迹 / 地面投影上的虚线段；判定不过时是红的。
- 这些行、带和逐行判定都是导出器用 `envelope.heading_word_rows` 与 `envelope.heading_words_inside` 算的，合计与标注器的
  判定核对过；前端不重算。

### AV24 · Training 的叠加层：执行器的回放与先验的预测，画在一个集合的航班上

- **叠加层在集合旁边，不在集合里面。** 每个机场的 `training/overlays.json`（`aeroviz-training-overlays-v2`）列出叠加层：
  种类（`executor-replay` / `prior-prediction`）、画在哪个集合上（`base`）、文件位置
  （`training/<叠加层 id>/executor.json` 或 `prior.json`）。集合的索引 `index.json` 不动。写它们的是
  `run_ts.py executor_training_export` 和 `prior_training_export`（ts 的 R13）。
- **绑定按内容，不按文件**（2026-09-28）：叠加层文件（`aeroviz-training-executor-v4` / `aeroviz-training-prior-v4`）
  自己写出它和所画集合共有的东西——集合 id、规格 sha、候选跑道 sha（`candidatesSha256`：跑道编号指向它）和机场坐标系
  （`airportFrame`），每架航班按集合的顺序一架一条；执行器的每条词与句子的词逐条对应（步、列、值），先验每架的步数等于
  句子的步数，画高度的叠加层每一行的 HAE − MSL 等于集合里这架航班的（`TRAINING_DATUM_TOLERANCE_M` = 0.02 m：两边各是
  两个写到 0.01 m 的高度之差），从集合的观测状态飞出的航迹第一点就是观测航迹那一行（执行器第 0 行，模型自己的句子第
  `firstPredictedRow` 行；位置差不超过写出的 1e-7°、高度 0.01 m；增强起点是挪过的，不比）。`trainingOverlays.ts` 逐项核对，对不上就整份拒读、说出是哪一项；`check-publication` 用同一个
  读取函数。**从不按集合文件的字节或写出时刻绑定**：集合用同样的航班重新导出，叠加层照读；航班、句子、规格、跑道、坐标系或
  高度基准有一样变了才拒读。导出时同样核对（`training_files.open_base_set`：候选跑道与坐标系是导出器自己产物的；
  `require_set_datum`：每架航班的高度基准）。执行器的格式 v2（2026-09-24，`instruction-v3`）：每个被判的航向词带上它在飞出航迹上的航向带和
  逐行判定（`heading`），每架被判的航班带上判决读到的飞出航迹（`judgedTrackDeg`：平滑、切到落地前，与 `track.trackDeg`
  同一分支，第 k 步就是 `track` 的第 k 个点，读取时核对）；动力学失败的航班没有这两样（判决读到了导出航迹有意不含的失败状态），
  词的状态与检查照旧。读取器要求：判决判过的每个航向词都带带子、别的词都不带；判定数 = 有判定的词数，离开航向词自己切入的
  那一个词若它的带也判过行则多数一次。v1 的航向判定是转弯与保持段，按名字拒读。
- **界面**（`useTrainingOverlays`）：面板的 Draw 里两个开关（有叠加层时默认开），没有就灰掉并写出生成它的命令；同一种
  有几个时给一个下拉，默认最后列出的；选中的那一个属于它所在的机场与集合（换了回到最后列出的，换回来恢复）。一个叠加层
  只下载一次，开关多少次都一样——除非下载失败（取不到或读不了）：那样把开关关掉再打开就重新下载。打开时把所选航班的那一条发布为 `trainingExecutor` / `trainingPrior`，与
  `trainingSelection` 分开，所以开关一个叠加层不会重新取景。执行器回放的门表、先验的读数表折叠在开关下面
  （`TrainingResults`），数字照抄产物。
- **执行器**（青色 `TRAINING_EXECUTOR_COLOR` #14b8a6，OKLab 与调色板里每种颜色的距离 ≥ 11.9）：句子条每条词左端一个点
  （青 = 在自己的包络里，红 = 出界，空心灰 = 不判 / 没说到 / 被同一步的后一条词取代；没有自己检查的词——跑道指针、
  下降角词、"未许可"、"未指定"——不画），悬停写出每项检查和原因；头部写结局、越过入口时的偏离和高度、词的合格数、
  evaluation 判定；航班列表每架下面一行结局与合格数；读数窗口平面图画它的航迹，三张图用虚线画它在**自己的时钟和自己
  飞过的路程**上的航向、高度、地速（它按自己的节奏飞，不和观测在时间上对齐，坐标轴放宽到能装下两者），航向图上还有它自己
  的航向带（青色边框，从它听到每个词的那一步加提前量算起），出界的行红色；三维画它的航迹（椭球高）、贴地的虚线投影、出界的
  行和终点标签。**执行器的词是按从它自己听到这条词的位置重新画的包络判的**，观测航班的包络不是它的。
- **先验**：头部写这架航班每步的负对数似然和 val 整体的；"Prior predictions" 窗口（`TrainingPriorWindow`）在游标处按列给出
  真值在这一步说了什么、先验给真值的概率、给"这一步说一个词"的概率、若说一个词最可能的几个词；下面每列一条带：说词的概率
  （列的颜色）、真值的概率（灰），真值在第 0 步之后说的每个词一根竖线——先验排第一的词就是它为青色，否则红色。**这是
  教师强制的读法**：每一步都看到真值句子在它之前的词，不是先验自己说出的句子。

### AV27 · Training 模块的代码结构（2026-09-25 整模块审查后）

- **一个读取器**（`src/data/trainingReader.ts`：`Reader`、`Refusal`、`attempt`、`Parsed`、清单的骨架 `parseManifest`）读三种
  文件：集合与样本（`trainingSample.ts`）、叠加层（`trainingOverlays.ts`）、实时执行器的答复（`trainingAutopilot.ts`）。拒读
  的信息带字段路径（`sample.flights[3].signals.tS has 40 values, expected 41`）。镜像常量仍在原文件里（Python 测试按文件路径
  用正则读它们）。
- 共用的读法各写一次：航向带（`readHeadingBand`：样本的带**正好**结束在 `heading_word_rows` 算的那一行，回放与实时执行器的带
  **不超过**它们判决读到的飞出航迹）、一个词的判定（`readWordVerdict`、`readJudgedBand`）、越过入口（`readCrossing`）、出界的
  行（`outsideSpans`，读数窗口与三维共用）。执行器没飞的航班是它自己的类型（`TrainingExecutorUnflown`）。
- 各视图的措辞在 `src/data/trainingText.ts`（判决的结局、检查、越过入口、sha、时长），回放和实时执行器不再各说各的。
- 读数窗口 = 纯函数的模型（`training/readbackModel.ts`，所有比例尺与范围）+ 四张图 + 画图小件（`chartKit.tsx`）+ 与先验窗口共用
  的外壳（`TrainingWindow.tsx`）；图宽由 `useMeasuredWidth` 量。
- 三维：`src/scene/trainingEntities.ts`（实体 id、坐标展开、线与点的几种画法、`entityGroup`）；观测航班的场景一架航班只建一次，
  Draw 开关只改 `show`；回放与实时执行器在 `useTrainingExecutorLayers.ts`；三维回放的计算（倍数、位置、标签）在
  `trainingAutopilot.ts`，不用 Cesium 就能测。
- 实时执行器与叠加层的视图只在"属于屏幕上这架航班"时画（`autopilotOnScreen`、`overlayOnScreen`：机场、集合、航班都对上）。
- **只在提示里的说明也写成页面文字**：读数窗口的色样与两行说明、先验窗口的读法、三维图例的每一行，各有一个 ⓘ
  （`training/NotesToggle.tsx`：`aria-expanded` 的按钮，展开是一个 `role="note"` 的区块，列表用 `NotesList`），键盘能到、读屏能读；
  面板上每个 Draw 开关与叠加层的说明、先验读数表原来只在每行提示里的那些数，写在详情页上（AV33）。
- 集合与叠加层里的名字镜像（`TRAINING_STRATA` ↔ `instructions.readout.STRATA`，`TRAINING_PRIOR_RULES` ↔ `prior.readout.RULES`）
  由后端的 `MirrorTest` 逐字比对；样本的其余镜像由 `test_instruction_training_export.py` 钉住。

### AV28 · Training 的游标只让读它的组件重渲染；Training 的会话跨任务保留（2026-09-26）

- `useApp()` 把八个 context 一起展开，所以任何一个变了，所有调用它的组件（约 50 个）都重新渲染。游标在图上每次鼠标移动都变，
  所以它是**单独的一个 context**（`TrainingCursorContext`，只能用 `useTrainingCursor()` 读），`useApp` 不读它；读游标的只有
  `TrainingSentenceBar` 和 `useTrainingTrackLayer`。后者由**叶子组件** `src/components/TrainingScene.tsx`（什么也不渲染）调用，
  不在 `FlightApp` 里——放回外壳，每次悬停整个工作台就又跟着重渲染。`src/__tests__/App.test.tsx` 钉住这一点（移动游标时
  `FlightApp` 不重渲染），`AppContext.test.tsx` 钉住 `useApp` 的读者不重渲染。
- `WorkbenchLeftDock` 在进入 Training 时挂上 `TrainingPanel`（记下当时的机场），之后一直挂着，在别的任务里只加 `hidden`
  （`.training-panel[hidden] { display: none }`——面板自己的 `display: flex` 会盖过这个属性）。所以面板的会话——索引、集合、
  样本、航班、叠加层、实时执行器——跨任务保留，回来时什么也不重新下载。**会话只属于打开它的那个机场**：在别的任务里换了机场，
  面板就卸下（它的清理把选择与叠加层清空），回到原机场也不再挂上，下次进 Training 为当时的机场重新打开；在 Training 里换机场，
  面板按机场重新挂上（`key`）。否则隐藏的面板会在后台为每个打开过的机场下载 Training 文件（审查发现，样本还下载两次）。
  已发布的选择在别的任务里仍然在，所以**画 Training 的东西必须看 `mode`**：句子条与三维图层只在 `mode === "training"` 时画。
  回到 Training 时三维重建、相机再取景一次；飞完的实时答复按 `playedAt` 计时，直接画成落地后的样子。

### AV29 · 频繁变化的值各有自己的 context，`useApp` 不读（2026-09-25）

- `useApp()` 把 `AppContext.tsx` 的八个 context 一起展开，任何一个变了，约 50 个调用它的组件都重新渲染。所以**每次鼠标移动、每个
  瓦片、滑块的每一步都会变的值，各放进自己的 context，用自己的 hook 读**，`useApp` 不展开它们；写它们的 setter 仍在原来的
  context 里（setter 不变，写的一方不会因此重渲染）。现在有三个：Training 的游标（`useTrainingCursor`，AV28），本地地形预加载的
  瓦片计数（`useAirportLocalTerrainProgress`，只有 HUD 读；`AirportLocalTerrainState` 不再带计数，按阶段——加载、预加载、可用、
  出错——才写一次，`airportLocalTerrainPhaseState`；换机场时计数归零；加载地形的 hook 自己不存计数；HUD 的文字是纯函数
  `localTerrainLabel`），测距环的半径（`useRangeRingRadiusKm`，测距环图层与 Layers 抽屉的滑块读）。
- 读这类值的组件要么本身就是显示它的（句子条、HUD、抽屉），要么是什么也不渲染的叶子（`TrainingScene`）；**不要在 `FlightApp` 这样的
  外壳里读**。`AppContext.test.tsx` 对三者各钉住"变了只重渲染它的读者、不重渲染 `useApp` 的读者"，`App.test.tsx` 钉住外壳不跟着游标动。
- 实测（KSTL，浏览器）：HUD 从 Preload 0/41 逐个数到 36/41 再到 Active，整个界面不再跟着每个瓦片重渲染。

### AV30 · Training 的左栏停在句子条上面；航班列表占剩下的高度（2026-09-26）

- 句子条横跨整个宽度，原来左栏一直伸到底，下半截（Draw 的开关、叠加层）被它盖住。现在句子条用 `ResizeObserver` 量自己的高度，
  写到页面的根元素上（`--training-bar-height`，句子条卸下时删掉；2026-09-28 之前写在 `.workbench` 上，Cesium 的署名在
  工作台之外也要读它，AV36）；`index.css` 里
  `.workbench:has(> .training-sentence-bar) > .cesium-overlay-container` 的底边距是句子条离底的距离（`--training-bar-bottom`，
  8 px；2026-09-28 之前 36 px，给 Cesium 的时间轴让位）+ 句子条高 + 8 px，
  左右两侧的面板因此都在它上面结束；原来给 Cesium 时钟盘留的底边距（`left-overlay-panel-stack` 的 92 px）这时归零——那一块本来就被
  句子条盖着。
- 左栏里 `TrainingPanel` 是 `flex: 1 0 auto`（填满左栏、不比内容矮），航班列表 `flex: 1 1 0; min-height: 176px`：列表拿剩下的高度，
  最少约五个两行的条目（执行器回放打开时每条两行）或九个一行的；内容再多时整个左栏滚动，列表下面的开关不会被挤掉，也不会被句子条盖住。
- 在演示模式里句子条隐藏但仍挂着，变量留着；两侧面板那时也隐藏，没有影响。

### AV31 · Training 读模型自己说的句子：`prior-generation` 叠加层（2026-09-26）

- 导出：`python run_ts.py prior_generation_training_export`（ts R13）——先验说、执行器飞，和正式自由生成读数同一条路径
  （`prior_free_generation.speak_and_fly`）；每架航班 `--samples` 个样本；只飞自己机型有动力学的航班。文件
  `training/<叠加层 id>/generation.json`，格式 `aeroviz-training-generation-v4`（Python `SCHEMA` 与 `TRAINING_GENERATION_SCHEMA`
  由 `test_training_overlays.py` 逐字比对），清单里的种类 `prior-generation`（不认识的种类只拒这一条）；与集合的绑定同 AV24。
- 每个样本：事件（行、列、值，行是航班自己的步号，从 `firstPredictedRow` = 8 起、第一步六列都说）、结局、结束时刻、越过入口
  （只随 `TRAINING_CROSSING_OUTCOMES` = `judge.CROSSINGS` 出现）、首末跑道、换跑道与复飞词数、结束时是否许可、屏蔽拿掉的概率
  （按列名的记录，列只能是六列之一——不镜像哪几列被屏蔽，后训练在加高度列的屏蔽）、航迹（航班自己的时钟，每点一步、最后一点
  可以不满一步，到结束时刻；动力学失败的早一个周期；MSL 与椭球高）。换跑道次数、复飞词数、结束时是否许可要和词对得上。读取器
  逐项核对这些账，对不上整份拒读（`trainingGeneration.test.ts`）。词写到执行器停下处，可能在结束时刻之后（越过入口没截获、失速
  截断）：照正式读数的数法写出，句子条把多说了一整步以上的那段涂暗。正式读数分这个机场与全部机场两份（`readout.prior.here/all`）。
- 读哪一句是 `AppContext.trainingSource`（null = 真值；或 `{overlayId, sample}`），只在句子条的标签页上选（左栏的 Sentences read 已删，2026-09-27）；换航班
  保留。`generationOnScreen` 找出屏幕上这架航班的那个样本；模型不飞这架航班时回到真值。**真值总是画**，模型的东西只在读它时画。
- 区分靠框不靠带：模型的带和真值的一样平涂（斜线 + 虚线边框让看板太乱，用户 2026-09-26），句子条边框、各行左边的竖色条
  （`training-sentence-model-strip`）、标签页、样本小块是模型的颜色；前 8 步淡灰底 "observed"，每行底边白色短竖线是真值在这一列
  说词的地方；模型航班结束的实线下面，时间轴上用模型的颜色写出结束时刻（`training-sentence-model-end`，总是写；离它太近的刻度
  让开，写在轴末端时右对齐）。颜色按模型的名字给（AV32），配色校验的数写在 `trainingWordColors.ts`。
- 三维（`useTrainingGenerationLayers`）：读的样本实线、贴地虚线、结束标签、它说航向词和许可的点（模型的颜色）；其余样本细线
  半透明（`TRAINING_OTHER_SAMPLE_ALPHA`）。选中的词画在模型飞出的航迹上，真值的包络不淡化（`useTrainingTrackLayer` 读模型时不画
  真值的选中）。
- 模型的许可与复飞用集合进近词表里的 "cleared" / "go-around"（`TRAINING_APPROACH_CLEARED` / `TRAINING_APPROACH_GO_AROUND`，
  导出器 `APPROACH_NAMES` 的镜像，`test_instruction_training_export.py` 钉住；样本读取器拒绝没有它们的词表）。
- 换模型从第 1 个样本开始；模型不飞的航班画真值连同真值的整个头部；Read-back / Prior 两个窗口读真值，只在读真值时给出。
  ▶ Fly 两边都有：模型的词交给实时执行器飞（AV26 "模型的词"），模型航班结束之后说的词灰着。
  左栏只留当前集合的模型下载，读不了的有 Retry（`useGenerationOverlays`）。

### AV32 · 模型按名字认：base / landing / augmented 加轮次，句子条上可以在各轮之间切换（2026-09-26）

- 用户 2026-09-26：Training 里只能在 base 和"后训练"之间切换，多轮后训练的各个模型分不开。先验的模型名字是用户定的
  （后训练设计开头的表）：**base**（只用数据训练）、**landing**（base 按落地奖励后训练，第一阶段）、**augmented**（landing
  在增广起点上再后训练，第二阶段）；某一轮写成 "augmented r3"。
- 导出器不再收 `--label`，名字从检查点的配置读出（`model_identity`）：没有 `fine_tuning` 就是 base；有的话，按训练它的方法
  （`fine_tuning.schema` 去掉版本号 `-vN`，`METHOD_MODELS`）给名字，同一方法的各版本是同一个模型（采纳的 landing 第 1 轮是 v1 写的）；
  没有名字的方法拒绝，新阶段的名字先和用户商定。载荷的 `model` 块：`name`、`round`（base 为 null）、`run`（放各轮的目录，
  base 是它自己的目录）、检查点 sha256、变体、训练时的提交、`fineTuning`（方法的 schema、起步模型的路径 `from`，以及
  起步模型的名字和轮次 `fromName` / `fromRound`——从起步模型自己的 `config.json` 读，在这一轮所在的 outputs 树里读，因为各轮是在
  工作树里跑的）。默认叠加层 id `generation_<名字>[_rNN]_<检查点 sha256 前 8 位>`。格式因此升到 v2，v1 按名字拒读。
- 读取器要求三者一致：base ⇔ 没有轮次 ⇔ 没有 `fineTuning`；起步模型是 base ⇔ 它没有轮次；`run` 必须在 `…/prior/` 下。名字的
  列表 `TRAINING_MODEL_NAMES` 是导出器 `MODEL_NAMES` 的镜像，`model` 块的字段与 `model_block` 的输出逐个比对
  （都在 `test_training_overlays.py`）。
- **分组只有一处**：`trainingModelGroups`。按名字的训练顺序（base、landing、augmented），再按 run，一个 run 的各轮按轮次排。
  同一个名字有两个 run（例如第二阶段停掉重跑过）时，每组都写上 run——写它的实验目录（"augmented r1 · v3_stage2_clip_20260926"），
  两个 run 同在一个实验目录里（不同种子）时写全（"…/aug_s1337"）；同一个
  run 的同一轮导出了两次（同一个检查点换了导出设置），各自单成一组，用叠加层 id 区分——不会出现两个一样的 "r3"。句子条、左栏、
  结果表、图例都用它给的 `title` / `memberLabel`，不自己拼名字。
- 句子条：每个模型一个标签页；正在读的模型发布了不止一轮时，标签页旁边是它的各轮 "r1 r2 …"，提示里写这一轮全称和这架航班上它的
  样本落地了几个。换轮次**保留样本号**，同一架航班可以一轮一轮地比；标签页回到这个模型上次读的那一轮，
  没读过的从第一轮开始、从第 1 个样本开始。结果表按同样的顺序和名字。
- 颜色按名字（`TRAINING_MODEL_COLOR`）：base 品红 `#d946ef`、landing 黄绿 `#a3e635`、augmented 树莓红 `#b82e7a`——配色校验器在
  这套配色里能找到的最好的一个：和 Training 的每种颜色 OKLab 色差正常视觉 ≥ 17.7（最近的是实时执行器的出界红，离 base 18.5），
  模拟色盲 ≥ 11.6；但对比度只有 3.3:1，所以只用在线、色块、边框和时间轴上的结束时刻（用户要这个时刻用模型的颜色写），名字和数字
  用正文颜色，旁边放色块。同一模型的各轮同色，轮次用文字说。
- **模型按它训练时的屏蔽说话**：词表规则总是开着；程序屏蔽是模型自己的（`procedure_masks.json`：base、landing 无，augmented
  `procedure-altitudes-v2`）。带程序高度的句子在飞到下滑道下边界以下的那一步截停，结局 `below_glidepath`（`TRAINING_FREE_OUTCOMES`
  = 判决的结局 + 它，`prior_free_generation.FREE_OUTCOMES` 的镜像），航迹到那一步结束、没有越过入口；读取器只在叠加层的
  `generation.procedureMasks` 含 `procedure-altitudes-v2`（`TRAINING_PROCEDURE_ALTITUDES`）时接受这个结局。实时执行器重飞时带回
  这几套屏蔽和数据摘要，后端核对后在同一步截停（AV26）。导出器拒绝在程序屏蔽下画的正式读数，结果表的说明写明这一点；修法见仓库
  `docs/code-health-followups.md`（2026-09-26）。

### AV33 · Training 的详情页：左栏不再展开长内容（2026-09-27）

- 用户（2026-09-27）："左边栏的详情……展开后都挤在左边栏里 根本没法读 可以单独设计一个详情页"。原来标题旁的 ⓘ 在左栏里展开
  模块说明、每个开关的说明、词表的各项数值；最下面三个 `<details>` 在左栏里展开回放门表、先验读数表、模型句子的落地表——
  左栏只有 220–280 px 宽，表格挤成一团。
- 现在：`src/components/training/TrainingDetails.tsx` 是一个**模态**的详情页（经 portal 挂到 `document.body`，AV7；背景变暗），
  左边一列分节标签（What this view shows · Vocabulary · The models' own sentences · The executor's replay gate · The prior's
  readout），右边是选中那一节，单独滚动。没有内容的节照样列出、灰掉，写明原因——先看叠加层清单（loading … / the overlays
  manifest cannot be read / none published for this set，清单拒了条目时后面写拒了几条），再看开关与下载
  （switch on … under Draw / loading … / cannot be read）。打开时 `document.body` 下它以外的元素都设 `inert`（Tab 出不去）；Esc
  （焦点在哪都行）、右上角 ×、点背景（左键）关闭，焦点回到打开它的那个按钮——按钮由面板传进来，因为 Safari 点按钮不给它焦点；
  面板被隐藏（切到别的任务）时详情页关掉，回来不会自己再开。↑↓ 在有内容的节之间移动。
- 左栏：标题旁一个 **ⓘ**（aria-label "Training details"，与模块里别处的 ⓘ 一致，2026-09-27 由 "Details" 按钮改来）打开第一节；Draw 下面每个读数一行——名字加结论（如 "Replay gate  val · spec 0d6a68a92c6f"、
  "Models' sentences  base 86% · landing r1 98% · …"），装不下时省略号，全文在提示里——点它打开详情页的那一节。左栏里不再有表格。
- 各表按宽页重排（`TrainingResults.tsx`）：表头大写小字、行间细线、数字右对齐等宽数字、合计行加底色；门表的 ✓ / ✗ 用绿 / 红，
  放在份额旁边；先验的似然表里每行三者中最低的加粗；**原来只在每行提示里的先验读数**（第一个预测步的第一名命中率、换词步数、
  换词处说词的概率、第一名 / 前五名命中率、不该说时说词的比例）成了自己的一张表；第一个预测步的跑道单独一张表；模型句子的落地表
  每个模型一组行（有正式读数时才出现 "sentences of" 一列，否则每行都是 "this set"），份额后面淡色写 "of N"；每个模型怎么抽样、
  怎么飞：所有模型相同的字段（每架样本数、温度、种子、执行器规格、时限、正式读数）在表上面写一次，不同的字段（run、从哪个模型
  后训练、程序屏蔽）是表的列。数都与原来相同，只是摆法变了。

### AV34 · 滚动条全局统一一种样式（2026-09-27）

- 用户（2026-09-27）：Training 左栏内外两个滚动条是浏览器默认样式，和配色不搭，"最好是全局性质的修改"。现在 `index.css` 开头
  一处定义所有滚动区的滚动条：`:root` 上的 `--scrollbar-size`（10 px）、`--scrollbar-thumb`（应用的蓝 `#7eb8f7`，28 %）、
  `--scrollbar-thumb-hover`（青 `#67e8f9`，55 %）、`--scrollbar-track`（透明）；Chromium / Safari 用 `::-webkit-scrollbar*`
  画内缩的圆头细条（透明边 3 px，悬停时 2 px）、不要箭头按钮；Firefox 只有标准属性（`scrollbar-width: thin` + `scrollbar-color`），
  放在 `@supports not selector(::-webkit-scrollbar)` 里——Chromium 121 起标准属性一设就压过 `::-webkit-scrollbar*`，不能两者都给它。
- **组件不再各自写滚动条样式**；要不同的只改这几个变量。原来唯一的一处（程序详情页的侧栏）已删掉。整个应用都是深色表面，所以
  只有一套颜色。

### AV35 · 增强起点上的模型句子：`prior-generation-augmented` 叠加层（2026-09-27）

- 用户（2026-09-27）：前端 40 架都是真实 val 航班，想看模型从第二阶段后训练用的**增强起点**怎么飞。导出器
  `prior_generation_training_export --augment-seed 1337`（ts R13）写另一种叠加层：种类 `prior-generation-augmented`，格式
  `aeroviz-training-augmented-generation-v2`（与集合的绑定同 AV24，增强后的第 0–7 行也核对高度基准）。每架航班多 `augmentDraws`、`augmentation`（转角、抬高、速度倍数）、
  `observed`（增强后的第 0–7 行）；同一个种子下每个模型的增强起点相同；没有正式读数。设计：文档 36 §2.8、§4.10。
- 读取器（`parseAugmentedStart`）核对：抽了几次在 1–`tries` 之间且只有自己动力学的航班才抽；有增强 ⇔ 飞了 ⇔ 有 `observed`；
  增强在上下限以内；`observed` 正好 `firstPredictedRow` 个点、从 0 起每点一步；一架航班的样本都从同一个起点起飞。
- 界面：句子条头部 `Real start | Augmented start`（只在发布了增强起点时出现），两边各自一组模型标签页（`trainingModelGroups`
  按起点分开调用，名字不串）；读增强起点的样本时没有 Truth 标签页、不画真值的发令短竖线和观测结束虚线，灰底写 "observed,
  moved"，样本小块写增强量；▶ Fly 照样能用：请求带上增强量，后端按 `augment_state` 挪执行器的起点（`fly.moved_inputs`，
  与第二阶段的 `augmented_inputs` 逐位相同）、按 `augment_signals` 挪观测行、时限 ×2，飞出来就是那个样本（导出的增强量
  取整到 0.0001° 时差约 2 cm，此后导出不取整）；详情页的模型一节多一张表；三维把增强后的前 8 行用模型颜色虚线画出，接到样本航迹起点。

### AV36 · Training 藏起 Cesium 的时钟；句子条贴底；下降角一行叫 Descent；实时执行器的小游标（2026-09-28）

- 用户（2026-09-28）：Training 里 Cesium 的控制台（时钟盘、时间轴）不起作用，藏起来让句子条沉底。Training 不载 CZML，句子条
  有自己的时间，`viewer.clock` 属于 Observe 的回放。`WorkbenchShell` 按 `mode` 给 `body` 加 `workbench-training-active`，
  `index.css` 把时钟盘、时间轴和全屏按钮设成 `visibility: hidden`，再 `viewer.forceResize()`：Cesium 排版时只看 `visibility`
  （`display: none` 它看不出，演示模式用的就是那种，署名不动），所以署名随之排到最底下；回到别的模式时同一个 effect 再强制排一次，
  时间轴重新排好。浏览器核对：Observe 的时钟盘、时间轴、全屏按钮照旧，Training 里都不见。
- 句子条离底 `--training-bar-bottom`（8 px，同左栏离顶栏的距离，定义在 `:root`）。Cesium 的署名（ion 标志与 "Data attribution"）
  必须看得见，而它们会落在句子条下面：Training 里用 `!important` 盖过 Cesium 每次排版写在行内的位置，放到句子条上方 4 px、左栏
  右边（`calc(16px + var(--overlay-left-width) + 16px)`）；为此 `--training-bar-height`（AV30）写到根元素上，
  `--overlay-left-width` 从 `.cesium-overlay-container` 挪到 `:root`（其余读它的地方都在根元素之下，值不变）。没选航班时没有
  句子条，署名离底 12 px。
- 下降角一行（`TRAINING_COLUMN_LABEL.angle`）的行名从 "Angle" 改成 "Descent"（用户：含糊），详情页词表的那一行也一样；带上的字
  （`trainingBandLabel`）写成 "编号: 标称角"——"1: 0.92°"（类名去掉开头的 "descent "，行名已经说了），爬升 "climb: -1.32°"，平飞
  "level"；带的提示、三维标签、读数与先验窗口仍用完整的词（`trainingWordLabel`）。
- 实时执行器在飞的时候（文档 36 §4.7），句子条上它飞的那个词所在的行有一根带白边的短竖条（`AutopilotCursor`），颜色同飞出的线：后端还
  在算时停在这个词说出的那一步、一闪一闪（`prefers-reduced-motion` 时不闪）；答复到了就与三维里的飞机一起走——两边读同一个时钟
  `autopilotPlaybackS(track, playedAt, now)`（实际时间 × 倍数，飞完停在段尾），过了听到下一个同列词的那一点（尾巴）变淡 0.45，
  飞完停在段尾；再飞一次（↻ Fly again）的新答复带新的 `playedAt`，它从头再走。它自己用 `requestAnimationFrame` 改自己的 `transform`，句子条不因它
  每帧重画（AV29）；飞完就停掉循环，轴的宽度或长度变了在绘制之前重新放。浏览器核对（KRDU AAL557）：真值的 heading 170°（64 s，×8）
  与 base 模型第 1 个样本的 heading 225°（74 s）——游标与三维标签的"已飞"同步，停在尾巴末端、变淡。

### AV37 · 执行器回放对一架航班的判定：三种颜色；出问题时句子条头部一个小块（2026-09-28）

- 用户（2026-09-28，看 KRDU SWA4462 的 "landed · 43/45" 为什么是红的）：落地了、45 个判了的词里 2 个出界，和没落地用同一种
  红色分不开。现在一个规则、一处定义（`replayVerdict`，`data/trainingOverlays.ts`，挨着 `executorWordCounts`）：**clean**——落地、
  判的词全在包络里；**flawed**——落地，但有词出界，或者标注器的门拒了它的航迹（那样一个词也不判，不能算干净）；**not landed**。
  出界的词数用回放门自己的计数（判的 − 包络内），与航班列表的 "43/45" 一致——门把"留给截获的航向词"算两次（`countedTwice`，
  AV24），按词的判定数会少一个。颜色表 `TRAINING_REPLAY_COLOR`（`utils/trainingWordColors.ts`，调色板里只有颜色、没有判定）：
  clean 用回放的青、not landed 用出界红、flawed 用琥珀 `#f59e0b`（OKLab ΔE 在句子条底色上离出界红 14.6、色觉异常下 ≥ 8.6，离青
  24.9，离选中的黄 11.2，对比度 8.6:1；离截获转弯的橙只有 4.2，那个颜色只画在图和三维里、不做文字）。
- 航班列表那一行（`ExecutorTag`）用这个颜色，提示里写出三种颜色的含义；详情页 Draw 一节多一条 "Executor replay"，三个色样
  与同样的说明。
- 句子条头部：读真值、回放飞了这架航班、且不是 clean 时，跑道之后多一个小块，字越少越好（`replayIssueText`，`data/trainingText.ts`）：
  "Replay · 2 words out"、"Replay · timed out · 1 word out"、"Replay · track refused"，颜色同上；出界的词逐个
  （"heading 095° at step 164"，被算两次且两项都没过的那个写 "counted twice: its band and the intercept"，名字加起来就是出界数；
  `replayOutsideWords`）写在它的提示里，也接在 ⓘ 说明里回放那一句之后。clean 时不出现——2026-09-27 删掉的回放小块（"冗余"）是一直
  在的那种，这个只在出问题时出现。
- 审查（opus，2026-09-29）：没有必须改的；按它改了——判定合成一处（原来颜色和文字各推一遍）、出界数改用门的计数并标出算两次的词、
  被拒的航迹不再算干净、头部注释补上这个小块。浏览器（5176）：KRDU 列表里 "landed · 43/45" 等为琥珀、"landed · 5/5" 等为青；
  SWA4462 头部 "Replay · 2 words out"。

### AV38 · 左栏的 "Autopilot (live)" 一栏删掉；点色块总是直接飞（2026-09-29）

- 用户（2026-09-29）："左边栏里 AUTOPILOT (LIVE) 这个面板中的信息也是多余的 把这块儿也删了……删干净，包括相关测试"。删掉的：
  面板的这一栏（开关 "Fly on band click" 与结果卡 `TrainingAutopilotCard`：词与步数、判定、尾巴的说明、模拟飞行与计算两个时间、
  检查项、Details 里的计算分项 / 规格与代码 sha / 词钟、模型词与导出样本的逐点核对、"Replay in 3D"、没飞成时的 "Fly again"），
  以及只为它存在的：`AppContext` 的 `replayTrainingAutopilot`、`trainingAutopilotAuto` / `setTrainingAutopilotAuto`，
  `trainingAutopilot.autopilotSampleGap`，答复视图的 `roundTripS`（浏览器往返时间），`trainingText.formatUtc`，详情页 "What each
  switch draws" 里的 "Fly on band click" 一条，`index.css` 里结果卡的样式，`ProblemBox` 的 `children`（只有结果卡的 "Fly again" 用），
  测试夹具 `onSampleLine`，以及它们的测试；`requestSource` 与夹具的 `mockGeneratedPoint` 只剩文件内使用，不再导出。
- 留下的：句子条头部那一行（`TrainingAutopilotStatus`，从 `TrainingAutopilotCard.tsx` 挪到自己的文件，测试同样挪到
  `TrainingAutopilotStatus.test.tsx`；时间写法的测试挪到 `trainingText.test.ts`）、小游标（AV36）、三维与读数核对窗口。点色块总是
  直接飞（原来开关默认就开）；没飞成时按 ↻ Fly again 再飞（再点一次选中的色块是取消选中，不是再飞）。答复的格式（`aeroviz-training-autopilot-segment-v7`，含
  `timing` 各项、规格与代码 sha）不变，读取器照样逐项核对——那是与后端的约定，不是界面。原来的 3D "Replay in 3D" 测试改成同一个
  词的再飞一次（↻ Fly again 的路径：先 flying 再 ready）：后端在飞的那段时间里三维不画上一次的答复（原来的 "Replay in 3D" 不请求，
  一直画着——这是删掉它之后唯一的行为差别），新答复到了照样飞出来，落地时画的撤掉、不重复添加。审查（opus，2026-09-29）：没有必须
  改的；留下的注释、`ProblemBox` 的 `children`、两个只剩文件内使用的导出、AV38 里"再点一次"的说法（点选中的色块是取消选中）、
  两个没走 flying 这一步的测试，都已改。

### AV39 · 多机模式：窗口集合与窗口里的模型句子（2026-09-30）

- 用户（2026-09-30）要在 Training 里看多机：同一机场 20 分钟的一个**窗口**，模型同时指挥这段时间进场、有句子的几架，其余照记录
  回放。方案用户审过（设计与定下的事在 `36-2026-09-20-training-module.zh.md` §2.9、§4.11）。数据由 ts `window_training_export`
  写：集合 `traffic.json`（`aeroviz-training-traffic-v1`，种类 `traffic-windows`，与模型无关：指挥的飞机就是普通的集合航班，另有
  每个窗口的其他飞机与"记录里的样子"）和每个模型一份叠加层 `window_generation.json`（`aeroviz-training-window-generation-v1`，
  种类 `window-generation`：每个窗口每个样本里每架的句子、目视与 IFR 两种读法下的失去间隔、落地）。
- **结构：一个场景层，加一架焦点飞机。** 窗口里被指挥的一架就是一架普通航班，模型对它说的就是一句普通的模型句子，所以句子条、
  读数窗口、单机的三维图层原样使用。共用代码推广了四处，没有加分支：
  - 游标存在选中项的时钟上（`TrainingSelection.clock = {scope, offsetS}`，`AppContext`：`trainingCursorS` 是这架自己的时间，
    `trainingSceneS` 是时钟的时间）；单机时时钟就是这架航班、偏移 0，与原来逐位相同；窗口时时钟是整个窗口，换一架时刻不变。
  - 模型叠加层拆出头部 `TrainingGenerationHead`（`readGenerationHead`、`readSaidWords`），窗口叠加层用 `windowGenerationView`
    投影成原来的视图。
  - 样本读取拆出集合头部 `readSetHead`。
  - 左栏拆成 `TrainingFlightSession` 与 `TrainingWindowSession`（共用 `PanelParts`）。
- **模式由集合的种类决定**（`data/trainingSets.ts`：`parseTrainingSet` 按种类打开，`TRAINING_OVERLAYS_OVER` 规定每种叠加层只画在
  哪种集合上，`parseOverlayOver`；`check-publication` 走同一个门）。
- **前端不判间隔、不排落地**：失去间隔段、被结束、落地都是导出器的判定器给的；读取器只核对账：点名的飞机在窗口里；被判定器结束的
  飞机，它的句子写 `lost_separation`，结束时刻等于目视读法里结束它的那一刻、同一个对象；落地正好是自己的结局为落地的那几架，按先后。
- **窗口里的飞机不现飞**（`TrainingSelection.liveExecutor` 为 false）：后端只打开读回集合，句子条不给 Fly 按钮，实时执行器的钩子
  什么也不问。
- 句子条上方的**窗口时间条**（`TrainingTrafficStrip.tsx`）：横轴是场景时间，跨度取被指挥的飞机（记录与当前读的句子），每架指挥的
  飞机一行、其他飞机合成一行；失去间隔在牵涉的指挥飞机行上画红段（IFR 每段描边、目视实心画在上面），两架都被指挥时连一条竖线；
  焦点那一段括起来。在图上按下或拖动移动场景时间；**按下**一行（飞机条或呼号）换焦点——按在图上会捕获指针，之后的点击到不了那一行，
  按在呼号上不动时间；▶ 按 1× / 10× / 30× 播放（每 50 ms 推进一次，计时器只随"在放、速度"重建；每个窗口一条新的时间条）。
- **游标在不在这架飞机上是一个判断**（`cursorOnFlight`）：单机时总在（越过轴的末端照旧夹住）；窗口时只在这架的轴上——之外时句子条、
  读数与先验窗口、三维都不画游标、不亮游标处的词，读数写 "outside this aircraft"。
- 左栏按集合的**种类**各留一个会话（`TrainingFlightSession` / `TrainingWindowSession`），同种集合加载时不卸下：开关与每个集合选中的叠加层
  切回时恢复。
- **三维**（`useTrainingTrafficLayer.ts`，由 `useTrainingTrackLayer` 调用，读游标，只在叶子组件里）：每架飞机按它的**角色**画
  （`TrainingAircraftRole`：`onScreen` / `commanded` / `replayed` / `background`，一张表 `ROLE_DRAW`；用户 2026-09-30：分不出哪条
  是正被控制的）。焦点（句子条读的那架）的航迹由单机图层画，游标处一个大白点、外圈是当前读法的颜色，呼号 "▶ …" 写在该颜色的
  底块上；**被指挥的是一个集合**（以后多架同时现飞，只是这个集合里放更多，没有地方假定只有一架）：航迹 2.5 px 不透明、点与呼号用
  当前读法的颜色（模型色，记录时观测航迹的近白色）；回放的石板灰 1.2 px、透明 0.6，背景的更暗更淡（`TRAINING_OTHER_AIRCRAFT_ALPHA`）。
  正处在失去间隔里的一对连红线并标"最近 / 要求"米数（目视实线，只有 IFR 有的虚线），被结束处一个固定的大红叉。时间条上焦点那一行
  的呼号前也有 ▶；图例在窗口时多五行（焦点、被指挥、回放、背景、失去间隔）。**取景**：一个时钟取景
  一次——单机是这架航班，窗口是整个窗口的全部航迹，换一架不再取景。
- 颜色：模型 `traffic`（M4，用户 2026-09-30 定名）天蓝 `#2b93ee`（用户同日嫌原来的橄榄金 `#9c8116` 不清新，换掉；对已有颜色
  OKLab ΔE ≥ 11.5，最近是航向带的蓝，对底色 5.8:1）；窗口判定三色沿用回放的青 / 琥珀 / 红（`TRAINING_WINDOW_VERDICT_COLOR`）；
  其他飞机 `TRAINING_OTHER_AIRCRAFT_COLOR`；失去间隔与被结束用失败大红（AV40）。

### AV40 · 失败一律大红；SVG 文字的颜色写在 style 里（2026-09-30）

- 用户（2026-09-30）：降落失败的航迹，看板上的粉红虚线不够醒目，换大红。**一个失败色** `TRAINING_FAILURE_COLOR = #ff2d2d`
  （`trainingWordColors.ts`；对底色 5.0:1）：实时执行器飞出包络的整段（`TRAINING_AUTOPILOT_OUTSIDE_COLOR` 就是它）、句子条上模型
  航班**没落地**时的结束线（3.5 px，落地的仍是模型色 2 px）与轴下的时刻、样本小块的文字、失去间隔（`TRAINING_LOSS_COLOR`，时间条
  上目视实心不透明 0.85、IFR 3–2 虚线，线宽 1.5）、被结束的 ✕（时间条上 14 px、黑描边、画在失去间隔段之上；三维 26 px）。逐行的
  "包络之外"仍是浅红 `TRAINING_OUTSIDE_COLOR`。
- **坑：SVG 元素上的 `fill` 属性会被样式表里的 `fill` 盖掉**（属性的优先级低于任何 CSS 规则）。时间条的 ✕ 原来被
  `.training-traffic-row text { fill: … }` 盖成灰色，句子条轴下模型航班的结束时刻被 `.training-sentence-tick` 盖成灰色——都从来没
  显示成设计的颜色。颜色随数据变的 SVG 文字写 `style={{ fill }}`，或让样式表的规则只选不带颜色的那几个元素（时间条的呼号现在是
  `.training-traffic-callsign`）。

### AV25 · Experiments 里的执行器回放：横轴模式 `sentence`

- 根目录的发布器 `--executor-replay` 把执行器的 val 回放记录（和 ts 预测同一个记录契约）发布成 Experiments 类别，每个机场
  一个（`experiment_executor_v2_20260924_2674ab8c71a9_val`），挂在 `intents.json` 的 `executor_val_replay_20260924` 下。
- 这些记录的 `horizon_mode` 是 `sentence`（飞完整句话），比较 CZML 的生成器把它写进类别的 `experiment.horizonMode`；前端的
  `EXPERIMENT_HORIZON_MODES` 在 `config.HORIZON_MODES` 之后加上它（`test_frontend_mirrors.py` 钉住，与
  `executor_replay.HORIZON` 比对）。不加的话，一个这样的类别就会让整个机场的选择器变空（AV6）。所以**先合并这个镜像，
  再发布**：正在运行的前端还不认 `sentence` 时发布，它的选择器立刻变空。
- 类别里是这个机场全部被飞的航班，自己机型动力学和 A320 替代动力学两组都在；标签写出两组各多少架，参数表写出规格 sha
  和每个参数。第一个周期就动力学失败的航班没有记录，不在类别里（参数表写出有几架）。

### AV26 · Training 的实时执行器：选中一个词，后端现飞它的一段

- 用途是验证执行器，所以**每次选中都现飞**：`POST /autopilot/segment`（`aeroviz_backend/autopilot_segment/` 包：`segment`、
  `verdict`、`fly`、`payload`、`backend`、`errors`），不读执行器的
  正式回放，也不读叠加层。**飞的是单条执行器**（`single.py`，2026-09-27，用户要求）：执行器一个周期的计算照 torch 执行器逐个
  表达式写成一架航班的纯 Python 浮点数——200 个周期约 9 ms，原来 torch 执行器一次一架约 0.55 s。一致按结果核对（逐位相同做不到：
  torch 的 atan2、hypot 同一架航班批量飞和单独飞就差最后一位）：`test_single_executor.py` 的合成航班与 `check_single` 对训练视图全部
  航班的每个词段，结局、每个词的判定、越过入口相同，状态只差舍入。它不进执行器规格的源码指纹，而是钉住它照写的代码
  （`MIRRORED_SOURCE_SHA256`：执行器规格的文件加它照写的动力学模块，按 `spec.logic` 算），后端选规格时核对，对不上就拒绝飞。
  按 `executor.fly` 的方式一个周期一个周期地飞（`fly_until`），在词钟把一个开始一步的周期放到段尾之前停下，段尾之后
  什么也不飞。
- 启动：选中一个词后按句子条头部的 **▶ Fly**（飞完变 **↻ Fly again**，同一选择的新一次尝试），或直接点色块（2026-09-29 起总是
  如此：面板的 "Autopilot (live)" 一栏连同开关 "Fly on band click" 删掉，AV38）；只有点击才请求（`trainingPick`），游标不触发，
  图表悬停会移动游标。
- **模型的词**（句子条读模型的样本时）：请求带上这句话（`sentence`：`overlayId`、`sample`、`firstRow`、`rows`、`events`，
  `trainingAutopilotRequest` 从屏幕上的样本取，`useTrainingAutopilot`），后端把整句从观测飞机在 `firstRow` 的状态照自由生成的
  飞法重飞（`segment.model_segment` / `fly.fly_segment`）：每个词在它说出的那一步听到（`TimeClock`）、时限 = 观测从 `firstRow`
  起剩下的时间 × 超时倍数（`fly.model_time_limit_s`，`prior_free_generation.limits_s` 的镜像，测试钉住）、判决按模型指的跑道；
  答复只给从词那一步起的航迹与判定。执行器是确定的，所以这就是导出样本自己的飞行：逐点比过（两者都有点的时刻，水平与高度取大；
  比较用的 `autopilotSampleGap` 随结果卡在 2026-09-29 删掉，AV38），2026-09-26 KRDU / KMSY 两个模型各 4 架 × 2 个样本、104 段
  最大差 0.000 m；测试
  `FreeGenerationTest` 让 `speak_and_fly` 自己的循环（照稿说话的说话者代替先验）和后端的飞法在真实执行器上逐状态比。只飞自己机型
  动力学的航班；词说出时或之前航班已结束的按名字拒绝（400，与前端 `row·step < endS` 同一个条件），标注器的门把航迹截在词之前的
  也拒绝；说在句子最后一步的词飞到结局（模型的最后一步不是落地）；句子从 `N_LOOK` 开口，步数不超过时限下能说的
  （`model_steps_max` = `rows_for`）；判决按句子最后指的跑道（与导出样本的结局相同）。下降角词：判决读的句子在它那一步
  重说生效的高度词（`judged_reading`），管子从它那一步起判，与真值相同；第二次许可不判（判决读第一次的截获）。答复 `source`
  写明飞的哪一句，前端核对；`observedS` 为 null，`offsetFromObserved` 只有真值有；`limits` 是整段重飞的。
  模型许可之后又说的航向词照判决原样判到航迹末尾（执行器在飞航道），读作出界——模型句子的读法，判决不改。
- 一段 = 被选中的那个色块：从词说出的一步飞到它的包络结束的
  `stopRow`——同列下一个词说出的一步，航向词再加一个提前量（它的带判到下一个航向词说出后一个提前量，下一个航向词照句子说出）；
  到了句子末尾就飞到落地，句子最后一步说的词按名字拒绝。**航向词多飞的那一截画成尾巴**：从执行器听到同列下一个词的周期
  （答复的 `segment.nextWordHeardS`，按判决的 `words_said`；前端 `tailFrom` 是它在航迹上的下标，`autopilotRunAndTail` 切开）起，
  三维与四张读数图都画淡色虚线（`AUTOPILOT_TAIL_OPACITY` 0.45、`AUTOPILOT_TAIL_DASH`），图例写出它是什么——飞机已在飞下一个
  词，评判还算这个词（用户 2026-09-26 看成"多飞了一段"）。初态是观测飞机在那一步的状态（`flight_inputs(anchor=row)`），第 0 步是那一步
  六列生效的词，之后是段内的词，每条在执行器到了观测飞机听到它的位置时说。
- 规格：`outputs/POOLED/executor/*/spec.json` 里恰好一份由现在的执行器代码、为这个集合所属产物的词表写的
  （`replay.open_executor`）；否则拒绝并列出每一份的原因；那个目录里的规格一有增、删、移动或改写就重新找。集合必须是这个词表的
  样本格式与读法、从 val 抽的；产物与划分从样本的 `producedBy.artefact` / `cohort.split` 读。
- **只飞一个页面最新的一次**：请求带页面的 `clientId`（`TRAINING_AUTOPILOT_CLIENT_ID`，每次打开页面由 `getRandomValues` 生成
  128 位——不用 `randomUUID`：它只在 https / localhost 下存在，http 打开时应用会白屏）和按页面发出先后递增的 `seq`；后端一次飞一段，
  同一页面编号更大的请求一到，这个页面编号更小的请求——排队的不飞，正在飞的在下一个周期前停下（`fly_until` 每个周期先问
  `superseded()`），晚到的立即拒绝——以 409 结束；以页面的编号为准，不以到达后端的先后为准。连点几个色块只飞最后一个；别的页面不受影响。页面早已断开的请求，后端写回时只记一行
  `client gone`（`http_server._send_json`），不报错。
- 状态码：请求不对（缺字段、没有 `clientId`、列名、不是机场代码的机场、那一步没说这列的词）400（`RequestRefused`）；集合或航班没
  列出 404（`NotListed`）；被同一页面的新请求取代 409（`Superseded`）；机型没有动力学、按数据飞不了 422（`NotFlyable`）；其余列出
  了却飞不了 500，写出原因。
- 航向词的判定：它生效期间（下一个航向词被听到之前）执行器离开它自己去切入航道的，不管有没有可判的行都记为出界；动力学失败时
  航向带照样画（判决读的是失败状态之前的行，航迹保留了这些行）。选中的航向带若被段尾截短（段尾按句子步、带按飞过的步，执行器
  更早听到下一个航向词时前者可能先到），答复按名字拒绝（`payload.band_cut_by_stop`，500）；只查段尾结束的飞行（`ended_at_stop`：
  到了段尾且结局是时限——越过入口没截获后执行器还会接着飞）。段尾与听词都只在开始一步的周期上，所以两步的提前量下听到的
  下一个航向词总放得下；会截短的是没有上限的距离词钟一步越过下一个词、或更长的提前量——2026-09-25 全部 3,526 个航向词试飞，0 次。
- 前端把答复绑到屏幕上这一段：同一架航班、`endRow` 是色块的终点、词表规格相同、**告诉执行器的词就是句子条这一段显示的词**；
  航向带的行是执行器自己飞过的步，以判决读到的飞出航迹为界、不以句子段尾为界（执行器可能晚听到下一个航向词）；
  对不上整份拒读；答复只画在它飞的那一句上（`autopilotOnScreen` 比 `source`）。答复格式 `aeroviz-autopilot-segment-v7` 两边钉住
  （v7：模型的句子带 `augmentation`，增强起点的样本从挪过的起点飞，AV35）
  （v5：模型的句子带它说话时的程序屏蔽 `procedureMasks`——名字和数据摘要，后端重建后核对，不一致按名字拒绝；在程序高度下说的句子
  飞完按自由生成的规则 `fly.glidepath_stop` 截在下滑道下边界的那一步，结局 `below_glidepath`；`GlidepathStopTest` 逐架钉住它与
  `glidepath_stops` 相同）（`SCHEMA` / `TRAINING_AUTOPILOT_SCHEMA`，判定状态与结局
  名也是镜像）；它带 `timing`（后端墙钟：等待；加起来等于总计的各项——集合与规格、重建航班或沿用、准备这一段、执行器与算了的周期数、判定、
  写答复；`flyS` 只是执行器的周期，装配物理量算在"准备"里）。单步飞法由
  `test_autopilot_segment.StepperTest` 钉住：单条执行器在后端的飞法下，不设段尾时就是 torch 执行器的 `executor.fly`（周期、词钟时刻、
  模式与限制相同，状态只差舍入），设了段尾时等于它在词钟首次把一个开始一步的周期放到段尾处截断；装配由 `SetupTest` 钉住（与
  `replay.fly_sentences` 的输入、跑道、图、进近速度、时限、词钟相同）；前端的四个镜像名由 `MirrorTest` 钉住。
- 显示：句子条一行只写**在不在包络内**、"N s flown · computed M ms"（飞得不好时加一个结束方式的短标签），不写词（选中的色块就是它；选中已移到别的词时才写出飞的是哪个词），
  词与完整读法在它的提示里，这一行不换行——头部的按钮不再被它挤到第二行（2026-09-27）；没飞成时这一行红色 "not flown"，原因在
  提示里，按钮变 ↻ Fly again；三维飞机标签走模拟时钟"已飞 / 全段 s simulated"。面板里原来的结果卡 2026-09-29 删掉（AV38）。
- **颜色按判定**：在包络内蓝 `#2563eb`，飞出包络整条换成醒目的红 `#ff2d2d`（`autopilotColour`：三维航迹、地面投影、飞机与
  标签、读数图的线、句子条那一行的判定）。
- 画法：与执行器回放的青色分开；三维里飞机按加速的实际时间把这一段飞出来（至少 8 倍、不超过 20 s，
  CallbackProperty，不碰 `viewer.clock`；飞完换成静态属性，不再每帧重建，被地形挡住的部分也画成虚线），读数窗口里四张图
  各一条蓝线，从观测线上说词的那一点出发。只有一个状态的答复（第一个周期就动力学失败）没有线可画（`autopilotHasLine`）。
- 后端第一次收到这个请求时才载入 torch 与 ts_transformer（常驻内存约多 470 MB）；一次一段（锁）。**一个集合的航班在它第一次被请求
  时一起重建**（`open_flights`：一次 `rebuild_series`，划分的信号、程序文件和到达清单各读一次；约 3 s，只一次），之后留在进程里，
  集合里每架航班的请求都是毫秒级（原来每架新航班重建 1.5–2 s）。
  后端不热更新：改了这部分要重启后端。
