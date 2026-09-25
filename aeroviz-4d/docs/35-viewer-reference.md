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
对不上就整份拒读并报出是哪一项，不做兼容分支。Python 一侧的对应常量（`instruction_training_export.py` 的
`SAMPLE_SCHEMA`、`INDEX_SCHEMA`、`KIND_READBACK`、`WORD_KINDS`，以及 `spec.READING_RULE`、`words.COLUMNS`、
`words.UNCHANGED`）由 `tests/test_instruction_training_export.py` 与 TypeScript 源码逐字比对。

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

`AppContext.trainingCursorS`（航班内的秒数）由句子条、读数核对窗口和三维图层共用；换航班时游标归零。

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
  一次换好（h = H + N，EGM96）。
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

- **叠加层在集合旁边，不在集合里面。** 每个机场的 `training/overlays.json`（`aeroviz-training-overlays-v1`）列出叠加层：
  种类（`executor-replay` / `prior-prediction`）、画在哪个集合上（`base`）、那个集合的样本文件的 sha256、文件位置
  （`training/<叠加层 id>/executor.json` 或 `prior.json`）。集合的索引 `index.json` 不动。写它们的是
  `run_ts.py executor_training_export` 和 `prior_training_export`（ts 的 R13）。
- **绑定**：叠加层文件（`aeroviz-training-executor-v2` / `aeroviz-training-prior-v3`）自己写出所画集合的 id、样本的写出
  时刻和规格 sha，每架航班按集合的顺序一架一条；执行器的每条词与句子的词逐条对应（步、列、值），先验每架的步数等于
  句子的步数。`trainingOverlays.ts` 逐项核对，对不上就整份拒读、说出是哪一项——例如集合按同一个 id 重新导出过，
  样本的写出时刻就不同，叠加层被拒读（"the set was re-exported after the overlay"）。`check-publication` 另外核对
  盘上样本文件的 sha256。执行器的格式 v2（2026-09-24，`instruction-v3`）：每个被判的航向词带上它在飞出航迹上的航向带和
  逐行判定（`heading`），每架被判的航班带上判决读到的飞出航迹（`judgedTrackDeg`：平滑、切到落地前，与 `track.trackDeg`
  同一分支，第 k 步就是 `track` 的第 k 个点，读取时核对）；动力学失败的航班没有这两样（判决读到了导出航迹有意不含的失败状态），
  词的状态与检查照旧。读取器要求：判决判过的每个航向词都带带子、别的词都不带；判定数 = 有判定的词数，离开航向词自己切入的
  那一个词若它的带也判过行则多数一次。v1 的航向判定是转弯与保持段，按名字拒读。
- **界面**（`useTrainingOverlays`）：面板的 Draw 里两个开关（有叠加层时默认开），没有就灰掉并写出生成它的命令；同一种
  有几个时给一个下拉，默认最后列出的。打开时把所选航班的那一条发布为 `trainingExecutor` / `trainingPrior`，与
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

- 用途是验证执行器，所以**每次选中都现飞**：`POST /autopilot/segment`（`aeroviz_backend/autopilot_segment.py`），不读执行器的
  正式回放，也不读叠加层。执行器代码原样使用、不改一行（它的源码 sha 绑着每份执行器规格）；后端用执行器的单步接口
  `Executor` 按 `executor.fly` 的方式一个周期一个周期地飞（`fly_until`），在词钟把一个开始一步的周期放到段尾之前停下，段尾之后
  什么也不飞——与"飞满时限再截断"逐位相同。
- 启动：选中一个词后按句子条头部的 **▶ Fly this segment**（飞完变 **↻ Fly again**，同一选择的新一次尝试），或在面板
  "Autopilot (live)" 一栏的开关开着时直接点色块；只有点击才请求（`trainingPick`），游标不触发，图表悬停会移动游标。
- 一段 = 被选中的那个色块：从词说出的一步飞到它的包络结束的
  `stopRow`——同列下一个词说出的一步，航向词再加一个提前量（它的带判到下一个航向词说出后一个提前量，下一个航向词照句子说出）；
  到了句子末尾就飞到落地，句子最后一步说的词按名字拒绝。初态是观测飞机在那一步的状态（`flight_inputs(anchor=row)`），第 0 步是那一步
  六列生效的词，之后是段内的词，每条在执行器到了观测飞机听到它的位置时说。
- 规格：`outputs/POOLED/executor/*/spec.json` 里恰好一份由现在的执行器代码、为这个集合所属产物的词表写的
  （`replay.open_executor`）；否则拒绝并列出每一份的原因。集合的产物与划分从样本的 `producedBy.artefact` / `cohort.split` 读。
- 前端把答复绑到屏幕上这一段：同一架航班、`endRow` 是色块的终点、词表规格相同、**告诉执行器的词就是句子条这一段显示的词**；
  对不上整份拒读。答复格式 `aeroviz-autopilot-segment-v2` 两边钉住（`SCHEMA` / `TRAINING_AUTOPILOT_SCHEMA`，判定状态与结局
  名也是镜像）；它带 `timing`（后端墙钟：等待；加起来等于总计的各项——集合与规格、重建航班或沿用、准备这一段、执行器与算了的周期数、判定、
  写答复），前端加上浏览器往返时间。单步飞法由 `test_autopilot_segment.StepperTest` 钉住：真实执行器上，不设段尾时与
  `executor.fly` 逐周期相同，设了段尾时等于它在词钟首次把一个开始一步的周期放到段尾处截断。
- 显示：句子条一行只写词、**在不在包络内**、"N s flown in M ms"（飞得不好时加怎么结束的）；结果卡第一行是判定，然后并排
  "模拟飞行时间"（对照观测）与"计算用时"（往返），检查项，其余收进 Details；三维飞机标签走模拟时钟"已飞 / 全段 s simulated"。
- **颜色按判定**：在包络内蓝 `#2563eb`，飞出包络整条换成醒目的红 `#ff2d2d`（`autopilotColour`：三维航迹、地面投影、飞机与
  标签、读数图的线、结果卡）。
- 画法：与执行器回放的青色分开；三维里飞机按加速的实际时间把这一段飞出来（至少 8 倍、不超过 20 s，
  CallbackProperty，不碰 `viewer.clock`），读数窗口里四张图各一条蓝线，从观测线上说词的那一点出发。
- 后端第一次收到这个请求时才载入 torch 与 ts_transformer（常驻内存约多 470 MB）；一次一段（锁），重建过的航班留最近 8 架。
  后端不热更新：改了这部分要重启后端。
