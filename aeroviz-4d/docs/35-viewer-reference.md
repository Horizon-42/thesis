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

### AV19

**一句话画出来的是一串包围盒，不是一条线**，而且**被取代的 Training 产物会被当场拒掉**
（2026-09-21，词表 `segment-v14` → `box-v2-wedge` → **`box-v3`**；本条取代了原来那版"走廊"的写法）。

三件事，都会让人以为界面坏了：

**一、Training 页上某个集合读不了，多半是产物被取代了，不是读者坏了。** 读者按名字钉住读法
（`TRAINING_READING_RULE = "box-v3"`）和 schema（`aeroviz-training-sample-v2`），对不上整份
文件拒绝，界面落到设计 §4.5 的 ③。面板**不会开在**一个读不了的集合上：清单里每个集合都写着自己的
读法，所以哪些是当前的、不用下载任何 sample 就知道；旧集合仍然列着，标成 `superseded`。
`npm run check-publication` 的报错里带上清单写的读法。**不要为此加兼容分支**（仓库规矩：兼容是禁用词）。

**二、一个词是一个区间，三维里画的是区域不是曲线。**

- **高度词是一个目标 + 它的后向可达楔形**（往上张 1.5°、往下张 1.0°，段末收到目标的 ±5 %）。
  它是这份词表里唯一一个**绝对的位置界**，三维里画成 Cesium `wall`
  （`altHaeLoM` / `altHaeHiM`，**HAE**；这一版 `Hi` 就是上界，不像上一版的走廊要反着读）。
  **一个高度段一面墙，不是一架一面**：高度界在每个高度词处跳一下（实测一架引导进近 9 次、最大
  59 m），一面墙硬穿过这些跳，在转弯的航迹上渲染成扭曲的面。每一面在自己段末收到目标的 ±5 %
  再交给下一面，所以那一跳看上去就是一跳。
- **航向词和速度词只给位移界，而那块空间是锥台，不是盒子也不是扇柱**：词表说的"包围盒"是
  **状态空间**里的盒子（航向 × 速度 × 高度）；它在**位置空间**里推出来的可达集，平面上是一个
  **顶点在飞机**、半径 `保持时长 × 速度盒上界`、张角等于航向盒的扇形，**竖直方向跟着收**
  （外轮廓上离顶点 d 米的点，离段末就少了 d 米航迹，楔形因此更窄）。所以它是**锥台**：平面扇形、
  径向剖面梯形。画法是一圈 `wall`（闭回顶点，两个径向面就是那两个梯形）加一张
  `perPositionHeight` 的斜顶盖 —— `polygon` 的 `extrudedHeight` 只有一个数，表达不了斜盖子，
  按一个高度拉平会把远端天花板抬高到盒子自身高度的 26 %（2026-09-21 被用户指出）。
  **不要画它的外接长方体**：长方体在顶点旁边多出一块航向盒里任何方向都到不了的地，看着像可飞
  空间（第一版就是这么画的，2026-09-21 被用户指出）。**速度盒的下界不参与** —— 保持结束前的
  任何一刻飞机都更近，所以扇形一直连回顶点，只有上界定半径。
  实测中位 441 m 深、远端 23 m 宽：这个窄本身就是结论，横向上一个词几乎什么都没说。
- **它是推出来的可达集，不是词本身**：词约束的是每一时刻的状态，不是路径。而且它**里面没有飞机** ——
  没有任何东西限制航向在盒子里转多快，因为词不限制；转弯速率是执行器的规矩，这里没有执行器。
- **没有"按规则飞出来的那条橙线"了**：飞一句包围盒需要跟高度的执行器（回放门，未做）。

**三、判决算在平滑信号上，不是原始行上。** 盒子是从滑动平均上读出来的（航道 6 s、速度和高度 10 s），
同一条航迹算在平滑线上 **100 %** 在盒子里、算在原始行上 **93.9 %**。导出器写 `readCourseDeg` /
`readSpeedMps` / `readHeightM` 三列专供判决，原始列画在它们后面（只画平滑线等于画了一条没人飞过的
信号）。**航向在 box-v3 起是在 UNWRAPPED 上平滑再 wrap 的**：上一版在 wrapped 上平滑，跨 ±180 会把
+179° 和 −179° 平均成 0°，按那个写法重建 box-v3 会有 1.4 % 的行落在盒外、**全在那个断口附近**。
换读法就要重新量一遍每一路信号，不能假定沿用。

**四、`box-v3` 起产物不再说列序**（`spec.kinds` 随 `edges`→`boxes` 的改名一起没了），所以导出器拿
**文件自己的数据**钉：跑道列必须是这架飞机自己的跑道、时长列乘格宽必须等于旁边的 hold、终止列必须
一路继续到最后一个落地。三条钉三列，另外三列由包含率钉。

**标注器在 2026-09-21 进了仓库**（`manoeuvre/box_vocabulary.py`）。在那之前导出器照 spec 重建盒子，
现在改成 import：表和楔形来自 `BoxVocabulary`，spec 回算的 sha 必须等于产物写的那个，而且**每一架都用
`read_boxes` 重读一遍**，句子必须逐词逐时刻相同（句子抄自产物、航迹在这边重建，除此之外没有东西说
它们是同一架）。前端仍然把包含率再算一遍，对不上就拒整份文件。
**重建期间有三处和原件不同**，import 之后才露出来，每一处当时都没让数据变红：楔形的半宽取的是
`|T|`（梯子上 13 个目标是负的，写成 `T` 会得到一个上下界翻过来的盒子）、段末取的是这一段自己的最后
一行、路径积分不加地速下限。

设计与决定表：`aeroviz-4d/docs/36-2026-09-20-training-module.zh.md` §5、V53–V58。

### AV20 · Training 选段与三维盒子高亮联动

`AppContext` 保存航班相对时间 `trainingCursorS`，底部句子条、读数核对窗口和三维图层共用。
点击事件编号或词带（键盘 Enter / 空格同样有效）沿用原来的选时规则：事件自己的时间或词带起点。
`useTrainingTrackLayer` 用 `eventInForce` 找到该时刻生效的盒子，将侧面、顶盖和起点标记改为
`TRAINING_WORD_COLOR` 黄色，并增强填充、放大标记；前一个盒子恢复原来的橙色和透明度。
时间恰好落在新事件上时选新盒子，航迹终点仍选最后一个盒子。

换航班时游标在渲染前归零；关闭再打开区域图层会重新应用当前高亮。选段只改 Cesium 实体样式并
请求重绘，不重建几何，不改相机或共享的 `viewer.clock`。盒子仍遵守 `trainingLayers.flown` 开关。

### AV21 · Training 目标高度参考面

每个事件盒内增加青色半透明目标高度面，平面范围沿用盒子的扇形轮廓，高度固定为
`altitudeTargetM + altHaeLoM[0] - altLoM[0]`：后两项恢复导出器给该盒使用的入口相对高度到 HAE
的偏移。因此参考面落在目标高度上，不取非对称楔形的中点，也不跟随飞机高度。
它随 `trainingLayers.flown` 显隐；选中段时增强青色填充并加黄色边框，与橙色盒体／黄色高亮区分。
左侧 Draw 下说明青色代表相对跑道入口的目标高度。
