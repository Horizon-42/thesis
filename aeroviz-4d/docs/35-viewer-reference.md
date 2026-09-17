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
