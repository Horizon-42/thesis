# aeroviz-4d — React + CesiumJS viewer (and its Python CZML tooling)

Frontend app plus `aeroviz-4d/python/` (CZML generation, comparison-CZML builder).
The Python tooling here **must not import the modeling tree**; the two files that duplicate
modeling logic (`python/vertical_datum.py`, `python/flight_identity.py`) are declared MIRRORS —
see `flight_scenarios/CLAUDE.md` before touching either.

**Gotchas, the colour contract and the CZML split are indexed below**: each line ends in the ID
of its full text in `docs/35-viewer-reference.md` (moved there verbatim 2026-09-16, with a dated
correction). A new fact gets a new ID there and one line here. Open items (local terrain vs
aircraft CZML ~33 m datum gap, the approach view): repo `docs/open-items.md`, "Viewer".

## Commands

```bash
cd aeroviz-4d
npm install
npm run dev                          # Vite dev server with HMR
npm run backend                      # the Python backend, from here
npm run build                        # tsc + vite production build
npm test                             # Vitest (watch mode)
npx vitest run                       # Single run, no watch
npx vitest run src/utils/__tests__/ocsGeometry.test.ts  # Single test file
npm run test:coverage                # Coverage report
npm run build:local-terrain          # Airport-local heightmap terrain tiles
npm run build:local-terrain:visual-assets
# Does the picker load what was just published? The frontend's own guards over
# categories.json + every comparison_index.json (names the rejected category and field),
# the referenced CZML/report files, every readable Training set through the Training reader (and every overlay
# drawn over one, against its sample's sha256), and — with --server — what the RUNNING dev server answers (a served
# Training sample or overlay is parsed too).
npm run check-publication -- --airport KSMF --server http://localhost:5173   # all airports if no --airport
npm run check-publication -- --airports-root /tmp/export   # another airports directory (not with --server)
npm run typecheck:scripts            # tsc over scripts/ (outside tsconfig.json's `src` include)

# Python side
pip install -r python/requirements.txt
python -m pytest python/tests/test_generate_czml.py -v
python -m pytest python/tests/ --cov=. --cov-report=html
```

## State & component structure

Global state lives in `AppContext` (context + useState, no Redux). Key state: `viewer` (CesiumJS
Viewer instance), `airport` config, `selectedFlightId`, `layers` visibility toggles,
`playbackSpeed`. Components read context via `useApp()`.

CesiumJS logic is encapsulated in custom hooks:
- `useCesiumViewer` — initializes Viewer, loads airport.json, sets camera
- `useCzmlLoader` — loads CZML data source, syncs Cesium clock
- `useRunwayLayer` / `useTerrainLayer` — data layer management
- `useAirportLocalTerrainLayer` — loads preprocessed `.f32` heightmap tiles via
  `terrain/airportLocalTerrain.ts`; returns `{ status, metadata, provider, error, … }`;
  controlled by the `layers.airportLocalTerrain` toggle

UI components (ControlPanel, HUD, FlightTable) overlay on the Cesium canvas via CSS grid with
`pointer-events: none`.

## Utility modules

- `ocsGeometry.ts` — pure PANS-OPS obstacle clearance surface math
- `czmlBuilder.ts` — pure CZML packet construction helpers
- `utils/procedureGeoMath.ts` — the single TS geo/units module (constants imported from generated
  `geoConstants.json`; regenerated from `geokit` — see `geokit/CLAUDE.md`)
- `src/data/evaluationReport.ts` — `EVALUATION_REPORT_SCHEMA_VERSION` is a declared MUST-match
  mirror of `evaluation.metrics.REPORT_SCHEMA_VERSION` (fixtures import it, never restate it); the
  reader also shows older reports behind a banner: `LEGACY_…` (v5, pre-speed-gate) and
  `PRIOR_SPEED_GATE_…` (v6–v8). Published reports on disk mix v5–v9, so v6-only fields stay
  optional in the TS types (AV1).
- `utils/trajectoryResultSources.ts` → `categoryResultSource` is the ONE classifier splitting
  categories into `optimization | prediction | experiment` — don't re-derive it locally (AV2).
- **Experiments picker = `ExperimentPicker` + `ExperimentDetails`**, reading the publisher-stamped
  `experiment.runName / variantLabel / intent / parameters` — optional, but SHAPE-checked when
  present, so a malformed one empties the airport's picker (AV3).

## Build config

- Requires `VITE_CESIUM_ION_TOKEN` in `.env` (Cesium Ion access token)
- Vite config uses `vite-plugin-cesium` (asset copying, `CESIUM_BASE_URL`)
- TypeScript strict mode (strict null checks, noUnusedLocals, noUnusedParameters)
- Test environment: jsdom with vitest globals
- `aeroviz-4d/public/data` is **git-ignored** (local artifacts; regenerate via preprocess scripts)

## Gotchas (recurring, verified)

- **Vite must never watch `public/data`** (`vite.config.ts` → `server.watch.ignored`): ~40k tiles
  exhaust `fs.inotify.max_user_watches` and vite dies with `ENOSPC`; `Port 5173 is in use` plus a
  restart streak means a stale dev server is alive — `ss -ltnp | grep 5173` (AV4).
- **…and the price of that ignore: a RUNNING dev server never sees a newly published category**
  (it 404s into the SPA fallback: "Expected JSON … but received HTML"). Restart the frontend after
  publishing NEW categories — **kill the `vite` NODE process, not the `npm run dev` wrapper**, or
  the old process keeps 5173 and the new one binds 5175 (AV5).
- **An EMPTY picker on every airport after a publication is the manifest validator, not the
  server** — ONE category with an unlisted `predictionOutput` fails `.every(isComparisonCategory)`;
  run `npm run check-publication`; the mirror is pinned by `test_frontend_mirrors.py` (AV6).
- `.flight-ops-panel` has `backdrop-filter` → floating windows must render via a portal into
  `document.body` (AV7).
- All aircraft CZML sets `forwardExtrapolationType: "HOLD"` — an outward time-walk must stop when
  the position stops changing, not only on null (AV8).
- Cesium `Clock.tick()` LOOP_STOP preserves overshoot — use `clock.onStop` for exact end-of-playback
  emits (AV9).
- CZML document clock intervals must use `iso_ms` (second precision stopped the clock up to 1 s
  early) (AV10).
- Comparison-overlay entities are TIME-WINDOWED — an empty scene is not a broken overlay; pause
  inside a window before diagnosing (AV11).
- The comparison reference must be requested on the `arrival` track window — see
  `aeroviz_backend/CLAUDE.md` (AV12).

- **Training reads ONE vocabulary: `instruction-v3`** — the sample schema (`aeroviz-training-sample-v7` since
  2026-09-24: heading words as per-row bands, no turn regions / funnels; a format name changes with its file's shape, both
  sides in one change), the reading rule, the spec sha (STILL instruction-v2's `103a6eae6b90` until the v3 spec is written:
  every v3 set is refused by it until then), the six columns IN ORDER and the labeller's word kinds are pinned mirrors,
  refused by name; `training/index.json` keeps its v1 schema across vocabularies, so a set of another vocabulary
  (`instruction_v1`, `instruction_v2`) stays listed and is refused from the manifest alone, never downloaded
  (`check-publication`: a warning) (AV19).
- **The Training views compute NO envelope**: every heading band and row verdict, capture turn, corridor, tube and speed
  band is exported from `instructions/display.py` (`envelope.heading_word_rows` / `heading_words_inside`, checked against
  the labeller's counts), every verdict is `Reading.checks`; the reader checks only the bookkeeping (AV20).
- **Training highlights ONE selected word, never a step**: `trainingColumn` (the word class) + the shared
  cursor → `trainingWordAt`; only that column's word lights up (its own envelope yellow — a line — or its hue deepened with a
  yellow edge — a fill — and its in-force rows), in the sentence bar, the read-back window and 3D alike; every other word's
  envelope recedes (× 0.3); red rows outside never fade (AV21).
- Training in 3D: what bounds position is draped on the ground under the track's ground trace — a heading word's judged rows,
  the capture turn's rows, the corridor; switches `headingBands` / `corridor` / `vertical` / `candidates`; a legend lists what
  is on; the tubes are walls in exported HAE with edge lines; a selected flight is framed once; the altitude chart's axis is
  the distance flown (AV22).
- **A heading word (instruction-v3) is a BAND over the rows it is judged on**: from its row plus the 4 s lead to the next
  heading word's, never past the clearance, the track within ±4.5° of its target row by row; a word the lead carries to the
  clearance has no row of its own (not drawn, not judged); drawn as rectangles on the heading chart, its judged rows on the
  ground in 3D, rows outside red everywhere; the capture turn is its rows, clearance → capture (AV23).
- **Training overlays sit BESIDE a set, never in it**: `training/overlays.json` (`aeroviz-training-overlays-v1`) lists the
  executor's replay (`aeroviz-training-executor-v2`: each judged heading word's band on the flown rows + `judgedTrackDeg`) and
  the prior's predictions (`aeroviz-training-prior-v1`), each bound to its set by id, the sample's `writtenUtc` and spec (and
  on disk its sha256) and flight by flight — refused whole on any mismatch; published as `trainingExecutor` /
  `trainingPrior`, apart from the selection; the executor's words are judged on envelopes re-drawn from where IT heard them,
  its lines and bands run on its own clock; the prior is teacher-forced (AV24).
- **`EXPERIMENT_HORIZON_MODES` = `config.HORIZON_MODES` + the executor replay's `sentence`** — its records' horizon, stamped
  by the comparison builder; unlisted, one executor category would empty the airport's picker (AV25).

## Comparison CZML colour contract

- Group status is entity `properties.status` ∈ solved/offTarget/failed; **the frontend repaint
  skip is keyed on "a verdict colour was baked", NOT on `status` alone** (AV13).
- `states_schema` dispatches on record keys: `opt-`/`sim-` entities, or `pred-` plus `look-` for
  predictions (AV14).
- **Predictions never get the off-target bake** (`mark_off_target = off_target and schema ==
  "optimizer"`) (AV15).
- **`look-` takes its forecast's verdict colour, faded — never a hue of its own**; this is a
  contract, not a preference (the builder still bakes purple: a known open item) (AV16).

## Comparison CZML is split by group count, not by runway alone

- One CZML per runway stops working at thesis scale; `build_scenario_comparison_czml.py
  --max-groups-per-czml N` writes `comparison_<ICAO>_<RW>_pNNN_<generation>.czml` (AV17).
- The split is transparent to the frontend: every `comparison_index.json` group carries its own
  `czml`, and pruning keeps files by the same set (AV18).
