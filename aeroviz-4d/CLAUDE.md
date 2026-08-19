# aeroviz-4d — React + CesiumJS viewer (and its Python CZML tooling)

Frontend app plus `aeroviz-4d/python/` (CZML generation, comparison-CZML builder).
The Python tooling here **must not import the modeling tree**; the two files that duplicate
modeling logic (`python/vertical_datum.py`, `python/flight_identity.py`) are declared MIRRORS —
see `flight_scenarios/CLAUDE.md` before touching either.

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
- `src/data/evaluationReport.ts` exports `EVALUATION_REPORT_SCHEMA_VERSION` as a declared
  MUST-match mirror of `evaluation.metrics.REPORT_SCHEMA_VERSION`; fixtures import it, never
  restate it (see `evaluation/CLAUDE.md`).

## Build config

- Requires `VITE_CESIUM_ION_TOKEN` in `.env` (Cesium Ion access token)
- Vite config uses `vite-plugin-cesium` (asset copying, `CESIUM_BASE_URL`)
- TypeScript strict mode (strict null checks, noUnusedLocals, noUnusedParameters)
- Test environment: jsdom with vitest globals
- `aeroviz-4d/public/data` is **git-ignored** (local artifacts; regenerate via preprocess scripts)

## Gotchas (recurring, verified)

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
- **`.flight-ops-panel` has `backdrop-filter`** → it becomes the containing block for
  `position:fixed` descendants AND clips overflow; floating windows must render via React portal
  into `document.body`.
- **All aircraft CZML sets `forwardExtrapolationType:"HOLD"`** — `position.getValue` returns the
  frozen final position forever forward; any outward time-walk must stop when the position stops
  changing, not only on null.
- **Cesium `Clock.tick()` LOOP_STOP wrap preserves overshoot**
  (`currentTime = startTime + (currentTime − stopTime)`); use `clock.onStop` (fires at the stop
  time for both CLAMPED and LOOP_STOP) for exact end-of-playback emits — never elapsed-based
  heuristics.
- **CZML document clock intervals must use `iso_ms`** — second-precision `iso()` truncation made
  the clock stop up to 1 s before the last sample (~25–75 m phantom position error).
- **Comparison-overlay entities are TIME-WINDOWED — an empty scene is not a broken overlay.**
  Each group only shows inside its own availability interval, so at a clock time outside it the
  map is legitimately blank. Pause inside a window before diagnosing (≈08:09 UTC on the KRDU data).
- **The comparison reference must be requested on the `arrival` track window**, not the full
  track — see `aeroviz_backend/CLAUDE.md` (median 5055 m of apparent "model error" otherwise).

## Comparison CZML colour contract

Group status lives on entity `properties.status` ∈ solved/offTarget/failed. Reference: white /
dark-red (failed) / dark-amber `OFF_TARGET_REF_COLOR` (off-target); simulator/result path bakes
bright yellow `OFF_TARGET_COLOR` (255,205,40) + "(off target)" name; optimizer plan keeps legend
orange/cyan. **The frontend repaint skip is keyed on "a verdict colour was baked", NOT on
`status` alone** — reference always, plus off-target optimizer/simulator paths.

`build_scenario_comparison_czml.states_schema` dispatches on the record keys
(`optimizer_states`/`simulator_states` → `opt-` + `sim-` entities; `predicted_states` +
`observed_states` → `pred-` (purple `PREDICTION_COLOR`, kind `predicted`) **plus `look-`**
(same RGB at alpha 85 `LOOKBACK_COLOR`, kind `lookback`) — see the anchor-shift gotcha in
`4dTrajectory/ts_transformer/CLAUDE.md`).

**Predictions never get the off-target bake** (`mark_off_target = off_target and schema ==
"optimizer"`): a forecast essentially always misses the 106.75 m gate, so marking it repainted
27/27 groups yellow and the kind colour was never visible. Their `status` stays accurate and they
ARE repainted from the legend — so `PREDICTION_COLOR` and the TS legend entry are not required to
agree.

**`look-` takes its forecast's verdict colour, faded — never a hue of its own.** The frontend
paints BOTH prediction halves from the group status (pass green / fail red / indeterminate gray);
the input window is separated from the forecast by `COMPARISON_KIND_ALPHA.lookback` (85/255)
alone. The purple `COMPARISON_KIND_COLORS.predicted`/`.lookback` is only the no-verdict fallback.
A distinct input hue reads as a third kind of result rather than as the first half of one track,
which is why this is a contract and not a preference. The builder still bakes purple into both —
that divergence is a known open item (see the README's "Future Improvements").

## Open items

- **Local terrain and aircraft CZML disagree by ~33 m in the viewer.** `local-terrain` heightmaps
  come from USGS TNM DSM (NAVD88 ≈ MSL) and the metadata records
  `vertical: "Source GeoTIFF elevation values, used directly as metres"` — no datum handling —
  while Cesium expects ellipsoidal heights and the aircraft CZML correctly supplies them. Found
  while chasing the datum bug; not investigated further.
- Approach view: the Observe 3-colour comparison overlay is a separate datasource not yet fed to
  the view (Observe-with-comparison plots neither source); the pre-existing `useCzmlLoader` clock
  write is still ungated for the Observe+comparison two-writer case.
- Approach-view interior-gap `break` is latent (current CZMLs are single-interval); the 07-07
  approach-view changes were verified via tests/tsc/build but not re-checked in-browser.

## Comparison CZML is split by group count, not by runway alone

- **One CZML per runway stops working at thesis scale.** At 38–54 KB of CZML per flight a
  2,000-flight runway is a single ~100 MB file, and the viewer `JSON.parse`s a whole file to
  show even one sampled group (the existing KRDU 23R prediction CZML is already 153 MB).
  `build_scenario_comparison_czml.py --max-groups-per-czml N` splits each runway into
  `comparison_<ICAO>_<RW>_pNNN_<generation>.czml`.
- **The split is transparent to the frontend by construction**: every
  `comparison_index.json` group record carries its own `czml` field and
  `selectComparisonGroups` derives the file list from those (`[...new Set(groups.map(g =>
  g.czml))]`). `prune_unreferenced_outputs` keeps files by the same set, so chunking needs no
  change on either side.
