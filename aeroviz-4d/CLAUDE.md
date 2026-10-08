# aeroviz-4d — React + CesiumJS viewer (and its Python CZML tooling)

Frontend app plus `aeroviz-4d/python/` (CZML generation, comparison-CZML builder).
The Python tooling here **must not import the modeling tree**; the file that duplicates
modeling logic (`python/flight_identity.py`) is a declared MIRROR — see `flight_scenarios/CLAUDE.md`
before touching it. (The EGM96 mirror `python/vertical_datum.py` had no caller and was deleted on
2026-09-25: the comparison CZML adds each record's own `source.hae_minus_msl_m` back.)

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
npx vitest run src/utils/__tests__/cameraDial.test.ts  # Single test file
npm run test:coverage                # Coverage report
npm run build:local-terrain          # Airport-local heightmap terrain tiles
npm run build:local-terrain:visual-assets
# Does the picker load what was just published? The frontend's own guards over
# categories.json + every comparison_index.json (names the rejected category and field),
# the referenced CZML/report files, every Training set `training/index_v4.json` lists through the Training reader, and
# — with --server — what the RUNNING dev server answers (a served Training sample is parsed too).
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
`playbackSpeed`. Components read context via `useApp()`, which spreads EVERY context — so **a value that changes
per mousemove, tile or slider step is its own context with its own hook, never in what `useApp` spreads**: the
Training cursor (`useTrainingCursor`), the local terrain's tile counts (`useAirportLocalTerrainProgress`), the range
ring radius (`useRangeRingRadiusKm`); read them only where they are shown or in a leaf (AV29).

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

- **Training reads ONE vocabulary: `instruction-v6`, stage A of the two-tier model** — the index `training/index_v4.json` (`aeroviz-training-index-v2`; the old view's `training/index.json` is NEVER read), the set kind `closed-loop-readback`, the sample `aeroviz-training-sample-v9`, the reading rule and the FIVE columns IN ORDER (runway with go-around, heading relative to the course of the runway in force, altitude above E or "no level-off", angle, speed — no approach column) are pinned mirrors (`data/trainingSample.ts`, pinned by `aeroviz_backend/tests/test_autopilot_segment.py::MirrorTest`); any other schema is refused BY NAME, with the schema found and the one expected (`check-publication` too) (AV19, AV45).
- **The Training views compute NO envelope and decode NO word**: every heading band and row verdict, tube and speed span is
  exported (`training_export.envelopes`, the labeller's and the judge's own checks), every word carries what it says
  (`says`, decoded by `Words`); the reader checks only the bookkeeping (AV20). One drawn shape is computed — the heading
  word's fan, a picture of its band; the verdict stays the export's rows (AV49).
- **Training highlights ONE selected word, never a step**: `trainingColumn` (the word class) + the shared
  cursor → `sentenceWordAt`; only that column's word lights up (its own envelope yellow — a line — or its hue deepened with a
  yellow edge — a fill — and its in-force rows), in the sentence bar, the read-back window and 3D alike; every other word's
  envelope recedes (× 0.3); red rows outside never fade (AV21).
- Training in 3D: what bounds position is draped on the ground under the judged track's ground trace — a heading word's judged rows and its FAN (target ± tolerance from its first judged row, the one shape the view computes: a picture, the verdict is the rows'); switches `observed` (the observed track, its aircraft, its chart lines, and the labelled sentence's envelopes) / `headingBands` / `vertical` / `candidates`; the procedure's limits in stage B and C/D sets (AV49); the observed track and the executor's FLOWN path (a closed-loop reading) are drawn beside each other, with the DA point and the words the reading added; the tubes are walls in exported HAE with edge lines; a selected flight is framed once; the charts' axis is flight time (AV22, AV45, AV49).
- **A heading word is a BAND over the rows it is judged on**: from its row plus the 4 s lead on, the track within ±4.5° of its target row by row; drawn as rectangles on the heading chart, its judged rows on the ground in 3D, rows outside red everywhere; the envelope of a closed-loop word is the judge's, found by its row × Δ/2 s (`trainingEnvelopeIndex`) (AV23, AV45).
- **`EXPERIMENT_HORIZON_MODES` = `config.HORIZON_MODES` + the executor replay's `sentence`** — its records' horizon, stamped
  by the comparison builder; unlisted, one executor category would empty the airport's picker (AV25).
- **Training's live executor flies the CLICKED word of the closed-loop sentence on the backend, every time** (`POST /autopilot/segment`, `aeroviz-autopilot-segment-v9`; `trainingPick`, never the hover cursor): request `{clientId, seq, airport, setId, flightKey, rowIntervalS, column, row}` — a column NAME and the word's Δ row in `closedLoop[Δ].words`; the executor is flown again from the sentence's first predicted step to where the word's segment stops (the next word of its column; a column's last word on to the judge's outcome, with the crossing and the DA check); the answer's cycles count 1 s from that step and are put on the flight clock; refused unless it is the segment of the word the bar shows; the pick belongs to the flight at its Δ (`trainingSelectionKey` + `trainingIntervalS`); status line "how it ended · N s flown · computed M ms"; blue `#2563eb`, the failure red `#ff2d2d` when flown on to an outcome other than a landing; a later request from the same page supersedes the earlier (409); backend 400 / 404 / 409 / 500 (AV26, AV45).
- **Training's code: one reader (`data/trainingReader.ts`) for every Training file; shared wording in `data/trainingText.ts`; the read-back is a pure model + four charts + a window shell (`components/training/`); the 3D scene is `scene/trainingEntities.ts`, built once per flight and reading — Draw switches set `show`, they rebuild nothing**; the sentence read is a `TrainingReading` (`trainingReadingOf`: the labelled sentence, or the closed loop at Δ — `trainingIntervalS`); a reading given only in a tooltip is also page text behind an ⓘ (`training/NotesToggle.tsx`) or on the details page (AV27).
- **The Training cursor is its own context (`useTrainingCursor`), which `useApp` does not read — it moves on every chart
  hover; read it only in leaves (`TrainingScene` runs the 3D layer; never a hook in `FlightApp`, or every hover re-renders
  the workbench). The Training panel stays mounted after a visit (`hidden` in other tasks), so its session
  survives a task switch at the same airport (another airport opened elsewhere drops it — no background download) —
  anything drawing Training state must check `mode`** (AV28).
- **The Training docks end ABOVE the sentence bar**: the bar publishes its measured height as `--training-bar-height` on
  the page's root (`document.documentElement`, 2026-09-28: Cesium's credits read it too), `.workbench:has(> .training-sentence-bar)`
  pads the overlay container by it; the flight list takes the leftover dock height (min ~5 two-line rows), the dock scrolls past
  that (AV30).
- **The Training dock never unfolds long content**: what is read once (the module, the switches, the vocabulary) and every
  readout's tables are on the modal DETAILS PAGE (`training/TrainingDetails.tsx`, portalled, a tab per section, a section
  with nothing to show listed disabled with its reason); the dock has a header ⓘ ("Training details") and one line per readout — its name
  and conclusion — opening the page on that section (AV33).
- **Scrollbars are styled ONCE, globally** (top of `index.css`: `--scrollbar-*` tokens, `::-webkit-scrollbar*` for
  Chromium/Safari, the standard properties only under `@supports not selector(::-webkit-scrollbar)` — Chromium 121+ lets them
  override the parts); a component never styles its own scrollbar, it changes the tokens (AV34).
- **Training hides Cesium's clock console** (dial, timeline, full-screen button — Training runs no clock): body class `workbench-training-active` from `mode` (WorkbenchShell), `visibility: hidden` (the viewer's `forceResize` reads it, so the credits are laid out without them); the sentence bar sits at the bottom edge (`--training-bar-bottom`), Cesium's credits are lifted above it right of the left dock (`!important` over the viewer's inline position; `--overlay-left-width` on `:root`); the live executor's cursor on its word's row reads the 3D fly-out's clock (`autopilotPlaybackS`), a leaf moving itself per frame (AV36).
- **The live executor has NO panel block** (2026-09-29, the user: redundant): its answer is the sentence bar's line (`TrainingAutopilotStatus`), the cursor, 3D and the read-back window; a band click of a closed-loop sentence always flies; ↻ Fly again asks anew; the labelled sentence is not flown (AV38).
- **A failure is ONE red, `TRAINING_FAILURE_COLOR` `#ff2d2d`** (a flown flight that did not land, a failed DA check, the live
  executor flown on to an outcome other than a landing); **an SVG `fill` attribute loses to any stylesheet `fill`
  rule** — colour data-driven SVG text with `style={{ fill }}` (two marks were grey for that reason) (AV40).
- **Every Training aircraft is the 3D model in the attitude the Python exporters computed** (`experiments/training_attitude.py`: heading, path angle, right bank, an attack READING; every track's `attitude` block — sample v9 observed and flown tracks, live answer v9); the viewer computes no attitude (`trainingAttitude.poseAt` only interpolates); **the drawn pitch is the path angle, the attack only in the label** (user, 2026-09-30: the lift curve has no flaps); a flight without an airframe has no bank/attack (null: wings level, said); ONE model orientation (`utils/aircraftOrientation.ts`) for Fly, Optimize and Training (AV42).
- **Four tasks, tabs in the user's order Fly ‖ Optimize · Learning · Evaluate ‖ Procedures (`TASK_GROUPS`, one gap
  rule): Evaluation (was Observe; tab label "Evaluate" — name only, mode `evaluation`; its observed source is
  Ground Truth, id `groundTruth`), Learning (only the
  module's NAME — tab and dock heading; it still shows training, so mode `training` and every code/data name stay), Fly,
  Optimize — Compare is part of Fly.** `PilotPanel` takes the workbench task as its mode (no tabs of its own). Fly is one
  aircraft and ONE control set flown two ways: live, or held fixed in the dynamics comparison (load factor only — off under
  Alpha); a control edit drops a computed comparison, and the controls are frozen while a request runs; the two runs hand
  over the screen, the live sim's state (`liveSnapshot`) is never a playback's (`playbackSnapshot`), and the bottom bar
  drives the live sim (`pilotTransport`) or, with a comparison loaded, the clock — which a playback stops when it unloads. The published key `rawKinematics.observedBaseline` keeps
  its name (a Python contract) (AV43).
- **The Camera panel is a drag disc, orbiting the screen-centre point** (`HUD.tsx`, math in `utils/cameraDial.ts`): heading/pitch are
  ACCUMULATED during a drag and never read back from the camera (they differ at long range); the 10 Hz readout is quantised and skips
  re-renders when nothing visible changed; orbit pitch is limited to −89°…−1°; no effect while following a flight (AV44). The layer toggles are `HudLayers`, the HUD's children under it (no top-bar button, no drawer; "Legacy FAF OCS Debug" is gone; Labels is a child of Obstacles — shown only while it is on).

## Comparison CZML colour contract

- Group status is entity `properties.status` ∈ solved/offTarget/failed; **the frontend repaint
  skip is keyed on "a verdict colour was baked", NOT on `status` alone** (AV13).
- **`otherRunway`** = a two-tier generation sentence that passed on ANOTHER runway than the observed flight's (the
  landed-runway grading, `--landed-runway-report`): light sky blue, never the pass green; the frontend must learn a status
  before any index carries it (an unknown status refuses the whole index) (AV41).
- **A traffic window's index group may carry `traffic: {outcome, recorded, startOffsetsS}`** (a `--summary` of mode `traffic:m1`, any other `traffic:` mode refused: its sidecars
  are required for every solved row, its arrivals roster is the one the summary names; no flags): the recorded neighbours as `ref-<flight_key>`
  ids, each with the seconds it enters after the commanded aircraft (no CZML holds an absolute time, so the frontend shifts each neighbour's
  epoch by it, per shown group); drawn pink `COMPARISON_TRAFFIC_COLOR`, the references' model budget, the clock stays the groups' own span and each neighbour's availability is clipped to it on purpose (airborne at the clock start: shown mid-flight; entering after the clock stop: not loaded; no clock: an error); their keys
  are requested in arrival-window requests of at most `MAX_FLIGHT_KEYS_PER_REQUEST` (1000); an unsolved row has no `traffic`; the `categories.json`
  entry carries no new field. **An index of windows is drawn ONE window at a time** — the selected flight's group, the first by default; its
  own reference, result and neighbours, on its own clock; the other windows stay in the Flights table, their neighbours are not fetched
  until shown (drawn together they flew beside another window's controlled aircraft); only a selected GROUP KEY changes the window (Reset view does not).
  Legend: "Controlled aircraft — its record" / "— optimized path" (+ the off-target yellow explained there), "Recorded traffic — not controlled".
  A traffic scene has a **"Scene time (UTC)"** readout (`SceneTimeReadout`: real = `sceneTime.startUtc` + clock − epoch; an M1 window's start is its flight's roster entry) (AV46).
- **An M2 run's index is ONE scene** (a `--summary` of mode `traffic:m2`): index `scene: {startUtc, background: {recorded, startOffsetsS}}`, every group
  `scene: {startOffsetS, outcome, delayS}` (never with `traffic`); the clock starts at the earliest group entry (solved and unsolved), result paths are written
  on it, the frontend shows every group (no sample, no runway filter), shifts each reference by its offset (available only while it flies), takes the clock
  from the groups' spans on the scene clock, and loads the background once (pink, model budget, clipped to the clock); its pink legend row reads
  "…, outside the scheduled set" (AV47).
- **Optimize has a mode selector: Single aircraft (the `PilotPanel`, unchanged) | Multi-aircraft one controlled (M1) | all controlled (M2)**; the multi-aircraft
  panel runs ONE backend job (`/traffic/*`, `aeroviz_backend/CLAUDE.md`), polls it every 2 s, and when it is done the comparison layer — which now has a
  SOURCE (a published category, or the job's files at `<backend>/traffic/jobs/<id>/files/`, `AppContext.trafficScene`) — draws its scene in the Optimize task;
  **a change of mode, another airport or leaving Optimize cancels a running job and removes the scene** (the hook's cleanup; Start is disabled while a job runs; every async continuation is epoch-gated; a failed poll keeps the job and retries with backoff); the PilotPanel stays mounted, `hidden` with its scene off, so no state is lost; a job source ignores the top-bar runway and the sample count and flies the camera to its scene. Outcomes are
  shown by plain names from ONE mapping (`utils/trafficOutcome.ts`; the raw string follows in the title; an unknown one is shown raw); its list, the block
  lengths and the settings are MIRRORS pinned by `aeroviz-4d/python/tests/test_traffic_mirrors.py`. **The scenario is chosen from the airport's catalog, never a date** (`GET /traffic/scenarios`, schema v2, written offline by `traffic_scenarios.py`; the dock is 242 px, so rows are two-line CARDS, not tables: M1 `callsign · type · runway` over `landing UTC · n seconds in loss`, light aircraft hidden by default; M2 the span over `n arrivals · m controllable · n seconds in loss · runways` (runways joined with commas, kept on one line); plain words with a title for the jargon ("loss of separation" spelled out once, in the header; "the records were checked every s s"; "Same settings as the batch experiments"); selects two to a row, labels ≤ 13 characters so none clips; a "Sort by" select, a runway select ("Show runway"), 100 cards at a time; ICAO type codes titled with the plain name where the backend's aircraft catalog has one; a FILTER that hides the pick clears it, a pick the view no longer shows stays named in the Selected line; starting a job leaves the list as set; a missing catalog is the backend's message, shown as it is; `TrafficScenarioList`, `TrafficCard`, `utils/trafficScenarios.ts`; the fields and JSON types read are pinned by a fixture that mirrors the writer). **A finished job's status carries `perAircraft`** (type, losses of the record → first solve → final, landing against the record, delay, and the SOLVER's wall/CPU time and solves, failed ones counted — summed from the sidecars' timed solves, failed sidecars included; the result cards show them — M2's in two labelled stages, "when flown: record → first solve → final" (the census's kind of count; final = the solve the loop KEPT, MD14: fewest counted losses) and "after the whole block" —, "lands … than it really did" with its reason on its own line (more than 60 s earlier: the real flight was vectored; M2 later: its slot; no footnote) and "slot delay … (after its earliest arrival)"; the summary names the aircraft off the landing gates and each yellow aircraft's line names the gates it failed in plain words, read from the job's evaluation report (`not_reached` with how far short; an unknown code shown raw); M2 lists which aircraft carry the block's recorded losses and names the arrivals that stayed their records (`stayedRecords`); a failed sidecar that lost its solves says "time not recorded", never 0 s) **and `timing`** (wall time by stage, shown under the summary), **a running job's progress a `phase`** ("optimizing k of n", shown with the wall time right under the header, above the list, for M1 and M2); **while a scene is shown the animation widget and the timeline's TICK labels read real UTC** (`SceneClockLabels`; the timeline is redrawn with `zoomTo` — `resize()` does nothing at an unchanged size —, and nothing is restored on a destroyed viewer); the job camera frames at margin 1.2 (AV48).
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
