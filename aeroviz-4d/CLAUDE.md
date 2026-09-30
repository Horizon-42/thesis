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
npx vitest run src/utils/__tests__/ocsGeometry.test.ts  # Single test file
npm run test:coverage                # Coverage report
npm run build:local-terrain          # Airport-local heightmap terrain tiles
npm run build:local-terrain:visual-assets
# Does the picker load what was just published? The frontend's own guards over
# categories.json + every comparison_index.json (names the rejected category and field),
# the referenced CZML/report files, every readable Training set through the Training reader (and every overlay
# drawn over one, read against its sample), and — with --server — what the RUNNING dev server answers (a served
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

- **Training reads ONE vocabulary: `instruction-v3`** — the sample schema (`aeroviz-training-sample-v8` since 2026-09-30, the attitude beside the signals, AV42; v7 since
  2026-09-24: heading words as per-row bands, no turn regions / funnels; a format name changes with its file's shape, both
  sides in one change), the reading rule, the spec sha (`145d6911e75b` since 2026-09-25: the day-split `v4_20260924` artefact
  today's executor flies — the flight-split `instruction_v3` set is refused by it; current set `instruction_v3_day_split`), the six columns IN ORDER and the labeller's word kinds are pinned mirrors,
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
- **Training overlays sit BESIDE a set, never in it**: `training/overlays.json` (`aeroviz-training-overlays-v2`) lists the
  executor's replay (`aeroviz-training-executor-v5`, v5 = the attitude, AV42: a crossing names its runway; each judged heading word's band on the flown rows + `judgedTrackDeg`) and
  the prior's predictions (`aeroviz-training-prior-v4`: from `firstPredictedRow` on; the rows before are observed only), each
  bound to its set BY CONTENT, never by the sample file's bytes or `writtenUtc` (2026-09-28): the set id, spec,
  `candidatesSha256` and `airportFrame`, and flight by flight (keys, the executor's words, the prior's steps, every drawn
  height's HAE − MSL = the set flight's, 0.02 m, a track flown from the observed state starting AT that observed row) —
  refused whole on any mismatch; a set exported again with the same
  flights keeps its overlays; published as `trainingExecutor` /
  `trainingPrior`, apart from the selection; the executor's words are judged on envelopes re-drawn from where IT heard them,
  its lines and bands run on its own clock; the prior is teacher-forced (AV24).
- **`EXPERIMENT_HORIZON_MODES` = `config.HORIZON_MODES` + the executor replay's `sentence`** — its records' horizon, stamped
  by the comparison builder; unlisted, one executor category would empty the airport's picker (AV25).
- **Training's live executor flies the CLICKED word's segment on the backend, every time** (`POST /autopilot/segment`,
  `aeroviz-autopilot-segment-v8` — v8 the attitude, AV42; `trainingPick`, never the hover cursor): from the observed state where the word is said to
  where its envelope ends (the next word of its column; a heading word's a lead later — drawn as a faded, dashed TAIL from
  where the executor heard the next heading word, `nextWordHeardS` / `autopilotRunAndTail`; the sentence's end: to the landing),
  the executor's own stepper driven cycle by cycle and stopped there (never fly-then-cut); the answer is refused unless the
  words it told are the sentence bar's for that segment; a heading band is bounded by the FLOWN track its judge read, never
  by the sentence's stop (the executor may hear the next word late); started by the sentence bar's "▶ Fly" (or a band
  click); the pick and the cursor belong to the flight on screen (`trainingSelectionKey`) and reset
  with it; one short unbroken status line (inside/outside · "N s flown · computed M ms"; the word only once the selection moves off it; the full reading in its tooltip); blue `#2563eb` inside its envelope, the
  whole line a loud red `#ff2d2d` outside (`autopilotColour`), never the replay's teal; each request names its page
  and its number there (`clientId`, `seq`): a later one from the same page supersedes the earlier still waiting or flying
  (409), so clicking through bands flies only the last; backend 400 / 404 / 409 / 422 / 500. A MODEL's word (the bar reading
  a sample) is asked WITH its sentence and flown as its free generation flew it — the whole sentence from the observed state at
  `firstRow`, each word at its own step (`TimeClock`), the generation's time limit, judged on the sentence's last runway — so it IS the
  exported sample's flight (checked point by point: 0.000 m over 104 segments, 2026-09-26); `source` echoed, no observed
  time or offset (AV26).
- **Training's code: one reader (`data/trainingReader.ts`) for every Training file; shared wording in `data/trainingText.ts`;
  the read-back is a pure model + four charts + a window shell (`components/training/`); the 3D scene is
  `scene/trainingEntities.ts`, built once per flight — Draw switches set `show`, they rebuild nothing**; a reading given only
  in a tooltip is also page text behind an ⓘ (`training/NotesToggle.tsx`) or on the details page (AV27).
- **The Training cursor is its own context (`useTrainingCursor`), which `useApp` does not read — it moves on every chart
  hover; read it only in leaves (`TrainingScene` runs the 3D layer; never a hook in `FlightApp`, or every hover re-renders
  the workbench). The Training panel stays mounted after a visit (`hidden` in other tasks), so its session
  survives a task switch at the same airport (another airport opened elsewhere drops it — no background download) —
  anything drawing Training state must check `mode`** (AV28).
- **The Training docks end ABOVE the sentence bar**: the bar publishes its measured height as `--training-bar-height` on
  the page's root (`document.documentElement`, 2026-09-28: Cesium's credits read it too), `.workbench:has(> .training-sentence-bar)`
  pads the overlay container by it; the flight list takes the leftover dock height (min ~5 two-line rows), the dock scrolls past
  that (AV30).
- **Training reads the models' OWN sentences** (`prior-generation` overlays, `aeroviz-training-generation-v5`, written by ts
  `prior_generation_training_export`): `trainingSource` (null = truth, or `{overlayId, sample}`) picks the sentence the bar,
  the panel and 3D read; the truth is ALWAYS drawn in 3D; a model's bands are FLAT like the truth's (hatching cluttered the
  bar — the user, 2026-09-26): the frame says whose (border, a strip down the rows, tab, chip in its colour), the truth's issues
  ticked under each row, its flight's end time written on the axis in its colour (AV31).
- **A model is NAMED, never labelled: base / landing / augmented / traffic + round** (`TRAINING_MODEL_NAMES`, a mirror of the exporter's
  `MODEL_NAMES`; the payload's `model` block, refused unless base ⇔ no round ⇔ no `fineTuning`); `trainingModelGroups` is the ONE
  grouping — by name in training order, then run, a run's rounds in order; a run named when its stage has two, a round exported
  twice named by overlay id — and every view prints its `title` / `memberLabel`, never a name of its own; the bar: a tab per
  model, the read model's rounds as `r1 r2 …` chips (a round keeps the sample number); colour by NAME (`TRAINING_MODEL_COLOR`:
  base `#d946ef`, landing `#a3e635`, augmented `#b82e7a` — 3.3:1, a mark colour, not for running text — traffic `#2b93ee`); **a model speaks under
  its OWN procedure's masks** (`generation.procedureMasks`); under the procedure's altitudes a sentence may end
  `below_glidepath` (`TRAINING_FREE_OUTCOMES`), and the live executor sends the masks back and is cut at the same step (AV32).
- **The Training dock never unfolds long content**: what is read once (the module, the switches, the vocabulary) and every
  readout's tables are on the modal DETAILS PAGE (`training/TrainingDetails.tsx`, portalled, a tab per section, a section
  with nothing to show listed disabled with its reason); the dock has a header ⓘ ("Training details") and one line per readout — its name
  and conclusion — opening the page on that section (AV33).
- **Scrollbars are styled ONCE, globally** (top of `index.css`: `--scrollbar-*` tokens, `::-webkit-scrollbar*` for
  Chromium/Safari, the standard properties only under `@supports not selector(::-webkit-scrollbar)` — Chromium 121+ lets them
  override the parts); a component never styles its own scrollbar, it changes the tokens (AV34).
- **A model's sentences from AUGMENTED starts are a kind of their own** (`prior-generation-augmented`,
  `aeroviz-training-augmented-generation-v3`, exporter `--augment-seed`): each flight's move and moved observed rows 0–7, the
  same move for every model of one seed, no readout; the bar's `Real start | Augmented start` switch filters the model tabs by
  start (group per start family, never mixed); no truth marks over a moved sample; its words fly live FROM the moved start
  (the request's `augmentation`, `fly.moved_inputs` = the generation's `augmented_inputs`, limit ×2) (AV35).
- **Training hides Cesium's clock console** (dial, timeline, full-screen button — Training runs no clock): body class
  `workbench-training-active` from `mode` (WorkbenchShell), `visibility: hidden` (the viewer's `forceResize` reads it, so the
  credits are laid out without them); the sentence bar sits at the bottom edge (`--training-bar-bottom`), Cesium's credits are
  lifted above it right of the left dock (`!important` over the viewer's inline position; `--overlay-left-width` on `:root`); the
  angle row is named "Descent", its bands "1: 0.92°" (`trainingBandLabel`); the live executor's cursor on its word's row reads the
  3D fly-out's clock (`autopilotPlaybackS`), a leaf moving itself per frame (AV36).
- **The replay's verdict on a flight is ONE rule** (`replayVerdict`, beside `executorWordCounts`): clean / flawed (landed with
  words out — the GATE's count, as the list's "43/45" — or its track refused) / not landed, coloured by `TRAINING_REPLAY_COLOR`
  (teal / amber `#f59e0b` / red) in the flight list's tag and a header chip "Replay · 2 words out" shown ONLY when not clean, on
  the truth's tab (`replayIssueText`; the words named in its tooltip and the ⓘ notes, `replayOutsideWords`) (AV37).
- **The live executor has NO panel block** (2026-09-29, the user: redundant): its answer is the sentence bar's line
  (`TrainingAutopilotStatus`), the cursor, 3D and the read-back window; a band click always flies; no "Replay in 3D" (↻ Fly
  again asks anew); the payload reader still checks every field (`timing`, shas) — the backend contract, not the view (AV38).
- **Training's MULTI-AIRCRAFT WINDOWS are a set kind of their own** (`traffic-windows`, `traffic.json` + per model a
  `window-generation` overlay, ts `window_training_export`); the mode is the set's kind (`trainingSets.ts`); a window's
  commanded aircraft is an ordinary set flight and a model's window sentences are projected onto the single-flight views
  (`windowGenerationView`); the cursor is ONE time on the selection's clock (`TrainingSelection.clock`: the flight's own, or the
  window's — `trainingCursorS` own, `trainingSceneS` the clock's); the window strip above the bar and the 3D traffic layer draw
  every aircraft; the frontend judges nothing (losses, ends, landings are the exporter's, only their books checked); a window's
  aircraft is never flown live (`liveExecutor`); the camera frames once per clock; in 3D each aircraft is drawn by its ROLE
  (`TrainingAircraftRole`: on screen — "▶" chip and ring in the selection yellow, never the reading's colour — / commanded, a
  SET, in `trainingWindowReadingColour` / replayed / background; one table `ROLE_DRAW`), `windowOnScreen` is the one "is a
  window on screen" test (AV39).
- **A failure is ONE red, `TRAINING_FAILURE_COLOR` `#ff2d2d`** (a model's flight that did not land on the bar, losses of
  separation and judge ends, the live executor outside its envelope); **an SVG `fill` attribute loses to any stylesheet `fill`
  rule** — colour data-driven SVG text with `style={{ fill }}` (two marks were grey for that reason) (AV40).
- **Every Training aircraft is the 3D model in the attitude the Python exporters computed** (`experiments/training_attitude.py`:
  heading, path angle, right bank, an attack READING; every track's `attitude` block — sample v8, traffic v2, executor v5,
  generation v5, augmented v3, window-generation v2, live Fly v8); the viewer computes no attitude (`trainingAttitude.poseAt`
  only interpolates); **the drawn pitch is the path angle, the attack only in the label** (user, 2026-09-30: the lift curve
  has no flaps); a flight without an airframe has no bank/attack (null: wings level, said); ONE model orientation
  (`utils/aircraftOrientation.ts`) for Fly, Optimize and Training (AV42).

## Comparison CZML colour contract

- Group status is entity `properties.status` ∈ solved/offTarget/failed; **the frontend repaint
  skip is keyed on "a verdict colour was baked", NOT on `status` alone** (AV13).
- **`otherRunway`** = a two-tier generation sentence that passed on ANOTHER runway than the observed flight's (the
  landed-runway grading, `--landed-runway-report`): light sky blue, never the pass green; the frontend must learn a status
  before any index carries it (an unknown status refuses the whole index) (AV41).
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
