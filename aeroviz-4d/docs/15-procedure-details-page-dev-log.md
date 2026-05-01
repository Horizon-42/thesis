# Procedure Details Page Dev Log

## 2026-04-23 22:53 CEST

### Goal Of This Session
- Implement the first working `/procedure-details` page.
- Add the browser-ready procedure-details export pipeline and chart publishing path.
- Generate KRDU procedure-details assets so the page can run against real data.

### Facts Discovered
- The existing CIFP pipeline in `python/preprocess_procedures.py` already had enough semantic data to build runway-scoped procedure documents: branches, legs, fixes, warnings, and route-point timing.
- The newer runway-profile work in `src/utils/runwayProfileGeometry.ts` already established the right altitude-repair strategy for missing/placeholder heights; the new page needed to follow that same interpolation idea instead of trusting `0 ft` placeholders.
- Local RNAV chart PDFs already existed under `data/RNAV_CHARTS/KRDU/`.
- Generated browser data under `aeroviz-4d/public/data/*` is ignored by `.gitignore`, so the published KRDU procedure-details JSON and chart copies are local build artifacts, not tracked source files.
- The repo-level `.gitignore` rule `data` was also catching `aeroviz-4d/src/data/*`, so source files in that folder had to be explicitly unignored.
- This environment does not currently have `pytest` installed, so the new Python test file could be written but not executed here.

### Decisions Locked
- The richer page reads only the new browser-ready intermediate JSON and chart manifest; it does not fall back to `procedures.geojson` when the richer dataset is missing.
- The page is a standalone route with lightweight history-based routing in `App.tsx`, not a Cesium overlay.
- KRDU is the first generated airport; other airports show a friendly empty state until their procedure-details assets are generated.
- The vertical profile uses repaired/interpolated display altitudes for missing values instead of treating unknown heights as real `0 ft`.
- The local chart experience is link-out only in v1.

### Files Changed
- `python/data_layout.py`
- `python/preprocess_procedures.py`
- `python/tests/test_preprocess_procedures.py`
- `.gitignore`
- `src/App.tsx`
- `src/components/ControlPanel.tsx`
- `src/components/ProcedureDetailsPage.tsx`
- `src/components/__tests__/ControlPanel.test.tsx`
- `src/components/__tests__/ProcedureDetailsPage.test.tsx`
- `src/data/airportData.ts`
- `src/data/__tests__/airportData.test.ts`
- `src/data/procedureDetails.ts`
- `src/index.css`
- `src/utils/navigation.ts`
- `src/utils/procedureDetailsGeometry.ts`

### Commands Run / Checks Passed
- `python -m py_compile aeroviz-4d/python/preprocess_procedures.py aeroviz-4d/python/data_layout.py`
- `python aeroviz-4d/python/preprocess_procedures.py --airport KRDU --include-all-rnav`
- `npm test -- --run`
- `npm run build`

### Commands Run / Checks Blocked
- `python -m pytest tests/test_preprocess_procedures.py`
  - blocked because `pytest` is not installed in this environment

### Current Status
- `/procedure-details` exists and renders a runway navigator, procedure picker, overview card, plan view, vertical profile, leg ladder, fix inspector, glossary, and reference links.
- `ControlPanel` now includes a `Procedure Details` navigation entry that opens the new page for the active airport.
- KRDU procedure-details documents and local chart copies are generated locally under:
  - `public/data/airports/KRDU/procedure-details/`
  - `public/data/airports/KRDU/charts/`
- Generated KRDU assets are not tracked by git because `public/data/*` is ignored.

### Known Blockers
- Fresh checkouts will not automatically contain the generated KRDU procedure-details assets unless the export command is rerun.
- Python tests for the exporter are present but still need execution in an environment with `pytest`.
- The current plan-view and profile charts are intentionally lightweight SVG visualizations; they are readable and interactive, but not yet a one-to-one recreation of the full FAA sheet layout.

### Exact Next Recommended Step
- Add a small documented regeneration command to the user-facing docs or README section for the page, then extend the page with one or both of:
  - a branch-specific filter/legend toggle
  - a tighter FAA-like profile annotation layer for minima, missed-approach transition, and threshold markers

## 2026-04-24 15:55 CEST

### Goal Of This Session
- Redesign the `Procedure Details` page so it feels more like an interactive explorer than a static report.
- Reduce always-visible dense detail blocks.
- Enlarge the plan view and vertical profile, and move them into a stacked up-down layout.

### Facts Discovered
- The original page structure was readable but too flat: sidebar + two medium charts + long always-visible leg/glossary lists.
- The existing procedure-detail data already had enough structure to support focus-driven interactions without changing the data pipeline: branches, per-leg endpoints, role hints, and chart point geometry were already available.
- The current SVG charts became much easier to read once labels were shown mainly for the focused branch/fix and important role points instead of every point all at once.

### Decisions Locked
- The page now uses a single-column main reading flow instead of a sidebar-led layout so the charts can take substantially more width.
- The interaction model is now “focus-driven”: branch pills, fix chips, chart points, and leg cards all feed one shared focused branch/fix state.
- Detailed procedure content is now layered:
  - overview is always visible
  - branch/fix explorer is always visible
  - focused sequence, focused fix metadata, and glossary explanation respond to current focus
  - the full always-open leg ladder and full glossary list are removed from the default page surface
- Plan view and vertical profile remain SVG-based for now, but are deliberately larger and stacked vertically.

### Files Changed
- `src/components/ProcedureDetailsPage.tsx`
- `src/index.css`

### Commands Run / Checks Passed
- `npm run build`
- `npm test -- --run src/components/__tests__/ProcedureDetailsPage.test.tsx`
- `npm test -- --run`

### Current Status
- The page now opens with airport/runway/procedure selection cards at the top, a larger overview, an interactive branch/fix explorer strip, and two enlarged charts stacked vertically.
- Fix and branch focus now drives:
  - chart emphasis
  - fix inspector content
  - focused branch sequence cards
  - glossary chip definition
- The leg ladder has been replaced by a focused sequence card deck, so the user sees only the currently relevant procedure path instead of a long global table.
- The glossary is now layered behind term chips instead of rendering every definition at once.

### Known Blockers
- The charts are larger and more interactive, but still intentionally lightweight SVG renderings rather than a full FAA-sheet recreation.
- Focus is currently driven by hover/click state only; there is not yet a pinned multi-fix comparison mode.
- Chart annotations are still intentionally selective; additional altitude callouts, minima, or missed-approach-specific symbols would require another visual pass.

### Exact Next Recommended Step
- Add one deeper annotation layer to the enlarged charts, prioritizing one of:
  - threshold / FAF / MAPt callout badges directly in the plan/profile views
  - branch merge markers and missed-approach continuation cues
  - a compact “compare branches” toggle that temporarily shows more than one branch at full emphasis

## 2026-05-01 23:11 CEST

### Goal Of This Session
- Continue the v3 protected-geometry migration after identifying that turn areas were visually discontinuous.
- Add a modular baseline for TF turn junction continuity without pretending to implement full FAA FB/FO/RF turn construction.
- Preserve the migration discipline: small geometry kernel first, render integration second, then log the design boundary.

### Facts Discovered
- The v3 visualization spec already says intermediate-to-final transitions are not simple line joins; they require taper or offset construction, and G-09 requires continuous boundaries.
- The current kernel still only supports TF centerlines and straight offset envelopes. This creates visible discontinuities or misleading kinks at turn junctions.
- Final segments with multiple TF legs can reveal turns that should be flagged, because LNAV final should not silently accept TF turns as compliant protected geometry.

### Decisions Locked
- `procedureTurnGeometry.ts` is a Cesium-independent module.
- The new TF turn patch is explicitly marked `VISUAL_FILL_ONLY`.
- The patch is allowed to improve visual continuity, but it is not a certified FB/FO/RF construction.
- If a final segment contains a TF turn junction, the geometry bundle emits `FINAL_HAS_TURN` diagnostic.
- Full compliance remains future work:
  - RF true arc centerline and parallel OEA;
  - FB turn construction with DTA, bisector, and boundary arcs;
  - FO construction with reaction/roll distance and 30-degree taper;
  - offset intermediate-to-final debug primitives.

### Files Changed
- `src/utils/procedureTurnGeometry.ts`
- `src/utils/__tests__/procedureTurnGeometry.test.ts`
- `src/utils/procedureSegmentGeometry.ts`
- `src/utils/__tests__/procedureSegmentGeometry.test.ts`
- `src/hooks/useProcedureSegmentLayer.ts`
- `src/hooks/__tests__/useProcedureSegmentLayer.test.ts`

### Commands Run / Checks Passed
- `npm test -- --run src/utils/__tests__/procedureTurnGeometry.test.ts`
- `npm test -- --run src/utils/__tests__/procedureSegmentGeometry.test.ts src/utils/__tests__/procedureTurnGeometry.test.ts src/hooks/__tests__/useProcedureSegmentLayer.test.ts src/data/__tests__/procedureRenderBundle.test.ts`
- `npm test -- --run`
- `npm run build`

### Current Status
- Protected mode now renders visual turn-fill patches at detected TF turn junctions.
- The patches are drawn as independent 3D entities above the base envelope surfaces so turn discontinuities are easier to see.
- Segment diagnostics now distinguish “visual continuity fill” from proper final-segment turn authorization.

### Known Blockers
- This does not yet satisfy G-04, G-05, or G-06; those need actual FB/FO/RF algorithms and source metadata.
- Source procedure-detail JSON does not yet expose enough FB/FO/RF metadata for full turn construction.
- Some discontinuities between different segment objects may still need branch-level junction construction, especially outside the final connector region.

### Exact Next Recommended Step
- Add branch-level turn junction construction between adjacent segment bundles, then start RF support only when radius/center data is available or exported.

## 2026-05-01 23:17 CEST

### Goal Of This Session
- Continue the turn-continuity migration by covering discontinuities between adjacent segment objects on the same protected-geometry branch.
- Keep the implementation modular: geometry kernel stays Cesium-independent, render-bundle assembly owns branch-level joins, and Cesium layer only renders prepared geometry.

### Facts Discovered
- Intra-segment TF turn patches do not cover joins where the procedure adapter splits an approach into separate intermediate/final/missed segment records.
- The new inter-segment kernel can safely build a small visual patch only when the two segment endpoints are adjacent within a tight gap tolerance.
- The render bundle is the right boundary for these joins because it has ordered branch segment bundles and can attach branch-level diagnostics before rendering.

### Decisions Locked
- Branch-level turn junctions are stored on `BranchGeometryBundle.turnJunctions`.
- Each built inter-segment junction emits `TURN_VISUAL_FILL_ONLY`.
- Width selection uses the wider adjacent segment tolerance so the fill does not under-cover either side of the join.
- The Cesium procedure layer renders these patches under the branch visibility map, so procedure selection and protected-mode visibility stay canonical.
- These patches remain visual continuity fills only; they are not FAA-compliant FB/FO/RF construction.

### Files Changed
- `src/data/procedurePackage.ts`
- `src/data/procedureRenderBundle.ts`
- `src/data/__tests__/procedureRenderBundle.test.ts`
- `src/hooks/useProcedureSegmentLayer.ts`
- `src/hooks/__tests__/useProcedureSegmentLayer.test.ts`

### Commands Run / Checks Passed
- `npm test -- --run src/data/__tests__/procedureRenderBundle.test.ts src/hooks/__tests__/useProcedureSegmentLayer.test.ts src/utils/__tests__/procedureTurnGeometry.test.ts`
- `npm run build`
- `npm test -- --run`

### Current Status
- Protected procedure rendering now has visual turn-fill coverage both inside multi-leg TF segments and between adjacent segment bundles on the same branch.
- Branch-level turn patches are branch-scoped and follow the canonical v3 branch identifiers.
- The implementation should reduce the visible gaps around intermediate-to-final or other split-segment turns, while preserving diagnostics that the construction is not yet compliant turn protection.

### Known Blockers
- Full G-04/G-05/G-06 acceptance still needs true RF arc handling and FB/FO turn construction from source metadata.
- The current inter-segment patch intentionally refuses joins with larger endpoint gaps instead of inventing missing path geometry.
- The procedure exporter may still need to expose richer turn metadata before compliant RF/FB/FO geometry can be built.

### Exact Next Recommended Step
- Implement RF geometry support when `arcRadiusNm` and center/turn metadata are available, then add acceptance tests for continuous RF centerline and parallel protected boundaries.

## 2026-05-01 23:21 CEST

### Goal Of This Session
- Add the first RF-capable geometry kernel without changing the current exporter contract or inventing missing RF metadata.
- Integrate RF centerlines into the segment geometry builder only when the canonical `ProcedurePackageLeg` already carries the required radius, center, and turn direction.

### Facts Discovered
- Current `ProcedureDetailDocument` records still do not expose RF `arcRadiusNm`, `centerLatDeg`, or `centerLonDeg`.
- The canonical package schema already has optional RF fields, so RF support can be implemented behind that boundary before the exporter is upgraded.
- TF visual turn-fill detection must not run over sampled RF arcs, otherwise normal arc curvature can be misdiagnosed as a sequence of TF corner turns.

### Decisions Locked
- `procedureRfGeometry.ts` owns RF centerline construction and remains independent of Cesium.
- RF construction requires:
  - positioned start and end fixes;
  - `arcRadiusNm`;
  - `centerLatDeg` and `centerLonDeg`;
  - explicit `turnDirection`.
- Missing RF center/radius metadata returns `RF_RADIUS_MISSING`.
- Missing direction or inconsistent radius/fix geometry returns `SOURCE_INCOMPLETE`.
- Segment bundles may now contain RF centerlines and straight offset ribbons, but RF-specific parallel OEA/Case 1/Case 2 protected-area construction is still future work.

### Files Changed
- `src/utils/procedureRfGeometry.ts`
- `src/utils/procedureSegmentGeometry.ts`
- `src/utils/__tests__/procedureSegmentGeometry.test.ts`

### Commands Run / Checks Passed
- `npm test -- --run src/utils/__tests__/procedureSegmentGeometry.test.ts src/data/__tests__/procedureRenderBundle.test.ts src/hooks/__tests__/useProcedureSegmentLayer.test.ts`
- `npm run build`
- `npm test -- --run src/components/__tests__/ProcedurePanel.test.tsx src/utils/__tests__/procedureSegmentGeometry.test.ts`
- `npm test -- --run`
- `npm run build`

### Current Status
- Synthetic RF legs with complete metadata now build sampled circular centerlines with radial error covered by unit tests.
- Segment geometry integrates TF and RF legs in order.
- RF segments do not receive TF visual turn-fill patches.
- Real exported documents will still report missing RF metadata until the Python/export schema is upgraded.

### Known Blockers
- RF envelope support is still a straight offset ribbon over sampled centerline, not the full RF parallel OEA construction required by G-06.
- The exporter/parser does not yet populate RF radius/center/direction fields.
- DF/CA/HM/HA/HF remain unsupported by the protected geometry kernel.

### Exact Next Recommended Step
- Upgrade the procedure-detail/export schema to carry RF radius, center, and turn direction from CIFP/source data, then add a fixture that proves an exported RF leg reaches the new kernel without synthetic test-only data.

## 2026-05-01 23:24 CEST

### Goal Of This Session
- Add schema plumbing so RF metadata can travel from the Python procedure model through exported procedure-detail JSON into the canonical frontend `ProcedurePackage`.
- Avoid guessing CIFP RF fields until the parser mapping is verified against source records.

### Facts Discovered
- `ProcedurePackageLeg` already had optional RF fields, but `ProcedureDetailLeg.path` and the Python export document did not.
- The current parser still produces `ProcedureLeg` records without populated RF radius/center/direction fields.
- The frontend adapter previously emitted `RF_RADIUS_MISSING` for every RF leg even if metadata were supplied.

### Decisions Locked
- RF metadata is carried on the procedure-detail leg `path` object:
  - `turnDirection`
  - `arcRadiusNm`
  - `centerLatDeg`
  - `centerLonDeg`
- The adapter promotes those fields onto canonical `ProcedurePackageLeg`.
- `RF_RADIUS_MISSING` is emitted only when RF center/radius fields are absent.
- Parser-level extraction remains a separate step because column/key mapping must be verified against real CIFP/cifparse records.

### Files Changed
- `python/cifp_parser.py`
- `python/preprocess_procedures.py`
- `python/tests/test_preprocess_procedures.py`
- `src/data/procedureDetails.ts`
- `src/data/procedurePackageAdapter.ts`
- `src/data/__tests__/procedurePackageAdapter.test.ts`

### Commands Run / Checks Passed
- `npm test -- --run src/data/__tests__/procedurePackageAdapter.test.ts src/utils/__tests__/procedureSegmentGeometry.test.ts`
- `python -m pytest python/tests/test_preprocess_procedures.py` failed because this Python environment does not have `pytest` installed.
- `python -m py_compile python/cifp_parser.py python/preprocess_procedures.py python/tests/test_preprocess_procedures.py`
- Python RF schema smoke test via direct `build_branch_document(...)` call.
- `npm test -- --run`
- `npm run build`

### Current Status
- The RF metadata path exists end-to-end once parser/exporter code can populate the fields.
- The TypeScript adapter no longer blocks RF legs that already include radius and center metadata.
- Python tests have a new RF metadata assertion, but full pytest execution still needs a Python environment with `pytest`.

### Known Blockers
- Real CIFP RF radius/center/direction parsing is not implemented yet.
- Existing generated KRDU procedure-detail JSON must be regenerated after parser/export changes before real RF metadata can appear in the app.
- RF protected envelope construction remains a straight sampled-ribbon approximation until RF-specific OEA cases are implemented.

### Exact Next Recommended Step
- Inspect real cifparse procedure-record keys for RF legs, map the verified radius/center/direction fields into `ProcedureLeg`, and regenerate a procedure-detail fixture that exercises the RF kernel from exported data.

## 2026-05-01 23:38 CEST

### Goal Of This Session
- Populate RF metadata from real CIFP/cifparse procedure records instead of synthetic test-only data.
- Resolve RF center fixes into exported procedure-detail JSON so frontend RF construction can receive center coordinates.

### Facts Discovered
- The `aviation` environment has a PATH issue: `conda run -n aviation python` resolves to Homebrew Python, but `conda run -n aviation pytest` uses the correct conda environment interpreter.
- Real cifparse RF procedure records expose:
  - `turn_direction`
  - `arc_radius`
  - `center_fix`
  - `center_fix_region`
- KATL `ZELAN4` branch `4RW27R` has a real RF leg suitable for regression coverage.

### Decisions Locked
- `ProcedureLeg` now stores both RF source identifiers and resolved metadata:
  - `turn_direction`
  - `arc_radius_nm`
  - `center_fix_ident`
  - `center_fix_region_code`
  - optional resolved center lat/lon.
- `build_fix_index(...)` includes RF center fixes when resolving the procedure fix catalog.
- Exported procedure-detail path metadata now includes `centerFixRef` plus `centerLatDeg/centerLonDeg` when the center fix resolves.
- `agent.md` now documents the explicit conda environment Python path and recommends `conda run -n aviation pytest ...` for Python tests.

### Files Changed
- `python/cifp_parser.py`
- `python/preprocess_procedures.py`
- `python/tests/test_preprocess_procedures.py`
- `src/data/procedureDetails.ts`
- `agent.md`

### Commands Run / Checks Passed
- `conda run -n aviation pytest python/tests/test_preprocess_procedures.py`
- `conda run -n aviation pytest python/tests`
- `npm test -- --run src/data/__tests__/procedurePackageAdapter.test.ts src/utils/__tests__/procedureSegmentGeometry.test.ts src/data/__tests__/procedureRenderBundle.test.ts`
- `npm run build`
- `npm test -- --run`

### Current Status
- Real CIFP RF fields now flow into the Python procedure model.
- RF center fixes are included in fix resolution and exported as center coordinates in procedure-detail JSON.
- Python tests now cover a real KATL RF leg through parser, fix resolution, and branch document export.

### Known Blockers
- Generated public procedure-detail assets still need regeneration to contain the new RF metadata.
- KRDU may not contain RF legs in the current selected RNAV procedures, so RF validation may need a non-KRDU fixture or a synthetic exported fixture for frontend acceptance.
- RF protected envelope construction is still sampled straight-ribbon geometry rather than full RF Case 1/Case 2 OEA.

### Exact Next Recommended Step
- Regenerate procedure-detail assets for an RF-containing airport/procedure fixture, then add an end-to-end frontend render-bundle test proving exported RF metadata reaches `buildRfLeg(...)`.

## 2026-05-01 23:40 CEST

### Goal Of This Session
- Close the frontend RF metadata test gap by proving exported procedure-detail style RF path metadata reaches render-bundle geometry construction.

### Decisions Locked
- The frontend regression test starts at `ProcedureDetailDocument`, not a prebuilt `ProcedurePackage`, so it covers:
  - procedure-detail schema;
  - `normalizeProcedurePackage(...)`;
  - `buildProcedureRenderBundle(...)`;
  - RF centerline construction.
- The test asserts RF metadata does not emit `RF_RADIUS_MISSING` and produces an arc centerline.

### Files Changed
- `src/data/__tests__/procedureRenderBundle.test.ts`
- `docs/15-procedure-details-page-dev-log.md`

### Commands Run / Checks Passed
- `npm test -- --run src/data/__tests__/procedureRenderBundle.test.ts src/data/__tests__/procedurePackageAdapter.test.ts src/utils/__tests__/procedureSegmentGeometry.test.ts`
- `npm test -- --run`
- `npm run build`

### Current Status
- Frontend RF metadata path is now covered from exported JSON shape through render-bundle geometry.
- Test count is now 86 frontend tests, all passing.

### Known Blockers
- This is still a synthetic frontend fixture; generated public RF procedure-detail assets have not been regenerated or committed.
- RF protected-area envelope construction remains future work.

### Exact Next Recommended Step
- Add RF-specific protected envelope construction for sampled RF centerlines, starting with a conservative parallel ribbon test and then separating full RF OEA Case 1/Case 2 behavior.

## 2026-05-01 23:42 CEST

### Goal Of This Session
- Improve RF protected ribbon geometry so pure RF segments use parallel circular boundaries instead of generic sampled-polyline offsets.

### Decisions Locked
- RF centerline geometry now preserves arc metadata:
  - center point;
  - radius;
  - start angle;
  - sweep angle;
  - turn direction.
- Pure RF segment envelopes use concentric left/right boundary arcs at `radius +/- halfWidth`.
- Mixed TF/RF segments continue to fall back to the existing generic sampled offset path until multi-leg RF/TF composition is designed.

### Files Changed
- `src/utils/procedureRfGeometry.ts`
- `src/utils/procedureSegmentGeometry.ts`
- `src/utils/__tests__/procedureSegmentGeometry.test.ts`
- `docs/15-procedure-details-page-dev-log.md`

### Commands Run / Checks Passed
- `npm test -- --run src/utils/__tests__/procedureSegmentGeometry.test.ts src/data/__tests__/procedureRenderBundle.test.ts`
- `npm run build`
- `npm test -- --run`

### Current Status
- Pure RF segment primary/secondary ribbons now preserve constant radial offset from the RF center.
- Unit tests verify the RF primary boundary radii against expected `center radius +/- halfWidth`.

### Known Blockers
- This is still not full FAA RF OEA Case 1/Case 2 construction.
- Mixed TF/RF protected area stitching still needs explicit transition rules.

### Exact Next Recommended Step
- Add RF debug/inspection primitives or diagnostics for RF metadata in the procedure panel/details view, then design full RF OEA Case 1/Case 2 acceptance tests.

## 2026-05-01 23:44 CEST

### Goal Of This Session
- Surface RF source metadata in the Procedure Details focused sequence so the user can inspect turn direction, radius, and center without opening raw JSON.

### Decisions Locked
- RF leg cards now display:
  - `Turn LEFT/RIGHT`
  - `Radius <n> NM`
  - `Center <fix>` or center coordinates when no center fix is available.
- This stays in the focused sequence metadata row so it supports validation without adding another panel.

### Files Changed
- `src/components/ProcedureDetailsPage.tsx`
- `src/components/__tests__/ProcedureDetailsPage.test.tsx`
- `docs/15-procedure-details-page-dev-log.md`

### Commands Run / Checks Passed
- `npm test -- --run src/components/__tests__/ProcedureDetailsPage.test.tsx`
- `npm test -- --run`
- `npm run build`

### Current Status
- Procedure Details now exposes RF metadata in the UI.
- Frontend test count is now 87 tests, all passing.

### Known Blockers
- RF metadata visibility is currently textual; there are no plan-view RF center/radius debug primitives yet.
- Full RF OEA Case 1/Case 2 acceptance remains future work.

### Exact Next Recommended Step
- Add optional plan-view RF center/radius markers for RF legs, then define the full RF OEA Case 1/Case 2 geometry acceptance fixtures.

## 2026-05-01 23:46 CEST

### Goal Of This Session
- Add plan-view RF debug markers so RF center and radius can be inspected visually in Procedure Details.

### Decisions Locked
- RF markers are derived from procedure-detail RF path metadata and rendered only when center coordinates and radius are present.
- The plan-view domain includes RF center/radius extents so the marker is not clipped out of the SVG.
- The marker includes:
  - a dashed radius circle;
  - a dashed line from center to RF endpoint;
  - a center marker and label.

### Files Changed
- `src/components/ProcedureDetailsPage.tsx`
- `src/components/__tests__/ProcedureDetailsPage.test.tsx`
- `src/utils/procedureDetailsGeometry.ts`
- `src/index.css`
- `docs/15-procedure-details-page-dev-log.md`

### Commands Run / Checks Passed
- `npm test -- --run src/components/__tests__/ProcedureDetailsPage.test.tsx`
- `npm test -- --run`
- `npm run build`

### Current Status
- Procedure Details now exposes RF metadata both as focused-sequence text and as plan-view center/radius debug geometry.

### Known Blockers
- RF debug marker is inspection geometry, not the final RF OEA Case 1/Case 2 protected-area implementation.
- The app still needs a generated RF-containing public fixture for browser-level manual validation.

### Exact Next Recommended Step
- Create or regenerate an RF-containing procedure-detail fixture and use it for manual visual validation, then start full RF OEA Case 1/Case 2 acceptance tests.

## 2026-05-01 23:52 CEST

### Goal Of This Session
- Turn the manually verified KATL ZELAN4 RF export path into an automated regression test.

### Decisions Locked
- The test uses the full `build_procedure_detail_document(...)` path instead of only `build_branch_document(...)`.
- KATL `ZELAN4` is retained as the real CIFP RF fixture because it has a populated RF leg from `CPARK` to `MPASS` with center fix `CFZJF`.
- The fixture remains test-only; no large generated public procedure-detail asset is committed in this phase.

### Files Changed
- `python/tests/test_preprocess_procedures.py`
- `docs/15-procedure-details-page-dev-log.md`

### Commands Run / Checks Passed
- `conda run -n aviation pytest python/tests/test_preprocess_procedures.py -k katl_zelan4`

### Current Status
- Full procedure-detail document generation now proves real RF metadata survives parser, fix resolution, branch assembly, and JSON document construction.
- The regression asserts RF start/end fix refs, turn direction, radius, center fix ref, and resolved center coordinates.

### Known Blockers
- This still does not publish a browser-visible RF public data fixture.
- RF protected-area Case 1/Case 2 construction remains incomplete.

### Exact Next Recommended Step
- Add RF envelope case metadata and Case 1/Case 2 acceptance tests in the TypeScript geometry kernel.

## 2026-05-01 23:55 CEST

### Goal Of This Session
- Add explicit RF envelope case metadata and prevent inner-side RF envelope collapse from breaking geometry construction.

### Decisions Locked
- `LateralEnvelopeGeometry` now identifies whether an envelope was built as `STRAIGHT_OFFSET` or `RF_PARALLEL_ARC`.
- RF envelopes now carry:
  - `rfEnvelopeCase`;
  - nominal radius;
  - inner radius;
  - outer radius.
- Case 1 means the inside parallel radius remains positive.
- Case 2 currently means the inside parallel radius collapses to the RF center and stays finite/renderable. This is a first acceptance-safe geometry guard, not a full FAA Figure 1-2-11 construction claim.

### Files Changed
- `src/utils/procedureRfGeometry.ts`
- `src/utils/procedureSegmentGeometry.ts`
- `src/utils/__tests__/procedureSegmentGeometry.test.ts`
- `docs/15-procedure-details-page-dev-log.md`

### Commands Run / Checks Passed
- `npm test -- --run src/utils/__tests__/procedureSegmentGeometry.test.ts`

### Current Status
- RF parallel envelopes no longer fall back to straight offsets when the inside radius is zero or negative.
- Tests cover both RF Case 1 metadata and finite Case 2 inner-collapsed geometry.

### Known Blockers
- This is still a conservative Case 2 rendering guard, not the full FAA RF OEA Case 2 construction.
- Mixed TF/RF stitching still needs explicit transition geometry.

### Exact Next Recommended Step
- Use the RF case metadata in render-bundle/debug output, then continue segment-level trajectory assessment for the 2D profile.

## 2026-05-01 23:59 CEST

### Goal Of This Session
- Start migrating the 2D runway trajectory profile from boolean horizontal-plate inclusion to segment-level assessment output.

### Decisions Locked
- Added `procedureSegmentAssessment.ts` as a standalone service-style utility.
- The first implementation consumes existing projected `HorizontalPlateRoute` data, but returns the v3-style fields needed by the UI:
  - `activeSegmentId`;
  - station;
  - cross-track error;
  - containment.
- The hook now filters aircraft with segment assessment rather than a plain `pointIsInsideHorizontalPlate(...)` boolean.
- The panel summary shows the selected aircraft assessment so users can inspect segment/station/cross-track without opening debug tools.

### Files Changed
- `src/utils/procedureSegmentAssessment.ts`
- `src/utils/__tests__/procedureSegmentAssessment.test.ts`
- `src/hooks/useRunwayTrajectoryProfile.ts`
- `src/components/RunwayTrajectoryProfilePanel.tsx`
- `src/components/__tests__/RunwayTrajectoryProfilePanel.test.tsx`
- `docs/15-procedure-details-page-dev-log.md`

### Commands Run / Checks Passed
- `npm test -- --run src/utils/__tests__/procedureSegmentAssessment.test.ts src/components/__tests__/RunwayTrajectoryProfilePanel.test.tsx`

### Current Status
- The 2D profile now carries per-aircraft segment assessment metadata through the hook into the UI.
- Unit tests cover nearest-segment projection, outside classification with retained context, and selected-aircraft assessment display.

### Known Blockers
- The assessment still uses the projected route/plate representation, not `ProcedureRenderBundle` segment envelopes.
- Only `PRIMARY` vs `OUTSIDE` is available because the current profile data does not yet carry secondary envelope widths.

### Exact Next Recommended Step
- Feed profile assessment from `ProcedureRenderBundle` envelopes so primary/secondary/outside can be classified against the same geometry used by Procedure Details and 3D protected mode.

## 2026-05-02 00:04 CEST

### Goal Of This Session
- Feed the 2D runway profile assessment from `ProcedureRenderBundle` segment geometry while preserving the existing readable fix-based profile display.

### Decisions Locked
- `HorizontalPlateRoute` now has optional `assessmentSegments`.
- Display points remain the existing route/fix points, so the profile does not render dense geometry samples as fix markers.
- Assessment segments are attached from `ProcedureRenderBundle` by matching `procedureUid + branchKey` to the scoped v3 branch id.
- Containment now supports:
  - `PRIMARY`;
  - `SECONDARY`;
  - `OUTSIDE`.
- If render-bundle assessment segments are absent, the profile falls back to the existing route segment geometry.

### Files Changed
- `src/utils/runwayProfileGeometry.ts`
- `src/utils/procedureSegmentAssessment.ts`
- `src/utils/__tests__/runwayProfileGeometry.test.ts`
- `src/utils/__tests__/procedureSegmentAssessment.test.ts`
- `src/hooks/useRunwayTrajectoryProfile.ts`
- `docs/15-procedure-details-page-dev-log.md`

### Commands Run / Checks Passed
- `npm test -- --run src/utils/__tests__/procedureSegmentAssessment.test.ts src/utils/__tests__/runwayProfileGeometry.test.ts src/components/__tests__/RunwayTrajectoryProfilePanel.test.tsx`
- `npm run build`

### Current Status
- 2D trajectory filtering and sample annotation can now use the same v3 segment centerlines and primary/secondary widths built for protected geometry.
- The UI still keeps compact readable fix labels instead of dense sampled geometry labels.

### Known Blockers
- The hook currently loads both route data and render-bundle data, which duplicates some procedure-detail fetching.
- Vertical error and event marker output are still not part of `SegmentAssessment`.

### Exact Next Recommended Step
- Refactor profile loading so one procedure-detail fetch builds both display routes and render bundles, then add vertical error/event markers for trajectory validation.

## 2026-05-02 00:07 CEST

### Goal Of This Session
- Remove duplicated procedure-detail loading from the runway trajectory profile data path.

### Decisions Locked
- `useRunwayTrajectoryProfile` now loads `ProcedureRenderBundleData` once.
- Display routes are rebuilt from `procedureRenderData.documents` with `buildProcedureRoutes(...)`.
- Assessment segments continue to come from `procedureRenderData.renderBundles`.

### Files Changed
- `src/hooks/useRunwayTrajectoryProfile.ts`
- `docs/15-procedure-details-page-dev-log.md`

### Commands Run / Checks Passed
- `npm run build`

### Current Status
- The profile has one procedure-detail/render-bundle fetch path instead of separate route and render-bundle loaders.
- This keeps display routes and assessment geometry tied to the same document snapshot.

### Known Blockers
- Vertical validation metrics still need to be added to the segment assessment payload.
- There is no hook-level test for the single-load behavior yet.

### Exact Next Recommended Step
- Add vertical error and event-marker fields to `HorizontalPlateSegmentAssessment`, then expose them in the selected aircraft summary.

## 2026-05-02 00:10 CEST

### Goal Of This Session
- Add vertical validation output and basic event markers to runway profile segment assessment.

### Decisions Locked
- `HorizontalPlateSegmentAssessment` now includes:
  - `verticalErrorM`;
  - `events`.
- Vertical error is measured against the interpolated segment centerline/profile height. This is an assessment reference, not yet a full vertical protection surface.
- Event markers currently include lateral containment and vertical deviation beyond 100 ft.
- The selected aircraft summary now displays vertical error in feet.

### Files Changed
- `src/utils/procedureSegmentAssessment.ts`
- `src/utils/__tests__/procedureSegmentAssessment.test.ts`
- `src/components/RunwayTrajectoryProfilePanel.tsx`
- `src/components/__tests__/RunwayTrajectoryProfilePanel.test.tsx`
- `docs/15-procedure-details-page-dev-log.md`

### Commands Run / Checks Passed
- `npm test -- --run src/utils/__tests__/procedureSegmentAssessment.test.ts src/components/__tests__/RunwayTrajectoryProfilePanel.test.tsx`
- `npm run build`

### Current Status
- 2D profile validation now reports horizontal segment context, lateral containment, cross-track error, vertical error, and basic events for the selected aircraft.

### Known Blockers
- Vertical error is centerline/profile-relative only; it is not LPV/GLS/LNAV/VNAV surface validation.
- Events are point-local markers; temporal enter/exit event derivation still needs trail/time-series processing.

### Exact Next Recommended Step
- Add missed approach sectionization (`MISSED_S1` / `MISSED_S2`) so post-threshold trajectory validation can stop treating missed approach as a generic continuation.

## 2026-05-02 00:13 CEST

### Goal Of This Session
- Add the first missed approach sectionization pass in the v3 package adapter.

### Decisions Locked
- Missed groups are split at the first hold/MAHF-style boundary:
  - before the boundary: `MISSED_S1`;
  - from the hold/MAHF boundary onward: `MISSED_S2`.
- If no hold/MAHF boundary is present, missed legs remain `MISSED_S1`.
- If the missed group starts with a hold/MAHF leg, it is classified as `MISSED_S2`.
- This is section typing only; no missed approach protected-surface or turning missed wind spiral is claimed yet.

### Files Changed
- `src/data/procedurePackageAdapter.ts`
- `src/data/__tests__/procedurePackageAdapter.test.ts`
- `docs/15-procedure-details-page-dev-log.md`

### Commands Run / Checks Passed
- `npm test -- --run src/data/__tests__/procedurePackageAdapter.test.ts`

### Current Status
- The canonical package can now represent `MISSED_S1` and `MISSED_S2` instead of collapsing every missed leg into a single generic missed segment.
- Regression coverage verifies the split at a hold/MAHF boundary.

### Known Blockers
- CA/DF missed geometry remains unsupported by the segment geometry kernel.
- Turning missed approach protection and wind spiral debug geometry remain future work.

### Exact Next Recommended Step
- Add CA/DF missed-leg preservation diagnostics in render bundles and prevent unsupported missed sections from emitting misleading protected geometry.

## 2026-05-02 00:15 CEST

### Goal Of This Session
- Make unsupported preserved missed approach legs explicit in geometry diagnostics.

### Decisions Locked
- The v3 package may preserve DF/CA/HM/HA/HF legs, but the current segment geometry kernel only constructs TF and RF.
- Non-constructible leg types now produce `UNSUPPORTED_LEG_TYPE` diagnostics instead of being silently ignored.
- If a missed section has only unsupported geometry legs, it emits diagnostics and no primary/secondary envelope.

### Files Changed
- `src/utils/procedureSegmentGeometry.ts`
- `src/utils/__tests__/procedureSegmentGeometry.test.ts`
- `docs/15-procedure-details-page-dev-log.md`

### Commands Run / Checks Passed
- `npm test -- --run src/utils/__tests__/procedureSegmentGeometry.test.ts`

### Current Status
- Missed approach sections can be semantically represented without misleading protected geometry.
- Tests cover DF/HM preserved-leg diagnostics and the no-envelope outcome for unsupported missed geometry.

### Known Blockers
- DF/CA/HM geometry construction remains future work.
- Procedure Details does not yet surface segment diagnostics as a dedicated validation table.

### Exact Next Recommended Step
- Surface segment diagnostics in Procedure Details so unsupported sections and RF/turn limitations are visible in the validation workspace.

## 2026-05-02 00:18 CEST

### Goal Of This Session
- Surface v3 render-bundle diagnostics in Procedure Details.

### Decisions Locked
- Procedure Details now derives a `ProcedureRenderBundle` from the selected `ProcedureDetailDocument`.
- Data Notes includes diagnostics relevant to the focused branch/legs.
- Diagnostics are displayed with severity and code so unsupported geometry, default tolerances, RF gaps, and turn limitations are visible in the validation workspace.

### Files Changed
- `src/components/ProcedureDetailsPage.tsx`
- `src/components/__tests__/ProcedureDetailsPage.test.tsx`
- `docs/15-procedure-details-page-dev-log.md`

### Commands Run / Checks Passed
- `npm test -- --run src/components/__tests__/ProcedureDetailsPage.test.tsx`
- `npm run build`

### Current Status
- Procedure Details is now a better validation workbench: source notes and geometry/render diagnostics are visible together.

### Known Blockers
- Diagnostics are text-only; selecting a diagnostic does not yet focus the related segment/leg on the chart.
- Full advanced geometry items remain incomplete: CA/DF/HM, FB/FO turns, LPV/GLS surfaces, RNP AR templates, and turning missed approach.

### Exact Next Recommended Step
- Add a final validation run across Python tests, frontend tests, and build, then summarize remaining v3 migration gaps clearly.

## 2026-05-02 00:23 CEST

### Goal Of This Session
- Run full validation after the RF, segment assessment, missed-section, and diagnostics stages.

### Commands Run / Checks Passed
- `conda run -n aviation pytest python/tests`
  - 73 passed.
- `npm test -- --run`
  - 95 passed.
- `npm run build`
  - TypeScript compile and Vite production build passed.

### Current Status
- Real RF metadata is tested from CIFP parser through full procedure-detail document export.
- RF centerline, RF parallel envelope metadata, and finite Case 2 inner-collapse handling are covered.
- 2D profile assessment now uses render-bundle segment geometry when available and reports station, cross-track, containment, vertical error, and basic events.
- Missed approach legs can be sectioned into `MISSED_S1` and `MISSED_S2`.
- Procedure Details surfaces render/geometry diagnostics in Data Notes.

### Remaining v3 Migration Gaps
- Full FAA RF OEA Case 1/Case 2 construction is not complete; the current Case 2 behavior is a conservative finite-geometry guard.
- FB/FO turn construction remains visual-fill only.
- DF/CA/HM/HA/HF geometry is preserved and diagnosed but not constructed.
- LPV/GLS W/X/Y, LNAV/VNAV vertical surfaces, RNP AR templates, and turning missed approach wind spiral remain future work.
- Diagnostic selection does not yet focus the related segment or leg on the Procedure Details charts.

## 2026-05-02 00:32 CEST

### Goal Of This Session
- Improve missed approach readability by making the `MISSED_S1` / `MISSED_S2` split visible in Procedure Details.

### Decisions Locked
- Missed section markers are derived from the selected procedure's `ProcedureRenderBundle`, not from ad hoc string parsing in the SVG layer.
- A `MISSED_S2` marker anchors at the segment start fix when that fix is visible in the current procedure-detail polyline.
- The same marker is drawn in both plan view and vertical profile so the section split is visible horizontally and vertically.
- Unsupported DF/HM missed geometry remains diagnosed in Data Notes; the marker only identifies the section transition.

### Files Changed
- `src/components/ProcedureDetailsPage.tsx`
- `src/components/__tests__/ProcedureDetailsPage.test.tsx`
- `src/index.css`
- `docs/15-procedure-details-page-dev-log.md`

### Commands Run / Checks Passed
- `npm test -- --run src/components/__tests__/ProcedureDetailsPage.test.tsx`
- `npm run build`

### Current Status
- Procedure Details now visually distinguishes the missed approach section split with an `S1/S2` marker.
- Regression coverage verifies the marker appears in both plan/profile contexts for a missed approach with a hold-boundary section split.

### Known Blockers
- Missed section surfaces are still not constructed.
- CA first-leg rendering and turning missed debug primitives remain future work.

### Exact Next Recommended Step
- Add explicit CA/DF/HM missed-leg badges/markers in the Procedure Details sequence and chart views before implementing their full geometry builders.

## 2026-05-02 00:38 CEST

### Goal Of This Session
- Make preserved missed approach leg semantics visible before full CA/DF/HM geometry construction exists.

### Decisions Locked
- Missed leg markers are derived from `ProcedureDetailDocument.branches[].legs[]`.
- CA/DF/HM/HA/HF legs in missed approach context are labeled on the nearest positioned start/end fix.
- Markers are rendered in both plan view and vertical profile.
- These markers are semantic badges only; unsupported geometry still emits diagnostics and no misleading envelope.

### Files Changed
- `src/components/ProcedureDetailsPage.tsx`
- `src/components/__tests__/ProcedureDetailsPage.test.tsx`
- `src/index.css`
- `docs/15-procedure-details-page-dev-log.md`

### Commands Run / Checks Passed
- `npm test -- --run src/components/__tests__/ProcedureDetailsPage.test.tsx`
- `npm run build`

### Current Status
- Procedure Details now distinguishes missed approach visually with:
  - dashed missed path;
  - outbound arrow;
  - `S1/S2` section split marker;
  - DF/HM/CA-style missed leg badges when those legs are present.

### Known Blockers
- CA/DF/HM/HA/HF geometry builders remain future work.
- Turning missed approach debug primitives remain future work.

### Exact Next Recommended Step
- Start implementing the first constructible missed approach geometry path, beginning with straight DF missed section support and tests.

## 2026-05-02 00:43 CEST

### Goal Of This Session
- Add the first constructible missed approach geometry path: positioned DF direct-to-fix legs.

### Decisions Locked
- DF is treated as a straight fix-to-fix geodesic when both start and end fixes are positioned.
- DF can now participate in segment centerline and primary/secondary envelope construction.
- CA/HM/HA/HF remain unsupported geometry and continue to emit diagnostics.
- The existing `buildTfLeg` path is generalized to accept TF/DF because both first-pass implementations are positioned fix-to-fix paths.

### Files Changed
- `src/utils/procedureSegmentGeometry.ts`
- `src/utils/__tests__/procedureSegmentGeometry.test.ts`
- `docs/15-procedure-details-page-dev-log.md`

### Commands Run / Checks Passed
- `npm test -- --run src/utils/__tests__/procedureSegmentGeometry.test.ts`
- `npm run build`

### Current Status
- Straight DF missed sections can now be rendered as protected segment geometry when source fixes are available.
- Unsupported missed geometry diagnostics now focus on CA/HM/HA/HF rather than all missed leg types.

### Known Blockers
- CA course-to-altitude geometry is still not constructible.
- HM/HA/HF hold geometry is still not constructible.
- Straight missed section 1/2 surface rules are not yet implemented beyond centerline/envelope.

### Exact Next Recommended Step
- Add render-bundle coverage for DF missed geometry and ensure Procedure Details diagnostics no longer report DF as unsupported when a positioned DF leg is present.

## 2026-05-02 00:47 CEST

### Goal Of This Session
- Cover DF missed geometry through the full procedure-detail normalizer and render-bundle path.

### Decisions Locked
- The regression starts from `ProcedureDetailDocument` shape rather than a hand-built `ProcedurePackage`.
- A positioned DF missed leg must produce:
  - `MISSED_S1` segment type;
  - sampled centerline;
  - primary envelope;
  - secondary envelope.
- Render diagnostics must not classify positioned DF as `UNSUPPORTED_LEG_TYPE`.

### Files Changed
- `src/data/__tests__/procedureRenderBundle.test.ts`
- `docs/15-procedure-details-page-dev-log.md`

### Commands Run / Checks Passed
- `npm test -- --run src/data/__tests__/procedureRenderBundle.test.ts src/components/__tests__/ProcedureDetailsPage.test.tsx`
- `npm run build`

### Current Status
- DF missed approach geometry is covered from source document through canonical package and render bundle.

### Known Blockers
- CA and holding missed geometry remain unsupported.
- Missed section surfaces remain future work.

### Exact Next Recommended Step
- Add a lightweight missed section surface/ribbon classification object so `MISSED_S1` and straight `MISSED_S2` can become independent renderable objects beyond centerline/envelope.
