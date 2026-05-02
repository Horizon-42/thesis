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

## 2026-05-02 00:52 CEST

### Goal Of This Session
- Add a lightweight missed section surface object to the render-bundle pipeline.

### Decisions Locked
- New `procedureMissedGeometry.ts` owns missed section surface classification.
- `MISSED_S1` with a primary envelope becomes `MISSED_SECTION1_ENVELOPE`.
- Straight `MISSED_S2` with a primary envelope becomes `MISSED_SECTION2_STRAIGHT_ENVELOPE`.
- Turning `MISSED_S2` is explicitly diagnosed instead of being represented as a straight surface.
- This remains an envelope/surface classification layer, not full FAA missed approach section surface construction.

### Files Changed
- `src/utils/procedureMissedGeometry.ts`
- `src/utils/__tests__/procedureMissedGeometry.test.ts`
- `src/data/procedureRenderBundle.ts`
- `src/data/__tests__/procedureRenderBundle.test.ts`
- `docs/15-procedure-details-page-dev-log.md`

### Commands Run / Checks Passed
- `npm test -- --run src/utils/__tests__/procedureMissedGeometry.test.ts src/data/__tests__/procedureRenderBundle.test.ts`
- `npm run build`

### Current Status
- Render bundles can now expose missed section surfaces as independent objects, separate from generic segment envelopes.

### Known Blockers
- The 3D layer does not yet render missed section surfaces with distinct styling.
- Full section 1 / section 2 FAA construction is still future work.

### Exact Next Recommended Step
- Render `missedSectionSurface` in `useProcedureSegmentLayer` with distinct missed-approach styling and tests.

## 2026-05-02 00:57 CEST

### Goal Of This Session
- Render missed section surface objects in the 3D protected-geometry layer.

### Decisions Locked
- `useProcedureSegmentLayer` now renders `missedSectionSurface.primary` and optional `secondaryOuter` as independent Cesium polygon entities.
- Missed section surfaces use separate entity ids:
  - `-missed-surface-primary`;
  - `-missed-surface-secondary`.
- Missed section surfaces are vertically offset above generic envelopes so they can be inspected as distinct protected-geometry objects.

### Files Changed
- `src/hooks/useProcedureSegmentLayer.ts`
- `src/hooks/__tests__/useProcedureSegmentLayer.test.ts`
- `docs/15-procedure-details-page-dev-log.md`

### Commands Run / Checks Passed
- `npm test -- --run src/hooks/__tests__/useProcedureSegmentLayer.test.ts`
- `npm run build`

### Current Status
- 3D protected mode can now render missed section surfaces independently from generic segment envelopes.

### Known Blockers
- Styling is still first-pass and not yet split by missed section 1 vs straight section 2 vs turning section 2.
- Full FAA missed section surface geometry remains future work.

### Exact Next Recommended Step
- Add final targeted/full validation, then continue with CA course-to-altitude geometry design in the next phase.

## 2026-05-02 01:03 CEST

### Goal Of This Session
- Validate the missed-approach visual/geometry stages before moving into CA course-to-altitude design.

### Commands Run / Checks Passed
- `npm test -- --run`
  - 100 passed.
- `conda run -n aviation pytest python/tests`
  - 73 passed.
- `npm run build`
  - TypeScript compile and Vite production build passed.

### Current Status
- Missed approach is now visibly differentiated in Procedure Details and independently represented in protected-geometry render bundles.
- DF missed legs with positioned fixes can construct centerline/envelope geometry.
- Missed section surfaces render in 3D protected mode as independent objects.

### Remaining v3 Migration Gaps
- CA course-to-altitude geometry is still not implemented.
- Holding legs (`HM`/`HA`/`HF`) remain semantic markers plus diagnostics.
- Full FAA missed section 1/2 surface construction and turning missed wind spiral remain future work.

## 2026-05-02 01:12 CEST

### Goal Of This Session
- Add source-data plumbing needed for future CA course-to-altitude geometry.

### Decisions Locked
- CIFP CA/CF/FA course is parsed from the ARINC course field as tenths of a degree.
- Procedure-detail leg paths now export `courseDeg` when available.
- `ProcedurePackageLeg.outboundCourseDeg` is populated from exported `courseDeg`.
- No CA geometry is constructed in this phase because climb/termination distance rules still need explicit design.

### Files Changed
- `python/cifp_parser.py`
- `python/preprocess_procedures.py`
- `python/tests/test_preprocess_procedures.py`
- `src/data/procedureDetails.ts`
- `src/data/procedurePackageAdapter.ts`
- `src/data/__tests__/procedurePackageAdapter.test.ts`
- `docs/15-procedure-details-page-dev-log.md`

### Commands Run / Checks Passed
- `conda run -n aviation pytest python/tests/test_preprocess_procedures.py -k "course or branch_document_exports_course"`
- `npm test -- --run src/data/__tests__/procedurePackageAdapter.test.ts`
- `npm run build`

### Current Status
- Future CA geometry builders can now consume real course metadata instead of guessing heading from adjacent fixes.

### Known Blockers
- CA termination still needs a validated endpoint model.
- Real procedure-detail public assets must be regenerated before app data contains CA course metadata.

### Exact Next Recommended Step
- Design and implement a conservative CA debug geometry that shows course direction and altitude termination semantics without pretending to know certified endpoint distance.

## 2026-05-02 01:20 CEST

### Goal Of This Session
- Make preserved missed-approach CA first-leg course semantics visible in Procedure Details while keeping it clearly separate from certified CA endpoint/surface construction.

### Decisions Locked
- CA missed-leg markers now include the parsed outbound course when available, for example `CA 305 deg`.
- The 2D Procedure Details plan view draws a short dashed CA course ray from the positioned CA anchor fix.
- The ray is intentionally fixed-length debug geometry; it communicates course direction only and does not claim a CA termination point, climb distance, or protected missed-approach surface.

### Files Changed
- `src/components/ProcedureDetailsPage.tsx`
- `src/components/__tests__/ProcedureDetailsPage.test.tsx`
- `src/index.css`
- `docs/15-procedure-details-page-dev-log.md`

### Commands Run / Checks Passed
- `npm test -- --run src/components/__tests__/ProcedureDetailsPage.test.tsx`
- `npm run build`

### Current Status
- Procedure Details can now visually distinguish the CA first leg in missed approach with both semantic labeling and course-direction context.

### Known Blockers
- CA course-to-altitude endpoint construction still needs a validated termination model.
- The CA debug ray is not yet a reusable geometry object in the render-bundle layer.

### Exact Next Recommended Step
- Promote CA debug/course metadata into a dedicated geometry helper and render-bundle object so 2D/3D views can consume the same conservative semantics without duplicating UI-only logic.

## 2026-05-02 01:45 CEST

### Goal Of This Session
- Promote CA missed-approach course direction from UI-local logic into a shared render-bundle geometry object.

### Decisions Locked
- Added `MissedCourseGuideGeometry` for preserved CA legs in missed approach segments.
- The guide stores:
  - segment/leg identity;
  - positioned start fix;
  - outbound course;
  - required altitude metadata;
  - a fixed-length guide line marked `COURSE_DIRECTION_ONLY`.
- Procedure Details now consumes CA course guide geometry from `ProcedureRenderBundle` instead of recomputing the ray inside the component.
- Missing CA course or start-fix position is diagnosed as `SOURCE_INCOMPLETE`.

### Files Changed
- `src/utils/procedureMissedGeometry.ts`
- `src/utils/__tests__/procedureMissedGeometry.test.ts`
- `src/data/procedureRenderBundle.ts`
- `src/data/__tests__/procedureRenderBundle.test.ts`
- `src/components/ProcedureDetailsPage.tsx`
- `src/hooks/__tests__/useProcedureSegmentLayer.test.ts`
- `docs/15-procedure-details-page-dev-log.md`

### Commands Run / Checks Passed
- `npm test -- --run src/utils/__tests__/procedureMissedGeometry.test.ts`
- `npm test -- --run src/data/__tests__/procedureRenderBundle.test.ts`
- `npm test -- --run src/components/__tests__/ProcedureDetailsPage.test.tsx`
- `npm run build`

### Current Status
- CA first-leg visibility is now backed by render-bundle geometry and can be reused by other views.

### Known Blockers
- The CA guide remains a semantic/debug guide, not a certified course-to-altitude endpoint or surface.
- 3D protected mode does not render the CA guide yet.

### Exact Next Recommended Step
- Render CA course guides in 3D protected mode with a distinct diagnostic style and tests.

## 2026-05-02 02:00 CEST

### Goal Of This Session
- Render shared CA missed-approach course guides in the 3D protected-procedure layer.

### Decisions Locked
- `useProcedureSegmentLayer` now renders each `missedCourseGuide` as an independent high-offset polyline.
- CA guide entities use a `-ca-course-guide-` id suffix so they remain separate from centerline, envelope, OEA, connector, and missed-surface entities.
- The visual remains diagnostic/semantic: a course-direction guide only, not a CA endpoint, surface, or containment object.

### Files Changed
- `src/hooks/useProcedureSegmentLayer.ts`
- `src/hooks/__tests__/useProcedureSegmentLayer.test.ts`
- `docs/15-procedure-details-page-dev-log.md`

### Commands Run / Checks Passed
- `npm test -- --run src/hooks/__tests__/useProcedureSegmentLayer.test.ts`
- `npm run build`

### Current Status
- CA first-leg direction can now be inspected in both Procedure Details 2D plan view and 3D protected mode.

### Known Blockers
- Full CA course-to-altitude termination geometry remains unresolved.
- CA guides are not yet surfaced in validation metrics; they are visual/semantic primitives only.

### Exact Next Recommended Step
- Add a data regeneration test path for `run_asd-b_fetch_and_generate.py` so real exported procedure assets can carry `courseDeg` into the app.

## 2026-05-02 02:15 CEST

### Goal Of This Session
- Add a root-pipeline procedure regeneration path so exported app assets can carry parsed CA `courseDeg` metadata without requiring a separate manual command.

### Decisions Locked
- `run_asd-b_fetch_and_generate.py` now supports `--generate-procedures`.
- When enabled, the root pipeline runs `aeroviz-4d/python/preprocess_procedures.py` after CZML generation with:
  - `--include-all-rnav`;
  - configurable `--cifp-root`;
  - configurable `--procedure-output`;
  - optional `--include-procedure-transitions`;
  - optional `--procedure-charts-root`.
- Added a Python test that imports the root script and verifies the generated subprocess command without running network fetches or real asset generation.

### Files Changed
- `../run_asd-b_fetch_and_generate.py`
- `python/tests/test_run_asd_b_pipeline.py`
- `docs/15-procedure-details-page-dev-log.md`

### Commands Run / Checks Passed
- `conda run -n aviation pytest python/tests/test_run_asd_b_pipeline.py`
- `conda run -n aviation pytest python/tests/test_preprocess_procedures.py -k "course or branch_document_exports_course" python/tests/test_run_asd_b_pipeline.py`

### Current Status
- The main root pipeline can now regenerate RNAV/RNP procedure assets and procedure-details JSON after trajectory CZML generation.

### Known Blockers
- The new pipeline option is opt-in; existing invocations do not regenerate procedure data unless `--generate-procedures` is provided.
- The test validates command construction, not the full external CIFP preprocessing output.

### Exact Next Recommended Step
- Run full frontend and Python validation before continuing into the next unresolved v3 design item.

## 2026-05-02 02:30 CEST

### Goal Of This Session
- Run full validation after the CA course guide and root pipeline stages.

### Commands Run / Checks Passed
- `npm test -- --run`
  - 104 passed.
- `conda run -n aviation pytest python/tests`
  - 76 passed.
- `npm run build`
  - TypeScript compile and Vite production build passed.

### Current Status
- CA first-leg missed approach visibility is covered in 2D Procedure Details and 3D protected mode.
- The root data pipeline can optionally regenerate procedure assets that carry CA `courseDeg`.
- Full current regression suite is green.

### Remaining v3 Migration Gaps
- Certified CA course-to-altitude endpoint and protected surface construction are still not implemented.
- Holding leg geometry (`HM`/`HA`/`HF`) remains semantic/diagnostic only.
- Turning missed approach section 2 wind spiral/TIA debug geometry remains future work.
- LPV/GLS W/X/Y and LNAV/VNAV vertical surfaces are still future work beyond the LNAV baseline.

### Exact Next Recommended Step
- Continue with a small, non-misleading missed-approach improvement: add explicit turning-missed diagnostics/flags for missed section 2 segments that contain holding or turn-trigger legs.

## 2026-05-02 02:45 CEST

### Goal Of This Session
- Make turning missed approach gaps explicit at the canonical package layer.

### Decisions Locked
- `MISSED_S2` segments containing `HM`/`HA`/`HF`/`RF` now get `constructionFlags.isTurningMissedApproach`.
- The adapter emits `TURNING_MISSED_UNIMPLEMENTED` diagnostics for these segments.
- This is diagnostic/schema clarity only; no TIA, early/late baseline, or wind-spiral geometry is constructed in this phase.

### Files Changed
- `src/data/procedurePackageAdapter.ts`
- `src/data/__tests__/procedurePackageAdapter.test.ts`
- `docs/15-procedure-details-page-dev-log.md`

### Commands Run / Checks Passed
- `npm test -- --run src/data/__tests__/procedurePackageAdapter.test.ts`
- `npm run build`

### Current Status
- Turning missed approach cases are no longer only implied by unsupported hold legs; they are explicitly flagged and diagnosed.

### Known Blockers
- Turning missed protected geometry remains future work.

### Exact Next Recommended Step
- Expose turning missed diagnostics in Procedure Details visual notes/markers so users can identify why section 2 has no surface.

## 2026-05-02 03:20 CEST

### Goal Of This Session
- Surface turning missed approach diagnostics in Procedure Details.

### Decisions Locked
- Procedure Details now filters render diagnostics using the canonical package branch id instead of `procedureUid`.
- The existing Data Notes panel displays `TURNING_MISSED_UNIMPLEMENTED` for focused missed section 2 cases.
- Added regression coverage to ensure a missed approach with HM section 2 shows the turning-missed diagnostic.

### Files Changed
- `src/components/ProcedureDetailsPage.tsx`
- `src/components/__tests__/ProcedureDetailsPage.test.tsx`
- `docs/15-procedure-details-page-dev-log.md`

### Commands Run / Checks Passed
- `npm test -- --run src/components/__tests__/ProcedureDetailsPage.test.tsx`
- `npm run build`

### Current Status
- Users can now see why a turning missed section is diagnostic-only instead of silently missing a protected surface.

### Known Blockers
- The diagnostic is text-only; no TIA/wind-spiral debug overlay exists yet.

### Exact Next Recommended Step
- Add a lightweight 2D/3D debug marker for turning missed section 2 anchor points without constructing wind spiral geometry.

## 2026-05-02 04:30 CEST

### Goal Of This Session
- Add a non-misleading turning missed approach debug anchor.

### Decisions Locked
- Added `MissedTurnDebugPointGeometry` for `MISSED_S2` segments flagged as turning missed approach.
- The debug point records:
  - anchor fix id;
  - trigger leg types such as `HM`;
  - `DEBUG_MARKER_ONLY` construction status;
  - positioned geo/world point.
- Render bundles now expose `missedTurnDebugPoint`.
- 3D protected mode renders the anchor as an independent `-turning-missed-anchor` point entity.
- No TIA, wind spiral, early/late turn baseline, or protected surface is implied.

### Files Changed
- `src/utils/procedureMissedGeometry.ts`
- `src/utils/__tests__/procedureMissedGeometry.test.ts`
- `src/data/procedureRenderBundle.ts`
- `src/data/__tests__/procedureRenderBundle.test.ts`
- `src/hooks/useProcedureSegmentLayer.ts`
- `src/hooks/__tests__/useProcedureSegmentLayer.test.ts`
- `docs/15-procedure-details-page-dev-log.md`

### Commands Run / Checks Passed
- `npm test -- --run src/utils/__tests__/procedureMissedGeometry.test.ts src/data/__tests__/procedureRenderBundle.test.ts src/hooks/__tests__/useProcedureSegmentLayer.test.ts`
- `npm run build`

### Current Status
- Turning missed approach section 2 now has schema flagging, Data Notes diagnostics, and a 3D debug anchor.

### Known Blockers
- Procedure Details 2D does not yet draw a distinct turning-missed anchor beyond the existing S1/S2 and HM markers.
- Full turning missed protected geometry remains future work.

### Exact Next Recommended Step
- Run a final full validation pass, then summarize the completed staged work and remaining v3 gaps.

## 2026-05-02 04:45 CEST

### Goal Of This Session
- Run final full validation for the staged CA and turning-missed migration work.

### Commands Run / Checks Passed
- `npm test -- --run`
  - 106 passed.
- `conda run -n aviation pytest python/tests`
  - 76 passed.
- `npm run build`
  - TypeScript compile and Vite production build passed.

### Current Status
- CA missed approach course metadata flows from parser/export into package/render-bundle objects.
- CA first-leg direction is visible in Procedure Details and 3D protected mode as course-guide/debug geometry.
- Turning missed section 2 is flagged, diagnosed, and visible in 3D as a debug-only anchor.
- The root `run_asd-b_fetch_and_generate.py` pipeline can optionally regenerate procedure assets after CZML generation.

### Remaining v3 Migration Gaps
- CA course-to-altitude endpoint construction still requires an explicit climb/termination model.
- Full missed approach section 1/2 protected surfaces are still first-pass envelope classifications, not certified FAA construction.
- Turning missed TIA/early-late baseline/wind-spiral geometry is not implemented.
- LPV/GLS W/X/Y, LNAV/VNAV, and RNP AR vertical/surface models remain future phases.

## 2026-05-02 10:30 CEST

### Goal Of This Session
- Make the turning missed debug anchor visible in Procedure Details 2D charts.

### Decisions Locked
- Added `MissedTurnDebugMarker` derived from `ProcedureRenderBundle.missedTurnDebugPoint`.
- The plan view and vertical profile now render a `Turn debug` anchor at the missed section 2 start/trigger point.
- The marker remains explicitly diagnostic: it identifies the turn-trigger anchor only and does not draw TIA, wind spiral, early/late baseline, or protected surface geometry.

### Files Changed
- `src/components/ProcedureDetailsPage.tsx`
- `src/components/__tests__/ProcedureDetailsPage.test.tsx`
- `src/index.css`
- `docs/15-procedure-details-page-dev-log.md`

### Commands Run / Checks Passed
- `npm test -- --run src/components/__tests__/ProcedureDetailsPage.test.tsx`
- `npm run build`

### Current Status
- Turning missed section 2 is now visible in Data Notes, 3D protected mode, and both Procedure Details 2D chart contexts.

### Known Blockers
- Turning missed remains debug-marker-only until TIA/wind-spiral geometry is designed and implemented.

### Exact Next Recommended Step
- Start the next v3 priority by adding a typed vertical-surface diagnostic bundle for final approach modes that are currently collapsed to LNAV.

## 2026-05-02 10:35 CEST

### Goal Of This Session
- Add structured final approach surface status for modes that are currently collapsed to LNAV baseline geometry.

### Decisions Locked
- Added `FinalApproachSurfaceStatus` and `FinalApproachSurfaceType`.
- LNAV baseline final OEA is now recorded as a constructed surface type.
- Collapsed LPV/LNAV-VNAV/LNAV procedures now expose missing mode-specific surfaces in render bundles:
  - `LPV_W`;
  - `LPV_X`;
  - `LPV_Y`;
  - `LNAV_VNAV_OCS`.
- Added `FINAL_VERTICAL_SURFACE_UNIMPLEMENTED` diagnostics when required final vertical/surface objects are missing.
- No LPV/GLS W/X/Y or LNAV/VNAV OCS geometry is constructed in this phase.

### Files Changed
- `src/data/procedurePackage.ts`
- `src/utils/procedureSurfaceGeometry.ts`
- `src/utils/__tests__/procedureSurfaceGeometry.test.ts`
- `src/data/procedureRenderBundle.ts`
- `src/data/__tests__/procedureRenderBundle.test.ts`
- `docs/15-procedure-details-page-dev-log.md`

### Commands Run / Checks Passed
- `npm test -- --run src/utils/__tests__/procedureSurfaceGeometry.test.ts src/data/__tests__/procedureRenderBundle.test.ts`
- `npm run build`

### Current Status
- Final approach mode gaps are now typed render-bundle state, not only adapter warnings.

### Known Blockers
- Mode-specific LPV/GLS W/X/Y and LNAV/VNAV OCS geometry remains future work.

### Exact Next Recommended Step
- Surface final approach missing-surface status in Procedure Details Data Notes and the focused sequence context.

## 2026-05-02 10:40 CEST

### Goal Of This Session
- Surface structured final approach surface status in Procedure Details.

### Decisions Locked
- Focused Sequence now shows built final surface types and missing final surface types for the focused branch.
- Data Notes now includes a structured final surface status line when mode-specific surfaces are missing.
- The existing `FINAL_VERTICAL_SURFACE_UNIMPLEMENTED` diagnostic remains visible alongside the structured summary.

### Files Changed
- `src/components/ProcedureDetailsPage.tsx`
- `src/components/__tests__/ProcedureDetailsPage.test.tsx`
- `src/index.css`
- `docs/15-procedure-details-page-dev-log.md`

### Commands Run / Checks Passed
- `npm test -- --run src/components/__tests__/ProcedureDetailsPage.test.tsx`
- `npm run build`

### Current Status
- Users can now see that LNAV baseline is built while LPV W/X/Y and LNAV/VNAV OCS are missing for collapsed multimode procedures.

### Known Blockers
- This is still status/diagnostic UI only; mode-specific vertical surfaces are not constructed yet.

### Exact Next Recommended Step
- Add a small 3D placeholder/status primitive for missing final vertical surfaces so protected mode can reveal collapsed final modes without drawing fake W/X/Y geometry.

## 2026-05-02 10:45 CEST

### Goal Of This Session
- Make missing final vertical/surface status visible in 3D protected mode without inventing unsupported geometry.

### Decisions Locked
- `useProcedureSegmentLayer` now renders a `-final-surface-status` point entity for final segments whose `FinalApproachSurfaceStatus` has missing surface types.
- The point is placed at a representative point on the final centerline and vertically offset above the other protected-mode primitives.
- This is a status marker only; it does not draw LPV W/X/Y, GLS W/X/Y, or LNAV/VNAV OCS geometry.

### Files Changed
- `src/hooks/useProcedureSegmentLayer.ts`
- `src/hooks/__tests__/useProcedureSegmentLayer.test.ts`
- `docs/15-procedure-details-page-dev-log.md`

### Commands Run / Checks Passed
- `npm test -- --run src/hooks/__tests__/useProcedureSegmentLayer.test.ts`
- `npm run build`

### Current Status
- Collapsed final modes are now visible in Procedure Details and protected 3D mode as explicit status/diagnostic objects.

### Known Blockers
- Real final vertical/surface construction remains future work.

### Exact Next Recommended Step
- Run full frontend/Python validation after the final-surface status stages.

## 2026-05-02 10:50 CEST

### Goal Of This Session
- Run full validation after the 2D turning missed marker and final-surface status stages.

### Commands Run / Checks Passed
- `npm test -- --run`
  - First run exposed an expected-count regression in `ProcedurePanel.test.tsx`: final-surface missing diagnostics increased the warning summary from 5 to 8.
  - Updated the test expectation.
  - Rerun passed: 108 passed.
- `conda run -n aviation pytest python/tests`
  - 76 passed.
- `npm run build`
  - TypeScript compile and Vite production build passed.

### Current Status
- Procedure Details 2D, ProcedurePanel warning counts, render bundles, and protected 3D mode now account for final-surface diagnostic status.

### Remaining v3 Migration Gaps
- Real LPV/GLS W/X/Y and LNAV/VNAV OCS geometry is still not implemented.
- CA endpoint construction and full turning missed wind-spiral/TIA geometry remain future work.

## 2026-05-02 11:00 CEST

### Goal Of This Session
- Preserve the remaining v3 migration priority list in the project docs so future work can continue after context compaction.

### Remaining Major Items And Priority

#### P0: CA course-to-altitude real geometry
- Current state:
  - CA `courseDeg` flows from CIFP parser/export into procedure package and render bundle.
  - Procedure Details and 3D protected mode show CA course guides.
  - Current CA primitive is `COURSE_DIRECTION_ONLY`; no endpoint, climb distance, or certified surface is claimed.
- Required functions:
  - Build a CA endpoint model from start fix, outbound course, required altitude, starting altitude, and explicit climb-gradient/default climb model.
  - Emit endpoint status such as `ESTIMATED_ENDPOINT`, `SOURCE_EXACT`, or `INSUFFICIENT_CLIMB_MODEL`.
  - Build CA centerline geometry from start fix to endpoint.
  - Build conservative CA missed section 1 envelope/surface when endpoint is available.
  - Keep diagnostics clear when altitude/climb inputs are missing.
- Acceptance checks:
  - CA with course and altitude can produce a typed endpoint and centerline.
  - CA without enough climb/altitude information produces diagnostics and no fake endpoint.
  - CA is not mislabeled or constructed as TF/DF.

#### P1: LNAV/VNAV OCS vertical surface
- Current state:
  - `FinalApproachSurfaceStatus` lists `LNAV_VNAV_OCS` as missing.
  - No LNAV/VNAV OCS geometry or vertical assessment exists.
- Required functions:
  - Add `LnavVnavOcsGeometry` as a final vertical surface object.
  - Use centerline stationing plus GPA/TCH or an explicitly diagnosed fallback.
  - Render OCS as an independent 3D translucent surface.
  - Extend segment assessment to report vertical error/clearance against the OCS.
- Acceptance checks:
  - `LNAV_VNAV_OCS` moves from missing to constructed when required source data exists.
  - Missing GPA/TCH or equivalent data produces diagnostics, not fake geometry.
  - Vertical deviation assessment can identify aircraft below the OCS.

#### P2: LPV/GLS W/X/Y surfaces
- Current state:
  - `FinalApproachSurfaceStatus` lists `LPV_W`, `LPV_X`, `LPV_Y` and GLS equivalents as missing.
  - Protected mode only shows a status marker; no W/X/Y geometry exists.
- Required functions:
  - Add LPV/GLS surface bundle objects for W, X, and Y surfaces.
  - Construct W/X/Y boundaries from final course, threshold/glidepath metadata, and mode-specific rules.
  - Render W/X/Y as independently identifiable protected-mode objects.
  - Update Procedure Details status so constructed W/X/Y surfaces replace missing-surface notes.
- Acceptance checks:
  - W/X/Y are separate typed geometry objects with independent entity ids.
  - Multimode LPV procedures no longer show LPV W/X/Y as missing once geometry is available.
  - LNAV baseline behavior remains unchanged.

#### P3: Turning missed approach debug primitives
- Current state:
  - Turning missed section 2 is flagged with `isTurningMissedApproach`.
  - Procedure Details and 3D protected mode show a debug-only anchor.
  - No TIA, early/late baseline, inside/outside turn, or wind-spiral primitive exists.
- Required functions:
  - Classify turn-at-altitude versus turn-at-fix.
  - Classify early/inside versus late/outside turn cases.
  - Add debug primitives for TIA boundary, early baseline, late baseline, nominal turn path, and wind spiral.
  - Mark estimated wind/turn primitives as `DEBUG_ESTIMATE_ONLY` or equivalent.
- Acceptance checks:
  - Turning missed cases produce explicit debug primitives when enough data exists.
  - Missing wind/turn inputs produce diagnostics.
  - Debug primitives are visually separate from certified protected surfaces.

#### P4: Full missed section 1 / section 2 protected surfaces
- Current state:
  - `MISSED_SECTION1_ENVELOPE` and straight `MISSED_SECTION2_STRAIGHT_ENVELOPE` are first-pass envelope classifications.
  - They are not full FAA section 1/2 protected-surface construction.
- Required functions:
  - Build section 1 surfaces by final type: LNAV, LNAV/VNAV, LPV/GLS.
  - Build straight section 2 surfaces from TF/DF centerlines with section-specific primary/secondary boundaries.
  - Build turning section 2 surfaces after P3 debug/classification primitives exist.
  - Connect missed surfaces to climb-gradient and vertical assessment.
- Acceptance checks:
  - Section 1 and section 2 are independent geometry objects.
  - Straight section 2 can be assessed for horizontal and vertical containment.
  - Turning section 2 does not claim compliance until TIA/wind-spiral rules are implemented.

#### P5: RNP AR final / RF-specific protected templates
- Current state:
  - RF centerline and first-pass RF envelope handling exist.
  - Full RNP AR final templates and RF final protected areas are not implemented.
- Required functions:
  - Preserve and render `FINAL_RNP_AR` as a distinct final type instead of collapsing to LNAV.
  - Add RF-to-PFAF and RF final protection templates.
  - Tighten RF Case 1/Case 2 acceptance and diagnostics.
  - Add FO/FB turn rule diagnostics where applicable.
- Acceptance checks:
  - RNP AR procedures produce distinct final-surface status/geometry.
  - RF final surfaces are not treated as straight LNAV OEA.
  - Unsupported FO/FB or RNP changes produce explicit diagnostics.

### Recommended Execution Order
1. P0 CA endpoint model scaffold: helper, types, diagnostics, tests; do not replace current guide until endpoint status is reliable.
2. P0 CA centerline/envelope integration into render bundle and 2D/3D views.
3. P1 LNAV/VNAV OCS geometry and vertical assessment.
4. P2 LPV/GLS W/X/Y surface bundle.
5. P3 turning missed debug primitives.
6. P4 full missed section surfaces.
7. P5 RNP AR / RF-specific final templates.

### Boundary Rules
- Do not claim certified FAA compliance for any estimated/debug geometry.
- If source data or model assumptions are incomplete, emit diagnostics and avoid fake protected surfaces.
- Keep every stage modular, covered by focused tests, logged here, and committed separately.

## 2026-05-02 11:15 CEST

### Goal Of This Session
- Start P0 by adding a CA course-to-altitude endpoint model scaffold.

### Decisions Locked
- Added `MissedCaEndpointGeometry` and `MissedCaEndpointStatus`.
- Added `buildMissedCaEndpoints` in `procedureMissedGeometry.ts`.
- The helper estimates endpoint distance from:
  - positioned CA start fix;
  - outbound CA course;
  - target altitude constraint;
  - start-fix elevation;
  - explicit climb gradient, or documented default 200 ft/NM climb model.
- Estimated endpoints are marked `ESTIMATED_ENDPOINT`.
- Missing source semantics produce `CA_ENDPOINT_NOT_CONSTRUCTIBLE` diagnostics and no geometry.
- This phase does not replace the existing CA course guide and does not construct CA missed section surfaces.

### Files Changed
- `src/data/procedurePackage.ts`
- `src/utils/procedureMissedGeometry.ts`
- `src/utils/__tests__/procedureMissedGeometry.test.ts`
- `docs/15-procedure-details-page-dev-log.md`

### Commands Run / Checks Passed
- `npm test -- --run src/utils/__tests__/procedureMissedGeometry.test.ts`
- `npm run build`

### Current Status
- P0 endpoint math and safety diagnostics are now covered by focused tests.

### Known Blockers
- CA endpoint geometry is not yet wired into `ProcedureRenderBundle`.
- CA centerline/envelope construction remains next work.

### Exact Next Recommended Step
- Add CA endpoint geometry to render bundles alongside course guides, with status/diagnostics visible but without replacing the current guide yet.

## 2026-05-02 11:20 CEST

### Goal Of This Session
- Expose estimated CA endpoints through render bundles without replacing the existing CA course guide.

### Decisions Locked
- Added `missedCaEndpoints` to `ProcedureSegmentRenderBundle`.
- `buildProcedureRenderBundle` now calls `buildMissedCaEndpoints` for missed segments.
- CA endpoint geometry is exposed alongside `missedCourseGuides`.
- Existing segment geometry remains unchanged: CA legs still do not become TF/DF centerlines in this phase.

### Files Changed
- `src/data/procedureRenderBundle.ts`
- `src/data/__tests__/procedureRenderBundle.test.ts`
- `docs/15-procedure-details-page-dev-log.md`

### Commands Run / Checks Passed
- `npm test -- --run src/data/__tests__/procedureRenderBundle.test.ts`
- `npm run build`

### Current Status
- Render bundles now carry both the conservative CA direction guide and the estimated CA course-to-altitude endpoint object.

### Known Blockers
- Procedure Details and 3D protected mode do not yet render the estimated CA endpoint/status.
- CA centerline and envelope are not constructed yet.

### Exact Next Recommended Step
- Render CA estimated endpoints as diagnostic/status markers in Procedure Details and protected 3D mode.

## 2026-05-02 11:30 CEST

### Goal Of This Session
- Render estimated CA endpoints as diagnostic/status markers.

### Decisions Locked
- Procedure Details plan view and vertical profile now render `CA end est` markers from `missedCaEndpoints`.
- Marker hover/click focuses the CA start fix, not a synthetic endpoint fix.
- `useProcedureSegmentLayer` now renders each estimated CA endpoint as a `-ca-endpoint-` point entity.
- The marker is diagnostic/status only; it does not imply certified CA termination or protected surface construction.

### Files Changed
- `src/components/ProcedureDetailsPage.tsx`
- `src/components/__tests__/ProcedureDetailsPage.test.tsx`
- `src/hooks/useProcedureSegmentLayer.ts`
- `src/hooks/__tests__/useProcedureSegmentLayer.test.ts`
- `src/index.css`
- `docs/15-procedure-details-page-dev-log.md`

### Commands Run / Checks Passed
- `npm test -- --run src/components/__tests__/ProcedureDetailsPage.test.tsx src/hooks/__tests__/useProcedureSegmentLayer.test.ts`
- `npm run build`

### Current Status
- P0 endpoint model is visible in render bundles, Procedure Details, and protected 3D mode.

### Known Blockers
- CA estimated endpoint is not yet used to build CA centerline/envelope geometry.
- CA surface construction remains future work.

### Exact Next Recommended Step
- Build CA centerline geometry from estimated endpoints, then use it as a constructible missed segment path while keeping estimated-status diagnostics visible.

## 2026-05-02 11:40 CEST

### Goal Of This Session
- Add CA centerline geometry helper from estimated CA endpoints.

### Decisions Locked
- Added `MissedCaCenterlineGeometry`.
- Added `buildMissedCaCenterlines`.
- Centerlines are sampled from estimated CA endpoint geometry and marked `ESTIMATED_CENTERLINE`.
- This phase does not yet replace `SegmentGeometryBundle.centerline`; it only creates the typed centerline object needed for the next integration step.

### Files Changed
- `src/utils/procedureMissedGeometry.ts`
- `src/utils/__tests__/procedureMissedGeometry.test.ts`
- `docs/15-procedure-details-page-dev-log.md`

### Commands Run / Checks Passed
- `npm test -- --run src/utils/__tests__/procedureMissedGeometry.test.ts`
- `npm run build`

### Current Status
- CA endpoint and sampled CA centerline math are now both available as typed helper geometry.

### Known Blockers
- Render bundles do not yet expose CA centerlines.
- The segment geometry kernel still treats CA as unsupported for centerline/envelope construction.

### Exact Next Recommended Step
- Expose CA centerlines in render bundles and render them separately before using them to backfill missed segment geometry.

## 2026-05-02 11:50 CEST

### Goal Of This Session
- Expose estimated CA centerlines through render bundles and protected 3D mode.

### Decisions Locked
- Added `missedCaCenterlines` to `ProcedureSegmentRenderBundle`.
- `buildProcedureRenderBundle` now builds CA centerlines from `missedCaEndpoints`.
- `useProcedureSegmentLayer` renders estimated CA centerlines as independent `-ca-centerline-` polylines.
- Segment kernel geometry is still unchanged; this is separate estimated CA geometry, not yet a replacement for generic segment centerline/envelope construction.

### Files Changed
- `src/data/procedureRenderBundle.ts`
- `src/data/__tests__/procedureRenderBundle.test.ts`
- `src/hooks/useProcedureSegmentLayer.ts`
- `src/hooks/__tests__/useProcedureSegmentLayer.test.ts`
- `docs/15-procedure-details-page-dev-log.md`

### Commands Run / Checks Passed
- `npm test -- --run src/data/__tests__/procedureRenderBundle.test.ts src/hooks/__tests__/useProcedureSegmentLayer.test.ts`
- `npm run build`

### Current Status
- CA estimated endpoint and centerline are both visible in render bundles and protected 3D mode.

### Known Blockers
- CA segment envelope/surface construction remains separate and incomplete.
- The segment geometry kernel still reports CA as unsupported for core centerline construction.

### Exact Next Recommended Step
- Backfill CA-only missed segment geometry from estimated CA centerlines, filtering obsolete unsupported-CA diagnostics while preserving estimated-status diagnostics.

## 2026-05-02 11:21 CEST

### Goal Of This Session
- Backfill CA-only missed segment geometry from estimated CA centerlines.

### Decisions Locked
- Added `buildMissedCaSegmentGeometry` as a missed-geometry helper instead of folding estimated CA behavior into the generic TF/RF/DF segment kernel.
- Backfill only applies when the missed segment's non-IF geometry legs are all CA and all CA centerlines are constructible.
- Backfilled CA geometry builds a straight estimated centerline plus primary/secondary envelopes, allowing section 1 missed surfaces to be produced.
- Obsolete `UNSUPPORTED_LEG_TYPE` diagnostics for backfilled CA legs are filtered.
- The replacement emits `ESTIMATED_CA_GEOMETRY` so the UI/data layer still states that this is debug geometry derived from a climb model, not certified source protection.

### Files Changed
- `src/data/procedurePackage.ts`
- `src/data/procedureRenderBundle.ts`
- `src/data/__tests__/procedureRenderBundle.test.ts`
- `src/utils/procedureMissedGeometry.ts`
- `docs/15-procedure-details-page-dev-log.md`

### Commands Run / Checks Passed
- `npm test -- --run src/data/__tests__/procedureRenderBundle.test.ts src/utils/__tests__/procedureMissedGeometry.test.ts`
- `npm run build`

### Current Status
- CA-only missed section 1 segments can now participate in the core render-bundle geometry path with centerline, envelope, and missed section surface.
- CA course guides, estimated endpoints, and estimated centerlines remain separately visible diagnostic primitives.

### Known Blockers
- Mixed CA/HM/HA/HF turning missed sections are still not backfilled as straight CA segments.
- The CA endpoint still uses the documented climb model and is not certified source geometry.

### Exact Next Recommended Step
- Add explicit Procedure Details/3D styling for estimated CA missed surfaces so users can distinguish CA-estimated protection from source-backed DF/TF/RF protection.

## 2026-05-02 11:23 CEST

### Goal Of This Session
- Make CA-estimated missed surfaces visually and structurally distinguishable from source-backed missed surfaces.

### Decisions Locked
- `MissedSectionSurfaceGeometry` now carries `constructionStatus`.
- Existing DF/TF/RF-backed missed section surfaces report `SOURCE_BACKED`.
- CA backfilled missed section surfaces report `ESTIMATED_CA` when the segment geometry contains `ESTIMATED_CA_GEOMETRY`.
- The 3D procedure layer renders estimated CA missed surfaces with CA-specific naming and orange material instead of the generic missed-surface yellow.

### Files Changed
- `src/utils/procedureMissedGeometry.ts`
- `src/utils/__tests__/procedureMissedGeometry.test.ts`
- `src/data/__tests__/procedureRenderBundle.test.ts`
- `src/hooks/useProcedureSegmentLayer.ts`
- `src/hooks/__tests__/useProcedureSegmentLayer.test.ts`
- `docs/15-procedure-details-page-dev-log.md`

### Commands Run / Checks Passed
- `npm test -- --run src/utils/__tests__/procedureMissedGeometry.test.ts src/data/__tests__/procedureRenderBundle.test.ts src/hooks/__tests__/useProcedureSegmentLayer.test.ts`
- `npm run build`

### Current Status
- Estimated CA missed surfaces are now data-tagged and visually distinct in protected 3D mode.

### Known Blockers
- Procedure Details 2D views still show CA endpoints/labels but do not yet draw surface polygons.
- Mixed turning missed geometry remains debug-marker-only unless straight constructible geometry is available.

### Exact Next Recommended Step
- Continue P0/P1 validation by adding broader full-suite checks, then move into the highest-priority remaining advanced missed/turning geometry item.

## 2026-05-02 11:24 CEST

### Goal Of This Session
- Run full validation after the CA missed geometry stages.

### Commands Run / Checks Passed
- `npm test -- --run`
  - 22 test files passed.
  - 113 tests passed.
- `conda run -n aviation pytest python/tests`
  - 76 tests passed in the `aviation` conda environment.

### Current Status
- P0 CA course-to-altitude estimated endpoint, centerline, envelope/surface integration, 2D markers, 3D primitives, and estimated-surface styling are validated.

### Exact Next Recommended Step
- Start P1 LNAV/VNAV OCS geometry as a separate staged implementation.

## 2026-05-02 11:30 CEST

### Goal Of This Session
- Start P1 by adding LNAV/VNAV OCS geometry when explicit vertical source data exists.

### Decisions Locked
- Added `LnavVnavOcsGeometry` as an independent final vertical surface object.
- OCS construction requires explicit `gpaDeg` and `tchFt` from `VerticalRule`; missing values emit `SOURCE_INCOMPLETE` and no fake geometry.
- The first OCS implementation is marked `GPA_TCH_SLOPE_ESTIMATE`; it uses GPA/TCH and LNAV lateral OEA widths, but does not claim VEB-specific certified construction.
- `FinalApproachSurfaceStatus` can now report `LNAV_VNAV_OCS` as constructed and distinguish `MODE_SPECIFIC_SURFACES_CONSTRUCTED`.
- The procedure package adapter now passes `glidepathAngleDeg` and `thresholdCrossingHeightFt` from vertical profiles into final vertical rules.
- Protected 3D mode renders LNAV/VNAV OCS as its own lime translucent surface.

### Files Changed
- `src/data/procedurePackage.ts`
- `src/data/procedurePackageAdapter.ts`
- `src/data/procedureRenderBundle.ts`
- `src/data/__tests__/procedurePackageAdapter.test.ts`
- `src/data/__tests__/procedureRenderBundle.test.ts`
- `src/hooks/useProcedureSegmentLayer.ts`
- `src/hooks/__tests__/useProcedureSegmentLayer.test.ts`
- `src/utils/procedureSurfaceGeometry.ts`
- `src/utils/__tests__/procedureSurfaceGeometry.test.ts`
- `docs/15-procedure-details-page-dev-log.md`

### Commands Run / Checks Passed
- `npm test -- --run src/utils/__tests__/procedureSurfaceGeometry.test.ts src/data/__tests__/procedureRenderBundle.test.ts src/hooks/__tests__/useProcedureSegmentLayer.test.ts src/data/__tests__/procedurePackageAdapter.test.ts`
- `npm run build`

### Current Status
- LNAV/VNAV OCS is available as a typed render-bundle geometry and visible in protected 3D mode when GPA/TCH are present.

### Known Blockers
- Vertical assessment still compares aircraft altitude against route/profile samples, not against the new OCS geometry.
- Existing generated procedure assets currently have null GPA/TCH until the exporter or source data provides those values.

### Exact Next Recommended Step
- Add OCS-specific vertical assessment helpers/events so aircraft below LNAV/VNAV OCS can be detected independently of the display profile.

## 2026-05-02 11:33 CEST

### Goal Of This Session
- Add LNAV/VNAV OCS-specific vertical assessment events.

### Decisions Locked
- `HorizontalPlateAssessmentSegment` can now carry `verticalReferenceSurfaceType`.
- When a render-bundle segment has `lnavVnavOcs`, runway-profile assessment uses the OCS centerline and OCS half-width samples instead of the generic segment centerline/envelope for that assessment segment.
- Vertical deviations against OCS emit `VERTICAL_OCS` events with `BELOW_OCS` or `ABOVE_OCS` labels.
- Existing profile-based vertical deviations still emit `VERTICAL_DEVIATION` with `BELOW_PROFILE` or `ABOVE_PROFILE`.

### Files Changed
- `src/utils/runwayProfileGeometry.ts`
- `src/utils/__tests__/runwayProfileGeometry.test.ts`
- `src/utils/procedureSegmentAssessment.ts`
- `src/utils/__tests__/procedureSegmentAssessment.test.ts`
- `docs/15-procedure-details-page-dev-log.md`

### Commands Run / Checks Passed
- `npm test -- --run src/utils/__tests__/procedureSegmentAssessment.test.ts src/utils/__tests__/runwayProfileGeometry.test.ts src/utils/__tests__/procedureSurfaceGeometry.test.ts`
- `npm run build`

### Current Status
- Aircraft/profile points can now be assessed against LNAV/VNAV OCS where OCS geometry is present in the render bundle.

### Known Blockers
- The visible runway-profile summary still labels the numeric value generically as vertical error.
- Generated procedure assets still need real GPA/TCH source values before real procedures construct OCS.

### Exact Next Recommended Step
- Run full frontend validation, then start P2 LPV/GLS W/X/Y surface scaffolding if no regressions appear.

## 2026-05-02 11:34 CEST

### Goal Of This Session
- Run full frontend validation after P1 LNAV/VNAV OCS geometry and assessment stages.

### Commands Run / Checks Passed
- `npm test -- --run`
  - 22 test files passed.
  - 118 tests passed.

### Current Status
- P1 LNAV/VNAV OCS typed geometry, 3D rendering, render-bundle status, and OCS-based vertical assessment are validated.

### Exact Next Recommended Step
- Start P2 LPV/GLS W/X/Y surface scaffolding as a separate staged implementation.

## 2026-05-02 11:37 CEST

### Goal Of This Session
- Start P2 by scaffolding independently typed LPV/GLS W/X/Y surface objects.

### Decisions Locked
- Added `PrecisionFinalSurfaceGeometry` for `LPV_W`, `LPV_X`, `LPV_Y`, `GLS_W`, `GLS_X`, and `GLS_Y`.
- W/X/Y construction requires explicit GPA/TCH and a positioned final centerline with LNAV lateral OEA.
- The scaffold is marked `GPA_TCH_DEBUG_ESTIMATE`; it uses GPA/TCH plus scaled LNAV lateral widths and does not claim certified LPV/GLS W/X/Y construction.
- `FinalApproachSurfaceStatus` can now remove LPV/GLS W/X/Y from missing surfaces when debug-estimate geometry exists.
- Protected 3D mode renders each precision surface with an independent `-precision-...` entity id.

### Files Changed
- `src/utils/procedureSurfaceGeometry.ts`
- `src/utils/__tests__/procedureSurfaceGeometry.test.ts`
- `src/data/procedureRenderBundle.ts`
- `src/data/__tests__/procedureRenderBundle.test.ts`
- `src/hooks/useProcedureSegmentLayer.ts`
- `src/hooks/__tests__/useProcedureSegmentLayer.test.ts`
- `docs/15-procedure-details-page-dev-log.md`

### Commands Run / Checks Passed
- `npm test -- --run src/utils/__tests__/procedureSurfaceGeometry.test.ts src/data/__tests__/procedureRenderBundle.test.ts src/hooks/__tests__/useProcedureSegmentLayer.test.ts`
- `npm run build`

### Current Status
- P2 has typed, independently rendered LPV/GLS W/X/Y debug-estimate surfaces behind explicit source-data requirements.

### Known Blockers
- Full LPV/GLS W/X/Y construction still needs VEB/mode-specific dimensions instead of scaled LNAV lateral widths.
- Generated procedure assets still need real GPA/TCH values to construct these surfaces for real procedures.

### Exact Next Recommended Step
- Run full frontend validation, then continue with P3 turning missed debug primitives.

## 2026-05-02 11:38 CEST

### Goal Of This Session
- Run full frontend validation after P2 precision final surface scaffold.

### Commands Run / Checks Passed
- `npm test -- --run`
  - 22 test files passed.
  - 120 tests passed.

### Current Status
- P2 debug-estimate LPV/GLS W/X/Y surface scaffolding is validated with the existing frontend suite.

### Exact Next Recommended Step
- Continue with P3 turning missed debug primitives.

## 2026-05-02 12:44 CEST

### Goal Of This Session
- Add first turning missed debug-estimate primitives beyond the existing anchor marker.

### Decisions Locked
- Added `MissedTurnDebugPrimitiveGeometry` for turning missed section 2 debug overlays.
- Built five primitive types when section 2 has a positioned anchor plus course and turn direction:
  - `TIA_BOUNDARY`
  - `EARLY_TURN_BASELINE`
  - `LATE_TURN_BASELINE`
  - `NOMINAL_TURN_PATH`
  - `WIND_SPIRAL`
- The primitives classify turn trigger semantics as turn-at-fix or turn-at-altitude from the leg type.
- All new primitives are marked `DEBUG_ESTIMATE_ONLY`.
- Missing modeled wind/aircraft turn inputs emit a `SOURCE_INCOMPLETE` diagnostic; the wind spiral and TIA are fixed debug assumptions, not certified geometry.
- Protected 3D mode renders each primitive as an independent `-turning-missed-...` polyline.

### Files Changed
- `src/utils/procedureMissedGeometry.ts`
- `src/utils/__tests__/procedureMissedGeometry.test.ts`
- `src/data/procedureRenderBundle.ts`
- `src/data/__tests__/procedureRenderBundle.test.ts`
- `src/hooks/useProcedureSegmentLayer.ts`
- `src/hooks/__tests__/useProcedureSegmentLayer.test.ts`
- `docs/15-procedure-details-page-dev-log.md`

### Commands Run / Checks Passed
- `npm test -- --run src/utils/__tests__/procedureMissedGeometry.test.ts src/data/__tests__/procedureRenderBundle.test.ts src/hooks/__tests__/useProcedureSegmentLayer.test.ts`
- `npm run build`

### Current Status
- P3 now has TIA, early/late baseline, nominal turn path, and wind-spiral debug overlays in render bundles and protected 3D mode.

### Known Blockers
- These primitives are not certified turning missed protected surfaces.
- Real wind, aircraft category/speed, bank angle, reaction time, and FAA TIA parameters are not yet modeled.
- Procedure Details 2D still shows only the turning-missed anchor marker.

### Exact Next Recommended Step
- Run full frontend validation, then continue with either 2D Procedure Details display for turning missed primitives or P4 full missed section surface refinement.
