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
