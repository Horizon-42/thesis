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
