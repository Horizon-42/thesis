# Global Review Implementation Dev Log

## 2026-05-02 Step 1 - RNAV(RNP) RNP AR classification

- Scope: fix the real generated-data path where RNAV(RNP) procedure-detail documents use `procedureFamily: "RNAV_RNP"` with `approachModes: ["RNP AR"]`.
- Change: centralized RNP AR detection in `procedurePackageAdapter`, maps those finals to `FINAL_RNP_AR`, `RNP_AR_0_3`, no secondary area, and `RNP_AR_VERTICAL`.
- Tests: added adapter and render-bundle coverage so generated RNAV(RNP) documents report the unsupported `RNP_AR_FINAL_TEMPLATE` instead of silently falling back to LNAV baseline geometry.
- Verification: `npm test -- --run src/data/__tests__/procedurePackageAdapter.test.ts src/data/__tests__/procedureRenderBundle.test.ts`.
- Commit: this entry is included in `Fix RNAV RNP AR final classification`.

## 2026-05-02 Step 2 - Legacy OCS isolation

- Scope: prevent the old FAF-to-threshold OCS hook from mixing by default with v3 procedure segment OEA/OCS geometry.
- Change: `useOcsLayer` now accepts an `enabled` flag and only loads legacy entities when the legacy toggle is on; the control label is renamed to `Legacy FAF OCS Debug`.
- Tests: `npm test -- --run src/components/__tests__/ControlPanel.test.tsx`; `npm run build`.
- Commit: this entry is included in `Isolate legacy OCS debug layer`.

## 2026-05-02 Step 3 - Shared render bundle load cache

- Scope: reduce duplicate procedure-detail fetch/parse/geometry-build work across `ProcedurePanel`, 3D procedure layer, and runway trajectory profile.
- Change: added an airport/context keyed promise cache inside `procedureRenderBundle`, with a test-only clear helper and retry support after failed loads.
- Tests: `npm test -- --run src/data/__tests__/procedureRenderBundle.test.ts`; `npm run build`.
- Commit: this entry is included in `Cache procedure render bundle loads`.

## 2026-05-02 Step 4 - Lazy procedure branch entity creation

- Scope: reduce Cesium entity count by avoiding eager creation for hidden procedure branches.
- Change: `useProcedureSegmentLayer` now stores loaded render bundle data, creates only visible branch entities, and lazily creates a hidden branch the first time selector state makes it visible.
- Tests: `npm test -- --run src/hooks/__tests__/useProcedureSegmentLayer.test.ts`; `npm run build`.
- Commit: this entry is included in `Create procedure branch entities lazily`.

## 2026-05-02 Step 5 - CIFP adapter caching

- Scope: reduce repeated `FAACIFP18` scans and cifparse path point extraction during multi-procedure preprocessing.
- Change: added bounded per-process caches for path point lines and exact source-line maps, plus `clear_cifp_parser_caches()` for tests and long-running tooling.
- Tests: `conda run -n aviation /Users/liudongxu/opt/miniconda3/envs/aviation/bin/python3.13 -m pytest aeroviz-4d/python/tests/test_preprocess_procedures.py -q`.
- Commit: pending.
