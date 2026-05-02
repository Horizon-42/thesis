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
- Commit: pending.
