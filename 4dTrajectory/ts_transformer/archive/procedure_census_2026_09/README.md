# procedure_census_2026_09 — how observed arrivals fly the coded RNAV(GPS) procedures

Archived 2026-09-26 from `docs/` (layout rule L20: `docs/` holds documents), with the report the scripts produced.
Moved byte for byte; off the import path. Read-only measurements over the harvest and the CIFP procedure documents,
run on the v5 roster (42,650 arrivals); the live harvest has since changed. They do not run where they sit
(`Path(__file__).parents[3]` was the repository root from `docs/`); a new census is a new runner and a new report.

- `measure_procedure_adherence.py` (2026-09-04, `388574f4`) — per runway: coded off-axis entry, where the flight became
  established on the LPV cone, and the inside-FAF share inside the cone and the glidepath window. Behind
  `docs/2026-09-04_procedure_constraints_design.zh.md`.
- `measure_procedure_compliance.py` (2026-09-12, `863acbae`) — the nested tiers A (final) / B (+ IF) / C (+ a coded
  transition). Its report and tier-C flight list are `docs/2026-09-12_procedure_compliance_census.zh.md` and
  `docs/2026-09-12_procedure_compliance_tier_C_flights.json` here (the report's commands name the scripts' old
  `docs/` paths).

The two-tier model's procedure constraint (the glidepath lower edge) is measured by the live runner
`prior_procedure_check` instead (post-training design §3).
