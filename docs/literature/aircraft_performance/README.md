# Aircraft performance parameters for the types OpenAP lacks: sources, licences and coverage (2026-09-23)

**Why this folder exists.** The point-mass approach model needs, per aircraft type, the landing mass
(MLW), wing area S, a landing lift limit (C_L,max,landing, or an approach-speed law), a drag polar
(C_D0, k) and the installed maximum thrust. Today these come from **OpenAP 2.4**. A type OpenAP does not
have falls back to A320 dynamics. That is wrong for business jets, turboprops and piston GA. This survey
checks which published sources could replace the fallback, and under which licence. It records, for each
source, the publisher, version, licence, the parameters it gives and its coverage of our fleet.

**The fleet.** `missing_types.csv` (199 rows, sha256 `bad4a0d4…a5ecc0`, written by the parent session
from the FAA Aircraft Characteristics Database and the census of the training cohort). It holds
**199 ICAO types with 13,643 training flights**: 188 types / 12,552 flights with `no_dynamics(A320 fallback)`
(not 183, as the task text said) and 11 types / 1,091 flights with `openap_2_4_synonym`. Everything below is
weighted by `train_flights`.

**How to read it.**
- **Quotes are copied** from the saved files and kept short; "…" marks a cut. Longer verbatim clauses
  are in [`excerpts/licence_and_scope_clauses.txt`](excerpts/licence_and_scope_clauses.txt).
- **"(reading)"** marks a step that no source states, for example "ANP ID X *is* type Y" or
  "this engine designation is the same engine as that EDB row".
- **Nothing comes from memory.** Where a source was not found or could not be opened, the cell says
  `unknown` or is blank, and [§9](#9-what-could-not-be-verified) lists it.
- **Per-type results** are in [`coverage_matrix.csv`](coverage_matrix.csv), one row per type. The column
  legend is in [§8](#8-coverage_matrixcsv-columns). **Pinned numbers** are in [`excerpts/`](excerpts/).
  Each excerpt file names its source file and that file's sha256 in its header.
- **Raw downloads** are under `data/aircraft_performance/<source>/`, which is git-ignored.
  [`download.sh`](download.sh) re-fetches every public file there. It never overwrites a file and prints
  each sha256. From a git worktree, set `AIRCRAFT_PERF_DATA` to the main tree's data directory. A test run
  on 2026-09-23 re-downloaded 17 files: 16 were byte-identical, and the one live HTML page differed.
- `airframe_facts*.{csv,md}`, `download_airframes.sh` and `data/aircraft_performance/airframes/` belong to a
  separate pack in this folder. This survey did not touch them.
- **Segments** used below are explicit lists in the build script: **airliner** = BCS1/BCS3, CRJ1/2/7,
  E75S, E135, B712, A306, A30B, B722, B735, B753, B762, B764, B78X, A339, MD82/83/88, AT43, AT73, SB20,
  D328, E120. **military** = C30J, C27J, T38, F5, TEX2. **rotorcraft** = ICAO description `H…`. The rest
  is split by the engine letter of the ICAO description: business jet (J), turboprop (T), piston GA (P).

| segment | types | train flights |
|---|---:|---:|
| airliner | 25 | 3,863 |
| business jet | 61 | 6,143 |
| turboprop | 29 | 1,162 |
| piston GA | 71 | 2,439 |
| rotorcraft | 8 | 20 |
| military | 5 | 16 |
| **all** | **199** | **13,643** |

## At a glance

| source | publisher · version · date | what it gives for the approach model | coverage of our 199 types (types / train flights) | may a thesis publish the values? | verdict |
|---|---|---|---|---|---|
| **OpenAP 2.6.2** (§1) | TU Delft, J. Sun · 2.6.2 · PyPI 2026-09-22 | MTOW, MLW, OEW, S, span, flap geometry, C_D0/k/e, engine options and static thrust | **0 native.** 11 synonyms / 1,091, the same as 2.4: the `data/` tree is byte-identical | yes. Package LGPL-3.0; `data/LICENSE` is the GPL-3.0 text | usable, but it closes none of the gap |
| **Poll–Schumann (PS)** in pycontrails (§2) | Contrails.org / Breakthrough Energy · pycontrails 0.63.5 (2026-09-10) · parameter file dated 2025-03-28 · papers in *Aeronaut. J.* 2021–2025 | MTOM, MLM, OEM, S, span, AR, sweep, ψ₀ (C_D0 = C_f(Re)·ψ₀), Oswald e from δ₂/AR/sweep, total static thrust F₀₀, C_L at design optimum. **Clean configuration only** | 17 native / 3,089 + 2 synonyms / 812. Airliners 18 of 25 types (3,738 of 3,863 flights); one business jet (GLF5); no turboprop, piston or GA | yes. Apache-2.0; the Part 3 paper is CC BY 4.0 | **usable now** for airliner masses, S, clean polar and thrust. It has no flap or gear drag and no C_L,max |
| **BADA 3 / BADA 4** (§3) | EUROCONTROL · BADA 3: 264 models + 1,641 synonyms; BADA 4: 126 models (Nov 2025 slides) | BADA 3: C_D0/C_D2 per configuration, gear drag, V_stall per configuration, thrust. BADA 4: non-clean drag and C_L,max per flap and gear position | **No public per-type list for current releases.** The newest public list is BADA 3.8 (2010): 43 model / 2,960 + 77 synonym / 4,191; 79 / 6,492 unknown. BADA 4 is unknown for all 199 | **no.** Datasets *and* "calculation results/output" are confidential. The licence forbids naming aircraft types in type-to-type comparisons and requires EUROCONTROL review before publication | licence is free, "up to two weeks" to answer. **Coefficients cannot be published**, and per-type results need EUROCONTROL clearance |
| **ANP** (§4) | EASA (legacy v2.3, from EUROCONTROL/FAA/Volpe, release note 2020-10-14) · substitution list 22/02/2018 · current verified data on request | per flap setting: **D** (approach CAS = D·√W) and **R** (drag/lift, gear down); B/C for take-off; jet thrust coefficients; propeller efficiency and power; MLW; default approach steps | native (reading) 38 / 3,621, of which **33 / 2,279 carry approach D/R**; official noise substitute for 40 more / 6,002; none 121 / 4,020. No light props (< 8,618 kg) in the substitution table | **unresolved.** The legacy tables are a public download and the EASA site says "Reproduction is authorised, provided the source is acknowledged, save where otherwise stated". The ANP T&C (Issue 3) restrict ANP Data to noise contours, forbid per-type benchmarking and forbid publication without the manufacturer's consent | **usable for modelling. Ask EASA** whether the T&C bind the public legacy tables before printing coefficients |
| **EUROCONTROL APD** (§5) | EUROCONTROL, contentzone.eurocontrol.int · live pages · retrieved 2026-09-23 | approach IAS, V_at, MTOW, APC, wing span, power plant, "Performance Similarity" | listed 129 / 11,274. 9 of the top 40 are absent (1,734 flights). "Performance Similarity" is filled for **5 types only** (BCS1, BCS3, CL30, CL35, A339) | **only with permission**: "Copyright permission must be sought from EUROCONTROL" | corroboration only. It has no substitution list worth the name |
| **ICAO Engine Emissions Databank** (§6) | ICAO, hosted by EASA · issue 32, 20 March 2026 | rated take-off thrust F₀₀ per engine (kN), BPR, OPR, LTO fuel flows | engine found by exact designation for 35 types / 4,516. A family match (reading) for 3 more / 1,983 (BCS3, E545, A339). No turboprop or piston types. 48 jet or airliner types / 3,507 flights have no EDB engine | yes, with acknowledgement (EASA site notice; no EDB-specific terms found) | **usable now** for installed thrust of airliners and the larger business jets |
| FAA **AEDT** (§7) | FAA · AEDT 4 | its own fleet DB, with BADA 3/4 and ANP inside | not examined (licensed software) | BADA terms apply, via the integrated-tool licence | US $1,200 site licence + a BADA licence. Not pursued |
| BlueSky "BS" model (§7) | TU Delft · GitHub master | OEW, MTOW, MLW, S, C_fe, C_L,max for landing | 16 types, 2 of ours (D328, SB20; 9 flights) | yes (MIT) | **not acceptable as a primary source.** The values come from handbooks and SKYbrary, and D328 has MLW > MTOW |

**Coverage by segment** (native + substitute, types / train flights; "substitute" is the source's *own*
synonym or substitution table):

| source | airliner (25 / 3,863) | business jet (61 / 6,143) | turboprop (29 / 1,162) | piston GA (71 / 2,439) | rotorcraft + military (13 / 36) | all (199 / 13,643) |
|---|---|---|---|---|---|---|
| OpenAP 2.6.2 | 0 + 4 / 230 | 0 + 7 / 861 | 0 | 0 | 0 | 0 + 11 / 1,091 |
| PS (pycontrails) | 16 / 2,926 + 2 / 812 | 1 / 163 + 0 | 0 | 0 | 0 | 17 / 3,089 + 2 / 812 |
| BADA 3.8 public lists (2010) | 17 / 493 + 2 / 812 | 14 / 872 + 27 / 2,021 | 5 / 157 + 15 / 857 | 7 / 1,438 + 24 / 480 | 0 + 9 / 21 | 43 / 2,960 + 77 / 4,191 |
| BADA 4 | unknown | unknown | unknown | unknown | unknown | unknown |
| ANP v2.3 (native = reading) + official noise substitute | 13 / 557 + 10 / 3,299 | 15 / 1,430 + 30 / 2,703 | 5 / 208 + 0 | 5 / 1,426 + 0 | 0 | 38 / 3,621 + 40 / 6,002 |
| APD listed | 23 / 3,679 | 42 / 4,537 | 21 / 994 | 36 / 2,042 | 7 / 22 | 129 / 11,274 |
| ICAO EDB (exact + family) | 17 / 2,046 + 2 / 1,761 | 18 / 2,470 + 1 / 222 | — | — | — | 35 / 4,516 + 3 / 1,983 |

**The 40 types with the most training flights** (11,615 of 13,643 flights):

| type | train flights | segment | OpenAP 2.6.2 | PS | BADA 3.8 (2010) | ANP | APD Vat (kt IAS) | APD performance similarity | EDB |
|---|---:|---|---|---|---|---|---:|---|---|
| BCS3 | 1,760 | airliner | none | native | unknown | subst. 737700 | 140 | ICAO Code: BCS1 Notes: Lower rates of climb due to higher MTOM | family |
| P28A | 1,335 | piston | none | none | model | native PA28 | 65 | No data | — |
| CRJ7 | 808 | airliner | none | syn CRJ9 | syn CRJ9 | subst. CL601 | 135 | No data | yes |
| E55P | 778 | bizjet | none | none | unknown | subst. CNA560XL | 115 | No data | — |
| BCS1 | 607 | airliner | none | native | unknown | subst. 737500 | 135 | ICAO Code: BCS3 Notes: Better rates of climb (lower MTOM) | yes |
| CL35 | 585 | bizjet | none | none | unknown | none | 125 | ICAO Code: CL30 Notes: CL35 has a bit more range | yes |
| C68A | 523 | bizjet | none | none | unknown | none | — | not in APD | — |
| PC12 | 336 | turboprop | none | none | syn BE9L | none | 85 | No data | — |
| C700 | 335 | bizjet | none | none | unknown | none | — | not in APD | — |
| C56X | 327 | bizjet | syn C550 | none | syn C560 | native CNA560XL | 117 | No data | — |
| CL60 | 256 | bizjet | none | none | model | native CL601/CL600 | 130 | No data | yes |
| CL30 | 254 | bizjet | none | none | syn F2TH | subst. CL600 | 125 | ICAO Code: CL35 Notes: CL30 has a bit less range | yes |
| E545 | 222 | bizjet | none | none | unknown | subst. CNA750 | — | not in APD | family |
| C25B | 214 | bizjet | none | none | syn C560 | subst. CNA525C | 108 | No data | — |
| C208 | 202 | turboprop | none | none | syn PA27 | native CNA208 | 75 | No data | — |
| B712 | 188 | airliner | none | native | model | native 717200 | 139 | No data | yes |
| E75S | 183 | airliner | none | native | unknown | native EMB175 | — | not in APD | yes |
| GLF4 | 175 | bizjet | none | none | syn FA7X | native GIV | 140 | No data | yes |
| B350 | 172 | turboprop | none | none | syn PAY3 | none | 110 | No data | — |
| GLF5 | 163 | bizjet | syn GLF6 | native | syn FA7X | native GV | 140 | No data | yes |
| C750 | 159 | bizjet | none | none | model | native CNA750 | 130 | No data | yes |
| GLEX | 159 | bizjet | none | none | syn C750 | subst. F10065 | 125 | No data | yes |
| S22T | 157 | piston | none | none | unknown | none | — | not in APD | — |
| G280 | 156 | bizjet | none | none | unknown | subst. EMB145 | 134 | No data | yes |
| SR22 | 148 | piston | none | none | syn TRIN | none | 70 | No data | — |
| A306 | 145 | airliner | syn A332 | native | model | native A300-622R | 139 | No data | yes |
| H25B | 127 | bizjet | none | none | syn H25A | subst. IA1125 | 125 | No data | yes |
| C680 | 120 | bizjet | none | none | syn F2TH | native CNA680 | 125 | No data | — |
| LJ45 | 113 | bizjet | syn GLF6 | none | model | subst. LEAR35 | 140 | No data | — |
| C560 | 107 | bizjet | none | none | model | native CNA560E/CNA560U | 110 | No data | yes |
| BE20 | 100 | turboprop | none | none | model | none | 100 | No data | — |
| GA6C | 89 | bizjet | none | none | unknown | none | — | not in APD | — |
| GL7T | 88 | bizjet | none | none | unknown | none | — | not in APD | — |
| GL5T | 82 | bizjet | syn GLF6 | none | syn C750 | subst. GV | 122 | No data | yes |
| F2TH | 80 | bizjet | none | none | model | subst. CL600/CL601 | 110 | No data | yes |
| BE36 | 77 | piston | none | none | syn DA42 | none | 70 | No data | — |
| CRJ2 | 75 | airliner | syn E145 | none | model | subst. CL601 | 140 | No data | yes |
| C25M | 73 | bizjet | none | none | unknown | none | 105 | No data | — |
| SF50 | 69 | bizjet | none | none | unknown | none | — | not in APD | — |
| PC24 | 68 | bizjet | syn C550 | none | unknown | none | — | not in APD | — |

## 1. OpenAP 2.6.2

| item | value |
|---|---|
| Publisher | Junzi Sun (TU Delft), GitHub `junzis/openap`. Paper: Sun, Hoekstra, Ellerbroek, *OpenAP: An Open-Source Aircraft Performance Model for Air Transportation Studies and Simulations*, Aerospace 7(8):104, 2020, doi:10.3390/aerospace7080104 (CC BY 4.0, per Crossref) |
| Version / date | **2.6.2**, uploaded to PyPI 2026-09-22T20:10. GitHub `master` HEAD `a19a1cd` (2026-09-22, "Fix BADA 3 idle fuel and convergence; release 2.6.2") is the release commit |
| Files | `openap/openap-2.6.2-py3-none-any.whl` sha256 `c4a79325…8bf508a295`; `openap-2.6.2.tar.gz` `9c1b370d…b9bb1121d9d`; `openap-2.4-py3-none-any.whl` `33c8d136…4e8b0bf260`. All three equal the PyPI digests. Fetched with curl from the PyPI file URLs, the same files `pip download --no-deps` would fetch. Nothing was installed |
| Licence | wheel METADATA: "License: GNU LGPL v3". The package LICENSE is the LGPL-3.0 text. **`openap/data/LICENSE` is the GPL-3.0 text** ("GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007"). Either way the values may be redistributed and printed with attribution |
| Parameters | per aircraft YAML: `mtow`, `mlw`, `oew`, `wing.area`, `wing.span`, `flaps.*`, `engine.default/options`, `drag.cd0/k/e/gears`. `data/engine/engines.csv` holds `max_thrust` for 426 engines. It has no C_L,max |

**Coverage.** 37 native aircraft files (A19N…GLF6) and 26 native drag polars.
`aircraft/_synonym.csv` has 22 rows. CRJ9 is both native and a synonym row (→E75L), which leaves the 21
synonym-only designators the task text mentions. **None of our 199 types is native in 2.6.2.** 11 of them
are synonyms (1,091 flights):

| our type | aircraft synonym | drag-polar synonym | flights |
|---|---|---|---:|
| C56X | C550 | C550 | 327 |
| GLF5 | GLF6 | GLF6 | 163 |
| A306 | A332 | A332 | 145 |
| LJ45 | GLF6 | GLF6 | 113 |
| GL5T | GLF6 | GLF6 | 82 |
| CRJ2 | **E145** | **E75L** | 75 |
| PC24 | **C550** | **GLF6** | 68 |
| C25A | C550 | C550 | 61 |
| C525 | C550 | C550 | 47 |
| B762 | **B763** | **B752** | 9 |
| B735 | B734 | B734 | 1 |

**2.6.2 vs 2.4: nothing changed in the data.** `diff -rq` of `openap/data/` between the 2.4 and 2.6.2 wheels
is empty, and the installed `aeroviz` copy equals the 2.4 wheel. The synonyms the task asked about are
therefore the same in 2.6.2: LJ45→GLF6, CRJ2→E145, A306→A332, and AT72→E145 (AT72 is not in our list; our
AT73 has no OpenAP synonym at all). The release notes for 2.5.0 (thrust upper-bound slope), 2.6.0 (CasADi
backend for the BADA add-ons), 2.6.1 (optional matplotlib) and 2.6.2 (BADA 3 idle fuel) mention no aircraft
data. Note that the aircraft and drag-polar synonym tables disagree for CRJ2, PC24 and B762 (bold above). A
caller that takes mass from one table and drag from the other mixes two airframes.

**Assessment.** Upgrading to 2.6.2 buys nothing for coverage. OpenAP ships `addon/bada3.py` and
`addon/bada4.py`, which read a user's own licensed BADA files, but they carry no data.
Excerpt: [`excerpts/openap_2.6.2_aircraft_and_synonyms.txt`](excerpts/openap_2.6.2_aircraft_and_synonyms.txt).

## 2. Poll–Schumann (PS) model, pycontrails

| item | value |
|---|---|
| Publisher | pycontrails, Contrails.org and the Breakthrough Energy Foundation ("Copyright (c) 2021-present Contrails.org and the Breakthrough Energy Foundation", NOTICE). GitHub `contrailcirrus/pycontrails` |
| Version | pycontrails **0.63.5** (PyPI 2026-09-10). Wheel `pycontrails-0.63.5-cp312-cp312-manylinux…whl` sha256 `b4daa0db…e5e8acfdc66` |
| Static files | `pycontrails/models/ps_model/static/ps-aircraft-params-20250328.csv` (68 models × 44 columns, sha256 `2d462e5a…2978a0db`) and `ps-synonym-list-20250328.csv` (103 rows; 35 map to a different model, sha256 `a61b6b7f…7fc6bdcf`). Their git blob ids equal GitHub `main` at `0d49123` (2026-09-10), so 0.63.5 is current |
| Licence | PyPI `license_expression`: **Apache-2.0**. LICENSE: Apache License 2.0. The NOTICE file names no separate terms for the PS data |
| Papers | Part 1: *Aeronaut. J.* 125(1284):257–295, **doi:10.1017/aer.2020.62**. Part 2 (*determining the aircraft's characteristic parameters*): 125(1284):296–340, **doi:10.1017/aer.2020.124**. Both are "© The Author(s), 2020. Published by Cambridge University Press"; Crossref gives only the Cambridge terms; the author copies are on DLR elib. Part 3 (*full flight profile when the trajectory is specified*): 129(1334):825–861, **doi:10.1017/aer.2024.141**, **CC BY 4.0**. Engine paper: *A simple model for the estimation of turbofan engine performance in all airborne phases of flight*, doi:10.1017/aer.2024.92, CC BY 4.0. All four PDFs are saved under `pycontrails/papers/` (sha256 in [§10](#10-files)) |

**Parameters (units from the column names and the `PSAircraftEngineParams` docstring).** `MTOM_kg`,
`MLM_kg`, `MZFM_kg`, `OEM_i_kg`, `MPM_i_kg` [kg]; `Sref_m2` [m²]; `span_m`, `bf_m` [m]; `AR`, `cos_sweep`,
`delta_2` (wing–fuselage induced-drag interference factor), `psi_0` (geometry drag parameter); `CL_do` (design
optimum C_L); wave-drag `Xo`, `wing_constant`, `j_1`, `j_2`; `nominal_F00_ISA_kn` — the docstring says
"summed over all engines" [kN]; `mf_max_T_O_SLS_kg_s`, `mf_idle_SLS_kg_s` [kg/s]; `M_des`, `CT_des`,
`eta_1`, `eta_2`, `Mec`, `Tec`; `FL_max`, `MMO`; `pi_max_pa`, `pinf_co_pa` [Pa]; `nominal_opr/bpr/fpr`.
The drag law, as coded in `ps_model.py` with the paper equation numbers:
C_D = C_D0 + K·C_L² + C_Dw, where C_D0 = C_f·ψ₀ and C_f = 0.0269/Re^0.14 (Part 2 eq. 28), and
K = 1/(π·AR·e_ls) with e_ls = (1.075 with winglets, else 1)/(1 + 0.03 + δ₂ + k₁·π·AR) and
k₁ = 0.8·(1 − 0.53·cos Λ)·C_D0. **C_D0 therefore depends on the Reynolds number; it is not a constant.**

**Scope: the clean configuration only.** Part 3: "The flight phases of primary interest are the climb,
cruise, descent and holding, when the flaps and undercarriage are fully retracted…". §8 of Part 3 says
"the method, in its current form, only addresses the clean configuration". Below 3,000 ft it falls back to
the ICAO LTO thrust fractions (30 % on approach). For the final approach, PS gives masses, S, the
clean-wing polar and thrust. **It gives no flap or gear drag increments and no C_L,max.**

**Coverage** (see [`excerpts/ps_params_rows_for_our_types.csv`](excerpts/ps_params_rows_for_our_types.csv)):
- native, 17 types / 3,089 flights: BCS3, BCS1, B712, E75S, A306, A30B, B762, B764, B753, B735, B722, B78X,
  A339, MD82, MD83, E135 (airliners) and **GLF5**, the only business jet;
- synonym, 2 types / 812 flights: **CRJ7→CRJ9** (808) and **MD88→MD82** (4);
- not covered: the other 7 airliners (CRJ2, CRJ1, AT43, AT73, SB20, D328, E120; 125 flights). PS covers
  turbofans only, so none of the turboprops, pistons or 60 of the 61 business jets.

**Data-quality observations (reading).** (1) The B722 row gives `n_engine = 2` for the 727-200, a trijet
(ANP `727200` lists 3 engines). Its F₀₀ of 204 kN is a three-engine total. (2) pycontrails hard-codes
`j_3 = 70.0`, while Part 3 says "j 3 is … currently taken to be 40". j_3 only acts in wave drag, which does
not matter at approach Mach numbers. (3) The pycontrails docstring still cites Part 3 as "(2022) … submitted"
with a different title.

**Assessment.** **Usable now, and publishable (Apache-2.0).** It is the best open source for the airliner
part of the gap: 3,738 of 3,863 airliner flights. For the approach it must be combined with a landing-speed
law and a non-clean drag increment from elsewhere (ANP D/R; §4).

## 3. EUROCONTROL BADA 3 and BADA 4

| item | value |
|---|---|
| Publisher | EUROCONTROL (Innovation Hub, Brétigny). Product page https://www.eurocontrol.int/model/bada (saved 2026-09-23) |
| Versions and size | 2019-07-11 news: "BADA Family 3 now contains 250 original aircraft models, while 1,159 additional aircraft types are covered by so-called synonyms". Nov 2025 showcase slides: **BADA 3 "264 (95%) models + 1641 (5%) synonyms"; BADA 4 "126 models"**, "84.5% of ECAC IFR operations". The product page gives no release numbers |
| Public specification | BADA 4 User Manual **v1.4 (July 2025)**, EIH Technical/Scientific Report 12/11/22-58, "Distribution: Public", on GitHub `eurocontrol-bada/model-specifications` (EUPL-1.2 + amendment). pyBADA (EUPL-1.2) ships **dummy** datasets only. BADA 3.8 User Manual (2010, public) |
| What the licensed data would give | BADA 3.8 manual §3: stall speed per configuration (CR, IC, TO, AP, LD); C_D0/C_D2 for CR, AP and LD; C_D0,ΔLDG for the gear; thrust coefficients. BADA 4 manual §3.2 and §6.3.2: separate clean and non-clean drag, and "C_L,max is defined for each high-lift devices and landing gear position as a constant". This is exactly what the approach model needs |

**Licence (verbatim in [`excerpts/licence_and_scope_clauses.txt`](excerpts/licence_and_scope_clauses.txt)).**
- Cost: "BADA is free to use, its distribution is regulated through a license agreement to protect the
  interests of data providers" (slides). The licence is "royalty-free" (Art. 1.1).
- How to apply: register at EUROCONTROL OneSky Online, then request through the "BADA User Interface"
  (product page; "User guide for requesting BADA access"). "The provision of access is subject to
  EUROCONTROL's approval." **Lead time:** the FAA AEDT page says "EUROCONTROL can take up to two weeks
  to respond to an individual license request." Students can be covered under their university's licence
  (Art. 3.4) or sign their own.
- Use: "only for modelling (in the framework of ATM research and development activities)…" (Art. 2.2 a).
- **Publishing:** "Confidential Information" includes "…Datasets, any calculation results/output thereof…".
  The licensee may not "…make available the … Datasets, any calculation results/output thereof … to any
  person other than the End-Users without the explicit and prior written approval by EUROCONTROL" (2.2 c i),
  nor "display BADA on any … worldwide web site" (2.2 c vi). Art. 2.2 d: "…strictly prohibited from using
  BADA for any comparisons of any kind between aircraft types… Therefore YOU shall not mention any name of
  aircraft type in the publication." Art. 6.1: EUROCONTROL "reserves the right to review in advance all
  documents to be released and to disapprove…". The 2025 template (`B3B4BH_OPEN_STD_V03062025`, the Family 4
  link) keeps the same confidentiality, results and type-name clauses.
- **BADA 4 data access:** the 2019 standard template says "For BADA 4 family a sample dataset consisting of
  one jet, one turboprop and one piston aircraft model will be provided (additional datasets are subject to
  the prior authorization of the aircraft manufacturers)". The 2025 template drops that sentence and says
  "BADA consists of Datasets only". Whether a university gets the full BADA 4 set today is **not stated**.

**Coverage of our types.** EUROCONTROL publishes **no per-type list** for current BADA 3 or BADA 4
releases. The newest public per-type lists are from **BADA 3.8 (2010)**: the *Coverage of 2009 European Air
Traffic* report (EEC TR 2010/008, Annex B, which lists models, synonyms and types not covered) and the
*Synonym Aircraft Report 3.8* (EEC TR 2010/007). Coverage per that list, with the synonym's reference model
in `coverage_matrix.csv`: **43 types / 2,960 flights as a 3.8 model, 77 / 4,191 as a 3.8 synonym, 79 /
6,492 unknown**. Unknown includes every type certified after 2010 (BCS1/3, E75S, CL35, C68A, C700, E545/E550,
GA5C–GA8C, GL7T, SF50, PC24, A339, B78X…). **BADA 4: unknown for all 199.** The slides name "Cessna
Longitude, Cirrus VisionJet, Embraer Legacy 450/500, HondaJet, Learjet 75" as "recent models", i.e. C700,
SF50, E545/E550, HDJT. They do not say which family those are in.

**What the FAA ACD `faa_model_bada` column is.** The FAA data dictionary defines `Model_BADA` as the
"Euecontrol [sic] Base of Aircraft Data (BADA) Aircraft Model Name". Its stated source is "BADA Aircraft
Performance Database" at `contentzone.eurocontrol.int/aircraftperformance`, i.e. the APD of §5, not a BADA
release. (Reading) The strings have the BADA 3 `SYNONYM.NEW` format: MANUFACTURER + "NAME OR MODEL". For
66 of the 67 of our types that appear in pyBADA's public demo `SYNONYM.NEW`, the FAA string equals the demo
file's manufacturer + name (case-insensitive). The one exception is PA32. The BADA 3.8 manual says
SYNONYM.NEW "lists all aircraft types, which are supported by the BADA revision… whether they are supported
directly or by equivalence". So a filled `faa_model_bada`, which 165 of our types / 13,355 flights have,
suggests the type is in *some* BADA 3 release, **as a model or as a synonym**. It says nothing about BADA 4.
It is kept as the `faa_acd_model_bada` column and is not counted as coverage.

**How good are BADA 3 synonyms? (reading).** The 3.8 synonym choice uses "engine type; wake turbulence
category (WTC); maximum take-off weight (MTOW); max operating speed (VMO)…" (Synonym Report 3.8 §2), not
aerodynamics. The results include C208→PA27 (Caravan → Aztec), GLF4/GLF5→FA7X, BE36→DA42, PC12→BE9L and
SR22/C182/PA32→TRIN.

**Assessment.** It is the most complete source for configuration-dependent drag and lift limits. **The
values can never appear in the thesis, and per-type results need EUROCONTROL clearance.** Art. 2.2 d
conflicts with any per-type validation table that names types. **Usable after a licence, for internal
modelling only.**

## 4. ANP — Aircraft Noise and Performance database (EASA; ECAC Doc 29 / ICAO Doc 9911)

| item | value |
|---|---|
| Publisher | EASA, under Regulation (EU) 598/2014 Art. 7(3). aircraftnoisemodel.org now redirects to the EASA page (saved as `anp/easa_anp_data_page.html`). "EUROCONTROL published consecutive versions of the ANP data (versions 1.0 to 2.3) which are now published on the EASA website…" |
| Version | **Legacy ANP v2.3.** Release note dated **14/10/2020** (adds ATR72-212A/PW127F; updates A350-941 and 737-8). `archive_anp_v2.3.zip` holds 10 CSVs dated 2023-06-26, 155 aircraft. Newer EASA-verified data (14 aircraft/engine entries, including A330-941/Trent 7000 and FAL900EX/TFE731-60, `easa_verified_anp_aircraft_types.xlsx`) is **on request only** |
| Parameters | `ANP2.3_Aircraft.csv`: MTOW, **max gross landing weight**, max landing distance, **max sea-level static thrust**, engine count and type. `ANP2.3_Aerodynamic_coefficients.csv`: per flap ID, **B, C** (take-off), **D** and **R**. `Jet_engine_coefficients`: E, F, Ga, Gb, H (+K1–K4) per thrust rating. `Propeller_engine_coefficients`: efficiency and installed power. Default approach procedural steps and fixed-point profiles |
| How D and R enter (ECAC Doc 29 4th ed., Vol. 2, App. B11) | "The landing approach calibrated airspeed, VCA, is related to the landing gross weight by an equation of the same form as equation B-11" (eq. B-24), "where the coefficient D (kt/√lbf) corresponds to the landing flap setting". So **V_CA = D·√W** (W in lbf, V in kt; the radical over W is a graphic lost from the text layer, and the unit prints "kt/√lbf"). Thrust on the glideslope uses "a drag-to-lift ratio R appropriate for the flap setting with landing gear extended". Example (reading): 1900D, flap 35-A, D = 0.915858 → 0.915858·√14,940 ≈ 112 kt at MLW |
| Substitution table | `anp_aircraft_substitutions_-_jets_heavy_props_22022018_.xlsx` (sheets "by aircraft configuration", 19,565 rows from the EASA TCDSN and the ICAO NoiseDB, and "by ICAO code", 210 rows). "The ANP proxies were selected based on … MTOW…, the engine static thrust to MTOW ratio or the certified noise levels." Also: "Light propeller-driven aeroplanes (MTOW < 8,618 kg) are not included" |

**Licence.** Two texts apply, and they pull in different directions.
1. The legacy tables are a **public download** with no terms attached on the page. The EASA site notice
   says: "Reproduction is authorised, provided the source is acknowledged, save where otherwise stated."
2. The **EASA ANP Database Terms and Conditions, Issue 3 (03/06/2024)**, which bind users of the ANP
   Database: "The ANP Data is designed and solely provided to calculate noise contours around airports…".
   Use beyond that "shall be subject to the prior written consent of the ANP Data Provider" (6 a). "The ANP
   Data shall not be used to benchmark the noise or performance and procedures of specific aircraft types…
   without the prior written consent…" (6 b). Users "shall not … publish … the ANP Data, with the trademarks
   of the ANP Data Provider, without prior written consent" (6 g). Only four providers require that consent
   (`List_of_ANP_Data_Providers.pdf`, 2025-01-17): **Airbus, Boeing, Dassault Aviation, Embraer.**
Whether the T&C also govern the publicly posted legacy v2.3 tables is **not stated anywhere found**.
(Reading) Using D/R to set approach speeds in a trajectory model is outside the noise-contour purpose of
6 a. The conservative course is to publish derived results, cite the coefficients by table and ID rather
than reprint them, and ask environment@easa.europa.eu before printing any values.

**Coverage** (`anp` column; excerpts
[`anp_substitution_rows_for_our_types.tsv`](excerpts/anp_substitution_rows_for_our_types.tsv) and
[`anp_2.3_aircraft_and_aero_rows_for_referenced_ids.csv`](excerpts/anp_2.3_aircraft_and_aero_rows_for_referenced_ids.csv)):
- **native (reading: the ANP description names the same airframe), 38 types / 3,621 flights.**
  - airliners: A306→A300-622R, A30B→A300B4-203, B712→717200, E75S→EMB175, B762→767CF6/767JT9, B764→767400,
    B753→757300, B735→737500, B722→727200 family, MD82, MD83, D328→DO328, E120→EMB120.
  - business jets: C56X→CNA560XL, C750→CNA750, C680→CNA680, C510→CNA510, C25C→CNA525C,
    C560→CNA560E/CNA560U, GLF4→GIV, GLF5→GV, GLF3→GIIB, CL60→CL601/CL600, C650→CIT3, LJ35→LEAR35
    (described as "Learjet 36"), FA20→FAL20, EA50→ECLIPSE500, ASTR→IA1125.
  - turboprops: C208→CNA208, B190→1900D, C441→CNA441, DHC6→DHC6, PAY3→PA42.
  - piston: P28A→PA28, BE58→BEC58P (58P), C182→CNA182, T206→CNA20T, PA31→PA31.
  - **5 of these IDs have no approach D/R coefficient, only fixed-point profiles: PA28, MD82, MD83, 757300,
    PA31.** That leaves **33 types / 2,279 flights with a usable V_CA = D·√W law**. P28A (1,335 flights)
    is among the five without it.
- **official substitute only, 40 types / 6,002 flights.** These are *noise* proxies, not aerodynamic
  ones: BCS3→737700, BCS1→737500, CRJ7/CRJ2/CRJ1→CL601, E55P→CNA560XL, GLEX→**F10065 (Fokker 100)**,
  FA7X→**CRJ9-ER**, F900→**EMB14L**, G280/GALX/HA4T/E135→EMB145, H25B/H25C/G150/SBR1→IA1125,
  LJ45/LJ31/LJ55/FA10→LEAR35, AT43→DHC8, AT73→HS748A (the list predates the v2.3 ATR72 entry), SB20→SF340,
  MD88→MD83, and so on.
- **none, 121 types / 4,020 flights**: every light prop without a native ID, all rotorcraft, and the business
  jets newer than 2018 (CL35, C68A, C700, GA5C–GA8C, GL7T, E550, SF50, PC24, C25M, FA8X) plus B78X and A339.
  A339 is in the request-only EASA-verified list.

**Data-quality observation.** `1900D` carries an approach row `A_40D` with D = 0.416345, R = 0.140491. That
is byte-identical to `717200`'s `A_40D` row. At MLW it would give 0.416·√14,940 ≈ 51 kt. The 1900D's own
DEFAULT approach uses flap `35-A` (D = 0.915858), so (reading) `A_40D` is an orphan copy.

**Assessment.** ANP is **the only open-download source with a flap-dependent approach-speed law (D) and a
gear-down drag/lift ratio (R)**, and it covers business jets and some turboprops and pistons. It is usable
for modelling now. Its publication status is unresolved (above). Its substitution table must not be cited
as aerodynamic similarity.

## 5. EUROCONTROL Aircraft Performance Database (APD)

| item | value |
|---|---|
| Publisher | EUROCONTROL (the pages say to send feedback to the EUROCONTROL Training Institute), https://contentzone.eurocontrol.int/aircraftperformance/details.aspx?ICAO=<TYPE> |
| Retrieved | 2026-09-23, 3 s between requests, browser User-Agent. **79 pages**: the 40 types with the most flights plus the other 39 jet designators that appear in the site's type list, to read their power plant. Also `_help.html`. Per-file sha256 in [`excerpts/eurocontrol_apd_fields_for_our_types.csv`](excerpts/eurocontrol_apd_fields_for_our_types.csv) |
| Terms | page footer: "All data presented is only indicative and should not be used operationally." / "Copyright permission must be sought from EUROCONTROL". The help page adds no terms |
| Parameters | take-off V2, distance and MTOW; climb, cruise and descent speeds and rates; **approach IAS, MCS and ROD; landing V_at (IAS), distance and APC**; wing span, length, height; **power plant** (text); "Performance Similarity" and "Recognition similarity" (ICAO code + note) |

**Coverage.** The type list on every page offers **398 designators** (the page says "398 aircraft have
been returned"). 129 of ours / 11,274 flights are among them. **Nine of the top 40 return the site's empty
"No" page** (C68A, C700, E545, E75S, S22T, GA6C, GL7T, SF50, PC24; 1,734 flights).

**"Performance Similarity", verbatim, for every page fetched.** Only five types have an entry:

| type | Performance Similarity (verbatim) |
|---|---|
| BCS3 | ICAO Code: BCS1 · Notes: Lower rates of climb due to higher MTOM |
| BCS1 | ICAO Code: BCS3 · Notes: Better rates of climb (lower MTOM) |
| CL30 | ICAO Code: CL35 · Notes: CL30 has a bit less range |
| CL35 | ICAO Code: CL30 · Notes: CL35 has a bit more range |
| A339 | ICAO Code: A338 · Notes: The A339 has less range but more pax for very similar perfromance [sic] · ICAO Code: A333 · Notes: The A339 offers greater aerodynamic efficiency |

The other 65 listed pages say "No data". **APD offers no usable substitution list for business jets,
turboprops or GA.** V_at for the 40 top types is in the top-40 table above and in `apd_vat_ias_kt`.

**Assessment.** Corroboration only, as `docs/reference_speeds/` already uses it. Printing its values needs
EUROCONTROL's permission.

## 6. ICAO Aircraft Engine Emissions Databank (EDB)

| item | value |
|---|---|
| Publisher | ICAO, hosted by EASA: "The information is provided by the engine manufacturers, who are solely responsible for its accuracy." https://www.easa.europa.eu/en/domains/environment/icao-aircraft-engine-emissions-databank |
| Version | **Issue 32, 20 March 2026** (Record of Changes). `edb-emissions-databank_v32__web_.xlsx` sha256 `57a9ff57…9302530`. 888 gaseous rows and 269 nvPM rows |
| Scope | "turbojet and turbofan engines with a static thrust greater than 26.7 kilonewtons". Rated output F₀₀: "The maximum thrust available for take-off under normal operating conditions at ISA sea level static conditions without the use of water injection…" in kN (introduction, rev. 2, 06/2023). A few older small engines are listed too (TFE731-2-2B, TFE731-3, JT15D) |
| Terms | none specific found. The EASA site notice ("Reproduction is authorised, provided the source is acknowledged…") applies. The introduction warns that its fuel flows "should not be used for comparing the fuel efficiency of different engines" |

**How engines were assigned to types.** First from the ANP substitution sheet "by aircraft configuration"
(`ENGINE_TYPE`, sourced from the EASA TCDSN). For types absent there, from the APD "Power plant" field
(CL35, C25M, C25C, B78X, A339, LJ40, T38, F5). Then matched to the EDB `Engine Identification`:
- **exact:** the designation equals an EDB designation after dropping a trailing "(…)" installation suffix
  and expanding "/" and "," lists;
- **family (reading, hand-checked):** only 5 cases. PW1521G-3/PW1524G-3 → PW1521G/PW1524G (BCS3);
  AS-907-3-1E → AS907-3-1E-A1/A2/A3 (E545); RB211-535E4(B)-37 → RB211-535E4(B) (B753, which also has exact
  PW2037/PW2040 matches); Trent 7000 → Trent7000-68…-72D (A339); GEnx-1B (B78X; rating unknown, no UID given).
  TFE731, JT15D and AE3007 prefix matches were **rejected**, because the EDB rows are different sub-models.

**Coverage.** Exact matches for **35 types / 4,516 flights** (17 airliners / 2,046 and 18 business jets /
2,470). Family matches for 3 more / 1,983. **48 jet or airliner types / 3,507 flights have no EDB engine.**
These are the small business-jet engines below 26.7 kN (FJ44, PW535/545, PW615/617, HF120, most TFE731
variants, PW305, CFE738) and the turboprop airliners. Rows:
[`excerpts/icao_edb_rows_for_our_engines.csv`](excerpts/icao_edb_rows_for_our_engines.csv).
The EDB also lists PW814GA, PW815GA and Passport20-19BB1A. No source opened here links them to a type
designator, so GA5C, GA6C and GL7T stay blank.

**Assessment.** **Usable now** for F₀₀ per engine where the engine is known. Installed thrust per type is
n_engines × F₀₀ (reading). Where a type has several engine options, the rating flown is not known from our data.

## 7. Other sources checked

| source | what it is | verdict |
|---|---|---|
| **FAA AEDT 4** (https://aedt.faa.gov/Purchase.aspx) | FAA environmental tool with its own fleet DB, ANP and BADA inside. "1 AEDT 4 Site License: US $1200". "You must request and acquire a BADA license in advance of purchasing AEDT." BADA inside a tool falls under the integrated-tool template, which says "The BADA 3 data shall not be provided to YOU directly" and repeats the no-type-names clause | not pursued. Costs money and inherits the BADA restrictions. The AEDT EULA was not read |
| **BlueSky "BS" performance model** (TU Delft, GitHub `TUDelft-CNS-ATM/bluesky`, MIT) | 16 XML aircraft files (I. Metz, 2015). Two of ours: D328 and SB20 (9 flights). Fields: OEW, MTOW, MLW, S, C_fe, C_L,max for take-off, cruise and landing | **not acceptable as a primary source.** The values cite SKYbrary, a coffee-table book (Kreuzer) and estimates ("estimate from Obert… and Raymer"). D328.xml gives MTOW 12,000 kg < MLW 12,250 kg; the TCDSN MTOW in the ANP sheet is 13,640–13,990 kg |
| **EASA / FAA type certificate data sheets** | certified MTOW, MLW and engines, but no aerodynamics | covered by the separate airframe-facts pack in this folder (`airframe_facts*`) |
| **FAA Aircraft Characteristics Database** (Oct 2024) | MTOW, MALW, approach speed, and the BADA-style model name discussed in §3 | already in use (`docs/reference_speeds/`) |
| Textbooks (Obert 2009; Raymer; Roskam; Gudmundsson) | C_L,max and drag data for some types | copyrighted books, not open data. They can be cited for individual values, one page at a time. Not surveyed |
| Wikipedia, SKYbrary type pages, blogs, manufacturer marketing | — | not acceptable as sources for parameter values (user rule). None used |

## 8. `coverage_matrix.csv` columns

| column | meaning |
|---|---|
| `typecode`, `status`, `train_flights` | from `missing_types.csv` |
| `openap_2_6_2` | `native` / `synonym:<X>` (aircraft `_synonym.csv`) / `none` |
| `ps_model` | `native` (row in `ps-aircraft-params-20250328.csv`) / `synonym:<X>` / `none` |
| `bada3` | `yes:3.8-model` / `yes:3.8-synonym:<reference model>` from the **2010** public lists / `unknown`. Never `no`: no current list is public |
| `bada4` | `unknown` for all rows (no public per-type list) |
| `anp` | `native:<ANP ID>` (reading of the ANP description) / `substitute:<ANP ID>` (EASA "by ICAO code" sheet) / `none` |
| `apd_performance_similarity` | verbatim (field separators flattened) / `No data` / `not in APD (page returns 'No')` / blank = not fetched |
| `emissions_databank_engine` | EDB UID(s) matched exactly / `family:<UIDs>` (reading) / blank |
| `notes` | flags: native = reading, missing D/R, family matches, data-quality notes |
| `segment` | airliner / business_jet / turboprop / piston_ga / rotorcraft / military (explicit lists above) |
| `openap_2_4` | same test on the 2.4 wheel (identical for all rows) |
| `anp_official_substitute` | the EASA substitution proxy even where the type is also native |
| `apd_listed`, `apd_vat_ias_kt` | in the APD type list; landing V_at from the fetched page |
| `engine_designations`, `engine_designation_source` | engine strings used for the EDB match and where they came from |
| `faa_acd_model_bada` | the FAA ACD `Model_BADA` string (BADA 3 SYNONYM.NEW naming; not coverage evidence) |

## 9. What could not be verified

1. **Current BADA 3 and BADA 4 coverage per type.** EUROCONTROL publishes only counts. The 2010 lists
   cannot say whether a post-2010 type is modelled now. Whether a university licence today gets the full
   BADA 4 dataset (the 2019 template: sample only; the 2025 template: silent) is also unknown.
2. **Whether the EASA ANP T&C bind the public legacy v2.3 tables.** No statement was found either way.
3. **ANP "native" is a reading** of the ANP descriptions. Examples: GV for GLF5 (G-V vs G550); CL601 for
   CL60 (Challenger 600/601 vs 604/605/650); LEAR35 is described as "Learjet 36"; BEC58P is the pressurised
   58P.
4. **Engines of GA5C, GA6C, GA7C, GA8C, GL7T, C68A, C700, E550, FA8X, SF50, PC24 and E35L.** They have no
   row in the ANP configuration sheet and no APD page with a power plant, so no EDB mapping was made.
5. **APD pages** were fetched for 79 types only (the top 40 plus jets). The other 120 types have a blank
   `apd_performance_similarity`.
6. **PS parameter quality** for our types was not checked against manufacturer data, except the B722 engine
   count noted above.
7. **The AEDT EULA and the content of the AEDT fleet database** were not read (licensed download).
8. The pyBADA `SYNONYM.NEW` is a **demo file** ("BADA Release: 3.x demo", Nov 24 2020). It is used only to
   show the naming format behind the FAA column, never as coverage.

## 10. Files

Raw downloads, all under `data/aircraft_performance/` (git-ignored), retrieved 2026-09-23. HTML pages are
live pages; their sha256 changes whenever the site changes.

| path | sha256 | source URL |
|---|---|---|
| `openap/openap-2.6.2-py3-none-any.whl` | `c4a793251231fb9a10ecd3b37266e1f564667949b0eff2d42346fb8bf508a295` | files.pythonhosted.org (PyPI) |
| `openap/openap-2.6.2.tar.gz` | `9c1b370d12f02aeacf2a8112591ac0cb2a733ad7044f4b7bcf251b9bb1121d9d` | PyPI |
| `openap/openap-2.4-py3-none-any.whl` | `33c8d1366c7b1fd9b6d13aff8c21f81b1f780f7f5917b2026f2d6b4e8b0bf260` | PyPI |
| `openap/gh_master_aircraft_synonym.csv` | `d223e24368f64ed671bcf4c4e14df0ea5630b105bdcde28511c96baa13bc1c3e` | raw.githubusercontent.com/junzis/openap/master |
| `pycontrails/pycontrails-0.63.5-cp312-…manylinux_2_28_x86_64.whl` | `b4daa0db3093bef45d1439fed7f42e56274256439cb878d373a17e5e8acfdc66` | PyPI |
| `pycontrails/papers/Poll_Schumann_2021_Part1_aer.2020.62_DLR-elib.pdf` | `952d258fe83ecb4f27e49397352b9a02abbaf8fbe73df92becc3d173307ccea3` | elib.dlr.de/135592 |
| `pycontrails/papers/Poll_Schumann_2021_Part2_aer.2020.124_DLR-elib.pdf` | `ad343f5eb95f52fb2e2b2a3b5ca004c46d98d4fdfd360ace08d7772cea7cc8c7` | elib.dlr.de/139037 |
| `pycontrails/papers/Poll_Schumann_2025_Part3_aer.2024.141_DLR-elib.pdf` | `ee868646aebd9279ef2582df436f6b439031dc289bb61aac93d99bc314fc2f09` | elib.dlr.de/212199 |
| `pycontrails/papers/Poll_Schumann_2024_turbofan_engine_all_phases_aer.2024.92_DLR-elib.pdf` | `e7f7b5ed22633f7579ef59d066152aa50d8895bcc902d01a301ac22f679ffefe` | elib.dlr.de/206626 |
| `pycontrails/pycontrails_main.bib` | `8e9459c0efae5caae615f6fe37863a14fb74919282acab4a7e54cc6d729ef77c` | GitHub main `docs/_static/pycontrails.bib` |
| `bada/bada-standard-licence-information.pdf` | `60a2eb5ff254ac106b738d63602cb8cb9bd885e9066859ee1d82ced17a81abb6` | eurocontrol.int/sites/default/files/2020-06/ |
| `bada/bada-licence-information-family-4.pdf` | `523e4f2b474d7840fd1729ba572779ffe4b610a4b07eb1602c984d768f75b04b` | eurocontrol.int/sites/default/files/2025-07/ |
| `bada/bada-integrated-standard-licence-for-information.pdf` | `63b5ac4094d1bde6f745303fdc72990593c0b068b65d0f8d2c4f892f43497d9c` | eurocontrol.int/sites/default/files/2020-06/ |
| `bada/eurocontrol-showcase-summit-bada-latest-evolutions.pdf` | `9ec0b5e6dbbbdc5a66b657bafb033a3f6fb00e8a1314ab14dc3580bf41b8db1c` | eurocontrol.int/sites/default/files/2025-11/ |
| `bada/BADA_3.8_Coverage_2009_traffic.pdf` (EEC TR 2010/008) | `d5f2fad85645b94f33806e8da67692a13c2282b26d05f54544226f54512faf79` | eurocontrol.int/archive_download/all/node/9692 |
| `bada/BADA_3.8_Synonym_Aircraft_Report_EEC-TR-2010-007.pdf` | `0e1d6687d87c3c1ebbb88a7219c14108ec368c80f85613a12d05ef6a6f20f988` | …/node/9689 |
| `bada/BADA_3.7_Synonym_Aircraft_Report_EEC-TR-2009-007.pdf` | `83f79b19047c3c32dd1f1902291b98c825ab0b8163d0170da0c9655a92035117` | …/node/9706 |
| `bada/BADA_3.8_User_Manual_EEC-TR-2010-003.pdf` | `5b79ed340508a42f55655cbf83302dde8e549831518e3082768372d45fcd5e95` | …/node/9690 |
| `bada/EIH-Technical-Report-121122-58-v1.4_BADA4_spec.pdf` (BADA 4 UM v1.4) | `866e14989f6e61663317f06c9e997eedbab34aa17d21974d9fe5a22bdc1b5883` | GitHub eurocontrol-bada/model-specifications |
| `bada/pybada_github/SYNONYM.NEW` (demo) | `828ab17948865661c13688dbfa4955fdd8dcaf85159bbc16757814f5559c0d9c` | GitHub eurocontrol-bada/pybada `src/aircraft/BADA3/DUMMY/` |
| `bada/eurocontrol_model_bada.html`, `bada/news_bada_just_got_better.html` | live pages | eurocontrol.int/model/bada; /news/bada-atms-most-comprehensive-… |
| `anp/archive_anp_v2.3.zip` (unpacked to `anp/v2.3/`) | `6147f05ea50f57df816ad2bbf8a1e870bcc74f253fb32bfeea5f468ea85d852f` | easa.europa.eu/en/downloads/138164/en |
| `anp/anp_database_v2.3.pdf` (release note) | `13a362b3746505a764029a258c9c34cc206d7f097dfff04a8a26512ae1c833d9` | …/downloads/138159/en |
| `anp/anp_aircraft_substitutions_-_jets_heavy_props_22022018_.xlsx` | `2691383eecae1bafb9995e315da1a817dc57c3f0390251ed403396784cd54ac2` | …/downloads/138165/en |
| `anp/easa_verified_anp_aircraft_types.xlsx` | `7deb81b6bc2957ab97bdfa6bafd027fdf926a215ed486813fe47a0dda7209821` | …/downloads/143284/en |
| `anp/EASA_ANP_Database_Terms_and_Conditions_Issue_3.pdf` | `1eebbd843c81b7da57d15326be3fd626a5ff99575150acec4541a31b2a387923` | …/downloads/116585/en |
| `anp/List_of_ANP_Data_Providers.pdf` | `c1e1a55a28f8a3bb74753cbe150a0bc1df4e4a137f0a435c428195d3978e1af7` | …/downloads/137505/en |
| `anp/ECAC-Doc_29_4th_edition_Dec_2016_Volume_2.pdf` | `15c23476fb6250cbd7dfc3bd8875accd90d7bea31f84b19122fc97686b6e6029` | ecac-ceac.org/images/documents/ |
| `anp/easa_anp_data_page.html`, `anp/easa_anp_data_request_form.html`, `anp/easa_copyright_disclaimer.html` | live pages | easa.europa.eu |
| `eurocontrol_apd/<TYPE>.html` ×79, `_help.html` | per file in the APD excerpt | contentzone.eurocontrol.int/aircraftperformance/ |
| `icao_emissions/edb-emissions-databank_v32__web_.xlsx` | `57a9ff572458ad3a3141afc1aea932b5faa5796d279f0ac74600b27869302530` | easa.europa.eu/en/downloads/131424/en (serves the current issue) |
| `icao_emissions/edb-introduction_text_-rev2.pdf` | `227f16669a59c4b2bbd3160ece94b6ccf54a3e664293e51dd3da034ee27a39bb` | …/downloads/45576/en |
| `other/aedt_purchase.html` | live page | aedt.faa.gov/Purchase.aspx |
| `other/bluesky_BS/D328.xml`, `SB20.xml` | `29a21dbe…1c45ddea`, `18c0f1cb…9ab6cd466d` | GitHub TUDelft-CNS-ATM/bluesky master |

Tracked in this folder: `README.md`, `download.sh`, `coverage_matrix.csv`, and `excerpts/`:
`openap_2.6.2_aircraft_and_synonyms.txt`, `ps_params_rows_for_our_types.csv`,
`ps_synonym_rows_for_our_types.csv`, `bada_3.8_coverage_rows_for_our_types.csv`,
`anp_substitution_rows_for_our_types.tsv`, `anp_2.3_aircraft_and_aero_rows_for_referenced_ids.csv`,
`eurocontrol_apd_fields_for_our_types.csv`, `eurocontrol_apd_index_398_designators.txt`,
`icao_edb_rows_for_our_engines.csv`, `licence_and_scope_clauses.txt`.
