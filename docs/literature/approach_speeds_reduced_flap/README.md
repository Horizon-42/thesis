# Reduced-flap approach speeds for E75L, B737, A319, E170, E190: a primary-source search

Retrieved and read 2026-09-25. Background: `docs/code-health-followups.md` entry 21 ("Single-valued
FAA approach-speed rows for multi-flap types"), `aircraft/reference_speeds.json`,
`docs/reference_speeds/README.md`. Files: `papers/` (re-fetch with `bash download.sh`; PDFs are
git-ignored).

## Status

| item | state |
|---|---|
| Reduced-flap V_REF at MLW, published in a primary document, for any of the five types | **not found** (all five) |
| Full-flap / single value at MLW from the manufacturer | A319: found (126 kt, CONF FULL). B737: found, but no flap setting stated (132 kt). E75L, E170, E190: not found |
| Does the FAA single value match the manufacturer's full-flap value? | A319: same speed, different weight. B737: no (130 vs 132 kt, and a different MLW). E-Jets: cannot be checked |
| Side finding: the premise that the FAA dual rows (B738 140/144 …) are flap pairs | **doubtful**, see §4 (a reading from the FAA's own data dictionary + the 737 FSB report) |
| Data change made | none (this folder only) |

Units: speeds in the documents' knots; 1 kt = 1852 m / 3600 s = 0.514444 m/s. Masses: 1 lb =
0.45359237 kg.

## 1. Answer at a glance

| type (FAA ACD row) | reduced-flap value | full-flap value | weight | source + location | confidence |
|---|---|---|---|---|---|
| **B737** (737-700; FAA 130 kt, MALW 145,600 lb) | **not found** (flaps 30) | 132 kt = 67.91 m/s, flap **not stated**; the FAA 737 FSB report says the category is set at flaps 40 | MLW 129,200 lb = 58,604 kg | Boeing, *FAA Reference Code and Approach Speeds for Boeing Aircraft*, 30 Mar 2016, p. 1, row 737-700/C | medium (flap setting inferred, not stated) |
| **A319** (FAA 126 kt, MALW 61,000 kg) | **not found** (CONF 3) | 126 kt = 64.82 m/s, CONF FULL (the certificated maximum flap setting) | MLW 62,500 kg | Airbus, *A319 Aircraft Characteristics – Airport and Maintenance Planning*, Jun 01/24, §3-5-0 (PDF p. 164) | high (full-flap value only) |
| **E75L** (E175 long wing; FAA 126 kt, MALW 34,000 kg) | **not found** (flaps 5) | **not found** | – | – | – |
| **E170** (FAA 124 kt, MALW 33,300 kg) | **not found** at MLW (nearest: flaps 5, 130 kt at 71,000 lb, an operator example) | **not found** at MLW (nearest: flaps FULL, 115 kt at 63,000 lb, same example) | – | NTSB docket DCA07MA072, Ops Att. 5 (Shuttle America ERJ-170 POH, Rev. 2, 15 Aug 2006, pp. 4-111/4-112) | low; not at MLW, not usable |
| **E190** (FAA 124 kt, MALW 43,000 kg) | **not found** (flaps 5) | **not found** | – | – | – |

No document gives a reduced/full pair at MLW for any of the five types, so no row can be added to
`reference_speeds.json` from this search. No value was estimated or scaled.

## 2. What each source means by "approach speed"

- **FAA InFO 23001** (9 Jan 2023), p. 1: the aircraft approach category is set by V_ref (or
  1.3 V_so) "both at the maximum certificated landing weight". That is the basis of every FAA
  ACD row.
- **Airbus AC §3-5-0**: final approach speed = indicated airspeed at the threshold in the landing
  configuration, "at the certificated maximum flap setting and Maximum Landing Weight (MLW)",
  standard atmosphere. So the Airbus AC only ever publishes the CONF FULL value.
- **Boeing ARC/approach-speed table**: a single "Approach Speed (knots)" column next to MLW; no
  definition and no flap setting on the page.
- **FAA FSB 737, Rev 17**, §13.3 (p. 17 of 67): the 737NG / MAX category "may be done using the
  certificated maximum flap setting of FLAPS 40" and the airplane's AFM MLW; §13.4: normal landing
  flaps are 15, 30 and 40.
- **Operator manuals in NTSB dockets**: V_REF = target speed at 50 ft over the threshold
  (Shuttle America ERJ-170 POH: 1.23 V_S), read from a weight-indexed speed card.

## 3. Per type

### 3.1 B737 (Boeing 737-700)

Found:
- Boeing, *FAA Reference Code and Approach Speeds for Boeing Aircraft*, 30 March 2016 (PDF made
  2017-02-08), p. 1: **737-700/C and 737-700W: 132 kt at MLW 129,200 lb (58,604 kg)**, reference
  code C-III. One value; no flap setting. (Same page: 737 BBJ 135 kt at 134,000 lb; 737-800 142 kt
  at 146,300 lb.)
- FAA FSB 737 Rev 17 (11/16/2020) §13.3: 737-600/700 category C; category determined at FLAPS 40
  and the AFM MLW. So the single Boeing value is most likely the flaps-40 V_REF (my reading).
- NTSB DCA13FA131 (Southwest 345, 737-7H4 N753SW), Operations Group Chairman's Factual Report,
  23 Jul 2014, §9.1 (report p. 18): landing flaps 40, **V_REF 128 kt (65.85 m/s)** at an estimated
  landing weight of 127,100 lb (57,652 kg); that airplane's MLW 128,000 lb. Flaps 40 only.
- NTSB DCA19IA036 (Southwest 278, 737-7H4 N752SW), Aircraft Performance Study, 29 Oct 2020, p. 9:
  flaps-40 V_REF 126 kt at 120,600 lb. Flaps 40 only, not at MLW.

Not found: any flaps-30 V_REF for the 737-700, at any weight, in an official document. The Boeing
ACAP (D6-58325-7 Rev C, already in `data/reference_speeds/boeing/`) has no approach speed.

FAA row check: the FAA's 130 kt matches neither Boeing's 132 kt (at 129,200 lb) nor the NTSB 128 kt,
and the FAA MALW (145,600 lb) matches no 737-700 MLW in these documents (the Boeing table gives
129,200 lb for the 737-700/C and 134,000 lb for the 737-700-based BBJ; 146,300 lb is the 737-800 and
BBJ2 figure). Not a match.

### 3.2 A319

Found:
- Airbus, *A319 Aircraft Characteristics – Airport and Maintenance Planning*, issue Jun 01/24,
  §3-5-0, Final Approach Speed (PDF p. 164), on A/C A319-100 only: **126 kt (64.82 m/s) at MLW
  62,500 kg (137,789 lb)**, category C, at the certificated maximum flap setting (CONF FULL). No
  CONF 3 value and no A319neo value.
- FAA FSB A318/A319/A320/A321 (Rev 7 **draft**) §13.2: A319 category C; no speeds.

Not found: V_LS CONF 3 (the Airbus reduced-flap landing speed) for the A319 at MLW. The only
official copies of a CONF 3 table found are **A320** tables (see §5.1); they are not A319 data.

FAA row check: the FAA's 126 kt equals Airbus's CONF FULL value, but the FAA pairs it with MALW
61,000 kg, while Airbus quotes it at 62,500 kg (both are A319 MLW options in the same AC).

### 3.3 E75L (E175), E170, E190

Found (category and flap policy only):
- EASA Operational Evaluation Board report, Embraer 170/175/190/195, Rev 5, 01 Jul 2013, p. 18:
  §4.2 normal final landing flap setting is FLAPS FULL; §4.9 all variants category C. No speeds.
- Transport Canada Operational Evaluation Report, Embraer E-Jets, Rev 1, 2023-11-03, p. 40:
  §13.4 category C, circling at flaps 5 or FULL and V_REF + 5 kt; §13.5 the "normal final landing
  flap setting is flaps 5º and Flaps FULL". No speeds.
- NTSB DCA07MA072 (Shuttle America 6448, ERJ-170 N862RW, 2007): Operations Group factual, §3.0.1
  (PDF p. 14): flaps 5, V_REF 129 kt, V_APP 139 kt, estimated landing weight 69,186 lb (31,382 kg),
  MLW 72,310 lb (32,799 kg). Ops Attachment 5 (company POH, Rev. 2, 15 Aug 2006; PDF pp. 4–5): worked
  examples read from the company speed card, flaps 5 V_REF 130 kt at 71,000 lb (32,205 kg) and
  flaps FULL V_REF 115 kt at 63,000 lb (28,576 kg). These are operator examples at weights below
  MLW; they are neither at MLW nor a pair at one weight, so they cannot fill the row.

Not found: any flaps-5 or flaps-FULL V_REF at MLW for the E170, E175 or E190 in an official
document. Embraer's airport planning manuals (E175 APM-2259 as filed in NTSB DCA20IA014; E190
APM-1901 Rev 24, 29 Nov 2024) and the E175 specification sheet carry no approach speed; the EASA
TCDS IM.A.001 (ERJ-170, Issue 13) has none.

FAA row check: not possible; no manufacturer full-flap value for these types was found.

## 4. Side finding: the FAA dual rows are probably not flap pairs

Follow-up 21 says the dual rows (B738 140/144, CRJ9 132/141) "carry the reduced-flap speed too".
Four primary texts argue otherwise (the texts are verified; the conclusion is my reading):

1. FAA ACD `Data_Dictionary`, note 2 (already in `docs/reference_speeds/excerpts/faa_acd_2024-10_definitions.txt`):
   where the FSB gives two categories but only one speed, the FAA **calculates** the missing speed as
   the value in the other category closest to the published one. 14 CFR 97.3 (eCFR, in force
   2026-09-01) bounds category C at 121 kt or more but less than 141 kt, so its top whole-knot value
   is 140 kt; every Boeing dual row in `reference_speeds.json` has a minimum of exactly 140 kt (B37M, B38M,
   B39M, B738, B739, B788, B789). Those 140 kt minima look like this category-boundary filler, not a
   published full-flap speed.
2. FAA FSB 737 Rev 17 §13.3, note: the 737-800/900/900ER C-or-D split is put down to the many
   maximum-landing-weight options, with the category evaluated at flaps 40: a weight split, not a
   flap split.
3. Airbus AC A321 (Dec 01/23, §3-5-0) and FSB A320 draft §13.2: the A321's two values (140 kt at
   75,500 kg, 142 kt at 77,800 kg) are two MLW options at maximum flap; the JSON already records this
   ("the dual value is by landing weight, not flap setting"), and the CRJ9 note says the same
   (141 kt at the Long Range MALW).

So the "reduced-flap upper edge" that the single-value rows are said to lack may not exist on the
dual rows either. The B738/B739/B38M/B39M/B788/B789 notes' "flap reading" rests on the data
dictionary's row wording (rows 15–16) and should be re-checked against note 2 before any window is
built on it.

## 5. Leads (recorded, not used as sources)

### 5.1 Official, but not what was asked
- **A320 CONF 3 tables** (not A319): NTSB DCA11IA040 (United A320 N409UA) Ops Att. 7, United
  A319/A320 Flight Manual p. 8.20.15 (17 Nov 06, page marked A320; PDF p. 12): landing reference
  speeds by gross weight, 100,000–170,000 lb, columns Green Dot, S, F, V_LS Conf 3, and
  "VREF (VLS Conf Full)". NTSB DCA12IA096 (JetBlue A320) Ops Att. 2, A320 QRH p. 13-18 (Rev 05,
  12/01/10; PDF p. 5): V_LS CONF FULL and V_LS CONF 3 by weight, 96,000–172,000 lb. An A319 page of
  either manual would answer the A319 question; none was found in a docket.
- **ICAO/ECAC ANP noise-model coefficients** (`ANP_archive_v2.3.zip`, EASA-hosted legacy
  ANP 2.3; current ANP data is by request only). `ANP2.3_Aerodynamic_coefficients.csv` gives a
  landing coefficient D per landing flap setting, and ECAC Doc 29 4th ed. Vol. 2, §B11 eq. (B-24),
  p. B-16, defines the "landing approach calibrated airspeed" as V_CA = D·√W (W in lbf), flown into
  an 8 kt reference headwind. Coefficients (kt/√lbf) and ANP maximum landing weights
  (`ANP2.3_Aircraft.csv`):

  | ANP id | D per landing flap | ANP max landing weight |
  |---|---|---|
  | 737700 (CFM56-7B24) | A_15 0.412200, A_30 0.398600, A_40 0.390700 | 129,200 lb |
  | A319-131 (V2522-A5) | 3_D 0.379931, FULL_D 0.355927 | 137,789 lb |
  | EMB170 | FULL 0.498900 only | 72,312 lb |
  | EMB175 | FULL 0.498200 only | 74,957 lb |
  | EMB190 | FULL 0.434400 only | 97,003 lb |

  This is an approach speed for noise modelling, not a V_REF, and it is defined by a formula, not
  tabulated; it is not evaluated here. It has no flaps-5 coefficient for the E-Jets.

### 5.2 Unofficial (not opened as sources; not acceptable)
- Scribd "Boeing 737-700 Performance Overview" (`scribd.com/document/470653342`) and a weebly copy
  of "Boeing 737-700 Performance and Selected Limitations CFM56-7B20" — appear to be copies of
  Boeing performance pages with a V_REF 40/30/15 table.
- Scribd/idoc.pub copies of an "Airbus A319-A320-A321 Quick Reference Handbook".
- A copy of an Embraer E170 AOM (Rev 21, 20 Oct 2016) on `caisatech.net` (host did not resolve,
  2026-09-25); Mesa Airlines E-175 study guide (employee portal); `flite.ch` E190 summary; forum
  and simulator pages (PPRuNe, airlinepilotforums, flyawaysimulation, Infinite Flight).

## 6. Documents in `papers/`

All retrieved 2026-09-25 by `download.sh`. "cited for" says whether a value or an absence was taken
from it.

| file | document | edition | cited for | URL | sha256 |
|---|---|---|---|---|---|
| `eCFR_14CFR_97.3_2026-09-01.xml` | 14 CFR 97.3, definitions (aircraft approach category) | eCFR, as in force 2026-09-01 | category bounds (§4) | https://www.ecfr.gov/api/versioner/v1/full/2026-09-01/title-14.xml?part=97&section=97.3 | `2e3c880a…758d1876` |
| `FAA_InFO23001_2023-01-09.pdf` | FAA InFO 23001, Use of Aircraft Approach Category During Instrument Approach Operations | 01/09/23 | definition, p. 1 | https://www.faa.gov/sites/faa.gov/files/InFO23001.pdf | `01d3ea2e…4b1db3b9` |
| `Boeing_FAA_Reference_Code_and_Approach_Speeds_2016-03-30.pdf` | Boeing, FAA Reference Code and Approach Speeds for Boeing Aircraft | 30 March 2016 | B737 value, p. 1 | https://www.boeing.com/content/dam/boeing/v2/airports/faq/arcandapproachspeeds.pdf | `396329a9…6994d967` |
| `FAA_FSB_Boeing737_Rev17_2020-11-16.pdf` | FAA Flight Standardization Board Report, The Boeing Company 737 | Revision 17, 11/16/2020 | §13.3–13.4, p. 17 | https://www.faa.gov/sites/faa.gov/files/2022-08/737_FSB_Report.pdf | `657d8269…65596aa6` |
| `NTSB_DCA13FA131_Operations_Group_Chairman_Factual.pdf` | NTSB DCA13FA131, Operational Factors Group Chairman's Factual Report | 23 Jul 2014 | B737 flaps 40, §9.1 p. 18 | data.ntsb.gov docBLOB ID 40418230 (full URL in `download.sh`) | `d87cbcb2…a3f27c18` |
| `NTSB_DCA19IA036_Aircraft_Performance_Study.pdf` | NTSB DCA19IA036, Aircraft Performance Study | 29 Oct 2020 | B737 flaps 40, p. 9 | docBLOB ID 11940941 | `8ff62dc0…fd00d538` |
| `Airbus_AC_A319_0624.pdf` | Airbus, A319 Aircraft Characteristics – Airport and Maintenance Planning | Jun 01/24 | A319 value, §3-5-0, PDF p. 164 | https://www.aircraft.airbus.com/sites/g/files/jlcbta126/files/2024-06/AC_A319_0624.pdf | `c59ecba5…3b341e9c` |
| `FAA_FSB_A320_family_Rev7_Draft.pdf` | FAA FSB Report, Airbus A318/A319/A320/A321 | Revision 7, **draft** (undated) | §13.2, PDF p. 16 | https://www.faa.gov/aircraft/draft_docs/fsb/FSBR_A320_Rev_7_Draft.pdf | `6b2f59de…171aca94` |
| `EASA_TCDS_A064_A318-A321_Iss62_2026-06-26.pdf` | EASA TCDS EASA.A.064, Airbus A318/A319/A320/A321 | Issue 62, 26 June 2026 | absence (no V_REF) | https://www.easa.europa.eu/en/downloads/16507/en | `957199ee…b538182f` |
| `NTSB_DCA11IA040_Ops2_Att7_UAL_A319-A320_Landing_Performance.pdf` | NTSB DCA11IA040, Ops Att. 7, United A319/A320 Flight Manual excerpts | pages dated 2006–2009 | A320 lead, PDF p. 12 | docBLOB ID 40355135 | `376c7534…49fd77c5` |
| `NTSB_DCA12IA096_Ops2_Att2_JetBlue_A320_QRH_Landing_Performance.pdf` | NTSB DCA12IA096, Ops Att. 2, JetBlue A320 QRH excerpts | Rev 05/07 | A320 lead, PDF p. 5 | docBLOB ID 40384278 | `bcd863b4…8229763f` |
| `Embraer_APM-1901_E190_Rev24_2024-11-29.pdf` | Embraer 190 Airport Planning Manual APM-1901 | Revision 24, 29 Nov 2024 | absence | https://embraer.com/media/jwsj1u1u/e190apm_apm_190-1.pdf | `f6b89339…387b76a0` |
| `Embraer_E175_APM_NTSB_DCA20IA014_Att2.pdf` | Embraer 175 Airport Planning Manual APM-2259 (NTSB DCA20IA014 Att. 2) | pages to May 25/18 | absence | docBLOB ID 13691419 | `e1df5f7d…6daf2c8c` |
| `Embraer_E175_spec.pdf` | Embraer E175 specification sheet | undated | absence | https://www.embraer.com/media/o3sjzbwl/e175_spec.pdf | `94530879…047617b5` |
| `EASA_OEB_Embraer170-195_Rev5_2013-07-01.pdf` | EASA OEB Report, Embraer 170/175/190/195 – Flight Crew Qualifications | Revision 5, 01 July 2013 | §4.2, §4.9, p. 18 | https://www.easa.europa.eu/sites/default/files/dfu/20130701%20Embraer%20ERJ%20170-175-190-195%20EASA%20OEB%20Report%20Rev%205.pdf | `242c18f0…0bd6f1ed` |
| `TCCA_OE_Report_Embraer_E-Jets_Rev1_2023-11-03.pdf` | Transport Canada Operational Evaluation Report, Embraer E-Jets | Revision 1, 2023-11-03 | §13.4–13.5, p. 40 | https://tc.canada.ca/sites/default/files/2023-12/OE_EMBRAER_E170_E-JETS_OE_REPORT_REVISION_1.pdf | `5b948260…e459a8ea` |
| `EASA_TCDS_IM_A_001_ERJ170_Iss13_2022-01-31.pdf` | EASA TCDS EASA.IM.A.001, Embraer ERJ-170 | Issue 13, 31 Jan 2022 | absence | https://www.easa.europa.eu/en/downloads/7525/en | `366e7e1e…8ae610f6` |
| `NTSB_AAR-08-01_Shuttle_America_6448_ERJ170.pdf` | NTSB AAR-08/01, Shuttle America 6448, ERJ-170, Cleveland | 2008 | absence (no V_REF stated) | https://www.ntsb.gov/investigations/AccidentReports/Reports/AAR0801.pdf | `064018df…b928d626` |
| `NTSB_DCA07MA072_Ops2_Factual.pdf` | NTSB DCA07MA072, Operations Group Chairman's Factual Report | 29 Jun 2007 | E170 flaps 5 at 69,186 lb, PDF p. 14 | docBLOB ID 40279863 | `b6e3178f…d291161c` |
| `NTSB_DCA07MA072_Ops2_Att5_ERJ170_POH_Normal_Procedures.pdf` | NTSB DCA07MA072, Ops Att. 5, Shuttle America ERJ-170 POH ch. 4 | Rev. 2, 15 Aug 2006 | E170 examples, PDF pp. 4–5 | docBLOB ID 40279825 | `e5fb6e41…025ca320` |
| `ANP_archive_v2.3.zip` | ANP database, legacy version 2.3 archive tables | v2.3 | lead (§5.1) | https://www.easa.europa.eu/en/downloads/138164/en | `6147f05e…a85d852f` |
| `ECAC_Doc29_4th_Ed_Vol2_Dec2016.pdf` | ECAC.CEAC Doc 29, 4th Edition, Volume 2 | Dec 2016 | lead, §B11 p. B-16 | https://www.ecac-ceac.org/images/documents/ECAC-Doc_29_4th_edition_Dec_2016_Volume_2.pdf | `15c23476…6b6e6029` |

`download.sh` prints the full sha256 of every file. The two EASA TCDS links serve the current
issue, so a later run may fetch a newer issue with a different hash.

## 7. Searched and found nothing usable

- Boeing ACAPs 737NG Rev C, 737 Classic Rev E, 737 MAX Rev K (`data/reference_speeds/boeing/`, text
  searched for "approach speed" / "VREF"): none. FAA FSB 737 Rev 21 draft: the same §13.3 category table and flaps-40 note as Rev 17.
- NTSB DCA06MA009 (Southwest 1248, 737-7H4): FOM/FRM excerpt has flaps 15/30/40 landing-distance
  tables, no V_REF table. NTSB DCA18MA142 (Southwest 1380): Ops attachments 1–7, no V_REF table.
- NTSB DCA19LA134 Att. 11 (American A319/320/321 QRH excerpt) and DCA20CA058 Att. 9 (United
  A319/320 Flight Manual excerpt): no weight-indexed V_LS table in their text.
- NTSB DCA20CA043 Att. 7 (American E190 OM Vol. 1 excerpt): refers to the "speed flip card or QRH",
  no table. NTSB DCA19CA081 Att. 9 (ERJ 170 AOM Vol. 1 excerpt): no V_REF in its text. NTSB DCA07MA072 Ops
  Att. 9 (ERJ-170 POH performance): landing-distance tables only. NTSB DCA20IA014 docket: takeoff
  pitch-trim event, no landing speeds.
- Embraer E170 APM (APM-1346): the Embraer links returned HTTP 404 (embraercommercialaviation.com)
  and 403 (flyembraer.com) on 2026-09-25; not examined.
- FAA FSB ERJ-170/190: `fsims.faa.gov` did not resolve and the regulations.gov copy returned HTML;
  not examined (EASA OEB and TCCA OE cover the same ground).
- Not checked: FAA TCDS A16WE (737), EASA TCDS for the 737 and the ERJ-190 (IM.A.071).
