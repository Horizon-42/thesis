# FAA certificated model → ICAO Doc 8643 designator: evidence (2026-09-23)

Research chore for the airframe-resolution gap, written before any code changed. **Applied the same
day** in `aircraft/faa_icao_crosswalk.json` (read by `aircraft/build_aircraft_identity_database.py`),
with these coordinator decisions:

- **A** rows applied as written, each row listing every registry spelling of the same certificated
  model (TC holder history: CESSNA / CESSNA AIRCRAFT CO / TEXTRON AVIATION INC, BEECH / RAYTHEON /
  HAWKER BEECHCRAFT / BEECHCRAFT CORP / TEXTRON, CANADAIR / BOMBARDIER …), so no other spelling of a
  documented model falls through to the name matcher or OpenSky.
- **F**: B300C is an `unresolved` row: its TCDS line names both MC-12W (Doc 8643 MC-12 → B350) and
  UC-12W (Doc 8643 UC-12 Huron → BE20), so no FAA document ties the civil model to one record; the row
  also stops OpenSky's BE30 (the Model 300). 208B, DA 50 C, FALCON 2000EX and FALCON 900EX get no row:
  they are distinct TC model designations that Doc 8643 does not name, so the lower-authority sources
  still decide them.
- **S**: all four split by serial number, the Cessna Model 525 by TCDS A1WI p1 (serial groups by
  approval date) + FSBR CE-525 p96 (the name of each group): C525 for 525-0001..0558, C25M for
  525-0685 and 525-0800 on (the M2 Gen2 has no serial group of its own); the CJ1+ (525-0600..0701),
  a separately approved TCDS group whose FSB name Doc 8643 does not carry, stays unresolved.
- **N**: ATR 72-212A unresolved, under both registry spellings.
- The incidental findings below (Bell 525 → C525, Cessna 525 → C25A, TBM majority vote) are fixed by
  the rows.

Before any code changed: Decision rule (user, 宁缺毋滥): a mapping is accepted only when
(1) the Doc 8643 snapshot has a record naming the model, (2) an FAA document ties the FAA certificated
model designation to that model, and (3) the FAA model maps to exactly one designator. Otherwise it is
AMBIGUOUS (with a serial split, if an FAA document states one) or NONE.

## Status

| Item | State |
|---|---|
| Gap measured | 2,737 flights have `speed_reason` "airframe could not be resolved…"; 703 have an icao24 absent from the FAA registry; **2,034** (73 FAA models) are in the registry but their model has no `typecode`. Reproduces the coordinator's list exactly. |
| Covered here | 38 FAA (manufacturer, model) strings = **1,927 of 2,034 flights** (every model with ≥ 9 flights, plus 7 smaller strings whose evidence was already downloaded). |
| A: accept, strict | 26 strings, **1,481 flights** (table below). |
| F: accept only at family level (coordinator decides) | 6 strings, **90 flights**: 208B, DA 50 C, FALCON 2000EX ×2, FALCON 900EX, B300C. |
| S: ambiguous, FAA serial split exists | 4 strings, **303 flights**: BD-100-1A10, FALCON 7X, TBM 700, TEXTRON 525. |
| N: ambiguous, no serial split | 1 string, **53 flights**: ATR 72-212A. |
| Not researched | 35 strings with ≤ 8 flights, 107 flights (list at the end). |
| Sources | 24 current FAA TCDS + 6 FAA FSB reports (drs.faa.gov), FAA Order JO 7360.1K + its only notice GENOT N JO 7360.7, FAA Aircraft Characteristics Database (supplementary). Doc 8643 = repo snapshot `aircraft/icao_doc8643.json` (10 July 2026). |
| Excerpts | `docs/aircraft_identity/excerpts/*.txt` (34 files, verbatim, PDF page numbers). |
| Downloads | `data/AIRCRAFT/sources/faa/` (+ `tcds/`, `fsbr/`), each directory has a `manifest.json` with URL, revision, size, sha256. 23 MB. `data/` is git-ignored, so the downloads are local only; the report and excerpts are new, uncommitted files. |

## What each FAA document contributes

- **TCDS** (`faa_tcds_<number>.txt`): the certificated model designation, the TC holder and holder history
  (for example "Cessna Aircraft Company transferred to Textron Aviation Inc. on July 29, 2015"), and, where
  it gives them, marketing designations with eligible serial numbers.
- **FSB report** (`faa_fsbr_<id>.txt`): its cover table maps "TCDS / TCDS Identifier (model) / Marketing Name
  / Pilot Type Rating". Used where the TCDS does not print the marketing name (Model 700 → Longitude,
  Model 408 → SkyCourier) and for the 525 serial table. The FSB report is an FAA document but is not one
  of the two types the rule names; this is flagged per row.
- **JO 7360.1K** (`faa_jo7360_1k.txt`): the FAA's own model → designator table. Appendix C is cited (one
  row per model: MODEL | MANUFACTURER | DESIGNATOR). Its model names are the Doc 8643 names. Appendix D
  decodes manufacturer codes: `CESSNA` = Cessna Aircraft Company, `TEXTRON` = Textron Aviation Inc,
  `CIRRUS` = Cirrus Design Corporation, `IAI` = Israel Aerospace/Aircraft Industries Ltd, `DIAMOND` =
  Diamond Aircraft Industries GmbH/Inc, `PIAGGIO` includes Piaggio Aero Industries SpA, `DASSAULT`
  includes Dassault Aviation, `DAHER` includes Compagnie Daher, `EPIC AIRCRAFT` = Epic Aircraft LLC. JO
  7360.1K lists the Textron-built Cessna types (408 SkyCourier, 700 Citation Longitude) under `CESSNA`.
  So the FAA itself files a Textron-held Cessna model under CESSNA, and the registry's "TEXTRON AVIATION
  INC" does not by itself block a CESSNA record.
- **GENOT N JO 7360.7** (effective 2025-10-14): adds ELECTRA EL-2 / EL2 only; changes nothing here.
- **FAA Aircraft Characteristics Database** (Oct 2024): one representative row per designator. Its
  `Model_BADA` column shows "Cessna 208B" on C208, "Dassault Falcon 2000EX" on F2TH and "Dassault Falcon
  900EX" on F900. It is supplementary only and is not a certificated-model crosswalk.

Page numbers below are PDF pages. In every TCDS/FSB cited, the printed page number equals the PDF page
(checked). JO pages are given as PDF page / printed label.

## A — accept (strict): the model designation or every FAA-listed marketing name is in Doc 8643, one designator

| FAA MFR | FAA MODEL | → | Doc 8643 record(s) (mfr \| model \| designator) | FAA evidence (page) | Flights |
|---|---|---|---|---|---|
| TEXTRON AVIATION INC | 700 | **C700** | CESSNA \| 700 Citation Longitude \| C700 | TCDS T00015WI Rev 9 p1: holder Textron Aviation Inc.; "I. Model 700 (Transport Category) Approved September 21, 2019". FSBR CE-700 Rev 2 p1: "T00015WI  700 (#0001 and on)  Longitude  CE-700". JO C-10 (p289): "700 Citation Longitude CESSNA C700" | 265 |
| TEXTRON AVIATION INC | 680A | **C68A** | CESSNA \| 680A Citation Latitude \| C68A | TCDS T00012WI Rev 18 p1 holder record Cessna → Textron (2015-07-29); p9 "II. Model 680A Latitude (Transport Category) S/N 680A0001 and On". JO C-10 (p289) | 192 |
| CIRRUS DESIGN CORP | SR22T | **S22T** | CIRRUS \| SR-22T \| S22T (also SR-22 Turbo \| S22T) | TCDS A00009CH Rev 25 p1 holder Cirrus Design Corporation; p7 "III - MODEL SR22T, NORMAL CATEGORY, APPROVED FEBRUARY 10, 2010". JO C-95 (p374) | 154 |
| TEXTRON AVIATION INC | 408 | **C408** | CESSNA \| 408 SkyCourier \| C408 | TCDS A00016WI Rev 6 p1 "I. Model 408 (Normal Category) Approved March 11, 2022". FSBR CE-408 Rev 0 p1 "A00016WI  408  SkyCourier  CE-408"; p3 "Textron Aviation Cessna Model 408 SkyCourier". JO C-7 (p286) | 103 |
| BOMBARDIER INC | BD-700-2A12 | **GL7T** | BOMBARDIER \| BD-700 Global 7500 / Global 7000 / Global 8000 \| GL7T | TCDS T00003NY Rev 24 p12 "VI - Model BD-700-2A12"; p24 NOTE 20 "Global 7500 (previously known as Global 7000) is a marketing designation for BD-700-2A12"; p25 NOTE 26 "Global 8000 is a marketing designation for BD-700-2A12 …". JO C-25 (p304) | 98 |
| AIRBUS | A330-343 | **A333** | AIRBUS \| A-330-300 \| A333 | TCDS A46NM Rev 42 p1 "A330-300 Series: … A330-341, A330-342, A330-343"; p13 "III. Airbus A330-300 Series …", "Airbus A330-343 - approved December 21, 2000". JO C-14 (p293) | 98 |
| GULFSTREAM AEROSPACE CORP | GVI (G650ER) | **GLF6** | GULFSTREAM AEROSPACE \| G-6 Gulfstream G650 \| GLF6 (the only G-6 record) | TCDS T00015AT Rev 22 p1 "I - GVI (Transport Category)"; p39–40 NOTE 5: ASC 001 "will designate those aircraft as Model GVI (G650)", ASC 014 "GVI Extended Range (G650ER)" "will designate those aircraft as Model GVI (G650ER)". JO C-52 (p331). *Note:* "G650ER" itself is not a Doc 8643 name; the model designation GVI (= G-6) is. | 69 |
| BOMBARDIER INC | BD-700-1A10 | **GLEX** | BOMBARDIER \| BD-700 Global Express / Global 6000 / Global 6500 \| GLEX | TCDS T00003NY p1 "I - Model BD-700-1A10"; p2 table "Marketing Designation / Eligible Serial Numbers": Global Express, Global Express XRS, Global 6000, Global 6500. JO C-25. *Note:* "Global Express XRS" is not a Doc 8643 name; the three named marketing names all → GLEX. | 55 |
| BOMBARDIER INC | CL-600-2B16 | **CL60** | CANADAIR \| CL-600 Challenger 601 / 604 / 605 \| CL60; BOMBARDIER \| CL-600 Challenger 650 \| CL60 | TCDS A21EA Rev 52 p6 "III - Model CL-600-2B16" (601-3A, 601-3R, 604 Variants); p19 NOTE 9 "The Challenger 605 is a marketing designation for the Challenger CL-600-2B16 (604 Variant) …"; NOTE 14 "The Challenger 650 is a marketing designation for the Challenger CL-600-2B16 (604 Variant) beginning with aircraft s/n 6050". JO C-33 (p312) | 55 |
| BOMBARDIER INC | BD-700-1A11 | **GL5T** | BOMBARDIER \| BD-700 Global 5000 / Global 5500 \| GL5T | TCDS T00003NY p3 "II - Model BD-700-1A11"; p4 table: Global 5000, Global 5000 ft. GVFD, Global 5500. JO C-25 | 52 |
| GULFSTREAM AEROSPACE | GIV-X (G450) | **GLF4** | GULFSTREAM AEROSPACE \| G-4X Gulfstream G450 \| GLF4 (G350 variant: G-4 Gulfstream G350 \| GLF4) | TCDS A12EA Rev 54 p41 "VIII - Model GIV-X"; "two variants of the GIV-X …: (1) The G450 … (2) the G350"; p61 NOTE 8 "eligible for identification as Model GIV-X (G450) when modified in accordance with … ASC 005". JO C-51 (p330) | 51 |
| GULFSTREAM AEROSPACE CORP | GVII-G600 | **GA6C** | GULFSTREAM AEROSPACE \| G-7 Gulfstream G600 \| GA6C | TCDS T00021AT Rev 13 p4 "II. - GVII-G600 (Transport Category), Approved June 28, 2019". JO C-52 | 47 |
| CIRRUS DESIGN CORP | SF50 | **SF50** | CIRRUS \| SF-50 Vision \| SF50 (also SJ-X Vision \| SF50) | TCDS A00018CH Rev 7 p1 "I - Model SF50 (7PCLM, Normal Category), Approved October 28, 2016". JO C-92 (p371) | 35 |
| GULFSTREAM AEROSPACE | GV-SP (G550) | **GLF5** | GULFSTREAM AEROSPACE \| G-5SP Gulfstream G550 \| GLF5 (G500 variant also GLF5) | TCDS A12EA p35 "VII. Model GV-SP"; "two variants of the GV-SP …: (1) The G550 … (2) the G500"; p61 NOTE 8 "identification as Model GV-SP (G550) … ASC 11". JO C-52 | 30 |
| TEXTRON AVIATION INC | 525B | **C25B** | CESSNA \| 525B Citation CJ3 \| C25B | TCDS A1WI Rev 34 p1 holder record Cessna → Textron; p16 "III. Model 525B (Commuter Category)". FSBR CE-525 Rev 10 p1 "A1WI  525B  CJ3, CJ3+, CJ3 Gen2". JO C-9 (p288) | 25 |
| TEXTRON AVIATION INC | 525C | **C25C** | CESSNA \| 525C Citation CJ4 \| C25C | TCDS A1WI p21 "IV. Model 525C (Commuter Category) Approved March 12, 2010". FSBR CE-525 p1 "A1WI  525C  CJ4". JO C-9 | 24 |
| EPIC AIRCRAFT LLC | E1000 | **EPIC** | EPIC AIRCRAFT \| E1000 \| EPIC | TCDS A00059SE Rev 7 p1 "TC Holder: Epic Aircraft, LLC", "Model E1000 (Utility Category), Approved November 6, 2019" (E1000 GX and E1000AX are AFM variants of Model E1000). JO C-41 (p320) | 24 |
| GULFSTREAM AEROSPACE CORP | GVII-G500 | **GA5C** | GULFSTREAM AEROSPACE \| G-7 Gulfstream G500 \| GA5C | TCDS T00021AT p1 "I. - GVII-G500 (Transport Category), Approved July 20, 2018". *Do not confuse* with "G-5SP Gulfstream G500 \| GLF5", which is GV-SP (G500) on TCDS A12EA. JO C-52 | 22 |
| PIAGGIO AERO INDUSTRIES SPA | P180 AVANTI II | **P180** | PIAGGIO \| P-180 Avanti \| P180 (the only Piaggio P-180 record) | TCDS A59EU Rev 27 p1: model P-180; holder record "… transferred TC A59EU to PIAGGIO AERO INDUSTRIES S.p.A. on November 17, 1998 …"; p9 Note 9 "S/N 1105 and up … Modification No. DMT 80-0587 … must use the 'P.180 AVANTI II Airplane Flight Manual'". JO C-76 (p355) | 21 |
| GULFSTREAM AEROSPACE CORP | GVIII-G700 | **GA7C** | GULFSTREAM AEROSPACE \| G-8 Gulfstream G700 \| GA7C | TCDS T00015AT p4 "II - GVIII-G700 (Transport Category), Approved March 29, 2024". JO C-52 | 19 |
| IAI LTD | GULFSTREAM G280 | **G280** | IAI \| Gulfstream G280 \| G280; GULFSTREAM AEROSPACE \| Gulfstream G280 \| G280 | TCDS A61NM Rev 12 p1 "I. Model Gulfstream G280", holder Gulfstream Aerospace LP (Israel); p5 "produced in Israel under production certificate PA-30 … (CAAI)". JO C-53 (p332); Appendix D IAI = Israel Aerospace Industries Ltd | 16 |
| TEXTRON AVIATION INC | B300 | **B350** | BEECH \| 300 (B300) Super King Air 350 \| B350; BEECHCRAFT \| 300 (B300) King Air 350 / 360 \| B350; HAWKER BEECHCRAFT, RAYTHEON 300 (B300) … \| B350 | TCDS A24CE Rev 132 p1 holder Textron Aviation Inc., record Beech → Raytheon → Hawker Beechcraft → Beechcraft Corp; p34 "IX. Model B300, Super King Air (Commuter Category)". *Contrast:* Model 300 (p30) is "300 Super King Air \| BE30". JO C-5 (p284) | 10 |
| GULFSTREAM AEROSPACE CORP | GVIII-G800 | **GA8C** | GULFSTREAM AEROSPACE \| G-8 Gulfstream G800 \| GA8C | TCDS T00015AT p7 "III - GVIII-G800 (Transport Category), Approved April 16, 2025" | 7 |
| GULFSTREAM AEROSPACE CORP | GVI | **GLF6** | as GVI (G650ER) | TCDS T00015AT p1 | 5 |
| TEXTRON AVIATION INC | 208 | **C208** | CESSNA \| 208 Caravan 1 (and every other CESSNA 208 record) \| C208 | TCDS A37CE Rev 25 p1 "I. Model 208, Caravan". JO C-4 (p283) | 2 |
| GULFSTREAM AEROSPACE | GV-SP | **GLF5** | both GV-SP variants (G550, G500) → GLF5 | TCDS A12EA p35 | 1 |
| CESSNA AIRCRAFT CO | 525C | **C25C** | as 525C | TCDS A1WI p21 | 1 |

On C700 and C408: their TCDS do not print a marketing name. The link from Model 700 to "Longitude" and
from Model 408 to "SkyCourier" comes from the FAA FSB report. JO 7360.1K and Doc 8643 then name "700 Citation
Longitude" and "408 SkyCourier" under CESSNA. If the coordinator accepts only TCDS or JO evidence, these two
rest on the model number: T00015WI is the only TC for a Textron/Cessna "Model 700", and C700 is the only Doc
8643 CESSNA/TEXTRON record whose model starts with "700". The same applies to 408.

## F — family-level only (Doc 8643 does not name this FAA model; coordinator decides)

The FAA model is a separately certificated derivative on the same TC as a named base model. Every Doc 8643
record for that family has one designator, and no competing designator exists. Criterion 1 is met only at
family level.

| FAA MFR | FAA MODEL | Candidate | Doc 8643 record(s) | FAA evidence | Flights |
|---|---|---|---|---|---|
| TEXTRON AVIATION INC | 208B | C208 | CESSNA \| 208 Caravan 1 / 208 Grand Caravan / 208 Cargomaster / 208 Super Cargomaster / 208 Caravan 675 / AC-208 Combat Caravan \| C208 | TCDS A37CE p4 "II. Model 208B, Caravan, 2 PCLM (Normal Category), Approved October 9, 1986". The TCDS never says "Grand Caravan". ACD row C208 has Model_BADA "Cessna 208B". | 28 |
| DIAMOND AIRCRAFT IND GMBH | DA 50 C | DA50 | DIAMOND \| DA-50 \| DA50 (also WANFENG DIAMOND \| DA-50) | TCDS A00064IB Rev 4 p1 "I. Model DA 50 C (Normal Category), approved July 25, 2023"; p4 NOTE 8 "… on the Diamond DA 50". JO C-37 (p316); Appendix D DIAMOND = Diamond Aircraft Industries GmbH. | 25 |
| DASSAULT / DASSAULT AVIATION | FALCON 2000EX | F2TH | DASSAULT \| Falcon 2000 \| F2TH (the only Falcon 2000 record) | TCDS A50NM Rev 16 p5 "II. Model FALCON 2000EX (Transport Category Airplane) approved March 21, 2003"; "The Falcon 2000EX is defined by Dassault modification M1802 and differs from the Falcon 2000 …". The 2000DX, LX, LXS and S are commercial designations of Model 2000EX (p9–10). ACD row F2TH has Model_BADA "Dassault Falcon 2000EX". | 19 + 9 |
| DASSAULT AVIATION | FALCON 900EX | F900 | DASSAULT \| Falcon 900 / Mystère 900 / T-18 \| F900 | TCDS A46EU Rev 23 p9 "III. Model FALCON 900EX (Transport Category Airplane), approved July 19, 1996"; p10 "defined by Dassault modification M3000 and differs from the Mystere-Falcon 900". ACD row F900 has Model_BADA "Dassault Falcon 900EX". | 8 |
| TEXTRON AVIATION INC | B300C | B350 | "300 (B300) …" records → B350; HAWKER BEECHCRAFT \| MC-12 \| B350 | TCDS A24CE p34 "Model B300C, B300C(MC-12W), B300C(UC-12W), Super King Air" | 1 |

Checked: none of "2000EX", "900EX", "G650ER", "CJ1+", "XRS", "B300C", "Gen2" or "TBM 980" appears in
the Doc 8643 snapshot or in JO 7360.1K.

## S — ambiguous, and an FAA document gives a serial split

The FAA registry MASTER file carries each aircraft's serial number. A rule would read that field.

1. **BOMBARDIER INC | BD-100-1A10 (254 flights)** → CL30 or CL35.
   Doc 8643: "BOMBARDIER | BD-100 Challenger 300 | CL30", "BD-100 Challenger 350 | CL35", "BD-100 Challenger 3500 | CL35".
   TCDS T00005NY Rev 12 (current, 2019-07-02) p7 NOTE 7, verbatim: *"'Challenger 300' is a marketing designation for the BD-100-1A10 up to aircraft S/N 20500. 'Challenger 350' is a marketing designation for the BD-100-1A10 starting at aircraft S/N 20501."* This matches p1: "AS907-1-1A for S/N 20002 to 20500 … AS907-2-1A for S/N 20501 and subsequent". FSBR BD-100-1A10 Rev 8 p1 gives "Challenger 300 and Challenger 350" as the marketing names on T00005NY.
   Rule the evidence supports: S/N 20002–20500 → CL30; S/N ≥ 20501 → CL35. The TCDS predates the 3500 name, but the 3500 is also CL35, so the split holds either way.
   Also: "BOMBARDIER AEROSPACE INC | BD-100-1A10" (model code 1390050, 5 icao24, 0 flights) is the same model.
2. **DASSAULT AVIATION | FALCON 7X (28 flights)** → FA7X or FA8X. **This is new: the Falcon 8X is certificated as Model Falcon 7X.**
   Doc 8643: "DASSAULT | Falcon 7X | FA7X", "DASSAULT | Falcon 8X | FA8X".
   TCDS A59NM Rev 10 p1 "(a) Basic Model Definition"; p4 "Serial Numbers Eligible   Serial numbers 0001 through 0400."; p4 "(b) Falcon 8X Definition … The Falcon 8X does not correspond to a model designation. The Falcon 8X is only a commercial designation for stretch version of the Falcon 7X airplanes that incorporates modifications M1000 and M1254 (EASy III) installed at production. … M1000 is basic on all Falcon 7X aircraft starting with serial number 0401."; p10 "DGT 125953 applicable to S/N 001 to S/N 400 (modification M1000 not included) / DGSM 151456 applicable to S/N 401 and up (modification M1000 included)". FSBR DA-7X Rev 6 p1 lists TCDS model "Falcon 7X" under both marketing names, Falcon 7X and Falcon 8X.
   Rule: S/N 0001–0400 → FA7X; S/N ≥ 0401 → FA8X. The 8X also needs M1254, which the TCDS does not say is basic from 0401; only M1000 (the stretch) is.
3. **COMPAGNIE DAHER | TBM 700 (13 flights)** → TBM7, TBM8 or TBM9.
   Doc 8643: SOCATA/TBM TBM-700/700A/700B/700C → TBM7; TBM-700N (TBM-850) → TBM8; TBM-700N (TBM-900/910/930/940/960) → TBM9. There is no TBM 980 record.
   TCDS A60EU Rev 40 p1 "I. Model TBM 700 … [See note 14 … for information on different versions and trade names of the TBM 700.]". p14 Note 14 (verbatim in excerpt): TBM 700A s/n 1-125; 700B s/n 126-243 except 205 and 240; 700C1/C2 s/n 244-345 except 269, plus 205 and 240; TBM 700N "(Trade name: TBM850)" s/n 346-433 except 269 and s/n 434-999 except 687; "(TBM900)" s/n 687, 1000-1169 (with MOD70-0176-00); TBM930 1111-1271; TBM910 1170-9999; TBM940 1172, 1275-9999; TBM960 1408-9999; **"(Trade name: TBM980) … = s/n 1627-9999"**.
   Rule the evidence supports: s/n 1–345 except 269 → TBM7; s/n 346–999 except 687 → TBM8; s/n 687 and 1000–1626 → TBM9. s/n 269 is excluded from every list, so leave it unresolved. **s/n ≥ 1627 may be a TBM 980, which the snapshot does not name, so leave it unresolved.** Within 1000–1626 the trade names overlap by modification, but they all map to TBM9.
   The registry also has "DAHER AIRCRAFT SAS | TBM 700" (22 icao24, 0 flights).
4. **TEXTRON AVIATION INC | 525 (8 flights)** → C525 or C25M.
   Doc 8643: "CESSNA | 525 CitationJet | C525", "525 Citation CJ1 | C525", "525 Citation M2 | C25M".
   FSBR CE-525 Rev 10 p1 lists Model 525 marketing names "CJ, CJ1, CJ1+, M2, M2 Gen2". p96 Appendix 4 serials: 525 (CJ) 525-0001 thru 525-0359; (CJ1) 525-0360 thru 525-0558; (CJ1+) 525-0600 thru 525-0701 excl. 525-0685; **(M2) 525-0685 and 525-0800 and on**. TCDS A1WI p1 has the same serial groups by approval date ("S/N 525-0685, 525-0800 & On Approved December 20, 2013").
   Rule: 525-0685 or ≥ 525-0800 → C25M; 525-0001–0558 → C525; CJ1+ (0600–0701 excl. 0685) → C525 at family level only, because CJ1+ is not a Doc 8643 name.
   A1WI NOTE 9: serials "manufactured under the name Textron Aviation Inc." are 525-0875, 0877, 0878, 0881 & On, which are all in the M2 range.

## N — ambiguous, no FAA serial split

**ATR-GIE AVIONS DE TRNSP RGNL | ATR 72-212A (53 flights, all KRDU)** → AT75 or AT76.
Doc 8643: "ATR | ATR-72-212A (500) | AT75", "ATR-72-212A (600) | AT76", "ATR-72-500 | AT75", "ATR-72-600 | AT76".
TCDS A53EU Rev 38 p27: *"The ATR72-212A '600 version' designation does not correspond to a model designation. This is only a commercial designation for an ATR72-212A on which Major modifications 5948, 6521, and 5977 have been embodied during production."* The 600F version is modification 7900. FSBR ATR-42/72 Rev 8 p1 gives the marketing names of TCDS model ATR72-212A as ATR 72-500, ATR 72-600 and ATR 72-600F.
The split is by production modification, which the registry does not carry, and neither FAA document gives a serial boundary. **Leave unresolved.** EASA TCDS A.084 was not consulted; it is outside the FAA-only rule.

## Why the current matcher misses these (for the coordinator; no code touched)

- `_manufacturer_family` has no alias for the registry spellings CIRRUS DESIGN CORP, EPIC AIRCRAFT LLC, IAI LTD,
  DIAMOND AIRCRAFT IND GMBH, DASSAULT AVIATION, PIAGGIO AERO INDUSTRIES SPA, COMPAGNIE DAHER or ATR-GIE…,
  so it compares them against an empty family. SR22T → "SR-22T", E1000 and "GULFSTREAM G280" would otherwise match exactly after `_compact`.
- TEXTRON AVIATION → CESSNA is right for Cessna TCs, but B300 is a Beech TC (A24CE). Its Doc 8643 records are
  under BEECH/BEECHCRAFT/HAWKER BEECHCRAFT/RAYTHEON, so a Textron→Cessna-only alias can never reach B350.
- The unique-prefix rule needs ≥ 5 compact characters, so "700", "408", "680A", "525B", "525C", "SF50" and "208B" never reach it.
- Certificated designations (BD-100-1A10, BD-700-xAxx, CL-600-2B16, GVI/GVII/GVIII, GIV-X, GV-SP, A330-343) never appear in Doc 8643 model names. Only the TCDS and FSB marketing tables connect them.

## Incidental findings (outside this task; verified against the cited documents)

- `faa_aircraft_identity.json` model 1182127 "BELL HELICOPTER TEXTRON INC | 525" carries `typecode` **C525**
  (registration_crosswalk, 3/3). Bell 525 is a helicopter: Doc 8643 "BELL | 525 Relentless | B525" and JO C-9 "525 Relentless BELL B525 Helicopter". The crosswalk value is wrong.
- Model 2076601 "CESSNA | 525" carries **C25A** (registration_crosswalk, 377/391 = 96.4%). By TCDS A1WI and FSBR
  CE-525, Model 525 is CJ/CJ1/CJ1+/M2 (C525/C25M); C25A is Model **525A** (CJ2), a different FAA model. Model 2076653 "CESSNA AIRCRAFT CO | 525" → C25A has the same problem.
- The TBM entries "SOCATA | TBM 700 → TBM7" (98.3% crosswalk), "EADS SOCATA | TBM 700 → TBM7" and
  "EADS SOCATA | TBM700N → TBM9" are all FAA model TBM 700. By A60EU Note 14 that model spans TBM7/8/9 by serial, so a majority vote is not evidence here.

## What was tried and how the sources were obtained

- JO 7360.1: the faa.gov "document.current" page names **JO 7360.1K** (issued 2025-04-10, effective 06/12/2025, cancels 7360.1J) and one current notice, GENOT N JO 7360.7. Both PDFs were downloaded. `curl` needs a browser User-Agent; without one faa.gov times out or returns 403.
- DRS (drs.faa.gov) is a single-page app. The document list and PDFs come from its JSON API after a guest session (`GET /guest/login?targetUrl=…`, which sets cookies):
  `POST /api/browse/doctype/TCDSMODEL/documents/metadatas {"page":n}` pages through all **2,227 current TCDS**, with model metadata. The FSB reports are doctype `FSB_REPORTS` (114 current).
  `GET /api/browse/documents/summaryguiddocview/<docUniqueId>` returns the revision, status and file id, and
  `GET /api/content/alf/<id>` returns the PDF (size checked against DRS `sizeInBytes`).
  Every TCDS used is the DRS **Current** revision as of 2026-09-23. The index also has a bulk archive, "TCDS_08312026.zip"; it was not downloaded.
- The TCDS for each model was found from the DRS Model metadata, not by guessing numbers. The index for Textron Model 700 and 408 was searched separately because the regex first missed them.
- FAA Aircraft Characteristics Database: kept only as supplementary evidence for the three family-level Falcon/Caravan rows.
- Not used (per rule): OpenSky, ADS-B Exchange, Wikipedia and planespotting sites. EASA TCDS were not used.

## Sources

| Source | Edition | URL | Local path | sha256 |
|---|---|---|---|---|
| FAA Order JO 7360.1K, Aircraft Type Designators | eff. 06/12/2025 | https://www.faa.gov/documentLibrary/media/Order/FAA_Order_JO_7360.1K_Aircraft_Type_Designators.pdf | data/AIRCRAFT/sources/faa/FAA_Order_JO_7360.1K_Aircraft_Type_Designators.pdf (11,971,409 B) | 64a709bf8863e177c3daffade253aa6362519186ba807943165f68490ee0c8fe |
| GENOT N JO 7360.7 (change to JO 7360.1) | eff. 10/14/2025 | https://www.faa.gov/documentLibrary/media/Notice/GENOT_N_JO_7360.7_Change_to_FAA_Order_JO_7360.1_Aircraft_Type_Designators.pdf | data/AIRCRAFT/sources/faa/GENOT_N_JO_7360.7_…pdf (196,098 B) | bc659054b08e5b1d9fa63d8e4e78d925fdff1b380ef613ecbe6c0777429746e3 |
| FAA Aircraft Characteristics Database (supplementary) | Oct 2024 (mod. 2025-01-03) | https://www.faa.gov/airports/engineering/aircraft_char_database/aircraft_data | data/AIRCRAFT/sources/faa/FAA_Aircraft_Characteristics_Data_2024-10.xlsx (125,570 B) | 3e68848b6bb99f9700e25d33c080b993a80dc962d581a391f89aee94b6c8fcb3 |
| ICAO Doc 8643 (repo snapshot, not re-downloaded) | last_updated 10 July 2026 | https://doc8643.icao.int/External/AircraftTypes | aircraft/icao_doc8643.json | raw_sha256 in file: 53c64ae8…9006 |

FAA TCDS (drs.faa.gov, all "Current" on 2026-09-23; local `data/AIRCRAFT/sources/faa/tcds/<file>`; URL `https://drs.faa.gov/browse/excelExternalWindow/<docUniqueId>` in `tcds/manifest.json`):

| File | Rev date | sha256 |
|---|---|---|
| T00015WI_Rev9.pdf (Textron 700) | 06/29/2023 | 794257f7262a8ff039e0afada08ad34442423962fa52be5e2120781c4f667d98 |
| T00005NY_Rev12.pdf (BD-100-1A10) | 07/02/2019 | 746c8f8deb808820baa77231f592f9b65b426fd08d167d1b9aa6c2bf9ce3b805 |
| T00012WI_Rev18.pdf (680, 680A) | 09/18/2025 | 19ac1c2660499133baef866d70f6e045c5d28129845e1d6e7d910b0adc8e9930 |
| A00009CH_Rev25.pdf (SR20/22/22T) | 12/26/2025 | dc688b1226e75ceeb10371e2211e203b40a04f0656f6468d77b738b91dee3ddd |
| A00016WI_Rev6.pdf (408) | 06/26/2026 | 395fb11926b51a3d393d2da84a43495d3f3d7a33d564009b0cfc4e987781ed13 |
| T00003NY_Rev24.pdf (BD-700) | 12/19/2025 | 2e6cd88f2bdcb6228eecb078ba3f9ce9efc309a582709a4d0c2b5abec2965dc5 |
| A46NM_Rev42.pdf (A330) | 06/02/2025 | f7921ce6875fc2fc8c3f765a120e7938eb14ca2bde58743687b0c6913825a0ee |
| T00015AT_Rev22.pdf (GVI, GVIII) | 04/16/2025 | 57dfc74d827b990828d9f2da5287e47ba4b16a9fde8aebec9648f197daf814b8 |
| A21EA_Rev52.pdf (CL-600) | 07/29/2026 | ad195be08f73bf96df65ffbfcd54d5d0cb87251d73a0f78c6374ac1c8000c31b |
| A53EU_Rev38.pdf (ATR 42/72) | 08/27/2026 | 7f014077b22e4f9e962f54fa81b9b35a89de227b41dac5bf7f927e628105a724 |
| A12EA_Rev54.pdf (G-1159…GIV-X, GV-SP) | 04/21/2026 | d611384c2866dca673c43ea56ccf0dd1ff23f6a3605b8853007f3161ba770789 |
| T00021AT_Rev13.pdf (GVII) | 04/09/2026 | 4f2793fbcb609a477e09252176184775cfb39fbed343cdcf8828c09ce860b0d8 |
| A00018CH_Rev7.pdf (SF50) | 12/26/2025 | ffc2bc93dae22239afad21e1d35c39aa484494334d7ff3c08e1f043d95115076 |
| A59NM_Rev10.pdf (Falcon 7X) | 07/02/2025 | 36f967dd850b1b9dc6e3eb6de5caa5d13b16d949f88ae791fd9e347b82a3ef5f |
| A37CE_Rev25.pdf (208, 208B) | 08/01/2025 | 4d31562f875b4b2d18a1d6f98404ffd7b2520a39f2f90783c8054dece06dd862 |
| A1WI_Rev34.pdf (525–525C) | 01/05/2026 | fd9fc2c3120a639f75b1cc50bb8e5eb21460680a0db28d00548bcf9721623e10 |
| A00064IB_Rev4.pdf (DA 50 C) | 01/16/2025 | 668a2083ed5fa1c98a7cfc3505ea1614bccdc76acc71eb16fcf4dc525f46c750 |
| A00059SE_Rev7.pdf (E1000) | 06/12/2026 | f37fc0759cf39b1399e20726fee63da0eb1dbe093dae4b46ae34007d944bc9be |
| A50NM_Rev16.pdf (Falcon 2000/2000EX) | 11/13/2025 | f3952bdd89e69257e863312293fa4c3fb1480f3a031e9fed50a2b67bc6ad70be |
| A46EU_Rev23.pdf (Falcon 50/900/900EX) | 01/16/2026 | a8c444d8dfaef39fb0e21178267facb2c7946729af6b486378b9c546d9b4dc06 |
| A59EU_Rev27.pdf (P-180) | 09/15/2026 | 1b8c162609b0c7815964c153ccd47d9237b491331f69f7d0ffb96aa56fecca2f |
| A60EU_Rev40.pdf (TBM 700) | 04/24/2026 | 9e3588247658bd833343d0b5427f4a372644bb031fd07ed0e95c5988c2006a84 |
| A24CE_Rev132.pdf (King Air 200/300/1900) | 08/21/2026 | 4591b36127f8a54f60fe77fc8a714d6ea071f459043b8ef9607b33ad4c621ad4 |
| A61NM_Rev12.pdf (G280) | 12/21/2021 | 832efadaf068af2f8e6e6f57c6ec71b190b4da6090c2dba4ecbee1f258220bf1 |

FAA FSB reports (local `data/AIRCRAFT/sources/faa/fsbr/`, metadata in `fsbr/manifest.json`):

| File | Issue date | sha256 |
|---|---|---|
| FSBR_CE-700_Rev_2.pdf | 10/05/2023 | 7379f88d58de733d9787a9b062718b608999988e53c887cc5bb548724d4f586e |
| FSBR_CE-408_Rev_0.pdf | 04/01/2022 | 5f5fb7b57d5d4ad915edb6e79815c5d2fa9ea1f38b2510a9c8e70c5855e1c8c2 |
| FSBR_CE-525_Rev_10.pdf | 09/02/2026 | 703e61fbc23b1c16834a73da0e9121af3692dd1479699886a135951021e36a94 |
| FSBR_BD-100-1A10_Rev_8.pdf | 09/24/2024 | a98a9e806003dc12ede514d2b62d4a65aa0205b91701485c02880756ee342a74 |
| FSBR_DA-7X_Rev_6.pdf | 03/17/2023 | 5469c40b0b24f4d5fca6554e57c27ad3b4a52e081029e17136d7fb8d875ce9b0 |
| FSBR_ATR-42_72_Rev_8.pdf | 07/20/2022 | ec019f8f8557fa3abf560a81881c7527c77bc69b94a3afed41ff29d7648caebf |

## Not researched (≤ 8 flights each, 107 flights total)

BOEING 737-3Y0 (8), EYE SEE EXPRESS LIV-P (8), TEXTRON 182T (8), TEXTRON 172S (8), BOEING 737-82R (7),
HAWKER BEECHCRAFT HAWKER 900XP (7), KODIAK 100 (7), GAME COMPOSITES GB1 (6), MHI RJ CL-600-2C10 (6),
DIAMOND DA 62 (4), EMBRAER ERJ 170-100 STD (4), PILATUS PC-12/47G (3), TECNAM P-MENTOR (3), TEXTRON T206H (3),
SAAB 2000 ×2 strings (3), TECNAM P2010 TDI (2), DASSAULT MYSTERE FALCON 900 (2), and 18 single-flight
amateur-built/LSA/other strings. Of these, the 737 (TCDS A16WE, Rev 81), CL-600-2C10 (TCDS A21EA-1, Rev 7,
holder MHI RJ Aviation ULC, "Regional Jet Series 700/701/…" so likely CRJ7 but needs checking),
Hawker 900XP, PC-12/47G and the Cessna 172S/182T/T206H are the likely cheap next steps.
