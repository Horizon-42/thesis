# Reference approach speeds and weights - provenance pack

Generated 2026-09-07. Machine-readable table: [`aircraft/reference_speeds.json`](../../aircraft/reference_speeds.json)
(schema `aircraft-reference-speeds-v1`). Verbatim backing excerpts: [`excerpts/`](excerpts/).

This pack exists so that every approach speed and every mass used by the evaluation gate can be
traced, in one hop, to a page of a published document that is on this disk.

## What is and is not in git

* **Everything under `data/reference_speeds/` is git-ignored** (`data/` is not tracked). The PDFs,
  the FAA spreadsheet and the 41 Eurocontrol HTML pages are local artifacts.
* **`fetch_sources.sh` in this directory re-downloads all of them** to the same paths, with a browser
  User-Agent, printing the sha256 of each file. It **never overwrites an existing file**, so running
  it on a populated tree only prints checksums you can compare against the tables below.
* This README, the `excerpts/` directory and `aircraft/reference_speeds.json` **are** tracked, so the
  numbers and the quotes that justify them survive without the source documents.
* OpenAP's YAML files are not copied here at all; they live in the `aeroviz` conda env. Their values
  and checksums are pinned in `excerpts/openap_2_4_oew.csv`.

## Conventions

* `approach_speed_kt` is **always** the FAA `Approach_Speed_knot`. Manufacturer and Eurocontrol values
  are recorded as corroboration only; they never replace it.
* `malw_kg` = FAA `MALW_lb` x 0.45359237, rounded to the kilogram. That rounding can differ by 1 kg
  from the kilogram figure printed by the manufacturer (e.g. CRJ900 33,339 here vs 33,340 printed),
  because the manufacturers round their own conversion. The single exception is **B739**, where the
  FAA cell is wrong - see the row note.
* `min_mass_kg` is the lowest published operating-empty / basic-operating / minimum-flight weight for
  the designator. Preference order: manufacturer minimum flight weight, then manufacturer OEW/BOW from
  a published document, then OpenAP 2.4. Where a manufacturer publishes both a passenger and a
  freighter/BCF variant, the **passenger baseline** is the recorded value and the freighter figure is
  named in the note, because the observed fleet is scheduled passenger traffic.
* `null` means nothing published was found. Four types are in that state; none of them is guessed.

## Sources

### faa_acd_2024_10

**Aircraft Characteristics Database** - FAA Office of Airports  
Document: Aircraft Characteristics (October 2024); page last updated 2025-01-07  
URL: https://www.faa.gov/airports/engineering/aircraft_char_database/aircraft_data  
Retrieved: 2026-09-07  
Local path: `data/reference_speeds/faa/FAA-Aircraft-Characteristics-Database-2024-10.xlsx` (125,570 bytes)  
sha256: `3e68848b6bb99f9700e25d33c080b993a80dc962d581a391f89aee94b6c8fcb3`

Read: Sheet `ACD_Data`, the one row per fleet ICAO type (all 40 present): columns 0 `ICAO_Code`, 7-9 `AAC`/`AAC_minimum`/`AAC_maximum`, 12-14 `Approach_Speed_knot`/`_minimum_knot`/`_maximum_knot`, 22 `MTOW_lb`, 23 `MALW_lb`, 39 `Remarks`. Sheet `Data_Dictionary` rows 14-16 and 24-25 for the column definitions and note 2 on dual AAC values. Rows copied verbatim to `excerpts/faa_acd_2024-10_fleet_rows.csv`, definitions to `excerpts/faa_acd_2024-10_definitions.txt`. This is the ONLY source of `approach_speed_kt` in the JSON.

### eurocontrol_apd

**Aircraft Performance Database** - EUROCONTROL  
Document: per-type web page, retrieved 2026-09-06  
URL: https://contentzone.eurocontrol.int/aircraftperformance/details.aspx?ICAO=<TYPE>  
Retrieved: 2026-09-06  
Local path: `data/reference_speeds/eurocontrol/<TYPE>.html`  
sha256: `per file, see docs/reference_speeds/excerpts/eurocontrol_apd_index.csv`

Read: One page per ICAO type, 41 files. Parsed fields: `Landing Vat (IAS)` kt, `MTOW` kg, `APC`. Used only as corroboration, never as the primary speed. E75L, GLF6 and PC24 return 'No ICAO' - the database has no entry for those designators. All 40 fleet rows with their per-file sha256 are in `excerpts/eurocontrol_apd_index.csv`.

### boeing_737ng_acap_rev_c

**737 Airplane Characteristics for Airport Planning** - Boeing  
Document: D6-58325-7 Rev C, October 2025  
URL: https://www.boeing.com/content/dam/boeing/v2/airports/acaps/737NG_REV_C.pdf  
Retrieved: 2026-09-07  
Local path: `data/reference_speeds/boeing/737NG_REV_C.pdf` (20,031,940 bytes)  
sha256: `22b1746201262dc34bb3ea46ae710b8802aff5c7df45ee73106f6f5ce9eb65bf`

Read: Section 2.1 tables 2.1.2 (737-700/-700W/-700C), 2.1.3 (737-800/-800W/-800BCF), 2.1.4 (737-900/-900W), 2.1.5 (737-900ER/-900ERW), printed pages 2-3 to 2-6, PDF pages 34-37. Taken: MAX DESIGN LANDING WEIGHT and OPERATING EMPTY WEIGHT per model and weight variant. Excerpt `excerpts/boeing_737NG_2.1_weights.txt`. **No approach speed is published anywhere in this document.**

### boeing_737cl_acap_rev_e

**737 Classic Airplane Characteristics for Airport Planning** - Boeing  
Document: D6-58325-6 Rev E, November 2023  
URL: https://www.boeing.com/content/dam/boeing/v2/airports/acaps/737CL_REV_E.pdf  
Retrieved: 2026-09-07  
Local path: `data/reference_speeds/boeing/737CL_REV_E.pdf` (17,289,799 bytes)  
sha256: `abe0baae09d582ecd63c5dd048117fd1ef3812b702c2e3096e0af10c4c8af360`

Read: Sections 2.1.7 (737-400) and 2.1.8 (737-500), printed pages 2-8 and 2-9. Taken: MAX DESIGN LANDING WEIGHT and OPERATING EMPTY WEIGHT. Excerpt `excerpts/boeing_737CL_2.1_weights.txt`. **No approach speed is published.**

### boeing_737max_acap_rev_k

**737 MAX Airplane Characteristics for Airport Planning** - Boeing  
Document: D6-38A004 Rev K, July 2025  
URL: https://www.boeing.com/content/dam/boeing/v2/airports/acaps/737MAX_RevK.pdf  
Retrieved: 2026-09-07  
Local path: `data/reference_speeds/boeing/737MAX_RevK.pdf` (18,127,573 bytes)  
sha256: `6ac41b3794c19e96ec7413fe45f2cc13f68d748561573a032d13ea2d6a9119ee`

Read: Sections 2.1.2 (737-8) and 2.1.5 (737-9), printed pages 2-3 and 2-6. Taken: MAX DESIGN LANDING WEIGHT. Excerpt `excerpts/boeing_737MAX_2.1_weights.txt`. **This document publishes no OPERATING EMPTY WEIGHT row and no approach speed** - that absence is why B38M/B39M fall back to OpenAP for minimum mass.

### boeing_757_acap_rev_h

**757 Airplane Characteristics for Airport Planning** - Boeing  
Document: D6-58327 Rev H, December 2024  
URL: https://www.boeing.com/content/dam/boeing/v2/airports/acaps/757_Rev_H.pdf  
Retrieved: 2026-09-07  
Local path: `data/reference_speeds/boeing/757_Rev_H.pdf` (9,620,693 bytes)  
sha256: `7f2dde89ecbdecbc8d3254cd4c028d8ece96aae4aa56f8a314406dd996dc4079`

Read: Sections 2.1.1 to 2.1.3 (757-200 with RB211 and with PW2037/PW2040 engines, 757-200PF), printed pages 2-6 to 2-8. Taken: MAX DESIGN LANDING WEIGHT and SPEC OPERATING EMPTY WEIGHT. Excerpt `excerpts/boeing_757_2.1_weights.txt`. **No approach speed is published.**

### boeing_767_acap_rev_k

**767 Airplane Characteristics for Airport Planning** - Boeing  
Document: D6-58328 Rev K, December 2024  
URL: https://www.boeing.com/content/dam/boeing/v2/airports/acaps/767_REV_K.pdf  
Retrieved: 2026-09-07  
Local path: `data/reference_speeds/boeing/767_REV_K.pdf` (21,785,773 bytes)  
sha256: `af2a91f1cba372f09592aadf0b3f60fdc6cd8144034941292b45ddd695d3d3eb`

Read: Sections 2.1.1 to 2.1.4 (767-200, -200ER, -300, -300ER), printed pages 2-2 to 2-5. Taken: MAX DESIGN LANDING WEIGHT and SPEC OPERATING EMPTY WEIGHT. Excerpt `excerpts/boeing_767_2.1_weights.txt`. **No approach speed is published.**

### boeing_777_acap_rev_e

**777-200/-200ER/-300 Airplane Characteristics for Airport Planning** - Boeing  
Document: D6-58329 Rev E, December 2024  
URL: https://www.boeing.com/content/dam/boeing/v2/airports/acaps/777-200-200ER-300_Rev_E.pdf  
Retrieved: 2026-09-07  
Local path: `data/reference_speeds/boeing/777-200-200ER-300_Rev_E.pdf` (20,273,897 bytes)  
sha256: `ef189aefb353fc15a33866453c51fe2e6251d54a8ca979ba639e9980e5cf19a5`

Read: Sections 2.1.1 to 2.1.3 (777-200 with GE, PW and RR engines), printed pages 2-2 to 2-4. Taken: MAX DESIGN LANDING WEIGHT and SPEC OPERATING EMPTY WEIGHT. Excerpt `excerpts/boeing_777_2.1_weights.txt`. **No approach speed is published.** This document does not cover the 777-300ER (B77W).

### boeing_777lr_300er_f_acap_rev_g

**777-200LR/-300ER/777F Airplane Characteristics for Airport Planning** - Boeing  
Document: D6-58329-2 Rev G, December 2024  
URL: https://www.boeing.com/content/dam/boeing/v2/airports/acaps/777-200LR-300ER-F_Rev_G.pdf  
Retrieved: 2026-09-07  
Local path: `data/reference_speeds/boeing/777-200LR-300ER-F_Rev_G.pdf` (12,858,800 bytes)  
sha256: `9c98f2e9c013615c0eaa9dc87ccb82544b7a3333ed1b725d996e283d50da142a`

Read: Section 2.1.1 (777-200LR, 777-300ER, 777F), printed page 2-2. Taken: MAX DESIGN LANDING WEIGHT for the 777-300ER. Excerpt `excerpts/boeing_777LR-300ER_2.1_weights.txt`. **This document publishes no OPERATING EMPTY WEIGHT row and no approach speed.**

### boeing_787_acap_rev_q

**787 Airplane Characteristics for Airport Planning** - Boeing  
Document: D6-58333 Rev Q, October 2025  
URL: https://www.boeing.com/content/dam/boeing/v2/airports/acaps/787_ACAP_Rev_Q.pdf  
Retrieved: 2026-09-07  
Local path: `data/reference_speeds/boeing/787_ACAP_Rev_Q.pdf` (11,726,333 bytes)  
sha256: `b9f0398de13da78aa0391b9aefa32b1ecdecf5032d83413381018aeae81c57f6`

Read: Sections 2.1.1 (787-8) and 2.1.2 (787-9), printed pages 2-2 and 2-3. Taken: MAX DESIGN LANDING WEIGHT. Excerpt `excerpts/boeing_787_2.1_weights.txt`. **This document publishes no OPERATING EMPTY WEIGHT row and no approach speed.**

### airbus_ac_a319_0624

**A319 Aircraft Characteristics - Airport and Maintenance Planning** - Airbus  
Document: issue Jun 01/24 (file AC_A319_0624)  
URL: https://www.aircraft.airbus.com/sites/g/files/jlcbta126/files/2024-06/AC_A319_0624.pdf  
Retrieved: 2026-09-07  
Local path: `data/reference_speeds/airbus/AC_A319_0624.pdf` (6,253,171 bytes)  
sha256: `c59ecba5b9449e35a0a68bd1ddafcaa69f24261ece7c807f38dcbd2e3b341e9c`

Read: Subject 3-5-0 Final Approach Speed, PDF page 164: 126 kt at MLW 62,500 kg (137,789 lb), AAC C. Subject 2-1-1 General Aircraft Characteristics Data: MLW per weight variant (61,000 kg for WV000/001/003/007, 62,500 kg for the rest). Excerpts `excerpts/airbus_A319_AC_3-5-0.txt` and `excerpts/airbus_A319_AC_2-1-1_mlw.txt`. **Airbus AC documents publish no OEW.**

### airbus_ac_a320_0624

**A320 Aircraft Characteristics - Airport and Maintenance Planning** - Airbus  
Document: issue Jun 01/24 (file AC_A320_0624)  
URL: https://www.aircraft.airbus.com/sites/g/files/jlcbta126/files/2025-01/AC_A320_0624.pdf  
Retrieved: 2026-09-07  
Local path: `data/reference_speeds/airbus/AC_A320_0624.pdf` (6,887,839 bytes)  
sha256: `f5cd06dfa3e6fe429301907e38e0470c37599f61bd808c8df47a37d5277836a2`

Read: Subject 3-5-0 Final Approach Speed, PDF page 180: A320-200 136 kt at MLW 66,000 kg (145,505 lb) AAC C; A320neo 131.5 kt at MLW 67,400 kg (148,592 lb) AAC C. Subject 2-1-1: MLW per weight variant (64,500 kg or 66,000 kg for the A320-200). Excerpts `excerpts/airbus_A320_AC_3-5-0.txt`, `excerpts/airbus_A320_AC_2-1-1_mlw.txt`. **No OEW is published.**

### airbus_ac_a321_1223

**A321 Aircraft Characteristics - Airport and Maintenance Planning** - Airbus  
Document: issue Dec 01/23 (file ac_a321_1223)  
URL: https://www.aircraft.airbus.com/sites/g/files/jlcbta126/files/2023-12/ac_a321_1223.pdf  
Retrieved: 2026-09-07  
Local path: `data/reference_speeds/airbus/ac_a321_1223.pdf` (8,143,679 bytes)  
sha256: `932dad53c83190f506070bc5b4dbb3af33d24a7c3da858ce3083dc843d3b53f3`

Read: Subject 3-5-0 Final Approach Speed, PDF page 191: A321-100/-200 140 kt at MLW 75,500 kg (166,449 lb) AAC C and 142 kt at MLW 77,800 kg (171,520 lb) AAC D; A321neo 136 kt at MLW 79,200 kg (174,606 lb) AAC C. Subject 2-1-1: MLW per weight variant. Excerpts `excerpts/airbus_A321_AC_3-5-0.txt`, `excerpts/airbus_A321_AC_2-1-1_mlw.txt`. **No OEW is published.**

### airbus_ac_a300_600_dec2009

**A300-600 Airplane Characteristics for Airport Planning** - Airbus  
Document: DEC 01/09  
URL: https://www.aircraft.airbus.com/sites/g/files/jlcbta126/files/2023-02/Airbus-Commercial-Aircraft-AC-A300-600-Dec-2009.pdf  
Retrieved: 2026-09-07  
Local path: `data/reference_speeds/airbus/Airbus-Commercial-Aircraft-AC-A300-600-Dec-2009.pdf` (4,720,011 bytes)  
sha256: `f656ed63a07fbca17c3755dec452350be750bbab647b4c910f58e448ba88f516`

Read: Chapter 2.1 General Airplane Characteristics Data (definitions page plus the A300B4-600 / A300C4-600 and A300B4-600R tables): MLW 138,000 kg (B4-600) / 140,000 kg (B4-600R) and **Estimated Operational Empty Weight** per engine - the only Airbus document in this pack that publishes an OEW. Excerpt `excerpts/airbus_A300-600_AC_2.1_weights.txt`. Chapter 3.5.1 'Landing Approach Speed At 1.3 Vs' is a graph; **it was not read for this pack**, so there is no manufacturer approach speed for A306 here.

### embraer_e175_apm

**Embraer 175 Airport Planning Manual** - Embraer  
Document: APM-2259, May 25/18 pages (copy filed as NTSB docket DCA20IA014 Attachment 2)  
URL: https://data.ntsb.gov/Docket/Document/docBLOB?ID=13691419&FileExtension=pdf&FileName=DCA20IA014+Attachment+2+-+Embraer+175+Airport+Planning+Manual-Rel.pdf  
Retrieved: 2026-09-07  
Local path: `data/reference_speeds/embraer/E175_APM_NTSB_DCA20IA014_Att2.pdf` (2,019,209 bytes)  
sha256: `e1df5f7db5849b9f6e6e7800b3f8e61f2321692be25d0a676c09de966daf2c8c`

Read: Table 2.1 Aircraft General Characteristics, PDF page 18: MLW 34,000 kg (74,957 lb) for STD/LR/LL and 34,100 kg for AR; BOW 21,500 kg (47,399 lb) for STD/LR/LL and 22,500 kg for AR. Excerpt `excerpts/embraer_E175_table_2.1.txt`. **No approach speed is published.** The copy is the one filed in the NTSB public docket for DCA20IA014; it is a manufacturer document reproduced by a government docket, not an Embraer-hosted download.

### bombardier_crj900_apm_r11

**CRJ900 Airport Planning Manual** - Bombardier  
Document: CSP C-020 Rev 11  
URL: https://customer.aero.bombardier.com/webd/BAG/CustSite/BRAD/RACSDocument.nsf/51aae8b2b3bfdf6685256c300045ff31/ec63f8639ff3ab9d85257c1500635bd8/$FILE/ATTQF1EY.pdf/CRJ900APMR11.pdf  
Retrieved: 2026-09-07  
Local path: `data/reference_speeds/bombardier/CRJ900APMR11.pdf` (3,406,885 bytes)  
sha256: `987d2962bd5ac61714a631aef93dd09b582301c25821c65266b81701b9c23ef8`

Read: Page 00-02-01 (PDF page 22): MTLW 73,500 lb (33,340 kg), Minimum Flight Weight 45,000 lb (20,412 kg). Section 00-03-03 Figure 3 (PDF page 118, serials 15001-15035/15038-15039/15042) and Figure 4 (PDF page 120, remaining serials), 'Landing Speed - VREF (KIAS), flaps 45 degrees/slats extended' versus gross weight - **read graphically**, see `excerpts/bombardier_CRJ900_fig3_reading.md` for the calibration and the result. Excerpt of the weights page: `excerpts/bombardier_CRJ900_00-02-01.txt`.

### bombardier_crj200_apm_r8

**CRJ100/200/440 Airport Planning Manual** - Bombardier  
Document: CSP A-020 Rev 8, Jan 10/2016  
URL: https://customer.aero.bombardier.com/webd/BAG/CustSite/BRAD/RACSDocument.nsf/51aae8b2b3bfdf6685256c300045ff31/ec63f8639ff3ab9d85257c1500635bd8/$FILE/ATT1ES4H.pdf/CRJ200APMR8.pdf  
Retrieved: 2026-09-07  
Local path: `data/reference_speeds/bombardier/CRJ200APMR8.pdf` (3,294,114 bytes)  
sha256: `db533e0c559e50b2ff940df11421811e3523ced7b3060dcd80acd797348ccca5`

Read: Page 00-02-01 Table 1 Aircraft Characteristics: MLW 44,700 lb (20,276 kg) for CRJ100/200 and 47,000 lb (21,319 kg) for the ER/LR variants, Operating Empty Weight 30,500 lb (13,835 kg) for every variant. Excerpt `excerpts/bombardier_CRJ200_00-02-01.txt`. **No approach speed is published.**

### bombardier_global5000_factsheet

**Global 5000 Fact Sheet** - Bombardier  
Document: served as 2018-10/Global_5000_Fact_Sheet_0.pdf  
URL: https://bombardier.com/en/media/2531/download  
Retrieved: 2026-09-07  
Local path: `data/reference_speeds/bombardier/Global_5000_Fact_Sheet.pdf` (775,273 bytes)  
sha256: `ce6f73149633d4036398da600015f2c60df7ff992ea401d027208bfa9148da5d`

Read: Weights block, page 2: Maximum landing weight 78,600 lb (35,652 kg), Basic operating weight 50,861 lb (23,070 kg). Excerpt `excerpts/bombardier_Global5000_factsheet_weights.txt`. **No approach speed is published.** The URL is Bombardier's media redirector; it serves the file `2018-10/Global_5000_Fact_Sheet_0.pdf`.

### pilatus_pc24_factsheet

**PC-24 - The Super Versatile Jet, Factsheet** - Pilatus Aircraft Ltd  
Document: doc code PIL|0823A (August 2023)  
URL: https://www.pilatus-aircraft.com/assets/files/Brochures/PC-24/Pilatus-Aircraft-Ltd-PC-24-Factsheet.pdf  
Retrieved: 2026-09-07  
Local path: `data/reference_speeds/pilatus/Pilatus-Aircraft-Ltd-PC-24-Factsheet.pdf` (1,338,906 bytes)  
sha256: `bd33088166a640dab3e4fe0df20ddfd06f952d8d5b5d57d0d6ad4f944a468280`

Read: WEIGHTS block: Maximum landing weight 17,340 lb (7,865 kg), Basic operating weight 11,561 lb (5,244 kg, executive configuration 6 seat incl. one pilot). Excerpt `excerpts/pilatus_PC24_factsheet_weights.txt`. **No approach speed is published** (the sheet gives a landing-configuration stall speed of 83 KIAS at MLW).

### textron_citation_ascend_product_card

**Cessna Citation Ascend product card (Model 560XL)** - Textron Aviation  
Document: undated product card PDF  
URL: https://cessna.txtav.com/-/media/cessna/files/product-cards/citation/citation_ascend_product_card.pdf  
Retrieved: 2026-09-07  
Local path: `data/reference_speeds/textron/citation_ascend_product_card.pdf` (266,092 bytes)  
sha256: `e32eef00e038e6c38e7b66f1a9ffd45222ff0614edb22f03faf475f6ea4f8ec3`

Read: WEIGHTS block: Max Takeoff Weight 20,500 lb, Basic Operating Weight 13,060 lb (5,924 kg). The matching model page `https://cessna.txtav.com/en/citation/xls-gen2` additionally lists Maximum Landing Weight 18,700 lb (8,482 kg) - the same value as the FAA MALW. Excerpt `excerpts/textron_C56X_ascend_product_card_weights.txt`. **No approach speed is published.**

### textron_skyhawk_product_card

**Cessna Skyhawk product card (Model 172S)** - Textron Aviation  
Document: undated product card PDF  
URL: https://cessna.txtav.com/-/media/cessna/files/product-cards/piston/skyhawk_product_card.pdf  
Retrieved: 2026-09-07  
Local path: `data/reference_speeds/textron/skyhawk_product_card.pdf` (146,388 bytes)  
sha256: `ff33311f8102476896ebeef76f402b19999ee736ea968e02b1d6f6e62b58fa50`

Read: WEIGHTS block: Max Takeoff Weight 2,550 lb, Basic Empty Weight 1,680 lb (762 kg), Useful Load 878 lb. The matching model page `https://cessna.txtav.com/en/piston/cessna-skyhawk` lists Maximum Landing Weight 2,550 lb. Excerpt `excerpts/textron_C172_skyhawk_product_card_weights.txt`. **No approach speed is published.**

### openap_2_4

**OpenAP aircraft property database** - OpenAP (TU Delft)  
Document: openap 2.4, package data files openap/data/aircraft/<type>.yml  
URL: https://github.com/TUDelft-CNS-ATM/openap  
Retrieved: 2026-09-07  
Local path: `conda env aeroviz: site-packages/openap/data/aircraft/<type>.yml (not copied into data/)`  
sha256: `per file, see docs/reference_speeds/excerpts/openap_2_4_oew.csv`

Read: Per-type YAML property files, field `oew` (also `mlw`, `mtow` recorded for context). Used ONLY as the last-resort minimum-mass fallback, for the 19 types whose manufacturer does not publish an empty weight in any document reachable here. The files live in the conda env, not under `data/`; every value with the file's sha256 is in `excerpts/openap_2_4_oew.csv`, which is what makes the number reproducible. OpenAP is a research database, not a manufacturer publication - rows sourced to it are marked as such in the table below.

## Per-type table

`rows` is the count of observed arrival records of that type in the five-airport cohort.
Speeds are knots IAS; the FAA triple is main / min / max (min and max equal the main value when the
FAA publishes a single figure). `EC Vat` is the EUROCONTROL Aircraft Performance Database landing Vat.
The `manufacturer` column names the document and the section it was read from; the full sentence for
every entry is the `corroboration[].note` of that type in the JSON.

| type | rows | FAA speed kt main/min/max | FAA MALW lb | malw_kg | manufacturer corroboration (speed @ MLW, doc, section) | EC Vat kt | minimum mass kg | kind | min-mass source | notes |
|---|---:|---|---:|---:|---|---:|---:|---|---|---|
| B38M | 5864 | 145 / 140 / 145 | 152,800 | 69,309 | no speed published; MLW 69,308 kg (same weight as the FAA MALW, -1 kg conversion rounding) -- `boeing_737max_acap_rev_k` 2.1.2 (737-8) | 145 | **45,000** | OEW | `openap_2_4` | min mass = OpenAP fallback; dual FSB value |
| B737 | 5537 | 130 / 130 / 130 | 145,600 | 66,043 | no speed published; MLW 58,604 kg (FAA MALW is 7,439 kg higher) -- `boeing_737ng_acap_rev_c` 2.1.2 (737-700) | 137 | **37,648** | OEW | `boeing_737ng_acap_rev_c` |  |
| B738 | 4963 | 144 / 140 / 144 | 146,275 | 66,349 | no speed published; MLW 66,360 kg (FAA MALW is 11 kg lower) -- `boeing_737ng_acap_rev_c` 2.1.3 (737-800) | 147 | **41,412** | OEW | `boeing_737ng_acap_rev_c` | dual FSB value |
| E75L | 3401 | 126 / 126 / 126 | 74,957 | 34,000 | no speed published; MLW 34,000 kg (same weight as the FAA MALW, +0 kg conversion rounding) -- `embraer_e175_apm` Table 2.1 | - | **21,500** | BOW | `embraer_e175_apm` | no EUROCONTROL entry |
| B739 | 2249 | 149 / 140 / 149 | 71400 (rejected) | 71,350 | no speed published; MLW 71,350 kg (this is the value carried as `malw_kg`) -- `boeing_737ng_acap_rev_c` 2.1.5 (737-900ER) | 150 | **42,900** | OEW | `boeing_737ng_acap_rev_c` | **FAA MALW cell is wrong**, Boeing value used; dual FSB value |
| A319 | 1895 | 126 / 126 / 126 | 134,482 | 61,000 | **126 kt** @ MLW 62,500 kg -- `airbus_ac_a319_0624` 3-5-0 | 130 | **40,800** | OEW | `openap_2_4` | min mass = OpenAP fallback |
| CRJ9 | 1650 | 141 / 132 / 141 | 73,500 | 33,339 | **139.5 kt** @ MLW 33,340 kg -- `bombardier_crj900_apm_r11` 00-03-03 Fig 3 (graph) | 135 | **20,412** | MFW | `bombardier_crj900_apm_r11` | dual FSB value |
| A321 | 1202 | 142 / 140 / 142 | 171,520 | 77,800 | **142 kt** @ MLW 77,800 kg -- `airbus_ac_a321_1223` 3-5-0 | 141 | **48,500** | OEW | `openap_2_4` | min mass = OpenAP fallback; dual FSB value |
| A320 | 1000 | 136 / 136 / 136 | 145,505 | 66,000 | **136 kt** @ MLW 66,000 kg -- `airbus_ac_a320_0624` 3-5-0 | 137 | **42,600** | OEW | `openap_2_4` | min mass = OpenAP fallback |
| A21N | 881 | 136 / 136 / 136 | 174,606 | 79,200 | **136 kt** @ MLW 79,200 kg -- `airbus_ac_a321_1223` 3-5-0 (A321neo) | 140 | **50,000** | OEW | `openap_2_4` | min mass = OpenAP fallback |
| B39M | 713 | 150 / 140 / 150 | 163,900 | 74,344 | no speed published; MLW 74,343 kg (same weight as the FAA MALW, -1 kg conversion rounding) -- `boeing_737max_acap_rev_k` 2.1.5 (737-9) | 150 | **45,000** | OEW | `openap_2_4` | min mass = OpenAP fallback; dual FSB value |
| A20N | 535 | 137 / 137 / 137 | 148,591 | 67,400 | **131.5 kt** @ MLW 67,400 kg -- `airbus_ac_a320_0624` 3-5-0 (A320neo) | 135 | **44,300** | OEW | `openap_2_4` | min mass = OpenAP fallback |
| C56X | 314 | 116 / 116 / 116 | 18,700 | 8,482 | no speed published; MLW 8,482 kg (same weight as the FAA MALW, +0 kg conversion rounding) -- `textron_citation_ascend_product_card` product card + model page | 117 | **5,924** | BOW | `textron_citation_ascend_product_card` |  |
| B763 | 239 | 140 / 140 / 140 | 320,000 | 145,150 | no speed published; MLW 145,149 kg (same weight as the FAA MALW, -1 kg conversion rounding) -- `boeing_767_acap_rev_k` 2.1.4 (767-300ER) | 140 | **84,540** | OEW | `boeing_767_acap_rev_k` |  |
| B752 | 178 | 137 / 137 / 137 | 198,000 | 89,811 | no speed published; MLW 89,811 kg (same weight as the FAA MALW, +0 kg conversion rounding) -- `boeing_757_acap_rev_h` 2.1.1 (757-200) | 130 | **56,748** | OEW | `boeing_757_acap_rev_h` |  |
| E170 | 175 | 124 / 124 / 124 | 73,413 | 33,300 | none published | 130 | **21,140** | OEW | `openap_2_4` | min mass = OpenAP fallback |
| E190 | 154 | 124 / 124 / 124 | 94,799 | 43,000 | none published | 131 | **27,753** | OEW | `openap_2_4` | min mass = OpenAP fallback |
| GLF5 | 147 | 136 / 136 / 136 | 75,300 | 34,156 | none published | 140 | **null** | - | none | see 'no published minimum mass' below |
| GLF6 | 141 | 137 / 137 / 137 | 83,500 | 37,875 | none published | - | **24,000** | OEW | `openap_2_4` | min mass = OpenAP fallback; no EUROCONTROL entry |
| A306 | 126 | 137 / 137 / 137 | 304,230 | 137,996 | no speed published; MLW 138,000 kg (same weight as the FAA MALW, +4 kg conversion rounding) -- `airbus_ac_a300_600_dec2009` chapter 2.1 | 139 | **86,727** | OEW | `airbus_ac_a300_600_dec2009` |  |
| C25A | 119 | 114 / 114 / 114 | 11,525 | 5,228 | none published | 110 | **null** | - | none | see 'no published minimum mass' below |
| LJ45 | 103 | 123 / 123 / 123 | 19,200 | 8,709 | none published | 140 | **null** | - | none | see 'no published minimum mass' below |
| PC24 | 75 | 107 / 107 / 107 | 16,900 | 7,666 | no speed published; MLW 7,865 kg (FAA MALW is 199 kg lower) -- `pilatus_pc24_factsheet` factsheet WEIGHTS | - | **5,244** | BOW | `pilatus_pc24_factsheet` | no EUROCONTROL entry |
| A333 | 71 | 137 / 137 / 137 | 412,264 | 187,000 | none published | 140 | **122,780** | OEW | `openap_2_4` | min mass = OpenAP fallback |
| C172 | 63 | 62 / 62 / 62 | 2,450 | 1,111 | no speed published; MLW 1,157 kg (FAA MALW is 46 kg lower) -- `textron_skyhawk_product_card` product card + model page | 65 | **762** | BEW | `textron_skyhawk_product_card` |  |
| B772 | 62 | 140 / 140 / 140 | 470,000 | 213,188 | no speed published; MLW 208,652 kg (FAA MALW is 4,536 kg higher) -- `boeing_777_acap_rev_e` 2.1.1 (777-200 HGW) | 140 | **133,084** | OEW | `boeing_777_acap_rev_e` |  |
| B788 | 60 | 144 / 140 / 144 | 380,000 | 172,365 | no speed published; MLW 172,365 kg (same weight as the FAA MALW, +0 kg conversion rounding) -- `boeing_787_acap_rev_q` 2.1.1 (787-8) | 140 | **119,000** | OEW | `openap_2_4` | min mass = OpenAP fallback; dual FSB value |
| CRJ2 | 43 | 141 / 141 / 141 | 44,700 | 20,276 | no speed published; MLW 21,319 kg (FAA MALW is 1,043 kg lower) -- `bombardier_crj200_apm_r8` 00-02-01 Table 1 | 140 | **13,835** | OWE | `bombardier_crj200_apm_r8` |  |
| C525 | 35 | 108 / 108 / 108 | 9,900 | 4,491 | none published | 110 | **null** | - | none | see 'no published minimum mass' below |
| GL5T | 35 | 128 / 128 / 128 | 78,600 | 35,652 | no speed published; MLW 35,652 kg (same weight as the FAA MALW, +0 kg conversion rounding) -- `bombardier_global5000_factsheet` fact sheet Weights | 122 | **23,070** | BOW | `bombardier_global5000_factsheet` |  |
| B789 | 33 | 144 / 140 / 144 | 425,000 | 192,777 | no speed published; MLW 192,776 kg (same weight as the FAA MALW, -1 kg conversion rounding) -- `boeing_787_acap_rev_q` 2.1.2 (787-9) | 150 | **128,000** | OEW | `openap_2_4` | min mass = OpenAP fallback; dual FSB value |
| C550 | 33 | 105 / 105 / 105 | 13,500 | 6,123 | none published | 110 | **3,655** | OEW | `openap_2_4` | min mass = OpenAP fallback |
| A332 | 27 | 136 / 136 / 136 | 401,241 | 182,000 | none published | 140 | **120,200** | OEW | `openap_2_4` | min mass = OpenAP fallback |
| A359 | 26 | 140 / 140 / 140 | 456,357 | 207,000 | none published | 140 | **142,400** | OEW | `openap_2_4` | min mass = OpenAP fallback |
| B734 | 25 | 139 / 139 / 139 | 123,700 | 56,109 | no speed published; MLW 56,245 kg (FAA MALW is 136 kg lower) -- `boeing_737cl_acap_rev_e` 2.1.7 (737-400) | 139 | **33,189** | OEW | `boeing_737cl_acap_rev_e` |  |
| B762 | 10 | 135 / 135 / 135 | 260,000 | 117,934 | no speed published; MLW 123,377 kg (FAA MALW is 5,443 kg lower) -- `boeing_767_acap_rev_k` 2.1.1 (767-200) | 135 | **78,974** | OEW | `boeing_767_acap_rev_k` |  |
| E145 | 5 | 124 / 124 / 124 | 41,226 | 18,700 | none published | 135 | **12,110** | OEW | `openap_2_4` | min mass = OpenAP fallback |
| A343 | 1 | 145 / 145 / 145 | 423,288 | 192,000 | none published | 150 | **130,000** | OEW | `openap_2_4` | min mass = OpenAP fallback |
| B77W | 1 | 149 / 149 / 149 | 554,000 | 251,290 | no speed published; MLW 251,290 kg (same weight as the FAA MALW, +0 kg conversion rounding) -- `boeing_777lr_300er_f_acap_rev_g` 2.1.1 (777-300ER) | 149 | **167,800** | OEW | `openap_2_4` | min mass = OpenAP fallback |
| B735 | 1 | 128 / 128 / 128 | 110,000 | 49,895 | no speed published; MLW 49,895 kg (same weight as the FAA MALW, +0 kg conversion rounding) -- `boeing_737cl_acap_rev_e` 2.1.8 (737-500) | 128 | **31,311** | OEW | `boeing_737cl_acap_rev_e` |  |

## Types with no published minimum mass

Four types carry `min_mass_kg: null`. Nothing is guessed for them; what was tried is recorded here.

* **GLF5** (147 rows) - no published figure found. gulfstream.com no longer serves a Gulfstream V / G550 aircraft page (HTTP 403 AccessDenied, checked 2026-09-07), Gulfstream airport planning manuals sit behind the MyGulfstream login, and OpenAP 2.4 has no glf5.yml. Third-party compilations quote 48,300 lb but they are not manufacturer publications, so nothing is recorded here
* **C25A** (119 rows) - no published figure found. Textron Aviation has retired the Citation CJ2 / Model 525A product page (cessna.txtav.com/en/citation/cj2 returns HTTP 404, checked 2026-09-07) and OpenAP 2.4 has no c25a.yml
* **LJ45** (103 rows) - no published figure found. Bombardier's public document index (RACSDocument.nsf) carries only CRJ and Dash 8 manuals, there is no Learjet 45 airport planning manual or spec sheet on a Bombardier domain, and OpenAP 2.4 has no lj45.yml
* **C525** (35 rows) - no published figure found. Textron Aviation has retired the CitationJet / CJ1 (Model 525) product page (cessna.txtav.com/en/citation/cj1 returns HTTP 404, checked 2026-09-07) and OpenAP 2.4 has no c525.yml

## Where the three sources disagree by more than 5 kt

The FAA figure is the one the JSON carries; these rows are the ones where a reader should not treat
the number as settled. All values in knots IAS at the respective maximum landing weight.

| type | FAA | other source | value | difference | remark |
|---|---:|---|---:|---:|---|
| B737 | 130 | `eurocontrol_apd` | 137 | -7 | the FAA MALW for B737 (145,600 lb) also matches no published Boeing 737-700 or BBJ landing weight, so this row is doubly suspect |
| CRJ9 | 141 | `eurocontrol_apd` | 135 | +6 | the FAA value is the AAC-D Long Range figure; the CRJ900 APM graph reading (139.5 kt at 33,340 kg, flaps 45) sits between the FAA 141 kt and the EUROCONTROL 135 kt |
| A20N | 137 | `airbus_ac_a320_0624` | 131.5 | +5.5 | Airbus publishes 131.5 kt at MLW 67,400 kg for the A320neo and the FAA MALW is the same weight, so the 5.5 kt gap is a genuine source disagreement, not a weight difference |
| B752 | 137 | `eurocontrol_apd` | 130 | +7 | EUROCONTROL's B752 MTOW (115,680 kg) is the high-gross-weight 757-200; its Vat is nevertheless 7 kt below the FAA figure |
| E170 | 124 | `eurocontrol_apd` | 130 | -6 | EUROCONTROL's E170 MTOW (35,995 kg) is a much lighter variant than the FAA row (85,098 lb = 38,600 kg), which explains part of the gap |
| E190 | 124 | `eurocontrol_apd` | 131 | -7 | EUROCONTROL's E190 MTOW (45,995 kg) is a lighter variant than the FAA row (110,892 lb = 50,300 kg) |
| LJ45 | 123 | `eurocontrol_apd` | 140 | -17 | the largest disagreement in the fleet. EUROCONTROL lists Vat 140 kt / APC C at MTOW 9,230 kg; the FAA lists 123 kt at MALW 19,200 lb. No manufacturer document was reachable to break the tie |
| GL5T | 128 | `eurocontrol_apd` | 122 | +6 | EUROCONTROL lists Vat 122 kt at MTOW 41,957 kg, which is the same MTOW as the Bombardier fact sheet, so the 6 kt gap is a source disagreement |
| B789 | 144 | `eurocontrol_apd` | 150 | -6 | EUROCONTROL lists 150 kt; the FAA gives 144 kt max / 140 kt min depending on configuration and performance package |
| E145 | 124 | `eurocontrol_apd` | 135 | -11 | EUROCONTROL lists Vat 135 kt at MTOW 21,198 kg against the FAA 124 kt at MALW 41,226 lb (18,700 kg) |

## Regenerating this pack

```bash
bash docs/reference_speeds/fetch_sources.sh          # re-download every remote file into data/, print sha256
```

The JSON and this README were written from the documents by hand-checked extraction; there is no
build step that reads the PDFs on every run. If a source document is revised, re-read the section
named in its **Read:** line above and update both files together.

