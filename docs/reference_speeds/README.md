# Reference approach speeds and weights - provenance pack

Generated 2026-09-07. Machine-readable table: [`aircraft/reference_speeds.json`](../../aircraft/reference_speeds.json)
(schema `aircraft-reference-speeds-v1`). Verbatim backing excerpts: [`excerpts/`](excerpts/).

This pack exists so that every approach speed and every mass used by the evaluation gate can be
traced, in one hop, to a page of a published document that is on this disk.

**172 types.** The pack was built in two passes. The first covered the **40 types the optimizer
already has dynamics for** (the table under [Per-type table](#per-type-table)). The second, on the
same day, added the **132 types the observed cohort flies but has no dynamics for** - 8,919 arrival
records - so that the speed gate has a published anchor for them too; those are the table under
[Types added for the no-dynamics cohort](#types-added-for-the-no-dynamics-cohort). Every added row's
speed and (with one stated exception) landing weight come from the same FAA spreadsheet as the
first 40; 27 of them also carry a published minimum operating mass.

## What is and is not in git

* **Everything under `data/reference_speeds/` is git-ignored** (`data/` is not tracked). The PDFs,
  the FAA spreadsheet, the 41 EUROCONTROL HTML pages and the five manufacturer product pages saved
  as HTML (three Bombardier, two Cirrus) are local artifacts.
* **`fetch_sources.sh` in this directory re-downloads all of them** to the same paths, with a browser
  User-Agent, printing the sha256 of each file. It **never overwrites an existing file**, so running
  it on a populated tree only prints checksums you can compare against the tables below.
* This README, the `excerpts/` directory and `aircraft/reference_speeds.json` **are** tracked, so the
  numbers and the quotes that justify them survive without the source documents.
* **Five sources are web pages, not files** (`bombardier_challenger3500_specs`,
  `bombardier_challenger650_specs`, `bombardier_global6500_specs`, `cirrus_sr_series_specs`,
  `cirrus_vision_jet_specs`). Their makers publish the weight block only on the product page - the
  downloadable brochures print none - so the page is what was saved, and its sha256 will change the
  next time the site is touched even if the numbers do not. The excerpt file is what pins the numbers.
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
* `min_mass_kind` is the manufacturer's own term for the figure, abbreviated: `MFW` minimum flight
  weight, `OEW` operating empty weight, `OWE` operational weight empty, `BOW` basic operating
  weight, `BEW` basic empty weight, `EW` empty weight, `SEW` standard empty weight (Piper),
  `BW` base weight (Cirrus). The term is never normalised away, because the definitions differ:
  an MFW is a certified floor, a BOW includes crew and standard items, an EW does not.
* `null` means nothing published was found. **109 of the 172 types are in that state** - 4 of the
  original 40 and 105 of the 132 added ones - and none of them is guessed. What was tried for each
  is written down under [Types with no published minimum mass](#types-with-no-published-minimum-mass).
* **EUROCONTROL corroboration exists only for the original 40 types.** The 132 added rows carry no
  `eurocontrol_apd` entry; the database was not queried for them, and their tables print no EC Vat
  column rather than an empty one.
* **OpenAP is not a fallback for any added type.** OpenAP 2.4 ships 38 aircraft files and **not one**
  of the 134 no-dynamics designators is among them (checked 2026-09-07 with
  `prop.aircraft(code, use_synonym=False)`), so for the added rows the ladder is manufacturer
  document or `null`, with no middle rung.

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

### airbus_a220_acp_issue013

**A220 Aircraft Characteristics Publication (ACP)** - Airbus  
Document: Issue 013-00, 27 November 2025 (BD500-3AB48-13800-00)  
URL: https://www.aircraft.airbus.com/sites/g/files/jlcbta126/files/2025-12/A220-ACP-Issue013-00-27Nov2025.pdf  
Retrieved: 2026-09-07  
Local path: `data/reference_speeds/airbus/A220-ACP-Issue013-00-27Nov2025.pdf` (32,733,365 bytes)  
sha256: `ac489084ac6fd0b68f7d8df65bde42f01644442dc6069b9c0562b42bc7ae678b`

Read: Data module BD500-A-J00-00-00-12AAA-030A-A, PDF page 73 (A220-100, BD-500-1A10) and BD500-A-J00-00-00-12AAB-030A-A, PDF page 109 (A220-300, BD-500-1A11), 'Table 2 Aircraft characteristics'. Taken: Maximum Landing Weight and **Minimum Flight Weight (MFW)** per model - A220-100 77,000 lb (34,927 kg) for both of its weight variants, A220-300 80,000 lb (36,287 kg). The Airbus-inherited overview table on PDF page 30 also prints an Operating Weight Empty (A220-100 77,650 lb / 35,221 kg, A220-300 81,750 lb / 37,081 kg); the MFW is the lower, and the pack's preferred, figure. Excerpt `excerpts/airbus_A220_ACP_aircraft_characteristics.txt`. **No approach speed is published.** This is the only Airbus document in the pack that publishes a minimum flight weight - it is a Bombardier-heritage data-module publication, not an Airbus AC document.

### bombardier_crj700_apm_r15

**CRJ700 Airport Planning Manual** - Bombardier  
Document: CSP B-020 Rev 15 (page 00-02-01 dated May 20/2010)  
URL: https://customer.aero.bombardier.com/webd/BAG/CustSite/BRAD/RACSDocument.nsf/51aae8b2b3bfdf6685256c300045ff31/ec63f8639ff3ab9d85257c1500635bd8/$FILE/ATTE8Q23.pdf/CRJ700APMR15.pdf  
Retrieved: 2026-09-07  
Local path: `data/reference_speeds/bombardier/CRJ700APMR15.pdf` (4,432,271 bytes)  
sha256: `195e92c0b358f347342a51d4f084a13f721fc099015b82439f7acd22b7ad45ce`

Read: Page 00-02-01 (PDF page 26), Model CL-600-2C10: Maximum Landing Weight (MTLW) 67,000 lb (30,390 kg) - the same weight as the FAA MALW - and Minimum Flight Weight (MFW) 42,000 lb (19,051 kg). Excerpt `excerpts/bombardier_CRJ700_00-02-01.txt`. **No approach speed is published in this revision** (the CRJ900 APM's VREF graph has no CRJ700 counterpart here).

### boeing_717_acap_rev_b

**717-200 Airplane Characteristics for Airport Planning** - Boeing  
Document: D6-58330 Rev B, November 2014  
URL: https://www.boeing.com/content/dam/boeing/v2/airports/acaps/717.pdf  
Retrieved: 2026-09-07  
Local path: `data/reference_speeds/boeing/717.pdf` (11,060,519 bytes)  
sha256: `5d7310882b997826ec78edc442ce766466710e0870984608dd14860886efa093`

Read: Section 2.1 General Characteristics, Model 717-200, printed page 7 = PDF page 11. Taken: MAX DESIGN LANDING WEIGHT (100,000 lb for the lowest basic weight variant, 110,000 lb with the high-gross-weight option) and SPEC OPERATING EMPTY WEIGHT (67,500 lb / 30,617 kg basic, 68,500 lb with the HGW option). Excerpt `excerpts/boeing_717_2.1_weights.txt`. **No approach speed is published.** **This PDF's text layer is a broken font subset** - `pdftotext` returns mojibake for every body font - so the table was read from the rendered page image, cell by cell; the excerpt says so and reproduces the two printed notes.

### bombardier_challenger3500_specs

**Challenger 3500 product page, Specifications block** - Bombardier  
Document: product page (HTML) as served 2026-09-07  
URL: https://bombardier.com/en/aircraft/challenger-3500  
Retrieved: 2026-09-07  
Local path: `data/reference_speeds/bombardier/bombardier_challenger-3500.html` (309,754 bytes)  
sha256: `ca1afcade8cd0ce0be3540769a29771b17425491f8102b80e1d27fda1b18d359`

Read: Specifications -> Weights block of the product page: Maximum landing weight 34,150 lb (15,490 kg), the same weight as the FAA MALW, and Basic operating weight 24,800 lb (11,249 kg). Excerpt `excerpts/bombardier_challenger3500_specs_weights.txt`. **No approach speed is published.** Bombardier's current fact sheets are served only after a request form (bombardier.com/en/aircraft/challenger-3500 offers 'DOWNLOAD' behind an e-mail form, and bombardier.com/en/media/media-library/... returns HTTP 403), so the product page - which prints the whole Weights block in the HTML - is the reachable published source. The Challenger 3500 is the current build of the Challenger 350 that carries the CL35 designator.

### bombardier_challenger650_specs

**Challenger 650 product page, Specifications block** - Bombardier  
Document: product page (HTML) as served 2026-09-07  
URL: https://bombardier.com/en/aircraft/challenger-650  
Retrieved: 2026-09-07  
Local path: `data/reference_speeds/bombardier/bombardier_challenger-650.html` (287,767 bytes)  
sha256: `fcc96c7b5012771265148e7956e2cb321b548d1a2ccc290f6302da59ce36909b`

Read: Specifications -> Weights block: Maximum landing weight 38,000 lb (17,237 kg), the same weight as the FAA MALW, and Basic operating weight 27,150 lb (12,315 kg). Excerpt `excerpts/bombardier_challenger650_specs_weights.txt`. **No approach speed is published.** Same 403/form situation as the Challenger 3500 above.

### bombardier_global6500_specs

**Global 6500 product page, Specifications block** - Bombardier  
Document: product page (HTML) as served 2026-09-07  
URL: https://bombardier.com/en/aircraft/global-6500  
Retrieved: 2026-09-07  
Local path: `data/reference_speeds/bombardier/bombardier_global-6500.html` (315,146 bytes)  
sha256: `f4e3b62cfe35759a9fb745781150259aa1f058f938a48beb96c27e403846d886`

Read: Specifications -> Weights block: Maximum landing weight 78,600 lb (35,652 kg), the same weight as the FAA MALW, and Basic operating weight 52,230 lb (23,691 kg). Excerpt `excerpts/bombardier_global6500_specs_weights.txt`. **No approach speed is published.** The Global 6500 is the current build of the BD-700-1A10 family that carries the GLEX designator; the Global 5500 page on the same site prints the identical MLW and a Basic operating weight of 50,861 lb, the value the existing GL5T row already carries from the Global 5000 fact sheet.

### pilatus_pc12_factsheet

**PC-12 PRO - Just the Facts** - Pilatus Aircraft Ltd  
Document: 2025 edition (file PC-12-PRO-Just-The-Facts-2025-web.pdf)  
URL: https://www.pilatus-aircraft.com/assets/files/PC-12-PRO-Just-The-Facts-2025-web.pdf  
Retrieved: 2026-09-07  
Local path: `data/reference_speeds/pilatus/PC-12-PRO-Just-The-Facts-2025-web.pdf` (8,477,502 bytes)  
sha256: `e19a79473f05458cd83b71807d7b8ddebb8edb557b578086770f311d8f990203`

Read: WEIGHTS block: Maximum landing weight 9,921 lb (4,500 kg) - the same weight as the FAA MALW - and Basic operating weight 6,703 lb (3,040 kg). Excerpt `excerpts/pilatus_PC12_factsheet_weights.txt`. **No approach speed is published** (the sheet gives a stall speed of 67 KIAS). The PC-12 page's own technical-data tab renders its numbers from JavaScript and yields nothing to a plain fetch; the factsheet is the static published file.

### textron_citation_latitude_product_card

**Cessna Citation Latitude product card (Model 680A)** - Textron Aviation  
Document: undated product card PDF  
URL: https://cessna.txtav.com/-/media/cessna/files/product-cards/citation/citation_latitude_product_card.pdf  
Retrieved: 2026-09-07  
Local path: `data/reference_speeds/textron/citation_latitude_product_card.pdf` (561,474 bytes)  
sha256: `377b8319cd88ed04c02b3a0f27371905f8a68ee2456ed67468ac97908c8972be`

Read: WEIGHTS block: Max Takeoff Weight 30,800 lb (the same as the FAA MTOW), Basic Operating Weight 18,656 lb (8,462 kg). Excerpt `excerpts/textron_C68A_latitude_product_card_weights.txt`. **No approach speed and no landing weight are published on the card.**

### textron_citation_longitude_product_card

**Cessna Citation Longitude product card (Model 700)** - Textron Aviation  
Document: undated product card PDF  
URL: https://cessna.txtav.com/-/media/cessna/files/product-cards/citation/citation_longitude_product_card.pdf  
Retrieved: 2026-09-07  
Local path: `data/reference_speeds/textron/citation_longitude_product_card.pdf` (203,484 bytes)  
sha256: `a1e9d2e42cfddfb2232d58f8218d275d4af15d5d54ecfd6c483f1143e22d5b6c`

Read: WEIGHTS block: Max Takeoff Weight 39,500 lb (the same as the FAA MTOW), Basic Operating Weight 23,600 lb (10,705 kg). Excerpt `excerpts/textron_C700_longitude_product_card_weights.txt`. **No approach speed and no landing weight are published.**

### textron_citation_cj3_gen3_product_card

**Cessna Citation CJ3 Gen3 product card (Model 525B)** - Textron Aviation  
Document: undated product card PDF  
URL: https://cessna.txtav.com/-/media/cessna/files/product-cards/citation/citation_cj3_gen3_product_card.pdf  
Retrieved: 2026-09-07  
Local path: `data/reference_speeds/textron/citation_cj3_gen3_product_card.pdf` (233,697 bytes)  
sha256: `c9f1fa1e73ee68e46366c3eed33c34dd5fdb70be8625f8d7c0b72d1dd56780e7`

Read: WEIGHTS block: Max Takeoff Weight 13,870 lb (the same as the FAA MTOW), Empty Weight 8,540 lb (3,874 kg). Excerpt `excerpts/textron_C25B_cj3gen3_product_card_weights.txt`. **No approach speed and no landing weight are published.** The card is for the CJ3 Gen3, the current build of the Model 525B that carries the C25B designator.

### textron_citation_cj4_gen3_product_card

**Cessna Citation CJ4 Gen3 product card (Model 525C)** - Textron Aviation  
Document: undated product card PDF  
URL: https://cessna.txtav.com/-/media/cessna/files/product-cards/citation/citation_cj4_gen3_product_card.pdf  
Retrieved: 2026-09-07  
Local path: `data/reference_speeds/textron/citation_cj4_gen3_product_card.pdf` (371,501 bytes)  
sha256: `9c67bf6a98de659f97841170aff6fd55a98afbeb4369ed58eb4501146a436f30`

Read: WEIGHTS block: Max Takeoff Weight 17,110 lb (the same as the FAA MTOW), Basic Operating Weight 10,300 lb (4,672 kg). Excerpt `excerpts/textron_C25C_cj4gen3_product_card_weights.txt`. **No approach speed and no landing weight are published.**

### textron_citation_m2_gen3_product_card

**Cessna Citation M2 Gen3 product card (Model 525)** - Textron Aviation  
Document: undated product card PDF  
URL: https://cessna.txtav.com/-/media/cessna/files/product-cards/citation/citation_m2_gen3_product_card.pdf  
Retrieved: 2026-09-07  
Local path: `data/reference_speeds/textron/citation_m2_gen3_product_card.pdf` (608,466 bytes)  
sha256: `0ae9432d51f63be557e0e035a7589c1ab951cd79bbc045246ebe92889c96496f`

Read: WEIGHTS block: Max Takeoff Weight 10,700 lb (the same as the FAA MTOW), Empty Weight 6,990 lb (3,171 kg). Excerpt `excerpts/textron_C25M_m2gen3_product_card_weights.txt`. **No approach speed and no landing weight are published.**

### textron_caravan_product_card

**Cessna Caravan product card (Model 208)** - Textron Aviation  
Document: undated product card PDF  
URL: https://cessna.txtav.com/-/media/cessna/files/product-cards/turboprop/caravan_product_card.pdf  
Retrieved: 2026-09-07  
Local path: `data/reference_speeds/textron/caravan_product_card.pdf` (292,430 bytes)  
sha256: `37248e00073bee31791ffee44bbea482753187729ac841358a26643a594019ba`

Read: WEIGHTS block: Max Takeoff Weight 8,000 lb, Empty Weight 4,730 lb (2,145 kg). Excerpt `excerpts/textron_C208_caravan_product_card_weights.txt`. **No approach speed and no landing weight are published.** The card is the Model 208 Caravan; the FAA C208 row's MTOW of 9,062 lb is the 208B Grand Caravan EX, which shares the designator.

### textron_skylane_product_card

**Cessna Skylane product card (Model 182T)** - Textron Aviation  
Document: undated product card PDF  
URL: https://cessna.txtav.com/-/media/cessna/files/product-cards/piston/skylane_product_card.pdf  
Retrieved: 2026-09-07  
Local path: `data/reference_speeds/textron/skylane_product_card.pdf` (165,699 bytes)  
sha256: `822d777046be61fc264dcf2947d5fb2965601fe90651dc2cbc463e10b75cd7c7`

Read: WEIGHTS block: Max Takeoff Weight 3,100 lb, Basic Empty Weight 2,000 lb (907 kg). Excerpt `excerpts/textron_C182_skylane_product_card_weights.txt`. **No approach speed and no landing weight are published.**

### textron_king_air_360_product_card

**Beechcraft King Air 360 product card (Model B300)** - Textron Aviation  
Document: undated product card PDF  
URL: https://beechcraft.txtav.com/-/media/beechcraft/files/product-card/king_air_360_product_card.pdf  
Retrieved: 2026-09-07  
Local path: `data/reference_speeds/textron/king_air_360_product_card.pdf` (375,870 bytes)  
sha256: `36a0ece4210b8487a8999d19a8ed66052420b47f3c8dcc763a3ef60aa3c50b3d`

Read: WEIGHTS block: Max Takeoff Weight 15,000 lb (the same as the FAA MTOW and MALW), Basic Operating Weight 9,955 lb (4,516 kg). Excerpt `excerpts/textron_B350_kingair360_product_card_weights.txt`. **No approach speed and no landing weight are published.** The King Air 360 (Model B300) is the current build of the Super King Air 350 that carries the B350 designator; it does NOT cover BE30 (Model 300), a different, out-of-production airframe.

### textron_king_air_260_product_card

**Beechcraft King Air 260 product card (Model B200GT)** - Textron Aviation  
Document: undated product card PDF  
URL: https://beechcraft.txtav.com/-/media/beechcraft/files/product-card/king_air_260_product_card.pdf  
Retrieved: 2026-09-07  
Local path: `data/reference_speeds/textron/king_air_260_product_card.pdf` (349,773 bytes)  
sha256: `ea962cb5c1ce2f596a9ea70912993ec43af19c1caabe9adf2469c770519fefcc`

Read: WEIGHTS block: Max Takeoff Weight 12,500 lb (the same as the FAA MTOW and MALW), Basic Operating Weight 8,830 lb (4,005 kg). Excerpt `excerpts/textron_BE20_kingair260_product_card_weights.txt`. **No approach speed and no landing weight are published.** The King Air 260 is the current build of the Model 200 family that carries the BE20 designator.

### textron_bonanza_product_card

**Beechcraft Bonanza G36 product card (Model G36)** - Textron Aviation  
Document: undated product card PDF  
URL: https://beechcraft.txtav.com/-/media/beechcraft/files/product-card/bonanza_product_card.pdf  
Retrieved: 2026-09-07  
Local path: `data/reference_speeds/textron/bonanza_product_card.pdf` (220,323 bytes)  
sha256: `587d951feca2d3b251e537082afa35e6afef70bbadecdcd6eb14a7d40e92addb`

Read: WEIGHTS block: Max Takeoff Weight 3,805 lb, Basic Empty Weight 2,605 lb (1,182 kg). Excerpt `excerpts/textron_BE36_bonanza_product_card_weights.txt`. **No approach speed and no landing weight are published.** Beechcraft has retired the Bonanza's model page (beechcraft.txtav.com/en/bonanza returns HTTP 404, checked 2026-09-07) but the product card itself is still served.

### textron_baron_product_card

**Beechcraft Baron G58 product card (Model G58)** - Textron Aviation  
Document: undated product card PDF  
URL: https://beechcraft.txtav.com/-/media/beechcraft/files/product-card/baron_product_card.pdf  
Retrieved: 2026-09-07  
Local path: `data/reference_speeds/textron/baron_product_card.pdf` (1,866,590 bytes)  
sha256: `bfef90840da9287bc94fb7f0fb9c539709652f6d4acb77b4cdc2973ad1314cd7`

Read: WEIGHTS block: Max Takeoff Weight 5,500 lb (the same as the FAA MTOW), Basic Empty Weight 3,965 lb (1,798 kg). Excerpt `excerpts/textron_BE58_baron_product_card_weights.txt`. **No approach speed and no landing weight are published.** As with the Bonanza, the model page is gone (HTTP 404) and only the card remains.

### cirrus_sr_series_specs

**SR Series product page, All Specs block (SR20 / SR22 / SR22T)** - Cirrus Aircraft  
Document: product page (HTML) as served 2026-09-07  
URL: https://cirrusaircraft.com/aircraft/sr22/  
Retrieved: 2026-09-07  
Local path: `data/reference_speeds/cirrus/cirrusaircraft_sr22.html` (2,092,005 bytes)  
sha256: `72459d59823f08d426b4ec1966413c00a86a28dbf688f89d4807209a0fa95af7`

Read: the 'All Specs' block of the SR Series page, which carries the SR20, SR22 and SR22T columns side by side (told apart by horsepower 215 / 310 / 315): Max Gross Weight 3,150 / 3,600 / 3,600 lb and Base Weight 2,147 lb (947 kg) / 2,272 lb (1,030 kg) / 2,362 lb (1,071 kg). Excerpt `excerpts/cirrus_SR_series_specs_weights.txt`. **No approach speed and no landing weight are published.** The SR Series product brochure (SRSeries_G7_ProductBrochure_2026.pdf) carries no weight table at all, which is why the page is the source.

### cirrus_vision_jet_specs

**Vision Jet (SF50) product page, Aircraft Specifications block** - Cirrus Aircraft  
Document: product page (HTML) as served 2026-09-07  
URL: https://cirrusaircraft.com/aircraft/vision-jet/  
Retrieved: 2026-09-07  
Local path: `data/reference_speeds/cirrus/cirrusaircraft_vision-jet.html` (850,698 bytes)  
sha256: `4ef8cbc0a6234fd847a22715f03aa17228399a06ce4778c20243a986920a0332`

Read: Aircraft Specifications -> Weight block: Maximum Takeoff Weight 6,000 lbs (2,727 kg), Basic Empty Weight 3,550 lbs (1,610 kg). Excerpt `excerpts/cirrus_VisionJet_specs_weights.txt`. **No approach speed and no landing weight are published** - neither here nor in the G3 Vision Jet product brochure - which is why the SF50 row's `malw_kg` comes from the EASA TCDS below.

### easa_tcds_sf50_issue6

**Type Certificate Data Sheet EASA.IM.A.615 - Cirrus SF50** - EASA  
Document: TCDS No. EASA.IM.A.615, Issue 6, 01 June 2026  
URL: https://www.easa.europa.eu/en/downloads/24242/en  
Retrieved: 2026-09-07  
Local path: `data/reference_speeds/easa/EASA_TCDS_IM_A_615_SF50_Issue6.pdf` (517,249 bytes)  
sha256: `24bae613e45c18994d014957fdd9a5944d3230285f94f51e1f88bc57a37bb738`

Read: item 11 'Maximum Certified Weights': Ramp 2740 kg (6040 lb), Takeoff 2722 kg (6000 lb), **Landing 2517 kg (5550 lb)**, Zero Fuel 2223 kg (4900 lb). Excerpt `excerpts/easa_tcds_SF50_weights.txt`. Used for ONE number: the SF50's landing weight, because the FAA row's MALW_lb cell reads 'N/A'. A TCDS publishes certified maxima only - it states no minimum weight and no operating empty weight, so it is never a minimum-mass source here.

### piper_archer_lx_2026

**2026 Archer LX specification foldover** - Piper Aircraft, Inc.  
Document: 2026_Archer_LX_Foldover.pdf  
URL: https://www.piper.com/wp-content/uploads/2019/01/2026_Archer_LX_Foldover.pdf  
Retrieved: 2026-09-07  
Local path: `data/reference_speeds/piper/2026_Archer_LX_Foldover.pdf` (4,587,052 bytes)  
sha256: `1c6146ddcfa9a441e5873728edf1f22b64a4bb9207c9981f4df7ae410154d137`

Read: the Weight block of the foldover: Maximum Takeoff Weight 2,550 lb (1,157 kg), Maximum Ramp Weight 2,558 lb, Standard Empty Weight 1,688 lb (766 kg), Standard Useful Load 870 lb. Excerpt `excerpts/piper_ArcherLX_2026_weights.txt`. **No approach speed and no landing weight are published.** The Archer LX is the current PA-28-181 that carries the P28A designator; the FAA P28A row is an older Cherokee (MTOW 2,440 lb, MALW 2,130 lb), so the two describe different weight variants of the same designator.

### piper_m500_2026

**2026 M500 brochure** - Piper Aircraft, Inc.  
Document: 2026_M500_Brochure_V2.pdf  
URL: https://www.piper.com/wp-content/uploads/2019/01/2026_M500_Brochure_V2.pdf  
Retrieved: 2026-09-07  
Local path: `data/reference_speeds/piper/2026_M500_Brochure_V2.pdf` (13,696,027 bytes)  
sha256: `6c42fcce6686b17548e9068a97c983f14274f803b25e68f152bc27aa066530dc`

Read: the Weight block: Maximum Takeoff Weight 5,092 lb (the same as the FAA MTOW), Maximum Ramp Weight 5,134 lb, Standard Empty Weight 3,436 lb (1,559 kg), Standard Useful Load 1,698 lb. Excerpt `excerpts/piper_M500_2026_weights.txt`. **No approach speed and no landing weight are published.** The M500 (PA-46-500TP) is the current build of the Malibu Meridian that carries the P46T designator.

### diamond_da40ng_folder_2025

**DA40 NG Product Folder 2025** - Diamond Aircraft Industries  
Document: DA40_NG_Product_Folder_2025_SCREEN.pdf  
URL: https://www.diamondaircraft.com/fileadmin/diamondaircraft/documents/da40/DA40_NG_Product_Folder_2025_SCREEN.pdf  
Retrieved: 2026-09-07  
Local path: `data/reference_speeds/diamond/DA40_NG_Product_Folder_2025_SCREEN.pdf` (9,571,266 bytes)  
sha256: `f9512ef36cf7e6277aa42d13c87617a19dd7eed63789c24c462d6e2e9d9895d3`

Read: 'DA40 NG FACTS AND SPECIFICATIONS': Max. take-off mass 1,310 kg (2,888 lbs, the same as the FAA MTOW), **Empty weight without options 903 kg (1,991 lbs)**, Max. useful load 407 kg. Excerpt `excerpts/diamond_DA40NG_2025_weights.txt`. **No approach speed and no landing weight are published.** The same folder carries a DA40 XLT column; only the NG column was taken.

### dassault_falcon2000lxs_backgrounder

**Falcon 2000LXS Backgrounder** - Dassault Aviation  
Document: Falcon-2000LXS-Backgrounder.pdf (served from /app/uploads/2023/02/)  
URL: https://www.dassaultfalcon.com/app/uploads/2023/02/Falcon-2000LXS-Backgrounder.pdf  
Retrieved: 2026-09-07  
Local path: `data/reference_speeds/dassault/Falcon-2000LXS-Backgrounder.pdf` (298,437 bytes)  
sha256: `55e2f344889141b387259998fe2a205f64bb01b7c1e36dc6fca2b9f2dc8b0ab3`

Read: the WEIGHTS/CAPACITIES block: Maximum Takeoff Weight 42,800 lb, **Maximum Landing Weight 39,300 lb (17,826 kg)** - the same weight as the FAA MALW - Maximum Zero-Fuel Weight 29,700 lb, Maximum Fuel Weight 16,660 lb. Excerpt `excerpts/dassault_Falcon2000LXS_weights.txt`. **No operating empty weight is published**, which is why F2TH still carries `min_mass_kg: null`. The backgrounder does print 'Approach Speed, Vref (Typical Landing Weight): 105 kias', but that is at a typical, not maximum, landing weight and so is not comparable with the FAA figure; it is recorded in the row's corroboration note and not used.

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

## Types added for the no-dynamics cohort

The 132 types below are the ones the five-airport cohort flies that the optimizer has **no dynamics
model** for; they were added on 2026-09-07 so that the speed gate can still anchor on a published
number. Columns are the same as the table above minus `EC Vat` (EUROCONTROL was not queried for
these types). `rows` is again the observed arrival count in the cohort.

| type | rows | FAA speed kt main/min/max | FAA MALW lb | malw_kg | manufacturer corroboration (doc, section) | minimum mass kg | kind | min-mass source | notes |
|---|---:|---|---:|---:|---|---:|---|---|---|
| BCS3 | 1582 | 135 / 135 / 135 | 133,600 | 60,600 | MLW 61,008 kg -- `airbus_a220_acp_issue013` ACP Table 2 | **36,287** | MFW | `airbus_a220_acp_issue013` |  |
| E55P | 719 | 116 / 116 / 116 | 16,685 | 7,568 | none published | **null** | - | none | see 'no published minimum mass' |
| CRJ7 | 674 | 135 / 135 / 135 | 67,000 | 30,391 | MLW 30,390 kg -- `bombardier_crj700_apm_r15` APM 00-02-01 | **19,051** | MFW | `bombardier_crj700_apm_r15` |  |
| BCS1 | 482 | 130 / 130 / 130 | 119,490 | 54,200 | MLW 54,658 kg -- `airbus_a220_acp_issue013` ACP Table 2 | **34,927** | MFW | `airbus_a220_acp_issue013` |  |
| P28A | 409 | 70 / 70 / 70 | 2,130 | 966 | none published | **766** | SEW | `piper_archer_lx_2026` |  |
| CL35 | 359 | 124 / 124 / 124 | 34,150 | 15,490 | MLW 15,490 kg -- `bombardier_challenger3500_specs` page Specifications/Weights | **11,249** | BOW | `bombardier_challenger3500_specs` |  |
| PC12 | 336 | 85 / 85 / 85 | 9,921 | 4,500 | MLW 4,500 kg -- `pilatus_pc12_factsheet` factsheet WEIGHTS | **3,040** | BOW | `pilatus_pc12_factsheet` |  |
| C68A | 267 | 100 / 100 / 100 | 27,575 | 12,508 | none published | **8,462** | BOW | `textron_citation_latitude_product_card` |  |
| CL30 | 196 | 126 / 126 / 126 | 33,750 | 15,309 | none published | **null** | - | none | see 'no published minimum mass' |
| E75S | 193 | 124 / 124 / 124 | 74,957 | 34,000 | MLW 34,000 kg -- `embraer_e175_apm` APM Table 2.1 | **21,500** | BOW | `embraer_e175_apm` |  |
| E545 | 180 | 111 / 111 / 111 | 32,518 | 14,750 | none published | **null** | - | none | see 'no published minimum mass' |
| C25B | 179 | 108 / 108 / 108 | 12,750 | 5,783 | none published | **3,874** | EW | `textron_citation_cj3_gen3_product_card` |  |
| C680 | 167 | 108 / 108 / 108 | 27,100 | 12,292 | none published | **null** | - | none | see 'no published minimum mass' |
| B712 | 165 | 139 / 139 / 139 | 100,000 | 45,359 | MLW 45,359 kg -- `boeing_717_acap_rev_b` ACAP 2.1 | **30,617** | OEW | `boeing_717_acap_rev_b` |  |
| CL60 | 165 | 137 / 137 / 137 | 38,000 | 17,237 | MLW 17,237 kg -- `bombardier_challenger650_specs` page Specifications/Weights | **12,315** | BOW | `bombardier_challenger650_specs` |  |
| C750 | 157 | 131 / 131 / 131 | 31,800 | 14,424 | none published | **null** | - | none | see 'no published minimum mass' |
| B350 | 134 | 107 / 107 / 107 | 15,000 | 6,804 | none published | **4,516** | BOW | `textron_king_air_360_product_card` |  |
| G280 | 127 | 125 / 125 / 125 | 32,700 | 14,832 | none published | **null** | - | none | see 'no published minimum mass' |
| H25B | 126 | 137 / 137 / 137 | 23,350 | 10,591 | none published | **null** | - | none | see 'no published minimum mass' |
| C208 | 121 | 79 / 79 / 79 | 7,800 | 3,538 | none published | **2,145** | EW | `textron_caravan_product_card` |  |
| GLF4 | 117 | 144 / 144 / 144 | 66,000 | 29,937 | none published | **null** | - | none | see 'no published minimum mass' |
| C700 | 115 | 121 / 121 / 121 | 33,500 | 15,195 | none published | **10,705** | BOW | `textron_citation_longitude_product_card` |  |
| GLEX | 111 | 131 / 131 / 131 | 78,600 | 35,652 | MLW 35,652 kg -- `bombardier_global6500_specs` page Specifications/Weights | **23,691** | BOW | `bombardier_global6500_specs` |  |
| C560 | 105 | 103 / 103 / 103 | 15,200 | 6,895 | none published | **null** | - | none | see 'no published minimum mass' |
| F2TH | 98 | 130 / 130 / 130 | 39,300 | 17,826 | MLW 17,826 kg -- `dassault_falcon2000lxs_backgrounder` backgrounder WEIGHTS | **null** | - | none | see 'no published minimum mass' |
| BE20 | 88 | 107 / 107 / 107 | 12,500 | 5,670 | none published | **4,005** | BOW | `textron_king_air_260_product_card` |  |
| SR22 | 70 | 78 / 78 / 78 | 3,400 | 1,542 | none published | **1,030** | BW | `cirrus_sr_series_specs` | **FAA MALW anomaly**, see row note |
| LJ60 | 59 | 125 / 125 / 125 | 19,500 | 8,845 | none published | **null** | - | none | see 'no published minimum mass' |
| BE40 | 58 | 114 / 114 / 114 | 15,700 | 7,121 | none published | **null** | - | none | see 'no published minimum mass' |
| TBM7 | 53 | 74 / 74 / 74 | 6,250 | 2,835 | none published | **null** | - | none | see 'no published minimum mass' |
| BE36 | 47 | 77 / 77 / 77 | 3,650 | 1,656 | none published | **1,182** | BEW | `textron_bonanza_product_card` |  |
| E550 | 47 | 113 / 113 / 113 | 34,524 | 15,660 | none published | **null** | - | none | see 'no published minimum mass' |
| GALX | 46 | 130 / 130 / 130 | 30,000 | 13,608 | none published | **null** | - | none | see 'no published minimum mass' |
| F900 | 46 | 130 / 130 / 130 | 44,500 | 20,185 | none published | **null** | - | none | see 'no published minimum mass' |
| BE30 | 44 | 107 / 107 / 107 | 14,000 | 6,350 | none published | **null** | - | none | see 'no published minimum mass' |
| DA40 | 40 | 77 / 77 / 77 | 2,407 | 1,092 | none published | **903** | EW | `diamond_da40ng_folder_2025` |  |
| SF50 | 40 | 87 / 87 / 87 | N/A (rejected) | 2,517 | no landing weight published -- `cirrus_vision_jet_specs` page Specifications | **1,610** | BEW | `cirrus_vision_jet_specs` | **FAA MALW anomaly**, see row note |
| C25C | 40 | 111 / 111 / 111 | 15,660 | 7,103 | none published | **4,672** | BOW | `textron_citation_cj4_gen3_product_card` |  |
| S22T | 39 | 77 / 77 / 77 | 3,600 | 1,633 | none published | **1,071** | BW | `cirrus_sr_series_specs` |  |
| BE9L | 38 | 100 / 100 / 100 | 9,600 | 4,354 | none published | **null** | - | none | see 'no published minimum mass' |
| BE58 | 38 | 95 / 95 / 95 | 5,400 | 2,449 | none published | **1,798** | BEW | `textron_baron_product_card` |  |
| T210 | 38 | 73 / 73 / 73 | 3,800 | 1,724 | none published | **null** | - | none | see 'no published minimum mass' |
| AT43 | 37 | 104 / 104 / 104 | 36,160 | 16,402 | none published | **null** | - | none | see 'no published minimum mass' |
| GA6C | 31 | 129 / 129 / 129 | 76,800 | 34,836 | none published | **null** | - | none | see 'no published minimum mass' |
| PRM1 | 29 | 120 / 120 / 120 | 11,600 | 5,262 | none published | **null** | - | none | see 'no published minimum mass' |
| HDJT | 28 | 111 / 111 / 111 | 9,859 | 4,472 | none published | **null** | - | none | **FAA MALW anomaly**, see row note; see 'no published minimum mass' |
| C650 | 27 | 126 / 126 / 126 | 17,000 | 7,711 | none published | **null** | - | none | see 'no published minimum mass' |
| M20P | 26 | 77 / 77 / 77 | 3,200 | 1,451 | none published | **null** | - | none | see 'no published minimum mass' |
| C425 | 24 | 98 / 98 / 98 | 8,000 | 3,629 | none published | **null** | - | none | see 'no published minimum mass' |
| GL7T | 24 | 125 / 125 / 125 | 85,799 | 38,918 | none published | **null** | - | none | see 'no published minimum mass' |
| P28R | 23 | 73 / 73 / 73 | 2,900 | 1,315 | none published | **null** | - | none | **FAA MALW anomaly**, see row note; see 'no published minimum mass' |
| P46T | 23 | 75 / 75 / 75 | 4,850 | 2,200 | none published | **1,559** | SEW | `piper_m500_2026` |  |
| E50P | 21 | 100 / 100 / 100 | 9,766 | 4,430 | none published | **null** | - | none | see 'no published minimum mass' |
| LJ31 | 21 | 120 / 120 / 120 | 15,300 | 6,940 | none published | **null** | - | none | see 'no published minimum mass' |
| FA50 | 20 | 124 / 124 / 124 | 35,715 | 16,200 | none published | **null** | - | none | see 'no published minimum mass' |
| P180 | 20 | 121 / 121 / 121 | 11,500 | 5,216 | none published | **null** | - | none | see 'no published minimum mass' |
| C510 | 20 | 105 / 105 / 105 | 8,000 | 3,629 | none published | **null** | - | none | see 'no published minimum mass' |
| BT36 | 20 | 73 / 73 / 73 | 3,850 | 1,746 | none published | **null** | - | none | see 'no published minimum mass' |
| BE35 | 19 | 72 / 72 / 72 | 3,400 | 1,542 | none published | **null** | - | none | see 'no published minimum mass' |
| C182 | 19 | 65 / 65 / 65 | 2,950 | 1,338 | none published | **907** | BEW | `textron_skylane_product_card` |  |
| PA38 | 16 | 60 / 60 / 60 | 1,670 | 757 | none published | **null** | - | none | see 'no published minimum mass' |
| E35L | 15 | 124 / 124 / 124 | 40,785 | 18,500 | none published | **null** | - | none | see 'no published minimum mass' |
| P32R | 15 | 80 / 80 / 80 | 3,600 | 1,633 | none published | **null** | - | none | see 'no published minimum mass' |
| FA7X | 15 | 104 / 104 / 104 | 62,400 | 28,304 | none published | **null** | - | none | see 'no published minimum mass' |
| A30B | 14 | 137 / 137 / 137 | 299,829 | 136,000 | none published | **null** | - | none | see 'no published minimum mass' |
| AA5 | 14 | 69 / 69 / 69 | 2,200 | 998 | none published | **null** | - | none | see 'no published minimum mass' |
| C150 | 14 | 55 / 55 / 55 | 1,600 | 726 | none published | **null** | - | none | see 'no published minimum mass' |
| C25M | 12 | 100 / 100 / 100 | 9,900 | 4,491 | none published | **3,171** | EW | `textron_citation_m2_gen3_product_card` |  |
| SW4 | 12 | 112 / 112 / 112 | 15,675 | 7,110 | none published | **null** | - | none | see 'no published minimum mass' |
| G150 | 12 | 130 / 130 / 130 | 21,700 | 9,843 | none published | **null** | - | none | see 'no published minimum mass' |
| PA34 | 12 | 81 / 81 / 81 | 4,513 | 2,047 | none published | **null** | - | none | see 'no published minimum mass' |
| PA46 | 12 | 75 / 75 / 75 | 3,900 | 1,769 | none published | **null** | - | none | see 'no published minimum mass' |
| C414 | 11 | 95 / 95 / 95 | 6,750 | 3,062 | none published | **null** | - | none | see 'no published minimum mass' |
| SR20 | 10 | 74 / 74 / 74 | 2,900 | 1,315 | none published | **947** | BW | `cirrus_sr_series_specs` | **FAA MALW anomaly**, see row note |
| C340 | 10 | 94 / 94 / 94 | 5,990 | 2,717 | none published | **null** | - | none | see 'no published minimum mass' |
| GA5C | 10 | 132 / 132 / 132 | 64,350 | 29,189 | none published | **null** | - | none | see 'no published minimum mass' |
| B78X | 10 | 149 / 149 / 149 | 445,000 | 201,849 | none published | **null** | - | none | see 'no published minimum mass' |
| DA42 | 9 | 88 / 88 / 88 | 3,748 | 1,700 | none published | **null** | - | none | see 'no published minimum mass' |
| C210 | 9 | 85 / 85 / 85 | 3,800 | 1,724 | none published | **null** | - | none | see 'no published minimum mass' |
| LJ35 | 8 | 128 / 128 / 128 | 14,300 | 6,486 | none published | **null** | - | none | see 'no published minimum mass' |
| FA20 | 8 | 107 / 107 / 107 | 27,320 | 12,392 | none published | **null** | - | none | see 'no published minimum mass' |
| EA50 | 7 | 91 / 91 / 91 | 5,415 | 2,456 | none published | **null** | - | none | see 'no published minimum mass' |
| C82R | 7 | 64 / 64 / 64 | 2,950 | 1,338 | none published | **null** | - | none | see 'no published minimum mass' |
| PA32 | 7 | 78 / 78 / 78 | 3,600 | 1,633 | none published | **null** | - | none | see 'no published minimum mass' |
| BE33 | 7 | 69 / 69 / 69 | 3,400 | 1,542 | none published | **null** | - | none | see 'no published minimum mass' |
| C30J | 6 | 128 / 128 / 128 | 130,000 | 58,967 | none published | **null** | - | none | see 'no published minimum mass' |
| PA44 | 6 | 66 / 66 / 66 | 3,800 | 1,724 | none published | **null** | - | none | see 'no published minimum mass' |
| P28B | 6 | 62 / 62 / 62 | 3,000 | 1,361 | none published | **null** | - | none | see 'no published minimum mass' |
| C421 | 5 | 96 / 96 / 96 | 7,200 | 3,266 | none published | **null** | - | none | see 'no published minimum mass' |
| PAY2 | 5 | 100 / 100 / 100 | 9,000 | 4,082 | none published | **null** | - | none | see 'no published minimum mass' |
| HA4T | 5 | 128 / 128 / 128 | 33,500 | 15,195 | none published | **null** | - | none | see 'no published minimum mass' |
| H25C | 4 | 132 / 132 / 132 | 25,000 | 11,340 | none published | **null** | - | none | see 'no published minimum mass' |
| SB20 | 4 | 122 / 122 / 122 | 48,501 | 22,000 | none published | **null** | - | none | see 'no published minimum mass' |
| FA10 | 3 | 107 / 107 / 107 | 17,640 | 8,001 | none published | **null** | - | none | see 'no published minimum mass' |
| B190 | 2 | 121 / 109 / 121 | 16,000 | 7,257 | none published | **null** | - | none | dual FSB value; see 'no published minimum mass' |
| C180 | 2 | 64 / 64 / 64 | 2,800 | 1,270 | none published | **null** | - | none | see 'no published minimum mass' |
| BE9T | 2 | 108 / 108 / 108 | 10,950 | 4,967 | none published | **null** | - | none | see 'no published minimum mass' |
| PA24 | 2 | 75 / 75 / 75 | 2,900 | 1,315 | none published | **null** | - | none | see 'no published minimum mass' |
| PA27 | 2 | 91 / 91 / 91 | 4,940 | 2,241 | none published | **null** | - | none | see 'no published minimum mass' |
| TEX2 | 2 | 103 / 103 / 103 | 6,900 | 3,130 | none published | **null** | - | none | **FAA MALW anomaly**, see row note; see 'no published minimum mass' |
| H25A | 2 | 125 / 125 / 125 | 22,046 | 10,000 | none published | **null** | - | none | see 'no published minimum mass' |
| COL4 | 2 | 78 / 78 / 78 | 3,230 | 1,465 | none published | **null** | - | none | see 'no published minimum mass' |
| AEST | 2 | 96 / 96 / 96 | 6,000 | 2,722 | none published | **null** | - | none | see 'no published minimum mass' |
| TBM9 | 2 | 85 / 85 / 85 | 7,024 | 3,186 | none published | **null** | - | none | see 'no published minimum mass' |
| B36T | 2 | 73 / 73 / 73 | 3,650 | 1,656 | none published | **null** | - | none | see 'no published minimum mass' |
| C152 | 2 | 56 / 56 / 56 | 1,675 | 760 | none published | **null** | - | none | see 'no published minimum mass' |
| C185 | 2 | 64 / 64 / 64 | 3,350 | 1,520 | none published | **null** | - | none | see 'no published minimum mass' |
| TB20 | 2 | 75 / 75 / 75 | 3,080 | 1,397 | none published | **null** | - | none | see 'no published minimum mass' |
| C177 | 1 | 52 / 52 / 52 | 2,500 | 1,134 | none published | **null** | - | none | see 'no published minimum mass' |
| DHC6 | 1 | 74 / 74 / 74 | 12,300 | 5,579 | none published | **null** | - | none | see 'no published minimum mass' |
| PA31 | 1 | 95 / 95 / 95 | 6,500 | 2,948 | none published | **null** | - | none | see 'no published minimum mass' |
| AT73 | 1 | 109 / 109 / 109 | 47,068 | 21,350 | none published | **null** | - | none | see 'no published minimum mass' |
| BL17 | 1 | 79 / 79 / 79 | 3,325 | 1,508 | none published | **null** | - | none | see 'no published minimum mass' |
| BE10 | 1 | 111 / 111 / 111 | 11,210 | 5,085 | none published | **null** | - | none | see 'no published minimum mass' |
| C240 | 1 | 78 / 78 / 78 | 3,420 | 1,551 | none published | **null** | - | none | see 'no published minimum mass' |
| B764 | 1 | 150 / 150 / 150 | 350,000 | 158,757 | none published | **null** | - | none | see 'no published minimum mass' |
| C310 | 1 | 87 / 87 / 87 | 5,500 | 2,495 | none published | **null** | - | none | see 'no published minimum mass' |
| A346 | 1 | 153 / 153 / 153 | 584,225 | 265,000 | none published | **null** | - | none | see 'no published minimum mass' |
| P32T | 1 | 80 / 80 / 80 | 3,600 | 1,633 | none published | **null** | - | none | see 'no published minimum mass' |
| BE60 | 1 | 98 / 98 / 98 | 6,775 | 3,073 | none published | **null** | - | none | see 'no published minimum mass' |
| CH7B | 1 | 56 / 56 / 56 | 1,800 | 816 | none published | **null** | - | none | see 'no published minimum mass' |
| CRJ1 | 1 | 141 / 141 / 141 | 47,000 | 21,319 | none published | **null** | - | none | see 'no published minimum mass' |
| E135 | 1 | 124 / 124 / 124 | 41,226 | 18,700 | none published | **null** | - | none | see 'no published minimum mass' |
| SBR1 | 1 | 126 / 126 / 126 | 17,500 | 7,938 | none published | **null** | - | none | see 'no published minimum mass' |
| GLF3 | 1 | 140 / 140 / 140 | 58,500 | 26,535 | none published | **null** | - | none | see 'no published minimum mass' |
| MU2 | 1 | 105 / 105 / 105 | 11,025 | 5,001 | none published | **null** | - | none | see 'no published minimum mass' |
| MD83 | 1 | 144 / 144 / 144 | 139,500 | 63,276 | none published | **null** | - | none | see 'no published minimum mass' |
| WW24 | 1 | 129 / 129 / 129 | 19,000 | 8,618 | none published | **null** | - | none | see 'no published minimum mass' |
| FA8X | 1 | 106 / 106 / 106 | 62,400 | 28,304 | none published | **null** | - | none | see 'no published minimum mass' |
| MD88 | 1 | 130 / 130 / 130 | 130,000 | 58,967 | none published | **null** | - | none | see 'no published minimum mass' |
| B722 | 1 | 133 / 133 / 133 | 150,000 | 68,039 | none published | **null** | - | none | see 'no published minimum mass' |
| C441 | 1 | 98 / 98 / 98 | 9,360 | 4,246 | none published | **null** | - | none | see 'no published minimum mass' |

### FAA rows that could not be turned into an entry

Two designators are in the FAA spreadsheet with an **`N/A` MALW cell**, and the pack's whole
convention is that the approach speed is quoted *at* the maximum allowable landing weight. Without
that weight the speed cannot be scaled, so no row was written and nothing was invented:

| type | rows | FAA row | FAA speed kt | FAA MTOW lb | why there is no entry |
|---|---:|---|---:|---:|---|
| BE95 | 10 | Beechcraft 95 Travel Air | 79 | 4,200 | MALW cell reads `N/A`; the type left production in 1968 and Textron Aviation publishes nothing for it |
| T38 | 6 | Northrop T-38 Talon | 160 | 12,093 | MALW cell reads `N/A`; a military trainer with no civil type certificate data sheet to fall back on |

A third such row, **SF50** (40 rows), *was* recoverable: EASA TCDS EASA.IM.A.615 publishes a
certified landing weight of 2,517 kg (5,550 lb), and that is what its `malw_kg` carries - the only
row in the pack besides B739 whose landing weight does not come from the FAA cell. Its row note
says so.

### FAA rows whose two weight cells contradict each other

Five added rows publish a MALW **above** their own MTOW, which no certified airframe can have. The
MALW is carried as published (it is the weight the FAA's own speed is defined at) and each row's
`malw_note` starts with `ANOMALY:`.

| type | rows | FAA MALW lb | FAA MTOW lb | what the manufacturer publishes |
|---|---:|---:|---:|---|
| SR22 | 70 | 3,400 | 2,358 | Cirrus: Max Gross Weight 3,600 lb, Base Weight 2,272 lb - the FAA MTOW cell is close to the *empty* weight |
| SR20 | 10 | 2,900 | 2,126 | Cirrus: Max Gross Weight 3,150 lb, Base Weight 2,147 lb - same swap |
| HDJT | 28 | 9,859 | 9,039 | Honda publishes an HA-420 maximum takeoff weight above 10,000 lb; the MTOW cell is the wrong one |
| P28R | 23 | 2,900 | 2,758 | not resolved; no reachable Piper document for the Arrow |
| TEX2 | 2 | 6,900 | 6,500 | not resolved |

### One added row whose speed is not an FSB figure

**B78X** (10 rows, Boeing 787-10) carries the FAA remark *"Approach speed estimated by Virginia
Tech."* - the only row in the pack whose `Approach_Speed_knot` the database itself says is a
third-party estimate rather than a value read out of a Flight Standardization Board report or a
manufacturer manual. It is carried as published, with that sentence as its `approach_speed_note`.

## Types not in the FAA database

33 designators of the cohort (121 observed rows between them) have **no row at all** in the October
2024 Aircraft Characteristics Database, so they get no entry - the pack has no published approach
speed for them and does not manufacture one. Listed with their observed row counts:

| type | rows | | type | rows | | type | rows |
|---|---:|---|---|---:|---|---|---:|
| C82T | 23 | | GA7C | 5 | | AS55 | 1 |
| T206 | 12 | | AS50 | 3 | | B429 | 1 |
| G200 | 11 | | RV6 | 2 | | B06 | 1 |
| B407 | 8 | | H60 | 2 | | H269 | 1 |
| EC20 | 7 | | SREY | 2 | | AS65 | 1 |
| DA62 | 6 | | RV7 | 2 | | F100 | 1 |
| C82S | 6 | | PTS1 | 1 | | RV14 | 1 |
| P28S | 5 | | TWEN | 1 | | GA8 | 1 |
| M700 | 5 | | P06T | 1 | | LGEZ | 1 |
| M600 | 5 | | | | | K100 | 1 |
| | | | | | | C27J | 1 |
| | | | | | | RV10 | 1 |
| | | | | | | PC7 | 1 |
| | | | | | | A119 | 1 |

Most are helicopters (B407, EC20, AS50/AS55/AS65, H60, B06, B429, H269, A119), homebuilts
(RV6/RV7/RV10/RV14, LGEZ, SREY) or recent models the October 2024 edition predates (M700, DA62,
C82T/C82S, T206, G200). The two largest, **C82T** (23 rows, Cessna Turbo Skylane) and **T206**
(12 rows, Turbo Stationair HD), do have current Textron product cards - the missing piece is the
FAA *speed*, not the mass, so the cards were not downloaded.

## Types with no published minimum mass

109 of the 172 types carry `min_mass_kg: null`. Nothing is guessed for any of them; what was tried
is recorded here.

### The original 40 types

Four of them, recorded when the pack was first built:

* **GLF5** (147 rows) - no published figure found. gulfstream.com no longer serves a Gulfstream V / G550 aircraft page (HTTP 403 AccessDenied, checked 2026-09-07), Gulfstream airport planning manuals sit behind the MyGulfstream login, and OpenAP 2.4 has no glf5.yml. Third-party compilations quote 48,300 lb but they are not manufacturer publications, so nothing is recorded here
* **C25A** (119 rows) - no published figure found. Textron Aviation has retired the Citation CJ2 / Model 525A product page (cessna.txtav.com/en/citation/cj2 returns HTTP 404, checked 2026-09-07) and OpenAP 2.4 has no c25a.yml
* **LJ45** (103 rows) - no published figure found. Bombardier's public document index (RACSDocument.nsf) carries only CRJ and Dash 8 manuals, there is no Learjet 45 airport planning manual or spec sheet on a Bombardier domain, and OpenAP 2.4 has no lj45.yml
* **C525** (35 rows) - no published figure found. Textron Aviation has retired the CitationJet / CJ1 (Model 525) product page (cessna.txtav.com/en/citation/cj1 returns HTTP 404, checked 2026-09-07) and OpenAP 2.4 has no c525.yml

### The 132 added types

105 of the added rows have no published minimum mass. **OpenAP is not available as a fallback for a
single one of them** (see Conventions), so every one of these is a manufacturer-document search that
came back empty.

Why so many: the FAA database is a *fleet* database, so most of these designators are airframes that
left production years or decades ago - Citation V / X / III-VI-VII / Mustang / 425, Hawker 800 and
Premier 1, Learjet 31/35/60, King Air 90 and 300, Falcon 50/900/7X/10/20, Gulfstream III/IV/G150/G200,
ATR 42-300, Beechjet 400, TBM 700, Mooney M20C, Cessna T210/340/414/421/441. Manufacturers do not host
spec sheets for retired models, and the airport planning manuals that would have the weights sit behind
customer logins. The pack does not substitute a third-party compilation for them.

The gaps that are **not** explained by retirement - types still in production whose maker simply does
not publish a weight in public - are worth naming, because a future revision could close them from a
document this session could not reach:

| type | rows | what was tried, 2026-09-07 |
|---|---:|---|
| E55P | 719 | Embraer publishes no weights at all in its public material: the Phenom 300E brochure (`embraer.com/media/b4lchr5u/phenom-300e-electronic.pdf`, downloaded and read) carries range, speed, thrust and field length but **no weight table**, and the model page's HTML contains the word "weight" zero times. Embraer's airport planning manuals live behind the flyembraer.com customer login; the one APM in this pack (E175) is a copy filed in an NTSB public docket, and no such docket copy exists for the Phenom or Praetor |
| CL30 | 196 | the Challenger 300 is out of production and `bombardier.com/en/aircraft/challenger-300` returns HTTP 404; the Challenger 3500 page covers the 350/3500 (CL35) only |
| E545 | 180 | as E55P - the Praetor 500 brochure (`embraer.com/media/asbhvchi/praetor-500-eletronic.pdf`) publishes no weights |
| C680 | 167 | Textron has retired the Sovereign product card (`cessna.txtav.com/-/media/cessna/files/product-cards/citation/citation_sovereign_product_card.pdf` and the `sovereign_` variant both HTTP 404) |
| C750 | 157 | Textron has retired the Citation X card (`citation_x_product_card.pdf` HTTP 404) |
| G280 | 127 | gulfstream.com serves a 1,103-byte bot interstitial for `/en/aircraft/gulfstream-g280/` (HTTP 200 but no content), and `/en/newsroom/press-kit/` returns HTTP 403. No Gulfstream document was reachable for any of its six designators |
| H25B | 126 | Hawker 800 out of production; Textron Aviation supports the fleet but publishes no card |
| GLF4 | 117 | as G280 |
| C560 | 105 | Citation V/Ultra/Encore out of production; no card |
| F2TH | 98 | the Falcon 2000LXS backgrounder **is** in the pack and gives MTOW/MLW/MZFW/max fuel - but Dassault publishes no operating empty weight in it, and the same is true of the 900/7X/8X sheets in that series |
| E550 | 47 | as E545 |
| GA6C | 31 | as G280 |
| HDJT | 28 | hondajet.com returns HTTP 403 to every path tried, including the site root |
| GL7T | 24 | `bombardier.com/en/aircraft/global-7500` now redirects to the Global 8000 page, whose HTML contains the word "weight" zero times; `bombardier.com/en/media/media-library/global-7500` returns HTTP 403 |
| E50P | 21 | as E55P - the Phenom 100EX brochure publishes no weights |
| P180 | 20 | piaggioaerospace.it renders its aircraft pages from JavaScript; `/en/aviation/p180-avanti-evo` and `/en/p180-avanti-evo` both return HTTP 404 and the site index exposes no Avanti brochure PDF |
| P28R | 23 | `piper.com/model/arrow/` still resolves but carries no weight block (the Arrow is no longer in the price list) |

The remaining 88 null rows are all out-of-production designators; 71 of them have fewer than 20
observed records each (369 rows in total) and no document was sought for them.

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

