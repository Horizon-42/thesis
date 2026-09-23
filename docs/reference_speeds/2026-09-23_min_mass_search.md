# Minimum-mass search, 2026-09-23 — authority documents first

Status: **applied 2026-09-23** for the accepted rows (6 of the 40 plus the 6 bonus designators, 13 rows
with FA7X/FA8X): each figure was re-read from its downloaded document, sha256 checked, before it went into
`aircraft/reference_speeds.json` and `README.md` ("2026-09-23: type certificate data sheets"). H25B is
recorded as `MZFW` (the data sheet's own term). The archived-only figures are NOT applied - they await the
user's decision.

Scope: the 40 null-`min_mass_kg` designators ranked by blocked observed flights (E55P … LJ31). The
2026-09-07 dead ends (manufacturer pages retired or behind logins, see README "Types with no published
minimum mass") were not repeated; the new angle was **authority documents**: EASA and FAA type-certificate
data sheets (TCDS), TCCA and ANAC TCDS, then manufacturer pages served today, then — kept apart —
manufacturer documents that survive only on web.archive.org.

Acceptance rule (the user's, 宁缺毋滥): (a) EASA/FAA TCDS printing a weight usable as a minimum; (b) a
manufacturer publication served from its own domain today; (c) another civil-aviation-authority document.
No third-party compilations, no OpenAP, no derivations. Archived manufacturer documents are listed
separately and are **not** accepted.

## Result per type

| type | flights | result | value as printed | kind (document's term) | model | proposed source key | page |
|---|---:|---|---|---|---|---|---|
| E55P | 1128 | ARCHIVED-ONLY | 11,583 lb (5,254 kg) | BOW ("Basic Operating Weight", standard configuration) | Phenom 300 (2012 page) | `archived_embraer_phenom300_performance_2012` | HTML |
| E545 | 317 | NONE | – | – | – | – | – |
| CL30 | 305 | ARCHIVED-ONLY | 23,700 lb (10,750 kg) | BOW ("Typical basic operating weight") | Challenger 300 (2010 fact sheet) | `archived_bombardier_challenger300_factsheet_2010` | 1 |
| C750 | 264 | ARCHIVED-ONLY | 22,131 lb (10,038 kg) | BOW ("Basic Oper Weight") | Citation X+ only | `archived_textron_xplus_productcard_2017` | 1 |
| C680 | 238 | ARCHIVED-ONLY | 18,235 lb (8,271 kg) | BOW ("Basic Oper Weight") | Sovereign+ only | `archived_textron_sovereignplus_productcard_2017` | 1 |
| GLF5 | 219 | ARCHIVED-ONLY | 48,300 lb (21,909 kg) | BOW ("Basic Operating (including 4 crew)") | G550 only | `archived_gulfstream_g550_2019` | HTML |
| GLF4 | 203 | NONE | – | – | – | – | – |
| H25B | 197 | ACCEPTED (kind needs a decision) | 13,100 lb / 5,942 kg | "Minimum zero fuel weight" (no MFW printed) | HS.125-700A/700B | `easa_tcds_im_a_085_issue05` (+ `faa_tcds_a3eu_rev44`) | 28 (EASA) / 34 (FAA) |
| C25A | 186 | NONE | – | – | – | – | – |
| G280 | 180 | ARCHIVED-ONLY | 24,150 lb (10,954 kg) | BOW ("Basic Operating (including 2 crew)") | G280 | `archived_gulfstream_g280_2019` | HTML |
| F2TH | 159 | ACCEPTED | 20,100 lb (at 32.5% CG) | MFW ("Minimum flight") | Falcon 2000 | `faa_tcds_a50nm_rev16` | 4 |
| LJ45 | 154 | ACCEPTED | 14,000 lb | MFW ("Min. Flight Weight") | Learjet 45 | `faa_tcds_t00008wi_rev34` | 3 |
| C560 | 144 | NONE | – | – | – | – | – |
| BE40 | 99 | NONE | – | – | – | – | – |
| LJ60 | 90 | ARCHIVED-ONLY | 14,896 lb (6,757 kg) | BOW ("Typical basic operating weight") | Learjet 60 XR only | `archived_bombardier_learjet60xr_factsheet_2012` | 1 |
| BE30 | 85 | NONE | – | – | – | – | – |
| TBM7 | 77 | NONE | – | – | – | – | – |
| E550 | 76 | NONE | – | – | – | – | – |
| F900 | 74 | ACCEPTED | 9,390 kg (20,700 lb) | MFW ("Minimum flight") | Mystère-Falcon 900 (same for 900EX) | `easa_tcds_a062_issue10` (+ `faa_tcds_a46eu_rev23`) | 32 (EASA) / 8 (FAA) |
| GALX | 72 | NONE | – | – | – | – | – |
| BE9L | 71 | ARCHIVED-ONLY | 7,265 lb (3,295 kg) | BOW ("Basic Operating Weight*", typ. equipped w/1 pilot) | King Air C90GTx only | `archived_textron_king_air_c90gtx_productcard_2017` | 1 |
| PA38 | 67 | NONE | – | – | – | – | – |
| T210 | 56 | NONE | – | – | – | – | – |
| AT43 | 55 | NONE | – | – | – | – | – |
| M20P | 55 | NONE | – | – | – | – | – |
| P28R | 55 | NONE | – | – | – | – | – |
| BE35 | 55 | NONE | – | – | – | – | – |
| C525 | 51 | NONE | – | – | – | – | – |
| PRM1 | 49 | NONE | – | – | – | – | – |
| GA6C | 47 | ACCEPTED | 50,900 lb / 23,088 kg | BOW ("Basic Operating (including 4 crew)") | G600 | `gulfstream_g600_specs` | HTML |
| C425 | 41 | NONE | – | – | – | – | – |
| HDJT | 39 | NONE | – | – | – | – | – |
| P32R | 37 | NONE | – | – | – | – | – |
| FA50 | 36 | ACCEPTED | 8,600 kg (18,959 lbs) | MFW ("Minimum flight") | Mystère-Falcon 50 | `easa_tcds_a062_issue10` (+ `faa_tcds_a46eu_rev23`) | 15 (EASA) / 3 (FAA) |
| C650 | 36 | NONE | – | – | – | – | – |
| GL7T | 36 | NONE | – | – | – | – | – |
| C510 | 34 | ARCHIVED-ONLY | 5,600 lb (2,540 kg) | BOW ("Basic Oper Weight") | Citation Mustang | `archived_textron_mustang_productcard_2017` | 1 |
| P180 | 34 | NONE | – | – | – | – | – |
| E50P | 33 | ARCHIVED-ONLY | 7,132 lb (3,235 kg) | BOW ("Basic Operating Weight") | Phenom 100 (2012 page) | `archived_embraer_phenom100_performance_2012` | HTML |
| LJ31 | 30 | NONE | – | – | – | – | – |

Totals over the 40: **6 ACCEPTED** (F2TH, LJ45, F900, FA50, GA6C, and H25B pending the kind decision),
**10 ARCHIVED-ONLY**, **24 NONE**.

Reading notes for the accepted rows:

* **F2TH** — the FAA minimum flight weight is CG-dependent: 23,075 lb at 14% CG, **20,100 lb at 32.5% CG**
  (Falcon 2000); 23,444 / 21,149 lb for the 2000EX. Lowest printed = 20,100 lb = 9,117 kg (×0.45359237;
  the TCDS prints pounds only). EASA A.008 Issue 13 prints the line "Minimum flight" three times **with
  no value** (checked on the rendered page).
* **LJ45** — 14,000 lb = 6,350 kg (pounds only in the TCDS). Learjet 40/70 (13,500 lb) and 75 are other
  designators and were not used.
* **F900 / FA50** — EASA prints kg and lb; FAA A46EU prints the same pounds. The MF900 "with M1196" column
  of A.062 misprints the label as "Maximum flight 9,390 kg"; the unambiguous column prints "Minimum flight".
* **H25B** — no Hawker section prints a minimum *flight* weight. Printed minima: "Minimum zero fuel
  weight" 13,100 lb (700A/B), 14,120 lb (800/800XP/850XP/900XP/750), and "Minimum operating weight"
  16,100 lb (850XP, 900XP only). Neither term is in the `min_mass_kind` vocabulary. The lowest printed
  floor is 13,100 lb (5,942 kg, HS.125-700). Taking the 16,100 lb "minimum operating weight" for the whole
  designator would put the floor above what the TCDS allows the 700/800/750 to weigh. Coordinator decides.
* **GA6C** — web page, not a file; sha256 will change with site edits, the excerpt pins the number.

## Bonus findings (designators outside the 40, found in the same documents)

| type | pack now | printed | kind | source key | page |
|---|---|---|---|---|---|
| GLEX | 23,691 kg BOW (Global 6500 page) | 48,200 lb / 21,865 kg | MFW ("Min. Flight Weight"), BD-700-1A10 | `faa_tcds_t00003ny_rev24` | 1 |
| GL5T | 23,070 kg BOW (Global 5000 fact sheet) | 51,200 lb / 23,224 kg | MFW, BD-700-1A11 — *higher* than the pack's BOW | `faa_tcds_t00003ny_rev24` | 3 |
| C56X | 5,924 kg BOW (Ascend card) | 12,400 lb / 5,625 kg | "Minimum Weight In-flight" (MFW), 560XL S/N 5001–6000 | `easa_tcds_im_a_207_issue10` (+ `faa_tcds_a22ce_rev74`) | 53 / 22 |
| FA7X, FA8X | null | 14,696 kg (32,400 lbs) aft CG; 15,694 kg fwd | MFW ("Minimum flight") | `easa_tcds_a155_issue17` | 16 |
| FA10 | null | 4,500 kg | MFW ("Minimum flight") | `easa_tcds_a173_issue1` | 8–9 |
| GA5C | null | 46,850 lb / 21,251 kg | BOW ("Basic Operating (including 3 crew)") | `gulfstream_g500_specs` | HTML |

GL5T shows the preference-order question in its sharpest form: the certified MFW (23,224 kg) is above the
manufacturer's typical BOW (23,070 kg). The G500 page is the GVII-G500 (GA5C); the older G-5SP marketed as
"G500" is filed under GLF5 by ICAO and must not take this number.

## Accepted documents (on disk; `data/` is git-ignored)

| source key | publisher | document | URL | local path | bytes | sha256 |
|---|---|---|---|---|---:|---|
| `easa_tcds_a062_issue10` | EASA | TCDS EASA.A.062 MF50/MF900/Falcon 900EX, Issue 10, 15 Sep 2026 | https://www.easa.europa.eu/en/downloads/7401/en | `data/reference_speeds/easa/EASA_TCDS_A_062_MF50_MF900_F900EX_Issue10.pdf` | 1,074,818 | `d42d1e5425ae4718eb1077a530b7d0c9e17c4e52520923c1ba0bf0c5a739ed5a` |
| `faa_tcds_a46eu_rev23` | FAA | TCDS A46EU Mystère-Falcon 50/900, Falcon 900EX, Rev 23, Jan 16 2026 | https://drs.faa.gov/browse/excelExternalWindow/DRSDOCID108734422520260121164435.0001 | `data/reference_speeds/faa/FAA_TCDS_A46EU_Rev23.pdf` | 458,071 | `a8c444d8dfaef39fb0e21178267facb2c7946729af6b486378b9c546d9b4dc06` |
| `faa_tcds_a50nm_rev16` | FAA | TCDS A50NM Falcon 2000/2000EX, Rev 16, Nov 13 2025 | https://drs.faa.gov/browse/excelExternalWindow/DRSDOCID173314097720251113164846.0001 | `data/reference_speeds/faa/FAA_TCDS_A50NM_Rev16.pdf` | 404,442 | `f3952bdd89e69257e863312293fa4c3fb1480f3a031e9fed50a2b67bc6ad70be` |
| `faa_tcds_t00008wi_rev34` | FAA | TCDS T00008WI Learjet 45, Rev 34, June 8 2026 | https://drs.faa.gov/browse/excelExternalWindow/DRSDOCID173477243220260610144929.0001 | `data/reference_speeds/faa/FAA_TCDS_T00008WI_Rev34.pdf` | 240,538 | `00317751673bb2ec3b346131ef133c115065ceb54b271341c1810ee740ae1bed` |
| `easa_tcds_im_a_085_issue05` | EASA | TCDS EASA.IM.A.085 Hawker Series, Issue 05, 27 Sep 2018 | https://www.easa.europa.eu/en/downloads/7356/en | `data/reference_speeds/easa/EASA_TCDS_IM_A_085_Hawker_Issue05.pdf` | 2,062,377 | `acde0ab102659c6f8ebe90e74e36c87fef776a4e7bcbea7e2a80affb1bf615b1` |
| `faa_tcds_a3eu_rev44` | FAA | TCDS A3EU Hawker series, Rev 44, Nov 6 2017 | https://drs.faa.gov/browse/excelExternalWindow/5871EF1C07FE2CFC862581D300724B70.0001 | `data/reference_speeds/faa/FAA_TCDS_A3EU_Rev44.pdf` | 760,759 | `b771840f4c82b7d83a1ca8c836666a50b46b7c63ab48242c3f9090e46a670dae` |
| `gulfstream_g600_specs` | Gulfstream | G600 product page (HTML), as served 2026-09-23 | https://www.gulfstream.com/en/aircraft/gulfstream-g600/ | `data/reference_speeds/gulfstream/gulfstream_g600.html` | 124,582 | `2bf221ef14e3b9bb38768f44b8d1fc4cc0ed0968959555907dd295a0bdc82626` |
| `faa_tcds_t00003ny_rev24` | FAA | TCDS T00003NY BD-700-1A10/1A11/2A12, Rev 24, Dec 19 2025 | https://drs.faa.gov/browse/excelExternalWindow/DRSDOCID116658665820251219185513.0001 | `data/reference_speeds/faa/FAA_TCDS_T00003NY_Rev24.pdf` | 495,841 | `2e6cd88f2bdcb6228eecb078ba3f9ce9efc309a582709a4d0c2b5abec2965dc5` |
| `faa_tcds_a22ce_rev74` | FAA | TCDS A22CE Cessna 500/550/S550/552/560/560XL, Rev 74, Dec 8 2025 | https://drs.faa.gov/browse/excelExternalWindow/DRSDOCID132119292720251209201014.0001 | `data/reference_speeds/faa/FAA_TCDS_A22CE_Rev74.pdf` | 276,375 | `006dc82b5d0c0be00ed8888b0cda08de5f1874ed5f9b3e496ff480e843476de9` |
| `easa_tcds_im_a_207_issue10` | EASA | TCDS EASA.IM.A.207 Cessna 500–560XL, Issue 10, 06 Jul 2022 | https://www.easa.europa.eu/en/downloads/7244/en | `data/reference_speeds/easa/EASA_TCDS_IM_A_207_Cessna_500_560XL_Issue10.pdf` | 738,739 | `a2c8fcdc55214623d33de8d65eb653726cc237d2b8d8cb7bc466e955808a48bd` |
| `easa_tcds_a155_issue17` | EASA | TCDS EASA.A.155 Falcon 7X, Issue 17, 23 Jul 2025 | https://www.easa.europa.eu/en/downloads/7288/en | `data/reference_speeds/easa/EASA_TCDS_A_155_Falcon7X_Issue17.pdf` | 343,270 | `6e988bab072039a44ad53b9c6464cc25ed380ef9cf0833da708f5e97c50bc80a` |
| `easa_tcds_a173_issue1` | EASA | TCDS EASA.A.173 Falcon 10, Issue 1, 9 Oct 2009 | https://www.easa.europa.eu/en/downloads/7270/en | `data/reference_speeds/easa/EASA_TCDS_A_173_Falcon10_Issue1.pdf` | 121,529 | `8bbf314c85a8e2a1f1d11c9570ece2c1b0ba46c0aa74ad4ec4011924e0bd25ca` |
| `gulfstream_g500_specs` | Gulfstream | G500 product page (HTML), as served 2026-09-23 | https://www.gulfstream.com/en/aircraft/gulfstream-g500/ | `data/reference_speeds/gulfstream/gulfstream_g500.html` | 122,625 | `511b4a86108d8b980395fc6c6a68eaf9fdbd4a105f385867f530afe4c0e95715` |

Each PDF has a `pdftotext -layout` `.txt` next to it. Excerpts: `docs/reference_speeds/excerpts/<source_key>_weights.txt`.

**FAA DRS access** (for `fetch_sources.sh`, not edited here): a plain `curl` of `drs.faa.gov/api/...`
returns HTTP 403. It works after `GET https://drs.faa.gov/guest/login` with a cookie jar; then
`GET /api/browse/documents/summaryguiddocview/<docUniqueId>` returns JSON whose `id` is the file id, and
`GET /api/content/alf/<id>` returns the PDF. Document search: `POST /api/drs/search/simpleSearch` with
`{"searchText":["\"A50NM\""], ...}`. The TCDS number search is exact; model-name full-text search ranks badly.

## Archived manufacturer documents (NOT accepted — for the user's decision)

Stored apart under `data/reference_speeds/_archived/<publisher>/`; excerpts are
`excerpts/archived_<key>_weights.txt`. All are the manufacturer's own former pages/files, captured by the
Internet Archive; none is served by the manufacturer today.

| key | type | printed | capture URL | local path | bytes | sha256 |
|---|---|---|---|---|---:|---|
| `archived_embraer_phenom300_performance_2012` | E55P | BOW 11,583 lb (5,254 kg) | web.archive.org/web/20120601233735/http://embraerexecutivejets.com:80/en-US/jets/phenom-300/Pages/performance.aspx | `_archived/embraer/wayback_20120601233735_embraerexecutivejets_phenom-300_performance.html` | 76,935 | `4894168a0db8c40c814ef61b3c9289e5c8a1309ee042322728e2cb8c8960f1cb` |
| `archived_embraer_phenom100_performance_2012` | E50P | BOW 7,132 lb (3,235 kg) | web.archive.org/web/20120605011528/http://embraerexecutivejets.com:80/en-US/jets/phenom-100/Pages/performance.aspx | `_archived/embraer/wayback_20120605011528_embraerexecutivejets_phenom-100_performance.html` | 77,388 | `32ba331ac99d359e2b0599801645de9667f9b99f05386139ee07e92bc4d3bda3` |
| `archived_bombardier_challenger300_factsheet_2010` | CL30 | Typical BOW 23,700 lb (10,750 kg) | web.archive.org/web/20100524190729/http://businessaircraft.bombardier.com:80/en/3_0/3_2/pdf/challenger_300_factsheet.pdf | `_archived/bombardier/wayback_20100524190729_challenger_300_factsheet.pdf` | 5,377,021 | `40ead2a7c91136b9e87e0adde5e49eae5d52022f8aa8dca659dfc2345ed40a53` |
| (same, 2013 issue) | CL30 | Typical BOW 23,850 lb (10,818 kg) | web.archive.org/web/20130219050248/http://businessaircraft.bombardier.com/content/dam/bombardier/en/aircraft/challenger/Fact%20Sheet/Challenger%20300%20factsheet.pdf | `_archived/bombardier/wayback_20130219050248_Challenger_300_factsheet.pdf` | 450,177 | `ea10aa013a9c4a319b843ad7423ccf3b0f23e56658914d87f453b1014c8ffe5f` |
| `archived_bombardier_learjet60xr_factsheet_2012` | LJ60 | Typical BOW 14,896 lb (6,757 kg), 60 XR | web.archive.org/web/20121021012715/http://businessaircraft.bombardier.com:80/content/dam/bombardier/en/aircraft/learjet/Fact%20Sheets/L60XR_EN.pdf | `_archived/bombardier/wayback_20121021012715_L60XR_EN.pdf` | 704,483 | `0bd4fa85acea4c3ae0301988c295b094cf7ba88326090c9b86d952b747018728` |
| `archived_gulfstream_g550_2019` | GLF5 | BOW (incl. 4 crew) 48,300 lb / 21,909 kg, G550 | web.archive.org/web/20190603084406/https://www.gulfstream.com/aircraft/gulfstream-g550 | `_archived/gulfstream/wayback_20190603084406_gulfstream_g550.html` | 93,748 | `aac04e0a5d13ab16a47710ad600d729401c30f15b27d4c786b92ff926d9266c2` |
| `archived_gulfstream_g280_2019` | G280 | BOW (incl. 2 crew) 24,150 lb / 10,954 kg | web.archive.org/web/20190520153203/https://www.gulfstream.com/aircraft/gulfstream-g280 | `_archived/gulfstream/wayback_20190520153203_gulfstream_g280.html` | 98,353 | `1517dd99e1c34e07456c424793a9e2b79619d26a7a1b465211ab0eafa4075138` |
| `archived_textron_xplus_productcard_2017` | C750 | Basic Oper Weight 22,131 lb (10,038 kg), X+ | web.archive.org/web/20170420093056/http://cessna.txtav.com:80/-/media/cessna/files/citation/xplus/xplus_productcard.ashx | `_archived/textron/wayback_20170420093056_xplus_productcard.pdf` | 302,508 | `6d09d3c473a2020498598b7f9ed2aa432d2d8b3939e2b5c9ec510ed35e6fc107` |
| `archived_textron_sovereignplus_productcard_2017` | C680 | Basic Oper Weight 18,235 lb (8,271 kg), Sovereign+ | web.archive.org/web/20170413004647/http://cessna.txtav.com:80/-/media/cessna/files/citation/sovereignplus/sovereignplus_productcard.ashx | `_archived/textron/wayback_20170413004647_sovereignplus_productcard.pdf` | 381,161 | `afca68d9c86b7c3feea7a9764cea0cd336a4e96aa18d6f690d5c5ba5ac81fb34` |
| `archived_textron_mustang_productcard_2017` | C510 | Basic Oper Weight 5,600 lb (2,540 kg) | web.archive.org/web/20170411054820/http://cessna.txtav.com:80/-/media/cessna/files/citation/mustang/mustang_productcard.ashx | `_archived/textron/wayback_20170411054820_mustang_productcard.pdf` | 255,132 | `86f030950c605494d0b7ec5a1a8f27b1477cb07708d0f107e2b78d866a0158f6` |
| `archived_textron_king_air_c90gtx_productcard_2017` | BE9L | Basic Operating Weight* 7,265 lb (3,295 kg), C90GTx | web.archive.org/web/20170413002200/http://beechcraft.txtav.com:80/-/media/beechcraft/files/litho/king_air_c90gtx_productcard.ashx | `_archived/textron/wayback_20170413002200_king_air_c90gtx_productcard.pdf` | 377,457 | `d238fd40da9e8bfb5ec9b8364a686a73a7656575a611f63d1a8b13fefc703ea8` |

Caveats: each archived figure is for one marketing variant at one date (Phenom 300 2012, X+, Sovereign+,
G550, C90GTx, 60 XR); none is shown to be the lowest across the designator's models. Later captures of
the Embraer page (2014/2016/2018) print no BOW value. Archived Embraer brochures (Phenom 100/300,
Legacy 450/500, 2014–2018) and the Legacy 450/500 performance pages (2012–2018) print no BOW.

## What was tried, per type (2026-09-23)

"EASA" = TCDS from `https://www.easa.europa.eu/en/downloads/<id>/en` (HTTP 200 for every one fetched).
"FAA" = TCDS from drs.faa.gov via the guest-login route above. "max only" = the weights section prints
maximum ramp/take-off/landing/zero-fuel weights and no minimum of any kind (grep for minimum/min./empty/
basic/operating weight over the whole text, then read of the weights section).

| type | authority documents checked | manufacturer today | other |
|---|---|---|---|
| E55P | EASA IM.A.158 Iss 11 (29 Aug 2024, dl 7282) max only; FAA A60CE Rev 14 max only; ANAC EA-2009T12-13 (sistemas.anac.gov.br/certificacao/Produtos/Espec/EA-2009T12-13i.pdf) max only — its CG envelope has vertices down to 5,150 kg but no labelled minimum, not taken | embraer.com/executive-jets-our-aircraft/phenom-300ev/en/ HTTP 200, 0 "weight"; only brochure linked is the known phenom-300e-electronic.pdf | archived 2012 page: BOW (see above) |
| E545 | EASA IM.A.526 Iss 12 (23 Sep 2026, dl 17998) max only; FAA TC00062IB Rev 17 max only; ANAC EA-2014T04-14 max only | praetor-500e page HTTP 200, 0 "weight" | archived Legacy 450 pages/brochures: BOW blank or absent |
| CL30 | EASA IM.A.080 Iss 11 (14 Feb 2025, dl 7364) max only; FAA T00005NY Rev 12 max only; TCCA A-234 Iss 10 (23 Sep 2021, NICO) max only | (Challenger 300 retired, 2026-09-07) | archived 2010/2013 fact sheets: typical BOW |
| C750 | EASA IM.A.097 Iss 03 (07 Jan 2016, dl 7335) max only; FAA T00007WI Rev 19 max only | guessed cessna.txtav.com card URLs (`citation_x_plus_product_card.pdf`, `xplus_productcard.ashx`) → 302 to /error/page-not-found (404) | archived X+ card 2017 |
| C680 | EASA IM.A.033 Iss 09 (25 Mar 2026, dl 7459) max only; FAA T00012WI Rev 18 max only | `citation_sovereign_plus_product_card.pdf`, `sovereignplus_productcard.ashx` → 404 | archived Sovereign+ card 2017 |
| GLF5 | EASA IM.A.070 Iss 13 (06 Aug 2026, dl 7384) max only; FAA A12EA Rev 54 max only | gulfstream.com/en/aircraft/gulfstream-g550/ HTTP 403 | archived G550 page 2019 |
| GLF4 | as GLF5. A12EA pages 23–25 carry a "G-IV Aircraft Zero Fuel Gross Weight Envelope" graph whose lower edge sits at 38 (×1000 lb) for S/N 1000–1213 — a graph reading, not a printed number, and G-IV only; not taken | g450 path HTTP 403 | no archived G450 spec page found (Wayback CDX offline intermittently) |
| H25B | EASA IM.A.085 Iss 05 and FAA A3EU Rev 44: minimum zero fuel / minimum operating weights (ACCEPTED, kind open) | (retired, README) | – |
| C25A | EASA IM.A.078 Iss 19 (07 May 2026, dl 7368) max only; FAA A1WI Rev 34 max only | `citation_cj2_product_card.pdf`, `citation_cj2plus_product_card.pdf` → 404 | no CJ2 file in the Wayback CDX of cessna.txtav.com |
| G280 | EASA IM.A.348 Iss 06 (26 May 2025, dl 7184) max only; FAA A61NM Rev 12 (Dec 21 2021) max only | /en/aircraft/gulfstream-g280/ now redirects to the NEW **G300** page (BOW 11,059 kg incl. 2 crew) — a different model, not taken | archived G280 page 2019 |
| F2TH | FAA A50NM Rev 16: MFW (ACCEPTED). EASA A.008 Iss 13 (15 Sep 2026, dl 7510): "Minimum flight" printed with no value | (backgrounder, README: no OEW) | – |
| LJ45 | FAA T00008WI Rev 34: MFW (ACCEPTED). EASA IM.A.020 Iss 18 max only | – | – |
| C560 | EASA IM.A.207 Iss 10 Section 4 max only; FAA A22CE Rev 74 Sections V/VIII/IX max only (the 550 Bravo and 560XL sections do print minima) | `citation_encore_product_card.pdf` → 404 | – |
| BE40 | FAA A16SW Rev 29 (MU-300-10/400/400A/400T) max only; no EASA TCDS found by library search | `hawker_400xp_product_card.pdf` → 404 | – |
| LJ60 | FAA A10CE Rev 69 max only; EASA IM.A.212 (Learjet M60) Iss 04 max only | – | archived 60 XR fact sheet 2012 |
| BE30 | FAA A24CE Rev 132 (200/300 series) max only; EASA IM.A.277 Iss 22 (B200/B300 — B300 is designator B350, not BE30) max only | – | ICAO BE30 = Beech 300 Super King Air only; the King Air 360 card (B350) does not apply |
| TBM7 | EASA A.010 Iss 22 (24 Apr 2026, dl 16508) max only; FAA A60EU Rev 40 max only | not searched on tbm.aero (TBM 700 out of production) | – |
| E550 | as E545 (same TCDS) | praetor-600e page HTTP 200, 0 "weight" | archived Legacy 500 pages 2012–2018: BOW blank |
| F900 | EASA A.062 Iss 10 + FAA A46EU Rev 23: MFW (ACCEPTED) | – | – |
| GALX | EASA IM.A.013 Iss 05 (26 May 2025, dl 7500) max only; FAA A53NM Rev 9 max only | – | no archived G200 page found |
| BE9L | EASA IM.A.503 Iss 9 (20 Jan 2020, dl 7106) max only; FAA 3A20 Rev 82 max only | `king_air_c90gtx_product_card.pdf`, `king_air_c90_product_card.pdf` → 404 | archived C90GTx card 2017 |
| PA38 | FAA A18SO Rev 6 max only; no EASA TCDS found | – | – |
| T210 | FAA 3A21 Rev 50 max only; no EASA TCDS found | – | – |
| AT43 | EASA A.084 Iss 14 (23 Feb 2026, dl 7358) max only; FAA A53EU Rev 38 max only | not searched on atr-aircraft.com (ATR 42-300 out of production) | – |
| M20P | EASA IM.A.266 Iss 02 (M20M–M20R only) max only; FAA 2A3 not retrievable (DRS search returned no TCDS hit) | mooney.com: connection refused | – |
| P28R | FAA 2A13 Rev 66 (incl. PA-28R) max only; EASA IM.A.234 does not cover PA-28R | (Arrow page, README: no weights) | – |
| BE35 | EASA IM.A.279 Iss 03 max only; FAA A-777 Rev 62 and 3A15 Rev 100 max only | (retired) | – |
| C525 | as C25A (A1WI, IM.A.078) | `citation_cj1_product_card.pdf` → 404 | – |
| PRM1 | EASA IM.A.073 Iss 02 (13 Mar 2018, dl 7378) max only; FAA A00010WI Rev 11 max only | `premier_product_card.pdf` → 404 | – |
| GA6C | EASA IM.A.595 Iss 15 and FAA T00021AT Rev 13 max only | G600 page: BOW (ACCEPTED) | – |
| C425 | FAA A7CE Rev 51 (Section X, Model 425) max only; no EASA TCDS found | `conquest_product_card.pdf` → 404 | – |
| HDJT | EASA IM.A.352 Iss 09 (05 Jun 2023, dl 20876) max only; FAA A00018AT Rev 11 max only | hondajet.com: plain curl HTTP 403, served with browser headers. Elite II page, `HondaJet-Elite-II-specifications.pdf` and `HondaJet-Elite-II-Brochure_091126.pdf`: no weights (only "gross weight increase") | – |
| P32R | FAA A3SO Rev 33 (incl. PA-32R) max only; EASA IM.A.239 Iss 01 max only | – | – |
| FA50 | EASA A.062 Iss 10 + FAA A46EU Rev 23: MFW (ACCEPTED) | – | – |
| C650 | FAA A9NM Rev 27 max only; no EASA TCDS found | – | – |
| GL7T | EASA IM.A.009 Iss 14 max only; FAA T00003NY Rev 24 Section VI max only; TCCA A-177 Iss 23 (5 Nov 2025, scanned, page 18) max only | bombardier.com/en/aircraft/global-8000 HTTP 200, no weights | archived Global 7000 brochure (2015, pre-certification) not opened |
| C510 | EASA IM.A.502 Iss 03 max only; FAA A00014WI Rev 7 max only | `citation_mustang_product_card.pdf`, `mustang_productcard.ashx` → 404 | archived Mustang card 2017 |
| P180 | EASA A.059 Iss 19 (02 Sep 2025, dl 7407) max only; FAA A59EU Rev 27 max only | piaggioaerospace.it/en/business: MTOW 5,489 kg / 12,100 lb only | – |
| E50P | EASA IM.A.157 Iss 09 max only; FAA A59CE Rev 12 max only; ANAC EA-2008T09-11 max only | phenom-100ex page HTTP 200, 0 "weight" | archived 2012 page: BOW |
| LJ31 | FAA A10CE Rev 69 max only | – | – |

## Things found and deliberately not used

* **Gulfstream G300 page** (today's target of the G280 URL): BOW 11,059 kg — a new model, not the G280;
  and not the 2000s-era "G-4 Gulfstream G300" that ICAO files under GLF4.
* **Gulfstream G500 page** is the GVII-G500 (GA5C, bonus row); the G-5SP "G500" is GLF5.
* **G-IV ZFGW envelope graph** (A12EA p.23–25): graph only, G-IV only.
* **ANAC CG-envelope vertices** (e.g. 5,150 kg on the EMB-505 aft limit): not labelled as a minimum weight.
* **FAA A45NM** turned out to be the Dornier 328-100 (MIN. FLIGHT WEIGHT 9,400 kg Mod 10/20, 9,600 kg
  Mod 00); D328 has no pack row, recorded only so nobody re-downloads it for a Falcon.
* **NZ CAA type acceptance reports** (aviation.govt.nz) exist for EMB-505 and Cessna 680; a civil aviation
  authority, but outside the FAA/EASA/TCCA/ANAC list given — not opened.
