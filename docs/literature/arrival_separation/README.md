# Arrival separation: the US rules a multi-runway arrival scheduler must honour, with ICAO and RECAT-EU beside them (2026-09-14)

Why this folder exists: R3 of the runway-intent plan
(`4dTrajectory/ts_transformer/docs/2026-09-13_runway_intent_plan.zh.md` §17) assigns every arrival a
(runway, landing time) over several open runways. Its constraints are same-runway spacing, closely spaced
parallels treated as one runway, dependent diagonal spacing, independent parallels and intersecting
runways. They have to come from the regulation text, not from memory. This folder holds that text, quoted
with paragraph numbers, for R3 and for the multi-aircraft work that follows it.

How to read it:
- **Quotes are copied** from the saved pages: the HTML for 7110.65BB, 7210.3EE and the P/CG, the PDFs for
  everything else. Each is kept under about 30 words, and "…" marks a cut. Paragraph numbers are printed
  with ASCII hyphens; the PDF prints them with a Unicode minus.
- **"(reading)"** marks a step the text does not state. That covers applying a rule to our runway
  geometry, turning a distance into a time, or deciding which rule governs when two apply.
- **Nothing is filled from memory.** §5 lists everything that could not be verified.
- The PDFs and the saved HTML pages are **not tracked in git** (this folder's `.gitignore`; the HTML is
  about 2.6 MB, mostly site navigation). `./download.sh` re-fetches them. The parsed 7360.1K CSV **is**
  tracked: the scheduler's type → CWT table (`inference/runway_schedule.py`) cites it.
- The 7110.65BB consolidated PDF, 7210.3EE, the AIM, ICAO Doc 9643 (SOIR) and the Erzberger–Itoh NASA TP
  live in **`../runway_assignment/`** and are cited there by PDF page, not downloaded again. Run
  `../runway_assignment/download.sh` to get them. Its 7110.65BB copy is byte-identical to the file the FAA
  serves today (md5 `c266c5f2…`, checked 2026-09-14).

## At a glance

| item | state (2026-09-14) |
|---|---|
| Edition in force | **FAA Order JO 7110.65BB, Change 3, effective 2026-07-09**. Basic 2025-02-20, Chg 1 2025-08-07, Chg 2 2026-01-22. Six notices are listed against it (§1.1) |
| CWT (A–I) | **Merged into 7110.65BB by Change 2 (2026-01-22), for terminal facilities only.** It sits in 5-5-4 g/h as TBL 5-5-1 and 5-5-2. JO 7110.126B is canceled. En route keeps the legacy classes in 5-5-4 f |
| Aircraft type → category | JO 7360.1K (2025-06-12), Appendix A, "CWT" column. All 24 dominant types found (§2.5). A parsed CSV of all 2,653 Appendix A rows is saved |
| Same runway | Runway clear before the threshold (3-10-3). 3 mi radar (5-5-4 a/b). 2.5 NM within 10 NM with average ROT ≤ 50 s (5-5-4 j, 7210.3EE 10-4-14). Wake at the threshold (5-5-4 h) |
| Parallels | < 2,500 ft is one runway for wake (5-5-4 h NOTE), except the JO 7110.308E list (STL listed, SJC not). Dependent diagonal 1.0 / 1.5 / 2 NM at 2,500–3,600 / 3,600–8,300 / 8,300–9,000 ft (5-9-6). Independent from 3,600 ft (5-9-7), 3,100 ft with HUR, PRM below 4,300 ft (5-9-8) |
| Intersecting / converging | 3-10-4, 7210.3EE 10-4-12 (SCIA), JO 7110.110B (DCIA with CRDA) |
| ICAO Doc 4444 | **Not freely available** (ICAO Store, USD 442). Paragraph numbers come from the US AIP GEN 1.7, an ICAO APAC paper and SKYbrary; values from EUROCONTROL RECAT-EU Table 1 and SKYbrary (§3.1) |
| RECAT-EU | EUROCONTROL Edition 2.0 (2024-11-08): six categories A–F and an arrival distance matrix (§3.3) |
| Not verified | Per-airport chart and facility authorizations, and FAA-published centerline separations (§5) |

## 1. Sources

### 1.1 FAA: primary text

HTML pages are at `https://www.faa.gov/air_traffic/publications/atpubs/atc_html/<page>#<paragraph>`, for
example `chap5_section_9.html#5-9-6`. One page holds a whole section, and each paragraph number is an
anchor on it.

| document | edition / effective | what was read | local copy |
|---|---|---|---|
| **JO 7110.65BB *Air Traffic Control*** | Basic 02/20/2025; Change 1 08/07/2025; Change 2 01/22/2026; **Change 3 07/09/2026**. The HTML index states "Effective: 7/9/2026 Change: Change 3" | 2-1-19/20, 3-9-6…3-9-9, 3-10-3/3-10-4, 5-5-4, 5-9-5…5-9-11, 7-4-4, Explanation of Changes | `papers/7110.65BB_*.html` (7 sections + index); PDF: `../runway_assignment/official/FAA_Order_JO_7110.65BB_Air_Traffic_Control_w_Chg1-3_2026-07-09.pdf` (927 pp.) |
| JO 7110.65BB **Basic** (superseded) | 02/20/2025 | 5-5-4 f/g/i, the legacy **terminal** wake minima that Change 2 replaced | `papers/FAA_JO_7110.65BB_Basic_eff2025-02-20_EXCERPT_5-5-4_legacy_wake.pdf` (PDF pp. 1, 310–314) |
| Pilot/Controller Glossary (P/CG) | Change 3, effective 7/9/26 (HTML index) | AIRCRAFT CLASSES, AIRCRAFT WAKE CATEGORIES, CLOSE PARALLEL RUNWAYS, SIDESTEP MANEUVER, SIMULTANEOUS … APPROACHES, PRM APPROACH, ATPA | `papers/PCG_glossary-{a,c,p,s}.html`, `PCG_index.html` |
| JO 7210.3EE *Facility Operation and Administration* | Change 3, effective 7/9/2026 | 10-4-10…10-4-14 | `papers/7210.3EE_chap10_section_4_Services.html`; PDF in `../runway_assignment/official/` |
| **JO 7360.1K *Aircraft Type Designators*** | 06/12/2025 | Ch. 2 (column definitions), Appendix A (the decode table with the CWT column) | `papers/FAA_JO_7360.1K_…_EXCERPT.pdf` (PDF pp. 1–121 = front matter + all of Appendix A); `papers/FAA_JO_7360.1K_AppendixA_categories_parsed.csv` |
| N JO 7360.7 (GENOT 25/41) | effective 10/14/2025, cancels 10/14/2026 | Adds one new type: Electra EL-2 ("EL2"), CWT I | `papers/FAA_N_JO_7360.7_…pdf` |
| **JO 7110.308E** *Simultaneous Dependent Approaches to Closely Spaced Parallel Runways* | 10/05/2023 (still listed as current on the publications page) | All of it; Appendix A lists STL | `papers/FAA_JO_7110.308E_…pdf` |
| JO 7110.110B *Dependent Converging Instrument Approaches (DCIA) with CRDA* | 11/17/2017 | ¶1–10 | `papers/FAA_JO_7110.110B_…pdf` |
| JO 7110.663A *Merging, Converging, Parallel Route Sequencing (MCPRS) with CRDA* | 03/01/2024 | ¶1–8 | `papers/FAA_JO_7110.663A_…pdf` |
| US AIP, GEN 1.7 *Differences from ICAO …* | AIP Amendment 1, effective 07/09/2026 | The PANS-ATM (Doc 4444, 16th ed.) difference rows | `papers/US_AIP_GEN_1.7_Differences_from_ICAO.html`, `US_AIP_index.html` |

Change documents and notices in force against 7110.65BB. Sources: the publications page
(`papers/FAA_atpubs_publications_index.html`) and the notices page (`papers/FAA_at_notices_7110.65BB.html`).
The publications page says "Current Notices (7)"; the notices page lists **six**. The seventh was not found.

| notice | effective | cancels | subject | bearing on arrival separation |
|---|---|---|---|---|
| N JO 7110.801 (GENOT 26/14) | 2026-03-18 | 2026-12-24 | Interim helicopter separation (7-2-1) | none for fixed-wing arrivals |
| N JO 7110.802 | 2026-05-19 | 2026-12-24 | Traffic advisories (2-1-21 d) | parallels ≤ 1,000 ft **within Class D** (§2.7) |
| N JO 7110.803 | **2026-10-28** (not yet in force) | 2026-12-24 | VFR/IFR separation, Ch. 7 | none for IFR arrivals on final |
| N JO 7110.804 | **2026-10-28** (not yet in force) | 2026-12-24 | Visual separation (7-2-1) | visual separation only |
| N JO 7110.805 | 2026-06-19 | 2026-12-24 | Approaches to multiple runways (7-4-4 b1, c1) | visual approaches to parallels < 2,500 ft (§2.7) |
| N JO 7110.806 | 2026-09-14 | 2026-12-24 | Micro-EARTS 3 NM (5-5-4) | en route / Micro-EARTS only |

The Change 3 Explanation of Changes touches 5-5-4 only editorially ("replacing Center Radar Approach
Control (CERAP) with Combined Control Facility (CCF) in TBL 1-2-1 and paragraph 5-5-4").

### 1.2 International and secondary sources

| document | edition | status | local copy |
|---|---|---|---|
| ICAO **Doc 4444** PANS-ATM | 16th edition (2016). The store page lists "Amendments 7-13"; Amendment 12 applies from 28 Nov 2024 | **not read**: sold at `https://store.icao.int/en/procedures-for-air-navigation-services-air-traffic-management-doc-4444` (USD 442.00). Unofficial copies found by search were not used | none |
| ICAO APAC SAIOSEACG/4 IP/03, *Update on the amendment concerning separation minima based on an ATS surveillance system to the PANS-ATM* | March 2025 (official ICAO) | read; quotes the amended 8.7.3.3 and 8.7.3.4 | `papers/ICAO_APAC_SAIOSEACG4_IP03_…pdf`. icao.int answers curl with HTTP 403, so this copy was saved through WebFetch |
| **EUROCONTROL RECAT-EU** *European Wake Turbulence Categorisation and Separation Minima on Approach and Departure* | **Edition 2.0, 08/11/2024** (published 5 Dec 2024) | read in full | `papers/EUROCONTROL_RECAT-EU_Edition_2.0_2024-11-08.pdf` |
| SKYbrary *Separation Standards*; *Mitigation of Wake Turbulence Hazard*; *ICAO Wake Turbulence Category* | page revisions 2025-09-04, 2025-12-10 and 2025-12-10 | read. SKYbrary is EUROCONTROL-run but secondary: it paraphrases Doc 4444 | `papers/SKYbrary_*.html` |
| ICAO **Doc 9643 SOIR** | **First Edition, 2004**. The copy in `../runway_assignment/` is named "2nd ed 2020", but its title page says "First Edition — 2004" | read (Ch. 1–2, 5, glossary) | `../runway_assignment/official/ICAO_Doc_9643_SOIR_Simultaneous_Operations_Parallel_Runways_2nd_ed_2020.pdf` |

### 1.3 Cross-links into `../runway_assignment/`

| file there | what it adds here |
|---|---|
| `official/FAA_Order_JO_7110.65BB_…_w_Chg1-3_2026-07-09.pdf` | The PDF pages cited below: 3-10-3 pp. 202–204; 3-10-4 pp. 204–207; 5-5-4 pp. 326–330 (TBL 5-5-1/5-5-2 on pp. 329–330); 5-9-5…5-9-11 pp. 360–370; 7-4-4 pp. 439–440; P/CG A-8/A-9 pp. 646–647; Change 2 CWT statement p. 14 |
| `official/FAA_Order_JO_7210.3EE_…pdf` | 10-4-10 p. 294, 10-4-12 p. 297, 10-4-14 p. 300 |
| `official/FAA_AIM_…pdf` | **5-4-14** Simultaneous Dependent Approaches (pp. 434–435, printed 5-4-42/43), quoted in §2.6–2.7; 5-4-15/16 independent and PRM; 5-4-19 side-step; **7-4-9** Air Traffic Wake Turbulence Separations (p. 613), see §5 |
| `official/ICAO_Doc_9643_SOIR_…pdf` | The parallel-runway operating modes and ICAO's spacing thresholds (§3.2) |
| `nasa_eurocontrol/NASA_TP-2014_Erzberger_Itoh_…pdf` | §2.2 *In-Trail Distance Separations*: how a scheduler turns the distance matrix into a time matrix (Table 1 on p. 6, Table 2 at 130 kt). The authors call these tables "representative examples for the purposes of this paper" (p. 7). Table 1 predates both today's legacy text and CWT (B757 → Large = 4, Large → Large = 2.5), so do not reuse its values |

## 2. FAA rules by topic (JO 7110.65BB Change 3 unless stated)

### 2.1 Same runway, arrival behind arrival

| ¶ | rule | quote |
|---|---|---|
| 3-10-3 a | runway occupancy | "Separate an arriving aircraft from another aircraft using the same runway by ensuring that the arriving aircraft does not cross the landing threshold until one of the following conditions exists…" |
| 3-10-3 a1 | preceding arrival | "The other aircraft has landed and is clear of the runway." |
| 3-10-3 a1 (a)(b) | daylight distance option, **only for SRS Category I/II followers** | "When a Category I aircraft is landing behind a Category I or II- 3,000 feet." / "When a Category II aircraft is landing behind a Category I or II- 4,500 feet." |
| 3-10-3 a2 (c) | preceding departure, airborne | "When either is a category III aircraft- 6,000 feet." |
| 3-9-6 a NOTE | SRS categories | "CATEGORY III− all other aircraft." (I is small single-engine props ≤ 12,500 lb and helicopters; II is small twin props ≤ 12,500 lb). Every jet among our 24 types is SRS III in 7360.1K |
| 3-10-3 b | wake advisories, same **or parallel < 2,500 ft** | "…departing or arriving aircraft on the same runway or parallel runways separated by less than 2,500 feet to:" (1) B–I behind A, B or D; (2) E–I behind C; (3) I behind E |
| 5-9-5 | who separates on the same final | "Radar final controllers ensure that established separation is maintained between aircraft under their control and other aircraft established on the same final approach course." |
| 2-1-19 b | to touchdown | "The separation minima must continue to touchdown for all IFR aircraft not making a visual approach or maintaining visual separation." |

(Reading) 3-10-3 a1 gives a distance option only to Category I/II followers. For an SRS III arrival behind
a landed aircraft, the text therefore requires the runway to be clear (or 3-10-10, altitude-restricted low
approach).

### 2.2 Radar minima on final and the reduced 2.5 NM

| ¶ | value | quote |
|---|---|---|
| 5-5-4 a1/a2 | 3 mi / 5 mi | "TERMINAL. Single Sensor ASR or Digital Terminal Automation System (DTAS):" "When less than 40 miles from the antenna- 3 miles." "When 40 miles or more from the antenna- 5 miles." |
| 5-5-4 b1/b2 | 3 mi / 5 mi | "TERMINAL. FUSION:" "Fusion target symbol – 3 miles." "When displaying ISR in the data block- 5 miles." |
| 5-5-4 c | 5 mi | "STARS Multi-Sensor Mode – 5 miles." |
| **5-5-4 j** | **2.5 NM** | "2.5 nautical miles (NM) separation is authorized between aircraft established on the final approach course within 10 NM of the landing runway when operating in FUSION, or single sensor slant range mode…" (and "…if the aircraft remains within 40 miles of the antenna and:") |
| 5-5-4 j1 | condition | "Wake turbulence separation must be applied in accordance with TBL 5-5-2;" |
| 5-5-4 j2 | condition | "An average runway occupancy time of 50 seconds or less is documented;" |
| 5-5-4 j3 | condition | "CTRDs are operational and used for quick glance references;" |
| 5-5-4 j4 | condition | "Turnoff points are visible from the control tower." |
| 7210.3EE 10-4-14 | ROT documentation | "…reduced to 2.5 NM in‐trail separation on the final approach course within 10 NM of the runway provided an average Runway Occupancy Time (ROT) of 50 seconds or less is documented for each runway." |
| 7210.3EE 10-4-14 | sample | "The average ROT is calculated by using the average of the ROT of no less than 250 arrivals." Revalidation "within 30 days if there is a significant change in runway/taxiway configuration, fleet mix…" |
| AIM 5-4-14 c (pilot side) | summary | "Aircraft on the same final approach course within 10 NM of the runway end are provided a minimum of 3 NM radar separation, reduced to 2.5 NM in certain circumstances." |

Before Change 2, 5-5-4 i of the Basic set two other 2.5 NM conditions, replaced by j1 above: "The leading
aircraft's weight class is the same or less than the trailing aircraft;" and "Super and heavy aircraft are
permitted to participate in the separation reduction as the trailing aircraft only;".

### 2.3 Wake turbulence: CWT categories A–I (terminal, in force since 2026-01-22)

Where CWT lives. The Change 2 Explanation of Changes, item d (PDF p. 14): "The contents of FAA Order JO
7110.126, Consolidated Wake Turbulence (CWT), have been incorporated into FAA Order JO 7110.65, Air Traffic
Control. At this time, the applicability of this change is the terminal air traffic facility. En route air
traffic facilities will continue to use the legacy standards as currently published. Upon publication of
this change, FAA Order JO 7110.126B is canceled."

The definitions are in the P/CG, entry AIRCRAFT WAKE CATEGORIES:
- "CATEGORY A. The Airbus A-380-800 (A388) is classified as a super aircraft."
- "CATEGORY B, C, and D. Aircraft capable of takeoff weights of 300,000 pounds or more whether or not they
  are operating at this weight during a particular phase of flight. These are categorized as heavy aircraft."
- "CATEGORY E. All B757 aircraft."
- "CATEGORY F, and G. Aircraft weighing 41,000 pounds or more maximum certificated takeoff weight, up to
  but not including 300,000 pounds."
- "CATEGORY H and I. Aircraft of less than 41,000 pounds maximum certificated takeoff weight."

The P/CG entry AIRCRAFT CLASSES links the two schemes: Super "is a Category A"; Heavy "are Category B, C,
or D"; Large "are Category F and G"; Small "are Category H and I". Neither entry states how B, C and D, or
F and G, or H and I split. The split comes from the per-type assignment in 7360.1K (§2.5).

Where the tables apply:

| ¶ | quote |
|---|---|
| 5-5-4 g | "TERMINAL. Separate aircraft by the minima specified in TBL 5-5-1 in accordance with the following:" |
| 5-5-4 g1 | "When following an aircraft conducting an instrument approach and/or operating within 2,500 feet and less than 1,000 feet below the flight path of a Category A, B, C, or D aircraft." |
| 5-5-4 g2 | "…within 2,500 feet and/or less than 500 feet below a Category E aircraft." |
| 5-5-4 g NOTE 2 | "Consider runways separated by less than 700 feet as a single runway because of the possible effects of wake turbulence." |
| 5-5-4 h | "TERMINAL. ON APPROACH. In addition to subparagraph g, separate an aircraft on approach behind another aircraft to the same runway by ensuring the separation minima in TBL 5-5-2 will exist at the time the preceding aircraft is over the landing threshold." |
| 5-5-4 h NOTE | "Consider parallel runways less than 2,500 feet apart as a single runway because of the possible effects of wake turbulence." |
| 5-5-4 i | "TERMINAL. When NOWGT is displayed in an aircraft data block, provide 10 miles separation behind the preceding aircraft and 10 miles separation to the succeeding aircraft." |

The tables, in NM. Rows are the leader and columns the follower. "·" is a cell that is blank in the
source. The HTML captions number the tables TBL 5-5-3 and 5-5-4; the text and the PDF number them TBL
5-5-1 and 5-5-2, and the PDF numbering is used here.

**TBL 5-5-1 Wake Turbulence Separation for Directly Behind** (5-5-4 g)

| leader \ follower | A | B | C | D | E | F | G | H | I |
|---|---|---|---|---|---|---|---|---|---|
| A | · | 5 | 6 | 6 | 7 | 7 | 7 | 8 | 8 |
| B | · | 3 | 4 | 4 | 5 | 5 | 5 | 5 | 5 |
| C | · | · | · | · | 3.5 | 3.5 | 3.5 | 5 | 5 |
| D | · | 3 | 4 | 4 | 5 | 5 | 5 | 5 | 5 |
| E | · | · | · | · | · | · | · | · | 4 |
| F, G, H, I | · | · | · | · | · | · | · | · | · |

**TBL 5-5-2 Wake Turbulence Separation for On Approach** (5-5-4 h, applied at the landing threshold)

| leader \ follower | A | B | C | D | E | F | G | H | I |
|---|---|---|---|---|---|---|---|---|---|
| A | · | 5 | 6 | 6 | 7 | 7 | 7 | 8 | 8 |
| B | · | 3 | 4 | 4 | 5 | 5 | 5 | 5 | **6** |
| C | · | · | · | · | 3.5 | 3.5 | 3.5 | 5 | **6** |
| D | · | 3 | 4 | 4 | 5 | 5 | 5 | **6** | **6** |
| E | · | · | · | · | · | · | · | · | 4 |
| F | · | · | · | · | · | · | · | · | **4** |
| G, H, I | · | · | · | · | · | · | · | · | · |

Bold marks the cells where TBL 5-5-2 is larger than TBL 5-5-1. Both matrices were cross-checked between
the HTML (parsed by the `headers` attribute of each cell) and the PDF text; they agree. (Reading) A blank
cell sets no wake minimum. The radar minima of §2.2 still apply; the text does not say this at the table.

### 2.4 Wake turbulence: legacy classes (Super / Heavy / B757 / Large / Small)

**In force today for EN ROUTE only** (5-5-4 f, Change 3):

| ¶ | quote |
|---|---|
| 5-5-4 f1 | "Separate aircraft operating directly behind, following an aircraft conducting an instrument approach and/or operating within 2,500 feet and less than 1,000 feet below, by the following:" |
| 5-5-4 f1(a) | "Behind super - 5 miles, unless the super is operating at or below FL240 and below 250 knots, then: (1) Heavy - 6 miles. (2) Large - 7 miles. (3) Small - 8 miles." |
| 5-5-4 f1(b) | "Behind heavy: (1) Heavy - 4 miles. (2) Large or small - 5 miles." |
| 5-5-4 f2 | "Separate a small aircraft behind a B757 – 4 miles when operating directly behind…" |
| 5-5-4 f3 | at the landing threshold: "(a) Small behind large - 4 miles. (b) Small behind heavy - 6 miles." NOTE: "Consider parallel runways less than 2,500 feet apart as a single runway…" |

**TERMINAL before 2026-01-22** (7110.65BB Basic, 5-5-4 f/g, PDF pp. 312–313 of the Basic):
"(a) TERMINAL. Behind super: (1) Heavy - 6 miles. (2) Large - 7 miles. (3) Small - 8 miles." "Behind heavy:
(1) Heavy - 4 miles. (2) Large or small - 5 miles." "Separate small aircraft behind a B757 by 4 miles…". At
the threshold: "Small behind large− 4 miles." "Small behind heavy− 6 miles." The Basic's 5-5-4 f NOTE reads
"Consider parallel runways less than 2,500 feet apart as a single runway…", and its NOTE 2 under f3 reads
"…less than 700 feet as a single runway…".

Legacy matrix as the Basic text gives it (terminal, NM; "·" = no wake minimum stated). Wake-class
membership comes from the P/CG AIRCRAFT CLASSES entry: Super is the A388; Heavy is 300,000 lb or more;
Large is more than 41,000 lb and below 300,000; Small is 41,000 lb or less. 7360.1K lists the B752/B753 as
weight class L. The B757 row is the only B757-specific arrival rule the Basic text states. A B757 follower
is a Large, so it gets the Large column.

| leader \ follower | Super | Heavy | Large | Small |
|---|---|---|---|---|
| Super | · | 6 | 7 | 8 |
| Heavy | · | 4 | 5 | 5 directly behind; 6 at threshold |
| B757 | · | · | · | 4 |
| Large | · | · | · | 4 at threshold |
| Small | · | · | · | · |

### 2.5 Aircraft type → category (JO 7360.1K)

- 7360.1K ¶2-11: "Consolidated Wake Turbulence (CWT) Categories. See FAA Order JO 7110.126, Consolidated
  Wake Turbulence (CWT), for definitions of these categories." This pointer is dangling; see §5.
- 7360.1K ¶2-10 b defines the weight class letters: "J – Super", "H – Heavy", "L – Large", "S – Small".
  Its NOTE reads: "A '+' denotes an aircraft weighing between 12,500 pounds and 41,000 pounds."
- 7360.1K ¶2-6 restates the ICAO WTC: "Heavy (H) – aircraft types of 300,000 pounds or more…", "Medium (M)
  – aircraft types less than 300,000 pounds and more than 15,400 pounds", "Light (L) – aircraft types of
  15,400 pounds or less."
- Appendix A ("Decode – Aircraft Type Designator") has these columns: Type Designator | Class | Engine
  Number-Type/FAA Weight Class | ICAO WTC | CWT | SRS | LAHSO | Manufacturer, Model.

The 24 types that dominate our data, all found in Appendix A:

| type | FAA weight | ICAO WTC | **CWT** | SRS | 7360.1K page | RECAT-EU Ed. 2.0 Table 2 (examples) |
|---|---|---|---|---|---|---|
| B738 | 2J/L | Medium | **F** | III | A-15 | D |
| B38M | 2J/L | Medium | **F** | III | A-13 | not listed |
| B737 | 2J/L | Medium | **F** | III | A-15 | D |
| B739 | 2J/L | Medium | **F** | III | A-15 | D |
| A319 | 2J/L | Medium | **F** | III | A-2 | D |
| A320 | 2J/L | Medium | **F** | III | A-2 | D |
| A321 | 2J/L | Medium | **F** | III | A-2 | D |
| A20N | 2J/L | Medium | **F** | III | A-2 | not listed |
| A21N | 2J/L | Medium | **F** | III | A-2 | not listed |
| E75L | 2J/L | Medium | **G** | III | A-41 | "E175" printed under E |
| E170 | 2J/L | Medium | **G** | III | A-40 | E |
| E190 | 2J/L | Medium | **F** | III | A-40 | E |
| CRJ2 | 2J/L | Medium | **G** | III | A-32 | E |
| CRJ7 | 2J/L | Medium | **G** | III | A-32 | E |
| CRJ9 | 2J/L | Medium | **G** | III | A-33 | E |
| B752 | 2J/L | Medium | **E** | III | A-15 | C |
| B763 | 2J/H | Heavy | **C** | III | A-15 | C |
| B772 | 2J/H | Heavy | **B** | III | A-15 | B |
| B788 | 2J/H | Heavy | **B** | III | A-15 | B |
| B789 | 2J/H | Heavy | **B** | III | A-15 | B |
| A332 | 2J/H | Heavy | **B** | III | A-2 | B |
| MD11 | 3J/H | Heavy | **C** | III | A-70 | C |
| C172 | 1P/S | Light | **I** | I | A-23 | not listed |
| PC12 | 1T/S | Light | **I** | I | A-82 | not listed |

Notes on the table:
- **The rows were copied from the PDF text, one line per type.** For example B738 reads "Fixed-wing 2J/L
  Medium F III 8" and B763 reads "Fixed-wing 2J/H Heavy C III 9 BOEING, 767-300". The model lists under
  the designators confirm the mapping; B737 covers "BOEING, 737-700" and E75L covers "EMBRAER, 175 (long
  wing)".
- Printed page numbers are the Appendix A page. The PDF page is that number plus 9 (A-1 is PDF p. 10).
- **In our 24 types, CWT is not a function of the FAA weight class.** Among the 2J/L types, E190 is F
  while E170, E75L and the CRJs are G. Among the heavies, B763 and MD11 are C while B772, B788, B789 and
  A332 are B.
- `papers/FAA_JO_7360.1K_AppendixA_categories_parsed.csv` is **derived**: all 2,653 Appendix A rows,
  parsed from `pdftotext -layout` output. The CWT, SRS and LAHSO codes are assigned to columns by
  position against each page's header. The parser is inline at the end of `download.sh`.
  - Check: Appendices B and C repeat the category columns for every designator. **10,184 of 10,194**
    repeated entries agree with the Appendix A parse.
  - The 8 disagreements are inconsistencies inside the source itself: WH4, MIRA*, L18, NH90, PA34, STAR,
    J328, E530*. For example, L18 is G in Appendix A and I in Appendix B/C. The raw Appendix A lines were
    re-read and match the CSV.
  - One row has no CWT in the source: Q4 ("1J/S Medium III").
  - CWT counts: A 2, B 16, C 12, D 41, E 2, F 146, G 97, H 184, I 2,152.

### 2.6 Parallel runways 2,500 ft apart or more

**Simultaneous dependent approaches, 5-9-6** (TERMINAL):

| ¶ | value | quote |
|---|---|---|
| 5-9-6 a1 | turn-on | "Provide a minimum of 1,000 feet vertical or a minimum of 3 miles radar separation between aircraft during turn on." |
| **5-9-6 a2** | **1.0 NM** diagonal, 2,500–3,600 ft | "…1 mile radar separation diagonally between successive aircraft on adjacent final approach courses when runway centerlines are at least 2,500 feet but no more than 3,600 feet apart." |
| **5-9-6 a3** | **1.5 NM**, 3,600–8,300 ft | "…1.5 miles radar separation diagonally … when runway centerlines are more than 3,600 feet but no more than 8,300 feet apart." |
| **5-9-6 a4** | **2 NM**, 8,300–9,000 ft | "…2 miles radar separation diagonally … where runway centerlines are more than 8,300 feet but no more than 9,000 feet apart." |
| 5-9-6 a5 | same final | "Provide the minimum approved radar separation between aircraft on the same final approach course." (REFERENCE: 5-5-4) |
| FIG 5-9-6 EXAMPLE | diagonal and in-trail both hold | "Aircraft 2 is 2 miles from heavy Aircraft 1. Aircraft 3 is a small aircraft and is 6 miles from Aircraft 1. *The resultant separation between Aircraft 2 and 3 is at least 4.7 miles." |
| 5-9-6 b1 | when | "Apply this separation standard only after aircraft are established on the parallel final approach course." |
| 5-9-6 b NOTE | charts; no EoR | "Simultaneous dependent approaches may only be conducted where instrument approach charts specifically authorize simultaneous approaches." / "Established on RNP (EoR) operations are not authorized in conjunction with simultaneous dependent approaches." |
| 5-9-6 b2–b5 | conditions | "Straight‐in landings will be made." "Missed approach procedures do not conflict." Aircraft informed (ATIS); interphone to local control |
| P/CG | definition | SIMULTANEOUS (PARALLEL) DEPENDENT APPROACHES: "…where prescribed diagonal spacing must be maintained. Aircraft are not permitted to pass each other during simultaneous dependent operations." |

**Simultaneous independent approaches, 5-9-7 (dual and triple), 5-9-8 (PRM), 5-9-10 (widely spaced)**:

| ¶ | value | quote |
|---|---|---|
| 5-9-7 a1 | turn-on | "Provide a minimum of 1,000 feet vertical or a minimum of 3 miles radar separation between aircraft : (a) during turn‐on to parallel final approach, or (b) until … established on a published segment of an approach authorized for … (EoR) operations." |
| **5-9-7 a2** | **dual ≥ 3,600 ft** (≥ 3,000 with offset) | "Dual parallel runway centerlines are at least 3,600 feet apart, or dual parallel runway centerlines are at least 3,000 feet apart with a 2.5° to 3.0° offset approach to either runway." |
| 5-9-7 a3 | triple | "(a) Parallel runway centerlines are at least 3,900 feet apart; or (b) … at least 3,000 feet apart, a 2.5° to 3.0° offset approach to both outside runways; or (c) …" |
| 5-9-7 a4 | same final | "Provide the minimum applicable radar separation between aircraft on the same final approach course." |
| **5-9-7 b / b1** | **HUR: dual ≥ 3,100 ft** (≥ 2,500 with offset) | "At locations with high update rate surveillance capable of update rates of 1.2 seconds or faster, and where fusion display mode is utilized…" / "Dual parallel runway centerlines are at least 3,100 feet apart, or … at least 2,500 feet apart with a 2.5° to 3.0° offset approach to either runway." |
| 5-9-7 b3 NOTE 2 | NOZ/NTZ | "Where RCLS is ≤3400 feet, the normal operating zone (NOZ) is constant at 700 feet; and for RCLS ≥3400 feet, the no transgression zone (NTZ) remains constant at 2000 feet." |
| **5-9-7 c1** | FMA required 2,500 to < 4,300 ft | "A color digital display set to a 4 to 1 (4:1) aspect ratio (AR) with visual and aural alerts, such as the STARS final monitor aid (FMA), and a surveillance update rate at 4.8 seconds or faster must be used…" where "Dual parallel runway centerlines are at least 2,500 and less than 4,300 feet apart." |
| 5-9-7 c NOTE | | "At locations where the airfield elevation is 2000 feet or less, FMA is not required to monitor the NTZ for runway centerlines 4,300 feet or greater for dual runways…" |
| 5-9-7 d NOTE | charts | "Simultaneous independent approaches may only be conducted where instrument approach charts specifically authorize simultaneous approaches." |
| 5-9-7 d3 | | "…or when runway centerlines are less than 4,300 feet, PRM approaches are in use, prior to aircraft departing an outer fix." |
| **5-9-8 b** | **PRM < 4,300 ft** | "PRM approaches must be assigned when conducting instrument approaches to dual and triple parallel runways with runway centerlines separated by less than 4,300 feet." |
| 5-9-10 b | widely spaced, no monitors | "…runway centerlines that are separated by more than 9,000 feet with a field elevation at or below 5,000 feet MSL, or 9,200 feet … above 5,000 feet MSL:" |
| 5-9-11 | go-around | "…prior to losing the approved reduced separation, control instructions must be expeditiously issued to increase separation between the applicable aircraft." |
| 7210.3EE 10-4-10 a/b/c | facility side | Same thresholds as 5-9-7 a/b, plus "Instrument approach procedures are annotated with 'simultaneous approach authorized.'" |
| P/CG | definitions | SIMULTANEOUS CLOSE PARALLEL APPROACHES: "…parallel runways separated by at least 3,000 feet and less than 4,300-feet between centerlines. Aircraft are permitted to pass each other…"; SIMULTANEOUS ILS APPROACHES: "…parallel runways separated by at least 4,300 feet between centerlines." |
| AIM 5-4-14 a | pilot side | "…parallel runway centerlines separated by at least 2,500 feet up to 9,000 feet." |

### 2.7 Parallel runways less than 2,500 ft apart

| source | rule | quote |
|---|---|---|
| **5-5-4 h NOTE** (arrivals, TERMINAL) | one runway for wake at the threshold | "Consider parallel runways less than 2,500 feet apart as a single runway because of the possible effects of wake turbulence." |
| 5-5-4 g NOTE 2 | one runway for directly-behind, **< 700 ft** | "Consider runways separated by less than 700 feet as a single runway because of the possible effects of wake turbulence." |
| 5-5-4 f3 NOTE (EN ROUTE) | same as h | "Consider parallel runways less than 2,500 feet apart as a single runway…" |
| 3-10-3 b | wake advisories | "…on the same runway or parallel runways separated by less than 2,500 feet…" |
| **AIM 5-4-14 e** | what applies by default | "The trailing aircraft is permitted reduced diagonal separation, instead of the single runway separation normally utilized for runways spaced less than 2,500 feet apart." |
| AIM 5-4-14 e1 | pairing | "Reduced diagonal spacing is only permitted when certain aircraft wake category pairings exist; typically when the leader is either in the large or small wake turbulence category…" |
| **JO 7110.308E ¶1** | the CSPR exception | "…supplemental criteria to apply … paragraph 5-9-6, Simultaneous Dependent Approaches, to parallel runways separated by less than 2,500 feet, also referred to as Closely Spaced Parallel Runways (CSPR)." |
| 7110.308E 12 c(1) | 1.0 NM | "At airports listed in Appendix A, separation may be reduced to 1.0 nautical mile." |
| 7110.308E 12 b(1)(b) | leader | "A wake category F, G, H, or I aircraft at CWT facilities." The leader must be assigned the lead (lower) approach; 10 b: "the lead approach is listed first and is the lower approach" |
| 7110.308E 12 d | pair to pair | "Provide approved radar separation between the trailing aircraft of one pair and the lead aircraft of the next pair in accordance with … 5-5-4, Minima, subparagraphs g and h." |
| 7110.308E 12 e, i | limits | "Reduced separation is not permitted if either of the aircraft in a reduced separation pair is conducting an instrument approach without vertical guidance." "…down to and including Category I minimums." |
| 7110.308E App. A | **STL listed** | "30R / 30L 1300 ILS / ILS 3.0 / 3.0 89 ft" and "12R /12L 1300 ILS / ILS 3.0 / 3.0 159 ft" (lead / trail, centerline separation ft, glidepath height difference 7 NM from the lead threshold). Also listed: BOS, CLE, EWR, MEM, PHL, SEA, SFO. **SJC is not listed.** App. B (WTMA-P) lists PHL and DTW |
| 5-9-9 a (SOIA) | 750 to < 3,000 ft, by authorization | "…parallel runways that have centerlines separated by at least 750 feet and less than 3,000 feet with one final approach course offset by 2.5 to 3.0 degrees…" |
| 5-9-9 g1/g2 | SOIA wake | "When runways are at least 2,500 feet apart, there are no wake turbulence requirements between aircraft on adjacent final approach courses." "For runways less than 2,500 feet apart, whenever the ceiling is greater than or equal to 500 feet above the MVA, wake vortex spacing … need not be applied." |
| 7-4-4 c1 as amended by **N JO 7110.805** (visual approaches) | | "Parallel runways separated by less than 2,500 feet, determine the preceding and succeeding aircraft, and issue an approach clearance to the succeeding aircraft if the following conditions are met:" (a) "Approved separation must be provided until the preceding aircraft is established on its extended runway centerline…"; (c) "…Do not permit an aircraft to overtake another aircraft when wake turbulence separation is required." |
| 7-4-4 c2 / c3 (visual) | regimes | "Parallel runways separated by 2,500 feet but less than 4,300 feet." / "Parallel runways separated by 4,300 feet or more." |
| P/CG SIDESTEP MANEUVER | geometry | "…a straight‐in landing on a parallel runway not more than 1,200 feet to either side of the runway to which the instrument approach was conducted." |
| N JO 7110.802 (2-1-21 d) | Class D only | "Within Class D airspace when parallel runways are separated by 1,000 feet or less, issue traffic advisories to aircraft on or approaching runway centerlines:" |

### 2.8 Intersecting and converging runways

| source | rule | quote |
|---|---|---|
| **3-10-4 a** | gating at the threshold / flight path | "Separate an arriving aircraft using one runway from another aircraft using an intersecting runway or a nonintersecting runway when the flight paths intersect by ensuring that the arriving aircraft does not cross the landing threshold or flight path of the other aircraft until…" |
| 3-10-4 a1/a2 | conditions | "The preceding aircraft has departed and passed the intersection/flight path or is airborne and turning to avert any conflict." / "A preceding arriving aircraft is clear of the landing runway, completed landing roll and will hold short of the intersection/flight path, or has passed the intersection/flight path." |
| 3-10-4 b | LAHSO | Hold-short landings under JO 7110.118 ("USA/USAF/USN NOT APPLICABLE") |
| **3-10-4 c** | wake, arrival behind departure on a crossing runway | "…by the appropriate radar separation or the following intervals:" "…behind Category A aircraft – 3 minutes." "…behind Category B or D aircraft – 2 minutes." "Category E, F, G, H, or I aircraft behind Category C aircraft – 2 minutes." "Category I aircraft behind Category E aircraft – 2 minutes." |
| 3-9-9 b | converging (departures) | "If the extended centerline of a runway crosses a converging runway or the extended centerline of a converging runway at a distance of 1 NM or less from either departure end, apply the provisions of paragraph 3-9-8…" |
| **7210.3EE 10-4-12 b** (SCIA) | requirements | "Non‐intersecting final approach segments." "Missed approach points (MAP) must be at least 3 nautical miles (NM) apart…" |
| 7210.3EE 10-4-12 b5 | intersecting runways | "Converging approaches must not be conducted simultaneously to runways that intersect, when the ceiling is less than 1,000 feet or the visibility is less than 3 miles." |
| 7210.3EE 10-4-12 b7 | | "Application of this procedure to intersecting runways does not relieve the controller of the responsibility to provide intersecting runways separation as required in … paragraph 3-10-4." |
| **JO 7110.110B ¶8** (DCIA with CRDA) | criteria | "Nonintersecting final approach courses." "Included angle between the runway approach courses of not less than 45 degrees or not greater than 110 degrees." |
| 7110.110B ¶10 f | stagger | "A minimum stagger of 5 nautical miles (NM) or more must be used when the lead aircraft is a heavy aircraft. A minimum stagger of 8 NM or more must be used when the lead aircraft is an Airbus A388 aircraft." Other staggers are in its Appendix 1 Tables 1-A/B/C, by approach minimums |
| 7110.110B ¶10 d | | "Aircraft must not be instructed to hold short of any intersection or intersecting runway while conducting DCIA during IMC." |
| JO 7110.663A ¶2 | CRDA is a tool | "This order provides for the use of CRDA as a planning or reference tool when applying separation. CRDA is not a form of separation." |
| P/CG | definition | SIMULTANEOUS (CONVERGING) DEPENDENT APPROACHES: "…approaches to runways or missed approach courses that intersect where required minimum spacing between the aircraft on each final approach course is required." |
| 7-4-4 c4 (visual) | | "Intersecting and converging runways. Visual approaches may be conducted simultaneously with visual or instrument approaches to other runways, provided:" |

## 3. International counterparts

### 3.1 ICAO Doc 4444 (PANS-ATM)

**Availability.** Doc 4444 is sold, not published: 16th edition, USD 442.00, "Amendments 7-13" on the ICAO
Store page. Its text was **not read**. The paragraph numbers below come only from these free sources:
- **ICAO APAC SAIOSEACG/4 IP/03, March 2025** (official ICAO) states the ANC "approved Amendment 12 to
  the sixteenth edition … for applicability on 28 November 2024". It quotes the new "8.7.3.3 Where the
  communications system used satisfies RCP 240, a horizontal separation minimum based on an ATS
  surveillance system of 28 km (15 NM) may be applied." and "8.7.3.4 The separation minimum or minima
  based on radar and/or ADS-B and/or MLAT systems to be applied shall be prescribed by the appropriate ATS
  authority…". **Amendment 12 therefore renumbered 8.7.3.** Older citations of 8.7.3.x (for example
  SOIR's "8.7.4.4") may point elsewhere today.
- **US AIP GEN 1.7** (official FAA; AIP Amendment 1, effective 07/09/2026) lists US differences against
  "PANS ATM Doc 4444 16th Edition" by paragraph:
  - "8.7.3.2 (b) | The U.S. only allows visual observance of runway turn‐off points."
  - "8.7.3.5 | FAA Consolidated Wake Turbulence (CWT) is based on nine weight groups. FAA distance‐based
    wake turbulence separation minima differs from ICAO standards."
  - 8.7.3.4 and 8.7.3.6, the Super/Heavy 6-mile difference and the < 2,500-ft single-runway note.
  - "6.7.3.2.1 a) Table 6-1 | When conducting Dual and Triple Simultaneous Independent Approaches using
    High Update Rate Surveillance, the FAA allows the minimum distance between runway centerlines to be
    3100 feet."
  - 4.9.1.1/4.9.1.2 (wake categories) and 5.8.x (time-based wake).
- **SKYbrary *Separation Standards*** (secondary): "…the minimum separation prescribed by ICAO Doc 4444
  is 5 nm … may be reduced … but not below: 3 nm when the surveillance systems' capabilities at a given
  location permit this; 2.5 nm between succeeding aircraft which are established on the same final
  approach track within 10 nm of the runway threshold." It adds: "A number of additional criteria must be
  met … (described in detail in ICAO Doc 4444, 8.7.3.2 b))." **Those criteria were not read.**

**ICAO wake categories and minima** as two free sources reproduce them (not checked against Doc 4444):

| | EUROCONTROL RECAT-EU Ed. 2.0, Table 1 "ICAO wake turbulence categories and separation minima" (p. 8) | SKYbrary *Mitigation of Wake Turbulence Hazard* |
|---|---|---|
| categories | "HEAVY MTOM ≥ 136 tons"; "MEDIUM 7 tons ≤ MTOM < 136 tons"; "LIGHT MTOM < 7 tons"; A380-800 as its own row | J (Super) per Doc 8643 (A380-800); H "136 000 kg (300 000 lb) or more"; M "less than 136 000 kg … and more than 7 000 kg"; L "7 000 kg (15 500 lb) or less" (from *ICAO Wake Turbulence Category*) |
| Super/A380 → Heavy, Medium, Light | **6**, 7, 8 NM (Note 1: "an ICAO State guidance released in 2008 recommends an increase…") | **5.0**, 7.0, 8.0 NM ("in the EU … and in the USA … the minimum for HEAVY aircraft after SUPER is defined as 6 NM rather than 5 NM") |
| Heavy → Heavy, Medium, Light | 4, 5, 6 NM | 4.0, 5.0, 6.0 NM |
| Medium → Light | 5 NM | 5.0 NM |
| blank pairs | Note 2: "…minimum radar separation (MRS) being 3NM (or 2.5NM under given conditions described in Doc 4444)" | — |
| where distance minima apply | — | "…if both aircraft are using the same runway or parallel runways separated by less than 760 m…" |
| time-based, arrivals not radar-separated | — | 2 min Medium behind Heavy and Heavy behind Super; 3 min Light behind Heavy or Medium, and Medium behind Super; 4 min Light behind Super |

The two sources **disagree on Super → Heavy (6 vs 5 NM)**. Doc 4444 itself was not read, so the disagreement
is not resolved. SKYbrary also notes that "wake turbulence groups A to G" exist in ICAO as an alternative
approved by the ATS authority. They were not read.

### 3.2 ICAO Doc 9643 SOIR (First Edition 2004; `../runway_assignment/` copy)

| ¶ (printed page) | mode | quote |
|---|---|---|
| 2.2.1.1 a) 1)–3) (2-1/2-2) | Mode 1, independent | "…where runway centre lines are spaced by less than 1 310 m (4 300 ft) but not less than 1 035 m (3 400 ft), suitable SSR equipment, with a minimum azimuth accuracy of 0.06 degrees (one sigma), an update period of 2.5 seconds or less…". Item 3) covers 1 525 m (5 000 ft) or more, with "suitable surveillance radar with a minimum azimuth accuracy of 0.3 degrees … and an update period of 5 seconds or less" |
| 2.2.1.6 (2-3) | Mode 1, same final | "…a minimum of 5.6 km (3.0 NM) radar separation shall be provided between aircraft on the same ILS localizer course … unless increased longitudinal separation is required due to wake turbulence…" |
| 2.3.1.1 (2-8) | Mode 2, dependent | "…a dependent approach procedure may be used when the runways are spaced by 915 m (3 000 ft) or more." |
| 2.3.2.2 (2-9) | Mode 2 minima | "a) 5.6 km (3.0 NM) between aircraft on the same ILS localizer course…" "b) 3.7 km (2.0 NM) between successive aircraft on adjacent ILS localizer courses or MLS final approach tracks" |
| 2.3.2.3 (2-9) | turn-on | "A minimum of 300 m (1 000 ft) vertical separation or a minimum of 5.6 km (3.0 NM) radar separation shall be provided between aircraft during turn-on…" |
| 2.3.3.2 (2-10) | single runway | "At present, for wake turbulence reasons, parallel runways spaced less than 760 m (2 500 ft) apart are considered to be a single runway." |
| Glossary (vii); 5.1.2 (5-1) | near-parallel | "Near-parallel runways. Non-intersecting runways whose extended centre lines have an angle of convergence/divergence of 15 degrees or less." "No special procedures have been developed for simultaneous operations to near-parallel runways." |

SOIR's notes cite PANS-ATM "Chapter 6, 6.7.3.2" (Mode 1), "6.7.3.4" (Mode 2) and "Chapter 8, 8.7.4.4"
(wake). Those are the 2004 paragraph numbers. The current Doc 4444 (Table 6-1, per the US AIP) was not read.

### 3.3 EUROCONTROL RECAT-EU (Edition 2.0, 08/11/2024)

PDF page numbers below equal the printed "Page N of 28".

- **Scheme** (pp. 5 and 13): "…splitting ICAO HEAVY and MEDIUM categories into 'Upper' ('Larger') and
  'Lower' ('Smaller')" and "creates a new SUPER HEAVY one for the AIRBUS A380". That gives six
  categories: A "Super Heavy", B "Upper Heavy", C "Lower Heavy", D "Upper Medium", E "Lower Medium", F
  "Light".
- **Criteria** (Figure 6, p. 14, an image transcribed here):
  - MTOW < 15 t: Light, CAT-F.
  - MTOW < 100 t: Medium. Span < 32 m is CAT-E; span > 32 m is CAT-D.
  - MTOW > 100 t: Heavy. Span < 52 m is CAT-C; 52–60 m needs specific analysis; 60–72 m is CAT-B; 72–80 m
    is CAT-A. Some branches carry an "optional specific analysis".
- **Coverage** (p. 15): "All certificated aircraft types (as per ICAO designators) before 1ST of January 2013 have
  been assigned in RECAT-EU scheme, with examples provided in Table 2." The full list is in Safety Case
  Appendix F (p. 26), which was **not found** in public form.
- **Regulatory hook** (p. 20, quoting EU 2017/373 AMC7 ATS.TR.220): "…an air traffic services provider may
  decide to implement RECAT-EU or parts thereof, subject to the approval of the competent authority."
  Partial and hybrid 6-CAT and 7-CAT schemes are allowed (Figures 9–10, p. 22).

**Table 3, RECAT-EU WT distance-based separation minima on approach and departure** (p. 16; NM; rows are
the leader; "·" = blank in the source):

| leader \ follower | A | B | C | D | E | F |
|---|---|---|---|---|---|---|
| A Super Heavy | 3 | 4 | 5 | 5 | 6 | 8 |
| B Upper Heavy | · | 3 | 4 | 4 | 5 | 7 |
| C Lower Heavy | · | (*) | 3 | 3 | 4 | 6 |
| D Upper Medium | · | · | · | · | · | 5 |
| E Lower Medium | · | · | · | · | · | 4 |
| F Light | · | · | · | · | · | 3 |

"(*) means minimum radar separation (MRS), set at 2.5 NM, is applicable as per current ICAO doc 4444
provisions." Table 4 gives the time-based departure minima (80–180 s).

### 3.4 How they differ from the FAA values

Same runway, on approach. FAA values are TBL 5-5-2 at the threshold; RECAT-EU is Table 3; ICAO is
EUROCONTROL's Table 1. Types map through the 7360.1K CWT and ICAO WTC columns and RECAT-EU Table 2.
Every cell is a table lookup.

| pair (leader → follower) | FAA CWT | RECAT-EU | ICAO (Table 1) |
|---|---|---|---|
| B772 / B789 / A332 → A320 / B738 | B→F **5** | B→D **4** | Heavy→Medium **5** |
| B772 → B789 | B→B **3** | B→B **3** | Heavy→Heavy **4** |
| B763 / MD11 → A320 / B738 | C→F **3.5** | C→D **3** | Heavy→Medium **5** |
| B752 → A320 / B738 | E→F **·** | C→D **3** | Medium→Medium **·** |
| A320 / B738 → E75L / CRJ9 | F→G **·** | D→E **·** | Medium→Medium **·** |
| A320 / B738 → C172 | F→I **4** | D→F **5** | Medium→Light **5** |
| B772 → C172 | B→I **6** | B→F **7** | Heavy→Light **6** |
| A388 → A320 | A→F **7** | A→D **5** | A380→Medium **7** |

- **ICAO vs FAA.** ICAO has three weight categories plus the A380. The FAA terminal scheme has nine
  (A–I), splitting the heavies into three and the 41,000–300,000 lb class into two. ICAO sets 4 NM for
  every Heavy → Heavy pair; FAA CWT sets 3 or 4 NM for B/D → B/C/D and nothing for C → B/C/D. The US AIP
  GEN 1.7 row for 8.7.3.5 states the difference in one line. ICAO's reduced 2.5 NM needs conditions in
  8.7.3.2 b) (not read). FAA's 2.5 NM needs the four conditions of 5-5-4 j, and wake per TBL 5-5-2 still
  applies.
- **RECAT-EU vs FAA.** RECAT-EU has six categories keyed on MTOM and wingspan; FAA CWT has nine. The B757
  is FAA E, with a single non-blank cell (→ I 4 NM), but RECAT-EU C (Lower Heavy). The A320 and B737
  families are FAA F / RECAT-EU D, and behind an upper heavy they get 5 NM (FAA) against 4 NM (RECAT-EU).
  RECAT-EU sets a minimum for A→A (3 NM) and F→F (3 NM) where FAA TBL 5-5-2 is blank. It also names MRS
  = 2.5 NM in its (*) cell.
- **SOIR vs FAA.** ICAO's dependent diagonal is a single 2.0 NM from 915 m (3,000 ft). FAA's is 1.0, 1.5
  or 2 NM by spacing, from 2,500 ft. ICAO independent approaches start at 1,035 m (3,400 ft) with
  high-precision monitoring; FAA's start at 3,600 ft standard or 3,100 ft with HUR. Both treat spacing
  below 2,500 ft (760 m) as one runway for wake.

## 4. Applied to our five airports

Spacings are the project's measurements, given in the brief and in the R3 plan §17.3. FAA-published
centerline separations were **not** checked, except STL, which 7110.308E lists at 1,300 ft. Every regime
the text offers is also **conditional on authorization**, and none of that was checked (§5): charts
annotated for simultaneous approaches, PRM or offset approaches, HUR, FMA, CTRDs and a documented ROT.

| airport, pair | spacing | rule the text puts it under | ambiguous or unverified |
|---|---|---|---|
| **KRDU** 05L/05R, 23L/23R | 3,498 ft | Not one runway for wake (≥ 2,500). **Dependent: 1.0 NM diagonal** (5-9-6 a2, 2,500–3,600). **Independent: below the 3,600 ft of 5-9-7 a2**, so only with a 2.5–3.0° offset approach to one runway (a2, ≥ 3,000), or under HUR (b1, ≥ 3,100) with 1.2 s surveillance and fusion display. In either case FMA is required (c1, < 4,300) and PRM approaches must be assigned (5-9-8 b, < 4,300). NTZ constant at 2,000 ft (RCLS ≥ 3,400). Visual: 7-4-4 c2 | 3,498 is 102 ft below 3,600, the boundary for both the 1.0/1.5 NM dependent split and standard independent operation. Whether KRDU's charts authorize simultaneous, PRM or offset approaches, or whether its TRACON has HUR, was not checked |
| **KSJC** 12L/12R, 30L/30R | 699 ft | **One runway for wake** (5-5-4 h NOTE, < 2,500); wake advisories (3-10-3 b). Also < 700 ft, so one runway for the directly-behind table too (5-5-4 g NOTE 2). **Not in 7110.308E**, so no CSPR diagonal. Below the SOIA lower bound (5-9-9 a, 750 ft). No 5-9-6 or 5-9-7 regime applies. AIM 5-4-14 e: "single runway separation normally utilized". Visual: 7-4-4 c1 as amended by N JO 7110.805. Side-step geometry holds (≤ 1,200 ft) | **699 vs 700 ft is a 1-ft knife edge** for 5-5-4 g NOTE 2, and for the 700-ft departure thresholds in 3-9-6 g/k and 3-9-7 a2. The FAA-published figure was not checked. N JO 7110.802 applies only "within Class D", and SJC's airspace class was not checked |
| **KSTL** 12L/12R, 30L/30R | ≈ 1,290 ft (7110.308E: 1300) | **One runway for wake** (5-5-4 h NOTE); ≥ 700, so g NOTE 2 does not apply. **7110.308E App. A lists both pairs**: lead 12R / trail 12L and lead 30R / trail 30L. The diagonal may be reduced to **1.0 NM** when the leader is CWT F–I on the lower (lead) approach; trail-to-next-lead per 5-5-4 g/h; ILS on both; down to CAT I. Visual: 7-4-4 c1 | Side-step: ≈ 1,290 > 1,200 ft, **outside** the P/CG definition (7110.308E's 1,300 gives the same answer). Whether STL is currently running 7110.308E operations was not checked |
| **KSTL** 11 vs 12R, 29 vs 30L | ≈ 2,750 ft | Not one runway (≥ 2,500). **Dependent: 1.0 NM diagonal** (5-9-6 a2). Independent: no standard dual (< 3,000 even with offset). HUR with a 2.5–3.0° offset (5-9-7 b1, ≥ 2,500) is the only independent route; FMA and PRM would be required. The SOIA geometric range (750 to < 3,000) is met, but SOIA needs an authorization not seen in any text read. Visual: 7-4-4 c2 | (reading) 11/29 and 12R/30L are not an L/R-designated pair, and the P/CG PARALLEL RUNWAYS entry ties designation to being parallel. OurAirports, in the project's `runways.csv`, gives both the same true heading (122°/302°). The order's rules speak of "parallel runway centerlines", and the pair is treated as parallel here on geometry |
| **KSTL** 11 vs 12L, 29 vs 30R | ≈ 4,037 ft | **Dependent: 1.5 NM diagonal** (5-9-6 a3). **Independent: meets 5-9-7 a2** (≥ 3,600); FMA required (c1) and PRM must be assigned (5-9-8 b), both because the spacing is below 4,300. Visual: 7-4-4 c2 | (reading) With 12R's final active between them, whether 11 and 12L count as "adjacent final approach courses" (5-9-6 a) is not stated. The triple rules (5-9-7 a3, ≥ 3,900 or ≥ 3,000 with offsets) cannot be met while the 12R/12L pair is 1,290 ft apart |
| **KSMF** 17L/17R, 35L/35R | 5,982 ft | **Dependent: 1.5 NM** (5-9-6 a3). **Independent: meets 5-9-7 a2**. FMA is not required at a field elevation of 2,000 ft or less for ≥ 4,300 ft (c NOTE); SMF threshold elevations are 22–27 ft in the project's OurAirports data. PRM is not required (≥ 4,300). Not widely spaced (5-9-10 needs > 9,000). Visual: 7-4-4 c3 | Chart authorization ("simultaneous approach authorized") was not checked |
| **KMSY** 02/20 × 11/29 | intersecting; no parallels | **3-10-4 a**: an arrival may not cross the threshold or flight path until the preceding aircraft has passed the intersection or is holding short or clear. **3-10-4 c** gives the wake intervals for an arrival behind a departure on the crossing runway (3 / 2 / 2 / 2 min). SCIA (7210.3EE 10-4-12) is barred to intersecting runways when the ceiling is below 1,000 ft or visibility below 3 mi, and needs non-intersecting final segments with MAPs ≥ 3 NM apart. DCIA (7110.110B) needs an included angle of 45–110°; the project data gives 02 at 015° T and 11 at 106° T, i.e. 91°. Visual: 7-4-4 c4 | Whether MSY uses SCIA or DCIA (both need facility approval) was not checked. (reading) Without either, the two finals are separated by the ordinary radar minima of 5-5-4 up to the 3-10-4 gate |

## 5. Could not verify, and inconsistencies found

1. **ICAO Doc 4444 text** was not read (sold). Its values here come from EUROCONTROL and SKYbrary, which
   disagree on Super → Heavy (6 vs 5 NM). The conditions of 8.7.3.2 b) and the wake turbulence groups
   A–G were not read. Amendment 13 appears on the store page; its content and applicability were not
   checked.
2. **Authorizations at our airports**, none checked:
   - charts annotated for simultaneous approaches;
   - PRM or offset approaches;
   - HUR, FMA and CTRDs;
   - documented ROT for the 2.5 NM rule;
   - current use of 7110.308E at STL;
   - SOIA, SCIA or DCIA anywhere;
   - whether each TRACON applies CWT. 7110.65BB Change 2 makes CWT the terminal standard, but US AIP GEN
     1.7 still says "Not all FAA facilities are authorized to use the provisions of FAA JO 7110.126."
3. **Centerline spacings** are project measurements; the FAA-published values were not checked. The
   knife edges are KSJC 699 vs 700 ft (5-5-4 g NOTE 2), KRDU 3,498 vs 3,600 ft, and KSTL 1,290 vs 1,200
   ft (side-step).
4. **Which NOTE governs a directly-behind arrival pair 700–2,500 ft apart** is not stated. 5-5-4 g, with
   TBL 5-5-1, carries the 700-ft NOTE; h, with TBL 5-5-2, carries the 2,500-ft NOTE. In the Basic the same
   700-ft NOTE sat under the departure item f3.
5. **The seventh notice.** The publications page says "Current Notices (7)"; the notices page lists six.
6. **Table numbering.** The HTML captions read TBL 5-5-3 and 5-5-4 while the text and the PDF read TBL
   5-5-1 and 5-5-2. The values are identical.
7. **Dangling references to the canceled JO 7110.126**:
   - 7360.1K ¶2-11 (category definitions);
   - 7110.308E ¶10 a and its 12 d NOTE;
   - US AIP GEN 1.7 row 4.9.1.2;
   - the P/CG entry "CWT- See CONSOLIDATED WAKE TURBULENCE", which has no target: glossary C has no such
     entry (checked 2026-09-14).
8. **B757 wording differs by document.** P/CG: "CATEGORY E. All B757 aircraft". 7360.1K: B752 and B753 are
   weight class L. US AIP GEN 1.7 row 5.8.4.1: "The U.S. includes B757 in heavy category for wake
   turbulence purposes." These are quoted as printed and not reconciled.
9. **The 41,000-lb boundary is worded two ways in the P/CG.** AIRCRAFT CLASSES has Large at "more than
   41,000 pounds" and Small at "41,000 pounds or less". AIRCRAFT WAKE CATEGORIES has F/G at "41,000
   pounds or more" and H/I at "less than 41,000 pounds".
10. **AIM 7-4-9** (pilot-side summary, `../runway_assignment/` copy, p. 613; revised in AIM Change 2,
    effective 1/22/26, whose Explanation of Changes says it "aligns the Aeronautical Information Manual (AIM) with the FAA
    effort to recategorize the existing fleet") lists:
    - "Heavy behind heavy −3 miles". The legacy text gives 4 miles, both en route (5-5-4 f1(b)) and in the
      Basic's terminal rule. 3 NM equals the CWT cells B→B and D→B, but the AIM states it for every heavy.
    - "Heavy behind super − 5 miles". This matches the general en-route value in 5-5-4 f1(a); the Basic's
      terminal rule gave 6.
    The controller order governs; the AIM is noted, not reconciled.
11. **7360.1K internal disagreements**: 8 designators differ between Appendix A and Appendices B/C (§2.5).
12. **RECAT-EU full type list** (Safety Case Appendix F) was not available. A20N, A21N, B38M, C172 and PC12
    are not in its Table 2 examples, and E75L appears as "E175".
13. **SOIR edition.** The `../runway_assignment/` file is the First Edition (2004), not "2nd ed 2020" as its
    filename and README say. The SOIR paragraph references to PANS-ATM are 2004-era numbers.
14. **JO 7110.126B itself** was not read. It is canceled, and CWT was taken from 7110.65BB, which now holds it.
15. **No text gives a diagonal minimum below 2,500 ft outside 7110.308E and SOIA.** "Single runway
    separation" for that case is stated only in the pilot-side AIM 5-4-14 e. The controller order says
    only "single runway" for wake (5-5-4 h NOTE).

## 6. Layout

| file | what |
|---|---|
| `README.md` | this index |
| `download.sh` | re-fetches everything in `papers/`, cuts the two PDF excerpts with ghostscript, and regenerates the parsed 7360.1K CSV. It does not fetch the `../runway_assignment/` files |
| `papers/7110.65BB_*.html` | 7110.65BB Change 3 sections: 0-0 (Explanation of Changes), 2-1, 3-9, 3-10, 5-5, 5-9, 7-4, plus the index |
| `papers/7210.3EE_*.html` | 7210.3EE Change 3, Ch. 10 Sec. 4, plus the index |
| `papers/PCG_*.html` | P/CG Change 3: letters A, C, P, S, plus the index |
| `papers/FAA_atpubs_publications_index.html`, `FAA_at_notices_*.html` | edition, change and notice evidence |
| `papers/FAA_JO_7110.65BB_Basic_…_EXCERPT_5-5-4_legacy_wake.pdf` | superseded legacy terminal wake text |
| `papers/FAA_JO_7360.1K_…_EXCERPT.pdf`, `…_AppendixA_categories_parsed.csv` | type → CWT / weight / ICAO WTC / SRS |
| `papers/FAA_JO_7110.308E_…pdf`, `…7110.110B…pdf`, `…7110.663A…pdf` | CSPR, DCIA and CRDA orders |
| `papers/FAA_N_JO_7110.80{1..6}_…pdf`, `FAA_N_JO_7360.7_…pdf` | the notices |
| `papers/US_AIP_*.html` | GEN 1.7, differences from ICAO |
| `papers/ICAO_APAC_…IP03…pdf` | the ICAO paper on PANS-ATM Amendment 12 |
| `papers/EUROCONTROL_RECAT-EU_Edition_2.0_2024-11-08.pdf` | RECAT-EU |
| `papers/SKYbrary_*.html` | the three SKYbrary articles |

## 7. Constraints for the R3 scheduler

What a scheduler over several runways would have to enforce, per the text, with sources. **Status**:
- **T**: the value is quoted above from primary text read on 2026-09-14.
- **T+R**: the value is in the text, but the way a scheduler applies it is our reading. Examples are
  turning distance into time with a groundspeed (as Erzberger & Itoh §2.2 do), treating a pair as
  "adjacent", or combining several minima by taking the maximum.
- **N**: not verified.

All distances are in NM at the stated point.

| # | constraint | value per the text | source | status |
|---|---|---|---|---|
| 1 | Same runway, radar minimum between arrivals on final | 3 mi (terminal single-sensor < 40 mi from the antenna, or FUSION); 5 mi at 40 mi or more, with ISR, or in STARS multi-sensor | 5-9-6 a5 / 5-9-7 a4 → 5-5-4 a, b, c | T |
| 2 | Same runway, reduced minimum | 2.5 NM between aircraft established on final within 10 NM of the runway (FUSION, or single-sensor slant range within 40 mi). Only if (1) TBL 5-5-2 wake is applied, (2) average ROT ≤ 50 s is documented over ≥ 250 arrivals, (3) CTRDs are in use, (4) turn-offs are visible from the tower | 5-5-4 j; 7210.3EE 10-4-14 | T. **Whether any of the five airports is authorized: N** |
| 3 | Same runway, wake at the threshold | TBL 5-5-2 (CWT A–I), measured when the leader crosses the threshold | 5-5-4 h | T |
| 4 | Same runway, wake while airborne behind a leader | TBL 5-5-1 when following an aircraft on an instrument approach and/or operating "within 2,500 feet and less than 1,000 feet below the flight path" of A–D ("…and/or less than 500 feet below" for E) | 5-5-4 g | T |
| 5 | Same runway, occupancy | The arrival may not cross the threshold until the preceding arrival is clear of the runway. The daylight distance option applies only to SRS I/II followers, and all our jets are SRS III | 3-10-3 a | T (SRS III consequence: T+R) |
| 6 | Wake holds to touchdown | "…must continue to touchdown for all IFR aircraft not making a visual approach…" | 2-1-19 b | T |
| 7 | Category of each type | CWT column of 7360.1K Appendix A (`papers/…parsed.csv`); legacy class from the weight-class column | 7360.1K; P/CG | T |
| 8 | Parallels < 2,500 ft (KSJC, KSTL 12L/12R) | **One runway for wake**: apply rows 3–4 across both runways. Below 700 ft, TBL 5-5-1 also treats them as one runway | 5-5-4 h NOTE, g NOTE 2; 3-10-3 b | T. Applying the full same-runway radar minimum across the pair: T+R (AIM 5-4-14 e says "single runway separation") |
| 9 | CSPR exception (KSTL only) | Diagonal 1.0 NM within a pair when the leader is CWT F–I on the lower approach (12R, 30R), both on ILS, CAT I or better. Pair to next pair per 5-5-4 g/h | 7110.308E 12 b–e, App. A | T. **Current use at STL: N** |
| 10 | Dependent parallels 2,500–3,600 ft (KRDU, KSTL 11/12R) | Diagonal ≥ 1.0 NM between successive aircraft on adjacent finals, **plus** each final's own in-trail minima (FIG 5-9-6 example). Turn-on: 1,000 ft or 3 mi. Only once established on the final | 5-9-6 a1, a2, a5, b1 | T (conversion to time, and "successive": T+R) |
| 11 | Dependent parallels 3,600–8,300 ft (KSTL 11/12L, KSMF) | Diagonal ≥ 1.5 NM, as above | 5-9-6 a3 | T |
| 12 | Dependent parallels 8,300–9,000 ft | Diagonal ≥ 2 NM | 5-9-6 a4 | T (none of our pairs) |
| 13 | Independent parallels | No cross-runway minimum on final once established. Dual needs ≥ 3,600 ft (≥ 3,000 with offset), or ≥ 3,100 ft (≥ 2,500 with offset) with HUR. FMA and PRM below 4,300 ft. Needs charts authorizing simultaneous approaches. 1,000 ft or 3 mi until established | 5-9-7 a–d; 5-9-8 b; 7210.3EE 10-4-10 | T. **Which of our pairs actually runs independent: N**. By spacing only KSMF and KSTL 11/12L qualify under a2; KRDU only via HUR or offset |
| 14 | Widely spaced, no monitors | > 9,000 ft (field ≤ 5,000 ft MSL) | 5-9-10 b | T (none of our pairs) |
| 15 | Intersecting runways (KMSY) | The arrival may not cross the threshold or flight path until the other aircraft has passed the intersection, is holding short or is clear. Behind a crossing-runway departure: 3 min behind A, 2 min behind B/D, 2 min for E–I behind C, 2 min for I behind E, or radar separation | 3-10-4 a, c | T |
| 16 | Converging / intersecting, simultaneous IMC | SCIA: not to intersecting runways below 1,000 ft ceiling or 3 mi visibility; MAPs ≥ 3 NM apart. DCIA: 45–110°, stagger per tables, ≥ 5 NM behind a heavy, ≥ 8 NM behind an A388 | 7210.3EE 10-4-12; 7110.110B ¶8, ¶10 | T. **Authorization at MSY: N** |
| 17 | Go-around from a reduced-separation pair | Controllers must re-establish approved separation | 5-9-11 | T (not a scheduling constraint) |
| 18 | Visual conditions | For parallels < 2,500 ft, visual approaches with pilot-applied visual separation; no overtaking where wake separation is required. The runway_intent plan measured parallel-runway gaps down to 22–51 s at p5 in our data | 7-4-4 c1 (N JO 7110.805) | T for the rule. **Share of our flights flown visually: N** |
| 19 | Legacy alternative (en route, or the pre-2026 terminal rule) | Super/Heavy/B757/Large/Small minima of §2.4 | 5-5-4 f (Chg 3); Basic 5-5-4 f/g | T (superseded in terminal) |
