# Runway assignment — reading list and index (2026-09-03)

Why this folder exists: the ts_transformer experiments of 2026-09-03 showed that the
threshold-anchored chart is the predictor's only runway knowledge, that a co-temporal
"active configuration" cue recovers the runway *direction* but not the parallel side, and
that the side is worth ~500–800 m of FDE at KRDU. The open design question is **whether
runway assignment belongs inside the thesis's scheduling layer or arrives as an ATC
input**. These documents are what operations, automation programmes and the literature
say about who assigns the runway, when, and on what basis.

The PDFs are **not tracked in git** (~140 MB; `.gitignore`). `./download.sh` re-fetches
every file from the public source recorded below. Everything here is either a government
publication, a NASA/EUROCONTROL/SESAR report, or an author/open-access copy; two ICAO
documents that are sold, not published, are listed with their store links only.

## Start here (the 90-minute version)

1. `official/FAA_Order_JO_7110.65BB…pdf` — Chapter 3 Section 5 *Runway Selection*
   (3-5-1 Selection, 3-5-3 Tailwind components), 4-8-1 *Approach Clearance*,
   5-9-1 / 5-9-2 *Vectors to final approach course / interception*, 3-8-1
   *Sequence/spacing application*, and the Pilot/Controller Glossary entry *Sidestep
   maneuver*. This is the rulebook for the airports in our data: runway-in-use is a
   tower/facility decision by wind and runway-use programme; the approach clearance names
   the runway; the parallel side can change late only by sidestep.
2. `official/FAA_Order_JO_7210.3EE…pdf` — 10-1-6 *Selecting active runways*, 10-1-7 *Use
   of active runways*: the facility-level (not per-aircraft) nature of the choice.
3. `official/FAA_AIM…pdf` — 4-1-13 ATIS (how the runway-in-use reaches the cockpit),
   4-3-2 Airports with an operating control tower, 5-4-19 *Side-step Maneuver*.
4. `nasa_eurocontrol/NASA_TP-2014_Erzberger_Itoh…pdf` — §3.2 *Multiple meter gates to
   one runway*, §3.3 *Multiple terminal gates to multiple runways*, §4.1 *Real-time
   scheduling with limited sequence and runway changes*. The reference design of an
   arrival scheduler in which runway assignment is an explicit, early decision variable.
5. `official/ICAO_Doc_9643_SOIR…pdf` — Chapter 1 (modes 1–4: independent / dependent
   parallel approaches, independent departures, segregated operations), Chapter 2. What
   "parallel runways in use" means procedurally; why the side is decided by ATC.
6. `papers/Ramanujam_Balakrishnan_2015…pdf` — the configuration-selection decision as a
   discrete-choice model of the controller; the formal version of our co-temporal cue.

## What the documents say, in one table

| Layer | Who decides | Timescale | Evidence |
|---|---|---|---|
| Runway configuration (which direction, which runways open) | tower / TRACON facility, by wind, capacity, runway-use programme | tens of minutes to hours | 7110.65 3-5; 7210.3 10-1-6/7; 8400.9; AC 150/5060-5; Doc 4444 7.2; Ramanujam & Avery models |
| Per-aircraft runway + sequence + landing time | approach control, increasingly by a scheduler (TMA/TBFM, FAST/TSAS, AMAN) | 20–40 min before landing, on the STAR / at the meter fix | Erzberger–Itoh TP; Krzeczowski 1995; pFAST 1998; ATD-1 ConOps App. G; AMAN guidelines; E-AMAN spec |
| Late change of the parallel side | approach/tower, by clearance | inside the final approach (sidestep) or by vectoring on downwind | 7110.65 4-8-1, 5-9; AIM 5-4-19; SOIR ch. 2 |
| Trajectory to the assigned runway | pilot / FMS, or our predictor and optimizer | seconds to minutes | this thesis |

Consequence for the thesis: runway assignment is a scheduling decision made with
airport-level information before the terminal ring, never a kinematic inference. If the
system is framed as a scheduling replacement, the runway is an **output of the scheduling
layer** (with sequence and spacing, the multi-flight problem); if it is framed as
decision support, the runway is an **input** (the clearance). Either way it stays outside
the trajectory predictor, which is what the 2026-09-03 experiments found empirically.

## Index

### official/ — regulations and guidance

| File | What it is | Read | Source |
|---|---|---|---|
| FAA_Order_JO_7110.65BB_…_w_Chg1-3_2026-07-09.pdf (927 p) | FAA *Air Traffic Control*, the US controller rulebook; edition BB with changes 1–3 (7 Jul 2026) | 3-5-1…3-5-3; 3-8-1; 4-8-1; 5-9-1, 5-9-2; 7-4-2; glossary "Sidestep maneuver" | faa.gov/documentLibrary/media/Order/7110.65BB_Bsc_w_Chg_1_2_and_3_dtd_7-9-26_Final.pdf |
| FAA_Order_JO_7210.3EE_…_w_Chg1-3_2026-07-09.pdf (728 p) | FAA *Facility Operation and Administration* | 10-1-6 Selecting active runways; 10-1-7 Use of active runways; 10-1-8 | faa.gov/documentLibrary/media/Order/7210.3EE_Bsc_w_Chg_1_2_and_3_dtd_7-9-26.pdf |
| FAA_AIM_…_w_Chg1-3_2026-07-09.pdf (918 p) | *Aeronautical Information Manual*, the pilot-side view | 4-1-13 ATIS; 4-3-2; 4-3-5; 5-4-19 Side-step Maneuver | faa.gov/air_traffic/publications/media/AIM_Basic_w_Chg_1_and_2_and_3_dtd_7-9-26_FINAL.pdf |
| FAA_Order_8400.9_Runway_Use_Programs.pdf (8 p) | *National Safety and Operational Criteria for Runway Use Programs* (1981, still in force): noise-driven preferential runways that override the wind rule | all | faa.gov/documentLibrary/media/Order/8400-9.pdf |
| FAA_AC_150-5060-5_Airport_Capacity_and_Delay.pdf (153 p) | Advisory Circular on runway-configuration capacity | Ch. 2–3 (capacity by configuration), Fig. 2-1 configuration diagrams | faa.gov/documentlibrary/media/advisory_circular/150_5060_5.pdf |
| ICAO_Doc_9643_SOIR_…_2nd_ed_2020.pdf (40 p) | ICAO *Manual on Simultaneous Operations on Parallel or Near-Parallel Instrument Runways* | Ch. 1 modes of operation; Ch. 2 simultaneous approaches; Ch. 5 near-parallel | skybrary.aero/sites/default/files/bookshelf/4647.pdf |

Not downloadable from an authorised public source (listed for completeness):

| Document | Why it matters | Where |
|---|---|---|
| ICAO Doc 4444 PANS-ATM, 16th ed. | 7.2 *Selection of runway-in-use* (the global baseline for 7110.65 3-5); 6.5 arriving aircraft; 8.9 vectoring to final | store.icao.int (USD 442); university library |
| ICAO Doc 9426 ATS Planning Manual (1984) | Part II runway capacity and configuration planning | store.icao.int; some national CAA libraries host it |

### nasa_eurocontrol/ — arrival-management automation

| File | What it is | Read | Source |
|---|---|---|---|
| NASA_TP-2014_Erzberger_Itoh_Arrival_Scheduling_Design_Principles.pdf (48 p) | *Design Principles and Algorithms for Air Traffic Arrival Scheduling* (NASA/TP-2014-218302): the TMA/TBFM scheduler, runway assignment as a scheduling variable | §3.2, §3.3, §4.1–4.3 | ntrs.nasa.gov/citations/20140010277 |
| NASA_1989_Erzberger_Nedell_Automated_Management_of_Arrival_Traffic.pdf (50 p) | NASA TM 102201, the CTAS origin: arrival scheduling architecture (TMA + FAST) | §2–4 | ntrs.nasa.gov/citations/19890014919 |
| NASA_1989_Design_of_FAST_for_TRACON.pdf (27 p) | NASA TM 102229, the first FAST design (sequencing + runway allocation in the TRACON) | all | ntrs.nasa.gov/citations/19900001525 |
| NASA_1995_Krzeczowski_Davis_Erzberger_Knowledge-based_Scheduling_of_Arrival_Aircraft_AIAA-95-3366.pdf (10 p) | Knowledge-based scheduler that "sequences, assigns landing times, and assigns runways" — the FAST runway-assignment logic | all | ntrs.nasa.gov/citations/19960016183 |
| NASA_1995_Development_of_FAST_Controller-Engineer_Design.pdf (12 p) | NASA TM 110359, how the FAST rules were derived with controllers | all | ntrs.nasa.gov/citations/19960001956 |
| NASA_1995_ATC_Automation_Closely_Spaced_Parallel_Runways.pdf (12 p) | AIAA-95-3367, automation effects on precision approaches to closely spaced parallel runways | all (our KSJC case) | ntrs.nasa.gov/citations/19960016111 |
| NASA_1998_Passive_Final_Approach_Spacing_Tool_pFAST.pdf (44 p) | Human-factors assessment of the pFAST operational evaluation at DFW: runway/sequence advisories in use | §1–3 | ntrs.nasa.gov/citations/19980237029 |
| NASA_2014_ATD-1_Concept_of_Operations_v2.pdf (83 p) | ATD-1 ConOps: TMA-TM + CMS + FIM; **Appendix G.1 "Route and runway assignment by en route controller"** (p. 66) | §2–3, App. G | ntrs.nasa.gov/citations/20140001370 |
| NASA_2015_TSAS_NextGen_on_STARS.pdf (16 p) | TSAS (Terminal Sequencing and Spacing) on the FAA STARS display | all | ntrs.nasa.gov/citations/20150001418 |
| EUROCONTROL_2010_Arrival_Manager_Implementation_Guidelines_and_Lessons_Learned.pdf (107 p) | AMAN guidelines: runway allocation rules as AMAN configuration input | sections on sequencing constraints and runway allocation (grep "runway allocation") | skybrary.aero/bookshelf/…-17-december-2010 |
| EUROCONTROL_2017_Airport_CDM_Implementation_Manual_v5.pdf (363 p) | A-CDM manual; runway configuration as a shared planning element | Annex 7 §5.9 Runway configuration; capacity sections | eurocontrol.int/publication/airport-collaborative-decision-making-cdm-implementation-manual |
| SESAR_Solution05_Extended_AMAN_Technical_Specification.pdf (49 p) | E-AMAN technical specification: "Arrival runway allocation" as an explicit AMAN function | grep "runway allocation" | sesarju.eu |
| SESAR_Extended_AMAN_Factsheet.pdf (2 p) | E-AMAN horizon (180–200 NM) in one page | all | sesarju.eu |

Not downloadable (no public file): Davis, Krzeczowski & Bergh, *The Final Approach Spacing
Tool*, 1994 (NTRS 20010048880, record only); Robinson, Davis & Isaacson, *Fuzzy
reasoning-based sequencing of arrival aircraft in the terminal area*, AIAA GNC 1997.

### papers/ — operations research and data-driven models

| File | What it is | Read | Source |
|---|---|---|---|
| Lee_Balakrishnan_2008_Tradeoffs_in_Scheduling_Terminal-Area_Operations_ProcIEEE.pdf (15 p) | Lee & Balakrishnan, *A Study of Tradeoffs in Scheduling Terminal-Area Operations*, Proc. IEEE 96(12), 2008: throughput / delay / fuel tradeoffs of terminal-area scheduling incl. runway assignment | all | dspace.mit.edu handle 1721.1/51953 |
| Lee_2008_MIT_thesis_Tradeoff_Evaluation_of_Scheduling_Algorithms_Terminal_Area.pdf (120 p) | H. Lee, MIT SM thesis behind the paper above; the CPS scheduling formulations in full | Ch. 2–4 | dspace.mit.edu handle 1721.1/45254 |
| Ramanujam_Balakrishnan_2015_Airport_Configuration_Selection_THMS.pdf (10 p) | Ramanujam & Balakrishnan, *Data-driven modeling of the airport configuration selection process*, IEEE THMS 45(4), 2015 | all | mit.edu/~hamsa/pubs |
| Avery_Balakrishnan_2015_Predicting_Runway_Configuration_ATMSeminar.pdf (10 p) | Avery & Balakrishnan, *Predicting airport runway configuration: a discrete-choice modeling approach*, 11th ATM R&D Seminar, 2015 | all | web.mit.edu/hamsa/www/pubs |

Not downloadable (paywalled; library access by DOI):

| Paper | Why it matters | DOI / venue |
|---|---|---|
| Beasley, Krishnamoorthy, Sharaiha, Abramson, *Scheduling aircraft landings — the static case*, Transportation Science 34(2):180–197, 2000 | the Aircraft Landing Problem; the multiple-runway MIP makes runway assignment a decision variable | 10.1287/trsc.34.2.180.12302 |
| Bennell, Mesgarpour, Potts, *Airport runway scheduling*, 4OR 9:115–138 (2011); Annals of OR 204:249–270 (2013) | the survey of the field | 10.1007/s10288-011-0172-x; 10.1007/s10479-012-1268-1 |
| Balakrishnan & Chandran, *Algorithms for scheduling runway operations under constrained position shifting*, Operations Research 58(6):1650–1665, 2010 | CPS runway scheduling; the tractable formulation the thesis's scheduling layer could adopt | 10.1287/opre.1100.0869 |
| Samà, D'Ariano, Pacciarelli, *Rolling horizon approach for aircraft scheduling in the terminal control area of busy airports*, Transportation Research E 60:140–155, 2013; D'Ariano et al., *Optimal aircraft scheduling and routing at a terminal control area during disturbances*, Transportation Research C 47:61–85, 2014 | joint runway, route and time decisions inside the TMA — closest to this thesis's optimizer scope | Elsevier |

## Reading guide by question

**Q1. Who assigns the runway, and can it still change inside 5 km?** 7110.65 3-5 →
4-8-1 → 5-9-1/5-9-2 → AIM 5-4-19 → SOIR ch. 1–2. Answer you should come out with: the
configuration is a facility decision; the approach clearance fixes the runway before the
final approach course is intercepted; inside the FAF the only changes are sidestep or
go-around. (Our data: 53 % of KRDU flights are geometrically committed beyond 20 km, 26 %
only inside 5 km because their base turn is late; 13 % at the 25 km ring still look like
the sibling runway.)

**Q2. How does automation assign runways?** Erzberger & Nedell 1989 (architecture) →
Erzberger & Itoh 2014 §3.3, §4.1 (runway assignment inside the scheduler, when it may
change) → Krzeczowski 1995 and pFAST 1998 (the terminal-area tool that actually assigned
runways at DFW) → ATD-1 ConOps App. G → AMAN guidelines and the E-AMAN specification
(European equivalent: "arrival runway allocation" as an AMAN function fed by rules).

**Q3. As an optimization problem, what does runway assignment look like?** Lee &
Balakrishnan 2008 → Lee thesis ch. 2–3 → (library) Beasley 2000, Balakrishnan & Chandran
2010, Bennell survey, D'Ariano 2013/2014. Runway is a decision variable coupled to
sequence and separation; the objective is throughput/delay/fuel, never a single
aircraft's trajectory error.

**Q4. Can the runway or configuration be predicted from data?** Ramanujam & Balakrishnan
2015 → Avery & Balakrishnan 2015. Both predict the *configuration* from wind, demand and
the current configuration — the airport-level state our co-temporal-landings rule proxies.
Neither predicts a single aircraft's parallel side, which is consistent with our finding
that the side is not in the data available at the ring.

**Q5. Parallel runways specifically.** SOIR ch. 1–2 (independent vs dependent approaches,
the separation regimes), NASA 1995 closely-spaced-parallel paper (automation effects), and
7110.65 5-9 for the vectoring rules onto parallel finals.

## Suggested order and time

| Session | Read | Time |
|---|---|---|
| 1 | 7110.65 3-5, 4-8-1, 5-9-1/2; AIM 5-4-19, 4-1-13 | 1 h |
| 2 | Erzberger & Itoh 2014 §1–4 | 2 h |
| 3 | SOIR ch. 1–2; NASA 1995 parallel runways | 1.5 h |
| 4 | Lee & Balakrishnan 2008; skim the thesis formulations | 2 h |
| 5 | Ramanujam 2015; Avery 2015 | 1.5 h |
| 6 | ATD-1 ConOps §2–3 + App. G; AMAN guidelines; E-AMAN spec (runway allocation) | 2 h |

## Related thesis documents

- `4dTrajectory/ts_transformer/docs/2026-09-03_airport_frame_ablation_results.md`
- `4dTrajectory/ts_transformer/docs/2026-09-03_runway_hypothesis_expansion.md`
- `4dTrajectory/ts_transformer/docs/2026-09-03_krdu_nw_endpoint_bias.md`
- `4dTrajectory/ts_transformer/docs/2026-09-03_state_v2_anchor_relative_results.md`
