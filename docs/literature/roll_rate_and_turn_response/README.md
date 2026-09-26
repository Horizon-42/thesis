# Roll rates of transport aircraft and the delay from a heading instruction to the turn (retrieved 2026-09-26)

This folder collects primary sources for two parameters of the thesis executor (the autopilot that
flies controller-like "words"):

- **p = 8°/s**, the roll rate, derived as 32° maximum bank ÷ 4 s;
- **the 4 s lead**: a heading word is said 4 s before the aircraft's track reaches that heading, and
  the executor completes the turn to it in 4 s.

It covers four questions: (1) roll rates and bank in normal airline operations and under the
autopilot / flight director; (2) certification roll-performance requirements for large transports;
(3) the reaction and bank-establishment allowances used in procedure design; (4) the measured delay
from a controller's heading instruction to readback, to the start of the turn and to its completion.

How to read it:
- **Quotes are copied** from the files in `papers/`, each under about 30 words; "…" marks a cut.
  Section, paragraph, table or page follows each quote.
- **"(reading)"** marks a step the source does not state: a unit conversion, an average computed
  from a quoted time, or what a value means for the executor.
- Units: the sources' own units are kept inside quotes; SI (deg/s, s, m) is given next to them.
  1 ft = 0.3048 m, 1 kt = 0.514444 m/s.
- The downloaded files are **not tracked in git** (PDFs are git-ignored). `bash download.sh`
  re-fetches every publicly downloadable source into `papers/`, never overwrites an existing file,
  and prints the SHA-256 of each; compare with §5.
- "Capability" (what the aircraft must be able to do with a full, abrupt control input) and "usage"
  (what pilots or autopilots actually do) are kept apart throughout. Most published roll numbers
  are capability numbers.

## At a glance

| item | value(s), SI | source | paragraph | what it means for p = 8°/s and the 4 s lead (reading) |
|---|---|---|---|---|
| Autopilot heading-select roll-rate cap (USAF flight-control spec, applies to large aircraft, Class III) | roll rate ≤ 10°/s, roll acceleration ≤ 5°/s² | MIL-F-9490D (1975) | §3.1.2.3 | p = 8°/s is under the rate cap. But rolling 0 → 32° under a 5°/s² acceleration cap takes at least 5.1 s, not 4 s (§1.4) |
| Roll rate of one autopilot's heading/track-hold law (Honeywell patent) | 2–3°/s during heading manoeuvres, 2°/s after capture | US 5,023,796 (1991) | col. 8, l. 25–29 | 3–4 times slower than p. One patented design, not a certified-system limit |
| Heading-select guidance in the flight-guidance-system AC / AMC | no number: "consistent with occupant comfort". Typical normal limit 35° roll (said of CWS) | FAA AC 25.1329-1C Chg 2; EASA AMC No. 1 to CS 25.1329 | AC ¶63.b, ¶68.b.(2) Note 1; AMC 11.1.2 | No certification number constrains p. 32° max bank is below the 35° "normal operational limit" |
| Bank reached by the autopilot in simulator heading turns (ATC breakouts) | A320: 23.7–25.6°; B747-400: about 27° | MIT LL ATC-265, ATC-263 (1998) | ATC-265 §4.4.3; ATC-263 §5.4.2 | The executor's 32° is 5–8° steeper than the autopilots observed |
| Roll rates pilots chose for transport manoeuvring (airborne simulator, cruise) | normal: 3–6°/s, mean ≈ 5°/s; fast: 10–25°/s, mean ≈ 17°/s | NASA TN D-5957 (1970) | "Roll-rate criteria", p. 14 | p = 8°/s lies between "normal" and "fast" manual rolling |
| A320 fly-by-wire manual normal law | full sidestick = 15°/s roll-rate demand | A319/A320/A321 FCOM 1.27.20 P 6 (NTSB docket DCA09MA026, Att. 23) | 1.27.20 P 6 | p is about half the full-stick rate |
| "Normally encountered" lateral control position used for jam analysis | control position for a steady 12°/s roll (≤ 50 % of input) | FAA AC 25.671-1 (2024); EASA AMC 25.671 | AC ¶8.3.2.2.1; AMC 7.b.(1)(ii) | The authorities treat up to 12°/s as normal in-flight use; p is below it |
| Certification roll capability, one engine out (FAA and EASA) | 30° → 30° the other way (60°) in ≤ 11 s at V2, i.e. ≥ 5.5°/s average (reading) | FAA AC 25-7D Chg 1; EASA AMC 25.147(d) | AC ¶5.3.2.4.2; AMC 25.147(d) | Minimum capability, full control. p exceeds it |
| Certification roll capability, all engines (EASA only; the FAA test is qualitative) | 60° reversal in ≤ 7 s, en route and approach, i.e. ≥ 8.6°/s average (reading) | EASA AMC 25.147(f); FAA AC 25-7D ¶5.3.2.6 | AMC 25.147(f) | p ≈ the **minimum all-engines capability** demanded with a full, abrupt input: the executor rolls as hard as the certification test does |
| Certification roll capability near VDF/MDF | 20° → 20° (40°) in ≤ 8 s, i.e. ≥ 5°/s (reading) | EASA AMC 25.253(a)(4) | AMC 25.253(a)(4) | Capability; below p |
| Military handling qualities, large aircraft (Class III), time to 30° bank change | Level 1: 2.0–2.3 s (cruise, Cat. B), 2.5 s (terminal, Cat. C), i.e. 12–15°/s average; Level 3, Cat. C: 6.0 s (5°/s) | MIL-F-8785C (1980) | §3.3.4.2, Table IXf | Capability with abrupt input; the aircraft can roll faster than p, which says nothing about how fast they are rolled in service |
| PANS-OPS turn construction: reaction and bank establishment | departure / missed approach: 3 s reaction + 3 s bank; approach turns, holding, reversal: 6 s + 5 s; en route: 10 s + 5 s; fly-by stabilisation: 5 s bank; flyover: 10 s | ICAO Doc 8168 Vol II (5th ed., Amdt 3) | Part I-2-3 Table I-2-3-1; I-3-3 3.3.4; I-4-3 3.6.2; I-4-6 6.4.3; II-3-1 1.4.3.3; II-4-1 1.3.9.1; III-2-1 1.4 | 25° in 5 s or 15° in 3 s = 5°/s average (reading); the executor's 32° in 4 s is faster. Design also allows 3–10 s of reaction **before** the roll starts; the 4 s lead has none |
| PANS-OPS bank angles and rate of turn | 15° (departure, missed approach), 20° (circling), 25° (approach, holding); rate of turn ≤ 3°/s | Doc 8168 Vol II | I-2-3 3.1.2.2 b), Table I-2-3-1 note 2 | 32° is steeper than any PANS-OPS design bank |
| FAA TERPS, simultaneous offset approaches (SOIA) | roll-in / roll-out nominally 3°/s; up to 5°/s and 25° bank allowed | FAA Order 8260.3G (2024) | App. E, Sec. 4, ¶6.a | p is 1.6–2.7 times the FAA's design roll rate. At 3°/s, rolling out of 32° takes 10.7 s, not 4 s |
| FAA TERPS holding; FAA PBN flyover turns | 6 s to recognise and react to fix passage; 6 s of flight for "pilot reaction time and roll-in time" | Order 8260.3G; Order 8260.58D (2025) | 8260.3G ¶16-1-3.b.(3); 8260.58D ¶1-2-5.d.(2)(a), Formula 1-2-12 | The FAA allows 6 s from the cue to the turn; the executor's 4 s covers the whole roll-out |
| En-route voice loop: turn instructions (not for traffic), 3 ARTCCs, 46 h of tapes | controller message 4.6 s mean; end of message → pilot reply 2.7 s mean; start of message → end of correct readback: mean 10.0 s, median 8 s, 90 % ≤ 16 s, 95 % ≤ 21 s (n = 250) | Cardosi & Boole, DOT/FAA/RD-91/20 (1991) | §3.2, Table 3-2 | The readback alone ends about 8 s (median) after the controller starts speaking; nothing has turned yet |
| Approach (KLAX), voice + ADS-B | end of clearance → heading changed by 2.5°: mean 16.7 s, SD 6.5, P25 10 s, P75 21 s; → within ±5° of the cleared heading: mean 69.3 s (P25 45, P75 81); readback delay 0.64 s | Lutz, Chatterji & Idris, NASA / AIAA 2022 | Table 5; §IV.B–C | In service the aircraft **reaches** a vectored heading about a minute after the word, and the first 2.5° takes ~17 s. A 4 s lead is roll-out anticipation, not an ATC-to-compliance model |
| Swedish airspace (SCAT), ATM-system clearances + ADS-B, 831,311 clearances | initiation delay (clearance entered → 5 % of the change done), all types: peak 10.75 s, mean 20.5 s (< 90 s); heading clearances fastest; heading changes "usually within 50 s" | Wüstenbecker et al., CEAS Aeronaut. J. (2025) | §2.1, §4.1, §4.1.1, §4.1.2 | Same picture: seconds-to-tens-of-seconds before the turn starts |
| Full-flight-simulator breakouts, "TURN … IMMEDIATELY" (urgent), start of instruction → roll ≥ 3° | A320 autopilot: median 6.7–6.9 s (max 10.4–16.8 s) above 400 ft (122 m) AGL and in descending breakouts, 11.5 s at decision altitude; hand-flown: median 4.5–5.1 s. B747-400 autopilot: roll ≥ 3° comes 3.3 s (mean) after the heading is set on the MCP; heading has changed 3° a further 3.3 s later | MIT LL ATC-265 (A320), ATC-263 (B747-400) (1998) | ATC-265 Fig. 4-5; ATC-263 §5.4.2 | Even urgent, expected instructions take ≈ 5–7 s to the first 3° of bank. From the autopilot's own heading input, the first 3° of heading change took ≈ 6.6 s in the B747-400 |
| PRM risk-analysis acceptance criterion for breakout response | start of instruction → start of turn: mean ≤ 8 s, maximum ≤ 17 s | ATC-265, ATC-263 (from the 1991 PRM programme) | ATC-265 Executive Summary, §1.1.3 | The FAA's own safety case assumed up to 8 s mean before the turn begins |
| EUROCONTROL STCA design parameter | "standard turn reaction time" 50–70 s (controller + communication + pilot + aircraft) | EUROCONTROL-GUID-159 Part III (2017) | §3.10, Table 10 | An alerting allowance, not a measurement; shows the whole loop is budgeted in tens of seconds |

### Reading for the executor (summary)

- **p = 8°/s sits at the fast end of everything found.** It is about equal to the **minimum
  all-engines roll capability** EASA demands with a full, abrupt control input (60° in 7 s), above
  the rates procedure designers assume (3–5°/s: TERPS SOIA, PANS-OPS bank-establishment times), above
  the "normal" manual roll rate pilots chose in the 1970 NASA study (≈ 5°/s) and the one autopilot
  design found (2–3°/s), and below the USAF heading-select cap (10°/s), the FAA/EASA "normally
  encountered" 12°/s and the A320 full-stick 15°/s. No airliner FCOM value for the autopilot's roll
  rate in HDG SEL / LNAV was found in a public primary source (§6).
- **The instantaneous 0 → 8°/s start** has no counterpart: the only published autopilot acceleration
  limit (MIL-F-9490D, 5°/s²) makes 0 → 32° take at least 5.1 s (§1.4).
- **32° bank** is steeper than every design value (PANS-OPS 25°) and than what the A320 and B747-400
  autopilots flew in the breakout simulations (≈ 25–27°), and below the 35° "normal operational
  limit" of AC 25.1329-1C.
- **The 4 s lead** is short against every measured delay. From the start of a controller's
  instruction, the first 3° of bank took a median of 4.5–6.9 s even in urgent simulator breakouts (11.5 s on
  the A320 autopilot at decision altitude);
  in service a 2.5° heading change is seen 10–21 s (interquartile, KLAX) after the end of the
  clearance, and the cleared heading is reached after about 45–81 s. Procedure design allows 3–10 s
  of reaction plus 3–5 s of bank establishment. If the word's timestamp is meant as the moment the
  autopilot target changes (not the moment the controller speaks), the B747-400 data still give
  ≈ 3.3 s to 3° of bank and ≈ 6.6 s to 3° of heading change. So the 4 s lead reproduces a fast
  roll-out, not the instruction-to-turn delay of real operations.

## 1. Roll rates and bank in normal operations and under the autopilot / flight director

### 1.1 What pilots actually use (manual flight): NASA TN D-5957

Holleman, E. C., *Flight Investigation of the Roll Requirements for Transport Airplanes in
Cruising Flight*, NASA TN D-5957, NASA Flight Research Center, September 1970. Setting: the
variable-stability Lockheed JetStar ("general purpose airborne simulator") at Mach 0.55 and about
20,000 ft (6,100 m), test pilots rating simulated transport roll dynamics.

> "The normal roll rates are concentrated about 3 to 6 deg/sec, with a mean of about 5 deg/sec."
> — "Roll-rate criteria", p. 14

> "The fast roll rates, as fast as would normally be used, are concentrated from 10 to 25 deg/sec
> roll rate, with a mean of about 17 deg/sec." — p. 14

The pilots were "asked to demonstrate normal and fast roll-rate maneuvers that they considered to
be acceptable for transport operation" (p. 14). Kind of value: **demonstrated usage distribution**
(histograms, Fig. 30), manual flight, cruise, a 1960s airframe; the report also notes "Transport
pilots seldom use high roll rate" (p. 13). The same report's capability findings (a steady roll
rate of about 15–20°/s needed for satisfactory ratings; "a time to bank 30° of about 1.5 seconds
would be satisfactory") are in §2.4.

Reading: p = 8°/s is 1.6 times the mean "normal" rate and about half the mean "fast" rate.

### 1.2 Airbus A320 family, manual normal law (the upper end of commanded roll rate)

A319/A320/A321 Flight Crew Operating Manual, section 1.27.20 "Flight Controls – Normal Law", P 6,
REV 40 (US Airways issue of the Airbus FCOM), released as NTSB docket DCA09MA026 (US Airways 1549),
Operations/Human Performance Attachment 23. The page is a scan; it was read from the rendered image.

> "The roll rate requested by the pilot during flight is proportional to the sidestick deflection,
> with a maximum rate of 15° per second when the sidestick is at the stop." — 1.27.20 P 6

> "If the bank angle exceeds 45°, the autopilot disconnects and the FD bars disappear." — 1.27.20 P 6

Same page: with the sidestick neutral the system holds bank up to 33° and returns the aircraft to
33° from a steeper bank; 67° is the maximum. Kind of value: **control-law limit** (manual flight).
Aircraft: A319/A320/A321. Reading: p is about half of full-stick.

### 1.3 Autopilot heading modes: what certification guidance says

FAA AC 25.1329-1C, *Approval of Flight Guidance Systems*, Change 2 (02/13/2026), and the harmonised
EASA AMC No. 1 to CS 25.1329 (CS-25 Amendment 28), §11.1.2:

> "the FGS should expeditiously acquire and maintain a 'selected' heading or track value consistent
> with occupant comfort." — AC ¶63.b (p. 46); same words in AMC No. 1 to CS 25.1329 §11.1.2

> "Normal operational limits are typically 35 degrees in roll and +20 degrees to –10 degrees in
> pitch." — AC ¶68.b.(2), Note 1 (pp. 53–54), said of control-wheel steering (CWS)

Neither document gives a roll rate or a heading-select bank limit. Kind of value: **qualitative
guidance** plus a typical attitude limit.

### 1.4 A roll-rate and roll-acceleration cap for heading select: MIL-F-9490D

MIL-F-9490D (USAF), *Flight Control Systems – Design, Installation and Test of Piloted Aircraft,
General Specification for*, 6 June 1975. A military specification, but its "Class III" means large,
heavy aircraft (the transport class of MIL-F-8785). Copy from the everyspec.com mirror.

> "The roll rate shall not exceed 10 deg/sec and roll acceleration shall not exceed 5 deg/sec/sec
> for MIL-F-8785 Classes I, II and III aircraft" — §3.1.2.3 "Heading select"

> "The contractor shall determine a bank angle limit which provides a satisfactory turn rate and
> precludes impending stall." — §3.1.2.3

Kind of value: **limit** (upper bound on the automatic heading-select manoeuvre). Aircraft: USAF
aircraft, Classes I–III.

Reading (arithmetic, not in the source): with roll acceleration capped at 5°/s², the fastest
0 → 32° bank change (accelerate, then decelerate) takes 2·√(32/5) = 5.06 s and peaks at 12.6°/s,
which the 10°/s cap then lowers: 5.2 s with both caps. With p = 8°/s and the 5°/s² cap, 0 → 32°
takes 1.6 s + 2.4 s + 1.6 s = 5.6 s. The executor's 4 s assumes infinite roll acceleration.

### 1.5 One autopilot design: Honeywell patent US 5,023,796

Kahler, J. A. (Honeywell Inc.), *Flight control apparatus with predictive heading/track hold
command*, US Patent 5,023,796, filed 3 Oct 1989, issued 11 June 1991. The heading/track hold mode
here also captures "a heading or track presented on the flight control panel" (col. 8, l. 10–14).

> "In the present invention, roll rates during heading maneuvers are limited to 2° or 3° per second
> depending on conditions or less and are limited to 2° per second after the reference has been
> captured" — col. 8, l. 25–29

In the patent's simulation (col. 8, l. 56–61) the aircraft rolled out from 12.5° to wings level
"within about 15 seconds", and from 25° settled within 30 s. Kind of value: **design limit of one
patented embodiment**, not a statement about any certified system.

Reading: this design's roll-out is exponential and several times slower than the executor's 4 s
constant-rate roll-out.

### 1.6 Bank angle actually flown by autopilots (simulator)

MIT Lincoln Laboratory, full-flight-simulator studies of ATC-directed breakouts (§4.4 has the
setting):

> "Maximum roll magnitude was consistently between 23.7 and 25.6 degrees for breakouts in which the
> autopilot remained engaged, except during engine-out scenarios." — ATC-265 (A320) §4.4.3

> "When the autopilot remained on during the turn, the maximum roll angle was consistently around
> 27 degrees." — ATC-263 (B747-400) §5.4.2

Hand-flown turns varied more; some exceeded 30° (ATC-265: 29 trials; ATC-263: 5–11 % of manual
breakouts per phase). Kind of value: **measured** maximum bank, heading turns of large magnitude.

Reading: consistent with a 25° autopilot bank limit on the A320 in this mode; the executor's 32°
maximum is steeper than both autopilots flew.

### 1.7 FAA/EASA: the 12°/s "normally encountered" roll

FAA AC 25.671-1, *Control Systems—General*, 08/30/2024, ¶8.3 "Determination of Normally
Encountered Flight Control Surface or Pilot Control Positions"; the same criterion is in EASA
AMC 25.671 §7.b.(1)(ii) (CS-25 Amdt 28, p. 380).

> "The lateral control position to sustain a 12 degree-per-second steady roll rate from 1.23 VSR1 to
> VMO/MMO or VFE, as appropriate, but not greater than 50 percent of the control input." — AC ¶8.3.2.2.1

¶8.3.1.1 says these positions "may be considered normally encountered positions for compliance with
§ 25.671(c)(3)". Kind of value: **regulatory assumption** of the largest normal in-flight lateral
input. Reading: the FAA and EASA treat a steady 12°/s roll as within normal use; p = 8°/s is below.

## 2. Certification roll performance for large transports

### 2.1 The rule: 14 CFR 25.147 (no number)

14 CFR 25.147 as in force on 2026-09-01 (eCFR versioner API; `papers/eCFR_14CFR_25.147_2026-09-01.xml`).
The rule has no time or rate:

> "Lateral control must be enough at any speed up to VFC/MFC to provide a peak roll rate necessary
> for safety, without excessive control forces or travel." — § 25.147(f)

§ 25.147(d) says the same for one engine inoperative ("a roll rate necessary for safety"). CS 25.147
(CS-25 Amendment 28) has the same text. The numbers are in the advisory material.

### 2.2 FAA AC 25-7D Change 1 (09/16/2025): one engine inoperative, 11 s

FAA AC 25-7D, *Flight Test Guide for Certification of Transport Category Airplanes*, 05/04/2018,
Change 1 dated 09/16/2025 (Chapter 5 pages 5-20 to 5-23 are unchanged, dated 05/04/18).

> "establish a steady 30° banked turn. Demonstrate that the airplane can be rolled to a 30° bank angle
> in the other direction in not more than 11 seconds." — ¶5.3.2.4.2 (§ 25.147(d))

Conditions (¶5.3.2.4.1): maximum takeoff weight, aft c.g., critical takeoff flaps, gear up,
operating engine(s) at maximum takeoff thrust, critical engine inoperative, speed V2; the manoeuvre
"may be unchecked". Reading: 60° in 11 s = 5.5°/s average, including the roll build-up.

For all engines operating the FAA gives no number:

> "This is primarily a qualitative evaluation that should be conducted throughout the test
> program." — ¶5.3.2.6.2 (§ 25.147(f))

### 2.3 EASA CS-25 Amendment 28 (as corrected 20/11/2025): all engines 7 s, high speed 8 s

> "roll the aeroplane from a steady 30° banked turn through an angle of 60° so as to reverse the
> direction of the turn in not more than 7 seconds." — AMC 25.147(f), pp. 154–155

Conditions: (a) en route, all speeds from the minimum all-engines climb speed to VMO/MMO, flaps
en route, gear up, flight idle to maximum continuous power; (b) approach, at the CS 25.125 approach
speed, landing flaps, gear down, power for a 5 % descent gradient. Rudder may be used to minimise
sideslip; "the manoeuvres may be unchecked". Reading: 60°/7 s = 8.6°/s average.

AMC 25.147(d) (p. 154) repeats the FAA's one-engine-out test: 30° → 30° "in not more than
11 seconds" at V2.

> "Using lateral control alone, it should be demonstrated that the aeroplane can be rolled to 20° bank
> angle in the other direction in not more than 8 seconds." — AMC 25.253(a)(4), p. 178

(starting from a steady 20° bank near VDF/MDF, flaps and gear up). Reading: 40°/8 s = 5°/s average.

AMC 25.671 §7.e.(1)(iv)(B) and FAA AC 25.671-1 use the same "60° … in not more than 11 seconds" roll
as one of the manoeuvres showing continued safe flight after a flight-control failure.

Kind of value for all of §2: **minimum capability** with a full, abrupt lateral input. Reading:
p = 8°/s is close to the all-engines average (8.6°/s); an executor rolling at p all the time rolls
about as hard as a certification capability demonstration.

### 2.4 Handling-qualities requirements (military standard and NASA)

MIL-F-8785C, *Flying Qualities of Piloted Airplanes*, 5 November 1980 (everyspec.com mirror).
§3.3.4: "Inputs shall be abrupt, with time measured from the initiation of control force
application." Table IXf, "Class III roll performance. Time to Achieve 30° Bank Angle Change
(Seconds)" (§3.3.4.2, p. 30):

| Level | speed range | Category A | Category B | Category C |
|---|---|---:|---:|---:|
| 1 | L / M / H | 1.8 / 1.5 / 2.0 | 2.3 / 2.0 / 2.3 | 2.5 / 2.5 / 2.5 |
| 2 | L / M / H | 2.4 / 2.0 / 2.5 | 3.9 / 3.3 / 3.9 | 4.0 / 4.0 / 4.0 |
| 3 | all | 3.0 | 5.0 | 6.0 |

(Category B = non-terminal phases such as climb, cruise and descent; Category C = terminal phases
such as approach and landing; Level 1 = clearly adequate.) Reading: Level 1, Category C = 30° in
2.5 s = 12°/s average.

NASA TN D-5957 (§1.1), capability side: "A steady-state roll rate of 15 to 20 degrees per second and
roll time constants of 1.8 seconds or less were required for acceptable and satisfactory pilot
ratings" (Summary, p. 1), and "a time to bank 30° of about 1.5 seconds would be satisfactory …
for transport operation" (p. 14).

Kind of value: **capability requirements**. They show that p = 8°/s is well within what a transport
can do; they say nothing about the rate used in routine turns.

## 3. Procedure-design allowances for pilot reaction and bank establishment

### 3.1 ICAO PANS-OPS, Doc 8168 Volume II

Copy used: Doc 8168 Vol II, **5th edition (2006) with Amendments 1–3** (Amendment 3 applicable
18 Nov 2010), as filed by the UK CAA for airspace change proposal ACP-2015-02 (Edinburgh SIDs and
STARs). Its pages carry "Provided by IHS under license with ICAO". The current edition is the 7th
(2020), sold by ICAO only; it was not checked (§6).

General turn parameters, Part I, Section 2, Chapter 3 "Turn area construction":

> "b) Rate of turn (R) in degrees/second. … up to a maximum value of 3 degrees/second." — 3.1.2.2 b)

> "f) c = 6 seconds pilot reaction time." — 3.1.2.2 f)

Table I-2-3-1 "Turn construction parameter summary" (pp. I-2-3-6/7, Amendment 1), columns "Bank
establishment time" and "Pilot reaction time" (together the flight technical tolerance, c):

| segment or fix of turn | bank angle | bank establishment (s) | pilot reaction (s) |
|---|---|---:|---:|
| Departure | 15° for the turn area; 15°/20°/25° for the average flight path by height | 3 | 3 |
| En route | 15° | 5 | 10 |
| Holding | 25° (RNP: 23° below FL245, 15° above) | 5 | 6 |
| Initial approach, reversal and racetrack | 25° | 5 | 6 |
| Initial approach, DR track | 25° | 5 | 6 |
| IAF, IF, FAF | 25° | 5 | 6 |
| Missed approach | 15° | 3 | 3 |
| Visual manoeuvring (prescribed track) / circling | 25° / 20° | N/A | N/A |

Table note 2: "The rate of turn associated with the stated bank angle values in this table shall not
be greater than 3º /s, except for visual manoeuvring using prescribed track."

Quotes from the chapters behind the table:

> "a distance equivalent to 6 seconds of flight (3 second pilot reaction and 3 second bank
> establishing time) at the specified speed." — Part I, Sec 3, Ch 3, 3.3.4 h) (turning departures)

> "c = a distance equivalent to 6 seconds of flight (3-second pilot reaction and 3-second bank
> establishing time)" — Part I, Sec 4, Ch 6, 6.4.3 h) 1) (turning missed approach; also 6.4.6.2 b))

> "2) pilot reaction time of 0 to + 6 s; 3) establishment of bank angle, + 5 s" — Part I, Sec 4,
> Ch 3, 3.6.2 h) (racetrack and reversal areas; bank "25° or … 3° per second, whichever is the lesser")

> "e) average achieved bank angle: 15°; f) maximum pilot reaction time: 10 s; g) bank establishment
> time: 5 s" — Part II, Sec 3, Ch 1, 1.4.3.3 (en-route turns, IAS 585 km/h (315 kt) = 162 m/s)

> "an overall tolerance of 11 seconds shall be applied … a) 6 seconds tolerance for pilot reaction;
> and b) 5 seconds for establishment of bank." — Part II, Sec 4, Ch 1, 1.3.9.1 (holding)

RNAV minimum stabilisation distance, Part III, Section 2, Chapter 1, 1.4:

> "L2 is a five-second delay to take into account the bank establishing time." — 1.4.2.1 (fly-by)

> "d) a 10-second delay to account for bank establishing time." — 1.4.1.1 (flyover)

Kind of value: **design allowances** (protection-area construction), not measurements. PANS-OPS gives
no roll rate. Reading: 25° established in 5 s, or 15° in 3 s, is 5°/s average; 32° in 4 s (8°/s) is
faster. And the design puts 3–10 s of pilot reaction **before** bank establishment starts.

### 3.2 FAA TERPS, Order 8260.3G (07/01/2024)

The only TERPS roll rate found is in the simultaneous-offset-approach design program:

> "The SOIA Design Program determines the approach geometry based on a nominal bank angle of 15
> degrees, roll-in/roll-out rates of nominally three degrees per second" — Appendix E, Section 4, ¶6.a (p. E-17)

> "Roll-in rates of up to five degrees per second and bank angles of 25 degrees may be used to
> determine the realignment flight track." — Appendix E, Section 4, ¶6.a

Holding (Chapter 16):

> "(3) Delay in recognizing and reacting to fix passage: six seconds for entry turn, applied in the
> direction most significant to protected airspace." — ¶16-1-3.b.(3) (p. 16-1)

Reading: at 3°/s a roll-out from 32° takes 10.7 s, from 25° 8.3 s; at 5°/s, 6.4 s and 5 s.

### 3.3 FAA PBN design, Order 8260.58D (01/15/2025)

> "FO turn construction incorporates a delay in start of turn to account for pilot reaction time and
> roll-in time." — ¶1-2-5.d.(2)(a), p. 1-18

Formula 1-2-12, "Reaction & Roll Distance": Drr = VKTAS × 6 / 3600 (NM), i.e. 6 s of flight at the
turn's true airspeed (reading: the 6 s covers reaction and roll-in together).

### 3.4 Related certification reaction times (flight-control failures, not ATC)

AC 25.671-1 ¶8.4.1.3.3, Table 1 (and AMC 25.671 §7.e.(1)(iii), pp. 384–385): reaction time after recognition (itself
"not normally … less than 1 second") is 1 s in manual flight and 3 s in "Automatic flight (>1,000
feet AGL)", 3 s if control passes between pilots. Listed only as the certification authorities'
human-reaction assumption; it concerns failures, not heading instructions.

## 4. ATC heading instruction → readback → turn: measured delays

### 4.1 Voice loop, en route: Cardosi & Boole (1991)

Cardosi, K. M., Boole, P. W., *Analysis of Pilot Response Time to Time-Critical Air Traffic Control
Calls*, DOT/FAA/RD-91/20 (DOT-VNTSC-FAA-91-12), Volpe Center, August 1991. Setting: 46 h of voice
tapes from Los Angeles, New York and Salt Lake ARTCCs (en route, high and low sectors, two workload
levels); 5,205 controller-to-pilot transmissions; times in whole seconds (reading: every minimum in
Tables 3-1/3-2 is 1).

"Turns not for traffic" (n = 250), Table 3-2 (mean, SD, min–max, s): controller message 4.62 (2.98,
1–26); **lag, end of controller message → start of pilot response, 2.68 (4.60, 1–41)**; pilot
response 2.66 (1.58, 1–11); total 10.04 (5.90, 4–52).

> "The median total time required for transmission of a turn not isuued [sic] for traffic avoidance
> was 8 sec." — §3.2, p. 3-4

> "In 95 percent of the cases, the total time was less than or equal to 21 seconds, and 90 percent of
> the total times were equal to or less than 16 seconds." — §3.2

"Maneuvers for traffic" (n = 80), Table 3-1: lag mean 3.31 s (SD 4.80); total mean 10.85 s, median
9 s, 90 % ≤ 17 s, 95 % < 23 s (§3.1). Total = start of controller speech → end of the pilot's
correct acknowledgement, including repeats (13–14 % of calls). The authors warn the ~10 s average "is
only valid for the en route environment" (Executive Summary). No aircraft response was measured.

### 4.2 Voice + ADS-B, approach: Lutz, Chatterji & Idris (NASA Ames, 2022)

Lutz, M., Chatterji, G. B., Idris, H., "Characterization of Response Times based on Voice
Communication and Traffic Surveillance Data", AIAA AVIATION 2022 Forum, paper 2022-3762 (NASA NTRS
20220007116). Setting: 24 h of LiveATC audio from KLAX Approach Zuma/NW arrival (frequency 124.500, printed "Hz"; MHz meant) on
1 March 2022, ADS-B Exchange tracks within 150 NM (278 km); 600 transmissions checked by an
IFR-rated pilot.

Definitions: "The Maneuver Initiation Detection Delay (MIDD) is the temporal difference between when
the aircraft maneuver is first recognized in the track data and the end of clearance delivery"
(§III). Heading thresholds: initiation at ± 2.5° (§III.C), completion at ± 5° of the cleared
heading (§III.B); track data searched in 5 s steps (§III.B), so values carry ± 5 s resolution
(reading).

Table 5, heading change commands: **MIDD mean 16.70 s, SD 6.47 s, P25 10.00 s, P75 21.00 s**;
completion (MCDD) mean 69.3 s, SD 43.5 s, P25 45.0 s, P75 81.0 s.

> "The average MIDD values for altitude, heading, and speed commands in isolation were 16.69
> seconds, 16.70 seconds, and 25.47 seconds, respectively." — §IV.B

Readback delay (end of ATC transmission → start of readback, heard on frequency; 257 clearances on
KLAX Tower South, ZLA sector 25 and KLAX Approach Final North): "The average readback delay was found
to be 0.639 seconds considering the three domains" (§IV.C); approach 0.669 s (SD 0.716 s).

### 4.3 ATM-system clearances + ADS-B, Sweden: Wüstenbecker et al. (DLR, 2025)

Wüstenbecker, N., Renkhoff, J., Zeppenfeld, D., Jameel, M., Schier-Morgenthal, S., "Analysis and
prediction of pilot response time to air traffic control clearances", CEAS Aeronautical Journal
17(2), 2025, doi:10.1007/s13272-025-00848-9, CC BY 4.0 (copy from DLR elib). Setting: Swedish Civil
Air Traffic Control (SCAT) dataset, 13 weeks over one year, en route and TMA; clearances as entered
in the Thales TopSky ATM system; ADS-B / Mode S at 5 s intervals; 831,311 clearances after filtering,
116,498 of them heading changes (Table 1).

Definitions: initiation delay runs from the clearance's entry in the ATM system (assumed equal to the
moment it was issued; voice or datalink not known) to "The first pilot reaction … once 5% or more of
the issued clearance delta is fulfilled" (§2.1). Heading tolerance 2.5° (§2.1).

> "we observe a right-skewed distribution with a peak at 10.75 s and a pronounced tail extending
> beyond 60 s." — §4.1 (all clearance types)

> "including all initiation delays below 90 s, the mean initiation delay is 20.50 s" — §4.1

"heading change commands consistently produce the quickest responses from pilots" (§4.1.1), and
"Heading changes are executed much more rapidly, usually within 50 s" (§4.1.2). Heading-only
initiation statistics are shown only as box plots by flight level (Fig. 5) and were not transcribed.
Initiation delay was hard to predict (heading: MAE 14.9 s, R² 0.24; Table 2).

### 4.4 Full-flight simulators, urgent heading instructions: MIT Lincoln Laboratory (1998)

Hollister, K. M., Rhoades, A. S., Lind, A. T.:
- *Airbus 320 Performance During ATC-Directed Breakouts on Final Approach*, Project Report ATC-265,
  20 Nov 1998. Full-motion A320 simulator at Northwest Aerospace Training Corporation, Eagan MN,
  1995; qualified A320 pilots.
- *Evaluation of Boeing 747-400 Performance During ATC-Directed Breakouts on Final Approach*, Project
  Report ATC-263, 7 Jan 1998. B747-400 simulator at NASA Ames; United and Northwest pilots.

Instruction (Phase 2): "(Aircraft) TRAFFIC ALERT. (Aircraft) TURN (left/right) IMMEDIATELY HEADING
(degrees). (Climb/Descend) AND MAINTAIN (altitude)." Times run from the **start** of the ATC
instruction.

> "Start of Turn: Data record at which the aircraft achieved a 3-degree or greater roll to the left
> and maintained the roll until the final heading was reached." — ATC-265 §1.3 "Report Terminology" (p. 8)

A320, Phase 2, time to start of roll (ATC-265 Figure 4-5; n, mean, median, SD, max, s):

| breakout | approach mode | n | mean | median | SD | max |
|---|---|---:|---:|---:|---:|---:|
| climbing, at decision altitude | hand-flown | 8 | 6.6 | 5.1 | 4.8 | 18.1 |
| climbing, at decision altitude | autopilot | 10 | 12.8 | 11.5 | 5.4 | 26.7 |
| climbing, above 400 ft (122 m) AGL | hand-flown | 45 | 5.3 | 4.5 | 2.8 | 21.1 |
| climbing, above 400 ft (122 m) AGL | autopilot | 82 | 7.6 | 6.7 | 2.5 | 16.8 |
| descending, 1800 ft (549 m) AGL | hand-flown | 8 | 5.3 | 5.1 | 0.9 | 6.9 |
| descending, 1800 ft (549 m) AGL | autopilot | 9 | 7.3 | 6.9 | 1.5 | 10.4 |

The descending breakouts (a heading change and a descent, no go-around mode) are the closest to an
ordinary vector (reading). Autopilot delays at decision altitude include TOGA mode interactions: once
TOGA engaged, "the flight director reverted to the pre-programmed go-around track" and "the start of
turn was delayed 4 to 18 seconds" (ATC-265 §4.4.3).

B747-400 (ATC-263 §5.4.2; start of roll = bank ≥ 3°, start of heading change = heading changed by 3°,
Table 3-1; heading-select time taken from video):

> "there was a correlation for autopilot-coupled breakouts: dt_roll always occurred after
> t_heading_select, and the mean time difference (dt_roll - t_heading_select) was 3.3 seconds." — §5.4.2

> "When the outliers were excluded, the mean difference between the two (dt_heading - dt_roll) was
> 3.3 seconds, with a range in values of 1.5 to 5.1 seconds." — §5.4.2

Controller breakout instruction duration (ATC-263 Table 5-2): median 4.9 s (Phase 1), 6.8 s
(Phase 2), 6.7 s (Phase 3); maximum 7.2–8.5 s. The roll began before the end of the transmission in
32 % of climbing breakouts (§5.4, subsection misnumbered "5.3.3 Start of Maneuver Relative to ATC
Transmission Length" in the report). B747-400 autopilot-coupled breakouts were slow: in Phase 3,
descending breakouts flown on the autopilot had mean 17.6 s, median 16.7 s (n = 9, Figure 5-11).

Reading (rough, assumptions stated): if a B747-400 at about 80–87 m/s (155–170 kt) rolls on at a
constant rate from 3° of bank, a 3° heading change 3.3 s later implies an average roll rate of about
2.7–3.1°/s over those seconds; at 8°/s the same 3° would come after about 2.2 s. The 3.3 s mean pools
hand-flown and autopilot trials.

Earlier data reproduced in ATC-265 Table 1-1 (originals not found, §6): B727 and DC-10 simulator
studies, start of instruction → 3° roll, means 3–8 s, maxima 11–23 s; a follow-on B727 study after
briefing pilots on "immediately": mean 5.6 s, max 11 s. The PRM risk analysis accepted evader
responses with a

> "mean time to start of turn less than or equal to 8 seconds and maximum time to start of turn of
> less than 17 seconds." — ATC-265 Executive Summary (test criteria)

### 4.5 A design allowance for the whole loop: EUROCONTROL STCA

EUROCONTROL Guidelines for Short Term Conflict Alert, Part III – Implementation and Optimisation
Examples, EUROCONTROL-GUID-159, edition 1.0, 18/01/2017. The reaction-time parameters

> "are supposed to cover the reaction times of the controller, communications, and also pilot and
> aircraft reaction." — §3.10

Table 10: StandardTurnReactionTime 50–70 s, en route and TMA. Kind of value: **alerting-system tuning
parameter**, not a measurement.

## 5. Files, sources and checksums

All retrieved 2026-09-26. Re-fetch with `bash download.sh`. Three files change SHA-256 on every
download with identical content (per-request PDF stamps): the USPTO patent and the two everyspec MIL
specs (marked †); compare their size and page count instead.

| file (`papers/`) | document | source URL | bytes | SHA-256 |
|---|---|---|---:|---|
| `NASA_TN_D-5957_Holleman_1970_roll_requirements_transports.pdf` | NASA TN D-5957 (1970) | https://ntrs.nasa.gov/api/citations/19700029309/downloads/19700029309.pdf | 4,394,061 | `317f172080c65b3161796367aaa4854ffc73a10e897b0b9b8358280d38514f98` |
| `NTSB_DCA09MA026_Att23_Airbus_A320_FCOM_1.27.20_Normal_Law.pdf` | A319/A320/A321 FCOM 1.27.20 Normal Law (US Airways, NTSB docket) | https://data.ntsb.gov/Docket/Document/docBLOB?FileExtension=.PDF&FileName=Operations%2FHuman+Performance+2X+-+Attachment+23%3A+Airbus+FCOM+1+Flight+Control+Normal+Law-Master.PDF&ID=40313401 | 295,206 | `798cbe833aec904191aa6cd6d47c5e3f113eb6bc830791b99225a9ef94b96d53` |
| `FAA_AC_25.1329-1C_Chg2.pdf` | FAA AC 25.1329-1C Change 2 (02/13/2026) | https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_25.1329-1C_CHG2.pdf | 6,645,674 | `d1122e89d61663cc44456dd1ebff460127469027d8d667b0fd695bc8436357e0` |
| `FAA_AC_25.671-1.pdf` | FAA AC 25.671-1 (08/30/2024) | https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_25.671-1.pdf | 2,618,452 | `1de2e7533b46983eee0d0b9110b76669aa42f8cb554d55d2256d8fa313da2c11` |
| `MIL-F-9490D_1975_via_everyspec.pdf` † | MIL-F-9490D (6 June 1975) | https://everyspec.com/MIL-SPECS/MIL-SPECS-MIL-F/download.php?spec=MIL-F-9490D.026060.pdf | 4,109,037 | `e11832af0153f5b5b6dcf1c9be8f3b48e6a4bcead2015adf11cbd3c972972e4b` |
| `Honeywell_US5023796A_1991_heading_track_hold_patent.pdf` † | US Patent 5,023,796 (1991) | https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/5023796 | 745,796 | `d64cf8b109bbb8270a9cf4976f467a1914128a4e78492204f28d5d244f051281` |
| `eCFR_14CFR_25.147_2026-09-01.xml` | 14 CFR 25.147 in force 2026-09-01 | https://www.ecfr.gov/api/versioner/v1/full/2026-09-01/title-14.xml?part=25&section=25.147 | 3,697 | `b3efcbfe9177bc83fd037156b78812b0494240b0d395b3022fe8d863605d6285` |
| `FAA_AC_25-7D_Chg1.pdf` | FAA AC 25-7D Change 1 (09/16/2025) | https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_25-7D_Chg_1.pdf | 8,240,583 | `da5783aec23cccc05345bb358f434e745b183f25cc8c50265c3da0e83325b16f` |
| `EASA_CS-25_Amdt28.pdf` | EASA CS-25 Amendment 28 (corrected 20/11/2025) | https://www.easa.europa.eu/en/downloads/139073/en | 23,584,619 | `32f1a9acf26e8ceceb291d206bf07a7071f7f59f40f10c096bb17f34c6a58007` |
| `MIL-F-8785C_1980_via_everyspec.pdf` † | MIL-F-8785C (5 Nov 1980) | https://everyspec.com/MIL-SPECS/MIL-SPECS-MIL-F/download.php?spec=MIL-F-8785C.028392.pdf | 3,465,219 | `0e22eaad6c2d65ee129f289ba5c018944151c637846180dd75e544d6fd0c2066` |
| `ICAO_Doc8168_VolII_via_UKCAA_ACP-2015-02.pdf` | ICAO Doc 8168 Vol II, 5th ed. + Amdt 1–3 (UK CAA ACP copy) | https://www.caa.co.uk/publication/download/17061 | 16,348,531 | `48136025d65928e96eed59c6b63f8a84da74c08306304d21c526ec199f7cb566` |
| `FAA_Order_8260.3G.pdf` | FAA Order 8260.3G TERPS (07/01/2024) | https://www.faa.gov/documentLibrary/media/Order/Order_8260.3G.pdf | 20,580,280 | `ea779404ff8d0cab17c7ff47976b1a45a88b51a9c82f8ba40ad330d574a93550` |
| `FAA_Order_8260.58D.pdf` | FAA Order 8260.58D PBN design (01/15/2025) | https://www.faa.gov/documentLibrary/media/Order/Order_8260.58D.pdf | 15,064,527 | `ead3c8e089a88dbde5de0a35a47ad72b9d484a54d408f1b0db15828581f31638` |
| `FAA_RD-91-20_Cardosi_Boole_1991_Pilot_Response_Time.pdf` | DOT/FAA/RD-91/20 (1991) | https://rosap.ntl.bts.gov/view/dot/9082/dot_9082_DS1.pdf | 717,656 | `53d2e2a6fad24b58e595860a2f82ad3d27b532ad7750877d9686cc3fff840b11` |
| `NASA_Lutz_Chatterji_Idris_2022_AIAA_response_times.pdf` | Lutz et al., AIAA 2022-3762 | https://ntrs.nasa.gov/api/citations/20220007116/downloads/20220007116_Lutz_Aviation2022_NLP.pdf | 648,014 | `f485bd2b6cb1af525af811c6f2b9b753871d7bd317a470996055bc5fd01a9b2a` |
| `Wuestenbecker2025_CEAS_pilot_response_time_DLR-elib.pdf` | Wüstenbecker et al., CEAS Aeronaut. J. 2025 | https://elib.dlr.de/219500/1/Analysis_and_Prediction_of_Pilot_Response_Time_to_Air_Traffic_Control_Clearances.pdf | 3,989,986 | `82ce33bfae3cfdaee68bdc707b19a03f68e052d0eb6cc5913852b33b36785ae1` |
| `MITLL_ATC-265_Hollister_1998_A320_breakouts.pdf` | MIT LL ATC-265 (1998) | https://archive.ll.mit.edu/mission/aviation/publications/publication-files/atc-reports/Hollister_1998_ATC-265_WW-15318.pdf | 8,093,325 | `f3e8c81ef10c77f2790e3bdb9228abc27985bca9be9c4e0222e9f07d35b0354b` |
| `MITLL_ATC-263_Hollister_1998_B747-400_breakouts.pdf` | MIT LL ATC-263 (1998) | https://archive.ll.mit.edu/mission/aviation/publications/publication-files/atc-reports/Hollister_1998_ATC-263_WW-15318.pdf | 9,291,757 | `04209480c2def681781d43d07ea652fa81e26ad7221b9e0fa7b5b1cb643439e0` |
| `EUROCONTROL_GUID-159_STCA_Part_III_Ed1.0.pdf` | EUROCONTROL-GUID-159 Part III (2017) | https://www.eurocontrol.int/sites/default/files/2019-09/eurocontrol-guidelines-159-part-iii-1.0.pdf | 781,541 | `ec30cb1cc6564f917e6a391f140f2328b13de725cebcc79c348865f02d855c95` |

Mirrors, not publisher copies: the two MIL specifications (everyspec.com, not the DoD ASSIST site),
Doc 8168 (UK CAA, not ICAO) and the Airbus FCOM page (NTSB public docket, not Airbus).

## 6. Not verified / not found

- **An airliner FCOM or FCTM value for the autopilot's roll rate in HDG SEL / LNAV (Boeing, Airbus,
  Embraer): not found in a public primary source.** Searched: FAA and EASA advisory material
  (AC 25.1329-1C, AMC No. 1 to CS 25.1329: comfort wording only), NTSB dockets (found the A320 FCOM
  normal-law page used in §1.2, A320 FCOM autopilot engagement/disengagement pages (only the 45°
  disconnect), a B777 mode-control-panel exhibit and B737NG FCTM excerpts, none with a roll rate),
  the BEA AF447 final report and interim report 3 (describe the A330 roll-rate law without a
  number), US patents. Boeing bank-limit-selector values (10–30°, "AUTO" 15–25° by TAS) and Airbus
  HDG/NAV bank limits (25°/30°) appear only on forums and unofficial FCOM copies and are **not cited**.
- **A passenger-comfort roll-rate number: not found** in an aviation primary source. The AC/AMC say
  only "consistent with occupant comfort". NASA ride-quality reports (TM X-71922, 1974; CR-159186,
  1980) treat random vibration, not turn entry. The "6 deg/sec … passenger comfort" autopilot
  requirement that web searches return is a MathWorks Simulink Test example file, not an aircraft
  document; not used.
- **Airline flight-data (FOQA/QAR) distributions of roll rate: not found.** The FAA statistical
  loads reports (e.g. A320, DOT/FAA/AR-02/35) were not checked: the tc.faa.gov server timed out, and
  DTIC answers 403 to scripted downloads.
- **The current PANS-OPS edition** (Doc 8168 Vol II, 7th edition 2020) is sold by ICAO; an
  ICAO-hosted courseware copy (icao.int APAC FPP) returned HTTP 404 on 2026-09-26. Every PANS-OPS
  value here is from the 5th edition with Amendments 1–3; whether the 7th edition changed Table
  I-2-3-1 is not verified.
- **Paywalled or not public, not read:** ICAO Doc 9905 (RNP AR design) and Doc 9643 (simultaneous
  parallel runways); RTCA DO-236 / DO-283; ARINC 702A; SAE AS402 (autopilot standard referenced by
  TSO-C9c); Rantanen, McCarley & Xu (2004), *Int. J. Aviation Psychology*, on ATC communication-loop
  delays; Cardosi (1993), *Int. J. Aviation Psychology* 3(4) (the journal version of §4.1; the FAA
  report was used instead).
- **Original B727 / DC-10 breakout reports** (FAA-AVN-500-51, -52, -54; PRM Demonstration Report
  DOT/FAA/RD-91/5): not found online. Their numbers are quoted only as reproduced in ATC-265 Table 1-1.
- **Airbus STL 945.7136/97** ("A319/A320/A321 Flight deck and systems briefing for pilots", the usual
  citation for "15°/s"): only on unofficial mirrors; not used. The 15°/s value is taken from the FCOM
  page in the NTSB docket.
- **Wüstenbecker et al., heading-only initiation delays** by flight level exist only as a figure
  (Fig. 5) and were not transcribed.
- **Status of MIL-F-9490D and MIL-F-8785C** (both superseded in the US military by later
  handbooks / joint service specifications) was not checked; they are quoted as published.
- **Reading, not measured:** the B747-400 roll-rate estimate in §4.4 (it assumes TAS and a constant
  roll rate and pools hand-flown and autopilot trials) and the MIL-F-9490D minimum-time arithmetic in
  §1.4.
