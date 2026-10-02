# Go-around and cancelling an approach — what the rules say (2026-10-02)

Why this folder exists: step 8 of the multi-aircraft design teaches the two-tier prior to say the
two words its vocabulary already has for leaving an approach: "go around" and "not cleared" said after
"cleared" (cancelling the approach). The user asked (2026-10-02) when these words are used in practice,
and whether an aircraft that goes around or has its approach cancelled must fly to the missed-approach
point and hold. The answers below are quoted from the FAA documents; the readings for the vocabulary
are mine and marked as such. Design: `4dTrajectory/ts_transformer/docs/two_tier/multi_aircraft_design.zh.md`
§6.6 step 8 details, items 10–11.

## Sources

No new files. The documents sit in `../runway_assignment/official/` (not tracked in git; run
`./download.sh`, which runs `../runway_assignment/download.sh`):

| Document | Edition | Paragraphs used |
|---|---|---|
| FAA Order JO 7110.65BB *Air Traffic Control* (`FAA_Order_JO_7110.65BB_Air_Traffic_Control_w_Chg1-3_2026-07-09.pdf`) | Change 3, effective 2026-07-09 | 3-8-1, 4-8-1, 4-8-9, 5-7-1 b.4; Pilot/Controller Glossary *BREAKOUT*, *GO AROUND*, *MISSED APPROACH* |
| FAA *Aeronautical Information Manual* (`FAA_AIM_Aeronautical_Information_Manual_w_Chg1-3_2026-07-09.pdf`) | Change 3, effective 2026-07-09 | 5-4-6 j; 5-4-21 a, b, d, f, g, h; 5-5-5 a.1, a.3, a.4 |

Quotes are copied from `pdftotext -layout` of those PDFs.

## What the documents say

**Go around: abandoning an approach to landing.**
- P/CG *GO AROUND*: "Instructions for a pilot to abandon his/her approach to landing. Additional
  instructions may follow. Unless otherwise advised by ATC, a VFR aircraft or an aircraft conducting
  visual approach should overfly the runway while climbing to traffic pattern altitude and enter the
  traffic pattern via the crosswind leg. A pilot on an IFR flight plan making an instrument approach
  should execute the published missed approach procedure or proceed as instructed by ATC; e.g., "Go
  around" (additional instructions if required)."
- 7110.65 3-8-1 *Sequence/Spacing Application* (tower): "Establish the sequence of arriving and departing
  aircraft by requiring them to adjust flight or ground operation, as necessary, to achieve proper
  spacing." Its phraseology lists, side by side, EXTEND DOWNWIND, MAKE SHORT APPROACH, CIRCLE THE AIRPORT,
  MAKE LEFT/RIGHT THREE−SIXTY/TWO SEVENTY, and "GO AROUND (additional instructions as necessary)."
- AIM 5-5-5 a.1: the pilot "Executes a missed approach when one of the following conditions exist:
  (a) Arrival at the Missed Approach Point (MAP) or the Decision Height (DH) and visual reference to the
  runway environment is insufficient to complete the landing. (b) Determines that a safe approach or
  landing is not possible (see subparagraph 5−4−21h). (c) Instructed to do so by ATC."

**Cancelling an approach clearance.**
- 7110.65 4-8-1 *Approach Clearance*, phraseology: "CANCEL APPROACH CLEARANCE (additional instructions as
  necessary) (When it is necessary to cancel a previously issued approach clearance)".
- AIM 5-4-6 j: "When necessary to cancel a previously issued approach clearance, the controller will advise
  the pilot "Cancel Approach Clearance" followed by any additional instructions when applicable." Neither
  text limits where on the approach it is said.
- P/CG *BREAKOUT*: "A technique to direct aircraft out of the approach stream. In the context of simultaneous
  (independent) parallel operations, a breakout is used to direct threatened aircraft away from a deviating
  aircraft."

**After a go-around / missed approach: the published procedure is the default, radar vectors replace it.**
- 7110.65 4-8-9 *Missed Approach*: "Except in the case of a VFR aircraft practicing an instrument
  approach, an approach clearance automatically authorizes the aircraft to execute the missed approach
  procedure depicted for the IAP being flown. … After an aircraft commences a missed approach, it may be
  vectored at or above the MVA/MIA or follow the provisions of paragraph 5-6-3, Vectors Below Minimum
  Altitude." Its NOTE: "In the event of a missed approach involving a turn, unless otherwise cleared, the
  pilot will proceed to the missed approach point before starting that turn." and mentions pilots "provided
  an initial heading to fly or radar vectors in lieu of published missed approach procedures".
- AIM 5-4-21 a: "When a landing cannot be accomplished, advise ATC and, upon reaching the missed approach
  point defined on the approach procedure chart, the pilot must comply with the missed approach
  instructions for the procedure being used or with an alternate missed approach procedure specified by ATC."
- AIM 5-4-21 b: "Obstacle protection for missed approach is predicated on the missed approach being initiated at the
  decision altitude/decision height (DA/DH) or at the missed approach point and not lower than minimum descent altitude
  (MDA). A climb gradient of at least 200 feet per nautical mile is required, (except for Copter approaches, where a climb
  of at least 400 feet per nautical mile is required), unless a higher climb gradient is published in the notes section of
  the approach procedure chart." (The executor climbs at this gradient while a go-around is in force — multi-aircraft
  design §6.6 step 8 item 7; R40 v2 compares the recorded go-arounds with it: p50 9.3 %, 96 % at or above 3.29 %.)
- AIM 5-4-21 d: "At locations where ATC radar service is provided, the pilot should conform to radar
  vectors when provided by ATC in lieu of the published missed approach procedure." (The same sentence is
  P/CG *MISSED APPROACH* c.)
- AIM 5-4-21 f: "When approach has been missed, request clearance for specific action; i.e., to
  alternative airport, another approach, etc."
- AIM 5-4-21 g: "… Additional climb may be required after reaching the holding pattern before proceeding
  back to the IAF or to an alternate." (the holding pattern at the end of a published missed approach).
- AIM 5-4-21 h: "A clearance for an instrument approach procedure includes a clearance to fly the
  published missed approach procedure, unless otherwise instructed by ATC."
- P/CG *MISSED APPROACH* a: "A pilot executing a missed approach prior to the Missed Approach Point (MAP)
  must continue along the final approach to the MAP." AIM 5-5-5 a.4: "If executing a missed approach prior
  to reaching the MAP, fly the lateral navigation path of the instrument procedure to the MAP."

**Few tools close to the runway.**
- 7110.65 5-7-1 b: "Do not assign speed adjustment to aircraft: … 4. Inside the final approach fix on
  final or a point 5 miles from the runway, whichever is closer to the runway."

## Readings for the vocabulary (mine)

- **Cancelling an approach and going around are different instructions.** A cancel withdraws the clearance;
  the aircraft flies the headings and altitudes that follow, does not climb by itself, and is re-sequenced. It
  can be said anywhere the clearance is in force, on final too. A go-around abandons the landing and climbs.
- A go-around abandons an *approach to landing*: it is said to an aircraft on the approach. In the
  vocabulary that is "cleared" in force (the labeller says "cleared" at the row the capture turn starts).
  A visual-pattern aircraft can be told to go around without an approach clearance, but it too is on
  final.
- After a cancelled approach the missed approach does not come into it at all. After a go-around the
  aircraft does **not** have to fly to the missed-approach holding fix: the published missed approach is the default, and in radar airspace vectors at or above
  the MVA replace it, followed by another approach. The executor's go-around (climb along the runway
  course until a new approach word) matches "continue along the final approach to the MAP"; the model
  then vectors with heading and altitude words, as the step-8 design already planned.
- The recorded go-arounds look vectored: go-around to landing p50 623 s, farthest from the airport p95
  28 km (`4dTrajectory/ts_transformer/docs/two_tier/readouts/2026-10-01_go_arounds.zh.md`).
