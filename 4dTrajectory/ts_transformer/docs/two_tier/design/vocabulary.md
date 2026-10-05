# Two-tier model: the vocabulary, the labeller and the executor

**Summary.** The two-tier model has two parts (outline §1): a prior that says rows of words, and an executor that flies
only these words. This document gives the lower part and the language between the two parts: the words (the
vocabulary), the program that reads words from observed tracks (the labeller), the autopilot that flies the words (the
executor) and the judge of what it flew. It is stage A of the plan. Other documents use this document only through its
public interface (§6) and its decisions (the D numbers).

**Language.** This document uses the writing rules of ASD-STE100 (Simplified Technical English): short sentences, active
voice, one meaning for each word, and technical names that the terms define. The words were not checked one by one
against the STE dictionary. Units are SI only (m, m/s, deg, s). Values in feet, knots or nautical miles occur only in
quotations of regulations, with the SI value given one time.

**Scope of this document.** This document is complete for its part: it gives every rule of the vocabulary, the labeller,
the executor and the judge. The principles and the rules that all parts share are in the outline (`outline.md`); this
document reads no other document. The code is on the branch `dev-two-tier-v4` (§0.3, §11). The evidence (§9) cites the
readouts and the data it comes from. Paths in backticks are relative to `4dTrajectory/ts_transformer/` unless they start
with `4dTrajectory/` or name the repository root; `readouts/` is `docs/two_tier/readouts/`. The documents of the earlier
design (`*_design.zh.md`, `two_tier_framework.zh.md`, `two_tier_stage_notes.zh.md`) are in
`archive/two_tier_v3_2026_10/docs/`.

---

## 0 Status

### 0.1 Decisions

The decision numbers are shared by all documents (outline §3).

| # | Item | State | Source |
|---|---|---|---|
| D1 | The runway column says the expected runway and "go-around". There is no approach column and no word "cleared". The model says the runway at the first predicted step | Decided | User, 2026-10-03 |
| D2 | The model flies the aircraft to the runway with heading words. No executor law turns the aircraft onto the final | Decided | User, 2026-10-03 |
| D3 | After the aircraft is on the final, the executor continues to fly the words of the model. The LPV minimums are a judgement only. The model must learn to land or to go around | Decided | User, 2026-10-03 |
| D4 | The speed value "unspecified" starts at the capture row of the labeller | Decided | User, 2026-10-03 |
| D6 | The grids of the words: heading 5°, four descent classes. A measurement or a rough labelling decides if they are sufficient (D35) | Decided | User, 2026-10-03 |
| D8 | The heading grid is aligned to the course of the runway in force (§3.3) | Decided | User, 2026-10-03 |
| D9 | The executor has none of the laws of §5.7, the glidepath floor included | Decided | User, 2026-10-03 |
| D10 | Go-around is an event: "abandon this approach". It does not change the runway in force; a runway word ends it. Its effects on the executor, the judge and the grammar: §3.2 | Decided | User, 2026-10-03 |
| D11 | The labeller stage includes an ablation of the row interval Δ (§4.8). The user chose Δ = 4 s on the readings of D34 of A25 (`readouts/2026-10-04_stage_a_a25_report.zh.md`; A30 gives them again, equal): a speaker says a row and the executor hears it every 4 s, and the step of a scene of several aircraft is 4 s. The artefact keeps the closed-loop sentences at Δ = 2, 4 and 8 s. The report showed the val rows too (outline D85) | Decided. Values: D25. Readings: D34. Δ = 4 s | User, 2026-10-03; Δ = 4 s: user, 2026-10-04 |
| D12 | No runway lock. While G is false the model can change the runway at any time; the judge reads R at the crossing (§3.2) | Decided | User, 2026-10-03 |
| D14 | While G is true, "no level-off" is not permitted (rule 5, §3.7). | Decided | User, 2026-10-03 |
| D15 | The spec measurement gives each value that it fits from data with rounder candidates and the fit that each leaves; the user chooses (§3.5) | Decided | User, 2026-10-03 |
| D18 | The labeller reads the real go-arounds inside a sentence and says "go-around" at the go-around point (§4.6) | Decided | User, 2026-10-03 |
| D19 | After a labelled go-around, the runway word that ends G is at the first level-off after the go-around climb, and not later than the row of the next "no level-off" (§4.6) | Decided | User, 2026-10-03 |
| D22 | Altitude words use the grid "optimal 40 levels" of the altitude-grid proposal: 60 m steps from 0 to 1,260 m, 120 m steps to 2,700 m, 450 m steps to 5,400 m (§3.4). Since D58 the heights are above the airport elevation, and the grid is fitted again on them | Decided | User, 2026-10-03 |
| D25 | The ablation reads Δ = 2, 4, 8 s: each divides the 16 s before the first predicted step (§4.8). The chosen Δ is 4 s (D11) | Decided | User, 2026-10-03 |
| D26 | The labeller reads a flight with a go-around approach by approach. An approach ends at the landing or at a go-around row. The last descent that reaches a go-around row says "no level-off". Each approach has its own capture row and its own "unspecified". The first row says the runway of the first approach. The go-around row is the row of the climb word (§4.2, §4.4–§4.6) | Decided | User, 2026-10-03 |
| D27 | "Go-around" changes no target of another column. A row that says "go-around" while "no level-off" is in force also says a level above the aircraft (rule 6). The executor keeps the heading word in force; it does not fly the course of R. While G is true, "unspecified" holds the airspeed (§3.2, §3.7, §5.4–§5.6) | Decided | User, 2026-10-03 |
| D28 | The vocabulary has one climb word. Its angle is a value of the executor. While G is true, the executor climbs at the steady climb angle that the thrust limit permits, not more than 3° and not less than 1.885° (200 ft per NM). While G is false, it climbs at the nominal angle of the climb class; that value comes from the spec measurement with D15 (§3.5, §5.5) | Decided (the range 1.885°–3° and the nominal from D15: the user; the thrust rule inside the range: Claude's proposal) | User, 2026-10-03 |
| D32 | Closed-loop reading. The labeller flies its sentence with the executor. When the flown path leaves the observed path by more than a tolerance, it says a correction word, and the observed word again when the flown path is back. These sentences and their flown states are the training sentences of the artefact (§4.9, §6). Tolerances: lateral 30 m, vertical 15 m (in the final descent H_final, D66) | Decided | User, 2026-10-03 |
| D33 | While G is true, no crossing of a threshold is an event: of R or of another candidate (§3.2, §5.8) | Decided | User, 2026-10-03 |
| D34 | The ablation of the row interval reads, at each Δ: the correction words of the closed-loop reading for each flight and each column; the errors left where §4.9 makes no correction; the replay outcomes. The user compares the Δ values on these readings (§4.8) | Decided | User, 2026-10-03 |
| D35 | No finer grids. A finer grid does not remove the drift of open-loop words (§9.6); the closed-loop reading does (D32). The grids are those of D6 | Decided | User, 2026-10-03 |
| D38 | The DA check uses one definition with the evaluation module. Vertical: within ±22 m of the published glidepath of R, the bound of `evaluation/thresholds.py` (`RNAV_TERMINAL_VERTICAL_BOUND_M`, ICAO Doc 9613). Lateral: inside the FAS cone at the distance of the DA point. Neither is a parameter (§5.8) | Decided | User, 2026-10-04 |
| D42 | The closed-loop reading says each observed word at the place where the observed aircraft heard it, not at the time: at the Δ row nearest to the place where the matched point of the flown aircraft reaches it (D45). The correction words do not change. The flight ends when the executor is done or at the time limit of the replay (§4.9) | Decided | User, 2026-10-04 |
| D43 | Speed words in steps. In a change of speed, the labeller says each grid value on the way, where the observed speed comes nearer to it than to the value before. The executor changes the speed of a speed word at the largest acceleration of the speed envelope (a_max, a value of the spec). The rate of a change thus comes from the words, as the turn rate comes from the heading words. "Unspecified" has its own rate (§3.6, §4.5, §5.6) | Decided | User, 2026-10-04 |
| D44 | No correction past the end of the observed path (the observed slice ends before the threshold; the flown aircraft flies on to it). There the closed-loop reading measures the lateral error against the line of the last segment, for the readouts only, and no vertical error (there is no observed height); it says no lateral or vertical correction: a correction in force ends, and the observed word is said again; the aircraft flies the last observed words to the threshold. The rows count as rows without correction (D34). The DA point always lies before the end of the observed path (§4.9) | Decided | User, 2026-10-04 |
| D45 | A word goes to the nearest row, not to the next row. The Δ grid puts each word of the 2 s reading on the nearest Δ row (§4.8). The closed-loop reading takes the observed words with their 2 s times and says a word at the first Δ row whose matched observed time is less than Δ/2 before the time of the word (§4.9). A word exactly between two rows goes to the later row in both. The mean lateness of a word alone in its row is then zero in the closed loop, and on the Δ grid zero at 2 s and 1 s (half a 2 s row, from the ties) at 4 and 8 s. When one row holds several words of a column, it says the last one, so the words said are early on average: in the closed loop, the observed heading words of train +0.04 / −0.30 / −1.48 s at Δ = 2 / 4 / 8 s, each inside ±Δ/2 (A21, `readouts/2026-10-04_stage_a_a21_report.zh.md` §3.4) | Decided | User, 2026-10-04; the lateness as measured: user, 2026-10-04 |
| D46 | A heading word is said in the frame where the executor hears it. Where the Δ grid or the closed-loop reading moves a heading word across a runway word, the word is the class nearest to its absolute track minus the course of the R in force when it is heard. A heading word is said when its absolute track differs from the track in force, also when its class is the class in force said under another course. A change of runway alone says no heading word. No sentence is refused for a heading word across a change of runway (§3.3, §4.8, §4.9) | Decided | User, 2026-10-04 |
| D47 | The turn law of the executor stays (§5.4). The measurement of A13: the words alone (the 5° grid and the lead) end a turn approximately 35 m inside the observed turn; the executor gives back approximately 18 m of it; the stopping-rate limit changes approximately 3 m (§9.10). The readout of a turn is its own part (the change of the displacement from the observed aircraft of the same time); the change of e_y is read beside it, because it holds the offsets that earlier turns left | Decided | User, 2026-10-04 |
| D48 | The 2 s reading follows D46 too. After a change of R, a heading word whose class is the class in force is said, because its absolute track differs from the track in force. No flight is refused for it (§4.3) | Decided | User, 2026-10-04 (Claude's check of stage A, §3 item 2) |
| D49 | The labeller conformance also covers the Δ grid: its reference sample stores the sentences at Δ = 4 and 8 s, and the check compares them again with the code on disk (§7.2) | Decided | User, 2026-10-04 (check §3 item 3) |
| D50 | The closed-loop reading does not keep every row inside the tolerances of D32. A correction is one class and needs time to bring the flown path back (§9.7: a 100 m offset takes approximately 20 s at one 5° class); in the stage A smoke 10–25 % of the correctable rows are more than 30 m off laterally. The check of §12.2 is a rule on the stored data: on every row where §4.9 permits a correction, \|e_y\| > Y is followed by a heading correction in force after that row, and \|e_h\| > H by an angle correction. The p90 of \|e_y\| at the last row is information. The share of the correctable rows outside the tolerances is a reading of the ablation (D34), not a check (§4.8, §12.2). Since D53 the reading keeps it on every row | Decided | User, 2026-10-04 (check §3 item 4) |
| D51 | The closed-loop artefact stores the flown states on the data's 2 s rows at every Δ, with the Δ rows marked. Observed and flown rows are then on one grid, and a motion input over the 2 s before a row exists at every Δ. The artefact stores raw states, not a derived input, so a change of the motion window needs no rebuild. The replay check of a closed-loop sentence compares on the 2 s rows. At Δ = 2 s nothing changes (§4.8, §4.9) | Decided | User, 2026-10-04 (check §3 item 1) |
| D52 | ε of a level = half the larger gap to its neighbouring levels + 10 m: 40 m for 0–1,200 m, 70 m for 1,260–2,580 m, 235 m for 2,700–5,400 m. A level's band then covers the largest rounding error of its word; the top level of a segment collects the heights up to half the next segment's step above it. One band for both sides (a one-sided band would differ at two levels only). "No level-off" has ε = 40 m: it has no level, so no rounding to cover; its tube's width comes mainly from the edges of the angle class (§3.4, §8) | Decided | User, 2026-10-04 (check §3 item 5) |
| D53 | An overshoot. When a correction is in force and the error changes its sign while it is more than the tolerance (the flown aircraft crossed the observed path in one row), the labeller says the opposite correction in the same row: one class toward the path from the observed word. When the error changes its sign and is within the tolerance, it says the observed word again. Laterally and vertically alike. The rule of D50 then holds on every row (§4.9, §12.2). Why: a correction ends at a sign change so that it does not push the aircraft further on the other side; waiting one row to correct that side adds a lag of one row Δ that comes from the reading, not from Δ, and the ablation compares Δ (A15 smoke: every break of D50 at Δ = 4 and 8 s was such a row) | Decided | User, 2026-10-04 |
| D54 | The descent classes are a k-means on tan(angle) of the descent pieces, with each piece weighted by the square of its length. A piece flown at the nominal angle of its class ends length · \|tan a − tan c\| from its observed end, so the k-means makes the sum of the squared end-of-piece height errors smallest; D15 gives the same error for each candidate. The climb nominal is the length-weighted median of the climb pieces (§3.5) | Decided | User, 2026-10-04 |
| D56 | The values of D15, chosen: the candidate rounded to 0.25° of the spec measurement on all train days (stage A code at `42f3ff62`, 44,703 train flights, 227,559 descent pieces, 1,339 climb pieces). Descent nominals 1.5 / 2.5 / 3.0 / 4.5°, edges −0.5 / 2.0 / 2.75 / 3.75 / 10°; climb nominal (G false) 1.5°. End-of-piece height error p50 / p90: 15.0 / 51.0 m (fitted 1.51 / 2.45 / 3.09 / 4.45°: 14.7 / 47.8 m). Descent 3 is the published 3.0° glidepath of 24 of the 25 candidate runways (§3.5, §8) | Decided | User, 2026-10-04 |
| D57 | One word clock: `time`. An open-loop sentence is said on its own rows, as a closed-loop sentence and a sentence of a speaker are. The clocks `distance` and `track` (an open-loop word said where the flown aircraft reaches the observed track) are removed with the next change of the executor spec, not in the formal build of A18 (A19). Why: no training sentence and no reading of D34 uses them; the closed-loop reading has the one rule of the place (D42, D45); the clock is a parameter of the executor spec, so a change of it gives a new spec identity and a rebuild of the closed-loop data that it does not change (D21); and the batch and the single-flight executors each carry it. The question "how far do the observed words alone carry a flight" is read, when a readout needs it, with the closed-loop reading with no corrections (§4.9) | Decided | User, 2026-10-04 |
| D58 | The altitude words are heights above the airport elevation E (the published field elevation, one value for each airport, `candidates.json` `reference.elevation_m`), not MSL. The labeller reads them from the height above E; the executor flies a level T at T + E MSL. E, not the threshold of R: R can change during a sentence, and the thresholds of one airport differ by up to 28 m (KSTL 160.8–188.4 m). The spec measurement fits the grid of D22 again on the level-offs above E, with the D22 grid among the candidates; the user chooses (D55). Why: at a high airport (KDEN, approximately 1,650 m) the MSL words of an approach are far from what the five training airports (1–188 m) said for the same manoeuvre, and they fall in the coarse segments of the grid; and the day-to-day spread of the geometric height of an assigned level grows with the height above the altimeter-setting source, the airport. The cost: the round MSL levels that controllers assign fall on different words at each airport (3,000 ft MSL: 900 m at KMSY, 780 m at KRDU, 720 m at KSTL) (§3.4). | Decided | User, 2026-10-04 |
| D59 | The grid of D58, chosen: the fitted row of the spec measurement on all train days (code `220e858e`, 44,703 train flights, 25,727 level words above E): 60 / 120 / 450 m, break points 1,260 and 2,700 m, 40 levels — the grid of D22 itself; rounding error of the level words p50 / p95 / largest 17 / 34 / 190 m. The next best of the 4,515 grids of the fit leaves 7.6 % more squared error. A21 builds with `--grid fitted` (§3.4, §9.5) | Decided | User, 2026-10-04 |
| D61 | The vertical path of each candidate (its TCH, its glidepath angle and its DA above the threshold) is part of the artefact: the first runner writes it into `candidates.json` beside the geometry, and every reader (the labeller's refusal, the judge, the closed-loop reading, the replay, the executor spec, the prior) takes it from there, never from the CIFP at run time. The names of the state columns of a closed-loop file and the function that reads a closed-loop file into its sentences are in `instructions/artefact.py`. Why: the public interface (§6, items 3 and 4) gives them, but the code held them in `autopilot/`, which the prior cannot import; and a value read from the CIFP at run time is not bound to the artefact (§6, milestone A22) | Decided | User, 2026-10-04, on the report of the stage B agent |
| D62 | The grammar's mask for a speaker (§6, item 2) is one function in `instructions/grammar.py`. It takes the words in force (with R and G; none before the first step), the words that the earlier columns of the row said, the column asked, the height above E, the number of candidates and, for each later column, the words that the speaker's other masks permit (all words when not given). It gives, for each word of the column ("unchanged" and, in the runway column, "go-around" included), whether some words of the later columns, each among its permitted words, make the row pass `apply`. Its definition is `apply`, not a second copy of rules 1–6. Why: a speaker says a row column by column, and a check with the later columns "unchanged" refuses good words (a level below the aircraft, which a descent class in the angle column makes grammatical); a copy of the rules in the prior breaks the one definition (§3.7); the permitted words of the later columns make sure that a row never reaches a column with no permitted word (milestone A22) | Decided | User, 2026-10-04, on the report of the stage B agent |
| D66 | While "no level-off" is in force (the final descent), the vertical tolerance of the closed-loop reading is H_final, smaller than H; elsewhere H = 15 m stays. The final descent is told by the words, not by procedure data (principle 3). The value is measured and the user chooses it (D55): the closed-loop reading and its replay at Δ = 2, 4, 8 s for H_final = 15, 10, 7.5 and 5 m, on the train sample of D34 and on every labelled select flight (milestone A24). A correction ends below half the tolerance in force; the rule of D50 and the readings of D34 read the tolerance in force at each row. Why: the DA check allows 22 m (D38); the real tracks lie up to +8 m (p90) above the glidepath at the DA point; with H = 15 m the flown path can stay up to 15 m above the observed one before a correction starts, and the correction then needs time. 67 of 1,999 train replays at Δ = 2 s ended `unstable_at_minimums`, 61 of them high, and 57 of these 67 flights passed the same check on their real track (`readouts/2026-10-04_stage_a_check_a15_a22.zh.md` §7). These sentences are the prior's training sentences | Decided; H_final = 10 m, the user's choice from A24 | User, 2026-10-04, on Claude's reading of the check; 10 m: user, 2026-10-04 |
| D67 | The start of a closed loop is part of the public interface (§6, item 5): one function in `autopilot/start.py`. It takes closed-loop sentences of the artefact as the reader of item 3 gives them (a split and a Δ), the executor spec of the artefact and the most go-arounds that a flight may say. It gives the executor at the first predicted step of each sentence's flight: the flight rebuilt from the harvest and compared row by row with the stored signals (§7.2 #4), its aircraft (its own dynamics or a stand-in's, by the rule of the replay) and its approach speed, its time limit (§5.8) and the time that its go-arounds may add (900 s each, up to the most given; a further go-around is refused, and the caller masks it). Then, at each Δ row, the caller gives the words of the row for each flight; the executor flies Δ seconds and gives the states of the 2 s rows flown (`STATE_COLUMNS`) and the flights that are done. The judge gives the outcome of a flight from what the executor recorded, with no observed words (§6, item 6). The closed-loop reading (§4.9) starts its flights with the same function. Why: a speaker's closed loop (the prior's free generation, the post-training) needs the start, and the code that makes it (`autopilot/replay.py`, `autopilot/flights.py`) is not in the public interface: a caller of that code breaks when stage A changes it. The start is not stored in the artefact: the artefact holds the signals, not the dynamics (§7.2 #4) (milestone A26) | Decided | User, 2026-10-04, on the report of the stage B agent |
| D69 | The check that closed-loop sentences may be read is part of the public interface of the sentence artefact (§6, item 3): `autopilot/closed_loop.py` `require_conforming_closed_loop`. It refuses unless the code on disk has passed the closed-loop conformance of the artefact (§7.2 #2). A runner of another document that reads closed-loop sentences calls it first and takes nothing else from that module. The function stays where it is. Why: item 3 names the closed-loop conformance record as part of its identity, but its code column named no function that checks it; the prior's training must check it (§7.2 #2), and the prior's architecture test lets a runner import only what §6 lists | Decided | User, 2026-10-04, on the report of the stage B agent |
| D70 | Each labelled flight's stratum (§4.3: straight-in or vectored) is stored in the sentence file by its name (`instructions/readout.py` `STRATA`) and is part of item 3 (§6). The labeller writes it with the one definition (`instructions/readout.py` `stratum`); the labeller conformance compares it as it compares the words. The sentence file and the labeller's reference get new format names. Like the capture row, the stratum uses later rows: it is for readouts and strata, never an input. Why: the readouts of the other documents are by stratum (the prior's free generation), and the stratum needs the labeller's turns before the capture row, which the artefact did not store; a document that computed it again would make a second definition (milestone A27, before A25) | Decided | User, 2026-10-04, on the report of the stage B agent |
| D71 | The start of a closed loop opens the executor spec itself (§6, item 5): it takes the directory of the spec, not its parameters, and opens it as the replay and the backend do (`autopilot/replay.py` `open_executor`). It is refused unless the spec was measured against the artefact's vocabulary and the labeller and the executor code on disk pass their conformance (§7.2 #2, #3); as before, it is also refused unless the closed-loop sentences were flown by these parameters. A caller handles no executor parameters, so no caller can start a closed loop with a spec that is not checked. The closed-loop reading, which writes the sentences, builds its loop from the same pieces and opens its spec as before. Why: the opener is not in the public interface, and naming it there would leave the checks to each caller (the prior's free generation, the loops of the post-training) (milestone A28) | Decided | User, 2026-10-04, on the report of the stage B agent |
| D73 | No code fingerprint (D21). The labeller, the executor and the closed-loop reading are checked by what they do on their reference samples (§7.2 #2, #3), in the process that uses them, every time. A runner that labels, opens an executor spec, reads the closed loop or replays sentences runs the checks of the code it uses before its work, and a difference refuses it by name; the backend runs them at its start. There is no passed record, no digest of code (not as a name, not as a guard of a run, not as information in a spec, an artefact or a reference) and no clean-checkout rule for a check. A reference is written only in the run that writes what it pins. A run records its commit and the checks it ran, as information. Data digests stay: the spec shas, the sha256 of files, the format names. Why: the passed records were named by a digest of the code's logic, so every change in `autopilot/`, even one that moves no track, refused every runner until the checks ran again from a clean checkout, and a list kept by hand decided which files the digest covered. The checks take 1.7 s (labeller), 26.7 s (executor) and 16.5 s (closed loop) on the formal artefact of A21 (one thread). The formats that recorded a code digest get new names, so the artefact of A25 is built again (milestones A29, A30) | Decided | User, 2026-10-04 ("代码指纹是毒瘤设计") |
| D74 | Each closed-loop sentence's outcome is stored in its closed-loop file and is part of item 3 (§6): the judge's outcome (§5.8) of the flight that the closed-loop reading flies, from the first predicted step to the end, by the outcome's name. Like the capture row and the stratum, it uses later rows: it is for readouts and selection, never an input. The closed-loop file gets a new format name. Why: the prior's base learns only from sentences whose own words land (prior D75), and the reading already flies every sentence to its end, so the outcome costs nothing more; another document that flew them again would make a second definition (milestone A31, before A30) | Decided | User, 2026-10-04 |
| D77 | The start of a closed loop reads no sample after the first predicted step. The executor's start state takes the position and the height of the observed row, and the airspeed, the track and the path angle from the observed rows at or before it, by a rule that the user chooses from a measurement (A33): the displacement in the 2 s before the row, or a least-squares line over the 8 s or over the 15 s before the row (the centred 15 s fit of today is measured beside them, for comparison only). The rule is a field of the executor spec. The stored states of the observed rows (§6, item 3) take their track, ground speed and vertical rate from the same rule. Why: the start took the data plane's velocity, a least-squares fit over 15 s centred on each sample, which reads up to 7.5 s after the row, and the interpolation onto the row reads the next sample too (Claude's review of stage A, 2026-10-05: on the select days at Δ = 4 s, in the 8 % of the flights that turn at the first predicted step, the start track is 4.4° (p50) from the observed direction of the next 8 s, against 11.5° for a fit over the 15 s before the row). Every closed loop starts there: the closed-loop reading, a speaker's closed loop and the live executor. Inputs give only what is known before the step (outline principle 7) | Decided; the rule waits for A33 and the user's choice | User, 2026-10-05, on Claude's review of stage A |
| D78 | The candidates of an airport are its runway ends with a published vertical path in the FAA CIFP (the LPV line, or the LNAV/VNAV line where no LPV line is published; D61). No flight decides a candidate. Why: the arrival manifest's `runway_targets` had an entry for each runway with an arrival on any day; KSTL 06 has one arrival, on a test day (2026-07-22), none on the other days, and it is a candidate of `v11_20261004`: a sealed day added a class to KSTL's runway column (C32). With published ends, an airport that is not in the training data, KAUS included, takes its candidates from published data only. A33 lists for each airport the ends that the rule adds and removes, and their arrivals by split, before the build | Decided | User, 2026-10-05, on Claude's review of stage A |
| D79 | The executor ends a flight where the judge ends it (§5.8): an approach crossing of R, ground contact, a lined-up crossing of another candidate with G false (inside that runway's own limit), a dynamics failure (the stall cut-off included), or the time limit. The single-aircraft batch, the multi-aircraft batch and the single flight alike; the start gives this end as "done" (§6, item 5). Why: the executor ended a flight only at a crossing of R, the ground, a state that is not finite and the time limit. A flight that crossed another runway or reached the stall cut-off flew on, and was given words, to its time limit (train, Δ = 8 s: one sentence with `crossed_other_runway` flown to its limit of 834 s) | Decided | User, 2026-10-05, on Claude's review of stage A |
| D80 | The start's `Loop.step` (§6, item 5) checks the row of each flight that flies with the grammar (`instructions/grammar.py` `apply`, at the height above E of the executor's state) before anything changes, and refuses the row by name. A value outside its column's words is refused too. Why: it took "go-around" while G was true (counted again, 900 s more time, the held airspeed set again), a runway value in another column (read with a negative index), and a refusal inside a cycle left the loop changed. A speaker applies the grammar as a mask; the start is the boundary of the public interface | Decided | User, 2026-10-05, on Claude's review of stage A |
| D81 | The executor holds nothing of the landed runway or of a procedure: the frame that it integrates in has its origin at the airport reference (with E), not at the landed runway's threshold and TCH. The rule "the executor's laws read no vertical path" (§5.2) is checked by behaviour, as D73 checks code: the executor's reference tracks, flown again with the vertical path of every candidate changed and the frame's origin moved, give the same states (within the executor conformance tolerance) and the same end cycles. This check replaces the scan of names in `tests/test_architecture.py`. Why: `FlightInputs.frame_params` held the landed runway's threshold and TCH. No law read it (a move of 3 km and 15 m changed the states by less than 1e-7 m), but a caller could read the landed runway from the executor, and the scan of names did not see a value read through `getattr`, a dictionary or the frame | Decided | User, 2026-10-05, on Claude's review of stage A |
| D82 | The reader of item 3 (§6) keeps what a model may read apart from what it must not read. For each sentence it gives (a) the rows: the words with their correction marks (targets), and the states; and (b) apart, by name, the fields that use later rows or the observed path: the landed runway (the flight's runway and its index), the landing time, the capture row, the go-around rows, the stratum, the outcome, `timed_out`, the matched and observed rows, the errors e_y and e_h, and the rows without a correction. The states of the observed rows follow D77; the track is in [0°, 360°) on every row. New format names. Why: these fields came with the rows, and only the capture row, the go-around rows, the stratum and the outcome were named as never inputs; the observed rows held the centred fit (up to 5.5 s after the first predicted step) and an unwrapped track | Decided | User, 2026-10-05, on Claude's review of stage A |
| D83 | The lateral error e_y of the closed-loop reading is the signed distance of the flown position from the observed path: from the line through the observed 2 s rows near the matched point (its segments, not their extensions), positive to the right of the path (§4.9). Why: the code measured the distance to the extended line of the matched segment; at a vertex on the outside of a turn, that line is nearer than the path | Decided | User, 2026-10-05, on Claude's review of stage A |
| D84 | The bank limit and the roll rate hold from the first cycle of a flight (§5.3); the start bank is 0. Why: the executor took any bank up to the bank limit in its first cycle (no roll-rate limit there), a law that §5.3 does not have. The user chose no exception and no start bank read from the data | Decided | User, 2026-10-05, on Claude's review of stage A |

| D86 | A23's export, and stage B's through `split_flights` (A36), read no formal replay row. They draw their flights from the flights of the closed-loop sentence files (at every Δ that the set shows), take each flight's stratum from the sentence file (D70), and check each sentence flown again against its stored states on the 2 s rows and its stored outcome (D74), for every split. Why: the formal replays are of train and select only (the replay of the val days waits for the user), so the export could not give the val flights of the base model's one validation readout (outline §6 item 4); the stored outcome is the one definition of a sentence's outcome (D74), and it equals the replay's on 8,195 of 8,195 replayed flights (A30) | Decided | User, 2026-10-05, on stage B's request (vocabulary O11 until then) |

### 0.2 Open items

| # | Item | Proposal | §  |
|---|---|---|---|
| O8 | Speed words in ground speed: wind can make speed words at turns | A check on data, later | 3.6 |
| O10 | The positions and heights of the observed 2 s rows are interpolated between the raw samples around each row, so a row reads the sample after it: p50 0.81 s, p99 1.69 s, at most 13.1 s later (KRDU, 300 train flights; Claude's review, 2026-10-05). Across a gap, the interpolation draws a line toward a later sample | Keep: the position of a row is where the aircraft was at that time, not a prediction, and the next sample is less than one row away on 99 % of the rows. A check on data of the rows whose next sample is more than one row away, later; the user decides | 4.1, 6 |

### 0.3 Implementation

| Part | State |
|---|---|
| Stage A: vocabulary, labeller, identities, executor, judge, replay, closed-loop reading (§12.1) | Built on `dev-two-tier-v4`, each milestone reviewed: A0 `e57c62d8` + `d2917c06`; A1–A3 `63639a54` + review fixes `feaf9558`, `d1e277f7`; A4–A6 `330ffbaf` + review fixes `c7603a4c`; A7 `14eb7946`; A8 `05f00be7` + review fixes in `c41ce7af`; A9 `c41ce7af` + review fixes `2e182e8d`, `983f2847`; D38 and D34 `6d1e8c4a` + review fixes `fdc3b832`; A10 `4ac34679` + review fixes `d1258c00`; A11 `efcbb5dc` + review fixes `0c0778c8`; A12 `b8077c34` + review fixes `b5b76c5a`; A13 `47865c8e` + review fixes `249804bd`, `b8994f58`; A14 `66daefa2` + review fix `ab295b18`. Smoke build of A12 and A13 (80 flights an airport and split, from the open-loop reading again, in the ignored `smoke_v4/data/a12/` of the worktree): the labeller's, the executor's and the closed loop's conformance checks pass; the closed loop read at Δ = 2, 4 and 8 s; the replays ran at Δ = 2 and 4 s open loop and 2, 4, 8 s closed loop; the turn readout of A13 ran on train and select; the full ts suite passes (1,523). Reports: `readouts/2026-10-04_stage_a_a8_a9_report.zh.md`, `readouts/2026-10-04_stage_a_a10_a11_report.zh.md`, `readouts/2026-10-04_stage_a_a12_a13_report.zh.md`. The closed loop of A14 built again in `smoke_v4/data/a14/`; the full ts suite at `ab295b18` passes (1,524). Claude's check of stage A (§12.2) at `ab295b18`: `readouts/2026-10-04_stage_a_check.zh.md` — items 2, 4, 5 pass; items 1 and 3 hold with points for the user (the motion input 2 s before a row is not stored at Δ = 4 and 8 s). The user's decisions on the check: D48–D52 and A15. A15 `4a2f4fc5` (three review rounds); smoke `smoke_v4/data/a15/`: the three conformance checks pass, the closed-loop sentences and the replays equal A14's on every Δ row, the replays pass on every 2 s row; the rule of D50 held at Δ = 2 s and broke at 4 and 8 s only on overshoot rows; the full ts suite passes (1,535). Report: `readouts/2026-10-04_stage_a_a15_report.zh.md`. A16 (D53, the overshoot corrected in its row) `3e70b4f3`, reviewed; smoke `smoke_v4/data/a16/`: the conformance checks pass, the rule of D50 holds on every row at Δ = 2, 4, 8 s, Δ = 2 s unchanged; the full ts suite passes (1,535). Report: `readouts/2026-10-04_stage_a_a16_report.zh.md`. A17 and A18 `688e945e`; A18's formal build stopped for D58. A19 + A20 (D57, D58) `220e858e`, reviewed (one break fixed: the publisher's mirrors of the executor names); the full ts suite passes (1,542). The spec measurement on all train days (scratch, 44,703 flights, code `220e858e`): the grid fitted on the level-offs above E is the grid of D22 itself; the rows of D56 unchanged. Report: `readouts/2026-10-04_stage_a_a19_a20_report.zh.md`. The first build of A21 (`v8_20261004`, `v13_20261004`, 13:36Z) finished before D61 and is superseded. A22 (D61, D62) `0c07f92f` + second-review fix `df84cbf1`, reviewed twice; the full ts suite passes at `0c07f92f` and at `df84cbf1` (1,547 each); `apply` on the one definition of the rules gives the old answers (reason and detail) on 60,000 random rows and the reviewers' 2.5 million; the superseded closed-loop files rewrite through the new reader and writer identically. A21's formal artefact built from `9a986c09` (2026-10-04 15:35–16:41Z): `instruction_language/v9_20261004`, `executor/v14_20261004`, read-only with `SHA256SUMS`; equal bit for bit to the build before D61 in every signal, sentence and closed-loop array; report `readouts/2026-10-04_stage_a_a21_report.zh.md`. Claude's check of A15–A22 (§12.2): all five items pass, `readouts/2026-10-04_stage_a_check_a15_a22.zh.md`; its §7 (the user's question): the landing share against instruction-v3 (99.6 % → 96.0 / 95.4 / 95.5 % on train) comes mostly from D38's check, which the real tracks pass at 98.4 % (train) / 98.0 % (select); the replay ends high at the DA (the user's answer: D66, A24, A25). A23, the code, on `v9_20261004` (§0.4 item 6), on `dev-two-tier-v4`: the export `experiments/training_export.py` (runner R54; index `training/index_v4.json` beside `index.json`, sample `aeroviz-training-sample-v9`) and the live executor on the closed-loop sentences (`aeroviz_backend/autopilot_segment/`, answer v9) `97053c7f`, each word decoded by `Words` `36c0ed5f`; reviewed (export and backend twice, frontend twice), fixes `def0127c`, `a6d40fca`, `6d79aa38` (the replay's flight to its outcome, `replay.track`, and the executor's cycle in the sample), `dccb785c`, `0e301993`, `7842a8c6`, `3bf96c92`; the frontend (written by a sonnet agent, accepted by Claude) `0c8e4c13`. Merged with A24/A26/A27 (`6ddeebdd`): the view reads `instruction-v6` (`7f3d0250`); the runner is R54 (R52, R53 are A24's and A26's). After the merge the full Python suite passes (2,588 + 161), Vitest 691, build; two independent browser checks pass on the scratch v9 trial (live words 0.000 m from the exported states); the backend's check over the scratch sets: 32,948 live segments, 0 differing. Both set their flights up through the replay's code (`experiments/training_flights.py`), as the formal replay did; once A26's start exists (D67) they can take it, which is stage A's own code either way. The user's choices for A23 (2026-10-04): the existing Training view adapted, its code written and debugged by a sonnet agent, Claude accepting it at the end, every visual check by a separate agent; the sample 20 train + 20 select flights an airport, half straight-in and half vectored, seed 1337; nothing published until the user says so (export and publication from A25's artefact). Checked in a scratch root only: 5 airports × 40 flights from `v9_20261004`/`v14_20261004`, every closed-loop sentence flown again equal to its stored states and formal outcome; KRDU's 4-flight trial: 781 live segments 0 m from the stored states, crossings equal to the export's. Open for the user: the choice of Δ (D7, D11). Decided by the user, not yet in the design text (the design's author writes it): D45's mean lateness worded as measured — Claude's proposed text: "the mean lateness of a word alone in its row is zero; when one row holds several words of a column it says the last one, so the said words are early on average (A21: +0.04 / −0.30 / −1.48 s at Δ = 2 / 4 / 8 s)" |
| A24, A26, A27 (D66, D67, D70) | **A24** on `dev-two-tier-v4-a24` (from `dev-two-tier-v4` `2ade7569`, merged `dev-two-tier` up to `68fc5c27`): `27442322`, reviewed (no blocking point; the fixes in the same commit) — the spec field `closed_loop_final_vertical_m` (H_final ≤ H; reading `instruction-v6`, spec `ts-instruction-spec-v7`), `closed_loop.vertical_tolerance_m` read by the corrections and by the rule of D50 / the readings of D34 at each row, `instruction_spec --closed-loop-final-vertical-m` (required when measuring), the measurement runner `final_descent_tolerance` (R52); the full ts suite passes at `27442322` (1,558). The measurement (scratch `.claude/worktrees/two-tier-v4-a24/smoke_v4/data/a24/`, four artefacts as `v9_20261004` but for H_final, train 400 an airport and every labelled select flight, Δ = 2, 4, 8 s): report `readouts/2026-10-04_stage_a_a24_report.zh.md`; at 15 m every number equals A21's; select Δ = 2 s landed 96.1 / 97.4 / 97.6 / 97.6 % at H_final = 15 / 10 / 7.5 / 5 m, angle correction words per final descent 5.65 / 8.23 / 10.97 / 16.29; KMSY apart at every value; the rule of D50 holds everywhere. Waits for the user's H_final. **A26** and **A27** on `dev-two-tier-v4-a26` (from `27442322`, merged `dev-two-tier` `9944f954`): A26 `6a5d688b` (`autopilot/start.py`: `start`, `Loop`, `GoAroundBeyondMost`; the closed-loop reading flies through `Loop`), review pending; A27 `390c79f9` + review fixes `7526845c` (sentences `ts-instruction-sentences-v5` with each flight's stratum, the labeller reference `ts-instruction-conformance-reference-v4` compares it). Claude's readings (proposals): `readout.py` joins the labeller code's name (`LABELLER_MODULES`), since D70 keeps `stratum` there and it is now stored — a logic edit anywhere in `readout.py` then asks for both conformance checks again; the closed-loop reading's Loop takes the most go-arounds of the batch's readings (no flight can exceed it). Note for the merges: after any of A24, A26, A27 no code reads `v9_20261004` / `v14_20261004`, so stage B's free generation and A23's export run on A25's artefact |
| A25, A28, A29 (D66, D71, D73) | **A25** built 2026-10-04 19:57–20:37Z from `dev-two-tier-v4` `3bf96c92` (the user's merge of A23–A27) plus the parallel closed loop (`dev-two-tier-v4-parallel` `b8d6909d`: `instruction_closed_loop --workers`, a split a process, the serial path's reading; a first try with a split × Δ a process ran out of memory and wrote nothing): `instruction_language/v10_20261004`, `executor/v15_20261004`, H_final = 10 m (the user's choice from A24), read-only with `SHA256SUMS`; report `readouts/2026-10-04_stage_a_a25_report.zh.md` — the readings of D34 at Δ = 2, 4, 8 s beside A21's, the rule of D50 on every cell, A26's start check (R53) on the artefact: 250 train flights at each Δ, 0 m. Waits for the user's Δ and the go to delete v9 / v14. **A28** `44ecbb84` and **A29** `bfeaeed0` on `dev-two-tier-v4-a28` (from the parallel branch, `dev-two-tier` merged up to `2fe51df4`), reviewed once; the review's points in work (the checks once a process and recorded in each run's output, a stronger architecture test, the backend through the closed-loop check, the override hook). Claude's readings (proposals): the checks run once in a process before its first use of an (executor spec, artefact) pair (`replay.CHECKED`) — the code of a process does not change, and a runner that starts a loop per chunk would otherwise fly the reference per chunk; `final_descent_tolerance` reads the closed loop itself, so it runs no closed-loop check; the Training export records its checks in its log only (its formats are the frontend's). The key code index (§11) for A24–A29 comes with the stage's end |
| The user's choice of Δ; the merge (D11, D34) | **Δ = 4 s** — the user's choice on the readings of D34 of A25 (`readouts/2026-10-04_stage_a_a25_report.zh.md`), 2026-10-04; decided by the user, not yet in the design text (D11, D25 and the values of §8 are the design's author's to write). `dev-two-tier-v4` fast-forwarded by Claude on the user's word to `4efbf5d3` (A24 `27442322`, A26, A27, the parallel closed loop `b8d6909d`, A28 `44ecbb84`, A29 `bfeaeed0` + review fixes `4efbf5d3`); targeted tests pass (428 + 79), the full ts suite at `4efbf5d3` before A30. The causes of the 2–3 % of A25's replays that do not land (Δ = 2 s, Claude's reading in the chat of 2026-10-04): high at the DA point with the real track already near +22 m (mostly KMSY) or too far off for one class; ground contact after drifting low under descent 1, which has no shallower class; crossing too high after a drift during a level hold or past the end of the observed path — not yet written up — `v9_20261004` and `v14_20261004` deleted on the user's go (2026-10-04; A24's scratch artefacts linked their signals, their readouts stay) |
| A31 (D74) | Done on `dev-two-tier-v4-a31` (from `dev-two-tier-v4` 4efbf5d3 + dev-two-tier 773e748f): `1429e0ef` the outcome of each closed-loop sentence — `closed_loop.read` takes the judge's (`judge.outcome_of`) on what its executor flew to the end, `ts-instruction-closed-loop-v7` stores it by name, `closed_loop_sentences` gives it back (C38), the closed loop's reference v7 compares it, the summary counts the sentences by outcome; review fixes `f39fb14a` + `c31e0b90` — the stored outcome is now required wherever a sentence is flown again (`executor_replay --closed-loop`, the start check R53 `ts-closed-loop-start-check-v3`, the Training export, R52), tests that can fail (a changed outcome refused, a timeout stored and replayed alike at Δ 2 and 8 s, a refused flight ahead of flown ones). Tests: stored = replay = start at Δ 2, 4, 8 s; a landing and each failure stored by name; a v6 file refused by name. Full suite at `4efbf5d3` passed (1,712); at `f39fb14a` 3 R52 tests failed (fixed in `c31e0b90`); at `c31e0b90` running, A30 chained behind it from a clean checkout of `c31e0b90` (3 closed-loop workers, as A25). Every reader now refuses A25's v10/v15 closed-loop files: the branch must not reach the live line before A30 is rebuilt |
| The closed loop's train in parts; dev-two-tier-v4 fast-forwarded; A30 started | On the user's word ("split train"): `instruction_closed_loop --train-parts P` (`a0d8b69e`, review fixes `e0cc22c1`) cuts train into P consecutive blocks of its one seeded permutation, each drawn and read in its own process; the sentences, the draw's description and the numbers are put together in order, so the files and the summary (text included: tie and group order) are those of train read whole — tested at 1/2 processes × 1/2/3 parts and on `draw_flights` in parts. Measured on A25's artefact with its code: a sixth of train draws in 126 s and reads at Δ 2 s in 84 s, peak 3.9 GB (train whole 1,837 s, 14.4 GB). Full suite (`run_all_tests.sh`, the user's parallel runner) at `e0cc22c1`: ts 1,534, modeling + backend 1,005, aeroviz-4d 161, exit 0. `dev-two-tier-v4` fast-forwarded by Claude on the user's word to `e0cc22c1` (A31 + train in parts), then `cf549e47` (R50 docs). **A30 started 2026-10-04 22:40Z** from a clean checkout of `e0cc22c1`: `--workers 4 --train-parts 8`, into `instruction_language/v11_20261004/` and `executor/v16_20261004/` |
| A30 (D73, D74) | Done: `instruction_language/v11_20261004`, `executor/v16_20261004` (read-only, `SHA256SUMS` checked), from a clean checkout of `e0cc22c1`, 22:40–23:10Z (30 min; closed loop 19.5 min, `--workers 4 --train-parts 8`). Against A25: signals, sentences and executor spec arrays equal (format names apart, A29's `labeller_code_sha256` gone); select and val closed-loop arrays equal; every D34 reading equal (closed loop and the 6 replays, 0 differences); train closed-loop `states` differ on 6–7 of ~40,500 flights per Δ by at most 9.1e-13 m (8 s; `vertical_m` 23 rows by 6.8e-13 m), words, corrections, outcomes equal — Claude's reading, not verified separately: float rounding from the different chunk mix of train read in parts. The stored outcome equals the replay's on every replayed flight (8,195 / 8,195); the start check passed (0 m). Report `readouts/2026-10-05_stage_a_a30_report.zh.md`. A25's `v10_20261004` / `v15_20261004` deleted on the user's go (2026-10-05; nothing read them) |
| The merge; A23 published from A30 (outline §6) | On the user's word: `dev-two-tier-v4` merged into `dev-two-tier` (`5c5fb4d3`), then `dev-two-tier-v4` fast-forwarded to `dev-two-tier` (`271bd7e7`). A23's export from `v11_20261004` / `v16_20261004` at `5c5fb4d3`: the set `closed_loop_v11_20261004` (KMSY, KRDU, KSJC, KSMF, KSTL; 20 train + 20 select flights an airport, seed 1337; Δ = 2, 4, 8 s) beside the old index; the main checkout's stack restarted (frontend 5173, backend 8765). Browser check by a one-shot agent: PASS on KRDU and KMSY (the set opens, the Δ tabs and outcomes show, a word's live segment answers in about 70 ms). `check_live` over the whole set, one process an airport (2026-10-04 23:46–23:59Z): 34,251 live segments, 0 differing from the export (KMSY 6,205, KRDU 7,242, KSJC 5,999, KSMF 7,413, KSTL 7,392); each process's closed-loop check first, 150 reference flights at 0 m. The user, after it: later checks of a set fly a sample, not every word |
| A36 (stage B's request) | On `dev-two-tier-v4-a36` (from `dev-two-tier` `bb2ef9b6`), worktree `.claude/worktrees/two-tier-v4-a36`: `354ff90d` — `split_flights` in `experiments/training_export.py` (the body of `build_airport`'s loop over the splits, moved unchanged; stage B's reference patch applied as it was), the stratum and kind read off the first Δ given, `formal_rows(executor, split, intervals=ROW_INTERVALS_S)`. Reviewed (an independent reviewer: no behaviour change; three low points fixed before the commit: a refusal by name for chosen flights of several airports or no row interval, a test with two Δ in their order, no global patch in the test). Byte for byte: each airport's sample built by the export's own steps from `v11_20261004` / `v16_20261004` with the published set's `git` record — KRDU before (`bb2ef9b6`) and after, the other four after — equals the published `closed_loop_v11_20261004` sample (`cmp`; KRDU sha256 `2b1bc145…`, 7.6 MB, again after the review's fixes). Full suite (`run_all_tests.sh`) at `354ff90d`: ts 1,535 passed + 1 skipped, modeling + backend 1,005, aeroviz-4d 161, exit 0. `dev-two-tier-v4` fast-forwarded to `354ff90d` on the user's go (2026-10-05); the worktree and the branch label removed |
| A32 (D77–D86) | Done on `dev-two-tier-v4-a32` (worktree `.claude/worktrees/two-tier-v4-a32`): `b7790be0`, one commit on `dev-two-tier-v4` `354ff90d` (A36 inside); nothing built. D77 start rules (`autopilot/params.py` `START_RULES`, `flights.start_state` / `observed_rows`; executor spec `ts-executor-spec-v10`, `executor_spec --start-rule`, the centred fit refused by `write_spec` and `load_spec`); D79 `autopilot/ends.py` (one definition for the judge, the batch, the staggered batch and `single.py`); D80 `Loop.step` (`RowRefused`); D81 the chart at the airport reference and the conformance's way `moved` (reference `ts-executor-conformance-reference-v5`; the name scan in `test_architecture.py` removed); D82 `ClosedLoopSentence(rows, withheld)` read by `closed_loop_sentences(artefact, split, Δ, spec)` (closed-loop format `ts-instruction-closed-loop-v8`); D83, D84; D78 candidates `ts-instruction-candidates-v4`; D85 the closed loop's summary; D86 the export (sample `aeroviz-training-sample-v10` on both sides); the references with up to 10 train flights with a labelled go-around an airport (labeller reference v6, closed-loop reference v8); the C32 checks; `keep_spec`; "unspecified" past an approach refused by name; A23's code through `start.require_startable` and `single.fly`; the backend keeps a failed check (a `ValueError`) and refuses val; `check_live --flights-per-airport`. Reviewed by two independent reviewers (executor core: no behaviour change beyond D77–D84, the three ways to fly within 1.1e-13 m on a cross-check; reader, runners, export: one high point, below, and medium points fixed: the kind read off the closed-loop words, memory, the backend's kept errors), fixes re-reviewed; full suite at the commit's tree: ts 1,548 + 1 skipped, modeling + backend 1,006, aeroviz-4d python 161, vitest 691. **The user's choices** (2026-10-05, asked by Claude): D77's window cut at row 0 (row 0 takes rows 0 and 1) and one start state for every start of the executor (above); `--list-candidates` compares with the former rule taken again on the same arrival manifests (`runway_targets`) instead of reading `v11_20261004`'s `candidates.json` (a v3 file, refused by the code) — **the design text of A32 ("against a given artefact") and A33 step 3 ("against `v11_20261004`") is the design author's to change**; Claude checked read-only that `v11_20261004`'s candidates equal the manifests' `runway_targets` at all five airports (the manifests unchanged since 2026-09-23) and that D78's candidates equal them too: on the live harvest D78 adds and removes no end (KSTL 06 stays, by its published vertical path). **Claude's readings (proposals)**: a rule's slope is turned from the airport frame's metres into metres on the ground (WGS84 radii at the row); a halted or done flight's row is not read by the grammar but is stored (the closed-loop reading gives a refused flight's held words at its first row); a flight's kind in the export is whether its closed-loop sentence at the first Δ says a go-around (as A23's replay rows had it); of val, the summary keeps the flights drawn and the sentences written; the backend keeps a `ValueError` raised by the checks (a malformed reference included) until a restart. A defect outside the milestone, recorded in `docs/code-health-followups.md`: two runway ends on one centreline halve the landing screen to float noise (no such pair at the five airports). **For stage B after the merge** (outline §5 rule 1): `prior_training_export.py` calls `split_flights(..., formal_rows(...))` and imports `formal_rows` (now `split_flights(instructions, split, chosen, (Δ,), params, words)`), `prior_free_generation.py` calls `closed_loop_sentences(load_closed_loop(...))` (now `closed_loop_sentences(artefact, split, Δ, spec)`, the sentence's fields under `.rows` and `.withheld`), and its backend calls `closed_loop_batch(drawn, stored, Δ, words)` (now with the artefact, split and params) |

### 0.4 Plan

1. A0–A31 are done (§0.3): A30's formal artefact (`instruction_language/v11_20261004`, `executor/v16_20261004`,
   read-only with `SHA256SUMS`; report `readouts/2026-10-05_stage_a_a30_report.zh.md`), and A23's Training sets
   published from it (`closed_loop_v11_20261004`). The user chose H_final = 10 m (D66) and Δ = 4 s (D11), the latter on
   the readings of D34 of A25, which A30 gives again.
2. Claude's review of stage A (2026-10-05; the user's question: data leaks, and information that the executor must not
   have) gave D77–D84 and outline D85. Their corrections, and one request of stage B (A36), come before the formal runs
   of stage B, which read the artefact of A34:
   1. A36 first (§12.1): the export of A23 gives the closed-loop part of chosen flights as one function
      (`split_flights`), for stage B's Training view (prior B6); a change of the code only, output unchanged. On its
      own branch, so that B6 does not wait for A32.
   2. A32: the corrections in the code (§12.1), on a new branch; nothing is built.
   3. A33: the measurements for the user's choices, on all train days, in a scratch directory (D55): the start rule
      of D77, the candidates of D78, the climb nominal. The user chooses the start rule.
   4. A34: the formal artefact once more, with A32's code and the chosen rule (`v12_<date>`, `v17_<date>`). Its report
      reads train and select only (D85).
   5. A35: A23's Training sets again, from A34's artefact. Then the user merges, and `v11_20261004`,
      `v16_20261004` and the sets of `closed_loop_v11_20261004` are deleted with the user's go.
   6. Claude's check of A32–A36 (§12.2, item 8).
3. D86 (the export reads no formal replay row, so it can give val flights) goes into A32's code.
4. The replay of the val days waits for the user.

---

## 1 Scope

- **This document owns** the packages `instructions/` (the vocabulary, the labeller, the artefact) and `autopilot/` (the
  executor, the judge, the replay, the closed-loop reading), and their runners.
- **It reads** the outline only: the principles (outline §2), the shared decisions D7, D20, D21 and D55 (outline §3) and
  the rules of the implementation (outline §5).
- **It gives** the public interface of §6, and nothing else, to the other documents. A change of §6 is a change of a
  format: it gets a new name (principle 8), and the documents that read it are changed with it.

---

## 2 Terms

| Term | Meaning |
|---|---|
| Row, step | One line of a sentence. A row is Δ seconds, on the UTC multiples of Δ. The data's rows are 2 s; the ablation of §4.8 reads Δ = 2, 4 and 8 s (D25), and the user chose Δ = 4 s (D11) |
| Sentence | The rows of one flight, from its first row to the row before the landing |
| Column | One kind of word in a row. A row has five columns (§3.1) |
| Word | The value of one column in one row. The value "unchanged" means that the column says nothing new |
| Word in force | The last word that a column said |
| Runway in force (R) | The candidate runway that the runway column said last |
| Go-around state (G) | True after the word "go-around", false after a runway word (§3.2) |
| Candidate runway | A runway end of the airport with a published vertical path in the FAA CIFP, and its geometry (D78). No flight decides a candidate |
| Course | The true direction of a candidate runway, in compass degrees |
| Relative heading | The target track minus the course of R, in (−180°, +180°]. Positive is clockwise |
| Final | The extended centreline of R, before the threshold, in the direction of the course |
| Capture corridor | On the final: lateral offset ≤ 20 m + d·tan 0.45°, track within 2° of the course (d: distance to the threshold). Spec values (§8) |
| Approach | The rows of a sentence from its first row, or from the runway word that ends G, to the landing or to the next go-around row (D26) |
| Go-around row | The row where the runway column says "go-around". In a labelled sentence: the first row of the climb after the low pass (§4.6) |
| Capture row | The labeller's first row from which the observed track stays in the capture corridor to the end of its approach (D26). Each approach has at most one. Only the labeller uses it (D4). It uses later rows, so it is never an input |
| Go-around angle | The angle of a climb word while G is true: a value of the executor, 1.885°–3° (D28, §5.5) |
| Open-loop reading | The labeller's reading of the words from the observed track alone (§4.1–§4.8). Its words are the observed words |
| Closed-loop reading | The labeller's second pass: it flies the sentence with the executor and adds correction words (D32, §4.9) |
| Flown states | The states of the executor in the closed-loop reading, on the data's 2 s rows. The training sentences carry them (§6) |
| Correction word | A word that the closed-loop reading adds to bring the flown path back to the observed path (§4.9) |
| Matched point | The point of the observed path nearest to the flown position, searched forward from the matched point of the row before (§4.9). The closed-loop reading compares the flown state with the observed path there, and says the observed words there (D42) |
| First predicted step | The row 16 s after row 0 (8 rows at 2 s). The rows before it are observed only: no word is said there. A sentence says every column at it, and the closed-loop reading starts there |
| Executor cycle | 1 s. The executor hears the words at the start of each row |
| Labeller | The program that reads a sentence from an observed track (`instructions/labeller/`) |
| Judge | The part of the executor package that classifies what the executor flew (`autopilot/judge.py`) |
| Airport elevation (E) | The published elevation of the airport (`candidates.json` `reference.elevation_m`). The altitude words are heights above it (D58) |
| Decision altitude (DA) | The published DA of the LPV minimums of R, as a height above the threshold (`Runway.decision_height_above_threshold_m`) |
| DA point | The first flown row where the aircraft, on the final of R and with G false, descends through the DA |
| Final approach segment (FAS) cone | The LPV lateral containment: half-width 106.7 m (350 ft) at the threshold, larger with distance (FAA Order 8260.58D Formula 3-1-1, `flight_scenarios/fas_geometry.py`) |

---

## 3 Vocabulary

### 3.1 Columns and order

| # | Column | Values (with "unchanged") | Classes |
|---|---|---|---|
| 0 | Runway | a candidate runway (pointer), "go-around" | candidates of the airport + 2 |
| 1 | Heading | relative heading, 5° grid (72 values) | 73 |
| 2 | Altitude | target level, geometric height above the airport elevation, 40 levels on three segments (60 / 120 / 450 m steps), 0–5,400 m; "no level-off" | 42 |
| 3 | Angle | level; descent 1–4; climb | 7 |
| 4 | Speed | target ground speed, 5 m/s grid, 20–250 m/s (47 values); "unspecified" | 49 |

In one row a speaker says the columns in the order of the table. A later column sees the values that the earlier columns
said in the same row.

### 3.2 Runway column

**Meaning of a runway word.** "Expect runway k." The executor uses R for two things only: the conversion of the heading
words (§5.4) and the distance for the deceleration to the approach speed (§5.6). The judge uses R for the landing
(§5.8).

**Regulation.** Approach control gives the expected approach and runway at the first contact (FAA JO 7110.65BB
4-7-10 a: "Approach clearance or type approach to be expected ... Runway if different from that to which the
instrument approach is made"). Thus the runway is known before the approach clearance. The approach clearance is an
authorization ("CLEARED (Type of) APPROACH − ATC authorization for an aircraft to execute a specific instrument approach
procedure", Pilot/Controller Glossary). After it, the aircraft flies the procedure (AIM 5-5-4 a.3). The labeller cannot
see when a controller gave it: the data has no radio. Thus this design has no clearance word. The model says the turns
with heading words (D2).

**States.** The column keeps two states: R (the runway in force) and G (the go-around state).

| Word said | R after it | G after it | Permitted when |
|---|---|---|---|
| candidate k at the first predicted step | k | false | always; the first predicted step must say a candidate |
| candidate k later, G false | k | false | k ≠ R (the word in force is not said again) |
| candidate k, G true | k | false | any candidate, R included |
| go-around | R (no change) | true | G false, after the first predicted step |

**No runway lock (D12).** While G is false the model can change the runway at any time, also on the final (a change to
a parallel runway is a real manoeuvre). The judge reads R at the crossing.

**Go-around (D10): "abandon this approach".** A change of runway does not stop a descent and does not tell the judge
anything. A go-around does. It does not change R: R stays the frame of the heading words while the model vectors the
aircraft back. Its effects:

| Who | Effect of "go-around" |
|---|---|
| Executor (§5.4–§5.6, D27, D28) | "Go-around" changes no target. While G is true, a climb word climbs at the go-around angle (1.885°–3°), not at the nominal angle of the climb class, and "unspecified" holds the airspeed. The climb comes from the level that the go-around row says (rule 6). The heading word in force stays |
| Judge (§5.8, D33) | While G is true, no crossing of a threshold is an event, of R or of another candidate (not a landing and not a failure): the flight continues. A go-around before the DA point answers a failed DA check. The time limit of the flight grows by 900 s (the user, 2026-10-02) |
| Grammar (§3.7, D14, D27) | Rule 5: no "no level-off" while G is true. Rule 6: a go-around row with "no level-off" in force also says a level above the aircraft |

A runway word ends G. The model can say R again or another candidate ("expect runway k" again). Thus one column says
the sequence approach → go-around → approach.

**Why "go-around" changes no target (D27).** The model flies the final with its own heading and altitude words (D2,
D3). An executor target for a go-around would replace a word of the model that is in force: a climb against "no
level-off", the course of R against the heading word. Then a target would come from the executor and not from a word
(§5.1). Thus a go-around changes only the angle of a climb word and the meaning of "unspecified" (§5.5, §5.6). The model
says the climb and the heading.

### 3.3 Heading column

**Meaning.** "Fly track θ and keep it", with θ = course(R) + 5k°, k the class of the word (D8). The executor turns in
the shorter direction for the first heading word (Pilot/Controller Glossary, FLY HEADING). Class 0 is the course of R.
Class 36 is the opposite direction (downwind). Classes 18 and 54 are the two base legs.

**Why a relative grid (D8).** Two reasons, both measured on the v6 train split (§9):

1. **The final must be on the grid.** The capture corridor permits 2° from the course. The four courses of KSTL (62.7°,
   122.3°, 242.7°, 302.3°) are 2.26–2.32° from the nearest absolute 5° value. With absolute words, no heading word holds
   a KSTL final. With D2 and D3 the model must hold the final with heading words.
2. **The same manoeuvre gets the same word at all airports.** The heading in force at the end of the labelled turn onto
   the final is very different between airports in absolute classes. It is much more similar in relative classes
   (§9.1). This helps a model to use the data of one airport at another airport.

**Conversion and change of runway.** The executor converts a heading word to an absolute track when it hears the word.
It keeps that absolute target until the next heading word. A change of R does not turn the aircraft.

**The frame where a word is heard (D46).** The labeller says each heading word relative to the course of the R in force
when the executor hears it. Where the Δ grid (§4.8) or the closed-loop reading (§4.9) moves a heading word across a
runway word, the word is the class nearest to its absolute track minus the course of the new R. For parallel runways
(courses 0.002–0.01° apart) the class does not change; for other runways the absolute track stays within half a step,
and the closed-loop reading corrects the rest. A class gives a track only with its course: a heading word is said when
its absolute track differs from the track in force, also when its class is the class in force said under another course
(the executor takes a heading word as new because it is said, not because its class is new). A change of R alone says no
heading word.

**Reading.** At each row, the labeller takes the smoothed track L = 4 s later (the lead; near the end, the track of the
last row). The word is the grid value, relative to the course of R at that row, nearest to that track. A new word is
said where that track leaves the target in force by more than half a step (2.5°). Thus a continuous turn is a series of
words, one for each 5°, and the time between the words gives the turn rate. The words continue from row 0 to the last
row of the sentence: they describe the turns onto the final and the final itself (§4.3).

**Envelope.** A word said at row r: from row r + L to the next heading word + L, the track is within ±4.5° of θ
(`envelope.heading_words_inside`). The rows of a word continue to the next heading word or to the end of the
sentence.

**Grid (D6, D35).** The grid is 5°. A word of class 0 gives no lateral correction: the aircraft keeps its lateral
offset until the model says a word of ±5°. The corrections come from the closed-loop reading (D32, §4.9), not from a
finer grid (§9.6).

### 3.4 Altitude column

**Meaning.** "Descend (or climb) to level T and keep it" (7110.65BB 4-5-7, DESCEND AND MAINTAIN). T is a geometric
height above the airport elevation E (D58); the executor flies it at T + E MSL. The direction comes from T and the
present height. The angle column says how steep. In this section a height is a height above E.

**Reference (D58).** E is one value for each airport: the published field elevation. It is not the threshold of R: R
can change during a sentence, and a level word must not change its meaning with it (as a heading word keeps its track,
§3.3); and the thresholds of one airport differ by up to 28 m (KSTL 160.8–188.4 m), about half a step of the grid.

**Grid (D22, D58).** 40 levels on three segments, each segment uniform:

| Segment | Levels | Step | Largest rounding error |
|---|---|---|---|
| 1 | 0, 60, …, 1,260 m | 60 m | 30 m |
| 2 | 1,380, …, 2,700 m | 120 m | 60 m |
| 3 | 3,150, …, 5,400 m | 450 m | 225 m |

The steps and the break points are constants of the vocabulary and go into the spec. A word is the nearest level; its
meaning is a height above the airport, the same at every airport and from every position. The grid of the table is an
exact dynamic-programming fit on MSL level-offs: at most three uniform segments, break points on a 15 m grid, steps from
a fixed set of 15–600 m, that make the squared rounding error of the observed level-offs smallest for 40 levels (§9.5).
The spec measurement makes the same fit on the level-offs above E of the train days and writes it beside the grid of
the table, each with its rounding error; the user chooses (D55, D58).

**Why a coarser grid higher up.** (1) 99.7 % of the level words are under 3,000 m (§9.5). (2) A controller assigns a
pressure altitude, which the altimeter setting of the airport makes correct at the airport. The geometric height of one
assigned level moves from day to day by about h × ΔT / 273, with h the height above the airport and ΔT the day's
temperature deviation (the model does not see it); ICAO gives its cold-temperature corrections by the height above the
altimeter-setting source (Claude's reading, the text not checked). At KRDU (E = 133 m) this is about 15 m at 600–900 m
MSL and about 90 m at 2,000–3,000 m MSL (§9.5). A fine step high up only divides this spread.

**What a coarse segment costs.** In segment 3 two assigned levels 1,000 ft (304.8 m) apart can round to one word, and
a small change of level inside one step has no word. Such words are rare (above 2,700 m: approximately 0.3 % of the
words, mostly the entry level at row 0). The 1,000 ft between opposite base legs (7110.65BB 5-9-1 b) is at low
altitude, where the step is 60 m. Because the heights are above the airport, the final approach lies in the 60 m
segment at every airport.

**"No level-off".** "Descend at the angle in force. Do not level off." The executor flies the nominal angle of the
class in force and has no other law for it (§5.5, §5.7). The judge stops the flight at the threshold or at the ground.

**Why above the airport and not MSL (D58).** A word must mean the same manoeuvre at every airport (principle 4). In MSL,
a high airport is outside what the training airports said: at KDEN (E approximately 1,650 m) an intercept of the final
600–900 m above the airport is at approximately 2,250–2,550 m MSL, a height that the five training airports (E = 1–188
m) use only far out, and it falls in the 120 m segment, a downwind in the 450 m segment. Above E, the same manoeuvre is
the same word, in the 60 m segment. The spread of an assigned level also grows with the height above the airport, not
above the sea (the grid above). The cost: controllers assign round MSL levels, and above E one assigned level is a
different word at each airport (3,000 ft MSL: 900 m at KMSY, 780 m at KRDU, 720 m at KSTL). A speaker that does not get
E cannot know where the round MSL levels lie; its level words spread over one to three steps more. This
changes where the aircraft levels by at most one to three steps; the final is held with angle words and the corrections
of §4.9, not with level words. The level words of the five airports differ mainly by their airspace in both frames
(§9.1: KSJC has 28 % of its level words on levels that the other four airports almost never use); that measurement
covers only airports between 1 and 188 m.

**Envelope.** The tube from the row of the word: max(T, h0 − s·tan γ_hi) − ε ≤ h ≤ max(T, h0 − s·tan γ_lo) + ε while it
descends, T ± ε after it arrives (s is the horizontal distance flown from the row of the word). ε depends on the level
(D52): **ε of a level = half the larger gap to its neighbouring levels + 10 m** (the fit residual). That gives 40 m for
0–1,200 m, 70 m for 1,260–2,580 m and 235 m for 2,700–5,400 m. The band covers the largest rounding error of the word:
the top level of a segment collects the heights up to half the next segment's step above it (60 m above 1,260 m, 225 m
above 2,700 m). For "no level-off" there is no lower bound T, and **ε = 40 m**: the word has no level, so no rounding to
cover, and the tube's width comes mainly from the edges of the angle class. A new angle word starts a new tube. A
containment rate is given with these widths (principle 6). The level detection of the labeller does not use ε (§4.4).

### 3.5 Angle column

| Class | Nominal | Range |
|---|---|---|
| Level | 0° | used only to hold a level that the aircraft reached |
| Descent 1 | 1.5° | −0.5° to 2.0° |
| Descent 2 | 2.5° | 2.0° to 2.75° |
| Descent 3 | 3.0° | 2.75° to 3.75° |
| Descent 4 | 4.5° | 3.75° to 10° |
| Climb | No angle in the word. The executor's angle (D28): the nominal of D15 while G is false (1.5°, D56); 1.885°–3° while G is true | 0.5° to 15° climb (the labeller's range of a climb piece) |

The four descent classes are a k-means on tan(angle) of the descent pieces of the train days, with each piece weighted
by the square of its length (D54): a piece flown at the nominal angle of its class ends length · |tan a − tan c| from
its observed end, so the k-means makes the sum of the squared end-of-piece height errors smallest. The values of the
table are the user's choice (D56): the fit on all train days rounded to 0.25° (fitted 1.51 / 2.45 / 3.09 / 4.45°, edges
1.98 / 2.77 / 3.77°, climb 1.47°). The old labeller's pieces (artefact `v6_20261002`) gave 0.92 / 2.13 / 3.06 / 4.41°
under the same rule: the same flights and a median piece angle of 2.98° against 2.97°, but its shallowest class (12 % of
the weight at 0.92°) is not in the new pieces.
With D3 the model holds the glidepath with these classes; the closed-loop reading gives the corrections (D32, §4.9).

**Rounder values (D15, D56).** The spec measurement writes each value that it fits from data (the descent nominals and
edges, the climb nominal) with candidates rounded to 0.5°, 0.25° and 0.1°, and for each candidate the end-of-piece
height error that it leaves on the descent and on the climb pieces (`instructions/measure.py` `rounding_candidates`).
The spec takes the row that the user chose (`instruction_spec --candidate`; D56: 0.25°). Descent 3 is then the 3.0°
glidepath that 24 of the 25 candidate runways publish (§9.3). The values that the measurement takes as percentiles
are rounded by their own rules (turn rates to 0.1°/s, the bank limit to 1°, the corridor to 5 m, 0.05° and 1°, the
acceleration to 0.1 m/s²).

**Climb (D28).** The vocabulary has one climb word: "climb". The word has no angle. The angle is a value of the
executor (§5.5), from the state G:

- **G true (a missed approach).** The executor climbs at the go-around angle: the steady climb angle that the thrust
  limit permits at the present airspeed, not more than 3° and not less than 1.885°. 1.885° is the minimum gradient of a
  missed approach, 200 ft per NM (AIM 5-4-21 b). 3° is the upper limit. The real go-around climbs are steeper (§9.4),
  so the 3° limit usually applies.
- **G false.** The executor climbs at the nominal angle of the climb class, 1.5° (D56): the length-weighted median of
  the climb pieces of the train days (1.47°), rounded to 0.25°. The train days have 1,339 climb pieces against 227,559
  descent pieces (0.6 %).

The labeller does not read a go-around angle: a climb piece is a climb when its angle is 0.5° to 15°, in and out of G.

### 3.6 Speed column

**Meaning.** "Hold ground speed V." Controllers assign indicated airspeed; the data has no airspeed and no wind, so the
words use ground speed. "Unspecified" means that the pilot flies the approach speed of the aircraft type. An approach
clearance cancels the assigned speeds (7110.65BB 5-7-1 d). In this design the labeller starts "unspecified" at the
capture row (D4, §4.5).

**Steps (D43).** In a change of speed, the labeller says each grid value on the way (§4.5). Each speed word is then a
step of 5 m/s, and the time between the words gives the rate of the change, as the time between the heading words
gives the rate of a turn. The executor makes each step at a_max (§5.6).

**Envelope.** A transition is monotone (a step back ≤ 5 m/s) with acceleration ≤ a_max (1.4 m/s², spec).
After it, the speed stays within V ± 5 m/s. "Unspecified" checks only the range.

**Wind (O8, open).** At constant airspeed, the ground speed changes when the aircraft turns in wind. Such a change can
become a speed word that no controller said. A check against the surface winds of the METAR data is open.

### 3.7 Grammar rules

The labeller checks these rules. A speaker applies the same function as masks when it decodes
(`instructions/grammar.py`, §6), column by column: a word of a column is permitted when some words of the later columns
of the row, each among the words that the speaker's other masks permit, make the row pass the rules (D62). The labeller
conformance (§7.2) covers them.

1. At the first predicted step, each column says a value. The runway column says a candidate.
2. The runway column follows the table of §3.2.
3. In a row that says an altitude word or an angle word, the level T in force and the angle in force agree at the
   height of the aircraft: a T more than its ε (§3.4) below the aircraft needs a descent class, a T more than its ε
   above it needs the climb class.
4. "No level-off" needs a descent class in force or in the same row.
5. "No level-off" is not permitted while G is true (D14). The model first says a runway word, which ends G; in the same
   row (the runway column comes first) or later it can say "no level-off".
6. A row that says "go-around" while "no level-off" is in force also says a level T more than its ε above the present
   height of the aircraft (D27). Rule 3 then makes the same row say "climb". With rules 5 and 6, "no level-off" is never
   in force while G is true, so every climb of a go-around has a target from an altitude word.

### 3.8 Example

A KRDU arrival that lands on 23R (course 225.0°). At row 0 it is on the left downwind of 23R: 3.0 km before the
threshold, 5.0 km left of the final, track 045°, level at 1,080 m above the airport (E = 133 m; 1,213 m MSL), 115 m/s.
Only rows with a word are shown; an empty cell is "unchanged". Relative headings and heights above the airport are
given.

| t (s) | Runway | Heading | Altitude | Angle | Speed | What happens |
|---|---|---|---|---|---|---|
| 0 | 23R | +180 | 1,080 m | level | 115 m/s | Expected runway 23R. Downwind, level |
| 40 | | | | | 105 m/s | Speed reduction |
| 106–134 | | +175 … +90 (one word each row) | | | | Left turn onto the base leg, 4 s before the track |
| 120 | | | 780 m | descent 2 | | Descend to 780 m above the airport |
| 156–200 | | +85 … 0 (one word each row) | | | | Left turn onto the final, with heading words (D2) |
| 204 | | | | | unspecified | Capture row; the pilot's own speed (D4) |
| 210 | | | no level-off | descent 3 | | Final descent |

At approximately t = 390 s the aircraft crosses the threshold of 23R. The judge classifies the crossing and the DA
check (§5.8). The table is the labelled sentence. Rows 0–7 are observed only. The first predicted step is row 8
(t = 16 s); there the sentence says the five words in force at that row.

## 4 Labeller

The labeller reads one sentence from one observed flight (`instructions/labeller/read.read_flight`), in two passes. The
open-loop reading (§4.1–§4.8) reads the observed words from the observed track. The closed-loop reading (§4.9) flies
them with the executor and adds the correction words; its sentences are the training sentences.

### 4.1 Signals, cut and gate

- **Signals** come from the ts data plane (`build_series`): the altitude repaired at read time, MSL, 2 s rows on even
  UTC seconds, the airport frame. The test days are sealed: the labeller never opens them (contract C32). The labeller
  reads the altitude words from the height above the airport elevation, MSL − E (D58).
- **Smoothing:** centred moving means of the track (6 s), the altitude (10 s) and the ground speed (10 s). The flown
  distance is the integral of the smoothed ground speed.
- **Cut** (`read.admit`). A landing passage is a row where the flight crosses the threshold plane of its landed runway
  inside the landing screen of the harvest: lateral ≤ 1,000 m and ≤ half the spacing to a parallel runway (a candidate
  whose course is within 5°), height ≤ 100 m, both interpolated between the two rows (`read.landing_passages`). The
  sentence ends before the last landing passage after which the flight does not come back before the threshold.
- **Gate.** The labeller refuses a flight, with its reason, when: its rows are not 2 s apart; its landed runway is not
  a candidate; fewer than two rows remain before the threshold; a raw ground speed is outside 15–350 m/s; a smoothed
  ground speed is outside the speed words (20–250 m/s ± 5 m/s); a landing passage that it comes back from is not inside
  the low pass of a go-around (§4.6); a go-around point is on the runway (§4.6). The readings of §4.2–§4.7 add their own
  refusals.

### 4.2 Runway word

The first row says the runway of the first approach (D26). For a flight without a go-around, this is the runway on
which the flight landed (the harvest assignment). For a flight with a go-around, it is the runway of the first low pass
(§4.6). The data shows a change of runway only at a go-around, so a labelled sentence changes R only with the runway
word that ends G (D19). A candidate without a published threshold crossing height (TCH), a
glidepath angle or an LPV DA is refused before labelling (the judge needs all three, §5.8). The first runner writes
the three values of each candidate into `candidates.json` (D61).

### 4.3 Heading words

- **Frame:** the course of R at each row (D8): the runway of the first approach, then, from the runway word that ends
  a go-around, the runway of the next approach (§4.2, §4.6).
- **Reading:** §3.3, at every row from row 0 to the last row of the sentence. A word in force keeps its track when R
  changes; only a new word reads the new course. After a change of R, a new word with the class in force is said,
  because its absolute track differs from the track in force (D48, as D46 on the Δ grid and in the closed loop).
- **Capture row** (D4, D26): the first row of an approach from which the track stays in the capture corridor of its
  runway, before the threshold, to the end of the approach. The landing approach must end in the corridor; otherwise
  the flight is refused. An approach that ends at a go-around row outside the corridor has no capture row.
- **Strata of the readouts.** A flight is vectored when, before the capture row of its landing approach, the net turns
  of its runs of rows that turn one way faster than 0.2°/s add up to 90° or more. Otherwise it is straight-in.
  The sentence file stores each labelled flight's stratum (§6, item 3; D70).

### 4.4 Altitude and angle words

**Pieces.** The smoothed height above the airport elevation is fitted against the flown distance by straight pieces.
From the first row of a piece, the piece grows as long as the least-squares line through its rows keeps every residual
within 10 m (the fit residual). A piece has at least two rows.

1. **Level.** A piece is level when it lasts at least 20 s and every row is within 25 m of the piece's own median. This
   test is physical and does not use the grid: a test against the 60–450 m steps would miss a level between two grid
   levels (a level at 630 m is 30 m from both neighbours) or take a slow descent for a level. The word of a level is the
   grid level nearest to its median. Consecutive level pieces with the same word are one level.
2. **Move.** Every other piece is a move, with its path angle (descending positive). Consecutive moves in one direction
   are a run.

**Words.**

1. A level at row 0 says its level and the angle "level".
2. A run says its altitude word at its first row: the word of the next level; the grid level nearest to the height
   where the direction changes without a level; or "no level-off" when the run is the last descent of its approach (it
   reaches the threshold, or the go-around row; D26, §4.6). A run whose word is the word in force (a move between two
   heights that round to one level) says nothing: the word cannot be said (§3.4).
3. Each move says its angle class where the class differs from the class in force: the descent class whose range holds
   the angle, or "climb" for a climb of 0.5° to 15° (§3.5).
4. A step is two levels with different words and no move between them. The altitude word is said at the last row of
   the first level. Its angle is read from there to the first row within 10 m of the height of the second level.
5. The flight is refused when it climbs at the end of the sentence, when a level is outside the grid, when an angle is
   outside the classes, or when a level word is on the wrong side of the height where its run starts (by more than its
   ε).

The tube of §3.4 checks each word, with the ε of its level (D52).

### 4.5 Speed words

**Pieces.** The smoothed ground speed is fitted against time by straight pieces, as the altitude (§4.4), with every
residual within 1.5 m/s. A piece is a hold when it lasts at least 20 s, its slope is at most 0.1 m/s², and every row is
within ±5 m/s of the grid value nearest to its median. A hold that follows a hold and stays inside the band of its value
is the same hold. Every other piece is a transition, accelerating or decelerating. Consecutive transitions in one
direction are a run. The target of a run is the value of the next hold; or the grid value nearest to the speed where the
direction changes without a hold; or, when the run continues into "unspecified" or to the end of the approach, the grid
value nearest to the speed at its last row before that.

**Words.**

1. Row 0 says the value of the hold that starts there, or else the grid value nearest to the smoothed speed there.
2. A hold that follows another hold says its value at its first row.
3. **Steps (D43).** In a run, the labeller says each grid value from the word in force to the target, in order. It says
   a grid value at the first row at which the smoothed speed is nearer to that value than to the value before it, and
   the target at the latest at the last row of the run. Thus, in a change of speed, the word in force is approximately
   the grid value nearest to the observed speed. Inside a run the labeller says no value in the other direction, so the
   noise of the speed gives no words.
4. A target outside the grid (20–250 m/s) refuses the flight.

**"Unspecified" (D4, D26).** Each approach has its own reading. "Unspecified" starts at the capture row of the
approach, with two exceptions:

1. A hold of at least 30 s that ends at or after the capture row, and ends 9,260 m (5 NM, 7110.65BB 5-7-1 b.4) or
   more before the threshold, keeps the speed word in force. "Unspecified" starts at the end of the last such hold.
2. A change of speed under way at that row, which started less than 20 s before it, is already the pilot's speed.
   "Unspecified" starts where it started.

From that row to the end of the approach the column says no other word. An approach that ends at a go-around row
outside the capture corridor has no capture row: its speed words continue to its end, without "unspecified". The
reading of the next approach starts at the go-around row.

**Why steps (D43):**

- A speed word gives a target, not a rate. The rate of a real change of speed is a choice of the pilot or the
  controller. On the smoke artefact it is 0.24 m/s² in level flight and 0.19 m/s² in a descent (median), and it changes
  by more than three times in each (§9.8). It does not follow from the aircraft.
- The dynamics cannot give it. Their drag polar is clean: no flaps, no landing gear, no speedbrakes. The thrust floor of
  −20 % of the installed thrust is a stand-in for these devices (`outputs/envelope.py`). Thus "the deceleration that
  the aircraft can do" is not a physical value in this model.
- One target word with a fixed rate of the executor moves the flown aircraft ahead of the observed aircraft along the
  path, or behind it, by more than 1 km in 31 % of the flights (before "unspecified"). A faster rate makes this worse.
  With the steps and the executor at a_max, no flight is more than 1 km ahead or behind (§9.8, a one-dimensional
  estimate).
- The speaker says the steps. Thus the speaker, not the executor, sets the rate of a change of speed. A controller who
  keeps the spacing between two aircraft does this with speed.

### 4.6 Go-around words (D18, D19)

**The data.** The training days hold 109 go-arounds (R40 v2, `go_around_census`; §9.4). In 73 the go-around point
lies inside a labelled sentence of artefact `v5`. In 11 more it lies inside the arrival slice of a flight that crosses
a threshold plane in its low pass; the gate of §4.1 reads these as go-arounds. In 23 it lies before the sentence: the
aircraft left the 25 km slice, and the slice starts at its return.

**The reading.**

1. **The go-around** (`instructions/labeller/go_around.py`).
   - A low pass is a run of rows (gaps of at most 3 rows) on the final of one candidate: at most 500 m from its
     centreline, from 10 km before its threshold to 3 km past it, at most 600 m above it, with at least 1 km of
     progress along the landing direction between its first and last rows (it flies the final; it does not cross it).
     Two passes on two candidates with common rows (close parallels) are one pass, on the candidate with the smaller
     median offset. The lowest row of the pass is the go-around point.
   - A low pass is a go-around when, before it (since the previous pass), the aircraft held a level for at least 20 s at
     least 150 m above the go-around point, and after it (before the next pass or the end) held a level for at least
     20 s at least 150 m above the point. The last pass of a sentence (the landing approach) is never a go-around.
   - A go-around point past the threshold and at most 15 m above it is on the runway: a touch-and-go, or a landing
     balked in the flare. The vocabulary cannot say that landing, so the flight is refused.
2. The go-around row is the first row of the climb after the low pass: the row where the altitude reading starts the
   climb run, with its level word and its climb word (D26). The runway column says "go-around" in that row; G becomes
   true. A go-around without a climb word is refused, with its reason.
3. The climb is read as altitude and angle words, as every climb. Its level word and its climb word are in the
   go-around row (rule 6). While G is true the executor flies the climb at the go-around angle (D28, §5.5).
4. The runway word that ends G (D19): at the first level after the go-around row held at least 150 m above the go-around
   point (a level flown at the low point before the climb is not it), and not later than the row of the next "no
   level-off" (rule 5). It says the runway of the next approach: the runway on which the flight landed, after its last
   go-around. A go-around without such a row, or whose row comes at or after the next go-around row, is refused
   ("go-around not ended").
5. A landing passage that the flight comes back from must lie inside the low pass of a go-around; otherwise the
   flight is refused (§4.1).
6. Each approach is read separately (D26): the last descent that reaches the go-around row says "no level-off" (§4.4);
   the capture row and "unspecified" belong to the approach (§4.5); the first row says the runway of the first low pass
   (§4.2).

**Why each approach is read separately (D26).** A word must not tell a model what only later rows show (principle
7). Two readings use the end of an approach: "no level-off" is the last descent that reaches the end, and the capture
row needs the track in the corridor up to the end. If the end were the landing for every approach, the words before a
go-around would be a low level and a speed value, never "no level-off" or "unspecified" (§9.4). These words then tell
a model that this approach will not land, a fact that only the rows after the go-around show. The model would also
learn "go-around" only after a low level and a speed value. In closed loop the model flies its final with "no
level-off" and "unspecified" in force, and there the model would give "go-around" a probability near zero, where the DA
check needs it (D3). With each approach read separately, "go-around" occurs with "no level-off" and "unspecified" in
force, as on a real final.

**Not read.** The 23 go-arounds before their sentence need a longer arrival slice, and 9 sentences of `v5`
end at a low go-around that the harvest took for the landing (it takes the best-aligned crossing under 100 m, not the
last; R40 v2 §5). Both are changes of the harvest or of the data plane, which other lines (evaluation, the optimizer,
the one-tier models) share.

### 4.7 Assembly and artefact

- All words go on the row grid (Δ; §4.8). A word equal to the word in force is not a word. The first row says all five
  columns. Two different words in one column in one row: refused.
- The reading is named `instruction-v4`. The spec records the grids, the classes, the tolerances and the reading name.
  The labeller conformance (§7.2) checks the reading code, the grammar included.
- The open-loop sentences are the input of the closed-loop reading (§4.9). The training sentences are the closed-loop
  sentences with their flown states (§6).
- The code refuses an artefact of another reading name (principle 8).

### 4.8 Row interval: an ablation (D11)

**What the row interval Δ sets.** The data plane gives a row every 2 s. Δ sets the grid of the words, how often a
speaker speaks and the executor hears, the length of a sentence (median 154 rows at 2 s) and the step of a scene of
several aircraft. Each column changes its word at only 1–2 % of the 2 s rows (artefact `v6_20261002`), so a larger Δ can
be sufficient. The ablation measures this at the labeller stage, before a model trains.

**How the ablation changes only Δ.** The labeller reads each flight at the 2 s rows of the data. Then it puts
the words on a Δ grid:

1. The Δ rows are the rows on UTC multiples of Δ, so the aircraft of a scene are on one grid.
2. Each word of the 2 s reading goes to the nearest Δ row; a word exactly between two Δ rows goes to the later one
   (D45). A Δ row says, for each column, the last word that goes to it, if that word differs from the word in force. A
   heading word that moves across a runway word is said in the frame where it is heard (D46, §3.3).
3. The first Δ row says all five columns. The grammar rules (§3.7) are checked again on the Δ grid. The rounding keeps
   the order of the words, and the words of one 2 s row stay in one Δ row (a level word and its angle word, a
   go-around row). A "go-around" and the runway word that ends it in one Δ row cancel. A sentence in a go-around at its
   first Δ row is refused (it cannot say a runway there). A word that goes past the last Δ row is not said.

**Why the nearest row (D45).** A word put on the next Δ row is late by (Δ − 2)/2 on average, and a word said when the
matched point has passed its place (§4.9) is late by Δ/2 more: approximately 1 s at Δ = 2 s, 3 s at 4 s and 7 s at 8 s.
A late heading word makes a turn end late: 3 s late at 70–100 m/s leaves approximately 200–300 m of lateral offset after
a turn of 90°. The ablation would then read the rounding, not the row interval (§9.9). With the nearest row, the mean
lateness of a word alone in its row is zero at Δ = 2 s and 1 s at 4 and 8 s (a word exactly between two rows goes to the
later one). A row that holds several words of a column says the last one, so the words said are early on average (D45).
The spread of ±Δ/2 stays; that spread is the cost of a coarser Δ.

Thus one reading gives every Δ. Δ must be a multiple of 2 s, and it must divide the 16 s before the first predicted
step (D25): the first predicted step is a Δ row.

**What a larger Δ changes (to read in the ablation; no criteria here, D7):**

| Part | Change |
|---|---|
| Heading words | In a 3°/s turn the track moves 12° in 4 s, so one word jumps two or three 5° classes. The lead L = 4 s is one row at Δ = 4 s and less than one row above it |
| Executor | The cycle stays 1 s. It hears the words at each Δ row. The heading law (arrive L after the hearing, the stopping rate) flies larger steps |
| Speed words | A change of speed faster than 5 m/s in Δ puts two steps (D43) into one row; the row says the last one, and the executor flies it at a_max |
| Final approach | With D2 and D3 the model corrects the final only every Δ; an error grows for a longer time before the next word |
| Speaker | Fewer steps for each flight and more changes for each step. The rows before the first predicted step stay 16 s (16 s / Δ rows: 8, 4, 2). A loss for each step cannot be compared between two Δ; a loss for each flight or each second can |
| Scene of several aircraft | The scene step is Δ |

**Values (D25).** Δ = 2, 4, 8 s. A Δ of 6 s is not used: 16 s is not a whole number of 6 s rows. The user chose
Δ = 4 s on the readings of D34 (D11); the artefact keeps the closed-loop sentences at all three.

**Readings (D34).** The closed-loop reading (§4.9) makes each replay follow the observed path, so the landed share no
longer shows how well a Δ carries a flight. At each Δ the ablation reads instead:

1. the correction words of the closed-loop reading, for each flight and each column;
2. the errors left where §4.9 makes no correction (a level hold, a climb, no steeper or shallower class): the largest
   |e_y| and |e_h| for each flight;
3. the share of the rows where §4.9 permits a correction and the flown path is outside the tolerances (|e_y| > Y,
   |e_h| > H; D50);
4. the replay outcomes (§5.8).

The user compares the Δ values on these readings; this document sets no threshold (D7).

### 4.9 Closed-loop reading (D32)

**What it does.** The open-loop reading (§4.1–§4.8) gives the observed words. Then the labeller flies the sentence with
the executor of the artefact's executor spec and compares the flown path with the observed path at each row. When the
flown path leaves the observed path by more than a tolerance, the labeller adds a correction word. The result is the
training sentence: the observed words, the correction words and the flown states.

**Why.** A heading word gives a direction, and an angle word gives a path angle. Neither gives a position. The real
flights stayed on their paths because of the aircraft's own lateral and vertical guidance (LOC, LPV) and the
controller's radar. Their corrections are smaller than half a grid step, so the open-loop reading shows none: in the
smoke artefact of stage A, the last heading word is class 0 in every sentence. Flown without these corrections, the
errors of the directions add up over 10–25 km. With an exact model of the words, 47 % of the vectored flights end more
than 106.7 m from the centreline, and the final descents end between 45 m low and 53 m high (p10, p90; §9.6). A finer
grid does not remove this: the real corrections are smaller than any grid step. The closed-loop reading puts the
missing corrections into words. Each correction is a reaction to an offset of the flown states, and a model trained on
these sentences reads the flown states (§6). Thus it learns "an offset gives a correction word" from states
that its own closed loop also gives.

**The flight.**

1. The executor starts at the first predicted step, from the observed state of that row, as a speaker's closed loop
   does: both through the start of a closed loop (§6, item 5; D67). The start state reads no sample after that row: the
   position and the height of the row, and the airspeed, the track and the path angle by the start rule of the
   executor spec, from the rows at or before it (D77); the bank is 0 (D84). The rows before it stay observed.
2. The aircraft flies with its own dynamics or with a stand-in's, by the rule of the replay (`autopilot/replay.py`
   `group_of`). A flight that the replay does not fly (no identified type, no aircraft dynamics, or no published
   approach speed) gives no training sentence. Nor does a sentence with fewer than two rows from the first predicted
   step, or in a go-around at it. The artefact counts these flights by reason.
3. At each row, the labeller finds the matched point, then decides the words of the row: the observed words that the
   matched point has reached ("When the observed words are said", below) and the correction words. The executor hears
   them together.
4. The closed-loop reading runs at each row interval Δ of the ablation. It takes the observed words with their 2 s
   times (the open-loop reading at 2 s, not the Δ grid of §4.8). The words, the corrections and the flown states are on
   the Δ rows and belong to that Δ.
5. The observed path includes a go-around and the next approach (D26). The comparison continues while G is true.
6. The flight ends where the judge ends it (§5.8; D79): an approach crossing of R, the ground, a crossing of another
   candidate, the dynamics, or the time limit of the replay. The closed-loop sentence has one row for each Δ row of the flight: its length is the flown time, not the
   observed time.

**The comparison.** The matched point is the point of the observed path nearest to the flown position. The search
starts at the matched point of the row before and goes forward, so that a path that crosses itself does not jump. At
the matched point:

- the lateral error e_y is the signed distance of the flown position from the observed path, the line through the
  observed rows near the matched point (its segments, not their extensions), positive to the right of the path (D83);
- the vertical error e_h is the flown height minus the observed height.

The observed height is the one that the labeller reads (smoothed, §4.1); the positions are as observed.

The reference is the matched point, not the observed point at the same time. Thus a difference along the path (a
difference of time) is not corrected, and the speed words stay the observed ones.

**Past the end of the observed path (D44).** The observed slice ends before the threshold, and the flown aircraft flies
on to it: approximately 150 m (median), at most approximately 750 m (§9.9). There the path continues along the line of
its last segment, and the lateral error is measured against that line for the readouts; no vertical error is measured
there. The labeller says no correction there, lateral or vertical: a correction in force ends at the end of the path,
and the observed word of its column is said again (in the readouts this word counts as a correction word; a speaker and
the executor read every word alike). The aircraft flies the last observed words to the threshold. These rows count as
rows without correction (D34). Why: there is no observation there; the slope of one 2 s segment is not a measurement
(the ADS-B altitude has steps of 7.6 m, so one segment can be approximately ±3° wrong, ±20 m over 400 m); and the DA
point, 0.9–2.1 km before the threshold, always lies before the end of the observed path, so the DA check does not
change.

**When the observed words are said (D42, D45, D46).** The observed time of the matched point is the time at which the
observed aircraft was at that point. At each Δ row, the labeller says every observed word whose 2 s time is less
than Δ/2 after the observed time of the matched point, and that it did not say before (D45; a word exactly Δ/2 after
waits for the next row, as on the Δ grid). Thus a word is said at the Δ row nearest to the place where the observed
aircraft heard it: a word alone in its row has a mean lateness of zero. When one Δ row says more than one observed word
of a column, it says the last one, so the words said are early on average (A21, train: +0.04 / −0.30 / −1.48 s at
Δ = 2 / 4 / 8 s; D45). A heading word is said in the frame where the executor
hears it (D46, §3.3). When the flown aircraft is behind the observed aircraft, the observed words wait. The first
predicted step says every column (rule 1): the observed words in force before Δ/2 after its observed time.

Why:

- A word gives a target, not a time. The speed steps (D43) give the rate of a change of speed, but the steps, the
  dynamics and the 5 m/s grid still leave a difference along the path: the flown aircraft moves ahead of the observed
  aircraft, or falls behind it (closed loop, select, Δ = 2 s, vectored: p50 221 m, p90 348 m; §9.9). The comparison
  does not correct this difference.
- A turn word must come where the observed aircraft turned. A word said at the observed time comes where the flown
  aircraft is at that time: an aircraft d ahead along the path before a turn of 90° is d to the side after the turn.
  One correction class closes 6–9 m/s, too slowly for such an offset (§9.7).
- The controller gives a turn at a place, not at a time. A model learns from the flown states when to say a word, so
  the words must agree with the flown states.
- A go-around word said at the observed time can come after the flown aircraft has crossed the threshold. Said at the
  place, it comes where the observed aircraft went around.

Thus the flown time to the threshold is not the observed time. The flown states keep the observed path, not the
observed time (D42).

**Lateral correction.**

1. When |e_y| > Y (the lateral tolerance), and the row says no new observed heading word, the labeller says the
   heading class one step (5°) from the observed word in force, toward the observed path.
2. When |e_y| < Y / 2, or e_y changes its sign, the labeller says the observed word in force again — except an
   overshoot (D53): when e_y changes its sign and |e_y| > Y, it says the opposite correction in the same row (item 1
   toward the other side).
3. A new observed heading word ends a correction. The labeller says the observed word, and the comparison continues.

**Vertical correction.** The vertical tolerance in force is H_final while "no level-off" is in force (the final
descent) and H elsewhere (D66). In the rules below, H is the tolerance in force.

1. Only while a descent class is in force (toward a level or with "no level-off"). During a level hold the level word
   is the target; its rounding to the grid (§3.4) is not corrected. A level hold is the executor's: the level in force
   is captured (its level-off has started, §5.5). A level reached by a descent says no angle word, so the descent class
   stays in force there, and the capture, not the angle word, tells the hold. In a level hold no correction starts, and
   one in force ends. A climb has one class, so a climb gets no correction.
2. When e_h > H (too high), the labeller says the next steeper descent class. When e_h < −H (too low), it says the next
   shallower descent class. From descent 4 there is no steeper class, and from descent 1 no shallower one: then there is
   no correction.
3. When |e_h| < H / 2, or e_h changes its sign, the labeller says the observed angle class in force again — except an
   overshoot (D53): when e_h changes its sign and |e_h| > H, it says the opposite correction in the same row (item 2,
   where that class exists). A new observed altitude word or angle word ends a correction.

**The words.** A correction word is an ordinary word of its column (§3.3, §3.5). The grammar (§3.7) checks each row at
the flown height; a row that it refuses refuses the flight (counted by reason). The envelopes of the closed-loop
sentence are checked on the flown states. The envelopes of the observed words on the observed track stay as the readout
of the open-loop reading. The capture row, the runway words and the go-around rows come from the open-loop reading: the
decisions come from the observed track; the closed-loop reading only adds corrections. The executor reads only words
(D2, D3).

**Tolerances (D32, D66).** Y = 30 m, H = 15 m and H_final, constants of the labeller in the spec. With Y = 30 m the
flown path ends within approximately 30 m of the observed path, which ends 2 m from the centreline (median), far inside
the runway limit of 106.7 m. H = 15 m is larger than the fit residual of the altitude pieces (10 m, §4.4), so that the
noise of the fit starts no correction. In the final descent, H = 15 m is too wide for the DA check (D38: 22 m at the DA
point, with the real tracks up to +8 m above the glidepath there, p90), so H_final is smaller there: the user chose
H_final = 10 m from the measurement of A24, which also counts the correction words that each value adds (D66).

**Where the code goes.** The labeller package (`instructions/`) does not import the executor; only the runners and
`autopilot/` read `instructions/` (`tests/test_architecture.py`). The closed-loop reading is a module in `autopilot/`.
It reads the open-loop artefact and the executor spec.

**Artefact.** For each split and each Δ: the closed-loop sentences (each correction word marked as a correction); the
flown states on the data's 2 s rows, with the Δ rows marked (position in the airport frame, MSL height, track, ground
speed, vertical rate; D51; on the observed rows, the track, ground speed and vertical rate by the start rule, D77); the
fields that use later rows or the observed path, apart from the rows (D82); the count of correction words for each
column; the flights without a training sentence, by reason.

## 5 Executor and judge

### 5.1 Principle

The executor flies the words with the point-mass dynamics of the ts control path. It has no state that the words do not
give. It has no law that makes a track from the runway or from a procedure (D2, D3). The user's rule of 2026-09-24: the
executor uses only the vocabulary; a parameter mined from data shows that it does not generalize.

### 5.2 What the executor reads

| Input | Use |
|---|---|
| The vocabulary spec | Grids, nominal angles, tolerances, lead, turn-rate limit, bank limit, speed change rate |
| The words of each row | The targets |
| The course of each candidate runway (`candidates.json`) | Conversion of heading words (§5.4) |
| The airport elevation E (`candidates.json`) | Conversion of level words: a level T is flown at T + E MSL (§5.5, D58) |
| The distance to the threshold of R | The deceleration to the approach speed (§5.6) |
| The aircraft: aerodynamic data, installed thrust, published maximum landing mass, published approach speed | Dynamics and "unspecified" speed |

The executor laws do not read the TCH, the glidepath angle or the DA. Only the judge reads them (§5.8), from
`candidates.json` (D61). The executor holds nothing of the landed runway or of a procedure: its frame has its origin at
the airport reference (D81). A behaviour check makes sure that the executor's laws read no vertical path: the reference
tracks flown with every candidate's vertical path changed and the origin moved give the same states (D81, §7.2).

### 5.3 One cycle

The laws of §5.4–§5.6 change the words in force into three wanted rates: the track rate χ̇*, the path-angle rate γ̇*
and the airspeed rate V̇*. The inner loop solves the point-mass equations of the ts control path
(`aerodynamic_model.torch_dynamics`) for the three controls that give these rates:

```
A = −χ̇*·V·cos γ / g          (= n·sin φ; compass and mathematical turns have opposite signs)
B =  γ̇*·V / g + cos γ        (= n·cos φ)
φ = atan2(A, B),   n = B / cos φ
T = m·(V̇* + g·sin γ) + D(V, n, h)
```

D is the drag of the dynamics' own polar at the commanded load factor and the ISA density at the height. The limits act
in this order, each on the value that it bounds:

1. Bank: not more than the bank limit (32°, spec), and changed at not more than the roll rate p = 5°/s, from the first
   cycle of a flight; the start bank is 0 (D84). Then n = B / cos φ keeps γ̇*: a limited bank costs turn rate, not the
   path.
2. Load factor: in [0.5, 2.0].
3. Stall floor: the speed law asks for no airspeed below 1.10·V_stall(n) at that load factor (§5.6).
4. Thrust: in [−0.2, 1.0]·T_max. The drag polar is clean: it has no flaps, no landing gear and no speedbrakes. The
   negative part of the range stands for these devices. A limited thrust costs airspeed rate.

One cycle is 1 s: the controls are held (zero-order hold), and RK4 integrates the point-mass equations. Layer 1 of the
judge records each limit that bound (§5.8).

### 5.4 Lateral law

- **Heading words.** When the executor hears a heading word, it sets the absolute target θ = course(R) + 5k° and keeps
  it until the next heading word. The word says the track L = 4 s later, so the executor turns to arrive at θ at that
  time. The track rate is the smallest of three rates (`lateral.word_rate`): the error e divided by the time left until
  L runs out (not less than two cycles; after that, the executor holds θ); the stopping rate √(2·g·p·|e| / V), the
  fastest rate whose bank the executor can still take out before e is gone (it cannot know which word ends a turn); and
  the turn-rate limit 4.7°/s. The first word turns in the shorter direction. A later word measures e from the target
  before it, so a series of words never reverses a turn while the aircraft lags behind it.
- **Go-around (D27).** "Go-around" does not change the heading target. The executor keeps flying the heading word in
  force. The model turns the aircraft with heading words, as at every other row.
- Nothing else. There is no capture state, no turn onto the final, no centreline tracking (§5.7).

### 5.5 Vertical law

The inner path-angle loop: γ̇* = sat((γ_ref − γ) / τ_γ, ±γ̇_max), τ_γ = 2 s. γ̇_max is twice the smallest rate that
keeps an entry into the steepest descent class inside its tube: 2 × V·γ_lo² / (2ε), with γ_lo the lower edge of the
steepest class and ε the narrowest level band of the grid. A level T is the height T + E MSL (D58); h is the height of
the aircraft above E. The modes choose γ_ref (climbing positive):

| Words in force | Mode | γ_ref |
|---|---|---|
| Level T + level | Hold | −(h − T) / (V·τ_h), τ_h = 4·τ_γ = 8 s; within the nominal angle of the steepest descent class and the climb angle (γ_GA while G is true) |
| Level T + descent class | Descend to T | −nominal angle of the class; level-off starts at V·γ²/(2·γ̇_max) above T |
| Level T + climb, G false | Climb to T | +nominal angle of the climb class (D15, D28); level-off as above |
| Level T + climb, G true | Go-around climb to T | +γ_GA, the go-around angle (below); level-off as above |
| "No level-off" + descent class | Descend | −nominal angle of the class; no level-off |

A target that the executor has captured (its level-off started) stays captured until a new altitude word or angle word,
so the mode does not change back and forth at the level.

**The go-around angle (D28).** γ_GA = min(3°, max(1.885°, γ_T)). γ_T is the steady climb angle at the thrust limit and
the present airspeed: sin γ_T = (T_max − D) / (m·g), with the drag D of the present state at load factor 1 (the
point-mass model of §5.3). The executor computes γ_T at each cycle from the aircraft data that it already reads (§5.2).
1.885° is the minimum gradient of a missed approach, 200 ft per NM (AIM 5-4-21 b); 3° is the upper limit. When γ_T is
less than 1.885°, the reference is 1.885° and the inner loop gives what the thrust permits;
layer 1 of the judge records the limit that bound (§5.8). No word of the vocabulary says an angle of a climb: the
vocabulary has one climb word (§3.5).

**No climb without a target.** Rules 5 and 6 (§3.7) make sure that "no level-off" is never in force while G is true.
Thus the executor has no mode "climb with no target", and "go-around" alone starts no climb.

### 5.6 Speed law (D43)

V_ref = V_g / cos γ (ground speed to airspeed without wind), not below 1.10·V_stall(n). For a speed word, the speed
changes at a_max, the largest acceleration of the speed envelope (§3.6, a value of the spec), and goes exponentially
into the last 5 m/s. A speed word is a step of 5 m/s (D43), so the executor makes each step in a few seconds; the rate
of a longer change comes from the words. Where the thrust limits cannot give a_max, the thrust limit binds (§5.3), and
the replay counts it. "Unspecified" is the published approach speed of the type (an indicated airspeed at the maximum
landing mass), scaled by √(m / maximum landing mass) to the mass of the aircraft and flown as the true airspeed at the
present density. A flight on a stand-in's dynamics has the stand-in's mass, so it takes the published speed of its own
type unscaled. The deceleration to it is the larger of a_U = 0.25 m/s² (a speed step of 5 m/s divided by the minimum
hold of 20 s) and the rate that reaches it over the straight-line distance to the threshold of R, not more than a_max.
While G is true, "unspecified" means the pilot's own speed in a missed approach: the executor holds the airspeed that
the aircraft has at the go-around row (D27; Claude's reading, not checked in the regulation text). A speed word of the
model ends it at any row.

### 5.7 What the executor does not do (D9)

The executor has no law of its own for the approach. It has:

- no runway lock (D12);
- no capture state, no turn onto the final, no heading bend and no own intercept (D2);
- no centreline tracking (D3);
- no special law for "no level-off": no aim at the threshold crossing point and no tube limit (D3);
- no glidepath floor and no level flight below the glidepath (D3, D9).

The words do these things. Only the judge and the masks read the procedure (principles 2 and 3).

### 5.8 Judge

**Layer 1, each cycle.** The wanted rates, the rates given, and the limit that bound.

**Layer 2, each word.** The labeller's checks on the flown track, from the row where the executor heard the
word: heading words to the end of the flight (§3.3); altitude and angle words in their tubes; speed words in their
spans.

**Layer 3, each flight.** A crossing of R is an approach crossing when G is false and the aircraft crosses the threshold
plane of R lined up: track within 30° of the course, lateral offset inside the landing screen (≤ 1,000 m and ≤ half the
spacing to a parallel). The runway limit is the FAS half-width at the threshold (106.7 m), and not more than half the
spacing to a parallel. The judge gives the first outcome that occurs. At one row it uses the order of the table.

| Outcome | Condition |
|---|---|
| `dynamics_failure` | A state that is not finite, no airspeed, or the stall cut-off of the dynamics |
| `ground_contact` | Below the threshold elevation of R while before the threshold |
| `crossed_too_high` | An approach crossing higher than 100 m above the threshold |
| `crossed_off_runway` | An approach crossing at ≤ 100 m, outside the runway limit |
| `unstable_at_minimums` | An approach crossing at ≤ 100 m, inside the runway limit, after a failed DA check (or with no DA point: the aircraft crossed above the DA) |
| `landed` | An approach crossing at ≤ 100 m, inside the runway limit, after a DA check that passed |
| `crossed_other_runway` | G false: the threshold plane of another candidate crossed lined up, inside that runway's own limit, at any height |
| `timeout` | None of these within the time limit (the remaining observed time × 1.5, plus 900 s for each go-around) |

A crossing that is not lined up (for example, abeam the threshold on a downwind) is not an event. While G is true, no
crossing is an event, of R or of another candidate (D33): the flight continues. The executor ends the flight at the
first of these outcomes, with the same tests (D79): nothing is flown or said after the end of a flight.

**The DA check (D3, D38).** At the DA point the judge checks that the aircraft is stable:

- **Vertical:** the height within ±22 m of the published glidepath of R (straight-line reference, §9.3; the TCH and the
  glidepath angle of R from `candidates.json`, D61). The value is
  the vertical bound of the evaluation module (`evaluation/thresholds.py` `RNAV_TERMINAL_VERTICAL_BOUND_M`; ICAO Doc
  9613, RNP APCH Baro-VNAV final approach segment). The judge imports it; it is not a parameter.
- **Lateral:** the lateral offset inside the FAS cone at the distance of the DA point (FAA Order 8260.58D Formula 3-1-1,
  `flight_scenarios/fas_geometry.py`), the same geometry as the runway limit of the crossing. The lateral bound of the
  evaluation module (half the runway width, 15.24–22.86 m) is a landing-geometry criterion at the threshold (evaluation
  EV1). At the DA point, 0.9–2.1 km before the threshold, it would be narrower than the lateral tolerance of the
  closed-loop reading (30 m, D32): a sentence that follows a real flight could then fail the check.

The regulation basis: the missed approach starts at the DA (AIM 5-4-21 b: "Obstacle
protection for missed approach is predicated on the missed approach being initiated at the decision altitude"). If
the model says "go-around" before the DA point, the flight continues (§3.2).

**Quality** belongs to the evaluation module (lateral, vertical and speed gates). The judge does not repeat it.

---

## 6 Public interface

The other documents and their code use this document only through these items and the decisions (the D numbers).
Everything else (the pieces of the labeller, the open-loop sentences, the laws inside the executor) can change without a
change of the other documents.

| # | Item | What it gives | Code | Identity |
|---|---|---|---|---|
| 1 | The vocabulary spec | The five columns in their order and their values (§3.1); the grids, the classes and their nominal angles, the ε of each level, the tolerances Y, H and H_final, the lead L, the turn, bank and speed limits, the reading name. `Words` converts a heading class to a track with the course of R, a level to a height with the airport elevation E, a speed value to m/s | `instructions/spec.py` `VocabularySpec`, `instructions/words.py` `Words` | The spec sha |
| 2 | The grammar | Rules 1–6 and the runway/G table (§3.2, §3.7) as one function: it checks a row (`apply`); for a speaker's mask, it gives the permitted words of a column after the earlier columns of the row: a word is permitted when some words of the later columns, each among the words that the caller permits, make the row pass `apply` (D62) | `instructions/grammar.py` `apply` | The labeller conformance |
| 3 | The sentence artefact | For each split of the day split (the test days sealed, contract C32) and each Δ of the ablation: the closed-loop sentences (the words of each row, each correction word marked) and the flown states on the data's 2 s rows with the Δ rows marked (position in the airport frame, MSL height, track, ground speed, vertical rate; observed before the first predicted step, flown from it; D51; on the observed rows, the track, ground speed and vertical rate by the start rule, D77, the track in [0°, 360°)); for each sentence, apart from its rows (D82), the fields that use later rows or the observed path: its flight, its runway and its index, its landing time, its capture row, its go-around rows, its stratum (§4.3; D70), the outcome of its closed-loop sentence (§5.8; D74), `timed_out`, the matched and observed rows, the errors e_y and e_h and the rows without a correction; the flights without a sentence, by reason. These fields are for readouts, strata and selection, never an input. The check that closed-loop sentences may be read: refused unless the code on disk reads the closed-loop reference as it was read, checked at the call (D69, D73) | `instructions/artefact.py`: the formats, `STATE_COLUMNS` (the names of the state columns) and the function that reads a closed-loop file into its sentences (for each: its flight, its first row, its words and correction marks, all its states on the 2 s rows from row 0 with the Δ rows marked) (D61); `instructions/readout.py` `STRATA` (the names of the strata, D70); the check: `autopilot/closed_loop.py` `require_conforming_closed_loop` (D69; it takes the artefact and the executor spec's directory, D73) | The format names; the labeller and the closed-loop conformance checks (§7.2, D73) |
| 4 | The candidates and their geometry | For each airport: E, and its candidates: its runway ends with a published vertical path, decided by no flight (D78); for each candidate: the threshold, the course, the threshold elevation, the length, and its vertical path (D61): the TCH, the glidepath angle and the DA above the threshold (the LPV line's; where a runway publishes no LPV line, KRDU 32 and KSMF 35R, the LNAV/VNAV line's: the reading of the code). The functions: the position, height and direction relative to a candidate (distance before its threshold along its course, offset right of its final, height above its threshold, direction minus its course); the height of its published glidepath at a distance before its threshold with the straight-line reference (§9.3), one function of the airport, the candidate and the distance, the radius of curvature included, which the judge and the prior both call; the lateral limit of a landing passage | `candidates.json` (a new format name, D61); `instructions/airport.py` `VerticalPath`, `relative_to_runway`, the glidepath height, `landing_cross_limit_m` | Part of the artefact (item 3) |
| 5 | The executor | Flies the words of one row at each Δ row, in 1 s cycles, from a given state (§5): one aircraft, a batch, or a batch in which each aircraft starts at its own cycle; it gives the state at each cycle and when the aircraft is done: where the judge ends the flight (D79). It reads only the words, the aircraft and the runway geometry (§5.2); it holds nothing of the landed runway (D81). The start of a closed loop (D67): for closed-loop sentences of the artefact (item 3) at a Δ, with the directory of the executor spec (opened and checked by the start, D71) and the most go-arounds of a flight, the executor at each flight's first predicted step (the flight rebuilt and compared with the stored signals, its start state from the rows at or before that step by the spec's start rule (D77), its aircraft and approach speed, its time limit and the time its go-arounds may add); then, for the words of one row of each flight, checked by the grammar before anything changes (D80), the states of the 2 s rows flown (`STATE_COLUMNS`) and the flights done | `autopilot/executor.py` `Executor`, `autopilot/single.py`; the start: `autopilot/start.py` | The executor spec sha and its conformance check (§7.2, D73) |
| 6 | The judge | The outcome of a flight and its order (§5.8), the DA check, the time limit (the remaining observed time × 1.5, plus 900 s for each go-around), the limits that bound in each cycle. The outcome of a flight flown from the start of a closed loop needs no observed words (D67) | `autopilot/judge.py` (the outcome: `outcome_of`) | With item 5 |
| 7 | The row grid | Rows on UTC multiples of Δ, Δ = 2, 4 or 8 s (D25); the chosen Δ is 4 s (D11); the first predicted step 16 s after row 0 | `instructions/labeller/interval.py` | — |

The package rules in the code: `instructions/` imports neither `autopilot/` nor a model package; `autopilot/` imports
`instructions/` and no model package; a runner joins a model to the executor (`tests/test_architecture.py`).

---

## 7 Gates and identities

### 7.1 Gates

Two gates belong to this document: the labeller (completeness, envelope containment with the envelope width) and the
replay (the executor flies the closed-loop sentences; §4.8, reading 4). The user sets their criteria (D7).

### 7.2 Identities

The rule is D21 (outline §3): an identity binds the format and the data rules; code is identified by what it does on
fixed inputs, data by their flights.

| # | What | Its identity |
|---|---|---|
| 1 | The vocabulary (grids, classes, tolerances, reading name) | The spec sha. It is the format |
| 2 | The labeller, open-loop and closed-loop reading | The labeller conformance: the artefact holds a reference sample (`conformance/`; train, seed 1337, 50 labelled and 10 refused flights for each airport, with their signals); `instruction_conformance` labels it again with the code on disk and requires the same words, the same strata (D70) and the same refusals, on the 2 s rows and on the Δ grid at Δ = 4 and 8 s (D49). The closed-loop reading (D32) has its own reference sample (train, seed 1337, 10 flights for each airport and the train flights with a labelled go-around, up to 5 for each airport, at each Δ of the artefact): with the artefact's executor spec, the same correction words and the same flown states on the 2 s rows (D51), within the tolerance of the executor conformance. Each check runs where it is needed, every time (D73): a runner that labels, reads the closed loop or replays sentences runs the checks of the code it uses before its work and refuses on a difference (the closed loop's: `require_conforming_closed_loop`, D69). No digest of code. The artefact records the commit that wrote it, as information |
| 3 | The executor | The sha of the spec's parameters and the conformance of its reference tracks (C33): the spec's labelled train flights, with the train flights that have a labelled go-around (up to 10 for each airport, so that the go-around climb and the held airspeed are flown), flown again in every way the executor flies (single-aircraft batch, multi-aircraft batch, single flight) within 1e-6 m, flown again before each use (D73); and the behaviour check of D81: the same tracks with every candidate's vertical path changed and the frame's origin moved. The open-loop sentences of these flights are said on their own rows (D57) |
| 4 | The flights of an artefact | The stored signals. A consumer that rebuilds a flight from the harvest compares it row by row with them (`autopilot/flights.py` `require_same_flight`). No byte hash of an arrival manifest |
| 5 | The day split and the sealed test days (C32) | The day split file. It is a data rule |

---

## 8 Values

| Item | Value | Source |
|---|---|---|
| Row | Δ = 4 s, the user's choice, on UTC multiples of 4 s; the artefact also keeps 2 and 8 s (the ablation); the data's rows are 2 s, on even UTC seconds | D11, D25 |
| First predicted step | 16 s after row 0 (8, 4, 2 rows at Δ = 2, 4, 8 s) | D25 |
| Executor cycle | 1 s | Fixed choice |
| Heading grid | 5°, relative to the course of R | Spec (grid); D8 (frame) |
| Heading lead L | 4 s | Measured (`instruction_vocabulary_design.zh.md` §10.1) |
| Heading tolerance | 4.5° | Half grid 2.5° + 2° |
| Turn-rate limit | 4.7°/s | Spec, measured (p99.9) |
| Bank limit | 32° | Spec, measured (p99.9) |
| Roll rate p | 5°/s | FAA Order 8260.3G Appendix E §4 ¶6.a ("roll-in rates of up to five degrees per second") |
| Altitude reference | The airport elevation E; a level T is T + E MSL | D58 |
| Level grid | 60 m to 1,260 m, 120 m to 2,700 m, 450 m to 5,400 m above E; 40 levels; fitted again on the level-offs above E, the user chooses | D22 (fit of the altitude-grid proposal, on MSL), D58 |
| Level envelope ε | half the larger gap to the neighbouring levels + 10 m: 40 m (0–1,200 m), 70 m (1,260–2,580 m), 235 m (2,700–5,400 m); "no level-off" 40 m | D22, D52 |
| Level detection | ≥ 20 s, rows within 25 m of the piece's median | Labeller constant (§4.4) |
| Descent classes | edges −0.5 / 2.0 / 2.75 / 3.75 / 10°; nominal 1.5 / 2.5 / 3.0 / 4.5° | Spec; k-means on the train days, weight length² (D54), rounded to 0.25° (D56) |
| Climb word | One class; a climb piece is 0.5–15° (labeller); executor angle while G is false: the nominal of D15, 1.5° | Spec; D15, D28, D56 |
| Go-around angle (executor, while G is true) | The steady climb angle at the thrust limit, within 1.885°–3° | AIM 5-4-21 b (minimum, 200 ft per NM); D28 (maximum) |
| Speed grid, range, tolerance | 5 m/s, 20–250 m/s, ±5 m/s | Spec |
| Speed change rate (executor), speed word | a_max (the transition acceleration limit below) | D43 |
| Speed change rate (executor), "unspecified" | a_U = 0.25 m/s², or the rate that reaches the approach speed at the threshold when larger; not more than a_max | 5 m/s ÷ 20 s (user, 2026-09-24) |
| Transition acceleration limit | 1.4 m/s² | Spec, measured (p99.9) |
| Capture corridor (labeller only) | 20 m + d·tan 0.45°, course ±2° | Spec, measured (p99) |
| Landing screen | lateral ≤ 1,000 m and ≤ half the parallel spacing; height ≤ 100 m | `final_approach.assign.LandingScreen` |
| Landed lateral limit | FAS half-width at the threshold, 106.7 m | FAA Order 8260.58D Formula 3-1-1 |
| Lined up | track within 30° of the course | 7110.65BB 5-9-2, TBL 5-9-1 |
| DA check | vertical ±22 m of the published glidepath; lateral inside the FAS cone at the DA distance | D38 (`evaluation/thresholds.py`; FAA Order 8260.58D Formula 3-1-1) |
| Time limit | remaining observed time × 1.5, plus 900 s for each go-around | Fixed choice; the 900 s: the user, 2026-10-02 |
| Closed-loop reading: tolerances | lateral Y = 30 m, vertical H = 15 m; vertical H_final = 10 m while "no level-off" is in force (the user's choice from A24); a correction ends below half the tolerance in force or at a change of sign; a change of sign beyond the tolerance says the opposite correction in the same row | D32, D53, D66 |
| Closed-loop reading: correction | heading: one class (5°) toward the observed path; angle: the next descent class | D32 |
| Closed-loop reading: observed words | at the first Δ row whose matched observed time is less than Δ/2 before the word's 2 s time (the Δ row nearest to its place; a tie: the later row); a heading word in the frame where it is heard | D42, D45, D46 |
| Closed-loop reading past the end of the observed path | no correction (one in force ends); lateral error for the readouts, no vertical error | D44 |
| Turn rate of a heading word | the smallest of: the error over the time left until L (not less than two cycles), the stopping rate √(2·g·p·\|e\| / V), the turn-rate limit | §5.4; D47 |
| Δ grid of the open-loop reading | each 2 s word on the nearest Δ row (a tie: the later row) | D45 |
| Word clock of a replay | `time`: a sentence is said on its own rows (the formal executor spec `v12_20261004` has `track` until A19) | D57 |

---

## 9 Evidence

All counts in §9.1–§9.3 come from two runners on the train split of artefact `v6_20261002` (44,375 sentences),
2026-10-03: R48 `instruction_word_frames` (§9.1, §9.2; output
`4dTrajectory/outputs/POOLED/analyses/word_frames_20261003/word_frames.json`) and R49 `instruction_final_approach`
(§9.3; output `4dTrajectory/outputs/POOLED/analyses/final_approach_20261003/final_approach.json`). Each output gives
the same numbers for each airport too. `docs/reference/runners.md` R48, R49 give the definitions.

The smoke builds of §9.6 and §9.7–§9.10 took the first 80 sorted flight keys of each airport and split (a key starts
with the callsign: approximately 400 train flights, 373 of them one airline), and each fitted its spec on its own train
flights (the A9 smoke: descent nominals 1.51 / 2.36 / 3.08 / 4.36°, the climb 4.24° from one climb piece). Their numbers
show how the mechanisms act; they are not numbers of the formal artefact (D55).

### 9.1 Frames of the heading and level words

Mean pairwise Jensen–Shannon divergence between the word distributions of the five airports (bits; 0 is the same
distribution, 1 is no class in common):

| Words | Absolute | Relative to R |
|---|---|---|
| All heading words | 0.108 | 0.052 |
| Heading in force at the clearance row of `instruction-v3` (the intercept heading) | 0.619 | 0.195 |
| Level words (relative = height above the threshold) | 0.426 | 0.484 |

Share of each airport's MSL level words on levels that the other four airports use for less than 0.2 % of theirs: KSJC
27.8 %, KRDU 6.8 %, KSTL 4.8 %, KSMF 3.7 %, KMSY 0.4 %.

### 9.2 Courses of KSTL against the 5° grid

KSTL courses 62.7°, 122.3°, 242.7°, 302.3° (all eight candidates): the nearest absolute grid values are 2.26–2.32° away.
The capture corridor permits 2°. The courses of the other four airports are 0.01–1.07° from the grid (KSJC the
largest).

### 9.3 Clearance, final descent and glidepath in the labelled data

- 47.1 % of all sentence rows are before the clearance row of `instruction-v3`. 25.3 % of the flights are cleared at
  row 0.
- "Descend to land" comes before the clearance row in 49.9 % of the flights (median 96 s before, p90 332 s). It comes
  before the capture row in 59.0 %. Its distance before the threshold along the course is −0.9 / 15.7 / 24.0 km
  (p10 / p50 / p90; a negative distance is past the threshold plane, on a downwind).
- After the capture row and more than 300 m before the threshold, the observed height minus the published glidepath of
  the landed runway, against two references (R49). "Flat" is TCH + d·tan(angle), the formula that the executor and the
  procedure masks use; it leaves out the earth's curvature. "Straight line" is the published path as a straight line in
  space; over the curved earth it is d²/(2R) higher (approximately 15 m at 14 km, 31 m at 20 km).

  | Measure | Flat | Straight line |
  |---|---|---|
  | Rows: p10 / p50 / p90 | −9 / +6 / +31 m | −20 / +2 / +21 m |
  | Rows within ±30 m / ±60 m | 83.6 % / 92.5 % | 86.2 % / 92.8 % |
  | Flights with ≥ 90 % of their rows within ±60 m | 78.0 % | 78.9 % |
  | At the capture row (median 13.9 km before the threshold): more than 60 m above / below | 11.2 % / 21.0 % | 5.7 % / 24.9 % |

  The flat formula of the executor is a few metres to 31 m low far from the threshold. This matters only for a design
  that uses it as a reference.
- Published glidepaths: 3.0° at all candidates, 3.5° at KRDU 32. TCH 13.7–19.5 m.

### 9.4 Go-around

- R40 v2: 105 real go-arounds on the training days; from the go-around to the next turn onto the final, median 421 s,
  p95 598 s ([readout](../readouts/2026-10-01_go_arounds.zh.md)).
- Words in force at the 70 real go-arounds inside the train sentences of `v5`, where each sentence is one
  approach to the landing (R40 v2 joined to the `v5` sentences; a one-off script, 2026-10-03): "descend to land" 0 of
  70; a level 70 of 70 (median 450 m MSL; the go-around heights above the threshold p10 / p50 / p90 90 / 236 / 481 m);
  "unspecified" 0 of 70; the capture row after the go-around row 70 of 70; a landing on another runway after the
  go-around 9 of 70 (D26).
- The row of the climb word in `v5`, relative to the lowest point of the go-around (69 of the 70 have a climb word):
  1–3 rows before 7; the same row 6; 1–4 rows after 45; 5–23 rows after 11. D26 puts "go-around" in the row of the
  climb word.
- The climb after a real go-around: median gradient 9.3 % (5.3°); 96 % are at least 200 ft per NM (R40 v2, readout §7).

### 9.5 The altitude grid

From the proposal [altitude_word_grid.zh.md](../../history/2026-10_two_tier_design/altitude_word_grid.zh.md)
(`v6_20261002`; fit on 11,937 train flights, read on 3,964 validation flights; continuous level-off heights). Rounding
error of the level words other than row 0, p50 / p95 / largest: the chosen grid (40 levels) 17 / 30 / 58 m; the same
shape with steps doubling and break points at 1,200 and 2,400 m (44 levels) 17 / 38 / 79 m; a uniform grid of 48 levels
over the range 35 / 55 / 57 m. The `instruction-v3` grid (30 m, 182 classes) has at most 15 m. These numbers come from a
one-off script of that proposal. The spec measurement gives them from a runner (`instruction_spec`, `level_rounding`):
on all train days with the labeller of stage A (code at `42f3ff62`, 25,654 level words), 16 / 34 / 223 m; the largest is
in the 450 m segment. These are MSL heights. Since D58 the measurement reads the heights above the airport elevation
(A20; code at `220e858e`, 25,727 level words): the exact fit of §3.4 gives the grid of D22 itself, 17 / 34 / 190 m;
among the 4,515 grids of at most three segments and 40 levels, the next best leaves 7.6 % more squared error (60 / 120 /
600 m with breaks at 1,200 and 3,000 m: 17 / 40 / 290 m). Fitted on the same level-offs in MSL, it is D22 too. The
five training airports lie at 1–188 m, and most level-offs fall in the 60 m segment either way.

- In `v6_20261002`, 99.7 % of the level words are under 3,000 m; its 30 m grid used 116 of its 182 classes.
- The geometric height of one assigned pressure level moves from day to day by about height × ΔT / 273 (ΔT: the day's
  temperature deviation). At KRDU this is about 15 m at 600–900 m and about 90 m at 2,000–3,000 m
  (`instruction_vocabulary_design.zh.md` §2.4).

### 9.6 Open-loop words on the smoke artefact

From the smoke build of stage A (`dev-two-tier-v4`, 80 flights for each airport and split; 1,189 sentences; the replay
at Δ = 2 s, 150 train flights on their own dynamics). One-off scripts, 2026-10-03; information, not a criterion (D7).

- Replay outcomes: landed 49, `crossed_off_runway` 36, `ground_contact` 23, `unstable_at_minimums` 16,
  `crossed_too_high` 15, `timeout` 6, `crossed_other_runway` 5. Lateral offset at the crossing, median / p90:
  straight-in 54 / 155 m, vectored 143 / 265 m.
- The last heading word is class 0 in 1,189 of 1,189 sentences. Where it takes effect (median 22.7 km before the
  threshold straight-in, 11.2 km vectored), the observed offset is 28 m and 30 m (median); the observed track after it
  stays within 2.1° of the course (median of the largest deviation, vectored); the observed offset at the last row is
  2 m (median).
- Where that word takes effect, the flown offset of the vectored flights is 157 m (median; p90 378 m) against 30 m
  observed; at the end, 155 m. The error is made before the final.
- An exact model of the heading words (each word's track exactly, reached 4 s after the word, at the observed ground
  speed, from the observed position at row 0), offset at the last row:

  | Heading grid | Straight-in: median / p90 | Vectored: median / p90 | Vectored: more than 106.7 m |
  |---|---|---|---|
  | Observed track | 2 / 5 m | 2 / 4 m | 0 % |
  | 5° | 34 / 82 m | 99 / 382 m | 47 % |
  | 2.5° | 31 / 63 m | 74 / 196 m | 31 % |
  | 1° | 40 / 72 m | 70 / 118 m | 15 % |

- The final descent ("no level-off" in force, median 20 km) flown at the nominal angles of its labelled classes, from
  the observed height where it starts: the height at the last row minus the observed height, p10 / p50 / p90:
  −45 / −4 / +53 m; 22 % more than 30 m high, 19 % more than 30 m low.

### 9.7 Closed-loop sentences with the observed words at the observed time

From the smoke build of A9 (`dev-two-tier-v4` at `751d0ec8`, `smoke_v4/data/`): closed-loop sentences at Δ = 2 s, each
observed word said at its observed time, replayed; the select split, the 395 flights on their own dynamics.
One-off scripts, 2026-10-04; information, not a criterion (D7).

- Landed: straight-in 222 of 229 (96.9 %), vectored 122 of 166 (73.5 %). The vectored failures: `timeout` 17,
  `crossed_off_runway` 13, `unstable_at_minimums` 6, `crossed_too_high` 5, `crossed_other_runway` 3.
- 41 of the 44 vectored failures leave the observed path laterally by more than 300 m (largest offset: median 1.55 km).
  At the row before, the flown aircraft is ahead of or behind the observed aircraft of the same time, along the
  observed track, by 1.3 km (median; p90 3.6 km; more than 500 m in 88 %).
- 76 of the 122 landed vectored flights also leave the path by more than 300 m (along the track before it: median
  0.6 km); they come back before the threshold. Heading correction words: 12.4 for each sentence (train, Δ = 2 s).
- The straight-in flights are also ahead or behind (median of the largest difference 0.7 km), but they do not turn:
  13 of the 222 landed straight-in flights leave the path by more than 300 m.
- KMSY `AAL2354_11_a9942e_20260826T153443Z` hears 70 m/s at 62 s. The observed aircraft slows at 0.36 m/s², the
  executor at 0.25 m/s². At the start of a turn of approximately 100° the flown aircraft is 1.8 km ahead; after the
  turn it is 2.5 km to the side (`timeout`).
- KSTL `AAL1010_30R_aae316_20260722T024358Z` hears 105 m/s at the first predicted step. The observed aircraft takes
  approximately 290 s from 146 m/s to 105 m/s, the executor approximately 160 s. The flown aircraft is 4 km behind;
  after the turns it is 11 km to the side (`timeout`).
- One correction class (5°) closes 6–9 m/s at a ground speed of 70–100 m/s: 2 km in 4–5 minutes. A new observed heading
  word ends a correction (§4.9).
- The 8 vertical failures of the DA check (select, 22–54 m above the glidepath at the DA point): in 3, the flown path is
  within 3 m of the observed path and the observed path is 19–23 m above the glidepath; in 3, the flown path is 9–16 m
  above the observed path and the observed path is 12–15 m above the glidepath; in 1, the flown path is 44 m above the
  observed path; in 1, the flown path is 850 m to the side of the observed path.

### 9.8 Speed words and the rate of a change of speed

From the smoke build of A9 (`dev-two-tier-v4` at `751d0ec8`, `smoke_v4/data/`), train and select. One-off scripts,
2026-10-04; information, not a criterion (D7).

- The observed rate of a deceleration (20 s windows that slow down; ground speed, 10 s mean), p25 / p50 / p75: level
  flight (|vertical rate| < 0.5 m/s, 844 windows) 0.12 / 0.24 / 0.40 m/s²; descent (vertical rate below −3 m/s, 3,540
  windows) 0.11 / 0.19 / 0.31 m/s².
- A one-dimensional estimate: 634 flights with at least 30 rows of 2 s before "unspecified"; from the first predicted
  step, the speed goes toward the word in force at a constant rate, the words at the observed time, no dynamics. The
  "steps" are the grid value nearest to the observed speed at each row. The largest difference along the path from the
  observed aircraft, for each flight:

  | Words | Rate | p50 | p90 | More than 1 km |
  |---|---|---|---|---|
  | One word for each run (its target) | 0.25 m/s² | 499 m | 2,235 m | 31 % |
  | One word for each run (its target) | 0.5 m/s² | 537 m | 2,359 m | 32 % |
  | One word for each run (its target) | 1.5 m/s² | 1,276 m | 4,527 m | 57 % |
  | Steps | 0.25 m/s² | 743 m | 2,251 m | 39 % |
  | Steps | 0.5 m/s² | 255 m | 563 m | 1 % |
  | Steps | 1.0 m/s² | 148 m | 297 m | 0 % |
  | Steps | 1.5 m/s² | 120 m | 243 m | 0 % |

- Speed words before "unspecified", for each flight (mean): 2.2 for one word for each run, 11.1 for the steps.
- With a thrust floor of 0 N, 40 % of the inverted teacher segments of the KSJC outer-train cohort needed less thrust
  than 0 N: the real aircraft slowed down more than the clean polar permits (`outputs/envelope.py`).

### 9.9 Closed-loop sentences with the words at the place and the speed words in steps

From the smoke build of A10 and A11 (`dev-two-tier-v4` at `0c0778c8`, `smoke_v4/data/a11/`, the spec of the A9 smoke;
words said at the first Δ row after their place, speed words in steps). Readouts of the stage A report
(`readouts/2026-10-04_stage_a_a10_a11_report.zh.md`) and one-off scripts, 2026-10-04; information, not a criterion (D7).

- Landed in the replay of the closed-loop sentences, select, Δ = 2 s: straight-in 225 of 231, vectored 157 of 166
  (94.6 %). Vectored flights that leave the observed path by more than 300 m: 4 (§9.7: 117).
- The 14 failures of select, Δ = 2 s (9 vectored, 5 straight-in), are DA checks that fail vertically: 13 between 22 m
  and 43 m above the glidepath, 1 23 m below it; 9 at KMSY.
- Heading correction words for each sentence, train, Δ = 2 s: straight-in 4.9, vectored 25.7.
- The largest difference along the path from the observed aircraft of the same time, before "unspecified", closed
  loop, select, Δ = 2 s, vectored: p50 221 m, p90 348 m; more than 1 km 1 %.
- Δ = 4 s, train: 35 of 40 vectored flights leave the observed path by more than 300 m (Δ = 2 s: 2 of 40); 39 of 40
  land.
- The flight past the end of the observed path, train and select: p50 149 m, p90 393 m, at most 754 m. Train, Δ = 2 s:
  82 of the 2,621 angle corrections and 6 heading corrections were said there.
- A synthetic flight at the observed speed, flown open loop: 75–180 m of lateral offset after each turn (a test of A11
  holds 93 m after the second turn).
- The courses of parallel runways differ by 0.002–0.01°.

### 9.10 The nearest row and the turn of the executor

From the smoke build of A12 and A13 (`dev-two-tier-v4` at `fd235e19` and `b8994f58`, `smoke_v4/data/a12/`, the spec of
the A9 smoke; the closed loop with a tie at the earlier row, before A14). Readouts of the stage A report
(`readouts/2026-10-04_stage_a_a12_a13_report.zh.md`); information, not a criterion (D7).

- Closed-loop sentences replayed, train, vectored: more than 300 m from the observed path at Δ = 4 s 0 of 40 (§9.9: 35
  of 40), at Δ = 8 s 2 of 40; all 40 land at each Δ. Select, Δ = 2 s, vectored: 159 of 166 land.
- The lateness of the observed heading words said in closed loop (the matched observed time of the row that says a word
  minus the word's 2 s time), train, mean: +0.03 s at Δ = 2 s, −0.3 s at 4 s, −1.6 s at 8 s; inside ±Δ/2. In a turn
  several words fall in one row and only the last, the earliest in time, is said: thus the words said are early on
  average at Δ = 8 s.
- The open-loop sentences at Δ = 4 s do not change: on a 4 s grid, the nearest row with a tie at the later row is the
  next row.
- The turn of the executor (A13, `experiments/executor_turns.py`), select, 837 turns measured in every way, the turn's
  own part toward the outside, mean: the exact words −36 m, the executor −18 m, the executor without the stopping-rate
  limit −21 m, the observed track integrated (the control) −2 m. Without the 292 turns whose next turn starts within
  30 s of the end: −34, −16, −18, +1 m. Larger at higher speed: at 100 m/s or more the words −39 m, the executor −18 m.
  The change of e_y across a turn: p50 58 m, p90 390 m (the executor); it holds the offsets of earlier turns.

---

## 10 Regulation sources

Checked on 2026-10-03 in the local PDF files (`docs/literature/runway_assignment/official/`, repository root; not in
git):

| Paragraph | Text used |
|---|---|
| Pilot/Controller Glossary, CLEARED (Type of) APPROACH | "ATC authorization for an aircraft to execute a specific instrument approach procedure to an airport" |
| AIM 5-5-4 a.3 | When radar vectored: "Maintains the last assigned altitude until established on a segment of a published route or IAP, at which time published altitudes apply" |
| AIM 5-5-4 b.1, b.2 | The controller "Issues an approach clearance based on known traffic"; "only after the aircraft is established ... or assigns an appropriate altitude ... until so established" |
| AIM 5-4-21 b | Missed approach at the DA; climb gradient of at least 200 feet per nautical mile (1.885°) |
| 7110.65BB 4-7-10 a | Approach information at the first contact: the approach to expect and the runway |
| 7110.65BB 5-9-1 a, c | Intercept the final at least 2 miles (3,704 m) outside the approach gate; for a precision approach "at an altitude not above the glideslope/glidepath" |
| 7110.65BB 5-9-2, TBL 5-9-1 | Intercept angle: 30°, or 20° less than 2 miles (3,704 m) from the approach gate |
| 7110.65BB 5-9-4 | Arrival instructions before the approach gate; example 2: "Turn right heading three four zero. Maintain two thousand until established on the localizer. Cleared I-L-S runway three six approach." |
| 7110.65BB 5-7-1 b.4, d | No speed assignment inside 5 NM (9,260 m) or the FAF; an approach clearance cancels assigned speeds (`instruction_vocabulary_design.zh.md` §9) |
| FAA Order 8260.58D Formula 3-1-1 | FAS course half-width ≥ 350 ft (106.7 m) at the threshold (cited in `flight_scenarios/fas_geometry.py`; not checked again on 2026-10-03) |

---

## 11 Key code index

The code of this document (stage A). Line numbers are at the end of A9 on `dev-two-tier-v4`; paths are relative to
`4dTrajectory/ts_transformer/`.

| What | Where |
|---|---|
| Columns; "go-around" in the runway column | `instructions/words.py:20` `COLUMNS`, `:24` `RUNWAY_GO_AROUND` |
| Heading classes relative to the course of R | `instructions/words.py:80` `heading_class`, `:84` `heading_track_deg` |
| The 40 levels, their bands | `instructions/words.py:47` `altitude_tolerances`, `:89` `altitude_index` |
| Grammar: rules 1–6 and the runway/G table written once; a row's check; a speaker's column mask (D62; at `df84cbf1`) | `instructions/grammar.py:147` `_rules`, `:191` `apply`, `:227` `column_words`, `:236` `column_mask` |
| Approaches; capture row per approach; heading words relative to the R of each row | `instructions/labeller/lateral.py:32` `Approach`, `:53` `capture_row`, `:71` `per_step_words`, `:105` `read_lateral` |
| Level test by the piece's own median; held height; the go-around climb and the final descent before it (D26); tubes | `instructions/labeller/vertical.py:65` `vertical_pieces`, `:86` `held_height` (above E since A20), `:122` `climb_after`, `:133` `read_vertical`, `:236` `tube_bounds`, `:267` `tube_checks` |
| Speed words per approach; "unspecified" from the capture row | `instructions/labeller/speed.py:90` `read_speed` |
| Go-around rule (R40's, on the rows) | `instructions/labeller/go_around.py:62` `low_passes`, `:109` `go_arounds` |
| Landing cut, go-arounds in the gate, approaches, runway words (D19, D26) | `instructions/labeller/read.py:59` `ApproachReading`, `:126` `admit`, `:168` `flight_go_arounds`, `:185` `runway_again_rows`, `:208` `read_flight` |
| Row interval (D11, D25); a heading word across a runway change | `instructions/labeller/interval.py:51` `first_interval_row`, `:75` `on_interval` |
| Labeller conformance (§7.2 #2) | `instructions/conformance.py:59` `labeller_code_sha256`, `:150` `check`, `:204` `require_conforming_labeller` |
| The words in force at each go-around row (readout) | `instructions/readout.py:30` `go_around_in_force` |
| Heading envelope | `instructions/envelope.py:31` `heading_words_inside` |
| Each candidate's vertical path; the published glidepath's height (D61; at `df84cbf1`) | `instructions/airport.py:34` `VerticalPath`, `:118` `vertical_path` (read once, refused without one), `:130` `airport_geometry`, `:220` `published_glidepath_height_m`, `:231` `glidepath_height_m`; `instructions/artefact.py:48` `CANDIDATES_SCHEMA`, `:122` `load_candidates` |
| Closed-loop sentences on disk; the public reader (D61; at `df84cbf1`) | `instructions/artefact.py:222` `CLOSED_LOOP_SCHEMA`, `:237` `STATE_COLUMNS`, `:241` `ClosedLoopSentence`, `:271` `write_closed_loop`, `:308` `load_closed_loop`, `:320` `closed_loop_sentences` |
| Heading law; conversion with the course of R ("go-around" changes no target, D27) | `autopilot/lateral.py:52` `word_rate`, `:127` `Lateral.word_error`, `:154` `Lateral.rate` |
| Vertical modes; the go-around angle (D28) | `autopilot/vertical.py:49` `GO_AROUND_MIN_RAD`, `:54` `go_around_angle_rad`, `:88` `Vertical.rate` |
| "Unspecified" under G holds the go-around row's airspeed | `autopilot/speed.py:71` `Speed.hear_go_around`, `:76` `Speed.rate` |
| The cycle; go-around time; approach crossing ends the flight | `autopilot/executor.py:60` `GO_AROUND_EXTRA_S`, `:160` `Executor.cycle` |
| R and G per row | `autopilot/sentence.py:89` `_filled` |
| Outcomes, their order; the DA check (D38) on each candidate's vertical path (D61); no crossing is an event under G (D33) (at `df84cbf1`) | `autopilot/judge.py:73` `OUTCOMES`, `:76` `EVENT_ORDER`, `:134` `decision_check` (D38: `evaluation/thresholds.py` `RNAV_TERMINAL_VERTICAL_BOUND_M`, repository root), `:157` `_outcome`, `:301` `judge` |
| A sentence on Δ; the readout | `autopilot/replay.py:77` `sentence_on_interval`, `:88` `instructions_of`, `:233` `batch_of`, `:398` `summary`; `experiments/executor_replay.py:110` `readout_table` |
| Closed-loop reading (D32): the matched point, the corrections, the rows without one (D34), the flight, its conformance, its replay (at `df84cbf1`) | `autopilot/closed_loop.py:126` `ObservedPath`, `:171` `uncorrected_m`, `:251` `Corrector`, `:390` `read`, `:501` `read_chunked`, `:529` `closed_loop_code_sha256`, `:605` `check`, `:677` `require_conforming_closed_loop`, `:692` `replay_batch`; `experiments/instruction_closed_loop.py:74` `main`; `experiments/executor_replay.py:236` `closed_loop_columns` |
| FAS cone; DA above the threshold | `flight_scenarios/fas_geometry.py:46` `fas_course_geometry`; `trajectory_data_process/harvest/airports.py:169` (repository root) |
| A15 (line numbers at `4a2f4fc5`): one test of "the same track" (D46, D48) | `instructions/words.py:173` `same_track`; the 2 s sentence `instructions/labeller/sentence.py:23` `assemble` |
| A15: a sentence on its UTC Δ grid; the Δ rows of the 2 s rows (D49, D51) | `instructions/labeller/interval.py:95` `on_utc_grid`, `:68` `on_interval_rows` |
| A15: the labeller reference on the Δ grids (D49) | `instructions/conformance.py:47` `INTERVALS_S`, `:81` `sentences_of` |
| A15: closed-loop states on the 2 s rows (D51) | `instructions/artefact.py:271` `write_closed_loop` (lengths and marks; at `df84cbf1`), `autopilot/closed_loop.py` `read` |
| A15: the rule of D50 and D34's third reading | `autopilot/closed_loop.py:215` `observed_tracks`, `:227` `outside_rows`; `experiments/instruction_closed_loop.py` `summarise` (`outside_the_tolerance`) |
| A16 (`3e70b4f3`): an overshoot corrected in its row (D53) | `autopilot/closed_loop.py:330` `Corrector.row` (lateral; the angle just below) |

---

## 12 Implementation plan

The rules of the implementation are in the outline (§5 there).

### 12.1 Stage A: vocabulary, labeller, identities, executor, judge, replay

A0–A17 are done (§0.3 gives the commits; the reports are in `readouts/`). The rules are in the design sections above;
the table gives what each milestone built. The module and the test that carry each decision of A0–A14 are in the table
of Claude's check (`readouts/2026-10-04_stage_a_check.zh.md` §2), those of A15–A17 in their reports.

| Milestone | What it built | Decisions |
|---|---|---|
| A0 | The archive of the modules of the old vocabulary that stage A does not rewrite (`archive/two_tier_v3_2026_10/`, with a `README.md`: what moved, why, which stage brings each part back; the backend tests that use the two-tier code fail until A23) | D20 |
| A1 | The vocabulary: five columns, the runway/G table, heading classes relative to the course of R, the altitude grid, grammar rules 1–5 as one function for the labeller and the speaker | D1, D8, D10, D12, D14, D22 |
| A2 | The labeller: heading words from row 0 to the end, the go-around reading, the level test without the grid, "unspecified" from the capture row, the Δ grid; candidates without TCH, glidepath angle or LPV DA refused | D4, D11, D18, D19, D25 |
| A3 | The artefact and the identities: the labeller conformance (§7.2 #2), no byte checks of the arrival manifests, the sentences schema, the spec measurement with its rounding candidates and the level rounding errors | D15, D21 |
| A4 | The executor without the laws of §5.7; heading words converted with the course of R when heard; the go-around; the words heard at each Δ row; the same in `autopilot/single.py` | D2, D3, D9, D10, D27 |
| A5 | The judge: the outcomes of §5.8 in their order, no event while G is true, the time limit + 900 s for each go-around, the DA check | D3, D33 |
| A6 | The executor spec (`ts-executor-spec-v7`) and its conformance; `executor_replay --row-interval-s` | D21 |
| A7 | The close of the first part: the full suite, the report `readouts/2026-10-03_stage_a_report.zh.md` | — |
| A8 | Each approach read separately; rule 6; "go-around" changes no target; the go-around angle | D26–D28 |
| A9 | The closed-loop reading (`autopilot/closed_loop.py`, runner `instruction_closed_loop`) and its conformance; the replay of closed-loop sentences; the DA check with the bounds of D38; the readings of D34 | D32–D34, D38 |
| A10 | The observed words said at the place; the flight to the end of the executor or the time limit | D42 |
| A11 | Speed words in steps; the executor at a_max | D43 |
| A12 | Each word on the nearest Δ row; a heading word in the frame where it is heard; no correction past the end of the observed path | D44–D46 |
| A13 | The turn measurement (`experiments/executor_turns.py`, R51); the user kept the turn law | D47 |
| A14 | A tie at the later row in the closed loop too | D45 |
| A15 | A heading word of the class in force after a change of R; the flown states on the 2 s rows; the Δ grid in the labeller conformance; the architecture test that the executor's laws do not import the runway data | D48, D49, D51 |
| A16 | An overshoot corrected in its row | D53 |
| A17 | The smoke sample random for each airport and split | D55 |

**A18. The chosen spec (D56).** Done: `experiments/instruction_spec.py` `--candidate NAME` (required when the runner
measures) takes the descent nominals and edges and the climb nominal of that row of `rounding_candidates` (`fitted`,
`0.5`, `0.25`, `0.1`); `measurements.json` records the chosen row beside the fitted values. Its formal build was stopped
for D58 (the user, 2026-10-04): the directories `4dTrajectory/outputs/POOLED/instruction_language/v7_20261004/` and
`4dTrajectory/outputs/POOLED/executor/v12_20261004/` are superseded. They are deleted with the user's go, never kept
beside the formal artefact of A21.

**A19. One word clock (D57).** Done with A20 (`220e858e`). Together with A20: A20 changes the executor, and one rebuild
serves both.

- Remove the clocks `distance` and `track`: `autopilot/sentence.py` (`DistanceClock`, `TrackClock`, `CLOCKS`),
  `autopilot/single.py` (`word_clock`), `autopilot/replay.py` (`word_clock`), `ExecutorParams.word_clock`,
  `executor_spec --word-clock`. A replay and the reference flights of the executor conformance say a sentence on its own
  rows. A new executor spec schema name (outline §5 rule 5).
- Tests: the tests of the two clocks go with them; an open-loop replay says each word on its own row; the batch and the
  single-flight executors give the same flown states on the conformance flights.
- The readout "the observed words alone" is not built here. When a readout needs it, it is the closed-loop reading with
  no corrections (a runner setting).

**A20. The altitude words above the airport (D58).** Done with A19 (`220e858e`); the spec measurement on all train
days done (§9.5); the user chose the fitted grid (D59). After A18; together with A19.

- Vocabulary (`instructions/words.py`, `instructions/spec.py`): a level is a height above the airport elevation E;
  `Words` maps a height above E to its level and a level to its height, with E as an argument (as the course for the
  heading words). E is `reference.elevation_m` of the airport in `candidates.json`. A new reading name and new names for
  every changed format (outline §5 rule 5).
- Labeller (`instructions/labeller/`): the altitude words from the smoothed height above E; the tubes of §3.4 and
  grammar rules 3 and 6 on the height above E. The path angles, the level detection and the pieces do not depend on a
  constant offset, so the descent and climb classes of D56 stay.
- Executor (`autopilot/vertical.py`, `autopilot/executor.py`, `autopilot/single.py`) and judge (layer 2): a level T is
  flown and judged at T + E MSL. The DA check, the crossings and the ground contact keep their MSL and threshold
  references. The closed-loop reading keeps e_h (a difference) and stores MSL heights (D51).
- Spec measurement (`instructions/measure.py`, `experiments/instruction_spec.py`): the grid fit of §3.4 as a step of the
  runner: on the level-offs above E of the train days (row 0 apart), at most three uniform segments, break points on a
  15 m grid, steps from 15, 30, 45, 60, 75, 90, 120, 150, 180, 240, 300, 450 and 600 m, 40 levels, the smallest sum of
  squared rounding errors (exact dynamic programming). It writes this fit and the grid of D22, each with its rounding
  error of the level words (p50 / p95 / largest). `--grid NAME` (required when the runner measures, as `--candidate`)
  takes the chosen grid into the spec; `measurements.json` records it.
- Tests: two airports with different E give one word for one height above E; the executor levels a word T at T + E MSL;
  rules 3 and 6 at an airport with E = 188 m; the grid fit finds a known grid in synthetic level-offs; `--grid` refuses
  a name that is not in the rows.
- The citations of the design in the code and the tests name the documents of `docs/two_tier/design/` and their
  sections, not the single document `two_tier_design.md` (now in `docs/history/2026-10_two_tier_design/`); the
  message of the commit that split it gives where each of its sections went.
- Then (outline §5 rule 12): the spec measurement on all train days in a scratch directory, with A19 and A20. The report
  gives the grid candidates and their rounding errors, and the rows of D56 again (they must not change). The user
  chooses the grid (D55).

**A21. The formal artefact (D56, D58, D59, D61).** Done: built from `9a986c09` on 2026-10-04 15:35–16:41Z into
`v9_20261004` / `v14_20261004` (report `readouts/2026-10-04_stage_a_a21_report.zh.md`); its signals, sentences and
closed-loop files equal the build before D61 bit for bit (A22 changed no word and no state), which is deleted with the
user's go. After A22.

- The formal artefact, from a clean checkout, into new directories:
  `4dTrajectory/outputs/POOLED/instruction_language/v9_<date>/` (signals of every development flight, the spec with
  `--candidate 0.25` and `--grid fitted` (D59), labels, the labeller conformance, the closed-loop reading at Δ = 2, 4, 8 s
  on every split) and `4dTrajectory/outputs/POOLED/executor/v14_<date>/` (the executor spec, its conformance).
- The build of 2026-10-04 13:36Z (`v8_20261004`, `v13_20261004`) was made before D61: its `candidates.json` has no
  vertical path. It is superseded and deleted with the user's go, never kept beside the formal artefact.
- The readings of D34 at Δ = 2, 4, 8 s: the closed-loop summary of every split (items 1–3) and the closed-loop replays
  (item 4) of train (400 flights an airport, seed 1337) and select (every labelled flight). The val replay waits for the
  user (as for every executor before). The data are made read-only with a `SHA256SUMS` beside them. The report gives
  the readings side by side; it sets no criterion (D7).

**A22. The public interface completed: the vertical path, the state columns, the grammar's column mask (D61, D62).**
Done: `0c07f92f` + `df84cbf1` (two reviews; §0.3). Before A21; stage B waits for it (B1, B4). The values do not change, only where they are: no word and no state of a
sentence changes.

- `instructions/airport.py`: `VerticalPath` (the TCH, the glidepath angle and the DA above the threshold), one for each
  candidate, moved from `autopilot/runway_data.py`. The first runner (`instruction_signals`) builds them from the
  harvest's runway data, as now, refuses a candidate without them (§4.2), and writes them into `candidates.json` beside
  the geometry. `candidates.json` gets a new format name (outline §5 rule 5).
- One function gives the height of the published glidepath of a candidate at a distance before its threshold, from the
  airport, the candidate and the distance (§6, item 4); the judge calls it.
- Every reader takes the vertical paths from `candidates.json`: the judge (the DA check), the closed-loop reading, the
  replay, the executor spec. Nothing reads the CIFP after the first runner. `autopilot/runway_data.py` goes.
- `instructions/artefact.py`: `STATE_COLUMNS` beside `CLOSED_LOOP_FIELDS`, and the function that reads a closed-loop
  file into its sentences (§6, item 3). `autopilot/closed_loop.py` imports both and keeps no copy.
- `instructions/grammar.py`: the column mask of D62 (the name is the code's). One call answers every word of the
  column, for a batch of aircraft: it runs at every aircraft, row and column of a speaker. `runway_words_allowed`
  (rule 2 only, no caller) goes.
- The architecture test of A15 (the executor's laws do not read the runway data) checks that the executor's laws read
  no `VerticalPath`.
- Tests: `candidates.json` writes and reads the vertical paths, and the old format name is refused; the DA check gives
  the same result from `candidates.json` as from the harvest's runway data on fixed flights; the reader gives back the
  words, the marks and the states that were written; the column mask equals its definition (every completion of the
  later columns through `apply`) for every column, on all rows of a small spec and on random rows of the spec of D59,
  with and without permitted words of the later columns; with a level angle in force, a level below the aircraft is
  permitted and the angle column then permits only the descent classes, but not when the caller permits no descent
  class; the labeller, executor and closed-loop conformance checks pass.
- The key code index (§11) follows the moves.

**A23. The Training view of stage A (outline §6).** Its code after A21, on `v9_20261004`; its export and publication
from the artefact of A25 (D66). The user sees the words, the closed-loop sentences and the executor of this stage in
the frontend.

- **Backend.** The live executor (`aeroviz_backend/autopilot_segment/`) on the new executor: `autopilot/single.py` with
  the executor spec of the formal artefact (`executor/v14_20261004`). It flies the words of this vocabulary from a
  state (five columns; a heading word with the course of R, D46; a level at T + E, D58; speed words in steps, D43) and
  imports no `prior/`. The backend tests that A0 left failing (`aeroviz_backend/tests/test_single_executor.py`,
  `test_autopilot_segment.py`, the route in `test_http_server.py`) are rewritten and pass.
- **Export** (the archived R11 and R13, rewritten). For each airport, a random sample of select flights and of train
  flights (seed 1337; the sizes stated in the export): the observed track; the open-loop sentence on the 2 s rows; at
  Δ = 2, 4 and 8 s the closed-loop sentence, each correction word marked, and its flown states; the judge's outcome and
  the DA check (its point, its vertical and lateral values); the attitudes. New schema names; its own index beside the
  old one (outline §6 item 3).
- **Frontend** (`aeroviz-4d`, the Training view). The five columns (runway and G, heading relative to the course of R,
  altitude above E, angle, speed); the correction words marked; a choice of Δ; the flown path beside the observed one;
  the outcome and the DA point; a click on a word flies its segment live with the new executor. It refuses the old
  schema names.
- **Publication and view.** The intent in `docs/experiments/intents.json`; the sets published for the five airports;
  a test stack from the worktree for the user (outline §6 items 4, 5).
- **Tests.** The backend's; the export (a sample written and read again); the frontend's readers (Vitest) on fixtures
  that the export writes; a live segment equals the export's flown states from the same state with the same words
  (the executor conformance tolerance); the browser check (outline §6 item 6).

**A24. The vertical tolerance of the final descent (D66).** After A22.

- The closed-loop reading (`autopilot/closed_loop.py`) uses H_final while "no level-off" is in force and H elsewhere;
  a correction ends below half the tolerance in force, and an overshoot is beyond the tolerance in force (D53). H_final
  is a value of the vocabulary spec: a new spec schema name and new names for every changed format (outline §5 rule 5).
  The rule of D50 and the readings of D34 (`experiments/instruction_closed_loop.py`) read the tolerance in force at each
  row.
- The measurement (outline §5 rule 12), in a scratch directory: for each H_final = 15, 10, 7.5 and 5 m, an artefact as
  `v9_20261004` in everything else, on the train sample of D34 (400 flights an airport, seed 1337) and on every
  labelled select flight; the closed-loop reading and its replay at Δ = 2, 4, 8 s. For each value, Δ and airport: the
  landed share and the outcomes (`unstable_at_minimums` high and low apart); the height above the glidepath at the DA
  point (p10 / p50 / p90); the flights whose real track passes the DA check and whose replay does not; the angle
  correction words in each final descent (the cost: words that the prior must learn). The report gives them side by
  side, with no criterion (D7); the user chooses H_final (D55). An airport that stays apart from the others at every
  value is named in the report: its cause can be in its data, not in the tolerance.
- Tests: the tolerance in force changes where "no level-off" starts and ends (a go-around included); in a final descent
  an e_h between H_final and H starts a correction, and before the final descent it does not; at H_final = H the
  sentences equal those of the spec without H_final.

**A25. The formal artefact with the chosen H_final (D56, D58, D59, D61, D66, D70).** After the user's choice of
H_final, A26 and A27. Built again by A30 (D73).

- A21's steps, from a clean checkout, into new directories:
  `4dTrajectory/outputs/POOLED/instruction_language/v10_<date>/` and `4dTrajectory/outputs/POOLED/executor/v15_<date>/`;
  the readings of D34 at Δ = 2, 4, 8 s; read-only with a `SHA256SUMS`. The report gives the readings beside those of
  A21.
- `v9_20261004` and `v14_20261004` are superseded: deleted with the user's go, never kept beside the formal artefact.

**A26. The start of a closed loop (D67).** After A24's code; before A25, so that A25's conformance records are those of
the code with it. The artefact's formats do not change.

- `autopilot/start.py`: for closed-loop sentences of the artefact at a Δ (a split, as the reader of `instructions/artefact.py`
  gives them), the executor spec and the most go-arounds of a flight: each sentence's flight rebuilt from the harvest
  and compared with the stored signals (`autopilot/flights.py`); its group and approach speed by the rule of the replay
  (`autopilot/replay.py` `group_of`); its time limit (the remaining observed time from the first predicted step × 1.5)
  and the time its go-arounds may add; the executor at the first predicted step. A step: the words of one row for each
  flight → the states of the 2 s rows flown (`STATE_COLUMNS`) and the flights done. A flight's outcome:
  `autopilot/judge.py` `outcome_of`.
- The closed-loop reading (`autopilot/closed_loop.py`) starts its flights with it; its sentences do not change.
- Tests: the words of a closed-loop sentence said through the start give its stored states on the 2 s rows (within
  the executor conformance tolerance) and its outcome, at Δ = 2, 4 and 8 s; the time limit equals the closed-loop
  reading's; a go-around beyond the most given is refused by name; the closed-loop and executor conformance checks
  pass.

**A27. The stratum in the sentence file (D70).** After A24's code; before A25, so that the artefact of A25 holds it.

- `instructions/artefact.py`: the sentence file holds each labelled flight's stratum by its name
  (`instructions/readout.py` `STRATA`), from `instructions/readout.py` `stratum` of its reading; a new format name.
- The labeller conformance (`instructions/conformance.py`): the record of a labelled flight holds its stratum, and
  the check compares it; the reference gets a new format name.
- Tests: the stored stratum of each flight equals `stratum` of its reading; a flight whose turns before the capture
  row add up to 90° is vectored, one at 89° straight-in; a changed stratum fails the labeller conformance by name; a
  sentence file of the former format is refused by name.

**A28. The start opens the executor spec (D71).** After A26. It changes no artefact.

- `autopilot/start.py` `start`: the directory of the executor spec in place of its parameters and the words. It
  opens the spec with `autopilot/replay.py` `open_executor` (the spec's vocabulary against the artefact's, the
  labeller and the executor conformance) and keeps the check that the closed-loop file was flown by these
  parameters; the words are those of the artefact's vocabulary. The runner of A26's check (R53) gives the directory.
- Tests: a spec measured against another vocabulary, a spec whose reference tracks the code on disk does not fly
  within the bounds, and a closed-loop file flown by other parameters are each refused by name; A26's test of the
  stored states passes through the new signature.

**A29. No code fingerprint (D73).** After A28.

- The labeller, executor and closed-loop checks (`instructions/conformance.py`, `autopilot/conformance.py`,
  `autopilot/closed_loop.py`: each `check`) run in the process that needs them, before its work: labelling an
  artefact; opening an executor spec (`replay.open_executor`: the executor check; the start of D71, the replay, the
  backend at its start, in its warm-up); reading the closed loop or flying its sentences again
  (`require_conforming_closed_loop`, which takes the artefact and the executor spec's directory). A difference
  refuses by name. The run records its commit and the checks' largest differences, as information.
- Deleted: the passed records and their schemas; `executor_source_files`, `executor_source_sha256`,
  `closed_loop_code_sha256`, `labeller_code_sha256`, `checker_sha256`, `REACHED_MODULES`, `UNHASHED_IMPORTS`; every
  code digest written into a spec, an artefact, a reference or a readout; the guards that compare a code digest
  before and after a run; the clean-checkout rule of a check; `executor_conformance --write-reference`. A reference
  is written only in the run that writes what it pins (the labeller's with the labels, the executor's with its
  spec, the closed loop's with the closed-loop reading). Each format that held a code digest gets a new name. The
  index lines of the ts `CLAUDE.md` and their reference text (C30, C33) follow, and the key code index (§11).
- Also deleted: code that stayed only because the digest covered it. The `override` hook of `outputs/base.py`
  answers None on every path and stayed because `data/dataset.py`, which asks it, was in the executor's digest; the
  hook and its call go.
- Tests: a check that finds a difference refuses by name in each place that runs it; a change of the code that
  keeps the behaviour opens everything with nothing run beforehand; no module of `instructions/`, `autopilot/`,
  `prior/`, the runners or the backend computes a digest of code (`tests/test_architecture.py`).

**A31. The outcome of each closed-loop sentence (D74).** After A29; before A30.

- `autopilot/closed_loop.py`: the reading gives each sentence's outcome by the judge (`judge.outcome_of`) on what
  its executor flew, to the end; `instructions/artefact.py` stores it in the closed-loop file by the outcome's name
  (a new format name) and the reader of item 3 gives it with each sentence.
- Tests: the stored outcome of a sentence equals the replay's outcome of the same sentence (`executor_replay
  --closed-loop`) and the start's (A26), at Δ = 2, 4 and 8 s; a landing and each kind of failure are stored by name;
  a closed-loop file of the former format is refused by name.

**A30. The formal artefact once more (D73, D74).** After A28, A29 and A31.

- The steps of A25, from a clean checkout, into new directories `instruction_language/v11_<date>/` and
  `executor/v16_<date>/`; read-only with a `SHA256SUMS`.
- The report compares it with A25: every signal, sentence and closed-loop array equal, the readings of D34 equal;
  for each split and Δ, the sentences by stored outcome (D74), and on the replayed flights the stored outcome equal
  to the replay's.
- `v9_20261004`, `v14_20261004` and the directories of A25 are superseded: deleted with the user's go.

**A32. The corrections of Claude's review (D77–D85).** After the user's decisions of 2026-10-05. On the branch
`dev-two-tier-v4-a32`, made from `dev-two-tier` (it holds `dev-two-tier-v4` and these decisions), in the worktree
`.claude/worktrees/two-tier-v4-a32`. Nothing is built. The formats change, so the code refuses `v11_20261004` and
`v16_20261004` after it (principle 8): the branch does not reach the live line before A34 and A35.

- D77 (`autopilot/flights.py`, `autopilot/start.py`): the start state from the 2 s rows at or before the first
  predicted step, by a named rule: `displacement-2s`, `trailing-fit-8s`, `trailing-fit-15s`, and `centred-fit-15s`
  (today's; for the comparison of A33 only, refused by a formal executor spec). The airspeed from the ground speed and
  the vertical rate without wind, as §5.6 does. The rule is a required field of the executor spec (a new spec name).
  The stored observed rows take their track, ground speed and vertical rate from the same rule, the track in
  [0°, 360°). The data plane's channels (`data/`, `flight_scenarios/`) do not change: the other paths read them.
- D78 (`experiments/instruction_signals.py`, `instructions/airport.py`): the candidates are every runway end of the
  airport with a published vertical path in the CIFP data that `runway_targets` reads today
  (`trajectory_data_process/harvest/arrivals.py`), with or without arrivals. A flight assigned to an end that is not a
  candidate is refused by name and counted. `--list-candidates` writes, for each airport, the ends added and removed
  against a given artefact and their arrivals by split, and builds nothing (A33). A new candidates format name.
- D79 (`autopilot/executor.py`, `autopilot/single.py`): the end of a flight with the judge's tests (one definition,
  called by the judge and the executor): a lined-up crossing of another candidate with G false, and the stall cut-off,
  besides the ends of today.
- D80 (`autopilot/start.py` `Loop.step`): the grammar's `apply` on each flying flight's row before `say`, at the height
  above E of the executor's state; a value outside its column refused.
- D81 (`autopilot/flights.py`, `autopilot/conformance.py`): the frame's origin at the airport reference; the behaviour
  check, run where the executor check runs (D73). It replaces the scan of names in `tests/test_architecture.py`; the
  scan of imports stays.
- D82 (`instructions/artefact.py`, `autopilot/closed_loop.py`): the reader's two parts, new format names. The
  consumers on this branch follow: `experiments/training_export.py`, `experiments/training_flights.py`,
  `aeroviz_backend/autopilot_segment/`, R52, R53. Stage B's code follows on its own branch after the merge (outline §5
  rule 1).
- D83 (`autopilot/closed_loop.py` `ObservedPath`): e_y to the path's segments.
- D84 (`autopilot/executor.py`, `autopilot/single.py`): the roll rate from the first cycle.
- D85 (`experiments/instruction_closed_loop.py` and its summary): no reading of the val days in the summary or the
  printed text; the val sentences are written as before.
- D86 (`experiments/training_export.py`: `formal_rows`, `choose`, `split_flights`, `replay_payload`): no formal replay
  row; the draw from the closed-loop sentence files, the stratum from the sentence file, the check against the stored
  states and outcome; a val flight exported in a test.
- The references (§7.2 #2, #3): the train flights with a labelled go-around added.
- Checks that the existing rules ask for: the day check of C32 where the labeller's reference reads its signals
  (`instructions/conformance.py`) and in `artefact.signals_flights`; `--spec-from` (`artefact.keep_spec`) refuses a
  source whose train days are not train days of this artefact's split; the labeller's "unspecified" at an approach's
  last row is refused by name, never an `IndexError` (`instructions/labeller/speed.py`, `labeller/sentence.py`).
- A23's code: `experiments/training_flights.py` starts its flights through the start of D67, so it has the start's
  refusals (the closed-loop file's executor parameters, the observed rows against the signals), and flies with one
  single-flight loop, the conformance's; `aeroviz_backend/autopilot_segment/backend.py` keeps a failed check for each
  (artefact, executor spec) and refuses at once with its reason; a request for a split other than train and select is
  refused.
- Tests, each one fails on the code before A32: a change of the observed samples after the first predicted step changes
  no start state and no stored observed row (D77); a flight added or removed on any day changes no candidate (D78);
  a lined-up crossing of another candidate and the stall cut-off end the flight at that cycle in the three ways to fly,
  and `Loop.step` gives it as done (D79); `Loop.step` refuses "go-around" while G is true, a value outside its column
  and "no level-off" while G is true, with nothing changed (D80); the executor's inputs hold no value of the landed
  runway, and the behaviour check refuses a test law that reads a vertical path (D81); the reader's two parts, and a
  former format refused by name (D82); e_y at a vertex on the outside of a turn is the distance to the path (D83); the
  bank after the first cycle is at most p · 1 s (D84); no reading of val in the summary (D85); the backend refuses at
  once after a failed check, and refuses a val request.
- The targeted tests, the full ts suite and the backend's tests pass (outline §5 rule 4); a code review (rule 2).

**A33. The measurements for the user's choices (D55, outline §5 rule 12).** After A32, on all train days, in a scratch
directory (nothing under `4dTrajectory/outputs/`), with A32's code.

1. A scratch artefact by the first steps of A30 (signals, spec, labels, conformance; about 4 min) with the candidates of
   D78.
2. The start rule of D77: a runner (R55, as R52) reads the closed loop of every train flight at Δ = 4 s (the chosen Δ)
   once for each rule: `displacement-2s`, `trailing-fit-8s`, `trailing-fit-15s`, and `centred-fit-15s` for the
   comparison. For each rule: the readings of D34 (items 1–4), the outcomes, and, on the flights that turn at the first
   predicted step (more than 5° between the 8 s before it and the 8 s after it), the start track against the observed
   direction of the next 8 s (a readout of the future, never an input). Train only (D85).
3. The candidates of D78: for each airport, the ends added and removed against `v11_20261004`, and their arrivals by
   split (the test days only counted, from the roster, C32).
4. The climb nominal (D54, D56) fitted again on the climb pieces with G false, inside the labeller's angle range, beside
   D56's 1.5°. A change of its 0.25° rounding goes to the user (D15).

The report gives the values to the user; the user chooses the start rule. A34 waits for the choice.

**A34. The formal artefact once more (D77–D85).** After the user's choice of the start rule, from a clean checkout of
the code of A32. The steps of A30 (the spec with `--candidate 0.25 --grid fitted --closed-loop-final-vertical-m 10`, the
executor spec with the chosen start rule, the closed loop at Δ = 2, 4, 8 s with `--workers 4 --train-parts 8`, the
replays of train (400 an airport) and select at each Δ, the start check), into `instruction_language/v12_<date>/` and
`executor/v17_<date>/`; read-only with a `SHA256SUMS`. The report compares it with A30 on train and select only
(D85): the candidates of each airport; at each Δ, the readings of D34, the outcomes and the landed share; the sentences
that changed, by cause (the start, the end, the candidates, e_y).

**A35. The Training sets of stage A again (A23, outline §6).** From A34's artefact: its intent in
`docs/experiments/intents.json` first; `training_export` into a new set beside `closed_loop_v11_20261004`; `check_live`
over the new set; the browser check by a one-shot agent; a test stack for the user. After the user's merge,
`v11_20261004`, `v16_20261004` and the set `closed_loop_v11_20261004` are deleted with the user's go (the backend's
live executor reads the artefact of its set, so they go together).

**A36. One function for the closed-loop part of the export (stage B's Training view, prior B6).** Before A32, on the
branch `dev-two-tier-v4-a36`, made from `dev-two-tier`; Claude fast-forwards `dev-two-tier-v4` on the user's word,
and stage B merges it (outline §5 rule 1: stage B does not change stage A's code). A change of the code only: no format,
no artefact and no published set changes.

- `experiments/training_export.py`: `split_flights(instructions, split, chosen, rows, params, words, *, device)` gives,
  for the flights `chosen` of one split and one airport, in that order, each flight's payload — its head
  (`flight_head`: the observed track, the open-loop sentence) and, at each Δ of `rows`, its closed-loop sentence flown
  again (`replay_payload`, against the stored states and the formal replay's row) — and the airport's geometry. It is
  the body of the loop over the splits in `build_airport`, moved unchanged; `build_airport` calls it. The stratum and
  the kind of a flight are read from the rows of the first Δ given (today `2.0`).
- `formal_rows(executor, split, intervals=ROW_INTERVALS_S)`: the Δ values to read; the default is today's.
- Stage B's export (`experiments/prior_training_export.py`) calls `split_flights` with the prior's Δ, for example
  `(4.0,)`. A32 changes the function with the rest of A23's code (D77, D82, D86: no formal rows).
- Tests: A23's export tests pass unchanged, and a sample written before and after the change is the same, byte for
  byte; `split_flights` with one Δ gives that Δ only. Stage B's session wrote a reference patch (90 lines; it can be
  gone from its scratchpad: `/tmp/claude-1000/-home-supercomputing-studys-thesis/7767b9f9-586d-4d01-9b82-2949db171a8a/scratchpad/split_flights.patch`;
  13 export tests passed with it).

### 12.2 Claude's check of stage A

The check of A0–A14 is done (`readouts/2026-10-04_stage_a_check.zh.md`, at `ab295b18`; its points became D48–D52).
After A21, Claude checks A15–A22 and the formal artefact. The formal runs of stage B wait for this check (outline §4).
Done at `9a986c09`: all five items pass (`readouts/2026-10-04_stage_a_check_a15_a22.zh.md`).

1. D48–D56 against the code: done at `688e945e`. D57, D58, D59, D61 and D62, and the code changed after `688e945e`,
   against the code of the formal build of A21.
2. The targeted tests of A15–A22 pass on the commit of the formal build (run again); the full suite of that commit
   passed (outline §5 rule 4; its log).
3. The formal artefact: the spec holds the values of D56 and the grid that the user chose (D58), and its measurement
   records both choices; `candidates.json` holds the vertical path of every candidate (D61); the labeller, the executor
   and the closed-loop conformance records pass for the code of the build; at each Δ, the closed-loop sentences of the
   replays give their flown states again on the 2 s rows, and the rule of D50 holds on every stored row; the readings of
   D34 (items 1–4) exist for every Δ, each split that A21 reads and each airport.
4. No write under a live root except the new directories of A21; no existing directory under `4dTrajectory/outputs/`
   changed; the superseded directories of A18 and of the build before D61 deleted; the `SHA256SUMS` beside the formal
   data match.
5. Nothing in the archive was edited after the move.
6. After A23: the published sets open and play in the browser; a live segment equals the export's flown states; the
   old Training index and its sets are unchanged; the backend tests pass (outline §6).
7. After A24 and A26–A31: D66, D67, D69–D71, D73 and D74 against the code (A26's test of the stored states run
   again on the artefact of A30, through the start of A28; the labeller conformance of A30 compares the strata; the
   stored outcomes equal the replay's on the replayed flights; no digest of code in the code, the artefact or the key
   code index); the spec of the formal artefact holds the chosen H_final and its measurement records the choice;
   the readings of D34 exist for every Δ, split and airport and equal A25's; the artefacts of A21 and A25 deleted
   with the user's go.
8. After A32–A36: D77–D86 against the code; A36's sample equal to A23's byte for byte; A32's tests fail on the code before it (`dev-two-tier-v4` at `cf549e47`);
   the behaviour check of D81 passes on A34's executor spec; A34's candidates are the published ends of each airport;
   no start state and no stored observed row reads a sample after the first predicted step (on A34's artefact: the
   observed samples after it changed, the start the same); no reading of the val days in A34's report or summaries;
   A35's set opens and plays, and its live segments equal its export; the superseded artefacts deleted with the user's
   go.
