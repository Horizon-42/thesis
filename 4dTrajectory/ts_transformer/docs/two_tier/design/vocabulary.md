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
| D11 | The labeller stage includes an ablation of the row interval Δ (§4.8). Δ = 4 s, the user's choice on the readings of D34 (`readouts/2026-10-04_stage_a_a25_report.zh.md`): a speaker says a row and the executor hears it every 4 s, and the step of a scene of several aircraft is 4 s. The artefact keeps the closed-loop sentences at Δ = 2, 4 and 8 s. That report also shows val rows: it was written before outline D85 | Decided. Values: D25. Readings: D34. Δ = 4 s | User, 2026-10-03; Δ = 4 s: user, 2026-10-04 |
| D12 | No runway lock. While G is false the model can change the runway at any time; the judge reads R at the crossing (§3.2) | Decided | User, 2026-10-03 |
| D14 | While G is true, "no level-off" is not permitted (rule 5, §3.7). | Decided | User, 2026-10-03 |
| D15 | The spec measurement gives each value that it fits from data with rounder candidates and the fit that each leaves; the user chooses (§3.5) | Decided | User, 2026-10-03 |
| D18 | The labeller reads the real go-arounds inside a sentence and says "go-around" at the go-around point (§4.6) | Decided | User, 2026-10-03 |
| D19 | After a labelled go-around, the runway word that ends G is at the first level-off after the go-around climb, and not later than the row of the next "no level-off" (§4.6) | Decided | User, 2026-10-03 |
| D22 | Altitude words use the grid "optimal 40 levels" of the altitude-grid proposal: 60 m steps from 0 to 1,260 m, 120 m steps to 2,700 m, 450 m steps to 5,400 m (§3.4). The heights are above the airport elevation (D58); the fit on them gives the same grid (D59) | Decided | User, 2026-10-03 |
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
| D50 | The closed-loop reading does not keep every row inside the tolerances of D32. A correction is one class and needs time to bring the flown path back (§9.7: a 100 m offset takes approximately 20 s at one 5° class); in the stage A smoke 10–25 % of the correctable rows are more than 30 m off laterally. The check is a rule on the stored data: on every row where §4.9 permits a correction, \|e_y\| > Y is followed by a heading correction in force after that row, and \|e_h\| > H by an angle correction; with D53 the reading keeps this rule on every row. The p90 of \|e_y\| at the last row is information. The share of the correctable rows outside the tolerances is a reading of the ablation (D34), not a check (§4.8) | Decided | User, 2026-10-04 (check §3 item 4) |
| D51 | The closed-loop artefact stores the flown states on the data's 2 s rows at every Δ, with the Δ rows marked. Observed and flown rows are then on one grid, and a motion input over the 2 s before a row exists at every Δ. The artefact stores raw states, not a derived input, so a change of the motion window needs no rebuild. The replay check of a closed-loop sentence compares on the 2 s rows. At Δ = 2 s nothing changes (§4.8, §4.9) | Decided | User, 2026-10-04 (check §3 item 1) |
| D52 | ε of a level = half the larger gap to its neighbouring levels + 10 m: 40 m for 0–1,200 m, 70 m for 1,260–2,580 m, 235 m for 2,700–5,400 m. A level's band then covers the largest rounding error of its word; the top level of a segment collects the heights up to half the next segment's step above it. One band for both sides (a one-sided band would differ at two levels only). "No level-off" has ε = 40 m: it has no level, so no rounding to cover; its tube's width comes mainly from the edges of the angle class (§3.4, §8) | Decided | User, 2026-10-04 (check §3 item 5) |
| D53 | An overshoot. When a correction is in force and the error changes its sign while it is more than the tolerance (the flown aircraft crossed the observed path in one row), the labeller says the opposite correction in the same row: one class toward the path from the observed word. When the error changes its sign and is within the tolerance, it says the observed word again. Laterally and vertically alike. The rule of D50 then holds on every row (§4.9, §12.2). Why: a correction ends at a sign change so that it does not push the aircraft further on the other side; waiting one row to correct that side adds a lag of one row Δ that comes from the reading, not from Δ, and the ablation compares Δ (A15 smoke: every break of D50 at Δ = 4 and 8 s was such a row) | Decided | User, 2026-10-04 |
| D54 | The descent classes are a k-means on tan(angle) of the descent pieces, with each piece weighted by the square of its length. A piece flown at the nominal angle of its class ends length · \|tan a − tan c\| from its observed end, so the k-means makes the sum of the squared end-of-piece height errors smallest; D15 gives the same error for each candidate. The climb nominal is the length-weighted median of the climb pieces with G false, at most the climb class's 15° (a go-around climb is flown at the go-around angle, D28, so it is not in the fit) (§3.5) | Decided | User, 2026-10-04; the pieces with G false: user, 2026-10-05 (A33) |
| D56 | The values of D15, chosen: the candidate rounded to 0.25° of the spec measurement on all train days. Descent nominals 1.5 / 2.5 / 3.0 / 4.5°, edges −0.5 / 2.0 / 2.75 / 3.75 / 10°; climb nominal (G false, D54) 1.25°. The formal artefact's measurement (`instruction_language/v12_20261005` `measurements.json`: 44,703 train flights, 227,559 descent pieces, 1,076 climb pieces with G false): fitted 1.51 / 2.45 / 3.09 / 4.45°, climb 1.236°; the end-of-piece height error of the descent pieces p50 / p90 15.0 / 51.0 m (fitted: 14.7 / 47.8 m). Descent 3 is the published 3.0° glidepath of 24 of the 25 candidate runways (§3.5, §8) | Decided | User, 2026-10-04; the climb nominal of the pieces with G false: user, 2026-10-05 (A33) |
| D57 | One word clock: `time`. Every sentence is said on its own rows: an open-loop sentence, a closed-loop sentence and a sentence of a speaker. The executor spec has no clock field. Why: no training sentence and no reading of D34 uses the clocks `distance` and `track` (an open-loop word said where the flown aircraft reaches the observed track); the closed-loop reading has the one rule of the place (D42, D45); a clock in the executor spec gives a new spec identity, and a rebuild, for a change that no closed-loop sentence sees (D21); and the batch and the single-flight executors would each carry it. The question "how far do the observed words alone carry a flight" is read, when a readout needs it, with the closed-loop reading with no corrections (§4.9) | Decided | User, 2026-10-04 |
| D58 | The altitude words are heights above the airport elevation E (the published field elevation, one value for each airport, `candidates.json` `reference.elevation_m`), not MSL. The labeller reads them from the height above E; the executor flies a level T at T + E MSL. E, not the threshold of R: R can change during a sentence, and the thresholds of one airport differ by up to 28 m (KSTL 160.8–188.4 m). The spec measurement fits the grid of D22 again on the level-offs above E, with the D22 grid among the candidates; the user chooses (D55). Why: at a high airport (KDEN, approximately 1,650 m) the MSL words of an approach are far from what the five training airports (1–188 m) said for the same manoeuvre, and they fall in the coarse segments of the grid; and the day-to-day spread of the geometric height of an assigned level grows with the height above the altimeter-setting source, the airport. The cost: the round MSL levels that controllers assign fall on different words at each airport (3,000 ft MSL: 900 m at KMSY, 780 m at KRDU, 720 m at KSTL) (§3.4). | Decided | User, 2026-10-04 |
| D59 | The grid of D58, chosen: the fitted row of the spec measurement on all train days (code `220e858e`, 44,703 train flights, 25,727 level words above E): 60 / 120 / 450 m, break points 1,260 and 2,700 m, 40 levels — the grid of D22 itself; rounding error of the level words p50 / p95 / largest 17 / 34 / 190 m. The next best of the 4,515 grids of the fit leaves 7.6 % more squared error. A21 builds with `--grid fitted` (§3.4, §9.5) | Decided | User, 2026-10-04 |
| D61 | The vertical path of each candidate (its TCH, its glidepath angle and its DA above the threshold) is part of the artefact: the first runner writes it into `candidates.json` beside the geometry, and every reader (the labeller's refusal, the judge, the closed-loop reading, the replay, the executor spec, the prior) takes it from there, never from the CIFP at run time. The names of the state columns of a closed-loop file and the function that reads a closed-loop file into its sentences are in `instructions/artefact.py`. Why: the public interface (§6, items 3 and 4) gives them, but the code held them in `autopilot/`, which the prior cannot import; and a value read from the CIFP at run time is not bound to the artefact (§6, milestone A22) | Decided | User, 2026-10-04, on the report of the stage B agent |
| D62 | The grammar's mask for a speaker (§6, item 2) is one function in `instructions/grammar.py`. It takes the words in force (with R and G; none before the first step), the words that the earlier columns of the row said, the column asked, the height above E, the number of candidates and, for each later column, the words that the speaker's other masks permit (all words when not given). It gives, for each word of the column ("unchanged" and, in the runway column, "go-around" included), whether some words of the later columns, each among its permitted words, make the row pass `apply`. Its definition is `apply`, not a second copy of rules 1–6. Why: a speaker says a row column by column, and a check with the later columns "unchanged" refuses good words (a level below the aircraft, which a descent class in the angle column makes grammatical); a copy of the rules in the prior breaks the one definition (§3.7); the permitted words of the later columns make sure that a row never reaches a column with no permitted word (milestone A22) | Decided | User, 2026-10-04, on the report of the stage B agent |
| D66 | While "no level-off" is in force (the final descent), the vertical tolerance of the closed-loop reading is H_final = 10 m, smaller than H; elsewhere H = 15 m. The final descent is told by the words, not by procedure data (principle 3). The user chose H_final from a measurement (D55, milestone A24): the closed-loop reading and its replay at Δ = 2, 4, 8 s for H_final = 15, 10, 7.5 and 5 m, on the train sample of D34 and on every labelled select flight. A correction ends below half the tolerance in force; the rule of D50 and the readings of D34 read the tolerance in force at each row. Why: the DA check allows 22 m (D38); the real tracks lie up to +8 m (p90) above the glidepath at the DA point; with H = 15 m the flown path can stay up to 15 m above the observed one before a correction starts, and the correction then needs time. 67 of 1,999 train replays at Δ = 2 s ended `unstable_at_minimums`, 61 of them high, and 57 of these 67 flights passed the same check on their real track (`readouts/2026-10-04_stage_a_check_a15_a22.zh.md` §7). These sentences are the prior's training sentences | Decided; H_final = 10 m, the user's choice from A24 | User, 2026-10-04, on Claude's reading of the check; 10 m: user, 2026-10-04 |
| D67 | The start of a closed loop is part of the public interface (§6, item 5): one function in `autopilot/start.py`. It takes closed-loop sentences of the artefact as the reader of item 3 gives them (a split and a Δ), the executor spec of the artefact and the most go-arounds that a flight may say. It gives the executor at the first predicted step of each sentence's flight: the flight rebuilt from the harvest and compared row by row with the stored signals (§7.2 #4), its aircraft (its own dynamics or a stand-in's, by the rule of the replay) and its approach speed, its time limit (§5.8) and the time that its go-arounds may add (900 s each, up to the most given; a further go-around is refused, and the caller masks it). Then, at each Δ row, the caller gives the words of the row for each flight; the executor flies Δ seconds and gives the states of the 2 s rows flown (`STATE_COLUMNS`) and the flights that are done. The judge gives the outcome of a flight from what the executor recorded, with no observed words (§6, item 6). The closed-loop reading (§4.9) starts its flights with the same function. Why: a speaker's closed loop (the prior's free generation, the post-training) needs the start, and the code that makes it (`autopilot/replay.py`, `autopilot/flights.py`) is not in the public interface: a caller of that code breaks when stage A changes it. The start is not stored in the artefact: the artefact holds the signals, not the dynamics (§7.2 #4) (milestone A26) | Decided | User, 2026-10-04, on the report of the stage B agent |
| D69 | The check that closed-loop sentences may be read is part of the public interface of the sentence artefact (§6, item 3): `autopilot/closed_loop.py` `require_conforming_closed_loop`. It refuses unless the code on disk has passed the closed-loop conformance of the artefact (§7.2 #2). A runner of another document that reads closed-loop sentences calls it first and takes nothing else from that module. Why: item 3 names the closed-loop conformance record as part of its identity, but its code column named no function that checks it; the prior's training must check it (§7.2 #2), and the prior's architecture test lets a runner import only what §6 lists | Decided | User, 2026-10-04, on the report of the stage B agent |
| D70 | Each labelled flight's stratum (§4.3: straight-in or vectored) is stored in the sentence file by its name (`instructions/readout.py` `STRATA`) and is part of item 3 (§6). The labeller writes it with the one definition (`instructions/readout.py` `stratum`); the labeller conformance compares it as it compares the words. The sentence file and the labeller's reference get new format names. Like the capture row, the stratum uses later rows: it is for readouts and strata, never an input. Why: the readouts of the other documents are by stratum (the prior's free generation), and the stratum needs the labeller's turns before the capture row, which the artefact did not store; a document that computed it again would make a second definition (milestone A27, before A25) | Decided | User, 2026-10-04, on the report of the stage B agent |
| D71 | The start of a closed loop opens the executor spec itself (§6, item 5): it takes the directory of the spec, not its parameters, and opens it as the replay does (`autopilot/replay.py` `open_executor`). It is refused unless the spec was measured against the artefact's vocabulary, the labeller and the executor code on disk pass their conformance (§7.2 #2, #3), and the closed-loop sentences were flown by these parameters. A caller handles no executor parameters, so no caller can start a closed loop with a spec that is not checked. The closed-loop reading, which writes the sentences, builds its loop from the same pieces and opens its spec with the same function. Why: the opener is not in the public interface, and naming it there would leave the checks to each caller (the prior's free generation, the loops of the post-training) (milestone A28) | Decided | User, 2026-10-04, on the report of the stage B agent |
| D73 | No code fingerprint (D21). The labeller, the executor and the closed-loop reading are checked by what they do on their reference samples (§7.2 #2, #3). A runner that labels, opens an executor spec, reads the closed loop or replays sentences runs the checks of the code it uses in its own process, before its work, and a difference refuses it by name. A change of the code in `autopilot/` or `instructions/` is checked when it is made: its milestone runs the checks on the formal artefact through their runners before its review (outline §5 rule 2). The backend runs no check at its start. It produces no result that anyone compares with another, and it checks what it shows by what it flies: an answer is refused by name (`ExecutorDiffers`) when the live flight is farther than `STATE_BOUND_M` from the artefact's stored flown states, or when a flight flown to its outcome ends otherwise than the stored outcome. What these do not see (a change of the judge's DA check, or of a limit or a mode that moves no state) the checks at the code change and `check_live` see. There is no passed record and no digest of code: not as a name, not as a guard of a run, not as information in a spec, an artefact or a reference; and no clean-checkout rule for a check. A reference is written only in the run that writes what it pins. A run records its commit and the checks it ran, as information. Data digests stay: the spec shas, the sha256 of files, the format names. Why: passed records named by a digest of the code's logic refuse every runner after any change in `autopilot/`, even one that moves no track, until the checks run again from a clean checkout, and a list kept by hand decides which files the digest covers. The checks take 1.7 s (labeller), 26.7 s (executor) and 16.5 s (closed loop) on the formal artefact of A21 (one thread); in the backend's start (A34's artefact, a busy machine) they took about 100 s, while the page showed "flying": the check belongs where the code changes (the user, 2026-10-05) | Decided | User, 2026-10-04 ("代码指纹是毒瘤设计"); the backend and the check at a code change: the user, 2026-10-05; the refusal at `STATE_BOUND_M` in each answer: Claude's reading |
| D74 | Each closed-loop sentence's outcome is stored in its closed-loop file and is part of item 3 (§6): the judge's outcome (§5.8) of the flight that the closed-loop reading flies, from the first predicted step to the end, by the outcome's name. Like the capture row and the stratum, it uses later rows: it is for readouts and selection, never an input. The closed-loop file gets a new format name. Why: the prior's base learns only from sentences whose own words land (prior D75), and the reading already flies every sentence to its end, so the outcome costs nothing more; another document that flew them again would make a second definition (milestone A31, before A30) | Decided | User, 2026-10-04 |
| D77 | The start of a closed loop reads no 2 s row after the first predicted step (the row's own position and height are interpolated between the raw samples around it, D89). The executor's start state takes the position and the height of the observed row, and the airspeed, the track and the path angle from the observed rows at or before it, by the start rule, a field of the executor spec: `displacement-2s`, the displacement in the 2 s before the row. A rule's window is cut at row 0: row 0 takes rows 0 and 1. Every start of the executor (the start of a closed loop, the first row of a replay, the live executor) takes its start state by the same rule, and the stored states of the observed rows (§6, item 3) take their track, ground speed and vertical rate from it. Inputs give only what is known before the step (outline principle 7). Why: the data plane's velocity is a least-squares fit over 15 s centred on each sample, which reads up to 7.5 s after the row (Claude's review of stage A, 2026-10-05: on the select days at Δ = 4 s, in the 8 % of the flights that turn at the first predicted step, the start track from the centred fit is 4.4° (p50) from the observed direction of the next 8 s, against 11.5° for a fit over the 15 s before the row). The rule is the user's choice among `displacement-2s`, `trailing-fit-8s` and `trailing-fit-15s` on A33's readings (`readouts/2026-10-05_stage_a_a33_report.zh.md`, every train day at Δ = 4 s; the centred fit measured beside them for comparison only): on the 7.4 % of the flights that turn at the first predicted step, the start track is 5.9° (p50) from the observed direction of the 8 s after (`trailing-fit-8s` 8.8°, `trailing-fit-15s` 11.5°; the centred fit 4.5°); landed 97.76 % (97.78 / 97.73 %; the centred fit 97.81 %); 14 dynamics failures (3 / 1; the centred fit 0) — Claude's reading: the 2 s displacement reads the steps of the ADS-B altitude | Decided; the rule `displacement-2s` | User, 2026-10-05, on Claude's review of stage A; the window cut at row 0 and one start state for every start: user, 2026-10-05; `displacement-2s`: user, 2026-10-05 (A33) |
| D78 | The candidates of an airport are its runway ends with a published vertical path in the FAA CIFP (the LPV line, or the LNAV/VNAV line where no LPV line is published; D61). No flight decides a candidate. Why: candidates taken from the arrival manifest's `runway_targets` (an entry for each runway with an arrival on any day) let a sealed day add a class to a runway column: KSTL 06 has one arrival, on a test day (2026-07-22), none on the other days (C32). With published ends, an airport that is not in the training data, KAUS included, takes its candidates from published data only. `instruction_signals --list-candidates` lists, for each airport, its candidates and the runway ends with arrivals that are no candidate (their flights are refused at the build), each with its arrivals by split. On the live harvest the rule adds and removes no end at the five airports (A32's listing, 2026-10-05): KSTL 06 stays a candidate, by its published vertical path; the set is the same, and no flight decides it | Decided | User, 2026-10-05, on Claude's review of stage A |
| D79 | The executor ends a flight where the judge ends it (§5.8): an approach crossing of R, ground contact, a lined-up crossing of another candidate with G false (inside that runway's own limit), a dynamics failure (the stall cut-off included), or the time limit. The single-aircraft batch, the multi-aircraft batch and the single flight alike; the start gives this end as "done" (§6, item 5). Why: an executor that ends a flight only at a crossing of R, the ground, a state that is not finite and the time limit flies a flight that crossed another runway or reached the stall cut-off on, and gives it words, to its time limit (before A32, train, Δ = 8 s: one sentence with `crossed_other_runway` flown to its limit of 834 s) | Decided | User, 2026-10-05, on Claude's review of stage A |
| D80 | The start's `Loop.step` (§6, item 5) checks the row of each flight that flies with the grammar (`instructions/grammar.py` `apply`, at the height above E of the executor's state) before anything changes, and refuses the row by name. A value outside its column's words is refused too. Why: without the check the start took "go-around" while G was true (counted again, 900 s more time, the held airspeed set again) and a runway value in another column (read with a negative index), and a refusal inside a cycle left the loop changed. A speaker applies the grammar as a mask; the start is the boundary of the public interface | Decided | User, 2026-10-05, on Claude's review of stage A |
| D81 | The executor holds nothing of the landed runway or of a procedure: the frame that it integrates in has its origin at the airport reference (with E), not at the landed runway's threshold and TCH. The rule "the executor's laws read no vertical path" (§5.2) is checked by behaviour, as D73 checks code: the executor's reference tracks, flown again with the vertical path of every candidate changed and the frame's origin moved, give the same states (horizontally within 1e-4 m: a move of the origin sideways moves the states by up to 2.2e-6 m from rounding, A33, the user's choice of 2026-10-05; otherwise within the executor conformance tolerance) and the same end cycles. The check is by behaviour, not by a scan of names in `tests/test_architecture.py`: a scan does not see a value read through `getattr`, a dictionary or the frame. Why: a frame with its origin at the landed runway's threshold and TCH (`FlightInputs.frame_params` before A32) let a caller read the landed runway from the executor, although no law read it (a move of 3 km and 15 m changed the states by less than 1e-7 m) | Decided | User, 2026-10-05, on Claude's review of stage A |
| D82 | The reader of item 3 (§6) keeps what a model may read apart from what it must not read. For each sentence it gives (a) the rows: the words with their correction marks (targets), and the states; and (b) apart, by name, the fields that use later rows or the observed path: the landed runway (the flight's runway and its index), the landing time, the capture row, the go-around rows, the stratum, the outcome, `timed_out`, the matched and observed rows, the errors e_y and e_h, and the rows without a correction. The states of the observed rows follow D77; the track is in [0°, 360°) on every row. New format names. Why: before A32 these fields came with the rows, and only the capture row, the go-around rows, the stratum and the outcome were named as never inputs; the observed rows held the centred fit (up to 5.5 s after the first predicted step) and an unwrapped track | Decided | User, 2026-10-05, on Claude's review of stage A |
| D83 | The lateral error e_y of the closed-loop reading is the signed distance of the flown position from the observed path: from the line through the observed 2 s rows near the matched point (its segments, not their extensions), positive to the right of the path (§4.9). At a vertex of the path (the matched point is a row's position), e_y is the smaller of the distances to the two segments around it, on the side of the nearer one; where both are nearest at the vertex (the outside of a turn), the side is that of the sum of their normals, and where the path turns there by more than 170° (a reversal: the normals nearly cancel and give no side) it is the matched segment's. The matched point never moves back. Why: at a vertex on the outside of a turn, the extended line of the matched segment is nearer than the path. At a reversal the sum of the normals gives no side: on A34's artefact, 4 sentences of 2 flights of train and select (val not shown, D85) had an e_y of the opposite side at a vertex where the observed path nearly reverses (a turn of 176.8°–178.4°, a fault of the observed track, D111). The value 170° is Claude's (A37): it is high so that every vertex with a smaller turn reads as in the formal artefact, and no artefact is built again; it is not the 120° of D111, which marks a flight | Decided | User, 2026-10-05, on Claude's review of stage A (the change at a reversal: the user, 2026-10-05; the value 170°: Claude's) |
| D84 | The bank limit and the roll rate hold from the first cycle of a flight (§5.3); the start bank is 0. Why: before A32 the executor took any bank up to the bank limit in its first cycle (no roll-rate limit there), a law that §5.3 does not have. The user chose no exception and no start bank read from the data | Decided | User, 2026-10-05, on Claude's review of stage A |
| D86 | A23's export, and stage B's through `split_flights` (A36), read no formal replay row. They draw their flights from the flights of the closed-loop sentence files (at every Δ that the set shows), take each flight's stratum from the sentence file (D70), and check each sentence flown again against its stored states on the 2 s rows and its stored outcome (D74), for every split. Why: the formal replays are of train and select only (the replay of the val days waits for the user), so an export from them cannot give the val flights of the base model's one validation readout (outline §6.1 item 4); the stored outcome is the one definition of a sentence's outcome (D74), and it equals the replay's on 8,195 of 8,195 replayed flights (A30) | Decided | User, 2026-10-05, on stage B's request |
| D87 | The readings of A32, accepted by the user: (1) a start rule's slope, in metres of the airport frame, is turned into metres on the ground with the WGS84 radii of curvature at the row (D77); (2) the row of a flight that is done or halted is not checked by the grammar and is stored as given (D80); (3) a flight's kind in the export is whether its closed-loop sentence at the first Δ says "go-around" (D86); (4) of the val days, the closed loop's summary keeps the flights drawn and the sentences written, and no reading (D85). Reading (5), the backend keeping an error of its start checks until it restarts, no longer applies: the backend runs no check at its start (D73) | Decided | User, 2026-10-05, on the readings of A32 |
| D88 | The readings of A33, accepted by the user: (1) R55 flies no replay: the outcome of each start rule is the closed-loop reading's own (D74); (2) a climb piece is in G when G is true at its first row (D54); (3) in a flight with a go-around whose vertical reading the labeller refuses, G is not known, and its climb pieces are left out of the fit and counted (D54); (4) the measurement reads G under the provisional spec's altitude grid (D22's) in its first pass (D54); (5) the way `moved` of the behaviour check keeps the executor conformance tolerance (1e-6 m) vertically (D81) | Decided | User, 2026-10-05, on the readings of A33 |
| D89 | The positions and heights of the observed 2 s rows stay interpolated between the raw samples around each row. A row reads the sample after it: p50 0.81 s, p99 1.69 s, at most 13.1 s later (KRDU, 300 train flights). The position of a row is where the aircraft was at that time, not a prediction, and on 99 % of the rows the next sample is less than one row away | Decided | User, 2026-10-05, on the second review of stage A |
| D90 | The start's `Loop` has no `timed_out()`: a caller reads why a flight ended from the judge's outcome (§6, items 5 and 6; the outcome `timeout` is the time limit's end). The time limit is a function of the observed landing time (§5.8). It is not an input of any model: the executor's laws read it only to end a flight, and a model's inputs come from the states and the words (outline principle 7), which a test on the inputs of the prior checks (the observed samples after the first predicted step changed, no input changed). The executor that the loop holds is public: item 5 lists what a caller reads from it. Why: `Loop.timed_out()` made the time limit readable to a caller (the second review of A32–A36). A private executor would hide nothing in Python and would break the runners of the prior (free generation and the export read the end cycle, the flown record and the aero parameters); a guard belongs at the inputs (the user, 2026-10-05) | Decided | User, 2026-10-05, on the second review of stage A; the scope: user, 2026-10-05 |
| D97 | The public interface gives what a speaker's closed loop in windows of recorded traffic needs; stage A makes the changes, code only: no artefact, executor spec or reading is built again, and the executor conformance (§7.2 #3) does not change. (1) Item 3 names the stored signals of every flight of a split (`instructions/artefact.py` `load_signals`, `signals_flights`): the positions in the airport frame and the MSL heights on the 2 s UTC rows, from the entry of the arrival slice to the last row before the observed threshold crossing. Their track, ground speed and vertical rate are fits that use later rows (as the fields of D82) and are never an input. A caller can replay them as recorded traffic, and reads a flight's row only at that row's time. (2) Item 5: a copy of chosen flights of a `Loop`, repeats permitted: everything that the loop holds of them (the executor's state and its record for the judge, the words said, the grammar's words in force, the go-arounds, the time limits, which stay hidden, D90). A copy flown on with the same words flies what the original flies and gets the same outcome (a test; the states within the bound of (3)). `Loop.halt` is part of item 5. (3) Item 5: a flight's states do not depend on the other flights of its loop (a test on the CPU: a flight alone and in a batch of others): the words, the end and the outcome are exact, the states within `STATE_BOUND_M` (1e-6 m). Measured by A38 (A34's artefact, train, Δ = 4 s, on the CPU): chunks of 2,048 flights and of 1,000 gave every flight the same states bit for bit; 200 flights flown alone and in chunks of 2,048 gave the same words, corrections and outcomes, and states that differ by at most 7.9e-10 m on 65 of them (A30's train read in parts differed from train read whole on 6–7 of about 40,500 flights for each Δ, by at most 9.1e-13 m). (4) Item 5: the start of a closed loop with moved starts: for each flight, a turn about the airport reference by an angle, a change of height and a change of speed, applied to its observed rows up to the first predicted step after the flight is checked against its stored signals (§7.2 #4), so that the start rule (D77) gives the moved state from the moved rows. The start gives the moved observed rows back, because a speaker reads them. A change of speed by the factor 1 + κ stretches the positions and the heights of the observed rows about the first predicted step's row, so that the path angle is kept (the rule of the archived instruction-v3); the time limit stays the flight's own (the user, 2026-10-05). A turn is made on the rows as stored, in the airport frame, so the ground scale of the start rule (D87) moves the start speed slightly (×0.99976 for 10°, about 0.4 % at 90°); this is not corrected (Claude's reading, A38). A move of zero gives the start without a move, bit for bit (a test). Why: a speaker's closed loop in windows of traffic replays recorded flights, copies a loop's state at its branch points and moves starts; without these, its runner would read the internals of `autopilot/`, and D71 keeps the executor parameters away from every caller (milestone A38) | Decided | User, 2026-10-05, on Claude's report of stage C's readiness; (3) measured first, the bound: the user, 2026-10-05; (4) the stretch of the rows and the time limit: the user, 2026-10-05 |
| D102 | Two functions that the post-training reads are part of §6, by name, with no change of code: `instructions/grammar.py` `column_words` (item 2: the order of a column's words, in which a caller gives a speaker its masks) and `instructions/artefact.py` `closed_loop_indices` (item 3: which flights of a split have a closed-loop sentence at a Δ, without reading a sentence). Why: the stage C implementer's code reads them (its proposal P10, `post/speed_mask.py`, `post/scene.py`), and its architecture test admits only the names that §6 lists | Decided | User, 2026-10-05, on the stage C implementer's proposal |
| D111 | A flight whose stored observed track has a faulty point is marked when the artefact is read (§6 item 3); the artefact is not changed and nothing is built again. A point is faulty where a 2 s step is more than 3 times, or less than a third of, the median of the 10 steps around it (5 on each side; a jump, or a position that stops while the aircraft flies on), or where the track turns more than 120° between two consecutive positions (a reversal). The mark is a function of the stored signals alone, by one definition, with the reasons it found. The prior leaves marked flights out of its selection (prior D111). Why: ADS-B faults in the observed tracks (jumps, held positions, reversals) gave A34's dynamics failures (a start rule reading a held position: 7–58 m/s), its 26 km lateral errors and A37's vertex flips; the closed-loop sentences of some such flights land and would be learned from. On A34's artefact at Δ = 4 s, among the flights with a closed-loop sentence: train 498 of 40,530 marked (469 landed), select 177 of 6,199 (157 landed), most of them held positions; the rule marks 11 of train's 14 and 5 of select's 6 dynamics failures (`instructions/faults.py`) | Decided | User, 2026-10-05 (the rule X = 3, the reversal at 120°, the mark read at reading time, the prior's selection) |
| D126 | §6 item 8 lists the functions of stage A's Training export that a later stage's export imports (stage C's window export, post-training C11); the later stage imports exactly these names (`tests/test_architecture.py` `TRAINING_EXPORT_NAMES`). Names only: no code changes. (The part of the prior: D126 there) | Decided | User, 2026-10-05, on stage C's request |

### 0.2 Open items

| # | Item | Proposal | §  |
|---|---|---|---|
| O8 | Speed words in ground speed: wind can make speed words at turns | A check on data, later | 3.6 |

### 0.3 Implementation

The commits, dates, branches, test counts and measurements of every milestone are in
`readouts/2026-10-05_stage_a_implementation_log.md` §1 (the design document holds no log).

| Part | State |
|---|---|
| A0–A42 | Done, each milestone reviewed; merged into `dev-two-tier` (A41 `362234ee`; A42 `06b8fde1`) |
| A43 (D73) | Done: `60bbc901` on `dev-two-tier-v4`, reviewed; merged into `dev-two-tier` with stage B's B12 (`645e4cf0`, 2026-10-05) |
| The formal artefact | `instruction_language/v12_20261005`, `executor/v17_20261005` (A34; read-only, `SHA256SUMS`); its Training sets `closed_loop_v12_20261005`, 5 airports (A35). The superseded `v11_20261004`, `v16_20261004` and `closed_loop_v11_20261004` are deleted |
| The replay of the val days | Not run; it waits for the user (outline D85) |
| Worktrees `two-tier-v4-a32`, `two-tier-v4-a37` | Deleted with their branches on 2026-10-06 (the user); their ignored scratch (`smoke_v4/data/a33`, `a34`, `a37`, `a38`), cited by the A33, A34, A37 and A38 reports, went with them |

### 0.4 Plan

No milestone of stage A is open (§12.1). What is left:

1. A44 (outline D138), before stage C's next campaign.
2. The Training view of stage A changes with the one layout of the three stages' views (outline §6.2, D133–D135): the
   block of a flown sentence becomes one function for every stage (`flown_sentence`, the track unrounded), and
   `closed_loop_v12_20261005` is exported again in a new sample format; built by stage B's implementer.
3. The replay of the val days, the stage's one validation readout (outline D85), waits for the user.

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
the table, each with its rounding error; the fit gives the grid of the table, the user's choice (D55, D58, D59).

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
| Climb | No angle in the word. The executor's angle (D28): the nominal of D15 while G is false (1.25°, D56); 1.885°–3° while G is true | 0.5° to 15° climb (the labeller's range of a climb piece) |

The four descent classes are a k-means on tan(angle) of the descent pieces of the train days, with each piece weighted
by the square of its length (D54): a piece flown at the nominal angle of its class ends length · |tan a − tan c| from
its observed end, so the k-means makes the sum of the squared end-of-piece height errors smallest. The values of the
table are the user's choice (D56): the fit on all train days rounded to 0.25° (fitted 1.51 / 2.45 / 3.09 / 4.45°, edges
1.98 / 2.77 / 3.77°; the climb pieces with G false 1.236°). With D3 the model holds the glidepath with these classes; the closed-loop reading gives the corrections (D32, §4.9).

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
- **G false.** The executor climbs at the nominal angle of the climb class (D54, D56): the length-weighted median of
  the climb pieces with G false of the train days, rounded to 0.25°: 1.25° (fitted 1.236° on 1,076 of the 1,339 climb
  pieces; the 251 with G true are flown at the go-around angle, and the 12 of flights whose vertical reading is refused
  have no known G, D88; neither is in the fit). The train days have 1,339 climb pieces against 227,559 descent pieces
  (0.6 %).

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
- The reading is named `instruction-v6` (`instructions/spec.py` `READING_RULE`). The spec records the grids, the
  classes, the tolerances and the reading name.
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

**Readings (D34).** The closed-loop reading (§4.9) makes each replay follow the observed path, so the landed share does not
show how well a Δ carries a flight. At each Δ the ablation reads instead:

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
   candidate, the dynamics, or the time limit of the replay. The closed-loop sentence has one row for each Δ row of the
   flight: its length is the flown time, not the observed time.

**The comparison.** The matched point is the point of the observed path nearest to the flown position. The search
starts at the matched point of the row before and goes forward, so that a path that crosses itself does not jump. At
the matched point:

- the lateral error e_y is the signed distance of the flown position from the observed path, the line through the
  observed rows near the matched point (its segments, not their extensions), positive to the right of the path; at a
  vertex, the smaller distance to the two segments around it, on the side of the sum of their normals, or of the matched
  segment where the path turns there by more than 170° (D83);
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
| 2 | The grammar | Rules 1–6 and the runway/G table (§3.2, §3.7) as one function: it checks a row (`apply`); for a speaker's mask, it gives the permitted words of a column after the earlier columns of the row: a word is permitted when some words of the later columns, each among the words that the caller permits, make the row pass `apply` (D62) | `instructions/grammar.py` `apply`; `column_words` (the order of a column's words, in which a caller gives the speaker its masks, D102) | The labeller conformance |
| 3 | The sentence artefact | For each split of the day split (the test days sealed, contract C32) and each Δ of the ablation: the closed-loop sentences (the words of each row, each correction word marked) and the flown states on the data's 2 s rows with the Δ rows marked (position in the airport frame, MSL height, track, ground speed, vertical rate; observed before the first predicted step, flown from it; D51; on the observed rows, the track, ground speed and vertical rate by the start rule, D77, the track in [0°, 360°)); for each sentence, apart from its rows (D82), the fields that use later rows or the observed path: its flight, its runway and its index, its landing time, its capture row, its go-around rows, its stratum (§4.3; D70), the outcome of its closed-loop sentence (§5.8; D74), `timed_out`, the matched and observed rows, the errors e_y and e_h and the rows without a correction; the flights without a sentence, by reason. These fields are for readouts, strata and selection, never an input. The stored signals of every flight of a split, with a sentence or without one (D97): the positions in the airport frame and the MSL heights on the 2 s rows, from the entry of the arrival slice to the last row before the observed threshold crossing; their track, ground speed and vertical rate are fits that use later rows, never an input. The check that closed-loop sentences may be read: refused unless the code on disk reads the closed-loop reference as it was read, checked at the call (D69, D73) | `instructions/artefact.py`: the formats, `load_signals` and `signals_flights` (the stored signals, D97), `instructions/faults.py` (`track_faults`, `faulty_flights`, `JUMP_RATIO`, `AROUND_STEPS`, `REVERSAL_DEG`: a flight's marked observed-track faults, D111), `closed_loop_indices` (which flights of a split have a closed-loop sentence at a Δ, no sentence read; D102), `STATE_COLUMNS` (the names of the state columns) and the function that reads a closed-loop file into its sentences (for each: its flight, its first row, its words and correction marks, all its states on the 2 s rows from row 0 with the Δ rows marked) (D61); `instructions/readout.py` `STRATA` (the names of the strata, D70); the check: `autopilot/closed_loop.py` `require_conforming_closed_loop` (D69; it takes the artefact and the executor spec's directory, D73) | The format names; the labeller and the closed-loop conformance checks (§7.2, D73) |
| 4 | The candidates and their geometry | For each airport: E, and its candidates: its runway ends with a published vertical path, decided by no flight (D78); for each candidate: the threshold, the course, the threshold elevation, the length, and its vertical path (D61): the TCH, the glidepath angle and the DA above the threshold (the LPV line's; where a runway publishes no LPV line, KRDU 32 and KSMF 35R, the LNAV/VNAV line's: the reading of the code). The functions: the position, height and direction relative to a candidate (distance before its threshold along its course, offset right of its final, height above its threshold, direction minus its course); the height of its published glidepath at a distance before its threshold with the straight-line reference (§9.3), one function of the airport, the candidate and the distance, the radius of curvature included, which the judge and the prior both call; the lateral limit of a landing passage | `candidates.json` (a new format name, D61); `instructions/airport.py` `VerticalPath`, `relative_to_runway`, the glidepath height, `landing_cross_limit_m` | Part of the artefact (item 3) |
| 5 | The executor | Flies the words of one row at each Δ row, in 1 s cycles, from a given state (§5): one aircraft, a batch, or a batch in which each aircraft starts at its own cycle; it gives the state at each cycle and when the aircraft is done: where the judge ends the flight (D79). It reads only the words, the aircraft and the runway geometry (§5.2); it holds nothing of the landed runway (D81). The start of a closed loop (D67): for closed-loop sentences of the artefact (item 3) at a Δ, with the directory of the executor spec (opened and checked by the start, D71) and the most go-arounds of a flight, the executor at each flight's first predicted step (the flight rebuilt and compared with the stored signals, its start state from the rows at or before that step by the spec's start rule (D77), its aircraft and approach speed, its time limit and the time its go-arounds may add); then, for the words of one row of each flight, checked by the grammar before anything changes (D80), the states of the 2 s rows flown (`STATE_COLUMNS`) and the flights done. The loop's executor (`Loop.executor`) gives a caller the end cycle of each flight (`done_cycle`: where its states end), the flown record (`flown()`: the record the judge reads) and the aero parameters (`inputs.aero_params`: the attitude); the loop gives no time limit (D90). The loop gives a copy of chosen flights, repeats permitted, with everything that it holds of them, and halts chosen flights (`Loop.halt`); a flight's states do not depend on the other flights of its loop; the start takes moved starts (a turn about the airport reference, a change of height and of speed of the observed rows up to the first predicted step; a move of zero is no move; each flight is checked against its stored signals before it is moved, and the start gives each flight's moved observed rows before the first predicted step back) (D97) | `autopilot/executor.py` `Executor` (`take`), `autopilot/single.py`; the start: `autopilot/start.py` (`Start`, opened once in a process: `moved`, `release`, outline D138, A44; start`, `start_moved`, `Move`, `NO_MOVE`, `Loop`: `step`, `halt`, `copy`, `outcome`, `executor`) | The executor spec sha and its conformance check (§7.2, D73) |
| 6 | The judge | The outcome of a flight and its order (§5.8), the DA check, the time limit (the remaining observed time × 1.5, plus 900 s for each go-around), the limits that bound in each cycle. The outcome of a flight flown from the start of a closed loop needs no observed words (D67) | `autopilot/judge.py` (the outcome: `outcome_of`) | With item 5 |
| 7 | The row grid | Rows on UTC multiples of Δ, Δ = 2, 4 or 8 s (D25); the chosen Δ is 4 s (D11); the first predicted step 16 s after row 0 | `instructions/labeller/interval.py` | — |
| 8 | The Training export | The functions of stage A's Training export that a later stage's export calls, so that the formats and blocks of a set have one definition (D126); the block of a flown sentence with its envelopes and the writer of the Training files, for every stage (outline D135) | `experiments/training_export.py` `FORMATS`, `candidate_hae_minus_msl_m`, `candidates_block`, `events`, `flown_sentence` (outline D135, to be built), `split_flights`, `vocabulary_block`; `instructions/training_files.py` the reading and writing of an index and a set (outline D135, to be built); `experiments/training_flights.py` `crossing_payload`, `last_state_cycle`; `experiments/training_attitude.py` `attitude_payload`, `executor_attitude` | — |

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
| 2 | The labeller, open-loop and closed-loop reading | The labeller conformance: the artefact holds a reference sample (`conformance/`; train, seed 1337, 50 labelled and 10 refused flights for each airport and up to 10 train flights with a labelled go-around for each airport, with their signals); `instruction_conformance` labels it again with the code on disk and requires the same words, the same strata (D70) and the same refusals, on the 2 s rows and on the Δ grid at Δ = 4 and 8 s (D49). The closed-loop reading (D32) has its own reference sample (train, seed 1337, 10 flights for each airport and up to 10 train flights with a labelled go-around for each airport, at each Δ of the artefact): with the artefact's executor spec, the same correction words and the same flown states on the 2 s rows (D51), within the tolerance of the executor conformance. Each check runs where it is needed, every time (D73): a runner that labels, reads the closed loop or replays sentences runs the checks of the code it uses before its work and refuses on a difference (the closed loop's: `require_conforming_closed_loop`, D69). No digest of code. The artefact records the commit that wrote it, as information |
| 3 | The executor | The sha of the spec's parameters and the conformance of its reference tracks (C33): the spec's labelled train flights, with the train flights that have a labelled go-around (up to 10 for each airport, so that the go-around climb and the held airspeed are flown), flown again in every way the executor flies (single-aircraft batch, multi-aircraft batch, single flight) within 1e-6 m (`STATE_BOUND_M`), by each runner that opens the spec, before its work (D73); and the behaviour check of D81: the same tracks with every candidate's vertical path changed and the frame's origin moved. The open-loop sentences of these flights are said on their own rows (D57) |
| 4 | The flights of an artefact | The stored signals. A consumer that rebuilds a flight from the harvest compares it row by row with them (`autopilot/flights.py` `require_same_flight`). No byte hash of an arrival manifest |
| 5 | The day split and the sealed test days (C32) | The day split file. It is a data rule |

---

## 8 Values

| Item | Value | Source |
|---|---|---|
| Row | Δ = 4 s, the user's choice, on UTC multiples of 4 s; the artefact also keeps 2 and 8 s (the ablation); the data's rows are 2 s, on even UTC seconds | D11, D25 |
| First predicted step | 16 s after row 0 (8, 4, 2 rows at Δ = 2, 4, 8 s) | D25 |
| Start rule | `displacement-2s`: the start state's velocity from the displacement in the 2 s before the row (row 0: rows 0 and 1); the same for every start of the executor and for the stored observed rows | D77 |
| Executor cycle | 1 s | Fixed choice |
| Heading grid | 5°, relative to the course of R | Spec (grid); D8 (frame) |
| Heading lead L | 4 s | Measured (`instruction_vocabulary_design.zh.md` §10.1) |
| Heading tolerance | 4.5° | Half grid 2.5° + 2° |
| Turn-rate limit | 4.7°/s | Spec, measured (p99.9) |
| Bank limit | 32° | Spec, measured (p99.9) |
| Roll rate p | 5°/s | FAA Order 8260.3G Appendix E §4 ¶6.a ("roll-in rates of up to five degrees per second") |
| Altitude reference | The airport elevation E; a level T is T + E MSL | D58 |
| Level grid | 60 m to 1,260 m, 120 m to 2,700 m, 450 m to 5,400 m above E; 40 levels; the fit on the level-offs above E gives this grid, the user's choice | D22 (fit of the altitude-grid proposal, on MSL), D58, D59 |
| Level envelope ε | half the larger gap to the neighbouring levels + 10 m: 40 m (0–1,200 m), 70 m (1,260–2,580 m), 235 m (2,700–5,400 m); "no level-off" 40 m | D22, D52 |
| Level detection | ≥ 20 s, rows within 25 m of the piece's median | Labeller constant (§4.4) |
| Descent classes | edges −0.5 / 2.0 / 2.75 / 3.75 / 10°; nominal 1.5 / 2.5 / 3.0 / 4.5° | Spec; k-means on the train days, weight length² (D54), rounded to 0.25° (D56) |
| Climb word | One class; a climb piece is 0.5–15° (labeller); executor angle while G is false: 1.25°, the length-weighted median of the climb pieces with G false, rounded to 0.25° | Spec; D15, D28, D54, D56 |
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
| Word clock | `time`: every sentence is said on its own rows; the executor spec has no clock field | D57 |

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
in the 450 m segment. These are MSL heights. On the heights above the airport elevation
(D58; A20, code at `220e858e`, 25,727 level words; the formal artefact's measurement gives the same), the exact fit of §3.4 gives the grid of D22 itself, 17 / 34 / 190 m;
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

The code of this document (stage A). Line numbers are at `60bbc901` (A43) on `dev-two-tier-v4`; paths are relative to
`4dTrajectory/ts_transformer/` unless marked "repository root".

| What | Where |
|---|---|
| Columns; "go-around" in the runway column | `instructions/words.py:22` `COLUMNS`, `:26` `RUNWAY_GO_AROUND` |
| Heading classes relative to the course of R; one test of "the same track" (D46, D48) | `instructions/words.py:79` `Words.heading_class`, `:83` `Words.heading_track_deg`, `:187` `same_track` |
| The 40 levels, their bands ε (D52) | `instructions/words.py:46` `altitude_tolerances`, `:88` `Words.altitude_index`, `:107` `Words.altitude_tolerance_m` |
| The vocabulary spec; the reading name | `instructions/spec.py:28` `READING_RULE`, `:44` `VocabularySpec` |
| Grammar: rules 1–6 and the runway/G table written once; a row's check; a speaker's column mask (D62, D102) | `instructions/grammar.py:166` `_rules`, `:210` `apply`, `:246` `column_words`, `:255` `column_mask` |
| Approaches; capture row per approach; heading words relative to the R of each row | `instructions/labeller/lateral.py:33` `Approach`, `:54` `capture_row`, `:72` `per_step_words`, `:106` `read_lateral` |
| Level test by the piece's own median; held height above E; the go-around climb and the final descent before it (D26); tubes | `instructions/labeller/vertical.py:69` `vertical_pieces`, `:90` `held_height`, `:126` `climb_after`, `:137` `read_vertical`, `:240` `tube_bounds`, `:271` `tube_checks` |
| Speed words per approach; "unspecified" from the capture row | `instructions/labeller/speed.py:95` `read_speed` |
| Go-around rule (R40's, on the rows) | `instructions/labeller/go_around.py:62` `low_passes`, `:109` `go_arounds` |
| Landing cut, go-arounds in the gate, approaches, runway words (D19, D26) | `instructions/labeller/read.py:59` `ApproachReading`, `:127` `admit`, `:169` `flight_go_arounds`, `:186` `runway_again_rows`, `:220` `read_flight` |
| The 2 s sentence | `instructions/labeller/sentence.py:24` `assemble` |
| Row interval; the Δ rows of the 2 s rows; a heading word across a runway change (D11, D25, D46, D51) | `instructions/labeller/interval.py:59` `first_interval_row`, `:68` `on_interval_rows`, `:95` `on_utc_grid`, `:103` `on_interval` |
| Labeller conformance, on the 2 s rows and the Δ grids (§7.2 #2, D49) | `instructions/conformance.py:51` `INTERVALS_S`, `:65` `sentences_of`, `:177` `check`, `:219` `require_conforming_labeller` |
| Strata (D70); the words in force at each go-around row (readout) | `instructions/readout.py:22` `STRATA`, `:27` `stratum`, `:33` `go_around_in_force` |
| Heading envelope | `instructions/envelope.py:31` `heading_words_inside` |
| Each candidate's vertical path; the position relative to a candidate; the published glidepath's height (D61) | `instructions/airport.py:35` `VerticalPath`, `:119` `vertical_path`, `:138` `airport_geometry`, `:169` `landing_cross_limit_m`, `:206` `relative_to_runway`, `:227` `published_glidepath_height_m`, `:238` `glidepath_height_m`; `instructions/artefact.py:53` `CANDIDATES_SCHEMA`, `:138` `load_candidates` |
| The splits read and sealed (D85); the stored signals (D97); the closed-loop sentences on disk and their public reader (D61, D74, D82, D102) | `instructions/artefact.py:59` `SEALED_READINGS`, `:60` `READ_SPLITS`, `:109` `load_signals`, `:120` `signals_flights`, `:255` `CLOSED_LOOP_SCHEMA`, `:271` `STATE_COLUMNS`, `:322` `ClosedLoopSentence`, `:331` `write_closed_loop`, `:370` `load_closed_loop`, `:382` `closed_loop_indices`, `:396` `closed_loop_sentences` |
| Faults of an observed track (D111) | `instructions/faults.py:29` `JUMP_RATIO`, `:31` `AROUND_STEPS`, `:33` `REVERSAL_DEG`, `:45` `track_faults`, `:67` `faulty_flights` |
| Heading law; conversion with the course of R ("go-around" changes no target, D27) | `autopilot/lateral.py:53` `word_rate`, `:140` `Lateral.word_error`, `:167` `Lateral.rate` |
| Vertical modes; the go-around angle (D28) | `autopilot/vertical.py:50` `GO_AROUND_MIN_RAD`, `:55` `go_around_angle_rad`, `:89` `Vertical.rate` |
| "Unspecified" under G holds the go-around row's airspeed | `autopilot/speed.py:73` `Speed.hear_go_around`, `:78` `Speed.rate` |
| The cycle; go-around time; the end of a flight where the judge ends it (D79); a copy of chosen flights | `autopilot/executor.py:61` `GO_AROUND_EXTRA_S`, `:81` `Executor`, `:145` `Executor.take`, `:198` `Executor.cycle` |
| The single-flight executor (the same laws) | `autopilot/single.py:425` `SingleExecutor`, `:556` `fly` |
| The start rules (D77); a flight rebuilt and compared with its stored signals (§7.2 #4) | `autopilot/params.py:34` `START_RULES`, `:41` `ExecutorParams`; `autopilot/flights.py:94` `require_same_flight`, `:176` `start_state` |
| R and G per row | `autopilot/sentence.py:62` `_filled` |
| Outcomes, their order; the DA check (D38) on each candidate's vertical path (D61); no crossing is an event under G (D33) | `autopilot/judge.py:78` `OUTCOMES`, `:81` `EVENT_ORDER`, `:130` `decision_check` (D38: `evaluation/thresholds.py` `RNAV_TERMINAL_VERTICAL_BOUND_M`, repository root), `:153` `_outcome`, `:282` `outcome_of`, `:295` `judge` |
| Opening a spec, with the checks or without (D71, D73); a sentence on Δ; the flights drawn; the readout | `autopilot/replay.py:77` `sentence_on_interval`, `:87` `instructions_of`, `:143` `open_executor`, `:167` `open_spec`, `:179` `group_of`, `:224` `draw_flights`, `:299` `batch_of`, `:443` `summary`; `experiments/executor_replay.py:123` `readout_table` |
| Closed-loop reading (D32): the matched point, the rows without a correction (D34), the rule of D50, the corrections (D53), the flight, its conformance, its replay | `autopilot/closed_loop.py:131` `ObservedPath`, `:220` `uncorrected_m`, `:238` `observed_tracks`, `:250` `outside_rows`, `:303` `Corrector`, `:328` `Corrector.row`, `:438` `read`, `:537` `read_chunked`, `:633` `check`, `:686` `require_conforming_closed_loop`, `:707` `replay_batch`; `experiments/instruction_closed_loop.py:142` `summarise`, `:247` `main`; `experiments/executor_replay.py:290` `closed_loop_columns` |
| The start of a closed loop (D67, D80, D90, D97) | `autopilot/start.py:64` `Move`, `:80` `NO_MOVE`, `:133` `Loop`, `:172` `Loop.step`, `:215` `Loop.halt`, `:221` `Loop.copy`, `:239` `Loop.outcome`, `:282` `start`, `:293` `start_moved` |
| Executor conformance (§7.2 #3); the behaviour check of D81 | `autopilot/conformance.py:80` `STATE_BOUND_M`, `:87` `MOVED_HORIZONTAL_BOUND_M`, `:202` `MOVED_PATH`, `:203` `MOVED_ORIGIN`, `:216` `fly_moved`, `:239` `MODES`, `:488` `check`, `:526` `require_conforming_executor` |
| The backend's check of each answer (D73) | `aeroviz_backend/autopilot_segment/fly.py:106` `refuse_past_bound`, `:117` `apart_from_stored`; `errors.py:19` `ExecutorDiffers` (repository root) |
| FAS cone; DA above the threshold | `flight_scenarios/fas_geometry.py:46` `fas_course_geometry`; `trajectory_data_process/harvest/airports.py:169` `decision_height_above_threshold_m` (repository root) |

---

## 12 Implementation plan

The rules of the implementation are in the outline (§5 there).

### 12.1 Stage A: vocabulary, labeller, identities, executor, judge, replay

A0–A43 are done. What each milestone built, its specification, its commits and its report are in
`readouts/2026-10-05_stage_a_implementation_log.md` (§1 the state and the commits, §3 the milestones); the rules they
build are §3–§8 and the decisions of §0.1. A new milestone is specified here before it is built, and its specification
moves to the log when it is done (outline §5 rule 10). Open: A44.

**A44. The start opened once** (outline D138; built and reviewed, `3b535db6`), by stage B's implementer (outline §5 rule 1: one implementer for all
stages) on `dev-two-tier-v4`, before stage C's next campaign.

- `autopilot/start.py`: an object `Start`, opened for one artefact, split, Δ and executor directory, holds what every
  call of the start reads today: the opened executor (`replay.open_executor`, whose checks run once a process, D73),
  the candidates, the split's signals by signal index (`load_signals`), the labelled sentences' offsets. Its call with
  the sentences and the moves (`Start.moved`; `start_moved`'s other arguments) does only the work of those flights, and
  each flight's series is rebuilt once and kept until `Start.release` (the second pass of a window reuses the first's).
  The arrival records are not held (outline D139 (7)): holding them would change the harvest's loader, which training
  reads; a flight's first start still reads its airport's manifests, about 1 % of a batch.
- `start` and `start_moved` stay: they open a `Start` and call it, so the one-call form stays the readable reference.
  Vocabulary §6 item 5 names `Start`.
- Tests: `Start.moved` gives `start_moved`'s loop, order and observed rows bit for bit (no move, and a moved start, on
  the synthetic artefact); a second call reads no file (the readers counted); a `Start` opened before a fork serves the
  forked process.
- Size: about 100 lines of `start.py` and `flights.py`, about 80 of tests. Expected: a quarter to a third of a speaking
  batch of stage C (outline D138), measured by stage C after C13.

### 12.2 Claude's check of stage A

Items 1–8 are done (the log §4; item 8, A32–A40: `readouts/2026-10-05_stage_a_check_a32_a40.zh.md`). Left from them:
the replay of the val days, which waits for the user (§0.4).
