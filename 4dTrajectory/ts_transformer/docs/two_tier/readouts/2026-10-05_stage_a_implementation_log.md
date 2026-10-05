# Stage A: implementation log (vocabulary document §0.3, §0.4, §12.1 A18–A37, §12.2)

Moved verbatim from `design/vocabulary.md` on 2026-10-05 (branch `docs-vocabulary-slim`): the design document keeps
the rules, and this file keeps the state, the commits, the milestone specifications and the checks of stage A.
Milestone numbers (A18…A37) and decision numbers (D…) are the same as before. Append new state here, not in the
design document.

---

## 1 Implementation state (was §0.3)

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
| A32 (D77–D87) | Done on `dev-two-tier-v4-a32` (worktree `.claude/worktrees/two-tier-v4-a32`): `b7790be0` + `37d632eb`, on `dev-two-tier-v4` `354ff90d` (A36 inside); `dev-two-tier-v4` fast-forwarded to `37d632eb` on the user's go (2026-10-05); A33 onward on the same branch, `dev-two-tier-v4` fast-forwarded after each milestone on the user's go; nothing built. D77 start rules (`autopilot/params.py` `START_RULES`, `flights.start_state` / `observed_rows`; executor spec `ts-executor-spec-v10`, `executor_spec --start-rule`, the centred fit refused by `write_spec` and `load_spec`); D79 `autopilot/ends.py` (one definition for the judge, the batch, the staggered batch and `single.py`); D80 `Loop.step` (`RowRefused`); D81 the chart at the airport reference and the conformance's way `moved` (reference `ts-executor-conformance-reference-v5`; the name scan in `test_architecture.py` removed); D82 `ClosedLoopSentence(rows, withheld)` read by `closed_loop_sentences(artefact, split, Δ, spec)` (closed-loop format `ts-instruction-closed-loop-v8`); D83, D84; D78 candidates `ts-instruction-candidates-v4`; D85 the closed loop's summary; D86 the export (sample `aeroviz-training-sample-v10` on both sides); the references with up to 10 train flights with a labelled go-around an airport (labeller reference v6, closed-loop reference v8); the C32 checks; `keep_spec`; "unspecified" past an approach refused by name; A23's code through `start.require_startable` and `single.fly`; the backend keeps a failed check (a `ValueError`) and refuses val; `check_live --flights-per-airport`. Reviewed by two independent reviewers (executor core: no behaviour change beyond D77–D84, the three ways to fly within 1.1e-13 m on a cross-check; reader, runners, export: one high point, below, and medium points fixed: the kind read off the closed-loop words, memory, the backend's kept errors), fixes re-reviewed; full suite at the commit's tree: ts 1,548 + 1 skipped, modeling + backend 1,006, aeroviz-4d python 161, vitest 691. **The user's choices** (2026-10-05, asked by Claude): D77's window cut at row 0 (row 0 takes rows 0 and 1) and one start state for every start of the executor (above); no former rule kept in the code: `--list-candidates` (`37d632eb`, reviewed) lists each airport's candidates and the ends with arrivals that are no candidate (their flights refused at the build), each with its arrivals by split, and records the runway data and day split it read; the comparison with `v11_20261004` (whose `candidates.json` is a v3 file the code refuses) is a one-off of A33's report, from the former rule taken again on the same manifests — **the design text of A32 ("the ends added and removed against a given artefact") and A33 step 3 ("against `v11_20261004`") is the design author's to change**; on the live harvest the listing (run by the reviewer) counts 72,247 arrivals, every one on a candidate; Claude checked read-only that `v11_20261004`'s candidates equal the manifests' `runway_targets` at all five airports (the manifests unchanged since 2026-09-23) and that D78's candidates equal them too: on the live harvest D78 adds and removes no end (KSTL 06 stays, by its published vertical path). **Claude's readings, decided by the user as D87 (2026-10-05)**: a rule's slope is turned from the airport frame's metres into metres on the ground (WGS84 radii at the row); a halted or done flight's row is not read by the grammar but is stored (the closed-loop reading gives a refused flight's held words at its first row); a flight's kind in the export is whether its closed-loop sentence at the first Δ says a go-around (as A23's replay rows had it); of val, the summary keeps the flights drawn and the sentences written; the backend keeps a `ValueError` raised by the checks (a malformed reference included) until a restart. A defect outside the milestone, recorded in `docs/code-health-followups.md`: two runway ends on one centreline halve the landing screen to float noise (no such pair at the five airports). **For stage B after the merge** (outline §5 rule 1): `prior_training_export.py` calls `split_flights(..., formal_rows(...))` and imports `formal_rows` (now `split_flights(instructions, split, chosen, (Δ,), params, words)`), `prior_free_generation.py` calls `closed_loop_sentences(load_closed_loop(...))` (now `closed_loop_sentences(artefact, split, Δ, spec)`, the sentence's fields under `.rows` and `.withheld`), and its backend calls `closed_loop_batch(drawn, stored, Δ, words)` (now with the artefact, split and params) |
| A33 (D55, D77, D78, D54) | In work on `dev-two-tier-v4-a32`: `84af1840` — R55 `start_rules` (`experiments/start_rules.py`: every labelled train flight read in closed loop at one Δ in memory once per start rule, the spec's parameters but for the rule; per rule the readings of D34, the outcomes by airport, and on the flights turning at the first predicted step the start track against the observed direction of the 8 s after; train in parts, the same readout whatever the parts; `closed_loop.first_predicted_rows` shared with the start); the climb nominal fitted on the climb pieces with G false (G's rows as the labeller reads them, `read.read_heights`) and at most the climb class's 15°, the others counted (`measure.climb_pieces`); `flights.ground_scale` out of `start_velocity`. Reviewed (8 points: G known false in a flight without a go-around, tests that could not fail, the parts test, the guard on the labelled rows, an empty fit refused; fixed, re-reviewed clean); full suite: ts 1,552 + 1 skipped, modeling + backend 1,006, aeroviz-4d python 161. **Claude's readings, decided by the user as D88 (2026-10-05)**: the outcome of each rule is the closed-loop reading's own (D74: equal to the replay's on every replayed flight), so R55 flies no replay; a climb piece is G-true when G is true at its first row; in a flight with a go-around whose vertical reading the labeller refuses, G is not known and its climbs are left out (counted); G is read under the provisional spec's altitude grid (D22's) in the measurement's first pass. `58fd8a2b` (the user's choice, 2026-10-05): the first scratch executor spec failed its own check in the way `moved` (D81) on real data — 11 of 285 reference flights 1.06–2.16e-6 m apart horizontally, every limit, mode and end cycle the same; the vertical paths changed alone fly the states bit for bit, the origin moved 15 m up alone 2.5e-8 m, moved sideways 1.7–2.2e-6 m — so `moved` has its own horizontal bound, 1e-4 m (`conformance.STATE_BOUNDS_M`, reference `ts-executor-conformance-reference-v6`); vertically it stays 1e-6 m (Claude's reading, decided by the user as D88); reviewed twice, full suite at `58fd8a2b`: ts 1,553 + 1 skipped, modeling + backend 1,006, aeroviz-4d python 161. **The measurements** (scratch `.claude/worktrees/two-tier-v4-a32/smoke_v4/data/a33/`, `SHA256SUMS`, results read-only; the steps of A30 with `--candidate 0.25 --grid fitted --closed-loop-final-vertical-m 10`, 09:02–09:06Z; executor spec `displacement-2s`; R55 at Δ = 4 s on every train day, 40,555 flights drawn, 40,530 sentences, 8 parts on 4 workers, 14.5 min): the start track on the 2,998 flights (7.4 %) turning at the first predicted step, p50 from the observed direction of the 8 s after: displacement-2s 5.9°, trailing-fit-8s 8.8°, trailing-fit-15s 11.5° (centred-fit-15s 4.5°); landed 97.76 / 97.78 / 97.73 % (97.81 %); largest |e_y| p50 58.2 / 61.1 / 63.3 m (57.7 m); dynamics failures 14 / 3 / 1 (0) — Claude's reading: the 2 s displacement reads ADS-B's altitude steps (start path angle up to 20.5° against at most 8.4°, a one-off probe on 4,000 train flights). D78 on the live harvest: no end added or removed against `v11_20261004` at the five airports, geometry and vertical paths equal; no end with arrivals is no candidate. The climb nominal refit on 1,076 of 1,339 climb pieces (251 with G true, 12 in refused readings): 1.236°, rounded to 0.25° 1.25° (D56: 1.5°) — a change for the user (D15). Report `readouts/2026-10-05_stage_a_a33_report.zh.md`. **The user's choices (2026-10-05, on this report): the start rule `displacement-2s`; the climb nominal the 0.25 row (1.25° on this measurement; A34 measures it again with `--candidate 0.25`)** — not yet in the design text (D77, D56 and §8 are the design author's to write). A33 is done; `dev-two-tier-v4` fast-forwarded to `58fd8a2b` on the user's go (2026-10-05) |
| A34 (D77–D87) | Started 2026-10-05 09:48Z from the clean checkout of `dev-two-tier-v4-a32` `58fd8a2b` (`.claude/worktrees/two-tier-v4-a32`, script `smoke_v4/data/a34/run.sh`): the steps of A30 into `instruction_language/v12_20261005` and `executor/v17_20261005` (spec `--candidate 0.25 --grid fitted --closed-loop-final-vertical-m 10`, executor spec `--start-rule displacement-2s`, the closed loop at Δ = 2, 4, 8 s with `--workers 4 --train-parts 8`, the replays of train (400 an airport) and select at each Δ, the start check), read-only with `SHA256SUMS` **Done 10:43Z** (55 min): the labeller conformance (336 flights, 0 read otherwise), the executor spec's check (batch / staggered 0 m, single 2.1e-8 m, moved 2.2e-6 m within its own bound), the closed loop (train Δ = 4 s equal to R55's `displacement-2s` reading: 40,530 sentences, the same corrections and outcomes), the six replays, the start check (250 train flights at each Δ, 0 failed, 0 m); both directories read-only with `SHA256SUMS` (29 and 73,853 files, 2.3 and 4.4 GB). Against A30 (train and select only, D85; one-off `smoke_v4/data/a34/compare_a30.py`): candidates and signals equal; the spec differs only in the climb nominal (1.5° → 1.25°); the landed share of the closed loop train 97.95 → 97.89 / 97.81 → 97.76 / 97.07 → 97.02 %, select 97.37 → 97.23 / 97.19 → 97.08 / 96.72 → 96.43 % at Δ = 2 / 4 / 8 s; why each sentence changed (start, end, e_y, climb nominal) is being read (`smoke_v4/data/a34/attribute.py`, two readings in memory with A34's code, one choice put back each); the report follows |
| A37 (D80, D83, D85, D90) | On the user's word (2026-10-05): branch `dev-two-tier-v4-a37` from `dev-two-tier-v4` `58fd8a2b`, worktree `.claude/worktrees/two-tier-v4-a37`. **`e7e01461`**, reviewed twice by an independent reviewer: D85 — one definition `artefact.SEALED_READINGS` / `READ_SPLITS` (`training_files.SPLITS` is it), the labeller's readout and printed text give val's counts only (`instruction_labels.shown`), `executor_replay`, `closed_loop_start_check` and `final_descent_tolerance` take `--split` train or select; D80 — `Loop.step` checks the values of every flight's row, a done or halted one's too (`grammar.require_values`); D90 — `Loop`'s executor private, no `timed_out()`, `captured()` for the closed-loop reading, `timed_out` from the judge's outcome (`judge.TIMEOUT`, the one literal); D83 — `ObservedPath._at_vertex`, from `match` at t = 0 and at t = 1 on a segment not the last: the smaller distance to the two segments, the other segment's side where its own nearest point is inside it (decided by its clamped t), else the side of the sum of their normals, the matched segment's where they cancel (**Claude's reading**: the sum of the normals alone gives the wrong side beside the segment before the vertex, inside a turn; the reviews added turns over 90° and the t = 1 case, both tested); the export's HAE − MSL of each candidate from the published runway data, checked by `signals.json`'s sha256 (equal to the manifests' on every runway they hold; `training_files.runway_hae_minus_msl_m` removed); the backend refuses a split other than train and select before any check; a 360° track stored as 0° (`flights.compass_track`; the executor's own angles unchanged); the labeller reference's day check before its arrays; R55's parts claim up to rounding. Tests for each (the reviewers ran them on `58fd8a2b`: those of new behaviour fail there; the D77 channels, the references' go-around draw, the field names, the export's draw order, the D81 frame law and `check_live.compare` pin behaviour that held). Each touched test file passes; the full suite waits for the machine (A34's attribution and stage B's work are running, rule 13). **For the user** (reported, not changed): stage B's code on `dev-two-tier-v4-prior` reads `loop.executor.time_limit_s`, `loop.timed_out()`, `executor.done_cycle`, `executor.flown()`, `executor.inputs.aero_params` and `training_files.runway_hae_minus_msl_m` (`prior_free_generation.py`, `prior_training_export.py`, their tests): after the merge it must follow, and the public interface has no accessor yet for a flight's end row, its flown record or its aero parameters; `instruction_figures` (val pages) and `executor_turns` (`--split` val) still read val, outside A37's list. **`b41b24a5`** (D90 narrowed by the user, 2026-10-05; reviewed twice): the rename to `_executor` undone, `Loop.executor` public; `Loop.timed_out()` stays removed, `timed_out` from the judge's outcome; `captured()` removed (it came only with the rename: the closed-loop reading reads `loop.executor.vertical.captured` again, as before `e7e01461`); a test that a caller reads the end cycle (`done_cycle`), the flown record (`flown()`, equal to the 2 s rows `step` returned) and the aero parameters (`inputs.aero_params`) from `Loop.executor`, and that `Loop` has no `timed_out()`. **The check that nothing changes on A34's artefact (`e7e01461`; `smoke_v4/data/a37/no_change.py`, `no_change.json`, read-only): DIFFERENT** — every split and Δ read again in memory in the build's parts and chunks; the three conformance checks pass; 5 of about 170,000 sentences differ (train 2 s and 4 s, select 2 s and 8 s, val 2 s: 1 each; the other cells equal), each with another e_y; stopped as ordered, no rebuild, v12's readout not written again. **The cause, measured** (`smoke_v4/data/a37/vertex.py`, `vertex.json`, read-only; nothing flown: e_y recomputed on the stored states by A34's `ObservedPath` at `58fd8a2b` and by A37's): A34's reproduces every stored e_y of every split and Δ; A37's differs on 3 flights only — KSMF SWA1414 17R 09-05 (train 2 s and 4 s, the last rows), KSTL DAL1400 30R 08-26 (select 2 s from Δ row 40, 8 s from row 10), KSJC CPJ007 30L 08-30 (val 2 s, row 106) — each at a vertex where the observed path nearly reverses (turn 176.8°–178.6°), both segments nearest at the vertex: A37's sum of the normals (which nearly cancel, |sum| ≈ 0.03) gives the opposite side to A34's matched segment, the same distance, so e_y changes sign and the correction word with it (train 4 s: one word at the end, the states unchanged; select: the flight after the word, up to 49 km apart). **For the user**: where a near-reversal counts as a reversal (A37: only |sum of the normals| < 1e-9) is Claude's reading of D83; and whether A34's artefact is built again. **The user's answer (2026-10-05)**: the faulty points stay an open item (repo `docs/open-items.md`: 53 train and 21 select flights turn over 120° between two 2 s rows, counted over all signals), the rule changes, no full check again — a sample. **`89729d66`** (reviewed twice): a turn above `REVERSAL_TURN_DEG` = 170° (**Claude's value**) at a vertex is a reversal, the matched segment's side; below it bit for bit as `b41b24a5` (the reviewer: 2.71 M cases, no bit apart); tests 1° either side of it and at 175°, left and right. The full suite passed on `b41b24a5` (1563 + 1008 + 161, 1 skipped); `test_closed_loop.py` and `test_executor_turns.py` on `89729d66`. **The sample check** (`smoke_v4/data/a37/sample_check.py`, `sample_check_*.json`, read-only): the 3 flights at every Δ give A34's stored sentences back (words, corrections, outcome, e_y, e_h); 20 other flights a split at every Δ the same words, corrections and outcomes, errors equal on train and select, on val 2 flights apart by ≤ 5.6e-10 m (a batch of 21, not the build's chunks: D97 (3)). **v12's `readout.json` and `readout.md` written again by A37's code** (val counts only; train and select rebuilt equal to the stored before writing) and `SHA256SUMS` (29 files, all verified after; the old three files' sha256 in `smoke_v4/data/a37/readout_before.sha256`). Next: A35, then A38 |

## 2 Plan (was §0.4)


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
   6. A37: the corrections of the second review of A32–A36 (§12.1), code and tests only; no artefact is built again.
      It runs beside A34 and A35 (the user, 2026-10-05).
   7. Claude's check of A32–A37 (§12.2, item 8).
3. D86 (the export reads no formal replay row, so it can give val flights) goes into A32's code.
4. The replay of the val days waits for the user.

## 3 Milestones A18–A37 (was §12.1)

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
  candidate is refused by name and counted. `--list-candidates` writes, for each airport, its candidates and the runway
  ends with arrivals that are no candidate, each with its arrivals by split, and the runway data and the day split that
  it read; it builds nothing. The code keeps no former rule (the user, 2026-10-05). A new candidates format name.
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
3. The candidates of D78: the listing of `--list-candidates` on the live harvest (the test days only counted, from the
   roster, C32). The comparison with the candidates of `v11_20261004` (the former rule applied again to the same
   manifests) is a one-off of the report, outside the code.
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

**A35.** Not done: its specification is in the design document (vocabulary §12.1) until it is done.

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

**A37.** Not done: its specification is in the design document (vocabulary §12.1) until it is done.

## 4 Claude's check of stage A (was §12.2)

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
Item 8 is not done: it is in the design document (vocabulary §12.2) until it is done.
