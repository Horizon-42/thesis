# Stage B: implementation log (prior document §0.3, §0.4 and the specifications B0–B4, B8)

Moved verbatim from `design/prior.md` on 2026-10-05 (branch `docs-prior-slim`), as stage A's log was moved from
`design/vocabulary.md` (outline §5 rule 10). The design document keeps the rules, the milestones that are not done
(B5, B6, B7, B9) and a status table. In the text below, "§12" means §2 of this file for the specifications of B0–B4
and B8 (the design document's §12 no longer holds them); "§0.3" means §1 of this file. Append new state here, not in
the design document.

---

## 1 Implementation state (was §0.3)

The implementer's log (outline §5 rule 10). Branch `dev-two-tier-v4-prior`, worktree `.claude/worktrees/two-tier-v4-prior`.
A proposal is a reading where the design says nothing; it holds only until the user decides.

| Part | State |
|---|---|
| B0: the package `prior/`; its import rules in `tests/test_architecture.py` (`prior/` reads only `instructions/`, the day split and the plain utilities; only the runners import it) | Done, `f3070978` |
| B2: the model (`prior/model.py`), a sentence's rows and their batch (`prior/batch.py`), the checkpoint (`prior/checkpoint.py`; `ts-prior-checkpoint-v7` since B8, whose identity holds the selection); `no_motion` (D60) | Done on synthetic sentences, `f3070978`, `de7d4994`. Full ts suite at `f3070978`: 1,559 passed |
| B3: the training loop (`prior/train.py`), the data of a run or a fold (`prior/runs.py`), the runner `prior_train` | The loop done on synthetic sentences, `f3070978`. The runner (one run or one fold; the memory check of the largest batches before training; `--sample` a smoke run, D55) on synthetic artefacts, `849e9cde` (reviewed); `--memory-check-only`, `5a214031`. The smoke on the formal artefact `v9_20261004` at Δ = 2 s (`98a5a9aa`, configuration A, 2.17 M parameters, variant `full`, all five airports, 200 sentences of each airport and split, seed 1337, 3 epochs, in the scratchpad): 1.8–1.9 s an epoch for 1,000 train and 1,000 select sentences; the select loss 18.74, 17.43, 15.41 per step (still in the warm-up); loading the train and select splits approximately 1 min 50 s. The memory check at the formal size (40,534 train sentences, 7.56 M rows; the longest 770 rows): the largest batches 128 × 128 rows and 8 × 770 rows, 8 candidates; GPU 1.86 GB reserved (7.55 GB free); host 3.9 GB. Claude's estimate from the smoke: approximately 55 s an epoch on all five airports, 45 s on a fold; at most approximately 25 min a run of 30 epochs |
| B1: the inputs of a row (`prior/inputs.py`: `state_inputs`, `Heard`; a sentence and a loop use both), the landings (`prior/landings.py`), the artefact as sentences and the identity of the data (`prior/source.py`) | Done on synthetic artefacts, `278b626b`; the landings digest as D63, `ff514325` |
| B4: the speaker (`prior/speaker.py`), the procedure masks (`prior/procedure.py`, set `procedure-masks-v4`) | Done on synthetic inputs, `278b626b`; the finals read on KRDU's CIFP. The masks and the glidepath scale as D64 and D65, with B4's tests of D64, `07f3f49b` (reviewed). The bound of D68 (the speaker counts each aircraft's go-arounds; `go_around_bound`, the caller's mask after the second), `26d05dab`. The architecture test of D69 (a runner of the prior takes from `autopilot/` only the modules of items 5 and 6 and `require_conforming_closed_loop`, by the names imported), `0a04331e`. Free generation's closed loop (through the start of A26: the observed rows, then each Δ row's inputs from the flown states through the prior's one input function — a row equal bit for bit to the sentence's — the speaker under D68's bound, the judge's outcome) and its readout by airport and stratum (D70), on synthetic artefacts, `47ece60f` (reviewed). The runner's `main` (the draw, the samples, the refusals, the files written; the checks it ran recorded, D73) through the start of D71 and the closed-loop check with the executor spec's directory (D69, A28 and A29 merged), `12127353` (reviewed); `prior_train` gives the check the executor spec, `12127353`. A smoke on real data waits for A30's artefact: A29's format names refuse A25's `v10_20261004`; free generation on the formal artefact waits for A25's artefact (the sentence files of A27's format) |
| D73 (no code fingerprint) | Stage B records no digest of code: its digests are of data (the spec sha, the sentence files' and the checkpoint's sha256, the landings, the CIFP documents). After A29 is on this branch, `prior_train` and the free-generation runner give `require_conforming_closed_loop` the executor spec's directory (B3, B4); the formal runs read A30's artefact, the smoke runs A25's |
| B8: the selection of the base's sentences (D75) | Done on synthetic artefacts, `228e73be` (reviewed), after `dev-two-tier-v4` with A31 was merged (`24c6be28`). `prior/selection.py`: the rules `all` and `landed`, the record of the sentences kept and left out by split, airport, stratum and outcome; the source applies the rule as it reads a split's closed-loop sentences (the stored outcome, D74; the stratum from the sentence file, by the flight's place); the identity holds the record, the checkpoint `ts-prior-checkpoint-v7`; `prior_train --selection` (train and select) prints the artefact's counts; free generation starts from every flight under the prior's own rule, its `readout.json` the flights inside and outside the selection apart, by their stored outcome. Tests of B8 as §12, the artefact's files compared before and after every runner test. The teacher-forced loss of the validation readout reads the same source and rule when B5 writes it. On the formal artefact after A30 |
| The full ts suite | `8fbc4f96` (with `07f3f49b`): 1,607 passed; `849e9cde`: 1,610 passed; `12127353`: 1,660 passed, 1 skipped. `228e73be` (`./run_all_tests.sh`): 1,613 passed, 1 skipped; 1,005 passed; 161 passed `415a7461` (A32 merged; B on its reader and export): 1,640 passed, 1 skipped; 1,013 passed; 161 passed; vitest 714 (the agent's run) |
| B6: the Training view of stage B | Started 2026-10-05: A23 is on `dev-two-tier-v4`, merged (`05cc8e8a`). The plan (the parts the design does not give are Claude's proposals, marked P): (1) free generation records, at each row, the words the procedure masks blocked in each column they rule, under the runway and G after the row's runway word; its files get a format name (P: `ts-prior-free-generation-v1`; they had none). (2) `experiments/prior_training_export.py` (Claude writes it): from one free-generation readout (and its prior), each flight of it: A23's flight payload (the observed track, the open-loop sentence) and the closed-loop sentence at the prior's Δ, through A23's export functions (accepted by the user 2026-10-05: the closed-loop part is flown again by A23's code, which reads `autopilot.replay`; the prior's runner imports no `autopilot/` module beyond D69's. It needs from stage A, on `dev-two-tier-v4`, a function of `experiments/training_export.py` that builds the payloads of given flights of one split at given Δ — `build_airport`'s loop over a split, its behaviour unchanged (the user, 2026-10-05: stage A writes it, stage B does not edit stage A's code; Claude's draft in the session's scratchpad, `split_flights.patch`). Until it is merged, the export is written without its closed-loop part); each sentence the prior said, flown again through the start (A28) with its words, refused unless it gives the readout's states within the executor conformance bound and its outcome; its words, flown track, attitude, crossing and DA check, the probability of "go-around" and the blocked words of each row; for each candidate the region (FAF and LPV cone, drawn as an outline), the glidepath lower edge, the DA and the entry height. P: sample `aeroviz-training-prior-sample-v1` in `<airport>/training/<set>/`, listed in its own index `index_prior_v1.json` (`aeroviz-training-prior-index-v1`), kind `prior-free-generation`; the judge's envelopes only for the closed-loop sentence (as A23), not for the prior's sentences. Tests: a sample written and read again; the fixtures of the frontend written by the export. (3) The frontend's view and the backend's live segment of a prior sentence: a sonnet agent (the user, 2026-10-05: the Training page is frontend work, delegated), on the export's fixtures; reviewed by Claude. (4) The smoke set (the model of B3's smoke on `v11` at Δ = 4 s) on a test stack from this worktree, the browser check by a one-shot subagent; the publication of the folds and the base after B5 Progress: (1) `b079ebef`; (2) `ce601bd7` (reviewed) after A36 was merged (`ab12b0d7`): the export calls `split_flights` at the prior's Δ; a seeded draw of each airport's flights of the readout (D55); every prior sentence flown again within the executor conformance's bound (a mirror pinned in a test: D69 keeps the runner from importing `autopilot.conformance`), its outcome, timeout and go-arounds; until A32 (D86) a set comes from a select readout. The export's check found a bug of B4's free generation: a sentence's go-arounds counted the words the speaker says for a flight already done (fixed in `ce601bd7`; only smoke readouts were made before). Smoke on `v11` at Δ = 4 s (the 3-epoch model of the formal-size check, select, 5 flights × 2 sentences of each airport, scratchpad): every sentence and closed-loop sentence flown again within the bounds, 0.5–1.0 MB an airport, 85 s. (3) The view and the live segment, by a sonnet agent: `aac93945` (a route `/autopilot/prior-segment`: the flight's closed-loop setup of A23 with the prior sentence's words; the frontend gives a prior sentence to A23's views as a derived flight; a Stage A / Stage B switch; the row inspector; the procedure's limits in 3D), reviewed; fixes `201b4afd` (a stale set choice that could stop the app, the stage switch, the runway in force highlighted, words said after the judge's outcome greyed — in A23's sentence bar too — and refused by the backend, labels) and `a280d984` (the set names its artefact and executor spec relative to the repository). Hooks in A23's files: exports and one argument in `trainingSample.ts`, one condition in `useTrainingAutopilot.ts`, the switch in `TrainingPanel.tsx`, one line in `App.tsx`, the greyed words in `TrainingSentenceBar.tsx`, `backend.py`, `http_server.py`. A live segment equals the readout's states within 1e-6 m (synthetic) and the written track within its 0.1 m rounding on 244 segments of the smoke set. Tests: vitest 714, backend 23 (+126 subtests). (4) The smoke set on a test stack from this worktree (vite 5179, backend 8791, a scratch public directory: the live data with the smoke set beside stage A's), the browser check by a one-shot subagent: every item passes. Open: the publication of the folds and the base after B5; stage A's `/autopilot/segment` refuses no word said after the outcome (an IndexError; reported to the user) |
| After A32 (D77–D87): stage B on its interfaces | `dev-two-tier-v4` at `37d632eb` merged; `3b9c3302`: the prior's input function reads a closed-loop sentence's rows alone (D82), the selection and the readouts its stratum and stored outcome from the withheld fields; free generation and the export the same; the export calls `split_flights` with the prior's Δ and no formal rows (D86: a set of any split); the stage-B fixtures written again. `415a7461` (the sonnet agent): the live segment on A32's `SetFlown(params)` and rows. `v11_20261004` is refused by A32's format names: the real-data checks wait for A34's artefact |
| B5's tools (built before A34, on synthetic artefacts; started 2026-10-05) | Plan, in order: (a) the first-step runway (top-1) at the held-out airport in a fold's `held_out.json` (`prior/train.py`); (b) the base's one validation readout without its free generation (`experiments/prior_validation.py`: the teacher-forced loss of the val sentences under `landed`, the share of the labelled words that the procedure masks block); (c) `experiments/prior_select.py`: §5's rules on the fold runs — the configuration after steps 1–2, the variant after step 3 — written as a choice; (d) `experiments/prior_campaign.py`: the 31 runs as one campaign from one commit on a clean checkout, one at a time on the GPU, each fold's free generation at its held-out airport (select, 200 flights × 2), the choices between the steps, the base and its validation readout; resumable. Then the intents (`docs/experiments/intents.json`) and `docs/reference/runners.md` (B7) Done on synthetic artefacts: (a)+(b) `0ea056f5`, (c)+(d) `f086d723`, review fixes `649a3d74` (reviewed: no wrong flag of any runner and no rule of §5 read otherwise; fixed: the base's last file, the lock and the running step's PID, the commit checked before each step, D85's guards, the arm's checks). **Decided by the user, 2026-10-05:** a tie of parameters among the configurations within twice the seed scale (A and D have one shape) goes to the lower score. Claude's readings, accepted by the user (2026-10-05): the second seed 2024; the base's free generation on the val days 200 flights × 2 an airport, as a fold's; D85 kept by a mark in the prior's run (`val_read_<runner>.json`) and a smoke that reads select, never val; a killed step's directory moved aside as `<dir>.aborted-<UTC>` by the campaign itself (outline E8), recorded; `--smoke N` runs every step small (N sentences, 5 flights) to check the chain on A34's artefact first. Tests: 132 of stage B. Open: the intents (`docs/experiments/intents.json`) before the publication; `docs/reference/runners.md` at B7 (after merging `dev-two-tier-v4`, for the next free R number) |
| B5, B7 (the close of stage B) | Wait for A30's artefact and Claude's check of stage A; the user chose Δ = 4 s (2026-10-04, recorded in vocabulary §0.3) |
| Before B5: the formal-size check at Δ = 4 s | Done 2026-10-05 at `228e73be` on `v11_20261004` / `v16_20261004`, Δ = 4 s, `landed`, all five airports, in the scratchpad (the user asked for it: B3's check was at Δ = 2 s, configuration A only). The selection keeps train 39,645 (886 left out, 2.2 %), select 6,025 (174), val 9,540 (210). The memory check: the largest batches 256 × 64 rows and 36 × 386 rows (the longest sentence 386 rows, 770 at Δ = 2 s), 8 candidates; GPU peak reserved 1.86 GB (A) and 2.43 GB (C) of 7.5 GB free; host 3.0–3.2 GB. Three epochs of A (2.17 M parameters) and of C: 25.3 s and 35.8 s an epoch (train and the select loss), approximately 1 min 45 s before the first epoch (the checks and the reading); the select loss per step 2.50, 1.85, 1.64 (A) and 2.25, 1.78, 1.61 (C). Claude's estimate of §5 from it, every run at most 30 epochs (`max_epochs`; patience 3 stops many earlier), a fold on four fifths of the sentences: A or D at most approximately 12 min a fold run, C approximately 16 min, B (not measured, between) approximately 14 min; steps 1–3 and the base at most approximately 7 h of GPU. Free generation on `v11` (the smoke of B4 on real data): the 3-epoch model of A, the select days, 40 flights of each airport × 2 samples, 3 min 15 s for 400 sentences (approximately 1 min of it the checks); 398 inside the selection, 2 outside; 38 of the 400 landed (an untrained model: a check that the loop runs, not a result); so approximately 3 min the free generation of one fold (200 flights × 2), approximately 1.5 h for the 30 fold runs if every run is read |

Proposals (where the design says nothing): none open. Decided: the probability of "go-around" on the final (`47ece60f`)
as D72 (the user, 2026-10-04). The nine proposals of `bbedfe9e` were decided by the user on
2026-10-04 as D64 (the procedure masks: 7, 8, 9; 7 and 9 changed) and D65 (the inputs and the model: 1, 2, 4, 5, 6; 4
changed); 3 (the runway head's class order) is a detail of the code. The code follows them at `07f3f49b`.


## 2 Plan of the milestones that are done (was the rows of §0.4)

| When | What |
|---|---|
| Now | B0. B2 and the training loop of B3, tested on synthetic inputs. The words of each column and their number come from the vocabulary spec (vocabulary §6, item 1), never from constants of the prior |
| After A19, A20 and A22 of stage A (the altitude words above E, the new format names, the executor that flies T + E; the vertical path of each candidate in `candidates.json`, the reader of a closed-loop file, D61; the grammar's column mask, D62) | B1, tested on synthetic artefacts (`tests/support.py`). The speaker and the masks of B4 |
| After A21 of stage A (the formal artefact), and again after A25 (the formal artefact with the vertical tolerance of the final descent, D66) and A30 (the same artefact with no code digest, vocabulary D73; the formal runs read it) | B1–B4 on a sample of the formal artefact at Δ = 2 s: the smoke run of B3, its time and the memory check at the formal size; free generation with the executor of the formal artefact |
| After A26 of stage A (the start of a closed loop, D67), merged into this branch | Free generation of B4 (the speaker with the executor and the judge, through the start), on synthetic artefacts and on the formal artefact |
| After A27 of stage A (each flight's stratum in the sentence file, D70), merged into this branch | B4's readout by stratum |
| After A31 of stage A (each closed-loop sentence's outcome, vocabulary D74), merged into this branch | B8 (the selection of the base's sentences, D75), on synthetic artefacts; on the formal artefact after A30 |
| After A28 of stage A (the start opens the executor spec, D71), merged into this branch | The main of the free-generation runner of B4 (it gives the start the directory of the executor spec) |
| After A23 of stage A (the Training view of stage A), merged into this branch | B6's export and view, on the smoke sets of B3 and B4 |
| After A36 of stage A (`split_flights`, the closed-loop part of A23's export as one function), merged into this branch | B6's export of the closed-loop sentences beside the prior's own, through `split_flights` at the prior's Δ; stage B does not change A23's code |
| After A32 of stage A (the corrections of Claude's review, vocabulary D77–D84), merged into this branch | The reader of item 3 with the later-row fields apart (D82), the start that reads no sample after the first predicted step (D77) and the candidates decided by no flight (D78), followed in `prior/` and its runners; the tests on synthetic artefacts again |

## 3 Specifications of the milestones that are done (was §12: B0–B4, B8)

**B0. Package and layout.**

- A new package `prior/`. It reads `instructions/` and the artefact files; it does not import `autopilot/`; only the
  runners join it to the executor (§1). These rules go into `tests/test_architecture.py` (the prior's rules cut into
  `tests/test_mirrors_cut_from_live_tests.py` come back, rewritten).
- The runners of B: `prior_train` (one run: all airports, or one fold with a held-out airport), `prior_select` (reads
  the campaign of B5 and writes the choice), `prior_free_generation` (the prior speaks, the executor flies). New code;
  the archived runners of the same names stay as they are.

**B1. Data** (§2, §7 item 2; D13, D17, D23–D25, D32, D41, D60, D63).

- The rows: before the first predicted step, the observed states; from it on, the flown states of the closed-loop
  sentences, on the rows of the chosen Δ.
- The own state: height above the airport elevation (D58), ground speed, vertical rate, `no_motion`; the motion from the
  displacement in the 2 s before the row; at row 0 the motion inputs 0 and `no_motion` 1 (D60).
- One vector for each candidate: the distance before its threshold along its course, the offset right of its final,
  the height above its threshold, the motion direction minus its course (sine, cosine), the height above its glidepath
  (straight-line reference), the landings on it in the 30 min before the step. No constant of the runway, except in the
  variant `constants` of D39 (length, threshold elevation).
- The words in force: the runway as the candidate vector of R ("none yet" up to the first predicted step); G; the
  heading as the sine and cosine of its angle relative to the course of R; the other columns as embeddings; the time
  since each column said its word, in seconds.
- The targets: the five columns of the closed-loop sentence.
- Every scale is a constant in SI units (D41). A flight without a training sentence (vocabulary §6, item 3) is not read.
- The landings from each airport's tracks roster, less the sealed test days, never a flight's own (§2, D63); their
  digest in the identity of the data (§8, item 1).
- Tests: a change of the runway word leaves the inputs of the rows up to the first predicted step the same, bit for bit
  (D23); the glidepath height against a hand computation; the motion from the 2 s displacement at Δ = 2, 4, 8 s; at row
  0 the motion inputs 0 and `no_motion` 1 at Δ = 2, 4, 8 s, also when the data have a 2 s row before row 0, and
  `no_motion` 0 at every other row (D60); a change of the stored track, ground speed and vertical rate of the states
  changes no input (only positions and heights give the motion); the flown states from the first predicted step on; a
  permutation of the candidates permutes their vectors and nothing else; an airport with more candidates than any
  training airport is read; a change of a landing's time or runway in the roster changes the digest of the landings, and
  a change of another field of the roster does not; a run refuses landings whose digest differs from the identity (D63).

**B2. Model** (§2, §3; D16, D41, D65).

- The time attention with RoPE on the seconds from the aircraft's row 0; no row-position and no airport embedding.
- The candidate tokens through one shared network and one attention over them (weights that sum to one); five heads in
  the order of the columns; the runway head points at the candidates, plus "unchanged" and "go-around"; the first
  predicted step masks "unchanged" and "go-around".
- Each layer has a place where a module with a zero output can be added (§7, item 5).
- The checkpoint: a new format name; its identity as §8.
- Tests: the row-by-row encoding gives what the encoding of the whole sentence gives; a shift of all times changes no
  output; a permutation of the candidates permutes the runway scores; any number of candidates.

**B3. Training** (§5; D31, D40).

- Teacher forcing; the airports of a run (all, or a fold without its held-out airport); the stop on the select days;
  the validation days not read. The closed-loop sentences are read only after the check of vocabulary §6, item 3
  (D69), which the runner calls with the artefact and the executor spec's directory and which runs the closed-loop
  check itself (vocabulary D73).
- Before a formal run: the check at the formal size of the host memory and the GPU memory of the largest batch.
- A smoke run on a sample of the formal artefact; it gives the time of one run for the cost of B5.
- Tests: the stop reads only the select days; a fold never reads its held-out airport in training.

**B4. Speaking and free generation** (§4, §7 item 3; vocabulary §6 items 2, 3, 5, 6; D14, D33, D38, D52, D62, D64,
D67–D72).

- The masks of §4: the grammar; the procedure masks of D64 (the glidepath lower edge inside the region, the DA outside
  it, no climb back with the band of the level nearest the entry height; never "unchanged"; the climb mask lifted while
  G is true and a go-around clearing the join, D14), a level checked with the band ε of its level (D52); the masks of a
  caller.
- The closed loop: the prior speaks, the executor flies, the judge decides (D33, D38); 900 s more time at each
  go-around; the observed rows before the first predicted step, the executor's states after it. The executor is started
  and flown a row at a time through the start of a closed loop (vocabulary §6, item 5; D67), and the outcome is the
  judge's (item 6). The runner gives the start the directory of the executor spec, which the start opens and checks
  (D71). Before it reads closed-loop sentences, the runner calls the check of item 3 (D69); it imports nothing else
  of `autopilot/`. At most 2 go-arounds a flight (D68).
- The readout: the outcomes for each airport and each stratum of the flight (straight-in or vectored, as the sentence
  file stores it: vocabulary §6, item 3; D70); the words for each column against the labelled ones; the go-arounds said
  and the flights that reached the bound of D68; the probability of "go-around" on the final (D72).
- Tests: one flight spoken and flown to its outcome; each mask; G; the time limit; the same seed gives the same
  sentence; no row reaches a column with no permitted word (D62); a procedure mask never blocks "unchanged", also when
  the word in force breaks a limit; a level below the DA is blocked after the join when the aircraft has left the LPV
  cone; an aircraft at the level nearest the entry height, less than its ε below it, may still climb; after a go-around
  the join and the passage below the entry height start again (D64); after a flight's second go-around, "go-around"
  is masked (D68); the runners of `prior/` import from `autopilot/` only what vocabulary §6 lists: the modules of
  items 5 and 6, and of `autopilot/closed_loop.py` only `require_conforming_closed_loop` (item 3, D69); the test reads
  the names imported (`tests/test_architecture.py`).

**B8. The selection of the base's sentences** (§8 item 1; vocabulary §6 item 3; D74, D75). After A31 is on this
branch; before B5.

- `prior/selection.py`: the rules `all` and `landed`, applied to the sentences of a split and Δ as the reader of
  item 3 gives them, with their stored outcomes; the artefact is not changed. `prior_train` takes the rule
  (`landed` for every run of B5) and applies it to train and select; the validation readout applies it to the
  teacher-forced loss, and free generation draws from every flight.
- The identity of the data (§8, item 1) holds the rule and, for each split, airport, stratum and outcome, the
  sentences kept and left out; a run refuses a selection that differs from its identity. The readouts print the
  counts.
- Tests: `landed` keeps exactly the sentences whose stored outcome is a landing and `all` keeps every one; the counts
  add up to the sentences of the split; the artefact's files are unchanged after a run; a run under another rule
  than its identity's is refused by name.

