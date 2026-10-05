# Two-tier model: the prior

**Summary.** The prior is the upper part of the two-tier model (outline §1): a language model of the controller. At
each row it says one row of words for each aircraft, and the executor flies them. This document gives its inputs, its
outputs, its decoding, its training and how it runs at a new airport. It is stage B of the plan. It reads the
vocabulary only through the vocabulary's public interface (vocabulary §6) and its decisions; the post-training reads
this document only through its public interface (§7) and its decisions.

**Language.** This document uses the writing rules of ASD-STE100 (Simplified Technical English): short sentences, active
voice, one meaning for each word, and technical names that the terms define. The words were not checked one by one
against the STE dictionary. Units are SI only (m, m/s, deg, s). Values in feet, knots or nautical miles occur only in
quotations of regulations, with the SI value given one time.

**Scope of this document.** This document is complete for its part. The principles and the shared rules are in the
outline (`outline.md`). The code of stage B is on the branch `dev-two-tier-v4-prior` (§0.3); the code that it
replaces is archived (§11).
Paths in backticks are relative to `4dTrajectory/ts_transformer/` unless they start with `4dTrajectory/` or name the
repository root; `readouts/` is `docs/two_tier/readouts/`. The documents of the earlier design (`*_design.zh.md`,
`two_tier_framework.zh.md`, `two_tier_stage_notes.zh.md`) are in `archive/two_tier_v3_2026_10/docs/`.

---

## 0 Status

### 0.1 Decisions

The decision numbers are shared by all documents (outline §3).

| # | Item | State | Source |
|---|---|---|---|
| D5 | Prior inputs are in the frame of the runway in force. The prior has no airport embedding, no absolute position and no absolute direction | Decided | User, 2026-10-03 |
| D13 | The prior gets the height above the published glidepath of R at each step (§2) | Decided | User, 2026-10-03 |
| D14 | While G is true, the procedure mask "no climb below the entry height" does not apply (§4). (The same decision has a part in the vocabulary: rule 5) | Decided | User, 2026-10-03 |
| D16 | The prior has no row position embedding and no input "time from row 0". The time attention uses RoPE with the row's time in seconds (§2) | Decided | User, 2026-10-03 |
| D17 | Every column has the input "time since this column said its word in force", in seconds; the runway column too (§2) | Decided | User, 2026-10-03 |
| D23 | How the prior gets the frame of R (D5) and the glidepath height (D13). The own state of the aircraft has no frame. Every position and direction is in the candidate vectors, each in the frame of its own candidate. R is an input only as its candidate vector. No input of a row up to the first predicted step is computed from R. Each candidate vector has the height above its own glidepath; the value of R is the input of D13 (§2) | Decided | User, 2026-10-03 |
| D24 | A candidate vector has no constant of its runway: no length, no elevation, no layout relative to R. Such a value comes back only as a variant that D39 selects (§2) | Decided | User, 2026-10-03 |
| D25 | At each Δ of the ablation (2, 4, 8 s), the motion inputs of a row come from the 2 s before the row (§2). (The Δ values: D25 in the vocabulary) | Decided | User, 2026-10-03 |
| D31 | The prior's training stops on the select days (§5). (The multi-aircraft part: D31 in the post-training) | Decided | User, 2026-10-03 |
| D39 | The prior's design is chosen by leave-one-airport-out cross-validation (5 folds: train on four airports, read the fifth). Two variants: `full` (§2) and `constants` (`full` and, in each candidate vector, the runway's length and threshold elevation). `constants` is chosen only if it is better by more than twice the seed scale (§5) | Decided | User, 2026-10-04 |
| D40 | Hyperparameters: configuration A (§5) is the start. The same folds compare four configurations (A, a smaller model, a larger model, a stronger regularization); the one with the fewest parameters within twice the seed scale of the best is chosen (§5). Every configuration has attention heads 32 wide: A 6 heads, B (d_model 128) 4, C (d_model 256) 8 — 128 and 256 are not six heads of one width | Decided | User, 2026-10-04; the head width: user, 2026-10-04, on Claude's proposal |
| D41 | The design runs at an airport that is not in the training data: the number of candidates is not fixed; every input has a fixed physical scale, never a statistic of the training data; each fold of D39 flies the full closed loop at its held-out airport. The altitude words are above the airport elevation (D58), so the levels of an approach lie in the 60 m segment at any airport (§6) | Decided | User, 2026-10-04 |
| D58 | The prior's own height is the height above the airport elevation E, as the altitude words are (vocabulary, D58). The prior gets neither E nor the MSL height (D24), so it cannot know where the round MSL levels that controllers assign lie (§2). (The words: D58 in the vocabulary) | Decided | User, 2026-10-04 |
| D60 | Row 0 of an aircraft has no motion inputs: no state 2 s before it is stored. There, the ground speed, the vertical rate and the motion direction in each candidate vector (sine, cosine) are 0, and the own input `no_motion` is 1 (0 at every other row). The same at every Δ, in training, in closed loop and in a loop of several aircraft. Not the fitted values of the data, which use 7.5 s after the row (§2) | Decided | User, 2026-10-04, on Claude's proposal |
| D63 | The identity of a prior's data (§8, item 1) also holds the landings that the candidate vectors count: for each airport, the sha256 of each landing's flight, runway and time, in time order, and of the number of landings left out on the sealed test days. The landings come from the airport's tracks roster, outside the artefact: without them, a changed roster would change the inputs and leave the identity the same. By their flights, never by the bytes of the roster (D21). A run that reads the landings again computes the digest again and refuses a difference (§2, §8) | Decided | User, 2026-10-04, on the reading of the stage B agent |
| D64 | The word rules of the procedure masks (§4). They block a word when it is said and never block "unchanged". Inside the region (the FAF and the LPV cone of R) a level T only where T ≥ the glidepath lower edge − ε(T), and "no level-off" not where the aircraft is more than ε("no level-off") below the edge. Wherever the aircraft is not inside the region (before the join, or after it when the aircraft has left the region), a level T only where T ≥ DA − ε(T). Before the join, once the aircraft has been below the entry height by more than the band ε of the level nearest the entry height, no level above the aircraft's height + ε and no climb class. A go-around clears the join and the passage below the entry height; while G is true neither is kept | Decided | User, 2026-10-04, on the proposals of the stage B agent and Claude's review |
| D65 | The inputs and the model where §2 left them open: the fixed scales of §9 (the height above the glidepath as asinh(h / 100 m)); the candidate tokens reach a row through one attention over them whose weights sum to one; the RoPE base is 10,000; the variant `constants` gives the threshold elevation MSL; the runway column's `since` starts again at a candidate word, not at "go-around" (§2) | Decided | User, 2026-10-04, on the proposals of the stage B agent and Claude's review |
| D68 | Free generation lets a flight say at most 2 go-arounds. After its second, the runner forbids "go-around" in the runway column (a mask of a caller, §4), and it gives 2 as the most go-arounds to the start of the closed loop (vocabulary §6, item 5; D67). The readout counts the flights that reached the bound. Why: the executor lays out the time that go-arounds add (900 s each) when it starts, and the number that a model says is not known before; no closed-loop sentence of `v9_20261004` at Δ = 2 s has more than one go-around (train 68 of 40,534 sentences, select 12 of 6,199, val 20 of 9,750); 2 lets a model say one more than the data, and no flight goes around without end (B4) | Decided | User, 2026-10-04, on the report of the stage B agent |
| D72 | The readout's probability of "go-around" on the final (B4, B5) is read on the rows on the final (inside the region of the runway in force, D64) where the masks permit "go-around": not while G is true, not after the bound of D68. The probability is that of the distribution the speaker draws the runway word from, the masks applied. Why: at a row where the word is masked, its probability is 0 because of the mask, not the model, and would pull the mean down | Decided | User, 2026-10-04, on the proposal of the stage B agent |
| D75 | The base learns only from closed-loop sentences whose own words land: a selection of the artefact's sentences by a named rule (`all`, `landed`), applied when they are read; the artefact is not changed. `landed` keeps a sentence whose stored outcome (vocabulary §6, item 3; D74) is a landing. It applies to train and select (the stop and the choices of D39 and D40 read the same kind of sentence that the training reads) and to the teacher-forced loss of the validation readout; free generation starts from every flight, and the readout gives the flights outside the selection apart. A run records the rule and, for each split, airport, stratum and outcome, the sentences kept and left out (§8, item 1). Why: about 2.2–2.6 % of the sentences do not land when their own words are flown (A25, Δ = 2 s: train 1,955 of 1,999 replayed, select 6,036 of 6,199; KMSY 94–95 %), mostly high at the DA; their last words teach an end that does not land, and the vocabulary and the executor, not the words, are often the cause, so imitation cannot correct them. They come back in the post-training as starts (post-training D76) (milestone B8) | Decided | User, 2026-10-04 |

### 0.2 Open items

None.

### 0.3 Implementation

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
| B5's tools (built before A34, on synthetic artefacts; started 2026-10-05) | Plan, in order: (a) the first-step runway (top-1) at the held-out airport in a fold's `held_out.json` (`prior/train.py`); (b) the base's one validation readout without its free generation (`experiments/prior_validation.py`: the teacher-forced loss of the val sentences under `landed`, the share of the labelled words that the procedure masks block); (c) `experiments/prior_select.py`: §5's rules on the fold runs — the configuration after steps 1–2, the variant after step 3 — written as a choice; (d) `experiments/prior_campaign.py`: the 31 runs as one campaign from one commit on a clean checkout, one at a time on the GPU, each fold's free generation at its held-out airport (select, 200 flights × 2), the choices between the steps, the base and its validation readout; resumable. Then the intents (`docs/experiments/intents.json`) and `docs/reference/runners.md` (B7) Done on synthetic artefacts: (a)+(b) `0ea056f5`, (c)+(d) `f086d723`, review fixes `649a3d74` (reviewed: no wrong flag of any runner and no rule of §5 read otherwise; fixed: the base's last file, the lock and the running step's PID, the commit checked before each step, D85's guards, the arm's checks). **Decided by the user, 2026-10-05:** a tie of parameters among the configurations within twice the seed scale (A and D have one shape) goes to the lower score. Claude's readings (proposals): the second seed 2024; the base's free generation on the val days 200 flights × 2 an airport, as a fold's; D85 kept by a mark in the prior's run (`val_read_<runner>.json`) and a smoke that reads select, never val; a killed step's directory moved aside as `<dir>.aborted-<UTC>` by the campaign itself (outline E8), recorded; `--smoke N` runs every step small (N sentences, 5 flights) to check the chain on A34's artefact first. Tests: 132 of stage B. Open: the intents (`docs/experiments/intents.json`) before the publication; `docs/reference/runners.md` at B7 (after merging `dev-two-tier-v4`, for the next free R number) |
| B5, B7 (the close of stage B) | Wait for A30's artefact and Claude's check of stage A; the user chose Δ = 4 s (2026-10-04, recorded in vocabulary §0.3) |
| Before B5: the formal-size check at Δ = 4 s | Done 2026-10-05 at `228e73be` on `v11_20261004` / `v16_20261004`, Δ = 4 s, `landed`, all five airports, in the scratchpad (the user asked for it: B3's check was at Δ = 2 s, configuration A only). The selection keeps train 39,645 (886 left out, 2.2 %), select 6,025 (174), val 9,540 (210). The memory check: the largest batches 256 × 64 rows and 36 × 386 rows (the longest sentence 386 rows, 770 at Δ = 2 s), 8 candidates; GPU peak reserved 1.86 GB (A) and 2.43 GB (C) of 7.5 GB free; host 3.0–3.2 GB. Three epochs of A (2.17 M parameters) and of C: 25.3 s and 35.8 s an epoch (train and the select loss), approximately 1 min 45 s before the first epoch (the checks and the reading); the select loss per step 2.50, 1.85, 1.64 (A) and 2.25, 1.78, 1.61 (C). Claude's estimate of §5 from it, every run at most 30 epochs (`max_epochs`; patience 3 stops many earlier), a fold on four fifths of the sentences: A or D at most approximately 12 min a fold run, C approximately 16 min, B (not measured, between) approximately 14 min; steps 1–3 and the base at most approximately 7 h of GPU. Free generation on `v11` (the smoke of B4 on real data): the 3-epoch model of A, the select days, 40 flights of each airport × 2 samples, 3 min 15 s for 400 sentences (approximately 1 min of it the checks); 398 inside the selection, 2 outside; 38 of the 400 landed (an untrained model: a check that the loop runs, not a result); so approximately 3 min the free generation of one fold (200 flights × 2), approximately 1.5 h for the 30 fold runs if every run is read |

Proposals (where the design says nothing): none open. Decided: the probability of "go-around" on the final (`47ece60f`)
as D72 (the user, 2026-10-04). The nine proposals of `bbedfe9e` were decided by the user on
2026-10-04 as D64 (the procedure masks: 7, 8, 9; 7 and 9 changed) and D65 (the inputs and the model: 1, 2, 4, 5, 6; 4
changed); 3 (the runway head's class order) is a detail of the code. The code follows them at `07f3f49b`.

### 0.4 Plan

1. Stage B is developed in parallel with the end of stage A (the user, 2026-10-04), on its own branch (outline §5
   rule 1). B0–B7 (§12): the package, the data, the model, the training, the speaking and free generation, the
   cross-validation and the base model, the Training view of stage B, the close. A milestone starts when the parts of
   stage A that it reads are on `dev-two-tier-v4` (outline §4):

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
   | After A34 of stage A (the formal artefact with D77–D84) | The smoke of B3, free generation and the check at the formal size at Δ = 4 s again, on A34's artefact (the candidates of an airport can change, D78); the formal runs read A34's artefact |
   | After Claude's check of stage A (with A32–A36, vocabulary §12.2 item 8) and the user's choice of Δ (outline §4); A32 holds vocabulary D86 (the export gives val flights for the base's validation readout) | B5; B6's publication of the folds and the base; B7 |

2. Then the post-training (`post_training.md`).

---

## 1 Scope

- **This document owns** the package `prior/` (the inputs, the model, the training, the speaker and its masks), its
  runners, and the Training view of its results (B6, outline §6).
- **It reads** the outline (the principles, the shared decisions D7, D20, D21, D55, the rules of the implementation)
  and the vocabulary's public interface (vocabulary §6): the vocabulary spec and the grammar (items 1, 2), the sentence
  artefact (item 3), the candidates and their geometry (item 4) and the row grid (item 7). The package `prior/` does
  not import the executor or the judge (items 5, 6): a runner joins them through the start of a closed loop (item 5,
  D67; free generation, B4).
- **It gives** the public interface of §7, and nothing else, to the post-training.

---

## 2 Inputs

- **Source of the states (D32).** The rows before the first predicted step are observed (ADS-B), as in closed loop. From
  the first predicted step on, every input that comes from a state (the own state, the candidate vectors, the motion)
  comes from the flown states of the closed-loop sentences (vocabulary §6, item 3), not from the observed track. The
  words in force are those of the closed-loop sentence, corrections included. In closed loop the states come from the
  executor.
- **Own state (D5, D23, D58, D60).** The own state of the aircraft has no frame: its height above the airport elevation
  (the level words are above it), its ground speed, its vertical rate and `no_motion` (1 only at row 0, below). No MSL
  height: with it, the airport elevation (a constant of the airport) would be an input (D24).
- **Frame (D5, D23).** Every position and every direction is in the candidate vectors. Each candidate vector gives the
  aircraft in the frame of that candidate: the distance to its threshold along its course, the offset right of its
  final, the height above its threshold, the motion direction minus its course (sine, cosine)
  (`instructions.airport.relative_to_runway`). "The frame of R" is the candidate vector of R, which the prior gets as
  the runway in force. There is no second set of values relative to R.
- **No input is computed from R before R is said (D23).** The first predicted step says R. The inputs of the rows before
  it and of the first predicted step itself use no value computed from R. (The heads of a row see the runway that the
  same row said, in the order of the columns; that is an output, not an input of the row.) The reason: the artefact
  writes the runway word at row 0 (vocabulary §6, item 3), and its value is the runway on which the flight landed. An
  input of these rows computed from it gives the answer of the first predicted step, and in closed loop the same input
  does not exist. A test: a change of the runway word of a sentence leaves the inputs of rows 0 to `N_LOOK` the same,
  bit for bit.
- **No airport identity (D5, D24).** The prior has no airport embedding, no absolute position (E, N), no absolute
  motion direction, and no table of candidate constants (the absolute threshold position, the course, the elevation,
  the length).
- **Candidates (D24).** A candidate vector has only values that change with the aircraft: the values of the frame above,
  the glidepath height below, and the landings on the candidate in the 30 min before the step. It has no
  constant of its runway. Such a constant identifies the runway:
  - In the candidate table of `v6_20261002`, the pair (length, elevation) is different for 24 of the 25 candidates. The
    length alone tells the left runway of KRDU from the right one (05L / 23R 3,048 m, 05R / 23L 2,286 m).
  - The spacing of parallel runways is different at each airport: KRDU 1,068 m, KSJC 213 m, KSMF 1,827 m, KSTL 397 /
    855 / 1,251 m. A layout relative to R is a code of the airport when R is known.
  - The runway-intent study found this channel (R1.1, gradient-boosted trees, split by day;
    [plan](../../history/2026-09_runway_intent/2026-09-13_runway_intent_plan.zh.md) §15): with a constant of each runway
    (`prior_share`), the trees recognised "30L" and were right on only 53 / 59 % of the flights on the two KSJC days
    with 30L closed. Without it (R1.1b) they were right on 96.8 / 97.6 %. For the prior, R46 shows that it uses an
    airport identity when it gets one (§10.1).

  The constants are also not necessary. The length is a cause only together with the aircraft type, and the prior has no
  aircraft type. The elevation is not needed: the words and the own height are above the airport, and each candidate
  gives the height above its own threshold. (The own height minus a candidate's height above its threshold is that
  threshold's height relative to E: 0–28 m at the five airports, a small constant of the runway that two physical inputs
  give together.) Which of two parallel runways is the left one shows at each step in the offsets right of their finals
  (the value of the left runway is always larger). The relative positions of all candidates together still show the
  layout of the airport. That is real geometry, and the prior has it; only the held-out airports of D39 can measure how
  much the prior uses it. A constant comes back only as the variant `constants` of D39: the length and the threshold
  elevation MSL (D65). That variant thus also gives the airport elevation E, which `full` does not (D58).
- **Landings (D63).** Offline, the landings come from the airport's tracks roster: every flight that the harvest
  assigned to a candidate, with a sentence or without one, less the landings on the sealed test days (C32). A flight
  never counts its own landing: a closed-loop sentence can fly past the time at which the observed aircraft landed, and
  the runway of that landing is the answer of the runway word. In a loop, the caller gives the landings that the loop
  knows. The roster is outside the artefact, so the identity of a prior's data holds a digest of the landings (§8,
  item 1).
- **Candidate tokens (D41, D65).** Each candidate vector goes through one shared network. The candidate tokens reach a
  row through one attention over them whose weights sum to one: a sum would grow with the number of candidates.
- **Scales (D41, D65).** Every input has a fixed scale in SI units (§9): the distances along and across a candidate
  asinh(d / 1 km), linear within a few hundred metres of a final; the heights (above E, above a threshold) h / 1 km; the
  height above a glidepath asinh(h / 100 m), linear near the glidepath, where the value has a meaning, and compressed
  far from it (an aircraft high above an opposite runway's extended glidepath is thousands of metres above it); the
  ground speed v / 100 m/s; the vertical rate v / 10 m/s; the landings in 30 min n / 10; in the variant `constants` the
  length l / 1 km and the threshold elevation MSL h / 1 km.
- **Words in force:** the runway in force as its candidate vector ("none yet" up to the first predicted step), the
  go-around state G, the heading in force as the sine and cosine of its angle relative to the course of R, the other
  columns as embeddings, and the time since each column said its word (D17).
- **Glidepath (D13, D23).** Each candidate vector has the height above the published glidepath of that candidate at the
  aircraft's distance before its threshold, with the straight-line reference of the vocabulary (vocabulary §6, item 4:
  TCH + d·tan(angle) + d²/(2R_e), R_e the earth's radius of curvature along the course). The value in the vector of R is
  the input of D13. Thus the value exists at every row, also before R is said. Every candidate has a TCH and a glidepath
  angle (the vocabulary refuses a candidate without them). It is procedure geometry as an input; the model still decides
  the profile (principle 2). The value is only meaningful near the final; the model also has the offset from the final
  to weigh it.
- **Motion (D25).** The ground speed, the vertical rate and the motion direction (in each candidate vector, minus its
  course) come from the displacement in the 2 s before the row, at every row interval Δ. Only past positions
  give them; never the fitted track, ground speed or vertical rate of the signals: a least-squares fit over 15 s
  centred on the row, which uses 7.5 s of the future.
- **Row 0 (D60).** Row 0 of an aircraft has no state 2 s before it: the observed data start at the entry of the 25 km
  slice, and the sentence artefact stores the states from row 0 of each sentence (vocabulary §6, item 3). At row 0 the
  ground speed, the vertical rate and the motion direction in each candidate vector (sine, cosine) are 0, and
  `no_motion` is 1; at every other row `no_motion` is 0. The rule is the same at every Δ, also when the data have a 2 s
  row before row 0, so that only the interval changes with Δ (D25). It is the same in training, in closed loop and in a
  loop of several aircraft (§7, item 2): the first row of an aircraft never has a state before it. The zeros alone
  already mark the row (no aircraft in flight has a ground speed of 0, and (0, 0) is not a direction); `no_motion` says
  it directly, so that the network does not have to learn it from an extreme value. Two other values were not chosen:
  - The fitted values of the data at row 0. They use 7.5 s after the row. For the aircraft's own first predicted step,
    16 s after row 0, this is no leak; but in a loop of several aircraft another aircraft reads the row at its own time,
    and then reads 7.5 s of the future (principle 7). Also, it would be a 15 s fit at one row and a 2 s displacement at
    every other row.
  - The displacement from row 0 to the next row: in a loop of several aircraft, that is 2 s of the future.
- **Time (D16).** No row position embedding and no input "time from row 0": both measure the time since the aircraft
  entered the 25 km slice, a cut of the data, and a long sentence (a go-around adds up to 900 s) reaches rows that
  training seldom saw. The causal time attention gives the order. RoPE in the time attention gives how far back each
  earlier row is: the rotation of a row's query and key uses its time in seconds, so the attention reads only time
  differences. Seconds, not rows, so that every row interval of D11 reads the same time. The time of a row is in seconds
  from the aircraft's own row 0. Only differences count, so the zero changes no result; but a UTC time (approximately
  1.76e9 s) in float32 has a step of 128 s and loses the rows. RoPE works with the row-by-row cache of the speaker
  (`Prior.extend`): a key is rotated once, when it is written. The RoPE base is 10,000 (D65): with heads 32 wide, the
  periods go from 6.3 s to approximately 35,000 s.
- **Time since each word (D17).** Each column has the input "time since this column said its word in force" (`since`):
  log(1 + t / 2 s) / 5, with t in seconds, counted from the first predicted step at the earliest. It is in seconds for
  every column, so that every row interval of D11 reads the same time. The runway column's value is, in the labelled
  data, the time since the first predicted step, because a labelled sentence says its runway only there, and again only
  at the runway word that ends a go-around (D19, D26). In closed loop it is the time since the runway was given or given
  again (a change of runway, or the runway word that ends a go-around), so it measures a real fact. A candidate word
  starts it again; "go-around" does not, because it keeps R (D10) and G is an input of its own (D65).

**Why the motion inputs do not change with Δ (D25).** If the motion of a row were the displacement since the row before,
it would be the mean over Δ. In a 3°/s turn, the direction of that mean is approximately 12° behind the track at
Δ = 8 s (3° at 2 s). The ablation of Δ would then compare a coarser input as well as a longer interval. With the
displacement in the 2 s before the row, only the interval changes. The data has a row every 2 s, and in closed loop the
executor has a state every 1 s, so the value exists at every Δ, at every row after row 0 (D60); the closed-loop
artefact stores the flown states on the 2 s rows (D51). At Δ = 2 s it is the value of now.

## 3 Outputs

Five heads in the order of the columns (vocabulary §6, item 1). The runway head scores each candidate (a pointer) plus
"unchanged" and "go-around". The number of candidates is the airport's own: no slot count is fixed by the model (D41).

## 4 Decoding

The speaker says a row column by column, in the order of the columns. A later column sees the words of the earlier
columns of the same row. Three kinds of masks block words:

**The grammar.** Rules 1–6 and the runway/G table, the vocabulary's one function (vocabulary §6, item 2). The first
predicted step masks "unchanged" in each column and "go-around" in the runway column. The speaker asks the grammar
column by column, after the earlier columns of the row (D62). It gives the grammar, for each later column, the words
that its other masks (below) permit there, so that a row never reaches a column with no permitted word.

**Procedure masks (D64).** The prior decodes under three masks from the procedure of R (principle 3). The first two
are lower limits on the altitude words; the third blocks the climb. The region is inside the FAF and the LPV cone of R;
the join is the first row inside it. ε is the band of a level (vocabulary §6, item 1; D52):

- **The glidepath lower edge**, inside the region: the published glidepath − 60 m (nowhere else: the RNAV floors outside
  the FAF disagree with 10–14 % of the recorded tracks). A level T only where T ≥ edge − ε(T). "No level-off" not where
  the aircraft is already more than ε("no level-off") below the edge. The edge is taken at the aircraft's position; it
  only falls toward the threshold, so a level permitted there stays permitted inbound.
- **The DA**, wherever the aircraft is not inside the region: before the join, and after it when the aircraft has left
  the region (out of the LPV cone before the threshold). A level T only where T ≥ DA − ε(T). After the join, outside the
  region, no other limit holds, so the DA holds there too.
- **No climb**, before the join, once the aircraft has been below the entry height (the glidepath at the FAF) by more
  than the band ε of the level nearest the entry height: no level T above the aircraft's height + ε(T), and no climb
  class. The band: the levels are on a grid and the executor holds a level exactly, so an aircraft at the level nearest
  the FAF altitude can be up to half a step below the entry height without a descent below it.

**A mask blocks a word when it is said, never "unchanged".** A word in force was permitted when it was said. A mask
that made the speaker change it, for example a level forced above an aircraft that sank under the edge with
"no level-off" in force, would fly the aircraft along a line that the procedure computes: the glidepath floor that the
executor does not have (D9). Whether to correct or to go around is the model's decision, and the DA check judges it
(vocabulary §6, item 6).

**Masks while G is true (D14).** Only one mask stops a go-around: "no climb below the entry height". It does not apply
while G is true. A go-around clears the join and the passage below the entry height, and while G is true neither is
kept: the next approach is read as a new one (D26). The other two masks are lower limits; a go-around climbs above
them, so they apply. The grammar applies while G is true as at every row.

A prior records the set of procedure masks that it was trained under (`procedure_masks.json`, contract C35) and speaks
under it.

**Masks of a caller.** The loop that runs the speaker can give a set of forbidden words for each column (§7, item 3).
The speaker applies them as it applies the others. The prior does not know what they mean. Free generation gives one:
"go-around" after a flight's second go-around (D68).

## 5 Training

Teacher forcing on the closed-loop sentences of the artefact, split by operating day (`data/day_split_20260924.json`;
test days sealed). KAUS is a held-out test airport only: it is read one time, at the end of the whole chain.

**Stop on the select days (D31).** The training keeps the epoch with the smallest per-step loss on the select days. The
training does not read the validation days: they are read one time for each stage.

**Hyperparameters (D40).** Configuration A, the start: d_model 192, 4 layers, 6 attention heads, feed-forward 768,
dropout 0.1, weight decay 0.01, learning rate 3·10⁻⁴ with 500 warm-up steps, gradient clip 1.0, 16,384 aircraft-steps
in a batch, at most 30 epochs, a stop after 3 epochs without a better select loss. The RoPE base is 10,000 (D65); a
test makes sure that a shift of all times changes nothing.

**Cross-validation and the choice of the design (D39, D40).** The folds are the five airports: a fold trains on the
train days of four airports, stops on their select days and reads the select days of the fifth (the held-out airport).
The goal is a new airport, so the folds are airports, not days. All runs use the chosen Δ (D11).

| Step | Runs | Training runs |
|---|---|---|
| 1 | Variant `full`, four configurations, each on the 5 folds. A: the start. B: d_model 128, 4 heads, feed-forward 512. C: d_model 256, 8 heads, feed-forward 1,024. D: dropout 0.2, weight decay 0.05. B, C and D have the layers and the learning rate of A; every head is 32 wide | 20 |
| 2 | The seed scale: configuration A with a second seed, on the 5 folds | 5 |
| 3 | Variant `constants` with the configuration chosen in step 1, on the 5 folds | 5 |
| 4 | The base: the chosen configuration and variant on all five airports, stopped on their select days; the validation days read one time | 1 |

The rules, fixed before the runs:

- **The score** of a run is the mean, over the 5 folds, of the per-step loss on the held-out airport's select days.
- **The seed scale** is the absolute difference between the scores of configuration A at the two seeds.
- **Configuration:** among the configurations whose score is within twice the seed scale of the best, the one with the
  fewest parameters.
- **Variant:** `constants` only if its score is lower than that of `full` (same configuration) by more than twice the
  seed scale; else `full`.
- **Readouts of each fold, not used for the choice:** the runway word of the first predicted step at the held-out
  airport (top-1); free generation at the held-out airport (200 flights × 2 sentences): its outcomes (vocabulary §6,
  item 6) and the words for each column.

The cost, Claude's estimate from the smoke of B3 (`v9_20261004`, Δ = 2 s, configuration A): approximately 45 s an
epoch on a fold and 55 s on all five airports, so at most approximately 25 min for one run of 30 epochs, and
approximately 13 h of GPU for the 31 runs if each run costs as much as A (B is smaller, C is larger). The runs use
Δ = 4 s, where a sentence has half the rows. Before B5, the memory check and the time are measured again at Δ = 4 s
with configurations A and C (§0.3); that measurement replaces this estimate. The free generation of the folds is not
measured on real data: Claude's estimate before the smoke, 1–2 h.

## 6 Running at a new airport

The prior has no identity of an airport (D5, D23, D24). A new airport needs only data that the five airports also
come from:

| What the airport needs | Source |
|---|---|
| Candidate runways and their geometry: threshold, course, TCH, glidepath angle, LPV DA; the airport elevation | FAA CIFP and the runway data (the vocabulary refuses a candidate without them) |
| The procedure data for the procedure masks: the FAF, the LPV cone | FAA CIFP procedure details |
| The landings in the 30 min before the step | The airport's tracks (offline) or a live surveillance feed |
| The dynamics of each aircraft type | Independent of the airport |

The design makes sure of three more points:

1. **No fixed count of candidates.** The runway head points at the airport's candidates, whatever their number; no
   slot count comes from the training airports.
2. **Fixed physical scales.** Every input is scaled by a constant in SI units. No input uses a mean, a spread or another
   statistic of the training data or of an airport: a new airport has none.
3. **The check is a flight.** Each fold of D39 runs the full closed loop at its held-out airport (the prior speaks,
   the executor flies, the judge decides), as well as the teacher-forced loss. The final test on KAUS is the same.

**Heights and speeds at a high airport.** The altitude words are above the airport elevation (D58), so a level word
means the same height above the airport, and the final approach lies in the 60 m segment, at every airport. The speed
words are ground speeds: at a high airport one indicated airspeed is a larger ground speed (at E = 1,650 m approximately
8 %, about 6 m/s at an approach speed of 70 m/s: one step). The speed words of a high airport thus lie about one step
above those of the training airports for the same phase. The five airports and KAUS lie below 200 m.

---

## 7 Public interface

The post-training and its code use this document only through these items and the decisions (the D numbers).

| # | Item | What it gives |
|---|---|---|
| 1 | The checkpoint | A trained prior in a format with its own name; its identity (§8) |
| 2 | The inputs of a row | One function from the states on the data's 2 s rows (observed before the first predicted step, flown from it), the words said, the candidates and the landings before the step to the inputs of a row (§2). Training, free generation and a loop of several aircraft use the same function |
| 3 | The speaker | Says one row from the inputs of the row, column by column, with a cache from row to row; under the masks of §4, the masks of a caller included; with a random generator that the caller gives |
| 4 | The teacher-forced loss | The loss of each step and each column for a batch of sentences |
| 5 | A place in each layer | A module added at each layer whose output starts at zero leaves every output of the prior unchanged until it learns |

---

## 8 Gates and identities

The gate of this document is the prior: the teacher-forced likelihood against baselines and free generation. The user
sets its criteria (D7). The identities follow D21 (outline §3):

| # | What | Its identity |
|---|---|---|
| 1 | The artefact of a prior | The spec sha, the day split, the candidate table, the sha256 of the sentence files, the selection of its sentences (D75: the rule, and the sentences kept and left out for each split, airport, stratum and outcome), and the landings that the candidate vectors count (D63): for each airport, the sha256 of each landing's flight, runway and time, in time order, and of the number left out on the sealed test days. A run that reads the landings again computes the digest again and refuses a difference |
| 2 | The procedure masks of a prior | The set name, the checkpoint sha and the digests of the procedure data (C35). The procedure data are the format of the masks |

---

## 9 Values

| Item | Value | Source |
|---|---|---|
| Motion inputs | Displacement in the 2 s before the row, at every Δ; at row 0 of an aircraft 0, with `no_motion` = 1 | D25, D60 |
| Rows before the first predicted step | 16 s: 8, 4, 2 rows at Δ = 2, 4, 8 s | Vocabulary §6, item 7 |
| Time since a word | log(1 + t / 2 s) / 5, t in seconds, from the first predicted step at the earliest | D17 |
| Landings of a candidate | In the 30 min before the step | §2 |
| Configuration A | d_model 192, 4 layers, 6 heads, feed-forward 768, dropout 0.1, weight decay 0.01, learning rate 3·10⁻⁴, 500 warm-up steps, clip 1.0, 16,384 aircraft-steps a batch, at most 30 epochs, stop after 3 | D40 |
| Configurations B, C, D | B: d_model 128, 4 heads, feed-forward 512. C: d_model 256, 8 heads, feed-forward 1,024. D: A with dropout 0.2, weight decay 0.05. A head is 32 wide in each | D40 |
| Procedure masks | Glidepath lower edge: published glidepath − 60 m inside the FAF and the LPV cone; the DA wherever the aircraft is not inside them; no climb before the join, once more than ε of the level nearest the entry height below the entry height; never "unchanged" blocked | §4, D64 |
| Input scales | Distances asinh(d / 1 km); heights / 1 km; height above a glidepath asinh(h / 100 m); ground speed / 100 m/s; vertical rate / 10 m/s; landings in 30 min / 10; length / 1 km and threshold elevation MSL / 1 km (`constants`) | D41, D65 |
| RoPE base | 10,000 (heads 32 wide: periods 6.3 s to approximately 35,000 s) | D65 |
| Go-arounds of a flight in free generation | At most 2; after the second, "go-around" is masked | D68 |
| Sentences of the base, the folds and the configurations | `landed`: only sentences whose own words land; train, select and the validation's teacher-forced loss | D75 |

---

## 10 Evidence

### 10.1 The prior uses its airport embedding (R46)

[Readout](../readouts/2026-10-03_airport_embedding.zh.md). On the validation days, a replacement of the airport
embedding with the mean of the four other embeddings increases the teacher-forced loss by 0.024 (base) and 0.028
(augmented) for each step. It decreases the landed share in free generation by 5.8 and 8.0 points. Approximately 60 % of
the increase is in the heading column. The five embeddings are almost orthogonal: an identity table, not a property of
the airports. The candidate table (absolute threshold positions and courses) is a second identity: R46 §6.

### 10.2 Free generation and the heading column of the earlier prior

- Base model, validation days: per-step loss 0.2648, of which heading 0.1214 (46 %) (R46, own embedding).
- Base model free generation (validation, 400 × 4 for each airport): landed 88.7 %, timeout 6.5 %, crossed too high 3.5
  % (R46). The timeouts are mostly a flight that gets heading words and no clearance (`two_tier_stage_notes.zh.md`
  §3.2).
- Labelled heading words for each flight: straight-in 3.0, vectored 33.7 (`instruction_vocabulary_design.zh.md` §10.1).

---

## 11 Key code index

The code that this design replaces, archived by stage A unchanged.

| What | Where |
|---|---|
| Prior: airport embedding, position embedding, candidate sum (line numbers at the move) | `archive/two_tier_v3_2026_10/prior/model.py:312`, `:313`, `:382`, `:385` |
| Prior: step, relative and candidate features (line numbers at the move) | `archive/two_tier_v3_2026_10/prior/data.py:63`, `:64`, `:70` |

---

## 12 Implementation plan: stage B

**Start.** In parallel with the end of stage A, on the branch `dev-two-tier-v4-prior` (outline §5 rule 1); §0.4 gives
when each milestone starts. Until the formal artefact (A21, then A25) exists, the tests use synthetic inputs and
artefacts. Then the code is tested on a sample of the formal artefact at Δ = 2 s, read-only. The smoke artefacts of
stage A are not used: their spec is not the spec of D56 and D58 (D55). The formal runs of B5 need Claude's check of
stage A and the chosen Δ (D11). Stage B changes no code of `instructions/` or `autopilot/`; it reads them only through
the vocabulary's public interface (vocabulary §6). The rules of outline §5 apply.

**Written from this document, not patched from the archive.** The archived `prior/`
(`archive/two_tier_v3_2026_10/prior/`) stays unchanged. A part of it comes back only where its logic fits this document,
rewritten into the new modules: the row-by-row encoding with a cache (`Prior.extend`), the candidate tokens with one
shared network and the pointer head, the ordered heads, the training loop (AdamW, warm-up, clipping, early stopping),
the index of the airport's landings, the rules of the procedure masks. Nothing else comes back: not the six-column
layout, the airport and row-position embeddings, the candidate constant table, the fixed count of candidate slots, the
variants and selection rule of `instruction-v3`, the aircraft attention of a one-aircraft scene, the checkpoint formats
`v3`–`v5`. No compatibility (principle 8).

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

**B5. Cross-validation and the base** (D39, D40, D41).

- The 31 training runs of §5, as one campaign from one commit on a clean checkout, one at a time on the GPU.
- Every run reads the sentences of the selection `landed` (D75). For each fold: the held-out loss, the first-step
  runway at the held-out airport, the free generation at the held-out airport (200 flights × 2, from every flight;
  the flights outside the selection given apart).
- `prior_select` applies the rules of §5 and writes the choice. Then the base on all five airports, and its one
  validation readout: the teacher-forced loss, the free generation, the share of the labelled words that the masks
  block, the probability of "go-around" on the final (D72).
- No criterion is applied: the user reads the results (D7).

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

**B6. The Training view of stage B (outline §6).** After A23 is on this branch; the publication of the folds and the
base after B5. The user sees what the prior says and how the executor flies it.

- **Export** (the archived prior Training exports, rewritten). For each flight of the sample: the observed rows before
  the first predicted step; the sentences that the prior says in free generation and their flown states (several
  sentences of one flight side by side); the closed-loop sentence of the same flight; the outcome and the DA check of
  each; at each row, the words that the procedure masks blocked; the region, the glidepath lower edge, the DA and the
  entry height of R. Sets: each fold of B5 at its held-out airport (the flights of its free generation), and the base
  model (its one validation readout). Before B5, a smoke set from the smoke model of B3 checks the view. New schema
  names; its own index beside the old one (outline §6 item 3).
- **Frontend.** The Training view of A23 with the prior's sentences: the five columns, a choice of sentence, the
  blocked words at a row, the procedure's limits drawn, the outcome. A click on a word flies its segment live with the
  executor of A23.
- **Publication and view.** The intent of each set in `docs/experiments/intents.json`; a test stack from the worktree
  (outline §6 items 4, 5).
- **Tests.** The export (a sample written and read again); the frontend's readers on fixtures that the export writes;
  a live segment equals the export's flown states; the browser check (outline §6 item 6).

**B7. Close of stage B.** The full ts suite passes (run detached). §0.3 (the log) and `docs/reference/runners.md` are
updated; the report gives the new code index for §11 (outline §5 rule 10). `dev-two-tier-v4` merges
`dev-two-tier-v4-prior` (outline §5 rule 1). Report to the user: the commits, the readings of each fold and of the base,
the choice and its rule, and what stage C needs.
