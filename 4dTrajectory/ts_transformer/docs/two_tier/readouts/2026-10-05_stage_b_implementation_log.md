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
| Before B5: the formal-size check at Δ = 4 s | Done 2026-10-05 at `228e73be` on `v11_20261004` / `v16_20261004`, Δ = 4 s, `landed`, all five airports, in the scratchpad (the user asked for it: B3's check was at Δ = 2 s, configuration A only). The selection keeps train 39,645 (886 left out, 2.2 %), select 6,025 (174); val not shown (D85). The memory check: the largest batches 256 × 64 rows and 36 × 386 rows (the longest sentence 386 rows, 770 at Δ = 2 s), 8 candidates; GPU peak reserved 1.86 GB (A) and 2.43 GB (C) of 7.5 GB free; host 3.0–3.2 GB. Three epochs of A (2.17 M parameters) and of C: 25.3 s and 35.8 s an epoch (train and the select loss), approximately 1 min 45 s before the first epoch (the checks and the reading); the select loss per step 2.50, 1.85, 1.64 (A) and 2.25, 1.78, 1.61 (C). Claude's estimate of §5 from it, every run at most 30 epochs (`max_epochs`; patience 3 stops many earlier), a fold on four fifths of the sentences: A or D at most approximately 12 min a fold run, C approximately 16 min, B (not measured, between) approximately 14 min; steps 1–3 and the base at most approximately 7 h of GPU. Free generation on `v11` (the smoke of B4 on real data): the 3-epoch model of A, the select days, 40 flights of each airport × 2 samples, 3 min 15 s for 400 sentences (approximately 1 min of it the checks); 398 inside the selection, 2 outside; 38 of the 400 landed (an untrained model: a check that the loop runs, not a result); so approximately 3 min the free generation of one fold (200 flights × 2), approximately 1.5 h for the 30 fold runs if every run is read |
| A34's artefact (`v12_20261005` / `v17_20261005`): the formal-size check at Δ = 4 s | 2026-10-05, at `1204fd8f`, `landed`, all five airports, in the scratchpad. The selection keeps train 39,623 (907 left out), select 6,018 (181); val not shown (D85). The memory check: the largest batches 256 × 64 rows and 28 × 386 rows, 8 candidates; GPU peak reserved 1.73 GB (A) and 2.26 GB (C); host 2.9 GB. Three epochs of C: 35.7–36.1 s an epoch, the select loss per step 2.27, 1.84, 1.59; 4 min 8 s the run. Three epochs of A (from a clean tree, at `bb86a39f`): 26 s an epoch, the select loss per step 2.47, 1.89, 1.63 |
| B9 (D96): the interface for the post-training | Done 2026-10-05 (after `dev-two-tier` `f62c1634` was merged, `08b1c5e1`), reviewed twice. `1204fd8f`: the speaker's cache grows as a sentence needs (`model.Past.grown`), so free generation reads no time limit of the loop (vocabulary D90). `6a68ac67`: (4) the inputs of a loop's row as one function of `prior/` (`prior/loop.py` `LoopRows`, with `select` for a copy's rows), free generation calls it — the 1,447 rows free generation gives the speaker on A26's synthetic artefact at Δ 2 and 4 s the same before and after the move, bit for bit (a scratch check), and a test: `LoopRows` walked on a sentence's states gives `sentence_rows`'s inputs bit for bit. `0b050797`: (1) `Speaker.observe` / `speak` pass ``extra`` to the model; (2) `speak(row, at, numbers, caller, extra)`, `speaker.draw` (the first class whose cumulative probability, float64, passes the number; a number past the sum gives the last class of positive probability), the shared generator gone; free generation's numbers `flight_numbers(seed, sample, index)` (numpy `default_rng([seed, sample, index])`, five a row said), format `ts-prior-free-generation-v2`; (3) `Speaker.permitted()` → `speaker.Permitted` (`select`), `train.masked_log_probability`; (5) `Speaker.copy`, `ProcedureMasks.select`, `Heard.copy`. The stage-B fixtures written again by the export and the service. Review fixes: `bb86a39f` (a record's runway column as wide as the batch's widest, padded with blocked classes; a row asked reads the record said at its time; a narrower record padded, a wider one refused unless its extra classes are blocked; `LoopRows` reads the Δ rows from `interval_rows`), `5313b6cd` (a record holds the own-state inputs of its row and a row asked with others is refused: records of another aircraft or sentence; the speaker refuses a row not later than its last and a model in training mode, at construction too). Tests (§12 B9): 142 of stage B pass; mutants checked (another aircraft's numbers, a copy without its words heard, a record not padded). Full suite at `5313b6cd` (`./run_all_tests.sh`): 1,669 passed, 1 skipped; 1,013 passed; 161 passed. The names of §7's "Code" column, for Claude: item 2 `prior/loop.py` `LoopRows` (`select`); the speaker `prior/speaker.py` `Speaker.observe(rows, positions, extra)`, `Speaker.speak(row, at, numbers, caller, extra)`, `draw`, `Speaker.permitted()` → `Permitted` (`select`), `Speaker.copy(indices)`; the masked log-probability `prior/train.py` `masked_log_probability(model, rows, permitted, extra)`; free generation's numbers `experiments/prior_free_generation.py` `flight_numbers`, its format `ts-prior-free-generation-v2`. Claude's readings (proposals): the numbers of a flight in free generation from numpy `default_rng([seed, sample, index])`; `masked_log_probability` gives 0 at a row not asked, and an asked row reads the record of the row said at its time, its own-state inputs checked against the record's |
| B10 (D64, D72, D85, D105–D108): the corrections of Claude's check of stage B | Done 2026-10-05, `e4e7ba42` (after `dev-two-tier` with `4d64c1cd` and `9e44a687` was merged), reviewed twice. D106: (1) the step of a speaker's closed loop as one module, `experiments/prior_speaking_loop.py` `SpeakingLoop` (`observe`, `step(numbers, caller, extra)`, `end`, `said`, `states`, `generated(flights)`, `sentences(split)`, `permitted`), `flight_numbers` and `Generated` moved into it; free generation calls it — words, states, outcomes and every record the same as at `5313b6cd`, bit for bit, on 15 runs of A26's synthetic artefact at Δ 2, 4, 8 s (a scratch check; the reviewer's own: 51 runs); (2) `prior/checkpoint.py` `open_prior` (`OpenedPrior`), used by `prior_free_generation`, `prior_validation` and `prior_training_export`; (3) `SpeakingLoop.sentences`; (4) `speaker.Permitted.join`; (5) `prior/inputs.py` `motion` / `Motion` with `east_mps`, `north_mps`; (6) `Speaker.heard`, `in_force`, `go_arounds`, `n_candidates` read only (copies). D105: `LoopRows` takes one `LandingIndex` for each aircraft; `LandingIndex` holds the day split and refuses a landing on a test day or a day outside the split. D107: `prior/model.py` `require_eval` in the speaker and `train.masked_log_probability`. The speaker changes nothing before a row is said, refuses `observe` after a `speak` and a row not later than the last. D64: `ProcedureMasks.after_row` (the speaker and the validation readout's walk); `procedure-masks-v5`. D72: the runway in force before the row; `ts-prior-free-generation-v3`, `ts-prior-validation-v2`. D85: `prior_train` prints train and select only; `open_prior` reads no val outcome (the val counts compared by the val file's sha256: `artefact_identity(…, counted=)`, `load_checkpoint(…, counted=)`); the export takes a val readout only when `checkpoint.validation_claim` names it; the two val counts of this log removed. D108: `experiments/prior_behaviour.py` (training two steps, the speaker, the input functions, the `constants` variant, the first-step runway, `SpeakingLoop` on a straight-flying stand-in; CPU, one thread; 1.8–2.5 s, the same answer twice and across processes), run by `prior_campaign` at its start and before each step, its settings (seeds, selection, configurations, free generation, temperature, D68's bound) compared too, each step's commit recorded as information; `ts-prior-campaign-v2`. Tests of §12 B10 (the reviewers' scripts made tests); the stage-B frontend fixtures written by the export's own `main` (vitest 714 passed); mutants checked (D64's keep, D72's runway, a refused row). Stage B tests 172 passed; full suite at `e4e7ba42` (`./run_all_tests.sh`): 1,692 passed, 1 skipped; 1,013 passed; 161 passed. The formal-size memory of free generation (one chunk of 400 flights, a sentence to its two go-arounds: the cache grown to 1,056 rows): GPU peak reserved 4.16 GB (A) and 5.55 GB (C) of 8 GB. The smoke on A34's artefact again (`v12`/`v17`, Δ = 4 s, `landed`, 200 sentences, 3 epochs; free generation of 40 flights of each airport × 2 on select): both exit 0, 400 sentences in 4 min 53 s (an untrained model: no result). The campaign smoke (`--smoke 50`, before B10, 2 h 37 min) ran every step to the base and its readouts. The step's copy for chosen flights (D106 item 1) waits for A38's `Loop.copy`; D109 for A39. Claude's readings (proposals): a flight the caller ends has the caller's outcome (`generated` refuses it, `said` and `states` give its sentence); the behaviour check reads the artefact's spec and first airport's finals as its fixed inputs; the loop keeps each row's inputs for `sentences` (~150 MB a chunk at Δ = 4 s) |
| Stage A's line merged (A37–A41), D109, B11 (D111) | Done 2026-10-05 (notes/stage_b.md, the order after B10). `c072e355`: `dev-two-tier-v4` `ed2530ae` (A37–A40) merged (the conflict in `trainingSample.ts`: `parseFlight` takes `closedIntervals` and A39's `splits`); A37 followed — a flight's timeout from the judge's outcome (`judge.TIMEOUT`) in `SpeakingLoop.generated` and the export's `fly_again`, the export's runway offset from `training_export.candidate_hae_minus_msl_m`; A38 — `SpeakingLoop.copy(flights)` on `Loop.copy` (tested: a copy flown with other numbers leaves the original as a loop never copied; one flown with the same numbers says its words and flies its states within the executor's bound); reviewed. `1087c6bd`, `c58d7772`, `89845fa6`: `dev-two-tier-v4` `706aa2bd` (A41) and `dev-two-tier` merged, no conflict; stage A and stage B fit — every name stage B reads exists, stage A's readers do not refuse val (only its runners narrow `--split`), `split_flights` takes any split; stage B tests 173 passed on the merged code. `df68c927` (reviewed twice): D109 — every set's source names the claim of the val read it was exported under (`validationClaim`, null but for the base's one validation readout); `aeroviz-training-prior-index-v2` / `-sample-v2`, `index_prior_v2.json`; the frontend reader (a sonnet agent) and the backend's live segment give a claimed set val alone and every other set stage A's splits, check the claim's reader (`prior.training_files.CLAIM_READER`), prior and readout, the backend the claim on disk, before the executor check; the claim names its readout relative to the repository (`checkpoint.validation_claim`, `holds_claim`); stage A's files unchanged (a change of `backend.py` by the agent reverted: a second stage A service with `splits=SEALED_READINGS` instead). B11 — `selection.left_out`, `kept(rule, outcome, faulty)`, `side`, `SIDES`; the record's cells `kept` / `left_out_fault` / `left_out_outcome`; the marks from `instructions.faults.faulty_flights`; `ts-prior-checkpoint-v8` (the identity also holds each split's signals sha256: the val marks bound before the claim); the readouts by side (`ts-prior-free-generation-v4`, `ts-prior-validation-v3`); D85 — readouts write `checkpoint.readable_identity` (no val count), the val readers recount val's selection after the claim (`source.require_selection_of`). Tests: stage B 183 passed; vitest 722; mutants checked (the fault mark ignored). Full suite at `89845fa6`: 1,725 passed, 1 skipped; 1,023 passed; 161 passed. The smoke on A34's artefact again (200 sentences, 3 epochs; free generation 40 flights × 2 on select): `landed` keeps train 39,154 (498 left out for a faulty track, 878 by their outcome), select 5,861 (177, 161); free generation 400 sentences — inside 384, outside for a faulty track 12, by the outcome 4 (an untrained model: no result); the memory of free generation at the formal size again 4.16 GB (A) and 5.55 GB (C). Claude's readings (proposals): a claimed set flies on a second stage A service (`splits=SEALED_READINGS`); the claim's names relative to the repository; a sentence of a marked flight that did not land is counted as left out for its fault; the val readers' recount after the claim kept as a last guard |
| D118 and A42 (notes/stage_b.md, the order after B11) | Done 2026-10-05. `0a8ad371`: `dev-two-tier` `931f4e5d` merged; §7's column "Code" checked against the code: every name and signature it gives exists. `1ec6980f`, `c7a105ab`: `dev-two-tier-v4` `06b8fde1` (A42: the synthetic artefact of `tests/support.py` records `runway_ends_from`) and `dev-two-tier` `65c77ecd` merged, no conflict. `5818fbbf` (reviewed): the stand-in `with_runway_ends` of `tests/test_prior_training_export.py` deleted; stage B's hook comment in `backend.py` cites `prior.training_files.INDEX_FILE` (no file version). Tests: `test_prior_training_export.py` 12 passed, `aeroviz_backend/tests/test_prior_segment.py` 11 passed; the frontend fixtures unchanged. B5 waits for the user's order |
| A43 and B12 (D108, D119–D122; notes/stage_b.md, the order while B5 runs) | Done 2026-10-05/06 on `dev-two-tier-v4-prior`, not merged into `dev-two-tier` while B5 runs. `d559e970`: `dev-two-tier-v4` `60bbc901` (A43) merged (the conflict in `backend.py`: A43's file and stage B's three hook lines); `1777b3ec` (reviewed): the prior warm-up opens each set under its own lock, never the request lock (tested), the validation service made once under a lock, the split comment "before the set is opened (A37)". Not done: `fly.refuse_past_bound` in `apart_from_exported` — it compares with the export's track written to 0.1 m, so the 1e-6 m bound refused every answer (0.045 m in the test); in the requests note. `39d02ef8` (B12, reviewed twice): D119 (a claim spent when its readout.json is written, recorded in the claim; run again to its own output only; options checked before the claim; the export and the backend take a written val readout only); D108 (the settings from the behaviour check's process; the answer covers `sentence_rows`, the selection, `prior_select`'s rules; `--chunk` passed; `ts-prior-campaign-v3`); the speaker's two refusals and `accept`; `SpeakingLoop` on the start's observed rows (free generation with `NO_MOVE`, bit for bit: the stage-B frontend fixtures unchanged), a refused row leaves the loop as it was; `LoopRows` one first step; `prior_select`'s refusals; the tests of §12 B12. Tests: stage B, `test_architecture`, `test_start` and the hook's backend tests 213 passed (CPU; the full suite after B5). Claude's readings (proposals): the claim's spent mark `spent_utc` in `val_read_<reader>.json`; the executor's step inside the speaker's `accept` |
| B5 restarted on B12; B13 (D108, D127, D128) | 2026-10-05 22:20Z the user stopped B5 (`prior_base_20261005`, step 17; its folders deleted at the user's word) and had B12 merged: `dev-two-tier` = `dev-two-tier-v4` = this branch at `645e4cf0` (`dev-two-tier-merge-a43` merged, no file changed), intents `278ffcfc`; full suite at `2190fe0a`: ts 1,738 passed 1 skipped, modeling and backend 1,030 (+138 subtests), frontend Python 161. B5 relaunched 22:40Z as `prior_base_20261006` (`ts-prior-campaign-v3`, the main tree). `142272f0` (B13, reviewed twice), on this branch, not merged while B5 runs: D128 (the claim records the reader's options and is written whole under the read's lock, `lock_val_read`; a written readout under an unspent claim is marked spent, by the readers and by the campaign's `settle_val_steps`; a negative seed or a repeated airport refused before the claim), D108 (the behaviour check's landings counted, each loop aircraft its own; `kept`, `side`, the campaign's plan and free generation's draw in the answer), D127 (the prior sentence's track unrounded, `aeroviz-training-prior-sample-v3`, the frontend mirror and fixtures by the export; the backend refuses past the executor's bound or at another outcome or end cycle; the export refuses a readout without readout.json before reading a file). Tests: stage B, `test_architecture`, `test_start` and the hook's backend tests 220 passed; vitest of stage B 29 passed (CPU; the full suite after B5). Decided by the user, 2026-10-06: the claim's options also hold the device (both readers) and free generation's temperature and bound of go-arounds (kept); an error found only after the claim (an airport with no val sentence) locks that val read for good (accepted) |
| B5 done; B13 merged; B6 published (2026-10-06) | B5 (`prior_base_20261006`) ended 07:12Z, every step done: configuration C (score 1.3601; the seed scale 0.0022, only C within twice it), variant `full` (`constants` 1.3731); the base (C, `full`, five airports, best epoch 22 of 25): its one val readout, teacher-forced 1.0843 per step; free generation on the val days, inside the selection 1,792 of 1,894 landed (94.6 %); both claims spent. Data read-only with `SHA256SUMS` (382 files, 439 MB); the readout `readouts/2026-10-06_b5_campaign.zh.md` (`e17aa643`, a sonnet agent). B13 merged: `dev-two-tier` = `dev-two-tier-v4` = this branch at `d14a2f76`; full suite: ts 1,742 passed 1 skipped, modeling and backend 1,033, frontend Python 161. B6's publication (intents `prior_sets_20261006`, `493dbe2b`): the chosen configuration's five folds at their held-out airports (`prior_fold_C_<airport>_20261006`, select) and the base's val readout at the five airports (`prior_base_val_20261006`, claimed), 10 flights × 2 sentences a set, `aeroviz-training-prior-sample-v3`, into `aeroviz-4d/public/data/airports/<airport>/training/` (its own `index_prior_v2.json`). The dev server was restarted (it served the base set, written after its start, as HTML). Checked: the browser (a one-shot subagent: the sets open, sentences shown, a word flown at KRDU); the live endpoint of the full stack's backend (port 8765) on 40 sentences (two flights of each set, both samples, each flown from its last word to the judge's outcome): every answer within 4e-9 m of the written track and of its outcome and end cycle |
| B7: the close of stage B (2026-10-06) | `docs/reference/runners.md` R56–R62 (`prior_train`, `prior_free_generation`, `prior_validation`, `prior_select`, `prior_campaign`, `prior_behaviour`, `prior_training_export`; the v3 entries R13, R15, R17 stay archived) and their index line in `ts_transformer/CLAUDE.md`; the full suite passed at `d14a2f76` (the code since unchanged); `dev-two-tier-v4` and `dev-two-tier-v4-prior` at `dev-two-tier`; the code index for §11 and what stage C needs: requests note §3 items 4 and 5. Open: the Training view of stages A, B and C (requests note §3 item 3, the user's decisions of 2026-10-06) |
| Vocabulary D90 and A37 (stage A) | D90 narrowed (`f728e704`, the user, 2026-10-05): `Loop.executor` stays public, so the gap reported earlier is closed (free generation and the export read a flight's end cycle `executor.done_cycle`, its flown record `executor.flown()` and its aero parameters `executor.inputs.aero_params` through it); A37 is being changed again (the user). After its merge stage B follows: the closed-loop reading's `timed_out` from the judge's outcome (`judge.TIMEOUT`), as D90 says (the user confirmed, 2026-10-05), in free generation (`experiments/prior_free_generation.py` `speak_and_fly`) and the export's check (`experiments/prior_training_export.py` `fly_again`); A23's `training_files.runway_hae_minus_msl_m` removed |

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

## 3 Specifications of the milestones that are done (was §12: B0–B13)

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


**B9. The interface for the post-training** (§4, §7; D96). Moved here from the design on 2026-10-05, after B9 was done.
Now; before B5's formal campaign, because it changes the draws of free generation, and a payload that an experiment
writes is settled before the experiment runs.

- The six parts of D96 in `prior/` and `experiments/prior_free_generation.py`: the speaker's input of the added
  modules, its random numbers (§4 "Drawing"), its records of the permitted words and its copy of chosen aircraft; the
  log-probability of given words under records; the function of a loop's row, moved out of `speak_and_fly` into
  `prior/`; the report gives the names of the new parts for the column "Code" of §7.
- Free generation: each flight's random numbers from the seed and the flight; a new format name for its files; B6's
  export and its smoke set follow the new name.
- Tests: an aircraft's words with given numbers are the same when other aircraft join its batch (on a synthetic batch;
  a difference only where a number falls within the float tolerance of a boundary, counted); the same numbers give the
  same sentence; the log-probability under the records equals the probability that the speaker drew the word from;
  a word that a record blocks has probability 0; a copy continued with the same inputs and numbers says what the
  original says, bit for bit, in a batch of the same layout; a module that reads the input of item 5 gets it from the
  speaker, and a module whose output is zero changes no word; free generation through the moved function of a loop's
  row gives the same inputs, bit for bit, as before the move.

**B10. The corrections of Claude's check of stage B** Moved here from the design on 2026-10-05, after B10 was done (`e4e7ba42`; its item of outline D109 in `df68c927`). (`readouts/2026-10-05_stage_b_check.zh.md`; D64, D72, D85,
D105–D108). Now; before B5's formal campaign, because it changes free generation and the campaign runner. On synthetic
artefacts; then B3's smoke and free generation on A34's artefact again. The report gives the names of the new parts
for the column "Code" of §7.

- D85: no printed text, log or summary of the prior shows a reading of the val days before the base's one validation
  readout: `prior_train` prints the selection of train and select only (the identity keeps every split, D75). A
  validation readout of the base is exported only when it is the one that the claim of the val read names (its output
  and the base's run).
- D105: the function of a loop's row takes each aircraft's landings; `Landing` and the construction of a
  `LandingIndex` from given landings are public; the index refuses a landing on a sealed test day.
- D106: (1) the step of a speaker's closed loop as one module under `experiments/`, with `flight_numbers` moved into it;
  free generation calls it; (2) the function that opens a prior run, used by every runner of the prior; (3) the rows of
  a sentence a loop said as one batch with its targets; (4) the join of records; (5) the motion of rows in §7; (6) the
  speaker's state named in §7.
- D107: the speaker and the log-probability under records refuse a model of which any module is in training mode (not
  only the top module).
- D108: the campaign's behaviour check before each step in place of the comparison of the commit.
- The speaker changes nothing before a row is said (as vocabulary D80 for the start): the state of the procedure
  masks, the cache and the records change only after every column of the row has a word; `observe` is refused after
  the first `speak`; both refuse a row not later than the last row encoded, observed or said.
- D64: at the row whose runway word ends G, the procedure masks keep the row's state (G is false after that word):
  the join and the passage below the entry height start again at that row, not one row later.
- D72: "on the final" is the region of the runway in force before the row, the runway under which the speaker drew
  the word.
- Free generation's memory at the formal size (outline §5 rule 13): a chunk in which one sentence reaches the time of
  its go-arounds (the cache of every aircraft of the chunk grows with it), measured on the GPU before B5.
- Tests: a change of the observed samples after the first predicted step and of the time limit changes no input of a
  loop's row (vocabulary D90); D23's test also changes the flight's record (its runway and landing time) and its own
  landing in the index; in free generation, a change of every field that it must not read (the stored flown states,
  the words, the correction marks, the withheld fields of vocabulary D82, the landing time, the own landing) changes
  no word, state or record of a flight (the readout itself reads the stratum, the stored outcome and the mark, D75, D111); the loop's rows equal the sentence's rows at Δ = 8 s too; a smoke of the validation
  readout reads no outcome of a val sentence (a test that fails on such a read); the export of a val readout (refused
  unless claimed); the frontend's fixtures of stage B written by the export, the index by its own writer; a refused row
  leaves the speaker as it was; a module in training mode inside an eval model is refused; free generation through the
  shared step gives the same words, states and readout, bit for bit, as before the move.
- Outline D109, after vocabulary A39 is on this branch: the prior's sets give the Training view's reader and live
  segment their splits; a set exported from the base's claimed validation readout gives val too, and only it. Tests: a
  val set of a claimed readout opens and flies live; any other set with a val flight is refused.

**B11. The selection leaves out flights with a faulty observed track (D111).** Moved here from the design on 2026-10-05, after B11 was done (`df68c927`). After vocabulary A40 is on the stage A
line that stage B merges; before B5's formal campaign. `prior/selection.py`: `landed` also leaves out a flight that
stage A's `instructions/faults.py` marks; the selection record counts them apart (by split, airport and stratum). The
change of the record is a change of the identity of a run's data (the checkpoint's schema; one new name with B10's
changes, as both come before B5). The mark is read like the stored outcome (D75): it keeps or leaves a sentence and
never reaches an input. Of the val days, the marks are counted in the identity and never printed (D85, as B10). Free
generation's readout gives the flights left out for a faulty track apart from those left out by their outcome (a
flight with a faulty start fails whatever is said: vocabulary D111). The report gives the selection's names for §7
item 4 if they change. Tests: a marked flight that landed is left out under `landed` and kept under `all`; the record
counts it apart; a change of a flight's mark changes no input of its rows; a marked flight's free generation is
unchanged; the readout counts the two reasons apart.

**B12. The corrections of Claude's second check of stage B** Moved here from the design on 2026-10-06, after B12 was done (`39d02ef8`, merged into `dev-two-tier` at `2190fe0a`). (`readouts/2026-10-05_stage_b_check_2.zh.md`; D108,
D119–D122). On stage B's branch, on synthetic artefacts; merged into `dev-two-tier` only after B5 has ended: its
behaviour check gives another answer, so a campaign started before it stops by name under it (§0.4). The report gives
the changed names for §7 items 2, 3 and 7.

- D119: a claim of the val read whose output holds no written readout may be run again to the same output (both
  readers, `prior_validation` and `prior_free_generation`); free generation checks its options (`--airports`) before
  it claims. The export and the backend take a val readout only when it is written and claimed.
- D108's guard (D118 item 7): the campaign's settings (the seeds, the selection, the configurations, free generation,
  the temperature, D68's bound) come from the code on the disk — the process of the behaviour check gives them in its
  answer —, not from the campaign's own process. The behaviour check also covers `inputs.sentence_rows` on a fixed
  synthetic sentence (with a flight key of the real format), the selection (`kept`, `left_out`) over every rule ×
  outcome × mark, and `prior_select`'s rules on a fixed table of scores (a tie of parameters, a score at exactly twice
  the seed scale, a seed scale of 0). The campaign's format gets a new name.
- The speaker (§7 item 3): a row whose mark of the first predicted step differs, for an aircraft, from "no word in
  force" is refused; `observe` refuses positions that are not one for each row and each aircraft; both before any
  change.
- The step of a speaker's closed loop (§7 item 7): `SpeakingLoop` takes the observed rows before the first predicted
  step as the start gives them back (`start_moved`'s; `NO_MOVE` is `start` bit for bit) and refuses rows of another
  shape; of a sentence it reads only its first row and its first predicted step. Free generation gives it the rows of
  `start_moved` with `NO_MOVE`: its words, states and readout stay the same, bit for bit (a test). A row that the
  executor refuses (vocabulary D80) leaves the speaker, the loop's records and its row as they were.
- `LoopRows` takes one first predicted step for the batch.
- `prior_select` refuses a score that is not finite, by name; it checks every training value of a fold against its
  arm (the learning rate, the weight decay, the patience, the epochs, the warm-up, the clip, the padded rows of a batch,
  the RoPE base); the campaign's selection rule has one definition.
- D121, D122: no change of the code; a test holds D122 (a change of runway outside a go-around keeps the join and the
  passage of the candidate).
- `prior/inputs.py`'s docstring and `tests/test_prior_inputs.py` name what the motion must not read as §2 does.
- Tests: the D23 test and free generation's test of the fields it must not read, with a flight key of the real format
  (`<airport>:<id>_<runway>_<icao24>_<landing time>`), its runway and time changed with the own landing; free
  generation's test gives the changed sentences to the start too; `prior_train` prints train and select only (its
  output tested); the val readers refuse every read of val before the claim (the sentences, the stored outcomes, the
  faulty-track marks); a fold through the runner: a change of the held-out airport's data leaves the checkpoint the
  same; the choice at its edges; the campaign's settings from a second process; the speaker's two refusals; a moved
  start's rows reach the inputs of the first predicted step.

**B13. The corrections of Claude's check of B12** Moved here from the design on 2026-10-06, after B13 was done (`142272f0`, on `dev-two-tier-v4-prior`). (`readouts/2026-10-06_stage_b_b12_check.zh.md`; D108, D127,
D128). On stage B's branch, on synthetic artefacts; merged into `dev-two-tier` only after B5 has ended: its behaviour
check gives another answer, so B5 would stop under it. It changes no input, model, training, draw or choice, so B5's
results stay valid. The report gives the changed names for §7.

- D128: the claim records the reader's options (free generation: its airports, flights of each airport, samples, seed
  and chunk; the validation readout: its split); a rerun to the same output with other options is refused by name; a
  runner that finds its output with its readout written and the claim not marked spent marks it spent, then refuses;
  the claim file is written whole (a temporary file linked to its name) and held under an exclusive lock while the
  reader runs. Free generation refuses a negative seed and a repeated airport before it claims. A spent claim without
  options (B5's, written by B12's code) is read for its output only.
- D108's guard: in the behaviour check, the fixed sentence has landings in the 30 min before its rows on two or more
  candidates, its own landing between two of its rows and another landing at the same second on another runway; the
  speaking loop's aircraft each have their own landing index, with landings before their rows; the answer also holds
  `kept` and `side` over every rule × outcome × mark, the campaign's plan of steps for a fixed record (each step's
  runner and arguments) and free generation's draw of flights from a fixed set. On a resume, the campaign's refusal
  names what differs from its start record. The comment of the choice's fixed table says which configurations are
  within.
- D127: the export writes each prior sentence's flown track unrounded (its sample format gets a new name; the
  frontend's reader and fixtures follow); the backend's `apart_from_exported` calls stage A's
  `fly.refuse_past_bound(…, "the readout's flown states")` and refuses an outcome or an end cycle that differs from the
  sentence's. The export calls `checkpoint.written_claim` (one definition of a written claim) and refuses by name a val
  output without its readout before it reads any of its files.
- Tests: a kill between the readout and the spent mark (both readers); free generation's val path to its end on a
  synthetic artefact with val days (the spent mark, a rerun after a crash with the same and with other options, the
  recount after the claim); a second run to one output refused while the first holds the lock; the backend against a
  real claim file (no stand-in for `written_claim`); the val set opened once by two threads; D127's refusals (a state
  past the bound, another outcome, another end cycle); `LoopRows`' refusal of a start that is not one int; a change of
  each new part of the behaviour answer changes the answer.
- Not ordered: tying the observed rows to the `Loop` that the start returns (a change of stage A's start); stage C
  passes them as the start gives them.

**B5. Cross-validation and the base** Moved here from the design on 2026-10-06, after B5 was done (`prior_base_20261006`). (D39, D40, D41, D75, D108).

- The 31 training runs of §5 as one campaign (`prior_campaign`) on a clean checkout, one at a time on the GPU: each
  fold's training and then its free generation at its held-out airport; after step 2 the choice of the configuration,
  after step 3 the choice of the variant (`prior_select`, the rules of §5, written as a choice); then the base and its
  one validation readout (`prior_validation`, then free generation on the val days). 65 steps in all.
- Before each step the behaviour check of D108; the commit of each step recorded as information. A step whose output
  lacks its last file (a crash, a kill) is moved aside and run again; a run is never repeated otherwise.
- Every run reads the sentences of the selection `landed` (D75, D111). For each fold: the held-out loss, the first-step
  runway at the held-out airport, the free generation at the held-out airport (§5).
- No criterion is applied: the user reads the results (D7).

**B6. The Training view of stage B (outline §6).** Moved here from the design on 2026-10-06, after B6's publication (`prior_sets_20261006`). The user sees what the prior says and how the executor flies it.

- **Export** (`prior_training_export`). For each flight of the sample: the observed rows before the first predicted
  step; the sentences that the prior says in free generation and their flown states (several sentences of one flight
  side by side); the closed-loop sentence of the same flight; the outcome and the DA check of each; at each row, the
  words that the procedure masks blocked; the region, the glidepath lower edge, the DA and the entry height of R. Every
  prior sentence is flown again and must equal its readout's states within the executor's bound; its track is written
  unrounded (D127). Sets: each fold of B5
  at its held-out airport (the flights of its free generation), and the base model (its one validation readout, only
  when claimed, outline D109). Its own schema names and its own index beside stage A's (outline §6 item 3).
- **Frontend.** The Training view of stage A with the prior's sentences: the five columns, a choice of sentence, the
  blocked words at a row, the procedure's limits drawn, the outcome. A click on a word flies its segment live with the
  executor of stage A, refused past the executor's bound from the exported track or with another outcome (D127); the base's val set opens and flies live, and no other set with a val flight opens (outline
  D109).
- **Publication and view** (after B5). The intent of each set in `docs/experiments/intents.json`; a test stack from the
  worktree; the browser check (outline §6 items 4–6).
- **Tests.** The export (a sample written and read again); the frontend's readers on fixtures that the export writes;
  a live segment equals the export's flown states.

**B7. Close of stage B.** Moved here from the design on 2026-10-06, after B7 was done (`d1386a77`). The full ts suite passes (run detached). The implementation log and
`docs/reference/runners.md` are updated (the runners `prior_train`, `prior_select`, `prior_free_generation`,
`prior_validation`, `prior_campaign`, `prior_behaviour`, `prior_training_export`); the report gives the code index for
§11 (outline §5 rule 10). `dev-two-tier-v4` merges `dev-two-tier-v4-prior` (outline §5 rule 1). Report to the user: the
commits, the readings of each fold and of the base, the choice and its rule, and what stage C needs.

## 4 The Training view's one layout (outline §6.2, D133)

Built by stage B's implementer on `dev-training-layout` (from `docs-training-view` `4fa9f2ee`, which holds D133; `dev-two-tier` merged in, with D133 as `b2911b89`), worktree
`.claude/worktrees/training-layout`, at the user's order of 2026-10-06 (notes/stage_b.md); for this work it changed the
view files of stages A and C and the backend's routes.

| Part | State |
|---|---|
| The layout (§6.2 items 1–5) | `38297528` (reviewed twice): the left panel the same in every stage — the stage switch; the set chooser with the set's own intent line and a SMOKE tag, opening the details page on "The set and the experiment"; the item list with four columns; one line per readout; the Draw box; no table (stage B keeps "At the cursor" and its probability strip). The sentence bar's tabs choose the sentence in every stage (one state, `data/trainingTabs.ts`: the session gives the tabs, the bar chooses; a dot in the outcome's colour; the arrow keys, Home and End); the bar's chips and notes follow the kind of sentence (closed loop, prior sample, round, labelled); the notes' first line is the set and its intent line, with a link to the details page. The details page in every stage, its ⓘ never disabled (a page with the reason when no session is on screen), its first section built from the set's index entry and intent. Backend `GET /experiments/intent?run=<set id>` (one campaign; none 404 or several 409 named; read at each request); the frontend shows it in the experiments picker's form (`ExperimentIntentBlock`, extracted from `ExperimentDetails.tsx` unchanged). Shared parts: `hooks/useTrainingSet.ts`, `training/SetParts.tsx` (`SetChooser`, `ItemList`, `ExperimentSection`), `PanelParts.tsx` (`useDetailsPage`, `requestTrainingDetails`, `DrawBox`) |
| Short tab labels | `e0aceffd` (reviewed; the user, 2026-10-06): stage B's samples by their number (0, 1 …), stage C's rounds r1, r2 …; the tooltips keep the full names |
| Tests | Vitest 756 (13 new: the tabs' state and keys, the intent's answers read from the fixture the backend test writes, the one loader, the details page's requests, the layout in stages B and C, the bar's chips and notes by kind, the short labels); every test of the three stages kept, adapted where it clicked a left-panel table. The backend: `test_http_server.py` (one campaign, none, several; the route pinned; `fixtures/training_intent/answers.json` written by it). Full suite on the branch: ts 1,840 passed 1 skipped, modeling and backend 1,042, frontend Python 161 |
| The check | A test stack from the worktree (vite 5185, backend 8795, a scratch airports root: the live data with stage C's smoke set `c11-smoke-windows` at KRDU, copied in by links). A one-shot browser agent, all three stages at KRDU: the tabs choose the sentence and the chips follow; every ⓘ and readout line opens the details page; the intent shows (stage C's smoke set says by name that it has none); a word flies live in each stage (A Δ 4 s, B sample 0, C round 0) |
| Proposals (where §6.2 says nothing) | In stage B's requests note §1 |

