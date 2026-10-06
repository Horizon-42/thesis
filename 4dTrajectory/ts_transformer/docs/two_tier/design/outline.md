# Two-tier model: outline

**Summary.** The two-tier model has two parts. The upper part is the prior: a language model of the controller. At each
row it says one row of words for each aircraft. The lower part is the executor: an autopilot that flies only these
words with point-mass dynamics. Between them is the vocabulary, the language of the words, and the labeller that reads
words from observed tracks. The post-training trains the prior further in closed loop, in windows of recorded traffic.
This outline gives the documents of the design, what passes between them, the principles and rules that all of them
share, and the plan.

**Language.** This document uses the writing rules of ASD-STE100 (Simplified Technical English): short sentences, active
voice, one meaning for each word, and technical names that the terms define. The words were not checked one by one
against the STE dictionary. Units are SI only (m, m/s, deg, s). Values in feet, knots or nautical miles occur only in
quotations of regulations, with the SI value given one time.

Paths in backticks are relative to `4dTrajectory/ts_transformer/` unless they start with `4dTrajectory/` or name the
repository root; `readouts/` is `docs/two_tier/readouts/`. The documents of the earlier design (`*_design.zh.md`,
`two_tier_framework.zh.md`, `two_tier_stage_notes.zh.md`) are in `archive/two_tier_v3_2026_10/docs/`.

---

## 1 The documents

| Document | Part | Stage | State |
|---|---|---|---|
| `vocabulary.md` | The words, the labeller (open-loop and closed-loop reading), the executor, the judge | A | A0–A43 done and merged into `dev-two-tier`; the formal artefact `v12_20261005` / `v17_20261005` and the Training view (`closed_loop_v12_20261005`) done, the superseded artefacts deleted; Claude's check of A32–A40 done; no milestone open; the replay of the val days waits for the user (vocabulary §0.4) |
| `prior.md` | The prior: inputs, outputs, decoding and its masks, training, a new airport | B | Closed (B0–B13 done, B7 on 2026-10-06): B5's campaign `prior_base_20261006` done (configuration C, variant `full`, the base and its one validation readout); B6's sets published (`prior_sets_20261006`); `dev-two-tier` = `dev-two-tier-v4` = `dev-two-tier-v4-prior` (the code at `d14a2f76`) |
| `post_training.md` | The post-training in windows of traffic with one commanded aircraft | C | Merged into `dev-two-tier` (2026-10-06); C0–C9 and C11 (the window view) done (C8 on B5's base); C10 and its validation readout built, the formal run waits for O13 and the user's criteria; C12 left. Further work on `dev-two-tier-v4` (§5 rule 1) |
| `multi_control.md` | The multi-aircraft control: every arrival of a window's span commanded; the window's reward and its credit by one varied aircraft; the model in two versions (the other aircraft's words in force, then their intent); what it needs of the other stages' public interfaces | D | Written and decided 2026-10-06 (D140–D153); nothing built; MC0 and MC1 next (§4 item 7), its formal runs after C10's chosen round |
| `frontend.md` | The Training view of every stage: its rules, the one layout, each stage's view (stage D's: stage C's window view with several commanded aircraft), the shared controls (the cursor slider), the backend's Training routes, the content and formats of the Training sets | — (fronter) | Written 2026-10-06 (moved from this outline's §6, D133–D136 and D109 with it; D154–D156 new). The one layout built on `dev-two-tier-v4`, its steps after C10 stage D's implementer's; F1–F4 not started |
| This outline | The principles, the shared rules, the plan | — | Holds the principles, the shared rules (§5) and the plan (§4); the Training view moved to `frontend.md` (2026-10-06) |

How a stage is reviewed against its design (leaks, what each consumer may read, the procedure, a checklist for each
stage) is in the review guide, `docs/two_tier/review_guide.md`; it is not a design document.

**Dependencies.** The documents depend on each other as the code does. The vocabulary reads no other document. The
prior reads the vocabulary. The post-training reads the vocabulary and the prior. The multi-aircraft control reads the vocabulary, the prior and the
post-training (its public interface, multi-aircraft control D149). The Training view (`frontend.md`) reads each
stage's public interface for its Training export. Each document reads another only
through that document's public interface and its decisions:

| Public interface | What it gives | Read by |
|---|---|---|
| Vocabulary §6 | The vocabulary spec, the grammar, the sentence artefact and the stored signals, the candidates and their geometry, the executor and the start of a closed loop, the judge, the row grid, the Training export | Prior, post-training, multi-aircraft control, Training view |
| Prior §7 | The checkpoint, the inputs of a row, the speaker, the teacher-forced loss, a place for an added module, the region of a final, the step of a speaker's closed loop, the Training export's procedure block; with the code of each item | Post-training, multi-aircraft control, Training view |
| Post-training §9 | The scene and a window, the landings of a window, the separation judge's inputs and the losses, the tokens, the speed-word mask, the reward, the traffic attention, the window loop, branch training, the loss, the campaign's steps, the checkpoint, the window sets; with the code of each item | Multi-aircraft control, Training view |

A document never cites another document's other sections; every document reads this outline. A change of a public
interface is a change of a format: it gets a new name (principle 8), and every document that reads it changes with it.
In the code: `instructions/` imports neither `autopilot/` nor `prior/`; `autopilot/` and `prior/` import `instructions/`
and not each other; the runners join them (`tests/test_architecture.py`).

**Decision numbers.** The decisions have one numbering for all documents, so that the code can cite a decision by its
number. A decision is in the document that owns it. A decision with parts in two documents (D14, D25, D31, D58) is in
both, each with its part. A new decision or open item takes the next free number of the index (§3.2).

---

## 2 Principles

1. **The words are the only interface.** The prior gives no control values. The executor reads the words, the dynamics
   of the aircraft and the runway geometry of the vocabulary (vocabulary §6, item 5). The executor reads no procedure
   data.
2. **The model decides the path.** No executor law turns onto the final, follows the centreline or follows the
   glidepath. The model must say the words that do these things (D2, D3). This is the user's rule of 2026-09-16: a
   track that the procedure computes is not a skill of the model.
3. **Procedures are masks and judgements only.** The procedure data (glidepath, decision altitude, final approach
   segment) blocks words during decoding (masks) and judges the flown track (judge, evaluation). It does not make a
   track.
4. **Airport facts are inputs, not words and not identities.** A word has the same meaning at all airports. Facts
   of the airport (runway geometry, landings before now) go into the inputs as measured values. No input is a code
   for the airport.
5. **One frame for the words of direction.** Heading words are relative to the course of the runway in force (D8).
   The same heading word means "downwind" or "final" at all airports.
6. **Each instruction has a target and an envelope.** The labeller, the judge and the display use the same envelope
   code (`instructions/envelope.py`). A containment rate is always given with the width of its envelope.
7. **Inputs give only what a controller knows before the step.** Labelled words are targets of the prior, never
   inputs, except the words said before the step.
8. **No compatibility.** A changed word, spec or payload gets a new name. The code refuses an artefact of another name.

---

## 3 Decisions

### 3.1 Shared decisions

| # | Item | State | Source |
|---|---|---|---|
| D7 | The design documents set no test criteria and no measurement criteria. The user sets them after the design is settled | Decided | User, 2026-10-03 |
| D20 | This design is a new version, on its own branch and worktree. It reads no artefact, executor spec or prior that an earlier version made (`v1`–`v6`). The running experiments keep their own checkouts | Decided | User, 2026-10-03 |
| D21 | Identities bind format and data rules only. A code identity is a behaviour check on fixed inputs, never a hash of source bytes; data are identified by their flights, never by the bytes of a manifest (vocabulary §7.2, prior §8, post-training §4) | Decided | User, 2026-10-03 |
| D85 | A readout that serves a choice of the user reports the train and select days only. Where a build needs the validation days (the closed-loop sentences of val), it writes them, but no report, summary or printed text shows a reading of them before the stage's one validation readout. Why: the user chose Δ = 4 s (vocabulary D11) on a report that showed the val rows of the readings of D34 (`readouts/2026-10-04_stage_a_a25_report.zh.md`); the train and select rows give the same order, but the validation days are read one time for each stage | Decided | User, 2026-10-05, on Claude's review of stage A |
| D95 | Stage C is developed in parallel with the end of stage B (the user, 2026-10-05), on its own branch and worktree (§5 rule 1), by its own implementer; Claude writes its design and runs the project. Stage C never changes the code of `instructions/`, `autopilot/` or `prior/`. What it needs of them is a change of their public interfaces, made by their own stages (vocabulary D97, prior D96): code only, so that nothing of stage A is built again (the user, 2026-10-05); the changes of stage B before B5's formal campaign, because they change the draws of free generation. The formal runs of stage C wait for B5's base | Decided | User, 2026-10-05 |
| D55 | A value that a runner fits from data and the user chooses (D15) is measured on all train days, in a scratch directory, directly after the milestone that writes the runner; the user chooses before a later milestone reads the value. A smoke build uses the chosen spec, and its flights are a random sample for each airport and split (seed 1337), not the first flights of the sorted flight keys (a key starts with the callsign, so the first flights are mostly one airline). Why: the smoke of stage A fitted its own spec on approximately 400 train flights, 373 of them one airline, with one climb piece; every smoke reading of A9–A16 used it (§5 rules 7 and 12) | Decided | User, 2026-10-04 |
| D131 | Every review of a stage follows one standard, the same in every round (`review_guide.md` §3 step 6 and "Later rounds"): each finding gets a severity. S1 (a leak into an input, a target selection or a choice; a split violation; a second read of the validation days; a defect that changes a number the user reads, on real data, beyond its tolerance) is corrected before any formal run. S2 (a latent leak, a boundary that does not refuse, a decided rule with no test, an effect not measured and not bounded) is corrected before the formal run when the correction is cheap; else it is listed as an open item of the stage's design document (§0.2, an O number), with no milestone and no order. S3 (a corner case below 0.1 % of the windows or steps that moves no reported number beyond its tolerance, a difference within a stated bound, style) is listed in one line of the review and not reported again. A later round reads the changed code and the S1 channels only; a finding of an earlier round is not reported again. A scenario against common sense is no finding at all, whatever its correction costs: it is not reviewed, not listed and not corrected — for example the code changed while a run of it goes on, an input that no step of the pipeline makes, a file moved or edited by hand to defeat a rule, a second person working against the first. Why: four rounds of the review of stage B found no leak, and their later findings came from a deeper search of the same guard in each round, not from new defects; the deepest were guards against a code change during a campaign, which does not happen (the user: "we are not in a spy war") | Decided | User, 2026-10-06 |
| D138 | The speed of the closed loop (§4 item 6). A faster form of a computation is added beside the readable one, never in its place: the readable one stays a mode and the reference, and a check on fixed inputs requires both to give the same result before the faster one is used in a process (as the executor's ways to fly, vocabulary D73); no result changes (words, masks, states, rewards). Three parts, each in its stage's code: (1) vocabulary A44, the start opened once in a process (the split's signals, the arrival records, the candidates, the sentences' offsets and the opened executor held by one object; each flight's series rebuilt once), `start` and `start_moved` kept as the one-call form; (2) prior B14, the procedure masks in two modes, one aircraft at a time (today's code, the reference) and the whole batch at once; (3) post-training C13, the selection readout's batches spoken by the speaking workers, one process kept as a mode, and the start opened before the workers fork so that they share its memory. Not done: speaking on the CPU with about ten workers (other float numbers: another campaign; the host's memory is shared), batches of 128, a compiled executor or judge. A campaign that runs keeps its code: C10 (`post_train_20261006`) gets none of it. Why: C10 speaks bound by the CPU. One batch of 28 windows took 74 s under cProfile: the start about 35 % (each call reads the whole split's signals and arrival records again), the procedure masks about 19 % (Python loops over the aircraft; the grammar's column mask is already over the batch), the inputs, edges and scene about 11 %, the set-up of compiled kernels about 13 % (once a process), the prior's forward pass about 7 %, the executor, the judge and the copies about 15 %. A round of C10's settings takes about 60 min with three workers, about 5 min of it the selection readout in the main process (stage C's log §26). Expected: a batch about 40 % shorter and a round about 35–40 min with three workers; each part is measured after it is built, never assumed | Decided | User, 2026-10-06 (stages A and C made faster; the readable mask code kept as a mode); the parts: stage C's proposals (a)–(c), with Claude's reading that the slow masks are stage B's procedure code, not the grammar |
| D139 | Stage B's readings of frontend §3 and D138, accepted as built (stage B's requests note, 2026-10-06): (1) the speed's batch of 400 is the 100 drawn flights (20 an airport) repeated in turn, copy k of a flight its sample k; (2) stage C's export requires `--speed` too, and a fold set of stage B names the base's readout, saying which model it timed; (3) the results route answers 400 for an airport that is no airport code and 409 by name for a set of another format, and a section whose file is of another format, elsewhere or missing gives that section's problem by name while the others show; (4) stage A's closed-loop section reads the replays of the set's own splits at each Δ of the set; (5) the earlier blocks are checked against the sha256 of each fixture's earlier block, held in the test (a digest of output data, not of code); (6) stage C's page has a section Speed too; (7) A44's `Start` keeps each flight's series until `release`, not the arrival records (holding them would change the harvest's loader, which training reads; a flight's first start still reads its airport's manifests, about 1 % of a batch); (8) B14's check takes 0.6–1.7 s an airport once a process (a grid about each final and a sweep down each course, so that a 0.2 m change of a limit is found); (9) C13's memory rule (O15): N times one worker's measured peak, the main process's growth in one update and the other workers' held memory must fit the host and the GPU — with the parent's shared pages taken out of a worker's peak and the worker's kept model and series counted in | Decided | User, 2026-10-06 (each reading in turn; (9) with the over-count corrected, Claude's review `readouts/2026-10-06_training_view_and_speed_check.zh.md`) |
| D158 | **KAUS is read in three parts** (the user, 2026-10-03). KAUS, the held-out test airport (prior §5), is not read all at once: its 97 operating days are dealt one time into three parts of 33, 32 and 32 days (grouped by half year, April–September and October–March; each group sorted by sha256(seed:day) with the seed 1337; the groups dealt in turn to the parts), so that later models still find flights that no model read. The deal is committed in code (`data/held_out_split_kaus_20261003.json`, `data/held_out_split.py`; branch `dev-kaus-parts`, `9e254341`). A part is opened by a commit, never by a flag; a part that is not opened is sealed as a test day is. Part 1 was opened on 2026-10-03 for a sentence artefact of the earlier vocabulary (instruction-v3: `4dTrajectory/outputs/POOLED/instruction_language/heldout_kaus_part1_20261003`). `dev-kaus-parts` is not merged (the user, 2026-10-06): its other work belongs to the archived design (R46 tests the airport embedding that D5 removed; R47 reads the instruction-v3 labeller). The deal comes into `dev-two-tier` with the final test of KAUS, which also names the part that it reads | Decided | User, 2026-10-03 (three parts, the seed, part 1 opened); not merged: user, 2026-10-06 |

**The identity rule (D21)** (the user, 2026-10-02 and 2026-10-03). An identity binds the format (what the words and the
payloads mean) and the data rules (the sealed test days). Code is identified by what it does on fixed inputs (a
conformance check), not by the bytes of its source. Data are identified by their flights, not by the bytes of a file
they came from. Each document lists the identities of its parts.

### 3.2 Index

| Document | Decisions | Open items |
|---|---|---|
| Outline | D7, D20, D21, D55, D85, D95, D131, D138, D139, D158 | — |
| Training view (`frontend.md`) | D109, D133–D136, D154–D156 | — |
| Vocabulary | D1–D4, D6, D8–D12, D14, D15, D18, D19, D22, D25–D28, D32–D35, D38, D42–D54, D56–D59, D61, D62, D66, D67, D69–D71, D73, D74, D77–D84, D86–D90, D97, D102, D111, D126 | O8 |
| Prior | D5, D13, D14, D16, D17, D23–D25, D31, D39–D41, D58, D60, D63–D65, D68, D72, D75, D96, D105–D108, D111, D118–D122, D126–D128 | — |
| Post-training | D29–D31, D36, D37, D76, D91–D94, D98–D101, D103–D105, D107, D110, D112–D117, D123–D125, D129, D130, D132, D137, D157 | O15 |
| Multi-aircraft control | D140–D153 | O16, O18 |

The next free numbers: D159, O19.

---

## 4 Plan

1. Stage A (vocabulary §0.4): A19 and A20, the user's choice of the altitude grid, A22 (the vertical path of each
   candidate, the state columns and the grammar's column mask in the public interface, D61, D62), A21 (the formal
   artefact and the readings of D34), Claude's check, A24 (the vertical tolerance of the final descent, D66: measured,
   the user chooses), A26 (the start of a closed loop in the public interface, D67), A27 (each flight's stratum in
   the sentence file, D70), A25 (the formal artefact again), A28 (the start opens the executor spec, D71), A29 (no code
   fingerprint, D73), A31 (each closed-loop sentence's outcome, D74), A30 (the formal artefact once more), A23 (the
   Training view of stage A). The user compares the readings of D34 and chooses Δ (D7, D11): Δ = 4 s. Then the
   corrections of Claude's review of 2026-10-05 (vocabulary D77–D84, D85): A36 first (one function of A23's export
   for stage B's Training view, prior B6), A32 (the code), A33 (the measurements; the user chooses the start rule of
   D77), A34 (the formal artefact once more), A35 (the Training sets again); D86 (the export reads no formal replay row) in A32.
2. Stage B (prior §0.4): the prior from the start, chosen by cross-validation over the airports, and the base model.
   Stage B is developed in parallel with the end of stage A (the user, 2026-10-04). A milestone of stage B starts when
   the parts of stage A that it reads are on `dev-two-tier-v4`. The formal runs of stage B waited for Claude's check of
   stage A (done: `readouts/2026-10-05_stage_a_check_a32_a40.zh.md`), the user's choice of Δ (4 s) and A34's artefact. B9 (the interface for stage C, prior D96) and B10 (the
   corrections of Claude's check of stage B, prior D105–D108) come before B5's formal campaign (D95). B12 (the
   corrections of Claude's second check, prior D119–D122) and B13 (the corrections of Claude's check of B12, prior D127,
   D128) are merged. B5's campaign ran on B12's code (2026-10-05 22:40 to 2026-10-06 07:12 UTC); B13 leaves its results
   valid. The Training view of stage B (prior B6) is published, and stage B is closed (prior B7, 2026-10-06).
3. Stage C (post-training §0.4): the post-training in windows of traffic with one aircraft commanded. Stage C is
   developed in parallel with the end of stage B (D95). A milestone of stage C starts when the parts of stages A and B
   that it reads are on its branch; what it needs of their public interfaces is made by them (vocabulary D97, prior
   D96). Its formal runs wait for B5's base and Claude's check of stage B. It ends with the Training view of stage C.
   C10 is done (2026-10-06 22:46, ten rounds). The user continues it to 14 rounds as the same campaign (post-training
   D157, C14): its code on `dev-two-tier-v4`, its rounds 10–13 after the user's fast-forward of `dev-two-tier` and
   C13's check on the GPU; then the user's criterion (D7) over the 14 rounds and the validation readout.
4. Each stage ends with its own Training view (`frontend.md`): the backend's live executor, the export and the frontend
   follow the stage, so that the user sees what the stage does in the frontend (the user, 2026-10-04). From 2026-10-06
   the Training view has its own design document and its own implementer, fronter (frontend D154).
5. The one layout of the three stages' Training views (frontend §3, D133–D136): stage B's implementer, on
   `dev-two-tier-v4` (§5 rule 1); the results page, the envelopes and the shared parts of the exports, the speed runner
   and the base's speed readout (frontend §3 item 10), then the Training sets of stages A and B exported again. It is
   built; its steps after C10 change no file of the view (frontend §3 item 9).
6. The speed of the closed loop (D138), by stage B's implementer on `dev-two-tier-v4` (§5 rule 1), before stage C's next
   campaign: vocabulary A44 first, since C13 uses it, then post-training C13 and prior B14. Each is measured after it is
   built (`post_profile` at C10's settings, `model_speed` of frontend §3 item 10). Nothing of it is merged into `dev-two-tier`
   while C10 runs. Items 5 and 6 are built and reviewed; their steps after C10 (B14's real-data check, C13's GPU check,
   the base's speed readout, the export of stages A's and B's sets, the browser check, the merge report) go to stage
   D's implementer (§5 rule 1). They come first now that C10 is done.
7. Stage D (multi-aircraft control §0.4), by stage D's implementer on `dev-multi-control` (§5 rule 1), after the
   user's decisions of its §0.1 (all decided 2026-10-06): MC0 (the interfaces of stages A, B and C that it needs, D150, D149, D152) and
   MC1 (the census). Their code and their tests on synthetic inputs while C10 runs; their checks on real data and the
   census after C10 ends, after the steps of items 5 and 6 that wait for C10. The order of the work: while C10 runs,
   MC0's and MC1's code; when C10 ends, first the steps of items 5 and 6 (they hold back the merge of
   `dev-two-tier-v4`), then MC0's checks on real data and MC1's census. While C14's rounds run (item 3), the
   multi-aircraft control's code and tests on synthetic inputs only (§5 rule 13); its checks on real data and its
   census after C14. Version 1's formal campaign after the chosen round of the 14; version 2 (D153) after version 1's.
8. The Training view (frontend §0.4), by fronter on `dev-frontend` (§5 rule 1): F1 (the cursor slider) and F2 (windows
   of several commanded aircraft, with stage C's) now, light on the host while C10 runs; F3 (stage D's parts) after
   stage D's MC0; F4 (stage D's export and sets) after its MC4.
9. Every branch but the multi-aircraft control's is merged into `dev-two-tier` (the user, 2026-10-06: merge what can
   be merged and go on from the newest commit): `dev-two-tier-v4` with the frontend document and C14's design
   (`7e581b88`), and `dev-traffic-scenarios` (the optimizer's, outside this design, `780b7bfc`). The work goes on from
   there: stage D's implementer brings `dev-two-tier-v4` level with `dev-two-tier` (a fast-forward) and does the steps
   of item 6 and C14's code on it; `dev-multi-control` merges `dev-two-tier`; fronter's `dev-frontend` is made from
   `dev-two-tier`. Not merged, kept as records: `dev-kaus-parts` and `dev-airport-embedding` (D158), `dev-step9-one-commanded`
   (the archived multi-aircraft design's step 9) and `wip-r32-leg-timing` (not adopted).
10. The user merges (§5 rule 11).

---

## 5 Rules of the implementation

These rules hold at every stage. The plan of each stage is in its document; before a milestone, read the design
sections that it names.

1. The branches. The main parts of stages A, B and C are merged into `dev-two-tier` (2026-10-06). From then on, every
   stage develops on one branch, `dev-two-tier-v4`, in the worktree `.claude/worktrees/two-tier-v4` (the user,
   2026-10-06), so that a review reads one difference (`dev-two-tier...dev-two-tier-v4`):
   - before its first commit, and before a milestone that a new design commit holds, the implementer brings
     `dev-two-tier-v4` level with `dev-two-tier` (a fast-forward, or a merge of `dev-two-tier` when both moved);
   - work begun on another branch (stage B's `dev-training-layout`) is merged into `dev-two-tier-v4` and goes on there;
     a stage branch whose work is all in `dev-two-tier` (`dev-two-tier-v4-prior`, `dev-two-tier-v4-post`) is deleted with
     its worktree;
   - the user merges `dev-two-tier-v4` into `dev-two-tier` (rule 11).

   One implementer develops for every stage: stage D's (the user, 2026-10-06; it takes over stage B's implementer's
   remaining steps, §4 items 5 and 6). Stages A's and B's implementers have no work; stage C's runs C10 to its end and
   does not change code. Stage D's implementer keeps two branches: the work of §4 items 5 and 6 on `dev-two-tier-v4`
   (worktree `.claude/worktrees/two-tier-v4`), and the multi-aircraft control on its own branch `dev-multi-control` in
   the worktree `.claude/worktrees/multi-control`, made from `dev-two-tier-v4`, so that the merge of
   `dev-two-tier-v4` does not wait for the multi-aircraft control's review. It
   brings `dev-multi-control` level with `dev-two-tier-v4` (a merge) before each milestone and before its report; it
   changes the code of stages A, B and C only where multi-aircraft control §6 says, each change written in that
   stage's public interface first; the user merges `dev-multi-control`. On the host, the measurements of §4 items 5
   and 6 after C10 (rule 13) come before the multi-aircraft control's runs on real data. The implementer stages files by explicit paths and reads
   `git diff --cached --stat` before each commit. A formal build or run starts from the main checkout after the user's
   merge, as B5 did, never from the development worktree. C10 (`post_train_20261006`) was launched from
   `.claude/worktrees/two-tier-v4-post` before this rule; its later rounds (C14) and its validation readout run from
   the main checkout and read the paths that it recorded there as `this_checkout` maps them (post-training D157).
   After the merge of §4 item 9, stage D's implementer removes the worktrees and deletes the branches whose work is all
   in `dev-two-tier` (each worktree's data links unlinked first; `git branch -d`, never `-D`).

   The Training view (`frontend.md`) has an implementer of its own, fronter (the user, 2026-10-06; frontend D154), on
   its own branch `dev-frontend` in the worktree `.claude/worktrees/frontend`, made from `dev-two-tier` and merged
   level with it before each milestone; the user merges `dev-frontend`. Fronter changes the Training view's files and
   the Training exports with their files' modules (frontend §8), and no other code of a stage. Stage D's implementer
   does not change those files; it runs the exports of A's and B's sets after C10 (§4 item 5).

   The code keeps its stages' order of imports (`tests/test_architecture.py`): `instructions/` imports neither
   `autopilot/` nor `prior/`, and so on; a change of a stage's public interface is written in that stage's document
   before it is built (vocabulary §6, prior §7). The ignored data trees of each worktree (`data`,
   `trajectory_data_process/outputs`, `4dTrajectory/outputs`, `aeroviz-4d/public/data/airports`) are absolute links to
   LIVE data.
2. Each milestone: read the code that it changes; write the code and its tests; run the milestone's test files; when it
   changes `autopilot/` or `instructions/`, run the checks of vocabulary D73 on the formal artefact through their runners
   (`closed_loop_start_check` runs all three first) and give their largest differences in the log; get a
   code review from a separate reviewer (code only, never documents); correct; commit with explicit paths. Never use
   `git add -A`. Before each commit, read `git diff --cached --stat`.
3. A test never writes under a live root. A test that calls a runner's `main()` gives every write root a tmp path
   (`tests/support.py` `labelled_instruction_artefact` is an example).
4. Run single test files in the foreground. Run the full ts suite at the end of each milestone that changes code, before
   its report, and before a formal build: on 8 workers, in the foreground (`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python
   -m pytest --import-mode=importlib -n 8 --dist worksteal 4dTrajectory/ts_transformer/tests`; about 3 min for about
   1,600 tests, 2026-10-05), or one test at a time, detached (about 15 min: `nohup setsid`, the script writes its own
   PID file). Rule 13 comes first: the 8 workers never start beside a formal build.
5. No compatibility (principle 8). Every changed format gets a new name (the reading name, each schema name; the
   names are constants of the code). The new code refuses an old artefact by its name. No `.get(key, default)`
   fallbacks, no branches on a schema version.
6. SI units only. A value from a regulation has its paragraph in a comment and is defined one time.
7. A formal build or a formal readout only where the user orders it (vocabulary A18, A21, A25). A smoke build (a stated
   limit for each airport) goes to a tmp or scratch directory. It uses the spec that the user chose (D15), and its
   flights are a random sample for each airport and split, seed 1337 (D55). The replay of the val days and every
   criterion wait for the user (D7). Never write into an existing directory under `4dTrajectory/outputs/`.
8. Do not touch the checkouts of running experiments (`.claude/worktrees/step9-run` and others) or the main checkout.
9. A defect that you find outside the milestone goes to `docs/code-health-followups.md` (an entry and a table row), not
   into the change.
10. The stage's implementation log is the implementer's, and the only text about a stage that the implementer writes
    (the user, 2026-10-04): at each milestone, the state and the commits, and each reading that the implementer made
    where the design says nothing (a reading is a proposal until the user decides). The log is a file in `readouts/`
    (stage A: `readouts/2026-10-05_stage_a_implementation_log.md`), not a section of the design document; §0.3 of the
    design document holds only a short status table (part, state), which the implementer keeps current. The design
    document keeps the specifications of the milestones that are not done and the check items that are not done: they
    are the plan that the implementer follows. When a milestone is done, Claude moves its specification to the log. The implementer
    changes no other part of a design document: not the decisions, the design sections, the plan, the values or the key
    code index. (Stage B: `readouts/2026-10-05_stage_b_implementation_log.md`.) A gap in the design, or a reading that needs a change of the design, goes to the user; Claude writes the
    text that the user decides. The log is committed to `dev-two-tier` (the user's checkout), with explicit paths. At
    the end of a stage, the report gives the new code index, and Claude puts it into the key code index of the
    document.
11. The user merges a branch into `dev-two-tier`, not before the Training view of its stage (`frontend.md`): before it, the
    backend's live executor (`aeroviz_backend/autopilot_segment/`) and the Training view of the branch do not work. A
    merge also moves the main checkout's services to the new formats; the sets published in the old formats stay on
    disk, and the new frontend does not read them (principle 8). Until the user merges, the user sees a stage on a test
    stack from its worktree (frontend §2 item 5).
12. A value that a runner fits from data and the user chooses (D15) is measured on all train days in a scratch
    directory, directly after the milestone that writes the runner. This measurement is not a formal build: it writes
    nothing under `4dTrajectory/outputs/`. Report the fitted values and their candidates to the user, and wait for the
    choice before a later milestone reads the value (D55).
13. Two stages run on one host. Before a full ts suite, a smoke build, a measurement or a run on the GPU, read the free
    memory, the GPU memory and the running jobs. Do not start a job that can stop or slow a formal build, a full-train
    measurement or a formal campaign of the other stage.

---

## 6 The Training view

The Training view of every stage — its rules, the one layout, each stage's view, the shared controls, the backend's
Training routes and the content of the Training sets — is in `frontend.md` (moved there on 2026-10-06, the user):
outline §6.1 is frontend §2, outline §6.2 is frontend §3 with the same item numbers, and D109 and D133–D136 are the
Training view's decisions.
