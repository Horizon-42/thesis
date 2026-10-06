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
| `vocabulary.md` | The words, the labeller (open-loop and closed-loop reading), the executor, the judge; their Training view | A | A0–A43 done and merged into `dev-two-tier`; the formal artefact `v12_20261005` / `v17_20261005` and the Training view (`closed_loop_v12_20261005`) done, the superseded artefacts deleted; Claude's check of A32–A40 done; no milestone open; the replay of the val days waits for the user (vocabulary §0.4) |
| `prior.md` | The prior: inputs, outputs, decoding and its masks, training, a new airport; its Training view | B | Closed (B0–B13 done, B7 on 2026-10-06): B5's campaign `prior_base_20261006` done (configuration C, variant `full`, the base and its one validation readout); B6's sets published (`prior_sets_20261006`); `dev-two-tier` = `dev-two-tier-v4` = `dev-two-tier-v4-prior` (the code at `d14a2f76`) |
| `post_training.md` | The post-training in windows of traffic; the multi-aircraft work; its Training view | C | Built on `dev-two-tier-v4-post` (post-training §0.3, D95), merged into `dev-two-tier` (2026-10-06); C0–C7, C9 and C11 (the window view) done on synthetic artefacts; C8 and C10 built on synthetic artefacts, their formal runs wait; C12 left |
| This outline | The principles, the shared rules, the plan; the rules of each stage's Training view and the one layout of the three views (§6) | — | The one layout (§6.2, D133) not built |

How a stage is reviewed against its design (leaks, what each consumer may read, the procedure, a checklist for each
stage) is in the review guide, `docs/two_tier/review_guide.md`; it is not a design document.

**Dependencies.** The documents depend on each other as the code does. The vocabulary reads no other document. The
prior reads the vocabulary. The post-training reads the vocabulary and the prior. Each document reads another only
through that document's public interface and its decisions:

| Public interface | What it gives | Read by |
|---|---|---|
| Vocabulary §6 | The vocabulary spec, the grammar, the sentence artefact and the stored signals, the candidates and their geometry, the executor and the start of a closed loop, the judge, the row grid | Prior, post-training |
| Prior §7 | The checkpoint, the inputs of a row, the speaker, the teacher-forced loss, a place for an added module, the region of a final, the step of a speaker's closed loop; with the code of each item | Post-training |

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
| D109 | The Training view opens the set of the base model's one validation readout (§6.1 item 4, prior B6) as it opens the others, and a click on a word flies its segment live: the backend rebuilds the val flight and flies the words that the readout already flew (the same states within the executor conformance tolerance), so it gives no new reading of the val days. Only that set: the frontend's reader and the backend's live segment take the splits a set may hold from its caller (stage A's own sets: train and select; stage B's prior sets: also val, for a set exported from the claimed validation readout, prior B10). Why: the export could write that set, but A23's reader and live segment accepted train and select only, so the planned publication did not open (Claude's check of stage B, 2026-10-05; vocabulary A39) | Decided | User, 2026-10-05 |
| D131 | Every review of a stage follows one standard, the same in every round (`review_guide.md` §3 step 6 and "Later rounds"): each finding gets a severity. S1 (a leak into an input, a target selection or a choice; a split violation; a second read of the validation days; a defect that changes a number the user reads, on real data, beyond its tolerance) is corrected before any formal run. S2 (a latent leak, a boundary that does not refuse, a decided rule with no test, an effect not measured and not bounded) is corrected before the formal run when the correction is cheap; else it is listed as an open item of the stage's design document (§0.2, an O number), with no milestone and no order. S3 (a corner case below 0.1 % of the windows or steps that moves no reported number beyond its tolerance, a difference within a stated bound, style) is listed in one line of the review and not reported again. A later round reads the changed code and the S1 channels only; a finding of an earlier round is not reported again. A scenario against common sense is no finding at all, whatever its correction costs: it is not reviewed, not listed and not corrected — for example the code changed while a run of it goes on, an input that no step of the pipeline makes, a file moved or edited by hand to defeat a rule, a second person working against the first. Why: four rounds of the review of stage B found no leak, and their later findings came from a deeper search of the same guard in each round, not from new defects; the deepest were guards against a code change during a campaign, which does not happen (the user: "we are not in a spy war") | Decided | User, 2026-10-06 |
| D133 | The Training views of stages A, B and C share one layout (§6.2). The user decided: one layout with the switch between the stages kept; a set's intent read from `docs/experiments/intents.json` through the backend, with no export, set or index format changed; stage C's branch merged before the views change; stage B's implementer builds it, and for this work may change the view files of stages A and C and the backend's routes. Claude's design of the parts, on stage B's request: the left panel holds the set chooser with the set's own line of the intent registry, the item list with the same four columns, one-line readouts and the Draw switches, and no table; the sentence bar's tabs choose the sentence on screen in every stage (B: Labelled, Closed loop and the samples; C: Labelled and the rounds), and the bar's chips and notes follow the kind of that sentence; the details page in every stage, its first section the set and the experiment; what follows the cursor (B's probability of "go-around" at the cursor) stays in the left panel, since the details page is modal; shared parts, not copies. Why: B's and C's views chose the sentence in a table of the left panel, so that the bar's tab "Δ 4 s" meant the table's choice; B's details page was disabled and C's ⓘ opened nothing; no view showed a set's intent | Decided | User, 2026-10-06 (the layout, the intent's source, the order); the parts: Claude, on stage B's request |

**The identity rule (D21)** (the user, 2026-10-02 and 2026-10-03). An identity binds the format (what the words and the
payloads mean) and the data rules (the sealed test days). Code is identified by what it does on fixed inputs (a
conformance check), not by the bytes of its source. Data are identified by their flights, not by the bytes of a file
they came from. Each document lists the identities of its parts.

### 3.2 Index

| Document | Decisions | Open items |
|---|---|---|
| Outline | D7, D20, D21, D55, D85, D95, D109, D131, D133 | — |
| Vocabulary | D1–D4, D6, D8–D12, D14, D15, D18, D19, D22, D25–D28, D32–D35, D38, D42–D54, D56–D59, D61, D62, D66, D67, D69–D71, D73, D74, D77–D84, D86–D90, D97, D102, D111, D126 | O8 |
| Prior | D5, D13, D14, D16, D17, D23–D25, D31, D39–D41, D58, D60, D63–D65, D68, D72, D75, D96, D105–D108, D111, D118–D122, D126–D128 | — |
| Post-training | D29–D31, D36, D37, D76, D91–D94, D98–D101, D103–D105, D107, D110, D112–D117, D123–D125, D129, D130, D132 | O13 |

The next free numbers: D134, O15.

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
4. Each stage ends with its own Training view (§6): the backend's live executor, the export and the frontend follow the
   stage, so that the user sees what the stage does in the frontend (the user, 2026-10-04). There is no separate
   frontend stage.
5. The one layout of the three stages' Training views (§6.2, D133): stage B's implementer, on one branch from
   `dev-two-tier`; no export is run again. While it is built, no other branch changes the files of the Training view
   (§6.2 item 7).
6. The user merges (§5 rule 11).

---

## 5 Rules of the implementation

These rules hold at every stage. The plan of each stage is in its document; before a milestone, read the design
sections that it names.

1. The branches. Stage A is on `dev-two-tier-v4`, in the worktree `.claude/worktrees/two-tier-v4`, made from
   `dev-two-tier`. Stage B is on `dev-two-tier-v4-prior`, in the worktree `.claude/worktrees/two-tier-v4-prior`, made
   from `dev-two-tier-v4`. The merges go in one direction:
   - before a milestone that a new design commit holds, the branch of the stage merges `dev-two-tier`;
   - when stage A commits a part that a milestone of stage B reads, `dev-two-tier-v4-prior` merges `dev-two-tier-v4`;
   - at the end of stage B (prior B7), `dev-two-tier-v4` merges `dev-two-tier-v4-prior`.

   Stage C is on `dev-two-tier-v4-post`, in the worktree `.claude/worktrees/two-tier-v4-post`, made from
   `dev-two-tier-v4-prior` (D95). When stage B commits a part that a milestone of stage C reads (stage A's parts reach
   stage C through stage B's merges), `dev-two-tier-v4-post` merges `dev-two-tier-v4-prior`; after prior B7 it merges
   `dev-two-tier-v4`; at the end of stage C (post-training C12), `dev-two-tier-v4` merges `dev-two-tier-v4-post`.

   Stage B never changes the code of `instructions/` or `autopilot/`; stage C never changes the code of
   `instructions/`, `autopilot/` or `prior/`. A defect in them goes to their stage, through the user; the stage that
   reads them gets the correction with the next merge. A part of a public interface that a later stage needs is
   requested from the stage that owns it, through the user (vocabulary A36, D97; prior D96). The ignored data trees of each worktree (`data`,
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
11. The user merges a branch into `dev-two-tier`, not before the Training view of its stage (§6): before it, the
    backend's live executor (`aeroviz_backend/autopilot_segment/`) and the Training view of the branch do not work. A
    merge also moves the main checkout's services to the new formats; the sets published in the old formats stay on
    disk, and the new frontend does not read them (principle 8). Until the user merges, the user sees a stage on a test
    stack from its worktree (§6).
12. A value that a runner fits from data and the user chooses (D15) is measured on all train days in a scratch
    directory, directly after the milestone that writes the runner. This measurement is not a formal build: it writes
    nothing under `4dTrajectory/outputs/`. Report the fitted values and their candidates to the user, and wait for the
    choice before a later milestone reads the value (D55).
13. Two stages run on one host. Before a full ts suite, a smoke build, a measurement or a run on the GPU, read the free
    memory, the GPU memory and the running jobs. Do not start a job that can stop or slow a formal build, a full-train
    measurement or a formal campaign of the other stage.

---

## 6 The Training view of each stage

Each stage ends with a milestone that brings the frontend and the backend to that stage (the user, 2026-10-04):
vocabulary A23, prior B6, and the Training view of the post-training (post-training C11). The user sees each stage in
the frontend before the next one builds on it. §6.1 gives the rules for these milestones; §6.2 gives the one layout that
the views of the three stages share.

### 6.1 The rules

1. **What follows the stage.** The backend's live executor (`aeroviz_backend/autopilot_segment/`, a click on a word
   flies its segment), the Training export of the stage's results, and the frontend's Training view (`aeroviz-4d`). The
   archived exports and display code (`archive/two_tier_v3_2026_10/`: `instructions/training_files.py`,
   `instructions/display.py`, the `*_training_export.py` runners, `experiments/training_attitude.py`) come back,
   rewritten, in the stage that needs them.
2. **Formats.** Every changed export format gets a new schema name, and the frontend refuses an old one (principle 8).
   The frontend reads the reading name and the format names, not the spec sha (D21). The fixtures of the frontend's
   tests are written by the export code, not by hand.
3. **Beside the old sets.** The worktree's `aeroviz-4d/public/data/airports` is a link to the live data (§5 rule 1).
   A stage's export writes its own index under a new name beside `training/index.json`; the old index and its sets
   stay unchanged, so the Training view of the main checkout keeps working until the user merges.
4. **Publication.** The intent of each published set is in `docs/experiments/intents.json`, committed before the
   export. Train and select flights; val flights only from a readout that a plan already makes (the base model's one
   validation readout, prior B5), never a new read of the val days. That set opens and flies live as the others do
   (D109).
5. **The user's view.** A test stack from the stage's worktree: the backend and vite on their own ports (§5 rule 13 for
   the memory). The report gives the address and the stop command (`kill $(lsof -t -iTCP:<port> -sTCP:LISTEN)`). The
   main checkout's services are not touched (§5 rule 8).
6. **The check.** A published set opens and plays in the browser, and a live segment equals the flown states of the
   export from the same state with the same words (the executor conformance tolerance). The browser check runs in a
   one-shot subagent that returns a short verdict; HTTP 200 is not "it loads".

### 6.2 One layout for the three stages (D133)

Stages A, B and C keep their own sets, readers, sessions and live segments, and the switch between them at the top of
the left panel ("Sets of": stage A, B or C, offered where the airport has their sets). Their views share one layout,
built from shared parts. No export runs again; no set, index or sample format changes; the backend's live segments do
not change.

Why: B's view (prior B6) and C's (post-training C11) were built on one template that chose the sentence in a table of
the left panel; the sentence bar's tabs then read "Labelled" and "Δ 4 s", and "Δ 4 s" meant the table's choice. B's ⓘ
was disabled, and C's ⓘ opened nothing, because only stage A's session draws the details page. No stage's view showed
the intent of its set. Stage A's view (A23) already has the layout that the user chose before: the sentence chosen in
the bar's tabs, a short left panel, the tables on a details page.

1. **The left panel**, top to bottom, the same in every stage:
   - the stage switch;
   - the set chooser: a choice of set (its id and its count of items), and under it one line: the set's own line of
     the intent registry (item 4) and a SMOKE tag for a set whose source says smoke (stages B and C; stage A's sets
     have no such flag). The line opens the details page on "The set and the experiment";
   - the list of items (flights; in stage C, windows), with four columns in every stage: the callsign; the runway
     (stage C: the recorded runway, post-training D129); the stratum (stage C: the window's kind); the outcomes of the
     item's own sentences, as the number landed of all (stage A: its closed-loop sentences, one for each Δ; B: the
     prior's samples; C: the rounds, and the count of other aircraft), with each sentence's outcome in the tooltip;
   - the readouts, one line each, each opening the details page on its section (item 3);
   - the Draw switches of the stage.

   No table is in the left panel, and no block of more than one line, except the probability strip of stage B
   (item 3).

2. **The sentence bar's tabs choose the sentence on screen**, in every stage. The left panel has no second chooser:
   one state of the app holds the choice, and the session gives the bar its tabs.

   | Stage | The tabs |
   |---|---|
   | A | Labelled · Δ 2 s · Δ 4 s · Δ 8 s (one for each Δ of the set, as now) |
   | B | Labelled · Closed loop · Sample 0 · Sample 1 … |
   | C | Labelled · Start (base) · Round 1 … |

   Labelled is the labeller's open-loop reading of the observed flight (stage C: of the recorded flight of the
   commanded aircraft; for a window of kind B, its tooltip says that the window starts from a moved start). B's Closed
   loop is stage A's closed-loop sentence of the flight at the prior's Δ. A tab of a sentence with an outcome carries a
   small dot in the outcome's colour (`trainingOutcomeColour`), and its tooltip gives the outcome, the time and the
   go-arounds. Many tabs wrap onto a second line (the bar measures its own height); the arrow keys move between the
   tabs, Home and End go to the first and the last. A word of the sentence on screen flies live as now: stage A's
   closed-loop sentences on stage A's segment, B's closed loop and samples on the prior's segment, C's rounds on the
   window's segment; a labelled sentence does not fly.

   The bar's chips and notes follow the kind of the sentence on screen: a closed-loop sentence — the outcome, the DA
   check and the corrections; a prior sample — the outcome, the DA check and the go-arounds; a round — the outcome, the
   reward and the loss of separation; a labelled sentence — none. The bar's ⓘ keeps its notes on how to read the bar,
   written for that kind (only the notes of a closed-loop sentence speak of words that the reading added). Their first
   line names the set and gives its intent line, with a link to the details page's "The set and the experiment".

3. **The readouts and the details page.** Every stage has the details page (`TrainingDetails`), and the ⓘ of the
   panel's header is never disabled: it opens the page on its first section. The page is modal (the scene and the bar
   are inert behind it), so what follows the cursor stays in the left panel.

   | Stage | Readouts in the left panel (one line each → its section) | Sections after the first two |
   |---|---|---|
   | A | Flown flight (at the tab's Δ: the outcome and its time, the DA check) → Flown flights; Flown flights (landed at each Δ) → Flown flights | Vocabulary; Flown flights (every flight at every Δ) |
   | B | This sentence (the outcome and its time, the DA check, the go-arounds) → Sentences; Sentences (landed of each sample) → Sentences; At the cursor (the probability of "go-around", whether the procedure permits one, on final or not, the count of words blocked in each masked column) → Row inspector | Sentences (every flight's sentences; landed by sample and by stratum); Row inspector (at a row: the words that the procedure masks block and permit in each column they rule; a row slider that moves the cursor) |
   | C | This round (the outcome, the reward, the loss of separation or none) → Rounds; Rounds (landed of each round) → Rounds; The window (its kind, its other aircraft by role) → The window | Rounds (every window's rounds: the outcome, the reward, the go-arounds, the words, the loss of separation); The window (its kind, its moves, its other aircraft, the round's end, the threshold, the rows where the speed-word mask acted, the faulty points) |

   The first two sections in every stage: "The set and the experiment" — the campaign's title, its intent and the
   design it names, the set's own line (item 4); what the set was made from, as the set's source gives it; its items
   and how they were drawn; for a set exported from a claimed validation readout, the path of that readout — and "What
   this view shows" — what an item is, what each tab is, what each Draw switch draws. Under B's line "At the cursor"
   stays the probability strip of the sentence on screen (one line high; a click moves the cursor). Stage C has no row
   inspector: a window set holds no records of the speaker (post-training D125).

4. **The intent.** The backend serves the intent of a set: `GET /experiments/intent?run=<set id>` reads
   `docs/experiments/intents.json` at each request (an entry committed later shows without a restart of the backend)
   and finds the set's id among the `runs` keys of every campaign. One campaign: the answer gives the campaign's id,
   title, intent and design and the run's line. No campaign, or more than one (run keys are not unique in the registry:
   `S20_cv_s1337` is a run of two campaigns): the answer is an error that names the set and the campaigns found; the
   view shows it by name, and the set still opens. The backend writes nothing. The frontend shows an intent in the one
   form of the experiments picker (`ExperimentIntent`, the intent block of `ExperimentDetails`).
5. **Shared parts, not copies.** One hook that loads a set (now stage A's panel and B's and C's sessions each have
   their own); one set chooser; one item list; the bar's tabs; the details page and its state, given to every session;
   the readout line (`DetailsLink`); the Draw box. `ProblemBox` and `trainingText` stay the one definition of problems
   and outcome words. Each stage keeps its reader, its session and its own sections.
6. **Tests and the check.** Vitest for each shared part, on fixtures that the exports write (§6.1 item 2); every test
   of the three stages kept (among them: a set of another airport never opens, a stale choice after a set changes, the
   stage switch, the cursor at a window's row 0); a test of the backend's intent answer (one campaign, none, more than
   one). The browser check of §6.1 item 6 in each stage: the tabs choose the sentence, the details page opens from
   every ⓘ and readout line, the intent shows, a word flies live.
7. **Who and where.** Stage B's implementer (the user, 2026-10-06), on the branch `dev-training-layout` from
   `dev-two-tier`, in the worktree `.claude/worktrees/training-layout`; for this work only, it may change the view files
   of stages A and C and the backend's routes (§5 rule 1 otherwise holds). Its log is a section of stage B's
   implementation log; a reading where this section says nothing is a proposal in stage B's requests note. While it is
   built, no other branch changes the files of the Training view (`aeroviz-4d/src/components/Training*.tsx`,
   `aeroviz-4d/src/components/training/`, `aeroviz-4d/src/data/training*.ts`, `aeroviz-4d/src/hooks/useTraining*.ts`,
   their tests) or the backend's routes (`aeroviz_backend/http_server.py`). The test stack and the report follow §6.1
   item 5.
