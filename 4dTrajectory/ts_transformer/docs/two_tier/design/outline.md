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
| `vocabulary.md` | The words, the labeller (open-loop and closed-loop reading), the executor, the judge; their Training view | A | Built on `dev-two-tier-v4` and merged into `dev-two-tier` (`a804633e`: A0–A40); the formal artefact `v12_20261005` / `v17_20261005` and the Training view (`closed_loop_v12_20261005`) done; Claude's check of A32–A40 done; A41 and the deletion of the superseded artefacts left (vocabulary §0.4) |
| `prior.md` | The prior: inputs, outputs, decoding and its masks, training, a new airport; its Training view | B | Built in parallel with the end of stage A, on `dev-two-tier-v4-prior` (prior §0.3); B9 (the interface for stage C, D96) done; Claude's check of stage B done, its corrections B10 (D105–D108) before B5's formal campaign |
| `post_training.md` | The post-training in windows of traffic; the multi-aircraft work; its Training view | C | Design complete; built in parallel with the end of stage B, on `dev-two-tier-v4-post` (post-training §0.3, D95); C0–C5 and C7 done on synthetic artefacts; C6 and C8–C12 left |
| This outline | The principles, the shared rules, the plan; the rules of each stage's Training view (§6) | — | — |

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
| D109 | The Training view opens the set of the base model's one validation readout (§6 item 4, prior B6) as it opens the others, and a click on a word flies its segment live: the backend rebuilds the val flight and flies the words that the readout already flew (the same states within the executor conformance tolerance), so it gives no new reading of the val days. Only that set: the frontend's reader and the backend's live segment take the splits a set may hold from its caller (stage A's own sets: train and select; stage B's prior sets: also val, for a set exported from the claimed validation readout, prior B10). Why: the export could write that set, but A23's reader and live segment accepted train and select only, so the planned publication did not open (Claude's check of stage B, 2026-10-05; vocabulary A39) | Decided | User, 2026-10-05 |

**The identity rule (D21)** (the user, 2026-10-02 and 2026-10-03). An identity binds the format (what the words and the
payloads mean) and the data rules (the sealed test days). Code is identified by what it does on fixed inputs (a
conformance check), not by the bytes of its source. Data are identified by their flights, not by the bytes of a file
they came from. Each document lists the identities of its parts.

### 3.2 Index

| Document | Decisions | Open items |
|---|---|---|
| Outline | D7, D20, D21, D55, D85, D95, D109 | — |
| Vocabulary | D1–D4, D6, D8–D12, D14, D15, D18, D19, D22, D25–D28, D32–D35, D38, D42–D54, D56–D59, D61, D62, D66, D67, D69–D71, D73, D74, D77–D84, D86–D90, D97, D102, D111 | O8 |
| Prior | D5, D13, D14, D16, D17, D23–D25, D31, D39–D41, D58, D60, D63–D65, D68, D72, D75, D96, D105–D108, D111 | — |
| Post-training | D29–D31, D36, D37, D76, D91–D94, D98–D101, D103–D105, D107, D110, D112, D113 | — |

The next free numbers: D114, O13.

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
   corrections of Claude's check of stage B, prior D105–D108) come before B5's formal campaign (D95). It ends with the
   Training view of stage B (prior B6).
3. Stage C (post-training §0.4): the post-training in windows of traffic with one aircraft commanded. Stage C is
   developed in parallel with the end of stage B (D95). A milestone of stage C starts when the parts of stages A and B
   that it reads are on its branch; what it needs of their public interfaces is made by them (vocabulary D97, prior
   D96). Its formal runs wait for B5's base and Claude's check of stage B. It ends with the Training view of stage C.
4. Each stage ends with its own Training view (§6): the backend's live executor, the export and the frontend follow the
   stage, so that the user sees what the stage does in the frontend (the user, 2026-10-04). There is no separate
   frontend stage.
5. The user merges (§5 rule 11).

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
2. Each milestone: read the code that it changes; write the code and its tests; run the milestone's test files; get a
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
vocabulary A23, prior B6, and the Training view of the post-training (post-training §8). The user sees each stage in
the frontend before the next one builds on it. The rules for these milestones:

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
