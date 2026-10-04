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
| `vocabulary.md` | The words, the labeller (open-loop and closed-loop reading), the executor, the judge | A | Built on `dev-two-tier-v4`; the formal artefact next (vocabulary §0.4) |
| `prior.md` | The prior: inputs, outputs, decoding and its masks, training, a new airport | B | Not started; it starts in parallel with the end of stage A (§4) |
| `post_training.md` | The post-training in windows of traffic; the multi-aircraft work | C | Not started; outline |
| This outline | The principles, the shared rules, the plan; the frontend and the backend (§6) | D | Not started |

**Dependencies.** The documents depend on each other as the code does. The vocabulary reads no other document. The
prior reads the vocabulary. The post-training reads the vocabulary and the prior. Each document reads another only
through that document's public interface and its decisions:

| Public interface | What it gives | Read by |
|---|---|---|
| Vocabulary §6 | The vocabulary spec, the grammar, the sentence artefact, the candidates and their geometry, the executor, the judge, the row grid | Prior, post-training, stage D |
| Prior §7 | The checkpoint, the inputs of a row, the speaker, the teacher-forced loss, a place for an added module | Post-training, stage D |

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
| D55 | A value that a runner fits from data and the user chooses (D15) is measured on all train days, in a scratch directory, directly after the milestone that writes the runner; the user chooses before a later milestone reads the value. A smoke build uses the chosen spec, and its flights are a random sample for each airport and split (seed 1337), not the first flights of the sorted flight keys (a key starts with the callsign, so the first flights are mostly one airline). Why: the smoke of stage A fitted its own spec on approximately 400 train flights, 373 of them one airline, with one climb piece; every smoke reading of A9–A16 used it (§5 rules 7 and 12) | Decided | User, 2026-10-04 |

**The identity rule (D21)** (the user, 2026-10-02 and 2026-10-03). An identity binds the format (what the words and the
payloads mean) and the data rules (the sealed test days). Code is identified by what it does on fixed inputs (a
conformance check), not by the bytes of its source. Data are identified by their flights, not by the bytes of a file
they came from. Each document lists the identities of its parts.

### 3.2 Index

| Document | Decisions | Open items |
|---|---|---|
| Outline | D7, D20, D21, D55 | — |
| Vocabulary | D1–D4, D6, D8–D12, D14, D15, D18, D19, D22, D25–D28, D32–D35, D38, D42–D54, D56–D59 | O8 |
| Prior | D5, D13, D14, D16, D17, D23–D25, D31, D39–D41, D58 | — |
| Post-training | D29–D31, D36, D37 | O6 |

The next free numbers: D60, O9.

---

## 4 Plan

1. Stage A (vocabulary §0.4): A19 and A20, the user's choice of the altitude grid, A21 (the formal artefact and the
   readings of D34), Claude's check. The user compares the readings of D34 and chooses Δ (D7, D11).
2. Stage B (prior §0.4): the prior from the start, chosen by cross-validation over the airports, and the base model.
   Stage B is developed in parallel with the end of stage A (the user, 2026-10-04). A milestone of stage B starts when
   the parts of stage A that it reads are on `dev-two-tier-v4`. The formal runs of stage B wait for Claude's check of
   stage A and the user's choice of Δ.
3. Stage C (post-training §0.4): the post-training in windows of traffic with one aircraft commanded.
4. Stage D (§6): the frontend and the backend.
5. The user merges the branch.

---

## 5 Rules of the implementation

These rules hold at every stage. The plan of each stage is in its document; before a milestone, read the design
sections that it names.

1. The branches. Stage A is on `dev-two-tier-v4`, in the worktree `.claude/worktrees/two-tier-v4`, made from
   `dev-two-tier`. Stage B is on `dev-two-tier-v4-prior`, in the worktree `.claude/worktrees/two-tier-v4-prior`, made
   from `dev-two-tier-v4`. The merges go in one direction:
   - before a milestone that a new design commit holds, the branch of the stage merges `dev-two-tier`;
   - when stage A commits a part that a milestone of stage B reads, `dev-two-tier-v4-prior` merges `dev-two-tier-v4`;
   - at the end of stage B (prior B6), `dev-two-tier-v4` merges `dev-two-tier-v4-prior`.

   Stage B never changes the code of `instructions/` or `autopilot/`. A defect in it goes to stage A, through the user;
   stage B gets the correction with the next merge. The ignored data trees of each worktree (`data`,
   `trajectory_data_process/outputs`, `4dTrajectory/outputs`, `aeroviz-4d/public/data/airports`) are absolute links to
   LIVE data.
2. Each milestone: read the code that it changes; write the code and its tests; run the milestone's test files; get a
   code review from a separate reviewer (code only, never documents); correct; commit with explicit paths. Never use
   `git add -A`. Before each commit, read `git diff --cached --stat`.
3. A test never writes under a live root. A test that calls a runner's `main()` gives every write root a tmp path
   (`tests/support.py` `labelled_instruction_artefact` is an example).
4. Run single test files in the foreground. Run the full ts suite (approximately 55 min, detached: `nohup setsid`, the
   script writes its own PID file) at the end of each milestone that changes code, before its report, and before a
   formal build.
5. No compatibility (principle 8). Every changed format gets a new name (the reading name, each schema name; the
   names are constants of the code). The new code refuses an old artefact by its name. No `.get(key, default)`
   fallbacks, no branches on a schema version.
6. SI units only. A value from a regulation has its paragraph in a comment and is defined one time.
7. A formal build or a formal readout only where the user orders it (vocabulary A18, A21). A smoke build (a stated limit
   for each airport) goes to a tmp or scratch directory. It uses the spec that the user chose (D15), and its flights are
   a random sample for each airport and split, seed 1337 (D55). The replay of the val days and every criterion wait for
   the user (D7). Never write into an existing directory under `4dTrajectory/outputs/`.
8. Do not touch the checkouts of running experiments (`.claude/worktrees/step9-run` and others) or the main checkout.
9. A defect that you find outside the milestone goes to `docs/code-health-followups.md` (an entry and a table row), not
   into the change.
10. At each milestone, update §0.3 of the stage's document (state, commit). At the end of a stage, update the key code
    index of its document to the new code.
11. The branches are not merged into `dev-two-tier` before stage D: the backend's live executor
    (`aeroviz_backend/autopilot_segment/`) and the frontend's Training view read the old format until then. The user
    merges.
12. A value that a runner fits from data and the user chooses (D15) is measured on all train days in a scratch
    directory, directly after the milestone that writes the runner. This measurement is not a formal build: it writes
    nothing under `4dTrajectory/outputs/`. Report the fitted values and their candidates to the user, and wait for the
    choice before a later milestone reads the value (D55).
13. Two stages run on one host. Before a full ts suite, a smoke build, a measurement or a run on the GPU, read the free
    memory, the GPU memory and the running jobs. Do not start a job that can stop or slow a formal build, a full-train
    measurement or a formal campaign of the other stage.

---

## 6 Stage D: frontend and backend (outline)

The Training exports and the backend's live executor (`aeroviz_backend/autopilot_segment/`) on the new format; the
frontend reads the reading name, not the spec sha: what the frontend reads is identified by the reading name (D21).
After stage D the user merges the branch.
