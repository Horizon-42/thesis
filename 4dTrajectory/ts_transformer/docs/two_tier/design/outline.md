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
| `vocabulary.md` | The words, the labeller (open-loop and closed-loop reading), the executor, the judge | A | A0–A44 done; no milestone open. The formal artefact `v19_20261014` / `v24_20261014` (D196–D199, D201, D203), read by B19; before it `v18_20261013` / `v23_20261013` (read by B16), `v17_20261013` / `v22_20261013` (read by B15) and `v12_20261005` / `v17_20261005` (read by B5's base and the stage C and D models; its Training view `closed_loop_v12_20261005`). The test days check the labelling at the end (D198) |
| `prior.md` | The prior: inputs, outputs, decoding and its masks, training, a new airport | B | B0–B15 done. B5's base `prior_base_20261006` (on v12; configuration C) is the start of the stage C and D models. B15 `prior_b15_20261013` (on v17): configuration C chosen, the base lands 90.3 % of the val days, the folds 50–63 % at their held-out airports (cause: the labels' soft lateral centring, vocabulary D201). B16 (B15's campaign on v18) paused after configuration A's folds (47.0 % at their held-out airports, 9–30 m off the centreline); D204 (the lateral offset on the glidepath's 100 m scale) took them to 80.2 %. B19 (B15's campaign on v19 with D204): folds 80–85 % at their held-out airports, configuration A, the base 87.2 % of the val days after 13 epochs; the user chose the 30-epoch base `prior_b19_base30_20261014`: 93.0 % of the val days (B15's base 90.3 %). Every run of B19 stopped early (100 epochs: the select days 96.2 %); D205 (15 epochs, at most 150) and B20, B19's campaign again under it, paused after three folds for a trial without the landings input; D206 (the prior reads no landings; the trial's KRDU fold 84.8 → 90.8 %) and B21, B20's campaign without it: running |
| `post_training.md` | The post-training in windows of traffic with one commanded aircraft | C | C0–C11 and C13–C27 done; C12 (the close) left. The user's chosen round: `post_seg60_20261007` round 5 (D7), its one validation readout 88.9 % landed; its sets published |
| `multi_control.md` | The multi-aircraft control: every arrival of a window's span commanded; the window's reward and its credit by one varied aircraft; the model in two versions (the other aircraft's words in force, then their intent); what it needs of the other stages' public interfaces | D | Version 1 done (MC0–MC7, MC11–MC14): the formal campaign `multi_train_majority_formal_20261011`, 12 rounds; the user's chosen round 7, its one validation readout 83.9 % landed, 63.8 % of its aircraft passed by the evaluation package; its sets published; the report `readouts/2026-10-12_stage_d_report.md`. D200 (one method of branch training for C and D) built, its comparison held by the user. Version 2 (MC8–MC10) not started |
| `frontend.md` | The Training view of every stage: its rules, the one layout, each stage's view (stage D's: stage C's window view with several commanded aircraft), the shared controls (the cursor slider), the backend's Training routes, the content and formats of the Training sets | — | F0–F5 done; the evaluation's landing verdict (`288b779b`) and the results route of B15's and D200's campaigns (`be3cdd49`) done. A change waits for the user's permission (§5 rule 11) |
| This outline | The principles, the shared rules, the plan | — | The principles (§2), the shared decisions (§3), the plan (§4), the rules (§5) |

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
   inputs, except the words said before the step. The one exception is a network that never speaks: the
   post-training's value network reads the recorded traffic's future and the time left (post-training D171).
8. **No compatibility.** A changed word, spec or payload gets a new name. The code refuses an artefact of another name.

---

## 3 Decisions

### 3.1 Shared decisions

| # | Item | State | Source |
|---|---|---|---|
| D7 | The design documents set no test criteria and no measurement criteria. The user sets them after the design is settled | Decided | User, 2026-10-03 |
| D20 | This design is a new version, on its own branch and worktree. It reads no artefact, executor spec or prior that an earlier version made (`v1`–`v6`). The running experiments keep their own checkouts | Decided | User, 2026-10-03 |
| D21 | Identities bind format and data rules only. A code identity is a behaviour check on fixed inputs, never a hash of source bytes; data are identified by their flights, never by the bytes of a manifest (vocabulary §7.2, prior §8, post-training §4) | Decided | User, 2026-10-03 |
| D85 | A readout that serves a choice of the user reports the train and select days only. Where a build needs the validation days (the closed-loop sentences of val), it writes them, but no report, summary or printed text shows a reading of them before the stage's one validation readout. Why: the user chose Δ = 4 s (vocabulary D11) on a report that showed the val rows of the readings of D34 (`readouts/2026-10-04_stage_a_a25_report.zh.md`); the train and select rows give the same order, but the validation days are read one time for each stage and each labelled artefact (new labels are a new data set: their base has its own one read). Stage A's labelling reads its val days (D198) | Decided | User, 2026-10-05, on Claude's review of stage A; one read for each labelled artefact: user, 2026-10-13 |
| D95 | Stage C is developed in parallel with the end of stage B (the user, 2026-10-05), on its own branch and worktree (§5 rule 1), by its own implementer; Claude writes its design and runs the project. Stage C never changes the code of `instructions/`, `autopilot/` or `prior/`. What it needs of them is a change of their public interfaces, made by their own stages (vocabulary D97, prior D96): code only, so that nothing of stage A is built again (the user, 2026-10-05); the changes of stage B before B5's formal campaign, because they change the draws of free generation. The formal runs of stage C wait for B5's base | Decided | User, 2026-10-05 |
| D55 | A value that a runner fits from data and the user chooses (D15) is measured on all train days, in a scratch directory, directly after the milestone that writes the runner; the user chooses before a later milestone reads the value. A smoke build uses the chosen spec, and its flights are a random sample for each airport and split (seed 1337), not the first flights of the sorted flight keys (a key starts with the callsign, so the first flights are mostly one airline). Why: the smoke of stage A fitted its own spec on approximately 400 train flights, 373 of them one airline, with one climb piece; every smoke reading of A9–A16 used it (§5 rules 7 and 12) | Decided | User, 2026-10-04 |
| D131 | Every review of a stage follows one standard, the same in every round (`review_guide.md` §3 step 6 and "Later rounds"): each finding gets a severity. S1 (a leak into an input, a target selection or a choice; a split violation; a second read of the validation days; a defect that changes a number the user reads, on real data, beyond its tolerance) is corrected before any formal run. S2 (a latent leak, a boundary that does not refuse, a decided rule with no test, an effect not measured and not bounded) is corrected before the formal run when the correction is cheap; else it is listed as an open item of the stage's design document (§0.2, an O number), with no milestone and no order. S3 (a corner case below 0.1 % of the windows or steps that moves no reported number beyond its tolerance, a difference within a stated bound, style) is listed in one line of the review and not reported again. A later round reads the changed code and the S1 channels only; a finding of an earlier round is not reported again. A scenario against common sense is no finding at all, whatever its correction costs: it is not reviewed, not listed and not corrected — for example the code changed while a run of it goes on, an input that no step of the pipeline makes, a file moved or edited by hand to defeat a rule, a second person working against the first. Why: four rounds of the review of stage B found no leak, and their later findings came from a deeper search of the same guard in each round, not from new defects; the deepest were guards against a code change during a campaign, which does not happen (the user: "we are not in a spy war") | Decided | User, 2026-10-06 |
| D138 | The speed of the closed loop (§4 item 6). A faster form of a computation gives the results of the code before it (words, masks, states, rewards), checked once before its commit, and then replaces it (D187). Three parts, each in its stage's code: (1) vocabulary A44, the start opened once in a process (the split's signals, the arrival records, the candidates, the sentences' offsets and the opened executor held by one object; each flight's series rebuilt once), `start` and `start_moved` kept as the one-call form; (2) prior B14, the procedure masks computed for the whole batch at once (the per-aircraft form retired, D187); (3) post-training C13, the selection readout's batches spoken by the speaking workers, one process kept as a mode, and the start opened before the workers fork so that they share its memory. Not done: speaking on the CPU with about ten workers (other float numbers: another campaign; the host's memory is shared), batches of 128, a compiled executor or judge. A campaign that runs keeps its code: C10 (`post_train_20261006`) gets none of it. Why: C10 speaks bound by the CPU. One batch of 28 windows took 74 s under cProfile: the start about 35 % (each call reads the whole split's signals and arrival records again), the procedure masks about 19 % (Python loops over the aircraft; the grammar's column mask is already over the batch), the inputs, edges and scene about 11 %, the set-up of compiled kernels about 13 % (once a process), the prior's forward pass about 7 %, the executor, the judge and the copies about 15 %. A round of C10's settings takes about 60 min with three workers, about 5 min of it the selection readout in the main process (stage C's log §26). Expected: a batch about 40 % shorter and a round about 35–40 min with three workers; each part is measured after it is built, never assumed | Decided | User, 2026-10-06 (stages A and C made faster; the readable mask code kept as a mode until D187, 2026-10-09); the parts: stage C's proposals (a)–(c), with Claude's reading that the slow masks are stage B's procedure code, not the grammar |
| D139 | Stage B's readings of frontend §3 and D138, accepted as built (stage B's requests note, 2026-10-06): (1) the speed's batch of 400 is the 100 drawn flights (20 an airport) repeated in turn, copy k of a flight its sample k; (2) stage C's export requires `--speed` too, and a fold set of stage B names the base's readout, saying which model it timed; (3) the results route answers 400 for an airport that is no airport code and 409 by name for a set of another format, and a section whose file is of another format, elsewhere or missing gives that section's problem by name while the others show; (4) stage A's closed-loop section reads the replays of the set's own splits at each Δ of the set; (5) the earlier blocks are checked against the sha256 of each fixture's earlier block, held in the test (a digest of output data, not of code); (6) stage C's page has a section Speed too; (7) A44's `Start` keeps each flight's series until `release`, not the arrival records (holding them would change the harvest's loader, which training reads; a flight's first start still reads its airport's manifests, about 1 % of a batch); (8) B14's check takes 0.6–1.7 s an airport once a process (a grid about each final and a sweep down each course, so that a 0.2 m change of a limit is found); (9) C13's memory rule (O15): N times one worker's measured peak, the main process's growth in one update and the other workers' held memory must fit the host and the GPU — with the parent's shared pages taken out of a worker's peak and the worker's kept model and series counted in | Decided | User, 2026-10-06 (each reading in turn; (9) with the over-count corrected, Claude's review `readouts/2026-10-06_training_view_and_speed_check.zh.md`) |
| D158 | **KAUS is read in three parts** (the user, 2026-10-03). KAUS, the held-out test airport (prior §5), is not read all at once: its 97 operating days are dealt one time into three parts of 33, 32 and 32 days (grouped by half year, April–September and October–March; each group sorted by sha256(seed:day) with the seed 1337; the groups dealt in turn to the parts), so that later models still find flights that no model read. The deal is committed in code (`data/held_out_split_kaus_20261003.json`, `data/held_out_split.py`; branch `dev-kaus-parts`, `9e254341`). A part is opened by a commit, never by a flag; a part that is not opened is sealed as a test day is. Part 1 was opened on 2026-10-03 for a sentence artefact of the earlier vocabulary (instruction-v3: `4dTrajectory/outputs/POOLED/instruction_language/heldout_kaus_part1_20261003`). `dev-kaus-parts` is not merged (the user, 2026-10-06): its other work belongs to the archived design (R46 tests the airport embedding that D5 removed; R47 reads the instruction-v3 labeller). The deal comes into `dev-two-tier` with the final test of KAUS, which also names the part that it reads | Decided | User, 2026-10-03 (three parts, the seed, part 1 opened); not merged: user, 2026-10-06 |
| D163 | **A formal run starts from a run worktree, and its intent is checked at publication** (the user, 2026-10-07). A formal build, campaign or readout starts from a detached worktree made for it at the commit that the user merged (`.claude/worktrees/run-<id>`, its data trees linked as every worktree's — `aeroviz-4d/public/data/airports` too, from which the readouts and the exports read the procedure details — a clean tree), and the worktree is removed when the run's data are sealed; recorded paths are read through `this_checkout` (post-training D157), so any checkout reads them. The main checkout stays free: merges and fast-forwards go on while a run runs. A run's intent is written in `docs/experiments/intents.json` before its publication, and the publisher refuses a run without one (L27); a launch does not check it. Why: C14 ran from the main checkout and other sessions fast-forwarded it eight times under the run (stage C's log §27); the run was not affected, but nothing kept it so | Decided | User, 2026-10-07 |
| D184 | **The closing phase: stage D's implementer fixes small defects of stages A, B and C directly** (the user, 2026-10-09: "我们已经到了收尾阶段，d 可以直接改 a b c 阶段的小问题，没必要留下错误的接口"). From 2026-10-09, a small defect of the code of `instructions/`, `autopilot/`, `prior/` or `post/` and their runners — a bug, memory or time spent for nothing, a mirror of another module's code, a wrong or missing public name or signature — is fixed by stage D's implementer in its own change, not worked around, mirrored or requested of another stage. A wrong interface is corrected, not kept beside the right one (no compatibility shim). The stage's behaviour checks run before the commit (the executor's conformance, vocabulary D73; the prior's behaviour check, prior D108; D149's check bit for bit on fixed inputs; the stage's tests). A name that changes goes into the implementer's report, and Claude writes it into that stage's public interface after. Not included, still the user's first: a format or an identity of a trained artefact or of a data artefact, a rebuild of an artefact, and a rule of a decided design. It replaces D95's and multi-aircraft control §6.2's "a change written in that stage's public interface first" for such fixes | Decided | User, 2026-10-09 |
| D187 | **A fast form replaces the code before it, checked once before its commit** (the user, 2026-10-09: "确认快速模式不会改变结果，就直接默认用快速模式，旧模式淘汰，只需要在提交代码前确认一次"). A faster form of a computation is checked before its commit to give the results of the code before it, on fixed inputs, on the CPU with one thread: the stage's behaviour checks (`window_behaviour` against the record of the code before, on stage C's and stage D's windows, multi-aircraft control D149; `prior_behaviour`, prior D108; the labeller's conformance where the grammar is read, vocabulary D73) and the stage's tests. It then replaces that code: the code before it is deleted (git keeps it), no mode chooses between the two, and no process compares them again. The form that stays is written to be read on its own (the user, 2026-10-09: "保证新写法的可读性就可以了"): its names say what it computes, and its docstring says what it computes and why the result is the definition's (for the grammar's mask, the property of the rules it rests on); a review of the change reads it for that. Where a later change of a rule must be checked against the rule's definition, a test computes the definition itself (the grammar's mask by `apply`, vocabulary D62), not a kept copy of the old code. Applied now: the masks' modes go — the procedure masks one aircraft at a time and their check once a process (`require_same_masks`, prior B14), the grammar's readable column mask in the speaker and the rows' inputs one aircraft at a time (multi-aircraft control D186 (2), (3)) are deleted; the forms over the batch are the only ones and the mode parameters go with them (D184: no shim). Kept, being no second form of one computation (Claude's reading): the selection readout in one process (post-training C13: the path with no workers), `start` and `start_moved` (calls of `Start`), the executor's three ways to fly (vocabulary D73: each serves its own callers). Why: the code does not change between its commit and its runs, so a comparison in every process answers again a question answered once, costs every run and keeps two codes alive | Decided | User, 2026-10-09 (Claude's first form, the two modes compared in every process, rejected by the user; the readability of the form that stays: the user, 2026-10-09) |
| D198 | Stage A's labelling reads its val days. The readouts of the vocabulary's labelling and its closed loop (the landed shares of the closed-loop sentences and of their replay) report the train, select and val days: they show that the labelling's ceiling, the share of flights that land on their own labels, is high enough. The test days stay closed until the end, where they check that the whole labelling generalises. D85 holds for every other reading | Decided | User, 2026-10-12 |

**The identity rule (D21)** (the user, 2026-10-02 and 2026-10-03). An identity binds the format (what the words and the
payloads mean) and the data rules (the sealed test days). Code is identified by what it does on fixed inputs (a
conformance check), not by the bytes of its source. Data are identified by their flights, not by the bytes of a file
they came from. Each document lists the identities of its parts.

### 3.2 Index

| Document | Decisions | Open items |
|---|---|---|
| Outline | D7, D20, D21, D55, D85, D95, D131, D138, D139, D158, D163, D184, D187, D198 | — |
| Training view (`frontend.md`) | D109, D133–D136, D154–D156, D159, D160, D177, D178 | — |
| Vocabulary | D1–D4, D6, D8–D12, D14, D15, D18, D19, D22, D25–D28, D32–D35, D38, D42–D54, D56–D59, D61, D62, D66, D67, D69–D71, D73, D74, D77–D84, D86–D90, D97, D102, D111, D126, D196, D197, D199, D201, D203 | O8, O22 |
| Prior | D5, D13, D14, D16, D17, D23–D25, D31, D39–D41, D58, D60, D63–D65, D68, D72, D75, D96, D105–D108, D111, D118–D122, D126–D128, D202, D204–D206 | O25 |
| Post-training | D29–D31, D36, D37, D76, D91–D94, D98–D101, D103–D105, D107, D110, D113–D117, D123–D125, D129, D130, D132, D137, D157, D161, D162, D165, D167–D171, D173–D176, D185, D207, D208 | O15, O26 |
| Multi-aircraft control | D140–D146, D148–D153, D164, D166, D172, D179–D183, D186, D188–D195, D200 | O18–O21 |

The next free numbers: D209, O27.

---

## 4 Plan

Stages A–D are built: each stage's §0.3 gives its state, its implementation log the steps. What is left, in this order:

1. B21 (prior §0.4): B20's campaign on `v19_20261014` / `v24_20261014` with D204 and the stop of prior D205, without
   the landings input (prior D206; the user, 2026-10-14): the folds, the choice, the base and its one val read. Prior
   O25 (wind) follows it.
2. One method of branch training for stages C and D (multi-aircraft control D200). Its comparison is held (the user,
   2026-10-13: "先查原因，对照暂缓"); it starts from the new base, B21's (the user, 2026-10-14), with the reward
   of post-training D207 (the runway of a landing does not change it) and D208 (the loops keep no landings).
   Post-training O26 (the window's landings that only D175 reads, the landings' digest in a prior's identity)
   waits for the user.
3. Version 2 of the multi-aircraft control (multi-aircraft control D153, MC8–MC10).
4. The test days at the end: the check of the labelling (D198), KAUS in its three parts (D158).
5. The frontend and the backend change last, each change with the user's permission (§5 rule 11).

---

## 5 Rules of the implementation

These rules hold at every stage. The plan of each stage is in its document; before a milestone, read the design
sections that it names.

1. The branches (the user, 2026-10-13: "two-tier2 维护这个就行 其他的都是临时分支"). The user keeps one branch,
   `two-tier2`. A change is built on a temporary branch from `two-tier2`, in its own worktree; after its review and
   tests, `two-tier2` is moved to it by its ref (`git update-ref refs/heads/two-tier2 <new> <old>`, never a merge
   inside a checkout), and the temporary branch and its worktree are deleted (each worktree's data links unlinked
   first; `git branch -d`, never `-D`). No other branch moves: not `dev-two-tier` (the main checkout, from which the frontend and the backend
   run), not a stage branch. A push of `two-tier2` waits for the user's word. The implementer stages files by
   explicit paths and reads `git diff --cached --stat` before each commit. A formal build or run starts from a run
   worktree at a `two-tier2` commit (D163), never from a development worktree or the main checkout; it reads
   recorded paths through `this_checkout` (post-training D157).

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
8. Do not touch the run worktrees of running builds and campaigns, or the main checkout.
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
    text that the user decides. The log is committed to `two-tier2` (rule 1), with explicit paths. At
    the end of a stage, the report gives the new code index, and Claude puts it into the key code index of the
    document.
11. The frontend and the backend change last (the user, 2026-10-12: "以后不要做任何前端改动 服务的后端，也要我的
    允许才能改。因为前后端改动意味着重要的设计修改"; "这是最后阶段的事情"). A change of `aeroviz-4d` (its code, tests and
    fixtures) or of `aeroviz_backend` waits for the user's permission, after the numbers of the main task. A change of
    a format that the Training view or the backend's live executor (`aeroviz_backend/autopilot_segment/`) reads first
    lists what it forces there; until the user allows it, its mirror tests stay red and are named in the commit. The
    sets published in an old format stay on disk, and the new frontend does not read them (principle 8). The
    services run from the main checkout, which the user moves.
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
