# Two-tier model: the Training view

**Summary.** The Training view is the page of the frontend (`aeroviz-4d`) where the user sees what each stage of the
two-tier model does: the words, the sentences that the labeller reads and that the models say, the tracks that the
executor flies, and the results of the experiment that made each set. The four stages (A, B, C, D) share one layout,
built from shared parts; stages C and D share one window view. This document gives the rules of the view, the layout,
each stage's view, the shared controls, the backend's routes that serve the view, and the content of the Training sets.

**Language.** This document uses the writing rules of ASD-STE100 (Simplified Technical English): short sentences, active
voice, one meaning for each word, and technical names that the terms define. The words were not checked one by one
against the STE dictionary. Units are SI only (m, m/s, deg, s).

**Scope of this document.** Paths in backticks are relative to `4dTrajectory/ts_transformer/` unless they start with
`aeroviz-4d/`, `aeroviz_backend/`, `4dTrajectory/` or name the repository root; frontend paths without a prefix are
under `aeroviz-4d/src/`; `readouts/` is `docs/two_tier/readouts/`. Moved here from the outline on 2026-10-06 (the
user): outline §6.1 is §2 here, outline §6.2 is §3 with the same item numbers, and D133–D136 are this document's.

**State of this document.** Written 2026-10-06. The one layout (§3) is built and reviewed on `dev-two-tier-v4`; its
steps after C10 are stage D's implementer's (outline §4 item 5). The cursor slider (D155) and stage D's view (D156) are
designed and not built; fronter builds them (D154, §8).

---

## 0 Status

### 0.1 Decisions

The decision numbers are shared by all documents (outline §3).

| # | Item | State | Source |
|---|---|---|---|
| D133 | The Training views of stages A, B and C share one layout (§3). The user decided: one layout with the switch between the stages kept; a set's intent read from `docs/experiments/intents.json` through the backend, with no export, set or index format changed; stage C's branch merged before the views change; stage B's implementer builds it, and for this work may change the view files of stages A and C and the backend's routes. Claude's design of the parts, on stage B's request: the left panel holds the set chooser with the set's own line of the intent registry, the item list with the same four columns, one-line readouts and the Draw switches, and no table; the sentence bar's tabs choose the sentence on screen in every stage (B: Labelled, Closed loop and the samples; C: Labelled and the rounds), and the bar's chips and notes follow the kind of that sentence; the details page in every stage, its first section the set and the experiment; what follows the cursor (B's probability of "go-around" at the cursor) stays in the left panel, since the details page is modal; shared parts, not copies. Why: B's and C's views chose the sentence in a table of the left panel, so that the bar's tab "Δ 4 s" meant the table's choice; B's details page was disabled and C's ⓘ opened nothing; no view showed a set's intent | Decided | User, 2026-10-06 (the layout, the intent's source, the order); the parts: Claude, on stage B's request |
| D134 | The Training view's details page holds the results of the experiment that made the set (§3 items 3, 4): stage A — the labelling (train and select only), the executor's closed-loop replay at each Δ, the set's flown flights, the vocabulary; B — the free generation the set was made from, the training, the validation (the base's val set only), the choice; C — the rounds, the checks at the campaign's start, the window. The first section is the set and the experiment, with one line of provenance; "What this view shows" and the row inspector are removed. The backend reads the results (`GET /training/results`) from the files the set names under `4dTrajectory/outputs/` and answers named fields only; no answer holds a reading that the design keeps unshown (stage A's val block, the val counts of a run's identity), and B's validation only for the set of the claimed readout. It replaces the sections of D133. Why: the user found D133's page wrong: its sections (the intent and a list of paths, usage notes, the set's own 20 sentences counted, a row inspector that repeats the bar) were Claude's reading of the page before A23 as a place for long text; that page held the experiment's results | Decided | User, 2026-10-06 (the results, read through the backend, the removals); the sections and the route: stage B's proposal, with Claude's guards of the val days |
| D135 | Every sentence on screen is drawn with the envelopes of its words (§3 items 6, 7), simpler than the instruction-v3 view: lateral, each heading word's judged rows on the ground with the rows outside its band in red; vertical, each altitude word's tube as a wall with its two edges; the speed bands in the charts only; no capture turn or corridor, and only the envelopes of the sentence on screen. The export computes them: one function of stage A's Training export (`flown_sentence`) gives the block of every flown sentence in every stage (the outcome, the crossing, the track unrounded, the attitude, the envelopes), and one writer of the Training files serves the three stages. The three sample formats get new names; stage A's and B's published sets are exported again with the same ids, and the old ones are deleted once the new view is merged and checked. Stage B's implementer builds it on `dev-two-tier-v4` (outline §5 rule 1) and may change stage A's and C's Training exports for it. The trained base is not affected: B's export opens it read-only and writes only under `aeroviz-4d/public/data`, the base's directory is read-only with its `SHA256SUMS`, and stage C opens the base itself and reads no Training set. Why: B's and C's exports wrote no envelopes, so their sentences were drawn without them, and three copies built the block of a flown sentence | Decided | User, 2026-10-06 |
| D136 | The Training view shows how fast the model speaks (§3 item 10): a runner (`model_speed`) measures, on select-day flights or windows and never on val or test, the time of the prior's step of a row and of the executor's steps of it, apart, on the CPU (one thread) and the GPU, for one aircraft at a time and for a batch of 400, with p50, p95 and the largest time of a row, the rows per second and the share of Δ = 4 s; stage B's base now, stage C's chosen round after C10; the sets name the readout and the details page shows it in a section Speed. Why: the view showed only the executor's computing time of a live segment; no readout recorded the prior's time to say a row, and only C8's profile timed the prior's step, for one batch of windows | Decided | User, 2026-10-06 (the speed in the view); the measurement and its settings: Claude's proposal |
| D154 | **The Training view has its own design document and its own implementer.** This document holds the rules and the layout of the Training view of every stage (was outline §6), the view of each stage (§4, §5) and the shared controls (§6). Fronter, an implementer of its own, builds and keeps the Training view: the frontend's Training page, the backend's Training routes and live segments, and the Training exports of every stage with their formats (§8). The stages' implementers build the data that the exports read; an export reads its stage only through that stage's public interface (vocabulary §6 item 8, prior §7 item 8, post-training §9 item 13, multi-aircraft control §10). Why: each stage's view was built by that stage's implementer, so parts were copied before they were shared (D133), and a fourth stage brings a fourth view | Decided | User, 2026-10-06 |
| D155 | **The cursor is one slider, the same in every stage** (§6.1). Under the readouts of the left panel, a slider moves the one cursor of the view over the rows of the sentence on screen: by drag, by a click on its track, or by keys. Its track carries each stage's own marks: B, the probability of "go-around" along the sentence (until now a strip that a click moved the cursor on); A, the rows of correction words; C and D, the losses of separation and, in D, the aircraft's silence; every stage, the first predicted step. It replaces B's strip; the sentence bar's steps and bands and the read-back charts still move the cursor. Why: the user (2026-10-06): moving the aircraft along its track from the left panel is good, but it should be a slider, and one control for stages A, B, C and D | Decided | User, 2026-10-06 (a slider common to every stage); its form: Claude |
| D156 | **Stage D's view is stage C's window view with several commanded aircraft** (§5). One window view serves stages C and D: a window holds a list of commanded aircraft (stage C's hold one), one of them selected. The sentence bar shows the selected aircraft's sentence with stage C's tabs; one line of the left panel, the aircraft strip, shows each commanded aircraft's outcome in the round on screen and selects one; the scene draws every commanded aircraft, the selected one with its envelopes. Stage D is a fourth entry of the stage switch with its own index; the window sample gets one new format for stages C and D. D's view adds no colour and no second chooser. Why: the user: one view if stage C's view can take stage D with a small change, a view of its own only if the change is large. Claude's count of stage C's window view (2026-10-06): about 250–400 lines in 6–8 files of the frontend, about 30 of the backend's window segment and 50–100 of the export change, against a copy of about 1,000 lines | Decided | User, 2026-10-06 (one view when the change is small); the count and the design: Claude |

### 0.2 Open items

None.

### 0.3 Implementation

Fronter keeps this table and its log, `readouts/2026-10-06_fronter_implementation_log.md` (outline §5 rule 10).

| Part | State |
|---|---|
| The one layout (§3, D133–D136) | Built and reviewed by stage B's implementer on `dev-two-tier-v4` (`65314499`, `92a56314`, `550fb0c2`, `67ff4ce9`, `a3cd6ae3`, `d870921f`; Claude's check `readouts/2026-10-06_training_view_and_speed_check.zh.md`). After C10, by stage D's implementer: the base's speed readout, the export of A's and B's sets in the new formats, the browser check (outline §4 item 5) |
| F1 The cursor slider (D155) | Not started |
| F2 Windows of several commanded aircraft: the format, stage C's export, the view's readers (D156) | Not started |
| F3 Stage D's parts of the window view (D156) | Not started; after stage D's MC0 |
| F4 Stage D's export and sets | Not started; after stage D's MC4 |

### 0.4 Plan

1. F1 now, on fronter's branch (§8).
2. F2 now: the window sample of several commanded aircraft with stage C's windows of one.
3. F3 after stage D's MC0 is on `dev-two-tier` (post-training §9 items 1, 3 and 8: a window of several commanded
   aircraft, its losses, its loop), on fixtures that stage C's export writes for a synthetic window of two.
4. F4 after stage D's MC4 (its runners): its export and a smoke set; its formal sets after MC6, their intent first.

The user merges fronter's branch.

---

## 1 Scope

- **This document owns** the Training view of every stage: the frontend's Training page (`components/TrainingPanel.tsx`,
  `components/Training*.tsx`, `components/training/`, `data/training*.ts`, `hooks/useTraining*.ts`,
  `scene/trainingEntities.ts`, `utils/trainingWordColors.ts`), the backend's routes that serve it (`GET
  /training/results`, `GET /experiments/intent`, the live segments `aeroviz_backend/autopilot_segment/`), what each
  stage's Training set holds and the names of its formats.
- **It reads** the outline; each stage's public interface for its Training export (vocabulary §6 item 8: `flown_sentence`
  and the one writer of the Training files; prior §7 item 8: the procedure block; post-training §9 item 13: the window
  sets) and the stage's decisions that the view shows (D109, D127, D129, multi-aircraft control D140–D152).
- **It gives** the user's view of every stage, and the check that a published set opens and flies (§2 item 6).

---

## 2 Rules

Each stage ends with a milestone that brings the view to that stage (the user, 2026-10-04): vocabulary A23, prior B6,
post-training C11; for stage D, fronter's F2–F4 (§8). The user sees each stage in the view before the next one builds
on it.

1. **What follows the stage.** The backend's live executor (`aeroviz_backend/autopilot_segment/`, a click on a word
   flies its segment), the Training export of the stage's results, and the frontend's Training view (`aeroviz-4d`). The
   archived exports and display code (`archive/two_tier_v3_2026_10/`: `instructions/training_files.py`,
   `instructions/display.py`, the `*_training_export.py` runners, `experiments/training_attitude.py`) come back,
   rewritten, in the stage that needs them.
2. **Formats.** Every changed export format gets a new schema name, and the frontend refuses an old one (outline principle 8).
   The frontend reads the reading name and the format names, not the spec sha (D21). The fixtures of the frontend's
   tests are written by the export code, not by hand.
3. **Beside the old sets.** The worktree's `aeroviz-4d/public/data/airports` is a link to the live data (outline §5 rule 1).
   A stage's export writes its own index under a new name beside `training/index.json`; the old index and its sets
   stay unchanged, so the Training view of the main checkout keeps working until the user merges.
4. **Publication.** The intent of each published set is in `docs/experiments/intents.json`, committed before the
   export. Train and select flights; val flights only from a readout that a plan already makes (the base model's one
   validation readout, prior B5), never a new read of the val days. That set opens and flies live as the others do
   (D109).
5. **The user's view.** A test stack from the stage's worktree: the backend and vite on their own ports (outline §5 rule 13 for
   the memory). The report gives the address and the stop command (`kill $(lsof -t -iTCP:<port> -sTCP:LISTEN)`). The
   main checkout's services are not touched (outline §5 rule 8).
6. **The check.** A published set opens and plays in the browser, and a live segment equals the flown states of the
   export from the same state with the same words (the executor conformance tolerance). The browser check runs in a
   one-shot subagent that returns a short verdict; HTTP 200 is not "it loads".

## 3 One layout for every stage (D133–D136, D155)

Stages A, B, C and D keep their own sets, readers, sessions and live segments, and the switch between them at the top
of the left panel ("Sets of": stage A, B, C or D, offered where the airport has their sets). Their views share one
layout, built from shared parts; stages C and D share one window view (§5, D156). The items below were written for
stages A, B and C (D133); stage D's rows are added, and §5 gives the rest of its view.

Why: B's view (prior B6) and C's (post-training C11) were built on one template that chose the sentence in a table of
the left panel; the sentence bar's tabs then read "Labelled" and "Δ 4 s", and "Δ 4 s" meant the table's choice. B's ⓘ
was disabled, and C's ⓘ opened nothing, because only stage A's session draws the details page. No stage's view showed
the intent of its set, or the results of the experiment that made it. The sentences of B and C were drawn without the
envelopes of their words, because their exports wrote none (D135).

1. **The left panel**, top to bottom, the same in every stage:
   - the stage switch;
   - the set chooser: a choice of set (its id and its count of items), and under it one line: the set's own line of
     the intent registry (item 5) and a SMOKE tag for a set whose source says smoke (stages B and C; stage A's sets
     have no such flag). The line opens the details page on "The set and the experiment";
   - the list of items (flights; in stage C, windows), with four columns in every stage: the callsign; the runway
     (stage C: the recorded runway, post-training D129); the stratum (stage C: the window's kind); the outcomes of the
     item's own sentences, as the number landed of all (stage A: its closed-loop sentences, one for each Δ; B: the
     prior's samples; C: the rounds, and the count of other aircraft), with each sentence's outcome in the tooltip;
   - the readouts of the item, one line each: stage A — the flown flight at the tab's Δ (the outcome and its time, the
     DA check); B — this sentence (the outcome and its time, the DA check, the go-arounds) and, at the cursor, the
     probability of "go-around", whether the procedure permits one, on final or not and the count of words blocked in
     each masked column (the probability of "go-around" along the sentence is drawn on the track of the cursor slider);
     C — this round (the outcome, the reward, the loss of separation or none) and the window (its kind,
     its other aircraft by role). These lines follow the item and the cursor; they open no page;
   - the cursor slider (§6.1, D155), one line, for the sentence on screen, in every stage;
   - the Draw switches: the envelopes' first (item 6), then the stage's own.

   No table is in the left panel, and no block of more than one line.

2. **The sentence bar's tabs choose the sentence on screen**, in every stage. The left panel has no second chooser:
   one state of the app holds the choice, and the session gives the bar its tabs.

   | Stage | The tabs |
   |---|---|
   | A | Labelled · Δ 2 s · Δ 4 s · Δ 8 s (one for each Δ of the set) |
   | B | Labelled · Closed loop · 0 · 1 … (the prior's samples, by their number) |
   | C | Labelled · Start (base) · r1 · r2 … (the rounds) |
   | D | Labelled · Start · r1 · r2 … (the rounds, of the selected commanded aircraft, §5.3) |

   Labelled is the labeller's open-loop reading of the observed flight (stage C: of the recorded flight of the
   commanded aircraft; for a window of kind B, its tooltip says that the window starts from a moved start). B's Closed
   loop is stage A's closed-loop sentence of the flight at the prior's Δ. A tab of a sentence with an outcome carries a
   small dot in the outcome's colour (`trainingOutcomeColour`), and its tooltip gives the outcome, the time and the
   go-arounds. Many tabs wrap onto a second line (the bar measures its own height); the arrow keys move between the
   tabs, Home and End go to the first and the last. A tab chosen stays chosen on the next item of the same stage when
   that item has it; else the stage's first choice (A: the set's first Δ; B: the first sample; C: the last round). A
   word of the sentence on screen flies live as now: stage A's closed-loop sentences on stage A's segment, B's closed
   loop and samples on the prior's segment, C's rounds on the window's segment; a labelled sentence does not fly.

   The bar's chips and notes follow the kind of the sentence on screen: a closed-loop sentence — the outcome, the DA
   check and the corrections; a prior sample — the outcome, the DA check and the go-arounds; a round — the outcome, the
   reward and the loss of separation; a labelled sentence — none. The bar's ⓘ holds the notes on how to read the bar,
   the tabs and the chips, written for that kind (only the notes of a closed-loop sentence speak of words that the
   reading added). Their first line names the set and gives its intent line, with a link to the details page's "The
   set and the experiment".

3. **The details page is the experiment's results (D134).** Every stage has the details page (`TrainingDetails`), and
   the ⓘ of the panel's header is never disabled: it opens the page on its first section. With no session on screen,
   the page opens on its first section and says why it is empty. The page holds the results of the experiment that
   made the set, as the Training view before A23 did (`TrainingResults.tsx`: the executor's replay gate, the prior's
   val readout per column, each model's landed beside its free generation) — not usage notes and not the set's own
   items again.

   | Stage | Sections after "The set and the experiment" | Read from (named by the set's `source` / `model`) |
   |---|---|---|
   | A | **Labelling**: labelled and refused, the reasons of the refusals, by airport — train and select only · **Closed loop**: the executor's replay of the closed-loop sentences of each split of the set at each Δ (landed and each other outcome) · **Flown flights**: the set's own flights at each Δ · **Vocabulary** | `instruction_language/<artefact>/readout.json` (its train and select blocks); `executor/<spec>/replay-closed-<split>-<Δ>s/replay.json` |
   | B | **Free generation**: the readout the set was made from — the outcomes by airport and by stratum, the sentences inside the selection and outside it (for a faulty track, for the outcome), the words per sentence beside the labelled ones; beside it the set's own sentences counted · **Training**: the loss per epoch on train and select, the best epoch · **Validation** (the base's val set only): the teacher-forced loss per step and per column, the share of the labelled words that the procedure masks block · **Speed** (item 10) · **The choice**: each configuration's and variant's score over its folds, the seed scale, the chosen configuration and variant | the free-generation `readout.json` of `source.readout`; `history.json` of the prior; `validation/readout.json` beside the base; `choice_configuration.json`, `choice_variant.json` of the campaign; the speed readout that `source.speed` names |
   | C | **Rounds**: each round of the campaign — the reward, landed, the losses of separation, the go-arounds · **Speed** (item 10, after C10) · **The checks**: the labeller's and the executor's checks at the campaign's start · **The window**: its kind, its moves, its other aircraft, the round's end, the threshold, the rows where the speed-word mask acted, the faulty points | `campaign.json` and `round_<n>/round.json` of `model.campaign`; the speed readout that `source.speed` names |
   | D | **Rounds**: each round — W per commanded aircraft, landed, go-arounds, silent aircraft, the losses of separation by pair (multi-aircraft control D145), the share of informative groups · **Speed** · **The checks** · **The window**: its kind and c, its commanded aircraft (each one's join time and outcome), its recorded aircraft, the losses by pair, the end | as stage C, of stage D's campaign |

   "The set and the experiment" holds the campaign's title, intent and design, the set's own line (item 5) and one line
   of provenance: the path of the readout or campaign the set was made from. No section shows a reading that the
   design keeps unshown: stage A's val block (vocabulary: the replay of the val days waits for the user; D85) is never
   read into an answer; B's Validation section and the free generation of the val days are shown only for the set
   exported from the base's claimed, written validation readout (D109; prior D119), and never the val counts of a run's
   identity (prior D120); stage C's validation readout is not shown until the post-training's design says so.

4. **The results route (D134).** `GET /training/results?stage=<A|B|C|D>&airport=<ICAO>&set=<id>`: the backend finds the
   set in that stage's index of that airport, reads only the files that the set's `source` and `model` name, each
   under `4dTrajectory/outputs/`, and answers the named fields of item 3's sections (never a whole file). A set whose
   files are elsewhere (a smoke set in a scratch directory) or missing gets an answer that names them, and the page says
   so: an airport that is no airport code 400, a set of another format 409 by name, a section whose file is of another
   format, elsewhere or missing that section's problem by name, the other sections shown (D139). It reads at each
   request and writes nothing.

5. **The intent.** `GET /experiments/intent?run=<set id>` reads `docs/experiments/intents.json` at each request (an
   entry committed later shows without a restart of the backend) and finds the set's id among the `runs` keys of every
   campaign. One campaign: the answer gives the campaign's id, title, intent and design and the run's line. No
   campaign (404), more than one (409: run keys are not unique in the registry, `S20_cv_s1337` is a run of two
   campaigns) or no run asked (400): the answer names the set and the campaigns found; the view shows it by name, and
   the set still opens. The view keeps an intent it found for the page's life. The backend writes nothing. The
   frontend shows an intent in the one form of the experiments picker (`ExperimentIntentBlock`, taken out of
   `ExperimentDetails` unchanged).

6. **The envelopes of the words (D135).** Every sentence on screen is drawn with the envelopes of its words, on the
   track they are judged on: the labelled sentence with the labeller's envelopes on the observed track; every sentence
   that the executor flew (stage A's and B's closed loop, B's samples, C's rounds) with the judge's envelopes on its
   flown track, each word from where the executor heard it. The export computes them, never the view. The views draw,
   for the sentence on screen only:
   - **lateral** (Draw "Heading bands"): each heading word's judged rows as a line on the ground under the judged track,
     and the rows where the track is outside the word's ±4.5° in red over it;
   - **vertical** (Draw "Altitude tubes"): each altitude word's tube as a wall over the judged track's ground position,
     between its lower and upper edge, and the two edges as lines in the tube's verdict colour;
   - in the read-back window's charts, as now: the heading bands, the tubes' edges, the speed bands.

   The selected word's envelope is highlighted as in stage A's view now. Not drawn (the instruction-v3 view had them):
   the capture turn and its corridor (words of instruction-v3), and the observed track's envelopes under a model's
   sentence.

7. **Shared parts, not copies**, in the exports and in the views:
   - **One flown sentence.** One function of stage A's Training export (vocabulary §6 item 8, `flown_sentence`) gives
     the block of every sentence that the executor flew, in every stage: the outcome, the end cycle, the crossing and
     the DA check, the track to the outcome on the 2 s rows, unrounded (the rule of prior D127, now for every stage),
     the attitude and the envelopes. A stage adds only its own fields (B: the timeout, the blocked words; C: the
     reward, the loss of separation). Today three copies build this block (stage A's `replay_payload`, B's
     `fly_again`, C's `sentence_payload`) and none of B's or C's has the envelopes. The labelled sentence keeps
     `envelopes` on the observed track.
   - **One writer of the Training files.** The index and set reading and writing (`listed_set`, `read_index`,
     `serialise`, the atomic write, `require_writable`, `write_set`) is one definition in `instructions/training_files.py`,
     given each stage's constants (its index file, its schemas, its set kind); today the three `training_files.py`
     carry the same code. The bytes written do not change.
   - **The views.** One reader of the flown-sentence block and its envelopes (stage A's) for every stage; one hook that
     loads a set; one set chooser; one item list; the bar's tabs; the details page and its state, given to every
     session; the Draw box with the envelopes' switches; the intent block. `ProblemBox` and `trainingText` stay the one
     definition of problems and outcome words. Each stage keeps its reader of its own fields, its session and its
     sections.
   - **Formats and sets.** The sample formats of the three stages get new names (outline principle 8). Stage A's
     `closed_loop_v12_20261005` and stage B's `prior_sets_20261006` are exported again in the new formats, with the same
     set ids and intent entries, under new index files beside the old ones; the old indexes and their sets are deleted
     once the new view is merged and checked (the user, 2026-10-06). Stage C has no published set. The exports open the
     base, the folds and the artefact read-only and write only under `aeroviz-4d/public/data`; the base's val set is
     the claimed readout's sentences flown again, no new read of the val days (D109).
8. **Tests and the check.** Vitest for each shared part and for the envelopes of an A, a B and a C sentence, on
   fixtures that the exports write (§2 item 2); every test of the three stages kept (among them: a set of another
   airport never opens, a stale choice after a set changes, the stage switch, the cursor at a window's row 0). The
   exports: the shared writer writes the same index bytes as before; `flown_sentence` gives stage A's earlier block
   within its earlier rounding (checked against each fixture's earlier block by its sha256, held in the test, D139), and B's and C's earlier blocks exactly, apart from the added envelopes; every export
   refuses an envelope that ends past the flown track; the base's directory has the same `SHA256SUMS` after B's
   export. The speed runner on synthetic artefacts: every time positive, each setting named, no val or test flight
   read, nothing written into the model's directory. The backend: the results answer (each stage; files missing or elsewhere; no val block of stage A in any
   answer) and the intent answer (one campaign, none, more than one, none asked). The browser check of §2 item 6 in
   each stage: the tabs choose the sentence, the details page opens from every ⓘ and from the set's line, the results
   and the intent show, B's section Speed shows the base's readout, every flown sentence shows its heading bands and
   altitude tubes, a word flies live.
9. **Who and where** (the layout's build, done). Stage B's implementer (the user, 2026-10-06), on `dev-two-tier-v4` in the worktree
   `.claude/worktrees/two-tier-v4` (outline §5 rule 1), after its first part on `dev-training-layout` is merged there. For this
   work only, it may change the view files
   of stages A and C, `ExperimentDetails.tsx`, the backend's routes, and the Training exports and their formats of
   stages A and C (`experiments/training_export.py`, `instructions/training_files.py`,
   `experiments/post_training_export.py`, `post/training_files.py`), and run the exports of item 7; outline §5 rule 1
   otherwise holds. Its log is a section of stage B's implementation log; a
   reading where this section says nothing is a proposal in stage B's requests note. While it is built, no other work
   changes these files: `aeroviz-4d/src/components/Training*.tsx`, `aeroviz-4d/src/components/training/`,
   `aeroviz-4d/src/data/training*.ts`, `aeroviz-4d/src/hooks/useTraining*.ts`, their tests,
   `aeroviz-4d/src/components/ExperimentDetails.tsx`, `aeroviz_backend/http_server.py`, `experiments/model_speed.py`,
   `experiments/training_export.py`, `instructions/training_files.py`, `experiments/prior_training_export.py`,
   `prior/training_files.py`, `experiments/post_training_export.py`, `post/training_files.py`. The test stack
   and the report follow §2 item 5. The layout is built and reviewed; its steps after C10 (the base's speed readout,
   the export of A's and B's sets in the new formats, the browser check) are stage D's implementer's (outline §4 item 5)
   and change no file of the view, so the lock of these files is lifted: the Training view is fronter's (§8).
10. **The model's speed (D136).** The view shows how fast the model speaks; until now it showed only the live
    executor's computing time of a flown segment ("computed M ms"), and no readout recorded the prior's time to say a
    row (free generation records no time; C8's profile times the prior's step of one batch of windows only). A runner
    (`experiments/model_speed.py`, with its tests and an entry in `docs/reference/runners.md`) measures it on fixed
    inputs:
    - **What:** a model run (stage B: the base; stage C: the chosen round, after C10, with its traffic attention and the
      separation masks), speaking select-day flights (B) or select windows (C) to their end through the shared step of a
      closed loop (prior §7 item 7; C's window loop), one sample at temperature 1. No val or test day is read.
    - **Times, each apart:** the prior's step of a row (its inputs, the model with its cache, the masks, the draw; in C,
      the traffic attention and the separation masks too) and the executor's steps of the row; on the GPU each timed
      between two synchronisations; the first 20 rows after loading are not timed.
    - **Settings** (Claude's proposal; the user may change them): the CPU with one thread (as the closed loop runs) and
      the GPU; one aircraft at a time (the time a controller in the loop waits) and a batch of 400 (free generation's
      chunk in B5; the 100 drawn flights repeated in turn, copy k of a flight its sample k, D139); 20 flights of each
      airport (seed 1337; C: the select windows of its selection readout).
    - **Readout:** for each setting, the time of a row (p50, p95, largest, in ms), of a sentence (s), the rows per
      second, and the share of Δ = 4 s that one row takes (the prior's step and the executor's); the device, the torch
      version, the threads, the load of the host at the start and the commit, as information. Written into a new
      directory, `4dTrajectory/outputs/POOLED/speed/<model run>_<date>/` (never into the read-only directory of the
      model). It runs with no other job on the host or the GPU (outline §5 rule 13).
    - **The view:** the sets of B and C name their model's speed readout (`source.speed`, written by the export, which
      requires `--speed` in both stages; a fold set of B names the base's readout and says which model it timed; B's
      sets are exported again after it, item 7), the results route reads it (item 4), and the details page shows it in
      its section **Speed**, the setting of each number beside it. Stage A's view keeps the executor's time on the live
      line.

---

## 4 The view of stages A, B and C

What each stage's view shows, as its milestone built it (the specifications: vocabulary A23, prior B6 and post-training
C11 in the stages' implementation logs) and as §3 lays it out. The cursor slider (§6.1) is the one change.

### 4.1 Stage A: the words and the closed loop (vocabulary A23)

- **Sets.** For each airport, a seeded sample of select and train flights (seed 1337), from the formal artefact: the
  observed track; the labelled sentence on the 2 s rows; at Δ = 2, 4 and 8 s the closed-loop sentence, each correction
  word marked, and its flown states; the judge's outcome and the DA check; the attitudes.
- **View.** The five columns (runway and G, heading relative to the course of R, altitude above E, angle, speed); the
  correction words marked; a tab for each Δ; the flown path beside the observed one; the outcome and the DA point.
- **Live.** A click on a word flies its segment with the single-flight executor (`autopilot/single.py`) under the
  formal artefact's executor spec (`POST /autopilot/segment`). The view refuses an old format by name.

### 4.2 Stage B: the prior's sentences (prior B6)

- **Sets.** Each fold of B5 at its held-out airport (the flights of its free generation) and the base model's one
  validation readout (only when claimed, D109). For each flight: the observed rows before the first predicted step; the
  prior's sentences side by side and their flown states, each flown again and equal to its readout within the
  executor's bound, its track unrounded (D127); stage A's closed-loop sentence of the same flight; the outcome and the
  DA check of each; at each row the words that the procedure masks blocked; the region, the glidepath lower edge, the
  DA and the entry height of each candidate.
- **View.** Stage A's view with the prior's sentences: a tab for each sample and the closed loop; the blocked words at
  the cursor; the procedure's limits drawn; the outcome.
- **Live.** A click on a word flies its segment with stage A's executor (`POST /autopilot/prior-segment`), refused past
  the executor's bound from the exported track or with another outcome (D127). The base's val set opens and flies; no
  other set with a val flight opens (D109).

### 4.3 Stage C: the window of traffic (post-training C11)

- **Sets.** Windows of recorded traffic with the commanded aircraft flown on the words of each round of the
  post-training and the other aircraft on their records; the separation judge's events; each round's outcome; the
  rounds side by side.
- **View.** The commanded aircraft's five columns in the bar; the other aircraft by role in the scene (recorded: slate;
  inserted, kind A: pink; moved, kind D: amber); a loss of separation as a red line between the two aircraft at the
  loss's time, with two markers that give the distance and the distance required. The list names a window's runway as
  its recorded runway; the shift of a window A is shown in days or hours; the cursor starts at the window's row 0, so
  that the other aircraft show from the start (D129).
- **Live.** A click on a word flies its segment in the window (`POST /autopilot/window-segment`), the other aircraft on
  their records.

---

## 5 The view of stage D (D156)

Stage D commands every arrival of a window's span (multi-aircraft control D152). Its view is stage C's window view in
which a window holds several commanded aircraft. It shows one thing more than stage C's: who the model commanded and
how each of them ended. One aircraft's sentence is on screen at a time, as in every stage.

### 5.1 The left panel

- **The stage switch** offers D where the airport has stage D's index; **the set chooser** as in every stage.
- **The item list:** the windows, with the four columns of every stage: the callsign (the anchor's, with "+k" for its
  k other commanded aircraft); the runway (the anchor's recorded runway, as D129); the stratum (the window's kind: real,
  or compressed with its c); the outcomes (the commanded aircraft landed of all in the last round, for example 5/6;
  each round's in the tooltip).
- **The readouts**, one line each:
  - this round: W and the count of commanded aircraft, landed, go-arounds, silent aircraft, the losses of separation
    (between commanded aircraft; with recorded aircraft);
  - this aircraft: its callsign, its outcome and its reward r in the round, the row from which it is silent (if it
    is), its join time in the window;
  - the window: its kind and c, its commanded aircraft, the most recorded aircraft at one time.
- **The aircraft strip** (§5.2), one line.
- **The cursor slider** (§6.1), for the selected aircraft's sentence.
- **The Draw switches:** the envelopes' first (§3 item 6, the selected aircraft's sentence only), then "Other commanded
  tracks" (on at the start), then stage C's.

### 5.2 The aircraft strip

One line of the left panel that shows the window's commanded aircraft and chooses one:

- one small square for each commanded aircraft, in the order in which they join the window; its fill is the aircraft's
  outcome in the round on screen (`trainingOutcomeColour`); on the Labelled tab, every square is neutral;
- a silent aircraft's square has one diagonal stroke; the selected aircraft's square is outlined;
- after the squares, on the same line: the selected aircraft's callsign and "k of n";
- a click on a square, or on a commanded aircraft in the scene, selects it; the keys `[` and `]` select the one before
  and after; the tab stays, and the cursor keeps its instant (§6.1);
- a window with one commanded aircraft (every window of stage C) shows no strip.

### 5.3 The sentence bar

The selected aircraft's sentence, with stage C's tabs: Labelled (the labeller's reading of the aircraft's recorded
flight) · Start · r1 · r2 …. The bar's first chip names the aircraft ("AAL123 · 2 of 6"). The chips of a round: the
aircraft's outcome, its reward r, its silence (from its row) or its loss of separation; the window's W of its count as
the last chip. The bar's ⓘ notes of a round tell how r and W are read (multi-aircraft control D140, D141). A word of
the sentence on screen flies live (§5.5).

### 5.4 The scene

- **The selected aircraft** as stage C's commanded aircraft: its observed rows, its flown track of the round in teal,
  the envelopes of its words, its live flight in blue.
- **Every other commanded aircraft:** its flown track of the round in the same teal, thinner and at half opacity, and
  its model at the cursor, from the row it joins to its end. Draw "Other commanded tracks" hides the tracks, never the
  models.
- **A silent aircraft:** its track from the row it became silent dashed, in the slate of the recorded traffic: from
  there it flies on its words in force and the model no longer speaks to it (multi-aircraft control D144).
- **The recorded aircraft** as in stage C.
- **Each loss of separation of the round,** between any two aircraft, as stage C draws one: a red line between them at
  the loss's time and two markers with the distance and the distance required. A loss that the records also have and
  that no commanded aircraft answers for (multi-aircraft control D145) is dashed.

Stage D adds no colour: teal is a commanded aircraft, slate is traffic, red is a loss, as in stage C.

### 5.5 A live flight

A word of the selected aircraft flies its segment in the window (`POST /autopilot/window-segment` with the aircraft
named): the selected aircraft flies the words from its state at the word's row; every other aircraft, commanded or
recorded, flies its states of the round and does not react. The answer is refused past the executor's bound from the
exported track, or with another outcome, as every live segment (D127).

### 5.6 The details page

The sections of §3 item 3's row D, drawn by stage C's section parts with stage D's fields; the results route takes
`stage=D` and reads stage D's campaign.

### 5.7 The set and its format

- **The window sample of stages C and D** (one new format, `aeroviz-training-window-sample-v4`): a window holds its kind
  and c; its commanded aircraft, a list — each one's dataset id, its join offset from the window's row 0 (s), its move
  in a compressed window, its labelled sentence, and each round's flown sentence (vocabulary §6 item 8's block, with
  its envelopes) with its reward and the row from which it is silent; the recorded aircraft on their records; each
  round's losses of separation — the two aircraft, which ones answer, the distance and the distance required, and
  whether it costs W. Stage C's export writes one commanded aircraft.
- **The indexes:** stage C's `index_post_v3.json`, stage D's `index_multi_v1.json`; one index format for both; the
  stage is the file's.
- **D's sets** (`experiments/multi_training_export.py`, multi-aircraft control §10): a seeded draw of the select
  windows of each airport from the formal campaign (seed 1337), every round; a smoke set from MC4's smoke.
- **The window segment** takes the aircraft: a new request name on both sides.

### 5.8 What stage D's view does not have

No view of the branch groups or of the varied aircraft, no map of the attention between aircraft, no second chooser of
the sentence, no table, and no sentence of more than one aircraft on screen at a time. The readouts and the details page
give the counts; the scene gives the place.

---

## 6 Shared controls

### 6.1 The cursor slider (D155)

- **Where:** the left panel, one line under the item's readouts, in every stage, for the sentence on screen. With no
  sentence on screen it is disabled and says "choose a sentence".
- **Form:** on the left, the time at the cursor ("t 128 s"); on the right, the track, from the sentence's row 0 to its
  last row, 12 px high, with the thumb at the cursor and the stage's marks on the track.
- **Input:** drag the thumb; a click on the track puts the thumb there; with the slider focused, ← and → move one Δ row
  (one 2 s row on a labelled sentence), Shift with ← and → ten rows, Home and End to the first and the last row. It is
  a `role="slider"` element with `aria-valuenow` (the row) and `aria-valuetext` ("t = 128 s, row 32 of 210").
- **State:** it reads and writes the one cursor of the view (`trainingCursorS`, `useTrainingCursor`); every reader of
  the cursor follows it: the aircraft in the scene, the word and the runway in force, the traffic, the bar's cursor
  line, the read-back charts. It does not drive the Cesium clock, and it has no playback.
- **Marks on the track** (each stage's session gives a list; the slider draws them; no stage draws a strip of its own):
  - every stage: a tick at the first predicted step;
  - A: a tick at each row with a correction word;
  - B, on a sample's tab: the probability of "go-around" at each row as a line (the line of B's former strip); B's
    readout line "At the cursor …" stays above the slider;
  - C and D: a red tick at each loss of separation that the aircraft on screen is in; in D, a slate band from the
    aircraft's silence to its end.
  The marks use the view's existing colours.
- **In a window** (stages C and D) the cursor keeps the window's instant: a change of the round or of the selected
  aircraft keeps the cursor's UTC time where the new sentence has it, else puts it at the nearer end of the sentence.

### 6.2 One definition of each part

A part that two stages use is one definition (§3 item 7). The shared parts of the view today, and what stage D adds:

| Part | Where | Used by |
|---|---|---|
| The panel's header, its ⓘ and the stage switch | `components/TrainingPanel.tsx` | A, B, C, D |
| The set chooser, the item list, the set and the experiment | `components/training/SetParts.tsx` (`SetChooser`, `ItemList`, `ExperimentSection`) | A, B, C, D |
| The Draw box and its switches, the details link, the readout line, the details page's state | `components/training/PanelParts.tsx` (`DrawBox`, `LayerSwitches`, `DetailsLink`, `ReadoutLine`, `useDetailsPage`) | A, B, C, D |
| The details page and its sections | `components/training/TrainingDetails.tsx`, `ResultSections.tsx` | A, B, C, D (C and D: the same section parts) |
| The bar's tabs | `data/trainingTabs.ts` | A, B, C, D |
| The sentence bar, its chips and its cursor | `components/TrainingSentenceBar.tsx` | A, B, C, D |
| The read-back window and its charts | `components/TrainingReadbackWindow.tsx`, `components/training/TrainingWindow.tsx`, `Readback*.tsx`, `chartKit.tsx`, `chartMarks.tsx` | A, B, C, D |
| A set's loading, its intent, its results | `hooks/useTrainingSet.ts`, `data/trainingSetIntent.ts`, `data/trainingSetResults.ts` | A, B, C, D |
| The track with its envelopes, the aircraft at the cursor, the live flight | `hooks/useTrainingTrackLayer.ts`, `useTrainingAircraftLayer.ts`, `useTrainingLiveLayer.ts` | A, B, C, D |
| The cursor slider (new, F1) | `components/training/` | A, B, C, D |
| The window view: its reader, session, scene layer, live request | `data/trainingWindowSample.ts`, `components/training/TrainingWindowSession.tsx`, `hooks/useTrainingWindowLayer.ts`, `data/trainingWindowAutopilot.ts` | C, D |
| The aircraft strip (new, F3) | `components/training/` | D (C: hidden) |

Each stage keeps its reader of its own fields, its session and its sections. `ProblemBox` and `trainingText` stay the
one definition of problems and outcome words.

---

## 7 Formats

Every changed format gets a new name; the view refuses an old one by name (outline principle 8). The formats of the
layout (§3 item 7): stage A's index `index_v5.json` and sample v11, stage B's `index_prior_v3.json` and sample v4, stage
C's `index_post_v2.json` and sample v3. Stage D's view (D156) renames stage C's: the window sample
`aeroviz-training-window-sample-v4` for C and D, the indexes `index_post_v3.json` (C) and `index_multi_v1.json` (D), and
the window segment's request. Stage C has no published set, so no set of C is exported again.

---

## 8 Who and where; the milestones

**Fronter** (D154) builds the Training view: on its own branch `dev-frontend`, in the worktree
`.claude/worktrees/frontend`, made from `dev-two-tier`; it merges `dev-two-tier` into its branch before each
milestone and before its report; the user merges `dev-frontend`. It changes the files of §1, the Training exports
(`experiments/training_export.py`, `experiments/prior_training_export.py`, `experiments/post_training_export.py`,
`experiments/multi_training_export.py`) and the Training files' modules (`instructions/training_files.py`,
`prior/training_files.py`, `post/training_files.py`), with their tests. It changes no other code of a stage; what it
needs of a stage's public interface goes to that stage's document first (outline §5 rule 1). The rules of outline §5
hold: a review before each commit, explicit paths, light tests while a campaign runs (rule 13: Vitest and pytest of the
changed files on few workers), smoke exports only in a scratch directory, a formal export only on the user's order, the
test stack and the browser check of §2 items 5 and 6. Stage D's implementer runs the exports of §3 item 7 on
`dev-two-tier-v4` after C10 (outline §4 item 5); fronter changes neither stage A's nor stage B's export format before
that export is done.

**F1. The cursor slider** (D155, §6.1). A shared slider in `components/training/`, given the cursor and the marks by
each session; B's strip (`AtTheCursor`'s SVG in `TrainingPriorSession.tsx`) removed, its readout line kept; A, C and
D give their marks. Tests (Vitest): drag, click and keys move the cursor by rows; the marks of each stage; the aria
values; the readers follow the cursor; in a window, a change of the round keeps the instant; every earlier test kept
(the cursor at a window's row 0, D129). The browser check on a test stack with smoke sets in the current formats.
About 150 lines and their tests.

**F2. Windows of several commanded aircraft** (D156, §5.7). The window sample v4 and the index of stages C and D; stage
C's export writes its windows as lists of one commanded aircraft; the window segment's request names the aircraft (the
backend and the frontend); the view's reader, session and scene layer take the list, and stage C's view stays as it is
(no strip for one aircraft). Tests: every test of stage C's view kept, on fixtures that the new export writes; the
export written and read again; the window segment with the aircraft named, and refused for an aircraft not in the
window. About 200 lines and their tests.

**F3. Stage D's parts** (§5.1–§5.6), after stage D's MC0 is on `dev-two-tier`. Fixtures: stage C's export on a
synthetic window of two commanded aircraft (through post-training §9 items 1, 3 and 8). The stage switch's D and its
index; the aircraft strip; the other commanded aircraft and the silent aircraft in the scene; every loss by pair; the
bar's chips; D's readouts; the slider's marks of D; D's sections of the details page and `stage=D` of the results
route. Tests for each, and the selection of an aircraft from the strip, the keys and the scene. About 300 lines and
their tests.

**F4. Stage D's export and sets**, after stage D's MC4: `experiments/multi_training_export.py` through the shared
writer and `flown_sentence`; a smoke set; the browser check of §2 item 6 in stage D (the strip selects, every
aircraft's tabs choose its sentence, the losses show by pair, a word flies live); the formal sets after MC6, their
intent first (§2 item 4).

**Close.** The report gives the code index of the parts (§6.2) and the formats (§7); Claude moves each finished
milestone's specification to fronter's log.
