# Requests from stage B to the designer

What Claude, as stage B's implementer, asks of the designer: the readings that wait for the user's decision, what stage
A must change for stage B, and the design text that stage B's work now needs. It holds only the current state: it is
rewritten in full each time, never appended to, and an item leaves when it is done. The evidence and the commits are in
the implementation log (`readouts/2026-10-05_stage_b_implementation_log.md` §1).

State of 2026-10-06. Stage B is closed (B7): B5 done, B13 merged, B6 published and checked live; `dev-two-tier` =
`dev-two-tier-v4` = `dev-two-tier-v4-prior` (the code at `d14a2f76`, full suite passed); `docs/reference/runners.md`
R56–R62. Open: the Training view's design (§3 item 3).

## 1 Readings for the user to decide

None open. Decided by the user, 2026-10-06:

1. The claim's options also hold the device (both readers) and free generation's temperature and bound of go-arounds —
   kept, as built in B13 (`142272f0`).
2. An error that free generation can find only after its claim (an airport with no val sentence) locks that val read
   for good — accepted.
3. The Training view of stages A, B and C: one layout, the stage switch kept (§3 item 3).
4. A set's experiment description (its intent) comes from `docs/experiments/intents.json`, its one source, read through
   the backend by the frontend — no export writes it into a set, no index format changes, nothing is exported again.
5. Stage C's branch is merged by the user before the Training view is changed. Who implements the view (it touches
   stage A's files) is the user's decision once the design is in the doc.

## 2 For stage A

None yet: the Training view's change (§3 item 3) touches stage A's files (`TrainingSentenceBar.tsx`,
`TrainingPanel.tsx`, `training/TrainingFlightSession.tsx`); the user decides who makes it.

## 3 For the design text

1. **§7, the names and signatures B13 changed:** `prior.checkpoint.lock_val_read(prior_dir, reader)` (new: the read's
   exclusive lock, taken before a reader looks at its output, kept to its end); `claim_validation_read(prior_dir,
   reader, out, options)` (now takes the reader's options, no longer locks or returns anything);
   `settle_written_claim`, `holds_written_claim` (new); `holds_claim` (deleted); `prior_campaign.settle_val_steps`,
   `VAL_READERS` (new); the sample format `aeroviz-training-prior-sample-v3` (D127); the backend's
   `apart_from_exported(result, said, batch, j, step_s)` (``said``: the prior sentence, no longer its track).
2. B13's "Not ordered" item (tying the observed rows to the `Loop` the start returns) is left as the design says.
3. **The Training view of stages A, B and C — please add this design to the doc** (the outline's §6, the Training view,
   which every stage's view follows; decided by the user, 2026-10-06, items 3–5 of §1). Why: B's view (`aac93945`) and
   C's (on `dev-two-tier-v4-post`) were built on one template that put the choice of sentence in a table of the left panel
   and has no details page (B's ⓘ is disabled, C has none); the earlier Training view had the sentences along the top
   edge of the bottom panel (`38873db5`, AV30/AV31), a details page opened from one-line readouts (`53fa6ca1`, ⓘ
   `40cde69f`), and an experiment's intent (`311ff59d`, the Experiments picker); no stage's Training view shows an
   intent now. Stage A's view (`training/TrainingFlightSession.tsx`) keeps the details page and the bar's tabs.

   1. **One layout, the stage switch kept.** Stage A, B and C stay separate views with one layout. The left panel, top
      to bottom: the set chooser (a choice of set; one line: the set's title and the first sentence of its intent; a
      SMOKE tag for a smoke set); the list of items (flights or windows) with the same columns in every stage (callsign,
      runway, stratum or the window's kind, a tally of outcomes); the item's readouts, one line each, each opening the
      details page on its section; the Draw switches. No table in the left panel: B's sentence table and row-inspector
      table and C's round table and window-end block become one-line readouts and sections of the details page.
   2. **The sentences along the top edge of the bottom panel.** The tabs at the head of the sentence bar
      (`training-source-tabs`) choose the sentence in every stage: A — Labelled | one per Δ (as now); B — Closed loop |
      Sample 0 | Sample 1 …; C — one per round. Each tab carries a small dot in its outcome's colour
      (`trainingOutcomeColour`), its tooltip the outcome, the time and the go-arounds; many tabs wrap or scroll; the
      keyboard moves between them.
   3. **The details page in every stage, its ⓘ never disabled.** Sections: "What this view shows"; "The set and the
      experiment" (the full intent, the design it follows, what the set was made from, its val claim); then the stage's
      own: B — the outcomes by sample, airport and stratum, the row inspector (the probability of "go-around", what the
      procedure masks blocked); C — each round's end (the reward, the loss of separation) and the window.
   4. **The experiment's description from both panels.** The ⓘ of the left panel's header and the ⓘ of the sentence
      bar open the details page on "The set and the experiment". The intent is read from `docs/experiments/intents.json`
      (served read-only by the backend; looked up by the set's id, a `runs` key there); a set without an entry says so by
      name.
   5. **Shared parts, not copies.** One set chooser, one item list, the bar's tabs, the one-line readouts that open the
      details page, the Draw switches, and one hook that loads a set (B's and C's sessions copy it now); `ProblemBox`
      and `trainingText` stay the one definition of problems and outcome words.
   6. **Tests and checks.** Vitest for each shared part; each stage's tests kept; a browser check by a one-shot agent.

4. **§11, the key code index at `d14a2f76`** (B13's names; the rows not listed keep their lines): the checkpoint, the
   val read and its claim — `prior/checkpoint.py:67` `load_checkpoint`, `:89` `CLAIM_SPENT_BY`, `:99` `lock_val_read`,
   `:113` `claim_validation_read`, `:152` `settle_written_claim`, `:164` `spend_validation_claim`, `:174`
   `holds_written_claim`, `:180` `written_claim`, `:190` `readable_identity`, `:208` `open_prior`; the selection —
   `prior/selection.py:51` `left_out`, `:61` `kept`, `:70` `side`; the speaker — `prior/speaker.py:208`
   `Speaker.observe`, `:230` `Speaker.speak` (with `accept`); the step of the closed loop —
   `experiments/prior_speaking_loop.py:78` `SpeakingLoop` (the start's observed rows), `:148` `SpeakingLoop.step`;
   the runners — `experiments/prior_train.py:69`, `experiments/prior_free_generation.py:164`,
   `experiments/prior_validation.py:115`, `experiments/prior_select.py:169` (`:119` `choose_configuration`),
   `experiments/prior_campaign.py:316` (`:199` `settings`, `:252` `settle_val_steps`, `:263` `run_campaign`),
   `experiments/prior_behaviour.py:339` (`:285` `behaviour`, `:185` `fixed_sentence`, `:220` `selection_rules`,
   `:230` `campaign_plan`, `:247` `free_generation_draw`, `:257` `select_rules`); the Training export —
   `experiments/prior_training_export.py:91` `procedure_block`, `:170` `fly_again`, `:228` `unrounded`, `:327`
   `main`, `prior/training_files.py:35` `SAMPLE_SCHEMA` (v3); the backend — `aeroviz_backend/autopilot_segment/prior.py:94`
   `PriorSegments`, `:55` `apart_from_exported`. The runners' manual: `docs/reference/runners.md` R56–R62.
5. **What stage C needs after B12 and B13** (its branch follows when it merges): `SpeakingLoop(model, loop, order,
   sentences, observed, flights, …)` — the start's observed rows (`start_moved`'s), `post_window_loop.py` still calls
   the old order; `LoopRows(…, start: int)`; `Speaker.speak(…, accept)` and its two refusals (a row marked first otherwise
   than "nothing in force", positions not one for each row); a val read: `lock_val_read` then
   `claim_validation_read(…, options)`, `spend_validation_claim` after the readout (D119, D128); the behaviour check's
   answer and `ts-prior-campaign-v3`; the Training sample `aeroviz-training-prior-sample-v3` (the track unrounded, D127).

## 4 The plan

- **The Training view** (§3 item 3): the designer adds it to the doc; the user merges stage C's branch and decides who
  implements it.
- Stage B's milestones are done (B0–B13); B7's report is §3 items 4 and 5 and the implementation log.
