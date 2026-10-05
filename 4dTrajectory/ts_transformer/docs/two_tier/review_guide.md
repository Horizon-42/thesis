# Two-tier model: how to review a stage

**Summary.** This guide tells how to review the code of a stage of the two-tier model against its design. The main
question of a review is: can a model, the executor, a mask or a judge get information that it must not have, so that a
result looks good for a wrong reason? The second question is: does the code do what the design says? The guide gives
the principles, the procedure, a checklist for stages B and C, and the findings of the review of stage A as examples.

**Language.** This guide uses the writing rules of ASD-STE100 (Simplified Technical English): short sentences, active
voice, one meaning for each word. The words were not checked one by one against the STE dictionary. Units are SI only.

**Sources.** The principles come from Claude's review of stage A (2026-10-05) and the decisions that followed it
(vocabulary D77–D87, outline D85). Paths are relative to `4dTrajectory/ts_transformer/` unless they start with
`4dTrajectory/` or name the repository root. The design documents are in `docs/two_tier/design/`.

---

## 1 Terms

| Term | Meaning |
|---|---|
| Input | A value that a model, the executor or a mask reads to make its next output |
| Target | A value that a model learns to give (for example, a labelled word). A target can use the future |
| Readout | A value that a person reads after a run (for example, a landed share). A readout can use the future |
| Judgement | A value that the judge or the evaluation gives about a flown track. It can use the procedure and the observed track, as the design permits |
| Leak | Information that reaches an input but that the input must not have: the future of the flight, a sealed day, an identity of the airport, or procedure data where the design forbids it |
| Latent leak | A leak that no consumer reads today, but that nothing in the code stops |
| Channel | A path from a source of data to a consumer: a value, a field of an object, a set, a file |
| Train days, select days | The days that a training reads. The gradients come from the train days. The select days are read in every epoch: they stop the training and choose the epoch, the round or a setting (the "validation" of a one-tier run is this set) |
| Validation days (val) | Days that no training and no choice reads. They are read one time for each stage, in the stage's validation readout, to give the number that the stage reports (outline D85) |
| Sealed day | A test day (contract C32). No code opens, labels, counts or reads it, except the final test |
| Seam | Shared code that more than one stage uses (the data plane, `flight_scenarios/`, the start of a closed loop) |

---

## 2 Principles

1. **Trace each input to its source and its time.** For each value that an input reads, find the code that makes it,
   down to the raw data. Find the latest time of the data that it uses. An input uses only data at or before its step
   (outline principle 7). Do not stop at the module under review: follow the value through the seams.
2. **Keep inputs, targets, readouts and judgements apart.** A target, a readout or a judgement can use the future. An
   input cannot. Each field that uses later rows or the observed path has a name that says so, and the reader gives it
   apart from the inputs (vocabulary D82).
3. **A set that data decides is a channel.** Which candidates an airport has, which flights a sample holds, which days
   a split holds: each can carry information. Find which data decides each set (vocabulary D78).
4. **A value fitted from data comes from the train days only.** Check the split of every fit: grids, classes, limits,
   rules, hyperparameters, the stop of a training. The select days choose. The validation days are read one time for
   each stage. A readout that serves a choice of the user shows the train and select days only (outline D85). The test
   days are sealed. When a rule seals a split, search every runner for the split (its name, each `--split` option); do not
   trust the list of runners that an order names (stage A: `instruction_figures` and `executor_turns` still read val
   after A37 had changed the runners of its list). A check of a code change runs on the train and select days.
5. **Check what a component holds, not only what it reads.** An object that holds a forbidden value is a channel, also
   when no law reads it today (vocabulary D81: the landed runway in the executor's frame).
6. **Check by behaviour, not by names.** A test that scans names misses a value read through `getattr`, a dictionary or
   a parameter. Change the forbidden input and require the same output (vocabulary D73, D81).
7. **A boundary refuses before it changes state.** A function of a public interface checks its input (the grammar, the
   value ranges) before it changes anything, and refuses by name (vocabulary D80).
8. **One end, one definition.** When one component ends a flight, an episode or a loop, every component ends it at the
   same event, with the same test (vocabulary D79).
9. **The code does what the design says, and nothing more.** A law, a limit or an exception that the design does not
   have is a finding, also when it looks harmless (vocabulary D84: an unlimited roll rate in the first cycle).
10. **Measure each leak.** Measure how much the leak changes on real data, and say what it changes. Report the size
    plainly; do not make it larger or smaller. A small leak is still corrected before a formal run when the correction
    is cheap.
11. **Verify before you report.** Read the source line of each finding. Run a small check on data where you can. A
    finding that two independent reviewers report is stronger, but still read the line.
12. **Report; do not decide.** The design sets no criteria (outline D7). Give the facts, the size, the cost of the
    correction and a proposal. The user decides. A reading where the design says nothing is a proposal until the user
    decides.

---

## 3 Procedure

1. **Read the design.** Read the outline (principles, shared decisions, rules), the document of the stage (its
   decisions, its public interface, its plan) and the public interfaces that the stage reads.
2. **List the consumers and what each may read.** For each consumer (the model's inputs, the executor, each mask, the
   judge, the readouts, the export, the backend), write the list of the values that the design permits. This list is
   the reference of the review.
3. **List the channels.** For each consumer, list the paths by which data reaches it: values, objects that it holds,
   sets, files, seams. Mark each channel that crosses a time (the step), a split (a day) or an identity (an airport).
4. **Cut the code into areas.** Give each area to a separate reviewer (the agent `opus-code-reviewer`). Give each
   reviewer:
   - the design sections of the area and the list of step 2;
   - the questions: leaks (each channel of step 3), forbidden laws or rules, bugs;
   - the rules: read only; no full test suite, no formal runner and no GPU job while other jobs run; small checks in a
     scratch directory;
   - the report format of §6.
5. **Verify.** Read the source line of each important finding. Reproduce it on data where you can (a count, a value).
   Check that a finding is not the design's own decision.
6. **Classify.** Put each finding into one class: leak, latent leak, split (a day or a sample), boundary, end of a
   flight, design mismatch, bug, test gap.
7. **Report to the user.** Give the findings, the most severe first. For each: what, where (`path:line`), the size on
   data, what a correction changes (a format, a rebuild of an artefact, a retraining) and its cost. Then the list of
   the channels that are clean. Ask the decisions that the user must make as questions.
8. **Write the decisions into the design.** Each decision gets a number (outline §3.2) and its text in the design
   sections. Each correction becomes a milestone of the stage's plan, and the check of the stage gets an item for it.
9. **Give the orders.** The stage's agent does the work. The orders name the milestones, the stop points and the rules;
   the details are in the design.
10. **After each milestone of the corrections.** Read the agent's log (§0.3). Read the key code of the correction. Write
    the user's choices into the design text. List the agent's proposals for the user.

---

## 4 Checklist for stage B (the prior)

Use the design `prior.md` (decisions, §2–§8) and the vocabulary's public interface (§6).

**Inputs of a row (prior §2).**

- No input of a row up to the first predicted step is computed from R (D23). Test: change the runway word of a
  sentence; the inputs of rows 0 to `N_LOOK` stay the same, bit for bit. Check the candidate vectors, the glidepath
  height, the landings and every derived value.
- The motion inputs come from the positions in the 2 s before the row (D25), never from the stored track, ground speed
  or vertical rate. Row 0 has no motion (D60). Test: change the stored velocity columns; no input changes.
- The reader of item 3 reads only the rows. Search the prior's code for the fields that the reader gives apart
  (`withheld`: the runway, the landing time, the capture row, the go-around rows, the stratum, the outcome, the errors).
  None of them reaches an input (vocabulary D82).
- The landings in the 30 min before the step: never the flight's own landing; no landing of a sealed day; in a loop,
  only the landings that the loop knows (D63, post-training D31).
- No identity of the airport: no embedding, no absolute position or direction, no constant of a runway in the variant
  `full` (D5, D24). Each scale is a fixed SI constant, never a statistic of the data (D41).
- The time attention reads only time differences (D16). Test: a shift of all times changes no output.

**Training and choices (prior §5).**

- The stop and every choice read the select days. The training never reads the validation days (D31).
- A fold never reads its held-out airport in training. Check the data of each fold, the landings and the candidates
  included (D39).
- The selection of sentences (D75) uses the stored outcome only to keep or leave a sentence, never as an input.
- The seed scale and the rules of the choice are fixed before the runs (§5).
- KAUS is read one time, at the end of the whole chain.

**Speaking and free generation (prior §4; B4's specification: `readouts/2026-10-05_stage_b_implementation_log.md` §3).**

- The procedure masks read the procedure data and the state at the step only. They never block "unchanged" (D64).
- Free generation starts through the start of a closed loop (vocabulary D67, D77): the start state uses no sample after
  the first predicted step. The runner gives the words; the start refuses a row that the grammar refuses (D80).
- The time limit only ends a flight. No input reads it.
- The outcome comes from the judge. The end of a flight is the judge's end (vocabulary D79).
- The bound on go-arounds is a mask of the caller (D68).

**Identities and the Training view (prior §8, outline §6).**

- The data of a checkpoint are identified by their flights and the landings (D21, D63). No digest of code (D73).
- The export holds train and select flights, and val flights only from the base model's one validation readout.
- A live segment equals the export's flown states from the same state with the same words.

---

## 5 Checklist for stage C (the post-training and the multi-aircraft work)

Use the design `post_training.md` and the public interfaces of the vocabulary (§6) and the prior (§7).

**The scene.**

- At step t, an aircraft reads the other aircraft only at step t or before. A replayed aircraft flies its record: its
  later positions exist in the data. Check that no input, edge feature or mask reads them. A predicted value (for
  example, a closest-approach time from the present states) is permitted; a value from the record's future is not.
- D23 holds for every aircraft of a scene (post-training D31). Test: change one aircraft's runway word; all inputs and
  edge features of the scene up to its first predicted step stay the same, bit for bit.
- The landing context counts the landings of the loop. The recorded landing of a commanded aircraft is never in it
  (post-training §3).
- "Established on the final" is a function of one row (post-training D31, O6). It does not use the capture row or any
  later row.

**The reward and the training.**

- The reward comes only from the judge's outcome and the separation judge (D30). No term reads the observed track of
  the commanded aircraft.
- Branch training (D37): a continuation with the random numbers of the first sentence repeats it, bit for bit. A saved
  state holds the executor, the speaker's cache, the judge and the loop, and nothing from after the branch point.
- The data term uses the selection `landed` (D36, D76).
- Training windows come from the train days. The choice of a round reads the select days. The validation days are read
  one time.
- An augmented window (a moved start, an inserted aircraft, a moved leader) gives the commanded aircraft no value of
  the record that the augmentation changed.

**The separation judge and the masks.**

- One judge, its readings named (contract C36). The masks are given by the loop as masks of a caller; the prior does
  not import them.
- A mask uses the states at the step and the predicted states only.

---

## 6 Report format of a reviewer

- **Findings, the most severe first.** For each: the class (§3 step 6); `path:line`; the defect in one sentence; a
  concrete scenario (inputs, then what goes wrong); how it was verified (read only, or run, with the scratch path).
- **Checked and clean.** Each channel traced and found clean, with its evidence line.
- No points of style or names.
- Length: at most approximately 1,500 words.

---

## 7 Examples: the findings of the review of stage A (2026-10-05)

| Finding | Class | Principle | Decision |
|---|---|---|---|
| The executor's start state took a velocity fitted over 15 s centred on the row: up to approximately 8.5 s of the future. Measured: in the 8 % of the flights that turn at the first predicted step, the start track was 4.4° (p50) from the next 8 s of observed direction, against 11.5° for a fit over the 15 s before | Leak | 1 (the source was in the seam `flight_scenarios/start_state.py`) | Vocabulary D77 |
| KSTL 06 was a candidate because of one arrival on a test day | Split (a set decided by a sealed day) | 3, 4 | D78: the candidates are the published runway ends |
| The user chose Δ on a report that also showed the val rows | Split (a choice saw val) | 4 | Outline D85 |
| The executor held the landed runway's threshold and TCH as the origin of its frame | Latent leak | 5 | D81 |
| The test "the executor reads no vertical path" scanned names only | Test gap | 6 | D81: a behaviour check |
| The reader of the artefact gave the landed runway, the landing time and the errors against the observed path beside the rows; the observed rows held the centred fit | Latent leak | 2 | D82 |
| The start's loop took "go-around" while a go-around was in force, and a value outside its column | Boundary | 7 | D80 |
| The executor flew on after a crossing of another runway or the stall cut-off, which the judge takes as the end | End of a flight | 8 | D79 |
| The first cycle of a flight had no roll-rate limit | Design mismatch | 9 | D84 |
| The lateral error used the extended line of a segment, not the path | Design mismatch | 9 | D83 |
| Two runners (`instruction_figures`, `executor_turns`) still read the val days after D85 had been applied to the runners of its list; a correction's own check and report also showed val flights (check of A32–A40) | Split | 4 | Vocabulary A41 |
| The design said that an executor's states are the same bit for bit in batches of different size (from a smoke of 395 flights); a sample of 200 flights showed differences up to 7.9e-10 m (check of A32–A40) | A claim without a measurement at the size and the layout that it applies to | 11 | Vocabulary D97 (3); post-training D94 and §6.4: the states are within `STATE_BOUND_M` |

**What the review found clean (examples of channels to trace):** each law of the executor read only what vocabulary
§5.2 lists; the mass was the published landing mass of the type and the "unspecified" speed its published approach
speed, never an observed speed; the time limit only ended a flight; the loop never read the stored words or states; the
judge read no observed word; every fit of the vocabulary spec read the train days only; the backend's request carried no
state.

**What made the review work.** Four reviewers, one area each, in parallel, with the list of what each consumer may
read. Two of them found the start-state leak from two sides (the executor and the closed-loop reading). Each important
finding was checked again at its source line and on data before the report.
