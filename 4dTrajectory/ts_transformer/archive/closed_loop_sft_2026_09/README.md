# closed_loop_sft_2026_09 — closed-loop supervised fine-tuning of the prior (CAT-K by segments)

The prior's first post-training step (prior design §9.2 as it was): each train flight flown in 4 branches, the prior
speaking in every branch, the branch closest to the observed flight kept every 10 steps and copied into the others
(`take` on the executor, its laws and context, `Spoken`, the `Speaker` and the model's cached keys and values); the
labelled words re-read against the words the kept chain had in force (`prior/relabel.py`: aligned to the label's own
words within ±5 rows, a clearance asked for until given, words the executor would ignore withheld); one pass a round
over every chain so far (`train.FineTuner`), 8 rounds, the round kept on the select days.

**Archived 2026-09-25, on the user's word** ("把CAT-K的代码移进archive"), after the formal run failed
(`4dTrajectory/outputs/POOLED/prior/v3_sft_20260925/`, readouts §6): val landed 90.2 → 93.1 %, all of it on straight-in
approaches (94.0 → 99.7 %) while the vectored ones got worse (83.3 → 81.0 %, crossing off the runway doubled), with half
the labelled heading words and the teacher-forced NLL 0.284 → 0.452 (1.114 after 8 rounds). The chains drifted from the
observed flights (1 km at 220 s), where the time-aligned labels no longer said what to say. The design keeps only why it
is not used (§9.2, §11); the post-training step is the landing reward (§9.3, `experiments/prior_landing_reward.py`).

Off the import path (`tests/test_architecture.py`); nothing here runs. To re-run it, check out `002998fb` (branch
`dev-prior-fast`) or `e760ab1d` (branch `dev-prior-v3`, the code of the formal run).

## What moved (`git mv`, byte for byte)

| here | was |
|---|---|
| `experiments/prior_closed_loop.py` | the runner (`run_ts.py prior_closed_loop`, runners R18) |
| `prior/relabel.py` | the chain's targets |
| `tests/test_prior_closed_loop.py` | its tests, unmodified; three of them test code that stays live and were copied into `tests/test_prior_speaker.py` (below) |

## Cut verbatim out of live files

- `cut_passages.md` — the code: `take` and the per-flight field lists (`PER_FLIGHT`, `CONTEXT`, `HISTORIES`) of
  `autopilot/executor.py`, `lateral.py` (`Runways`, `Lateral`), `vertical.py`, `speed.py`, `flights.py` (`FlightInputs`),
  `frame.py` (`AirportCharts`), `sentence.py` (`Spoken`), `prior/generate.py` (`Speaker`), `prior/model.py` (`Past`),
  `experiments/prior_free_generation.py` (`ClosedLoop.take` and its per-step record of the executor's clearance and
  capture); `train.FineTuneConfig` / `FineTuner`; `data.in_force_words`; the docstring sentences that named them.
- `docs/reference/entries.md` — runners R18 (a one-line pointer is left in `docs/reference/runners.md`).
- `docs/cut_sections.md` — the ts `CLAUDE.md` index sentence and the runners passages that named the runner.

## What stayed live

- `autopilot/` is back to its content before the step (`4205e186`) byte for byte, so its source hash is the one executor
  spec `executor/v7_20260925` recorded: v7 is this code's spec again; `executor/v8_20260925` (the same content sha,
  written for the code with `take`) no longer opens with it.
- `Prior.extend` / `Past` (row-by-row encoding), `data.chain_record`, `Flight.asked` and the loss over the asked columns,
  `prior_free_generation.ClosedLoop` (without `take`) and `steps_said` — the free generation and the landing reward use
  them. `select_readout` and `TIE_SHARE` moved into `experiments/prior_landing_reward.py`, their only live user, unchanged.
- `tests/test_prior_speaker.py` — the three live tests of the archived test file, copied verbatim (the speaker's rows are
  a chain's training rows; the loss counts only the asked columns — its `FineTuner` half dropped and the test renamed from
  `test_one_pass_counts_only_the_asked_columns` to `test_the_loss_counts_only_the_asked_columns`; rows encoded one at a
  time are the rows encoded together — its `Past.take` lines dropped), and the fixtures the landing reward's tests use.
- The asked-column mask (`Flight.asked`, `column_nll(..., asked)`) stays, though nothing live now asks for fewer than
  every column (the relabelling was its only partial user) — whether to remove it is the user's call.
