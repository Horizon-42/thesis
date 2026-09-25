# Reference entries cut from `docs/reference/runners.md` (2026-09-25)

Verbatim as they stood at `002998fb`.

### R18 · `run_ts.py prior_closed_loop` — closed-loop supervised fine-tuning of the prior (prior design §9.2)

2026-09-25. `prior_closed_loop --prior <chosen run> --instructions <artefact> --executor <executor spec dir> --out <new dir>
[--rounds 8] [--per-airport 800] [--select-per-airport 200] [--select-samples 2] [--branches 4] [--segment-steps 10]
[--window 5] [--temperature 1] [--learning-rate 1e-4] [--warmup-steps 100] [--weight-decay 0.01] [--clip-norm 1]
[--tokens-per-batch 16384] [--seed 1337] [--chunk 64] [--device cuda] [--smoke]` — CAT-K by segments. Each round draws
`--per-airport` train-day flights (own dynamics, `replay.draw` with seed + round, so rounds may share flights: counted as
`repeated_flights`) and flies each in `--branches` branches of `prior_free_generation.ClosedLoop` (the prior speaking
exactly as in free generation); every `--segment-steps` steps the branch closest (3D metres) to the observed flight at that
row is kept and copied into the others (`Executor.take`, `Spoken.take`, `Speaker.take` — every per-flight field, the laws'
state, the records and the model's cached keys and values); a branch the executor finished other than by crossing the
threshold captured is passed over while another branch flies. The chain's targets are the labelled words aligned to the
label's own words (`prior.relabel`, `--window`; a clearance the label has in force is asked for until the chain gives it,
the handover's too); its rows are the rows the speaker read (`data.chain_record`). One pass
(`train.FineTuner`, AdamW, optimiser state carried across rounds) over every chain so far, then free generation on the
select days (the same flights and seed every round; round 0 = the prior before fine-tuning) and the teacher-forced NLL
there. `choice.json` keeps the round with the highest select landed share, the earliest within `TIE_SHARE` (0.015) of it.
Writes `config.json`, `round_00/readout.json`, `round_<k>/{chains.npz, chains.json, checkpoint.pt, config.json,
readout.json}` (a round directory is a prior run `prior_free_generation` reads — the val readout, once), `history.json`,
`choice.json`; from a clean tree unless `--smoke`. Val and the sealed test days are never read. Cost (2026-09-25): one
chunk of 64 flights × 4 branches ≈ 30 s (the speaker encodes row by row, `Prior.extend`; each layer's keys and values
allocated once for the chunk's longest flight and written in place), ≈ 31 min a 4,000-flight round.
The executor's source hash moves with `take`: write a new executor spec (same content sha) at the commit before a
formal run.
