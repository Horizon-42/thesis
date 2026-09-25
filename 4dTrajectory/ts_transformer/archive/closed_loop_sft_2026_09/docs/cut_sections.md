# Passages cut from mixed live documents (2026-09-25)

Verbatim as they stood at `002998fb` (branch `dev-prior-fast`), each under the file it came from.

## `4dTrajectory/ts_transformer/CLAUDE.md` — the runners paragraph, after R17

```
**Closed-loop fine-tuning** (`prior_closed_loop`, design §9.2): CAT-K by segments — each
flight flown in branches, the one closest to the observed flight kept every 10 steps (`take` on the executor, `Spoken`,
`Speaker`), targets = the labelled words aligned to the label's own words (`prior.relabel`), one pass a round over every
chain so far (`train.FineTuner`), the round chosen on select; the speaker encodes row by row (`Prior.extend`) (R18).
```

## `4dTrajectory/ts_transformer/docs/reference/runners.md`

```
Since `Prior.extend` (2026-09-25, R18) the speaker encodes row by row:
```

## `4dTrajectory/ts_transformer/docs/reference/runners.md`

```
× a teacher-forced batch of the train split per update (a round with no flight to train on is refused by name). The select readout is `prior_closed_loop`'s, in the same batches of 64 flights (`SELECT_CHUNK`: round 0 from the step-1 prior reproduces §9.2's round 0), plus the share landed against the landing direction. `choice.json`: among
```

## `4dTrajectory/ts_transformer/docs/reference/runners.md` (after the review)

```
`prior_landing_reward --prior <the §9.2 round kept, or the step-1 run> --instructions
```

## `4dTrajectory/ts_transformer/docs/reference/runners.md` (after the review)

```
new code dumped on the same seeds — free generation, CAT-K chains, the select split's inputs — 29/29 arrays identical;
```
