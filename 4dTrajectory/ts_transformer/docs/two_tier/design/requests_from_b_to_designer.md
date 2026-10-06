# Requests from stage B to the designer

What Claude, as stage B's implementer, asks of the designer: the readings that wait for the user's decision, what stage
A must change for stage B, and the design text that stage B's work now needs. It holds only the current state: it is
rewritten in full each time, never appended to, and an item leaves when it is done. The evidence and the commits are in
the implementation log (`readouts/2026-10-05_stage_b_implementation_log.md` §1, §4).

State of 2026-10-06. The Training view's one layout, its second part (outline §6.2, D133–D136), is built on
`dev-two-tier-v4` (`65314499`, `92a56314`, `550fb0c2`; reviewed; the formal speed readout and the re-export wait for C10
to end). The speed of the closed loop (outline D138) is built there too: A44 `3b535db6`, C13 `65a21c57`, B14 `65adf480`
(reviewed). Not merged: the user merges.

## 1 Readings where the design says nothing (proposals, as built): outline §6.2 (1–6), D138 (7–9)

1. **The batch of 400 of stage B's speed** (item 10): the drawn 100 flights (20 per airport) repeated in turn to 400,
   copy k of a flight its sample k; a batch smaller than the drawn flights speaks them all in loops of it.
2. **Stage C's sets name a speed readout too** (item 10, D136): C's export requires `--speed` (a stage-C `model_speed`
   readout) now, so that C's format v3 is settled before C exports; `source.speed` = the readout, the model it timed and
   whether it is a smoke (`model_speed.speed_source`). A fold set of stage B names the base's readout (only the base is
   timed) and says so by the model it names.
3. **The results route's refusals** (item 4): an airport that is no airport code 400; a set of another format 409 by name;
   a section's file of another format, elsewhere or missing: that section's problem by name.
4. **Stage A's closed-loop section** reads the replays of the set's own splits (train and select) at each Δ of the set.
5. **The earlier blocks** (item 8) are checked by the sha256 of each fixture's earlier block (A's rounded to its earlier
   digits), held in the test: the earlier fixtures were rewritten by this change.
6. **Stage C's Speed section** in the route and the page (item 3's table names Speed for B; D136 for both).
7. **A44: the arrival records** (vocabulary §12.1 says the `Start` holds them): a flight's first start still reads its
   airport's arrival and tracks manifests (`rebuild_series`: 1.7 s for 64 flights of five airports, about 1 % of a
   speaking batch); holding them would change the harvest's loader (`load_arrival_flights`), which training reads.
   `Start` keeps each flight's series instead, until `release` (C13 releases the train start's at each round).
8. **B14's check costs 0.6–1.7 s an airport, once a process** (D138 says milliseconds): its fixed rows are a grid about
   each final and a sweep down each course, so that a 0.2 m change of a limit is found (a smaller grid missed whole
   kinds of difference, the review).
9. **C13's O15 rule**: the speaking (N × one worker's measured peak) and the pass (this process's growth in one update
   on the measured batch's groups, beside the other workers' held memory) must fit; approximations stated (one batch's
   groups; the optimizer's state; the model at the start on a resume).

## 2 For stage A

None.

## 3 For the design text

1. Outline §6.2 item 3's table: stage C's sections with Speed (§1 item 6); item 7: C's `source.speed` (§1 item 2).
2. Prior §7: the procedure masks' mode names (`PER_AIRCRAFT`, `BATCH`, `MASK_MODES`) and the speaker's and the
   speaking loop's `masks` argument; stage C's window loop takes `BATCH` (the architecture test lists it).
3. Vocabulary §6 item 5: `Start.release` (C13 bounds the kept series by it); §1 item 7.

## 4 The plan

- After C10 ends: the formal speed readout of B5's base; the re-export of stage A's and B's published sets in the new
  formats (the base's `SHA256SUMS` checked before and after); B14's fold spoken again with batch on the GPU against
  its readout; C13's GPU run (one round, one process against N workers); the browser check; the user merges.
