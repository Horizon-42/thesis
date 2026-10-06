# Requests from stage B to the designer

What Claude, as stage B's implementer, asks of the designer: the readings that wait for the user's decision, what stage
A must change for stage B, and the design text that stage B's work now needs. It holds only the current state: it is
rewritten in full each time, never appended to, and an item leaves when it is done. The evidence and the commits are in
the implementation log (`readouts/2026-10-05_stage_b_implementation_log.md` §1).

State of 2026-10-06. Stage B is closed (B7). The Training view's one layout (outline §6.2, D133) is built on
`dev-training-layout` (`38297528`, `e0aceffd`; from `docs-training-view`, which holds D133), reviewed, the full suite
passed, checked in the browser; the user merges it.

## 1 Readings for the user to decide

Claude's readings where outline §6.2 says nothing (proposals, as built):

1. **The tab chosen outlives the item**: a tab chosen stays chosen on the next flight, window or set of the same stage when
   it has that tab (stage B: a sample; stage C: a round; stage A: a Δ, as before); otherwise the stage's first choice —
   B the first sample, C the last round, A the set's first Δ.
2. **The intent's answer**: none 404, several 409, no run asked 400; the frontend keeps a found intent for the page's
   life (an edited line shows after a reload; a missing one is asked again on the next view).
3. **The details page with no session on screen** (no set, an index that cannot be read) opens with its first section
   saying why there is nothing.
4. **"At the cursor" on a tab that is not a sample** (Labelled, Closed loop) says to choose a sample's tab; its line still
   opens the details page, on its first section (the Row inspector has nothing to show there).
5. **The short tab labels** (the user, 2026-10-06): stage B's samples by their number (0, 1 …), stage C's rounds r1, r2 …
   (Start (base) unchanged) — §6.2 item 2's table says "Sample 0" and "Round 1": the design text should follow.
6. `ExperimentIntentBlock` was extracted from `ExperimentDetails.tsx` (the experiments picker, outside §6.2 item 7's list
   of files), its output unchanged, so that the Training view shows an intent in the picker's one form.

## 2 For stage A

None.

## 3 For the design text

1. §6.2 item 2's tab labels: as §1 item 5.

## 4 The plan

- The user merges `dev-training-layout` (and with it `docs-training-view`); the services restart (the backend does not
  hot-reload: it serves `GET /experiments/intent` only after a restart).
