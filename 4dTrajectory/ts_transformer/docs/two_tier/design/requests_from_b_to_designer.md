# Requests from stage B to the designer

What Claude, as stage B's implementer, asks of the designer: the readings that wait for the user's decision, what stage
A must change for stage B, and the design text that stage B's work now needs. It holds only the current state: it is
rewritten in full each time, never appended to, and an item leaves when it is done. The evidence and the commits are in
the implementation log (`readouts/2026-10-05_stage_b_implementation_log.md` §1, §4).

State of 2026-10-06. The Training view's second part (outline §6.2, D133–D136) and the speed of the closed loop
(outline D138: A44 `3b535db6`, C13 `65a21c57`, B14 `65adf480`) are built and reviewed on `dev-two-tier-v4`; the items
of Claude's check (`readouts/2026-10-06_training_view_and_speed_check.zh.md`): `8e911092` (S2 2, 3) and `8cac5191`
(O15, D139). The nine readings and the three requests for the design text of the last version are in the design (D139,
prior §7, vocabulary §6). Not merged: the user merges.

## 1 Readings where the design says nothing (proposals, as built)

None.

## 2 For stage A

None.

## 3 For the design text

None.

## 4 The plan

- After C10 ends, in this order: B14's real-data check (before any run of `BATCH` on real data); C13's check on the
  GPU; the formal speed readout of B5's base; `sha256sum -c` of the base, the re-export of stage A's and B's published
  sets in the new formats, `sha256sum -c` again; the browser check; the user merges.
