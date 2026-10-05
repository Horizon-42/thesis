# Requests from stage C to the designer

What Claude, as stage C's implementer, asks of the designer: the design text that the user's decisions still need, the
names a public interface must give, and the questions that only the user can decide. It holds only the current state:
it is rewritten in full each time, never appended to, and an item leaves when it is done. The evidence and the commits
are in the implementation log (`readouts/2026-10-05_stage_c_implementation_log.md`, cited by §).

State of 2026-10-06. **No open request.** Every item of the note of 2026-10-05 is resolved:

| Item | Resolved by |
|---|---|
| Window B's ranges (P28) | Post-training D123 (`51f9a404`) |
| The readings of C6, C9, D114 and C10 (P26, P27, P29–P31, P33–P36, P38, P39) | Post-training D125 |
| A round's windows share their flights (P32); the loss branch point by branch point; D113 at the draw; one command of a flight a batch (P31) | Post-training D124 |
| The Training export's rows in vocabulary §6 and prior §7 | Vocabulary §6 item 8, prior §7 item 8, D126 |
| P37: a window set carries no per-row speaker records | Post-training D125 (the window view does not show them) |
| The window view (P42–P44, log §19) | Post-training D129, as proposed; ordered in `notes/stage_c.md` |

The plan: C11's last points (D129) now; then B13 when stage B commits it. The formal C8 (`post_profile`) and C10
(`post_train`) wait for B5's base and Claude's check of stage B, C10 also for the user's criteria; C12 after C10.
