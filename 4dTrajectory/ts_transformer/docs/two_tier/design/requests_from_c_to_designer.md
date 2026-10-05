# Requests from stage C to the designer

What Claude, as stage C's implementer, asks of the designer: the design text that the user's decisions still need, the
names a public interface must give, and the questions that only the user can decide. It holds only the current state:
it is rewritten in full each time, never appended to, and an item leaves when it is done. The evidence and the commits
are in the implementation log (`readouts/2026-10-05_stage_c_implementation_log.md`, cited by §).

State of 2026-10-05. Branch `dev-two-tier-v4-post` at `35289115`.

## 1 Design text for decisions the user made (post_training.md)

1. **A quiet window's landing direction** (log §12; the user, 2026-10-05). With no landing in the 30 min before the
   first predicted step, every runway counts as the present landing direction. Code: `post/reward.py`
   `present_runways`, `a1fdd2a8`. Text needed in §2 item 2 and §5. On A34's artefact, 2.8 % of the train windows had
   no landing.
2. **Windows that open inside a loss of separation are left out of the draw** (log §12, §13; the user). The test: the
   commanded aircraft on its record at its first predicted step, with no runway in force, loses separation it answers
   for. The rule applies to real and augmented windows alike. Code: `post/traffic.py` `opens_inside_loss`, `2bb77065`.
   Text needed in §2 item 4 and C1. Train windows left out: 58 of 40,530 real, 2,082 of 40,530 A, 1,304 of 16,234 D.

## 2 Names for prior §7's "Code" column (B10 still reads "new, B10")

Stage C's architecture test already admits these names, taken from B10's code (`e4e7ba42`):

| Item | Name |
|---|---|
| 1 | `prior/checkpoint.py` `open_prior`, `OpenedPrior` |
| 2 | `prior/inputs.py` `motion`; `prior/loop.py` `LoopRows` takes one `LandingIndex` per aircraft |
| 3 | `prior/speaker.py` `Permitted.join` |
| 4 | the rows of a sentence a loop said: `SpeakingLoop.sentences` (in item 7's module) |
| 7 | `experiments/prior_speaking_loop.py` `SpeakingLoop`, `Generated`, `flight_numbers` |

## 3 Questions for the user

1. **D103's shift ranges.** Under the rule of §1 item 2, the augmented windows lose 5.1 % (A) and 8.0 % (D). The
   shifts (A within ±180 s, D within ±120 s) often put the moved aircraft inside 3 NM and 1,000 ft at the first
   predicted step (log §13). Keep the ranges, or narrow them?
2. **Faulty tracks in the scene** (vocabulary D111; log §14).
   - On train, 1,062 of 40,530 windows (2.6 %) have a marked recorded aircraft in the air.
   - 796 steps of 3.8 million read a faulty point, through 914 tokens.
   - Only 2 of 1,374 losses on the records have a faulty point at or 2 Δ before the event.
   - Is a rule needed for these steps or windows?
3. **The weight of a sample in the loss** (P13, log §9). Each sample is divided by its counted rows, so a late branch's
   continuation weighs as much as a whole first sentence. Keep that, or weigh every row the same?
4. **The traffic module's memory** (P16, log §9). The reviewer estimates about 9 GB at a plausible formal batch; C8
   measures it. One token network shared by the layers would cut it. That is a design choice.
5. **The readings still open** (log §4, §9, §10, §11, §14):
   - P1, P2, P4–P6, P8, P11, P12, P14, P15, P18, P21–P25.
   - Each is built as written in the log. None changes a result the user has seen.

## 4 The plan: C6 and C9 can start

A38 is on this branch with stage A's line (`fdfa81d3`): `Loop.copy`, `Loop.halt` and `start_moved`. B9 gives
`Speaker.copy`. C6 (branch training, D94) and C9 (window B, the moved start) now have what §0.4 waits for. An order
for them is needed. C8 and C10 still wait for B5's base.
