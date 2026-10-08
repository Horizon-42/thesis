# Requests from stage C to the designer

What Claude, as stage C's implementer, asks of the designer: the design text that the user's decisions still need, the
names a public interface must give, and the questions that only the user can decide. It holds only the current state:
it is rewritten in full each time, never appended to, and an item leaves when it is done. The evidence and the commits
are in the implementation log (`readouts/2026-10-05_stage_c_implementation_log.md`, cited by §).

**State: 2026-10-08, evening (local).**
- C24 (D175) is merged (`bf2a4f04`) and read: the experiment log, "诊断读数"; the look at 17 "follower" losses beside it.
- Claude's readings of D175 where its text left a choice are in the log §30 (the C24 entry), for the designer to confirm.

**Resolved:** none since the last version.

## 1 The loss windows as a Training set — an exporter option `--windows` (the user, 2026-10-08)

The user's words: index the select windows that lost separation, fly them with several models and publish the result to
the frontend so the models can be compared on them; a sample of its own for stage C; **the windows stay in the select
split, nothing is taken out**; as simple as possible, little new code.

**Done without code** (data only, read-only): the index `outputs/POOLED/post/loss_windows_select_20261008/index.json`
(`SHA256SUMS`): the 193 select windows that lost separation with traffic on in at least one of three readouts — C10's
start (`ceiling_20261007` draw 0: 149), C10 round 8 (`diagnose_post_train_20261006_r8_20261008`: 117), P55 round 5
(`diagnose_post_seg60_20261007_r5_20261008`: 99); 62 lost by all three; KMSY 22, KRDU 52, KSJC 40, KSMF 38, KSTL 41.
Each entry: the window's place in `selection_windows` (select seed 1337), its flight and start time (`row0_s`), the
models that lost it, and D175's fields where read.

**Asked of the design** (C11's runner `post_training_export`, R-entry of its manual): today it draws its own windows
(`--per-airport`, `--seed`, `--kinds`) and flies each with `readout_numbers` of its place in its own draw. Asked: an
option `--windows <index>` (default: none, the draw as today) that takes the windows of such an index instead — matched
to the split's selection windows by flight and start time, refused by name when one is missing or the index's split is
not `--split`; refused together with `--per-airport`, `--seed` or `--kinds` — and flies each with
`readout_numbers(select_seed, place)` of its place in the selection readout, so that each model says exactly the
sentence its readout judged (the set then shows the readout's own losses). The set records the index's path and
sha256. My estimate: about 40 lines and three tests.

**Then (on the note):** two sets, each from one campaign as the exporter takes them, from a run worktree after the
user's merge:
- `post_train_20261006` rounds `start` and `8` (C10's start, the base with zero-output traffic, and C10 round 8);
- `post_seg60_20261007` rounds `start` and `5` (its start is C10 round 8, D162; and P55 round 5, stage C's model).
Set names to fix (mine: `post_loss_c10_v1`, `post_loss_p55_v1`); the intent entry in `docs/experiments/intents.json`
before publication if the publisher asks for one.
