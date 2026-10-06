# Requests from stage C to the designer

What Claude, as stage C's implementer, asks of the designer: the design text that the user's decisions still need, the
names a public interface must give, and the questions that only the user can decide. It holds only the current state:
it is rewritten in full each time, never appended to, and an item leaves when it is done. The evidence and the commits
are in the implementation log (`readouts/2026-10-05_stage_c_implementation_log.md`, cited by §).

**State: 2026-10-06, late evening.**
- C14's code is `21547ad4` on `dev-two-tier-v4-post` and reviewed; the user merges it.
- Rounds 10–13 of `post_train_20261006` wait for that merge and for stage D's GPU and timing steps.

**Items of the earlier note that are now resolved:**
- the update in pieces: post-training §2 item 5;
- the extension after C10: D157, C14;
- stage B's item 5: it has left B's note.

| # | Request | For | Log |
|---|---|---|---|
| 1 | **P46, how C14 records a raise of the rounds.** The raise has no list of its own. Each resume's entry in `campaign.json` holds `rounds: {before, after}` and `inputs`, the paths that resume read, beside its time, commit and checks. The raises are the entries where `before` and `after` differ. The record's `settings.rounds` holds the current count, and its paths stay as recorded. The format's name stays `ts-post-train-v1`, because D157 keeps the same campaign. Only new resume entries carry the two keys; C10's four earlier entries do not. No reader, in Python or in the frontend, reads the entries of `resumed` | The user | §27 |
| 2 | **P47, a raise after the validation read.** The code does not refuse a raise of the rounds once the campaign's validation read is claimed (`val_read_post_validation.json`, D132). C10's validation read has not been made, so C14 does not reach this case. Proposal: `open_campaign` refuses a raise when the claim exists, because the round would then be chosen after val had been seen | The designer, the user | §27 |
| 3 | **S3 findings from C14's review, outside C14's files.** First: `post_training_export.py:309` and `model_speed.py:376` each write out their own list of the five path keys, so the same list is now in three places. Each could import `post_train.inputs_here` / `INPUT_PATHS` instead (stage D's implementer, a small change). Second: a resume entry's `inputs` is the path after `this_checkout`, not the path as typed. The two differ only when another checkout's absolute path is given, and both name the same data | The designer | §27 |
| 4 | **P45, a reading for the user.** A window set writes its flown tracks without rounding (prior D127 followed for windows), so a set's file is larger: 1.3 times on the smoke set. A formal set's size is read when it is exported | The user | §21 |
