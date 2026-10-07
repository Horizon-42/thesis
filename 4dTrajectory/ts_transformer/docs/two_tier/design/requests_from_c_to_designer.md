# Requests from stage C to the designer

What Claude, as stage C's implementer, asks of the designer: the design text that the user's decisions still need, the
names a public interface must give, and the questions that only the user can decide. It holds only the current state:
it is rewritten in full each time, never appended to, and an item leaves when it is done. The evidence and the commits
are in the implementation log (`readouts/2026-10-05_stage_c_implementation_log.md`, cited by §).

**State: 2026-10-07, 21:20 (local).**
- C21 (`a72829f3`) and C22 (`c1e39964`) are merged into `dev-two-tier` (the user's word each time).
- P55's first campaign (`post_seg60_20261007`) runs; the value method's first campaign (item 4) follows it.

**Resolved:** items 1–5 of the last version (D167's reading, `settings_type` in §9 item 11, stage D's merge of C18,
P55 as D170, P54 as D171).

| # | Request | For | Log |
|---|---|---|---|
| 1 | **C22's readings where D171 / §2 item 10 is silent (built so; the user can change any):** (a) an update takes `Settings.update_groups` samples (one sentence each; a piece a sample, as D165's kept sentences); (b) under `value`, `continuations` is required to be 1 (one sentence a window), refused by name otherwise; (c) V's "edge features" at +30/60/120 s are `edges.EDGE_FEATURES` (14), not the motion features, each with an in-air flag; the time to the landing reads the record's `landing_s`, 0 once landed; (d) the time left is the executor's limit in force (its go-arounds' extensions included) less the flight's cycles flown, over 900 s; (e) V's head: d + 1 → 64 → 1 (GELU), its weights drawn from a seed of the campaign's own (`value_seed`); (f) the warm-up's check for a base start compares rounds 1… with round 0 (round 0 has nothing before it), and for a start from a round compares only when the select seed and the select windows a airport are the source's (otherwise recorded as not compared) | The designer, the user | §30 |
| 2 | **The skeleton's two new parts (post-training §9 item 11).** `Stage.companion` (what a campaign keeps beside its model from round to round: stage C's value method keeps V; None for the others) and `Stage.close_round` (after the round's record, before its checkpoint: the value method writes `value.pt` with the round's identity there and adds the warm-up's check to `round.json`); `Stage.train` receives the round and the companion. Stage D takes the defaults (nothing kept, nothing added), its digest unchanged. §9 item 11 could name them | The designer, stage D | §30 |
| 3 | **`WindowLoop`'s `value_reader` (post-training §9 item 9).** An optional reader of what a value network reads at each row beside the model, kept apart from the tokens (`WindowLoop.values`); stage D does not pass one | The designer, stage D | §30 |
| 4 | **The value method's first campaign — decided by the user (2026-10-07):** from C10's round 8, seed 2029, select seed 1337; 4,000 windows a round; `update_groups` 32 samples; `epochs` 4; `value_lr` 1e-4; `value_warmup` 2; C10's rates for the model; `clip_norm` 1.0; 6 rounds. It starts when P55's campaign ends, from `dev-two-tier` `c1e39964` (C22 merged on the user's word) | — | experiment log |
