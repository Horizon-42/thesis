# Requests from stage C to the designer

What Claude, as stage C's implementer, asks of the designer: the design text that the user's decisions still need, the
names a public interface must give, and the questions that only the user can decide. It holds only the current state:
it is rewritten in full each time, never appended to, and an item leaves when it is done. The evidence and the commits
are in the implementation log (`readouts/2026-10-05_stage_c_implementation_log.md`, cited by §).

**State: 2026-10-08, evening (local).**
- Stage C stopped at `post_seg60_20261007` round 5 (the user's word): select 87.7 % landed; val read once
  (`validation_post_seg60_r5_20261008`): 88.9 % landed, 9.1 % lost separation, mean reward 0.880. Nothing runs.
- The diagnostics proposal `readouts/2026-10-08_stage_c_next_steps_proposal.zh.md` (G1–G4, T1–T3) was written before
  the segment (D170) and value (D171) results. **The user's decision, 2026-10-08:** do G2 and then G1, cut as in item 1
  below, on P55 round 5 with C10 round 8 as the comparison; G3 deferred; G4 not done; T1–T3 not asked for.

**Resolved:** every item of the last version (none was open).

## 1 A diagnostic runner `post_diagnose` (G2 traffic ablation, G1 failure breakdown) — needs a D number and a note

Readouts only, no training. Select days only, never val (outline D85).

**Models read** (each with the selection readout's windows and numbers, seed 1337, `readout_numbers`):

| model | traffic on | traffic off |
|---|---|---|
| `post_seg60_20261007` round 5 (the stage's model) | yes, with `--per-window` (the self-check and G1's data) | yes (G2) |
| `post_train_20261006` (C10) round 8 | recorded in its `round.json`; read again only as the self-check | yes (G2) |
| C10's start (base + the zero traffic module, D116) | one read (off is the same model: its output layer is zero) | — |

**Options of the runner** (only in this runner; training and `selection_readout` do not get them):
- `--traffic-off` (default off): after the round is loaded, set the weight and the bias of every `TrafficAttention.out`
  (`post/traffic_attention.py`, the layers `traffic_modules` returns) to zero — the same as the start's initialisation
  (D116); the rest of the model unchanged.
- `--per-window` (default off): one line per window in `per_window.jsonl`: window index, airport, kind, outcome,
  reward, go-arounds, `loss_step`, `other`, `loss_reads_fault` (all on `WindowResult` today), and G1's fields below.

**The self-check:** a read with traffic on of a round whose `round.json` has a `selection_readout` must equal it, else
refused by name (as `post_ceiling`'s draw 0). To decide: what checks a traffic-off read. My proposal: the runner reads
a round with traffic on first (checked), then off, in one process; for C10 round 8 that costs one extra read
(~12 min). And what the start's read is checked against (C10's campaign records no start readout; `ceiling_20261007`
reads starts — its draw 0 if it read this one).

**G1's fields** (each lost-separation window, from its `WindowResult` and the window's scene; definitions are the
designer's to fix):
1. the other aircraft: the leader on the same runway (lands there before the commanded aircraft) / the follower on the
   same runway / an approach to another runway / an aircraft that does not land in the window;
2. the commanded aircraft's distance to its threshold at `loss_step`, bins 0–10, 10–20, 20–40, > 40 km;
3. the time from the first predicted step to `loss_step`, bins < 60, 60–120, 120–300, > 300 s;
4. already in conflict at the start: both aircraft extrapolated in a straight line at their speeds at the first
   predicted step; is the closest distance (to decide: up to which time — my proposal, `loss_step`'s) below the
   separation standard of the one judge (`inference/separation.py`, `VISUAL`);
5. for C10 round 8 only: the windows that lost separation in all 32 draws of `ceiling_20261007` (that readout is C10's,
   not P55's).

**Outputs** (a new dir per model under `4dTrajectory/outputs/POOLED/post/`, name to fix): `intent.json` (the decision
rules below, written before the read), `summary.json` (landed, lost separation, mean reward, by airport, on and off;
G2's paired differences with a 95 % interval from 2,000 window bootstrap resamples), `g1_failures.json` (counts and
shares of each class, by airport), `per_window.jsonl`, the run log, `SHA256SUMS`; sealed read-only.

**Decision rules (the proposal's values, on P55 round 5; the user may change them before the read):**
- G2: off below on by < 2 points with the interval holding 0 → the model hardly uses the traffic; stop RL-side work and
  write a proposal on what the model sees (D117 token features, the traffic module's size and rate). ≥ 5 points → the
  traffic is used. Between → report the numbers, the user decides. Also reported: C10 round 8 off against the start
  (how far the prior itself moved in post-training).
- G1: (in conflict at the start) as a share of the lost-separation windows ≥ 40 % → that part is out of training's
  reach; re-estimate the reachable ceiling and look at D113's window rule. One class of other aircraft or one distance
  bin ≥ 50 % → later work aims at it.

**Cost (my estimate):** five select reads, about one hour; the runner and its tests about 200–300 lines, built on
`post_train.selection_windows`, `read_batch`, `round_model`, `counted_ends` and `post_ceiling`'s check.

**Not in this request:** the proposal's G1 on a train draw (windows A/D/B apart; later if wanted), G3 (temperature:
needs the temperature passed through `WindowLoop`, and acting on it changes D125 / P39 — a user decision), G4 (PIPO is
not planned, and a round's select change sits inside the readout's ±1.2 point noise, so the sign test reads noise).
