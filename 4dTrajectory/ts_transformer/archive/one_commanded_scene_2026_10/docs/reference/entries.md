# Runner manual entries cut from `docs/reference/runners.md` (2026-10-02)

Cut verbatim when the one-aircraft scene loop was archived (`../../README.md`); they describe the archived code.

### R31 · `run_ts.py traffic_free_generation` — multi-aircraft M3: the post-training's start speaking to one aircraft of each scene (design §6.1, §6.6 step 4)

2026-09-28. `traffic_free_generation --prior <single-aircraft prior (augmented)> --executor <spec> --instructions <artefact>
--split select --out <new dir> [--per-airport 400] [--samples 4] [--aircraft-steps 300000] [--augment-seed N] [--device]`. The start of M4: the prior with a
traffic attention at zero (`prior.model.with_traffic`, C37 — it answers as the prior does, to rounding) speaking in the
closed loop of `experiments/traffic_speaking.py` ("one aircraft commanded": its scene's other flights replayed, the edge
features of the steps encoded and the two separation masks computed each step and handed to `prior.scene_speaker.
SceneSpeaker`, the speaking aircraft judged afterwards from its first predicted step by `traffic_loop.Loop`; an other in the
air more than 20 min before is read from there, counted). Each drawn flight read four ways, all judged the same way in its
scene (VISUAL ends it, IFR beside): **scene** (the model, `--samples` times), **alone** (the same model with no other in its
scene — the prior's single-aircraft free generation — judged against the scene: how much separation comes for free),
**labelled** (its labelled words flown), **recorded** (its record). Per source, pooled and per airport: outcomes with
`lost_separation`, loss episodes per flight and per hour flown, relations; for the model's sources, the share of steps the
separation masks took a word away (recorded by the loop) and the probability on what the masks removed. The procedure's
masks are the prior's own. A flight already in a loss it answers for at its first predicted step is counted and left out.
Batches are bounded by `--aircraft-steps` (scenes × their most aircraft × (longest pre-roll + rows)); the pre-roll is
encoded 64 steps at a time (`scene_speaker.BLOCK_STEPS`); each batch's rows are appended to `flights.jsonl` as it ends, and
each batch's loop is closed (`SceneLoop.close`: the loop and its speaker hold each other, and the first formal run ran out of
memory at 476 / 2,000 flights before it — kept as `free_generation_20260928.aborted-20260928T1330Z`). Writes
`free_generation.json` (`ts-traffic-free-generation-v1`).

**`--augment-seed`** (§6.6 step 5, `experiments/traffic_augment.py`): every flight's scene augmented one of three ways, a
third each — **D** the leader (by landing order, same runway or a pair separated as one) moved whole by U[−60, 60] s, **B**
the speaking flight's start moved as stage 2 moves it (the only kind with stage 2's × 2.0 time limit), **A** a flight with a
sentence inserted `g` × the required gap before the speaking one on the approach clock (`g` ~ U[0.5, 2]; same runway, single
pair, dependent parallel only) — drawn until no loss it answers for through its observed rows (judged not established),
at most 10 draws, else left out and counted; the speaking flight's landing context is the scene's. Read as scene and alone
only (a moved start has no record).

**Formal** (2026-09-28, `a0c31f3c`, `outputs/POOLED/traffic/free_generation_20260928/`, select, 400 per airport × 4,
79 min; readout `docs/two_tier/readouts/2026-09-28_m3_free_generation.zh.md`): lost separation under VISUAL — scene 11.5 %
(IFR 13.2 %), alone 12.3 %, labelled 4.9 %, recorded 2.6 %; landed 84.9 / 84.2 / 94.9 / 97.4 %. Vectored 21.6 % against
straight-in 4.0 %; a vectored landing is −111 s / +121 s (p10 / p90) off the record's time against the labelled words'
−36 s / +21 s. The masks acted on 2.7 % (approach) and 1.05 % (speed) of the steps. 8 scene / alone samples (2 flights)
started in a loss. The augmented run: `free_generation_aug_20260928/` (augment seed 7919, `de9a539a`; B 815, A 839,
D 346, none left out).

### R32 · `run_ts.py traffic_reward` — multi-aircraft M4: the traffic post-training (design §6.2, §6.6 step 6)

2026-09-28. `traffic_reward --prior <augmented> --base <base> --instructions <artefact> --executor <spec> --out <new dir>
[--rounds 8] [--resume] [--speakers 4] [--real-per-airport 200] [--augmented-per-airport 200] [--samples 8] [--select-per-airport 200]
[--select-samples 2] [--traffic-learning-rate 3e-4] [--aircraft-steps 100000] [--seed 1337] [--device] [--smoke]` plus
`RewardConfig`'s flags (stage 2's recipe). The start is `with_traffic(augmented)` (C37: augmented's answers to rounding).
Each round: a pool of 1.25 × 400 training-day flights per airport (the round's seed); the first 200 speak in their real
scenes, the rest in pool order in augmented ones (`traffic_augment`, D / B / A) until 200 qualify (refused when an airport
runs short); each scene spoken to 8 times in the scene loop (`traffic_free_generation.scene_sentences`: the start's
procedure's masks and the two separation masks), judged under VISUAL, each sentence kept to its judged end; reward 1 =
landed in the landing direction (the scene's landings) with no loss ending it first; advantages per scene; a scene
starting in a loss it answers for, or whose sentences agree, is not trained on. One pass (`traffic_tuner.SceneRewardTuner`):
stage 2's loss with each sentence scored in its scene by the speaker's own layout and edge code
(`scene_speaker.scene_inputs`, `traffic_speaking.speaking_edges` — the start's distribution is the sampling one, tested
with a traffic attention that reads the others and the others off the steps), base scoring the aircraft alone, the data
term on M2's scene samples of the training days (2 of M2's batches an update, ≈ stage 2's asked steps), the traffic
attention at 3e-4, the rest at 1e-5; a batch (stage 2's grouping, its own rows) is scored in parts of ≤ 16,384 padded
aircraft-steps with the gradients summed. The loop reads each step's edge features from two steps back (an other off the
step is carried forward at its row-before's motion; one step back read it "unknown" — the review of step 6). Select
readouts every round (round 0 = the start): 200 select flights per airport × 2 in their real scenes and one fixed
augmentation each (seed + 7919); the teacher-forced NLL on the select scene samples and the traffic attention's output
over the residual stream per layer; the select flights' record once. Guards and choice: design §6.6 step 6 item 8 (the
start is checked against the ordering guards before training and refused if it fails them). Writes `config.json`,
`round_00/readout.json`, `round_<k>/{sentences.npz, sentences.json, checkpoint.pt (v5, with its start), config.json,
procedure_masks.json, readout.json}`, `history.json`, `choice.json`. The training days are built once (≈ 4.4 GB host
memory, the loop's scenes and the data term's samples share each flight's rows). The pass scores in parts sized by the
GPU memory measured (`traffic_tuner.PAIR_COST`, `SCORE_BUDGET`: 64 KB an aircraft-step + 8 KB a pair-step with the
layers recomputed in the backward — `Prior.encode(checkpoint=True)`; kept, 145 KB + 31 KB, and the first GPU smoke ran
out of memory). GPU smoke (1 round, 20 + 20 scenes an airport × 8, select 20 × 2; `4d2c9c77`…`f3a8f4eb`): 24 min, host
peak 7.5 GB, GPU peak 4.0 GB; 1,600 training sentences in 13 min, the pass over 680 (18 updates) 72 s, a select
readout of 400 sentences ≈ 4 min. At the formal size (16,000 + 4,000 sentences a round) about 2.5 h a round, about 21 h
for round 0 and 8 rounds (the loop's batches are fuller at size: M3 spoke 8,000 scene sentences in about 45 min).

**Round by round** (design §6.6 step 6 item 11, user 2026-09-28): `--rounds k` is the last round an invocation runs
(0: round 0 alone) and `--resume` continues the run at `--out` from its last finished round. Every stream a round draws
from is its own — the pool (seed + round), the augmentations ([seed, round]), the sentences said and the pass's batches and
data ([seed, round, 1 / 2]) — and each round saves `optimiser.pt` (AdamW's moments, the warm-up's step) beside its v5
checkpoint, so rounds run one per invocation are the computation of one invocation (bit for bit on the CPU, tested on the
pass; on the GPU the same streams, the kernels' sums differing at rounding as between any two runs). A resume is refused
when `config.json` differs apart from `written_utc`, `rounds`, `resumed` (the git commit and the device are in it: run
from a checkout fixed at one commit — a commit made in it under a run refuses the resume, the first resume smoke showed
it), when a round did not finish (its directory without `readout.json`, written last — move it aside as
`round_<k>.aborted-<UTC>`; a run that did not finish round 0 is started again) or when the start failed the ordering
guards. `choice.json` is written over the rounds finished at the end of each invocation; `--resume --rounds <the last
finished>` writes it alone.

**In several processes** (design §6.6 step 6 items 12–13, 2026-09-29): the loop's Python ran on one core while the GPU
waited (20–30 % busy). The sentences — a round's and the select readouts' — are spoken by `--speakers` processes
(`Speakers`), forked once the data are built (the collector frozen first: the data stay shared) and before the parent
starts the GPU; each rebuilds a training round from its number with the parent's own `train_round` (a scene holds its
airport's whole data — rebuilt, not sent; the parent checks each process's round fingerprint) and speaks the loop batches
its index deals it, each batch from its own stream (`batch_seed`), so the sentences do not depend on `--speakers` (tested:
one and two processes = one process, bit for bit, on one thread; `--speakers` is not part of the run). The parent frees
its GPU cache before every speaking, builds its round while the processes speak, and runs the pass, the distances and
the teacher-forced readout; a speaking process that fails or is gone ends the run with what it said; the processes die
with the parent. A speaking process's batch is 100,000 aircraft-steps (four within 8 GB); each process's GPU peak is
logged beside the pass's. The loop's edge features are computed for all pairs at once and one airport's scenes in one
call (`inference.scene_edges.scene_edge_blocks`), the separation masks share the others' state per scene and step
(`traffic_speaking.others_at`) — identical to e76ca5ff on every select sample and on 20 real scenes spoken end to end.
The edge code's hash moved with it (`038df9ab…` → new): the run begun at e76ca5ff (`m4_traffic_20260928`, paused after
round 2 for the restart) was deleted on 2026-09-29.

**Several passes and the paired choice** (design §6.6 step 6 item 14, §6.2, §9 items 27–28, 2026-09-29): `--passes N`
sweeps a round's sentences N times (default 1, the first run's), each sweep a new order of batches from the round's pass
stream, every one scored against the model frozen once at the round's start — the one the sentences were said by, so the
clip bounds how far the sweeps go together (PPO's epochs); `passes` is part of the run, and the pass record carries each
sweep's KL and share of words outside the clip (`sweeps`). The round kept: within the guards, a round beating round 0 on
the augmented scenes by at least `TIE_STANDARD_ERRORS` = 2 paired standard errors (the select readouts speak the same
draws every round: √(sentences whose reward flipped) / sentences, over the sentences both count — `paired_difference`,
`select_rewards` from each round's `readout.json`), of those the highest and the earliest it does not beat by as much;
none: round 0. `choice.json` carries each round's difference and standard error against round 0 (`against_round_0`).
The first run (`m4_traffic_20260929`) was chosen by the old tie (0.015) and is not re-chosen.

### R33 · `run_ts.py traffic_reward_readout` — multi-aircraft M4's readout, round by round (design §7)

2026-09-29. `traffic_reward_readout --run <a finished, formal traffic_reward run> --reference <M3's real-scene free
generation> --out <new dir>`. Reads only files: the run's `config.json`, `history.json`, `choice.json` (the run must be
finished — every round it asked for, `config.rounds`, read and covered by both files; a smoke run is refused), each round's `readout.json` (the select flights) and
`sentences.npz` (the training sentences); M3's `flights.jsonl` (`ts-traffic-free-generation-v1`, real select scenes, the
run's executor sha256 and instruction artefact, named by its place under `4dTrajectory/outputs` so a path through a removed
worktree still names it — refused otherwise, as is a flight/source row twice) for each real select
flight's loss on its labelled words and along its record: every real select flight must have both, and M3's records over
them must reproduce the run's own recorded reading (`config.select.recorded`) exactly. Per round: each side's reward, lost
separation (VISUAL and IFR), landed and observed-runway shares (the run's numbers); lost separation by approach type
(real) and by kind (augmented), over sentences, a sentence starting in a loss it answers for left out as the run's own
separation readout does; sentences with a go-around; the traffic attention's output over the residual stream; from round 1
the round's sentence summary and pass (traces left out) and where the reward term's signal comes from — each scene's `K`
sentences in one of five classes per kind, each class's share of the summed |advantage|, and `differing_scenes` beside the
run's `scenes_with_contrast` (the difference: scenes starting in a loss, not trained on, unmarked in the npz).
**The select readout speaks the same random draws every round** (the select generator is seeded alike), so rounds are
paired, not repeated tries (on the formal run 88 % of real (flight, sample) outcomes are the same in rounds 0–7): the
readout gives that share, the paired change of lost separation from round 0 to the last (fixed, newly lost, net, standard
error √(fixed + newly lost) / n), and in round 0 and the last round the flights grouped by how many of their samples lost
separation, with each group's share of losses, of vectored approaches (real) and of the same flights' losses on their
labelled words and along their records. A flight with a sentence starting in a loss in any round (the first step's runway
word decides the runway it is judged against), or whose record or labelled reading starts in one, is left out of this part
and counted. The last round is not necessarily the kept one (`choice.round`). Writes `traffic_reward_readout.json`.
