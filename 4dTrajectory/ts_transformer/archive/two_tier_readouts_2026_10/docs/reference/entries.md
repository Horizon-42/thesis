# Runner entries cut from `docs/reference/runners.md` (2026-10-13)

Verbatim, as at `two-tier2-eval-landing` `7621d5b3`.

### R51 · `run_ts.py executor_turns` — the turn of the executor measured: how much of the offset after a turn is the heading law's, how much the words' (vocabulary §12.1 A13)

2026-10-04 (vocabulary §5.4, §4.3, §9.6, §12.1 A13; `experiments/executor_turns.py`). `executor_turns --instructions
<artefact> --executor <spec dir> --split train [--per-airport N] [--seed 1337] [--chunk 500] --out <new dir>` (`--split` train
or select — val refused by name, D85, A41; select from a clean tree): the replay's flights (`replay.draw`), their open-loop sentences at Δ = 2 s, flown from the first
row, said on their own rows, in three ways — `executor` (the executor at the observed ground speed: the airspeed set at each
cycle's start so that the ground speed is the observed one, and moved within the cycle to the observed one at its end in
place of the speed law), `executor_no_stopping` (the same, without the stopping-rate limit of §5.4) and `exact_words` (each
heading word's track reached 4 s after the word, linearly, at the observed ground speed, from the observed position at row
0), with a CONTROL `observed_track` (the observed track integrated the same way: what integration and the data's own
track-against-position agreement leave). The settings of the first two live in the runner (subclasses of `Executor`,
`Lateral`, `Speed`); the executor's laws do not change. For each observed turn (runs of the labeller's smoothed track
faster than 0.2°/s, `turn_runs`) and way: the change of e_y (the closed loop's matched point) from the turn's start row to
30 s after its end row (A13 as written; it holds what earlier turns left — a 90° turn makes an along-track lead lateral),
and the turn's OWN part (the change of the flown-minus-observed displacement at the same time over those rows, on the
right of the observed track at the later row; Claude's reading), each also toward the outside of the turn, and the way's
e_y at the turn's start, and whether the next turn starts before the read row (`overlaps_next`, counted per cell); not measured
where the later row lies past the observed flight or the way's flight. `turns.json`
(`ts-executor-turns-readout-v1`) holds every turn (with its group) and the table by stratum × ground-speed band (0–70,
70–85, 85–100, ≥ 100 m/s): per way, |change| and |own| p50 / p90 / mean and their outward parts; and over the turns
measured in every way, the same and the paired differences of the own outward part — executor ways minus exact words (the
heading law), exact words minus the control (the words). On the test fixture the control itself leaves 99 m per 90° turn:
the fixture's track leads its positions by 1 s. Information for the user's decision on the heading law (D7).

    python run_ts.py executor_turns --split train \
        --instructions 4dTrajectory/outputs/POOLED/instruction_language/<artefact> \
        --executor 4dTrajectory/outputs/POOLED/executor/<name> --out <new dir>

### R52 · `run_ts.py final_descent_tolerance` — the closed loop's vertical tolerance in the final descent, H_final, measured for the user's choice (vocabulary §12.1 A24, D66)

**2026-10-12** (the user: a landing is the evaluation's verdict, C33; the decision-altitude check stays, a record):
readout v3, table v2 — the outcomes are `landed` / `landing_failed`, and ``da_not_passed`` (every approach crossing
whose DA check did not pass, by why) replaces ``unstable_at_minimums``. The text below is from 2026-10-04.

2026-10-04 (vocabulary §4.9, D34, D38, D55, D66, §12.1 A24; `experiments/final_descent_tolerance.py`). `final_descent_tolerance
--instructions <artefact> --executor <spec dir> --split train|select [--per-airport N] [--seed 1337] --row-interval-s 2 4 8
[--chunk 2048] [--device cpu] --out <new dir>` (select from a clean tree; val is refused): one artefact built as the formal
one in everything but the spec's `closed_loop_final_vertical_m`, with its executor spec. The flights of the split (train:
`--per-airport` of each airport, D34's seeded sample; select: every labelled flight; own dynamics or a stand-in's), each read
in closed loop at each Δ IN MEMORY (`closed_loop.read_chunked`, nothing written into the artefact), its sentence flown again
as the replay flies it (every flight must fly its stored states again) and judged (`judge.outcome_of`). `readout.json`
(`ts-final-descent-tolerance-readout-v1`): per Δ the split's closed-loop numbers (`instruction_closed_loop.summarise`, D34
readings 1–3 with the tolerance in force at each row) and, per airport and pooled, the outcomes and landed share, the
`unstable_at_minimums` flights by why (no DA point / high / low / lateral only), the height above the glidepath at the DA
point p10 / p50 / p90, the DA check of the OBSERVED track (the judge's `decision_check` on the raw observed rows of the
landing approach, from its last runway word, on its labelled runway — check A15–A22 §7.3's reading, which it reproduces:
select 97.97 %, −6.4 / +1.0 / +8.7 m) against the replay's, and the angle correction words per final descent (runs of rows
with "no level-off" in force) and per flight; every flight's row. `--table DIR ... --out <new dir>` puts readouts side by
side (refused unless their specs differ only in H_final and a split's readouts drew the same flights at the same Δ):
`table.json` (`ts-final-descent-tolerance-table-v1`) and `table.md`. No criterion is read (D7): the user chooses (D55).

    python run_ts.py final_descent_tolerance --split train --per-airport 400 --row-interval-s 2 4 8 \
        --instructions <scratch artefact at one H_final> --executor <its executor spec> --out <new dir>
    python run_ts.py final_descent_tolerance --table <readout dir> ... --out <new dir>

### R55 · `run_ts.py start_rules` — the start rules of D77 measured for the user's choice (vocabulary §12.1 A33, D55, D77)

2026-10-05 (`experiments/start_rules.py`). `start_rules --instructions <artefact> --executor <spec dir> --row-interval-s 4
[--rules …] [--workers N --parts P] [--chunk 2048] [--device cpu] --out <new dir>`: every labelled TRAIN flight (train only,
D85; own dynamics or a stand-in's) read in closed loop at one Δ in memory (`instruction_closed_loop.read_interval`, nothing
written into the artefact) once for each start rule (`autopilot.params.START_RULES`; `centred-fit-15s` — the data plane's
centred fit, no rule of a formal spec — for the comparison only), the spec's executor parameters with that rule in place of
its own. Per rule (`readout.json`, `ts-start-rules-readout-v1`): the closed loop's numbers (`instruction_closed_loop.summarise`:
the readings of D34 1–3 and the outcomes, D74), the outcomes and landed share by airport, and the start track at the first
predicted step: on the flights that turn there (more than 5° between the observed directions of the 8 s before and after,
in metres on the ground, `flights.ground_scale`) the distance of the rule's start track (`flights.start_velocity`) from the
direction of the 8 s after (a readout of the future, never an input), pooled and by airport. Train is drawn in `--parts`
consecutive blocks of its one seeded permutation, each drawn once and read at every rule in its own process (up to
`--workers`), put together in order (`merge_tallies`): the readout is the same whatever N and P up to float rounding (a
part flies its flights in other chunks; a rounding can move a discrete step). No criterion is read (D7):
the user chooses (D55). Tests: `tests/test_start_rules.py`.

    python run_ts.py start_rules --row-interval-s 4 --workers 4 --parts 8 \
        --instructions <scratch artefact> --executor <its executor spec> --out <new dir>

### R70 · `run_ts.py pass_gradients` — which part of the loss sets a round's pass, stages C and D

2026-10-10 (the user: is a flat reward the learning rate's or the reward's?). A round whose groups are still on disk (a
round left half done; a round's groups are deleted after its pass), read at the model that spoke it (the round before's
checkpoint, or the start), its pass's updates in the pass's own order (`update_pairs` with the round's numbers), every
`--every`-th, no step taken (every ratio 1). Of each update, by parameter group (the prior's own, the traffic modules')
and over all: the gradient of the surrogate (the advantages), of the pull to the base (× 0.04) and of the data term (×
1) — `post.loss.update_gradients`, whose three parts add up to `update_step`'s gradient — each part's norm, its share
⟨g_part, g⟩ / |g|² of the update's direction (the shares add up to 1) and the cosine of each pair; the means over the
updates read. Under AdamW a parameter's step is about its learning rate whatever its gradient's size: the shares say
which part points the update, the learning rate how far. On the CPU by default (`--device`, `--threads`); the data
term's dropout masks are the pass's in distribution only (another device's generator, or updates left out by `--every`);
of a campaign of several passes a round (D169), the first pass's order. And whether the updates read point one way
(after a 3× learning rate did worse, 2026-10-10), for the surrogate, the data term (a reference) and the whole, by group
and over all: the mean pairwise cosine of the updates' gradients, the cosine of the sums of the even and odd updates,
the coherence |Σ g_i| / Σ |g_i| (1 aligned, about 1/√n independent). Writes `ts-pass-gradients-v2`; a clean tree unless
`--smoke`, the checks first. Tests: `tests/test_pass_gradients.py`, `tests/test_post_loss.py`.

    python run_ts.py pass_gradients --campaign 4dTrajectory/outputs/POOLED/multi/<campaign> --round <r> \
        --out <a new file>.json [--every K] [--device cpu] [--threads T]

### R71 · `run_ts.py lead_ablation` — the words' leads measured one column at a time, for the user's choice (executor design V12, §17, D7, D8)

2026-10-12 (`experiments/lead_ablation.py`). `lead_ablation --instructions <artefact> --executor <spec dir> --column
heading|angle|speed --leads-s 0 2 4 6 8 --per-airport N [--seed 1337] --row-interval-s 2 4 [--chunk 2048] [--device cpu]
--out <new dir>` (train only; no split option, so val and select are never read). ONE lead changed — the heading words'
`heading_lead_s`, the angle words' `angle_lead_s` or the speed words' `speed_lead_s` — to each value, every other value of
the artefact's spec kept (`spec_with_lead`; a lead off the 2 s steps is refused by the spec before any flight). The train
flights drawn as the replay draws them (`replay.draw_flights`, D34's seeded sample, own dynamics or a stand-in's) are READ
AGAIN by the labeller under each value, in memory (no stored sentence is read: those are the artefact's lead's; refusals
counted by reason), and read in closed loop at each Δ in memory by the executor of this code, which reads the leads from
the vocabulary. Per value and Δ (`readout.json`, `ts-lead-ablation-readout-v1`; `lead_<value>s.json` as each ends;
`table.md`): the labeller's flights read / refused and the column's envelopes on the observed tracks (heading rows inside
their bands; tubes or speed spans contained); the closed loop's split numbers (`instruction_closed_loop.summarise`), the
column's correction words per sentence and the landed share (the judge's, D74, a readout); and the error of the flown path
around each OBSERVED word of the column that changes its value, at the said rows −1…+4 from it (e_y for heading, e_h for
angle, flown − observed ground speed at the matched point for speed): p10 / p50 / p90 and the share beyond Y, H or δv.
No criterion is read (D7). Needs an `instruction-v7` artefact and an executor spec of this code (`ts-executor-spec-v11`).

### R74 · `run_ts.py multi_val_evaluation` — a stage D round's val windows flown again and graded by the evaluation package

2026-10-12 (the user: "用 evaluation 跑一下 r7 的 val 轨迹"). Round `--round`'s model flown again on the campaign's val
windows as `multi_validation` flew them (`multi_train.selection_readout` with `window_ends`: the same windows and random
numbers, one process on `--device`); the summed readout must be the val read's own (`--val-read`'s `readout.json`), or the
run stops and writes nothing. The val claim is neither checked nor written: the run flies the windows of the read the claim
spent, by the user's order, and chooses nothing. Each commanded aircraft of each window becomes an evaluation record
(`aircraft_record`: subject `predicted`, no controls — the speed gate reads the declared 1 g; latitude and longitude from the
airport frame, the MSL height, ψ from the compass track, V and γ from the ground speed and vertical rate, no wind; the runway
its crossing landed on, else its observed one; the target the evaluation context's threshold, LTP + TCH; the observed
record's source without its observed-only fields and `aircraft_type`; the type and mass the executor flew, the scenario's —
never the observed record's mass, the harvest's nominal 60 t for a type it does not resolve, which put a Phenom 300's
speed window at 168 m/s in the first run of 2026-10-12), one a window and aircraft;
`python -m evaluation` grades each airport's records, and the same flights' observed records are graded by the same code
(`executor_replay.observed_verdicts`, the two reports' schema and methodology required equal). Writes, into a new directory,
`config.json` (`ts-multi-val-evaluation-v1`), `records/<ICAO>/`, `observed/<ICAO>/`, `aircraft.jsonl` (an aircraft a line:
window, span, outcome, runway, the evaluation's verdict and gates, the observed verdict), `readout.json` (`readout_table`:
pooled and by airport, each by span: outcomes, verdicts and pass share, gates of the landed, verdict by outcome, and of the
aircraft whose observed track passes the share whose flown track passes) and `SHA256SUMS`, read-only. A clean tree; the
checks first; no criterion is read (D7). Tests: `tests/test_multi_val_evaluation.py`.

    python run_ts.py multi_val_evaluation --campaign 4dTrajectory/outputs/POOLED/multi/<campaign> --round <r> \
        --val-read 4dTrajectory/outputs/POOLED/multi/<val read> --out 4dTrajectory/outputs/POOLED/multi/<id> [--device cuda]
