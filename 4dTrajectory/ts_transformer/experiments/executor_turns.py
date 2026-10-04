"""Executor A13: the turn of the executor measured (design §14.2 A13) — how much of the lateral offset after a turn comes
from the heading law of the executor (§5.4), and how much from the words (the 5° grid, the lead). Information, before
the formal artefact: the user decides from it if the heading law changes. NO CRITERION IS READ (D7).

Who: the replay's flights (`replay.draw`: ``--split``, ``--per-airport`` of each airport in a seeded sample, on their own
dynamics or a stand-in's), their open-loop sentences at Δ = 2 s, each flown from its first row on the TIME clock in
three ways (`WAYS`):

(a) ``executor`` — the executor at the observed ground speed (`ObservedSpeedExecutor`; past the observed flight's end,
    its last), so that a difference of speed moves nothing: its airspeed set at the start of every cycle so that its
    ground speed is the observed one, and moved within the cycle to the observed one at its end in place of the speed
    law (Claude's reading of "its airspeed set to the observed ground speed at every cycle": the same ground speed, not
    the same airspeed — they differ by cos γ — and no speed word pulling it away inside a cycle);
(b) ``executor_no_stopping`` — as (a), with the stopping-rate limit of the heading law lifted (`NoStoppingLateral`: the
    error over the time left of the lead, up to the vocabulary's largest turn rate);
(c) ``exact_words`` — the exact model of the heading words (§11.9, `exact_positions`): each word's track reached the
    lead after the word, linearly in time from the track there (a later word measured from the word before it, as the
    executor measures it; the first the shorter way from the observed track at row 0), at the observed ground speed,
    from the observed position at row 0.

The settings of (a) and (b) belong to this runner only: the laws of the executor do not change.

THE READOUT. The turns are the observed flight's runs of 2 s rows that turn one way faster than the turn onset rate
(0.2°/s, §4.3, `labeller.lateral.turn_runs` on the labeller's smoothed track). For each turn and way, the change of the
lateral offset from the observed path (e_y, the closed loop's matched point, `closed_loop.ObservedPath`, right of the
observed track positive) from the row the turn starts at to `AFTER_S` after the row it ends at — and that change toward
the outside of the turn (``outward_m``: positive where the flown path ends outside the observed turn). A turn whose end
+ `AFTER_S` lies past the observed flight, or past the end of a way's flight, is counted as not measured for that way.
Per stratum (straight-in / vectored, `instructions.readout`) and band of the observed ground speed over the turn
(`SPEED_BANDS_MPS`), and pooled: the turns, |change| p50 / p90 and the outward change p50 / mean, per way.

    python run_ts.py executor_turns --split train --per-airport 0 \\
        --instructions 4dTrajectory/outputs/POOLED/instruction_language/<artefact> \\
        --executor 4dTrajectory/outputs/POOLED/executor/<name> --out <new directory>
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ts_transformer.autopilot import replay
from ts_transformer.autopilot.closed_loop import ObservedPath
from ts_transformer.autopilot.executor import Executor, Flown
from ts_transformer.autopilot.frame import GAMMA, SPEED, AirportCharts
from ts_transformer.autopilot.judge import flown_track
from ts_transformer.autopilot.lateral import Lateral, Runways
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.autopilot.sentence import Sentences, TimeClock, WordsNow
from ts_transformer.autopilot.speed import Speed
from ts_transformer.instructions.artefact import SPLITS
from ts_transformer.instructions.labeller.lateral import turn_runs
from ts_transformer.instructions.labeller.read import smooth, truncated
from ts_transformer.instructions.readout import STRATA, stratum
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.instructions.words import HEADING, Words, wrap180
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.repo_layout import REPO_ROOT, git_state

READOUT_SCHEMA = "ts-executor-turns-readout-v1"
WAYS = ("executor", "executor_no_stopping", "exact_words")
#: How long after a turn's end its offset is read, s (§14.2 A13).
AFTER_S = 30.0
#: Bands of the observed ground speed over a turn, m/s: [low, high).
SPEED_BANDS_MPS = ((0.0, 70.0), (70.0, 85.0), (85.0, 100.0), (100.0, math.inf))
ROW_INTERVAL_S = 2.0


def observed_speed_mps(signals: FlightSignals, cycles: int, cycle_s: float) -> np.ndarray:
    """The observed ground speed at the start of each of ``cycles`` + 1 cycles from the flight's row 0, linear between
    its rows (its last past its end)."""
    return np.interp(np.arange(cycles + 1) * cycle_s, signals.time_s, signals.ground_speed_mps)


class NoStoppingLateral(Lateral):
    """The heading law of §5.4 without its stopping-rate limit (way (b)): the error over the time left until the lead
    runs out (floored at 2Δt), up to the vocabulary's largest turn rate."""

    def rate(self, state, relative_deg, issued, runway, runways, time_s, *, fresh=None):  # type: ignore[override]
        course = runways.pointed(runway)[2]
        error = self.word_error(state, relative_deg, issued, course, time_s, fresh)
        to_go = self.heard_s + self.spec.heading_lead_s - time_s
        rate = (error / to_go.clamp(min=2.0 * self.params.cycle_s)).abs()
        return torch.sign(error) * rate.clamp(max=self.spec.turn_rate_max_deg_s)


class ObservedSpeedLaw(Speed):
    """Ways (a), (b): instead of the speed words, the airspeed rate that moves the ground speed from the observed one at
    the cycle's start to the observed one at its end (``ground_speed_mps`` ``[B, cycles + 1]``); the cycle counted by
    the executor (``cycle``, set before each cycle). The thrust box still binds (and is recorded)."""

    def __init__(self, approach_ias_mps: torch.Tensor, spec: VocabularySpec, ground_speed_mps: torch.Tensor,
                 cycle_s: float) -> None:
        super().__init__(approach_ias_mps, spec)
        self.ground_speed_mps, self.cycle_s, self.cycle = ground_speed_mps, cycle_s, 0

    def rate(self, state, speed_mps, unspecified, go_around, load_factor, aero_params,  # type: ignore[override]
             straight_m):
        k = min(self.cycle, self.ground_speed_mps.shape[1] - 2)
        accel = (self.ground_speed_mps[:, k + 1] - self.ground_speed_mps[:, k]) / (self.cycle_s * torch.cos(
            state.gamma_rad))
        return accel, accel, {"stall_floor": torch.zeros_like(accel, dtype=torch.bool)}


class ObservedSpeedExecutor(Executor):
    """The executor flown at the observed ground speed (ways (a), (b)): its airspeed set at the start of every cycle so
    that its ground speed is the observed one (``ground_speed_mps`` ``[B, cycles + 1]``; past the observed flight's end,
    its last), and moved within the cycle to the observed one at its end (`ObservedSpeedLaw`); ``stopping`` False flies
    `NoStoppingLateral`."""

    def __init__(self, *args: Any, ground_speed_mps: torch.Tensor, stopping: bool, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.ground_speed_mps = ground_speed_mps
        self.speed = ObservedSpeedLaw(self.speed.approach_ias_mps, self.words.spec, ground_speed_mps, self.params.cycle_s)
        if not stopping:
            self.lateral = NoStoppingLateral(len(ground_speed_mps), self.params, self.words.spec, ground_speed_mps.device)

    def cycle(self, force: WordsNow, sentence_s: torch.Tensor) -> None:
        state = self.state.clone()
        k = min(self.count, self.ground_speed_mps.shape[1] - 1)
        state[:, SPEED] = self.ground_speed_mps[:, k] / torch.cos(state[:, GAMMA])
        self.state = state
        self.speed.cycle = self.count
        super().cycle(force, sentence_s)


def fly_way(batch: replay.Batch, params: ExecutorParams, words: Words, *, stopping: bool,
            device: torch.device) -> Flown:
    """Every flight's sentence flown on the time clock at the observed ground speed (ways (a), (b)), as `executor.fly`
    flies it."""
    f64 = torch.float64
    params = replace(params, word_clock="time")
    sentences = Sentences([s.grid for s in batch.sentences], words, step_s=batch.row_interval_s, device=device)
    limits = replay.time_limits_s(batch, params, words.spec.step_s)
    cycles = int(math.ceil((max(limits) + replay.reserve_s(batch)) / params.cycle_s))
    speed = torch.tensor(np.array([observed_speed_mps(s, cycles, params.cycle_s) for s in batch.signals]),
                         dtype=f64, device=device)
    executor = ObservedSpeedExecutor(
        batch.inputs(device), Runways.of(batch.geometries, words.spec, dtype=f64, device=device),
        AirportCharts.of(batch.geometries, dtype=f64, device=device),
        torch.tensor(batch.approach_ias_mps, dtype=f64, device=device), params, words, step_s=batch.row_interval_s,
        time_limit_s=torch.tensor(limits, dtype=f64, device=device), reserve_s=replay.reserve_s(batch),
        ground_speed_mps=speed, stopping=stopping)
    clock = TimeClock(params.cycle_s)
    for cycle in range(executor.cycles):
        sentence_s = clock.now(cycle, executor.now())
        if cycle % executor.step_rows == 0:
            step_start_s = sentence_s
        executor.cycle(sentences.at(step_start_s), sentence_s)
        if bool(executor.done.all()):
            break
    return executor.flown()


def exact_positions(sentence: replay.Sentence, signals: FlightSignals, lead_s: float, step_s: float,
                    cycle_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Way (c), the exact model of the heading words (module docstring): east and north, m, at each of the sentence's
    rows (``step_s`` apart), integrated in cycles of ``cycle_s`` at the observed ground speed."""
    rows = len(sentence.grid)
    cycles = int(round((rows - 1) * step_s / cycle_s))
    per_row = int(round(step_s / cycle_s))
    said = sorted((i.row * step_s, float(i.info["target_deg"])) for i in sentence.instructions if i.column == HEADING)
    times = np.arange(cycles + 1) * cycle_s
    tracks = np.empty(len(times))
    # the ramp in force (unwrapped degrees): from `begin` at `heard` to `target` at `heard + lead_s`
    begin = target = float(signals.track_deg[0])
    heard, next_word = -math.inf, 0

    def on_ramp(t: float) -> float:
        return target if t >= heard + lead_s else begin + (target - begin) * (t - heard) / lead_s

    for k, t in enumerate(times):
        while next_word < len(said) and said[next_word][0] <= t:
            begin, heard = on_ramp(t), said[next_word][0]
            # the first word turns the shorter way from the track; a later one from the word before it (§5.4)
            base = begin if next_word == 0 else target
            target = base + float(wrap180(said[next_word][1] - base))
            next_word += 1
        tracks[k] = on_ramp(t)
    speed = observed_speed_mps(signals, cycles, cycle_s)
    rad = np.radians(tracks)
    step_e = 0.5 * (speed[1:] * np.sin(rad[1:]) + speed[:-1] * np.sin(rad[:-1])) * cycle_s
    step_n = 0.5 * (speed[1:] * np.cos(rad[1:]) + speed[:-1] * np.cos(rad[:-1])) * cycle_s
    e = float(signals.e_m[0]) + np.concatenate(([0.0], np.cumsum(step_e)))
    n = float(signals.n_m[0]) + np.concatenate(([0.0], np.cumsum(step_n)))
    return e[::per_row], n[::per_row]


def offsets_m(signals: FlightSignals, span: int, e_m: np.ndarray, n_m: np.ndarray) -> np.ndarray:
    """e_y of flown positions at the observed rows (one per row from row 0) against the observed path of the first
    ``span`` rows: the closed loop's matched point, searched forward row by row."""
    path = ObservedPath(signals.e_m[:span], signals.n_m[:span], np.zeros(span), 0)
    return np.array([path.match(float(e), float(n), 0.0).lateral_m for e, n in zip(e_m, n_m)])


def turns(signals: FlightSignals, span: int, words: Words) -> list[tuple[int, int, float, float]]:
    """``(start row, end row, net turn deg (compass, right positive), mean observed ground speed m/s)`` of every turn of
    the observed flight's first ``span`` rows (module docstring)."""
    smoothed = smooth(truncated(signals, span), words.spec)
    track = np.degrees(np.unwrap(np.radians(smoothed.track_deg)))
    rate = np.diff(track) / words.spec.step_s
    return [(start, stop, float(track[stop] - track[start]),
             float(np.mean(signals.ground_speed_mps[start: stop + 1])))
            for start, stop in turn_runs(rate, words.spec.turn_onset_rate_deg_s)]


def turn_rows(batch: replay.Batch, flown: dict[str, Flown | list[tuple[np.ndarray, np.ndarray]]],
              words: Words) -> list[dict[str, Any]]:
    """One record per turn of ``batch``'s flights (module docstring); ``flown`` each way's flight: a `Flown` of ways (a)
    and (b), the exact positions of way (c)."""
    step_s = words.spec.step_s
    after = int(round(AFTER_S / step_s))
    out = []
    for j, signals in enumerate(batch.signals):
        reading, sentence = batch.readings[j], batch.sentences[j]
        span = min(len(reading.words) - sentence.first_row, len(sentence.grid))
        offsets = {}
        for way in WAYS:
            result = flown[way]
            if isinstance(result, Flown):
                cycles = int(round(step_s / result.cycle_s))
                last = min(int(result.done_cycle[j]) + 1, result.states.shape[1] - 1)
                rows = min(span, last // cycles + 1)
                track = flown_track(result.states[j, : (rows - 1) * cycles + 1].cpu().numpy(), batch.geometries[j])
                e_m, n_m = track["e"][::cycles], track["n"][::cycles]
            else:
                e_m, n_m = result[j]
            # a flight's rows up to its end, and up to a state the dynamics left (a dynamics failure)
            finite = np.isfinite(e_m) & np.isfinite(n_m)
            rows = min(span, len(e_m), int(np.argmin(finite)) if not finite.all() else len(e_m))
            offsets[way] = offsets_m(signals, span, e_m[:rows], n_m[:rows])
        for start, stop, turned, speed in turns(signals, span, words):
            record: dict[str, Any] = {"dataset_id": reading.dataset_id, "airport": reading.airport,
                                      "stratum": stratum(reading), "start_row": start, "end_row": stop,
                                      "turn_deg": turned, "ground_speed_mps": speed}
            for way in WAYS:
                lateral = offsets[way]
                measured = stop + after < len(lateral)
                change = float(lateral[stop + after] - lateral[start]) if measured else None
                record[way] = None if change is None else {"change_m": change,
                                                           "outward_m": -math.copysign(1.0, turned) * change}
            out.append(record)
    return out


def band_name(speed_mps: float) -> str:
    low, high = next(b for b in SPEED_BANDS_MPS if b[0] <= speed_mps < b[1])
    return f"{low:g}-{high:g} m/s"


def readout_table(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per stratum (and "all") and speed band (and "all"), per way: the turns measured and not, |change| p50 / p90, the
    outward change p50 and mean (module docstring)."""
    table: dict[str, Any] = {}
    for part in (*STRATA, "all"):
        for band in (*(band_name(low) for low, _ in SPEED_BANDS_MPS), "all"):
            members = [r for r in records if part in (r["stratum"], "all")
                       and band in (band_name(r["ground_speed_mps"]), "all")]
            cell: dict[str, Any] = {"turns": len(members)}
            for way in WAYS:
                measured = [r[way] for r in members if r[way] is not None]
                size = [abs(m["change_m"]) for m in measured]
                outward = [m["outward_m"] for m in measured]
                cell[way] = {"measured": len(measured), "not_measured": len(members) - len(measured),
                             "abs_change_m": None if not size else {"p50": float(np.percentile(size, 50)),
                                                                     "p90": float(np.percentile(size, 90))},
                             "outward_m": None if not outward else {"p50": float(np.percentile(outward, 50)),
                                                                    "mean": float(np.mean(outward))}}
            table.setdefault(part, {})[band] = cell
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True, help="the executor spec directory")
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--per-airport", type=int, default=0, help="0: every labelled flight of the split")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--chunk", type=int, default=500)
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    instructions = args.instructions if args.instructions.is_absolute() else REPO_ROOT / args.instructions
    executor = args.executor if args.executor.is_absolute() else REPO_ROOT / args.executor
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if out.exists():
        parser.error(f"{out} exists; a readout is never overwritten")
    if args.split != "train" and git_state()["dirty"]:
        parser.error(f"the {args.split} readout runs from a clean tree")
    params, record, words = replay.open_executor(executor, instructions)
    device, started = torch.device(args.device), time.perf_counter()
    batch = replay.draw(instructions, args.split, words.spec, words, per_airport=args.per_airport, seed=args.seed,
                        groups=(replay.OWN, replay.STAND_IN), row_interval_s=ROW_INTERVAL_S)
    print(f"{len(batch.sentences)} {args.split} flights ({batch.drawn['by_group']}; not flown "
          f"{batch.drawn['excluded']}), {time.perf_counter() - started:.0f}s", flush=True)
    records: list[dict[str, Any]] = []
    for first in range(0, len(batch.sentences), args.chunk):
        part = replay.subset(batch, list(range(first, min(first + args.chunk, len(batch.sentences)))))
        flown: dict[str, Any] = {
            "executor": fly_way(part, params, words, stopping=True, device=device),
            "executor_no_stopping": fly_way(part, params, words, stopping=False, device=device),
            "exact_words": [exact_positions(s, f, words.spec.heading_lead_s, words.spec.step_s, params.cycle_s)
                            for s, f in zip(part.sentences, part.signals)]}
        records += turn_rows(part, flown, words)
        print(f"  {first + len(part.sentences)} flights, {len(records)} turns, {time.perf_counter() - started:.0f}s",
              flush=True)
    out.mkdir(parents=True)
    table = readout_table(records)
    write_json_atomic(out / "turns.json", {
        "schema": READOUT_SCHEMA, "written_utc": utc_now(), "split": args.split, "row_interval_s": ROW_INTERVAL_S,
        "after_s": AFTER_S, "executor_spec_sha256": record["sha256"], "vocabulary_spec_sha256": words.spec.sha256,
        "drawn": batch.drawn, "ways": list(WAYS), "readout": table, "turns": records, "git": git_state(),
        "elapsed_s": time.perf_counter() - started})
    for part, bands in table.items():
        for band, cell in bands.items():
            text = "  ".join(f"{way} {cell[way]['abs_change_m']} out {cell[way]['outward_m']}" for way in WAYS)
            print(f"  {part:12s} {band:14s} turns {cell['turns']:5d}  {text}")
    print(f"→ {out / 'turns.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
