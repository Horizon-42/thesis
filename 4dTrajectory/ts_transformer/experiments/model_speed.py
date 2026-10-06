"""D136: how fast the model speaks (outline §6.2 item 10) — the time of the prior's step of a row and of the executor's
steps of it, apart, on fixed inputs of the select days (never val or test).

WHAT. Stage B (``--prior``): a prior run (the base) speaking select-day flights to their end through the shared step of
a closed loop (`prior_speaking_loop.SpeakingLoop`, prior §7 item 7), one sample at temperature 1 — ``--per-airport``
flights of each airport drawn with ``--seed`` (`prior_free_generation.draw`). Stage C (``--campaign``, ``--round``): a
post-training round's model speaking the select windows of its campaign's selection readout
(`post_train.selection_windows`) in the window loop (`post_window_loop.WindowLoop`: the traffic attention and the
separation masks with the prior's step).

TIMES, EACH APART (`TimedLoop`). The executor's steps of a row are the closed loop's `step` (and its `halt` after the
row), timed between two synchronisations on the GPU; the prior's step of the row is everything else of the row — from the end of the executor's
previous step (or the end of the observed rows) to the start of this one: the row's inputs, the model with its cache,
the masks, the draw and the loop's records (C: the traffic attention and the separation masks too). A row's time is
the two together. Before each setting is timed, a warm-up speaks the setting's first group for `WARMUP_ROWS` rows and
is thrown away: the first rows after loading are never timed.

SETTINGS (Claude's proposal, outline §6.2 item 10; the user may change them). Each ``--devices`` (the CPU with one
thread, as the closed loop runs; the GPU, the executor on it too; every setting with one CPU thread) × each ``--batches``: 1, one aircraft at a time (the
time a controller in the loop waits), and `BATCH`, free generation's chunk in B5. Stage B's batch of `BATCH` holds the
drawn flights repeated in turn (copy k of a flight is its sample k, `flight_numbers`), as many as make `BATCH`; a batch
smaller than the drawn flights speaks them in loops of that size (`prior_groups`); stage C's batches command each flight once (`post_train.batches`), so a batch holds at most `BATCH`
windows — the readout records how many each held.

READOUT (``speed.json``, `SPEED_SCHEMA`), for each setting: the loops and their sizes, the rows timed, a row's prior
step, executor steps and whole time (p50, p95, largest; ms), a sentence's time (p50, p95, largest; s: the rows of its
loop it was said in — the encoding of its observed rows before the first predicted step not included), the
flight rows said a second, and the share of the row interval Δ that one row takes (p50 and p95, the prior's step and
the executor's apart); as information: the device, torch's version and threads, the host's load and the GPU's free
memory at the start, the commit. Writes into ``--out`` (new; `4dTrajectory/outputs/POOLED/speed/<model run>_<date>/`,
never the model's read-only directory). Run with no other job on the host or the GPU (outline §5 rule 13): the load is
recorded, not judged.

    python run_ts.py model_speed --prior 4dTrajectory/outputs/POOLED/prior/prior_base_20261006/base/run \\
        --instructions 4dTrajectory/outputs/POOLED/instruction_language/v12_20261005 \\
        --executor 4dTrajectory/outputs/POOLED/executor/v17_20261005 \\
        --out 4dTrajectory/outputs/POOLED/speed/prior_base_20261006_base_20261006
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from ts_transformer.instructions.words import COLUMNS
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.repo_layout import REPO_ROOT, git_state, repo_relative

#: The format of ``speed.json``.
SPEED_SCHEMA = "ts-model-speed-v1"
#: The rows of the warm-up spoken before a setting is timed, then thrown away.
WARMUP_ROWS = 20
#: The batch of free generation's chunk in B5.
BATCH = 400
#: The split read: the select days only (never val or test).
SPLIT = "select"


class Warmed(Exception):
    """The warm-up has said its rows: the loop it was spoken in is thrown away."""


@dataclass
class TimedLoop:
    """A closed loop (`autopilot.start.Loop`) whose executor steps are timed, given to the loop that speaks in it as the
    loop itself (every other attribute is the loop's). ``sync`` waits for the device (the GPU's synchronisation; nothing
    on the CPU). ``limit``: the rows after which it stops the speaking (`Warmed`), for the warm-up; None for none. Each
    step records the prior's time before it (from the previous step's end, or the end of the observed rows: `rows`),
    the executor's time of it, and the flights flying in it."""

    loop: Any
    sync: Callable[[], None]
    limit: int | None = None
    prior_s: list[float] = field(default_factory=list)
    executor_s: list[float] = field(default_factory=list)
    flying: list[np.ndarray] = field(default_factory=list)
    #: when the observed rows ended (`rows`): every moment after it is in one timing or the other
    started: float | None = None
    _since: float | None = None

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or name in ("loop", "sync", "limit", "prior_s", "executor_s", "flying", "started"):
            raise AttributeError(name)
        return getattr(self.loop, name)

    def rows(self) -> np.ndarray:
        """The loop's states now: read once, at the end of the observed rows — the prior's first step is timed from
        here."""
        out = self.loop.rows()
        self.sync()
        self._since = self.started = time.perf_counter()
        return out

    def halt(self, flights: np.ndarray) -> None:
        """The loop's halt (the executor's work after a row: holding the flights done), timed as the executor's, into
        the row just flown (a halt before the first step is refused: no row holds it)."""
        self.sync()
        begin = time.perf_counter()
        self.loop.halt(flights)
        self.sync()
        end = time.perf_counter()
        # into the row just flown; what the loop did between the step and the halt stays the prior's
        self.executor_s[-1] += end - begin
        self._since += end - begin

    def step(self, words_row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.limit is not None and len(self.prior_s) >= self.limit:
            raise Warmed()
        if self._since is None:
            raise ValueError("a step before the end of the observed rows: the prior's time has no start")
        executor = self.loop.executor
        flying = (~(executor.done | executor.halted)).cpu().numpy()
        self.sync()
        begin = time.perf_counter()
        out = self.loop.step(words_row)
        self.sync()
        end = time.perf_counter()
        self.prior_s.append(begin - self._since)
        self.executor_s.append(end - begin)
        self.flying.append(flying)
        self._since = end
        return out


def speed_source(directory: Path, stage: str) -> dict[str, Any]:
    """The speed readout a Training set names (``source.speed``, outline §6.2 item 10, D136): ``directory`` a readout of
    this runner of ``stage`` ("B" or "C"), refused otherwise; its directory (repository-relative), the model it timed (a
    stage-B fold set names the base's readout and says so by the prior it names) and whether it is a smoke."""
    record = json.loads((directory / "speed.json").read_text(encoding="utf-8"))
    if record["schema"] != SPEED_SCHEMA or record["model"]["stage"] != stage:
        raise ValueError(f"{directory} is a {record['schema']} readout of stage {record['model']['stage']}, not a "
                         f"{SPEED_SCHEMA} readout of stage {stage}")
    timed = record["model"]["prior"] if stage == "B" else f"{record['model']['campaign']} round {record['model']['round']}"
    return {"readout": repo_relative(directory), "model": timed, "smoke": bool(record["smoke"])}


def prior_groups(drawn: Sequence[int], batch: int) -> list[list[tuple[int, int]]]:
    """Stage B's loops of ``batch`` flights (module docstring), each a list of ``(flight, sample)``: the drawn flights in
    loops of ``batch`` (the last one shorter), each spoken once (sample 0) — or, when ``batch`` is larger than the drawn
    flights, one loop of the drawn flights repeated in turn to ``batch``, copy k of a flight its sample k. Every drawn
    flight is spoken; the readout records each loop's size."""
    if batch <= len(drawn):
        return [[(f, 0) for f in drawn[begin: begin + batch]] for begin in range(0, len(drawn), batch)]
    return [[(drawn[k % len(drawn)], k // len(drawn)) for k in range(batch)]]


def percentiles(values: Sequence[float], scale: float) -> dict[str, float]:
    """p50, p95 and the largest of ``values``, times ``scale``."""
    array = np.asarray(values, dtype=np.float64) * scale
    return {"p50": float(np.percentile(array, 50)), "p95": float(np.percentile(array, 95)), "max": float(array.max())}


def summary(timed: Sequence[TimedLoop], interval_s: float) -> dict[str, Any]:
    """A setting's readout (module docstring) from its timed loops."""
    prior = np.concatenate([np.asarray(t.prior_s) for t in timed])
    executor = np.concatenate([np.asarray(t.executor_s) for t in timed])
    if not len(prior):
        raise ValueError("no row was timed")
    row = prior + executor
    flight_rows = sum(int(f.sum()) for t in timed for f in t.flying)
    # a sentence's time: its loop's rows while it flew (each flight is flying in a row it is said in)
    sentences = [float(sum(p + e for p, e, f in zip(t.prior_s, t.executor_s, t.flying) if f[b]))
                 for t in timed for b in range(len(t.flying[0]))]
    return {"rowsTimed": int(len(row)), "sentences": len(sentences),
            "priorStepMs": percentiles(prior, 1e3), "executorStepsMs": percentiles(executor, 1e3),
            "rowMs": percentiles(row, 1e3), "sentenceS": percentiles(sentences, 1.0),
            "flightRowsPerS": flight_rows / float(row.sum()),
            "shareOfInterval": {"intervalS": interval_s,
                                "prior": {k: v / 1e3 / interval_s for k, v in percentiles(prior, 1e3).items()
                                          if k != "max"},
                                "executor": {k: v / 1e3 / interval_s for k, v in percentiles(executor, 1e3).items()
                                             if k != "max"},
                                "row": {k: v / 1e3 / interval_s for k, v in percentiles(row, 1e3).items()
                                        if k != "max"}}}


def synchroniser(device: torch.device) -> Callable[[], None]:
    """The device's wait: the GPU's synchronisation, nothing on the CPU."""
    if device.type == "cuda":
        return lambda: torch.cuda.synchronize(device)
    return lambda: None


def host_info(device: torch.device) -> dict[str, Any]:
    """What the setting ran on, as information (never judged)."""
    out = {"device": str(device), "torch": torch.__version__, "threads": torch.get_num_threads(),
           "loadAverage": list(os.getloadavg()), "cpus": os.cpu_count()}
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        out.update(name=torch.cuda.get_device_name(device), freeMemoryBytes=int(free), totalMemoryBytes=int(total))
    return out


# ---- stage B: a prior run
@dataclass(frozen=True)
class PriorRun:
    """What stage B's settings fly: the prior opened (`checkpoint.open_prior`), the artefact, the executor spec, the
    select days' closed-loop sentences and records, and the flights drawn."""

    prior: Any
    instructions: Path
    executor: Path
    sentences: Mapping[int, Any]
    flights: Mapping[int, Mapping[str, Any]]
    finals: Mapping[str, Any]
    drawn: list[int]


def open_prior_run(prior_dir: Path, instructions: Path, executor: Path, per_airport: int, seed: int) -> PriorRun:
    from ts_transformer.experiments.prior_free_generation import draw
    from ts_transformer.instructions.artefact import closed_loop_sentences, load_spec, signals_flights
    from ts_transformer.prior.checkpoint import open_prior
    from ts_transformer.prior.procedure import airport_finals

    prior = open_prior(prior_dir, instructions)
    sentences = closed_loop_sentences(instructions, SPLIT, prior.interval_s, load_spec(instructions))
    flights = signals_flights(instructions, SPLIT)
    drawn = draw(sentences, flights, sorted(prior.geometries), per_airport, seed)
    return PriorRun(prior, instructions, executor, sentences, dict(enumerate(flights)),
                    {code: airport_finals(g) for code, g in prior.geometries.items()}, drawn)


def speak_prior(run: PriorRun, model: Any, group: Sequence[tuple[int, int]], seed: int, device: torch.device,
                limit: int | None = None) -> TimedLoop:
    """The flights ``group`` (``(flight, sample)``) spoken to their end in one closed loop on ``device``, timed."""
    from ts_transformer.autopilot.start import NO_MOVE, start_moved
    from ts_transformer.experiments.prior_speaking_loop import SpeakingLoop, flight_numbers
    from ts_transformer.prior.speaker import MOST_GO_AROUNDS

    flights = sorted({f for f, _ in group})
    loop, order, observed = start_moved(run.instructions, SPLIT, run.prior.interval_s,
                                        {i: run.sentences[i] for i in flights}, run.executor,
                                        {i: NO_MOVE for i in flights}, most_go_arounds=MOST_GO_AROUNDS, device=device)
    place = {flight: p for p, flight in enumerate(order)}
    loop = loop.copy([place[f] for f, _ in group]) if [f for f, _ in group] != list(order) else loop
    timed = TimedLoop(loop, synchroniser(device), limit)
    speaking = SpeakingLoop(model, timed, [f for f, _ in group], run.sentences, observed, run.flights,
                            run.prior.geometries, [run.prior.landings[run.flights[f]["airport"]] for f, _ in group],
                            run.finals, loop.words, interval_s=run.prior.interval_s, variant=model.config.variant,
                            device=device, temperature=1.0)
    numbers = [flight_numbers(seed, sample, f) for f, sample in group]
    try:
        while speaking.observing:
            speaking.observe()
        while speaking.alive.any():
            speaking.step(np.stack([n.random(len(COLUMNS)) for n in numbers]))
    except Warmed:
        pass
    return timed


def prior_setting(run: PriorRun, device: torch.device, batch: int, seed: int) -> dict[str, Any]:
    model = run.prior.checkpoint.model.to(device).eval()
    groups = prior_groups(run.drawn, batch)
    info = host_info(device)
    with torch.no_grad():
        speak_prior(run, model, groups[0], seed, device, limit=WARMUP_ROWS)                    # thrown away
        timed = [speak_prior(run, model, group, seed, device) for group in groups]
    return {"device": str(device), "batch": batch, "loops": len(groups), "loopSizes": [len(g) for g in groups],
            **summary(timed, run.prior.interval_s), "host": info}


# ---- stage C: a round of a post-training campaign
def window_setting(context: Any, model: Any, windows: Sequence[Any], seed: int, device: torch.device, batch: int
                   ) -> dict[str, Any]:
    """Stage C's setting (module docstring): ``model`` (a round's, `post_train.round_model`) on ``windows`` (the select
    windows of the campaign's selection readout) in the window loop, in batches that command each flight once
    (`post_train.batches`), each loop timed; the numbers of each window the selection readout's (`readout_numbers`)."""
    from ts_transformer.experiments.post_train import batches, readout_numbers
    from ts_transformer.experiments.post_window_loop import WindowLoop

    groups = [[p] for p in range(len(windows))] if batch == 1 else batches(windows, batch)
    select = context.splits[SPLIT]

    def fly(places: Sequence[int], limit: int | None) -> TimedLoop:
        chosen = [windows[p] for p in places]
        loop, order, observed = context.start_loop(SPLIT, chosen)([w.signal_index for w in chosen])
        timed = TimedLoop(loop, synchroniser(device), limit)
        flown = WindowLoop(model, timed, order, chosen, select["sentences"], select["flights"], context.geometries,
                           context.rosters, context.finals, context.words, interval_s=context.interval_s,
                           variant=context.variant, edges_reference=context.edges_reference, faults=select["faults"],
                           observed=observed, device=device)
        try:
            flown.run([readout_numbers(seed, p) for p in places])
        except Warmed:
            pass
        return timed

    info = host_info(device)
    with torch.no_grad():
        fly(groups[0], WARMUP_ROWS)                                                            # thrown away
        timed = [fly(group, None) for group in groups]
    return {"device": str(device), "batch": batch, "loops": len(groups), "loopSizes": [len(g) for g in groups],
            **summary(timed, context.interval_s), "host": info}


def open_round(campaign: Path, record: Mapping[str, Any], round_: str, device: torch.device) -> tuple[Any, Any, list]:
    """A campaign's round opened on ``device`` (``record``: its ``campaign.json``, checked by the caller): the context of
    the select days, the round's model and the selection readout's windows."""
    from ts_transformer.experiments.post_train import open_context, round_model, selection_windows, settings_of

    inputs = record["inputs"]
    context = open_context(Path(inputs["prior"]), Path(inputs["instructions"]), Path(inputs["executor"]),
                           Path(inputs["windows"]) / "conformance" / "edges.npz", device, Path(inputs["procedure_root"]),
                           formal=False, data=False, splits=(SPLIT,))
    settings = settings_of(record)
    model = round_model(context, settings, campaign, None if round_ == "start" else int(round_)).eval()
    return context, model, selection_windows(context, settings, SPLIT)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--prior", type=Path, help="stage B: a prior_train run's directory")
    model.add_argument("--campaign", type=Path, help="stage C: a post_train campaign's directory (with --round)")
    parser.add_argument("--round", help="stage C: 'start' or a round's number")
    parser.add_argument("--instructions", type=Path, help="stage B: the artefact")
    parser.add_argument("--executor", type=Path, help="stage B: the directory of the artefact's executor spec")
    parser.add_argument("--per-airport", type=int, default=20, help="stage B: flights of each airport")
    parser.add_argument("--seed", type=int, default=1337, help="stage B: the draw of the flights and their numbers")
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda"])
    parser.add_argument("--batches", nargs="+", type=int, default=[1, BATCH])
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    parser.add_argument("--smoke", action="store_true", help="SMOKE: allowed from a tree with changes; recorded")
    args = parser.parse_args(argv)
    if (args.campaign is None) != (args.round is None):
        parser.error("--round goes with --campaign, and only with it")
    if args.prior is not None and (args.instructions is None or args.executor is None):
        parser.error("--prior needs --instructions and --executor")
    if args.campaign is not None and (args.instructions is not None or args.executor is not None):
        parser.error("--campaign reads its artefact and executor spec from its campaign.json")
    if args.per_airport < 1 or any(b < 1 for b in args.batches):
        parser.error("--per-airport and --batches are at least 1")
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if out.exists():
        parser.error(f"{out} exists; a readout is never overwritten")
    git = git_state()
    if git["dirty"] and not args.smoke:
        parser.error("a readout that is not a smoke needs a clean tree")
    torch.set_num_threads(1)                    # every setting: one CPU thread, as the closed loop runs
    settings = []
    if args.prior is not None:
        from ts_transformer.autopilot.closed_loop import require_conforming_closed_loop

        prior_dir, instructions, executor = (p if p.is_absolute() else REPO_ROOT / p
                                             for p in (args.prior, args.instructions, args.executor))
        require_conforming_closed_loop(instructions, executor)                  # D69: the checks run here (D73)
        run = open_prior_run(prior_dir, instructions, executor, args.per_airport, args.seed)
        subject = {"stage": "B", "prior": repo_relative(prior_dir), "instructions": repo_relative(instructions),
                   "executor": repo_relative(executor), "split": SPLIT, "perAirport": args.per_airport,
                   "seed": args.seed, "flights": len(run.drawn), "intervalS": run.prior.interval_s}
        for name in args.devices:
            for batch in args.batches:
                settings.append(prior_setting(run, torch.device(name), batch, args.seed))
                print(json.dumps({k: settings[-1][k] for k in ("device", "batch", "rowMs")}), flush=True)
    else:
        from ts_transformer.autopilot.closed_loop import require_conforming_closed_loop
        from ts_transformer.experiments.post_train import CAMPAIGN_SCHEMA, settings_of

        campaign = args.campaign if args.campaign.is_absolute() else REPO_ROOT / args.campaign
        record = json.loads((campaign / "campaign.json").read_text(encoding="utf-8"))
        if record["schema"] != CAMPAIGN_SCHEMA:
            parser.error(f"{campaign} is a {record['schema']} campaign, not {CAMPAIGN_SCHEMA}")
        require_conforming_closed_loop(Path(record["inputs"]["instructions"]),
                                       Path(record["inputs"]["executor"]))     # D69: the checks run here (D73)
        seed = settings_of(record).seed
        subject = {"stage": "C", "campaign": repo_relative(campaign), "round": args.round, "split": SPLIT}
        for name in args.devices:
            context, model, windows = open_round(campaign, record, args.round, torch.device(name))
            subject["windows"] = len(windows)
            for batch in args.batches:
                settings.append(window_setting(context, model, windows, seed, torch.device(name), batch))
                print(json.dumps({k: settings[-1][k] for k in ("device", "batch", "rowMs")}), flush=True)
    out.mkdir(parents=True)
    write_json_atomic(out / "speed.json", {
        "schema": SPEED_SCHEMA, "writtenUtc": utc_now(), "model": subject, "warmupRows": WARMUP_ROWS,
        "temperature": 1.0, "settings": settings, "git": git, "smoke": args.smoke})
    print(json.dumps({"out": str(out), "settings": len(settings)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
