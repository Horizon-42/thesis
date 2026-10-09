"""MC11: where a batch of stage D's round spends its time (multi-aircraft control D181 (56)). No criterion is applied
(D7): the report goes to the user (item 57 waits on it). The memory measure is `multi_profile` (D179); this is the time.

WHAT IT MEASURES: batch ``--batch`` of round ``--round``'s draw of a stage D campaign (``--campaign``: its
``campaign.json``, its settings and inputs), spoken once by the stage's own batch speaker (`multi_train.speak_batch`,
what a speaking worker runs: its two passes, its groups written to a scratch directory) with that round's model (round
0: the stage's start; a later round: the round before's checkpoint, `multi_train.round_model`), in this process, torch
on one thread as a worker. Timers at its parts:

- ``first_pass``: the window loop of every window (`WindowLoop.run`);
- ``second_pass``: the windows spoken again, their rows to the last branch point (`WindowLoop.observe`, `step`);
- ``continuations``: the copies at each branch point flown to their ends (`WindowLoop.copy`, `finish`);
- ``samples``: the sentences for the loss of the first pass and of the copies (`WindowLoop.samples`);
- ``other``: the rest of the batch (the starts of the loops, the groups written).

Inside each part, apart: the speaker's steps (`Speaker.say`, `observe`: the model's forward and its masks, the
executor's steps inside them taken out), the executor's steps (`autopilot.start.Loop.step`, `halt`), and the window
loop's own work (the scenes, the tokens, the judge: the part's time less those two). Each timer starts and stops
between two synchronisations of the GPU (as `model_speed`), so the parts add up to the batch. The GPU's peak (reserved
and allocated) is reset when a part is entered and read when it is left; a part's peak is the largest of its
entries. The first part holds the device's warm-up (as a worker's first batch does), stated, never removed.

A profiler (``--cprofile PART``, cProfile, the CPU only) runs only inside that part, only when the timers leave it
unexplained (D181 (56): on the GPU it nearly doubles a batch and counts a kernel's time where the host waits); its
40 functions of most cumulative time are written beside. Run it with no other job on the host or the GPU: the times are
the host's.

The record is ``speed.json`` (`MULTI_SPEED_SCHEMA`) under ``--out`` (a new directory).

    python run_ts.py multi_speed --campaign 4dTrajectory/outputs/POOLED/multi/multi_train_20261009 --round 0 \\
        --batch <k> --device cuda --out <a new directory> [--cprofile PART]
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import torch

from ts_transformer.autopilot.closed_loop import require_conforming_closed_loop
from ts_transformer.autopilot.start import Loop
from ts_transformer.experiments.model_speed import host_info, synchroniser
from ts_transformer.experiments.multi_profile import rows_of
from ts_transformer.experiments.multi_train import multi_settings_of, round_model, span_key, stage_d
from ts_transformer.experiments.post_train import Context, inputs_here, open_context
from ts_transformer.experiments.post_window_loop import WindowLoop, checked_edges
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.prior.model import Prior
from ts_transformer.prior.speaker import Speaker
from ts_transformer.repo_layout import REPO_ROOT, git_state

MULTI_SPEED_SCHEMA = "ts-multi-speed-v1"
#: The parts of a batch (module docstring), in the order a batch enters them.
PARTS = ("first_pass", "second_pass", "continuations", "samples")
#: The timed steps inside a part, each by its methods.
STEPS = {"speaker": ((Speaker, "say"), (Speaker, "observe")), "executor": ((Loop, "step"), (Loop, "halt"))}
#: The window loop's methods that open a part, when no part is open: the part (a method called inside an open part
#: stays in it: `run` calls `observe` and `finish`).
ENTRIES = {"run": "first_pass", "observe": "second_pass", "step": "second_pass", "copy": "continuations",
           "finish": "continuations", "samples": "samples"}
#: The functions of a profile written.
PROFILED_FUNCTIONS = 40


@dataclass
class Timers:
    """The batch's timers (module docstring): each part's time, its entries and its GPU peaks, each step's time inside it
    (its own: a step inside another is taken out of the outer one), ``sync`` the device's wait, ``cuda`` whether the
    GPU's peaks are read, ``profiled`` the part under the profiler (None: none)."""

    sync: Callable[[], None]
    cuda: bool
    profiled: str | None = None
    parts: dict[str, dict[str, Any]] = field(default_factory=lambda: {
        part: {"s": 0.0, "entries": 0, "speaker_s": 0.0, "executor_s": 0.0, "gpu_reserved_peak": None,
               "gpu_allocated_peak": None} for part in PARTS})
    profiler: cProfile.Profile | None = None
    _part: str | None = None
    #: the open steps: their names and the time of the steps inside each
    _open: list[list[Any]] = field(default_factory=list)

    @contextmanager
    def part(self, name: str) -> Iterator[None]:
        if self._part is not None:                     # inside an open part: its own
            yield
            return
        self.sync()
        if self.cuda:
            torch.cuda.reset_peak_memory_stats()
        if name == self.profiled:
            self.profiler.enable()
        self._part, begin = name, time.perf_counter()
        try:
            yield
        finally:
            self.sync()
            end = time.perf_counter()
            if name == self.profiled:
                self.profiler.disable()
            self._part = None
            held = self.parts[name]
            held["s"] += end - begin
            held["entries"] += 1
            if self.cuda:
                for key, peak in (("gpu_reserved_peak", torch.cuda.max_memory_reserved()),
                                  ("gpu_allocated_peak", torch.cuda.max_memory_allocated())):
                    held[key] = max(held[key] or 0, int(peak))

    @contextmanager
    def step(self, name: str) -> Iterator[None]:
        if self._part is None:                         # outside every part (a loop's start): the batch's other time
            yield
            return
        self.sync()
        self._open.append([name, 0.0])
        begin = time.perf_counter()
        try:
            yield
        finally:
            self.sync()
            elapsed = time.perf_counter() - begin
            _, inner = self._open.pop()
            self.parts[self._part][f"{name}_s"] += elapsed - inner
            if self._open:
                self._open[-1][1] += elapsed

    @contextmanager
    def installed(self) -> Iterator[None]:
        """The timers on the window loop's, the speaker's and the executor's methods (module docstring), taken off after;
        this process only."""
        kept = []

        def wrap(owner: type, name: str, opened: Callable[[], Any]) -> None:
            method = getattr(owner, name)
            kept.append((owner, name, method))

            def timed(*args: Any, **kwargs: Any) -> Any:
                with opened():
                    return method(*args, **kwargs)

            setattr(owner, name, timed)

        try:
            for name, part in ENTRIES.items():
                wrap(WindowLoop, name, lambda part=part: self.part(part))
            for step, methods in STEPS.items():
                for owner, name in methods:
                    wrap(owner, name, lambda step=step: self.step(step))
            yield
        finally:
            for owner, name, method in reversed(kept):
                setattr(owner, name, method)


def timed_batch(model: Prior, context: Context, settings: Any, round_: int, batch: int,
                profiled: str | None = None) -> dict[str, Any]:
    """Batch ``batch`` of round ``round_``'s draw spoken once with ``model`` under the timers (module docstring): its
    record (`speak_batch`'s), its windows, rows and span, each part's times and peaks, the rest (``other_s``) and the
    whole, and the profile of ``profiled`` (its text; the CPU only)."""
    if profiled is not None and (profiled not in PARTS or context.device.type != "cpu"):
        raise ValueError(f"a profile of one of {PARTS}, on the CPU only (D181 (56)), not {profiled!r} on "
                         f"{context.device}")
    stage = stage_d()
    windows, _ = stage.draw(context, settings, round_)
    found = stage.batches(windows, settings)
    places = found[batch]
    timers = Timers(synchroniser(context.device), context.device.type == "cuda", profiled,
                    profiler=cProfile.Profile() if profiled is not None else None)
    with tempfile.TemporaryDirectory(prefix="multi_speed_") as scratch, timers.installed():
        timers.sync()
        begin = time.perf_counter()
        record = stage.speak_batch(model, context, windows, places, settings, round_, Path(scratch), batch)
        timers.sync()
        whole = time.perf_counter() - begin
    parts = {name: {**held, "window_loop_s": held["s"] - held["speaker_s"] - held["executor_s"]}
             for name, held in timers.parts.items()}
    out = {"batch": batch, "batches": len(found), "span": span_key(windows[places[0]].span_s), "windows": len(places),
           "rows": rows_of(windows, places), "record": record, "parts": parts,
           "other_s": whole - sum(held["s"] for held in parts.values()), "s": whole}
    if profiled is not None:
        text = io.StringIO()
        pstats.Stats(timers.profiler, stream=text).sort_stats("cumulative").print_stats(PROFILED_FUNCTIONS)
        out["profile"] = {"part": profiled, "text": text.getvalue()}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--campaign", type=Path, required=True, help="a campaign of stage D (its campaign.json)")
    parser.add_argument("--round", type=int, default=0, help="the round whose draw and model speak (default 0)")
    parser.add_argument("--batch", type=int, required=True, help="the batch of that round's draw")
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--cprofile", choices=PARTS, help="a profile inside this part (the CPU only)")
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    args = parser.parse_args(argv)
    if args.cprofile is not None and args.device != "cpu":
        parser.error("--cprofile runs on the CPU only (D181 (56))")
    campaign, out = (p if p.is_absolute() else REPO_ROOT / p for p in (args.campaign, args.out))
    if out.exists():
        parser.error(f"{out} exists: the measure writes a new directory")
    record = json.loads((campaign / "campaign.json").read_text(encoding="utf-8"))
    settings = multi_settings_of(record)
    if not 0 <= args.round < settings.rounds:
        parser.error(f"round {args.round}: the campaign has rounds 0 to {settings.rounds - 1}")
    inputs = inputs_here(record["inputs"])
    _, opened, _ = require_conforming_closed_loop(Path(inputs["instructions"]), Path(inputs["executor"]))   # D73
    edges_reference = Path(inputs["windows"]) / "conformance" / "edges.npz"
    checked_edges(edges_reference)                                                                   # D104
    torch.set_num_threads(1)                                                    # as a speaking worker (`Speakers`)
    device = torch.device(args.device)
    context = open_context(Path(inputs["prior"]), Path(inputs["instructions"]), Path(inputs["executor"]),
                           edges_reference, device, Path(inputs["procedure_root"]), formal=False, data=False,
                           splits=("train",))
    model = (stage_d().start(context, settings)[0] if args.round == 0
             else round_model(context, settings, campaign, args.round - 1)).eval()
    out.mkdir(parents=True)
    head = {"schema": MULTI_SPEED_SCHEMA, "started_utc": utc_now(), "git": git_state(), "checks": opened["checks"],
            "inputs": {"campaign": str(campaign), "round": args.round, "batch": args.batch}, "host": host_info(device)}
    speed = timed_batch(model, context, settings, args.round, args.batch, args.cprofile)
    write_json_atomic(out / "speed.json", json.loads(json.dumps({**head, **speed, "finished_utc": utc_now()},
                                                                default=str)))
    print(json.dumps({key: speed[key] for key in ("span", "windows", "rows", "s", "other_s", "parts")}, indent=1),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
