"""The executor checked by what it flies, not by its source (executor design §12.3, the user 2026-10-01).

A spec's REFERENCE is a fixed batch of labelled train flights — the replay's draw (`replay.draw`: `SPLIT`, `PER_AIRPORT`
an airport, `SEED`, flights on their own dynamics, at the data's row interval) — flown from row 0 on their labelled words
by the code that measured the spec. Every cycle is recorded: states, commands, wanted rates, every limit and mode, the sentence time, and the
cycle each flight was done at. Every flight is judged (`autopilot.judge`: outcome, end row, crossing, each limit's count
and every word's verdict and check numbers). It is written once into ``<spec>/conformance/`` (``reference.json`` +
``reference.npz``), from a clean checkout whose executor code measured the spec (`executor_spec` writes it with the
spec; v11's was written by `a7a32324`, before the executor's code first changed).

The CHECK flies the reference's flights again with the code on disk in every way the executor flies (`MODES`: a
single-aircraft batch, a multi-aircraft batch with staggered starts, the single-flight executor) and compares each
flight with its reference up to the cycle it was done at:
- its states no further apart than `STATE_BOUND_M`, horizontally and vertically;
- every other float (speed, angles — ψ round the circle — mass, commands, wanted rates, sentence times, a verdict's
  check numbers) no further apart than `ROUNDOFF`;
- every limit and mode each cycle, the done cycle, the outcome, the end row and every word's verdict the same;
- every flight of the reference flown.

Before flying, each flight's INPUTS are compared with the reference's digests (`input_digests`: the state it starts
from, its airframe, frame, thrust and approach speed, its time limit, its words and runway, the runways' geometry and
vertical paths): inputs that moved are refused by name — the data under the
reference changed, which says nothing about the executor.

A check that passes writes ``passed-<code>.json`` beside the reference (`spec.passed_path`), only from a clean
checkout: the code it checked (`spec.executor_source_sha256`, which covers this module too), each way's largest
differences, when and from which commit — what `spec.require_conforming_executor` asks for before a spec is opened.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import platform
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from geokit import METRES_PER_DEG_LAT
from ts_transformer.autopilot import replay, single
from ts_transformer.autopilot.executor import LIMITS, MODES as EXECUTOR_MODES, Executor, Flown
from ts_transformer.autopilot.flights import FlightInputs
from ts_transformer.autopilot.frame import ALT, LAT, LON, PSI, AirportCharts
from ts_transformer.autopilot.judge import Verdict, judge
from ts_transformer.autopilot.lateral import Runways
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.autopilot.sentence import Sentences
from ts_transformer.autopilot.spec import (
    CONFORMANCE_DIRECTORY as DIRECTORY, PASSED_SCHEMA, executor_source_sha256, passed_path, reference_sha256,
)
from ts_transformer.instructions.words import Words
from ts_transformer.io_utils import logic, utc_now, write_json_atomic
from ts_transformer.repo_layout import git_state, repo_relative

#: v2 (two-tier v4): the flights' runway in force per cycle, the five-column words, the go-around's time reserve.
#: v3 (A19, A20): no word clock's observed rows in a flight's input digest; the airport elevation E in it (D58).
REFERENCE_SCHEMA = "ts-executor-conformance-reference-v3"
#: The reference's flights: the replay gate's draw on the training days, every airport alike.
SPLIT, PER_AIRPORT, SEED, GROUPS = "train", 50, 1337, (replay.OWN,)
#: How far apart two flown states may be, metres, horizontally or vertically — the single-flight executor's bound
#: against the batched one (`aeroviz_backend.autopilot_segment.check_single`, 7,426 segments within 1.6e-8 m).
STATE_BOUND_M = 1e-6
#: How far apart any other two floats may be (speed m/s, angles rad, mass kg, commands, wanted rates, sentence times,
#: a verdict's check numbers).
ROUNDOFF = 1e-6
DEVICE = torch.device("cpu")


@dataclasses.dataclass(frozen=True)
class FlightResult:
    """One flight as flown, up to the cycle it was done at (``done``): ``states`` ``[done + 2, 7]``, the per-cycle
    arrays ``[done + 1, …]``, and its verdict as JSON (`verdict_json`)."""

    states: np.ndarray
    commands: np.ndarray
    wanted: np.ndarray
    sentence_s: np.ndarray
    runway: np.ndarray
    limits: dict[str, np.ndarray]
    modes: dict[str, np.ndarray]
    done: int
    verdict: dict[str, Any]


def jsonable(value: Any) -> Any:
    """``value`` with numpy scalars, tuples and arrays as JSON's numbers and lists."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def verdict_json(verdict: Verdict) -> dict[str, Any]:
    return jsonable(dataclasses.asdict(verdict))


def flight_results(flown: Flown, verdicts: Sequence[Verdict]) -> list[FlightResult]:
    """Each flight of a batch, cut at the cycle it was done at."""
    out = []
    for j, verdict in enumerate(verdicts):
        done = int(flown.done_cycle[j])
        out.append(FlightResult(
            states=flown.states[j, :done + 2].cpu().numpy(), commands=flown.commands[j, :done + 1].cpu().numpy(),
            wanted=flown.wanted[j, :done + 1].cpu().numpy(), sentence_s=flown.sentence_s[j, :done + 1].cpu().numpy(),
            runway=flown.runway[j, :done + 1].cpu().numpy(),
            limits={name: flown.limits[name][j, :done + 1].cpu().numpy() for name in LIMITS},
            modes={name: flown.modes[name][j, :done + 1].cpu().numpy() for name in EXECUTOR_MODES},
            done=done, verdict=verdict_json(verdict)))
    return out


def fly_batch(batch: replay.Batch, params: ExecutorParams, words: Words) -> list[FlightResult]:
    """Single-aircraft batch: every flight from its row 0, one executor (`replay.fly_batch`)."""
    flown, verdicts = replay.fly_batch(batch, params, words, device=DEVICE)
    return flight_results(flown, verdicts)


#: A multi-aircraft batch's starts: each flight from a seeded step in ``[0, STAGGER_STEPS]`` of the batch.
STAGGER_STEPS = 30


def fly_staggered(batch: replay.Batch, params: ExecutorParams, words: Words) -> list[FlightResult]:
    """Multi-aircraft batch: the same flights in one executor, each from its own seeded start
    (`Executor`'s ``start_cycle``), its words on its own rows from its own first cycle (`replay.fly_sentences`'s
    limits), each halted (`Executor.halt`) from the cycle after it is done while the others fly on. (The
    words said a step at a time to such a batch, `Spoken`'s ``start_step``, are the window loop's: checked there, each
    aircraft against the same flight flown alone — `tests/test_traffic_window.py`.)"""
    f64 = torch.float64
    step_rows = int(round(batch.row_interval_s / params.cycle_s))
    starts = np.random.default_rng(SEED).integers(0, STAGGER_STEPS + 1, size=len(batch.sentences)) * step_rows
    limits = torch.tensor(replay.time_limits_s(batch, params, words.spec.step_s), dtype=f64, device=DEVICE)
    sentences = Sentences([s.grid for s in batch.sentences], words, step_s=batch.row_interval_s, device=DEVICE)
    executor = Executor(batch.inputs(DEVICE), Runways.of(batch.geometries, words.spec, dtype=f64, device=DEVICE),
                        AirportCharts.of(batch.geometries, dtype=f64, device=DEVICE),
                        torch.tensor(batch.approach_ias_mps, dtype=f64, device=DEVICE), params, words,
                        step_s=batch.row_interval_s, time_limit_s=limits, start_cycle=torch.as_tensor(starts, device=DEVICE),
                        reserve_s=replay.reserve_s(batch))
    for _ in range(executor.cycles):
        sentence_s = executor.own_cycle().clamp(min=0).to(f64) * params.cycle_s
        executor.cycle(sentences.at(sentence_s), sentence_s)
        if bool(executor.done.all()):
            break
        executor.halt(executor.done)          # held from the cycle after it is done, as the window loop halts a cohort
    flown = executor.flown()
    return flight_results(flown, replay.judge_batch(batch, flown, words))


def fly_single(batch: replay.Batch, params: ExecutorParams, words: Words) -> list[FlightResult]:
    """Single flight: each flight alone in the single-flight executor (`autopilot.single`), driven as `executor.fly`
    drives a batch — a step's words heard on the cycle that starts it."""
    spec = words.spec
    inputs = batch.inputs(DEVICE)
    limits, reserve = replay.time_limits_s(batch, params, words.spec.step_s), replay.reserve_s(batch)
    out = []
    for j, sentence_flown in enumerate(batch.sentences):
        flight = FlightInputs(**{f.name: getattr(inputs, f.name)[j: j + 1] for f in dataclasses.fields(FlightInputs)})
        executor = single.SingleExecutor(flight, batch.geometries[j], batch.approach_ias_mps[j], params, words,
                                         step_s=batch.row_interval_s, time_limit_s=limits[j], reserve_s=reserve)
        sentence = single.Sentence(sentence_flown.grid, words, step_s=batch.row_interval_s)
        for cycle in range(executor.cycles):
            sentence_s = cycle * params.cycle_s
            executor.cycle(sentence.at(sentence_s), sentence_s)
            if executor.done:
                break
        flown = executor.flown()
        verdict = judge(flown, 0, batch.geometries[j], sentence_flown.instructions,
                        batch.row_interval_s, batch.signals[j], spec, words)
        out += flight_results(flown, [verdict])
    return out


#: Every way the executor flies, each checked against the reference (executor design §12.4).
MODES: dict[str, Callable[[replay.Batch, ExecutorParams, Words], list[FlightResult]]] = {
    "batch": fly_batch, "staggered": fly_staggered, "single": fly_single}


# ---- the reference on disk

def _stack(rows: Sequence[np.ndarray], fill: Any) -> np.ndarray:
    """Rows of unequal length padded with ``fill`` into one array."""
    length = max(len(r) for r in rows)
    out = np.full((len(rows), length, *rows[0].shape[1:]), fill, dtype=rows[0].dtype)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


def save_results(path: Path, results: Sequence[FlightResult]) -> None:
    arrays = {"done": np.array([r.done for r in results], dtype=np.int64),
              "states": _stack([r.states for r in results], np.nan),
              "commands": _stack([r.commands for r in results], np.nan),
              "wanted": _stack([r.wanted for r in results], np.nan),
              "sentence_s": _stack([r.sentence_s for r in results], np.nan),
              "runway": _stack([r.runway for r in results], -1)}
    for name in LIMITS:
        arrays[f"limit_{name}"] = _stack([r.limits[name] for r in results], False)
    for name in EXECUTOR_MODES:
        arrays[f"mode_{name}"] = _stack([r.modes[name] for r in results], False)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)


def load_results(path: Path, verdicts: Sequence[dict[str, Any]]) -> list[FlightResult]:
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    keys = {"done", "states", "commands", "wanted", "sentence_s", "runway", *(f"limit_{n}" for n in LIMITS),
            *(f"mode_{n}" for n in EXECUTOR_MODES)}
    if set(arrays) != keys or any(len(v) != len(verdicts) for v in arrays.values()):
        raise ValueError(f"{path} does not hold one row of every limit and mode for each of its {len(verdicts)} flights")
    out = []
    for j, verdict in enumerate(verdicts):
        done = int(arrays["done"][j])
        out.append(FlightResult(
            states=arrays["states"][j, :done + 2], commands=arrays["commands"][j, :done + 1],
            wanted=arrays["wanted"][j, :done + 1], sentence_s=arrays["sentence_s"][j, :done + 1],
            runway=arrays["runway"][j, :done + 1],
            limits={name: arrays[f"limit_{name}"][j, :done + 1] for name in LIMITS},
            modes={name: arrays[f"mode_{name}"][j, :done + 1] for name in EXECUTOR_MODES},
            done=done, verdict=verdict))
    return out


# ---- comparing a flight with its reference

def checker_sha256() -> str:
    """This module's logic (`spec.logic`): the checker a reference and a passed record were made by."""
    return hashlib.sha256(logic(Path(__file__).read_text(encoding="utf-8")).encode("utf-8")).hexdigest()


def bounds() -> dict[str, float]:
    return {"state_m": STATE_BOUND_M, "roundoff": ROUNDOFF}


@dataclasses.dataclass
class Difference:
    """How far one way of flying lies from the reference over its flights: the largest differences, and every flight
    that differs beyond the bounds (`mismatches`: flight → what). It passes only over every flight of the reference
    (``expected``)."""

    expected: int
    flights: int = 0
    horizontal_m: float = 0.0
    vertical_m: float = 0.0
    other: float = 0.0
    mismatches: dict[str, list[str]] = dataclasses.field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.mismatches and self.flights == self.expected

    def summary(self) -> dict[str, Any]:
        return {"expected": self.expected, "flights": self.flights, "horizontal_m": self.horizontal_m,
                "vertical_m": self.vertical_m, "other": self.other, "mismatched_flights": len(self.mismatches)}


def _max_abs(a: np.ndarray, b: np.ndarray, *, angle: bool = False) -> tuple[float, bool]:
    """The largest |a − b| over the entries finite in both (an angle's difference wrapped to ±π), and whether the two
    differ where either is not finite (a NaN against a number, an infinity against anything but itself)."""
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    infinite = np.isinf(a) | np.isinf(b)
    apart = bool((nan_a != nan_b).any() or (infinite & ~(nan_a | nan_b) & (a != b)).any())
    both = np.isfinite(a) & np.isfinite(b)
    gap = a[both] - b[both]
    if angle:
        gap = np.remainder(gap + math.pi, 2.0 * math.pi) - math.pi
    return (float(np.abs(gap).max()) if gap.size else 0.0), apart


def _verdict_differences(a: Any, b: Any, where: str, out: list[str]) -> None:
    """Every place two verdicts differ: floats beyond `ROUNDOFF`, anything else not equal."""
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(a.keys() | b.keys()):
            if key not in a or key not in b:
                out.append(f"verdict {where}.{key}: only in one")
            else:
                _verdict_differences(a[key], b[key], f"{where}.{key}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"verdict {where}: {len(a)} entries, {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            _verdict_differences(x, y, f"{where}[{i}]", out)
    elif isinstance(a, float) and isinstance(b, float):
        if not (a == b or (math.isnan(a) and math.isnan(b)) or abs(a - b) <= ROUNDOFF):
            out.append(f"verdict {where}: {a!r}, {b!r}")
    elif a != b or type(a) is not type(b):
        out.append(f"verdict {where}: {a!r}, {b!r}")


def compare(reference: FlightResult, flown: FlightResult, difference: Difference, name: str) -> None:
    """Add flight ``name``'s comparison to ``difference``."""
    problems: list[str] = []
    for result in (reference, flown):
        rows = {"states": len(result.states),
                **{f: len(getattr(result, f)) for f in ("commands", "wanted", "sentence_s", "runway")},
                **{f"limit {k}": len(v) for k, v in result.limits.items()},
                **{f"mode {k}": len(v) for k, v in result.modes.items()}}
        wrong = {k: n for k, n in rows.items() if n != result.done + (2 if k == "states" else 1)}
        if wrong or set(result.limits) != set(LIMITS) or set(result.modes) != set(EXECUTOR_MODES):
            raise ValueError(f"{name}: a flight result not cut at its done cycle {result.done}, or without every limit "
                             f"and mode ({wrong})")
    if flown.done != reference.done:
        problems.append(f"done at cycle {flown.done}, the reference at {reference.done}")
    cycles = min(flown.done, reference.done) + 1
    a, b = reference.states[:cycles + 1], flown.states[:cycles + 1]
    lat_m, apart_lat = _max_abs(a[:, LAT] * METRES_PER_DEG_LAT, b[:, LAT] * METRES_PER_DEG_LAT)
    scale = METRES_PER_DEG_LAT * np.cos(np.radians(np.where(np.isfinite(a[:, LAT]), a[:, LAT], 0.0)))
    lon_m, apart_lon = _max_abs(a[:, LON] * scale, b[:, LON] * scale)
    horizontal_m = math.hypot(lat_m, lon_m)
    vertical_m, apart_alt = _max_abs(a[:, ALT], b[:, ALT])
    others = [_max_abs(a[:, c], b[:, c], angle=c == PSI) for c in range(a.shape[1]) if c not in (LAT, LON, ALT)]
    for field in ("commands", "wanted", "sentence_s"):
        others.append(_max_abs(getattr(reference, field)[:cycles], getattr(flown, field)[:cycles]))
    other = max(gap for gap, _ in others)
    if apart_lat or apart_lon or apart_alt or any(apart for _, apart in others):
        problems.append("a NaN or an infinity where the reference has another value")
    if horizontal_m > STATE_BOUND_M or vertical_m > STATE_BOUND_M:
        problems.append(f"states {horizontal_m:.3g} m apart horizontally, {vertical_m:.3g} m vertically")
    if other > ROUNDOFF:
        problems.append(f"a float {other:.3g} apart")
    apart = np.flatnonzero(flown.runway[:cycles] != reference.runway[:cycles])
    if len(apart):
        problems.append(f"the runway in force differs at cycle {int(apart[0])} ({len(apart)} cycles)")
    for kind, ours, theirs in (("limit", flown.limits, reference.limits), ("mode", flown.modes, reference.modes)):
        for key in theirs:
            apart = np.flatnonzero(ours[key][:cycles] != theirs[key][:cycles])
            if len(apart):
                problems.append(f"{kind} {key} differs at cycle {int(apart[0])} ({len(apart)} cycles)")
    _verdict_differences(reference.verdict, flown.verdict, "", problems)
    difference.flights += 1
    difference.horizontal_m = max(difference.horizontal_m, horizontal_m)
    difference.vertical_m = max(difference.vertical_m, vertical_m)
    difference.other = max(difference.other, other)
    if problems:
        difference.mismatches[name] = problems


# ---- what each flight is flown from

def input_digests(batch: replay.Batch, params: ExecutorParams, words: Words) -> list[str]:
    """Each flight's inputs as one sha256 (module docstring): what `replay.fly_sentences` flies it from, and what the
    judge reads it against."""
    inputs = batch.inputs(DEVICE)
    limits, reserve = replay.time_limits_s(batch, params, words.spec.step_s), replay.reserve_s(batch)
    out = []
    for j, sentence in enumerate(batch.sentences):
        geometry = batch.geometries[j]
        parts = [inputs.initial_state[j], inputs.aero_params[j], inputs.frame_params[j], inputs.max_thrust_n[j],
                 np.array([batch.approach_ias_mps[j], limits[j], reserve, batch.readings[j].runway_index,
                           batch.row_interval_s, sentence.first_row]),
                 np.asarray(sentence.grid),
                 np.array([[c.threshold_e_m, c.threshold_n_m, c.course_deg, c.elevation_m] for c in geometry.candidates]),
                 np.array([geometry.elevation_m]),
                 np.array([[float(getattr(path, f.name)) for f in dataclasses.fields(path)]
                           for path in (c.vertical_path for c in geometry.candidates)], dtype=np.float64)]
        digest = hashlib.sha256()
        for part in parts:
            array = np.ascontiguousarray(part.cpu().numpy() if isinstance(part, torch.Tensor) else part)
            digest.update(str(array.dtype).encode() + str(array.shape).encode() + array.tobytes())
        out.append(digest.hexdigest())
    return out


# ---- the reference and the check

def spec_identity(executor_dir: Path, record: Mapping[str, Any], instructions: Path) -> dict[str, Any]:
    return {"executor": repo_relative(executor_dir), "spec_sha256": record["sha256"],
            "instructions": repo_relative(instructions), "vocabulary_spec_sha256": record["vocabulary_spec_sha256"]}


def draw_reference_batch(instructions: Path, words: Words, draw: Mapping[str, Any]) -> replay.Batch:
    return replay.draw(instructions, draw["split"], words.spec, words, per_airport=draw["per_airport"],
                       seed=draw["seed"], groups=tuple(draw["groups"]), row_interval_s=draw["row_interval_s"])


def keys_of(batch: replay.Batch) -> list[str]:
    return [s.dataset_id for s in batch.signals]


def write_reference(executor_dir: Path, instructions: Path, *, batch: replay.Batch | None = None) -> Path:
    """Fly the reference with the code on disk and write it into ``<executor_dir>/conformance/`` (a new directory).
    Only from a clean checkout, and only with the code that measured the spec (its `executor_source_sha256`)."""
    params, record, words = replay.open_spec(executor_dir, instructions)
    git = git_state()
    if git["dirty"]:
        raise RuntimeError("the reference is written from a clean checkout only (it records the commit)")
    if record["source"]["executor_source_sha256"] != executor_source_sha256():
        raise ValueError(f"{executor_dir.name} was measured by other executor code "
                         f"({record['source']['executor_source_sha256'][:12]}, now {executor_source_sha256()[:12]}): its "
                         f"reference is flown by the code that measured it")
    directory = executor_dir / DIRECTORY
    if directory.exists():
        raise FileExistsError(f"{directory} exists: a spec's reference is written once")
    draw = {"split": SPLIT, "per_airport": PER_AIRPORT, "seed": SEED, "groups": list(GROUPS),
            "row_interval_s": words.spec.step_s}
    batch = draw_reference_batch(instructions, words, draw) if batch is None else batch
    results = MODES["batch"](batch, params, words)
    payload = {"schema": REFERENCE_SCHEMA, "written_utc": utc_now(), "git": git, "python": platform.python_version(),
               **spec_identity(executor_dir, record, instructions), "draw": draw, "drawn": batch.drawn,
               "flown_by": {"executor_source_sha256": executor_source_sha256(), "mode": "batch"},
               "checker_sha256": checker_sha256(), "bounds": bounds(), "flights": keys_of(batch),
               "inputs": input_digests(batch, params, words), "verdicts": [r.verdict for r in results]}
    text = json.dumps(payload, indent=2, allow_nan=True)
    staging = executor_dir / f".{DIRECTORY}.writing-{os.getpid()}"
    staging.mkdir()
    save_results(staging / "reference.npz", results)
    (staging / "reference.json").write_text(text, encoding="utf-8")
    staging.rename(directory)
    return directory


@dataclasses.dataclass(frozen=True)
class Checked:
    """A check's differences, way by way, and the executor code and commit it flew — taken before it flew."""

    differences: dict[str, Difference]
    executor_source_sha256: str
    git: dict[str, Any]


def check(executor_dir: Path, instructions: Path, *, modes: Sequence[str] | None = None,
          batch: replay.Batch | None = None) -> Checked:
    """Fly the reference's flights in every way of ``modes`` (every one of `MODES` by default) and compare each with the
    reference."""
    code, git = executor_source_sha256(), git_state()
    params, record, words = replay.open_spec(executor_dir, instructions)
    directory = executor_dir / DIRECTORY
    payload = json.loads((directory / "reference.json").read_text(encoding="utf-8"))
    if payload["schema"] != REFERENCE_SCHEMA:
        raise ValueError(f"{directory / 'reference.json'} is {payload['schema']!r}, this code reads {REFERENCE_SCHEMA!r}")
    identity = spec_identity(executor_dir, record, instructions)
    if {k: payload[k] for k in identity} != identity:
        raise ValueError(f"the reference was flown for {({k: payload[k] for k in identity})}, not {identity}")
    if payload["bounds"] != bounds():
        raise ValueError(f"the reference was made under the bounds {payload['bounds']}, this checker's are {bounds()}")
    batch = draw_reference_batch(instructions, words, payload["draw"]) if batch is None else batch
    if keys_of(batch) != payload["flights"]:
        raise ValueError("the reference's draw no longer gives its flights (the data under it moved)")
    moved = [name for name, ours, theirs in zip(payload["flights"], input_digests(batch, params, words),
                                                payload["inputs"], strict=True) if ours != theirs]
    if moved:
        raise ValueError(f"the inputs of {len(moved)} reference flights moved (the data under the reference changed, not "
                         f"the executor): {moved[:5]}")
    reference = load_results(directory / "reference.npz", payload["verdicts"])
    out = {}
    for mode in (tuple(MODES) if modes is None else modes):
        difference = Difference(expected=len(payload["flights"]))
        flown = MODES[mode](batch, params, words)
        if len(flown) != len(reference):
            raise ValueError(f"{mode} flew {len(flown)} of the reference's {len(reference)} flights")
        for name, ref, result in zip(payload["flights"], reference, flown, strict=True):
            compare(ref, result, difference, name)
        out[mode] = difference
    return Checked(out, code, git)


def write_passed(executor_dir: Path, checked: Checked) -> Path:
    """The record that the code on disk flies the reference within the bounds in every way (refused otherwise, and
    when the code or the commit is not the one the check flew)."""
    differences = checked.differences
    if set(differences) != set(MODES) or not all(d.passed for d in differences.values()):
        raise ValueError("a passed record needs every way of flying checked and within the bounds")
    git, code = git_state(), executor_source_sha256()
    if checked.git["dirty"] or git != checked.git or code != checked.executor_source_sha256:
        raise RuntimeError("a passed record is written from a clean checkout only, for the code the check flew (the "
                           "code or the commit changed while it flew)")
    path = passed_path(executor_dir, code)
    write_json_atomic(path, {"schema": PASSED_SCHEMA, "written_utc": utc_now(), "git": git,
                             "python": platform.python_version(), "executor_source_sha256": code,
                             "checker_sha256": checker_sha256(),
                             "reference_sha256": reference_sha256(executor_dir / DIRECTORY), "bounds": bounds(),
                             "modes": {mode: d.summary() for mode, d in differences.items()}})
    return path
