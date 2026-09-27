"""Heading-lead ablation, first level (two-tier progress report 2026-09-27, P0 item 1): the heading lead L, the bank
limit and the roll rate moved over ONE train sample, relabelled and replayed by the current executor — no model trained,
no formal artefact written.

Why: L = 4 s (the vocabulary's `heading_lead_s`) was set on 2026-09-24 under the old first-order heading law
(vocabulary design §10.1); the law now arrives on each heading word exactly L after it is heard, and method A derives
τ_ψ = L and p = bank limit ÷ L (`autopilot/derive.py`), so L also sets how fast the executor rolls — 8°/s, the fast end
of every source (repo `docs/literature/roll_rate_and_turn_response/`: procedure design assumes 3–5°/s, PANS-OPS a 25°
bank).

A cell is (L, bank limit φ, roll rate p). Its vocabulary is the formal spec's (``--instructions``) with `heading_lead_s`
= L and `turn_bank_max_deg` = φ, nothing else changed; its executor is the formal spec's parameters (``--executor``,
which must be the current executor code's) with τ_ψ = L and p = φ ÷ L (``derived``: method A, `derive`) or a given p.
A given p breaks method A's p·τ_ψ = φ, on which the executor's own turns rely (`lateral.rate_for_error`: its intercept,
a go-around, the line) — they may overshoot, and that is part of what such a cell measures. Every cell re-reads the SAME
sample with the labeller under its own vocabulary (a flight the labeller refuses under it is counted, not flown) and
flies it (`executor_replay.fly_airport`: judged, written as records, graded by evaluation and paired with the observed
flights' verdicts, graded once, here). The sample is the formal train replay gate's (`replay.draw`: own and stand-in
dynamics, ``--per-airport`` of each airport, seeded), and the cell with the formal values is flown first and must
reproduce that replay (``--reference-replay``) flight for flight in every field it stores but the two verdicts — or
nothing else is flown.

Each cell reports, per dynamics group (own: the gate's; stand-in: reported) and stratum (read off the track's turns
before the capture, which no cell moves, so a stratum holds the same flights in every cell): landed; words inside their
envelopes, judged under the cell's OWN vocabulary (a longer lead gives a heading word longer before it is judged, and
the capture turn is judged against the cell's bank limit, so this column compares neither across L nor across the bank
limit — landed, evaluation and the distances do); evaluation paired with the observed verdict; heading words per flight
after step 0; over the LANDED flights, the distance to the observed track — time-aligned over the sentence rows both
reach (the mean, `replay.flight_alignment`, and the largest) and time-free (the symmetric Hausdorff distance between the
observed track closed to the threshold it landed on and the flown track, both read as lines, `DENSIFY_M`) — and the
landing time against the observed; the share of cycles the bank limit and the roll rate bound; the airport × stratum
cells that clear the three gates (`executor_replay.gate_table`); the labeller's refusals; and, against the reference
cell on the flights both flew, how many changed outcome or evaluation verdict each way. A non-finite distance (a
dynamics failure's last state) is counted apart, never averaged.

Writes into ``--out`` (a new directory; ``--resume`` continues one written by the same plan, commit and libraries — a
cell cut off mid-way is moved aside as ``<cell>.aborted-<UTC>`` and flown again): ``plan.json``; ``observed/<ICAO>/``
(the observed records linked read-only, and their evaluation report); per cell ``cells/<nn>_<cell>/`` holding
``cell.json`` (every flight's row) and ``evaluation/<ICAO>.json`` — the flown records themselves are dropped once the
cell is graded unless ``--keep-records``; then ``ablation.json`` and ``ablation.md``. Train only, from a clean tree (checked
again before every cell), on the CPU (the reference must reproduce a CPU replay bit for bit).

    python run_ts.py heading_lead_ablation \\
        --instructions 4dTrajectory/outputs/POOLED/instruction_language/v5_20260926 \\
        --executor 4dTrajectory/outputs/POOLED/executor/v10_20260926 \\
        --reference-replay 4dTrajectory/outputs/POOLED/executor/v10_20260926/replay-train/replay.json \\
        --out 4dTrajectory/outputs/POOLED/analyses/heading_lead_ablation_20260927
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import torch
from scipy.spatial import cKDTree

from ts_transformer.autopilot import derive, replay
from ts_transformer.autopilot.executor import Flown
from ts_transformer.autopilot.inverse import BANK_MAX_RAD
from ts_transformer.autopilot.judge import CROSSINGS, Verdict, flown_track
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.autopilot.spec import params_sha256
from ts_transformer.experiments.executor_replay import (
    REPLAY_SCHEMA, fly_airport, gate_table, observed_verdicts, require_same_grading,
)
from ts_transformer.instructions.labeller.read import Reading, read_flight
from ts_transformer.instructions.labeller.records import Refused
from ts_transformer.instructions.readout import STRATA, flight_record
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.instructions.words import Words
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.repo_layout import REPO_ROOT, git_state

ABLATION_SCHEMA = "ts-heading-lead-ablation-v1"
SPLIT = "train"
CPU = torch.device("cpu")
#: p = the bank limit over the lead (method A, `derive.roll_rate_deg_s`).
DERIVED = "derived"
#: The grid proposed in the progress report (§3 item 1, the user's choice 2026-09-27): L on the 2 s step, the bank limit
#: at PANS-OPS's 25° and the vocabulary's measured 32°, p derived or at the sources' 3 and 5°/s.
LEADS_S = (2.0, 4.0, 6.0, 8.0)
BANK_LIMITS_DEG = (25.0, 32.0)
BANK_RATES = (DERIVED, "3", "5")
#: A stored replay row's fields the reference cell is not held to: the evaluation's verdicts, compared and reported
#: apart (the evaluation code may have moved since the formal replay).
VERDICTS = ("replay_verdict", "observed_verdict")
#: The fields this runner adds to a replay row.
ABLATION_FIELDS = ("max_horizontal_distance_m", "hausdorff_horizontal_distance_m", "heading_words")
#: The time-free distance reads both tracks as lines, with points no farther apart than this (so within half of it of
#: the true point-to-line distance).
DENSIFY_M = 5.0
#: The limits whose bound cycles the table reads: the vocabulary's bank limit and the roll rate p.
BANK_LIMITS = ("bank_cap", "bank_rate")
GROUPS = (replay.OWN, replay.STAND_IN)


@dataclass(frozen=True)
class Cell:
    lead_s: float
    bank_limit_deg: float
    #: None: derived, the bank limit over the lead (method A).
    bank_rate_deg_s: float | None

    @property
    def name(self) -> str:
        rate = DERIVED if self.bank_rate_deg_s is None else f"{self.bank_rate_deg_s:g}"
        return f"L{self.lead_s:g}_bank{self.bank_limit_deg:g}_p{rate}"

    def vocabulary(self, base: VocabularySpec) -> VocabularySpec:
        """The formal vocabulary with this cell's lead and bank limit, nothing else changed."""
        return replace(base, heading_lead_s=self.lead_s, turn_bank_max_deg=self.bank_limit_deg)

    def executor(self, base: ExecutorParams, vocabulary: VocabularySpec) -> ExecutorParams:
        """The formal executor parameters with τ_ψ = the lead and this cell's roll rate, checked against the design's
        constraints under the cell's vocabulary, and the bank limit against the grader's (`inverse.attitude` checks it
        only when it is first flown)."""
        if not 0.0 < math.radians(self.bank_limit_deg) <= BANK_MAX_RAD:
            raise ValueError(f"bank limit {self.bank_limit_deg:g}° outside (0, {math.degrees(BANK_MAX_RAD):g}°]")
        tau = derive.heading_time_constant_s(vocabulary, base.cycle_s)      # refuses a lead under 2Δt first
        rate = derive.roll_rate_deg_s(vocabulary) if self.bank_rate_deg_s is None else self.bank_rate_deg_s
        params = replace(base, heading_time_constant_s=tau, bank_rate_deg_s=rate)
        params.check(vocabulary)
        return params


def positive(value: str) -> float:
    """An argument that must be a finite positive number (`ExecutorParams.check` lets a NaN through)."""
    number = float(value)
    if not (math.isfinite(number) and number > 0.0):
        raise argparse.ArgumentTypeError(f"{value} is not a finite positive number")
    return number


def parse_rate(value: str) -> float | None:
    return None if value == DERIVED else positive(value)


def cells_of(reference: Cell, leads: list[float], bank_limits: list[float], rates: list[float | None]) -> list[Cell]:
    """The reference first, then the grid (L × bank limit × p) without it, each once."""
    grid = [Cell(lead, bank, rate) for bank in bank_limits for rate in rates for lead in leads]
    return [reference, *dict.fromkeys(cell for cell in grid if cell != reference)]


def reference_cell(spec: VocabularySpec, params: ExecutorParams) -> Cell:
    """The formal values as a cell: the vocabulary's lead and bank limit, p derived — refused unless that IS the formal
    executor's p and τ_ψ (the formal spec was written by method A)."""
    cell = Cell(spec.heading_lead_s, spec.turn_bank_max_deg, None)
    if cell.executor(params, spec) != params:
        raise ValueError("the executor spec's τ_ψ and p are not method A's from its vocabulary: no cell reproduces it")
    return cell


def relabel(batch: replay.Batch, vocabulary: VocabularySpec,
            words: Words) -> tuple[list[int], list[Reading], list[tuple[int, str]]]:
    """Every flight of ``batch`` read under ``vocabulary``: the indices read, their readings, and each refused flight's
    index and reason."""
    keep, readings, refused = [], [], []
    for j, flight in enumerate(batch.signals):
        try:
            readings.append(read_flight(flight, batch.geometries[j], vocabulary, words))
        except Refused as refusal:
            refused.append((j, refusal.reason))
            continue
        keep.append(j)
    return keep, readings, refused


def densify(points: np.ndarray, spacing_m: float) -> np.ndarray:
    """A polyline ``[N, 2]`` as its first vertex and, along every segment, points no farther apart than ``spacing_m``
    ending on the segment's far vertex."""
    segments = np.diff(points, axis=0)
    counts = np.maximum(1, np.ceil(np.hypot(segments[:, 0], segments[:, 1]) / spacing_m).astype(np.int64))
    index = np.repeat(np.arange(len(segments)), counts)
    fraction = (np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts) + 1) / np.repeat(counts, counts)
    return np.concatenate((points[:1], points[index] + segments[index] * fraction[:, None]))


def track_gaps(batch: replay.Batch, flown: Flown, verdicts: list[Verdict]) -> list[dict[str, float]]:
    """Per flight, the largest horizontal distance between the flown and the observed track, two ways (module
    docstring): time-aligned at the sentence rows both reach, time-free as the symmetric Hausdorff distance between the
    observed rows closed to the pointed threshold (the flight landed just past its last row) and the flown track, both
    read as lines — a crossing's flown line ends at its interpolated crossing (`Verdict.crossing["at_row"]`), not at the
    first cycle past the plane, so neither line runs past the threshold. A dynamics failure is read to the cycle before
    it, as its record is (`executor_replay.executor_forecast`) — unlike `replay.flight_alignment`, which reads its failure
    row too."""
    out = []
    for j, verdict in enumerate(verdicts):
        observed, reading, geometry = batch.signals[j], batch.readings[j], batch.geometries[j]
        rows = len(reading.words)
        end = verdict.end_row - 1 if verdict.outcome == "dynamics_failure" else verdict.end_row
        track = flown_track(flown.states[j, : end + 1].cpu().numpy(), geometry)
        step_rows = int(round((observed.time_s[1] - observed.time_s[0]) / flown.cycle_s))
        both = min(rows, end // step_rows + 1)
        at = np.arange(both) * step_rows
        aligned = np.hypot(track["e"][at] - observed.e_m[:both], track["n"][at] - observed.n_m[:both])
        threshold = geometry.candidates[reading.runway_index]
        seen = densify(np.vstack((np.column_stack((observed.e_m[:rows], observed.n_m[:rows])),
                                  [[threshold.threshold_e_m, threshold.threshold_n_m]])), DENSIFY_M)
        line = np.column_stack((track["e"], track["n"]))
        if verdict.outcome in CROSSINGS:
            whole = int(math.floor(verdict.crossing["at_row"]))
            fraction = verdict.crossing["at_row"] - whole
            line = np.vstack((line[: whole + 1], line[whole] + fraction * (line[whole + 1] - line[whole])))
        path = densify(line, DENSIFY_M)
        hausdorff = max(cKDTree(path).query(seen)[0].max(), cKDTree(seen).query(path)[0].max())
        out.append({"max_horizontal_distance_m": float(aligned.max()),
                    "hausdorff_horizontal_distance_m": float(hausdorff)})
    return out


def canonical(value: Any) -> str:
    """A row field as JSON text: tuples read as lists and NaN equals NaN, as in a stored replay."""
    return json.dumps(value, sort_keys=True)


def reproduction(rows: list[dict[str, Any]], stored: list[dict[str, Any]]) -> dict[str, Any]:
    """The reference cell against the formal replay's rows: every field a stored row holds but the two verdicts must be
    equal (the flights that differ are listed, and must be none); the verdicts that differ are reported. Refused unless
    the two hold the same flights and a row here is a stored row's fields plus `ABLATION_FIELDS` — a field the replay
    gains later escapes no comparison."""
    fields = set(stored[0])
    if any(set(row) != fields for row in stored) or any(set(row) != fields | set(ABLATION_FIELDS) for row in rows):
        raise ValueError("the reference cell's rows and the formal replay's do not hold the same fields")
    by_id = {row["dataset_id"]: row for row in stored}
    if sorted(by_id) != sorted(row["dataset_id"] for row in rows):
        raise ValueError("the reference cell flew other flights than the formal replay")
    compared = sorted(fields - set(VERDICTS))
    differ = [row["dataset_id"] for row in rows
              if any(canonical(row[key]) != canonical(by_id[row["dataset_id"]][key]) for key in compared)]
    verdicts = {key: [row["dataset_id"] for row in rows if row[key] != by_id[row["dataset_id"]][key]] for key in VERDICTS}
    return {"flights": len(rows), "fields_compared": compared, "executor_fields_differ": differ,
            "evaluation_verdicts_differ": verdicts["replay_verdict"],
            "observed_verdicts_differ": verdicts["observed_verdict"]}


def _percentiles(values: list[float]) -> dict[str, float] | None:
    """Mean, p50 and p90 of the finite values, with how many were not finite; None when none is."""
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return None
    return {"n": int(len(finite)), "non_finite": int(len(array) - len(finite)), "mean": float(finite.mean()),
            "p50": float(np.percentile(finite, 50)), "p90": float(np.percentile(finite, 90))}


def _share(part: int, whole: int) -> float | None:
    return part / whole if whole else None


def summarise(rows: list[dict[str, Any]], refused: int) -> dict[str, Any]:
    """One group of flown rows (``refused``: the group's flights the cell's labeller refused): the replay's shares, the
    heading words, the distances and the landing time over the landed flights, the bank limits' bound cycles."""
    judged = [ok for row in rows if row["words"] is not None for _, ok in row["words"]]
    paired = [row for row in rows if row["observed_verdict"] == "pass"]
    landed = [row for row in rows if row["outcome"] == "landed"]
    cycles = sum(row["limits"]["cycles"]["cycles"] for row in rows)
    return {
        "flights": len(rows), "refused": refused,
        "landed": _share(len(landed), len(rows)),
        "flew_the_sentence": _share(sum(row["flew_the_sentence"] for row in rows), len(rows)),
        "words_judged": len(judged), "words_inside": _share(sum(judged), len(judged)),
        "observed_passes": len(paired),
        "replay_passes_where_observed_passes": _share(sum(row["replay_verdict"] == "pass" for row in paired), len(paired)),
        "outcomes": dict(Counter(row["outcome"] for row in rows).most_common()),
        "heading_words_per_flight": _percentiles([row["heading_words"] for row in rows]),
        **{name: _percentiles([row[name] for row in landed])
           for name in ("mean_horizontal_distance_m", "max_horizontal_distance_m", "hausdorff_horizontal_distance_m",
                        "landing_time_minus_observed_s")},
        "limit_bound_cycle_share": {name: _share(sum(row["limits"][name]["cycles"] for row in rows), cycles)
                                    for name in BANK_LIMITS},
    }


def in_stratum(row: dict[str, Any], stratum: str) -> bool:
    return stratum in ("all", row["stratum"])


def by_group(rows: list[dict[str, Any]], refused: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    """`summarise` per dynamics group and stratum (``all`` beside the strata)."""
    return {group: {stratum: summarise([row for row in rows if row["group"] == group and in_stratum(row, stratum)],
                                       sum(r["group"] == group and in_stratum(r, stratum) for r in refused))
                    for stratum in ("all", *STRATA)}
            for group in GROUPS}


def paired_changes(rows: list[dict[str, Any]], reference: list[dict[str, Any]], stratum: str) -> dict[str, Any]:
    """Against the reference cell, own dynamics, on the stratum's flights both flew: landed lost / gained, and — where
    the observed flight passes evaluation — the replay's pass lost / gained."""
    before = {row["dataset_id"]: row for row in reference if row["group"] == replay.OWN and in_stratum(row, stratum)}
    pairs = [(before[row["dataset_id"]], row) for row in rows if row["dataset_id"] in before]
    landed = [(a["outcome"] == "landed", b["outcome"] == "landed") for a, b in pairs]
    passed = [(a["replay_verdict"] == "pass", b["replay_verdict"] == "pass") for a, b in pairs
              if a["observed_verdict"] == "pass"]
    return {"flights": len(pairs),
            "landed_lost": sum(a and not b for a, b in landed), "landed_gained": sum(b and not a for a, b in landed),
            "evaluation_pairs": len(passed),
            "evaluation_lost": sum(a and not b for a, b in passed),
            "evaluation_gained": sum(b and not a for a, b in passed)}


def gates_cleared(gates: dict[str, Any], stratum: str) -> dict[str, int]:
    """Own dynamics, the airport × stratum cells of ``stratum`` (every one for ``all``): how many, and how many clear
    all three gates."""
    cells = [cell for airport, strata in gates[replay.OWN].items() if airport != "all"
             for name, cell in strata.items() if name != "all" and stratum in ("all", name)]
    return {"cells": len(cells), "clear": sum(all(cell["clears"].values()) for cell in cells)}


def require_plan_commit(plan: dict[str, Any]) -> None:
    """The tree is still clean at the plan's commit: every cell of one ablation is flown and graded by the same code."""
    if git_state() != {"head": plan["commit"], "dirty": False}:
        raise SystemExit(f"the tree is no longer clean at {plan['commit'][:12]}; resume from a clean tree at it")


def fly_cell(batch: replay.Batch, cell: Cell, number: int, base: VocabularySpec, base_params: ExecutorParams, *,
             chunk: int, directory: Path, observed_reports: dict[str, dict[str, Any]],
             keep_records: bool) -> dict[str, Any]:
    """Relabel, fly, judge and grade one cell into ``directory``; returns what its ``cell.json`` holds (written by
    `main`, once the tree is checked to be still the plan's)."""
    started = time.perf_counter()
    vocabulary = cell.vocabulary(base)
    words = Words(vocabulary)
    params = cell.executor(base_params, vocabulary)
    keep, readings, refused = relabel(batch, vocabulary, words)
    flown_batch = replace(replay.subset(batch, keep), readings=readings)
    by_airport: dict[str, list[int]] = defaultdict(list)
    for j, flight in enumerate(flown_batch.signals):
        by_airport[flight.airport].append(j)
    rows: list[dict[str, Any]] = []
    for airport, members in sorted(by_airport.items()):
        airport_rows, graded = fly_airport(
            flown_batch, members, params, words, chunk=chunk, device=CPU, records=directory / "records" / airport,
            split=SPLIT, checkpoint=f"heading_lead_ablation:{cell.name}",
            extra_summary={"executor_params_sha256": params_sha256(params), "vocabulary_spec_sha256": vocabulary.sha256},
            per_flight=track_gaps)
        require_same_grading(graded, observed_reports[airport], airport)
        observed = {row["flight_key"]: row["verdict"] for row in observed_reports[airport]["trajectories"]}
        for row in airport_rows:
            row["observed_verdict"] = observed[row["flight_key"]]
        rows += airport_rows
        write_json_atomic(directory / "evaluation" / f"{airport}.json", graded)
    if not keep_records:
        shutil.rmtree(directory / "records")
    heading_words = {reading.dataset_id: flight_record(reading)["words_after_step0"]["heading"] for reading in readings}
    for row in rows:
        row["heading_words"] = heading_words[row["dataset_id"]]
    result = {"number": number, "cell": asdict(cell), "name": cell.name, "vocabulary_spec_sha256": vocabulary.sha256,
              "params": asdict(params), "params_sha256": params_sha256(params), "labelled": len(keep),
              # a refused flight's stratum and group are its formal reading's and its draw's
              "refused": [{"dataset_id": batch.readings[j].dataset_id, "reason": reason, "group": batch.groups[j],
                           "stratum": flight_record(batch.readings[j])["stratum"]} for j, reason in refused],
              "records": "kept" if keep_records else "dropped once graded",
              "gates": gate_table(rows), "rows": rows, "elapsed_s": time.perf_counter() - started}
    return result


def render(result: dict[str, Any]) -> str:
    def pct(value: float | None) -> str:
        return "—" if value is None else f"{100 * value:.1f}%"

    def km(p: dict[str, float] | None) -> str:
        return "—" if p is None else f"{p['p50'] / 1000:.2f}/{p['p90'] / 1000:.2f}"

    def mean(p: dict[str, float] | None, spec: str) -> str:
        return "—" if p is None else format(p["mean"], spec)

    lines = [f"heading-lead ablation, {SPLIT}: {result['sample']['flights']} flights drawn "
             f"({result['sample']['by_group']}); own dynamics below (the gate's group)",
             "words inside are judged under each cell's own vocabulary: they compare neither across L nor across the "
             "bank limit", "distances (p50/p90, km) and the landing time are over the landed flights", ""]
    head = (f"{'cell':22s} {'τψ':>3s} {'p':>5s} {'n':>5s} {'landed':>7s} {'eval':>6s} {'words':>6s} {'gates':>6s} "
            f"{'hdg/flt':>7s} {'mean km':>10s} {'max km':>10s} {'Hausd. km':>10s} {'Δt mean':>7s} "
            f"{'bank-lim':>8s} {'p-lim':>6s} {'refused':>7s} {'landed −/+':>10s} {'eval −/+':>8s}")
    for stratum in ("all", *STRATA):
        lines += [f"[{stratum}]", head]
        for name, cell in result["cells"].items():
            s = cell["summary"][replay.OWN][stratum]
            params, limits = cell["params"], s["limit_bound_cycle_share"]
            changes, cleared = cell["against_reference"][stratum], cell["gates_cleared"][stratum]
            lines.append(
                f"{name:22s} {params['heading_time_constant_s']:3g} {params['bank_rate_deg_s']:5.2f} {s['flights']:5d} "
                f"{pct(s['landed']):>7s} {pct(s['replay_passes_where_observed_passes']):>6s} "
                f"{pct(s['words_inside']):>6s} {cleared['clear']:>2d}/{cleared['cells']:<3d} "
                f"{mean(s['heading_words_per_flight'], '.1f'):>7s} {km(s['mean_horizontal_distance_m']):>10s} "
                f"{km(s['max_horizontal_distance_m']):>10s} {km(s['hausdorff_horizontal_distance_m']):>10s} "
                f"{mean(s['landing_time_minus_observed_s'], '+.1f'):>7s} "
                f"{pct(limits['bank_cap']):>8s} {pct(limits['bank_rate']):>6s} {s['refused']:7d} "
                f"{changes['landed_lost']:>4d}/{changes['landed_gained']:<5d} "
                f"{changes['evaluation_lost']:>3d}/{changes['evaluation_gained']:<4d}")
        lines.append("")
    reproduced = result["reference_reproduction"]
    lines.append(f"reference {result['reference']}: reproduces the formal replay on {reproduced['flights']} flights "
                 f"(evaluation verdicts that differ: {len(reproduced['evaluation_verdicts_differ'])} replayed, "
                 f"{len(reproduced['observed_verdicts_differ'])} observed)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--instructions", type=Path, required=True, help="the formal instruction artefact")
    parser.add_argument("--executor", type=Path, required=True, help="the formal executor spec directory")
    parser.add_argument("--reference-replay", type=Path, required=True,
                        help="the formal train replay (replay.json) the reference cell must reproduce")
    parser.add_argument("--per-airport", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--chunk", type=int, default=500)
    parser.add_argument("--leads", type=positive, nargs="+", default=list(LEADS_S), help="heading leads L, seconds")
    parser.add_argument("--bank-limits", type=positive, nargs="+", default=list(BANK_LIMITS_DEG), help="degrees")
    parser.add_argument("--bank-rates", type=parse_rate, nargs="+", default=[parse_rate(r) for r in BANK_RATES],
                        help=f"roll rates p, deg/s, or {DERIVED!r} (the bank limit over the lead)")
    parser.add_argument("--keep-records", action="store_true", help="keep every cell's flown records")
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    parser.add_argument("--resume", action="store_true", help="continue --out, written by the same plan and commit")
    args = parser.parse_args(argv)
    resolve = lambda path: path if path.is_absolute() else REPO_ROOT / path  # noqa: E731
    instructions, executor, out = resolve(args.instructions), resolve(args.executor), resolve(args.out)
    if out.exists() and not args.resume:
        parser.error(f"{out} exists; an ablation is never overwritten (--resume continues it)")
    git = git_state()
    if git["dirty"]:
        parser.error("the tree has uncommitted changes; the ablation records the commit it ran at")
    params, record, words = replay.open_executor(executor, instructions)
    spec = words.spec
    stored = json.loads(resolve(args.reference_replay).read_text(encoding="utf-8"))
    if (stored["schema"], stored["split"], stored["executor_spec_sha256"], stored["vocabulary_spec_sha256"]) != (
            REPLAY_SCHEMA, SPLIT, record["sha256"], spec.sha256):
        parser.error(f"{args.reference_replay} is not this executor's {SPLIT} replay of this vocabulary")
    reference = reference_cell(spec, params)
    cells = cells_of(reference, args.leads, args.bank_limits, args.bank_rates)
    for cell in cells:
        cell.executor(params, cell.vocabulary(spec))          # every cell is flyable before anything is flown
    plan = {"schema": ABLATION_SCHEMA, "split": SPLIT, "instructions": str(args.instructions),
            "executor": str(args.executor), "executor_spec_sha256": record["sha256"],
            "vocabulary_spec_sha256": spec.sha256, "reference_replay": str(args.reference_replay),
            "per_airport": args.per_airport, "seed": args.seed, "chunk": args.chunk,
            "cells": [cell.name for cell in cells], "keep_records": args.keep_records, "commit": git["head"],
            "libraries": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                          "scipy": scipy.__version__}}
    if args.resume and out.exists():
        written = json.loads((out / "plan.json").read_text(encoding="utf-8"))
        if written != plan:
            parser.error(f"{out / 'plan.json'} is another plan, commit or library set; --resume continues only the "
                         "same one")
    started = time.perf_counter()
    batch = replay.draw(instructions, SPLIT, spec, words, per_airport=args.per_airport, seed=args.seed,
                        groups=GROUPS)
    if canonical(batch.drawn) != canonical(stored["drawn"]):
        raise SystemExit("the drawn sample differs from the formal replay's")
    print(f"{batch.drawn['flights']} {SPLIT} flights {batch.drawn['by_group']}, {time.perf_counter() - started:.0f}s",
          flush=True)

    if not out.exists():
        out.mkdir(parents=True)
        write_json_atomic(out / "plan.json", plan)
    observed_reports = {}
    for airport in sorted({flight.airport for flight in batch.signals}):
        directory = out / "observed" / airport
        if (directory / "evaluation_report.json").exists():
            observed_reports[airport] = json.loads((directory / "evaluation_report.json").read_text(encoding="utf-8"))
            continue
        if directory.exists():
            directory.rename(directory.with_name(f"{airport}.aborted-{utc_now()}"))
        keys = {series.scenario.source["flight_key"] for flight, series in zip(batch.signals, batch.series)
                if flight.airport == airport}
        observed_reports[airport] = observed_verdicts(airport, keys, directory)
    print(f"observed flights graded, {time.perf_counter() - started:.0f}s", flush=True)

    results: dict[str, dict[str, Any]] = {}
    for number, cell in enumerate(cells):
        directory = out / "cells" / f"{number:02d}_{cell.name}"
        if (directory / "cell.json").exists():
            results[cell.name] = json.loads((directory / "cell.json").read_text(encoding="utf-8"))
        else:
            require_plan_commit(plan)
            if directory.exists():
                directory.rename(directory.with_name(f"{directory.name}.aborted-{utc_now()}"))
            results[cell.name] = fly_cell(batch, cell, number, spec, params, chunk=args.chunk, directory=directory,
                                          observed_reports=observed_reports, keep_records=args.keep_records)
            require_plan_commit(plan)       # graded by the plan's code too (evaluation runs as a subprocess)
            write_json_atomic(directory / "cell.json", results[cell.name])
        if cell == reference:
            reproduced = reproduction(results[cell.name]["rows"], stored["flights"])
            if reproduced["executor_fields_differ"]:
                raise SystemExit(f"the reference cell does not reproduce the formal replay: "
                                 f"{len(reproduced['executor_fields_differ'])} flights differ, e.g. "
                                 f"{reproduced['executor_fields_differ'][:3]}")
        row = results[cell.name]["gates"][replay.OWN]["all"]["all"]
        print(f"  {number:02d} {cell.name:22s} labelled {results[cell.name]['labelled']}  landed {row['landed']:.4f}  "
              f"words {row['words_inside']:.4f}  eval {row['replay_passes_where_observed_passes']:.4f}  "
              f"{time.perf_counter() - started:.0f}s", flush=True)

    reference_rows = results[reference.name]["rows"]
    strata = ("all", *STRATA)
    result = {
        "schema": ABLATION_SCHEMA, "written_utc": utc_now(), "plan": plan, "git": git, "sample": batch.drawn,
        "reference": reference.name, "reference_reproduction": reproduction(reference_rows, stored["flights"]),
        "cells": {name: {"cell": cell["cell"], "params": cell["params"],
                         "vocabulary_spec_sha256": cell["vocabulary_spec_sha256"], "labelled": cell["labelled"],
                         "refused": dict(Counter(r["reason"] for r in cell["refused"]).most_common()),
                         "summary": by_group(cell["rows"], cell["refused"]),
                         "gates_cleared": {stratum: gates_cleared(cell["gates"], stratum) for stratum in strata},
                         "against_reference": {stratum: paired_changes(cell["rows"], reference_rows, stratum)
                                               for stratum in strata}}
                  for name, cell in results.items()},
        "elapsed_s": time.perf_counter() - started}
    write_json_atomic(out / "ablation.json", result)
    text = render(result)
    (out / "ablation.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"→ {out / 'ablation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
