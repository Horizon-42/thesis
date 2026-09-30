"""A free-generation readout's sentences as evaluation records (the prior speaks, the executor flies): each stored
sentence flown again from its first predicted step, written as control-path prediction records and graded by the
evaluation package — ADE / FDE against the observed flight and the evaluation's verdict, per sentence.

The readout (`prior_free_generation`'s directory: ``generation.json`` + ``sentences.npz``) stores the words the prior
said and each sentence's outcome, not the flown track. The executor is deterministic and a said sentence is heard as the
closed loop heard it — step k's words from its first cycle, on the sentence's own clock (`TimeClock`), from the observed
state at row `N_LOOK` (an augmented start: moved by its augmentation) to the same time limit (`prior_free_generation
.limits_s`) — so flying the stored words again (`fly_said`) is flying the sentence. The readout's own draw is rebuilt
(`replay.draw` with its split, seed and per-airport count; its ``drawn`` must be the readout's) and flown in the readout's
chunks (``--chunk``: the readout's, its ``run.sh`` says it; the batch a flight is flown in is the readout's), and every
sentence re-flown must reproduce its row of the readout — outcome, end, runways, words said (`flight_rows` /
`said_rows`, the glidepath stop included) — or nothing is written. So the records are the readout's sentences.

What is written, per sentence kind and airport (``records/<kind>/<ICAO>``: ``labelled`` — the labelled words flown from
the same row, the readout's reference, real starts only — and ``sample_<k>`` — the prior's k-th sentence of each flight):
the flown states from the first predicted step to the row its outcome is read at (`executor_replay.executor_forecast`
at anchor `N_LOOK`; a sentence the glidepath lower edge stopped ends at its stopping step), the observed flight from
the same row as its reference, and the evaluation report of the directory. ADE / FDE are `observed_series_metrics`'s:
against the observed track from the first predicted step, on its own common time grid. A sentence whose flight failed
at its first cycle has no state to record: no record, counted.

**An augmented start has no truth** (post-training design §4): its start is the source flight's moved (rotated about the
airport, raised, sped up), and no aircraft flew on from there. Its record's reference is the source flight's observed
track moved by the same augmentation (`augmented_series`: the record's first state must be the flown start); the
evaluation's verdict does not read the reference, so the pass rate is the start's, but its ADE / FDE are the distance to a
moved track, not an error — the readout leaves them out, and each record directory's summary says so.

The readout (``records.json``), per kind and pooled over the prior's samples, in all / per airport / per approach kind /
per airport × approach kind: the sentences, the ones recorded, the evaluation's verdicts and the pass rate (verdict
``pass`` over EVERY sentence: one not recorded does not pass), the outcomes; on real starts ADE / FDE over the recorded
sentences and over the landed ones, and over the prior's samples each flight's best (the smallest ADE and, apart, the
smallest FDE of its recorded samples) and whether any of its samples passes.

    python run_ts.py prior_generation_records \\
        --readout 4dTrajectory/outputs/POOLED/prior/v3_reread_v11_20260927/val_base_400x4 \\
        --instructions 4dTrajectory/outputs/POOLED/instruction_language/v5_20260926 \\
        --executor 4dTrajectory/outputs/POOLED/executor/v11_20260927 --chunk 32 \\
        --out 4dTrajectory/outputs/POOLED/prior/v3_reread_v11_20260927/records_val_base_400x4

Writes into ``--out`` (a new directory; from a clean tree unless ``--chunks`` limits the run to the readout's first
chunks, which the readout then records as partial).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ts_transformer.autopilot import replay
from ts_transformer.autopilot.executor import Flown, fly
from ts_transformer.autopilot.flights import FlightInputs, flight_inputs
from ts_transformer.autopilot.judge import Outcome
from ts_transformer.autopilot.lateral import Runways
from ts_transformer.autopilot.frame import AirportCharts
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.autopilot.plant import EXECUTOR_DYNAMICS
from ts_transformer.autopilot.sentence import Sentences, TimeClock
from ts_transformer.data.channels import channels_from_states, states_from_channels
from ts_transformer.data.dataset import FlightSeries
from ts_transformer.experiments.executor_replay import evaluate_records, executor_forecast
from ts_transformer.experiments.prior_free_generation import (
    GENERATION_SCHEMA, _physics, augmented_inputs, fly_reference, flight_rows, glidepath_stops,
    limits_s, reference_grid, said_rows,
)
from ts_transformer.experiments.prior_generation_training_export import outputs_path
from ts_transformer.inference.export import build_prediction_record, observed_series_metrics, write_batch
from ts_transformer.inference.forecast import Forecast
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.artefact import load_candidates
from ts_transformer.instructions.readout import STRATA
from ts_transformer.instructions.words import Words
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.prior.augment import Augmentation, augment_signals, rotate
from ts_transformer.prior.masks import PROCEDURE_ALTITUDES, ProcedureMasks
from ts_transformer.prior.scene import N_LOOK
from ts_transformer.repo_layout import REPO_ROOT, git_state

RECORDS_SCHEMA = "ts-prior-generation-records-v1"
#: The predictor name each kind of sentence's records carry.
PREDICTORS = {"labelled": "executor", "prior": "prior"}
HORIZON = "sentence"
#: Which runway the evaluation's verdict judges (its context is the record's ``source.runway``).
VERDICT_RUNWAY = ("the observed flight's runway (the record's source.runway): a sentence ending on another runway is "
                  "judged against the observed one — its verdicts are counted apart")
#: A readout row's fields the re-flown sentence must reproduce are all of them but these: the prior's probability on what
#: the masks removed (the speaker's, not the flight's) and the readouts before the join (the speaker's rows).
SPEAKER_FIELDS = ("forbidden_mass", "pre_join", "pre_join_observed")
#: What an augmented start's records are measured against (the summary of each of its record directories says so).
AUGMENTED_REFERENCE = ("the source flight's observed track moved by the start's augmentation: no aircraft flew on from "
                       "an augmented start, so ADE / FDE here are a distance to a moved track, not an error; the "
                       "evaluation's verdict does not read the reference")


def fly_said(inputs: FlightInputs, grids: Sequence[np.ndarray], runways: Runways, charts: AirportCharts,
             approach_ias_mps: torch.Tensor, limits: Sequence[float], words: Words, params: ExecutorParams) -> Flown:
    """Sentences said in a closed loop flown again: step k's words heard from its first cycle, on the sentence's own
    clock, to each flight's time limit — the flight the closed loop flew (`tests/test_prior_generation_records.py`)."""
    device = inputs.initial_state.device
    return fly(inputs, Sentences(grids, words, device=device), TimeClock(params.cycle_s), runways, charts,
               approach_ias_mps, params, words,
               time_limit_s=torch.tensor(limits, dtype=torch.float64, device=device))


def augmented_series(series: FlightSeries, geometry: AirportGeometry, augmentation: Augmentation) -> FlightSeries:
    """The observed flight moved as an augmentation moves its start (`prior.augment`): every row — observed and
    supervision — rotated about the airport, stretched by the speed scale about the first predicted step's row
    (positions and heights), raised; its airspeed scaled, its track turned, its path angle kept. Its row `N_LOOK` is
    the executor's augmented start (`augment.augment_state`); its rows the moved signals' (`augment.augment_signals`)."""
    mass = float(series.scenario.initial.m)
    frame, turn, k = geometry.frame, augmentation.rotation_deg, augmentation.speed_scale
    centre = states_from_channels(series.times[N_LOOK: N_LOOK + 1], series.values[N_LOOK: N_LOOK + 1], series.frame,
                                  mass_kg=mass)[0][1]
    e0, n0 = rotate(*(np.float64(v) for v in frame.horizontal_from_latlon(centre.latitude, centre.longitude)), turn)

    def moved(times: np.ndarray, values: np.ndarray) -> np.ndarray:
        rows = []
        for t, state in states_from_channels(times, values, series.frame, mass_kg=mass):
            e, n = rotate(*(np.float64(v) for v in frame.horizontal_from_latlon(state.latitude, state.longitude)), turn)
            lat, lon = frame.latlon_from_horizontal(float(e0 + k * (e - e0)), float(n0 + k * (n - n0)))
            rows.append((t, replace(state, latitude=float(lat), longitude=float(lon),
                                    altitude=centre.altitude + augmentation.altitude_m + k * (state.altitude - centre.altitude),
                                    V=state.V * k, psi=state.psi - math.radians(turn))))
        return channels_from_states(rows, series.frame)[1]

    return replace(series, values=moved(series.times, series.values),
                   supervision_values=moved(series.supervision_times, series.supervision_values))


@dataclass
class Flight:
    """One re-flown sentence waiting to be written: its series (moved for an augmented start), the flown track as a
    forecast (None: not recorded), its metrics, and its readout row."""

    series: FlightSeries
    forecast: Forecast | None
    metrics: dict[str, Any] | None
    row: dict[str, Any]


def recorded_flights(flown: Flown, inputs: FlightInputs, rows: list[dict[str, Any]],
                     series: Sequence[FlightSeries]) -> list[Flight]:
    """Each flight of a flown chunk as a record (``rows``: its readout rows, `flight_rows`): its states to the row its
    outcome is read at — a stopped sentence's to its stopping step's end — unless it has no state past its start (a
    failure at its first cycle)."""
    out = []
    for j, row in enumerate(rows):
        end_row = int(round(row["end_s"] / flown.cycle_s))
        # the last state the record keeps (`executor_forecast`: before a dynamics failure's): none past the start
        recorded = (end_row - 1 if row["outcome"] == "dynamics_failure" else end_row) >= 1
        forecast = metrics = None
        if recorded:
            forecast = executor_forecast(flown, j, Outcome(row["outcome"], end_row, None, {}), inputs, series[j],
                                         anchor=N_LOOK)
            metrics = observed_series_metrics(series[j], forecast)
        out.append(Flight(series[j], forecast, metrics, {**row, "recorded": recorded}))
    return out


def reproduced(stored: Sequence[dict[str, Any]], again: Sequence[dict[str, Any]], ignore: Sequence[str]) -> list[str]:
    """The re-flown rows that differ from the readout's (one line each), every field but ``ignore``."""
    wrong = []
    for old, new in zip(stored, again, strict=True):
        fields = [key for key in old if key not in ignore and old[key] != new[key]]
        if fields:
            wrong.append(f"{old['dataset_id']} sample {old['sample']}: " + ", ".join(
                f"{key} {old[key]!r} → {new[key]!r}" for key in fields))
    return wrong


def write_kind(flights: list[Flight], directory: Path, *, predictor: str, split: str, checkpoint: str,
               params: ExecutorParams, extra_summary: dict[str, Any]) -> dict[str, str]:
    """One kind's records of one airport, written and graded (the flights with no record counted in its summary):
    each recorded flight's verdict by its flight key."""
    written = [f for f in flights if f.forecast is not None]
    unrecorded = len(flights) - len(written)
    records = [build_prediction_record(f.series, f.forecast, index=i, model_name=predictor, horizon_mode=HORIZON,
                                       split=split) for i, f in enumerate(written)]
    for record, flight in zip(records, written):
        record.states_payload["source"]["freeGeneration"] = {
            "sentence": flight.row["source"], "sample": flight.row["sample"], "outcome": flight.row["outcome"],
            "augmentation": flight.row["augmentation"]}
    write_batch(records, output_dir=directory,
                config_dict={"model": predictor, "horizon_mode": HORIZON,
                             "prediction_output": EXECUTOR_DYNAMICS.prediction_output,
                             "executor_params": asdict(params)},
                flight_metrics=[f.metrics for f in written], checkpoint=checkpoint, split=split,
                skipped={"no_state_past_the_start": unrecorded} if unrecorded else None, extra_summary=extra_summary)
    graded = evaluate_records(directory)
    return {row["flight_key"]: row["verdict"] for row in graded["trajectories"]}


def _stats(values: Sequence[float]) -> dict[str, float] | None:
    if not len(values):
        return None
    values = np.asarray(values, dtype=np.float64)
    return {"median": float(np.median(values)), "mean": float(values.mean()), "p90": float(np.percentile(values, 90)),
            "count": int(len(values))}


def summarise(rows: list[dict[str, Any]], *, errors: bool) -> dict[str, Any]:
    """Shares over ``rows`` (sentences): recorded, verdicts, the pass rate over every sentence, outcomes; the verdict
    judges the observed flight's runway (`VERDICT_RUNWAY`), so the pass rate again over the sentences ending on it and
    the verdicts of the others apart; with ``errors``, ADE / FDE over the recorded ones and the landed ones."""
    count = len(rows)
    recorded = [r for r in rows if r["recorded"]]
    verdicts: dict[str, int] = defaultdict(int)
    for r in rows:
        verdicts[r["verdict"]] += 1
    outcomes: dict[str, int] = defaultdict(int)
    for r in rows:
        outcomes[r["outcome"]] += 1
    out = {"sentences": count, "recorded": len(recorded), "verdicts": dict(sorted(verdicts.items())),
           "pass_rate": verdicts["pass"] / count, "outcomes": {k: v / count for k, v in sorted(outcomes.items())}}
    observed = [r for r in rows if r["last_runway"] == r["observed_runway"]]
    other: dict[str, int] = defaultdict(int)
    for r in rows:
        if r["last_runway"] != r["observed_runway"]:
            other[r["verdict"]] += 1
    out.update({"ending_on_observed_runway": {
                    "sentences": len(observed),
                    "pass_rate": (sum(r["verdict"] == "pass" for r in observed) / len(observed) if observed else None)},
                "ending_on_another_runway": {"sentences": count - len(observed), "verdicts": dict(sorted(other.items()))}})
    if errors:
        landed = [r for r in recorded if r["outcome"] == "landed"]
        out.update({"ade_m": _stats([r["ade_m"] for r in recorded]), "fde_m": _stats([r["fde_m"] for r in recorded]),
                    "landed_ade_m": _stats([r["ade_m"] for r in landed]),
                    "landed_fde_m": _stats([r["fde_m"] for r in landed])})
    return out


def best_of_samples(rows: list[dict[str, Any]], *, errors: bool) -> dict[str, Any]:
    """Over the prior's samples, per flight: whether any passes, and with ``errors`` the smallest ADE and, apart, the
    smallest FDE of its recorded samples (a flight with none recorded has neither)."""
    flights: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        flights[r["dataset_id"]].append(r)
    out = {"flights": len(flights),
           "any_sample_passes": sum(any(r["verdict"] == "pass" for r in group) for group in flights.values()) / len(flights)}
    if errors:
        recorded = [[r for r in group if r["recorded"]] for group in flights.values()]
        out.update({"min_ade_m": _stats([min(r["ade_m"] for r in group) for group in recorded if group]),
                    "min_fde_m": _stats([min(r["fde_m"] for r in group) for group in recorded if group])})
    return out


def grouped(rows: list[dict[str, Any]], *, errors: bool, samples: bool) -> dict[str, Any]:
    """`summarise` (and over samples `best_of_samples`) in all, per airport, per approach kind, per airport × kind."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        for key in ("all", r["airport"], r["stratum"], f"{r['airport']} {r['stratum']}"):
            groups[key].append(r)
    return {key: {**summarise(value, errors=errors),
                  **({"best_of_samples": best_of_samples(value, errors=errors)} if samples else {})}
            for key, value in sorted(groups.items())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--readout", type=Path, required=True, help="a prior_free_generation directory")
    parser.add_argument("--instructions", type=Path, required=True, help="the readout's instruction artefact")
    parser.add_argument("--executor", type=Path, required=True, help="the readout's executor spec directory")
    parser.add_argument("--chunk", type=int, required=True, help="the readout's --chunk (its run.sh)")
    parser.add_argument("--chunks", type=int, default=0,
                        help="0: every chunk; N: the readout's first N chunks only (a partial run, recorded as one)")
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    args = parser.parse_args(argv)

    def resolved(path: Path) -> Path:
        return path if path.is_absolute() else REPO_ROOT / path

    readout, instructions, executor_dir, out = map(resolved, (args.readout, args.instructions, args.executor, args.out))
    if out.exists():
        parser.error(f"{out} exists; records are never overwritten")
    git = git_state()
    if git["dirty"] and not args.chunks:
        parser.error("a whole readout's records are written from a clean tree (--chunks N for a partial run)")
    generation = json.loads((readout / "generation.json").read_text(encoding="utf-8"))
    if generation["schema"] != GENERATION_SCHEMA:
        parser.error(f"{readout} is a {generation['schema']} readout; this code reads {GENERATION_SCHEMA}")
    started = time.perf_counter()
    params, spec_record, words = replay.open_executor(executor_dir, instructions)
    if spec_record["sha256"] != generation["executor"]["sha256"]:
        parser.error(f"{executor_dir} is not the readout's executor spec ({generation['executor']['sha256'][:12]})")
    if outputs_path(generation["instructions"]) != outputs_path(instructions):
        parser.error(f"the readout read {generation['instructions']}, not {instructions}")
    geometries = load_candidates(instructions)
    masks = (ProcedureMasks.build((PROCEDURE_ALTITUDES,), geometries) if generation["procedure_masks"]
             else ProcedureMasks.none())
    drawn = generation["drawn"]
    batch = replay.draw(instructions, generation["split"], words.spec, words, per_airport=drawn["per_airport"],
                        seed=generation["seed"])
    if batch.drawn != drawn:
        raise SystemExit("the draw rebuilt differs from the readout's")
    augmented = generation["augment_seed"] is not None
    augmentations: list[Augmentation | None] = [None] * len(batch.readings)
    if augmented:
        moves = {entry["dataset_id"]: Augmentation(entry["rotation_deg"], entry["altitude_m"], entry["speed_scale"])
                 for entry in generation["augmentations"]}
        kept = [j for j, s in enumerate(batch.signals) if s.dataset_id in moves]
        if len(kept) != len(moves) or len(batch.readings) - len(kept) != generation["augmented_left_out"]:
            raise SystemExit("the readout's augmented starts are not flights of its draw")
        batch = replay.subset(batch, kept)
        augmentations = [moves[s.dataset_id] for s in batch.signals]
    samples, step_s, cpu = generation["samples"], words.spec.step_s, torch.device("cpu")
    stored = np.load(readout / "sentences.npz")
    offsets, said_words = stored["offsets"], stored["words"].astype(np.int64)
    prior_rows = [r for r in generation["flights"] if r["source"] == "prior"]
    labelled_rows = [r for r in generation["flights"] if r["source"] == "labelled"]
    if [r["dataset_id"] for r in prior_rows] != stored["dataset_id"].tolist():
        raise SystemExit("the readout's sentences and its rows are not in one order")
    order = sorted(range(len(batch.readings)), key=lambda j: len(batch.readings[j].words))
    chunks = [order[s: s + args.chunk] for s in range(0, len(order), args.chunk)]
    if args.chunks:
        if args.chunks >= len(chunks):
            parser.error(f"--chunks {args.chunks}: the readout has {len(chunks)} chunks of {args.chunk} — a whole run is "
                         f"made without --chunks")
        chunks = chunks[: args.chunks]
    print(f"{readout.name}: {len(batch.readings)} {generation['split']} flights × {samples} sentences"
          f"{' from augmented starts' if augmented else ''}, {len(chunks)} chunk(s) of {args.chunk}, "
          f"{time.perf_counter() - started:.0f}s", flush=True)

    kinds: dict[tuple[str, str], list[Flight]] = defaultdict(list)
    sentence = labelled_at = 0
    for number, chunk in enumerate(chunks):
        part = replay.subset(batch, chunk)
        if not augmented:
            reference = fly_reference(part, words, params, device=cpu)
            grids = [reference_grid(r.words) for r in part.readings]
            stops = (glidepath_stops(reference, grids, part.geometries, [masks.finals[g.code] for g in part.geometries],
                                     words) if masks.altitudes else None)
            rows = flight_rows(part, reference, grids, words, "labelled", [None] * len(chunk), None, stops)
            wrong = reproduced(labelled_rows[labelled_at: labelled_at + len(rows)], rows, ())
            if wrong:
                raise SystemExit(f"{len(wrong)} labelled sentence(s) of chunk {number} flew otherwise than in the "
                                 f"readout, e.g. {wrong[:3]}")
            labelled_at += len(rows)
            for flight in recorded_flights(reference, flight_inputs(part.series, device=cpu, anchor=N_LOOK),
                                           [{**r, "augmentation": None} for r in rows], part.series):
                kinds[("labelled", flight.row["airport"])].append(flight)
        index = [j for j in range(len(chunk)) for _ in range(samples)]
        repeated = replay.subset(part, index)
        inputs = flight_inputs(repeated.series, device=cpu, anchor=N_LOOK)
        moves = [augmentations[chunk[j]] for j in index]
        series = list(repeated.series)
        if augmented:
            repeated = replace(repeated, signals=[augment_signals(s, a) for s, a in zip(repeated.signals, moves)])
            inputs = augmented_inputs(inputs, repeated.geometries, moves)
            series = [augmented_series(s, g, a) for s, g, a in zip(series, repeated.geometries, moves)]
        said = [said_words[offsets[i]: offsets[i + 1]] for i in range(sentence, sentence + len(index))]
        if len({len(grid) for grid in said}) != 1:
            raise SystemExit(f"chunk {number}'s sentences were not said in one closed loop: --chunk {args.chunk} is not "
                             f"the readout's")
        runways, charts, approach = _physics(repeated, cpu)
        flown = fly_said(inputs, said, runways, charts, approach,
                         limits_s(repeated, params, step_s, augmented=augmented), words, params)
        rows, _, _ = said_rows(repeated, flown, np.stack(said), {}, words, [j % samples for j in range(len(index))],
                               masks)
        wrong = reproduced(prior_rows[sentence: sentence + len(rows)], rows, SPEAKER_FIELDS)
        if wrong:
            raise SystemExit(f"{len(wrong)} sentence(s) of chunk {number} flew otherwise than in the readout, "
                             f"e.g. {wrong[:3]}")
        sentence += len(rows)
        rows = [{**r, "augmentation": None if a is None else asdict(a)} for r, a in zip(rows, moves)]
        for flight in recorded_flights(flown, inputs, rows, series):
            kinds[(f"sample_{flight.row['sample']}", flight.row["airport"])].append(flight)
        del flown
        print(f"  chunk {number + 1}/{len(chunks)}: {sentence} sentences reproduced, "
              f"{time.perf_counter() - started:.0f}s", flush=True)

    if not args.chunks and (sentence, labelled_at) != (len(prior_rows), len(labelled_rows)):
        raise SystemExit(f"the chunks flew {sentence} of the readout's {len(prior_rows)} sentences and {labelled_at} of "
                         f"its {len(labelled_rows)} labelled ones")
    out.mkdir(parents=True)
    extra = {"free_generation": {"readout": str(readout), "augmented": augmented,
                                 "reference": AUGMENTED_REFERENCE if augmented else "the observed flight"},
             "executor_spec_sha256": spec_record["sha256"]}
    rows_out: list[dict[str, Any]] = []
    for (kind, airport), flights in sorted(kinds.items()):
        source = "labelled" if kind == "labelled" else "prior"
        checkpoint = str(executor_dir) if source == "labelled" else generation["prior"]["directory"]
        verdicts = write_kind(flights, out / "records" / kind / airport, predictor=PREDICTORS[source],
                              split=generation["split"], checkpoint=checkpoint, params=params, extra_summary=extra)
        for flight in flights:
            key = flight.series.scenario.source["flight_key"]
            rows_out.append({
                "dataset_id": flight.row["dataset_id"], "flight_key": key, "airport": airport,
                "stratum": flight.row["stratum"], "kind": kind, "source": source, "sample": flight.row["sample"],
                "outcome": flight.row["outcome"], "end_s": flight.row["end_s"], "recorded": flight.forecast is not None,
                "observed_runway": flight.row["observed_runway"], "last_runway": flight.row["last_runway"],
                "verdict": verdicts[key] if flight.forecast is not None else "not recorded",
                "ade_m": None if flight.metrics is None else flight.metrics["ade_m"],
                "fde_m": None if flight.metrics is None else flight.metrics["fde_m"]})
        print(f"  {kind} {airport}: {len(flights)} written and graded, {time.perf_counter() - started:.0f}s", flush=True)
        del flights[:]

    errors = not augmented
    readouts = {"prior": grouped([r for r in rows_out if r["source"] == "prior"], errors=errors, samples=True)}
    for k in range(samples):
        readouts[f"sample_{k}"] = grouped([r for r in rows_out if r["kind"] == f"sample_{k}"], errors=errors,
                                          samples=False)
    if not augmented:
        readouts["labelled"] = grouped([r for r in rows_out if r["source"] == "labelled"], errors=True, samples=False)
    write_json_atomic(out / "records.json", {
        "schema": RECORDS_SCHEMA, "written_utc": utc_now(), "git": git,
        "readout": {"directory": str(readout), "schema": generation["schema"], "prior": generation["prior"],
                    "split": generation["split"], "samples": samples, "augment_seed": generation["augment_seed"],
                    "procedure_masks": generation["procedure_masks"]},
        "executor": {"directory": str(executor_dir), "sha256": spec_record["sha256"]},
        "instructions": str(instructions), "chunk": args.chunk,
        "partial": {"chunks": len(chunks), "of": math.ceil(len(order) / args.chunk)} if args.chunks else None,
        "reference": AUGMENTED_REFERENCE if augmented else "the observed flight from the first predicted step",
        "verdict": VERDICT_RUNWAY,
        "errors": ("ADE / FDE: observed_series_metrics against the observed track from the first predicted step "
                   "(row N_LOOK)" if errors else "none: an augmented start has no truth"),
        "readouts": readouts, "sentences": rows_out, "elapsed_s": time.perf_counter() - started})
    for name, readout_block in readouts.items():
        print(f"{name}:")
        for key in ("all", *STRATA):
            if key not in readout_block:
                continue
            part = readout_block[key]
            line = f"  {key:12s} n={part['sentences']:5d}  pass {part['pass_rate']:.3f}"
            if errors:
                line += (f"  ADE p50 {part['ade_m']['median']:7.0f} m  FDE p50 {part['fde_m']['median']:7.0f} m"
                         if part["ade_m"] else "")
            if "best_of_samples" in part:
                best = part["best_of_samples"]
                line += f"  | best of {samples}: any passes {best['any_sample_passes']:.3f}"
                if errors and best["min_ade_m"]:
                    line += (f"  minADE p50 {best['min_ade_m']['median']:7.0f} m"
                             f"  minFDE p50 {best['min_fde_m']['median']:7.0f} m")
            print(line)
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
