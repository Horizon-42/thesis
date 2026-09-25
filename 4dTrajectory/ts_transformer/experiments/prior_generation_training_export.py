"""The prior's OWN sentences over a Training set's flights, for the frontend: each flight flown from its first predicted
step with the prior speaking and the executor flying (single-aircraft free generation, prior design §9.1) — the words it
said, the flown track and how the flight ended — ``--samples`` times (the Training module:
`aeroviz-4d/docs/36-2026-09-20-training-module.zh.md`).

    python run_ts.py prior_generation_training_export \\
        --prior 4dTrajectory/outputs/POOLED/prior/<the prior run> --label "base model" \\
        --instructions 4dTrajectory/outputs/POOLED/instruction_language/<its artefact> \\
        --executor 4dTrajectory/outputs/POOLED/executor/<a spec this executor code opens> \\
        --readout 4dTrajectory/outputs/POOLED/prior/<campaign>/<its val free generation> \\
        --airports-root aeroviz-4d/public/data/airports --set <a read-back set of that artefact> \\
        --airport KMSY --airport KRDU --airport KSJC --airport KSMF --airport KSTL

Per airport, writes ``<airports-root>/<ICAO>/training/<overlay-id>/generation.json`` (schema `SCHEMA`; refused if the
directory exists) and adds the overlay to ``training/overlays.json`` (kind `KIND_GENERATION`; refused if it lists the
id already). The set (``--set``) must be a read-back set of the prior's vocabulary drawn from val.

**The loop is the val readout's** (`prior_free_generation.speak_and_fly`): the first `N_LOOK` rows are observed, the
executor starts from the observed state at row `N_LOOK`, the prior says every column there and then speaks step by step
from where the executor flew, masked by the vocabulary's compatibility rules and the listener's runway lock; each flight
flies until the executor is done with it or its time limit (`prior_free_generation.limits_s`). Only the flights the val
readout flies — their own dynamics (`replay.OWN`) — are flown; the rest of the set is listed with the reason. The
outcome is the executor's judge's (`prior_free_generation.flight_rows`: `judge.outcome_of` on the runway pointed at the
end). One generator, seeded, draws every sample of every airport in the order the airports are named.

``--readout`` (optional) carries the prior's formal val readout beside the set's own flights: the landed share in all
and per approach kind, at the payload's airport and pooled over every airport the readout drew, refused unless it is a
free generation of THIS prior (its directory from ``4dTrajectory/outputs/`` on — the readouts were run from worktrees
whose outputs are the same tree), on this executor spec, this artefact, val, with the same samples and temperature.

What is written per sample: the words said as events (``row``: the flight's own step, the first `N_LOOK` observed), the
flown track every 2 s step on the flight's own clock (``tS`` from row `N_LOOK`'s time) to its outcome's row — a dynamics
failure to the row before, as the executor's replay export keeps it — in MSL and in the ellipsoid height Cesium draws in
(h = H + N), the outcome, the crossing, the runway pointed first and last, and the probability the prior put on what the
masks removed. The words carry no verdicts: the judge would say how the EXECUTOR flew the prior's words, not what the
prior said; the landing is the prior's answer. SI units.

**The words run to where the executor stopped, the track to the outcome**: `flight_rows` counts a sentence to the step
whose cycles ended the flight for the executor, which flies on after two outcomes its judge reads earlier — crossing the
threshold without the capture, and the stall cut-off (`executor.Executor.finished` stops at neither) — so a sample's last
words can follow its ``endS``. They are written as the formal readout counts them (its runway pointed at the end is read
from them); the frontend shades the rows after the end.

"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from flight_scenarios.datum import geoid_undulation_m
from ts_transformer.autopilot import replay
from ts_transformer.autopilot.executor import Flown
from ts_transformer.autopilot.flights import flight_inputs, rebuild_series
from ts_transformer.autopilot.frame import ALT, LAT, LON
from ts_transformer.autopilot.judge import flown_track, outcome_of
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.autopilot.runway_data import published_crossing_heights
from ts_transformer.experiments.prior_free_generation import (
    GENERATION_SCHEMA, _physics, flight_rows, limits_s, speak_and_fly,
)
from ts_transformer.experiments.prior_train import rosters
from ts_transformer.experiments.prior_training_export import open_trained_prior
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.artefact import load_candidates, load_sentences, load_signals
from ts_transformer.instructions.labeller.read import read_flight
from ts_transformer.instructions.readout import STRATA
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.training_files import (
    KIND_GENERATION, SPLIT, BaseSet, base_flights, open_base_set, overlay_entry, read_overlays,
    require_overlays_unchanged, rounded, serialise, write_overlay,
)
from ts_transformer.instructions.words import COLUMNS, UNCHANGED, Words
from ts_transformer.io_utils import utc_now
from ts_transformer.prior.data import VARIANTS, airport_landings
from ts_transformer.prior.model import Prior
from ts_transformer.prior.scene import N_LOOK
from ts_transformer.repo_layout import REPO_ROOT, git_state, repo_relative

#: MIRROR of `aeroviz-4d/src/data/trainingOverlays.ts` (`TRAINING_GENERATION_SCHEMA`); the reader refuses anything else
#: by name. A name changes with its file's shape or meaning, on both sides, in one change.
SCHEMA = "aeroviz-training-generation-v1"
PAYLOAD_FILE = "generation.json"
RUNNER = "ts_transformer.experiments.prior_generation_training_export"
#: The one tree every checkout's outputs are (a worktree links it): a readout's prior is known by its path from here on.
OUTPUTS_MARK = "4dTrajectory/outputs/"
#: MIRROR of the phrase `replay.draw_flights` writes for ``per_airport`` 0 (every labelled flight of the split).
EVERY_FLIGHT = "every labelled flight"


def outputs_path(path: str | Path) -> str:
    """A path read from ``4dTrajectory/outputs/`` on — the same artefact from any checkout (worktrees link the tree)."""
    text = Path(path).as_posix()
    if OUTPUTS_MARK not in text:
        raise ValueError(f"{text} is not under {OUTPUTS_MARK}")
    return text[text.rindex(OUTPUTS_MARK):]


def readout_block(generation: dict[str, Any], prior_dir: Path, executor_sha256: str, instructions: Path, samples: int,
                  temperature: float, airport: str) -> dict[str, Any]:
    """The prior's formal val free generation (`prior_free_generation`'s ``generation.json``) as the frontend reads it —
    the landed share in all and per approach kind, at ``airport`` (``here``) and over every airport the readout drew
    (``all``), of the prior's sentences and of the labelled words flown from the same row — refused unless it is this
    prior's, on this executor spec and artefact, val, with these samples and this temperature."""
    if generation["schema"] != GENERATION_SCHEMA:
        raise ValueError(f"the readout is a {generation['schema']} file, not {GENERATION_SCHEMA}")
    wanted = {"prior": outputs_path(prior_dir), "executor": executor_sha256,
              "instructions": outputs_path(instructions), "split": SPLIT, "n_look": N_LOOK, "samples": samples,
              "temperature": temperature}
    found = {"prior": outputs_path(generation["prior"]["directory"]),
             "executor": generation["executor"]["sha256"], "instructions": outputs_path(generation["instructions"]),
             "split": generation["split"], "n_look": generation["n_look"], "samples": generation["samples"],
             "temperature": generation["temperature"]}
    differ = {key: (found[key], wanted[key]) for key in wanted if found[key] != wanted[key]}
    if differ:
        raise ValueError("the readout is not this prior's val free generation: " +
                         ", ".join(f"{key} {theirs!r}, expected {ours!r}" for key, (theirs, ours) in differ.items()))

    def cells(source: str, prefix: str | None) -> dict[str, Any]:
        # "all", and each approach kind — null where the draw held no flight of it (the readout lists only the kinds
        # it counted); the airport's own are keyed "<ICAO>" and "<ICAO> <stratum>"
        part = generation["readout"][source]
        keys = {"all": "all" if prefix is None else prefix,
                **{stratum: stratum if prefix is None else f"{prefix} {stratum}" for stratum in STRATA}}
        if keys["all"] not in part:
            raise ValueError(f"the readout counted no {source} flight at {prefix}")
        return {name: ({"flights": part[key]["flights"], "landed": part[key]["outcomes"]["landed"]} if key in part else None)
                for name, key in keys.items()}

    per_airport = generation["drawn"]["per_airport"]
    if not (isinstance(per_airport, int) or per_airport == EVERY_FLIGHT):
        raise ValueError(f"the readout's draw names {per_airport!r} flights an airport")
    return {"split": generation["split"], "writtenUtc": generation["written_utc"], "seed": generation["seed"],
            # the draw's size per airport; 0 = every labelled flight of the split
            "drawn": {"flights": generation["drawn"]["flights"], "perAirport": 0 if per_airport == EVERY_FLIGHT else per_airport},
            "prior": {"here": cells("prior", airport), "all": cells("prior", None)},
            "labelled": {"here": cells("labelled", airport), "all": cells("labelled", None)}}


def track_payload(flown: Flown, index: int, end_row: int, outcome: str, geometry: AirportGeometry, step_s: float,
                  start_s: float) -> dict[str, Any]:
    """The flown track every sentence step from its first state to its outcome's row (a dynamics failure: to the row
    before, as the replay export keeps it — the failed state may not be finite), on the flight's own clock
    (``start_s``: the time of the row the executor started at)."""
    end = end_row - 1 if outcome == "dynamics_failure" else end_row
    states = flown.states[index, : end + 1].cpu().numpy()
    step_rows = int(round(step_s / flown.cycle_s))
    rows = list(range(0, end + 1, step_rows))
    if rows[-1] != end:
        rows.append(end)
    track = flown_track(states[rows], geometry)
    lat, lon, height = states[rows, LAT], states[rows, LON], states[rows, ALT]
    undulation = geoid_undulation_m(lat, lon)
    return {"tS": rounded(start_s + np.asarray(rows) * flown.cycle_s, 3), "lon": rounded(lon, 7), "lat": rounded(lat, 7),
            "altitudeM": rounded(height, 2), "altitudeHaeM": rounded(height + undulation, 2),
            "groundSpeedMps": rounded(track["ground_speed"], 3)}


def sample_payload(flown: Flown, index: int, row: dict[str, Any], grid: np.ndarray, geometry: AirportGeometry,
                   words: Words) -> dict[str, Any]:
    """One sample as the frontend reads it: `flight_rows`'s ``row`` of it (its outcome and bookkeeping), the words
    it said up to its end as events on the flight's own steps (``grid``: [steps, 6], UNCHANGED where a column says
    nothing; step 0 is row `N_LOOK`), the crossing and the flown track."""
    step_s = words.spec.step_s
    said = np.asarray(grid)[: row["steps_said"]]
    if (said[0] == UNCHANGED).any():
        raise ValueError(f"{row['dataset_id']}: the first predicted step does not say every column")
    # the crossing and the outcome's row: `flight_rows` read them with this call and kept only the outcome and the time
    outcome = outcome_of(flown, index, geometry, row["last_runway"], words.spec)
    crossing = outcome.crossing
    start_s = N_LOOK * step_s
    return {
        "sample": row["sample"], "outcome": row["outcome"], "endS": round(start_s + row["end_s"], 3),
        "crossing": None if crossing is None else {"crossM": round(crossing["cross_m"], 2),
                                                   "heightM": round(crossing["height_m"], 2),
                                                   "atS": round(start_s + crossing["at_row"] * flown.cycle_s, 3)},
        "firstRunway": row["first_runway"], "lastRunway": row["last_runway"], "runwayChanges": row["runway_changes"],
        "goArounds": row["go_arounds"], "clearedAtEnd": row["cleared_at_end"],
        "forbiddenMass": {column: round(mass, 6) for column, mass in row["forbidden_mass"].items()},
        "rows": N_LOOK + len(said),
        "events": [{"row": N_LOOK + int(step), "column": int(column), "value": int(said[step, column])}
                   for step, column in zip(*np.nonzero(said != UNCHANGED))],
        "track": track_payload(flown, index, outcome.end_row, row["outcome"], geometry, step_s, start_s),
    }


def build_airport(base: BaseSet, flights: list[FlightSignals], sentences: dict[str, np.ndarray], instructions: Path,
                  geometry: AirportGeometry, model: Prior, params: ExecutorParams, words: Words, landings: Any,
                  samples: int, *, generator: torch.Generator, temperature: float) -> list[dict[str, Any]]:
    """Every flight of the set: those the val readout flies (their own dynamics) flown ``samples`` times with the
    prior speaking, in one batch; the rest listed with the reason."""
    spec = words.spec
    located = base_flights(base, flights, sentences)
    signals = [flight for flight, _ in located]
    series = rebuild_series(instructions, signals)
    groups = [replay.group_of(item) for item in series]
    flyable = [j for j, group in enumerate(groups) if group == replay.OWN]
    readings = []
    for j in flyable:
        reading = read_flight(signals[j], geometry, spec, words)
        k = located[j][1]
        if not np.array_equal(reading.words, sentences["words"][sentences["offsets"][k]: sentences["offsets"][k + 1]]):
            raise ValueError(f"{signals[j].dataset_id}: the re-read sentence differs from the stored one")
        readings.append(reading)
    by_flight: dict[int, list[dict[str, Any]]] = {j: [] for j in flyable}
    if flyable:
        batch = replay.Batch(signals=[signals[j] for j in flyable], series=[series[j] for j in flyable],
                             readings=readings, geometries=[geometry] * len(flyable),
                             crossing_heights=[published_crossing_heights(geometry)] * len(flyable),
                             approach_ias_mps=[replay.flight_approach_ias_mps(series[j], replay.OWN) for j in flyable],
                             groups=[replay.OWN] * len(flyable), drawn={})
        cpu = torch.device("cpu")
        repeated = replay.subset(batch, [n for n in range(len(flyable)) for _ in range(samples)])
        runways, charts, approach = _physics(repeated, cpu)
        flown, said, forbidden, _ = speak_and_fly(model, repeated.signals, repeated.geometries,
                                                  flight_inputs(repeated.series, device=cpu, anchor=N_LOOK), runways,
                                                  charts, approach, limits_s(repeated, params, spec.step_s), words,
                                                  params, landings, generator=generator, temperature=temperature)
        grids = [said[i] for i in range(len(said))]
        rows = flight_rows(repeated, flown, grids, words, "prior", [i % samples for i in range(len(grids))], forbidden)
        for i, row in enumerate(rows):
            by_flight[flyable[i // samples]].append(sample_payload(flown, i, row, grids[i], geometry, words))
    return [{"flightKey": item["flightKey"], "datasetId": item["datasetId"], "group": groups[j],
             "flown": j in by_flight, "samples": by_flight.get(j, [])}
            for j, item in enumerate(base.sample["flights"])]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--prior", type=Path, required=True, help="the prior run whose sentences are drawn")
    parser.add_argument("--label", required=True, help="what the frontend calls it, e.g. 'base model'")
    parser.add_argument("--instructions", type=Path, required=True, help="the instruction artefact it was trained on")
    parser.add_argument("--executor", type=Path, required=True, help="an executor spec directory this code opens")
    parser.add_argument("--readout", type=Path, default=None, help="this prior's val free generation (optional)")
    parser.add_argument("--airports-root", type=Path, required=True,
                        help="the frontend's airports directory (…/public/data/airports)")
    parser.add_argument("--set", required=True, help="the Training set whose flights are flown")
    parser.add_argument("--airport", action="append", required=True, help="an ICAO code; repeat for several")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--overlay-id", default=None, help="default: generation_<the prior's parent>_<its name>")
    args = parser.parse_args(argv)

    def resolved(path: Path) -> Path:
        return path if path.is_absolute() else REPO_ROOT / path

    prior_dir, instructions, executor, root = (resolved(p) for p in (args.prior, args.instructions, args.executor,
                                                                       args.airports_root))
    airports = [code.upper() for code in args.airport]
    if len(set(airports)) != len(airports):
        parser.error(f"an airport is named twice in {airports}")
    if args.samples < 1:
        parser.error("--samples is at least 1")
    overlay_id = args.overlay_id or f"generation_{prior_dir.parent.name}_{prior_dir.name}"
    started = time.perf_counter()
    params, record, words = replay.open_executor(executor, instructions)
    spec = words.spec
    model, config_file, checkpoint_sha = open_trained_prior(prior_dir, instructions)
    missing = [code for code in airports if code not in model.config.airports]
    if missing:
        parser.error(f"the prior knows no airport {missing} ({list(model.config.airports)})")
    formal = None if args.readout is None else json.loads((resolved(args.readout) / "generation.json").read_text(encoding="utf-8"))
    readouts = {code: None if formal is None else readout_block(formal, prior_dir, record["sha256"], instructions,
                                                                 args.samples, args.temperature, code)
                for code in airports}
    geometries = load_candidates(instructions)
    landings = (airport_landings(instructions, rosters(instructions))
                if VARIANTS[model.config.variant].landing_context else None)
    flights = load_signals(instructions, SPLIT)
    sentences = load_sentences(instructions, SPLIT, spec)
    bases: dict[str, BaseSet] = {}
    existing = {}
    for code in airports:
        training = root / code / "training"
        if (training / overlay_id).exists():
            parser.error(f"{training / overlay_id} exists; an overlay is never overwritten")
        existing[code] = read_overlays(training, code, overlay_id)
        bases[code] = open_base_set(training, code, args.set, spec)

    source = {"runner": RUNNER, "prior": repo_relative(prior_dir), "executor": repo_relative(executor),
              "instructions": repo_relative(instructions),
              "readout": None if args.readout is None else repo_relative(resolved(args.readout)), "git": git_state()}
    # a post-trained round's config names its method, round and start model (`prior_landing_reward` writes the key);
    # a prior trained on data alone (`prior_train`) has none
    tuning = config_file["fine_tuning"] if "fine_tuning" in config_file else None
    model_block = {
        "label": args.label, "checkpointSha256": checkpoint_sha, "variant": model.config.variant,
        "trainedAt": config_file["git"],
        "fineTuning": None if tuning is None else {"schema": tuning["schema"], "round": tuning["round"],
                                                   "from": outputs_path(tuning["from"])},
    }
    generation = {"samples": args.samples, "temperature": args.temperature, "seed": args.seed,
                  "firstPredictedRow": N_LOOK, "stepS": spec.step_s, "executor": {
                      "specSha256": record["sha256"], "wordClock": params.word_clock, "cycleS": params.cycle_s,
                      "timeoutFactor": params.timeout_factor}}
    title = (f"{args.label} · {prior_dir.parent.name}/{prior_dir.name} · its own sentences, {args.samples} a flight, flown "
             f"by executor spec {record['sha256'][:12]} from step {N_LOOK}")
    generator = torch.Generator().manual_seed(args.seed)
    built = {}
    for code in airports:
        payloads = build_airport(bases[code], flights, sentences, instructions, geometries[code], model, params, words,
                                 landings, args.samples, generator=generator, temperature=args.temperature)
        payload = {"schema": SCHEMA, "overlayId": overlay_id, "airport": code, "writtenUtc": utc_now(),
                   "producedBy": source, "base": bases[code].block, "model": model_block, "generation": generation,
                   "readout": readouts[code], "columns": list(COLUMNS), "flights": payloads}
        entry = overlay_entry(overlay_id, KIND_GENERATION, bases[code], title, PAYLOAD_FILE, len(payloads), source)
        built[code] = (serialise(payload), entry)
        said = [s for item in payloads for s in item["samples"]]
        landed = sum(s["outcome"] == "landed" for s in said)
        print(f"  {code}: {sum(item['flown'] for item in payloads)} of {len(payloads)} flights flown, {landed} of "
              f"{len(said)} samples landed; {time.perf_counter() - started:.0f} s", flush=True)
    # no airport is written while another's manifest changed since the start (each write checks its own again)
    for code in airports:
        require_overlays_unchanged(root / code / "training", code, overlay_id, existing[code])
    for code, (text, entry) in built.items():
        out = write_overlay(root / code / "training", code, entry, text, existing[code])
        print(f"  {code}: {out.stat().st_size / 1e6:.1f} MB → {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
