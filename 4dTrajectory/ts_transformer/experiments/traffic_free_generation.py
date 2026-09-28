"""Multi-aircraft M3 (design §6.1, §6.6 step 4, §7): the start of the multi-aircraft post-training — the single-aircraft
prior (augmented) with a traffic attention at zero (`prior.model.with_traffic`: it answers as the prior does, to
rounding) — speaking to one aircraft of each scene in the closed loop, the others replayed (`traffic_speaking`): M4's
round 0. A readout only.

On the split's drawn flights (`autopilot.replay.draw`: flown on their own dynamics, ``--per-airport`` a seeded sample per
airport), each flight is read four ways, every one judged the same way from its first predicted step with the others of
its scene replayed (`traffic_speaking.judged`: the VISUAL reading ends it; IFR beside it):

- **scene** — the model speaking, ``--samples`` times: the others in its edge features, the separation masks on;
- **alone** — the same model with no other in its scene (nothing to read, no separation mask), ``--samples`` times: the
  prior as it speaks alone (single-aircraft free generation), judged against the same scene — how much separation comes
  for free;
- **labelled** — its labelled words flown by the executor from the first predicted step;
- **recorded** — its own record on the loop's steps.

Per source, pooled and per airport: the outcomes (the executor's judge's; `below_glidepath` under the procedure's
altitudes; `lost_separation` where a loss it answered for ended it first), the losses it was in per flight and per hour
flown — flown to its judged end (a loss that ends it ends what is counted) — by relation, and for the model's sources,
over its steps to that end, the separation masks. A flight already in a loss it answers for at its first predicted step
(the record's: every reading starts from the same state) is counted and left out of the summaries, as an augmented scene
starting in one is redrawn (design §5.3) — no reading can be blamed for it. The separation masks: the share of its steps where they took a word away, per
column, and the probability the model put on what the masks removed (the approach column's counts the vocabulary's own
rule too: a runway changed under a clearance takes the approach with it). The procedure's masks are the prior's own.

With ``--augment-seed`` every flight's scene is augmented instead (`traffic_augment`: the leader moved, the start moved,
a flight inserted, a third each, drawn until the scene qualifies; a flight with no qualifying draw is left out, counted)
and read as **scene** and **alone** only — a moved start has no record, and its time limit is stage 2's
(`prior.augment.TIMEOUT_FACTOR`); the readout adds each kind's.

Writes into a NEW directory ``flights.jsonl`` (every flight's rows, appended as each batch ends, so a run cut short
keeps what it read) and ``free_generation.json`` (the readout); ``--per-airport`` is the sample's size (the readout says
so). A batch holds at most ``--aircraft-steps`` scenes × their most aircraft × (their longest pre-roll + rows): the past
the model keeps is 6 KB an aircraft-step.

    python run_ts.py traffic_free_generation \\
        --prior 4dTrajectory/outputs/POOLED/prior/v3_stage2_clip_20260926/aug_s1337/round_07 \\
        --executor 4dTrajectory/outputs/POOLED/executor/<spec> \\
        --instructions 4dTrajectory/outputs/POOLED/instruction_language/v5_20260926 \\
        --split select --out 4dTrajectory/outputs/POOLED/traffic/free_generation_<date>
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ts_transformer.autopilot import replay
from ts_transformer.autopilot.flights import flight_inputs
from ts_transformer.autopilot.judge import outcome_of
from ts_transformer.experiments.prior_free_generation import (
    _physics, augmented_inputs, fly_reference, flight_rows, glidepath_stops, limits_s, reference_grid, said_rows,
    start_altitude_windows,
)
from ts_transformer.experiments.traffic_augment import KINDS, augment
from ts_transformer.prior.augment import Augmentation
from ts_transformer.prior.generate import rows_for
from ts_transformer.experiments.prior_train import PRIOR_CHECKPOINT_SCHEMA, load_prior, rosters
from ts_transformer.experiments.traffic_loop import LOST_SEPARATION, Controlled, recorded
from ts_transformer.experiments.traffic_speaking import (
    HISTORY_S, MASK_COLUMNS, Scene, SceneLoop, judged, scene_airports, scene_of, speaking_aircraft,
)
from ts_transformer.inference.scene_edges import EDGE_FEATURES
from ts_transformer.inference.separation import IFR, VISUAL
from ts_transformer.instructions.artefact import SPLITS
from ts_transformer.instructions.words import COLUMNS, RUNWAY, UNCHANGED, Words
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.prior.data import VARIANTS, airport_landings
from ts_transformer.prior.model import Prior, with_traffic
from ts_transformer.prior.scene import N_LOOK, hang, presence
from ts_transformer.repo_layout import REPO_ROOT, git_state, repo_relative

SCHEMA = "ts-traffic-free-generation-v1"
SOURCES = ("scene", "alone", "labelled", "recorded")
#: A batch's most aircraft-steps (module docstring): about 1.8 GB of the model's past on its d and layers.
AIRCRAFT_STEPS = 300_000


def separation_fields(scene: Scene, aircraft: Controlled, step_s: float) -> dict[str, Any]:
    """How the flight fared with the others of ``scene``, under each reading: its outcome, the loss that ended it, and
    the loss episodes it was in (per relation; how many it answered for; the least distance over the required)."""
    out: dict[str, Any] = {}
    for reading in (VISUAL, IFR):
        outcome, end, run = judged(scene, aircraft, step_s, reading)
        mine = [e for e in run.episodes if aircraft.key in e["pair"]]
        last = end["t_s"] if end is not None else aircraft.last_step_s
        if reading == VISUAL:
            out["starts_in_a_loss"] = end is not None and end["t_s"] <= aircraft.first_step_s
        out[reading] = {
            "outcome": outcome, "end": end, "episodes": len(mine), "flown_s": float(last - aircraft.first_step_s),
            "relations": dict(Counter(e["relation"] for e in mine)),
            "answered": sum(any(r["key"] == aircraft.key for r in e["responsible"]) for e in mine),
            "min_ratio": min((e["min_ratio"] for e in mine), default=None)}
    return out


def _mask_steps(masked: dict[int, list[np.ndarray]], j: int, steps: int) -> dict[str, int]:
    """Of flight ``j``'s first ``steps`` steps, how many had a separation mask take a word away, per column
    (`traffic_speaking.SceneLoop.separation_masked`)."""
    return {COLUMNS[c]: int(sum(bool(masked[c][k][j]) for k in range(steps))) for c in MASK_COLUMNS}


def judged_steps(fields: dict[str, Any], aircraft: Controlled, steps_said: int, step_s: float) -> int:
    """The steps said up to the judged end under VISUAL: to the step whose state a loss ended it at, else every one."""
    end = fields[VISUAL]["end"]
    if end is None:
        return steps_said
    return min(steps_said, int(round((end["t_s"] - aircraft.first_step_s) / step_s)))


def model_rows(model: Prior, batch: replay.Batch, scenes: Sequence[Scene], judge_scenes: Sequence[Scene], source: str,
               words: Words, params: Any, landings: Any, samples: int, *, generator: torch.Generator,
               temperature: float, procedure_masks: Any,
               moves: Sequence[Augmentation | None] | None = None) -> list[dict[str, Any]]:
    """Every flight of ``batch`` flown ``samples`` times, the model speaking in ``scenes`` and judged with the others of
    ``judge_scenes`` (the same scenes, or — alone — the scenes it was taken out of); ``moves``: each flight's moved start
    (`batch.signals` are already its moved rows; None: its own), and stage 2's time limit."""
    cpu, step_s = torch.device("cpu"), words.spec.step_s
    index = [j for j in range(len(batch.readings)) for _ in range(samples)]
    repeated = replay.subset(batch, index)
    runways, charts, approach = _physics(repeated, cpu)
    inputs = flight_inputs(repeated.series, device=cpu, anchor=N_LOOK)
    if moves is not None:
        inputs = augmented_inputs(inputs, repeated.geometries, [moves[j] for j in index])
    loop = SceneLoop(model, repeated.signals, repeated.geometries, inputs, runways, charts, approach,
                     limits_s(repeated, params, step_s, augmented=moves is not None), words, params,
                     landings, scenes=[scenes[j] for j in index], generator=generator, temperature=temperature,
                     procedure_masks=procedure_masks)
    while loop.running:
        loop.step()
    flown, said = loop.executor.flown(), loop.spoken.sentences()
    forbidden = {c: np.stack(masses, axis=1) for c, masses in loop.speaker.forbidden.items()}
    rows, grids, stops = said_rows(repeated, flown, said, forbidden, words, [j % samples for j in range(len(said))],
                                   procedure_masks)
    for j, row in enumerate(rows):
        grid = grids[j][: row["steps_said"]]
        pointer = grid[:, RUNWAY][grid[:, RUNWAY] != UNCHANGED]
        ended = outcome_of(flown, j, repeated.geometries[j], int(pointer[-1]), words.spec)
        scene = judge_scenes[index[j]]
        aircraft = speaking_aircraft(scene, flown, j, grid, ended.outcome, ended.end_row, ended.crossing,
                                     -1 if stops is None else int(stops.step[j]), step_s)
        fields = separation_fields(scene, aircraft, step_s)
        counted = judged_steps(fields, aircraft, row["steps_said"], step_s)
        row.update(source=source, others=len(scene.others), speaking_with=len(scenes[index[j]].others),
                   history_cut=sum(o.first_step < -loop.speaker.pre for o in loop.speaker.others[j]),
                   judged_steps=counted, mask_steps=_mask_steps(loop.separation_masked, j, counted),
                   masked_mass={COLUMNS[c]: float(forbidden[c][j, :counted].mean()) if counted else 0.0
                                for c in MASK_COLUMNS},
                   **fields)
    return rows


def labelled_rows(batch: replay.Batch, scenes: Sequence[Scene], words: Words, params: Any,
                  procedure_masks: Any) -> list[dict[str, Any]]:
    """Every flight's labelled words flown by the executor from the first predicted step, judged in its scene."""
    step_s = words.spec.step_s
    flown = fly_reference(batch, words, params, device=torch.device("cpu"))
    grids = [reference_grid(r.words) for r in batch.readings]
    stops = (glidepath_stops(flown, grids, batch.geometries, [procedure_masks.finals[g.code] for g in batch.geometries],
                             words) if procedure_masks.altitudes else None)
    rows = flight_rows(batch, flown, grids, words, "labelled", [None] * len(grids), None, stops)
    for j, row in enumerate(rows):
        ended = outcome_of(flown, j, batch.geometries[j], batch.readings[j].runway_index, words.spec)
        aircraft = speaking_aircraft(scenes[j], flown, j, grids[j], ended.outcome, ended.end_row, ended.crossing,
                                     -1 if stops is None else int(stops.step[j]), step_s)
        row.update(others=len(scenes[j].others), **separation_fields(scenes[j], aircraft, step_s))
    return rows


def recorded_rows(batch: replay.Batch, scenes: Sequence[Scene], words: Words) -> list[dict[str, Any]]:
    """Every flight along its own record on the loop's steps (`traffic_loop.recorded`, landed at its crossing read off
    its rows), from its first predicted step, judged in its scene."""
    step_s, rows = words.spec.step_s, []
    for flight, reading, geometry, scene in zip(batch.signals, batch.readings, batch.geometries, scenes):
        own = scene.track(scene.key)
        seen = presence(flight, len(reading.words), geometry)
        capture = int(np.searchsorted(seen.times_s, own.captured_s))
        whole = recorded(seen, flight, capture, replay.observed_landing_s(flight, reading, geometry), geometry,
                         scene.airport.flights.separation, scene.speaking.category, step_s)
        cut = slice(N_LOOK, None)
        aircraft = dataclasses.replace(
            whole, times_s=whole.times_s[cut], e_m=whole.e_m[cut], n_m=whole.n_m[cut], height_m=whole.height_m[cut],
            runway=whole.runway[cut], along_m=whole.along_m[cut], track_minus_course_deg=whole.track_minus_course_deg[cut],
            right_of_course_m=whole.right_of_course_m[cut], along_speed_mps=whole.along_speed_mps[cut],
            established=whole.established[cut])
        rows.append({"dataset_id": flight.dataset_id, "airport": flight.airport, "source": "recorded",
                     "outcome": aircraft.outcome, "others": len(scene.others),
                     **separation_fields(scene, aircraft, step_s)})
    return rows


def augmented_scenes(batch: replay.Batch, scenes: Sequence[Scene], airports: dict[str, Any], seed: int,
                     windows: dict[str, tuple[float, float]], step_s: float
                     ) -> tuple[replay.Batch, list[Scene], list[Augmentation | None], list[dict[str, Any]], int]:
    """Every flight's scene augmented (`traffic_augment.augment`, one generator from ``seed`` in the batch's order):
    ``(the flights kept — their rows moved where the start is —, their scenes, their moved starts, what was drawn, how
    many were left out)``."""
    rng = np.random.default_rng(seed)
    pools = {code: [k for k, f in airport.flights.flights.items() if f.presence.speaking]
             for code, airport in airports.items()}
    drawn = []
    for j, (scene, signals) in enumerate(zip(scenes, batch.signals)):
        inputs = flight_inputs(batch.series[j: j + 1], device=torch.device("cpu"), anchor=N_LOOK)
        drawn.append(augment(scene, signals, inputs, pools[signals.airport], rng, windows, step_s))
    kept = [j for j, a in enumerate(drawn) if a is not None]
    part = replay.subset(batch, kept)
    part = dataclasses.replace(part, signals=[drawn[j].signals for j in kept])
    return (part, [drawn[j].scene for j in kept], [drawn[j].augmentation for j in kept],
            [{"kind": drawn[j].kind, "draws": drawn[j].draws, **drawn[j].drawn} for j in kept], len(drawn) - len(kept))


def summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per source, the flights not starting in a loss (those counted): flights, the outcomes under VISUAL (and the lost-separation share under IFR), the losses per flight
    and per hour flown, their relations, and — the model's sources — the separation masks."""
    out: dict[str, Any] = {}
    for source in SOURCES:
        every = [r for r in rows if r["source"] == source]
        part = [r for r in every if not r["starts_in_a_loss"]]
        if not part:
            continue
        hours = sum(r[VISUAL]["flown_s"] for r in part) / 3600.0
        entry: dict[str, Any] = {
            "flights": len(part), "left_out_starting_in_a_loss": len(every) - len(part),
            "outcomes": {k: v / len(part) for k, v in Counter(r[VISUAL]["outcome"] for r in part).most_common()},
            "lost_separation": sum(r[VISUAL]["outcome"] == LOST_SEPARATION for r in part) / len(part),
            "lost_separation_ifr": sum(r[IFR]["outcome"] == LOST_SEPARATION for r in part) / len(part),
            "episodes_per_flight": sum(r[VISUAL]["episodes"] for r in part) / len(part),
            "episodes_per_hour": sum(r[VISUAL]["episodes"] for r in part) / hours if hours else None,
            "relations": dict(sum((Counter(r[VISUAL]["relations"]) for r in part), Counter())),
            "others_per_flight": sum(r["others"] for r in part) / len(part)}
        if "mask_steps" in part[0]:
            steps = sum(r["judged_steps"] for r in part)
            entry["mask_steps_share"] = {COLUMNS[c]: sum(r["mask_steps"][COLUMNS[c]] for r in part) / steps
                                         for c in MASK_COLUMNS}
            entry["masked_mass_per_step"] = {
                COLUMNS[c]: sum(r["masked_mass"][COLUMNS[c]] * r["judged_steps"] for r in part) / steps
                for c in MASK_COLUMNS}
            entry["history_cut"] = sum(r["history_cut"] for r in part)
        out[source] = entry
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--prior", type=Path, required=True, help="the single-aircraft prior it starts from (augmented)")
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True, help="the executor spec directory")
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--per-airport", type=int, default=400, help="0: every flight of the split on its own dynamics")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--aircraft-steps", type=int, default=AIRCRAFT_STEPS, help="a batch's most (module docstring)")
    parser.add_argument("--augment-seed", type=int, default=None, help="augmented scenes drawn with this seed")
    parser.add_argument("--device", default="cuda", help="the prior's; the executor flies on CPU")
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    args = parser.parse_args(argv)

    def resolved(path: Path) -> Path:
        return path if path.is_absolute() else REPO_ROOT / path

    prior_dir, instructions, executor_dir, out = map(resolved, (args.prior, args.instructions, args.executor, args.out))
    if out.exists():
        parser.error(f"{out} exists; a readout is never overwritten")
    git = git_state()
    started = time.perf_counter()
    params, record, words = replay.open_executor(executor_dir, instructions)
    single, payload, prior_config, own_masks = load_prior(prior_dir, instructions)
    if prior_config["smoke"] or payload["schema"] != PRIOR_CHECKPOINT_SCHEMA:
        parser.error(f"{prior_dir} is not a single-aircraft prior's formal run")
    torch.manual_seed(args.seed)                       # the traffic attention's weights: at zero it reads nothing
    model = with_traffic(single, EDGE_FEATURES).to(torch.device(args.device)).eval()
    landings = (airport_landings(instructions, rosters(instructions))
                if VARIANTS[model.config.variant].landing_context else None)
    batch = replay.draw(instructions, args.split, words.spec, words, per_airport=args.per_airport, seed=args.seed)
    airports, built = scene_airports(instructions, args.split, words.spec, model.config.airports, landings,
                                     model.config.max_rows)
    augmented = args.augment_seed is not None
    limits = limits_s(batch, params, words.spec.step_s, augmented=augmented)
    scenes = [scene_of(airports[g.code], s.dataset_id, limit, words.spec.step_s)
              for s, g, limit in zip(batch.signals, batch.geometries, limits)]
    moves: list[Augmentation | None] | None = None
    draws: list[dict[str, Any]] = [{} for _ in scenes]
    left_out = 0
    if augmented:
        batch, scenes, moves, draws, left_out = augmented_scenes(batch, scenes, airports, args.augment_seed,
                                                                 start_altitude_windows(instructions), words.spec.step_s)
        limits = limits_s(batch, params, words.spec.step_s, augmented=True)
        print(f"augmented scenes: {dict(Counter(d['kind'] for d in draws))}, {left_out} with no qualifying draw left "
              f"out", flush=True)
    print(f"{len(batch.readings)} {args.split} flights ({batch.drawn['excluded']} not flown), scenes built "
          f"({built}), {time.perf_counter() - started:.0f}s", flush=True)

    generator = torch.Generator(device=torch.device(args.device)).manual_seed(args.seed)
    step_s = words.spec.step_s
    history = int(HISTORY_S // step_s)

    def size(j: int) -> tuple[int, int]:
        """A flight's scene: its aircraft and its steps (its pre-roll, as the speaker caps it, and its rows)."""
        scene = scenes[j]
        pre = max([0] + [int(round((scene.first_step_s - hang(scene.rows(k).presence.start_s, step_s))
                                   / step_s)) for k in scene.others])
        return 1 + len(scene.others), min(pre, history) + rows_for(limits[j] + step_s, step_s)

    order = sorted(range(len(scenes)), key=lambda j: (size(j), len(batch.readings[j].words)))
    batches: list[list[int]] = [[]]
    for j in order:
        grown = batches[-1] + [j]
        if batches[-1] and (len(grown) * args.samples * max(size(k)[0] for k in grown)
                            * max(size(k)[1] for k in grown)) > args.aircraft_steps:
            batches.append([j])
        else:
            batches[-1] = grown
    out.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    done = 0
    for chunk in batches:
        part, here = replay.subset(batch, chunk), [scenes[j] for j in chunk]
        alone = [dataclasses.replace(s, others=(), moved=()) for s in here]
        chunk_moves = None if moves is None else [moves[j] for j in chunk]
        new: list[dict[str, Any]] = []
        for source, speaking in (("scene", here), ("alone", alone)):
            rows_here = model_rows(model, part, speaking, here, source, words, params, landings, args.samples,
                                   generator=generator, temperature=args.temperature, procedure_masks=own_masks,
                                   moves=chunk_moves)
            for k, row in enumerate(rows_here):
                row["augmented"] = draws[chunk[k // args.samples]] or None
            new += rows_here
        if not augmented:
            new += labelled_rows(part, here, words, params, own_masks)
            new += recorded_rows(part, here, words)
        with (out / "flights.jsonl").open("a", encoding="utf-8") as stream:
            for row in new:
                stream.write(json.dumps(row) + "\n")
        rows += new
        done += len(chunk)
        print(f"  {done}/{len(order)} flights ({len(batches)} batches), {time.perf_counter() - started:.0f}s", flush=True)

    by_airport: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_airport[row["airport"]].append(row)
    readout = {"pooled": summary(rows), "airports": {code: summary(part) for code, part in sorted(by_airport.items())}}
    if augmented:
        readout["kinds"] = {kind: summary([r for r in rows if r["augmented"]["kind"] == kind]) for kind in KINDS}
    write_json_atomic(out / "free_generation.json", {
        "schema": SCHEMA, "written_utc": utc_now(), "git": git, "split": args.split, "drawn": batch.drawn,
        "per_airport": args.per_airport, "samples": args.samples, "temperature": args.temperature, "seed": args.seed,
        "prior": {"directory": repo_relative(prior_dir), "procedure_masks": list(own_masks.names)},
        "traffic_attention": "zero (with_traffic): the prior's answers to rounding",
        "executor": {"directory": repo_relative(executor_dir), "sha256": record["sha256"]},
        "instructions": repo_relative(instructions), "scenes": built, "history_s": HISTORY_S,
        "readings": {"ends": VISUAL, "beside": IFR}, "aircraft_steps": args.aircraft_steps, "batches": len(batches),
        "augment_seed": args.augment_seed, "augmented_left_out": left_out,
        "readout": readout, "flights_file": "flights.jsonl", "elapsed_s": time.perf_counter() - started})
    for source, entry in readout["pooled"].items():
        shares = "  ".join(f"{name} {share:.3f}" for name, share in entry["outcomes"].items())
        print(f"{source:9s} n={entry['flights']:5d}  lost separation {entry['lost_separation']:.3f} "
              f"(IFR {entry['lost_separation_ifr']:.3f})  episodes/h {entry['episodes_per_hour']}  {shares}")
        if "mask_steps_share" in entry:
            print(f"          masks: steps {entry['mask_steps_share']}  mass {entry['masked_mass_per_step']}")
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
