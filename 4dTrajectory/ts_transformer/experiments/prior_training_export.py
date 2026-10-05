"""B6: the Training export of stage B (prior design §12 B6, outline §6) — for each airport, a set of the flights of one
free-generation readout (`prior_free_generation`) that the frontend's Training view reads: each flight's observed track
and open-loop sentence (A23's payload), and every sentence the prior said, flown again here — its words, its flown
track and attitude, its outcome with the crossing and decision-altitude check, the probability of "go-around" and, at
each row, the words the procedure masks blocked; for each candidate the procedure's limits (the region, the glidepath
lower edge, the DA and the entry height).

WHO. ``--per-airport`` flights of each airport of the readout, drawn at random with ``--seed`` (D55), every sample of each.

CHECKED, NOT TRUSTED. Each sentence is flown again through the start of a closed loop (`autopilot.start`, D67) with its
own words, and must give the readout's states on every 2 s row (within the executor conformance's bound) and its outcome,
timeout and go-arounds. The executor spec opens, and the closed-loop sentences are read, only for the code that passes
the labeller's, the executor's and the closed loop's checks, run here first (`require_conforming_closed_loop`, D69,
D73). The readout's prior opens as every runner of the prior opens it (`checkpoint.open_prior`: today's artefact
identity, its procedure masks on today's procedure data) and must be the checkpoint and identity the readout records.
A readout of the val days is exported only when it is the one the prior's claim of its val read names (D85): the base's
one validation readout.

THE CLOSED-LOOP SENTENCE of each flight at the prior's Δ, with its head (the observed track, the open-loop sentence), is
A23's payload, built and flown again by A23's code (`training_export.split_flights`, A36; the prior's runner imports
from `autopilot/` only what D69 lists): checked against its stored states and outcome, on any split (vocabulary D86).

WRITES a set ``<root>/<airport>/training/<set-id>/sample.json`` and its entry in
``<root>/<airport>/training/index_prior_v2.json`` (`prior.training_files`); refused when the set exists. Every airport is
built before any is written. From a clean tree (the set records the commit) unless ``--smoke``.

    python run_ts.py prior_training_export --readout <a prior_free_generation directory> --set-id <id> --per-airport 10
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from flight_scenarios.fas_geometry import course_halfwidth_m
from ts_transformer.autopilot.closed_loop import require_conforming_closed_loop
from ts_transformer.autopilot.judge import ALT, LAT, LON, TIMEOUT, flown_track
from ts_transformer.autopilot.start import start
from ts_transformer.experiments import training_flights
from ts_transformer.experiments.prior_free_generation import FREE_GENERATION_SCHEMA, Stored, read_sentences
from ts_transformer.experiments.training_attitude import attitude_payload, executor_attitude
from ts_transformer.experiments.training_export import (
    FORMATS, candidate_hae_minus_msl_m, candidates_block, events, split_flights, vocabulary_block,
)
from ts_transformer.instructions import training_files as stage_a_files
from ts_transformer.instructions.airport import AirportGeometry, RunwayCandidate
from ts_transformer.instructions.artefact import STATE_COLUMNS, ClosedLoopSentence
from ts_transformer.instructions.grammar import column_words
from ts_transformer.instructions.labeller.interval import interval_rows
from ts_transformer.instructions.spec import READING_RULE
from ts_transformer.instructions.words import COLUMNS, UNCHANGED, Words
from ts_transformer.io_utils import file_sha256
from ts_transformer.prior import training_files as files
from ts_transformer.prior.checkpoint import CHECKPOINT_SCHEMA, holds_claim, open_prior, readable_identity, validation_claim
from ts_transformer.prior.procedure import (
    GLIDEPATH_BELOW_M, PROCEDURE_MASKS, Final, airport_finals,
)
from ts_transformer.prior.speaker import MOST_GO_AROUNDS
from ts_transformer.repo_layout import REPO_ROOT, git_state, repo_relative

#: The points along each final its region's outline and its glidepath lower edge are drawn with.
OUTLINE_POINTS = 61
#: The state column of the compass track (`STATE_COLUMNS`).
TRACK = STATE_COLUMNS.index("track_deg")


# ---- the procedure's limits (the views draw them, they compute none)
def runway_point(candidate: RunwayCandidate, before_m: np.ndarray, right_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The airport frame's east and north of the points ``before_m`` before the threshold along the course and ``right_m``
    right of it (the inverse of `instructions.airport.relative_to_runway`)."""
    course = math.radians(candidate.course_deg)
    along_e, along_n = math.sin(course), math.cos(course)
    return (candidate.threshold_e_m - before_m * along_e + right_m * along_n,
            candidate.threshold_n_m - before_m * along_n - right_m * along_e)


def placed(geometry: AirportGeometry, e: np.ndarray, n: np.ndarray) -> dict[str, list[float]]:
    lat, lon = geometry.frame.latlon_from_horizontal(e, n)
    return {"eM": stage_a_files.rounded(e, 1), "nM": stage_a_files.rounded(n, 1),
            "latDeg": stage_a_files.rounded(lat, 7), "lonDeg": stage_a_files.rounded(lon, 7)}


def procedure_block(finals: Sequence[Final]) -> list[dict[str, Any]]:
    """Each candidate's final as the procedure masks read it (`prior.procedure`; heights MSL): the outline of its region
    (inside the FAF and the LPV cone), the glidepath lower edge along the course inside it, the DA where the glidepath
    reaches it and the entry height at the FAF."""
    out = []
    for final in finals:
        geometry = final.geometry
        candidate = geometry.candidates[final.index]
        elevation = geometry.elevation_m
        before = np.linspace(0.0, final.faf_m, OUTLINE_POINTS)
        half = course_halfwidth_m(before, final.cone)
        ring_before = np.concatenate((before, before[::-1], before[:1]))
        ring_right = np.concatenate((half, -half[::-1], half[:1]))
        glidepath = final.glidepath_m(before)
        da_before = float(np.interp(final.decision_m, glidepath, before))
        on_course = np.zeros_like(before)
        out.append({
            "index": final.index, "ident": candidate.ident, "fafBeforeThresholdM": round(final.faf_m, 1),
            "glidepathBelowM": GLIDEPATH_BELOW_M,
            "region": placed(geometry, *runway_point(candidate, ring_before, ring_right)),
            "glidepathLowerEdge": {"beforeThresholdM": stage_a_files.rounded(before, 1),
                                   **placed(geometry, *runway_point(candidate, before, on_course)),
                                   "heightMslM": stage_a_files.rounded(glidepath - GLIDEPATH_BELOW_M + elevation, 1)},
            "decision": {"beforeThresholdM": round(da_before, 1), "heightMslM": round(final.decision_m + elevation, 1),
                         **placed(geometry, *runway_point(candidate, np.array([da_before]), np.zeros(1)))},
            "entry": {"beforeThresholdM": round(final.faf_m, 1), "heightMslM": round(final.entry_m + elevation, 1),
                      **placed(geometry, *runway_point(candidate, np.array([final.faf_m]), np.zeros(1)))}})
    return out


# ---- the prior's sentences, flown again
def blocked_payload(blocked: Mapping[int, np.ndarray], candidates: int, words: Words) -> dict[str, list[list[int]]]:
    """At each row, the words (vocabulary values) the procedure masks blocked, by column name."""
    out = {}
    for column, mask in blocked.items():
        classes = column_words(column, words, candidates)
        out[COLUMNS[column]] = [[int(classes[k]) for k in np.flatnonzero(row)] for row in mask]
    return out


def step_words(loop: Any, said: Sequence[np.ndarray]) -> list[list[np.ndarray]]:
    """Each flight of ``loop`` (`autopilot.start.Loop`, from its first predicted step) told its own words ``said`` a Δ row
    at a time, as free generation told them (`prior_free_generation.speak_and_fly`): a done flight is halted and told
    "unchanged" from then on; a flight must be done at its last word row, or it is refused. Each flight's flown 2 s rows,
    from its first predicted step's state, to the end of the row it was done in."""
    count = len(said)
    flown: list[list[np.ndarray]] = [[row] for row in loop.rows()]
    alive, t = np.ones(count, dtype=bool), 0
    while alive.any():
        # a flight still flying has a word for this row: one not done at its last word was refused there (below)
        row = np.stack([said[b][t] if t < len(said[b]) else np.full(len(COLUMNS), UNCHANGED) for b in range(count)])
        rows, done = loop.step(row)
        for b in np.flatnonzero(alive):
            flown[b] += list(rows[b])
        wrong = [b for b in np.flatnonzero(alive) if done[b] != (t == len(said[b]) - 1)]
        if wrong:
            raise ValueError(f"flight(s) {wrong[:3]} of the loop: done at another row than their last word when flown "
                             f"again (row {t})")
        alive &= ~done
        loop.halt(~alive)
        t += 1
    return flown


#: The bound on a state flown again, m. MIRROR of `autopilot.conformance.STATE_BOUND_M` (the executor conformance's
#: tolerance), which a runner of the prior may not import (D69); `tests/test_prior_training_export.py` pins it there.
STATE_BOUND_M = 1e-6


def apart(states: np.ndarray, stored: np.ndarray) -> float:
    """The largest difference of two state sequences (`STATE_COLUMNS`), the track wrapped (a track near north may read
    0° on one and 360° on the other); infinite when their shapes differ, NaN when either holds a NaN."""
    if states.shape != stored.shape:
        return math.inf
    difference = np.abs(states - stored)
    difference[:, TRACK] = np.abs((states[:, TRACK] - stored[:, TRACK] + 180.0) % 360.0 - 180.0)
    return float(difference.max())


def fly_again(instructions: Path, executor: Path, split: str, interval_s: float,
              sentences: Mapping[int, ClosedLoopSentence], stored: Sequence[Stored], words: Words, *,
              device: torch.device) -> list[dict[str, Any]]:
    """One sample's sentences (``stored``, each of its own flight) flown again through the start with their own words,
    each refused unless it gives its stored states, outcome, timeout and go-arounds (module docstring); their flown
    payloads, in the order of ``stored``."""
    loop, order = start(instructions, split, interval_s, {s.index: sentences[s.index] for s in stored}, executor,
                        most_go_arounds=MOST_GO_AROUNDS, device=device)
    by_index = {s.index: s for s in stored}
    every = interval_rows(interval_s, words.spec.step_s)
    first = sentences[order[0]].rows.start
    flown = step_words(loop, [by_index[i].words for i in order])
    ended = np.ceil((loop.executor.done_cycle.cpu().numpy() + 1) / loop.row_cycles).astype(int)
    executed = loop.executor.flown()
    aero = loop.executor.inputs.aero_params.cpu().numpy()
    out = {}
    for b, index in enumerate(order):
        item, sentence, geometry = by_index[index], sentences[index], loop.geometries[b]
        states = np.concatenate((sentence.rows.states[: first * every], np.array(flown[b][: ended[b] + 1])))
        distance = apart(states, item.states)
        if not distance <= STATE_BOUND_M:                  # a NaN state is refused too
            raise ValueError(f"{item.row['dataset_id']} sample {item.sample}: flown again {distance:.3g} from its "
                             f"readout's states")
        outcome = loop.outcome(b)
        timed_out = outcome.outcome == TIMEOUT                      # why it ended: the judge's (vocabulary D90)
        if (outcome.outcome, timed_out, int(loop.go_arounds[b])) != (
                item.row["outcome"], item.row["timed_out"], item.row["go_arounds"]):
            raise ValueError(f"{item.row['dataset_id']} sample {item.sample}: flown again to {outcome.outcome} "
                             f"(timeout {timed_out}, {int(loop.go_arounds[b])} go-arounds), the readout's "
                             f"{item.row['outcome']} ({item.row['timed_out']}, {item.row['go_arounds']})")
        # the flight to its outcome on the 2 s rows from the first predicted step, as A23 draws a replay
        last = training_flights.last_state_cycle(outcome.outcome, outcome.end_row)
        cycles = np.arange(0, last + 1, loop.row_cycles)
        whole = flown_track(executed.states[b, : last + 1].cpu().numpy(), geometry)
        at = executed.states[b, cycles].cpu().numpy()
        out[index] = {
            "sample": item.sample, "rows": len(item.words), "words": item.words.astype(int).tolist(),
            "events": events(item.words, None, geometry, words), "firstRow": sentence.rows.first_row, "startRow": first,
            # the row of the readout's states the executor flew from (the first predicted step)
            "flownFromRow": first * every,
            "outcome": outcome.outcome, "endCycle": int(outcome.end_row), "timedOut": timed_out,
            "goArounds": int(loop.go_arounds[b]),
            "crossing": training_flights.crossing_payload(outcome, executed, b, geometry),
            "track": {"rows": len(cycles), "lastCycle": int(last), "eM": stage_a_files.rounded(whole["e"][cycles], 1),
                      "nM": stage_a_files.rounded(whole["n"][cycles], 1), "latDeg": stage_a_files.rounded(at[:, LAT], 7),
                      "lonDeg": stage_a_files.rounded(at[:, LON], 7),
                      "heightMslM": stage_a_files.rounded(at[:, ALT], 1),
                      "trackDeg": stage_a_files.rounded(np.mod(whole["track"][cycles], 360.0), 2),
                      "groundSpeedMps": stage_a_files.rounded(whole["ground_speed"][cycles], 2),
                      "verticalRateMps": stage_a_files.rounded(whole["vertical_rate"][cycles], 2)},
            "attitude": attitude_payload(executor_attitude(executed, b, cycles, aero[b])),
            "goAroundProbability": stage_a_files.rounded(item.go_around_probability, 4),
            "goAroundPermitted": [int(v) for v in item.go_around_permitted],
            "onFinal": [int(v) for v in item.on_final],
            "blocked": blocked_payload(item.blocked, len(geometry.candidates), words)}
    return [out[s.index] for s in stored]


# ---- one airport
def chosen_flights(stored: Sequence[Stored], airport: str, per_airport: int, seed: int) -> list[int]:
    """At most ``per_airport`` of the airport's flights in the readout, drawn at random with ``seed`` (D55: never the
    first in order — the readout's flights are in the order of the signals, by callsign), by their place in the signals,
    in order."""
    pool = sorted({item.index for item in stored if item.row["airport"] == airport})
    order = np.random.default_rng(seed).permutation(len(pool))
    return sorted(pool[k] for k in order[:per_airport])


def build_airport(airport: str, readout: dict[str, Any], stored: Sequence[Stored], params: Any, words: Words, *,
                  per_airport: int, seed: int, device: torch.device) -> tuple[list[dict[str, Any]], AirportGeometry]:
    """The airport's flights (module docstring): each one's payload of A23 (`training_export.split_flights` at the
    prior's Δ: the head and the closed-loop sentence flown again) with its sentences of every sample."""
    instructions, executor = Path(readout["instructions"]), Path(readout["executor"])
    split, interval = readout["split"], float(readout["row_interval_s"])
    indices = chosen_flights(stored, airport, per_airport, seed)
    mine = [s for s in stored if s.index in indices]
    ids = [next(s.row["dataset_id"] for s in mine if s.index == i) for i in indices]
    heads, geometry = split_flights(instructions, split, ids, (interval,), params, words, device=device)
    stored_loop = training_flights.stored_closed_loop(instructions, split, interval, words)
    sentences = {i: training_flights.owned_sentence(stored_loop[i]) for i in indices}
    del stored_loop
    said: dict[int, list[dict[str, Any]]] = {i: [] for i in indices}
    for sample in sorted({s.sample for s in mine}):
        one = [s for s in mine if s.sample == sample]
        for s, payload in zip(one, fly_again(instructions, executor, split, interval, sentences, one, words,
                                             device=device)):
            said[s.index].append(payload)
    for head, index in zip(heads, indices):
        head["prior"] = said[index]
    return heads, geometry


# ---- the set's head
def model_block(readout: dict[str, Any], prior_dir: str) -> dict[str, Any]:
    """Which prior said the set's sentences and how (the readout's record)."""
    return {"prior": prior_dir, "run": readout["prior_run"], "checkpointSha256": readout["checkpoint_sha256"],
            "selection": readout["selection"], "rowIntervalS": float(readout["row_interval_s"]),
            "temperature": readout["temperature"], "mostGoArounds": readout["most_go_arounds"],
            "procedureMasks": readout["procedure_masks"]}


def source_block(readout_dir: str, readout: dict[str, Any], words: Words, opened: dict[str, Any], git: dict[str, Any],
                 *, smoke: bool, device: str, claim: dict[str, str] | None) -> dict[str, Any]:
    """What the set was exported from, and the checks the export ran (D73). The artefact and the executor spec are named
    relative to the repository (`repo_relative`): the live executor opens them from its own checkout, never from the
    worktree a readout happened to run in. ``claim``: the claim of the val read the set was exported under (outline
    D109: its flights are of val), None for every other set."""
    return {"readout": readout_dir, "readoutGit": readout["git"], "readoutSmoke": readout["smoke"],
            "validationClaim": claim,
            "instructions": repo_relative(Path(readout["instructions"])),
            "executor": repo_relative(Path(readout["executor"])), "specSha256": words.spec.sha256,
            "executorSpecSha256": opened["sha256"], "checks": opened["checks"], "git": git, "smoke": smoke,
            "device": device}


def cohort_block(readout: dict[str, Any], flights: int, per_airport: int, seed: int, readout_flights: int
                 ) -> dict[str, Any]:
    """Which of the readout's flights the set holds (no silent sample: how many, of how many, drawn how)."""
    return {"split": readout["split"], "flights": flights, "perAirport": per_airport, "seed": seed,
            "samples": readout["samples"], "readoutSeed": readout["seed"], "readoutFlights": readout_flights,
            "drawnFrom": "a seeded draw of the airport's flights in the free-generation readout (itself a seeded draw)"}


def sample_of(set_id: str, geometry: AirportGeometry, hae_minus_msl_m: dict[str, float], source: dict[str, Any],
              model: dict[str, Any], cohort: dict[str, Any], words: Words, cycle_s: float, finals: Sequence[Final],
              flights: list[dict[str, Any]]) -> dict[str, Any]:
    """A set's sample: its head (formats, source, the model, cohort, vocabulary, the airport frame, the candidates with
    their HAE − MSL, the procedure's limits) and its flights, each given its runway's HAE − MSL."""
    for flight in flights:
        flight["haeMinusMslM"] = hae_minus_msl_m[flight["runway"]]
    return {"schema": files.SAMPLE_SCHEMA, "setId": set_id, "airport": geometry.code, "readingRule": READING_RULE,
            "formats": {**FORMATS, "freeGeneration": FREE_GENERATION_SCHEMA, "checkpoint": CHECKPOINT_SCHEMA},
            "source": source, "model": model, "cohort": cohort, "vocabulary": vocabulary_block(words.spec, words),
            "executor": {"cycleS": cycle_s},
            "airportFrame": {"code": geometry.code, "lat": geometry.frame.lat0, "lon": geometry.frame.lon0,
                             "elevationM": geometry.elevation_m},
            "candidatesSha256": stage_a_files.candidates_sha256(geometry),
            "candidates": candidates_block(geometry, hae_minus_msl_m), "procedure": procedure_block(finals),
            "flights": flights}


def index_entry(set_id: str, sample: dict[str, Any]) -> dict[str, Any]:
    """A set as its airport's index lists it."""
    model = sample["model"]
    return {"id": set_id, "kind": files.SET_KIND, "readingRule": READING_RULE,
            "title": f"Stage B · prior ({model['selection']}) · free generation at Δ {model['rowIntervalS']:g} s · "
                     f"{sample['cohort']['split']}",
            "file": f"{set_id}/{files.SAMPLE_FILE}", "flights": len(sample["flights"]),
            "sentences": sum(len(flight["prior"]) for flight in sample["flights"]), "formats": sample["formats"],
            "model": model, "cohort": sample["cohort"], "source": sample["source"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--readout", type=Path, required=True, help="a prior_free_generation directory")
    parser.add_argument("--set-id", required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "aeroviz-4d" / "public" / "data" / "airports")
    parser.add_argument("--airports", nargs="+", default=None, help="default: every airport of the readout")
    parser.add_argument("--per-airport", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337, help="the draw of each airport's flights from the readout")
    parser.add_argument("--device", default="cpu", help="the executor's device when the sentences are flown again")
    parser.add_argument("--smoke", action="store_true", help="SMOKE: allowed from a tree with changes; recorded")
    args = parser.parse_args(argv)
    git = git_state()
    if git["dirty"] and not args.smoke:
        parser.error("the tree has uncommitted changes; a Training set is exported from a commit")
    readout_dir = args.readout if args.readout.is_absolute() else REPO_ROOT / args.readout
    readout, stored = read_sentences(readout_dir)
    if readout["smoke"] and not args.smoke:
        parser.error(f"{readout_dir} is a smoke readout: only a --smoke set is made from it")
    readout_airports = sorted({s.row["airport"] for s in stored})
    airports = args.airports or readout_airports
    unknown = sorted(set(airports) - set(readout_airports))
    if unknown:
        parser.error(f"airports {unknown} are not in the readout ({readout_airports})")
    instructions, executor = Path(readout["instructions"]), Path(readout["executor"])
    interval = float(readout["row_interval_s"])
    params, opened, words = require_conforming_closed_loop(instructions, executor)   # D69: the checks run here (D73)
    prior_dir = Path(readout["prior"])
    prior = open_prior(prior_dir, instructions)        # today's identity and procedure masks (§7 item 1, D106)
    geometries = prior.geometries
    if (readable_identity(prior.checkpoint.identity) != readout["identity"] or prior.interval_s != interval
            or file_sha256(prior_dir / "checkpoint.pt") != readout["checkpoint_sha256"]
            or readout["procedure_masks"] != PROCEDURE_MASKS):
        raise SystemExit(f"{readout_dir}: not said by {prior_dir}'s checkpoint on this artefact's identity under "
                         f"{PROCEDURE_MASKS} (its data or its prior changed since the readout)")
    claim = None
    if readout["split"] == "val":           # the base's one validation readout, and only it (D85)
        claimed = validation_claim(prior_dir, files.CLAIM_READER)
        if not holds_claim(prior_dir, files.CLAIM_READER, readout_dir):
            raise SystemExit(f"{readout_dir}: a readout of the val days that {prior_dir}'s claim of its val read does "
                             f"not name ({claimed}); only the base's one validation readout is exported (D85)")
        claim = {"reader": files.CLAIM_READER, "prior": repo_relative(prior_dir),
                 "readout": repo_relative(readout_dir)}
    signals_record = json.loads((instructions / "signals.json").read_text(encoding="utf-8"))
    started = time.perf_counter()
    existing = {airport: files.read_index(args.root / airport / "training", airport, args.set_id) for airport in airports}
    model = model_block(readout, repo_relative(prior_dir))
    source = source_block(repo_relative(readout_dir), readout, words, opened, git, smoke=args.smoke, device=args.device,
                          claim=claim)
    built = {}
    for airport in airports:
        flights, geometry = build_airport(airport, readout, stored, params, words, per_airport=args.per_airport,
                                          seed=args.seed, device=torch.device(args.device))
        hae = candidate_hae_minus_msl_m(signals_record["runway_ends_from"], geometry)
        cohort = cohort_block(readout, len(flights), args.per_airport, args.seed,
                              len({s.index for s in stored if s.row["airport"] == airport}))
        sample = sample_of(args.set_id, geometry, hae, source, model, cohort, words, params.cycle_s,
                           airport_finals(geometries[airport]), flights)
        entry = index_entry(args.set_id, sample)
        built[airport] = (entry, files.serialise(sample))
        print(f"{airport}: {len(flights)} flights, {entry['sentences']} sentences, "
              f"{len(built[airport][1]) / 1e6:.1f} MB, {time.perf_counter() - started:.0f}s", flush=True)
    for airport, (entry, _) in built.items():          # every airport writable before any is written
        files.require_writable(args.root / airport / "training", airport, entry, existing[airport])
    for airport, (entry, text) in built.items():
        out = files.write_set(args.root / airport / "training", airport, entry, text, existing[airport])
        print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
