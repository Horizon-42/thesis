"""The prior's OWN sentences over a Training set's flights, for the frontend: each flight flown from its first predicted
step with the prior speaking and the executor flying (single-aircraft free generation, prior design §9.1) — the words it
said, the flown track and how the flight ended — ``--samples`` times (the Training module:
`aeroviz-4d/docs/36-2026-09-20-training-module.zh.md`).

    python run_ts.py prior_generation_training_export \\
        --prior 4dTrajectory/outputs/POOLED/prior/<the prior run, or one round of a post-training run> \\
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
end). One generator, seeded, draws every sample of every airport in the order the airports are named — on the speaker's
``--device`` (cpu by default; cuda as the formal readout runs it, other samples than cpu's), recorded in
``producedBy.device``; the executor flies on CPU whatever the device, as the live backend flies a sample again.

**The prior speaks as it was trained to**: under the vocabulary's rules and its OWN procedure's masks (`prior.masks`,
recorded beside its checkpoint — none for base and landing, the procedure's altitudes for augmented), read as the formal
readout reads a free sentence (`prior_free_generation.said_rows`): under the procedure's altitudes a sentence stops at
the first flown step more than the track tolerance below the glidepath lower edge (outcome `BELOW_GLIDEPATH`; nothing
said after it counts, its track ends at that step's end state). The payload names the sets and the digest of the data
each read, which the live backend checks before it flies a sample again (`aeroviz_backend/autopilot_segment`). A
``--readout`` must have been drawn under the same masks.

**Which model it is, by name** (`MODEL_NAMES`, the post-training design's table): a prior trained on data alone is
``base``; a post-trained round is named by the method that trained it (`fine_tuning.schema` less its version,
`METHOD_MODELS`; M4 in windows by how a window's aircraft were commanded, `WINDOW_MODELS`), with its round, the run holding its rounds and the model it started from, named the same way — so the
frontend can switch between the rounds of every stage. A method with no name is refused: a new stage's name is agreed
with the user and added here.

``--readout`` (optional) carries the prior's formal val readout beside the set's own flights: the landed share in all
and per approach kind, at the payload's airport and pooled over every airport the readout drew, refused unless it is a
free generation of THIS prior (its directory from ``4dTrajectory/outputs/`` on — the readouts were run from worktrees
whose outputs are the same tree), on this executor spec, this artefact, val, with the same samples and temperature.

What is written per sample: the words said as events (``row``: the flight's own step, the first `N_LOOK` observed), the
flown track every 2 s step on the flight's own clock (``tS`` from row `N_LOOK`'s time) to its outcome's row — a dynamics
failure to the row before, as the executor's replay export keeps it — in MSL and in the ellipsoid height Cesium draws in
(plus the flight's runway's HAE − MSL offset, `training_files.runway_hae_minus_msl_m`), the outcome, the crossing, the
runway pointed first and last, and the probability the prior put on what the masks removed. The words carry no
verdicts: the judge would say how the EXECUTOR flew the prior's words, not what the prior said; the landing is the
prior's answer. SI units.

**From augmented starts** (``--augment-seed``; the Training module §2.8, post-training design §4): each flight flies not
from its own start but from an augmented one — the same code as stage 2 and its val readouts
(`prior_free_generation.augmented_starts`, `prior.augment`): rotated about the airport, raised, sped up, drawn until
plausible, at most `AUGMENT_TRIES` draws; a flight with no plausible draw is not flown. The draw is the flyable flights'
in the set's order, from a generator seeded with ``--augment-seed`` afresh at each airport and apart from the samples' —
so every model exported with one seed over one set and artefact flies each flight from the same moved start. The prior
reads the moved observed rows and the executor starts from the moved state; the time limit is the source's observed
remaining time × `augment.TIMEOUT_FACTOR`, as in training. Written as a kind of its own (`KIND_AUGMENTED_GENERATION`,
`AUGMENTED_SCHEMA`): each flight adds its augmentation and the moved observed rows the prior read (rows 0 to
`N_LOOK` − 1; the samples' tracks start at row `N_LOOK`). No formal readout: stage 2's augmented val readouts drew their
own moves.

**The words run to where the executor stopped, the track to the outcome**: `flight_rows` counts a sentence to the step
whose cycles ended the flight for the executor, which flies on after three outcomes its judge reads earlier — crossing
the threshold without the capture, crossing another runway's threshold, and the stall cut-off (the executor stops at none
of them) — so a sample's last
words can follow its ``endS``. They are written as the formal readout counts them (its runway pointed at the end is read
from them); the frontend shades the rows after the end.

"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ts_transformer.autopilot import replay
from ts_transformer.autopilot.executor import Flown
from ts_transformer.autopilot.flights import flight_inputs, rebuild_series
from ts_transformer.autopilot.frame import ALT, LAT, LON
from ts_transformer.autopilot.judge import flown_track, outcome_of
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.autopilot.runway_data import published_vertical_paths
from ts_transformer.experiments.prior_augmented_reward import AUGMENTED_REWARD_SCHEMA
from ts_transformer.experiments.prior_free_generation import (
    AUGMENT_TRIES, BELOW_GLIDEPATH, GENERATION_SCHEMA, _physics, augmented_inputs, augmented_starts, limits_s, said_rows,
    speak_and_fly, start_altitude_windows,
)
from ts_transformer.experiments.prior_landing_reward import LANDING_REWARD_SCHEMA
from ts_transformer.experiments.prior_train import rosters
from ts_transformer.experiments.traffic_window_reward import SCHEMA as TRAFFIC_WINDOW_REWARD_SCHEMA
from ts_transformer.experiments.training_attitude import attitude_payload, executor_attitude
from ts_transformer.experiments.prior_training_export import open_trained_prior
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.artefact import load_candidates, load_sentences, load_signals
from ts_transformer.instructions.labeller.read import read_flight
from ts_transformer.instructions.readout import STRATA
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.training_files import (
    KIND_AUGMENTED_GENERATION, KIND_GENERATION, KIND_READBACK, SPLIT, BaseSet, base_flights, open_base_set, overlay_entry, read_overlays,
    require_overlays_unchanged, require_set_datum, rounded, runway_hae_minus_msl_m, serialise, write_overlay,
)
from ts_transformer.instructions.words import COLUMNS, UNCHANGED, Words
from ts_transformer.io_utils import utc_now
from ts_transformer.prior import augment
from ts_transformer.prior.augment import Augmentation, augment_signals
from ts_transformer.prior.data import VARIANTS, airport_landings
from ts_transformer.prior.masks import ProcedureMasks
from ts_transformer.prior.model import Prior
from ts_transformer.prior.scene import N_LOOK
from ts_transformer.repo_layout import REPO_ROOT, arrival_manifest_path, git_state, repo_relative

#: MIRROR of `aeroviz-4d/src/data/trainingOverlays.ts` (`TRAINING_GENERATION_SCHEMA`); the reader refuses anything else
#: by name. A name changes with its file's shape or meaning, on both sides, in one change. v4 (2026-09-28): ``base``
#: records the set's spec, candidates and frame, no longer its sample file's sha256 or time of writing
#: (`training_files.BaseSet.block`). v5 (2026-09-30): every sample's flown track carries the attitude it is drawn in
#: (``track.attitude``, `training_attitude`).
SCHEMA = "aeroviz-training-generation-v5"
#: MIRROR of `TRAINING_AUGMENTED_GENERATION_SCHEMA` in the same file: the same sentences flown from augmented starts, each
#: flight with its augmentation and moved observed rows, no formal readout; v2 (2026-09-28): ``base`` as `SCHEMA`'s v4;
#: v3 (2026-09-30): the attitude as `SCHEMA`'s v5.
AUGMENTED_SCHEMA = "aeroviz-training-augmented-generation-v3"
PAYLOAD_FILE = "generation.json"
RUNNER = "ts_transformer.experiments.prior_generation_training_export"
#: The one tree every checkout's outputs are (a worktree links it): a readout's prior is known by its path from here on.
OUTPUTS_MARK = "4dTrajectory/outputs/"
#: MIRROR of the phrase `replay.draw_flights` writes for ``per_airport`` 0 (every labelled flight of the split).
EVERY_FLIGHT = "every labelled flight"
#: The prior's models by name, in the order they are trained (the post-training design's table; the user, 2026-09-26;
#: ``traffic``, the multi-aircraft post-training M4, the user 2026-09-30; ``window``, M4 in windows — every aircraft of a
#: window commanded — the user 2026-10-01):
#: MIRROR of `TRAINING_MODEL_NAMES` in `aeroviz-4d/src/data/trainingOverlays.ts`, which orders the views by it.
MODEL_NAMES = ("base", "landing", "augmented", "traffic", "window")


def method_of(schema: str) -> str:
    """A post-training method's name: its checkpoint schema less the version (``ts-prior-landing-reward-v3`` →
    ``ts-prior-landing-reward``) — every version of a method trains the same model."""
    match = re.fullmatch(r"(.+)-v\d+", schema)
    if match is None:
        raise ValueError(f"{schema!r} is not a versioned post-training schema")
    return match.group(1)


#: MIRROR of the checkpoint schema of M4's first runner (R32, archived: `archive/one_commanded_scene_2026_10/experiments/
#: traffic_reward.py`, nothing imports the archive), whose rounds are ``traffic``.
TRAFFIC_REWARD_SCHEMA = "ts-traffic-reward-v1"
#: The model a post-training method makes (``base`` has none: it is trained on data alone); M4 in windows (R37) makes
#: two, by `WINDOW_MODELS`.
METHOD_MODELS = {method_of(LANDING_REWARD_SCHEMA): "landing", method_of(AUGMENTED_REWARD_SCHEMA): "augmented",
                 method_of(TRAFFIC_REWARD_SCHEMA): "traffic"}
#: R37's rounds by how a window's aircraft were commanded (its ``fine_tuning.commanded``, `traffic_window.COMMANDED`):
#: one a window is M4's own setting, so ``traffic`` (the user, 2026-10-03: multi-aircraft step 9.4's model); every
#: aircraft ``window`` (the user, 2026-10-01).
WINDOW_MODELS = {"one": "traffic", "every": "window"}


def model_identity(config_file: dict[str, Any]) -> tuple[str, int | None]:
    """A prior's name and round from its config file: ``base`` and no round without a ``fine_tuning`` block, else the
    name of the method that post-trained it and the round it wrote."""
    if "fine_tuning" not in config_file:
        return "base", None
    tuning = config_file["fine_tuning"]
    method = method_of(tuning["schema"])
    if method == method_of(TRAFFIC_WINDOW_REWARD_SCHEMA):
        if "commanded" not in tuning:
            raise ValueError(f"a {tuning['schema']} round written before its rounds recorded how a window's aircraft "
                             f"were commanded (multi-aircraft step 9.4): its model is not named")
        return WINDOW_MODELS[tuning["commanded"]], tuning["round"]
    if method not in METHOD_MODELS:
        raise ValueError(f"no model is named for {tuning['schema']}: a new stage's name is agreed with the user and "
                         f"added to MODEL_NAMES / METHOD_MODELS")
    return METHOD_MODELS[method], tuning["round"]


def display_name(name: str, round_number: int | None) -> str:
    """How a model is called: ``base``, ``landing r1``, ``augmented r3``."""
    return name if round_number is None else f"{name} r{round_number}"


def model_block(prior_dir: Path, config_file: dict[str, Any], checkpoint_sha: str, variant: str) -> dict[str, Any]:
    """Who the prior is: its name and round, the run holding its rounds (``base``: its own directory), its checkpoint,
    and — post-trained — the method's schema and the model it started from, named from that model's own config file
    (read in ``prior_dir``'s outputs tree: the rounds were run from worktrees whose outputs are the same tree)."""
    name, round_number = model_identity(config_file)
    fine_tuning = None
    if name != "base":
        tuning = config_file["fine_tuning"]
        start = outputs_path(tuning["from"])
        start_config = in_tree_of(prior_dir, start) / "config.json"
        start_name, start_round = model_identity(json.loads(start_config.read_text(encoding="utf-8")))
        fine_tuning = {"schema": tuning["schema"], "from": start, "fromName": start_name, "fromRound": start_round}
    return {"name": name, "round": round_number,
            "run": outputs_path(prior_dir if round_number is None else prior_dir.parent),
            "checkpointSha256": checkpoint_sha, "variant": variant, "trainedAt": config_file["git"],
            "fineTuning": fine_tuning}


def generation_block(samples: int, temperature: float, seed: int, step_s: float, procedure_masks: ProcedureMasks,
                     executor_sha256: str, params: ExecutorParams) -> dict[str, Any]:
    """How the prior's sentences were drawn, as an overlay of them records it (this runner's, and a window set's
    `window_training_export`): the samples, temperature and seed, the first predicted row, the procedure's masks it
    spoke under — its own (`prior.masks`), each set with the digest of the data it read — and the executor that flew
    them."""
    digests = procedure_masks.data_sha256()
    return {"samples": samples, "temperature": temperature, "seed": seed, "firstPredictedRow": N_LOOK, "stepS": step_s,
            "procedureMasks": [{"name": mask, "dataSha256": digests[mask]} for mask in procedure_masks.names],
            "executor": {"specSha256": executor_sha256, "wordClock": params.word_clock, "cycleS": params.cycle_s,
                         "timeoutFactor": params.timeout_factor}}


def outputs_path(path: str | Path) -> str:
    """A path read from ``4dTrajectory/outputs/`` on — the same artefact from any checkout (worktrees link the tree)."""
    text = Path(path).as_posix()
    if OUTPUTS_MARK not in text:
        raise ValueError(f"{text} is not under {OUTPUTS_MARK}")
    return text[text.rindex(OUTPUTS_MARK):]


def in_tree_of(path: Path, other: str) -> Path:
    """``other`` (a path from `OUTPUTS_MARK` on) in the outputs tree ``path`` is in."""
    text = path.as_posix()
    if OUTPUTS_MARK not in text:
        raise ValueError(f"{text} is not under {OUTPUTS_MARK}")
    return Path(text[: text.rindex(OUTPUTS_MARK)]) / outputs_path(other)


def readout_block(generation: dict[str, Any], prior_dir: Path, executor_sha256: str, instructions: Path, samples: int,
                  temperature: float, airport: str,
                  procedure_altitudes: bool) -> dict[str, Any]:
    """The prior's formal val free generation (`prior_free_generation`'s ``generation.json``) as the frontend reads it —
    the landed share in all and per approach kind, at ``airport`` (``here``) and over every airport the readout drew
    (``all``), of the prior's sentences and of the labelled words flown from the same row — refused unless it is this
    prior's, on this executor spec and artefact, val, with these samples and this temperature, drawn under the
    procedure's altitudes exactly when the export is (``procedure_altitudes``: the model's own)."""
    if generation["schema"] != GENERATION_SCHEMA:
        raise ValueError(f"the readout is a {generation['schema']} file, not {GENERATION_SCHEMA}")
    # the export speaks under the model's own procedure's masks: a readout drawn with the procedure's altitudes where the
    # model has none (or without them where it has them) is another generation — the readout records only that flag
    wanted = {"prior": outputs_path(prior_dir), "executor": executor_sha256,
              "instructions": outputs_path(instructions), "split": SPLIT, "n_look": N_LOOK, "samples": samples,
              "temperature": temperature, "procedure_masks": procedure_altitudes}
    found = {"prior": outputs_path(generation["prior"]["directory"]),
             "executor": generation["executor"]["sha256"], "instructions": outputs_path(generation["instructions"]),
             "split": generation["split"], "n_look": generation["n_look"], "samples": generation["samples"],
             "temperature": generation["temperature"], "procedure_masks": generation["procedure_masks"]}
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
                  start_s: float, hae_minus_msl_m: float, aero_params: np.ndarray) -> dict[str, Any]:
    """The flown track every sentence step from its first state to its outcome's row (a dynamics failure: to the row
    before, as the replay export keeps it — the failed state may not be finite), on the flight's own clock
    (``start_s``: the time of the row the executor started at); ``hae_minus_msl_m``: the flight's runway's; and the
    attitude it is drawn in (``aero_params``: its airframe's, `training_attitude.executor_attitude`)."""
    end = end_row - 1 if outcome == "dynamics_failure" else end_row
    states = flown.states[index, : end + 1].cpu().numpy()
    step_rows = int(round(step_s / flown.cycle_s))
    rows = list(range(0, end + 1, step_rows))
    if rows[-1] != end:
        rows.append(end)
    track = flown_track(states[rows], geometry)
    lat, lon, height = states[rows, LAT], states[rows, LON], states[rows, ALT]
    return {"tS": rounded(start_s + np.asarray(rows) * flown.cycle_s, 3), "lon": rounded(lon, 7), "lat": rounded(lat, 7),
            "altitudeM": rounded(height, 2), "altitudeHaeM": rounded(height + hae_minus_msl_m, 2),
            "groundSpeedMps": rounded(track["ground_speed"], 3),
            "attitude": attitude_payload(executor_attitude(flown, index, rows, aero_params))}


def sample_payload(flown: Flown, index: int, row: dict[str, Any], grid: np.ndarray, geometry: AirportGeometry,
                   words: Words, hae_minus_msl_m: float, aero_params: np.ndarray, stop: int = -1) -> dict[str, Any]:
    """One sample as the frontend reads it: `flight_rows`'s ``row`` of it (its outcome and bookkeeping), the words
    it said up to its end as events on the flight's own steps (``grid``: [steps, 6], UNCHANGED where a column says
    nothing; step 0 is row `N_LOOK`), the crossing and the flown track (``hae_minus_msl_m``: the flight's runway's).
    ``stop``: the step the glidepath lower edge stopped it at (`said_rows`; -1: it did not) — no crossing, the track to
    that step's end state."""
    step_s = words.spec.step_s
    said = np.asarray(grid)[: row["steps_said"]]
    if (said[0] == UNCHANGED).any():
        raise ValueError(f"{row['dataset_id']}: the first predicted step does not say every column")
    # the crossing and the outcome's row: `flight_rows` read them with this call and kept only the outcome and the time
    outcome = outcome_of(flown, index, geometry, row["last_runway"], words.spec)
    step_rows = int(round(step_s / flown.cycle_s))
    crossing, end_row = (outcome.crossing, outcome.end_row) if stop < 0 else (None, (stop + 1) * step_rows)
    if (stop >= 0) != (row["outcome"] == BELOW_GLIDEPATH):
        raise ValueError(f"{row['dataset_id']}: outcome {row['outcome']} with the glidepath stop at step {stop}")
    start_s = N_LOOK * step_s
    return {
        "sample": row["sample"], "outcome": row["outcome"], "endS": round(start_s + row["end_s"], 3),
        "crossing": None if crossing is None else {"crossM": round(crossing["cross_m"], 2),
                                                   "heightM": round(crossing["height_m"], 2),
                                                   "atS": round(start_s + crossing["at_row"] * flown.cycle_s, 3),
                                                   "runway": crossing["runway_index"]},
        "firstRunway": row["first_runway"], "lastRunway": row["last_runway"], "runwayChanges": row["runway_changes"],
        "goArounds": row["go_arounds"], "clearedAtEnd": row["cleared_at_end"],
        "forbiddenMass": {column: round(mass, 6) for column, mass in row["forbidden_mass"].items()},
        "rows": N_LOOK + len(said),
        "events": [{"row": N_LOOK + int(step), "column": int(column), "value": int(said[step, column])}
                   for step, column in zip(*np.nonzero(said != UNCHANGED))],
        "track": track_payload(flown, index, end_row, row["outcome"], geometry, step_s, start_s, hae_minus_msl_m,
                               aero_params),
    }


def observed_payload(signals: FlightSignals, geometry: AirportGeometry, hae_minus_msl_m: float) -> dict[str, Any]:
    """The observed rows the prior reads before it speaks (rows 0 to `N_LOOK` − 1), on the flight's own clock, in MSL and
    in the ellipsoid height Cesium draws in (``hae_minus_msl_m``: the flight's runway's)."""
    rows = slice(0, N_LOOK)
    lat, lon = geometry.frame.latlon_from_horizontal(np.asarray(signals.e_m[rows], dtype=np.float64),
                                                     np.asarray(signals.n_m[rows], dtype=np.float64))
    height = np.asarray(signals.altitude_m[rows], dtype=np.float64)
    return {"tS": rounded(signals.time_s[rows], 3), "lon": rounded(lon, 7), "lat": rounded(lat, 7),
            "altitudeM": rounded(height, 2), "altitudeHaeM": rounded(height + hae_minus_msl_m, 2)}


def augmentation_payload(move: Augmentation) -> dict[str, float]:
    """The move as drawn, unrounded: the live backend flies a sample again from it, and a rotation rounded to 0.0001°
    already moves a start ~20 km out by ~2 cm (the flight would no longer be the sample's own to the centimetre)."""
    return {"rotationDeg": move.rotation_deg, "altitudeM": move.altitude_m, "speedScale": move.speed_scale}


def build_airport(base: BaseSet, flights: list[FlightSignals], sentences: dict[str, np.ndarray], instructions: Path,
                  geometry: AirportGeometry, model: Prior, params: ExecutorParams, words: Words, landings: Any,
                  samples: int, *, generator: torch.Generator, temperature: float,
                  procedure_masks: ProcedureMasks, augment_seed: int | None = None,
                  windows: dict[str, tuple[float, float]] | None = None) -> list[dict[str, Any]]:
    """Every flight of the set: those the val readout flies (their own dynamics) flown ``samples`` times with the
    prior speaking under ``procedure_masks`` (its own), in one batch; the rest listed with the reason. With
    ``augment_seed``, each flies from an augmented start (module docstring; ``windows``: `start_altitude_windows`) — one
    with no plausible draw is not flown — and each flight carries its augmentation and moved observed rows."""
    spec = words.spec
    offsets = runway_hae_minus_msl_m(instructions, geometry.code, arrival_manifest_path(geometry.code))
    located = base_flights(base, flights, sentences)
    signals = [flight for flight, _ in located]
    require_set_datum(base, signals, offsets)
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
    cpu = torch.device("cpu")
    batch = replay.Batch(signals=[signals[j] for j in flyable], series=[series[j] for j in flyable],
                         readings=readings, geometries=[geometry] * len(flyable),
                         vertical_paths=[published_vertical_paths(geometry)] * len(flyable),
                         approach_ias_mps=[replay.flight_approach_ias_mps(series[j], replay.OWN) for j in flyable],
                         groups=[replay.OWN] * len(flyable), drawn={})
    # from augmented starts: one move a flyable flight, drawn afresh at each airport; a flight none fits is not flown
    moves: list[Augmentation | None] = [None] * len(flyable)
    draws: list[int] = [0] * len(flyable)
    if augment_seed is not None and flyable:
        drawn = augmented_starts(batch.signals, flight_inputs(batch.series, device=cpu, anchor=N_LOOK),
                                 np.random.default_rng(augment_seed), windows)
        moves, draws = drawn.moves, drawn.draws
    kept = [n for n in range(len(flyable)) if augment_seed is None or moves[n] is not None]
    by_flight: dict[int, list[dict[str, Any]]] = {flyable[n]: [] for n in kept}
    if kept:
        index = [n for n in kept for _ in range(samples)]
        repeated = replay.subset(batch, index)
        inputs = flight_inputs(repeated.series, device=cpu, anchor=N_LOOK)
        if augment_seed is not None:
            repeated = replace(repeated, signals=[augment_signals(s, moves[n]) for s, n in zip(repeated.signals, index)])
            inputs = augmented_inputs(inputs, repeated.geometries, [moves[n] for n in index])
        runways, charts, approach = _physics(repeated, cpu)
        limits = limits_s(repeated, params, spec.step_s, augmented=augment_seed is not None)
        flown, said, forbidden, _ = speak_and_fly(model, repeated.signals, repeated.geometries, inputs, runways,
                                                  charts, approach, limits, words, params, landings,
                                                  generator=generator, temperature=temperature,
                                                  procedure_masks=procedure_masks)
        rows, grids, stops = said_rows(repeated, flown, said, forbidden, words, [i % samples for i in range(len(said))],
                                       procedure_masks)
        for i, row in enumerate(rows):
            j = flyable[index[i]]
            by_flight[j].append(sample_payload(flown, i, row, grids[i], geometry, words, offsets[signals[j].runway],
                                               inputs.aero_params[i].numpy(), -1 if stops is None else int(stops.step[i])))
    payloads = [{"flightKey": item["flightKey"], "datasetId": item["datasetId"], "group": groups[j],
                 "flown": j in by_flight, "samples": by_flight.get(j, [])}
                for j, item in enumerate(base.sample["flights"])]
    if augment_seed is not None:
        at = {j: n for n, j in enumerate(flyable)}
        for j, payload in enumerate(payloads):
            move = moves[at[j]] if j in at else None
            # null draws: not drawn (not on its own dynamics); null augmentation: none of the draws fitted
            payload["augmentDraws"] = draws[at[j]] if j in at else None
            payload["augmentation"] = None if move is None else augmentation_payload(move)
            payload["observed"] = None if move is None else observed_payload(augment_signals(signals[j], move), geometry,
                                                                             offsets[signals[j].runway])
    return payloads


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--prior", type=Path, required=True, help="the prior run whose sentences are drawn")
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
    parser.add_argument("--device", default="cpu",
                        help="the prior's (the speaker's), as the formal readout's `--device`; the executor always flies on "
                             "CPU — the live backend flies a sample again there — and the samples' generator is on this "
                             "device, so a cuda export draws other samples than a cpu one")
    parser.add_argument("--augment-seed", type=int, default=None,
                        help="fly every flight from an augmented start drawn with this seed (a kind of its own)")
    parser.add_argument("--overlay-id", default=None,
                        help="default: generation_[augstart_]<its name>[_r<round>]_<its checkpoint's sha256, 8 digits>")
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
    augmented = args.augment_seed is not None
    if augmented and args.readout is not None:
        parser.error("--readout is the prior's own starts' readout: an export from augmented starts carries none")
    started = time.perf_counter()
    params, record, words = replay.open_executor(executor, instructions)
    spec = words.spec
    model, config_file, checkpoint_sha, procedure_masks = open_trained_prior(prior_dir, instructions)
    device = torch.device(args.device)
    model.to(device)
    model_part = model_block(prior_dir, config_file, checkpoint_sha, model.config.variant)
    name = display_name(model_part["name"], model_part["round"])
    suffix = "" if model_part["round"] is None else f"_r{model_part['round']:02d}"
    start = "augstart_" if augmented else ""
    overlay_id = args.overlay_id or f"generation_{start}{model_part['name']}{suffix}_{checkpoint_sha[:8]}"
    missing = [code for code in airports if code not in model.config.airports]
    if missing:
        parser.error(f"the prior knows no airport {missing} ({list(model.config.airports)})")
    formal = None if args.readout is None else json.loads((resolved(args.readout) / "generation.json").read_text(encoding="utf-8"))
    readouts = {code: None if formal is None else readout_block(formal, prior_dir, record["sha256"], instructions,
                                                                 args.samples, args.temperature, code,
                                                                 procedure_masks.altitudes)
                for code in airports}
    geometries = load_candidates(instructions)
    landings = (airport_landings(instructions, rosters(instructions))
                if VARIANTS[model.config.variant].landing_context else None)
    flights = load_signals(instructions, SPLIT)
    sentences = load_sentences(instructions, SPLIT, spec)
    windows = start_altitude_windows(instructions) if augmented else None
    bases: dict[str, BaseSet] = {}
    existing = {}
    for code in airports:
        training = root / code / "training"
        if (training / overlay_id).exists():
            parser.error(f"{training / overlay_id} exists; an overlay is never overwritten")
        existing[code] = read_overlays(training, code, overlay_id)
        bases[code] = open_base_set(training, code, args.set, KIND_READBACK, spec, geometries[code])

    source = {"runner": RUNNER, "prior": repo_relative(prior_dir), "executor": repo_relative(executor),
              "instructions": repo_relative(instructions),
              "readout": None if args.readout is None else repo_relative(resolved(args.readout)), "git": git_state(),
              # the speaker's device: with the seed it names the samples drawn (the executor flies on CPU either way)
              "device": args.device}
    generation = generation_block(args.samples, args.temperature, args.seed, spec.step_s, procedure_masks,
                                  record["sha256"], params)
    if augmented:
        limits = augment.LIMITS
        generation["augment"] = {"seed": args.augment_seed, "tries": AUGMENT_TRIES,
                                 "limits": {"rotationDeg": limits.rotation_deg, "altitudeM": limits.altitude_m,
                                            "speedFraction": limits.speed_fraction},
                                 "timeoutFactor": augment.TIMEOUT_FACTOR}
    masks_text = ", ".join(procedure_masks.names) or "no procedure's masks"
    starts = f"augmented starts (seed {args.augment_seed})" if augmented else f"step {N_LOOK}"
    title = (f"{name} · {prior_dir.parent.name}/{prior_dir.name} · its own sentences, {args.samples} a flight "
             f"({masks_text}), flown by executor spec {record['sha256'][:12]} from {starts}")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    built = {}
    for code in airports:
        payloads = build_airport(bases[code], flights, sentences, instructions, geometries[code], model, params, words,
                                 landings, args.samples, generator=generator, temperature=args.temperature,
                                 procedure_masks=procedure_masks, augment_seed=args.augment_seed, windows=windows)
        payload = {"schema": AUGMENTED_SCHEMA if augmented else SCHEMA, "overlayId": overlay_id, "airport": code,
                   "writtenUtc": utc_now(), "producedBy": source, "base": bases[code].block, "model": model_part,
                   "generation": generation, **({} if augmented else {"readout": readouts[code]}),
                   "columns": list(COLUMNS), "flights": payloads}
        entry = overlay_entry(overlay_id, KIND_AUGMENTED_GENERATION if augmented else KIND_GENERATION, bases[code], title,
                              PAYLOAD_FILE, len(payloads), source)
        built[code] = (serialise(payload), entry)
        said = [s for item in payloads for s in item["samples"]]
        landed = sum(s["outcome"] == "landed" for s in said)
        unfitted = sum(item["augmentDraws"] is not None and item["augmentation"] is None for item in payloads) if augmented else 0
        print(f"  {code}: {sum(item['flown'] for item in payloads)} of {len(payloads)} flights flown"
              + (f" ({unfitted} with no plausible augmentation in {AUGMENT_TRIES} draws)" if augmented else "")
              + f", {landed} of {len(said)} samples landed; {time.perf_counter() - started:.0f} s", flush=True)
    # no airport is written while another's manifest changed since the start (each write checks its own again)
    for code in airports:
        require_overlays_unchanged(root / code / "training", code, overlay_id, existing[code])
    for code, (text, entry) in built.items():
        out = write_overlay(root / code / "training", code, entry, text, existing[code])
        print(f"  {code}: {out.stat().st_size / 1e6:.1f} MB → {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
