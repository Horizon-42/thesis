"""The prior's own sentences in multi-aircraft WINDOWS, for the frontend's Training module
(`aeroviz-4d/docs/36-2026-09-20-training-module.zh.md` §2.9, §4.11): every commanded aircraft of a window spoken to by the
prior together and flown by its own executor, judged in its window as the formal window readouts judge it
(`traffic_window.WindowLoop`), every other aircraft of the window replayed.

    python run_ts.py window_training_export \\
        --prior 4dTrajectory/outputs/POOLED/prior/<augmented r7, or an M4 round> \\
        --instructions 4dTrajectory/outputs/POOLED/instruction_language/v5_20260926 \\
        --executor 4dTrajectory/outputs/POOLED/executor/v11_20260927 \\
        [--readout 4dTrajectory/outputs/POOLED/traffic/<this prior's formal window readout>] \\
        --airports-root aeroviz-4d/public/data/airports --set traffic_windows_select \\
        --airport KMSY --airport KRDU --airport KSJC --airport KSMF --airport KSTL

**The windows** are the formal window readouts' draw (`traffic_window.draw_windows` over `training_files.TRAFFIC_SPLIT`,
`WINDOWS_PER_AIRPORT` an airport, ``--seed``), of which ``--windows`` an airport are chosen (`choose_windows`): split by
their commanded aircraft (`traffic_window_generation.SIZES`) as evenly as ``--windows`` divides (`shares`: 20 → 6, 7, 7,
the larger sizes first), each size in a permutation of a generator seeded by ``--seed`` and the airport (the same windows
whichever airports are exported); a size short of its share is filled from the others' rest, and counted.

**The set** (`KIND_TRAFFIC`: ``<set>/traffic.json``, `TRAFFIC_SCHEMA`) does not depend on the prior: the vocabulary,
candidates and frame as a read-back set's, and the commanded flights as its flights
(`instruction_training_export.head_block`, `flights_block`); per window (`window_payload`) its opening, its commanded
aircraft (each one's row 0 on the scene's steps, its time limit and its recorded rows), every other aircraft of it —
replayed with a sentence, or a background arrival — on its recorded rows, and the window AS RECORDED: its commanded
aircraft along their records (`traffic_window_generation.fixed_paths` "recorded"), their losses under VISUAL and IFR
(`losses_payload`) and their landings in order. Written by the first export that names it; a later one (another prior)
builds it again and refuses unless it is the same but for the time of writing and the run that wrote it.

**The overlay** (`KIND_WINDOW_GENERATION`: ``<overlay-id>/window_generation.json``, `SCHEMA`) is the prior's: every
chosen window spoken to ``--samples`` times (a sample is the whole window) by `traffic_window_generation.fly_windows`, as
the formal readouts fly them — in batches of at most ``--aircraft-steps`` (`window_batches`), each batch on its own
seeded stream (`batch_seed`, "scene"), in ``--workers`` forked processes (what is drawn does not depend on their number)
— **on its own draws, not the readout's** (the user, 2026-09-30, as `prior_generation_training_export` draws its own):
what is shown is this prior in these windows, never a readout's rows. A batch's windows share its stream, so the sentences
drawn depend on the windows flown together — the airports named, ``--windows``, ``--aircraft-steps`` — which the file
records (``producedBy``); the windows chosen do not. Per window and sample: each commanded aircraft's
sentence (`sentence_payload`), the window's losses under VISUAL (the loop's: they end aircraft) and IFR (its paths judged
again afterwards), and its landings in order. ``--readout`` (optional): the prior's formal window readout's summaries
copied beside (`readout_block`), refused unless it is this prior's, on this executor spec, artefact, split, draw,
samples and temperature.

**Clocks.** Scene times are seconds from the window's opening; a sentence's are its flight's own (row 0 at 0 s), its
row 0 on the scene's steps at its ``rowZeroS`` — every row is on the scene's even-second steps (multi-aircraft design
§2.1), so its recorded rows (``recorded``) are at its sentence's times.
An aircraft the judge ends (``lost_separation``) says nothing more and flies on, passive, in the scene (design §9 item
29): its words end at ``endS``, its track at its own end (``ownEndS``). Units SI; heights MSL, and the ellipsoid height
Cesium draws in (plus the flight's runway's HAE − MSL, `training_files.runway_hae_minus_msl_m`).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import gc
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ts_transformer.autopilot import replay
from ts_transformer.autopilot.executor import Executor
from ts_transformer.autopilot.judge import CROSSINGS, flown_track, outcome_of
from ts_transformer.experiments.instruction_training_export import Globe, flights_block, head_block
from ts_transformer.experiments.prior_free_generation import sentence_counts
from ts_transformer.experiments.prior_generation_training_export import (
    display_name, generation_block, model_block, outputs_path,
)
from ts_transformer.experiments.prior_train import rosters
from ts_transformer.experiments.traffic_loop import Run
from ts_transformer.experiments.traffic_speaking import scene_airports
from ts_transformer.experiments.traffic_window import Window, draw_windows
from ts_transformer.experiments.training_attitude import attitude_payload, executor_attitude, observed_attitudes
from ts_transformer.experiments.traffic_window_generation import (
    AIRCRAFT_STEPS, SCHEMA as READOUT_SCHEMA, SIZES, WINDOWS_PER_AIRPORT, WORKERS, Drawn, FixedWindow, Flown,
    batch_seed, drawn_subset, drawn_windows, fixed_paths, fly_windows, in_processes, size_of, window_batches,
    window_prior,
)
from ts_transformer.inference.separation import IFR
from ts_transformer.instructions.artefact import load_candidates, load_signals, spec_labeller_source
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.readout import flight_record
from ts_transformer.instructions.spec import READING_RULE
from ts_transformer.instructions.training_files import (
    INDEX_FILE, KIND_TRAFFIC, KIND_WINDOW_GENERATION, TRAFFIC_FILE, TRAFFIC_SCHEMA, TRAFFIC_SPLIT, BaseSet,
    candidates_sha256, check_set, listed_set, overlay_entry, read_index, read_overlays, require_index_unchanged,
    require_overlays_unchanged, rounded, runway_hae_minus_msl_m, serialise, write_overlay, write_set,
)
from ts_transformer.instructions.words import COLUMNS, UNCHANGED, Words
from ts_transformer.io_utils import file_sha256, utc_now
from ts_transformer.prior.data import VARIANTS, airport_landings
from ts_transformer.prior.scene import N_LOOK
from ts_transformer.repo_layout import REPO_ROOT, arrival_manifest_path, git_state, repo_relative

#: MIRROR of `aeroviz-4d/src/data/trainingTraffic.ts` (`TRAINING_WINDOW_GENERATION_SCHEMA`); the reader refuses anything
#: else by name. A name changes with its file's shape or meaning, on both sides, in one change. v2 (2026-09-30): every
#: aircraft's track carries the attitude it is drawn in (``track.attitude``, `training_attitude`).
SCHEMA = "aeroviz-training-window-generation-v2"
PAYLOAD_FILE = "window_generation.json"
RUNNER = "ts_transformer.experiments.window_training_export"
#: Windows chosen an airport (the user, 2026-09-30: 20, not the 12 first proposed).
WINDOWS = 20
#: The set a window export names by default.
SET_ID = "traffic_windows_select"
#: What a later export may find different in a set it names (`require_same_set`): when and by which run it was written.
SET_WRITTEN = ("writtenUtc", "producedBy")


# ---- the windows
def shares(count: int) -> dict[str, int]:
    """``count`` windows split over the window sizes as evenly as it divides, the larger sizes taking the rest."""
    base, extra = divmod(count, len(SIZES))
    return {size: base + (k >= len(SIZES) - extra) for k, size in enumerate(SIZES)}


def choose_windows(openings: Sequence[tuple[str, float, tuple[str, ...]]], codes: Sequence[str], per_airport: int,
                   seed: int) -> tuple[list[int], dict[str, dict[str, Any]]]:
    """``per_airport`` of the drawn windows (``openings``: `WindowDraw.openings`) at each airport of ``codes`` (module
    docstring): their places in the draw — an airport's by opening — and, per airport, what was drawn and taken."""
    chosen: list[int] = []
    counts: dict[str, dict[str, Any]] = {}
    for code in codes:
        mine = [w for w, (airport, _, _) in enumerate(openings) if airport == code]
        # a generator of the seed and the airport: the same windows whichever airports an export names
        permuted = np.random.default_rng([seed, *code.encode("ascii")]).permutation(len(mine))
        order = [mine[int(k)] for k in permuted]
        wanted = shares(per_airport)
        by_size = {size: [w for w in order if size_of(len(openings[w][2])) == size] for size in SIZES}
        taken = [w for size in SIZES for w in by_size[size][: wanted[size]]]
        filled = [w for w in order if w not in taken][: per_airport - len(taken)]
        if len(taken) + len(filled) < per_airport:
            raise ValueError(f"{code}: {len(mine)} windows drawn, {per_airport} wanted")
        picked = sorted(taken + filled, key=lambda w: (openings[w][1], w))
        chosen += picked
        counts[code] = {"drawn": len(mine),
                        "drawnBySize": {size: len(by_size[size]) for size in SIZES},
                        "shares": wanted,
                        "taken": {size: sum(size_of(len(openings[w][2])) == size for w in picked) for size in SIZES},
                        "filled": len(filled)}
    return chosen, counts


# ---- what is written
def callsign_of(dataset_id: str) -> str:
    """The callsign in a flight's key (``<ICAO>:<callsign>_<runway>_<icao24>_<landing>``)."""
    return dataset_id.split(":", 1)[1].rsplit("_", 3)[0]


def utc_text(epoch_s: float) -> str:
    return datetime.datetime.fromtimestamp(epoch_s, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def recorded_track(window: Window, key: str, globe: Globe, flight: FlightSignals,
                   attitude: dict[str, np.ndarray | None]) -> dict[str, Any]:
    """An aircraft of ``window`` on its recorded rows (`Window.track`: the rows the judge replays it on — its signals'
    first rows, checked), scene time, with the attitude it is drawn in there (``attitude``: its observed attitude over
    all its rows, `training_attitude.observed_attitudes`)."""
    track = window.track(key)
    rows = len(track.presence.times_s)
    if not (np.array_equal(track.e_m, flight.e_m[:rows]) and np.array_equal(track.n_m, flight.n_m[:rows])):
        raise ValueError(f"{key}: the window's recorded rows are not its signals' first {rows}")
    lat, lon = globe.latlon(track.e_m, track.n_m)
    return {"tS": rounded(track.presence.times_s - window.opens_s, 3), "lon": rounded(lon, 7), "lat": rounded(lat, 7),
            "altitudeM": rounded(track.height_m, 2),
            "altitudeHaeM": rounded(globe.hae_m(track.height_m, track.presence.runway), 2),
            "attitude": attitude_payload(attitude, slice(0, rows))}


def losses_payload(run: Run, opens_s: float) -> dict[str, Any]:
    """A window's losses as the judge recorded them under one reading, scene time: every episode (a pair under its
    minimum on consecutive steps — its relation, span, kinds, the closest it came against the minimum then, who answers
    for it and whom it ended), every wake shortfall at a landing, and every aircraft ended, with why."""
    return {
        "episodes": [{"pair": list(e["pair"]), "relation": e["relation"], "fromS": round(e["first_s"] - opens_s, 3),
                      "toS": round(e["last_s"] - opens_s, 3), "steps": e["steps"], "kinds": list(e["kinds"]),
                      "closestM": round(e["closest_m"], 1), "requiredM": round(e["required_m"], 1),
                      "wakeKnown": e["wake_known"], "responsible": [r["key"] for r in e["responsible"]],
                      "ended": list(e["ended"])} for e in run.episodes],
        "atThreshold": [{"atS": round(a["t_s"] - opens_s, 3), "leader": a["leader"], "follower": a["follower"],
                         "gapM": round(a["gap_m"], 1), "requiredM": round(a["required_m"], 1),
                         "relation": a["relation"]} for a in run.at_threshold],
        "ended": [{"datasetId": key, "atS": round(end["t_s"] - opens_s, 3), "kind": end["kind"],
                   "relation": end["relation"], "with": end["with"]} for key, end in run.ended.items()]}


def landings_payload(landings: Sequence[tuple[str, float]], opens_s: float) -> list[dict[str, Any]]:
    """The commanded aircraft that landed, in the order they crossed their thresholds, scene time."""
    return [{"datasetId": key, "atS": round(at - opens_s, 3)} for key, at in sorted(landings, key=lambda x: (x[1], x[0]))]


def window_payload(window: Window, limits: Sequence[float], recorded: FixedWindow, globe: Globe, step_s: float,
                   flights: Mapping[str, FlightSignals], observed: Mapping[str, dict[str, np.ndarray | None]]
                   ) -> dict[str, Any]:
    """One window of the set (module docstring): who is in it, where, and what the record made of it (``flights`` and
    ``observed``: each aircraft's signals and observed attitude, by dataset id)."""
    opens = window.opens_s

    def track(key: str) -> dict[str, Any]:
        return recorded_track(window, key, globe, flights[key], observed[key])

    return {
        "opensUtc": utc_text(opens),
        "commanded": [{"datasetId": key, "rowZeroS": round(window.first_step_s(key) - opens, 3),
                       "limitS": round(limit, 3), "recorded": track(key)}
                      for key, limit in zip(window.commanded, limits, strict=True)],
        "others": [{"datasetId": key, "callsign": callsign_of(key), "category": window.rows(key).category,
                    "role": "replayed" if window.rows(key).presence.speaking else "background",
                    "track": track(key)} for key in window.others],
        "recorded": {"visual": losses_payload(recorded.run, opens), "ifr": losses_payload(recorded.again, opens),
                     "landings": landings_payload([(path.key, path.landing_s) for path, own in
                                                   zip(recorded.paths, recorded.owns) if own == "landed"], opens)},
    }


def sentence_payload(flown: Flown, i: int, sample: int, executor: Executor, executor_flown: Any, globe: Globe,
                     words: Words) -> dict[str, Any]:
    """Commanded aircraft ``i`` of a flown batch, one sample (module docstring): the words it said as events on its own
    steps (step 0 is row `N_LOOK`; nothing after the judge ended it), its outcome under the loop's reading and its own
    end, when each came (``endS``: the judge's end, else its last judged instant; ``ownEndS``: its last state), the
    crossing (its own end's, read on the runway in force then, `autopilot.judge.outcome_of`), what the sentence did with
    its runway and approach, the probability the prior put on what the masks removed, and its states to its own end
    (the loop's: what the judge read), on its own clock, with the attitude it is drawn in (the executor's states at the
    rows the loop recorded — one a step from its first, `WindowLoop._record`: checked to be the loop's own positions —
    and its commands there, `training_attitude.executor_attitude`). ``executor``: its executor, ``executor_flown`` its
    `Flown`."""
    loop, got = flown.loop, flown.results[i]
    step_s = words.spec.step_s
    said = np.asarray(got.said)
    if len(said) == 0 or (said[0] == UNCHANGED).any():
        raise ValueError(f"{got.key}: the first predicted step does not say every column")
    states = loop.states[i][: min(int(loop.in_scene_to[i]), len(loop.states[i]) - 1) + 1]
    row_zero_s = states[0].t_s - N_LOOK * step_s                    # its row 0 on the scene's steps
    geometry, signals = flown.part.geometries[i], flown.part.signals[i]
    crossing = None
    if got.own in CROSSINGS:
        ended = outcome_of(executor_flown, int(loop.place[i]), geometry, got.runway, words.spec)
        if ended.outcome != got.own:
            raise ValueError(f"{got.key}: the loop ended it {got.own}, its executor's flight reads {ended.outcome}")
        crossing = {"crossM": round(ended.crossing["cross_m"], 2), "heightM": round(ended.crossing["height_m"], 2),
                    "atS": round(N_LOOK * step_s + ended.crossing["at_row"] * executor_flown.cycle_s, 3),
                    "runway": ended.crossing["runway_index"]}
    end_s = got.end["t_s"] if got.end is not None else loop.judged_until_s(i)
    counts = sentence_counts(said)
    e = np.array([s.e_m for s in states])
    n = np.array([s.n_m for s in states])
    height = np.array([s.height_m for s in states])
    lat, lon = globe.latlon(e, n)
    place = int(loop.place[i])
    rows = np.minimum(np.arange(len(states)) * executor.step_rows, executor_flown.commands.shape[1])
    read = flown_track(executor_flown.states[place].cpu().numpy()[rows], geometry)
    if not (np.array_equal(read["e"], e) and np.array_equal(read["n"], n)):
        raise ValueError(f"{got.key}: its executor's states at the rows a step apart are not the loop's")
    attitude = executor_attitude(executor_flown, place, rows, executor.inputs.aero_params[place].cpu().numpy())
    return {
        "datasetId": got.key, "sample": sample, "outcome": got.outcome, "own": got.own,
        "end": None if got.end is None else {"kind": got.end["kind"], "relation": got.end["relation"],
                                             "with": got.end["with"]},
        "endS": round(end_s - row_zero_s, 3), "ownEndS": round(states[-1].t_s - row_zero_s, 3), "crossing": crossing,
        "firstRunway": counts["first_runway"], "lastRunway": counts["last_runway"],
        "runwayChanges": counts["runway_changes"], "goArounds": counts["go_arounds"],
        "clearedAtEnd": counts["cleared_at_end"],
        # over the steps it spoke to its judged end (after the judge ended it the speaker is silent), as the readout
        # averages it; none spoken (ended at its first predicted step): 0
        "forbiddenMass": {COLUMNS[c]: round(float(mass[i, : got.counted].mean()) if got.counted else 0.0, 6)
                          for c, mass in sorted(loop.speaker.forbidden.items())},
        "rows": N_LOOK + len(said),
        "events": [{"row": N_LOOK + int(step), "column": int(column), "value": int(said[step, column])}
                   for step, column in zip(*np.nonzero(said != UNCHANGED))],
        "track": {"tS": rounded(np.array([s.t_s for s in states]) - row_zero_s, 3), "lon": rounded(lon, 7),
                  "lat": rounded(lat, 7), "altitudeM": rounded(height, 2),
                  "altitudeHaeM": rounded(globe.hae_m(height, signals.runway), 2),
                  "groundSpeedMps": rounded(np.array([s.ground_speed_mps for s in states]), 3),
                  "attitude": attitude_payload(attitude)},
    }


def batch_payloads(flown: Flown, drawn: Drawn, samples: int, globes: Mapping[str, Globe],
                   words: Words) -> list[tuple[int, dict[str, Any]]]:
    """Each window sample of a flown batch as the overlay writes it — its aircraft's sentences, its losses under VISUAL
    (the loop's) and IFR (judged again) and its landings — with its window's place in ``drawn``, in the loop's order
    (a window's samples in order)."""
    step_s = words.spec.step_s
    loop = flown.loop
    executors = [(executor, executor.flown()) for _, _, executor, _ in loop.executors]
    out = []
    for b, w in enumerate(flown.instances):
        window = drawn.windows[w]
        globe = globes[window.airport.flights.code]
        here = flown.members(b)
        if [flown.results[i].key for i in here] != list(window.commanded):
            raise ValueError(f"window {w}'s loop holds {[flown.results[i].key for i in here]}, not its commanded aircraft")
        sample = b % samples
        out.append((w, {
            "sample": sample,
            "aircraft": [sentence_payload(flown, i, sample, *executors[int(loop.group[i])], globe, words) for i in here],
            "visual": losses_payload(loop.runs[b], window.opens_s),
            "ifr": losses_payload(flown.judged_again(b, IFR, step_s, window), window.opens_s),
            # the aircraft whose OWN end is a landing, as the record's are: the loop keeps a landing time for one the
            # glidepath lower edge stopped first (its own end), which is not a landing here
            "landings": landings_payload([(flown.results[i].key, flown.results[i].landing_s) for i in here
                                          if flown.results[i].own == "landed"], window.opens_s)}))
    return out


def readout_block(readout: dict[str, Any], directory: Path, *, prior_dir: Path, checkpoint_sha256: str,
                  executor_sha256: str, instructions: Path, samples: int, temperature: float, seed: int,
                  airport: str) -> dict[str, Any]:
    """The prior's formal window readout (`traffic_window_generation`'s ``window_generation.json``) as the frontend reads
    it — per source (the model in the scene, the record) the aircraft counted, the landed share and the lost separation
    under VISUAL and IFR, at ``airport`` and over every airport — refused unless it is this prior's (its directory from
    ``4dTrajectory/outputs/`` on, and its checkpoint), on this executor spec and artefact, over the windows these are
    chosen from (every aircraft of a window commanded), with these samples and this temperature, as drawn (not
    augmented)."""
    if readout["schema"] != READOUT_SCHEMA:
        raise ValueError(f"the readout is a {readout['schema']} file, not {READOUT_SCHEMA}")
    wanted = {"prior": outputs_path(prior_dir), "checkpoint": checkpoint_sha256, "executor": executor_sha256,
              "instructions": outputs_path(instructions), "split": TRAFFIC_SPLIT, "commanded": "every",
              "windows_per_airport": WINDOWS_PER_AIRPORT, "seed": seed, "samples": samples, "temperature": temperature,
              "augment_seed": None}
    found = {"prior": outputs_path(readout["prior"]["directory"]), "checkpoint": readout["prior"]["checkpoint_sha256"],
             "executor": readout["executor"]["sha256"], "instructions": outputs_path(readout["instructions"]),
             "split": readout["split"], "commanded": readout["commanded"],
             "windows_per_airport": readout["windows_per_airport"], "seed": readout["seed"],
             "samples": readout["samples"], "temperature": readout["temperature"],
             "augment_seed": readout["augment_seed"]}
    differ = {key: (found[key], wanted[key]) for key in wanted if found[key] != wanted[key]}
    if differ:
        raise ValueError("the readout is not this prior's over these windows: " +
                         ", ".join(f"{key} {theirs!r}, expected {ours!r}" for key, (theirs, ours) in differ.items()))

    def cell(entry: dict[str, Any]) -> dict[str, Any]:
        # the outcomes list the ones that happened (a share each): none landed is a share of 0
        return {"aircraft": entry["aircraft"],
                "landed": entry["outcomes"]["landed"] if "landed" in entry["outcomes"] else 0.0,
                "lostSeparation": entry["lost_separation"], "lostSeparationIfr": entry["lost_separation_ifr"]}

    summaries = readout["readout"]
    return {"directory": outputs_path(directory), "writtenUtc": readout["written_utc"],
            # what the cells count: every sample of its draw's windows, not the export's few
            "windowsPerAirport": readout["windows_per_airport"], "samples": readout["samples"],
            **{source: {"here": cell(summaries["airports"][airport][source]), "all": cell(summaries["pooled"][source])}
               for source in ("scene", "recorded")}}


def require_same_set(existing: dict[str, Any], built: dict[str, Any], path: Path) -> None:
    """A set a later export names is the one it would write (module docstring): the same but for `SET_WRITTEN`."""
    def kept(payload: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in payload.items() if key not in SET_WRITTEN}

    theirs, ours = kept(existing), kept(json.loads(serialise(built)))
    if theirs != ours:
        differ = sorted(key for key in set(theirs) | set(ours) if theirs.get(key) != ours.get(key))
        raise ValueError(f"{path} is not the set this export builds (differs in {differ}); export into another --set")


# ---- the disk
@dataclasses.dataclass(frozen=True)
class OnDisk:
    """What an export found at an airport before it built anything: the index as it stands when the set is to be written
    (None: an earlier export wrote it — the one built is checked against it, `require_same_set`) and the overlays."""

    index: list[dict[str, Any]] | None
    overlays: list[dict[str, Any]]


def on_disk(root: Path, airports: Sequence[str], set_id: str, overlay_id: str) -> dict[str, OnDisk]:
    """Every refusal about what is on disk, before the windows are drawn and flown: a set an earlier export wrote is a
    window set listed as one (a directory without its file — a write that stopped half-way — is refused); a new one's
    index must not list it; the overlay is new."""
    found = {}
    for code in airports:
        training = root / code / "training"
        path = training / set_id / TRAFFIC_FILE
        if path.parent.exists() and not path.exists():
            raise ValueError(f"{path.parent} exists without its {TRAFFIC_FILE}: an export stopped there; remove it")
        if path.exists():
            index = training / INDEX_FILE
            listed = listed_set(json.loads(index.read_text(encoding="utf-8")), index, code, set_id)
            check_set(listed, json.loads(path.read_text(encoding="utf-8")), path, code, KIND_TRAFFIC)
        if (training / overlay_id).exists():
            raise ValueError(f"{training / overlay_id} exists; an overlay is never overwritten")
        found[code] = OnDisk(None if path.exists() else read_index(training, code, set_id),
                             read_overlays(training, code, overlay_id))
    return found


def write_export(root: Path, found: Mapping[str, OnDisk], sets: Mapping[str, tuple[dict[str, Any], dict[str, Any]]],
                 overlays: Mapping[str, tuple[str, dict[str, Any]]]) -> list[Path]:
    """Each airport's set (``sets``: payload and index entry — written only where `on_disk` found none; one found is
    checked to be it) and overlay (``overlays``: text and manifest entry); nothing is written while a manifest another run
    wrote meanwhile would lose its entry. The files written."""
    for code, (payload, entry) in sets.items():
        training = root / code / "training"
        if found[code].index is None:
            path = training / entry["file"]
            require_same_set(json.loads(path.read_text(encoding="utf-8")), payload, path)
        else:
            require_index_unchanged(training, code, entry["id"], found[code].index)
        require_overlays_unchanged(training, code, overlays[code][1]["id"], found[code].overlays)
    written = []
    for code, (payload, entry) in sets.items():
        training = root / code / "training"
        if found[code].index is not None:
            written.append(write_set(training, code, entry, serialise(payload), found[code].index))
        text, overlay = overlays[code]
        written.append(write_overlay(training, code, overlay, text, found[code].overlays))
    return written


# ---- the run


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--prior", type=Path, required=True,
                        help="the prior commanding the windows: a single-aircraft prior (augmented) or an M4 round")
    parser.add_argument("--instructions", type=Path, required=True, help="the instruction artefact it was trained on")
    parser.add_argument("--executor", type=Path, required=True, help="an executor spec directory this code opens")
    parser.add_argument("--readout", type=Path, default=None,
                        help="this prior's formal window readout over the same draw (optional; its summaries copied)")
    parser.add_argument("--airports-root", type=Path, required=True,
                        help="the frontend's airports directory (…/public/data/airports)")
    parser.add_argument("--set", default=SET_ID, help="the window set (written by the first export that names it)")
    parser.add_argument("--airport", action="append", required=True, help="an ICAO code; repeat for several")
    parser.add_argument("--windows", type=int, default=WINDOWS, help="windows an airport")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337, help="the draw's (the readouts'), the choice's and the samples'")
    parser.add_argument("--aircraft-steps", type=int, default=AIRCRAFT_STEPS, help="a batch's most")
    parser.add_argument("--workers", type=int, default=WORKERS, help="reading processes (what is drawn does not "
                        "depend on it)")
    parser.add_argument("--device", default="cuda", help="the prior's, as the formal readouts'; the executors fly on CPU")
    parser.add_argument("--overlay-id", default=None,
                        help="default: windows_<its name>[_r<round>]_<its checkpoint's sha256, 8 digits>")
    args = parser.parse_args(argv)
    if args.windows < 1 or args.samples < 1 or args.workers < 1:
        parser.error("at least one window an airport, one sample and one reading process")

    def resolved(path: Path) -> Path:
        return path if path.is_absolute() else REPO_ROOT / path

    prior_dir, instructions, executor, root = (resolved(p) for p in (args.prior, args.instructions, args.executor,
                                                                       args.airports_root))
    airports = [code.upper() for code in args.airport]
    if len(set(airports)) != len(airports):
        parser.error(f"an airport is named twice in {airports}")
    started = time.perf_counter()
    params, record, words = replay.open_executor(executor, instructions)
    spec = words.spec
    step_s = spec.step_s
    try:                                               # on the CPU until the reading processes are forked
        model, attention, config_file, procedure_masks = window_prior(prior_dir, instructions, args.seed)
    except ValueError as refusal:
        parser.error(str(refusal))
    checkpoint_sha = file_sha256(prior_dir / "checkpoint.pt")
    model_part = model_block(prior_dir, config_file, checkpoint_sha, model.config.variant)
    suffix = "" if model_part["round"] is None else f"_r{model_part['round']:02d}"
    overlay_id = args.overlay_id or f"windows_{model_part['name']}{suffix}_{checkpoint_sha[:8]}"
    missing = [code for code in airports if code not in model.config.airports]
    if missing:
        parser.error(f"the prior knows no airport {missing} ({list(model.config.airports)})")
    formal = None
    if args.readout is not None:
        formal = json.loads((resolved(args.readout) / "window_generation.json").read_text(encoding="utf-8"))
    readouts = {code: None if formal is None else readout_block(
        formal, resolved(args.readout), prior_dir=prior_dir, checkpoint_sha256=checkpoint_sha,
        executor_sha256=record["sha256"], instructions=instructions, samples=args.samples,
        temperature=args.temperature, seed=args.seed, airport=code) for code in airports}

    try:                                               # every refusal about the disk before any work
        found = on_disk(root, airports, args.set, overlay_id)
    except ValueError as refusal:
        parser.error(str(refusal))

    # the windows: the formal readouts' draw, of which --windows an airport
    landings = (airport_landings(instructions, rosters(instructions))
                if VARIANTS[model.config.variant].landing_context else None)
    scenes, built = scene_airports(instructions, TRAFFIC_SPLIT, spec, model.config.airports, landings,
                                   model.config.max_rows)
    draw = draw_windows(instructions, TRAFFIC_SPLIT, spec, words, scenes, per_airport=WINDOWS_PER_AIRPORT,
                        seed=args.seed, step_s=step_s)
    chosen, choice = choose_windows(draw.openings, airports, args.windows, args.seed)
    drawn = drawn_subset(drawn_windows(draw, scenes, params, step_s), chosen)
    print(f"{len(drawn.windows)} {TRAFFIC_SPLIT} windows of {len(draw.openings)} drawn, "
          f"{len(drawn.batch.readings)} commanded aircraft ({choice}); scenes built ({built}), "
          f"{time.perf_counter() - started:.0f}s", flush=True)

    # the sets (the prior's no part of them): the flights, and each window as recorded
    geometries = load_candidates(instructions)
    globes = {code: Globe(geometries[code], runway_hae_minus_msl_m(instructions, code, arrival_manifest_path(code)))
              for code in airports}
    _, _, recorded = fixed_paths(drawn, range(len(drawn.windows)), "recorded", words, params, procedure_masks)
    artefact_name = repo_relative(instructions)
    labeller = spec_labeller_source(instructions)
    split_signals = {flight.dataset_id: flight for flight in load_signals(instructions, TRAFFIC_SPLIT)}
    git = git_state()
    sets: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for code in airports:
        places = [w for w, window in enumerate(drawn.windows) if window.airport.flights.code == code]
        members = {drawn.batch.signals[j].dataset_id: j for w in places for j in drawn.members[w]}
        chosen_flights = [(flight_record(drawn.batch.readings[j])["stratum"], drawn.batch.signals[j],
                           drawn.batch.readings[j]) for j in members.values()]
        # every aircraft of its windows, commanded or not: its signals and observed attitude (one rebuild an airport)
        keys = sorted({key for w in places for key in (*drawn.windows[w].commanded, *drawn.windows[w].others)})
        observed = observed_attitudes(instructions, [split_signals[key] for key in keys])
        flights, centreline_m = flights_block(chosen_flights, geometries[code], spec, words, globes[code], observed)
        drawn_from = (f"the formal window readouts' draw — {WINDOWS_PER_AIRPORT} windows an airport of the "
                      f"{TRAFFIC_SPLIT} split, seed {args.seed} (traffic_window.draw_windows) — of which "
                      f"{args.windows} at {code}: split by their commanded aircraft {dict(zip(SIZES, shares(args.windows).values()))}, "
                      f"each size in a permutation seeded by the seed and the airport, a size short of its share filled "
                      f"from the others' rest ({choice[code]['filled']} filled)")
        cohort = {"split": TRAFFIC_SPLIT, "windows": args.windows, "seed": args.seed, "drawnFrom": drawn_from}
        payload = {
            "schema": TRAFFIC_SCHEMA, "setId": args.set, "airport": code, "writtenUtc": utc_now(),
            "producedBy": {"runner": RUNNER, "artefact": artefact_name, "git": git},
            "cohort": {**cohort, **choice[code]},
            **head_block(geometries[code], spec, words, labeller, globes[code], centreline_m),
            "flights": flights,
            "windows": [window_payload(drawn.windows[w], [drawn.limits[j] for j in drawn.members[w]], recorded[w],
                                       globes[code], step_s, split_signals, observed) for w in places],
        }
        entry = {"id": args.set, "kind": KIND_TRAFFIC,
                 "title": f"Multi-aircraft windows · {TRAFFIC_SPLIT} · {args.windows} an airport",
                 "file": f"{args.set}/{TRAFFIC_FILE}", "vocabularySha256": spec.sha256,
                 "runwaySha256": candidates_sha256(geometries[code]), "readingRule": READING_RULE,
                 "flights": len(flights), "cohort": cohort,
                 "source": {"artefact": artefact_name, "specSha256": spec.sha256, "labellerSourceSha256": labeller,
                            "exporter": RUNNER, "git": git}}
        sets[code] = (payload, entry)
    print(f"sets built, {time.perf_counter() - started:.0f}s", flush=True)

    # the prior in every window, --samples times
    batches = window_batches(drawn, args.samples, args.aircraft_steps, step_s)
    device = torch.device(args.device)

    def read(number: int) -> list[tuple[int, dict[str, Any]]]:
        speaking = model.to(device)
        generator = torch.Generator(device=device).manual_seed(batch_seed(args.seed, "scene", number))
        flown = fly_windows(speaking, drawn, batches[number], "scene", words, params, landings, args.samples,
                            generator=generator, temperature=args.temperature, procedure_masks=procedure_masks)
        try:
            return batch_payloads(flown, drawn, args.samples, globes, words)
        finally:
            flown.loop.close()

    gc.collect()
    gc.freeze()                                         # the reading processes share the parent's data, not copy it
    by_window: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for number, got, peak in in_processes(args.workers, list(range(len(batches))), read):
        for w, sample in got:
            by_window[w].append(sample)
        print(f"  batch {number + 1}/{len(batches)}, {time.perf_counter() - started:.0f}s, GPU {peak:.2f} GB",
              flush=True)

    name = display_name(model_part["name"], model_part["round"])
    masks_text = ", ".join(procedure_masks.names) or "no procedure's masks"
    title = (f"{name} · {prior_dir.parent.name}/{prior_dir.name} · commanding every aircraft of {args.windows} windows an "
             f"airport, {args.samples} samples a window ({masks_text}), flown by executor spec {record['sha256'][:12]}")
    source = {"runner": RUNNER, "prior": repo_relative(prior_dir), "executor": repo_relative(executor),
              "instructions": repo_relative(instructions),
              "readout": None if args.readout is None else repo_relative(resolved(args.readout)), "git": git_state(),
              "device": args.device, "workers": args.workers, "aircraftSteps": args.aircraft_steps,
              # the windows flown together (a batch's windows share its stream): every airport of this export
              "airports": airports, "windows": args.windows}
    generation = {**generation_block(args.samples, args.temperature, args.seed, step_s, procedure_masks,
                                     record["sha256"], params), "trafficAttention": attention}
    built_overlays = {}
    for code in airports:
        places = [w for w, window in enumerate(drawn.windows) if window.airport.flights.code == code]
        windows = []
        for w in places:
            samples = sorted(by_window[w], key=lambda item: item["sample"])
            if [item["sample"] for item in samples] != list(range(args.samples)):
                raise ValueError(f"{code} window {w} was read {len(samples)} times, not {args.samples}")
            windows.append({"opensUtc": utc_text(drawn.windows[w].opens_s),
                            "commanded": list(drawn.windows[w].commanded), "samples": samples})
        base = BaseSet(code, sets[code][1], sets[code][0])
        payload = {"schema": SCHEMA, "overlayId": overlay_id, "airport": code, "writtenUtc": utc_now(),
                   "producedBy": source, "base": base.block, "model": model_part, "generation": generation,
                   "readout": readouts[code], "columns": list(COLUMNS), "windows": windows}
        entry = overlay_entry(overlay_id, KIND_WINDOW_GENERATION, base, title, PAYLOAD_FILE,
                              len(sets[code][0]["flights"]), source)
        built_overlays[code] = (serialise(payload), entry)
        said = [a for window in windows for s in window["samples"] for a in s["aircraft"]]
        print(f"  {code}: {len(windows)} windows, {len(said)} aircraft sentences — "
              f"{sum(a['outcome'] == 'landed' for a in said)} landed, "
              f"{sum(a['outcome'] == 'lost_separation' for a in said)} lost separation", flush=True)

    for out in write_export(root, found, sets, built_overlays):
        print(f"  {out.stat().st_size / 1e6:.1f} MB → {out}", flush=True)
    print(f"done in {time.perf_counter() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
