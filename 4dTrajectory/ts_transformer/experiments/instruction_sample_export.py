"""Export a few flights' sentences from a vocabulary artefact for the frontend's Training view (design: `aeroviz-4d/docs/36-2026-09-20-training-module.zh.md`).

    python run_ts.py instruction_sample_export --vocabulary <…/vocabulary_tau10> \\
        --out aeroviz-4d/public/data/airports/KRDU/training/<set> [--prior <…/prior_s1337>] \\
        [--flights 40] [--split train] [--set-id vocabulary_tau10] [--title "…"]

**It re-reads nothing.** The sentences come from the artefact's own ``sentences_<split>.json``
only the TRACK is rebuilt, straight from the airport's arrival manifest. Re-running the labeller
here would let the view drift from the artefact it claims to show: the published object is the
artefact, so the view shows the artefact.

With ``--prior`` the set also carries what the MODEL says: at every event of every drawn flight,
the prior's top-1 next words and how sure it was, plus that sentence flown by the same kinematics.
It is teacher-forced (`PRIOR_METHOD`) — at each event the model saw the truth's history and the
truth's state — and the file says so, because a reader who assumed free generation would be
reading a different experiment.

**The draw is the exporter's own** (2026-09-21): a seeded permutation of the split, rebuilt and
then stratified by `approach_difficulty`, because the stratum is a property of the track. The rule
and the seed go into the manifest. It used to be the prefix of ``hand_check/index.csv`` so the
screen showed the pages a human had marked; a pooled artefact's pages span five airports, so that
prefix cannot fill one airport's strata.

Written under ``--out`` (refused if it exists), plus an entry in the parent's ``index.json``:

    <out>/sample.json    the vocabulary's spec, its runway classes, the `geometry` block of
                         assumptions the flown sentences are drawn under, and per flight the
                         sentence (events × 6 words), the instructions, the absorbed manoeuvres,
                         the observed track IN THE RUNWAY FRAME (what the read-back charts plot)
                         and the `geometric` track flown from the words alone
    ../index.json        the manifest the frontend lists; this set added or replaced in place

Both tracks carry geodetic columns for the 3D layer, and their altitude is HAE: a record is MSL,
Cesium reads the ellipsoid, and the conversion belongs at this boundary (`geodetic_columns`).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from flight_scenarios.datum import geoid_undulation_m

from ts_transformer.config import TSConfig, default_anchor
from ts_transformer.data.dataset import build_series, load_flight_dicts
from ts_transformer.repo_layout import arrival_manifest_path
from ts_transformer.data.approach_difficulty import (
    STRAIGHT_TORTUOSITY, STRATUM_SHORT, STRATUM_STRAIGHT_IN, STRATUM_VECTORED, approach_difficulty,
)
from ts_transformer.experiments.support import REPO_ROOT
from ts_transformer.io_utils import file_sha256, utc_now, write_json_atomic
from ts_transformer.manoeuvre import instruction_kinematics as kinematics
from ts_transformer.manoeuvre.instructions import (
    INSTRUCTION_KINDS, TERMINAL_LANDED, VOCABULARY_FILE, Reading, Vocabulary, course_frame,
    load_vocabulary, runway_sha256, word_counts,
)
from ts_transformer.manoeuvre.segments import MINIMUM_GROUND_SPEED_MPS


#: MIRROR of `src/data/trainingSample.ts` (`TRAINING_INDEX_SCHEMA` / `TRAINING_SAMPLE_SCHEMA`).
#: The reader refuses anything else by name, so these two move together or not at all.
INDEX_SCHEMA = "aeroviz-training-index-v1"
SAMPLE_SCHEMA = "aeroviz-training-sample-v1"
#: The set kinds the frontend knows (`TRAINING_SET_KINDS` there).
KIND_READBACK = "vocabulary-readback"
#: A set that also carries what the PRIOR said at every event. The kind is what the reader keys
#: the prior block's presence on: a `vocabulary-readback` set must not have one and a
#: `prior-generated` set must, so neither is read leniently.
KIND_PRIOR = "prior-generated"

INDEX_FILE = "index.json"
SAMPLE_FILE = "sample.json"


#: How many candidates are rebuilt per flight wanted. The stratum is a property of the TRACK
#: (`approach_difficulty` at the executor's anchor), so the draw has to rebuild before it can
#: stratify; this is the margin that makes both strata fill on a cohort whose mix is unknown.
POOL_FACTOR = 5


def drawn_flights(artefact: Path, per_stratum: int, seed: int) -> list[str]:
    """The candidate pool, in draw order: a seeded permutation of the split's flights.

    THE DRAW MOVED OFF THE HAND CHECK (2026-09-21). It used to be the prefix of
    ``hand_check/index.csv``, so that the flights on screen were the pages a human had marked
    一致 / 漏读 / 误读 — the two views showing the same aircraft, which was the point. The
    `segment-v12` artefacts were written with ``--hand-check 0``: there are no pages, so there is
    nothing to be a prefix of, and that property is vacuous until a hand check is run over this
    vocabulary. Until then the draw is this: seeded, stated in the manifest, and reproducible
    from the artefact alone. If pages are ever written for this vocabulary, this goes back to
    reading their order so the two sets match again.

    The stratum is NOT decided here, because it cannot be: it is a property of the rebuilt track.
    This returns the order to rebuild in, and `stratify` takes the first `per_stratum` of each.
    """
    payload_ids = sorted(artefact)      # `artefact` is the reading map: sorted for determinism
    if len(payload_ids) < per_stratum * 2:
        raise SystemExit(f"the split holds {len(payload_ids)} flights, fewer than the {per_stratum * 2} asked for")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(payload_ids))
    return [payload_ids[int(i)] for i in order[:min(len(payload_ids), per_stratum * 2 * POOL_FACTOR)]]


def stratify(candidates: list[str], series: list[Any], config: TSConfig,
             per_stratum: int) -> list[tuple[str, str, Any]]:
    """``[(flight_id, stratum, series)]``: the first ``per_stratum`` of each stratum, in the draw
    order above.

    The two strata are the package's own (`approach_difficulty` at the executor's anchor):
    straight-in is a route tortuosity below the fleet's cut, vectored is above it AND not yet
    established at the anchor. A flight in neither (tortuous but already established) is not
    drawn — the same rule the vocabulary runner's hand-check draw applies, so the two draws
    remain comparable even though they are no longer the same flights.

    A stratum that cannot be filled from the pool is REFUSED. Quietly returning 31 straight-in
    and 9 vectored would publish a set whose stratum column says something the draw does not.
    """
    anchor = default_anchor(config)
    taken: dict[str, list[tuple[str, str, Any]]] = {STRATUM_STRAIGHT_IN: [], STRATUM_VECTORED: []}
    for flight_id, item in zip(candidates, series, strict=True):
        difficulty = approach_difficulty(item, anchor)
        if difficulty.route_tortuosity < STRAIGHT_TORTUOSITY:
            stratum = STRATUM_STRAIGHT_IN
        elif not difficulty.established_at_anchor:
            stratum = STRATUM_VECTORED
        else:
            continue
        if len(taken[stratum]) < per_stratum:
            taken[stratum].append((flight_id, STRATUM_SHORT[stratum], item))
    short = {STRATUM_SHORT[name]: len(rows) for name, rows in taken.items() if len(rows) < per_stratum}
    if short:
        raise SystemExit(
            f"the draw filled {short} of the {per_stratum} per stratum asked for, out of "
            f"{len(candidates)} rebuilt candidates — raise --flights' pool or lower --flights"
        )
    return [row for name in sorted(taken) for row in taken[name]]


def series_by_dataset_id(airport: str, dataset_ids: list[str], config: TSConfig):
    """The drawn flights' tracks, rebuilt STRAIGHT FROM the airport's arrival manifest.

    The sibling door is `support.cohort_from_manifests`, which the vocabulary runner and the
    prior use; this is the same door for a handful of named flights instead of a whole cohort.
    It replaced a checkpoint (`rebuild_cohort`) on 2026-09-21: the checkpoint was only ever "the
    door to the data" here — nothing of the model was used — and the per-airport checkpoints that
    could open the other four airports are `prediction_output='plan'`, retired 2026-09-18, so
    `TSConfig.from_dict` refuses them. The manifest is the thing the sentences were read from
    anyway, and its digest is what the export records.

    `TSConfig()` is the config the sentences were read under (the vocabulary runner's own
    `--airports` branch uses it), so the course frame here is the one the words were fitted in.
    """
    manifest = arrival_manifest_path(airport)
    built, report = build_series(
        load_flight_dicts([manifest], include_flight_keys=set(dataset_ids), verbose=False),
        config, aircraft_type=config.aircraft_type,
    )
    print(f"  {report.format()}", flush=True)
    by_id = {item.dataset_id: item for item in built}
    missing = [key for key in dataset_ids if key not in by_id]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(dataset_ids)} drawn flights could not be rebuilt from "
            f"{manifest} (first {missing[0]!r}) — a readout over a silent subset is a different draw"
        )
    return [by_id[key] for key in dataset_ids], manifest


def readings_by_flight(artefact: Path, split: str, sha256: str) -> dict[str, dict[str, Any]]:
    """The artefact's sentences for one split, keyed by ``flight_id``; refused if they were read
    under a different vocabulary than the spec beside them."""
    path = artefact / f"sentences_{split}.json"
    if not path.exists():
        raise SystemExit(f"{path} is missing: --split {split} is not in this artefact")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["vocabulary_sha256"] != sha256:
        raise SystemExit(
            f"{path} was read under vocabulary {payload['vocabulary_sha256'][:12]}…, "
            f"but the artefact's spec is {sha256[:12]}… — these are not one vocabulary"
        )
    return {item["flight_id"]: item for item in payload["flights"]}


def _round(values: np.ndarray, digits: int) -> list[float]:
    """A column at display precision — the view plots kilometres and degrees, and full float64
    would triple the file for digits nothing draws."""
    return [round(float(value), digits) for value in values]


def geodetic_columns(series, to_go_m, cross_m, height_m) -> dict[str, list[float]]:
    """A course-frame track as ``(lon, lat, altHaeM)`` — the 3D layer's input.

    THE VERTICAL DATUM IS CONVERTED HERE, on the way out, exactly as the CZML exporter does
    it: a record's altitude is MSL (observed ADS-B is ellipsoidal and is converted once at the
    `flight_scenarios` seam), while Cesium reads `cartographicDegrees` as metres above the
    WGS84 ELLIPSOID. Since h = H + N and N is NEGATIVE here (-33.5 m at KRDU), a line handed
    the MSL number renders |N| too HIGH — above its own terrain and above the observed CZML it
    is read against. The field is named `altHaeM` so nothing downstream has to remember which
    it got.

    The undulation comes from `flight_scenarios.datum.geoid_undulation_m` and is added by hand
    because the production MSL->HAE function lives in `aeroviz-4d/python/vertical_datum.py`,
    which the modeling tree must not import (and vice versa). Adding one to `flight_scenarios`
    is a change to a shared package, which is not this module's to make.

    The horizontal inverse is `course_frame_rows`' own algebra read backwards: with
    ``to_go = -(e·cosψ + n·sinψ)`` and ``cross = e·sinψ - n·cosψ``, the offsets from the
    threshold are ``e = -to_go·cosψ + cross·sinψ`` and ``n = -to_go·sinψ - cross·cosψ``.
    """
    psi = float(series.scenario.target.psi)
    cosine, sine = math.cos(psi), math.sin(psi)
    to_go = np.asarray(to_go_m, dtype=np.float64)
    cross = np.asarray(cross_m, dtype=np.float64)
    east = -to_go * cosine + cross * sine
    north = -to_go * sine - cross * cosine

    lons: list[float] = []
    lats: list[float] = []
    for east_m, north_m in zip(east, north):
        first, second = series.frame.from_world_horizontal(float(east_m), float(north_m))
        lat, lon = series.frame.latlon_from_horizontal(
            series.target_chart[0] + first, series.target_chart[1] + second
        )
        lons.append(lon)
        lats.append(lat)

    # The chart's vertical axis is measured from the FRAME's anchor, not from sea level
    # (`channels.py`: u = altitude - frame.alt0, and its inverse adds alt0 back). So an
    # absolute MSL altitude is the anchor's elevation, plus the threshold's height above it,
    # plus the height above the threshold. Leaving the anchor out put touchdown at HAE -4 m.
    msl = (
        float(series.frame.alt0)
        + float(series.target_chart[2])
        + np.asarray(height_m, dtype=np.float64)
    )
    hae = msl + np.asarray(geoid_undulation_m(lats, lons), dtype=np.float64)
    return {
        "lon": [round(value, 7) for value in lons],
        "lat": [round(value, 7) for value in lats],
        "altHaeM": [round(float(value), 1) for value in hae],
        # The same conversion as an OFFSET, so another height column over the same ground track
        # (the vertical band's two edges) becomes HAE by adding it. Recomputing the geodesy per
        # edge would be a second inverse of the same rows, and the two could drift.
        "_haeOffsetM": (hae - np.asarray(height_m, dtype=np.float64)).tolist(),
    }


def hae_from_height(offset: list[float], height_m) -> list[float]:
    """A height above the threshold as HAE, over the ground track `geodetic_columns` just did."""
    return [round(float(value), 1)
            for value in np.asarray(offset, dtype=np.float64) + np.asarray(height_m, dtype=np.float64)]


def cumulative_path_m(times: np.ndarray, ground_speed_mps: np.ndarray) -> np.ndarray:
    """Cumulative horizontal distance along the track — the axis the VERTICAL word is read on.

    MIRRORS the accumulation in `instructions._profile`, which fits height against this and takes
    the word to be its slope: each step is the ROW'S OWN speed (floored at
    `MINIMUM_GROUND_SPEED_MPS`, as there) times the gap before it. The frontend draws the word as
    a ramp over this column, so it has to be the same quantity the labeller fitted over — which
    is why it is exported rather than re-derived on the other side of the wire.
    """
    step = np.maximum(np.asarray(ground_speed_mps)[1:], MINIMUM_GROUND_SPEED_MPS) * np.diff(times)
    return np.concatenate([[0.0], np.cumsum(step)])


def observed_track(frame: dict[str, Any], series) -> dict[str, Any]:
    """The flight in the FINAL APPROACH COURSE's frame — the same `course_frame` the labeller
    read the words from, so the charts and the words cannot disagree about where the aircraft
    was — plus the geodetic columns the 3D layer needs."""
    geodetic = geodetic_columns(series, frame["to_go_m"], frame["cross_m"], frame["height_m"])
    geodetic.pop("_haeOffsetM")
    return {
        **geodetic,
        "pathM": _round(cumulative_path_m(frame["t"], frame["ground_speed_mps"]), 1),
        "tS": _round(frame["t"] - frame["t"][0], 1),
        "toGoM": _round(frame["to_go_m"], 1),
        "crossM": _round(frame["cross_m"], 1),
        "heightM": _round(frame["height_m"], 1),
        "relCourseDeg": _round(frame["relative_course_deg"], 2),
        "courseUnwrappedDeg": _round(frame["course_unwrapped_deg"], 2),
        "groundSpeedMps": _round(frame["ground_speed_mps"], 2),
        "established": [int(value) for value in frame["established"]],
    }


def _camel(instruction: dict[str, Any]) -> dict[str, Any]:
    """`Instruction.to_dict()` with the frontend's key spelling. The VALUES are untouched: a
    target is the plateau's own value, not a rounded one."""
    return {
        "kind": instruction["kind"], "word": instruction["word"], "target": instruction["target"],
        "issuedS": instruction["issued_s"], "settledS": instruction["settled_s"],
        "clamped": instruction["clamped"],
    }


def _absorbed(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": item["kind"], "startS": item["start_s"], "endS": item["end_s"],
        "word": item["word"], "change": item["change"], "reason": item["reason"],
    }


def _rebuilt_reading(reading: dict[str, Any]) -> Reading:
    """The artefact's own sentence as a `Reading` — copied, never re-read (V19). The instruction
    list is left empty on purpose: flying reads `words_at`, and the instructions travel to the
    frontend straight from the artefact."""
    return Reading(
        dataset_id=reading["dataset_id"], flight_id=reading["flight_id"], instructions=(),
        event_times_s=np.asarray(reading["event_times_s"], dtype=np.float64),
        words=np.asarray(reading["words"], dtype=np.int64),
        runway=reading["runway"], established_from_start=reading["established_from_start"],
        duration_s=reading["duration_s"],
    )


def flown_sentence(
    reading: dict[str, Any], vocabulary: Vocabulary, frame: dict[str, Any],
) -> tuple[kinematics.GeometricTrack, dict[str, Any]]:
    """What the words alone say, flown by `instruction_kinematics` and measured against the
    aircraft that was actually there.

    The reading is rebuilt from the artefact's own event times and words — the sentence is NOT
    re-read from the track (V19), so this flies exactly the sentence the view shows.
    """
    rebuilt = _rebuilt_reading(reading)
    # THE seam V19 opens: the sentence is copied from the artefact while the track is rebuilt
    # here, so nothing but this line says they are the same flight's. They agree on all 40
    # today; a rebuild that ever drifts would draw one flight's words over another's track and
    # look entirely reasonable.
    rebuilt_s = float(frame["t"][-1] - frame["t"][0])
    if abs(rebuilt_s - reading["duration_s"]) > 0.05:
        raise SystemExit(
            f"{reading['flight_id']}: the artefact's sentence spans {reading['duration_s']:g} s but the "
            f"rebuilt track spans {rebuilt_s:g} s — these are not the same flight"
        )
    track = kinematics.fly(
        rebuilt, vocabulary, kinematics.Start.from_course_frame(frame), observed_s=rebuilt_s,
    )
    return track, {**track.to_dict(), **kinematics.gap_to_observed(track, frame)}


def geometric_track(
    reading: dict[str, Any], vocabulary: Vocabulary, frame: dict[str, Any], series,
    bands: bool = True,
) -> dict[str, Any]:
    """The flown sentence, the corridor its words allow, and the geodetic columns the 3D layer
    draws both from.

    The vertical band rides THIS track's own ground positions — the commanded angle never touches
    the horizontal step — so its two heights are converted with the offset `geodetic_columns`
    just computed rather than by inverting the same rows twice.

    ``bands=False`` for the PRIOR's sentence. The corridor says what the vocabulary's tolerance
    allows around a sentence; a second corridor on the same chart is two of them overlapping, and
    the question the model's line answers is which WORDS it said, not how much slack those words
    would have had. It is also half the bytes of the block.
    """
    track, payload = flown_sentence(reading, vocabulary, frame)
    geodetic = geodetic_columns(series, track.to_go_m, track.cross_m, track.height_m)
    offset = geodetic.pop("_haeOffsetM")
    rebuilt = _rebuilt_reading(reading)
    start = kinematics.Start.from_course_frame(frame)
    observed_s = float(frame["t"][-1] - frame["t"][0])
    if not bands:
        return {**payload, **geodetic}
    return {
        **payload,
        **geodetic,
        "verticalBand": {
            "heightLoM": [round(float(v), 1) for v in track.height_lo_m],
            "heightHiM": [round(float(v), 1) for v in track.height_hi_m],
            "altHaeLoM": hae_from_height(offset, track.height_lo_m),
            "altHaeHiM": hae_from_height(offset, track.height_hi_m),
        },
        "speedBand": kinematics.speed_band(rebuilt, vocabulary, start, observed_s),
    }


PRIOR_FILE = "prior.pt"
#: MIRROR of `instruction_prior.PRIOR_SCHEMA`.
PRIOR_SCHEMA = "ts-instruction-prior-v1"
#: What the prior was asked, in the file, because it is NOT what a reader assumes. At every event
#: the model saw the TRUTH's words and the TRUTH's state up to that point and said what the next
#: event would be. It did not generate the sentence: a free run feeds the model its own words and
#: an executor's state, which is the closed loop (design §4 gate 3) and is not this.
PRIOR_METHOD = "teacher-forced-next-word"


def load_prior(directory: Path, vocabulary_sha256: str):
    """The trained prior, refused unless it was trained on THIS vocabulary.

    The sha is the whole point of the check: the model's heads are one per kind and sized by the
    class counts, so a prior from another vocabulary would still load, still run, and still emit
    word indices — into a different vocabulary's classes.
    """
    import torch
    from ts_transformer.manoeuvre.context import TypeVocabulary
    from ts_transformer.manoeuvre.instruction_prior import InstructionPrior, PriorConfig

    path = directory / PRIOR_FILE
    if not path.exists():
        raise SystemExit(f"{path} is missing: --prior takes the directory a prior run wrote")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != PRIOR_SCHEMA:
        raise SystemExit(f"{path} has schema {payload.get('schema')!r}, not {PRIOR_SCHEMA!r}")
    if payload["vocabulary_sha256"] != vocabulary_sha256:
        raise SystemExit(
            f"{path} was trained on vocabulary {payload['vocabulary_sha256'][:12]}…, the artefact is "
            f"{vocabulary_sha256[:12]}… — the heads are sized by THAT vocabulary's classes"
        )
    config = PriorConfig.from_dict(payload["prior_config"])
    model = InstructionPrior(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, config, TypeVocabulary.from_dict(payload["types"]), payload


def prior_sentences(model, config, types, readings: list[dict[str, Any]], series: list[Any]) -> list[dict[str, Any]]:
    """What the prior says at every event of each drawn flight, and how sure it was.

    One batch, because the drawn set is forty flights. Position k's logits are the model's answer
    for event k + 1, so the said sentence is the truth's OPENING EVENT followed by the model's
    answers at positions 0 … E-2: the first event is given (it is what the model is asked from),
    and `givenEvents` says so in the file.

    The event TIMES stay the truth's. The duration word is predicted like every other kind and is
    carried here, but it is not used to place the events: each prediction was conditioned on the
    truth's state at the truth's instant, so re-timing the sentence by the model's own gaps would
    put its words at moments its conditioning never saw. What the model says about timing is
    therefore READ on the duration row, not flown.
    """
    import torch
    from ts_transformer.manoeuvre.instruction_prior import collate
    from ts_transformer.manoeuvre.instruction_sequences import flight_sequence

    sequences = [flight_sequence(item, _rebuilt_reading(reading))
                 for reading, item in zip(readings, series, strict=True)]
    with torch.no_grad():
        output = model(collate(sequences, config, types))

    said: list[dict[str, Any]] = []
    terminal_column = INSTRUCTION_KINDS.index("terminal")
    for index, sequence in enumerate(sequences):
        length = sequence.length
        words = [[int(v) for v in sequence.words[0]]]
        confidence = [[1.0] * len(INSTRUCTION_KINDS)]       # the opening event is given, not said
        for position in range(length - 1):
            row, sure = [], []
            for kind in INSTRUCTION_KINDS:
                probabilities = torch.softmax(output.logits[kind][index, position], dim=-1)
                word = int(probabilities.argmax())
                row.append(word)
                sure.append(round(float(probabilities[word]), 4))
            words.append(row)
            confidence.append(sure)
        landed = next((float(sequence.positions_s[k]) for k in range(1, length)
                       if words[k][terminal_column] == TERMINAL_LANDED), None)
        said.append({"words": words, "confidence": confidence, "givenEvents": 1, "landedAtS": landed})
    return said


def flight_payload(
    reading: dict[str, Any], series, stratum: str, vocabulary: Vocabulary,
    said: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One flight as the frontend reads it. ``words`` is copied from the artefact UNCHANGED —
    six columns in `INSTRUCTION_KINDS` order, which the reader checks positionally.

    ``said`` is what the prior said at each of this flight's events (`prior_sentences`). Its
    sentence is flown by the SAME kinematics as the truth's, on the truth's event times, so the
    two tracks differ only in the words — which is the comparison the view exists to draw.
    """
    frame = course_frame(series)
    prior = None
    if said is not None:
        prior = {**said,
                 "geometric": geometric_track({**reading, "words": said["words"]}, vocabulary,
                                              frame, series, bands=False)}
    return {
        "flightKey": reading["flight_id"],
        "callsign": reading["flight_id"].split("_", 1)[0],
        "runway": reading["runway"],
        "stratum": stratum,
        "durationS": reading["duration_s"],
        "establishedFromStart": reading["established_from_start"],
        "sentence": {
            "eventTimesS": reading["event_times_s"],
            "words": reading["words"],
            "durationClamped": reading["duration_clamped"],
        },
        "instructions": [_camel(item) for item in reading["instructions"]],
        "absorbed": [_absorbed(item) for item in reading["absorbed"]],
        "observed": observed_track(frame, series),
        "geometric": geometric_track(reading, vocabulary, frame, series),
        **({"prior": prior} if prior is not None else {}),
    }


def update_index(directory: Path, airport: str, entry: dict[str, Any]) -> Path:
    """Add or replace this set in the parent's manifest, keeping every other set. A manifest of
    another schema is refused rather than rewritten — it is not ours to reinterpret."""
    path = directory / INDEX_FILE
    sets: list[dict[str, Any]] = []
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != INDEX_SCHEMA:
            raise SystemExit(f"{path} has schema {payload.get('schema')!r}, not {INDEX_SCHEMA!r}")
        if payload.get("airport") != airport:
            raise SystemExit(f"{path} is {payload.get('airport')!r}'s manifest, not {airport}'s")
        sets = [item for item in payload["sets"] if item.get("id") != entry["id"]]
    sets.append(entry)
    sets.sort(key=lambda item: item["id"])
    write_json_atomic(path, {"schema": INDEX_SCHEMA, "writtenUtc": utc_now(), "airport": airport, "sets": sets})
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--vocabulary", type=Path, required=True,
                        help="the artefact DIRECTORY (or its instruction_vocabulary.json)")
    parser.add_argument("--prior", type=Path, default=None,
                        help="a prior run's directory: adds what the model says at every event, and its track")
    parser.add_argument("--out", type=Path, required=True, help="…/airports/<ICAO>/training/<set id>")
    parser.add_argument("--flights", type=int, default=40, help="drawn half per stratum (default 40)")
    parser.add_argument("--split", default="train", choices=("train", "val"))
    parser.add_argument("--seed", type=int, default=1337, help="the draw's seed; it is written into the manifest")
    parser.add_argument("--set-id", default=None, help="defaults to the output directory's name")
    parser.add_argument("--title", default=None)
    args = parser.parse_args(argv)

    if args.flights % 2:
        parser.error(f"--flights draws half per stratum: {args.flights} is odd")
    artefact = args.vocabulary if args.vocabulary.is_absolute() else REPO_ROOT / args.vocabulary
    if artefact.name == VOCABULARY_FILE:
        artefact = artefact.parent
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if out.exists():
        parser.error(f"{out} exists; a sample export is never overwritten")
    training = out.parent                          # …/airports/<ICAO>/training/<set> → training
    airport = training.parent.name                 # → <ICAO>
    set_id = args.set_id or out.name
    started = time.perf_counter()

    vocabulary, runways, payload = load_vocabulary(artefact / VOCABULARY_FILE)
    readings = readings_by_flight(artefact, args.split, vocabulary.sha256)
    if True:
        # A pooled artefact holds five airports' sentences; ONE TRAINING SET IS ONE AIRPORT'S, and
        # the airport is the one the --out path names. The runway word carries the prefix, so this
        # is the artefact's own answer to "whose flight is this", not a guess from the flight id.
        readings = {key: row for key, row in readings.items()
                    if row["runway"].split(":")[0] == airport}
        if not readings:
            raise SystemExit(f"no flight in sentences_{args.split}.json lands at {airport}")
    per_stratum = args.flights // 2
    candidates = drawn_flights(readings, per_stratum, args.seed)

    config = TSConfig()
    print(f"  {len(candidates)} candidates drawn (seed {args.seed}); rebuilding their tracks to stratify", flush=True)
    rebuilt, manifest = series_by_dataset_id(
        airport, [readings[flight_id]["dataset_id"] for flight_id in candidates], config)
    draw = stratify(candidates, rebuilt, config, per_stratum)
    print(f"  {len(draw)} flights drawn ({per_stratum} per stratum)", flush=True)

    said_by_flight: list[dict[str, Any] | None] = [None] * len(draw)
    prior_block = None
    if args.prior:
        prior_dir = args.prior if args.prior.is_absolute() else REPO_ROOT / args.prior
        model, prior_config, types, prior_payload = load_prior(prior_dir, vocabulary.sha256)
        said_by_flight = prior_sentences(
            model, prior_config, types,
            [readings[flight_id] for flight_id, _, _ in draw], [item for _, _, item in draw])
        print(f"  the prior said {sum(len(s['words']) for s in said_by_flight)} events over {len(draw)} flights", flush=True)
        # WHICH flights the model has seen. The draw is one split; a set drawn from `train` shows
        # the model reciting what it was fitted on, and the difference matters enough to be in
        # the file rather than in whoever remembers the command.
        trained_on = set(prior_payload["split"]["train"])
        drawn_ids = {readings[flight_id]["dataset_id"] for flight_id, _, _ in draw}
        prior_block = {
            "sha256": file_sha256(prior_dir / PRIOR_FILE),
            "method": PRIOR_METHOD,
            "seed": prior_payload["settings"]["seed"],
            "bestEpoch": prior_payload["best_epoch"],
            "trainedOnTheseFlights": len(drawn_ids & trained_on),
            "readout": json.loads((prior_dir / "readings.json").read_text(encoding="utf-8"))["readings"],
        }

    flights = [flight_payload(readings[flight_id], item, stratum, vocabulary, said)
               for (flight_id, stratum, item), said in zip(draw, said_by_flight, strict=True)]
    spec = payload["spec"]
    out.mkdir(parents=True)
    write_json_atomic(out / SAMPLE_FILE, {
        "schema": SAMPLE_SCHEMA, "setId": set_id, "airport": airport, "writtenUtc": utc_now(),
        "kinds": list(INSTRUCTION_KINDS),
        "geometry": kinematics.assumptions(),
        **({"prior": prior_block} if prior_block is not None else {}),
        "vocabulary": {
            "sha256": vocabulary.sha256, "runwaySha256": runway_sha256(runways),
            "readingRule": spec["reading_rule"], "tokenStepS": spec["token_step_s"],
            "headingBinDeg": spec["heading_bin_deg"],
            "verticalModesDeg": list(spec["vertical_modes_deg"]),
            "verticalSegments": spec["vertical_segments"],
            "speedCentresMps": list(spec["speed_centres_mps"]),
            "verticalLevelToleranceDeg": spec["vertical_level_tolerance_deg"],
            "verticalToleranceFraction": spec["vertical_tolerance_fraction"],
            "speedToleranceFraction": spec["speed_tolerance_fraction"],
            "durationBinS": spec["duration_bin_s"], "durationMaxS": spec["duration_max_s"],
            "runwayIdents": list(runways.idents),
            "words": word_counts(vocabulary, runways),
        },
        "flights": flights,
    })

    entry = {
        "id": set_id, "kind": KIND_PRIOR if prior_block is not None else KIND_READBACK,
        "title": args.title or f"Instruction vocabulary · {spec['reading_rule']} · {args.split}",
        "file": f"{out.name}/{SAMPLE_FILE}",
        "vocabularySha256": vocabulary.sha256, "runwaySha256": runway_sha256(runways),
        "readingRule": spec["reading_rule"], "flights": len(flights),
        # the draw is stated, never implied: these ARE hand-check pages, and which ones
        "cohort": {"split": args.split, "perStratum": per_stratum, "seed": args.seed,
                   "drawnFrom": (f"a seeded permutation of the {args.split} split"
                                 + f" at {airport}"
                                 + f", stratified by approach_difficulty at the executor's anchor "
                                   f"(pool {len(candidates)})")},
        # The MODEL, in the manifest and not only in the sample: the picker has to
        # say which model a set carries before anyone downloads ten megabytes of it.
        **({"prior": {"sha256": prior_block["sha256"], "seed": prior_block["seed"],
                      "method": prior_block["method"]}} if prior_block is not None else {}),
        "source": {"artefact": str(artefact), "manifest": str(manifest),
                   "manifestSha256": file_sha256(manifest)},
    }
    index = update_index(training, airport, entry)
    print(f"  wrote {out / SAMPLE_FILE} and {index} in {time.perf_counter() - started:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
