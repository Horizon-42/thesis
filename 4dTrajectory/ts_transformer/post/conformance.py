"""The identity of the edge features (post-training §4 item 1, D21; as vocabulary D73): fixed reference scenes give the
same tokens. The reference holds, for each of its steps, the inputs of `edges.tokens` (the commanded aircraft and the
other aircraft at the step, their airport) and the tokens written with the code; every process that computes tokens
reads them again first (`require_conforming_edges`) and refuses a difference. No digest of code.

The reference steps come from windows of the train days drawn with seed 1337 (`reference_steps`): the commanded
aircraft on its own record, its R in force after its first predicted step, at its first predicted step and every
`REFERENCE_EVERY_S` after it while its record lasts.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.io_utils import write_bytes_atomic
from ts_transformer.post.edges import EDGES_SCHEMA, TOKEN_FEATURES, tokens
from ts_transformer.post.runways import airport_separation
from ts_transformer.post.scene import AircraftAt, Window

EDGES_REFERENCE_SCHEMA = "post-edges-reference-v1"
#: The windows of each airport in a reference, and the time between two of a window's reference steps, s.
REFERENCE_WINDOWS, REFERENCE_EVERY_S = 10, 60.0
REFERENCE_SEED = 1337
#: The largest difference of a token feature from its reference (the features are of order 1; float32 tokens).
TOLERANCE = 1e-6
_FIELDS = {"schema", "edges_schema", "features", "step_s", "geometries", "airport", "offsets", "own_at", "own_before", "own_known",
           "own_runway", "own_category", "at", "before", "known", "runway", "category", "tokens"}


@dataclass(frozen=True)
class ReferenceStep:
    airport: str
    own: AircraftAt
    others: AircraftAt


@dataclass(frozen=True)
class Checked:
    steps: int
    tokens: int
    max_difference: float


def reference_steps(windows: Sequence[Window], rng: np.random.Generator, per_airport: int = REFERENCE_WINDOWS,
                    every_s: float = REFERENCE_EVERY_S) -> list[ReferenceStep]:
    """``per_airport`` windows of each airport (drawn by ``rng``, in the airports' order) and their reference steps
    (module docstring)."""
    by_airport: dict[str, list[Window]] = {}
    for window in windows:
        by_airport.setdefault(window.scene.geometry.code, []).append(window)
    out = []
    for code in sorted(by_airport):
        items = by_airport[code]
        for k in sorted(rng.choice(len(items), size=min(per_airport, len(items)), replace=False)):
            window = items[int(k)]
            own, interval = window.commanded, window.scene.interval_s
            time_s = window.first_step_s
            while time_s <= own.end_s:
                out.append(ReferenceStep(code, AircraftAt.of([own.at_step(time_s, interval)]),
                                         window.scene.others_at(time_s, own.key)))
                time_s += round(every_s / interval) * interval
    return out


def _categories(values: Sequence[str | None]) -> np.ndarray:
    return np.asarray(["" if v is None else v for v in values], dtype=np.str_)


def write_edge_reference(path: Path, steps: Sequence[ReferenceStep], geometries: Mapping[str, AirportGeometry],
                         step_s: float) -> None:
    """The reference at ``path`` (new): ``steps``, the geometry of their airports (so that the check reads fixed inputs,
    whatever artefact is at hand) and their tokens by this code."""
    if path.exists():
        raise FileExistsError(f"{path} exists; a reference is written once")
    if not steps:
        raise ValueError("a reference of no step checks nothing")
    separations = {code: airport_separation(g) for code, g in geometries.items()}
    expected = [tokens(s.own, s.others, geometries[s.airport], separations[s.airport], step_s) for s in steps]
    sizes = np.array([len(s.others) for s in steps], dtype=np.int64)

    def stacked(name: str) -> np.ndarray:
        return np.concatenate([getattr(s.others, name) for s in steps])

    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    airports = sorted({s.airport for s in steps})
    np.savez_compressed(
        buffer, schema=np.array(EDGES_REFERENCE_SCHEMA), edges_schema=np.array(EDGES_SCHEMA),
        features=np.asarray(TOKEN_FEATURES, dtype=np.str_), step_s=np.array(step_s),
        geometries=np.array(json.dumps({code: geometries[code].to_dict() for code in airports})),
        airport=np.asarray([s.airport for s in steps], dtype=np.str_),
        offsets=np.concatenate(([0], np.cumsum(sizes))).astype(np.int64),
        own_at=np.concatenate([s.own.at for s in steps]), own_before=np.concatenate([s.own.before for s in steps]),
        own_known=np.concatenate([s.own.known for s in steps]),
        own_runway=np.concatenate([s.own.runway_index for s in steps]),
        own_category=_categories([s.own.category[0] for s in steps]),
        at=stacked("at"), before=stacked("before"), known=stacked("known"), runway=stacked("runway_index"),
        category=_categories([c for s in steps for c in s.others.category]),
        tokens=np.concatenate(expected) if sizes.sum() else np.zeros((0, len(TOKEN_FEATURES)), dtype=np.float32))
    write_bytes_atomic(path, buffer.getvalue())


def _aircraft(at, before, known, runway, category) -> AircraftAt:
    count = len(known)
    return AircraftAt(keys=tuple(str(k) for k in range(count)), at=np.asarray(at, dtype=np.float64).reshape(-1, 3),
                      before=np.asarray(before, dtype=np.float64).reshape(-1, 3), known=np.asarray(known, dtype=bool),
                      runway_index=np.asarray(runway, dtype=np.int64),
                      category=tuple(None if c == "" else str(c) for c in category),
                      last_step=np.zeros(count, dtype=bool))


def require_conforming_edges(path: Path) -> Checked:
    """Compute the tokens of the reference at ``path`` again with the code on disk, on the reference's own inputs and
    geometry; refused unless the reference is of this schema and of these features and every token is finite and within
    `TOLERANCE` of its reference."""
    with np.load(path) as arrays:
        data = {name: arrays[name] for name in arrays.files}
    if set(data) != _FIELDS or str(data["schema"]) != EDGES_REFERENCE_SCHEMA:
        raise ValueError(f"{path} is not a {EDGES_REFERENCE_SCHEMA} file")
    if str(data["edges_schema"]) != EDGES_SCHEMA or tuple(data["features"].tolist()) != TOKEN_FEATURES:
        raise ValueError(f"{path} holds the tokens of {str(data['edges_schema'])}, not {EDGES_SCHEMA}")
    geometries = {code: AirportGeometry.from_dict(g) for code, g in json.loads(str(data["geometries"])).items()}
    separations = {code: airport_separation(g) for code, g in geometries.items()}
    step_s, offsets = float(data["step_s"]), data["offsets"]
    worst = 0.0
    for s, code in enumerate(data["airport"].tolist()):
        rows = slice(int(offsets[s]), int(offsets[s + 1]))
        own = _aircraft(data["own_at"][s], data["own_before"][s], data["own_known"][s:s + 1],
                        data["own_runway"][s:s + 1], data["own_category"][s:s + 1])
        others = _aircraft(data["at"][rows], data["before"][rows], data["known"][rows], data["runway"][rows],
                           data["category"][rows])
        got = tokens(own, others, geometries[code], separations[code], step_s)
        if len(got):
            difference = np.abs(got.astype(np.float64) - data["tokens"][rows].astype(np.float64))
            if not np.isfinite(difference).all():
                raise ValueError(f"the edge features or their reference {path} are not finite at step {s}: the code "
                                 "that computes them does not conform")
            worst = max(worst, float(difference.max()))
    if worst > TOLERANCE:
        raise ValueError(f"the edge features differ from their reference {path} by up to {worst:.3g} "
                         f"(tolerance {TOLERANCE:g}): the code that computes them does not conform")
    return Checked(steps=len(data["airport"]), tokens=int(offsets[-1]), max_difference=worst)
