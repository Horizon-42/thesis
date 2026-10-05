"""The sentence artefact as the prior's sentences (prior design §5, §8; milestone B1): a `runs.SentenceSource` over the
closed-loop sentences of one row interval Δ, and the artefact's identity that a checkpoint records.

Read-only, through the vocabulary's public interface (vocabulary §6 items 3, 4): the spec, the day split, the
candidates and their vertical paths (`candidates.json`), the split's flight records, its sentence file (the strata)
and its closed-loop file (the sentences and their outcomes). A flight without a closed-loop sentence is not read (§12
B1), nor one the selection leaves out (D75, `prior.selection`). The test days are never asked for: the artefact holds only the
development splits, and the prior asks only for train, select and (for the base's one readout) val. Whether today's
code may read the closed-loop sentences (their passed conformance record) is the runner's check: it imports the
executor, the prior does not.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ts_transformer.data.day_split import DaySplit
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.artefact import (
    SPLITS, closed_loop_path, closed_loop_sentences, load_candidates, load_day_split, load_spec, signals_flights,
)
from ts_transformer.instructions.words import Words
from ts_transformer.io_utils import file_sha256
from ts_transformer.prior.batch import SentenceRows
from ts_transformer.prior.inputs import sentence_rows
from ts_transformer.prior.landings import LandingIndex, read_roster_landings
from ts_transformer.prior.selection import Stored, kept, require_rule, selection_record
from ts_transformer.repo_layout import tracks_manifest_path


def stored_sentences(directory: Path, interval_s: float, split: str) -> list[Stored]:
    """What the selection reads of each closed-loop sentence of ``split`` at Δ (vocabulary §6 item 3): its flight's
    airport, and of the withheld fields (D82) its stratum (D70) and its stored outcome (D74)."""
    flights = signals_flights(directory, split)
    return [Stored(split, flights[index]["airport"], sentence.withheld.stratum, sentence.withheld.outcome)
            for index, sentence in closed_loop_sentences(directory, split, interval_s, load_spec(directory)).items()]


def artefact_identity(directory: Path, interval_s: float, landings: Mapping[str, LandingIndex],
                      selection: str) -> dict[str, Any]:
    """§8 item 1 (D21, D63, D75): the spec sha, the day split, the candidate table (`candidates.json`, the vertical paths
    in it), the sha256 of the closed-loop sentence files of Δ = ``interval_s``, every development split; the landings
    the candidate vectors count (`LandingIndex.digest`, by their flights); and the selection — its rule and, for each
    split, airport, stratum and outcome, the sentences kept and left out (`prior.selection.selection_record` of the rule
    ``selection``)."""
    return {"spec_sha256": load_spec(directory).sha256, "day_split": load_day_split(directory).to_dict(),
            "candidates": json.loads((directory / "candidates.json").read_text(encoding="utf-8")),
            "row_interval_s": interval_s,
            "sentence_files": {split: file_sha256(closed_loop_path(directory, split, interval_s)) for split in SPLITS},
            "landings": {code: index.digest() for code, index in sorted(landings.items())},
            "selection": selection_record(selection, (s for split in SPLITS
                                                     for s in stored_sentences(directory, interval_s, split)))}


def airport_landings(geometries: Mapping[str, AirportGeometry], days: DaySplit) -> dict[str, LandingIndex]:
    """Each airport's landings on its candidates from its tracks roster, less the sealed test days (`landings`)."""
    return {code: read_roster_landings(tracks_manifest_path(code), [c.ident for c in geometry.candidates], days)
            for code, geometry in geometries.items()}


class ArtefactSource:
    """The closed-loop sentences of an artefact at one Δ that the rule ``selection`` keeps (D75), as `SentenceRows` of a
    variant (module docstring); each split's file is read once, on first use. The artefact is not changed."""

    def __init__(self, directory: Path, interval_s: float, variant: str, landings: Mapping[str, LandingIndex],
                 selection: str) -> None:
        self.directory, self.interval_s, self.variant = directory, interval_s, variant
        self.selection = require_rule(selection)
        self.words = Words(load_spec(directory))
        self.geometries = load_candidates(directory)
        if set(landings) != set(self.geometries):
            raise ValueError(f"landings of {sorted(landings)}, the artefact's airports are {sorted(self.geometries)}")
        self.landings = dict(landings)
        self._split: dict[str, dict[str, list[SentenceRows]]] = {}

    def sentences(self, split: str, airport: str) -> list[SentenceRows]:
        if split not in SPLITS:
            raise ValueError(f"the prior reads the development splits {SPLITS}, not {split!r}")
        if airport not in self.geometries:
            raise ValueError(f"{airport} is not an airport of {self.directory}")
        if split not in self._split:
            self._split[split] = self._read(split)
        return self._split[split][airport]

    def _read(self, split: str) -> dict[str, list[SentenceRows]]:
        stored = closed_loop_sentences(self.directory, split, self.interval_s, self.words.spec)
        flights = signals_flights(self.directory, split)
        # every airport of the artefact, also one with no sentence in this split (a small split can have none)
        out: dict[str, list[SentenceRows]] = {code: [] for code in self.geometries}
        for index, sentence in stored.items():
            if not kept(self.selection, sentence.withheld.outcome):
                continue
            flight = flights[index]
            code = flight["airport"]
            out[code].append(sentence_rows(           # the rows alone are an input (D82)
                sentence.rows, flight, self.geometries[code], self.landings[code], self.words, interval_s=self.interval_s,
                split=split, variant=self.variant))
        return out
