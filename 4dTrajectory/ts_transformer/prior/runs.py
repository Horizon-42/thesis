"""The airports of a training run and the sentences it reads (prior design §5; D31, D39).

A run trains on the train days of its training airports and stops on their select days (D31). A fold of the
cross-validation (D39) holds one airport out: its training airports are the other four, and its held-out airport is
read only after the training, on its select days (`held_out_sentences`). No run reads the validation days: they are
read one time for each stage, by the readout of the base model, not here. The sentences come from a `SentenceSource`
(the artefact's, milestone B1), which this module asks only for the train and select days of the airports it names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np

from ts_transformer.prior.batch import SentenceRows

TRAIN, SELECT = "train", "select"


@dataclass(frozen=True)
class Run:
    """``airports``: every airport of the campaign; ``held_out``: the fold's held-out airport, None for a run on all."""

    airports: tuple[str, ...]
    held_out: str | None = None

    def __post_init__(self) -> None:
        if len(set(self.airports)) != len(self.airports):
            raise ValueError(f"airports {self.airports} repeat")
        if self.held_out is not None and self.held_out not in self.airports:
            raise ValueError(f"held-out airport {self.held_out} is not one of {self.airports}")
        if not self.training_airports:
            raise ValueError("a run needs a training airport")

    @property
    def training_airports(self) -> tuple[str, ...]:
        return tuple(a for a in self.airports if a != self.held_out)

    def to_dict(self) -> dict[str, Any]:
        return {"airports": list(self.airports), "held_out": self.held_out}


class SentenceSource(Protocol):
    def sentences(self, split: str, airport: str) -> list[SentenceRows]:
        """The sentences of one split of the day split at one airport."""


@dataclass(frozen=True)
class RunData:
    """What a run trains on and stops on; the constructor refuses a sentence of another split or airport."""

    run: Run
    train: list[SentenceRows]
    select: list[SentenceRows]

    def __post_init__(self) -> None:
        for split, sentences in ((TRAIN, self.train), (SELECT, self.select)):
            if not sentences:
                raise ValueError(f"no {split} sentence for {self.run.training_airports}")
            wrong = [s.flight_key for s in sentences
                     if s.split != split or s.airport not in self.run.training_airports]
            if wrong:
                raise ValueError(f"{len(wrong)} {split} sentences are not of the {split} days of "
                                 f"{self.run.training_airports}, e.g. {wrong[0]}")


def _read(source: SentenceSource, split: str, airports: Sequence[str]) -> list[SentenceRows]:
    return [sentence for airport in airports for sentence in source.sentences(split, airport)]


def run_data(source: SentenceSource, run: Run) -> RunData:
    """The train and select days of the run's training airports."""
    return RunData(run, _read(source, TRAIN, run.training_airports), _read(source, SELECT, run.training_airports))


def held_out_sentences(source: SentenceSource, run: Run) -> list[SentenceRows]:
    """The select days of a fold's held-out airport: the fold's score (§5), read after its training."""
    if run.held_out is None:
        raise ValueError("a run on every airport has no held-out airport")
    return _read(source, SELECT, (run.held_out,))


#: The seed of a smoke run's sample (D55): the same flights at every run.
SAMPLE_SEED = 1337


def sampled_run_data(source: SentenceSource, run: Run, per_airport: int, seed: int = SAMPLE_SEED) -> RunData:
    """`run_data` on a random sample of at most ``per_airport`` sentences of each training airport and split (a smoke
    run, D55: never the first sentences in order), drawn with ``seed``."""
    rng = np.random.default_rng(seed)

    def sample(split: str) -> list[SentenceRows]:
        out = []
        for airport in run.training_airports:
            sentences = source.sentences(split, airport)
            keep = np.sort(rng.choice(len(sentences), size=min(per_airport, len(sentences)), replace=False))
            out += [sentences[i] for i in keep]
        return out

    return RunData(run, sample(TRAIN), sample(SELECT))
