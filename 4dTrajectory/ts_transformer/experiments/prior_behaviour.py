"""B5's behaviour check of the prior's code (prior design §12 B5; D108): the losses of two steps of training on a fixed
synthetic set, and the words and probabilities the speaker says with fixed numbers — on the CPU, with one thread, as
JSON. With them: the configuration values a run reads from the code (`TrainConfig`'s and `PriorConfig`'s defaults:
`prior_train`'s), the first-step runway readout of the trained model, the teacher-forced loss of the variant
`constants`, the inputs of fixed rows (`inputs.state_inputs` of both variants with the landings counted by a
`LandingIndex`, `inputs.Heard.inputs` after the words said), and the step of a speaker's closed loop
(`prior_speaking_loop.SpeakingLoop`) on an executor that flies straight (`Straight`): its words, states and records. `prior_campaign` runs it as its own process at its start and before each step, and stops when the answer
differs: results of different code are compared once the code is shown to behave the same on fixed inputs, never by
an equal commit (the user, 2026-10-02; no code fingerprint, 2026-10-04).

The fixed inputs: the artefact's vocabulary spec and its first airport's candidates and finals (the procedure masks
the speaker speaks under), configuration A's shape (`PriorConfig`'s defaults) and training values (`TrainConfig`'s), a
set of sentences drawn from a fixed seed (`fixed_sentences`), the speaker's numbers from a fixed seed. The answer
holds every number as its float's hex, compared exactly.

    python run_ts.py prior_behaviour --instructions <artefact>
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from flight_scenarios.procedure_final import DEFAULT_PROCEDURE_ROOT
from ts_transformer.data.day_split import DaySplit
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.artefact import load_candidates, load_day_split, load_spec
from ts_transformer.experiments.prior_speaking_loop import SpeakingLoop, flight_numbers
from ts_transformer.instructions.words import COLUMNS, RUNWAY, UNCHANGED, Words
from ts_transformer.prior.batch import (
    CANDIDATE_MOTION_FEATURES, IN_FORCE_WORDS, OWN_FEATURES, OWN_MOTION_FEATURES, SentenceRows, collate,
    variant_features,
)
from ts_transformer.prior.inputs import Heard, state_inputs
from ts_transformer.prior.landings import Landing, LandingIndex
from ts_transformer.prior.model import Prior, PriorConfig
from ts_transformer.prior.procedure import Final, airport_finals
from ts_transformer.prior.runs import Run, RunData
from ts_transformer.prior.speaker import Position, Speaker
from ts_transformer.prior.train import TrainConfig, batch_nll, first_step_runway, train
from ts_transformer.repo_layout import REPO_ROOT

#: The fixed set: sentences, rows, the first predicted step's row, the rows said and the seed.
SENTENCES, ROWS, FIRST, SAID, SEED = 4, 24, 6, 12, 1337


def fixed_sentences(words: Words, candidates: int, word_values: Sequence[int], *, variant: str = "full",
                    seed: int = SEED) -> list[SentenceRows]:
    """`SENTENCES` synthetic sentences of ``candidates`` candidates (`batch.SentenceRows`'s contract: row 0 without
    motion, nothing in force to the first predicted step, which says every column), their inputs and words drawn with
    ``seed``; ``word_values`` the model's counts of each word column."""
    rng = np.random.default_rng(seed)
    features = variant_features(variant)
    width = len(features)
    out = []
    for k in range(SENTENCES):
        time_s = np.arange(ROWS, dtype=np.float32) * 4.0
        own = rng.normal(size=(ROWS, len(OWN_FEATURES))).astype(np.float32)
        own[:, OWN_FEATURES.index("no_motion")] = 0.0
        own[0, [OWN_FEATURES.index(name) for name in OWN_MOTION_FEATURES]] = 0.0
        own[0, OWN_FEATURES.index("no_motion")] = 1.0
        vectors = rng.normal(size=(ROWS, candidates, width)).astype(np.float32)
        vectors[0][:, [features.index(name) for name in CANDIDATE_MOTION_FEATURES]] = 0.0
        targets = np.full((ROWS, len(COLUMNS)), UNCHANGED, dtype=np.int64)
        targets[FIRST] = [rng.integers(candidates), *(rng.integers(n) for n in word_values)]
        for r in range(FIRST + 1, ROWS):
            column = int(rng.integers(1, len(COLUMNS)))
            if rng.random() < 0.4:
                targets[r, column] = rng.integers(word_values[column - 1])
        runway = np.full(ROWS, -1, dtype=np.int64)
        runway[FIRST + 1:] = targets[FIRST, RUNWAY]
        heading = np.zeros((ROWS, 2), dtype=np.float32)
        heading[FIRST + 1:] = (0.6, 0.8)
        in_force = np.full((ROWS, len(IN_FORCE_WORDS)), -1, dtype=np.int64)
        in_force[FIRST + 1:] = targets[FIRST, 2:]
        since = np.zeros((ROWS, len(COLUMNS)), dtype=np.float32)
        since[FIRST + 1:] = rng.random((ROWS - FIRST - 1, len(COLUMNS))).astype(np.float32)
        out.append(SentenceRows(flight_key=f"BEHAVIOUR:{k}", airport="BEHAVIOUR", split="train", first_step=FIRST,
                                time_s=time_s, own=own, candidates=vectors, runway_in_force=runway,
                                go_around=np.zeros(ROWS, dtype=bool), heading_in_force=heading,
                                words_in_force=in_force, since=since, targets=targets))
    return out


def hexed(values: Any) -> list:
    """Numbers as the hex of their floats (exact), nested as given."""
    return np.vectorize(lambda v: float(v).hex(), otypes=[object])(np.asarray(values, dtype=np.float64)).tolist()


def fixed_inputs(geometry: AirportGeometry, days: DaySplit) -> dict[str, Any]:
    """The inputs of fixed rows of ``geometry``'s airport (module docstring): the states of a descent along its first
    candidate's extended course, the landings of fixed flights on a train day of ``days`` counted before each row."""
    day = days.days["train"][0]
    noon = float(np.datetime64(f"{day}T12:00:00", "s").astype(np.int64))
    times = noon + 2.0 * np.arange(ROWS)
    runways = tuple(c.ident for c in geometry.candidates)
    index = LandingIndex(runways, tuple(Landing(noon - 600.0 + 30.0 * k, runways[k % len(runways)], f"L{k}")
                                        for k in range(20)), 0, days)
    counts = index.counts_before(times, without="L0")
    at = np.stack([-12_000.0 + 140.0 * np.arange(ROWS), 30.0 * np.sin(np.arange(ROWS)),
                   geometry.elevation_m + 900.0 - 7.0 * np.arange(ROWS)], axis=1)
    before = np.concatenate((at[:1], at[:-1]))
    out = {"landings": counts.tolist()}
    for variant in ("full", "constants"):
        own, candidates = state_inputs(at, before, np.arange(ROWS) > 0, counts, geometry, variant, 2.0)
        out[variant] = {"own": hexed(own), "candidates": hexed(candidates)}
    return out


class Straight:
    """An executor of a loop (the start's `Loop`, as `SpeakingLoop` reads it) whose flights fly on in a straight line,
    120 m east and 6 m lower each 2 s row, flight b done after ``3 + 2 b`` rows."""

    def __init__(self, starts: np.ndarray, every: int, most_go_arounds: int) -> None:
        self.state, self.every, self.most_go_arounds, self.row_cycles = starts.copy(), every, most_go_arounds, 2
        self.ends = 3 + 2 * np.arange(len(starts))
        self.steps = 0
        self.halted = np.zeros(len(starts), dtype=bool)
        self.executor = type("Done", (), {"done_cycle": torch.as_tensor(2 * every * self.ends - 1)})()

    def rows(self) -> np.ndarray:
        return self.state.copy()

    def step(self, words_row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        flown = []
        for _ in range(self.every):
            self.state[~self.halted, 0] += 120.0
            self.state[~self.halted, 2] -= 6.0
            flown.append(self.state.copy())
        self.steps += 1
        return np.stack(flown, axis=1), self.steps >= self.ends

    def halt(self, flights: np.ndarray) -> None:
        self.halted |= flights


def speaking(model: Prior, words: Words, finals: Sequence[Final], days: DaySplit) -> dict[str, Any]:
    """Three flights of ``finals``' airport in `SpeakingLoop` on `Straight` at Δ = 4 s (module docstring)."""
    geometry = finals[0].geometry
    count, every, start = 3, 2, 4
    noon = float(np.datetime64(f"{days.days['train'][0]}T12:00:00", "s").astype(np.int64))
    states = np.zeros((count, start * every + 1, 6))
    states[:, :, 0] = -14_000.0 + 120.0 * np.arange(start * every + 1)[None] + 2_000.0 * np.arange(count)[:, None]
    states[:, :, 2] = geometry.elevation_m + 900.0 - 6.0 * np.arange(start * every + 1)[None]
    rows = {b: type("Rows", (), {"rows": type("R", (), {"start": start, "first_row": 0, "states": states[b]})()})()
            for b in range(count)}
    flights = {b: {"airport": geometry.code, "entry_time_utc": f"{days.days['train'][0]}T11:50:00Z",
                   "dataset_id": f"{geometry.code}:L{b}"} for b in range(count)}
    runways = tuple(c.ident for c in geometry.candidates)
    landings = LandingIndex(runways, tuple(Landing(noon + 10.0 * b, runways[0], f"L{b}") for b in range(count)), 0,
                            days)
    loop = SpeakingLoop(model, Straight(states[:, -1], every, 2), list(range(count)), rows, flights,
                        {geometry.code: geometry}, [landings] * count, {geometry.code: finals}, words, interval_s=4.0,
                        variant="full", device=torch.device("cpu"))
    while loop.observing:
        loop.observe()
    numbers = [flight_numbers(SEED, 0, b) for b in range(count)]
    while loop.alive.any():
        loop.step(np.stack([n.random(len(COLUMNS)) for n in numbers]))
    return {"words": [loop.said(b).tolist() for b in range(count)],
            "states": [hexed(loop.states(b)) for b in range(count)],
            "probabilities": hexed(np.stack(loop.speaker.drawn_probability))}


def behaviour(words: Words, finals: Sequence[Final], days: DaySplit) -> dict[str, Any]:
    """The answer of the check (module docstring) on one thread of the CPU; ``finals`` one airport's, ``days`` the
    artefact's day split."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cpu = torch.device("cpu")
        torch.manual_seed(SEED)
        model = Prior(PriorConfig.from_words(words, "full"))
        sentences = fixed_sentences(words, len(finals), model.config.word_values)
        # two steps: two epochs of one batch each (every sentence in one batch), stopped on the same sentences
        config = replace(TrainConfig(), max_epochs=2, seed=SEED, tokens_per_batch=SENTENCES * ROWS)
        data = RunData(Run(("BEHAVIOUR",)), sentences, [replace(s, split="select") for s in sentences])
        result = train(model, data, config, cpu, lambda line: None)
        model.eval()
        rows = collate(sentences, cpu)
        speaker = Speaker(model, words, [finals] * SENTENCES, capacity=8)
        at = [Position(np.full(SENTENCES, -15_000.0 + 400.0 * r) + 3_000.0 * np.arange(SENTENCES),
                       np.zeros(SENTENCES), np.full(SENTENCES, 900.0 - 10.0 * r)) for r in range(FIRST + SAID)]
        speaker.observe(rows.between(0, FIRST), at[:FIRST])
        numbers = np.random.default_rng(SEED).random((SAID, SENTENCES, len(COLUMNS)))
        said = [speaker.speak(rows.between(r, r + 1), at[r], numbers[r - FIRST]).tolist()
                for r in range(FIRST, FIRST + SAID)]
        runway = first_step_runway(model, sentences, config.tokens_per_batch, cpu)
        heard = Heard(finals[0].geometry, words)
        for r, row in enumerate(said):
            heard.hear(np.array(row[0]), float(at[FIRST + r].height_m[0]), (FIRST + r) * 4.0)
        runway_in_force, go_around, heading, levels, since = heard.inputs((FIRST + SAID) * 4.0)
        heard_inputs = {"runway": int(runway_in_force), "go_around": bool(go_around), "heading": hexed(heading),
                        "words": levels.tolist(), "since": hexed(since)}
        torch.manual_seed(SEED)
        constants = Prior(PriorConfig.from_words(words, "constants")).eval()
        with torch.no_grad():
            nll, asked = batch_nll(constants, collate(fixed_sentences(words, len(finals), constants.config.word_values,
                                                                      variant="constants"), cpu))
        inputs = fixed_inputs(finals[0].geometry, days)
        loop = speaking(model, words, finals, days)
    finally:
        torch.set_num_threads(threads)
    return {"train_config": asdict(TrainConfig()), "model_config": model.config.to_dict(),
            "first_step_runway": runway, "constants_loss": hexed(nll / asked),
            "heard": heard_inputs,
            "inputs": inputs, "speaking_loop": loop,
            "train_loss": hexed([row["train_loss_per_step"] for row in result.history]),
            "select_loss": hexed([row["select_loss_per_step"] for row in result.history]),
            "words": said, "probabilities": hexed(np.stack(speaker.drawn_probability))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--procedure-root", type=Path, default=DEFAULT_PROCEDURE_ROOT)
    args = parser.parse_args(argv)
    instructions = args.instructions if args.instructions.is_absolute() else REPO_ROOT / args.instructions
    geometries = load_candidates(instructions)
    first = geometries[sorted(geometries)[0]]
    print(json.dumps(behaviour(Words(load_spec(instructions)), airport_finals(first, root=args.procedure_root),
                               load_day_split(instructions))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
