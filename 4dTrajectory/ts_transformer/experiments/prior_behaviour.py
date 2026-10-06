"""B5's behaviour check of the prior's code (prior design §12 B5; D108): the losses of two steps of training on a fixed
synthetic set, and the words and probabilities the speaker says with fixed numbers — on the CPU, with one thread, as
JSON. With them: the configuration values a run reads from the code (`TrainConfig`'s and `PriorConfig`'s defaults:
`prior_train`'s), the first-step runway readout of the trained model, the teacher-forced loss of the variant
`constants`, the inputs of fixed rows (`inputs.state_inputs` of both variants with the landings counted by a
`LandingIndex`, `inputs.Heard.inputs` after the words said), the inputs of a fixed sentence (`inputs.sentence_rows`,
its flight key of the real format), the step of a speaker's closed loop (`prior_speaking_loop.SpeakingLoop`) on an
executor that flies straight (`Straight`): its words, states and records; the selection (`selection.left_out`) over
every rule, stored outcome and mark; `prior_select`'s rules on a fixed table of scores (`select_rules`: a tie of
parameters, a score at exactly twice the seed scale, a seed scale of 0); and the campaign's settings
(`prior_campaign.settings`: the seeds, the selection, the configurations, free generation, the temperature, D68's
bound) as the code on the disk sets them (D108). `prior_campaign` runs it as its own process at its start and before
each step, and stops when the answer differs: results of different code are compared once the code is shown to behave
the same on fixed inputs, never by an equal commit (the user, 2026-10-02; no code fingerprint, 2026-10-04).

The fixed inputs: the artefact's vocabulary spec and its first airport's candidates and finals (the procedure masks
the speaker speaks under), configuration A's shape (`PriorConfig`'s defaults) and training values (`TrainConfig`'s), a
set of sentences drawn from a fixed seed (`fixed_sentences`), the speaker's numbers from a fixed seed. The answer
holds every number as its float's hex, compared exactly. The speaker and the speaking loop run with the procedure
masks in both modes (B14) and a difference is refused by name; the answer is the per-aircraft mode's.

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
from ts_transformer.autopilot.judge import OUTCOMES
from ts_transformer.instructions.artefact import STATE_COLUMNS, SentenceRows as ClosedLoopRows
from ts_transformer.instructions.artefact import load_candidates, load_day_split, load_spec
from ts_transformer.instructions.labeller.interval import on_interval_rows
from ts_transformer.experiments.prior_speaking_loop import SpeakingLoop, flight_numbers
from ts_transformer.instructions.words import COLUMNS, RUNWAY, UNCHANGED, Words
from ts_transformer.prior.batch import (
    CANDIDATE_MOTION_FEATURES, IN_FORCE_WORDS, OWN_FEATURES, OWN_MOTION_FEATURES, SentenceRows, collate,
    variant_features,
)
from ts_transformer.prior.inputs import Heard, sentence_rows, state_inputs
from ts_transformer.prior.landings import Landing, LandingIndex
from ts_transformer.prior.model import Prior, PriorConfig
from ts_transformer.prior.procedure import BATCH, MASK_MODES, PER_AIRCRAFT, Final, airport_finals
from ts_transformer.prior.runs import Run, RunData
from ts_transformer.prior.selection import RULES, kept, left_out, side
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


def speaking(model: Prior, words: Words, finals: Sequence[Final], days: DaySplit, masks: str = PER_AIRCRAFT
             ) -> dict[str, Any]:
    """Three flights of ``finals``' airport in `SpeakingLoop` on `Straight` at Δ = 4 s (module docstring), the procedure
    masks in the mode ``masks``."""
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
    # each aircraft its own landings (D105): the three flights' own landings after their rows (a scene's), and others
    # in the 30 min before its rows, on each candidate, one more than the aircraft before it
    landings = [LandingIndex(runways, tuple(sorted(
        (*(Landing(noon - 1_500.0 + 60.0 * k, runways[k % len(runways)], f"O{b}{k}") for k in range(3 + b)),
         *(Landing(noon + 10.0 * a, runways[0], f"L{a}") for a in range(count))), key=lambda landing: landing.time_s)),
        0, days) for b in range(count)]
    observed = {b: states[b, : start * every] for b in range(count)}
    loop = SpeakingLoop(model, Straight(states[:, -1], every, 2), list(range(count)), rows, observed, flights,
                        {geometry.code: geometry}, landings, {geometry.code: finals}, words, interval_s=4.0,
                        variant="full", device=torch.device("cpu"), masks=masks)
    while loop.observing:
        loop.observe()
    numbers = [flight_numbers(SEED, 0, b) for b in range(count)]
    while loop.alive.any():
        loop.step(np.stack([n.random(len(COLUMNS)) for n in numbers]))
    return {"words": [loop.said(b).tolist() for b in range(count)],
            "states": [hexed(loop.states(b)) for b in range(count)],
            "probabilities": hexed(np.stack(loop.speaker.drawn_probability))}


def fixed_sentence(said: Sequence[Sequence[int]], geometry: AirportGeometry, words: Words, days: DaySplit
                   ) -> dict[str, Any]:
    """The inputs and targets of a fixed closed-loop sentence (`inputs.sentence_rows`, both variants) at Δ = 4 s: the
    words ``said`` from its first predicted step (`FIRST`) where the speaker said them (`behaviour`'s first aircraft:
    the same heights, so the grammar takes them), a flight key of the real format
    (``<airport>:<id>_<runway>_<icao24>_<landing time>``) whose own landing the landings hold between two of its rows on
    the last candidate, with another landing at the same second on the first, others in the 30 min before its rows on
    each candidate, one at a row's time and one exactly 30 min before a row (the window's two edges)."""
    every, rows = 2, (FIRST + len(said) - 1) * 2 + 1
    states = np.zeros((rows, len(STATE_COLUMNS)))
    states[:, STATE_COLUMNS.index("e_m")] = -15_000.0 + 200.0 * np.arange(rows)
    states[:, STATE_COLUMNS.index("height_m")] = geometry.elevation_m + 900.0 - 5.0 * np.arange(rows)
    day = days.days["train"][0]
    runway = geometry.candidates[-1].ident
    flight = {"airport": geometry.code, "entry_time_utc": f"{day}T11:50:00Z",
              "dataset_id": f"{geometry.code}:BEH123_{runway}_abc123_{day.replace('-', '')}T115031Z"}
    entry = float(np.datetime64(f"{day}T11:50:00", "s").astype(np.int64))
    runways = tuple(c.ident for c in geometry.candidates)
    own = entry + 31.0                # between two of its Δ rows (2 s row 3 at 11:50:06, a Δ row every 4 s from it)
    landings = LandingIndex(runways, tuple(sorted(
        (*(Landing(entry - 1_700.0 + 97.0 * k, runways[k % len(runways)], f"L{k}") for k in range(17)),
         Landing(own, runway, flight["dataset_id"].partition(":")[2]), Landing(own, runways[0], "SAME_SECOND"),
         Landing(entry + 14.0, runways[0], "AT_A_ROW"), Landing(entry + 18.0 - 1_800.0, runways[-1], "WINDOW_EDGE")),
        key=lambda landing: landing.time_s)), 0, days)
    sentence = ClosedLoopRows(first_row=3, start=FIRST, grid=np.array(said, dtype=np.int16),
                              correction=np.zeros((len(said), len(COLUMNS)), dtype=bool), states=states,
                              on_interval=on_interval_rows(rows, every))
    out = {}
    for variant in ("full", "constants"):
        got = sentence_rows(sentence, flight, geometry, landings, words, interval_s=4.0, split="train", variant=variant)
        out[variant] = {name: (hexed(value) if np.asarray(value).dtype.kind == "f" else np.asarray(value).tolist())
                        for name, value in vars(got).items() if name not in ("flight_key", "airport", "split")}
    return out


def selection_rules() -> dict[str, Any]:
    """For every rule, stored outcome of the judge and mark of a faulty track: why the rule leaves a sentence out
    (`selection.left_out`, None: kept), whether it keeps it (`selection.kept`, what training reads) and its side
    (`selection.side`, what the readouts read)."""
    return {name: {rule: {outcome: {str(faulty): function(rule, outcome, faulty) for faulty in (False, True)}
                          for outcome in OUTCOMES}
                   for rule in RULES}
            for name, function in (("left_out", left_out), ("kept", kept), ("side", side))}


def campaign_plan() -> list[list[Any]]:
    """`prior_campaign.plan` of a fixed formal record, both choices made (configuration B, variant `constants`): each
    step's name, runner and arguments (the campaign's directory as ``<campaign>``)."""
    import tempfile

    from ts_transformer.experiments.prior_campaign import SEEDS, plan

    record = {"instructions": "artefact", "executor": "executor", "row_interval_s": 4.0, "airports": ["KAAA", "KBBB"],
              "seeds": list(SEEDS), "smoke": None}
    with tempfile.TemporaryDirectory() as directory:
        campaign = Path(directory)
        for step, chosen in (("configuration", "B"), ("variant", "constants")):
            (campaign / f"choice_{step}.json").write_text(json.dumps({"chosen": chosen}), encoding="utf-8")
        return [[step.name, step.runner, [str(a).replace(directory, "<campaign>") for a in step.argv]]
                for step in plan(campaign, record, "cuda")]


def free_generation_draw() -> list[int]:
    """`prior_free_generation.draw` on a fixed set: 30 flights of three airports (KBBB's even places without a
    sentence), 4 of each, seed 1337."""
    from ts_transformer.experiments.prior_free_generation import draw

    flights = [{"airport": ("KAAA", "KBBB", "KCCC")[i % 3]} for i in range(30)]
    sentences = {i: None for i in range(30) if not (i % 3 == 1 and i % 2 == 0)}
    return draw(sentences, flights, ["KAAA", "KBBB", "KCCC"], 4, SEED)


def select_rules() -> dict[str, Any]:
    """`prior_select`'s rules (§5, D40) on fixed tables of scores, each at an edge: configurations tied in parameters
    within twice the seed scale (the lower score), a score at exactly twice the seed scale (within), a seed scale of 0
    (the best alone), and the variant at exactly twice the seed scale (kept `full`) and with a seed scale of 0."""
    from ts_transformer.experiments.prior_select import arm_name, choose_configuration, choose_variant

    seeds = (1, 2)

    def table(scores: dict[str, float], parameters: dict[str, int], second: float) -> dict[str, dict[str, Any]]:
        """Each configuration's arm (variant `full`, the first seed) and configuration A's at the second seed."""
        out = {arm_name(c, "full", seeds[0]): {"score": v, "parameters": parameters[c]} for c, v in scores.items()}
        out[arm_name("A", "full", seeds[1])] = {"score": second, "parameters": parameters["A"]}
        return out

    parameters = {"A": 100, "B": 200, "C": 300, "D": 100}
    # scale 0.25: all four within 1.5 (C at exactly it); A and D tie in parameters, D the lower score
    tie = table({"A": 1.25, "B": 1.0, "C": 1.5, "D": 1.125}, parameters, 1.5)
    edge = table({"A": 1.5, "B": 1.0, "C": 2.0, "D": 1.75}, parameters, 1.75)       # A at exactly best + 2 · 0.25
    zero = table({"A": 1.25, "B": 1.0, "C": 1.5, "D": 1.125}, parameters, 1.25)     # scale 0: B alone
    variant_edge = {**table({"A": 1.5}, parameters, 1.625),                         # scale 0.125: 0.25 is not more
                    arm_name("A", "constants", seeds[0]): {"score": 1.25, "parameters": 100}}
    variant_zero = {**table({"A": 1.5}, parameters, 1.5),
                    arm_name("A", "constants", seeds[0]): {"score": 1.375, "parameters": 100}}
    return {"tie": choose_configuration(tie, seeds), "edge": choose_configuration(edge, seeds),
            "zero": choose_configuration(zero, seeds), "variant_edge": choose_variant(variant_edge, "A", seeds),
            "variant_zero": choose_variant(variant_zero, "A", seeds)}


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
        at = [Position(np.full(SENTENCES, -15_000.0 + 400.0 * r) + 3_000.0 * np.arange(SENTENCES),
                       np.zeros(SENTENCES), np.full(SENTENCES, 900.0 - 10.0 * r)) for r in range(FIRST + SAID)]
        numbers = np.random.default_rng(SEED).random((SAID, SENTENCES, len(COLUMNS)))
        spoken = {}
        for masks in MASK_MODES:                     # B14: the procedure masks' two modes say the same, bit for bit
            speaker = Speaker(model, words, [finals] * SENTENCES, capacity=8, masks=masks)
            speaker.observe(rows.between(0, FIRST), at[:FIRST])
            said = [speaker.speak(rows.between(r, r + 1), at[r], numbers[r - FIRST]).tolist()
                    for r in range(FIRST, FIRST + SAID)]
            spoken[masks] = (said, hexed(np.stack(speaker.drawn_probability)),
                             [{c: v.tolist() for c, v in row.items()} for row in speaker.procedure_blocked],
                             [m.tolist() for m in speaker.permitted().masks])
        require_same_modes("the speaker", spoken)
        said, probabilities = spoken[PER_AIRCRAFT][:2]
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
        sentence = fixed_sentence([row[0] for row in said], finals[0].geometry, words, days)
        loops = {masks: speaking(model, words, finals, days, masks) for masks in MASK_MODES}
        require_same_modes("the speaking loop", loops)
        loop = loops[PER_AIRCRAFT]
    finally:
        torch.set_num_threads(threads)
    from ts_transformer.experiments.prior_campaign import settings

    return {"settings": settings(), "selection": selection_rules(), "select_rules": select_rules(),
            "campaign_plan": campaign_plan(), "free_generation_draw": free_generation_draw(),
            "sentence_rows": sentence,
            "train_config": asdict(TrainConfig()), "model_config": model.config.to_dict(),
            "first_step_runway": runway, "constants_loss": hexed(nll / asked),
            "heard": heard_inputs,
            "inputs": inputs, "speaking_loop": loop,
            "train_loss": hexed([row["train_loss_per_step"] for row in result.history]),
            "select_loss": hexed([row["select_loss_per_step"] for row in result.history]),
            "words": said, "probabilities": probabilities}


def require_same_modes(what: str, answers: dict[str, Any]) -> None:
    """B14: what ``what`` said and flew with the procedure masks in each mode (``answers`` by mode) is the same — refused
    by name otherwise; the answer is the per-aircraft mode's (the reference)."""
    if answers[BATCH] != answers[PER_AIRCRAFT]:
        raise ValueError(f"{what} says otherwise with the procedure masks' batch mode than with the per-aircraft mode "
                         f"(B14)")


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
