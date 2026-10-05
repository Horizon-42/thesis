"""Free generation (prior design §12 B4): the prior speaks, the executor flies, the judge decides.

`speak_and_fly` flies a batch of flights started at their first predicted step through the start of a closed loop
(vocabulary §6 item 5, D67: `autopilot.start`) in the step of a speaker's closed loop (`prior_speaking_loop`, §7 item 7,
D106: the observed rows, then at each Δ row the inputs from the flown states, the speaker under its masks and the bound
of D68, the executor's step), with each flight's own random numbers (`flight_numbers`, D96) and its airport's roster
landings (D105), to the end of every flight; its outcome is the judge's (`Loop.outcome`, item 6). The runner imports
nothing else of `autopilot/` (`tests/test_architecture.py`, D69).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ts_transformer.autopilot.closed_loop import require_conforming_closed_loop
from ts_transformer.autopilot.start import NO_MOVE, Loop, start_moved
from ts_transformer.experiments.prior_speaking_loop import Generated, SpeakingLoop, flight_numbers
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.artefact import ClosedLoopSentence, closed_loop_sentences, load_spec, signals_flights
from ts_transformer.instructions.faults import faulty_flights
from ts_transformer.instructions.words import COLUMNS, UNCHANGED, Words
from ts_transformer.io_utils import file_sha256, utc_now, write_json_atomic
from ts_transformer.prior.checkpoint import (CLAIM_SPENT_BY, claim_validation_read, open_prior, readable_identity,
                                             spend_validation_claim)
from ts_transformer.prior.source import require_selection_of
from ts_transformer.prior.training_files import CLAIM_READER
from ts_transformer.prior.landings import LandingIndex
from ts_transformer.prior.model import Prior
from ts_transformer.prior.procedure import PROCEDURE_MASKS, Final, ProcedureMasks, airport_finals
from ts_transformer.prior.selection import SIDES, side
from ts_transformer.prior.speaker import MOST_GO_AROUNDS
from ts_transformer.repo_layout import REPO_ROOT, git_state

#: The temperature of free generation: the model's own distribution (the masks applied).
TEMPERATURE = 1.0
#: The format of a readout's files (``config.json``, ``sentences.npz``, ``readout.json``; `read_sentences`). v4 (B11,
#: D111): ``readout.json`` by side (`selection.SIDES`: the flights left out for a faulty track apart), the identity without
#: the val counts (D85). v3 (B10): "on the final" under
#: the runway in force before the row (D72), the masks of `procedure-masks-v5` (D64). v2 (B9, D96): each flight's
#: words drawn with its own random numbers (`flight_numbers`), whatever the batch. v1 (B6): the words the
#: procedure masks blocked at each row (``blocked_<column>``).
FREE_GENERATION_SCHEMA = "ts-prior-free-generation-v4"
#: The arrays of ``sentences.npz``.
SENTENCES_FIELDS = {"schema", "index", "sample", "offsets", "words", "go_around_probability", "go_around_permitted",
                    "on_final", "state_offsets", "states", *(f"blocked_{COLUMNS[c]}" for c in ProcedureMasks.columns)}


@dataclass(frozen=True)
class Stored:
    """One sentence of a readout as `read_sentences` gives it back: its flight's place in the split's signals, its
    sample, its words, its states on the 2 s rows from row 0 and its rows' probability and permission of "go-around",
    place on the final and blocked words (`Generated`'s arrays); its row of ``sentences.jsonl``."""

    index: int
    sample: int
    words: np.ndarray
    states: np.ndarray
    go_around_probability: np.ndarray
    go_around_permitted: np.ndarray
    on_final: np.ndarray
    blocked: dict[int, np.ndarray]
    row: dict[str, Any]


def read_sentences(out: Path) -> tuple[dict[str, Any], list[Stored]]:
    """A readout's config and its sentences in the order written, refused unless they are this format's."""
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    with np.load(out / "sentences.npz") as arrays:
        data = {name: arrays[name] for name in arrays.files}
    if config["schema"] != FREE_GENERATION_SCHEMA or set(data) != SENTENCES_FIELDS \
            or str(data["schema"]) != FREE_GENERATION_SCHEMA:
        raise ValueError(f"{out} is not a {FREE_GENERATION_SCHEMA} readout")
    rows = [json.loads(line) for line in (out / "sentences.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(rows) != len(data["index"]):
        raise ValueError(f"{out}: sentences.jsonl holds {len(rows)} sentences, sentences.npz {len(data['index'])}")
    stored = []
    for k, row in enumerate(rows):
        said = slice(int(data["offsets"][k]), int(data["offsets"][k + 1]))
        states = slice(int(data["state_offsets"][k]), int(data["state_offsets"][k + 1]))
        if (int(data["index"][k]), int(data["sample"][k])) != (row["index"], row["sample"]):
            raise ValueError(f"{out}: sentence {k} of sentences.npz is not the one of sentences.jsonl")
        stored.append(Stored(int(data["index"][k]), int(data["sample"][k]), data["words"][said], data["states"][states],
                             data["go_around_probability"][said], data["go_around_permitted"][said],
                             data["on_final"][said],
                             {c: data[f"blocked_{COLUMNS[c]}"][said] for c in ProcedureMasks.columns}, row))
    return config, stored


def speak_and_fly(model: Prior, loop: Loop, order: Sequence[int], sentences: Mapping[int, ClosedLoopSentence],
                  observed: Mapping[int, np.ndarray], flights: Mapping[int, Mapping[str, Any]], geometries: Mapping[str, AirportGeometry],
                  landings: Mapping[str, LandingIndex], finals: Mapping[str, Sequence[Final]], words: Words, *,
                  interval_s: float, variant: str, numbers: Sequence[np.random.Generator],
                  device: torch.device, temperature: float = 1.0) -> list[Generated]:
    """The closed loop of the flights ``order`` of ``loop`` (`autopilot.start.start_moved`; module docstring):
    ``sentences``, ``observed`` (the start's observed rows) and ``flights`` (their records in the split's signals) by
    their place in the signals; ``landings`` each airport's
    roster landings (each flight is given its airport's, D105); ``numbers`` each flight's source of random numbers, in
    ``order`` (`flight_numbers`: five a row said, D96)."""
    if len(numbers) != len(order):
        raise ValueError(f"{len(numbers)} sources of random numbers for {len(order)} flights")
    if loop.most_go_arounds != MOST_GO_AROUNDS:
        raise ValueError(f"the loop was started for {loop.most_go_arounds} go-arounds a flight, the bound is "
                         f"{MOST_GO_AROUNDS} (D68)")
    speaking = SpeakingLoop(model, loop, order, sentences, observed, flights, geometries,
                            [landings[flights[i]["airport"]] for i in order], finals, words,
                            interval_s=interval_s, variant=variant, device=device, temperature=temperature)
    while speaking.observing:
        speaking.observe()
    while speaking.alive.any():
        speaking.step(np.stack([n.random(len(COLUMNS)) for n in numbers]))
    return speaking.generated()


def readout(generated: Sequence[Generated], airports: Mapping[int, str], strata: Mapping[int, str],
            labelled: Mapping[int, np.ndarray]) -> dict[str, Any]:
    """The readout of free generation (§12 B4), for each airport and each stratum of the flight (straight-in or vectored,
    as the sentence file stores it, D70): the sentences and their outcomes; the words said in each column for each
    sentence, beside the labelled closed-loop sentence's (``labelled``: by the flight's place, its words from the first
    predicted step); the go-arounds said and the sentences that reached the bound of D68; the probability of
    "go-around" on the rows on the final (inside the region of the runway in force before the row, under which the
    speaker drew the word) where the masks permitted it (not while G, not after the bound; D72), and those rows. No
    criterion is applied (D7)."""
    groups: dict[tuple[str, str], list[Generated]] = {}
    for g in generated:
        groups.setdefault((airports[g.index], strata[g.index]), []).append(g)
    out: dict[str, Any] = {}
    for (airport, stratum), members in sorted(groups.items()):
        said = np.array([(g.words != UNCHANGED).sum(axis=0) for g in members])
        reference = np.array([(labelled[g.index] != UNCHANGED).sum(axis=0) for g in members])
        final = np.concatenate([g.go_around_probability[g.on_final & g.go_around_permitted] for g in members])
        outcomes = Counter(g.outcome for g in members)
        out.setdefault(airport, {})[stratum] = {
            "sentences": len(members), "outcomes": dict(sorted(outcomes.items())),
            "timed_out": int(sum(g.timed_out for g in members)),
            "words_per_sentence": dict(zip(COLUMNS, said.mean(axis=0).tolist())),
            "labelled_words_per_sentence": dict(zip(COLUMNS, reference.mean(axis=0).tolist())),
            "go_arounds": int(sum(g.go_arounds for g in members)),
            "at_the_bound": int(sum(g.go_arounds >= MOST_GO_AROUNDS for g in members)),
            "go_around_probability_on_final": {"rows": int(len(final)),
                                               "mean": float(final.mean()) if len(final) else None}}
    return out


def draw(sentences: Mapping[int, ClosedLoopSentence], flights: Sequence[Mapping[str, Any]], airports: Sequence[str],
         per_airport: int, seed: int) -> list[int]:
    """At most ``per_airport`` flights of each of ``airports`` with a closed-loop sentence, drawn at random with ``seed``
    (D55: never the first in order), by their place in the split's signals, in order."""
    rng = np.random.default_rng(seed)
    out: list[int] = []
    for airport in airports:
        pool = np.array(sorted(i for i in sentences if flights[i]["airport"] == airport))
        out += sorted(rng.choice(pool, size=min(per_airport, len(pool)), replace=False).tolist())
    return out


def main(argv: list[str] | None = None) -> int:
    """`python run_ts.py prior_free_generation --prior <a prior_train run> --instructions <artefact> --executor <spec>
    --split select --airports KSJC --out <a new directory>`: free generation of a prior at the airports given (a fold's
    held-out airport, or every airport of the base) on a split's flights: ``--per-airport`` flights drawn at random
    (seed), each spoken ``--samples`` times. Writes into ``--out`` (new, never over an existing one; a clean tree unless
    ``--smoke``): ``config.json``, ``sentences.jsonl`` (one row a sentence: its flight, sample, outcome, crossing, the
    go-arounds), ``sentences.npz`` (the words, the states, the probability and permission of "go-around" and the rows on
    the final, each sentence's by its offsets) and ``readout.json`` (`readout`, all samples together, the flights the
    prior's selection keeps and those it leaves out for each reason apart: `selection.SIDES`, D75, D111)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--prior", type=Path, required=True, help="a prior_train run's directory")
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True, help="the directory of the artefact's executor spec")
    parser.add_argument("--split", required=True, choices=("train", "select", "val"),
                        help="a fold reads select (its held-out airport), the base its one val readout")
    parser.add_argument("--airports", nargs="+", default=None, help="the airports spoken at (default: every one)")
    parser.add_argument("--per-airport", type=int, default=200)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--chunk", type=int, default=400, help="flights flown in one loop")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    parser.add_argument("--smoke", action="store_true", help="SMOKE: allowed from a tree with changes; recorded")
    args = parser.parse_args(argv)
    absolute = [p if p.is_absolute() else REPO_ROOT / p for p in (args.prior, args.instructions, args.executor, args.out)]
    prior_dir, instructions, executor_dir, out = absolute
    if out.exists() and not (out / CLAIM_SPENT_BY).exists():
        parser.error(f"{out} exists without its {CLAIM_SPENT_BY}: a run that stopped; move it aside "
                     f"({out.name}.aborted-<UTC>, outline E8) and run again to the same output")
    if out.exists():
        parser.error(f"{out} exists; a readout is never overwritten")
    small = [name for name in ("per_airport", "samples", "chunk") if getattr(args, name) < 1]
    if small:                                   # options are checked before the val read is claimed (D119)
        parser.error(f"--{small[0].replace('_', '-')} is at least 1")
    git = git_state()
    if git["dirty"] and not args.smoke:
        parser.error("a readout that is not a smoke needs a clean tree")
    _, opened, _ = require_conforming_closed_loop(instructions, executor_dir)   # D69: the checks run here (D73)
    device = torch.device(args.device)

    prior = open_prior(prior_dir, instructions)        # its record, checkpoint and procedure masks (§7 item 1, D106)
    geometries, landings, interval_s = prior.geometries, prior.landings, prior.interval_s
    selection, checkpoint = prior.selection, prior.checkpoint           # the prior's own rule (D75)
    model = checkpoint.model.to(device)
    finals = {code: airport_finals(geometry) for code, geometry in geometries.items()}
    spec = load_spec(instructions)
    airports = args.airports or sorted(geometries)
    unknown = sorted(set(airports) - set(geometries))
    if unknown:                                 # an option is checked before the val read is claimed (D119)
        parser.error(f"airports {unknown} are not the artefact's {sorted(geometries)}")
    if args.split == "val":                     # the base's one validation readout (D85): never a smoke, read once
        if args.smoke or checkpoint.run["held_out"] is not None or checkpoint.run["sample"] is not None:
            parser.error("the validation days are read only by the base's formal readout (D85)")
        claim_validation_read(prior_dir, CLAIM_READER, out)
        require_selection_of(instructions, interval_s, checkpoint.identity, args.split)
    sentences = closed_loop_sentences(instructions, args.split, interval_s, spec)
    flights = signals_flights(instructions, args.split)
    # withheld from the model (D82), read for the readout: each flight's stratum (D70) and stored outcome (D74)
    strata = {i: sentence.withheld.stratum for i, sentence in sentences.items()}
    empty = sorted(set(airports) - {flights[i]["airport"] for i in sentences})
    if empty:
        parser.error(f"airports {empty} have no closed-loop sentence in {args.split}")
    drawn = draw(sentences, flights, airports, args.per_airport, args.seed)

    generated: list[tuple[int, Generated]] = []
    for sample in range(args.samples):
        for begin in range(0, len(drawn), args.chunk):
            chunk = drawn[begin: begin + args.chunk]
            # the start's own observed rows (`NO_MOVE`: the stored sentence's, bit for bit) are what the speaker reads
            loop, order, observed = start_moved(instructions, args.split, interval_s, {i: sentences[i] for i in chunk},
                                                executor_dir, {i: NO_MOVE for i in chunk},
                                                most_go_arounds=MOST_GO_AROUNDS, device=device)
            generated += [(sample, g) for g in speak_and_fly(
                model, loop, order, sentences, observed, dict(enumerate(flights)), geometries, landings, finals, loop.words,
                interval_s=interval_s, variant=model.config.variant,
                numbers=[flight_numbers(args.seed, sample, i) for i in order], device=device,
                temperature=TEMPERATURE)]
            print(f"sample {sample}: {begin + len(chunk)} of {len(drawn)} flights", flush=True)

    out.mkdir(parents=True)
    write_json_atomic(out / "config.json", {
        "schema": FREE_GENERATION_SCHEMA,
        "written_utc": utc_now(), "prior": str(prior_dir), "prior_run": checkpoint.run,
        "checkpoint_sha256": file_sha256(prior_dir / "checkpoint.pt"), "identity": readable_identity(checkpoint.identity),
        "instructions": str(instructions), "executor": str(executor_dir), "checks": opened["checks"],
        "split": args.split, "row_interval_s": interval_s, "selection": selection, "airports": airports,
        "per_airport": args.per_airport,
        "samples": args.samples, "seed": args.seed, "chunk": args.chunk, "temperature": TEMPERATURE, "drawn": len(drawn),
        "numbers": "each flight's own: numpy default_rng([seed, sample, its place in the split's signals]), five "
                   "uniform numbers a row said (D96)",
        "procedure_masks": PROCEDURE_MASKS, "most_go_arounds": MOST_GO_AROUNDS, "git": git, "smoke": args.smoke,
        "device": str(device)})
    with open(out / "sentences.jsonl", "w", encoding="utf-8") as rows:
        for sample, g in generated:
            rows.write(json.dumps({"index": g.index, "dataset_id": flights[g.index]["dataset_id"],
                                   "airport": flights[g.index]["airport"], "stratum": strata[g.index], "sample": sample,
                                   "outcome": g.outcome, "crossing": g.crossing, "timed_out": g.timed_out,
                                   "go_arounds": g.go_arounds, "rows": len(g.words)}) + "\n")

    def offsets(items):
        return np.concatenate(([0], np.cumsum([len(item) for item in items]))).astype(np.int64)

    sentences_only = [g for _, g in generated]
    np.savez_compressed(
        out / "sentences.npz", schema=np.array(FREE_GENERATION_SCHEMA), index=np.array([g.index for g in sentences_only]),
        sample=np.array([s for s, _ in generated]), offsets=offsets([g.words for g in sentences_only]),
        words=np.concatenate([g.words for g in sentences_only]),
        go_around_probability=np.concatenate([g.go_around_probability for g in sentences_only]),
        go_around_permitted=np.concatenate([g.go_around_permitted for g in sentences_only]),
        on_final=np.concatenate([g.on_final for g in sentences_only]),
        state_offsets=offsets([g.states for g in sentences_only]),
        states=np.concatenate([g.states for g in sentences_only]),
        **{f"blocked_{COLUMNS[c]}": np.concatenate([g.blocked[c] for g in sentences_only]) for c in ProcedureMasks.columns})
    # free generation starts from every flight; the readout gives the flights outside the prior's selection apart, by
    # why the selection leaves them out: a faulty observed track (D111; such a flight's start can fail whatever is
    # said) or the stored outcome (D75)
    airports_of = {i: flights[i]["airport"] for i in drawn}
    grids = {i: sentences[i].rows.grid for i in drawn}
    faulty = faulty_flights(instructions, args.split)
    sides = {g.index: side(selection, sentences[g.index].withheld.outcome, g.index in faulty) for g in sentences_only}
    write_json_atomic(out / "readout.json", {"selection": selection, **{
        name: readout([g for g in sentences_only if sides[g.index] == name], airports_of, strata, grids)
        for name in SIDES}})
    if args.split == "val":
        spend_validation_claim(prior_dir, CLAIM_READER, out)               # its readout written: the read is spent (D119)
    print(json.dumps({"out": str(out), "sentences": len(generated)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
