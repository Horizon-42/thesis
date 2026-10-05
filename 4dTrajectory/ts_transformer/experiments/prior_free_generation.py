"""Free generation (prior design §12 B4): the prior speaks, the executor flies, the judge decides.

`speak_and_fly` is the closed loop of a batch of flights started at their first predicted step through the start of a
closed loop (vocabulary §6 item 5, D67: `autopilot.start`): the rows before the first predicted step are the flight's
observed rows (the closed-loop sentence's states there); from it on, at each Δ row, the inputs come from the states the
executor flew (D32, §2) through the prior's one input function (`prior.inputs.state_inputs`, `prior.inputs.Heard`,
`prior.batch.row_tensors`), the speaker says the row under its masks and a caller's — the bound of D68, "go-around"
forbidden after a flight's second — and the loop flies it for Δ seconds. A flight ends when the executor is done with
it (its crossing, its time limit with 900 s for each go-around, or the dynamics); its outcome is the judge's
(`Loop.outcome`, item 6). The runner imports nothing else of `autopilot/` (`tests/test_architecture.py`, D69).
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
from ts_transformer.autopilot.start import Loop, start
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.artefact import (
    ClosedLoopSentence, closed_loop_sentences, load_candidates, load_day_split, load_spec, signals_flights,
)
from ts_transformer.io_utils import file_sha256, utc_now, write_json_atomic
from ts_transformer.prior.checkpoint import CHECKPOINT_SCHEMA, claim_validation_read, load_checkpoint
from ts_transformer.prior.procedure import PROCEDURE_MASKS, ProcedureMasks, airport_finals, procedure_digests
from ts_transformer.prior.selection import kept
from ts_transformer.prior.source import airport_landings, artefact_identity
from ts_transformer.repo_layout import REPO_ROOT, git_state
from ts_transformer.instructions.labeller.interval import interval_rows
from ts_transformer.instructions.words import COLUMNS, RUNWAY, RUNWAY_GO_AROUND, UNCHANGED, Words
from ts_transformer.prior.batch import row_tensors
from ts_transformer.prior.inputs import own_flight_key, state_inputs
from ts_transformer.prior.landings import LandingIndex, utc_s
from ts_transformer.prior.model import Prior
from ts_transformer.prior.procedure import Final
from ts_transformer.prior.speaker import MOST_GO_AROUNDS, Position, Speaker, go_around_bound


#: The temperature of free generation: the model's own distribution (the masks applied).
TEMPERATURE = 1.0
#: The predicted rows the speaker's cache has room for at first (it grows as the flights need, `model.Past.grown`).
FIRST_ROWS = 128
#: The format of a readout's files (``config.json``, ``sentences.npz``; `read_sentences`). v1 (B6): the words the
#: procedure masks blocked at each row (``blocked_<column>``).
FREE_GENERATION_SCHEMA = "ts-prior-free-generation-v1"
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


@dataclass
class Generated:
    """One flight's free sentence: the words said from its first predicted step to the row it was done in, its states on
    the 2 s rows from its row 0 (observed before the first predicted step, flown from it; D51's layout), and the judge's
    outcome."""

    index: int                         # the flight's place in the split's signals
    words: np.ndarray                  # [M, 5]
    #: [rows, 6] `STATE_COLUMNS` on the 2 s rows from row 0, to the first row at or after the cycle the executor was done in
    states: np.ndarray
    outcome: str
    crossing: dict[str, Any] | None
    timed_out: bool
    go_arounds: int
    go_around_probability: np.ndarray  # [M]: of "go-around" in the distribution each runway word was drawn from
    go_around_permitted: np.ndarray    # [M] bool: the masks permitted "go-around" (G false, D68's bound not met)
    on_final: np.ndarray               # [M] bool: inside the region of the runway in force (the FAF and the LPV cone)
    #: by column the procedure masks rule (`ProcedureMasks.columns`): [M, words] bool, the classes they blocked at each
    #: row (`Speaker.procedure_blocked`)
    blocked: dict[int, np.ndarray]


def speak_and_fly(model: Prior, loop: Loop, order: Sequence[int], sentences: Mapping[int, ClosedLoopSentence],
                  flights: Mapping[int, Mapping[str, Any]], geometries: Mapping[str, AirportGeometry],
                  landings: Mapping[str, LandingIndex], finals: Mapping[str, Sequence[Final]], words: Words, *,
                  interval_s: float, variant: str, generator: torch.Generator, device: torch.device,
                  temperature: float = 1.0) -> list[Generated]:
    """The closed loop of the flights ``order`` of ``loop`` (`autopilot.start.start`; module docstring): ``sentences``
    and ``flights`` (their records in the split's signals) by their place in the signals."""
    if loop.most_go_arounds != MOST_GO_AROUNDS:
        raise ValueError(f"the loop was started for {loop.most_go_arounds} go-arounds a flight, the bound is "
                         f"{MOST_GO_AROUNDS} (D68)")
    step_s = words.spec.step_s
    every, start = interval_rows(interval_s, step_s), sentences[order[0]].rows.start
    count = len(order)
    flight_geometries = [geometries[flights[i]["airport"]] for i in order]
    elevations = np.array([g.elevation_m for g in flight_geometries])
    # the cache's first room: the observed rows and some predicted ones (it grows as the flights need; the loop's time
    # limit is not the speaker's to read, vocabulary D90)
    speaker = Speaker(model, words, [finals[g.code] for g in flight_geometries], capacity=start + FIRST_ROWS,
                      generator=generator, temperature=temperature)
    entry = np.array([utc_s(flights[i]["entry_time_utc"]) for i in order])
    first_rows = np.array([sentences[i].rows.first_row for i in order])
    keys = [own_flight_key(flights[i]) for i in order]

    def row(t: int, at: np.ndarray, before: np.ndarray, known: bool) -> tuple[Any, Position]:
        """The inputs of Δ row ``t`` of every flight at the states ``at`` [B, 6] with ``before`` the 2 s row before."""
        utc = entry + (first_rows + t * every) * step_s
        own, candidates = [], []
        for b, geometry in enumerate(flight_geometries):
            counts = landings[geometry.code].counts_before(utc[b: b + 1], without=keys[b])
            index = [landings[geometry.code].runways.index(c.ident) for c in geometry.candidates]
            o, c = state_inputs(at[b: b + 1, :3], before[b: b + 1, :3], np.array([known]), counts[:, index], geometry,
                                variant, step_s)
            own.append(o[0])
            candidates.append(c[0])
        heard = [h.inputs(t * interval_s) for h in speaker.heard]
        tensors = row_tensors(np.full(count, t * interval_s), np.stack(own), candidates,
                              np.array([h[0] for h in heard]), np.array([h[1] for h in heard]),
                              np.stack([h[2] for h in heard]), np.stack([h[3] for h in heard]),
                              np.stack([h[4] for h in heard]), np.full(count, t == start), np.full(count, t >= start),
                              device)
        return tensors, Position(at[:, 0], at[:, 1], at[:, 2] - elevations)

    # the observed rows before the first predicted step
    observed = np.stack([sentences[i].rows.states[: start * every] for i in order])     # [B, start·every, 6]
    for t in range(start):
        r = t * every
        tensors, at = row(t, observed[:, r], observed[:, max(r - 1, 0)], r > 0)
        speaker.observe(tensors, [at])
    # from the first predicted step on: the flown states
    current, before = loop.rows(), observed[:, start * every - 1]
    said: list[list[np.ndarray]] = [[] for _ in order]
    flown: list[list[np.ndarray]] = [[current[b]] for b in range(count)]
    probability: list[list[float]] = [[] for _ in order]
    permitted: list[list[bool]] = [[] for _ in order]
    on_final: list[list[bool]] = [[] for _ in order]
    blocked: list[list[dict[int, np.ndarray]]] = [[] for _ in order]
    alive = np.ones(count, dtype=bool)
    t = start
    while alive.any():
        tensors, at = row(t, current, before, True)
        caller = {RUNWAY: go_around_bound(speaker.go_arounds, words, int(speaker.n_candidates.max()))}
        words_row = speaker.speak(tensors, at, caller)
        rows, done = loop.step(words_row)
        for b in np.flatnonzero(alive):
            final = finals[flight_geometries[b].code][speaker.in_force[b].runway]
            said[b].append(words_row[b])
            flown[b] += list(rows[b])
            probability[b].append(float(speaker.go_around_probability[-1][b]))
            permitted[b].append(bool(speaker.go_around_permitted[-1][b]))
            on_final[b].append(bool(final.inside(np.array(at.e_m[b]), np.array(at.n_m[b]))))
            blocked[b].append({c: mask[b] for c, mask in speaker.procedure_blocked[-1].items()})
        alive &= ~done
        # a done flight is halted and keeps, as its inputs, the finite state of the row it ended in: the executor flies
        # a done flight on (a non-finite state is one of its ends) and the speaker still says its rows
        loop.halt(~alive)
        before = np.where(alive[:, None], rows[:, -2] if every > 1 else current, before)
        current = np.where(alive[:, None], rows[:, -1], current)
        t += 1
    timed_out = loop.timed_out()
    # each flight's flown 2 s rows to the first at or after the end of the cycle it was done in
    ended = np.ceil((loop.executor.done_cycle.cpu().numpy() + 1) / loop.row_cycles).astype(int)
    out = []
    for b, i in enumerate(order):
        outcome = loop.outcome(b)
        out.append(Generated(index=i, words=np.array(said[b], dtype=np.int64),
                             states=np.concatenate((observed[b], np.array(flown[b][: ended[b] + 1]))),
                             outcome=outcome.outcome,
                             crossing=outcome.crossing, timed_out=bool(timed_out[b]),
                             # the go-arounds of its own words: the speaker says a done flight's rows too (above)
                             go_arounds=int(sum(row[RUNWAY] == RUNWAY_GO_AROUND for row in said[b])),
                             go_around_probability=np.array(probability[b]),
                             go_around_permitted=np.array(permitted[b], dtype=bool),
                             on_final=np.array(on_final[b], dtype=bool),
                             blocked={c: np.array([row[c] for row in blocked[b]], dtype=bool) for c in blocked[b][0]}))
    return out


def readout(generated: Sequence[Generated], airports: Mapping[int, str], strata: Mapping[int, str],
            labelled: Mapping[int, np.ndarray]) -> dict[str, Any]:
    """The readout of free generation (§12 B4), for each airport and each stratum of the flight (straight-in or vectored,
    as the sentence file stores it, D70): the sentences and their outcomes; the words said in each column for each
    sentence, beside the labelled closed-loop sentence's (``labelled``: by the flight's place, its words from the first
    predicted step); the go-arounds said and the sentences that reached the bound of D68; the probability of
    "go-around" on the rows on the final (inside the region of the runway in force) where the masks permitted it (not
    while G, not after the bound: Claude's reading of "on the final", a proposal), and those rows. No criterion is
    applied (D7)."""
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
    prior's selection keeps and those it leaves out apart, D75)."""
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
    if out.exists():
        parser.error(f"{out} exists; a readout is never overwritten")
    git = git_state()
    if git["dirty"] and not args.smoke:
        parser.error("a readout that is not a smoke needs a clean tree")
    _, opened, _ = require_conforming_closed_loop(instructions, executor_dir)   # D69: the checks run here (D73)
    device = torch.device(args.device)

    geometries = load_candidates(instructions)
    landings = airport_landings(geometries, load_day_split(instructions))
    trained = json.loads((prior_dir / "config.json").read_text(encoding="utf-8"))
    if trained["schema"] != CHECKPOINT_SCHEMA:
        raise SystemExit(f"{prior_dir}: a {trained['schema']!r} prior, not {CHECKPOINT_SCHEMA!r}")
    interval_s = float(trained["identity"]["row_interval_s"])
    selection = trained["identity"]["selection"]["rule"]               # the prior's own rule (D75), checked below
    checkpoint = load_checkpoint(prior_dir / "checkpoint.pt",
                                 artefact_identity(instructions, interval_s, landings, selection))
    masks = json.loads((prior_dir / "procedure_masks.json").read_text(encoding="utf-8"))
    if masks != {"set": PROCEDURE_MASKS, "checkpoint_sha256": file_sha256(prior_dir / "checkpoint.pt"),
                 "procedure_data": procedure_digests(geometries)}:
        raise SystemExit(f"{prior_dir}: the prior's procedure masks are not {PROCEDURE_MASKS} on today's procedure data "
                         f"(§8 item 2)")
    model = checkpoint.model.to(device)
    finals = {code: airport_finals(geometry) for code, geometry in geometries.items()}
    spec = load_spec(instructions)
    if args.split == "val":                     # the base's one validation readout (D85): never a smoke, read once
        if args.smoke or checkpoint.run["held_out"] is not None or checkpoint.run["sample"] is not None:
            parser.error("the validation days are read only by the base's formal readout (D85)")
        claim_validation_read(prior_dir, "prior_free_generation", out)
    sentences = closed_loop_sentences(instructions, args.split, interval_s, spec)
    flights = signals_flights(instructions, args.split)
    # withheld from the model (D82), read for the readout: each flight's stratum (D70) and stored outcome (D74)
    strata = {i: sentence.withheld.stratum for i, sentence in sentences.items()}
    airports = args.airports or sorted(geometries)
    unknown = sorted(set(airports) - set(geometries))
    if unknown:
        parser.error(f"airports {unknown} are not the artefact's {sorted(geometries)}")
    empty = sorted(set(airports) - {flights[i]["airport"] for i in sentences})
    if empty:
        parser.error(f"airports {empty} have no closed-loop sentence in {args.split}")
    drawn = draw(sentences, flights, airports, args.per_airport, args.seed)

    generated: list[tuple[int, Generated]] = []
    for sample in range(args.samples):
        for begin in range(0, len(drawn), args.chunk):
            chunk = drawn[begin: begin + args.chunk]
            loop, order = start(instructions, args.split, interval_s, {i: sentences[i] for i in chunk}, executor_dir,
                                most_go_arounds=MOST_GO_AROUNDS, device=device)
            generator = torch.Generator(device=device).manual_seed(args.seed + 1_000_003 * sample + begin)
            generated += [(sample, g) for g in speak_and_fly(
                model, loop, order, sentences, dict(enumerate(flights)), geometries, landings, finals, loop.words,
                interval_s=interval_s, variant=model.config.variant, generator=generator, device=device,
                temperature=TEMPERATURE)]
            print(f"sample {sample}: {begin + len(chunk)} of {len(drawn)} flights", flush=True)

    out.mkdir(parents=True)
    write_json_atomic(out / "config.json", {
        "schema": FREE_GENERATION_SCHEMA,
        "written_utc": utc_now(), "prior": str(prior_dir), "prior_run": checkpoint.run,
        "checkpoint_sha256": file_sha256(prior_dir / "checkpoint.pt"), "identity": checkpoint.identity,
        "instructions": str(instructions), "executor": str(executor_dir), "checks": opened["checks"],
        "split": args.split, "row_interval_s": interval_s, "selection": selection, "airports": airports,
        "per_airport": args.per_airport,
        "samples": args.samples, "seed": args.seed, "chunk": args.chunk, "temperature": TEMPERATURE, "drawn": len(drawn),
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
    # free generation starts from every flight; the readout gives the flights outside the prior's selection apart (D75)
    airports_of = {i: flights[i]["airport"] for i in drawn}
    grids = {i: sentences[i].rows.grid for i in drawn}
    inside = [g for g in sentences_only if kept(selection, sentences[g.index].withheld.outcome)]
    outside = [g for g in sentences_only if not kept(selection, sentences[g.index].withheld.outcome)]
    write_json_atomic(out / "readout.json", {
        "selection": selection, "inside": readout(inside, airports_of, strata, grids),
        "outside": readout(outside, airports_of, strata, grids)})
    print(json.dumps({"out": str(out), "sentences": len(generated)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
