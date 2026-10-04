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

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ts_transformer.autopilot.executor import GO_AROUND_EXTRA_S
from ts_transformer.autopilot.start import Loop
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.artefact import ClosedLoopSentence
from ts_transformer.instructions.labeller.interval import interval_rows
from ts_transformer.instructions.words import COLUMNS, RUNWAY, UNCHANGED, Words
from ts_transformer.prior.batch import row_tensors
from ts_transformer.prior.inputs import own_flight_key, state_inputs
from ts_transformer.prior.landings import LandingIndex, utc_s
from ts_transformer.prior.model import Prior
from ts_transformer.prior.procedure import Final
from ts_transformer.prior.speaker import MOST_GO_AROUNDS, Position, Speaker, go_around_bound


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


def speak_and_fly(model: Prior, loop: Loop, order: Sequence[int], sentences: Mapping[int, ClosedLoopSentence],
                  flights: Mapping[int, Mapping[str, Any]], geometries: Mapping[str, AirportGeometry],
                  landings: Mapping[str, LandingIndex], finals: Mapping[str, Sequence[Final]], words: Words, *,
                  interval_s: float, variant: str, generator: torch.Generator, device: torch.device
                  ) -> list[Generated]:
    """The closed loop of the flights ``order`` of ``loop`` (`autopilot.start.start`; module docstring): ``sentences``
    and ``flights`` (their records in the split's signals) by their place in the signals."""
    if loop.most_go_arounds != MOST_GO_AROUNDS:
        raise ValueError(f"the loop was started for {loop.most_go_arounds} go-arounds a flight, the bound is "
                         f"{MOST_GO_AROUNDS} (D68)")
    step_s = words.spec.step_s
    every, start = interval_rows(interval_s, step_s), sentences[order[0]].start
    count = len(order)
    flight_geometries = [geometries[flights[i]["airport"]] for i in order]
    elevations = np.array([g.elevation_m for g in flight_geometries])
    limit = float(loop.executor.time_limit_s.max()) + GO_AROUND_EXTRA_S * loop.most_go_arounds
    speaker = Speaker(model, words, [finals[g.code] for g in flight_geometries],
                      capacity=start + math.ceil(limit / interval_s) + 2, generator=generator)
    entry = np.array([utc_s(flights[i]["entry_time_utc"]) for i in order])
    first_rows = np.array([sentences[i].first_row for i in order])
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
    observed = np.stack([sentences[i].states[: start * every] for i in order])          # [B, start·every, 6]
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
                             go_arounds=int(speaker.go_arounds[b]), go_around_probability=np.array(probability[b]),
                             go_around_permitted=np.array(permitted[b], dtype=bool),
                             on_final=np.array(on_final[b], dtype=bool)))
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
