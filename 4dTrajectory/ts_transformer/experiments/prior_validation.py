"""B5: the base's one validation readout, but its free generation (prior design §12 B5; D72, D75, D85): the
teacher-forced loss of the val sentences that the prior's own selection keeps, for each airport and in all, and the
share of the labelled words of every val sentence that the procedure masks block, for each airport and column, the
sentences inside and outside the selection apart. The free generation of the readout (and its probability of
"go-around" on the final, D72) is `prior_free_generation --split val`.

The validation days are read once for each stage (D85): the readout is the base's alone (a prior trained on every
airport, no held-out airport), from a clean tree, and marks the prior's run as read (`claim_validation_read`) before it
reads; a second one is refused by name. A smoke (``--smoke``, a tree with changes) never reads the validation days:
it reads the select days (``--split select``), so the chain is checked before the one readout.

THE MASKS ON THE LABELLED WORDS. Each val closed-loop sentence's rows (D82) walked as a speaker walks them
(`prior.speaker.Speaker`): the procedure masks take on every Δ row's state (observed before the first predicted step,
flown from it), with G of the words in force before the row (`ProcedureMasks.track`); at each word row, each altitude and
angle word said is checked against the masks under the runway and G after the row's runway word
(`prior.speaker.runway_after`), and the row is then heard (`prior.inputs.Heard`, the grammar's walk) and its state kept
when its word ended G (`ProcedureMasks.after_row`, D64). "Unchanged" is never counted (the masks never block it, D64).

    python run_ts.py prior_validation --prior <the base's prior_train run> --instructions <artefact> \\
        --executor <its executor spec> --out <a new directory>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from flight_scenarios.procedure_final import DEFAULT_PROCEDURE_ROOT
from ts_transformer.autopilot.closed_loop import require_conforming_closed_loop
from ts_transformer.instructions.artefact import STATE_COLUMNS
from ts_transformer.instructions.artefact import SentenceRows as ClosedLoopRows
from ts_transformer.instructions.artefact import closed_loop_sentences, load_spec, signals_flights
from ts_transformer.instructions.faults import faulty_flights
from ts_transformer.instructions.grammar import column_words
from ts_transformer.instructions.labeller.interval import interval_rows, on_interval_rows
from ts_transformer.instructions.words import COLUMNS, RUNWAY, UNCHANGED, Words
from ts_transformer.io_utils import file_sha256, utc_now, write_json_atomic
from ts_transformer.prior.checkpoint import claim_validation_read, open_prior, readable_identity
from ts_transformer.prior.inputs import Heard
from ts_transformer.prior.procedure import PROCEDURE_MASKS, Final, ProcedureMasks, airport_finals
from ts_transformer.prior.selection import SIDES, side
from ts_transformer.prior.source import ArtefactSource, require_selection_of
from ts_transformer.prior.speaker import runway_after
from ts_transformer.prior.train import evaluate
from ts_transformer.repo_layout import REPO_ROOT, git_state

#: The format of the readout's files (``config.json``, ``readout.json``).
#: v3 (B11, D111): ``readout.json``'s masks counted by side (`selection.SIDES`: the flights left out for a faulty track
#: apart); the identity without the val counts (D85). v2 (B10): the masks of D64 keep the row whose runway word ends G.
VALIDATION_SCHEMA = "ts-prior-validation-v3"
#: The columns of a state row the masks read (`STATE_COLUMNS`).
POSITION = [STATE_COLUMNS.index(name) for name in ("e_m", "n_m", "height_m")]


def blocked_by_masks(rows: ClosedLoopRows, finals: Sequence[Final], words: Words, interval_s: float
                     ) -> dict[int, tuple[int, int]]:
    """One closed-loop sentence's labelled words in each column the procedure masks rule: ``{column: (said, blocked)}``
    (module docstring); ``finals`` its airport's, one for each candidate."""
    if not np.array_equal(rows.on_interval, on_interval_rows(len(rows.states), interval_rows(interval_s,
                                                                                               words.spec.step_s))):
        raise ValueError("the sentence's Δ rows are not marked as its row interval's")
    geometry = finals[0].geometry
    states = rows.states[rows.on_interval][:, POSITION]
    masks = ProcedureMasks([finals], words)
    heard = Heard(geometry, words)
    candidates = len(geometry.candidates)
    classes = {c: list(column_words(c, words, candidates)) for c in ProcedureMasks.columns}
    out = {c: [0, 0] for c in ProcedureMasks.columns}
    for t, state in enumerate(states):
        e, n, height = np.array([state[0]]), np.array([state[1]]), np.array([state[2] - geometry.elevation_m])
        in_force = heard.state
        masks.track(e, n, height, np.array([in_force is not None and in_force.go_around]))
        if t < rows.start:
            continue
        row = rows.grid[t - rows.start]
        runway, go_around = runway_after(np.array([int(row[RUNWAY])]), [in_force], np.array([candidates]))
        for column in ProcedureMasks.columns:
            word = int(row[column])
            if word == UNCHANGED:
                continue
            permitted = masks.permitted(column, runway, go_around, e, n, height)[0]
            out[column][0] += 1
            out[column][1] += int(not permitted[classes[column].index(word)])
        heard.hear(row, float(height[0]), t * interval_s)
        masks.after_row(e, n, height, go_around)
    return {column: (said, blocked) for column, (said, blocked) in out.items()}


def masks_readout(sentences: Mapping[int, Any], flights: Sequence[Mapping[str, Any]],
                  finals: Mapping[str, Sequence[Final]], words: Words, interval_s: float, selection: str,
                  faulty: Mapping[int, Any]) -> dict[str, Any]:
    """The share of the labelled words the masks block, ``{side: {airport: {column: {said, blocked, share}}}}``, the
    sentences inside the prior's selection and outside it for each reason apart (`selection.SIDES`: the stored outcome,
    D75; a faulty observed track, ``faulty`` the marked flights by their place in the signals, D111)."""
    counts: dict[str, dict[str, dict[str, list[int]]]] = {name: {} for name in SIDES}
    for index, sentence in sentences.items():
        airport = flights[index]["airport"]
        where = side(selection, sentence.withheld.outcome, index in faulty)
        for column, (said, blocked) in blocked_by_masks(sentence.rows, finals[airport], words, interval_s).items():
            cell = counts[where].setdefault(airport, {}).setdefault(COLUMNS[column], [0, 0])
            cell[0] += said
            cell[1] += blocked
    return {side: {airport: {column: {"said": said, "blocked": blocked, "share": blocked / said if said else None}
                             for column, (said, blocked) in sorted(columns.items())}
                   for airport, columns in sorted(airports.items())}
            for side, airports in counts.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--prior", type=Path, required=True, help="the base's prior_train run")
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True, help="the directory of the artefact's executor spec")
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--procedure-root", type=Path, default=DEFAULT_PROCEDURE_ROOT)
    parser.add_argument("--split", default="val", choices=("val", "select"),
                        help="val: the one readout (D85); select: a smoke's, which never reads val")
    parser.add_argument("--smoke", action="store_true", help="SMOKE: allowed from a tree with changes; reads select")
    args = parser.parse_args(argv)
    if args.smoke != (args.split == "select"):
        parser.error("the formal readout reads the val days, a smoke the select days (D85)")
    prior_dir, instructions, executor_dir, out = (path if path.is_absolute() else REPO_ROOT / path
                                                  for path in (args.prior, args.instructions, args.executor, args.out))
    if out.exists():
        parser.error(f"{out} exists; the validation days are read once (D85)")
    git = git_state()
    if git["dirty"] and not args.smoke:
        parser.error("a readout that is not a smoke needs a clean tree")
    _, opened, _ = require_conforming_closed_loop(instructions, executor_dir)   # D69: the checks run here (D73)
    device = torch.device(args.device)
    prior = open_prior(prior_dir, instructions, procedure_root=args.procedure_root)   # §7 item 1 (D106)
    if prior.config["run"]["held_out"] is not None:
        raise SystemExit(f"{prior_dir} is a fold (held out {prior.config['run']['held_out']}): the validation readout is "
                         f"the base's (§12 B5)")
    geometries, landings, interval_s = prior.geometries, prior.landings, prior.interval_s
    selection, checkpoint = prior.selection, prior.checkpoint
    if checkpoint.run["sample"] is not None and not args.smoke:
        raise SystemExit(f"{prior_dir} is a smoke prior (a sample of the sentences): no formal readout")
    model = checkpoint.model.to(device)
    variant = model.config.variant
    tokens = int(checkpoint.train_config["tokens_per_batch"])
    if args.split == "val":
        claim_validation_read(prior_dir, "prior_validation", out)
        require_selection_of(instructions, interval_s, checkpoint.identity, args.split)
    source = ArtefactSource(instructions, interval_s, variant, landings, selection)
    airports = sorted(geometries)
    loss = {airport: evaluate(model, source.sentences(args.split, airport), tokens, device) for airport in airports}
    pooled = evaluate(model, [s for airport in airports for s in source.sentences(args.split, airport)], tokens, device)
    spec = load_spec(instructions)
    finals = {code: airport_finals(geometry, root=args.procedure_root) for code, geometry in geometries.items()}
    blocked = masks_readout(closed_loop_sentences(instructions, args.split, interval_s, spec),
                            signals_flights(instructions, args.split), finals, source.words, interval_s, selection,
                            faulty_flights(instructions, args.split))
    out.mkdir(parents=True)
    write_json_atomic(out / "config.json", {
        "schema": VALIDATION_SCHEMA, "written_utc": utc_now(), "prior": str(prior_dir),
        "checkpoint_sha256": file_sha256(prior_dir / "checkpoint.pt"), "identity": readable_identity(checkpoint.identity),
        "instructions": str(instructions), "executor": str(executor_dir), "checks": opened["checks"],
        "row_interval_s": interval_s, "selection": selection, "variant": variant, "procedure_masks": PROCEDURE_MASKS,
        "tokens_per_batch": tokens, "git": git, "smoke": args.smoke, "device": str(device)})
    write_json_atomic(out / "readout.json", {
        "schema": VALIDATION_SCHEMA, "split": args.split, "selection": selection,
        "teacher_forced": {"pooled": pooled, "airports": loss}, "masks_on_labelled_words": blocked})
    print(json.dumps({"out": str(out), "loss_per_step": pooled["loss_per_step"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
