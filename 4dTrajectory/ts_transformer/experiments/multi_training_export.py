"""F4: the Training export of stage D (frontend §5.7, §8 F4; multi-aircraft control §10) — for each airport, a set of
windows in which every arrival of the window's span is commanded, which the frontend's Training view reads in stage D:
stage C's set format (`post_training_export`, one format for both stages), listed in the airport's stage D index
(``training/index_multi_v1.json``).

WHICH WINDOWS. The campaign's readout windows of ``--split`` (`multi_train.selection_windows`: of each airport, its real
windows of each span, D166 (33)), and of each airport ``--per-airport`` of them drawn once with ``--seed``. Every round
flies the same windows with the same numbers — `multi_train.readout_numbers` of ``--seed``, the window's place in the
set and the aircraft's in the window (as stage C's sets: not the readout's own places) — so the rounds stand side by
side.

WHICH ROUNDS. ``--rounds``: ``start`` (the campaign's start, a round of stage C's campaign with the token part at zero,
D164: `stage_d().start`) and the numbers of rounds whose checkpoint the campaign holds (`multi_train.round_model`).

HOW THEY ARE FLOWN. Stage D's window loop (`multi_train.loop_options`: its rule of who answers, D145, and its token part,
D152), its batches (`multi_train.span_batches`). A window's losses of a round: the ones its commanded aircraft answer for
(the loop's, `post_training_export.round_end_payload`, costing W), and every other loss that holds a commanded aircraft
at the states they flew, judged as the readout's census judges them (`census_losses`) — answered by none (D145),
costing nothing, drawn dashed: an answered loss at the step it was answered, a pair's unanswered run of steps once, at
its first step (`round_end`).

No speed readout is named: `model_speed` times stages B and C (D136); a stage D set's ``source.speed`` is null.

WRITES ``<root>/<airport>/training/<set-id>/sample.json`` and its entry in ``<root>/<airport>/training/index_multi_v1.json``;
refused when the set exists. Every airport is built before any is written. From a clean tree unless ``--smoke``; a smoke
campaign gives only a smoke set.

    python run_ts.py multi_training_export --campaign <a multi_train directory> --rounds start 0 --split select \\
        --per-airport 10 --set-id <id>
"""

from __future__ import annotations

import argparse
from typing import Any, Sequence

import numpy as np

from ts_transformer.experiments.multi_train import (
    MULTI_CAMPAIGN_SCHEMA, MULTI_CHECKPOINT_SCHEMA, loop_options, multi_settings_of, readout_numbers,
    round_model, selection_windows, span_batches, stage_d,
)
from ts_transformer.experiments.post_training_export import (
    START, Flying, StageExport, add_arguments, export_sets, round_end_payload,
)
from ts_transformer.experiments.post_window_loop import STEPS_BEFORE_EVENT, WindowLoop, WindowResult
from ts_transformer.multi.census import flown_positions
from ts_transformer.multi.separation import classify, judged_step
from ts_transformer.multi.windows import first_step_rows
from ts_transformer.post.scene import Window


def chosen_windows(context: Any, settings: Any, split: str, airport: str, per_airport: int, seed: int
                   ) -> tuple[list[Window], dict[str, Any]]:
    """The airport's windows (module docstring): ``per_airport`` of the campaign's readout windows of ``split`` at
    ``airport`` (real ones, D166 (33)), drawn once with ``seed`` (in their order), and the counts of the draw."""
    pool = [w for w in selection_windows(context, settings, split) if w.scene.geometry.code == airport]
    picked = sorted(np.random.default_rng([seed, 1 << 29]).permutation(len(pool))[:per_airport])
    return [pool[int(p)] for p in picked], {"pool": len(pool), "real": len(picked), "leftOut": {}}


def census_losses(context: Any, flown: WindowLoop, w: int, window: Window, ends: Sequence[WindowResult]
                  ) -> list[tuple[int, str, list[str], Any]]:
    """Every loss holding a commanded aircraft at the states window ``w``'s commanded aircraft flew, step by step: its
    step, its pair kind (`multi.separation.classify`), its two keys and the judge's loss. MIRROR of the readout's census
    (`multi.census.window_losses` through `multi_train.window_losses_of`: the same positions — an aircraft that landed
    judged once over its threshold, its landed runway read only in a window of several —, steps, landed ones counted as
    recorded, and kinds, except that the census splits a recorded-only loss by whether the records have it too; the
    census counts them where this lists them; pinned by `test_multi_training_export`)."""
    positions = flown_positions(window, [(e.states, e.words, flown.speaking.start,
                                          int(e.crossing["runway_index"]) if len(ends) > 1 and e.crossing is not None
                                          else None) for e in ends], context.words)
    code = window.scene.geometry.code
    separation, finals, step_s = context.separations[code], context.finals[code], context.words.spec.step_s
    first = int(first_step_rows(window)[0])
    last = int(round((positions.end_s - window.row0_s) / window.scene.interval_s))
    out = []
    for step in range(first, last + 1):
        judged = judged_step(window, positions, step, separation, finals, step_s)
        if judged is None:
            continue
        aircraft, commanded, losses = judged
        over = frozenset(int(k) for k in np.flatnonzero(aircraft.last_step[:commanded]))
        for loss in losses:
            kind = classify(loss.i, loss.j, loss.responsible, commanded, over)
            if kind is not None:
                out.append((step, kind, [aircraft.keys[loss.i], aircraft.keys[loss.j]], loss))
    return out


def round_end(context: Any, flown: WindowLoop, w: int, window: Window, ends: Sequence[WindowResult]) -> dict[str, Any]:
    """Window ``w``'s end in a round (module docstring): one entry for each loss a commanded aircraft answered (the
    loop's, at the step it answered it), and one for each run of a pair's loss over successive steps that no commanded
    aircraft answered at any of its steps (`census_losses`), at the run's first step, answered by none and costing
    nothing — a run holding an answered step is that answer's, never written again (a loss that starts while its
    responsible aircraft is still observed, D145, or goes on after its aircraft went silent); each with whether its
    aircraft read a faulty point near it (D114)."""
    keys = [flown.records[b].key for b in flown.members[w]]
    out = round_end_payload(ends, keys, window)
    answered = {(loss["step"], frozenset(loss["aircraft"])) for loss in out["losses"]}
    # the steps each recorded aircraft read a faulty point at, as the loop read them (D114): `WindowLoop`'s own record
    reading = flown._reading[w]                                                           # noqa: SLF001
    runs: list[list[tuple[int, list[str], Any]]] = []
    last: dict[frozenset[str], list[tuple[int, list[str], Any]]] = {}
    for step, _, pair, loss in census_losses(context, flown, w, window, ends):
        key = frozenset(pair)
        run = last.get(key)
        if run is None or run[-1][0] != step - 1:
            run = []
            runs.append(run)
            last[key] = run
        run.append((step, pair, loss))
    unanswered = []
    for run in runs:
        if any((step, frozenset(pair)) in answered for step, pair, _ in run):
            continue
        step, pair, loss = run[0]
        unanswered.append({
            "step": step, "timeS": round(window.step_s(step) - window.row0_s, 3), "aircraft": pair, "answering": [],
            "kind": loss.kind, "relation": loss.relation, "requiredM": round(loss.required_m, 1),
            "distanceM": round(loss.distance_m, 1), "verticalM": round(loss.vertical_m, 1),
            "wakeKnown": bool(loss.wake_known),
            "readsFault": any(key_ in reading.get(s, frozenset()) for key_ in pair
                              for s in range(step - STEPS_BEFORE_EVENT, step + 1)),
            "costsW": False})
    return {**out, "losses": sorted(out["losses"] + unanswered, key=lambda loss: loss["step"])}


def stage_d_flying(context: Any, settings: Any, seed: int) -> Flying:
    """Stage D's flying (module docstring)."""
    return Flying(batches=lambda windows: span_batches(windows, settings),
                  numbers=lambda place, member: readout_numbers(seed, place, member), options=loop_options(context),
                  end=lambda flown, w, window, ends: round_end(context, flown, w, window, ends))


#: Stage D's export (module docstring).
STAGE_D = StageExport(
    stage="D", campaign_schema=MULTI_CAMPAIGN_SCHEMA,
    campaign_formats={"multiCheckpoint": MULTI_CHECKPOINT_SCHEMA, "campaign": MULTI_CAMPAIGN_SCHEMA},
    settings_of=multi_settings_of,
    windows=lambda context, settings, split, airport, per_airport, kinds, seed: chosen_windows(
        context, settings, split, airport, per_airport, seed),
    model_of=lambda context, settings, campaign, r: (stage_d().start(context, settings)[0] if r == START
                                                     else round_model(context, settings, campaign, int(r))),
    flying=stage_d_flying,
    drawn_from="a seeded draw of the airport's readout windows (real, of every span: D166 (33))", speed=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    add_arguments(parser, STAGE_D)
    return export_sets(parser, parser.parse_args(argv), STAGE_D)


if __name__ == "__main__":
    raise SystemExit(main())
