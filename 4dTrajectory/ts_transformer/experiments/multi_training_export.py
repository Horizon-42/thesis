"""F4: the Training export of stage D (frontend §5.7, §8 F4; multi-aircraft control §10) — for each airport, a set of
windows in which every arrival of the window's span is commanded, which the frontend's Training view reads in stage D:
stage C's set format (`post_training_export`, one format for both stages), listed in the airport's stage D index
(``training/index_multi_v2.json``).

WHICH WINDOWS. A window list's (``--windows``, post-training D176 (3): each window flown with its readout's numbers,
`multi_train.readout_numbers` of the campaign's seed and its readout place), or the campaign's readout windows of
``--split`` (`multi_train.selection_windows`: of each airport, its real
windows of each span, D166 (33)), and of each airport ``--per-airport`` of them drawn once with ``--seed``. Every round
flies the same windows with the same numbers — `multi_train.readout_numbers` of ``--seed``, the window's place in the
set and the aircraft's in the window (as stage C's sets: not the readout's own places) — so the rounds stand side by
side.

WHICH ROUNDS. ``--rounds``: ``start`` (the campaign's start, a round of stage C's campaign with the token part at zero,
D164: `stage_d().start`) and the numbers of rounds whose checkpoint the campaign holds (`multi_train.round_model`).

HOW THEY ARE FLOWN. Stage D's window loop (`multi_train.loop_options`: its rule of who answers, D145, and its token part,
D152), its batches (`multi_train.span_batches`). A window's losses of a round: the ones its commanded aircraft answer for
(the loop's, `post_training_export.round_end_payload`, costing W), and every other loss that holds a commanded aircraft
at the states they flew, as the readout's census judges them (`multi.census.judged_steps`) — answered by none (D145),
costing nothing, drawn dashed: an answered loss at the step it was answered, a pair's unanswered run of steps once, at
its first step (`round_end`).

A set names no speed readout: `model_speed` writes its own file, which nothing binds to a set (the user, 2026-10-08).

WRITES ``<root>/<airport>/training/<set-id>/sample.json`` and its entry in ``<root>/<airport>/training/index_multi_v2.json``;
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
    START, Flying, Numbers, StageExport, add_arguments, export_sets, round_end_payload,
)
from ts_transformer.experiments.post_window_loop import STEPS_BEFORE_EVENT, WindowLoop, WindowResult
from ts_transformer.multi.census import flown_positions, judged_steps
from ts_transformer.post.scene import Window


def chosen_windows(context: Any, settings: Any, split: str, airport: str, per_airport: int, seed: int
                   ) -> tuple[list[Window], dict[str, Any]]:
    """The airport's windows (module docstring): ``per_airport`` of the campaign's readout windows of ``split`` at
    ``airport`` (real ones, D166 (33)), drawn once with ``seed`` (in their order), and the counts of the draw."""
    pool = [w for w in selection_windows(context, settings, split) if w.scene.geometry.code == airport]
    picked = sorted(np.random.default_rng([seed, 1 << 29]).permutation(len(pool))[:per_airport])
    return [pool[int(p)] for p in picked], {"pool": len(pool), "real": len(picked), "leftOut": {}}


def round_end(context: Any, flown: WindowLoop, w: int, window: Window, ends: Sequence[WindowResult]) -> dict[str, Any]:
    """Window ``w``'s end in a round (module docstring): one entry for each loss a commanded aircraft answered (the
    loop's, at the step it answered it), and one for each run of a pair's loss over successive steps that no commanded
    aircraft answered at any of its steps — the census's losses holding a commanded aircraft at the states they flew
    (`multi.census.judged_steps`, which the readout's census reads, frontend D177 (12)) —, at the run's first step,
    answered by none and costing nothing; a run holding an answered step is that answer's, never written again (a loss
    that starts while its responsible aircraft is still observed, D145, or goes on after its aircraft went silent); each
    with whether its aircraft read a faulty point near it (D114: the loop's `WindowLoop.fault_readings`, D177 (15))."""
    keys = [flown.records[b].key for b in flown.members[w]]
    out = round_end_payload(ends, keys, window)
    answered = {(loss["step"], frozenset(loss["aircraft"])) for loss in out["losses"]}
    reading = flown.fault_readings(w)
    # the commanded aircraft at the states they flew: MIRROR of `multi_train.window_losses_of`'s positions (the readout's
    # census; pinned by `test_multi_training_export`; one helper of stage D's would replace it)
    positions = flown_positions(window, [(e.states, e.words, flown.speaking.start,
                                          int(e.crossing["runway_index"]) if len(ends) > 1 and e.crossing is not None
                                          else None) for e in ends], context.words)
    code = window.scene.geometry.code
    runs: list[list[tuple[int, Any]]] = []
    last: dict[frozenset[str], list[tuple[int, Any]]] = {}
    for judged in judged_steps(window, positions, context.separations[code], context.finals[code],
                               context.words.spec.step_s):
        for item in judged.losses:
            key = frozenset(item.keys)
            run = last.get(key)
            if run is None or run[-1][0] != judged.step - 1:
                run = []
                runs.append(run)
                last[key] = run
            run.append((judged.step, item))
    unanswered = []
    for run in runs:
        if any((step, frozenset(item.keys)) in answered for step, item in run):
            continue
        step, item = run[0]
        loss = item.loss
        unanswered.append({
            "step": step, "timeS": round(window.step_s(step) - window.row0_s, 3), "aircraft": list(item.keys),
            "answering": [], "kind": loss.kind, "relation": loss.relation, "requiredM": round(loss.required_m, 1),
            "distanceM": round(loss.distance_m, 1), "verticalM": round(loss.vertical_m, 1),
            "wakeKnown": bool(loss.wake_known),
            "readsFault": any(key in reading.get(s, frozenset()) for key in item.keys
                              for s in range(step - STEPS_BEFORE_EVENT, step + 1)),
            "costsW": False})
    return {**out, "losses": sorted(out["losses"] + unanswered, key=lambda loss: loss["step"])}


def stage_d_flying(context: Any, settings: Any, numbers: Numbers) -> Flying:
    """Stage D's flying (module docstring), with the set's ``numbers``."""
    return Flying(batches=lambda windows: span_batches(windows, settings), numbers=numbers, options=loop_options(context),
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
    drawn_numbers=lambda settings, seed: lambda place, member: readout_numbers(seed, place, member),
    # the readout's numbers of each aircraft (`multi_train.read_batch`, D166 (34)): the campaign's seed, the place
    readout_numbers=lambda settings: lambda place, member: readout_numbers(settings.seed, place, member),
    selection=lambda context, settings: selection_windows(context, settings, "select"),
    identity=lambda window: {"flight": window.commanded.key, "row0_s": window.row0_s, "span_s": window.span_s},
    # the seed that orders the readout's windows and draws their numbers (`multi_train.selection_windows`)
    selection_fields=lambda settings: {"select_seed": settings.seed, "per_airport": settings.select_per_airport,
                                       "spans_s": list(settings.spans_s)},
    drawn_from="a seeded draw of the airport's readout windows (real, of every span: D166 (33))")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    add_arguments(parser, STAGE_D)
    return export_sets(parser, parser.parse_args(argv), STAGE_D)


if __name__ == "__main__":
    raise SystemExit(main())
