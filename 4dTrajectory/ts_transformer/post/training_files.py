"""The files of the frontend's Training view of stage C (post-training §8 C11, outline §6): the index of an airport's
window sets and a set's sample, which the export (`experiments/post_training_export.py`) writes and the backend's live
executor (`aeroviz_backend/autopilot_segment/`) and the frontend read.

Not a runner, and torch-free: its reader and writer (`FILES`) raise `ValueError` (a set that is not listed:
`instructions.training_files.NotListed`, one of them), never `SystemExit` — a server thread would let that escape its
handler and drop the request unanswered.

**Beside the other sets (outline §6 item 3).** A window set is ``<airport>/training/<set-id>/sample.json``
(`SAMPLE_SCHEMA`), listed in ``<airport>/training/index_post_v3.json`` (`INDEX_FILE`, `INDEX_SCHEMA`) — an index of its
own beside stage A's and stage B's, which this code never reads or writes. Read
and written by the one definition of the three stages (`instructions.training_files.TrainingFiles`): this module holds
only stage C's constants (`FILES`).

**One window format for stages C and D (frontend D156, §5.7).** A window holds a list of commanded aircraft — stage C's
one, stage D's several — and the index format is one for both stages: the stage is the index FILE's (stage C's
``index_post_v3.json``; stage D's ``index_multi_v1.json``, its own module).
"""

from __future__ import annotations

from ts_transformer.instructions.training_files import TrainingFiles

#: MIRROR of the frontend's reader of the window sets (C11); the reader refuses anything else by name, so these move
#: together. A name changes with its file's shape, on both sides, in the same change. Index v1 / sample v1 (C11,
#: 2026-10-05): windows of recorded traffic — the commanded flight's observed track, open-loop and closed-loop sentence
#: (stage A's head), the other aircraft on their records, and for each round of a post-training campaign the sentence
#: its model said for the commanded aircraft, flown, with the window's end (its outcome, a loss of separation and its
#: other aircraft, the reward). Sample v2 (prior D127, followed for windows, 2026-10-06): a round's flown track is written
#: unrounded — the live executor's answer is checked against it within the executor's bound. Index v2
#: (``index_post_v2.json``) / sample v3 (D135, 2026-10-06): each round's block is stage A's
#: (`experiments.training_export.flown_sentence`), with the envelopes of its words. Index v3 (``index_post_v3.json``) /
#: sample v4 (frontend D156, 2026-10-07): the window format of stages C and D — a window's commanded aircraft are a list
#: (stage C writes one), each with its join offset, its move, its first predicted step and each round's flown sentence
#: with its reward and the row from which it is silent; each round's losses of separation are the window's, each with its
#: two aircraft, the ones that answer for it and whether it costs W; the set kind is the two stages' one.
INDEX_SCHEMA = "aeroviz-training-window-index-v3"
INDEX_FILE = "index_post_v3.json"
SAMPLE_SCHEMA = "aeroviz-training-window-sample-v4"
SAMPLE_FILE = "sample.json"
SET_KIND = "training-windows"

#: Stage C's Training files.
FILES = TrainingFiles(index_schema=INDEX_SCHEMA, index_file=INDEX_FILE, sample_schema=SAMPLE_SCHEMA, set_kind=SET_KIND)
