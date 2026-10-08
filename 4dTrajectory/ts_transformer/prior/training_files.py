"""The files of the frontend's Training view of stage B (prior design §12 B6, outline §6): the index of an airport's
prior sets and a set's sample, which the export (`experiments/prior_training_export.py`) writes and the backend's live
executor (`aeroviz_backend/autopilot_segment/`) and the frontend read.

Not a runner, and torch-free: its reader and writer (`FILES`) raise `ValueError` (a set that is not listed:
`instructions.training_files.NotListed`, one of them), never `SystemExit` — a server thread would let that escape its
handler and drop the request unanswered.

**Beside the other sets (outline §6 item 3).** A prior set is ``<airport>/training/<set-id>/sample.json``
(`SAMPLE_SCHEMA`), listed in ``<airport>/training/index_prior_v3.json`` (`INDEX_FILE`, `INDEX_SCHEMA`) — an index of its
own beside stage A's and the older ones, which this code never reads or writes. Read and written by the one definition of the three stages (`instructions.training_files.TrainingFiles`): this
module holds only stage B's constants (`FILES`).
"""

from __future__ import annotations

from ts_transformer.instructions.training_files import TrainingFiles

#: MIRROR of the frontend's reader of the prior's sets (B6); the reader refuses anything else by name, so these move
#: together. A name changes with its file's shape, on both sides, in the same change. Index v1 / sample v1 (B6,
#: 2026-10-05): the flights of one free-generation readout — the observed track, the open-loop sentence, the closed-loop
#: sentence at the prior's Δ and the sentences the prior said, flown again, with the words the procedure masks blocked.
#: Index v2 / sample v2 (B10, outline D109): a set's source names the claim of the val read it was exported under
#: (``validationClaim``: null for every set but the base's one validation readout, whose flights are of val); the
#: readers give val to that set alone.
#: Sample v3 (B13, D127): each prior sentence's flown track written unrounded (the live segment is checked against it
#: within the executor's bound).
#: Index v3 (``index_prior_v3.json``) / sample v4 (D135, D136, 2026-10-06): each sentence's block is stage A's
#: (`experiments.training_export.flown_sentence`), with the envelopes of its words. From 2026-10-08 a set's source
#: names no speed readout (the user unbound it; no reader ever read it): sets written before keep an unread ``speed``,
#: and the names stay.
INDEX_SCHEMA = "aeroviz-training-prior-index-v3"
INDEX_FILE = "index_prior_v3.json"
SAMPLE_SCHEMA = "aeroviz-training-prior-sample-v4"
SAMPLE_FILE = "sample.json"
SET_KIND = "prior-free-generation"
#: The reader of the val days whose claim a set's ``validationClaim`` names (`checkpoint.claim_validation_read`): the
#: base's one validation free generation. The export writes it, the backend's live service checks it; the frontend's
#: `TRAINING_PRIOR_CLAIM_READER` is its MIRROR.
CLAIM_READER = "prior_free_generation"

#: Stage B's Training files.
FILES = TrainingFiles(index_schema=INDEX_SCHEMA, index_file=INDEX_FILE, sample_schema=SAMPLE_SCHEMA, set_kind=SET_KIND)
