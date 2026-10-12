"""One-off readout (paper tables, 2026-10-12; the user chose stage D's windows of the val days): another model than a
stage D round read exactly as `multi_validation` (R-, MC4) reads a round — the same val windows (`selection_windows`
with the campaign's seed, spans and ``select_per_airport``), the same random numbers (draw 0), the same first pass and
`selection_readout` — so that the base, stage D's start and the chosen round stand side by side on one set.

The models, each given stage D's shape (`multi_train.stage_d`):
  * ``base``: `Stage.start_model` — the base with zero-output traffic modules and a token part at 0 (post-training D29):
    the base's own behaviour;
  * ``start``: `Stage.start` — the campaign's start (a stage C round), its token part at 0.
The campaign's val claim is neither checked nor written (the campaign directory is not touched): the read is
information beside the chosen round's own val read.

usage (from a clean checkout's root, in the aeroviz env):
  python multi_val_models.py <stage D campaign dir> <base|start> <out dir> <val|select>
The select split is the behaviour check: ``start`` on select must give the start's select readout
(`outputs/POOLED/multi/handover_20261010/start_read/start_readout.json`).
"""
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

REPO = Path.cwd()
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "4dTrajectory"))

import torch  # noqa: E402

from ts_transformer.autopilot.closed_loop import require_conforming_closed_loop  # noqa: E402
from ts_transformer.experiments.multi_train import (  # noqa: E402
    multi_settings_of, selection_readout, selection_windows, span_key, stage_d, window_record,
)
from ts_transformer.experiments.multi_validation import MULTI_VALIDATION_SCHEMA  # noqa: E402
from ts_transformer.experiments.post_train import inputs_here, open_context, split_data  # noqa: E402
from ts_transformer.experiments.post_window_loop import checked_edges  # noqa: E402
from ts_transformer.io_utils import utc_now, write_json_atomic  # noqa: E402
from ts_transformer.prior.checkpoint import readable_identity  # noqa: E402
from ts_transformer.prior.source import require_selection_of  # noqa: E402
from ts_transformer.repo_layout import git_state  # noqa: E402

MODELS = ("base", "start")
campaign, which, out, SPLIT = REPO / sys.argv[1], sys.argv[2], REPO / sys.argv[3], sys.argv[4]
device = torch.device("cuda")
if SPLIT not in ("val", "select"):
    raise SystemExit(f"the split is val or select, not {SPLIT!r}")
if which not in MODELS:
    raise SystemExit(f"the model is one of {MODELS}, not {which!r}")
if out.exists():
    raise SystemExit(f"{out} exists")
git = git_state()
if git["dirty"]:
    raise SystemExit("a readout needs a clean tree")

record = json.loads((campaign / "campaign.json").read_text(encoding="utf-8"))
settings = multi_settings_of(record)
inputs = inputs_here(record["inputs"])
instructions, executor = Path(inputs["instructions"]), Path(inputs["executor"])
_, opened, _ = require_conforming_closed_loop(instructions, executor)
edges_reference = Path(inputs["windows"]) / "conformance" / "edges.npz"
checked_edges(edges_reference)
context = open_context(Path(inputs["prior"]), instructions, executor, edges_reference, device,
                       Path(inputs["procedure_root"]), formal=True, data=False, splits=())
stage = stage_d()
model, _ = stage.start_model(context, settings) if which == "base" else stage.start(context, settings)
model.eval()
require_selection_of(instructions, context.interval_s, context.base_identity, SPLIT)
context = replace(context, splits={SPLIT: split_data(instructions, SPLIT, context.words, context.interval_s,
                                                     context.geometries, executor)})
windows = selection_windows(context, settings, SPLIT)
readout = selection_readout(model, context, windows, settings, None, stage=stage, split=SPLIT)
anchors = Counter(w.scene.geometry.code for w in context.splits[SPLIT]["windows"])
read = Counter((w.scene.geometry.code, span_key(w.span_s)) for w in windows)
coverage = {code: {"anchors": anchors[code], "read": {span_key(span): read[code, span_key(span)]
                                                      for span in settings.spans_s}} for code in sorted(anchors)}
out.mkdir(parents=True)
write_json_atomic(out / "config.json", {
    "schema": MULTI_VALIDATION_SCHEMA, "written_utc": utc_now(), "campaign": str(campaign), "model": which,
    "model_is": {"base": "stage D's start_model: the base, zero-output traffic modules, token part at 0",
                 "start": f"stage D's start: {settings.start}, token part at 0"}[which],
    "base": readable_identity(context.base_identity), "settings": inputs["settings"],
    "instructions": str(instructions), "executor": str(executor), "checks": opened["checks"],
    "edges_reference": str(edges_reference), "git": git, "device": str(device),
    "split": SPLIT, "val_claim": "not checked, not written (one-off read for the paper tables, 2026-10-12)",
    "script": "multi_val_models.py (copied beside this output's run directory)"})
write_json_atomic(out / "readout.json", {
    "schema": MULTI_VALIDATION_SCHEMA, "split": SPLIT, "model": which, "coverage": coverage,
    "windows": [window_record(w) for w in windows], "readout": readout})
print(json.dumps({"out": str(out), "reward_mean": {code: r["reward_mean"] for code, r in readout.items()}}))
