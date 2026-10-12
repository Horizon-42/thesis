"""One-off readout (paper tables, 2026-10-12): a stage D round's whole model flown single-aircraft on the base's val free
generation — the same read as stage C's `outputs/POOLED/post/single_val_post_seg60_r5_20261010/single_val.py`, for a
stage D campaign's round.

It runs `prior_free_generation` (R57) unchanged, with the base run and the base's val options, except in memory:
  * the model: after the runner opens the base, the model is given stage D's shape (`multi_train.stage_d` `shaped`:
    stage C's start model of the source campaign's traffic shape, then the token part), and the round's whole
    `state["model"]` is loaded, strict. Single-aircraft free generation gives the model no traffic (``extra`` None), and
    the traffic modules then output zero (`post/traffic_attention.py` `TrafficAttention.forward`);
  * the base's val claim (D85/D128, spent by the base's own read) is neither checked nor written: nothing is written
    into the base's run directory or the stage D campaign.
Afterwards config.json gains a `weights_from` block naming what flew.

usage (from a clean checkout's root, repo-relative paths):
  python single_val_stage_d.py <stage D campaign dir> <round> <out dir> [prior_free_generation options...]
"""
import json
import sys
from pathlib import Path

REPO = Path.cwd()
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "4dTrajectory"))

import torch  # noqa: E402

import ts_transformer.experiments.prior_free_generation as pfg  # noqa: E402
from ts_transformer.experiments.multi_train import MULTI_CHECKPOINT_SCHEMA  # noqa: E402
from ts_transformer.io_utils import file_sha256  # noqa: E402
from ts_transformer.multi.tokens import TOKENS_SCHEMA, TokenPart  # noqa: E402
from ts_transformer.post.traffic_attention import TrafficConfig, add_token_part, add_traffic_attention  # noqa: E402

campaign, round_, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
rest = sys.argv[4:]
checkpoint_path = REPO / campaign / f"round_{round_}" / "checkpoint.pt"
real_open = pfg.open_prior
loaded: dict = {}


def opened(prior_dir, instructions, **kw):
    prior = real_open(prior_dir, instructions, **kw)
    start = json.loads((REPO / campaign / "campaign.json").read_text())["inputs"]["settings"]["start"]
    source = json.loads((REPO / start["campaign"] / "campaign.json").read_text())["inputs"]["settings"]
    config = TrafficConfig(source["traffic_hidden"], source["traffic_heads"])
    state = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    held = state["identity"]
    if held["schema"] != MULTI_CHECKPOINT_SCHEMA or held["token_part"]["schema"] != TOKENS_SCHEMA \
            or len(held["rounds"]) != round_ + 1:
        raise ValueError(f"{checkpoint_path}: not round {round_} of a stage D campaign")
    model = prior.checkpoint.model
    add_traffic_attention(model, config)                    # stage C's `start_model` of the source campaign
    add_token_part(model, TokenPart.width, TOKENS_SCHEMA)   # stage D's `shaped`
    model.load_state_dict(state["model"], strict=True)      # every parameter of the round
    model.eval()
    loaded.update(parameters=len(state["model"]), held_rounds=len(held["rounds"]), traffic=config.to_dict(),
                  start=start)
    print(f"opened {checkpoint_path} in stage D's shape: {len(state['model'])} parameters", flush=True)
    return prior


def nothing(*args, **kwargs):
    return None


pfg.open_prior = opened
pfg.lock_val_read = nothing
pfg.settle_written_claim = nothing
pfg.claim_validation_read = nothing
pfg.spend_validation_claim = nothing

code = pfg.main([*rest, "--out", str(out)])
config_path = out / "config.json"
config = json.loads(config_path.read_text())
config["weights_from"] = {
    "what": "the round's whole model in stage D's shape (stage C's start model + the token part), flown with no "
            "traffic; the runner's prior/checkpoint_sha256 fields name the base it was built on",
    "campaign": campaign, "round": round_, "checkpoint_sha256": file_sha256(checkpoint_path), **loaded,
    "val_claim": "not checked, not written (one-off read for the paper tables, 2026-10-12)",
    "script": "single_val_stage_d.py (copied beside this config)"}
config_path.write_text(json.dumps(config, indent=2))
raise SystemExit(code)
