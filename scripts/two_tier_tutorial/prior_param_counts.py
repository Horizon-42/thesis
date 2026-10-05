"""Parameter counts of the prior's four configurations (prior design D40), counted from the stage B code.

The stage B code is on the branch `dev-two-tier-v4-prior` and is not merged into `dev-two-tier` yet, so this script
reads it from its worktree (a separate process: two copies of `ts_transformer` must never load together).

    conda run -n aeroviz python scripts/two_tier_tutorial/prior_param_counts.py <checkout with prior/>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if len(sys.argv) != 2:
    sys.exit("usage: prior_param_counts.py <checkout that holds ts_transformer/prior>")
PRIOR = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(PRIOR), str(PRIOR / "4dTrajectory"), str(PRIOR / "geokit" / "src")]

from ts_transformer.instructions.artefact import load_spec  # noqa: E402
from ts_transformer.instructions.words import Words  # noqa: E402
from ts_transformer.prior.model import Prior, PriorConfig  # noqa: E402

ARTEFACT = Path("/home/supercomputing/studys/thesis/4dTrajectory/outputs/POOLED/instruction_language/v12_20261005")
# prior design D40: the four configurations; every head is 32 wide
CONFIGS = {
    "A": dict(d_model=192, layers=4, heads=6, feedforward=768, dropout=0.1),
    "B": dict(d_model=128, layers=4, heads=4, feedforward=512, dropout=0.1),
    "C": dict(d_model=256, layers=4, heads=8, feedforward=1024, dropout=0.1),
    "D": dict(d_model=192, layers=4, heads=6, feedforward=768, dropout=0.2),
}


def main() -> None:
    words = Words(load_spec(ARTEFACT))
    out = {}
    for name, shape in CONFIGS.items():
        for variant in ("full", "constants"):
            config = PriorConfig.from_words(words, variant, **shape)
            out[f"{name}/{variant}"] = sum(p.numel() for p in Prior(config).parameters())
    path = HERE / "data" / "prior_params.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
