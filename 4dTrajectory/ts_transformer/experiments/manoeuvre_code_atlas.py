"""The code atlas (plan §2.4 "码怎么看"): every code flown by the executor from a few typical
val states, the K segment paths and end points in each state's start frame beside the truth's.

    python run_ts.py manoeuvre_code_atlas --executor <arm>/checkpoint.pt --codebook <dir> --out <dir> \\
        [--flights 8] [--seed 1337] [--device auto]

The flights are drawn at random (seeded) from the executor's val split, half straight-in and half
vectored where the cohort allows; the codebook must be this executor's (a joint executor's
exported one, or the one it was trained against). Writes ``manoeuvre_code_atlas.json``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from ts_transformer.backbone.adapters import resolve_device
from ts_transformer.data.approach_difficulty import STRAIGHT_TORTUOSITY, approach_difficulty
from ts_transformer.data.data_provenance import checkpoint_data_provenance, require_matching_data_provenance
from ts_transformer.data.dataset import build_series, load_flight_dicts
from ts_transformer.experiments.support import REPO_ROOT
from ts_transformer.io_utils import file_sha256, utc_now, write_json_atomic
from ts_transformer.manoeuvre.readout import ATLAS_PATH_STRIDE, code_atlas
from ts_transformer.manoeuvre.tokenizer import load_codebook
from ts_transformer.repo_layout import arrival_manifest_path
from ts_transformer.training.train import load_checkpoint, usable_series
from ts_transformer.config import default_anchor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--executor", type=Path, required=True)
    parser.add_argument("--codebook", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--flights", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if out.exists():
        parser.error(f"{out} exists; an atlas is never overwritten")
    started = time.perf_counter()
    device = resolve_device(args.device)
    executor_path = args.executor if args.executor.is_absolute() else REPO_ROOT / args.executor
    model, config, normalizer, payload = load_checkpoint(executor_path)
    model = model.to(device).eval()
    codebook = load_codebook(args.codebook if args.codebook.is_absolute() else REPO_ROOT / args.codebook)
    executor_sha = file_sha256(executor_path)
    bound = model.codebook_sha256 == codebook.sha256 if config.manoeuvre_codebook else codebook.source.get("checkpoint_sha256") == executor_sha
    if not bound:
        parser.error(f"{codebook.path} is not this executor's codebook")
    airports = tuple(entry["airport"] for entry in payload["data_provenance"]["manifests"])
    manifests = [arrival_manifest_path(item) for item in airports]
    require_matching_data_provenance(payload, checkpoint_data_provenance(payload, manifests))
    # the draw: seeded, then half straight-in / half vectored where possible — the series are
    # built for a generous prefix and the covariates decide
    rng = np.random.default_rng(args.seed)
    val = list(payload["split"]["val"])
    candidates = [val[i] for i in rng.permutation(len(val))[: max(8 * args.flights, 64)]]
    built, _report = build_series(load_flight_dicts(manifests, include_flight_keys=set(candidates), verbose=False), config,
                                  aircraft_type=config.aircraft_type)
    by_id = {item.dataset_id: item for item in usable_series(built, config, verbose=False)}
    anchor = default_anchor(config)
    straight, vectored = [], []
    for key in candidates:
        item = by_id.get(key)
        if item is None:
            continue
        (straight if approach_difficulty(item, anchor).route_tortuosity < STRAIGHT_TORTUOSITY else vectored).append(item)
    half = args.flights // 2
    chosen = straight[:half] + vectored[: args.flights - half]
    if len(chosen) < args.flights:
        chosen = (straight + vectored)[: args.flights]
    atlas = code_atlas(model, config, normalizer, codebook, chosen, device)
    result = {
        "schema": "ts-manoeuvre-code-atlas-v1", "written_utc": utc_now(), "executor": str(executor_path),
        "executor_sha256": executor_sha, "codebook": str(codebook.path), "codebook_sha256": codebook.sha256,
        "code_count": codebook.code_count, "segment_s": codebook.segment_s, "anchor": anchor, "path_stride_rows": ATLAS_PATH_STRIDE,
        "frame": "each flight's segment-start frame at the anchor: x along its course, y left, z up, metres",
        "flights": atlas, "elapsed_s": time.perf_counter() - started,
    }
    out.mkdir(parents=True)
    write_json_atomic(out / "manoeuvre_code_atlas.json", result)
    for entry in atlas:
        ends = np.array([code["end"] for code in entry["codes"]])
        spread = float(np.linalg.norm(ends - ends.mean(axis=0), axis=1).mean())
        print(f"  {entry['flight_id']:<40} tortuosity {entry['route_tortuosity']:.2f}  truth code {entry['truth_code']:>3}  "
              f"end-point spread over codes {spread:.0f} m  truth end x/y {entry['truth_end'][0]:.0f}/{entry['truth_end'][1]:.0f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
