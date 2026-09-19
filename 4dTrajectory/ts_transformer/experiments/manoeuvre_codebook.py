"""Export a jointly trained executor's tokenizer as a FROZEN codebook artefact (plan §2.4).

    python run_ts.py manoeuvre_codebook --checkpoint <run>/checkpoint.pt \\
        --out 4dTrajectory/outputs/codebooks/<name>

The checkpoint must be a `plan_conditioning = manoeuvre-code` executor whose tokenizer trained
JOINTLY (``manoeuvre_codebook`` empty); an executor trained AGAINST a codebook already has one
and is refused with its path. The artefact (`manoeuvre.tokenizer.write_codebook`) records the
tokenizer's kind, levels and segment, the fitted cohort's identity (the checkpoint's
`data_provenance` — C26: the eligible SET digests, never roster bytes) and the source
checkpoint (path, sha256, run name). It is never overwritten: an existing directory refuses.
The codebook's sha256 is what every prior (P2) and every executor re-trained against it
(P3.3, ``manoeuvre_codebook``) binds to.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ts_transformer.config import PLAN_CONDITIONING_MANOEUVRE_CODE, token_span_s
from ts_transformer.data.data_provenance import provenance_eligible_set_digests
from ts_transformer.io_utils import file_sha256, sha256_bytes
from ts_transformer.manoeuvre.tokenizer import write_codebook
from ts_transformer.run_naming import run_display_name
from ts_transformer.training.train import load_checkpoint


def export_codebook(checkpoint: Path, out: Path):
    """The codebook of ``checkpoint``'s tokenizer, written to ``out``."""
    model, config, _normalizer, payload = load_checkpoint(checkpoint)
    if config.plan_conditioning != PLAN_CONDITIONING_MANOEUVRE_CODE:
        raise ValueError(f"{checkpoint}: plan_conditioning={config.plan_conditioning!r} has no tokenizer to export")
    if config.manoeuvre_codebook:
        raise ValueError(
            f"{checkpoint} was trained AGAINST the codebook at {config.manoeuvre_codebook}; "
            "that directory is its codebook, there is nothing new to export"
        )
    provenance = payload["data_provenance"]
    return write_codebook(
        out, model.manoeuvre_tokenizer, segment_s=token_span_s(config), dt_s=config.dt_s,
        data_identity={
            "schema_version": provenance.get("schema_version"),
            "eligible_set_sha256": provenance_eligible_set_digests(provenance),
            # the split's IDENTITY (the sorted key lists' shas), never its counts
            "split_sha256": {name: sha256_bytes("\n".join(sorted(keys)).encode()) for name, keys in payload["split"].items()},
            "split_counts": {name: len(keys) for name, keys in payload["split"].items()},
        },
        source={
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "run": run_display_name(config.to_dict()),
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="the codebook directory (must not exist)")
    args = parser.parse_args(argv)
    codebook = export_codebook(args.checkpoint, args.out)
    print(
        f"codebook {codebook.path}: {codebook.kind}, levels {list(codebook.levels)}, K = {codebook.code_count}, "
        f"segment {codebook.segment_s:g} s, sha256 {codebook.sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
