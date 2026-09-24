"""The prior's predictions over a Training set's flights, for the frontend: at every predicted step of each flight's
truth sentence, what the trained prior gives each column — the probability that a word is said, the words most likely
if one is, and the probability of the truth — with its val readout against the two baselines (prior design §5, §8; the
Training module: `aeroviz-4d/docs/36-2026-09-20-training-module.zh.md`).

    python run_ts.py prior_training_export \\
        --prior 4dTrajectory/outputs/POOLED/prior/<campaign>/<the chosen run> \\
        --instructions 4dTrajectory/outputs/POOLED/instruction_language/<its artefact> \\
        --airports-root aeroviz-4d/public/data/airports --set <a read-back set of that artefact> \\
        --airport KMSY --airport KRDU --airport KSJC --airport KSMF --airport KSTL

Per airport, writes ``<airports-root>/<ICAO>/training/<overlay-id>/prior.json`` (schema `SCHEMA`; refused if the
directory exists) and adds the overlay to ``training/overlays.json`` (`instruction_training_export.OVERLAYS_SCHEMA`;
refused if it lists the id already). The set (``--set``) must be a read-back set of the prior's vocabulary drawn from
val — flights the prior never trained on.

**The predictions are teacher-forced**, as the prior was trained and read out: each step sees the truth sentence's
words before it (`prior.data.flight_record` over the stored sentence, which must be the set's own), never the prior's
own earlier guesses — this is not the prior speaking a sentence of its own (that needs the executor in the loop).
The checkpoint is refused unless `prior_train.load_prior` opens it on this artefact (spec, labeller, day split,
candidate table, a whole state), it is not a smoke run, it holds a val readout (``readout.json``, the chosen variant's,
`prior_select`) — which travels with the predictions, unchanged — and today's tracks rosters are the ones it was
trained with (the landing context is read from them).

**Only the predicted steps** (from `scene.N_LOOK`, ``firstPredictedRow``) carry predictions: the rows before are
observed only. Per column and predicted step: ``changeP``, the probability that a word is said (1 − P(unchanged); 1 at
the first predicted step, which says every column); ``words`` / ``wordsP``, the column's ``k`` most likely words GIVEN
that one is said (the readout's top-k convention), each a value as the sentence writes it (the runway pointer's: the
candidate's index) — ``k`` is `TOP_K`, or fewer when the column has fewer values (an airport's candidate runways);
``truthP``, the probability of what the truth sentence says there (a word, or unchanged). The probabilities are rounded
to `PROBABILITY_DIGITS`; each flight's negative log-likelihood per predicted step is computed before rounding.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from ts_transformer.experiments.instruction_training_export import (
    KIND_PRIOR, SPLIT, BaseSet, base_flights, open_base_set, overlay_entry, read_overlays, serialise_overlay, write_overlay,
)
from ts_transformer.experiments.prior_train import PRIOR_CHECKPOINT_SCHEMA, load_prior, roster_record, rosters
from ts_transformer.instructions.artefact import load_candidates, load_sentences, load_signals, load_spec
from ts_transformer.instructions.words import COLUMNS
from ts_transformer.io_utils import file_sha256, utc_now
from ts_transformer.prior.data import VARIANTS, Split, airport_landings, batches, flight_record, runway_names
from ts_transformer.prior.model import Prior
from ts_transformer.prior.scene import N_LOOK
from ts_transformer.prior.train import TrainConfig, batch_logits, to_batch
from ts_transformer.repo_layout import REPO_ROOT, git_state

#: MIRROR of `aeroviz-4d/src/data/trainingOverlays.ts` (`TRAINING_PRIOR_SCHEMA`); the reader refuses anything else by
#: name. A name changes with its file's shape or meaning, on both sides, in one change. v3 (2026-09-24, the prior's
#: third version): the per-step arrays cover the predicted steps only (``firstPredictedRow`` on); a column that never
#: changes after the first step has no change metrics (null); the first step's runway is read against the rules.
SCHEMA = "aeroviz-training-prior-v3"
PAYLOAD_FILE = "prior.json"
RUNNER = "ts_transformer.experiments.prior_training_export"
TOP_K = 3
PROBABILITY_DIGITS = 4


def open_prior(directory: Path, instructions: Path) -> tuple[Prior, dict[str, Any], dict[str, Any], str]:
    """The prior at ``directory`` on CPU, in eval mode, with its config and readout files and its checkpoint's
    sha256 — refused unless `prior_train.load_prior` opens it on ``instructions``, it is not a smoke run, it holds
    a val readout (`prior_select` writes one, on the chosen variant only) and today's tracks rosters are its own."""
    model, _, config_file = load_prior(directory, instructions)
    if roster_record(rosters(instructions)) != config_file["tracks_rosters"]:
        raise SystemExit(f"the tracks rosters changed since {directory} was trained (its landing context)")
    if config_file["smoke"]:
        raise SystemExit(f"{directory} is a smoke run (--limit {config_file['limit']}), not a trained prior")
    if not (directory / "readout.json").exists():
        raise SystemExit(f"{directory} holds no val readout: it is not a chosen prior (prior_select)")
    readout = json.loads((directory / "readout.json").read_text(encoding="utf-8"))
    return model, config_file, readout, file_sha256(directory / "checkpoint.pt")


@torch.no_grad()
def predictions(model: Prior, split: Split) -> list[dict[str, Any]]:
    """Every flight of ``split``, in its order: per column and predicted step, the probability that a word is said,
    the ``k`` most likely words given that one is (values: class − 1) and their probabilities, and the probability of
    the truth; and the flight's negative log-likelihood per predicted step, in all and per column."""
    out: list[dict[str, Any] | None] = [None] * len(split.flights)
    for indices in batches(split.flights, TrainConfig().tokens_per_batch, None):
        batch = to_batch(split, indices, torch.device("cpu"))
        logits = batch_logits(model, batch)
        for b, i in enumerate(indices):
            rows = split.flights[i].rows
            columns, nll = [], []
            for c, logit in enumerate(logits):
                logit = logit[b, 0, N_LOOK:rows].double()
                log_p = torch.log_softmax(logit, dim=-1)
                target = batch["targets"][b, 0, N_LOOK:rows, c]
                truth = log_p.gather(1, target[:, None])[:, 0]
                nll.append(float(-truth.sum()) / (rows - N_LOOK))
                # a word's probability given that one is said; the runway head's slots past the airport's candidates
                # are masked to -inf and never ranked
                said = torch.softmax(logit[:, 1:], dim=-1)
                k = min(TOP_K, int(torch.isfinite(logit[:, 1:]).sum(dim=1).max()))
                top_p, top = said.topk(k, dim=-1)
                columns.append({"k": k, "changeP": _round(1.0 - log_p[:, 0].exp()), "words": top.flatten().tolist(),
                                "wordsP": _round(top_p.flatten()), "truthP": _round(truth.exp())})
            out[i] = {"rows": rows, "firstPredictedRow": N_LOOK, "nllPerStep": round(sum(nll), 6),
                      "columnNllPerStep": [round(v, 6) for v in nll], "columns": columns}
    return out   # type: ignore[return-value]


def _round(values: torch.Tensor) -> list[float]:
    return [round(float(value), PROBABILITY_DIGITS) for value in values]


def readout_block(readout: dict[str, Any]) -> dict[str, Any]:
    """The prior's val readout (`prior.readout`), as its file holds it, in the frontend's spelling."""
    model = readout["model"]

    def runway(part: dict[str, Any]) -> dict[str, Any]:
        return {"top1": part["top1"], "direction": part["direction"], "sideGivenDirection": part["side_given_direction"]}

    return {
        "split": readout["split"], "steps": model["steps"], "bestEpoch": readout["best_epoch"],
        "model": {"nllPerStep": model["nll_per_step"], "perplexityPerStep": model["perplexity_per_step"],
                  "perColumn": {name: {"nllPerStep": values["nll_per_step"], "changeSteps": values["change_steps"],
                                       "firstStepTop1": values["first_step_top1"],
                                       "changeProbabilityWhereChanged": values["mean_change_probability_where_changed"],
                                       "top1GivenChange": values["top1_given_change"],
                                       "top5GivenChange": values["top5_given_change"],
                                       "falseChangeShareWhereKept": values["false_change_share_where_kept"]}
                                for name, values in model["per_column"].items()}},
        "baselines": {"repeat": readout["baselines"]["repeat"],
                      "previousWord": readout["baselines"]["previous word"]},
        "firstStepRunway": {"model": runway(model["first_step_runway"]),
                            "airportFrequency": runway(readout["airport_runway_frequency"]),
                            "rules": {rule: runway(part) for rule, part in readout["rules"].items()}},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--prior", type=Path, required=True, help="the trained prior's directory (prior_train --out)")
    parser.add_argument("--instructions", type=Path, required=True, help="the instruction artefact it was trained on")
    parser.add_argument("--airports-root", type=Path, required=True,
                        help="the frontend's airports directory (…/public/data/airports)")
    parser.add_argument("--set", required=True, help="the Training set whose flights are predicted, e.g. instruction_v3")
    parser.add_argument("--airport", action="append", required=True, help="an ICAO code; repeat for several")
    parser.add_argument("--overlay-id", default=None, help="default: prior_<the prior directory's name>")
    args = parser.parse_args(argv)

    def resolved(path: Path) -> Path:
        return path if path.is_absolute() else REPO_ROOT / path

    prior_dir, instructions, root = (resolved(p) for p in (args.prior, args.instructions, args.airports_root))
    airports = [code.upper() for code in args.airport]
    if len(set(airports)) != len(airports):
        parser.error(f"an airport is named twice in {airports}")
    overlay_id = args.overlay_id or f"prior_{prior_dir.name}"
    started = time.perf_counter()
    model, config_file, readout, checkpoint_sha = open_prior(prior_dir, instructions)
    spec = load_spec(instructions)
    missing = [code for code in airports if code not in model.config.airports]
    if missing:
        parser.error(f"the prior knows no airport {missing} ({list(model.config.airports)})")
    geometries = load_candidates(instructions)
    landings = airport_landings(instructions, rosters(instructions))
    context = VARIANTS[model.config.variant].landing_context
    flights = load_signals(instructions, SPLIT)
    sentences = load_sentences(instructions, SPLIT, spec)
    bases: dict[str, BaseSet] = {}
    existing = {}
    for code in airports:
        training = root / code / "training"
        if (training / overlay_id).exists():
            parser.error(f"{training / overlay_id} exists; an overlay is never overwritten")
        existing[code] = read_overlays(training, code, overlay_id)
        bases[code] = open_base_set(training, code, args.set, spec)

    def relative(path: Path) -> str:
        return path.relative_to(REPO_ROOT).as_posix() if path.is_relative_to(REPO_ROOT) else path.as_posix()

    source = {"runner": RUNNER, "prior": relative(prior_dir), "instructions": relative(instructions), "git": git_state()}
    model_config = model.config
    prior_block = {
        "checkpointSha256": checkpoint_sha, "schema": PRIOR_CHECKPOINT_SCHEMA, "specSha256": spec.sha256,
        "parameters": config_file["parameters"], "writtenUtc": config_file["written_utc"],
        "trainedAt": config_file["git"],
        "model": {"dModel": model_config.d_model, "layers": model_config.layers, "heads": model_config.heads,
                  "feedforward": model_config.feedforward, "dropout": model_config.dropout,
                  "airports": list(model_config.airports), "classes": list(model_config.classes)},
        "train": asdict(TrainConfig(**config_file["train"])),
        "method": (f"teacher-forced: each step sees the truth sentence's words before it; variant {model_config.variant}; "
                   f"the first {N_LOOK} rows are observed only, the first predicted step says every column"),
    }
    title = f"Prior {prior_dir.name} · its predictions at each step of the truth sentence (teacher-forced)"
    built = {}
    for code in airports:
        base = bases[code]
        located = base_flights(base, flights, sentences)
        split = Split([flight_record(signal, sentences["words"][sentences["offsets"][k]: sentences["offsets"][k + 1]],
                                     geometries[code], landings[code] if context else None,
                                     model_config.airports.index(code), int(sentences["capture_row"][k]))
                       for signal, k in located],
                      model_config.airports, model.candidates.numpy(), *runway_names(geometries, model_config.airports),
                      model_config.classes, model_config.variant)
        per_flight = predictions(model, split)
        payloads = [{"flightKey": item["flightKey"], "datasetId": item["datasetId"], **values}
                    for item, values in zip(base.sample["flights"], per_flight)]
        payload = {"schema": SCHEMA, "overlayId": overlay_id, "airport": code, "writtenUtc": utc_now(),
                   "producedBy": source, "base": base.block, "prior": prior_block, "readout": readout_block(readout),
                   "columns": list(COLUMNS), "flights": payloads}
        entry = overlay_entry(overlay_id, KIND_PRIOR, base, title, PAYLOAD_FILE, len(payloads), source)
        built[code] = (serialise_overlay(payload), entry)
        steps = sum(item["rows"] - item["firstPredictedRow"] for item in payloads)
        nll = sum(item["nllPerStep"] * (item["rows"] - item["firstPredictedRow"]) for item in payloads) / steps
        print(f"  {code}: {len(payloads)} flights, {steps} predicted steps, NLL {nll:.4f} per step (val "
              f"{readout['model']['nll_per_step']:.4f}); {time.perf_counter() - started:.0f} s", flush=True)
    for code, (text, entry) in built.items():
        out = write_overlay(root / code / "training", code, overlay_id, entry, text, existing[code])
        print(f"  {code}: {out.stat().st_size / 1e6:.1f} MB → {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
