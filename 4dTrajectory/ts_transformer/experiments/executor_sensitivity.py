"""Executor E7 check: how much the executor's result moves with the parameters method A and B set (executor
design §9, the E7 plan item 5) — on TRAIN, never val.

One seeded train sample (`replay.draw`: per airport, own dynamics, a published approach speed) is flown once
per variant: the spec's own parameters, then each of τ_ψ, p, the γ̇ factor and the three delays moved alone.
τ_ψ runs from 2 s up to the spec's value (a larger one breaks the split-turn constraint), p over 2–5°/s, the
γ̇ factor over 1–3, each delay by ± `DELAY_STEP_S`. A delay moved below 0 makes a word act BEFORE it is said:
such a variant is a probe, flown with `ExecutorParams.check`'s ``early_words`` and marked ``probe`` in the
readout — it measures what the labeller's late reading of a manoeuvre's onset costs (method B's finding), and
could never be a spec's value. Every variant reports the landed share, the share flown as said, the words
inside their envelopes (per word judged), the word checks that failed, and how far the flown tracks lie from
the observed ones (`replay.alignment`).

Writes into ``--out`` (a new directory; default beside the spec): ``variants/<nn>_<variant>.json`` as each is
flown, then ``sensitivity.json`` with every row.

    python run_ts.py executor_sensitivity \\
        --instructions 4dTrajectory/outputs/POOLED/instruction_language/v2_20260924 \\
        --executor 4dTrajectory/outputs/POOLED/executor/<name>
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from ts_transformer.autopilot import replay
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.autopilot.sentence import DELAY_GROUPS
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.repo_layout import REPO_ROOT, git_state

HEADING_TIME_CONSTANTS_S = (2.0, 3.0, 4.0)
BANK_RATES_DEG_S = (2.0, 3.0, 4.0, 5.0)
PATH_RATE_FACTORS = (1.0, 2.0, 3.0)
DELAY_STEP_S = 4.0


def variants(params: ExecutorParams) -> list[tuple[str, ExecutorParams, bool]]:
    """``(name, params, probe)``: the spec's own first, then one parameter moved at a time (a value equal to
    the spec's is not flown twice)."""
    out = [("spec", params, False)]
    for tau in HEADING_TIME_CONSTANTS_S:
        if tau < params.heading_time_constant_s:
            out.append((f"heading_time_constant_s={tau:g}", replace(params, heading_time_constant_s=tau), False))
    for rate in BANK_RATES_DEG_S:
        if rate != params.bank_rate_deg_s:
            out.append((f"bank_rate_deg_s={rate:g}", replace(params, bank_rate_deg_s=rate), False))
    for factor in PATH_RATE_FACTORS:
        if factor != params.path_rate_factor:
            out.append((f"path_rate_factor={factor:g}", replace(params, path_rate_factor=factor), False))
    for name in DELAY_GROUPS:
        for sign in (-1.0, 1.0):
            value = getattr(params.delays, name) + sign * DELAY_STEP_S
            moved = replace(params, delays=replace(params.delays, **{name: value}))
            out.append((f"delays.{name}={value:g}", moved, value < 0.0))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True, help="the executor spec directory")
    parser.add_argument("--per-airport", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", type=Path, default=None,
                        help="default: <executor>/sensitivity-train-<per-airport>-seed<seed>")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    instructions = args.instructions if args.instructions.is_absolute() else REPO_ROOT / args.instructions
    executor = args.executor if args.executor.is_absolute() else REPO_ROOT / args.executor
    out = args.out or executor / f"sensitivity-train-{args.per_airport}-seed{args.seed}"
    out = out if out.is_absolute() else REPO_ROOT / out
    if out.exists():
        parser.error(f"{out} exists; a readout is never overwritten")
    params, record, words = replay.open_executor(executor, instructions)
    spec = words.spec
    device = torch.device(args.device)
    started = time.perf_counter()
    batch = replay.draw(instructions, "train", spec, words, per_airport=args.per_airport, seed=args.seed)
    print(f"{batch.drawn['flights']} train flights drawn ({batch.drawn['read']} read, excluded "
          f"{batch.drawn['excluded']}), {time.perf_counter() - started:.0f}s", flush=True)

    (out / "variants").mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for number, (name, moved, probe) in enumerate(variants(params)):
        moved.check(spec, early_words=probe)
        flown, verdicts = replay.fly_batch(batch, moved, words, device=device, early_words=probe)
        row = {"variant": name, "probe": probe, "params": asdict(moved), **replay.summary(verdicts),
               **replay.alignment(batch, flown, verdicts)}
        write_json_atomic(out / "variants" / f"{number:02d}_{name}.json", row)
        rows.append(row)
        landing = row["landing_time_minus_observed_s"]
        print(f"  {name:32s}{' (probe)' if probe else '':8s} landed {row['landed_share']:.3f}  "
              f"flew as said {row['flew_the_sentence_share']:.3f}  words inside {row['words_inside_share']:.3f}  "
              f"landing Δt p50 "
              f"{landing['p50'] if landing else float('nan'):+.0f} s  {time.perf_counter() - started:.0f}s", flush=True)
        del flown, verdicts

    write_json_atomic(out / "sensitivity.json", {
        "written_utc": utc_now(), "split": "train", "executor_spec_sha256": record["sha256"],
        "vocabulary_spec_sha256": spec.sha256, "params": asdict(params), "drawn": batch.drawn,
        "grid": {"heading_time_constant_s": HEADING_TIME_CONSTANTS_S, "bank_rate_deg_s": BANK_RATES_DEG_S,
                 "path_rate_factor": PATH_RATE_FACTORS, "delay_step_s": DELAY_STEP_S},
        "variants": rows, "git": git_state(), "elapsed_s": time.perf_counter() - started})
    print(f"→ {out / 'sensitivity.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
