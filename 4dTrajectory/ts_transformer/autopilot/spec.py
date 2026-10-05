"""The executor spec on disk (executor design §10, §12): the parameters, with a sha, written once — the vocabulary's rule
for its own spec.

``spec.json`` carries the parameters (`ExecutorParams`), their sha, the vocabulary spec they were derived from and the
git state; ``measurements.json`` where every value comes from. Nothing here is ever overwritten.

THE EXECUTOR IS CHECKED BY WHAT IT FLIES, NEVER BY ITS SOURCE (executor design §12.2–§12.3, the user 2026-10-01; D73):
beside the spec, ``conformance/`` holds its reference tracks — labelled flights flown by the code that wrote the spec, in
the same run — and every process that opens the spec flies them again first, in every way the executor flies
(`autopilot.conformance.require_conforming_executor`, through `replay.open_executor`). There is no passed record and no
digest of code: a code change whose tracks stay within the bounds opens everything, and one that moves a track is
refused by name where it is used.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from ts_transformer.autopilot.params import FORMAL_START_RULES, ExecutorParams
from ts_transformer.io_utils import write_json_atomic

#: v8 (vocabulary §12.1 A19, A20): one word clock — a sentence is said on its own rows, no ``word_clock``
#: parameter (D57) — and a level word flown at T + E MSL, E the airport elevation (D58); the laws of §5 otherwise as v7
#: (no capture, no landing aim, no glidepath floor, no parameter of the decision-altitude check, D38).
#: v9 (A29, D73): no digest of the executor's or the labeller's code beside the parameters.
#: v10 (A32): the start rule (D77) among the parameters; the flight ends where the judge ends it (D79), the bank's limits
#: from the first cycle (D84), the frame at the airport reference (D81).
EXECUTOR_SPEC_SCHEMA = "ts-executor-spec-v10"


def params_to_dict(params: ExecutorParams) -> dict[str, Any]:
    return asdict(params)


def params_from_dict(data: dict[str, Any]) -> ExecutorParams:
    names = {field.name for field in fields(ExecutorParams)}
    if set(data) != names:
        raise ValueError(f"an executor spec needs exactly {sorted(names)}; missing {sorted(names - set(data))}, "
                         f"extra {sorted(set(data) - names)}")
    return ExecutorParams(**data)


def params_sha256(params: ExecutorParams) -> str:
    canonical = json.dumps(params_to_dict(params), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()



def _fresh(path: Path) -> Path:
    if path.exists():
        raise FileExistsError(f"{path} exists; an executor spec is never overwritten")
    return path


def write_spec(directory: Path, params: ExecutorParams, vocabulary_sha256: str, measurements: dict[str, Any],
               source: dict[str, Any]) -> None:
    if params.start_rule not in FORMAL_START_RULES:
        raise ValueError(f"start rule {params.start_rule!r} is none of {FORMAL_START_RULES}: no spec holds another (D77)")
    directory.mkdir(parents=True, exist_ok=True)
    sha = params_sha256(params)
    write_json_atomic(_fresh(directory / "spec.json"), {
        "schema": EXECUTOR_SPEC_SCHEMA, "sha256": sha, "params": params_to_dict(params),
        "vocabulary_spec_sha256": vocabulary_sha256, "source": source})
    write_json_atomic(_fresh(directory / "measurements.json"), {"spec_sha256": sha, **measurements})


def load_spec(directory: Path) -> tuple[ExecutorParams, dict[str, Any]]:
    """The params and the whole record; refused unless the file is this format and its sha is its params'."""
    record = json.loads((directory / "spec.json").read_text(encoding="utf-8"))
    if record["schema"] != EXECUTOR_SPEC_SCHEMA:
        raise ValueError(f"{directory / 'spec.json'} is not a {EXECUTOR_SPEC_SCHEMA} file")
    params = params_from_dict(record["params"])
    if params_sha256(params) != record["sha256"]:
        raise ValueError(f"{directory / 'spec.json'}: the params do not hash to the recorded sha")
    if params.start_rule not in FORMAL_START_RULES:
        raise ValueError(f"{directory / 'spec.json'}: start rule {params.start_rule!r} is none of {FORMAL_START_RULES} "
                         f"(D77: the centred fit is A33's comparison only)")
    return params, record


#: Beside a spec: its reference tracks, written with it (`autopilot.conformance`).
CONFORMANCE_DIRECTORY = "conformance"
