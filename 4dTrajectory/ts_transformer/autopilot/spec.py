"""The executor spec on disk (executor design §10, the E7 plan in §9): the parameters, with a sha, written
once — the vocabulary's rule for its own spec.

``spec.json`` carries the parameters (`ExecutorParams`), their sha, the vocabulary spec they were measured
against and the executor's source hash (`EXECUTOR_MODULES`: the laws, the loop and the dynamics seam — a
spec measured by other code is refused at replay, `require_current_executor`), and the git state;
``measurements.json`` the numbers behind every value. Nothing here is ever overwritten.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.autopilot.sentence import Delays
from ts_transformer.io_utils import write_json_atomic

EXECUTOR_SPEC_SCHEMA = "ts-executor-spec-v1"
#: What decides a flown track and the values it is flown with: every module of the executor except the
#: spec file itself and the judge (how a flight is graded is not how it is flown).
EXECUTOR_MODULES = ("__init__.py", "frame.py", "sentence.py", "flights.py", "plant.py", "inverse.py", "params.py",
                    "lateral.py", "vertical.py", "speed.py", "executor.py", "replay.py", "observe.py", "measure.py")


def params_to_dict(params: ExecutorParams) -> dict[str, Any]:
    return asdict(params)


def params_from_dict(data: dict[str, Any]) -> ExecutorParams:
    names = {field.name for field in fields(ExecutorParams)}
    if set(data) != names:
        raise ValueError(f"an executor spec needs exactly {sorted(names)}; missing {sorted(names - set(data))}, "
                         f"extra {sorted(set(data) - names)}")
    return ExecutorParams(**{**data, "delays": Delays(**data["delays"])})


def params_sha256(params: ExecutorParams) -> str:
    canonical = json.dumps(params_to_dict(params), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def executor_source_sha256() -> str:
    package = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in sorted(EXECUTOR_MODULES):
        digest.update(name.encode("utf-8") + b"\0" + (package / name).read_bytes() + b"\0")
    return digest.hexdigest()


def _fresh(path: Path) -> Path:
    if path.exists():
        raise FileExistsError(f"{path} exists; an executor spec is never overwritten")
    return path


def write_spec(directory: Path, params: ExecutorParams, vocabulary_sha256: str, measurements: dict[str, Any],
               source: dict[str, Any]) -> None:
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
    return params, record


def require_current_executor(record: dict[str, Any]) -> None:
    """A spec measured by other executor code is refused: its values were fitted to other laws."""
    if record["source"]["executor_source_sha256"] != executor_source_sha256():
        raise ValueError("the executor spec was measured by other executor code "
                         f"({record['source']['executor_source_sha256'][:12]}, now {executor_source_sha256()[:12]})")
