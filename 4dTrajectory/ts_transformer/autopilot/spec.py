"""The executor spec on disk (executor design §10, §12): the parameters, with a sha, written once — the vocabulary's rule
for its own spec.

``spec.json`` carries the parameters (`ExecutorParams`), their sha, the vocabulary spec they were derived from, the
labeller that reads the flights, the logic hash of the executor code that measured it (`executor_source_sha256`: where
the spec came from) and the git state; ``measurements.json`` where every value comes from. Nothing here is ever
overwritten.

THE EXECUTOR IS CHECKED BY WHAT IT FLIES, NOT BY ITS SOURCE (executor design §12.2–§12.3, the user 2026-10-01): beside
the spec, ``conformance/`` holds its reference tracks — labelled flights flown by the spec's own code — and a
``passed-<code>.json`` for every executor code that flew them again within the bounds in every way the executor flies
(`autopilot.conformance`, runner `executor_conformance`). `require_conforming_executor` opens a spec only for executor
code with such a record: a code change whose tracks stay within the bounds needs one check, and no spec, training or
readout is redone. The code is named by `executor_source_sha256` — the logic of `autopilot/` and the repository modules
it imports directly (`logic`: docstrings and comments are free) — which here only names the record ("this code was
checked"); the normalised text is the running Python's, so another Python version is another code and is checked again.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.machinery
import importlib.util
import json
import platform
import sysconfig
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.io_utils import logic, logic_sha256, write_json_atomic

#: v6 (2026-09-27, executor v11): the source hash is over the code's logic (`logic`), no longer its bytes
#: (v5, 2026-09-24: nothing measured from data is left — the speed changes' pace is the vocabulary's, the landing
#: crosses at the runway's published TCH).
EXECUTOR_SPEC_SCHEMA = "ts-executor-spec-v6"
PACKAGE = Path(__file__).resolve().parent
#: Imported by the executor but not part of what decides a flown track or a value: the instruction language
#: (its own hash, the labeller's, is recorded in the spec and checked at replay), the path and file helpers, and the
#: evaluation CLI (it only names the runway data's files; a replay records the crossing heights it flew to).
UNHASHED_IMPORTS = ("ts_transformer.instructions", "ts_transformer.io_utils", "ts_transformer.repo_layout",
                    "evaluation.cli")
#: What the executor's integration reaches beyond its direct imports — the plant's rollout down to the right-hand side
#: (`Plant.step` → `outputs.dynamics.rollout` → `backends` → `aerodynamic_model`'s scaled transport-chart RK4) — named
#: whoever imports them: the code a passed record names must hold every module whose change moves a flown track (the
#: single-flight executor once pinned these itself). Not every transitive import: that reaches the harvest, evaluation
#: and training (100 modules), and any change there would ask for a check that cannot change an answer.
REACHED_MODULES = (
    "ts_transformer.outputs.constraints.speed_floor",
    "ts_transformer.outputs.dynamics.rollout",
    "ts_transformer.outputs.dynamics.backends",
    "aerodynamic_model.torch_piecewise_rollout",
    "aerodynamic_model.torch_scaled_transport_chart_dynamics",
    "aerodynamic_model.torch_transport_chart_dynamics",
    "aerodynamic_model.torch_dynamics",
)


def _imported_modules(path: Path) -> dict[str, Path]:
    """The modules ``path`` imports, by name, with their source files (absolute imports; for ``from package import
    name``, the submodule ``package.name`` when there is one)."""
    modules: dict[str, Path] = {}

    def add(name: str) -> importlib.machinery.ModuleSpec | None:
        found = importlib.util.find_spec(name)
        if found is not None and found.origin not in (None, "built-in", "frozen"):
            modules[name] = Path(found.origin).resolve()
        return found

    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            base = add(node.module)
            if base is not None and base.submodule_search_locations is not None:
                for alias in node.names:
                    add(f"{node.module}.{alias.name}")
    return modules


def _installed(file: Path) -> bool:
    """A module of the environment itself — the standard library or site-packages — not of this repository's code."""
    places = {Path(sysconfig.get_paths()[key]).resolve() for key in ("stdlib", "platstdlib", "purelib", "platlib")}
    return any(file.is_relative_to(place) for place in places)


def executor_source_files() -> list[tuple[str, Path]]:
    """What the hash covers, as ``(label, file)``: every module of the package except this file (labelled by its path in
    the package), every module they import directly that is the repository's code rather than the environment's
    (labelled by its module name: the dynamics, the envelope, the approach-speed table, geokit…), except `UNHASHED_IMPORTS`,
    and `REACHED_MODULES`; each file once. Labels, not paths, so the hash is the same from any checkout — geokit is
    installed editable from the main checkout, outside a worktree."""
    own = sorted(path.resolve() for path in PACKAGE.glob("*.py") if path.name != "spec.py")
    external: dict[str, Path] = {}
    for path in own:
        for name, file in _imported_modules(path).items():
            if (file.parent != PACKAGE.resolve() and not _installed(file)
                    and not any(name == unhashed or name.startswith(unhashed + ".") for unhashed in UNHASHED_IMPORTS)):
                external[name] = file
    for name in REACHED_MODULES:
        external[name] = Path(importlib.util.find_spec(name).origin).resolve()
    held = set()
    named = []
    for name in sorted(external):
        if external[name] not in held:
            held.add(external[name])
            named.append((name, external[name]))
    return [(f"{PACKAGE.name}/{path.name}", path) for path in own] + named


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


def executor_source_sha256() -> str:
    """sha256 over `executor_source_files` (label and `logic`, in order: `io_utils.logic_sha256`)."""
    return logic_sha256(executor_source_files())


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


#: Beside a spec: its reference tracks and the records of the code that flew them within the bounds.
CONFORMANCE_DIRECTORY = "conformance"
PASSED_SCHEMA = "ts-executor-conformance-passed-v1"


def reference_sha256(directory: Path) -> str:
    """sha256 over a spec's reference tracks (``reference.json``, ``reference.npz``: name and bytes)."""
    digest = hashlib.sha256()
    for name in ("reference.json", "reference.npz"):
        digest.update(name.encode("utf-8") + b"\0" + (directory / name).read_bytes() + b"\0")
    return digest.hexdigest()


def passed_path(spec_dir: Path, code_sha256: str) -> Path:
    return spec_dir / CONFORMANCE_DIRECTORY / f"passed-{code_sha256[:12]}.json"


def require_conforming_executor(spec_dir: Path) -> None:
    """Refused unless the executor code on disk has flown ``spec_dir``'s reference tracks within the bounds in every way
    it flies: a passed record for this code against this reference (module docstring)."""
    code = executor_source_sha256()
    path = passed_path(spec_dir, code)
    command = (f"python run_ts.py executor_conformance --executor {spec_dir} --instructions <the artefact it flies>"
               f" (Python {platform.python_version()})")
    if not path.exists():
        raise ValueError(f"the executor code on disk ({code[:12]}) has not been checked against {spec_dir.name}'s "
                         f"reference tracks: {command}")
    record = json.loads(path.read_text(encoding="utf-8"))
    if (record["schema"] != PASSED_SCHEMA or record["executor_source_sha256"] != code
            or record["reference_sha256"] != reference_sha256(spec_dir / CONFORMANCE_DIRECTORY)):
        raise ValueError(f"{path} is not a passed record of this code against {spec_dir.name}'s reference: {command}")
