"""The executor spec on disk (executor design §10, the E7 plan in §9): the parameters, with a sha, written
once — the vocabulary's rule for its own spec.

``spec.json`` carries the parameters (`ExecutorParams`), their sha, the vocabulary spec they were measured
against, the labeller that read the flights and the executor's source hash (`executor_source_files`: the code
that flies a sentence and measures the values — a spec measured by other code is refused at replay,
`require_current_executor`), and the git state; ``measurements.json`` the numbers behind every value. Nothing
here is ever overwritten.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.machinery
import importlib.util
import json
import sysconfig
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.io_utils import write_json_atomic

#: v4 (2026-09-24): the executor takes nothing beyond the vocabulary — r_turn, φ_cap and the word delays are gone
#: (the vocabulary's turn rates and bank limit; a word acts when said), p is the bank limit over the lead.
EXECUTOR_SPEC_SCHEMA = "ts-executor-spec-v4"
PACKAGE = Path(__file__).resolve().parent
#: Imported by the executor but not part of what decides a flown track or a value: the instruction language
#: (its own hash, the labeller's, is recorded in the spec and checked at replay) and the path and file helpers.
UNHASHED_IMPORTS = ("ts_transformer.instructions", "ts_transformer.io_utils", "ts_transformer.repo_layout")


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
    the package), and every module they import directly that is the repository's code rather than the environment's
    (labelled by its module name: the dynamics, the envelope, the approach-speed table, geokit…), except `UNHASHED_IMPORTS`. Labels, not paths, so the hash is the same from any checkout — geokit
    is installed editable from the main checkout, outside a worktree. Direct imports only, as the labeller's hash."""
    own = sorted(path.resolve() for path in PACKAGE.glob("*.py") if path.name != "spec.py")
    external: dict[str, Path] = {}
    for path in own:
        for name, file in _imported_modules(path).items():
            if (file.parent != PACKAGE.resolve() and not _installed(file)
                    and not any(name == unhashed or name.startswith(unhashed + ".") for unhashed in UNHASHED_IMPORTS)):
                external[name] = file
    return ([(f"{PACKAGE.name}/{path.name}", path) for path in own]
            + [(name, external[name]) for name in sorted(external)])


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
    """sha256 over `executor_source_files` (label and bytes, in order)."""
    digest = hashlib.sha256()
    for label, path in executor_source_files():
        digest.update(label.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
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
