"""Small file/hash helpers shared across the package's writers and runners.

They were copied byte-for-byte into five modules (2026-09-07 package audit); one
definition each. Deliberately torch-free: `experiments/pipeline.py` is import-light and
`experiment_index` runs before any model is built.
"""

from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: dict[str, Any], *, allow_nan: bool = True) -> None:
    """Write via a sibling temp file and rename, so a reader never sees a partial file.

    ``allow_nan=False`` refuses a payload with NaN/inf instead of writing invalid JSON —
    cross-validation results use it because a NaN fold must fail loudly, not serialise.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=allow_nan), encoding="utf-8")
    temporary.replace(path)


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    """`write_json_atomic` for bytes as they are (a file copied byte for byte)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_DOCUMENTED = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def logic(source: str) -> str:
    """A module's logic as text: its syntax tree with the docstring of the module, every class and every function
    removed (a body left empty holds ``pass``), written back by `ast.unparse` — no comment, docstring or layout in it.
    The text is the running Python's, so another Python version may write another text."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (isinstance(node, _DOCUMENTED) and node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str)):
            node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def logic_sha256(files: Sequence[tuple[str, Path]]) -> str:
    """sha256 over ``(label, file)`` pairs in order: each label and the file's `logic` — what names a piece of code by
    what it does, not by its comments (the executor's and the labeller's conformance records are named by it)."""
    digest = hashlib.sha256()
    for label, path in files:
        digest.update(label.encode("utf-8") + b"\0" + logic(path.read_text(encoding="utf-8")).encode("utf-8") + b"\0")
    return digest.hexdigest()
