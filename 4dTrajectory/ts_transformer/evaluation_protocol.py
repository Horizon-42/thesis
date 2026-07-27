"""One-way release gate for the checkpoint's outer-test flights.

Development code may evaluate train and validation repeatedly.  Test evaluation is different:
it must be explicitly frozen against one checkpoint, and every claimed flight becomes
irreversibly recorded before its data are loaded.  A failed/partial evaluation therefore
cannot silently be repeated and treated as a fresh blind test.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from dataset import require_matching_data_provenance

TEST_RELEASE_NAME = "test_release.json"
TEST_RELEASE_SCHEMA = "ts-test-release-v1-checkpoint-bound-one-shot"
TEST_RELEASE_PROTOCOL_FIELD = "test_release_protocol"


class TestReleaseError(RuntimeError):
    """The requested operation would violate the frozen outer-test protocol."""


def test_release_path(checkpoint: str | Path) -> Path:
    """The sole release ledger associated with ``checkpoint``."""
    return Path(checkpoint).resolve().with_name(TEST_RELEASE_NAME)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _flight_keys(payload: dict[str, Any]) -> list[str]:
    split = payload.get("split")
    keys = split.get("test") if isinstance(split, dict) else None
    if (
        not isinstance(keys, list)
        or not keys
        or any(not isinstance(key, str) or not key for key in keys)
        or len(set(keys)) != len(keys)
    ):
        raise TestReleaseError("checkpoint has no valid unique outer-test flight identities")
    return keys


def _write_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


@contextmanager
def _locked(checkpoint: Path) -> Iterator[None]:
    """Serialize claims from parallel per-airport publishers."""
    lock_path = test_release_path(checkpoint).with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _read_bound_release(
    checkpoint: Path,
    checkpoint_payload: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    release_path = test_release_path(checkpoint)
    if not release_path.is_file():
        raise TestReleaseError(
            "outer-test is sealed; run the explicit freeze-test command after all model "
            "and hyperparameter decisions are final"
        )
    try:
        release = json.loads(release_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TestReleaseError(f"cannot audit test release {release_path}: {exc}") from exc
    test_keys = _flight_keys(checkpoint_payload)
    stored_provenance = checkpoint_payload.get("data_provenance")
    if not isinstance(release, dict) or release.get("schema_version") != TEST_RELEASE_SCHEMA:
        raise TestReleaseError("test release has an unsupported schema")
    if release.get("checkpoint_sha256") != _file_sha256(checkpoint):
        raise TestReleaseError("test release is bound to a different checkpoint")
    if release.get("test_split_sha256") != _json_sha256(sorted(test_keys)):
        raise TestReleaseError("test release is bound to a different outer-test split")
    if release.get("data_provenance_sha256") != _json_sha256(stored_provenance):
        raise TestReleaseError("test release is bound to different arrival data")
    if not isinstance(release.get("claims"), list):
        raise TestReleaseError("test release has no auditable claims list")
    return release_path, release


def create_test_release(
    checkpoint: str | Path,
    checkpoint_payload: dict[str, Any],
    current_provenance: dict[str, Any],
) -> Path:
    """Freeze one checkpoint/data/split identity; refuse replacement or reuse."""
    checkpoint_path = Path(checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise TestReleaseError(f"checkpoint does not exist: {checkpoint_path}")
    if checkpoint_payload.get(TEST_RELEASE_PROTOCOL_FIELD) != TEST_RELEASE_SCHEMA:
        raise TestReleaseError(
            "checkpoint predates the sealed-test protocol; retrain it before claiming a "
            "new blind test release"
        )
    try:
        require_matching_data_provenance(
            checkpoint_payload, current_provenance, allow_subset=False
        )
    except ValueError as exc:
        raise TestReleaseError(str(exc)) from exc
    test_keys = _flight_keys(checkpoint_payload)
    release_path = test_release_path(checkpoint_path)
    with _locked(checkpoint_path):
        if release_path.exists():
            raise TestReleaseError(
                f"outer-test release already exists at {release_path}; refusing to reset "
                "its exposure history"
            )
        _write_atomic(release_path, {
            "schema_version": TEST_RELEASE_SCHEMA,
            "status": "frozen",
            "created_at": _utc_now(),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _file_sha256(checkpoint_path),
            "data_provenance_sha256": _json_sha256(
                checkpoint_payload.get("data_provenance")
            ),
            "test_split_sha256": _json_sha256(sorted(test_keys)),
            "test_flights": len(test_keys),
            "claims": [],
        })
    return release_path


def begin_test_evaluation(
    checkpoint: str | Path,
    checkpoint_payload: dict[str, Any],
    current_provenance: dict[str, Any],
    selected_test_keys: Sequence[str],
    *,
    output_dir: str | Path,
) -> str:
    """Irreversibly claim test flights before loading or predicting them."""
    checkpoint_path = Path(checkpoint).resolve()
    try:
        require_matching_data_provenance(
            checkpoint_payload, current_provenance, allow_subset=True
        )
    except ValueError as exc:
        raise TestReleaseError(str(exc)) from exc
    selected = list(selected_test_keys)
    frozen = set(_flight_keys(checkpoint_payload))
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(key not in frozen for key in selected)
    ):
        raise TestReleaseError("test claim is not a unique subset of the frozen test split")

    with _locked(checkpoint_path):
        release_path, release = _read_bound_release(checkpoint_path, checkpoint_payload)
        previously_exposed = {
            key
            for claim in release["claims"]
            for key in claim.get("flight_keys", [])
        }
        overlap = sorted(previously_exposed.intersection(selected))
        if overlap:
            raise TestReleaseError(
                f"{len(overlap)} outer-test flight(s) were already exposed; refusing "
                f"repeated evaluation (first: {overlap[0]!r})"
            )
        claim_id = f"claim-{len(release['claims']) + 1:04d}"
        release["claims"].append({
            "claim_id": claim_id,
            "status": "started",
            "started_at": _utc_now(),
            "output_dir": str(Path(output_dir).resolve()),
            "flight_count": len(selected),
            "flight_keys_sha256": _json_sha256(sorted(selected)),
            # Keeping identities makes overlap detection auditable across airport subsets.
            "flight_keys": sorted(selected),
        })
        release["status"] = "evaluating"
        _write_atomic(release_path, release)
    return claim_id


def complete_test_evaluation(checkpoint: str | Path, claim_id: str) -> None:
    """Mark a successful claim complete without making failed claims reusable."""
    checkpoint_path = Path(checkpoint).resolve()
    with _locked(checkpoint_path):
        release_path = test_release_path(checkpoint_path)
        try:
            release = json.loads(release_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TestReleaseError(f"cannot audit test release {release_path}: {exc}") from exc
        claims = release.get("claims") if isinstance(release, dict) else None
        claim = next(
            (
                item for item in claims or []
                if isinstance(item, dict) and item.get("claim_id") == claim_id
            ),
            None,
        )
        if claim is None or claim.get("status") != "started":
            raise TestReleaseError(f"test claim {claim_id!r} is absent or not active")
        claim["status"] = "complete"
        claim["completed_at"] = _utc_now()
        completed = {
            key
            for item in claims
            if item.get("status") == "complete"
            for key in item.get("flight_keys", [])
        }
        if len(completed) == release.get("test_flights"):
            release["status"] = "complete"
            release["completed_at"] = _utc_now()
        elif any(item.get("status") == "started" for item in claims):
            release["status"] = "evaluating"
        else:
            release["status"] = "partially_evaluated"
        _write_atomic(release_path, release)
