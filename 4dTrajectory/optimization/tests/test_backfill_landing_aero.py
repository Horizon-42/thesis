"""The landing_aero backfill writes the block build_scenario would have, and nothing else."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_OPT_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _OPT_DIR.parents[1]
for entry in (_OPT_DIR, _REPO_ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import backfill_landing_aero as bl  # noqa: E402
from aircraft.aero_params import aero_params_for_aircraft  # noqa: E402
from flight_scenarios.scenario import aircraft_for_code  # noqa: E402


def _record(subject: str, *, typecode: str | None = "A320", landing_aero=None) -> dict:
    source = {"subject": subject, "id": "X", "flight_key": "X_05L_abc_20260812T000000Z"}
    if typecode is not None:
        source["dynamics_typecode"] = typecode
    if landing_aero is not None:
        source["landing_aero"] = landing_aero
    return {"source": source, "states": [], "controls": []}


def _batch(root: Path, records: list[dict]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, record in enumerate(records):
        name = f"r{index}_eval.json"
        (root / name).write_text(json.dumps(record), encoding="utf-8")
        rows.append({"eval_file": name})
    (root / "summary.json").write_text(json.dumps({"results": rows}), encoding="utf-8")
    return root


def test_dry_run_counts_and_apply_writes_the_producer_block(tmp_path):
    batch = _batch(tmp_path / "runway", [
        _record("optimized"),
        _record("optimized", landing_aero={"wing_area_m2": 1.0, "cl_max_landing": 1.0}),
    ])
    _batch(tmp_path / "ts_pred", [_record("predicted")])
    before = (batch / "r0_eval.json").read_text()

    counts = bl.BackfillCounts()
    for manifest in sorted(tmp_path.rglob("summary.json")):
        bl.backfill_batch(manifest.parent, apply=False, counts=counts)
    assert (counts.batches, counts.skipped_batches, counts.already, counts.backfilled) == (1, 1, 1, 1)
    assert (batch / "r0_eval.json").read_text() == before        # dry run touches nothing

    assert bl.main(["--root", str(tmp_path), "--apply"]) == 0
    written = json.loads((batch / "r0_eval.json").read_text())["source"]["landing_aero"]
    aero = aero_params_for_aircraft(aircraft_for_code("A320"))
    assert written == {"wing_area_m2": aero.S, "cl_max_landing": aero.Cl_max}
    # The record that already carried a block keeps it; a second pass is a no-op.
    kept = json.loads((batch / "r1_eval.json").read_text())["source"]["landing_aero"]
    assert kept == {"wing_area_m2": 1.0, "cl_max_landing": 1.0}
    again = bl.BackfillCounts()
    bl.backfill_batch(batch, apply=False, counts=again)
    assert (again.already, again.backfilled) == (2, 0)


def test_a_record_without_a_dynamics_typecode_is_refused_loudly(tmp_path):
    batch = _batch(tmp_path, [_record("optimized", typecode=None)])
    with pytest.raises(ValueError, match="dynamics_typecode"):
        bl.backfill_batch(batch, apply=False, counts=bl.BackfillCounts())
