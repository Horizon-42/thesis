"""The frontend's mirror of the scenario catalog (design §10.6) against the census writer, field by field and type by type.

The viewer reads `aeroviz-4d/src/data/__tests__/fixtures/trafficScenarioCatalog.json` in its own tests (the guard of
`fetchTrafficScenarios`, the panel's tables). That file is a MIRROR of what `traffic_scenarios.py` writes. This test runs the
writer's `main` on two arrivals (its loaders and its worker replaced by fakes, every path under tmp) and compares the catalog it
writes with the mirror: the same fields, each with the same JSON type. The groups are compared separately — the document's
own fields, `config`, `counts`, the M1 rows, the M2 rows of each block length. A null in the mirror is allowed only for the
fields the writer can leave null: `callsign`, `type` (a flight the roster has no callsign or airframe of), `category` (a type
with no CWT category) and `limit` (a whole roster). The judge's row (`judge_record`) cannot be run without a real window: its value types are read from the constructors
in its return literal and the fake worker returns exactly those.
"""

import ast
import inspect
import json
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

import traffic_scenarios as ts
from traffic_job_files import BLOCK_LENGTHS_S, CATALOG_SCHEMA, catalog_path

MIRROR = Path(__file__).resolve().parents[4] / "aeroviz-4d" / "src" / "data" / "__tests__" / "fixtures" / "trafficScenarioCatalog.json"
#: The fields whose value may be null in the catalog, and so in the mirror (a field not listed is never null).
NULLABLE = {"callsign", "type", "category", "limit"}
#: The constructors `judge_record`'s return literal uses, and the JSON type each gives.
JUDGE_CONSTRUCTORS = {"len": int, "sorted": list, "min": float}


def json_type(value) -> type:
    return type(value)


def assert_same_types(written: dict, mirror: dict, where: str) -> None:
    """The mirror has the writer's fields, each of the writer's JSON type (null only where the writer may leave one)."""
    assert set(mirror) == set(written), f"{where}: fields differ"
    for field, value in mirror.items():
        if value is None:
            assert field in NULLABLE, f"{where}.{field} is null in the mirror, and the writer never leaves it null"
        else:
            assert json_type(value) is json_type(written[field]), (
                f"{where}.{field}: the mirror has {type(value).__name__}, the writer {type(written[field]).__name__}")


def judge_row_types() -> dict[str, type]:
    """The JSON type of each field of `judge_record`'s return literal, from the call that builds it."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(ts.judge_record)))
    literal = [node for node in ast.walk(tree) if isinstance(node, ast.Return)][-1].value
    assert isinstance(literal, ast.Dict)
    types = {}
    for key, value in zip(literal.keys, literal.values):
        if isinstance(value, ast.Attribute):                      # window.flight_key, window.category (a string or null)
            types[key.value] = str
            continue
        assert isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id in JUDGE_CONSTRUCTORS, (
            f"judge_record's field {key.value!r} is built by something this test does not know")
        types[key.value] = JUDGE_CONSTRUCTORS[value.func.id]
    return types


class _InProcessPool:
    """A stand-in for the process pool: the worker runs here, in order."""

    def __init__(self, max_workers: int) -> None:
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None

    def map(self, fn, iterable, chunksize=1):
        return map(fn, iterable)


@pytest.fixture(scope="module")
def written(tmp_path_factory) -> dict:
    """The catalog `main` writes for two arrivals, read back from the file (JSON types), under a tmp outputs root."""
    root = tmp_path_factory.mktemp("census")
    flights = [
        {"key": "AAL1_05L_a_1", "callsign": "AAL1", "runway": "05L", "entry_time_utc": "2026-05-01T10:00:00.000Z",
         "landing_time_utc": "2026-05-01T10:05:00Z", "typecode": "A320"},
        {"key": "DAL2_05R_b_2", "callsign": "DAL2", "runway": "05R", "entry_time_utc": "2026-05-01T10:10:00.000Z",
         "landing_time_utc": "2026-05-01T10:20:00Z", "typecode": "B738"},
    ]
    traffic = SimpleNamespace(flight=lambda key: SimpleNamespace(
        typecode=next(f["typecode"] for f in flights if f["key"] == key), start_utc_s=0.0, end_utc_s=300.0))
    sample = {int: 3, list: ["in_trail"], float: 0.8}
    row_types = judge_row_types()                                # before `judge_record` is replaced

    def judge(payload):                                          # payload[0] is the arrival's key (see `window_scenario` below)
        return {field: payload[0] if kind is str else sample[kind] for field, kind in row_types.items()}

    patch = pytest.MonkeyPatch()
    patch.setattr(ts, "load_model_arrivals", lambda manifest: list(flights))
    patch.setattr(ts, "traffic_from_arrivals", lambda arrivals: traffic)
    patch.setattr(ts, "flight_key", lambda flight, _index: flight["key"])
    patch.setattr(ts, "build_scenario", lambda flight, **_kw: flight["key"])
    patch.setattr(ts, "window_scenario", lambda scenario, _traffic, _horizon: scenario)
    patch.setattr(ts, "judge_record", judge)
    patch.setattr(ts, "ProcessPoolExecutor", _InProcessPool)
    patch.setattr(ts, "limit_solver_threads", lambda: None)      # it sets this process's environment
    patch.setattr(sys, "argv", ["traffic_scenarios.py", "--airport", "krdu", "--jobs", "1", "--limit", "2",
                                "--harvest-root", str(root / "harvest"), "--outputs-root", str(root / "outputs")])
    try:
        ts.main()
    finally:
        patch.undo()
    return json.loads(catalog_path(root / "outputs", "KRDU").read_text(encoding="utf-8"))


def _mirror() -> dict:
    return json.loads(MIRROR.read_text(encoding="utf-8"))


def test_the_fake_worker_returns_what_judge_record_returns_by_construction():
    assert judge_row_types() == {"flightKey": str, "category": str, "lossInstants": int, "kinds": list, "tightest": float,
                                 "recordedAircraft": int}


def test_the_writers_catalog_is_one_the_mirror_describes_group_by_group(written):
    mirror = _mirror()
    own = lambda catalog: {key: value for key, value in catalog.items() if key not in ("config", "counts", "m1", "m2")}
    assert_same_types(own(written), own(mirror), "document")
    assert_same_types(written["config"], mirror["config"], "config")
    assert_same_types(written["counts"], mirror["counts"], "counts")
    assert mirror["schema"] == written["schema"] == CATALOG_SCHEMA


def test_the_mirrors_m1_rows_have_the_writers_fields_and_types(written):
    assert written["m1"], "the fake census must leave arrivals with a loss to compare"
    assert any(row["callsign"] is None for row in _mirror()["m1"])         # the mirror shows the nulls the panel must read
    for n, row in enumerate(_mirror()["m1"]):
        assert_same_types(written["m1"][0], row, f"m1[{n}]")


def test_the_mirrors_m2_blocks_have_the_writers_fields_and_types_for_every_block_length(written):
    mirror = _mirror()
    assert set(mirror["m2"]) == set(written["m2"]) == {str(length) for length in BLOCK_LENGTHS_S}
    for length, blocks in mirror["m2"].items():
        assert blocks and written["m2"][length]
        for n, block in enumerate(blocks):
            assert_same_types(written["m2"][length][0], block, f"m2[{length}][{n}]")


def test_a_null_is_allowed_only_where_the_writer_can_leave_one():
    written = {"callsign": "A", "tightest": 0.5}
    assert_same_types(written, {"callsign": None, "tightest": 0.5}, "row")
    with pytest.raises(AssertionError, match="never leaves it null"):
        assert_same_types(written, {"callsign": "A", "tightest": None}, "row")
    with pytest.raises(AssertionError, match="the mirror has int, the writer float"):
        assert_same_types(written, {"callsign": "A", "tightest": 1}, "row")
    with pytest.raises(AssertionError, match="fields differ"):
        assert_same_types(written, {"callsign": "A"}, "row")
