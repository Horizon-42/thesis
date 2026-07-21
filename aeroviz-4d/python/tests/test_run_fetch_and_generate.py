import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUNNER_PATH = ROOT / "run_asd-b_fetch_and_generate.py"
spec = importlib.util.spec_from_file_location("run_asdb_fetch_and_generate", RUNNER_PATH)
assert spec is not None
assert spec.loader is not None
runner_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner_module)

parse_args = runner_module.parse_args


def test_frontend_data_is_parsed_locally_while_harvest_options_are_forwarded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frontend_data = tmp_path / "public-data"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_asd-b_fetch_and_generate.py",
            "--airport",
            "KRDU",
            "--frontend-data",
            str(frontend_data),
            "--count",
            "25",
        ],
    )

    args, passthrough = parse_args()

    assert args.airport == "KRDU"
    assert args.frontend_data == str(frontend_data)
    assert passthrough == ["--count", "25"]
