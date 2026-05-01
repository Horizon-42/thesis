from pathlib import Path
import importlib.util
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUNNER_PATH = ROOT / "run_asd-b_fetch_and_generate.py"
spec = importlib.util.spec_from_file_location("run_asdb_fetch_and_generate", RUNNER_PATH)
assert spec is not None
assert spec.loader is not None
runner_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner_module)

_airport_output_dir = runner_module._airport_output_dir
_default_czml_output = runner_module._default_czml_output


def test_airport_output_dir_is_airport_scoped(tmp_path):
    output_dir = _airport_output_dir(tmp_path, "CYVR")
    assert output_dir == tmp_path / "cyvr"


def test_default_czml_output_is_airport_scoped(tmp_path):
    output_path = _default_czml_output(tmp_path, "krdu")
    assert output_path == tmp_path / "public" / "data" / "airports" / "KRDU" / "trajectories.czml"
