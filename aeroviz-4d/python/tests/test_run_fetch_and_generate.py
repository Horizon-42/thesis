from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_fetch_and_generate import _airport_output_dir, _default_czml_output


def test_airport_output_dir_is_airport_scoped(tmp_path):
    output_dir = _airport_output_dir(tmp_path, "CYVR")
    assert output_dir == tmp_path / "cyvr"


def test_default_czml_output_is_airport_scoped(tmp_path):
    output_path = _default_czml_output(tmp_path, "krdu")
    assert output_path == tmp_path / "public" / "data" / "airports" / "KRDU" / "trajectories.czml"
