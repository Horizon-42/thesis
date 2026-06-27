"""Drift guard: the frontend's generated geoConstants.json must stay in sync with
geokit.constants. If this fails, regenerate it:

    python geokit/scripts/export_constants_json.py
"""

import importlib.util
import json
from pathlib import Path

_HERE = Path(__file__).resolve()
_GEOKIT_ROOT = _HERE.parents[1]
_REPO_ROOT = _HERE.parents[2]
_SCRIPT = _GEOKIT_ROOT / "scripts" / "export_constants_json.py"
_JSON = _REPO_ROOT / "aeroviz-4d" / "src" / "generated" / "geoConstants.json"


def _load_export_module():
    spec = importlib.util.spec_from_file_location("export_constants_json", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_geoconstants_json_in_sync_with_geokit():
    expected = _load_export_module().build_payload()
    actual = json.loads(_JSON.read_text())
    assert actual == expected, (
        "aeroviz-4d/src/generated/geoConstants.json is stale relative to geokit.constants. "
        "Regenerate it: python geokit/scripts/export_constants_json.py"
    )
