from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).parents[3]


def test_generation_helper_passes_chart_root_parent_directory() -> None:
    script = (REPO_ROOT / "generate_aeroviz_airport_procedure_data.sh").read_text(encoding="utf-8")

    assert '--charts-root "$SCRIPT_DIR/data/RNAV_CHARTS"' in script
    assert '--charts-root "$SCRIPT_DIR/data/RNAV_CHARTS/$ICAO"' not in script
