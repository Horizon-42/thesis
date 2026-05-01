from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).parents[3]
PIPELINE_SCRIPT = REPO_ROOT / "run_asd-b_fetch_and_generate.py"


def load_pipeline_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_asd_b_fetch_and_generate", PIPELINE_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_root_pipeline_can_regenerate_procedure_assets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pipeline = load_pipeline_module()
    aeroviz_root = tmp_path / "aeroviz-4d"
    python_dir = aeroviz_root / "python"
    python_dir.mkdir(parents=True)
    (python_dir / "generate_czml.py").write_text("print('fake czml')\n", encoding="utf-8")
    (python_dir / "preprocess_procedures.py").write_text("print('fake procedures')\n", encoding="utf-8")

    input_json = tmp_path / "krdu_czml_input_fixture.json"
    input_json.write_text("[]\n", encoding="utf-8")
    cifp_root = tmp_path / "CIFP_260319"
    charts_root = tmp_path / "RNAV_CHARTS"
    procedure_output = tmp_path / "procedures.geojson"
    czml_output = tmp_path / "trajectories.czml"
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], check: bool) -> None:
        assert check is True
        calls.append(cmd)

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    monkeypatch.setattr(
        pipeline.sys,
        "argv",
        [
            "run_asd-b_fetch_and_generate.py",
            "--airport",
            "KRDU",
            "--aeroviz-root",
            str(aeroviz_root),
            "--input-json",
            str(input_json),
            "--czml-output",
            str(czml_output),
            "--generate-procedures",
            "--cifp-root",
            str(cifp_root),
            "--procedure-output",
            str(procedure_output),
            "--include-procedure-transitions",
            "--procedure-charts-root",
            str(charts_root),
        ],
    )

    pipeline.main()

    assert len(calls) == 2
    assert calls[0][1] == str(python_dir / "generate_czml.py")
    procedure_cmd = calls[1]
    assert procedure_cmd[1] == str(python_dir / "preprocess_procedures.py")
    assert procedure_cmd[procedure_cmd.index("--airport") + 1] == "KRDU"
    assert procedure_cmd[procedure_cmd.index("--cifp-root") + 1] == str(cifp_root)
    assert procedure_cmd[procedure_cmd.index("--output") + 1] == str(procedure_output)
    assert "--include-all-rnav" in procedure_cmd
    assert "--include-transitions" in procedure_cmd
    assert procedure_cmd[procedure_cmd.index("--charts-root") + 1] == str(charts_root)
