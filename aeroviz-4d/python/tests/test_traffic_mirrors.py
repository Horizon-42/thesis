"""The frontend's mirrors of the multi-aircraft job's vocabulary and settings, pinned (design §10).

The viewer cannot import the optimizer's constants, so `src/utils/trafficOutcome.ts` (the outcomes of a window),
`src/data/trafficJobs.ts` (the block lengths a job takes, the solver settings it shows read-only) and
`src/utils/comparisonSource.ts` (the name of the index it starts from) restate them. A mirror that differs from its
source shows the user a name or a number the job does not have. Read as text, not imported: the optimizer modules pull
in casadi, and this package must not depend on the modeling tree. The comparison of a list is ORDER-SENSITIVE on purpose.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "aeroviz-4d" / "src"
OPTIMIZATION = ROOT / "4dTrajectory" / "optimization"
_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def ts_strings(path: Path, name: str) -> list[str]:
    """The string literals of `export const <name> = [...] as const` in a TypeScript file."""
    match = re.search(rf"export const {name} = \[(?P<body>.*?)\]\s*as\s+const", read(path), re.DOTALL)
    assert match is not None, f"{name} not found in {path}"
    return re.findall(r'"([^"]+)"', _COMMENT.sub("", match.group("body")))


def ts_numbers(path: Path, name: str) -> list[float]:
    match = re.search(rf"export const {name} = \[(?P<body>.*?)\]\s*as\s+const", read(path), re.DOTALL)
    assert match is not None, f"{name} not found in {path}"
    return [float(n) for n in re.findall(r"[\d.]+", _COMMENT.sub("", match.group("body")))]


def ts_settings(path: Path) -> dict[str, float]:
    match = re.search(r"export const TRAFFIC_JOB_SETTINGS = \{(?P<body>.*?)\}\s*as\s+const", read(path), re.DOTALL)
    assert match is not None
    return {key: float(value) for key, value in re.findall(r"(\w+):\s*([\d.]+)", _COMMENT.sub("", match.group("body")))}


def py_default(path: Path, pattern: str) -> float:
    match = re.search(pattern, read(path))
    assert match is not None, f"{pattern!r} not found in {path}"
    return float(match.group(1))


def test_the_outcome_names_cover_the_loops_outcomes_in_its_order():
    source = read(OPTIMIZATION / "traffic" / "loop.py")
    match = re.search(
        r"SEPARATED_AT_BASELINE, SEPARATED, UNRESOLVED, SOLVE_FAILED, WAKE_AT_FIXED_TIME = \((?P<body>.*?)\)", source, re.DOTALL)
    assert match is not None
    outcomes = re.findall(r'"([^"]+)"', match.group("body"))
    assert len(outcomes) == 5
    assert ts_strings(SRC / "utils" / "trafficOutcome.ts", "TRAFFIC_OUTCOMES") == outcomes


def test_the_failure_reasons_the_names_match_are_the_ones_the_batch_writes():
    # `BaselineFailed` is the exception of loop.py; the block records prefix their reason with the stage that failed
    block = read(OPTIMIZATION / "traffic" / "block.py")
    ts = read(SRC / "utils" / "trafficOutcome.ts")
    for prefix in ("BaselineFailed: slot solve", "BaselineFailed: ETA solve"):
        assert f'f"{prefix}' in block, prefix
        assert f'"{prefix}"' in ts, prefix
    assert "class BaselineFailed" in read(OPTIMIZATION / "traffic" / "loop.py")
    assert 'f"block failed: {error}"' in read(OPTIMIZATION / "traffic_optimization.py")
    assert '"block failed:"' in ts


def test_the_block_lengths_a_job_takes_are_the_backends():
    source = read(OPTIMIZATION / "traffic_job_files.py")
    backend = re.search(r"BLOCK_LENGTHS_S = \((?P<body>[^)]*)\)", source)
    assert backend is not None
    assert ts_numbers(SRC / "data" / "trafficJobs.ts", "TRAFFIC_BLOCK_LENGTHS_S") == [
        float(n) for n in re.findall(r"\d+", backend.group("body"))]


def test_the_solver_settings_the_panel_shows_are_the_batch_defaults():
    settings = ts_settings(SRC / "data" / "trafficJobs.ts")
    config = OPTIMIZATION.parent.parent / "optimization_run_config.py"
    assert settings["maxDurationS"] == py_default(config, r"DEFAULT_MAX_DURATION_S = ([\d.]+)")
    assert settings["rolloutDtS"] == py_default(config, r"DEFAULT_ROLLOUT_DT_S = ([\d.]+)")
    assert settings["maxIterations"] == py_default(
        OPTIMIZATION / "collocation" / "components.py", r"(?m)^DEFAULT_MAX_ITERATIONS = (\d+)")
    loop = OPTIMIZATION / "traffic" / "loop.py"
    assert settings["stepS"] == py_default(loop, r"step_s: float = ([\d.]+)")
    assert settings["rowWindowS"] == py_default(loop, r"row_window_s: float = ([\d.]+)")
    assert settings["margin"] == py_default(loop, r"margin: float = ([\d.]+)")
    assert settings["maxRounds"] == py_default(loop, r"max_rounds: int = (\d+)")


def test_the_index_a_job_starts_from_is_the_builders_and_the_jobs():
    names = re.search(r'JOB_INDEX_FILE = "([^"]+)"', read(SRC / "utils" / "comparisonSource.ts"))
    assert names is not None
    assert f'INDEX_FILE = "{names.group(1)}"' in read(OPTIMIZATION / "traffic_job_files.py")
    assert f'out_dir / "{names.group(1)}"' in read(ROOT / "aeroviz-4d" / "python" / "build_scenario_comparison_czml.py")


def test_the_routes_the_client_calls_are_the_backends():
    client = read(SRC / "data" / "trafficJobs.ts")
    backend = read(ROOT / "aeroviz_backend" / "http_server.py")
    assert '"/traffic/arrivals"' in backend and "/traffic/arrivals?airport=" in client
    assert '"/traffic/jobs"' in backend and '"/traffic/jobs"' in client
    assert "/traffic/jobs/${jobId}" in client and "/files/" in client and "/cancel" in client
    assert r"/files/[^/]+|/cancel" in backend


def test_the_readout_fields_the_panel_reads_are_the_readouts():
    client = read(SRC / "data" / "trafficJobs.ts")
    readout = read(OPTIMIZATION / "traffic" / "readout.py")
    for field in ("aircraft", "skipped_no_dynamics", "flown_aircraft_with_a_loss_left_after_the_block", "not_answered"):
        assert f"{field}" in client, field
        assert f'"{field}"' in readout or f"{field}" in readout, field
    # the M2 readout sums the dynamics-less arrivals over EVERY block, a failed one's included
    assert 'sum(block["skipped_no_dynamics"] for block in summary["blocks"])' in readout
