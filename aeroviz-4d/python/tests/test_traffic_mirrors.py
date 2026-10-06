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
    for field in ("aircraft", "flown_aircraft_with_a_loss_left_after_the_block", "not_answered"):
        assert f"{field}" in client, field
        assert f'"{field}"' in readout or f"{field}" in readout, field
    # the arrivals of a block that stayed their records come from the job's `stayedRecords` (named, with callsign and type), not from
    # a count in the readout: the panel reads no `skipped_no_dynamics`
    assert "skipped_no_dynamics" not in client and "skipped_no_dynamics" not in read(SRC / "utils" / "trafficJobResult.ts")
    job = read(OPTIMIZATION / "traffic_job.py")
    assert '"stayedRecords"' in job and '"callsign"' in job and '"type"' in job
    assert "stayedRecords?: Record<string, TrafficStayedRecord>" in client
    assert "export interface TrafficStayedRecord" in client and "callsign: string | null;" in client


def test_the_scenario_catalog_schema_and_loss_kinds_the_panel_names_are_the_backends_and_the_judges():
    source = read(OPTIMIZATION / "traffic_job_files.py")
    schema = re.search(r'CATALOG_SCHEMA = "([^"]+)"', source)
    assert schema is not None
    assert f'TRAFFIC_CATALOG_SCHEMA = "{schema.group(1)}"' in read(SRC / "data" / "trafficJobs.ts")
    kinds = re.search(r'IN_TRAIL, DIAGONAL, RADAR_OR_VERTICAL, AT_THRESHOLD = ((?:"\w+",? ?)+)',
                      read(OPTIMIZATION.parent / "ts_transformer" / "inference" / "separation.py"))
    assert kinds is not None
    assert ts_strings(SRC / "utils" / "trafficScenarios.ts", "TRAFFIC_LOSS_KINDS") == re.findall(r'"(\w+)"', kinds.group(1))


def test_the_scenario_route_the_client_calls_is_the_backends():
    assert '"/traffic/scenarios"' in read(ROOT / "aeroviz_backend" / "http_server.py")
    assert "/traffic/scenarios?airport=" in read(SRC / "data" / "trafficJobs.ts")


def test_the_phase_before_the_jobs_first_answer_and_the_lightest_wake_category_are_the_backends():
    phase = re.search(r'PHASE_STARTING = "([^"]+)"', read(OPTIMIZATION / "traffic_job_files.py"))
    assert phase is not None
    assert f'TRAFFIC_PHASE_STARTING = "{phase.group(1)}"' in read(SRC / "data" / "trafficJobs.ts")
    categories = re.search(r'CWT_CATEGORIES = frozenset\("([A-Z]+)"\)',
                           read(OPTIMIZATION.parent / "ts_transformer" / "inference" / "runway_schedule.py"))
    assert categories is not None
    # the categories run from the heaviest (A) to the lightest: the light aircraft hidden by default are the last
    assert categories.group(1) == "".join(sorted(categories.group(1)))
    assert f'LIGHT_AIRCRAFT_CATEGORY = "{categories.group(1)[-1]}"' in read(SRC / "utils" / "trafficScenarios.ts")


def test_the_fields_of_a_jobs_per_aircraft_the_panel_reads_are_the_jobs():
    client = read(SRC / "data" / "trafficJobs.ts")
    names = []
    for list_name in ("AIRCRAFT_CHANGE_FIELDS", "AIRCRAFT_SOLVER_FIELDS"):          # may be null / always numbers
        fields = re.search(rf"{list_name} = \[(?P<body>[^\]]*)\]", client)
        assert fields is not None, list_name
        names += re.findall(r'"(\w+)"', fields.group("body"))
    # the numeric fields; `type` (text or null) is read beside them
    assert names == ["firstSolveLosses", "finalLosses", "landingVsRecordS", "delayS", "blockCheckLosses", "optimizeS",
                     "optimizeCpuS", "solves", "failedSolves"]
    job = read(OPTIMIZATION / "traffic_job.py")
    layout = read(OPTIMIZATION / "traffic_job_files.py")
    for name in ["type", *names]:          # the job writes each field, and its file layout documents it
        assert f'"{name}"' in job or f"{name}=" in job, name
        assert f"``{name}``" in layout, name
    # the time and the solves are the SOLVER's (the sidecars' `solves`, `wallS`, `cpuS`, `ok`), not the job's phases
    solve_time = read(OPTIMIZATION / "scenario_optimization.py")
    for key in ("wallS", "cpuS", '"ok"'):
        assert key in solve_time and key in job, key
    assert "next_solve_s" not in job and "optimize_s" not in job
    # the loss counts are the LOOP's own (MD14): the sidecar the loop writes carries them, the job reads them, never a recount
    loop = read(OPTIMIZATION / "traffic" / "loop.py")
    for key in ("answered_loss_instants", "kept_round"):          # shown: the census's count of the first and of the kept round
        assert f'"{key}"' in loop and f'["{key}"]' in job, key
    assert '"counted_loss_instants"' in loop and "counted_loss_instants" not in job.replace("``counted_loss_instants``", "")
    assert "``answered_loss_instants``" in layout and "``kept_round``" in layout
    # and the timing the panel shows under the summary: the job writes it, the layout documents it
    assert '"timing"' in job and '"totalS"' in job and "``timing``" in layout
    client = read(SRC / "data" / "trafficJobs.ts")
    assert "totalS: number" in client and "timing?: TrafficJobTiming" in client
    # the job's stages, in the words the panel prints them in
    stages = re.search(r"STAGE_READING, STAGE_ETA, STAGE_SCHEDULE, STAGE_FLYING, STAGE_EVALUATION, STAGE_SCENE = \((?P<body>.*?)\)\n",
                       layout, re.DOTALL)
    assert stages is not None
    assert re.findall(r'"([^"]+)"', stages.group("body")) == [
        "reading traffic", "earliest arrivals", "schedule", "optimizing", "evaluation", "building the scene"]
    for field in ("perAircraft", "timing", "stayedRecords"):
        assert field in read(ROOT / "aeroviz_backend" / "traffic_jobs.py"), field
        assert f"``{field}``" in layout, field


def test_the_report_fields_and_gate_names_the_panel_words_for_a_yellow_path_are_the_evaluations():
    metrics = read(ROOT / "evaluation" / "metrics.py")
    words = read(SRC / "utils" / "trafficJobResult.ts")
    # the gates a row can fail, as `violations` names them, and the report fields their numbers are read from
    for gate in ("lateral", "vertical", "speed"):
        assert f'violations.append("{gate}")' in metrics, gate
        assert f'violation === "{gate}"' in words, gate
    for field in ("lateral_m", "vertical_m", "crossing_speed_ms"):
        assert f"{field}=" in metrics or f'"{field}":' in metrics, field
        assert f"row.{field}" in words, field
    # the bounds: the vertical ones are metrics.py's; the speed window's are the speed gate's own (`SpeedWindow.to_dict`, where they are
    # defined — metrics.py only has the null placeholders of an unjudged row)
    speed_gate = read(ROOT / "evaluation" / "speed_gate.py")
    for bound, source in (("vertical_lower_m", metrics), ("vertical_upper_m", metrics),
                          ("speed_lower_ms", speed_gate), ("speed_upper_ms", speed_gate)):
        assert f'"{bound}": ' in source, bound
        assert f"row.bounds.{bound}" in words, bound
    assert '"lateral_m": item.lateral_bound_m' in metrics and "row.bounds.lateral_m" in words
    # the events that never gave a threshold crossing: their codes are the evaluation's `event_status`, and the one amount there is
    # (how far short `not_reached` stopped) is in the row's `reason`, in the sentence the panel reads it from
    arrival = read(ROOT / "evaluation" / "arrival.py")
    for code, const in (("not_reached", "NOT_REACHED_CODE"), ("threshold_not_bracketed", "NOT_BRACKETED_CODE")):
        assert f'None, "{code}"' in arrival, code
        assert f'{const} = "{code}"' in words, const
    assert 'f"trajectory ended {abs(final_projected.along_m):.1f} m before the threshold plane"' in arrival
    assert "/^trajectory ended ([0-9.]+) m before the threshold plane$/" in words
    assert 'violations=((outcome.event_status,) if computed_failure else ())' in metrics
    # the job names the report the index points at, and the viewer reads it from the job's files
    assert "evaluationReport" in read(SRC / "hooks" / "useTrafficJob.ts")
