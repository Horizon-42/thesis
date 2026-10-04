"""Test functions cut verbatim from live test files on 2026-10-03 (two-tier v4, stage A0): each checks a live mirror
against a module now in this archive. They do not run (the live side of each mirror is unchanged until its stage
brings the archived side back)."""

# ---- cut from tests/test_autopilot.py

def test_the_executors_glidepath_edge_is_the_post_training_checks():
    """`vertical.GLIDEPATH_BELOW_M` mirrors `prior.procedure.GLIDEPATH_BELOW_M` (`autopilot` imports no model package)."""
    from ts_transformer.autopilot.vertical import GLIDEPATH_BELOW_M
    from ts_transformer.prior import procedure

    assert GLIDEPATH_BELOW_M == procedure.GLIDEPATH_BELOW_M


# ---- cut from tests/test_publish_ts_experiment_trajectories.py

def test_the_generation_names_mirror_the_runner():
    from ts_transformer.experiments.prior_generation_grading import GRADING_SCHEMA
    from ts_transformer.experiments.prior_generation_records import HORIZON, PREDICTORS, RECORDS_SCHEMA

    assert publisher.GENERATION_RECORDS_SCHEMA == RECORDS_SCHEMA
    assert publisher.GENERATION_GRADING_SCHEMA == GRADING_SCHEMA
    assert (publisher.GENERATION_PREDICTORS, publisher.GENERATION_HORIZON) == (PREDICTORS, HORIZON)


# ---- cut from tests/test_architecture.py (the prior's layout rules)

PRIOR = TS_DIR / "prior"
#: The prior sits on the instruction language (framework document §2): inside this package it reads the words and the
#: artefact, the day split the artefact was dealt by (its landing context leaves the sealed test days out) and the causal
#: runway rules its first-step runway is read against (`data.runway_context`, design §8) — never the executor, a model
#: of the prediction paths, the training plane or a runner. Outside it, the glidepath lower edge (`prior/procedure.py`,
#: post-training design §3) reads the coded approach, the harvest's runway data and the LPV cone
#: (`PROCEDURE_MAY_IMPORT_OUTSIDE`) and stays torch-free.
PRIOR_MAY_IMPORT = ("prior.", "instructions.", "data.day_split", "data.runway_context", "io_utils", "repo_layout")
PROCEDURE_MAY_IMPORT_OUTSIDE = {"evaluation.cli", "flight_scenarios.fas_geometry", "flight_scenarios.procedure_final",
                                "trajectory_data_process.harvest.airports"}


def test_the_prior_reads_only_the_instruction_language():
    groups = {p.name for p in TS_DIR.iterdir() if (p / "__init__.py").is_file()} | {p.stem for p in TS_DIR.glob("*.py")}
    for path in PRIOR.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(TS_DIR).as_posix()
        for name in _imported_names(path):
            if name.split(".")[0] not in groups or name == "prior":
                continue
            allowed = any((name == item[:-1] or name.startswith(item)) if item.endswith(".") else
                          (name == item or name.startswith(item + ".")) for item in PRIOR_MAY_IMPORT)
            assert allowed, f"{rel} imports {name}"


def test_the_glidepath_edge_reads_only_the_procedure_sources_and_no_torch():
    groups = {p.name for p in TS_DIR.iterdir() if (p / "__init__.py").is_file()} | {p.stem for p in TS_DIR.glob("*.py")}
    for name in _imported_names(PRIOR / "procedure.py"):
        top = name.split(".")[0]
        assert top != "torch", f"prior/procedure.py imports {name}"
        if top in groups or top in {"__future__", "collections", "math", "dataclasses", "typing", "numpy"}:
            continue
        assert any(name == allowed or name.startswith(allowed + ".") for allowed in PROCEDURE_MAY_IMPORT_OUTSIDE), \
            f"prior/procedure.py imports {name}"


def test_only_the_runners_reach_the_prior_for_now():
    for path in _module_files():
        if path.is_relative_to(PRIOR):
            continue
        rel = path.relative_to(TS_DIR).as_posix()
        if any(name.split(".")[0] == "prior" for name in _imported_names(path)):
            assert rel.startswith("experiments/"), f"{rel} imports the prior"
