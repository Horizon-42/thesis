"""The control package's membership rule, enforced rather than described.

`control/` exists because twelve `control_*.py` files at the top level named their subject
but not their role. Two things keep that from coming back, and both are checkable:

1. no new `control_*.py` may appear at the top level;
2. a module belongs in `control/` only if EVERY consumer of it is control-specific — which
   is why `prediction_outputs`, `terminal_state_loss`, `arc_length_geometry`,
   `fixed_dt_supervision` and `flyability` stay outside it.

The second rule is the one worth testing: without it the package slowly absorbs shared
modules and starts claiming ownership it does not have, which is worse than the flat
listing because the name now lies.
"""
from __future__ import annotations

import ast
from pathlib import Path
import sys

TS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TS_DIR.parents[1]
if str(TS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(TS_DIR.parent))

CONTROL = TS_DIR / "control"
#: Completed campaigns kept in the repository as the record behind published numbers. They
#: are NOT part of the package: no live module may import them, and the module walk below
#: must not count them as consumers of anything.
ARCHIVE = TS_DIR / "archive"
# Shared with the state path through `fixed_anchor_validation`, `dataset` or `batching`.
SHARED_BY_DESIGN = {
    "prediction_outputs",
    "terminal_state_loss",
    "arc_length_geometry",
    "fixed_dt_supervision",
    "flyability",
}


def _module_files() -> list[Path]:
    return [
        path
        for path in TS_DIR.rglob("*.py")
        if "__pycache__" not in path.parts
        and "vendor" not in path.parts
        and "tests" not in path.parts
        and "docs" not in path.parts
        and "archive" not in path.parts
    ]


def _archive_import_candidates() -> list[Path]:
    """Everything that could import the archive and be RUN — wider than `_module_files`.

    The layering checks below deliberately skip `tests/` and the root runners (a test may
    import anything; a runner is not part of the package). The archive rule is not about
    layering: `CLAUDE.md` says the archive is off the import path, and a test file or a
    root `run_ts_*.py` importing it would break that claim just as loudly as a package
    module would.
    """
    return [
        *_module_files(),
        *sorted(p for p in (TS_DIR / "tests").glob("*.py")),
        *sorted(REPO_ROOT.glob("run_ts_*.py")),
    ]


def _imported_names(path: Path) -> set[str]:
    """Every module name a file imports, with relative imports resolved.

    `from .common import x` inside `cli/` reads as `node.module == "common"` and
    `node.level == 1`; without the resolution below the layering checks would see a
    top-level `common` that does not exist and miss a real edge.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    # Root `run_ts_*.py` runners live outside the package and have no relative imports.
    package = (
        path.parent.relative_to(TS_DIR).parts
        if path.is_relative_to(TS_DIR) else ()
    )
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                prefix = package[: len(package) - node.level + 1]
                base = ".".join((*prefix, node.module)) if node.module else ".".join(prefix)
                names.add(base)
                names.update(f"{base}.{alias.name}" for alias in node.names)
            elif node.module:
                names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    # The layering rules below are written in package-relative names (`control.envelope`,
    # `dataset`); every live import is qualified (`ts_transformer.control.envelope`).
    return {_package_relative(name) for name in names}


PACKAGE = "ts_transformer"


def _package_relative(name: str) -> str:
    return name[len(PACKAGE) + 1:] if name.startswith(PACKAGE + ".") else name


def _package_module_names() -> set[str]:
    """The flat names the modules used to be importable under, from the directory itself."""
    names = {p.stem for p in TS_DIR.glob("*.py")} - {"__init__", "__main__"}
    names |= {p.name for p in TS_DIR.iterdir() if (p / "__init__.py").is_file()}
    return names


def _flat_import_candidates() -> list[Path]:
    """Everything that imports the package and is RUN: the modules, the tests, the docs
    scripts, the root runners and the trajectory_data_process tests that drive them."""
    return [
        *_archive_import_candidates(),
        *sorted((TS_DIR / "docs").glob("*.py")),
        REPO_ROOT / "publish_ts_experiment_trajectories.py",
        *sorted((REPO_ROOT / "trajectory_data_process" / "tests").glob("test_ts_*.py")),
    ]


def test_no_module_is_imported_by_its_flat_name():
    """`ts_transformer` is a package (2026-09-09, review §4.1): every import of one of its
    modules is qualified. A flat `from config import TSConfig` would only resolve while
    `ts_transformer/` itself sat on `sys.path` — and then it would load a SECOND copy of
    the module beside `ts_transformer.config`, with its own dataclass, registries and
    module constants, and every `isinstance` and identity check between the two would
    silently fail. So the flat names are refused outright."""
    flat = _package_module_names()
    offending: dict[str, set[str]] = {}
    for path in _flat_import_candidates():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        found = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and not node.level and node.module:
                if node.module.split(".")[0] in flat:
                    found.add(node.module)
            elif isinstance(node, ast.Import):
                found.update(a.name for a in node.names if a.name.split(".")[0] in flat)
        if found:
            offending[path.relative_to(REPO_ROOT).as_posix()] = found
    assert not offending, f"flat imports of package modules: {offending}"


def test_the_package_directory_itself_is_never_put_on_sys_path():
    """The companion rule: no bootstrap may insert `ts_transformer/` into `sys.path` — the
    package's PARENT, `4dTrajectory/`, is what resolves the qualified names."""
    offending = []
    for path in _flat_import_candidates():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "sys.path" not in line:
                continue
            if '"4dTrajectory" / "ts_transformer")' in line:
                offending.append(f"{path.relative_to(REPO_ROOT)}:{number}")
    assert not offending, offending


def test_no_control_prefixed_module_returns_to_the_top_level():
    stragglers = sorted(p.name for p in TS_DIR.glob("control_*.py"))
    assert not stragglers, (
        f"{stragglers} belong under control/ by role, not at the top level behind a prefix"
    )
    assert (CONTROL / "__init__.py").is_file()
    for sub in ("dynamics", "loss", "training"):
        assert (CONTROL / sub / "__init__.py").is_file(), f"control/{sub} is not a package"


def test_nothing_live_imports_the_archive():
    """`archive/` holds completed campaigns, not package code.

    They are kept so a published number has its code, and they are off the import path on
    purpose: an archived module is not maintained against the current contracts (its
    checkpoints are already refused, its objective and dynamics backend are retired). A
    live import would quietly make it load-bearing again.
    """
    campaigns = sorted(p.name for p in ARCHIVE.iterdir() if p.is_dir())
    assert campaigns and all((ARCHIVE / name / "README.md").is_file() for name in campaigns), (
        f"every archived campaign needs a README saying what it is: {campaigns}"
    )
    for path in _archive_import_candidates():
        offending = {name for name in _imported_names(path) if name.split(".")[0] == "archive"}
        assert not offending, (
            f"{path} imports {sorted(offending)}; archive/ is a record of completed "
            f"campaigns, never a dependency"
        )
    # And nothing in the archive is reachable as a package: no __init__.py on the way in.
    assert not (ARCHIVE / "__init__.py").exists()
    for campaign in ARCHIVE.iterdir():
        if campaign.is_dir():
            assert not (campaign / "__init__.py").exists(), (
                f"{campaign.name} is importable; the archive must stay off the import path"
            )


def test_shared_modules_stay_outside_the_control_package():
    """Each name here has at least one consumer that is NOT control-specific."""
    for name in SHARED_BY_DESIGN:
        assert (TS_DIR / f"{name}.py").is_file(), (
            f"{name} moved into control/, but the state path reaches it — check its "
            f"consumers before claiming it as control-only"
        )
        consumers = {
            path.relative_to(TS_DIR).as_posix()
            for path in _module_files()
            if name in _imported_names(path)
        }
        outside = {c for c in consumers if not c.startswith("control/")}
        assert outside, (
            f"{name} is now imported only from control/ — it may have become genuinely "
            f"control-specific, in which case move it in and drop it from SHARED_BY_DESIGN"
        )


def test_the_control_package_does_not_import_the_training_loop():
    """control/ is imported BY the training loop; it must never reach back up into it.

    `dataset` is deliberately NOT on this list. `Normalizer` and the window types are
    data-plane value types the loss modules genuinely consume, and `Normalizer.fit`
    balances over `FlightSeries`, so it belongs with the data plane rather than under
    `control`. `objective` and `validation` ARE on it: both import `control/`, so one
    importing either back would be a cycle as well as a layering inversion. The direction
    that matters is this one: a loss module that imported `train`, `objective` or
    `validation` would make the package unusable outside the loop it was extracted from.
    """
    consumers = {
        "train", "objective", "validation", "forecast", "__main__", "models", "batching",
    }
    for path in CONTROL.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        offending = _imported_names(path) & consumers
        assert not offending, (
            f"{path.relative_to(TS_DIR)} imports {sorted(offending)}; control/ is imported "
            f"BY the training loop, never the other way round"
        )


#: `dataset` imports these four `control/` modules — the data plane genuinely needs the
#: control envelope, the conditioning names, the inverse dynamics and the teacher table to
#: BUILD a batch. That is the one edge that runs downward into `control/`, and it only
#: stays acyclic while these four stay `dataset`-free.
CONTROL_MODULES_DATASET_IMPORTS = (
    "control/basis_fit.py",
    "control/conditioning.py",
    "control/dynamics/inverse.py",
    "control/envelope.py",
)


def test_the_dataset_control_edge_runs_one_way_only():
    """`dataset` imports four `control/` modules; none of them may import `dataset` back.

    The general rule is the other way round — `control/` MAY import `dataset` (`Normalizer`
    and the window types are data-plane values a loss genuinely consumes), which is why the
    layering test above does not ban it. These four are the exception inside the exception:
    `dataset` needs them to build a batch at all, so a `dataset` import in any of them
    closes a cycle that only fails at import time, in whichever order a caller happens to
    hit first.
    """
    dataset_imports = _imported_names(TS_DIR / "dataset.py")
    for name in CONTROL_MODULES_DATASET_IMPORTS:
        module = name[: -len(".py")].replace("/", ".")
        assert module in dataset_imports, (
            f"{module} is no longer imported by dataset — drop it from "
            f"CONTROL_MODULES_DATASET_IMPORTS rather than leaving a rule about nothing"
        )
        assert "dataset" not in _imported_names(TS_DIR / name), (
            f"{name} imports dataset, which imports it: the one downward edge into "
            f"control/ has become a cycle"
        )


def test_every_subcommand_module_exposes_the_same_triple():
    """`__main__` is a table, not a 700-line `if` chain, and this is what makes that safe.

    A command is `(help, add_cli_arguments, run_cli)`. `approach-cohorts` and
    `benchmark-batch` supply their two callables from their own modules and carry their
    help text in the table, which is why they have no module under `cli/`.
    """
    import argparse
    import importlib.util

    spec = importlib.util.spec_from_file_location("ts_cli_main", TS_DIR / "__main__.py")
    main_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main_module)

    assert set(main_module.COMMANDS) == {
        "train", "cross-validate", "approach-cohorts", "benchmark-batch",
        "evaluate-fit", "freeze-test", "predict",
    }
    for name, (help_text, add_cli_arguments, run_cli) in main_module.COMMANDS.items():
        assert help_text and callable(add_cli_arguments) and callable(run_cli), name
        # It has to actually build: a subparser that raises is a --help that never prints.
        add_cli_arguments(argparse.ArgumentParser(prog=name))


#: Every training-parser dest that is NOT a `TSConfig` field the CLI sets. Two kinds:
#: infrastructure (where the data is, where the output goes, the experiment identity), and
#: the three documented exceptions where the flag deliberately is not the field's name.
#: Frozen here so that DELETING a name from `CLI_CONFIG_FIELDS` fails loudly instead of
#: leaving its flag parsed, accepted and silently ignored.
NON_CONFIG_DESTS = {
    "help",
    # infrastructure
    "data", "eligibility_roster", "airport", "output_dir",
    "config_overrides", "campaign_id", "experiment_id",
    # the documented exceptions (see CLI_CONFIG_FIELDS' comment)
    "batch_size",          # the flag also accepts "auto", which is not a config value
    "instance_norm",       # one flag, two backbone-specific fields (use_norm / revin)
    "control_recipe_name",  # its flag IS its name, but it resolves before the rest
}


def _training_parser():
    import argparse

    from ts_transformer.cli.common import add_data_args, add_training_args

    parser = argparse.ArgumentParser()
    add_data_args(parser)          # --aircraft-type / --aircraft-filter live here
    add_training_args(parser)
    return parser


def test_every_training_flag_is_named_after_the_field_it_sets():
    """The rename that turned a hand-written flag→field table into a list of field names.

    `cli.common` asserts at import that every name in `CLI_CONFIG_FIELDS` is a `TSConfig`
    field. The two other directions are checked here: that each listed field has a flag
    spelled exactly as the field, and — the one that catches a DELETION — that every dest
    the parser defines is either a listed field or a frozen non-config dest. Without the
    second, dropping `"patience"` from the list leaves `--patience 99` parsed and ignored.
    """
    from ts_transformer.cli.common import CLI_CONFIG_FIELDS

    parser = _training_parser()
    flags = {option for action in parser._actions for option in action.option_strings}
    missing = [
        name for name in CLI_CONFIG_FIELDS
        if f"--{name.replace('_', '-')}" not in flags
    ]
    assert not missing, f"{missing} are config fields the CLI claims to set with no flag"

    dests = {action.dest for action in parser._actions}
    assert dests - set(CLI_CONFIG_FIELDS) == NON_CONFIG_DESTS, (
        "a training flag exists whose dest is neither in CLI_CONFIG_FIELDS nor a declared "
        f"non-config dest: {sorted(dests - set(CLI_CONFIG_FIELDS) - NON_CONFIG_DESTS)}; "
        f"or one was removed: {sorted(NON_CONFIG_DESTS - dests)}"
    )


#: The 2026-09-07 (T3-19) renames, old spelling -> new. Every old one must be REFUSED:
#: argparse's default prefix matching accepted four of them silently.
RENAMED_FLAGS_2026_09_07 = {
    "--closure-labels": "--closure-labels-path",
    "--dt": "--dt-s",
    "--fitted-tail-weight": "--fitted-tail-position-weight",
    "--fitted-terminal-weight": "--fitted-terminal-position-weight",
    "--kinematic-consistency-weight": "--kinematic-consistency-loss-weight",
    "--control-thrust-tau-s": "--control-thrust-time-constant-s",
    "--control-bank-tau-s": "--control-bank-time-constant-s",
    "--control-load-tau-s": "--control-load-time-constant-s",
    "--control-state-clock": "--control-state-supervision-clock",
    "--control-fitted-teacher": "--control-fitted-teacher-path",
    "--control-heading-rate-weight": "--control-heading-rate-loss-weight",
    "--control-heading-rate-scale-dps": "--control-heading-rate-loss-scale-dps",
    "--control-bank-tv-weight": "--control-bank-tv-loss-weight",
    "--control-rollout-dt": "--control-rollout-integrator-dt-s",
    "--control-recipe": "--control-recipe-name",
}


def test_a_renamed_flag_is_refused_not_prefix_matched(capsys):
    """`--dt 2` must not keep working as `--dt-s 2`.

    argparse accepts any unambiguous PREFIX by default, so four of the fifteen renamed
    flags (`--dt`, `--control-recipe`, `--closure-labels`, `--control-fitted-teacher`)
    still parsed after T3-19 — a stale command line would have set the field it looks like
    it sets while reading as up to date. Every subparser passes `allow_abbrev=False`.
    """
    import importlib.util

    import pytest

    spec = importlib.util.spec_from_file_location("ts_cli_abbrev", TS_DIR / "__main__.py")
    main_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main_module)

    prefixes = {old for old, new in RENAMED_FLAGS_2026_09_07.items() if new.startswith(old)}
    assert prefixes == {
        "--dt", "--control-recipe", "--closure-labels", "--control-fitted-teacher",
    }, f"the set of old spellings argparse could prefix-match has changed: {sorted(prefixes)}"

    for old in RENAMED_FLAGS_2026_09_07:
        with pytest.raises(SystemExit) as info:
            main_module.main([
                "train", "--data", "x.json", "--output-dir", "out", old, "1",
            ])
        assert info.value.code == 2, old
        assert "unrecognized arguments" in capsys.readouterr().err, old


def test_the_conditioning_names_and_their_scalings_are_one_source():
    """`mass_100t` must actually be divided by 100 t, and nothing used to check that.

    The names sat in `dataset` and the divisors in `dynamics_arrays` fifty lines apart, so
    renaming a channel without changing its divisor was a silent mislabel of the vector the
    head is conditioned on.
    """
    from types import SimpleNamespace

    from ts_transformer.control.conditioning import (
        CONDITION_CHANNELS,
        DYNAMICS_CONDITION_NAMES,
        condition_vector,
    )

    assert DYNAMICS_CONDITION_NAMES == tuple(name for name, _ in CONDITION_CHANNELS)

    aero = SimpleNamespace(S=500.0, Cl_max=3.0, Cd0=0.1, k=0.1, stall_threshold=0.8,
                           k_stall=0.2)
    vector = condition_vector(mass_kg=100_000.0, max_thrust_n=1_000_000.0, aero=aero)

    # Every channel fed exactly the quantity its name declares, so each must read 1.0 —
    # except stall_threshold, which the name says is already dimensionless.
    named = dict(zip(DYNAMICS_CONDITION_NAMES, vector))
    assert named["stall_threshold"] == 0.8
    for name, value in named.items():
        if name != "stall_threshold":
            assert value == 1.0, f"{name} does not divide by the unit its name states"


def test_every_new_run_vocabulary_is_actually_refused(tmp_path, capsys):
    """A value a STORED config may carry but a NEW run may not select must be refused at
    BOTH doors, and `--config-overrides` is the one with no argparse `choices` to stop it.
    `cli.common._NEW_RUN_VOCABULARIES` is the list; nothing asserted that any entry on it
    bites, so an axis added to the vocabulary and forgotten here would stay selectable.

    Each entry needs the companion settings its own field validation demands (a hook needs
    the lag dynamics, a CTA needs the control output). A new entry with no companion row
    fails here with a KeyError rather than passing vacuously.
    """
    import importlib.util
    import json

    import pytest

    from ts_transformer.cli.common import _NEW_RUN_VOCABULARIES
    from ts_transformer.config import (
        CONTROL_HOOKS, CONTROL_STATE_LOSS_GRIDS, CTA_CONDITIONINGS, INTENT_CONDITIONINGS,
        PREDICTION_OUTPUTS, STATE_POSITION_REFERENCES,
    )
    from ts_transformer.coordinate_frames import COORDINATE_FRAMES

    #: field -> (its full stored vocabulary, the other settings that value needs to be legal)
    VOCABULARY_CONTEXT = {
        "control_command_hook": (CONTROL_HOOKS, {
            "prediction_output": "control",
            "control_dynamics_model": "first-order-lag",
            "control_dynamics_backend": "scaled-transport-chart-velocity",
            "control_state_loss_grid": "native-segment-endpoints",
        }),
        "cta_conditioning": (CTA_CONDITIONINGS, {"prediction_output": "control"}),
        "state_position_reference": (STATE_POSITION_REFERENCES, {}),
        # Frozen 2026-09-09 (package review §5).
        "prediction_output": (PREDICTION_OUTPUTS, {
            "closure_labels_path": "labels.json",
            "checkpoint_selection_metric": "fixed-anchor-objective",
        }),
        "intent_conditioning": (INTENT_CONDITIONINGS, {}),
        "control_state_loss_grid": (CONTROL_STATE_LOSS_GRIDS, {
            "prediction_output": "control",
            "control_state_supervision_clock": "observed",
        }),
        "coordinate_frame": (COORDINATE_FRAMES, {}),
    }

    spec = importlib.util.spec_from_file_location("ts_cli_vocabulary", TS_DIR / "__main__.py")
    main_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main_module)

    assert _NEW_RUN_VOCABULARIES, "the vocabulary list is empty; this test would pass vacuously"
    for field, available, _why in _NEW_RUN_VOCABULARIES:
        full, companions = VOCABULARY_CONTEXT[field]
        # An entry whose every value is available is a bound that can never bind.
        refused = [value for value in full if value not in available]
        assert refused, f"{field}: every value is available, so this entry never binds"

        overrides = tmp_path / f"{field}.json"
        overrides.write_text(json.dumps({field: refused[0], **companions}))
        with pytest.raises(SystemExit) as info:
            main_module.main([
                "train", "--data", "x.json", "--output-dir", str(tmp_path / "out"),
                "--config-overrides", str(overrides),
            ])
        assert info.value.code == 2, field
        message = capsys.readouterr().err
        assert f"{field}={refused[0]!r} cannot be selected for a new run" in message, field
