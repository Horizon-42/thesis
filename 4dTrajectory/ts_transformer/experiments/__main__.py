"""`python -m ts_transformer.experiments <name> [args]` — the runners' one entry point.

Run by path (`python 4dTrajectory/ts_transformer/experiments/__main__.py <name>`, which is
what the repository-root `run_ts.py` does) it puts `4dTrajectory/` on `sys.path` itself,
the way the package CLI does; run as a module it needs nothing.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
from pathlib import Path

if __package__ in (None, ""):  # run by path: the package's parent goes on the path
    _PACKAGE_PARENT = Path(__file__).resolve().parents[2]
    if str(_PACKAGE_PARENT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_PARENT))

PACKAGE = "ts_transformer.experiments"
#: Modules here that are not runners: shared code the runners import (the scene judge's closed loop: M0 step 5 on; the
#: scene samples: M2 on; a commanded aircraft's view of its scene, the window loop, its augmentation and tuner: step 7
#: on; the round protocol of the post-training: M4 on).
NOT_RUNNERS = {"__main__", "support", "traffic_loop", "traffic_rounds", "traffic_scene_data", "traffic_speaking",
               "traffic_window", "traffic_window_augment", "traffic_window_tuner"}


def runner_names() -> list[str]:
    package = importlib.import_module(PACKAGE)
    return sorted(
        info.name for info in pkgutil.iter_modules(package.__path__)
        if info.name not in NOT_RUNNERS
    )


def describe(name: str) -> str:
    doc = importlib.import_module(f"{PACKAGE}.{name}").__doc__ or ""
    return doc.strip().splitlines()[0] if doc.strip() else ""


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    names = runner_names()
    if not argv or argv[0] in ("-h", "--help", "--list"):
        width = max(len(name) for name in names)
        print("usage: run_ts.py <name> [args]   (each runner takes --help)\n")
        for name in names:
            print(f"  {name:<{width}}  {describe(name)}")
        return 0 if argv else 2
    name = argv[0].replace("-", "_")
    if name not in names:
        print(f"unknown runner {argv[0]!r}; one of: {', '.join(names)}", file=sys.stderr)
        return 2
    module = importlib.import_module(f"{PACKAGE}.{name}")
    # Every runner parses `sys.argv[1:]` when its `main` takes no arguments; the ones that
    # take `argv` read it the same way, so one calling convention covers both.
    sys.argv = [f"run_ts.py {name}", *argv[1:]]
    result = module.main()
    return 0 if result is None else int(result)


if __name__ == "__main__":
    raise SystemExit(main())
