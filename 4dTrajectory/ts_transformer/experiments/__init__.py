"""The re-runnable experiment runners (review §4.5) — one module each, one entry point.

    python run_ts.py <name> [args]            # from the repository root
    python -m ts_transformer.experiments <name> [args]   # with 4dTrajectory/ on the path

`<name>` is the module: `pipeline`, `cv`, `frame_ablation`, `anytime_curve`, … (`python
run_ts.py --list` prints them with their first docstring line). They used to be 22
`run_ts_*.py` files at the repository root, each inserting `4dTrajectory/` into `sys.path`
by hand and five of them carrying a byte-identical `_parse_airports`; the paths and the
shared helpers are `support.py`, the one-shot drivers are under `archive/`, and the tests
sit beside them in `tests/`. A runner is a CONSUMER of the package — it may import anything
in it, and nothing in the package imports a runner (`tests/test_architecture.py`).
"""
