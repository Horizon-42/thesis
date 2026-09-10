"""One module per subcommand, each exposing ``HELP``, ``add_cli_arguments`` and ``run_cli``.

`__main__` builds the top-level parser from that triple and dispatches on the command name,
so adding a subcommand is adding a module and one row of the table — never another branch
in a 700-line function.

`common.py` holds what more than one command needs: the shared argument groups, the
config assembly (`config_from_args`), the cohort loading `train` and `cross-validate`
share, and `split_keys_for_current_data`. `approach-cohorts` follows the same triple from
its own package (`approach_clustering.cli`), which is why it has no module here;
`benchmark-batch` is `benchmark_batch.py` (the top-level `batch_benchmark.py` until the
2026-09-10 grouping).

This package re-exports nothing on purpose: the command modules import each other's
helpers through `common`, never through the package.
"""
