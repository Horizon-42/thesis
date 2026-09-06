"""One module per subcommand, each exposing ``HELP``, ``add_cli_arguments`` and ``run_cli``.

`__main__` builds the top-level parser from that triple and dispatches on the command name,
so adding a subcommand is adding a module and one row of the table — never another branch
in a 700-line function.

`common.py` holds what more than one command needs: the shared argument groups, the
config assembly (`config_from_args`), the cohort loading `train` and `cross-validate`
share, and `split_keys_for_current_data`. `approach-cohorts` and `benchmark-batch` follow
the same triple from their own modules (`approach_clustering.cli`, `batch_benchmark`),
which is why they have no module here.

This package re-exports nothing on purpose: the command modules import each other's
helpers through `common`, never through the package.
"""
