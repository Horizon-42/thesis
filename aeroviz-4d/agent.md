# AeroViz-4D Agent Notes

## Python Environment

- Run all Python scripts and Python tests in the conda `aviation` environment.
- Use `conda run -n aviation ...` for non-interactive commands, for example:
  - `conda run -n aviation python python/preprocess_procedures.py ...`
  - `conda run -n aviation python -m pytest python/tests/test_preprocess_procedures.py`
- Do not run project Python commands with the system Python, Homebrew Python, or an unqualified `python` command.
