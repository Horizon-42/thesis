# AeroViz-4D Agent Notes

## Python Environment

- Run all Python scripts and Python tests in the conda `aviation` environment.
- Use `conda run -n aviation ...` for non-interactive commands, for example:
  - `conda run -n aviation /Users/liudongxu/opt/miniconda3/envs/aviation/bin/python3.13 python/preprocess_procedures.py ...`
  - `conda run -n aviation pytest python/tests/test_preprocess_procedures.py`
- In this workspace, `conda run -n aviation python` may resolve to Homebrew Python instead of the conda environment interpreter. Prefer the explicit environment Python path above for scripts, and `conda run -n aviation pytest ...` for tests.
- Do not run project Python commands with the system Python, Homebrew Python, or an unqualified `python` command.
