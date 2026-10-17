"""Where these archived scripts find the repository, the data and the results (README §2).

On the Mac (2026-10-11/12) they sat in RESULTS beside their outputs and read a COPY of SP-AI's live harvest
(`spai_copy_20261011/harvest`, pulled read-only on 2026-10-11 by `pull.sh`). Archived, they read the repository they
sit in and SP-AI's live harvest; RESULTS (git-ignored) exists only where it was made or copied to.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]           # scripts, this archive, archive, ts_transformer, 4dTrajectory, the repository
POOLED = REPO / "4dTrajectory/outputs/POOLED"
#: The experiments' outputs and the inputs they read back (`inputs/`): on the Mac only, unless copied.
RESULTS = POOLED / "spai_copy_20261011/plant_comparison"
#: SP-AI's live harvest; the Mac runs read its 2026-10-11 copy, `POOLED / "spai_copy_20261011/harvest"`.
HARVEST = REPO / "trajectory_data_process/outputs/harvest"
