"""The two optimizer constants `outputs/guidance/skeleton.py` mirrors (the optimizer's package is
not on this package's import path) are asserted equal to their owners, the way
`test_final_approach_geometry.py` pins the corridor's."""

from __future__ import annotations

import sys

from ts_transformer.outputs.guidance.skeleton import RNP_HALF_WIDTH_M, THRESHOLD_TOLERANCE_M
from ts_transformer.repo_layout import REPO_ROOT


def test_the_mirrored_constants_equal_their_owners():
    optimization = REPO_ROOT / "4dTrajectory" / "optimization"
    if str(optimization) not in sys.path:
        sys.path.insert(0, str(optimization))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from geokit import NM_M

    import scenario_optimization
    from aeroviz_backend import procedure_segments

    assert RNP_HALF_WIDTH_M == procedure_segments._DEFAULT_RNP_NM * NM_M
    assert THRESHOLD_TOLERANCE_M == scenario_optimization._FRAME_ANCHOR_TOLERANCE_M
