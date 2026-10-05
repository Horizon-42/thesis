"""Direct-collocation trajectory optimiser (procedure constraints optional).

  * :class:`CollocationOptimizer` — the optimiser (``segments=None`` unconstrained; a list of legs
    for the procedure-constrained form).
  * :mod:`.schemes` (dynamics × fitting + metric-position normalization) and :mod:`.components`
    (bounds, altitude floor, control costs, terminal bank, solver factory) hold the reusable NLP
    internals — import those submodules directly when a script needs them.
"""

from .optimizer import CollocationOptimizer
from .components import altitude_floor_m, ALTITUDE_FLOOR_MARGIN_M

__all__ = [
    "CollocationOptimizer",
    "altitude_floor_m",
    "ALTITUDE_FLOOR_MARGIN_M",
]
