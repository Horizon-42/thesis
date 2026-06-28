"""Backend dispatch — one implementation that works for **NumPy** and **CasADi**.

The whole point of this package is that the constraint functions you write are usable *as-is*
inside the CasADi NLP (not numpy-only throwaway). Two facts make that possible:

1. The arithmetic operators ``+  -  *  /`` are already overloaded for NumPy arrays **and** for
   CasADi ``SX``/``MX``. So write the linear parts of a constraint with plain operators — they
   work on both backends untouched.
2. Only the **nonlinear** ops differ (``abs``, ``sqrt``, ``atan2``, ``sin``, ``cos``, ``tan``,
   ``max``, conditional). Route those through the helpers here: each picks the CasADi version if
   **any** argument is a CasADi type, otherwise the NumPy version.

Rules of thumb for your TODO code:
- coordinates of a state node (``n, e, h, V, psi, gamma``) are the *variables* — pass them as
  scalar components (a NumPy array of node values, **or** a CasADi symbol); use ``+ - * /`` and
  the ``mathx`` helpers on them, **never** ``np.abs``/``np.hypot``/``np.arctan2`` directly.
- a **fix** (``A``, ``B``, GARP, LTP) and the procedure numbers (course width, GPA, …) are
  *constants*; computing with plain NumPy/``math`` on them is fine — they fold in as constants.
"""

from __future__ import annotations

import numpy as np

try:  # CasADi is present in the `aviation` env; keep the package importable without it.
    import casadi as ca
except ImportError:  # pragma: no cover
    ca = None

_CA_TYPES = () if ca is None else (ca.SX, ca.MX, ca.DM)


def is_casadi(*xs) -> bool:
    """True if any argument is a CasADi symbolic/numeric type."""
    return bool(_CA_TYPES) and any(isinstance(x, _CA_TYPES) for x in xs)


def fabs(x):
    return ca.fabs(x) if is_casadi(x) else np.abs(x)


def sqrt(x):
    return ca.sqrt(x) if is_casadi(x) else np.sqrt(x)


def sin(x):
    return ca.sin(x) if is_casadi(x) else np.sin(x)


def cos(x):
    return ca.cos(x) if is_casadi(x) else np.cos(x)


def tan(x):
    return ca.tan(x) if is_casadi(x) else np.tan(x)


def atan2(y, x):
    return ca.atan2(y, x) if is_casadi(y, x) else np.arctan2(y, x)


def fmax(a, b):
    return ca.fmax(a, b) if is_casadi(a, b) else np.maximum(a, b)


def if_else(cond, a, b):
    """Elementwise select: ``cond ? a : b``. CasADi ``if_else`` or NumPy ``where``."""
    if is_casadi(cond, a, b):
        return ca.if_else(cond, a, b)
    return np.where(cond, a, b)
