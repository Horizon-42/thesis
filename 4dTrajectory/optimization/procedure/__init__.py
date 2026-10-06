"""Approach procedures for the optimizer.

  * :mod:`.constraint` — :class:`ProcedureConstraint`, the canonical procedure (the Python side of the
    frontend's ``procedureConstraint.ts``), parsed from a request or a procedure detail document.
  * :mod:`.segments` — ``build_constraint_segments``: a procedure as ``approach_constraints`` legs,
    anchored at the optimizer target (the LTP).
  * :mod:`.iaf` — the IAF→runway paths of a procedure document and their length.

Dependency direction: backend → {procedure, collocation}; procedure → approach_constraints,
flight_scenarios.fas_geometry, geokit; nothing here imports the backend.
"""
