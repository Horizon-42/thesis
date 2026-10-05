"""Multi-aircraft optimization: one optimized aircraft in recorded traffic (M1), a scheduled block (M2).

Design: ``4dTrajectory/docs/multi_aircraft_optimization/design.md``.

  * :mod:`.scene` — the recorded traffic of an airport (the arrivals manifest, MSL) and a window's
    aircraft;
  * :mod:`.frame` — one metric frame per window: the commanded aircraft's target frame;
  * :mod:`.runways` — the per-runway inputs the separation rules need (approach clock, offsets,
    "established");
  * :mod:`.rules` — the ONLY module that imports ``ts_transformer`` (the separation judge, read-only);
  * :mod:`.rows` — a judged loss as smooth NLP rows;
  * :mod:`.loop` — the M1 loop: baseline solve → replay → judge → rows → re-solve.
"""
