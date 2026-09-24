"""Stage 3 of the two-tier framework: the executor, an autopilot that flies the instruction words
through the point-mass dynamics (``docs/2026-09-23_executor_design.zh.md``).

Once per control cycle it reads the aircraft's state and the words in force, turns the words into
the rates it wants, inverts the point-mass equations for the controls that produce them, and
integrates the cycle with the control path's own dynamics. It reads no procedure and learns
nothing.

- ``frame``    the dynamics' state read the way the words read a flight (compass, airport frame)
- ``sentence`` which word of each column is in force at each cycle
- ``flights``  a labelled flight's inputs, rebuilt from the data plane and checked row for row
- ``plant``    one control cycle of the dynamics
- ``inverse``  wanted rates → bank, load factor, thrust; the limits, in their order
- ``lateral`` / ``vertical`` / ``speed``  the three laws: words in force → the rates wanted
- ``params``   the executor's own parameters and the design's constraints on them
- ``executor`` the cycle loop (``fly``); ``judge`` the three-layer verdict on what was flown
- ``replay``   a batch of labelled flights: who is flown, flying and judging them
- ``derive``   the executor's own parameters, from the vocabulary alone
- ``spec``     the executor spec on disk, written once
"""
