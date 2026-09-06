"""What a predicted schedule is scored against.

``components`` assembles the tracking objectives (position, velocity, imitation, terminal),
``terminal_clock`` chooses which rollout endpoint the terminal term reads, and ``fixed_dt`` is
the fixed-physical-time supervision variant.
"""
