"""The guidance layer: a deterministic controller that flies a plan inside the skeleton
(design §4), through the shared point-mass rollout's command hook.

    route          the route builder — a plan's join, side and pre-final length laid as a
                   polyline in the chart (Dubins turns at the aircraft's turn radius, a
                   dog-leg for the length the shortest path does not use, the final leg),
                   and the time the speed schedule needs to fly it
    controller     `PlanGuidance`, the `CommandHook`: per rollout segment, the bank that
                   tracks the route, the load factor that flies the height profile, the
                   thrust that holds the speed schedule — every command inside the
                   flyability envelope and above the stall margin, so the trajectory
                   conforms by construction

Nothing here composes the barrier / speed-floor / trombone hooks after the fact: their
laws (the lagged bank demand, the thrust inversion, the speed floor) are used as parts of
one controller.
"""
