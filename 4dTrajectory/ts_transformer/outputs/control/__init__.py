"""Everything specific to predicting a CONTROL SCHEDULE rather than a state sequence.

The path's one door for the spine is ``strategy.ControlStrategy`` (review §4.2); the role
is the submodule:

    strategy               what the spine asks of the path: the model, the objective, the
                           batch context, the probe, the epoch record, the forecast, the replay
    supervision            the per-flight physical context and the two supervision references
    forecast               the schedule rolled densely under the command hook; the CTA and
                           calibrated-interval plumbing; the latent decodes
    envelope               the dimensionless box the head predicts in
    heads                  the prediction contract and the output heads that emit a schedule
    latent                 the latent intent z on the control output (L2)
    basis_fit              batched shooting fit of a schedule to a known future (the fitted
                           teacher table)
    dynamics.backends      the flight models a schedule can be rolled through
    dynamics.rollout       the one rollout API training/forecast/evaluation share
    dynamics.inverse       the same models solved backwards, for targets
    dynamics.hooks         the command-hook contract the rollout calls per segment
    constraints.*          the command hooks (barrier, speed floor, trombone, their composite)
    loss.objective         the control objective assembled into `LossComponents`
    loss.components/fixed_dt   the tracking terms on the two supervision grids
    training.diagnostics   gradient diagnostics

The 2026-08 future-aware teachers that used to be ``control.oracle`` are a completed
campaign, archived under ``archive/oracle_teacher_2026_08/`` (off the import path) and
superseded by ``simple-v3``'s in-training imitation term.

**Membership rule**: a module belongs here only if EVERY consumer of it is control-specific.
``terminal_state_loss``, ``arc_length_geometry``, ``fixed_dt_supervision`` and ``flyability``
stay at the top level — ``fixed_anchor_validation`` and ``dataset`` share them with the state
path, so filing them here would claim an ownership that does not exist; the duration heads
both paths build live in ``outputs/duration_heads``. ``tests/test_architecture.py`` enforces
the rule, and the direction: only the strategy seam (``strategy``, ``forecast``,
``supervision``, ``loss.objective``) reaches the spine, and nothing here imports the loop.

Nothing is re-exported here on purpose: a package that flattens forty names back into one
namespace would restore exactly the undifferentiated listing this package exists to remove.
Import the submodule that owns what you need.
"""
