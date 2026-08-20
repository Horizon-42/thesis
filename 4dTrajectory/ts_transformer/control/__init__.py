"""Everything specific to predicting a CONTROL SCHEDULE rather than a state sequence.

These modules used to sit at the top level behind a ``control_`` prefix, which named the
subject but not the role, so the loss, the flight models, the head and the teacher were
interleaved with the shared modules in one flat listing. The prefix is now the package and
the role is the submodule:

    control.envelope           the dimensionless box the head predicts in
    control.heads              the output heads that emit a schedule
    control.duration           the uniform segment-duration parameterization
    control.dynamics.backends  the flight models a schedule can be rolled through
    control.dynamics.rollout   the one rollout API training/forecast/evaluation share
    control.dynamics.inverse   the same models solved backwards, for teachers and targets
    control.loss.*             tracking objectives, terminal clocks, regularizers
    control.training.*         curriculum views and gradient diagnostics
    control.oracle.*           future-aware teachers (direct shooting + the imitation path)

**Membership rule**: a module belongs here only if EVERY consumer of it is control-specific.
That is why ``prediction_outputs`` (it holds ``StatePrediction`` too), ``terminal_state_loss``,
``arc_length_geometry``, ``fixed_dt_supervision`` and ``flyability`` stay at the top level —
``fixed_anchor_validation`` and ``dataset`` share them with the state path, so filing them
under ``control`` would claim an ownership that does not exist. ``tests/test_architecture.py``
enforces both halves of the rule.

Nothing is re-exported here on purpose: a package that flattens forty names back into one
namespace would restore exactly the undifferentiated listing this package exists to remove.
Import the submodule that owns what you need.
"""
