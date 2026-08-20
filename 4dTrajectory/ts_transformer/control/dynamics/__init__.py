"""Flight models: rolling a control schedule forward, and solving one backwards.

``backends`` registers the forward models under ``(control_dynamics_model,
control_dynamics_backend)``; ``inverse`` registers their algebraic inverses under the SAME
model key, which is what stops a teacher being solved against equations the training rollout
does not integrate. ``rollout`` is the single entry point training, forecast and evaluation
all go through.
"""
