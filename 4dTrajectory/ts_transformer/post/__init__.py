"""The post-training and the multi-aircraft work (`docs/two_tier/design/post_training.md`; stage C): the scene and its
steps, the edge features, "established", the separation masks, the traffic module, the reward, the branch groups and
the loss.

It reads the instruction language (`instructions/`, through vocabulary §6), the separation rules
(`inference.separation`, `inference.runway_schedule`) and from the prior only the names of prior §7; it never imports
the executor (`autopilot/`): the window loop and the runners under `experiments/` join them (post-training §8 C0,
`tests/test_architecture.py`).
"""
