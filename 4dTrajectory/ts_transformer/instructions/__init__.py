"""The second layer's language: the instruction vocabulary, the per-step signals it is read
from, the envelopes, the labeller and the sentence artefact.

Design: ``docs/2026-09-23_instruction_vocabulary_design.zh.md`` (the words) and
``docs/2026-09-23_two_tier_framework.zh.md`` (where this group sits). Torch-free; a group's
``__init__`` re-exports nothing (layout rule L2).
"""
