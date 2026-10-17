import hashlib

import json

import tarfile

from dataclasses import replace

from pathlib import Path

import pytest

import publish_ts_experiment_trajectories as publisher

def test_the_publishers_variant_blocks_mirror_the_runners():
    """The publisher restates the runners' block names (it cannot import torch); pinned here —
    the chain runner's and both manoeuvre-token readouts' (a lockstep directory published bare
    would otherwise be filed as the executor's own prediction; 2026-09-18)."""
    from ts_transformer.experiments import chain_sensitivity, executor_failure_modes, manoeuvre_lockstep

    assert chain_sensitivity.RECORDS_BLOCK in publisher.VARIANT_RECORD_BLOCKS
    # `manoeuvre_readout` is a MIRROR of `archive/manoeuvre_codes_2026_09/readout.RECORDS_BLOCK`
    # (2026-09-20: the intent-code readout is archived and cannot be imported), kept because the
    # categories published from it carry the block and the publisher must keep filing them.
    assert "manoeuvre_readout" in publisher.VARIANT_RECORD_BLOCKS
    assert manoeuvre_lockstep.LOCKSTEP_RECORDS_BLOCK in publisher.VARIANT_RECORD_BLOCKS
    assert executor_failure_modes.FAILURE_MODES_BLOCK in publisher.VARIANT_RECORD_BLOCKS
