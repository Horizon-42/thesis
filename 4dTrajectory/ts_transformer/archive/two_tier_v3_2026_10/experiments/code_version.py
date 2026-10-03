"""A readout's code version (multi-aircraft design §6.6 step 9.9.1): what read it, apart from what it was asked to read
(its configuration) and what it read (its data) — the commit and whether the checkout was clean, the Python, torch and
CUDA it ran on, the GPU and the device, and the code's own constants that decide how it reads.

Two readouts' code versions are compared whole (`traffic_window_compare`): equal, or a conformance record
(`traffic_window_conformance`) shows the one readout reads the same under the other's code.
"""

from __future__ import annotations

import platform
from typing import Any, Mapping

import torch

#: `code_version`'s keys, in order.
KEYS = ("commit", "dirty", "python", "torch", "cuda", "gpu", "device", "constants")


def code_version(git: Mapping[str, Any], device: torch.device, constants: Mapping[str, Any]) -> dict[str, Any]:
    """The code version of a run whose checkout was ``git`` (`repo_layout.git_state`, taken when it started) on
    ``device``, reading by ``constants``. Asks the GPU's name, so in a program that forks reading processes call it
    once they have ended (the parent never starts CUDA before)."""
    cuda = device.type == "cuda"
    return {"commit": git["head"], "dirty": git["dirty"], "python": platform.python_version(),
            "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device) if cuda
            else None, "device": device.type, "constants": dict(constants)}

