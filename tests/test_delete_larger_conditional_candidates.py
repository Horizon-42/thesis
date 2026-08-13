from __future__ import annotations

import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "delete_larger_conditional_candidates.sh"


def _mount_is_rejected(mountpoint: str, target: str) -> bool:
    command = (
        f'source "{SCRIPT}"; '
        f'mount_is_at_or_inside_target "{mountpoint}" "{target}"'
    )
    return subprocess.run(["bash", "-c", command], check=False).returncode == 0


def test_mount_guard_rejects_target_itself_and_descendants() -> None:
    target = "/home/supercomputing/works/build_mesh2/.deps"

    assert _mount_is_rejected(target, target)
    assert _mount_is_rejected(f"{target}/nested", target)
    assert not _mount_is_rejected(f"{target}-other", target)
