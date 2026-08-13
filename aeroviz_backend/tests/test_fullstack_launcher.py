from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPOSITORY_ROOT / "start_aeroviz_fullstack.sh"


def _wait_until(predicate, *, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("condition was not met before timeout")


def _stop(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _make_launcher_fixture(tmp_path: Path) -> tuple[Path, dict[str, str], Path]:
    root = tmp_path / "fixture"
    app = root / "aeroviz-4d"
    (app / "public" / "data").mkdir(parents=True)
    (root / "aeroviz_backend").mkdir()

    launcher = root / LAUNCHER.name
    launcher.write_bytes(LAUNCHER.read_bytes())
    launcher.chmod(0o755)

    fake_bin = root / "fake-bin"
    fake_bin.mkdir()
    service = fake_bin / "service"
    service.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s %s\\n" "$(basename "$0")" "$$" >> "$AEROVIZ_TEST_SERVICE_LOG"\n'
        "trap 'exit 0' INT TERM\n"
        "while true; do sleep 0.1; done\n",
        encoding="utf-8",
    )
    service.chmod(0o755)
    (fake_bin / "python").symlink_to(service)
    (fake_bin / "npm").symlink_to(service)

    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    service_log = tmp_path / "services.log"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "PYTHON_BIN": str(fake_bin / "python"),
            "XDG_RUNTIME_DIR": str(runtime_dir),
            "AEROVIZ_TEST_SERVICE_LOG": str(service_log),
            "AEROVIZ_MIN_HEALTHY_S": "30",
        }
    )
    return launcher, env, service_log


def test_replace_stops_only_the_recorded_previous_supervisor(tmp_path: Path) -> None:
    launcher, env, service_log = _make_launcher_fixture(tmp_path)
    first = subprocess.Popen(
        [str(launcher)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    second: subprocess.Popen[str] | None = None
    duplicate: subprocess.Popen[str] | None = None
    try:
        _wait_until(
            lambda: service_log.exists()
            and len(service_log.read_text(encoding="utf-8").splitlines()) >= 2
        )

        duplicate = subprocess.Popen(
            [str(launcher)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        duplicate_output, _ = duplicate.communicate(timeout=5)
        assert duplicate.returncode != 0
        assert "already running" in duplicate_output
        assert first.poll() is None

        second = subprocess.Popen(
            [str(launcher), "--replace"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        _wait_until(lambda: first.poll() is not None)
        _wait_until(
            lambda: len(service_log.read_text(encoding="utf-8").splitlines()) >= 4
        )
        assert second.poll() is None
    finally:
        if duplicate is not None:
            _stop(duplicate)
        if second is not None:
            _stop(second)
        _stop(first)
    service_pids = [
        int(line.rsplit(" ", 1)[1])
        for line in service_log.read_text(encoding="utf-8").splitlines()
    ]
    _wait_until(lambda: not any(_pid_exists(pid) for pid in service_pids))
