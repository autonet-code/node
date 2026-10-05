"""Daemon singleton lock: a reused pid in a stale lock file must not block start.

Container case: the daemon runs as the same small pid under tini on every
start, so after a hard stop the lock file names a pid that is alive again
(often the new daemon itself). Only the OS lock is authoritative.
"""
import json
import os
import sys

import pytest

from atn.lock_manager import LockManager


@pytest.fixture
def lock(tmp_path, monkeypatch):
    monkeypatch.setattr(LockManager, "_instance", None)
    lm = LockManager()
    lm.set_data_dir(tmp_path)
    yield lm
    lm.release_lock()


def _write_info(path, pid):
    path.write_text(json.dumps({"pid": pid, "started_at": "2026-01-01T00:00:00"}))


def _os_lock(fh):
    if sys.platform == "win32":
        import msvcrt
        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_stale_file_with_live_reused_pid_is_not_running(lock):
    _write_info(lock.lock_file, os.getpid())     # pid alive, nobody holds the lock
    assert lock.is_daemon_running() is None
    assert lock.acquire_lock() is True


def test_held_os_lock_is_running(lock, tmp_path):
    _write_info(lock.lock_file, os.getpid())
    with open(lock.lock_file, "r+") as fh:
        _os_lock(fh)
        assert lock._lock_is_held() is True
        if sys.platform != "win32":   # Windows refuses the read of a locked byte
            info = lock.is_daemon_running()
            assert info is not None and info["pid"] == os.getpid()
        assert lock.acquire_lock() is False
