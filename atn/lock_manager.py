"""
Lock Manager for ATN Daemon

Ensures only one ATN daemon can run at a time using OS-level file locking.
Cross-platform support for Windows (msvcrt) and Unix (fcntl).

Ported from the Chevin daemon lock_manager pattern.
"""

import json
import logging
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import psutil

log = logging.getLogger(__name__)

_LOCK_FILENAME = "daemon.lock"


class LockManager:
    """
    Singleton daemon enforcement via OS-level file locking.

    Uses platform-specific file locking so the OS automatically releases the
    lock when the process dies — no stale-pidfile problem.

    The lock file contains JSON with process info for diagnostics.
    """

    _instance: Optional["LockManager"] = None
    _cls_lock = threading.Lock()

    def __new__(cls) -> "LockManager":
        if cls._instance is None:
            with cls._cls_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._lock_file_handle = None
        self._is_locked = False
        self._started_at: Optional[datetime] = None
        self._data_dir: Optional[Path] = None

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    def set_data_dir(self, data_dir: Path) -> None:
        self._data_dir = data_dir

    @property
    def lock_file(self) -> Path:
        if self._data_dir is None:
            raise RuntimeError("LockManager.set_data_dir() must be called before use")
        return self._data_dir / _LOCK_FILENAME

    # ------------------------------------------------------------------
    # Lock info helpers
    # ------------------------------------------------------------------

    def _write_lock_info(self) -> None:
        if self._lock_file_handle:
            self._lock_file_handle.seek(0)
            self._lock_file_handle.truncate()
            info = {
                "pid": os.getpid(),
                "started_at": self._started_at.isoformat() if self._started_at else None,
                "hostname": os.environ.get(
                    "COMPUTERNAME", os.environ.get("HOSTNAME", "unknown")
                ),
                "python_version": sys.version,
            }
            self._lock_file_handle.write(json.dumps(info, indent=2))
            self._lock_file_handle.flush()

    def _read_lock_info(self) -> Optional[dict]:
        if not self.lock_file.exists():
            return None
        try:
            with open(self.lock_file, "r") as f:
                content = f.read().strip()
                if content:
                    return json.loads(content)
        except (json.JSONDecodeError, IOError):
            pass
        return None

    @staticmethod
    def _is_process_alive(pid: int) -> bool:
        try:
            proc = psutil.Process(pid)
            return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

    # ------------------------------------------------------------------
    # Stale-lock cleanup
    # ------------------------------------------------------------------

    def _cleanup_stale_lock(self) -> None:
        info = self._read_lock_info()
        if not info:
            return
        pid = info.get("pid")
        if pid and not self._is_process_alive(pid):
            log.info("Removing stale lock from dead PID %d", pid)
            try:
                self.lock_file.unlink()
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def acquire_lock(self) -> bool:
        """
        Try to acquire the exclusive daemon lock.

        Returns True on success, False if another instance holds the lock.
        """
        if self._is_locked:
            return True

        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._cleanup_stale_lock()

        try:
            self._lock_file_handle = open(self.lock_file, "w+")

            if sys.platform == "win32":
                import msvcrt

                try:
                    msvcrt.locking(
                        self._lock_file_handle.fileno(), msvcrt.LK_NBLCK, 1
                    )
                except IOError:
                    self._lock_file_handle.close()
                    self._lock_file_handle = None
                    return False
            else:
                import fcntl

                try:
                    fcntl.flock(
                        self._lock_file_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                    )
                except IOError:
                    self._lock_file_handle.close()
                    self._lock_file_handle = None
                    return False

            self._is_locked = True
            self._started_at = datetime.now()
            self._write_lock_info()
            return True

        except Exception as exc:
            if self._lock_file_handle:
                self._lock_file_handle.close()
                self._lock_file_handle = None
            raise RuntimeError(f"Failed to acquire daemon lock: {exc}") from exc

    def release_lock(self) -> None:
        """Release the daemon lock on shutdown."""
        if not self._is_locked or not self._lock_file_handle:
            return

        try:
            if sys.platform == "win32":
                import msvcrt

                try:
                    msvcrt.locking(
                        self._lock_file_handle.fileno(), msvcrt.LK_UNLCK, 1
                    )
                except IOError:
                    pass
            else:
                import fcntl

                try:
                    fcntl.flock(self._lock_file_handle.fileno(), fcntl.LOCK_UN)
                except IOError:
                    pass

            self._lock_file_handle.close()
            self._lock_file_handle = None
            self._is_locked = False

            try:
                self.lock_file.unlink()
            except OSError:
                pass
        except Exception:
            pass

    def is_daemon_running(self) -> Optional[dict]:
        """
        Check if another daemon instance is running.

        Returns dict with PID / start time if running, None otherwise.
        """
        if self._is_locked:
            return None  # We hold the lock ourselves

        info = self._read_lock_info()
        if not info:
            return None

        pid = info.get("pid")
        if pid and self._is_process_alive(pid):
            return {
                "pid": pid,
                "started_at": info.get("started_at"),
                "hostname": info.get("hostname"),
            }
        return None

    def force_release_stale_lock(self) -> bool:
        """
        Force-remove a lock left by a dead process.

        Returns True if removed (or no lock file), False if the process is alive.
        """
        info = self._read_lock_info()
        if not info:
            return True

        pid = info.get("pid")
        if pid and self._is_process_alive(pid):
            return False

        try:
            self.lock_file.unlink()
            return True
        except OSError:
            return False

    def get_lock_info(self) -> Optional[dict]:
        if self._is_locked:
            return {
                "pid": os.getpid(),
                "started_at": self._started_at.isoformat() if self._started_at else None,
                "is_self": True,
            }
        return self._read_lock_info()

    @property
    def is_locked(self) -> bool:
        return self._is_locked

    @property
    def started_at(self) -> Optional[datetime]:
        return self._started_at
