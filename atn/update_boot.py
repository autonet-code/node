"""Boot-time staged-update application — runs before any heavy imports.

The daemon's auto-updater (``nodes/common/updater.py``) only *stages* a
verified release wheel while the daemon runs; it never restarts the live
process. This module is the other half: at the very start of ``main()``,
before ``asyncio`` or any ``atn``/``nodes`` runtime module is imported, it
checks for a staged wheel newer than the running version, ``pip install``s
it into the current environment, and re-execs the process once into the new
code.

Deliberately import-light: stdlib only (os, sys, json, subprocess,
pathlib). Importing ``atn.*`` here would load the very modules a staged
update is meant to replace, defeating the point — so this file must stay
self-contained. See docs/auto_update_design.md.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# Mirror of nodes/common/updater.py constants — kept local so this module
# imports nothing from the package being updated.
_STAGED_DIRNAME = "staged_update"
_PENDING_MARKER = "pending.json"
# Set right before re-exec so the freshly-installed process doesn't loop
# back into another apply attempt.
_APPLIED_ENV = "ATN_UPDATE_APPLIED"


def _data_dir() -> Path:
    """The daemon data dir (~/.atn), honoring ATN_DATA_DIR if set.

    Mirrors atn.config's default without importing it.
    """
    env = os.environ.get("ATN_DATA_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".atn"


def _running_version() -> str:
    """Current installed version, via package metadata (no atn import)."""
    try:
        from importlib.metadata import version
        return version("autonet-computer")
    except Exception:
        return "0"


def dev_environment_reason() -> str:
    """Return a non-empty reason string if auto-update must be SKIPPED here.

    Auto-update is for end users who ``pip install autonet-computer`` and run
    the daemon from that installed package. It must NEVER touch a developer's
    working tree. We refuse to update when the code we'd be replacing isn't the
    code that's actually running, i.e.:

      - the package is an **editable / -e install** (site-packages points back
        at a source tree), or
      - ``atn`` imports from a directory that is NOT pip's install location
        (the classic ``python -m atn`` from a git checkout, where cwd shadows
        site-packages — exactly the dev-machine case), or
      - that import directory lives inside a git working tree.

    Returns "" when it's a normal installed environment and updating is safe.
    Import-light: only importlib + os/sys/pathlib.
    """
    # 1. Editable install marker on the distribution.
    try:
        import json as _json
        from importlib.metadata import distribution
        dist = distribution("autonet-computer")
        durl = dist.read_text("direct_url.json")
        if durl:
            info = _json.loads(durl)
            if info.get("dir_info", {}).get("editable"):
                return "editable (-e) install"
    except Exception:
        pass

    # 2. Where does `atn` actually import from, vs where is it installed?
    try:
        import importlib.util
        spec = importlib.util.find_spec("atn")
        if spec and spec.origin:
            import_dir = Path(spec.origin).resolve().parent  # .../atn
            pkg_root = import_dir.parent                      # the dir on sys.path
            # 2a. Inside a git working tree → developer checkout.
            probe = import_dir
            for _ in range(6):
                if (probe / ".git").exists():
                    return f"running from a git working tree ({pkg_root})"
                if probe.parent == probe:
                    break
                probe = probe.parent
            # 2b. Import dir differs from pip's install location → source tree
            #     shadowing site-packages (e.g. `python -m atn` from project root).
            try:
                from importlib.metadata import distribution
                dist = distribution("autonet-computer")
                installed_root = Path(dist.locate_file("")).resolve()
                # If the importable atn/ is not under pip's install root, the
                # running code is not the installed package.
                if installed_root not in import_dir.resolve().parents \
                        and installed_root != pkg_root:
                    return (
                        f"running code ({pkg_root}) is not the installed "
                        f"package ({installed_root})"
                    )
            except Exception:
                pass
    except Exception:
        pass

    return ""


def _parse_version(v: str) -> tuple:
    try:
        from packaging.version import Version
        return (1, Version(v))
    except Exception:
        parts = []
        for seg in str(v).replace("-", ".").split("."):
            num = "".join(ch for ch in seg if ch.isdigit())
            parts.append(int(num) if num else 0)
        return (0, tuple(parts))


def _is_newer(candidate: str, current: str) -> bool:
    if not candidate or candidate == current:
        return False
    try:
        c, cur = _parse_version(candidate), _parse_version(current)
        if c[0] == cur[0]:
            return c[1] > cur[1]
        return candidate != current
    except Exception:
        return candidate != current


def apply_staged_update_if_any() -> None:
    """Apply a staged update and re-exec, if one is pending and newer.

    Safe to call unconditionally at process start. No-ops when:
      - we've already applied an update in this process lineage
        (``ATN_UPDATE_APPLIED`` set), or
      - no ``pending.json`` marker exists, or
      - the staged version isn't newer than the running version.

    On a usable marker: ``pip install`` the staged wheel, clear the marker
    (whether install succeeds OR fails — a bad wheel must not be retried
    every boot), and on success re-exec the process once.
    """
    # Re-exec loop guard: only attempt one apply per process lineage.
    if os.environ.get(_APPLIED_ENV) == "1":
        return

    # Never overwrite a developer's working tree / editable install. If the
    # running code isn't the pip-installed package, refuse to apply.
    reason = dev_environment_reason()
    if reason:
        sys.stderr.write(
            f"[auto-update] skipping update apply: {reason}. "
            "Auto-update only manages pip-installed daemons.\n"
        )
        return

    pending_path = _data_dir() / _STAGED_DIRNAME / _PENDING_MARKER
    try:
        if not pending_path.exists():
            return
        marker = json.loads(pending_path.read_text(encoding="utf-8"))
    except Exception:
        # Unreadable marker — remove it and carry on with the running version.
        _safe_unlink(pending_path)
        return

    version = str(marker.get("version", ""))
    wheel_path = marker.get("wheel_path", "")
    current = _running_version()

    if not version or not wheel_path or not _is_newer(version, current):
        # Stale or not-newer marker (e.g. we already updated past it). Clear it.
        _safe_unlink(pending_path)
        return

    wheel = Path(wheel_path)
    if not wheel.exists():
        sys.stderr.write(
            f"[auto-update] staged wheel missing ({wheel_path}); clearing marker\n"
        )
        _safe_unlink(pending_path)
        return

    sys.stderr.write(
        f"[auto-update] applying staged update {current} -> {version} ...\n"
    )
    sys.stderr.flush()

    # Clear the marker BEFORE installing: a wheel that fails to install must
    # not re-trigger on every subsequent boot. (Idempotent re-stage will
    # re-create it on the next poll if the release is still newer.)
    _safe_unlink(pending_path)

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade",
             "--no-input", str(wheel)],
            capture_output=True, text=True, timeout=600,
        )
    except Exception as exc:
        sys.stderr.write(f"[auto-update] pip install errored: {exc}\n")
        return

    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-5:]
        sys.stderr.write(
            "[auto-update] pip install failed; staying on "
            f"{current}:\n  " + "\n  ".join(tail) + "\n"
        )
        return

    # Success — re-exec once into the new code. Set the guard first so the
    # replacement process doesn't loop.
    sys.stderr.write(f"[auto-update] installed {version}; restarting daemon\n")
    sys.stderr.flush()
    os.environ[_APPLIED_ENV] = "1"
    try:
        # Best-effort cleanup of the staged wheel dir for this version.
        _cleanup_version_dir(wheel)
    except Exception:
        pass
    os.execv(sys.executable, [sys.executable, *sys.argv])


def _safe_unlink(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def _cleanup_version_dir(wheel: Path) -> None:
    """Remove the staged <version>/ dir containing the applied wheel."""
    import shutil
    parent = wheel.parent
    if parent.name and parent.parent.name == _STAGED_DIRNAME:
        shutil.rmtree(parent, ignore_errors=True)
