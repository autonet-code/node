"""The tools this daemon has pruned (Decision 2026-11-10).

The close lists, in its carried state, every tool with an uncleared malware
verdict (the reviewers' or the DAO's), every tool built on one, and every
registration it refused for importing one (``tool_rounds["pruned"]``).
Daemons stop circulating them: the library drops their manifests from the
index and deletes their blobs, the blob handlers refuse to serve them, the
fetch paths refuse to pull them, and adoption refuses them.

A household keeps what it already installed: the tool store registers a
keeper, and a blob it keeps is not deleted (what to run stays the
household's policy). It is still never served to peers.

The set holds manifest digests and the code digests they name. It is
process-wide because the blob handlers, the library and the tool store are
separate objects that each need it; the library persists it.
"""

from __future__ import annotations

import threading
from typing import Callable, Iterable, List, Set

_lock = threading.Lock()
_pruned: Set[str] = set()
_keepers: List[Callable[[str], bool]] = []


def _norm(digest: str) -> str:
    return str(digest or "").lower().removeprefix("0x")


def replace(digests: Iterable[str]) -> None:
    """Make ``digests`` the pruned set (a DAO clearing removes a tool)."""
    new = {_norm(d) for d in digests if _norm(d)}
    with _lock:
        _pruned.clear()
        _pruned.update(new)


def is_pruned(digest: str) -> bool:
    with _lock:
        return _norm(digest) in _pruned


def snapshot() -> Set[str]:
    with _lock:
        return set(_pruned)


def add_keeper(keeper: Callable[[str], bool]) -> None:
    """Register ``keeper(digest) -> bool``: True keeps a blob on disk."""
    with _lock:
        if keeper not in _keepers:
            _keepers.append(keeper)


def kept(digest: str) -> bool:
    with _lock:
        keepers = list(_keepers)
    for k in keepers:
        try:
            if k(_norm(digest)):
                return True
        except Exception:  # noqa: BLE001 - a keeper never breaks pruning
            continue
    return False
