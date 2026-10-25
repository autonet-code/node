"""The robot's worklog as a signed hash chain (docs/robots.md, 2026-10-25).

The recorder writes MCAP chunks into one directory per stream
(``<bag_dir>/<stream>/...``), so each stream can be thinned on its own
schedule. Every closed chunk becomes a leaf ``{stream, name, sha256, bytes}``.
A **link** commits the leaves closed since the previous link under one
merkle root, is signed by the robot's agent key and carries the previous
link's hash, so the whole worklog is one chain: no chunk can be dropped or
inserted between links.

The robot commits a link every time chunks close and pushes it to its
witnesses and pollers (continuous commitment). A poll answer is a link that
also carries the poll's challenge. The robot keeps its links and chunks; a
dispute opens the chunks, which must hash to the leaves their link signed.
A thinned (deleted) chunk only removes evidence, it never breaks the chain.

The chunk still being written is left out: its bytes are still changing.
So is a chunk the recorder has moved past but not yet finished (it flushes
the old file after opening the next): a chunk counts as closed only once it
ends with the MCAP magic, which the writer appends last.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable

from . import crypto as C

LINK_TYPE = "autonet.robot.link/1"
CHUNK_SUFFIX = ".mcap"
MCAP_MAGIC = b"\x89MCAP0\r\n"
MAIN_STREAM = "main"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def leaf_hash(leaf: dict[str, Any]) -> str:
    return C.sha256_hex(json.dumps(leaf, sort_keys=True, separators=(",", ":")).encode())


def _finished(path: Path) -> bool:
    try:
        size = path.stat().st_size
        if size < len(MCAP_MAGIC):
            return False
        with path.open("rb") as f:
            f.seek(size - len(MCAP_MAGIC))
            return f.read() == MCAP_MAGIC
    except OSError:
        return False


def closed_chunks(bag_dir: str | Path) -> list[tuple[str, str, Path]]:
    """(stream, name, path) of every closed chunk, oldest first per stream.
    A stream is the first directory under ``bag_dir`` (files directly in it
    are the ``main`` stream); its newest chunk is the one being written, and
    an older one is closed once the writer finished it. Chunks are taken in
    order, so an unfinished one holds back the ones after it."""
    root = Path(bag_dir)
    by_stream: dict[str, list[Path]] = {}
    for p in root.rglob(f"*{CHUNK_SUFFIX}"):
        rel = p.relative_to(root)
        stream = rel.parts[0] if len(rel.parts) > 1 else MAIN_STREAM
        by_stream.setdefault(stream, []).append(p)
    out = []
    for stream in sorted(by_stream):
        files = sorted(by_stream[stream], key=lambda p: (p.stat().st_mtime, p.name))
        for p in files[:-1]:
            if not _finished(p):
                break
            out.append((stream, p.relative_to(root).as_posix(), p))
    return out


def build_link(*, robot: str, seq: int, prev: str, leaves: list[dict[str, Any]],
               packs: list[str], challenge: str = "", poll: str = "",
               scene: list[dict[str, Any]] | None = None,
               now: float | None = None) -> dict[str, Any]:
    """``scene``: the robot's scene samples since its previous link
    (``atn/robot/scene.py``), signed with the rest so witnesses can check
    them against what they saw."""
    return {"type": LINK_TYPE, "robot": robot.lower(), "seq": seq, "prev": prev,
            "ts": time.time() if now is None else now, "packs": list(packs),
            "challenge": challenge, "poll": poll, "leaves": leaves,
            "root": C.merkle_root([leaf_hash(lf) for lf in leaves]),
            "scene": list(scene or [])}


def verify_link(link: dict[str, Any], *, robot: str,
                prev: dict[str, Any] | None = None) -> list[str]:
    """Why the link fails (empty = passes). With ``prev``, also that it
    follows ``prev`` directly in the chain."""
    errors = []
    if link.get("type") != LINK_TYPE:
        errors.append("wrong link type")
    try:
        if C.signer(link) != robot.lower():
            errors.append("signed by a different key")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"bad signature: {exc}")
    leaves = link.get("leaves") or []
    if link.get("root") != C.merkle_root([leaf_hash(lf) for lf in leaves]):
        errors.append("root does not match the leaves")
    if prev is not None:
        if link.get("seq") != prev.get("seq", -1) + 1:
            errors.append(f"seq {link.get('seq')} does not follow {prev.get('seq')}")
        if link.get("prev") != C.obj_hash(prev):
            errors.append("prev does not match the previous link")
    elif link.get("seq") == 0 and link.get("prev") != C.ZERO:
        errors.append("first link must have a zero prev")
    return errors


def verify_chain(links: list[dict[str, Any]], *, robot: str) -> list[str]:
    """A whole chain, or a tail of one (its first link is taken as given)."""
    errors = []
    for i, link in enumerate(links):
        errs = verify_link(link, robot=robot, prev=links[i - 1] if i else None)
        errors += [f"link {link.get('seq')}: {e}" for e in errs]
    return errors


def verify_chunks(link: dict[str, Any], files: dict[str, bytes]) -> list[str]:
    """Check produced chunks against the leaves the link signed. Leaves with
    no produced file are thinned or withheld; the caller decides which."""
    errors = []
    for leaf in link.get("leaves") or []:
        data = files.get(leaf["name"])
        if data is None:
            continue
        if hashlib.sha256(data).hexdigest() != leaf["sha256"]:
            errors.append(f"chunk {leaf['name']} does not match its leaf")
    return errors


def challenge_recorded(challenge: str, files: Iterable[bytes]) -> bool:
    """Whether the challenge string appears in any of the chunks. The
    recorder writes uncompressed chunks, and a 64-hex challenge cannot be
    guessed, so a byte match is proof the robot recorded it after the poll."""
    needle = challenge.encode()
    return any(needle in data for data in files)


class Worklog:
    """The robot's own chain: links on disk (``links.jsonl``), chunks in
    ``bag_dir``."""

    def __init__(self, bag_dir: str | Path, state_dir: str | Path,
                 robot: str, private_key: str) -> None:
        self.bag_dir = Path(bag_dir)
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.robot = robot.lower()
        self._key = private_key
        self._path = self.state_dir / "links.jsonl"
        self.links: list[dict[str, Any]] = []
        if self._path.exists():
            for line in self._path.read_text("utf-8").splitlines():
                if line.strip():
                    self.links.append(json.loads(line))
        self._committed = {lf["name"] for lk in self.links for lf in lk["leaves"]}

    @property
    def scene_until(self) -> float:
        """Time of the newest scene sample already committed (0 = none)."""
        for link in reversed(self.links):
            if link.get("scene"):
                return max(s["t"] for s in link["scene"])
        return 0.0

    @property
    def head(self) -> dict[str, Any] | None:
        return self.links[-1] if self.links else None

    def pending_leaves(self) -> list[dict[str, Any]]:
        return [{"stream": s, "name": n, "sha256": _sha256_file(p),
                 "bytes": p.stat().st_size}
                for s, n, p in closed_chunks(self.bag_dir) if n not in self._committed]

    def commit(self, *, packs: list[str], challenge: str = "", poll: str = "",
               scene: list[dict[str, Any]] | None = None,
               force: bool = False, now: float | None = None) -> dict[str, Any] | None:
        """Sign a link over the chunks closed since the last one. Returns
        None when nothing closed, unless ``force`` (a poll answer always
        commits, possibly with no new leaves)."""
        leaves = self.pending_leaves()
        if not leaves and not force:
            return None
        head = self.head
        link = C.sign(build_link(
            robot=self.robot, seq=(head["seq"] + 1) if head else 0,
            prev=C.obj_hash(head) if head else C.ZERO, leaves=leaves,
            packs=packs, challenge=challenge, poll=poll, scene=scene, now=now), self._key)
        self.links.append(link)
        self._committed.update(lf["name"] for lf in leaves)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(link, sort_keys=True) + "\n")
        return link

    def links_since(self, seq: int) -> list[dict[str, Any]]:
        return [lk for lk in self.links if lk["seq"] > seq]

    def produce(self, names: Iterable[str]) -> dict[str, bytes]:
        """Chunk bytes for a dispute; thinned chunks are simply absent."""
        out = {}
        for name in names:
            p = self.bag_dir / name
            if p.exists():
                out[name] = p.read_bytes()
        return out

    def prune(self, retention_s: dict[str, float], now: float | None = None) -> list[str]:
        """Thin committed chunks older than their stream's retention. A
        stream not listed is kept. Links are never pruned."""
        now = time.time() if now is None else now
        removed = []
        for link in self.links:
            for leaf in link["leaves"]:
                keep = retention_s.get(leaf["stream"])
                if keep is None or now - link["ts"] <= keep:
                    continue
                p = self.bag_dir / leaf["name"]
                if p.exists():
                    p.unlink()
                    removed.append(leaf["name"])
        return removed
