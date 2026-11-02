"""Tools as codebases (docs/tool_codebases.md).

A tool's codebase lives anywhere on the machine, with an
``autonet-tool.json`` at its root. Registering it snapshots the tree:

- the **tree digest** is sha256 over the canonical file list, one line per
  file: ``<relative path, forward slashes> NUL <sha256 of the file> LF``,
  sorted by path. It names the content, independent of any archive format,
  so a copy rebuilt from anywhere (a peer, a git host) is checked the same
  way;
- the **archive** is a zip of exactly those files (sorted, fixed
  timestamps). Its own sha256 is the manifest's ``code_digest``: the blob
  peers already move and verify.

What runs is always an extracted snapshot, in a directory of its own,
re-verified against the tree digest before every run.
"""
from __future__ import annotations

import fnmatch
import hashlib
import io
import json
import os
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

MANIFEST_FILE = "autonet-tool.json"
MAX_ARCHIVE_BYTES = 16 * 1024 * 1024
MAX_FILES = 2000
_ALWAYS_EXCLUDED = (".git", ".git/*", "*/.git", "*/.git/*", "__pycache__",
                    "*/__pycache__", "*/__pycache__/*", "__pycache__/*",
                    "*.pyc")
_ZIP_TIME = (1980, 1, 1, 0, 0, 0)

# Fields a codebase manifest may carry, besides the required ones.
MANIFEST_FIELDS = ("name", "description", "input_schema", "output_schema",
                   "entrypoint", "runtime", "capabilities", "dependencies",
                   "novelty", "exclude", "source")


class TreeError(ValueError):
    pass


def read_manifest(root: Path) -> dict[str, Any]:
    path = Path(root) / MANIFEST_FILE
    if not path.is_file():
        raise TreeError(f"no {MANIFEST_FILE} at {root}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise TreeError(f"{MANIFEST_FILE} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise TreeError(f"{MANIFEST_FILE} must be a JSON object")
    unknown = sorted(set(data) - set(MANIFEST_FIELDS))
    if unknown:
        raise TreeError(f"{MANIFEST_FILE}: unknown fields {unknown}")
    for key in ("name", "description", "entrypoint"):
        if not str(data.get(key) or "").strip():
            raise TreeError(f"{MANIFEST_FILE}: '{key}' is required")
    if not isinstance(data.get("input_schema"), dict):
        raise TreeError(f"{MANIFEST_FILE}: 'input_schema' must be an object")
    src = data.get("source")
    if src is not None:
        if not isinstance(src, dict) or not str(src.get("repo") or "").startswith("https://") \
                or not str(src.get("commit") or "").strip():
            raise TreeError(f"{MANIFEST_FILE}: 'source' is "
                            "{\"repo\": \"https://...\", \"commit\": \"<sha>\"}")
    return data


def _excluded(rel: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(rel, p) for p in patterns)


def list_files(root: Path, exclude: list[str] | None = None, *,
               strict: bool = False) -> list[str]:
    """Relative paths (forward slashes, sorted) of the files a snapshot
    holds: everything under ``root`` except ``.git``, caches and the
    manifest's ``exclude`` globs. Symlinks are refused. ``strict`` excludes
    nothing (checking an extracted snapshot: any extra file fails it)."""
    root = Path(root)
    patterns = () if strict else (
        _ALWAYS_EXCLUDED + tuple(str(p) for p in (exclude or [])))
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        rel_dir = "" if rel_dir == "." else rel_dir
        keep = []
        for d in sorted(dirnames):
            rel = f"{rel_dir}/{d}" if rel_dir else d
            full = Path(dirpath) / d
            if full.is_symlink():
                raise TreeError(f"symlinks are not allowed in a tool: {rel}")
            if not _excluded(rel, patterns):
                keep.append(d)
        dirnames[:] = keep
        for f in filenames:
            rel = f"{rel_dir}/{f}" if rel_dir else f
            full = Path(dirpath) / f
            if full.is_symlink():
                raise TreeError(f"symlinks are not allowed in a tool: {rel}")
            if not _excluded(rel, patterns):
                out.append(rel)
    out.sort()
    if len(out) > MAX_FILES:
        raise TreeError(f"a tool holds at most {MAX_FILES} files ({len(out)})")
    return out


def tree_digest_of(files: dict[str, bytes]) -> str:
    """The tree digest of ``{relative path: content}``."""
    h = hashlib.sha256()
    for rel in sorted(files):
        h.update(rel.encode("utf-8") + b"\0"
                 + hashlib.sha256(files[rel]).hexdigest().encode("ascii") + b"\n")
    return h.hexdigest()


def snapshot(root: Path) -> tuple[dict[str, Any], str, bytes]:
    """``(manifest file contents, tree digest, archive bytes)`` for the
    codebase at ``root``."""
    root = Path(root).resolve()
    meta = read_manifest(root)
    files = {rel: (root / rel).read_bytes()
             for rel in list_files(root, meta.get("exclude"))}
    entry = str(meta["entrypoint"]).replace("\\", "/")
    if entry not in files:
        raise TreeError(f"entrypoint '{entry}' is not in the snapshot")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in sorted(files):
            info = zipfile.ZipInfo(rel, date_time=_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, files[rel])
    data = buf.getvalue()
    if len(data) > MAX_ARCHIVE_BYTES:
        raise TreeError(f"snapshot is {len(data)} bytes; the cap is "
                        f"{MAX_ARCHIVE_BYTES}")
    return meta, tree_digest_of(files), data


def unpack(archive: bytes) -> dict[str, bytes]:
    """``{relative path: content}`` of an archive, refusing anything that
    could land outside its directory."""
    if len(archive) > MAX_ARCHIVE_BYTES:
        raise TreeError("archive too large")
    out: dict[str, bytes] = {}
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        infos = zf.infolist()
        if len(infos) > MAX_FILES:
            raise TreeError("archive holds too many files")
        total = 0
        for info in infos:
            if info.is_dir():
                continue
            p = PurePosixPath(info.filename)
            if p.is_absolute() or ".." in p.parts or "\\" in info.filename \
                    or ":" in info.filename:
                raise TreeError(f"unsafe path in archive: {info.filename}")
            total += info.file_size
            if total > 8 * MAX_ARCHIVE_BYTES:
                raise TreeError("archive expands too far")
            out[p.as_posix()] = zf.read(info)
    return out


def materialize(archive: bytes, tree_digest: str, cache_root: Path) -> Path:
    """The verified, extracted snapshot under ``cache_root/<tree digest>``.
    Re-verified on every call; re-extracted when anything changed."""
    target = Path(cache_root) / tree_digest
    if target.is_dir() and verify_dir(target, tree_digest):
        return target
    files = unpack(archive)
    if tree_digest_of(files) != tree_digest:
        raise TreeError("archive does not match its tree digest")
    tmp = target.with_name(target.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    for rel, content in files.items():
        dest = tmp / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
    shutil.rmtree(target, ignore_errors=True)
    os.replace(tmp, target)
    return target


def verify_dir(path: Path, tree_digest: str) -> bool:
    """Whether the files under ``path``, every one of them, hash to the
    digest."""
    try:
        files = {rel: (Path(path) / rel).read_bytes()
                 for rel in list_files(Path(path), strict=True)}
    except (OSError, TreeError):
        return False
    return tree_digest_of(files) == tree_digest
