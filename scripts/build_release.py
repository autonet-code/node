#!/usr/bin/env python3
"""Build a release: obfuscate integrity modules, build wheel, compute package hash.

Usage:
    python scripts/build_release.py              # build only
    python scripts/build_release.py --hash-only  # just print the core package hash

Requires: python-minifier, build
    pip install python-minifier build

The integrity system uses a two-tier fingerprint.  The on-chain hash covers
ONLY the core-protected files (constitution injection, lineage verification,
integrity checking).  Community-modifiable files (providers, tools, connectors,
prompts, etc.) are NOT included in the on-chain hash, so modifications to
those files do not break integrity verification.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent

# Modules to obfuscate (the integrity-check system)
OBFUSCATE_TARGETS = [
    ROOT / "atn" / "_cache.py",
]

# Core-protected files — must match _CORE_FILES in atn/_cache.py.
# Only these are included in the on-chain integrity hash.
_CORE_FILES = frozenset({
    "atn/jurisdiction.py",
    "atn/runtime/execution_engine.py",
    "atn/delegate_prompts.py",
    "atn/agent_identity.py",
    "atn/on_chain.py",
    "atn/autonet_service.py",
    "nodes/core/constitution.py",
})


def _iter_sources(package_root: Path, pkg: str):
    """Yield (posix_rel_path, raw_bytes) for .py files in a subpackage."""
    cache_path = (package_root / "atn" / "_cache.py").resolve()
    base = package_root / pkg
    if not base.is_dir():
        return
    for f in sorted(base.rglob("*.py"),
                    key=lambda p: PurePosixPath(p.relative_to(package_root))):
        if f.resolve() == cache_path:
            continue
        rel = PurePosixPath(f.relative_to(package_root))
        content = f.read_bytes().replace(b"\r\n", b"\n")
        yield str(rel), content


def compute_core_hash(package_root: Path) -> str:
    """Compute the SHA-256 hash of only the core-protected files.

    This matches the runtime core_fingerprint() algorithm in _cache.py.
    Only files in _CORE_FILES are included.
    """
    h = hashlib.sha256()
    for pkg in ("atn", "nodes"):
        for rel, content in _iter_sources(package_root, pkg):
            if rel in _CORE_FILES:
                h.update(rel.encode("utf-8"))
                h.update(len(content).to_bytes(8, "big"))
                h.update(content)
    return "0x" + h.hexdigest()


def compute_full_hash(package_root: Path) -> str:
    """Compute the SHA-256 hash of ALL .py files (diagnostic, not enforced)."""
    left = hashlib.sha256()
    for rel, content in _iter_sources(package_root, "atn"):
        left.update(rel.encode("utf-8"))
        left.update(len(content).to_bytes(8, "big"))
        left.update(content)

    right = hashlib.sha256()
    for rel, content in _iter_sources(package_root, "nodes"):
        right.update(rel.encode("utf-8"))
        right.update(len(content).to_bytes(8, "big"))
        right.update(content)

    h = hashlib.sha256()
    h.update(left.digest())
    h.update(right.digest())
    return "0x" + h.hexdigest()


def obfuscate(targets: list[Path]) -> None:
    """Minify the given Python files in-place using python-minifier."""
    try:
        import python_minifier
    except ImportError:
        print("ERROR: python-minifier not installed. Run: pip install python-minifier",
              file=sys.stderr)
        sys.exit(1)

    for path in targets:
        if not path.exists():
            print(f"WARNING: {path} not found, skipping", file=sys.stderr)
            continue
        source = path.read_text(encoding="utf-8")
        minified = python_minifier.minify(
            source,
            filename=path.name,
            remove_literal_statements=True,  # strip docstrings
            rename_locals=True,
            rename_globals=False,  # keep public API names
            remove_annotations=True,
            hoist_literals=False,
            remove_pass=True,
            remove_object_base=True,
            combine_imports=True,
        )
        path.write_text(minified, encoding="utf-8")
        print(f"  Obfuscated: {path.relative_to(ROOT)}"
              f" ({len(source)} -> {len(minified)} bytes)")


def build_wheel() -> Path:
    """Build a wheel using the 'build' package. Returns the wheel path."""
    dist = ROOT / "dist"
    if dist.exists():
        shutil.rmtree(dist)

    subprocess.check_call(
        [sys.executable, "-m", "build", "--wheel", str(ROOT)],
        cwd=str(ROOT),
    )

    wheels = list(dist.glob("*.whl"))
    if not wheels:
        print("ERROR: No wheel produced", file=sys.stderr)
        sys.exit(1)
    return wheels[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an autonet release")
    parser.add_argument("--hash-only", action="store_true",
                        help="Just compute and print the package hash")
    parser.add_argument("--no-obfuscate", action="store_true",
                        help="Skip obfuscation step")
    args = parser.parse_args()

    if args.hash_only:
        h = compute_core_hash(ROOT)
        print(h)
        return

    print("=== Autonet Release Build ===\n")

    # Step 1: Obfuscate integrity modules
    if not args.no_obfuscate:
        print("1. Obfuscating integrity modules...")
        obfuscate(OBFUSCATE_TARGETS)
    else:
        print("1. Skipping obfuscation (--no-obfuscate)")

    # Step 2: Build wheel
    print("\n2. Building wheel...")
    wheel = build_wheel()
    print(f"   Wheel: {wheel.name}")

    # Step 3: Compute core hash (covers only protected files)
    print("\n3. Computing core integrity hash...")
    core_hash = compute_core_hash(ROOT)
    full_hash = compute_full_hash(ROOT)
    print(f"   Core hash: {core_hash}")
    print(f"   Full hash: {full_hash} (diagnostic only, not enforced)")
    print(f"\n   Publish the CORE hash on-chain:")
    print(f"   Key:   node.code.hash.{_get_version()}")
    print(f"   Value: {core_hash}")
    print(f"\n   Core-protected files ({len(_CORE_FILES)}):")
    for f in sorted(_CORE_FILES):
        print(f"     - {f}")

    # Write hash to a file for CI to pick up
    hash_file = ROOT / "dist" / "package_hash.txt"
    hash_file.write_text(core_hash, encoding="utf-8")
    print(f"\n   Written to: {hash_file}")

    print("\n=== Done ===")
    print(f"   Wheel: dist/{wheel.name}")
    print(f"   Hash:  dist/package_hash.txt")


def _get_version() -> str:
    """Read the package version from atn/__init__.py."""
    init = ROOT / "atn" / "__init__.py"
    for line in init.read_text(encoding="utf-8").splitlines():
        if line.startswith("__version__"):
            return line.split("=")[1].strip().strip('"').strip("'")
    return "0.0.0"


if __name__ == "__main__":
    main()
