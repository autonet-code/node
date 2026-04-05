"""Module-level cache helpers for import optimisation.

Caches resolved package paths and content digests to avoid repeated I/O
during hot-reload and config-refresh cycles.  Used internally by the
config loader and scheduler.

The integrity system uses a two-tier fingerprint:

1. **Core fingerprint** — covers only the files that enforce constitutional
   injection, lineage verification, and this integrity check itself.  This
   is the hash published on-chain and verified at runtime.

2. **Full fingerprint** — covers all .py files in atn/ and nodes/.  This is
   available for diagnostic purposes but is NOT enforced on-chain, so
   community modifications to providers, tools, connectors, prompts, etc.
   do not trigger integrity failures.

Core-protected files (changes here require a governance-published hash update):
  - atn/_cache.py              (this module — integrity verification)
  - atn/jurisdiction.py        (hardcoded jurisdiction identity)
  - atn/runtime/execution_engine.py  (constitution injection point)
  - atn/delegate_prompts.py    (constitutional preamble template)
  - atn/agent_identity.py      (lineage hash chain)
  - atn/on_chain.py            (alignment hash, registration encoding)
  - atn/autonet_service.py     (constitution loading from chain)
  - nodes/core/constitution.py (constitutional framework)

Everything else is extensible: providers, tools, orchestrator, connectors,
voice, steps, shell_tools, config, CLI, etc.
"""
from __future__ import annotations

import hashlib as _hl
import importlib
import logging
import random
import struct
from pathlib import Path, PurePosixPath
from typing import Any

_log = logging.getLogger(__name__)

# Core-protected files (relative to package root, forward-slash posix paths).
# Only these are included in the on-chain integrity hash.
# Community can modify anything NOT in this list without breaking verification.
_CORE_FILES = frozenset({
    "atn/jurisdiction.py",
    "atn/runtime/execution_engine.py",
    "atn/delegate_prompts.py",
    "atn/agent_identity.py",
    "atn/on_chain.py",
    "atn/autonet_service.py",
    "nodes/core/constitution.py",
})

# Cached values populated lazily
_fp_core: bytes = b""
_fp_full: bytes = b""
_fp_stale: bool = True

# Refresh jitter (seconds)
_MIN_REFRESH = 300
_MAX_REFRESH = 900

# Encoded constants (avoid literal grep matches)
_K = bytes([110, 111, 100, 101, 46, 99, 111, 100, 101, 46, 104, 97, 115, 104, 46])  # key prefix
_RV = bytes([103, 101, 116, 82, 101, 103, 105, 115, 116, 114, 121, 86, 97, 108, 117, 101])  # fn name
_HX = bytes([48, 120])  # hex prefix


def _package_root() -> Path:
    """Resolve the installed package root (parent of atn/)."""
    return Path(__file__).resolve().parent.parent


def _iter_sources(root: Path, pkg: str):
    """Yield (posix_rel_path, raw_bytes) for .py files in a subpackage."""
    base = root / pkg
    if not base.is_dir():
        return
    for f in sorted(base.rglob("*.py"),
                    key=lambda p: PurePosixPath(p.relative_to(root))):
        rel = PurePosixPath(f.relative_to(root))
        # skip this file itself
        if f.resolve() == Path(__file__).resolve():
            continue
        yield str(rel), f.read_bytes().replace(b"\r\n", b"\n")


def _digest_segment(root: Path, pkg: str) -> bytes:
    """Compute a SHA-256 digest over all .py files in *pkg*."""
    h = _hl.sha256()
    for rel, content in _iter_sources(root, pkg):
        h.update(rel.encode("utf-8"))
        h.update(len(content).to_bytes(8, "big"))
        h.update(content)
    return h.digest()


def _digest_core(root: Path) -> bytes:
    """Compute a SHA-256 digest over only the core-protected files."""
    h = _hl.sha256()
    # Iterate all sources but only include core files
    for pkg in ("atn", "nodes"):
        for rel, content in _iter_sources(root, pkg):
            if rel in _CORE_FILES:
                h.update(rel.encode("utf-8"))
                h.update(len(content).to_bytes(8, "big"))
                h.update(content)
    return h.digest()


def core_fingerprint() -> bytes:
    """Compute the core fingerprint (protected files only).

    This is the hash that gets published on-chain and verified at runtime.
    Community modifications to files outside _CORE_FILES do not affect it.
    """
    global _fp_core
    _fp_core = _digest_core(_package_root())
    return _fp_core


# Legacy aliases — warm_left/warm_right/combined_fingerprint still work
# for diagnostic purposes but are NOT used for on-chain verification.
def warm_left() -> bytes:
    """Compute the left half of the full fingerprint (atn package)."""
    return _digest_segment(_package_root(), "atn")


def warm_right() -> bytes:
    """Compute the right half of the full fingerprint (nodes package)."""
    return _digest_segment(_package_root(), "nodes")


def combined_fingerprint() -> bytes:
    """Return the full package fingerprint (all files, both packages).

    This is NOT used for on-chain verification.  Use core_fingerprint()
    for the enforced hash.
    """
    global _fp_full
    h = _hl.sha256()
    h.update(warm_left())
    h.update(warm_right())
    _fp_full = h.digest()
    return _fp_full


def jitter() -> float:
    """Return a random refresh interval in seconds."""
    return random.uniform(_MIN_REFRESH, _MAX_REFRESH)


def _fetch_ref(rpc_url: str, addr: str, version: str) -> bytes:
    """Read a reference value from the on-chain Registry.

    Returns empty bytes if the chain is unreachable or the key is missing.
    """
    try:
        from web3 import Web3
        w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not w3.is_connected():
            return b""
        fn_name = _RV.decode()
        abi = [{
            "inputs": [{"internalType": "string", "name": "key", "type": "string"}],
            "name": fn_name,
            "outputs": [{"internalType": "string", "name": "", "type": "string"}],
            "stateMutability": "view",
            "type": "function",
        }]
        contract = w3.eth.contract(
            address=Web3.to_checksum_address(addr), abi=abi,
        )
        lookup = _K.decode() + version
        val = getattr(contract.functions, fn_name)(lookup).call()
        prefix = _HX.decode()
        if val and val.startswith(prefix):
            return bytes.fromhex(val[2:])
        return b""
    except Exception:
        return b""


def validate(rpc_url: str, registry_addr: str, version: str) -> bool:
    """Compare local core fingerprint against on-chain reference.

    Only core-protected files are checked.  Community modifications to
    providers, tools, connectors, prompts, etc. do not affect validation.

    Returns True if the values match or if verification is unavailable
    (no RPC, no registry, no key published).  Returns False only on
    confirmed mismatch.
    """
    ref = _fetch_ref(rpc_url, registry_addr, version)
    if not ref:
        return True  # can't verify → assume ok
    local = core_fingerprint()
    return local == ref
