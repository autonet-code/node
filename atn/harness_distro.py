"""Reference harness distro bootstrap (docs/tool_substrate.md —
"Resident tools, loadouts, distros").

At runtime init the daemon registers its own reference harness as the
FIRST distro manifest: one PINNED module manifest per resident tool
bundle (``atn_<bundle>``), then a distro manifest (``atn_reference_harness``)
whose ``dependencies`` are those module digests. The distro digest is
stamped onto the ToolStore as the ``active_loadout`` so cognitive
attestations carry which harness produced them (adoption accrues to the
distro, then fans over its module deps — damp-then-split).

All failures are guarded: a bad bootstrap logs and returns None; it must
NEVER break daemon boot. On a standalone atn install (no ``nodes.*``
substrate package) registration cannot content-address, so we return
None quietly — the daemon runs, just without a distro identity.

Design notes:
  - Module manifests describe CAPABILITY, not a callable signature, so
    ``input_schema = {"type": "object"}``. Their ``code`` blob is the
    concatenated source text of the bundle's executors (what the module
    actually is), hash-locking the module's identity to its behavior.
  - The distro manifest's ``code`` is CONFIG, not executable: the daemon
    identity line plus the sorted module digest list. Its deps carry the
    composition DAG the mint fan-out walks.
  - Idempotent by construction: every blob is content-addressed, and
    ``ToolStore.register`` keys ``_records`` by digest, so a second
    bootstrap over identical source overwrites same-digest records
    (no duplicates) and yields the same distro digest.
"""

from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .runtime import Runtime

log = logging.getLogger(__name__)

# Author recorded for the bootstrap manifests. The reference harness is
# platform-authored on the OWNER's own daemon; "user" is the owner author
# id (OWNER_AUTHOR), matching how owner-registered tools are attributed.
_BOOTSTRAP_AUTHOR = "user"

DISTRO_NAME = "atn_reference_harness"


def _bundle_code(bundle: str, tool_names: list[str], executors: dict) -> str:
    """Concatenated source text for a bundle's executor functions.

    Sorted by tool name for determinism. ``inspect.getsource`` per
    executor; a name with no importable source (or missing from the
    executor map — e.g. shell-tool names, handled by the caller) falls
    back to a stub line so the module still hash-locks to a stable,
    content-addressed identity.
    """
    parts: list[str] = [f"# atn module manifest: {bundle}\n"]
    for name in sorted(tool_names):
        fn = executors.get(name)
        src = ""
        if fn is not None:
            try:
                src = inspect.getsource(fn)
            except (OSError, TypeError):
                src = ""
        if not src:
            src = f"# {name}: source unavailable (stub)\n"
        parts.append(src)
    return "\n".join(parts)


def _shell_code() -> str:
    """Source of atn/shell_tools.py — the shell bundle's tools live there,
    not in ``_EXECUTORS`` (docs comment in ``_TOOL_CATEGORIES``). Hash the
    module file so the ``atn_shell`` module still has a real code blob."""
    from . import shell_tools

    try:
        return inspect.getsource(shell_tools)
    except (OSError, TypeError):
        return "# atn_shell: source unavailable (stub)\n"


def _shell_tool_names() -> list[str]:
    from . import shell_tools

    names = [str(t.get("name") or "") for t in getattr(shell_tools, "SHELL_TOOLS", [])]
    return sorted(n for n in names if n)


def bootstrap_reference_distro(runtime: "Runtime") -> str | None:
    """Register the reference harness as the first distro manifest and
    stamp ``runtime.tool_store.active_loadout`` with its digest.

    Idempotent (content-addressed). Returns the distro digest, or None on
    any failure / when the substrate package is unavailable. NEVER raises.
    """
    try:
        store = getattr(runtime, "tool_store", None)
        if store is None:
            return None

        # nodes.* gate: register() needs the blob store to content-address.
        # Probe it early so a standalone install returns None quietly rather
        # than logging a warning per module.
        try:
            store._blob_store()
        except Exception:
            log.debug("harness distro: substrate package unavailable; skipping")
            return None

        from .orchestrator.tools import _EXECUTORS, _TOOL_CATEGORIES

        module_digests: list[str] = []

        for bundle in sorted(_TOOL_CATEGORIES.keys()):
            tool_names = sorted(_TOOL_CATEGORIES[bundle])
            caps: dict | None = None
            schema: dict = {"type": "object"}

            if bundle == "shell":
                # shell tools live in shell_tools.py, not _EXECUTORS.
                names = _shell_tool_names()
                code = _shell_code()
                desc_names = names or ["bash", "read_file", "write_file",
                                       "list_directory", "search_files"]
                # The shell module is the one bundle whose code blob is a
                # RUNNABLE program (shell_tools.py grew a __main__ that speaks
                # the sealed tool protocol), so its manifest can describe
                # itself honestly: the envelope it accepts, the tool names it
                # provides, and the host access it genuinely needs. Every
                # other bundle stays an identity-only manifest — its blob is
                # concatenated source, not a program, and claiming otherwise
                # would be a lie in the one place the network reads.
                caps = {
                    "provides": sorted(desc_names),
                    "net": True, "fs": True, "spawn": True,
                }
                schema = {
                    "type": "object",
                    "properties": {
                        "tool": {"type": "string", "enum": sorted(desc_names)},
                        "args": {"type": "object"},
                    },
                    "required": ["tool"],
                }
            elif not tool_names:
                # Other empty placeholder bundles carry no capability — skip.
                continue
            else:
                code = _bundle_code(bundle, tool_names, _EXECUTORS)
                desc_names = tool_names

            description = (
                f"atn resident module {bundle}: " + ", ".join(desc_names)
            )
            try:
                res = store.register(
                    name=f"atn_{bundle}",
                    description=description,
                    input_schema=schema,
                    author=_BOOTSTRAP_AUTHOR,
                    code=code,
                    publish=False,
                    capabilities=caps,
                )
            except Exception as exc:
                log.warning("harness distro: module %r failed: %s", bundle, exc)
                continue
            module_digests.append(res["digest"])
            # Bootstrap hygiene: dev code drift mints a new module digest
            # (content-addressed, no version chain), and pre-idempotency
            # boots left one copy per start. Superseded local records are
            # noise — drop them, keeping only the digest just registered.
            try:
                store.prune_superseded_local(
                    name=f"atn_{bundle}",
                    author=_BOOTSTRAP_AUTHOR,
                    keep_digest=res["digest"],
                )
            except Exception as exc:  # noqa: BLE001 — hygiene, never fatal
                log.debug("harness distro: prune failed for %r: %s",
                          bundle, exc)

        if not module_digests:
            log.warning("harness distro: no modules registered; skipping distro")
            return None

        module_digests = sorted(set(module_digests))

        # Distro blob is CONFIG, not executable: daemon identity + the
        # sorted module digest list. Version string is best-effort.
        try:
            import atn

            version = getattr(atn, "__version__", "dev")
        except Exception:
            version = "dev"
        distro_code = "\n".join(
            [f"atn reference harness\n{version}", *module_digests]
        )

        try:
            distro = store.register(
                name=DISTRO_NAME,
                description=(
                    "atn reference harness distro: the daemon's default "
                    "resident loadout (" + str(len(module_digests)) + " modules)"
                ),
                input_schema={"type": "object"},
                author=_BOOTSTRAP_AUTHOR,
                code=distro_code,
                dependencies=module_digests,
                publish=False,
            )
        except Exception as exc:
            log.warning("harness distro: distro manifest failed: %s", exc)
            return None

        distro_digest = distro["digest"]
        try:
            store.prune_superseded_local(
                name=DISTRO_NAME,
                author=_BOOTSTRAP_AUTHOR,
                keep_digest=distro_digest,
            )
        except Exception as exc:  # noqa: BLE001 — hygiene, never fatal
            log.debug("harness distro: distro prune failed: %s", exc)
        store.active_loadout = distro_digest
        log.info("harness distro bootstrapped: %s (%d modules)",
                 distro_digest[:16], len(module_digests))
        return distro_digest
    except Exception as exc:  # absolute backstop — never break boot
        log.warning("harness distro bootstrap failed: %s", exc)
        return None
