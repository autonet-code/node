"""Base class for host environment probes.

Ported near-verbatim from the deprecated native daemon
(``daemon/integrations/host_probe.py``). Two additions for the autonet
secrets surface:

  * ``ProbeResult.env_keys`` — credential-shaped environment variable NAMES a
    probe discovered (never values, never value fragments). Serialized only
    when non-empty.
  * ``ProbeResult.importable`` — env var names this probe's result exposes that
    the owner could import into the vault. Populated post-scan by the runner.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

# On Windows, prevent console windows from flashing for each subprocess.
_SUBPROCESS_FLAGS: dict[str, Any] = {}
if sys.platform == "win32":
    _SUBPROCESS_FLAGS["creationflags"] = (
        subprocess.CREATE_NO_WINDOW
    )

log = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    """What a single probe discovered about one tool/service."""

    id: str
    name: str
    available: bool = False
    authenticated: bool = False
    version: str | None = None
    account: str | None = None       # username, email, project-id, etc.
    provider: str = ""
    capabilities: list[str] = field(default_factory=list)
    auth_method: str = ""            # "cli", "env_var", "config_file", "none"
    detail: str = ""                 # human-readable extra info
    error: str | None = None
    # Credential-shaped env var NAMES this probe found. NAMES ONLY — never a
    # value or any fragment of one. Serialized only when non-empty.
    env_keys: list[str] = field(default_factory=list)
    # Env var names the owner could import into the vault from this result.
    # Populated after the scan by HostDiscovery (see discovery.py).
    importable: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "available": self.available,
            "authenticated": self.authenticated,
            "provider": self.provider,
            "capabilities": self.capabilities,
            "auth_method": self.auth_method,
            "category": "host",
        }
        if self.version:
            d["version"] = self.version
        if self.account:
            d["account"] = self.account
        if self.detail:
            d["detail"] = self.detail
        if self.error:
            d["error"] = self.error
        if self.env_keys:
            d["env_keys"] = list(self.env_keys)
        if self.importable:
            d["importable"] = list(self.importable)
        return d


class HostProbe(ABC):
    """Detects whether a tool or service is available on the host."""

    id: str = ""
    name: str = ""
    provider: str = ""
    description: str = ""
    tags: list[str] = []
    install_hint: str = ""   # e.g. "winget install GitHub.cli"
    auth_hint: str = ""      # e.g. "Run: gh auth login"
    url: str = ""            # e.g. "https://cli.github.com"

    @abstractmethod
    async def probe(self) -> ProbeResult:
        """Run detection and return a result."""
        ...

    # -- helpers for subclasses --------------------------------------------

    @staticmethod
    def has_command(cmd: str) -> bool:
        return shutil.which(cmd) is not None

    @staticmethod
    async def run_cmd(
        *args: str, timeout: float = 10.0
    ) -> tuple[int, str, str]:
        """Run a command, return (returncode, stdout, stderr)."""
        try:
            # On Windows, .CMD/.BAT scripts (e.g. firebase, netlify, npm)
            # can't be launched by create_subprocess_exec directly.
            # Resolve the full path so Windows knows how to execute them.
            resolved = shutil.which(args[0])
            if resolved:
                args = (resolved, *args[1:])
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **_SUBPROCESS_FLAGS,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            return (
                proc.returncode or 0,
                stdout.decode(errors="replace").strip(),
                stderr.decode(errors="replace").strip(),
            )
        except asyncio.TimeoutError:
            return (-1, "", "timeout")
        except FileNotFoundError:
            return (-1, "", "not found")
        except Exception as exc:
            return (-1, "", str(exc))
