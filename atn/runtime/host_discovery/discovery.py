"""Host discovery runner — probes the local environment for available tools.

Ported from the deprecated native daemon (``daemon/integrations/host_discovery.py``)
with three autonet-specific changes:

  * the cache file lives at ``<data_dir>/security/host_scan.json`` (the
    ``security`` dir already exists under ``~/.atn``);
  * a concurrency cap (``asyncio.Semaphore``) so a scan of ~40 subprocess-forking
    probes can't fork-bomb the host;
  * a post-scan ``importable`` pass computing, per result, the env var names the
    owner could import into the vault (all found names for the env-credential
    probe; the specific env var a keyed probe checks, when present).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from .probe import HostProbe, ProbeResult
from .probes import ALL_PROBES

log = logging.getLogger(__name__)

# Max probes running concurrently. Each probe may fork several short-lived
# subprocesses (version/auth checks); an unbounded gather over ~40 probes could
# spawn a fork-bomb-shaped burst. 8 keeps the scan quick without starving the
# host or the websocket keepalive loop.
_MAX_CONCURRENCY = 8

# Per-probe env vars an owner could import into the vault. Keyed on probe id;
# the first name present in os.environ (if any) is offered as importable. The
# env-credential probe is handled separately (it carries its own env_keys).
_PROBE_IMPORT_ENV: dict[str, tuple[str, ...]] = {
    "host_anthropic": ("ANTHROPIC_API_KEY",),
    "host_openai": ("OPENAI_API_KEY",),
    "host_github": ("GITHUB_TOKEN", "GH_TOKEN"),
    "host_huggingface": ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"),
    "host_etherscan": ("ETHERSCAN_API_KEY",),
    "host_stripe": ("STRIPE_SECRET_KEY", "STRIPE_API_KEY"),
    "host_slack": ("SLACK_BOT_TOKEN", "SLACK_TOKEN"),
    "host_discord": ("DISCORD_BOT_TOKEN", "DISCORD_TOKEN"),
    "host_notion": ("NOTION_API_KEY", "NOTION_TOKEN"),
    "host_figma": ("FIGMA_TOKEN", "FIGMA_ACCESS_TOKEN"),
    "host_sendgrid": ("SENDGRID_API_KEY",),
    "host_twilio": ("TWILIO_AUTH_TOKEN",),
    "host_coingecko": ("COINGECKO_API_KEY",),
    "host_cloudflare": ("CLOUDFLARE_API_TOKEN",),
}


def validate_secret_name(name: str) -> str | None:
    """Validate a vault service/secret NAME (identical rules to the WS
    ``secrets_put`` handler). Returns an error reason string, or ``None`` when
    the name is acceptable.

    The name doubles as an allowance-spec token and a vault key, so commas,
    whitespace, empty strings, and the reserved keywords ``none``/``all`` are
    rejected. Dotted names are reserved for daemon-internal credentials
    (CredentialStore ``app.*``, ``agent-key.*``) and are also rejected —
    belt-and-braces, since env var names never contain dots.
    """
    if not name:
        return "empty name"
    if "," in name or any(c.isspace() for c in name):
        return "commas/whitespace not allowed"
    if "." in name:
        return "dotted names are reserved for daemon-internal credentials"
    if name.lower() in ("none", "all"):
        return "reserved allowance-spec keyword"
    return None


class HostDiscovery:
    """Runs all host probes and caches the results.

    Results are persisted to ``<data_dir>/security/host_scan.json`` so that a
    daemon restart doesn't require an immediate rescan.
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        self._probes: list[HostProbe] = [cls() for cls in ALL_PROBES]
        self._results: list[ProbeResult] = []
        self._last_scan: float = 0
        self._scanning = False
        self._cache_path: Path | None = (
            Path(data_dir) / "security" / "host_scan.json" if data_dir else None
        )
        self._load_cache()

    # -- importable post-processing --------------------------------------

    @staticmethod
    def _compute_importable(result: ProbeResult) -> list[str]:
        """Env var names this result exposes that the owner could import into
        the vault. NAMES ONLY."""
        if result.env_keys:
            # The env-credential probe already carries the exact names.
            return list(result.env_keys)
        candidates = _PROBE_IMPORT_ENV.get(result.id)
        if not candidates:
            return []
        return [n for n in candidates if os.environ.get(n)]

    # -- cache persistence -----------------------------------------------

    def _load_cache(self) -> None:
        """Load cached scan results from disk if available."""
        if not self._cache_path or not self._cache_path.exists():
            return
        try:
            data = json.loads(self._cache_path.read_text(encoding="utf-8"))
            self._last_scan = data.get("last_scan", 0)
            self._results = [
                ProbeResult(
                    id=item["id"],
                    name=item["name"],
                    available=item.get("available", False),
                    authenticated=item.get("authenticated", False),
                    version=item.get("version"),
                    account=item.get("account"),
                    provider=item.get("provider", ""),
                    capabilities=item.get("capabilities", []),
                    auth_method=item.get("auth_method", ""),
                    detail=item.get("detail", ""),
                    error=item.get("error"),
                    env_keys=item.get("env_keys", []),
                    importable=item.get("importable", []),
                )
                for item in data.get("items", [])
            ]
            log.info(
                "loaded cached host scan  %d results  age=%.0fs",
                len(self._results),
                time.time() - self._last_scan,
            )
        except Exception:
            log.debug("failed to load host scan cache", exc_info=True)

    def _save_cache(self) -> None:
        """Persist current scan results to disk."""
        if not self._cache_path:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "last_scan": self._last_scan,
                "items": [r.to_dict() for r in self._results],
            }
            self._cache_path.write_text(
                json.dumps(data, indent=2), encoding="utf-8",
            )
        except Exception:
            log.debug("failed to save host scan cache", exc_info=True)

    def registered_probes(self) -> list[dict[str, Any]]:
        """Return the list of all known probe IDs, names, providers, descriptions, and tags."""
        return [
            {
                "id": p.id,
                "name": p.name,
                "provider": p.provider,
                "description": p.description,
                "tags": p.tags,
            }
            for p in self._probes
        ]

    async def scan(
        self, probe_ids: list[str] | None = None,
    ) -> list[ProbeResult]:
        """Run probes concurrently (capped) and cache results.

        If *probe_ids* is given, only run the probes whose id is in the list.
        """
        if self._scanning:
            while self._scanning:
                await asyncio.sleep(0.1)
            return self._results

        probes = self._probes
        if probe_ids is not None:
            allowed = set(probe_ids)
            probes = [p for p in self._probes if p.id in allowed]

        self._scanning = True
        t0 = time.monotonic()
        sem = asyncio.Semaphore(_MAX_CONCURRENCY)

        async def _run(p: HostProbe) -> ProbeResult:
            async with sem:
                return await p.probe()

        try:
            tasks = [_run(p) for p in probes]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            self._results = []
            for probe, result in zip(probes, results):
                if isinstance(result, Exception):
                    log.warning("probe failed  id=%s  error=%s", probe.id, result)
                    self._results.append(ProbeResult(
                        id=probe.id,
                        name=probe.name,
                        provider=probe.provider,
                        error=str(result),
                    ))
                else:
                    result.importable = self._compute_importable(result)
                    self._results.append(result)

            self._last_scan = time.time()
            self._save_cache()
            elapsed = time.monotonic() - t0
            available = [r for r in self._results if r.available]
            authed = [r for r in self._results if r.authenticated]
            log.info(
                "host scan complete  %.1fs  %d/%d available  %d authenticated",
                elapsed, len(available), len(self._results), len(authed),
            )
            return self._results
        finally:
            self._scanning = False

    @property
    def results(self) -> list[ProbeResult]:
        return self._results

    @property
    def last_scan(self) -> float:
        return self._last_scan

    @property
    def scanning(self) -> bool:
        return self._scanning

    @property
    def has_scanned(self) -> bool:
        return self._last_scan > 0

    def to_dict(self, available_only: bool = False) -> dict[str, Any]:
        items = self._results
        if available_only:
            items = [r for r in items if r.available]
        return {
            "scanned": self.has_scanned,
            "scanning": self._scanning,
            "items": [r.to_dict() for r in items],
        }

    def to_list(self, available_only: bool = False) -> list[dict[str, Any]]:
        items = self._results
        if available_only:
            items = [r for r in items if r.available]
        return [r.to_dict() for r in items]
