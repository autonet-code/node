"""The robot side of a daemon (docs/robots.md).

When ``robot.enabled``, the daemon:

- exports the required safety packs from its tool store into the robot
  bundle the filter node reads (``export_packs``);
- writes a heartbeat into the bundle every ``heartbeat_interval_s``, carrying
  whether the sponsor is reachable. A dead daemon stops writing, so the filter
  sees a stale heartbeat and stops the robot; an unreachable sponsor stops it
  too, since no judgment is available;
- answers worklog challenges (``attest``).

Sponsor reachability uses the sponsored provider's own discovery: the named
sponsor (``autonet.sponsor_address``) must be advertising on the p2p network.
``ATN_ROBOT_SPONSOR_PROBE=assume`` replaces the probe for local tests that run
without a network.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any

from . import packs as P
from . import worklog as W

log = logging.getLogger(__name__)


class RobotService:
    def __init__(self, runtime: Any, config: Any) -> None:
        self._runtime = runtime
        self._config = config
        self.bundle_dir = Path(config.robot.bundle_dir or
                               Path(config.data_dir) / "robot" / "bundle")
        self.bag_dir = Path(config.robot.bag_dir or
                            Path(config.data_dir) / "robot" / "bags")
        self.chain_dir = Path(config.data_dir) / "robot" / "chain"
        self._worklog: W.Worklog | None = None
        self.sponsor_ok = False
        self.exported: list[str] = []
        self.export_errors: list[str] = []
        self._task: asyncio.Task | None = None

    # -- packs --------------------------------------------------------------

    def export_packs(self) -> dict[str, Any]:
        """Resolve ``robot.required_packs`` in the tool store and write the
        bundle. On any error nothing is exported and the old bundle stays."""
        store = self._runtime.tool_store
        blobs = store._blob_store()
        pairs, errors = [], []
        for ref in self._config.robot.required_packs:
            rec = store.resolve(ref)
            if rec is None:
                errors.append(f"{ref}: not in the tool store")
                continue
            if not rec.enabled:
                errors.append(f"{ref}: disabled")
                continue
            errs = P.pack_errors(rec.manifest)
            if errs:
                errors.append(f"{ref}: {'; '.join(errs)}")
                continue
            code = blobs.get_bytes_local(rec.manifest["code_digest"])
            if code is None:
                errors.append(f"{ref}: code blob missing")
                continue
            pairs.append((rec.manifest, code.decode("utf-8")))
        if not pairs and not errors:
            errors.append("robot.required_packs is empty")
        self.export_errors = errors
        if errors:
            log.warning("robot packs not exported: %s", errors)
            return {"ok": False, "errors": errors}
        self.exported = P.write_bundle(self.bundle_dir, pairs)
        log.info("robot packs exported: %s", [d[:12] for d in self.exported])
        return {"ok": True, "packs": self.exported}

    # -- heartbeat ----------------------------------------------------------

    async def _probe_sponsor(self) -> bool:
        if os.environ.get("ATN_ROBOT_SPONSOR_PROBE", "").strip() == "assume":
            return True
        sponsor = (self._config.autonet.sponsor_address or "").strip()
        p2p = getattr(self._runtime.providers, "_p2p_host", None)
        if not sponsor or p2p is None:
            return False
        from ..providers.rpb import RPBNetworkProvider
        found = await RPBNetworkProvider(p2p_host=p2p,
                                         sponsor_address=sponsor).discover_providers()
        return bool(found)

    async def _loop(self) -> None:
        interval = max(0.2, float(self._config.robot.heartbeat_interval_s))
        while True:
            try:
                self.sponsor_ok = await self._probe_sponsor()
            except Exception:  # noqa: BLE001
                log.debug("sponsor probe failed", exc_info=True)
                self.sponsor_ok = False
            try:
                P.write_heartbeat(self.bundle_dir, sponsor_ok=self.sponsor_ok)
            except OSError:
                log.warning("robot heartbeat write failed", exc_info=True)
            await asyncio.sleep(interval)

    def start(self) -> None:
        self.export_packs()
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    # -- attestation --------------------------------------------------------

    def _robot_key(self) -> tuple[str, str]:
        agent_id = self._config.robot.agent_id
        defn = self._runtime.get_agent(agent_id) if agent_id else None
        if defn is None:
            raise ValueError("robot.agent_id does not name an agent on this daemon")
        key = self._runtime.registry.get_agent_key(agent_id)
        address = str(getattr(getattr(defn, "identity", None), "address", "") or "")
        if not key or not address:
            raise ValueError(f"agent {agent_id!r} has no signing key")
        return address, key

    @property
    def worklog(self) -> W.Worklog:
        """The robot's signed worklog chain, opened on first use (it needs
        the robot agent's key)."""
        if self._worklog is None:
            address, key = self._robot_key()
            self._worklog = W.Worklog(self.bag_dir, self.chain_dir, address, key)
        return self._worklog

    def write_challenge(self, challenge: str) -> None:
        P.write_challenge(self.bundle_dir, challenge)

    def attest(self, challenge: str) -> dict[str, Any]:
        """An owner's own poll: record the challenge and commit a link that
        carries it. Returns the signed link."""
        if not challenge or len(challenge) > 256:
            raise ValueError("challenge must be 1 to 256 characters")
        self.write_challenge(challenge)
        return self.worklog.commit(packs=self.exported, challenge=challenge, force=True)

    def status(self) -> dict[str, Any]:
        hb = P.read_heartbeat(self.bundle_dir)
        return {"bundle_dir": str(self.bundle_dir), "bag_dir": str(self.bag_dir),
                "agent_id": self._config.robot.agent_id,
                "required": list(self._config.robot.required_packs),
                "exported": self.exported, "export_errors": self.export_errors,
                "sponsor_ok": self.sponsor_ok,
                "heartbeat_age_s": (time.time() - float(hb["ts"])) if hb.get("ts") else None}
