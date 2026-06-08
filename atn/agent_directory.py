"""Agent directory — publish this daemon's browser-reachable WS endpoint.

The chain (via the indexer) owns the permanent half of an agent's record:
0x address, lineage, registration. But a daemon's reachable wss:// URL is
*presence* data — it changes on restart / IP change / proxy move, and nobody
verifies it on-chain. So the daemon writes it directly to the Firestore agent
directory (collection `agents`, doc id = checksum address), the same collection
the frontend already reads for the network roster.

This is what lets a remote frontend connect by an agent's 0x address alone:
resolve address -> wsEndpoint in Firestore -> dial that WS -> sign the auth
challenge -> get scoped to that agent. The browser can reach Firestore (HTTPS)
but not the libp2p mesh, which is why peerId in the directory isn't enough.

Two disjoint writers on one doc:
  - indexer: address, lineageHash, peerId, registration metadata (chain-derived)
  - daemon (here): wsEndpoint, lastSeen (mutable presence)
Both use merge writes, so neither clobbers the other's fields.

Best-effort by construction: a missing dependency, missing credentials, or a
Firestore error must NEVER block daemon boot or agent work. Publishing is
skipped entirely unless ``public_ws_endpoint`` is configured.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

log = logging.getLogger(__name__)

_COLLECTION = "agents"


class AgentDirectoryPublisher:
    """Writes {wsEndpoint, lastSeen} into each hosted agent's Firestore doc.

    Constructed by the runtime when ``public_ws_endpoint`` is set. ``start()``
    publishes once — reachability only changes on a daemon restart (new IP /
    proxy move), and a restart re-runs start(), so a single write per boot
    covers every event that changes the endpoint. No heartbeat: a dead daemon
    is discovered when the WS dial fails, not by polling lastSeen. All Firestore
    access is lazy and guarded — if anything is unavailable the publisher logs
    once and becomes a no-op.
    """

    def __init__(self, runtime: Any, *, endpoint: str, project: str = "") -> None:
        self.runtime = runtime
        self.endpoint = endpoint
        self.project = project
        self._client: Any = None
        self._enabled = bool(endpoint)

    # -- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        if not self._enabled:
            return
        if not self._connect():
            return                              # logged; stays a no-op
        await self._publish_once()

    async def stop(self) -> None:
        # Nothing to tear down — the publish is one-shot at start().
        return

    # -- internals ----------------------------------------------------------

    def _connect(self) -> bool:
        """Lazily build the Firestore client. Returns False (and disables the
        publisher) on any failure — never raises into the caller."""
        try:
            from google.cloud import firestore
        except ImportError:
            log.info("Agent directory: google-cloud-firestore not installed; "
                     "reachability will not be published.")
            self._enabled = False
            return False
        try:
            self._client = (firestore.Client(project=self.project)
                            if self.project else firestore.Client())
            return True
        except Exception as exc:
            log.warning("Agent directory: could not init Firestore (%s); "
                        "reachability will not be published.", exc)
            self._enabled = False
            return False

    def _hosted_agents(self) -> list[dict]:
        """Advertisement dicts for agents this daemon hosts that have an
        identity. Carries address + name + agent_type (the indexer can't know
        name/type — they aren't on-chain — so the daemon publishes them so the
        directory can show a real name, not just an address)."""
        try:
            ads = self.runtime.registry.build_agent_advertisements()
        except Exception:
            log.debug("Agent directory: advertisement enumeration failed",
                      exc_info=True)
            return []
        return [a for a in ads if a.get("address")]

    async def _publish_once(self) -> None:
        """Merge reachability + display fields into every hosted agent's doc.
        Runs the blocking Firestore calls off the event loop."""
        if self._client is None:
            return
        agents = self._hosted_agents()
        if not agents:
            return
        now = int(time.time())
        try:
            await asyncio.to_thread(self._write_batch, agents, now)
            log.info("Agent directory: published directory entry for %d agent(s)",
                     len(agents))
        except Exception as exc:
            log.warning("Agent directory: publish failed (%s)", exc)

    def _write_batch(self, agents: list[dict], now: int) -> None:
        """Blocking: one merge-set per agent. Merge so the indexer's
        chain-derived fields (address/peerId/registration) are untouched; we
        only write the daemon-owned fields: reachability + display name/type."""
        for ad in agents:
            payload = {
                "wsEndpoint": self.endpoint,
                "lastSeen": now,
                "displayName": ad.get("name", ""),
                "agentType": ad.get("agent_type", ""),
            }
            self._client.collection(_COLLECTION).document(ad["address"]).set(
                payload, merge=True)
