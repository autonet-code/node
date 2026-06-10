"""Agent directory — publish this daemon's browser-reachable WS endpoint.

The chain owns identity (address, lineage, peerId — the libp2p mesh id). The
browser-reachable wss:// URL is *presence*: it changes every daemon restart
when fronted by an ephemeral tunnel. We publish it ON-CHAIN via the agent's own
`updateEndpoint` tx (msg.sender == agent, so no one can set another agent's
endpoint — the directory is hijack-proof). An off-chain indexer mirrors the
`EndpointUpdated` event into the agent directory (Firestore `agents`
collection) that the browser reads to resolve an agent's 0x address -> wss.

Why on-chain and not a direct Firestore write: a Firestore security rule can't
verify a 0x signature, so a daemon writing Firestore directly can't be stopped
from overwriting another agent's endpoint. The chain verifies the signer
natively (it IS msg.sender). The daemon therefore never touches Firestore.

Each call costs gas, so we publish only when the endpoint actually CHANGED
(read the current on-chain value first). With an ephemeral tunnel the URL
changes every restart, so expect one tx per restart while remote access is on;
with a static URL it's rare.

Best-effort by construction: a missing dependency, an unconfigured chain, an
unregistered agent, or an RPC error must NEVER block daemon boot or agent work.
Publishing is skipped entirely unless ``public_ws_endpoint`` is configured.
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


class AgentDirectoryPublisher:
    """Publishes this daemon's wss endpoint on-chain for each hosted agent.

    Constructed by the runtime when ``public_ws_endpoint`` is set. ``start()``
    publishes once per boot, per hosted registered agent, only when the
    endpoint differs from what's already on-chain (an ephemeral tunnel makes
    that every restart). The indexer mirrors the resulting EndpointUpdated
    event to the off-chain directory.
    """

    def __init__(self, runtime: Any, *, endpoint: str, project: str = "") -> None:
        self.runtime = runtime
        self.endpoint = endpoint
        # project kept for signature compatibility with the prior Firestore
        # implementation; no longer used (no direct Firestore access).
        self.project = project
        self._enabled = bool(endpoint)

    # -- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        if not self._enabled:
            return
        try:
            await self._publish_once()
        except Exception:
            log.debug("Agent directory: publish failed (non-fatal)", exc_info=True)

    async def stop(self) -> None:
        # Nothing to tear down — the publish is one-shot at start().
        return

    # -- internals ----------------------------------------------------------

    def _svc(self) -> Any:
        """The on-chain service, or None if the chain isn't configured/usable."""
        try:
            from .on_chain import OnChainService
            svc = OnChainService(self.runtime._config.autonet)
            return svc if svc.available else None
        except Exception:
            return None

    async def _publish_once(self) -> None:
        """For each hosted, on-chain-registered agent whose on-chain endpoint
        differs from ours, sign an updateEndpoint tx with that agent's key."""
        svc = self._svc()
        if svc is None:
            log.info("Agent directory: chain not configured; endpoint not published.")
            return
        reg = self.runtime.registry
        published = 0
        for defn, _status in reg.list_agents():
            ident = getattr(defn, "identity", None)
            if not ident or not ident.address:
                continue
            if not getattr(ident, "registered_on_chain", False):
                continue            # only registered agents have an on-chain slot
            key = reg.get_agent_key(defn.id)
            if not key:
                continue
            try:
                current = await svc.get_agent_endpoint(ident.address)
                if current == self.endpoint:
                    continue        # unchanged — no tx, no gas
                result = await svc.update_endpoint(key, self.endpoint)
                if result.get("success"):
                    published += 1
                else:
                    log.warning("Agent directory: updateEndpoint failed for %s: %s",
                                defn.id, result.get("error"))
            except Exception:
                log.debug("Agent directory: publish error for %s", defn.id, exc_info=True)
        if published:
            log.info("Agent directory: published wss endpoint on-chain for %d agent(s)",
                     published)
