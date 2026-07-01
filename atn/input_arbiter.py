"""InputArbiter — the runtime-owned single-writer gate above every Surface.

Output in ATN fans out to ALL surfaces (the EventBus, unchanged). INPUT is the
asymmetric half: at any instant at most one surface may drive an agent. That
"who holds the mic" decision is NOT a per-surface InputPolicy (those decide
*whether a given author* may speak on *their own* channel); it is a single
cross-surface arbitration that sits ABOVE every policy. This module owns it.

A Surface (voice / chat / a WS connection) is named by a SurfaceId. Surfaces
``register()`` when they come online and ``release_for()`` when they go away.
The arbiter tracks the connected set and a single ``_holder``. The mic is
acquired lazily — the first genuine inbound from a surface, when no one holds
it, auto-acquires — or explicitly via ``request_input``. An idle surface that
merely registered (e.g. a Discord bot sitting in a channel) never grabs the
mic just by existing.

Enforcement is a single backstop: SessionManager.send_agent_message consults
``is_active(surface)`` at its chokepoint. Trusted internal callers (the
orchestrator tool, the scheduler) pass ``surface=None`` and are never gated.

The switch-authorization seam (should requester X be allowed to *take* the mic
from holder Y?) is STUBBED here — ``AllowSwitch`` always says yes. Mic-preemption
policy is deferred; this builds the seam, not the check.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Protocol

from .events import Event, EventBus, EventType

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SurfaceId (SC-1) — the stable name for one input surface.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SurfaceId:
    """Identity of a single input surface.

    kind:     'voice' | 'chat' | 'ws'
    instance: for ws, the per-connection conn_id; for an in-process singleton
              surface, a fixed name ('local' for voice, 'discord' for chat).
    label:    human-readable, shown in input.* events / the snapshot.
    in_process: True for voice/chat (same process); False for a remote ws peer.

    ``token`` (kind:instance) is the canonical string key used everywhere the
    holder must be represented as a plain string (events, snapshot, the
    connected map).
    """

    kind: str
    instance: str
    label: str = ""
    in_process: bool = False

    @property
    def token(self) -> str:
        return f"{self.kind}:{self.instance}"


# ---------------------------------------------------------------------------
# Switch-authorization seam (STUBBED — mic-preemption policy is deferred).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SwitchContext:
    """Everything an authorizer needs to rule on a mic hand-over request."""

    requester: SurfaceId
    current_holder: SurfaceId | None
    requester_is_owner: bool = False
    credential: str | None = None


class SwitchAuthorizer(Protocol):
    """Decides whether ``requester`` may take the mic from ``current_holder``.

    Returns (allowed, reason). The reason is surfaced to the requester on deny.
    This is the seam for a future preemption policy (owner-priority, credits,
    consensus); today the only implementation is AllowSwitch.
    """

    async def authorize(self, ctx: "SwitchContext") -> tuple[bool, str]:
        ...


class AllowSwitch:
    """The default: any switch is allowed. The seam exists so a deployment can
    later slot in a real preemption check without re-plumbing the arbiter."""

    async def authorize(self, ctx: "SwitchContext") -> tuple[bool, str]:
        return (True, "")


# ---------------------------------------------------------------------------
# InputArbiter — runtime-owned, single-writer arbitration.
# ---------------------------------------------------------------------------

class InputArbiter:
    """Runtime-owned single-writer gate above every Surface's InputPolicy.

    State:
      _holder:    the SurfaceId currently allowed to drive agents (or None).
      _connected: token -> SurfaceId for every registered surface.

    The mic is acquired lazily (empty-holder auto-acquire on first genuine
    inbound) or explicitly (request_input). On disconnect, if the leaver held
    the mic, it auto-hands to the most-recently-registered remaining surface.
    """

    def __init__(self, events: EventBus,
                 authorizer: SwitchAuthorizer | None = None) -> None:
        self._events = events
        self._authorizer: SwitchAuthorizer = authorizer or AllowSwitch()
        self._holder: SurfaceId | None = None
        # Insertion-ordered: dict preserves registration order (py3.7+), which
        # is exactly the "most-recently-registered remaining surface" we hand
        # the mic to on an involuntary release.
        self._connected: dict[str, SurfaceId] = {}
        # Serializes grant decisions so two concurrent request_input calls
        # can't both win.
        self._grant_lock = asyncio.Lock()

    # -- registration -------------------------------------------------------

    def register(self, sid: SurfaceId) -> None:
        """Record a surface as connected. Does NOT grab the mic — an idle
        surface holds nothing until it actually speaks (or asks)."""
        self._connected[sid.token] = sid
        log.debug("[arbiter] registered %s (holder=%s)",
                  sid.token, self._holder.token if self._holder else None)

    def release_for(self, sid_or_token: "SurfaceId | str") -> None:
        """Drop a surface from the connected set. If it held the mic, auto-hand
        to the most-recently-registered remaining surface (empty otherwise) and
        emit input.active_changed{auto:true}."""
        token = sid_or_token.token if isinstance(sid_or_token, SurfaceId) else sid_or_token
        self._connected.pop(token, None)
        if self._holder is not None and self._holder.token == token:
            prior = self._holder
            # Most-recently-registered remaining surface = last dict entry.
            successor = next(reversed(self._connected.values()), None) \
                if self._connected else None
            self._holder = successor
            self._emit(EventType.INPUT_ACTIVE_CHANGED, {
                "active_writer": successor.token if successor else None,
                "active_label": successor.label if successor else "",
                "prior": prior.token,
                "auto": True,
            })
            log.info("[arbiter] %s released the mic; auto-handed to %s",
                     prior.token, successor.token if successor else "(none)")

    # -- queries ------------------------------------------------------------

    def is_active(self, sid: SurfaceId) -> bool:
        """True if ``sid`` currently holds the mic, OR the mic is free (empty
        holder auto-acquire: the first genuine inbound from a surface takes it).
        A None holder means "next speaker wins" — so we grant and record it."""
        if self._holder is None:
            # Auto-acquire on first genuine inbound.
            self._holder = sid
            self._connected.setdefault(sid.token, sid)
            self._emit(EventType.INPUT_ACTIVE_CHANGED, {
                "active_writer": sid.token,
                "active_label": sid.label,
                "prior": None,
                "auto": True,
            })
            log.info("[arbiter] %s auto-acquired the mic (was free)", sid.token)
            return True
        return self._holder.token == sid.token

    def holder(self) -> SurfaceId | None:
        return self._holder

    def holder_token(self) -> str | None:
        return self._holder.token if self._holder is not None else None

    def state(self) -> dict:
        """Serializable arbiter state for the snapshot + input_status."""
        return {
            "active_writer": self._holder.token if self._holder else None,
            "active_label": self._holder.label if self._holder else "",
            "connected": [
                {"token": s.token, "label": s.label,
                 "kind": s.kind, "in_process": s.in_process}
                for s in self._connected.values()
            ],
        }

    # -- explicit grant -----------------------------------------------------

    async def request_input(self, sid: SurfaceId, requester_is_owner: bool = False,
                            credential: str | None = None) -> dict:
        """Explicitly request the mic for ``sid``.

        Grants immediately if the mic is free or already held by the requester.
        Otherwise consults the SwitchAuthorizer (stubbed AllowSwitch today).
        Emits input.grant_requested, then input.granted or input.denied, and
        input.active_changed on a successful hand-over. Returns a dict the WS
        handler relays verbatim.
        """
        self._connected.setdefault(sid.token, sid)
        self._emit(EventType.INPUT_GRANT_REQUESTED, {
            "requester": sid.token,
            "requester_label": sid.label,
            "current_holder": self._holder.token if self._holder else None,
        })
        async with self._grant_lock:
            prior = self._holder
            if prior is not None and prior.token == sid.token:
                # Already holds it — idempotent grant.
                self._emit(EventType.INPUT_GRANTED, {
                    "active_writer": sid.token, "active_label": sid.label,
                    "prior": prior.token, "auto": False,
                })
                return {"granted": True, "active_writer": sid.token,
                        "holder": sid.token}
            if prior is None:
                allowed, reason = True, ""
            else:
                allowed, reason = await self._authorizer.authorize(SwitchContext(
                    requester=sid, current_holder=prior,
                    requester_is_owner=requester_is_owner, credential=credential))
            if not allowed:
                self._emit(EventType.INPUT_GRANT_DENIED, {
                    "requester": sid.token, "requester_label": sid.label,
                    "current_holder": prior.token if prior else None,
                    "reason": reason,
                })
                return {"granted": False, "reason": reason,
                        "holder": prior.token if prior else None}
            self._holder = sid
            self._emit(EventType.INPUT_GRANTED, {
                "active_writer": sid.token, "active_label": sid.label,
                "prior": prior.token if prior else None, "auto": False,
            })
            self._emit(EventType.INPUT_ACTIVE_CHANGED, {
                "active_writer": sid.token, "active_label": sid.label,
                "prior": prior.token if prior else None, "auto": False,
            })
            log.info("[arbiter] %s granted the mic (prior=%s)",
                     sid.token, prior.token if prior else None)
            return {"granted": True, "active_writer": sid.token,
                    "holder": sid.token}

    # -- internals ----------------------------------------------------------

    def _emit(self, etype: EventType, data: dict) -> None:
        """Fire an input.* event. Fire-and-forget: the arbiter's mutators are
        sync (called from sync seams like release_for on disconnect), so we
        schedule the emit on the running loop rather than awaiting it. Falls
        back to a direct history append if no loop is running (e.g. tests)."""
        evt = Event(type=etype, source="input_arbiter", data=data)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            loop.create_task(self._events.emit(evt))
        else:  # no loop (unit test / sync context) — record without awaiting
            self._events._history.append(evt)
