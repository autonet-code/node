"""Surface — the integration entity for external real-time channels.

ATN has two kinds of integration, and until now only one had a name:

  • Connector (atn/connectors/): OUTBOUND, agent-driven, request/response. An
    agent calls a tool and gets a result. The thing an agent *uses*. Stateless
    from the framework's view. (Google Calendar, the member roster.)

  • Surface (this module): a long-lived, BIDIRECTIONAL bridge from an external
    real-time channel to agent *sessions*. The thing the world *reaches agents
    through*. It owns a persistent connection, reacts to inbound events, routes
    them to agents (gated at an input seam), and streams agent output back out
    via the EventBus.

VoiceService and ChatService are both Surfaces — same shape, different channel
(audio vs. a chat platform). This module names that shape.

Defining properties of a Surface:
  - owns a persistent connection (audio device; Discord gateway)
  - inbound-reactive: events arrive unprompted and must be routed/gated BEFORE
    reaching an agent
  - EventBus-bound for output (agent output → channel), not a tool return
  - has an INPUT SEAM — the one place an inbound message becomes
    runtime.send_agent_message. Access control / credits / rep-weighted
    consensus all live here, as an InputPolicy. This is where a deployment's
    gating belongs — NOT in a connector.
  - runtime-managed lifecycle: start()/stop(), config-gated at boot, same as
    the voice service.

The Surface protocol below is intentionally small: the lifecycle contract the
runtime drives. A Surface is free to add channel-specific machinery (routing
tables, renderers, adapters) on top. Policy is expressed via InputPolicy so the
gating seam is uniform across surfaces.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .events import EventBus


# ---------------------------------------------------------------------------
# Input-seam policy — the gating decision at a Surface's input seam.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PolicyDecision:
    """Outcome of evaluating an inbound message at a Surface's input seam."""

    allow: bool
    reason: str = ""          # human-readable; shown to the author when denied


@runtime_checkable
class InputPolicy(Protocol):
    """Decides whether an inbound message may reach an agent.

    Evaluated at the Surface input seam, before runtime.send_agent_message.
    Implementations: an open allow-all, an operator gate, a credit meter, a
    rep-weighted consensus gate. ``author`` is the platform-neutral sender (any
    object exposing at least ``id`` and ``is_bot``); ``agent_id`` is the target
    the message would be routed to.
    """

    def evaluate(self, author: Any, agent_id: str, content: str) -> PolicyDecision:
        ...


# ---------------------------------------------------------------------------
# Surface — the lifecycle contract the runtime drives.
# ---------------------------------------------------------------------------

@runtime_checkable
class Surface(Protocol):
    """A long-lived bidirectional bridge between an external channel and agents.

    The runtime constructs a Surface with the EventBus + itself, optionally
    config-gated at boot (like the voice service), and drives start()/stop().
    Everything else — the connection, inbound routing, the input seam, output
    rendering — is the Surface's own business.
    """

    async def start(self) -> None:
        """Bring the surface online: open the connection, subscribe to the
        EventBus for output, begin routing inbound events to agents."""
        ...

    async def stop(self) -> None:
        """Tear down: unsubscribe, close the connection."""
        ...

    # -- surface tools (optional) -------------------------------------------
    # A Surface is the only thing that can let an agent ACT BACK on the
    # external channel (e.g. read the channel's history to "read the room").
    # It contributes tools to the agents it binds and executes them itself.
    # This is what makes "give this agent a Discord presence" a capability
    # rather than a special case. Default: a surface contributes nothing.

    def agent_tools(self, agent_id: str) -> list[dict]:
        """Tool definitions ({name, description, input_schema}) this surface
        offers to `agent_id`. Tool names are prefixed `surface_` so the engine
        routes their calls back here. Return [] if the agent isn't bound to
        this surface or the surface offers no tools."""
        return []

    async def call_surface_tool(self, name: str, tool_input: dict, agent_id: str) -> dict:
        """Execute a surface tool (the `surface_`-prefixed name) for `agent_id`.
        Returns a result dict (or {"error": …})."""
        return {"error": f"surface tool {name} not handled"}
