"""Chat-interface subsystem — binds autonet agent conversations to chat
platforms (Discord first; Telegram/Matrix/Signal later).

This is the in-framework counterpart to the VoiceService: an external real-time
surface bound to agent sessions at the EventBus seam. Output flows agent →
EventBus → platform; input flows platform → ``runtime.send_agent_message`` →
agent. One dedicated channel maps to the orchestrator's conversation, top-level
delegates get their own threads, and nested sub-agents render as pinned tiles —
mirroring autonet's own orchestrator+delegate hierarchy.

Platform specifics live behind the platform-neutral ``MessagingClient``
protocol (``protocol.py``); the service never imports discord.py directly.
Adapters live under ``adapters/``.

DОrg-specific application logic (member-agent registration, credit metering)
does NOT live here — it sits at the input seam as an ``InputPolicy`` supplied
by the deployment.
"""

from .protocol import (
    Author, Channel, ChannelKind, EmbedSpec, Message, MessagingClient,
    PostResult, StatusEmbedHandle,
)
from .policy import AllowAll, InputPolicy, OperatorGate, PolicyDecision
from .routing import (
    ORCHESTRATOR, agent_depth, is_top_level_delegate, top_level_delegate,
)
from .chat_service import ChatService

__all__ = [
    "ChatService",
    "MessagingClient",
    "Author", "Channel", "ChannelKind", "EmbedSpec", "Message",
    "PostResult", "StatusEmbedHandle",
    "InputPolicy", "AllowAll", "OperatorGate", "PolicyDecision",
    "ORCHESTRATOR", "agent_depth", "is_top_level_delegate", "top_level_delegate",
]
