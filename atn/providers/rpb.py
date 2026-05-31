"""RPB Network Provider - routes inference through the P2P network.

Path A (v1): Sponsor nodes proxy inference through their centralized API
subscriptions to dependent nodes via P2P. The requesting node discovers
sponsor nodes via gossip, selects the best by latency and budget, and
sends the request over libp2p. The sponsor node forwards it to its own
provider (Claude, OpenAI, etc.) and returns the response.

Path B (mature, future): Two-speed architecture with distributed VL-JEPA
latent reasoning. Not implemented in v1.
"""

import logging
from typing import Any, Optional

from .base import (
    Provider,
    ProviderResponse,
    ToolCall,
    ToolDefinition,
    Usage,
)

logger = logging.getLogger(__name__)


class RPBNetworkProvider(Provider):
    """Provider that routes inference requests through the RPB P2P network.

    v1 (Path A): Discovers sponsor nodes via gossip registry, selects
    the best available by latency and budget, and routes the request
    over libp2p. The sponsor node serves it through their own provider.

    No-chaining rule: Inference received through RPB cannot be
    re-advertised through RPB. This prevents infinite loops.
    """

    def __init__(
        self,
        agent_id: str = "",
        agent_address: str = "",
        p2p_host: Any = None,
        model: str = "",
        sponsor_address: str = "",
    ):
        self._agent_id = agent_id
        self._agent_address = agent_address
        self._p2p = p2p_host
        self._model = model
        # When set, the dependent routes only to this specific sponsor agent
        # (the employer named in config), instead of any is_sponsor peer.
        self._sponsor_address = (sponsor_address or "").strip().lower()
        self._discovered_providers: list[dict] = []
        self._remaining_budget_tokens: int = -1

    async def discover_providers(self, model: str = "") -> list[dict]:
        """Discover sponsor nodes offering the requested model via gossip.

        Scans the P2P host's known capabilities for nodes that advertise
        agents with the requested model and have sponsor status.
        """
        target_model = model or self._model
        if not self._p2p:
            logger.warning("RPB provider: No P2P host configured")
            return []

        providers = []
        capabilities = getattr(self._p2p, "_known_capabilities", {})
        latency_tracker = getattr(self._p2p, "_latency_tracker", None)

        for peer_id, cap in capabilities.items():
            # Check if this peer advertises the target model
            for agent_ad in getattr(cap, "agents", []):
                def _field(name, default=""):
                    if hasattr(agent_ad, name):
                        return getattr(agent_ad, name)
                    if isinstance(agent_ad, dict):
                        return agent_ad.get(name, default)
                    return default

                ad_model = _field("model", "")
                ad_is_sponsor = _field("is_sponsor", False)
                ad_address = (_field("address", "") or "").strip().lower()

                # If the dependent named a specific sponsor, only that one
                # qualifies — employment is bound, not market-discovered.
                if self._sponsor_address and ad_address != self._sponsor_address:
                    continue

                # When targeting a named sponsor, the employer dictates the
                # model: accept the sponsor regardless of the dependent's
                # requested model. Otherwise fall back to model matching.
                model_ok = (
                    bool(self._sponsor_address and ad_address == self._sponsor_address)
                    or ad_model == target_model
                    or not target_model
                )
                if model_ok:
                    latency = 9999.0
                    if latency_tracker:
                        lat = latency_tracker.get_latency(peer_id)
                        if lat and lat.ema_rtt_ms > 0:
                            latency = lat.ema_rtt_ms

                    providers.append({
                        "peer_id": peer_id,
                        "model": ad_model,
                        "address": ad_address,
                        "is_sponsor": ad_is_sponsor,
                        "latency_ms": latency,
                        "remaining_budget": agent_ad.get("remaining_budget", 0) if isinstance(agent_ad, dict) else 0,
                    })

        # Prefer sponsors, then sort by latency
        providers.sort(key=lambda p: (not p.get("is_sponsor", False), p.get("latency_ms", 9999)))
        self._discovered_providers = providers

        logger.info("RPB: Discovered %d providers for model=%s%s",
                    len(providers), target_model,
                    f" (sponsor={self._sponsor_address[:10]})" if self._sponsor_address else "")
        return providers

    async def send(
        self,
        *,
        messages: list[dict[str, Any]],
        system: str = "",
        model: str = "",
        max_tokens: int = 4096,
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
    ) -> ProviderResponse:
        """Send an inference request through the RPB network.

        Discovers sponsor nodes, selects the best one, and routes the
        request over P2P. The sponsor node serves it through their own
        centralized provider and returns the response.
        """
        target_model = model or self._model

        # Discover available providers
        providers = await self.discover_providers(target_model)

        if not providers:
            raise RuntimeError(
                f"RPB: No providers available for model={target_model}. "
                "Ensure sponsor nodes are online or fall back to direct providers."
            )

        # Select best provider (lowest latency sponsor)
        provider = self._select_best_provider(providers)

        # Build request payload
        request = {
            "messages": messages,
            "system": system,
            "model": target_model,
            "max_tokens": max_tokens,
            "tools": [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in tools
            ] if tools else [],
            "temperature": temperature,
            "agent_address": self._agent_address,
            "via_rpb": True,  # No-chaining flag
        }

        # Route through P2P
        peer_id = provider["peer_id"]
        logger.info("RPB: Routing to provider %s (latency=%.0fms)",
                     str(peer_id)[:16], provider.get("latency_ms", 0))

        response = await self._p2p.request_inference(peer_id, request)

        # Parse response into ProviderResponse
        tool_calls = []
        for tc in response.get("tool_calls", []):
            tool_calls.append(ToolCall(
                id=tc.get("id", ""),
                name=tc.get("name", ""),
                input=tc.get("input", {}),
            ))

        usage_data = response.get("usage", {})

        # Track how much of the sponsor's grant remains (-1 = unlimited).
        self._remaining_budget_tokens = response.get("remaining_budget_tokens", -1)
        if 0 <= self._remaining_budget_tokens < 10_000:
            logger.warning(
                "RPB: sponsor grant low — %d tokens remaining",
                self._remaining_budget_tokens,
            )

        return ProviderResponse(
            text=response.get("text", ""),
            tool_calls=tool_calls if tool_calls else None,
            usage=Usage(
                input_tokens=usage_data.get("input_tokens", 0),
                output_tokens=usage_data.get("output_tokens", 0),
            ),
            model=response.get("model", target_model),
            stop_reason=response.get("stop_reason", "end_turn"),
        )

    async def send_stream(
        self,
        *,
        messages: list[dict[str, Any]],
        system: str = "",
        model: str = "",
        max_tokens: int = 4096,
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        on_chunk: Any = None,
        on_thinking: Any = None,
    ) -> ProviderResponse:
        """Streaming send — delegates to send() (P2P streaming not yet implemented)."""
        return await self.send(
            messages=messages,
            system=system,
            model=model,
            max_tokens=max_tokens,
            tools=tools,
            temperature=temperature,
        )

    def _select_best_provider(self, providers: list[dict]) -> dict:
        """Select the best provider based on latency and budget."""
        if not providers:
            raise ValueError("No providers to select from")

        # Sort by: sponsor first, then latency ascending, then budget descending
        sorted_providers = sorted(
            providers,
            key=lambda p: (
                not p.get("is_sponsor", False),
                p.get("latency_ms", 9999),
                -p.get("remaining_budget", 0),
            ),
        )
        return sorted_providers[0]

    @property
    def name(self) -> str:
        return "rpb"

    @property
    def is_available(self) -> bool:
        """RPB provider is available if P2P host is connected."""
        return self._p2p is not None
