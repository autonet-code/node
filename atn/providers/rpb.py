"""RPB Network Provider - routes inference through the P2P network."""

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

    This provider discovers sponsor nodes or decentralized model nodes
    via the gossip registry, selects the best available based on latency
    and budget, and routes the request over libp2p.

    Two resolution paths:
    - Path A (bootstrap): Sponsor nodes with centralized subscriptions
    - Path B (mature): Jurisdiction's VL-JEPA distributed inference
    """

    def __init__(
        self,
        agent_id: str = "",
        agent_address: str = "",
        p2p_host: Any = None,
        model: str = "",
    ):
        self._agent_id = agent_id
        self._agent_address = agent_address
        self._p2p = p2p_host
        self._model = model
        self._discovered_providers: list[dict] = []

    async def discover_providers(self, model: str = "") -> list[dict]:
        """Discover nodes offering the requested model via gossip registry."""
        target_model = model or self._model
        if not self._p2p:
            logger.warning("RPB provider: No P2P host configured")
            return []

        # Query gossip registry cache for nodes advertising this model
        # TODO: Wire to the actual P2P discovery when available
        logger.info(f"RPB: Discovering providers for model={target_model}")
        return self._discovered_providers

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

        Finds the best available provider node, routes the request
        over P2P, and returns the response.
        """
        target_model = model or self._model

        # Discover available providers
        providers = await self.discover_providers(target_model)

        if not providers:
            raise RuntimeError(
                f"RPB: No providers available for model={target_model}. "
                "Ensure sponsor nodes are online or fall back to direct providers."
            )

        # Select best provider (lowest latency + available budget)
        provider = self._select_best_provider(providers)

        # Build signed request
        request = {
            "messages": messages,
            "system": system,
            "model": target_model,
            "max_tokens": max_tokens,
            "tools": [{"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in tools] if tools else [],
            "temperature": temperature,
            "agent_address": self._agent_address,
        }

        # Route through P2P
        logger.info(f"RPB: Routing to provider {provider.get('peer_id', 'unknown')}")

        # TODO: Wire to actual P2P inference protocol
        # response = await self._p2p.request_inference(provider['peer_id'], request)

        raise NotImplementedError(
            "RPB P2P inference routing not yet wired. "
            "This provider will be functional when the /rpb/inference/1.0.0 protocol is implemented."
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

        # Sort by latency (ascending), then by remaining budget (descending)
        sorted_providers = sorted(
            providers,
            key=lambda p: (p.get("latency_ms", 9999), -p.get("remaining_budget", 0)),
        )
        return sorted_providers[0]

    @property
    def name(self) -> str:
        return "rpb"

    @property
    def is_available(self) -> bool:
        """RPB provider is available if P2P host is connected."""
        return self._p2p is not None
