"""Service provider — inference bought off the marketplace rail.

Consumer side of ``docs/services_market.md``, "Decision (2026-07-26): LLM
inference as a marketplace service" (item 2). A purchased marketplace
service backs an agent's PROVIDER: the agent does ordinary chat
completions, and underneath each call is

    read ask (cached)  →  payForService  →  service_request  →  completion

Indistinguishable from any other model from the agent's seat. Any wallet
holding ATN can buy cognition — no sponsor relationship required.

Configured with the two facts that name a purchase:

    providers:
      service:
        provider_address: "0x..."     # the serving agent's 0x
        spec_digest: "<sha256 hex>"   # the service spec being bought

Two ways a purchase reaches an agent, differing ONLY in whose key signs:

- **Daemon-level (owner) purchase** — ``providers.service`` in config.yaml,
  signed with ``autonet.private_key``. Buying cognition for the fleet is an
  owner act, the same doctrine as the sponsor pipe where the dependent
  identity IS the owner wallet.
- **Per-agent binding** (ratified 2026-07-26, employer-chooses-the-tool) —
  ``AgentDefinition.service_provider = {provider_address, spec_digest}``, set
  ONLY by the agent's PARENT, signed with the BOUND AGENT'S OWN key. A parent
  may acquire a marketplace inference service and provision a CHILD bound to
  it, then scrutinize the child's output from outside the purchased substrate.
  The child pays from its own wallet: spend authority is literal token custody,
  so the parent funds that wallet on-chain and can starve it by not refilling.

In BOTH cases the agent has no say in the matter — an agent must never switch
its own substrate, and there is no self-set surface anywhere on this rail.
- **Fresh request_id per call.** The provider-side gate persists seen
  request ids, so a reused id is a replay and is refused.
- **Non-streaming by design (v1).** Responses buffer; the agent loop
  consumes whole completions anyway. ``send_stream`` yields the whole
  completion once via ``on_chunk``.
- **No tool calls (v1).** The wire contract carries ``messages`` and
  ``max_tokens`` only; a tools list is dropped with a warning rather
  than silently pretending tool use is available.
- **Degrades honestly.** With NO chain config the payment is SKIPPED
  with a loud warning and the request still goes out — the provider-side
  gate degrades open in exactly the same condition, which is what makes
  a local two-daemon demo possible. With chain configured, a failed
  payment ABORTS the call.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import (
    Provider,
    ProviderError,
    ProviderResponse,
    ToolDefinition,
    Usage,
)

logger = logging.getLogger(__name__)


class ServiceProvider(Provider):
    """Provider backed by a purchased marketplace inference service."""

    def __init__(
        self,
        *,
        config: Any,
        provider_address: str = "",
        spec_digest: str = "",
        model: str = "",
        owner_private_key: str = "",
        client_address: str = "",
        timeout: float = 0.0,
        signing_key: str = "",
        payer_kind: str = "owner",
        payer_id: str = "",
    ) -> None:
        # ``config`` is the AutonetConfig (ATNConfig.rpb) — the chain surface.
        self._config = config
        self._provider_address = (provider_address or "").strip()
        self._spec_digest = (spec_digest or "").strip()
        self._model = model or ""
        # The signing identity. ``signing_key`` is the general form (an agent's
        # own key for a per-agent binding); ``owner_private_key`` is the
        # daemon-owner spelling kept for the config-driven purchase. They are
        # the same slot — whoever holds the key is who pays, which is the whole
        # of spend authority on this rail.
        self._owner_key = (signing_key or owner_private_key or "").strip()
        self._client_address = (client_address or "").strip()
        self._timeout = timeout or 0.0
        # Provenance, for error messages and diagnostics only: "owner" (daemon
        # config purchase) or "agent" (per-agent parent-set binding).
        self._payer_kind = payer_kind or "owner"
        self._payer_id = payer_id or ""
        # Cached on-chain ask (base units) and the resolved wss endpoint.
        # Both are stable for a registered service, so one read serves many
        # completions; ``updateServiceAsk`` / ``updateEndpoint`` are rare and
        # a stale value fails loudly at the gate rather than silently.
        self._ask_amount: int | None = None
        self._endpoint: str = ""

    # ------------------------------------------------------------------
    # Provider interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "service"

    @property
    def is_available(self) -> bool:
        """Configured means: we know WHO to call and WHAT we bought."""
        return bool(self._provider_address and self._spec_digest)

    @property
    def provider_address(self) -> str:
        return self._provider_address

    @property
    def spec_digest(self) -> str:
        return self._spec_digest

    @property
    def cached_ask(self) -> int | None:
        """Last-read on-chain ask in ATN base units (None = not yet read)."""
        return self._ask_amount

    @property
    def payer_kind(self) -> str:
        """"owner" (daemon config purchase) or "agent" (per-agent binding)."""
        return self._payer_kind

    @property
    def payer_id(self) -> str:
        """The bound agent's id when ``payer_kind == "agent"``, else ""."""
        return self._payer_id

    # ------------------------------------------------------------------
    # Chain helpers
    # ------------------------------------------------------------------

    def _chain_configured(self) -> bool:
        from ..on_chain import OnChainService
        try:
            return bool(OnChainService(self._config).available)
        except Exception:  # noqa: BLE001 — a broken config is "not configured"
            logger.debug("service provider: chain availability probe failed",
                         exc_info=True)
            return False

    async def _resolve_ask(self) -> int:
        """The service's on-chain ask, cached after the first read.

        Raises ProviderError when the service can't be found on chain —
        paying an unknown ask is not something to guess at.
        """
        if self._ask_amount is not None:
            return self._ask_amount
        from .. import service_client
        found = await service_client.lookup_service_ask(
            self._config, self._provider_address, self._spec_digest)
        if found.get("error"):
            raise ProviderError(
                f"service provider: cannot read the on-chain ask for "
                f"{self._spec_digest[:16]} — {found['error']}",
                provider=self.name,
            )
        if not found.get("active"):
            raise ProviderError(
                f"service provider: service {self._spec_digest[:16]} is "
                f"retired on chain — nothing to buy.",
                provider=self.name,
            )
        self._ask_amount = int(found.get("ask_amount") or 0)
        logger.info("service provider: ask for %s = %d ATN base units "
                    "(service_id=%s)", self._spec_digest[:16],
                    self._ask_amount, found.get("service_id"))
        return self._ask_amount

    async def _pay(self, max_tokens: int) -> dict[str, Any]:
        """Pay this call's ask. Returns the payment proof, or {} when the
        payment was legitimately skipped (no chain config).

        Raises ProviderError when chain IS configured and payment fails —
        an unpaid request would just be refused by the provider's gate.
        """
        from .. import service_client

        if not self._chain_configured():
            logger.warning(
                "SERVICE PROVIDER PAYMENT SKIPPED: no chain config "
                "(substrate_address/rpc_url unset) — sending an UNPAID "
                "inference request to %s for spec %s. The provider gate "
                "degrades open in the same condition; this is local-dev "
                "only and MUST NOT be relied on against a real counterparty.",
                self._provider_address[:10] or "(unset)",
                self._spec_digest[:16] or "(unset)",
            )
            return {}

        if not self._owner_key:
            if self._payer_kind == "agent":
                raise ProviderError(
                    f"service provider: chain is configured but agent "
                    f"'{self._payer_id or '?'}' has no stored wallet key — a "
                    f"bound child pays for its own inference, so it needs its "
                    f"own key (register it on-chain first).",
                    provider=self.name,
                )
            raise ProviderError(
                "service provider: chain is configured but no owner signing "
                "key is available (autonet.private_key) — the daemon-level "
                "purchase is an owner-level act. For a child that pays from "
                "its own wallet, set a per-agent service_provider binding.",
                provider=self.name,
            )

        ask = await self._resolve_ask()
        if ask <= 0:
            # A zero ask means the provider is giving it away; nothing to
            # sign, and payForService rejects a zero amount anyway.
            logger.info("service provider: ask is 0 — no payment needed")
            return {}

        # Fresh request id per call: the provider persists seen ids, so a
        # reused one is a replay and is refused.
        request_id = service_client.new_request_id()
        paid = await service_client.pay_for_service(
            self._config,
            self._owner_key,
            self._provider_address,
            ask,
            request_id=request_id,
        )
        if paid.get("error"):
            raise ProviderError(
                f"service provider: payment failed, aborting the call — "
                f"{paid['error']}",
                provider=self.name,
            )
        logger.info("service provider: %s paid %d for request %s (tx %s)",
                    f"agent {self._payer_id}" if self._payer_kind == "agent"
                    else "owner",
                    ask, request_id[:12], str(paid.get("tx_hash"))[:16])
        return {
            "tx_hash": str(paid.get("tx_hash") or ""),
            "request_id": str(paid.get("request_id") or request_id),
        }

    # ------------------------------------------------------------------
    # send
    # ------------------------------------------------------------------

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
        """One completion: pay → service_request → buffered result."""
        from .. import service_client

        if not self.is_available:
            raise ProviderError(
                "service provider is not configured: both provider_address "
                "and spec_digest are required (providers.service in "
                "config.yaml).",
                provider=self.name,
            )
        if tools:
            # v1 wire contract is {messages, max_tokens}. Be loud rather
            # than let an agent believe its tools are on the table.
            logger.warning(
                "service provider: %d tool definition(s) dropped — the v1 "
                "inference-service wire contract carries messages/max_tokens "
                "only. Point tool-using agents at a local provider.",
                len(tools),
            )

        wire_messages = self._wire_messages(messages)
        if not wire_messages:
            raise ProviderError(
                "service provider: 'messages' must be non-empty",
                provider=self.name,
            )
        proof = await self._pay(max_tokens)

        # Wire contract (docs/services_market.md, provider-side dispatch):
        # {messages (required, non-empty), max_tokens?, system?, temperature?}.
        # max_tokens is clamped provider-side to the spec's declared cap.
        args: dict[str, Any] = {"messages": wire_messages}
        if max_tokens:
            args["max_tokens"] = int(max_tokens)
        if system:
            args["system"] = system
        if temperature:
            args["temperature"] = float(temperature)

        kwargs: dict[str, Any] = {}
        if self._timeout:
            kwargs["timeout"] = self._timeout
        reply = await service_client.request_service(
            self._config,
            self._provider_address,
            self._spec_digest,
            args,
            tx_hash=proof.get("tx_hash", ""),
            request_id=proof.get("request_id", ""),
            client_address=self._client_address,
            endpoint=self._endpoint,
            **kwargs,
        )
        # Cache the resolved endpoint for subsequent calls.
        if reply.get("endpoint"):
            self._endpoint = str(reply["endpoint"])

        result = reply.get("result")
        # Two failure shapes: a transport/gate refusal carries a top-level
        # ``error``, a provider-side dispatch failure comes back as
        # ``{error}`` inside the result envelope. Prefer whichever is
        # present so the message the agent sees is the real cause.
        inner_error = result.get("error") if isinstance(result, dict) else None
        if reply.get("error") or inner_error or not reply.get("ok"):
            raise ProviderError(
                f"service provider: request to "
                f"{self._provider_address[:10]} failed — "
                f"{reply.get('error') or inner_error or 'provider returned not-ok'}",
                provider=self.name,
            )

        if not isinstance(result, dict):
            raise ProviderError(
                "service provider: malformed reply (no result object)",
                provider=self.name,
            )

        # Success envelope (provider-side): {request_id, content, model, usage,
        # stop_reason, max_tokens}. An EMPTY served model is legitimate — the
        # spec may leave the model unset and let the seller's daemon resolve
        # its own default — so fall back to our config label, never error.
        usage_raw = result.get("usage") or {}
        served_model = str(result.get("model") or "") or (model or self._model)
        self._active_model = served_model
        return ProviderResponse(
            text=str(result.get("content") or ""),
            model=served_model,
            stop_reason=str(result.get("stop_reason") or "end_turn"),
            usage=Usage(
                input_tokens=int(usage_raw.get("input_tokens") or 0),
                output_tokens=int(usage_raw.get("output_tokens") or 0),
                cache_read_tokens=int(usage_raw.get("cache_read_tokens") or 0),
                cache_creation_tokens=int(
                    usage_raw.get("cache_creation_tokens") or 0),
            ),
            raw=result,
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
        """v1 is deliberately non-streaming: buffer, then emit once."""
        response = await self.send(
            messages=messages,
            system=system,
            model=model,
            max_tokens=max_tokens,
            tools=tools,
            temperature=temperature,
        )
        if on_chunk and response.text:
            await on_chunk(response.text)
        return response

    # ------------------------------------------------------------------
    # Wire shaping
    # ------------------------------------------------------------------

    @staticmethod
    def _wire_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Flatten canonical messages into the {role, content: str} shape the
        service wire contract carries.

        The system prompt is NOT folded in here — the contract carries it as
        its own ``system`` arg. Structured content blocks are flattened to
        their text parts (v1 has no tool-use round trip, so nothing else
        survives the wire anyway); a block-only message that flattens to
        nothing is dropped rather than sent as an empty turn.
        """
        out: list[dict[str, Any]] = []
        for m in messages or []:
            role = str(m.get("role") or "user")
            content = m.get("content")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                parts: list[str] = []
                for block in content:
                    if isinstance(block, str):
                        parts.append(block)
                    elif isinstance(block, dict):
                        if block.get("type") == "text" and block.get("text"):
                            parts.append(str(block["text"]))
                        elif block.get("type") == "tool_result":
                            parts.append(str(block.get("content") or ""))
                text = "\n".join(p for p in parts if p)
            else:
                text = "" if content is None else str(content)
            out.append({"role": role, "content": text})
        return out
