"""Service provider: a native agent's model calls bought from a request
service (docs/services_market.md). The agent does ordinary chat
completions, and underneath each call is one ``service_request`` to a
request service whose tool answers ``{messages, max_tokens}``:

    price (the provider's signed price list)  ->  pay  ->  service_request

Payment is the buyer's credits with the provider when it holds them (a
payment channel, opened and topped up by the Agent with ``open_credits``;
``service_channels.request_on_credits``: a failed or refused call costs
nothing), else one direct payment per call. Native agents are an opt-in
mode; external Agents buy model access as sessions (an OpenAI-compatible
protocol) instead.

Configured with the two facts that name a purchase:

    providers:
      service:
        provider_address: "0x..."     # the serving agent's 0x
        spec_digest: "<sha256 hex>"   # the service spec being bought

Two ways a purchase reaches an agent, differing ONLY in whose key signs:

- **Daemon-level purchase**: ``providers.service`` in config.yaml, paid as
  the household's operator (the configured daemon key, else the vault-held
  operator key; Decision 2026-11-08).
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
  a local two-daemon demo possible. With chain configured, a call that
  cannot be paid (no key, no PaymentChannel, no channel opened) ABORTS
  before anything is sent.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
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
        channel_book: Any = None,
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
        # The service's ServiceRegistry number, found by one registry scan
        # and kept; the entry is re-read on every call for whether the
        # service is still active. ``_ask_amount`` is the last request price
        # read from the provider's price list (Decision 2026-11-12). The wss
        # endpoint is cached.
        self._service_id: int | None = None
        self._ask_amount: int | None = None
        self._endpoint: str = ""
        # The daemon's one ChannelBook (atn/service_channels.py), shared with
        # the request_service agent tool: the payer's credits per provider.
        # A path is opened on the first paid call (``book_at``: one book per
        # path).
        self._book = channel_book
        # The last call's payment: {credits, paid} on credits (paid 0 when
        # the request was not served, so its charge was refunded), {tx_hash,
        # paid} for a direct payment, {} for an unpaid call (no chain, or a
        # free service).
        self._last_payment: dict[str, Any] = {}

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
        """Last-read request price in ATN base units (None = not yet read)."""
        return self._ask_amount

    @property
    def payer_kind(self) -> str:
        """"owner" (daemon config purchase) or "agent" (per-agent binding)."""
        return self._payer_kind

    @property
    def payer_id(self) -> str:
        """The bound agent's id when ``payer_kind == "agent"``, else ""."""
        return self._payer_id

    @property
    def last_payment(self) -> dict[str, Any]:
        """The last call's payment: ``{credits, paid}`` on credits (``paid``
        0 when it was not served), ``{tx_hash, paid}`` paid directly, ``{}``
        when the call was unpaid."""
        return dict(self._last_payment)

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
        """The price of one request: the provider's signed price list for
        the service (Decision 2026-11-12), with the registry entry read for
        whether the service is still active. Raises ProviderError when the
        service can't be found or is not a request service."""
        from .. import service_client
        found: dict[str, Any] = {}
        if self._service_id is not None:
            found = await service_client.read_service(self._config, self._service_id)
            if (found.get("error")
                    or str(found.get("provider") or "").lower()
                    != self._provider_address.lower()
                    or service_client.normalize_digest(str(found.get("spec_digest") or ""))
                    != service_client.normalize_digest(self._spec_digest)):
                found = {}
                self._service_id = None
        if not found:
            found = await service_client.lookup_service_ask(
                self._config, self._provider_address, self._spec_digest)
            if found.get("service_id") is not None and not found.get("error"):
                self._service_id = int(found["service_id"])
        if found.get("error"):
            raise ProviderError(
                f"service provider: cannot find service {self._spec_digest[:16]} on "
                f"chain: {found['error']}", provider=self.name)
        if not found.get("active"):
            raise ProviderError(
                f"service provider: service {self._spec_digest[:16]} is retired on "
                "chain: nothing to buy.", provider=self.name)
        catalog = service_client._catalog
        price_list = (catalog.price_list(self._spec_digest, self._provider_address)
                      if catalog is not None else None)
        spec = catalog.spec(self._spec_digest) if catalog is not None else None
        if isinstance(spec, dict) and spec.get("mode") != "request":
            raise ProviderError(
                "service provider: that is a session service; this provider sends "
                "requests (bind a request service, or use the session's protocol "
                "directly)", provider=self.name)
        if price_list is None:
            raise ProviderError(
                f"service provider: no price list is known yet for "
                f"{self._spec_digest[:16]}; try again once the catalog has it",
                provider=self.name)
        from ..service_prices import request_price, voucher_size
        self._ask_amount = request_price(price_list)
        self._voucher = voucher_size(price_list)
        return self._ask_amount

    async def _refuse_if_sanctioned(self) -> None:
        """Raise when the provider belongs to a household the DAO sanctioned
        (same rule the request_service tool applies)."""
        sanctions = getattr(self, "_sanctions", None)
        if sanctions is None:
            from ..sanctions import SanctionList
            sanctions = self._sanctions = SanctionList.from_config(self._config)
        try:
            entry = await asyncio.to_thread(sanctions.check, self._provider_address)
        except Exception:
            entry = None
        if entry is not None:
            from ..sanctions import refusal
            raise ProviderError(
                "service provider: " + refusal(
                    self._provider_address, entry, "buy services from")["error"],
                provider=self.name)

    def _channel_book(self) -> Any:
        """The ChannelBook, opened from its path on first use."""
        if isinstance(self._book, (str, Path)):
            from ..service_channels import book_at
            self._book = book_at(Path(self._book))
        return self._book

    def _payer_address(self) -> str:
        """The address that pays: the signing key's (the channel's client),
        else the configured client address."""
        if self._owner_key:
            try:
                from eth_account import Account
                return str(Account.from_key(self._owner_key).address)
            except Exception:  # noqa: BLE001 - a malformed key pays nothing
                logger.debug("service provider: unusable signing key",
                             exc_info=True)
        return self._client_address

    async def _price(self) -> int:
        """This call's price in ATN base units, after every check that
        refuses a call before anything is sent; 0 when nothing is paid
        (no chain config, or a zero ask).

        Raises ProviderError when chain IS configured and the call cannot
        be paid: an unpaid request would just be refused by the provider's
        gate.
        """
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
            return 0

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
                "service provider: chain is configured but no signing key is "
                "available for the daemon-level purchase (the operator key, "
                "or autonet.private_key). For a child that pays from its own "
                "wallet, set a per-agent service_provider binding.",
                provider=self.name,
            )

        # Agents may not buy from a sanctioned household
        # (docs/accountability.md); this path pays like request_service does.
        await self._refuse_if_sanctioned()

        ask = await self._resolve_ask()
        if ask <= 0:
            # A zero ask means the provider is giving it away: no voucher.
            logger.info("service provider: ask is 0 — no payment needed")
            return 0
        return ask

    async def _purchase(self, args: dict[str, Any]) -> dict[str, Any]:
        """Send one call and pay for it: from the payer's credits with the
        provider when it has them (``request_on_credits``: a failed or
        refused call costs nothing), else with one direct payment."""
        from .. import service_client

        kwargs: dict[str, Any] = {}
        if self._timeout:
            kwargs["timeout"] = self._timeout
        self._last_payment = {}
        ask = await self._price()
        if ask <= 0:
            return await service_client.request_service(
                self._config, self._provider_address, self._spec_digest, args,
                client_address=self._client_address, endpoint=self._endpoint,
                payer_key=self._owner_key, **kwargs)

        from ..on_chain import ServiceMarketClient
        from ..service_channels import request_on_credits

        smc = ServiceMarketClient(self._config)
        book = self._channel_book()
        payer = self._payer_address()
        provider = self._provider_address
        if book is not None and smc.channel_available and book.get(payer, provider):
            reply = await request_on_credits(
                book, smc, self._config, self._owner_key, payer, provider,
                self._spec_digest, args, price=ask,
                voucher=int(getattr(self, "_voucher", 0) or 0),
                endpoint=self._endpoint, **kwargs)
            if reply.get("error") and not reply.get("delivered"):
                raise ProviderError(f"service provider: payment failed: {reply['error']}",
                                    provider=self.name)
            self._last_payment = {"credits": reply.get("credits"),
                                  "paid": ask if reply.get("ok") else 0}
            return reply
        request_id = service_client.new_request_id()
        paid = await service_client.pay_for_service(self._config, self._owner_key,
                                                    provider, ask, request_id=request_id)
        if paid.get("error"):
            raise ProviderError(f"service provider: payment failed: {paid['error']}",
                                provider=self.name)
        self._last_payment = {"tx_hash": paid.get("tx_hash"), "paid": ask}
        return await service_client.request_service(
            self._config, provider, self._spec_digest, args,
            tx_hash=str(paid.get("tx_hash") or ""),
            request_id=str(paid.get("request_id") or request_id),
            client_address=self._client_address or payer, endpoint=self._endpoint,
            payer_key=self._owner_key, **kwargs)

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
        """One completion: voucher → service_request → buffered result."""
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

        reply = await self._purchase(args)
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

        # The provider's envelope is {request_id, result}: the tool's output
        # is its ``result``.
        if isinstance(result, dict) and isinstance(result.get("result"), dict):
            result = result["result"]
        if not isinstance(result, dict):
            raise ProviderError(
                "service provider: malformed reply (no result object)",
                provider=self.name,
            )

        # The tool's output: {content, model?, usage?, stop_reason?}. The
        # spec's settings name the served model, but a tool may report none,
        # so fall back to our config label rather than error.
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
