"""Consumer side of the services market rail (docs/services_market.md).

The three mechanics a buyer needs, factored out of the agent tool
handlers so the ``service`` inference provider can reuse them verbatim
instead of re-implementing chain code:

  1. ``new_request_id()``     — a fresh bytes32 request id per call.
  2. ``pay_for_service()``    — sign a ``Substrate.payForService`` transfer
                                and return ``{tx_hash, request_id}``.
  3. ``request_service()``    — resolve the provider's on-chain wss
                                endpoint, dial it one-shot, send a
                                ``service_request`` carrying the payment
                                proof, and return the reply.
  4. ``lookup_service_ask()`` — read a service's on-chain ask by
                                (provider, spec_digest).

``atn/agent_tools.py`` (``pay_for_service`` / ``request_service``
agent tools) and ``atn/providers/service.py`` (the inference provider)
are both thin wrappers over these.

Everything here is key-agnostic: the CALLER supplies the signing key. An
agent tool passes the agent's key; the inference provider passes the
daemon owner's key, because buying cognition for the whole fleet is an
owner-level act (same doctrine as the sponsor pipe, where the dependent
identity IS the owner wallet).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any

log = logging.getLogger(__name__)

# Default wait for a provider daemon's reply to a service_request.
DEFAULT_REQUEST_TIMEOUT = 60.0


def new_request_id() -> str:
    """A fresh bytes32-shaped hex request id (two uuid4s = 32 bytes)."""
    return "0x" + (uuid.uuid4().bytes + uuid.uuid4().bytes).hex()


def normalize_digest(digest: str) -> str:
    """Lowercase, 0x-stripped form of a sha256/bytes32 hex digest.

    Chain reads return bytes32 as hex (with or without the ``0x``
    prefix depending on the web3 version); local spec digests are bare
    sha256 hex. Compare in this normalized space.
    """
    d = str(digest or "").strip().lower()
    return d[2:] if d.startswith("0x") else d


async def pay_for_service(
    config: Any,
    private_key: str,
    recipient: str,
    amount: int,
    request_id: str = "",
) -> dict[str, Any]:
    """Sign + submit ``Substrate.payForService``.

    Returns ``{tx_hash, request_id}`` on success, or ``{"error": ...}``.
    ``config`` is the ``AutonetConfig`` (``ATNConfig.rpb``).
    """
    from .on_chain import OnChainService

    if not recipient:
        return {"error": "Missing recipient for payment"}
    try:
        amount = int(amount)
    except (TypeError, ValueError):
        return {"error": "amount must be an integer (ATN base units)"}
    if amount <= 0:
        return {"error": "amount must be positive"}
    if not private_key:
        return {"error": "No signing key available for the payment"}

    request_id = str(request_id or "").strip() or new_request_id()

    svc = OnChainService(config)
    if not svc.available:
        return {"error": "Substrate not configured (missing substrate_address "
                         "or rpc_url)."}
    result = await svc.pay_for_service(private_key, recipient, amount, request_id)
    if not result.get("success"):
        return {"error": f"Payment failed: {result.get('error')}",
                "tx_hash": result.get("tx_hash")}
    return {
        "tx_hash": result.get("tx_hash"),
        "request_id": result.get("request_id") or request_id,
    }


async def request_service(
    config: Any,
    provider_address: str,
    spec_digest: str,
    args: dict[str, Any],
    *,
    tx_hash: str = "",
    request_id: str = "",
    client_address: str = "",
    timeout: float = DEFAULT_REQUEST_TIMEOUT,
    endpoint: str = "",
) -> dict[str, Any]:
    """Cross-daemon ``service_request``: resolve endpoint, dial, send, read.

    Returns ``{ok, result, error, endpoint}``; ``{"error": ...}`` on a
    setup failure. ``endpoint`` may be passed in to skip the chain read
    (the provider caches it).
    """
    from .on_chain import OnChainService

    if not provider_address:
        return {"error": "Missing provider_address"}
    if not spec_digest:
        return {"error": "Missing spec_digest"}
    if not isinstance(args, dict):
        return {"error": "'args' must be an object"}

    if not endpoint:
        svc = OnChainService(config)
        if not svc.available:
            return {"error": "Substrate not configured — cannot resolve the "
                             "provider endpoint."}
        endpoint = await svc.get_agent_endpoint(provider_address)
    if not endpoint:
        return {"error": f"Provider {provider_address[:10]} published no WS "
                         "endpoint on chain — cannot reach it."}

    try:
        import websockets
    except ImportError:
        return {"error": "websockets library not installed on this daemon."}

    msg_id = uuid.uuid4().hex[:12]
    frame: dict[str, Any] = {
        "type": "service_request",
        "msg_id": msg_id,
        "spec_digest": spec_digest,
        "request_id": request_id,
        "args": args,
        "client": client_address,
        "client_address": client_address,
    }
    if tx_hash:
        # Payment proof (direct payForService path).
        frame["tx_hash"] = tx_hash

    try:
        async with websockets.connect(endpoint, open_timeout=10) as ws:
            await ws.send(json.dumps(frame))
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                raw = await asyncio.wait_for(
                    ws.recv(), timeout=max(0.1, remaining))
                try:
                    reply = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                if reply.get("msg_id") == msg_id:
                    return {
                        "ok": bool(reply.get("ok")),
                        "result": reply.get("result"),
                        "error": reply.get("error"),
                        "endpoint": endpoint,
                    }
            return {"error": f"Provider did not reply within {int(timeout)}s.",
                    "endpoint": endpoint}
    except Exception as exc:  # noqa: BLE001 — network failure is a normal outcome
        return {"error": f"Service request failed: {exc}", "endpoint": endpoint}


async def lookup_service_ask(
    config: Any,
    provider_address: str,
    spec_digest: str,
) -> dict[str, Any]:
    """Read a registered service's on-chain ask by (provider, spec_digest).

    ``ServiceRegistry`` has no digest→id index, so this scans
    ``list_services`` and matches on both fields (a digest alone is not
    unique across providers). Returns
    ``{service_id, provider, ask_amount, active}`` or ``{"error": ...}``.
    """
    from .on_chain import ServiceMarketClient

    smc = ServiceMarketClient(config)
    if not smc.registry_available:
        return {"error": "ServiceRegistry not configured (missing "
                         "service_registry_address or rpc_url)."}
    target_digest = normalize_digest(spec_digest)
    target_provider = str(provider_address or "").strip().lower()
    try:
        services = await smc.list_services()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"ServiceRegistry read failed: {exc}"}
    for s in services:
        if normalize_digest(str(s.get("spec_digest") or "")) != target_digest:
            continue
        if target_provider and str(s.get("provider") or "").lower() != target_provider:
            continue
        try:
            ask_amount = int(str(s.get("ask_amount") or "0"))
        except (TypeError, ValueError):
            ask_amount = 0
        return {
            "service_id": s.get("service_id"),
            "provider": s.get("provider"),
            "ask_amount": ask_amount,
            "active": bool(s.get("active")),
        }
    return {"error": f"No service registered for digest "
                     f"{target_digest[:16]} under provider "
                     f"{target_provider[:10] or '(any)'}."}
