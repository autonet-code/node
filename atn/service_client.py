"""Consumer side of the services market rail (docs/services_market.md).

The three mechanics a buyer needs, factored out of the agent tool
handlers so the ``service`` inference provider can reuse them verbatim
instead of re-implementing chain code:

  1. ``new_request_id()``     — a fresh bytes32 request id per call.
  2. ``pay_for_service()``    — sign a ``Substrate.payForService`` transfer
                                and return ``{tx_hash, request_id}``.
  3. ``request_service()``    — send a ``service_request`` carrying the
                                payment proof to the provider and return
                                the reply: over libp2p to the provider's
                                on-chain peer id (direct, or one hop
                                through a connected relay), else over its
                                on-chain wss endpoint.
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

# The buyer signs each paid request with the key that paid for it, so the
# provider serves a payment only to its payer: a request id read off the
# chain is worthless without that key, and the signature binds the args.
REQUEST_AUTH_DOMAIN = "autonet:service_request:v1"


def request_auth_hash(request_id: str, spec_digest: str,
                      args: dict[str, Any], provider: str) -> bytes:
    """keccak over (domain, request id, spec digest, sha256 of the args'
    canonical JSON, provider address): what the payer signs."""
    import hashlib

    from eth_abi import encode
    from eth_utils import keccak, to_checksum_address

    rid_hex = str(request_id).lower().removeprefix("0x")
    if len(rid_hex) > 64:
        raise ValueError("request_id longer than 32 bytes")
    rid = bytes.fromhex(rid_hex.rjust(64, "0"))
    args_hash = hashlib.sha256(json.dumps(
        args, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")).digest()
    return keccak(encode(
        ["string", "bytes32", "string", "bytes32", "address"],
        [REQUEST_AUTH_DOMAIN, rid, str(spec_digest), args_hash,
         to_checksum_address(provider)]))


def sign_request(private_key: str, request_id: str, spec_digest: str,
                 args: dict[str, Any], provider: str) -> str:
    from eth_account import Account
    from eth_account.messages import encode_defunct

    msg = encode_defunct(primitive=request_auth_hash(
        request_id, spec_digest, args, provider))
    key = private_key if private_key.startswith("0x") else "0x" + private_key
    return "0x" + bytes(Account.sign_message(msg, key).signature).hex()


def recover_request_signer(signature: str, request_id: str, spec_digest: str,
                           args: dict[str, Any], provider: str) -> str:
    from eth_account import Account
    from eth_account.messages import encode_defunct

    msg = encode_defunct(primitive=request_auth_hash(
        request_id, spec_digest, args, provider))
    return Account.recover_message(msg, signature=signature)

log = logging.getLogger(__name__)

# Default wait for a provider daemon's reply to a service_request. Generous:
# a service may be a long inference or a slow tool.
DEFAULT_REQUEST_TIMEOUT = 300.0

# The daemon's libp2p host (AutonetHost), registered by the autonet bridge
# while it runs, so every caller of request_service gets the libp2p path
# without threading the host through. Tests pass ``p2p_host=`` instead.
_p2p_host: Any = None


def set_p2p_host(host: Any) -> None:
    global _p2p_host
    _p2p_host = host


def clear_p2p_host(host: Any) -> None:
    """Unregister ``host`` (only if it is still the registered one)."""
    global _p2p_host
    if _p2p_host is host:
        _p2p_host = None


def _running_host(host: Any) -> Any:
    host = host if host is not None else _p2p_host
    if host is None or getattr(host, "_trio_token", None) is None:
        return None
    return host


async def resolve_provider_peer_id(config: Any, provider_address: str,
                                   host: Any = None) -> str:
    """The provider agent's libp2p peer id, as bound on chain
    (``Substrate.getAgentPeerId``), base58. Falls back to the host's
    cached peer attribution; '' when neither knows it."""
    from nodes.common.peer_attribution import PeerAttribution

    from .on_chain import OnChainService

    try:
        svc = OnChainService(config)
        if svc.available:
            raw = await asyncio.to_thread(
                lambda: asyncio.run(svc.get_agent_peer_id(provider_address)))
            peer = PeerAttribution._decode_peer_id(raw or b"")
            if peer:
                return peer
    except Exception:  # noqa: BLE001 (a failed read falls through to the cache)
        log.debug("getAgentPeerId read failed for %s", provider_address,
                  exc_info=True)
    attribution = getattr(host, "_peer_attribution", None) if host else None
    if attribution is not None:
        try:
            rec = attribution.peer_for_agent(provider_address)
            if rec is not None:
                return rec.peer_id
        except Exception:  # noqa: BLE001
            pass
    return ""


async def p2p_request(host: Any, peer_id: str, frame: dict[str, Any],
                      timeout: float) -> dict[str, Any]:
    """Run ``host.request_service`` (trio, in the host's thread) from
    asyncio, blocking only an executor thread.

    Raises ``ConnectionError`` when the request reached no path to the
    provider, ``TimeoutError`` when it was delivered but not answered."""
    import trio

    token = getattr(host, "_trio_token", None)
    if token is None:
        raise ConnectionError("libp2p host not running")

    def call() -> dict:
        async def _do():
            return await host.request_service(peer_id, frame, timeout=timeout)
        return trio.from_thread.run(_do, trio_token=token)
    return await asyncio.get_running_loop().run_in_executor(None, call)


def host_service_handler(serve: Any, loop: asyncio.AbstractEventLoop,
                         timeout: float = 900.0) -> Any:
    """Wrap an asyncio ``serve(msg) -> reply`` coroutine function (the
    daemon's shared ``service_request`` handler) as the trio handler
    ``AutonetHost.set_service_handler`` expects. The call runs on
    ``loop``; the trio side waits in a worker thread."""
    import trio

    async def handle(msg: dict) -> dict:
        def run() -> dict:
            return asyncio.run_coroutine_threadsafe(
                serve(msg), loop).result(timeout)
        try:
            return await trio.to_thread.run_sync(run, abandon_on_cancel=True)
        except Exception as exc:  # noqa: BLE001
            log.warning("service request over libp2p failed: %s", exc)
            return {"msg_id": msg.get("msg_id"), "ok": False,
                    "error": f"provider failed to serve the request: {exc}"}
    return handle


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
    payer_key: str = "",
    p2p_host: Any = None,
    peer_id: str = "",
) -> dict[str, Any]:
    """Cross-daemon ``service_request``: send it to the provider, read the reply.

    Transport order (docs/services_market.md, "How requests travel"):

      1. libp2p, when this daemon's host is running and the provider's
         peer id is known (``peer_id``, else the on-chain
         ``getAgentPeerId``): direct if connected, else one hop through a
         connected relay. Works for providers with no public address, and
         the noise handshake proves the far end holds the peer id the
         provider bound on chain.
      2. The provider's on-chain wss endpoint, only if libp2p could not
         deliver the request. A request libp2p delivered is never resent
         (a paid request would be refused as a replay anyway), so a
         provider error or a reply timeout is final.

    Returns ``{ok, result, error, endpoint, transport}``; ``{"error": ...}``
    on a setup failure. ``endpoint`` may be passed in to skip the chain
    read (the provider caches it).
    """
    from .on_chain import OnChainService

    if not provider_address:
        return {"error": "Missing provider_address"}
    if not spec_digest:
        return {"error": "Missing spec_digest"}
    if not isinstance(args, dict):
        return {"error": "'args' must be an object"}

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
        # Payment proof (direct payForService path), signed by the payer.
        frame["tx_hash"] = tx_hash
        if payer_key and request_id:
            frame["payer_signature"] = sign_request(
                payer_key, request_id, spec_digest, args, provider_address)

    def _reply(reply: dict[str, Any], via: str, where: str) -> dict[str, Any]:
        return {
            "ok": bool(reply.get("ok")),
            "result": reply.get("result"),
            "error": reply.get("error"),
            "endpoint": where if via == "wss" else endpoint,
            "transport": via,
        }

    # --- 1. libp2p ---------------------------------------------------------
    p2p_error = ""
    host = _running_host(p2p_host)
    if host is not None:
        peer = peer_id or await resolve_provider_peer_id(
            config, provider_address, host)
        if not peer:
            p2p_error = "no peer id on chain for the provider"
        else:
            try:
                reply = await p2p_request(host, peer, frame, timeout)
                return _reply(reply, "libp2p", peer)
            except TimeoutError as exc:
                return {"error": f"Provider did not reply within "
                                 f"{int(timeout)}s over libp2p ({exc}).",
                        "endpoint": endpoint, "transport": "libp2p"}
            except ConnectionError as exc:
                p2p_error = str(exc)
            except Exception as exc:  # noqa: BLE001 (treated as undelivered)
                p2p_error = f"libp2p call failed: {exc}"
            log.info("service request to %s: libp2p could not deliver (%s); "
                     "trying the wss endpoint", provider_address[:10], p2p_error)

    # --- 2. wss endpoint ---------------------------------------------------
    if not endpoint:
        svc = OnChainService(config)
        if not svc.available:
            return {"error": "Substrate not configured — cannot resolve the "
                             "provider endpoint."
                             + (f" libp2p: {p2p_error}" if p2p_error else "")}
        endpoint = await svc.get_agent_endpoint(provider_address)
    if not endpoint:
        return {"error": f"Provider {provider_address[:10]} is unreachable: "
                         + (f"libp2p: {p2p_error}; " if p2p_error else
                            "libp2p not running here; ")
                         + "and it published no WS endpoint on chain."}

    try:
        import websockets
    except ImportError:
        return {"error": "websockets library not installed on this daemon."}

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
                    return _reply(reply, "wss", endpoint)
            return {"error": f"Provider did not reply within {int(timeout)}s.",
                    "endpoint": endpoint, "transport": "wss"}
    except Exception as exc:  # noqa: BLE001 — network failure is a normal outcome
        return {"error": f"Service request failed: {exc}"
                         + (f" (libp2p: {p2p_error})" if p2p_error else ""),
                "endpoint": endpoint, "transport": "wss"}


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
