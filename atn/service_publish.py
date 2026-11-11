"""Publishing a service (docs/services_market.md, Decision 2026-11-12): the
one path the owner's app (WS ``register_service``) and the Agents'
``register_service`` command share.

The form a provider fills in: how buyers use it (``mode``: ``request`` or
``session``), the tool that does the work (``backing_tool``, a tool
registered on this daemon), what sets it apart (``settings``, such as the
model), the listing (``name``, ``description``, ``image_uri``), what it
charges for and its prices (``prices``: ``{measure: amount}``; a request
service may give ``price``, ATN per request), how much each voucher pays
ahead (``voucher``), and optionally a price recorded in the registry
(``registry_price``, unenforced). Amounts are ATN base units (6 decimals).
A codebase may keep these fields in ``autonet-services/<name>.json``
(``path``); the arguments given override the file's.

Publishing builds and signs the spec and its price list, registers the
spec's digest on chain (``ServiceRegistry.registerService``, under the
signing key's address: the provider, paid for every unit), and gossips
both once the chain numbers it. A chain failure leaves the service
published locally only, and says so.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

FIELDS = ("name", "description", "mode", "backing_tool", "settings", "measures",
          "prices", "price", "voucher", "registry_price", "image_uri",
          "version_of", "input_schema", "output_schema")


def _amount(value: Any, what: str) -> int:
    try:
        n = int(str(value).strip())
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a whole number of ATN base units") from None
    if n < 0:
        raise ValueError(f"{what} must not be negative")
    return n


def read_service_file(path: str) -> dict[str, Any]:
    """The fields of a service file (``autonet-services/<name>.json``)."""
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read the service file {p}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{p} must hold a JSON object")
    unknown = sorted(set(data) - set(FIELDS))
    if unknown:
        raise ValueError(f"{p}: unknown fields {unknown}")
    return data


def form_from(msg: dict[str, Any]) -> dict[str, Any]:
    """The publish form from a message: the service file's fields (when
    ``path`` names one), overridden by the message's. Amounts as ints."""
    fields: dict[str, Any] = {}
    if msg.get("path"):
        fields.update(read_service_file(str(msg["path"])))
    for k in FIELDS:
        if msg.get(k) is not None:
            fields[k] = msg[k]
    if not fields.get("backing_tool") and msg.get("tool_digest"):
        fields["backing_tool"] = msg["tool_digest"]
    mode = str(fields.get("mode") or "").strip()
    prices = fields.get("prices")
    if mode == "request" and prices is None and fields.get("price") is not None:
        prices = {"requests": fields["price"]}
    if not isinstance(prices, dict) or not prices:
        raise ValueError("name the prices: 'price' (ATN base units per request) for "
                         "a request service, or 'prices' ({measure: amount}) for a "
                         "session service")
    fields["prices"] = {str(k): _amount(v, f"the price of {k}") for k, v in prices.items()}
    fields["voucher"] = _amount(fields.get("voucher") or 0, "voucher")
    fields["registry_price"] = _amount(fields.get("registry_price") or 0, "registry_price")
    for key in ("settings", "input_schema", "output_schema"):
        if fields.get(key) is not None and not isinstance(fields[key], dict):
            raise ValueError(f"{key} must be an object")
    if fields.get("measures") is not None and not isinstance(fields["measures"], list):
        raise ValueError("measures must be a list of {measure, quantity}")
    return fields


async def publish_service(runtime: Any, *, author: str, signing_key: str,
                          form: dict[str, Any], offload: Any = None) -> dict[str, Any]:
    """Publish a service from ``form`` (``form_from``). Returns
    ``{digest, spec, price_list, on_chain, service_id?, tx_hash?, reason?}``
    or ``{"error"}``."""
    import asyncio

    from .on_chain import ServiceMarketClient

    store = getattr(runtime, "service_store", None)
    if store is None:
        return {"error": "Service store not available on this daemon."}
    try:
        built = store.register(
            name=str(form.get("name") or ""),
            description=str(form.get("description") or ""),
            mode=str(form.get("mode") or ""),
            author=author,
            backing_tool=str(form.get("backing_tool") or ""),
            prices=form["prices"],
            voucher=int(form["voucher"]),
            settings=form.get("settings"),
            measures=form.get("measures"),
            input_schema=form.get("input_schema"),
            output_schema=form.get("output_schema"),
            image_uri=str(form.get("image_uri") or ""),
            version_of=form.get("version_of"),
        )
    except (ValueError, RuntimeError) as exc:
        return {"error": str(exc)}
    out: dict[str, Any] = {"digest": built["digest"], "spec": built["spec"],
                           "price_list": built["price_list"]}
    smc = ServiceMarketClient(runtime._config.rpb)
    if not smc.registry_available:
        out.update({"on_chain": False,
                    "reason": "ServiceRegistry not configured (missing "
                              "service_registry_address or rpc_url), so the service "
                              "is published locally only and no buyer can find it."})
        return out
    if not signing_key:
        out.update({"on_chain": False,
                    "reason": f"No signing key held for '{author}', so the service "
                              "is published locally only."})
        return out

    def _run() -> Any:
        return asyncio.run(smc.register_service(
            signing_key, built["digest"], int(form.get("registry_price") or 0)))
    try:
        res = await asyncio.to_thread(_run) if offload is None else await offload(
            lambda: smc.register_service(signing_key, built["digest"],
                                         int(form.get("registry_price") or 0)))
    except Exception as exc:  # noqa: BLE001
        log.warning("register_service on-chain leg failed: %s", exc)
        out.update({"on_chain": False, "reason": str(exc)})
        return out
    if not res.get("success"):
        out.update({"on_chain": False,
                    "reason": f"On-chain registration failed: {res.get('error')}"})
        return out
    if res.get("service_id") is not None:
        store.set_service_id(built["digest"], int(res["service_id"]))
    out.update({"on_chain": True, "service_id": res.get("service_id"),
                "tx_hash": res.get("tx_hash")})
    return out
