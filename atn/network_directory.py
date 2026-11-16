"""The published list of networks and their jurisdictions.

A data source is a set of files an indexer publishes (werule-indexer's
publisher, ``publisher/CONTRACT.md`` there): ``networks.json`` names each
network with its chain id, RPC, gas token and explorer, and each network's
``daos`` list holds one document per DAO. The daemon reads them to list
what can be joined and to turn "this DAO on this network" into what it
needs to join: the DAO Registry's address and the network's facts. It keeps
no table of chains of its own.

Anyone can publish such files for their own deployment and point the
daemon at them (``ATN_DATA_URL`` or ``autonet.data_url``). A jurisdiction
that no data source lists is joined by naming its Registry and an RPC URL
directly; this module is not involved then.

Plain HTTP reads; nothing here touches a chain. Objects named by hash never
change, so they are cached for the process's life.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_DATA_URL = "https://data.autonet.computer"
_TIMEOUT = 8.0
_ADDR = re.compile(r"^0x[0-9a-fA-F]{40}$")
_NETWORK_NAME = re.compile(r"^[A-Za-z0-9-]{1,64}$")
_HASH = re.compile(r"^[0-9a-f]{8,64}$")
# The Registry key that makes a DAO an Autonet jurisdiction.
SUBSTRATE_KEY = "jurisdiction.autonet.substrate"

_OBJECTS: dict[tuple[str, str, str], Any] = {}


class DirectoryError(Exception):
    """The data source could not answer (unreachable, or not such files)."""


def data_url(configured: str = "") -> str:
    """Where the published files live: ``ATN_DATA_URL``, else the daemon's
    ``autonet.data_url``, else the project's own publisher."""
    url = (os.environ.get("ATN_DATA_URL", "").strip() or str(configured or "").strip()
           or DEFAULT_DATA_URL)
    return url.rstrip("/")


def _get(url: str) -> Any:
    try:
        import httpx

        resp = httpx.get(url, timeout=_TIMEOUT, follow_redirects=True)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        raise DirectoryError(f"can't read {url}: {exc}") from exc


def _object(base: str, network: str, name: str) -> Any:
    if not _HASH.match(str(name or "")):
        raise DirectoryError(f"{base}: malformed object name in {network}")
    key = (base, network, name)
    if key not in _OBJECTS:
        _OBJECTS[key] = _get(f"{base}/{network}/o/{name}.json")
    return _OBJECTS[key]


def _network_row(net_id: str, d: dict[str, Any]) -> dict[str, Any] | None:
    """One networks.json entry as the daemon uses it, or None when it lacks
    what a jurisdiction needs (an RPC, a chain id, the jurisdiction factory)."""
    try:
        chain_id = int(d.get("chainId") or 0)
    except (TypeError, ValueError):
        chain_id = 0
    rpc = str(d.get("rpc") or "").strip()
    if not rpc or not chain_id or not _ADDR.match(str(d.get("jurisdictionFactory") or "")):
        return None
    return {
        "name": net_id,
        "chain_id": chain_id,
        "rpc_url": rpc,
        "gas_symbol": str(d.get("symbol") or d.get("nativeCurrency") or "").strip(),
        "explorer": str(d.get("blockExplorer") or "").strip(),
    }


def list_networks(base: str = "") -> list[dict[str, Any]]:
    """Every network the data source publishes that can hold jurisdictions:
    ``{name, chain_id, rpc_url, gas_symbol, explorer}``, by name."""
    base = data_url(base)
    doc = _get(f"{base}/networks.json")
    rows = []
    for n in (doc.get("networks") if isinstance(doc, dict) else None) or []:
        if not isinstance(n, dict) or not isinstance(n.get("data"), dict):
            continue
        net_id = str(n.get("id") or "")
        if not _NETWORK_NAME.match(net_id):
            continue
        row = _network_row(net_id, n["data"])
        if row is not None:
            rows.append(row)
    return sorted(rows, key=lambda r: r["name"].lower())


def get_network(network: str, base: str = "") -> dict[str, Any]:
    """One network by name (case-insensitive). DirectoryError when the data
    source does not list it."""
    nets = list_networks(base)
    for row in nets:
        if row["name"].lower() == str(network or "").strip().lower():
            return row
    raise DirectoryError(
        f"no network '{network}' in {data_url(base)} "
        f"(it lists: {', '.join(r['name'] for r in nets) or 'none'})")


def _dao_docs(base: str, network: str) -> list[dict[str, Any]]:
    """Every DAO document of ``network`` (each a doc entry's ``data`` with
    its id), from the network scope's ``daos`` list."""
    ptr = _get(f"{base}/{network}/ptr.json")
    head_name = ptr.get("head") if isinstance(ptr, dict) else None
    if not head_name:
        return []
    head = _object(base, network, head_name)
    index = ((head.get("lists") or {}).get("daos") or {}) if isinstance(head, dict) else {}
    docs: list[dict[str, Any]] = []
    for page in index.get("pages") or []:
        obj = _object(base, network, page.get("h"))
        for entry in (obj.get("docs") if isinstance(obj, dict) else None) or []:
            if (isinstance(entry, dict) and not entry.get("deleted")
                    and isinstance(entry.get("data"), dict)):
                docs.append({"id": str(entry.get("id") or ""), **entry["data"]})
    return docs


def list_jurisdictions(network: str, base: str = "", *, query: str = "",
                       limit: int = 20, offset: int = 0) -> dict[str, Any]:
    """The DAOs on ``network`` that carry an Autonet jurisdiction (a
    Substrate in their Registry), by name, a page at a time:
    ``{network, total, offset, limit, jurisdictions: [{name, dao,
    dao_registry}]}``. ``query`` keeps the names or addresses containing it."""
    base = data_url(base)
    net = get_network(network, base)
    rows = []
    for d in _dao_docs(base, net["name"]):
        registry = d.get("registry")
        if not isinstance(registry, dict):
            continue
        dao = str(d.get("address") or d.get("id") or "")
        dao_registry = str(d.get("registryAddress") or "")
        if not (_ADDR.match(str(registry.get(SUBSTRATE_KEY) or ""))
                and _ADDR.match(dao_registry) and _ADDR.match(dao)):
            continue
        rows.append({"name": str(d.get("name") or ""), "dao": dao,
                     "dao_registry": dao_registry})
    q = str(query or "").strip().lower()
    if q:
        rows = [r for r in rows
                if q in r["name"].lower() or q in r["dao"].lower()]
    rows.sort(key=lambda r: (r["name"].lower(), r["dao"].lower()))
    limit = max(1, min(int(limit or 20), 100))
    offset = max(0, int(offset or 0))
    return {"network": net["name"], "total": len(rows), "offset": offset,
            "limit": limit, "jurisdictions": rows[offset:offset + limit]}


def find_jurisdiction(network: str, dao: str, base: str = "") -> dict[str, Any]:
    """What joining "``dao`` on ``network``" needs, from the data source:
    ``{network: {name, chain_id, rpc_url, gas_symbol, explorer}, name, dao,
    dao_registry}``. DirectoryError when either is not listed."""
    base = data_url(base)
    net = get_network(network, base)
    want = str(dao or "").strip().lower()
    if not _ADDR.match(want):
        raise DirectoryError(f"'{dao}' is not a DAO address")
    offset = 0
    while True:
        page = list_jurisdictions(net["name"], base, limit=100, offset=offset)
        for row in page["jurisdictions"]:
            if row["dao"].lower() == want:
                return {"network": net, **row}
        offset += len(page["jurisdictions"])
        if not page["jurisdictions"] or offset >= page["total"]:
            break
    raise DirectoryError(
        f"no Autonet jurisdiction with DAO {dao} on {net['name']} in {base}")
