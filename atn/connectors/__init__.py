"""Bundled MCP connectors for ATN agents.

Provides auto-discovered ConnectorSpec entries for connectors that ship
with ATN.  Each bundled connector has a `connector.json` manifest that
declares its ID, metadata, and launch configuration.

Adding a new bundled connector:
  1. Create a directory under atn/connectors/ (e.g. atn/connectors/my_tool/)
  2. Add a connector.json with at minimum: {"id": "my_tool", "mode": "local", "entry": "server.py"}
  3. Add server.py — a FastMCP server that runs over stdio
  4. Restart ATN — the connector is auto-discovered and available via list_connectors
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Resolve at import time so paths are absolute regardless of cwd.
_CONNECTORS_DIR = Path(__file__).parent


def get_bundled_specs() -> dict:
    """Discover and return ConnectorSpec instances for bundled connectors.

    Scans atn/connectors/*/connector.json for manifest files.  Each manifest
    is parsed and converted to a ConnectorSpec with resolved absolute paths.

    Returns a dict of {connector_id: ConnectorSpec}.
    """
    from ..connectors_manager import ConnectorSpec

    specs: dict[str, ConnectorSpec] = {}

    for manifest_path in _CONNECTORS_DIR.glob("*/connector.json"):
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            log.warning("Failed to read connector manifest: %s", manifest_path)
            continue

        connector_id = data.get("id")
        if not connector_id:
            log.warning("Connector manifest missing 'id': %s", manifest_path)
            continue

        connector_dir = manifest_path.parent
        mode = data.get("mode", "local")

        # Resolve entry path to absolute for local mode
        entry = data.get("entry", "server.py")
        if mode == "local":
            entry = str(connector_dir / entry)

        specs[connector_id] = ConnectorSpec(
            mode=mode,
            package=data.get("package", ""),
            entry=entry,
            command=data.get("command"),
            args=data.get("args", []),
            env=data.get("env", {}),
            env_required=data.get("env_required", []),
            name=data.get("name", ""),
            description=data.get("description", ""),
        )

    if specs:
        log.debug("Discovered %d bundled connector(s): %s", len(specs), list(specs.keys()))

    return specs
