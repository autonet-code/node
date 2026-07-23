"""Host-discovery subsystem — probes the local OS for available tools/services.

Ported from the deprecated native daemon. The runner (``HostDiscovery``) scans
~40 stdlib-only availability probes concurrently (capped), caches names-only
results to ``<data_dir>/security/host_scan.json``, and feeds the WS secrets
surface (``secrets_probe`` / ``secrets_import``).
"""

from __future__ import annotations

from .discovery import HostDiscovery, validate_secret_name
from .probe import HostProbe, ProbeResult

__all__ = ["HostDiscovery", "HostProbe", "ProbeResult", "validate_secret_name"]
