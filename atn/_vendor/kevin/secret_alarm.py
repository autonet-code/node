#!/usr/bin/env python3
"""secret_alarm — the durable NAMES-only alarm store (tripwire backstop).

Vendored minimal implementation. ``security_monitor.py`` does the live
scanning itself; what it needs from this module is the durable record:
``record_alarm(names, instance_name, when=...)`` appending to the JSON store
at the module-level ``ALARM_STORE`` path (the monitor points that constant at
``<data_dir>/security/secret_alarms.json`` before calling).

Contract (SAFETY-CRITICAL): a record carries secret NAMES only — never a
value, never a staged path. The store is a JSON list of
``{"names": [...], "instance": str, "ts": float}``.

``scan_text`` is the shared literal-substring matcher kept for parity with
the original kevin module (the monitor has its own tail-buffered variant).
"""

import json
import os
import threading

# Default store location; callers (security_monitor) override this module
# constant to relocate the store under their own data dir.
ALARM_STORE = os.path.join(
    os.path.expanduser("~"), ".atn", "keystore", "secret_alarms.json")

# Minimum value length worth scanning for (false-positive guard). Mirrored by
# security_monitor._MIN_VALUE_LEN.
_MIN_LEN = 6

_lock = threading.Lock()


def scan_text(text, values):
    """Literal substring scan: which of ``values`` (a {name: value} map)
    appear in ``text``. Returns the set of leaked NAMES. Values shorter than
    ``_MIN_LEN`` are skipped."""
    hits = set()
    if not text:
        return hits
    for name, value in (values or {}).items():
        if isinstance(value, str) and len(value) >= _MIN_LEN and value in text:
            hits.add(name)
    return hits


def _read_store(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def record_alarm(names, instance_name, when=None):
    """Append one NAMES-only alarm record to the durable store. Best-effort:
    never raises into the caller's scan path."""
    import time
    record = {
        "names": [str(n) for n in (names or [])],
        "instance": str(instance_name or ""),
        "ts": float(when) if when is not None else time.time(),
    }
    try:
        with _lock:
            path = ALARM_STORE
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)
            records = _read_store(path)
            records.append(record)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
    except OSError:
        pass
    return record


def read_alarms(store_path=None):
    """All recorded alarms (names-only records), oldest first."""
    with _lock:
        return _read_store(store_path or ALARM_STORE)
