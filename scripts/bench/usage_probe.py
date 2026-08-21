"""Claude Max usage probe — replicates atn bridge.py refresh_usage()."""
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx

creds = json.loads((Path.home() / ".claude" / ".credentials.json").read_text())
token = creds["claudeAiOauth"]["accessToken"]
resp = httpx.post(
    "https://api.anthropic.com/v1/messages",
    headers={
        "Authorization": f"Bearer {token}",
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "oauth-2025-04-20",
        "content-type": "application/json",
    },
    json={"model": "claude-haiku-4-5", "max_tokens": 1,
          "messages": [{"role": "user", "content": "hi"}]},
    timeout=30,
)
h = resp.headers
for prefix, label in (("5h", "5-hour"), ("7d", "7-day"),
                      ("7d-sonnet", "7-day sonnet"), ("7d-opus", "7-day opus")):
    status = h.get(f"anthropic-ratelimit-unified-{prefix}-status")
    util = h.get(f"anthropic-ratelimit-unified-{prefix}-utilization")
    reset = h.get(f"anthropic-ratelimit-unified-{prefix}-reset")
    if status or util or reset:
        when = ""
        if reset:
            try:
                dt = datetime.fromtimestamp(int(reset), tz=timezone.utc)
                hrs = (dt - datetime.now(timezone.utc)).total_seconds() / 3600
                when = f" resets {dt.isoformat()} ({hrs:.1f}h)"
            except Exception:
                when = f" reset={reset}"
        print(f"{label:14} status={status} used={util}%{when}")
