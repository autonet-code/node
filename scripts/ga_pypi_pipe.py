"""Daily pipe: PyPI download count -> GA4 custom event.

Fetches the previous day's download count for the autonet-computer package
from pypistats.org and posts it to GA4 as event `pypi_install` (one event
per category, with_mirrors and without_mirrors). Lets you chart daily
installs in the same GA dashboard as the autonet.computer web analytics.

Intended to run daily via GitHub Actions cron. Can also be invoked with
--backfill to seed historical data from pypistats' full series.

Required env vars (CI sets these from GitHub secrets):
    GA_MEASUREMENT_ID   GA4 stream measurement id, e.g. G-XXXXXXXXXX
    GA_API_SECRET       Measurement Protocol API secret for that stream

Optional:
    PYPI_PACKAGE        default: autonet-computer
"""
import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import date, timedelta

MEASUREMENT_ID = os.environ.get("GA_MEASUREMENT_ID", "G-GVN1WMFZJ5")
PACKAGE = os.environ.get("PYPI_PACKAGE", "autonet-computer")

# Stable pseudo-client id so all daily pipes land under one logical "user"
# in GA. Not associated with any real visitor.
CLIENT_ID = "pypi-pipe.0001"


def load_secret() -> str:
    secret = os.environ.get("GA_API_SECRET")
    if not secret:
        print("GA_API_SECRET env var is required.", file=sys.stderr)
        sys.exit(2)
    return secret.strip()


def fetch_overall() -> list[dict]:
    url = f"https://pypistats.org/api/packages/{PACKAGE}/overall"
    req = urllib.request.Request(url, headers={"User-Agent": "autonet-ga-pipe/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload.get("data", [])


def counts_for(rows: list[dict], target_date: str) -> tuple[int, int]:
    wo = 0
    w = 0
    for r in rows:
        if r.get("date") != target_date:
            continue
        if r.get("category") == "without_mirrors":
            wo = int(r.get("downloads", 0))
        elif r.get("category") == "with_mirrors":
            w = int(r.get("downloads", 0))
    return wo, w


def post_event(secret: str, category: str, count: int, event_date: str) -> None:
    endpoint = (
        f"https://www.google-analytics.com/mp/collect"
        f"?measurement_id={MEASUREMENT_ID}&api_secret={secret}"
    )
    body = {
        "client_id": CLIENT_ID,
        "events": [
            {
                "name": "pypi_install",
                "params": {
                    "count": count,
                    "category": category,
                    "event_date": event_date,
                    "package": PACKAGE,
                },
            }
        ],
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            if resp.status != 204:
                print(f"  unexpected status {resp.status} for {category}",
                      file=sys.stderr)
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}",
              file=sys.stderr)
        raise


def send_one_day(secret: str, rows: list[dict], target_date: str) -> bool:
    """Send both category events for one date. Returns True if anything went out."""
    wo, w = counts_for(rows, target_date)
    if wo == 0 and w == 0:
        return False
    print(f"  {target_date}: without_mirrors={wo}, with_mirrors={w}")
    post_event(secret, "without_mirrors", wo, target_date)
    post_event(secret, "with_mirrors", w, target_date)
    return True


def backfill(secret: str, rows: list[dict]) -> int:
    """Send every distinct date present in the pypistats series."""
    dates = sorted({r["date"] for r in rows if r.get("date")})
    sent = 0
    for d in dates:
        if send_one_day(secret, rows, d):
            sent += 1
            # Be polite to MP; ~5 events/sec is well under the limit.
            time.sleep(0.2)
    return sent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true",
                    help="Send every date in the pypistats series, not just yesterday.")
    args = ap.parse_args()

    secret = load_secret()
    print(f"Fetching pypistats for {PACKAGE} ...")
    rows = fetch_overall()

    if args.backfill:
        n = backfill(secret, rows)
        print(f"Backfilled {n} day(s).")
        return 0

    yesterday = (date.today() - timedelta(days=1)).isoformat()
    if not send_one_day(secret, rows, yesterday):
        print(f"  no rows for {yesterday} yet (pypistats lags ~1d). Exit clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
