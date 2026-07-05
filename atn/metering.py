"""Deterministic metering service — dollar cost accounting + subscription
quota inference.

This is a SERVICE (a daemon-owned module), not an agent. It answers two
questions the per-agent budget rail (``runtime/agent_registry.py``) cannot:

1. **Real dollar spend per provider.** For metered API providers (anthropic,
   openai, gemini, deepseek, …) it cross-references a maintained
   cost-per-token table (keyed by canonical model id) against the ACTUAL
   parsed token counts the runtime records per turn. It enforces a
   daemon-wide HARD dollar ceiling per provider, checked pre-flight exactly
   like a budget.

2. **Hidden subscription quota.** For subscription providers (claude_max,
   codex_max) the absolute quota is never disclosed by the vendor — only a
   rolling ``utilization`` fraction lands in ``provider._rate_limits``. This
   service persists ``(timestamp, utilization, cumulative_tokens)`` snapshots
   and fits ``quota ≈ Δtokens / Δutilization`` with a rolling linear fit +
   confidence, exposing the current estimate and the remaining allocation.

Persistence: two JSONL files under ``{data_dir}/metering/``:
  * ``spend.jsonl``        — append-only per-turn dollar-cost events.
  * ``quota_snapshots.jsonl`` — append-only subscription snapshots.
Both are derived/rebuildable ledgers; nothing here is gossiped or anchored.

The static cost table is intentionally a maintained constant with a dated
source comment. Vendor pricing APIs are not uniformly machine-readable, so a
static table with model-id keys is the honest V1. Update the table and bump
``_PRICING_AS_OF`` when prices change.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cost-per-token table
# ---------------------------------------------------------------------------
# USD per 1,000,000 tokens (per-MTok), split by lane. Cache-read is the
# discounted re-send of a cached prefix (~10% of input on Anthropic); cache
# write is a one-time premium (~125% of input on Anthropic). Where a vendor
# does not distinguish, cache lanes fall back to the input rate.
#
# Keys are canonical model ids matched by longest-prefix (see model_specs.py),
# so "claude-opus-4-8-20260101" resolves to the "claude-opus-4-8" entry.
#
# _PRICING_AS_OF documents when these figures were last reconciled against the
# vendors' public pricing pages. BUMP THIS when you touch the table.
#
# Source: vendor public pricing pages (anthropic.com/pricing,
# openai.com/api/pricing, ai.google.dev/pricing, deepseek published rates),
# reconciled 2026-07-05. Subscription models (claude_max) have no per-token
# dollar price — their cost is a flat monthly fee, metered by quota instead.
_PRICING_AS_OF = "2026-07-05"


@dataclass(frozen=True)
class ModelPricing:
    """Per-MTok USD prices for one model. cache_read / cache_write default to
    the input rate when a vendor doesn't break them out."""
    input: float
    output: float
    cache_read: float | None = None
    cache_write: float | None = None

    def read_rate(self) -> float:
        return self.cache_read if self.cache_read is not None else self.input

    def write_rate(self) -> float:
        return self.cache_write if self.cache_write is not None else self.input


# per-MTok USD. See _PRICING_AS_OF above.
_PRICING: dict[str, ModelPricing] = {
    # ---- Anthropic / Claude (direct API; NOT the claude_max subscription) ----
    "claude-fable-5":    ModelPricing(input=10.0, output=50.0, cache_read=1.0,  cache_write=12.5),
    "claude-mythos-5":   ModelPricing(input=10.0, output=50.0, cache_read=1.0,  cache_write=12.5),
    "claude-opus-4-8":   ModelPricing(input=5.0,  output=25.0, cache_read=0.5,  cache_write=6.25),
    "claude-opus-4-7":   ModelPricing(input=5.0,  output=25.0, cache_read=0.5,  cache_write=6.25),
    "claude-opus-4-6":   ModelPricing(input=5.0,  output=25.0, cache_read=0.5,  cache_write=6.25),
    "claude-sonnet-4-6": ModelPricing(input=3.0,  output=15.0, cache_read=0.3,  cache_write=3.75),
    "claude-sonnet-4":   ModelPricing(input=3.0,  output=15.0, cache_read=0.3,  cache_write=3.75),
    "claude-haiku-4-5":  ModelPricing(input=1.0,  output=5.0,  cache_read=0.1,  cache_write=1.25),
    "claude-haiku-4":    ModelPricing(input=0.8,  output=4.0,  cache_read=0.08, cache_write=1.0),
    # ---- OpenAI ----
    "gpt-5.5":  ModelPricing(input=2.5,  output=10.0, cache_read=0.25),
    "gpt-5":    ModelPricing(input=2.5,  output=10.0, cache_read=0.25),
    "gpt-4.1":  ModelPricing(input=2.0,  output=8.0,  cache_read=0.5),
    "gpt-4o":   ModelPricing(input=2.5,  output=10.0, cache_read=1.25),
    "o3":       ModelPricing(input=10.0, output=40.0, cache_read=2.5),
    "o4-mini":  ModelPricing(input=1.1,  output=4.4,  cache_read=0.275),
    "o1":       ModelPricing(input=15.0, output=60.0, cache_read=7.5),
    # ---- DeepSeek ----
    "deepseek-reasoner": ModelPricing(input=0.55, output=2.19, cache_read=0.14),
    "deepseek-chat":     ModelPricing(input=0.27, output=1.10, cache_read=0.07),
    "deepseek-v4-pro":   ModelPricing(input=0.55, output=2.19, cache_read=0.14),
    "deepseek-v4-flash": ModelPricing(input=0.14, output=0.28, cache_read=0.035),
    # ---- Google Gemini ----
    "gemini-3-pro":     ModelPricing(input=1.25, output=10.0, cache_read=0.3125),
    "gemini-3-flash":   ModelPricing(input=0.30, output=2.5,  cache_read=0.075),
    "gemini-2.5-pro":   ModelPricing(input=1.25, output=10.0, cache_read=0.3125),
    "gemini-2.5-flash": ModelPricing(input=0.30, output=2.5,  cache_read=0.075),
}

# Providers billed by subscription (flat fee) — NO per-token dollar price.
# These are metered by inferred quota, never a dollar cap.
SUBSCRIPTION_PROVIDERS = frozenset({"claude_max", "codex_max"})

_PERIOD_SECONDS: dict[str, int] = {
    "hourly": 3_600,
    "daily": 86_400,
    "weekly": 604_800,
    "monthly": 2_592_000,  # 30-day approximation, matches agent_registry
}

# Cap the retained snapshot series per provider — the rolling fit needs only
# the current window, and old resets are dropped anyway. This bounds memory and
# replay time while leaving the append-only file as the full audit trail.
_MAX_SNAPSHOTS = 500


def _price_for_model(model_id: str) -> ModelPricing | None:
    """Longest-prefix lookup into the pricing table. Returns None for a model
    with no published per-token price (e.g. a local/ollama model, or a
    subscription-only model)."""
    if not model_id:
        return None
    exact = _PRICING.get(model_id)
    if exact is not None:
        return exact
    best_key = ""
    for key in _PRICING:
        if model_id.startswith(key) and len(key) > len(best_key):
            best_key = key
    return _PRICING.get(best_key) if best_key else None


def cost_usd(
    model_id: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float | None:
    """Deterministic dollar cost of a turn given its parsed token counts.

    Returns None when the model has no published per-token price (subscription
    or local model) — the caller then knows to fall back to quota metering
    rather than treating $0 as "free".
    """
    pricing = _price_for_model(model_id)
    if pricing is None:
        return None
    return (
        input_tokens * pricing.input
        + output_tokens * pricing.output
        + cache_read_tokens * pricing.read_rate()
        + cache_write_tokens * pricing.write_rate()
    ) / 1_000_000.0


# ---------------------------------------------------------------------------
# Quota inference (subscription providers)
# ---------------------------------------------------------------------------

@dataclass
class QuotaEstimate:
    """Result of the rolling fit over (utilization, cumulative_tokens) snapshots.

    ``quota_tokens`` is the estimated absolute number of tokens that fill 100%
    of the subscription window. ``confidence`` is 0..1: 0 = no usable fit yet,
    rising with the number of consistent observations and the utilization span
    they cover. ``remaining_tokens`` is quota × (1 − current_utilization)."""
    provider: str
    quota_tokens: float | None = None
    confidence: float = 0.0
    remaining_tokens: float | None = None
    current_utilization: float | None = None
    observations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "quota_tokens": self.quota_tokens,
            "confidence": round(self.confidence, 4),
            "remaining_tokens": self.remaining_tokens,
            "current_utilization": self.current_utilization,
            "observations": self.observations,
        }


def infer_quota(
    snapshots: list[dict[str, Any]],
    *,
    provider: str = "",
) -> QuotaEstimate:
    """Fit ``cumulative_tokens ≈ quota × utilization`` over the snapshot series.

    Pure function of the persisted snapshots (closed-form testable). Each
    snapshot is ``{utilization: 0..1, cumulative_tokens: int, ts: iso}``.

    Method — a rolling least-squares slope through the origin, restricted to
    the CURRENT window (we drop everything before the last utilization RESET,
    i.e. a point where utilization dropped, since the 5h/7d window rolled over
    and cumulative_tokens is measured from a different zero on each side).

    slope = Σ(u·t) / Σ(u²)  is the tokens-per-unit-utilization, and since
    utilization runs 0..1, slope IS the full-window quota. Robust to noise
    because it uses every point, not just the endpoints.

    confidence blends (a) the number of points and (b) the utilization span
    covered — a fit over u∈[0.02, 0.04] is far less trustworthy than one over
    u∈[0.05, 0.80]. Both must be present to earn confidence.
    """
    est = QuotaEstimate(provider=provider)
    if not snapshots:
        return est

    # Restrict to the current window: walk back from the end, stop at the last
    # point where utilization dropped vs its predecessor (a window reset).
    series = [
        s for s in snapshots
        if isinstance(s.get("utilization"), (int, float))
        and isinstance(s.get("cumulative_tokens"), (int, float))
    ]
    if not series:
        return est
    cut = 0
    for i in range(1, len(series)):
        if series[i]["utilization"] < series[i - 1]["utilization"] - 1e-9:
            cut = i  # window rolled over here; current window starts at i
    window = series[cut:]
    est.current_utilization = float(window[-1]["utilization"])

    # Rebase cumulative_tokens to the window's start so the through-origin fit
    # is measured from this window's zero.
    base_tokens = float(window[0]["cumulative_tokens"])
    base_util = float(window[0]["utilization"])
    pts: list[tuple[float, float]] = []
    for s in window:
        u = float(s["utilization"]) - base_util
        t = float(s["cumulative_tokens"]) - base_tokens
        if u > 1e-9 and t >= 0:
            pts.append((u, t))
    est.observations = len(pts)
    if len(pts) < 1:
        return est

    sum_uu = sum(u * u for u, _ in pts)
    sum_ut = sum(u * t for u, t in pts)
    if sum_uu <= 0:
        return est
    slope = sum_ut / sum_uu  # tokens per 1.0 (=100%) utilization
    if slope <= 0:
        return est
    est.quota_tokens = slope
    est.remaining_tokens = max(0.0, slope * (1.0 - est.current_utilization))

    # Confidence: saturating in observation count, scaled by utilization span.
    span = max(u for u, _ in pts)
    n = len(pts)
    count_conf = n / (n + 3.0)          # 0.25 @1obs, 0.57 @4, 0.77 @10, →1
    span_conf = min(1.0, span / 0.10)   # full credit once ≥10% of window spanned
    est.confidence = round(count_conf * span_conf, 4)
    return est


# ---------------------------------------------------------------------------
# Operational analysis (admin-agent facing)
# ---------------------------------------------------------------------------
# Everything below is a PURE function of the persisted spend events + snapshot
# series. The admin agent's ``metering_report`` tool is a thin wrapper that
# reads the ledgers and calls these; there is NO new math in the tool layer.


def _parse_ts(ts: Any) -> datetime | None:
    if not isinstance(ts, str):
        return None
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _bucket_key(dt: datetime, bucket: str) -> str:
    """ISO bucket label for a timestamp. 'hour' → YYYY-MM-DDTHH, 'day' → date."""
    if bucket == "day":
        return dt.date().isoformat()
    return dt.strftime("%Y-%m-%dT%H")


def analyze_spend_events(
    events: list[dict[str, Any]],
    *,
    now: datetime | None = None,
    window_hours: float = 24.0,
    bucket: str = "hour",
) -> dict[str, Any]:
    """Deterministic operational rollup over persisted spend events.

    Pure function of the event list (each event is one ``spend.jsonl`` line:
    ``{ts, provider, model, usd, input_tokens, output_tokens,
    cache_read_tokens, cache_write_tokens, agent_id?}``). Returns:

      * ``per_provider``: for each provider, a time-bucketed series of
        ``{bucket, usd, input_tokens, output_tokens, cache_read_tokens,
        cache_write_tokens, cache_hit_ratio, cost_per_ktok}`` plus totals.
        ``cache_hit_ratio`` = cache_read / (input + cache_read) — the fraction
        of prompt tokens served from cache. A DROP in this ratio across
        buckets is the "prompt caching appears broken" signal. ``cost_per_ktok``
        = usd / (all tokens / 1000) — a RISE is the same anomaly seen from the
        cost side.
      * ``per_agent``: burn ranking — ``[{agent_id, usd, tokens}]`` descending
        by usd, over the window. Agents with no attributed events are absent.
      * ``window``: the (start, end) actually covered and event count.

    Only events within ``window_hours`` of ``now`` are included, so the report
    reflects the recent operational window, not all history.
    """
    now = now or datetime.now(timezone.utc)
    horizon = now.timestamp() - window_hours * 3600.0
    bucket = "day" if bucket == "day" else "hour"

    # provider -> bucket_key -> accumulator
    prov_buckets: dict[str, dict[str, dict[str, float]]] = {}
    prov_totals: dict[str, dict[str, float]] = {}
    agent_burn: dict[str, dict[str, float]] = {}
    kept = 0
    first_ts: datetime | None = None
    last_ts: datetime | None = None

    for ev in events:
        dt = _parse_ts(ev.get("ts"))
        if dt is None or dt.timestamp() < horizon:
            continue
        usd = ev.get("usd")
        if not isinstance(usd, (int, float)):
            continue
        prov = ev.get("provider") or "unknown"
        kept += 1
        first_ts = dt if first_ts is None or dt < first_ts else first_ts
        last_ts = dt if last_ts is None or dt > last_ts else last_ts

        it = float(ev.get("input_tokens", 0) or 0)
        ot = float(ev.get("output_tokens", 0) or 0)
        cr = float(ev.get("cache_read_tokens", 0) or 0)
        cw = float(ev.get("cache_write_tokens", 0) or 0)

        bkey = _bucket_key(dt, bucket)
        b = prov_buckets.setdefault(prov, {}).setdefault(
            bkey, {"usd": 0.0, "input_tokens": 0.0, "output_tokens": 0.0,
                   "cache_read_tokens": 0.0, "cache_write_tokens": 0.0})
        b["usd"] += float(usd)
        b["input_tokens"] += it
        b["output_tokens"] += ot
        b["cache_read_tokens"] += cr
        b["cache_write_tokens"] += cw

        t = prov_totals.setdefault(
            prov, {"usd": 0.0, "input_tokens": 0.0, "output_tokens": 0.0,
                   "cache_read_tokens": 0.0, "cache_write_tokens": 0.0})
        t["usd"] += float(usd)
        t["input_tokens"] += it
        t["output_tokens"] += ot
        t["cache_read_tokens"] += cr
        t["cache_write_tokens"] += cw

        aid = ev.get("agent_id")
        if aid:
            ab = agent_burn.setdefault(aid, {"usd": 0.0, "tokens": 0.0})
            ab["usd"] += float(usd)
            ab["tokens"] += it + ot + cr + cw

    def _cache_hit_ratio(inp: float, cread: float) -> float | None:
        denom = inp + cread
        return round(cread / denom, 4) if denom > 0 else None

    def _cost_per_ktok(usd: float, inp: float, out: float, cread: float,
                       cwrite: float) -> float | None:
        toks = inp + out + cread + cwrite
        return round(usd / (toks / 1000.0), 6) if toks > 0 else None

    per_provider: dict[str, Any] = {}
    for prov, buckets in prov_buckets.items():
        series = []
        for bkey in sorted(buckets):
            acc = buckets[bkey]
            series.append({
                "bucket": bkey,
                "usd": round(acc["usd"], 6),
                "input_tokens": int(acc["input_tokens"]),
                "output_tokens": int(acc["output_tokens"]),
                "cache_read_tokens": int(acc["cache_read_tokens"]),
                "cache_write_tokens": int(acc["cache_write_tokens"]),
                "cache_hit_ratio": _cache_hit_ratio(
                    acc["input_tokens"], acc["cache_read_tokens"]),
                "cost_per_ktok": _cost_per_ktok(
                    acc["usd"], acc["input_tokens"], acc["output_tokens"],
                    acc["cache_read_tokens"], acc["cache_write_tokens"]),
            })
        tot = prov_totals[prov]
        per_provider[prov] = {
            "series": series,
            "total_usd": round(tot["usd"], 6),
            "cache_hit_ratio": _cache_hit_ratio(
                tot["input_tokens"], tot["cache_read_tokens"]),
            "cost_per_ktok": _cost_per_ktok(
                tot["usd"], tot["input_tokens"], tot["output_tokens"],
                tot["cache_read_tokens"], tot["cache_write_tokens"]),
        }

    per_agent = sorted(
        ({"agent_id": a, "usd": round(v["usd"], 6), "tokens": int(v["tokens"])}
         for a, v in agent_burn.items()),
        key=lambda r: r["usd"], reverse=True,
    )

    return {
        "per_provider": per_provider,
        "per_agent": per_agent,
        "window": {
            "hours": window_hours,
            "bucket": bucket,
            "events": kept,
            "start": first_ts.isoformat() if first_ts else None,
            "end": last_ts.isoformat() if last_ts else None,
        },
    }


# ---------------------------------------------------------------------------
# The service
# ---------------------------------------------------------------------------

class MeteringService:
    """Daemon-wide dollar accounting + subscription quota inference.

    Thread-safe (a lock guards the in-memory ledgers and the append writes);
    the runtime records usage from the execution engine's per-turn recorder,
    which may run on the worker-host thread.
    """

    def __init__(self, data_dir: Path, config: Any = None) -> None:
        self._dir = Path(data_dir) / "metering"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._spend_path = self._dir / "spend.jsonl"
        self._snap_path = self._dir / "quota_snapshots.jsonl"
        self._config = config
        self._lock = threading.RLock()

        # provider -> cumulative USD spent in the current period.
        self._spend: dict[str, float] = {}
        # provider -> iso timestamp of the current period start (for rollover).
        self._period_start: dict[str, str] = {}
        # provider -> list of {ts, utilization, cumulative_tokens} snapshots.
        self._snapshots: dict[str, list[dict[str, Any]]] = {}
        self._load()

    # ------------------------------------------------------------------
    # Config access
    # ------------------------------------------------------------------

    def _provider_cfg(self, provider: str) -> Any:
        providers = getattr(self._config, "providers", None)
        if isinstance(providers, dict):
            return providers.get(provider)
        return None

    def dollar_limit(self, provider: str) -> tuple[float, str]:
        """(limit, period) for a provider's daemon-wide dollar cap. (0.0, 'none')
        when no cap is configured."""
        cfg = self._provider_cfg(provider)
        if cfg is None:
            return 0.0, "none"
        limit = float(getattr(cfg, "dollar_limit", 0.0) or 0.0)
        period = str(getattr(cfg, "dollar_period", "none") or "none").lower()
        if period not in _PERIOD_SECONDS and period != "none":
            period = "none"
        return limit, period

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Rebuild in-memory ledgers by replaying the JSONL files.

        spend.jsonl is replayed to recover the current-period cumulative
        spend; quota_snapshots.jsonl is replayed to recover the snapshot
        series per provider (capped to the most recent _MAX_SNAPSHOTS each)."""
        if self._spend_path.exists():
            try:
                for line in self._spend_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    ev = json.loads(line)
                    prov = ev.get("provider")
                    usd = ev.get("usd")
                    ts = ev.get("ts")
                    if not prov or not isinstance(usd, (int, float)):
                        continue
                    # Period rollover on replay: reset the running total when a
                    # newer event crosses the period boundary.
                    self._apply_spend_event(prov, float(usd), ts)
            except Exception:
                log.warning("metering: failed to replay spend.jsonl", exc_info=True)
        if self._snap_path.exists():
            try:
                for line in self._snap_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    s = json.loads(line)
                    prov = s.get("provider")
                    if not prov:
                        continue
                    self._snapshots.setdefault(prov, []).append({
                        "ts": s.get("ts"),
                        "utilization": s.get("utilization"),
                        "cumulative_tokens": s.get("cumulative_tokens"),
                    })
                for prov, lst in self._snapshots.items():
                    if len(lst) > _MAX_SNAPSHOTS:
                        self._snapshots[prov] = lst[-_MAX_SNAPSHOTS:]
            except Exception:
                log.warning("metering: failed to replay quota_snapshots.jsonl", exc_info=True)

    def _append(self, path: Path, obj: dict[str, Any]) -> None:
        """Append one JSON line atomically-enough (a single write() call)."""
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(obj) + "\n")
        except Exception:
            log.warning("metering: failed to append to %s", path.name, exc_info=True)

    # ------------------------------------------------------------------
    # Period rollover
    # ------------------------------------------------------------------

    def _maybe_reset_period(self, provider: str, now: datetime) -> None:
        _limit, period = self.dollar_limit(provider)
        if period == "none" or period not in _PERIOD_SECONDS:
            return
        start_iso = self._period_start.get(provider)
        if not start_iso:
            self._period_start[provider] = now.isoformat()
            return
        try:
            start_dt = datetime.fromisoformat(start_iso)
        except ValueError:
            self._period_start[provider] = now.isoformat()
            return
        if (now - start_dt).total_seconds() >= _PERIOD_SECONDS[period]:
            self._spend[provider] = 0.0
            self._period_start[provider] = now.isoformat()

    def _apply_spend_event(self, provider: str, usd: float, ts: str | None) -> None:
        """Fold one spend event into the running period total (replay + live)."""
        now = datetime.now(timezone.utc)
        if ts:
            try:
                now = datetime.fromisoformat(ts)
            except ValueError:
                pass
        self._maybe_reset_period(provider, now)
        self._spend[provider] = self._spend.get(provider, 0.0) + usd
        self._period_start.setdefault(provider, now.isoformat())

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_usage(
        self,
        provider: str,
        model_id: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        utilization: float | None = None,
        cumulative_tokens: int | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        """Record one turn's usage.

        For metered providers: compute the dollar cost from the table × the
        supplied token counts, fold it into the provider's running spend, and
        append a spend event. Returns ``{usd, provider_spend, over_limit}``.

        For subscription providers: if ``utilization`` and ``cumulative_tokens``
        are supplied, append a quota snapshot so the estimator can refine.

        Both may fire for one call — a subscription turn still has parsed token
        counts, and passing utilization lets the quota fit advance.
        """
        with self._lock:
            now = datetime.now(timezone.utc)
            ts = now.isoformat()
            result: dict[str, Any] = {"provider": provider, "usd": None}

            usd = cost_usd(
                model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
            )
            if usd is not None and usd > 0:
                self._apply_spend_event(provider, usd, ts)
                event: dict[str, Any] = {
                    "ts": ts, "provider": provider, "model": model_id,
                    "usd": usd,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cache_read_tokens": cache_read_tokens,
                    "cache_write_tokens": cache_write_tokens,
                }
                if agent_id:
                    event["agent_id"] = agent_id
                self._append(self._spend_path, event)
                result["usd"] = usd
                result["provider_spend"] = self._spend.get(provider, 0.0)
                limit, _period = self.dollar_limit(provider)
                result["over_limit"] = bool(limit > 0 and self._spend[provider] >= limit)

            if utilization is not None and cumulative_tokens is not None:
                self.record_snapshot(provider, float(utilization),
                                     int(cumulative_tokens), _ts=ts, _locked=True)
            return result

    def record_snapshot(
        self,
        provider: str,
        utilization: float,
        cumulative_tokens: int,
        *,
        _ts: str | None = None,
        _locked: bool = False,
    ) -> None:
        """Persist a (utilization, cumulative_tokens) snapshot for a
        subscription provider. Skipped if the last snapshot is identical (no
        new information) to keep the fit from over-weighting idle polls."""
        def _do() -> None:
            ts = _ts or datetime.now(timezone.utc).isoformat()
            lst = self._snapshots.setdefault(provider, [])
            if lst:
                last = lst[-1]
                if (last.get("utilization") == utilization
                        and last.get("cumulative_tokens") == cumulative_tokens):
                    return
            snap = {"ts": ts, "utilization": utilization,
                    "cumulative_tokens": cumulative_tokens}
            lst.append(snap)
            if len(lst) > _MAX_SNAPSHOTS:
                del lst[:-_MAX_SNAPSHOTS]
            self._append(self._snap_path, {"provider": provider, **snap})

        if _locked:
            _do()
        else:
            with self._lock:
                _do()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def check_provider_spend(self, provider: str) -> tuple[bool, str | None]:
        """Pre-flight dollar check. Returns (ok, reason). ``ok=True`` when the
        provider has no cap, isn't a metered provider, or is still under its
        cap. Mirrors ``AgentRegistry.check_budget``'s contract so the engine
        can gate on it the same way."""
        with self._lock:
            limit, period = self.dollar_limit(provider)
            if limit <= 0:
                return True, None
            self._maybe_reset_period(provider, datetime.now(timezone.utc))
            spent = self._spend.get(provider, 0.0)
            if spent >= limit:
                return False, (
                    f"Daemon-wide dollar cap reached for '{provider}': "
                    f"${spent:.2f} of ${limit:.2f}"
                    + (f" ({period})" if period != "none" else "")
                )
            return True, None

    def provider_spend(self, provider: str) -> dict[str, Any]:
        """Current dollar posture for a provider: limit, spent, remaining."""
        with self._lock:
            limit, period = self.dollar_limit(provider)
            spent = self._spend.get(provider, 0.0)
            return {
                "provider": provider,
                "limit_usd": limit,
                "spent_usd": round(spent, 6),
                "remaining_usd": (round(max(0.0, limit - spent), 6)
                                  if limit > 0 else None),
                "period": period,
                "period_started_at": self._period_start.get(provider),
                "pricing_as_of": _PRICING_AS_OF,
            }

    def estimate_quota(self, provider: str) -> QuotaEstimate:
        """Current inferred subscription quota + remaining allocation."""
        with self._lock:
            snaps = list(self._snapshots.get(provider, []))
        return infer_quota(snaps, provider=provider)

    def _read_spend_events(self) -> list[dict[str, Any]]:
        """Read all persisted spend events from spend.jsonl (append-only audit
        trail — the full history, not the capped in-memory running total)."""
        out: list[dict[str, Any]] = []
        if not self._spend_path.exists():
            return out
        try:
            for line in self._spend_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
        except Exception:
            log.warning("metering: failed to read spend.jsonl for report",
                        exc_info=True)
        return out

    def report(
        self,
        *,
        window_hours: float = 24.0,
        bucket: str = "hour",
    ) -> dict[str, Any]:
        """Operational report for the admin agent — the deterministic view the
        ``metering_report`` tool surfaces.

        Combines three deterministic sources:
          * ``providers``: the time-bucketed cost/cache/burn analysis over the
            persisted spend events (``analyze_spend_events``), keyed by provider,
            plus the current dollar posture (limit/spent/remaining) for each.
          * ``agents``: per-agent burn ranking over the window.
          * ``subscriptions``: for each subscription provider with snapshots, the
            inferred quota + remaining + confidence (``estimate_quota``).

        Pure over the persisted ledgers — no live provider calls. Bounded by
        ``window_hours`` so it reflects the recent operational window.
        """
        with self._lock:
            events = self._read_spend_events()
            snap_providers = list(self._snapshots.keys())
        analysis = analyze_spend_events(
            events, window_hours=window_hours, bucket=bucket)

        # Fold in dollar posture per provider so a leak is legible against its
        # cap in the same object.
        providers: dict[str, Any] = {}
        seen = set(analysis["per_provider"].keys())
        for prov in seen:
            entry = dict(analysis["per_provider"][prov])
            entry["posture"] = self.provider_spend(prov)
            providers[prov] = entry
        # Also include capped providers with no spend in the window, so a
        # provider that fell silent is still visible.
        for posture in self.all_provider_spend():
            prov = posture["provider"]
            if prov not in providers:
                providers[prov] = {
                    "series": [], "total_usd": 0.0,
                    "cache_hit_ratio": None, "cost_per_ktok": None,
                    "posture": posture,
                }

        subscriptions: dict[str, Any] = {}
        for prov in snap_providers:
            subscriptions[prov] = self.estimate_quota(prov).to_dict()

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "pricing_as_of": _PRICING_AS_OF,
            "window": analysis["window"],
            "providers": providers,
            "agents": analysis["per_agent"],
            "subscriptions": subscriptions,
        }

    def all_provider_spend(self) -> list[dict[str, Any]]:
        """Dollar posture for every provider that has a configured cap or any
        recorded spend."""
        with self._lock:
            names = set(self._spend.keys())
            providers = getattr(self._config, "providers", None)
            if isinstance(providers, dict):
                names |= {n for n, c in providers.items()
                          if float(getattr(c, "dollar_limit", 0.0) or 0.0) > 0}
        return [self.provider_spend(p) for p in sorted(names)]
