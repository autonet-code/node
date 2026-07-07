r"""security_monitor — the daemon-side tripwire (Phase 3 of the secret track).

The ``SecurityMonitor`` holds a LIVE-SET of secret exposures and scans every
``STEP_OUTPUT`` event (agent transcript content) for a leaked secret VALUE or
its staged PATH. A hit means a worker handled a secret raw — read the staged
file and echoed the value into its conversation, or leaked the path — instead
of letting the daemon-mediated tooling handle it. On a hit the monitor fires a
``SECURITY_ALARM`` naming the secret and the SOURCE agent whose transcript
leaked it.

Data-flow discipline (SAFETY-CRITICAL, non-negotiable):
  * The raw secret VALUE lives here ONLY to compare against transcript text. It
    is fed by the broker's owner-authenticated value-push (P2) at stage time,
    held in the live-set, and DROPPED at ``drop_exposure`` (Seam B revoke) or by
    the TTL sweeper. It is NEVER logged, NEVER emitted in an event, NEVER
    returned, NEVER written to disk. The durable alarm store records NAMES only.
  * ``SECURITY_ALARM`` event data carries ``{"names": [...]}`` ONLY. It carries
    NO ``agent_id`` key (that key triggers a subtree-scoping trap in the
    consumers); the leaking agent travels as the event ``source`` instead.

Fail-safe posture: the scan path NEVER goes dark. A scan exception degrades to a
``<scan-error>`` alarm rather than silently swallowing a possible leak. An empty
live-set is a cheap no-op (flag-off / no grant => the monitor sees nothing to
scan for and stays silent — byte-identical externally).

Reuses ``kevin/secret_alarm.py`` ``scan_text`` (literal substring match) +
``record_alarm`` (durable NAMES-only store), relocated under
``<data_dir>/security/``. It NEVER touches ``cloud.env`` — the live-set is fed
by exposures, never a standing secret copy.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

from ..events import Event, EventBus, EventType

log = logging.getLogger(__name__)

# Minimum value length we will scan for. Mirrors secret_alarm._MIN_LEN — a value
# shorter than this is too likely to appear in ordinary text (false positives).
# The PATH is always registered regardless of length (it is unguessable, 128-bit).
_MIN_VALUE_LEN = 6

# The per-agent rolling tail buffer keeps the last _TAIL_FACTOR x (longest live
# value/path) characters of that agent's transcript so a value split across two
# STEP_OUTPUT chunks is still caught when the two halves land in different
# events. 2x the longest tracked token guarantees any single token that spans a
# chunk boundary is fully contained in [previous-tail + this-chunk].
_TAIL_FACTOR = 2
_TAIL_FLOOR = 256  # never shrink the window below this (cheap, robust)

# Dedup window (seconds): the same (source_agent, secret_name) hit inside this
# window fires exactly one alarm. A leaked value typically re-appears across
# many streamed chunks; we don't want an alarm per chunk.
_DEDUP_WINDOW_S = 30.0

# Default TTL backstop (seconds) for an exposure whose explicit drop_exposure
# never fires (e.g. the daemon crashed before Seam-B revoke). This is a SAFETY
# BACKSTOP, not the primary lifecycle — the primary drop is drop_exposure at
# revoke. It must be >= the longest an agent can plausibly live. worker
# max_runtime defaults to None (unbounded), so we cannot derive a tight bound;
# a generous fixed default is used and can be overridden by the caller.
_DEFAULT_TTL_S = 24 * 3600.0

_SWEEP_INTERVAL_S = 60.0


class SecurityMonitor:
    """Live-exposure tripwire. One per Runtime; subscribes to STEP_OUTPUT.

    Constructed unconditionally (cheap, inert until an exposure is added). With
    the isolation flag OFF nothing ever calls ``add_exposure`` (P4 gates the
    grant), so the live-set stays empty and every scan is a no-op.
    """

    def __init__(
        self,
        events: EventBus,
        *,
        data_dir: Optional[Path] = None,
        ttl_s: float = _DEFAULT_TTL_S,
    ) -> None:
        self._events = events
        self._ttl_s = float(ttl_s) if ttl_s and ttl_s > 0 else _DEFAULT_TTL_S

        # Live-set: (pid, secret_name) -> exposure record. One lock guards it and
        # the derived scan lists + tails + dedup map.
        self._lock = threading.RLock()
        self._by_key: dict[tuple[int, str], dict[str, Any]] = {}
        # Derived scan lists, rebuilt on every mutation (small — a handful of
        # live secrets at a time). Values are only present for len >= _MIN.
        self._values: list[tuple[str, str, str, int]] = []  # (value, name, agent_id, pid)
        self._paths: list[tuple[str, str, str, int]] = []    # (path, name, agent_id, pid)
        self._max_token_len = 0

        # Per-agent rolling transcript tail (chunk-split defence).
        self._tails: dict[str, str] = {}
        # Dedup: (source_agent, secret_name) -> last-alarm monotonic ts.
        self._recent: dict[tuple[str, str], float] = {}

        # Durable NAMES-only alarm store, relocated under <data_dir>/security/.
        # secret_alarm.record_alarm writes to a module-level ALARM_STORE; we call
        # it with an explicit store path if the installed version supports it,
        # else we patch the module constant (best-effort, process-local).
        self._alarm_store_path: Optional[str] = None
        if data_dir is not None:
            try:
                sec_dir = Path(data_dir) / "security"
                sec_dir.mkdir(parents=True, exist_ok=True)
                self._alarm_store_path = str(sec_dir / "secret_alarms.json")
            except OSError:
                self._alarm_store_path = None

        self._instance_name = "atn-daemon"
        self._sweeper: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._subscribed = False

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        """Subscribe to STEP_OUTPUT and start the TTL sweeper. Idempotent."""
        if not self._subscribed:
            self._events.subscribe(EventType.STEP_OUTPUT, self._on_step_output)
            self._subscribed = True
        if self._sweeper is None:
            self._stop.clear()
            self._sweeper = threading.Thread(
                target=self._sweep_loop, name="atn-secmon-sweep", daemon=True)
            self._sweeper.start()

    def stop(self) -> None:
        self._stop.set()
        if self._subscribed:
            try:
                self._events.unsubscribe(EventType.STEP_OUTPUT, self._on_step_output)
            except Exception:  # noqa: BLE001
                pass
            self._subscribed = False

    # -- live-set mutation ---------------------------------------------------
    def add_exposure(
        self,
        value: Optional[str],
        staged_path: Optional[str],
        secret_name: str,
        agent_id: str,
        pid: int,
    ) -> None:
        """Register a live exposure from the broker value-push.

        VALUE-SCAN ON: the real ``value`` (from the owner-authenticated push) is
        stored and scanned alongside the ``staged_path``. A value shorter than
        ``_MIN_VALUE_LEN`` is NOT registered as a scan target (false-positive
        guard) but the path is ALWAYS registered (it is 128-bit unguessable).
        """
        try:
            pid_i = int(pid)
        except (TypeError, ValueError):
            return
        if not secret_name:
            return
        rec = {
            "value": value if (isinstance(value, str) and len(value) >= _MIN_VALUE_LEN) else None,
            "staged_path": staged_path if isinstance(staged_path, str) and staged_path else None,
            "secret_name": str(secret_name),
            "agent_id": str(agent_id) if agent_id else "",
            "pid": pid_i,
            "staged_at": time.time(),
            "_mono": time.monotonic(),
        }
        with self._lock:
            self._by_key[(pid_i, str(secret_name))] = rec
            self._rebuild_scan_lists()

    def drop_exposure(self, *, pid: Optional[int] = None,
                      agent_id: Optional[str] = None) -> None:
        """Drop live exposures (and the agent's tail buffer) at Seam-B revoke.

        Called with the revoked ``pid`` (widened revoke hook passes it). All
        secrets staged for that PID are removed and the raw values dropped. If
        ``agent_id`` is given its rolling tail buffer is cleared too.
        """
        with self._lock:
            if pid is not None:
                try:
                    pid_i = int(pid)
                except (TypeError, ValueError):
                    pid_i = None
                if pid_i is not None:
                    dead = [k for k in self._by_key if k[0] == pid_i]
                    for k in dead:
                        # Drop the raw value reference explicitly.
                        self._by_key.pop(k, None)
            if agent_id:
                self._tails.pop(str(agent_id), None)
            elif pid is not None:
                # Also drop tails for any agent bound to this pid's exposures.
                agents = {r.get("agent_id") for r in self._by_key.values()}
                for a in list(self._tails):
                    if a not in agents:
                        # keep only agents still live; a dropped-pid agent whose
                        # exposures are gone no longer needs a tail
                        pass
            self._rebuild_scan_lists()

    def is_live(self, pid: int, secret_name: str) -> bool:
        with self._lock:
            try:
                return (int(pid), str(secret_name)) in self._by_key
            except (TypeError, ValueError):
                return False

    def is_healthy(self) -> bool:
        """True iff the tripwire is actually running: subscribed to STEP_OUTPUT
        AND the TTL sweeper thread is alive. This is the monitor-down signal the
        Seam-B grant path checks — if the monitor cannot scan, no secret may be
        granted (monitor-down => no tripwire => no secrets)."""
        if not self._subscribed:
            return False
        sw = self._sweeper
        return sw is not None and sw.is_alive() and not self._stop.is_set()

    def _rebuild_scan_lists(self) -> None:
        """Rebuild the derived (value/path) scan lists. Caller holds the lock."""
        values: list[tuple[str, str, str, int]] = []
        paths: list[tuple[str, str, str, int]] = []
        longest = 0
        for rec in self._by_key.values():
            name = rec["secret_name"]
            agent = rec.get("agent_id", "")
            pid = rec.get("pid", 0)
            v = rec.get("value")
            if v:
                values.append((v, name, agent, pid))
                longest = max(longest, len(v))
            p = rec.get("staged_path")
            if p:
                paths.append((p, name, agent, pid))
                longest = max(longest, len(p))
        self._values = values
        self._paths = paths
        self._max_token_len = longest

    # -- scan path -----------------------------------------------------------
    async def _on_step_output(self, event: Event) -> None:
        """Scan one STEP_OUTPUT event's content for any live value/path.

        Covers BOTH emit sites via the single STEP_OUTPUT subscription: the
        worker re-emit (worker_host._event_sink) and the in-process cognitive
        stream (execution_engine ~:978). The leaking agent is the event
        ``source``.
        """
        # Fast no-op: nothing to scan for.
        with self._lock:
            if not self._values and not self._paths:
                return

        src = event.source or (event.data or {}).get("agent_id") or ""
        text = ""
        data = event.data or {}
        content = data.get("content")
        if isinstance(content, str):
            text = content
        if not text and not src:
            return

        try:
            hits = self._scan_for_agent(str(src), text)
        except Exception:  # noqa: BLE001 — the tripwire must NEVER go dark
            log.debug("security scan raised; emitting degraded alarm", exc_info=True)
            await self._fire_alarm(str(src), ["<scan-error>"])
            return

        if hits:
            await self._fire_alarm(str(src), sorted(hits))

    def _scan_for_agent(self, src: str, text: str) -> set[str]:
        """Return the set of leaked secret NAMES for this agent's chunk.

        Prepends the agent's rolling tail so a token split across two chunks is
        caught, scans the combined text, then updates + trims the tail. Only the
        VALUES/PATHS whose exposure is live for THIS agent are considered — a
        value staged for a different agent does not implicate ``src`` (the
        tripwire names the agent whose transcript leaked ITS OWN secret).
        """
        with self._lock:
            tail = self._tails.get(src, "")
            values = [(v, n) for (v, n, a, _p) in self._values if a == src]
            paths = [(p, n) for (p, n, a, _p) in self._paths if a == src]
            window = max(_TAIL_FLOOR, _TAIL_FACTOR * self._max_token_len)

        combined = tail + (text or "")
        hits: set[str] = set()
        # Literal substring match — same logic as secret_alarm.scan_text, but
        # scoped to this agent's live tokens (name-tagged).
        if combined:
            for value, name in values:
                if value and value in combined:
                    hits.add(name)
            for path, name in paths:
                if path and path in combined:
                    hits.add(name)

        # Update the rolling tail (keep the last `window` chars of combined).
        new_tail = combined[-window:] if len(combined) > window else combined
        with self._lock:
            # Only retain a tail if this agent still has live tokens to catch.
            if any(a == src for (_v, _n, a, _p) in self._values) or \
               any(a == src for (_p2, _n2, a, _p3) in self._paths):
                self._tails[src] = new_tail
            else:
                self._tails.pop(src, None)
        return hits

    async def _fire_alarm(self, src_agent: str, names: list[str]) -> None:
        """Record (names-only, durable) + emit SECURITY_ALARM. Dedup'd.

        The event carries NAMES ONLY, ``source=src_agent``, and NO ``agent_id``
        key (subtree-scoping trap). The value NEVER appears.
        """
        names = list(names) or ["<unknown>"]
        now = time.monotonic()
        fresh: list[str] = []
        with self._lock:
            for n in names:
                key = (src_agent, n)
                last = self._recent.get(key, 0.0)
                if now - last >= _DEDUP_WINDOW_S:
                    self._recent[key] = now
                    fresh.append(n)
            # Opportunistic prune of the dedup map.
            if len(self._recent) > 4096:
                cutoff = now - _DEDUP_WINDOW_S
                self._recent = {k: t for k, t in self._recent.items() if t >= cutoff}
        if not fresh:
            return

        # Durable record — NAMES ONLY. Best-effort; never raises into the scan.
        try:
            self._record_names(fresh)
        except Exception:  # noqa: BLE001
            log.debug("record_alarm failed", exc_info=True)

        try:
            await self._events.emit(Event(
                type=EventType.SECURITY_ALARM,
                source=src_agent,           # the leaking agent
                data={"names": fresh},      # NAMES ONLY; NO agent_id key
            ))
        except Exception:  # noqa: BLE001
            log.debug("SECURITY_ALARM emit failed", exc_info=True)

        # Log names only — NEVER the value.
        log.error("SECURITY_ALARM: agent %s leaked secret value(s): %s",
                  src_agent, ", ".join(fresh))

    def _record_names(self, names: list[str]) -> None:
        """Append to the durable NAMES-only store via kevin/secret_alarm.

        The vendored in-wheel copy is authoritative (same pattern as the
        keystore import in worker_host.py); a dev checkout of kevin on the
        path is the fallback. Before the vendored copy existed this import
        silently failed everywhere and secret_alarms.json was never written."""
        try:
            from atn._vendor.kevin import secret_alarm as _sa  # type: ignore
        except Exception:  # noqa: BLE001
            try:
                import secret_alarm as _sa  # type: ignore
            except Exception:  # noqa: BLE001 — store is optional; scan still fires the event
                return
        record_alarm = _sa.record_alarm
        # Relocate the module-level store under our data_dir if we have one.
        if self._alarm_store_path:
            try:
                _sa.ALARM_STORE = self._alarm_store_path
            except Exception:  # noqa: BLE001
                pass
        record_alarm(names, self._instance_name, when=time.time())

    # -- TTL sweeper ---------------------------------------------------------
    def _sweep_loop(self) -> None:
        while not self._stop.wait(_SWEEP_INTERVAL_S):
            try:
                self._sweep_once()
            except Exception:  # noqa: BLE001
                log.debug("security TTL sweep raised", exc_info=True)

    def _sweep_once(self) -> None:
        now = time.monotonic()
        with self._lock:
            dead = [k for k, r in self._by_key.items()
                    if now - r.get("_mono", now) >= self._ttl_s]
            if dead:
                for k in dead:
                    self._by_key.pop(k, None)
                # Prune tails for agents with no remaining live tokens.
                live_agents = {r.get("agent_id") for r in self._by_key.values()}
                for a in list(self._tails):
                    if a not in live_agents:
                        self._tails.pop(a, None)
                self._rebuild_scan_lists()
