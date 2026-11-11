#!/usr/bin/env python3
"""S0 fixes check: four recent fixes proven on the S0 local network.

Reuses scripts/s0_local_run.py's setup (hardhat, the packaged jurisdiction
from trustless-contracts, a local relay seeded in the Registry, four
household daemons onboarded the way an owner would) and then runs one
targeted check per fix, with a short volume loop in the background so
epochs keep closing:

  #46 adoption waits for a code holder: h1 publishes a fresh tool, h2
      proposes it, h1 stops, h2's owner approves (expect approved_pending),
      h1 comes back, the adoption completes in the background.
  #45 services on credits: h3 sells a tool-backed request service, h4 finds
      it (spec and price list present), opens credits with h3 and requests
      it twice (one channel, two prices used), then requests it while h3 is
      stopped (not delivered, nothing more used).
  #41 batch catch-up: h3 stays down about a minute mid-epoch, comes back,
      and its next closes must match the other daemons' closes (same epoch
      id, root and per-sender batch set) after catching up missed batches.
  #44 relay keeps anchored state: every anchor's payload, carry-over and
      world checkpoint blobs are in the relay's store (daemons offer them).
  #42 seed refresh: the only relay is killed and restarted (daemons must
      reconnect on their own), then a second relay is started and added to
      the Registry bootstrap key (daemons must dial it without a restart).

The daemons run with p2p.seed_refresh_interval 15 s and the periodic
bootstrap redial off (bootstrap_redial_interval 0), so reconnects come only
from the code under test. The Registry key is edited by impersonating its
owner (the timelock) on hardhat: the fix under test is the daemon re-read,
not governance.

Writes ``<run>/fixes_report.json`` and prints PASS/FAIL per fix.

Run:
    python scripts/s0_fixes_check.py

Env: S0_DIR (default D:/autonet_s0), TRUSTLESS_CONTRACTS_DIR, as for S0.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s0_local_run as s0  # noqa: E402

log = s0.log
RELAY2_PORT = s0.RELAY_PORT + 1
SEED_REFRESH_SECS = 15
SERVICE_ASK = 1_000_000           # 1 ATN (6 decimals) per request
EPOCH_SECS = 120                  # payments aimed at floor-length epochs


# ---------------------------------------------------------------------------
# Processes and daemons
# ---------------------------------------------------------------------------

def kill(proc: Optional[subprocess.Popen]) -> None:
    """Stop one process this script started (by its own PID only)."""
    if proc is None or proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    else:
        proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass


class Household(s0.Household):
    """An S0 household whose daemon can be stopped and started again, with
    the p2p settings under test."""

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.boots = 0
        self.registry = ""

    def autonet_yaml(self) -> Path:
        return self.home / "autonet.yaml"

    def write_config(self) -> None:
        super().write_config()
        import yaml
        self.autonet_yaml().write_text(yaml.safe_dump({"p2p": {
            "seed_refresh_interval": SEED_REFRESH_SECS,
            "bootstrap_redial_interval": 0,
        }}), encoding="utf-8")

    def env(self, registry: str) -> Dict[str, str]:
        env = super().env(registry)
        env["AUTONET_CONFIG"] = str(self.autonet_yaml())
        return env

    def log_path(self) -> Path:
        return self.home / ("daemon.log" if self.boots <= 1 else f"daemon.{self.boots}.log")

    def logs(self) -> List[Path]:
        return sorted(self.home.glob("daemon*.log"))

    def start(self, procs: s0.Procs, registry: str) -> None:
        self.registry = registry
        self.boots += 1
        if self.boots == 1:
            # A restart keeps the config the daemon has written since.
            self.write_config()
        self.proc = procs.start(f"daemon-{self.name}", [sys.executable, "-m", "atn", "--headless"],
                                cwd=self.home, env=self.env(registry),
                                log_path=self.log_path(), stdin_pipe=True)

    async def stop(self) -> None:
        """Shut the daemon down the way a terminal close does (stdin EOF),
        falling back to a kill of its own PID."""
        if self.client is not None and self.client._ws is not None:
            try:
                await self.client._ws.close()
            except Exception:
                pass
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            deadline = time.time() + 45
            while time.time() < deadline and self.proc.poll() is None:
                await asyncio.sleep(1)
            kill(self.proc)

    async def restart(self, procs: s0.Procs) -> None:
        self.start(procs, self.registry)
        self.client = s0.DaemonClient(self.client_url(), self.name)
        await self.client.connect()


def log_lines(h: Household, since: float = 0.0) -> List[str]:
    out = []
    for p in h.logs():
        for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
            ts = line_ts(ln)
            if ts is None or ts >= since:
                out.append(ln)
    return out


def line_ts(ln: str) -> Optional[float]:
    m = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d+)", ln)
    if not m:
        return None
    return time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")) + int(m.group(2)) / 1000


async def peers(h: Household) -> Optional[int]:
    try:
        r = await h.client.call({"type": "network_status"}, timeout=30)
        return ((r.get("result") or {}).get("p2p") or {}).get("peers")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Relays and the Registry
# ---------------------------------------------------------------------------

def relay_args(port: int, data_dir: Path) -> List[str]:
    return [sys.executable, "-m", "nodes.relay", "--port", str(port), "--host", "127.0.0.1",
            "--transport", "ws", "--no-upnp", "--data-dir", str(data_dir)]


def relay_peer(log_path: Path, timeout: float = 90) -> str:
    addr = s0.wait_for(lambda: next((ln.split("relay address: ", 1)[1].strip()
                                     for ln in log_path.read_text(
                                         encoding="utf-8", errors="replace").splitlines()
                                     if "relay address: " in ln), None), timeout, "relay address")
    return addr.rsplit("/p2p/", 1)[1]


async def relay_peer_async(log_path: Path, timeout: float = 90) -> str:
    """relay_peer without blocking the event loop (daemon sockets live on it)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            return relay_peer(log_path, timeout=0.5)
        except Exception:
            await asyncio.sleep(1)
    raise RuntimeError(f"timed out waiting for relay address in {log_path}")


def set_bootstrap(w3, registry: str, value: str) -> None:
    """Edit jurisdiction.autonet.bootstrap as the Registry's owner (the DAO
    timelock), impersonated on hardhat."""
    art = s0.artifact("Registry.sol/Registry.json")
    reg = w3.eth.contract(address=w3.to_checksum_address(registry), abi=art["abi"])
    owner = reg.functions.owner().call()
    w3.provider.make_request("hardhat_impersonateAccount", [owner])
    w3.provider.make_request("hardhat_setBalance", [owner, hex(10**18)])
    h = reg.functions.editRegistry("jurisdiction.autonet.bootstrap", value).transact(
        {"from": owner, "gas": 500_000})
    rc = w3.eth.wait_for_transaction_receipt(h, timeout=120)
    w3.provider.make_request("hardhat_stopImpersonatingAccount", [owner])
    if rc.status != 1:
        raise RuntimeError("editRegistry reverted")
    got = reg.functions.getRegistryValue("jurisdiction.autonet.bootstrap").call()
    if got != value:
        raise RuntimeError(f"bootstrap key reads {got!r}")


# ---------------------------------------------------------------------------
# Background volume, so epochs close during the checks
# ---------------------------------------------------------------------------

async def volume_loop(act: s0.Activity, stop: asyncio.Event) -> None:
    import random
    rng = random.Random(11)
    while not stop.is_set():
        t0 = time.time()
        live = [h for h in act.hs if h.alive() and h.client and h.client._ws is not None]
        try:
            threshold = act.epoch_threshold()
        except Exception:  # noqa: BLE001
            threshold = 0
        per_tick = threshold / s0.BURN_SHARE * s0.TICK_SECS / EPOCH_SECS if threshold else 0
        per_agent = int(per_tick / max(1, len(live)))
        try:
            await asyncio.gather(*(act.work(h, rng) for h in live))
            if per_agent > 0:
                await asyncio.gather(*(act.pay(h, per_agent, rng) for h in live))
        except Exception as exc:  # noqa: BLE001
            log(f"volume tick: {exc!r}")
        try:
            await asyncio.wait_for(stop.wait(), timeout=max(0.0, s0.TICK_SECS - (time.time() - t0)))
        except asyncio.TimeoutError:
            pass


# ---------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------

class Report:
    def __init__(self, run: Path) -> None:
        self.run = run
        self.fixes: Dict[str, Dict[str, Any]] = {}

    def set(self, fix: str, ok: bool, **evidence) -> None:
        self.fixes[fix] = {"pass": bool(ok), **evidence}
        log(f"{fix}: {'PASS' if ok else 'FAIL'} {json.dumps(evidence, default=str)[:1500]}")
        (self.run / "fixes_report.json").write_text(
            json.dumps(self.fixes, indent=1, default=str), encoding="utf-8")


async def proposal_status(h: Household, digest: str) -> Dict[str, Any]:
    r = await h.client.ok({"type": "list_adoption_proposals"})
    return next((p for p in r.get("proposals") or [] if p.get("digest") == digest), {})


async def check_adoption(hs: List[Household], procs: s0.Procs, rep: Report) -> None:
    """#46: approve while the only code holder is down; complete once it is up."""
    author, adopter = hs[0], hs[1]
    ev: Dict[str, Any] = {}
    name = "fix46_rot13"
    body = "import codecs\nprint(json.dumps({'text': codecs.encode(args['text'], 'rot13')}))"
    r = await author.client.ok({"type": "register_tool", "caller_id": author.agent_id, "name": name,
                                "description": "ROT13 a text.", "input_schema": s0.TEXT_SCHEMA,
                                "code": s0._prog(body),
                                "novelty": {"statement": "ROT13 a text. Nothing comparable "
                                            "is in the library yet.", "prior_art": []}})
    digest = r.get("digest") or (r.get("tool") or {}).get("digest")
    await author.client.ok({"type": "publish_tool", "caller_id": author.agent_id, "digest": digest})
    ev["digest"] = digest
    prop = await adopter.client.ok({"type": "adopt_tool", "caller_id": adopter.agent_id,
                                    "digest": digest, "reason": "s0 fixes check"})
    ev["proposed"] = prop.get("status")
    ev["adopter_peers_before"] = await peers(adopter)
    await author.stop()
    ev["author_stopped_at"] = time.time()
    log(f"#46: {author.name} stopped; {adopter.name} owner approves {digest[:16]}")
    t0 = time.time()
    appr = await adopter.client.call({"type": "approve_adoption", "digest": digest}, timeout=240)
    ev["approve_secs"] = round(time.time() - t0, 1)
    res = appr.get("result") or {}
    ev["approve_reply"] = {"ok": appr.get("ok"), "status": res.get("status"),
                           "error": appr.get("error"), "note": res.get("note")}
    st = await proposal_status(adopter, digest)
    ev["status_while_down"] = st.get("status")
    ev["retry_while_down"] = st.get("retry")
    pending_ok = res.get("status") == "approved_pending" and st.get("status") == "approved_pending"
    await asyncio.sleep(20)
    await author.restart(procs)
    ev["author_restarted_at"] = time.time()
    log(f"#46: {author.name} back up; waiting for {adopter.name}'s background install")
    done: Dict[str, Any] = {}
    deadline = time.time() + 420
    while time.time() < deadline:
        st = await proposal_status(adopter, digest)
        if st.get("status") != "approved_pending":
            done = st
            break
        await asyncio.sleep(10)
    ev["status_after"] = done.get("status") or "still approved_pending"
    ev["resolved_after_restart_secs"] = (round(time.time() - ev["author_restarted_at"], 1)
                                         if done else None)
    tools = await adopter.client.ok({"type": "list_tools", "caller_id": adopter.agent_id})
    local = next((t for t in tools.get("tools") or [] if t.get("digest") == digest), None)
    ev["installed_on_adopter"] = bool(local)
    if local:
        use = await adopter.client.call({"type": "use_tool", "caller_id": adopter.agent_id,
                                         "name": local.get("name"), "arguments": {"text": "abc"}})
        ev["use_after_install"] = (use.get("result") or {}).get("result", use.get("error"))
    rep.set("#46 adoption", pending_ok and done.get("status") == "approved" and bool(local), **ev)


async def check_services(act: s0.Activity, hs: List[Household], procs: s0.Procs,
                         rep: Report, j: Dict[str, Any]) -> Household:
    """#45: register, find (spec), request twice on one channel, then once to
    a stopped provider."""
    provider, buyer = hs[2], hs[3]
    ev: Dict[str, Any] = {}
    backing = next(d for d, m in act.published.items()
                   if m["author"] == provider.name and m["name"] == "slugify_text")
    reg = await provider.client.ok({"type": "register_service", "agent_id": provider.agent_id,
                                    "mode": "request", "name": "slugify",
                                    "description": "Turn a text into a url slug.",
                                    "backing_tool": backing, "price": SERVICE_ASK,
                                    "voucher": SERVICE_ASK})
    spec_digest = reg.get("digest")
    ev["register"] = {"digest": spec_digest, "on_chain": reg.get("on_chain"),
                      "service_id": reg.get("service_id"), "reason": reg.get("reason")}
    await asyncio.sleep(5)
    found = await buyer.client.ok({"type": "find_services", "caller_id": buyer.agent_id,
                                   "limit": 20})
    row = next((s for s in found.get("services") or [] if s.get("spec_digest") == spec_digest), None)
    ev["find_row"] = row
    spec_ok = bool(row and row.get("spec_available") and row.get("input_schema")
                   and row.get("name") == "slugify")

    from atn.config import RPBConfig
    from atn.on_chain import ServiceMarketClient
    mc = ServiceMarketClient(RPBConfig(rpc_url=s0.RPC_URL, chain_id=s0.CHAIN_ID,
                                       substrate_address=j["autonet"]["substrate"],
                                       payment_channel_address=j["autonet"]["paymentChannel"],
                                       service_registry_address=j["autonet"]["serviceRegistry"]))
    book_path = buyer.home / ".atn" / "service_channels.json"

    def book() -> Dict[str, Any]:
        try:
            data = json.loads(book_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return next(iter((data.get("open") or {}).values()), {})

    opened = await buyer.client.call({"type": "open_credits", "caller_id": buyer.agent_id,
                                      "provider_address": provider.agent_address,
                                      "deposit": 10 * SERVICE_ASK}, timeout=240)
    ev["open_credits"] = opened
    replies = []
    for text in ("Hello Channel World", "second request on the same channel"):
        r = await buyer.client.call({"type": "request_service", "caller_id": buyer.agent_id,
                                     "provider_address": provider.agent_address,
                                     "spec_digest": spec_digest, "payload": {"text": text}},
                                    timeout=240)
        res = r.get("result") or {}
        replies.append({"ok": r.get("ok"), "error": r.get("error") or res.get("error"),
                        "result": res.get("result"), "credits": res.get("credits"),
                        "transport": res.get("transport"),
                        "delivered": res.get("delivered"), "book": book()})
    ev["requests"] = replies
    cid = (replies[0].get("credits") or {}).get("channel_id")
    chain = await mc.get_channel(int(cid)) if cid is not None else None
    ev["channel_on_chain"] = chain
    voucher_ok = (all(x["ok"] and not x["error"] for x in replies)
                  and replies[1]["book"].get("consumed") == 2 * SERVICE_ASK
                  and replies[1]["book"].get("cumulative") == 2 * SERVICE_ASK
                  and chain is not None and int(chain["deposit"]) == 10 * SERVICE_ASK
                  and chain["provider"].lower() == provider.agent_address.lower())
    # The provider side: the vouchers it holds for this channel.
    pv = provider.home / ".atn" / "services"
    ev["provider_store_files"] = sorted(p.name for p in pv.glob("*")) if pv.exists() else []

    await provider.stop()
    log(f"#45: provider {provider.name} stopped; requesting again")
    before = book().get("consumed")
    r = await buyer.client.call({"type": "request_service", "caller_id": buyer.agent_id,
                                 "provider_address": provider.agent_address,
                                 "spec_digest": spec_digest, "payload": {"text": "nobody home"}},
                                timeout=300)
    res = r.get("result") or {}
    after = book().get("consumed")
    ev["stopped_provider_request"] = {"ok": r.get("ok"), "error": r.get("error") or res.get("error"),
                                      "delivered": res.get("delivered"),
                                      "used_before": before, "used_after": after}
    # Nothing more used; the voucher signed for it is at most one paid ahead.
    stopped_ok = before == after == 2 * SERVICE_ASK and not res.get("delivered")
    rep.set("#45 service channel", spec_ok and voucher_ok and stopped_ok,
            spec_ok=spec_ok, voucher_ok=voucher_ok, stopped_ok=stopped_ok, **ev)
    return provider


CLOSE_RE = re.compile(r"federated close inputs: (epoch=(\S+) .*)$")
CAUGHT_RE = re.compile(r"caught up (\d+) batch\(es\) for epoch (\d+)")


def closes(h: Household, since: float = 0.0) -> Dict[str, str]:
    out = {}
    for ln in log_lines(h, since):
        m = CLOSE_RE.search(ln)
        if m:
            out[m.group(2)] = m.group(1)
    return out


async def check_catchup(h3: Household, hs: List[Household], procs: s0.Procs, w3, j,
                        rep: Report, down_since: float) -> None:
    """#41: h3 was stopped mid-epoch; after it comes back its closes must
    match everyone else's."""
    sub = w3.eth.contract(address=w3.to_checksum_address(j["autonet"]["substrate"]),
                          abi=s0.abi("Substrate"))
    await asyncio.sleep(max(0.0, 60 - (time.time() - down_since)))
    await h3.restart(procs)
    back = time.time()
    n0 = int(sub.functions.anchorCount().call())
    log(f"#41: {h3.name} back after {back - down_since:.0f}s down; anchors {n0}")
    deadline = time.time() + 900
    while time.time() < deadline:
        mine = closes(h3, back)
        others = [closes(o) for o in hs if o is not h3]
        common = [e for e in mine if all(e in c for c in others)]
        # Two closes everyone made, both anchored on chain.
        n = int(sub.functions.anchorCount().call())
        ids = {sub.functions.getAnchor(i).call()[0] for i in range(n)}
        if len(common) >= 2 and all(e in ids for e in common):
            break
        await asyncio.sleep(15)
    mine = closes(h3, back)
    others = {o.name: closes(o) for o in hs if o is not h3}
    rows = []
    for e, line in mine.items():
        cmp = {n: (c.get(e) == line if e in c else None) for n, c in others.items()}
        rows.append({"epoch": e, "h3": line, "match": cmp})
    caught = [ln for ln in log_lines(h3, back) if CAUGHT_RE.search(ln)]
    rewinds = [ln for ln in log_lines(h3, back) if "rewinding" in ln]
    n = int(sub.functions.anchorCount().call())
    anchored = [sub.functions.getAnchor(i).call()[0] for i in range(n)]
    compared = [r for r in rows if all(v is not None for v in r["match"].values())]
    ok = (bool(caught) and bool(compared)
          and all(all(r["match"].values()) for r in compared) and not rewinds
          and all(r["epoch"] in anchored for r in compared))
    rep.set("#41 batch catch-up", ok, down_secs=round(back - down_since),
            caught_up_lines=caught[:5], closes_after_restart=rows, rewinds=rewinds[:3],
            anchored_epoch_ids=anchored)


def check_relay_state(w3, j, relay_dir: Path, rep: Report) -> None:
    """#44: the relay holds every anchored epoch's state (payload, and the
    carry-over and world checkpoint it names), offered by the daemons."""
    sub = w3.eth.contract(address=w3.to_checksum_address(j["autonet"]["substrate"]),
                          abi=s0.abi("Substrate"))
    blobs = relay_dir / "blobs"
    rows = []
    for i in range(int(sub.functions.anchorCount().call())):
        a = sub.functions.getAnchor(i).call()
        ph = bytes(a[5]).hex()
        row = {"epoch": a[0], "payload": (blobs / ph).exists()}
        if row["payload"]:
            payload = json.loads((blobs / ph).read_bytes())
            for k in ("carry_cid",):
                if payload.get(k):
                    row[k] = (blobs / payload[k]).exists()
        rows.append(row)
    ok = bool(rows) and all(all(v for k, v in r.items() if k != "epoch") for r in rows)
    rep.set("#44 relay keeps anchored state", ok, anchors=rows)


async def check_seeds(hs: List[Household], procs: s0.Procs, w3, j, run: Path,
                      relay1: subprocess.Popen, relay1_args: List[str], relay1_peer: str,
                      rep: Report) -> None:
    """#42: kill/restart the only relay; then add a second relay to the
    Registry mid-run."""
    ev: Dict[str, Any] = {}
    ev["peers_before"] = {h.name: await peers(h) for h in hs}
    kill(relay1)
    killed = time.time()
    log("#42: relay killed")
    await asyncio.sleep(25)
    ev["peers_relay_down"] = {h.name: await peers(h) for h in hs}
    procs.start("seed-relay-a2", relay1_args, cwd=s0.REPO, env=s0.base_env(),
                          log_path=run / "relay.2.log")
    restarted = time.time()
    same_peer = await relay_peer_async(run / "relay.2.log") == relay1_peer
    deadline = time.time() + 240
    reconnect: Dict[str, Optional[float]] = {h.name: None for h in hs}
    while time.time() < deadline and any(v is None for v in reconnect.values()):
        for h in hs:
            if reconnect[h.name] is None:
                for ln in log_lines(h, restarted):
                    if f"Connected to {relay1_peer[:16]}" in ln:
                        reconnect[h.name] = round(line_ts(ln) - restarted, 1)
                        break
        await asyncio.sleep(5)
    ev["relay_restart_same_peer_id"] = same_peer
    ev["relay_down_secs"] = round(restarted - killed)
    ev["reconnect_secs_after_restart"] = reconnect
    ev["peers_after_restart"] = {h.name: await peers(h) for h in hs}
    reconnect_ok = all(v is not None for v in reconnect.values())

    # A second relay, added to the Registry while the daemons run.
    relay2_dir = run / "relay2"
    relay2_dir.mkdir(exist_ok=True)
    jkey = f"{s0.CHAIN_ID}:{j['dao']['governor'].lower()}"
    args2 = relay_args(RELAY2_PORT, relay2_dir) + ["--jurisdiction", jkey]
    procs.start("seed-relay-b", args2, cwd=s0.REPO, env=s0.base_env(), log_path=run / "relay2.log")
    peer2 = await relay_peer_async(run / "relay2.log")
    seeds = f"{s0.seed_addr(relay1_peer)},/dns4/localhost/tcp/{RELAY2_PORT}/ws/p2p/{peer2}"
    set_bootstrap(w3, j["dao"]["registry"], seeds)
    edited = time.time()
    log(f"#42: Registry bootstrap now lists relay2 {peer2[:16]}")
    dialled: Dict[str, Any] = {h.name: None for h in hs}
    changed: Dict[str, Any] = {h.name: None for h in hs}
    deadline = time.time() + 180
    while time.time() < deadline and any(v is None for v in dialled.values()):
        for h in hs:
            for ln in log_lines(h, edited):
                if changed[h.name] is None and "Seed list changed" in ln and peer2 in ln:
                    changed[h.name] = round(line_ts(ln) - edited, 1)
                if dialled[h.name] is None and f"Connected to {peer2[:16]}" in ln:
                    dialled[h.name] = round(line_ts(ln) - edited, 1)
        await asyncio.sleep(5)
    ev["relay2_peer"] = peer2
    ev["seed_list_changed_secs_after_edit"] = changed
    ev["dialled_relay2_secs_after_edit"] = dialled
    ev["peers_after_relay2"] = {h.name: await peers(h) for h in hs}
    add_ok = all(v is not None for v in dialled.values())
    rep.set("#42 seed refresh", reconnect_ok and add_ok, reconnect_ok=reconnect_ok,
            add_ok=add_ok, **ev)


# ---------------------------------------------------------------------------

async def amain() -> int:
    run = s0.S0_DIR / time.strftime("fixes-%Y%m%d-%H%M%S")
    run.mkdir(parents=True, exist_ok=True)
    log(f"run dir {run}")
    procs = s0.Procs()
    hs: List[Household] = []
    rep = Report(run)
    stop = asyncio.Event()
    bg: List[asyncio.Task] = []
    try:
        procs.start("hardhat", ["npx.cmd" if os.name == "nt" else "npx", "hardhat", "node",
                                "--port", str(s0.RPC_PORT)], cwd=s0.DAO_DIR, env=s0.base_env(),
                    log_path=run / "hardhat.log")
        w3, accts = s0.w3_and_accounts()
        s0.wait_for(lambda: w3.is_connected() and w3.eth.chain_id == s0.CHAIN_ID, 120, "hardhat")
        w3.provider.make_request("evm_setAutomine", [False])
        w3.provider.make_request("evm_setIntervalMining", [1000])
        deployer, owners = accts[0], accts[1:1 + s0.HOUSEHOLDS]
        usdc_addr = s0.deploy_test_usdc(w3, deployer)
        usdc = w3.eth.contract(address=usdc_addr,
                               abi=s0.artifact("test/TestToken6.sol/TestToken6.json")["abi"])
        for o in owners:
            s0.send(w3, deployer, usdc.functions.transfer(o.address, s0.USDC_PER_OWNER))

        relay_dir = run / "relay"
        relay_dir.mkdir()
        rargs = relay_args(s0.RELAY_PORT, relay_dir)
        probe = procs.start("relay-probe", rargs + ["--jurisdiction", "31337:0x0"], cwd=s0.REPO,
                            env=s0.base_env(), log_path=run / "relay_probe.log")
        peer = relay_peer(run / "relay_probe.log")
        kill(probe)
        j = s0.deploy_jurisdiction(run, usdc_addr, [o.address for o in owners], s0.seed_addr(peer))
        jkey = f"{s0.CHAIN_ID}:{j['dao']['governor'].lower()}"
        relay1_args = rargs + ["--jurisdiction", jkey]
        relay1 = procs.start("seed-relay-a", relay1_args, cwd=s0.REPO, env=s0.base_env(),
                             log_path=run / "relay.log")
        log(f"jurisdiction up; relay {peer[:16]}")

        for i, o in enumerate(owners):
            h = Household(i, run, o)
            h.start(procs, j["dao"]["registry"])
            hs.append(h)
        for h in hs:
            h.client = s0.DaemonClient(h.client_url(), h.name)
            await h.client.connect()
        for h in hs:
            await s0.onboard(h, j, w3, usdc)
        log("all households onboarded")

        # #46 first: before any other daemon could hold the probe tool's code.
        try:
            await check_adoption(hs, procs, rep)
        except Exception:
            rep.set("#46 adoption", False, crash=traceback.format_exc()[-1500:])

        act = s0.Activity(run, hs, j, w3)
        await act.author()
        await act.adopt()
        bg.append(asyncio.create_task(s0.collector(run, hs, j, w3, stop)))
        bg.append(asyncio.create_task(volume_loop(act, stop)))
        # Let a couple of epochs anchor with everyone up.
        sub = w3.eth.contract(address=w3.to_checksum_address(j["autonet"]["substrate"]),
                              abi=s0.abi("Substrate"))
        # Polled without blocking the loop (a blocking wait starves the
        # websocket keepalives and the daemons drop the connections).
        deadline = time.time() + 900
        while time.time() < deadline and int(sub.functions.anchorCount().call()) < 1:
            await asyncio.sleep(10)

        provider = None
        try:
            provider = await check_services(act, hs, procs, rep, j)
        except Exception:
            rep.set("#45 service channel", False, crash=traceback.format_exc()[-1500:])
        down_since = time.time()
        h3 = hs[2]
        if provider is None and h3.alive():
            await h3.stop()
        try:
            await check_catchup(h3, hs, procs, w3, j, rep, down_since)
        except Exception:
            rep.set("#41 batch catch-up", False, crash=traceback.format_exc()[-1500:])
        try:
            check_relay_state(w3, j, relay_dir, rep)
        except Exception:
            rep.set("#44 relay keeps anchored state", False, crash=traceback.format_exc()[-1500:])
        try:
            await check_seeds(hs, procs, w3, j, run, relay1, relay1_args, peer, rep)
        except Exception:
            rep.set("#42 seed refresh", False, crash=traceback.format_exc()[-1500:])

        stop.set()
        for t in bg:
            try:
                await asyncio.wait_for(t, timeout=30)
            except Exception:
                t.cancel()
        log("SUMMARY " + " | ".join(f"{k}: {'PASS' if v['pass'] else 'FAIL'}"
                                    for k, v in rep.fixes.items()))
        return 0 if rep.fixes and all(v["pass"] for v in rep.fixes.values()) else 1
    except Exception:
        log(traceback.format_exc())
        return 1
    finally:
        stop.set()
        for t in bg:
            t.cancel()
        for h in hs:
            if h.client and h.client._ws is not None:
                try:
                    await h.client._ws.close()
                except Exception:
                    pass
        procs.stop()


def main() -> int:
    try:
        return asyncio.run(amain())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
