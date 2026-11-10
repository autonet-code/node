#!/usr/bin/env python3
"""The P2P mesh behind real NATs, in Docker (docs/p2p_networking.md).

S1 style: one machine, Docker networks standing in for the internet.

    mesh-pub (172.31.0.0/24, "the internet")
      mesh-relay   atn relay (python -m nodes.relay): circuit relay v2 hop,
                   DHT, gossip relay; TCP 4002 + WebSocket 4001
      pub-p        a reachable daemon (public address)
      mesh-gw-a    NAT gateway for LAN A (iptables MASQUERADE, unsolicited
                   inbound dropped, like a home router)
      mesh-gw-b    NAT gateway for LAN B
      mesh-proxy   HTTP CONNECT proxy, the only way out of LAN P, allowing
                   only the relay's WebSocket port (scripts/s1/gateway.py)
    mesh-lan-a (internal): nat-a1, nat-a2   default route via mesh-gw-a
    mesh-lan-b (internal): nat-b1, nat-b2   default route via mesh-gw-b
    mesh-lan-p (internal): proxied-h        HTTPS_PROXY=mesh-proxy, no route

Every node runs scripts/mesh/mesh_node.py (the daemon's AutonetHost with a
blob store and the jurisdiction topic) from the repo mounted at /repo, so
code changes need no image rebuild. Seeds: the relay only.

Checks, printed as PASS/FAIL lines:
  (a) NAT'd daemons end up directly connected to each other, NAT'd to
      NAT'd across the two gateways by hole punching (DCUtR);
  (b) a tool published on one NAT'd daemon is found through the DHT
      providers and fetched by every other daemon;
  (d) the proxied HTTPS-only daemon gets gossip and tools and serves its
      own through the relay;
  (c) with the relay stopped, gossip still reaches every daemon (the
      proxied one excepted: the relay is its only way in) and new tools
      are still found and fetched;
  (e) a daemon restarted with its only seed down rejoins through its peer
      cache.

Run:
    docker build -f docker/Dockerfile.mesh -t autonet-mesh:dev .
    python scripts/mesh_nat_harness.py

Env: MESH_DIR (default D:/autonet_mesh) for logs, MESH_IMAGE, MESH_KEEP=1
to leave the containers up. The transcript is also written to
experiments/mesh_nat/harness_output.txt.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
IMAGE = os.environ.get("MESH_IMAGE", "autonet-mesh:dev")
RUN = Path(os.environ.get("MESH_DIR", "D:/autonet_mesh")) / time.strftime("run-%Y%m%d-%H%M%S")
TRANSCRIPT = REPO / "experiments" / "mesh_nat" / "harness_output.txt"
KEEP = os.environ.get("MESH_KEEP") == "1"
TOPIC_KEY = "31337:0xmesh"

PUB, LAN_A, LAN_B, LAN_P = "mesh-pub", "mesh-lan-a", "mesh-lan-b", "mesh-lan-p"
SUBNETS = {PUB: "172.31.0.0/24", LAN_A: "10.31.1.0/24", LAN_B: "10.31.2.0/24",
           LAN_P: "10.31.3.0/24"}
RELAY_IP, PUB_P_IP = "172.31.0.10", "172.31.0.20"
GW = {"mesh-gw-a": ("172.31.0.2", LAN_A, "10.31.1.2"),
      "mesh-gw-b": ("172.31.0.3", LAN_B, "10.31.2.2")}
PROXY = ("mesh-proxy", "172.31.0.4", "10.31.3.2")
# name: (network, ip, gateway ip or None, ws port, tcp port, kind)
NODES = {
    "nat-a1": (LAN_A, "10.31.1.11", "10.31.1.2", 4001, 4002, "nat"),
    "nat-a2": (LAN_A, "10.31.1.12", "10.31.1.2", 4011, 4012, "nat"),
    "nat-b1": (LAN_B, "10.31.2.11", "10.31.2.2", 4001, 4002, "nat"),
    "nat-b2": (LAN_B, "10.31.2.12", "10.31.2.2", 4011, 4012, "nat"),
    "pub-p": (PUB, PUB_P_IP, None, 4001, 4002, "public"),
    "proxied-h": (LAN_P, "10.31.3.11", None, 4001, 4002, "proxied"),
}
NAT_IP = {"nat-a1": "172.31.0.2", "nat-a2": "172.31.0.2",
          "nat-b1": "172.31.0.3", "nat-b2": "172.31.0.3"}

_lines: list = []
_results: list = []


def say(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    _lines.append(line)


def check(ok: bool, what: str) -> bool:
    _results.append((ok, what))
    say(("PASS " if ok else "FAIL ") + what)
    return ok


def docker(*args: str, check_: bool = True) -> str:
    r = subprocess.run(["docker", *args], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if check_ and r.returncode != 0:
        raise RuntimeError(f"docker {' '.join(args[:4])}: {r.stderr.strip()[-600:]}")
    return (r.stdout or "").strip()


def repo_mount() -> str:
    return str(REPO).replace("\\", "/") + ":/repo:ro"


def cleanup() -> None:
    names = [n for n in docker("ps", "-aq", "--filter", "name=^mesh-", check_=False).split()]
    names += [n for n in docker("ps", "-aq", "--filter", "name=^nat-", check_=False).split()]
    for name in ("pub-p", "proxied-h"):
        names += docker("ps", "-aq", "--filter", f"name=^{name}$", check_=False).split()
    if names:
        docker("rm", "-f", *names, check_=False)
    for net in (LAN_A, LAN_B, LAN_P, PUB):
        docker("network", "rm", net, check_=False)


def run_container(name: str, net: str, ip: str, cmd: str, env: dict | None = None,
                  caps: bool = False, sysctl: bool = False) -> None:
    args = ["run", "-d", "--name", name, "--hostname", name, "--network", net,
            "--ip", ip, "-v", repo_mount(), "-e", "PYTHONPATH=/repo"]
    if caps:
        args += ["--cap-add", "NET_ADMIN"]
    if sysctl:
        args += ["--sysctl", "net.ipv4.ip_forward=1"]
    for k, v in (env or {}).items():
        args += ["-e", f"{k}={v}"]
    docker(*args, IMAGE, "sh", "-c", cmd)


def ctl(node: str, cmd: str, args: dict | None = None, timeout: float = 150) -> dict:
    argv = ["docker", "exec", node, "python", "/repo/scripts/mesh/mesh_ctl.py", cmd]
    if args:
        argv.append(json.dumps(args))
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="replace")
        return json.loads(r.stdout or "{}")
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


def wait(cond, timeout: float, step: float = 5.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            if cond():
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(step)
    return False


def start_network() -> str:
    """Networks, gateways, proxy, relay. Returns the relay's peer id."""
    cleanup()
    docker("network", "create", "--subnet", SUBNETS[PUB], PUB)
    # The NAT'd LANs are plain bridges: an --internal network drops every
    # packet whose destination is outside its subnet on the host bridge
    # (br_netfilter), even one addressed to our gateway container. Nodes
    # route through their gateway, and Docker's isolation between
    # networks keeps unsolicited inbound out (checked below).
    for net in (LAN_A, LAN_B):
        docker("network", "create", "--subnet", SUBNETS[net], net)
    # LAN P has no way out at all but the proxy, which sits on it.
    docker("network", "create", "--internal", "--subnet", SUBNETS[LAN_P], LAN_P)
    nat_setup = (
        "set -e; WAN=$(ip -o -4 addr | awk '/172\\.31\\.0\\./{print $2}'); "
        "IPT=iptables; $IPT -t nat -L >/dev/null 2>&1 || IPT=iptables-legacy; "
        "$IPT -t nat -A POSTROUTING -o $WAN -j MASQUERADE; "
        "$IPT -A INPUT -i $WAN -p tcp -m conntrack --ctstate NEW -j DROP; "
        "$IPT -A FORWARD -i $WAN -m conntrack --ctstate NEW -j DROP; "
        # Private addresses are not routed across the internet.
        "$IPT -A FORWARD -o $WAN -d 10.31.0.0/16 -j DROP; "
        "echo nat up on $WAN")
    for name, (wan_ip, lan, lan_ip) in GW.items():
        run_container(name, PUB, wan_ip, "sleep infinity", caps=True, sysctl=True)
        docker("network", "connect", "--ip", lan_ip, lan, name)
        say(f"{name}: {docker('exec', name, 'sh', '-c', nat_setup)} "
            f"(public {wan_ip}, LAN {lan_ip})")
    pname, pub_ip, lan_ip = PROXY
    run_container(pname, PUB, pub_ip,
                  f"exec python /repo/scripts/s1/gateway.py proxy --port 3128 "
                  f"--allow {RELAY_IP}:4001")
    docker("network", "connect", "--ip", lan_ip, LAN_P, pname)
    run_container(
        "mesh-relay", PUB, RELAY_IP,
        "exec python -m nodes.relay --port 4001 --tcp-port 4002 --host 0.0.0.0 "
        f"--no-upnp --announce /ip4/{RELAY_IP}/tcp/4002,/ip4/{RELAY_IP}/tcp/4001/ws "
        f"--jurisdiction {TOPIC_KEY} --data-dir /data --advertise-interval 15")
    peer = ""

    def relay_up() -> bool:
        nonlocal peer
        for line in docker("logs", "mesh-relay", check_=False).splitlines():
            if line.startswith("relay address: ") and "/p2p/" in line:
                peer = line.rsplit("/p2p/", 1)[1].strip()
        return bool(peer)
    if not wait(relay_up, 90, 1):
        raise RuntimeError("relay did not start:\n" + docker("logs", "mesh-relay", check_=False))
    say(f"relay {peer[:20]}... on {RELAY_IP} (tcp 4002, ws 4001)")
    return peer


def gw_mac(lan_ip: str) -> str:
    for name, (_wan, _lan, gw_lan_ip) in GW.items():
        if gw_lan_ip == lan_ip:
            return docker("exec", name, "sh", "-c",
                          r"ip -o -4 addr | awk '/10\.31\./{print $2}' | head -1 | "
                          r"xargs -I{} cat /sys/class/net/{}/address")
    raise KeyError(lan_ip)


def start_nodes(relay_peer: str) -> None:
    for name, (net, ip, gw, ws, tcp, kind) in NODES.items():
        env = {"MESH_NAME": name, "MESH_PORT": str(ws), "MESH_TCP_PORT": str(tcp),
               "MESH_TOPIC_KEY": TOPIC_KEY}
        if kind == "proxied":
            proxy = f"http://{PROXY[2]}:3128"
            env.update({"HTTPS_PROXY": proxy, "HTTP_PROXY": proxy,
                        "MESH_SEEDS": f"/ip4/{RELAY_IP}/tcp/4001/ws/p2p/{relay_peer}"})
        else:
            env["MESH_SEEDS"] = f"/ip4/{RELAY_IP}/tcp/4002/p2p/{relay_peer}"
        route = ""
        if gw:
            # Out through the gateway only, and in only through it: Docker
            # routes between its bridges on the host, so a packet that did
            # not come from the gateway's MAC (or the LAN) is dropped, as no
            # packet reaches a home LAN except through its router.
            mac, lan = gw_mac(gw), SUBNETS[net]
            route = (f"ip route replace default via {gw} && "
                     f"(iptables -A INPUT -i eth0 ! -s {lan} -m mac ! --mac-source {mac} -j DROP "
                     f"|| iptables-legacy -A INPUT -i eth0 ! -s {lan} -m mac ! --mac-source {mac} -j DROP) && ")
        run_container(name, net, ip, route + "exec python /repo/scripts/mesh/mesh_node.py",
                      env=env, caps=bool(gw))
        say(f"{name}: {kind} on {net} {ip}" + (f" via {gw}" if gw else ""))


def isolation_checks() -> None:
    """The NAT'd LANs are unreachable from outside, the proxied one has no
    route out: raw TCP connects that must fail."""
    probe = ("import socket,sys\n"
             "try:\n"
             "    socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=4)\n"
             "    print('open')\n"
             "except Exception as e:\n"
             "    print('blocked', type(e).__name__)\n")
    for src, ip, port in (("pub-p", "10.31.1.11", 4002), ("pub-p", "10.31.2.11", 4002),
                          ("nat-a1", "10.31.2.11", 4002), ("proxied-h", RELAY_IP, 4002),
                          ("proxied-h", PUB_P_IP, 4002)):
        r = docker("exec", src, "python", "-c", probe, ip, str(port), check_=False)
        check(r.startswith("blocked"), f"isolation: {src} -> {ip}:{port} {r}")
    r = docker("exec", "nat-a1", "python", "-c", probe, RELAY_IP, "4002", check_=False)
    check(r == "open", f"isolation: nat-a1 -> relay {RELAY_IP}:4002 through its NAT {r}")


def statuses() -> dict:
    return {n: ctl(n, "status") for n in NODES}


def links(st: dict, a: str, b: str) -> list:
    """Connections from a to b: [(kind, remote ip:port)]."""
    pid = st.get(b, {}).get("peer_id")
    out = []
    for c in st.get(a, {}).get("connections", {}).get(pid, []) if pid else []:
        remote = c.get("remote")
        out.append((c["kind"], f"{remote[0]}:{remote[1]}" if remote else "-"))
    return out


def has_direct(st: dict, a: str, b: str) -> bool:
    return any(k == "direct" for k, _ in links(st, a, b))


def table(st: dict) -> None:
    for a in NODES:
        s = st.get(a, {})
        row = []
        for b in NODES:
            if a == b:
                continue
            ls = links(st, a, b)
            row.append(f"{b}={'+'.join(k + '@' + r for k, r in ls) if ls else 'none'}")
        say(f"  {a} [{s.get('transport')}, dht {s.get('dht_mode')}/{s.get('dht_peers')}, "
            f"reservations {len(s.get('reservations') or [])}]: " + "  ".join(row))


def dump_logs() -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    for name in list(NODES) + ["mesh-relay", "mesh-proxy"] + list(GW):
        r = subprocess.run(["docker", "logs", name], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        (RUN / f"{name}.log").write_text((r.stdout or "") + (r.stderr or ""), encoding="utf-8")


def tool_fetched_everywhere(tool: str, nodes: list) -> dict:
    out = {}
    for n in nodes:
        rec = ctl(n, "received")
        hit = [f for f in rec.get("fetches", []) if f.get("tool") == tool]
        out[n] = hit[-1] if hit else None
    return out


def phase_a() -> None:
    say("(a) waiting for the mesh: direct links, NAT'd to NAT'd by hole punching")
    nat = [n for n, v in NODES.items() if v[5] == "nat"]
    pairs = [(x, y) for x in nat for y in nat if x < y]

    def formed() -> bool:
        st = statuses()
        return (all(has_direct(st, x, y) for x, y in pairs)
                and all(has_direct(st, n, "pub-p") for n in nat))
    wait(formed, 240, 10)
    st = statuses()
    table(st)
    for x, y in pairs:
        ls = links(st, x, y)
        same_lan = NODES[x][0] == NODES[y][0]
        how = "same LAN" if same_lan else f"across NATs ({NAT_IP[x]} <-> {NAT_IP[y]})"
        evidence = ", ".join(f"{k}@{r}" for k, r in ls) or "no connection"
        ok = has_direct(st, x, y)
        if ok and not same_lan:
            # The remote end of a hole-punched link is the other NAT's
            # public address on the peer's TCP listen port.
            remote_ok = any(k == "direct" and r.split(":")[0] == NAT_IP[y] for k, r in ls) or \
                any(k == "direct" and r.split(":")[0] == NAT_IP[x] for k, r in links(st, y, x))
            ok = ok and remote_ok
        check(ok, f"(a) {x} <-> {y} direct, {how}: {evidence}")
    leaks = [(x, y, r) for x in nat for y in nat if NODES[x][0] != NODES[y][0]
             for k, r in links(st, x, y) if k == "direct" and r.startswith("10.31.")]
    check(not leaks, f"(a) no link between the LANs uses a private address: {leaks or 'none'}")
    for n in nat:
        check(has_direct(st, n, "pub-p"), f"(a) {n} <-> pub-p direct: "
              + ", ".join(f"{k}@{r}" for k, r in links(st, n, "pub-p")))
    check(any(links(st, "proxied-h", n) for n in nat + ["pub-p"]),
          "(a) proxied-h reaches daemons through relay circuits: "
          + ", ".join(f"{n}={'+'.join(k for k, _ in links(st, 'proxied-h', n))}"
                      for n in nat + ["pub-p"] if links(st, "proxied-h", n)))


def phase_b() -> None:
    say("(b) nat-b1 publishes a tool: provide manifest + code, then gossip the registration")
    pub = ctl("nat-b1", "publish_tool", {"name": "tool-b1"})
    say(f"  nat-b1 manifest {pub.get('manifest', '?')[:16]} code {pub.get('code', '?')[:16]} "
        f"provider records placed: {pub.get('provided')}")
    others = [n for n in NODES if n != "nat-b1"]
    wait(lambda: all(tool_fetched_everywhere("tool-b1", others).values()), 180, 5)
    b1 = ctl("nat-b1", "status").get("peer_id")
    for n, f in tool_fetched_everywhere("tool-b1", others).items():
        if not f:
            check(False, f"(b) {n} fetched tool-b1: never")
            continue
        provs = f["results"][0]["providers"]
        via = f["results"][0].get("via") or {}
        detail = (f"providers found in the DHT: {len(provs)} (nat-b1 among them: "
                  f"{b1 in provs}); fetched over {via.get(b1)} in "
                  f"{[r['seconds'] for r in f['results']]} s")
        tag = "(d)" if NODES[n][5] == "proxied" else "(b)"
        check(f["ok"] and b1 in provs, f"{tag} {n} found tool-b1 via the DHT and fetched it: {detail}")


def phase_d() -> None:
    say("(d) the proxied daemon publishes a tool and gossips; it can only leave through the relay")
    pub = ctl("proxied-h", "publish_tool", {"name": "tool-h"})
    say(f"  proxied-h provider records placed: {pub.get('provided')}")
    others = [n for n in NODES if n != "proxied-h"]
    wait(lambda: all(tool_fetched_everywhere("tool-h", others).values()), 180, 5)
    h = ctl("proxied-h", "status").get("peer_id")
    for n, f in tool_fetched_everywhere("tool-h", others).items():
        via = ((f or {}).get("results") or [{}])[0].get("via") or {}
        check(bool(f and f["ok"]), f"(d) {n} fetched proxied-h's tool via the DHT over {via.get(h)}")
    ctl("nat-a1", "gossip", {"text": "hello-before-relay-stop"})
    time.sleep(8)
    got = [e for e in ctl("proxied-h", "received").get("received", [])
           if e.get("text") == "hello-before-relay-stop"]
    check(bool(got), "(d) proxied-h received nat-a1's gossip")
    proxy_log = docker("logs", "mesh-proxy", check_=False)
    allowed = sorted({ln.split(" ", 1)[1] for ln in proxy_log.splitlines() if ln.startswith("ALLOW ")})
    denied = sorted({ln.split(" ", 1)[1] for ln in proxy_log.splitlines() if ln.startswith("DENY ")})
    say(f"  proxy allowed {allowed}, denied {denied or 'nothing'}")


def phase_c() -> None:
    say("(c) stopping the relay: every daemon's links to the others must hold")
    docker("stop", "mesh-relay")
    time.sleep(75)  # DCUtR closes relayed legs a minute after the upgrade
    st = statuses()
    table(st)
    ctl("nat-a1", "gossip", {"text": "hello-without-relays"})
    reachable = [n for n in NODES if NODES[n][5] != "proxied" and n != "nat-a1"]
    wait(lambda: all(any(e.get("text") == "hello-without-relays"
                         for e in ctl(n, "received").get("received", [])) for n in reachable), 60, 3)
    for n in reachable:
        got = any(e.get("text") == "hello-without-relays"
                  for e in ctl(n, "received").get("received", []))
        check(got, f"(c) {n} received nat-a1's gossip with the relay down")
    got_h = any(e.get("text") == "hello-without-relays"
                for e in ctl("proxied-h", "received").get("received", []))
    say(f"  proxied-h received it: {got_h} (expected False: the relay is its only way in)")
    pub = ctl("nat-a2", "publish_tool", {"name": "tool-a2-no-relay"})
    say(f"  nat-a2 published tool-a2-no-relay, provider records placed: {pub.get('provided')}")
    others = [n for n in reachable if n != "nat-a2"] + ["nat-a1"]
    others = list(dict.fromkeys(others))
    wait(lambda: all(tool_fetched_everywhere("tool-a2-no-relay", others).values()), 150, 5)
    a2 = ctl("nat-a2", "status").get("peer_id")
    for n, f in tool_fetched_everywhere("tool-a2-no-relay", others).items():
        provs = ((f or {}).get("results") or [{}])[0].get("providers") or []
        via = ((f or {}).get("results") or [{}])[0].get("via") or {}
        check(bool(f and f["ok"] and a2 in provs),
              f"(c) {n} found tool-a2-no-relay in the DHT and fetched it over {via.get(a2)}")


def phase_e() -> None:
    say("(e) restarting nat-b1 with its only seed (the relay) down")
    docker("restart", "nat-b1")
    time.sleep(3)

    def rejoined() -> bool:
        s = ctl("nat-b1", "status")
        return bool(s.get("connections"))
    ok = wait(rejoined, 90, 3)
    s = ctl("nat-b1", "status")
    names = {v.get("peer_id"): k for k, v in statuses().items() if v.get("peer_id")}
    peers = [names.get(p, p[:12]) for p in (s.get("connections") or {})]
    check(ok, f"(e) nat-b1 rejoined through its peer cache ({s.get('peer_cache')} cached): "
              f"connected to {peers}")


def main() -> int:
    if not docker("images", "-q", IMAGE, check_=False):
        print(f"image {IMAGE} missing: docker build -f docker/Dockerfile.mesh -t {IMAGE} .")
        return 2
    RUN.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        relay = start_network()
        start_nodes(relay)
        isolation_checks()
        phase_a()
        phase_b()
        phase_d()
        phase_c()
        phase_e()
    finally:
        dump_logs()
        if not KEEP:
            cleanup()
        passed = sum(1 for ok, _ in _results if ok)
        say(f"{passed}/{len(_results)} checks passed in {(time.time() - t0) / 60:.1f} min; "
            f"logs in {RUN}")
        TRANSCRIPT.parent.mkdir(parents=True, exist_ok=True)
        TRANSCRIPT.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    return 0 if _results and all(ok for ok, _ in _results) else 1


if __name__ == "__main__":
    sys.exit(main())
