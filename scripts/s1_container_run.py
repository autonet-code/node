#!/usr/bin/env python3
"""S1: the S0 run with every daemon in a container behind HTTPS-only egress
(wan_test_plan, "S1").

The cloud VMs that host the WAN run's daemons reach the internet only
through an agent proxy: HTTP CONNECT to allowlisted hostnames, no raw TCP,
no direct DNS. S1 recreates that on one machine:

    host:  hardhat :8545, atn relay :4101 (ws), this harness
    s1-edge network:
      s1-caddy   stands in for Cloudflare: TLS for relay.autonet.test (wss)
                 and rpc.autonet.test (https, token in the path), proxying
                 to the host's relay and hardhat
      s1-proxy   the agent proxy: CONNECT to those two names on 443 only,
                 every other target refused and logged
      s1-ctl     port forwards so the harness can drive each daemon's
                 owner listener (the owner's app, not part of the network)
    s1-internal network (no route out):
      s1-h1..h4  autonet-daemon:s1 containers, HTTPS_PROXY=s1-proxy

The seed in the Registry is /dns4/relay.autonet.test/tcp/443/wss/p2p/<id>,
the shape the real run uses. Everything else (chain, deploy, onboarding,
activity, canary, collector, success criterion) is S0's.

Run:
    docker build -f docker/Dockerfile.daemon -t autonet-daemon:s1 .
    S0_MINUTES=30 python scripts/s1_container_run.py

Env: S1_DIR (default D:/autonet_s1), S1_IMAGE (default autonet-daemon:s1),
plus S0's S0_MINUTES / S0_KEEP / TRUSTLESS_CONTRACTS_DIR.
"""
from __future__ import annotations

import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

os.environ.setdefault("S0_DIR", os.environ.get("S1_DIR", r"D:/autonet_s1"))
import s0_local_run as s0  # noqa: E402

IMAGE = os.environ.get("S1_IMAGE", "autonet-daemon:s1")
RELAY_HOST = "relay.autonet.test"
RPC_HOST = "rpc.autonet.test"
NET_EDGE = "s1-edge"
NET_INTERNAL = "s1-internal"
CTL_PORT_BASE = 7901            # host 127.0.0.1:7901+i -> s1-h<i+1>:7800
IN_CONTAINER_WS = 7700
IN_CONTAINER_FWD = 7800
RPC_TOKEN = secrets.token_hex(12)
DAEMON_RPC = f"https://{RPC_HOST}/{RPC_TOKEN}"
S1_SCRIPTS = REPO / "scripts" / "s1"


def docker(*args: str, check: bool = True, capture: bool = True) -> str:
    r = subprocess.run(["docker", *args], capture_output=capture, text=True,
                       encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        raise RuntimeError(f"docker {' '.join(args[:3])} failed: {r.stderr.strip()[-800:]}")
    return (r.stdout or "").strip()


def mount_path(p: Path) -> str:
    """A host path in the form Docker Desktop accepts for -v."""
    return str(p.resolve()).replace("\\", "/")


def cleanup() -> None:
    names = docker("ps", "-aq", "--filter", "name=^s1-", check=False).split()
    if names:
        docker("rm", "-f", *names, check=False)
    for net in (NET_INTERNAL, NET_EDGE):
        docker("network", "rm", net, check=False)


def write_caddyfile(run: Path) -> Path:
    path = run / "Caddyfile"
    path.write_text(f"""{{
    local_certs
    skip_install_trust
    admin off
}}
{RELAY_HOST} {{
    tls internal
    reverse_proxy host.docker.internal:{s0.RELAY_PORT}
}}
{RPC_HOST} {{
    tls internal
    handle_path /{RPC_TOKEN}* {{
        reverse_proxy host.docker.internal:{s0.RPC_PORT}
    }}
    respond 403
}}
""", encoding="utf-8")
    return path


def start_network(run: Path) -> Path:
    """Networks, Caddy, the proxy and the control forwards. Returns the CA
    certificate the daemons must trust (Caddy's local root)."""
    cleanup()
    docker("network", "create", NET_EDGE)
    docker("network", "create", "--internal", NET_INTERNAL)
    caddyfile = write_caddyfile(run)
    docker("run", "-d", "--name", "s1-caddy", "--network", NET_EDGE,
           "--network-alias", RELAY_HOST, "--network-alias", RPC_HOST,
           "-v", f"{mount_path(caddyfile)}:/etc/caddy/Caddyfile:ro",
           "caddy:2-alpine")
    ca = run / "ca.crt"
    src = "s1-caddy:/data/caddy/pki/authorities/local/root.crt"
    s0.wait_for(lambda: subprocess.run(["docker", "cp", src, str(ca)],
                                       capture_output=True).returncode == 0 and ca.exists(),
                60, "Caddy's local CA")
    allow = f"{RELAY_HOST}:443,{RPC_HOST}:443"
    docker("run", "-d", "--name", "s1-proxy", "--network", NET_EDGE,
           "-v", f"{mount_path(S1_SCRIPTS)}:/s1:ro", "--entrypoint", "python", IMAGE,
           "/s1/gateway.py", "proxy", "--port", "3128", "--allow", allow)
    docker("network", "connect", NET_INTERNAL, "s1-proxy")
    specs = [f"{CTL_PORT_BASE + i}=s1-h{i + 1}:{IN_CONTAINER_FWD}"
             for i in range(s0.HOUSEHOLDS)]
    ports = []
    for i in range(s0.HOUSEHOLDS):
        ports += ["-p", f"127.0.0.1:{CTL_PORT_BASE + i}:{CTL_PORT_BASE + i}"]
    docker("run", "-d", "--name", "s1-ctl", "--network", NET_EDGE, *ports,
           "-v", f"{mount_path(S1_SCRIPTS)}:/s1:ro", "--entrypoint", "python", IMAGE,
           "/s1/gateway.py", "forward", *specs)
    docker("network", "connect", NET_INTERNAL, "s1-ctl")
    s0.log(f"S1 network up: caddy ({RELAY_HOST}, {RPC_HOST}), proxy, ctl :{CTL_PORT_BASE}+")
    return ca


class ContainerHousehold(s0.Household):
    """One household's daemon as a container on the internal network."""

    ca: Path = Path()

    @property
    def container(self) -> str:
        return f"s1-{self.name}"

    def client_url(self) -> str:
        return f"ws://127.0.0.1:{CTL_PORT_BASE + self.i}"

    def log_tail(self) -> str:
        return docker("logs", "--tail", "40", self.container, check=False)

    def alive(self) -> bool:
        return docker("inspect", "-f", "{{.State.Running}}", self.container,
                      check=False) == "true"

    def write_config(self) -> None:
        import yaml
        seed = self.home / "seed"
        seed.mkdir(parents=True, exist_ok=True)
        cfg = {
            "autonet": {
                "enabled": True,
                "local_ws_port": IN_CONTAINER_WS,
                "owner_wallet": self.owner.address,
                "rpc_url": DAEMON_RPC,
                "chain_id": s0.CHAIN_ID,
            },
            "voice": {"enabled": False},
            "auto_update": {"enabled": False},
        }
        (seed / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False),
                                          encoding="utf-8")

    def start(self, procs, registry: str) -> None:
        self.write_config()
        proxy = "http://s1-proxy:3128"
        env = {
            "ATN_JURISDICTION_REGISTRY": registry,
            "ATN_JURISDICTION_RPC": DAEMON_RPC,
            "AUTONET_P2P_TRANSPORT": "ws",
            "HTTPS_PROXY": proxy, "https_proxy": proxy,
            "HTTP_PROXY": proxy, "http_proxy": proxy,
            "NO_PROXY": "localhost,127.0.0.1", "no_proxy": "localhost,127.0.0.1",
            "SSL_CERT_FILE": "/s1run/ca.crt",
            "REQUESTS_CA_BUNDLE": "/s1run/ca.crt",
        }
        envargs = []
        for k, v in env.items():
            envargs += ["-e", f"{k}={v}"]
        # The owner listener binds the container's loopback; a forward on
        # 0.0.0.0 lets s1-ctl reach it. The daemon keeps stdin open (-i):
        # EOF on stdin shuts it down.
        boot = (f"cp /s1seed/config.yaml /data/atn/config.yaml && "
                f"(python /s1/gateway.py forward "
                f"{IN_CONTAINER_FWD}=127.0.0.1:{IN_CONTAINER_WS} &) && "
                f"exec atn --headless")
        docker("run", "-d", "-i", "--name", self.container, "--hostname", self.container,
               "--network", NET_INTERNAL,
               "--cap-drop", "ALL", "--cap-add", "SETUID", "--cap-add", "SETGID",
               "--security-opt", "no-new-privileges:true",
               "-v", f"{mount_path(S1_SCRIPTS)}:/s1:ro",
               "-v", f"{mount_path(self.home / 'seed')}:/s1seed:ro",
               "-v", f"{mount_path(self.ca.parent)}:/s1run:ro",
               *envargs, IMAGE, "sh", "-c", boot)


def seed_addr(peer: str) -> str:
    return f"/dns4/{RELAY_HOST}/tcp/443/wss/p2p/{peer}"


def dump_logs(run: Path) -> None:
    for name in ["s1-proxy", "s1-caddy", "s1-ctl"] + [f"s1-h{i + 1}" for i in range(s0.HOUSEHOLDS)]:
        r = subprocess.run(["docker", "logs", name], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        (run / f"{name}.log").write_text((r.stdout or "") + (r.stderr or ""), encoding="utf-8")
    proxy_log = (run / "s1-proxy.log").read_text(encoding="utf-8", errors="replace")
    denied = sorted({ln.split(" ", 1)[1] for ln in proxy_log.splitlines()
                     if ln.startswith("DENY ")})
    s0.log(f"proxy denied: {denied or 'nothing'}")


def main() -> int:
    if not docker("images", "-q", IMAGE, check=False):
        s0.log(f"image {IMAGE} missing: docker build -f docker/Dockerfile.daemon -t {IMAGE} .")
        return 2
    s0.seed_addr = seed_addr
    s0.Household = ContainerHousehold

    # The run dir is created inside s0.amain; build the network as soon as it
    # exists, before the first household starts.
    real_start = ContainerHousehold.start
    state: dict = {}

    def start_with_network(self, procs, registry):
        if "ca" not in state:
            state["run"] = self.home.parent
            state["ca"] = start_network(state["run"])
            ContainerHousehold.ca = state["ca"]
        real_start(self, procs, registry)

    ContainerHousehold.start = start_with_network
    t0 = time.time()
    try:
        return s0.main()
    finally:
        if "run" in state:
            dump_logs(state["run"])
        if not s0.KEEP:
            cleanup()
        s0.log(f"S1 done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    sys.exit(main())
