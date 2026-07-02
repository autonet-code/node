"""POSIX integration tests for the cross-platform isolation port.

Gated to POSIX (skipped on Windows). Covers the two hard parts the Windows
build could not exercise:
  * tracked-PID kill + process-group backstop (P1 containment), incl. the
    documented setsid-escape residual;
  * the broker's portable ``_identity`` (Linux /proc starttime) PID-reuse guard;
  * AF_UNIX broker + SO_PEERCRED peer-PID auth end-to-end (P2), incl. the
    fail-closed invariants (unregistered / no-tripwire / post-release denied).

These formalize the manual EC2 verification runs into the committed suite.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time

import pytest

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="POSIX isolation path; Windows uses Job Objects + named pipes")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, OSError):
        return False


# ── P1: tracked-PID kill + group backstop ─────────────────────────────────────
def test_tracked_pid_kill_reaps_worker_and_child():
    """A worker in its own session (start_new_session) that spawns a child (the
    bridge grandchild) is fully reaped by per-PID kill + group sweep."""
    worker_src = (
        "import subprocess,time;"
        "c=subprocess.Popen(['sleep','30']);"
        "print(c.pid,flush=True); time.sleep(30)"
    )
    proc = subprocess.Popen([sys.executable, "-c", worker_src],
                            stdout=subprocess.PIPE, start_new_session=True)
    try:
        child_pid = int(proc.stdout.readline().decode().strip())
        time.sleep(0.2)
        assert _alive(proc.pid) and _alive(child_pid)

        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            pgid = None

        def _killpg(sig):
            if pgid is None:
                return
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                pass

        def _killpid(sig):
            try:
                proc.send_signal(sig)
            except ProcessLookupError:
                pass

        # primary: tracked PID; backstop: the group
        _killpid(signal.SIGTERM)
        _killpg(signal.SIGTERM)
        time.sleep(0.4)
        if _alive(proc.pid) or _alive(child_pid):
            _killpid(signal.SIGKILL)
            _killpg(signal.SIGKILL)
            time.sleep(0.4)
        proc.wait(timeout=5)
        assert not _alive(proc.pid), "worker survived"
        assert not _alive(child_pid), "bridge grandchild survived — containment leak"
    finally:
        if proc.poll() is None:
            proc.kill()


def test_setsid_escape_residual_is_documented():
    """A grandchild that detaches (start_new_session) BEFORE we observe it escapes
    the group kill. This is the accepted, cross-platform residual — assert it so
    the boundary is pinned (and clean it up)."""
    worker_src = (
        "import subprocess,time;"
        "c=subprocess.Popen(['sleep','30'], start_new_session=True);"  # detached
        "print(c.pid,flush=True); time.sleep(30)"
    )
    proc = subprocess.Popen([sys.executable, "-c", worker_src],
                            stdout=subprocess.PIPE, start_new_session=True)
    detached = None
    try:
        detached = int(proc.stdout.readline().decode().strip())
        time.sleep(0.2)
        pgid = os.getpgid(proc.pid)
        proc.send_signal(signal.SIGKILL)
        os.killpg(pgid, signal.SIGKILL)
        time.sleep(0.4)
        proc.wait(timeout=5)
        assert _alive(detached), "detached grandchild unexpectedly reaped by group kill"
    finally:
        if proc.poll() is None:
            proc.kill()
        if detached is not None and _alive(detached):
            try:
                os.kill(detached, signal.SIGKILL)
            except OSError:
                pass


# ── P0/P2: portable _identity (Linux /proc) ───────────────────────────────────
@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="/proc starttime is Linux-specific")
def test_identity_proc_parsing_and_reuse_signal():
    from atn._vendor.kevin.vault import vault_broker as vb
    me = vb._identity(os.getpid())
    assert me and me.startswith(f"{os.getpid()}:")
    # dead pid -> None
    assert vb._identity(999999) is None
    # comm with spaces/parens must not break field indexing
    p = subprocess.Popen(["bash", "-c", 'exec -a "weird ) name" sleep 5'])
    try:
        time.sleep(0.2)
        ident = vb._identity(p.pid)
        assert ident and ident.startswith(f"{p.pid}:")
    finally:
        p.kill()


# ── P2: AF_UNIX broker + SO_PEERCRED end-to-end ───────────────────────────────
pytest.importorskip("pyrage", reason="vault keystore needs pyrage")

# The broker binds KEYSTORE_DIR at import (its own top-level ``keystore`` module
# via the _KEVIN_ROOT bootstrap). To mirror production (one KEYSTORE_DIR per
# process) and avoid cross-test module-cache bleed, each scenario runs in a fresh
# subprocess driven by this script. It returns a JSON verdict on stdout.
_BROKER_DRIVER = r'''
import os, sys, json, time, threading, stat

REPO = os.environ["ATN_REPO_ROOT"]
sys.path.insert(0, REPO)
scenario = os.environ["ATN_SCENARIO"]
SECRET = "s3cret-value-xyz"

from atn._vendor.kevin import keystore
keystore.init_identity()
keystore.put_secret("github", SECRET)
with open(os.environ["ATN_VAULT_POLICY_MAP"], "w") as f:
    json.dump({"github": "github"}, f)

from atn._vendor.kevin.vault import vault_broker as vb
threading.Thread(target=vb.serve_forever, daemon=True).start()
sock = os.environ["ATN_VAULT_BROKER_SOCK"]
for _ in range(50):
    if os.path.exists(sock):
        break
    time.sleep(0.1)

from atn.runtime import broker_client as bc
out = {}
try:
    if scenario == "grant_flow":
        pushed = {}
        dc = bc.DaemonBrokerClient()
        out["armed"] = dc.arm_value_push(lambda e: pushed.update(e))
        time.sleep(0.3)
        wc = bc.WorkerBrokerClient()
        out["pre_register_denied"] = not wc.request("github").get("ok")
        mint = dc.mint_nonce(["github"], agent_id="agent-x")
        out["mint_ok"] = bool(mint.get("ok"))
        out["register_ok"] = bool(dc.register(os.getpid(), mint["nonce"]).get("ok"))
        req = wc.request("github")
        out["request_ok"] = bool(req.get("ok"))
        out["no_value_in_reply"] = "value" not in req
        out["has_var_and_path"] = bool(req.get("var_name") and req.get("path"))
        time.sleep(0.3)
        out["tripwire_value_ok"] = pushed.get("value") == SECRET
        out["tripwire_pid_ok"] = pushed.get("pid") == os.getpid()
        out["tripwire_agent_ok"] = pushed.get("agent_id") == "agent-x"
        p = req.get("path")
        out["staged_content_ok"] = bool(p) and open(p).read() == SECRET
        out["staged_mode_0600"] = bool(p) and stat.S_IMODE(os.stat(p).st_mode) == 0o600
        out["release_ok"] = bool(dc.release_session(os.getpid()).get("ok"))
        out["staged_unlinked"] = not (p and os.path.exists(p))
        out["post_release_denied"] = not wc.request("github").get("ok")
    elif scenario == "fail_closed_no_tripwire":
        dc = bc.DaemonBrokerClient()     # do NOT arm_value_push
        wc = bc.WorkerBrokerClient()
        mint = dc.mint_nonce(["github"], agent_id="a")
        dc.register(os.getpid(), mint["nonce"])
        req = wc.request("github")
        out["request_denied"] = not req.get("ok")
        out["no_value_in_reply"] = "value" not in req
    elif scenario == "pid_reuse_guard":
        wc = bc.WorkerBrokerClient()
        with vb._LOCK:
            vb._SESSIONS[os.getpid()] = {
                "identity": "%d:BOGUS_STARTTIME" % os.getpid(),
                "policies": ["github"],
            }
        out["stale_identity_denied"] = not wc.request("github").get("ok")
    out["_error"] = None
except Exception as e:  # surface the failure to the parent
    out["_error"] = repr(e)
print("RESULT " + json.dumps(out))
'''


def _run_broker_scenario(tmp_path, scenario):
    ks_dir = str(tmp_path / "ks")
    os.makedirs(ks_dir, exist_ok=True)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ)
    env.update({
        "ATN_REPO_ROOT": repo_root,
        "ATN_SCENARIO": scenario,
        "KEYSTORE_DIR": ks_dir,
        "ATN_VAULT_BROKER_SOCK": os.path.join(ks_dir, "vault-broker.sock"),
        "ATN_VAULT_POLICY_MAP": os.path.join(ks_dir, "service_policy_map.json"),
        "BROKER_OWNER_SECRET": "test-owner",
    })
    proc = subprocess.run([sys.executable, "-c", _BROKER_DRIVER],
                          env=env, capture_output=True, text=True, timeout=60)
    line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None)
    assert line is not None, (
        f"driver produced no RESULT.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    out = json.loads(line[len("RESULT "):])
    assert out.get("_error") is None, f"driver error: {out['_error']}\n{proc.stderr}"
    return out


def test_afunix_broker_grant_flow(tmp_path):
    out = _run_broker_scenario(tmp_path, "grant_flow")
    # every recorded invariant must hold
    for k, v in out.items():
        if k == "_error":
            continue
        assert v is True, f"grant-flow invariant failed: {k}={v!r} (all={out})"


def test_afunix_fail_closed_without_tripwire(tmp_path):
    """M7: with no value-push sink armed, a request for a granted service is
    DENIED (no un-scanned secret is ever handed out)."""
    out = _run_broker_scenario(tmp_path, "fail_closed_no_tripwire")
    assert out["request_denied"] is True
    assert out["no_value_in_reply"] is True


def test_afunix_pid_reuse_identity_guard(tmp_path):
    """A session whose recorded _identity no longer matches the live PID is
    rejected (defeats PID reuse)."""
    out = _run_broker_scenario(tmp_path, "pid_reuse_guard")
    assert out["stale_identity_denied"] is True
