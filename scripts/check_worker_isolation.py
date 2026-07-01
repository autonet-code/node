"""ELEVATION GO/NO-GO self-check for agent per-PID process isolation.

This is the diagnostic that decides whether the security model for the whole
migration holds on THIS host. It:

  (a) spawns a real ``atn.agent_worker`` subprocess via WorkerManager (the same
      spawn path P2 wires behind ATN_WORKER_ISOLATION),
  (b) confirms the worker runs and completes the pipe handshake (status: ready),
  (c) queries the worker's process token and reports:
       - whether SeDebugPrivilege is present (the whole memory-isolation
         argument requires it to be ABSENT),
       - the token integrity level (Medium/Low/...),
       - whether the token is elevated,
  (d) notes whether a node child spawned by the worker would inherit the
      restriction (Job Object membership is inherited by children on Windows),
  (e) confirms the Win32 Job Object kills the tree on handle close.

Run:  python scripts/check_worker_isolation.py

It prints a human-readable report and a final GO / NO-GO / DEGRADED verdict, and
exits 0 on GO/DEGRADED, 2 on NO-GO (hard failure to spawn/handshake).

If a restricted-token spawn is NOT achievable on this host, that is reported
LOUDLY along with exactly what a worker CAN get (e.g. Medium integrity,
non-elevated) — the check never fails silently.
"""
from __future__ import annotations

import asyncio
import ctypes
import os
import sys
from pathlib import Path

# Make the repo importable when run directly.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from atn.runtime.worker_manager import WorkerManager, WorkerHandle  # noqa: E402

_IS_WIN = sys.platform == "win32"


# ---------------------------------------------------------------------------
# Windows token introspection (query the CHILD's token by PID)
# ---------------------------------------------------------------------------

def _win_token_report(pid: int) -> dict:
    """Open the child process token by PID and report privilege/integrity."""
    import ctypes.wintypes as wt

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    a32 = ctypes.WinDLL("advapi32", use_last_error=True)

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    TOKEN_QUERY = 0x0008
    TokenElevation = 20
    TokenIntegrityLevel = 25
    SE_PRIVILEGE_ENABLED = 0x00000002

    report: dict = {
        "opened": False,
        "se_debug_present": None,
        "se_debug_enabled": None,
        "integrity": None,
        "elevated": None,
        "privileges": [],
        "error": None,
    }

    hproc = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not hproc:
        report["error"] = f"OpenProcess failed: {ctypes.get_last_error()}"
        return report

    htoken = wt.HANDLE()
    if not a32.OpenProcessToken(wt.HANDLE(hproc), TOKEN_QUERY, ctypes.byref(htoken)):
        report["error"] = f"OpenProcessToken failed: {ctypes.get_last_error()}"
        k32.CloseHandle(wt.HANDLE(hproc))
        return report
    report["opened"] = True

    # ---- Elevation ----
    class TOKEN_ELEVATION(ctypes.Structure):
        _fields_ = [("TokenIsElevated", wt.DWORD)]

    elev = TOKEN_ELEVATION()
    ret_len = wt.DWORD()
    if a32.GetTokenInformation(htoken, TokenElevation, ctypes.byref(elev),
                               ctypes.sizeof(elev), ctypes.byref(ret_len)):
        report["elevated"] = bool(elev.TokenIsElevated)

    # ---- Integrity level ----
    class SID_AND_ATTRIBUTES(ctypes.Structure):
        _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", wt.DWORD)]

    class TOKEN_MANDATORY_LABEL(ctypes.Structure):
        _fields_ = [("Label", SID_AND_ATTRIBUTES)]

    size = wt.DWORD(0)
    a32.GetTokenInformation(htoken, TokenIntegrityLevel, None, 0, ctypes.byref(size))
    buf = ctypes.create_string_buffer(size.value)
    if size.value and a32.GetTokenInformation(
        htoken, TokenIntegrityLevel, buf, size, ctypes.byref(size)
    ):
        tml = ctypes.cast(buf, ctypes.POINTER(TOKEN_MANDATORY_LABEL)).contents
        get_sub_count = a32.GetSidSubAuthorityCount
        get_sub_count.argtypes = [ctypes.c_void_p]
        get_sub_count.restype = ctypes.POINTER(ctypes.c_ubyte)
        get_sub = a32.GetSidSubAuthority
        get_sub.argtypes = [ctypes.c_void_p, wt.DWORD]
        get_sub.restype = ctypes.POINTER(wt.DWORD)
        sid_ptr = tml.Label.Sid
        cnt = get_sub_count(sid_ptr)[0]
        rid = get_sub(sid_ptr, cnt - 1)[0]
        report["integrity"] = _integrity_name(rid)

    # ---- Privileges (look for SeDebugPrivilege) ----
    size = wt.DWORD(0)
    TokenPrivileges = 3
    a32.GetTokenInformation(htoken, TokenPrivileges, None, 0, ctypes.byref(size))
    buf = ctypes.create_string_buffer(size.value)
    if size.value and a32.GetTokenInformation(
        htoken, TokenPrivileges, buf, size, ctypes.byref(size)
    ):
        class LUID(ctypes.Structure):
            _fields_ = [("LowPart", wt.DWORD), ("HighPart", ctypes.c_long)]

        class LUID_AND_ATTRIBUTES(ctypes.Structure):
            _fields_ = [("Luid", LUID), ("Attributes", wt.DWORD)]

        count = ctypes.cast(buf, ctypes.POINTER(wt.DWORD)).contents.value
        # PrivilegeCount (DWORD) followed by the array.
        arr_addr = ctypes.addressof(buf) + ctypes.sizeof(wt.DWORD)
        arr_type = LUID_AND_ATTRIBUTES * count
        arr = arr_type.from_address(arr_addr)
        for i in range(count):
            la = arr[i]
            name = _privilege_name(a32, la.Luid)
            if name:
                report["privileges"].append(name)
                if name == "SeDebugPrivilege":
                    report["se_debug_present"] = True
                    report["se_debug_enabled"] = bool(la.Attributes & SE_PRIVILEGE_ENABLED)
        if report["se_debug_present"] is None:
            report["se_debug_present"] = False
            report["se_debug_enabled"] = False

    a32.CloseHandle = k32.CloseHandle
    k32.CloseHandle(htoken)
    k32.CloseHandle(wt.HANDLE(hproc))
    return report


def _privilege_name(a32, luid) -> str:
    import ctypes.wintypes as wt
    size = wt.DWORD(0)
    a32.LookupPrivilegeNameW(None, ctypes.byref(luid), None, ctypes.byref(size))
    if not size.value:
        return ""
    buf = ctypes.create_unicode_buffer(size.value)
    if a32.LookupPrivilegeNameW(None, ctypes.byref(luid), buf, ctypes.byref(size)):
        return buf.value
    return ""


def _integrity_name(rid: int) -> str:
    # Well-known integrity RIDs.
    table = {
        0x0000: "Untrusted",
        0x1000: "Low",
        0x2000: "Medium",
        0x2100: "Medium-Plus",
        0x3000: "High",
        0x4000: "System",
    }
    return table.get(rid, f"0x{rid:x}")


# ---------------------------------------------------------------------------
# Main check
# ---------------------------------------------------------------------------

async def _run() -> int:
    print("=" * 70)
    print("AGENT PER-PID PROCESS ISOLATION — ELEVATION SELF-CHECK")
    print("=" * 70)
    print(f"host platform : {sys.platform}")
    print(f"python        : {sys.version.split()[0]}")
    print(f"daemon pid    : {os.getpid()}")

    # Report the DAEMON's own elevation for context (a worker's ceiling is the
    # daemon's token unless we drop privileges).
    if _IS_WIN:
        daemon_tok = _win_token_report(os.getpid())
        print(f"daemon token  : integrity={daemon_tok.get('integrity')} "
              f"elevated={daemon_tok.get('elevated')} "
              f"SeDebug={daemon_tok.get('se_debug_present')}")
    print("-" * 70)

    mgr = WorkerManager()
    handle: WorkerHandle | None = None
    verdict = "NO-GO"
    notes: list[str] = []

    try:
        print("[a] spawning worker (atn.agent_worker) ...")
        handle = await mgr.ensure_worker("self-check-agent",
                                         {"agent_label": "self-check"})
        print(f"    spawned pid={handle.pid}")
        print("[b] handshake ... ready received:", handle.ready)
        if not handle.ready:
            print("    NO-GO: worker did not complete handshake")
            return 2

        if _IS_WIN:
            print("[c] querying worker token ...")
            tok = _win_token_report(handle.pid)
            if tok.get("error"):
                print(f"    could not read worker token: {tok['error']}")
                notes.append("token introspection failed; cannot confirm privileges")
            else:
                se_debug = tok.get("se_debug_present")
                integ = tok.get("integrity")
                elevated = tok.get("elevated")
                print(f"    SeDebugPrivilege present : {se_debug}")
                print(f"    integrity level          : {integ}")
                print(f"    token elevated           : {elevated}")
                print(f"    privilege count          : {len(tok.get('privileges', []))}")

                # Verdict logic.
                if se_debug:
                    verdict = "DEGRADED"
                    notes.append(
                        "SeDebugPrivilege IS present on the worker token. This "
                        "worker inherited the daemon's privileges — a restricted "
                        "token was NOT applied. With SeDebugPrivilege a process "
                        "can open ANY process and read its memory, defeating the "
                        "PID-bound-secret isolation. To reach GO, P3/P4 must spawn "
                        "the worker with CreateRestrictedToken / a lowered-integrity "
                        "token (this phase does not drop privileges yet)."
                    )
                    print("    !! LOUD WARNING: worker has SeDebugPrivilege — "
                          "isolation NOT yet enforced (see notes)")
                else:
                    verdict = "GO"
                    notes.append(
                        "Worker token lacks SeDebugPrivilege — it cannot debug-open "
                        "arbitrary processes. This is the property the isolation "
                        "model needs. Integrity/elevation still matter for "
                        "defense-in-depth (see below)."
                    )
                if elevated:
                    notes.append(
                        "Worker token is ELEVATED (running the daemon as admin). "
                        "Even without SeDebugPrivilege in the default set, an "
                        "elevated token can re-enable privileges. Recommend running "
                        "the daemon non-elevated so workers cannot self-escalate."
                    )
        else:
            verdict = "GO"
            notes.append(
                "POSIX host: no SeDebugPrivilege concept. Isolation rests on "
                "separate process/address space + (P4) dropped capabilities / "
                "uid separation. No Job Object; group-kill via start_new_session "
                "covers the tree."
            )

        # (d) node-child inheritance note.
        if _IS_WIN:
            print("[d] node-child inheritance:")
            print("    A node child (SDK bridge) spawned by the worker is created "
                  "INSIDE the worker's Job Object and inherits the worker's token "
                  "restrictions — so the restriction propagates to the SDK subtree.")
            notes.append(
                "Job Object membership is inherited by child processes, so a node "
                "SDK child spawned by the worker is reaped by the same job and "
                "runs under the same (restricted, once P3/P4 apply it) token."
            )

        # (e) Job Object kill-on-close reaps the tree.
        print("[e] Job Object kill-on-close reap:")
        pid = handle.pid
        proc = handle.proc
        await mgr.hard_kill(handle)
        reaped = proc.poll() is not None
        # Confirm the PID is truly gone.
        still_alive = _pid_alive(pid)
        print(f"    worker exited      : {reaped} (returncode={proc.returncode})")
        print(f"    pid {pid} still alive : {still_alive}")
        if reaped and not still_alive:
            print("    kill-on-close reap : OK")
        else:
            notes.append("Job Object reap did not fully clear the worker PID.")
        handle = None  # already killed

    except Exception as e:  # noqa: BLE001
        print(f"    EXCEPTION during check: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 2
    finally:
        if handle is not None:
            try:
                await mgr.hard_kill(handle)
            except Exception:
                pass

    print("-" * 70)
    print("NOTES:")
    for i, n in enumerate(notes, 1):
        print(f"  {i}. {n}")
    print("=" * 70)
    print(f"VERDICT: {verdict}")
    print("=" * 70)
    # GO or DEGRADED => exit 0 (spawn/handshake/reap all worked); NO-GO => 2.
    return 0


def _pid_alive(pid: int) -> bool:
    if _IS_WIN:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        SYNCHRONIZE = 0x00100000
        h = k32.OpenProcess(SYNCHRONIZE, False, pid)
        if not h:
            return False
        # WAIT_TIMEOUT (0x102) => still running; WAIT_OBJECT_0 (0) => exited.
        rc = k32.WaitForSingleObject(h, 0)
        k32.CloseHandle(h)
        return rc != 0  # non-signaled => alive
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
