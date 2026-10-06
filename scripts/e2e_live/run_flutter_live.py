#!/usr/bin/env python3
"""Boot the live harness, run atn_web's live integration tests, tear down.

    python scripts/e2e_live/run_flutter_live.py            # windows desktop
    python scripts/e2e_live/run_flutter_live.py --web      # + chrome (flutter drive)
    python scripts/e2e_live/run_flutter_live.py --keep-up  # leave harness running

Steps: ``harness.py up`` -> ``flutter test integration_test/live -d windows
--dart-define=DAEMON_WS=<ws>`` in atn_web -> optional web variant ->
``harness.py down`` (always, unless --keep-up). Exit code is non-zero if the
harness failed to come up or any Flutter run failed.

The web variant needs ``chromedriver`` on PATH matching the installed Chrome;
when it is missing the variant is reported SKIP, not FAIL.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness  # noqa: E402

DEFAULT_APP = Path(os.environ.get("ATN_WEB_DIR", r"C:\code\atn_web"))
LIVE_DIR = "integration_test/live"


def _flutter() -> str:
    exe = shutil.which("flutter.bat" if os.name == "nt" else "flutter") \
        or shutil.which("flutter")
    if not exe:
        raise SystemExit("flutter not found on PATH")
    return exe


def run(cmd: list[str], cwd: Path, timeout: float) -> int:
    print(f"\n$ {' '.join(cmd)}   (cwd {cwd})", flush=True)
    started = time.time()
    try:
        rc = subprocess.run(cmd, cwd=str(cwd), timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT after {timeout:.0f}s", flush=True)
        rc = 124
    print(f"-> rc={rc} in {time.time() - started:.0f}s", flush=True)
    return rc


WEB_ENTRY = "_live_web_entry.dart"


def run_web(app: Path, defines: list[str], flutter: str,
            files: list[str]) -> dict[str, str]:
    """Web variant via flutter drive + chromedriver, one file at a time.

    The web compiler roots ``org-dartlang-app:/`` at the TARGET's directory,
    so a target inside integration_test/live cannot import
    ../../test/support or ../support ("File not found"). Each file is run
    through a throwaway entrypoint at the app root (removed afterwards).
    Returns {"web <file>": PASS/FAIL} or {"web": "SKIP"}.
    """
    driver = shutil.which("chromedriver")
    if not driver:
        print("web variant: SKIP (no chromedriver on PATH; it must match the "
              "installed Chrome major version)", flush=True)
        return {"web": "SKIP"}
    proc = subprocess.Popen([driver, "--port=4444"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    entry = app / WEB_ENTRY
    results: dict[str, str] = {}
    try:
        time.sleep(2.0)
        for f in files:
            t = Path(f)
            entry.write_text(
                "// Temporary web entrypoint written by run_flutter_live.py\n"
                f"import '{t.as_posix()}' as t;\n\n"
                "void main() => t.main();\n", encoding="utf-8")
            # web-server + chromedriver: ONE browser instance. `-d chrome`
            # next to a running chromedriver ran the app twice (Flutter's own
            # debug Chrome plus the WebDriver session), both against the same
            # daemon: they fought over the input arbiter and the driver
            # reported the hidden instance's timeouts. 1400x900 keeps the
            # desktop layout (the headless default is much smaller).
            rc = run([flutter, "drive", "-d", "web-server",
                      "--browser-name=chrome", "--browser-dimension=1400,900",
                      f"--driver={LIVE_DIR}/driver.dart",
                      f"--target={WEB_ENTRY}", *defines],
                     app, timeout=1200)
            results[f"web {t.name}"] = "PASS" if rc == 0 else "FAIL"
        return results
    finally:
        entry.unlink(missing_ok=True)
        harness.kill_tree(proc.pid)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--app-dir", default=str(DEFAULT_APP))
    ap.add_argument("--web", action="store_true",
                    help="also run the chrome variant (needs chromedriver)")
    ap.add_argument("--keep-up", action="store_true",
                    help="leave the harness running afterwards")
    ap.add_argument("tests", nargs="*", default=[LIVE_DIR],
                    help=f"test paths under the app (default {LIVE_DIR})")
    args = ap.parse_args()
    app = Path(args.app_dir)
    flutter = _flutter()

    pubspec = (app / "pubspec.yaml").read_text(encoding="utf-8")
    if "integration_test:" not in pubspec:
        raise SystemExit(f"{app}/pubspec.yaml has no integration_test "
                         "dev_dependency; add `integration_test: {sdk: "
                         "flutter}` first")

    up = subprocess.run([sys.executable, str(HERE / "harness.py"), "up"])
    if up.returncode != 0:
        print("harness up FAILED", flush=True)
        return 2
    state = harness.load_state()
    ws_url = state["ws_url"]
    defines = [f"--dart-define=DAEMON_WS={ws_url}"]
    adoptable = ((state.get("fixtures") or {}).get("adoptable_tool")
                 or {}).get("digest")
    if adoptable:
        # A5's adoption approve path (a pinned foreign tool in blobs only).
        defines.append(f"--dart-define=E2E_ADOPTABLE_DIGEST={adoptable}")
    results: dict[str, str] = {}
    try:
        # One `flutter test` per file: with several files in one invocation
        # the Windows device starts the first app and then fails every later
        # one ("The log reader stopped unexpectedly ... Unable to start the
        # app on the device"), so a directory run reported 6 load failures
        # for files that pass on their own.
        files: list[str] = []
        for t in args.tests:
            path = app / t
            if path.is_dir():
                files += sorted(f"{t}/{f.name}"
                                for f in path.glob("*_test.dart"))
            else:
                files.append(t)
        for f in files:
            rc = run([flutter, "test", f, "-d", "windows", *defines],
                     app, timeout=1800)
            results[f"windows {Path(f).name}"] = "PASS" if rc == 0 else "FAIL"
        if args.web:
            results.update(run_web(app, defines, flutter, files))
        st = subprocess.run([sys.executable, str(HERE / "harness.py"),
                             "status"])
        results["harness still healthy"] = "PASS" if st.returncode == 0 else "FAIL"
    finally:
        if not args.keep_up:
            subprocess.run([sys.executable, str(HERE / "harness.py"), "down"])
    print("\n" + json.dumps(results, indent=2), flush=True)
    return 0 if all(v != "FAIL" for v in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
