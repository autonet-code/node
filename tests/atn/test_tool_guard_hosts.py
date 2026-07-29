"""tool_guard destination allowlist (docs/tool_secret_binding.md — change 2).

The guard is explicitly NOT an egress firewall (see its module docstring).
These tests pin what it DOES catch, and — just as importantly — record what it
does not, so nobody later mistakes it for a stronger control than it is.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

GUARD = Path(__file__).resolve().parents[2] / "atn" / "tool_guard.py"

_RESOLVE = (
    "import socket\n"
    "try:\n"
    "    socket.getaddrinfo({host!r}, 443)\n"
    "    print('ALLOWED')\n"
    "except PermissionError:\n"
    "    print('BLOCKED')\n"
    "except Exception:\n"
    "    print('ALLOWED')\n"  # DNS failure still means the guard let it through
)

_CONNECT = (
    "import socket\n"
    "s = socket.socket(); s.settimeout(0.2)\n"
    "try:\n"
    "    s.connect(({host!r}, 9))\n"
    "    print('ALLOWED')\n"
    "except PermissionError:\n"
    "    print('BLOCKED')\n"
    "except Exception:\n"
    "    print('ALLOWED')\n"
)


def _run(code: str, policy: dict) -> str:
    d = Path(tempfile.mkdtemp())
    script = d / "t.py"
    script.write_text(code, encoding="utf-8")
    env = {
        "PATH": os.path.dirname(sys.executable),
        "SystemRoot": os.environ.get("SystemRoot", ""),
        "PYTHONIOENCODING": "utf-8",
        "TEMP": str(d), "TMP": str(d),
        "ATN_TOOL_POLICY": json.dumps(policy),
    }
    proc = subprocess.run(
        [sys.executable, str(GUARD), str(script)],
        env=env, cwd=str(d), capture_output=True, text=True, timeout=60,
    )
    return proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "ERROR"


_ALLOWLIST = {"net": True, "hosts": ["api.openai.com", "example.com"]}


@pytest.mark.parametrize("host,expected", [
    ("api.openai.com", "ALLOWED"),      # exact match
    ("cdn.example.com", "ALLOWED"),     # subdomain of a listed name
    ("evil.attacker.com", "BLOCKED"),   # unrelated
    ("notexample.com", "BLOCKED"),      # lookalike suffix — must NOT match
    ("example.com.evil.net", "BLOCKED"),  # listed name as a prefix label
])
def test_destination_allowlist(host, expected):
    """PROPERTY: suffix matching is on LABEL boundaries, so a lookalike domain
    that merely ends with the allowed string is rejected."""
    assert _run(_RESOLVE.format(host=host), _ALLOWLIST) == expected


def test_raw_ip_connect_is_checked():
    """PROPERTY: connecting to a literal IP does not skip the check — the IP
    string itself must be listed.

    This is the DNS-bypass case: a tool that resolves a name itself and then
    connects by address still meets the allowlist at socket.connect.
    """
    assert _run(_CONNECT.format(host="127.0.0.1"), _ALLOWLIST) == "BLOCKED"
    assert _run(_CONNECT.format(host="127.0.0.1"),
                {"net": True, "hosts": ["127.0.0.1"]}) == "ALLOWED"


def test_empty_allowlist_is_unrestricted():
    """PROPERTY: BACKWARD COMPATIBILITY. No configured hosts means no claim
    about where the tool may go — prior behavior, unchanged."""
    assert _run(_RESOLVE.format(host="anywhere.example.org"),
                {"net": True}) == "ALLOWED"


def test_net_false_still_wins():
    """PROPERTY: the allowlist NARROWS net; it never widens it. A listed host
    is still denied when net was not declared at all."""
    assert _run(_RESOLVE.format(host="api.openai.com"),
                {"net": False, "hosts": ["api.openai.com"]}) == "BLOCKED"


def test_unix_socket_path_is_not_a_network_destination():
    """PROPERTY: an AF_UNIX connect carries a path, not a (host, port). It must
    not be misread as an unauthorized host.

    This matters because the broker itself is reached over AF_UNIX on POSIX —
    misclassifying it would make a secret-bound tool unable to read its own
    secret.
    """
    if sys.platform == "win32":
        pytest.skip("AF_UNIX connect shape is POSIX-specific")
    code = (
        "import socket\n"
        "s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
        "s.settimeout(0.2)\n"
        "try:\n"
        "    s.connect('/tmp/definitely-not-here.sock')\n"
        "    print('ALLOWED')\n"
        "except PermissionError:\n"
        "    print('BLOCKED')\n"
        "except Exception:\n"
        "    print('ALLOWED')\n"
    )
    assert _run(code, _ALLOWLIST) == "ALLOWED"
