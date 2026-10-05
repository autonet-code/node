"""`atn integration-token`: mint / revoke / list integration-listener tokens.

Offline: edits ``<data_dir>/integration_tokens.json`` directly; the running
daemon picks up changes on its next handshake or message (no restart). The
plaintext token is printed ONCE at create time and only its sha256 is stored.
See docs/integration_listener.md.

    atn integration-token create --agent <id> [--label <l>] [--data-dir <p>]
    atn integration-token revoke <token-id-or-label> [--data-dir <p>]
    atn integration-token list [--json] [--data-dir <p>]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .ws_auth import IntegrationTokenStore, is_bindable_agent_id


def _data_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg).expanduser()
    from .config import load_config
    return load_config().data_dir


def _known_agent_ids(data_dir_arg: str | None) -> set[str] | None:
    """Best-effort set of agent ids on disk (None = could not tell)."""
    try:
        from .config import load_config
        from .loader import load_agents_dir
        cfg = load_config()
        if data_dir_arg and Path(data_dir_arg).expanduser() != cfg.data_dir:
            return None     # a different daemon's data dir: no reliable agents dir
        defs, _errors = load_agents_dir(cfg.agents_dir)
        return {d.id for d in defs}
    except Exception:
        return None


def _fmt_ts(ts: float | None) -> str:
    if not ts:
        return "-"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="atn integration-token",
        description="Manage bearer tokens for the agent-clamped integration listener.")
    parser.add_argument("--data-dir", default=None,
                        help="daemon data dir (default: from the ATN config)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_create = sub.add_parser("create", help="mint a token bound to one agent")
    p_create.add_argument("--agent", required=True, help="agent id the token acts as")
    p_create.add_argument("--label", default="", help="human label (also a revoke handle)")
    p_create.add_argument("--force", action="store_true",
                          help="mint even if the agent is not found on disk")

    p_revoke = sub.add_parser("revoke", help="revoke by token id or label")
    p_revoke.add_argument("ident", help="token id (from `list`) or exact label")

    p_list = sub.add_parser("list", help="list tokens (never shows the secret)")
    p_list.add_argument("--json", action="store_true", help="machine-readable output")

    args = parser.parse_args(argv)
    store = IntegrationTokenStore(_data_dir(args.data_dir))

    if args.cmd == "create":
        if not is_bindable_agent_id(args.agent):
            print(f"error: {args.agent!r} is an owner sentinel or invalid id; "
                  "an integration token must name a real agent", file=sys.stderr)
            return 2
        known = _known_agent_ids(args.data_dir)
        if known is not None and args.agent not in known and not args.force:
            print(f"error: no agent '{args.agent}' in the agents dir "
                  "(use --force to mint anyway; the daemon refuses the token "
                  "until that agent exists)", file=sys.stderr)
            return 2
        token, rec = store.create(args.agent, args.label)
        print(token)
        print(f"# id={rec.token_id} agent={rec.agent_id} label={rec.label or '-'}",
              file=sys.stderr)
        print("# Shown once. Store it as a secret; only its sha256 is kept.",
              file=sys.stderr)
        return 0

    if args.cmd == "revoke":
        hit = store.revoke(args.ident)
        if not hit:
            print(f"error: no active token matches '{args.ident}'", file=sys.stderr)
            return 1
        for rec in hit:
            print(f"revoked {rec.token_id} (agent={rec.agent_id}, label={rec.label or '-'})")
        return 0

    rows = store.list()
    if args.json:
        print(json.dumps([r.public_view() for r in rows], indent=2))
        return 0
    if not rows:
        print("no integration tokens")
        return 0
    for r in rows:
        state = "active" if r.active else f"revoked {_fmt_ts(r.revoked_at)}"
        print(f"{r.token_id}  agent={r.agent_id}  label={r.label or '-'}  "
              f"created={_fmt_ts(r.created_at)}  {state}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
