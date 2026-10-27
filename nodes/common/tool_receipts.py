"""Signed tool-use receipts (docs/tool_substrate.md, Decision 2026-10-27).

An attested ``tool_used`` event is what credits a tool's author, weighted by
the calling household's REP, so it must come from the agent it names: the
agent's own key signs it (EIP-191), and the close counts only receipts whose
signature recovers to their ``author_agent``. Without this any daemon could
publish receipts claiming any REP holder used its tools.

The signed message names the jurisdiction (the gossip domain, e.g.
``31337:0x<dao>``), so a receipt cannot be replayed into another one, and
the receipt's ``seq``, so the close can count each receipt once.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

RECEIPT_VERSION = "autonet:tool_used:v1"


def receipt_message(domain: str, event: Dict[str, Any]) -> str:
    """The exact text an agent signs for one attested receipt."""
    own = event.get("own_share")
    return "|".join([
        RECEIPT_VERSION,
        str(domain).lower(),
        str(event.get("author_agent") or "").lower(),
        str(event.get("manifest_digest") or "").lower(),
        str(int(event.get("seq") or 0)),
        "1" if event.get("ok", True) else "0",
        str(event.get("receipt_digest") or ""),
        "" if own is None else repr(float(own)),
    ])


def sign_receipt(private_key: str, domain: str, event: Dict[str, Any]) -> str:
    from eth_account import Account
    from eth_account.messages import encode_defunct
    signed = Account.sign_message(encode_defunct(text=receipt_message(domain, event)),
                                  private_key)
    return "0x" + signed.signature.hex().removeprefix("0x")


def receipt_signer_ok(domain: str, event: Dict[str, Any]) -> bool:
    """True when the receipt's signature recovers to its author_agent."""
    sig = str(event.get("agent_sig") or "")
    agent = str(event.get("author_agent") or "")
    if not sig or not agent.startswith("0x"):
        return False
    try:
        from eth_account import Account
        from eth_account.messages import encode_defunct
        who = Account.recover_message(encode_defunct(text=receipt_message(domain, event)),
                                      signature=sig)
    except Exception:
        return False
    return who.lower() == agent.lower()


def verified_receipts(events: Iterable[Dict[str, Any]],
                      domain: Optional[str]) -> List[Dict[str, Any]]:
    """``events`` with every attested ``tool_used`` that is unsigned,
    wrongly signed or a repeat of an earlier (author_agent, seq) dropped.
    Other events pass through unchanged. ``domain`` None (no chain, tests)
    keeps the old behaviour: nothing is dropped."""
    if domain is None:
        return list(events)
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for ev in events:
        if ev.get("kind") == "tool_used" and ev.get("attested"):
            key = (str(ev.get("author_agent") or "").lower(), int(ev.get("seq") or 0))
            if key in seen or not receipt_signer_ok(domain, ev):
                continue
            seen.add(key)
        out.append(ev)
    return out
