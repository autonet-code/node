"""Cross-machine sponsor/dependent inference smoke (Phase 8).

Exercises the REAL libp2p `/rpb/inference/1.0.0` round trip across two
machines (or two hosts on a ZeroTier mesh), proving the work-AI flow:

  - Sponsor side: starts an AutonetHost, registers the Phase-8 sponsor
    inference handler backed by a real SponsorBindingStore + a pluggable
    provider, advertises is_sponsor with its agent address.
  - Dependent side: starts an AutonetHost, dials the sponsor, then issues
    an inference request carrying its agent_address. Asserts authorize +
    meter behaviour.

Driven entirely by env vars so the same file runs on both ends.

SPONSOR (server):
    INF_ROLE=sponsor \
    INF_LISTEN_PORT=4002 \
    INF_SPONSOR_ADDR=0xSPONSOR... \
    INF_DEPENDENT_ADDR=0xDEPENDENT...  (the bound dependent it will serve) \
    INF_BUDGET=200 \
    INF_PROVIDER=echo  (echo|anthropic|openai|...) \
    INF_MODEL=test-model \
    python -m scripts.cross_machine_inference_smoke

DEPENDENT (dialer):
    INF_ROLE=dependent \
    INF_LISTEN_PORT=4002 \
    INF_AGENT_ADDR=0xDEPENDENT...  (this dependent's own address) \
    INF_SPONSOR_ADDR=0xSPONSOR...  (target sponsor) \
    INF_MODEL=test-model \
    INF_BOOTSTRAP=/ip4/<sponsor-ZT>/tcp/4002/p2p/<sponsor-peer-id> \
    python -m scripts.cross_machine_inference_smoke

The dependent prints PASS/FAIL lines the harness can grep.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if (_REPO_ROOT / "nodes" / "__init__.py").exists():
    sys.path.insert(0, str(_REPO_ROOT))

import trio  # noqa: E402


# ---------------------------------------------------------------------------
# Minimal provider shims (no paid key needed for the dummy tier)
# ---------------------------------------------------------------------------

class _EchoProvider:
    """Returns a deterministic completion echoing the last user message.
    Reports a fixed token usage so metering is observable."""
    def __init__(self, in_tok=12, out_tok=8):
        self._in, self._out = in_tok, out_tok

    async def send(self, *, messages, system="", model="", max_tokens=4096,
                   tools=None, temperature=0.0):
        from atn.providers.base import ProviderResponse, Usage
        last = ""
        for m in reversed(messages or []):
            if m.get("role") == "user":
                last = str(m.get("content", ""))
                break
        return ProviderResponse(
            text=f"echo: {last}",
            tool_calls=None,
            usage=Usage(input_tokens=self._in, output_tokens=self._out),
            model=model or "echo-model",
            stop_reason="end_turn",
        )


def _resolve_provider(name: str, model: str):
    """Resolve a real provider from creds, or the echo stub for 'echo'."""
    if name == "echo" or not name:
        return _EchoProvider()
    # Real provider via the daemon's credential store.
    from atn.credentials import CredentialStore
    creds = CredentialStore()
    api_key = creds.load(f"provider_{name}").get("api_key", "")
    if name == "anthropic" and api_key:
        from atn.providers.anthropic import AnthropicProvider
        return AnthropicProvider(api_key=api_key, default_model=model)
    if name in ("openai", "gemini") and api_key:
        from atn.providers.openai_compat import OpenAICompatibleProvider
        base = {
            "openai": "https://api.openai.com/v1",
            "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
        }[name]
        return OpenAICompatibleProvider(name=f"smoke-{name}", base_url=base,
                                        api_key=api_key, default_model=model)
    print(f"[inf] WARNING: provider '{name}' unavailable, falling back to echo", flush=True)
    return _EchoProvider()


# ---------------------------------------------------------------------------
# Sponsor handler (mirrors AutonetBridge._create_sponsor_handler, standalone)
# ---------------------------------------------------------------------------

def _make_sponsor_handler(store, provider, sponsor_model: str):
    from atn.providers.base import ToolDefinition

    async def _handle(request: dict) -> dict:
        dependent = request.get("agent_address", "")
        if not dependent:
            return {"error": "Missing agent_address — sponsor only serves bound dependents"}
        binding = store.get(dependent)
        if binding is None:
            return {"error": "not an authorized dependent"}
        if not binding.unlimited and binding.remaining() <= 0:
            return {"error": "budget exhausted"}

        model = request.get("model", sponsor_model or "")
        messages = request.get("messages", [])
        system = request.get("system", "")
        tools_raw = request.get("tools", [])
        tools = None
        if tools_raw:
            tools = [ToolDefinition(name=t.get("name", ""),
                                    description=t.get("description", ""),
                                    input_schema=t.get("input_schema", {}))
                     for t in tools_raw]
        try:
            resp = await provider.send(
                messages=messages, system=system, model=model,
                max_tokens=request.get("max_tokens", 1024), tools=tools,
                temperature=request.get("temperature", 0.0),
            )
            in_tok = resp.usage.input_tokens if resp.usage else 0
            out_tok = resp.usage.output_tokens if resp.usage else 0
            store.record_spend(dependent, in_tok + out_tok)
            return {
                "text": resp.text or "",
                "model": resp.model or model,
                "stop_reason": resp.stop_reason or "end_turn",
                "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
                "remaining_budget_tokens": store.remaining(dependent),
            }
        except Exception as e:
            return {"error": f"Sponsor inference failed: {e}"}

    return _handle


def _run_sponsor():
    from atn.sponsor_bindings import SponsorBindingStore
    from nodes.common.p2p import AutonetHost, NodeCapability

    listen_port = int(os.environ.get("INF_LISTEN_PORT", "4002"))
    listen_host = os.environ.get("INF_LISTEN_HOST", "0.0.0.0")
    sponsor_addr = os.environ["INF_SPONSOR_ADDR"]
    dependent_addr = os.environ["INF_DEPENDENT_ADDR"]
    budget = int(os.environ.get("INF_BUDGET", "200"))
    provider_name = os.environ.get("INF_PROVIDER", "echo")
    model = os.environ.get("INF_MODEL", "test-model")
    data_dir = Path(os.environ.get("INF_DATA", "/tmp/inf_sponsor")).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)

    store = SponsorBindingStore(data_dir)
    store.add(dependent_addr, budget_tokens=budget, label="cross-machine smoke")
    provider = _resolve_provider(provider_name, model)
    handler = _make_sponsor_handler(store, provider, model)

    cap = NodeCapability(peer_id="", node_id="sponsor")
    # Advertise this sponsor agent so the dependent can target by address.
    cap.agents = [{"address": sponsor_addr, "name": "sponsor", "model": model,
                   "is_sponsor": True, "is_root": True, "parent_address": "",
                   "registered_on_chain": False, "agent_type": "sponsor"}]
    host = AutonetHost(node_id="sponsor", listen_port=listen_port,
                       listen_host=listen_host, bootstrap_peers=[], capability=cap)
    host.set_inference_handler(handler)

    async def _main():
        async with host.run():
            print(f"[inf] role=sponsor peer_id={host.peer_id}", flush=True)
            print(f"[inf] addrs={host.addrs}", flush=True)
            print(f"[inf] bound dependent={dependent_addr} budget={budget} provider={provider_name}", flush=True)
            print("[inf] SPONSOR_READY", flush=True)
            last = -1
            while True:
                await trio.sleep(3)
                b = store.get(dependent_addr)
                if b and b.spent_tokens != last:
                    last = b.spent_tokens
                    print(f"[inf] spend update: spent={b.spent_tokens} remaining={b.remaining()}", flush=True)
    trio.run(_main)


def _run_dependent():
    from nodes.common.p2p import AutonetHost, NodeCapability

    listen_port = int(os.environ.get("INF_LISTEN_PORT", "4002"))
    listen_host = os.environ.get("INF_LISTEN_HOST", "0.0.0.0")
    agent_addr = os.environ["INF_AGENT_ADDR"]
    sponsor_addr = os.environ.get("INF_SPONSOR_ADDR", "")
    model = os.environ.get("INF_MODEL", "test-model")
    bootstrap = os.environ["INF_BOOTSTRAP"].strip()
    n_calls = int(os.environ.get("INF_CALLS", "3"))

    cap = NodeCapability(peer_id="", node_id="dependent")
    host = AutonetHost(node_id="dependent", listen_port=listen_port,
                       listen_host=listen_host, bootstrap_peers=[bootstrap], capability=cap)

    async def _main():
        async with host.run():
            print(f"[inf] role=dependent peer_id={host.peer_id}", flush=True)
            await trio.sleep(3)
            ok = await host.connect_to_peer(bootstrap)
            print(f"[inf] dialed sponsor ok={ok}", flush=True)
            # Extract target peer id from the bootstrap multiaddr and convert
            # to a libp2p ID — host.new_stream resolves the peerstore by ID,
            # not by the base58 string.
            from libp2p.peer.id import ID
            target_peer = ID.from_base58(bootstrap.split("/p2p/")[-1])
            await trio.sleep(2)

            results = []
            for i in range(n_calls):
                req = {
                    "messages": [{"role": "user", "content": f"hello {i}"}],
                    "system": "", "model": model, "max_tokens": 256,
                    "temperature": 0.0, "agent_address": agent_addr,
                    "via_rpb": True,
                }
                try:
                    resp = await host.request_inference(target_peer, req)
                    rem = resp.get("remaining_budget_tokens")
                    print(f"[inf] call {i}: text={resp.get('text','')[:40]!r} remaining={rem}", flush=True)
                    results.append(("ok", resp))
                except Exception as e:
                    print(f"[inf] call {i}: REJECTED {e}", flush=True)
                    results.append(("err", str(e)))
                await trio.sleep(1)

            # Now prove authorization: an UNBOUND address must be rejected.
            req_unbound = dict(req)
            req_unbound["agent_address"] = "0xUNBOUND0000000000000000000000000000dead"
            try:
                await host.request_inference(target_peer, req_unbound)
                print("[inf] UNBOUND: unexpectedly served — FAIL", flush=True)
                unbound_rejected = False
            except Exception as e:
                print(f"[inf] UNBOUND rejected as expected: {e}", flush=True)
                unbound_rejected = "not an authorized dependent" in str(e)

            served = sum(1 for s, _ in results if s == "ok")
            print(f"[inf] SUMMARY served={served}/{n_calls} unbound_rejected={unbound_rejected}", flush=True)
            if served >= 1 and unbound_rejected:
                print("[inf] RESULT PASS", flush=True)
            else:
                print("[inf] RESULT FAIL", flush=True)
    trio.run(_main)


def main():
    role = os.environ.get("INF_ROLE", "").lower()
    if role == "sponsor":
        _run_sponsor()
    elif role == "dependent":
        _run_dependent()
    else:
        print("Set INF_ROLE=sponsor or INF_ROLE=dependent", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
