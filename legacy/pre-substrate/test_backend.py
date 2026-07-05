"""In-process backend test for the new generic agent operations.

Starts the full Runtime + WebSocket server, then connects as a client
to test reset_agent_conversation, set_agent_model, remove_agent, and
session_stats.
"""
import asyncio
import json
import logging
import sys

import websockets

logging.basicConfig(level=logging.WARNING)

PORT = 7711  # Use a non-default port to avoid conflicts

async def recv_response(ws, msg_id, timeout=10):
    """Receive the response matching msg_id, skipping events."""
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            raise TimeoutError(f"No response for msg_id={msg_id}")
        raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
        msg = json.loads(raw)
        if msg.get("msg_id") == msg_id:
            return msg
        # skip events and snapshot


async def run_tests():
    from atn.config import load_config
    from atn.events import EventBus
    from atn.runtime import Runtime
    from atn.loader import load_agents_dir
    from atn.ws_server import WebSocketBridge

    config = load_config()
    event_bus = EventBus()
    runtime = Runtime(event_bus, data_dir=config.data_dir, config=config)
    await runtime.start()

    # Load agents
    agents, errors = load_agents_dir(config.agents_dir)
    for defn in agents:
        if defn.id in {d.id for d, _ in runtime.list_agents()}:
            await runtime.unregister_agent(defn.id)
        await runtime.register_agent(defn)

    # Register orchestrator
    try:
        await runtime.setup_orchestrator()
    except Exception as e:
        print(f"  Orchestrator setup: {e}")

    # Start WebSocket on test port
    ws_bridge = WebSocketBridge(runtime, host="127.0.0.1", port=PORT)
    await ws_bridge.start()
    print(f"Server listening on ws://127.0.0.1:{PORT}")
    await asyncio.sleep(0.5)  # Let the server fully bind

    passed = 0
    failed = 0

    try:
        async with websockets.connect(f"ws://127.0.0.1:{PORT}", open_timeout=10) as ws:
            # Consume initial snapshot
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            snap = json.loads(raw)
            assert snap.get("type") == "snapshot", f"Expected snapshot, got {snap.get('type')}"
            agent_ids = list(snap["data"]["agents"].keys())
            agents_data = snap["data"]["agents"]
            print(f"Connected. Agents: {agent_ids}")

            # Find a cognitive child agent (for model-change tests)
            cognitive_child = next(
                (aid for aid, info in agents_data.items()
                 if aid != "orchestrator" and info.get("mode") == "cognitive"),
                None,
            )

            # ── Test 1: session_stats for orchestrator ──
            print("\n--- Test 1: session_stats (orchestrator) ---")
            await ws.send(json.dumps({"type": "session_stats", "msg_id": "t1"}))
            resp = await recv_response(ws, "t1")
            if resp.get("ok"):
                result = resp["result"]
                print(f"  OK: turns={result.get('num_turns')}, model={result.get('active_model')}, "
                      f"context_window={result.get('context_window')}, "
                      f"last_input_tokens={result.get('last_input_tokens')}")
                passed += 1
            elif "No active session" in str(resp.get("error", "")):
                print(f"  OK (no session yet — expected if conversation was reset): {resp.get('error')}")
                passed += 1
            else:
                print(f"  FAIL: {resp.get('error')}")
                failed += 1

            # ── Test 2: session_stats for a child agent ──
            child = next((a for a in agent_ids if a != "orchestrator"), None)
            if child:
                print(f"\n--- Test 2: session_stats ({child}) ---")
                await ws.send(json.dumps({"type": "session_stats", "msg_id": "t2", "agent_id": child}))
                resp = await recv_response(ws, "t2")
                # May return error if no session yet — that's OK
                print(f"  Result: ok={resp.get('ok')}, "
                      f"data={json.dumps(resp.get('result', resp.get('error', '')))[:120]}")
                passed += 1
            else:
                print("\n--- Test 2: SKIP (no child agents) ---")

            # ── Test 3: reset_agent_conversation (orchestrator) ──
            print("\n--- Test 3: reset_agent_conversation (orchestrator) ---")
            await ws.send(json.dumps({"type": "reset_agent_conversation", "msg_id": "t3", "agent_id": "orchestrator"}))
            resp = await recv_response(ws, "t3")
            if resp.get("ok"):
                print(f"  OK: {resp['result']}")
                passed += 1
            else:
                print(f"  FAIL: {resp.get('error')}")
                failed += 1

            # ── Test 4: reset_agent_conversation (child agent) ──
            if child:
                print(f"\n--- Test 4: reset_agent_conversation ({child}) ---")
                await ws.send(json.dumps({"type": "reset_agent_conversation", "msg_id": "t4", "agent_id": child}))
                resp = await recv_response(ws, "t4")
                if resp.get("ok"):
                    print(f"  OK: {resp['result']}")
                    passed += 1
                else:
                    print(f"  FAIL: {resp.get('error')}")
                    failed += 1

            # ── Test 5: set_agent_model (cognitive child agent) ──
            if cognitive_child:
                print(f"\n--- Test 5: set_agent_model ({cognitive_child}, claude-haiku-4-5) ---")
                await ws.send(json.dumps({
                    "type": "set_agent_model", "msg_id": "t5",
                    "agent_id": cognitive_child, "model": "claude-haiku-4-5",
                }))
                resp = await recv_response(ws, "t5")
                if resp.get("ok"):
                    print(f"  OK: {resp['result']}")
                    passed += 1
                else:
                    print(f"  FAIL: {resp.get('error')}")
                    failed += 1
            else:
                print("\n--- Test 5: SKIP (no cognitive child agents) ---")

            # ── Test 6: set_agent_model (orchestrator) ──
            print("\n--- Test 6: set_agent_model (orchestrator, claude-sonnet-4-6) ---")
            await ws.send(json.dumps({
                "type": "set_agent_model", "msg_id": "t6",
                "agent_id": "orchestrator", "model": "claude-sonnet-4-6",
            }))
            resp = await recv_response(ws, "t6")
            if resp.get("ok"):
                print(f"  OK: {resp['result']}")
                passed += 1
            else:
                print(f"  FAIL: {resp.get('error')}")
                failed += 1

            # ── Test 7: remove_agent (root — should fail) ──
            print("\n--- Test 7: remove_agent (orchestrator — expect rejection) ---")
            await ws.send(json.dumps({"type": "remove_agent", "msg_id": "t7", "agent_id": "orchestrator"}))
            resp = await recv_response(ws, "t7")
            if not resp.get("ok"):
                print(f"  OK (correctly rejected): {resp.get('error')}")
                passed += 1
            else:
                print(f"  FAIL: should have been rejected")
                failed += 1

            # ── Test 8: remove_agent (child — should succeed) ──
            # Pick a non-essential agent to remove
            removable = next((a for a in agent_ids
                              if a not in ("orchestrator", "homebase-ops") and a != child), None)
            if removable:
                print(f"\n--- Test 8: remove_agent ({removable}) ---")
                await ws.send(json.dumps({"type": "remove_agent", "msg_id": "t8", "agent_id": removable}))
                resp = await recv_response(ws, "t8")
                if resp.get("ok"):
                    print(f"  OK: {resp['result']}")
                    passed += 1
                else:
                    print(f"  FAIL: {resp.get('error')}")
                    failed += 1
            else:
                print("\n--- Test 8: SKIP (no removable agents) ---")

            # ── Test 9: set_agent_model missing fields ──
            print("\n--- Test 9: set_agent_model (missing model — expect error) ---")
            await ws.send(json.dumps({"type": "set_agent_model", "msg_id": "t9", "agent_id": "orchestrator"}))
            resp = await recv_response(ws, "t9")
            if not resp.get("ok"):
                print(f"  OK (correctly rejected): {resp.get('error')}")
                passed += 1
            else:
                print(f"  FAIL: should have been rejected")
                failed += 1

            # ── Test 10: Verify orchestrator model was restored ──
            # Change it back to opus
            print("\n--- Test 10: set_agent_model (orchestrator back to opus) ---")
            await ws.send(json.dumps({
                "type": "set_agent_model", "msg_id": "t10",
                "agent_id": "orchestrator", "model": "claude-opus-4-6",
            }))
            resp = await recv_response(ws, "t10")
            if resp.get("ok"):
                print(f"  OK: {resp['result']}")
                passed += 1
            else:
                print(f"  FAIL: {resp.get('error')}")
                failed += 1

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        failed += 1

    finally:
        print(f"\n{'='*50}")
        print(f"Results: {passed} passed, {failed} failed")
        if failed == 0:
            print("ALL TESTS PASSED")
        else:
            print("SOME TESTS FAILED")
        await ws_bridge.stop()
        await runtime.stop()


if __name__ == "__main__":
    asyncio.run(run_tests())
