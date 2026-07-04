"""Kill switches, interrupts, and message injection for running agents."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .execution_engine import ExecutionEngine
    from .provider_manager import ProviderManager
    from .agent_supervisor import AgentSupervisor

log = logging.getLogger(__name__)


class ExecutionControl:
    """Kill execution, agent, or all; interrupt delegates; inject messages.

    Dual-mode (P3): each kill front-end keeps its EXACT existing asyncio
    task-cancel path when the flag is OFF (or no worker is registered for the
    target). When the flag is ON *and* the target agent has a live supervised
    worker, the front-end routes to the ``AgentSupervisor`` kill ladder instead
    (cooperative IPC → SIGTERM/Job → SIGKILL). Signatures are UNCHANGED so every
    existing caller works untouched.
    """

    def __init__(
        self,
        engine: ExecutionEngine,
        provider_manager: ProviderManager,
        supervisor: "Optional[AgentSupervisor]" = None,
    ) -> None:
        self.engine = engine
        self.provider_manager = provider_manager
        # The per-Runtime worker supervisor (P3). May be None in bare-engine
        # tests that construct ExecutionControl directly; treated as "no workers".
        self.supervisor = supervisor

    # ------------------------------------------------------------------
    # Dual-mode helpers
    # ------------------------------------------------------------------

    def _worker_isolation_enabled(self) -> bool:
        """Flag state, read off the engine's config (single source of truth)."""
        cfg = getattr(self.engine, "_config", None)
        wi = getattr(cfg, "worker_isolation", None)
        return bool(getattr(wi, "enabled", False))

    def _supervised(self, agent_id: str) -> bool:
        """True iff the flag is ON and a live worker exists for this agent, i.e.
        the kill must go through the supervisor rather than the in-process task."""
        if not self._worker_isolation_enabled() or self.supervisor is None:
            return False
        return self.supervisor.is_supervised(agent_id)

    # ------------------------------------------------------------------
    # Kill switches
    # ------------------------------------------------------------------

    async def kill_execution(self, execution_id: str) -> bool:
        # Dual-mode: if this execution belongs to a supervised worker, route the
        # kill through the supervisor ladder (cooperative IPC → SIGTERM/Job →
        # SIGKILL). Otherwise fall through to the exact in-process path below.
        rec = self.engine._executions.get(execution_id)
        if rec is not None and self._supervised(rec.agent_id):
            return await self.supervisor.kill(rec.agent_id, reason="kill_execution")

        cancel = self.engine._cancels.get(execution_id)
        if not cancel:
            return False

        hook = self.engine._interrupt_hooks.get(execution_id)
        if hook:
            try:
                await hook()
            except Exception:
                log.debug("Interrupt hook error for %s", execution_id, exc_info=True)

        cancel.set()
        task = self.engine._tasks.get(execution_id)
        if task:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=10)
            except asyncio.TimeoutError:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            except (asyncio.CancelledError, Exception):
                pass
        return True

    async def kill_agent(self, agent_id: str) -> int:
        # Dual-mode: a supervised worker is killed once via the ladder (which
        # also reaps its logical delegate subtree leaf-first). We still count the
        # in-process executions for this agent so the return value is unchanged.
        eids = [eid for eid, rec in self.engine._executions.items() if rec.agent_id == agent_id]
        if self._supervised(agent_id):
            await self.supervisor.kill(agent_id, reason="kill_agent")
            return len(eids) or 1
        for eid in eids:
            await self.kill_execution(eid)
        # §11 kill ladder: a cooperative interrupt cannot stop a spinning/wedged
        # bridge SDK loop, so a "killed" agent must not leave the bun subprocess
        # running. If this agent's provider owns a subprocess, verify it is dead
        # (grace window then hard kill of the tracked PID).
        await self._verify_provider_subprocess_killed(agent_id)
        return len(eids)

    async def _verify_provider_subprocess_killed(self, agent_id: str) -> None:
        """Escalate a cooperative interrupt to a verified subprocess kill for
        providers that host an external process (the bridge/bun subprocess).

        No-op for in-process providers (no ``terminate_subprocess``) and for
        supervised workers (handled by the supervisor ladder)."""
        provider = self.provider_manager._active_providers.get(agent_id)
        if provider is None:
            return
        terminate = getattr(provider, "terminate_subprocess", None)
        if not callable(terminate):
            return
        try:
            killed = await terminate(reason=f"kill_agent:{agent_id}")
            if not killed:
                log.error(
                    "kill_agent(%s): provider subprocess could not be confirmed dead",
                    agent_id,
                )
        except Exception:
            log.warning("kill_agent(%s): terminate_subprocess raised", agent_id, exc_info=True)

    async def kill_all(self) -> int:
        # Dual-mode: kill every supervised worker via the ladder first (roots
        # last, leaf-first per subtree). This is additive to — not a replacement
        # for — the in-process teardown below, which still handles any agents
        # NOT running in a worker (and is a full no-op when there are no
        # in-process tasks). With the flag OFF the supervisor holds no workers,
        # so this call returns 0 and behavior is exactly as before.
        worker_killed = 0
        if self._worker_isolation_enabled() and self.supervisor is not None:
            try:
                worker_killed = await self.supervisor.kill_all(reason="kill_all")
            except Exception:
                log.warning("kill_all: supervisor teardown raised", exc_info=True)

        for hook in list(self.engine._interrupt_hooks.values()):
            try:
                await hook()
            except Exception:
                pass
        for cancel in self.engine._cancels.values():
            cancel.set()
        tasks = list(self.engine._tasks.values())
        if not tasks:
            return worker_killed
        # Bound the wait so a misbehaving task can't block shutdown forever.
        # Cooperative interrupts get a short window; tasks still running are
        # cancelled hard and we fall through to subprocess teardown.
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=5,
            )
        except asyncio.TimeoutError:
            log.warning(
                "kill_all: %d task(s) did not exit within 5s — cancelling hard",
                sum(1 for t in tasks if not t.done()),
            )
            for t in tasks:
                if not t.done():
                    t.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=2,
                )
            except asyncio.TimeoutError:
                log.warning("kill_all: tasks still alive after hard cancel — abandoning")
        return len(tasks) + worker_killed

    # ------------------------------------------------------------------
    # Interrupt hooks
    # ------------------------------------------------------------------

    def register_interrupt_hook(self, execution_id: str, hook: Callable) -> None:
        self.engine._interrupt_hooks[execution_id] = hook

    def unregister_interrupt_hook(self, execution_id: str) -> None:
        self.engine._interrupt_hooks.pop(execution_id, None)

    # ------------------------------------------------------------------
    # Message injection & interrupt
    # ------------------------------------------------------------------

    async def send_delegate_message(self, agent_id: str, content: str) -> "str | bool":
        """Inject a steering message into a running delegate.

        Returns a truthy status string when the message was handled:
          - "injected"        — consumed by the SDK/provider loop this turn.
          - "inbox_fallback"  — the live injection raced turn completion (or the
                                provider dropped it); re-posted to the agent's
                                inbox (HIGH WORK) so the next run picks it up.
        Returns False only when there is no provider AND no agent to fall back
        to (caller treats that as "not running").

        §5 delivery-ack (finding 8): a bridge injection is only "delivered" once
        the bridge acks it consumed. If the injection is not consumed within a
        short window, we re-post the content to the inbox so nothing is silently
        dropped — the daemon must never report ``injected`` without a real
        delivery or a fallback guarantee behind it.
        """
        # Dual-mode (C1): a supervised worker's provider object lives IN THE
        # WORKER, not in the daemon's _active_providers map. Route the injection
        # over the IPC "send_user_message" cmd and AWAIT the worker's consumption
        # ack (the worker runs the same injection_consumed poll the in-process
        # path runs, because the provider is local to it). The worker returns
        # {"status": "injected"} when the loop consumed it, or
        # {"status": "inbox_fallback", "content": <reclaimed>} when it raced turn
        # end / was dropped — in which case the DAEMON does the inbox re-post
        # here (it owns the InboxManager; the worker cannot). This gives the
        # caller the SAME "injected"/"inbox_fallback" semantics as in-process.
        if self._supervised(agent_id):
            worker = self.supervisor.get(agent_id)
            channel = getattr(getattr(worker, "handle", None), "channel", None)
            if channel is not None:
                try:
                    ack = await channel.send_cmd_await(
                        "send_user_message", {"content": content})
                except Exception:
                    log.debug("send_user_message cmd failed for %s", agent_id, exc_info=True)
                    return self._inbox_fallback(agent_id, content) or "inbox_fallback"
                status = (ack or {}).get("status") if isinstance(ack, dict) else None
                if status == "injected":
                    return "injected"
                # inbox_fallback (or an unexpected ack): re-post the reclaimed
                # content (worker returns the exact text it couldn't deliver).
                pending = (ack or {}).get("content") if isinstance(ack, dict) else None
                return self._inbox_fallback(agent_id, pending or content) or "inbox_fallback"
            return self._inbox_fallback(agent_id, content) or "inbox_fallback"

        provider = self.provider_manager._active_providers.get(agent_id)
        if provider is None:
            # No live provider — deliver via the inbox if the agent exists,
            # else report "not running".
            return self._inbox_fallback(agent_id, content) or False

        inj_id = await provider.send_user_message(content)

        # Providers whose send_user_message doesn't report an id (older / non-
        # bridge) can't be verified — treat as injected (prior behavior).
        if not isinstance(inj_id, str) or not inj_id:
            return "injected"

        # Wait briefly for the consumption ack. The bridge acks as soon as the
        # SDK loop dequeues the message; if the turn was already ending it will
        # never be consumed and we fall back.
        consumed = await self._await_injection_consumed(provider, inj_id)
        if consumed:
            return "injected"

        # Unconsumed — reclaim the content and re-post it to the inbox so the
        # next run picks it up. Nothing is lost.
        take = getattr(provider, "take_unconsumed_injection", None)
        pending = take(inj_id) if callable(take) else content
        if pending is None:
            # Consumed in the race between the check and the take — good.
            return "injected"
        log.info(
            "delegate_message to %s not consumed (id=%s) — falling back to inbox",
            agent_id, inj_id,
        )
        return self._inbox_fallback(agent_id, pending) or "inbox_fallback"

    async def _await_injection_consumed(
        self, provider: Any, inj_id: str, timeout: float = 2.0, interval: float = 0.05,
    ) -> bool:
        """Poll ``provider.injection_consumed(inj_id)`` up to ``timeout`` s."""
        checker = getattr(provider, "injection_consumed", None)
        if not callable(checker):
            return True  # can't verify — assume delivered
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            if checker(inj_id):
                return True
            await asyncio.sleep(interval)
        return checker(inj_id)

    def _inbox_fallback(self, agent_id: str, content: str) -> str:
        """Re-post steering content to an agent's inbox as a HIGH WORK message.

        Mirrors orchestrator/tools._post_message's InboxMessage shape. Returns
        "inbox_fallback" on success, "" if the agent is unknown / no inbox."""
        from ..models import InboxMessage, MessagePriority, MessageType
        inbox = getattr(self.engine, "inbox", None)
        registry = getattr(self.engine, "registry", None)
        if inbox is None:
            return ""
        # If we can tell the agent doesn't exist, don't stage an orphan message.
        if registry is not None:
            known = getattr(registry, "_agents", None)
            if isinstance(known, dict) and agent_id not in known:
                return ""
        inbox.post(InboxMessage(
            id=InboxMessage.generate_id(),
            source="orchestrator",
            target=agent_id,
            type=MessageType.WORK,
            priority=MessagePriority.HIGH,
            data={"instruction": content},
        ))
        return "inbox_fallback"

    async def interrupt_delegate(self, agent_id: str) -> bool:
        # Dual-mode: for a supervised worker, a cooperative interrupt is an IPC
        # "interrupt" cmd (the provider loop lives in the worker, not here).
        # kill is NOT a message; interrupt IS cooperative and droppable-safe.
        if self._supervised(agent_id):
            worker = self.supervisor.get(agent_id)
            channel = getattr(getattr(worker, "handle", None), "channel", None)
            if channel is not None:
                try:
                    await channel.send_cmd("interrupt", {})
                    return True
                except Exception:
                    log.debug("interrupt cmd send failed for %s", agent_id, exc_info=True)
                    return False
            return False
        provider = self.provider_manager._active_providers.get(agent_id)
        if provider is None:
            return False
        await provider.interrupt()
        return True

    async def interrupt_orchestrator(self) -> bool:
        from ..orchestrator import ORCHESTRATOR_ID
        return await self.interrupt_delegate(ORCHESTRATOR_ID)
