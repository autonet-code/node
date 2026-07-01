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
        return len(eids)

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

    async def send_delegate_message(self, agent_id: str, content: str) -> bool:
        # Dual-mode (C1): a supervised worker's provider object lives IN THE
        # WORKER, not in the daemon's _active_providers map. Route the injection
        # over the IPC "send_user_message" cmd (the worker's on_cmd handles it:
        # live-inject if the provider supports it, else stage for the inbox
        # fallback). Mirrors interrupt_delegate's worker branch.
        if self._supervised(agent_id):
            worker = self.supervisor.get(agent_id)
            channel = getattr(getattr(worker, "handle", None), "channel", None)
            if channel is not None:
                try:
                    await channel.send_cmd("send_user_message", {"content": content})
                    return True
                except Exception:
                    log.debug("send_user_message cmd failed for %s", agent_id, exc_info=True)
                    return False
            return False
        provider = self.provider_manager._active_providers.get(agent_id)
        if provider is None:
            return False
        await provider.send_user_message(content)
        return True

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
