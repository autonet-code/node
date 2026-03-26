"""Kill switches, interrupts, and message injection for running agents."""
from __future__ import annotations

import asyncio
import logging
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .execution_engine import ExecutionEngine
    from .provider_manager import ProviderManager

log = logging.getLogger(__name__)


class ExecutionControl:
    """Kill execution, agent, or all; interrupt delegates; inject messages."""

    def __init__(
        self,
        engine: ExecutionEngine,
        provider_manager: ProviderManager,
    ) -> None:
        self.engine = engine
        self.provider_manager = provider_manager

    # ------------------------------------------------------------------
    # Kill switches
    # ------------------------------------------------------------------

    async def kill_execution(self, execution_id: str) -> bool:
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
        eids = [eid for eid, rec in self.engine._executions.items() if rec.agent_id == agent_id]
        for eid in eids:
            await self.kill_execution(eid)
        return len(eids)

    async def kill_all(self) -> int:
        for hook in list(self.engine._interrupt_hooks.values()):
            try:
                await hook()
            except Exception:
                pass
        for cancel in self.engine._cancels.values():
            cancel.set()
        tasks = list(self.engine._tasks.values())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        return len(tasks)

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
        provider = self.provider_manager._active_providers.get(agent_id)
        if provider is None:
            return False
        await provider.send_user_message(content)
        return True

    async def interrupt_delegate(self, agent_id: str) -> bool:
        provider = self.provider_manager._active_providers.get(agent_id)
        if provider is None:
            return False
        await provider.interrupt()
        return True

    async def interrupt_orchestrator(self) -> bool:
        from ..orchestrator import ORCHESTRATOR_ID
        return await self.interrupt_delegate(ORCHESTRATOR_ID)
