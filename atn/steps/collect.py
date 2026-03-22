"""Collect step executor — await the result of a prior async message step.

Used after an async message step to retrieve the target agent's output
once it's available.  Polls the target's output store for a result newer
than the timestamp recorded by the message step.

Config keys:
  from_step   (int, required)  Index of the earlier async message step whose
              result we're collecting.
  timeout     (int)  Seconds to wait.  Default: 120.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from ..models import ExecutionStatus, StepDefinition, StepResult, StepType
from .base import StepContext, StepExecutor


class CollectStepExecutor(StepExecutor):

    async def execute(
        self,
        step: StepDefinition,
        step_index: int,
        context: StepContext,
    ) -> StepResult:
        from_step: int = step.config["from_step"]
        timeout: int = step.config.get("timeout", 120)

        result = StepResult(
            step_index=step_index,
            step_name=step.name or f"collect_{step_index}",
            step_type=StepType.COLLECT,
            status=ExecutionStatus.RUNNING,
            started_at=datetime.now(timezone.utc),
        )

        if context.cancel_event.is_set():
            result.status = ExecutionStatus.KILLED
            result.completed_at = datetime.now(timezone.utc)
            return result

        if context.output_store is None:
            result.status = ExecutionStatus.FAILED
            result.error = "No output_store in context"
            result.completed_at = datetime.now(timezone.utc)
            return result

        # Retrieve the async message step's output to find target and timestamp.
        if from_step >= len(context.previous_outputs):
            result.status = ExecutionStatus.FAILED
            result.error = f"from_step={from_step} but only {len(context.previous_outputs)} previous outputs"
            result.completed_at = datetime.now(timezone.utc)
            return result

        msg_output = context.previous_outputs[from_step]
        if not isinstance(msg_output, dict) or "target" not in msg_output:
            result.status = ExecutionStatus.FAILED
            result.error = f"Step {from_step} output is not an async message handle: {msg_output}"
            result.completed_at = datetime.now(timezone.utc)
            return result

        target = msg_output["target"]
        posted_at = datetime.fromisoformat(msg_output["posted_at"])

        # Poll for result
        try:
            data = await self._wait_for_output(
                target, posted_at, timeout, context,
            )
            result.status = ExecutionStatus.COMPLETED
            result.output = data
        except TimeoutError as exc:
            result.status = ExecutionStatus.FAILED
            result.error = str(exc)

        result.completed_at = datetime.now(timezone.utc)
        return result

    @staticmethod
    async def _wait_for_output(
        target: str,
        posted_at: datetime,
        timeout: int,
        context: StepContext,
    ) -> dict:
        store = context.output_store
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            if context.cancel_event.is_set():
                raise TimeoutError("Killed while waiting")
            output = store.read(target)
            if output and output.timestamp > posted_at:
                return {
                    "source": output.agent_id,
                    "data": output.data,
                    "execution_id": output.execution_id,
                }
            await asyncio.sleep(0.3)
        raise TimeoutError(f"Timed out collecting result from '{target}' after {timeout}s")
