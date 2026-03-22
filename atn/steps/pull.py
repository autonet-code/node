"""Pull step executor — read another agent's output store.

Non-triggering: does NOT wake the source agent.  Pure data access.
Useful when an agent just needs another agent's latest result without
requiring a fresh run.

Config keys:
  source      (str, required)  Agent ID whose output to read.
  max_age     (int, optional)  Maximum acceptable age in seconds.
              If the output is older, the step fails.  Omit for no age check.
"""
from __future__ import annotations

from datetime import datetime, timezone

from ..models import ExecutionStatus, StepDefinition, StepResult, StepType
from .base import StepContext, StepExecutor


class PullStepExecutor(StepExecutor):

    async def execute(
        self,
        step: StepDefinition,
        step_index: int,
        context: StepContext,
    ) -> StepResult:
        source: str = step.config["source"]
        max_age: int | None = step.config.get("max_age")

        result = StepResult(
            step_index=step_index,
            step_name=step.name or f"pull_{step_index}",
            step_type=StepType.PULL,
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

        output = context.output_store.read(source)

        if output is None:
            result.status = ExecutionStatus.FAILED
            result.error = f"No output available from agent '{source}'"
            result.completed_at = datetime.now(timezone.utc)
            return result

        # Age check
        if max_age is not None:
            age = (datetime.now(timezone.utc) - output.timestamp).total_seconds()
            if age > max_age:
                result.status = ExecutionStatus.FAILED
                result.error = (
                    f"Output from '{source}' is {age:.0f}s old (max_age={max_age}s)"
                )
                result.completed_at = datetime.now(timezone.utc)
                return result

        result.status = ExecutionStatus.COMPLETED
        result.output = {
            "source": source,
            "data": output.data,
            "timestamp": output.timestamp.isoformat(),
            "execution_id": output.execution_id,
        }
        result.completed_at = datetime.now(timezone.utc)
        return result
