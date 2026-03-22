"""Script step executor — runs shell commands / scripts.

Features:
  - Subprocess with configurable timeout
  - Kill-switch monitoring (cancel_event) terminates the process
  - Previous step output piped via stdin
  - PID tracked for process-level observability
  - **Streaming**: stdout/stderr emitted line-by-line as step.output events
"""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone

from ..events import Event, EventType
from ..models import ExecutionStatus, StepDefinition, StepResult, StepType
from .base import StepContext, StepExecutor


class ScriptStepExecutor(StepExecutor):

    async def execute(
        self,
        step: StepDefinition,
        step_index: int,
        context: StepContext,
    ) -> StepResult:
        command: str = step.config.get("command", "")
        timeout: int = step.config.get("timeout", 60)

        result = StepResult(
            step_index=step_index,
            step_name=step.name or f"script_{step_index}",
            step_type=StepType.SCRIPT,
            status=ExecutionStatus.RUNNING,
            started_at=datetime.now(timezone.utc),
        )

        # ---- pre-flight kill-switch check ----
        if context.cancel_event.is_set():
            result.status = ExecutionStatus.KILLED
            result.completed_at = datetime.now(timezone.utc)
            return result

        # ---- build env ----
        env = {**os.environ, **context.env}
        env["AGENT_ID"] = context.agent_id
        env["EXECUTION_ID"] = context.execution_id
        env["STEP_INDEX"] = str(step_index)

        # Feed previous step output as stdin
        stdin_data: bytes | None = None
        if context.previous_outputs:
            last = context.previous_outputs[-1]
            if last is not None:
                stdin_data = str(last).encode()

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE if stdin_data else None,
                cwd=str(context.work_dir),
                env=env,
            )
            result.pid = process.pid

            # Write stdin and close it, then stream stdout/stderr
            if stdin_data and process.stdin:
                process.stdin.write(stdin_data)
                await process.stdin.drain()
                process.stdin.close()

            try:
                stdout, stderr = await asyncio.wait_for(
                    self._stream_with_cancel(
                        process, context, step_index,
                        step.name or f"script_{step_index}",
                    ),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                await self._terminate(process)
                result.status = ExecutionStatus.FAILED
                result.error = f"Timed out after {timeout}s"
                result.completed_at = datetime.now(timezone.utc)
                return result

            if context.cancel_event.is_set():
                result.status = ExecutionStatus.KILLED
                result.completed_at = datetime.now(timezone.utc)
                return result

            result.output = {
                "stdout": stdout.strip(),
                "stderr": stderr.strip(),
                "exit_code": process.returncode,
            }

            if process.returncode == 0:
                result.status = ExecutionStatus.COMPLETED
            else:
                result.status = ExecutionStatus.FAILED
                result.error = (
                    f"Exit code {process.returncode}"
                    + (f": {result.output['stderr'][:200]}" if result.output["stderr"] else "")
                )

        except Exception as exc:
            result.status = ExecutionStatus.FAILED
            result.error = str(exc)

        result.completed_at = datetime.now(timezone.utc)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _stream_with_cancel(
        self,
        process: asyncio.subprocess.Process,
        context: StepContext,
        step_index: int,
        step_name: str,
    ) -> tuple[str, str]:
        """Read stdout/stderr line by line, emitting step.output events.

        Returns the full accumulated (stdout, stderr) strings on completion.
        """
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        async def _read_stream(
            stream: asyncio.StreamReader | None,
            channel: str,
            acc: list[str],
        ) -> None:
            if stream is None:
                return
            while True:
                line_bytes = await stream.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode(errors="replace")
                acc.append(line)
                await self._emit_output(
                    context, step_index, step_name, channel, line.rstrip("\n\r"),
                )

        async def _watch_cancel() -> None:
            while not context.cancel_event.is_set():
                await asyncio.sleep(0.1)
            await self._terminate(process)

        watcher = asyncio.create_task(_watch_cancel())
        try:
            await asyncio.gather(
                _read_stream(process.stdout, "stdout", stdout_lines),
                _read_stream(process.stderr, "stderr", stderr_lines),
            )
            await process.wait()
        finally:
            watcher.cancel()
            try:
                await watcher
            except asyncio.CancelledError:
                pass

        return "".join(stdout_lines), "".join(stderr_lines)

    @staticmethod
    async def _emit_output(
        context: StepContext,
        step_index: int,
        step_name: str,
        channel: str,
        content: str,
    ) -> None:
        """Emit a step.output event if the EventBus is available."""
        if not context.event_bus or not content:
            return
        await context.event_bus.emit(Event(
            type=EventType.STEP_OUTPUT,
            source=context.agent_id,
            data={
                "agent_id": context.agent_id,
                "execution_id": context.execution_id,
                "step_index": step_index,
                "step_name": step_name,
                "channel": channel,
                "content": content,
            },
        ))

    @staticmethod
    async def _terminate(process: asyncio.subprocess.Process) -> None:
        """Best-effort terminate then kill."""
        try:
            process.terminate()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=2)
        except asyncio.TimeoutError:
            try:
                process.kill()
            except ProcessLookupError:
                pass
