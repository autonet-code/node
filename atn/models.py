"""Core data models for the agent framework.

These are plain dataclasses — no ORM, no Pydantic (yet).
Keep them serialisable to JSON for persistence and wire transport.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class StepType(Enum):
    SCRIPT = "script"           # deterministic shell/script execution
    COGNITIVE = "cognitive"     # LLM reasoning (future)
    MESSAGE = "message"         # post to another agent's inbox
    PULL = "pull"               # read another agent's output store
    COLLECT = "collect"         # await result of prior async message step


class AgentStatus(Enum):
    REGISTERED = "registered"   # defined but not active
    ACTIVE = "active"           # armed — scheduler + inbox triggers honoured
    RUNNING = "running"         # at least one execution in progress
    STOPPED = "stopped"         # manually deactivated
    ERROR = "error"             # last execution failed


class ExecutionStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"


class MessagePriority(Enum):
    LOW = "low"                 # pick up whenever
    NORMAL = "normal"           # process on next run
    HIGH = "high"               # wake up if idle
    URGENT = "urgent"           # wake up immediately


class MessageType(Enum):
    TRIGGER = "trigger"         # "run now"
    WORK = "work"               # data / task to process
    INFO = "info"               # informational, no action required
    ALERT = "alert"             # something needs attention


class OnboardingStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class TaskStatus(Enum):
    PROPOSED = "proposed"       # orchestrator suggested it
    APPROVED = "approved"       # user approved it
    ACTIVE = "active"           # being worked on (agent running or calendar event created)
    COMPLETED = "completed"
    REJECTED = "rejected"       # user rejected the proposal


class TaskType(Enum):
    AUTOMATION = "automation"   # create/run an agent
    CALENDAR = "calendar"       # block time on user's calendar
    REMINDER = "reminder"       # remind the user to do something


# ---------------------------------------------------------------------------
# Step & Agent definitions
# ---------------------------------------------------------------------------

@dataclass
class StepDefinition:
    """One step in an agent's execution pipeline."""
    type: StepType
    config: dict[str, Any]
    name: str | None = None

    def __post_init__(self) -> None:
        if self.name is None:
            self.name = self.type.value


@dataclass
class AgentDefinition:
    """Blueprint for an agent — its step pipeline, scheduling, concurrency."""
    id: str
    name: str
    steps: list[StepDefinition]
    concurrency: int = 1                    # 1 = singleton
    schedule: str | None = None             # e.g. "30s", "5m", "1h"
    output_schema: dict | None = None       # expected shape of final output
    description: str = ""
    budgets: dict[str, int] = field(default_factory=dict)
    # Token budgets per provider.  e.g. {"gemini": 50000, "claude_max": 100000}
    # Limit is total tokens (input+output) per execution.  0 or absent = unlimited.
    connector_ids: list[str] = field(default_factory=list)
    # MCP connectors this agent uses.  Matched against ConnectorManager specs.

    @property
    def model(self) -> str | None:
        """Derive the model name from cognitive steps for display purposes.

        Returns None if no cognitive steps, the model string if one,
        or a comma-separated list if multiple distinct models are used.
        """
        models: list[str] = []
        for s in self.steps:
            if s.type == StepType.COGNITIVE:
                m = s.config.get("model", "")
                if m and m not in models:
                    models.append(m)
        if not models:
            return None
        return ", ".join(models)

    @staticmethod
    def generate_id() -> str:
        return uuid4().hex[:8]


# ---------------------------------------------------------------------------
# Inbox messages
# ---------------------------------------------------------------------------

@dataclass
class InboxMessage:
    """A message in an agent's inbox."""
    id: str
    source: str                 # who sent this
    target: str                 # agent_id
    type: MessageType
    priority: MessagePriority
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @staticmethod
    def generate_id() -> str:
        return uuid4().hex[:10]


# ---------------------------------------------------------------------------
# Token tracking
# ---------------------------------------------------------------------------

@dataclass
class TokenUsage:
    """Cumulative token usage for one provider within one execution."""
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_creation_tokens + self.output_tokens


# ---------------------------------------------------------------------------
# Execution records
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """Result of a single step execution."""
    step_index: int
    step_name: str
    step_type: StepType
    status: ExecutionStatus
    output: Any = None
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    pid: int | None = None                  # subprocess PID (script steps)


@dataclass
class ExecutionRecord:
    """Full record of one pipeline execution."""
    execution_id: str
    agent_id: str
    status: ExecutionStatus
    trigger_source: str                     # "scheduler", "user", "agent:<id>", etc.
    step_results: list[StepResult] = field(default_factory=list)
    current_step: int = 0
    output: Any = None                      # final pipeline output
    error: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    token_usage: dict[str, TokenUsage] = field(default_factory=dict)
    # Per-provider cumulative token usage for this execution (own + children).

    @staticmethod
    def generate_id() -> str:
        return uuid4().hex[:12]


# ---------------------------------------------------------------------------
# User profile & goal-oriented planning
# ---------------------------------------------------------------------------

@dataclass
class UserProfile:
    """User profile with goals, projects, and onboarding state.

    Forward-compatible with web3: jurisdiction_id field ready for DAO membership.
    """
    id: str = "local"                                   # user-scoped ID
    onboarding_status: OnboardingStatus = OnboardingStatus.NOT_STARTED
    summary: str = ""                                    # 2-3 sentence description
    standards: list[dict[str, Any]] = field(default_factory=list)
    # [{title, description}]
    goals: list[dict[str, Any]] = field(default_factory=list)
    # [{id, title, timeframe, description, motivation, status}]
    projects: list[dict[str, Any]] = field(default_factory=list)
    # [{id, title, description, goal_link, next_steps, status}]
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    jurisdiction_id: str | None = None                   # V0: None. Future: DAO ID
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CreditBudget:
    """Per-provider token budget for a billing period.

    The planning loop checks utilization and proactively allocates unused
    budget to goal-aligned tasks.
    """
    provider: str
    period: str = "monthly"                              # "daily" | "weekly" | "monthly"
    token_limit: int = 0                                 # 0 = unlimited
    tokens_used: int = 0
    period_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    auto_allocate: bool = True                           # planning loop can spend unused budget


@dataclass
class PlanningTask:
    """A task proposed or managed by the planning loop.

    Links a user's goal to a concrete action: an agent automation, a calendar
    event, or a reminder.
    """
    id: str
    goal_id: str                                         # which goal this serves
    title: str
    description: str = ""
    task_type: TaskType = TaskType.AUTOMATION
    status: TaskStatus = TaskStatus.PROPOSED
    agent_id: str | None = None                          # linked agent (automation tasks)
    calendar_event_id: str | None = None                 # linked gcal event (calendar tasks)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

    @staticmethod
    def generate_id() -> str:
        return uuid4().hex[:10]
