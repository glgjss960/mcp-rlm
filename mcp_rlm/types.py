from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class GroupStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"


class ActionType(str, Enum):
    READ_MEMORY = "READ_MEMORY"
    WRITE_MEMORY = "WRITE_MEMORY"
    CALL_OBJECT = "CALL_OBJECT"
    SPAWN_GROUPS = "SPAWN_GROUPS"
    JOIN_GROUPS = "JOIN_GROUPS"
    FINALIZE = "FINALIZE"


class MemoryObjectType(str, Enum):
    GOAL = "GOAL"
    FACT = "FACT"
    PLAN = "PLAN"
    DECISION = "DECISION"
    ARTIFACT = "ARTIFACT"
    METRIC = "METRIC"


class WriteReason(str, Enum):
    SPAWN_PREP = "SPAWN_PREP"
    JOIN_RESULT = "JOIN_RESULT"
    CONTEXT_PRESSURE = "CONTEXT_PRESSURE"
    VALUE_EVENT = "VALUE_EVENT"
    FINALIZE = "FINALIZE"


@dataclass
class Budget:
    max_steps: int = 64
    max_children: int = 32
    max_wall_seconds: float = 120.0
    max_object_calls: int = 256


@dataclass
class GroupSpec:
    group_id: str
    episode_id: str
    goal: str
    program: str
    parent_group_id: Optional[str] = None
    depth: int = 0
    budget: Budget = field(default_factory=Budget)
    input_payload: Dict[str, Any] = field(default_factory=dict)
    status: GroupStatus = GroupStatus.PENDING
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    result: Any = None
    error: Optional[str] = None
    immediate_reward: float = 0.0
    total_reward: float = 0.0


@dataclass
class MemoryEvent:
    event_id: str
    episode_id: str
    group_id: str
    key: str
    object_type: MemoryObjectType
    reason: WriteReason
    content: Any
    confidence: float = 1.0
    evidence_refs: List[str] = field(default_factory=list)
    parent_event_ids: List[str] = field(default_factory=list)
    version: int = 0
    timestamp: str = field(default_factory=utc_now_iso)


@dataclass
class StepRecord:
    step_id: str
    episode_id: str
    group_id: str
    step_index: int
    action_type: ActionType
    action_name: str
    payload: Dict[str, Any]
    result: Any
    ok: bool
    started_at: str
    ended_at: str
    error: Optional[str] = None


@dataclass
class GroupResult:
    group_id: str
    status: GroupStatus
    output: Any
    reward: float
    child_group_ids: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class EpisodeTrace:
    episode_id: str
    root_group_id: str
    goal: str
    started_at: str
    ended_at: str
    success: bool
    root_output: Any
    groups: List[GroupSpec]
    steps: List[StepRecord]
    memory_events: List[MemoryEvent]
