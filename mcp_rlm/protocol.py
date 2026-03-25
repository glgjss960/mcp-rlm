from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Any, Optional

from .memory import SharedMemory
from .types import MemoryEvent, MemoryObjectType, WriteReason


@dataclass
class WriteIntent:
    episode_id: str
    group_id: str
    key: str
    object_type: MemoryObjectType
    reason: WriteReason
    content: Any
    confidence: float = 1.0
    context_usage: float = 0.0
    expected_version: Optional[int] = None
    force: bool = False


class WritePolicy:
    """Controls when writes are committed to shared memory."""

    def __init__(
        self,
        *,
        context_pressure_threshold: float = 0.7,
        value_confidence_threshold: float = 0.75,
        min_interval_seconds: float = 0.0,
    ) -> None:
        self.context_pressure_threshold = context_pressure_threshold
        self.value_confidence_threshold = value_confidence_threshold
        self.min_interval_seconds = min_interval_seconds
        self._required_reasons = {
            WriteReason.SPAWN_PREP,
            WriteReason.JOIN_RESULT,
            WriteReason.FINALIZE,
        }

    def should_commit(self, intent: WriteIntent, *, last_write_time: Optional[float]) -> bool:
        if intent.force:
            return True
        if intent.reason in self._required_reasons:
            return True
        if (
            self.min_interval_seconds > 0.0
            and last_write_time is not None
            and monotonic() - last_write_time < self.min_interval_seconds
        ):
            return False
        if intent.reason == WriteReason.CONTEXT_PRESSURE:
            return intent.context_usage >= self.context_pressure_threshold
        if intent.context_usage >= self.context_pressure_threshold:
            return True
        if intent.reason == WriteReason.VALUE_EVENT and intent.confidence >= self.value_confidence_threshold:
            return True
        return False

    async def commit(
        self,
        *,
        memory: SharedMemory,
        intent: WriteIntent,
        last_write_time: Optional[float],
    ) -> Optional[MemoryEvent]:
        if not self.should_commit(intent, last_write_time=last_write_time):
            return None
        return await memory.append(
            episode_id=intent.episode_id,
            group_id=intent.group_id,
            key=intent.key,
            object_type=intent.object_type,
            reason=intent.reason,
            content=intent.content,
            confidence=intent.confidence,
            expected_version=intent.expected_version,
        )

