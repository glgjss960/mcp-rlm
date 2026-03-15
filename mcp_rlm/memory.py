from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Iterable, List, Optional
import asyncio

from .types import MemoryEvent, MemoryObjectType, WriteReason, new_id


class MemoryConflictError(RuntimeError):
    pass


@dataclass
class MemoryValue:
    key: str
    version: int
    content: Any
    object_type: MemoryObjectType
    event_id: str
    timestamp: str


class SharedMemory:
    """Append-only shared memory with per-key versioning and CAS writes."""

    def __init__(self) -> None:
        self._events: List[MemoryEvent] = []
        self._events_by_key: Dict[str, List[MemoryEvent]] = {}
        self._versions: Dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def append(
        self,
        *,
        episode_id: str,
        group_id: str,
        key: str,
        object_type: MemoryObjectType,
        reason: WriteReason,
        content: Any,
        confidence: float = 1.0,
        evidence_refs: Optional[List[str]] = None,
        parent_event_ids: Optional[List[str]] = None,
        expected_version: Optional[int] = None,
    ) -> MemoryEvent:
        event = MemoryEvent(
            event_id=new_id("mem"),
            episode_id=episode_id,
            group_id=group_id,
            key=key,
            object_type=object_type,
            reason=reason,
            content=content,
            confidence=confidence,
            evidence_refs=evidence_refs or [],
            parent_event_ids=parent_event_ids or [],
        )
        return await self.write_event(event, expected_version=expected_version)

    async def write_event(
        self,
        event: MemoryEvent,
        *,
        expected_version: Optional[int] = None,
    ) -> MemoryEvent:
        async with self._lock:
            current_version = self._versions.get(event.key, 0)
            if expected_version is not None and current_version != expected_version:
                raise MemoryConflictError(
                    f"Version conflict for key={event.key}: expected {expected_version}, got {current_version}"
                )

            new_event = replace(event, version=current_version + 1)
            self._versions[event.key] = new_event.version
            self._events.append(new_event)
            self._events_by_key.setdefault(event.key, []).append(new_event)
            return new_event

    async def read_latest(self, key: str) -> Optional[MemoryValue]:
        async with self._lock:
            events = self._events_by_key.get(key)
            if not events:
                return None
            event = events[-1]
            return MemoryValue(
                key=key,
                version=event.version,
                content=event.content,
                object_type=event.object_type,
                event_id=event.event_id,
                timestamp=event.timestamp,
            )

    async def read_all(self, key: str) -> List[MemoryValue]:
        async with self._lock:
            events = self._events_by_key.get(key, [])
            return [
                MemoryValue(
                    key=key,
                    version=e.version,
                    content=e.content,
                    object_type=e.object_type,
                    event_id=e.event_id,
                    timestamp=e.timestamp,
                )
                for e in events
            ]

    async def snapshot(self, keys: Optional[Iterable[str]] = None) -> Dict[str, MemoryValue]:
        async with self._lock:
            target_keys = list(keys) if keys is not None else list(self._events_by_key.keys())
            out: Dict[str, MemoryValue] = {}
            for key in target_keys:
                events = self._events_by_key.get(key)
                if not events:
                    continue
                event = events[-1]
                out[key] = MemoryValue(
                    key=key,
                    version=event.version,
                    content=event.content,
                    object_type=event.object_type,
                    event_id=event.event_id,
                    timestamp=event.timestamp,
                )
            return out

    async def current_version(self, key: str) -> int:
        async with self._lock:
            return self._versions.get(key, 0)

    async def event_count(self) -> int:
        async with self._lock:
            return len(self._events)

    async def export_events(self, *, episode_id: Optional[str] = None) -> List[MemoryEvent]:
        async with self._lock:
            if episode_id is None:
                return list(self._events)
            return [e for e in self._events if e.episode_id == episode_id]

    async def export_events_dict(self, *, episode_id: Optional[str] = None) -> List[Dict[str, Any]]:
        events = await self.export_events(episode_id=episode_id)
        return [asdict(e) for e in events]
