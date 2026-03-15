from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import asyncio
import json
import os
import time

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


class FileSharedMemory:
    """Append-only shared memory persisted as JSONL with file-level lock and CAS semantics."""

    def __init__(
        self,
        root_dir: str | Path,
        *,
        lock_timeout_seconds: float = 10.0,
        lock_poll_seconds: float = 0.05,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

        self.log_path = self.root_dir / "events.jsonl"
        self.lock_path = self.root_dir / "events.lock"

        if not self.log_path.exists():
            self.log_path.write_text("", encoding="utf-8")

        self.lock_timeout_seconds = lock_timeout_seconds
        self.lock_poll_seconds = lock_poll_seconds
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
            await asyncio.to_thread(self._acquire_file_lock_sync)
            try:
                versions = await asyncio.to_thread(self._load_versions_sync)
                current_version = versions.get(event.key, 0)
                if expected_version is not None and current_version != expected_version:
                    raise MemoryConflictError(
                        f"Version conflict for key={event.key}: expected {expected_version}, got {current_version}"
                    )

                new_event = replace(event, version=current_version + 1)
                await asyncio.to_thread(self._append_event_sync, new_event)
                return new_event
            finally:
                await asyncio.to_thread(self._release_file_lock_sync)

    async def read_latest(self, key: str) -> Optional[MemoryValue]:
        events = await self.export_events()
        latest: Optional[MemoryEvent] = None
        for event in events:
            if event.key != key:
                continue
            if latest is None or event.version > latest.version:
                latest = event
        if latest is None:
            return None
        return self._to_memory_value(latest)

    async def read_all(self, key: str) -> List[MemoryValue]:
        events = await self.export_events()
        values = [self._to_memory_value(event) for event in events if event.key == key]
        values.sort(key=lambda item: item.version)
        return values

    async def snapshot(self, keys: Optional[Iterable[str]] = None) -> Dict[str, MemoryValue]:
        wanted = set(keys) if keys is not None else None
        events = await self.export_events()
        latest: Dict[str, MemoryEvent] = {}
        for event in events:
            if wanted is not None and event.key not in wanted:
                continue
            prev = latest.get(event.key)
            if prev is None or event.version > prev.version:
                latest[event.key] = event
        return {key: self._to_memory_value(event) for key, event in latest.items()}

    async def current_version(self, key: str) -> int:
        latest = await self.read_latest(key)
        return 0 if latest is None else latest.version

    async def event_count(self) -> int:
        return len(await self.export_events())

    async def export_events(self, *, episode_id: Optional[str] = None) -> List[MemoryEvent]:
        raw_events = await asyncio.to_thread(self._load_events_sync)
        if episode_id is None:
            return raw_events
        return [event for event in raw_events if event.episode_id == episode_id]

    async def export_events_dict(self, *, episode_id: Optional[str] = None) -> List[Dict[str, Any]]:
        events = await self.export_events(episode_id=episode_id)
        return [asdict(event) for event in events]

    def _load_versions_sync(self) -> Dict[str, int]:
        versions: Dict[str, int] = {}
        for event in self._load_events_sync():
            prev = versions.get(event.key, 0)
            if event.version > prev:
                versions[event.key] = event.version
        return versions

    def _load_events_sync(self) -> List[MemoryEvent]:
        events: List[MemoryEvent] = []
        if not self.log_path.exists():
            return events
        with self.log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                events.append(self._event_from_dict(payload))
        return events

    def _append_event_sync(self, event: MemoryEvent) -> None:
        payload = self._event_to_dict(event)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _acquire_file_lock_sync(self) -> None:
        deadline = time.time() + self.lock_timeout_seconds
        while True:
            try:
                fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w", encoding="utf-8") as lock_file:
                    lock_file.write(str(os.getpid()))
                return
            except FileExistsError:
                if time.time() >= deadline:
                    raise TimeoutError(f"Timed out acquiring memory lock: {self.lock_path}")
                time.sleep(self.lock_poll_seconds)

    def _release_file_lock_sync(self) -> None:
        try:
            self.lock_path.unlink(missing_ok=True)
        except TypeError:
            if self.lock_path.exists():
                self.lock_path.unlink()

    @staticmethod
    def _to_memory_value(event: MemoryEvent) -> MemoryValue:
        return MemoryValue(
            key=event.key,
            version=event.version,
            content=event.content,
            object_type=event.object_type,
            event_id=event.event_id,
            timestamp=event.timestamp,
        )

    @staticmethod
    def _event_to_dict(event: MemoryEvent) -> Dict[str, Any]:
        return {
            "event_id": event.event_id,
            "episode_id": event.episode_id,
            "group_id": event.group_id,
            "key": event.key,
            "object_type": event.object_type.value,
            "reason": event.reason.value,
            "content": event.content,
            "confidence": event.confidence,
            "evidence_refs": list(event.evidence_refs),
            "parent_event_ids": list(event.parent_event_ids),
            "version": event.version,
            "timestamp": event.timestamp,
        }

    @staticmethod
    def _event_from_dict(payload: Dict[str, Any]) -> MemoryEvent:
        return MemoryEvent(
            event_id=str(payload["event_id"]),
            episode_id=str(payload["episode_id"]),
            group_id=str(payload["group_id"]),
            key=str(payload["key"]),
            object_type=MemoryObjectType(str(payload["object_type"])),
            reason=WriteReason(str(payload["reason"])),
            content=payload.get("content"),
            confidence=float(payload.get("confidence", 1.0)),
            evidence_refs=list(payload.get("evidence_refs", [])),
            parent_event_ids=list(payload.get("parent_event_ids", [])),
            version=int(payload.get("version", 0)),
            timestamp=str(payload.get("timestamp", "")),
        )
