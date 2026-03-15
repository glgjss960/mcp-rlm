from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

from ..types import EpisodeTrace


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def to_episode_rows(trace: EpisodeTrace) -> List[Dict[str, Any]]:
    return [
        {
            "episode_id": trace.episode_id,
            "root_group_id": trace.root_group_id,
            "goal": trace.goal,
            "started_at": trace.started_at,
            "ended_at": trace.ended_at,
            "success": trace.success,
            "root_output": _jsonable(trace.root_output),
            "num_groups": len(trace.groups),
            "num_steps": len(trace.steps),
            "num_memory_events": len(trace.memory_events),
        }
    ]


def to_group_rows(trace: EpisodeTrace) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for group in trace.groups:
        row = asdict(group)
        row["status"] = group.status.value
        row["budget"] = asdict(group.budget)
        row["result"] = _jsonable(group.result)
        rows.append(row)
    return rows


def to_step_rows(trace: EpisodeTrace) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for step in trace.steps:
        row = asdict(step)
        row["action_type"] = step.action_type.value
        row["payload"] = _jsonable(step.payload)
        row["result"] = _jsonable(step.result)
        rows.append(row)
    return rows


def to_memory_rows(trace: EpisodeTrace) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for event in trace.memory_events:
        row = asdict(event)
        row["object_type"] = event.object_type.value
        row["reason"] = event.reason.value
        row["content"] = _jsonable(event.content)
        rows.append(row)
    return rows


def to_cold_start_turn_rows(trace: EpisodeTrace) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    history_by_group: Dict[str, List[Dict[str, Any]]] = {}

    for step in trace.steps:
        history = history_by_group.setdefault(step.group_id, [])
        row = {
            "episode_id": trace.episode_id,
            "group_id": step.group_id,
            "step_id": step.step_id,
            "step_index": step.step_index,
            "history": _jsonable(history),
            "input_payload": _jsonable(step.payload),
            "target_action_type": step.action_type.value,
            "target_action_name": step.action_name,
            "target_result": _jsonable(step.result),
            "ok": step.ok,
        }
        rows.append(row)
        history.append(
            {
                "action_type": step.action_type.value,
                "action_name": step.action_name,
                "payload": _jsonable(step.payload),
                "result": _jsonable(step.result),
                "ok": step.ok,
            }
        )
    return rows


def to_agentic_rl_row(trace: EpisodeTrace) -> Dict[str, Any]:
    return {
        "episode_id": trace.episode_id,
        "goal": trace.goal,
        "root_group_id": trace.root_group_id,
        "success": trace.success,
        "root_output": _jsonable(trace.root_output),
        "groups": to_group_rows(trace),
        "steps": to_step_rows(trace),
        "memory_events": to_memory_rows(trace),
    }
