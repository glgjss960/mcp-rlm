from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List
import json

from ..types import EpisodeTrace
from .schema import (
    to_agentic_rl_row,
    to_cold_start_turn_rows,
    to_episode_rows,
    to_group_rows,
    to_memory_rows,
    to_step_rows,
)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_trace(trace: EpisodeTrace, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "episodes.jsonl", to_episode_rows(trace))
    _write_jsonl(out_dir / "groups.jsonl", to_group_rows(trace))
    _write_jsonl(out_dir / "steps.jsonl", to_step_rows(trace))
    _write_jsonl(out_dir / "memory_events.jsonl", to_memory_rows(trace))


def export_traces(traces: List[EpisodeTrace], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    episode_rows: List[Dict[str, Any]] = []
    group_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    memory_rows: List[Dict[str, Any]] = []

    for trace in traces:
        episode_rows.extend(to_episode_rows(trace))
        group_rows.extend(to_group_rows(trace))
        step_rows.extend(to_step_rows(trace))
        memory_rows.extend(to_memory_rows(trace))

    _write_jsonl(out_dir / "episodes.jsonl", episode_rows)
    _write_jsonl(out_dir / "groups.jsonl", group_rows)
    _write_jsonl(out_dir / "steps.jsonl", step_rows)
    _write_jsonl(out_dir / "memory_events.jsonl", memory_rows)


def export_cold_start(traces: List[EpisodeTrace], out_path: Path) -> None:
    rows: List[Dict[str, Any]] = []
    for trace in traces:
        rows.extend(to_cold_start_turn_rows(trace))
    _write_jsonl(out_path, rows)


def export_agentic_rl(traces: List[EpisodeTrace], out_path: Path) -> None:
    rows = [to_agentic_rl_row(trace) for trace in traces]
    _write_jsonl(out_path, rows)


def export_verl(traces: List[EpisodeTrace], out_path: Path) -> None:
    """Export a simple SFT-style JSONL usable as VERL warm-start input."""
    rows: List[Dict[str, Any]] = []
    for trace in traces:
        for row in to_cold_start_turn_rows(trace):
            rows.append(
                {
                    "prompt": json.dumps(
                        {
                            "history": row["history"],
                            "input_payload": row["input_payload"],
                        },
                        ensure_ascii=False,
                    ),
                    "response": json.dumps(
                        {
                            "action_type": row["target_action_type"],
                            "action_name": row["target_action_name"],
                            "result": row["target_result"],
                        },
                        ensure_ascii=False,
                    ),
                    "metadata": {
                        "episode_id": row["episode_id"],
                        "group_id": row["group_id"],
                        "step_id": row["step_id"],
                    },
                }
            )
    _write_jsonl(out_path, rows)


def export_openrlhf(traces: List[EpisodeTrace], out_path: Path) -> None:
    """Export episode-level trajectory rows for policy/reward training."""
    rows: List[Dict[str, Any]] = []
    for trace in traces:
        root_group = next(g for g in trace.groups if g.group_id == trace.root_group_id)
        rows.append(
            {
                "query": trace.goal,
                "response": trace.root_output,
                "reward": root_group.total_reward,
                "trajectory": to_agentic_rl_row(trace),
            }
        )
    _write_jsonl(out_path, rows)
