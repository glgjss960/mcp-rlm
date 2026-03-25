from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import json
import random

from ..types import ActionType, EpisodeTrace, WriteReason


@dataclass
class QueryItem:
    query: str
    answer: Optional[str] = None
    metadata: Dict[str, Any] | None = None


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


def load_query_items(path: Path) -> List[QueryItem]:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f'Query file not found: {resolved}')

    suffix = resolved.suffix.lower()
    items: List[QueryItem] = []

    if suffix in {'.jsonl', '.json'}:
        rows = _read_jsonl(resolved)
        for row in rows:
            query = str(row.get('query') or row.get('question') or row.get('goal') or '').strip()
            if not query:
                continue
            answer_raw = row.get('answer')
            if answer_raw is None:
                answer_raw = row.get('ground_truth')
            if answer_raw is None:
                answer_raw = row.get('target')
            answer = None if answer_raw is None else str(answer_raw).strip() or None
            metadata = {
                k: v
                for k, v in row.items()
                if k not in {'query', 'question', 'goal', 'answer', 'ground_truth', 'target'}
            }
            items.append(QueryItem(query=query, answer=answer, metadata=metadata))
        return items

    text = resolved.read_text(encoding='utf-8').replace('\ufeff', '', 1)
    for line in text.splitlines():
        query = line.strip()
        if not query:
            continue
        items.append(QueryItem(query=query, answer=None, metadata=None))
    return items


def _build_prompt_text(
    *,
    trace: EpisodeTrace,
    group: Any,
    step: Any,
    history: List[Dict[str, Any]],
) -> str:
    payload = {
        'episode_goal': trace.goal,
        'group_goal': group.goal,
        'group_depth': group.depth,
        'step_index': step.step_index,
        'history': history[-6:],
        'current_input': _jsonable(step.payload),
        'task': (
            'Predict the next action in MCP-RLM execution. '
            'Return compact JSON with action_type, action_name, and rationale.'
        ),
    }
    return json.dumps(payload, ensure_ascii=False)


def build_verl_step_rows(
    traces: List[EpisodeTrace],
    *,
    data_source: str = 'mcp_rlm/trajectory_step',
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for trace in traces:
        groups = {g.group_id: g for g in trace.groups}
        root_group = groups.get(trace.root_group_id)
        root_task_score = 0.0
        expected_answer: Optional[str] = None
        predicted_answer: Optional[str] = None

        if isinstance(trace.root_output, dict):
            root_task_score = float(trace.root_output.get('task_score', 0.0))
            if trace.root_output.get('expected_answer') is not None:
                expected_answer = str(trace.root_output.get('expected_answer')).strip() or None
            if trace.root_output.get('answer') is not None:
                predicted_answer = str(trace.root_output.get('answer')).strip() or None

        if root_group is not None and expected_answer is None:
            raw_expected = root_group.input_payload.get('expected_answer')
            if raw_expected is not None:
                expected_answer = str(raw_expected).strip() or None

        answer_correct = 0.0
        if expected_answer and predicted_answer:
            answer_correct = 1.0 if expected_answer.lower() in predicted_answer.lower() else 0.0

        steps_by_group: Dict[str, int] = {}
        object_calls_by_group: Dict[str, int] = {}
        writes_by_group: Dict[str, int] = {}
        context_pressure_writes_by_group: Dict[str, int] = {}
        value_writes_by_group: Dict[str, int] = {}

        for step in trace.steps:
            steps_by_group[step.group_id] = steps_by_group.get(step.group_id, 0) + 1
            if step.action_type == ActionType.CALL_OBJECT:
                count = 1
                if step.action_name == 'call_objects':
                    try:
                        count = max(1, int(step.payload.get('num_calls', 1)))
                    except Exception:
                        count = 1
                object_calls_by_group[step.group_id] = object_calls_by_group.get(step.group_id, 0) + count

        for event in trace.memory_events:
            writes_by_group[event.group_id] = writes_by_group.get(event.group_id, 0) + 1
            if event.reason == WriteReason.CONTEXT_PRESSURE:
                context_pressure_writes_by_group[event.group_id] = (
                    context_pressure_writes_by_group.get(event.group_id, 0) + 1
                )
            elif event.reason == WriteReason.VALUE_EVENT:
                value_writes_by_group[event.group_id] = value_writes_by_group.get(event.group_id, 0) + 1

        history_by_group: Dict[str, List[Dict[str, Any]]] = {}
        for step in trace.steps:
            group = groups.get(step.group_id)
            if group is None:
                continue

            history = history_by_group.setdefault(step.group_id, [])
            prompt_text = _build_prompt_text(trace=trace, group=group, step=step, history=history)

            group_step_count = max(1, steps_by_group.get(step.group_id, 1))
            group_object_calls = object_calls_by_group.get(step.group_id, 0)
            group_writes = writes_by_group.get(step.group_id, 0)
            group_pressure_writes = context_pressure_writes_by_group.get(step.group_id, 0)
            group_value_writes = value_writes_by_group.get(step.group_id, 0)

            step_progress = float(step.step_index) / float(group_step_count)
            object_call_ratio = float(group_object_calls) / float(group_step_count)
            write_ratio = float(group_writes) / float(group_step_count)
            context_pressure_write_ratio = float(group_pressure_writes) / float(group_step_count)
            value_write_ratio = float(group_value_writes) / float(group_step_count)

            ground_truth = {
                'target_action_type': step.action_type.value,
                'target_action_name': step.action_name,
                'target_payload': _jsonable(step.payload),
                'target_result': _jsonable(step.result),
                'target_ok': bool(step.ok),
            }

            extra_info = {
                'episode_id': trace.episode_id,
                'group_id': step.group_id,
                'step_id': step.step_id,
                'episode_success': 1.0 if trace.success else 0.0,
                'episode_task_score': root_task_score,
                'answer_correct': answer_correct,
                'expected_answer': expected_answer,
                'group_depth': int(group.depth),
                'group_status': group.status.value,
                'group_total_reward': float(group.total_reward),
                'group_step_count': group_step_count,
                'step_progress': step_progress,
                'object_call_ratio': object_call_ratio,
                'write_ratio': write_ratio,
                'context_pressure_write_ratio': context_pressure_write_ratio,
                'context_pressure_write_count': int(group_pressure_writes),
                'value_write_ratio': value_write_ratio,
                'value_write_count': int(group_value_writes),
                'step_ok': 1.0 if step.ok else 0.0,
            }

            rows.append(
                {
                    'uid': f'{trace.episode_id}:{step.step_id}',
                    'prompt': [
                        {
                            'role': 'user',
                            'content': prompt_text,
                        }
                    ],
                    'data_source': data_source,
                    'reward_model': {'ground_truth': ground_truth},
                    'extra_info': extra_info,
                }
            )

            history.append(
                {
                    'action_type': step.action_type.value,
                    'action_name': step.action_name,
                    'payload': _jsonable(step.payload),
                    'result': _jsonable(step.result),
                    'ok': bool(step.ok),
                }
            )

    return rows


def export_verl_on_policy_dataset(
    traces: List[EpisodeTrace],
    output_dir: Path,
    *,
    val_ratio: float = 0.1,
    seed: int = 7,
) -> Dict[str, Any]:
    rows = build_verl_step_rows(traces)
    rng = random.Random(seed)
    rng.shuffle(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / 'train.jsonl'
    val_path = output_dir / 'val.jsonl'

    if not rows:
        _write_jsonl(train_path, [])
        _write_jsonl(val_path, [])
        return {
            'train_path': str(train_path),
            'val_path': str(val_path),
            'num_rows': 0,
            'num_train_rows': 0,
            'num_val_rows': 0,
        }

    val_count = int(len(rows) * max(0.0, min(0.5, float(val_ratio))))
    val_count = max(1, val_count) if len(rows) > 1 else 0
    train_count = max(1, len(rows) - val_count)
    if train_count + val_count > len(rows):
        val_count = len(rows) - train_count

    train_rows = rows[:train_count]
    val_rows = rows[train_count : train_count + val_count]
    if not val_rows:
        val_rows = train_rows[:1]

    _write_jsonl(train_path, train_rows)
    _write_jsonl(val_path, val_rows)

    return {
        'train_path': str(train_path),
        'val_path': str(val_path),
        'num_rows': len(rows),
        'num_train_rows': len(train_rows),
        'num_val_rows': len(val_rows),
    }
