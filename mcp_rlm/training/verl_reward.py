from __future__ import annotations

from typing import Any, Dict
import json
import re


_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.startswith('{') and text.endswith('}'):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                return {}
    return {}


def _extract_json_object(text: str) -> Dict[str, Any]:
    start = text.find('{')
    end = text.rfind('}')
    if start < 0 or end <= start:
        return {}
    snippet = text[start : end + 1]
    try:
        parsed = json.loads(snippet)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _normalize_tokens(text: str) -> list[str]:
    return [t for t in _WORD_RE.findall(text.lower()) if t]


def _token_f1(pred: str, truth: str) -> float:
    pred_tokens = _normalize_tokens(pred)
    truth_tokens = _normalize_tokens(truth)
    if not pred_tokens or not truth_tokens:
        return 0.0

    pred_counts: Dict[str, int] = {}
    truth_counts: Dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in truth_tokens:
        truth_counts[token] = truth_counts.get(token, 0) + 1

    overlap = 0
    for token, pred_count in pred_counts.items():
        overlap += min(pred_count, truth_counts.get(token, 0))

    if overlap <= 0:
        return 0.0

    precision = overlap / float(len(pred_tokens))
    recall = overlap / float(len(truth_tokens))
    denom = precision + recall
    if denom <= 0.0:
        return 0.0
    return 2.0 * precision * recall / denom


def _contains_casefold(haystack: str, needle: str) -> bool:
    if not needle:
        return False
    return needle.casefold() in haystack.casefold()


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Dict[str, Any] | None = None,
    **_: Any,
) -> Dict[str, Any]:
    """Trajectory-aware reward for MCP-RLM step policy optimization on VERL.

    Returns a dict with mandatory key `score` and additional diagnostic metrics.
    """

    _ = data_source  # reserved for multi-source extension

    solution = str(solution_str or '')
    gt = _safe_dict(ground_truth)
    extra = extra_info if isinstance(extra_info, dict) else {}

    target_action_name = str(gt.get('target_action_name', '')).strip()
    target_action_type = str(gt.get('target_action_type', '')).strip()
    target_result = gt.get('target_result')
    target_result_text = json.dumps(target_result, ensure_ascii=False, sort_keys=True)

    parsed_solution = _extract_json_object(solution)
    parsed_action_name = str(
        parsed_solution.get('action_name')
        or parsed_solution.get('target_action_name')
        or parsed_solution.get('action')
        or ''
    ).strip()
    parsed_action_type = str(
        parsed_solution.get('action_type')
        or parsed_solution.get('target_action_type')
        or ''
    ).strip()

    action_name_match = 0.0
    if target_action_name:
        if parsed_action_name and parsed_action_name.casefold() == target_action_name.casefold():
            action_name_match = 1.0
        elif _contains_casefold(solution, target_action_name):
            action_name_match = 0.75

    action_type_match = 0.0
    if target_action_type:
        if parsed_action_type and parsed_action_type.casefold() == target_action_type.casefold():
            action_type_match = 1.0
        elif _contains_casefold(solution, target_action_type):
            action_type_match = 0.75

    result_similarity = _token_f1(solution, target_result_text)

    episode_task_score = _clip(_to_float(extra.get('episode_task_score', 0.0)))
    episode_success = _clip(_to_float(extra.get('episode_success', 0.0)))
    step_ok = _clip(_to_float(extra.get('step_ok', 0.0)))
    process_bonus = _clip(0.5 * episode_task_score + 0.3 * episode_success + 0.2 * step_ok)

    answer_correct = _clip(_to_float(extra.get('answer_correct', 0.0)))
    expected_answer = str(extra.get('expected_answer') or '').strip()
    answer_match = answer_correct
    if expected_answer and _contains_casefold(solution, expected_answer):
        answer_match = max(answer_match, 1.0)

    object_call_ratio = _clip(_to_float(extra.get('object_call_ratio', 0.0)))
    write_ratio = _clip(_to_float(extra.get('write_ratio', 0.0)))
    context_pressure_write_ratio = _clip(_to_float(extra.get('context_pressure_write_ratio', 0.0)))
    step_progress = _clip(_to_float(extra.get('step_progress', 0.0)))

    # Context-pressure writes are hygiene operations; discount part of their cost.
    effective_write_ratio = _clip(max(0.0, write_ratio - (0.5 * context_pressure_write_ratio)))
    cost_penalty = _clip(0.5 * object_call_ratio + 0.3 * effective_write_ratio + 0.2 * step_progress)
    memory_hygiene_bonus = _clip(min(0.2, context_pressure_write_ratio))

    verbosity_penalty = 0.0
    if len(solution) > 3000:
        verbosity_penalty = 0.1

    score = (
        0.42 * action_name_match
        + 0.14 * action_type_match
        + 0.20 * result_similarity
        + 0.16 * process_bonus
        + 0.12 * answer_match
        - 0.16 * cost_penalty
        + 0.04 * memory_hygiene_bonus
        - verbosity_penalty
    )
    score = _clip(score)

    return {
        'score': score,
        'action_name_match': action_name_match,
        'action_type_match': action_type_match,
        'result_similarity': result_similarity,
        'process_bonus': process_bonus,
        'answer_match': answer_match,
        'cost_penalty': cost_penalty,
        'effective_write_ratio': effective_write_ratio,
        'context_pressure_write_ratio': context_pressure_write_ratio,
        'memory_hygiene_bonus': memory_hygiene_bonus,
        'verbosity_penalty': verbosity_penalty,
    }
