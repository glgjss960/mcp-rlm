from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .types import GroupSpec, GroupStatus


@dataclass
class RewardWeights:
    task: float = 0.6
    process: float = 0.2
    memory: float = 0.1
    cost: float = 0.1


class HierarchicalRewardModel:
    def __init__(self, *, discount: float = 0.9, weights: RewardWeights | None = None) -> None:
        self.discount = discount
        self.weights = weights or RewardWeights()

    def immediate_reward(
        self,
        *,
        group: GroupSpec,
        step_count: int,
        write_count: int,
        object_call_count: int,
    ) -> float:
        task_score = self._task_score(group)
        process_score = 1.0 if group.status == GroupStatus.SUCCEEDED else -1.0
        memory_score = min(1.0, write_count / 3.0)

        cost_ratio = 0.0
        if group.budget.max_steps > 0:
            cost_ratio += min(1.0, step_count / group.budget.max_steps)
        if group.budget.max_object_calls > 0:
            cost_ratio += min(1.0, object_call_count / group.budget.max_object_calls)
        cost_ratio /= 2.0

        reward = (
            self.weights.task * task_score
            + self.weights.process * process_score
            + self.weights.memory * memory_score
            - self.weights.cost * cost_ratio
        )
        return round(float(reward), 6)

    def total_reward(self, *, immediate: float, child_totals: list[float]) -> float:
        if not child_totals:
            return immediate
        mean_child = sum(child_totals) / len(child_totals)
        return round(float(immediate + self.discount * mean_child), 6)

    @staticmethod
    def _task_score(group: GroupSpec) -> float:
        result: Any = group.result
        if isinstance(result, dict) and "task_score" in result:
            try:
                return float(result["task_score"])
            except (TypeError, ValueError):
                pass
        return 1.0 if group.status == GroupStatus.SUCCEEDED else 0.0
