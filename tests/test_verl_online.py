from __future__ import annotations

from pathlib import Path
import json
import shutil
import unittest
import uuid

from mcp_rlm.training import build_verl_step_rows, export_verl_on_policy_dataset
from mcp_rlm.training.verl_reward import compute_score
from mcp_rlm.types import (
    ActionType,
    Budget,
    EpisodeTrace,
    GroupSpec,
    GroupStatus,
    StepRecord,
)


def _build_trace() -> EpisodeTrace:
    episode_id = 'ep_test'
    root_group_id = 'grp_root'

    group = GroupSpec(
        group_id=root_group_id,
        episode_id=episode_id,
        goal='Find code',
        program='mvp_root',
        parent_group_id=None,
        depth=0,
        budget=Budget(),
        input_payload={'query': 'What is launch code?', 'expected_answer': '123-B77'},
        status=GroupStatus.SUCCEEDED,
        started_at='2026-01-01T00:00:00Z',
        ended_at='2026-01-01T00:00:01Z',
        result={'answer': '123-B77'},
        total_reward=0.9,
    )

    steps = [
        StepRecord(
            step_id='step_1',
            episode_id=episode_id,
            group_id=root_group_id,
            step_index=1,
            action_type=ActionType.CALL_OBJECT,
            action_name='call_object',
            payload={'object_name': 'analysis/merge_facts'},
            result={'answer': '123-B77'},
            ok=True,
            started_at='2026-01-01T00:00:00Z',
            ended_at='2026-01-01T00:00:00Z',
        ),
        StepRecord(
            step_id='step_2',
            episode_id=episode_id,
            group_id=root_group_id,
            step_index=2,
            action_type=ActionType.FINALIZE,
            action_name='finalize',
            payload={},
            result={'task_score': 0.9},
            ok=True,
            started_at='2026-01-01T00:00:00Z',
            ended_at='2026-01-01T00:00:01Z',
        ),
    ]

    return EpisodeTrace(
        episode_id=episode_id,
        root_group_id=root_group_id,
        goal='What is launch code?',
        started_at='2026-01-01T00:00:00Z',
        ended_at='2026-01-01T00:00:01Z',
        success=True,
        root_output={
            'answer': 'The code is 123-B77',
            'confidence': 0.95,
            'task_score': 0.9,
            'expected_answer': '123-B77',
        },
        groups=[group],
        steps=steps,
        memory_events=[],
    )


class TestVerlOnlineBridge(unittest.TestCase):
    def test_build_rows(self) -> None:
        rows = build_verl_step_rows([_build_trace()])
        self.assertEqual(len(rows), 2)

        first = rows[0]
        self.assertIn('prompt', first)
        self.assertIn('reward_model', first)
        self.assertIn('extra_info', first)
        self.assertEqual(first['data_source'], 'mcp_rlm/trajectory_step')

        gt = first['reward_model']['ground_truth']
        self.assertIn('target_action_name', gt)
        self.assertIn('target_result', gt)

    def test_export_dataset(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        out = repo_root / 'artifacts' / 'test_verl_online' / uuid.uuid4().hex
        try:
            info = export_verl_on_policy_dataset([_build_trace()], out, val_ratio=0.5, seed=3)
            self.assertGreaterEqual(info['num_rows'], 1)
            self.assertTrue((out / 'train.jsonl').exists())
            self.assertTrue((out / 'val.jsonl').exists())

            with (out / 'train.jsonl').open('r', encoding='utf-8') as f:
                line = f.readline().strip()
                self.assertTrue(line)
                parsed = json.loads(line)
                self.assertIn('prompt', parsed)
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def test_reward_shape(self) -> None:
        ground_truth = {
            'target_action_name': 'call_object',
            'target_action_type': 'CALL_OBJECT',
            'target_result': {'answer': '123-B77'},
        }
        extra_info = {
            'episode_task_score': 0.9,
            'episode_success': 1.0,
            'step_ok': 1.0,
            'expected_answer': '123-B77',
            'object_call_ratio': 0.1,
            'write_ratio': 0.1,
            'step_progress': 0.2,
        }

        good = compute_score(
            data_source='mcp_rlm/trajectory_step',
            solution_str='{"action_name":"call_object","action_type":"CALL_OBJECT","rationale":"answer 123-B77"}',
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        bad = compute_score(
            data_source='mcp_rlm/trajectory_step',
            solution_str='random text unrelated',
            ground_truth=ground_truth,
            extra_info=extra_info,
        )

        self.assertIn('score', good)
        self.assertGreater(good['score'], bad['score'])


if __name__ == '__main__':
    unittest.main()
