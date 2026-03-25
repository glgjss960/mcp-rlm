from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

from .verl_reward import compute_score as mcp_rlm_compute_score


@register('mcp_rlm')
class MCPRLMRewardManager(AbstractRewardManager):
    """Reward manager tuned for MCP-RLM trajectory-step training data."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key='data_source', **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = int(num_examine)
        self.compute_score = compute_score or mcp_rlm_compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self._seen_data_source: dict[str, int] = {}
        self._debug_print = bool(kwargs.get('debug_print', False))

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            reward_model = data_item.non_tensor_batch.get('reward_model', {})
            ground_truth = reward_model.get('ground_truth', {}) if isinstance(reward_model, dict) else {}
            data_source = data_item.non_tensor_batch.get(self.reward_fn_key, 'mcp_rlm/trajectory_step')

            extra_info = data_item.non_tensor_batch.get('extra_info', {})
            if not isinstance(extra_info, dict):
                extra_info = {}
            num_turns = data_item.non_tensor_batch.get('__num_turns__', None)
            rollout_reward_scores = data_item.non_tensor_batch.get('reward_scores', {})
            extra_info['num_turns'] = num_turns
            extra_info['rollout_reward_scores'] = rollout_reward_scores

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = float(score.get('score', 0.0))
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = float(score)
                reward_extra_info['score'].append(reward)

            if valid_response_length > 0:
                reward_tensor[i, valid_response_length - 1] = reward

            seen_count = self._seen_data_source.get(str(data_source), 0)
            if self._debug_print and seen_count < self.num_examine:
                self._seen_data_source[str(data_source)] = seen_count + 1
                print('[prompt]', prompt_str)
                print('[response]', response_str)
                print('[ground_truth]', ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f'[{key}]', value)
                else:
                    print('[score]', score)

        if return_dict:
            return {
                'reward_tensor': reward_tensor,
                'reward_extra_info': reward_extra_info,
            }
        return reward_tensor
