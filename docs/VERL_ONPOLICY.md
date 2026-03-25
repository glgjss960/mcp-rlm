# MCP-RLM + VERL On-Policy RL

## What this pipeline does

Each iteration runs:

1. MCP-RLM recursive inference rollout on sampled queries.
2. Export full trajectories and convert them into VERL RL dataset rows.
3. Run VERL PPO with trajectory-aware reward.
4. (Optional) update next-iteration rollout model to latest actor checkpoint.

## 1) Prepare inputs

- Long-context store manifest: `artifacts/context_store/manifest.json`
- Query file (`.jsonl` recommended):

```json
{"query": "question 1", "answer": "optional expected answer"}
{"query": "question 2"}
```

## 2) Run online loop

Use registered reward manager (`mcp_rlm`) in `verl==0.7.0`:

```bash
python examples/run_online_ppo_with_verl.py \
  --manifest artifacts/context_store/manifest.json \
  --queries data/queries.jsonl \
  --out artifacts/online_rl \
  --iterations 2 \
  --episodes-per-iter 8 \
  --policy-mode huggingface \
  --model Qwen/Qwen2.5-3B-Instruct \
  --actor-model-path Qwen/Qwen2.5-3B-Instruct \
  --verl-repo ../verl \
  --reward-manager-source register \
  --reward-manager-name mcp_rlm
```

If your VERL clone does not contain registered `mcp_rlm`, use importlib mode:

```bash
python examples/run_online_ppo_with_verl.py \
  --manifest artifacts/context_store/manifest.json \
  --queries data/queries.jsonl \
  --skip-verl-train \
  --reward-manager-source importlib \
  --reward-manager-name MCPRLMRewardManager \
  --reward-manager-module-path mcp_rlm/training/verl_reward_manager.py
```

## 3) MCP transport mode

- Default: official MCP Python SDK transport.
- `--require-official-mcp-sdk`: fail fast if SDK/deps unavailable.
- `--legacy-mcp`: force legacy JSON-RPC transport (debug/fallback).

## 4) Important flags

- `--skip-verl-train`: rollout + dataset export only.
- `--rollout-engine hf|vllm|sglang`: VERL rollout backend.
- `--total-training-steps`: PPO steps per iteration.
- `--sync-rollout-hf-from-ckpt`: if rollout policy is HuggingFace mode, auto use latest actor ckpt in next iteration.

## 5) Output layout

Per iteration (`iter_000`, `iter_001`, ...):

- `episodes.jsonl`, `groups.jsonl`, `steps.jsonl`, `memory_events.jsonl`
- `cold_start_turns.jsonl`, `agentic_rl.jsonl`, `verl_warm_start.jsonl`, `openrlhf_episodes.jsonl`
- `verl_online_dataset/train.jsonl`
- `verl_online_dataset/val.jsonl`
- `verl_checkpoints/global_step_*/actor` (if VERL training enabled)
- `summary.json`

Global summary:

- `artifacts/online_rl/summary.json`

## 6) Reward and loss design

- Reward (`mcp_rlm/training/verl_reward.py`):
  - action-name match
  - action-type match
  - result text similarity
  - process bonus (episode success + task score + step ok)
  - answer match bonus
  - trajectory cost penalty (object/write ratios + step progress)
- Loss: VERL PPO default (policy clip objective + value loss + entropy regularization; KL disabled by default in this script).
