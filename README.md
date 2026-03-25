# MCP-RLM MVP

MCP-RLM MVP implementation with:

- LLM policy routing (`heuristic`, `openai`, `openrouter`, `vllm`, `ollama`, `huggingface`)
- File-backed shared memory (append-only JSONL + CAS)
- Hierarchical long-context preprocessing and retrieval
- Multi-server MCP orchestration (`ctx` + `analysis` servers) via official MCP Python SDK
- Recursive multi-group runtime with `spawn/join`
- Default object-parallel fan-out in both root and leaf programs

## Key modules

- `mcp_rlm/runtime.py`: recursive group runtime and scheduler
- `mcp_rlm/policy.py`: policy layer (OpenAI-compatible + optional local HuggingFace)
- `mcp_rlm/file_memory.py`: file-based shared memory
- `mcp_rlm/long_context.py`: preprocessing + hierarchical context store
- `mcp_rlm/multi_mcp.py`: multi-server MCP client orchestration
- `mcp_rlm/mvp_programs.py`: MVP recursive programs (root and leaf object-parallel fan-out)
- `mcp_rlm/stdio_mcp_client.py`: stdio MCP client (official `mcp.ClientSession` transport)
- `mcp_rlm/stdio_mcp_server.py`: stdio MCP server (official `mcp.server.Server`)

## MCP servers and objects

Servers:

- `examples/run_context_server.py`
- `examples/run_analysis_server.py`

Objects:

- `ctx/context_stats`
- `ctx/list_level`
- `ctx/read_segment`
- `ctx/search_hierarchical`
- `analysis/extract_facts`
- `analysis/analyze_segment`
- `analysis/merge_facts`
- `analysis/sleep` (test/demo)

## Quick start (one command)

```bash
cd mcp-rlm
python examples/run_mvp_pipeline.py --input data/long_context.txt --query "your question"
```

You can also pass query by file:

```bash
python examples/run_mvp_pipeline.py --input data/long_context.txt --query-file data/query.txt
```

## Two-step inference

1. Preprocess long text:

```bash
python examples/preprocess_long_context.py --input data/long_context.txt --out artifacts/context_store
```

2. Run inference:

```bash
python examples/run_mvp_inference.py --manifest artifacts/context_store/manifest.json --query "your question" --out artifacts/mvp
```

Or with query file:

```bash
python examples/run_mvp_inference.py --manifest artifacts/context_store/manifest.json --query-file data/query.txt --out artifacts/mvp
```

## Provider runbook (end-to-end)

Common flags for `run_mvp_inference.py` and `run_mvp_pipeline.py`:

- `--policy-mode heuristic|openai|openrouter|vllm|ollama|huggingface`
- `--model <model-name-or-path>`
- `--api-base <openai-compatible-base-url>`
- `--api-key <api-key>`

### 1) OpenRouter

No local service required.

```bash
python examples/run_mvp_pipeline.py \
  --input data/long_context.txt \
  --query "your question" \
  --policy-mode openrouter \
  --model openai/gpt-4o-mini \
  --api-key <OPENROUTER_API_KEY>
```

Optional headers:

- `--openrouter-site-url https://your.site`
- `--openrouter-app-name mcp-rlm`

### 2) vLLM (local OpenAI-compatible endpoint)

1. Install and start endpoint:

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000
```

2. Run MCP-RLM:

```bash
python examples/run_mvp_pipeline.py \
  --input data/long_context.txt \
  --query "your question" \
  --policy-mode vllm \
  --model Qwen/Qwen2.5-7B-Instruct \
  --api-base http://127.0.0.1:8000/v1
```

### 3) Ollama (local OpenAI-compatible endpoint)

1. Install and start Ollama service, then pull model:

```bash
ollama serve
ollama pull qwen2.5:7b
```

2. Run MCP-RLM:

```bash
python examples/run_mvp_pipeline.py \
  --input data/long_context.txt \
  --query "your question" \
  --policy-mode ollama \
  --model qwen2.5:7b \
  --api-base http://127.0.0.1:11434/v1
```

### 4) HuggingFace local model (direct `transformers`)

1. Install dependencies:

```bash
pip install transformers torch accelerate
```

2. Run MCP-RLM (no separate endpoint required):

```bash
python examples/run_mvp_pipeline.py \
  --input data/long_context.txt \
  --query "your question" \
  --policy-mode huggingface \
  --model Qwen/Qwen2.5-3B-Instruct
```

Optional flags:

- `--hf-device-map auto`
- `--hf-torch-dtype auto`
- `--hf-max-new-tokens 256`

## Outputs

Inference outputs are written to `--out`:

- `episodes.jsonl`
- `groups.jsonl`
- `steps.jsonl`
- `memory_events.jsonl`
- `cold_start_turns.jsonl`
- `agentic_rl.jsonl`
- `verl_warm_start.jsonl`
- `openrlhf_episodes.jsonl`
- `memory/events.jsonl` (file shared memory log)

## Tests

```bash
python -m unittest discover -s tests -v
```




## On-policy VERL RL loop (MCP-RLM rollout -> VERL PPO)

This repo now includes an online RL loop script:

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
  --verl-repo ../verl
```

`data/queries.jsonl` row format (minimal):

```json
{"query": "your question", "answer": "optional ground truth"}
```

Per iteration outputs:

- rollout traces: `episodes/groups/steps/memory_events`
- warm-start exports: `cold_start_turns.jsonl`, `agentic_rl.jsonl`, `verl_warm_start.jsonl`
- VERL on-policy dataset: `iter_xxx/verl_online_dataset/train.jsonl` + `val.jsonl`
- reward function for VERL: `mcp_rlm/training/verl_reward.py`
- PPO checkpoints (if enabled): `iter_xxx/verl_checkpoints/global_step_*/actor`

Use `--skip-verl-train` to generate rollout + RL datasets only.
Use `--sync-rollout-hf-from-ckpt` to update the next-iteration rollout model from latest VERL actor checkpoint.

## Official MCP SDK strict mode

All runner scripts support:

- `--require-official-mcp-sdk`: require official MCP Python SDK transport, fail fast if unavailable
- `--legacy-mcp`: force legacy JSON-RPC stdio mode

Examples:

```bash
python examples/run_mvp_inference.py --manifest artifacts/context_store/manifest.json --query-file data/query.txt --require-official-mcp-sdk
python examples/run_mvp_pipeline.py --input data/long_context.txt --query-file data/query.txt --legacy-mcp
```

## VERL reward-manager integration (verl==0.7.0)

This workspace now includes:

- `verl/verl/workers/reward_manager/mcp_rlm.py`
- registration in `verl/verl/workers/reward_manager/__init__.py`

Default online loop settings use:

- `--reward-manager-source register`
- `--reward-manager-name mcp_rlm`

Alternative importlib manager path:

- `mcp_rlm/training/verl_reward_manager.py`



## LongBench v2 integration

This repo includes LongBench-v2-specific recursive programs:

- `longbench_v2_root`
- `longbench_v2_leaf_segment`

LongBench-v2-related analysis objects:

- `analysis/score_mcq_choices`
- `analysis/aggregate_mcq_scores`
- `analysis/normalize_mcq_answer`

Run benchmark inference from local `data.json` or `data.jsonl`:

```bash
python examples/run_longbench_v2_eval.py --dataset-file ../LongBench/data.json --out artifacts/longbench_v2 --policy-mode heuristic
```

With OpenRouter:

```bash
python examples/run_longbench_v2_eval.py --dataset-file ../LongBench/data.json --out artifacts/longbench_v2 --policy-mode openrouter --model openai/gpt-4o-mini --api-key <OPENROUTER_API_KEY>
```

With local vLLM:

```bash
python examples/run_longbench_v2_eval.py --dataset-file ../LongBench/data.json --out artifacts/longbench_v2 --policy-mode vllm --model Qwen/Qwen2.5-7B-Instruct --api-base http://127.0.0.1:8000/v1
```

Useful controls:

- `--start-index N`
- `--limit N`
- `--ids id1,id2,...`
- `--resume`

## High-Score LongBench v2 object pack (custom + official MCP)

This repo now includes an expanded object set for LongBench-v2 score-oriented inference.

Custom context objects (`ctx/*`):

- `ctx/search_multi_query`
- `ctx/search_hierarchical_mmr`
- `ctx/read_segment_windows`

Custom analysis objects (`analysis/*`):

- `analysis/rerank_hits_with_choices`
- `analysis/score_mcq_choices`
- `analysis/score_mcq_windows`
- `analysis/eliminate_choices`
- `analysis/extract_code_cues`
- `analysis/extract_table_cues`
- `analysis/vote_choice_scores`
- `analysis/aggregate_mcq_scores`
- `analysis/normalize_mcq_answer`

`longbench_v2_root` and `longbench_v2_leaf_segment` now fan-out these objects in parallel by default.

### Official MCP servers integration

You can mount official MCP servers (filesystem/memory/fetch/git/sequential-thinking) as extra aliases:

```bash
python examples/run_longbench_v2_eval.py \
  --dataset-file ../LongBench/data.json \
  --out artifacts/longbench_v2 \
  --policy-mode heuristic \
  --enable-official-mcp-presets \
  --skip-unavailable-extra-servers
```

Or provide explicit server + extra fan-out config via JSON (`--mcp-server-config`):

```json
{
  "servers": [
    {
      "alias": "ofs",
      "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem", "D:/AINet/LLM/M-A-P/RLM"],
      "cwd": "D:/AINet/LLM/M-A-P/RLM"
    }
  ],
  "root_extra_object_fanout": [
    {
      "object_name": "ofs/list_directory",
      "payload": {"path": "D:/AINet/LLM/M-A-P/RLM/LongBench"},
      "timeout_seconds": 20
    }
  ],
  "leaf_extra_object_fanout": [
    {
      "object_name": "ofs/read_file",
      "payload": {"path": "D:/AINet/LLM/M-A-P/RLM/LongBench/prompts/0shot.txt"},
      "timeout_seconds": 20
    }
  ]
}
```

Run with config:

```bash
python examples/run_longbench_v2_eval.py \
  --dataset-file ../LongBench/data.json \
  --out artifacts/longbench_v2 \
  --policy-mode heuristic \
  --mcp-server-config artifacts/mcp_servers.json
```

The same extra-server options are available in:

- `examples/run_mvp_inference.py`
- `examples/run_mvp_pipeline.py`
