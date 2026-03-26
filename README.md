# MCP-RLM Manual (Install -> Extend Object -> Benchmark -> RL -> Troubleshooting)

MCP-RLM is a recursive multi-agent runtime that combines:

- hierarchical long-context preprocessing and retrieval,
- file-backed shared memory,
- multi-server MCP orchestration,
- object-level parallel fan-out,
- trajectory export for cold-start and on-policy RL.

This README is the operational manual for the latest code in this repository.

## 0) Scope and Current Status

Implemented and ready:

- MVP end-to-end inference (`run_mvp_pipeline.py`, `run_mvp_inference.py`)
- LongBench-v2 runner (`run_longbench_v2_eval.py`)
- MCP transport with official SDK (default) and legacy fallback
- On-policy loop bridge to VERL (`run_online_ppo_with_verl.py`)
- Custom + external MCP servers and extra fan-out config (`--mcp-server-config`)

Not yet implemented as first-class benchmark runners:

- BabiLong
- RepoQA

You can still evaluate them by building dataset adapters (see Benchmark section).

## 1) Install

### 1.1 Prerequisites

Required:

- Python 3.10+
- pip

Recommended:

- Git
- virtual environment (`venv`)

Optional, depending on what you run:

- vLLM (local OpenAI-compatible endpoint)
- Ollama (local OpenAI-compatible endpoint)
- `transformers` + `torch` + `accelerate` (local HuggingFace policy)
- Node.js + `npx` (official MCP preset servers such as filesystem/sequential-thinking)
- `uv` / `uvx` (official MCP preset servers such as fetch/git)
- local VERL clone (`../verl`) for on-policy PPO

### 1.2 Base install

From repository root (`mcp-rlm/`):

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux/macOS
# source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
```

The project dependency currently includes:

- `mcp>=1.0.0` (official MCP Python SDK package)

### 1.3 Optional provider installs

For local HuggingFace policy mode:

```bash
pip install transformers torch accelerate
```

For local vLLM endpoint:

```bash
pip install vllm
```

### 1.4 Optional MCP SDK path override

If the `mcp` package is not importable from current env, MCP-RLM can also load SDK from a local checkout:

- clone path example: `../python-sdk/src`
- set environment variable:

```bash
# Windows PowerShell
$env:MCP_PYTHON_SDK_PATH = "D:/AINet/LLM/M-A-P/RLM/python-sdk/src"

# Linux/macOS
# export MCP_PYTHON_SDK_PATH=/path/to/python-sdk/src
```

### 1.5 Quick sanity checks

```bash
python examples/run_demo.py
python -m unittest tests.test_context_pressure tests.test_runtime_parallel tests.test_verl_online -v
```

## 2) Extend Object

This section explains how to add new MCP objects and make programs call them.

### 2.1 Object naming and routing rule

In multi-server mode, object names must be:

- `<alias>/<tool_name>`

Examples:

- `ctx/search_hierarchical`
- `analysis/merge_facts`
- `ofs/read_file` (external filesystem server alias)

### 2.2 Add a new built-in analysis object

File to edit:

- `mcp_rlm/mcp.py`

Pattern:

```python
def _my_object(payload: dict, ctx: MCPInvocationContext) -> dict:
    text = str(payload.get("text", ""))
    return {"length": len(text)}


def register_builtin_objects(registry: MCPRegistry) -> None:
    # existing registrations ...
    registry.register("my_object", _my_object)
```

Then call it from programs as:

- `analysis/my_object`

because `run_analysis_server.py` exposes `register_builtin_objects(...)`.

### 2.3 Add a new built-in context object

File to edit:

- `examples/run_context_server.py`

Pattern:

```python
def my_context_tool(payload, _ctx):
    # implement logic
    return {"ok": True}

registry.register("my_context_tool", my_context_tool)
```

Then call it from programs as:

- `ctx/my_context_tool`

### 2.4 Mount external MCP servers (official or custom)

Use `--mcp-server-config` JSON. Example:

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
python examples/run_mvp_pipeline.py   --input data/long_context.txt   --query-file data/query.txt   --mcp-server-config artifacts/mcp_servers.json
```

### 2.5 Make recursive programs use the new objects

Main program files:

- `mcp_rlm/mvp_programs.py`
- `mcp_rlm/longbench_v2_programs.py`

Use:

- `await ctx.call_object(...)` for one object
- `await ctx.call_objects([...])` for object-parallel fan-out

Tip: when adding heavy fan-out, keep payload compact and use `ctx.flush_context_pressure(...)` if you intentionally checkpoint memory under high context pressure.

## 3) Benchmark

### 3.1 MVP task inference (single long-context task)

One-command pipeline:

```bash
python examples/run_mvp_pipeline.py --input data/long_context.txt --query "your question"
```

Or query from file:

```bash
python examples/run_mvp_pipeline.py --input data/long_context.txt --query-file data/query.txt
```

Two-step mode:

```bash
python examples/preprocess_long_context.py --input data/long_context.txt --out artifacts/context_store
python examples/run_mvp_inference.py --manifest artifacts/context_store/manifest.json --query-file data/query.txt --out artifacts/mvp
```

### 3.2 Provider runbook

OpenRouter:

```bash
python examples/run_mvp_pipeline.py   --input data/long_context.txt   --query "your question"   --policy-mode openrouter   --model openai/gpt-4o-mini   --api-key <OPENROUTER_API_KEY>
```

vLLM:

```bash
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000
python examples/run_mvp_pipeline.py   --input data/long_context.txt   --query "your question"   --policy-mode vllm   --model Qwen/Qwen2.5-7B-Instruct   --api-base http://127.0.0.1:8000/v1
```

Ollama:

```bash
ollama serve
ollama pull qwen2.5:7b
python examples/run_mvp_pipeline.py   --input data/long_context.txt   --query "your question"   --policy-mode ollama   --model qwen2.5:7b   --api-base http://127.0.0.1:11434/v1
```

HuggingFace local:

```bash
python examples/run_mvp_pipeline.py   --input data/long_context.txt   --query "your question"   --policy-mode huggingface   --model Qwen/Qwen2.5-3B-Instruct
```

### 3.3 LongBench-v2 (supported)

Run:

```bash
python examples/run_longbench_v2_eval.py   --dataset-file ../LongBench/data.json   --out artifacts/longbench_v2   --policy-mode heuristic
```

With OpenRouter:

```bash
python examples/run_longbench_v2_eval.py   --dataset-file ../LongBench/data.json   --out artifacts/longbench_v2   --policy-mode openrouter   --model openai/gpt-4o-mini   --api-key <OPENROUTER_API_KEY>
```

With vLLM:

```bash
python examples/run_longbench_v2_eval.py   --dataset-file ../LongBench/data.json   --out artifacts/longbench_v2   --policy-mode vllm   --model Qwen/Qwen2.5-7B-Instruct   --api-base http://127.0.0.1:8000/v1
```

Useful controls:

- `--start-index N`
- `--limit N`
- `--ids id1,id2,...`
- `--resume`

### 3.4 BabiLong and RepoQA (adapter workflow, no built-in runner yet)

Current status:

- no dedicated `run_babilong_eval.py`
- no dedicated `run_repoqa_eval.py`

Recommended adapter path:

1. Convert benchmark samples into your own JSONL task file.
2. For QA-style tasks, run `run_mvp_inference.py` / `run_mvp_pipeline.py` per sample.
3. Save prediction + ground truth and score externally.
4. If needed, add a dedicated runner following `examples/run_longbench_v2_eval.py` structure.

## 4) RL (On-policy with VERL)

Reference doc:

- `docs/VERL_ONPOLICY.md`

### 4.1 Prepare inputs

- manifest: `artifacts/context_store/manifest.json`
- query file (`.jsonl` recommended):

```json
{"query": "question 1", "answer": "optional expected answer"}
{"query": "question 2"}
```

### 4.2 Rollout + dataset export only (no PPO)

```bash
python examples/run_online_ppo_with_verl.py   --manifest artifacts/context_store/manifest.json   --queries data/queries.jsonl   --out artifacts/online_rl   --iterations 1   --episodes-per-iter 8   --policy-mode huggingface   --model Qwen/Qwen2.5-3B-Instruct   --skip-verl-train
```

### 4.3 Full on-policy loop with VERL PPO

Assume VERL repo exists at `../verl`.

```bash
python examples/run_online_ppo_with_verl.py   --manifest artifacts/context_store/manifest.json   --queries data/queries.jsonl   --out artifacts/online_rl   --iterations 2   --episodes-per-iter 8   --policy-mode huggingface   --model Qwen/Qwen2.5-3B-Instruct   --actor-model-path Qwen/Qwen2.5-3B-Instruct   --verl-repo ../verl   --reward-manager-source register   --reward-manager-name mcp_rlm
```

Importlib reward manager fallback:

```bash
python examples/run_online_ppo_with_verl.py   --manifest artifacts/context_store/manifest.json   --queries data/queries.jsonl   --actor-model-path Qwen/Qwen2.5-3B-Instruct   --verl-repo ../verl   --reward-manager-source importlib   --reward-manager-name MCPRLMRewardManager   --reward-manager-module-path mcp_rlm/training/verl_reward_manager.py
```

### 4.4 RL outputs

Each iteration directory (`iter_000`, `iter_001`, ...) contains:

- trajectory exports: `episodes.jsonl`, `groups.jsonl`, `steps.jsonl`, `memory_events.jsonl`
- warm-start exports: `cold_start_turns.jsonl`, `agentic_rl.jsonl`, `verl_warm_start.jsonl`, `openrlhf_episodes.jsonl`
- VERL dataset: `verl_online_dataset/train.jsonl`, `verl_online_dataset/val.jsonl`
- checkpoints if PPO enabled: `verl_checkpoints/global_step_*/actor`
- `summary.json`

Global summary:

- `artifacts/online_rl/summary.json`

## 5) Troubleshooting

### 5.1 Official MCP SDK import error

Symptom:

- `Official MCP SDK is required but unavailable`
- `Unable to import MCP SDK`

Fix:

- install in current env: `pip install mcp`
- or set `MCP_PYTHON_SDK_PATH=/path/to/python-sdk/src`
- if needed, run with `--legacy-mcp` as fallback

### 5.2 Unknown MCP alias or object

Symptom:

- `Unknown MCP server alias: ...`
- `MCP object not found: ...`

Fix:

- verify object name format `<alias>/<tool_name>`
- verify server alias in `--mcp-server-config`
- verify registration in server code (`registry.register(...)`)

### 5.3 Connection refused to local model endpoint

Symptom:

- failures when `--policy-mode vllm` or `--policy-mode ollama`

Fix:

- start endpoint/service first
- verify `--api-base`
- verify model name

### 5.4 npx / uvx missing for official MCP presets

Symptom:

- extra server startup failure with preset mode

Fix:

- install Node.js (for `npx`) and/or `uv` (for `uvx`)
- use `--skip-unavailable-extra-servers` to skip unavailable presets

### 5.5 Windows permission denied on temp/artifact paths

Symptom:

- `PermissionError: [WinError 5] Access is denied`

Fix:

- ensure output and temp directories are writable
- avoid running in restricted temp locations
- set writable temp folder in PowerShell session:

```powershell
$env:TEMP = "D:\tmp"
$env:TMP = "D:\tmp"
```

### 5.6 Query input errors

Symptom:

- `Provide either --query or --query-file`
- `Query file is empty`

Fix:

- pass exactly one of `--query` / `--query-file`
- ensure UTF-8 text file with non-empty content

### 5.7 VERL training fails to start

Checklist:

- `--verl-repo` points to valid VERL clone
- `--actor-model-path` provided (unless `--skip-verl-train`)
- `PYTHONPATH` can import both `verl` and `mcp_rlm`
- GPU / CUDA / torch stack is available for selected rollout engine

## 6) Useful Commands Summary

MVP one-command:

```bash
python examples/run_mvp_pipeline.py --input data/long_context.txt --query-file data/query.txt
```

LongBench-v2:

```bash
python examples/run_longbench_v2_eval.py --dataset-file ../LongBench/data.json --out artifacts/longbench_v2 --policy-mode heuristic
```

On-policy RL (dataset only):

```bash
python examples/run_online_ppo_with_verl.py --manifest artifacts/context_store/manifest.json --queries data/queries.jsonl --skip-verl-train
```

Run tests:

```bash
python -m unittest discover -s tests -v
```

