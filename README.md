# MCP-RLM MVP

MCP-RLM MVP implementation with:

- LLM policy routing (`heuristic` or `openai` compatible)
- File-backed shared memory (append-only JSONL + CAS)
- Hierarchical long-context preprocessing and retrieval
- Multi-server MCP orchestration (`ctx` + `analysis` servers)
- Recursive multi-group runtime with `spawn/join`

## Key modules

- `mcp_rlm/runtime.py`: recursive group runtime and scheduler
- `mcp_rlm/policy.py`: LLM policy layer
- `mcp_rlm/file_memory.py`: file-based shared memory
- `mcp_rlm/long_context.py`: preprocessing + hierarchical context store
- `mcp_rlm/multi_mcp.py`: multi-server MCP client orchestration
- `mcp_rlm/mvp_programs.py`: MVP recursive policy-driven programs
- `mcp_rlm/stdio_mcp_server.py`: stdio MCP server

## End-to-end inference (two-step)

1. Preprocess long text

```bash
cd mcp-rlm
python examples/preprocess_long_context.py --input path/to/long.txt --out artifacts/context_store
```

2. Run MVP inference

```bash
python examples/run_mvp_inference.py --manifest artifacts/context_store/manifest.json --query "your question" --out artifacts/mvp
```

## End-to-end inference (one command)

```bash
python examples/run_mvp_pipeline.py --input path/to/long.txt --query "your question" --store artifacts/context_store --out artifacts/mvp_pipeline
```

## LLM policy mode

Default is heuristic policy.

To use OpenAI-compatible chat endpoint:

```bash
set MCP_RLM_POLICY_MODE=openai
set MCP_RLM_API_BASE=https://your-endpoint/v1
set MCP_RLM_MODEL=your-model-name
set MCP_RLM_API_KEY=your-key
```

Then run `run_mvp_inference.py` or `run_mvp_pipeline.py`.

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

## MCP servers

- `examples/run_context_server.py`: context tools (`context_stats`, `list_level`, `read_segment`, `search_hierarchical`)
- `examples/run_analysis_server.py`: analysis tools (`extract_facts`, `analyze_segment`, `merge_facts`)

## Tests

```bash
python -m unittest discover -s tests -v
```
