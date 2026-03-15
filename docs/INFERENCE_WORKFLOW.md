# End-to-End Inference Workflow

## 1) Prepare input

Create one UTF-8 text file containing the full long context.

Example: `data/long_context.txt`

## 2) Preprocess into hierarchical store

```bash
python examples/preprocess_long_context.py \
  --input data/long_context.txt \
  --out artifacts/context_store \
  --chunk-chars 16000 \
  --overlap-chars 400 \
  --branch-factor 8
```

Output: `artifacts/context_store/manifest.json` + segmented files.

## 3) Run multi-server MCP inference

```bash
python examples/run_mvp_inference.py \
  --manifest artifacts/context_store/manifest.json \
  --query "Your question" \
  --out artifacts/mvp
```

This launches context + analysis MCP servers and runs recursive group inference.

## 4) Read final answer

Main answer is in:

- `artifacts/mvp/episodes.jsonl` (`root_output`)
- also printed in terminal

## 5) Inspect trajectory and memory

- `artifacts/mvp/groups.jsonl`
- `artifacts/mvp/steps.jsonl`
- `artifacts/mvp/memory_events.jsonl`
- `artifacts/mvp/memory/events.jsonl`

These files contain full recursive trace and file-backed shared-memory events.

## 6) One-command shortcut

```bash
python examples/run_mvp_pipeline.py --input data/long_context.txt --query "Your question"
```
