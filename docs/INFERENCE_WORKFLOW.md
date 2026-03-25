# End-to-End Inference Workflow

## 1) Prepare input

Create two UTF-8 files:

- long context file, example: `data/long_context.txt`
- optional query file, example: `data/query.txt`

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

With inline query:

```bash
python examples/run_mvp_inference.py \
  --manifest artifacts/context_store/manifest.json \
  --query "Your question" \
  --out artifacts/mvp
```

With query file:

```bash
python examples/run_mvp_inference.py \
  --manifest artifacts/context_store/manifest.json \
  --query-file data/query.txt \
  --out artifacts/mvp
```

This launches context + analysis MCP servers and runs recursive group inference.

## 4) Provider-specific startup and run

### OpenRouter

No local service required.

```bash
python examples/run_mvp_inference.py \
  --manifest artifacts/context_store/manifest.json \
  --query "Your question" \
  --policy-mode openrouter \
  --model openai/gpt-4o-mini \
  --api-key <OPENROUTER_API_KEY>
```

### vLLM

Start endpoint:

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000
```

Run inference:

```bash
python examples/run_mvp_inference.py \
  --manifest artifacts/context_store/manifest.json \
  --query "Your question" \
  --policy-mode vllm \
  --model Qwen/Qwen2.5-7B-Instruct \
  --api-base http://127.0.0.1:8000/v1
```

### Ollama

Start service and pull model:

```bash
ollama serve
ollama pull qwen2.5:7b
```

Run inference:

```bash
python examples/run_mvp_inference.py \
  --manifest artifacts/context_store/manifest.json \
  --query "Your question" \
  --policy-mode ollama \
  --model qwen2.5:7b \
  --api-base http://127.0.0.1:11434/v1
```

### HuggingFace local

Install dependencies:

```bash
pip install transformers torch accelerate
```

Run inference (no separate endpoint):

```bash
python examples/run_mvp_inference.py \
  --manifest artifacts/context_store/manifest.json \
  --query "Your question" \
  --policy-mode huggingface \
  --model Qwen/Qwen2.5-3B-Instruct
```

## 5) Read final answer

Main answer is in:

- `artifacts/mvp/episodes.jsonl` (`root_output`)
- also printed in terminal

## 6) Inspect trajectory and memory

- `artifacts/mvp/groups.jsonl`
- `artifacts/mvp/steps.jsonl`
- `artifacts/mvp/memory_events.jsonl`
- `artifacts/mvp/memory/events.jsonl`

These files contain full recursive trace and file-backed shared-memory events.

## 7) One-command shortcut

With inline query:

```bash
python examples/run_mvp_pipeline.py --input data/long_context.txt --query "Your question"
```

With query file:

```bash
python examples/run_mvp_pipeline.py --input data/long_context.txt --query-file data/query.txt
```
