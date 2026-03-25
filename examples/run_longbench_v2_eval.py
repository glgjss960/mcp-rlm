from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List
import argparse
import asyncio
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp_rlm import (
    FileSharedMemory,
    MCPRLMRuntime,
    MCPServerSpec,
    MultiServerMCPClient,
    ProgramRegistry,
    load_mcp_extension_config,
    preprocess_long_context,
    register_builtin_programs,
    register_longbench_v2_programs,
    register_mvp_programs,
)


POLICY_MODES = ["heuristic", "openai", "openrouter", "vllm", "ollama", "huggingface"]
_MCQL = ("A", "B", "C", "D")


def build_policy_config(args: argparse.Namespace) -> dict[str, object]:
    cfg: dict[str, object] = {
        "mode": args.policy_mode,
    }
    if args.model:
        cfg["model"] = args.model
    if args.api_base:
        cfg["api_base"] = args.api_base
    if args.api_key:
        cfg["api_key"] = args.api_key
    if args.openrouter_site_url:
        cfg["openrouter_site_url"] = args.openrouter_site_url
    if args.openrouter_app_name:
        cfg["openrouter_app_name"] = args.openrouter_app_name
    if args.hf_revision:
        cfg["hf_revision"] = args.hf_revision
    if args.hf_device_map:
        cfg["hf_device_map"] = args.hf_device_map
    if args.hf_torch_dtype:
        cfg["hf_torch_dtype"] = args.hf_torch_dtype
    if args.hf_max_new_tokens > 0:
        cfg["hf_max_new_tokens"] = args.hf_max_new_tokens
    return cfg


def build_runtime(
    manifest_path: Path,
    memory_dir: Path,
    *,
    require_official_mcp_sdk: bool = False,
    legacy_mcp: bool = False,
    extra_specs: List[MCPServerSpec] | None = None,
) -> tuple[MCPRLMRuntime, MultiServerMCPClient]:
    ctx_server = ROOT / "examples" / "run_context_server.py"
    analysis_server = ROOT / "examples" / "run_analysis_server.py"

    ctx_cmd = [sys.executable, str(ctx_server), "--manifest", str(manifest_path)]
    analysis_cmd = [sys.executable, str(analysis_server)]
    if legacy_mcp:
        ctx_cmd.append("--legacy-mcp")
        analysis_cmd.append("--legacy-mcp")
    if require_official_mcp_sdk:
        ctx_cmd.append("--require-official-sdk")
        analysis_cmd.append("--require-official-sdk")

    specs: List[MCPServerSpec] = [
        MCPServerSpec(
            alias="ctx",
            command=ctx_cmd,
            cwd=str(ROOT),
            max_concurrency=24,
            prefer_official_sdk=not legacy_mcp,
            strict_official_sdk=require_official_mcp_sdk,
        ),
        MCPServerSpec(
            alias="analysis",
            command=analysis_cmd,
            cwd=str(ROOT),
            max_concurrency=24,
            prefer_official_sdk=not legacy_mcp,
            strict_official_sdk=require_official_mcp_sdk,
        ),
    ]

    if extra_specs:
        specs.extend(extra_specs)

    multi_client = MultiServerMCPClient(specs=specs)

    registry = ProgramRegistry()
    register_builtin_programs(registry)
    register_mvp_programs(registry)
    register_longbench_v2_programs(registry)

    runtime = MCPRLMRuntime(
        program_registry=registry,
        mcp_client=multi_client,
        memory=FileSharedMemory(memory_dir),
        max_group_concurrency=64,
    )
    return runtime, multi_client


def extract_letter(text: str) -> str | None:
    raw = str(text or "").strip().upper()
    if not raw:
        return None
    patterns = [
        r"THE\s+CORRECT\s+ANSWER\s+IS\s*\(?([A-D])\)?",
        r"ANSWER\s*[:]\s*\(?([A-D])\)?",
        r"\(([A-D])\)",
        r"\b([A-D])\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, raw)
        if m:
            letter = str(m.group(1)).strip().upper()
            if letter in _MCQL:
                return letter
    return None


def load_records(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []

    if path.suffix.lower() == ".jsonl":
        out = []
        for line in text.splitlines():
            row = line.strip()
            if not row:
                continue
            out.append(json.loads(row))
        return out

    loaded = json.loads(text)
    if isinstance(loaded, list):
        return [row for row in loaded if isinstance(row, dict)]
    if isinstance(loaded, dict):
        return [loaded]
    raise ValueError(f"Unsupported dataset format in {path}")


def iter_selected(records: List[Dict[str, Any]], *, start: int, limit: int, ids: set[str]) -> Iterable[Dict[str, Any]]:
    sliced = records[start:]
    if ids:
        sliced = [row for row in sliced if str(row.get("_id", "")) in ids]
    if limit > 0:
        sliced = sliced[:limit]
    return sliced


async def run_one(
    item: Dict[str, Any],
    *,
    out_dir: Path,
    chunk_chars: int,
    overlap_chars: int,
    branch_factor: int,
    max_children: int,
    policy_config: Dict[str, Any],
    require_official_mcp_sdk: bool,
    legacy_mcp: bool,
    extra_specs: List[MCPServerSpec],
    root_extra_object_fanout: List[Dict[str, Any]],
    leaf_extra_object_fanout: List[Dict[str, Any]],
) -> Dict[str, Any]:
    item_id = str(item.get("_id", "unknown"))
    context = str(item.get("context", ""))
    question = str(item.get("question", "")).strip()

    if not question:
        raise ValueError(f"Missing question for item: {item_id}")

    choices = {
        "A": str(item.get("choice_A", "")).strip(),
        "B": str(item.get("choice_B", "")).strip(),
        "C": str(item.get("choice_C", "")).strip(),
        "D": str(item.get("choice_D", "")).strip(),
    }

    item_dir = out_dir / "runs" / item_id
    store_dir = item_dir / "context_store"
    memory_dir = item_dir / "memory"
    item_dir.mkdir(parents=True, exist_ok=True)

    context_file = item_dir / "context.txt"
    context_file.write_text(context, encoding="utf-8")

    manifest_path = preprocess_long_context(
        input_file=context_file,
        output_dir=store_dir,
        chunk_chars=chunk_chars,
        overlap_chars=overlap_chars,
        branch_factor=branch_factor,
    )

    runtime, mcp_client = build_runtime(
        manifest_path,
        memory_dir,
        require_official_mcp_sdk=require_official_mcp_sdk,
        legacy_mcp=legacy_mcp,
        extra_specs=extra_specs,
    )

    try:
        trace = await runtime.run_episode(
            goal=question,
            program="longbench_v2_root",
            input_payload={
                "question": question,
                "choices": choices,
                "manifest_path": str(manifest_path),
                "max_children": max_children,
                "policy_config": policy_config,
                "root_extra_object_fanout": root_extra_object_fanout,
                "leaf_extra_object_fanout": leaf_extra_object_fanout,
            },
        )
    finally:
        await mcp_client.close()

    output = trace.root_output if isinstance(trace.root_output, dict) else {}
    pred = str(output.get("pred", "")).strip().upper()
    if pred not in _MCQL:
        pred = extract_letter(str(output.get("response", ""))) or ""
    if pred not in _MCQL:
        pred = extract_letter(str(output.get("answer", ""))) or ""

    response = str(output.get("response", "")).strip()
    if not response and pred in _MCQL:
        response = f"The correct answer is ({pred})"

    answer = str(item.get("answer", "")).strip().upper()
    judge = pred == answer if pred in _MCQL else False

    row = {
        "_id": item_id,
        "domain": item.get("domain"),
        "sub_domain": item.get("sub_domain"),
        "difficulty": item.get("difficulty"),
        "length": item.get("length"),
        "question": question,
        "choice_A": choices["A"],
        "choice_B": choices["B"],
        "choice_C": choices["C"],
        "choice_D": choices["D"],
        "answer": answer,
        "response": response,
        "pred": pred if pred in _MCQL else None,
        "judge": bool(judge),
        "context": context[:1000],
        "mcp_rlm": {
            "episode_id": trace.episode_id,
            "success": bool(trace.success),
            "root_output": output,
            "manifest_path": str(manifest_path),
            "runtime_dir": str(item_dir),
        },
    }
    return row


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run LongBench v2 with MCP-RLM")
    parser.add_argument("--dataset-file", type=str, required=True, help="Path to LongBench v2 data.json or data.jsonl")
    parser.add_argument("--out", type=str, default="artifacts/longbench_v2")
    parser.add_argument("--result-file", type=str, default="", help="Optional output jsonl filename")

    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 means all")
    parser.add_argument("--ids", type=str, default="", help="Comma-separated _id filters")
    parser.add_argument("--resume", action="store_true", help="Skip items already in result file")

    parser.add_argument("--chunk-chars", type=int, default=16000)
    parser.add_argument("--overlap-chars", type=int, default=400)
    parser.add_argument("--branch-factor", type=int, default=8)
    parser.add_argument("--max-children", type=int, default=16)

    parser.add_argument("--policy-mode", type=str, default="heuristic", choices=POLICY_MODES)
    parser.add_argument("--model", type=str, default="", help="Model id/name for openai-compatible or huggingface mode")
    parser.add_argument("--api-base", type=str, default="", help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", type=str, default="", help="API key for remote providers")
    parser.add_argument("--openrouter-site-url", type=str, default="", help="Optional OpenRouter HTTP-Referer")
    parser.add_argument("--openrouter-app-name", type=str, default="mcp-rlm", help="Optional OpenRouter X-Title")

    parser.add_argument("--hf-revision", type=str, default="", help="Optional HuggingFace revision")
    parser.add_argument("--hf-device-map", type=str, default="auto", help="HuggingFace device_map")
    parser.add_argument("--hf-torch-dtype", type=str, default="auto", help="HuggingFace torch_dtype")
    parser.add_argument("--hf-max-new-tokens", type=int, default=256, help="HuggingFace max generation tokens")

    parser.add_argument("--legacy-mcp", action="store_true", help="Use legacy JSON-RPC transport instead of official MCP SDK")
    parser.add_argument("--require-official-mcp-sdk", action="store_true", help="Fail fast if official MCP SDK cannot be used")

    parser.add_argument("--mcp-server-config", type=str, default="", help="JSON file with extra MCP server specs and optional root/leaf extra fan-out calls")
    parser.add_argument("--enable-official-mcp-presets", action="store_true", help="Enable official MCP server presets (filesystem/memory/fetch/git/sequential-thinking)")
    parser.add_argument("--skip-unavailable-extra-servers", action="store_true", help="Skip extra servers whose executable is not found")

    args = parser.parse_args()

    dataset_path = Path(args.dataset_file).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    out_dir = (ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    result_file = Path(args.result_file).resolve() if args.result_file else (out_dir / "results.jsonl")
    result_file.parent.mkdir(parents=True, exist_ok=True)

    records = load_records(dataset_path)
    id_filter = {x.strip() for x in args.ids.split(",") if x.strip()}
    selected = list(iter_selected(records, start=max(0, args.start_index), limit=max(0, args.limit), ids=id_filter))

    done_ids: set[str] = set()
    if args.resume and result_file.exists():
        with result_file.open("r", encoding="utf-8") as f:
            for line in f:
                row = line.strip()
                if not row:
                    continue
                try:
                    parsed = json.loads(row)
                    done_ids.add(str(parsed.get("_id", "")))
                except Exception:
                    continue

    policy_config = build_policy_config(args)

    extra_specs, fanout_cfg = load_mcp_extension_config(
        workspace_root=str(ROOT),
        config_path=args.mcp_server_config,
        enable_official_presets=bool(args.enable_official_mcp_presets),
        skip_unavailable=bool(args.skip_unavailable_extra_servers),
    )
    root_extra_object_fanout = fanout_cfg.get("root_extra_object_fanout", []) if isinstance(fanout_cfg, dict) else []
    leaf_extra_object_fanout = fanout_cfg.get("leaf_extra_object_fanout", []) if isinstance(fanout_cfg, dict) else []

    total = 0
    correct = 0
    with result_file.open("a", encoding="utf-8") as fout:
        for idx, item in enumerate(selected, start=1):
            item_id = str(item.get("_id", ""))
            if args.resume and item_id in done_ids:
                continue

            row = await run_one(
                item,
                out_dir=out_dir,
                chunk_chars=args.chunk_chars,
                overlap_chars=args.overlap_chars,
                branch_factor=args.branch_factor,
                max_children=args.max_children,
                policy_config=policy_config,
                require_official_mcp_sdk=bool(args.require_official_mcp_sdk),
                legacy_mcp=bool(args.legacy_mcp),
                extra_specs=extra_specs,
                root_extra_object_fanout=root_extra_object_fanout,
                leaf_extra_object_fanout=leaf_extra_object_fanout,
            )
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()

            total += 1
            correct += int(bool(row.get("judge")))
            running = 0.0 if total <= 0 else (100.0 * correct / total)
            print(f"[{idx}/{len(selected)}] _id={item_id} pred={row.get('pred')} gold={row.get('answer')} judge={row.get('judge')} acc={running:.2f}%")

    print("Result file:", result_file)
    print("Extra MCP servers:", [spec.alias for spec in extra_specs])
    if total > 0:
        print("Processed:", total)
        print("Accuracy:", round(100.0 * correct / total, 3))


if __name__ == "__main__":
    asyncio.run(main())
