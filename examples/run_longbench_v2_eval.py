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
    register_llm_manager_programs,
    register_mvp_programs,
)
from mcp_rlm.training import export_trace
from mcp_rlm.types import Budget


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
    if args.hf_chat_timeout_seconds > 0:
        cfg["hf_chat_timeout_seconds"] = float(args.hf_chat_timeout_seconds)
    if args.hf_load_timeout_seconds > 0:
        cfg["hf_load_timeout_seconds"] = float(args.hf_load_timeout_seconds)
    if args.hf_generate_timeout_seconds > 0:
        cfg["hf_generate_timeout_seconds"] = float(args.hf_generate_timeout_seconds)
    if args.request_timeout_seconds > 0:
        cfg["request_timeout_seconds"] = float(args.request_timeout_seconds)
    cfg["hf_use_worker_process"] = bool(args.hf_use_worker_process)
    if args.hf_worker_module:
        cfg["hf_worker_module"] = args.hf_worker_module
    return cfg


def build_runtime(
    manifest_path: Path,
    memory_dir: Path,
    *,
    require_official_mcp_sdk: bool = False,
    legacy_mcp: bool = False,
    extra_specs: List[MCPServerSpec] | None = None,
    group_max_wall_seconds: float = 120.0,
    enable_llm_manager: bool = False,
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
    if enable_llm_manager:
        register_llm_manager_programs(registry)

    runtime = MCPRLMRuntime(
        program_registry=registry,
        mcp_client=multi_client,
        memory=FileSharedMemory(memory_dir),
        max_group_concurrency=64,
        default_group_budget=Budget(max_wall_seconds=max(10.0, float(group_max_wall_seconds))),
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
    track: str,
    chunk_chars: int,
    overlap_chars: int,
    branch_factor: int,
    max_children: int,
    group_max_wall_seconds: float,
    policy_config: Dict[str, Any],
    prompt_style: str,
    longbench_prompt_dir: str,
    manager_max_turns: int,
    manager_max_history: int,
    manager_list_objects_timeout_seconds: float,
    manager_policy_chat_timeout_seconds: float,
    manager_policy_mode: str,
    manager_policy_model: str,
    manager_policy_api_base: str,
    manager_policy_api_key: str,
    manager_action_max_new_tokens: int,
    manager_finalize_max_new_tokens: int,
    manager_json_mode: str,
    manager_json_retry: int,
    manager_json_repair: bool,
    default_child_program: str,
    manager_system_prompt: str,
    manager_finalize_system_prompt: str,
    export_trace_episode: bool,
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

    normalized_track = str(track).strip().lower()
    if normalized_track not in {"mvp", "llm_manager"}:
        raise ValueError(f"Unsupported track: {track}")

    runtime, mcp_client = build_runtime(
        manifest_path,
        memory_dir,
        require_official_mcp_sdk=require_official_mcp_sdk,
        legacy_mcp=legacy_mcp,
        extra_specs=extra_specs,
        group_max_wall_seconds=group_max_wall_seconds,
        enable_llm_manager=(normalized_track == "llm_manager"),
    )

    if normalized_track == "llm_manager":
        program = "llm_managed_root"
        payload = {
            "query": question,
            "question": question,
            "choices": choices,
            "manifest_path": str(manifest_path),
            "policy_config": policy_config,
            "manager_max_turns": max(1, int(manager_max_turns)),
            "manager_max_history": max(1, int(manager_max_history)),
            "manager_list_objects_timeout_seconds": max(0.1, float(manager_list_objects_timeout_seconds)),
            "manager_policy_chat_timeout_seconds": max(0.1, float(manager_policy_chat_timeout_seconds)),
            "manager_action_max_new_tokens": max(8, int(manager_action_max_new_tokens)),
            "manager_finalize_max_new_tokens": max(8, int(manager_finalize_max_new_tokens)),
            "manager_json_mode": str(manager_json_mode).strip().lower(),
            "manager_json_retry": max(0, int(manager_json_retry)),
            "manager_json_repair": bool(manager_json_repair),
            "default_child_program": default_child_program,
            "prompt_style": prompt_style,
            "longbench_prompt_dir": longbench_prompt_dir,
            "root_extra_object_fanout": root_extra_object_fanout,
            "leaf_extra_object_fanout": leaf_extra_object_fanout,
        }
        if manager_policy_mode.strip():
            payload["manager_policy_mode"] = manager_policy_mode.strip()
        if manager_policy_model.strip():
            payload["manager_policy_model"] = manager_policy_model.strip()
        if manager_policy_api_base.strip():
            payload["manager_policy_api_base"] = manager_policy_api_base.strip()
        if manager_policy_api_key.strip():
            payload["manager_policy_api_key"] = manager_policy_api_key.strip()
        if manager_system_prompt.strip():
            payload["manager_system_prompt"] = manager_system_prompt
        if manager_finalize_system_prompt.strip():
            payload["manager_finalize_system_prompt"] = manager_finalize_system_prompt
    else:
        program = "longbench_v2_root"
        payload = {
            "question": question,
            "choices": choices,
            "manifest_path": str(manifest_path),
            "max_children": max_children,
            "policy_config": policy_config,
            "prompt_style": prompt_style,
            "longbench_prompt_dir": longbench_prompt_dir,
            "root_extra_object_fanout": root_extra_object_fanout,
            "leaf_extra_object_fanout": leaf_extra_object_fanout,
        }

    try:
        trace = await runtime.run_episode(
            goal=question,
            program=program,
            input_payload=payload,
        )
    finally:
        await mcp_client.close()

    if export_trace_episode:
        export_trace(trace, item_dir / "trace")

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
            "track": normalized_track,
            "program": program,
            "root_output": output,
            "manifest_path": str(manifest_path),
            "runtime_dir": str(item_dir),
        },
    }
    return row


def build_error_row(item: Dict[str, Any], *, error: BaseException, stage: str) -> Dict[str, Any]:
    item_id = str(item.get("_id", "unknown"))
    question = str(item.get("question", "")).strip()
    choices = {
        "A": str(item.get("choice_A", "")).strip(),
        "B": str(item.get("choice_B", "")).strip(),
        "C": str(item.get("choice_C", "")).strip(),
        "D": str(item.get("choice_D", "")).strip(),
    }
    answer = str(item.get("answer", "")).strip().upper()
    context = str(item.get("context", ""))

    return {
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
        "response": "",
        "pred": None,
        "judge": False,
        "context": context[:1000],
        "mcp_rlm": {
            "success": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "stage": stage,
        },
    }


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
    parser.add_argument("--group-max-wall-seconds", type=float, default=480.0, help="Per-group wall-clock budget in seconds")

    parser.add_argument("--track", type=str, default="mvp", choices=["mvp", "llm_manager"], help="Run fixed-program MVP track or LLM-managed root track")
    parser.add_argument("--prompt-style", type=str, default="hybrid", choices=["internal", "0shot", "0shot_cot", "hybrid"], help="Prompt style for LongBench answer formatting")
    parser.add_argument("--longbench-prompt-dir", type=str, default="", help="Optional directory containing official LongBench prompts (0shot.txt / 0shot_cot.txt)")
    parser.add_argument("--export-trace", action="store_true", help="Export full EpisodeTrace jsonl files per sample")

    parser.add_argument("--manager-max-turns", type=int, default=56)
    parser.add_argument("--manager-max-history", type=int, default=10)
    parser.add_argument("--manager-list-objects-timeout-seconds", type=float, default=20.0, help="Timeout for manager list_objects stage")
    parser.add_argument("--manager-policy-chat-timeout-seconds", type=float, default=120.0, help="Timeout for each manager policy chat call")
    parser.add_argument("--manager-policy-mode", type=str, default="", help="Optional manager-only policy mode override")
    parser.add_argument("--manager-policy-model", type=str, default="", help="Optional manager-only model override")
    parser.add_argument("--manager-policy-api-base", type=str, default="", help="Optional manager-only OpenAI-compatible API base")
    parser.add_argument("--manager-policy-api-key", type=str, default="", help="Optional manager-only API key override")
    parser.add_argument("--manager-action-max-new-tokens", type=int, default=96, help="Max new tokens for each manager action turn")
    parser.add_argument("--manager-finalize-max-new-tokens", type=int, default=160, help="Max new tokens for manager finalize call")
    parser.add_argument("--manager-json-mode", type=str, default="json_object", choices=["none", "json_object", "json_schema"], help="JSON constraint mode for manager calls")
    parser.add_argument("--manager-json-retry", type=int, default=1, help="Retries for manager JSON parse/generation errors")
    parser.add_argument("--manager-json-repair", action=argparse.BooleanOptionalAction, default=True, help="Enable JSON repair fallback for manager outputs")
    parser.add_argument("--default-child-program", type=str, default="llm_managed_child")
    parser.add_argument("--manager-system-prompt-file", type=str, default="")
    parser.add_argument("--manager-finalize-system-prompt-file", type=str, default="")

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
    parser.add_argument("--hf-chat-timeout-seconds", type=float, default=120.0, help="HF policy total chat timeout (kept for backward compatibility)")
    parser.add_argument("--hf-load-timeout-seconds", type=float, default=1800.0, help="HF model/pipeline load timeout")
    parser.add_argument("--hf-generate-timeout-seconds", type=float, default=120.0, help="HF generation timeout per chat call")
    parser.add_argument("--hf-use-worker-process", action=argparse.BooleanOptionalAction, default=True, help="Use subprocess worker for HF generation (recommended)")
    parser.add_argument("--hf-worker-module", type=str, default="mcp_rlm.hf_worker", help="Worker module path for HF subprocess mode")
    parser.add_argument("--request-timeout-seconds", type=float, default=25.0, help="HTTP timeout for OpenAI-compatible policy calls")

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

    manager_system_prompt = (
        "You are the root execution manager for LongBench-v2 in MCP-RLM. "
        "Use MCP objects to retrieve evidence, score choices, and finalize one option. "
        "Use call_many and spawn_groups when useful. "
        "When finalizing, output JSON with fields pred, response, confidence, and ensure response uses: "
        "The correct answer is (X)."
    )
    manager_finalize_system_prompt = (
        "Finalize LongBench-v2 output. Return ONLY JSON object with keys pred,response,confidence. "
        "response must follow exact format: The correct answer is (X)."
    )
    if args.manager_system_prompt_file:
        manager_system_prompt = Path(args.manager_system_prompt_file).resolve().read_text(encoding="utf-8")
    if args.manager_finalize_system_prompt_file:
        manager_finalize_system_prompt = Path(args.manager_finalize_system_prompt_file).resolve().read_text(encoding="utf-8")

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

            try:
                row = await run_one(
                    item,
                    out_dir=out_dir,
                    track=args.track,
                    chunk_chars=args.chunk_chars,
                    overlap_chars=args.overlap_chars,
                    branch_factor=args.branch_factor,
                    max_children=args.max_children,
                    group_max_wall_seconds=args.group_max_wall_seconds,
                    policy_config=policy_config,
                    prompt_style=args.prompt_style,
                    longbench_prompt_dir=args.longbench_prompt_dir,
                    manager_max_turns=args.manager_max_turns,
                    manager_max_history=args.manager_max_history,
                    manager_list_objects_timeout_seconds=args.manager_list_objects_timeout_seconds,
                    manager_policy_chat_timeout_seconds=args.manager_policy_chat_timeout_seconds,
                    manager_policy_mode=args.manager_policy_mode,
                    manager_policy_model=args.manager_policy_model,
                    manager_policy_api_base=args.manager_policy_api_base,
                    manager_policy_api_key=args.manager_policy_api_key,
                    manager_action_max_new_tokens=args.manager_action_max_new_tokens,
                    manager_finalize_max_new_tokens=args.manager_finalize_max_new_tokens,
                    manager_json_mode=args.manager_json_mode,
                    manager_json_retry=args.manager_json_retry,
                    manager_json_repair=bool(args.manager_json_repair),
                    default_child_program=args.default_child_program,
                    manager_system_prompt=manager_system_prompt,
                    manager_finalize_system_prompt=manager_finalize_system_prompt,
                    export_trace_episode=bool(args.export_trace),
                    require_official_mcp_sdk=bool(args.require_official_mcp_sdk),
                    legacy_mcp=bool(args.legacy_mcp),
                    extra_specs=extra_specs,
                    root_extra_object_fanout=root_extra_object_fanout,
                    leaf_extra_object_fanout=leaf_extra_object_fanout,
                )
            except asyncio.CancelledError as exc:
                row = build_error_row(item, error=exc, stage="run_one_cancelled")
            except Exception as exc:
                row = build_error_row(item, error=exc, stage="run_one_exception")

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()

            total += 1
            correct += int(bool(row.get("judge")))
            running = 0.0 if total <= 0 else (100.0 * correct / total)
            if row.get("pred") is None and isinstance(row.get("mcp_rlm"), dict) and row["mcp_rlm"].get("error"):
                etype = row["mcp_rlm"].get("error_type")
                print(f"[{idx}/{len(selected)}] _id={item_id} ERROR={etype} acc={running:.2f}%")
            else:
                print(f"[{idx}/{len(selected)}] _id={item_id} pred={row.get('pred')} gold={row.get('answer')} judge={row.get('judge')} acc={running:.2f}%")

    print("Result file:", result_file)
    print("Extra MCP servers:", [spec.alias for spec in extra_specs])
    if total > 0:
        print("Processed:", total)
        print("Accuracy:", round(100.0 * correct / total, 3))


if __name__ == "__main__":
    asyncio.run(main())
