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
    register_llm_manager_programs,
    register_longbench_v2_programs,
    register_mvp_programs,
)
from mcp_rlm.types import Budget


POLICY_MODES = ["heuristic", "openai", "openrouter", "vllm", "ollama", "huggingface"]
_MCQL = ("A", "B", "C", "D")


def build_policy_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
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
    group_max_wall_seconds: float = 600.0,
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
        match = re.search(pattern, raw)
        if match:
            letter = str(match.group(1)).strip().upper()
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
            parsed = json.loads(row)
            if isinstance(parsed, dict):
                out.append(parsed)
        return out

    loaded = json.loads(text)
    if isinstance(loaded, list):
        return [row for row in loaded if isinstance(row, dict)]
    if isinstance(loaded, dict):
        return [loaded]
    raise ValueError(f"Unsupported dataset format in {path}")


def iter_selected(records: List[Dict[str, Any]], *, start: int, limit: int, ids: set[str], id_field: str) -> Iterable[Dict[str, Any]]:
    sliced = records[start:]
    if ids:
        sliced = [row for row in sliced if str(row.get(id_field, "")) in ids]
    if limit > 0:
        sliced = sliced[:limit]
    return sliced


def _choices_from_item(item: Dict[str, Any], choice_fields: List[str]) -> Dict[str, str]:
    labels = ["A", "B", "C", "D"]
    choices: Dict[str, str] = {}
    for idx, field in enumerate(choice_fields[:4]):
        letter = labels[idx]
        value = str(item.get(field, "")).strip()
        if value:
            choices[letter] = value
    return choices


async def run_one(
    item: Dict[str, Any],
    *,
    out_dir: Path,
    id_field: str,
    query_field: str,
    context_field: str,
    answer_field: str,
    choice_fields: List[str],
    program: str,
    manager_max_turns: int,
    manager_max_history: int,
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
    chunk_chars: int,
    overlap_chars: int,
    branch_factor: int,
    policy_config: Dict[str, Any],
    require_official_mcp_sdk: bool,
    legacy_mcp: bool,
    extra_specs: List[MCPServerSpec],
    group_max_wall_seconds: float,
) -> Dict[str, Any]:
    item_id = str(item.get(id_field, "unknown"))
    query = str(item.get(query_field, "")).strip()
    context = str(item.get(context_field, ""))
    answer = str(item.get(answer_field, "")).strip().upper() if answer_field else ""
    choices = _choices_from_item(item, choice_fields)

    if not query:
        raise ValueError(f"Missing query field '{query_field}' for item: {item_id}")

    item_dir = out_dir / "runs" / item_id
    store_dir = item_dir / "context_store"
    memory_dir = item_dir / "memory"
    trace_dir = item_dir / "trace"
    item_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)

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
        group_max_wall_seconds=group_max_wall_seconds,
    )

    try:
        payload = {
            "query": query,
            "question": query,
            "choices": choices,
            "manifest_path": str(manifest_path),
            "policy_config": dict(policy_config),
            "manager_max_turns": manager_max_turns,
            "manager_max_history": manager_max_history,
            "manager_action_max_new_tokens": max(8, int(manager_action_max_new_tokens)),
            "manager_finalize_max_new_tokens": max(8, int(manager_finalize_max_new_tokens)),
            "manager_json_mode": str(manager_json_mode).strip().lower(),
            "manager_json_retry": max(0, int(manager_json_retry)),
            "manager_json_repair": bool(manager_json_repair),
            "default_child_program": default_child_program,
            "manager_system_prompt": manager_system_prompt,
            "manager_finalize_system_prompt": manager_finalize_system_prompt,
            "runtime_dir": str(item_dir),
            "manager_events_log_path": str(trace_dir / "events.jsonl"),
        }
        if manager_policy_mode.strip():
            payload["manager_policy_mode"] = manager_policy_mode.strip()
        if manager_policy_model.strip():
            payload["manager_policy_model"] = manager_policy_model.strip()
        if manager_policy_api_base.strip():
            payload["manager_policy_api_base"] = manager_policy_api_base.strip()
        if manager_policy_api_key.strip():
            payload["manager_policy_api_key"] = manager_policy_api_key.strip()

        trace = await runtime.run_episode(
            goal=query,
            program=program,
            input_payload=payload,
        )
    finally:
        await mcp_client.close()

    output = trace.root_output if isinstance(trace.root_output, dict) else {"output": trace.root_output}
    root_group = next((g for g in trace.groups if g.group_id == trace.root_group_id), None)
    pred = str(output.get("pred", "")).strip().upper()
    if pred not in _MCQL:
        pred = extract_letter(str(output.get("response", ""))) or ""
    if pred not in _MCQL:
        pred = extract_letter(str(output.get("answer", ""))) or ""

    response = str(output.get("response", "")).strip()
    if not response and pred in _MCQL:
        response = f"The correct answer is ({pred})"

    judge = bool(pred == answer) if answer and pred in _MCQL else False

    return {
        "_id": item_id,
        "query": query,
        "answer": answer or None,
        "pred": pred if pred in _MCQL else None,
        "judge": judge,
        "response": response,
        "choices": choices,
        "mcp_rlm": {
            "episode_id": trace.episode_id,
            "success": bool(trace.success),
            "program": program,
            "root_output": output,
            "root_group_status": (root_group.status.value if root_group is not None else None),
            "root_group_error": (root_group.error if root_group is not None else None),
            "manifest_path": str(manifest_path),
            "runtime_dir": str(item_dir),
        },
    }


def build_error_row(item: Dict[str, Any], *, id_field: str, query_field: str, answer_field: str, error: BaseException, stage: str) -> Dict[str, Any]:
    item_id = str(item.get(id_field, "unknown"))
    query = str(item.get(query_field, "")).strip()
    answer = str(item.get(answer_field, "")).strip().upper() if answer_field else ""
    return {
        "_id": item_id,
        "query": query,
        "answer": answer or None,
        "pred": None,
        "judge": False,
        "response": "",
        "mcp_rlm": {
            "success": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "stage": stage,
        },
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run external dataset with LLM-managed root MCP-RLM")
    parser.add_argument("--dataset-file", type=str, required=True, help="Path to dataset (.json or .jsonl)")
    parser.add_argument("--out", type=str, default="artifacts/llm_manager_eval")
    parser.add_argument("--result-file", type=str, default="", help="Optional output jsonl filename")

    parser.add_argument("--id-field", type=str, default="_id")
    parser.add_argument("--query-field", type=str, default="question")
    parser.add_argument("--context-field", type=str, default="context")
    parser.add_argument("--answer-field", type=str, default="answer")
    parser.add_argument("--choice-fields", type=str, default="choice_A,choice_B,choice_C,choice_D")

    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 means all")
    parser.add_argument("--ids", type=str, default="", help="Comma-separated _id filters")
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--chunk-chars", type=int, default=16000)
    parser.add_argument("--overlap-chars", type=int, default=400)
    parser.add_argument("--branch-factor", type=int, default=8)
    parser.add_argument("--group-max-wall-seconds", type=float, default=900.0)

    parser.add_argument("--program", type=str, default="llm_managed_root")
    parser.add_argument("--manager-max-turns", type=int, default=48)
    parser.add_argument("--manager-max-history", type=int, default=8)
    parser.add_argument("--manager-policy-mode", type=str, default="", help="Optional manager-only policy mode override")
    parser.add_argument("--manager-policy-model", type=str, default="", help="Optional manager-only model override")
    parser.add_argument("--manager-policy-api-base", type=str, default="", help="Optional manager-only OpenAI-compatible API base")
    parser.add_argument("--manager-policy-api-key", type=str, default="", help="Optional manager-only API key override")
    parser.add_argument("--manager-action-max-new-tokens", type=int, default=96)
    parser.add_argument("--manager-finalize-max-new-tokens", type=int, default=160)
    parser.add_argument("--manager-json-mode", type=str, default="json_schema", choices=["none", "json_object", "json_schema"])
    parser.add_argument("--manager-json-retry", type=int, default=1)
    parser.add_argument("--manager-json-repair", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--default-child-program", type=str, default="llm_managed_child")
    parser.add_argument("--manager-system-prompt-file", type=str, default="")
    parser.add_argument("--manager-finalize-system-prompt-file", type=str, default="")

    parser.add_argument("--policy-mode", type=str, default="heuristic", choices=POLICY_MODES)
    parser.add_argument("--model", type=str, default="", help="Model id/name")
    parser.add_argument("--api-base", type=str, default="", help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", type=str, default="", help="API key")
    parser.add_argument("--openrouter-site-url", type=str, default="")
    parser.add_argument("--openrouter-app-name", type=str, default="mcp-rlm")

    parser.add_argument("--hf-revision", type=str, default="")
    parser.add_argument("--hf-device-map", type=str, default="auto")
    parser.add_argument("--hf-torch-dtype", type=str, default="auto")
    parser.add_argument("--hf-max-new-tokens", type=int, default=256)
    parser.add_argument("--hf-chat-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--hf-load-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--hf-generate-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--hf-use-worker-process", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hf-worker-module", type=str, default="mcp_rlm.hf_worker")
    parser.add_argument("--request-timeout-seconds", type=float, default=25.0)

    parser.add_argument("--legacy-mcp", action="store_true")
    parser.add_argument("--require-official-mcp-sdk", action="store_true")

    parser.add_argument("--mcp-server-config", type=str, default="")
    parser.add_argument("--enable-official-mcp-presets", action="store_true")
    parser.add_argument("--skip-unavailable-extra-servers", action="store_true")

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
    selected = list(
        iter_selected(
            records,
            start=max(0, args.start_index),
            limit=max(0, args.limit),
            ids=id_filter,
            id_field=args.id_field,
        )
    )

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

    extra_specs, _ = load_mcp_extension_config(
        workspace_root=str(ROOT),
        config_path=args.mcp_server_config,
        enable_official_presets=bool(args.enable_official_mcp_presets),
        skip_unavailable=bool(args.skip_unavailable_extra_servers),
    )

    manager_system_prompt = ""
    if args.manager_system_prompt_file:
        manager_system_prompt = Path(args.manager_system_prompt_file).resolve().read_text(encoding="utf-8")

    manager_finalize_system_prompt = ""
    if args.manager_finalize_system_prompt_file:
        manager_finalize_system_prompt = Path(args.manager_finalize_system_prompt_file).resolve().read_text(encoding="utf-8")

    choice_fields = [x.strip() for x in args.choice_fields.split(",") if x.strip()]

    total = 0
    correct = 0
    with result_file.open("a", encoding="utf-8") as fout:
        for idx, item in enumerate(selected, start=1):
            item_id = str(item.get(args.id_field, ""))
            if args.resume and item_id in done_ids:
                continue

            try:
                row = await run_one(
                    item,
                    out_dir=out_dir,
                    id_field=args.id_field,
                    query_field=args.query_field,
                    context_field=args.context_field,
                    answer_field=args.answer_field,
                    choice_fields=choice_fields,
                    program=args.program,
                    manager_max_turns=max(1, int(args.manager_max_turns)),
                    manager_max_history=max(1, int(args.manager_max_history)),
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
                    chunk_chars=args.chunk_chars,
                    overlap_chars=args.overlap_chars,
                    branch_factor=args.branch_factor,
                    policy_config=policy_config,
                    require_official_mcp_sdk=bool(args.require_official_mcp_sdk),
                    legacy_mcp=bool(args.legacy_mcp),
                    extra_specs=extra_specs,
                    group_max_wall_seconds=args.group_max_wall_seconds,
                )
            except asyncio.CancelledError as exc:
                row = build_error_row(
                    item,
                    id_field=args.id_field,
                    query_field=args.query_field,
                    answer_field=args.answer_field,
                    error=exc,
                    stage="run_one_cancelled",
                )
            except Exception as exc:
                row = build_error_row(
                    item,
                    id_field=args.id_field,
                    query_field=args.query_field,
                    answer_field=args.answer_field,
                    error=exc,
                    stage="run_one_exception",
                )

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()

            total += 1
            correct += int(bool(row.get("judge")))
            running = 0.0 if total <= 0 else (100.0 * correct / total)
            if row.get("pred") is None and isinstance(row.get("mcp_rlm"), dict) and row["mcp_rlm"].get("error"):
                etype = row["mcp_rlm"].get("error_type")
                print(f"[{idx}/{len(selected)}] _id={item_id} ERROR={etype} acc={running:.2f}%")
            else:
                print(
                    f"[{idx}/{len(selected)}] _id={item_id} pred={row.get('pred')} "
                    f"gold={row.get('answer')} judge={row.get('judge')} acc={running:.2f}%"
                )

    print("Result file:", result_file)
    print("Extra MCP servers:", [spec.alias for spec in extra_specs])
    if total > 0:
        print("Processed:", total)
        print("Accuracy:", round(100.0 * correct / total, 3))


if __name__ == "__main__":
    asyncio.run(main())
