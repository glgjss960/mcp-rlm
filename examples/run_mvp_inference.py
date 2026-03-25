from __future__ import annotations

from pathlib import Path
import argparse
import asyncio
import json
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
    register_builtin_programs,
    register_mvp_programs,
)
from mcp_rlm.training import export_agentic_rl, export_cold_start, export_openrlhf, export_trace, export_verl


POLICY_MODES = ['heuristic', 'openai', 'openrouter', 'vllm', 'ollama', 'huggingface']


def build_runtime(
    manifest_path: Path,
    memory_dir: Path,
    *,
    require_official_mcp_sdk: bool = False,
    legacy_mcp: bool = False,
    extra_specs: list[MCPServerSpec] | None = None,
) -> tuple[MCPRLMRuntime, MultiServerMCPClient]:
    ctx_server = ROOT / 'examples' / 'run_context_server.py'
    analysis_server = ROOT / 'examples' / 'run_analysis_server.py'

    ctx_cmd = [sys.executable, str(ctx_server), '--manifest', str(manifest_path)]
    analysis_cmd = [sys.executable, str(analysis_server)]
    if legacy_mcp:
        ctx_cmd.append('--legacy-mcp')
        analysis_cmd.append('--legacy-mcp')
    if require_official_mcp_sdk:
        ctx_cmd.append('--require-official-sdk')
        analysis_cmd.append('--require-official-sdk')

    specs: list[MCPServerSpec] = [
        MCPServerSpec(
            alias='ctx',
            command=ctx_cmd,
            cwd=str(ROOT),
            max_concurrency=24,
            prefer_official_sdk=not legacy_mcp,
            strict_official_sdk=require_official_mcp_sdk,
        ),
        MCPServerSpec(
            alias='analysis',
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

    runtime = MCPRLMRuntime(
        program_registry=registry,
        mcp_client=multi_client,
        memory=FileSharedMemory(memory_dir),
        max_group_concurrency=64,
    )
    return runtime, multi_client


def build_policy_config(args: argparse.Namespace) -> dict[str, object]:
    cfg: dict[str, object] = {
        'mode': args.policy_mode,
    }
    if args.model:
        cfg['model'] = args.model
    if args.api_base:
        cfg['api_base'] = args.api_base
    if args.api_key:
        cfg['api_key'] = args.api_key
    if args.openrouter_site_url:
        cfg['openrouter_site_url'] = args.openrouter_site_url
    if args.openrouter_app_name:
        cfg['openrouter_app_name'] = args.openrouter_app_name
    if args.hf_revision:
        cfg['hf_revision'] = args.hf_revision
    if args.hf_device_map:
        cfg['hf_device_map'] = args.hf_device_map
    if args.hf_torch_dtype:
        cfg['hf_torch_dtype'] = args.hf_torch_dtype
    if args.hf_max_new_tokens > 0:
        cfg['hf_max_new_tokens'] = args.hf_max_new_tokens
    return cfg


def resolve_query(args: argparse.Namespace) -> str:
    if args.query:
        query = str(args.query).strip()
        if query:
            return query
    if args.query_file:
        path = Path(args.query_file).resolve()
        if not path.exists():
            raise FileNotFoundError(f'Query file not found: {path}')
        text = path.read_text(encoding='utf-8').replace('\ufeff', '', 1).strip()
        if text:
            return text
        raise ValueError(f'Query file is empty: {path}')
    raise ValueError('Provide either --query or --query-file')


async def main() -> None:
    parser = argparse.ArgumentParser(description='Run MCP-RLM MVP inference with multi-server MCP orchestration')
    parser.add_argument('--manifest', type=str, required=True, help='Path to context manifest.json')

    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument('--query', type=str, default='', help='Query text')
    query_group.add_argument('--query-file', type=str, default='', help='Path to UTF-8 text file containing query')

    parser.add_argument('--out', type=str, default='artifacts/mvp')
    parser.add_argument('--max-children', type=int, default=16)

    parser.add_argument('--policy-mode', type=str, default='heuristic', choices=POLICY_MODES)
    parser.add_argument('--model', type=str, default='', help='Model id/name for openai-compatible or huggingface mode')
    parser.add_argument('--api-base', type=str, default='', help='OpenAI-compatible base URL')
    parser.add_argument('--api-key', type=str, default='', help='API key for remote providers')
    parser.add_argument('--openrouter-site-url', type=str, default='', help='Optional OpenRouter HTTP-Referer')
    parser.add_argument('--openrouter-app-name', type=str, default='mcp-rlm', help='Optional OpenRouter X-Title')

    parser.add_argument('--hf-revision', type=str, default='', help='Optional HuggingFace revision')
    parser.add_argument('--hf-device-map', type=str, default='auto', help='HuggingFace device_map')
    parser.add_argument('--hf-torch-dtype', type=str, default='auto', help='HuggingFace torch_dtype')
    parser.add_argument('--hf-max-new-tokens', type=int, default=256, help='HuggingFace max generation tokens')

    parser.add_argument('--legacy-mcp', action='store_true', help='Use legacy JSON-RPC transport instead of official MCP SDK')
    parser.add_argument('--require-official-mcp-sdk', action='store_true', help='Fail fast if official MCP SDK cannot be used')

    parser.add_argument('--mcp-server-config', type=str, default='', help='JSON file with extra MCP server specs and optional fan-out call templates')
    parser.add_argument('--enable-official-mcp-presets', action='store_true', help='Enable official MCP server presets (filesystem/memory/fetch/git/sequential-thinking)')
    parser.add_argument('--skip-unavailable-extra-servers', action='store_true', help='Skip extra servers whose executable is unavailable')
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f'Manifest not found: {manifest_path}')

    query = resolve_query(args)

    out_dir = (ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    memory_dir = out_dir / 'memory'

    extra_specs, _ = load_mcp_extension_config(
        workspace_root=str(ROOT),
        config_path=args.mcp_server_config,
        enable_official_presets=bool(args.enable_official_mcp_presets),
        skip_unavailable=bool(args.skip_unavailable_extra_servers),
    )

    runtime, multi_client = build_runtime(
        manifest_path,
        memory_dir,
        require_official_mcp_sdk=bool(args.require_official_mcp_sdk),
        legacy_mcp=bool(args.legacy_mcp),
        extra_specs=extra_specs,
    )
    policy_config = build_policy_config(args)

    try:
        trace = await runtime.run_episode(
            goal=query,
            program='mvp_root',
            input_payload={
                'query': query,
                'manifest_path': str(manifest_path),
                'max_children': args.max_children,
                'policy_config': policy_config,
            },
        )

        export_trace(trace, out_dir)
        export_cold_start([trace], out_dir / 'cold_start_turns.jsonl')
        export_agentic_rl([trace], out_dir / 'agentic_rl.jsonl')
        export_verl([trace], out_dir / 'verl_warm_start.jsonl')
        export_openrlhf([trace], out_dir / 'openrlhf_episodes.jsonl')

        print('Episode:', trace.episode_id)
        print('Success:', trace.success)
        print('Policy mode:', args.policy_mode)
        if args.model:
            print('Model:', args.model)
        print('Extra MCP servers:', [spec.alias for spec in extra_specs])
        print('Query:', query)
        print('Root output:')
        print(json.dumps(trace.root_output, ensure_ascii=False, indent=2))
        print('Output directory:', out_dir)
        print('Memory directory:', memory_dir)
    finally:
        await multi_client.close()


if __name__ == '__main__':
    asyncio.run(main())
