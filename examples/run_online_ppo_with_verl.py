from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import argparse
import asyncio
import json
import os
import random
import subprocess
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
    register_builtin_programs,
    register_mvp_programs,
)
from mcp_rlm.training import (
    QueryItem,
    export_agentic_rl,
    export_cold_start,
    export_openrlhf,
    export_trace,
    export_traces,
    export_verl,
    export_verl_on_policy_dataset,
    load_query_items,
)


POLICY_MODES = ['heuristic', 'openai', 'openrouter', 'vllm', 'ollama', 'huggingface']


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


def build_runtime(
    manifest_path: Path,
    memory_dir: Path,
    *,
    require_official_mcp_sdk: bool = False,
    legacy_mcp: bool = False,
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

    multi_client = MultiServerMCPClient(
        specs=[
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
    )

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


def _choose_samples(items: List[QueryItem], episodes_per_iter: int, rng: random.Random) -> List[QueryItem]:
    if episodes_per_iter <= 0 or episodes_per_iter >= len(items):
        return list(items)
    indexes = rng.sample(range(len(items)), episodes_per_iter)
    return [items[i] for i in indexes]


async def run_rollouts(
    *,
    samples: List[QueryItem],
    manifest_path: Path,
    out_dir: Path,
    max_children: int,
    policy_config: Dict[str, Any],
    require_official_mcp_sdk: bool,
    legacy_mcp: bool,
) -> List[Any]:
    runtime, mcp_client = build_runtime(
        manifest_path,
        out_dir / 'memory',
        require_official_mcp_sdk=require_official_mcp_sdk,
        legacy_mcp=legacy_mcp,
    )
    traces: List[Any] = []

    try:
        for sample in samples:
            payload: Dict[str, Any] = {
                'query': sample.query,
                'manifest_path': str(manifest_path),
                'max_children': max_children,
                'policy_config': dict(policy_config),
            }
            if sample.answer:
                payload['expected_answer'] = sample.answer
            if sample.metadata:
                payload['query_metadata'] = sample.metadata

            trace = await runtime.run_episode(
                goal=sample.query,
                program='mvp_root',
                input_payload=payload,
            )

            if isinstance(trace.root_output, dict):
                if sample.answer and trace.root_output.get('expected_answer') is None:
                    trace.root_output['expected_answer'] = sample.answer
                if sample.answer:
                    answer_text = str(trace.root_output.get('answer', '')).lower()
                    trace.root_output['answer_correct'] = sample.answer.lower() in answer_text

            traces.append(trace)
    finally:
        await mcp_client.close()

    return traces


def summarize_traces(traces: List[Any]) -> Dict[str, Any]:
    if not traces:
        return {
            'num_episodes': 0,
            'success_rate': 0.0,
            'avg_steps': 0.0,
            'avg_groups': 0.0,
            'avg_confidence': 0.0,
        }

    success = 0
    total_steps = 0
    total_groups = 0
    total_conf = 0.0
    conf_count = 0

    for trace in traces:
        if trace.success:
            success += 1
        total_steps += len(trace.steps)
        total_groups += len(trace.groups)
        if isinstance(trace.root_output, dict) and trace.root_output.get('confidence') is not None:
            total_conf += float(trace.root_output.get('confidence', 0.0))
            conf_count += 1

    return {
        'num_episodes': len(traces),
        'success_rate': success / float(len(traces)),
        'avg_steps': total_steps / float(len(traces)),
        'avg_groups': total_groups / float(len(traces)),
        'avg_confidence': (total_conf / float(conf_count)) if conf_count > 0 else 0.0,
    }


def _latest_actor_ckpt(checkpoint_root: Path) -> Path | None:
    if not checkpoint_root.exists():
        return None

    best_step = -1
    best_path: Path | None = None
    for child in checkpoint_root.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith('global_step_'):
            continue
        try:
            step = int(name.split('_')[-1])
        except ValueError:
            continue
        actor_path = child / 'actor'
        if step > best_step and actor_path.exists():
            best_step = step
            best_path = actor_path
    return best_path


def run_verl_ppo(
    *,
    python_bin: str,
    verl_repo: Path,
    train_path: Path,
    val_path: Path,
    reward_fn_path: Path,
    reward_manager_source: str,
    reward_manager_name: str,
    reward_manager_module_path: str,
    actor_model_path: str,
    rollout_engine: str,
    n_gpus: int,
    train_batch_size: int,
    max_prompt_length: int,
    max_response_length: int,
    ppo_epochs: int,
    total_training_steps: int,
    experiment_name: str,
    checkpoint_dir: Path,
) -> None:
    if not actor_model_path:
        raise ValueError('actor_model_path is required when VERL training is enabled')

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if reward_manager_source not in {'register', 'importlib'}:
        raise ValueError("reward_manager_source must be 'register' or 'importlib'")

    cmd = [
        python_bin,
        '-m',
        'verl.trainer.main_ppo',
        f'data.train_files={train_path.as_posix()}',
        f'data.val_files={val_path.as_posix()}',
        'data.prompt_key=prompt',
        'data.reward_fn_key=data_source',
        'data.return_raw_chat=True',
        'data.filter_overlong_prompts=False',
        'data.dataloader_num_workers=0',
        f'data.max_prompt_length={max_prompt_length}',
        f'data.max_response_length={max_response_length}',
        f'data.train_batch_size={train_batch_size}',
        f'data.val_batch_size={max(1, min(train_batch_size, 64))}',
        f'custom_reward_function.path={reward_fn_path.as_posix()}',
        'custom_reward_function.name=compute_score',
        f'reward_manager.source={reward_manager_source}',
        f'reward_manager.name={reward_manager_name}',
        f'actor_rollout_ref.model.path={Path(actor_model_path).as_posix()}',
        f'actor_rollout_ref.rollout.name={rollout_engine}',
        'actor_rollout_ref.rollout.mode=async',
        'actor_rollout_ref.rollout.load_format=hf',
        'actor_rollout_ref.rollout.skip_tokenizer_init=False',
        'actor_rollout_ref.rollout.tensor_model_parallel_size=1',
        'actor_rollout_ref.rollout.data_parallel_size=1',
        f'actor_rollout_ref.rollout.prompt_length={max_prompt_length}',
        f'actor_rollout_ref.rollout.response_length={max_response_length}',
        f'actor_rollout_ref.rollout.max_model_len={max_prompt_length + max_response_length}',
        'actor_rollout_ref.rollout.gpu_memory_utilization=0.4',
        f'actor_rollout_ref.actor.ppo_epochs={ppo_epochs}',
        'actor_rollout_ref.actor.use_kl_loss=False',
        'algorithm.use_kl_in_reward=False',
        'algorithm.adv_estimator=gae',
        'algorithm.gamma=1.0',
        'algorithm.lam=0.95',
        f'trainer.total_training_steps={total_training_steps}',
        'trainer.total_epochs=1',
        'trainer.save_freq=1',
        'trainer.test_freq=-1',
        'trainer.val_before_train=False',
        'trainer.log_val_generations=0',
        'trainer.resume_mode=disable',
        'trainer.nnodes=1',
        f'trainer.n_gpus_per_node={max(1, int(n_gpus))}',
        'trainer.project_name=mcp_rlm_online',
        f'trainer.experiment_name={experiment_name}',
        f'trainer.default_local_dir={checkpoint_dir.as_posix()}',
        'trainer.logger=[console]',
    ]

    if reward_manager_source == 'importlib':
        module_path = reward_manager_module_path.strip()
        if not module_path:
            raise ValueError('reward_manager.module.path is required when reward_manager_source=importlib')
        cmd.append(f'reward_manager.module.path={Path(module_path).resolve().as_posix()}')

    env = dict(os.environ)
    py_paths = [str(verl_repo), str(ROOT)]
    if env.get('PYTHONPATH'):
        py_paths.append(env['PYTHONPATH'])
    env['PYTHONPATH'] = os.pathsep.join(py_paths)

    print('[VERL CMD]')
    print(' '.join(cmd))

    subprocess.run(
        cmd,
        cwd=str(verl_repo),
        env=env,
        check=True,
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description='MCP-RLM + VERL on-policy online RL loop')
    parser.add_argument('--manifest', type=str, required=True, help='Path to context manifest.json')
    parser.add_argument('--queries', type=str, required=True, help='Path to queries (.txt/.jsonl)')
    parser.add_argument('--out', type=str, default='artifacts/online_rl')
    parser.add_argument('--iterations', type=int, default=1)
    parser.add_argument('--episodes-per-iter', type=int, default=8)
    parser.add_argument('--max-children', type=int, default=16)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=7)

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

    parser.add_argument('--skip-verl-train', action='store_true', help='Only rollout and export data, do not run VERL PPO')
    parser.add_argument('--python-bin', type=str, default=sys.executable)
    parser.add_argument('--verl-repo', type=str, default=str((ROOT / '..' / 'verl').resolve()))
    parser.add_argument('--actor-model-path', type=str, default='', help='Base model/checkpoint path for VERL actor')
    parser.add_argument('--rollout-engine', type=str, default='hf', choices=['hf', 'vllm', 'sglang'])
    parser.add_argument('--n-gpus', type=int, default=1)
    parser.add_argument('--train-batch-size', type=int, default=64)
    parser.add_argument('--max-prompt-length', type=int, default=1536)
    parser.add_argument('--max-response-length', type=int, default=256)
    parser.add_argument('--ppo-epochs', type=int, default=1)
    parser.add_argument('--total-training-steps', type=int, default=8)
    parser.add_argument('--experiment-name', type=str, default='mcp_rlm_online')

    parser.add_argument('--reward-manager-source', type=str, default='register', choices=['register', 'importlib'])
    parser.add_argument('--reward-manager-name', type=str, default='mcp_rlm')
    parser.add_argument('--reward-manager-module-path', type=str, default='', help='Required only when --reward-manager-source=importlib')

    parser.add_argument(
        '--sync-rollout-hf-from-ckpt',
        action='store_true',
        help='After each iteration, update rollout huggingface model path to latest VERL actor checkpoint',
    )

    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f'Manifest not found: {manifest_path}')

    query_items = load_query_items(Path(args.queries))
    if not query_items:
        raise ValueError('No valid queries loaded from --queries')

    out_dir = (ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    policy_config = build_policy_config(args)
    actor_model_path = args.actor_model_path

    if not args.skip_verl_train and not actor_model_path:
        raise ValueError('--actor-model-path is required unless --skip-verl-train is set')

    verl_repo = Path(args.verl_repo).resolve()
    if not args.skip_verl_train and not verl_repo.exists():
        raise FileNotFoundError(f'VERL repo not found: {verl_repo}')

    reward_fn_path = (ROOT / 'mcp_rlm' / 'training' / 'verl_reward.py').resolve()
    if args.reward_manager_source == 'importlib' and not args.reward_manager_module_path:
        args.reward_manager_module_path = str((ROOT / 'mcp_rlm' / 'training' / 'verl_reward_manager.py').resolve())

    rng = random.Random(args.seed)

    summaries: List[Dict[str, Any]] = []

    for iter_index in range(max(1, int(args.iterations))):
        iter_dir = out_dir / f'iter_{iter_index:03d}'
        iter_dir.mkdir(parents=True, exist_ok=True)

        sampled = _choose_samples(query_items, int(args.episodes_per_iter), rng)
        traces = await run_rollouts(
            samples=sampled,
            manifest_path=manifest_path,
            out_dir=iter_dir,
            max_children=int(args.max_children),
            policy_config=policy_config,
            require_official_mcp_sdk=bool(args.require_official_mcp_sdk),
            legacy_mcp=bool(args.legacy_mcp),
        )

        export_traces(traces, iter_dir)
        export_cold_start(traces, iter_dir / 'cold_start_turns.jsonl')
        export_agentic_rl(traces, iter_dir / 'agentic_rl.jsonl')
        export_verl(traces, iter_dir / 'verl_warm_start.jsonl')
        export_openrlhf(traces, iter_dir / 'openrlhf_episodes.jsonl')
        if traces:
            export_trace(traces[0], iter_dir / 'first_episode')

        dataset_info = export_verl_on_policy_dataset(
            traces,
            iter_dir / 'verl_online_dataset',
            val_ratio=float(args.val_ratio),
            seed=args.seed + iter_index,
        )

        summary = {
            'iteration': iter_index,
            'policy_config': dict(policy_config),
            'num_sampled_queries': len(sampled),
            'trace_metrics': summarize_traces(traces),
            'dataset': dataset_info,
            'mcp_transport': {
                'legacy_mcp': bool(args.legacy_mcp),
                'require_official_mcp_sdk': bool(args.require_official_mcp_sdk),
            },
        }

        if not args.skip_verl_train:
            ckpt_dir = iter_dir / 'verl_checkpoints'
            run_verl_ppo(
                python_bin=args.python_bin,
                verl_repo=verl_repo,
                train_path=Path(dataset_info['train_path']),
                val_path=Path(dataset_info['val_path']),
                reward_fn_path=reward_fn_path,
                reward_manager_source=args.reward_manager_source,
                reward_manager_name=args.reward_manager_name,
                reward_manager_module_path=args.reward_manager_module_path,
                actor_model_path=actor_model_path,
                rollout_engine=args.rollout_engine,
                n_gpus=args.n_gpus,
                train_batch_size=args.train_batch_size,
                max_prompt_length=args.max_prompt_length,
                max_response_length=args.max_response_length,
                ppo_epochs=args.ppo_epochs,
                total_training_steps=args.total_training_steps,
                experiment_name=f"{args.experiment_name}_iter_{iter_index:03d}",
                checkpoint_dir=ckpt_dir,
            )

            latest_ckpt = _latest_actor_ckpt(ckpt_dir)
            if latest_ckpt is not None:
                summary['latest_actor_ckpt'] = str(latest_ckpt)

                if args.sync_rollout_hf_from_ckpt and str(policy_config.get('mode', '')) in {'huggingface', 'hf', 'transformers'}:
                    policy_config['model'] = str(latest_ckpt)
                    actor_model_path = str(latest_ckpt)
                    summary['rollout_model_updated'] = str(latest_ckpt)

        summary_path = iter_dir / 'summary.json'
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        summaries.append(summary)

        print(f'[ITER {iter_index}] summary -> {summary_path}')

    final_summary_path = out_dir / 'summary.json'
    final_summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Online RL loop finished. Summary: {final_summary_path}')


if __name__ == '__main__':
    asyncio.run(main())
