from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import argparse
import asyncio
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp_rlm import (
    MCPCall,
    MCPInvocationContext,
    MCPServerSpec,
    MultiServerMCPClient,
    preprocess_long_context,
)


def _load_records(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding='utf-8-sig').strip()
    if not text:
        return []

    if path.suffix.lower() == '.jsonl':
        rows: List[Dict[str, Any]] = []
        for line in text.splitlines():
            row = line.strip()
            if not row:
                continue
            item = json.loads(row)
            if isinstance(item, dict):
                rows.append(item)
        return rows

    parsed = json.loads(text)
    if isinstance(parsed, list):
        return [x for x in parsed if isinstance(x, dict)]
    if isinstance(parsed, dict):
        if isinstance(parsed.get('data'), list):
            return [x for x in parsed['data'] if isinstance(x, dict)]
        return [parsed]
    raise ValueError(f'Unsupported dataset format: {path}')


def _pick_record(records: List[Dict[str, Any]], *, sample_id: str, index: int) -> Dict[str, Any]:
    if sample_id:
        for row in records:
            if str(row.get('_id', '')).strip() == sample_id:
                return row
        raise KeyError(f'Sample id not found: {sample_id}')

    if not records:
        raise ValueError('No records found in dataset file')

    idx = max(0, min(len(records) - 1, index))
    return records[idx]


def _extract_choices(row: Dict[str, Any]) -> Dict[str, str]:
    choices: Dict[str, str] = {}
    if isinstance(row.get('choices'), dict):
        for key, value in row['choices'].items():
            letter = str(key).strip().upper()
            if letter in {'A', 'B', 'C', 'D'}:
                text = str(value).strip()
                if text:
                    choices[letter] = text

    for letter in ('A', 'B', 'C', 'D'):
        if letter in choices:
            continue
        field = f'choice_{letter}'
        if field in row:
            text = str(row.get(field, '')).strip()
            if text:
                choices[letter] = text

    if len(choices) < 2:
        raise ValueError('Record does not contain enough multiple-choice options')
    return choices


async def _call_checked(
    client: MultiServerMCPClient,
    *,
    call: MCPCall,
    ctx: MCPInvocationContext,
    required_keys: Tuple[str, ...] = (),
) -> Dict[str, Any]:
    result = await client.call(call, ctx)
    if not result.ok:
        raise RuntimeError(f'{call.object_name} failed: {result.error}')

    if not isinstance(result.output, dict):
        raise RuntimeError(f'{call.object_name} output is not a dict: {type(result.output).__name__}')

    for key in required_keys:
        if key not in result.output:
            raise RuntimeError(f"{call.object_name} output missing key '{key}'")

    return {
        'output': result.output,
        'latency_ms': int(result.latency_ms),
    }


async def run_smoke(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_path = Path(args.dataset_file).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f'Dataset file not found: {dataset_path}')

    records = _load_records(dataset_path)
    record = _pick_record(records, sample_id=args.id, index=args.index)

    sample_id = str(record.get('_id', f'idx_{max(0, args.index)}'))
    question = str(record.get('question') or record.get('query') or '').strip()
    context = str(record.get('context', ''))
    if not question:
        raise ValueError('Sample missing question/query')
    if not context:
        raise ValueError('Sample missing context')

    choices = _extract_choices(record)

    out_dir = Path(args.out).resolve()
    run_dir = out_dir / 'runs' / sample_id
    store_dir = run_dir / 'context_store'
    run_dir.mkdir(parents=True, exist_ok=True)

    context_file = run_dir / 'context.txt'
    context_file.write_text(context, encoding='utf-8')

    manifest_path = preprocess_long_context(
        input_file=context_file,
        output_dir=store_dir,
        chunk_chars=args.chunk_chars,
        overlap_chars=args.overlap_chars,
        branch_factor=args.branch_factor,
    )

    ctx_server = ROOT / 'examples' / 'run_context_server.py'
    analysis_server = ROOT / 'examples' / 'run_analysis_server.py'

    ctx_cmd = [sys.executable, str(ctx_server), '--manifest', str(manifest_path)]
    analysis_cmd = [sys.executable, str(analysis_server)]

    strict_sdk = not bool(args.allow_legacy_fallback)
    if strict_sdk:
        ctx_cmd.append('--require-official-sdk')
        analysis_cmd.append('--require-official-sdk')

    specs = [
        MCPServerSpec(
            alias='ctx',
            command=ctx_cmd,
            cwd=str(ROOT),
            max_concurrency=8,
            prefer_official_sdk=True,
            strict_official_sdk=strict_sdk,
        ),
        MCPServerSpec(
            alias='analysis',
            command=analysis_cmd,
            cwd=str(ROOT),
            max_concurrency=8,
            prefer_official_sdk=True,
            strict_official_sdk=strict_sdk,
        ),
    ]

    client = MultiServerMCPClient(specs=specs)
    mcp_ctx = MCPInvocationContext(
        episode_id=f'smoke_{int(time.time())}',
        group_id='grp_smoke_root',
    )

    report: Dict[str, Any] = {
        'dataset_file': str(dataset_path),
        'sample_id': sample_id,
        'question': question,
        'manifest_path': str(manifest_path),
        'strict_sdk': strict_sdk,
        'objects_checked': [
            'ctx/context_stats',
            'ctx/search_hierarchical',
            'analysis/score_mcq_choices',
        ],
        'repeats': int(args.repeat),
        'results': {
            'ctx/context_stats': [],
            'ctx/search_hierarchical': [],
            'analysis/score_mcq_choices': [],
        },
        'ok': False,
    }

    try:
        await client.start()

        transport_mode = {}
        internal_clients = getattr(client, '_clients', {})
        if isinstance(internal_clients, dict):
            for alias, subclient in internal_clients.items():
                transport_mode[alias] = bool(getattr(subclient, 'using_official_sdk', False))
        report['transport_mode'] = transport_mode

        listed = await client.list_objects()
        report['listed_objects_count'] = len(listed)

        for _ in range(max(1, int(args.repeat))):
            stats = await _call_checked(
                client,
                call=MCPCall(
                    object_name='ctx/context_stats',
                    payload={'manifest_path': str(manifest_path)},
                    timeout_seconds=float(args.timeout_seconds),
                ),
                ctx=mcp_ctx,
                required_keys=('leaf_segments',),
            )
            report['results']['ctx/context_stats'].append(
                {
                    'latency_ms': stats['latency_ms'],
                    'leaf_segments': int(stats['output'].get('leaf_segments', 0)),
                    'num_levels': int(stats['output'].get('num_levels', 0)),
                }
            )

        search_hits = []
        for _ in range(max(1, int(args.repeat))):
            search = await _call_checked(
                client,
                call=MCPCall(
                    object_name='ctx/search_hierarchical',
                    payload={
                        'manifest_path': str(manifest_path),
                        'query': question,
                        'top_k': int(args.search_top_k),
                        'coarse_k': int(args.search_coarse_k),
                    },
                    timeout_seconds=float(args.timeout_seconds),
                ),
                ctx=mcp_ctx,
                required_keys=('hits',),
            )
            hits = list(search['output'].get('hits', []))
            search_hits = hits
            report['results']['ctx/search_hierarchical'].append(
                {
                    'latency_ms': search['latency_ms'],
                    'num_hits': len(hits),
                    'top_segment_id': str(hits[0].get('segment_id', '')) if hits and isinstance(hits[0], dict) else '',
                }
            )

        segment_text = context[: max(1000, int(args.segment_max_chars))]
        segment_id = 'raw_context_fallback'
        if search_hits and isinstance(search_hits[0], dict):
            seg_id = str(search_hits[0].get('segment_id', '')).strip()
            if seg_id:
                read_seg = await _call_checked(
                    client,
                    call=MCPCall(
                        object_name='ctx/read_segment',
                        payload={
                            'manifest_path': str(manifest_path),
                            'segment_id': seg_id,
                            'max_chars': int(args.segment_max_chars),
                        },
                        timeout_seconds=float(args.timeout_seconds),
                    ),
                    ctx=mcp_ctx,
                    required_keys=('text',),
                )
                segment_text = str(read_seg['output'].get('text', '') or '')
                segment_id = seg_id

        if not segment_text:
            raise RuntimeError('Segment text is empty before analysis/score_mcq_choices')

        for _ in range(max(1, int(args.repeat))):
            score = await _call_checked(
                client,
                call=MCPCall(
                    object_name='analysis/score_mcq_choices',
                    payload={
                        'question': question,
                        'choices': choices,
                        'text': segment_text,
                        'segment_id': segment_id,
                        'max_evidence': 4,
                    },
                    timeout_seconds=float(args.timeout_seconds),
                ),
                ctx=mcp_ctx,
                required_keys=('choice_scores',),
            )

            choice_scores = score['output'].get('choice_scores', {})
            if not isinstance(choice_scores, dict):
                raise RuntimeError('analysis/score_mcq_choices choice_scores is not dict')

            best_choice = None
            if choice_scores:
                best_choice = max(choice_scores.items(), key=lambda kv: float(kv[1]))[0]

            report['results']['analysis/score_mcq_choices'].append(
                {
                    'latency_ms': score['latency_ms'],
                    'best_choice': best_choice,
                    'num_choice_scores': len(choice_scores),
                }
            )

        report['ok'] = True
        return report
    finally:
        await client.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Minimal LongBench-v2 SDK-mode smoke test for ctx/context_stats, ctx/search_hierarchical, analysis/score_mcq_choices'
    )
    parser.add_argument('--dataset-file', type=str, required=True)
    parser.add_argument('--id', type=str, default='', help='Optional _id to pick one sample')
    parser.add_argument('--index', type=int, default=0, help='Sample index if --id not provided')
    parser.add_argument('--out', type=str, default='artifacts/lb2_sdk_smoke')

    parser.add_argument('--chunk-chars', type=int, default=16000)
    parser.add_argument('--overlap-chars', type=int, default=400)
    parser.add_argument('--branch-factor', type=int, default=8)

    parser.add_argument('--search-top-k', type=int, default=12)
    parser.add_argument('--search-coarse-k', type=int, default=24)
    parser.add_argument('--segment-max-chars', type=int, default=12000)

    parser.add_argument('--repeat', type=int, default=3, help='Repeat each object call this many times to check stability')
    parser.add_argument('--timeout-seconds', type=float, default=30.0)
    parser.add_argument('--allow-legacy-fallback', action='store_true', help='Allow fallback to legacy transport if SDK startup fails')

    args = parser.parse_args()

    report = asyncio.run(run_smoke(args))

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / 'smoke_report.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'Smoke report: {report_path}')

    if not bool(report.get('ok')):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
