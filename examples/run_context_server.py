from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import argparse
import asyncio
import os
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp_rlm import MCPRegistry, MCPServerInfo, StdioMCPServer
from mcp_rlm.mvp import LongContextStoreCache


def _resolve_manifest(params: Dict[str, Any], default_manifest: str | None) -> str:
    manifest = str(params.get('manifest_path') or '').strip()
    if manifest:
        return manifest
    if default_manifest:
        return default_manifest
    raise RuntimeError('manifest_path is required for context tools')


def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-zA-Z0-9_]+", str(text).lower()) if len(t) > 1]


def _window_text(text: str, *, window_chars: int, stride_chars: int) -> List[Dict[str, Any]]:
    if window_chars <= 0:
        window_chars = 2400
    if stride_chars <= 0:
        stride_chars = max(1, window_chars // 2)

    windows: List[Dict[str, Any]] = []
    cursor = 0
    idx = 0
    n = len(text)
    while cursor < n:
        chunk = text[cursor : cursor + window_chars]
        if not chunk:
            break
        windows.append(
            {
                'window_id': f'W{idx:05d}',
                'offset': cursor,
                'num_chars': len(chunk),
                'text': chunk,
            }
        )
        if cursor + window_chars >= n:
            break
        cursor += stride_chars
        idx += 1
    return windows


def _focus_score(text: str, terms: List[str]) -> float:
    if not terms:
        return 0.0
    low = text.lower()
    return float(sum(low.count(t) for t in terms))


def _jaccard(a: List[str], b: List[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


async def main() -> None:
    parser = argparse.ArgumentParser(description='Run context MCP server')
    parser.add_argument('--manifest', type=str, default='', help='Default manifest path')
    parser.add_argument('--legacy-mcp', action='store_true', help='Use legacy JSON-RPC transport instead of official MCP SDK')
    parser.add_argument('--require-official-sdk', action='store_true', help='Fail fast if official MCP SDK cannot be used')
    args = parser.parse_args()

    default_manifest = args.manifest.strip() or os.getenv('MCP_RLM_CONTEXT_MANIFEST', '').strip() or None
    cache = LongContextStoreCache()

    registry = MCPRegistry()

    def context_stats(payload: Dict[str, Any], _) -> Dict[str, Any]:
        manifest = _resolve_manifest(payload, default_manifest)
        store = cache.get(manifest)
        return store.context_stats()

    def list_level(payload: Dict[str, Any], _) -> Dict[str, Any]:
        manifest = _resolve_manifest(payload, default_manifest)
        store = cache.get(manifest)
        level = int(payload.get('level', 0))
        limit = int(payload.get('limit', 100))
        offset = int(payload.get('offset', 0))
        return store.list_level(level, limit=limit, offset=offset)

    def read_segment(payload: Dict[str, Any], _) -> Dict[str, Any]:
        manifest = _resolve_manifest(payload, default_manifest)
        store = cache.get(manifest)
        segment_id = str(payload.get('segment_id', '')).strip()
        if not segment_id:
            raise RuntimeError('read_segment requires segment_id')
        max_chars = payload.get('max_chars')
        max_chars_int = None if max_chars is None else int(max_chars)
        offset = int(payload.get('offset', 0))
        return store.read_segment(segment_id, max_chars=max_chars_int, offset=offset)

    def read_segment_windows(payload: Dict[str, Any], _) -> Dict[str, Any]:
        manifest = _resolve_manifest(payload, default_manifest)
        store = cache.get(manifest)
        segment_id = str(payload.get('segment_id', '')).strip()
        if not segment_id:
            raise RuntimeError('read_segment_windows requires segment_id')

        response = store.read_segment(segment_id, max_chars=None, offset=0)
        full_text = str(response.get('text', ''))
        segment_meta = response.get('segment', {}) if isinstance(response, dict) else {}

        window_chars = int(payload.get('window_chars', 3200))
        stride_chars = int(payload.get('stride_chars', max(1, window_chars // 2)))
        max_windows = max(1, int(payload.get('max_windows', 8)))

        focus_terms: List[str] = []
        query = str(payload.get('query', '')).strip()
        if query:
            focus_terms.extend(_tokenize(query))
        raw_choices = payload.get('choices')
        if isinstance(raw_choices, dict):
            for _, value in raw_choices.items():
                focus_terms.extend(_tokenize(str(value)))
        explicit_terms = payload.get('focus_terms')
        if isinstance(explicit_terms, list):
            for term in explicit_terms:
                focus_terms.extend(_tokenize(str(term)))

        windows = _window_text(full_text, window_chars=window_chars, stride_chars=stride_chars)
        for w in windows:
            w['score'] = _focus_score(str(w.get('text', '')), focus_terms)

        windows.sort(key=lambda x: float(x.get('score', 0.0)), reverse=True)
        selected = windows[:max_windows]
        selected.sort(key=lambda x: int(x.get('offset', 0)))

        return {
            'segment': segment_meta,
            'segment_id': segment_id,
            'window_chars': window_chars,
            'stride_chars': stride_chars,
            'focus_terms': focus_terms[:64],
            'num_windows_total': len(windows),
            'num_windows_selected': len(selected),
            'windows': selected,
        }

    def search_hierarchical(payload: Dict[str, Any], _) -> Dict[str, Any]:
        manifest = _resolve_manifest(payload, default_manifest)
        store = cache.get(manifest)
        query = str(payload.get('query', '')).strip()
        if not query:
            raise RuntimeError('search_hierarchical requires query')
        top_k = int(payload.get('top_k', 12))
        coarse_k = int(payload.get('coarse_k', 24))
        return store.search_hierarchical(query=query, top_k=top_k, coarse_k=coarse_k)

    def search_multi_query(payload: Dict[str, Any], _) -> Dict[str, Any]:
        manifest = _resolve_manifest(payload, default_manifest)
        store = cache.get(manifest)

        raw_queries = payload.get('queries')
        queries: List[str] = []
        if isinstance(raw_queries, list):
            for q in raw_queries:
                text = str(q).strip()
                if text:
                    queries.append(text)
        else:
            query = str(payload.get('query', '')).strip()
            if query:
                queries.append(query)

        if not queries:
            raise RuntimeError('search_multi_query requires query or non-empty queries')

        top_k = int(payload.get('top_k', 12))
        coarse_k = int(payload.get('coarse_k', 24))
        per_query_top_k = max(2, int(payload.get('per_query_top_k', max(8, top_k // 2))))

        weights_payload = payload.get('query_weights')
        query_weights: List[float] = []
        if isinstance(weights_payload, list):
            for x in weights_payload:
                try:
                    query_weights.append(float(x))
                except (TypeError, ValueError):
                    query_weights.append(1.0)

        merged: Dict[str, Dict[str, Any]] = {}
        query_results: List[Dict[str, Any]] = []

        for idx, query in enumerate(queries):
            search = store.search_hierarchical(
                query=query,
                top_k=per_query_top_k,
                coarse_k=coarse_k,
            )
            query_results.append({'query': query, 'hits': search.get('hits', [])})
            hits = search.get('hits', [])
            if not isinstance(hits, list):
                continue
            weight = query_weights[idx] if idx < len(query_weights) else 1.0
            for hit in hits:
                if not isinstance(hit, dict):
                    continue
                seg_id = str(hit.get('segment_id', '')).strip()
                if not seg_id:
                    continue
                base = float(hit.get('score', 0.0))
                item = merged.get(seg_id)
                if item is None:
                    item = {
                        'segment_id': seg_id,
                        'score': 0.0,
                        'start_char': hit.get('start_char'),
                        'end_char': hit.get('end_char'),
                        'num_chars': hit.get('num_chars'),
                        'preview': hit.get('preview', ''),
                        'sources': [],
                    }
                    merged[seg_id] = item
                item['score'] = float(item['score']) + (base * float(weight))
                sources = item.get('sources', [])
                if query not in sources:
                    sources.append(query)
                item['sources'] = sources

        hits = list(merged.values())
        hits.sort(key=lambda x: float(x.get('score', 0.0)), reverse=True)

        return {
            'queries': queries,
            'top_k': top_k,
            'coarse_k': coarse_k,
            'per_query_top_k': per_query_top_k,
            'hits': hits[:top_k],
            'query_results': query_results,
        }

    def search_hierarchical_mmr(payload: Dict[str, Any], _) -> Dict[str, Any]:
        manifest = _resolve_manifest(payload, default_manifest)
        store = cache.get(manifest)
        query = str(payload.get('query', '')).strip()
        if not query:
            raise RuntimeError('search_hierarchical_mmr requires query')

        top_k = max(1, int(payload.get('top_k', 12)))
        coarse_k = int(payload.get('coarse_k', 24))
        candidate_k = max(top_k, int(payload.get('candidate_k', top_k * 4)))
        lambda_mult = float(payload.get('lambda_mult', 0.75))
        lambda_mult = max(0.0, min(1.0, lambda_mult))

        base = store.search_hierarchical(query=query, top_k=candidate_k, coarse_k=coarse_k)
        base_hits = [x for x in base.get('hits', []) if isinstance(x, dict)]
        if not base_hits:
            return {
                'query': query,
                'top_k': top_k,
                'hits': [],
                'candidate_k': candidate_k,
                'lambda_mult': lambda_mult,
            }

        max_score = max(float(x.get('score', 0.0)) for x in base_hits) if base_hits else 1.0
        if max_score <= 0:
            max_score = 1.0

        selected: List[Dict[str, Any]] = []
        remaining = list(base_hits)

        while remaining and len(selected) < top_k:
            best_idx = -1
            best_mmr = -1e9
            for idx, hit in enumerate(remaining):
                rel = float(hit.get('score', 0.0)) / max_score
                preview_i = _tokenize(str(hit.get('preview', '')))

                max_sim = 0.0
                for chosen in selected:
                    preview_j = _tokenize(str(chosen.get('preview', '')))
                    sim = _jaccard(preview_i, preview_j)
                    if sim > max_sim:
                        max_sim = sim

                mmr = (lambda_mult * rel) - ((1.0 - lambda_mult) * max_sim)
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = idx

            if best_idx < 0:
                break
            selected.append(remaining.pop(best_idx))

        return {
            'query': query,
            'top_k': top_k,
            'candidate_k': candidate_k,
            'lambda_mult': lambda_mult,
            'hits': selected,
        }

    registry.register('context_stats', context_stats)
    registry.register('list_level', list_level)
    registry.register('read_segment', read_segment)
    registry.register('read_segment_windows', read_segment_windows)
    registry.register('search_hierarchical', search_hierarchical)
    registry.register('search_multi_query', search_multi_query)
    registry.register('search_hierarchical_mmr', search_hierarchical_mmr)

    server = StdioMCPServer(
        registry=registry,
        server_info=MCPServerInfo(name='mcp-rlm-context-server', version='0.2.0'),
        prefer_official_sdk=not args.legacy_mcp,
        strict_official_sdk=bool(args.require_official_sdk),
    )
    await server.serve_forever()


if __name__ == '__main__':
    asyncio.run(main())
