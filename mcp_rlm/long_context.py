from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import json
import math
import re


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "what",
    "which",
    "who",
    "why",
    "with",
}

_SEMANTIC_DIM_DEFAULT = 256


def _tokenize(text: str) -> List[str]:
    terms = [t for t in re.findall(r"[a-zA-Z0-9_]+", text.lower()) if len(t) > 1]
    out: List[str] = []
    seen = set()
    for term in terms:
        if term in _STOPWORDS:
            continue
        if term in seen:
            continue
        seen.add(term)
        out.append(term)
    return out


def _score_terms(text: str, terms: List[str]) -> int:
    low = text.lower()
    return sum(low.count(term) for term in terms)


def _dense_embed(text: str, *, dim: int) -> List[float]:
    if dim <= 0:
        dim = _SEMANTIC_DIM_DEFAULT
    vec = [0.0 for _ in range(dim)]
    terms = _tokenize(text)
    if not terms:
        return vec

    for term in terms:
        idx = hash(term) % dim
        vec[idx] += 1.0

    low = str(text).lower()
    for i in range(max(0, len(low) - 2)):
        tri = low[i : i + 3]
        if " " in tri:
            continue
        idx = hash(f"tri:{tri}") % dim
        vec[idx] += 0.2

    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    size = min(len(a), len(b))
    if size <= 0:
        return 0.0
    return float(sum(a[i] * b[i] for i in range(size)))


@dataclass
class SegmentMeta:
    segment_id: str
    level: int
    start_char: int
    end_char: int
    num_chars: int
    preview: str
    file: Optional[str] = None
    children: Optional[List[str]] = None


class LongContextStore:
    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.root_dir = self.manifest_path.parent
        self.source_file = payload["source_file"]
        self.total_chars = int(payload["total_chars"])
        self.chunk_chars = int(payload["chunk_chars"])
        self.overlap_chars = int(payload["overlap_chars"])
        self.branch_factor = int(payload["branch_factor"])
        self.levels: Dict[int, List[str]] = {int(k): list(v) for k, v in payload["levels"].items()}

        self.segments: Dict[str, SegmentMeta] = {}
        for seg_id, seg_payload in payload["segments"].items():
            self.segments[seg_id] = SegmentMeta(
                segment_id=seg_id,
                level=int(seg_payload["level"]),
                start_char=int(seg_payload["start_char"]),
                end_char=int(seg_payload["end_char"]),
                num_chars=int(seg_payload["num_chars"]),
                preview=str(seg_payload.get("preview", "")),
                file=seg_payload.get("file"),
                children=list(seg_payload.get("children", [])) if seg_payload.get("children") else None,
            )

        self.semantic_index_file = str(payload.get("semantic_index_file", "")).strip()
        self.semantic_dim = int(payload.get("semantic_dim", _SEMANTIC_DIM_DEFAULT))
        self.semantic_backend = str(payload.get("semantic_backend", "none"))
        self._semantic_vectors: Dict[str, List[float]] = {}
        self._load_semantic_index()

    def _leaf_ids(self) -> List[str]:
        return list(self.levels.get(0, []))

    def _load_semantic_index(self) -> None:
        self._semantic_vectors = {}
        if self.semantic_dim <= 0:
            self.semantic_dim = _SEMANTIC_DIM_DEFAULT

        if self.semantic_index_file:
            path = self.root_dir / self.semantic_index_file
            if path.exists() and path.is_file():
                with path.open("r", encoding="utf-8") as f:
                    for raw in f:
                        line = raw.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(item, dict):
                            continue
                        seg_id = str(item.get("segment_id", "")).strip()
                        emb = item.get("embedding")
                        if not seg_id or not isinstance(emb, list):
                            continue
                        vec: List[float] = []
                        for x in emb:
                            try:
                                vec.append(float(x))
                            except (TypeError, ValueError):
                                vec.append(0.0)
                        if not vec:
                            continue
                        self._semantic_vectors[seg_id] = vec

        if self._semantic_vectors:
            return

        # Fallback: lazily derive embeddings from preview/text without external deps.
        for seg_id in self._leaf_ids():
            seg = self.segments.get(seg_id)
            if seg is None:
                continue
            source = seg.preview
            if seg.file:
                try:
                    source = (self.root_dir / seg.file).read_text(encoding="utf-8")
                except Exception:
                    source = seg.preview
            self._semantic_vectors[seg_id] = _dense_embed(source, dim=self.semantic_dim)
        if self._semantic_vectors:
            self.semantic_backend = "hashed_bow_fallback"

    def context_stats(self) -> Dict[str, Any]:
        return {
            "source_file": self.source_file,
            "manifest_path": str(self.manifest_path),
            "total_chars": self.total_chars,
            "chunk_chars": self.chunk_chars,
            "overlap_chars": self.overlap_chars,
            "branch_factor": self.branch_factor,
            "num_levels": len(self.levels),
            "segments_per_level": {str(level): len(seg_ids) for level, seg_ids in self.levels.items()},
            "leaf_segments": len(self.levels.get(0, [])),
            "semantic_enabled": bool(self._semantic_vectors),
            "semantic_dim": self.semantic_dim,
            "semantic_backend": self.semantic_backend,
            "semantic_index_file": self.semantic_index_file,
        }

    def list_level(self, level: int, *, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        seg_ids = self.levels.get(level, [])
        sliced = seg_ids[offset : offset + limit]
        items = [self._segment_to_dict(self.segments[seg_id]) for seg_id in sliced]
        return {
            "level": level,
            "offset": offset,
            "limit": limit,
            "total": len(seg_ids),
            "segments": items,
        }

    def read_segment(self, segment_id: str, *, max_chars: Optional[int] = None, offset: int = 0) -> Dict[str, Any]:
        if segment_id not in self.segments:
            raise KeyError(f"Unknown segment_id: {segment_id}")

        seg = self.segments[segment_id]
        text = self._materialize_segment_text(seg)

        if offset < 0:
            offset = 0
        if offset > len(text):
            offset = len(text)
        if max_chars is None:
            chunk = text[offset:]
        else:
            chunk = text[offset : offset + max(0, max_chars)]

        return {
            "segment": self._segment_to_dict(seg),
            "offset": offset,
            "max_chars": max_chars,
            "returned_chars": len(chunk),
            "text": chunk,
        }

    def search_hierarchical(
        self,
        *,
        query: str,
        top_k: int = 12,
        coarse_k: int = 24,
    ) -> Dict[str, Any]:
        terms = _tokenize(query)
        if not terms:
            return {"query": query, "hits": []}

        max_level = max(self.levels.keys()) if self.levels else 0
        coarse_level = max_level if max_level > 0 else 0
        coarse_ids = self.levels.get(coarse_level, [])

        coarse_scored: List[tuple[str, int]] = []
        for seg_id in coarse_ids:
            seg = self.segments[seg_id]
            score = _score_terms(seg.preview, terms)
            if score > 0:
                coarse_scored.append((seg_id, score))
        coarse_scored.sort(key=lambda x: x[1], reverse=True)

        candidate_leaves: List[str]
        if coarse_level == 0:
            candidate_leaves = [seg_id for seg_id, _ in coarse_scored] or list(self.levels.get(0, []))
        else:
            selected_coarse = [seg_id for seg_id, _ in coarse_scored[:coarse_k]]
            if not selected_coarse:
                selected_coarse = list(self.levels.get(coarse_level, []))[:coarse_k]
            leaves: List[str] = []
            for seg_id in selected_coarse:
                leaves.extend(self._leaf_descendants(seg_id))
            candidate_leaves = list(dict.fromkeys(leaves))

        leaf_scored: List[Dict[str, Any]] = []
        for leaf_id in candidate_leaves:
            seg = self.segments[leaf_id]
            text = self._materialize_segment_text(seg)
            score = _score_terms(text, terms)
            if score <= 0:
                continue
            leaf_scored.append(
                {
                    "segment_id": leaf_id,
                    "score": score,
                    "start_char": seg.start_char,
                    "end_char": seg.end_char,
                    "num_chars": seg.num_chars,
                    "preview": seg.preview,
                }
            )
        leaf_scored.sort(key=lambda x: x["score"], reverse=True)

        return {
            "query": query,
            "terms": terms,
            "top_k": top_k,
            "coarse_k": coarse_k,
            "hits": leaf_scored[:top_k],
            "searched_candidates": len(candidate_leaves),
        }

    def search_semantic(
        self,
        *,
        query: str,
        top_k: int = 12,
        candidate_k: int = 160,
        lexical_boost_weight: float = 0.12,
    ) -> Dict[str, Any]:
        q = str(query).strip()
        if not q:
            return {"query": query, "hits": []}

        q_vec = _dense_embed(q, dim=self.semantic_dim)
        terms = _tokenize(q)

        leaf_ids = self._leaf_ids()
        candidate_ids = list(leaf_ids)
        if candidate_k > 0 and len(leaf_ids) > candidate_k:
            lexical_seed = self.search_hierarchical(
                query=q,
                top_k=max(candidate_k, top_k * 6),
                coarse_k=max(24, candidate_k),
            )
            seeded = [
                str(item.get("segment_id", "")).strip()
                for item in lexical_seed.get("hits", [])
                if isinstance(item, dict)
            ]
            candidate_ids = [sid for sid in seeded if sid]
            if len(candidate_ids) < candidate_k:
                for sid in leaf_ids:
                    if sid in candidate_ids:
                        continue
                    candidate_ids.append(sid)
                    if len(candidate_ids) >= candidate_k:
                        break

        ranked: List[Dict[str, Any]] = []
        denom = max(1.0, float(len(terms)))
        for seg_id in candidate_ids:
            seg = self.segments.get(seg_id)
            if seg is None:
                continue
            vec = self._semantic_vectors.get(seg_id)
            if vec is None:
                vec = _dense_embed(seg.preview, dim=self.semantic_dim)
                self._semantic_vectors[seg_id] = vec

            sem = _cosine(q_vec, vec)
            lex = 0.0
            if terms:
                lex = min(1.0, float(_score_terms(seg.preview, terms)) / denom)
            score = ((1.0 - lexical_boost_weight) * sem) + (lexical_boost_weight * lex)

            if sem <= 0 and lex <= 0:
                continue
            ranked.append(
                {
                    "segment_id": seg_id,
                    "score": float(score),
                    "semantic_score": float(sem),
                    "lexical_boost": float(lex),
                    "start_char": seg.start_char,
                    "end_char": seg.end_char,
                    "num_chars": seg.num_chars,
                    "preview": seg.preview,
                }
            )

        ranked.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return {
            "query": q,
            "top_k": top_k,
            "candidate_k": len(candidate_ids),
            "semantic_backend": self.semantic_backend,
            "hits": ranked[:top_k],
        }

    def search_hybrid(
        self,
        *,
        query: str,
        top_k: int = 12,
        coarse_k: int = 24,
        semantic_top_k: int = 64,
        lexical_weight: float = 0.45,
        semantic_weight: float = 0.55,
    ) -> Dict[str, Any]:
        q = str(query).strip()
        if not q:
            return {"query": query, "hits": []}

        lexical = self.search_hierarchical(
            query=q,
            top_k=max(top_k * 4, 32),
            coarse_k=coarse_k,
        )
        semantic = self.search_semantic(
            query=q,
            top_k=max(semantic_top_k, top_k * 4),
            candidate_k=max(semantic_top_k * 2, top_k * 8),
        )

        lexical_hits = [x for x in lexical.get("hits", []) if isinstance(x, dict)]
        semantic_hits = [x for x in semantic.get("hits", []) if isinstance(x, dict)]

        max_lex = max([float(x.get("score", 0.0)) for x in lexical_hits], default=1.0)
        if max_lex <= 0:
            max_lex = 1.0
        max_sem = max([float(x.get("semantic_score", x.get("score", 0.0))) for x in semantic_hits], default=1.0)
        if max_sem <= 0:
            max_sem = 1.0

        merged: Dict[str, Dict[str, Any]] = {}

        for hit in lexical_hits:
            seg_id = str(hit.get("segment_id", "")).strip()
            if not seg_id:
                continue
            base = merged.get(seg_id)
            if base is None:
                seg = self.segments.get(seg_id)
                if seg is None:
                    continue
                base = {
                    "segment_id": seg_id,
                    "score": 0.0,
                    "lexical_score": 0.0,
                    "semantic_score": 0.0,
                    "start_char": seg.start_char,
                    "end_char": seg.end_char,
                    "num_chars": seg.num_chars,
                    "preview": seg.preview,
                    "sources": [],
                }
                merged[seg_id] = base
            lex_norm = float(hit.get("score", 0.0)) / max_lex
            base["lexical_score"] = max(float(base.get("lexical_score", 0.0)), lex_norm)
            base["score"] = float(base.get("score", 0.0)) + (lexical_weight * lex_norm)
            sources = base.get("sources", [])
            if "lexical" not in sources:
                sources.append("lexical")
            base["sources"] = sources

        for hit in semantic_hits:
            seg_id = str(hit.get("segment_id", "")).strip()
            if not seg_id:
                continue
            base = merged.get(seg_id)
            if base is None:
                seg = self.segments.get(seg_id)
                if seg is None:
                    continue
                base = {
                    "segment_id": seg_id,
                    "score": 0.0,
                    "lexical_score": 0.0,
                    "semantic_score": 0.0,
                    "start_char": seg.start_char,
                    "end_char": seg.end_char,
                    "num_chars": seg.num_chars,
                    "preview": seg.preview,
                    "sources": [],
                }
                merged[seg_id] = base
            sem_norm = float(hit.get("semantic_score", hit.get("score", 0.0))) / max_sem
            base["semantic_score"] = max(float(base.get("semantic_score", 0.0)), sem_norm)
            base["score"] = float(base.get("score", 0.0)) + (semantic_weight * sem_norm)
            sources = base.get("sources", [])
            if "semantic" not in sources:
                sources.append("semantic")
            base["sources"] = sources

        hits = list(merged.values())
        hits.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return {
            "query": q,
            "top_k": top_k,
            "coarse_k": coarse_k,
            "semantic_top_k": semantic_top_k,
            "weights": {
                "lexical": lexical_weight,
                "semantic": semantic_weight,
            },
            "hits": hits[:top_k],
            "total_candidates": len(hits),
        }

    def _leaf_descendants(self, segment_id: str) -> List[str]:
        seg = self.segments[segment_id]
        if seg.level == 0:
            return [segment_id]
        leaves: List[str] = []
        for child_id in seg.children or []:
            leaves.extend(self._leaf_descendants(child_id))
        return leaves

    def _materialize_segment_text(self, seg: SegmentMeta) -> str:
        if seg.file:
            path = self.root_dir / seg.file
            if not path.exists():
                raise FileNotFoundError(f"Segment file missing: {path}")
            return path.read_text(encoding="utf-8")

        parts: List[str] = []
        for child_id in seg.children or []:
            child = self.segments[child_id]
            if child.file:
                parts.append((self.root_dir / child.file).read_text(encoding="utf-8"))
            else:
                parts.append(child.preview)
        return "\n".join(parts)

    @staticmethod
    def _segment_to_dict(seg: SegmentMeta) -> Dict[str, Any]:
        return {
            "segment_id": seg.segment_id,
            "level": seg.level,
            "start_char": seg.start_char,
            "end_char": seg.end_char,
            "num_chars": seg.num_chars,
            "preview": seg.preview,
            "file": seg.file,
            "children": list(seg.children or []),
        }


def preprocess_long_context(
    *,
    input_file: str | Path,
    output_dir: str | Path,
    chunk_chars: int = 16000,
    overlap_chars: int = 400,
    branch_factor: int = 8,
    semantic_index: bool = True,
    semantic_dim: int = _SEMANTIC_DIM_DEFAULT,
) -> Path:
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    if overlap_chars < 0 or overlap_chars >= chunk_chars:
        raise ValueError("overlap_chars must satisfy 0 <= overlap_chars < chunk_chars")
    if branch_factor < 2:
        raise ValueError("branch_factor must be >= 2")

    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    out_dir = Path(output_dir)
    seg_root = out_dir / "segments" / "level0"
    seg_root.mkdir(parents=True, exist_ok=True)

    stride = chunk_chars - overlap_chars
    level0_ids: List[str] = []
    segments: Dict[str, Dict[str, Any]] = {}

    cursor = 0
    seg_idx = 0
    buffer = ""
    total_chars = 0

    with input_path.open("r", encoding="utf-8") as f:
        while True:
            needed = max(1, chunk_chars - len(buffer))
            block = f.read(needed)
            if not block:
                break
            buffer += block

            while len(buffer) >= chunk_chars:
                text = buffer[:chunk_chars]
                seg_id = f"L0_{seg_idx:07d}"
                rel_path = Path("segments") / "level0" / f"{seg_id}.txt"
                (out_dir / rel_path).write_text(text, encoding="utf-8")

                start = cursor
                end = cursor + len(text)
                total_chars = max(total_chars, end)

                segments[seg_id] = {
                    "level": 0,
                    "start_char": start,
                    "end_char": end,
                    "num_chars": len(text),
                    "preview": text[:200].replace("\n", " "),
                    "file": str(rel_path),
                }
                level0_ids.append(seg_id)

                seg_idx += 1
                cursor += stride
                buffer = buffer[stride:]

    if buffer:
        seg_id = f"L0_{seg_idx:07d}"
        rel_path = Path("segments") / "level0" / f"{seg_id}.txt"
        (out_dir / rel_path).write_text(buffer, encoding="utf-8")

        start = cursor
        end = cursor + len(buffer)
        total_chars = max(total_chars, end)

        segments[seg_id] = {
            "level": 0,
            "start_char": start,
            "end_char": end,
            "num_chars": len(buffer),
            "preview": buffer[:200].replace("\n", " "),
            "file": str(rel_path),
        }
        level0_ids.append(seg_id)

    levels: Dict[int, List[str]] = {0: level0_ids}
    current_ids = level0_ids
    level = 1
    while len(current_ids) > 1:
        next_ids: List[str] = []
        for i in range(0, len(current_ids), branch_factor):
            children = current_ids[i : i + branch_factor]
            if not children:
                continue
            seg_id = f"L{level}_{len(next_ids):07d}"

            child_meta = [segments[cid] for cid in children]
            start = min(int(m["start_char"]) for m in child_meta)
            end = max(int(m["end_char"]) for m in child_meta)
            preview_parts = [str(child_meta[0]["preview"])]
            if len(child_meta) > 1:
                preview_parts.append(str(child_meta[-1]["preview"]))

            segments[seg_id] = {
                "level": level,
                "start_char": start,
                "end_char": end,
                "num_chars": end - start,
                "preview": " ... ".join(preview_parts)[:400],
                "children": children,
            }
            next_ids.append(seg_id)

        levels[level] = next_ids
        current_ids = next_ids
        level += 1

    semantic_index_file = ""
    semantic_backend = "none"
    if semantic_index and level0_ids:
        dim = max(64, int(semantic_dim))
        sem_path = out_dir / "semantic_index.jsonl"
        with sem_path.open("w", encoding="utf-8") as f:
            for seg_id in level0_ids:
                rel = str(segments[seg_id].get("file", "")).strip()
                text = ""
                if rel:
                    try:
                        text = (out_dir / rel).read_text(encoding="utf-8")
                    except Exception:
                        text = str(segments[seg_id].get("preview", ""))
                vec = _dense_embed(text, dim=dim)
                row = {
                    "segment_id": seg_id,
                    "embedding": [round(float(v), 6) for v in vec],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        semantic_index_file = str(sem_path.relative_to(out_dir))
        semantic_backend = "hashed_bow_v1"
        semantic_dim = dim

    manifest = {
        "created_at": utc_now_iso(),
        "source_file": str(input_path),
        "total_chars": total_chars,
        "chunk_chars": chunk_chars,
        "overlap_chars": overlap_chars,
        "branch_factor": branch_factor,
        "levels": {str(k): v for k, v in levels.items()},
        "segments": segments,
        "semantic_index_file": semantic_index_file,
        "semantic_dim": int(semantic_dim),
        "semantic_backend": semantic_backend,
    }

    manifest_path = out_dir / "manifest.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path
