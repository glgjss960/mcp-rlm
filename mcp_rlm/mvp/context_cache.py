from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from ..long_context import LongContextStore


@dataclass
class _StoreEntry:
    manifest_path: Path
    store: LongContextStore


class LongContextStoreCache:
    def __init__(self) -> None:
        self._cache: Dict[str, _StoreEntry] = {}

    def get(self, manifest_path: str | Path) -> LongContextStore:
        path = Path(manifest_path).resolve()
        key = str(path)
        entry = self._cache.get(key)
        if entry is not None and entry.manifest_path == path:
            return entry.store

        store = LongContextStore(path)
        self._cache[key] = _StoreEntry(manifest_path=path, store=store)
        return store

