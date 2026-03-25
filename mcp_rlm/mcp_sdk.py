from __future__ import annotations

from pathlib import Path
from typing import Iterable
import importlib
import os
import sys


def _candidate_sdk_paths() -> Iterable[Path]:
    env_path = os.getenv('MCP_PYTHON_SDK_PATH', '').strip()
    if env_path:
        yield Path(env_path)

    here = Path(__file__).resolve()
    yield here.parents[2] / 'python-sdk' / 'src'
    yield Path.cwd() / 'python-sdk' / 'src'


def mcp_sdk_status() -> tuple[bool, str]:
    """Return MCP SDK availability and a human-readable detail string."""

    try:
        importlib.import_module('mcp')
        return True, 'Imported MCP SDK from current sys.path'
    except Exception as exc:  # pragma: no cover - environment dependent
        first_error = exc

    for candidate in _candidate_sdk_paths():
        if not candidate.exists():
            continue
        path_str = str(candidate)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
        try:
            importlib.import_module('mcp')
            return True, f'Imported MCP SDK from {candidate}'
        except Exception as exc:  # pragma: no cover - environment dependent
            first_error = exc

    return False, f'Unable to import MCP SDK: {type(first_error).__name__}: {first_error}'


def ensure_mcp_sdk() -> None:
    ok, detail = mcp_sdk_status()
    if ok:
        return
    raise ModuleNotFoundError(
        'Official MCP Python SDK import failed. '
        'Install mcp dependencies or set MCP_PYTHON_SDK_PATH to <python-sdk>/src. '
        f'Detail: {detail}'
    )
