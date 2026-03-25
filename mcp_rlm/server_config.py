from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import shutil

from .multi_mcp import MCPServerSpec


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"1", "true", "yes", "y", "on"}:
            return True
        if raw in {"0", "false", "no", "n", "off"}:
            return False
    return default


def official_server_presets(workspace_root: str) -> List[Dict[str, Any]]:
    root = str(Path(workspace_root).resolve())
    return [
        {
            "alias": "ofs",
            "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem", root],
            "cwd": root,
        },
        {
            "alias": "omemory",
            "command": ["npx", "-y", "@modelcontextprotocol/server-memory"],
            "cwd": root,
        },
        {
            "alias": "ofetch",
            "command": ["uvx", "mcp-server-fetch"],
            "cwd": root,
        },
        {
            "alias": "ogit",
            "command": ["uvx", "mcp-server-git", "--repository", root],
            "cwd": root,
        },
        {
            "alias": "oseq",
            "command": ["npx", "-y", "@modelcontextprotocol/server-sequential-thinking"],
            "cwd": root,
        },
    ]


def _parse_server_specs(raw_servers: List[Dict[str, Any]], *, skip_unavailable: bool) -> List[MCPServerSpec]:
    specs: List[MCPServerSpec] = []
    seen_alias = set()

    for item in raw_servers:
        if not isinstance(item, dict):
            continue

        alias = str(item.get("alias", "")).strip()
        command = item.get("command", [])

        if not alias or not isinstance(command, list) or not command:
            continue
        if alias in seen_alias:
            continue

        executable = str(command[0]).strip()
        if skip_unavailable and executable and (shutil.which(executable) is None):
            continue

        spec = MCPServerSpec(
            alias=alias,
            command=[str(x) for x in command],
            cwd=str(item.get("cwd", "")).strip() or None,
            env=item.get("env") if isinstance(item.get("env"), dict) else None,
            max_concurrency=int(item.get("max_concurrency", 24)),
            prefer_official_sdk=_as_bool(item.get("prefer_official_sdk", True), True),
            strict_official_sdk=_as_bool(item.get("strict_official_sdk", False), False),
        )
        specs.append(spec)
        seen_alias.add(alias)

    return specs


def load_mcp_extension_config(
    *,
    workspace_root: str,
    config_path: str = "",
    enable_official_presets: bool = False,
    skip_unavailable: bool = False,
) -> Tuple[List[MCPServerSpec], Dict[str, Any]]:
    raw_servers: List[Dict[str, Any]] = []
    fanout: Dict[str, Any] = {
        "root_extra_object_fanout": [],
        "leaf_extra_object_fanout": [],
    }

    if enable_official_presets:
        raw_servers.extend(official_server_presets(workspace_root))

    if config_path:
        cfg_file = Path(config_path).resolve()
        payload = json.loads(cfg_file.read_text(encoding="utf-8-sig"))
        if isinstance(payload, dict):
            if isinstance(payload.get("servers"), list):
                raw_servers.extend([x for x in payload["servers"] if isinstance(x, dict)])
            if isinstance(payload.get("root_extra_object_fanout"), list):
                fanout["root_extra_object_fanout"] = payload.get("root_extra_object_fanout", [])
            if isinstance(payload.get("leaf_extra_object_fanout"), list):
                fanout["leaf_extra_object_fanout"] = payload.get("leaf_extra_object_fanout", [])

    specs = _parse_server_specs(raw_servers, skip_unavailable=skip_unavailable)
    return specs, fanout
