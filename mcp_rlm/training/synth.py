from __future__ import annotations

from typing import Any, Dict, List
import asyncio
import random

from .. import MCPClient, MCPRegistry, MCPRLMRuntime, ProgramRegistry, SharedMemory
from ..mcp import register_builtin_objects
from ..programs import register_builtin_programs
from ..types import EpisodeTrace


def _build_runtime() -> MCPRLMRuntime:
    object_registry = MCPRegistry()
    register_builtin_objects(object_registry)

    program_registry = ProgramRegistry()
    register_builtin_programs(program_registry)

    return MCPRLMRuntime(
        program_registry=program_registry,
        mcp_client=MCPClient(object_registry, max_concurrency=64),
        memory=SharedMemory(),
    )


def build_task(seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    topic = rng.choice(["atlas", "quasar", "mercury", "aurora", "nova"])
    code = f"{rng.randint(100, 999)}-{rng.choice(['A', 'B', 'C'])}{rng.randint(10, 99)}"
    noise_words = ["alpha", "beta", "gamma", "delta", "sigma", "kappa"]

    documents: List[str] = []
    for i in range(18):
        noise = " ".join(rng.choice(noise_words) for _ in range(20))
        documents.append(f"Doc {i}: general notes about {topic}. {noise}.")

    answer_doc = rng.randint(0, len(documents) - 1)
    documents[answer_doc] += (
        f" Confidential statement: the secret code for project {topic} is {code}."
    )

    query = f"What is the secret code for project {topic}?"
    return {
        "query": query,
        "documents": documents,
        "expected_answer": code,
    }


async def _run_one(seed: int) -> EpisodeTrace:
    task = build_task(seed)
    runtime = _build_runtime()
    trace = await runtime.run_episode(
        goal=task["query"],
        program="root_map_reduce",
        input_payload={
            "query": task["query"],
            "documents": task["documents"],
            "chunk_size": 4,
        },
    )

    if isinstance(trace.root_output, dict):
        answer_text = str(trace.root_output.get("answer", "")).lower()
        expected = str(task["expected_answer"]).lower()
        trace.root_output["expected_answer"] = task["expected_answer"]
        trace.root_output["is_correct"] = expected in answer_text

    return trace


async def synthesize_traces(
    *,
    num_episodes: int,
    parallelism: int = 8,
    seed: int = 7,
) -> List[EpisodeTrace]:
    semaphore = asyncio.Semaphore(parallelism)

    async def worker(offset: int) -> EpisodeTrace:
        async with semaphore:
            return await _run_one(seed + offset)

    tasks = [asyncio.create_task(worker(i)) for i in range(num_episodes)]
    return await asyncio.gather(*tasks)
