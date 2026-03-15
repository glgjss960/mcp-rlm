from __future__ import annotations

import asyncio
import time
import unittest

from mcp_rlm import (
    MCPClient,
    MCPRegistry,
    MCPRLMRuntime,
    ProgramRegistry,
    SharedMemory,
    register_builtin_objects,
)


async def child_program(ctx):
    await ctx.call_object("sleep", {"seconds": 0.25})
    return await ctx.finalize({"task_score": 1.0})


async def parent_program(ctx):
    specs = [
        {
            "goal": f"child-{i}",
            "program": "child",
            "input_payload": {},
        }
        for i in range(5)
    ]
    ids = await ctx.spawn_groups(specs)
    joined = await ctx.join_groups(ids)
    return await ctx.finalize({"children": len(joined), "task_score": 1.0})


class TestRuntimeParallel(unittest.TestCase):
    def test_spawn_join_parallel(self) -> None:
        async def runner() -> float:
            object_registry = MCPRegistry()
            register_builtin_objects(object_registry)

            program_registry = ProgramRegistry()
            program_registry.register("child", child_program)
            program_registry.register("parent", parent_program)

            runtime = MCPRLMRuntime(
                program_registry=program_registry,
                mcp_client=MCPClient(object_registry, max_concurrency=32),
                memory=SharedMemory(),
                max_group_concurrency=32,
            )

            start = time.perf_counter()
            trace = await runtime.run_episode(goal="parallel test", program="parent", input_payload={})
            elapsed = time.perf_counter() - start

            self.assertTrue(trace.success)
            return elapsed

        elapsed = asyncio.run(runner())
        self.assertLess(elapsed, 0.8)


if __name__ == "__main__":
    unittest.main()
