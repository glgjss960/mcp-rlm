from __future__ import annotations

import asyncio
import unittest

from mcp_rlm import MCPClient, MCPRLMRuntime, MCPRegistry, ProgramRegistry, SharedMemory
from mcp_rlm.protocol import WritePolicy
from mcp_rlm.training import build_verl_step_rows
from mcp_rlm.types import WriteReason


async def _large_result_program(ctx):
    await ctx.call_object("diag/huge", {"chars": 18000})
    return await ctx.finalize({"task_score": 1.0})


async def _manual_flush_program(ctx):
    await ctx.call_object("diag/huge", {"chars": 2000})
    await ctx.flush_context_pressure(note="manual-force", force=True)
    return await ctx.finalize({"task_score": 1.0})


class TestContextPressure(unittest.TestCase):
    def _build_runtime(self, *, threshold: float, min_gap: int = 1, min_delta: float = 0.0) -> MCPRLMRuntime:
        registry = MCPRegistry()

        def huge(payload, _ctx):
            chars = max(1, int(payload.get("chars", 1000)))
            return {"blob": "x" * chars}

        registry.register("diag/huge", huge)

        programs = ProgramRegistry()
        programs.register("large_result", _large_result_program)
        programs.register("manual_flush", _manual_flush_program)

        runtime = MCPRLMRuntime(
            program_registry=programs,
            mcp_client=MCPClient(registry, max_concurrency=8),
            memory=SharedMemory(),
            write_policy=WritePolicy(context_pressure_threshold=threshold),
            context_pressure_auto_write=True,
            context_pressure_min_step_gap=min_gap,
            context_pressure_min_delta=min_delta,
        )
        return runtime

    def test_auto_context_pressure_write(self) -> None:
        async def runner() -> None:
            runtime = self._build_runtime(threshold=0.2, min_gap=1, min_delta=0.0)
            trace = await runtime.run_episode(goal="pressure", program="large_result", input_payload={})

            reasons = [event.reason for event in trace.memory_events]
            self.assertIn(WriteReason.CONTEXT_PRESSURE, reasons)
            self.assertIn(WriteReason.FINALIZE, reasons)

            pressure_events = [e for e in trace.memory_events if e.reason == WriteReason.CONTEXT_PRESSURE]
            self.assertGreaterEqual(len(pressure_events), 1)
            self.assertTrue(all("context_pressure" in e.key for e in pressure_events))

            rows = build_verl_step_rows([trace])
            self.assertTrue(rows)
            extra = rows[0].get("extra_info", {})
            self.assertIn("context_pressure_write_ratio", extra)
            self.assertIn("context_pressure_write_count", extra)
            self.assertGreaterEqual(float(extra.get("context_pressure_write_ratio", 0.0)), 0.0)

        asyncio.run(runner())

    def test_manual_flush_force_write(self) -> None:
        async def runner() -> None:
            runtime = self._build_runtime(threshold=0.99)
            trace = await runtime.run_episode(goal="manual", program="manual_flush", input_payload={})

            pressure_events = [e for e in trace.memory_events if e.reason == WriteReason.CONTEXT_PRESSURE]
            self.assertGreaterEqual(len(pressure_events), 1)
            has_manual = any(str(e.content.get("note", "")) == "manual-force" for e in pressure_events)
            self.assertTrue(has_manual)

        asyncio.run(runner())


if __name__ == "__main__":
    unittest.main()
