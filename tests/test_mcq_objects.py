from __future__ import annotations

import unittest

from mcp_rlm.mcp import MCPCall, MCPClient, MCPInvocationContext, MCPRegistry, register_builtin_objects


class TestMCQObjects(unittest.IsolatedAsyncioTestCase):
    async def test_score_mcq_choices_prefers_correct_option(self) -> None:
        registry = MCPRegistry()
        register_builtin_objects(registry)
        client = MCPClient(registry)

        result = await client.call(
            MCPCall(
                "score_mcq_choices",
                {
                    "question": "Which city is the capital of France?",
                    "choices": {
                        "A": "Berlin",
                        "B": "Madrid",
                        "C": "Paris",
                        "D": "Rome",
                    },
                    "text": "The capital of France is Paris. Berlin is the capital of Germany.",
                    "segment_id": "seg-1",
                },
            ),
            MCPInvocationContext(episode_id="ep", group_id="grp"),
        )

        self.assertTrue(result.ok)
        output = result.output if isinstance(result.output, dict) else {}
        self.assertEqual(output.get("best_choice"), "C")

    async def test_vote_choice_scores(self) -> None:
        registry = MCPRegistry()
        register_builtin_objects(registry)
        client = MCPClient(registry)

        result = await client.call(
            MCPCall(
                "vote_choice_scores",
                {
                    "maps": [
                        {"choice_scores": {"A": 0.1, "B": 0.2, "C": 1.3, "D": 0.1}},
                        {"choice_scores": {"A": 0.2, "B": 0.1, "C": 0.8, "D": 0.0}},
                    ],
                    "weights": [1.0, 0.8],
                    "elimination_penalties": {"A": 0.1, "B": 0.0, "C": 0.0, "D": 0.2},
                },
            ),
            MCPInvocationContext(episode_id="ep", group_id="grp"),
        )

        self.assertTrue(result.ok)
        output = result.output if isinstance(result.output, dict) else {}
        self.assertEqual(output.get("best_choice"), "C")

    async def test_rerank_hits_with_choices(self) -> None:
        registry = MCPRegistry()
        register_builtin_objects(registry)
        client = MCPClient(registry)

        result = await client.call(
            MCPCall(
                "rerank_hits_with_choices",
                {
                    "query": "capital of France",
                    "choices": {
                        "A": "Berlin",
                        "B": "Madrid",
                        "C": "Paris",
                        "D": "Rome",
                    },
                    "hits": [
                        {"segment_id": "s1", "score": 1.0, "preview": "Berlin is in Germany."},
                        {"segment_id": "s2", "score": 0.9, "preview": "The capital of France is Paris."},
                    ],
                    "top_k": 2,
                },
            ),
            MCPInvocationContext(episode_id="ep", group_id="grp"),
        )

        self.assertTrue(result.ok)
        output = result.output if isinstance(result.output, dict) else {}
        hits = output.get("hits", []) if isinstance(output.get("hits", []), list) else []
        self.assertGreaterEqual(len(hits), 1)
        self.assertEqual(str(hits[0].get("segment_id")), "s2")

    async def test_normalize_mcq_answer(self) -> None:
        registry = MCPRegistry()
        register_builtin_objects(registry)
        client = MCPClient(registry)

        result = await client.call(
            MCPCall("normalize_mcq_answer", {"response": "The correct answer is (B)", "fallback": "A"}),
            MCPInvocationContext(episode_id="ep", group_id="grp"),
        )

        self.assertTrue(result.ok)
        output = result.output if isinstance(result.output, dict) else {}
        self.assertEqual(output.get("answer"), "B")
        self.assertTrue(bool(output.get("is_valid")))


if __name__ == "__main__":
    unittest.main()
