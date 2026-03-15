from __future__ import annotations

import unittest

from mcp_rlm.memory import MemoryConflictError, SharedMemory
from mcp_rlm.types import MemoryObjectType, WriteReason


class TestSharedMemory(unittest.IsolatedAsyncioTestCase):
    async def test_cas_conflict(self) -> None:
        memory = SharedMemory()

        first = await memory.append(
            episode_id="ep_1",
            group_id="grp_1",
            key="fact/demo",
            object_type=MemoryObjectType.FACT,
            reason=WriteReason.VALUE_EVENT,
            content={"x": 1},
            expected_version=0,
        )
        self.assertEqual(first.version, 1)

        with self.assertRaises(MemoryConflictError):
            await memory.append(
                episode_id="ep_1",
                group_id="grp_2",
                key="fact/demo",
                object_type=MemoryObjectType.FACT,
                reason=WriteReason.VALUE_EVENT,
                content={"x": 2},
                expected_version=0,
            )

        second = await memory.append(
            episode_id="ep_1",
            group_id="grp_2",
            key="fact/demo",
            object_type=MemoryObjectType.FACT,
            reason=WriteReason.VALUE_EVENT,
            content={"x": 2},
            expected_version=1,
        )
        self.assertEqual(second.version, 2)


if __name__ == "__main__":
    unittest.main()
