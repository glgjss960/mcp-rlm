from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from mcp_rlm import FileSharedMemory
from mcp_rlm.types import MemoryObjectType, WriteReason


class TestFileSharedMemory(unittest.IsolatedAsyncioTestCase):
    async def test_persistence_and_cas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mem1 = FileSharedMemory(Path(tmp) / "memory")

            first = await mem1.append(
                episode_id="ep1",
                group_id="grp1",
                key="fact/demo",
                object_type=MemoryObjectType.FACT,
                reason=WriteReason.VALUE_EVENT,
                content={"v": 1},
                expected_version=0,
            )
            self.assertEqual(first.version, 1)

            second = await mem1.append(
                episode_id="ep1",
                group_id="grp2",
                key="fact/demo",
                object_type=MemoryObjectType.FACT,
                reason=WriteReason.VALUE_EVENT,
                content={"v": 2},
                expected_version=1,
            )
            self.assertEqual(second.version, 2)

            mem2 = FileSharedMemory(Path(tmp) / "memory")
            latest = await mem2.read_latest("fact/demo")
            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertEqual(latest.version, 2)
            self.assertEqual(latest.content["v"], 2)


if __name__ == "__main__":
    unittest.main()
