from __future__ import annotations

import unittest

from mcp_rlm.mcp_sdk import mcp_sdk_status
from mcp_rlm.multi_mcp import MCPServerSpec, MultiServerMCPClient
from mcp_rlm.stdio_mcp_client import StdioMCPClient


class TestMCPSDKMode(unittest.TestCase):
    def test_sdk_status_shape(self) -> None:
        ok, detail = mcp_sdk_status()
        self.assertIsInstance(ok, bool)
        self.assertIsInstance(detail, str)
        self.assertTrue(detail)

    def test_strict_mode_matches_sdk_availability(self) -> None:
        ok, _ = mcp_sdk_status()
        if ok:
            client = StdioMCPClient(command=['python', '-c', 'print(1)'], strict_official_sdk=True)
            self.assertTrue(client.using_official_sdk)
        else:
            with self.assertRaises(RuntimeError):
                StdioMCPClient(command=['python', '-c', 'print(1)'], strict_official_sdk=True)

    def test_multi_server_spec_flags(self) -> None:
        specs = [
            MCPServerSpec(
                alias='x',
                command=['python', '-c', 'print(1)'],
                prefer_official_sdk=False,
                strict_official_sdk=False,
            )
        ]
        client = MultiServerMCPClient(specs=specs)
        self.assertIsNotNone(client)


if __name__ == '__main__':
    unittest.main()
