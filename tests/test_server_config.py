from __future__ import annotations

import unittest

from mcp_rlm.server_config import load_mcp_extension_config, official_server_presets


class TestServerConfig(unittest.TestCase):
    def test_official_presets_contains_expected_aliases(self) -> None:
        presets = official_server_presets('.')
        aliases = {str(x.get('alias', '')) for x in presets if isinstance(x, dict)}
        self.assertIn('ofs', aliases)
        self.assertIn('omemory', aliases)
        self.assertIn('ofetch', aliases)
        self.assertIn('ogit', aliases)
        self.assertIn('oseq', aliases)

    def test_load_extension_default(self) -> None:
        specs, fanout = load_mcp_extension_config(workspace_root='.', config_path='', enable_official_presets=False)
        self.assertEqual(specs, [])
        self.assertIsInstance(fanout, dict)
        self.assertIn('root_extra_object_fanout', fanout)
        self.assertIn('leaf_extra_object_fanout', fanout)


if __name__ == '__main__':
    unittest.main()
