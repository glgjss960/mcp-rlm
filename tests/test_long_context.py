from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from mcp_rlm import LongContextStore, preprocess_long_context


class TestLongContext(unittest.TestCase):
    def test_preprocess_and_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_file = tmp_path / "long.txt"
            lines = []
            for i in range(300):
                lines.append(f"line {i}: noise text about unrelated info.")
            lines.append("critical note: project atlas secret code is 991-C88.")
            for i in range(300, 600):
                lines.append(f"line {i}: more unrelated data.")
            input_file.write_text("\n".join(lines), encoding="utf-8")

            manifest = preprocess_long_context(
                input_file=input_file,
                output_dir=tmp_path / "store",
                chunk_chars=1200,
                overlap_chars=100,
                branch_factor=4,
            )
            store = LongContextStore(manifest)

            stats = store.context_stats()
            self.assertGreater(stats["leaf_segments"], 2)

            search = store.search_hierarchical(query="What is the secret code for project atlas?", top_k=5, coarse_k=8)
            self.assertGreater(len(search["hits"]), 0)

            first_id = search["hits"][0]["segment_id"]
            text = store.read_segment(first_id)["text"]
            self.assertIn("atlas", text.lower())


if __name__ == "__main__":
    unittest.main()
