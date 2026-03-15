from __future__ import annotations

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp_rlm import preprocess_long_context


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess long text into hierarchical context store")
    parser.add_argument("--input", type=str, required=True, help="Path to raw text file")
    parser.add_argument("--out", type=str, required=True, help="Output context store directory")
    parser.add_argument("--chunk-chars", type=int, default=16000)
    parser.add_argument("--overlap-chars", type=int, default=400)
    parser.add_argument("--branch-factor", type=int, default=8)
    args = parser.parse_args()

    manifest_path = preprocess_long_context(
        input_file=args.input,
        output_dir=args.out,
        chunk_chars=args.chunk_chars,
        overlap_chars=args.overlap_chars,
        branch_factor=args.branch_factor,
    )

    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
