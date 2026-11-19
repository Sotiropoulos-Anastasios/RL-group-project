#!/usr/bin/env python3
"""Build the processed DialOp dataset used by the RL project."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from preprocessing import build_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert DialOp optimization JSONL into RL-friendly JSONL."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=PROJECT_DIR / "data" / "raw" / "optimization.jsonl",
        help="DialOp optimization JSONL input.",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=PROJECT_DIR / "data" / "processed",
        help="Directory for processed JSONL and manifest files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_dataset(args.input, args.output_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
