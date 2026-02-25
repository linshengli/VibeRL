from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.generator import SFTDataGenerator, load_samples_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SFT jsonl to verl parquet")
    parser.add_argument("--input", required=True, help="Input samples.jsonl path")
    parser.add_argument("--output-dir", required=True, help="Output parquet directory")
    parser.add_argument("--split", type=float, default=0.9, help="Train split ratio")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"input file not found: {input_path}")

    samples = load_samples_jsonl(str(input_path))
    generator = SFTDataGenerator()
    train_path, val_path = generator.save_to_parquet(samples, args.output_dir, split=args.split)

    print(f"train: {train_path}")
    print(f"val: {val_path}")


if __name__ == "__main__":
    main()
