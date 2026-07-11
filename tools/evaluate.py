from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.evaluator import evaluate_manifest_results, write_manifest_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and aggregate Plan2 embodied-agent evaluation results."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/evaluation/plan2_multiscene_v1.json"),
        help="Path to the frozen evaluation manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory containing per-episode result files and receiving summary.json.",
    )
    parser.add_argument("--split", help="Optional split filter.")
    parser.add_argument("--group", help="Optional group filter, e.g. oracle/non_oracle.")
    parser.add_argument(
        "--dry-run",
        "--validate-only",
        action="store_true",
        dest="dry_run",
        help="Validate manifest and emit metadata without reading episode results.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print summary only; do not write output-dir/summary.json.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = evaluate_manifest_results(
            args.manifest,
            args.output_dir,
            split=args.split,
            group=args.group,
            dry_run=args.dry_run,
        )
        if not args.no_write:
            write_manifest_summary(summary, args.output_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    except Exception as exc:
        print(f"evaluation failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
