from __future__ import annotations

import argparse
import sys

from .runner import BenchmarkError, execute


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark OpenAI-compatible streamed chat completions")
    parser.add_argument("--config", required=True, help="JSON run configuration")
    parser.add_argument("--output", default="runs", help="directory for JSON results and Markdown report")
    parser.add_argument("--include-text", action="store_true", help="opt in to storing prompts and responses")
    args = parser.parse_args()
    try:
        result, report = execute(args.config, args.output, include_text=args.include_text)
    except (BenchmarkError, KeyError, TypeError, ValueError) as exc:
        print(f"inference-bench: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(f"Results: {result}")
    print(f"Report:  {report}")


if __name__ == "__main__":
    main()
