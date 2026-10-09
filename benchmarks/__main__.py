"""Entry point for ``python -m benchmarks`` (run from the repository root)."""

from __future__ import annotations

import asyncio
import sys

from benchmarks.cli import BenchmarkConfigError, main

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except BenchmarkConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
