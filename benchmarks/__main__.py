"""Entry point for ``python -m benchmarks`` (run from the repository root)."""

from __future__ import annotations

import asyncio
import sys

from benchmarks.cli import BenchmarkConfigError, main
from benchmarks.display import print_error

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except BenchmarkConfigError as exc:
        print_error(str(exc))
        sys.exit(2)
    except BaseExceptionGroup as exc_group:
        leaf: BaseException = exc_group.exceptions[0]
        while isinstance(leaf, BaseExceptionGroup):
            leaf = leaf.exceptions[0]
        print_error(f"{type(leaf).__name__}: {leaf}")
        sys.exit(1)
