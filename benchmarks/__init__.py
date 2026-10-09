"""LongMemEval benchmark harness — a standalone CLI, not a test suite.

Run from the repository root with ``python -m benchmarks``.  The harness
ingests LongMemEval conversations into a live OpenZync instance, measures
retrieval quality (R@k) and end-to-end QA accuracy, and checkpoints every
question so interrupted runs resume instead of restarting.

Modules:
    - ``cli`` — argument parsing, env loading, dependency factories, and
      checkpoint resolution (``--fresh`` / ``--resume``).
    - ``checkpoint`` — manifest management under ``results/.in_progress/``.
    - ``run_longmemeval`` — the benchmark flow itself.
"""
