"""Rich terminal presentation for the LongMemEval benchmark harness.

This is the ONLY module in ``benchmarks/`` that imports ``rich`` — all
terminal presentation lives here so pipeline logic in
``run_longmemeval.py`` stays free of output formatting.

Stream discipline:
    - Dynamic output (progress bars, spinners, failure lines) → stderr
      via ``live_console``.
    - Final report (header panel, tables, closing panel) → stdout via
      ``console``.
    - Standard ``logging`` is untouched and stays on stderr.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskID, TaskProgressColumn, TextColumn
from rich.status import Status
from rich.table import Table

logger = logging.getLogger(__name__)

console = Console()
"""Stdout console for the final report (header, tables, summary)."""

live_console = Console(stderr=True)
"""Stderr console for dynamic output (progress, spinners, errors)."""

_STATS_TEMPLATE = (
    "acc {task.fields[acc]} · R@1 {task.fields[r1]}"
    " · R@5 {task.fields[r5]} · R@10 {task.fields[r10]}"
)
"""Progress-bar stats template, fed by ``format_progress_fields`` keys."""


def print_run_header(
    *,
    variant: str,
    limit: int | None,
    reranker: bool,
    baseline: bool,
    judge_model: str | None,
    manifest_path: str | Path,
    resumed: bool,
    completed_count: int = 0,
    total_count: int = 0,
) -> None:
    """Print the benchmark run header panel to stdout.

    Args:
        variant: Dataset variant key (``"s"``, ``"oracle"``, ...).
        limit: ``--benchmark-limit`` value (``None`` for all questions).
        reranker: Whether the reranker is enabled for this run.
        baseline: Whether the pure-vector baseline run is included.
        judge_model: Judge/answer model name (may be ``None``).
        manifest_path: Checkpoint manifest path for this run.
        resumed: Whether this run resumes a previous manifest.
        completed_count: Questions already answered in the manifest.
        total_count: Total questions in this run.
    """
    questions = str(limit) if limit is not None else str(total_count)
    run_status = (
        f"Resuming — {completed_count}/{total_count} complete"
        if resumed
        else "Fresh run"
    )
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold")
    grid.add_column()
    grid.add_row("Variant", variant)
    grid.add_row("Questions", questions)
    grid.add_row("Reranker", "enabled" if reranker else "disabled")
    grid.add_row("Baseline", "included (pure vector)" if baseline else "—")
    grid.add_row("Judge model", judge_model or "unknown")
    grid.add_row("Manifest", str(manifest_path))
    grid.add_row("Run", run_status)
    console.print(Panel(grid, title="LongMemEval Benchmark"))


def format_progress_fields(
    correct_so_far: int,
    total_so_far: int,
    r1_so_far: int,
    r5_so_far: int,
    r10_so_far: int,
) -> dict[str, str]:
    """Format running stats for the question progress bar fields.

    Pure function of counts — no ``Progress`` instance needed, so the
    formatting is unit-testable without a live terminal.

    Args:
        correct_so_far: Questions judged correct so far.
        total_so_far: Questions judged so far (including resumed ones).
        r1_so_far: Questions with recall@1 hit so far.
        r5_so_far: Questions with recall@5 hit so far.
        r10_so_far: Questions with recall@10 hit so far.

    Returns:
        Field dict with keys ``acc``, ``r1``, ``r5``, ``r10`` holding
        pre-formatted percentages (``"—"`` when nothing is judged yet).
    """
    if total_so_far <= 0:
        return {"acc": "—", "r1": "—", "r5": "—", "r10": "—"}
    return {
        "acc": f"{correct_so_far / total_so_far:.1%}",
        "r1": f"{r1_so_far / total_so_far:.1%}",
        "r5": f"{r5_so_far / total_so_far:.1%}",
        "r10": f"{r10_so_far / total_so_far:.1%}",
    }


def make_question_progress(total: int) -> tuple[Progress, TaskID]:
    """Create the question progress bar (stderr, transient).

    The caller pre-advances resumed work with
    ``progress.update(task, completed=len(already_done))`` before the
    loop, then ``progress.update(task, advance=1,
    **format_progress_fields(...))`` per newly judged question —
    skipped-on-resume questions never advance the bar.

    Args:
        total: Total questions in this run (bar denominator).

    Returns:
        A ``(Progress, TaskID)`` pair; the caller drives it as a
        context manager (``with progress:``).
    """
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn(_STATS_TEMPLATE),
        console=live_console,
        transient=True,
    )
    task_id = progress.add_task(
        "Questions", total=total, acc="—", r1="—", r5="—", r10="—"
    )
    return progress, task_id


def question_failed(question_id: str, reasoning: str) -> None:
    """Print a single red failure line to stderr.

    Correct answers print nothing — the progress bar carries them.

    Args:
        question_id: The failed question's id.
        reasoning: Judge reasoning, collapsed to one line and truncated
            to 120 chars.
    """
    one_line = " ".join(reasoning.splitlines())
    short = one_line if len(one_line) <= 120 else f"{one_line[:117]}..."
    live_console.print(f"✗ {question_id} — {short}", style="red")


@contextmanager
def enrichment_status(label: str) -> Iterator[Status]:
    """Show a spinner on stderr for an ingest/enrichment phase.

    Additive to the existing ``logger.info`` lines, which stay as-is.

    Args:
        label: Spinner text (e.g. ``"[full] Waiting for enrichment"``).

    Yields:
        The active ``rich.status.Status`` object.
    """
    with live_console.status(label) as status:
        yield status


def print_results(
    *,
    system_rows: list[dict[str, str]],
    category_rows: list[dict[str, str]],
    saved_path: str | Path,
    manifest_path: str | Path,
    judge_errors: int = 0,
    graded_accuracy: float = 0.0,
    total: int = 0,
) -> None:
    """Print the final results report to stdout.

    Main system table, per-category breakdown, judge-error line, and a
    closing panel with the results path, manifest path, and resume hint.
    Rows are prebuilt by ``benchmarks.run_longmemeval.build_comparison_rows``
    — no metrics are computed or transformed here, only rendered.

    Args:
        system_rows: Prebuilt system-comparison rows with keys
            ``system``, ``accuracy``, ``r1``, ``r5``, ``r10``,
            ``conditions`` (all pre-formatted strings).
        category_rows: Prebuilt per-category rows with keys ``category``,
            ``accuracy``, ``count`` (all pre-formatted strings).
        saved_path: Path of the saved results JSON file.
        manifest_path: Checkpoint manifest path of this run.
        judge_errors: Questions whose judge LLM call failed.
        graded_accuracy: Accuracy excluding judge-error entries.
        total: Total questions judged.
    """
    console.print()
    console.rule("LongMemEval Benchmark Results")
    console.print()

    systems = Table("System", "Accuracy", "R@1", "R@5", "R@10", "Conditions")
    for row in system_rows:
        system = row.get("system")
        accuracy = row.get("accuracy")
        r1 = row.get("r1")
        r5 = row.get("r5")
        r10 = row.get("r10")
        conditions = row.get("conditions")
        if (
            system is None
            or accuracy is None
            or r1 is None
            or r5 is None
            or r10 is None
            or conditions is None
        ):
            logger.warning("skipping malformed system row: %r", row)
            continue
        systems.add_row(system, accuracy, r1, r5, r10, conditions)
    console.print(systems)
    console.print()

    categories = Table("Category", "Accuracy", "Count")
    for row in category_rows:
        category = row.get("category")
        accuracy = row.get("accuracy")
        count = row.get("count")
        if category is None or accuracy is None or count is None:
            logger.warning("skipping malformed category row: %r", row)
            continue
        categories.add_row(category, accuracy, count)
    console.print(categories)
    console.print()
    if total <= 0:
        console.print("Judge errors: 0/0")
    else:
        console.print(
            f"Judge errors: {judge_errors}/{total} — graded accuracy "
            f"(excluding errors): {graded_accuracy:.1%}"
        )
    console.print()

    console.print(
        Panel(
            f"Results: {saved_path}\n"
            f"Manifest: {manifest_path}\n"
            "Re-run the same command to resume an interrupted run.",
            title="Done",
        )
    )


def print_error(message: str) -> None:
    """Print an error message in red on stderr.

    Args:
        message: Error text (rendered as ``error: {message}``).
    """
    live_console.print(f"error: {message}", style="red")
