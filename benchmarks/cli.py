"""CLI entry, environment loading, and dependency factories for the harness.

Benchmark options are argparse flags on ``python -m benchmarks``; missing
credentials fail loudly with exit code 2 instead of silently skipping the
run.

This module is the composition root — everything the benchmark needs is
built here (env, args, LLM backend, HTTP client, checkpoint manifest).
``run_longmemeval`` imports from here; this module never imports from it
at module level (``main`` defers that import to keep the graph acyclic).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess  # noqa: S404 — intentional git metadata read
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import httpx
from dotenv import load_dotenv

from benchmarks.checkpoint import IN_PROGRESS_DIR, Checkpoint
from benchmarks.longmemeval_evaluator import RETRIEVAL_JUDGE_SYSTEM_PROMPT
from benchmarks.longmemeval_utils import DATASET_FILES, load_dataset
from core.llm_backends import OpenAIBackend, OpenAILikeBackend

if TYPE_CHECKING:
    from collections.abc import Sequence

    from core.llm import LLMBackend

logger = logging.getLogger(__name__)

_ENV_FILE: Path = Path(__file__).resolve().parent / ".env"
"""Path to the benchmark-local .env (credentials, API keys)."""

_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
"""Repository root — parent of the ``benchmarks`` package."""


class BenchmarkConfigError(Exception):
    """Raised when benchmark configuration is missing or inconsistent.

    ``benchmarks/__main__.py`` maps this to ``sys.exit(2)`` with the
    message on stderr — never to a silent skip.
    """


def load_env() -> None:
    """Load benchmark-local environment variables from ``benchmarks/.env``.

    The .env file is optional — if absent, the process relies on exported
    environment variables.  Exported variables always win over file
    values (``load_dotenv`` default).
    """
    if not _ENV_FILE.exists():
        logger.info(
            "No benchmark .env at %s — relying on exported env vars",
            _ENV_FILE,
        )
        return
    load_dotenv(_ENV_FILE)
    logger.info("Loaded benchmark environment from %s", _ENV_FILE)


def configure_logging() -> None:
    """Send harness INFO logs to stderr with timestamps.

    Without an explicit root handler, ``python -m benchmarks`` drops all
    INFO logs (Python's last-resort handler only emits WARNING and up) —
    the harness would run silently, with no enrichment progress, no
    checkpoint skips, and nothing to debug a run from.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def parse_args(argv: Sequence[str] | None = None) -> SimpleNamespace:
    """Parse benchmark CLI arguments.

    Args:
        argv: Argument list to parse (defaults to ``sys.argv[1:]``).

    Returns:
        A ``SimpleNamespace`` with ``variant``, ``benchmark_limit``,
        ``baseline``, ``reranker``, ``resume``, ``fresh``, ``base_url``,
        ``ingest``, and ``workers``.
    """
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks",
        description=(
            "Run the LongMemEval benchmark against a live OpenZync "
            "instance. Requires BENCH_EMAIL/BENCH_PASSWORD and an LLM key "
            "(NVIDIA_API_KEY or OPENAI_API_KEY); interrupted runs resume "
            "automatically."
        ),
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="s",
        choices=sorted(DATASET_FILES),
        help="Dataset variant: 's' (small, default) or 'oracle'",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        default=False,
        help="Run pure vector baseline for comparison (no reranker)",
    )
    parser.add_argument(
        "--reranker",
        action="store_true",
        default=False,
        help="Run with cross-encoder reranker enabled",
    )
    parser.add_argument(
        "--benchmark-limit",
        type=int,
        default=None,
        help="Limit number of questions for quick runs (default: all)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Resume a specific checkpoint manifest (default: auto-resume "
            "the newest incomplete manifest matching the current config)"
        ),
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        default=False,
        help="Ignore existing checkpoints and start a new run",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("OPENZYNC_BASE_URL", "http://localhost:8000"),
        help=(
            "OpenZync API base URL "
            "(default: $OPENZYNC_BASE_URL or http://localhost:8000)"
        ),
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        default=False,
        help=(
            "Permit ingesting dataset conversations into a new or empty "
            "project. Without it the harness is query-only and errors if "
            "ingestion would be required."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Questions judged concurrently (default: 1). Higher values "
            "shorten wall-clock only — LLM cost is identical "
            "(1 call per question)."
        ),
    )
    parsed = parser.parse_args(argv)
    if not 1 <= parsed.workers <= 32:
        raise BenchmarkConfigError("--workers must be between 1 and 32")
    return SimpleNamespace(
        variant=parsed.variant,
        benchmark_limit=parsed.benchmark_limit,
        baseline=parsed.baseline,
        reranker=parsed.reranker,
        resume=parsed.resume,
        fresh=parsed.fresh,
        base_url=parsed.base_url,
        ingest=parsed.ingest,
        workers=parsed.workers,
    )


def require_login_creds() -> None:
    """Verify benchmark login credentials before any network call.

    Raises:
        BenchmarkConfigError: If ``BENCH_EMAIL`` or ``BENCH_PASSWORD`` is
            unset, naming every missing variable.
    """
    missing = [
        name for name in ("BENCH_EMAIL", "BENCH_PASSWORD") if not os.environ.get(name)
    ]
    if missing:
        raise BenchmarkConfigError(
            f"Missing benchmark login credentials: {', '.join(missing)}. "
            "Set them in benchmarks/.env or export them before running."
        )


def build_llm_backend() -> LLMBackend:
    """Create the LLM backend for retrieval judging (NVIDIA-first).

    ``NVIDIA_API_KEY`` configures an ``OpenAILikeBackend`` against the
    NVIDIA OpenAI-compatible endpoint; ``OPENAI_API_KEY`` configures
    ``OpenAIBackend`` with its default chat model.

    Returns:
        A configured ``LLMBackend``.

    Raises:
        BenchmarkConfigError: If neither ``NVIDIA_API_KEY`` nor
            ``OPENAI_API_KEY`` is set.
    """
    nvidia_key = os.environ.get("NVIDIA_API_KEY")
    if nvidia_key:
        return OpenAILikeBackend(
            base_url=os.environ.get(
                "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
            ),
            api_key=nvidia_key,
            model=os.environ.get("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b"),
        )

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return OpenAIBackend(api_key=api_key)

    raise BenchmarkConfigError(
        "No LLM backend configured: set NVIDIA_API_KEY (preferred; "
        "optional NVIDIA_BASE_URL and NVIDIA_MODEL overrides) or "
        "OPENAI_API_KEY in benchmarks/.env or the environment."
    )


def make_api_client(base_url: str) -> httpx.AsyncClient:
    """Create the HTTP client for the OpenZync API.

    Args:
        base_url: API base URL (e.g. ``http://localhost:8000``).

    Returns:
        An ``httpx.AsyncClient`` with a 60 s default timeout, shared for
        the whole run (connection reuse keeps ingest fast).
    """
    return httpx.AsyncClient(base_url=base_url, timeout=60.0)


def get_git_info() -> tuple[str | None, str | None]:
    """Read the current git commit hash and project version.

    Returns:
        A tuple of ``(git_commit_hash, project_version)``; the version
        is display-only metadata and may be ``None``.

    Raises:
        BenchmarkConfigError: If the git commit cannot be determined —
            without it the run has no benchmark identity and could
            auto-resume a manifest from different code.
    """
    git_commit: str | None = None
    version: str | None = None

    try:
        result = subprocess.run(  # noqa: S607
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
        )
    except Exception as exc:
        raise BenchmarkConfigError(
            "cannot determine git commit: 'git rev-parse HEAD' failed "
            f"({exc}). Run from a git checkout so the benchmark run is "
            "identifiable."
        ) from exc
    if result.returncode != 0:
        raise BenchmarkConfigError(
            "cannot determine git commit: 'git rev-parse HEAD' exited "
            f"with status {result.returncode} "
            f"({result.stderr.strip()}). Run from a git checkout so the "
            "benchmark run is identifiable."
        )
    git_commit = result.stdout.strip()
    if not git_commit:
        raise BenchmarkConfigError(
            "cannot determine git commit: 'git rev-parse HEAD' returned "
            "empty output. Run from a git checkout so the benchmark run "
            "is identifiable."
        )

    try:
        import tomllib  # Python 3.11+

        pyproject = _REPO_ROOT / "pyproject.toml"
        if pyproject.exists():
            with open(pyproject, "rb") as f:
                data = tomllib.load(f)
            version = data.get("project", {}).get("version", None)
    except Exception:  # noqa: S110 — best-effort, safe to ignore
        pass

    return git_commit, version


def dataset_question_ids(dataset: list[dict[str, Any]]) -> list[str]:
    """Return the stable id of every question in a dataset, in order.

    Single source of truth for the id-extraction rule so fingerprinting
    and the query loop can never disagree about which questions exist.

    Args:
        dataset: LongMemEval question entries.

    Returns:
        List of question id strings, one per entry, in dataset order.
    """
    return [
        str(entry.get("question_id", str(idx))) for idx, entry in enumerate(dataset)
    ]


def build_fingerprint(
    *,
    variant: str,
    label: str,
    reranker: bool,
    judge_model: str,
    git_commit: str | None,
    question_ids: list[str],
    limit: int | None,
) -> dict[str, Any]:
    """Build the run fingerprint that gates checkpoint resumption.

    A manifest is only resumed when every field matches the current run,
    so a resumed run answers exactly the same questions with the same
    dataset, model, and configuration. The fingerprint includes the
    retrieval-judge prompt hash, so old manifests holding answer-based
    verdicts (no hash key) never match — a fresh run auto-starts instead
    of resuming incomparable results.

    Args:
        variant: Dataset variant key.
        label: Run label (``"full"`` or ``"baseline"``).
        reranker: Whether the reranker is enabled for this run.
        judge_model: Judge model name.
        git_commit: Current git commit hash (may be ``None``).
        question_ids: Ordered question ids of the dataset subset.
        limit: ``--benchmark-limit`` value (``None`` for all questions).

    Returns:
        Fingerprint dict with keys ``variant``, ``label``, ``reranker``,
        ``judge_model``, ``git_commit``, ``dataset_sha256``,
        ``judge_prompt_sha``, and ``limit``.
    """
    return {
        "variant": variant,
        "label": label,
        "reranker": reranker,
        "judge_model": judge_model,
        "git_commit": git_commit,
        "dataset_sha256": hashlib.sha256(
            json.dumps(sorted(question_ids)).encode()
        ).hexdigest(),
        "judge_prompt_sha": hashlib.sha256(
            RETRIEVAL_JUDGE_SYSTEM_PROMPT.encode()
        ).hexdigest(),
        "limit": limit,
    }


def resolve_checkpoint(
    fingerprint: dict[str, Any],
    args: SimpleNamespace,
) -> Checkpoint:
    """Resolve the checkpoint manifest for a run, honouring resume flags.

    Resolution order:
        1. ``--fresh`` — always start a new manifest.
        2. ``--resume PATH`` — load exactly that manifest; missing file,
           corruption, fingerprint mismatch, or an already-completed
           manifest are all hard errors.
        3. Auto-resume — the newest incomplete manifest whose fingerprint
           matches every field; otherwise a new manifest.

    Args:
        fingerprint: Current run fingerprint (see ``build_fingerprint``).
        args: Parsed CLI args (``resume``, ``fresh``).

    Returns:
        A ``Checkpoint`` — loaded from a matching manifest or newly created.

    Raises:
        BenchmarkConfigError: If an explicit ``--resume`` manifest cannot
            be used, or a manifest in the auto-resume directory is
            unreadable.
    """
    label = str(fingerprint.get("label", "run"))

    if args.fresh:
        logger.info("--fresh set — starting a new checkpoint manifest")
        return Checkpoint.create_new(fingerprint, label=label)

    if args.resume:
        source = f"--resume manifest {args.resume}"
        return _load_verified_checkpoint(Path(args.resume), fingerprint, source)

    return _autodiscover_checkpoint(fingerprint, label)


def _load_verified_checkpoint(
    path: Path,
    fingerprint: dict[str, Any],
    source: str,
) -> Checkpoint:
    """Load a manifest and verify it fits the current run.

    Args:
        path: Manifest path to load.
        fingerprint: Current run fingerprint.
        source: Human-readable origin for error messages.

    Returns:
        The loaded, fingerprint-matching ``Checkpoint``.

    Raises:
        BenchmarkConfigError: If the file is missing, unreadable,
            corrupt, already completed, or its fingerprint differs.
    """
    if not path.is_file():
        raise BenchmarkConfigError(
            f"{source}: manifest not found at {path}. Drop --resume to "
            "auto-resume the newest matching manifest, or pass a valid "
            "path."
        )
    try:
        checkpoint = Checkpoint.load(path)
    except (OSError, ValueError) as exc:
        raise BenchmarkConfigError(
            f"{source}: cannot read manifest {path}: {exc}"
        ) from exc

    _require_fingerprint_match(checkpoint.fingerprint, fingerprint, source=source)
    if checkpoint.completed:
        raise BenchmarkConfigError(
            f"{source}: manifest {path} is already completed. Start a new "
            "run with --fresh."
        )
    logger.info("%s: resuming from %s", source, path)
    return checkpoint


def _autodiscover_checkpoint(
    fingerprint: dict[str, Any],
    label: str,
) -> Checkpoint:
    """Resume the newest incomplete manifest matching the current run.

    Args:
        fingerprint: Current run fingerprint.
        label: Run label for a newly created manifest.

    Returns:
        The matched ``Checkpoint``, or a brand-new one when nothing
        matches.

    Raises:
        BenchmarkConfigError: If a candidate manifest cannot be read.
    """
    if not IN_PROGRESS_DIR.is_dir():
        return Checkpoint.create_new(fingerprint, label=label)

    candidates: list[Path] = []
    for path in sorted(IN_PROGRESS_DIR.glob("run_*.json")):
        try:
            manifest: dict[str, Any] = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise BenchmarkConfigError(
                f"Cannot read checkpoint manifest {path}: {exc}. Fix or "
                "remove the file, or pass --fresh to start a new run."
            ) from exc
        if not isinstance(manifest, dict) or manifest.get("completed"):
            continue
        if manifest.get("fingerprint") == fingerprint:
            candidates.append(path)

    if not candidates:
        return Checkpoint.create_new(fingerprint, label=label)

    newest = max(candidates, key=lambda path: path.stat().st_mtime)
    return _load_verified_checkpoint(
        newest,
        fingerprint,
        source=f"auto-resumed manifest {newest}",
    )


def _require_fingerprint_match(
    stored: dict[str, Any],
    current: dict[str, Any],
    source: str,
) -> None:
    """Raise if a stored manifest fingerprint differs from the current run.

    Args:
        stored: Fingerprint recorded in the manifest.
        current: Fingerprint of the current run.
        source: Human-readable origin for the error message.

    Raises:
        BenchmarkConfigError: Listing every differing field.
    """
    differing = [
        f"{key}: manifest={stored.get(key)!r} current={current.get(key)!r}"
        for key in sorted(set(stored) | set(current))
        if stored.get(key) != current.get(key)
    ]
    if not differing:
        return
    raise BenchmarkConfigError(
        f"{source} does not match the current run configuration:\n  "
        + "\n  ".join(differing)
        + "\nStart a new run with --fresh, or pass a manifest that matches."
    )


async def main() -> None:
    """Run the benchmark end-to-end — entry point for ``python -m benchmarks``.

    Loads environment, validates credentials, builds the LLM backend and
    API client, resolves the full-run checkpoint manifest, and delegates
    to ``run_benchmark``.

    Raises:
        BenchmarkConfigError: On any missing or inconsistent configuration
            (mapped to exit code 2 by ``benchmarks/__main__.py``).
    """
    load_env()
    configure_logging()
    args = parse_args()
    if args.resume and args.baseline:
        raise BenchmarkConfigError(
            f"--resume {args.resume} names a full-run manifest, which can "
            "never match the baseline run (different label fingerprint). "
            "Drop --resume to auto-resolve the baseline manifest, or drop "
            "--baseline to resume the full run."
        )
    require_login_creds()
    llm_backend = build_llm_backend()
    api_client = make_api_client(args.base_url)

    # The fingerprint hashes the question ids, so the dataset is loaded
    # here — before the harness — to decide resume-vs-fresh.  Downloads
    # are cached under benchmarks/data/.  This is the single load: the
    # limit-sliced list is passed into run_benchmark, which must not
    # re-load (and re-parse 265 MB of) the dataset.
    dataset = load_dataset(args.variant)
    if args.benchmark_limit and args.benchmark_limit < len(dataset):
        dataset = dataset[: args.benchmark_limit]
    git_commit, _version = get_git_info()

    fingerprint = build_fingerprint(
        variant=args.variant,
        label="full",
        reranker=args.reranker,
        judge_model=llm_backend.model_name,
        git_commit=git_commit,
        question_ids=dataset_question_ids(dataset),
        limit=args.benchmark_limit,
    )
    checkpoint = resolve_checkpoint(fingerprint, args)
    logger.info("Full-run checkpoint manifest: %s", checkpoint.path)

    # Deferred import: run_longmemeval imports the factories above from
    # this module — a top-level import here would be circular.
    from benchmarks.run_longmemeval import run_benchmark

    await run_benchmark(args, llm_backend, api_client, checkpoint, dataset)
