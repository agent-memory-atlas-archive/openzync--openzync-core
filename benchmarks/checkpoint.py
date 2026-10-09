"""Checkpoint manifests for resumable LongMemEval benchmark runs.

A manifest records the run's identity (fingerprint) and its per-question
results under ``benchmarks/results/.in_progress/``.  Manifests are written
atomically after every question, so an interrupted run loses at most the
question in flight and resumes instead of restarting.

Manifest schema (schema_version 1)::

    {
        "schema_version": 1,
        "label": "full",                  # or "baseline"
        "completed": false,
        "ingested": false,              # true once ingest + enrichment wait done
        "project_id": null,               # OpenZync project UUID, once known
        "fingerprint": {...},             # run identity — see cli.build_fingerprint
        "started_at": "2026-10-08T12:00:00+00:00",
        "updated_at": "2026-10-08T12:05:00+00:00",
        "results": [...]                  # per-question result dicts
    }
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_VERSION: int = 1
"""Current manifest schema version; older manifests are rejected on load."""

RESULTS_DIR: Path = Path(__file__).resolve().parent / "results"
"""Directory holding final result JSON files and in-progress manifests."""

IN_PROGRESS_DIR: Path = RESULTS_DIR / ".in_progress"
"""Directory holding checkpoint manifests for unfinished runs."""

_REQUIRED_KEYS: tuple[str, ...] = (
    "label",
    "fingerprint",
    "started_at",
    "updated_at",
    "results",
)
"""Keys every manifest must carry for ``load`` to accept it."""


def _utcnow_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


class Checkpoint:
    """A resumable benchmark run: identity fingerprint plus per-question results.

    The manifest is the single source of truth for what a run has already
    answered.  Mutating methods update ``updated_at``; ``set_project_id``
    and ``mark_completed`` persist immediately, while ``append_result``
    leaves persistence to an explicit ``save`` call (the harness saves
    after every question).

    Attributes:
        path: Manifest file path.
        label: Run label (``"full"`` or ``"baseline"``).
        completed: Whether the run finished and its results were saved.
        ingested: Whether ingest + the enrichment wait completed. A
            resumed manifest with a project id but ``ingested=False``
            was interrupted mid-ingest and must re-ingest.
        project_id: OpenZync project id used by the run, if known.
        fingerprint: Run identity dict — see ``cli.build_fingerprint``.
        started_at: ISO timestamp of run start (preserved across resumes).
        updated_at: ISO timestamp of the last manifest mutation.
        results: Per-question result dicts in completion order.
    """

    def __init__(
        self,
        path: Path,
        label: str,
        fingerprint: dict[str, Any],
        started_at: str,
        updated_at: str,
        results: list[dict[str, Any]],
        completed: bool = False,
        project_id: str | None = None,
        ingested: bool = False,
    ) -> None:
        """Initialize a checkpoint in memory.

        Prefer ``create_new`` and ``load`` — they apply the on-disk
        naming and validation rules.
        """
        self.path = path
        self.label = label
        self.completed = completed
        self.ingested = ingested
        self.project_id = project_id
        self.fingerprint = fingerprint
        self.started_at = started_at
        self.updated_at = updated_at
        self.results = results

    @classmethod
    def create_new(
        cls,
        fingerprint: dict[str, Any],
        label: str,
        project_id: str | None = None,
    ) -> Checkpoint:
        """Create a new checkpoint manifest on disk.

        Args:
            fingerprint: Run identity dict; must contain ``variant``.
            label: Run label (``"full"`` or ``"baseline"``) used in the
                manifest filename.
            project_id: OpenZync project id if already known.

        Returns:
            A new ``Checkpoint`` with an empty result list, already written
            to ``results/.in_progress/run_{variant}_{label}_{ts}.json``.
        """
        variant = str(fingerprint.get("variant", "unknown"))
        # Microseconds in the filename: two manifests created within the
        # same second must never collide.
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
        IN_PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint = cls(
            path=IN_PROGRESS_DIR / f"run_{variant}_{label}_{timestamp}.json",
            label=label,
            fingerprint=dict(fingerprint),
            started_at=_utcnow_iso(),
            updated_at=_utcnow_iso(),
            results=[],
            completed=False,
            project_id=project_id,
            ingested=False,
        )
        checkpoint.save()
        logger.info("Created checkpoint manifest %s", checkpoint.path)
        return checkpoint

    @classmethod
    def load(cls, path: Path) -> Checkpoint:
        """Load a checkpoint manifest from disk.

        Args:
            path: Manifest JSON path.

        Returns:
            The loaded ``Checkpoint``.

        Raises:
            ValueError: If the file is not valid JSON, has an unsupported
                ``schema_version``, or is missing required keys.
        """
        with open(path) as f:
            manifest: dict[str, Any] = json.load(f)
        return cls._from_manifest(path, manifest)

    @classmethod
    def _from_manifest(cls, path: Path, manifest: dict[str, Any]) -> Checkpoint:
        """Validate a parsed manifest dict and build a ``Checkpoint``.

        Args:
            path: Manifest path (used in error messages).
            manifest: Parsed manifest contents.

        Returns:
            The validated ``Checkpoint``.

        Raises:
            ValueError: If the manifest fails any schema check.
        """
        version = manifest.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"{path}: unsupported schema_version {version!r} "
                f"(expected {SCHEMA_VERSION})"
            )
        missing = [key for key in _REQUIRED_KEYS if key not in manifest]
        if missing:
            raise ValueError(f"{path}: manifest missing keys: {', '.join(missing)}")
        results = manifest["results"]
        if not isinstance(results, list):
            raise ValueError(
                f"{path}: 'results' must be a list, got {type(results).__name__}"
            )
        fingerprint = manifest["fingerprint"]
        if not isinstance(fingerprint, dict):
            raise ValueError(
                f"{path}: 'fingerprint' must be an object, "
                f"got {type(fingerprint).__name__}"
            )
        return cls(
            path=path,
            label=str(manifest["label"]),
            completed=bool(manifest.get("completed", False)),
            # ``ingested`` postdates some unreleased manifests — default
            # False (not ingested) so they still load, and the pipeline
            # re-runs ingest for them rather than trusting partial data.
            ingested=bool(manifest.get("ingested", False)),
            project_id=manifest.get("project_id"),
            fingerprint=fingerprint,
            started_at=str(manifest["started_at"]),
            updated_at=str(manifest["updated_at"]),
            results=results,
        )

    def save(self) -> None:
        """Write the manifest atomically (temp file + ``os.replace``).

        The harness calls this after every question, so a killed run
        loses at most the question in flight.
        """
        manifest = self._to_manifest()
        tmp_path = self.path.with_suffix(".json.tmp")
        with open(tmp_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        os.replace(tmp_path, self.path)

    def set_project_id(self, project_id: str) -> None:
        """Record the OpenZync project id and persist the manifest.

        Resumed runs read this back instead of re-listing projects.

        Args:
            project_id: Project UUID used for this run.
        """
        self.project_id = project_id
        self.updated_at = _utcnow_iso()
        self.save()
        logger.info("Checkpoint %s now pins project %s", self.path, project_id)

    def mark_ingested(self) -> None:
        """Record that ingest + the enrichment wait completed, persistently.

        The pipeline calls this right after ``_wait_for_enrichment``
        returns. Resumed runs with ``ingested=False`` re-run ingest into
        the pinned project instead of querying partial data.
        """
        self.ingested = True
        self.updated_at = _utcnow_iso()
        self.save()
        logger.info("Checkpoint %s marked ingested", self.path)

    def append_result(self, entry: dict[str, Any]) -> None:
        """Append a per-question result and refresh ``updated_at``.

        Persistence is an explicit ``save()`` call so the caller controls
        write frequency (the harness saves after every question).

        Args:
            entry: Per-question result dict; must contain an ``id`` key.

        Raises:
            ValueError: If ``entry`` has no ``id`` key.
        """
        if "id" not in entry:
            raise ValueError(f"result entry missing 'id' key: {sorted(entry)}")
        self.results.append(entry)
        self.updated_at = _utcnow_iso()

    def completed_ids(self) -> set[str]:
        """Return the ids of questions already recorded in this manifest.

        Returns:
            Set of question id strings (the ``id`` field of each stored
            result entry).
        """
        return {str(entry["id"]) for entry in self.results if "id" in entry}

    def mark_completed(self) -> None:
        """Mark the run complete and persist the manifest.

        The manifest stays in ``results/.in_progress/`` for audit; the
        human-readable results live in the timestamped files beside it.
        """
        self.completed = True
        self.updated_at = _utcnow_iso()
        self.save()
        logger.info("Checkpoint %s marked completed", self.path)

    def _to_manifest(self) -> dict[str, Any]:
        """Serialize the checkpoint to the manifest schema.

        Returns:
            A dict with exactly the schema_version 1 keys.
        """
        return {
            "schema_version": SCHEMA_VERSION,
            "label": self.label,
            "completed": self.completed,
            "ingested": self.ingested,
            "project_id": self.project_id,
            "fingerprint": self.fingerprint,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "results": self.results,
        }
