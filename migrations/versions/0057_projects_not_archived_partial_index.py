"""Add a partial index on ``projects (id) WHERE is_archived = false``.

The four episode workers (enrich/embed/compute_observations/extract_blob_text)
early-return on archived projects, so archived-project episodes never reach
``enrichment_status = ENRICHMENT_ALL``.  Every read-side **progress** aggregate
therefore excludes archived projects via
``col.project_id.not_in(archived_project_ids())``, which resolves this
subquery once per query.

``projects.is_archived`` had no index at all before this migration, so that
subquery was a sequential scan of the projects table.  The index is partial on
``is_archived = false`` and covers only ``id`` — the archived set is tiny by
construction, and the index stays near-zero cost when nothing is archived.

Deliberately NOT expressed as ``index=True`` on the model column: a partial
index cannot be declared that way, and adding one would create a duplicate.

Not ``CONCURRENTLY``: ``projects`` holds one row per project (tiny table), so
the ``ACCESS EXCLUSIVE`` lock window is instantaneous.

Revision ID: 0057
Revises: 0056
Create Date: 2026-09-28
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0057"
down_revision: str | None = "0056"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

INDEX_NAME = "ix_projects_not_archived"


def upgrade() -> None:
    """Create the partial non-archived-projects index."""
    op.execute(
        f"CREATE INDEX IF NOT EXISTS {INDEX_NAME} "
        "ON projects (id) WHERE is_archived = false"
    )


def downgrade() -> None:
    """Drop the partial non-archived-projects index."""
    op.execute(f"DROP INDEX IF EXISTS {INDEX_NAME}")
