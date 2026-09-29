"""Tracking-only LLM metering for ``llm_usage`` — drop costing, add scope.

Drops ``cost_estimate`` and adds the tracking-only metering columns:
``provider``, ``worker``, entity scopes (``project_id``, ``episode_id``,
``session_id``, ``community_id``, ``task_run_id``), ``reasoning_tokens``,
``embed_count``/``embed_dim``, and the ``idempotency_key`` dedup key.
Enables RLS with the standard org-isolation policy and adds the
``(organization_id, created_at)`` index for admin reads.

Existing rows are backfilled: ``provider``/``reasoning_tokens`` default
in place, ``worker`` is copied from ``task_type`` where NULL, and
``idempotency_key`` is minted per row before the NOT NULL
+ UNIQUE constraints are applied.

Revision ID: 0058
Revises: 0057
Create Date: 2026-09-28
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0058"
down_revision: str | None = "0057"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Drop costing, add metering scope, index, and RLS."""
    op.drop_column("llm_usage", "cost_estimate")

    op.add_column(
        "llm_usage",
        sa.Column("provider", sa.Text(), nullable=False, server_default=sa.text("''")),
    )
    op.add_column("llm_usage", sa.Column("worker", sa.Text(), nullable=True))
    for entity_col in (
        "project_id",
        "episode_id",
        "session_id",
        "community_id",
        "task_run_id",
    ):
        op.add_column("llm_usage", sa.Column(entity_col, sa.Uuid(), nullable=True))
    op.add_column(
        "llm_usage",
        sa.Column(
            "reasoning_tokens",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
    )
    op.add_column("llm_usage", sa.Column("embed_count", sa.Integer(), nullable=True))
    op.add_column("llm_usage", sa.Column("embed_dim", sa.Integer(), nullable=True))

    # Idempotency key: add nullable, mint per row, then constrain.
    op.add_column("llm_usage", sa.Column("idempotency_key", sa.Text(), nullable=True))
    op.execute(
        "UPDATE llm_usage SET idempotency_key = gen_random_uuid()::text "
        "WHERE idempotency_key IS NULL"
    )
    op.execute("UPDATE llm_usage SET worker = task_type WHERE worker IS NULL")
    op.alter_column("llm_usage", "idempotency_key", nullable=False)
    op.create_unique_constraint(
        "uq_llm_usage_idempotency_key", "llm_usage", ["idempotency_key"]
    )

    op.create_index(
        "ix_llm_usage_org_created",
        "llm_usage",
        ["organization_id", sa.text("created_at DESC")],
    )

    op.execute("ALTER TABLE llm_usage ENABLE ROW LEVEL SECURITY")
    op.execute("""
        CREATE POLICY org_isolation_llm_usage ON llm_usage
        FOR ALL
        USING (
            current_setting('app.bypass_rls', true) = 'true'
            OR organization_id = current_setting('app.org_id')::UUID
        )
    """)


def downgrade() -> None:
    """Remove RLS, index, metering columns; restore ``cost_estimate``.

    Rollback note: run ``alembic downgrade -1`` with the same
    ``OZ_DATABASE_URL`` used for the upgrade. Metering scope written
    since the upgrade is lost with the columns.
    """
    op.execute("DROP POLICY IF EXISTS org_isolation_llm_usage ON llm_usage")
    op.execute("ALTER TABLE llm_usage DISABLE ROW LEVEL SECURITY")
    op.drop_index("ix_llm_usage_org_created", table_name="llm_usage")
    op.drop_constraint("uq_llm_usage_idempotency_key", "llm_usage", type_="unique")
    op.drop_column("llm_usage", "idempotency_key")
    op.drop_column("llm_usage", "embed_dim")
    op.drop_column("llm_usage", "embed_count")
    op.drop_column("llm_usage", "reasoning_tokens")
    for entity_col in (
        "task_run_id",
        "community_id",
        "session_id",
        "episode_id",
        "project_id",
    ):
        op.drop_column("llm_usage", entity_col)
    op.drop_column("llm_usage", "worker")
    op.drop_column("llm_usage", "provider")
    op.add_column(
        "llm_usage",
        sa.Column(
            "cost_estimate",
            sa.Numeric(12, 8),
            server_default=sa.text("0"),
            nullable=False,
        ),
    )
