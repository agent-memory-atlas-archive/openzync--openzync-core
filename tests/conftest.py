"""Root test configuration — shared fixtures for all test levels.

Fixtures requiring the application stack (``app``, ``async_client``, etc.)
live in ``tests/integration/conftest.py`` to avoid import-time failures when
the application hasn't been built yet (unit tests).

Testcontainers helpers live here so they can be shared between
``tests/integration/conftest.py`` and ``tests/security/conftest.py``.
"""

from __future__ import annotations

import os

import pytest

# ═══════════════════════════════════════════════════════════════════════════════
# Testcontainers helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _ensure_testcontainers_env() -> None:
    """Set environment variables required by testcontainers.

    Disables Ryuk (resource reaper) in CI since the container runtime
    may not support it.  Also disables Docker host checks.
    """
    os.environ.setdefault("TESTCONTAINERS_RYUK_DISABLED", "true")
    os.environ.setdefault("TC_HOST", "localhost")


def _start_postgres_container() -> object:
    """Start a PostgreSQL 15 + pgvector testcontainer.

    Returns:
        The started container instance.  Connection URL is available via
        ``container.get_connection_url()``.
    """
    from testcontainers.postgres import PostgresContainer

    container = PostgresContainer(
        image="pgvector/pgvector:pg15",
        username="openzync",
        password="openzync",
        dbname="openzync_test",
        driver="asyncpg",
    )
    container.start()
    return container


def _start_redis_container() -> object:
    """Start a Redis 7 testcontainer.

    Returns:
        The started container instance.  Host and port are available
        via ``container.get_container_host_ip()`` and
        ``container.get_exposed_port(6379)``.
    """
    from testcontainers.redis import RedisContainer

    container = RedisContainer(image="redis:7-alpine")
    container.start()
    return container


def _start_falkordb_container() -> object:
    """Start a FalkorDB testcontainer.

    FalkorDB speaks the Redis protocol on 6379 — same image tag and
    port mapping as the dev profile in
    ``infra/docker-compose.backend.yml`` (host 6381 → container 6379).
    Host and port are available via
    ``container.get_container_host_ip()`` and
    ``container.get_exposed_port(6379)``; the connection URL is
    ``redis://{host}:{port}``.

    Returns:
        The started container instance.
    """
    from testcontainers.redis import RedisContainer

    container = RedisContainer(image="falkordb/falkordb:v4.20.1-alpine")
    container.start()
    return container


def sync_database_url(url: str) -> str:
    """Force the psycopg2 dialect on a sync database URL.

    SQLAlchemy 2.1 changed the default ``postgresql://`` dialect from
    psycopg2 to psycopg (v3). ``psycopg2-binary`` is the declared dev
    dependency, so the driver must be pinned explicitly instead of
    inherited from whatever the installed SQLAlchemy defaults to.

    Accepts either scheme: a ``postgresql+asyncpg://`` URL (what
    ``testcontainers`` hands out) or a bare ``postgresql://`` one.
    Already-pinned URLs are returned unchanged.
    """
    return url.replace("postgresql+asyncpg://", "postgresql+psycopg2://").replace(
        "postgresql://", "postgresql+psycopg2://"
    )


def _run_alembic_upgrade(driver_url: str) -> None:
    """Run Alembic migrations up to ``head`` against the given database.

    Args:
        driver_url: Full asyncpg connection URL for the database.
    """

    from alembic.command import upgrade as alembic_upgrade
    from alembic.config import Config as AlembicConfig
    from sqlalchemy import create_engine

    # Alembic needs a sync engine for its migration runner
    sync_url = sync_database_url(driver_url)
    sync_engine = create_engine(sync_url, pool_pre_ping=True)

    try:
        alembic_cfg = AlembicConfig("alembic.ini")
        alembic_cfg.attributes["connection"] = sync_engine.connect()
        alembic_upgrade(alembic_cfg, "head")
    finally:
        sync_engine.dispose()


# ═══════════════════════════════════════════════════════════════════════════════
# Auth helpers
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def test_api_key() -> str:
    """Return a synthetic API key for use in auth tests."""
    return "oz_test_" + "a" * 64
