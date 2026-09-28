"""Shared fixtures for worker-task unit tests.

The episode workers (compute_observations, embed_episode, enrich_episode,
extract_blob_text) each open a fail-closed archived-project guard —
``ProjectRepository(db).is_archived(...)`` — *before* any real work.  Against
a mocked session the real query returns a truthy mock, so every task
early-returns "project_archived_skipping" and no assertion past the guard is
ever reached.

Default this directory's projects to **not archived**.  A test that needs the
archived path patches ``ProjectRepository`` again inside its own ``with``
block, which takes precedence over this fixture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _project_not_archived() -> Iterator[None]:
    """Make the archived-project guard non-archived for every task test."""
    with patch("repositories.project_repository.ProjectRepository") as project_cls:
        project_cls.return_value.is_archived = AsyncMock(return_value=False)
        yield
