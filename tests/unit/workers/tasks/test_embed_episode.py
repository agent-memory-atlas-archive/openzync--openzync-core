"""Unit tests for embed_episode task."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

_EPISODE_ID = str(uuid4())
_ORG_ID = str(uuid4())
_PROJECT_ID = str(uuid4())
_CONTENT = "Test episode content for embedding."
_TRACE_ID = "trace-101"


@pytest.fixture
def embedder():
    """Patch the single embedder entrypoint; auto-undone after the test.

    Yields:
        The ``AsyncMock`` standing in for ``embed_passage``.
    """
    mock = AsyncMock(return_value=[[0.1] * 768])
    with patch("core.embeddings.embed_passage", mock):
        yield mock


@pytest.mark.unit
class TestEmbedEpisode:
    """embed_episode task tests."""

    def _make_db(self) -> AsyncMock:
        db = AsyncMock()
        db.__aenter__.return_value = db
        db.__aexit__.return_value = None
        return db

    def _factory(self, db: AsyncMock) -> MagicMock:
        f = MagicMock()
        f.return_value = db
        return f

    def _ctx(self, db: AsyncMock) -> dict:
        return {
            "db_engine": MagicMock(),
            "db_session_factory": self._factory(db),
        }

    @pytest.mark.asyncio
    async def test_success(self, embedder: AsyncMock) -> None:
        """Embedding generated and stored successfully."""
        with (
            patch("workers.tasks.base.with_retry", lambda **kw: lambda f: f),
            patch("repositories.episode_repository.EpisodeRepository") as mock_repo_cls,
        ):
            episode = MagicMock()
            episode.id = _EPISODE_ID
            episode.enrichment_status = 0

            mock_repo = AsyncMock()
            mock_repo.get_by_id.return_value = episode
            mock_repo_cls.return_value = mock_repo

            db = self._make_db()
            from workers.tasks.embed_episode import embed_episode

            await embed_episode(
                ctx=self._ctx(db),
                episode_id=_EPISODE_ID,
                org_id=_ORG_ID,
                project_id=_PROJECT_ID,
                content=_CONTENT,
                trace_id=_TRACE_ID,
            )

            embedder.assert_awaited_once()
            mock_repo.apply_enrichment_bits.assert_called_once()

    @pytest.mark.asyncio
    async def test_already_embedded(self) -> None:
        """Bit 1 already set → skip embedding."""
        with (
            patch("workers.tasks.base.with_retry", lambda **kw: lambda f: f),
            patch("repositories.episode_repository.EpisodeRepository") as mock_repo_cls,
        ):
            episode = MagicMock()
            episode.id = _EPISODE_ID
            episode.enrichment_status = 1 << 1

            mock_repo = AsyncMock()
            mock_repo.get_by_id.return_value = episode
            mock_repo_cls.return_value = mock_repo

            db = self._make_db()
            from workers.tasks.embed_episode import embed_episode

            await embed_episode(
                ctx=self._ctx(db),
                episode_id=_EPISODE_ID,
                org_id=_ORG_ID,
                project_id=_PROJECT_ID,
                content=_CONTENT,
            )

            mock_repo.apply_enrichment_bits.assert_not_called()

    @pytest.mark.asyncio
    async def test_episode_not_found(self) -> None:
        """Missing episode raises EpisodeNotFoundError."""
        with (
            patch("workers.tasks.base.with_retry", lambda **kw: lambda f: f),
            patch("repositories.episode_repository.EpisodeRepository") as mock_repo_cls,
        ):
            mock_repo = AsyncMock()
            mock_repo.get_by_id.return_value = None
            mock_repo_cls.return_value = mock_repo

            db = self._make_db()
            from workers.tasks.embed_episode import embed_episode

            with pytest.raises(Exception):
                await embed_episode(
                    ctx=self._ctx(db),
                    episode_id=_EPISODE_ID,
                    org_id=_ORG_ID,
                    project_id=_PROJECT_ID,
                    content=_CONTENT,
                )

    @pytest.mark.asyncio
    async def test_embedder_never_queried_for_org_config(
        self, embedder: AsyncMock
    ) -> None:
        """Embedding needs no org config — the ARQ ctx is not consulted for it.

        Guards the regression that re-introduced a per-org embedding
        backend: a stubbed org config would be silently ignored if some
        caller reached for it again.
        """
        with (
            patch("workers.tasks.base.with_retry", lambda **kw: lambda f: f),
            patch("core.org_config.get_org_config") as mock_get_cfg,
            patch("repositories.episode_repository.EpisodeRepository") as mock_repo_cls,
        ):
            episode = MagicMock()
            episode.id = _EPISODE_ID
            episode.enrichment_status = 0

            mock_repo = AsyncMock()
            mock_repo.get_by_id.return_value = episode
            mock_repo_cls.return_value = mock_repo

            db = self._make_db()
            from workers.tasks.embed_episode import embed_episode

            await embed_episode(
                ctx=self._ctx(db),
                episode_id=_EPISODE_ID,
                org_id=_ORG_ID,
                project_id=_PROJECT_ID,
                content=_CONTENT,
            )

            mock_get_cfg.assert_not_called()

    @pytest.mark.asyncio
    async def test_embedding_failure(self, embedder: AsyncMock) -> None:
        """Embedding API failure propagates."""
        with (
            patch("workers.tasks.base.with_retry", lambda **kw: lambda f: f),
            patch("repositories.episode_repository.EpisodeRepository") as mock_repo_cls,
        ):
            embedder.side_effect = Exception("ONNX inference error")

            episode = MagicMock()
            episode.id = _EPISODE_ID
            episode.enrichment_status = 0

            mock_repo = AsyncMock()
            mock_repo.get_by_id.return_value = episode
            mock_repo_cls.return_value = mock_repo

            db = self._make_db()
            from workers.tasks.embed_episode import embed_episode

            with pytest.raises(Exception, match="ONNX inference error"):
                await embed_episode(
                    ctx=self._ctx(db),
                    episode_id=_EPISODE_ID,
                    org_id=_ORG_ID,
                    project_id=_PROJECT_ID,
                    content=_CONTENT,
                )

    @pytest.mark.asyncio
    async def test_empty_content(self, embedder: AsyncMock) -> None:
        """Empty content generates embedding (still valid)."""
        with (
            patch("workers.tasks.base.with_retry", lambda **kw: lambda f: f),
            patch("repositories.episode_repository.EpisodeRepository") as mock_repo_cls,
        ):
            episode = MagicMock()
            episode.id = _EPISODE_ID
            episode.enrichment_status = 0

            mock_repo = AsyncMock()
            mock_repo.get_by_id.return_value = episode
            mock_repo_cls.return_value = mock_repo

            db = self._make_db()
            from workers.tasks.embed_episode import embed_episode

            await embed_episode(
                ctx=self._ctx(db),
                episode_id=_EPISODE_ID,
                org_id=_ORG_ID,
                project_id=_PROJECT_ID,
                content="",
            )

            embedder.assert_awaited_once()
            mock_repo.apply_enrichment_bits.assert_called_once()

    @pytest.mark.asyncio
    async def test_db_content_fetch(self, embedder: AsyncMock) -> None:
        """When content is None, fetch from DB."""
        with (
            patch("workers.tasks.base.with_retry", lambda **kw: lambda f: f),
            patch("repositories.episode_repository.EpisodeRepository") as mock_repo_cls,
        ):
            episode = MagicMock()
            episode.id = _EPISODE_ID
            episode.enrichment_status = 0
            episode.content = _CONTENT

            mock_repo = AsyncMock()
            mock_repo.get_by_id.return_value = episode
            mock_repo_cls.return_value = mock_repo

            db = self._make_db()
            from workers.tasks.embed_episode import embed_episode

            await embed_episode(
                ctx=self._ctx(db),
                episode_id=_EPISODE_ID,
                org_id=_ORG_ID,
                project_id=_PROJECT_ID,
                content=None,
            )

            embedder.assert_awaited_once()
