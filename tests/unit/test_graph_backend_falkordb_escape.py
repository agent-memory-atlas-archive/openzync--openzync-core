"""Unit tests for FalkorDB RediSearch query escaping (hostile names).

Covers ``_escape_redisearch_query`` and its use as the ``$query`` param in
``search_entities`` / ``bulk_search_entities``. All FalkorDB I/O is mocked.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

pytest.importorskip("falkordb")

from packages.graph_backend.falkordb import (
    FalkorGraphBackend,
    _escape_redisearch_query,
)

ORG_ID = UUID("00000000-0000-0000-0000-000000000001")
PROJ_ID = UUID("00000000-0000-0000-0000-000000000002")


class MockQueryResult:
    """Simulates ``falkordb.asyncio.AsyncGraph.query()`` return value."""

    def __init__(self, result_set: list[tuple[object, ...]]) -> None:
        self.result_set = result_set


@pytest.fixture
def mock_graph() -> AsyncMock:
    """Mocked ``AsyncGraph`` with a pre-assigned ``query`` AsyncMock."""
    graph_mock = AsyncMock()
    graph_mock.query = AsyncMock(return_value=MockQueryResult([]))
    return graph_mock


@pytest.fixture
def backend(mock_graph: AsyncMock) -> FalkorGraphBackend:
    """Backend whose ``_get_graph`` returns the mocked graph."""
    client = MagicMock()
    client.select_graph.return_value = mock_graph
    bk = FalkorGraphBackend(client=client)
    bk._get_graph = MagicMock(return_value=mock_graph)  # type: ignore[method-assign]
    return bk


@pytest.mark.unit
class TestEscapeRediSearchQuery:
    """Direct unit tests for the escape helper."""

    @staticmethod
    def test_percent_escaped() -> None:
        assert _escape_redisearch_query("94%") == "94\\%"

    @staticmethod
    def test_plus_escaped() -> None:
        assert _escape_redisearch_query("C++") == "C\\+\\+"

    @staticmethod
    def test_colon_escaped() -> None:
        assert _escape_redisearch_query("a:b") == "a\\:b"

    @staticmethod
    def test_quotes_escaped() -> None:
        assert _escape_redisearch_query('"quoted"') == '\\"quoted\\"'

    @staticmethod
    def test_preexisting_backslash_escaped_once() -> None:
        assert _escape_redisearch_query("a\\b") == "a\\\\b"

    @staticmethod
    def test_empty_string() -> None:
        assert _escape_redisearch_query("") == ""

    @staticmethod
    def test_plain_names_unchanged() -> None:
        assert _escape_redisearch_query("Alice") == "Alice"

    @staticmethod
    def test_spaces_preserved_as_separators() -> None:
        assert _escape_redisearch_query("foo bar") == "foo bar"


@pytest.mark.unit
class TestEscapedQueryParam:
    """Callers pass the escaped (not raw) string as ``$query``."""

    @staticmethod
    async def test_search_entities_escapes_query(
        backend: FalkorGraphBackend,
        mock_graph: AsyncMock,
    ) -> None:
        raw = "94% C++ a:b"
        await backend.search_entities(ORG_ID, PROJ_ID, raw)
        params = mock_graph.query.call_args[0][1]
        assert params["query"] == _escape_redisearch_query(raw)
        assert params["query"] != raw

    @staticmethod
    async def test_bulk_search_entities_escapes_query(
        backend: FalkorGraphBackend,
        mock_graph: AsyncMock,
    ) -> None:
        raw = "94% C++ a:b"
        await backend.bulk_search_entities(ORG_ID, PROJ_ID, raw)
        params = mock_graph.query.call_args[0][1]
        assert params["query"] == _escape_redisearch_query(raw)
        assert params["query"] != raw
