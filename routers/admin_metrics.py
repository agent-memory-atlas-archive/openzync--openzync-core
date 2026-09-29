"""Admin metrics endpoints — HTTP adapter layer only.

Provides aggregated metrics for the admin panel frontend, combining
DB-sourced counts with Prometheus-backed latency/error metrics.

Endpoints:
    GET /metrics/summary   — Aggregated RED + DB metrics for the admin panel
    GET /metrics/targets   — List Prometheus scrape targets and health
    GET /metrics/batch     — Run all 12 predefined queries at once (partial OK)

    ``GET /metrics/batch`` takes EITHER ``days`` (one of 7/30/90, window is
    ``[now-days, now)``) OR ``from``+``to`` (``YYYY-MM-DD``, window is
    ``[from 00:00 UTC, to 00:00 UTC + 1 day)``). The styles are mutually
    exclusive — mixing them, omitting both, or ``from >= to`` is a 422.
    Every handler filters on the explicit ``[start, end)`` window.

The single-query surface (``GET /metrics/queries``, ``GET /metrics/query``)
was removed — the dashboard renders every predefined query in one batched
call instead of fanning out N single-query requests.

All endpoints require API key or JWT authentication.
"""

from __future__ import annotations

import asyncio
import math
import time
from datetime import UTC, date, datetime, timedelta
from datetime import time as dtime
from typing import Any
from uuid import UUID

import httpx
import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.exceptions import GraphBackendUnavailableError, MetricsUnavailableError
from dependencies.auth import require_permission
from dependencies.db import get_db
from dependencies.org_config import get_org_config
from models.audit_log import AuditLog
from models.episode import Episode
from models.fact import Fact
from models.project import Project
from models.user import User
from packages.graph_backend.interface import GraphBackend
from repositories.project_repository import archived_project_ids
from schemas.admin_metrics import (
    EpisodeStats,
    GraphStats,
    MetricsSummaryResponse,
)
from schemas.organization_config import OrgConfigBase
from schemas.sorting import MonitorTargetSortBy, SortDir
from services.graph_stats_service import GraphStatsService
from services.metrics_service import MetricsService
from workers.tasks.base import ENRICHMENT_ALL

logger = structlog.get_logger(__name__)

router = APIRouter(
    prefix="/metrics",
    tags=["Admin - Metrics"],
)


async def _resolve_graph_backend(
    request: Request,
    db: AsyncSession,
    org_config: OrgConfigBase,
    org_id: UUID,
) -> GraphBackend | None:
    """Resolve the org-configured graph backend, fail-soft to ``None``.

    Mirrors the ``context``/``search`` router pattern (dispatcher from app
    state, SurrealDB pool only when configured).  Unlike worker paths, admin
    observability degrades to zeros instead of failing — a down graph must
    not take down the whole metrics summary.  Every degraded path logs.
    """
    dispatcher = getattr(request.app.state, "graph_backend_dispatcher", None)
    if dispatcher is None:
        logger.warning("admin_metrics.no_graph_dispatcher")
        return None
    surreal = None
    if org_config.graph_backend == "surrealdb":
        pool = getattr(request.app.state, "surreal_connection_pool", None)
        if pool is not None:
            try:
                surreal = await pool.get_or_create(org_id, org_config)
            except Exception as exc:
                logger.warning(
                    "admin_metrics.surreal_connection_failed",
                    error=str(exc),
                )
                return None
    try:
        return dispatcher.resolve_and_create(
            org_config,
            db,
            surreal=surreal,
            falkordb_client=getattr(request.app.state, "falkordb_client", None),
        )
    except (GraphBackendUnavailableError, ValueError) as exc:
        logger.warning("admin_metrics.graph_backend_unresolved", error=str(exc))
        return None
    except Exception as exc:
        # GoneError (postgres) included — PG graph backend is retired.
        logger.warning("admin_metrics.graph_backend_failed", error=str(exc))
        return None


# ── Dependency ────────────────────────────────────────────────────────────────


def _get_metrics_service() -> MetricsService:
    """Dependency factory for ``MetricsService``."""
    return MetricsService(prometheus_url=get_settings().PROMETHEUS_URL)


# ── Shared Prometheus client ──────────────────────────────────────────────────
# ⚠️ One module-level client reused by every Prom helper/targets call.
# Per-call ``async with httpx.AsyncClient()`` opens a new connection pool per
# query — under /metrics/batch (12 handlers, 4 hitting Prom) that is 4 pools
# per request. Per-request timeouts still override the default below.

_prom_client: httpx.AsyncClient | None = None


def _get_prom_client() -> httpx.AsyncClient:
    """Return the shared Prometheus HTTP client (lazy singleton)."""
    global _prom_client
    if _prom_client is None:
        _prom_client = httpx.AsyncClient(timeout=10.0)
    return _prom_client


async def close_prom_client() -> None:
    """Close the shared Prometheus HTTP client (lifespan shutdown).

    Safe to call when the client was never created — a no-op in that case.
    """
    global _prom_client
    if _prom_client is not None:
        await _prom_client.aclose()
        _prom_client = None


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get(
    "/summary",
    response_model=MetricsSummaryResponse,
    summary="Aggregated admin dashboard metrics",
    description=(
        "Returns a combined view of DB counts (episodes, users, graphs) and "
        "Prometheus-backed performance metrics (latency, error rate, request "
        "rate).  The ``status`` field is ``\"degraded\"`` if Prometheus is "
        "unreachable — DB counts are still returned."
    ),
)
async def get_metrics_summary(
    request: Request,
    days: int | None = Query(
        default=None,
        description="Look-back window — one of 7/30/90. "
        "Mutually exclusive with from/to. "
        "Omit all window params for the last 24h.",
    ),
    from_date: str | None = Query(
        default=None,
        alias="from",
        description="Start date YYYY-MM-DD inclusive (00:00 UTC). "
        "Requires to; mutually exclusive with days.",
    ),
    to_date: str | None = Query(
        default=None,
        alias="to",
        description="End date YYYY-MM-DD inclusive "
        "(window ends to 00:00 UTC + 1 day). "
        "Requires from; mutually exclusive with days.",
    ),
    db: AsyncSession = Depends(get_db),
    org_id: str = Depends(require_permission("members:read")),
    org_config: OrgConfigBase = Depends(get_org_config),
    prom: MetricsService = Depends(_get_metrics_service),
) -> MetricsSummaryResponse:
    """Get aggregated metrics for the admin dashboard.

    Merges DB counts and Prometheus metrics into a single response.
    DB counts are scoped to the authenticated organization.
    Only the Prometheus range time-series honor the window; DB totals
    and instant queries stay snapshot semantics.
    """
    if days is not None and (from_date is not None or to_date is not None):
        raise HTTPException(
            status_code=422,
            detail="Pass either days or from/to, not both.",
        )
    if days is not None:
        if days not in (7, 30, 90):
            raise HTTPException(
                status_code=422,
                detail="days must be one of 7, 30, 90.",
            )
        end = datetime.now(UTC)
        start = end - timedelta(days=days)
    elif from_date is not None or to_date is not None:
        if from_date is None or to_date is None:
            raise HTTPException(
                status_code=422,
                detail="Pass either days or both from and to.",
            )
        try:
            from_day = date.fromisoformat(from_date)
            to_day = date.fromisoformat(to_date)
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail="from/to must be YYYY-MM-DD.",
            ) from exc
        if from_day >= to_day:
            raise HTTPException(
                status_code=422,
                detail="from must be earlier than to.",
            )
        start = datetime.combine(from_day, dtime.min, tzinfo=UTC)
        end = datetime.combine(to_day, dtime.min, tzinfo=UTC) + timedelta(days=1)
    else:
        end = datetime.now(UTC)
        start = end - timedelta(hours=24)
    org_uuid = UUID(org_id)
    backend = await _resolve_graph_backend(request, db, org_config, org_uuid)

    # ── DB counts (run concurrently) ─────────────────────────────────────
    episode_stats, graph_stats, user_count = await _fetch_db_counts(
        db, org_uuid, backend
    )

    # ── Prometheus metrics (org-scoped) ──────────────────────────────────
    # Degraded mode: a down Prometheus must not take down the summary —
    # DB counts are still returned with Prometheus fields at defaults.
    # Only MetricsUnavailableError is caught; every other exception
    # propagates to the global handler.
    try:
        perf = await prom.get_summary(org_id=str(org_uuid), start=start, end=end)
    except MetricsUnavailableError as exc:
        logger.warning(
            "admin_metrics.prometheus_degraded",
            org_id=str(org_uuid),
            error=str(exc),
        )
        return MetricsSummaryResponse(
            episodes=episode_stats,
            graphs=graph_stats,
            users_total=user_count,
            status="degraded",
            message=f"Prometheus unreachable: {exc}",
        )

    # Overwrite DB fields into the response
    perf.episodes = episode_stats
    perf.graphs = graph_stats
    perf.users_total = user_count

    return perf


@router.get(
    "/targets",
    summary="Prometheus scrape targets",
    description=(
        "Lists all Prometheus scrape targets and their current health. "
        "Useful for the admin panel's health indicator.  Returns 502 if "
        "Prometheus is unreachable."
    ),
)
async def get_prometheus_targets(
    _org_id: str = Depends(require_permission("members:read")),
    sort_by: MonitorTargetSortBy | None = Query(
        default=None,
        description="Sort key — ``name`` (job), ``created_at`` "
        "(last_scrape), ``type`` (instance), ``status`` (health).",
    ),
    sort_dir: SortDir = Query(
        default="asc",
        description="Sort direction (default asc).",
    ),
) -> dict:
    """Get Prometheus scrape target health.

    Default order is the Prometheus response order (preserved); sort keys
    map to target fields (``name``→job, ``created_at``→last_scrape,
    ``type``→instance, ``status``→health).

    Returns:
        Dict with ``targets`` list and ``status``.
    """
    base_url = get_settings().PROMETHEUS_URL.rstrip("/")
    try:
        resp = await _get_prom_client().get(
            f"{base_url}/api/v1/targets", timeout=3.0
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Prometheus targets unavailable: {exc}",
        ) from exc

    targets = []
    for t in data.get("data", {}).get("activeTargets", []):
        targets.append({
            "job": t.get("labels", {}).get("job", ""),
            "instance": t.get("labels", {}).get("instance", ""),
            "health": t.get("health", "unknown"),
            "last_scrape": t.get("lastScrape", ""),
            "last_error": t.get("lastError", "") or None,
        })

    if sort_by is not None:
        # note: explicit key selection — no getattr on raw input.
        def _target_key(target: dict) -> tuple[str, str]:
            if sort_by == "name":
                return (target["job"], target["instance"])
            if sort_by == "created_at":
                return (target["last_scrape"], target["job"])
            if sort_by == "type":
                return (target["instance"], target["job"])
            return (target["health"], target["job"])

        targets.sort(key=_target_key, reverse=(sort_dir == "desc"))

    return {"status": "ok", "targets": targets}


# ── Predefined query helpers ──────────────────────────────────────────────

ENRICHMENT_STATUS_LABELS: dict[int, str] = {
    0: "pending",
    1: "fact_extraction",
    3: "fact_extraction + classification",
    7: "fact_extraction + classification + summarization",
    15: "+ entity_links",
    31: "+ embedding",
    63: "fully_enriched",
}


def _result(
    query_name: str,
    org_scoped: bool,
    columns: list[str],
    rows: list,
    params: dict,
    warning: str | None = None,
) -> dict:
    """Build the standard query response dict."""
    resp: dict = {
        "query": query_name,
        "org_scoped": org_scoped,
        "columns": columns,
        "rows": rows,
        "total": len(rows),
        "parameters": params,
    }
    if warning:
        resp["warning"] = warning
    return resp


async def _prom_instant(promql: str) -> float:
    """Run a PromQL instant query and return the scalar value."""
    base_url = get_settings().PROMETHEUS_URL.rstrip("/")
    resp = await _get_prom_client().get(
        f"{base_url}/api/v1/query", params={"query": promql}, timeout=5.0
    )
    resp.raise_for_status()
    data = resp.json()
    if data["status"] != "success":
        raise HTTPException(
            status_code=502,
            detail=f"Prometheus error: {data.get('error', '')}",
        )
    results = data["data"]["result"]
    if not results:
        return 0.0
    value = float(results[0]["value"][1])
    # histogram_quantile yields "NaN" on sparse/fresh buckets — not JSON compliant.
    return value if math.isfinite(value) else 0.0


def _prom_step(start: datetime, end: datetime) -> str:
    """Range step capped at ~180 points: ``max(1h, window/180)``."""
    window_hours = (end - start).total_seconds() / 3600
    hours = max(1, math.ceil(window_hours / 180))
    return f"{hours}h"


async def _prom_range(promql: str, start: datetime, end: datetime) -> list[list]:
    """Run a PromQL range query and return rows as [[timestamp, value]]."""
    base_url = get_settings().PROMETHEUS_URL.rstrip("/")
    resp = await _get_prom_client().get(
        f"{base_url}/api/v1/query_range",
        params={
            "query": promql,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "step": _prom_step(start, end),
        },
        timeout=10.0,
    )
    resp.raise_for_status()
    data = resp.json()
    if data["status"] != "success":
        raise HTTPException(
            status_code=502,
            detail=f"Prometheus error: {data.get('error', '')}",
        )
    results = data["data"]["result"]
    if not results:
        return []
    rows: list[list] = []
    for v in results[0].get("values", []):
        point = float(v[1])
        # Prometheus returns Unix-epoch floats; frontend parses with
        # new Date(...), which needs ISO-8601 strings.
        ts = datetime.fromtimestamp(float(v[0]), tz=UTC).isoformat()
        rows.append([ts, point if math.isfinite(point) else 0.0])
    return rows


# ── DB query handlers ─────────────────────────────────────────────────────


async def _episodes_per_day_core(
    db: AsyncSession,
    org_uuid: UUID,
    start: datetime,
    end: datetime,
    project_id: UUID | None,
    query_name: str,
) -> dict:
    """Shared episodes/messages-per-day core — one query, two aliases.

    Episodes are message turns, so ``messages_per_day`` is the same query
    shape with a different ``query`` label (kept as a separate key so the
    dashboard can chart both without client-side renaming).

    Window is the explicit ``[start, end)`` — ORM comparisons bind
    ``start``/``end`` as parameters, so the window never interpolates
    request input into SQL text.
    """
    conditions = [
        Episode.organization_id == org_uuid,
        Episode.is_deleted.is_(False),
        Episode.created_at >= start,
        Episode.created_at < end,
    ]
    if project_id:
        conditions.append(Episode.project_id == project_id)
    stmt = (
        select(
            func.date_trunc("day", Episode.created_at.op("AT TIME ZONE")("UTC")).label(
                "date"
            ),
            func.count(Episode.id).label("count"),
        )
        .select_from(Episode)
        .where(*conditions)
        .group_by(text("date"))
        .order_by(text("date DESC"))
    )
    result = await db.execute(stmt)
    rows = [[str(r.date), r.count] for r in result]
    params = {"from": start.isoformat(), "to": end.isoformat()}
    return _result(query_name, True, ["date", "count"], rows, params)


async def _episodes_per_day(
    db: AsyncSession,
    org_uuid: UUID,
    start: datetime,
    end: datetime,
    limit: int,
    project_id: UUID | None,
) -> dict:
    return await _episodes_per_day_core(
        db, org_uuid, start, end, project_id, "episodes_per_day"
    )


async def _messages_per_day(
    db: AsyncSession,
    org_uuid: UUID,
    start: datetime,
    end: datetime,
    limit: int,
    project_id: UUID | None,
) -> dict:
    return await _episodes_per_day_core(
        db, org_uuid, start, end, project_id, "messages_per_day"
    )


async def _activity_per_day(
    db: AsyncSession,
    org_uuid: UUID,
    start: datetime,
    end: datetime,
    limit: int,
    project_id: UUID | None,
) -> dict:
    # AuditLog has no project column — project_id is explicitly ignored.
    _ = project_id
    stmt = (
        select(
            func.date_trunc("day", AuditLog.created_at.op("AT TIME ZONE")("UTC")).label(
                "date"
            ),
            func.count(AuditLog.id).label("count"),
        )
        .select_from(AuditLog)
        .where(
            AuditLog.organization_id == org_uuid,
            AuditLog.created_at >= start,
            AuditLog.created_at < end,
        )
        .group_by(text("date"))
        .order_by(text("date DESC"))
    )
    result = await db.execute(stmt)
    rows = [[str(r.date), r.count] for r in result]
    params = {"from": start.isoformat(), "to": end.isoformat()}
    return _result("activity_per_day", True, ["date", "count"], rows, params)


async def _entities_per_day(
    db: AsyncSession,
    org_uuid: UUID,
    start: datetime,
    end: datetime,
    limit: int,
    project_id: UUID | None,
    backend: GraphBackend | None = None,
) -> dict:
    stats = GraphStatsService(db, backend)
    # Inventory: archived projects keep their entities, so they stay in scope.
    project_ids = await stats.resolve_project_ids(
        org_uuid, project_id, include_archived=True
    )
    per_day = await stats.entity_counts_per_day(
        org_uuid, project_ids, start, end
    )
    # Row shape preserved: [[<midnight-UTC datetime str>, count]], newest first
    # (matches the old date_trunc PG output format).
    rows = [
        [
            str(datetime.combine(date.fromisoformat(day), dtime.min, tzinfo=UTC)),
            count,
        ]
        for day, count in sorted(per_day.items(), reverse=True)
    ]
    params = {"from": start.isoformat(), "to": end.isoformat()}
    return _result("entities_per_day", True, ["date", "count"], rows, params)


async def _facts_per_day(
    db: AsyncSession,
    org_uuid: UUID,
    start: datetime,
    end: datetime,
    limit: int,
    project_id: UUID | None,
) -> dict:
    conditions = [
        Fact.organization_id == org_uuid,
        Fact.created_at >= start,
        Fact.created_at < end,
    ]
    if project_id:
        conditions.append(Fact.project_id == project_id)
    stmt = (
        select(
            func.date_trunc("day", Fact.created_at.op("AT TIME ZONE")("UTC")).label(
                "date"
            ),
            func.count(Fact.id).label("count"),
        )
        .select_from(Fact)
        .where(*conditions)
        .group_by(text("date"))
        .order_by(text("date DESC"))
    )
    result = await db.execute(stmt)
    rows = [[str(r.date), r.count] for r in result]
    params = {"from": start.isoformat(), "to": end.isoformat()}
    return _result("facts_per_day", True, ["date", "count"], rows, params)


async def _enrichment_progress(
    db: AsyncSession,
    org_uuid: UUID,
    start: datetime,
    end: datetime,
    limit: int,
    project_id: UUID | None,
) -> dict:
    # Snapshot metric — the [start, end) window is explicitly ignored.
    _ = (start, end)
    # Progress metric — archived projects are excluded because the episode
    # workers early-return on them, so those rows are terminal, not pending.
    conditions = [
        Episode.organization_id == org_uuid,
        Episode.is_deleted.is_(False),
        Episode.project_id.not_in(archived_project_ids()),
    ]
    if project_id:
        conditions.append(Episode.project_id == project_id)
    stmt = (
        select(
            Episode.enrichment_status,
            func.count(Episode.id).label("count"),
        )
        .select_from(Episode)
        .where(*conditions)
        .group_by(Episode.enrichment_status)
        .order_by(text("count DESC"))
    )
    result = await db.execute(stmt)
    rows = [[r.enrichment_status, r.count] for r in result]
    labels = {
        str(k): v
        for k, v in ENRICHMENT_STATUS_LABELS.items()
        if any(row[0] == k for row in rows)
    }
    resp = _result(
        "enrichment_progress", True, ["enrichment_status", "count"], rows, {}
    )
    resp["labels"] = labels
    return resp


async def _top_projects_by_episodes(
    db: AsyncSession,
    org_uuid: UUID,
    start: datetime,
    end: datetime,
    limit: int,
    project_id: UUID | None,
    include_archived: bool = False,
) -> dict:
    # Snapshot ranking — the [start, end) window is explicitly ignored.
    _ = (start, end)
    # Volume metric — archiving preserves episodes, so archived projects are
    # excluded by default (like every other ranking here) but each row still
    # carries the flag, and ``include_archived`` lets the dashboard show them.
    conditions = [
        Episode.organization_id == org_uuid,
        Episode.is_deleted.is_(False),
    ]
    if project_id:
        conditions.append(Episode.project_id == project_id)
    if not include_archived:
        conditions.append(Project.is_archived.is_(False))
    stmt = (
        select(
            Project.name.label("project_name"),
            func.count(Episode.id).label("episode_count"),
            Project.is_archived.label("is_archived"),
        )
        .select_from(Episode)
        .join(Project, Project.id == Episode.project_id)
        .where(*conditions)
        .group_by(Project.id, Project.name, Project.is_archived)
        .order_by(text("episode_count DESC"))
        .limit(limit)
    )
    result = await db.execute(stmt)
    rows = [[r.project_name, r.episode_count, r.is_archived] for r in result]
    return _result(
        "top_projects_by_episodes",
        True,
        ["project_name", "episode_count", "is_archived"],
        rows,
        {"limit": limit, "include_archived": include_archived},
    )


async def _top_users_by_messages(
    db: AsyncSession,
    org_uuid: UUID,
    start: datetime,
    end: datetime,
    limit: int,
    project_id: UUID | None,
) -> dict:
    # Snapshot ranking — the [start, end) window is explicitly ignored.
    _ = (start, end)
    conditions = [
        Episode.organization_id == org_uuid,
        Episode.is_deleted.is_(False),
    ]
    if project_id:
        conditions.append(Episode.project_id == project_id)
    stmt = (
        select(
            func.coalesce(
                func.nullif(User.name, ""),
                func.nullif(User.email, ""),
            ).label("user"),
            User.id.label("user_id"),
            func.count(Episode.id).label("message_count"),
        )
        .select_from(Episode)
        .join(User, User.id == Episode.user_id)
        .where(User.organization_id == org_uuid, *conditions)
        .group_by(User.id, User.name, User.email)
        .order_by(text("message_count DESC"))
        .limit(limit)
    )
    result = await db.execute(stmt)
    # note: COALESCE yields NULL only when both name and email are NULL/'';
    # the ``or`` fallback keeps the display column never null/empty.
    rows = [[r.user or str(r.user_id), r.message_count] for r in result]
    return _result(
        "top_users_by_messages",
        True,
        ["user", "message_count"],
        rows,
        {"limit": limit},
    )


# ── Prometheus query handlers (org-scoped via org_id label) ───────────────


async def _error_rate_by_day(
    db: AsyncSession,
    org_uuid: UUID,
    start: datetime,
    end: datetime,
    limit: int,
    project_id: UUID | None,
) -> dict:
    # Prometheus has no project scope — project_id is explicitly ignored.
    _ = (project_id, limit)
    promql = f'sum(increase(openzync_http_requests_total{{status="5xx",org_id="{org_uuid}"}}[1d]))'
    rows = await _prom_range(promql, start, end)
    params = {"from": start.isoformat(), "to": end.isoformat()}
    return _result(
        "error_rate_by_day", True, ["timestamp", "value"], rows, params
    )


async def _latency_percentiles(
    db: AsyncSession,
    org_uuid: UUID,
    start: datetime,
    end: datetime,
    limit: int,
    project_id: UUID | None,
) -> dict:
    # Instant snapshot — the [start, end) window is explicitly ignored, and
    # Prometheus has no project scope so project_id is ignored too.
    _ = (start, end, project_id, limit)
    queries = {
        "overall_p50": f'histogram_quantile(0.50, sum(rate(openzync_http_request_duration_seconds_bucket{{org_id="{org_uuid}"}}[5m])) by (le)) * 1000',
        "overall_p95": f'histogram_quantile(0.95, sum(rate(openzync_http_request_duration_seconds_bucket{{org_id="{org_uuid}"}}[5m])) by (le)) * 1000',
        "overall_p99": f'histogram_quantile(0.99, sum(rate(openzync_http_request_duration_seconds_bucket{{org_id="{org_uuid}"}}[5m])) by (le)) * 1000',
        "context_p50": f'histogram_quantile(0.50, sum(rate(openzync_context_latency_seconds_bucket{{org_id="{org_uuid}"}}[5m])) by (le)) * 1000',
        "context_p95": f'histogram_quantile(0.95, sum(rate(openzync_context_latency_seconds_bucket{{org_id="{org_uuid}"}}[5m])) by (le)) * 1000',
        "context_p99": f'histogram_quantile(0.99, sum(rate(openzync_context_latency_seconds_bucket{{org_id="{org_uuid}"}}[5m])) by (le)) * 1000',
        "graph_p50": f'histogram_quantile(0.50, sum(rate(openzync_graph_search_latency_seconds_bucket{{org_id="{org_uuid}"}}[5m])) by (le)) * 1000',
        "graph_p95": f'histogram_quantile(0.95, sum(rate(openzync_graph_search_latency_seconds_bucket{{org_id="{org_uuid}"}}[5m])) by (le)) * 1000',
        "graph_p99": f'histogram_quantile(0.99, sum(rate(openzync_graph_search_latency_seconds_bucket{{org_id="{org_uuid}"}}[5m])) by (le)) * 1000',
    }
    # ⚠️ Was 9 sequential awaits (~9 RTTs); one gather pays a single RTT.
    values = await asyncio.gather(
        *(_prom_instant(promql) for promql in queries.values())
    )
    rows = [
        [name, round(val, 1)] for name, val in zip(queries.keys(), values, strict=True)
    ]
    return _result(
        "latency_percentiles", True, ["metric", "value_ms"], rows, {}
    )


async def _queue_depth_over_time(
    db: AsyncSession,
    org_uuid: UUID,
    start: datetime,
    end: datetime,
    limit: int,
    project_id: UUID | None,
) -> dict:
    # Was Prometheus, now DB: pending enrichments (bits 0-5 not all set)
    # per day
    # Progress metric — archived projects are excluded because the episode
    # workers early-return on them, so including them would report a permanent
    # non-draining phantom backlog that looks like a wedged queue.
    conditions = [
        Episode.organization_id == org_uuid,
        Episode.is_deleted.is_(False),
        (Episode.enrichment_status.op("&")(ENRICHMENT_ALL))
        != ENRICHMENT_ALL,
        Episode.project_id.not_in(archived_project_ids()),
        Episode.created_at >= start,
        Episode.created_at < end,
    ]
    if project_id:
        conditions.append(Episode.project_id == project_id)
    stmt = (
        select(
            func.date_trunc("day", Episode.created_at.op("AT TIME ZONE")("UTC")).label(
                "date"
            ),
            func.count(Episode.id).label("count"),
        )
        .select_from(Episode)
        .where(*conditions)
        .group_by(text("date"))
        .order_by(text("date DESC"))
    )
    result = await db.execute(stmt)
    rows = [[str(r.date), r.count] for r in result]
    params = {"from": start.isoformat(), "to": end.isoformat()}
    return _result("queue_depth_over_time", True, ["date", "count"], rows, params)


async def _context_retrieval_rate(
    db: AsyncSession,
    org_uuid: UUID,
    start: datetime,
    end: datetime,
    limit: int,
    project_id: UUID | None,
) -> dict:
    # Prometheus has no project scope — project_id is explicitly ignored.
    _ = (project_id, limit)
    promql = f'sum(rate(openzync_context_latency_seconds_count{{org_id="{org_uuid}"}}[5m]))'
    rows = await _prom_range(promql, start, end)
    params = {"from": start.isoformat(), "to": end.isoformat()}
    return _result(
        "context_retrieval_rate", True, ["timestamp", "rate"], rows, params
    )


# ── Dispatch ──────────────────────────────────────────────────────────────

_QUERY_HANDLERS = {
    "episodes_per_day": _episodes_per_day,
    "messages_per_day": _messages_per_day,
    "activity_per_day": _activity_per_day,
    "entities_per_day": _entities_per_day,
    "facts_per_day": _facts_per_day,
    "enrichment_progress": _enrichment_progress,
    "top_projects_by_episodes": _top_projects_by_episodes,
    "top_users_by_messages": _top_users_by_messages,
    "error_rate_by_day": _error_rate_by_day,
    "latency_percentiles": _latency_percentiles,
    "queue_depth_over_time": _queue_depth_over_time,
    "context_retrieval_rate": _context_retrieval_rate,
}

# Handlers hitting Prometheus (8s budget) vs DB (5s budget) in /metrics/batch.
_PROM_QUERY_NAMES = frozenset(
    {"error_rate_by_day", "latency_percentiles", "context_retrieval_rate"}
)


def _classify_batch_error(query_name: str, exc: BaseException) -> dict:
    """Map a batch handler failure to the ``errors`` envelope entry."""
    if isinstance(exc, HTTPException):
        code = "bad_gateway" if exc.status_code == 502 else "handler_error"
        message = str(exc.detail)
    elif isinstance(exc, httpx.HTTPError):
        # raise_for_status() in _prom_instant/_prom_range surfaces transport
        # and status errors as httpx.HTTPError — Prometheus is down, not a
        # handler bug, so report bad_gateway like the 502 HTTPException path.
        code = "bad_gateway"
        message = str(exc) or repr(exc)
    elif isinstance(exc, TimeoutError):
        code = "timeout"
        message = f"Query '{query_name}' timed out"
    else:
        code = "handler_error"
        message = str(exc) or repr(exc)
    logger.warning(
        "admin_metrics.batch_query_failed",
        query=query_name,
        code=code,
        error=message,
    )
    return {"query": query_name, "code": code, "message": message}


@router.get(
    "/batch",
    summary="Run all predefined metric queries at once",
    description=(
        "Runs all 12 predefined org-scoped queries concurrently over an "
        "explicit ``[start, end)`` window and returns a partial-OK envelope: "
        "per-query failures are reported in ``errors`` with HTTP 200, and "
        "the failing ``results`` entry is omitted. Pass EITHER ``days`` "
        "(one of 7/30/90, window ``[now-days, now)``) OR ``from``+``to`` "
        "(``YYYY-MM-DD``, window ``[from 00:00 UTC, to 00:00 UTC + 1 day)``). "
        "Per-query timeouts are 5s (DB) / 8s (Prometheus) with a "
        "~15s overall budget."
    ),
)
async def get_metrics_batch(
    request: Request,
    days: int | None = Query(
        default=None,
        description="Look-back window — one of 7/30/90. "
        "Mutually exclusive with from/to.",
    ),
    from_date: str | None = Query(
        default=None,
        alias="from",
        description="Start date YYYY-MM-DD inclusive (00:00 UTC). "
        "Requires to; mutually exclusive with days.",
    ),
    to_date: str | None = Query(
        default=None,
        alias="to",
        description="End date YYYY-MM-DD inclusive "
        "(window ends to 00:00 UTC + 1 day). "
        "Requires from; mutually exclusive with days.",
    ),
    limit: int = Query(default=20, ge=1, le=50, description="Max rows for rankings"),
    include_archived: bool = Query(
        default=False,
        description="Include archived projects (top_projects_by_episodes only).",
    ),
    project_id: UUID | None = Query(
        default=None, description="Optional project filter"
    ),
    db: AsyncSession = Depends(get_db),
    org_id: str = Depends(require_permission("members:read")),
    org_config: OrgConfigBase = Depends(get_org_config),
) -> dict:
    """Run every predefined query concurrently; partial failures stay HTTP 200.

    Returns:
        ``{"results": {name: query-dict}, "errors": [...], "meta": {...}}``
        where each ``results`` value has ``query/org_scoped/columns/rows/
        total/parameters`` and each ``errors`` entry has
        ``query/code/message``. ``meta`` echoes the window as either
        ``days`` or ``from``+``to``, plus ``limit``, ``duration_ms``,
        and ``partial``.
    """
    started = time.perf_counter()
    if days is not None and (from_date is not None or to_date is not None):
        raise HTTPException(
            status_code=422,
            detail="Pass either days or from/to, not both.",
        )
    if days is not None:
        if days not in (7, 30, 90):
            raise HTTPException(
                status_code=422,
                detail="days must be one of 7, 30, 90.",
            )
        end = datetime.now(UTC)
        start = end - timedelta(days=days)
        window_meta: dict = {"days": days}
    else:
        if from_date is None or to_date is None:
            raise HTTPException(
                status_code=422,
                detail="Pass either days or both from and to.",
            )
        try:
            from_day = date.fromisoformat(from_date)
            to_day = date.fromisoformat(to_date)
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail="from/to must be YYYY-MM-DD.",
            ) from exc
        if from_day >= to_day:
            raise HTTPException(
                status_code=422,
                detail="from must be earlier than to.",
            )
        start = datetime.combine(from_day, dtime.min, tzinfo=UTC)
        end = datetime.combine(to_day, dtime.min, tzinfo=UTC) + timedelta(days=1)
        window_meta = {"from": from_date, "to": to_date}
    org_uuid = UUID(org_id)
    backend = await _resolve_graph_backend(request, db, org_config, org_uuid)

    async def _guarded(name: str, coro: Any) -> dict:
        timeout = 8.0 if name in _PROM_QUERY_NAMES else 5.0
        return await asyncio.wait_for(coro, timeout=timeout)

    async def _invoke(name: str) -> dict:
        if name == "entities_per_day":
            return await _guarded(
                name,
                _entities_per_day(
                    db, org_uuid, start, end, limit, project_id, backend
                ),
            )
        if name == "top_projects_by_episodes":
            return await _guarded(
                name,
                _top_projects_by_episodes(
                    db,
                    org_uuid,
                    start,
                    end,
                    limit,
                    project_id,
                    include_archived,
                ),
            )
        handler = _QUERY_HANDLERS[name]
        return await _guarded(
            name, handler(db, org_uuid, start, end, limit, project_id)
        )

    names = list(_QUERY_HANDLERS)
    try:
        # Concurrent ceiling is ~8s (slowest Prom query), so the 15s overall
        # budget below rarely fires — it bounds pathological scheduling, not
        # the common case.
        outcomes = await asyncio.wait_for(
            asyncio.gather(*(_invoke(n) for n in names), return_exceptions=True),
            timeout=15.0,
        )
    except TimeoutError as exc:
        duration_ms = int((time.perf_counter() - started) * 1000)
        return {
            "results": {},
            "errors": [_classify_batch_error(n, exc) for n in names],
            "meta": {
                **window_meta,
                "limit": limit,
                "duration_ms": duration_ms,
                "partial": True,
            },
        }

    results: dict[str, dict] = {}
    errors: list[dict] = []
    for name, outcome in zip(names, outcomes, strict=True):
        if isinstance(outcome, BaseException):
            # CancelledError is control flow, not a per-query failure.
            if isinstance(outcome, asyncio.CancelledError):
                raise outcome
            errors.append(_classify_batch_error(name, outcome))
        else:
            results[name] = outcome

    duration_ms = int((time.perf_counter() - started) * 1000)
    return {
        "results": results,
        "errors": errors,
        "meta": {
            **window_meta,
            "limit": limit,
            "duration_ms": duration_ms,
            "partial": bool(errors),
        },
    }


# ── DB helper functions ───────────────────────────────────────────────────────


async def _fetch_db_counts(
    db: AsyncSession, org_id: UUID, backend: GraphBackend | None
) -> tuple[EpisodeStats, GraphStats, int]:
    """Run all DB count queries for the admin summary.

    Args:
        db: Async database session.
        org_id: Organization UUID for tenant isolation.
        backend: Resolved graph backend for entity counts (``None`` →
            zeros, degraded — see :func:`_resolve_graph_backend`).

    Returns:
        Tuple of (EpisodeStats, GraphStats, user_count).
    """
    # ── Episode counts ──────────────────────────────────────────────────
    # Inventory below (total/24h) deliberately includes archived projects:
    # archiving is a soft delete, so volume must not shrink.  The progress
    # counts further down exclude them — the episode workers early-return on
    # archived projects, so those rows are terminal, never "pending".
    #
    # Invariant: added_total == enrichable_total + archived_episodes. Both
    # sides use the identical org + is_deleted filters and partition on the
    # archived subquery, so this holds by construction — keep it that way if
    # you add a third bucket.

    # Total episodes
    total_ep_result = await db.execute(
        select(func.count(Episode.id)).where(
            Episode.organization_id == org_id,
            Episode.is_deleted.is_(False),
        )
    )
    episodes_total = total_ep_result.scalar() or 0

    # Episodes in last 24h
    ep_24h_result = await db.execute(
        select(func.count(Episode.id)).where(
            Episode.organization_id == org_id,
            Episode.is_deleted.is_(False),
            Episode.created_at >= func.now() - text("interval '24 hours'"),
        )
    )
    episodes_24h = ep_24h_result.scalar() or 0

    # Episodes in archived projects — excluded from every progress count below
    archived_ep_result = await db.execute(
        select(func.count(Episode.id)).where(
            Episode.organization_id == org_id,
            Episode.is_deleted.is_(False),
            Episode.project_id.in_(archived_project_ids()),
        )
    )
    episodes_archived = archived_ep_result.scalar() or 0

    # Episodes eligible for enrichment — the progress denominator
    enrichable_ep_result = await db.execute(
        select(func.count(Episode.id)).where(
            Episode.organization_id == org_id,
            Episode.is_deleted.is_(False),
            Episode.project_id.not_in(archived_project_ids()),
        )
    )
    episodes_enrichable = enrichable_ep_result.scalar() or 0

    # Episodes with incomplete enrichment (some bits still 0)
    in_prog_result = await db.execute(
        select(func.count(Episode.id)).where(
            Episode.organization_id == org_id,
            Episode.is_deleted.is_(False),
            (Episode.enrichment_status.op("&")(ENRICHMENT_ALL))
            != ENRICHMENT_ALL,
            Episode.project_id.not_in(archived_project_ids()),
        )
    )
    episodes_in_progress = in_prog_result.scalar() or 0

    # Episodes with no enrichment started
    pending_result = await db.execute(
        select(func.count(Episode.id)).where(
            Episode.organization_id == org_id,
            Episode.is_deleted.is_(False),
            Episode.enrichment_status == 0,
            Episode.project_id.not_in(archived_project_ids()),
        )
    )
    episodes_pending = pending_result.scalar() or 0

    # Fully enriched episodes (all active bits set)
    fully_enriched_result = await db.execute(
        select(func.count(Episode.id)).where(
            Episode.organization_id == org_id,
            Episode.is_deleted.is_(False),
            (Episode.enrichment_status.op("&")(ENRICHMENT_ALL))
            == ENRICHMENT_ALL,
            Episode.project_id.not_in(archived_project_ids()),
        )
    )
    episodes_fully_enriched = fully_enriched_result.scalar() or 0

    # Episodes with embedding populated.  embed_episode also early-returns on
    # archived projects, so this must be filtered too — otherwise the
    # "Embedded" tile would exceed the "Enriched" tile, which reads as a bug.
    with_embeddings_result = await db.execute(
        select(func.count(Episode.id)).where(
            Episode.organization_id == org_id,
            Episode.is_deleted.is_(False),
            Episode.embedding.isnot(None),
            Episode.project_id.not_in(archived_project_ids()),
        )
    )
    episodes_with_embeddings = with_embeddings_result.scalar() or 0

    episode_stats = EpisodeStats(
        added_total=episodes_total,
        added_24h=episodes_24h,
        in_progress=episodes_in_progress,
        enrichment_pending=episodes_pending,
        fully_enriched=episodes_fully_enriched,
        with_embeddings=episodes_with_embeddings,
        archived_episodes=episodes_archived,
        enrichable_total=episodes_enrichable,
        fully_enriched_pct=round(
            episodes_fully_enriched / episodes_enrichable * 100, 1
        ) if episodes_enrichable > 0 else 0.0,
    )

    # ── Graph counts (backend-backed, org-wide fan-out) ─────────────
    # The PG graph_entities stub is never written by the FalkorDB/SurrealDB
    # paths — counts come from the backend's get_all_entities + len().
    # entities_total is inventory, so archived projects are included here;
    # org-wide fan-out already included them via OrgStatsResponse.
    stats = GraphStatsService(db, backend)
    project_ids = await stats.resolve_project_ids(org_id, None, include_archived=True)
    entities_total, entities_24h = await stats.entity_totals(org_id, project_ids)

    graph_stats = GraphStats(
        entities_total=entities_total,
        entities_24h=entities_24h,
        relationships_total=0,  # GraphRelationship model TBD
    )

    # ── User count ──────────────────────────────────────────────────────
    users_result = await db.execute(
        select(func.count(User.id)).where(
            User.organization_id == org_id,
            User.is_deleted.is_(False),
        )
    )
    users_total = users_result.scalar() or 0

    return episode_stats, graph_stats, users_total
