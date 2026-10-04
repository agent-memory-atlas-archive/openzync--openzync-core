# Changelog

All notable changes to this project will be documented in this file.

<!-- towncrier release notes start -->

## [1.0.0rc4] - 2026-10-04

### Breaking Changes

- Replace provider-routed embeddings with a single frozen embedder served from ONE shared Ollama container (`nomic-ai/nomic-embed-text-v1.5`, 274MB F16, canonical `VECTOR(768)` unchanged): per-org `embedding_*` overrides are removed, dead `OZ_EMBEDDING_BACKEND`/`MODEL`/`DIM` env vars and Helm `embedding.*` keys are deleted (BREAKING cleanup), and the `/ready` payload gains an `embeddings` readiness check. (+onnx-embedder-freeze)
### Changed

- Serve embeddings from the shared Ollama container over HTTP (api/worker call Ollama, boot prewarm with `/ready` gated until warm; CD pulls and warms the model before app restart; equivalence probe cosine 0.999999 so no remodel needed); `fastembed`/`onnxruntime` removed from dependencies. Worker resource limits set to 1 CPU/2G, Helm `pullPolicy` set to Always, and the `numpy==2.2.6` pin kept. (+onnx-weights-baked)
## [1.0.0rc3] - 2026-09-30

### Breaking Changes

- Freeze embeddings to the canonical model `nomic-ai/nomic-embed-text-v1.5` (768 dims): migration 0054 converts `episodes`/`facts` embeddings to native `VECTOR(768)` with HNSW cosine indexes (non-conforming rows are nulled for re-embed), and per-org `embedding_model`/`embedding_dim` overrides are now rejected with 400 `embedding_frozen`. Ollama `nomic-embed-text` remains as the dim-compatible dev fallback. (+embedding-freeze)
- **BREAKING:** `DELETE /v1/projects/{project_id}/memory` now requires a JSON body `{"confirm": "<project_id>"}` matching the path project — a missing or mismatched confirm is rejected with 422 and nothing is deleted. A successful wipe is recorded as a destructive-action `memory.wipe` audit log entry carrying the actor, project ID, confirm-matched flag, timestamp, and deletion counts. (+p0-memory-delete-confirm)
- Message/episode listing now uses the v1 cursor envelope: legacy or malformed cursors are rejected with 400 `cursor_expired` (restart from page 1), and `get_by_project_id` ordering is aligned to `(sequence_number, id)` to match the keyset predicate.
  PII redaction is now fail-closed: an OpenBao/redaction outage returns 503 `pii_unavailable` with `Retry-After: 30` instead of persisting unredacted content.
  The legacy `quotas -> pii` config fallback is removed — orgs still carrying quota-stored PII config must migrate to the dedicated PII store before upgrading. (+p0-pagination-pii)
### Fixed

- Fixed API crash-loop on boot: the pgvector codec hook in `core/db.py` drove `register_vector` through the adapted connection's `await_` method, which SQLAlchemy 2.1 removed, while the unpinned `.[llm]` Docker install floated to 2.1.x — so every pooled connection raised `AttributeError` during lifespan startup and uvicorn exited. The hook now uses the documented `sqlalchemy.util.concurrency.await_only` bridge (identical behavior on 2.0.x, where the dialect's own `await_` is that same function), and both Docker images now build with a constraints file pinning SQLAlchemy/asyncpg/pgvector to the CI-tested versions (2.0.50/0.31/0.4.2), so a rebuild can no longer float past what the code is verified against, plus `numpy==2.2.6` (last release before the x86-64-v2 baseline wheels that cannot load on the production CPU).
- Fixed `/health` version reporting going stale at the hatch-vcs fallback: Docker builds and the CD pipeline now inject the release tag via the `APP_VERSION`/`OPENZYNC_VERSION` build-arg, which `core/_version.py` prefers over the packaged metadata. This was needed because the Docker build context ships without git history, so hatch-vcs could never resolve the tag and every image reported the fallback. Deployed environments now return the actually-released version, so the dashboard sidebar footer matches the running release instead of a frozen `1.0.0rc1`. (+app-version-build-arg)
- Fixed community detection running on stale state and crashing on real backend data: prior-run `community` nodes and `MEMBER_OF` edges are now excluded from the Label Propagation input (keyed on the cross-backend `type` contract) and prior communities are deleted before fresh ones are stored, so reruns replace instead of duplicating; the summarisation prompt reads `type` instead of `relationship_type` (which raised `KeyError` on every real edge); and `GET /graph/communities` `member_count` is computed at read time from `MEMBER_OF` edges instead of a stored attribute. Note: SurrealDB `delete_entity` removes only the node record, so incident `member_of` edge rows may survive a rerun cleanup as orphans — excluded from detection input regardless. (+community-detection)
- Fixed enrichment progress stalling at 41%: episodes in archived projects are now excluded from the progress counts (the workers never enrich them, so they sat as a permanent phantom backlog of ~3089). The dashboard percentage now runs 0–100% over enrichable episodes only, with new `archived_episodes`/`enrichable_total` fields on the summary so the excluded volume stays visible. (+enrichment-archived-progress)
- Fixed episodes wedged in the enrichment queue forever when extraction finds nothing: empty structured output and empty/filtered/deduplicated fact output now stamp their completion bit (assessed, nothing to store) instead of leaving the episode permanently pending, so the queue drains. (+enrichment-assessed-empty)
- Fixed FalkorDB `create_relationship` cloning endpoint nodes: the edge-pattern `MERGE` created bare duplicate `:Entity` nodes (id only, no name/type) and attached new edges to the duplicates instead of the real entities. Endpoints are now `MATCH`ed first and only the edge is merged; a missing endpoint raises `NotFoundError` instead of silently ghosting. (+falkordb-ghost-nodes)
- Fix FalkorDB v4 dialect (`<>` comparisons, v4 index DDL, loud schema bootstrap with a versioned re-run guard) and fail fast with a named error when OpenAI/Azure/OpenAI-like chat returns no choices instead of indexing into an empty list. (+falkordb-llm-hardening)
- Fixed session-scoped graph returning empty: `GET /graph/nodes?session_id=` traversed `:Session` stub nodes that are never created, so it always yielded `[]` (and the session graph page showed 0 nodes/0 edges). Added `GraphBackend.get_entities_for_episodes`; `GraphService` now resolves the session's episode IDs from PostgreSQL and filters `MENTIONS` edges by episode (capped, newest first). Existing graphs light up with no backfill; response shapes unchanged. (+session-graph)
- Fixed message/episode cursor keyset predicate to use tuple comparison on `(sequence_number, id)`, eliminating duplicate/skipped rows at page boundaries.
  Episode `sequence_number` is now server-assigned and contiguous via session `FOR UPDATE` + `MAX+1`, guarded by a partial unique index and backfill migration 0053 (duplicate sequence numbers eliminated; conflicts surface as 409 for client retry).
  Blob extraction failures are now fail-closed (raise + ARQ retry, success bit unset) instead of being marked successful.
  Content dedup is atomic via a Lua GET-or-SET claim: concurrent identical ingests replay the winner `job_id` and write a single row. (+p0-ingest-correctness)
### Changed

- Decoupled the FalkorDB/SurrealDB graph path from the PostgreSQL `graph_entities` stub table (never written by those backends). Migration 0056 drops the `facts` subject/object and `graph_observations` subject/related foreign keys to `graph_entities` (UUID columns kept; downgrade is not clean once orphan UUIDs accumulate). Admin stats/metrics entity counts, enrichment prompt user-entities, and the new `GraphBackend.get_entities_for_user` now read from the configured graph backend instead of the stub table — no API contract changes. FalkorDB/SurrealDB schema bootstrap is idempotent across duplicate-DEFINE variants (`already exists` / `already indexed`). (+graph-decouple)
- Changed the periodic enrichment reconciliation scan to match only episodes missing assessed LLM work and to skip soft-deleted rows, so it re-enqueues genuinely stale episodes instead of churning on rows that can never complete. (+reconcile-scan-mask)
## [Unreleased]
