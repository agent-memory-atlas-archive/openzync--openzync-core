#!/usr/bin/env python3
"""Probe Ollama embedding equivalence after the fastembed removal.

Operator tool, NOT a test: embeds a fixed sample corpus via the NEW
Ollama client twice (idempotence) plus once on the query side, and
checks the shipped contract — 768 dims, unit norms, passage/query
asymmetry (cosine < 0.999). With ``--reference-vectors`` it also
compares passage vectors against pre-recorded fastembed vectors
(cosine per row, PASS when mean >= 0.999).

Reference JSON format — either a bare list of vectors (positional
against the sample corpus below) or an object::

    {"texts": [...], "vectors": [[...], ...]}

When ``texts`` is present, rows are aligned by text; otherwise
positionally. Supply the file from the prod backup at probe time.

Usage:
    python scripts/probe_ollama_equivalence.py [--base-url URL]
        [--reference-vectors refs.json]

Exit codes: 0 all checks pass, 1 a check failed, 2 infra error.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.embeddings import (  # noqa: E402
    CANONICAL_EMBED_DIM,
    CANONICAL_EMBED_MODEL,
    OLLAMA_KEEP_ALIVE,
    PASSAGE_PREFIX,
    QUERY_PREFIX,
)

SAMPLE_CORPUS: tuple[str, ...] = (
    "OpenZync persists agent memory in a pgvector graph.",
    "The worker backfills embeddings for new episodes overnight.",
    "Hybrid retrieval fuses vector search with full-text ranking.",
    "Credits are deducted per tool call with idempotency keys.",
    "Readiness probes gate traffic until the embedder is warm.",
    "A short note about nothing in particular.",
)

TIMEOUT_S = 30.0
ASYMMETRY_MAX_COSINE = 0.999
REFERENCE_MIN_MEAN_COSINE = 0.999
IDEMPOTENCE_MIN_COSINE = 0.999999
NORM_TOLERANCE = 1e-6


def _cosine(a: list[float], b: list[float]) -> float:
    """Return the cosine similarity of two vectors."""
    va = np.array(a, dtype=np.float64)
    vb = np.array(b, dtype=np.float64)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


async def _embed(
    client: httpx.AsyncClient, base_url: str, texts: list[str]
) -> list[list[float]]:
    """POST one batch to Ollama /api/embed — mirrors core.embeddings."""
    resp = await client.post(
        f"{base_url}/api/embed",
        json={
            "model": CANONICAL_EMBED_MODEL,
            "input": texts,
            "keep_alive": OLLAMA_KEEP_ALIVE,
        },
    )
    resp.raise_for_status()
    rows = resp.json()["embeddings"]
    norms = np.linalg.norm(np.array(rows, dtype=np.float64), axis=-1, keepdims=True)
    return (np.array(rows, dtype=np.float64) / norms).tolist()


def _load_reference(path: Path) -> tuple[list[str], list[list[float]]]:
    """Load reference vectors, aligned to the sample corpus.

    Returns:
        (texts, vectors) positionally aligned with the compared run.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        vectors = raw["vectors"]
        texts = raw.get("texts")
        if texts is not None:
            by_text = dict(zip(texts, vectors, strict=True))
            missing = [t for t in SAMPLE_CORPUS if t not in by_text]
            if missing:
                raise ValueError(f"reference missing texts: {missing[:3]}")
            return list(SAMPLE_CORPUS), [by_text[t] for t in SAMPLE_CORPUS]
        return list(SAMPLE_CORPUS[: len(vectors)]), vectors
    return list(SAMPLE_CORPUS[: len(raw)]), raw


async def _run(base_url: str, reference: Path | None) -> int:
    """Run all probe checks. Returns the process exit code."""
    failures: list[str] = []
    base_url = base_url.rstrip("/")
    async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
        passage_a = await _embed(
            client, base_url, [PASSAGE_PREFIX + t for t in SAMPLE_CORPUS]
        )
        passage_b = await _embed(
            client, base_url, [PASSAGE_PREFIX + t for t in SAMPLE_CORPUS]
        )
        query = await _embed(client, base_url, [QUERY_PREFIX + SAMPLE_CORPUS[0]])

    for label, vecs in (("passage/run-a", passage_a), ("passage/run-b", passage_b)):
        if len(vecs) != len(SAMPLE_CORPUS):
            failures.append(
                f"{label}: got {len(vecs)} vectors, want {len(SAMPLE_CORPUS)}"
            )
        for i, vec in enumerate(vecs):
            if len(vec) != CANONICAL_EMBED_DIM:
                failures.append(f"{label}[{i}]: dim {len(vec)}")
            norm = math.sqrt(sum(v * v for v in vec))
            if abs(norm - 1.0) > NORM_TOLERANCE:
                failures.append(f"{label}[{i}]: |v|={norm:.6f}")
    print(
        f"dim+norm: {'OK' if not failures else 'FAIL'} "
        f"({len(passage_a)}x{CANONICAL_EMBED_DIM}, unit-norm)"
    )

    idem = min(_cosine(a, b) for a, b in zip(passage_a, passage_b, strict=True))
    print(
        f"idempotence: min cosine {idem:.9f} "
        f"({'OK' if idem >= IDEMPOTENCE_MIN_COSINE else 'FAIL'})"
    )
    if idem < IDEMPOTENCE_MIN_COSINE:
        failures.append(f"idempotence cosine {idem:.9f}")

    asym = _cosine(passage_a[0], query[0])
    print(
        f"asymmetry: passage/query cosine {asym:.4f} "
        f"({'OK' if asym < ASYMMETRY_MAX_COSINE else 'FAIL'})"
    )
    if asym >= ASYMMETRY_MAX_COSINE:
        failures.append(f"asymmetry cosine {asym:.4f}")

    if reference is not None:
        texts, ref_vecs = _load_reference(reference)
        mine = dict(zip(SAMPLE_CORPUS, passage_a, strict=True))
        cosines = [_cosine(mine[t], r) for t, r in zip(texts, ref_vecs, strict=True)]
        mean_cos = sum(cosines) / len(cosines)
        print(
            f"reference: mean cosine {mean_cos:.6f} "
            f"min {min(cosines):.6f} over {len(cosines)} rows "
            f"({'PASS' if mean_cos >= REFERENCE_MIN_MEAN_COSINE else 'FAIL'})"
        )
        if mean_cos < REFERENCE_MIN_MEAN_COSINE:
            failures.append(f"reference mean cosine {mean_cos:.6f}")

    if failures:
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PROBE PASS")
    return 0


def main() -> int:
    """Parse args and run the probe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://ollama:11434")
    parser.add_argument("--reference-vectors", type=Path, default=None)
    args = parser.parse_args()
    try:
        return asyncio.run(_run(args.base_url, args.reference_vectors))
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
        print(f"INFRA ERROR: Ollama unreachable at {args.base_url}: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
