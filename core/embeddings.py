"""The embedder — one local ONNX model, one frozen dimension.

Embeddings are produced in-process by a single hardcoded model:
``nomic-ai/nomic-embed-text-v1.5`` served by ``qdrant/fastembed`` (ONNX
runtime — no torch, no ``trust_remote_code``, no network call per
inference). There is no provider routing, no per-org backend selection,
and no env-var override: the model name below is the whole
configuration surface.

Two columns enforce the dimension at the schema level —
``episodes.embedding`` and ``facts.embedding`` are both ``VECTOR(768)``
with a CHECK constraint and an HNSW cosine index — so every vector must
be exactly :data:`CANONICAL_EMBED_DIM` floats. ``nomic-embed-text-v1.5``
is 768-dimensional, which is why the swap to a local model was possible
at all.

Rules enforced through this module:

- **Asymmetric prefixes.** ``nomic-embed-text-v1.5`` is a Matryoshka
  model trained with distinct search/document prefixes. Corrupting them
  is silent — you get 768 valid floats and quietly degraded recall — so
  write paths must call :func:`embed_passage` and the query path must
  call :func:`embed_query`. Never unify them.
- **The prefixes are applied here, not by fastembed.** The model card
  marks them mandatory and fastembed's own registry metadata says
  *"Prefixes for queries/documents: necessary"*, yet
  ``passage_embed``/``query_embed`` pass their input straight through to
  ``embed()`` for this model (only ``JinaEmbeddingV3`` overrides them,
  with a ``task_id``). Both entrypoints below therefore prepend
  :data:`PASSAGE_PREFIX` / :data:`QUERY_PREFIX` themselves.
- **Unit norms are guaranteed here.** fastembed registers this model as
  ``PooledEmbedding`` (mean-pool only, *unnormalized* — observed norms
  around 20.0), so the raw vectors are normalised at this boundary.
- Every returned vector passes through :func:`validate_embedding_dim`
  before it reaches a caller, so a wrong-shape vector fails loud at this
  single choke point instead of at the ``CAST(... AS vector(768))``.

⚠️ A same-dimension model swap is invisible to
:func:`validate_embedding_dim`. Stored vectors from the previous model
stay 768-wide and pass every check while occupying a different vector
space. After changing :data:`CANONICAL_EMBED_MODEL`, run
``scripts/reset_embeddings_for_remodel.py`` to null them, then let
``workers/tasks/reconcile_enrichment.py`` re-enqueue the backfill.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from core.exceptions import ExternalServiceError

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

CANONICAL_EMBED_MODEL: str = "nomic-ai/nomic-embed-text-v1.5"
"""The single embedding model all stored vectors are produced with."""

CANONICAL_EMBED_DIM: int = 768
"""The single embedding dimension. Matches ``VECTOR(768)`` DDL."""

PASSAGE_PREFIX: str = "search_document: "
"""Mandatory task prefix for the corpus side. Applied by :func:`embed_passage`."""

QUERY_PREFIX: str = "search_query: "
"""Mandatory task prefix for the query side. Applied by :func:`embed_query`."""

# ── Module-level model cache (double-checked locking) ──────────────────────

_MODEL: Any = None
"""Lazily-constructed ``fastembed.TextEmbedding``. ``None`` until first use."""

_MODEL_LOCK: asyncio.Lock = asyncio.Lock()
"""Guards construction so concurrent first-callers load the model once."""


# ── Public API ──────────────────────────────────────────────────────────────


async def embed_passage(texts: list[str]) -> list[list[float]]:
    """Embed corpus-side text, prefixing it with :data:`PASSAGE_PREFIX`.

    Use this for everything that is written to ``episodes.embedding`` or
    ``facts.embedding``. Calling :func:`embed_query` here silently
    degrades retrieval quality — the prefixes are not interchangeable.

    The prefix is prepended by this module, because fastembed does not
    prepend it (see the module docstring) — callers pass bare text.

    Args:
        texts: Texts to embed, one vector returned per input, in order.

    Returns:
        A list of 768-float unit vectors, positionally matching ``texts``.

    Raises:
        ExternalServiceError: If the embedder returns a vector that is not
            exactly :data:`CANONICAL_EMBED_DIM` floats, or a zero-norm
            vector (unnormalisable, and NaN-poisoning downstream).
        ImportError: If ``fastembed`` is not installed.
        Exception: Whatever ``fastembed``/``onnxruntime`` raises on load or
            inference — propagated unmodified, never swallowed.
    """
    model = await _ensure_model()
    # note: the prefix is NOT optional — fastembed's passage_embed does not
    # add it for this model, and omitting it silently degrades recall with no
    # error. Do not "simplify" this into a bare pass-through of `texts`.
    vectors = await _run_off_loop(
        lambda: _consume(
            model.passage_embed, texts, prefix=PASSAGE_PREFIX, source="embed_passage"
        ),
        source="embed_passage",
    )
    for vec in vectors:
        validate_embedding_dim(vec, source="embed_passage")
    return vectors


async def embed_query(texts: list[str]) -> list[list[float]]:
    """Embed search queries, prefixing each with :data:`QUERY_PREFIX`.

    Use this only for the query side of retrieval (see
    :meth:`services.hybrid_retriever.HybridRetriever._embed_query`).
    Calling :func:`embed_passage` here silently degrades recall.

    The prefix is prepended by this module, because fastembed does not
    prepend it (see the module docstring) — callers pass bare text.

    Args:
        texts: Query strings to embed, one vector returned per input, in
            order.

    Returns:
        A list of 768-float unit vectors, positionally matching ``texts``.

    Raises:
        ExternalServiceError: If the embedder returns a vector that is not
            exactly :data:`CANONICAL_EMBED_DIM` floats, or a zero-norm
            vector (unnormalisable, and NaN-poisoning downstream).
        ImportError: If ``fastembed`` is not installed.
        Exception: Whatever ``fastembed``/``onnxruntime`` raises on load or
            inference — propagated unmodified, never swallowed.
    """
    model = await _ensure_model()
    # note: see embed_passage — the prefix is mandatory and applied here.
    vectors = await _run_off_loop(
        lambda: _consume(
            model.query_embed, texts, prefix=QUERY_PREFIX, source="embed_query"
        ),
        source="embed_query",
    )
    for vec in vectors:
        validate_embedding_dim(vec, source="embed_query")
    return vectors


def validate_embedding_dim(vec: list[float], *, source: str) -> None:
    """Reject any embedding that is not exactly canonical-dim.

    Args:
        vec: The embedding vector returned by the provider.
        source: Caller name for the error detail (e.g. ``"embed_fact"``).

    Raises:
        ExternalServiceError: If ``len(vec) != CANONICAL_EMBED_DIM`` or any
            element is not a float/int.
    """
    if len(vec) != CANONICAL_EMBED_DIM or not all(
        isinstance(v, (float, int)) and not isinstance(v, bool) for v in vec
    ):
        raise ExternalServiceError(
            message=(
                f"Invalid embedding in {source}: len {len(vec)} "
                f"(expected canonical {CANONICAL_EMBED_DIM} float elements). "
                "Refusing to store."
            ),
            detail={
                "source": source,
                "got": len(vec),
                "expected": CANONICAL_EMBED_DIM,
            },
        )


# ── Internal helpers ────────────────────────────────────────────────────────


def _consume(
    embed: Callable[[list[str]], Any],
    texts: list[str],
    *,
    prefix: str,
    source: str,
) -> list[list[float]]:
    """Prefix, drain and normalise one fastembed batch into nested lists.

    Must be called on a worker thread — ``passage_embed``/``query_embed``
    are generators over synchronous ONNX inference, so merely constructing
    the generator costs nothing while iterating it blocks.

    Args:
        embed: Bound fastembed method (``passage_embed``/``query_embed``).
            Typed loosely because it yields numpy arrays and numpy is
            imported lazily below.
        texts: Raw caller text, prefixed here — fastembed does not prefix.
        prefix: The task prefix to prepend to every text, e.g.
            :data:`PASSAGE_PREFIX`.
        source: Caller name for the error detail.

    Returns:
        One list of floats per input text, each exactly unit-norm.

    Raises:
        ExternalServiceError: If any returned row has zero (or NaN) norm,
            which cannot be normalised into a unit vector.
    """
    import numpy as np

    matrix = np.array(list(embed([prefix + text for text in texts])))
    if matrix.size == 0:
        return []

    # fastembed registers this model as PooledEmbedding — mean-pool only, no
    # normalisation (observed norms ~20). Normalise at the boundary so every
    # stored vector has a guaranteed unit norm regardless of what fastembed
    # hands back, and so a future switch to dot-product / <#> index ops cannot
    # silently change the ranking scale.
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    if not bool(np.all(norms > 0)):
        raise ExternalServiceError(
            message=(
                f"Invalid embedding in {source}: model returned a zero-norm "
                "vector, which cannot be normalised. Refusing to store."
            ),
            detail={
                "source": source,
                "zero_norm_rows": np.flatnonzero(norms <= 0).tolist(),
            },
        )
    return (matrix / norms).tolist()


async def _run_off_loop(
    fn: Callable[[], list[list[float]]], *, source: str
) -> list[list[float]]:
    """Run *fn* on the default executor so ONNX inference never blocks asyncio.

    Args:
        fn: Zero-arg callable doing the full consume-and-convert work.
        source: Caller name for the error log.

    Returns:
        Whatever *fn* returned.

    Raises:
        Exception: Whatever *fn* raised, after a structured error log.
    """
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, fn)
    except Exception as exc:
        logger.error(
            "embeddings.inference_failed",
            extra={"source": source, "model": CANONICAL_EMBED_MODEL, "error": str(exc)},
            exc_info=True,
        )
        raise


async def _ensure_model() -> Any:
    """Load the ONNX model with double-checked locking.

    The instance is cached at module level, so every ``embed_passage`` /
    ``embed_query`` caller in the process shares one loaded model.

    Returns:
        The loaded ``fastembed.TextEmbedding`` instance.

    Raises:
        ImportError: If ``fastembed`` is not installed.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    async with _MODEL_LOCK:
        # Double-check — another coroutine may have loaded it while we
        # were waiting for the lock.
        if _MODEL is not None:
            return _MODEL

        try:
            from fastembed import TextEmbedding  # noqa: PLC0415
        except ImportError as err:
            raise ImportError(
                "fastembed is not installed — it is a base dependency. "
                "Install with: pip install -e ."
            ) from err

        # Constructing the model hits the HF cache / disk — off the loop.
        logger.info(
            "embeddings.loading_model",
            extra={"model": CANONICAL_EMBED_MODEL, "dim": CANONICAL_EMBED_DIM},
        )
        loop = asyncio.get_running_loop()
        model: Any = await loop.run_in_executor(
            None,
            lambda: TextEmbedding(CANONICAL_EMBED_MODEL, lazy_load=True),
        )
        _MODEL = model
        logger.info("embeddings.model_loaded", extra={"model": CANONICAL_EMBED_MODEL})
        return _MODEL


if __name__ == "__main__":  # pragma: no cover — operator self-check
    import asyncio as _asyncio
    import math as _math

    async def _self_check() -> None:
        texts = ["openzyc persists agent memory in a pgvector graph", "unrelated text"]

        for fn in (embed_passage, embed_query):
            vecs = await fn(texts)
            assert len(vecs) == len(texts), f"{fn.__name__}: wrong vector count"
            for vec in vecs:
                assert len(vec) == CANONICAL_EMBED_DIM, (
                    f"{fn.__name__}: got {len(vec)}, want {CANONICAL_EMBED_DIM}"
                )
                assert all(isinstance(v, float) for v in vec), (
                    f"{fn.__name__}: non-float element"
                )
                norm = _math.sqrt(sum(v * v for v in vec))
                assert abs(norm - 1.0) < 1e-3, f"{fn.__name__}: not normalized ({norm})"
            print(f"{fn.__name__}: OK — {len(vecs)}x{CANONICAL_EMBED_DIM}, |v|=1")

        # The prefixes must actually reach the model: the same text embedded on
        # both sides must NOT land on the same vector (both are unit vectors, so
        # the dot product IS the cosine). Before the fix this was exactly 1.0.
        probe = "openzyc persists agent memory in a pgvector graph"
        passage_vec = (await embed_passage([probe]))[0]
        query_vec = (await embed_query([probe]))[0]
        same_text_cosine = sum(
            a * b for a, b in zip(passage_vec, query_vec, strict=True)
        )
        assert same_text_cosine < 0.999, (
            f"prefixes not applied — identical text scored {same_text_cosine:.4f}"
        )
        print(
            f"prefix asymmetry: OK — identical text cosine "
            f"{same_text_cosine:.4f} < 0.999"
        )

        # Dim validation must reject a wrong-width vector, not pad or truncate.
        try:
            validate_embedding_dim([0.1] * 512, source="self_check")
        except Exception as exc:
            assert type(exc).__name__ == "ExternalServiceError", type(exc).__name__
            print("validate_embedding_dim: OK — rejects 512")
        else:
            raise AssertionError("validate_embedding_dim accepted a 512-dim vector")

    _asyncio.run(_self_check())
