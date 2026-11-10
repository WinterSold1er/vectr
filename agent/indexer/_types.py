"""Chunk dataclass, EmbedProvider protocol, embed provider implementations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from agent.config import (
    EMBEDDING_API_KEY_ENV as _EMBEDDING_API_KEY_ENV,
    EMBEDDING_DEFAULT_MODEL as _EMBEDDING_DEFAULT_MODEL,
    EMBEDDING_MAX_SEQ_LENGTH as _EMBEDDING_MAX_SEQ_LENGTH,
    EMBEDDING_OUTPUT_DIMENSIONALITY as _EMBEDDING_OUTPUT_DIMENSIONALITY,
    EMBEDDING_THREAD_CAP as _EMBEDDING_THREAD_CAP,
    vectr_cache_root as _vectr_cache_root,
)


# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------

@dataclass
class CodeChunk:
    chunk_id: str
    content: str
    file_path: str
    language: str
    node_type: str
    start_line: int
    end_line: int
    symbol_name: str


# ---------------------------------------------------------------------------
# Embedding provider protocol
# ---------------------------------------------------------------------------

class EmbedProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, texts: list[str]) -> list[list[float]]:
        """Embed search-query text. Distinct from `embed()` (document/indexing side)
        because some asymmetric embedding models require a different, model-
        registered prompt for queries than for the passages they're matched
        against. Providers with no such distinction may just delegate to
        `embed()`.
        """
        ...


class LocalEmbedProvider:
    """Uses sentence-transformers (no API key). Default model: config.yaml embedding.default_model
    (UPG-EMBEDDER-SWAP-GRANITE)."""

    def __init__(self, model_name: str = _EMBEDDING_DEFAULT_MODEL) -> None:
        from agent.model_cache import load_with_offline_preference, suppress_model_load_noise
        suppress_model_load_noise()
        import torch
        from sentence_transformers import SentenceTransformer

        # UPG-EMBED-THREAD-CONTENTION: cap torch's CPU thread pool at
        # construction time (config.yaml embedding.thread_cap /
        # thread_cap_auto_fraction). This is a process-wide torch setting —
        # applying it once here also caps the cross-encoder reranker, which
        # shares the same pool — so it belongs at the one embed-provider
        # construction site, not scattered across call sites. Without a cap,
        # a debounced watcher batch embed saturates every core and the HTTP
        # event loop gets no scheduler slice, so a live session's first MCP
        # call (vectr_status/vectr_recall) can time out even though no lock
        # is held.
        torch.set_num_threads(_EMBEDDING_THREAD_CAP)
        try:
            # Interop-thread count can only be set once per process, before
            # any parallel torch work has started — a later
            # LocalEmbedProvider construction in the same process (e.g.
            # across tests) raises RuntimeError on a second call. Safe to
            # ignore: the first call already applied the cap process-wide.
            torch.set_num_interop_threads(_EMBEDDING_THREAD_CAP)
        except RuntimeError:
            pass

        cache_dir = _vectr_cache_root() / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        # UPG-RERANKER-HF-NETWORK: prefer an offline (local_files_only) load
        # when this model is already fully cached, so a warm daemon start
        # never makes live huggingface.co calls (HEAD/GET on config.json,
        # tokenizer_config.json, repo tree listings, ...) just to re-confirm
        # a cache it already has. Falls back to a network-enabled load on a
        # genuine cache miss (first run) so that UX is unchanged.
        self._model = load_with_offline_preference(
            # torch_dtype float32 is load-bearing on CPU (UPG-EMBED-CPU-DTYPE):
            # transformers honors the checkpoint's declared dtype, and a model
            # shipping bfloat16 weights (the current default does) runs through
            # software-emulated bf16 matmuls on CPU — measured 12x slower than
            # float32 for identical inputs. On CPU float32 is strictly correct;
            # revisit only if a GPU/MPS device path is ever added.
            lambda local_only: SentenceTransformer(
                model_name,
                cache_folder=str(cache_dir),
                trust_remote_code=True,
                device="cpu",
                local_files_only=local_only,
                model_kwargs={"torch_dtype": torch.float32},
            ),
            model_name,
            str(cache_dir),
        )
        # Cap the encoder's sequence length (config.yaml embedding.max_seq_length).
        # A long-context model's own config otherwise wins — the current default
        # declares 8192, and encoding long chunks at full length on CPU has
        # quadratic-attention cost that stalls a full-corpus index for hours.
        # Only ever lowers: a model that already declares less keeps its own cap.
        current_cap = getattr(self._model, "max_seq_length", None)
        if current_cap is None or current_cap > _EMBEDDING_MAX_SEQ_LENGTH:
            self._model.max_seq_length = _EMBEDDING_MAX_SEQ_LENGTH
        # Asymmetric embedding models (arctic-embed and others) register a "query"
        # prompt in their sentence-transformers config that must be prepended to
        # search queries but NOT to the documents/chunks being indexed. Detected from
        # the loaded model itself (never hardcoded) so symmetric models — which
        # register no such prompt — embed queries and documents identically, unchanged.
        self._has_query_prompt = "query" in getattr(self._model, "prompts", {})

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, texts: list[str]) -> list[list[float]]:
        if not self._has_query_prompt:
            return self.embed(texts)
        embeddings = self._model.encode(
            texts,
            prompt_name="query",
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()


class VoyageEmbedProvider:
    """Uses Voyage AI code embedding model (requires VOYAGE_API_KEY).

    No CPU thread cap (UPG-EMBED-THREAD-CONTENTION) applies here — embedding
    happens server-side over HTTP; there is no local torch thread pool to
    contend with the event loop.
    """

    def __init__(self, model_name: str = "voyage-code-2") -> None:
        import voyageai  # type: ignore
        self._client = voyageai.Client()
        self._model = model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = self._client.embed(texts, model=self._model)
        return result.embeddings

    def embed_query(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)


class OpenAIEmbedProvider:
    """Uses OpenAI embedding model (requires OPENAI_API_KEY).

    No CPU thread cap (UPG-EMBED-THREAD-CONTENTION) applies here — same
    reasoning as VoyageEmbedProvider above: embedding is a remote HTTP call,
    not local CPU-bound torch work.
    """

    def __init__(self, model_name: str = "text-embedding-3-small") -> None:
        from openai import OpenAI
        self._client = OpenAI()
        self._model = model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in response.data]

    def embed_query(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)


def _resolve_google_api_key(env_name: str) -> str:
    """Key for the Google embeddings API: `env_name` first, then the
    "gemini_api_key" entry of ~/.vectr/secrets.json.

    Only the channel is ever named — the key value is never logged and never
    quoted in an error message. A secrets file that exists but is unreadable or
    malformed raises instead of silently falling back to a keyless request, and
    a group/other-readable file is warned about (it should be mode 600).
    """
    import json
    import logging
    import os
    from pathlib import Path

    key = os.environ.get(env_name)
    if key:
        return key

    path = Path.home() / ".vectr" / "secrets.json"
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raw = None
    except OSError as exc:
        raise RuntimeError(f"Google embeddings: {path} could not be read: {exc}") from exc

    if raw is not None:
        try:
            stored = json.loads(raw).get("gemini_api_key")
        except ValueError as exc:
            raise RuntimeError(f"Google embeddings: {path} is not valid JSON: {exc}") from exc
        if stored:
            mode = path.stat().st_mode & 0o777
            if mode & 0o077:
                logging.getLogger("vectr.embed").warning(
                    "vectr: %s is readable by group/other (mode %03o) — chmod 600 it", path, mode
                )
            return str(stored)

    raise RuntimeError(
        f"Google embeddings need an API key: set {env_name}, or store "
        f'"gemini_api_key" in {path} (mode 600)'
    )


def _embed_chars_per_second() -> float:
    """Embedding request budget in characters per second, used to pace
    batchEmbedContents against the key's tokens-per-minute quota.

    Measured 2026-11-10 against gemini-embedding-2: 100-text batches (~14k tokens
    each) sent back to back return HTTP 429, the same batches spaced 40s apart
    return 200 — about 72k chars/min on the free tier. Override with
    VECTR_EMBED_CHARS_PER_SECOND once the key moves to a paid quota tier.
    """
    import os

    default = 1200.0
    raw = os.environ.get("VECTR_EMBED_CHARS_PER_SECOND")
    try:
        value = float(raw) if raw else default
    except ValueError:
        return default
    if value <= 0:
        return default
    # A misconfigured or absurd rate must not stall indexing nor hammer the quota.
    return min(10000.0, max(50.0, value))


class GoogleEmbedProvider:
    """Google Generative Language embeddings via the batchEmbedContents REST
    endpoint (no SDK: plain JSON over httpx, the dependency vectr already has).

    Nothing about the model is hardcoded here: the model name comes from the
    `google/<model>` spec, the output size from `embedding.output_dimensionality`
    and the key's env-var name from `embedding.api_key_env`. As with the other
    API providers there is no local torch thread pool to cap
    (UPG-EMBED-THREAD-CONTENTION).
    """

    _BATCH_LIMIT = 100          # batchEmbedContents accepts at most 100 requests
    _MAX_ATTEMPTS = 4
    _TIMEOUT_SECONDS = 30.0

    def __init__(
        self,
        model_name: str,
        output_dimensionality: int | None = None,
        api_key_env: str = _EMBEDDING_API_KEY_ENV,
    ) -> None:
        if not model_name:
            raise ValueError(
                "a google/ embedding spec must name a model, e.g. google/gemini-embedding-2"
            )
        self._model = model_name
        self._dim = output_dimensionality
        self._api_key = _resolve_google_api_key(api_key_env)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, texts: list[str]) -> list[list[float]]:
        # Google's asymmetric task types: indexed chunks are embedded as
        # RETRIEVAL_DOCUMENT, search queries as RETRIEVAL_QUERY.
        return self._embed(texts, "RETRIEVAL_QUERY")

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        import time

        vectors: list[list[float]] = []
        budget = _embed_chars_per_second()
        for start in range(0, len(texts), self._BATCH_LIMIT):
            batch = texts[start:start + self._BATCH_LIMIT]
            vectors.extend(self._embed_batch(batch, task_type))
            # Hold the request rate under the key's tokens-per-minute budget: pause
            # for as long as this batch's payload would take at that rate, but never
            # long enough to stall a pass. Only when another batch follows.
            if start + self._BATCH_LIMIT < len(texts):
                time.sleep(min(60.0, sum(len(t) for t in batch) / budget))
        return vectors

    def _embed_batch(self, texts: list[str], task_type: str) -> list[list[float]]:
        import time

        import httpx

        requests: list[dict] = []
        for text in texts:
            request: dict = {
                "model": f"models/{self._model}",
                "content": {"parts": [{"text": text}]},
                "taskType": task_type,
            }
            if self._dim is not None:
                request["outputDimensionality"] = self._dim
            requests.append(request)

        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model}:batchEmbedContents"
        )
        for attempt in range(1, self._MAX_ATTEMPTS + 1):
            try:
                response = httpx.post(
                    url,
                    json={"requests": requests},
                    headers={"x-goog-api-key": self._api_key},
                    timeout=self._TIMEOUT_SECONDS,
                )
            except httpx.HTTPError as exc:
                # Connection-level failures (TLS handshake resets such as
                # UNEXPECTED_EOF_WHILE_READING, read timeouts, refused connections)
                # are transient: retry them with the same capped backoff as a 429
                # instead of aborting a whole indexing pass half-indexed. The key
                # is not in play here — it rides in a header, not in the URL that
                # httpx quotes in the error.
                if attempt < self._MAX_ATTEMPTS:
                    time.sleep(min(60.0, 2.0 ** (attempt - 1)))
                    continue
                raise RuntimeError(
                    f"Google embeddings request for {self._model} failed: {exc}"
                ) from exc

            if response.status_code == 429 and attempt < self._MAX_ATTEMPTS:
                # Honour the server's own wait when it sends one; a missing or
                # unparseable Retry-After falls back to exponential backoff.
                try:
                    delay = float(response.headers.get("retry-after"))
                except (TypeError, ValueError):
                    delay = min(60.0, 2.0 ** (attempt - 1))
                time.sleep(max(0.0, delay))
                continue

            if response.status_code != 200:
                # The key is never echoed: only the status and the provider's own
                # (key-free) error body are reported.
                raise RuntimeError(
                    f"Google batchEmbedContents failed for {self._model}: "
                    f"HTTP {response.status_code} {response.reason_phrase} "
                    f"(attempt {attempt}/{self._MAX_ATTEMPTS}): {response.text[:200]}"
                )

            embeddings = response.json().get("embeddings")
            if not isinstance(embeddings, list) or len(embeddings) != len(texts):
                got = len(embeddings) if isinstance(embeddings, list) else "none"
                raise RuntimeError(
                    f"Google batchEmbedContents returned {got} embeddings for "
                    f"{len(texts)} texts ({self._model})"
                )
            vectors: list[list[float]] = []
            for item in embeddings:
                values = item.get("values") if isinstance(item, dict) else None
                if not isinstance(values, list) or not values:
                    raise RuntimeError(
                        f"Google batchEmbedContents returned a malformed embedding "
                        f"entry for {self._model}"
                    )
                vectors.append(self._normalize(values))
            return vectors

    def _normalize(self, values: list[float]) -> list[float]:
        """Google returns an already unit-length vector only at its full 3072-dim
        output; a truncated `outputDimensionality` comes back unnormalized.
        Renormalize whenever the vector is not already unit length, so cosine
        scoring against vectr's other vectors stays comparable whatever the
        model or dimension."""
        import math

        norm = math.sqrt(sum(float(v) * float(v) for v in values))
        if norm == 0.0:
            raise RuntimeError(f"Google embeddings returned a zero-length vector from {self._model}")
        if abs(norm - 1.0) <= 1e-6:
            return [float(v) for v in values]
        return [float(v) / norm for v in values]


def get_embed_provider(model_spec: str) -> EmbedProvider:
    """Factory: parse VECTR_EMBED_MODEL and return the right provider."""
    if model_spec.startswith("voyage"):
        return VoyageEmbedProvider(model_spec)
    if model_spec.startswith("openai/"):
        return OpenAIEmbedProvider(model_spec.split("/", 1)[1])
    if model_spec.startswith("google/"):
        return GoogleEmbedProvider(
            model_spec.split("/", 1)[1],
            output_dimensionality=_EMBEDDING_OUTPUT_DIMENSIONALITY,
            api_key_env=_EMBEDDING_API_KEY_ENV,
        )
    return LocalEmbedProvider(model_spec)
