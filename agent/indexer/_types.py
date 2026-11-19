"""Chunk dataclass, EmbedProvider protocol, embed provider implementations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from agent.config import (
    EMBEDDING_DEFAULT_MODEL as _EMBEDDING_DEFAULT_MODEL,
    EMBEDDING_MAX_SEQ_LENGTH as _EMBEDDING_MAX_SEQ_LENGTH,
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



# --- embedding settings resolved here (installed 1.12.0 has no config overlay) ---
def _embed_config_for(workspace_root: str | None) -> dict:
    """`embedding:` settings from <workspace>/.vectr/config.yaml, then the global
    ~/.vectr/config.yaml; the workspace file wins key by key. An existing file
    that is unreadable or malformed raises instead of being ignored."""
    import os
    from pathlib import Path

    import yaml

    merged: dict = {}
    ws = workspace_root or os.environ.get("VECTR_WORKSPACE") or os.getcwd()
    for path in (Path(ws) / ".vectr" / "config.yaml", Path.home() / ".vectr" / "config.yaml"):
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise RuntimeError(f"vectr: {path} could not be read: {exc}") from exc
        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise RuntimeError(f"vectr: {path} is not valid YAML: {exc}") from exc
        section = data.get("embedding") if isinstance(data, dict) else None
        if isinstance(section, dict):
            merged = {**section, **merged}
    return merged


def resolve_embed_model(explicit_env: str | None, workspace_root: str | None, default: str) -> str:
    """Embed-model precedence: VECTR_EMBED_MODEL env > workspace .vectr/config.yaml
    embedding.model > ~/.vectr/config.yaml embedding.model > bundled default."""
    if explicit_env:
        return explicit_env
    model = _embed_config_for(workspace_root).get("model")
    return str(model) if model else default


def _output_dimensionality(cfg: dict) -> int | None:
    """Output vector size for an API embedding call: VECTR_EMBED_OUTPUT_DIM env,
    then embedding.output_dimensionality from the config files. Nothing is invented
    here — unset means "ask the API for its own default". Shared by every provider
    that takes an explicit dimension (Google's `outputDimensionality`, DashScope's
    `parameters.dimension`)."""
    import os

    raw = os.environ.get("VECTR_EMBED_OUTPUT_DIM") or cfg.get("output_dimensionality")
    if raw is None or raw == "":
        return None
    try:
        dim = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"vectr: invalid embedding output dimension {raw!r}") from exc
    if dim <= 0:
        raise RuntimeError(f"vectr: embedding output dimension must be positive, got {dim}")
    return dim


# Back-compat alias: the dimension helper was Google-only until DashScope needed
# the same layering. Existing callers keep working; new code uses
# `_output_dimensionality`.
_google_output_dimensionality = _output_dimensionality


def _resolve_api_key(env_name: str, secrets_key: str, provider: str) -> str:
    """Key for an embeddings API: `env_name` first, then the `secrets_key` entry of
    ~/.vectr/secrets.json — GEMINI_API_KEY / "gemini_api_key" for Google,
    DASHSCOPE_API_KEY / "dashscope_api_key" for DashScope.

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
        raise RuntimeError(f"{provider} embeddings: {path} could not be read: {exc}") from exc

    if raw is not None:
        try:
            stored_data = json.loads(raw)
        except ValueError as exc:
            raise RuntimeError(f"{provider} embeddings: {path} is not valid JSON: {exc}") from exc
        if not isinstance(stored_data, dict):
            raise RuntimeError(
                f"{provider} embeddings: {path} must contain a JSON object of "
                f'key names, e.g. {{"{secrets_key}": "…"}}'
            )
        stored = stored_data.get(secrets_key)
        if stored:
            mode = path.stat().st_mode & 0o777
            if mode & 0o077:
                logging.getLogger("vectr.embed").warning(
                    "vectr: %s is readable by group/other (mode %03o) — chmod 600 it", path, mode
                )
            return str(stored)

    raise RuntimeError(
        f"{provider} embeddings need an API key: set {env_name}, or store "
        f'"{secrets_key}" in {path} (mode 600)'
    )


def _embed_chars_per_second() -> float:
    """Embedding request budget in characters per second, used to pace
    batchEmbedContents against the key's tokens-per-minute quota.

    Measured 2026-11-10 against gemini-embedding-2: batches of ~14k tokens sent
    back to back return HTTP 429, the same batches spaced 40s apart return 200
    (about 72k chars/min on the free tier). Override with
    VECTR_EMBED_CHARS_PER_SECOND once the key is on a paid quota tier.
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


def _normalize_vector(values: list[float], provider: str, model: str) -> list[float]:
    """Unit-length guarantee for an embedding an API handed back.

    None of the remote providers can be trusted to return a unit vector: Google
    does at its full 3072-dim output but not for a truncated `outputDimensionality`,
    and DashScope makes no promise at all. Renormalize whenever the vector is not
    already unit length, so cosine scoring against vectr's other vectors stays
    comparable whatever the model or dimension.
    """
    import math

    try:
        norm = math.sqrt(sum(float(v) * float(v) for v in values))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{provider} embeddings returned non-numeric vector components from {model}"
        ) from exc
    if norm == 0.0:
        raise RuntimeError(
            f"{provider} embeddings returned a zero-length vector from {model}"
        )
    if abs(norm - 1.0) <= 1e-6:
        return [float(v) for v in values]
    return [float(v) / norm for v in values]


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
    _MAX_ATTEMPTS = 8           # 429 windows last about a minute; a short retry chain just fails
    _TIMEOUT_SECONDS = 30.0

    def __init__(
        self,
        model_name: str,
        output_dimensionality: int | None = None,
        api_key_env: str | None = None,
    ) -> None:
        if not model_name:
            raise ValueError(
                "a google/ embedding spec must name a model, e.g. google/gemini-embedding-2"
            )
        self._model = model_name
        self._dim = output_dimensionality
        self._api_key = _resolve_api_key(
            api_key_env or _embed_config_for(None).get("api_key_env") or "GEMINI_API_KEY",
            "gemini_api_key",
            "Google",
        )

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
            # Hold the request rate under the key's tokens-per-minute budget:
            # pause for as long as this batch's payload would take at that rate.
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
                # Connection-level failures (TLS handshake resets, read timeouts)
                # are transient: retry them with the same capped backoff as a 429
                # instead of aborting the whole indexing pass.
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

            if response.status_code >= 500 and attempt < self._MAX_ATTEMPTS:
                # Upstream 5xx (500/502/503/504) are server-side and transient
                # like a 429: retry on the same capped backoff instead of
                # aborting the whole indexing pass on a single bad response.
                time.sleep(min(60.0, 2.0 ** (attempt - 1)))
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
        return _normalize_vector(values, "Google", self._model)


# ---------------------------------------------------------------------------
# DashScope (Alibaba Model Studio / 千问) embeddings
# ---------------------------------------------------------------------------

_DASHSCOPE_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com"
_DASHSCOPE_EMBED_PATH = "/api/v1/services/embeddings/text-embedding/text-embedding"


def _dashscope_base_url(configured: object) -> str:
    """Origin for the DashScope-native call: VECTR_EMBED_BASE_URL env, then
    embedding.base_url from the config files, then the public endpoint.

    A VPC endpoint or an in-house gateway is a deployment choice, so the origin is
    configurable rather than assumed; a value that is not an http(s) origin raises
    here instead of surfacing as an opaque transport error on the first call.
    """
    import os

    raw = os.environ.get("VECTR_EMBED_BASE_URL") or configured or _DASHSCOPE_DEFAULT_BASE_URL
    base = str(raw).strip().rstrip("/")
    if not base.startswith(("http://", "https://")):
        raise RuntimeError(
            f"DashScope embeddings: base_url {base!r} must be an http(s) URL"
        )
    return base


class DashScopeEmbedProvider:
    """Alibaba DashScope (Model Studio / 千问) text embeddings through the native
    text-embedding endpoint (no SDK: plain JSON over httpx, the dependency vectr
    already has).

    Asymmetric retrieval is what this endpoint is tuned for and what the two
    protocol methods exist for: indexed chunks go out as `text_type: "document"`,
    search queries as `text_type: "query"`. Nothing about the model is hardcoded —
    the model name comes from the `dashscope/<model>` spec, the vector size from
    `embedding.output_dimensionality` / VECTR_EMBED_OUTPUT_DIM, the key's env-var
    name from `embedding.api_key_env` and the endpoint origin from
    `embedding.base_url`. As with the other API providers there is no local torch
    thread pool to cap (UPG-EMBED-THREAD-CONTENTION).
    """

    _BATCH_LIMIT = 20            # qwen3.7 text embeddings take at most 20 texts per call
    _LEGACY_BATCH_LIMIT = 10     # text-embedding-v3/v4 are documented at 10 texts per call
    _MAX_ATTEMPTS = 8            # Throttling.RateQuota windows last about a minute; a shorter chain just fails
    _TIMEOUT_SECONDS = 30.0

    def __init__(
        self,
        model_name: str,
        output_dimensionality: int | None = None,
        api_key_env: str | None = None,
        base_url: str | None = None,
    ) -> None:
        if not model_name:
            raise ValueError(
                "a dashscope/ embedding spec must name a model, "
                "e.g. dashscope/qwen3.7-text-embedding-flash"
            )
        self._model = model_name
        self._dim = output_dimensionality
        self._batch_limit = self._batch_limit_for(model_name)
        cfg = _embed_config_for(None)
        self._base_url = _dashscope_base_url(base_url or cfg.get("base_url"))
        self._key_env = str(api_key_env or cfg.get("api_key_env") or "DASHSCOPE_API_KEY")
        self._api_key = _resolve_api_key(self._key_env, "dashscope_api_key", "DashScope")

    @classmethod
    def _batch_limit_for(cls, model_name: str) -> int:
        """Per-request text cap for the model the spec names.

        DashScope caps texts per call per model: text-embedding-v3/v4 are
        documented at 10, the qwen3.7 text-embedding models take 20. An oversized
        batch is rejected outright, so the cap is read off the model name rather
        than assumed for every model.
        """
        import re

        if re.search(r"v[34](?!\d)", model_name):
            return cls._LEGACY_BATCH_LIMIT
        return cls._BATCH_LIMIT

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "document")

    def embed_query(self, texts: list[str]) -> list[list[float]]:
        # DashScope's asymmetric text_type: chunks are embedded as "document"
        # (optimised to be matched against), search queries as "query". The API
        # only accepts it on the native endpoint, which is why the native call is
        # used instead of the OpenAI-compatible one.
        return self._embed(texts, "query")

    def _embed(self, texts: list[str], text_type: str) -> list[list[float]]:
        import time

        vectors: list[list[float]] = []
        budget = _embed_chars_per_second()
        for start in range(0, len(texts), self._batch_limit):
            batch = texts[start:start + self._batch_limit]
            vectors.extend(self._embed_batch(batch, text_type))
            if start + self._batch_limit < len(texts):
                # Pace the batches against the key's quota the same way the Google
                # provider does: chars/sec, not a fixed sleep.
                time.sleep(min(60.0, sum(len(t) for t in batch) / budget))
        return vectors

    def _embed_batch(self, texts: list[str], text_type: str) -> list[list[float]]:
        import time

        import httpx

        parameters: dict = {"text_type": text_type}
        if self._dim is not None:
            parameters["dimension"] = self._dim
        payload = {
            "model": self._model,
            "input": {"texts": list(texts)},
            "parameters": parameters,
        }
        url = f"{self._base_url}{_DASHSCOPE_EMBED_PATH}"
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

        for attempt in range(1, self._MAX_ATTEMPTS + 1):
            try:
                response = httpx.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self._TIMEOUT_SECONDS,
                )
            except httpx.HTTPError as exc:
                # Connection-level failures (TLS handshake resets, read timeouts)
                # are transient: retry them with the same capped backoff as a 429
                # instead of aborting the whole indexing pass.
                if attempt < self._MAX_ATTEMPTS:
                    time.sleep(min(60.0, 2.0 ** (attempt - 1)))
                    continue
                raise RuntimeError(
                    f"DashScope embeddings request for {self._model} failed: "
                    f"{self._redact(str(exc))}"
                ) from exc

            if response.status_code != 200:
                error = self._error_fields(response)
                code = str(error.get("code") or "")
                throttled = response.status_code == 429 or code.lower().startswith("throttling")
                if throttled and attempt < self._MAX_ATTEMPTS:
                    # HTTP 429 and a body-level Throttling.* code are the same
                    # event seen through two endpoints; both back off.
                    time.sleep(self._retry_delay(response, attempt))
                    continue
                if response.status_code >= 500 and attempt < self._MAX_ATTEMPTS:
                    time.sleep(self._retry_delay(response, attempt))
                    continue
                raise RuntimeError(self._error_message(response, error, attempt))

            try:
                body = response.json()
            except ValueError as exc:
                # HTTP 200 claims the call succeeded, so replaying the same payload
                # would only hide a broken endpoint: report what actually came back.
                raise RuntimeError(
                    f"DashScope text-embedding for {self._model} returned a response "
                    f"body that is not JSON: {self._redact(response.text)[:200]}"
                ) from exc

            output = body.get("output") if isinstance(body, dict) else None
            embeddings = output.get("embeddings") if isinstance(output, dict) else None
            if not isinstance(embeddings, list) or len(embeddings) != len(texts):
                got = len(embeddings) if isinstance(embeddings, list) else "none"
                raise RuntimeError(
                    f"DashScope text-embedding returned {got} embeddings for "
                    f"{len(texts)} texts ({self._model})"
                )

            # text_index names the request position each vector belongs to. The
            # array is normally in order, but the API does not promise it, so the
            # reply is re-keyed onto the input order rather than trusted.
            placed: list[list[float] | None] = [None] * len(texts)
            for position, item in enumerate(embeddings):
                values = item.get("embedding") if isinstance(item, dict) else None
                if not isinstance(values, list) or not values:
                    raise RuntimeError(
                        f"DashScope text-embedding returned a malformed embedding "
                        f"entry for {self._model}"
                    )
                try:
                    slot = int(item.get("text_index", position))
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"DashScope text-embedding returned a malformed text_index "
                        f"for {self._model}"
                    ) from exc
                if not 0 <= slot < len(placed) or placed[slot] is not None:
                    raise RuntimeError(
                        f"DashScope text-embedding returned text_index {slot} outside "
                        f"the {len(texts)} texts it was given ({self._model})"
                    )
                placed[slot] = self._normalize(values)
            missing = sum(1 for vector in placed if vector is None)
            if missing:
                raise RuntimeError(
                    f"DashScope text-embedding left {missing} of {len(texts)} texts "
                    f"without a vector ({self._model})"
                )
            return [vector for vector in placed if vector is not None]

    @staticmethod
    def _error_fields(response) -> dict:
        """DashScope's error envelope: code / message / request_id at the top level
        on the native endpoint, the same fields nested under "error" on the
        OpenAI-compatible one. A body that is not JSON — an in-house gateway's
        error page — yields no fields instead of a second exception stacked on top
        of the first."""
        try:
            body = response.json()
        except ValueError:
            return {}
        if not isinstance(body, dict):
            return {}
        nested = body.get("error")
        if isinstance(nested, dict):
            return {**body, **nested}
        return body

    def _error_message(self, response, error: dict, attempt: int) -> str:
        """Status plus whatever DashScope named, with the key redacted from the
        free-form body fallback: an upstream that echoes the request it was given
        would otherwise put the credential into a log."""
        parts = [
            f"{name}={error[name]}"
            for name in ("code", "message", "request_id")
            if error.get(name)
        ]
        detail = ", ".join(parts) or self._redact(response.text)[:200] or response.reason_phrase
        hint = ""
        if response.status_code in (401, 403):
            hint = (
                f" — the key resolved from {self._key_env} or "
                f'~/.vectr/secrets.json "dashscope_api_key" was rejected'
            )
        return (
            f"DashScope text-embedding failed for {self._model}: "
            f"HTTP {response.status_code} {response.reason_phrase} "
            f"(attempt {attempt}/{self._MAX_ATTEMPTS}): {self._redact(detail)[:300]}{hint}"
        )

    @staticmethod
    def _retry_delay(response, attempt: int) -> float:
        """Retry-After when the endpoint sends one (its 429s do), otherwise
        exponential backoff 1s, 2s, 4s … capped at 60s — the shape DashScope's own
        throttling guidance recommends. A header is clamped to the same cap so one
        malformed response cannot park indexing for an hour."""
        try:
            delay = float(response.headers.get("retry-after"))
        except (TypeError, ValueError):
            return min(60.0, 2.0 ** (attempt - 1))
        return max(0.0, min(60.0, delay))

    def _redact(self, text: str) -> str:
        """The API key never reaches an error message or a log line, whatever the
        upstream put in its own body."""
        if not text:
            return text
        return text.replace(self._api_key, "[redacted]")

    def _normalize(self, values: list[float]) -> list[float]:
        return _normalize_vector(values, "DashScope", self._model)


def get_embed_provider(model_spec: str) -> EmbedProvider:
    """Factory: parse VECTR_EMBED_MODEL and return the right provider."""
    if model_spec.startswith("voyage"):
        return VoyageEmbedProvider(model_spec)
    if model_spec.startswith("openai/"):
        return OpenAIEmbedProvider(model_spec.split("/", 1)[1])
    if model_spec.startswith("google/"):
        cfg = _embed_config_for(None)
        return GoogleEmbedProvider(
            model_spec.split("/", 1)[1],
            output_dimensionality=_output_dimensionality(cfg),
            api_key_env=str(cfg.get("api_key_env") or "GEMINI_API_KEY"),
        )
    if model_spec.startswith("dashscope/"):
        cfg = _embed_config_for(None)
        return DashScopeEmbedProvider(
            model_spec.split("/", 1)[1],
            output_dimensionality=_output_dimensionality(cfg),
            api_key_env=str(cfg.get("api_key_env") or "DASHSCOPE_API_KEY"),
        )
    return LocalEmbedProvider(model_spec)
