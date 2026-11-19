"""Chunk dataclass, EmbedProvider protocol, embed provider implementations.

The HTTP transport every remote provider shares lives in `_remote_base.py`
(`RemoteEmbedProvider`): batching, pacing, the retry chain, Retry-After clamping,
key redaction and vector normalization. What is here is each API's wire format
plus `ProviderRegistry`, which turns an `embedding.model` spec into the provider
it names.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from agent.config import (
    EMBEDDING_DEFAULT_MODEL as _EMBEDDING_DEFAULT_MODEL,
    EMBEDDING_MAX_SEQ_LENGTH as _EMBEDDING_MAX_SEQ_LENGTH,
    EMBEDDING_THREAD_CAP as _EMBEDDING_THREAD_CAP,
    vectr_cache_root as _vectr_cache_root,
)

# The embedding settings helpers used to live in this module; they moved next to
# the transport that consumes them. The names are re-exported here because
# `agent/indexer/__init__.py`, `app/service.py` and the tests import them from
# `agent.indexer._types`.
from agent.indexer._remote_base import (
    RemoteEmbedProvider,
    _embed_chars_per_second,
    _embed_config_for,
    _normalize_vector,
    _output_dimensionality,
    _resolve_api_key,
    resolve_embed_model,
)

# Back-compat alias: the dimension helper was Google-only until DashScope needed
# the same layering. Existing callers keep working; new code uses
# `_output_dimensionality`.
_google_output_dimensionality = _output_dimensionality


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


# ---------------------------------------------------------------------------
# Google Generative Language embeddings (batchEmbedContents)
# ---------------------------------------------------------------------------

class GoogleEmbedProvider(RemoteEmbedProvider):
    """Google Generative Language embeddings via the batchEmbedContents REST
    endpoint (no SDK: plain JSON over httpx, the dependency vectr already has).

    Nothing about the model is hardcoded here: the model name comes from the
    `google/<model>` spec, the output size from `embedding.output_dimensionality`
    and the key's env-var name from `embedding.api_key_env`. As with the other API
    providers there is no local torch thread pool to cap
    (UPG-EMBED-THREAD-CONTENTION).
    """

    _PROVIDER_LABEL = "Google"
    _ERROR_PREFIX = "Google batchEmbedContents"
    _SPEC_PREFIX = "google"
    _SPEC_EXAMPLE = "google/gemini-embedding-2"
    _DEFAULT_BATCH_LIMIT = 100      # batchEmbedContents accepts at most 100 requests
    _DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    _DEFAULT_KEY_ENV = "GEMINI_API_KEY"
    _SECRETS_KEY = "gemini_api_key"

    # Google's asymmetric retrieval task types, keyed by the protocol's kinds:
    # indexed chunks are RETRIEVAL_DOCUMENT, search queries RETRIEVAL_QUERY.
    _TASK_TYPES = {"document": "RETRIEVAL_DOCUMENT", "query": "RETRIEVAL_QUERY"}

    def _endpoint(self) -> str:
        # The key rides in the `x-goog-api-key` header, never in the URL.
        return f"{self._base_url}/models/{self._model}:batchEmbedContents"

    def _headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self._api_key}

    def _build_payload(self, texts: list[str], kind: str) -> dict[str, Any]:
        requests: list[dict] = []
        for text in texts:
            request: dict = {
                "model": f"models/{self._model}",
                "content": {"parts": [{"text": text}]},
                "taskType": self._TASK_TYPES[kind],
            }
            if self._dim is not None:
                request["outputDimensionality"] = self._dim
            requests.append(request)
        return {"requests": requests}

    def _parse_embeddings(self, data: dict[str, Any], expected_count: int) -> list[list[float]]:
        # A gateway can answer 200 with valid JSON that is not an object (a bare
        # list, a string); the guard routes that case into the shared
        # "none embeddings for N texts" error instead of a bare AttributeError.
        return self._parse_vector_list(
            data.get("embeddings") if isinstance(data, dict) else None,
            "values",
            expected_count,
        )


# ---------------------------------------------------------------------------
# DashScope (Alibaba Model Studio / 千问) embeddings
# ---------------------------------------------------------------------------

class DashScopeEmbedProvider(RemoteEmbedProvider):
    """Alibaba DashScope (Model Studio / 千问) text embeddings through the native
    text-embedding endpoint (no SDK: plain JSON over httpx, the dependency vectr
    already has).

    Asymmetric retrieval is what this endpoint is tuned for and what the two
    protocol methods exist for: indexed chunks go out as `text_type: "document"`,
    search queries as `text_type: "query"`. The API only accepts the field on the
    native endpoint, which is why the native call is used rather than the
    OpenAI-compatible one. Nothing about the model is hardcoded — the model name
    comes from the `dashscope/<model>` spec, the vector size from
    `embedding.output_dimensionality` / VECTR_EMBED_OUTPUT_DIM, the key's env-var
    name from `embedding.api_key_env` and the endpoint origin from
    `embedding.base_url` (D5).
    """

    _PROVIDER_LABEL = "DashScope"
    _ERROR_PREFIX = "DashScope text-embedding"
    _SPEC_PREFIX = "dashscope"
    _SPEC_EXAMPLE = "dashscope/qwen3.7-text-embedding-flash"
    _DEFAULT_BATCH_LIMIT = 20       # qwen3.7 text embeddings take at most 20 texts per call
    _LEGACY_BATCH_LIMIT = 10        # text-embedding-v3/v4 are documented at 10 texts per call
    _DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com"
    _EMBED_PATH = "/api/v1/services/embeddings/text-embedding/text-embedding"
    _DEFAULT_KEY_ENV = "DASHSCOPE_API_KEY"
    _SECRETS_KEY = "dashscope_api_key"

    def _endpoint(self) -> str:
        return f"{self._base_url}{self._EMBED_PATH}"

    def _build_payload(self, texts: list[str], kind: str) -> dict[str, Any]:
        parameters: dict[str, Any] = {"text_type": kind}
        if self._dim is not None:
            parameters["dimension"] = self._dim
        return {
            "model": self._model,
            "input": {"texts": list(texts)},
            "parameters": parameters,
        }

    @classmethod
    def _batch_limit_for(cls, model_name: str) -> int:
        """Per-request text cap for the model the spec names.

        DashScope caps texts per call per model: text-embedding-v3/v4 are
        documented at 10, the qwen3.7 text-embedding models take 20. An oversized
        batch is rejected outright, so the cap is read off the model name rather
        than assumed for every model.
        """
        import re

        cap = cls._LEGACY_BATCH_LIMIT if re.search(r"v[34](?!\d)", model_name) else cls._DEFAULT_BATCH_LIMIT
        return cls._with_batch_limit_override(cap)

    def _is_throttled(self, response) -> bool:
        # HTTP 429 and a body-level Throttling.* code are the same event seen
        # through two endpoints; both back off.
        code = str(self._error_fields(response).get("code") or "")
        return response.status_code == 429 or code.lower().startswith("throttling")

    def _parse_embeddings(self, data: dict[str, Any], expected_count: int) -> list[list[float]]:
        output = data.get("output") if isinstance(data, dict) else None
        # text_index names the request position each vector belongs to. The array is
        # normally in order, but the API does not promise it, so the reply is
        # re-keyed onto the input order rather than trusted.
        return self._place_by_index(
            output.get("embeddings") if isinstance(output, dict) else None,
            values_key="embedding",
            index_key="text_index",
            expected_count=expected_count,
        )


# ---------------------------------------------------------------------------
# OpenAI-compatible /v1/embeddings: one adapter for every gateway
# ---------------------------------------------------------------------------

class OpenAICompatibleProvider(RemoteEmbedProvider):
    """`POST {base_url}/embeddings` — the request and reply shape every
    OpenAI-compatible server speaks: OpenAI, SiliconFlow, Jina, Ollama, vLLM,
    LiteLLM, an in-house gateway.

    Onboarding a new server is a config change, not a code change (D3, D5):
    `embedding.model: openai-compatible/<model>`, `embedding.base_url` pointing at
    it, `embedding.api_key_env` naming the key's channel, and — for a server that
    caps the input array below OpenAI's 2048 — `embedding.batch_limit: 256`. The
    batching, pacing, retry chain, redaction and normalization all come from
    `RemoteEmbedProvider`; the only thing written here is the JSON shape.

    `OpenAIEmbedProvider` (the official SDK client) stays behind the `openai/`
    prefix for callers that already use it; this class is the dependency-free path
    that also reaches servers no SDK knows about.
    """

    _PROVIDER_LABEL = "OpenAI-compatible"
    _ERROR_PREFIX = "OpenAI-compatible embeddings"
    _SPEC_PREFIX = "openai-compatible"
    _SPEC_EXAMPLE = "openai-compatible/BAAI/bge-m3"
    _DEFAULT_BATCH_LIMIT = 2048     # OpenAI's own per-request input cap
    _DEFAULT_BASE_URL = "https://api.openai.com/v1"
    _DEFAULT_KEY_ENV = "OPENAI_API_KEY"
    _SECRETS_KEY = "openai_api_key"

    def _endpoint(self) -> str:
        return f"{self._base_url}/embeddings"

    def _parse_embeddings(self, data: dict[str, Any], expected_count: int) -> list[list[float]]:
        # `data[i].index` names the request position each vector belongs to; a
        # server is free to answer out of order, so the reply is re-keyed.
        return self._place_by_index(
            data.get("data") if isinstance(data, dict) else None,
            values_key="embedding",
            index_key="index",
            expected_count=expected_count,
        )


class SiliconFlowEmbedProvider(OpenAICompatibleProvider):
    """SiliconFlow (硅基流动): its origin and key channel, same wire format."""

    _PROVIDER_LABEL = "SiliconFlow"
    _ERROR_PREFIX = "SiliconFlow embeddings"
    _SPEC_PREFIX = "siliconflow"
    _SPEC_EXAMPLE = "siliconflow/BAAI/bge-m3"
    _DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
    _DEFAULT_KEY_ENV = "SILICONFLOW_API_KEY"
    _SECRETS_KEY = "siliconflow_api_key"


class JinaEmbedProvider(OpenAICompatibleProvider):
    """Jina Embeddings through its OpenAI-compatible endpoint.

    `jina/` is the hosted API. The HuggingFace repos of the same models live under
    the `jinaai/` namespace, which is a model id and still loads locally.
    """

    _PROVIDER_LABEL = "Jina"
    _ERROR_PREFIX = "Jina embeddings"
    _SPEC_PREFIX = "jina"
    _SPEC_EXAMPLE = "jina/jina-embeddings-v3"
    _DEFAULT_BASE_URL = "https://api.jina.ai/v1"
    _DEFAULT_KEY_ENV = "JINA_API_KEY"
    _SECRETS_KEY = "jina_api_key"


class OllamaEmbedProvider(OpenAICompatibleProvider):
    """A local Ollama server: OpenAI-compatible endpoint, no key required."""

    _PROVIDER_LABEL = "Ollama"
    _ERROR_PREFIX = "Ollama embeddings"
    _SPEC_PREFIX = "ollama"
    _SPEC_EXAMPLE = "ollama/bge-m3"
    _DEFAULT_BASE_URL = "http://localhost:11434/v1"
    _DEFAULT_KEY_ENV = "OLLAMA_API_KEY"
    _SECRETS_KEY = "ollama_api_key"
    _API_KEY_OPTIONAL = True


class VllmEmbedProvider(OpenAICompatibleProvider):
    """A local vLLM server: OpenAI-compatible endpoint, keyless by default."""

    _PROVIDER_LABEL = "vLLM"
    _ERROR_PREFIX = "vLLM embeddings"
    _SPEC_PREFIX = "vllm"
    _SPEC_EXAMPLE = "vllm/BAAI/bge-m3"
    _DEFAULT_BASE_URL = "http://localhost:8000/v1"
    _DEFAULT_KEY_ENV = "VLLM_API_KEY"
    _SECRETS_KEY = "vllm_api_key"
    _API_KEY_OPTIONAL = True


# ---------------------------------------------------------------------------
# provider registry: an `embedding.model` spec -> the provider it names
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProviderEntry:
    """One spec prefix and how to build the provider it names.

    `factory` is a class *name* resolved in this module when a spec is used, not a
    class object captured here: callers and tests replace
    `agent.indexer._types.<Provider>` and expect the routing to honour that, which
    the if-elif chain this replaces did implicitly.
    """

    prefix: str
    factory: str
    model_name_only: bool = False   # the SDK-backed providers take a model name, nothing else
    default_api_key_env: str = ""
    example: str = ""


# Vendor names that are provider intent without being a provider vectr implements.
# Deliberately small: the HuggingFace namespace is full of vendor-named repos
# (`Qwen/Qwen3-Embedding-8B`, `nvidia/nv-embedqa-e5-v5`) that are model ids, so
# only names that are not HF namespaces qualify.
_UNIMPLEMENTED_VENDORS = frozenset(
    {
        "aliyun",
        "azureopenai",
        "bedrock",
        "chatgpt",
        "gemini",
        "generativelanguage",
        "openaicompat",
        "openaicompatible",
    }
)

# Namespaces that really do host model weights *and* share a provider's name, so
# they are model ids rather than a mistyped prefix: `jinaai/jina-embeddings-v3` is
# the local HuggingFace repo of the model `jina/jina-embeddings-v3` embeds through
# the hosted API. A spec that stays local is never rejected.
_MODEL_NAMESPACES = frozenset({"jinaai"})


class ProviderRegistry:
    """`embedding.model` spec -> provider, replacing the old if-elif chain.

    The grammar is `[prefix/]model`, and a prefix names a provider only when this
    registry knows it. The chain had two failure modes this closes:

    * D4 — anything unrecognized fell through to `LocalEmbedProvider`, so a typo
      (`googel/gemini-embedding-2`, `dashscpe/qwen…`) or an unimplemented vendor
      quietly started a HuggingFace download for what was meant to be an API call.
      A prefix that is near a registered one, or a known vendor name, now raises
      and lists the known prefixes.
    * A HuggingFace model id *contains* a slash (`ibm-granite/granite-embedding-
      english-r2`, `BAAI/bge-m3`), so a slash alone cannot mean "provider". An
      unknown prefix that is not a vendor name still selects the local model, as
      before, and `local/<model>` states that choice explicitly.
    """

    def __init__(self, entries: Sequence[ProviderEntry] | None = None) -> None:
        self._entries = {
            entry.prefix: entry for entry in (_default_entries() if entries is None else entries)
        }

    @property
    def prefixes(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def entry_for(self, prefix: str) -> ProviderEntry | None:
        return self._entries.get(str(prefix).lower())

    def create(self, model_spec: str) -> EmbedProvider:
        spec = str(model_spec).strip()
        prefix, slash, remainder = spec.partition("/")

        entry = self.entry_for(prefix)
        if entry is not None and slash:
            if not remainder:
                raise ValueError(
                    f"a {entry.prefix}/ embedding spec must name a model, e.g. {entry.example}"
                )
            return self._build(entry, remainder)

        if not slash:
            # No prefix: the legacy `voyage-code-2` form still means Voyage, and a
            # bare model name still means the local model.
            if prefix.lower().startswith("voyage"):
                return VoyageEmbedProvider(spec)
            return LocalEmbedProvider(spec)

        meant = self._vendor_meant(prefix)
        if meant is not None:
            suggestion = "" if meant == prefix.lower() else f" — did you mean {meant}/?"
            raise ValueError(
                f"unknown embedding provider prefix {prefix}/ in embedding.model "
                f"{model_spec!r}{suggestion}. Known prefixes: "
                f"{', '.join(f'{name}/' for name in self.prefixes)}. A HuggingFace "
                f'model id such as "BAAI/bge-m3" — or an explicit "local/BAAI/bge-m3" '
                f"— is the local model; a prefix that names a provider is rejected "
                f"rather than quietly downloaded."
            )

        # The leading segment is a HuggingFace namespace, not a provider.
        logging.getLogger("vectr.embed").info(
            "vectr: embedding.model %r matches no provider prefix; treating %r as a "
            "local model id (prefixes: %s)", spec, spec, ", ".join(self.prefixes)
        )
        return LocalEmbedProvider(spec)

    def _vendor_meant(self, prefix: str) -> str | None:
        """The provider this spec looks like it was reaching for: a vendor vectr
        does not implement, or a near-miss of a registered prefix. A namespace that
        really does host model weights is never treated as a near-miss."""
        import difflib

        lowered = prefix.lower()
        if lowered in _MODEL_NAMESPACES:
            return None
        if lowered in _UNIMPLEMENTED_VENDORS:
            return lowered
        near = difflib.get_close_matches(
            lowered, (*self.prefixes, *_UNIMPLEMENTED_VENDORS), n=1, cutoff=0.8
        )
        return near[0] if near else None

    @staticmethod
    def _build(entry: ProviderEntry, model_name: str) -> EmbedProvider:
        # Late binding: `entry.factory` is a name, so a provider class replaced in
        # this module (a test, or a downstream patch) is the one that gets built.
        provider_class = globals()[entry.factory]
        if entry.model_name_only:
            return provider_class(model_name)
        cfg = _embed_config_for(None)
        return provider_class(
            model_name,
            output_dimensionality=_output_dimensionality(cfg),
            api_key_env=str(cfg.get("api_key_env") or entry.default_api_key_env),
        )


def _default_entries() -> tuple[ProviderEntry, ...]:
    """The shipped registry. Each entry reads its defaults off the class it names
    — the same constants that class uses when built directly — so a prefix and its
    provider cannot drift apart.
    """

    def entry(prefix: str, provider_class: type, *, model_name_only: bool = False) -> ProviderEntry:
        return ProviderEntry(
            prefix=prefix,
            factory=provider_class.__name__,
            model_name_only=model_name_only,
            default_api_key_env=str(getattr(provider_class, "_DEFAULT_KEY_ENV", "")),
            example=str(getattr(provider_class, "_SPEC_EXAMPLE", "")) or f"{prefix}/<model>",
        )

    return (
        entry("dashscope", DashScopeEmbedProvider),
        entry("google", GoogleEmbedProvider),
        entry("jina", JinaEmbedProvider),
        entry("local", LocalEmbedProvider, model_name_only=True),
        entry("ollama", OllamaEmbedProvider),
        entry("openai", OpenAIEmbedProvider, model_name_only=True),
        entry("openai-compatible", OpenAICompatibleProvider),
        entry("siliconflow", SiliconFlowEmbedProvider),
        entry("vllm", VllmEmbedProvider),
        entry("voyage", VoyageEmbedProvider, model_name_only=True),
    )


PROVIDER_REGISTRY = ProviderRegistry()


def get_embed_provider(model_spec: str) -> EmbedProvider:
    """Factory: parse an `embedding.model` spec and return the provider it names.

    The routing itself lives in `ProviderRegistry`; this stays the import site
    `agent/indexer/_core.py` reaches through the package namespace so test-time
    patching of `agent.indexer.get_embed_provider` keeps working.
    """
    return PROVIDER_REGISTRY.create(model_spec)
