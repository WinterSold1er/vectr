"""Shared HTTP transport for remote embedding providers (UPG-EMBED-TRANSPORT-BASE).

`RemoteEmbedProvider` owns everything that is identical for every embeddings API
vectr calls — batching at the API's per-request cap, pacing between batches
against the key's character budget, one retry chain (transport errors, throttling,
5xx) with exponential backoff capped at 60s and a clamped `Retry-After`, API-key
redaction, "the body was not JSON" reporting, re-keying a reply onto the input
order, and L2 normalization — so a provider adapter only has to describe its own
wire format: endpoint, headers, payload, reply parsing, and what counts as
throttling on that API.

vectr's embedding settings are read here too (`embedding:` in
<workspace>/.vectr/config.yaml and ~/.vectr/config.yaml, ~/.vectr/secrets.json,
the VECTR_EMBED_* environment variables), so no provider re-implements them.
`agent/indexer/_types.py` re-exports these helpers for the callers that already
import them from there.
"""
from __future__ import annotations

import time
from typing import Any

import httpx


# ---------------------------------------------------------------------------
# embedding settings resolved here (installed 1.12.0 has no config overlay)
# ---------------------------------------------------------------------------

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
    `parameters.dimension`, an OpenAI-compatible server's `dimensions`)."""
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


def _resolve_api_key(env_name: str, secrets_key: str, provider: str, *, required: bool = True) -> str:
    """Key for an embeddings API: `env_name` first, then the `secrets_key` entry of
    ~/.vectr/secrets.json — GEMINI_API_KEY / "gemini_api_key" for Google,
    DASHSCOPE_API_KEY / "dashscope_api_key" for DashScope.

    Only the channel is ever named — the key value is never logged and never
    quoted in an error message. A secrets file that exists but is unreadable or
    malformed raises instead of silently falling back to a keyless request, and a
    group/other-readable file is warned about (it should be mode 600). With
    `required=False` (a keyless localhost gateway) the answer is simply "".
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

    if not required:
        return ""
    raise RuntimeError(
        f"{provider} embeddings need an API key: set {env_name}, or store "
        f'"{secrets_key}" in {path} (mode 600)'
    )


def _embed_chars_per_second() -> float:
    """Embedding request budget in characters per second, used to pace batch
    requests against the key's tokens-per-minute quota.

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


def _embed_batch_limit_override() -> int | None:
    """Texts-per-request override from VECTR_EMBED_BATCH_LIMIT, then
    embedding.batch_limit. These APIs cap the input array differently — OpenAI
    takes 2048 entries, many vLLM / Ollama / Jina builds 256, DashScope 10 or 20
    per model — so a narrower cap is a setting, not a code change (D3)."""
    import os

    raw = os.environ.get("VECTR_EMBED_BATCH_LIMIT") or _embed_config_for(None).get("batch_limit")
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"vectr: invalid embedding batch limit {raw!r}") from exc
    if value <= 0:
        raise RuntimeError(f"vectr: embedding batch limit must be positive, got {value}")
    return value


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


# ---------------------------------------------------------------------------
# the transport every remote embeddings call shares
# ---------------------------------------------------------------------------

class RemoteEmbedProvider:
    """Base for HTTP embedding providers: transport here, wire format in the adapter.

    An adapter implements `_endpoint`, `_build_payload` and `_parse_embeddings`,
    and overrides `_headers`, `_is_throttled`, `_batch_limit_for` or the provider
    constants when its API differs. Nothing about a model, a dimension, an API key
    name or an endpoint origin is invented here: each comes from the `embedding:`
    config block or the VECTR_EMBED_* environment, with the adapter's constants as
    the fallback.
    """

    _DEFAULT_BATCH_LIMIT = 100      # texts per request when the API names no other cap
    _MAX_ATTEMPTS = 8               # quota windows last about a minute; a shorter chain just fails
    _TIMEOUT_SECONDS = 30.0
    _MAX_RETRY_SECONDS = 60.0       # also the clamp on a server's own Retry-After (D1)
    _MAX_REPORTED_BODY = 200        # how much of an error body is quoted
    _MAX_REPORTED_DETAIL = 300

    _PROVIDER_LABEL = ""            # "Google" — transport failures and normalization
    _ERROR_PREFIX = ""              # "Google batchEmbedContents" — HTTP and reply failures
    _SPEC_PREFIX = ""               # the `embedding.model` prefix that selects this provider
    _SPEC_EXAMPLE = ""              # quoted by the "must name a model" error
    _DEFAULT_BASE_URL = ""          # origin when neither the env nor the config names one
    _DEFAULT_KEY_ENV = ""           # env var tried when `embedding.api_key_env` is unset
    _SECRETS_KEY = ""               # key name inside ~/.vectr/secrets.json
    _API_KEY_OPTIONAL = False       # a localhost gateway (Ollama, vLLM) may need no key
    _ERROR_FIELDS = ("code", "message", "type", "request_id")

    def __init__(
        self,
        model_name: str,
        output_dimensionality: int | None = None,
        api_key_env: str | None = None,
        base_url: str | None = None,
    ) -> None:
        if not (self._PROVIDER_LABEL and self._ERROR_PREFIX and self._SPEC_PREFIX):
            # A half-declared adapter would produce messages like
            # " embeddings request for … failed" with the provider name missing.
            raise TypeError(
                f"{type(self).__name__} must declare _PROVIDER_LABEL, _ERROR_PREFIX and "
                f"_SPEC_PREFIX: RemoteEmbedProvider is the shared transport, not a provider"
            )
        if not model_name:
            raise ValueError(
                f"a {self._SPEC_PREFIX}/ embedding spec must name a model, "
                f"e.g. {self._SPEC_EXAMPLE}"
            )
        self._model = model_name
        self._dim = output_dimensionality
        cfg = _embed_config_for(None)
        self._key_env = str(api_key_env or cfg.get("api_key_env") or self._DEFAULT_KEY_ENV)
        self._api_key = _resolve_api_key(
            self._key_env,
            self._SECRETS_KEY,
            self._PROVIDER_LABEL,
            required=not self._API_KEY_OPTIONAL,
        )
        self._base_url = self._resolve_base_url(base_url or cfg.get("base_url"))
        self._batch_limit = self._batch_limit_for(self._model)

    # -- EmbedProvider protocol ------------------------------------------------

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed document / index-side text."""
        return self._embed(texts, "document")

    def embed_query(self, texts: list[str]) -> list[list[float]]:
        """Embed search-query text. Asymmetric APIs map the two kinds onto their own
        request fields; symmetric ones ignore the distinction."""
        return self._embed(texts, "query")

    def _embed(self, texts: list[str], kind: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        budget = _embed_chars_per_second()
        for start in range(0, len(texts), self._batch_limit):
            batch = texts[start:start + self._batch_limit]
            vectors.extend(self._embed_batch(batch, kind))
            # Hold the request rate under the key's quota: pause for as long as this
            # batch's payload would take at the configured chars/second. Only a
            # batch with a follower is paced — the last one has nothing to protect.
            if start + self._batch_limit < len(texts):
                time.sleep(min(self._MAX_RETRY_SECONDS, sum(len(t) for t in batch) / budget))
        return vectors

    def _embed_batch(self, texts: list[str], kind: str) -> list[list[float]]:
        """One request for at most `_batch_limit` texts, retried until the attempts
        are spent. The loop always returns or raises on the final attempt."""
        payload = self._build_payload(texts, kind)
        url = self._endpoint()
        headers = self._headers()
        for attempt in range(1, self._MAX_ATTEMPTS + 1):
            try:
                response = httpx.post(
                    url, json=payload, headers=headers, timeout=self._TIMEOUT_SECONDS
                )
            except httpx.HTTPError as exc:
                # Connection-level failures (TLS handshake resets, read timeouts)
                # are transient: retry them with the same capped backoff as a 429
                # instead of aborting the whole indexing pass.
                if attempt < self._MAX_ATTEMPTS:
                    time.sleep(self._backoff(attempt))
                    continue
                raise RuntimeError(
                    f"{self._PROVIDER_LABEL} embeddings request for {self._model} failed: "
                    f"{self._redact(str(exc))}"
                ) from exc

            if (self._is_throttled(response) or response.status_code >= 500) and attempt < self._MAX_ATTEMPTS:
                # Throttling and upstream 5xx are both transient: same capped backoff.
                time.sleep(self._retry_delay(response, attempt))
                continue

            if response.status_code != 200:
                raise self._http_error(response, attempt)

            try:
                data = response.json()
            except ValueError as exc:
                # HTTP 200 claims the call succeeded, so replaying the same payload
                # would only hide a broken endpoint: report what actually came back
                # (D2), with the key redacted out of the body.
                raise RuntimeError(
                    f"{self._ERROR_PREFIX} for {self._model} returned a response body "
                    f"that is not JSON: {self._redact(response.text)[:self._MAX_REPORTED_BODY]}"
                ) from exc

            return self._parse_embeddings(data, len(texts))

    # -- adapter hooks ---------------------------------------------------------

    def _endpoint(self) -> str:
        """Absolute URL this provider POSTs to."""
        raise NotImplementedError

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _build_payload(self, texts: list[str], kind: str) -> dict[str, Any]:
        """Request body for `texts`. `kind` is "document" or "query"; an asymmetric
        API maps it onto its own field, a symmetric one leaves it unused."""
        payload: dict[str, Any] = {"model": self._model, "input": list(texts)}
        if self._dim is not None:
            payload["dimensions"] = self._dim
        return payload

    def _parse_embeddings(self, data: dict[str, Any], expected_count: int) -> list[list[float]]:
        """Reply body -> `expected_count` unit-length vectors, in input order."""
        raise NotImplementedError

    def _is_throttled(self, response) -> bool:
        """What counts as "you are over quota" on this API. Overridable because some
        report it in the body rather than the status."""
        return response.status_code == 429

    @classmethod
    def _batch_limit_for(cls, model: str) -> int:
        """Texts per request for the model the spec names."""
        return cls._with_batch_limit_override(cls._DEFAULT_BATCH_LIMIT)

    @classmethod
    def _with_batch_limit_override(cls, api_cap: int) -> int:
        """Narrow the API's cap by the configured one; never widen past the API."""
        override = _embed_batch_limit_override()
        return api_cap if override is None else max(1, min(api_cap, override))

    # -- shared plumbing -------------------------------------------------------

    def _resolve_base_url(self, configured: object) -> str:
        """Origin for this provider's calls: VECTR_EMBED_BASE_URL env, then
        embedding.base_url (or the value the caller passed), then the provider's own
        default endpoint.

        A VPC endpoint or an in-house gateway is a deployment choice, so the origin
        is configurable rather than assumed; a value that is not an http(s) origin
        raises here instead of surfacing as an opaque transport error on the first
        call (D5)."""
        import os

        raw = os.environ.get("VECTR_EMBED_BASE_URL") or configured or self._DEFAULT_BASE_URL
        base = str(raw).strip().rstrip("/")
        if not base.startswith(("http://", "https://")):
            raise RuntimeError(
                f"{self._PROVIDER_LABEL} embeddings: base_url {base!r} must be an http(s) URL"
            )
        return base

    @classmethod
    def _backoff(cls, attempt: int) -> float:
        """Exponential backoff 1s, 2s, 4s … capped at 60s."""
        return min(cls._MAX_RETRY_SECONDS, 2.0 ** (attempt - 1))

    def _retry_delay(self, response, attempt: int) -> float:
        """Retry-After when the endpoint sends one, otherwise the exponential backoff.

        The header is clamped to the same 60s cap the backoff has: one malformed or
        hostile 429 must not park indexing for an hour (D1). A missing, unparseable
        or non-finite value falls back to the backoff.
        """
        import math

        try:
            delay = float(response.headers.get("retry-after"))
        except (TypeError, ValueError):
            return self._backoff(attempt)
        if not math.isfinite(delay):
            return self._backoff(attempt)
        return max(0.0, min(self._MAX_RETRY_SECONDS, delay))

    def _http_error(self, response, attempt: int) -> RuntimeError:
        """Status plus whatever the provider named, with the key redacted out of the
        free-form body fallback: an upstream that echoes the request it was given
        would otherwise put the credential into a log."""
        detail = self._redact(self._error_detail(response))[:self._MAX_REPORTED_DETAIL]
        return RuntimeError(
            f"{self._ERROR_PREFIX} failed for {self._model}: "
            f"HTTP {response.status_code} {response.reason_phrase} "
            f"(attempt {attempt}/{self._MAX_ATTEMPTS}): {detail}{self._auth_hint(response)}"
        )

    def _error_detail(self, response) -> str:
        """The fields the API named in its error envelope, else its body."""
        error = self._error_fields(response)
        parts = [f"{name}={error[name]}" for name in self._ERROR_FIELDS if error.get(name)]
        return ", ".join(parts) or response.text[:self._MAX_REPORTED_BODY] or response.reason_phrase

    @staticmethod
    def _error_fields(response) -> dict:
        """The error envelope these APIs share: code / message / request_id at the top
        level on a native endpoint, the same fields nested under "error" on an
        OpenAI-compatible one. A body that is not JSON — an in-house gateway's error
        page — yields no fields instead of a second exception stacked on top of the
        first."""
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

    def _auth_hint(self, response) -> str:
        """401 / 403 say the credential was refused: name the channel, never the key."""
        if response.status_code in (401, 403):
            return (
                f" — the key resolved from {self._key_env} or "
                f'~/.vectr/secrets.json "{self._SECRETS_KEY}" was rejected'
            )
        return ""

    def _redact(self, text: str) -> str:
        """The API key never reaches an error message or a log line, whatever the
        upstream put in its own body."""
        if not text or not self._api_key:
            return text
        return text.replace(self._api_key, "[redacted]")

    def _normalize(self, values: list[float]) -> list[float]:
        return _normalize_vector(values, self._PROVIDER_LABEL, self._model)

    def _count_mismatch(self, expected: int, got: object) -> RuntimeError:
        return RuntimeError(
            f"{self._ERROR_PREFIX} returned {got} embeddings for {expected} texts ({self._model})"
        )

    def _malformed_entry(self) -> RuntimeError:
        return RuntimeError(
            f"{self._ERROR_PREFIX} returned a malformed embedding entry for {self._model}"
        )

    def _values_of(self, item: object, values_key: str) -> list[float]:
        values = item.get(values_key) if isinstance(item, dict) else None
        if not isinstance(values, list) or not values:
            raise self._malformed_entry()
        return values

    def _parse_vector_list(self, items: object, values_key: str, expected_count: int) -> list[list[float]]:
        """Vectors in array order, for an API that promises the order it was given."""
        if not isinstance(items, list) or len(items) != expected_count:
            got = len(items) if isinstance(items, list) else "none"
            raise self._count_mismatch(expected_count, got)
        return [self._normalize(self._values_of(item, values_key)) for item in items]

    def _place_by_index(
        self, items: object, *, values_key: str, index_key: str, expected_count: int
    ) -> list[list[float]]:
        """Vectors re-keyed onto the input order by the position the API names.

        The array order is not promised — Google's `embeddings`, DashScope's
        `text_index`, an OpenAI server's `index` — so a reply that carries a
        position is re-keyed rather than trusted, and a position outside the range
        or claimed twice is an error rather than a silent shuffle.
        """
        if not isinstance(items, list) or len(items) != expected_count:
            got = len(items) if isinstance(items, list) else "none"
            raise self._count_mismatch(expected_count, got)
        placed: list[list[float] | None] = [None] * expected_count
        for position, item in enumerate(items):
            values = self._values_of(item, values_key)
            claimed = item.get(index_key, position) if isinstance(item, dict) else position
            try:
                slot = int(claimed)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"{self._ERROR_PREFIX} returned a malformed {index_key} for {self._model}"
                ) from exc
            if not 0 <= slot < expected_count or placed[slot] is not None:
                raise RuntimeError(
                    f"{self._ERROR_PREFIX} returned {index_key} {slot} outside the "
                    f"{expected_count} texts it was given ({self._model})"
                )
            placed[slot] = self._normalize(values)
        missing = sum(1 for vector in placed if vector is None)
        if missing:
            raise RuntimeError(
                f"{self._ERROR_PREFIX} left {missing} of {expected_count} texts "
                f"without a vector ({self._model})"
            )
        return [vector for vector in placed if vector is not None]
