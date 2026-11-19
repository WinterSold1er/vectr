"""Contract tests for the shared embed-provider transport (UPG-EMBED-TRANSPORT-BASE).

`RemoteEmbedProvider` is where batching, pacing, the retry chain, Retry-After
clamping, key redaction, not-JSON reporting, reply re-keying and normalization now
live. These tests exercise that base through a minimal adapter that is neither
Google nor DashScope: the point is that a new provider inherits all of it instead
of re-implementing it, which is what the duplicated Google/DashScope transport code
used to mean. The OpenAI-compatible adapter and the provider registry are covered
here too; the Google- and DashScope-specific suites keep covering their own wire
formats.
"""
from __future__ import annotations

import math
import time

import httpx
import pytest

from agent.indexer import _types as types_mod
from agent.indexer._remote_base import RemoteEmbedProvider, _normalize_vector
from agent.indexer._types import (
    PROVIDER_REGISTRY,
    DashScopeEmbedProvider,
    GoogleEmbedProvider,
    OllamaEmbedProvider,
    OpenAICompatibleProvider,
    ProviderEntry,
    ProviderRegistry,
    SiliconFlowEmbedProvider,
    VllmEmbedProvider,
    get_embed_provider,
)

FAKE_KEY = "sk-contract-key-must-never-leak"
MODEL = "demo-embedding-1"
ENDPOINT = "https://contract.example/v1/embeddings"
ATTEMPTS = RemoteEmbedProvider._MAX_ATTEMPTS
BACKOFF = [min(60.0, 2.0 ** attempt) for attempt in range(ATTEMPTS - 1)]


def _unit(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in values))
    return [v / norm for v in values]


class _Response:
    """Stands in for an httpx.Response: status, headers, body text, and a json()
    that fails the way a real one does on a body that is not JSON."""

    def __init__(self, status_code=200, payload=None, text="", headers=None) -> None:
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}
        self.reason_phrase = "OK" if status_code == 200 else "Error"
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("body is not JSON")
        return self._payload


def _reply(vectors) -> _Response:
    """A well-formed OpenAI-shaped success body: data[].index / data[].embedding."""
    return _Response(
        payload={"data": [{"index": i, "embedding": list(v)} for i, v in enumerate(vectors)]}
    )


class _FakePost:
    """Stands in for httpx.post: records every call, replays a scripted reply, and
    raises the exception placed in a reply slot."""

    def __init__(self, replies=None, echo=False) -> None:
        self.calls: list[dict] = []
        self._replies = list(replies) if replies else []
        self._echo = echo

    def __call__(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        if self._echo:
            # The vector encodes the text it was asked about, so a reply that lands
            # on the wrong input position shows up as a failed ordering assertion.
            return _reply([[float(len(text) + 1), 1.0] for text in json["input"]])
        if not self._replies:
            raise AssertionError("the test ran out of scripted replies")
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class _MinimalProvider(RemoteEmbedProvider):
    """The smallest adapter that can exist: endpoint, payload shape, reply parsing.
    Everything these tests assert is the base doing the rest."""

    _PROVIDER_LABEL = "Contract"
    _ERROR_PREFIX = "Contract embeddings"
    _SPEC_PREFIX = "contract"
    _SPEC_EXAMPLE = "contract/demo-embedding-1"
    _DEFAULT_BATCH_LIMIT = 3
    _DEFAULT_BASE_URL = "https://contract.example/v1"
    _DEFAULT_KEY_ENV = "CONTRACT_API_KEY"
    _SECRETS_KEY = "contract_api_key"

    def _endpoint(self) -> str:
        return f"{self._base_url}/embeddings"

    def _build_payload(self, texts, kind):
        return {"model": self._model, "input": list(texts), "kind": kind}

    def _parse_embeddings(self, data, expected_count):
        return self._place_by_index(
            data.get("data") if isinstance(data, dict) else None,
            values_key="embedding",
            index_key="index",
            expected_count=expected_count,
        )


class _StandIn:
    """Records how the registry built a provider, so routing can be asserted
    without importing the provider's own dependencies."""

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch) -> None:
    """No real config, secrets file or VECTR_* setting may leak into these tests."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("VECTR_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("CONTRACT_API_KEY", FAKE_KEY)
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    for name in (
        "VECTR_EMBED_MODEL",
        "VECTR_EMBED_OUTPUT_DIM",
        "VECTR_EMBED_BASE_URL",
        "VECTR_EMBED_CHARS_PER_SECOND",
        "VECTR_EMBED_BATCH_LIMIT",
        "GEMINI_API_KEY",
        "DASHSCOPE_API_KEY",
        "SILICONFLOW_API_KEY",
        "JINA_API_KEY",
        "OLLAMA_API_KEY",
        "VLLM_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    """Record every pause the transport asks for instead of waiting for it."""
    recorded: list[float] = []
    monkeypatch.setattr(time, "sleep", recorded.append)
    return recorded


def _provider(model: str = MODEL, **kwargs) -> _MinimalProvider:
    return _MinimalProvider(model, **kwargs)


def _expected(texts: list[str]) -> list[list[float]]:
    return [_unit([float(len(t) + 1), 1.0]) for t in texts]


class TestExtensionContract:
    def test_a_minimal_adapter_inherits_the_whole_transport(
        self, isolated, monkeypatch
    ) -> None:
        post = _FakePost(echo=True)
        monkeypatch.setattr(httpx, "post", post)
        provider = _provider()

        vectors = provider.embed(["a", "bb", "ccc"])

        assert post.calls[0]["url"] == ENDPOINT
        assert post.calls[0]["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"
        assert post.calls[0]["timeout"] == RemoteEmbedProvider._TIMEOUT_SECONDS
        assert vectors == _expected(["a", "bb", "ccc"])

    def test_the_base_is_not_a_provider_by_itself(self, isolated) -> None:
        class _Undeclared(RemoteEmbedProvider):
            _DEFAULT_BASE_URL = "https://example.invalid/v1"

        with pytest.raises(TypeError, match="must declare _PROVIDER_LABEL"):
            _Undeclared(MODEL)

    def test_an_empty_spec_names_the_prefix_it_expected(self, isolated) -> None:
        with pytest.raises(ValueError, match="must name a model"):
            _provider("")

    def test_a_named_provider_needs_a_key_and_says_where(self, isolated, monkeypatch) -> None:
        monkeypatch.delenv("CONTRACT_API_KEY")

        with pytest.raises(RuntimeError) as exc_info:
            _provider()

        message = str(exc_info.value)
        assert "CONTRACT_API_KEY" in message and "contract_api_key" in message
        assert FAKE_KEY not in message

    def test_a_keyless_gateway_needs_no_credential(self, isolated, monkeypatch) -> None:
        monkeypatch.delenv("CONTRACT_API_KEY")

        class _Keyless(_MinimalProvider):
            _API_KEY_OPTIONAL = True

        assert _Keyless(MODEL)._api_key == ""

    def test_base_url_layering_is_env_then_argument_then_default(
        self, isolated, monkeypatch
    ) -> None:
        assert _provider()._base_url == "https://contract.example/v1"
        assert _provider(base_url="http://127.0.0.1:9099/gateway/")._base_url == "http://127.0.0.1:9099/gateway"

        monkeypatch.setenv("VECTR_EMBED_BASE_URL", "https://env.example/v1")
        assert _provider(base_url="http://127.0.0.1:1/ignored")._base_url == "https://env.example/v1"

    def test_base_url_comes_from_the_workspace_config(self, isolated, monkeypatch, tmp_path) -> None:
        (tmp_path / ".vectr").mkdir()
        (tmp_path / ".vectr" / "config.yaml").write_text(
            "embedding:\n"
            "  base_url: http://127.0.0.1:9099/gateway\n"
            "  api_key_env: MY_CONTRACT_KEY\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("MY_CONTRACT_KEY", FAKE_KEY)

        provider = _provider()

        assert provider._endpoint() == "http://127.0.0.1:9099/gateway/embeddings"
        assert provider._key_env == "MY_CONTRACT_KEY"
        assert provider._api_key == FAKE_KEY

    def test_a_non_http_base_url_raises_instead_of_a_transport_error(
        self, isolated
    ) -> None:
        with pytest.raises(RuntimeError, match="must be an http\\(s\\) URL"):
            _provider(base_url="contract.internal")


class TestBatchingAndPacing:
    def test_texts_are_split_at_the_providers_batch_limit(
        self, isolated, monkeypatch, sleeps
    ) -> None:
        post = _FakePost(echo=True)
        monkeypatch.setattr(httpx, "post", post)
        texts = [f"t{i}" for i in range(7)]

        vectors = _provider().embed(texts)

        assert [len(call["json"]["input"]) for call in post.calls] == [3, 3, 1]
        assert post.calls[1]["json"]["input"][0] == "t3"
        assert vectors == _expected(texts), "output must stay in input order"

    def test_empty_input_makes_no_request(self, isolated, monkeypatch) -> None:
        post = _FakePost(echo=True)
        monkeypatch.setattr(httpx, "post", post)

        assert _provider().embed([]) == []
        assert post.calls == []

    def test_each_batch_is_paced_by_its_own_payload_size(
        self, isolated, monkeypatch, sleeps
    ) -> None:
        post = _FakePost(echo=True)
        monkeypatch.setattr(httpx, "post", post)
        texts = ["x" * 400] * 4  # first batch is 1200 chars at the 1200 chars/s default

        _provider().embed(texts)

        assert [len(call["json"]["input"]) for call in post.calls] == [3, 1]
        assert sleeps == [1.0]

    def test_a_batch_with_no_follower_is_not_paced(self, isolated, monkeypatch, sleeps) -> None:
        post = _FakePost(echo=True)
        monkeypatch.setattr(httpx, "post", post)

        _provider().embed(["x" * 400] * 3)

        assert len(post.calls) == 1
        assert sleeps == []

    def test_the_pause_is_capped_so_one_huge_batch_cannot_stall_indexing(
        self, isolated, monkeypatch, sleeps
    ) -> None:
        monkeypatch.setenv("VECTR_EMBED_CHARS_PER_SECOND", "50")
        post = _FakePost(echo=True)
        monkeypatch.setattr(httpx, "post", post)

        _provider().embed(["y" * 3000] * 4)  # 9000 chars would ask for 180s

        assert sleeps == [60.0]

    def test_query_embedding_uses_the_same_transport_with_its_own_kind(
        self, isolated, monkeypatch
    ) -> None:
        post = _FakePost(echo=True)
        monkeypatch.setattr(httpx, "post", post)

        _provider().embed(["for the index"])
        _provider().embed_query(["for the search"])

        assert [call["json"]["kind"] for call in post.calls] == ["document", "query"]


class TestRetryChain:
    def test_throttling_and_upstream_failures_share_one_backoff_chain(
        self, isolated, monkeypatch, sleeps
    ) -> None:
        post = _FakePost([_Response(429, text="slow down"), _Response(503, text="down"), _reply([[1.0, 0.0]])])
        monkeypatch.setattr(httpx, "post", post)

        vectors = _provider().embed(["text"])

        assert len(post.calls) == 3
        assert sleeps == [1.0, 2.0]
        assert vectors == [_unit([1.0, 0.0])]

    def test_throttling_that_never_lifts_gives_up_after_the_attempt_budget(
        self, isolated, monkeypatch, sleeps
    ) -> None:
        post = _FakePost([_Response(429, text="rate limited")] * ATTEMPTS)
        monkeypatch.setattr(httpx, "post", post)

        with pytest.raises(RuntimeError) as exc_info:
            _provider().embed(["text"])

        message = str(exc_info.value)
        assert len(post.calls) == ATTEMPTS
        assert sleeps == BACKOFF
        assert "429" in message and f"attempt {ATTEMPTS}/{ATTEMPTS}" in message

    def test_retry_after_is_honoured_within_the_cap(self, isolated, monkeypatch, sleeps) -> None:
        post = _FakePost([_Response(429, text="slow down", headers={"retry-after": "9"}), _reply([[1.0, 0.0]])])
        monkeypatch.setattr(httpx, "post", post)

        _provider().embed(["text"])

        assert sleeps == [9.0]

    @pytest.mark.parametrize("header", ["3600", "86400"])
    def test_retry_after_is_clamped_to_the_backoff_cap(self, isolated, monkeypatch, sleeps, header) -> None:
        # D1: a single malformed or hostile 429 must not park an indexing pass for an hour.
        post = _FakePost([_Response(429, headers={"retry-after": header}), _reply([[1.0, 0.0]])])
        monkeypatch.setattr(httpx, "post", post)

        _provider().embed(["text"])

        assert sleeps == [60.0]

    @pytest.mark.parametrize("header", ["not-a-number", "", "nan", "inf"])
    def test_an_unusable_retry_after_falls_back_to_the_backoff(
        self, isolated, monkeypatch, sleeps, header
    ) -> None:
        post = _FakePost([_Response(429, headers={"retry-after": header}), _reply([[1.0, 0.0]])])
        monkeypatch.setattr(httpx, "post", post)

        _provider().embed(["text"])

        assert sleeps == [1.0]

    def test_a_client_error_is_not_retried(self, isolated, monkeypatch, sleeps) -> None:
        post = _FakePost([_Response(400, text="input is empty")] * ATTEMPTS)
        monkeypatch.setattr(httpx, "post", post)

        with pytest.raises(RuntimeError, match="HTTP 400"):
            _provider().embed(["text"])

        assert len(post.calls) == 1
        assert sleeps == []

    def test_connection_failures_are_retried_then_succeed(
        self, isolated, monkeypatch, sleeps
    ) -> None:
        post = _FakePost([httpx.ConnectError("[Errno 111] Connection refused"), _reply([[1.0, 0.0]])])
        monkeypatch.setattr(httpx, "post", post)

        assert _provider().embed(["text"]) == [_unit([1.0, 0.0])]
        assert sleeps == [1.0]

    def test_an_exhausted_transport_failure_keeps_the_original_error(
        self, isolated, monkeypatch, sleeps
    ) -> None:
        post = _FakePost([httpx.ReadTimeout("gateway never answered")] * ATTEMPTS)
        monkeypatch.setattr(httpx, "post", post)

        with pytest.raises(RuntimeError) as exc_info:
            _provider().embed(["text"])

        message = str(exc_info.value)
        assert "gateway never answered" in message and MODEL in message
        assert isinstance(exc_info.value.__cause__, httpx.ReadTimeout)
        assert len(post.calls) == ATTEMPTS
        assert sleeps == BACKOFF


class TestErrorReporting:
    def test_an_error_body_that_echoes_the_request_is_scrubbed(self, isolated, monkeypatch) -> None:
        post = _FakePost([_Response(400, text=f"bad request: Bearer {FAKE_KEY}")])
        monkeypatch.setattr(httpx, "post", post)

        with pytest.raises(RuntimeError) as exc_info:
            _provider().embed(["text"])

        assert FAKE_KEY not in str(exc_info.value)
        assert "[redacted]" in str(exc_info.value)

    def test_a_rejected_key_names_the_channel_not_the_key(self, isolated, monkeypatch) -> None:
        post = _FakePost([_Response(401, payload={
            "error": {"message": f"invalid key {FAKE_KEY}", "type": "invalid_request_error", "code": "bad_key"}
        })])
        monkeypatch.setattr(httpx, "post", post)

        with pytest.raises(RuntimeError) as exc_info:
            _provider().embed(["text"])

        message = str(exc_info.value)
        assert FAKE_KEY not in message
        assert "bad_key" in message  # what the provider actually said, kept for the operator
        assert "CONTRACT_API_KEY" in message and "contract_api_key" in message  # where to look

    def test_the_fields_the_api_named_are_reported(self, isolated, monkeypatch, sleeps) -> None:
        post = _FakePost([_Response(500, payload={"code": "InternalError", "message": "boom", "request_id": "req-500"})] * ATTEMPTS)
        monkeypatch.setattr(httpx, "post", post)

        with pytest.raises(RuntimeError) as exc_info:
            _provider().embed(["text"])

        message = str(exc_info.value)
        assert "HTTP 500" in message
        assert "code=InternalError" in message and "request_id=req-500" in message
        assert len(post.calls) == ATTEMPTS

    def test_a_success_body_that_is_not_json_is_reported_once(self, isolated, monkeypatch) -> None:
        # D2: HTTP 200 claims success, so the body is what gets reported — replaying
        # the payload would only hide a broken endpoint.
        post = _FakePost([_Response(200, text="<html>the gateway said OK</html>")])
        monkeypatch.setattr(httpx, "post", post)

        with pytest.raises(RuntimeError, match="not JSON") as exc_info:
            _provider().embed(["text"])

        message = str(exc_info.value)
        assert MODEL in message and "the gateway said OK" in message
        assert isinstance(exc_info.value.__cause__, ValueError)
        assert len(post.calls) == 1, "a body that is not JSON must not be replayed"

    def test_the_key_is_never_used_as_a_search_pattern(self, isolated, monkeypatch) -> None:
        class _KeylessGateway(_MinimalProvider):
            _API_KEY_OPTIONAL = True

        monkeypatch.delenv("CONTRACT_API_KEY")
        post = _FakePost([_Response(400, text="model not found")])
        monkeypatch.setattr(httpx, "post", post)

        with pytest.raises(RuntimeError) as exc_info:
            _KeylessGateway(MODEL).embed(["text"])

        # An empty key must not turn every character of the body into "[redacted]".
        assert str(exc_info.value).count("[redacted]") == 0
        assert "model not found" in str(exc_info.value)


class TestReplyParsing:
    def test_vectors_are_re_keyed_from_the_reply_not_the_array_order(
        self, isolated, monkeypatch
    ) -> None:
        post = _FakePost([_Response(payload={"data": [
            {"index": 1, "embedding": [0.0, 1.0]},
            {"index": 0, "embedding": [3.0, 4.0]},
        ]})])
        monkeypatch.setattr(httpx, "post", post)

        vectors = _provider().embed(["first", "second"])

        assert vectors[0] == pytest.approx([0.6, 0.8])
        assert vectors[1] == pytest.approx([0.0, 1.0])

    @pytest.mark.parametrize(
        "body, expected",
        [
            ({"data": [{"index": 0, "embedding": [1.0, 0.0]}]}, "1 embeddings for 2 texts"),
            ({"unexpected": []}, "none embeddings for 2 texts"),
            ({"data": [{"index": 0, "embedding": []}, {"index": 1, "embedding": [1.0, 0.0]}]}, "malformed embedding entry"),
            ({"data": [{"index": 7, "embedding": [1.0, 0.0]}, {"index": 1, "embedding": [0.0, 1.0]}]}, "index 7 outside the 2 texts"),
            ({"data": [{"index": 0, "embedding": [1.0, 0.0]}, {"index": 0, "embedding": [0.0, 1.0]}]}, "index 0 outside the 2 texts"),
            ({"data": [{"index": "not-a-number", "embedding": [1.0, 0.0]}, {"index": 1, "embedding": [0.0, 1.0]}]}, "malformed index"),
        ],
    )
    def test_a_reply_that_does_not_match_the_request_is_an_error(
        self, isolated, monkeypatch, body, expected
    ) -> None:
        post = _FakePost([_Response(payload=body)])
        monkeypatch.setattr(httpx, "post", post)

        with pytest.raises(RuntimeError) as exc_info:
            _provider().embed(["first", "second"])

        assert expected in str(exc_info.value)
        assert len(post.calls) == 1, "a malformed reply is not a transient failure"

    def test_a_vector_that_is_not_unit_length_is_renormalized(self, isolated, monkeypatch) -> None:
        post = _FakePost([_reply([[3.0, 4.0]])])
        monkeypatch.setattr(httpx, "post", post)

        vector = _provider().embed(["text"])[0]

        assert vector == pytest.approx([0.6, 0.8])
        assert math.sqrt(sum(v * v for v in vector)) == pytest.approx(1.0)

    def test_a_zero_vector_is_an_error_not_a_quiet_passthrough(self, isolated, monkeypatch) -> None:
        post = _FakePost([_reply([[0.0, 0.0]])])
        monkeypatch.setattr(httpx, "post", post)

        with pytest.raises(RuntimeError, match="zero-length vector"):
            _provider().embed(["text"])

    def test_non_numeric_components_are_an_error(self, isolated, monkeypatch) -> None:
        post = _FakePost([_reply([["nan", None]])])
        monkeypatch.setattr(httpx, "post", post)

        with pytest.raises(RuntimeError, match="non-numeric vector components"):
            _provider().embed(["text"])


class TestReplyThatIsJsonButNotAnObject:
    """HTTP 200 whose body is valid JSON and *not* an object — a bare list, a bare
    string, a bare number — is a malformed reply, and every adapter must report it
    the way the transport reports a malformed reply: a `RuntimeError` that names the
    provider. An adapter that reaches for its field without checking what it is
    standing on raises `AttributeError` from inside the parse step instead, which is
    not a provider error the caller can recognize or the retry chain can classify.

    Each adapter reads its own field (`embeddings` for Google, `output.embeddings`
    for DashScope, `data` for the OpenAI-compatible family), so the guard has to be
    in every adapter; these cases run the real ones, not a stand-in.
    """

    ADAPTERS = [
        pytest.param(_MinimalProvider, "CONTRACT_API_KEY", id="base-contract-adapter"),
        pytest.param(GoogleEmbedProvider, "GEMINI_API_KEY", id="google"),
        pytest.param(DashScopeEmbedProvider, "DASHSCOPE_API_KEY", id="dashscope"),
        pytest.param(OpenAICompatibleProvider, "OPENAI_API_KEY", id="openai-compatible"),
        pytest.param(SiliconFlowEmbedProvider, "SILICONFLOW_API_KEY", id="siliconflow"),
        pytest.param(OllamaEmbedProvider, None, id="ollama-keyless"),
    ]

    BODIES = [
        pytest.param([], id="empty-list"),
        pytest.param([[1.0, 0.0], [0.0, 1.0]], id="bare-vector-list"),
        pytest.param("foo", id="string"),
        pytest.param(7, id="number"),
        pytest.param(True, id="boolean"),
    ]

    @pytest.mark.parametrize("adapter, key_env", ADAPTERS)
    @pytest.mark.parametrize("body", BODIES)
    def test_the_reply_becomes_a_provider_error_not_a_crash(
        self, isolated, monkeypatch, adapter, key_env, body
    ) -> None:
        if key_env is not None:
            monkeypatch.setenv(key_env, FAKE_KEY)
        post = _FakePost([_Response(status_code=200, payload=body, text=str(body))])
        monkeypatch.setattr(httpx, "post", post)

        with pytest.raises(RuntimeError) as exc_info:
            adapter(MODEL).embed(["first", "second"])

        message = str(exc_info.value)
        # the shared transport error, carrying the adapter's own prefix: the parse
        # step reported the reply, it did not blow up reading it. Whatever scalar or
        # array the server answered with, the adapter reports "I found no embeddings
        # here" rather than an AttributeError on `.get`.
        assert adapter._ERROR_PREFIX in message
        assert "none embeddings for 2 texts" in message
        assert MODEL in message
        # the request really was made and answered, and a malformed reply is not a
        # transient failure worth replaying.
        assert len(post.calls) == 1

    def test_a_non_object_body_is_not_retried_as_if_it_were_transient(
        self, isolated, monkeypatch
    ) -> None:
        """The AttributeError version of this bug escaped the retry chain entirely;
        the reported error must be the one the chain recognizes, once."""
        post = _FakePost([_Response(status_code=200, payload="[]")])
        monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
        monkeypatch.setattr(httpx, "post", post)

        with pytest.raises(RuntimeError) as exc_info:
            GoogleEmbedProvider(MODEL).embed(["first"])

        assert "Google batchEmbedContents" in str(exc_info.value)
        assert "none embeddings for 1 texts" in str(exc_info.value)
        assert len(post.calls) == 1
        assert exc_info.value.__cause__ is None


class TestOpenAICompatibleProvider:
    def test_the_request_is_the_shape_every_compatible_server_reads(
        self, isolated, monkeypatch
    ) -> None:
        post = _FakePost(echo=True)
        monkeypatch.setattr(httpx, "post", post)

        OpenAICompatibleProvider(MODEL).embed(["one", "two"])

        assert post.calls[0]["url"] == "https://api.openai.com/v1/embeddings"
        assert post.calls[0]["json"] == {"model": MODEL, "input": ["one", "two"]}
        assert post.calls[0]["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"

    def test_the_configured_output_dimension_is_sent_as_the_common_field(
        self, isolated, monkeypatch
    ) -> None:
        post = _FakePost(echo=True)
        monkeypatch.setattr(httpx, "post", post)

        OpenAICompatibleProvider(MODEL, output_dimensionality=512).embed(["one"])

        assert post.calls[0]["json"]["dimensions"] == 512

    def test_a_narrower_gateway_cap_is_a_setting_not_a_code_change(
        self, isolated, monkeypatch, sleeps
    ) -> None:
        assert OpenAICompatibleProvider(MODEL)._batch_limit == 2048

        monkeypatch.setenv("VECTR_EMBED_BATCH_LIMIT", "256")
        provider = OpenAICompatibleProvider(MODEL)
        post = _FakePost(echo=True)
        monkeypatch.setattr(httpx, "post", post)
        texts = [f"t{i}" for i in range(600)]

        vectors = provider.embed(texts)

        assert provider._batch_limit == 256
        assert [len(call["json"]["input"]) for call in post.calls] == [256, 256, 88]
        assert vectors == _expected(texts)

    def test_a_narrowing_setting_never_widens_past_what_the_api_accepts(
        self, isolated, monkeypatch
    ) -> None:
        monkeypatch.setenv("VECTR_EMBED_BATCH_LIMIT", "999")

        assert _provider()._batch_limit == _MinimalProvider._DEFAULT_BATCH_LIMIT

    def test_an_invalid_batch_limit_is_rejected_at_once(self, isolated, monkeypatch) -> None:
        monkeypatch.setenv("VECTR_EMBED_BATCH_LIMIT", "0")
        with pytest.raises(RuntimeError, match="batch limit must be positive"):
            OpenAICompatibleProvider(MODEL)

        monkeypatch.setenv("VECTR_EMBED_BATCH_LIMIT", "whole-corpus")
        with pytest.raises(RuntimeError, match="invalid embedding batch limit"):
            OpenAICompatibleProvider(MODEL)

    @pytest.mark.parametrize(
        "spec, provider_class, base_url, key_env",
        [
            ("siliconflow/BAAI/bge-m3", SiliconFlowEmbedProvider, "https://api.siliconflow.cn/v1", "SILICONFLOW_API_KEY"),
            ("ollama/bge-m3", OllamaEmbedProvider, "http://localhost:11434/v1", "OLLAMA_API_KEY"),
            ("vllm/BAAI/bge-m3", VllmEmbedProvider, "http://localhost:8000/v1", "VLLM_API_KEY"),
        ],
    )
    def test_a_registered_gateway_prefills_only_its_origin_and_key_channel(
        self, isolated, monkeypatch, spec, provider_class, base_url, key_env
    ) -> None:
        monkeypatch.setenv(key_env, FAKE_KEY)

        provider = get_embed_provider(spec)

        assert isinstance(provider, provider_class)
        assert provider._base_url == base_url
        assert provider._key_env == key_env
        assert provider._endpoint() == f"{base_url}/embeddings"

    def test_a_local_gateway_needs_no_key_at_all(self, isolated) -> None:
        provider = get_embed_provider("ollama/bge-m3")

        assert provider._api_key == ""
        assert "Authorization" not in provider._headers()

    def test_a_hosted_gateway_without_a_key_says_which_channel_it_wanted(
        self, isolated
    ) -> None:
        with pytest.raises(RuntimeError, match="SILICONFLOW_API_KEY"):
            get_embed_provider("siliconflow/BAAI/bge-m3")


class TestProviderRegistry:
    def test_the_shipped_prefixes_are_advertised(self, isolated) -> None:
        assert {"google", "dashscope", "openai", "openai-compatible", "siliconflow", "jina", "ollama", "vllm", "local", "voyage"} <= set(PROVIDER_REGISTRY.prefixes)

    def test_routing_is_late_bound_so_a_replaced_provider_is_honoured(
        self, isolated, monkeypatch
    ) -> None:
        monkeypatch.setattr(types_mod, "GoogleEmbedProvider", _StandIn)
        monkeypatch.setattr(types_mod, "OpenAIEmbedProvider", _StandIn)

        google = get_embed_provider("google/gemini-embedding-2")
        openai = get_embed_provider("openai/text-embedding-3-small")

        assert isinstance(google, _StandIn)
        # Settings a remote provider needs come from the registry; the SDK-backed
        # ones take the model name and nothing else.
        assert google.args == ("gemini-embedding-2",)
        assert google.kwargs["api_key_env"] == "GEMINI_API_KEY"
        assert isinstance(openai, _StandIn)
        assert openai.args == ("text-embedding-3-small",)
        assert openai.kwargs == {}

    def test_the_default_key_channel_comes_from_the_provider_itself(
        self, isolated, monkeypatch
    ) -> None:
        monkeypatch.setattr(types_mod, "SiliconFlowEmbedProvider", _StandIn)

        provider = get_embed_provider("siliconflow/BAAI/bge-m3")

        assert provider.kwargs["api_key_env"] == "SILICONFLOW_API_KEY"

    def test_a_configured_key_channel_overrides_the_default(
        self, isolated, monkeypatch, tmp_path
    ) -> None:
        (tmp_path / ".vectr").mkdir()
        (tmp_path / ".vectr" / "config.yaml").write_text(
            "embedding:\n  api_key_env: MY_GATEWAY_KEY\n", encoding="utf-8"
        )
        monkeypatch.setattr(types_mod, "OpenAICompatibleProvider", _StandIn)

        provider = get_embed_provider("openai-compatible/BAAI/bge-m3")

        assert provider.kwargs["api_key_env"] == "MY_GATEWAY_KEY"

    def test_prefixes_are_matched_case_insensitively(self, isolated, monkeypatch) -> None:
        monkeypatch.setattr(types_mod, "GoogleEmbedProvider", _StandIn)

        assert get_embed_provider("Google/gemini-embedding-2").args == ("gemini-embedding-2",)

    @pytest.mark.parametrize("spec", ["google/", "dashscope/", "openai-compatible/"])
    def test_a_bare_prefix_says_which_model_it_wanted(self, isolated, spec) -> None:
        with pytest.raises(ValueError, match="must name a model"):
            get_embed_provider(spec)

    def test_a_huggingface_model_id_is_still_the_local_model(self, isolated, monkeypatch) -> None:
        monkeypatch.setattr(types_mod, "LocalEmbedProvider", _StandIn)

        for spec in (
            "ibm-granite/granite-embedding-english-r2",
            "BAAI/bge-m3",
            "Qwen/Qwen3-Embedding-8B",
            "jinaai/jina-embeddings-v3",
        ):
            provider = get_embed_provider(spec)
            assert isinstance(provider, _StandIn), spec
            assert provider.args == (spec,), spec

        # `local/` states the same choice explicitly; the prefix is a marker, not
        # part of the model id.
        assert get_embed_provider("local/BAAI/bge-m3").args == ("BAAI/bge-m3",)

    @pytest.mark.parametrize(
        "spec, meant",
        [
            ("googel/gemini-embedding-2", "google"),
            ("dashscpe/qwen3.7-text-embedding-flash", "dashscope"),
            ("openiai/text-embedding-3-small", "openai"),
            ("siliconflw/BAAI/bge-m3", "siliconflow"),
        ],
    )
    def test_a_typo_of_a_known_prefix_raises_instead_of_downloading_a_model(
        self, isolated, monkeypatch, spec, meant
    ) -> None:
        # D4: the old chain fell through to a HuggingFace download, so a mistyped
        # provider name failed as a model fetch instead of a routing mistake.
        monkeypatch.setattr(types_mod, "LocalEmbedProvider", _StandIn)

        with pytest.raises(ValueError) as exc_info:
            get_embed_provider(spec)

        message = str(exc_info.value)
        assert "unknown embedding provider prefix" in message
        assert f"did you mean {meant}/?" in message
        assert "dashscope/" in message and "openai-compatible/" in message
        assert get_embed_provider("BAAI/bge-m3").args == ("BAAI/bge-m3",), "a model id is still local"

    @pytest.mark.parametrize(
        "spec", ["gemini/gemini-embedding-2", "aliyun/text-embedding-v4", "bedrock/amzn2-embed-english"]
    )
    def test_a_vendor_name_vectr_does_not_implement_raises(self, isolated, monkeypatch, spec) -> None:
        monkeypatch.setattr(types_mod, "LocalEmbedProvider", _StandIn)

        with pytest.raises(ValueError, match="unknown embedding provider prefix"):
            get_embed_provider(spec)

    def test_legacy_unprefixed_specs_keep_their_providers(self, isolated, monkeypatch) -> None:
        monkeypatch.setattr(types_mod, "VoyageEmbedProvider", _StandIn)
        monkeypatch.setattr(types_mod, "LocalEmbedProvider", _StandIn)

        voyage = get_embed_provider("voyage-code-2")
        local = get_embed_provider("all-MiniLM-L6-v2")

        assert isinstance(voyage, _StandIn) and voyage.args == ("voyage-code-2",)
        assert isinstance(local, _StandIn) and local.args == ("all-MiniLM-L6-v2",)

    def test_a_new_provider_needs_one_entry_and_no_change_to_the_registry(
        self, isolated, monkeypatch
    ) -> None:
        class _Mine(_StandIn):
            pass

        monkeypatch.setattr(types_mod, "MineEmbedProvider", _Mine, raising=False)
        registry = ProviderRegistry([ProviderEntry(prefix="mine", factory="MineEmbedProvider")])

        assert isinstance(registry.create("mine/demo-model"), _Mine)
        assert registry.entry_for("MINE").factory == "MineEmbedProvider"


class TestSharedSettingsAreOneImplementation:
    def test_the_helpers_are_the_same_objects_through_either_module(self, isolated) -> None:
        assert types_mod._normalize_vector is _normalize_vector
        assert types_mod.RemoteEmbedProvider is RemoteEmbedProvider
        assert set(dir(types_mod)) >= {"_embed_config_for", "_output_dimensionality", "_google_output_dimensionality", "resolve_embed_model", "_embed_chars_per_second"}

    def test_normalization_is_shared_by_every_provider(self, isolated) -> None:
        assert _normalize_vector([3.0, 4.0], "Contract", MODEL) == pytest.approx([0.6, 0.8])
        assert _normalize_vector([0.0, 1.0], "Contract", MODEL) == [0.0, 1.0]

        for bad, expected in (([0.0, 0.0], "zero-length vector"), ([None], "non-numeric")):
            with pytest.raises(RuntimeError, match=expected):
                _normalize_vector(bad, "Contract", MODEL)
