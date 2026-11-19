"""Tests for DashScopeEmbedProvider and the `dashscope/<model>` spec.

All HTTP is mocked and every API key here is a fake literal, so no network call
and no real credential is involved. HOME and VECTR_WORKSPACE are pointed at
tmp_path in every test, so neither the real ~/.vectr/secrets.json nor the repo's
own .vectr/config.yaml is ever read. Assertions cover the request the provider
builds (URL, headers, body, text_type, dimension), its per-model batch cap and
batch stitching, its retry/backoff behaviour, the errors it raises, the
normalization it applies — and that no error message ever carries the key.
"""
from __future__ import annotations

import logging
import math
import time

import httpx
import pytest

import agent.indexer._types as types_mod
from agent.indexer import DashScopeEmbedProvider as PackageDashScopeEmbedProvider
from agent.indexer._types import (
    DashScopeEmbedProvider,
    GoogleEmbedProvider,
    get_embed_provider,
    _normalize_vector,
    _output_dimensionality,
)

FAKE_KEY = "sk-fake-dashscope-key-never-a-real-one"
SECRETS_KEY = "sk-from-secrets-file-never-a-real-one"
QWEN_MODEL = "qwen3.7-text-embedding-flash"
ENDPOINT_PATH = "/api/v1/services/embeddings/text-embedding/text-embedding"
PUBLIC_URL = f"https://dashscope.aliyuncs.com{ENDPOINT_PATH}"


class _Response:
    """Stand-in for an httpx.Response with just what the provider reads."""

    def __init__(self, status_code=200, payload=None, headers=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.text = text
        self.reason_phrase = {
            200: "OK", 400: "Bad Request", 401: "Unauthorized", 403: "Forbidden",
            404: "Not Found", 429: "Too Many Requests",
            500: "Internal Server Error", 502: "Bad Gateway",
            503: "Service Unavailable", 504: "Gateway Timeout",
        }.get(status_code, "Unknown Status")

    def json(self):
        # payload=None models the real thing: a body that is not JSON.
        if self._payload is None:
            raise ValueError("body is not JSON")
        return self._payload


def _request_texts(body: dict) -> list[str]:
    """The texts a provider sent, whatever its wire shape: DashScope nests them
    under input.texts, Google wraps each one in its own request object."""
    if "input" in body:
        return list(body["input"]["texts"])
    return [request["content"]["parts"][0]["text"] for request in body["requests"]]


class _FakePost:
    """Stand-in for httpx.post: records every call, replays queued outcomes.

    A queued BaseException is raised instead of returned — that is how the
    transport-retry tests inject a connection-level failure.
    """

    def __init__(self, responses=None, echo=False):
        self._responses = list(responses or [])
        self._echo = echo
        self.calls: list[dict] = []

    def __call__(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        if self._echo:
            # One vector per input text, keyed by the number written into the text
            # itself, so the test can prove both ordering and normalization. The
            # vectors are deliberately not unit length.
            vectors = [[float(text) + 1.0, 1.0] for text in _request_texts(json)]
            if "input" in json:
                entries = [{"text_index": i, "embedding": v} for i, v in enumerate(vectors)]
                return _Response(payload=_payload(entries))
            return _Response(payload={"embeddings": [{"values": v} for v in vectors]})
        if not self._responses:
            raise AssertionError("provider made more HTTP calls than the test queued")
        outcome = self._responses.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _payload(entries: list[dict]) -> dict:
    """Shape of a DashScope success body: output.embeddings[], plus request_id."""
    return {
        "output": {"embeddings": list(entries)},
        "usage": {"total_tokens": 1},
        "request_id": "req-echo-0001",
    }


def _vectors(values: list[list[float]]) -> dict:
    return _payload([{"text_index": i, "embedding": list(v)} for i, v in enumerate(values)])


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector]


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    """A provider's-eye view with no real config and no real secrets in reach.

    HOME and VECTR_WORKSPACE both land on tmp_path, so the ~/.vectr/secrets.json
    and .vectr/config.yaml a test writes are tmp_path's own, and the real
    ~/.vectr and the repo's own .vectr/config.yaml are never read.
    """
    monkeypatch.setenv("DASHSCOPE_API_KEY", FAKE_KEY)
    monkeypatch.setenv("HOME", str(tmp_path))                     # no secrets.json until the test writes one
    monkeypatch.setenv("VECTR_WORKSPACE", str(tmp_path))          # no .vectr/config.yaml until the test writes one
    for name in (
        "VECTR_EMBED_MODEL",
        "VECTR_EMBED_OUTPUT_DIM",
        "VECTR_EMBED_BASE_URL",
        "VECTR_EMBED_CHARS_PER_SECOND",
        "GEMINI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def _provider(model=QWEN_MODEL, dim=None, api_key_env=None, base_url=None):
    return DashScopeEmbedProvider(model, output_dimensionality=dim, api_key_env=api_key_env,
                                  base_url=base_url)


class TestRequestShape:
    def test_embed_posts_the_native_dashscope_request(self, isolated, monkeypatch) -> None:
        post = _FakePost([_Response(payload=_vectors([[3.0, 4.0]]))])
        monkeypatch.setattr(httpx, "post", post)

        vectors = _provider().embed(["def index_files(): ..."])

        assert len(post.calls) == 1
        call = post.calls[0]
        assert call["url"] == PUBLIC_URL
        assert call["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"
        assert call["headers"]["Content-Type"] == "application/json"
        assert call["timeout"] == DashScopeEmbedProvider._TIMEOUT_SECONDS
        assert call["json"]["model"] == QWEN_MODEL
        assert call["json"]["input"] == {"texts": ["def index_files(): ..."]}
        assert call["json"]["parameters"] == {"text_type": "document"}
        # A unit-normalized [3,4] is what comes back out.
        assert vectors == [[pytest.approx(0.6), pytest.approx(0.8)]]

    def test_dimension_is_sent_only_when_configured(self, isolated, monkeypatch) -> None:
        post = _FakePost([_Response(payload=_vectors([[1.0, 0.0]]))])
        monkeypatch.setattr(httpx, "post", post)
        monkeypatch.setenv("VECTR_EMBED_OUTPUT_DIM", "1024")

        _provider(dim=_output_dimensionality(types_mod._embed_config_for(None))).embed(["a"])

        assert post.calls[0]["json"]["parameters"] == {"text_type": "document", "dimension": 1024}

    def test_no_dimension_key_is_invented_when_unset(self, isolated, monkeypatch) -> None:
        post = _FakePost([_Response(payload=_vectors([[1.0, 0.0]]))])
        monkeypatch.setattr(httpx, "post", post)

        _provider().embed(["a"])

        assert "dimension" not in post.calls[0]["json"]["parameters"]

    def test_base_url_comes_from_the_env_override(self, isolated, monkeypatch) -> None:
        post = _FakePost([_Response(payload=_vectors([[1.0, 0.0]]))])
        monkeypatch.setattr(httpx, "post", post)
        monkeypatch.setenv("VECTR_EMBED_BASE_URL", "https://dashscope-intl.aliyuncs.com/")

        _provider().embed(["a"])

        # Trailing slash normalized away, path appended: no doubled slash.
        assert post.calls[0]["url"] == f"https://dashscope-intl.aliyuncs.com{ENDPOINT_PATH}"

    def test_base_url_comes_from_the_config_file(self, isolated, monkeypatch) -> None:
        (isolated / ".vectr").mkdir(parents=True)
        (isolated / ".vectr" / "config.yaml").write_text(
            "embedding:\n  base_url: http://127.0.0.1:8080/dashscope\n", encoding="utf-8"
        )
        post = _FakePost([_Response(payload=_vectors([[1.0, 0.0]]))])
        monkeypatch.setattr(httpx, "post", post)

        _provider().embed(["a"])

        assert post.calls[0]["url"] == f"http://127.0.0.1:8080/dashscope{ENDPOINT_PATH}"

    def test_a_non_http_base_url_is_rejected(self, isolated) -> None:
        with pytest.raises(RuntimeError, match="must be an http\\(s\\) URL"):
            _provider(base_url="dashscope.internal")

    def test_empty_input_makes_no_request(self, isolated, monkeypatch) -> None:
        post = _FakePost([])
        monkeypatch.setattr(httpx, "post", post)

        assert _provider().embed([]) == []
        assert _provider().embed_query([]) == []
        assert post.calls == []

    def test_an_unnamed_dashscope_spec_is_rejected(self, isolated) -> None:
        with pytest.raises(ValueError, match="must name a model"):
            _provider(model="")


class TestAsymmetricRetrieval:
    def test_embed_is_document_and_embed_query_is_query(self, isolated, monkeypatch) -> None:
        post = _FakePost(
            [
                _Response(payload=_vectors([[1.0, 0.0]])),
                _Response(payload=_vectors([[0.0, 1.0]])),
            ]
        )
        monkeypatch.setattr(httpx, "post", post)
        provider = _provider()

        provider.embed(["indexed chunk"])
        provider.embed_query(["what parses python?"])

        assert [call["json"]["parameters"]["text_type"] for call in post.calls] == [
            "document",
            "query",
        ]
        assert post.calls[0]["json"]["input"]["texts"] == ["indexed chunk"]
        assert post.calls[1]["json"]["input"]["texts"] == ["what parses python?"]

    def test_both_sides_hit_the_same_endpoint_with_the_same_model(self, isolated, monkeypatch) -> None:
        post = _FakePost(
            [
                _Response(payload=_vectors([[1.0, 0.0]])),
                _Response(payload=_vectors([[0.0, 1.0]])),
            ]
        )
        monkeypatch.setattr(httpx, "post", post)
        provider = _provider(model="text-embedding-v4")

        provider.embed(["a"])
        provider.embed_query(["b"])

        assert {call["url"] for call in post.calls} == {PUBLIC_URL}
        assert {call["json"]["model"] for call in post.calls} == {"text-embedding-v4"}


class TestBatching:
    def test_qwen_cap_is_20_texts_per_request(self) -> None:
        assert DashScopeEmbedProvider._batch_limit_for(QWEN_MODEL) == 20
        assert DashScopeEmbedProvider._batch_limit_for("qwen3.7-text-embedding") == 20

    def test_text_embedding_v3_and_v4_cap_is_10(self) -> None:
        assert DashScopeEmbedProvider._batch_limit_for("text-embedding-v3") == 10
        assert DashScopeEmbedProvider._batch_limit_for("text-embedding-v4") == 10

    def test_45_texts_split_20_20_5_and_keep_their_order(self, isolated, monkeypatch) -> None:
        post = _FakePost(echo=True)
        monkeypatch.setattr(httpx, "post", post)
        monkeypatch.setenv("VECTR_EMBED_CHARS_PER_SECOND", "100000")
        texts = [str(i) for i in range(45)]

        vectors = _provider().embed(texts)

        assert [len(call["json"]["input"]["texts"]) for call in post.calls] == [20, 20, 5]
        assert post.calls[0]["json"]["input"]["texts"] == texts[:20]
        assert post.calls[2]["json"]["input"]["texts"] == texts[40:]
        assert len(vectors) == 45
        for index, vector in enumerate(vectors):
            expected = _unit([float(index) + 1.0, 1.0])
            assert vector == [pytest.approx(expected[0]), pytest.approx(expected[1])]
        # 1/hypot(i+1, 1) strictly decreases with the input index, so a strictly
        # decreasing second component proves no batch was reordered or repeated.
        assert all(vectors[i][1] > vectors[i + 1][1] for i in range(44))

    def test_v3_spec_batches_by_10(self, isolated, monkeypatch) -> None:
        post = _FakePost(echo=True)
        monkeypatch.setattr(httpx, "post", post)
        monkeypatch.setenv("VECTR_EMBED_CHARS_PER_SECOND", "100000")

        vectors = _provider(model="text-embedding-v3").embed([str(i) for i in range(25)])

        assert [len(call["json"]["input"]["texts"]) for call in post.calls] == [10, 10, 5]
        assert len(vectors) == 25

    def test_batches_are_paced_by_characters_not_a_fixed_sleep(self, isolated, monkeypatch) -> None:
        post = _FakePost(echo=True)
        monkeypatch.setattr(httpx, "post", post)
        sleeps: list[float] = []
        monkeypatch.setattr(time, "sleep", sleeps.append)
        monkeypatch.setenv("VECTR_EMBED_CHARS_PER_SECOND", "1000")
        texts = ["0123456789"] * 20  # 2 batches of 10 texts (v3 cap), 100 chars each

        _provider(model="text-embedding-v3").embed(texts)

        # One sleep per inter-batch gap (100 chars / 1000 chars-per-second), and
        # none after the final batch.
        assert sleeps == [pytest.approx(0.1)]

    def test_vectors_are_re_keyed_from_text_index_not_array_order(self, isolated, monkeypatch) -> None:
        entries = [
            {"text_index": 2, "embedding": [3.0, 4.0]},
            {"text_index": 0, "embedding": [1.0, 0.0]},
            {"text_index": 1, "embedding": [0.0, 2.0]},
        ]
        post = _FakePost([_Response(payload=_payload(entries))])
        monkeypatch.setattr(httpx, "post", post)

        vectors = _provider().embed(["a", "b", "c"])

        assert vectors == [
            [pytest.approx(1.0), pytest.approx(0.0)],
            [pytest.approx(0.0), pytest.approx(1.0)],
            [pytest.approx(0.6), pytest.approx(0.8)],
        ]

    def test_text_index_outside_the_batch_is_rejected(self, isolated, monkeypatch) -> None:
        entries = [{"text_index": 0, "embedding": [1.0, 0.0]}, {"text_index": 7, "embedding": [0.0, 1.0]}]
        post = _FakePost([_Response(payload=_payload(entries))])
        monkeypatch.setattr(httpx, "post", post)

        with pytest.raises(RuntimeError, match=r"text_index 7 outside"):
            _provider().embed(["a", "b"])

    def test_duplicate_text_index_is_rejected(self, isolated, monkeypatch) -> None:
        entries = [{"text_index": 0, "embedding": [1.0, 0.0]}, {"text_index": 0, "embedding": [0.0, 1.0]}]
        post = _FakePost([_Response(payload=_payload(entries))])
        monkeypatch.setattr(httpx, "post", post)

        with pytest.raises(RuntimeError, match=r"text_index 0 outside"):
            _provider().embed(["a", "b"])

    def test_malformed_text_index_is_rejected(self, isolated, monkeypatch) -> None:
        entries = [{"text_index": "not-a-number", "embedding": [1.0, 0.0]}]
        post = _FakePost([_Response(payload=_payload(entries))])
        monkeypatch.setattr(httpx, "post", post)

        with pytest.raises(RuntimeError, match="malformed text_index"):
            _provider().embed(["a"])


class TestRetries:
    def test_429_is_retried_with_exponential_backoff(self, isolated, monkeypatch) -> None:
        post = _FakePost(
            [
                _Response(429, payload={"code": "Throttling.RateQuota", "message": "Too many requests",
                                        "request_id": "req-429"}),
                _Response(payload=_vectors([[1.0, 0.0]])),
            ]
        )
        monkeypatch.setattr(httpx, "post", post)
        sleeps: list[float] = []
        monkeypatch.setattr(time, "sleep", sleeps.append)

        vectors = _provider().embed(["a"])

        assert len(post.calls) == 2
        assert sleeps == [1.0]
        assert vectors == [[1.0, 0.0]]

    def test_a_body_level_throttling_code_is_retried_whatever_the_status(
        self, isolated, monkeypatch
    ) -> None:
        # Some DashScope endpoints report quota throttling as 400 + Throttling.*
        # rather than 429; both are the same transient event.
        post = _FakePost(
            [
                _Response(400, payload={"code": "Throttling.AllocationQuota",
                                        "message": "Allocated quota exceeded",
                                        "request_id": "req-quota"}),
                _Response(payload=_vectors([[1.0, 0.0]])),
            ]
        )
        monkeypatch.setattr(httpx, "post", post)
        sleeps: list[float] = []
        monkeypatch.setattr(time, "sleep", sleeps.append)

        _provider().embed(["a"])

        assert len(post.calls) == 2
        assert sleeps == [1.0]

    def test_retry_after_header_wins_over_the_backoff_curve(self, isolated, monkeypatch) -> None:
        post = _FakePost(
            [
                _Response(429, headers={"retry-after": "7"},
                          payload={"code": "Throttling.RateQuota", "message": "slow down"}),
                _Response(payload=_vectors([[1.0, 0.0]])),
            ]
        )
        monkeypatch.setattr(httpx, "post", post)
        sleeps: list[float] = []
        monkeypatch.setattr(time, "sleep", sleeps.append)

        _provider().embed(["a"])

        assert sleeps == [7.0]

    def test_an_absurd_retry_after_is_clamped_to_the_backoff_cap(self, isolated, monkeypatch) -> None:
        post = _FakePost(
            [
                _Response(429, headers={"retry-after": "3600"}, payload={"code": "Throttling.RateQuota"}),
                _Response(payload=_vectors([[1.0, 0.0]])),
            ]
        )
        monkeypatch.setattr(httpx, "post", post)
        sleeps: list[float] = []
        monkeypatch.setattr(time, "sleep", sleeps.append)

        _provider().embed(["a"])

        assert sleeps == [60.0]

    def test_5xx_is_retried_then_succeeds(self, isolated, monkeypatch) -> None:
        post = _FakePost(
            [
                _Response(502, text="Bad Gateway"),
                _Response(503, text="Service Unavailable"),
                _Response(payload=_vectors([[1.0, 0.0]])),
            ]
        )
        monkeypatch.setattr(httpx, "post", post)
        sleeps: list[float] = []
        monkeypatch.setattr(time, "sleep", sleeps.append)

        vectors = _provider().embed(["a"])

        assert len(post.calls) == 3
        assert sleeps == [1.0, 2.0]
        assert vectors == [[1.0, 0.0]]

    def test_retries_are_capped_and_then_reported(self, isolated, monkeypatch) -> None:
        post = _FakePost([_Response(500, payload={"code": "InternalError", "message": "boom",
                                                  "request_id": "req-500"})
                          for _ in range(DashScopeEmbedProvider._MAX_ATTEMPTS)])
        monkeypatch.setattr(httpx, "post", post)
        sleeps: list[float] = []
        monkeypatch.setattr(time, "sleep", sleeps.append)

        with pytest.raises(RuntimeError, match=r"attempt 8/8"):
            _provider().embed(["a"])

        assert len(post.calls) == 8
        assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0]

    def test_transport_failures_are_retried_like_a_429(self, isolated, monkeypatch) -> None:
        post = _FakePost(
            [
                httpx.ConnectError("connection reset by peer"),
                httpx.ReadTimeout("read timed out"),
                _Response(payload=_vectors([[1.0, 0.0]])),
            ]
        )
        monkeypatch.setattr(httpx, "post", post)
        sleeps: list[float] = []
        monkeypatch.setattr(time, "sleep", sleeps.append)

        vectors = _provider().embed(["a"])

        assert len(post.calls) == 3
        assert sleeps == [1.0, 2.0]
        assert vectors == [[1.0, 0.0]]

    def test_an_exhausted_connection_retries_reports_the_original_failure(
        self, isolated, monkeypatch
    ) -> None:
        post = _FakePost([httpx.ConnectError("network unreachable")
                          for _ in range(DashScopeEmbedProvider._MAX_ATTEMPTS)])
        monkeypatch.setattr(httpx, "post", post)
        sleeps: list[float] = []
        monkeypatch.setattr(time, "sleep", sleeps.append)

        with pytest.raises(RuntimeError) as excinfo:
            _provider().embed(["a"])

        assert len(post.calls) == 8
        assert "network unreachable" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, httpx.ConnectError)

    def test_a_non_retryable_client_error_is_reported_after_one_call(
        self, isolated, monkeypatch
    ) -> None:
        post = _FakePost(
            [_Response(400, payload={"code": "InvalidParameter", "message": "texts is empty",
                                     "request_id": "req-400"})]
        )
        monkeypatch.setattr(httpx, "post", post)
        sleeps: list[float] = []
        monkeypatch.setattr(time, "sleep", sleeps.append)

        with pytest.raises(RuntimeError) as excinfo:
            _provider().embed(["a"])

        assert len(post.calls) == 1
        assert sleeps == []
        message = str(excinfo.value)
        assert "InvalidParameter" in message and "req-400" in message
        assert "texts is empty" in message


class TestErrorsAndKeySafety:
    def test_missing_key_names_the_channel_not_a_key(self, isolated, monkeypatch) -> None:
        monkeypatch.delenv("DASHSCOPE_API_KEY")

        with pytest.raises(RuntimeError) as excinfo:
            _provider()

        message = str(excinfo.value)
        assert "DASHSCOPE_API_KEY" in message
        assert "dashscope_api_key" in message  # the secrets.json entry it looked for
        assert FAKE_KEY not in message

    def test_a_rejected_key_is_reported_without_leaking_it(self, isolated, monkeypatch) -> None:
        post = _FakePost([_Response(401, payload={"code": "InvalidApiKey",
                                                  "message": f"Invalid API-key provided: {FAKE_KEY}",
                                                  "request_id": "req-401"})])
        monkeypatch.setattr(httpx, "post", post)

        with pytest.raises(RuntimeError) as excinfo:
            _provider().embed(["a"])

        message = str(excinfo.value)
        assert FAKE_KEY not in message
        assert "InvalidApiKey" in message and "req-401" in message
        assert "DASHSCOPE_API_KEY" in message  # tells the operator where to look

    def test_a_body_that_echoes_the_request_is_scrubbed_before_reporting(
        self, isolated, monkeypatch
    ) -> None:
        # Free-form body, no error envelope: the fallback quotes the body, so the
        # key has to be redacted out of it.
        post = _FakePost([_Response(400, payload=None, text=f"bad request: Bearer {FAKE_KEY}")])
        monkeypatch.setattr(httpx, "post", post)

        with pytest.raises(RuntimeError) as excinfo:
            _provider().embed(["a"])

        assert FAKE_KEY not in str(excinfo.value)
        assert "[redacted]" in str(excinfo.value)

    def test_secrets_json_supplies_the_key_when_the_env_var_is_absent(
        self, isolated, monkeypatch, caplog
    ) -> None:
        monkeypatch.delenv("DASHSCOPE_API_KEY")
        secrets = isolated / ".vectr" / "secrets.json"
        secrets.parent.mkdir(parents=True)
        secrets.write_text(f'{{"dashscope_api_key": "{SECRETS_KEY}"}}', encoding="utf-8")
        secrets.chmod(0o600)
        post = _FakePost([_Response(payload=_vectors([[1.0, 0.0]]))])
        monkeypatch.setattr(httpx, "post", post)

        _provider().embed(["a"])

        assert post.calls[0]["headers"]["Authorization"] == f"Bearer {SECRETS_KEY}"
        assert "chmod 600" not in caplog.text

    def test_the_env_var_wins_over_the_secrets_file(self, isolated, monkeypatch) -> None:
        secrets = isolated / ".vectr" / "secrets.json"
        secrets.parent.mkdir(parents=True)
        secrets.write_text(f'{{"dashscope_api_key": "{SECRETS_KEY}"}}', encoding="utf-8")
        secrets.chmod(0o600)
        post = _FakePost([_Response(payload=_vectors([[1.0, 0.0]]))])
        monkeypatch.setattr(httpx, "post", post)

        _provider().embed(["a"])

        assert post.calls[0]["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"

    def test_a_group_readable_secrets_file_is_warned_about(
        self, isolated, monkeypatch, caplog
    ) -> None:
        monkeypatch.delenv("DASHSCOPE_API_KEY")
        secrets = isolated / ".vectr" / "secrets.json"
        secrets.parent.mkdir(parents=True)
        secrets.write_text(f'{{"dashscope_api_key": "{SECRETS_KEY}"}}', encoding="utf-8")
        secrets.chmod(0o644)
        monkeypatch.setattr(httpx, "post", _FakePost([_Response(payload=_vectors([[1.0, 0.0]]))]))
        caplog.set_level(logging.WARNING, logger="vectr.embed")

        _provider().embed(["a"])

        assert "chmod 600" in caplog.text
        assert SECRETS_KEY not in caplog.text

    def test_malformed_secrets_file_raises_instead_of_going_keyless(
        self, isolated, monkeypatch
    ) -> None:
        monkeypatch.delenv("DASHSCOPE_API_KEY")
        secrets = isolated / ".vectr" / "secrets.json"
        secrets.parent.mkdir(parents=True)
        secrets.write_text("definitely not json", encoding="utf-8")
        secrets.chmod(0o600)

        with pytest.raises(RuntimeError, match="not valid JSON"):
            _provider()

    def test_a_json_secrets_file_that_is_not_an_object_is_reported(self, isolated, monkeypatch) -> None:
        monkeypatch.delenv("DASHSCOPE_API_KEY")
        secrets = isolated / ".vectr" / "secrets.json"
        secrets.parent.mkdir(parents=True)
        secrets.write_text('["just-a-list"]', encoding="utf-8")
        secrets.chmod(0o600)

        with pytest.raises(RuntimeError, match="must contain a JSON object"):
            _provider()

    def test_a_custom_key_env_name_is_used(self, isolated, monkeypatch) -> None:
        monkeypatch.delenv("DASHSCOPE_API_KEY")
        monkeypatch.setenv("MY_DS_KEY", FAKE_KEY)
        post = _FakePost([_Response(payload=_vectors([[1.0, 0.0]]))])
        monkeypatch.setattr(httpx, "post", post)

        _provider(api_key_env="MY_DS_KEY").embed(["a"])

        assert post.calls[0]["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"

    def test_google_keeps_reading_GEMINI_API_KEY_through_the_generic_resolver(
        self, isolated, monkeypatch
    ) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
        post = _FakePost([_Response(payload={"embeddings": [{"values": [1.0, 0.0]}]})])
        monkeypatch.setattr(httpx, "post", post)

        vectors = GoogleEmbedProvider("gemini-embedding-2").embed(["a"])

        assert post.calls[0]["headers"]["x-goog-api-key"] == FAKE_KEY
        assert vectors == [[1.0, 0.0]]

    def test_google_falls_back_to_the_gemini_api_key_secret_entry(
        self, isolated, monkeypatch
    ) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)  # isolated keeps it unset already
        secrets = isolated / ".vectr" / "secrets.json"
        secrets.parent.mkdir(parents=True)
        secrets.write_text(f'{{"gemini_api_key": "{SECRETS_KEY}"}}', encoding="utf-8")
        secrets.chmod(0o600)
        post = _FakePost([_Response(payload={"embeddings": [{"values": [1.0, 0.0]}]})])
        monkeypatch.setattr(httpx, "post", post)

        GoogleEmbedProvider("gemini-embedding-2").embed(["a"])

        assert post.calls[0]["headers"]["x-goog-api-key"] == SECRETS_KEY

    def test_a_dashscope_key_does_not_feed_google_and_vice_versa(self, isolated, monkeypatch) -> None:
        # The two channels stay separate in both directions: dashscope_api_key
        # never authorizes Google, and GEMINI_API_KEY never authorizes DashScope.
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        secrets = isolated / ".vectr" / "secrets.json"
        secrets.parent.mkdir(parents=True)
        secrets.write_text(f'{{"dashscope_api_key": "{SECRETS_KEY}"}}', encoding="utf-8")
        secrets.chmod(0o600)

        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            GoogleEmbedProvider("gemini-embedding-2")

        monkeypatch.delenv("DASHSCOPE_API_KEY")
        monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
        secrets.write_text(f'{{"gemini_api_key": "{SECRETS_KEY}"}}', encoding="utf-8")

        with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY"):
            _provider()

    def test_embeddings_count_mismatch_is_reported(self, isolated, monkeypatch) -> None:
        post = _FakePost([_Response(payload=_vectors([[1.0, 0.0], [0.0, 1.0]]))])
        monkeypatch.setattr(httpx, "post", post)

        with pytest.raises(RuntimeError, match=r"2 embeddings for 3 texts"):
            _provider().embed(["a", "b", "c"])

    def test_a_missing_embedding_array_is_reported(self, isolated, monkeypatch) -> None:
        post = _FakePost([_Response(payload={"output": {}, "request_id": "req-x"})])
        monkeypatch.setattr(httpx, "post", post)

        with pytest.raises(RuntimeError, match="none embeddings for 1 texts"):
            _provider().embed(["a"])

    def test_an_empty_embedding_entry_is_reported(self, isolated, monkeypatch) -> None:
        post = _FakePost([_Response(payload=_vectors([[]]))])
        monkeypatch.setattr(httpx, "post", post)

        with pytest.raises(RuntimeError, match="malformed embedding entry"):
            _provider().embed(["a"])

    def test_a_200_with_a_non_json_body_is_reported(self, isolated, monkeypatch) -> None:
        post = _FakePost([_Response(200, payload=None, text="<html>gateway said OK</html>")])
        monkeypatch.setattr(httpx, "post", post)
        sleeps: list[float] = []
        monkeypatch.setattr(time, "sleep", sleeps.append)

        with pytest.raises(RuntimeError, match="not JSON") as excinfo:
            _provider().embed(["a"])

        assert "gateway said OK" in str(excinfo.value)
        # HTTP 200 claims success: the provider reports it instead of replaying it.
        assert sleeps == []
        assert len(post.calls) == 1

    def test_an_error_body_nested_under_error_is_read_too(self, isolated, monkeypatch) -> None:
        # The OpenAI-compatible envelope nests the same fields under "error".
        post = _FakePost([_Response(400, payload={"error": {"code": "Model.NotFound",
                                                            "message": "no such model"}})])
        monkeypatch.setattr(httpx, "post", post)

        with pytest.raises(RuntimeError, match="Model.NotFound"):
            _provider().embed(["a"])


class TestNormalization:
    def test_a_non_unit_vector_is_normalized(self, isolated, monkeypatch) -> None:
        post = _FakePost([_Response(payload=_vectors([[3.0, 4.0], [1.0, 1.0, 1.0]]))])
        monkeypatch.setattr(httpx, "post", post)

        vectors = _provider().embed(["a", "b"])

        assert vectors[0] == [pytest.approx(0.6), pytest.approx(0.8)]
        assert vectors[1] == [pytest.approx(1 / math.sqrt(3))] * 3
        for vector in vectors:
            assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, abs_tol=1e-12)

    def test_a_vector_that_is_already_unit_length_is_left_alone(self, isolated, monkeypatch) -> None:
        post = _FakePost([_Response(payload=_vectors([[1.0, 0.0, 0.0]]))])
        monkeypatch.setattr(httpx, "post", post)

        assert _provider().embed(["a"]) == [[1.0, 0.0, 0.0]]

    def test_a_zero_vector_is_rejected(self, isolated, monkeypatch) -> None:
        post = _FakePost([_Response(payload=_vectors([[0.0, 0.0]]))])
        monkeypatch.setattr(httpx, "post", post)

        with pytest.raises(RuntimeError, match="zero-length vector"):
            _provider().embed(["a"])

    def test_non_numeric_components_are_rejected(self, isolated, monkeypatch) -> None:
        post = _FakePost([_Response(payload=_vectors([["0.6", "n/a"]]))])
        monkeypatch.setattr(httpx, "post", post)

        with pytest.raises(RuntimeError, match="non-numeric vector components"):
            _provider().embed(["a"])

    def test_both_api_providers_share_one_normalizer(self, isolated, monkeypatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
        monkeypatch.setattr(httpx, "post", _FakePost(echo=True))

        dashscope = DashScopeEmbedProvider(QWEN_MODEL).embed(["2"])
        google = GoogleEmbedProvider("gemini-embedding-2").embed(["2"])

        expected = _unit([3.0, 1.0])
        assert dashscope == [[pytest.approx(expected[0]), pytest.approx(expected[1])]]
        assert google == [[pytest.approx(expected[0]), pytest.approx(expected[1])]]
        assert _normalize_vector([3.0, 4.0], "DashScope", "m") == pytest.approx([0.6, 0.8])


class TestFactoryRouting:
    class _StandIn:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    def test_dashscope_spec_selects_the_provider(self, isolated, monkeypatch) -> None:
        monkeypatch.setattr(httpx, "post", _FakePost([_Response(payload=_vectors([[1.0, 0.0]]))]))

        provider = get_embed_provider(f"dashscope/{QWEN_MODEL}")

        assert isinstance(provider, DashScopeEmbedProvider)
        assert provider._model == QWEN_MODEL
        assert provider._base_url == "https://dashscope.aliyuncs.com"

    def test_factory_reads_dimension_key_env_and_base_url_from_the_config(
        self, isolated, monkeypatch
    ) -> None:
        (isolated / ".vectr").mkdir(parents=True)
        (isolated / ".vectr" / "config.yaml").write_text(
            "embedding:\n"
            "  model: dashscope/qwen3.7-text-embedding-flash\n"
            "  output_dimensionality: 1024\n"
            "  api_key_env: MY_DS_KEY\n"
            "  base_url: http://127.0.0.1:9099/gateway\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("MY_DS_KEY", FAKE_KEY)
        post = _FakePost([_Response(payload=_vectors([[1.0, 0.0]]))])
        monkeypatch.setattr(httpx, "post", post)

        provider = get_embed_provider("dashscope/qwen3.7-text-embedding-flash")
        provider.embed(["a"])

        call = post.calls[0]
        assert call["url"] == f"http://127.0.0.1:9099/gateway{ENDPOINT_PATH}"
        assert call["json"]["parameters"] == {"text_type": "document", "dimension": 1024}
        assert call["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"

    def test_the_env_var_beats_the_config_dimension(self, isolated, monkeypatch) -> None:
        (isolated / ".vectr").mkdir(parents=True)
        (isolated / ".vectr" / "config.yaml").write_text(
            "embedding:\n  output_dimensionality: 512\n", encoding="utf-8"
        )
        monkeypatch.setenv("VECTR_EMBED_OUTPUT_DIM", "1536")
        post = _FakePost([_Response(payload=_vectors([[1.0, 0.0]]))])
        monkeypatch.setattr(httpx, "post", post)

        get_embed_provider(f"dashscope/{QWEN_MODEL}").embed(["a"])

        assert post.calls[0]["json"]["parameters"]["dimension"] == 1536

    def test_dashscope_spec_without_a_model_is_rejected(self, isolated) -> None:
        with pytest.raises(ValueError, match="must name a model"):
            get_embed_provider("dashscope/")

    def test_other_specs_keep_their_existing_providers(self, isolated, monkeypatch) -> None:
        monkeypatch.setattr(types_mod, "GoogleEmbedProvider", self._StandIn)
        monkeypatch.setattr(types_mod, "OpenAIEmbedProvider", self._StandIn)
        monkeypatch.setattr(types_mod, "LocalEmbedProvider", self._StandIn)

        google = get_embed_provider("google/gemini-embedding-2")
        assert isinstance(google, self._StandIn)
        assert google.kwargs["api_key_env"] == "GEMINI_API_KEY"

        assert isinstance(get_embed_provider("openai/text-embedding-3-small"), self._StandIn)
        assert isinstance(
            get_embed_provider("ibm-granite/granite-embedding-english-r2"), self._StandIn
        )

    def test_the_package_reexports_the_dashscope_provider(self) -> None:
        assert PackageDashScopeEmbedProvider is DashScopeEmbedProvider
