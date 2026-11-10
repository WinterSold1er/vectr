"""Tests for GoogleEmbedProvider and the `google/<model>` spec (UPG-EMBED-CONFIG-LAYERING).

All HTTP is mocked and every API key here is a fake literal, so no network call
and no real credential is involved — and HOME is pointed at tmp_path in every
key test so the real ~/.vectr/secrets.json is never read. Assertions cover the
request the provider builds, the response it parses, the normalization it
applies, its retry/backoff behaviour, and the errors it raises.
"""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import httpx
import pytest

import agent.indexer._types as types_mod
from agent.indexer._types import GoogleEmbedProvider, get_embed_provider

FAKE_KEY = "fake-key-never-a-real-one"


class _Response:
    """Stand-in for an httpx.Response with just what the provider reads."""

    def __init__(self, status_code=200, payload=None, headers=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {"embeddings": []}
        self.headers = headers or {}
        self.text = text
        self.reason_phrase = {200: "OK", 400: "Bad Request", 429: "Too Many Requests"}[status_code]

    def json(self):
        return self._payload


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
            # One vector per request, whose components are strictly increasing in
            # the request's index — so the test can prove output ordering.
            payloads = []
            for i, request in enumerate(json["requests"]):
                value = 1.0 + float(request["content"]["parts"][0]["text"].split("t")[1])
                payloads.append({"values": [value, 1.0]})
            return _Response(payload={"embeddings": payloads})
        if not self._responses:
            raise AssertionError("provider made more HTTP calls than the test queued")
        outcome = self._responses.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _payload(values_per_text: list[list[float]]) -> dict:
    return {"embeddings": [{"values": list(v)} for v in values_per_text]}


@pytest.fixture
def fake_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    monkeypatch.delenv("VECTR_EMBED_MODEL", raising=False)
    return FAKE_KEY


@pytest.fixture
def no_home_secrets(monkeypatch, tmp_path):
    """Keep the secrets-file fallback inside tmp_path, never the real ~/.vectr."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return tmp_path / "home"


def _provider(model="gemini-embedding-2", dim=None, api_key_env="GEMINI_API_KEY"):
    return GoogleEmbedProvider(model, output_dimensionality=dim, api_key_env=api_key_env)


class TestRequestShape:
    def test_document_request_matches_the_batchembedcontents_api(
        self, monkeypatch, fake_key
    ) -> None:
        post = _FakePost([_Response(payload=_payload([[1.0, 0.0, 0.0, 0.0]]))])
        monkeypatch.setattr(httpx, "post", post)
        vectors = _provider().embed(["chunk text"])

        call = post.calls[0]
        assert call["url"] == (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-embedding-2:batchEmbedContents"
        )
        assert FAKE_KEY not in call["url"], "the key must not ride in the URL"
        assert call["headers"]["x-goog-api-key"] == FAKE_KEY
        assert call["timeout"] == GoogleEmbedProvider._TIMEOUT_SECONDS
        request = call["json"]["requests"][0]
        assert request["model"] == "models/gemini-embedding-2"
        assert request["content"] == {"parts": [{"text": "chunk text"}]}
        assert request["taskType"] == "RETRIEVAL_DOCUMENT"
        assert "outputDimensionality" not in request, "unset dim keeps the provider default"
        assert vectors == [[1.0, 0.0, 0.0, 0.0]]

    def test_query_request_uses_the_query_task_type(self, monkeypatch, fake_key) -> None:
        post = _FakePost([_Response(payload=_payload([[1.0, 0.0]]))])
        monkeypatch.setattr(httpx, "post", post)
        _provider().embed_query(["what locks the workspace?"])
        assert post.calls[0]["json"]["requests"][0]["taskType"] == "RETRIEVAL_QUERY"

    def test_configured_output_dimensionality_is_sent(self, monkeypatch, fake_key) -> None:
        post = _FakePost([_Response(payload=_payload([[1.0, 0.0]]))])
        monkeypatch.setattr(httpx, "post", post)
        _provider(dim=1536).embed(["text"])
        assert post.calls[0]["json"]["requests"][0]["outputDimensionality"] == 1536

    def test_batch_is_split_at_the_api_limit_and_order_is_preserved(
        self, monkeypatch, fake_key
    ) -> None:
        post = _FakePost(echo=True)
        monkeypatch.setattr(httpx, "post", post)
        texts = [f"t{i}" for i in range(150)]
        vectors = _provider().embed(texts)

        assert [len(call["json"]["requests"]) for call in post.calls] == [100, 50]
        assert post.calls[1]["json"]["requests"][0]["content"]["parts"][0]["text"] == "t100"
        assert len(vectors) == 150
        firsts = [v[0] for v in vectors]
        assert firsts == sorted(firsts), "output must stay in input order"

    def test_empty_input_makes_no_request(self, monkeypatch, fake_key) -> None:
        post = _FakePost()
        monkeypatch.setattr(httpx, "post", post)
        assert _provider().embed([]) == []
        assert post.calls == []


class TestNormalization:
    def test_truncated_output_is_renormalized(self, monkeypatch, fake_key) -> None:
        # Google returns an unnormalized vector below its full 3072-dim output.
        post = _FakePost([_Response(payload=_payload([[3.0, 4.0]]))])
        monkeypatch.setattr(httpx, "post", post)
        vector = _provider(dim=2).embed(["text"])[0]
        assert vector[0] == pytest.approx(0.6)
        assert vector[1] == pytest.approx(0.8)
        assert math.sqrt(sum(v * v for v in vector)) == pytest.approx(1.0)

    def test_already_unit_vectors_are_left_alone(self, monkeypatch, fake_key) -> None:
        post = _FakePost([_Response(payload=_payload([[0.0, 1.0, 0.0]]))])
        monkeypatch.setattr(httpx, "post", post)
        assert _provider().embed(["text"])[0] == [0.0, 1.0, 0.0]

    def test_zero_vector_is_an_error_not_a_quiet_passthrough(self, monkeypatch, fake_key) -> None:
        post = _FakePost([_Response(payload=_payload([[0.0, 0.0]]))])
        monkeypatch.setattr(httpx, "post", post)
        with pytest.raises(RuntimeError, match="zero-length"):
            _provider().embed(["text"])


class TestRateLimiting:
    def test_429_retries_with_backoff_then_succeeds(self, monkeypatch, fake_key) -> None:
        sleeps: list[float] = []
        monkeypatch.setattr(time, "sleep", sleeps.append)
        post = _FakePost(
            [
                _Response(status_code=429, text="rate limited"),
                _Response(payload=_payload([[1.0, 0.0]])),
            ]
        )
        monkeypatch.setattr(httpx, "post", post)
        vectors = _provider().embed(["text"])
        assert len(post.calls) == 2
        assert sleeps == [1.0]
        assert vectors == [[1.0, 0.0]]

    def test_retry_after_header_is_honoured(self, monkeypatch, fake_key) -> None:
        sleeps: list[float] = []
        monkeypatch.setattr(time, "sleep", sleeps.append)
        post = _FakePost(
            [
                _Response(status_code=429, headers={"retry-after": "7"}, text="slow down"),
                _Response(payload=_payload([[1.0, 0.0]])),
            ]
        )
        monkeypatch.setattr(httpx, "post", post)
        _provider().embed(["text"])
        assert sleeps == [7.0]

    def test_unparseable_retry_after_falls_back_to_backoff(
        self, monkeypatch, fake_key
    ) -> None:
        sleeps: list[float] = []
        monkeypatch.setattr(time, "sleep", sleeps.append)
        post = _FakePost(
            [
                _Response(status_code=429, headers={"retry-after": "not-a-number"}),
                _Response(payload=_payload([[1.0, 0.0]])),
            ]
        )
        monkeypatch.setattr(httpx, "post", post)
        _provider().embed(["text"])
        assert sleeps == [1.0]

    def test_persistent_429_raises_after_the_attempts_are_spent(
        self, monkeypatch, fake_key
    ) -> None:
        sleeps: list[float] = []
        monkeypatch.setattr(time, "sleep", sleeps.append)
        post = _FakePost([_Response(status_code=429, text="rate limited")] * 4)
        monkeypatch.setattr(httpx, "post", post)
        with pytest.raises(RuntimeError) as exc_info:
            _provider().embed(["text"])
        message = str(exc_info.value)
        assert "429" in message and "attempt 4/4" in message
        assert len(post.calls) == 4
        assert sleeps == [1.0, 2.0, 4.0]


class TestTransientTransport:
    """Connection-level failures are transient and are retried exactly like a 429.

    Without this, a TLS EOF mid-run aborted the whole pass: the twoplus indexing
    run stopped at 588 of 1107 files and left a partial index behind.
    """

    def _tls_eof(self) -> httpx.RemoteProtocolError:
        # The shape httpx raises when the TLS stream dies mid-response. The key is
        # attached to the request so that any leak into the raised text would show.
        request = httpx.Request(
            "POST",
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-embedding-2:batchEmbedContents",
            headers={"x-goog-api-key": FAKE_KEY},
        )
        return httpx.RemoteProtocolError(
            "[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol",
            request=request,
        )

    def test_tls_eof_is_retried_then_succeeds(self, monkeypatch, fake_key) -> None:
        sleeps: list[float] = []
        monkeypatch.setattr(time, "sleep", sleeps.append)
        post = _FakePost([self._tls_eof(), _Response(payload=_payload([[1.0, 0.0]]))])
        monkeypatch.setattr(httpx, "post", post)
        assert _provider().embed(["text"]) == [[1.0, 0.0]]
        assert len(post.calls) == 2
        assert sleeps == [1.0]

    def test_read_timeout_and_connect_error_get_the_same_backoff(
        self, monkeypatch, fake_key
    ) -> None:
        sleeps: list[float] = []
        monkeypatch.setattr(time, "sleep", sleeps.append)
        post = _FakePost(
            [
                httpx.ReadTimeout("timed out"),
                httpx.ConnectError("[Errno 111] Connection refused"),
                _Response(payload=_payload([[1.0, 0.0]])),
            ]
        )
        monkeypatch.setattr(httpx, "post", post)
        assert len(_provider().embed(["text"])) == 1
        assert sleeps == [1.0, 2.0]

    def test_transport_failure_exhausts_the_attempts_and_raises(
        self, monkeypatch, fake_key
    ) -> None:
        sleeps: list[float] = []
        monkeypatch.setattr(time, "sleep", sleeps.append)
        attempts = GoogleEmbedProvider._MAX_ATTEMPTS
        post = _FakePost([self._tls_eof()] * attempts)
        monkeypatch.setattr(httpx, "post", post)
        with pytest.raises(RuntimeError) as exc_info:
            _provider().embed(["text"])
        message = str(exc_info.value)
        assert len(post.calls) == attempts
        # One backoff per retry, none after the last attempt.
        assert sleeps == [2.0 ** i for i in range(attempts - 1)]
        assert "gemini-embedding-2" in message
        assert "UNEXPECTED_EOF_WHILE_READING" in message
        assert FAKE_KEY not in message, "the key must never reach the error text"
        assert FAKE_KEY not in post.calls[0]["url"]

    def test_backoff_chain_is_capped_at_sixty_seconds(self, monkeypatch, fake_key) -> None:
        sleeps: list[float] = []
        monkeypatch.setattr(time, "sleep", sleeps.append)
        attempts = 8
        monkeypatch.setattr(GoogleEmbedProvider, "_MAX_ATTEMPTS", attempts)
        post = _FakePost([httpx.ConnectError("connection reset by peer")] * attempts)
        monkeypatch.setattr(httpx, "post", post)
        with pytest.raises(RuntimeError, match="failed"):
            _provider().embed(["text"])
        assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0]


class TestPacing:
    def test_budget_defaults_to_the_measured_free_tier_rate(self, monkeypatch) -> None:
        monkeypatch.delenv("VECTR_EMBED_CHARS_PER_SECOND", raising=False)
        assert types_mod._embed_chars_per_second() == 1200.0

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("5000", 5000.0),  # a paid quota tier
            ("70", 70.0),  # inside the clamp
            ("1", 50.0),  # absurdly slow: clamped up, indexing must not stall
            ("999999", 10000.0),  # absurdly fast: clamped down to spare the quota
            ("0", 1200.0),  # zero would divide by zero: falls back
            ("-3", 1200.0),
            ("not-a-number", 1200.0),
            ("nan", 50.0),  # a NaN rate would make every pause NaN: clamped to the floor
            ("inf", 10000.0),
            ("", 1200.0),  # set-but-empty is not a rate
        ],
    )
    def test_env_rate_is_parsed_and_clamped(self, monkeypatch, raw, expected) -> None:
        monkeypatch.setenv("VECTR_EMBED_CHARS_PER_SECOND", raw)
        assert types_mod._embed_chars_per_second() == expected

    def test_batches_are_spaced_by_their_own_payload_size(self, monkeypatch, fake_key) -> None:
        sleeps: list[float] = []
        monkeypatch.setattr(time, "sleep", sleeps.append)
        monkeypatch.setenv("VECTR_EMBED_CHARS_PER_SECOND", "1000")
        post = _FakePost(echo=True)
        monkeypatch.setattr(httpx, "post", post)
        texts = [f"t{i}" for i in range(150)]
        assert len(_provider().embed(texts)) == 150
        assert [len(call["json"]["requests"]) for call in post.calls] == [100, 50]
        # One pause, for the batch that has a follower: 280 chars at 1000 chars/s.
        assert sleeps == [sum(len(t) for t in texts[:100]) / 1000.0]

    def test_pacing_pause_is_capped_so_a_pass_cannot_stall(self, monkeypatch, fake_key) -> None:
        sleeps: list[float] = []
        monkeypatch.setattr(time, "sleep", sleeps.append)
        monkeypatch.setenv("VECTR_EMBED_CHARS_PER_SECOND", "50")  # the clamp floor
        post = _FakePost(echo=True)
        monkeypatch.setattr(httpx, "post", post)
        texts = [f"t{'0' * 400}{i}" for i in range(150)]  # ~40k chars in the first batch
        assert len(_provider().embed(texts)) == 150
        assert sleeps == [60.0]


class TestErrorPaths:
    def test_error_response_reports_status_and_body_but_never_the_key(
        self, monkeypatch, fake_key
    ) -> None:
        post = _FakePost(
            [
                _Response(
                    status_code=400,
                    text='{"error":{"message":"Model `models/nope` not found"}}',
                )
            ]
        )
        monkeypatch.setattr(httpx, "post", post)
        with pytest.raises(RuntimeError) as exc_info:
            _provider("nope").embed(["text"])
        message = str(exc_info.value)
        assert "HTTP 400" in message and "not found" in message
        assert FAKE_KEY not in message

    def test_embedding_count_mismatch_is_reported(self, monkeypatch, fake_key) -> None:
        post = _FakePost([_Response(payload=_payload([[1.0, 0.0]]))])
        monkeypatch.setattr(httpx, "post", post)
        with pytest.raises(RuntimeError, match="1 embeddings for 2 texts"):
            _provider().embed(["first", "second"])

    def test_malformed_embedding_entry_is_reported(self, monkeypatch, fake_key) -> None:
        post = _FakePost([_Response(payload={"embeddings": [{"foo": 1}]})])
        monkeypatch.setattr(httpx, "post", post)
        with pytest.raises(RuntimeError, match="malformed embedding"):
            _provider().embed(["text"])

    def test_non_json_success_body_is_reported(self, monkeypatch, fake_key) -> None:
        class _BrokenJsonResponse(_Response):
            def json(self):
                raise ValueError("not json")

        post = _FakePost([_BrokenJsonResponse()])
        monkeypatch.setattr(httpx, "post", post)
        with pytest.raises(ValueError, match="not json"):
            _provider().embed(["text"])

    def test_transport_failure_becomes_a_clear_error(self, monkeypatch, fake_key) -> None:
        def _boom(*_args, **_kwargs):
            raise httpx.ConnectError("connection refused")

        sleeps: list[float] = []
        monkeypatch.setattr(time, "sleep", sleeps.append)
        monkeypatch.setattr(httpx, "post", _boom)
        with pytest.raises(RuntimeError, match="gemini-embedding-2.*failed"):
            _provider().embed(["text"])
        assert sleeps == [1.0, 2.0, 4.0]


class TestApiKeyResolution:
    def test_missing_key_names_both_channels_and_leaks_nothing(
        self, monkeypatch, no_home_secrets
    ) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(RuntimeError) as exc_info:
            _provider()
        message = str(exc_info.value)
        assert "GEMINI_API_KEY" in message
        assert str(Path("home") / ".vectr" / "secrets.json") in message

    def test_env_key_wins_over_the_secrets_file(self, monkeypatch, no_home_secrets) -> None:
        secrets = no_home_secrets / ".vectr" / "secrets.json"
        secrets.parent.mkdir(parents=True)
        secrets.write_text('{"gemini_api_key": "from-file"}', encoding="utf-8")
        secrets.chmod(0o600)
        monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
        post = _FakePost([_Response(payload=_payload([[1.0, 0.0]]))])
        monkeypatch.setattr(httpx, "post", post)
        _provider().embed(["text"])
        assert post.calls[0]["headers"]["x-goog-api-key"] == FAKE_KEY

    def test_secrets_file_is_the_fallback(self, monkeypatch, no_home_secrets) -> None:
        secrets = no_home_secrets / ".vectr" / "secrets.json"
        secrets.parent.mkdir(parents=True)
        secrets.write_text('{"gemini_api_key": "from-file"}', encoding="utf-8")
        secrets.chmod(0o600)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        post = _FakePost([_Response(payload=_payload([[1.0, 0.0]]))])
        monkeypatch.setattr(httpx, "post", post)
        _provider().embed(["text"])
        assert post.calls[0]["headers"]["x-goog-api-key"] == "from-file"

    def test_group_readable_secrets_file_is_warned_about(
        self, monkeypatch, no_home_secrets, caplog
    ) -> None:
        secrets = no_home_secrets / ".vectr" / "secrets.json"
        secrets.parent.mkdir(parents=True)
        secrets.write_text('{"gemini_api_key": "from-file"}', encoding="utf-8")
        secrets.chmod(0o644)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        post = _FakePost([_Response(payload=_payload([[1.0, 0.0]]))])
        monkeypatch.setattr(httpx, "post", post)
        with caplog.at_level(logging.WARNING, logger="vectr.embed"):
            _provider().embed(["text"])
        assert any("chmod 600" in record.message for record in caplog.records)
        assert "from-file" not in " ".join(record.message for record in caplog.records)

    def test_secrets_file_without_a_gemini_key_raises(self, monkeypatch, no_home_secrets) -> None:
        secrets = no_home_secrets / ".vectr" / "secrets.json"
        secrets.parent.mkdir(parents=True)
        secrets.write_text('{"other_key": "x"}', encoding="utf-8")
        secrets.chmod(0o600)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="gemini_api_key"):
            _provider()

    def test_malformed_secrets_file_raises(self, monkeypatch, no_home_secrets) -> None:
        secrets = no_home_secrets / ".vectr" / "secrets.json"
        secrets.parent.mkdir(parents=True)
        secrets.write_text("definitely not json", encoding="utf-8")
        secrets.chmod(0o600)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="not valid JSON"):
            _provider()

    def test_custom_key_env_name_is_used(self, monkeypatch, no_home_secrets) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("TEAM_GEMINI_KEY", "team-key")
        post = _FakePost([_Response(payload=_payload([[1.0, 0.0]]))])
        monkeypatch.setattr(httpx, "post", post)
        _provider(api_key_env="TEAM_GEMINI_KEY").embed(["text"])
        assert post.calls[0]["headers"]["x-goog-api-key"] == "team-key"


class TestFactoryRouting:
    def test_google_spec_selects_the_provider_and_the_model_comes_from_the_spec(
        self, monkeypatch, fake_key
    ) -> None:
        post = _FakePost([_Response(payload=_payload([[1.0, 0.0]]))])
        monkeypatch.setattr(httpx, "post", post)
        provider = get_embed_provider("google/gemini-embedding-2")
        assert isinstance(provider, GoogleEmbedProvider)
        provider.embed(["text"])
        assert post.calls[0]["url"].endswith("models/gemini-embedding-2:batchEmbedContents")

    def test_factory_passes_the_layered_dimension_and_key_env(
        self, monkeypatch, fake_key
    ) -> None:
        monkeypatch.setattr(types_mod, "_EMBEDDING_API_KEY_ENV", "LAYERED_GEMINI_KEY")
        monkeypatch.setattr(types_mod, "_EMBEDDING_OUTPUT_DIMENSIONALITY", 768)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("LAYERED_GEMINI_KEY", "layered-key")
        post = _FakePost([_Response(payload=_payload([[1.0, 0.0]]))])
        monkeypatch.setattr(httpx, "post", post)
        get_embed_provider("google/gemini-embedding-2").embed(["text"])
        assert post.calls[0]["headers"]["x-goog-api-key"] == "layered-key"
        assert post.calls[0]["json"]["requests"][0]["outputDimensionality"] == 768

    def test_google_spec_without_a_model_name_is_rejected(self, fake_key) -> None:
        with pytest.raises(ValueError, match="must name a model"):
            get_embed_provider("google/")

    def test_other_specs_keep_their_existing_providers(self, monkeypatch, fake_key) -> None:
        # The `google/` branch must not sweep other specs in: `openai/` and a
        # bare (local) spec still reach the providers they reached before.
        class _StandIn:
            def __init__(self, model_name=None, **kwargs):
                self.model_name = model_name

        monkeypatch.setattr(types_mod, "OpenAIEmbedProvider", _StandIn)
        monkeypatch.setattr(types_mod, "LocalEmbedProvider", _StandIn)

        openai_provider = get_embed_provider("openai/text-embedding-3-small")
        assert isinstance(openai_provider, _StandIn)
        assert openai_provider.model_name == "text-embedding-3-small"
        assert isinstance(
            get_embed_provider("ibm-granite/granite-embedding-english-r2"), _StandIn
        )
