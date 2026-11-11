"""Proxy integration tests (UPG-PRO-14/15/16).

Every upstream call goes to a local in-process mock — NEVER the real API. The
injection provider is faked. Covers transparent pass-through (streaming +
non-streaming), tool_use SSE byte-exactness, error passthrough, upstream-down
handling, cache-safe injection, fail-open, key hygiene, response caching, and
concurrency.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json

import httpx
import pytest

from agent.proactive.cache import ResponseCache
from agent.proactive.gate import LedgerStore, ProactiveGate
from agent.proactive.proxy import build_proxy_app
from agent.proactive.settings import ProactiveSettings
from agent.proactive.types import Candidate, InjectionResult
from tests._proactive_upstream import MockUpstream, full_sse_bytes, unreachable_client_factory

_BASE = dict(
    enabled=True, min_similarity=0.35, max_items_per_event=3, max_chars_per_event=800,
    cooldown_items=30, cooldown_ttl_seconds=None,
    matcher_structural_note=True, matcher_semantic_note=True,
    matcher_code_search=False, proxy_host="127.0.0.1", proxy_port=19000,
    proxy_upstream_base_url="http://upstream", proxy_connect_timeout_s=10.0,
    proxy_read_timeout_s=600.0, proxy_inject=True, proxy_inject_budget_ms=40,
    proxy_inject_provider_timeout_fraction=0.8, proxy_inject_provider_timeout_max_s=2.0,
    proxy_exclude_directive_notes=True,
    cache_enabled=False, cache_max_entries=2048, cache_ttl_seconds=0.0,
    cache_similarity_threshold=1.0, response_cache_enabled=False,
    response_cache_ttl_seconds=60.0, response_cache_max_entries=256,
    structural_kinds=("gotcha", "finding", "decision", "operational", "reference"),
    structural_overfetch_multiplier=4, structural_overfetch_ceiling=60,
    structural_score_declared_anchor=1.0, structural_score_declared_trigger=0.95,
    structural_score_gotcha_mention=0.9,
    structural_score_mention=0.6, max_weak_structural_items=1,
)


def _settings(**over):
    d = dict(_BASE)
    d.update(over)
    return ProactiveSettings(**d)


class _Provider:
    def __init__(self, context="PROACTIVE: note #1", delay=0.0, raise_exc=False):
        self.context = context
        self.delay = delay
        self.raise_exc = raise_exc
        self.seen = []

    async def inject(self, window, *, session_id, channel):
        self.seen.append((window, session_id, channel))
        if self.raise_exc:
            raise RuntimeError("provider boom")
        if self.delay:
            await asyncio.sleep(self.delay)
        if not self.context:
            return InjectionResult.empty()
        return InjectionResult(context=self.context, item_count=1,
                               anchor_ids=("note:1",), scores=(0.9,))


def _body():
    return {
        "model": "claude-x",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "explain the lock"}]},
        ],
    }


async def _drive(app, body=None, headers=None, method="POST", path="/v1/messages"):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
        if method == "GET":
            return await client.get(path, headers=headers or {})
        return await client.post(path, content=json.dumps(body or _body()), headers=headers or {})


# -- transparent pass-through ----------------------------------------------

async def test_non_streaming_passthrough():
    up = MockUpstream()
    app = build_proxy_app(_settings(proxy_inject=False), client_factory=up.client_factory())
    resp = await _drive(app)
    assert resp.status_code == 200
    assert resp.json() == up.response_json
    assert up.call_count == 1


async def test_streaming_sse_byte_exact_with_tool_use():
    up = MockUpstream()
    up.mode = "sse"
    app = build_proxy_app(_settings(proxy_inject=False), client_factory=up.client_factory())
    resp = await _drive(app)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.content == full_sse_bytes()  # byte-exact relay incl. tool_use blocks


async def test_error_status_passthrough():
    up = MockUpstream()
    up.mode = "error"
    up.status = 429
    app = build_proxy_app(_settings(proxy_inject=False), client_factory=up.client_factory())
    resp = await _drive(app)
    assert resp.status_code == 429
    assert resp.json()["type"] == "error"


async def test_upstream_unreachable_returns_shaped_error():
    app = build_proxy_app(_settings(proxy_inject=False), client_factory=unreachable_client_factory())
    resp = await _drive(app)
    assert resp.status_code == 502
    body = resp.json()
    assert body["type"] == "error" and body["error"]["type"] == "api_error"


# -- injection --------------------------------------------------------------

async def test_injection_appends_context_cache_safe():
    up = MockUpstream()
    prov = _Provider(context="PROACTIVE: note #1")
    app = build_proxy_app(_settings(), injection_provider=prov, client_factory=up.client_factory())
    resp = await _drive(app)
    assert resp.status_code == 200
    forwarded = up.last_request["json"]
    last_content = forwarded["messages"][-1]["content"]
    # Injected block appended as newest content; original block untouched.
    assert last_content[0] == {"type": "text", "text": "explain the lock"}
    assert last_content[-1] == {"type": "text", "text": "PROACTIVE: note #1"}
    assert app.state.proxy.metrics.as_dict()["injected"] == 1


async def test_injection_empty_forwards_unmodified():
    up = MockUpstream()
    prov = _Provider(context="")  # nothing to inject
    app = build_proxy_app(_settings(), injection_provider=prov, client_factory=up.client_factory())
    await _drive(app)
    forwarded = up.last_request["json"]
    assert forwarded["messages"][-1]["content"] == [{"type": "text", "text": "explain the lock"}]


async def test_injection_provider_error_is_fail_open():
    up = MockUpstream()
    prov = _Provider(raise_exc=True)
    app = build_proxy_app(_settings(), injection_provider=prov, client_factory=up.client_factory())
    resp = await _drive(app)
    assert resp.status_code == 200  # request still went through
    forwarded = up.last_request["json"]
    assert forwarded["messages"][-1]["content"] == [{"type": "text", "text": "explain the lock"}]
    assert app.state.proxy.metrics.as_dict()["inject_bypassed_error"] == 1


async def test_injection_timeout_is_fail_open():
    up = MockUpstream()
    prov = _Provider(context="LATE", delay=0.5)  # 500ms > 20ms budget
    app = build_proxy_app(
        _settings(proxy_inject_budget_ms=20), injection_provider=prov,
        client_factory=up.client_factory(),
    )
    resp = await _drive(app)
    assert resp.status_code == 200
    forwarded = up.last_request["json"]
    assert forwarded["messages"][-1]["content"] == [{"type": "text", "text": "explain the lock"}]
    assert app.state.proxy.metrics.as_dict()["inject_bypassed_error"] == 1


# -- bypass diagnostics (UPG-PROXY-SILENT-BYPASS) ----------------------------


async def test_bypass_error_logs_one_warning_with_exception_and_elapsed_ms(caplog):
    import logging

    up = MockUpstream()
    prov = _Provider(raise_exc=True)
    app = build_proxy_app(_settings(), injection_provider=prov, client_factory=up.client_factory())
    with caplog.at_level(logging.WARNING, logger="agent.proactive.proxy"):
        resp = await _drive(app)
    assert resp.status_code == 200  # original bytes still forwarded (fail-open)
    forwarded = up.last_request["json"]
    assert forwarded["messages"][-1]["content"] == [{"type": "text", "text": "explain the lock"}]
    assert app.state.proxy.metrics.as_dict()["inject_bypassed_error"] == 1

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    msg = warnings[0].getMessage()
    assert "RuntimeError" in msg          # exception class name
    assert "ms" in msg                    # elapsed-ms figure present
    # Metadata only — never the conversation content or the provider's context.
    assert "explain the lock" not in msg
    assert "PROACTIVE" not in msg


async def test_bypass_timeout_logs_one_warning_with_timeout_error_and_elapsed_ms(caplog):
    import logging

    up = MockUpstream()
    prov = _Provider(context="LATE", delay=0.5)  # 500ms > 20ms budget
    app = build_proxy_app(
        _settings(proxy_inject_budget_ms=20), injection_provider=prov,
        client_factory=up.client_factory(),
    )
    with caplog.at_level(logging.WARNING, logger="agent.proactive.proxy"):
        resp = await _drive(app)
    assert resp.status_code == 200
    assert app.state.proxy.metrics.as_dict()["inject_bypassed_error"] == 1

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    msg = warnings[0].getMessage()
    assert "TimeoutError" in msg
    assert "ms" in msg


async def test_skip_branches_log_debug_reason_no_warning(caplog):
    import logging

    up = MockUpstream()
    prov = _Provider(context="")  # provider returns nothing to inject
    app = build_proxy_app(_settings(), injection_provider=prov, client_factory=up.client_factory())
    with caplog.at_level(logging.DEBUG, logger="agent.proactive.proxy"):
        await _drive(app)
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]
    debugs = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
    assert any("empty result" in m for m in debugs)


# -- key hygiene ------------------------------------------------------------

async def test_api_key_forwarded_untouched_never_leaked():
    up = MockUpstream()
    app = build_proxy_app(_settings(proxy_inject=False), client_factory=up.client_factory())
    secret = "sk-ant-SECRETKEY123"
    resp = await _drive(app, headers={"x-api-key": secret, "anthropic-version": "2023-06-01"})
    # Forwarded verbatim to upstream.
    assert up.last_request["headers"].get("x-api-key") == secret
    # Never surfaced in the proxy's own health/metrics output.
    health = await _drive(app, method="GET", path="/__vectr_proxy/health")
    assert secret not in health.text


async def test_upstream_error_body_has_no_key():
    app = build_proxy_app(_settings(proxy_inject=False), client_factory=unreachable_client_factory())
    resp = await _drive(app, headers={"x-api-key": "sk-ant-SECRET"})
    assert "sk-ant-SECRET" not in resp.text


# -- response cache ---------------------------------------------------------

async def test_response_cache_exact_match_serves_from_cache():
    up = MockUpstream()
    cache = ResponseCache(ttl_seconds=60.0)
    app = build_proxy_app(
        _settings(proxy_inject=False, response_cache_enabled=True),
        response_cache=cache, client_factory=up.client_factory(),
    )
    r1 = await _drive(app)
    r2 = await _drive(app)  # byte-identical request
    assert up.call_count == 1              # upstream hit once
    assert r2.content == r1.content        # byte-exact replay
    assert app.state.proxy.metrics.as_dict()["response_cache_hits"] == 1


# -- concurrency ------------------------------------------------------------

async def test_concurrent_requests_all_succeed():
    up = MockUpstream()
    app = build_proxy_app(_settings(proxy_inject=False), client_factory=up.client_factory())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
        results = await asyncio.gather(*[
            client.post("/v1/messages", content=json.dumps(_body())) for _ in range(10)
        ])
    assert all(r.status_code == 200 for r in results)
    assert up.call_count == 10


# -- cooldown session identity (UPG-PROXY-COOLDOWN-SESSION-IDENTITY) --------
#
# These drive REAL cooldown machinery (a real ProactiveGate over a real
# shared LedgerStore), not a stub provider, so the assertions pin actual
# injected content/anchor ids rather than the session_id string alone.


def _lock_candidate(anchor_id="note:1"):
    return Candidate(
        kind="note_structural",
        line="note #1 (gotcha, anchored to resolver.py): drops on scope exit",
        score=1.0,
        anchor_id=anchor_id,
        is_structural=True,
    )


def _messages_body(text):
    return {
        "model": "claude-x",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": [{"type": "text", "text": text}]}],
    }


class _GateBackedProvider:
    """Wraps a REAL `ProactiveGate` (over a REAL, injectable `LedgerStore`)
    so a test exercises actual cooldown/dedup logic end to end -- never a
    stub that only records the session_id it was called with."""

    def __init__(self, gate: ProactiveGate, candidates: list[Candidate]):
        self._gate = gate
        self._candidates = candidates
        self.seen_session_ids: list[str] = []

    async def inject(self, window, *, session_id, channel):
        self.seen_session_ids.append(session_id)
        return self._gate.select(self._candidates, session_id=session_id)


async def test_cooldown_shared_across_different_first_messages_same_proxy_instance():
    """Headline defect repro + fix pin. Two requests carrying DIFFERENT first
    user messages through the SAME VectrProxy instance must share one
    cooldown ledger: the second request must NOT re-inject the note the
    first one just emitted. Pre-fix (session_id = sha256 of the first user
    message), two different first messages produced two different cooldown
    keys and the note injected twice; post-fix both share the proxy
    instance's one stable identity and the second is suppressed."""
    ledger_store = LedgerStore(cooldown_items=30)
    gate = ProactiveGate(
        min_similarity=0.0, max_items_per_event=3, max_chars_per_event=800,
        cooldown_items=30, max_weak_structural_items=1, ledger_store=ledger_store,
    )
    provider = _GateBackedProvider(gate, [_lock_candidate()])
    up = MockUpstream()
    app = build_proxy_app(_settings(), injection_provider=provider, client_factory=up.client_factory())

    await _drive(app, body=_messages_body("explain the lock"))
    await _drive(app, body=_messages_body("totally different opening message entirely"))

    assert up.call_count == 2
    first_content = up.requests[0]["json"]["messages"][-1]["content"]
    second_content = up.requests[1]["json"]["messages"][-1]["content"]

    # First request: nothing seen yet for this identity -> the note injects.
    assert len(first_content) == 2
    assert "drops on scope exit" in first_content[-1]["text"]

    # Second request, SAME proxy instance, DIFFERENT first user message:
    # cooldown suppresses the repeat -- the fix pin. Forwarded unmodified.
    assert second_content == [{"type": "text", "text": "totally different opening message entirely"}]

    # Both requests carried the SAME session identity into the gate/ledger.
    assert len(provider.seen_session_ids) == 2
    assert provider.seen_session_ids[0] == provider.seen_session_ids[1]
    assert provider.seen_session_ids[0] == app.state.proxy._instance_id
    assert app.state.proxy._instance_id.startswith("proxy-")


async def test_different_proxy_instances_get_independent_cooldown_ledgers():
    """Inverse isolation. Two separately-constructed VectrProxy instances
    sharing the same LedgerStore get DIFFERENT identities, hence
    independent ledgers -- both inject the same note, even for the
    byte-identical first message, because cooldown keys off the proxy
    instance, never off message content."""
    ledger_store = LedgerStore(cooldown_items=30)
    gate = ProactiveGate(
        min_similarity=0.0, max_items_per_event=3, max_chars_per_event=800,
        cooldown_items=30, max_weak_structural_items=1, ledger_store=ledger_store,
    )
    candidate = _lock_candidate()

    provider_a = _GateBackedProvider(gate, [candidate])
    up_a = MockUpstream()
    app_a = build_proxy_app(_settings(), injection_provider=provider_a, client_factory=up_a.client_factory())

    provider_b = _GateBackedProvider(gate, [candidate])
    up_b = MockUpstream()
    app_b = build_proxy_app(_settings(), injection_provider=provider_b, client_factory=up_b.client_factory())

    assert app_a.state.proxy._instance_id != app_b.state.proxy._instance_id

    await _drive(app_a, body=_messages_body("explain the lock"))
    await _drive(app_b, body=_messages_body("explain the lock"))  # identical text

    content_a = up_a.requests[0]["json"]["messages"][-1]["content"]
    content_b = up_b.requests[0]["json"]["messages"][-1]["content"]
    assert "drops on scope exit" in content_a[-1]["text"]
    assert "drops on scope exit" in content_b[-1]["text"]


async def test_ledger_bound_under_one_stable_proxy_identity():
    """Ledger bound under a stable key. Many distinct anchor ids injected
    through ONE stable proxy identity -> LedgerStore holds exactly 1
    session entry, and that session's ring stays capped at cooldown_items."""
    cooldown_items = 5
    ledger_store = LedgerStore(cooldown_items=cooldown_items)
    gate = ProactiveGate(
        min_similarity=0.0, max_items_per_event=1, max_chars_per_event=800,
        cooldown_items=cooldown_items, max_weak_structural_items=1, ledger_store=ledger_store,
    )

    class _RotatingProvider:
        """Returns a distinct, never-before-seen anchor id on every call, so
        every call is eligible to inject and only the ledger's own capacity
        bounds what gets retained."""

        def __init__(self):
            self.n = 0

        async def inject(self, window, *, session_id, channel):
            self.n += 1
            candidate = Candidate(
                kind="note_structural", line=f"note #{self.n}: distinct anchor",
                score=1.0, anchor_id=f"note:{self.n}", is_structural=True,
            )
            return gate.select([candidate], session_id=session_id)

    up = MockUpstream()
    app = build_proxy_app(_settings(), injection_provider=_RotatingProvider(), client_factory=up.client_factory())

    for i in range(20):
        await _drive(app, body=_messages_body(f"message {i}"))

    assert up.call_count == 20
    assert len(ledger_store._ledgers) == 1
    session_ledger = next(iter(ledger_store._ledgers.values()))
    assert len(session_ledger._ring) == cooldown_items


# -- startup banner (UPG-PROXY-HIDDEN-MASTER-SWITCH) -------------------------

def test_banner_injection_off_single_line():
    from main import _render_injection_lines
    lines = _render_injection_lines(False, "http://localhost:8766", None)
    assert lines == ["  Injection : off (transparent pass-through)"]


def test_banner_daemon_unreachable_warns_fail_open():
    from main import _render_injection_lines
    lines = _render_injection_lines(True, "http://localhost:8766", None)
    assert len(lines) == 1
    assert "WARNING" in lines[0] and "fail open" in lines[0]


def test_banner_proxy_channel_consent_with_master_switch_off():
    from main import _render_injection_lines
    lines = _render_injection_lines(True, "http://localhost:8766", {"proactive_enabled": False})
    assert "launch consent" in lines[0]
    assert len(lines) == 2 and "ambient (hook) injection is off" in lines[1]


def test_banner_master_switch_on_single_line():
    from main import _render_injection_lines
    lines = _render_injection_lines(True, "http://localhost:8766", {"proactive_enabled": True})
    assert len(lines) == 1 and "launch consent" in lines[0]
