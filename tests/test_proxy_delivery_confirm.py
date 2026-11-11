"""Proxy delivery confirmation (UPG-PROXY-APPEND-BURNS-COOLDOWN + the audit
companion UPG-PROXY-AUDIT-CLAIMS-UNDELIVERED-INJECTION).

The defect these lock down: a real editor's FIRST request ends with a
`system`-role message, so `append_context_block` refuses it (deliberate
fail-open) — yet the retrieval that produced the block had already charged the
note's cooldown slot daemon-side, and the proxy's session id is
process-scoped, so the slot stayed spent for the life of the process. Requests
2..N WERE appendable and carried the file anchor, and were suppressed. The
durable audit log meanwhile recorded PROACTIVE_INJECT for every one of those
undelivered blocks.

The fix has two halves, and each has its own mutation check below:
  (a) decide appendability BEFORE spending a retrieval, so an unappendable
      request costs nothing and the note survives to the next request;
  (b) charge the cooldown ledger (and emit PROACTIVE_INJECT) on CONFIRMED
      delivery, carried back as an opaque token on the next call.

Every upstream call goes to an in-process mock; the daemon is the real
FastAPI app over an ASGI transport backed by a real memory-only VectrService.
Nothing here touches the network.
"""
from __future__ import annotations

import json
import logging
from unittest.mock import patch

import httpx
import pytest

from agent.proactive.provider import DaemonInjectionProvider
from agent.proactive.proxy import build_proxy_app
from agent.proactive.request_window import (
    append_context_block,
    cache_prefix_signature,
    can_append_context,
)
from agent.proactive.settings import ProactiveSettings
from agent.proactive.types import InjectionResult
from tests._proactive_upstream import MockUpstream

_BASE = dict(
    enabled=True, min_similarity=0.35, max_items_per_event=3, max_chars_per_event=800,
    cooldown_items=30, cooldown_ttl_seconds=None,
    matcher_structural_note=True, matcher_semantic_note=True,
    matcher_code_search=False, proxy_host="127.0.0.1", proxy_port=19000,
    proxy_upstream_base_url="http://upstream", proxy_connect_timeout_s=10.0,
    proxy_read_timeout_s=600.0, proxy_inject=True, proxy_inject_budget_ms=20000,
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


@pytest.fixture(autouse=True)
def _reset_audit_logger():
    """`_get_audit_logger()` caches handlers on the process-level singleton
    `logging.getLogger("vectr.audit")`, so a case that ran with the opt-in unset
    would otherwise pin a NullHandler for the rest of the session. Same hygiene
    pattern as tests/test_proactive_audit.py."""
    logging.getLogger("vectr.audit").handlers.clear()
    yield
    logging.getLogger("vectr.audit").handlers.clear()


def _service(tmp_path, monkeypatch, **env):
    """A real memory-only VectrService with the dummy embedder (no model
    download, no network) — the same construction the service tests use."""
    from agent import indexer as idx_module
    from tests.conftest import _DummyEmbedProvider

    monkeypatch.setattr(idx_module, "get_embed_provider", lambda _: _DummyEmbedProvider())
    monkeypatch.delenv("VECTR_BIND_HOST", raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    with patch("integrations.vscode_bridge.configure_all"), \
         patch("integrations.workspace_detect.find_workspace_root", return_value=str(tmp_path)), \
         patch.dict("os.environ", {"VECTR_DB_DIR": str(tmp_path / "db")}):
        from app.service import VectrService
        return VectrService(workspace_root=str(tmp_path), memory_only=True)


def _daemon_client(svc) -> httpx.AsyncClient:
    """The real `/v1/proactive` route over ASGI, backed by `svc`."""
    from api import app as api_app

    api_app.state.service = svc
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_app), base_url="http://daemon"
    )


# -- request shapes ---------------------------------------------------------
#
# Skeletons of the shapes an editor actually puts on the wire, captured from a
# live session through a recording proxy. Only the structure matters here (roles,
# block layout, where `cache_control` sits); the prose is stand-in text.

def _read_tool_use(path: str) -> dict:
    return {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": path}}


def _edit_tool_use(path: str) -> dict:
    """An Edit-shaped tool call (UPG-PROXY-INJECT-SINGLE-TURN) — as opposed to
    `_read_tool_use`, this is one of the genuine edit-tool names
    `assemble_window()` tracks into `ProactiveWindow.edited_file_paths`."""
    return {
        "type": "tool_use", "id": "toolu_2", "name": "Edit",
        "input": {"file_path": path, "old_string": "x", "new_string": "y"},
    }


def _first_request(path: str) -> dict:
    """The captured FIRST request of a session: roles are ['user', 'system'],
    and the trailing harness `system` message carries the LAST cache
    breakpoint. `messages[-1]` is not a user turn, so no block can land on it.
    """
    return {
        "model": "claude-x",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "why does the lock drop early"},
                _read_tool_use(path),
            ]},
            {"role": "system", "content": [
                {"type": "text",
                 "text": "<system-reminder>harness preamble</system-reminder>",
                 "cache_control": {"type": "ephemeral"}},
            ]},
        ],
    }


def _followup_request(path: str, n: int, *, edited: bool = False) -> dict:
    """A captured FOLLOW-UP request: the trailing message IS a user turn and
    carries the last cache breakpoint on its own final block.

    `edited=True` (UPG-PROXY-INJECT-SINGLE-TURN, default False so every
    other existing use of this helper is unchanged) swaps the assistant's
    tool call for an Edit-type one, so `assemble_window()` reports this
    request's window as having actually edited `path` — the signal a
    declared-anchor note's event-anchored retirement waits for."""
    action = _edit_tool_use(path) if edited else _read_tool_use(path)
    return {
        "model": "claude-x",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "why does the lock drop early"}]},
            {"role": "assistant", "content": [action]},
            {"role": "user", "content": [
                {"type": "text", "text": f"follow-up {n}"},
                _read_tool_use(path),
            ]},
        ],
    }


def _mark_last_block_cached(body: dict) -> dict:
    last = body["messages"][-1]["content"][-1]
    last["cache_control"] = {"type": "ephemeral"}
    return body


async def _post(app, body):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
        return await client.post("/v1/messages", content=json.dumps(body))


def _forwarded_texts(up: MockUpstream) -> list[str]:
    """The full JSON each request arrived upstream as — what the model saw."""
    return [json.dumps(r["json"]) for r in up.requests]


@pytest.fixture
def anchored(tmp_path, monkeypatch):
    """A real service holding one gotcha anchored to a real file, plus that
    file's path."""
    svc = _service(tmp_path, monkeypatch)
    f = tmp_path / "resolver.py"
    f.write_text("# resolver\n")
    svc.remember("the workspace lock drops on scope exit", kind="gotcha",
                 anchors=["resolver.py"])
    return svc, str(f)


# -- 1. headline: the four-request replay -----------------------------------

async def test_four_request_replay_delivers_on_first_appendable_request(anchored):
    """End-to-end through the real proxy, the real provider, the real
    `/v1/proactive` route and a real service: the note that the pre-fix build
    burned on the unappendable first request is instead DELIVERED on the second,
    and is then correctly suppressed for the rest of the process.

    Follow-up #2 is the one that also EDITS the anchored file
    (UPG-PROXY-INJECT-SINGLE-TURN: event-anchored retirement replaces the
    fixed cooldown for a declared-anchor note) — the same request that
    delivers it is the one that retires it, so requests #3/#4 (read-only)
    still see it suppressed, exactly as before that feature existed. A
    read-only follow-up alone would no longer retire the note (see
    test_declared_anchor_note_survives_reads_and_retires_on_edit in
    tests/test_proactive_service.py for that half of the contract)."""
    svc, path = anchored
    up = MockUpstream()
    async with _daemon_client(svc) as client:
        provider = DaemonInjectionProvider("http://daemon", client=client)
        app = build_proxy_app(
            _settings(), injection_provider=provider, client_factory=up.client_factory()
        )
        await _post(app, _first_request(path))
        for n in (2, 3, 4):
            edited = n == 2
            await _post(app, _mark_last_block_cached(_followup_request(path, n, edited=edited)))

    assert up.call_count == 4, "every request must reach upstream (fail-open)"
    seen = _forwarded_texts(up)
    carried = ["scope exit" in body for body in seen]
    # The pre-fix build produced [False, False, False, False]: req#1 spent the
    # note, req#2..4 were suppressed by the cooldown slot it burned.
    assert carried == [False, True, False, False], (
        "the note must survive the unappendable first request and land on the "
        f"first appendable one, got {carried}"
    )
    # And it landed exactly once — a delivered note is charged, not re-sent.
    assert sum(carried) == 1


async def test_first_request_is_forwarded_byte_identical(anchored):
    """The unappendable request is not merely uninjected — it is forwarded
    completely untouched, which is what makes skipping it free."""
    svc, path = anchored
    up = MockUpstream()
    body = _first_request(path)
    async with _daemon_client(svc) as client:
        provider = DaemonInjectionProvider("http://daemon", client=client)
        app = build_proxy_app(
            _settings(), injection_provider=provider, client_factory=up.client_factory()
        )
        await _post(app, body)
    assert up.requests[0]["json"] == body
    health = [r for r in up.requests if r["path"].endswith("health")]
    assert not health


# -- 2. executable cache-safety invariant -----------------------------------

def test_appending_to_a_non_trailing_user_message_would_break_the_cache_prefix():
    """Why the fix defers rather than reaching back for the last user message.

    On the captured first request the trailing `system` message holds the last
    cache breakpoint, so the last USER message sits INSIDE the protected
    prefix. Appending there rewrites cached bytes and invalidates the prompt
    cache for the whole conversation. This asserts that directly against the
    real `cache_prefix_signature`, so the rejected alternative can never be
    reintroduced silently.
    """
    body = _first_request("/x/resolver.py")
    before = cache_prefix_signature(body)

    # The real code path refuses this shape outright.
    assert can_append_context(body) is False
    _unchanged, ok = append_context_block(body, "PROACTIVE: note #1")
    assert ok is False
    assert cache_prefix_signature(body) == before

    # The rejected alternative (A1): append to the last USER message instead.
    import copy
    mutated = copy.deepcopy(body)
    for msg in reversed(mutated["messages"]):
        if msg["role"] == "user":
            msg["content"].append({"type": "text", "text": "PROACTIVE: note #1"})
            break
    assert cache_prefix_signature(mutated) != before, (
        "appending to a user message inside the cached prefix MUST be detected "
        "as a prefix change — this is the measured reason A1 was rejected"
    )


def test_append_on_a_followup_request_leaves_the_cache_prefix_identical():
    """The delivery the fix actually makes is cache-safe: the block lands after
    the last breakpoint, so the protected region is byte-identical."""
    body = _mark_last_block_cached(_followup_request("/x/resolver.py", 2))
    before = cache_prefix_signature(body)
    assert can_append_context(body) is True
    new_body, ok = append_context_block(body, "PROACTIVE: note #1")
    assert ok is True
    assert cache_prefix_signature(new_body) == before
    assert cache_prefix_signature(body) == before  # original not mutated


def test_can_append_context_reads_only_shape_never_content():
    """Structural predicate (no-query-heuristics rule): identical shapes decide
    identically regardless of what the conversation is about."""
    a = {"messages": [{"role": "user", "content": [{"type": "text", "text": "deploy the release"}]}]}
    b = {"messages": [{"role": "user", "content": [{"type": "text", "text": "zzzz"}]}]}
    assert can_append_context(a) == can_append_context(b) is True
    # Same content, trailing role flipped -> the ONLY thing that changes it.
    c = {"messages": [
        {"role": "user", "content": [{"type": "text", "text": "deploy the release"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "deploy the release"}]},
    ]}
    assert can_append_context(c) is False
    assert can_append_context({"messages": []}) is False
    assert can_append_context({}) is False


# -- 3. mutation checks: each half is load-bearing ---------------------------

async def test_mutation_without_the_appendability_precheck_the_retrieval_is_wasted(anchored):
    """Half (a). Removing the pre-check (simulated by forcing it True) makes the
    proxy spend a real retrieval on a request that then cannot carry the block:
    the daemon parks a pending delivery nobody will ever confirm."""
    svc, path = anchored
    up = MockUpstream()
    async with _daemon_client(svc) as client:
        provider = DaemonInjectionProvider("http://daemon", client=client)
        app = build_proxy_app(
            _settings(), injection_provider=provider, client_factory=up.client_factory()
        )
        # With the pre-check: the daemon is never even asked.
        await _post(app, _first_request(path))
        assert svc._proactive_pending == {}

        with patch("agent.proactive.proxy.can_append_context", return_value=True):
            await _post(app, _first_request(path))
        # Mutation observed: a retrieval happened and was parked, unconfirmed.
        assert len(svc._proactive_pending) == 1

    # Under the OLD charge-at-retrieval semantics that same wasted retrieval
    # would have been a charge. Deferred, the slot is still unspent, so the
    # next appendable request can still get the note.
    assert svc.proactive_context(
        text="", file_paths=[path], session_id="s-live", channel="proxy",
    )["item_count"] == 1


async def test_mutation_charging_at_retrieval_burns_the_note(anchored):
    """Half (b). `defer_charge=False` is the pre-fix behaviour: the note is
    charged the moment it is retrieved, delivered or not — so the very next
    request for the same file gets nothing.

    `edited_file_paths=[path]` on both calls (UPG-PROXY-INJECT-SINGLE-TURN):
    this note is declared-anchored, so it would otherwise be exempted from
    cooldown charging until its file is edited — an orthogonal feature to the
    one under test here. Supplying the edit signal isolates charge-at-
    retrieval mechanics exactly as this test intends."""
    svc, path = anchored
    first = svc.proactive_context(
        text="", file_paths=[path], edited_file_paths=[path],
        session_id="s1", channel="proxy", defer_charge=False,
    )
    assert first["item_count"] == 1
    assert "delivery_token" not in first, "no token when the charge already happened"
    second = svc.proactive_context(
        text="", file_paths=[path], edited_file_paths=[path],
        session_id="s1", channel="proxy", defer_charge=False,
    )
    assert second["item_count"] == 0, "pre-fix: burned at retrieval"


async def test_deferred_charge_lands_only_on_confirm(anchored):
    """Half (b), the fixed path. Retrieval alone does not charge; the note stays
    available until a token comes back, and is suppressed immediately after.

    `edited_file_paths=[path]` (UPG-PROXY-INJECT-SINGLE-TURN, see the sibling
    mutation test above for why): keeps this test isolated to deferred-charge
    mechanics rather than event-anchored retirement."""
    svc, path = anchored
    first = svc.proactive_context(
        text="", file_paths=[path], edited_file_paths=[path],
        session_id="s1", channel="proxy", defer_charge=True,
    )
    assert first["item_count"] == 1
    assert first["charge_deferred"] is True
    token = first["delivery_token"]
    assert token

    # Unconfirmed: still selectable (the slot was never spent).
    again = svc.proactive_context(
        text="", file_paths=[path], edited_file_paths=[path],
        session_id="s1", channel="proxy", defer_charge=True,
    )
    assert again["item_count"] == 1

    # Confirm the FIRST delivery; the charge is applied before this call's own
    # selection, so the note is already suppressed by the time the gate looks.
    after = svc.proactive_context(
        text="", file_paths=[path], edited_file_paths=[path], session_id="s1", channel="proxy",
        defer_charge=True, confirm_token=token,
    )
    assert after["item_count"] == 0, "confirmed delivery must charge before selection"


# -- 4. the proxy path never allocates a turn ledger -------------------------

async def test_proxy_path_never_allocates_a_turn_ledger_including_confirm(anchored):
    """The proxy's session id is process-scoped, not the editor's. Allocating a
    turn-ledger entry for it would evict a live editor session from the bounded
    LRU. Selection was already lookup-only; the NEW confirm/charge step must be
    too."""
    svc, path = anchored
    assert svc._turn_ledgers == {}
    first = svc.proactive_context(
        text="", file_paths=[path], session_id="proxy-proc-1", channel="proxy",
        defer_charge=True,
    )
    token = first["delivery_token"]
    assert token
    assert svc._turn_ledgers == {}, "selection must not allocate"

    def _boom(_self, _session_id):  # pragma: no cover - only fires on regression
        raise AssertionError("confirm path must not call _turn_ledger_for")

    from app.service import VectrService
    with patch.object(VectrService, "_turn_ledger_for", _boom):
        assert svc._confirm_proactive_delivery(token) is True
    assert svc._turn_ledgers == {}, "confirm must not allocate either"


# -- 5. backward compatibility, both directions ------------------------------

async def test_old_caller_against_new_daemon_keeps_charging_at_retrieval(anchored):
    """A caller that never sends `defer_charge` (the hook/trigger-engine
    channel) must be byte-identical to before: charged at retrieval, no token,
    no pending entry.

    `edited_file_paths=[path]` (UPG-PROXY-INJECT-SINGLE-TURN, see the mutation
    tests above): this note is declared-anchored, so it would otherwise stay
    eligible until its file is edited — an orthogonal feature. Supplying the
    edit signal keeps this test's subject exactly what its docstring says:
    charge-at-retrieval backward compatibility."""
    svc, path = anchored
    out = svc.proactive_context(
        text="", file_paths=[path], edited_file_paths=[path], session_id="h1", channel="hook",
    )
    assert out["item_count"] == 1
    assert set(out) == {"context", "item_count", "anchor_ids", "scores"}
    assert svc._proactive_pending == {}
    repeat = svc.proactive_context(
        text="", file_paths=[path], edited_file_paths=[path], session_id="h1", channel="hook",
    )
    assert repeat["item_count"] == 0, "the hook channel's cooldown semantics are unchanged"


async def test_new_provider_against_old_daemon_response_confirms_nothing():
    """Feature detection: a daemon that predates deferred charging returns no
    `charge_deferred`, so the provider carries no token and confirms nothing —
    that daemon's retrieval-time charge simply stands."""
    calls: list[dict] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={
            # Exactly the old four-field response model.
            "context": "PROACTIVE: note #1", "item_count": 1,
            "anchor_ids": ["note:1"], "scores": [0.9], "processing_ms": 1,
        })

    from agent.proactive.types import ProactiveWindow
    client = httpx.AsyncClient(transport=httpx.MockTransport(_handler), base_url="http://old")
    provider = DaemonInjectionProvider("http://old", client=client)
    result = await provider.inject(
        ProactiveWindow(text="lock", file_paths=["/x/resolver.py"]),
        session_id="s1", channel="proxy",
    )
    await client.aclose()
    assert result.context == "PROACTIVE: note #1"
    assert result.delivery_token == "", "no token to confirm against an old daemon"
    # The new fields are still SENT; an old daemon's pydantic model ignores them.
    assert calls[0]["defer_charge"] is True
    assert calls[0]["confirm_token"] == ""


async def test_new_daemon_ignores_unknown_fields_from_any_caller(anchored):
    """The other direction over the real route: extra/absent fields never error
    the caller."""
    svc, _path = anchored
    async with _daemon_client(svc) as client:
        resp = await client.post("/v1/proactive", json={
            "text": "lock", "file_paths": [], "symbols": [],
            "session_id": "s1", "channel": "proxy",
            "some_future_field": {"nested": True},
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["delivery_token"] == ""
    assert data["charge_deferred"] is False


# -- 6. the audit log stops asserting undelivered injections -----------------

def _audit_lines(path, event: str) -> list[str]:
    """Lines for one event. The handler's formatter prefixes an asctime, so the
    event name is matched as a substring, not a prefix (audit.py's own format is
    `<ts> EVENT k=v ...`)."""
    if not path.exists():
        return []
    return [ln for ln in path.read_text().splitlines() if event in ln]


async def test_audit_separates_retrieval_from_confirmed_delivery(tmp_path, monkeypatch):
    """UPG-PROXY-AUDIT-CLAIMS-UNDELIVERED-INJECTION: selection writes
    PROACTIVE_RETRIEVE; only a confirmed delivery writes PROACTIVE_INJECT."""
    log_file = tmp_path / "audit.log"
    svc = _service(tmp_path, monkeypatch, VECTR_PROACTIVE="1", VECTR_AUDIT_LOG=str(log_file))
    f = tmp_path / "resolver.py"
    f.write_text("# resolver\n")
    svc.remember("the lock drops on scope exit", kind="gotcha", anchors=["resolver.py"])

    out = svc.proactive_context(
        text="", file_paths=[str(f)], session_id="s1", channel="proxy", defer_charge=True,
    )
    token = out["delivery_token"]
    assert len(_audit_lines(log_file, "PROACTIVE_RETRIEVE")) == 1
    assert _audit_lines(log_file, "PROACTIVE_INJECT") == [], (
        "a retrieved-but-unconfirmed block must NOT be logged as injected"
    )

    svc._confirm_proactive_delivery(token)
    inject = _audit_lines(log_file, "PROACTIVE_INJECT")
    assert len(inject) == 1
    retrieve = _audit_lines(log_file, "PROACTIVE_RETRIEVE")[0]
    # Both lines describe the same delivery, and are joinable by token.
    assert f"token={token}" in retrieve
    assert f"token={token}" in inject[0]
    assert "items=1" in inject[0]
    # Metadata only: never the note body.
    assert "scope exit" not in retrieve
    assert "scope exit" not in inject[0]


async def test_status_counts_only_confirmed_deliveries(tmp_path, monkeypatch):
    """`proactive_injection_counts` follows the same truth as the log."""
    svc = _service(tmp_path, monkeypatch, VECTR_PROACTIVE="1")
    f = tmp_path / "resolver.py"
    f.write_text("# resolver\n")
    svc.remember("the lock drops on scope exit", kind="gotcha", anchors=["resolver.py"])

    before = dict(svc._proactive_injection_counts)
    out = svc.proactive_context(
        text="", file_paths=[str(f)], session_id="s1", channel="proxy", defer_charge=True,
    )
    assert out["item_count"] == 1
    assert dict(svc._proactive_injection_counts) == before, "retrieval is not a delivery"
    svc._confirm_proactive_delivery(out["delivery_token"])
    assert svc._proactive_injection_counts["proxy"] == before.get("proxy", 0) + 1


async def test_audit_log_stays_opt_in(tmp_path, monkeypatch):
    """VECTR_AUDIT_LOG contract preserved exactly: unset means no file, no
    directory, and no raise — on the new PROACTIVE_RETRIEVE path too."""
    monkeypatch.delenv("VECTR_AUDIT_LOG", raising=False)
    svc = _service(tmp_path, monkeypatch, VECTR_PROACTIVE="1")
    f = tmp_path / "resolver.py"
    f.write_text("# resolver\n")
    svc.remember("the lock drops on scope exit", kind="gotcha", anchors=["resolver.py"])
    out = svc.proactive_context(
        text="", file_paths=[str(f)], session_id="s1", channel="proxy", defer_charge=True,
    )
    assert out["delivery_token"]
    svc._confirm_proactive_delivery(out["delivery_token"])
    assert not (tmp_path / "audit.log").exists()
    assert not list(tmp_path.glob("**/*.log"))


# -- 7. idempotency + both pending stores are bounded ------------------------

async def test_confirm_is_idempotent_and_unknown_tokens_are_silent(anchored):
    svc, path = anchored
    out = svc.proactive_context(
        text="", file_paths=[path], session_id="s1", channel="proxy", defer_charge=True,
    )
    token = out["delivery_token"]
    assert svc._confirm_proactive_delivery(token) is True
    counts = dict(svc._proactive_injection_counts)
    # Replay: charges nothing, counts nothing, raises nothing.
    assert svc._confirm_proactive_delivery(token) is False
    assert dict(svc._proactive_injection_counts) == counts
    assert svc._confirm_proactive_delivery("not-a-real-token") is False
    assert svc._confirm_proactive_delivery("") is False


async def test_daemon_pending_store_is_bounded(anchored):
    """Unconfirmed deliveries accumulate only up to a hard cap, oldest evicted
    first — a caller that never confirms cannot grow daemon memory."""
    svc, path = anchored
    from app.service import PROACTIVE_MAX_PENDING_DELIVERIES

    first_token = ""
    for i in range(PROACTIVE_MAX_PENDING_DELIVERIES + 10):
        out = svc.proactive_context(
            text="", file_paths=[path], session_id=f"s{i}", channel="proxy", defer_charge=True,
        )
        if i == 0:
            first_token = out["delivery_token"]
    assert len(svc._proactive_pending) <= PROACTIVE_MAX_PENDING_DELIVERIES
    assert first_token not in svc._proactive_pending, "oldest evicted first"


def test_proxy_pending_confirm_queue_is_bounded():
    """The proxy side is bounded too (it only grows while the daemon is
    unreachable). An evicted token just leaves its note uncharged — noise, and
    the safe direction."""
    from agent.proactive.provider import _MAX_PENDING_CONFIRMS

    provider = DaemonInjectionProvider("http://daemon")
    for i in range(_MAX_PENDING_CONFIRMS + 25):
        provider.confirm_delivery(f"t{i}")
    assert len(provider._pending_confirms) == _MAX_PENDING_CONFIRMS
    assert "t0" not in provider._pending_confirms
    assert f"t{_MAX_PENDING_CONFIRMS + 24}" in provider._pending_confirms


async def test_a_failed_daemon_call_retries_the_confirmation(anchored):
    """A token is dropped only once the daemon has ACCEPTED it, so a transient
    failure retries rather than silently losing the charge."""
    state = {"fail": True}

    def _handler(request: httpx.Request) -> httpx.Response:
        if state["fail"]:
            return httpx.Response(503, json={"error": "down"})
        return httpx.Response(200, json={
            "context": "", "item_count": 0, "anchor_ids": [], "scores": [],
            "delivery_token": "", "charge_deferred": False, "processing_ms": 1,
        })

    from agent.proactive.types import ProactiveWindow
    client = httpx.AsyncClient(transport=httpx.MockTransport(_handler), base_url="http://d")
    provider = DaemonInjectionProvider("http://d", client=client)
    provider.confirm_delivery("tok-1")
    window = ProactiveWindow(text="lock", file_paths=["/x/resolver.py"])

    assert await provider.inject(window, session_id="s", channel="proxy") == InjectionResult.empty()
    assert list(provider._pending_confirms) == ["tok-1"], "kept for retry"

    state["fail"] = False
    await provider.inject(window, session_id="s", channel="proxy")
    await client.aclose()
    assert list(provider._pending_confirms) == [], "dropped once accepted"


# -- proxy metrics -----------------------------------------------------------

async def test_not_appendable_skips_are_counted_separately(anchored):
    """The skip is observable: `inject_skipped_not_appendable` is a labelled
    subset of `inject_skipped`, so a deployment can tell 'nothing matched' from
    'the request could not carry it'."""
    svc, path = anchored
    up = MockUpstream()
    async with _daemon_client(svc) as client:
        provider = DaemonInjectionProvider("http://daemon", client=client)
        app = build_proxy_app(
            _settings(), injection_provider=provider, client_factory=up.client_factory()
        )
        await _post(app, _first_request(path))
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as c:
            metrics = (await c.get("/__vectr_proxy/health")).json()["metrics"]
    assert metrics["inject_skipped_not_appendable"] == 1
    assert metrics["inject_skipped"] >= 1
    assert metrics["injected"] == 0
