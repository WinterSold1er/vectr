"""DaemonInjectionProvider tests (UPG-PRO-16): the proxy's daemon client.

Drives a stub daemon in-process — never a real daemon. Fail-open is the key
behaviour: any daemon error becomes "inject nothing".
"""
from __future__ import annotations

import httpx
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from agent.proactive.provider import DaemonInjectionProvider
from agent.proactive.settings import ProactiveSettings, derive_provider_timeout_s
from agent.proactive.types import ProactiveWindow


def _stub_daemon(response_json, status=200, capture=None):
    async def _proactive(request):
        if capture is not None:
            capture["body"] = await request.json()
            capture["headers"] = {k.lower(): v for k, v in request.headers.items()}
        return JSONResponse(status_code=status, content=response_json)

    return Starlette(routes=[Route("/v1/proactive", _proactive, methods=["POST"])])


def _provider(app, **kw):
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://daemon")
    return DaemonInjectionProvider("http://daemon", client=client, **kw)


async def test_provider_returns_injection_result():
    capture = {}
    app = _stub_daemon(
        {"context": "CTX-BODY", "item_count": 2, "anchor_ids": ["note:1", "note:2"], "scores": [1.0, 0.8]},
        capture=capture,
    )
    prov = _provider(app, api_key="k1")
    w = ProactiveWindow(text="lock", file_paths=["/x/a.py"], symbols=["Foo"])
    result = await prov.inject(w, session_id="s1", channel="proxy")
    assert result.context == "CTX-BODY"
    assert result.item_count == 2
    assert result.anchor_ids == ("note:1", "note:2")
    # Window + attribution threaded to the daemon.
    assert capture["body"]["text"] == "lock"
    assert capture["body"]["file_paths"] == ["/x/a.py"]
    assert capture["body"]["session_id"] == "s1"
    assert capture["headers"].get("x-api-key") == "k1"
    await prov.aclose()


async def test_provider_empty_context_is_empty_result():
    app = _stub_daemon({"context": "", "item_count": 0, "anchor_ids": [], "scores": []})
    prov = _provider(app)
    result = await prov.inject(ProactiveWindow(text="x"), session_id="", channel="proxy")
    assert result.is_empty()
    await prov.aclose()


async def test_provider_daemon_error_is_fail_open():
    app = _stub_daemon({"error": "boom"}, status=500)
    prov = _provider(app)
    result = await prov.inject(ProactiveWindow(text="x"), session_id="", channel="proxy")
    assert result.is_empty()  # a 500 -> inject nothing, never raise
    await prov.aclose()


async def test_provider_unreachable_is_fail_open():
    class _Down(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            raise httpx.ConnectError("down", request=request)

    client = httpx.AsyncClient(transport=_Down(), base_url="http://daemon")
    prov = DaemonInjectionProvider("http://daemon", client=client)
    result = await prov.inject(ProactiveWindow(text="x"), session_id="", channel="proxy")
    assert result.is_empty()
    await prov.aclose()


# -- budget-ordering invariant (UPG-PROXY-BUDGET-40MS) ----------------------
#
# main.py's `cmd_proxy` builds DaemonInjectionProvider with
# `timeout_s=derive_provider_timeout_s(settings)` — this is the same wiring,
# exercised end-to-end from settings through to the constructed instance.


def test_provider_constructed_via_wiring_has_timeout_below_budget():
    settings = ProactiveSettings(
        enabled=True, min_similarity=0.35, max_items_per_event=3, max_chars_per_event=800,
        cooldown_items=30, cooldown_ttl_seconds=None,
        matcher_structural_note=True, matcher_semantic_note=True,
        matcher_code_search=False, proxy_host="127.0.0.1",
        proxy_port=8785, proxy_upstream_base_url="https://api.anthropic.com",
        proxy_connect_timeout_s=10.0, proxy_read_timeout_s=600.0, proxy_inject=True,
        proxy_inject_budget_ms=750, proxy_inject_provider_timeout_fraction=0.8,
        proxy_inject_provider_timeout_max_s=2.0, proxy_exclude_directive_notes=True,
        cache_enabled=False, cache_max_entries=2048,
        cache_ttl_seconds=0.0, cache_similarity_threshold=1.0, response_cache_enabled=False,
        response_cache_ttl_seconds=60.0, response_cache_max_entries=256,
        structural_kinds=("gotcha", "finding", "decision", "operational", "reference"),
        structural_overfetch_multiplier=4, structural_overfetch_ceiling=60,
        structural_score_declared_anchor=1.0, structural_score_declared_trigger=0.95,
        structural_score_gotcha_mention=0.9,
        structural_score_mention=0.6, max_weak_structural_items=1,
    )
    # The exact construction main.py performs.
    timeout_s = derive_provider_timeout_s(settings)
    prov = DaemonInjectionProvider("http://daemon", timeout_s=timeout_s)
    assert prov._timeout == timeout_s
    assert prov._timeout < settings.proxy_inject_budget_ms / 1000.0
