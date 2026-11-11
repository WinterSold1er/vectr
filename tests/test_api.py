"""Integration tests for FastAPI routes. Uses a fully mocked VectrService — no model loading."""
from __future__ import annotations

import os
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api import app
from agent.searcher import SearchResult
from agent.symbol_graph import LocateResult, Symbol
from tests.conftest import _base_mock_service


def _make_service():
    svc = MagicMock()
    svc._embed_model = "BAAI/bge-base-en-v1.5"
    svc.total_chunks = 500
    # UPG-8.2: /v1/health sources last_indexed from the same VectrService
    # property that populates svc.status()["last_indexed"] below.
    svc.last_indexed = "2026-01-01T00:00:00Z"

    _result = SearchResult(
        file_path="src/auth/middleware.py",
        lines="42-67",
        symbol_name="verify_jwt_token",
        language="python",
        score=0.94,
        content="def verify_jwt_token(token: str) -> dict:\n    ...",
    )

    svc.search.return_value = ([_result], 18)
    # UPG-QUERYTYPE-REROUTE: additive symbol-graph hint — empty by default.
    svc.identifier_hint_symbols.return_value = []
    svc.index.return_value = (12, 500, 240)
    svc.status.return_value = {
        "indexed_files": 12,
        "total_chunks": 500,
        "last_indexed": "2026-01-01T00:00:00Z",
        "embed_model": "BAAI/bge-base-en-v1.5",
        "workspace_root": "/repo",
        "symbol_count": 0,
        # UPG-3.3: per-language coverage + symbol availability (real shape)
        "languages": [
            {"language": "python", "files": 10, "chunks": 400, "symbols": True},
            {"language": "markdown", "files": 2, "chunks": 100, "symbols": False},
        ],
        # UPG-8.7: symbol-graph build trust signals (real shape)
        "symbol_graph_complete": True,
        "symbol_graph_failed_files": 0,
        # UPG-STATUS-STALE-GRAPH: staleness signals (real shape)
        "symbol_graph_stale": False,
        "symbol_graph_rebuild_in_progress": False,
        # UPG-HOOK-INJECT-OBSERVABILITY: hook injection counters are always
        # present in the real service.status() output — mock the real shape.
        "hook_injection_counts": {"SessionStart": 3, "PreToolUse": 2},
        # UPG-PURPOSE-RESUME-HOLE: ARCH-4 dual-vector purpose-chunk count and
        # per-run backfill-gap count are always present in the real
        # service.status() output — mock the real shape.
        "total_purpose_chunks": 480,
        "purpose_backfill_pending_files": 0,
        "purpose_vectors_pending": False,
    }
    svc.get_map.return_value = "# Codebase Passport\nFastAPI service."
    # UPG-6.2: save_map returns a shaped result — real shape.
    svc.save_map.return_value = {"saved": True, "existing_summary": None}
    # locate_with_snippets returns a LocateResult wrapper (not a bare list) —
    # the REST route must unwrap .symbols. Mock the real shape so the route
    # contract is actually exercised.
    svc.locate_with_snippets.return_value = LocateResult(
        symbols=[
            Symbol(
                symbol_id=1,
                workspace="/repo",
                name="PyDict_New",
                kind="function",
                file_path="Objects/dictobject.c",
                start_line=812,
                end_line=824,
                snippet="PyObject *\nPyDict_New(void)\n{",
            )
        ],
        resolution_strategy="exact",
        query="PyDict_New",
    )
    svc.format_locate.return_value = "PyDict_New  Objects/dictobject.c:812"
    svc.trace_with_snippets.return_value = {}
    svc.format_trace.return_value = "No trace."
    svc.should_evict.return_value = False
    svc.eviction_hint.return_value = ""
    # Default mode is full (not memory-only); must be an explicit bool, not a MagicMock.
    svc.memory_only = False
    return svc


@pytest.fixture
def client():
    """
    Fast fixture: patches VectrService so no model is loaded during lifespan.
    All service calls go to a MagicMock with realistic return values.
    """
    svc = _make_service()
    with patch("app.service.VectrService", return_value=svc):
        with TestClient(app, raise_server_exceptions=True) as c:
            # Ensure mock is in app state regardless of lifespan behaviour
            app.state.service = svc
            yield c


# ---------------------------------------------------------------------------
# UPG-WORKSPACE-ENV-VALIDATE: daemon startup fails loudly on a bad
# VECTR_WORKSPACE rather than silently falling back to cwd detection.
# ---------------------------------------------------------------------------

class TestLifespanWorkspaceEnvValidation:
    def test_bad_vectr_workspace_env_fails_startup(self, tmp_path, monkeypatch) -> None:
        from integrations.workspace_detect import WorkspaceEnvError

        bad_path = str(tmp_path / "does-not-exist")
        monkeypatch.setenv("VECTR_WORKSPACE", bad_path)
        with pytest.raises(WorkspaceEnvError) as excinfo:
            with TestClient(app, raise_server_exceptions=True):
                pass
        assert bad_path in str(excinfo.value)

    def test_valid_vectr_workspace_env_starts_normally(self, tmp_path, monkeypatch) -> None:
        svc = _make_service()
        monkeypatch.setenv("VECTR_WORKSPACE", str(tmp_path))
        with patch("app.service.VectrService", return_value=svc):
            with TestClient(app, raise_server_exceptions=True) as c:
                resp = c.get("/v1/health")
                assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health(client) -> None:
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "embed_model" in data


def test_health_last_indexed_agrees_with_status(client) -> None:
    """UPG-8.2: /v1/health and /v1/status must report the same last_indexed —
    both source it from VectrService.last_indexed, the single source of truth."""
    health_data = client.get("/v1/health").json()
    status_data = client.get("/v1/status").json()
    assert health_data["last_indexed"] == status_data["last_indexed"]


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def test_search_happy_path(client) -> None:
    resp = client.post("/v1/search", json={"query": "JWT token validation"})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 1
    assert data["results"][0]["file"] == "src/auth/middleware.py"
    assert data["results"][0]["score"] == 0.94
    assert "processing_ms" in data
    assert "chunks_searched" in data


def test_search_result_has_score_source_field(client) -> None:
    """UPG-SCORE-DISPLAY-MIXED-SCALE: every REST search result must carry an
    additive `score_source` field ("reranker" | "dense") naming which scale
    its `score` was drawn from. The default mock's SearchResult never sets
    score_source explicitly, so it must default to "dense" — matching
    `score`'s own dense_sim fallback — rather than being omitted."""
    resp = client.post("/v1/search", json={"query": "JWT token validation"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["results"][0]["score_source"] == "dense"


def test_search_result_score_source_reranker_passthrough(client) -> None:
    """A SearchResult the agent layer stamped score_source="reranker" on must
    pass that through to the REST response unchanged."""
    reranked_result = SearchResult(
        file_path="src/auth/middleware.py",
        lines="42-67",
        symbol_name="verify_jwt_token",
        language="python",
        score=0.87,
        content="def verify_jwt_token(token: str) -> dict:\n    ...",
        ce_relevance=0.87,
        score_source="reranker",
    )
    svc = app.state.service
    svc.search.return_value = ([reranked_result], 18)

    resp = client.post("/v1/search", json={"query": "JWT token validation"})
    assert resp.status_code == 200
    assert resp.json()["results"][0]["score_source"] == "reranker"


def test_search_response_has_no_routing_fields(client) -> None:
    """UPG-QUERYTYPE-REROUTE: the deleted regex query-classification layer's
    RoutingDecision (query_type/routing/decision/resolution_strategy) must not
    appear anywhere on the REST /v1/search response — the route always
    returns a plain hybrid-retrieval result set."""
    resp = client.post("/v1/search", json={"query": "JWT token validation"})
    assert resp.status_code == 200
    data = resp.json()
    for forbidden_key in ("routing", "decision", "query_type", "resolution_strategy"):
        assert forbidden_key not in data
        for result in data["results"]:
            assert forbidden_key not in result


def test_search_missing_query(client) -> None:
    resp = client.post("/v1/search", json={"n_results": 5})
    assert resp.status_code == 422


def test_search_invalid_n_results_too_high(client) -> None:
    resp = client.post("/v1/search", json={"query": "auth", "n_results": 999})
    assert resp.status_code == 422


def test_search_unindexed_language_no_422(client) -> None:
    # UPG-3.1: a real-but-unindexed language must NOT 422 (c/zig were the pain).
    # It is accepted; unindexed languages simply yield no matches (+ MCP hint).
    for lang in ("c", "zig", "cobol"):
        resp = client.post("/v1/search", json={"query": "auth", "language": lang})
        assert resp.status_code == 200, f"{lang} should be accepted, got {resp.status_code}"


def test_search_language_normalized(client) -> None:
    # mixed case + whitespace normalised to lower/stripped
    resp = client.post("/v1/search", json={"query": "auth", "language": "  Rust "})
    assert resp.status_code == 200


def test_search_with_language_filter(client) -> None:
    resp = client.post("/v1/search", json={"query": "auth", "language": "python"})
    assert resp.status_code == 200


def test_search_request_accepts_any_language() -> None:
    # UPG-3.1: model no longer rejects/normalises — it just carries the value.
    # Normalisation is shared in CodeSearcher.search (see test_indexer_searcher).
    from app.models import SearchRequest
    assert SearchRequest(query="q", language="c").language == "c"
    assert SearchRequest(query="q", language="zig").language == "zig"
    assert SearchRequest(query="q", language=None).language is None


# ---------------------------------------------------------------------------
# UPG-NOTFOUND-FLOOR (F46) — REST low_confidence field
# ---------------------------------------------------------------------------

def test_search_default_low_confidence_false(client) -> None:
    """The default mock returns a plain list (no low_confidence attribute) —
    the REST route must default to False rather than error or omit the field."""
    resp = client.post("/v1/search", json={"query": "JWT token validation"})
    assert resp.status_code == 200
    assert resp.json()["low_confidence"] is False


def test_search_surfaces_low_confidence_true(client) -> None:
    """When CodeSearcher flags the result set low_confidence, /v1/search must
    carry that through as a top-level `low_confidence: true` field — results
    are still returned in full (isError/suppression is never involved here,
    this is REST, but the results list must remain non-empty)."""
    from agent.searcher import SearchResultList

    svc = app.state.service
    weak_result = SearchResult(
        file_path="src/unrelated/module.py",
        lines="1-3",
        symbol_name="unrelated_fn",
        language="python",
        score=0.81,
        content="def unrelated_fn():\n    pass",
    )
    flagged = SearchResultList([weak_result])
    flagged.low_confidence = True
    svc.search.return_value = (flagged, 12)

    resp = client.post("/v1/search", json={"query": "CORS handling implementation"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["low_confidence"] is True
    assert len(data["results"]) == 1  # results are never suppressed


# ---------------------------------------------------------------------------
# Locate (symbol graph) — route must unwrap LocateResult.symbols
# ---------------------------------------------------------------------------

def test_locate_happy_path(client) -> None:
    resp = client.post("/v1/locate", json={"name": "PyDict_New"})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 1
    sym = data["results"][0]
    assert sym["name"] == "PyDict_New"
    assert sym["file_path"] == "Objects/dictobject.c"
    assert sym["start_line"] == 812
    assert "formatted" in data
    assert "processing_ms" in data


def test_locate_empty(client) -> None:
    """A LocateResult with no symbols returns 200 + empty results, not a 500."""
    app.state.service.locate_with_snippets.return_value = LocateResult(
        symbols=[], resolution_strategy="none", query="nope"
    )
    resp = client.post("/v1/locate", json={"name": "nope"})
    assert resp.status_code == 200
    assert resp.json()["results"] == []


def test_trace_happy_path(client) -> None:
    resp = client.post("/v1/trace", json={"name": "get_gc_state"})
    assert resp.status_code == 200
    data = resp.json()
    assert "formatted" in data
    assert "processing_ms" in data


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Map save (UPG-6.2) — every REST route gets its own test, not just MCP
# ---------------------------------------------------------------------------

def test_map_save_happy_path(client) -> None:
    resp = client.post("/v1/map", json={"summary": "Python FastAPI service."})
    assert resp.status_code == 200
    data = resp.json()
    assert data["saved"] is True
    assert "saved" in data["message"].lower()


def test_map_save_blocked_when_passport_exists_and_not_overwrite(client) -> None:
    app.state.service.save_map.return_value = {
        "saved": False, "existing_summary": "Existing passport summary.",
    }
    resp = client.post("/v1/map", json={"summary": "New summary."})
    assert resp.status_code == 200
    data = resp.json()
    assert data["saved"] is False
    assert "already exists" in data["message"].lower()
    assert "Existing passport summary." in data["message"]
    app.state.service.save_map.assert_called_once_with("New summary.", overwrite=False)


def test_map_save_overwrite_true_forwarded_to_service(client) -> None:
    resp = client.post("/v1/map", json={"summary": "New summary.", "overwrite": True})
    assert resp.status_code == 200
    app.state.service.save_map.assert_called_once_with("New summary.", overwrite=True)


def test_index_happy_path(client) -> None:
    resp = client.post("/v1/index", json={"path": ".", "force": False})
    assert resp.status_code == 200
    data = resp.json()
    assert data["indexed_files"] == 12
    assert "processing_ms" in data


def test_index_does_not_block_concurrent_status(client) -> None:
    """UPG-REST-STARVATION: POST /v1/index runs svc.index() in a threadpool
    (see app/routes.py) instead of directly on the event-loop thread, so a
    slow/bulk index call must never block GET /v1/status from answering
    promptly on a concurrent request."""
    svc = client.app.state.service
    entered = threading.Event()
    release = threading.Event()

    def slow_index(path, force=False):
        entered.set()
        assert release.wait(timeout=5), "test never released the slow index call"
        return (12, 500, 240)

    svc.index.side_effect = slow_index

    results: dict[str, object] = {}

    def do_index() -> None:
        results["resp"] = client.post("/v1/index", json={"path": ".", "force": False})

    t = threading.Thread(target=do_index)
    t.start()
    assert entered.wait(timeout=2), "slow /v1/index handler never started"

    t0 = time.monotonic()
    status_resp = client.get("/v1/status")
    elapsed = time.monotonic() - t0

    release.set()
    t.join(timeout=5)

    assert status_resp.status_code == 200
    assert elapsed < 1.0, f"/v1/status took {elapsed:.2f}s while /v1/index was running"
    assert results["resp"].status_code == 200


def test_mcp_tools_call_does_not_block_concurrent_status(client) -> None:
    """UPG-EMBED-THREAD-CONTENTION: POST /mcp's tools/call dispatch runs
    handle_tools_call() in a threadpool (see app/routes.py), mirroring
    /v1/index above — an embed-touching tool call (e.g. vectr_search) must
    never block GET /v1/status from answering promptly on a concurrent
    request, the same event-loop-starvation class as the index route."""
    svc = client.app.state.service
    entered = threading.Event()
    release = threading.Event()

    def slow_search(query, n_results=5, language=None):
        entered.set()
        assert release.wait(timeout=5), "test never released the slow search call"
        return ([], 0)

    svc.search.side_effect = slow_search

    results: dict[str, object] = {}

    def do_tool_call() -> None:
        results["resp"] = client.post("/mcp", json={
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "vectr_search", "arguments": {"query": "rate limiter"}},
        })

    t = threading.Thread(target=do_tool_call)
    t.start()
    assert entered.wait(timeout=2), "slow /mcp tools/call handler never started"

    t0 = time.monotonic()
    status_resp = client.get("/v1/status")
    elapsed = time.monotonic() - t0

    release.set()
    t.join(timeout=5)

    assert status_resp.status_code == 200
    assert elapsed < 1.0, f"/v1/status took {elapsed:.2f}s while /mcp tools/call was running"
    assert results["resp"].status_code == 200
    assert results["resp"].json()["result"]["isError"] is False


def test_legacy_mcp_tools_call_does_not_block_concurrent_status(client) -> None:
    """Same guard for the legacy POST /mcp/tools/call route."""
    svc = client.app.state.service
    entered = threading.Event()
    release = threading.Event()

    def slow_search(query, n_results=5, language=None):
        entered.set()
        assert release.wait(timeout=5), "test never released the slow search call"
        return ([], 0)

    svc.search.side_effect = slow_search

    results: dict[str, object] = {}

    def do_tool_call() -> None:
        results["resp"] = client.post(
            "/mcp/tools/call", json={"name": "vectr_search", "arguments": {"query": "rate limiter"}},
        )

    t = threading.Thread(target=do_tool_call)
    t.start()
    assert entered.wait(timeout=2), "slow legacy /mcp/tools/call handler never started"

    t0 = time.monotonic()
    status_resp = client.get("/v1/status")
    elapsed = time.monotonic() - t0

    release.set()
    t.join(timeout=5)

    assert status_resp.status_code == 200
    assert elapsed < 1.0, f"/v1/status took {elapsed:.2f}s while /mcp/tools/call was running"
    assert results["resp"].status_code == 200


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def test_status_reindex_in_progress_field_present(client) -> None:
    """UPG-REST-STARVATION requirement #2: /v1/status always carries a
    reindex_in_progress flag (defaults to False when the service doesn't
    report otherwise, and surfaces True when it does)."""
    resp = client.get("/v1/status")
    assert resp.status_code == 200
    assert resp.json()["reindex_in_progress"] is False

    svc = client.app.state.service
    svc.status.return_value = {**svc.status.return_value, "reindex_in_progress": True}
    resp = client.get("/v1/status")
    assert resp.json()["reindex_in_progress"] is True


def test_status(client) -> None:
    resp = client.get("/v1/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["indexed_files"] == 12
    assert data["total_chunks"] == 500
    assert "workspace_root" in data
    assert "notes_count" in data, "/v1/status must include notes_count for agent recall decisions"
    assert isinstance(data["notes_count"], int)
    # UPG-8.7: symbol-graph build trust signals surface in /v1/status
    assert data["symbol_graph_complete"] is True
    assert data["symbol_graph_failed_files"] == 0
    # UPG-STATUS-STALE-GRAPH: staleness signals surface in /v1/status
    assert data["symbol_graph_stale"] is False
    assert data["symbol_graph_rebuild_in_progress"] is False
    # UPG-3.3: per-language coverage with symbol availability
    langs = {l["language"]: l for l in data["languages"]}
    assert langs["python"]["symbols"] is True
    assert langs["python"]["files"] == 10
    assert langs["markdown"]["symbols"] is False
    # UPG-HOOK-INJECT-OBSERVABILITY: hook injection counters surface in
    # /v1/status so `vectr status`/vectr_status can render them.
    assert data["hook_injection_counts"] == {"SessionStart": 3, "PreToolUse": 2}
    # UPG-PURPOSE-RESUME-HOLE: ARCH-4 purpose-chunk count and per-run
    # backfill-gap count surface in /v1/status so a body/purpose divergence
    # cannot hide behind a green status.
    assert data["total_purpose_chunks"] == 480
    assert data["purpose_backfill_pending_files"] == 0


def test_status_purpose_backfill_gap_surfaces(client) -> None:
    """UPG-PURPOSE-RESUME-HOLE: when the indexer's most recent run found
    files with current body content but incomplete purpose vectors,
    /v1/status must report the exact count via
    purpose_backfill_pending_files rather than requiring the caller to
    infer a gap by dividing total_chunks by total_purpose_chunks (a ratio
    that is always < 1 even on a fully healthy index, since only
    symbol-bearing chunks get a purpose vector at all)."""
    svc = client.app.state.service
    svc.status.return_value = {
        **svc.status.return_value,
        "total_chunks": 500,
        "total_purpose_chunks": 40,
        "purpose_backfill_pending_files": 7,
    }
    resp = client.get("/v1/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_chunks"] == 500
    assert data["total_purpose_chunks"] == 40
    assert data["purpose_backfill_pending_files"] == 7


# ---------------------------------------------------------------------------
# MCP protocol
# ---------------------------------------------------------------------------

def test_mcp_info(client) -> None:
    resp = client.get("/mcp")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "vectr"


def test_mcp_tools_list(client) -> None:
    resp = client.post("/mcp/tools/list", json={})
    assert resp.status_code == 200
    data = resp.json()
    names = {t["name"] for t in data["tools"]}
    assert "vectr_search" in names
    assert "vectr_status" in names
    assert "vectr_map_save" in names


def test_mcp_tools_call_search(client) -> None:
    resp = client.post(
        "/mcp/tools/call",
        json={"name": "vectr_search", "arguments": {"query": "JWT token validation"}},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["isError"] is False
    assert "verify_jwt_token" in data["content"][0]["text"]


def test_mcp_tools_call_map_save(client) -> None:
    resp = client.post(
        "/mcp/tools/call",
        json={"name": "vectr_map_save", "arguments": {"summary": "Python FastAPI service for auth."}},
    )
    assert resp.status_code == 200
    assert resp.json()["isError"] is False


def test_mcp_tools_call_missing_name(client) -> None:
    resp = client.post("/mcp/tools/call", json={"arguments": {}})
    assert resp.status_code == 400


def test_mcp_tools_call_status(client) -> None:
    resp = client.post("/mcp/tools/call", json={"name": "vectr_status", "arguments": {}})
    assert resp.status_code == 200
    assert resp.json()["isError"] is False


def test_mcp_status_renders_language_capability(client) -> None:
    """UPG-3.3: vectr_status text must tell the agent where locate/trace work."""
    resp = client.post("/mcp/tools/call", json={"name": "vectr_status", "arguments": {}})
    text = resp.json()["content"][0]["text"]
    assert "Languages" in text
    assert "python" in text and "locate/trace" in text
    assert "markdown" in text and "search-only" in text


def test_mcp_status_warns_when_primary_language_has_no_symbols(client) -> None:
    """If the dominant language is search-only, the agent should be told to prefer
    search over locate/trace — the adoption-critical routing hint."""
    app.state.service.status.return_value = {
        "indexed_files": 240, "total_chunks": 900,
        "last_indexed": "2026-01-01T00:00:00Z", "embed_model": "x",
        "workspace_root": "/repo", "symbol_count": 0,
        "languages": [
            {"language": "markdown", "files": 240, "chunks": 800, "symbols": False},
            {"language": "python", "files": 5, "chunks": 100, "symbols": True},
        ],
    }
    resp = client.post("/mcp/tools/call", json={"name": "vectr_status", "arguments": {}})
    text = resp.json()["content"][0]["text"]
    assert "Primary language (markdown)" in text
    assert "prefer vectr_search" in text


# ---------------------------------------------------------------------------
# T15: VECTR_API_KEY enforcement
# ---------------------------------------------------------------------------

class TestApiKeyEnforcement:
    """T15: VECTR_API_KEY header enforcement.

    The middleware checks os.environ at request time (not at import time),
    so patch.dict works cleanly without module reloading.
    """

    def test_no_key_set_allows_all_requests(self, client) -> None:
        with patch.dict("os.environ", {"VECTR_API_KEY": ""}):
            resp = client.get("/v1/health")
        assert resp.status_code == 200

    def test_health_endpoint_bypasses_key_check(self, client) -> None:
        with patch.dict("os.environ", {"VECTR_API_KEY": "test-secret-key"}):
            resp = client.get("/v1/health")
        assert resp.status_code == 200  # health always allowed

    def test_missing_key_returns_401(self, client) -> None:
        with patch.dict("os.environ", {"VECTR_API_KEY": "test-secret-key"}):
            resp = client.get("/v1/status")
        assert resp.status_code == 401

    def test_wrong_key_returns_401(self, client) -> None:
        with patch.dict("os.environ", {"VECTR_API_KEY": "test-secret-key"}):
            resp = client.get("/v1/status", headers={"X-Api-Key": "wrong-key"})
        assert resp.status_code == 401

    def test_correct_x_api_key_header_passes(self, client) -> None:
        with patch.dict("os.environ", {"VECTR_API_KEY": "test-secret-key"}):
            resp = client.get("/v1/status", headers={"X-Api-Key": "test-secret-key"})
        assert resp.status_code == 200

    def test_bearer_token_passes(self, client) -> None:
        with patch.dict("os.environ", {"VECTR_API_KEY": "test-secret-key"}):
            resp = client.get("/v1/status", headers={"Authorization": "Bearer test-secret-key"})
        assert resp.status_code == 200

    def test_401_response_includes_error_field(self, client) -> None:
        with patch.dict("os.environ", {"VECTR_API_KEY": "test-secret-key"}):
            resp = client.get("/v1/status")
        assert resp.status_code == 401
        body = resp.json()
        assert "error" in body
        assert body["error"] == "unauthorized"

    def test_empty_key_env_var_allows_requests(self, client) -> None:
        with patch.dict("os.environ", {"VECTR_API_KEY": ""}):
            resp = client.get("/v1/status")
        assert resp.status_code == 200

    def test_mcp_endpoint_requires_key(self, client) -> None:
        """The MCP surface (what the editor's LLM actually talks to) is protected
        by the same middleware, not just the REST /v1 routes."""
        with patch.dict("os.environ", {"VECTR_API_KEY": "test-secret-key"}):
            resp = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        assert resp.status_code == 401

    def test_mcp_endpoint_passes_with_key(self, client) -> None:
        with patch.dict("os.environ", {"VECTR_API_KEY": "test-secret-key"}):
            resp = client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
                headers={"X-Api-Key": "test-secret-key"},
            )
        assert resp.status_code == 200
        assert resp.json()["result"]["serverInfo"]["name"]

    def test_401_body_never_leaks_key(self, client) -> None:
        """A rejection must never echo the configured key or the provided key."""
        with patch.dict("os.environ", {"VECTR_API_KEY": "super-secret-value-123"}):
            resp = client.get("/v1/status", headers={"X-Api-Key": "attacker-guess-456"})
        assert resp.status_code == 401
        assert "super-secret-value-123" not in resp.text
        assert "attacker-guess-456" not in resp.text

    def test_constant_time_comparator_rejects_prefix_match(self, client) -> None:
        """A key sharing a long prefix but differing at the end is still rejected
        — the comparison covers the full value, not an early-exit prefix check."""
        with patch.dict("os.environ", {"VECTR_API_KEY": "abcdefghijklmnop"}):
            resp_prefix = client.get("/v1/status", headers={"X-Api-Key": "abcdefghijklmnoZ"})
            resp_short = client.get("/v1/status", headers={"X-Api-Key": "abcdefghij"})
            resp_exact = client.get("/v1/status", headers={"X-Api-Key": "abcdefghijklmnop"})
        assert resp_prefix.status_code == 401
        assert resp_short.status_code == 401
        assert resp_exact.status_code == 200


# ---------------------------------------------------------------------------
# Team mode: client attribution via the X-Vectr-Client header
# ---------------------------------------------------------------------------

class TestClientAttributionHeader:
    """A team client's label (X-Vectr-Client, written by `vectr connect --label`)
    becomes the default note author when vectr_remember declares no `agent`."""

    def _remember_call(self, client, headers, arguments):
        # The working-memory layer must be enabled for the remember branch to run.
        from api import app
        from app.service import RememberOutcome
        app.state.service.search_only = False
        app.state.service.remember_with_extras.return_value = RememberOutcome(
            note_id=7, related=[], revoked_related=[], proxy_anchor_suggestions=[],
        )
        return client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": {"name": "vectr_remember", "arguments": arguments},
            },
            headers=headers,
        )

    def test_client_label_becomes_author(self, client) -> None:
        from api import app
        svc = app.state.service
        resp = self._remember_call(
            client, {"X-Vectr-Client": "bob"}, {"content": "a team finding"},
        )
        assert resp.status_code == 200
        _, kwargs = svc.remember_with_extras.call_args
        assert kwargs["agent"] == "bob"

    def test_explicit_agent_wins_over_client_label(self, client) -> None:
        from api import app
        svc = app.state.service
        resp = self._remember_call(
            client, {"X-Vectr-Client": "bob"},
            {"content": "a finding", "agent": "coder-2"},
        )
        assert resp.status_code == 200
        _, kwargs = svc.remember_with_extras.call_args
        assert kwargs["agent"] == "coder-2"

    def test_no_header_no_attribution(self, client) -> None:
        from api import app
        svc = app.state.service
        resp = self._remember_call(client, {}, {"content": "a solo finding"})
        assert resp.status_code == 200
        _, kwargs = svc.remember_with_extras.call_args
        assert kwargs["agent"] == ""


# ---------------------------------------------------------------------------
# UPG-STDIO-MEMORY-READY: HTTP-transport readiness gating.
#
# `svc.fully_ready` models phase 2 of VectrService construction (embedder,
# indexer, searcher, watcher, symbol graph) not having completed yet.
# Search-touching REST routes and non-memory-ready MCP tools must 503/
# graceful-degrade in that state; memory routes and memory-ready MCP tools
# must serve normally, keyed on route/tool identity + service state only.
# ---------------------------------------------------------------------------

def _not_fully_ready_client():
    """A fresh TestClient wired to a mocked service with fully_ready=False —
    independent of the module-level `client` fixture so each test gets its
    own isolated mock (no cross-test state bleed from a shared instance).
    Returns (client, svc, client) — the third element is the same object,
    kept only so call sites can `ctx.__exit__(...)` symmetrically with the
    other constructors in this file."""
    svc = _make_service()
    svc.fully_ready = False
    c = TestClient(app, raise_server_exceptions=True)
    with patch("app.service.VectrService", return_value=svc):
        c.__enter__()
    app.state.service = svc
    return c, svc, c


class TestSearchRoutesGatedWhileNotFullyReady:
    @pytest.mark.parametrize("method,path,json_body", [
        ("post", "/v1/index", {"path": ".", "force": False}),
        ("post", "/v1/search", {"query": "auth"}),
        ("post", "/v1/fetch", {"ids": ["a.py:1-2"]}),
        ("get", "/v1/map", None),
        ("post", "/v1/map", {"summary": "a summary"}),
        ("post", "/v1/locate", {"name": "foo"}),
        ("post", "/v1/trace", {"name": "foo"}),
        ("get", "/v1/evict-hint", None),
    ])
    def test_returns_503_still_initialising(self, method, path, json_body) -> None:
        c, svc, ctx = _not_fully_ready_client()
        try:
            fn = getattr(c, method)
            resp = fn(path, json=json_body) if json_body is not None else fn(path)
            assert resp.status_code == 503
            assert resp.json()["detail"]["error"] == "still_initialising"
        finally:
            ctx.__exit__(None, None, None)

    @pytest.mark.parametrize("method,path,json_body", [
        ("post", "/v1/index", {"path": ".", "force": False}),
        ("post", "/v1/search", {"query": "auth"}),
        ("post", "/v1/locate", {"name": "foo"}),
        ("post", "/v1/trace", {"name": "foo"}),
    ])
    def test_returns_200_once_fully_ready(self, method, path, json_body) -> None:
        """Same routes, `fully_ready=True` (the default mock state) — must
        behave exactly as every other test in this file (no 503)."""
        svc = _make_service()
        svc.fully_ready = True
        with patch("app.service.VectrService", return_value=svc):
            with TestClient(app, raise_server_exceptions=True) as c:
                app.state.service = svc
                fn = getattr(c, method)
                resp = fn(path, json=json_body) if json_body is not None else fn(path)
        assert resp.status_code == 200


def _not_fully_ready_memory_client():
    """Same shape as `_not_fully_ready_client` but built on `_base_mock_service`
    (conftest), which already stubs realistic remember/recall/snapshot_session/
    forget_all/list_snapshots return values — the local `_make_service` above
    is search-route-focused and leaves those unset."""
    svc = _base_mock_service()
    svc.fully_ready = False
    c = TestClient(app, raise_server_exceptions=True)
    with patch("app.service.VectrService", return_value=svc):
        c.__enter__()
    app.state.service = svc
    return c, svc, c


class TestMemoryRoutesServeWhileNotFullyReady:
    @pytest.mark.parametrize("method,path,json_body", [
        ("post", "/v1/remember", {"content": "note stored during warm-up"}),
        ("post", "/v1/recall", {}),
        ("post", "/v1/snapshot", {"label": "warm-up-checkpoint"}),
        ("post", "/v1/forget", {"all": True}),
        ("get", "/v1/status", None),
        ("get", "/v1/resume", None),
        ("post", "/v1/memory/clear", {}),
    ])
    def test_returns_200_not_gated(self, method, path, json_body) -> None:
        c, svc, ctx = _not_fully_ready_memory_client()
        try:
            fn = getattr(c, method)
            resp = fn(path, json=json_body) if json_body is not None else fn(path)
            assert resp.status_code == 200
        finally:
            ctx.__exit__(None, None, None)

    def test_status_reflects_fully_ready_false(self) -> None:
        c, svc, ctx = _not_fully_ready_memory_client()
        try:
            svc.status.return_value = {**svc.status.return_value, "fully_ready": False, "embedder_ready": False}
            data = c.get("/v1/status").json()
            assert data["fully_ready"] is False
            assert data["embedder_ready"] is False
        finally:
            ctx.__exit__(None, None, None)


class TestMcpToolReadinessGating:
    """Both MCP surfaces (POST /mcp JSON-RPC and POST /mcp/tools/call REST)
    share the same `_mcp_tool_still_initialising` gate — verify each."""

    @pytest.mark.parametrize("endpoint,payload_fn", [
        ("/mcp", lambda name, args: {
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": name, "arguments": args},
        }),
        ("/mcp/tools/call", lambda name, args: {"name": name, "arguments": args}),
    ])
    def test_memory_tool_dispatches_while_not_fully_ready(self, endpoint, payload_fn) -> None:
        c, svc, ctx = _not_fully_ready_memory_client()
        try:
            resp = c.post(endpoint, json=payload_fn("vectr_status", {}))
            assert resp.status_code == 200
            body = resp.json()["result"] if endpoint == "/mcp" else resp.json()
            assert "starting up" not in body["content"][0]["text"].lower()
        finally:
            ctx.__exit__(None, None, None)

    @pytest.mark.parametrize("endpoint,payload_fn", [
        ("/mcp", lambda name, args: {
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": name, "arguments": args},
        }),
        ("/mcp/tools/call", lambda name, args: {"name": name, "arguments": args}),
    ])
    def test_search_tool_gated_while_not_fully_ready(self, endpoint, payload_fn) -> None:
        c, svc, ctx = _not_fully_ready_client()
        try:
            resp = c.post(endpoint, json=payload_fn("vectr_search", {"query": "auth"}))
            assert resp.status_code == 200
            body = resp.json()["result"] if endpoint == "/mcp" else resp.json()
            assert body["isError"] is False
            assert "starting up" in body["content"][0]["text"].lower()
        finally:
            ctx.__exit__(None, None, None)

    def test_search_tool_dispatches_once_fully_ready(self) -> None:
        svc = _make_service()
        svc.fully_ready = True
        with patch("app.service.VectrService", return_value=svc):
            with TestClient(app, raise_server_exceptions=True) as c:
                app.state.service = svc
                resp = c.post("/mcp", json={
                    "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                    "params": {"name": "vectr_search", "arguments": {"query": "auth"}},
                })
        assert resp.status_code == 200
        text = resp.json()["result"]["content"][0]["text"]
        assert "starting up" not in text.lower()

    def test_tools_list_unaffected_by_fully_ready(self) -> None:
        c, svc, ctx = _not_fully_ready_client()
        try:
            resp = c.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
            assert resp.status_code == 200
            assert "tools" in resp.json()["result"]
        finally:
            ctx.__exit__(None, None, None)
