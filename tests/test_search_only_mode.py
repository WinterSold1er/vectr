"""Tests for vectr start --search-only mode (UPG-SEARCH-ONLY-MODE).

search-only is the dual of memory-only: indexing + the file watcher run
normally (semantic search / symbol graph / codebase map are all active), but
the working-memory layer is fully disabled — nothing is written, and no notes
database is ever created for a workspace that never writes a note.

Covers:
- VectrService: search_only flag read from env + constructor arg; mutual
  exclusion with memory_only
- WorkingContextStore is never constructed in search-only mode — no
  working_context.sqlite file is created on disk
- start_background_index: index thread + watcher run normally; TTL purge
  is skipped (no context store to purge)
- service.status(): mode field is "search-only"
- memory-facing service methods raise rather than write in search-only mode
- MCP dispatch: remember/recall/forget/snapshot/snapshot_list return the
  search-only message; vectr_evict_hint, vectr_map, vectr_map_save stay active
- REST routes: memory routes return 503 in search-only mode; search/locate/
  trace/map succeed normally
- CLI: _do_start env construction includes VECTR_SEARCH_ONLY=1 when flag is
  set; --memory-only + --search-only together is a CLI error
"""
from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# _RealVectrService is the constructor saved at collection time, BEFORE the
# session-scoped real_service_client fixture patches app.service.VectrService.
# See tests/test_memory_only_mode.py for why this indirection is needed.
from tests.conftest import make_py, _DummyEmbedProvider, _RealVectrService


# ---------------------------------------------------------------------------
# Helper: build a VectrService with the dummy embedder, no real model load
# ---------------------------------------------------------------------------

def _make_service(tmp_path, monkeypatch, search_only: bool = False):
    from agent import indexer as idx_module
    monkeypatch.setattr(idx_module, "get_embed_provider", lambda _: _DummyEmbedProvider())
    make_py(tmp_path, "sample.py", "def foo(): pass\n")

    with patch("integrations.vscode_bridge.configure_all"), \
         patch("integrations.workspace_detect.find_workspace_root", return_value=str(tmp_path)), \
         patch.dict("os.environ", {"VECTR_DB_DIR": str(tmp_path / "db")}):
        svc = _RealVectrService(workspace_root=str(tmp_path), search_only=search_only)
    return svc


# ---------------------------------------------------------------------------
# VectrService: search_only property + env read + mutual exclusion
# ---------------------------------------------------------------------------

class TestSearchOnlyServiceFlag:
    def test_constructor_arg_sets_search_only_true(self, tmp_path, monkeypatch):
        svc = _make_service(tmp_path, monkeypatch, search_only=True)
        assert svc.search_only is True
        assert svc.memory_only is False

    def test_constructor_default_is_full(self, tmp_path, monkeypatch):
        svc = _make_service(tmp_path, monkeypatch, search_only=False)
        assert svc.search_only is False

    def test_env_var_sets_search_only_true(self, tmp_path, monkeypatch):
        from agent import indexer as idx_module
        monkeypatch.setattr(idx_module, "get_embed_provider", lambda _: _DummyEmbedProvider())
        make_py(tmp_path, "sample.py", "def foo(): pass\n")
        with patch("integrations.vscode_bridge.configure_all"), \
             patch("integrations.workspace_detect.find_workspace_root", return_value=str(tmp_path)), \
             patch.dict("os.environ", {"VECTR_DB_DIR": str(tmp_path / "db"), "VECTR_SEARCH_ONLY": "1"}):
            svc = _RealVectrService(workspace_root=str(tmp_path))
        assert svc.search_only is True

    def test_env_var_absent_means_full_mode(self, tmp_path, monkeypatch):
        from agent import indexer as idx_module
        monkeypatch.setattr(idx_module, "get_embed_provider", lambda _: _DummyEmbedProvider())
        make_py(tmp_path, "sample.py", "def foo(): pass\n")
        with patch("integrations.vscode_bridge.configure_all"), \
             patch("integrations.workspace_detect.find_workspace_root", return_value=str(tmp_path)), \
             patch.dict("os.environ", {"VECTR_DB_DIR": str(tmp_path / "db")}, clear=False):
            import os
            os.environ.pop("VECTR_SEARCH_ONLY", None)
            svc = _RealVectrService(workspace_root=str(tmp_path))
        assert svc.search_only is False

    def test_memory_only_and_search_only_together_raises(self, tmp_path, monkeypatch):
        from agent import indexer as idx_module
        monkeypatch.setattr(idx_module, "get_embed_provider", lambda _: _DummyEmbedProvider())
        make_py(tmp_path, "sample.py", "def foo(): pass\n")
        with patch("integrations.vscode_bridge.configure_all"), \
             patch("integrations.workspace_detect.find_workspace_root", return_value=str(tmp_path)), \
             patch.dict("os.environ", {"VECTR_DB_DIR": str(tmp_path / "db")}):
            with pytest.raises(ValueError):
                _RealVectrService(workspace_root=str(tmp_path), memory_only=True, search_only=True)


# ---------------------------------------------------------------------------
# No notes DB is ever created in search-only mode
# ---------------------------------------------------------------------------

class TestNoNotesDbCreated:
    def test_context_store_is_none_in_search_only_mode(self, tmp_path, monkeypatch):
        svc = _make_service(tmp_path, monkeypatch, search_only=True)
        assert svc._context_store is None

    def test_no_sqlite_file_written_to_disk(self, tmp_path, monkeypatch):
        db_dir = tmp_path / "db"
        svc = _make_service(tmp_path, monkeypatch, search_only=True)
        svc.index(str(tmp_path))
        assert not (db_dir / "working_context.sqlite").exists()

    def test_full_mode_does_create_sqlite_file(self, tmp_path, monkeypatch):
        """Sanity control: full mode DOES create the notes DB (proves the
        assertion above is meaningful, not just a path-typo false negative)."""
        db_dir = tmp_path / "db"
        svc = _make_service(tmp_path, monkeypatch, search_only=False)
        assert svc._context_store is not None
        assert (db_dir / "working_context.sqlite").exists()


# ---------------------------------------------------------------------------
# start_background_index: indexing + watcher run normally; TTL purge skipped
# ---------------------------------------------------------------------------

class TestStartBackgroundIndexSearchOnly:
    def test_search_only_starts_index_thread_and_watcher(self, tmp_path, monkeypatch):
        svc = _make_service(tmp_path, monkeypatch, search_only=True)

        watcher_start_calls = []
        original_watcher_start = svc._watcher.start
        svc._watcher.start = lambda: watcher_start_calls.append(True)

        svc.start_background_index()

        assert svc._index_thread is not None
        assert watcher_start_calls == [True]
        svc._watcher.stop = lambda: None
        svc._watcher.start = original_watcher_start

    def test_search_only_skips_ttl_purge_no_context_store(self, tmp_path, monkeypatch):
        """TTL purge must be a no-op in search-only mode — there is no
        context store to purge (and constructing one just to purge it would
        defeat the entire point: no notes DB should ever be created)."""
        svc = _make_service(tmp_path, monkeypatch, search_only=True)
        assert svc._context_store is None

        with patch.dict("os.environ", {"VECTR_NOTES_TTL_DAYS": "30"}):
            svc.start_background_index()  # must not raise (no context store)

        assert svc._context_store is None


# ---------------------------------------------------------------------------
# service.status() — mode field
# ---------------------------------------------------------------------------

class TestStatusModeFieldSearchOnly:
    def test_search_only_returns_search_only(self, tmp_path, monkeypatch):
        svc = _make_service(tmp_path, monkeypatch, search_only=True)
        status = svc.status()
        assert status["mode"] == "search-only"

    def test_cli_mode_label_matches_the_service_status_vocabulary(self, tmp_path, monkeypatch):
        """UPG-RESTART-DROPS-MODE drift guard: `main._mode_name` is what the
        instance registry stores and the staleness banner prints, and it has
        to keep meaning the same thing as the daemon's own `status()["mode"]`.
        """
        import main
        svc = _make_service(tmp_path, monkeypatch, search_only=True)
        assert main._mode_name(memory_only=False, search_only=True) == svc.status()["mode"]

    def test_search_only_notes_count_is_zero(self, tmp_path, monkeypatch):
        svc = _make_service(tmp_path, monkeypatch, search_only=True)
        status = svc.status()
        assert status["notes_count"] == 0
        assert svc.count_notes() == 0


# ---------------------------------------------------------------------------
# Memory-facing service methods are guarded in search-only mode
# ---------------------------------------------------------------------------

class TestMemoryLayerGuardedInSearchOnlyMode:
    def test_remember_raises(self, tmp_path, monkeypatch):
        svc = _make_service(tmp_path, monkeypatch, search_only=True)
        with pytest.raises(RuntimeError):
            svc.remember("a finding")

    def test_recall_raises(self, tmp_path, monkeypatch):
        svc = _make_service(tmp_path, monkeypatch, search_only=True)
        with pytest.raises(RuntimeError):
            svc.recall(query="anything")

    def test_forget_note_raises(self, tmp_path, monkeypatch):
        svc = _make_service(tmp_path, monkeypatch, search_only=True)
        with pytest.raises(RuntimeError):
            svc.forget_note(1)

    def test_forget_all_raises(self, tmp_path, monkeypatch):
        svc = _make_service(tmp_path, monkeypatch, search_only=True)
        with pytest.raises(RuntimeError):
            svc.forget_all()

    def test_snapshot_session_raises(self, tmp_path, monkeypatch):
        svc = _make_service(tmp_path, monkeypatch, search_only=True)
        with pytest.raises(RuntimeError):
            svc.snapshot_session("a label")

    def test_list_snapshots_raises(self, tmp_path, monkeypatch):
        svc = _make_service(tmp_path, monkeypatch, search_only=True)
        with pytest.raises(RuntimeError):
            svc.list_snapshots()

    def test_resume_raises(self, tmp_path, monkeypatch):
        svc = _make_service(tmp_path, monkeypatch, search_only=True)
        with pytest.raises(RuntimeError):
            svc.resume()

    def test_search_still_works(self, tmp_path, monkeypatch):
        svc = _make_service(tmp_path, monkeypatch, search_only=True)
        svc.index(str(tmp_path))
        results, _ = svc.search("foo")
        assert isinstance(results, list)

    def test_locate_still_works(self, tmp_path, monkeypatch):
        svc = _make_service(tmp_path, monkeypatch, search_only=True)
        svc.index(str(tmp_path))
        result = svc.locate_with_snippets("foo")
        assert result is not None


# ---------------------------------------------------------------------------
# MCP dispatch: memory tools return the search-only message; evict_hint stays active
# ---------------------------------------------------------------------------

class TestMcpDispatchSearchOnly:
    def _make_mock_service(self, search_only: bool = True):
        svc = MagicMock()
        svc.total_chunks = 100
        svc.memory_only = False
        svc.search_only = search_only
        svc._eviction_advisor = MagicMock()
        svc.auto_eviction_hint.return_value = ""
        return svc

    def test_remember_returns_search_only_message(self):
        from integrations.mcp_server import handle_tools_call
        from app.service import _SEARCH_ONLY_MSG
        svc = self._make_mock_service(search_only=True)

        result = handle_tools_call("vectr_remember", {"content": "a finding"}, svc)
        assert result["isError"] is False
        assert _SEARCH_ONLY_MSG in result["content"][0]["text"]
        svc.remember_with_extras.assert_not_called()

    def test_recall_returns_search_only_message(self):
        from integrations.mcp_server import handle_tools_call
        from app.service import _SEARCH_ONLY_MSG
        svc = self._make_mock_service(search_only=True)

        result = handle_tools_call("vectr_recall", {"query": "anything"}, svc)
        assert result["isError"] is False
        assert _SEARCH_ONLY_MSG in result["content"][0]["text"]
        svc.recall.assert_not_called()

    def test_snapshot_returns_search_only_message(self):
        from integrations.mcp_server import handle_tools_call
        from app.service import _SEARCH_ONLY_MSG
        svc = self._make_mock_service(search_only=True)

        result = handle_tools_call("vectr_snapshot", {"label": "checkpoint"}, svc)
        assert result["isError"] is False
        assert _SEARCH_ONLY_MSG in result["content"][0]["text"]
        svc.snapshot_session.assert_not_called()

    def test_snapshot_list_returns_search_only_message(self):
        from integrations.mcp_server import handle_tools_call
        from app.service import _SEARCH_ONLY_MSG
        svc = self._make_mock_service(search_only=True)

        result = handle_tools_call("vectr_snapshot_list", {}, svc)
        assert result["isError"] is False
        assert _SEARCH_ONLY_MSG in result["content"][0]["text"]
        svc.list_snapshots.assert_not_called()

    def test_forget_returns_search_only_message(self):
        from integrations.mcp_server import handle_tools_call
        from app.service import _SEARCH_ONLY_MSG
        svc = self._make_mock_service(search_only=True)

        result = handle_tools_call("vectr_forget", {"note_id": 1}, svc)
        assert result["isError"] is False
        assert _SEARCH_ONLY_MSG in result["content"][0]["text"]
        svc.forget_note.assert_not_called()

    def test_resume_returns_search_only_message(self):
        from integrations.mcp_server import handle_tools_call
        from app.service import _SEARCH_ONLY_MSG
        svc = self._make_mock_service(search_only=True)

        result = handle_tools_call("vectr_resume", {}, svc)
        assert result["isError"] is False
        assert _SEARCH_ONLY_MSG in result["content"][0]["text"]
        svc.resume.assert_not_called()

    def test_distill_returns_search_only_message(self):
        """memoization-l3-distiller-design §4: arcs/distillation are
        memory-layer, same availability rule as the rest of the memory
        surface — suppressed entirely in search-only mode."""
        from integrations.mcp_server import handle_tools_call
        from app.service import _SEARCH_ONLY_MSG
        svc = self._make_mock_service(search_only=True)

        result = handle_tools_call("vectr_distill", {}, svc)
        assert result["isError"] is False
        assert _SEARCH_ONLY_MSG in result["content"][0]["text"]
        svc.list_arcs.assert_not_called()

        result = handle_tools_call(
            "vectr_distill", {"dismiss": [1], "reason": "x"}, svc,
        )
        assert result["isError"] is False
        assert _SEARCH_ONLY_MSG in result["content"][0]["text"]
        svc.resolve_arcs_dismissed.assert_not_called()

    def test_evict_hint_stays_active_in_search_only_mode(self):
        """vectr_evict_hint advises on re-retrievable SEARCH chunks, not notes
        — it must remain active in search-only mode."""
        from integrations.mcp_server import handle_tools_call
        svc = self._make_mock_service(search_only=True)
        svc.eviction_hint.return_value = "3 chunks can be re-retrieved"

        result = handle_tools_call("vectr_evict_hint", {}, svc)
        assert result["isError"] is False
        assert "3 chunks can be re-retrieved" in result["content"][0]["text"]

    def test_map_stays_active_in_search_only_mode(self):
        from integrations.mcp_server import handle_tools_call
        svc = self._make_mock_service(search_only=True)
        svc.get_map.return_value = "# Passport\nSome codebase."

        result = handle_tools_call("vectr_map", {}, svc)
        assert result["isError"] is False
        assert "Passport" in result["content"][0]["text"]

    def test_map_save_stays_active_in_search_only_mode(self):
        from integrations.mcp_server import handle_tools_call
        svc = self._make_mock_service(search_only=True)
        svc.save_map.return_value = {"saved": True, "existing_summary": None}

        result = handle_tools_call("vectr_map_save", {"summary": "a summary"}, svc)
        assert result["isError"] is False
        svc.save_map.assert_called_once()

    def test_search_not_blocked_in_search_only_mode(self):
        """vectr_search must always work regardless of mode."""
        from agent.searcher import SearchResult
        from integrations.mcp_server import handle_tools_call

        svc = self._make_mock_service(search_only=True)
        _result = SearchResult(
            file_path="auth.py", lines="1-10", symbol_name="verify_token",
            language="python", score=0.9, content="def verify_token(): ...",
        )
        svc.search.return_value = ([_result], 15)
        svc.identifier_hint_symbols.return_value = []

        result = handle_tools_call("vectr_search", {"query": "auth flow"}, svc)
        assert result["isError"] is False
        assert "verify_token" in result["content"][0]["text"]


# ---------------------------------------------------------------------------
# MCP vectr_status text includes the search-only guidance line
# ---------------------------------------------------------------------------

class TestMcpStatusIncludesSearchOnlyLine:
    def _mock_service_with_mode(self, mode: str):
        svc = MagicMock()
        svc.total_chunks = 0
        svc.memory_only = (mode == "memory-only")
        svc.search_only = (mode == "search-only")
        svc.count_notes.return_value = 0
        svc.status.return_value = {
            "indexed_files": 0,
            "total_chunks": 0,
            "last_indexed": "never",
            "embed_model": "dummy",
            "workspace_root": "/test",
            "symbol_count": 0,
            "languages": [],
            "notes_count": 0,
            "grammars_unavailable": [],
            "mode": mode,
        }
        svc.suggest_instruction_style.return_value = "additive"
        return svc

    def test_status_text_includes_search_only_guidance(self):
        from integrations.mcp_server import handle_tools_call
        svc = self._mock_service_with_mode("search-only")

        result = handle_tools_call("vectr_status", {}, svc)
        text = result["content"][0]["text"]
        assert "search-only" in text
        assert "working-memory tools" in text

    def test_status_text_full_mode_no_search_only_guidance(self):
        from integrations.mcp_server import handle_tools_call
        svc = self._mock_service_with_mode("full")

        result = handle_tools_call("vectr_status", {}, svc)
        text = result["content"][0]["text"]
        assert "search-only mode:" not in text


# ---------------------------------------------------------------------------
# REST routes: memory routes 503 in search-only mode; search/locate/trace/map succeed
# ---------------------------------------------------------------------------

class TestRestRoutesSearchOnly:
    def test_remember_returns_503_in_search_only_mode(self):
        from fastapi.testclient import TestClient
        from api import app
        from tests.conftest import _base_mock_service

        svc = _base_mock_service()
        svc.search_only = True

        with patch("app.service.VectrService", return_value=svc):
            with TestClient(app, raise_server_exceptions=False) as c:
                app.state.service = svc
                resp = c.post("/v1/remember", json={"content": "test note"})
        assert resp.status_code == 503
        body = resp.json()
        assert body["detail"]["error"] == "search_only_mode"

    def test_recall_returns_503_in_search_only_mode(self):
        from fastapi.testclient import TestClient
        from api import app
        from tests.conftest import _base_mock_service

        svc = _base_mock_service()
        svc.search_only = True

        with patch("app.service.VectrService", return_value=svc):
            with TestClient(app, raise_server_exceptions=False) as c:
                app.state.service = svc
                resp = c.post("/v1/recall", json={"query": "anything"})
        assert resp.status_code == 503
        body = resp.json()
        assert body["detail"]["error"] == "search_only_mode"

    def test_snapshot_returns_503_in_search_only_mode(self):
        from fastapi.testclient import TestClient
        from api import app
        from tests.conftest import _base_mock_service

        svc = _base_mock_service()
        svc.search_only = True

        with patch("app.service.VectrService", return_value=svc):
            with TestClient(app, raise_server_exceptions=False) as c:
                app.state.service = svc
                resp = c.post("/v1/snapshot", json={"label": "checkpoint"})
        assert resp.status_code == 503
        body = resp.json()
        assert body["detail"]["error"] == "search_only_mode"

    def test_forget_returns_503_in_search_only_mode(self):
        from fastapi.testclient import TestClient
        from api import app
        from tests.conftest import _base_mock_service

        svc = _base_mock_service()
        svc.search_only = True

        with patch("app.service.VectrService", return_value=svc):
            with TestClient(app, raise_server_exceptions=False) as c:
                app.state.service = svc
                resp = c.post("/v1/forget", json={"all": True})
        assert resp.status_code == 503
        body = resp.json()
        assert body["detail"]["error"] == "search_only_mode"

    def test_resume_returns_503_in_search_only_mode(self):
        from fastapi.testclient import TestClient
        from api import app
        from tests.conftest import _base_mock_service

        svc = _base_mock_service()
        svc.search_only = True

        with patch("app.service.VectrService", return_value=svc):
            with TestClient(app, raise_server_exceptions=False) as c:
                app.state.service = svc
                resp = c.get("/v1/resume")
        assert resp.status_code == 503
        body = resp.json()
        assert body["detail"]["error"] == "search_only_mode"

    def test_memory_clear_returns_503_in_search_only_mode(self):
        from fastapi.testclient import TestClient
        from api import app
        from tests.conftest import _base_mock_service

        svc = _base_mock_service()
        svc.search_only = True

        with patch("app.service.VectrService", return_value=svc):
            with TestClient(app, raise_server_exceptions=False) as c:
                app.state.service = svc
                resp = c.post("/v1/memory/clear")
        assert resp.status_code == 503
        body = resp.json()
        assert body["detail"]["error"] == "search_only_mode"

    def test_arcs_returns_503_in_search_only_mode(self):
        from fastapi.testclient import TestClient
        from api import app
        from tests.conftest import _base_mock_service

        svc = _base_mock_service()
        svc.search_only = True

        with patch("app.service.VectrService", return_value=svc):
            with TestClient(app, raise_server_exceptions=False) as c:
                app.state.service = svc
                resp = c.get("/v1/arcs")
        assert resp.status_code == 503
        body = resp.json()
        assert body["detail"]["error"] == "search_only_mode"

    def test_arcs_dismiss_returns_503_in_search_only_mode(self):
        from fastapi.testclient import TestClient
        from api import app
        from tests.conftest import _base_mock_service

        svc = _base_mock_service()
        svc.search_only = True

        with patch("app.service.VectrService", return_value=svc):
            with TestClient(app, raise_server_exceptions=False) as c:
                app.state.service = svc
                resp = c.post(
                    "/v1/arcs/dismiss",
                    json={"arc_ids": [1], "reason": "test"},
                )
        assert resp.status_code == 503
        body = resp.json()
        assert body["detail"]["error"] == "search_only_mode"

    def test_search_succeeds_in_search_only_mode(self):
        from fastapi.testclient import TestClient
        from api import app
        from tests.conftest import _base_mock_service

        svc = _base_mock_service()
        svc.search_only = True

        with patch("app.service.VectrService", return_value=svc):
            with TestClient(app, raise_server_exceptions=True) as c:
                app.state.service = svc
                resp = c.post("/v1/search", json={"query": "auth flow"})
        assert resp.status_code == 200

    def test_locate_succeeds_in_search_only_mode(self):
        from fastapi.testclient import TestClient
        from api import app
        from tests.conftest import _base_mock_service

        svc = _base_mock_service()
        svc.search_only = True

        with patch("app.service.VectrService", return_value=svc):
            with TestClient(app, raise_server_exceptions=True) as c:
                app.state.service = svc
                resp = c.post("/v1/locate", json={"name": "AuthToken"})
        assert resp.status_code == 200

    def test_trace_succeeds_in_search_only_mode(self):
        from fastapi.testclient import TestClient
        from api import app
        from tests.conftest import _base_mock_service

        svc = _base_mock_service()
        svc.search_only = True

        with patch("app.service.VectrService", return_value=svc):
            with TestClient(app, raise_server_exceptions=True) as c:
                app.state.service = svc
                resp = c.post("/v1/trace", json={"name": "verify_token"})
        assert resp.status_code == 200

    def test_map_succeeds_in_search_only_mode(self):
        from fastapi.testclient import TestClient
        from api import app
        from tests.conftest import _base_mock_service

        svc = _base_mock_service()
        svc.search_only = True

        with patch("app.service.VectrService", return_value=svc):
            with TestClient(app, raise_server_exceptions=True) as c:
                app.state.service = svc
                resp = c.get("/v1/map")
        assert resp.status_code == 200

    def test_status_returns_search_only_mode_field(self):
        from fastapi.testclient import TestClient
        from api import app
        from tests.conftest import _base_mock_service

        svc = _base_mock_service()
        svc.search_only = True
        svc.status.return_value = {
            "indexed_files": 0, "total_chunks": 0,
            "last_indexed": "never", "embed_model": "dummy",
            "workspace_root": "/test", "symbol_count": 0,
            "mode": "search-only",
        }

        with patch("app.service.VectrService", return_value=svc):
            with TestClient(app, raise_server_exceptions=True) as c:
                app.state.service = svc
                resp = c.get("/v1/status")
        assert resp.status_code == 200
        assert resp.json()["mode"] == "search-only"


# ---------------------------------------------------------------------------
# CLI: _do_start passes VECTR_SEARCH_ONLY=1 when search_only=True
# ---------------------------------------------------------------------------

class TestDoStartEnvConstructionSearchOnly:
    def test_search_only_flag_adds_env_var(self, tmp_path):
        import main as m
        from agent.instance_registry import workspace_hash
        from pathlib import Path as _Path

        ws = str(tmp_path)
        wh = workspace_hash(ws)

        captured_env: dict = {}

        def _mock_popen(cmd, env, **kwargs):
            captured_env.update(env)
            proc = MagicMock()
            proc.pid = 99999
            return proc

        # UPG-CLI-START-TEXT-EXIT-CODE: this asserts on the env handed to
        # Popen, captured before the readiness verdict is reached. Readiness
        # is pinned so the subject is the spawn env alone, not whether PID
        # 99999 happens to be alive on the running machine. The exit contract
        # itself is pinned by TestDoStartReadinessBranches.
        # UPG-TEST-HOME-DIR-WRITES-FROM-UNIT-SUITE: redirect Path.home so
        # _do_start's secure_dir(Path.home() / ".vectr"[/logs]) mkdir lives
        # in tmp_path, not the developer's home. The call still runs
        # end-to-end — a real crash in secure_dir would still surface here.
        with patch("subprocess.Popen", side_effect=_mock_popen), \
             patch("main.InstanceRegistry") as MockReg, \
             patch("main._migrate_legacy_files"), \
             patch("main._wait_for_daemon_ready", return_value=True), \
             patch("pathlib.Path.home", lambda: _Path(tmp_path)), \
             patch("builtins.open", MagicMock()):
            MockReg.return_value.register = MagicMock()
            m._do_start(ws, 8765, wh, search_only=True)

        assert captured_env.get("VECTR_SEARCH_ONLY") == "1"

    def test_default_mode_does_not_add_search_only_env_var(self, tmp_path):
        import main as m
        from agent.instance_registry import workspace_hash
        from pathlib import Path as _Path

        ws = str(tmp_path)
        wh = workspace_hash(ws)

        captured_env: dict = {}

        def _mock_popen(cmd, env, **kwargs):
            captured_env.update(env)
            proc = MagicMock()
            proc.pid = 99999
            return proc

        # UPG-CLI-START-TEXT-EXIT-CODE: this asserts on the env handed to
        # Popen, captured before the readiness verdict is reached. Readiness
        # is pinned so the subject is the spawn env alone, not whether PID
        # 99999 happens to be alive on the running machine. The exit contract
        # itself is pinned by TestDoStartReadinessBranches.
        # UPG-TEST-HOME-DIR-WRITES-FROM-UNIT-SUITE: see the comment in the
        # companion test above.
        with patch("subprocess.Popen", side_effect=_mock_popen), \
             patch("main.InstanceRegistry") as MockReg, \
             patch("main._migrate_legacy_files"), \
             patch("main._wait_for_daemon_ready", return_value=True), \
             patch("pathlib.Path.home", lambda: _Path(tmp_path)), \
             patch("builtins.open", MagicMock()):
            MockReg.return_value.register = MagicMock()
            m._do_start(ws, 8765, wh, search_only=False)

        assert captured_env.get("VECTR_SEARCH_ONLY", "") != "1"

    def test_do_start_raises_when_both_modes_set(self, tmp_path):
        import main as m
        from agent.instance_registry import workspace_hash

        ws = str(tmp_path)
        wh = workspace_hash(ws)

        with pytest.raises(ValueError):
            m._do_start(ws, 8765, wh, memory_only=True, search_only=True)

    def test_cmd_start_search_only_arg_threads_to_do_start(self, tmp_path):
        import main as m
        from agent.instance_registry import InstanceRegistry

        ws = str(tmp_path)
        reg = InstanceRegistry(registry_path=tmp_path / "instances.json")

        with patch("main.InstanceRegistry", return_value=reg), \
             patch("agent.instance_registry._is_pid_alive", return_value=False), \
             patch("main._is_pid_alive", return_value=False), \
             patch("agent.instance_registry._port_is_free", return_value=True), \
             patch("main._write_workspace_config"), \
             patch("main._do_start") as mock_do_start:
            args = argparse.Namespace(
                workspace=None, paths=[ws], port=8765, memory_only=False, search_only=True,
            )
            m.cmd_start(args)

        _, call_kwargs = mock_do_start.call_args
        assert call_kwargs.get("search_only") is True

    def test_cmd_restart_search_only_arg_threads_to_do_start(self, tmp_path):
        import main as m
        from agent.instance_registry import InstanceRegistry

        reg = InstanceRegistry(registry_path=tmp_path / "instances.json")

        with patch("main.InstanceRegistry", return_value=reg), \
             patch("agent.instance_registry._port_is_free", return_value=True), \
             patch("main._stop_server"), \
             patch("main._write_workspace_config"), \
             patch("main._do_start") as mock_do_start:
            args = argparse.Namespace(
                workspace=None, paths=[str(tmp_path)], port=8765, memory_only=False, search_only=True,
            )
            m.cmd_restart(args)

        _, call_kwargs = mock_do_start.call_args
        assert call_kwargs.get("search_only") is True

    def test_cmd_start_both_flags_exits_nonzero(self, tmp_path, capsys):
        import main as m

        args = argparse.Namespace(
            workspace=None, paths=[str(tmp_path)], port=8765, memory_only=True, search_only=True,
        )
        with pytest.raises(SystemExit) as exc_info:
            m.cmd_start(args)
        assert exc_info.value.code != 0

    def test_cmd_restart_both_flags_exits_nonzero(self, tmp_path):
        import main as m

        args = argparse.Namespace(
            workspace=None, paths=[str(tmp_path)], port=8765, memory_only=True, search_only=True,
        )
        with pytest.raises(SystemExit) as exc_info:
            m.cmd_restart(args)
        assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# CLAUDE.md rendering: search-only variant has no working-memory section
# ---------------------------------------------------------------------------

class TestClaudeMdSearchOnlyVariant:
    def test_search_only_render_omits_working_memory_section(self):
        """No working-memory tool TABLE or session-start recall instructions —
        vectr_remember may still be named in the one-line explanation of why
        it's disabled, but there is no tool-usage documentation for it."""
        import main as m
        text = m._render_claude_md(hooks_installed=False, search_only=True)
        assert "Working memory —" not in text
        assert "vectr_remember(content" not in text
        assert "call vectr_recall(query=" not in text
        assert "notes_count" not in text

    def test_search_only_render_keeps_search_tools(self):
        import main as m
        text = m._render_claude_md(hooks_installed=False, search_only=True)
        assert "vectr_search" in text
        assert "vectr_locate" in text
        assert "vectr_trace" in text
        assert "vectr_map" in text

    def test_full_mode_render_unregressed(self):
        import main as m
        text = m._render_claude_md(hooks_installed=False, search_only=False)
        assert "Working memory" in text
        assert "vectr_remember" in text

    def test_search_only_variant_also_drops_unsourced_stat_and_uses_keyword_examples(self):
        """UPG-VERBALIZE-CONFIRM / UPG-TRACE-EMPTY-HINT apply to the search-only
        variant too — it carries its own copy of the verbalization sentence and
        the vectr_locate/vectr_trace example row."""
        import main as m
        text = m._render_claude_md(hooks_installed=False, search_only=True)
        assert "26" not in text
        assert "always run one cheap confirming call" in text
        assert 'vectr_locate(name="' in text
        assert 'vectr_trace(name="' in text
        assert 'vectr_locate("' not in text and "vectr_locate('" not in text
        assert 'vectr_trace("' not in text and "vectr_trace('" not in text


class TestGetDaemonMode:
    def test_returns_mode_from_status_endpoint(self):
        import main as m

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"mode": "search-only"}
        mock_resp.raise_for_status = MagicMock()
        with patch("httpx.get", return_value=mock_resp):
            assert m._get_daemon_mode(8765) == "search-only"

    def test_returns_none_when_daemon_unreachable(self):
        import main as m
        with patch("httpx.get", side_effect=Exception("connection refused")):
            assert m._get_daemon_mode(8765) is None
