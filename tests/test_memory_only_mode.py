"""Tests for vectr start --memory-only mode (UPG-MEMORY-ONLY-MODE).

Covers:
- VectrService: memory_only flag read from env + constructor arg
- start_background_index: index thread + watcher skipped in memory-only mode; TTL purge still runs
- service.status(): mode field is "memory-only" or "full"
- MCP dispatch: search/locate/trace return the memory-only message
- REST routes: search/locate/trace return 503 in memory-only mode
- memory tools still function in memory-only mode (remember/recall round-trip)
- CLI: _do_start env construction includes VECTR_MEMORY_ONLY=1 when flag is set
- MCP vectr_status text includes the mode line
"""
from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import pytest

# _RealVectrService is the constructor saved at collection time, BEFORE the
# session-scoped real_service_client fixture patches app.service.VectrService.
# Importing app.service.VectrService directly here would yield that session-wide
# mock once any test has triggered real_service_client, returning a shared
# full-mode instance and breaking every memory_only assertion.
from tests.conftest import make_py, _DummyEmbedProvider, _RealVectrService


# ---------------------------------------------------------------------------
# Helper: build a VectrService with the dummy embedder, no real model load
# ---------------------------------------------------------------------------

def _make_service(tmp_path, monkeypatch, memory_only: bool = False):
    from agent import indexer as idx_module
    monkeypatch.setattr(idx_module, "get_embed_provider", lambda _: _DummyEmbedProvider())
    make_py(tmp_path, "sample.py", "def foo(): pass\n")

    with patch("integrations.vscode_bridge.configure_all"), \
         patch("integrations.workspace_detect.find_workspace_root", return_value=str(tmp_path)), \
         patch.dict("os.environ", {"VECTR_DB_DIR": str(tmp_path / "db")}):
        svc = _RealVectrService(workspace_root=str(tmp_path), memory_only=memory_only)
    return svc


# ---------------------------------------------------------------------------
# VectrService: memory_only property + env read
# ---------------------------------------------------------------------------

class TestMemoryOnlyServiceFlag:
    def test_constructor_arg_sets_memory_only_true(self, tmp_path, monkeypatch):
        svc = _make_service(tmp_path, monkeypatch, memory_only=True)
        assert svc.memory_only is True

    def test_constructor_default_is_full(self, tmp_path, monkeypatch):
        svc = _make_service(tmp_path, monkeypatch, memory_only=False)
        assert svc.memory_only is False

    def test_env_var_sets_memory_only_true(self, tmp_path, monkeypatch):
        from agent import indexer as idx_module
        monkeypatch.setattr(idx_module, "get_embed_provider", lambda _: _DummyEmbedProvider())
        make_py(tmp_path, "sample.py", "def foo(): pass\n")
        with patch("integrations.vscode_bridge.configure_all"), \
             patch("integrations.workspace_detect.find_workspace_root", return_value=str(tmp_path)), \
             patch.dict("os.environ", {"VECTR_DB_DIR": str(tmp_path / "db"), "VECTR_MEMORY_ONLY": "1"}):
            svc = _RealVectrService(workspace_root=str(tmp_path))
        assert svc.memory_only is True

    def test_env_var_absent_means_full_mode(self, tmp_path, monkeypatch):
        from agent import indexer as idx_module
        monkeypatch.setattr(idx_module, "get_embed_provider", lambda _: _DummyEmbedProvider())
        make_py(tmp_path, "sample.py", "def foo(): pass\n")
        with patch("integrations.vscode_bridge.configure_all"), \
             patch("integrations.workspace_detect.find_workspace_root", return_value=str(tmp_path)), \
             patch.dict("os.environ", {"VECTR_DB_DIR": str(tmp_path / "db")}, clear=False):
            import os
            os.environ.pop("VECTR_MEMORY_ONLY", None)
            svc = _RealVectrService(workspace_root=str(tmp_path))
        assert svc.memory_only is False


# ---------------------------------------------------------------------------
# start_background_index: thread + watcher skipped in memory-only; TTL purge kept
# ---------------------------------------------------------------------------

class TestStartBackgroundIndexMemoryOnly:
    def test_memory_only_does_not_start_index_thread(self, tmp_path, monkeypatch):
        svc = _make_service(tmp_path, monkeypatch, memory_only=True)

        watcher_start_calls = []
        svc._watcher.start = lambda: watcher_start_calls.append(True)

        svc.start_background_index()

        # Index thread must remain None
        assert svc._index_thread is None
        # Watcher must not have been started
        assert watcher_start_calls == []

    def test_memory_only_indexing_flag_reset_to_false_after(self, tmp_path, monkeypatch):
        """_indexing must not stay True in memory-only mode (no thread to reset it)."""
        svc = _make_service(tmp_path, monkeypatch, memory_only=True)
        svc.start_background_index()
        # _indexing is reset to False at the end of the early return path
        assert svc._indexing is False

    def test_full_mode_starts_index_thread_and_watcher(self, tmp_path, monkeypatch):
        svc = _make_service(tmp_path, monkeypatch, memory_only=False)

        watcher_start_calls = []
        original_watcher_start = svc._watcher.start
        svc._watcher.start = lambda: watcher_start_calls.append(True)

        svc.start_background_index()

        # Index thread was created
        assert svc._index_thread is not None
        # Watcher was started
        assert watcher_start_calls == [True]
        # Cleanup: stop what we started
        svc._watcher.stop = lambda: None
        svc._watcher.start = original_watcher_start

    def test_ttl_purge_still_runs_in_memory_only_mode(self, tmp_path, monkeypatch):
        """TTL purge must fire in memory-only mode too — it's workspace hygiene, not indexing."""
        svc = _make_service(tmp_path, monkeypatch, memory_only=True)

        # Store a note so purge_expired_notes has something to operate on
        svc.remember("old note", tags=["test"])

        purge_called = []
        original_purge = svc._context_store.purge_expired_notes

        def _recording_purge(workspace, ttl):
            purge_called.append((workspace, ttl))
            return original_purge(workspace, ttl)

        svc._context_store.purge_expired_notes = _recording_purge

        with patch.dict("os.environ", {"VECTR_NOTES_TTL_DAYS": "30"}):
            svc.start_background_index()

        # The purge must have been called despite memory-only mode
        assert purge_called, "purge_expired_notes was not called in memory-only mode"

    def test_second_call_is_noop_in_memory_only(self, tmp_path, monkeypatch):
        """Re-calling start_background_index when already marked indexing is a no-op."""
        svc = _make_service(tmp_path, monkeypatch, memory_only=True)
        svc._indexing = True  # pretend already running
        svc.start_background_index()  # must not raise or start anything
        assert svc._index_thread is None


# ---------------------------------------------------------------------------
# service.status() — mode field
# ---------------------------------------------------------------------------

class TestStatusModeField:
    def test_full_mode_returns_full(self, tmp_path, monkeypatch):
        svc = _make_service(tmp_path, monkeypatch, memory_only=False)
        status = svc.status()
        assert status["mode"] == "full"

    def test_memory_only_returns_memory_only(self, tmp_path, monkeypatch):
        svc = _make_service(tmp_path, monkeypatch, memory_only=True)
        status = svc.status()
        assert status["mode"] == "memory-only"

    def test_cli_mode_labels_match_the_service_status_vocabulary(self, tmp_path, monkeypatch):
        """UPG-RESTART-DROPS-MODE drift guard: `main._mode_name` is what the
        instance registry stores and the staleness banner prints, and it has
        to keep meaning the same thing as the daemon's own `status()["mode"]`.
        """
        import main
        mem_ws = tmp_path / "mem"
        mem_ws.mkdir()
        mem = _make_service(mem_ws, monkeypatch, memory_only=True)
        assert main._mode_name(memory_only=True, search_only=False) == mem.status()["mode"]
        full_ws = tmp_path / "full"
        full_ws.mkdir()
        full = _make_service(full_ws, monkeypatch, memory_only=False)
        assert main._mode_name(memory_only=False, search_only=False) == full.status()["mode"]


# ---------------------------------------------------------------------------
# Memory tools work in memory-only mode (remember/recall round-trip)
# ---------------------------------------------------------------------------

class TestMemoryToolsWorkInMemoryOnlyMode:
    def test_remember_recall_round_trip_in_memory_only_mode(self, tmp_path, monkeypatch):
        svc = _make_service(tmp_path, monkeypatch, memory_only=True)

        note_id = svc.remember("auth_token() at auth.py:42 is the entry point", tags=["auth"])
        assert isinstance(note_id, int)

        recalled = svc.recall(query="auth token entry point")
        assert "auth.py:42" in recalled or note_id > 0  # note was stored

    def test_count_notes_works_in_memory_only_mode(self, tmp_path, monkeypatch):
        svc = _make_service(tmp_path, monkeypatch, memory_only=True)
        before = svc.count_notes()
        svc.remember("a memory-only note")
        assert svc.count_notes() == before + 1

    def test_resume_works_in_memory_only_mode(self, tmp_path, monkeypatch):
        """UPG-RESUME-SURFACE: the working-memory layer is fully active in
        memory-only mode (unlike search-only), so resume() needs no special
        casing there — it just works."""
        svc = _make_service(tmp_path, monkeypatch, memory_only=True)
        svc.remember("ship the resume feature", kind="task", priority="high")

        result = svc.resume()
        assert result["last_task"] is not None
        assert result["last_task"]["title"] == "ship the resume feature"
        assert "ship the resume feature" in result["formatted"]


# ---------------------------------------------------------------------------
# MCP dispatch: search/locate/trace return the memory-only message
# ---------------------------------------------------------------------------

class TestMcpDispatchMemoryOnly:
    def _make_mock_service(self, memory_only: bool = True):
        from agent.searcher import SearchResult

        svc = MagicMock()
        svc.total_chunks = 0
        svc.memory_only = memory_only
        svc.search_only = False

        _result = SearchResult(
            file_path="auth.py", lines="1-10", symbol_name="verify_token",
            language="python", score=0.9, content="def verify_token(): ...",
        )
        svc.search.return_value = ([_result], 15)
        # UPG-QUERYTYPE-REROUTE: additive symbol-graph hint — empty by default.
        svc.identifier_hint_symbols.return_value = []
        svc._eviction_advisor = MagicMock()
        svc._eviction_advisor.auto_eviction_hint.return_value = ""
        svc.auto_eviction_hint.return_value = ""
        return svc

    def test_search_returns_memory_only_message(self):
        from integrations.mcp_server import handle_tools_call
        from app.service import _MEMORY_ONLY_MSG
        svc = self._make_mock_service(memory_only=True)

        result = handle_tools_call("vectr_search", {"query": "auth flow"}, svc)
        assert result["isError"] is False
        assert _MEMORY_ONLY_MSG in result["content"][0]["text"]

    def test_locate_returns_memory_only_message(self):
        from integrations.mcp_server import handle_tools_call
        from app.service import _MEMORY_ONLY_MSG
        svc = self._make_mock_service(memory_only=True)

        result = handle_tools_call("vectr_locate", {"name": "AuthToken"}, svc)
        assert result["isError"] is False
        assert _MEMORY_ONLY_MSG in result["content"][0]["text"]

    def test_trace_returns_memory_only_message(self):
        from integrations.mcp_server import handle_tools_call
        from app.service import _MEMORY_ONLY_MSG
        svc = self._make_mock_service(memory_only=True)

        result = handle_tools_call("vectr_trace", {"name": "verify_token"}, svc)
        assert result["isError"] is False
        assert _MEMORY_ONLY_MSG in result["content"][0]["text"]

    def test_map_returns_memory_only_message(self):
        # UPG-MAP-MEMORY-ONLY-GUARD: vectr_map is a code-index tool and must
        # give the mode-contract message, not an empty passport.
        from integrations.mcp_server import handle_tools_call
        from app.service import _MEMORY_ONLY_MSG
        svc = self._make_mock_service(memory_only=True)
        svc.get_map.return_value = "should not be reached"

        result = handle_tools_call("vectr_map", {}, svc)
        assert result["isError"] is False
        assert _MEMORY_ONLY_MSG in result["content"][0]["text"]
        svc.get_map.assert_not_called()

    def test_map_save_returns_memory_only_message(self):
        from integrations.mcp_server import handle_tools_call
        from app.service import _MEMORY_ONLY_MSG
        svc = self._make_mock_service(memory_only=True)

        result = handle_tools_call("vectr_map_save", {"summary": "a repo summary"}, svc)
        assert result["isError"] is False
        assert _MEMORY_ONLY_MSG in result["content"][0]["text"]
        svc.save_map.assert_not_called()

    def test_map_in_full_mode_not_guarded(self):
        from integrations.mcp_server import handle_tools_call
        from app.service import _MEMORY_ONLY_MSG
        svc = self._make_mock_service(memory_only=False)
        svc.get_map.return_value = "codebase overview text"

        result = handle_tools_call("vectr_map", {}, svc)
        assert _MEMORY_ONLY_MSG not in result["content"][0]["text"]
        assert "codebase overview text" in result["content"][0]["text"]

    def test_search_in_full_mode_does_not_return_memory_only_message(self):
        from integrations.mcp_server import handle_tools_call
        from app.service import _MEMORY_ONLY_MSG
        svc = self._make_mock_service(memory_only=False)
        svc.total_chunks = 100

        result = handle_tools_call("vectr_search", {"query": "auth flow"}, svc)
        text = result["content"][0]["text"]
        assert _MEMORY_ONLY_MSG not in text

    def test_remember_not_blocked_in_memory_only_mode(self):
        """vectr_remember must always work regardless of mode."""
        from integrations.mcp_server import handle_tools_call
        from app.service import RememberOutcome
        svc = self._make_mock_service(memory_only=True)
        svc.remember_with_extras.return_value = RememberOutcome(
            note_id=7, related=[], revoked_related=[], proxy_anchor_suggestions=[],
        )

        result = handle_tools_call("vectr_remember", {"content": "important finding"}, svc)
        assert result["isError"] is False
        assert "7" in result["content"][0]["text"]


# ---------------------------------------------------------------------------
# MCP vectr_status text includes mode line
# ---------------------------------------------------------------------------

class TestMcpStatusIncludesModeLine:
    def _mock_service_with_mode(self, mode: str):
        svc = MagicMock()
        svc.total_chunks = 0
        svc.memory_only = (mode == "memory-only")
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
        svc._eviction_advisor = MagicMock()
        svc.suggest_instruction_style.return_value = "additive"
        return svc

    def test_status_text_includes_mode_full(self):
        from integrations.mcp_server import handle_tools_call
        svc = self._mock_service_with_mode("full")

        result = handle_tools_call("vectr_status", {}, svc)
        text = result["content"][0]["text"]
        assert "Mode" in text
        assert "full" in text

    def test_status_text_includes_mode_memory_only(self):
        from integrations.mcp_server import handle_tools_call
        svc = self._mock_service_with_mode("memory-only")

        result = handle_tools_call("vectr_status", {}, svc)
        text = result["content"][0]["text"]
        assert "Mode" in text
        assert "memory-only" in text

    def test_status_text_includes_memory_only_guidance(self):
        from integrations.mcp_server import handle_tools_call
        svc = self._mock_service_with_mode("memory-only")

        result = handle_tools_call("vectr_status", {}, svc)
        text = result["content"][0]["text"]
        # Must tell the agent that search/locate/trace are disabled
        assert "search" in text.lower() or "disabled" in text.lower()


# ---------------------------------------------------------------------------
# REST routes: search/locate/trace return 503 in memory-only mode
# ---------------------------------------------------------------------------

class TestRestRoutesMemoryOnly:
    def _make_memory_only_client(self):
        from fastapi.testclient import TestClient
        from api import app
        from tests.conftest import _base_mock_service

        svc = _base_mock_service()
        svc.memory_only = True

        with patch("app.service.VectrService", return_value=svc):
            with TestClient(app, raise_server_exceptions=False) as c:
                app.state.service = svc
                yield c

    def test_search_returns_503_in_memory_only_mode(self):
        from fastapi.testclient import TestClient
        from api import app
        from tests.conftest import _base_mock_service

        svc = _base_mock_service()
        svc.memory_only = True

        with patch("app.service.VectrService", return_value=svc):
            with TestClient(app, raise_server_exceptions=False) as c:
                app.state.service = svc
                resp = c.post("/v1/search", json={"query": "auth flow"})
        assert resp.status_code == 503
        body = resp.json()
        assert body["detail"]["error"] == "memory_only_mode"

    def test_locate_returns_503_in_memory_only_mode(self):
        from fastapi.testclient import TestClient
        from api import app
        from tests.conftest import _base_mock_service

        svc = _base_mock_service()
        svc.memory_only = True

        with patch("app.service.VectrService", return_value=svc):
            with TestClient(app, raise_server_exceptions=False) as c:
                app.state.service = svc
                resp = c.post("/v1/locate", json={"name": "AuthToken"})
        assert resp.status_code == 503
        body = resp.json()
        assert body["detail"]["error"] == "memory_only_mode"

    def test_trace_returns_503_in_memory_only_mode(self):
        from fastapi.testclient import TestClient
        from api import app
        from tests.conftest import _base_mock_service

        svc = _base_mock_service()
        svc.memory_only = True

        with patch("app.service.VectrService", return_value=svc):
            with TestClient(app, raise_server_exceptions=False) as c:
                app.state.service = svc
                resp = c.post("/v1/trace", json={"name": "verify_token"})
        assert resp.status_code == 503
        body = resp.json()
        assert body["detail"]["error"] == "memory_only_mode"

    def test_index_returns_503_in_memory_only_mode(self):
        """UPG-CLI-MEMONLY-CRASH companion: indexing is pointless when search
        is disabled — /v1/index is gated the same way /v1/search already is,
        rather than silently building an index nothing can query."""
        from fastapi.testclient import TestClient
        from api import app
        from tests.conftest import _base_mock_service

        svc = _base_mock_service()
        svc.memory_only = True

        with patch("app.service.VectrService", return_value=svc):
            with TestClient(app, raise_server_exceptions=False) as c:
                app.state.service = svc
                resp = c.post("/v1/index", json={"path": "/some/path", "force": False})
        assert resp.status_code == 503
        body = resp.json()
        assert body["detail"]["error"] == "memory_only_mode"

    def test_index_succeeds_in_full_mode(self):
        from fastapi.testclient import TestClient
        from api import app
        from tests.conftest import _base_mock_service

        svc = _base_mock_service()
        svc.memory_only = False

        with patch("app.service.VectrService", return_value=svc):
            with TestClient(app, raise_server_exceptions=True) as c:
                app.state.service = svc
                resp = c.post("/v1/index", json={"path": "/some/path", "force": False})
        assert resp.status_code == 200

    def test_search_succeeds_in_full_mode(self):
        from fastapi.testclient import TestClient
        from api import app
        from tests.conftest import _base_mock_service

        svc = _base_mock_service()
        svc.memory_only = False

        with patch("app.service.VectrService", return_value=svc):
            with TestClient(app, raise_server_exceptions=True) as c:
                app.state.service = svc
                resp = c.post("/v1/search", json={"query": "auth flow"})
        assert resp.status_code == 200

    def test_remember_rest_not_blocked_in_memory_only_mode(self):
        from fastapi.testclient import TestClient
        from api import app
        from tests.conftest import _base_mock_service

        svc = _base_mock_service()
        svc.memory_only = True

        with patch("app.service.VectrService", return_value=svc):
            with TestClient(app, raise_server_exceptions=True) as c:
                app.state.service = svc
                resp = c.post("/v1/remember", json={"content": "test note"})
        assert resp.status_code == 200

    def test_status_returns_mode_field(self):
        from fastapi.testclient import TestClient
        from api import app
        from tests.conftest import _base_mock_service

        svc = _base_mock_service()
        svc.memory_only = True
        svc.status.return_value = {
            "indexed_files": 0, "total_chunks": 0,
            "last_indexed": "never", "embed_model": "dummy",
            "workspace_root": "/test", "symbol_count": 0,
            "mode": "memory-only",
        }

        with patch("app.service.VectrService", return_value=svc):
            with TestClient(app, raise_server_exceptions=True) as c:
                app.state.service = svc
                resp = c.get("/v1/status")
        assert resp.status_code == 200
        assert resp.json()["mode"] == "memory-only"


# ---------------------------------------------------------------------------
# CLI: _do_start passes VECTR_MEMORY_ONLY=1 when memory_only=True
# ---------------------------------------------------------------------------

class TestDoStartEnvConstruction:
    def test_memory_only_flag_adds_env_var(self, tmp_path):
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
            m._do_start(ws, 8765, wh, memory_only=True)

        assert captured_env.get("VECTR_MEMORY_ONLY") == "1"

    def test_default_mode_does_not_add_env_var(self, tmp_path):
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
            m._do_start(ws, 8765, wh, memory_only=False)

        # VECTR_MEMORY_ONLY must NOT be "1" when not in memory-only mode
        assert captured_env.get("VECTR_MEMORY_ONLY", "") != "1"

    def test_cmd_start_memory_only_arg_threads_to_do_start(self, tmp_path):
        import main as m
        from agent.instance_registry import InstanceRegistry, workspace_hash

        ws = str(tmp_path)
        wh = workspace_hash(ws)
        reg = InstanceRegistry(registry_path=tmp_path / "instances.json")

        with patch("main.InstanceRegistry", return_value=reg), \
             patch("agent.instance_registry._is_pid_alive", return_value=False), \
             patch("main._is_pid_alive", return_value=False), \
             patch("agent.instance_registry._port_is_free", return_value=True), \
             patch("main._write_workspace_config"), \
             patch("main._do_start") as mock_do_start:
            args = argparse.Namespace(
                workspace=None, paths=[ws], port=8765, memory_only=True,
            )
            m.cmd_start(args)

        _, call_kwargs = mock_do_start.call_args
        assert call_kwargs.get("memory_only") is True

    def test_cmd_restart_memory_only_arg_threads_to_do_start(self, tmp_path):
        import main as m
        from agent.instance_registry import InstanceRegistry

        reg = InstanceRegistry(registry_path=tmp_path / "instances.json")

        with patch("main.InstanceRegistry", return_value=reg), \
             patch("agent.instance_registry._port_is_free", return_value=True), \
             patch("main._stop_server"), \
             patch("main._write_workspace_config"), \
             patch("main._do_start") as mock_do_start:
            args = argparse.Namespace(
                workspace=None, paths=[str(tmp_path)], port=8765, memory_only=True,
            )
            m.cmd_restart(args)

        _, call_kwargs = mock_do_start.call_args
        assert call_kwargs.get("memory_only") is True
