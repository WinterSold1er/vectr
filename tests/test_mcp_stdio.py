"""Tests for the MCP stdio transport (`vectr mcp-stdio`).

Newline-delimited JSON-RPC 2.0 on stdin/stdout — no daemon, no port. For MCP
clients and hosting platforms that spawn the server as a subprocess.

Layers tested:
  TestServiceHandle          — the thread-safe holder, no I/O
  TestDispatchLine            — line-codec/dispatch glue (mocked VectrService)
  TestRunStdioLoop            — the read/dispatch/write loop, in-process (StringIO)
  TestConfigureIdeOptOut      — the configure_ide constructor param this
                                 transport depends on to skip HTTP-pointing
                                 IDE config writes, plus its CLI env propagation
  TestMcpStdioSubprocess      — real subprocess, fast path only (no model load
                                 required — every assertion holds true whether
                                 or not the background VectrService has
                                 finished constructing)
  TestMcpStdioSubprocessReady — real subprocess, waits for the workspace index
                                 to actually finish (real embedder + reranker
                                 load); marked integration like the rest of the
                                 suite's real-model tests
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import _base_mock_service

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MAIN_PY = _REPO_ROOT / "main.py"


# ---------------------------------------------------------------------------
# ServiceHandle
# ---------------------------------------------------------------------------

class TestServiceHandle:
    def test_not_ready_initially(self) -> None:
        from integrations.mcp_server import ServiceHandle
        handle = ServiceHandle()
        assert handle.is_ready is False
        assert handle.service is None
        assert handle.error is None

    def test_set_service_marks_ready(self) -> None:
        from integrations.mcp_server import ServiceHandle
        handle = ServiceHandle()
        svc = _base_mock_service()
        handle.set_service(svc)
        assert handle.is_ready is True
        assert handle.service is svc
        assert handle.error is None

    def test_set_error_marks_ready_with_error_set(self) -> None:
        from integrations.mcp_server import ServiceHandle
        handle = ServiceHandle()
        exc = RuntimeError("boom")
        handle.set_error(exc)
        assert handle.is_ready is True
        assert handle.service is None
        assert handle.error is exc


# ---------------------------------------------------------------------------
# dispatch_line — protocol glue against a mocked VectrService
# ---------------------------------------------------------------------------

class TestDispatchLine:
    def test_initialize_returns_protocol_version(self) -> None:
        from integrations.mcp_server import ServiceHandle, dispatch_line
        handle = ServiceHandle()
        resp = dispatch_line(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                        "clientInfo": {"name": "test", "version": "1.0"}}},
            handle, "session-1",
        )
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 1
        assert resp["result"]["protocolVersion"] == "2024-11-05"
        assert resp["result"]["serverInfo"]["name"] == "vectr"

    def test_initialize_answers_before_service_ready(self) -> None:
        """The whole point of ServiceHandle: initialize never waits on it."""
        from integrations.mcp_server import ServiceHandle, dispatch_line
        handle = ServiceHandle()
        assert handle.is_ready is False
        resp = dispatch_line({"jsonrpc": "2.0", "id": 1, "method": "initialize"}, handle, "s")
        assert "result" in resp

    def test_notifications_initialized_returns_none(self) -> None:
        from integrations.mcp_server import ServiceHandle, dispatch_line
        handle = ServiceHandle()
        resp = dispatch_line({"jsonrpc": "2.0", "method": "notifications/initialized"}, handle, "s")
        assert resp is None

    def test_ping_returns_ok(self) -> None:
        from integrations.mcp_server import ServiceHandle, dispatch_line
        handle = ServiceHandle()
        resp = dispatch_line({"jsonrpc": "2.0", "id": 5, "method": "ping"}, handle, "s")
        assert resp["result"] == {}
        assert "error" not in resp

    def test_unknown_method_returns_method_not_found(self) -> None:
        from integrations.mcp_server import ServiceHandle, dispatch_line
        handle = ServiceHandle()
        resp = dispatch_line({"jsonrpc": "2.0", "id": 1, "method": "nonexistent/method"}, handle, "s")
        assert resp["error"]["code"] == -32601

    def test_tools_list_before_service_ready_returns_base_tools(self) -> None:
        """service=None must never raise — handle_tools_list already tolerates
        it (falls through to the base tool set), which is exactly what a
        not-yet-constructed service looks like from dispatch_line's side."""
        from integrations.mcp_server import ServiceHandle, dispatch_line
        handle = ServiceHandle()
        resp = dispatch_line({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, handle, "s")
        assert "tools" in resp["result"]
        assert len(resp["result"]["tools"]) > 0

    def test_tools_list_all_tools_env_independent_of_service(self, monkeypatch) -> None:
        from integrations.mcp_server import ServiceHandle, dispatch_line
        monkeypatch.setenv("VECTR_MCP_ALL_TOOLS", "1")
        handle = ServiceHandle()  # never set — service stays None throughout
        resp = dispatch_line({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, handle, "s")
        names = {t["name"] for t in resp["result"]["tools"]}
        assert "vectr_recall" in names  # a memory-read tool, gated off in the base set

    def test_tools_call_missing_name_returns_invalid_params(self) -> None:
        from integrations.mcp_server import ServiceHandle, dispatch_line
        handle = ServiceHandle()
        resp = dispatch_line(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"arguments": {}}},
            handle, "s",
        )
        assert resp["error"]["code"] == -32602

    def test_tools_call_before_service_ready_is_graceful(self) -> None:
        """Mirrors the existing "still indexing" degradation in
        handle_tools_call's vectr_search branch, one level up: the service
        itself isn't constructed yet."""
        from integrations.mcp_server import ServiceHandle, dispatch_line
        handle = ServiceHandle()
        resp = dispatch_line(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
             "params": {"name": "vectr_status", "arguments": {}}},
            handle, "s",
        )
        assert resp["result"]["isError"] is False
        assert "starting up" in resp["result"]["content"][0]["text"]

    def test_tools_call_after_service_error_reports_error(self) -> None:
        from integrations.mcp_server import ServiceHandle, dispatch_line
        handle = ServiceHandle()
        handle.set_error(RuntimeError("model load failed"))
        resp = dispatch_line(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
             "params": {"name": "vectr_status", "arguments": {}}},
            handle, "s",
        )
        assert resp["result"]["isError"] is True
        assert "model load failed" in resp["result"]["content"][0]["text"]

    def test_tools_call_dispatches_to_real_service_once_ready(self) -> None:
        from integrations.mcp_server import ServiceHandle, dispatch_line
        handle = ServiceHandle()
        handle.set_service(_base_mock_service())
        resp = dispatch_line(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
             "params": {"name": "vectr_status", "arguments": {}}},
            handle, "s",
        )
        assert resp["result"]["isError"] is False
        text = resp["result"]["content"][0]["text"]
        assert "workspace" in text.lower() or "indexed" in text.lower()

    def test_jsonrpc_id_echoed(self) -> None:
        from integrations.mcp_server import ServiceHandle, dispatch_line
        handle = ServiceHandle()
        resp = dispatch_line({"jsonrpc": "2.0", "id": 42, "method": "ping"}, handle, "s")
        assert resp["id"] == 42


# ---------------------------------------------------------------------------
# UPG-STDIO-MEMORY-READY: memory tools dispatch as soon as the service object
# exists (phase 1), independent of `service.fully_ready` (phase 2 — embedder/
# indexer/searcher/watcher/symbol graph); search-side tools still wait for
# `fully_ready`. Gating is keyed on tool NAME + service STATE only.
# ---------------------------------------------------------------------------

_MEMORY_TOOL_ARGS = {
    "vectr_remember": {"content": "a note stored during the phase-2 warm-up window"},
    "vectr_recall": {},
    "vectr_forget": {"all": True},
    "vectr_status": {},
    "vectr_snapshot": {"label": "warm-up-checkpoint"},
    "vectr_snapshot_list": {},
}

_SEARCH_TOOL_ARGS = {
    "vectr_search": {"query": "auth flow"},
    "vectr_locate": {"name": "verify_token"},
    "vectr_trace": {"name": "verify_token"},
    "vectr_map": {},
    "vectr_fetch": {"ids": ["src/auth.py:10-30"]},
    "vectr_ingest_traces": {"events": []},
    "vectr_evict_hint": {},
}


class TestMemoryReadyGating:
    def _handle_not_fully_ready(self):
        from integrations.mcp_server import ServiceHandle
        handle = ServiceHandle()
        svc = _base_mock_service()
        svc.fully_ready = False  # phase 1 done, phase 2 (embedder/indexer/...) not
        handle.set_service(svc)
        return handle

    def _handle_fully_ready(self):
        from integrations.mcp_server import ServiceHandle
        handle = ServiceHandle()
        svc = _base_mock_service()
        svc.fully_ready = True
        handle.set_service(svc)
        return handle

    @pytest.mark.parametrize("tool_name", sorted(_MEMORY_TOOL_ARGS))
    def test_memory_tool_served_while_not_fully_ready(self, tool_name) -> None:
        from integrations.mcp_server import dispatch_line
        handle = self._handle_not_fully_ready()
        resp = dispatch_line(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
             "params": {"name": tool_name, "arguments": _MEMORY_TOOL_ARGS[tool_name]}},
            handle, "s",
        )
        text = resp["result"]["content"][0]["text"]
        assert "starting up" not in text, f"{tool_name} was gated but should be memory-ready"

    @pytest.mark.parametrize("tool_name", sorted(_SEARCH_TOOL_ARGS))
    def test_search_tool_gated_while_not_fully_ready(self, tool_name) -> None:
        from integrations.mcp_server import dispatch_line
        handle = self._handle_not_fully_ready()
        resp = dispatch_line(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
             "params": {"name": tool_name, "arguments": _SEARCH_TOOL_ARGS[tool_name]}},
            handle, "s",
        )
        assert resp["result"]["isError"] is False
        assert "starting up" in resp["result"]["content"][0]["text"]

    @pytest.mark.parametrize("tool_name", sorted(_SEARCH_TOOL_ARGS))
    def test_search_tool_dispatches_once_fully_ready(self, tool_name) -> None:
        """Same tools, `fully_ready=True` — must reach the real service (mocked
        here) rather than the still-starting-up placeholder."""
        from integrations.mcp_server import dispatch_line
        handle = self._handle_fully_ready()
        resp = dispatch_line(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
             "params": {"name": tool_name, "arguments": _SEARCH_TOOL_ARGS[tool_name]}},
            handle, "s",
        )
        assert "starting up" not in resp["result"]["content"][0]["text"]

    def test_initialize_tools_list_ping_unaffected_by_fully_ready(self) -> None:
        """The three fast, model-independent methods must answer identically
        whether `fully_ready` is True or False — they never touch it."""
        from integrations.mcp_server import dispatch_line
        handle = self._handle_not_fully_ready()

        init_resp = dispatch_line(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                        "clientInfo": {"name": "test", "version": "1.0"}}},
            handle, "s",
        )
        assert init_resp["result"]["protocolVersion"] == "2024-11-05"

        list_resp = dispatch_line({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, handle, "s")
        assert "tools" in list_resp["result"]

        ping_resp = dispatch_line({"jsonrpc": "2.0", "id": 3, "method": "ping"}, handle, "s")
        assert ping_resp["result"] == {}


# ---------------------------------------------------------------------------
# run_stdio_loop — the read/dispatch/write loop, in-process
# ---------------------------------------------------------------------------

class TestRunStdioLoop:
    def _run(self, lines: list[str], handle=None):
        import io
        from integrations.mcp_server import ServiceHandle, run_stdio_loop

        if handle is None:
            handle = ServiceHandle()
            handle.set_service(_base_mock_service())
        stdin = io.StringIO("\n".join(lines) + "\n")
        stdout = io.StringIO()
        run_stdio_loop(handle, stdin=stdin, stdout=stdout)
        return [ln for ln in stdout.getvalue().split("\n") if ln]

    def test_initialize_then_tools_list(self) -> None:
        out = self._run([
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
            json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
        ])
        assert len(out) == 2
        r1, r2 = json.loads(out[0]), json.loads(out[1])
        assert r1["id"] == 1
        assert r2["id"] == 2
        assert "tools" in r2["result"]

    def test_notification_produces_no_output_line(self) -> None:
        out = self._run([
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}),
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
            json.dumps({"jsonrpc": "2.0", "id": 2, "method": "ping"}),
        ])
        # Exactly two responses — the notification wrote nothing.
        assert len(out) == 2
        assert json.loads(out[0])["id"] == 1
        assert json.loads(out[1])["id"] == 2

    def test_blank_lines_are_skipped(self) -> None:
        out = self._run(["", "  ", json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}), ""])
        assert len(out) == 1

    def test_malformed_json_returns_parse_error(self) -> None:
        out = self._run(["not valid json {{{"])
        assert len(out) == 1
        resp = json.loads(out[0])
        assert resp["error"]["code"] == -32700

    def test_every_output_line_is_valid_json(self) -> None:
        out = self._run([
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
            "garbage",
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
            json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                        "params": {"name": "vectr_status", "arguments": {}}}),
        ])
        for line in out:
            json.loads(line)  # raises if not valid JSON

    def test_eof_ends_loop_cleanly(self) -> None:
        # An empty stdin (immediate EOF) must return without raising or hanging.
        out = self._run([])
        assert out == []

    def test_session_id_consistent_across_calls(self) -> None:
        """A single stdio process is one implicit session — the same session_id
        must be used for every dispatched line, not regenerated per call."""
        import io
        from integrations.mcp_server import ServiceHandle
        from integrations.mcp_server import _stdio as stdio_module

        seen_session_ids = []
        real_dispatch = stdio_module.dispatch_line

        def _spy(body, handle, session_id):
            seen_session_ids.append(session_id)
            return real_dispatch(body, handle, session_id)

        handle = ServiceHandle()
        handle.set_service(_base_mock_service())
        stdin = io.StringIO(
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}) + "\n"
            + json.dumps({"jsonrpc": "2.0", "id": 2, "method": "ping"}) + "\n"
        )
        stdout = io.StringIO()
        with patch.object(stdio_module, "dispatch_line", side_effect=_spy):
            stdio_module.run_stdio_loop(handle, stdin=stdin, stdout=stdout)
        assert len(seen_session_ids) == 2
        assert seen_session_ids[0] == seen_session_ids[1]


# ---------------------------------------------------------------------------
# configure_ide opt-out (the defect this transport surfaced: VectrService.
# __init__ unconditionally wrote HTTP-pointing IDE config files even when the
# caller has no real HTTP port)
# ---------------------------------------------------------------------------

class TestConfigureIdeOptOut:
    def test_configure_ide_false_skips_configure_all(self, tmp_path, monkeypatch) -> None:
        from tests.conftest import _DummyEmbedProvider, _RealVectrService, make_py
        make_py(tmp_path, "sample.py", "def foo(): pass\n")
        monkeypatch.setenv("VECTR_MEMORY_ONLY", "1")  # skip indexer/reranker construction cost
        with patch("agent.indexer.get_embed_provider", return_value=_DummyEmbedProvider()), \
             patch("integrations.vscode_bridge.configure_all") as mock_configure, \
             patch.dict("os.environ", {"VECTR_DB_DIR": str(tmp_path / "db")}):
            _RealVectrService(workspace_root=str(tmp_path), workspace_explicit=True, configure_ide=False)
        mock_configure.assert_not_called()

    def test_configure_ide_default_true_preserves_existing_behavior(self, tmp_path, monkeypatch) -> None:
        from tests.conftest import _DummyEmbedProvider, _RealVectrService, make_py
        make_py(tmp_path, "sample.py", "def foo(): pass\n")
        monkeypatch.setenv("VECTR_MEMORY_ONLY", "1")
        with patch("agent.indexer.get_embed_provider", return_value=_DummyEmbedProvider()), \
             patch("integrations.vscode_bridge.configure_all") as mock_configure, \
             patch.dict("os.environ", {"VECTR_DB_DIR": str(tmp_path / "db")}):
            _RealVectrService(workspace_root=str(tmp_path), workspace_explicit=True)
        mock_configure.assert_called()

    def test_do_start_propagates_no_ide_config_env(self, tmp_path) -> None:
        """`vectr start --no-ide-config` previously stopped only the CLI's own
        7-file write; the daemon subprocess's VectrService.__init__ still
        called configure_all() unconditionally, writing .cursor/mcp.json and
        .claude/settings.json anyway. VECTR_CONFIGURE_IDE=0 closes that gap."""
        import main as main_module
        from agent.instance_registry import workspace_hash
        from pathlib import Path as _Path

        ws = str(tmp_path)
        captured_env: dict = {}

        def _fake_popen(cmd, env, **kwargs):
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
        # _do_start's secure_dir(Path.home() / ".vectr") and the matching
        # .../.vectr/logs mkdir live in tmp_path, not the developer's home.
        # The call is still exercised end-to-end — a real crash in
        # secure_dir would still surface here, just at a tmp_path.
        with patch("subprocess.Popen", side_effect=_fake_popen), \
             patch("main.InstanceRegistry") as mock_registry_cls, \
             patch("main._migrate_legacy_files"), \
             patch("main._wait_for_daemon_ready", return_value=True), \
             patch("pathlib.Path.home", lambda: _Path(tmp_path)), \
             patch("builtins.open", MagicMock()):
            mock_registry_cls.return_value.register = MagicMock()
            main_module._do_start(ws, 8765, workspace_hash(ws), no_ide_config=True)

        assert captured_env.get("VECTR_CONFIGURE_IDE") == "0"

    def test_do_start_omits_configure_ide_env_by_default(self, tmp_path) -> None:
        import main as main_module
        from agent.instance_registry import workspace_hash
        from pathlib import Path as _Path

        ws = str(tmp_path)
        captured_env: dict = {}

        def _fake_popen(cmd, env, **kwargs):
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
        with patch("subprocess.Popen", side_effect=_fake_popen), \
             patch("main.InstanceRegistry") as mock_registry_cls, \
             patch("main._migrate_legacy_files"), \
             patch("main._wait_for_daemon_ready", return_value=True), \
             patch("pathlib.Path.home", lambda: _Path(tmp_path)), \
             patch("builtins.open", MagicMock()):
            mock_registry_cls.return_value.register = MagicMock()
            main_module._do_start(ws, 8765, workspace_hash(ws))

        assert "VECTR_CONFIGURE_IDE" not in captured_env


# ---------------------------------------------------------------------------
# Subprocess integration — real `vectr mcp-stdio` process, fast path only.
#
# No assertion below depends on the background VectrService construction
# (embedder + reranker load, seconds) having finished — every one holds
# whether the service is ready or still "starting up". This keeps the test
# in the default (non -m integration) suite while still exercising the real
# CLI entry point, real subprocess framing, and real stdout purity.
# ---------------------------------------------------------------------------

def _spawn_mcp_stdio(tmp_path: Path, extra_env: dict | None = None) -> subprocess.Popen:
    py = sys.executable
    env = dict(os.environ)
    env["VECTR_DB_DIR"] = str(tmp_path / "_db")
    env.pop("VECTR_MCP_ALL_TOOLS", None)
    if extra_env:
        env.update(extra_env)
    # UPG-TEST-REAL-EMBEDDER-DOWNLOAD-GUARD: this child's background
    # VectrService construction may build a real LocalEmbedProvider (the
    # in-process guard can't see a subprocess). Unless the run explicitly
    # opted into real models, pin HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE so any
    # accidental hub access fails fast and LOCALLY inside the child instead
    # of reaching huggingface.co (CI hit real HTTP 429s). Safe by
    # construction: every assertion in this file holds whether the background
    # service finished loading or errored into "Vectr failed to start" —
    # initialize/tools-list responses are independent of service health, and
    # the reranker is already disabled via the inherited empty
    # VECTR_RERANKER_MODEL.
    if env.get("VECTR_TEST_ALLOW_REAL_NETWORK") != "1":
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
    return subprocess.Popen(
        [py, str(_MAIN_PY), "mcp-stdio", str(tmp_path)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1, env=env, cwd=str(_REPO_ROOT),
    )


def _send(proc: subprocess.Popen, body: dict) -> None:
    proc.stdin.write(json.dumps(body) + "\n")
    proc.stdin.flush()


class TestMcpStdioSubprocess:
    def test_initialize_responds_fast(self, tmp_path) -> None:
        (tmp_path / "hello.py").write_text("def hello():\n    return 1\n")
        proc = _spawn_mcp_stdio(tmp_path)
        try:
            t0 = time.monotonic()
            _send(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                         "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                                    "clientInfo": {"name": "test", "version": "1.0"}}})
            line = proc.stdout.readline()
            elapsed = time.monotonic() - t0
            resp = json.loads(line)
            assert resp["result"]["protocolVersion"] == "2024-11-05"
            assert resp["result"]["serverInfo"]["name"] == "vectr"
            # Spec target: ~2-3s of spawn. Generous bound for CI/interpreter
            # startup variance while still catching a real regression (e.g.
            # someone moving VectrService construction back onto this path).
            assert elapsed < 8.0, f"initialize took {elapsed:.2f}s — should answer before indexing"
        finally:
            proc.stdin.close()
            proc.wait(timeout=15)

    def test_tools_list_tool_count_with_and_without_all_tools(self, tmp_path) -> None:
        proc = _spawn_mcp_stdio(tmp_path)
        try:
            _send(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize"})
            proc.stdout.readline()
            _send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})
            _send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
            base_count = len(json.loads(proc.stdout.readline())["result"]["tools"])
        finally:
            proc.stdin.close()
            proc.wait(timeout=15)

        proc2 = _spawn_mcp_stdio(tmp_path, extra_env={"VECTR_MCP_ALL_TOOLS": "1"})
        try:
            _send(proc2, {"jsonrpc": "2.0", "id": 1, "method": "initialize"})
            proc2.stdout.readline()
            _send(proc2, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
            all_count = len(json.loads(proc2.stdout.readline())["result"]["tools"])
        finally:
            proc2.stdin.close()
            proc2.wait(timeout=15)

        assert all_count > base_count

    def test_stdout_purity_during_startup(self, tmp_path) -> None:
        """Every line vectr writes to stdout while its background service is
        still constructing (embedder/reranker load, indexing) must be valid
        JSON — a single stray byte would corrupt the protocol stream for
        whatever spawned this process."""
        (tmp_path / "hello.py").write_text("def hello():\n    return 1\n")
        proc = _spawn_mcp_stdio(tmp_path)
        try:
            _send(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize"})
            proc.stdout.readline()
            _send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})
            _send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
            proc.stdout.readline()
            # A tools/call issued immediately after spawn lands while the
            # background VectrService is (almost certainly) still loading —
            # exactly the window where a stray print() from a third-party
            # model-loading library would corrupt the stream if it leaked
            # to stdout instead of stderr.
            _send(proc, {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                         "params": {"name": "vectr_status", "arguments": {}}})
            line = proc.stdout.readline()
            json.loads(line)  # raises if corrupted
        finally:
            proc.stdin.close()
            proc.wait(timeout=15)

    def test_clean_eof_shutdown(self, tmp_path) -> None:
        proc = _spawn_mcp_stdio(tmp_path)
        try:
            _send(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize"})
            proc.stdout.readline()
            proc.stdin.close()  # EOF
            returncode = proc.wait(timeout=15)
            assert returncode == 0
        finally:
            if proc.poll() is None:
                proc.kill()


@pytest.mark.integration
class TestMcpStdioSubprocessReady:
    """Waits for the real embedder/reranker to load and the workspace to
    finish indexing (~10-15s on a warm local model cache), then makes one
    real tools/call and asserts its actual content — not just protocol
    shape. Marked integration like the rest of the suite's real-model tests
    (see pytest.ini); run with `pytest -m integration`.
    """

    def test_status_call_returns_real_workspace_after_ready(self, tmp_path) -> None:
        (tmp_path / "hello.py").write_text("def hello():\n    return 1\n")
        proc = _spawn_mcp_stdio(tmp_path)
        try:
            _send(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize"})
            proc.stdout.readline()
            _send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})

            deadline = time.monotonic() + 120
            text = ""
            call_id = 2
            while time.monotonic() < deadline:
                _send(proc, {"jsonrpc": "2.0", "id": call_id, "method": "tools/call",
                             "params": {"name": "vectr_status", "arguments": {}}})
                resp = json.loads(proc.stdout.readline())
                text = resp["result"]["content"][0]["text"]
                if "starting up" not in text:
                    break
                call_id += 1
                time.sleep(1)
            assert "starting up" not in text, "service never became ready within 120s"
            assert str(tmp_path) in text
        finally:
            proc.stdin.close()
            proc.wait(timeout=15)
