"""Parity coverage for UPG-HOOK-SUBPROCESS-IMPORT-TAX.

`agent/hook_cli.py`'s stdlib-only `run_hook()` is a second implementation of
`main.py`'s `cmd_hook` — deliberately, so the `vectr` binary's real hook
subprocess invocation never pays `main.py`'s full import cost (see that
module's docstring for why a shared import would defeat the point). A
second implementation that silently drifts from the one it mirrors is worse
than no fast path at all, so this file drives BOTH implementations against
a real local HTTP server with the same fixture stdin and asserts their
captured stdout is byte-for-byte identical, across every hook event and the
representative edge cases (empty prompt, no file path, no daemon
registered, session_id present/absent, PreCompact triggers).
"""
from __future__ import annotations

import argparse
import io
import json
import socketserver
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import main as m
from agent import hook_cli
from agent.instance_registry import InstanceRegistry, workspace_hash
from tests.conftest import allow_loopback_port


class _FastBindHTTPServer(ThreadingHTTPServer):
    """`HTTPServer.server_bind()` calls `socket.getfqdn(host)`, a reverse-DNS
    lookup that can hang for many seconds on hosts with slow or
    VPN-shadowed DNS — observed locally binding even 127.0.0.1. The test
    only needs a bound loopback socket, not a resolved hostname, so bind via
    `TCPServer`'s own `server_bind` and skip the FQDN lookup entirely."""

    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = host
        self.server_port = port


class _StubDaemonHandler(BaseHTTPRequestHandler):
    """Canned JSON bodies for /v1/recall, /v1/snapshot, /v1/trigger/reset,
    /v1/boundary/precompact — enough of the real daemon's contract for both
    httpx (main.cmd_hook) and the stdlib socket client (agent.hook_cli.run_hook)
    to round-trip through identically."""

    RESPONSES = {
        "/v1/recall": {"notes": "[1] lock_workspace() at resolver.rs:214"},
        "/v1/snapshot": {"ok": True},
        "/v1/trigger/reset": {"ok": True},
    }

    # (status_code, body) for GET /v1/boundary/precompact — overridden per
    # test via the server instance's own `boundary_response` attribute
    # (see `stub_daemon`'s `request.param` below) so different scenarios
    # (normal text / disabled-config empty text / search-only 503) reuse
    # the same handler class.
    DEFAULT_BOUNDARY_RESPONSE = (200, {"text": "Preserve concrete findings from this session.", "arcs_pending": 0})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)  # drain the request body
        body = json.dumps(self.RESPONSES.get(self.path, {})).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/boundary/precompact":
            status_code, body_dict = getattr(
                self.server, "boundary_response", self.DEFAULT_BOUNDARY_RESPONSE,
            )
            body = json.dumps(body_dict).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, *args):
        pass  # keep test output quiet


@pytest.fixture
def stub_daemon(request):
    """`request.param`, when set via `@pytest.mark.parametrize(...,
    indirect=True)`, is a `(status_code, body_dict)` tuple overriding
    GET /v1/boundary/precompact's response for that test only — every
    other fixture in this file leaves it unset and gets the handler's
    default non-empty-text response."""
    server = _FastBindHTTPServer(("127.0.0.1", 0), _StubDaemonHandler)
    boundary_response = getattr(request, "param", None)
    if boundary_response is not None:
        server.boundary_response = boundary_response
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    # This is the suite's one legitimate live-probe user: both hook
    # implementations dial a REAL local HTTP server with real sockets. The
    # session-wide socket guard (tests/conftest.py) blocks unregistered
    # loopback connects, so the ephemeral port this fixture owns is
    # allow-listed for exactly the fixture's lifetime.
    port = server.server_address[1]
    with allow_loopback_port(port):
        try:
            yield port
        finally:
            server.shutdown()
            thread.join(timeout=2)


def _registry_pointing_at(tmp_path, port):
    reg = InstanceRegistry(registry_path=tmp_path / "instances.json")
    reg.register(workspace_hash("/p"), "/p", port, 12345)
    return reg


# (hook_event, stdin JSON) — one representative fixture per branch of
# main.cmd_hook / agent.hook_cli.run_hook, including the edge cases each
# branch early-returns on.
FIXTURES = [
    ("session-start", '{"cwd": "/p"}'),
    ("session-start", '{"cwd": "/p", "source": "compact"}'),
    ("session-start", '{"cwd": "/p", "session_id": "abc-123"}'),
    ("user-prompt-submit", '{"cwd": "/p", "prompt": "fix the workspace lock"}'),
    ("user-prompt-submit", '{"cwd": "/p", "prompt": "   "}'),
    ("user-prompt-submit", '{"cwd": "/p", "prompt": "lock flow", "session_id": "abc-123"}'),
    ("pre-tool-use", '{"cwd": "/p", "tool_input": {"file_path": "/p/agent/symbol_graph.py"}}'),
    ("pre-tool-use", '{"cwd": "/p", "tool_input": {}}'),
    # Command-family injection (memoization-l1-capture-design §5.2, B1
    # follow-up): the branch that was DEAD in run_hook prior to this fix —
    # this is the exact fixture shape whose absence let that regression
    # ship silently (both implementations must recall via `command`, never
    # `file_path`, for a Bash tool call).
    ("pre-tool-use", '{"cwd": "/p", "tool_name": "Bash", "tool_input": {"command": "pytest -q"}}'),
    ("pre-tool-use", '{"cwd": "/p", "tool_name": "Bash", "tool_input": {"command": "  "}}'),  # blank command -> early return
    ("pre-tool-use", '{"cwd": "/p", "tool_name": "Bash", "tool_input": {}}'),  # no command key at all
    ("pre-tool-use", '{"cwd": "/p", "tool_name": "Bash", "tool_input": {"command": "pytest -q"}, "session_id": "abc-123"}'),
    # A Bash tool_input that also happens to carry a file_path key (should
    # never occur from a real harness, but proves tool_name=="Bash" is
    # checked FIRST and always wins over a coincidentally-present file_path).
    ("pre-tool-use", '{"cwd": "/p", "tool_name": "Bash", "tool_input": {"command": "pytest -q", "file_path": "/p/x.py"}}'),
    ("pre-compact", '{"cwd": "/p", "trigger": "auto", "session_id": "abc-123"}'),
    ("pre-compact", '{"cwd": "/p"}'),
    ("session-start", '{"cwd": "/nowhere"}'),  # no daemon registered
]

# Separate fixture list for post-tool-use (memoization-l1-capture-design §2):
# this branch never calls `_fetch_recall`/`_emit_hook_context` at all — its
# stdout is unconditionally empty in both implementations — so parity here
# means "spawns a detached worker with an equivalent payload", not "prints
# the same recall text". See TestPostToolUseParity below.
POST_TOOL_USE_FIXTURES = [
    '{"cwd": "/p", "session_id": "abc-123", "tool_name": "Bash", '
    '"tool_input": {"command": "git commit -m \\"fix\\"", "description": "commit"}, '
    '"tool_response": {"stdout": "ok\\n", "stderr": "", "is_error": false, "exit_code": 0}}',
    '{"cwd": "/p", "tool_name": "Edit", "tool_input": {"file_path": "/p/agent/symbol_graph.py"}, '
    '"tool_response": {}}',
    '{"cwd": "/p", "tool_name": "Read", "tool_input": {"file_path": "/p/agent/symbol_graph.py"}}',
    '{"cwd": "/nowhere", "tool_name": "Bash", "tool_input": {"command": "echo hi"}}',
    # apply_patch (adversarial-review fix B4) — Codex's edit tool, captured
    # identically to Edit/Write/MultiEdit.
    '{"cwd": "/p", "tool_name": "apply_patch", "tool_input": {"file_path": "/p/agent/symbol_graph.py"}, '
    '"tool_response": {}}',
    # PostToolUseFailure (adversarial-review fix B5, G0-verified shape) — no
    # tool_response; a top-level "error" string and "is_interrupt" instead.
    '{"cwd": "/p", "session_id": "abc-123", "tool_name": "Bash", '
    '"hook_event_name": "PostToolUseFailure", "tool_input": {"command": "false"}, '
    '"error": "Exit code 7\\nfailing-probe"}',
    '{"cwd": "/p", "tool_name": "Bash", "hook_event_name": "PostToolUseFailure", '
    '"tool_input": {"command": "sleep 100"}, "error": "Exit code -1\\ninterrupted", '
    '"is_interrupt": true}',
    # UPG-EPISODE-OUTCOME-CLASSIFICATION: explicit success-routed event name
    # (harness_success=True derivation) — both implementations must agree.
    '{"cwd": "/p", "tool_name": "Bash", "hook_event_name": "PostToolUse", '
    '"tool_input": {"command": "jq \'.\' file.json"}, '
    '"tool_response": {"stdout": "{}\\n", "stderr": ""}}',
]


class TestHookCliParity:
    @pytest.mark.parametrize("hook_event,stdin_json", FIXTURES)
    def test_slim_path_matches_canonical_cmd_hook(
        self, hook_event, stdin_json, tmp_path, stub_daemon, monkeypatch, capsys,
    ):
        reg = _registry_pointing_at(tmp_path, stub_daemon)
        monkeypatch.setattr(m, "InstanceRegistry", lambda *a, **k: reg)
        monkeypatch.setattr(hook_cli, "InstanceRegistry", lambda *a, **k: reg)

        monkeypatch.setattr("sys.stdin", io.StringIO(stdin_json))
        m.cmd_hook(argparse.Namespace(hook_event=hook_event))
        canonical_out = capsys.readouterr().out

        monkeypatch.setattr("sys.stdin", io.StringIO(stdin_json))
        hook_cli.run_hook(hook_event)
        slim_out = capsys.readouterr().out

        assert slim_out == canonical_out


class TestPostToolUseParity:
    """post-tool-use (memoization-l1-capture-design §2) never prints
    anything and never awaits the daemon write — parity here is: (1) both
    implementations produce empty stdout, always; (2) both spawn a detached
    worker with the SAME episode payload for the same input, or spawn
    nothing at all for the same inputs (no daemon / uncaptured tool_name)."""

    @pytest.mark.parametrize("stdin_json", POST_TOOL_USE_FIXTURES)
    def test_both_implementations_spawn_the_same_payload(
        self, stdin_json, tmp_path, stub_daemon, monkeypatch, capsys,
    ):
        reg = _registry_pointing_at(tmp_path, stub_daemon)
        monkeypatch.setattr(m, "InstanceRegistry", lambda *a, **k: reg)
        monkeypatch.setattr(hook_cli, "InstanceRegistry", lambda *a, **k: reg)

        canonical_calls: list[dict] = []
        slim_calls: list[dict] = []

        def _fake_spawn_canonical(port, payload):
            canonical_calls.append({"port": port, "payload": payload})

        def _fake_spawn_slim(port, payload):
            slim_calls.append({"port": port, "payload": payload})

        monkeypatch.setattr(m, "_spawn_episode_worker", _fake_spawn_canonical)
        monkeypatch.setattr(hook_cli, "_spawn_episode_worker", _fake_spawn_slim)

        monkeypatch.setattr("sys.stdin", io.StringIO(stdin_json))
        m.cmd_hook(argparse.Namespace(hook_event="post-tool-use"))
        canonical_out = capsys.readouterr().out

        monkeypatch.setattr("sys.stdin", io.StringIO(stdin_json))
        hook_cli.run_hook("post-tool-use")
        slim_out = capsys.readouterr().out

        assert canonical_out == ""
        assert slim_out == ""
        assert canonical_calls == slim_calls


class TestPreCompactBoundaryTextParity:
    """PreCompact boundary-preservation text (LANE-L3): both implementations
    fetch GET /v1/boundary/precompact and print its `text` field verbatim as
    PLAIN TEXT — never the `hookSpecificOutput` JSON envelope every other
    hook event uses. Split out from the shared FIXTURES table above because
    each scenario needs the stub daemon's own boundary-route response to
    vary (normal text / disabled-config empty text / search-only 503)."""

    def _run_both(self, tmp_path, stub_daemon, monkeypatch, capsys, stdin_json):
        reg = _registry_pointing_at(tmp_path, stub_daemon)
        monkeypatch.setattr(m, "InstanceRegistry", lambda *a, **k: reg)
        monkeypatch.setattr(hook_cli, "InstanceRegistry", lambda *a, **k: reg)

        monkeypatch.setattr("sys.stdin", io.StringIO(stdin_json))
        m.cmd_hook(argparse.Namespace(hook_event="pre-compact"))
        canonical_out = capsys.readouterr().out

        monkeypatch.setattr("sys.stdin", io.StringIO(stdin_json))
        hook_cli.run_hook("pre-compact")
        slim_out = capsys.readouterr().out
        return canonical_out, slim_out

    @pytest.mark.parametrize("stub_daemon", [
        (200, {"text": "Preserve concrete findings from this session.", "arcs_pending": 2}),
    ], indirect=True)
    def test_prints_plain_text_never_json_envelope(self, tmp_path, stub_daemon, monkeypatch, capsys):
        canonical_out, slim_out = self._run_both(
            tmp_path, stub_daemon, monkeypatch, capsys,
            '{"cwd": "/p", "trigger": "auto", "session_id": "abc-123"}',
        )
        assert canonical_out == slim_out
        assert canonical_out.strip() == "Preserve concrete findings from this session."
        assert "hookSpecificOutput" not in canonical_out
        with pytest.raises(json.JSONDecodeError):
            json.loads(canonical_out)

    @pytest.mark.parametrize("stub_daemon", [
        (200, {"text": "", "arcs_pending": 0}),
    ], indirect=True)
    def test_prints_nothing_when_text_is_empty(self, tmp_path, stub_daemon, monkeypatch, capsys):
        canonical_out, slim_out = self._run_both(
            tmp_path, stub_daemon, monkeypatch, capsys,
            '{"cwd": "/p", "trigger": "auto", "session_id": "abc-123"}',
        )
        assert canonical_out == slim_out == ""

    @pytest.mark.parametrize("stub_daemon", [
        (503, {"detail": {"error": "search_only_mode"}}),
    ], indirect=True)
    def test_prints_nothing_when_route_returns_503(self, tmp_path, stub_daemon, monkeypatch, capsys):
        canonical_out, slim_out = self._run_both(
            tmp_path, stub_daemon, monkeypatch, capsys,
            '{"cwd": "/p", "trigger": "auto", "session_id": "abc-123"}',
        )
        assert canonical_out == slim_out == ""

    @pytest.mark.parametrize("stub_daemon", [
        (200, "not-a-json-object-with-text"),  # malformed shape: a bare string body
    ], indirect=True)
    def test_prints_nothing_on_malformed_body(self, tmp_path, stub_daemon, monkeypatch, capsys):
        canonical_out, slim_out = self._run_both(
            tmp_path, stub_daemon, monkeypatch, capsys,
            '{"cwd": "/p", "trigger": "auto", "session_id": "abc-123"}',
        )
        assert canonical_out == slim_out == ""

    def test_prints_nothing_when_daemon_unreachable(self, tmp_path, monkeypatch, capsys):
        # No matching registry entry at all -> both implementations return
        # before ever attempting the boundary-text GET.
        reg = InstanceRegistry(registry_path=tmp_path / "instances.json")
        monkeypatch.setattr(m, "InstanceRegistry", lambda *a, **k: reg)
        monkeypatch.setattr(hook_cli, "InstanceRegistry", lambda *a, **k: reg)
        stdin_json = '{"cwd": "/nowhere", "trigger": "auto", "session_id": "abc-123"}'

        monkeypatch.setattr("sys.stdin", io.StringIO(stdin_json))
        m.cmd_hook(argparse.Namespace(hook_event="pre-compact"))
        canonical_out = capsys.readouterr().out

        monkeypatch.setattr("sys.stdin", io.StringIO(stdin_json))
        hook_cli.run_hook("pre-compact")
        slim_out = capsys.readouterr().out

        assert canonical_out == slim_out == ""
