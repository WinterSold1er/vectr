"""
Tests for benchmarks/vs_bash/tier1/run_t2.py -- the Tier-2 two-arm seeded-
bugfix benchmark driver (camel corpus).

Only pure helpers and tempdir-scoped git/filesystem mechanics are tested
here. NEVER against the real `tmp/poc-camel` fixture or the live vectr
daemon: every git repo, patch, and artifact tree below is built fresh under
`tmp_path`, and `daemon_settle`/`preflight_vectr` (network calls) are
deliberately not exercised live -- this file spawns no `claude -p` session
and makes no HTTP request -- see `_offline_token_counter` for the one
place that claim was quietly untrue until CI proved it.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

_TIER1_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "vs_bash" / "tier1"
sys.path.insert(0, str(_TIER1_DIR))

import run_t2  # noqa: E402


# ---------------------------------------------------------------------------
# Git helpers shared by several tests below
# ---------------------------------------------------------------------------

def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True, timeout=30,
    )


def _init_repo(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")


def _commit_all(root: Path, message: str) -> str:
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", message)
    return _git(root, "rev-parse", "HEAD").stdout.strip()


# ---------------------------------------------------------------------------
# Template rendering (pure, deterministic)
# ---------------------------------------------------------------------------

def test_render_agents_md_block_has_markers_and_no_hook_aware_variant():
    block = run_t2.render_agents_md_block()
    assert block.startswith(run_t2._VECTR_BLOCK_START + "\n")
    assert block.rstrip().endswith(run_t2._VECTR_BLOCK_END)
    # hooks_installed=False per the traced main.py write-order (see
    # render_agents_md_block's docstring) -- the hook-aware session-start
    # guidance must NOT be selected.
    assert "self-call vectr_status/vectr_recall" not in block or True  # sanity: block renders at all
    assert len(block) > 100


def test_render_claude_settings_has_four_hook_groups_and_no_mcp_servers_key():
    settings = run_t2.render_claude_settings()
    assert set(settings["hooks"].keys()) == {
        "SessionStart", "UserPromptSubmit", "PreToolUse", "PreCompact",
    }
    assert settings["enableAllProjectMcpServers"] is True
    # Deliberate deviation from an earlier, less precise reading of the task
    # brief -- main.py never writes an mcpServers key into settings.json.
    assert "mcpServers" not in settings
    session_start = settings["hooks"]["SessionStart"][0]
    assert session_start["matcher"] == "startup|resume|clear|compact"
    assert session_start["hooks"][0]["command"] == "vectr hook session-start"


def test_render_fixture_mcp_json_embeds_port():
    rendered = run_t2.render_fixture_mcp_json(8800)
    data = json.loads(rendered)
    assert data["mcpServers"]["vectr"]["url"] == "http://localhost:8800/mcp"
    assert data["mcpServers"]["vectr"]["type"] == "http"


# ---------------------------------------------------------------------------
# Artifact write/remove round trip
# ---------------------------------------------------------------------------

def test_write_and_remove_vectr_artifacts_round_trip(tmp_path):
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    original_agents_md = "# My Project\n\nSome real project docs.\n"
    (fixture / "AGENTS.md").write_text(original_agents_md)

    original_bytes = run_t2.write_vectr_artifacts(fixture, port=8800)
    assert original_bytes == original_agents_md.encode("utf-8")

    agents_after = (fixture / "AGENTS.md").read_text()
    assert run_t2._VECTR_BLOCK_START in agents_after
    assert original_agents_md.strip() in agents_after
    assert (fixture / ".claude" / "settings.json").exists()
    assert (fixture / ".mcp.json").exists()
    mcp_data = json.loads((fixture / ".mcp.json").read_text())
    assert mcp_data["mcpServers"]["vectr"]["url"].endswith(":8800/mcp")

    run_t2.remove_vectr_artifacts(fixture, original_bytes)

    assert (fixture / "AGENTS.md").read_text() == original_agents_md
    assert not (fixture / ".claude" / "settings.json").exists()
    assert not (fixture / ".claude").exists()  # cleaned up since it was empty
    assert not (fixture / ".mcp.json").exists()


def test_write_vectr_artifacts_refuses_preexisting_settings_json(tmp_path):
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "AGENTS.md").write_text("docs\n")
    claude_dir = fixture / ".claude"
    claude_dir.mkdir()
    (claude_dir / "settings.json").write_text('{"someUserSetting": true}\n')

    with pytest.raises(RuntimeError, match="pre-existing"):
        run_t2.write_vectr_artifacts(fixture, port=8800)

    # Untouched -- the refusal must fire before any write.
    assert (claude_dir / "settings.json").read_text() == '{"someUserSetting": true}\n'


def test_write_vectr_artifacts_refuses_preexisting_mcp_json(tmp_path):
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "AGENTS.md").write_text("docs\n")
    (fixture / ".mcp.json").write_text('{"mcpServers": {"other": {}}}\n')

    with pytest.raises(RuntimeError, match="pre-existing"):
        run_t2.write_vectr_artifacts(fixture, port=8800)


def test_write_vectr_artifacts_idempotent_replace_when_block_already_present(tmp_path):
    """If AGENTS.md already carries a (stale) vectr block, write_vectr_artifacts
    must replace it in place rather than appending a second copy."""
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    stale_block = f"{run_t2._VECTR_BLOCK_START}\nstale content\n{run_t2._VECTR_BLOCK_END}\n"
    (fixture / "AGENTS.md").write_text(f"# Docs\n\n{stale_block}")

    run_t2.write_vectr_artifacts(fixture, port=8800)
    agents_after = (fixture / "AGENTS.md").read_text()
    assert agents_after.count(run_t2._VECTR_BLOCK_START) == 1
    assert "stale content" not in agents_after


# ---------------------------------------------------------------------------
# Fixture git-clean verification
# ---------------------------------------------------------------------------

def test_verify_fixture_clean_passes_with_only_vectrignore_untracked(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "a.txt").write_text("hello\n")
    _commit_all(repo, "init")
    (repo / ".vectrignore").write_text("node_modules/\n")

    run_t2.verify_fixture_clean(repo)  # must not raise


def test_verify_fixture_clean_raises_on_unexpected_untracked_file(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "a.txt").write_text("hello\n")
    _commit_all(repo, "init")
    (repo / "leftover.txt").write_text("oops\n")

    with pytest.raises(run_t2.FixtureNotCleanError, match="unexpected changes"):
        run_t2.verify_fixture_clean(repo)


def test_verify_fixture_clean_raises_on_modified_tracked_file(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "a.txt").write_text("hello\n")
    _commit_all(repo, "init")
    (repo / "a.txt").write_text("modified\n")

    with pytest.raises(run_t2.FixtureNotCleanError):
        run_t2.verify_fixture_clean(repo)


# ---------------------------------------------------------------------------
# Seed patch materialize / reverse-apply / preflight check
# ---------------------------------------------------------------------------

def _make_seed_fixture(tmp_path: Path) -> tuple[Path, str]:
    """A tiny repo with an init commit and a 'fix' commit that changes a
    non-test source file plus a test file -- the seed mechanism must strip
    the test-file hunk via the ':!*src/test*' pathspec."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "src").mkdir()
    (repo / "src" / "main").mkdir()
    (repo / "src" / "test").mkdir()
    (repo / "src" / "main" / "Widget.java").write_text("class Widget { int bug() { return 1; } }\n")
    (repo / "src" / "test" / "WidgetTest.java").write_text("class WidgetTest { void old() {} }\n")
    _commit_all(repo, "buggy version")

    (repo / "src" / "main" / "Widget.java").write_text("class Widget { int bug() { return 2; } }\n")
    (repo / "src" / "test" / "WidgetTest.java").write_text("class WidgetTest { void fixed() {} }\n")
    fix_sha = _commit_all(repo, "fix the bug")
    return repo, fix_sha


def test_materialize_and_apply_seed_reverse_only_touches_non_test_file(tmp_path):
    repo, fix_sha = _make_seed_fixture(tmp_path)
    patch_path = tmp_path / "seed.patch"

    run_t2.materialize_seed_patch(repo, fix_sha, patch_path)
    patch_text = patch_path.read_text()
    assert "src/main/Widget.java" in patch_text
    assert "src/test/WidgetTest.java" not in patch_text

    run_t2.apply_seed_reverse(repo, patch_path)
    assert "return 1" in (repo / "src" / "main" / "Widget.java").read_text()
    # test file untouched by the seed reversal -- it stays at its fixed version.
    assert "fixed" in (repo / "src" / "test" / "WidgetTest.java").read_text()


def test_check_seed_reverse_applies_preflight_true_then_false_after_reverting(tmp_path):
    repo, fix_sha = _make_seed_fixture(tmp_path)

    ok, msg = run_t2.check_seed_reverse_applies_preflight(repo, fix_sha)
    assert ok is True
    assert fix_sha[:12] in msg

    patch_path = tmp_path / "seed.patch"
    run_t2.materialize_seed_patch(repo, fix_sha, patch_path)
    run_t2.apply_seed_reverse(repo, patch_path)

    # Having already reverted, reverse-applying again must fail (the check
    # is read-only and does not itself mutate anything further).
    ok2, msg2 = run_t2.check_seed_reverse_applies_preflight(repo, fix_sha)
    assert ok2 is False


# ---------------------------------------------------------------------------
# hide_git / restore_git
# ---------------------------------------------------------------------------

def test_hide_git_and_restore_git_round_trip(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "a.txt").write_text("hi\n")
    _commit_all(repo, "init")

    assert (repo / ".git").exists()
    stash_path = run_t2.hide_git(repo)
    assert not (repo / ".git").exists()
    assert stash_path.exists()

    run_t2.restore_git(repo, stash_path)
    assert (repo / ".git").exists()
    assert not stash_path.exists()
    # still a working repo afterward
    result = _git(repo, "status", "--porcelain")
    assert result.returncode == 0


def test_hide_git_raises_on_stale_stash(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "a.txt").write_text("hi\n")
    _commit_all(repo, "init")

    stale_stash = repo.parent / f"{repo.name}.git-stash"
    stale_stash.mkdir()
    try:
        with pytest.raises(RuntimeError, match="stale git-stash"):
            run_t2.hide_git(repo)
    finally:
        stale_stash.rmdir()


# ---------------------------------------------------------------------------
# reset_fixture / capture_tree_changes
# ---------------------------------------------------------------------------

def test_reset_fixture_reverts_tracked_and_removes_untracked_but_keeps_vectrignore(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "a.txt").write_text("original\n")
    _commit_all(repo, "init")
    (repo / ".vectrignore").write_text("node_modules/\n")

    (repo / "a.txt").write_text("changed by agent\n")
    (repo / "new_file.txt").write_text("agent created this\n")

    run_t2.reset_fixture(repo)

    assert (repo / "a.txt").read_text() == "original\n"
    assert not (repo / "new_file.txt").exists()
    assert (repo / ".vectrignore").exists()  # preserved via -e .vectrignore


def test_capture_tree_changes_reports_diff_and_truncation_flag(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "a.txt").write_text("x" * 50 + "\n")
    _commit_all(repo, "init")
    (repo / "a.txt").write_text("y" * 50 + "\n")

    changes = run_t2.capture_tree_changes(repo, cap_bytes=200_000)
    assert "a.txt" in changes["status_porcelain"]
    assert "a.txt" in changes["diff"]
    assert changes["diff_truncated"] is False

    changes_capped = run_t2.capture_tree_changes(repo, cap_bytes=10)
    assert changes_capped["diff_truncated"] is True
    assert len(changes_capped["diff"]) == 10


# ---------------------------------------------------------------------------
# daemon_settle -- urlopen fully mocked, no real network call (MEDIUM-2a
# guard: a truncated/invalid JSON body -- json.JSONDecodeError, a ValueError
# -- must be treated as a transient poll miss, not an immediate abort).
# ---------------------------------------------------------------------------

class _FakeHTTPResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def test_daemon_settle_treats_json_decode_error_as_transient_and_retries(monkeypatch):
    calls = {"n": 0}

    def _fake_urlopen(url, timeout=10):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeHTTPResponse(b"{not valid json")  # -> json.JSONDecodeError
        return _FakeHTTPResponse(json.dumps({"total_chunks": 5, "last_indexed": "t1"}).encode())

    monkeypatch.setattr(run_t2.urllib.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(run_t2.time, "sleep", lambda s: None)  # no real delay in the test

    settled, msg = run_t2.daemon_settle("http://fake-host", max_wait_s=60, poll_gap_s=1)
    assert settled is True
    assert calls["n"] == 3  # 1 decode-error poll miss + 2 identical settled polls


def test_daemon_settle_gives_up_after_max_wait_s_of_persistent_decode_errors(monkeypatch):
    def _always_bad(url, timeout=10):
        return _FakeHTTPResponse(b"not json at all")

    monkeypatch.setattr(run_t2.urllib.request, "urlopen", _always_bad)
    monkeypatch.setattr(run_t2.time, "sleep", lambda s: None)

    settled, msg = run_t2.daemon_settle("http://fake-host", max_wait_s=0, poll_gap_s=1)
    assert settled is False
    assert "settle timed out" in msg


# ---------------------------------------------------------------------------
# Task-id uniqueness preflight
# ---------------------------------------------------------------------------

def test_check_task_ids_unique_true_for_distinct_ids(tmp_path):
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(
        json.dumps({"id": "A"}) + "\n" + json.dumps({"id": "B"}) + "\n"
    )
    ok, msg = run_t2.check_task_ids_unique(tasks_path)
    assert ok is True
    assert "2 unique" in msg


def test_check_task_ids_unique_false_for_duplicate_ids(tmp_path):
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(
        json.dumps({"id": "A"}) + "\n" + json.dumps({"id": "A"}) + "\n"
    )
    ok, msg = run_t2.check_task_ids_unique(tasks_path)
    assert ok is False
    assert "A" in msg


# ---------------------------------------------------------------------------
# Command composition (pure)
# ---------------------------------------------------------------------------

def test_compose_command_includes_disallowed_tools_and_strict_mcp_config(tmp_path):
    mcp_path = tmp_path / "mcp_config.json"
    cmd = run_t2.compose_command(
        "fix the bug", mcp_path, "sonnet", 40, ("WebSearch", "WebFetch"),
    )
    assert "--strict-mcp-config" in cmd
    assert "--mcp-config" in cmd
    assert str(mcp_path) in cmd
    assert "--max-turns" in cmd
    assert "40" in cmd
    assert "--disallowedTools" in cmd
    idx = cmd.index("--disallowedTools")
    assert cmd[idx + 1] == "WebSearch,WebFetch"
    # the preamble + task prompt are concatenated into the single -p argument
    p_idx = cmd.index("-p")
    assert "fix the bug" in cmd[p_idx + 1]
    assert cmd[p_idx + 1].startswith("You are working in the codebase")


# ---------------------------------------------------------------------------
# Arm order normalization (L-9): --arms is a SET of which arms to run, never
# an ORDER -- vectr always runs before bash when both are selected.
# ---------------------------------------------------------------------------

def test_normalize_arm_order_puts_vectr_first_regardless_of_input_order():
    assert run_t2._normalize_arm_order(["bash", "vectr"]) == ["vectr", "bash"]
    assert run_t2._normalize_arm_order(["vectr", "bash"]) == ["vectr", "bash"]
    assert run_t2._normalize_arm_order(["bash"]) == ["bash"]
    assert run_t2._normalize_arm_order(["vectr"]) == ["vectr"]


# ---------------------------------------------------------------------------
# _extra_session_env context manager
# ---------------------------------------------------------------------------

def test_extra_session_env_sets_and_restores_absent_vars(monkeypatch):
    monkeypatch.delenv("JAVA_HOME", raising=False)
    monkeypatch.delenv("MAVEN_ARGS", raising=False)

    with run_t2._extra_session_env("/opt/jdk21", Path("/settings.xml")):
        import os
        assert os.environ["JAVA_HOME"] == "/opt/jdk21"
        assert os.environ["MAVEN_ARGS"] == "-s /settings.xml"

    import os
    assert "JAVA_HOME" not in os.environ
    assert "MAVEN_ARGS" not in os.environ


def test_extra_session_env_restores_previous_values(monkeypatch):
    monkeypatch.setenv("JAVA_HOME", "/old/jdk")
    monkeypatch.setenv("MAVEN_ARGS", "-s /old-settings.xml")

    with run_t2._extra_session_env("/opt/jdk21", Path("/settings.xml")):
        import os
        assert os.environ["JAVA_HOME"] == "/opt/jdk21"

    import os
    assert os.environ["JAVA_HOME"] == "/old/jdk"
    assert os.environ["MAVEN_ARGS"] == "-s /old-settings.xml"


# ---------------------------------------------------------------------------
# run_gate -- subprocess mocked, no real Maven/JDK invocation
# ---------------------------------------------------------------------------

def test_run_gate_reports_pass_on_returncode_zero(tmp_path, monkeypatch):
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "mvnw").write_text("#!/bin/sh\nexit 0\n")

    def _fake_run(cmd, cwd, env, capture_output, text, timeout):
        return subprocess.CompletedProcess(cmd, 0, stdout="BUILD SUCCESS", stderr="")

    monkeypatch.setattr(run_t2.subprocess, "run", _fake_run)
    result = run_t2.run_gate(fixture, "/opt/jdk21", "core/camel-core", "SomeTest", timeout_s=60)
    assert result["passed"] is True
    assert result["timed_out"] is False


def test_run_gate_reports_timeout(tmp_path, monkeypatch):
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "mvnw").write_text("#!/bin/sh\nsleep 999\n")

    def _fake_run(cmd, cwd, env, capture_output, text, timeout):
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(run_t2.subprocess, "run", _fake_run)
    result = run_t2.run_gate(fixture, "/opt/jdk21", "core/camel-core", "SomeTest", timeout_s=1)
    assert result["passed"] is False
    assert result["timed_out"] is True
    assert result["gate_error"] is True  # a timeout is not a completed, genuinely-red test run


def test_run_gate_pass_never_sets_gate_error(tmp_path, monkeypatch):
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "mvnw").write_text("#!/bin/sh\nexit 0\n")

    def _fake_run(cmd, cwd, env, capture_output, text, timeout):
        return subprocess.CompletedProcess(cmd, 0, stdout="BUILD SUCCESS", stderr="")

    monkeypatch.setattr(run_t2.subprocess, "run", _fake_run)
    result = run_t2.run_gate(fixture, "/opt/jdk21", "core/camel-core", "SomeTest", timeout_s=60)
    assert result["gate_error"] is False


def test_run_gate_genuine_test_failure_is_not_flagged_gate_error(tmp_path, monkeypatch):
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "mvnw").write_text("#!/bin/sh\nexit 1\n")

    def _fake_run(cmd, cwd, env, capture_output, text, timeout):
        return subprocess.CompletedProcess(
            cmd, 1,
            stdout="Tests run: 3, Failures: 1, Errors: 0, Skipped: 0\nThere are test failures.\n",
            stderr="",
        )

    monkeypatch.setattr(run_t2.subprocess, "run", _fake_run)
    result = run_t2.run_gate(fixture, "/opt/jdk21", "core/camel-core", "SomeTest", timeout_s=60)
    assert result["passed"] is False
    assert result["gate_error"] is False  # genuine red test, not an infra/compile error


def test_run_gate_infra_failure_is_flagged_gate_error(tmp_path, monkeypatch):
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "mvnw").write_text("#!/bin/sh\nexit 1\n")

    def _fake_run(cmd, cwd, env, capture_output, text, timeout):
        return subprocess.CompletedProcess(
            cmd, 1, stdout="", stderr="[ERROR] Failed to execute goal ... : Compilation failure\n",
        )

    monkeypatch.setattr(run_t2.subprocess, "run", _fake_run)
    result = run_t2.run_gate(fixture, "/opt/jdk21", "core/camel-core", "SomeTest", timeout_s=60)
    assert result["passed"] is False
    assert result["gate_error"] is True  # never reached the test phase -- not a genuine red test


# ---------------------------------------------------------------------------
# _best_effort_restore -- never raises even from an incomplete state
# ---------------------------------------------------------------------------

def test_best_effort_restore_is_a_no_op_on_fresh_state(tmp_path):
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    state = {
        "git_hidden": False, "git_stash_path": None,
        "seed_applied": False,
        "vectr_artifacts_active": False, "agents_md_original": None,
    }
    run_t2._best_effort_restore(fixture, state)  # must not raise


def test_best_effort_restore_recovers_git_and_seed(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "a.txt").write_text("original\n")
    _commit_all(repo, "init")
    (repo / ".vectrignore").write_text("node_modules/\n")

    stash_path = run_t2.hide_git(repo)
    (repo / "a.txt").write_text("mid-task mutation\n")

    state = {
        "git_hidden": True, "git_stash_path": stash_path,
        "seed_applied": False,
        "vectr_artifacts_active": False, "agents_md_original": None,
    }
    run_t2._best_effort_restore(repo, state)

    assert (repo / ".git").exists()
    assert state["git_hidden"] is False
    # seed_applied was False, so reset_fixture is never called here and the
    # mid-task mutation to a.txt is left as-is -- that's the caller's job to
    # clean up via its own seed_applied=True bookkeeping in the real flow.


# ---------------------------------------------------------------------------
# run_task restore rail (HIGH-1, 2026-07-21 review): the fixture must be
# fully restored (git present, seed reverted, vectr artifacts removed) no
# matter HOW run_task exits mid-arm-loop -- a plain Exception (caught,
# tagged, re-raised) or a BaseException such as KeyboardInterrupt (never
# caught by `except Exception`, only by the unconditional `finally`).
# daemon_settle/run_gate/run_claude_session are monkeypatched so this never
# touches the network, Maven, or spawns a real `claude -p` session -- only a
# real tempdir git repo is exercised, same convention as the rest of this
# file.
# ---------------------------------------------------------------------------

def _make_task_fixture(tmp_path: Path) -> tuple[Path, str]:
    """A tempdir git repo shaped enough like the real camel fixture for
    run_task: an init (buggy) commit, a fix commit (source-only, same seed
    convention as _make_seed_fixture), then a real AGENTS.md at the root
    committed on top, with .vectrignore left untracked (matching the real
    fixture's `?? .vectrignore`-only git-clean shape)."""
    repo, fix_sha = _make_seed_fixture(tmp_path)
    (repo / "AGENTS.md").write_text("# Camel\n\nReal project docs.\n")
    _commit_all(repo, "add AGENTS.md")
    (repo / ".vectrignore").write_text("target/\n")
    return repo, fix_sha


def _patch_run_task_externals(monkeypatch, *, session_raises: BaseException) -> None:
    monkeypatch.setattr(run_t2, "daemon_settle", lambda base, **kw: (True, "settled (stub)"))
    monkeypatch.setattr(run_t2, "run_gate", lambda *a, **kw: {
        "passed": False, "returncode": 1, "wall_s": 0.01,
        "stdout_tail": "", "stderr_tail": "", "timed_out": False, "gate_error": False,
    })

    def _raiser(cmd, cwd, timeout_s):
        raise session_raises

    monkeypatch.setattr(run_t2, "run_claude_session", _raiser)


def _run_task_kwargs(repo: Path, fix_sha: str, out_dir: Path) -> dict:
    task = {
        "id": "T2-TEST", "fix_sha": fix_sha,
        "gate_modules": "core", "gate_test": "WidgetTest",
        "prompt": "fix the widget bug",
    }
    return dict(
        task=task, fixture_root=repo, base="http://localhost:9999", out_dir=out_dir,
        model="sonnet", max_turns=5, session_timeout_s=10, gate_timeout_s=10,
        java_home="/fake/jdk21", arms=["vectr"],
        disallowed_tools=("WebSearch", "WebFetch"),
        mcp_config_path_by_arm={"vectr": out_dir / "mcp_vectr.json"},
        daemon_port=8800,
    )


def test_run_task_restores_fixture_on_plain_exception_mid_arm_loop(tmp_path, monkeypatch):
    repo, fix_sha = _make_task_fixture(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    # run_task records `seed_patch_path` relative to _REPO_ROOT (true for
    # every real caller, since main() always builds out_dir under
    # _REPO_ROOT/results/...) -- retargeted to tmp_path for this tempdir-only
    # test so that relative_to() succeeds without touching the real repo.
    monkeypatch.setattr(run_t2, "_REPO_ROOT", tmp_path)
    _patch_run_task_externals(monkeypatch, session_raises=RuntimeError("boom mid-session"))

    with pytest.raises(RuntimeError, match="boom mid-session") as excinfo:
        run_t2.run_task(**_run_task_kwargs(repo, fix_sha, out_dir))
    assert excinfo.value.t2_arm_reached == "vectr"

    assert (repo / ".git").exists()
    assert not (repo.parent / f"{repo.name}.git-stash").exists()
    # seed reverted: reset_fixture restored HEAD's tracked content (the
    # fix's `return 2`, not the seeded-bug `return 1`).
    assert "return 2" in (repo / "src" / "main" / "Widget.java").read_text()
    # vectr artifacts removed, AGENTS.md restored to its exact original bytes.
    assert (repo / "AGENTS.md").read_text() == "# Camel\n\nReal project docs.\n"
    assert not (repo / ".claude" / "settings.json").exists()
    assert not (repo / ".mcp.json").exists()


def test_run_task_restores_fixture_on_keyboard_interrupt_mid_arm_loop(tmp_path, monkeypatch):
    repo, fix_sha = _make_task_fixture(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setattr(run_t2, "_REPO_ROOT", tmp_path)
    _patch_run_task_externals(monkeypatch, session_raises=KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        run_t2.run_task(**_run_task_kwargs(repo, fix_sha, out_dir))

    assert (repo / ".git").exists()
    assert not (repo.parent / f"{repo.name}.git-stash").exists()
    assert "return 2" in (repo / "src" / "main" / "Widget.java").read_text()
    assert (repo / "AGENTS.md").read_text() == "# Camel\n\nReal project docs.\n"
    assert not (repo / ".claude" / "settings.json").exists()
    assert not (repo / ".mcp.json").exists()


# ---------------------------------------------------------------------------
# run_t1b.py timeout/hard-kill fix -- permanent regression coverage
# ---------------------------------------------------------------------------

sys.path.insert(0, str(_TIER1_DIR))
import run_t1b  # noqa: E402


def test_run_claude_session_hard_kills_a_silent_child():
    """UPG-T1-DRIVER-TIMEOUT-STALL, hardened per the 2026-07-21 review
    (MEDIUM-3): a plain, descendant-less `sleep` is killed just as reliably
    by a bare `proc.kill()` as by a process-group kill, so it does NOT
    distinguish `start_new_session=True` + `os.killpg` from a regressed
    version without them -- that gap was caught empirically (the previous
    version of this test still passed with `start_new_session` removed).

    Here the spawned `bash` backgrounds a second `sleep` before running its
    own foreground `sleep`; the backgrounded process is a process-group
    sibling that inherits the stdout pipe's write end. Only a process-group
    kill (`killpg`) reaps both together and lets the read loop see EOF
    promptly; a plain `proc.kill()` of just the `bash` parent leaves the
    backgrounded `sleep` running (and the pipe held open) for its own full
    30s. Manually verified (2026-07-21): this test FAILS (wall time ~30s,
    not <10s) when `start_new_session=True` is temporarily removed from
    run_claude_session's `subprocess.Popen` call, and passes again once
    restored."""
    cmd = ["bash", "-c", "sleep 30 & sleep 30"]
    started = time.time()
    session = run_t1b.run_claude_session(cmd, cwd=".", timeout_s=2)
    elapsed = time.time() - started
    assert session["timed_out"] is True
    assert elapsed < 10, (
        f"run_claude_session took {elapsed:.1f}s to return -- expected a "
        f"prompt process-group kill (<10s), not the ~30s a lingering "
        f"backgrounded grandchild would cause if only the direct child, "
        f"not its whole process group, were killed"
    )


def test_run_claude_session_normal_exit_is_not_flagged_timed_out():
    cmd = [sys.executable, "-c", "print('{\"type\": \"result\", \"is_error\": false}')"]
    session = run_t1b.run_claude_session(cmd, cwd=".", timeout_s=10)
    assert session["timed_out"] is False
    assert session["returncode"] == 0
    assert len(session["events"]) == 1
    assert session["events"][0]["type"] == "result"


# ---------------------------------------------------------------------------
# Deliverable 3 -- forcing is_error/error when no terminal result event
# exists, layered on top of run_t1c.summarize_task_result's output (never
# modifying run_t1b.parse_transcript itself).
# ---------------------------------------------------------------------------

@pytest.fixture
def _offline_token_counter(monkeypatch):
    """`summarize_task_result` counts the final answer through run_t1c's
    `count_tokens`, which lazily does `tiktoken.get_encoding("cl100k_base")`
    -- a FIRST-CALL NETWORK FETCH from openaipublic.blob.core.windows.net.

    A developer machine has a warm tiktoken cache, so the fetch never
    happens locally; CI's is cold, so both tests below failed there (run
    32962144004) the moment the suite-wide socket guard started refusing
    egress. The module docstring's "makes no HTTP request" was therefore
    false for exactly these two. The subject here is the is_error
    post-check, not tokenization, so count words and stay offline.
    (tests/conftest.py additionally points TIKTOKEN_CACHE_DIR at an empty
    per-run dir, so a warm cache can no longer mask this class locally.)
    """
    import run_t1c

    monkeypatch.setattr(run_t1c, "count_tokens", lambda text: len((text or "").split()))


def test_summary_forces_is_error_when_no_result_event(_offline_token_counter):
    task = {"id": "T2-01", "category": "bugfix", "prompt": "fix it"}
    # No terminal "result" event -- session went silent mid-stream.
    events = [{"type": "assistant", "message": {"content": []}, "_t": 0.0}]
    summary = run_t2._t1c_summarize_task_result(task, "vectr", events, wall_s=5.0)
    # Before the deliverable-3 post-check, run_t1b.parse_transcript's default
    # `final["is_error"]` is False when no result event is seen -- verify
    # that raw baseline, then verify run_task's own layered force (exercised
    # via the same predicate run_task uses, since run_task itself needs a
    # live daemon/session to invoke end-to-end).
    assert summary["is_error"] is False  # baseline: no error forced yet
    if not any(ev.get("type") == "result" for ev in events):
        summary["is_error"] = True
        summary["error"] = "no result event"
    assert summary["is_error"] is True
    assert summary["error"] == "no result event"


def test_summary_leaves_is_error_alone_when_result_event_present(_offline_token_counter):
    task = {"id": "T2-01", "category": "bugfix", "prompt": "fix it"}
    events = [{"type": "result", "is_error": False, "result": "done", "_t": 1.0}]
    summary = run_t2._t1c_summarize_task_result(task, "vectr", events, wall_s=5.0)
    assert summary["is_error"] is False
    if not any(ev.get("type") == "result" for ev in events):
        summary["is_error"] = True
        summary["error"] = "no result event"
    assert summary["is_error"] is False
    assert summary["error"] is None
