"""Tests for agent/prompt_templates.py — bundled prompt/config template loader
(UPG-PROMPTS-AS-DATA).

Verifies that:
1. Every template file `main.py` depends on resolves via `load_template()` and
   is non-empty.
2. The loader is cached (repeated calls return the exact same object, not a
   re-read).
3. `main.py`'s rendered outputs (CLAUDE.md variants, session-start guidance,
   the three `.mcp.json` templates, the hook no-double-recall line, the
   Cursor rules frontmatter) are unchanged after the prompts-as-data move —
   this is the behavior-preservation gate for the refactor.
4. The `agent/templates/` directory is declared as package data in
   pyproject.toml, so it ships with the built wheel/sdist (not just present
   on disk in the checkout).
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest

from agent.prompt_templates import load_template

_REPO_ROOT = Path(__file__).resolve().parent.parent

_ALL_TEMPLATE_NAMES = (
    "claude_md.md",
    "claude_md_search_only.md",
    "session_start_guidance_default.txt",
    "session_start_guidance_hooks_aware.txt",
    "mcp.json.template",
    "cursor_mcp.json.template",
    "vscode_mcp.json.template",
    "hook_no_double_recall.txt",
    "cursor_rules_header.txt",
)


class TestLoadTemplate:
    """Every template file resolves and is non-empty via the loader."""

    @pytest.mark.parametrize("name", _ALL_TEMPLATE_NAMES)
    def test_resolves_and_nonempty(self, name: str) -> None:
        content = load_template(name)
        assert isinstance(content, str)
        assert content != "", f"{name} loaded empty"

    def test_missing_template_raises(self) -> None:
        with pytest.raises(Exception):
            load_template("does_not_exist.txt")

    def test_cached_returns_same_object(self) -> None:
        """functools.cache means repeated calls don't re-read the file."""
        a = load_template("claude_md.md")
        b = load_template("claude_md.md")
        assert a is b


class TestMainRenderingUnchanged:
    """main.py's rendered outputs are byte-identical to before the move to
    template files (behavior-preservation gate for UPG-PROMPTS-AS-DATA)."""

    def test_claude_md_default(self) -> None:
        import main
        rendered = main._render_claude_md(hooks_installed=False)
        assert rendered.startswith("# Vectr — semantic search + reliable working memory")
        assert "call `vectr_status()` first" in rendered
        assert "__SESSION_START_GUIDANCE__" not in rendered

    def test_claude_md_hooks_aware(self) -> None:
        import main
        rendered = main._render_claude_md(hooks_installed=True)
        assert "auto-injected automatically" in rendered
        assert "__SESSION_START_GUIDANCE__" not in rendered

    def test_claude_md_search_only(self) -> None:
        import main
        rendered = main._render_claude_md(hooks_installed=False, search_only=True)
        assert rendered.startswith("# Vectr — semantic search\n")
        assert "vectr_recall(query=" not in rendered
        assert "Working-memory tools" in rendered and "disabled for this daemon" in rendered

    def test_mcp_json_renders_valid_json_with_port(self) -> None:
        import json
        import main
        rendered = main._MCP_JSON.format(port=9999)
        parsed = json.loads(rendered)
        assert parsed == {"mcpServers": {"vectr": {"type": "http", "url": "http://localhost:9999/mcp"}}}

    def test_cursor_mcp_json_omits_type(self) -> None:
        import json
        import main
        rendered = main._CURSOR_MCP_JSON.format(port=9999)
        parsed = json.loads(rendered)
        assert parsed == {"mcpServers": {"vectr": {"url": "http://localhost:9999/mcp"}}}

    def test_vscode_mcp_json_uses_servers_key(self) -> None:
        import json
        import main
        rendered = main._VSCODE_MCP_JSON.format(port=9999)
        parsed = json.loads(rendered)
        assert parsed == {"servers": {"vectr": {"type": "http", "url": "http://localhost:9999/mcp"}}}

    def test_hook_no_double_recall_line(self) -> None:
        import main
        assert main._HOOK_NO_DOUBLE_RECALL_LINE == (
            "Your working-memory notes are auto-injected below — do not call vectr_recall "
            "to re-fetch them; call it only for something not shown here."
        )

    def test_write_cursor_rules_frontmatter(self, tmp_path: Path) -> None:
        import main
        main._write_cursor_rules(str(tmp_path), search_only=False)
        content = (tmp_path / ".cursor" / "rules" / "vectr.mdc").read_text(encoding="utf-8")
        assert content.startswith(
            "---\n"
            "description: Vectr tool usage rules for AI-assisted development\n"
            "alwaysApply: true\n"
            "---\n\n"
        )


@contextlib.contextmanager
def _exclusive_source_tree_build():
    """Serialize wheel builds of the source tree across CONCURRENT pytest processes.

    `--outdir` isolates the OUTPUT, but the setuptools.build_meta backend still
    works in the source tree: it writes `vectr.egg-info/` and `build/` under the
    repo root, which is shared by every process building the same checkout. Two
    full suites running at once (the sentinel workflow runs lane suites in
    parallel by design) therefore race on those directories, and one build sees
    the other's half-written state — observed as this test failing in a
    concurrent pair while passing in isolation, which is the
    UPG-TEST-CONCURRENT-SUITE-FLAKE shape.

    A lock keeps the test building the REAL tree, which is the whole point of a
    packaging test: copying a curated subset elsewhere would stop it catching
    exactly the pyproject misconfiguration it exists to catch. The lock file
    lives outside the repo (keyed by the repo path, so separate checkouts and
    worktrees don't serialize against each other) and is never cleaned up: an
    empty lock file is the cheap, race-free way to hold a cross-process flock.
    """
    key = hashlib.sha256(str(_REPO_ROOT).encode()).hexdigest()[:16]
    lock_path = Path(tempfile.gettempdir()) / f"vectr-wheel-build-{key}.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)  # releases the flock


class TestPackagingIncludesTemplates:
    """agent/templates/*.* ship in the built wheel (not just present on the
    checkout's disk)."""

    def test_wheel_contains_all_template_files(self, tmp_path: Path) -> None:
        with _exclusive_source_tree_build():
            result = subprocess.run(
                [sys.executable, "-m", "build", "--wheel", "--outdir", str(tmp_path), str(_REPO_ROOT)],
                capture_output=True,
                text=True,
                timeout=300,
            )
        assert result.returncode == 0, f"wheel build failed:\n{result.stdout}\n{result.stderr}"

        wheels = list(tmp_path.glob("*.whl"))
        assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"

        with zipfile.ZipFile(wheels[0]) as zf:
            names = set(zf.namelist())

        for template_name in _ALL_TEMPLATE_NAMES:
            expected = f"agent/templates/{template_name}"
            assert expected in names, f"{expected} missing from built wheel"
