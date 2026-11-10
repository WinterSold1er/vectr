"""CLI entry point: vectr start / restart / stop / index / search / status / init."""
from __future__ import annotations

import sys

# UPG-HOOK-SUBPROCESS-IMPORT-TAX — `vectr hook <event>` is the highest-
# frequency invocation of this CLI: the editor harness spawns a fresh
# subprocess for it on every SessionStart/UserPromptSubmit/PreToolUse/
# PostToolUse/PreCompact, so this module's own import cost (dotenv, the full
# agent.config surface, argparse's subcommand tree) is pure per-turn latency
# paid before a single byte of stdin is even read. Short-circuit to the
# stdlib-only implementation in agent/hook_cli.py BEFORE any of those heavy
# imports run, for exactly the well-formed `hook <event>` invocation the
# installed console-script entry point (`vectr = "main:main"`) actually
# produces. Any other shape of argv (missing/extra args, --help, an
# unrecognized event) falls through unchanged to the normal argparse-driven
# path below, which already handles it — this is a fast path alongside the
# existing one, not a replacement for its validation. `agent.hook_cli`
# reimplements the same branches (see its module docstring for why this is
# a second implementation, parity-tested against this file's `cmd_hook`,
# rather than a shared import).
#
# `post-tool-use` (episode capture) belongs in this tuple for the same
# reason as the other four: it fires on the harness's
# synchronous turn loop (one PostToolUse per Bash/Edit/Write/MultiEdit call —
# the highest-frequency event of all), and its own foreground budget (≤50ms
# p95, spec §7 gate G2) cannot be met while paying this module's argparse
# import cost. Unlike those four, its own handler does no HTTP I/O inline —
# it only spawns a detached worker — but the import tax alone already
# dominates the budget, so the fast path still matters here.
#
# `post-commit` (UPG-COMMIT-MEMORY-HOOK) is deliberately NOT in this tuple —
# it is fired by `git`, not the editor harness's synchronous turn loop, and
# its own shell hook wrapper backgrounds the whole `vectr hook post-commit`
# invocation (see agent/templates/git_post_commit_hook.sh.template), so
# `git commit` returns before this module's import cost is even paid. The
# fast path exists to shave per-turn latency off events that block the
# harness; a backgrounded, once-per-commit event has nothing to shave.
_HOOK_EVENTS = (
    "session-start", "user-prompt-submit", "pre-tool-use", "post-tool-use", "pre-compact",
)
if len(sys.argv) == 3 and sys.argv[1] == "hook" and sys.argv[2] in _HOOK_EVENTS:
    from agent.hook_cli import run_hook
    run_hook(sys.argv[2])
    sys.exit(0)

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import threading
import time
import tomllib
from pathlib import Path

from dotenv import load_dotenv

from agent.config import (
    CLI_START_READY_POLL_INTERVAL_S,
    CLI_START_READY_POLL_TIMEOUT_S,
    CLI_START_READY_PROBE_TIMEOUT_S,
    CLI_VERSION_SKEW_PROBE_TIMEOUT_S,
    DEFAULT_PORT,
    HOOKS_POST_COMMIT_TIMEOUT_S,
    vectr_cache_root,
)
from agent.instance_registry import (
    InstanceRegistry,
    PortBusyError,
    _is_pid_alive,
    workspace_hash,
)
from agent.prompt_templates import load_template
from agent.version_stamp import compute_version_stamp

load_dotenv()

# Legacy single-instance files — removed on first registry write, kept here only
# so migration can clean them up.
_LEGACY_PID_FILE = Path.home() / ".vectr" / "vectr.pid"
_LEGACY_PORT_FILE = Path.home() / ".vectr" / "vectr.port"

# The daemon's operating-mode vocabulary — protocol labels, not tunables:
# these are the exact strings `VectrService.status()` publishes in its "mode"
# field (app/service.py), so the CLI's mode lines, the instance-registry
# record (UPG-RESTART-DROPS-MODE) and the staleness banner's remediation
# command all speak the same language as the daemon they describe. Pinned
# against the service's real output by tests/test_memory_only_mode.py and
# tests/test_search_only_mode.py so the two sides cannot drift apart.
MODE_FULL = "full"
MODE_MEMORY_ONLY = "memory-only"
MODE_SEARCH_ONLY = "search-only"

# The CLI flag that re-selects each mode explicitly on `restart`, keyed by the
# mode label above — used both to override an inherited mode and to print a
# mode-complete restart command in the staleness banner.
MODE_RESTART_FLAG = {
    MODE_FULL: "--full",
    MODE_MEMORY_ONLY: "--memory-only",
    MODE_SEARCH_ONLY: "--search-only",
}

# Per-turn recall hook tuning (UPG-9.5). Small N + a relevance floor keep the
# UserPromptSubmit injection tight: only notes genuinely related to the prompt,
# nothing on an off-topic turn. Override via env without re-running init.
# The relevance floor itself is config-driven (agent/config.yaml
# hooks.min_similarity) — applied server-side by `VectrService.recall`
# whenever `hook_event` is set and the request omits `min_similarity`
# (UPG-HOOK-SUBPROCESS-IMPORT-TAX), so this CLI never needs to import
# agent.config just to send that one float on its hottest path.
_HOOK_RECALL_LIMIT = 3

# Episode capture — hard cap on how much of a Bash/Edit tool's raw stdout/stderr this
# FOREGROUND process (the one the editor harness is synchronously waiting
# on, budget ≤50ms p95 per spec §7 gate G2) will ever json.dump into the
# handoff temp file. Measured before this cap existed: a 1MB tool output
# cost ~30ms p50/205ms p95 to serialize foreground, a 10MB one ~66/230ms —
# well over budget, and the size is attacker/tool-controlled (a verbose test
# run, a huge `cat`), not bounded by anything upstream. Kept as a raw
# hardcoded constant rather than an agent/config.yaml entry for the same
# reason as `_HOOK_RECALL_LIMIT` immediately below: this is the stdlib-only
# fast path (agent/hook_cli.py mirrors it) that exists specifically to avoid
# paying agent.config's import cost, so a config-sourced value here would
# defeat its own purpose. Set well above the worker's own
# EPISODES_CLIENT_TRUNCATE_CHARS default (8000) so nothing the daemon-side
# digest step would have kept is lost here first — this is a coarse
# foreground safety cap, not the real truncation (that stays config-driven,
# server/worker-side). Tail-kept (last N chars, not first): failure markers
# characteristically appear near the end of long output (see
# agent/outcome.py), so truncating the head and keeping the tail preserves
# the signal that actually determines outcome.
_EPISODE_FOREGROUND_TRUNCATE_CHARS = 65536

# UPG-11.5 — hook-injected notes announce themselves so the model doesn't also
# self-call vectr_recall for the same purpose. Without this, SessionStart/
# UserPromptSubmit injection and the model's own recall both fire and pay for
# the same memory twice (found in the eval v2 N=1 audit: arm C self-recalled
# on top of its hook injection). Prepended only to the events that duplicate
# what CLAUDE.md's session-start guidance already tells the model to fetch.
_HOOK_NO_DOUBLE_RECALL_LINE = load_template("hook_no_double_recall.txt")
_HOOK_EVENTS_ANNOUNCE_INJECTION = ("SessionStart", "UserPromptSubmit")

_CLAUDE_MD = load_template("claude_md.md")

# UPG-SEARCH-ONLY-MODE — this workspace's daemon runs with working memory
# disabled (no notes DB exists). Only the search/locate/trace/map tools are
# documented; there is no working-memory section, no session-start recall
# instructions, and no mention of vectr_remember/vectr_recall.
_CLAUDE_MD_SEARCH_ONLY = load_template("claude_md_search_only.md")

# Default session-start guidance: no hooks installed, so the model must
# self-call vectr_status/vectr_recall to ever see prior notes.
_SESSION_START_GUIDANCE_DEFAULT = load_template("session_start_guidance_default.txt")

# UPG-11.5 — hook-aware variant: when Claude Code hooks are installed for this
# workspace, SessionStart/UserPromptSubmit already inject recalled notes
# automatically (see `_emit_hook_context`), so repeating "call vectr_recall at
# session start" pays for the same memory twice. Redirect vectr_recall to the
# genuinely on-demand cases instead.
_SESSION_START_GUIDANCE_HOOKS_AWARE = load_template("session_start_guidance_hooks_aware.txt")

# UPG-INSTRUCTION-VET (V3): the tool-loading blockquote describes one host's
# deferred-tool mechanics (ToolSearch, the mcp__vectr__ tool-name prefix) that
# do not exist in other AI IDEs. It is spliced in only for the CLAUDE.md
# render; every other IDE config file (AGENTS.md, .cursorrules, GEMINI.md,
# copilot-instructions.md, .cursor/rules) gets the shared body with the
# placeholder removed.
_TOOL_LOADING_GUIDANCE_CLAUDE = load_template("tool_loading_guidance_claude.txt")
_TOOL_LOADING_GUIDANCE_CLAUDE_SEARCH_ONLY = load_template("tool_loading_guidance_claude_search_only.txt")


def _splice(template: str, placeholder: str, text: str) -> str:
    """Replace `placeholder` with `text`; an empty splice also collapses the
    blank lines that framed the placeholder so no gap is left behind."""
    if text:
        return template.replace(placeholder, text.rstrip("\n"))
    return template.replace(f"\n{placeholder}\n", "")


def _render_claude_md(hooks_installed: bool, search_only: bool = False, tool_loading: bool = False) -> str:
    """Render the CLAUDE.md guidance block.

    `search_only` selects the search-only variant (UPG-SEARCH-ONLY-MODE) — no
    working-memory section, no session-start recall instructions, since this
    daemon has no notes DB. `hooks_installed` selects the session-start
    guidance matching whether hooks are installed for this workspace
    (UPG-11.5). `tool_loading` splices the deferred-tool loading blockquote —
    True only for the CLAUDE.md write, whose host actually has that mechanism
    (UPG-INSTRUCTION-VET V3)."""
    if search_only:
        loading = _TOOL_LOADING_GUIDANCE_CLAUDE_SEARCH_ONLY if tool_loading else ""
        return _splice(_CLAUDE_MD_SEARCH_ONLY, "__TOOL_LOADING_GUIDANCE__", loading)
    guidance = _SESSION_START_GUIDANCE_HOOKS_AWARE if hooks_installed else _SESSION_START_GUIDANCE_DEFAULT
    rendered = _CLAUDE_MD.replace("__SESSION_START_GUIDANCE__", guidance)
    loading = _TOOL_LOADING_GUIDANCE_CLAUDE if tool_loading else ""
    return _splice(rendered, "__TOOL_LOADING_GUIDANCE__", loading)


_MCP_JSON = load_template("mcp.json.template")

# Cursor omits the "type" key (it infers HTTP from the url scheme)
_CURSOR_MCP_JSON = load_template("cursor_mcp.json.template")

# VSCode 1.99+ / GitHub Copilot Agent Mode uses "servers" (not "mcpServers")
_VSCODE_MCP_JSON = load_template("vscode_mcp.json.template")

# Codex CLI registers MCP servers in a TOML config (project-scoped
# `<workspace>/.codex/config.toml`), one `[mcp_servers.<name>]` table per
# server (UPG-CODEX-PARITY, research/codex-parity-2026-07.md §Q1). vectr is
# already an HTTP MCP server so it maps directly onto the `url` field with no
# stdio wrapper. Unlike the JSON MCP configs above (which `_write_or_update`
# overwrites whole), a user's `.codex/config.toml` is far more likely to hold
# unrelated content (project trust, model settings, other servers), so the
# writer edits only vectr's own comment-delimited region and leaves the rest
# byte-for-byte intact — the same strip-then-append idiom the markdown vectr
# block uses. `__HEADERS__` is spliced (an `http_headers` inline table for an
# authenticated daemon, or nothing for the keyless default).
_CODEX_MCP_TOML = load_template("codex_mcp.toml.template")
_CODEX_TOML_BLOCK_START = "# vectr-start"
_CODEX_TOML_BLOCK_END = "# vectr-end"
# Matches vectr's TOML block plus any blank lines immediately before it.
_CODEX_TOML_BLOCK_RE = re.compile(
    r"\n*" + re.escape(_CODEX_TOML_BLOCK_START) + r"\b.*?" + re.escape(_CODEX_TOML_BLOCK_END) + r"\n?",
    re.DOTALL,
)

# DeepSeek Harness (dsh) integration (UPG-DSH-PARITY): dsh loads MCP servers
# as `@deepseek-ai/dsh-mcp-client` plugin rows in a profile's
# `cordis.patch.yml` — a top-level YAML array of patch entries, NOT a
# per-workspace file like the editors above. vectr registers itself by
# managing one comment-delimited `- insert:` item in the ACTIVE profile's
# patch file (default `web` — the GUI profile — overridable via
# `$VECTR_DSH_PROFILE` or `$DSH_PROFILE`). Same strip-then-append idiom as
# the Codex TOML block and the markdown vectr block: only vectr's own
# `# vectr-start`…`# vectr-end` region is touched, anything else the user
# has in the patch (their own inserts, disables) survives verbatim.
_DSH_PATCH_TEMPLATE = load_template("dsh_mcp.patch.yml.template")
_DSH_BLOCK_START = "# vectr-start"
_DSH_BLOCK_END = "# vectr-end"
# Matches vectr's dsh patch block plus any blank lines immediately before it.
_DSH_BLOCK_RE = re.compile(
    r"\n*" + re.escape(_DSH_BLOCK_START) + r"\b.*?" + re.escape(_DSH_BLOCK_END) + r"\n?",
    re.DOTALL,
)
# Matches a lone `[]` empty top-level YAML array (optionally surrounded by
# comment lines), which a fresh `cordis.patch.yml` ships as.
_DSH_EMPTY_ARRAY_RE = re.compile(r"^(\s*(?:#.*\n)*)\s*\[\s*\]\s*$", re.MULTILINE)
# Default profile: the browser GUI profile ships as `web`; `headless` is the
# one-shot runner. `$DSH_PROFILE` is honored first, then `$VECTR_DSH_PROFILE`
# (vectr-specific override), then `web`.
_DSH_DEFAULT_PROFILE = "web"

# dsh hook bridge (UPG-DSH-PARITY): dsh has its own Claude Code hook bridge
# (`@deepseek-ai/dsh-hooks-claude-code`) that runs a user's `.claude/
# settings.json` `hooks` key on dsh's interception points. vectr's hook rows
# (`vectr hook <event>`) speak the same Claude Code stdin/exit contract, so
# the bridge consumes them unchanged. This is a SEPARATE comment-delimited
# region from the MCP block above (`# vectr-hooks-start`…`# vectr-hooks-end`,
# distinct markers so each block is managed independently): it points the
# bridge at this workspace's `.claude/settings.json`. dsh's bridge does not
# support `PreCompact` or `PostToolUseFailure` (ignored at parse), so those
# two vectr hook groups simply never fire on dsh — every other event works.
_DSH_HOOKS_PATCH_TEMPLATE = load_template("dsh_hooks.patch.yml.template")
_DSH_HOOKS_BLOCK_START = "# vectr-hooks-start"
_DSH_HOOKS_BLOCK_END = "# vectr-hooks-end"
# Matches vectr's dsh hooks block plus any blank lines immediately before it.
_DSH_HOOKS_BLOCK_RE = re.compile(
    r"\n*" + re.escape(_DSH_HOOKS_BLOCK_START) + r"\b.*?" + re.escape(_DSH_HOOKS_BLOCK_END) + r"\n?",
    re.DOTALL,
)


_VECTR_BLOCK_START = "<!-- vectr-start -->"
_VECTR_BLOCK_END = "<!-- vectr-end -->"
# Matches the vectr block plus any blank lines immediately before it.
_VECTR_BLOCK_RE = re.compile(
    r"\n*<!-- vectr-start -->.*?<!-- vectr-end -->\n?",
    re.DOTALL,
)

# UPG-COMMIT-MEMORY-HOOK: the git post-commit hook's own comment-delimited
# region (a shell script, so `#`-style markers — same literal text as the
# Codex TOML markers above by coincidence of both being comment syntax, but
# declared as its own constants/regex rather than shared, matching this
# module's existing per-artifact-type convention (markdown vs. TOML each get
# their own trio above)).
_GIT_HOOK_BLOCK_START = "# vectr-start"
_GIT_HOOK_BLOCK_END = "# vectr-end"
_GIT_HOOK_BLOCK_RE = re.compile(
    r"\n*" + re.escape(_GIT_HOOK_BLOCK_START) + r"\b.*?" + re.escape(_GIT_HOOK_BLOCK_END) + r"\n?",
    re.DOTALL,
)
# Interpreters vectr's `if command -v ...; then ... fi` POSIX-shell block may
# be safely appended under — matched against the shebang's own basename
# (handling both `#!/bin/sh` and `#!/usr/bin/env bash` forms), never a
# substring, so e.g. `#!/usr/bin/env pwsh` (PowerShell) is correctly treated
# as incompatible rather than false-positiving on a stray "sh"-like name.
_POSIX_SHELL_NAMES: tuple[str, ...] = ("sh", "bash", "dash", "zsh", "ksh")
_GIT_POST_COMMIT_BLOCK = load_template("git_post_commit_hook.sh.template")

# Local `git` subprocess safety timeout — a hang guard against a stuck git
# process, mirroring `_current_git_branch`'s own literal (agent/
# working_context_store/_store.py). Not an operator-tunable value (unlike
# agent/config.yaml's hooks.post_commit_timeout_s, which governs the actual
# tunable this data feeds: the daemon HTTP call in _post_commit_note below).
_GIT_SUBPROCESS_TIMEOUT_S = 2

# IDE config files that get the vectr block appended (not created from scratch).
# CODEX.md is deliberately absent: Codex CLI reads ONLY AGENTS.md (plus global
# ~/.codex/AGENTS.md) and has never read a CODEX.md file, so appending a vectr
# block into a user's pre-existing CODEX.md was dead copy no tool consumes
# (UPG-CODEX-STALE-AGENTS-FILE). Codex adoption rides the AGENTS.md entry above.
_IDE_CONFIG_APPEND_ONLY: tuple[str, ...] = (
    "AGENTS.md",
    ".cursorrules",
    "GEMINI.md",
)


def _make_vectr_block(*, hooks_installed: bool = False, search_only: bool = False, tool_loading: bool = False) -> str:
    """`hooks_installed` selects the session-start guidance variant (UPG-11.5) —
    meaningful for CLAUDE.md (Claude Code hooks) and AGENTS.md (Codex hooks,
    UPG-CODEX-PARITY); the other IDE config files have no automatic-injection
    path and always get the default.
    `search_only` selects the no-working-memory variant (UPG-SEARCH-ONLY-MODE)
    and takes precedence over `hooks_installed`. `tool_loading` splices the
    host-specific deferred-tool loading blockquote — CLAUDE.md only
    (UPG-INSTRUCTION-VET V3)."""
    return f"{_VECTR_BLOCK_START}\n{_render_claude_md(hooks_installed, search_only=search_only, tool_loading=tool_loading).rstrip()}\n{_VECTR_BLOCK_END}\n"


def _write_ide_config_merge_safe(
    path: Path, *, create_if_missing: bool, hooks_installed: bool = False, search_only: bool = False,
    tool_loading: bool = False,
) -> None:
    """Write the vectr guidance block into an IDE config file.

    - File missing + create_if_missing=True  → create file containing just the block.
    - File missing + create_if_missing=False → no-op.
    - File exists, no vectr block            → append block after existing content.
    - File exists, vectr block present       → replace block in-place (idempotent).
    """
    block = _make_vectr_block(hooks_installed=hooks_installed, search_only=search_only, tool_loading=tool_loading)

    if not path.exists():
        if not create_if_missing:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(block, encoding="utf-8")
        print(f"  Created {path}", file=sys.stderr)
        return

    existing = path.read_text(encoding="utf-8")

    if _VECTR_BLOCK_START in existing:
        stripped = _VECTR_BLOCK_RE.sub("", existing).rstrip()
        new_content = f"{stripped}\n\n{block}" if stripped else block
        if new_content == existing:
            return
        path.write_text(new_content, encoding="utf-8")
        print(f"  Updated vectr block in {path}", file=sys.stderr)
    else:
        path.write_text(f"{existing.rstrip()}\n\n{block}", encoding="utf-8")
        print(f"  Appended vectr block to {path}", file=sys.stderr)


def _remove_vectr_block(path: Path) -> None:
    """Remove the vectr block from a file. Delete the file if it becomes empty."""
    if not path.exists():
        return
    content = path.read_text(encoding="utf-8")
    if _VECTR_BLOCK_START not in content:
        return
    stripped = _VECTR_BLOCK_RE.sub("", content).rstrip()
    if stripped:
        path.write_text(stripped + "\n", encoding="utf-8")
        print(f"  Removed vectr block from {path}", file=sys.stderr)
    else:
        path.unlink()
        print(f"  Deleted {path} (was vectr-only)", file=sys.stderr)


def _write_cursor_rules(workspace: str, *, search_only: bool = False) -> None:
    """Write .cursor/rules/vectr.mdc for Cursor IDE (vectr-owned file, always current)."""
    path = Path(workspace) / ".cursor" / "rules" / "vectr.mdc"
    content = (
        f"{load_template('cursor_rules_header.txt')}"
        f"{_render_claude_md(hooks_installed=False, search_only=search_only).rstrip()}\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists()
    if not existed or path.read_text(encoding="utf-8") != content:
        path.write_text(content, encoding="utf-8")
        print(f"  {'Updated' if existed else 'Created'} {path}", file=sys.stderr)


def _mcp_auth_headers(api_key: str = "", client_label: str = "") -> dict[str, str]:
    """Build the header block an MCP client must send to an authenticated vectr
    instance: the shared API key, plus an optional client-attribution label."""
    headers: dict[str, str] = {}
    if api_key:
        headers["X-Api-Key"] = api_key
    if client_label:
        headers["X-Vectr-Client"] = client_label
    return headers


def _inject_mcp_headers(config_json: str, headers: dict[str, str]) -> str:
    """Add a `headers` block to the vectr server entry of a rendered MCP config
    JSON string. Returns config_json unchanged when `headers` is empty, so the
    default keyless output stays byte-for-byte identical to before."""
    if not headers:
        return config_json
    data = json.loads(config_json)
    servers = data.get("mcpServers") or data.get("servers") or {}
    entry = servers.get("vectr")
    if isinstance(entry, dict):
        entry["headers"] = headers
    return json.dumps(data, indent=2) + "\n"


def _api_base(port: int) -> str:
    return f"http://localhost:{port}"


def _daemon_error_detail(exc: "httpx.HTTPStatusError") -> str:
    """Extract the server's structured `detail` message from a daemon error
    response, e.g. {"detail": {"error": "memory_only_mode", "detail": "..."}}
    (see app/routes.py's mode-gated 503s). Falls back to the raw status code
    when the body isn't in that shape.
    """
    try:
        body = exc.response.json()
    except Exception:
        return f"Vectr returned HTTP {exc.response.status_code}."
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, dict):
        return detail.get("detail") or detail.get("error") or str(detail)
    if isinstance(detail, str):
        return detail
    return f"Vectr returned HTTP {exc.response.status_code}."


def _handle_daemon_call_error(exc: Exception, port: int) -> None:
    """Uniform CLI error contract for every subcommand that talks to the
    daemon over REST (UPG-CLI-MEMONLY-CRASH): a down daemon and a daemon
    that declines the request (e.g. search/index disabled in memory-only
    mode, memory tools disabled in search-only mode) must both print one
    clean line to stderr and exit 1 — never an unhandled traceback exposing
    internal paths. Re-raises anything it doesn't recognise so a genuine bug
    still surfaces instead of being silently swallowed.
    """
    import httpx
    if isinstance(exc, httpx.ConnectError):
        print(f"Error: Vectr is not running on port {port}. Run: vectr start", file=sys.stderr)
        sys.exit(1)
    if isinstance(exc, httpx.HTTPStatusError):
        print(f"Error: {_daemon_error_detail(exc)}", file=sys.stderr)
        sys.exit(1)
    raise exc


def _stamp_parts(stamp: str) -> tuple[str, str | None]:
    """Split a version stamp into `(packaged-version, git-sha | None)`.

    `compute_version_stamp` builds the stamp by appending `+<short-sha>` to
    whatever the packaging metadata reports, and that metadata may already
    carry its own PEP 440 local segment (`1.7.0+dirty`) — so the sha is the
    part after the LAST `+`, never the first. A stamp with no `+`, or with an
    empty trailing component, has no sha: code identity is simply unknown,
    which is not the same as "no revision".
    """
    version, sep, sha = stamp.rpartition("+")
    if not sep:
        return stamp, None
    if not sha:
        return version, None
    return version, sha


def _same_revision(daemon_sha: str, local_sha: str) -> bool:
    """True when two git short shas name the same commit. Short-sha LENGTH is
    a per-repo/per-git-version setting (`core.abbrev`, and git widens it as a
    repo grows), so the same revision can legitimately print as `6925d4c` on
    one side and `6925d4c8a` on the other. Both are abbreviations of one
    40-hex id, so a prefix relation is the correct identity test — and git
    only ever emits an abbreviation that is unambiguous in the repo."""
    return daemon_sha.startswith(local_sha) or local_sha.startswith(daemon_sha)


def _version_skew_verdict(daemon_stamp: str, local_stamp: str) -> str | None:
    """Classify a daemon/CLI stamp pair as `None` (nothing to report),
    `"code"` (different revisions), `"version"` (no revision to compare, and
    the packaged versions differ) or `"metadata"` (same revision, different
    packaged version).

    UPG-VERSION-SKEW-FALSE-POSITIVE: the decision keys on the git sha,
    because the sha is what identifies the code and the version prefix does
    not. The live false positive was a daemon reporting `1.7.0+6925d4c`
    against a CLI reporting `1.6.0+6925d4c`, announced as "running older
    code": same sha (so the daemon was running exactly the code the CLI was
    invoked from) and, if anything, the NEWER version — the CLI's `1.6.0`
    was stale editable-install dist metadata that `pyproject.toml` had
    already moved past. Two independent lessons encoded here: a version
    prefix can be stale while the code is current, and "older" is not
    something a lexical comparison of two version strings can establish at
    all, so the wording never claims it.
    """
    if daemon_stamp == local_stamp:
        return None
    daemon_version, daemon_sha = _stamp_parts(daemon_stamp)
    local_version, local_sha = _stamp_parts(local_stamp)
    if daemon_sha and local_sha:
        if not _same_revision(daemon_sha, local_sha):
            return "code"
        # Same revision: whatever else differs, it is not the code. Only a
        # genuine disagreement between the two packaged version strings is
        # worth a word — differing sha ABBREVIATION lengths are not.
        return "metadata" if daemon_version != local_version else None
    # At least one side isn't running from a git checkout (installed copy),
    # so there is no revision to compare. The packaged version is then the
    # only identity available — and for a wheel install it is a real one.
    # Equal versions with only one sha present says nothing either way.
    return "version" if daemon_version != local_version else None


def _check_version_skew(
    port: int, daemon_status: dict | None = None, registry_entry: dict | None = None,
) -> None:
    """Warn once, to stderr, when the daemon on `port` is running code that
    differs from this CLI invocation's (UPG-CLI-DAEMON-VERSION-SKEW). This is
    the ONE shared choke point every daemon-talking subcommand calls — the
    comparison logic itself is never copy-pasted per subcommand.

    Never blocks or fails the calling command: any probe failure (daemon
    down, timeout, older daemon missing the field entirely) or a stamp that
    isn't available on either side is silently treated as "can't tell,"
    never as a mismatch. `daemon_status` lets a caller that already fetched
    `/v1/status` (e.g. `vectr status`) reuse that payload instead of a
    second round-trip; otherwise this fetches it directly with a short,
    best-effort timeout. `registry_entry` (optional — this function never
    looks one up itself, so it never touches the registry on a caller that
    has none to give) is passed straight through to `_restart_command`,
    which prefers its `code_workspace_file` for the printed suggestion
    (UPG-RESTART-DROPS-EXTRA-ROOTS).
    """
    try:
        if daemon_status is None:
            import httpx
            resp = httpx.get(
                f"{_api_base(port)}/v1/status", timeout=CLI_VERSION_SKEW_PROBE_TIMEOUT_S
            )
            resp.raise_for_status()
            daemon_status = resp.json()
        daemon_stamp = daemon_status.get("version_stamp")
        local_stamp = compute_version_stamp()
        if not daemon_stamp or not local_stamp:
            return
        verdict = _version_skew_verdict(daemon_stamp, local_stamp)
        if verdict is None:
            return
        if verdict == "metadata":
            # Same revision on both sides — the daemon IS running this code,
            # so there is nothing to restart and nothing to fix in vectr;
            # only the two dist-metadata version strings disagree (a stale
            # editable install being the usual cause). Reported rather than
            # hidden, because the mismatch is real and confusing wherever
            # either version is displayed.
            print(
                f"vectr: version metadata mismatch with the daemon on port {port} "
                f"({daemon_stamp} vs {local_stamp}) — same code revision "
                f"{_stamp_parts(daemon_stamp)[1]}, no restart needed "
                f"(reinstall to refresh this CLI's version metadata)",
                file=sys.stderr,
            )
            return
        difference = "different code" if verdict == "code" else "a different version"
        print(
            f"vectr: daemon on port {port} is running {difference} "
            f"({daemon_stamp} vs {local_stamp}) — run "
            f"'{_restart_command(daemon_status, registry_entry)}'",
            file=sys.stderr,
        )
    except Exception:
        return


def _restart_command(daemon_status: dict, registry_entry: dict | None = None) -> str:
    """The exact `vectr restart` invocation that brings THIS daemon back the
    way it is running now — workspace path plus, for a non-full daemon, its
    mode flag (UPG-RESTART-DROPS-MODE).

    The remediation text is the whole point of the banner, so it must not be
    the one thing that changes the daemon's mode: before `restart` inherited
    the recorded mode, following this advice on a memory-only instance turned
    full indexing + the file watcher back on. The flag is printed even now
    that inheritance exists, because it is also correct when the previous
    instance's registry record is gone (pruned, or a fresh machine).

    UPG-RESTART-DROPS-EXTRA-ROOTS: `cmd_restart` itself now reproduces a
    multi-root instance's `extra_roots` even from a bare primary-path
    restart (see `_resolve_restart_roots`), so this banner's suggestion
    already works either way. `registry_entry`, when given, lets it print
    the more literal form anyway — the `.code-workspace` file the instance
    was actually launched from, when the registry recorded one — instead of
    just the primary folder, since that is the more exact reproduction and
    doesn't rely on the reader trusting that the bare-path form silently
    restores every root. Falls back to `daemon_status["workspace_root"]`
    (the pre-existing behaviour) when no registry entry is available or it
    has no `code_workspace_file` on record.
    """
    workspace = (
        (registry_entry or {}).get("code_workspace_file")
        or daemon_status.get("workspace_root")
    )
    mode = daemon_status.get("mode", MODE_FULL)
    # A daemon too old to report its workspace leaves the original
    # fill-in-the-blank placeholder, which must stay unquoted to read as one.
    parts = ["vectr", "restart", shlex.quote(workspace) if workspace else "<workspace>"]
    if mode in (MODE_MEMORY_ONLY, MODE_SEARCH_ONLY):
        parts.append(MODE_RESTART_FLAG[mode])
    return " ".join(parts)


def _is_server_alive(port: int, timeout: float = 2.0) -> tuple[bool, str | None]:
    """Return (alive, workspace_root). Non-blocking within timeout."""
    try:
        import httpx
        resp = httpx.get(f"{_api_base(port)}/v1/status", timeout=timeout)
        resp.raise_for_status()
        return True, resp.json().get("workspace_root")
    except Exception:
        return False, None


def _wait_for_daemon_ready(port: int, pid: int) -> bool:
    """Poll /v1/status until the just-spawned daemon actually answers, or
    `cli.start_ready_poll_timeout_s` elapses (UPG-CLI-START-READY-RACE).
    Returns True only once the daemon is genuinely listening and responding
    — never optimistically. Returns False early (without waiting out the
    full timeout) if the spawned process has already exited, since a dead
    process will never start responding.
    """
    deadline = time.monotonic() + CLI_START_READY_POLL_TIMEOUT_S
    while time.monotonic() < deadline:
        alive, _ = _is_server_alive(port, timeout=CLI_START_READY_PROBE_TIMEOUT_S)
        if alive:
            return True
        if not _is_pid_alive(pid):
            return False
        time.sleep(CLI_START_READY_POLL_INTERVAL_S)
    return False


def _get_daemon_mode(port: int, timeout: float = 2.0) -> str | None:
    """Return the live daemon's mode ("full" / "memory-only" / "search-only"),
    or None if the daemon isn't reachable. Used by `vectr init` to decide
    whether hooks make sense (search-only has no working-memory layer to
    inject) — queried live rather than persisted, since mode is a daemon
    property, not workspace-static state."""
    try:
        import httpx
        resp = httpx.get(f"{_api_base(port)}/v1/status", timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("mode")
    except Exception:
        return None


def _stop_server(pid: int, timeout_s: int = 8) -> bool:
    """SIGTERM → wait → SIGKILL. Returns True if process is gone."""
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        time.sleep(0.3)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
    try:
        os.kill(pid, signal.SIGKILL)
        time.sleep(0.5)
    except ProcessLookupError:
        return True
    try:
        os.kill(pid, 0)
        return False  # still alive after SIGKILL — caller should log and continue
    except ProcessLookupError:
        return True


def _write_or_update(path: Path, content: str, label: str) -> None:
    """Write file if missing; overwrite if content changed (port update)."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"  Created {path}", file=sys.stderr)
    elif path.read_text(encoding="utf-8") != content:
        path.write_text(content, encoding="utf-8")
        print(f"  Updated {path} ({label})", file=sys.stderr)


_IDE_CONFIG_MARKER_REL = Path(".vectr") / "ide_config"


def _ide_config_disabled(workspace: str) -> bool:
    """True if this workspace durably opted out of vectr's automatic IDE
    integration file writes (UPG-CLI-WRITES-DISCLOSURE) via a prior
    `--no-ide-config` run. Persisted at `.vectr/ide_config` so subsequent
    start/restart/init calls honor the choice without repeating the flag —
    delete the file to re-enable."""
    marker = Path(workspace) / _IDE_CONFIG_MARKER_REL
    if not marker.exists():
        return False
    try:
        return marker.read_text(encoding="utf-8").strip() == "disabled"
    except OSError:
        return False


def _persist_ide_config_disabled(workspace: str) -> None:
    marker = Path(workspace) / _IDE_CONFIG_MARKER_REL
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("disabled\n", encoding="utf-8")


def _maybe_write_workspace_config(
    workspace: str, port: int, args: argparse.Namespace, *, search_only: bool = False,
) -> None:
    """Gate `_write_workspace_config` on `--no-ide-config` or a prior durable
    opt-out (UPG-CLI-WRITES-DISCLOSURE): `start`/`restart`/`init` previously
    wrote 7 IDE integration files into the workspace root on every call with
    no disclosure in --help and no way to opt out. `--no-ide-config` skips
    the writes for this call AND persists the choice (`.vectr/ide_config`)
    so later start/restart calls keep honoring it without repeating the flag.
    """
    if getattr(args, "no_ide_config", False):
        _persist_ide_config_disabled(workspace)
        print(
            f"  Skipped IDE config file writes for {workspace} (--no-ide-config). "
            f"This choice persists at {Path(workspace) / _IDE_CONFIG_MARKER_REL} "
            f"for future start/restart/init calls; delete that file to re-enable.",
            file=sys.stderr,
        )
        return
    if _ide_config_disabled(workspace):
        print(
            f"  IDE config file writes are disabled for {workspace} "
            f"({Path(workspace) / _IDE_CONFIG_MARKER_REL}). Delete that file to re-enable.",
            file=sys.stderr,
        )
        return
    _write_workspace_config(workspace, port, search_only=search_only)


def _render_codex_mcp_block(port: int, headers: dict[str, str]) -> str:
    """Render vectr's `[mcp_servers.vectr]` TOML block for Codex CLI, splicing
    an `http_headers` inline table when the daemon is authenticated (keyless
    daemons — the default — get the header-free block, byte-identical to the
    template). Codex's HTTP MCP schema names this field `http_headers`, not the
    JSON `headers` key the editor MCP configs use (research §Q1)."""
    hdr = ""
    if headers:
        inline = ", ".join(f'"{k}" = "{v}"' for k, v in headers.items())
        hdr = f"\nhttp_headers = {{ {inline} }}"
    return _CODEX_MCP_TOML.format(port=port).replace("__HEADERS__", hdr)


def _write_codex_mcp_config(path: Path, port: int, headers: dict[str, str]) -> None:
    """Write/refresh vectr's `[mcp_servers.vectr]` region in `.codex/config.toml`.

    Merge-safe by comment-delimited region (mirrors `_write_ide_config_merge_safe`
    for markdown): only vectr's own `# vectr-start`…`# vectr-end` block is
    touched; any other TOML the user has (trust, model, other servers) is
    preserved verbatim. Idempotent — a port change rewrites the block in place;
    an unchanged block is a no-op. The merged output is validated as parseable
    TOML before it is written, so a pre-existing malformed config is left
    untouched rather than compounded."""
    block = _render_codex_mcp_block(port, headers)
    existed = path.exists()

    if not existed:
        new_content = block
    else:
        existing = path.read_text(encoding="utf-8")
        stripped = _CODEX_TOML_BLOCK_RE.sub("", existing).rstrip()
        new_content = f"{stripped}\n\n{block}" if stripped else block
        if new_content == existing:
            return

    try:
        tomllib.loads(new_content)
    except tomllib.TOMLDecodeError:
        print(
            f"  Skipped {path}: existing content is not valid TOML — "
            f'add [mcp_servers.vectr] url = "http://localhost:{port}/mcp" by hand.',
            file=sys.stderr,
        )
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new_content, encoding="utf-8")
    print(f"  {'Updated vectr block in' if existed else 'Created'} {path}", file=sys.stderr)


def _dsh_home() -> Path:
    """Locate the DeepSeek Harness home directory: `$DSH_HOME` if set, else
    `~/.dsh` (the default the harness itself uses)."""
    env_home = os.getenv("DSH_HOME")
    if env_home:
        return Path(env_home)
    return Path.home() / ".dsh"


def _dsh_profile() -> str:
    """Resolve the dsh profile vectr should register itself in: `$DSH_PROFILE`
    first (the harness's own selector), then `$VECTR_DSH_PROFILE` (a
    vectr-specific override for hosts that pin a different profile), then the
    `web` default (the browser GUI profile, the most common local setup)."""
    return os.getenv("DSH_PROFILE") or os.getenv("VECTR_DSH_PROFILE") or _DSH_DEFAULT_PROFILE


def _dsh_patch_path() -> Path:
    """Path to the active dsh profile's `cordis.patch.yml`, the file vectr's
    `mcp-client.vectr` insert row is merged into."""
    return _dsh_home() / "profiles" / _dsh_profile() / "cordis.patch.yml"


def _render_dsh_mcp_block(port: int, headers: dict[str, str]) -> str:
    """Render vectr's `- insert:` MCP row for a dsh `cordis.patch.yml`,
    splicing a `headers` block when the daemon is authenticated (keyless
    daemons — the default — get the header-free block, byte-identical to the
    template). dsh's Streamable HTTP config names the field `headers`, the
    same key the editor JSON MCP configs use."""
    hdr = ""
    if headers:
        hdr = "\n        headers:"
        for key, value in headers.items():
            hdr += f"\n          {key}: {value}"
    return _DSH_PATCH_TEMPLATE.format(port=port).replace("__HEADERS__", hdr)


def _render_dsh_hooks_block(config_path: str) -> str:
    """Render vectr's `- insert:` hook-bridge row for a dsh `cordis.patch.yml`.
    `config_path` is the ABSOLUTE path to the workspace's `.claude/settings.
    json` (dsh resolves `configPath` once at load against the server's launch
    cwd, so a relative path would point at the wrong tree)."""
    return _DSH_HOOKS_PATCH_TEMPLATE.format(config_path=json.dumps(config_path))


def _merge_dsh_patch_block(
    existing: str, block: str, block_start: str, block_re: "re.Pattern[str]",
) -> str | None:
    """Merge one vectr comment-delimited block into the text of a dsh
    `cordis.patch.yml`, returning the new content or `None` when the result
    would be byte-identical to the input (a no-op caller skips the write).

    Shared by the MCP-row and hook-bridge writers. When the block's own marker
    region is already present it is replaced IN PLACE (so a file holding both
    the MCP and hooks blocks keeps each block where it is, no matter which
    writer runs last); otherwise a fresh `[]` empty array is replaced by the
    block, and a file with real content gets the block appended as a new
    top-level item. In every case any other patch entries survive verbatim."""
    if block_start in existing:
        new_content = block_re.sub(block, existing)
        return None if new_content == existing else new_content
    match = _DSH_EMPTY_ARRAY_RE.search(existing)
    if match:
        # Fresh `[]` (optionally under header comments): replace the empty
        # array with vectr's block, keeping the comments.
        return f"{match.group(1).rstrip()}\n{block}" if match.group(1).strip() else block
    # Real content: append the block as a new top-level item.
    return f"{existing.rstrip()}\n\n{block}"


def _write_dsh_mcp_config(path: Path, port: int, headers: dict[str, str]) -> None:
    """Write/refresh vectr's `mcp-client.vectr` insert row in a dsh profile's
    `cordis.patch.yml`.

    Merge-safe by comment-delimited region (mirrors `_write_codex_mcp_config`
    for TOML): only vectr's own `# vectr-start`…`# vectr-end` block is
    touched; any other patch entries the user has (their own inserts,
    disables) are preserved verbatim. Idempotent — a port change rewrites the
    block in place; an unchanged block is a no-op.

    The file is a top-level YAML array. A fresh `cordis.patch.yml` ships as
    `[]` (possibly under a header comment), so appending a bare `- insert:`
    item after it would be invalid YAML — the empty array is replaced by the
    block instead. A file with real content gets the block appended as a new
    top-level item.
    """
    block = _render_dsh_mcp_block(port, headers)
    existed = path.exists()
    if not existed:
        new_content = block
    else:
        merged = _merge_dsh_patch_block(
            path.read_text(encoding="utf-8"), block, _DSH_BLOCK_START, _DSH_BLOCK_RE,
        )
        if merged is None:
            return
        new_content = merged
    _write_dsh_patch_validated(path, new_content, existed, "mcp-client.vectr insert row")


def _write_dsh_hooks_config(path: Path, config_path: str) -> None:
    """Write/refresh vectr's `hooks-claude-code` hook-bridge row in a dsh
    profile's `cordis.patch.yml`, pointing the bridge at the workspace's
    `.claude/settings.json`.

    Same merge-safe contract as `_write_dsh_mcp_config`, under its own
    `# vectr-hooks-start`…`# vectr-hooks-end` markers so the two blocks are
    managed independently (hooks are installed by `vectr init --hooks`, MCP by
    every `vectr start`/`init`)."""
    block = _render_dsh_hooks_block(config_path)
    existed = path.exists()
    if not existed:
        new_content = block
    else:
        merged = _merge_dsh_patch_block(
            path.read_text(encoding="utf-8"), block, _DSH_HOOKS_BLOCK_START, _DSH_HOOKS_BLOCK_RE,
        )
        if merged is None:
            return
        new_content = merged
    _write_dsh_patch_validated(path, new_content, existed, "hooks-claude-code row")


def _write_dsh_patch_validated(path: Path, new_content: str, existed: bool, what: str) -> None:
    """Validate `new_content` parses as a top-level YAML array, then write it.
    A merged file that would not parse is left untouched with a README pointer
    rather than compounding a malformed pre-existing patch."""
    try:
        _validate_dsh_patch_yaml(new_content)
    except Exception as exc:
        print(
            f"  Skipped {path}: merged content is not valid YAML ({exc}) — "
            f"add the {what} by hand (see README).",
            file=sys.stderr,
        )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new_content, encoding="utf-8")
    print(f"  {'Updated vectr block in' if existed else 'Created'} {path}", file=sys.stderr)



def _validate_dsh_patch_yaml(content: str) -> None:
    """Parse `content` as a top-level YAML array of patch entries to prove the
    merged `cordis.patch.yml` is loadable before it is written. A pre-existing
    malformed patch is left untouched rather than compounded.

    `!!js` tags (which dsh's loader evaluates) are unknown to PyYAML, so the
    check registers the `!!js` tag on a `SafeLoader` subclass as a passthrough
    that returns the raw scalar: any OTHER unknown tag or a structurally
    invalid document still raises, while a user's `!!js` expressions (rare in
    a patch file, but legal) don't false-positive the guard."""
    import yaml

    def _js_constructor(loader: yaml.Loader, node: yaml.Node):
        return loader.construct_scalar(node)

    class _Loader(yaml.SafeLoader):
        pass

    _Loader.add_constructor("tag:yaml.org,2002:js", _js_constructor)
    parsed = yaml.load(content, Loader=_Loader)
    if not isinstance(parsed, list):
        raise ValueError("top-level value is not a YAML array")


def _write_workspace_config(workspace: str, port: int, *, search_only: bool = False) -> None:
    """Write per-IDE MCP config files and IDE guidance into the workspace root.

    CLAUDE.md's session-start guidance is hook-aware (UPG-11.5): if Claude Code
    hooks are already installed for this workspace, the injected notes make a
    self-recall at session start redundant, so CLAUDE.md is written with the
    hook-aware variant instead. Other IDE config files always get the default
    variant — Claude Code hooks are the only automatic-injection path today.

    `search_only` (UPG-SEARCH-ONLY-MODE) takes precedence over the hook-aware
    variant: this daemon has no working-memory layer, so every IDE config file
    gets the search-only guidance (search tools only, no memory section).
    """
    root = Path(workspace)
    hooks_installed = _hooks_installed(workspace)

    # UPG-13.2: seed a default .vectrignore on first start/init so users get
    # sensible excludes (node_modules, .venv, __pycache__, ...) without hand-
    # authoring them. No-op (never overwrites) if one already exists.
    from integrations.workspace_detect import write_default_vectrignore
    if write_default_vectrignore(workspace):
        print(f"  Created {root / '.vectrignore'} (default excludes)", file=sys.stderr)

    # AGENTS.md (Codex CLI) gets the hook-aware guidance variant when Codex
    # hooks are installed for this workspace (UPG-CODEX-PARITY), mirroring how
    # CLAUDE.md keys on Claude Code hooks above; the other append-only files
    # have no automatic-injection path, so they always get the default variant.
    codex_hooks_installed = _codex_hooks_installed(workspace)
    _write_ide_config_merge_safe(
        root / "CLAUDE.md", create_if_missing=True, hooks_installed=hooks_installed, search_only=search_only,
        tool_loading=True,
    )
    for _rel in _IDE_CONFIG_APPEND_ONLY:
        file_hooks = codex_hooks_installed if _rel == "AGENTS.md" else False
        _write_ide_config_merge_safe(
            root / _rel, create_if_missing=False, hooks_installed=file_hooks, search_only=search_only,
        )
    _write_ide_config_merge_safe(
        root / ".github" / "copilot-instructions.md", create_if_missing=False, search_only=search_only,
    )
    _write_cursor_rules(workspace, search_only=search_only)

    # When this workspace's daemon runs with authentication enabled
    # (VECTR_API_KEY set), the local editor's MCP config must carry the key
    # header too, or the editor can no longer reach its own daemon. Keyless
    # daemons (the default) get the unchanged, header-free config.
    headers = _mcp_auth_headers(api_key=os.getenv("VECTR_API_KEY", ""))
    _write_or_update(root / ".mcp.json", _inject_mcp_headers(_MCP_JSON.format(port=port), headers), f"port {port}")
    _write_or_update(root / ".cursor" / "mcp.json", _inject_mcp_headers(_CURSOR_MCP_JSON.format(port=port), headers), f"port {port}")
    _write_or_update(root / ".vscode" / "mcp.json", _inject_mcp_headers(_VSCODE_MCP_JSON.format(port=port), headers), f"port {port}")
    # Codex CLI (UPG-CODEX-PARITY): project-scoped .codex/config.toml. Written
    # unconditionally like the other MCP configs; note that Codex only loads a
    # project-scoped config after the repo is trusted through its one-time
    # interactive trust prompt (research §Q1) — an expected first-run step, not
    # something an auto-writer can (or should) clear on the user's behalf.
    _write_codex_mcp_config(root / ".codex" / "config.toml", port, headers)

    # DeepSeek Harness (UPG-DSH-PARITY): register vectr in the ACTIVE dsh
    # profile's cordis.patch.yml (profile-global, unlike the per-workspace
    # editor files above). Skipped silently when dsh is not installed
    # (`~/.dsh` absent) — the harness is not the target host for every user.
    if _dsh_home().exists():
        _write_dsh_mcp_config(_dsh_patch_path(), port, headers)

    # Merge-safe (not create-only): UPG-11.5 reordered `vectr init --hooks` to
    # write hooks before workspace config in the same run, so settings.json can
    # already exist (hooks-only) by the time we get here — still needs this key.
    _ensure_enable_all_project_mcp_servers(root)


# Editor MCP config files `_write_workspace_config` above writes SYNCHRONOUSLY
# in this (CLI) process, relative to a workspace root — kept in one place so
# the stale-config check below stays in sync with the writers.
#
# Deliberately excludes `.claude/settings.json`: that file's vectr MCP entry
# is written by the daemon SUBPROCESS itself (`integrations.vscode_bridge.
# configure_claude_code`), asynchronously, near the end of its phase-2
# startup (after the embedding model loads) — well after `_do_start` below
# returns. Checking it here, synchronously, would false-positive on almost
# every start (the daemon just hasn't gotten to it yet), which would train
# users to ignore the warning. Its port instead self-heals once the daemon's
# write runs, now that `configure_claude_code`'s merge always syncs vectr's
# own managed keys (see `_merge_json_file`'s `owned_keys` in vscode_bridge.py).
_IDE_MCP_CONFIG_FILES: tuple[str, ...] = (
    ".mcp.json",
    ".cursor/mcp.json",
    ".vscode/mcp.json",
    ".codex/config.toml",
)

_MCP_URL_PORT_RE = re.compile(r"localhost:(\d+)/mcp")


def _configured_mcp_ports(workspace: str) -> dict[str, int]:
    """Port each known editor MCP config file under `workspace` currently
    points vectr's own entry at, keyed by path relative to `workspace`. Files
    that don't exist, or whose contents don't contain a recognizable vectr
    MCP URL, are omitted — never guessed."""
    found: dict[str, int] = {}
    root = Path(workspace)
    for rel in _IDE_MCP_CONFIG_FILES:
        path = root / rel
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        match = _MCP_URL_PORT_RE.search(text)
        if match:
            found[rel] = int(match.group(1))
    return found


def _warn_on_stale_mcp_configs(workspace: str, port: int) -> None:
    """Loud, explicit warning (UPG-RESTART-PORT-WALK-BREAKS-MCP) when any
    editor MCP config file under `workspace` still points at a port other
    than the one vectr actually just bound — e.g. IDE config writes were
    disabled for this workspace (`--no-ide-config`), or a write failed after
    a previous run. Never silent: an AI code editor reading a stale port
    gets a dead MCP connection with nothing in vectr's own output naming the
    cause, which is exactly the failure mode this exists to prevent."""
    stale = {
        rel: configured_port
        for rel, configured_port in _configured_mcp_ports(workspace).items()
        if configured_port != port
    }
    if not stale:
        return
    print(
        f"Warning: vectr is running on port {port}, but the following editor "
        f"MCP config file(s) still point at a different port. The editor's "
        f"vectr connection will fail until these are reconciled:",
        file=sys.stderr,
    )
    for rel, configured_port in sorted(stale.items()):
        print(f"  {Path(workspace) / rel} -> port {configured_port}", file=sys.stderr)
    print(
        "  Reconcile with: vectr start (rewrites these files to the running "
        "port) — or, if you passed --no-ide-config, edit them by hand.",
        file=sys.stderr,
    )


def _ensure_enable_all_project_mcp_servers(root: Path) -> None:
    """Merge `enableAllProjectMcpServers: true` into .claude/settings.json,
    preserving any existing keys. Shared by the local and remote config writers."""
    settings = root / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True, exist_ok=True)
    if not settings.exists():
        settings.write_text('{\n  "enableAllProjectMcpServers": true\n}\n')
        print(f"  Created {settings}", file=sys.stderr)
    else:
        try:
            data = json.loads(settings.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        if not data.get("enableAllProjectMcpServers"):
            data["enableAllProjectMcpServers"] = True
            settings.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            print(f"  Updated {settings} (enableAllProjectMcpServers)", file=sys.stderr)


def _normalize_mcp_url(url: str) -> str:
    """Return the MCP endpoint URL for a remote vectr instance. Accepts either
    a base URL (`http://host:8765`) or one already ending in `/mcp`."""
    url = url.strip().rstrip("/")
    if not url.endswith("/mcp"):
        url = url + "/mcp"
    return url


def _host_from_url(url: str) -> str:
    """Extract the hostname from an http(s) URL (no port), for the loopback
    check in `vectr connect`. Best-effort; returns "" if it can't be parsed."""
    from urllib.parse import urlparse
    try:
        return urlparse(url).hostname or ""
    except Exception:
        return ""


def _remote_mcp_configs(url: str, headers: dict[str, str]) -> dict[str, str]:
    """Build the three editor MCP config JSON texts pointing at a remote vectr
    instance, with optional auth/attribution headers. Keyed by relative path."""
    claude = {"mcpServers": {"vectr": {"type": "http", "url": url}}}
    cursor = {"mcpServers": {"vectr": {"url": url}}}
    vscode = {"servers": {"vectr": {"type": "http", "url": url}}}
    if headers:
        claude["mcpServers"]["vectr"]["headers"] = headers
        cursor["mcpServers"]["vectr"]["headers"] = headers
        vscode["servers"]["vectr"]["headers"] = headers

    def _dump(d: dict) -> str:
        return json.dumps(d, indent=2) + "\n"

    return {
        ".mcp.json": _dump(claude),
        ".cursor/mcp.json": _dump(cursor),
        ".vscode/mcp.json": _dump(vscode),
    }


def _write_remote_workspace_config(workspace: str, url: str, headers: dict[str, str]) -> list[str]:
    """Configure a local editor to use a REMOTE vectr instance (team / central
    mode). Writes the editor guidance blocks and the three MCP configs pointing
    at `url` with the given headers — but spawns no daemon and registers no
    instance. This is the client half of the client/server split.

    Returns the workspace-relative paths it manages (created, updated, or an
    already-present append-only guidance file it merged into), so the caller can
    print the exact file list for the user (UPG-TEAM-4). Append-only guidance
    files that do not already exist are never created, so they are excluded."""
    root = Path(workspace)
    managed: list[str] = []
    # Editor guidance so the MCP client's LLM knows the tools exist.
    _write_ide_config_merge_safe(root / "CLAUDE.md", create_if_missing=True, tool_loading=True)
    managed.append("CLAUDE.md")
    for _rel in _IDE_CONFIG_APPEND_ONLY:
        _write_ide_config_merge_safe(root / _rel, create_if_missing=False)
        if (root / _rel).exists():
            managed.append(_rel)
    copilot = Path(".github") / "copilot-instructions.md"
    _write_ide_config_merge_safe(root / copilot, create_if_missing=False)
    if (root / copilot).exists():
        managed.append(copilot.as_posix())
    _write_cursor_rules(workspace)
    managed.append(".cursor/rules/vectr.mdc")

    for rel, text in _remote_mcp_configs(url, headers).items():
        _write_or_update(root / rel, text, "remote")
        managed.append(rel)
    _ensure_enable_all_project_mcp_servers(root)
    managed.append(".claude/settings.json")
    return managed


# The connect-written MCP configs that embed the API key in plaintext (UPG-TEAM-4).
_REMOTE_KEY_BEARING_FILES = frozenset({".mcp.json", ".cursor/mcp.json", ".vscode/mcp.json"})


def _is_vectr_hook_group(group: dict) -> bool:
    """True if a hook group contains a vectr-managed command (for idempotent re-init)."""
    for h in group.get("hooks", []):
        if isinstance(h, dict) and str(h.get("command", "")).startswith("vectr hook"):
            return True
    return False


def _data_has_vectr_hook_group(data: object) -> bool:
    """True if a parsed hooks-config dict (`.claude/settings.json` or
    `.codex/hooks.json`) carries a vectr-managed hook group under `hooks`."""
    if not isinstance(data, dict):
        return False
    hooks = data.get("hooks")
    if not isinstance(hooks, dict):
        return False
    for groups in hooks.values():
        if not isinstance(groups, list):
            continue
        for group in groups:
            if isinstance(group, dict) and _is_vectr_hook_group(group):
                return True
    return False


def _config_has_vectr_hooks(path: Path) -> bool:
    """True if `path` parses and holds a vectr-managed hook group. Never raises:
    a missing/malformed file just means "no vectr hooks installed yet"."""
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return _data_has_vectr_hook_group(data)


def _hooks_installed(workspace: str) -> bool:
    """True if `<workspace>/.claude/settings.json` already has a vectr-managed
    hook group (UPG-11.5) — selects CLAUDE.md's hook-aware session-start
    guidance."""
    return _config_has_vectr_hooks(Path(workspace) / ".claude" / "settings.json")


def _codex_hooks_installed(workspace: str) -> bool:
    """True if `<workspace>/.codex/hooks.json` already has a vectr-managed hook
    group (UPG-CODEX-PARITY) — selects AGENTS.md's hook-aware session-start
    guidance, mirroring `_hooks_installed` for Claude Code. Codex injects
    recalled notes automatically once its hooks are installed, so AGENTS.md
    should stop telling the model to self-call vectr_recall at session start,
    exactly as CLAUDE.md does for Claude Code hooks."""
    return _config_has_vectr_hooks(Path(workspace) / ".codex" / "hooks.json")


def _install_hook_group(hooks: dict, event: str, *, command: str, matcher: str | None = None) -> None:
    """Insert (or replace) the vectr-managed hook group for an event, in place.

    Idempotent: any prior vectr group for this event is dropped first, so
    re-running `vectr init --hooks` never duplicates entries, and non-vectr
    hook groups the user added are left untouched.
    """
    groups = hooks.setdefault(event, [])
    groups[:] = [g for g in groups if not _is_vectr_hook_group(g)]
    group: dict = {"hooks": [{"type": "command", "command": command}]}
    if matcher is not None:
        group = {"matcher": matcher, **group}
    groups.append(group)


def _write_claude_hooks(workspace: str) -> None:
    """Merge vectr's hook entries into <workspace>/.claude/settings.json (UPG-9.4+).

    Preserves any existing settings (e.g. enableAllProjectMcpServers) and any
    non-vectr hooks. Each vectr hook calls the `vectr hook <event>` subcommand,
    which owns the Claude Code output contract.
    """
    settings = Path(workspace) / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    if settings.exists():
        try:
            data = json.loads(settings.read_text(encoding="utf-8"))
        except Exception:
            data = {}  # malformed — overwrite rather than crash init
    hooks = data.setdefault("hooks", {})

    # UPG-9.4 — SessionStart: inject the boot set (directives + high tasks) before turn 1.
    _install_hook_group(hooks, "SessionStart", matcher="startup|resume|clear|compact",
                        command="vectr hook session-start")
    # UPG-9.5 — UserPromptSubmit (no matcher): per-turn semantic recall keyed to the prompt.
    _install_hook_group(hooks, "UserPromptSubmit", command="vectr hook user-prompt-submit")
    # UPG-9.6 — PreToolUse (Edit|Write|Read): surface the gotcha recorded against
    # the file about to be read or edited. Extended to Read (UPG-HOOK-INJECT-
    # OBSERVABILITY): a file-reading tool has just as deterministic a
    # tool_input.file_path as Edit/Write, so it gets the same gotcha injection
    # at the moment the model is about to look at that file, not only when
    # it's about to change it. Extended to Bash: a command-running tool call
    # carries just as deterministic a
    # tool_input.command as Edit/Write carries a file_path, so a note anchored
    # to a command family (the `command` trigger axis) fires at the moment the
    # model is about to run it, not only when it's about to touch a file.
    _install_hook_group(hooks, "PreToolUse", matcher="Edit|Write|Read|Bash",
                        command="vectr hook pre-tool-use")
    # Episode capture — PostToolUse (Bash|Edit|Write|MultiEdit): record one
    # deterministic episode row per tool call. Foreground work is a stdin
    # read + a detached-worker spawn only (see cmd_hook's "post-tool-use"
    # branch) — never blocks the turn loop on the actual HTTP write.
    _install_hook_group(hooks, "PostToolUse", matcher="Bash|Edit|Write|MultiEdit",
                        command="vectr hook post-tool-use")
    # G0 live capture: PostToolUse fires ONLY on tool SUCCESS — a failed Bash call (non-zero exit) fires
    # PostToolUseFailure instead, and PostToolUse never fires for it at all.
    # Without this second group, every RED run (the failure half of every
    # failure->fix->success arc, arguably the most valuable episode class)
    # is silently never captured. Same matcher and same `vectr hook
    # post-tool-use` command — the two events share one CLI invocation;
    # `_build_episode_payload` tells them apart via the JSON's own
    # `hook_event_name` field, not a separate subcommand.
    _install_hook_group(hooks, "PostToolUseFailure", matcher="Bash|Edit|Write|MultiEdit",
                        command="vectr hook post-tool-use")
    # UPG-9.7 — PreCompact (manual|auto): snapshot working memory before /compact replaces context.
    _install_hook_group(hooks, "PreCompact", matcher="manual|auto",
                        command="vectr hook pre-compact")

    settings.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"  Wrote vectr hooks to {settings}", file=sys.stderr)


def _write_codex_hooks(workspace: str) -> None:
    """Merge vectr's hook entries into <workspace>/.codex/hooks.json (UPG-CODEX-PARITY).

    Codex CLI's hooks.json schema, event names, matcher vocabulary, and the
    `hookSpecificOutput.additionalContext` envelope are the same as Claude
    Code's (research/codex-parity-2026-07.md §Q3), so the same `vectr hook
    <event>` commands and `_install_hook_group` idempotency apply unchanged —
    `_emit_hook_context` emits the identical envelope Codex reads. Two
    differences from `_write_claude_hooks`: the file is `.codex/hooks.json`
    (whose top level IS the hooks object, not nested under a settings file),
    and the PreToolUse matcher also covers Codex's native `apply_patch` edit
    tool. Codex's exact PreToolUse tool name is the one matcher the research
    could not confirm without a live smoke test; the `Edit|Write|apply_patch`
    union is harmless either way — an absent tool name simply never fires the
    hook — so it needs no live verification to ship safely. The PostToolUse
    matcher includes `apply_patch` too: without it, no edit episode is ever
    captured under Codex, and the edit-mediated arc (the dominant real fix
    loop) can never form there.

    Codex's hook surface documents no distinct failure-path event
    (research/codex-parity-2026-07.md lists only "PostToolUse" among its
    event names) — this writer deliberately does NOT install a
    PostToolUseFailure group the way `_write_claude_hooks` does; inventing
    an event name Codex doesn't fire would just be a silent no-op, and a
    real one should be added here once confirmed, not guessed.

    Preserves any existing hooks.json content and any non-vectr hook groups.
    """
    hooks_file = Path(workspace) / ".codex" / "hooks.json"
    hooks_file.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    if hooks_file.exists():
        try:
            data = json.loads(hooks_file.read_text(encoding="utf-8"))
        except Exception:
            data = {}  # malformed — overwrite rather than crash init
    hooks = data.setdefault("hooks", {})

    _install_hook_group(hooks, "SessionStart", matcher="startup|resume|clear|compact",
                        command="vectr hook session-start")
    _install_hook_group(hooks, "UserPromptSubmit", command="vectr hook user-prompt-submit")
    # Adds Bash alongside apply_patch (same rationale as the PreToolUse group
    # in _write_claude_hooks): an absent
    # tool name is a harmless no-op, so this needs no live verification of
    # Codex's exact shell-tool name to ship safely.
    _install_hook_group(hooks, "PreToolUse", matcher="Edit|Write|apply_patch|Bash",
                        command="vectr hook pre-tool-use")
    # Episode capture — see _write_claude_hooks's PostToolUse group for the
    # rationale; same event and command apply unchanged under Codex's
    # identical schema. The matcher adds `apply_patch` (see module
    # docstring) alongside the Claude-compat tool names, mirroring the
    # PreToolUse matcher above.
    _install_hook_group(hooks, "PostToolUse", matcher="Bash|Edit|Write|MultiEdit|apply_patch",
                        command="vectr hook post-tool-use")
    _install_hook_group(hooks, "PreCompact", matcher="manual|auto",
                        command="vectr hook pre-compact")

    hooks_file.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"  Wrote vectr hooks to {hooks_file}", file=sys.stderr)


def _remove_vectr_hooks(workspace: str) -> None:
    """Strip vectr-managed hook groups from .claude/settings.json (for --reset-config).

    Leaves all other settings and any non-vectr hooks intact; drops now-empty
    hook-event lists and an empty `hooks` key.
    """
    settings = Path(workspace) / ".claude" / "settings.json"
    if not settings.exists():
        return
    try:
        data = json.loads(settings.read_text(encoding="utf-8"))
    except Exception:
        return
    hooks = data.get("hooks")
    if not isinstance(hooks, dict):
        return
    changed = False
    for event in list(hooks.keys()):
        groups = hooks[event]
        if not isinstance(groups, list):
            continue
        kept = [g for g in groups if not _is_vectr_hook_group(g)]
        if len(kept) != len(groups):
            changed = True
        if kept:
            hooks[event] = kept
        else:
            del hooks[event]
    if not hooks:
        data.pop("hooks", None)
    if changed:
        settings.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"  Removed vectr hooks from {settings}", file=sys.stderr)


def _remove_codex_hooks(workspace: str) -> None:
    """Strip vectr-managed hook groups from .codex/hooks.json (for --reset-config).

    Mirrors `_remove_vectr_hooks`, but `.codex/hooks.json`'s top level IS the
    hooks object, so once vectr's groups are gone and nothing else remains the
    file is deleted (it was vectr-only) rather than left as an empty `{}`.
    Non-vectr hook groups a user added are preserved."""
    hooks_file = Path(workspace) / ".codex" / "hooks.json"
    if not hooks_file.exists():
        return
    try:
        data = json.loads(hooks_file.read_text(encoding="utf-8"))
    except Exception:
        return
    hooks = data.get("hooks")
    if not isinstance(hooks, dict):
        return
    changed = False
    for event in list(hooks.keys()):
        groups = hooks[event]
        if not isinstance(groups, list):
            continue
        kept = [g for g in groups if not _is_vectr_hook_group(g)]
        if len(kept) != len(groups):
            changed = True
        if kept:
            hooks[event] = kept
        else:
            del hooks[event]
    if not hooks:
        data.pop("hooks", None)
    if not changed:
        return
    if data:
        hooks_file.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"  Removed vectr hooks from {hooks_file}", file=sys.stderr)
    else:
        hooks_file.unlink()
        print(f"  Deleted {hooks_file} (was vectr-only)", file=sys.stderr)


def _remove_codex_mcp_config(workspace: str) -> None:
    """Remove vectr's `[mcp_servers.vectr]` block from .codex/config.toml (for
    --reset-config). Deletes the file if it becomes empty (was vectr-only);
    otherwise leaves the user's other TOML intact."""
    path = Path(workspace) / ".codex" / "config.toml"
    if not path.exists():
        return
    content = path.read_text(encoding="utf-8")
    if _CODEX_TOML_BLOCK_START not in content:
        return
    stripped = _CODEX_TOML_BLOCK_RE.sub("", content).rstrip()
    if stripped:
        path.write_text(stripped + "\n", encoding="utf-8")
        print(f"  Removed vectr block from {path}", file=sys.stderr)
    else:
        path.unlink()
        print(f"  Deleted {path} (was vectr-only)", file=sys.stderr)


def _git_hooks_dir(workspace: str) -> Path | None:
    """Resolve the git hooks directory for `workspace` via `git rev-parse
    --git-path hooks` (UPG-COMMIT-MEMORY-HOOK) — correct for both a plain
    repo (`.git/hooks`) and a linked worktree, where `.git` is a FILE
    pointing at a shared common git dir whose hooks/ subdirectory every
    worktree of that repo shares (a hardcoded `.git/hooks` join is wrong
    there). Returns None when `workspace` is not inside a git working tree,
    or `git` itself is unavailable — callers treat that as "skip silently,
    with a disclosed message," never an error."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-path", "hooks"],
            cwd=workspace, capture_output=True, text=True, timeout=_GIT_SUBPROCESS_TIMEOUT_S,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    raw = result.stdout.strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = Path(workspace) / path
    return path.resolve()


def _shell_shebang_compatible(first_line: str) -> bool:
    """True if `first_line` (a hook file's own first line) is a shebang for
    one of `_POSIX_SHELL_NAMES` — handles both a direct (`#!/bin/sh`) and an
    `env`-wrapped (`#!/usr/bin/env bash`) shebang form, matched against the
    interpreter path's own basename (never a substring, so `#!/usr/bin/env
    pwsh` is correctly rejected rather than false-positiving on a stray
    "sh"-like name). A missing/absent shebang is treated the same as an
    incompatible one — a hook file with no `#!` line isn't reliably
    executable as a shell script in the first place, so vectr's POSIX-shell
    block must not be appended under it either."""
    first_line = first_line.strip()
    if not first_line.startswith("#!"):
        return False
    parts = first_line[2:].strip().split()
    if not parts:
        return False
    interpreter = parts[0]
    if Path(interpreter).name == "env" and len(parts) > 1:
        interpreter = parts[1]
    return Path(interpreter).name in _POSIX_SHELL_NAMES


def _write_git_post_commit_hook(workspace: str) -> None:
    """Install the git post-commit hook that records one deterministic
    working-memory note per commit (UPG-COMMIT-MEMORY-HOOK): git already
    records WHAT changed; this captures the commit's identity, touched
    files, and active task context so a future session (or this one, past
    /compact) can recall WHY without re-deriving it from `git log`/`git
    show`. The hook itself only invokes `vectr hook post-commit`
    (backgrounded — see the template); all interpretation happens in
    `cmd_hook`'s post-commit branch.

    Merge-safe like `_write_ide_config_merge_safe`/`_write_codex_mcp_config`:
    only vectr's own comment-delimited region is touched; a pre-existing
    user hook's other content is preserved verbatim, and re-running this
    (idempotent re-init) replaces the vectr block in place rather than
    duplicating it.

    Skips silently, with a disclosed stderr message (never an error), when
    `workspace` is not a git working tree, when a pre-existing foreign
    `post-commit` hook's interpreter is not POSIX-shell-compatible
    (`_shell_shebang_compatible`) — vectr's block would not run correctly
    appended under e.g. a Python or PowerShell hook — or when the resolved
    hooks directory is not writable (e.g. a stale/misconfigured repo-local
    `core.hooksPath` pointing at a path this user has no permission to
    create — `git rev-parse --git-path hooks` answers truthfully from git's
    own config, so an unwritable answer is an environment condition to
    report, not a vectr defect to raise past this function). This writer is
    the third of three independent hook installers `cmd_init --hooks` runs
    in sequence; a failure here must never take down the two that already
    succeeded.
    """
    hooks_dir = _git_hooks_dir(workspace)
    if hooks_dir is None:
        print(
            f"  Skipped git post-commit hook: {workspace} is not a git working tree.",
            file=sys.stderr,
        )
        return

    try:
        hook_path = hooks_dir / "post-commit"
        if not hook_path.exists():
            hooks_dir.mkdir(parents=True, exist_ok=True)
            hook_path.write_text(f"#!/bin/sh\n{_GIT_POST_COMMIT_BLOCK}", encoding="utf-8")
            hook_path.chmod(hook_path.stat().st_mode | 0o111)
            print(f"  Created {hook_path}", file=sys.stderr)
            return

        existing = hook_path.read_text(encoding="utf-8")
        if _GIT_HOOK_BLOCK_START not in existing:
            first_line = existing.splitlines()[0] if existing.splitlines() else ""
            if not _shell_shebang_compatible(first_line):
                print(
                    f"  Skipped {hook_path}: existing hook's interpreter is not a "
                    "POSIX shell vectr's post-commit block can run under. Add "
                    "`vectr hook post-commit </dev/null >/dev/null 2>&1 &` to it "
                    "by hand if you want commit provenance capture.",
                    file=sys.stderr,
                )
                return
            new_content = f"{existing.rstrip()}\n\n{_GIT_POST_COMMIT_BLOCK}"
        else:
            stripped = _GIT_HOOK_BLOCK_RE.sub("", existing).rstrip()
            new_content = f"{stripped}\n\n{_GIT_POST_COMMIT_BLOCK}" if stripped else f"#!/bin/sh\n{_GIT_POST_COMMIT_BLOCK}"
            if new_content == existing:
                return

        hook_path.write_text(new_content, encoding="utf-8")
        hook_path.chmod(hook_path.stat().st_mode | 0o111)
        print(f"  Updated vectr block in {hook_path}", file=sys.stderr)
    except OSError as exc:
        print(
            f"  Skipped git post-commit hook: could not write to {hooks_dir} "
            f"({exc}). This repo's `git config core.hooksPath` may point "
            "somewhere unwritable — fix that value directly, or install the "
            "hook by hand: `vectr hook post-commit </dev/null >/dev/null "
            "2>&1 &`.",
            file=sys.stderr,
        )


def _remove_git_post_commit_hook(workspace: str) -> None:
    """Strip vectr's block from the git post-commit hook (for
    --reset-config). Deletes the file if it becomes vectr-only-and-empty
    (mirrors `_remove_vectr_block`/`_remove_codex_mcp_config`); leaves a
    user's own pre-existing hook content untouched otherwise."""
    hooks_dir = _git_hooks_dir(workspace)
    if hooks_dir is None:
        return
    hook_path = hooks_dir / "post-commit"
    if not hook_path.exists():
        return
    content = hook_path.read_text(encoding="utf-8")
    if _GIT_HOOK_BLOCK_START not in content:
        return
    stripped = _GIT_HOOK_BLOCK_RE.sub("", content).rstrip()
    # A vectr-only file's only remaining content after stripping is its own
    # shebang line — nothing left worth keeping as an executable hook.
    if stripped and stripped.strip() != "#!/bin/sh":
        hook_path.write_text(stripped + "\n", encoding="utf-8")
        print(f"  Removed vectr block from {hook_path}", file=sys.stderr)
    else:
        hook_path.unlink()
        print(f"  Deleted {hook_path} (was vectr-only)", file=sys.stderr)


def _migrate_legacy_files() -> None:
    """Remove old single-instance PID/port files if they exist."""
    _LEGACY_PID_FILE.unlink(missing_ok=True)
    _LEGACY_PORT_FILE.unlink(missing_ok=True)


def _parse_code_workspace(path: str) -> list[str]:
    """Parse a .code-workspace file and return the absolute folder paths it lists."""
    ws_file = Path(path).resolve()
    data = json.loads(ws_file.read_text(encoding="utf-8"))
    ws_dir = ws_file.parent
    roots: list[str] = []
    for folder in data.get("folders", []):
        folder_path = folder.get("path", "")
        p = Path(folder_path)
        if not p.is_absolute():
            p = ws_dir / p
        roots.append(str(p.resolve()))
    return roots


def _resolve_workspace_roots(args: argparse.Namespace) -> list[str]:
    """Return ordered list of workspace roots from CLI args.

    Priority:
      1. Positional .code-workspace file  →  all folders listed in the file
      2. Positional directory             →  that single directory
      3. --path flags (one or more)       →  those directories in order
      4. VECTR_WORKSPACE env / default .  →  single directory
    """
    ws = getattr(args, "workspace", None)
    if ws:
        p = Path(ws)
        if str(ws).endswith(".code-workspace"):
            return _parse_code_workspace(ws)
        return [str(p.resolve())]
    paths = getattr(args, "paths", None) or []
    if paths:
        return [str(Path(p).resolve()) for p in paths]
    from integrations.workspace_detect import validate_workspace_env, WorkspaceEnvError

    raw = os.getenv("VECTR_WORKSPACE", ".")
    # UPG-WORKSPACE-ENV-VALIDATE: a typo'd VECTR_WORKSPACE must fail loudly
    # here rather than silently falling back to cwd detection below.
    try:
        validate_workspace_env(raw)
    except WorkspaceEnvError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    return [str(Path(raw).resolve())]


def _code_workspace_file_arg(args: argparse.Namespace) -> str | None:
    """The resolved `.code-workspace` file path the user gave `start`/`restart`,
    or None if they gave a plain directory / --path flags / nothing.

    Recorded in the InstanceRegistry so `vectr status` can show the file the
    instance was actually launched from instead of just its primary folder
    (UPG-CLI-STATUS-MODE).
    """
    ws = getattr(args, "workspace", None)
    if ws and str(ws).endswith(".code-workspace"):
        return str(Path(ws).resolve())
    return None


def _is_explicit_workspace(args: argparse.Namespace) -> bool:
    """True if the user gave an explicit workspace path to start/restart
    (positional `.code-workspace`/directory arg, or one or more --path
    flags), as opposed to defaulting to cwd/VECTR_WORKSPACE.

    UPG-WS-ROOT-MISDETECT: an explicit path must always win over the
    git-toplevel walk-up in `find_workspace_root` — that walk-up is only
    appropriate when the user gave vectr no path to go on at all.
    """
    return bool(getattr(args, "workspace", None)) or bool(getattr(args, "paths", None) or [])


def _warn_if_enclosing_repo(workspace: str) -> None:
    """One-line warning when an explicitly-given workspace path is not
    itself a git repo root but is nested inside an enclosing one
    (UPG-WS-ROOT-MISDETECT). Informational only: the enclosing repo is
    never indexed instead of the path the user gave.
    """
    from integrations.workspace_detect import find_workspace_root

    resolved = str(Path(workspace).resolve())
    enclosing = find_workspace_root(resolved)
    if enclosing != resolved:
        print(
            f"Note: {resolved} is not itself a git repo root; it is nested inside "
            f"the git repo at {enclosing}. Indexing only the given path, not the "
            f"enclosing repo.",
            file=sys.stderr,
        )


def _apply_exclude_args(workspace: str, exclude_entries: list[str]) -> None:
    """Append `--exclude` entries to .vectrignore (shared by `init` and `start`).

    Same repeatable append semantics either way — a directory name, a file
    glob, or a `re:<pattern>` path regex (UPG-EXCLUDE-REGEX). Every `re:`
    entry is validated with re.compile BEFORE anything is written: a bad
    pattern must exit non-zero with a clear message rather than land silently
    in .vectrignore (where the daemon would then have to warn and skip it).
    """
    if not exclude_entries:
        return
    from integrations.workspace_detect import write_vectrignore, VECTRIGNORE_REGEX_PREFIX

    for entry in exclude_entries:
        if entry.startswith(VECTRIGNORE_REGEX_PREFIX):
            pattern_text = entry[len(VECTRIGNORE_REGEX_PREFIX):]
            try:
                re.compile(pattern_text)
            except re.error as exc:
                print(f"Error: invalid regex in --exclude {entry!r}: {exc}", file=sys.stderr)
                sys.exit(1)

    write_vectrignore(workspace, exclude_entries)
    print(f"  Added to .vectrignore: {', '.join(exclude_entries)}", file=sys.stderr)


# Map from normalised language key to the Python module name of its grammar
# (used to derive the pip package name — module_name.replace("_", "-")).
# Only languages declared in SYMBOL_LANGUAGES are relevant; this covers all of them.
_GRAMMAR_MODULE: dict[str, str] = {
    "python":     "tree_sitter_python",
    "javascript": "tree_sitter_javascript",
    "typescript": "tree_sitter_typescript",
    "go":         "tree_sitter_go",
    "rust":       "tree_sitter_rust",
    "java":       "tree_sitter_java",
    "zig":        "tree_sitter_zig",
    "c":          "tree_sitter_c",
    "cpp":        "tree_sitter_cpp",
}


def _preflight_install_grammar(
    lang: str,
    pinned_req: str,
    *,
    _run_pip=None,
) -> bool:
    """Attempt to install one missing grammar package. Returns True on success.

    `pinned_req` is the requirement string (e.g. 'tree-sitter-c==0.24.2').
    `_run_pip` is injectable for tests (defaults to subprocess.run on sys.executable).
    """
    if _run_pip is None:
        def _run_pip(req: str):
            return subprocess.run(
                [sys.executable, "-m", "pip", "install", req],
                capture_output=True, text=True,
            )

    result = _run_pip(pinned_req)
    if result.returncode == 0:
        # Re-verify: clear parser cache so the re-import picks up the newly installed module.
        try:
            from agent.indexer._chunking import _PARSER_CACHE
            _PARSER_CACHE.clear()
        except Exception:
            pass
        from agent.symbol_graph import grammar_available
        if grammar_available(lang):
            print(f"  [vectr] grammar installed: {pinned_req}", file=sys.stderr)
            return True
    return False


def _preflight_grammars(*, _run_pip=None) -> None:
    """Check declared tree-sitter grammars and auto-install any that are missing.

    Called before starting the daemon (cmd_start) and before indexing (cmd_watch)
    so a missing grammar is repaired before the symbol graph is first built.

    Strategy:
    1. Compute missing = SYMBOL_LANGUAGES - available_symbol_languages().
    2. For each missing language, derive the pinned requirement from the installed
       vectr package metadata (importlib.metadata.requires("vectr")). Falls back
       to an unpinned package name if metadata lookup fails.
    3. Attempt pip install. On success, re-verify via grammar_available().
    4. On failure (offline / externally-managed env / permissions): print a clear
       remediation message and CONTINUE — the language is search-only for this run.
       Never crash, never silently add --break-system-packages.
    """
    from agent.symbol_graph import SYMBOL_LANGUAGES, available_symbol_languages

    missing_langs = sorted(SYMBOL_LANGUAGES - available_symbol_languages())
    if not missing_langs:
        return  # all grammars present — nothing to do

    # Derive pinned requirements from installed package metadata.
    pinned: dict[str, str] = {}
    try:
        import importlib.metadata as _meta
        reqs = _meta.requires("vectr") or []
        for req in reqs:
            # req looks like 'tree-sitter-c>=0.24.2' or 'tree-sitter-c==0.24.2'
            # We want the whole string (including the version constraint).
            req_name = req.split(";")[0].strip()  # strip environment markers
            if req_name.lower().startswith("tree-sitter-"):
                # Normalise: tree-sitter-cpp -> tree_sitter_cpp
                module = req_name.split(">=")[0].split("==")[0].split("!=")[0].strip()
                module_key = module.replace("-", "_").lower()
                pinned[module_key] = req_name  # e.g. "tree_sitter_c" -> "tree-sitter-c>=0.24.2"
    except Exception:
        pass  # metadata unavailable — will fall back to bare package names below

    failed: list[str] = []
    for lang in missing_langs:
        module_name = _GRAMMAR_MODULE.get(lang, f"tree_sitter_{lang}")
        pip_name = module_name.replace("_", "-")
        # Use pinned requirement if found, otherwise bare package name.
        req = pinned.get(module_name, pip_name)

        print(
            f"  [vectr] tree-sitter grammar missing for '{lang}' — installing {req!r} ...",
            file=sys.stderr,
        )
        success = _preflight_install_grammar(lang, req, _run_pip=_run_pip)
        if not success:
            failed.append((lang, req))

    if failed:
        lang_list = ", ".join(l for l, _ in failed)
        pip_cmd = " ".join(r for _, r in failed)
        print(
            f"\n[vectr] WARNING: could not auto-install grammar(s) for: {lang_list}",
            file=sys.stderr,
        )
        print(
            f"  locate/trace will be DISABLED for these languages in this session.",
            file=sys.stderr,
        )
        print(
            f"  To fix, run:  pip install {pip_cmd}",
            file=sys.stderr,
        )
        print(
            "  Note: externally-managed environments (Homebrew/system Python) may need "
            "--break-system-packages or use a virtualenv.",
            file=sys.stderr,
        )
        print(
            "  Continuing startup — affected languages are search-only.\n",
            file=sys.stderr,
        )


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "0:0:0:0:0:0:0:1"})


def _is_loopback_host(host: str) -> bool:
    """True when `host` is a loopback address the daemon binds to by default —
    reachable only from the same machine, so no authentication is required."""
    return host.strip().lower() in _LOOPBACK_HOSTS


def _enforce_bind_auth(host: str) -> None:
    """Refuse to bind beyond loopback without authentication.

    Binding the full codebase index + shared working memory to a network
    interface with no key would serve them to anyone who can reach the port.
    A non-loopback bind (team / central-instance mode) therefore requires
    VECTR_API_KEY to be set. Loopback binds are unaffected.
    """
    if _is_loopback_host(host) or os.getenv("VECTR_API_KEY", ""):
        return
    print(
        f"Error: refusing to bind to {host} without authentication.\n"
        "  Binding beyond localhost exposes the index and working memory to the network.\n"
        "  Set a shared key first:\n"
        "    export VECTR_API_KEY=$(vectr key)\n"
        "  then re-run. To keep vectr local-only, omit --host (defaults to 127.0.0.1).",
        file=sys.stderr,
    )
    sys.exit(1)


def _mode_name(memory_only: bool, search_only: bool) -> str:
    """The daemon's operating mode as one label, in the same vocabulary the
    daemon itself reports in `/v1/status` ("full" | "memory-only" |
    "search-only") — one spelling shared by the registry record, the CLI's
    mode lines, and the staleness banner's remediation command."""
    if memory_only:
        return MODE_MEMORY_ONLY
    if search_only:
        return MODE_SEARCH_ONLY
    return MODE_FULL


def _effective_mode_flags(memory_only: bool, search_only: bool) -> tuple[bool, bool]:
    """`(memory_only, search_only)` as `VectrService` will resolve them —
    the given flags OR'd with the VECTR_MEMORY_ONLY / VECTR_SEARCH_ONLY
    environment variables (app/service.py does exactly this at construction).
    Both-set is rejected here for the same reason the service rejects it."""
    memory_only = memory_only or os.getenv("VECTR_MEMORY_ONLY", "") == "1"
    search_only = search_only or os.getenv("VECTR_SEARCH_ONLY", "") == "1"
    if memory_only and search_only:
        raise ValueError("Cannot start vectr in both --memory-only and --search-only mode simultaneously")
    return memory_only, search_only


def _do_start(
    workspace: str,
    port: int,
    ws_hash: str,
    extra_roots: list[str] | None = None,
    memory_only: bool = False,
    search_only: bool = False,
    workspace_explicit: bool = False,
    code_workspace_file: str | None = None,
    host: str = "127.0.0.1",
    no_ide_config: bool = False,
    json_output: bool = False,
    foreground_fast: bool = False,
) -> None:
    if memory_only and search_only:
        raise ValueError("Cannot start vectr in both --memory-only and --search-only mode simultaneously")

    # UPG-RESTART-DROPS-MODE: resolve the mode the daemon will ACTUALLY run in
    # before anything records or reports it. `VectrService.__init__` ORs the
    # constructor flags with VECTR_MEMORY_ONLY/VECTR_SEARCH_ONLY (which this
    # process passes through in `env` below), so a daemon launched with the
    # env var and no flag runs memory-only while the flag says otherwise —
    # resolving it the same way here keeps the registry record, the printed
    # mode line, and the daemon's own `/v1/status` mode in agreement.
    memory_only, search_only = _effective_mode_flags(memory_only, search_only)

    from agent.fs_permissions import secure_dir
    secure_dir(Path.home() / ".vectr")
    log_dir = secure_dir(Path.home() / ".vectr" / "logs")
    log_path = log_dir / f"{ws_hash}.log"

    env = {
        **os.environ,
        "VECTR_WORKSPACE": workspace,
        "VECTR_PORT": str(port),
        "VECTR_EXTRA_ROOTS": json.dumps(extra_roots or []),
        # UPG-PROXY-LOOPBACK-BYPASS: the daemon process has no other way to
        # learn the host it was actually bound to (the `--host` argument
        # below is only visible to this launcher process) — plumb it through
        # so in-process gates (agent/proactive/settings.py's unconditional
        # bind check) can refuse to serve proactive injection over a
        # non-loopback bind regardless of config toggles or client-supplied
        # fields. Absence of this var (e.g. a daemon started some other way,
        # or in-process/test usage) is treated as this function's own
        # documented default bind, 127.0.0.1 — never as non-loopback.
        "VECTR_BIND_HOST": host,
    }
    if memory_only:
        env["VECTR_MEMORY_ONLY"] = "1"
    if search_only:
        env["VECTR_SEARCH_ONLY"] = "1"
    if workspace_explicit:
        # UPG-WS-ROOT-MISDETECT: tells VectrService to trust `workspace`
        # verbatim instead of walking up to the nearest enclosing .git root.
        env["VECTR_WORKSPACE_EXPLICIT"] = "1"
    if no_ide_config:
        # UPG-CLI-WRITES-DISCLOSURE follow-through: `--no-ide-config` skipped
        # this process's own 7-file `_write_workspace_config` write, but the
        # daemon subprocess's `VectrService.__init__` used to call
        # `configure_all()` unconditionally regardless — silently writing
        # .cursor/mcp.json and .claude/settings.json anyway. Propagate the
        # choice across the subprocess boundary so the opt-out actually holds.
        env["VECTR_CONFIGURE_IDE"] = "0"
    if foreground_fast:
        # UPG-INDEX-RESOURCE-GOVERNOR: restores pre-governor behaviour
        # exactly — read by agent/indexer/_core.py's _foreground_fast_enabled()
        # to skip both the OS priority clamp and the duty-cycle pacing.
        env["VECTR_FOREGROUND_FAST"] = "1"
    vectr_dir = Path(__file__).resolve().parent
    with open(log_path, "a") as log_file:
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "api:app", "--host", host, "--port", str(port)],
            env=env,
            cwd=str(vectr_dir),
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=log_file,
        )

    _migrate_legacy_files()
    InstanceRegistry().register(
        ws_hash, workspace, port, proc.pid,
        extra_roots=extra_roots, code_workspace_file=code_workspace_file,
        # UPG-RESTART-DROPS-MODE: record the launch configuration `restart`
        # must reproduce, alongside the port it already reuses.
        mode=_mode_name(memory_only, search_only), host=host,
    )
    mode_tag = " [memory-only]" if memory_only else (" [search-only]" if search_only else "")

    # UPG-CLI-START-READY-RACE: block until the daemon is actually reachable
    # (or the readiness window elapses) before printing a success message.
    # Before this fix, `start` printed success right after Popen() returned —
    # only proof a process was spawned, not that it had bound its port or
    # finished loading the embedder in its FastAPI lifespan. Running
    # `vectr status` right after that "success" line could still see a
    # connection refused, reading as start having lied.
    ready = _wait_for_daemon_ready(port, proc.pid)
    alive = ready or _is_pid_alive(proc.pid)
    status = "ready" if ready else ("not_ready" if alive else "failed")

    if json_output:
        # UPG-CLI-STOP-NO-PORT-FLAG companion: a single deterministic line on
        # stdout so scripts/harnesses can discover the actually-bound port
        # (find_free_port may have walked past a busy --port) without
        # scraping the human stderr text. Unlike the text branch below,
        # "failed" exits non-zero here — this surface is opt-in specifically
        # for callers that check results programmatically.
        print(json.dumps({
            "status": status,
            "port": port,
            "pid": proc.pid,
            "workspace": workspace,
            "extra_roots": extra_roots or [],
            "mode": _mode_name(memory_only, search_only),
            "mcp_url": f"http://{'localhost' if _is_loopback_host(host) else host}:{port}/mcp",
            "host": host,
            "logs": str(log_path),
        }))
        if status == "failed":
            sys.exit(1)
        return

    if ready:
        print(f"Vectr started{mode_tag} (PID {proc.pid}) on port {port}", file=sys.stderr)
        print(f"Workspace : {workspace}", file=sys.stderr)
        if extra_roots:
            for r in extra_roots:
                print(f"          + {r}", file=sys.stderr)
        _url_host = "localhost" if _is_loopback_host(host) else host
        print(f"MCP URL   : http://{_url_host}:{port}/mcp", file=sys.stderr)
        if not _is_loopback_host(host):
            print(
                f"Bind      : {host} (authenticated). Clients connect with: "
                f"vectr connect --url http://{host}:{port} --api-key <key>",
                file=sys.stderr,
            )
        print(f"Logs      : {log_path}", file=sys.stderr)
        if memory_only:
            print(
                f"Mode      : memory-only (no code indexing/watcher; memory tools + hooks active)",
                file=sys.stderr,
            )
        elif search_only:
            print(
                f"Mode      : search-only (no working-memory layer; search/locate/trace/map active)",
                file=sys.stderr,
            )
        else:
            print(f"Check indexing progress: vectr status --path {workspace}", file=sys.stderr)
    elif alive:
        print(
            f"Vectr started{mode_tag} (PID {proc.pid}) on port {port}, but has not "
            f"started responding yet after {CLI_START_READY_POLL_TIMEOUT_S:.0f}s.",
            file=sys.stderr,
        )
        print(
            "It may still be loading (large workspace / first-run embedding-model "
            "download) — this is not necessarily a failure.",
            file=sys.stderr,
        )
        print(f"Logs      : {log_path}", file=sys.stderr)
        print(f"Poll readiness with: vectr status --path {workspace}", file=sys.stderr)
    else:
        print(
            f"Vectr failed to start: the process exited before becoming ready.",
            file=sys.stderr,
        )
        print(f"Logs      : {log_path}", file=sys.stderr)


def _get_port_for_workspace(workspace: str, fallback: int) -> int:
    """Resolve the port serving `workspace` for every `--path`-accepting
    query subcommand (index/search/fetch/remember/recall/resume/status/
    forget/proxy).

    UPG-CLI-PATH-IGNORED: this used to do an EXACT `workspace_hash` lookup
    only, so `--path <subdir-of-a-registered-workspace>` (or any workspace
    the caller resolves slightly differently than the one it was started
    with) never matched and silently fell through to `fallback` — usually
    the default port, which commonly has some OTHER, unrelated instance
    listening (e.g. an always-on daemon for a different project). The CLI
    then happily reported on that wrong instance with no hint the requested
    `--path` was ignored. Reuses `_resolve_hook_instance`'s walk-up-parents
    registry resolution (the same one hook dispatch relies on for exactly
    this reason) instead of reimplementing it, so a subagent whose cwd is a
    subdirectory of a registered workspace reaches that workspace's own
    daemon here too.
    """
    entry = _resolve_hook_instance(workspace)
    return entry["port"] if entry is not None else fallback


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_key(args: argparse.Namespace) -> None:
    """Print a fresh high-entropy API key for authenticated / team deployments.

    The key goes to stdout (so `KEY=$(vectr key)` works); usage guidance goes
    to stderr. Vectr never persists the key — the operator sets it in the
    server's environment (VECTR_API_KEY) and shares it with clients out of band.
    """
    import secrets
    key = secrets.token_urlsafe(32)
    # A leading '-' makes the key parse as a flag in `--api-key <key>` and
    # most other CLI contexts; regenerate until the first character is safe.
    while key.startswith("-"):
        key = secrets.token_urlsafe(32)
    print(key)
    print(
        "\nShared key for authenticated / team (central instance) deployments:\n"
        "  server:  VECTR_API_KEY=<key> vectr start --host 0.0.0.0 --path /srv/repo\n"
        "  client:  vectr connect --url http://<server-host>:<port> --api-key=<key>\n"
        "Store it in each client's environment or MCP config; vectr never persists it.\n"
        "Configs that embed the key (.mcp.json, .cursor/mcp.json, .vscode/mcp.json)\n"
        "hold it in plaintext — treat them as secrets and keep them out of shared or\n"
        "public version control.",
        file=sys.stderr,
    )


def cmd_connect(args: argparse.Namespace) -> None:
    """Configure the local editor to use a REMOTE vectr instance (team mode).

    Writes the editor MCP configs + guidance pointing at a central daemon over
    the network, with the shared API key (and optional client label) as request
    headers. Spawns no local daemon. This is how a developer joins a shared
    working-memory + index instance served by one central host.
    """
    workspace = str(Path(getattr(args, "path", ".") or ".").resolve())
    url = _normalize_mcp_url(args.url)
    api_key = getattr(args, "api_key", "") or os.getenv("VECTR_API_KEY", "")
    label = getattr(args, "label", "") or ""

    if url.startswith("http://") and not _is_loopback_host(_host_from_url(url)) and not api_key:
        print(
            "Warning: connecting to a non-local instance over plain HTTP without a key.\n"
            "  If the server requires authentication, pass --api-key or set VECTR_API_KEY.\n"
            "  Terminate TLS at a reverse proxy for network transport.",
            file=sys.stderr,
        )

    headers = _mcp_auth_headers(api_key=api_key, client_label=label)
    managed = _write_remote_workspace_config(workspace, url, headers)

    print(f"Configured this workspace to use the remote vectr instance at {url}", file=sys.stderr)
    print(f"  Workspace : {workspace}", file=sys.stderr)
    # UPG-TEAM-4: print the exact list of files connect writes/manages so the
    # configuration is never a surprise, flagging the ones that embed the key.
    print("  Files     : written/updated in this workspace —", file=sys.stderr)
    for rel in managed:
        marker = "   [holds the API key in plaintext]" if (api_key and rel in _REMOTE_KEY_BEARING_FILES) else ""
        print(f"                - {rel}{marker}", file=sys.stderr)
    if api_key:
        print("  Auth      : X-Api-Key header written to the editor MCP configs", file=sys.stderr)
        print(
            "              (.mcp.json, .cursor/mcp.json, .vscode/mcp.json now hold the key\n"
            "              in plaintext — treat them as secrets; keep them out of shared or\n"
            "              public version control)",
            file=sys.stderr,
        )
    if label:
        print(f"  Client    : notes/audit will be attributed to '{label}'", file=sys.stderr)
    print(
        "  Note      : the remote instance indexes its own checkout — search/locate "
        "results reference the server's files/lines, which may differ from your local tree.",
        file=sys.stderr,
    )


def cmd_start(args: argparse.Namespace) -> None:
    memory_only = getattr(args, "memory_only", False)
    search_only = getattr(args, "search_only", False)
    if memory_only and search_only:
        print("Error: --memory-only and --search-only are mutually exclusive.", file=sys.stderr)
        sys.exit(1)

    host = getattr(args, "host", "127.0.0.1") or "127.0.0.1"
    # Refuse a non-loopback bind without a shared key BEFORE doing any work.
    _enforce_bind_auth(host)

    roots = _resolve_workspace_roots(args)
    workspace = roots[0]
    extra_roots = roots[1:]
    ws_hash = workspace_hash(workspace)
    preferred_port = args.port
    explicit = _is_explicit_workspace(args)
    if explicit:
        _warn_if_enclosing_repo(workspace)

    # write user-defined exclusions to .vectrignore before indexing starts
    _apply_exclude_args(workspace, getattr(args, "exclude", None) or [])

    json_output = getattr(args, "json_output", False)

    registry = InstanceRegistry()
    registry.prune_dead()

    entry = registry.get(ws_hash)
    if entry is not None and _is_pid_alive(entry["pid"]):
        port = entry["port"]
        for root in roots:
            _maybe_write_workspace_config(root, port, args, search_only=search_only)
        if json_output:
            print(json.dumps({
                "status": "already_running",
                "port": port,
                "pid": entry["pid"],
                "workspace": workspace,
                "extra_roots": extra_roots,
                "mode": entry.get("mode", _mode_name(memory_only, search_only)),
                "mcp_url": f"http://localhost:{port}/mcp",
                "host": entry.get("host", "127.0.0.1"),
            }))
        else:
            print("Vectr is already running for this workspace.", file=sys.stderr)
            print(f"  Workspace : {workspace}", file=sys.stderr)
            print(f"  Port      : {port}", file=sys.stderr)
            print(f"  MCP URL   : http://localhost:{port}/mcp", file=sys.stderr)
        for root in roots:
            _warn_on_stale_mcp_configs(root, port)
        return

    # UPG-CLI-STOP-NO-PORT-FLAG companion: --strict-port asks find_free_port
    # to fail loudly instead of silently walking past a busy --port, for
    # scripts/harnesses that need the daemon on an exact, predictable port.
    try:
        port = registry.find_free_port(ws_hash, preferred_port, strict=getattr(args, "strict_port", False))
    except PortBusyError as e:
        if json_output:
            print(json.dumps({"status": "failed", "error": str(e)}))
        else:
            print(f"Vectr failed to start: {e}", file=sys.stderr)
        sys.exit(1)
    for root in roots:
        _maybe_write_workspace_config(root, port, args, search_only=search_only)
    if not memory_only:
        _preflight_grammars()
    _do_start(
        workspace, port, ws_hash, extra_roots=extra_roots,
        memory_only=memory_only, search_only=search_only, workspace_explicit=explicit,
        code_workspace_file=_code_workspace_file_arg(args), host=host,
        no_ide_config=getattr(args, "no_ide_config", False),
        json_output=json_output,
        foreground_fast=getattr(args, "foreground_fast", False),
    )
    for root in roots:
        _warn_on_stale_mcp_configs(root, port)


def cmd_proxy(args: argparse.Namespace) -> None:
    """Run the experimental localhost Anthropic-shaped proxy (UPG-PRO-16).

    Point the agent harness at it with
    `ANTHROPIC_BASE_URL=http://127.0.0.1:<port>`. It forwards to the real API
    and — when injection is enabled and the workspace daemon is running — appends
    deterministic proactive context after the last prompt-cache breakpoint. The
    listener is localhost-only and refuses any non-loopback bind (Proactive
    context is a solo/localhost-only feature, mutually exclusive with team mode).
    """
    import dataclasses

    from agent.proactive.settings import ProactiveSettings, derive_provider_timeout_s
    from agent.proactive.proxy import build_proxy_app
    from agent.proactive.provider import DaemonInjectionProvider
    from agent.proactive.cache import ResponseCache

    settings = ProactiveSettings.from_env()
    host = getattr(args, "host", None) or settings.proxy_host
    port = getattr(args, "port", None) or settings.proxy_port
    upstream = getattr(args, "upstream", None) or settings.proxy_upstream_base_url
    inject = settings.proxy_inject and not getattr(args, "no_inject", False)
    settings = dataclasses.replace(
        settings, proxy_host=host, proxy_port=port,
        proxy_upstream_base_url=upstream, proxy_inject=inject,
    )

    # Localhost-only listener (design §9/§10). A non-loopback bind is refused:
    # the proxy sees the full conversation, so it is a solo/localhost-only
    # feature and is mutually exclusive with team / shared-instance mode.
    if not _is_loopback_host(host):
        print(
            f"Error: the vectr proxy binds localhost only; refusing to bind {host}.\n"
            f"  Proactive context reads the conversation and is a solo-only feature,\n"
            f"  mutually exclusive with team mode. Use --host 127.0.0.1 (the default).",
            file=sys.stderr,
        )
        sys.exit(1)

    workspace = str(Path(getattr(args, "path", ".") or ".").resolve())
    daemon_port = getattr(args, "daemon_port", None) or _get_port_for_workspace(workspace, DEFAULT_PORT)
    api_key = os.getenv("VECTR_API_KEY") or None

    provider = None
    if inject:
        provider = DaemonInjectionProvider(
            _api_base(daemon_port),
            # Strictly below proxy_inject_budget_ms (see derive_provider_timeout_s):
            # the provider's own timeout must trip before the proxy's outer
            # wait_for backstop, so a slow daemon resolves to a clean, logged
            # "nothing to inject" rather than an abrupt mid-flight cancellation.
            timeout_s=derive_provider_timeout_s(settings),
            api_key=api_key,
        )
    response_cache = None
    if settings.response_cache_enabled:
        response_cache = ResponseCache(
            max_entries=settings.response_cache_max_entries,
            ttl_seconds=settings.response_cache_ttl_seconds,
        )

    app = build_proxy_app(
        settings, injection_provider=provider, response_cache=response_cache
    )

    base = f"http://{host}:{port}"
    print("Vectr proxy (experimental — Proactive context) starting.", file=sys.stderr)
    print(f"  Listening : {base}", file=sys.stderr)
    print(f"  Upstream  : {upstream}", file=sys.stderr)
    daemon_status = None
    if inject:
        import httpx as _httpx
        # Two attempts with a generous timeout: the first /v1/status call on a
        # freshly started daemon computes language stats and can exceed a short
        # timeout, which would falsely print the unreachable warning.
        for _ in range(2):
            try:
                daemon_status = _httpx.get(f"{_api_base(daemon_port)}/v1/status", timeout=5.0).json()
                break
            except Exception:
                daemon_status = None
    for _line in _render_injection_lines(inject, _api_base(daemon_port), daemon_status):
        print(_line, file=sys.stderr)
    print(f"  Resp cache: {'on' if response_cache is not None else 'off'}", file=sys.stderr)
    print("  Wire it up:", file=sys.stderr)
    print(f"    export ANTHROPIC_BASE_URL={base}", file=sys.stderr)
    print("  Bypass at any time by unsetting it:", file=sys.stderr)
    print("    unset ANTHROPIC_BASE_URL", file=sys.stderr)
    print("  Caveats on a non-first-party base URL (documented upstream):", file=sys.stderr)
    print("    - MCP tool search is disabled unless ENABLE_TOOL_SEARCH=true and the proxy forwards tool_reference blocks.", file=sys.stderr)
    print("    - Remote Control is disabled on a non-api.anthropic.com base URL.", file=sys.stderr)
    print("  The upstream API key is forwarded untouched and never stored or logged.", file=sys.stderr)

    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="warning")


def cmd_index(args: argparse.Namespace) -> None:
    import httpx

    workspace = str(Path(args.path).resolve())
    port = _get_port_for_workspace(workspace, args.port)
    _check_version_skew(port, registry_entry=_resolve_hook_instance(workspace))
    try:
        resp = httpx.post(
            f"{_api_base(port)}/v1/index",
            json={"path": workspace, "force": args.force},
            timeout=600,
        )
        resp.raise_for_status()
        data = resp.json()
        # UPG-CLI-SMALL-UX: human-readable text, matching every sibling
        # subcommand, instead of a raw json.dumps of IndexResponse. `model`
        # (the embedding model name) is dropped here — it answers "which
        # model did this?", not "how did indexing go?"; still present on
        # the REST response for callers that do want it.
        print(f"Indexed {data['indexed_files']} files, {data['total_chunks']} chunks "
              f"in {data['processing_ms']}ms.")
    except (httpx.ConnectError, httpx.HTTPStatusError) as exc:
        _handle_daemon_call_error(exc, port)


def cmd_search(args: argparse.Namespace) -> None:
    import httpx

    workspace = str(Path(os.getenv("VECTR_WORKSPACE", ".")).resolve())
    port = _get_port_for_workspace(workspace, args.port)
    _check_version_skew(port, registry_entry=_resolve_hook_instance(workspace))
    payload: dict = {"query": args.query, "n_results": args.n}
    if args.language:
        payload["language"] = args.language
    try:
        resp = httpx.post(f"{_api_base(port)}/v1/search", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            print("No results found.", file=sys.stderr)
            return
        # UPG-CLI-SEARCH-FLOOR: `low_confidence` is an EXISTING signal — the
        # MCP surface has rendered it as a banner since UPG-NOTFOUND-FLOOR;
        # the CLI silently dropped it, so a nonsense query printed 10
        # formatted results indistinguishable from a real hit. Render the
        # same signal here, in CLI-appropriate wording (no `vectr_locate`
        # reference — no such subcommand exists).
        if data.get("low_confidence"):
            from agent.config import NOTFOUND_FLOOR_BANNER_CLI
            print(f"\n--- Low confidence ---\n{NOTFOUND_FLOOR_BANNER_CLI}", file=sys.stderr)
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] {r['file']}  lines {r['lines']}  score {r['score']:.3f}")
            if r["symbol"]:
                print(f"    {r['symbol']}  ({r['language']})")
            print()
            print(r["content"][:1000])
        print(f"\n— {data['query_time_ms']}ms  {data['chunks_searched']} chunks searched", file=sys.stderr)
    except (httpx.ConnectError, httpx.HTTPStatusError) as exc:
        _handle_daemon_call_error(exc, port)


def cmd_fetch(args: argparse.Namespace) -> None:
    """Re-fetch one or more chunks by exact id, verbatim (UPG-CTX-EVICT).

    Shell path to the same deterministic re-fetch surface as the MCP
    `vectr_fetch` tool and `POST /v1/fetch` — restores a chunk shown in an
    earlier `vectr search`/`vectr_locate`/`vectr_trace` result after it has
    left context, with no re-search or file re-read.
    """
    import httpx
    from integrations.mcp_server._dispatch import _storage_cap_truncation_warning

    workspace = str(Path(os.getenv("VECTR_WORKSPACE", ".")).resolve())
    port = _get_port_for_workspace(workspace, args.port)
    _check_version_skew(port, registry_entry=_resolve_hook_instance(workspace))
    try:
        resp = httpx.post(f"{_api_base(port)}/v1/fetch", json={"ids": args.ids}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        for entry in data.get("results", []):
            if not entry.get("found"):
                print(f"\n[{entry['id']}] not found", file=sys.stderr)
                continue
            print(f"\n[{entry['id']}]  {entry['file_path']}  lines {entry['lines']}")
            if entry.get("symbol"):
                print(f"    {entry['symbol']}  ({entry['language']})")
            print()
            print(entry["content"])
            # UPG-FETCH-TRUNCATION-SILENT: `lines` carries the symbol's full
            # recorded span even when the stored content was capped shorter
            # at index time — same detection MCP's vectr_fetch applies.
            start_s, _, end_s = entry.get("lines", "").partition("-")
            try:
                start_line, end_line = int(start_s), int(end_s)
            except ValueError:
                start_line = end_line = 0
            warning = _storage_cap_truncation_warning(
                entry["content"], entry.get("file_path", ""), start_line, end_line,
            )
            if warning:
                print(warning)
        if data.get("note"):
            print(f"\n{data['note']}", file=sys.stderr)
    except (httpx.ConnectError, httpx.HTTPStatusError) as exc:
        _handle_daemon_call_error(exc, port)


def cmd_remember(args: argparse.Namespace) -> None:
    """Store a working-memory note via the workspace daemon (UPG-9.1).

    Gives `command`-type hooks (and humans) a shell path to the note store,
    mirroring the MCP `vectr_remember` tool.
    """
    import httpx

    workspace = str(Path(args.path).resolve())
    port = _get_port_for_workspace(workspace, args.port)
    _check_version_skew(port, registry_entry=_resolve_hook_instance(workspace))
    payload: dict = {"content": args.content, "priority": args.priority}
    if getattr(args, "kind", None):
        payload["kind"] = args.kind
    if args.tags:
        payload["tags"] = args.tags
    if getattr(args, "title", None):
        payload["title"] = args.title
    if getattr(args, "agent", None):
        payload["agent"] = args.agent
    try:
        resp = httpx.post(f"{_api_base(port)}/v1/remember", json=payload, timeout=30)
        resp.raise_for_status()
        print(resp.json().get("message", "Stored note."))
    except (httpx.ConnectError, httpx.HTTPStatusError) as exc:
        _handle_daemon_call_error(exc, port)


def cmd_recall(args: argparse.Namespace) -> None:
    """Print recalled working-memory notes to stdout (UPG-9.1).

    This is the human-facing subcommand — `vectr recall` at a shell prompt or
    in a script — and follows the same error contract as every sibling
    subcommand (daemon down or declining the request -> stderr message,
    exit 1). It is NOT the hook path: hook entries written by
    `_write_claude_hooks` invoke `vectr hook <event>` (see cmd_hook), which
    resolves its own daemon connection via `_fetch_recall` and never raises
    regardless of daemon state — that resilience lives entirely in cmd_hook,
    not here, so a down daemon can never look like a successful recall to a
    script that shells out to this command (UPG-CLI-RECALL-EXITCODE).
    """
    import httpx

    workspace = str(Path(args.path).resolve())
    port = _get_port_for_workspace(workspace, args.port)
    _check_version_skew(port, registry_entry=_resolve_hook_instance(workspace))
    payload: dict = {"limit": args.limit}
    if getattr(args, "boot", False):
        # Boot mode ignores all filters server-side; send only the flag.
        payload = {"boot": True}
    else:
        if getattr(args, "note_id", None) is not None:
            # Single-note expand path — send only the note_id, detail=full.
            payload = {"note_id": args.note_id, "detail": "full"}
        else:
            if args.query:
                payload["query"] = args.query
            if args.tags:
                payload["tags"] = args.tags
            if args.priority:
                payload["priority"] = args.priority
            if getattr(args, "kind", None):
                payload["kind"] = args.kind
            if getattr(args, "min_similarity", None) is not None:
                payload["min_similarity"] = args.min_similarity
            if getattr(args, "max_age_days", None) is not None:
                payload["max_age_days"] = args.max_age_days
            if getattr(args, "sort_by", None):
                payload["sort_by"] = args.sort_by
            detail = getattr(args, "detail", "index") or "index"
            payload["detail"] = detail
    # UPG-CLI-RECALL-HINT: this is the human terminal surface — the response's
    # expand hint must be the real `vectr recall --id N` shell form, not the
    # MCP tool-call form the daemon renders by default.
    payload["surface"] = "cli"
    try:
        resp = httpx.post(f"{_api_base(port)}/v1/recall", json=payload, timeout=30)
        resp.raise_for_status()
        notes = resp.json().get("notes", "")
        if notes:
            print(notes)
    except (httpx.ConnectError, httpx.HTTPStatusError) as exc:
        _handle_daemon_call_error(exc, port)


def cmd_resume(args: argparse.Namespace) -> None:
    """Print 'pick up where you left off' to stdout (UPG-RESUME-SURFACE).

    One named subcommand over the same selection SessionStart boot injection
    and `vectr_resume` use — the most recent current-task note, the latest
    saved snapshot, and open gotchas. GET, no body: reads existing state only.
    """
    import httpx

    workspace = str(Path(args.path).resolve())
    port = _get_port_for_workspace(workspace, args.port)
    _check_version_skew(port, registry_entry=_resolve_hook_instance(workspace))
    try:
        resp = httpx.get(f"{_api_base(port)}/v1/resume", timeout=30)
        resp.raise_for_status()
        formatted = resp.json().get("formatted", "")
        if formatted:
            print(formatted)
    except (httpx.ConnectError, httpx.HTTPStatusError) as exc:
        _handle_daemon_call_error(exc, port)


def _fetch_recall(port: int, payload: dict) -> str:
    """POST /v1/recall and return the notes text, or '' on ANY failure.

    Never raises — this feeds harness-injected hook context and must not break
    the session if the daemon is down, slow, or returns an error.
    """
    import httpx
    try:
        resp = httpx.post(f"{_api_base(port)}/v1/recall", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json().get("notes", "") or ""
    except Exception:
        return ""


def _fetch_boundary_precompact(port: int) -> str:
    """GET /v1/boundary/precompact and return its `text` field, or '' on ANY
    failure (daemon absent, connection refused, non-2xx status such as the
    search-only-mode 503, malformed JSON) — never raises.

    Used by the PreCompact hook branch to fetch the boundary-preservation
    text; the branch prints whatever this returns as PLAIN TEXT (never the
    `hookSpecificOutput` JSON envelope other hook events use), matching the
    editor harness's PreCompact contract: its stdout is appended verbatim as
    custom compact instructions, not parsed as JSON.
    """
    import httpx
    try:
        resp = httpx.get(f"{_api_base(port)}/v1/boundary/precompact", timeout=30)
        resp.raise_for_status()
        return resp.json().get("text", "") or ""
    except Exception:
        return ""


def _post_snapshot(port: int, label: str) -> bool:
    """POST /v1/snapshot; True on success, False on any failure (never raises).

    Used by the PreCompact hook to seal working memory before context is
    replaced — a snapshot failure must never block compaction.
    """
    import httpx
    try:
        resp = httpx.post(f"{_api_base(port)}/v1/snapshot", json={"label": label}, timeout=30)
        resp.raise_for_status()
        return True
    except Exception:
        return False


def _post_trigger_reset(port: int, session_id: str) -> bool:
    """POST /v1/trigger/reset; True on success, False on any failure (never
    raises).

    Used by the PreCompact hook (TRIGGER-ENGINE wave 2a,
    bm2-design-skeleton.md §3: "cleared on compaction") to clear this
    session's per-session fire ledger and cumulative injection budget —
    a reset failure must never block compaction, mirroring `_post_snapshot`
    immediately above.
    """
    import httpx
    try:
        resp = httpx.post(
            f"{_api_base(port)}/v1/trigger/reset", json={"session_id": session_id}, timeout=30,
        )
        resp.raise_for_status()
        return True
    except Exception:
        return False


def _git_fact(cwd: str, *args: str) -> str:
    """Run one read-only `git` subprocess call from `cwd` and return its
    trimmed stdout, or "" on ANY failure (missing git, not a repo, non-zero
    exit, timeout) — never raises (UPG-COMMIT-MEMORY-HOOK). Used by the
    post-commit hook branch to gather the sha/subject/branch/files facts a
    commit-provenance note is built from — all interpretation of these raw
    facts happens server-side (`VectrService.record_commit_note`), this only
    gathers them."""
    try:
        result = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=_GIT_SUBPROCESS_TIMEOUT_S,
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _post_commit_note(port: int, sha: str, subject: str, branch: str, files: list[str]) -> bool:
    """POST /v1/commit-note; True on success, False on any failure (never
    raises) — mirrors `_post_snapshot`/`_post_trigger_reset` immediately
    above. A down, slow, or erroring daemon must never surface as a git hook
    failure (UPG-COMMIT-MEMORY-HOOK fail-safe contract)."""
    import httpx
    try:
        resp = httpx.post(
            f"{_api_base(port)}/v1/commit-note",
            json={"sha": sha, "subject": subject, "branch": branch, "files": files},
            timeout=HOOKS_POST_COMMIT_TIMEOUT_S,
        )
        resp.raise_for_status()
        return True
    except Exception:
        return False


def _read_hook_stdin() -> dict:
    """Read the Claude Code hook event JSON from stdin; {} if absent/invalid."""
    try:
        raw = sys.stdin.read()
        return json.loads(raw) if raw.strip() else {}
    except Exception:
        return {}


_FAILURE_EXIT_CODE_LINE_RE = re.compile(r"^Exit code (-?\d+)\n?")


def _parse_failure_error(error: str) -> tuple[int | None, str]:
    """Structurally parse the editor harness's PostToolUseFailure error
    string (G0 live capture, 2026-07-22, real `claude -p` session): a fixed
    `"Exit code N\\n<merged stdout+stderr>"` shape, e.g. `"Exit code 7\\n
    failing-probe"` or `"Exit code 1\\nls: /x: No such file or directory"`.

    Returns `(rc, remainder)` — `rc` is None (never guessed) when the
    leading line doesn't match this exact shape; `remainder` is the merged
    output digest to run marker matching over either way. This reads the
    harness's own fixed error-message structure, not task-prompt content or
    a query — R5-sanctioned (tool output/argv only)."""
    m = _FAILURE_EXIT_CODE_LINE_RE.match(error)
    if m is None:
        return None, error
    try:
        rc = int(m.group(1))
    except ValueError:
        return None, error
    return rc, error[m.end():]


def _tail_truncate(text: str, cap: int) -> str:
    """Keep the LAST `cap` characters of `text` (never the first) — see
    `_EPISODE_FOREGROUND_TRUNCATE_CHARS`'s docstring for why tail, not head."""
    return text[-cap:] if len(text) > cap else text


def _build_episode_payload(event: dict) -> dict | None:
    """Build the `POST /v1/episode` payload from a raw PostToolUse (or
    PostToolUseFailure) hook JSON event, or None when `tool_name` isn't one
    episode capture handles.

    Shapes below are G0 live-verified (2026-07-22, real `claude -p` session,
    claude-code 2.1.181) — no longer defensive guesses:

    - PostToolUse fires ONLY on tool success. Bash's `tool_response` is
      `{stdout, stderr, interrupted, isImage, noOutputExpected}` — there is
      NO exit-code field on this path, ever (rc is structurally absent, not
      merely omitted). Edit/Write/MultiEdit/apply_patch's `tool_response`
      carries `filePath`/`structuredPatch`/etc.; only `tool_input.file_path`
      is read from any of them (the path only, never file content).
    - A FAILED Bash call (non-zero exit) fires PostToolUseFailure instead —
      PostToolUse never fires for it. `tool_response` is absent; instead a
      top-level `error` string (`"Exit code N\\n<merged output>"`, see
      `_parse_failure_error`) and `is_interrupt` (note the different key
      name vs. `interrupted` on the success path — normalized to the same
      `interrupted` episode field here).
    - `harness_success` is derived from `hook_event_name` alone (which
      discrete event the harness routed this call through — tool-call
      structure, the same R5-sanctioned category as `rc`): True only for
      the literal string "PostToolUse", False for "PostToolUseFailure",
      None otherwise (payload drift / an editor that sends neither name —
      never guessed). UPG-EPISODE-OUTCOME-CLASSIFICATION: on the success
      path Bash's `tool_response` structurally never carries an exit code
      at all (see above), so most ordinary, non-build-tool commands
      previously had no signal to escape outcome="unknown" with; the fact
      that PostToolUse (not PostToolUseFailure) fired at all is itself the
      missing signal — `agent/outcome.py`'s cascade still checks content
      markers and `rc` first, so a piped/quieted build failure that prints
      a failure marker despite a clean top-level exit still wins.

    Every field is still read defensively (`.get()`, isinstance checks) in
    case a future harness version or Codex's own payload drifts from this
    shape — absent/malformed fields silently yield None/"" rather than
    raising. All further interpretation (argv normalization, outcome
    derivation, digest canonicalization) happens server-side in
    `VectrService.record_episode` — this only gathers raw facts, mirroring
    `_git_fact`'s own division of labor for the post-commit hook.

    stdout/stderr (or the failure-path error remainder) are tail-truncated
    to `_EPISODE_FOREGROUND_TRUNCATE_CHARS` HERE, before the temp-file
    handoff — `agent/episode_worker.py` applies
    its own, smaller, config-driven cap afterward as the real truncation;
    this is only a coarse foreground safety cap bounding this synchronous
    process's own json.dump cost against an arbitrarily large tool output.
    """
    tool_name = (event.get("tool_name") or "").strip()
    hook_event_name = (event.get("hook_event_name") or "").strip()
    tool_input = event.get("tool_input") or {}
    if not isinstance(tool_input, dict):
        tool_input = {}

    if tool_name == "Bash":
        tool = "bash"
        command = tool_input.get("command")
        description = tool_input.get("description")
        file_path = None
    elif tool_name in ("Edit", "Write", "MultiEdit", "apply_patch"):
        tool = "edit"
        command = None
        description = None
        file_path = tool_input.get("file_path")
    else:
        return None  # a tool_name this lane does not capture episodes for

    def _text(value: object) -> str:
        if isinstance(value, str):
            return value
        return "" if value is None else str(value)

    if hook_event_name == "PostToolUseFailure":
        harness_success = False
        rc, remainder = _parse_failure_error(_text(event.get("error")))
        is_error = True
        interrupted = bool(event.get("is_interrupt", False))
        stdout_tail = remainder
        stderr_tail = ""
    else:
        harness_success = True if hook_event_name == "PostToolUse" else None
        tool_response = event.get("tool_response") or {}
        if not isinstance(tool_response, dict):
            tool_response = {}
        rc_raw = tool_response.get("exit_code", tool_response.get("returncode"))
        rc = rc_raw if isinstance(rc_raw, int) else None
        is_error = bool(tool_response.get("is_error", False))
        interrupted = bool(tool_response.get("interrupted", False))
        stdout_tail = _text(tool_response.get("stdout"))
        stderr_tail = _text(tool_response.get("stderr"))

    return {
        "session_id": (event.get("session_id") or None),
        "cwd": event.get("cwd") or "",
        "tool": tool,
        "command": command,
        "description": description,
        "file_path": file_path,
        "rc": rc,
        "is_error": is_error,
        "interrupted": interrupted,
        "harness_success": harness_success,
        "stdout_tail": _tail_truncate(stdout_tail, _EPISODE_FOREGROUND_TRUNCATE_CHARS),
        "stderr_tail": _tail_truncate(stderr_tail, _EPISODE_FOREGROUND_TRUNCATE_CHARS),
    }


def _spawn_episode_worker(port: int, payload: dict) -> None:
    """Hand `payload` to a fully detached child process
    (`agent/episode_worker.py`) that truncates the stdout/stderr text to the
    config cap and performs the `POST /v1/episode` HTTP call — the
    PostToolUse hook's own foreground budget is ≤50ms p95 (spec §7 gate G2),
    which neither an HTTP round-trip nor importing the full `agent.config`
    surface (measured ~50–90ms alone, see agent/hook_cli.py's own module
    docstring) can reliably meet.

    The payload is written to a private temp file rather than piped to the
    child's stdin: a raw tool-output capture can be large (e.g. a verbose
    test-suite log), and a pipe write from this process would BLOCK once the
    OS pipe buffer fills if the child hasn't started reading yet — which it
    can't, until its own `agent.config` import finishes. A local temp-file
    write is a fast, non-blocking disk write regardless of payload size, so
    this process can hand off and return immediately. The child deletes the
    file after reading it.

    Mirrors `_do_start`'s Popen-detached idiom: unlike the git post-commit
    hook (which backgrounds itself at the shell level via its own generated
    wrapper script), PostToolUse's `command` field is invoked directly by the
    editor harness with no custom shell available, so detachment happens
    entirely inside this process via `start_new_session=True` — the child
    outlives this one's exit. Never raises: a spawn failure silently drops
    this one episode rather than breaking the tool call it's reporting on."""
    try:
        import tempfile
        fd, tmp_path = tempfile.mkstemp(prefix="vectr-episode-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"port": port, "payload": payload}, f)
        subprocess.Popen(
            [sys.executable, "-m", "agent.episode_worker", tmp_path],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(Path(__file__).resolve().parent),
            start_new_session=True,
        )
    except Exception:
        pass


def _emit_hook_context(event_name: str, text: str) -> None:
    """Print the Claude Code additionalContext envelope — only when there's text.

    Emitting nothing (instead of an empty envelope) means a fresh workspace
    injects nothing rather than noise. SessionStart/UserPromptSubmit injections
    are prefixed with a one-line notice (UPG-11.5) so the model doesn't also
    self-call vectr_recall for notes it was just handed.
    """
    if not text.strip():
        return
    if event_name in _HOOK_EVENTS_ANNOUNCE_INJECTION:
        text = f"{_HOOK_NO_DOUBLE_RECALL_LINE}\n\n{text}"
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": event_name,
            "additionalContext": text,
        }
    }))


def _emit_hook_plain_text(text: str) -> None:
    """Print `text` verbatim to stdout — only when non-empty.

    Deliberately NOT `_emit_hook_context`: the PreCompact hook's stdout is
    consumed by the editor harness as raw trimmed plain text and appended
    as custom compact instructions, NOT parsed as a `hookSpecificOutput`
    JSON envelope — emitting that envelope here would inject the literal
    JSON string into the summarizer's instructions as garbage. Used only by
    the pre-compact branch below.
    """
    if not text.strip():
        return
    print(text)


def _resolve_hook_instance(cwd: str) -> dict | None:
    """Find the running daemon serving `cwd`, or None.

    Multi-instance-safe with NO hardcoded port: the InstanceRegistry keys each
    `vectr start` workspace by its resolved path, so different folders resolve
    to different ports purely from cwd. We try the exact cwd first, then walk up
    parent directories so a hook fired from a subdirectory still finds the
    enclosing registered workspace. We never fall back to a default port — that
    could belong to an unrelated workspace and leak its memory into this session.
    """
    registry = InstanceRegistry()
    here = Path(cwd).resolve()
    for d in (here, *here.parents):
        entry = registry.get(workspace_hash(str(d)))
        if entry is not None:
            return entry
    return None


def cmd_hook(args: argparse.Namespace) -> None:
    """Emit hook output for harness-injected vectr memory (UPG-9.4+;
    TRIGGER-ENGINE wave 2a wires the per-memory trigger engine into every
    branch below).

    Invoked by the hook entries that `vectr init --hooks` writes — not meant to
    be called by hand. Resolves the workspace from the event's cwd (the harness
    runs hooks at the project root), then injects the right memory for the
    event. ALWAYS exits 0 and never raises: a hook must never break the session.

    Event mapping onto `agent.trigger_engine.EVENT_VALUES` (each hook JSON's
    own `session_id` — the same id across a `/compact` — threads through as
    the engine's per-session identity; absent, the call degrades to today's
    ledger-less/budget-less/scope-unenforced behaviour, never an error):

    | this hook event   | harness source(s)             | engine event(s)                    | REST call |
    |-------------------|-------------------------------|-------------------------------------|-----------|
    | session-start     | any `source`                  | session-start                       | POST /v1/recall {boot, session_id, hook_event} |
    | session-start     | `source == "compact"`          | session-start + post-compaction     | POST /v1/recall {..., events: [both]} |
    | user-prompt-submit| —                              | prompt-submit                       | POST /v1/recall {query, ..., events: [prompt-submit], session_id, hook_event} |
    | pre-tool-use      | file-bearing tool (Edit/Write/Read) | pre-edit + that file's path   | POST /v1/recall {file_path, kind: gotcha, session_id, hook_event} |
    | pre-tool-use      | command-running tool (Bash)    | pre-run + that command's normalized verb | POST /v1/recall {command, session_id, hook_event} |
    | post-tool-use     | Bash/Edit/Write/MultiEdit      | n/a — a WRITE path (episode capture), not a recall/trigger event | POST /v1/episode {session_id, cwd, tool, command/file_path, ...}, via a DETACHED worker (never awaited here) |
    | pre-compact       | any `trigger`                  | ledger reset (§3)                   | POST /v1/snapshot (unchanged) + POST /v1/trigger/reset {session_id} + GET /v1/boundary/precompact (printed as PLAIN TEXT via `_emit_hook_plain_text`, never the `hookSpecificOutput` envelope — see that function's docstring) |
    | post-commit       | n/a (fired by `git`, not the editor harness) | n/a — a WRITE path, not a recall/trigger event | POST /v1/commit-note {sha, subject, branch, files} |

    `session-start` with `source == "compact"` is the deterministic first
    delivery point after PreCompact's reset: it is the SAME lifecycle moment
    that already re-delivers the boot set (UPG-9.4's `compact` matcher), so a
    note whose ONLY explicit trigger is `post-compaction` (not covered by the
    directive kind-default bundle, which already includes `session-start`) is
    folded into that one call rather than inventing a new delivery point.
    `pre-run` is now live (wave 3, §5.2 — see the pre-tool-use row above);
    `pre-commit` remains declared-but-inert this wave (no lifecycle moment
    maps to it — never an error, bm2-design-skeleton.md §2).

    `post-commit` (UPG-COMMIT-MEMORY-HOOK) is fundamentally different from
    the five events above: it is invoked directly by `git` (installed by
    `_write_git_post_commit_hook`, not `_write_claude_hooks`/
    `_write_codex_hooks`), carries no editor-harness JSON on stdin (`event`
    degrades to `{}`, so `cwd` falls through to `os.getcwd()` — correct,
    since git runs hooks from the working-tree root), and never calls
    `_emit_hook_context` — nothing reads this process's stdout as a
    `hookSpecificOutput` envelope. It is also deliberately NOT part of the
    stdlib-only fast path (`agent/hook_cli.py`, see `main.py`'s own
    `_HOOK_EVENTS` comment): the shell wrapper backgrounds this whole
    invocation (`vectr hook post-commit </dev/null >/dev/null 2>&1 &`), so
    `git commit` returns before this process's imports even begin — the
    fast path's ~100ms import-cost saving exists for the five hooks above,
    which block the editor's turn loop synchronously; it buys nothing here.
    """
    try:
        event = _read_hook_stdin()
        cwd = event.get("cwd") or os.getcwd()
        entry = _resolve_hook_instance(cwd)
        if entry is None:
            return  # no daemon serves this workspace → inject nothing
        port = entry["port"]
        # TRIGGER-ENGINE wave 2a: the harness's own hook-JSON `session_id` —
        # present on every real hook invocation, stable across a /compact —
        # is the per-session identity the engine's fire ledger, cumulative
        # injection budget, and scope="session"/"branch" enforcement key on.
        # Reused, not invented: the same field the harness already threads
        # through its transcript_path/session lifecycle. Omitted from the
        # outgoing payload entirely when absent (older/stubbed callers), so
        # the wire shape is unchanged for them.
        session_id = (event.get("session_id") or "").strip() or None

        if args.hook_event == "session-start":
            # Unconditional boot set: directives + high-priority tasks (UPG-9.2),
            # the MEMORY.md equivalent — present before turn 1, zero model agency.
            # detail is NOT sent for boot=True because the service renders directives
            # at full and tasks at index automatically in the boot path.
            # hook_event (UPG-HOOK-INJECT-OBSERVABILITY) is the wire that makes this
            # firing visible in `vectr status` — without it, the daemon has no way
            # to tell a harness-injected recall apart from a direct one.
            payload = {"boot": True, "hook_event": "SessionStart"}
            if session_id:
                payload["session_id"] = session_id
            source = (event.get("source") or "").strip()
            if source == "compact":
                payload["events"] = ["session-start", "post-compaction"]
            notes = _fetch_recall(port, payload)
            _emit_hook_context("SessionStart", notes)

        elif args.hook_event == "user-prompt-submit":
            # Per-turn semantic recall (UPG-9.5): recall notes keyed to THIS prompt
            # and inject them before the model sees it. The relevance cutoff
            # (UPG-5.1) keeps an off-topic prompt from injecting anything.
            # detail="index" keeps the injected context token-bounded (UPG-RECALL-HIERARCHY).
            # events=["prompt-submit"] (TRIGGER-ENGINE wave 2a) additionally
            # fires any note with an EXPLICIT prompt-submit trigger override
            # (no kind's default bundle uses this event) — merged with, and
            # deduped against, the semantic query results server-side.
            prompt = (event.get("prompt") or "").strip()
            if not prompt:
                return
            limit = int(os.getenv("VECTR_HOOK_RECALL_LIMIT", str(_HOOK_RECALL_LIMIT)))
            payload = {
                "query": prompt, "limit": limit, "detail": "index",
                "hook_event": "UserPromptSubmit", "events": ["prompt-submit"],
            }
            # min_similarity is omitted here on purpose — the daemon applies
            # its own HOOKS_MIN_SIMILARITY default whenever hook_event is set
            # (see VectrService.recall), which is the single source of truth
            # for that floor. VECTR_HOOK_MIN_SIMILARITY still overrides it
            # explicitly, same as before.
            env_min_sim = os.getenv("VECTR_HOOK_MIN_SIMILARITY")
            if env_min_sim is not None:
                payload["min_similarity"] = float(env_min_sim)
            if session_id:
                payload["session_id"] = session_id
            notes = _fetch_recall(port, payload)
            _emit_hook_context("UserPromptSubmit", notes)

        elif args.hook_event == "pre-tool-use":
            # Gotcha injection (UPG-9.6): about to read or edit a file — surface any
            # caveat recorded against THAT file, at the moment of the tool call. Static
            # .claude/rules path-scoping can't do this; the gotcha is accrued + semantic.
            # `file_path` extraction is deterministic and tool-agnostic (Read, Edit,
            # Write, ... all put the target path under tool_input.file_path) — which
            # tool names actually reach this hook is a matcher decision made by
            # `_write_claude_hooks`, not a content-based guess made here. The
            # engine's pre-edit event (TRIGGER-ENGINE wave 2a) is merged in
            # server-side alongside this same file_path recall.
            #
            # Command-family injection: a Bash call carries no
            # `tool_input.file_path` at all, only `tool_input.command` — this
            # is a SEPARATE deterministic anchor (the 'command' trigger
            # primitive), not a file caveat, so it gets its own payload shape
            # rather than being forced through file_path. Which tool name
            # reaches this hook as Bash is, again, a matcher decision made by
            # `_write_claude_hooks`/`_write_codex_hooks`, not a guess here.
            # No `kind="gotcha"` filter: unlike the legacy file_path recall
            # below, there is no pre-existing content-substring fallback for
            # command-anchored notes — the 'command' trigger primitive (any
            # kind) is the entire surface, server-side in
            # `VectrService._recall_impl`'s `command` branch.
            tool_name = (event.get("tool_name") or "").strip()
            tool_input = event.get("tool_input") or {}
            if tool_name == "Bash":
                command = (tool_input.get("command") or "").strip()
                if not command:
                    return
                payload = {"command": command, "hook_event": "PreToolUse"}
                if session_id:
                    payload["session_id"] = session_id
                notes = _fetch_recall(port, payload)
                _emit_hook_context("PreToolUse", notes)
                return
            file_path = (tool_input.get("file_path") or "").strip()
            if not file_path:
                return
            payload = {"file_path": file_path, "kind": "gotcha", "hook_event": "PreToolUse"}
            if session_id:
                payload["session_id"] = session_id
            notes = _fetch_recall(port, payload)
            _emit_hook_context("PreToolUse", notes)

        elif args.hook_event == "post-tool-use":
            # Episode capture: never emits a hookSpecificOutput envelope
            # (nothing to inject) and never awaits the actual write — build
            # the payload, hand it to a detached worker, return. A
            # tool_name episode capture doesn't handle (anything but
            # Bash/Edit/Write/MultiEdit) yields None and is a silent no-op.
            payload = _build_episode_payload(event)
            if payload is not None:
                _spawn_episode_worker(port, payload)

        elif args.hook_event == "pre-compact":
            # Seal working memory before /compact replaces the conversation (UPG-9.7).
            # No hookSpecificOutput context is emitted here — compaction discards it
            # anyway; the boot set is re-injected afterwards by the SessionStart
            # `compact` matcher (UPG-9.4).
            trigger = (event.get("trigger") or "manual").strip() or "manual"
            label = f"pre-compact-{trigger}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
            _post_snapshot(port, label)
            # TRIGGER-ENGINE wave 2a (§3 "cleared on compaction"): reset this
            # session's fire ledger + cumulative injection budget so the
            # SessionStart `compact` call above gets a fresh budget and
            # re-eligibility for every trigger axis. A session with no
            # tracked ledger (or no session_id at all) is a no-op.
            if session_id:
                _post_trigger_reset(port, session_id)
            # Boundary-preservation nudge (distinct from the two calls above):
            # steers the summarizer itself, so it must reach stdout as PLAIN
            # TEXT via `_emit_hook_plain_text`, never the JSON envelope
            # `_emit_hook_context` emits (see that function's docstring for
            # why the two are not interchangeable on this hook event).
            _emit_hook_plain_text(_fetch_boundary_precompact(port))

        elif args.hook_event == "post-commit":
            # UPG-COMMIT-MEMORY-HOOK: fired by git itself, post-commit — no
            # editor-harness JSON, so `event` is `{}` and `cwd` above already
            # resolved to os.getcwd() (git runs hooks from the working-tree
            # root). All git facts are gathered HERE via read-only `git`
            # subprocess calls rather than in the shell wrapper, so there is
            # no shell string-escaping to get right for an arbitrary commit
            # subject/branch/file name. All interpretation (file-list
            # capping, active-task lookup, content formatting) happens
            # server-side in VectrService.record_commit_note — this branch
            # only gathers and forwards raw facts.
            sha = _git_fact(cwd, "rev-parse", "--short", "HEAD")
            if not sha:
                return  # not a git repo, or git unavailable -> nothing to record
            subject = _git_fact(cwd, "log", "-1", "--format=%s", "HEAD")
            branch = _git_fact(cwd, "rev-parse", "--abbrev-ref", "HEAD")
            if branch == "HEAD":
                branch = ""  # detached HEAD — git's own sentinel, not a real branch name
            # `log -m --first-parent --root` instead of `diff-tree -r HEAD`:
            # diff-tree prints NOTHING for a parentless initial commit (no
            # --root) and for merge commits (its --first-parent does not
            # constrain -m, so adding both concatenates every parent's diff).
            # This form yields the initial commit's full file list and a
            # merge's diff vs first parent — the "what landed on this
            # branch" reading — while normal commits are unchanged.
            files_raw = _git_fact(
                cwd, "log", "-1", "--format=", "--name-only", "-m",
                "--first-parent", "--root", "HEAD",
            )
            files = [f for f in files_raw.splitlines() if f]
            _post_commit_note(port, sha, subject, branch, files)
    except Exception:
        pass  # hook safety: never propagate


def _workspace_display_lines(entry: dict, fallback_workspace: str, label: str) -> list[str]:
    """Format the "Workspace" line(s) for `status` output.

    Shows the originating `.code-workspace` file when the instance was
    started from one (recorded in the registry at `start`/`restart` time),
    otherwise the primary folder plus any extra roots — never just the
    primary folder alone when there's more to the workspace than that
    (UPG-CLI-STATUS-MODE).
    """
    code_workspace_file = (entry or {}).get("code_workspace_file")
    primary = (entry or {}).get("workspace") or fallback_workspace
    if code_workspace_file:
        return [f"{label} : {code_workspace_file}"]
    lines = [f"{label} : {primary}"]
    for extra in (entry or {}).get("extra_roots") or []:
        lines.append(f"{' ' * len(label)}   + {extra}")
    return lines


def _mode_and_index_lines(
    data: dict, files_label: str, chunks_label: str, last_indexed_label: str | None,
    mode_label: str = "Mode         ",
) -> list[str]:
    """Format the Mode + indexed-files/chunks/last-indexed lines.

    In memory-only/search-only mode, indexing never runs in this process —
    `indexed_files` reflects only files this process has walked (0, since
    the startup walk is skipped), while `total_chunks` reads the persisted
    index and can be nonzero from an earlier full-mode run. Framing both as
    if they were live counts is misleading (UPG-CLI-STATUS-MODE): the mode
    line makes the gap self-explanatory, and the row wording says the
    figures are what's persisted, not what this run has done.
    """
    mode = data.get("mode", "full")
    lines = [f"{mode_label} : {mode}"]
    if mode == "full":
        lines.append(f"{files_label} : {data['indexed_files']}")
        lines.append(f"{chunks_label} : {data['total_chunks']}")
        if last_indexed_label:
            lines.append(f"{last_indexed_label} : {data['last_indexed']}")
    else:
        lines.append(
            f"{chunks_label} : {data['total_chunks']} "
            f"(persisted from the last full index; indexing is disabled in {mode} mode)"
        )
    return lines


def _watcher_backlog_line(data: dict) -> str | None:
    """One-line watcher backlog summary for `vectr status` (UPG-WATCHER-
    PRESSURE-GOVERNOR) — None when the workspace is quiet, so a healthy
    workspace's output stays terse instead of always printing a zero line."""
    pending = data.get("watcher_pending_files", 0)
    burst = data.get("watcher_burst_mode", False)
    running = data.get("watcher_batch_running", False)
    if not (pending or burst or running):
        return None
    last_ms = data.get("watcher_last_batch_duration_ms", 0)
    parts = [f"{pending} file(s) pending"]
    if burst:
        parts.append("burst mode")
    if running:
        parts.append("batch running")
    if last_ms:
        parts.append(f"last batch {last_ms}ms")
    return f"Watcher       : {', '.join(parts)}"


def _hook_injection_line(data: dict) -> str | None:
    """One-line hook-injection-count summary for `vectr status` (UPG-HOOK-
    INJECT-OBSERVABILITY) — None when no hook has injected anything yet, so a
    workspace with no Claude Code hooks installed (or whose hooks haven't
    fired) stays terse instead of a permanent zero line. Without this, hook
    injection is invisible: notes land silently in the model's context and
    the human has no way to tell a working memory system from a dead one."""
    counts = data.get("hook_injection_counts") or {}
    if not counts:
        return None
    parts = ", ".join(f"{kind} {n}" for kind, n in counts.items())
    return f"Hook injections : {parts}"


def _render_injection_lines(inject: bool, daemon_base: str, daemon_status: dict | None) -> list[str]:
    """Proxy-banner lines describing the END-TO-END injection state
    (UPG-PROXY-HIDDEN-MASTER-SWITCH): the proxy-side flag alone used to print
    "on" while the daemon returned empty context for every request. The banner
    now reflects what will actually happen. Metadata only — daemon_status is
    the /v1/status payload or None when unreachable."""
    if not inject:
        return ["  Injection : off (transparent pass-through)"]
    if daemon_status is None:
        return [
            f"  Injection : on (daemon {daemon_base}) — WARNING: daemon unreachable;"
            " injections fail open (requests forward unmodified) until it is up."
        ]
    lines = [f"  Injection : on (daemon {daemon_base}, proxy-channel launch consent)"]
    if not daemon_status.get("proactive_enabled", False):
        lines.append(
            "              ambient (hook) injection is off daemon-side; the proxy"
            " channel injects because you launched it."
        )
    return lines


def _proactive_injection_line(data: dict) -> str | None:
    """One-line proactive-injection summary for `vectr status` (UPG-PRO) — None
    until proactive context has injected something, so a workspace not using the
    feature stays terse instead of a permanent zero line."""
    counts = data.get("proactive_injection_counts") or {}
    if not counts:
        return None
    parts = ", ".join(f"{channel} {n}" for channel, n in counts.items())
    return f"Proactive injections : {parts}"


def _artifact_cache_line(data: dict) -> str | None:
    """One-line org-wide artifact-cache summary for `vectr status` (UPG-PRO) —
    None when the cache is off, so its metrics only appear once it is enabled."""
    metrics = data.get("artifact_cache")
    if not metrics:
        return None
    return (
        f"Artifact cache : {metrics.get('hits', 0)} hits / "
        f"{metrics.get('misses', 0)} misses "
        f"(hit rate {metrics.get('hit_rate', 0.0)}, "
        f"{metrics.get('entries', 0)} entries, "
        f"~{metrics.get('est_tokens_saved', 0)} tokens saved)"
    )


def cmd_status(args: argparse.Namespace) -> None:
    import httpx

    registry = InstanceRegistry()

    if getattr(args, "all", False):
        registry.prune_dead()
        instances = registry.list_all()
        if not instances:
            print("No running Vectr instances.")
            return
        for entry in instances.values():
            port = entry["port"]
            print()
            for line in _workspace_display_lines(entry, entry["workspace"], "Workspace"):
                print(line)
            print(f"Port      : {port}")
            print(f"PID       : {entry['pid']}")
            print(f"Started   : {entry.get('started_at', 'unknown')}")
            try:
                resp = httpx.get(f"{_api_base(port)}/v1/status", timeout=2)
                resp.raise_for_status()
                d = resp.json()
                for line in _mode_and_index_lines(d, "Files    ", "Chunks   ", None, mode_label="Mode     "):
                    print(line)
                watcher_line = _watcher_backlog_line(d)
                if watcher_line:
                    print(watcher_line)
                hook_line = _hook_injection_line(d)
                if hook_line:
                    print(hook_line)
                _check_version_skew(port, daemon_status=d, registry_entry=entry)
            except httpx.ConnectError:
                print("  (not listening — not running)")
            except httpx.HTTPError:
                print("  (listening, but not responding yet — still starting up)")
        return

    workspace = str(Path(args.path).resolve())
    entry = registry.get(workspace_hash(workspace))
    port = _get_port_for_workspace(workspace, args.port)
    try:
        resp = httpx.get(f"{_api_base(port)}/v1/status", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        for line in _workspace_display_lines(entry, data["workspace_root"], "Workspace    "):
            print(line)
        for line in _mode_and_index_lines(
            data, "Indexed files", "Total chunks ", "Last indexed ", mode_label="Mode         ",
        ):
            print(line)
        print(f"Embed model   : {data['embed_model']}")
        watcher_line = _watcher_backlog_line(data)
        if watcher_line:
            print(watcher_line)
        hook_line = _hook_injection_line(data)
        if hook_line:
            print(hook_line)
        proactive_line = _proactive_injection_line(data)
        if proactive_line:
            print(proactive_line)
        cache_line = _artifact_cache_line(data)
        if cache_line:
            print(cache_line)
        _check_version_skew(port, daemon_status=data, registry_entry=entry)
    except httpx.ConnectError:
        # UPG-CLI-START-READY-RACE: nothing is listening on this port at all
        # — genuinely not running, as opposed to the case below.
        print(f"Vectr is not listening on port {port} (not running). Run: vectr start", file=sys.stderr)
        sys.exit(1)
    except httpx.HTTPStatusError as exc:
        print(f"Vectr is listening on port {port} but returned an error: {_daemon_error_detail(exc)}", file=sys.stderr)
        sys.exit(1)
    except httpx.HTTPError:
        # Connection accepted but the request didn't complete (timeout,
        # dropped connection, ...) — the daemon is listening but not yet
        # ready to serve requests, e.g. still inside FastAPI lifespan
        # startup loading the embedder. Distinct from "not running": the
        # process exists, it just isn't answering yet.
        print(
            f"Vectr is listening on port {port} but is not responding yet "
            f"(still starting up). Try again in a few seconds.",
            file=sys.stderr,
        )
        sys.exit(1)


def cmd_stop(args: argparse.Namespace) -> None:
    registry = InstanceRegistry()

    if getattr(args, "all", False):
        instances = registry.list_all()
        if not instances:
            print("No running Vectr instances.", file=sys.stderr)
            return
        for ws_hash, entry in list(instances.items()):
            pid = entry["pid"]
            print(f"Stopping {entry['workspace']} (PID {pid})...", file=sys.stderr)
            _stop_server(pid)
            registry.unregister(ws_hash)
            print(f"  Stopped PID {pid}")
        return

    # UPG-CLI-STOP-NO-PORT-FLAG: port-addressed stop, for scripts/harnesses
    # that manage scratch daemons by the port they launched on and have no
    # reason to know (or reconstruct) the exact registered workspace path.
    # Wins over the positional workspace / --path, matching how the
    # positional already wins over --path in `_resolve_stop_workspace` —
    # the most specific identifier given wins.
    port = getattr(args, "port", None)
    if port is not None:
        registry.prune_dead()
        match = next(
            ((wh, e) for wh, e in registry.list_all().items() if e["port"] == port),
            None,
        )
        if match is None:
            print(f"No vectr instance registered on port {port}.", file=sys.stderr)
            sys.exit(1)
        ws_hash, entry = match
        pid = entry["pid"]
        _stop_server(pid)
        registry.unregister(ws_hash)
        print(f"Vectr stopped (PID {pid}) on port {port}")
        return

    workspace = _resolve_stop_workspace(args)
    ws_hash = workspace_hash(workspace)
    registry.prune_dead()
    entry = registry.get(ws_hash)
    if entry is None:
        print(f"No registered instance for workspace: {workspace}", file=sys.stderr)
        return
    pid = entry["pid"]
    _stop_server(pid)
    registry.unregister(ws_hash)
    print(f"Vectr stopped (PID {pid})")


def _resolve_stop_workspace(args: argparse.Namespace) -> str:
    """Resolve the single workspace path `stop` should act on.

    The optional positional `workspace` argument (UPG-CLI-STOP-PATH-POSITIONAL)
    mirrors `start`/`restart`'s positional — a `.code-workspace` file or a plain
    directory — and wins over `--path` when both are given, matching how
    `_resolve_workspace_roots` prioritises the positional for start/restart.
    Falls back to `--path` (or its default) when no positional was given.
    """
    workspace_arg = getattr(args, "workspace", None)
    if workspace_arg:
        if str(workspace_arg).endswith(".code-workspace"):
            return _parse_code_workspace(workspace_arg)[0]
        return str(Path(workspace_arg).resolve())
    return str(Path(args.path).resolve())


def _resolve_restart_mode(
    entry: dict | None, memory_only: bool, search_only: bool, full: bool,
) -> tuple[bool, bool, str | None]:
    """`(memory_only, search_only, inherited_mode)` for a `restart`.

    UPG-RESTART-DROPS-MODE: `restart` means "the same daemon again", so when
    the caller names no mode it reproduces the mode recorded for the previous
    instance instead of silently falling back to full mode. Explicit flags
    always win — `--memory-only` / `--search-only` select their mode, and
    `--full` is the way back out of an inherited one (without it a workspace
    once started memory-only could never be restarted into full mode).
    `inherited_mode` is the label that was inherited, or None when the caller
    was explicit / there was nothing to inherit — the caller announces it so
    the mode change is never invisible.
    """
    if memory_only or search_only or full:
        return memory_only, search_only, None
    recorded = (entry or {}).get("mode")
    if recorded == MODE_MEMORY_ONLY:
        return True, False, MODE_MEMORY_ONLY
    if recorded == MODE_SEARCH_ONLY:
        return False, True, MODE_SEARCH_ONLY
    return False, False, None


def _restart_roots_explicit(args: argparse.Namespace) -> bool:
    """True when `restart`'s caller named an explicit multi-root workspace:
    a `.code-workspace` file (which enumerates its own full folder list) or
    one-or-more `--path` flags. False for a bare positional directory or the
    VECTR_WORKSPACE/cwd default — those name only the PRIMARY root and leave
    `_resolve_restart_roots` free to reproduce the previous instance's
    `extra_roots` from the registry, the same "no flag named it, so inherit"
    rule `_resolve_restart_mode` above already applies to mode
    (UPG-RESTART-DROPS-EXTRA-ROOTS).
    """
    ws = getattr(args, "workspace", None)
    if ws:
        return str(ws).endswith(".code-workspace")
    return bool(getattr(args, "paths", None) or [])


def _resolve_restart_roots(
    entry: dict | None, roots: list[str], roots_explicit: bool,
) -> tuple[list[str], list[str] | None]:
    """`(roots, inherited_extra_roots)` for a `restart` — the roots
    counterpart to `_resolve_restart_mode`, with the same precedence: an
    explicit caller-given root list always wins outright, with no merge
    against the registry (silently growing an explicit list back out would
    be its own surprise). An under-specified one (just the primary
    workspace — a bare positional directory, or the VECTR_WORKSPACE/cwd
    default) instead reproduces the `extra_roots` recorded for the previous
    instance at that same primary path (UPG-RESTART-DROPS-EXTRA-ROOTS — the
    live bug this fixes: a multi-root instance's own staleness banner told
    the operator to run `vectr restart <primary-path>`, which came back
    with only 1 of 9 roots because `cmd_restart` built its root list purely
    from CLI args and never consulted the registry entry it already had in
    hand for mode/host). `inherited_extra_roots` is the list that was
    inherited, or None when the caller was explicit / there was nothing to
    inherit, so the caller can announce it — never silent, symmetric with
    `inherited_mode`.
    """
    if roots_explicit:
        return roots, None
    recorded = (entry or {}).get("extra_roots") or []
    if not recorded:
        return roots, None
    primary = roots[0]
    inherited = [r for r in recorded if r != primary]
    if not inherited:
        return roots, None
    return [primary, *inherited], inherited


def cmd_restart(args: argparse.Namespace) -> None:
    memory_only = getattr(args, "memory_only", False)
    search_only = getattr(args, "search_only", False)
    full = getattr(args, "full", False)
    if memory_only and search_only:
        print("Error: --memory-only and --search-only are mutually exclusive.", file=sys.stderr)
        sys.exit(1)
    if full and (memory_only or search_only):
        print(
            "Error: --full cannot be combined with --memory-only/--search-only.",
            file=sys.stderr,
        )
        sys.exit(1)

    roots = _resolve_workspace_roots(args)
    workspace = roots[0]
    ws_hash = workspace_hash(workspace)
    preferred_port = args.port
    explicit = _is_explicit_workspace(args)
    if explicit:
        _warn_if_enclosing_repo(workspace)

    registry = InstanceRegistry()
    entry = registry.get(ws_hash)

    # Mode, bind host, and extra roots are all inherited from the recorded
    # previous instance for the same reason its port already is
    # (UPG-RESTART-PORT-WALK-BREAKS-MCP): a restart that quietly changes how
    # the daemon runs breaks whatever was relying on the old configuration.
    # Resolved before `_enforce_bind_auth` so an inherited non-loopback bind
    # is authenticated exactly like an explicitly-passed one.
    memory_only, search_only, inherited_mode = _resolve_restart_mode(
        entry, memory_only, search_only, full,
    )
    if inherited_mode:
        print(
            f"Restarting in {inherited_mode} mode (inherited from the previous "
            f"instance for this workspace). Pass "
            f"{MODE_RESTART_FLAG[MODE_FULL]} to restart in full mode instead.",
            file=sys.stderr,
        )
    roots, inherited_extra_roots = _resolve_restart_roots(
        entry, roots, _restart_roots_explicit(args),
    )
    extra_roots = roots[1:]
    if inherited_extra_roots:
        print(
            f"Restarting with {len(inherited_extra_roots)} extra root(s) "
            "inherited from the previous instance for this workspace: "
            f"{', '.join(inherited_extra_roots)}. Pass --path to give an "
            "explicit root list instead.",
            file=sys.stderr,
        )
    host = getattr(args, "host", None) or (entry or {}).get("host") or "127.0.0.1"
    _enforce_bind_auth(host)

    if entry is not None:
        pid = entry["pid"]
        print(f"Stopping PID {pid}...", file=sys.stderr)
        _stop_server(pid)

    # UPG-RESTART-PORT-WALK-BREAKS-MCP: `find_free_port` must run BEFORE the
    # entry is unregistered — its "reuse the previous port" step (step 2)
    # only fires when a dead registry entry is still present. Unregistering
    # first (the old order here) erased that hint on every restart, so a
    # restart always fell through to scanning upward from `preferred_port`
    # instead of reusing the workspace's actual previous port — which then
    # only happened to work when `preferred_port` matched that port exactly.
    # `_do_start` below re-registers this ws_hash unconditionally, so the
    # explicit unregister() has no effect on final state either way.
    port = registry.find_free_port(ws_hash, preferred_port)
    registry.unregister(ws_hash)
    for root in roots:
        _maybe_write_workspace_config(root, port, args, search_only=search_only)
    _do_start(
        workspace, port, ws_hash, extra_roots=extra_roots,
        memory_only=memory_only, search_only=search_only, workspace_explicit=explicit,
        code_workspace_file=_code_workspace_file_arg(args), host=host,
        no_ide_config=getattr(args, "no_ide_config", False),
        foreground_fast=getattr(args, "foreground_fast", False),
    )
    for root in roots:
        _warn_on_stale_mcp_configs(root, port)


def cmd_forget(args: argparse.Namespace) -> None:
    import httpx

    # --all clears notes across ALL workspaces directly via SQLite,
    # bypassing the running server (server may be down, or multiple instances).
    if getattr(args, "all", False):
        from agent.working_context_store import WorkingContextStore
        cache_root = vectr_cache_root()
        # Layout of record: ~/.cache/vectr/<workspace-hash>/working_context.sqlite
        # (app.service._default_db_dir). Earlier builds nested the same file under
        # ~/.cache/vectr/db/<hash>/ — sweep both layouts so --all cannot silently
        # miss notes and report success having deleted nothing.
        db_files = sorted(
            set(cache_root.glob("*/working_context.sqlite"))
            | set(cache_root.glob("db/*/working_context.sqlite"))
        )
        print("--all sweeps every workspace database on this machine (--port/--path ignored).")
        total = 0
        for db_file in db_files:
            store = WorkingContextStore(str(db_file.parent))
            total += store.forget_all_workspaces()
        print(f"Deleted {total} working-memory notes across {len(db_files)} workspace databases.")
        return

    workspace = str(Path(args.path).resolve())
    port = _get_port_for_workspace(workspace, args.port)
    _check_version_skew(port, registry_entry=_resolve_hook_instance(workspace))
    try:
        resp = httpx.post(f"{_api_base(port)}/v1/memory/clear", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        print(f"Deleted {data['deleted']} working-memory notes for {workspace}")
    except (httpx.ConnectError, httpx.HTTPStatusError) as exc:
        _handle_daemon_call_error(exc, port)


def cmd_cache(args: argparse.Namespace) -> None:
    """`vectr cache prune` — remove empty per-workspace cache dirs (UPG-CACHE-LITTER).

    Opt-in cleanup for the hundreds of empty <hash>/ dirs the cache root can
    accumulate from path resolutions that never wrote anything. Only ever
    removes EMPTY dirs, skips the reserved db/ and models/ subdirs, and never
    touches a dir belonging to a live instance (rail R1)."""
    action = getattr(args, "cache_command", None)
    if action != "prune":
        print("usage: vectr cache prune [--dry-run]", file=sys.stderr)
        return
    from agent.cache_maintenance import live_instance_slugs, prune_empty_cache_dirs

    cache_root = vectr_cache_root()
    protected = live_instance_slugs()
    dry_run = getattr(args, "dry_run", False)
    affected = prune_empty_cache_dirs(cache_root, protected_slugs=protected, dry_run=dry_run)
    if not affected:
        print("No empty workspace cache dirs to prune.")
        return
    verb = "Would remove" if dry_run else "Removed"
    print(f"{verb} {len(affected)} empty workspace cache dir(s) under {cache_root}.")
    if protected:
        print(f"  ({len(protected)} live-instance dir(s) skipped.)")
    if dry_run:
        for d in affected:
            print(f"  {d.name}")


def cmd_init(args: argparse.Namespace) -> None:
    workspace = str(Path(args.path).resolve())

    if getattr(args, "reset_config", False):
        root = Path(workspace)
        for _rel in ("CLAUDE.md", *_IDE_CONFIG_APPEND_ONLY):
            _remove_vectr_block(root / _rel)
        _remove_vectr_block(root / ".github" / "copilot-instructions.md")
        cursor_mdc = root / ".cursor" / "rules" / "vectr.mdc"
        if cursor_mdc.exists():
            cursor_mdc.unlink()
            print(f"  Deleted {cursor_mdc}", file=sys.stderr)
        _remove_vectr_hooks(workspace)
        _remove_codex_hooks(workspace)
        _remove_codex_mcp_config(workspace)
        _remove_git_post_commit_hook(workspace)
        print(f"Vectr config reset for: {workspace}", file=sys.stderr)
        return

    # UPG-CLI-PATH-IGNORED: walk-up resolution (same as `_get_port_for_workspace`
    # / `_resolve_hook_instance`), not an exact-hash-only lookup — `init` run
    # from a subdirectory of an already-registered workspace should discover
    # that instance's real port and mode, not guess a provisional one.
    entry = _resolve_hook_instance(workspace)
    port_is_provisional = entry is None
    port = entry["port"] if entry is not None else int(os.getenv("VECTR_PORT", str(DEFAULT_PORT)))
    if port_is_provisional:
        # UPG-CLI-SMALL-UX: no instance is registered for this workspace yet,
        # so this port is a guess (VECTR_PORT or the 8765 fallback), not a
        # confirmed bound port — a real collision target if something else
        # already owns it. `vectr start` always finds an actually-free port
        # and overwrites these same files with it (see the CLI table entry
        # for `vectr init`), so this is silently self-correcting, but a user
        # who never runs `start` right after `init` would otherwise have no
        # indication the written port is unconfirmed.
        print(
            f"  Note: no vectr instance is running for this workspace yet — "
            f"MCP config files below use provisional port {port}. "
            f"`vectr start` will assign the real free port and correct them "
            f"automatically if it differs.",
            file=sys.stderr,
        )

    # Search-only mode (UPG-SEARCH-ONLY-MODE) has no working-memory layer to
    # inject — detected live from the running daemon (mode is a daemon
    # property, not a static workspace marker) rather than a CLI flag on init.
    search_only = _get_daemon_mode(port) == "search-only"

    # Hooks are written BEFORE the workspace config (UPG-11.5): CLAUDE.md's
    # (and AGENTS.md's, UPG-CODEX-PARITY) session-start guidance is hook-aware,
    # detected by reading back .claude/settings.json / .codex/hooks.json — so
    # within a single `vectr init --hooks` run, the hooks must already be on
    # disk when _write_workspace_config runs, or the guidance blocks would ship
    # the pre-hooks (double-recall) variant.
    if getattr(args, "hooks", False):
        if search_only:
            print(
                "Warning: this workspace's vectr daemon runs in search-only mode — "
                "there is no working-memory layer for hooks to inject notes from. "
                "Skipping hook installation. Restart without --search-only to use hooks.",
                file=sys.stderr,
            )
        else:
            _write_claude_hooks(workspace)
            _write_codex_hooks(workspace)
            # UPG-COMMIT-MEMORY-HOOK: a git-level hook, not an editor-harness
            # one — installed under the same --hooks flag and the same
            # search-only gate (it writes a note, so it needs the same
            # working-memory layer the two hooks above inject from).
            _write_git_post_commit_hook(workspace)
            # DeepSeek Harness (UPG-DSH-PARITY): point dsh's Claude Code hook
            # bridge at this workspace's .claude/settings.json (written just
            # above). Skipped silently when dsh is not installed. The path is
            # ABSOLUTE because dsh resolves configPath once at load against
            # the server's launch cwd, not per-session.
            if _dsh_home().exists():
                settings_path = Path(workspace) / ".claude" / "settings.json"
                _write_dsh_hooks_config(_dsh_patch_path(), str(settings_path.resolve()))

    _maybe_write_workspace_config(workspace, port, args, search_only=search_only)

    # write user-defined exclusions to .vectrignore
    _apply_exclude_args(workspace, getattr(args, "exclude", None) or [])

    # write style override if --style is specified
    if getattr(args, "style", None):
        style = args.style
        # UPG-TOOLSTYLE-LABEL-COLLISION: "memory-only" is accepted here for
        # backward compatibility (argparse choices above) but always
        # normalized to "memory-first" before being persisted — only the
        # current label is ever written to .vectr/style.
        if style == "memory-only":
            style = "memory-first"
        if style not in ("additive", "directed", "memory-first"):
            print(f"Error: --style must be one of: additive, directed, memory-first", file=sys.stderr)
            sys.exit(1)
        style_dir = Path(workspace) / ".vectr"
        style_dir.mkdir(parents=True, exist_ok=True)
        (style_dir / "style").write_text(style, encoding="utf-8")
        print(f"  Instruction style set: {style}", file=sys.stderr)

    print(f"Workspace configured: {workspace}", file=sys.stderr)
    print(f"  Run 'vectr start --path {workspace}' to index and start the server.", file=sys.stderr)


def cmd_watch(args: argparse.Namespace) -> None:
    """Index workspace(s) and start filesystem watcher without launching the MCP server."""
    import hashlib
    from agent.config import embed_model_spec
    from agent.indexer import CodeIndexer
    from agent.watcher import CodeWatcher
    from integrations.workspace_detect import find_workspace_root

    _preflight_grammars()

    roots = _resolve_workspace_roots(args)
    # UPG-WS-ROOT-MISDETECT: an explicit path/positional workspace arg wins
    # verbatim; the git-toplevel walk-up only applies when none was given.
    if _is_explicit_workspace(args):
        workspace = str(Path(roots[0]).resolve())
        _warn_if_enclosing_repo(workspace)
    else:
        workspace = find_workspace_root(roots[0])
    extra_roots = roots[1:]
    embed_model = embed_model_spec()

    # Use same db layout as VectrService so a later `vectr start` shares the index.
    db_hash = hashlib.md5(workspace.encode()).hexdigest()[:12]
    db_dir = vectr_cache_root() / db_hash
    db_dir.mkdir(parents=True, exist_ok=True)

    indexer = CodeIndexer(workspace, embed_model=embed_model, db_path=str(db_dir / "chroma"),
                          extra_roots=extra_roots)
    watcher = CodeWatcher(indexer)

    all_roots_str = ", ".join([workspace] + extra_roots)
    print(f"Indexing {all_roots_str} ...", file=sys.stderr)
    files, chunks = indexer.index_workspace()
    print(f"  Indexed {files} files, {chunks} chunks", file=sys.stderr)
    print(f"Watching for changes. Press Ctrl+C to stop.", file=sys.stderr)
    print(f"  Run 'vectr start --path {workspace}' to also serve MCP.", file=sys.stderr)

    watcher.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        watcher.stop()
        print("\nWatcher stopped.", file=sys.stderr)


def cmd_mcp_stdio(args: argparse.Namespace) -> None:
    """Run the MCP server on stdio: newline-delimited JSON-RPC 2.0 on stdin/
    stdout, no daemon, no port. For MCP clients and hosting platforms that
    spawn the server as a subprocess rather than connecting to a listening
    HTTP port.

    Stdout discipline is critical: stdout must carry ONLY protocol JSON, or a
    single stray byte corrupts the stream for whatever spawned this process.
    The real stdout is captured before anything else touches the stream, then
    `sys.stdout` is redirected to stderr for the rest of the process — a
    defense-in-depth net against a stray `print()` or third-party logging
    default (e.g. during model loading) landing on stdout.

    `VectrService` construction (embedder + reranker model loads) is slow —
    seconds, not milliseconds — so it runs on a background thread via
    `ServiceHandle` while the stdio read loop starts immediately: `initialize`,
    `notifications/initialized`, `ping`, and `tools/list` all answer before
    the workspace index (or even the service itself) is ready; `tools/call`
    reports "still starting up" gracefully until it is.

    `VectrService` itself is constructed with `defer_search_init=True`
    (UPG-STDIO-MEMORY-READY): phase 1 (fast — no model load) finishes almost
    immediately, `handle.set_service()` is called right there, and
    working-memory tools (remember/recall/forget/status/snapshot/
    snapshot_list) become servable well before the embedder/indexer/
    watcher/symbol-graph (phase 2) finish loading. `tools/call` for the
    remaining (search-side) tools still waits for `service.fully_ready`.
    """
    import logging

    protocol_stdout = sys.stdout
    sys.stdout = sys.stderr
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        stream=sys.stderr,
        force=True,
    )

    from integrations.mcp_server import ServiceHandle, run_stdio_loop

    memory_only = getattr(args, "memory_only", False)
    search_only = getattr(args, "search_only", False)
    if memory_only and search_only:
        print("Error: --memory-only and --search-only are mutually exclusive.", file=sys.stderr)
        sys.exit(1)

    roots = _resolve_workspace_roots(args)
    workspace = roots[0]
    extra_roots = roots[1:]
    explicit = _is_explicit_workspace(args)

    handle = ServiceHandle()

    def _build_service() -> None:
        try:
            from app.service import VectrService
            svc = VectrService(
                workspace_root=workspace,
                extra_roots=extra_roots,
                memory_only=memory_only,
                search_only=search_only,
                workspace_explicit=explicit,
                # No HTTP port exists on this transport — writing IDE config
                # files that point at one would be meaningless, and possibly
                # disruptive for a hosting platform's mounted workspace.
                configure_ide=False,
                # UPG-STDIO-MEMORY-READY: phase 1 only here — working-memory
                # tools become servable as soon as `set_service` below
                # returns, well before the embedder/indexer/watcher/symbol
                # graph (phase 2) finish loading in the background.
                defer_search_init=True,
            )
        except BaseException as exc:  # a background thread must never crash silently
            logging.getLogger(__name__).exception("mcp-stdio: service construction failed")
            handle.set_error(exc)
            return

        handle.set_service(svc)
        try:
            svc.complete_search_init()
            svc.start_background_index()
        except BaseException:
            # Phase 2 (embedder/indexer/watcher/symbol graph) failed to come
            # up. Working-memory tools already work (phase 1 succeeded above)
            # — never fail the whole service for this. `fully_ready` simply
            # never flips, so search-side tools keep answering "still
            # starting up" rather than the process crashing outright.
            logging.getLogger(__name__).exception(
                "mcp-stdio: phase 2 (search layer) initialisation failed — "
                "working-memory tools remain available; search tools will "
                "keep reporting still-starting-up"
            )

    threading.Thread(
        target=_build_service, daemon=True, name="vectr-mcp-stdio-service-init",
    ).start()

    print(f"vectr mcp-stdio: workspace={workspace}", file=sys.stderr)
    try:
        run_stdio_loop(handle, stdin=sys.stdin, stdout=protocol_stdout)
    finally:
        if handle.service is not None:
            try:
                handle.service.shutdown()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

# Shared --exclude help text (init and start accept identical entry forms —
# UPG-EXCLUDE-REGEX). PATTERN accepts a bare directory name, a file glob
# (e.g. *.generated.py), or a `re:<pattern>` path regex matched against the
# workspace-relative path (e.g. re:legacy/.* or re:.*_(backup|old)\.\w+$).
_EXCLUDE_HELP = (
    "Append a directory name, file glob, or `re:<pattern>` path regex to "
    ".vectrignore (repeatable). Examples: --exclude vendor "
    "--exclude '*.generated.py' --exclude 're:legacy/.*'"
)

# Shared IDE-config-writes disclosure (UPG-CLI-WRITES-DISCLOSURE): start,
# restart, and init all write IDE integration files into the workspace root
# as a side effect, with no prior --help mention and no opt-out.
_IDE_CONFIG_WRITES_DISCLOSURE = (
    "On first run for a workspace, vectr writes IDE integration files into "
    "the workspace root so an AI editor can auto-discover the MCP server: "
    "CLAUDE.md, .cursor/rules/vectr.mdc, .mcp.json, .cursor/mcp.json, "
    ".vscode/mcp.json, .codex/config.toml, .claude/settings.json, and "
    ".vectrignore (default excludes) — 8 files. Files for other editors/agents "
    "(AGENTS.md, .cursorrules, GEMINI.md, .github/copilot-instructions.md) "
    "only get a vectr guidance block appended if the file already exists — "
    "they are never created from scratch (Codex reads AGENTS.md for guidance, "
    "so no separate CODEX.md is written). Pass --no-ide-config to skip all "
    "of this; the choice persists at .vectr/ide_config for future start/"
    "restart/init calls on this workspace (delete that file to re-enable). "
    "See `vectr init --reset-config` to remove already-written blocks."
)

_NO_IDE_CONFIG_HELP = (
    "Skip writing IDE integration config files for this workspace and "
    "persist that choice (see the command description above). Delete "
    ".vectr/ide_config to re-enable on a later run."
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="vectr",
        description=(
            "Vectr: persistent external memory and semantic code search for AI coding "
            "agents. Indexes a codebase for fast search/symbol/call-graph lookups over "
            "MCP (search/locate/trace/map), and provides working memory (remember/"
            "recall/snapshot) that survives context compaction and session restarts. "
            "This CLI runs the daemon (start/stop/status) and also exposes search/"
            "remember/recall directly from a shell."
        ),
    )
    sub = parser.add_subparsers(dest="command")

    _default_path = os.getenv("VECTR_WORKSPACE", ".")
    _default_port = int(os.getenv("VECTR_PORT", str(DEFAULT_PORT)))

    p_start = sub.add_parser(
        "start",
        help="Start the Vectr daemon and index the workspace. "
             "Accepts a .code-workspace file, one or more --path flags, or defaults to cwd.",
        description=_IDE_CONFIG_WRITES_DISCLOSURE,
    )
    p_start.add_argument(
        "workspace", nargs="?", default=None,
        help="Path to a .code-workspace file or a single workspace directory",
    )
    p_start.add_argument(
        "--path", action="append", dest="paths", metavar="DIR",
        help="Workspace root to index (repeatable for multi-root). "
             "Example: vectr start --path dir1 --path dir2",
    )
    p_start.add_argument("--port", type=int, default=_default_port)
    p_start.add_argument(
        "--host", default="127.0.0.1",
        help="Bind address for the daemon (default 127.0.0.1, local-only). "
             "Use 0.0.0.0 or a specific interface to serve remote clients "
             "(team / central instance) — requires VECTR_API_KEY to be set.",
    )
    p_start.add_argument(
        "--exclude", action="append", metavar="PATTERN", dest="exclude",
        help=_EXCLUDE_HELP,
    )
    p_start.add_argument(
        "--memory-only",
        action="store_true",
        default=False,
        dest="memory_only",
        help=(
            "Run the daemon for working memory + Claude Code hooks WITHOUT "
            "indexing, embedding, or watching the codebase. "
            "Memory tools (remember/recall/snapshot) and hooks remain active; "
            "search/locate/trace are disabled. Useful on actively-edited projects "
            "where the full code index + watcher cause performance issues."
        ),
    )
    p_start.add_argument(
        "--search-only",
        action="store_true",
        default=False,
        dest="search_only",
        help=(
            "Run the daemon for semantic search + symbol graph + codebase map "
            "WITHOUT the working-memory layer — no notes DB is created for this "
            "workspace, and remember/recall/forget/snapshot are disabled. "
            "Indexing and the file watcher run normally. Useful for read-only "
            "consumers of an indexed codebase (reviewers, CI, a shared search "
            "server) where nothing should be written to a per-workspace note "
            "store. Mutually exclusive with --memory-only."
        ),
    )
    p_start.add_argument(
        "--no-ide-config",
        action="store_true",
        default=False,
        dest="no_ide_config",
        help=_NO_IDE_CONFIG_HELP,
    )
    p_start.add_argument(
        "--foreground-fast",
        action="store_true",
        default=False,
        dest="foreground_fast",
        help=(
            "Disable the indexer's resource governor (UPG-INDEX-RESOURCE-"
            "GOVERNOR): no OS priority clamp, no duty-cycle pacing between "
            "embedding batches. Indexing runs at the same full-tilt, "
            "multi-core pace as before the governor existed. Use on a "
            "dedicated machine (CI, a scratch box) where interactive "
            "responsiveness during indexing does not matter and finishing "
            "sooner does. Checkpoint/resume is a correctness guarantee, not "
            "a throttle, and stays on regardless of this flag."
        ),
    )
    p_start.add_argument(
        "--strict-port",
        action="store_true",
        default=False,
        dest="strict_port",
        help=(
            "Fail instead of walking to a different port when --port is not "
            "free (UPG-CLI-STOP-NO-PORT-FLAG companion). By default `start` "
            "treats --port as a preference and scans forward for the first "
            "free port past it, so the daemon can end up bound somewhere "
            "other than what was requested with no error. --strict-port "
            "exits non-zero instead, for scripts/harnesses that need the "
            "daemon on an exact, predictable port."
        ),
    )
    p_start.add_argument(
        "--json",
        action="store_true",
        default=False,
        dest="json_output",
        help=(
            "Print a single-line JSON object to stdout describing the "
            "outcome (status, port, pid, workspace, mode, mcp_url) instead "
            "of the human-readable text on stderr, for scripts/harnesses "
            "that need to discover the actually-bound port programmatically. "
            "Exits non-zero when status is \"failed\"."
        ),
    )

    p_stop = sub.add_parser("stop", help="Stop the daemon for a workspace")
    p_stop.add_argument(
        "workspace", nargs="?", default=None,
        help="Path to a .code-workspace file or a single workspace directory "
             "(same positional `start`/`restart` accept; wins over --path if both given)",
    )
    p_stop.add_argument("--path", default=_default_path)
    p_stop.add_argument(
        "--port", type=int, default=None,
        help="Stop the instance bound to this port instead of resolving by workspace "
             "path (UPG-CLI-STOP-NO-PORT-FLAG). Deterministic for scripts/harnesses "
             "managing scratch daemons by port; wins over the positional workspace "
             "and --path when given. Exits non-zero if nothing is registered on "
             "this port.",
    )
    p_stop.add_argument("--all", action="store_true", help="Stop all running instances")

    p_restart = sub.add_parser(
        "restart",
        help="Stop and restart the daemon for a workspace, reusing its "
             "recorded mode/port/bind address/extra roots unless flags "
             "override them",
        description=(
            "Restart reproduces the configuration of the instance it is "
            "replacing — mode (full / --memory-only / --search-only), port, "
            "bind address, and extra roots — so restarting never silently "
            "changes how the daemon runs. Pass a mode flag (or --full) to "
            "change the mode. A bare primary-path positional (or no "
            "positional at all) also restores every extra root the previous "
            "instance served; passing a .code-workspace file or one-or-more "
            "--path flags instead REPLACES the root list outright — it is "
            "never merged with the previous instance's roots.\n\n"
            + _IDE_CONFIG_WRITES_DISCLOSURE
        ),
    )
    p_restart.add_argument(
        "workspace", nargs="?", default=None,
        help="Path to a .code-workspace file or a single workspace directory. "
             "A single directory restores any extra roots recorded for the "
             "previous instance at that path; a .code-workspace file replaces "
             "them with exactly the folders it lists.",
    )
    p_restart.add_argument(
        "--path", action="append", dest="paths", metavar="DIR",
        help="Additional workspace root (repeatable). Giving one or more "
             "--path flags REPLACES any extra roots recorded for the "
             "previous instance — list every root you want restarted.",
    )
    p_restart.add_argument("--port", type=int, default=_default_port)
    p_restart.add_argument(
        # Default None (not "127.0.0.1") so `cmd_restart` can tell "the caller
        # asked for this bind" from "the caller said nothing", and only
        # inherit the previous instance's host in the second case
        # (UPG-RESTART-DROPS-MODE).
        "--host", default=None,
        help="Bind address for the daemon. Defaults to the bind address of "
             "the instance being restarted, or 127.0.0.1 if there is none. "
             "Non-loopback binds require VECTR_API_KEY (see vectr start --host).",
    )
    p_restart.add_argument(
        "--memory-only",
        action="store_true",
        default=False,
        dest="memory_only",
        help="Restart in memory-only mode (no indexing/watcher; see vectr start --memory-only).",
    )
    p_restart.add_argument(
        "--search-only",
        action="store_true",
        default=False,
        dest="search_only",
        help="Restart in search-only mode (no working-memory layer; see vectr start --search-only).",
    )
    p_restart.add_argument(
        "--full",
        action="store_true",
        default=False,
        dest="full",
        help="Restart in full mode (indexing + watcher + working memory), "
             "discarding the mode of the instance being restarted. Without any "
             "mode flag, restart reproduces that instance's mode — a "
             "--memory-only daemon comes back memory-only.",
    )
    p_restart.add_argument(
        "--no-ide-config",
        action="store_true",
        default=False,
        dest="no_ide_config",
        help=_NO_IDE_CONFIG_HELP,
    )
    p_restart.add_argument(
        "--foreground-fast",
        action="store_true",
        default=False,
        dest="foreground_fast",
        help="Restart without the indexer's resource governor (see vectr start --foreground-fast).",
    )

    p_forget = sub.add_parser("forget", help="Delete working-memory notes for a workspace")
    p_forget.add_argument("--path", default=_default_path)
    p_forget.add_argument("--port", type=int, default=_default_port)
    p_forget.add_argument(
        "--all", action="store_true",
        help="Delete notes across ALL workspaces (operates directly on SQLite, no server needed)",
    )

    p_watch = sub.add_parser("watch", help="Index workspace(s) and watch for changes (no MCP server)")
    p_watch.add_argument("workspace", nargs="?", default=None,
        help="Path to a .code-workspace file or a single workspace directory")
    p_watch.add_argument("--path", action="append", dest="paths", metavar="DIR")

    p_mcp_stdio = sub.add_parser(
        "mcp-stdio",
        help="Run the MCP server on stdio (no daemon, no port)",
        description=(
            "Foreground MCP server using the stdio transport: newline-delimited "
            "JSON-RPC 2.0 on stdin/stdout, one JSON object per line, no daemon "
            "and no port. For MCP clients and hosting platforms that spawn the "
            "server as a subprocess. Workspace defaults to VECTR_WORKSPACE or "
            "the current directory."
        ),
    )
    p_mcp_stdio.add_argument("workspace", nargs="?", default=None,
        help="Path to a workspace directory (default: $VECTR_WORKSPACE or cwd)")
    p_mcp_stdio.add_argument("--memory-only", action="store_true", default=False,
        dest="memory_only", help="Disable code indexing/watcher; memory tools only")
    p_mcp_stdio.add_argument("--search-only", action="store_true", default=False,
        dest="search_only", help="Disable the working-memory layer; search/locate/trace/map only")

    p_init = sub.add_parser(
        "init",
        help="Write CLAUDE.md and .mcp.json to a workspace (no server)",
        description=_IDE_CONFIG_WRITES_DISCLOSURE,
    )
    p_init.add_argument("--path", default=_default_path)
    p_init.add_argument(
        "--exclude", action="append", metavar="PATTERN", dest="exclude",
        help=_EXCLUDE_HELP,
    )
    p_init.add_argument(
        "--no-ide-config",
        action="store_true",
        default=False,
        dest="no_ide_config",
        help=_NO_IDE_CONFIG_HELP,
    )
    p_init.add_argument(
        "--style",
        # "memory-only" accepted for backward compatibility with pre-rename
        # scripts/muscle memory (UPG-TOOLSTYLE-LABEL-COLLISION renamed this
        # style label to "memory-first" so it never looks identical to the
        # unrelated memory_only *operating mode*); always normalized before
        # being written to .vectr/style — see cmd_init.
        choices=["additive", "directed", "memory-first", "memory-only"],
        default=None,
        help="Override adaptive instruction style (T14). Stored in .vectr/style.",
    )
    p_init.add_argument(
        "--reset-config",
        action="store_true",
        default=False,
        dest="reset_config",
        help="Remove all vectr blocks from IDE config files in the workspace.",
    )
    p_init.add_argument(
        "--hooks",
        action="store_true",
        default=False,
        help="Also write Claude Code hook entries (.claude/settings.json) for "
             "harness-injected vectr memory (SessionStart boot recall, etc.).",
    )

    p_hook = sub.add_parser(
        "hook",
        help="Emit Claude Code hook output, or record commit provenance (invoked by "
             "`vectr init --hooks` entries; not called directly)",
    )
    p_hook.add_argument("hook_event",
                        choices=["session-start", "user-prompt-submit", "pre-tool-use", "pre-compact", "post-commit"],
                        help="Which hook event to emit output for")

    p_index = sub.add_parser("index", help="(Re)index a directory or file")
    p_index.add_argument("--path", default=_default_path)
    p_index.add_argument("--port", type=int, default=_default_port)
    p_index.add_argument("--force", action="store_true", help="Force full re-index")

    p_search = sub.add_parser(
        "search",
        help="Semantic search",
        description=(
            "Semantic search over the indexed codebase. Each result's `score` "
            "is an absolute per-(query, result) relevance value in [0, 1], not "
            "a rank-derived percentile — it can legitimately be low even for "
            "the best available match if nothing in the codebase is a strong "
            "fit. A '--- Low confidence ---' banner is printed above the "
            "results when the query itself has no lexical anchor anywhere in "
            "the index, or the top result's own relevance score is below the "
            "configured floor (`ranking.notfound_floor.*` in config.yaml) — "
            "results are still shown in full even when it fires."
        ),
    )
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--n", type=int, default=10)
    p_search.add_argument("--language", help="Filter by language")
    p_search.add_argument("--port", type=int, default=_default_port)

    p_fetch = sub.add_parser(
        "fetch",
        help="Re-fetch chunks by exact id, verbatim",
        description=(
            "Deterministic re-fetch by chunk id — restores a chunk shown in an "
            "earlier search/locate/trace result after it has left context, with "
            "no re-search or file re-read. Ids not present in the index are "
            "reported as not found (the file likely changed since indexing)."
        ),
    )
    p_fetch.add_argument("ids", nargs="+", help="Chunk id(s), e.g. path/to/file.py:10-42")
    p_fetch.add_argument("--port", type=int, default=_default_port)

    p_remember = sub.add_parser("remember", help="Store a working-memory note (shell path to the note store)")
    p_remember.add_argument("content", help="The note content to store")
    p_remember.add_argument("--tags", action="append", metavar="TAG", help="Topic tag (repeatable)")
    p_remember.add_argument("--priority", choices=["high", "medium", "low"], default="medium")
    p_remember.add_argument("--kind", choices=["directive", "task", "gotcha", "finding", "reference", "decision", "operational"],
                            default="finding", help="Memory kind (controls injection policy)")
    p_remember.add_argument("--title", default="", help="Short label for index-tier display (optional; derived from first content line if empty)")
    p_remember.add_argument("--agent", default="", help="Optional agent/subagent identifier for multi-agent shared-memory attribution (e.g. 'coder-2'); never inferred")
    p_remember.add_argument("--path", default=_default_path)
    p_remember.add_argument("--port", type=int, default=_default_port)

    p_recall = sub.add_parser("recall", help="Print recalled working-memory notes to stdout")
    p_recall.add_argument("query", nargs="?", default=None, help="Semantic recall query (optional)")
    p_recall.add_argument("--tags", action="append", metavar="TAG", help="Filter by tag (repeatable)")
    p_recall.add_argument("--priority", choices=["high", "medium", "low"], default=None)
    p_recall.add_argument("--kind", choices=["directive", "task", "gotcha", "finding", "reference", "decision", "operational"],
                          default=None, help="Filter to one memory kind")
    p_recall.add_argument("--boot", action="store_true",
                          help="Boot mode: unconditional directives + high-priority tasks (for SessionStart hooks)")
    p_recall.add_argument("--min-similarity", type=float, default=None, dest="min_similarity",
                          help="Relevance cutoff [0..1]: drop semantic matches below this cosine similarity")
    p_recall.add_argument("--max-age-days", type=float, default=None, dest="max_age_days",
                          help="Time filter: only return notes created within this many days")
    p_recall.add_argument("--sort-by", choices=["relevance", "recency", "priority", "chronological"], default="relevance",
                          dest="sort_by", help="Sort order: relevance | recency | priority | chronological (oldest-first; index lines show the creation date)")
    p_recall.add_argument("--detail", choices=["index", "full"], default="index",
                          help="Detail level: 'index' = one-line summaries (default); 'full' = bodies")
    p_recall.add_argument("--id", type=int, default=None, dest="note_id",
                          help="Expand a single note by ID (returns full body)")
    p_recall.add_argument("--limit", type=int, default=10)
    p_recall.add_argument("--path", default=_default_path)
    p_recall.add_argument("--port", type=int, default=_default_port)

    p_resume = sub.add_parser(
        "resume",
        help="Print 'pick up where you left off' to stdout",
        description=(
            "Deterministic 'pick up where you left off': the most recent "
            "current-task note (the same selection SessionStart boot "
            "injection uses), the latest saved snapshot, and open gotchas "
            "with their file anchors. Not a search — reads existing state "
            "only, one call instead of several vectr recall calls."
        ),
    )
    p_resume.add_argument("--path", default=_default_path)
    p_resume.add_argument("--port", type=int, default=_default_port)

    p_status = sub.add_parser("status", help="Show status for a workspace")
    p_status.add_argument("--path", default=_default_path)
    p_status.add_argument("--port", type=int, default=_default_port)
    p_status.add_argument("--all", action="store_true", help="List all running instances")

    sub.add_parser(
        "key",
        help="Print a fresh high-entropy API key for authenticated / team deployments",
    )

    p_cache = sub.add_parser(
        "cache",
        help="Maintain the ~/.cache/vectr cache root",
        description=(
            "Cache-root hygiene. `vectr cache prune` removes empty per-workspace "
            "hash dirs left by path resolutions that never indexed or stored a "
            "note. Only empty dirs are removed; the model cache (models/), the "
            "legacy db/ layout's non-empty dirs, and any live instance's dir are "
            "never touched."
        ),
    )
    cache_sub = p_cache.add_subparsers(dest="cache_command")
    p_cache_prune = cache_sub.add_parser(
        "prune", help="Remove empty per-workspace cache dirs (safe; live instances skipped)"
    )
    p_cache_prune.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="List the dirs that would be removed without deleting anything",
    )

    p_connect = sub.add_parser(
        "connect",
        help="Configure this workspace's editor to use a REMOTE vectr instance "
             "(team / central server) instead of spawning a local daemon",
        description=(
            "Point the local editor's MCP config at a central vectr instance over "
            "the network, carrying the shared API key (and optional client label) as "
            "request headers. No local daemon is started. The remote instance serves "
            "its own indexed checkout and a shared working-memory store — a note one "
            "connected agent stores is recallable by every other."
        ),
    )
    p_connect.add_argument("--url", required=True,
                           help="Base URL of the remote instance, e.g. http://central-host:8765")
    p_connect.add_argument("--api-key", dest="api_key", default="",
                           help="Shared key for the remote instance (falls back to VECTR_API_KEY)")
    p_connect.add_argument("--label", default="",
                           help="Optional client attribution label for notes/audit (e.g. your name)")
    p_connect.add_argument("--path", default=_default_path,
                           help="Workspace directory to write the editor config into (default cwd)")

    p_proxy = sub.add_parser(
        "proxy",
        help="Run the experimental localhost Anthropic-shaped proxy (Proactive context)",
        description=(
            "Start a localhost proxy the agent harness targets with "
            "ANTHROPIC_BASE_URL. It forwards to the real Anthropic API "
            "transparently (streaming SSE + tool_use passthrough), forwarding "
            "the upstream API key untouched (never stored, never logged), and — "
            "when injection is on and the workspace daemon is running — appends "
            "deterministic proactive context after the last prompt-cache "
            "breakpoint. EXPERIMENTAL, off by default, localhost-only (a "
            "non-loopback bind is refused). To bypass it at any time, unset "
            "ANTHROPIC_BASE_URL."
        ),
    )
    p_proxy.add_argument("--path", default=_default_path,
                         help="Workspace whose daemon supplies injection context (default cwd)")
    p_proxy.add_argument("--host", default=None,
                         help="Proxy bind address (default from config; localhost only)")
    p_proxy.add_argument("--port", type=int, default=None,
                         help="Proxy listener port (default from config, e.g. 8785)")
    p_proxy.add_argument("--upstream", default=None,
                         help="Upstream base URL to forward to (default https://api.anthropic.com)")
    p_proxy.add_argument("--daemon-port", type=int, default=None, dest="daemon_port",
                         help="Port of the local vectr daemon to query for injection "
                              "(default: resolved from the workspace, else 8765)")
    p_proxy.add_argument("--no-inject", action="store_true", default=False, dest="no_inject",
                         help="Run as a pure transparent pass-through (no context injection)")

    args = parser.parse_args()
    dispatch = {
        "start":   cmd_start,
        "restart": cmd_restart,
        "watch":   cmd_watch,
        "mcp-stdio": cmd_mcp_stdio,
        "init":    cmd_init,
        "index":   cmd_index,
        "search":  cmd_search,
        "fetch":   cmd_fetch,
        "status":  cmd_status,
        "stop":    cmd_stop,
        "forget":  cmd_forget,
        "remember": cmd_remember,
        "recall":  cmd_recall,
        "resume":  cmd_resume,
        "hook":    cmd_hook,
        "key":     cmd_key,
        "cache":   cmd_cache,
        "connect": cmd_connect,
        "proxy":   cmd_proxy,
    }
    if args.command in dispatch:
        dispatch[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
