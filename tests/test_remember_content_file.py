"""Tests for vectr_remember's `content_file` parameter (MCP + REST) —
UPG-REMEMBER-MCP-LONG-PAYLOAD-PARSE-LOSS.

Layers covered, smallest to largest:
  - TestReadContentFile / TestResolveRememberContent: pure unit tests
    against agent.working_context_store._content_file — no service, no
    HTTP, no MCP dispatch. This is where every distinct error case (both/
    neither/escaping-path/missing-file/oversize/bad-UTF-8/empty) is pinned
    down at the smallest level.
  - TestMcpRememberContentFile*: the real MCP `vectr_remember` tool
    (integrations.mcp_server.handle_tools_call) against a REAL VectrService
    (memory-only, dummy embedder — no model load), proving the byte-
    identical round trip through vectr_recall end to end.
  - TestRestRememberContentFile*: the REST `POST /v1/remember` route
    (tests.conftest's client_real_memory fixture — real WorkingContextStore,
    mocked search) — a REST-specific pass, not just "the MCP test passed so
    REST must too" (this repo has a documented history of a /v1 route 500
    hiding behind a mock that returned the wrong type).
"""
from __future__ import annotations

import re

import pytest

from agent.working_context_store import (
    read_content_file,
    resolve_content_file_path,
    resolve_remember_content,
)
from integrations.mcp_server import handle_tools_call
from tests.test_memory_only_mode import _make_service


def _code_heavy_note(target_chars: int = 4200) -> str:
    """~4.2KB note body mirroring UPG-REMEMBER-MCP-LONG-PAYLOAD-PARSE-LOSS's
    own reproduction case: nested quoted code, backslash escape sequences,
    non-ASCII punctuation. Starts and ends on a non-whitespace character —
    so MCP's existing `content.strip()` is a no-op and the round trip
    below is byte-identical with no "modulo a trailing newline" caveat."""
    block = (
        'def handle_payload(raw: str) -> dict:\n'
        '    """Parses a nested JSON-in-JSON payload -- the exact shape that\n'
        '    corrupted mid-stream as an MCP tool-call argument. Quotes \\"like\n'
        '    this\\", backslashes \\\\, an em dash \u2014 an arrow \u2192 and\n'
        '    curly quotes \u201clike this\u201d."""\n'
        '    payload = "{\\"key\\": \\"value with \\\\n newline and \\\\t tab\\"}"\n'
        '    nested = {\'inner\': "a \'single\' and \\"double\\" value \u2014 done"}\n'
        '    return json.loads(payload), nested\n\n'
    )
    reps = target_chars // len(block) + 2
    text = (block * reps)[:target_chars]
    return text.rstrip() + "\n# end_of_note_marker"


def _note_id_from_confirmation(text: str) -> int:
    match = re.search(r"Stored note #(\d+)", text)
    assert match, f"no note id in confirmation text: {text!r}"
    return int(match.group(1))


# ---------------------------------------------------------------------------
# Pure unit tests: agent.working_context_store._content_file
# ---------------------------------------------------------------------------

class TestReadContentFile:
    def test_reads_utf8_file_verbatim(self, tmp_path) -> None:
        body = _code_heavy_note()
        (tmp_path / "note.txt").write_text(body, encoding="utf-8")
        assert read_content_file(str(tmp_path), "note.txt") == body

    def test_absolute_path_inside_workspace_is_accepted(self, tmp_path) -> None:
        f = tmp_path / "note.txt"
        f.write_text("hello world", encoding="utf-8")
        assert read_content_file(str(tmp_path), str(f)) == "hello world"

    def test_relative_escape_via_dotdot_is_rejected(self, tmp_path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (tmp_path / "secret.txt").write_text("nope", encoding="utf-8")
        with pytest.raises(ValueError, match="outside every workspace root"):
            read_content_file(str(workspace), "../secret.txt")

    def test_absolute_escape_outside_workspace_is_rejected(self, tmp_path, tmp_path_factory) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        outside = tmp_path_factory.mktemp("cf-outside") / "secret.txt"
        outside.write_text("nope", encoding="utf-8")
        with pytest.raises(ValueError, match="outside every workspace root"):
            read_content_file(str(workspace), str(outside))

    def test_missing_file_names_the_resolved_path(self, tmp_path) -> None:
        with pytest.raises(ValueError) as excinfo:
            read_content_file(str(tmp_path), "does-not-exist.txt")
        assert str(tmp_path / "does-not-exist.txt") in str(excinfo.value)
        assert "does not exist" in str(excinfo.value)

    def test_directory_path_is_rejected_as_not_a_regular_file(self, tmp_path) -> None:
        (tmp_path / "a_dir").mkdir()
        with pytest.raises(ValueError, match="not a regular file"):
            read_content_file(str(tmp_path), "a_dir")

    def test_oversize_file_is_rejected(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(
            "agent.working_context_store._content_file.MEMORY_WRITE_CONTENT_FILE_MAX_BYTES", 10,
        )
        (tmp_path / "big.txt").write_text("x" * 100, encoding="utf-8")
        with pytest.raises(ValueError, match="over the 10-byte limit"):
            read_content_file(str(tmp_path), "big.txt")

    def test_invalid_utf8_is_rejected(self, tmp_path) -> None:
        (tmp_path / "bad.txt").write_bytes(b"\xff\xfe\x00bad")
        with pytest.raises(ValueError, match="not valid UTF-8"):
            read_content_file(str(tmp_path), "bad.txt")

    def test_empty_file_is_rejected(self, tmp_path) -> None:
        (tmp_path / "empty.txt").write_bytes(b"")
        with pytest.raises(ValueError, match="is empty"):
            read_content_file(str(tmp_path), "empty.txt")


class TestResolveContentFilePath:
    def test_relative_path_resolves_under_workspace_root(self, tmp_path) -> None:
        resolved = resolve_content_file_path(str(tmp_path), "sub/note.txt")
        assert resolved == (tmp_path / "sub" / "note.txt").resolve()


class TestReadContentFileMultiRoot:
    """UPG-REMEMBER-CONTENT-FILE-EXTRA-ROOTS: a vectr instance can serve a
    primary workspace root plus one or more extra_roots (a multi-root IDE
    workspace) — content_file must accept a file under ANY served root, not
    only the primary, without weakening containment against every root."""

    def test_absolute_path_under_extra_root_is_accepted(self, tmp_path, tmp_path_factory) -> None:
        primary = tmp_path
        extra_root = tmp_path_factory.mktemp("extra-root")
        note_file = extra_root / "note.txt"
        note_file.write_text("body from an extra root", encoding="utf-8")
        assert read_content_file(
            str(primary), str(note_file), extra_roots=[str(extra_root)],
        ) == "body from an extra root"

    def test_path_outside_every_served_root_is_still_rejected(self, tmp_path, tmp_path_factory) -> None:
        primary = tmp_path
        extra_root = tmp_path_factory.mktemp("extra-root")
        outside = tmp_path_factory.mktemp("cf-outside") / "secret.txt"
        outside.write_text("nope", encoding="utf-8")
        with pytest.raises(ValueError, match="outside every workspace root"):
            read_content_file(str(primary), str(outside), extra_roots=[str(extra_root)])

    def test_dotdot_escape_from_extra_root_is_still_rejected(self, tmp_path, tmp_path_factory) -> None:
        primary = tmp_path
        extra_root = tmp_path_factory.mktemp("extra-root")
        outside_root = tmp_path_factory.mktemp("cf-outside")
        (outside_root / "secret.txt").write_text("nope", encoding="utf-8")
        # pytest's tmp_path_factory places every mktemp'd dir as an
        # immediate sibling under the same base temp dir, so ".." from
        # inside extra_root reaches outside_root -- exactly the same
        # escape shape as a ".." escape from the primary root, now
        # exercised from an extra root instead.
        escaping = str(extra_root / ".." / outside_root.name / "secret.txt")
        with pytest.raises(ValueError, match="outside every workspace root"):
            read_content_file(str(primary), escaping, extra_roots=[str(extra_root)])

    def test_symlink_inside_extra_root_pointing_outside_all_roots_is_rejected(
        self, tmp_path, tmp_path_factory,
    ) -> None:
        primary = tmp_path
        extra_root = tmp_path_factory.mktemp("extra-root")
        outside = tmp_path_factory.mktemp("cf-outside") / "secret.txt"
        outside.write_text("nope", encoding="utf-8")
        link = extra_root / "link.txt"
        link.symlink_to(outside)
        with pytest.raises(ValueError, match="outside every workspace root"):
            read_content_file(str(primary), str(link), extra_roots=[str(extra_root)])

    def test_relative_path_is_never_resolved_against_extra_roots(self, tmp_path, tmp_path_factory) -> None:
        """Documented rule: a relative raw_path resolves against the PRIMARY
        root only -- it is never trial-resolved against extra_roots, so a
        file that exists ONLY under an extra root is not found by a
        relative path even though it would be found by the absolute
        equivalent (see test_absolute_path_under_extra_root_is_accepted)."""
        primary = tmp_path
        extra_root = tmp_path_factory.mktemp("extra-root")
        (extra_root / "note.txt").write_text("only in extra root", encoding="utf-8")
        with pytest.raises(ValueError, match="does not exist"):
            read_content_file(str(primary), "note.txt", extra_roots=[str(extra_root)])


class TestReadContentFileAdditionalReadableRoots:
    """UPG-REMEMBER-CONTENT-FILE-PATH-REFUSAL: the operator can opt in to a
    configured set of additional absolute roots content_file is allowed to
    read from, BEYOND the primary workspace root and any extra_roots — for
    a directory the workspace does not own but agents are instructed to
    write to (a harness scratchpad dir, a per-session temp dir, etc.). The
    trust decision is operator-only (config or per-call argument passed
    down from a service built by the operator), never a flag the caller
    can simply always pass to widen containment."""

    def test_default_config_has_no_additional_readable_roots(self) -> None:
        """Off by default — the boundary is unchanged until the operator
        explicitly lists at least one root in
        `memory_write.content_file.additional_readable_roots`."""
        from agent.config import MEMORY_WRITE_CONTENT_FILE_ADDITIONAL_READABLE_ROOTS
        assert MEMORY_WRITE_CONTENT_FILE_ADDITIONAL_READABLE_ROOTS == ()

    def test_absolute_path_under_additional_root_is_accepted(
        self, tmp_path, tmp_path_factory,
    ) -> None:
        primary = tmp_path
        scratch = tmp_path_factory.mktemp("scratch")
        note_file = scratch / "note.txt"
        note_file.write_text("body from the harness scratchpad", encoding="utf-8")
        assert read_content_file(
            str(primary), str(note_file), additional_readable_roots=[str(scratch)],
        ) == "body from the harness scratchpad"

    def test_path_outside_every_accepted_root_is_still_rejected(
        self, tmp_path, tmp_path_factory,
    ) -> None:
        primary = tmp_path
        scratch = tmp_path_factory.mktemp("scratch")
        outside = tmp_path_factory.mktemp("cf-outside") / "secret.txt"
        outside.write_text("nope", encoding="utf-8")
        with pytest.raises(ValueError, match="outside every workspace root"):
            read_content_file(
                str(primary), str(outside), additional_readable_roots=[str(scratch)],
            )

    def test_dotdot_escape_from_additional_root_is_still_rejected(
        self, tmp_path, tmp_path_factory,
    ) -> None:
        primary = tmp_path
        scratch = tmp_path_factory.mktemp("scratch")
        outside_root = tmp_path_factory.mktemp("cf-outside")
        (outside_root / "secret.txt").write_text("nope", encoding="utf-8")
        # Same shape as the extra_root test above, now from an additional
        # readable root: a `..` from inside the additional root that
        # reaches a sibling of the additional root must still be rejected.
        escaping = str(scratch / ".." / outside_root.name / "secret.txt")
        with pytest.raises(ValueError, match="outside every workspace root"):
            read_content_file(
                str(primary), escaping, additional_readable_roots=[str(scratch)],
            )

    def test_symlink_inside_additional_root_pointing_outside_all_roots_is_rejected(
        self, tmp_path, tmp_path_factory,
    ) -> None:
        primary = tmp_path
        scratch = tmp_path_factory.mktemp("scratch")
        outside = tmp_path_factory.mktemp("cf-outside") / "secret.txt"
        outside.write_text("nope", encoding="utf-8")
        link = scratch / "link.txt"
        link.symlink_to(outside)
        with pytest.raises(ValueError, match="outside every workspace root"):
            read_content_file(
                str(primary), str(link), additional_readable_roots=[str(scratch)],
            )

    def test_relative_path_is_never_resolved_against_additional_roots(
        self, tmp_path, tmp_path_factory,
    ) -> None:
        """Documented rule (same as extra_roots): a relative raw_path
        resolves against the PRIMARY root only — it is never trial-resolved
        against additional_readable_roots, so a file that exists ONLY
        under an additional root is not found by a relative path even
        though it would be found by the absolute equivalent."""
        primary = tmp_path
        scratch = tmp_path_factory.mktemp("scratch")
        (scratch / "note.txt").write_text("only in scratch", encoding="utf-8")
        with pytest.raises(ValueError, match="does not exist"):
            read_content_file(
                str(primary), "note.txt", additional_readable_roots=[str(scratch)],
            )

    def test_empty_additional_readable_roots_explicit_disables_default(
        self, tmp_path, tmp_path_factory, monkeypatch,
    ) -> None:
        """An explicit `additional_readable_roots=[]` is a "no additional
        roots this call" — distinct from the default None that falls back
        to the config singleton. Even with the config singleton populated
        (here monkeypatched to include a real root), an explicit empty
        list on a single call must NOT widen containment for that call."""
        from agent.working_context_store import _content_file as _cf
        primary = tmp_path
        scratch = tmp_path_factory.mktemp("scratch")
        note_file = scratch / "note.txt"
        note_file.write_text("body", encoding="utf-8")
        monkeypatch.setattr(
            _cf, "MEMORY_WRITE_CONTENT_FILE_ADDITIONAL_READABLE_ROOTS", (str(scratch),),
        )
        with pytest.raises(ValueError, match="outside every workspace root"):
            read_content_file(
                str(primary), str(note_file), additional_readable_roots=[],
            )

    def test_configured_default_additional_root_is_honored(
        self, tmp_path, tmp_path_factory, monkeypatch,
    ) -> None:
        """`additional_readable_roots=None` (the default) reads from the
        config singleton — a calling agent that did NOT opt in to anything
        still gets the operator-configured set. The MCP and REST dispatch
        layers both pass None, so the config is the only thing that
        matters in production."""
        from agent.working_context_store import _content_file as _cf
        primary = tmp_path
        scratch = tmp_path_factory.mktemp("scratch")
        note_file = scratch / "note.txt"
        note_file.write_text("body from config-default root", encoding="utf-8")
        monkeypatch.setattr(
            _cf, "MEMORY_WRITE_CONTENT_FILE_ADDITIONAL_READABLE_ROOTS", (str(scratch),),
        )
        # No `additional_readable_roots` kwarg — falls back to the
        # config-default (here monkeypatched to include the scratch dir).
        assert read_content_file(str(primary), str(note_file)) == "body from config-default root"


class TestResolveRememberContent:
    def test_content_only_is_returned_unmodified(self, tmp_path) -> None:
        assert resolve_remember_content(str(tmp_path), "hello", None) == "hello"

    def test_content_file_only_reads_the_file(self, tmp_path) -> None:
        (tmp_path / "n.txt").write_text("from file", encoding="utf-8")
        assert resolve_remember_content(str(tmp_path), None, "n.txt") == "from file"

    def test_both_content_and_content_file_is_rejected(self, tmp_path) -> None:
        (tmp_path / "n.txt").write_text("x", encoding="utf-8")
        with pytest.raises(ValueError, match="mutually exclusive"):
            resolve_remember_content(str(tmp_path), "hello", "n.txt")

    def test_neither_content_nor_content_file_is_rejected(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="content or content_file is required"):
            resolve_remember_content(str(tmp_path), None, None)

    def test_whitespace_only_content_is_treated_as_absent(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="content or content_file is required"):
            resolve_remember_content(str(tmp_path), "   ", None)

    def test_content_file_under_additional_root_resolves(
        self, tmp_path, tmp_path_factory,
    ) -> None:
        scratch = tmp_path_factory.mktemp("scratch")
        (scratch / "n.txt").write_text("from scratch", encoding="utf-8")
        assert resolve_remember_content(
            str(tmp_path), None, str(scratch / "n.txt"),
            additional_readable_roots=[str(scratch)],
        ) == "from scratch"

    def test_content_file_outside_every_accepted_root_is_rejected(
        self, tmp_path, tmp_path_factory,
    ) -> None:
        scratch = tmp_path_factory.mktemp("scratch")
        outside = tmp_path_factory.mktemp("cf-outside") / "n.txt"
        outside.write_text("nope", encoding="utf-8")
        with pytest.raises(ValueError, match="outside every workspace root"):
            resolve_remember_content(
                str(tmp_path), None, str(outside),
                additional_readable_roots=[str(scratch)],
            )


# ---------------------------------------------------------------------------
# MCP vectr_remember (real VectrService, memory-only mode, dummy embedder)
# ---------------------------------------------------------------------------

class TestMcpRememberContentFileRoundTrip:
    def test_round_trips_byte_identically_via_mcp_recall(self, tmp_path, monkeypatch) -> None:
        svc = _make_service(tmp_path, monkeypatch, memory_only=True)
        body = _code_heavy_note()
        (tmp_path / "note_body.txt").write_text(body, encoding="utf-8")

        result = handle_tools_call(
            "vectr_remember",
            {"content_file": "note_body.txt", "kind": "finding", "tags": ["payload-test"]},
            svc,
        )
        assert result["isError"] is False
        note_id = _note_id_from_confirmation(result["content"][0]["text"])

        stored = svc._context_store.get_note(svc._workspace_root, note_id)
        assert stored.content == body

        recall_result = handle_tools_call("vectr_recall", {"note_id": note_id}, svc)
        assert body in recall_result["content"][0]["text"]

    def test_combines_with_kind_tags_priority_title_agent(self, tmp_path, monkeypatch) -> None:
        svc = _make_service(tmp_path, monkeypatch, memory_only=True)
        (tmp_path / "n.txt").write_text("gotcha body text", encoding="utf-8")
        result = handle_tools_call(
            "vectr_remember",
            {
                "content_file": "n.txt", "kind": "gotcha", "tags": ["x", "y"],
                "priority": "high", "title": "custom title", "agent": "lane-p1-payload",
            },
            svc,
        )
        assert result["isError"] is False
        note_id = _note_id_from_confirmation(result["content"][0]["text"])
        stored = svc._context_store.get_note(svc._workspace_root, note_id)
        assert stored.content == "gotcha body text"
        assert stored.kind == "gotcha"
        assert stored.tags == ["x", "y"]
        assert stored.priority == "high"
        assert stored.title == "custom title"
        assert stored.author_id == "lane-p1-payload"

    def test_mcp_strips_trailing_whitespace_same_as_inline_content(self, tmp_path, monkeypatch) -> None:
        """UPG-REMEMBER-MCP-LONG-PAYLOAD-PARSE-LOSS design constraint: content_file
        must behave identically to inline content at each surface. The MCP
        vectr_remember handler already strips leading/trailing whitespace off
        an inline `content` argument (unchanged by this feature) — a body
        sourced from content_file goes through the exact same strip."""
        svc = _make_service(tmp_path, monkeypatch, memory_only=True)
        (tmp_path / "n.txt").write_text("hello world\n\n", encoding="utf-8")
        result = handle_tools_call("vectr_remember", {"content_file": "n.txt"}, svc)
        note_id = _note_id_from_confirmation(result["content"][0]["text"])
        stored = svc._context_store.get_note(svc._workspace_root, note_id)
        assert stored.content == "hello world"


class TestMcpRememberContentFileErrors:
    @staticmethod
    def _mock_service(ws_root: str):
        from unittest.mock import MagicMock
        svc = MagicMock()
        svc._workspace_root = ws_root
        svc.search_only = False
        return svc

    def test_both_content_and_content_file_is_an_mcp_error(self, tmp_path) -> None:
        (tmp_path / "n.txt").write_text("x", encoding="utf-8")
        svc = self._mock_service(str(tmp_path))
        result = handle_tools_call(
            "vectr_remember", {"content": "hi", "content_file": "n.txt"}, svc,
        )
        assert result["isError"] is True
        assert "mutually exclusive" in result["content"][0]["text"]
        svc.remember_with_extras.assert_not_called()

    def test_neither_content_nor_content_file_is_an_mcp_error(self, tmp_path) -> None:
        svc = self._mock_service(str(tmp_path))
        result = handle_tools_call("vectr_remember", {}, svc)
        assert result["isError"] is True
        assert "content or content_file is required" in result["content"][0]["text"]

    def test_escaping_path_is_an_mcp_error(self, tmp_path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (tmp_path / "secret.txt").write_text("nope", encoding="utf-8")
        svc = self._mock_service(str(workspace))
        result = handle_tools_call("vectr_remember", {"content_file": "../secret.txt"}, svc)
        assert result["isError"] is True
        assert "outside every workspace root" in result["content"][0]["text"]

    def test_missing_file_is_an_mcp_error(self, tmp_path) -> None:
        svc = self._mock_service(str(tmp_path))
        result = handle_tools_call("vectr_remember", {"content_file": "nope.txt"}, svc)
        assert result["isError"] is True
        assert "does not exist" in result["content"][0]["text"]

    def test_oversize_file_is_an_mcp_error(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(
            "agent.working_context_store._content_file.MEMORY_WRITE_CONTENT_FILE_MAX_BYTES", 10,
        )
        (tmp_path / "big.txt").write_text("x" * 100, encoding="utf-8")
        svc = self._mock_service(str(tmp_path))
        result = handle_tools_call("vectr_remember", {"content_file": "big.txt"}, svc)
        assert result["isError"] is True
        assert "over the 10-byte limit" in result["content"][0]["text"]

    def test_bad_utf8_file_is_an_mcp_error(self, tmp_path) -> None:
        (tmp_path / "bad.txt").write_bytes(b"\xff\xfe\x00bad")
        svc = self._mock_service(str(tmp_path))
        result = handle_tools_call("vectr_remember", {"content_file": "bad.txt"}, svc)
        assert result["isError"] is True
        assert "not valid UTF-8" in result["content"][0]["text"]


class TestMcpRememberContentFileMultiRoot:
    """A vectr instance can serve a primary root plus extra_roots (a
    multi-root IDE workspace) -- content_file must be usable for a file
    under any served root, end to end through the real vectr_remember tool,
    not only at the resolver-function level."""

    def test_file_under_extra_root_is_accepted_via_mcp(
        self, tmp_path, tmp_path_factory, monkeypatch,
    ) -> None:
        svc = _make_service(tmp_path, monkeypatch, memory_only=True)
        extra_root = tmp_path_factory.mktemp("extra-root")
        svc._extra_roots = [str(extra_root)]
        note_file = extra_root / "note.txt"
        note_file.write_text("multi-root body", encoding="utf-8")

        result = handle_tools_call("vectr_remember", {"content_file": str(note_file)}, svc)
        assert result["isError"] is False
        note_id = _note_id_from_confirmation(result["content"][0]["text"])
        stored = svc._context_store.get_note(svc._workspace_root, note_id)
        assert stored.content == "multi-root body"

    def test_file_outside_every_served_root_is_still_an_mcp_error(
        self, tmp_path, tmp_path_factory,
    ) -> None:
        svc = TestMcpRememberContentFileErrors._mock_service(str(tmp_path))
        svc._extra_roots = [str(tmp_path_factory.mktemp("extra-root"))]
        outside = tmp_path_factory.mktemp("cf-outside") / "secret.txt"
        outside.write_text("nope", encoding="utf-8")
        result = handle_tools_call("vectr_remember", {"content_file": str(outside)}, svc)
        assert result["isError"] is True
        assert "outside every workspace root" in result["content"][0]["text"]


class TestMcpRememberContentFileAdditionalReadableRoots:
    """UPG-REMEMBER-CONTENT-FILE-PATH-REFUSAL: end-to-end proof through the
    real MCP `vectr_remember` tool that the operator-configured additional
    readable roots are honored — and that the dispatcher does NOT pass any
    per-call widening argument, so the boundary is config-only, not
    caller-controlled."""

    def test_file_under_additional_root_is_accepted_via_mcp(
        self, tmp_path, tmp_path_factory, monkeypatch,
    ) -> None:
        from agent.working_context_store import _content_file as _cf
        svc = _make_service(tmp_path, monkeypatch, memory_only=True)
        scratch = tmp_path_factory.mktemp("scratch")
        # The MCP dispatcher does NOT take a per-call additional_readable_roots
        # argument — the only way for the path to be accepted is via the
        # config singleton. This pins the contract that an operator opts in
        # by setting `memory_write.content_file.additional_readable_roots`
        # in config.yaml, and both surfaces pick it up.
        monkeypatch.setattr(
            _cf, "MEMORY_WRITE_CONTENT_FILE_ADDITIONAL_READABLE_ROOTS",
            (str(scratch),),
        )
        note_file = scratch / "note.txt"
        note_file.write_text("scratch body", encoding="utf-8")

        result = handle_tools_call("vectr_remember", {"content_file": str(note_file)}, svc)
        assert result["isError"] is False
        note_id = _note_id_from_confirmation(result["content"][0]["text"])
        stored = svc._context_store.get_note(svc._workspace_root, note_id)
        assert stored.content == "scratch body"

    def test_file_outside_every_accepted_root_is_still_an_mcp_error(
        self, tmp_path, tmp_path_factory, monkeypatch,
    ) -> None:
        from agent.working_context_store import _content_file as _cf
        svc = _make_service(tmp_path, monkeypatch, memory_only=True)
        scratch = tmp_path_factory.mktemp("scratch")
        monkeypatch.setattr(
            _cf, "MEMORY_WRITE_CONTENT_FILE_ADDITIONAL_READABLE_ROOTS",
            (str(scratch),),
        )
        outside = tmp_path_factory.mktemp("cf-outside") / "secret.txt"
        outside.write_text("nope", encoding="utf-8")
        result = handle_tools_call("vectr_remember", {"content_file": str(outside)}, svc)
        assert result["isError"] is True
        assert "outside every workspace root" in result["content"][0]["text"]


# ---------------------------------------------------------------------------
# REST POST /v1/remember — its own coverage, not inferred from the MCP pass
# ---------------------------------------------------------------------------

class TestRestRememberContentFileRoundTrip:
    def test_round_trips_byte_identically_via_rest(self, client_real_memory, tmp_path) -> None:
        body = _code_heavy_note()
        (tmp_path / "note_body.txt").write_text(body, encoding="utf-8")

        resp = client_real_memory.post("/v1/remember", json={"content_file": "note_body.txt"})
        assert resp.status_code == 200
        note_id = resp.json()["note_id"]

        full = client_real_memory.post(
            "/v1/recall", json={"note_id": note_id, "detail": "full"},
        ).json()["notes"]
        assert body in full

    def test_rest_does_not_strip_trailing_whitespace_same_as_inline_content(
        self, client_real_memory, tmp_path,
    ) -> None:
        """REST's existing /v1/remember handling of an inline `content` body
        applies no stripping (unlike MCP) — content_file must match that,
        not introduce a third convention."""
        (tmp_path / "n.txt").write_text("hello world\n\n", encoding="utf-8")
        resp = client_real_memory.post("/v1/remember", json={"content_file": "n.txt"})
        assert resp.status_code == 200
        note_id = resp.json()["note_id"]
        full = client_real_memory.post(
            "/v1/recall", json={"note_id": note_id, "detail": "full"},
        ).json()["notes"]
        assert "hello world\n\n" in full

    def test_content_file_combines_with_other_parameters_via_rest(
        self, client_real_memory, tmp_path,
    ) -> None:
        (tmp_path / "n.txt").write_text("gotcha body text", encoding="utf-8")
        resp = client_real_memory.post("/v1/remember", json={
            "content_file": "n.txt", "kind": "gotcha", "tags": ["x", "y"],
            "priority": "high", "title": "custom title", "agent": "lane-p1-payload",
        })
        assert resp.status_code == 200


class TestRestRememberContentFileErrors:
    def test_both_content_and_content_file_returns_422(self, client_real_memory, tmp_path) -> None:
        (tmp_path / "n.txt").write_text("x", encoding="utf-8")
        resp = client_real_memory.post(
            "/v1/remember", json={"content": "hi", "content_file": "n.txt"},
        )
        assert resp.status_code == 422
        assert resp.json()["detail"]["error"] == "invalid_memory_object"
        assert "mutually exclusive" in resp.json()["detail"]["detail"]

    def test_neither_content_nor_content_file_returns_422(self, client_real_memory) -> None:
        resp = client_real_memory.post("/v1/remember", json={})
        assert resp.status_code == 422
        assert resp.json()["detail"]["error"] == "invalid_memory_object"
        assert "content or content_file is required" in resp.json()["detail"]["detail"]

    def test_escaping_path_returns_422(self, client_real_memory, tmp_path, tmp_path_factory) -> None:
        outside = tmp_path_factory.mktemp("cf-outside") / "secret.txt"
        outside.write_text("nope", encoding="utf-8")
        resp = client_real_memory.post("/v1/remember", json={"content_file": str(outside)})
        assert resp.status_code == 422
        assert "outside every workspace root" in resp.json()["detail"]["detail"]

    def test_missing_file_returns_422_naming_resolved_path(self, client_real_memory, tmp_path) -> None:
        resp = client_real_memory.post("/v1/remember", json={"content_file": "nope.txt"})
        assert resp.status_code == 422
        detail = resp.json()["detail"]["detail"]
        assert "does not exist" in detail
        assert str(tmp_path / "nope.txt") in detail

    def test_oversize_file_returns_422(self, client_real_memory, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(
            "agent.working_context_store._content_file.MEMORY_WRITE_CONTENT_FILE_MAX_BYTES", 10,
        )
        (tmp_path / "big.txt").write_text("x" * 100, encoding="utf-8")
        resp = client_real_memory.post("/v1/remember", json={"content_file": "big.txt"})
        assert resp.status_code == 422
        assert "over the 10-byte limit" in resp.json()["detail"]["detail"]

    def test_bad_utf8_file_returns_422(self, client_real_memory, tmp_path) -> None:
        (tmp_path / "bad.txt").write_bytes(b"\xff\xfe\x00bad")
        resp = client_real_memory.post("/v1/remember", json={"content_file": "bad.txt"})
        assert resp.status_code == 422
        assert "not valid UTF-8" in resp.json()["detail"]["detail"]

    def test_missing_content_still_returns_422_with_empty_body(self, client_real_memory) -> None:
        """Pre-existing REST contract (content used to be pydantic-required
        with min_length=1): an entirely empty JSON body must still 422, now
        via resolve_remember_content's "neither" branch instead of pydantic's
        own field validation, since content became optional to allow
        content_file-only calls."""
        resp = client_real_memory.post("/v1/remember", json={})
        assert resp.status_code == 422


class TestRestRememberContentFileMultiRoot:
    """Mirrors TestMcpRememberContentFileMultiRoot for the REST surface --
    same multi-root acceptance/rejection behaviour, exercised through the
    real /v1/remember route rather than assumed from the MCP pass."""

    def test_file_under_extra_root_is_accepted_via_rest(
        self, client_real_memory, tmp_path_factory,
    ) -> None:
        extra_root = tmp_path_factory.mktemp("extra-root")
        client_real_memory.app.state.service._extra_roots = [str(extra_root)]
        note_file = extra_root / "note.txt"
        note_file.write_text("multi-root body", encoding="utf-8")

        resp = client_real_memory.post("/v1/remember", json={"content_file": str(note_file)})
        assert resp.status_code == 200
        note_id = resp.json()["note_id"]
        full = client_real_memory.post(
            "/v1/recall", json={"note_id": note_id, "detail": "full"},
        ).json()["notes"]
        assert "multi-root body" in full

    def test_file_outside_every_served_root_still_returns_422(
        self, client_real_memory, tmp_path_factory,
    ) -> None:
        extra_root = tmp_path_factory.mktemp("extra-root")
        client_real_memory.app.state.service._extra_roots = [str(extra_root)]
        outside = tmp_path_factory.mktemp("cf-outside") / "secret.txt"
        outside.write_text("nope", encoding="utf-8")

        resp = client_real_memory.post("/v1/remember", json={"content_file": str(outside)})
        assert resp.status_code == 422
        assert "outside every workspace root" in resp.json()["detail"]["detail"]


class TestRestRememberContentFileAdditionalReadableRoots:
    """UPG-REMEMBER-CONTENT-FILE-PATH-REFUSAL: REST-side mirror of
    TestMcpRememberContentFileAdditionalReadableRoots. Both surfaces
    resolve through the same `resolve_remember_content` and read the
    config singleton for `additional_readable_roots` — same configuration
    knob, identical behaviour, no per-call widening."""

    def test_file_under_additional_root_is_accepted_via_rest(
        self, client_real_memory, tmp_path_factory, monkeypatch,
    ) -> None:
        from agent.working_context_store import _content_file as _cf
        scratch = tmp_path_factory.mktemp("scratch")
        monkeypatch.setattr(
            _cf, "MEMORY_WRITE_CONTENT_FILE_ADDITIONAL_READABLE_ROOTS",
            (str(scratch),),
        )
        note_file = scratch / "note.txt"
        note_file.write_text("scratch body via REST", encoding="utf-8")

        resp = client_real_memory.post("/v1/remember", json={"content_file": str(note_file)})
        assert resp.status_code == 200
        note_id = resp.json()["note_id"]
        full = client_real_memory.post(
            "/v1/recall", json={"note_id": note_id, "detail": "full"},
        ).json()["notes"]
        assert "scratch body via REST" in full

    def test_file_outside_every_accepted_root_still_returns_422(
        self, client_real_memory, tmp_path_factory, monkeypatch,
    ) -> None:
        from agent.working_context_store import _content_file as _cf
        scratch = tmp_path_factory.mktemp("scratch")
        monkeypatch.setattr(
            _cf, "MEMORY_WRITE_CONTENT_FILE_ADDITIONAL_READABLE_ROOTS",
            (str(scratch),),
        )
        outside = tmp_path_factory.mktemp("cf-outside") / "secret.txt"
        outside.write_text("nope", encoding="utf-8")

        resp = client_real_memory.post("/v1/remember", json={"content_file": str(outside)})
        assert resp.status_code == 422
        assert "outside every workspace root" in resp.json()["detail"]["detail"]
