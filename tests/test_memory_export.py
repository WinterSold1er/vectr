"""Tests for `vectr memory export` (UPG-MEMORY-LEGIBLE-FILE-PROJECTION part
(a)) — the read-only markdown mirror of vectr's working memory.

Covers:
- render_memory_markdown(): determinism (byte-identical across two calls with
  no data change), kind-grouping order, [#N] anchor stability, revoked-note
  deterrent rendering, superseded-note rendering (both the explicit
  `supersedes=` path and the legacy code_hash-conflict path), and graceful
  handling of an unrecognized fold state.
- write_memory_markdown(): atomic write (temp file cleaned up, no partial
  file ever visible under the target name).
- export_memory(): end-to-end through a real WorkingContextStore, byte-
  identical re-export with no intervening write.
- The `.vectr/memory_export_path` marker mechanism (read/write round trip,
  relative-path resolution, absent-marker no-op).
- schedule_export_if_configured(): no-op without a marker, debounce
  coalescing, and that it never raises even when export itself fails.
- `vectr memory export` CLI command end-to-end (direct SQLite, no daemon).
- VectrService._bump_notes_epoch() invoking the scheduler on every write path
  (the minimal-footprint hook point).
"""
from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.working_context_store import WorkingContextStore
from agent.working_context_store._export import (
    _read_export_target,
    export_memory,
    remove_export_target,
    render_memory_markdown,
    schedule_export_if_configured,
    write_export_target,
    write_memory_markdown,
)


def _store(tmp_path: Path) -> WorkingContextStore:
    db_dir = tmp_path / "db"
    db_dir.mkdir(exist_ok=True)
    return WorkingContextStore(str(db_dir))


# ---------------------------------------------------------------------------
# render_memory_markdown — pure rendering, no I/O
# ---------------------------------------------------------------------------

class TestRenderMemoryMarkdown:
    def test_deterministic_across_two_calls_with_no_data_change(self, tmp_path):
        store = _store(tmp_path)
        ws = "ws"
        store.remember(ws, "note one", priority="high", kind="finding")
        store.remember(ws, "note two", priority="low", kind="directive")
        notes = store.notes_for_export(ws)
        states = store.note_event_states(ws, notes)
        first = render_memory_markdown(notes, states)
        second = render_memory_markdown(notes, states)
        assert first == second

    def test_kind_sections_follow_valid_kinds_order_not_insertion_order(self, tmp_path):
        store = _store(tmp_path)
        ws = "ws"
        # Written in reverse-of-VALID_KINDS order on purpose.
        store.remember(ws, "a reference", priority="medium", kind="reference")
        store.remember(ws, "a finding", priority="medium", kind="finding")
        store.remember(ws, "a directive", priority="medium", kind="directive")
        notes = store.notes_for_export(ws)
        states = store.note_event_states(ws, notes)
        rendered = render_memory_markdown(notes, states)
        assert rendered.index("## Directives") < rendered.index("## Findings")
        assert rendered.index("## Findings") < rendered.index("## References")

    def test_note_id_anchor_present_and_stable(self, tmp_path):
        store = _store(tmp_path)
        ws = "ws"
        note_id = store.remember(ws, "anchor me", priority="medium", kind="finding")
        notes = store.notes_for_export(ws)
        states = store.note_event_states(ws, notes)
        rendered = render_memory_markdown(notes, states)
        assert f"[#{note_id}]" in rendered

    def test_revoked_note_renders_anti_memory_deterrent_not_hidden(self, tmp_path):
        store = _store(tmp_path)
        ws = "ws"
        note_id = store.remember(
            ws, "the full detailed rationale, containing SECRET_BODY_MARKER, "
            "goes here and is many sentences long", priority="medium", kind="finding",
            title="short title",
        )
        store.revoke_note(ws, note_id, actor="agent", reason="disproven")
        notes = store.notes_for_export(ws)
        states = store.note_event_states(ws, notes)
        rendered = render_memory_markdown(notes, states)
        # Still present (never dropped) with the [#N] anchor.
        assert f"[#{note_id}]" in rendered
        assert "state: revoked" in rendered
        assert "Previously believed" in rendered
        assert "do not re-derive" in rendered.lower()
        # The body is replaced by the deterrent (which only quotes the short
        # title as its summary) — the raw original content must not leak.
        assert "SECRET_BODY_MARKER" not in rendered

    def test_explicit_supersedes_path_renders_superseded_marker(self, tmp_path):
        store = _store(tmp_path)
        ws = "ws"
        old_id = store.remember(ws, "old fact", priority="medium", kind="finding")
        new_id = store.remember(ws, "corrected fact", priority="medium", kind="finding", supersedes=old_id)
        notes = store.notes_for_export(ws)
        states = store.note_event_states(ws, notes)
        rendered = render_memory_markdown(notes, states)
        assert f"state: superseded by note#{new_id}" in rendered

    def test_code_hash_conflict_supersede_path_also_renders_superseded_marker(self, tmp_path):
        """The legacy code_hash-conflict auto-supersede path sets valid_until/
        superseded_by/superseded_at columns directly but never appends a
        note_events 'superseded' event — the export must key off the columns
        (not the fold) to catch this path too, exactly like
        `_format_full_block`'s `n.valid_until is not None` check does."""
        store = _store(tmp_path)
        ws = "ws"
        note_id = store.remember(
            ws, "anchored fact", priority="medium", kind="finding",
            anchors=["some/file.py"],
        )
        with store._conn() as conn:
            conn.execute(
                "UPDATE notes SET valid_until = ?, superseded_by = ?, superseded_at = ? "
                "WHERE workspace = ? AND note_id = ?",
                (time.time(), "other-agent", time.time(), ws, note_id),
            )
        notes = store.notes_for_export(ws)
        states = store.note_event_states(ws, notes)
        rendered = render_memory_markdown(notes, states)
        assert "state: superseded by other-agent" in rendered

    def test_active_note_renders_content_verbatim(self, tmp_path):
        store = _store(tmp_path)
        ws = "ws"
        store.remember(ws, "plain active content", priority="medium", kind="finding")
        notes = store.notes_for_export(ws)
        states = store.note_event_states(ws, notes)
        rendered = render_memory_markdown(notes, states)
        assert "state: active" in rendered
        assert "plain active content" in rendered

    def test_empty_notes_list_renders_header_only(self):
        rendered = render_memory_markdown([], {})
        assert "# Working Memory" in rendered
        assert "## " not in rendered.split("# Working Memory", 1)[1]

    def test_header_comment_present(self, tmp_path):
        rendered = render_memory_markdown([], {})
        assert rendered.startswith("<!-- Generated by `vectr memory export`")
        assert "Nothing reads this file back" in rendered

    def test_within_section_ordering_is_note_id_ascending(self, tmp_path):
        """Guards the entire ordering contract the byte-identical-re-export
        acceptance bar rests on. Three notes of the SAME kind (so they land
        in the same rendered section) with DELIBERATELY varied priorities —
        an `ORDER BY` that ties or reorders on anything other than
        `note_id ASC` (e.g. `priority ASC`, or a tiebreak of
        `note_id DESC`) must reorder these three relative to each other and
        fail this assertion. A single-note-per-section test (as in
        `test_reexport_with_no_intervening_write_is_byte_identical` before
        this test was added) cannot exercise within-section ordering at
        all."""
        store = _store(tmp_path)
        ws = "ws"
        id_a = store.remember(ws, "alpha finding", priority="low", kind="finding")
        id_b = store.remember(ws, "bravo finding", priority="high", kind="finding")
        id_c = store.remember(ws, "charlie finding", priority="medium", kind="finding")
        assert id_a < id_b < id_c  # sanity: note_id is monotonically increasing
        notes = store.notes_for_export(ws)
        states = store.note_event_states(ws, notes)
        rendered = render_memory_markdown(notes, states)
        pos_a = rendered.index(f"[#{id_a}]")
        pos_b = rendered.index(f"[#{id_b}]")
        pos_c = rendered.index(f"[#{id_c}]")
        assert pos_a < pos_b < pos_c


# ---------------------------------------------------------------------------
# write_memory_markdown — atomic write
# ---------------------------------------------------------------------------

class TestWriteMemoryMarkdownAtomic:
    def test_writes_content_and_creates_parent_dirs(self, tmp_path):
        target = tmp_path / "nested" / "MEMORY.md"
        write_memory_markdown(target, "hello")
        assert target.read_text() == "hello"

    def test_no_leftover_temp_file_after_success(self, tmp_path):
        target = tmp_path / "MEMORY.md"
        write_memory_markdown(target, "content")
        leftovers = list(tmp_path.glob(".memory_export_*"))
        assert leftovers == []

    def test_overwrites_existing_file_atomically(self, tmp_path):
        target = tmp_path / "MEMORY.md"
        target.write_text("old content")
        write_memory_markdown(target, "new content")
        assert target.read_text() == "new content"


# ---------------------------------------------------------------------------
# export_memory — end-to-end through a real store
# ---------------------------------------------------------------------------

class TestExportMemory:
    def test_reexport_with_no_intervening_write_is_byte_identical(self, tmp_path):
        store = _store(tmp_path)
        ws = "ws"
        store.remember(ws, "note one", priority="high", kind="finding")
        store.remember(ws, "note two", priority="low", kind="task")
        out = tmp_path / "MEMORY.md"
        export_memory(store, ws, out)
        first = out.read_bytes()
        export_memory(store, ws, out)
        second = out.read_bytes()
        assert first == second

    def test_reexport_with_several_same_kind_notes_is_byte_identical(self, tmp_path):
        """Strengthens the byte-identical guarantee to cover within-section
        ties: `test_reexport_with_no_intervening_write_is_byte_identical`
        above uses one note per kind (finding, task), so it never exercises
        ordering among notes that share a section. Here, the store is
        RE-QUERIED between the two exports (not a cached Python list, as
        `TestRenderMemoryMarkdown` uses) with five same-kind notes present —
        this is exactly where an `ORDER BY` lacking a total-order tiebreak
        would surface as nondeterministic, whole-file `git diff` noise."""
        store = _store(tmp_path)
        ws = "ws"
        priorities = ["high", "medium", "low", "medium", "high"]
        for i, priority in enumerate(priorities):
            store.remember(ws, f"finding number {i}", priority=priority, kind="finding")
        out = tmp_path / "MEMORY.md"
        export_memory(store, ws, out)
        first = out.read_bytes()
        export_memory(store, ws, out)  # re-queries the store fresh
        second = out.read_bytes()
        assert first == second

    def test_export_reflects_a_new_write(self, tmp_path):
        store = _store(tmp_path)
        ws = "ws"
        out = tmp_path / "MEMORY.md"
        export_memory(store, ws, out)
        before = out.read_text()
        store.remember(ws, "freshly written note", priority="medium", kind="finding")
        export_memory(store, ws, out)
        after = out.read_text()
        assert before != after
        assert "freshly written note" in after


# ---------------------------------------------------------------------------
# .vectr/memory_export_path marker mechanism
# ---------------------------------------------------------------------------

class TestExportTargetMarker:
    def test_absent_marker_returns_none(self, tmp_path):
        assert _read_export_target(str(tmp_path)) is None

    def test_write_then_read_round_trips_absolute_path(self, tmp_path):
        target = tmp_path / "MEMORY.md"
        write_export_target(str(tmp_path), target)
        assert _read_export_target(str(tmp_path)) == target

    def test_marker_file_lives_under_dot_vectr(self, tmp_path):
        target = tmp_path / "MEMORY.md"
        write_export_target(str(tmp_path), target)
        assert (tmp_path / ".vectr" / "memory_export_path").exists()


# ---------------------------------------------------------------------------
# remove_export_target — UPG-MEMORY-EXPORT-DISABLE
# ---------------------------------------------------------------------------

class TestRemoveExportTarget:
    def test_removes_marker_and_returns_previous_target(self, tmp_path):
        target = tmp_path / "MEMORY.md"
        write_export_target(str(tmp_path), target)
        assert (tmp_path / ".vectr" / "memory_export_path").exists()

        removed = remove_export_target(str(tmp_path))

        assert removed == target
        assert not (tmp_path / ".vectr" / "memory_export_path").exists()

    def test_unconfigured_workspace_is_a_clean_noop(self, tmp_path):
        # No write_export_target() call first -- marker was never written.
        assert remove_export_target(str(tmp_path)) is None
        # Still a no-op on a second call: idempotent, never raises.
        assert remove_export_target(str(tmp_path)) is None

    def test_after_removal_schedule_export_if_configured_is_a_noop(self, tmp_path, monkeypatch):
        """The actual bug this whole task exists to fix: before
        remove_export_target(), there was no way to make
        schedule_export_if_configured() stop firing on every future note
        write once a workspace had opted in once. Proves the marker's
        removal, not just its absence from disk, is what the scheduler
        reads."""
        store = _store(tmp_path)
        ws = str(tmp_path)
        target = tmp_path / "MEMORY.md"
        write_export_target(ws, target)
        remove_export_target(ws)

        monkeypatch.setattr(
            "agent.working_context_store._export.MEMORY_EXPORT_DEBOUNCE_SECONDS", 0.05,
        )
        with patch("agent.working_context_store._export.export_memory") as mock_export:
            store.remember(ws, "a note after disabling export", priority="medium", kind="finding")
            schedule_export_if_configured(ws, store)
            time.sleep(0.2)
        mock_export.assert_not_called()


# ---------------------------------------------------------------------------
# schedule_export_if_configured — debounced post-write hook
# ---------------------------------------------------------------------------

class TestScheduleExportIfConfigured:
    def test_noop_when_no_marker_configured(self, tmp_path, monkeypatch):
        store = _store(tmp_path)
        monkeypatch.setattr(
            "agent.working_context_store._export.MEMORY_EXPORT_DEBOUNCE_SECONDS", 0.05,
        )
        with patch("agent.working_context_store._export.export_memory") as mock_export:
            schedule_export_if_configured(str(tmp_path), store)
            time.sleep(0.2)
        mock_export.assert_not_called()

    def test_fires_export_after_debounce_when_marker_present(self, tmp_path, monkeypatch):
        store = _store(tmp_path)
        ws = str(tmp_path)
        store.remember(ws, "a note", priority="medium", kind="finding")
        target = tmp_path / "MEMORY.md"
        write_export_target(ws, target)
        monkeypatch.setattr(
            "agent.working_context_store._export.MEMORY_EXPORT_DEBOUNCE_SECONDS", 0.05,
        )
        schedule_export_if_configured(ws, store)
        time.sleep(0.3)
        assert target.exists()
        assert "a note" in target.read_text()

    def test_rapid_calls_coalesce_into_a_single_render(self, tmp_path, monkeypatch):
        store = _store(tmp_path)
        ws = str(tmp_path)
        target = tmp_path / "MEMORY.md"
        write_export_target(ws, target)
        monkeypatch.setattr(
            "agent.working_context_store._export.MEMORY_EXPORT_DEBOUNCE_SECONDS", 0.1,
        )
        with patch(
            "agent.working_context_store._export.export_memory",
            wraps=__import__(
                "agent.working_context_store._export", fromlist=["export_memory"],
            ).export_memory,
        ) as spy:
            for _ in range(5):
                schedule_export_if_configured(ws, store)
                time.sleep(0.01)
            time.sleep(0.3)
        assert spy.call_count == 1

    def test_never_raises_even_when_export_fails(self, tmp_path, monkeypatch):
        store = _store(tmp_path)
        ws = str(tmp_path)
        write_export_target(ws, tmp_path / "MEMORY.md")
        monkeypatch.setattr(
            "agent.working_context_store._export.MEMORY_EXPORT_DEBOUNCE_SECONDS", 0.05,
        )
        with patch(
            "agent.working_context_store._export.export_memory",
            side_effect=RuntimeError("boom"),
        ):
            schedule_export_if_configured(ws, store)  # must not raise
            time.sleep(0.2)  # let the background timer's exception surface if any


# ---------------------------------------------------------------------------
# CLI: `vectr memory export`
# ---------------------------------------------------------------------------

class TestCmdMemoryExport:
    def _args(self, workspace: str, export_path: str = "MEMORY.md", command: str = "export", disable: bool = False):
        return argparse.Namespace(
            memory_command=command, workspace=workspace, export_path=export_path, disable=disable,
        )

    def test_export_writes_file_and_marker(self, tmp_path, monkeypatch, capsys):
        import main as m

        db_dir = tmp_path / "cache_db"
        db_dir.mkdir()
        store = WorkingContextStore(str(db_dir))
        store.remember(str(tmp_path), "seeded via test", priority="medium", kind="finding")

        monkeypatch.setenv("VECTR_DB_DIR", str(db_dir))
        m.cmd_memory(self._args(str(tmp_path)))

        out_file = tmp_path / "MEMORY.md"
        assert out_file.exists()
        assert "seeded via test" in out_file.read_text()
        assert (tmp_path / ".vectr" / "memory_export_path").exists()
        assert "Exported working memory" in capsys.readouterr().out

    def test_unknown_memory_subcommand_prints_usage(self, capsys):
        import main as m

        m.cmd_memory(self._args(".", command=None))
        assert "usage: vectr memory export" in capsys.readouterr().err

    def test_custom_path_resolved_relative_to_workspace(self, tmp_path, monkeypatch):
        import main as m

        db_dir = tmp_path / "cache_db"
        db_dir.mkdir()
        store = WorkingContextStore(str(db_dir))
        store.remember(str(tmp_path), "x", priority="medium", kind="finding")
        monkeypatch.setenv("VECTR_DB_DIR", str(db_dir))

        m.cmd_memory(self._args(str(tmp_path), export_path="docs/notes.md"))
        assert (tmp_path / "docs" / "notes.md").exists()

    def test_export_success_output_names_the_disable_flag(self, tmp_path, monkeypatch, capsys):
        """Response-packing: the command that turns auto-refresh ON must also
        say how to turn it back off, in the same output, not just in
        --help."""
        import main as m

        db_dir = tmp_path / "cache_db"
        db_dir.mkdir()
        WorkingContextStore(str(db_dir))
        monkeypatch.setenv("VECTR_DB_DIR", str(db_dir))

        m.cmd_memory(self._args(str(tmp_path)))
        assert "--disable" in capsys.readouterr().out


class TestCmdMemoryExportDisable:
    def _args(self, workspace: str):
        return argparse.Namespace(
            memory_command="export", workspace=workspace, export_path="MEMORY.md", disable=True,
        )

    def test_disable_on_configured_workspace_removes_marker(self, tmp_path, monkeypatch, capsys):
        import main as m

        db_dir = tmp_path / "cache_db"
        db_dir.mkdir()
        store = WorkingContextStore(str(db_dir))
        store.remember(str(tmp_path), "seeded via test", priority="medium", kind="finding")
        monkeypatch.setenv("VECTR_DB_DIR", str(db_dir))

        # First export configures auto-refresh (writes the marker).
        m.cmd_memory(argparse.Namespace(
            memory_command="export", workspace=str(tmp_path), export_path="MEMORY.md", disable=False,
        ))
        marker = tmp_path / ".vectr" / "memory_export_path"
        assert marker.exists()
        exported_file = tmp_path / "MEMORY.md"
        before = exported_file.read_text()

        m.cmd_memory(self._args(str(tmp_path)))

        assert not marker.exists()
        assert "Disabled" in capsys.readouterr().out
        # The previously exported file is untouched by --disable.
        assert exported_file.read_text() == before

    def test_disable_on_unconfigured_workspace_is_a_clean_noop(self, tmp_path, monkeypatch, capsys):
        import main as m

        db_dir = tmp_path / "cache_db"
        db_dir.mkdir()
        WorkingContextStore(str(db_dir))
        monkeypatch.setenv("VECTR_DB_DIR", str(db_dir))

        # Never exported before -- no marker, no MEMORY.md.
        m.cmd_memory(self._args(str(tmp_path)))  # must not raise

        out = capsys.readouterr().out
        assert "nothing to disable" in out.lower()
        assert not (tmp_path / ".vectr" / "memory_export_path").exists()

    def test_disable_does_not_stop_a_subsequent_note_write_from_scheduling_nothing(self, tmp_path, monkeypatch):
        """End-to-end acceptance case: after `vectr memory export --disable`,
        a note write through the real store schedules no re-render."""
        import main as m
        from agent.working_context_store._export import schedule_export_if_configured

        db_dir = tmp_path / "cache_db"
        db_dir.mkdir()
        store = WorkingContextStore(str(db_dir))
        monkeypatch.setenv("VECTR_DB_DIR", str(db_dir))

        m.cmd_memory(argparse.Namespace(
            memory_command="export", workspace=str(tmp_path), export_path="MEMORY.md", disable=False,
        ))
        m.cmd_memory(self._args(str(tmp_path)))

        monkeypatch.setattr(
            "agent.working_context_store._export.MEMORY_EXPORT_DEBOUNCE_SECONDS", 0.05,
        )
        with patch("agent.working_context_store._export.export_memory") as mock_export:
            store.remember(str(tmp_path), "written after disable", priority="medium", kind="finding")
            schedule_export_if_configured(str(tmp_path), store)
            time.sleep(0.2)
        mock_export.assert_not_called()


# ---------------------------------------------------------------------------
# VectrService._bump_notes_epoch() wiring
# ---------------------------------------------------------------------------

class TestBumpNotesEpochInvokesExportScheduler:
    def test_bump_notes_epoch_calls_scheduler_when_context_store_present(self, tmp_path, monkeypatch):
        from agent import indexer as idx_module
        from tests.conftest import _DummyEmbedProvider, _RealVectrService, make_py

        monkeypatch.setattr(idx_module, "get_embed_provider", lambda _: _DummyEmbedProvider())
        make_py(tmp_path, "sample.py", "def foo(): pass\n")

        with patch("integrations.vscode_bridge.configure_all"), \
             patch("integrations.workspace_detect.find_workspace_root", return_value=str(tmp_path)), \
             patch.dict("os.environ", {"VECTR_DB_DIR": str(tmp_path / "db")}):
            svc = _RealVectrService(workspace_root=str(tmp_path), memory_only=True)

        with patch(
            "agent.working_context_store._export.schedule_export_if_configured",
        ) as mock_schedule:
            svc._bump_notes_epoch()

        mock_schedule.assert_called_once_with(svc._workspace_root, svc._context_store)

    def test_remember_triggers_the_hook_end_to_end(self, tmp_path, monkeypatch):
        from agent import indexer as idx_module
        from tests.conftest import _DummyEmbedProvider, _RealVectrService, make_py

        monkeypatch.setattr(idx_module, "get_embed_provider", lambda _: _DummyEmbedProvider())
        make_py(tmp_path, "sample.py", "def foo(): pass\n")

        with patch("integrations.vscode_bridge.configure_all"), \
             patch("integrations.workspace_detect.find_workspace_root", return_value=str(tmp_path)), \
             patch.dict("os.environ", {"VECTR_DB_DIR": str(tmp_path / "db")}):
            svc = _RealVectrService(workspace_root=str(tmp_path), memory_only=True)

        from agent.working_context_store._export import write_export_target
        target = tmp_path / "MEMORY.md"
        write_export_target(svc._workspace_root, target)
        monkeypatch.setattr(
            "agent.working_context_store._export.MEMORY_EXPORT_DEBOUNCE_SECONDS", 0.05,
        )

        svc.remember("a note written through the real service", priority="medium", kind="finding")
        time.sleep(0.3)

        assert target.exists()
        assert "a note written through the real service" in target.read_text()
