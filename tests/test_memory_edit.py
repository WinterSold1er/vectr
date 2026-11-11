"""Tests for `vectr memory edit` (UPG-MEMORY-LEGIBLE-FILE-PROJECTION part
(b)) — transactional check-out/check-in editing of vectr working memory.

Covers:
- notes_for_edit() / update_note_fields() (agent/working_context_store/
  _store.py): the two store-layer primitives this feature adds.
- parse_edit_buffer(): pure parsing of the store's own rendered buffer
  grammar (heading/meta/body extraction, forget-marker detection, new-block
  detection).
- plan_edit_operations(): the pure translation core — every acceptance
  case in the task brief, plus edge cases (duplicate/unknown [#N],
  tags-cleared-vs-unchanged, combined field changes).
- apply_edit_plan(): the one writing function — conflict detection applies
  nothing on drift, forget/revoke/supersede/field_update/remember all reach
  the real store.
- edit_memory_interactive(): the thin wrapper, editor launch injected.
- `vectr memory edit` CLI dispatch (main.cmd_memory).
"""
from __future__ import annotations

import argparse

import pytest

from agent.working_context_store import WorkingContextStore
from agent.working_context_store._memory_edit import (
    DELETE_REVOKE_REASON,
    EDIT_HEADER_COMMENT,
    FORGET_MARKER_LINE,
    ApplyResult,
    EditSnapshot,
    MemoryEditConflict,
    MemoryEditParseError,
    apply_edit_plan,
    capture_edit_snapshot,
    edit_memory_interactive,
    parse_edit_buffer,
    plan_edit_operations,
)


def _store(tmp_path) -> WorkingContextStore:
    db_dir = tmp_path / "db"
    db_dir.mkdir(exist_ok=True)
    return WorkingContextStore(str(db_dir))


def _body_replace(buffer_text: str, old_body: str, new_body: str) -> str:
    """Replace a note's BODY (never its heading — the heading's title text
    can collide with the body when a note's title is content-derived, e.g.
    `### [#1] short body` followed by the body `short body` again) by
    anchoring on the metadata block's terminating blank line, which only
    ever precedes the body."""
    anchor = f"\n\n{old_body}"
    assert anchor in buffer_text, f"body {old_body!r} not found in buffer"
    return buffer_text.replace(anchor, f"\n\n{new_body}", 1)


# ---------------------------------------------------------------------------
# Store layer: notes_for_edit / update_note_fields
# ---------------------------------------------------------------------------

class TestNotesForEdit:
    def test_active_notes_are_included(self, tmp_path):
        store = _store(tmp_path)
        ws = "ws"
        note_id = store.remember(ws, "active note", priority="medium", kind="finding")
        notes = store.notes_for_edit(ws)
        assert [n.note_id for n in notes] == [note_id]

    def test_superseded_notes_are_excluded(self, tmp_path):
        store = _store(tmp_path)
        ws = "ws"
        old_id = store.remember(ws, "old", priority="medium", kind="finding")
        new_id = store.remember(ws, "new", priority="medium", kind="finding", supersedes=old_id)
        notes = store.notes_for_edit(ws)
        assert [n.note_id for n in notes] == [new_id]

    def test_revoked_notes_are_excluded(self, tmp_path):
        """A revoked note keeps valid_until NULL — only the note_events fold
        distinguishes it from active, so a naive valid_until filter would
        wrongly include it (design call 2's core requirement)."""
        store = _store(tmp_path)
        ws = "ws"
        note_id = store.remember(ws, "will be revoked", priority="medium", kind="finding")
        store.revoke_note(ws, note_id, reason="wrong")
        assert store.notes_for_edit(ws) == []

    def test_note_id_ascending_order(self, tmp_path):
        store = _store(tmp_path)
        ws = "ws"
        id_a = store.remember(ws, "a", priority="low", kind="finding")
        id_b = store.remember(ws, "b", priority="high", kind="finding")
        notes = store.notes_for_edit(ws)
        assert [n.note_id for n in notes] == [id_a, id_b]


class TestUpdateNoteFields:
    def test_updates_only_passed_fields(self, tmp_path):
        store = _store(tmp_path)
        ws = "ws"
        note_id = store.remember(ws, "content", priority="medium", kind="finding", tags=["a"])
        assert store.update_note_fields(ws, note_id, priority="high") is True
        note = store.get_note(ws, note_id)
        assert note.priority == "high"
        assert note.kind == "finding"
        assert note.tags == ["a"]

    def test_content_is_never_touched(self, tmp_path):
        store = _store(tmp_path)
        ws = "ws"
        note_id = store.remember(ws, "original content", priority="medium", kind="finding")
        store.update_note_fields(ws, note_id, kind="task", priority="low", tags=["x"])
        note = store.get_note(ws, note_id)
        assert note.content == "original content"

    def test_returns_false_for_missing_note(self, tmp_path):
        store = _store(tmp_path)
        assert store.update_note_fields("ws", 999, priority="high") is False

    def test_raises_without_any_field(self, tmp_path):
        store = _store(tmp_path)
        ws = "ws"
        note_id = store.remember(ws, "x", priority="medium", kind="finding")
        with pytest.raises(ValueError):
            store.update_note_fields(ws, note_id)


# ---------------------------------------------------------------------------
# capture_edit_snapshot — the render entry point for the edit surface
# ---------------------------------------------------------------------------

class TestCaptureEditSnapshot:
    def test_buffer_uses_edit_header_not_export_header(self, tmp_path):
        store = _store(tmp_path)
        ws = str(tmp_path)
        store.remember(ws, "x", priority="medium", kind="finding")
        snap = capture_edit_snapshot(store, ws)
        assert snap.buffer_text.startswith(EDIT_HEADER_COMMENT)

    def test_revoked_and_superseded_notes_are_absent_from_buffer(self, tmp_path):
        store = _store(tmp_path)
        ws = str(tmp_path)
        revoked_id = store.remember(ws, "SECRET_REVOKED_MARKER", priority="medium", kind="finding")
        store.revoke_note(ws, revoked_id, reason="wrong")
        old_id = store.remember(ws, "SECRET_SUPERSEDED_MARKER", priority="medium", kind="finding")
        store.remember(ws, "replacement", priority="medium", kind="finding", supersedes=old_id)
        snap = capture_edit_snapshot(store, ws)
        assert "SECRET_REVOKED_MARKER" not in snap.buffer_text
        assert "SECRET_SUPERSEDED_MARKER" not in snap.buffer_text
        assert f"[#{revoked_id}]" not in snap.buffer_text
        assert f"[#{old_id}]" not in snap.buffer_text

    def test_snapshot_notes_match_buffer_ids(self, tmp_path):
        store = _store(tmp_path)
        ws = str(tmp_path)
        note_id = store.remember(ws, "content here", priority="medium", kind="finding")
        snap = capture_edit_snapshot(store, ws)
        assert snap.by_id().keys() == {note_id}
        assert f"[#{note_id}]" in snap.buffer_text


# ---------------------------------------------------------------------------
# parse_edit_buffer — pure parsing of vectr's own rendered grammar
# ---------------------------------------------------------------------------

class TestParseEditBuffer:
    def test_existing_block_extracts_id_meta_and_body(self, tmp_path):
        store = _store(tmp_path)
        ws = str(tmp_path)
        store.remember(ws, "the body text", priority="high", kind="finding", tags=["a", "b"])
        snap = capture_edit_snapshot(store, ws)
        blocks = parse_edit_buffer(snap.buffer_text)
        assert len(blocks) == 1
        b = blocks[0]
        assert b.note_id == 1
        assert b.meta["kind"] == "finding"
        assert b.meta["priority"] == "high"
        assert b.meta["tags"] == "a, b"
        assert b.body == "the body text"

    def test_new_block_has_no_note_id(self, tmp_path):
        text = EDIT_HEADER_COMMENT + "\n\n# Working Memory\n\n### A new title\n\nsome fresh body\n"
        blocks = parse_edit_buffer(text)
        assert len(blocks) == 1
        assert blocks[0].note_id is None
        assert blocks[0].title == "A new title"
        assert blocks[0].body == "some fresh body"

    def test_forget_marker_is_recognized_as_metadata(self, tmp_path):
        store = _store(tmp_path)
        ws = str(tmp_path)
        store.remember(ws, "body", priority="medium", kind="finding", provenance="human")
        snap = capture_edit_snapshot(store, ws)
        buf = snap.buffer_text.replace("- provenance: human", "- provenance: human\n" + FORGET_MARKER_LINE)
        blocks = parse_edit_buffer(buf)
        assert blocks[0].meta.get("forget") == "PERMANENTLY-DELETE-CONFIRMED"

    def test_multiple_blocks_across_sections_parse_independently(self, tmp_path):
        store = _store(tmp_path)
        ws = str(tmp_path)
        id_a = store.remember(ws, "finding body", priority="medium", kind="finding")
        id_b = store.remember(ws, "task body", priority="medium", kind="task")
        snap = capture_edit_snapshot(store, ws)
        blocks = parse_edit_buffer(snap.buffer_text)
        ids = {b.note_id for b in blocks}
        assert ids == {id_a, id_b}
        bodies = {b.note_id: b.body for b in blocks}
        assert bodies[id_a] == "finding body"
        assert bodies[id_b] == "task body"

    def test_body_with_leading_bullet_line_not_mistaken_for_metadata(self, tmp_path):
        """A body that happens to start with `- something: value` where
        `something` isn't a recognized metadata key must fold into body,
        not silently vanish as an unrecognized bullet."""
        text = (
            EDIT_HEADER_COMMENT + "\n\n# Working Memory\n\n"
            "### A checklist note\n\n- not-a-real-key: value\nmore text\n"
        )
        blocks = parse_edit_buffer(text)
        assert blocks[0].meta == {}
        assert "not-a-real-key: value" in blocks[0].body


# ---------------------------------------------------------------------------
# plan_edit_operations — the 7 required acceptance cases
# ---------------------------------------------------------------------------

class TestPlanEditOperationsAcceptance:
    def test_1_body_change_produces_exactly_one_supersede_and_no_mutation(self, tmp_path):
        """Acceptance 1: exactly one supersedes write, zero row content
        mutations. Asserts on the event log AND that the original row still
        holds its original content."""
        store = _store(tmp_path)
        ws = str(tmp_path)
        note_id = store.remember(ws, "original body", priority="medium", kind="finding", provenance="human")
        snap = capture_edit_snapshot(store, ws)

        buf = _body_replace(snap.buffer_text, "original body", "edited body")
        plan = plan_edit_operations(snap, buf)

        assert len(plan.operations) == 1
        op = plan.operations[0]
        assert op.kind == "supersede"
        assert op.note_id == note_id
        assert op.content == "edited body"

        result = apply_edit_plan(store, ws, snap, plan)
        assert result.supersedes == [(note_id, note_id + 1)]
        assert result.field_updates == []
        assert result.remembers == []
        assert result.revokes == []
        assert result.forgets == []

        # Original row's content column is untouched -- only valid_until/
        # superseded_at/superseded_by_note_id are set, never content.
        original = store.get_note(ws, note_id)
        assert original.content == "original body"
        assert original.valid_until is not None
        assert original.superseded_by_note_id == note_id + 1

        # Event log: exactly one 'superseded' event on the original, one
        # 'created' event on the new note -- no mutation-style event exists
        # in NOTE_EVENT_KINDS at all, so this also proves no such event was
        # (or could have been) appended.
        states = store.note_event_states(ws, [original])
        assert states[note_id]["state"] == "superseded"

    def test_2_unchanged_buffer_produces_zero_writes(self, tmp_path):
        """Acceptance 2: no supersedes, no field updates, no events."""
        store = _store(tmp_path)
        ws = str(tmp_path)
        store.remember(ws, "note one", priority="high", kind="finding")
        store.remember(ws, "note two", priority="low", kind="task")
        snap = capture_edit_snapshot(store, ws)

        plan = plan_edit_operations(snap, snap.buffer_text)
        assert plan.operations == []
        assert plan.touched_note_ids == []

        before_count = store.count_notes(ws)
        result = apply_edit_plan(store, ws, snap, plan)
        assert result.is_empty()
        assert store.count_notes(ws) == before_count

    def test_3_deleted_block_produces_revoke_and_is_reinstatable(self, tmp_path):
        """Acceptance 3: a `revoke` (with reason); note is reinstatable."""
        store = _store(tmp_path)
        ws = str(tmp_path)
        keep_id = store.remember(ws, "keep me", priority="medium", kind="finding")
        gone_id = store.remember(ws, "delete me", priority="medium", kind="finding")
        snap = capture_edit_snapshot(store, ws)

        blocks = parse_edit_buffer(snap.buffer_text)
        # Rebuild the buffer with the gone_id's whole block excised.
        start = snap.buffer_text.index(f"### [#{gone_id}]")
        rest_after = snap.buffer_text[start:]
        next_boundary = rest_after.find("\n### ", 1)
        next_section = rest_after.find("\n## ", 1)
        candidates = [x for x in (next_boundary, next_section) if x != -1]
        end = start + (min(candidates) + 1 if candidates else len(rest_after))
        buf = snap.buffer_text[:start] + snap.buffer_text[end:]
        assert f"[#{gone_id}]" not in buf
        assert f"[#{keep_id}]" in buf

        plan = plan_edit_operations(snap, buf)
        assert len(plan.operations) == 1
        assert plan.operations[0] == plan.operations[0]  # sanity
        op = plan.operations[0]
        assert op.kind == "revoke"
        assert op.note_id == gone_id
        assert op.reason == DELETE_REVOKE_REASON

        result = apply_edit_plan(store, ws, snap, plan)
        assert result.revokes == [gone_id]

        note = store.get_note(ws, gone_id)
        assert note is not None  # row still present -- soft delete
        states = store.note_event_states(ws, [note])
        assert states[gone_id]["state"] == "revoked"
        assert states[gone_id]["reason"] == DELETE_REVOKE_REASON

        assert store.reinstate_note(ws, gone_id, actor="human") is True
        states_after = store.note_event_states(ws, [note])
        assert states_after[gone_id]["state"] == "active"

    def test_4_new_block_produces_exactly_one_fresh_remember(self, tmp_path):
        """Acceptance 4: a new block with no [#N] produces exactly one fresh
        remember."""
        store = _store(tmp_path)
        ws = str(tmp_path)
        store.remember(ws, "existing note", priority="medium", kind="finding")
        snap = capture_edit_snapshot(store, ws)

        buf = snap.buffer_text + "\n\n### Brand new note\n\nfresh content\n"
        plan = plan_edit_operations(snap, buf)

        assert len(plan.operations) == 1
        op = plan.operations[0]
        assert op.kind == "remember"
        assert op.note_id is None
        assert op.content == "fresh content"
        assert op.title == "Brand new note"

        before_count = store.count_notes(ws)
        result = apply_edit_plan(store, ws, snap, plan)
        assert len(result.remembers) == 1
        assert store.count_notes(ws) == before_count + 1
        new_note = store.get_note(ws, result.remembers[0])
        assert new_note.content == "fresh content"

    def test_5_kind_priority_tags_change_is_field_update_not_supersede(self, tmp_path):
        """Acceptance 5: kind/priority/tags change produces a field update,
        NOT a supersedes."""
        store = _store(tmp_path)
        ws = str(tmp_path)
        note_id = store.remember(
            ws, "unchanged body", priority="medium", kind="finding", tags=["old"],
        )
        snap = capture_edit_snapshot(store, ws)

        buf = snap.buffer_text.replace("- kind: finding", "- kind: task")
        buf = buf.replace("- priority: medium", "- priority: high")
        buf = buf.replace("- tags: old", "- tags: old, new")
        plan = plan_edit_operations(snap, buf)

        assert len(plan.operations) == 1
        op = plan.operations[0]
        assert op.kind == "field_update"
        assert op.note_id == note_id
        assert op.new_kind == "task"
        assert op.new_priority == "high"
        assert op.new_tags == ["old", "new"]

        before_count = store.count_notes(ws)
        result = apply_edit_plan(store, ws, snap, plan)
        assert result.field_updates == [note_id]
        assert result.supersedes == []
        assert store.count_notes(ws) == before_count  # no new row

        note = store.get_note(ws, note_id)
        assert note.kind == "task"
        assert note.priority == "high"
        assert note.tags == ["old", "new"]
        assert note.content == "unchanged body"  # untouched
        assert note.valid_until is None  # never tombstoned

    def test_6_concurrent_write_to_touched_note_fails_loudly_applies_nothing(self, tmp_path):
        """Acceptance 6: a session racing a concurrent write to a touched
        note fails loudly with a conflict, applies nothing, names the
        conflicting ids."""
        store = _store(tmp_path)
        ws = str(tmp_path)
        note_id = store.remember(ws, "original body", priority="medium", kind="finding", provenance="human")
        snap = capture_edit_snapshot(store, ws)

        # Concurrent write drifts the SAME note this session is about to touch.
        store.update_note_fields(ws, note_id, priority="low")

        buf = _body_replace(snap.buffer_text, "original body", "edited body")
        plan = plan_edit_operations(snap, buf)
        assert plan.touched_note_ids == [note_id]

        before_count = store.count_notes(ws)
        with pytest.raises(MemoryEditConflict) as exc_info:
            apply_edit_plan(store, ws, snap, plan)
        assert exc_info.value.note_ids == [note_id]
        assert f"#{note_id}" in str(exc_info.value)

        # Zero writes applied: no new row, and the concurrent write's own
        # value (priority=low) is exactly what remains -- nothing rolled
        # back, nothing additionally written.
        assert store.count_notes(ws) == before_count
        note = store.get_note(ws, note_id)
        assert note.priority == "low"
        assert note.content == "original body"

    def test_6b_untouched_note_drifting_concurrently_is_not_a_conflict(self, tmp_path):
        """The other half of acceptance 6's contract: an UNTOUCHED note
        drifting concurrently must NOT block an unrelated edit."""
        store = _store(tmp_path)
        ws = str(tmp_path)
        touched_id = store.remember(ws, "will be edited", priority="medium", kind="finding", provenance="human")
        untouched_id = store.remember(ws, "left alone", priority="medium", kind="finding", provenance="human")
        snap = capture_edit_snapshot(store, ws)

        # Drift the note this session's buffer never touches.
        store.update_note_fields(ws, untouched_id, priority="high")

        buf = _body_replace(snap.buffer_text, "will be edited", "was edited")
        plan = plan_edit_operations(snap, buf)
        assert plan.touched_note_ids == [touched_id]

        result = apply_edit_plan(store, ws, snap, plan)  # must not raise
        assert result.supersedes == [(touched_id, touched_id + 2)]

    def test_7_roundtrip_apply_unmodified_buffer_then_reexport_is_byte_identical(self, tmp_path):
        """Acceptance 7: render -> apply unmodified buffer -> re-render is
        byte-identical."""
        store = _store(tmp_path)
        ws = str(tmp_path)
        store.remember(ws, "note one", priority="high", kind="finding", tags=["a"])
        store.remember(ws, "note two", priority="low", kind="task")
        snap = capture_edit_snapshot(store, ws)

        plan = plan_edit_operations(snap, snap.buffer_text)
        result = apply_edit_plan(store, ws, snap, plan)
        assert result.is_empty()

        snap2 = capture_edit_snapshot(store, ws)
        assert snap2.buffer_text == snap.buffer_text


# ---------------------------------------------------------------------------
# plan_edit_operations — additional edge cases
# ---------------------------------------------------------------------------

class TestPlanEditOperationsEdgeCases:
    def test_duplicate_note_id_raises_parse_error(self, tmp_path):
        store = _store(tmp_path)
        ws = str(tmp_path)
        store.remember(ws, "body", priority="medium", kind="finding")
        snap = capture_edit_snapshot(store, ws)
        buf = snap.buffer_text + "\n\n### [#1] Duplicate\n\nduplicate body\n"
        with pytest.raises(MemoryEditParseError):
            plan_edit_operations(snap, buf)

    def test_unknown_note_id_raises_parse_error(self, tmp_path):
        store = _store(tmp_path)
        ws = str(tmp_path)
        store.remember(ws, "body", priority="medium", kind="finding")
        snap = capture_edit_snapshot(store, ws)
        buf = snap.buffer_text + "\n\n### [#999] Stray id\n\nsome body\n"
        with pytest.raises(MemoryEditParseError):
            plan_edit_operations(snap, buf)

    def test_forget_marker_produces_forget_op_not_revoke(self, tmp_path):
        store = _store(tmp_path)
        ws = str(tmp_path)
        note_id = store.remember(ws, "body", priority="medium", kind="finding", provenance="human")
        snap = capture_edit_snapshot(store, ws)
        buf = snap.buffer_text.replace(
            "- provenance: human", "- provenance: human\n" + FORGET_MARKER_LINE,
        )
        plan = plan_edit_operations(snap, buf)
        assert len(plan.operations) == 1
        assert plan.operations[0].kind == "forget"
        assert plan.operations[0].note_id == note_id

        result = apply_edit_plan(store, ws, snap, plan)
        assert result.forgets == [note_id]
        assert store.get_note(ws, note_id) is None  # hard-deleted, unlike revoke

    def test_tags_bullet_removed_clears_tags(self, tmp_path):
        store = _store(tmp_path)
        ws = str(tmp_path)
        note_id = store.remember(ws, "body", priority="medium", kind="finding", tags=["one"])
        snap = capture_edit_snapshot(store, ws)
        assert "- tags: one" in snap.buffer_text
        buf = snap.buffer_text.replace("- tags: one\n", "")
        plan = plan_edit_operations(snap, buf)
        assert len(plan.operations) == 1
        assert plan.operations[0].kind == "field_update"
        assert plan.operations[0].new_tags == []

    def test_no_tags_bullet_and_originally_empty_tags_is_unchanged(self, tmp_path):
        store = _store(tmp_path)
        ws = str(tmp_path)
        store.remember(ws, "body untouched", priority="medium", kind="finding")
        snap = capture_edit_snapshot(store, ws)
        assert "- tags:" not in snap.buffer_text
        plan = plan_edit_operations(snap, snap.buffer_text)
        assert plan.operations == []

    def test_title_edit_alone_on_existing_block_is_a_noop(self, tmp_path):
        """Title isn't in the editable-field list (only body/kind/priority/
        tags are) -- editing only the heading text of an existing block
        must not itself produce any write."""
        store = _store(tmp_path)
        ws = str(tmp_path)
        note_id = store.remember(
            ws, "body text", priority="medium", kind="finding", title="Custom Title",
        )
        snap = capture_edit_snapshot(store, ws)
        assert f"### [#{note_id}] Custom Title" in snap.buffer_text
        buf = snap.buffer_text.replace(
            f"### [#{note_id}] Custom Title", f"### [#{note_id}] Renamed Title",
        )
        plan = plan_edit_operations(snap, buf)
        assert plan.operations == []

    def test_supersede_preserves_original_title_not_heading_edit(self, tmp_path):
        """When a body change DOES trigger a supersede, the new note keeps
        the original note's stored title verbatim, even if the heading text
        in the buffer also happened to differ (not fed through as a
        separate title edit — see the module's documented design call)."""
        store = _store(tmp_path)
        ws = str(tmp_path)
        note_id = store.remember(
            ws, "body text", priority="medium", kind="finding", title="Custom Title",
        )
        snap = capture_edit_snapshot(store, ws)
        buf = snap.buffer_text.replace(
            f"### [#{note_id}] Custom Title", f"### [#{note_id}] Renamed Title",
        )
        buf = _body_replace(buf, "body text", "edited body text")
        plan = plan_edit_operations(snap, buf)
        assert len(plan.operations) == 1
        assert plan.operations[0].title == "Custom Title"
        result = apply_edit_plan(store, ws, snap, plan)
        new_id = result.supersedes[0][1]
        assert store.get_note(ws, new_id).title == "Custom Title"


# ---------------------------------------------------------------------------
# apply_edit_plan — direct behavior not already covered above
# ---------------------------------------------------------------------------

class TestApplyEditPlan:
    def test_empty_plan_never_touches_the_store(self, tmp_path):
        store = _store(tmp_path)
        ws = str(tmp_path)
        store.remember(ws, "x", priority="medium", kind="finding")
        snap = capture_edit_snapshot(store, ws)
        result = apply_edit_plan(store, ws, snap, plan_edit_operations(snap, snap.buffer_text))
        assert isinstance(result, ApplyResult)
        assert result.is_empty()

    def test_supersede_default_provenance_is_human(self, tmp_path):
        store = _store(tmp_path)
        ws = str(tmp_path)
        note_id = store.remember(ws, "body", priority="medium", kind="finding", provenance="human")
        snap = capture_edit_snapshot(store, ws)
        buf = _body_replace(snap.buffer_text, "body", "edited body")
        plan = plan_edit_operations(snap, buf)
        result = apply_edit_plan(store, ws, snap, plan)
        new_note = store.get_note(ws, result.supersedes[0][1])
        assert new_note.provenance == "human"


# ---------------------------------------------------------------------------
# edit_memory_interactive — the thin wrapper, editor launch injected
# ---------------------------------------------------------------------------

class TestEditMemoryInteractive:
    def test_injected_editor_round_trip_applies_real_writes(self, tmp_path):
        store = _store(tmp_path)
        ws = str(tmp_path)
        note_id = store.remember(ws, "seed body", priority="medium", kind="finding", provenance="human")

        def fake_editor(path):
            text = path.read_text()
            text = text.replace("\n\nseed body\n", "\n\nseed body EDITED\n", 1)
            path.write_text(text, encoding="utf-8")

        result = edit_memory_interactive(store, ws, launch_editor=fake_editor)
        assert len(result.supersedes) == 1
        new_note = store.get_note(ws, result.supersedes[0][1])
        assert new_note.content == "seed body EDITED"

    def test_editor_that_makes_no_change_applies_nothing(self, tmp_path):
        store = _store(tmp_path)
        ws = str(tmp_path)
        store.remember(ws, "seed body", priority="medium", kind="finding")

        def noop_editor(path):
            pass

        result = edit_memory_interactive(store, ws, launch_editor=noop_editor)
        assert result.is_empty()

    def test_temp_file_is_removed_after_apply(self, tmp_path):
        store = _store(tmp_path)
        ws = str(tmp_path)
        store.remember(ws, "seed body", priority="medium", kind="finding")
        captured_path = {}

        def capturing_editor(path):
            captured_path["path"] = path
            assert path.exists()

        edit_memory_interactive(store, ws, launch_editor=capturing_editor)
        assert not captured_path["path"].exists()

    def test_temp_file_is_removed_even_on_conflict(self, tmp_path):
        store = _store(tmp_path)
        ws = str(tmp_path)
        note_id = store.remember(ws, "seed body", priority="medium", kind="finding", provenance="human")
        captured_path = {}

        def racing_editor(path):
            captured_path["path"] = path
            # Simulate a concurrent write landing while $EDITOR is open.
            store.update_note_fields(ws, note_id, priority="high")
            text = path.read_text()
            text = text.replace("\n\nseed body\n", "\n\nseed body EDITED\n", 1)
            path.write_text(text, encoding="utf-8")

        with pytest.raises(MemoryEditConflict):
            edit_memory_interactive(store, ws, launch_editor=racing_editor)
        assert not captured_path["path"].exists()


# ---------------------------------------------------------------------------
# CLI: `vectr memory edit`
# ---------------------------------------------------------------------------

class TestCmdMemoryEdit:
    def test_conflict_exits_nonzero_and_prints_conflicting_ids(self, tmp_path, monkeypatch, capsys):
        import main as m
        from unittest.mock import patch

        db_dir = tmp_path / "cache_db"
        db_dir.mkdir()
        WorkingContextStore(str(db_dir))
        monkeypatch.setenv("VECTR_DB_DIR", str(db_dir))

        with patch(
            "agent.working_context_store._memory_edit.edit_memory_interactive",
            side_effect=MemoryEditConflict([3, 4]),
        ):
            with pytest.raises(SystemExit) as exc_info:
                m.cmd_memory(argparse.Namespace(memory_command="edit", workspace=str(tmp_path)))
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "#3" in err and "#4" in err

    def test_no_changes_prints_no_changes(self, tmp_path, monkeypatch, capsys):
        import main as m
        from unittest.mock import patch

        db_dir = tmp_path / "cache_db"
        db_dir.mkdir()
        WorkingContextStore(str(db_dir))
        monkeypatch.setenv("VECTR_DB_DIR", str(db_dir))

        with patch(
            "agent.working_context_store._memory_edit.edit_memory_interactive",
            return_value=ApplyResult(),
        ):
            m.cmd_memory(argparse.Namespace(memory_command="edit", workspace=str(tmp_path)))
        assert "No changes." in capsys.readouterr().out

    def test_summary_names_each_operation_kind(self, tmp_path, monkeypatch, capsys):
        import main as m
        from unittest.mock import patch

        db_dir = tmp_path / "cache_db"
        db_dir.mkdir()
        WorkingContextStore(str(db_dir))
        monkeypatch.setenv("VECTR_DB_DIR", str(db_dir))

        result = ApplyResult(
            supersedes=[(1, 5)], field_updates=[2], remembers=[6], revokes=[3], forgets=[4],
        )
        with patch(
            "agent.working_context_store._memory_edit.edit_memory_interactive",
            return_value=result,
        ):
            m.cmd_memory(argparse.Namespace(memory_command="edit", workspace=str(tmp_path)))
        out = capsys.readouterr().out
        assert "#1 -> #5" in out
        assert "Updated fields on #2" in out
        assert "Remembered new note #6" in out
        assert "Revoked #3" in out
        assert "Permanently deleted #4" in out

    def test_end_to_end_through_real_store_no_mocks(self, tmp_path, monkeypatch):
        """Exercises the real translation path through the CLI entry point,
        not a mock of plan_edit_operations/apply_edit_plan. `edit_memory_
        interactive`'s `launch_editor` default binds `_default_launch_
        editor` at function-definition time, so patching that name later
        would not be observed -- $EDITOR is a real environment seam instead,
        pointed at a real script, so this drives the actual `subprocess.run`
        call `_default_launch_editor` makes."""
        import main as m
        import sys as _sys

        db_dir = tmp_path / "cache_db"
        db_dir.mkdir()
        store = WorkingContextStore(str(db_dir))
        store.remember(str(tmp_path), "seed body", priority="medium", kind="finding", provenance="human")
        monkeypatch.setenv("VECTR_DB_DIR", str(db_dir))

        editor_script = tmp_path / "fake_editor.py"
        editor_script.write_text(
            "import sys\n"
            "path = sys.argv[1]\n"
            "with open(path) as f:\n"
            "    text = f.read()\n"
            "text = text.replace('\\n\\nseed body\\n', '\\n\\nseed body EDITED\\n', 1)\n"
            "with open(path, 'w') as f:\n"
            "    f.write(text)\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("EDITOR", f"{_sys.executable} {editor_script}")

        m.cmd_memory(argparse.Namespace(memory_command="edit", workspace=str(tmp_path)))

        notes = store.notes_for_edit(str(tmp_path))
        assert len(notes) == 1
        assert notes[0].content == "seed body EDITED"

    def test_usage_mentions_both_subcommands(self, capsys):
        import main as m

        m.cmd_memory(argparse.Namespace(memory_command=None, workspace="."))
        err = capsys.readouterr().err
        assert "vectr memory export" in err
        assert "vectr memory edit" in err
