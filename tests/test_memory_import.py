"""Tests for `vectr memory import` (UPG-MEMORY-IMPORT) — the one-way-in
path that brings existing on-disk memory files into vectr's working-
memory store, idempotently on content, with structural-only kind
classification.

Pins:

* Discovery: default candidate list resolution (closed list, ordered).
* Section split: heading boundaries become section boundaries; a file
  with no headings becomes one section.
* Classification: imperative-keyword heading + short body = directive
  (provenance promoted auto->agent so the auto+directive write-time
  guard passes); otherwise finding/auto. Long body under an imperative
  heading is NOT promoted (a 200-line doc-block is documentation, not
  a rule). The source FILE is also evidence: a short section in
  CLAUDE.md / AGENTS.md / .cursorrules / .github/copilot-instructions.md
  is classified as directive regardless of the heading's first word.
* Idempotency: a re-run with the same body creates no duplicates; the
  same file with edited body creates new notes (the original is
  preserved, not silently overwritten). The same idempotency holds
  for re-importing a `vectr memory export` file (the body, after the
  export's own strip(), is byte-identical between renders).
* Export round-trip: a file written by `vectr memory export` is
  recognized by its header comment and parsed through the same
  `parse_edit_buffer()` grammar `vectr memory edit` uses, recovering
  each block as a note with its kind/priority/provenance/tags carried
  forward.
* Dry-run: a `--dry-run` call does not change the store.
* Provenance: every imported note is distinguishable at read time from
  an observed note — the `imported` tag is the at-read-time signal, and
  the `src:` / `src-sha:` / `src-lines:` tags are the at-trace-time
  signals (a user can see WHERE a note came from without a separate
  audit log lookup).
* CLI: `vectr memory import` dispatch through `cmd_memory` (the
  argparse-driven entry point).
"""
from __future__ import annotations

import argparse

import pytest

from agent.working_context_store import WorkingContextStore
from agent.working_context_store._import import (
    ImportPlan,
    ParsedSection,
    PlannedNote,
    _build_tags_for_planned,
    _content_hash,
    _infer_kind_and_provenance,
    _is_already_imported,
    _is_export_file,
    _make_title,
    _resolve_candidate_files,
    _split_into_sections,
    execute_import,
    plan_import,
)
from agent.working_context_store._export import (
    _HEADER_COMMENT,
    render_memory_markdown,
)


def _store(tmp_path) -> WorkingContextStore:
    db_dir = tmp_path / "db"
    db_dir.mkdir(exist_ok=True)
    return WorkingContextStore(str(db_dir))


def _workspace(tmp_path) -> "Path":
    """A bare workspace root (no MEMORY.md yet)."""
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    return ws


# ---------------------------------------------------------------------------
# Section split: heading boundaries
# ---------------------------------------------------------------------------

class TestSplitIntoSections:
    def test_no_headings_yields_one_section(self):
        body = "just plain text, no headings, single block."
        sections = _split_into_sections(body)
        assert len(sections) == 1
        assert sections[0].heading is None
        assert sections[0].body == body

    def test_single_heading_yields_one_section(self):
        body = "# Title\nbody line one\nbody line two"
        sections = _split_into_sections(body)
        assert len(sections) == 1
        assert sections[0].heading == "Title"
        assert "body line one" in sections[0].body
        assert "body line two" in sections[0].body

    def test_multiple_h2_headings_yield_separate_sections(self):
        body = (
            "## Alpha\nalpha body\n\n"
            "## Beta\nbeta body line one\nbeta body line two\n\n"
            "## Gamma\ngamma body"
        )
        sections = _split_into_sections(body)
        assert len(sections) == 3
        assert [s.heading for s in sections] == ["Alpha", "Beta", "Gamma"]
        assert "alpha body" in sections[0].body
        assert "beta body line one" in sections[1].body
        assert "gamma body" in sections[2].body

    def test_mixed_heading_levels(self):
        body = "# Top\ntop body\n\n## Sub\nsub body"
        sections = _split_into_sections(body)
        assert [s.heading for s in sections] == ["Top", "Sub"]

    def test_line_numbers_are_one_indexed(self):
        body = "# First\nfirst body\n# Second\nsecond body"
        sections = _split_into_sections(body)
        # "# First" is on line 1, "# Second" is on line 3.
        assert sections[0].start_line == 1
        assert sections[1].start_line == 3

    def test_heading_inside_fenced_code_block_does_not_split(self):
        body = (
            "# Real heading\nreal body\n\n"
            "```\n# Not a heading\nstill in the body\n```\n"
            "more body"
        )
        sections = _split_into_sections(body)
        assert len(sections) == 1
        assert sections[0].heading == "Real heading"
        assert "# Not a heading" in sections[0].body
        assert "more body" in sections[0].body

    def test_empty_file_yields_no_sections(self):
        assert _split_into_sections("") == []


# ---------------------------------------------------------------------------
# Classification: structural, no query-side heuristics
# ---------------------------------------------------------------------------

class TestInferKindAndProvenance:
    def test_imperative_heading_with_short_body_is_directive(self):
        section = ParsedSection(
            heading="MUST run tests before commit",
            body="Always run pytest before pushing.",
            start_line=1, end_line=2,
        )
        kind, priority, provenance = _infer_kind_and_provenance(section)
        assert kind == "directive"
        assert provenance == "agent"
        assert priority == "high"

    def test_long_body_under_imperative_heading_is_finding(self):
        section = ParsedSection(
            heading="MUST be careful with paths",
            body="line 1\nline 2\nline 3\nline 4",
            start_line=1, end_line=5,
        )
        kind, priority, provenance = _infer_kind_and_provenance(section)
        assert kind == "finding"
        assert provenance == "auto"
        assert priority == "medium"

    def test_non_imperative_heading_is_finding(self):
        section = ParsedSection(
            heading="Project history",
            body="We tried X, then Y, now Z.",
            start_line=1, end_line=2,
        )
        kind, priority, provenance = _infer_kind_and_provenance(section)
        assert kind == "finding"
        assert provenance == "auto"

    def test_never_keyword_is_imperative(self):
        section = ParsedSection(
            heading="NEVER use eval() on user input",
            body="Always validate.",
            start_line=1, end_line=2,
        )
        kind, _, _ = _infer_kind_and_provenance(section)
        assert kind == "directive"

    def test_always_keyword_is_imperative(self):
        section = ParsedSection(
            heading="ALWAYS escape shell args",
            body="Use shlex.quote.",
            start_line=1, end_line=2,
        )
        kind, _, _ = _infer_kind_and_provenance(section)
        assert kind == "directive"

    def test_required_keyword_is_imperative(self):
        section = ParsedSection(
            heading="REQUIRED prepend workspace_root to all paths",
            body="Do it.",
            start_line=1, end_line=2,
        )
        kind, _, _ = _infer_kind_and_provenance(section)
        assert kind == "directive"

    def test_lowercase_imperative_heading_word_is_still_caught(self):
        """The keyword set is uppercase; the test compares against an
        uppercased copy of the heading's first word, so a hand-written
        'must run tests' still classifies correctly."""
        section = ParsedSection(
            heading="must run tests before commit",
            body="Always run pytest before pushing.",
            start_line=1, end_line=2,
        )
        kind, _, _ = _infer_kind_and_provenance(section)
        assert kind == "directive"

    def test_imperative_keyword_mid_heading_does_not_match(self):
        """'It must be done' is prose, not a rule. The first word is
        'It' — not in the keyword set. This is the closed-set's
        advantage: prose with 'must' embedded does NOT become a
        directive on the strength of the word alone."""
        section = ParsedSection(
            heading="It must be done carefully",
            body="Single line body.",
            start_line=1, end_line=2,
        )
        kind, _, _ = _infer_kind_and_provenance(section)
        assert kind == "finding"

    def test_directive_typical_file_short_section_is_directive(self):
        """The file-as-evidence rule: a short section under a
        non-imperative heading in CLAUDE.md / AGENTS.md / .cursorrules
        / .github/copilot-instructions.md is classified as `directive`
        (provenance=`agent`) because the file itself is a known rule-
        bearing instruction file."""
        from pathlib import Path
        section = ParsedSection(
            heading="Test policy",
            body="Run pytest before pushing.",
            start_line=1, end_line=2,
        )
        for filename in ("CLAUDE.md", "AGENTS.md", ".cursorrules",
                         ".github/copilot-instructions.md"):
            kind, priority, provenance = _infer_kind_and_provenance(
                section, Path(filename),
            )
            assert kind == "directive", f"expected directive for {filename}"
            assert provenance == "agent"
            assert priority == "high"

    def test_directive_typical_file_long_section_is_finding(self):
        """A long body in a directive-typical file is still a finding:
        a 200-line prose block is documentation, not a rule, regardless
        of which file it lives in."""
        from pathlib import Path
        section = ParsedSection(
            heading="Test policy",
            body="line 1\nline 2\nline 3\nline 4\nline 5",
            start_line=1, end_line=6,
        )
        kind, _, provenance = _infer_kind_and_provenance(section, Path("CLAUDE.md"))
        assert kind == "finding"
        assert provenance == "auto"

    def test_non_directive_typical_file_short_section_is_finding(self):
        """A short section in MEMORY.md or any other non-directive
        file is NOT promoted to directive without an imperative
        heading — the file signal only kicks in for the closed
        directive-typical set."""
        from pathlib import Path
        section = ParsedSection(
            heading="Test policy",
            body="Run pytest before pushing.",
            start_line=1, end_line=2,
        )
        kind, _, provenance = _infer_kind_and_provenance(section, Path("MEMORY.md"))
        assert kind == "finding"
        assert provenance == "auto"


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

class TestIsAlreadyImported:
    def test_no_existing_notes_returns_false(self, tmp_path):
        store = _store(tmp_path)
        assert _is_already_imported(store, str(_workspace(tmp_path)),
                                     _workspace(tmp_path) / "MEMORY.md",
                                     "body text") is False

    def test_matching_src_and_sha_returns_true(self, tmp_path):
        ws = _workspace(tmp_path)
        body = "the body to be matched"
        tags = ["imported", "src:MEMORY.md", f"src-sha:{_content_hash(body)}"]
        store = _store(tmp_path)
        store.remember(str(ws), body, tags=tags, priority="medium", kind="finding",
                       provenance="auto", author_id="import")
        assert _is_already_imported(store, str(ws), ws / "MEMORY.md", body) is True

    def test_matching_src_only_returns_false(self, tmp_path):
        ws = _workspace(tmp_path)
        # Same source filename, different body — should NOT be a hit.
        store = _store(tmp_path)
        store.remember(str(ws), "first body", tags=["imported", "src:MEMORY.md"],
                       priority="medium", kind="finding", provenance="auto",
                       author_id="import")
        assert _is_already_imported(store, str(ws), ws / "MEMORY.md", "second body") is False

    def test_different_source_same_body_returns_false(self, tmp_path):
        ws = _workspace(tmp_path)
        body = "shared body"
        store = _store(tmp_path)
        store.remember(str(ws), body, tags=["imported", "src:OTHER.md",
                                            f"src-sha:{_content_hash(body)}"],
                       priority="medium", kind="finding", provenance="auto",
                       author_id="import")
        assert _is_already_imported(store, str(ws), ws / "MEMORY.md", body) is False

    def test_revoked_note_still_counts(self, tmp_path):
        """A re-import must skip even when the previous import's note
        was revoked: silently re-creating a duplicate alongside the
        revoked original is the silent-loss the idempotency check
        defends against."""
        ws = _workspace(tmp_path)
        body = "the body"
        store = _store(tmp_path)
        nid = store.remember(str(ws), body, tags=["imported", "src:MEMORY.md",
                                                   f"src-sha:{_content_hash(body)}"],
                             priority="medium", kind="finding", provenance="auto",
                             author_id="import")
        store.revoke_note(str(ws), nid, reason="removed")
        assert _is_already_imported(store, str(ws), ws / "MEMORY.md", body) is True


# ---------------------------------------------------------------------------
# plan_import: end-to-end through candidate discovery
# ---------------------------------------------------------------------------

class TestPlanImport:
    def test_no_candidates_yields_empty_plan(self, tmp_path):
        ws = _workspace(tmp_path)
        plan = plan_import(ws, [], None)
        assert plan.planned == []
        assert plan.skipped_existing == []
        assert plan.parse_errors == []

    def test_discovers_default_memory_md(self, tmp_path):
        ws = _workspace(tmp_path)
        (ws / "MEMORY.md").write_text(
            "# Project notes\n"
            "First learning.\n\n"
            "## More notes\n"
            "Second learning.\n"
        )
        plan = plan_import(ws, [], None)
        assert len(plan.planned) == 2
        # First section has a heading -> title is the heading.
        assert plan.planned[0].title == "Project notes"
        assert plan.planned[1].title == "More notes"

    def test_each_planned_note_has_imported_tag(self, tmp_path):
        ws = _workspace(tmp_path)
        (ws / "MEMORY.md").write_text("# Alpha\nbody content here")
        plan = plan_import(ws, [], None)
        assert len(plan.planned) == 1
        assert "imported" in plan.planned[0].tags

    def test_each_planned_note_has_src_tag(self, tmp_path):
        ws = _workspace(tmp_path)
        (ws / "MEMORY.md").write_text("# Alpha\nbody content here")
        plan = plan_import(ws, [], None)
        assert any(t.startswith("src:") for t in plan.planned[0].tags)

    def test_each_planned_note_has_src_sha_tag(self, tmp_path):
        ws = _workspace(tmp_path)
        (ws / "MEMORY.md").write_text("# Alpha\nbody content here")
        plan = plan_import(ws, [], None)
        sha_tags = [t for t in plan.planned[0].tags if t.startswith("src-sha:")]
        assert len(sha_tags) == 1
        assert sha_tags[0].endswith(_content_hash(plan.planned[0].content))

    def test_each_planned_note_has_src_lines_tag(self, tmp_path):
        ws = _workspace(tmp_path)
        (ws / "MEMORY.md").write_text("# Alpha\nbody content here")
        plan = plan_import(ws, [], None)
        line_tags = [t for t in plan.planned[0].tags if t.startswith("src-lines:")]
        assert len(line_tags) == 1
        # src-lines:<start>-<end>
        assert "-" in line_tags[0].removeprefix("src-lines:")

    def test_short_body_is_skipped(self, tmp_path):
        ws = _workspace(tmp_path)
        (ws / "MEMORY.md").write_text("# Tiny\nok")  # 2 chars body, below floor
        plan = plan_import(ws, [], None)
        assert plan.planned == []
        assert len(plan.skipped_too_short) == 1

    def test_imperative_directive_classified_with_agent_provenance(self, tmp_path):
        ws = _workspace(tmp_path)
        (ws / "MEMORY.md").write_text("# MUST run tests before commit\nAlways run pytest.")
        plan = plan_import(ws, [], None)
        assert len(plan.planned) == 1
        assert plan.planned[0].kind == "directive"
        assert plan.planned[0].provenance == "agent"

    def test_finding_classified_with_auto_provenance(self, tmp_path):
        ws = _workspace(tmp_path)
        (ws / "MEMORY.md").write_text("# Project history\nWe tried X then Y.")
        plan = plan_import(ws, [], None)
        assert plan.planned[0].kind == "finding"
        assert plan.planned[0].provenance == "auto"

    def test_explicit_path_argument(self, tmp_path):
        ws = _workspace(tmp_path)
        extra = tmp_path / "extra.md"
        extra.write_text("# From extra file\nbody content here\n")
        plan = plan_import(ws, [extra], None)
        # The extra file shows up; default candidates don't (none of them
        # exist on disk for this workspace).
        assert any(p.source_path.resolve() == extra.resolve() for p in plan.planned)

    def test_claude_md_short_section_promoted_to_directive(self, tmp_path):
        """The file-as-evidence rule applied end-to-end: a short
        section in CLAUDE.md under a non-imperative heading ('Test
        policy') is classified as `directive` with provenance `agent`,
        even though the heading itself does not start with a keyword
        in the imperative set."""
        ws = _workspace(tmp_path)
        (ws / "CLAUDE.md").write_text(
            "# Test policy\nRun pytest before pushing.\n"
        )
        plan = plan_import(ws, [], None)
        assert len(plan.planned) == 1
        assert plan.planned[0].kind == "directive"
        assert plan.planned[0].provenance == "agent"

    def test_memory_md_short_section_not_promoted_without_imperative(self, tmp_path):
        """The same short section in MEMORY.md stays a `finding`
        because MEMORY.md is not in the directive-typical set."""
        ws = _workspace(tmp_path)
        (ws / "MEMORY.md").write_text(
            "# Test policy\nRun pytest before pushing.\n"
        )
        plan = plan_import(ws, [], None)
        assert len(plan.planned) == 1
        assert plan.planned[0].kind == "finding"
        assert plan.planned[0].provenance == "auto"

    def test_idempotent_re_run_creates_no_duplicates(self, tmp_path):
        ws = _workspace(tmp_path)
        (ws / "MEMORY.md").write_text("# Title\nbody content here")
        store = _store(tmp_path)
        first = plan_import(ws, [], store)
        for n in first.planned:
            store.remember(str(ws), n.content, tags=n.tags, priority=n.priority,
                           kind=n.kind, title=n.title, provenance=n.provenance,
                           author_id=n.author_id)
        second = plan_import(ws, [], store)
        # No new planned notes (all skipped as already-imported).
        assert second.planned == []
        assert len(second.skipped_existing) == len(first.planned)

    def test_edited_body_creates_new_note_without_deleting_old(self, tmp_path):
        ws = _workspace(tmp_path)
        (ws / "MEMORY.md").write_text("# Title\noriginal body content here")
        store = _store(tmp_path)
        first = plan_import(ws, [], store)
        first_result = execute_import(store, str(ws), first)
        assert len(first_result.written) == 1
        # Edit the source file's body.
        (ws / "MEMORY.md").write_text("# Title\nedited body content here")
        second = plan_import(ws, [], store)
        # New note planned (different body -> different hash), original preserved.
        assert len(second.planned) == 1
        assert second.planned[0].content == "edited body content here"
        second_result = execute_import(store, str(ws), second)
        assert len(second_result.written) == 1
        # The original note is still in the store alongside the new one.
        all_notes = store.notes_for_export(str(ws))
        bodies = [n.content for n in all_notes]
        assert "original body content here" in bodies
        assert "edited body content here" in bodies

    def test_dry_run_with_store_uses_idempotency(self, tmp_path):
        """When a store IS passed, plan_import consults it for the
        idempotency check even if the caller is building a dry-run
        preview (the no-store dry-run case is also supported, tested
        in test_dry_run_with_none_store)."""
        ws = _workspace(tmp_path)
        (ws / "MEMORY.md").write_text("# Title\nbody content here")
        store = _store(tmp_path)
        first = plan_import(ws, [], None)
        for n in first.planned:
            store.remember(str(ws), n.content, tags=n.tags, priority=n.priority,
                           kind=n.kind, title=n.title, provenance=n.provenance,
                           author_id=n.author_id)
        # Plan again WITH a store -> idempotency kicks in.
        second = plan_import(ws, [], store)
        assert second.planned == []
        assert len(second.skipped_existing) == 1

    def test_dry_run_with_none_store_skips_idempotency(self, tmp_path):
        """A dry-run plan with a None store pretends the workspace is
        empty so a first-time user sees what they would get on a clean
        import. The CLI is the only caller that passes None here."""
        ws = _workspace(tmp_path)
        (ws / "MEMORY.md").write_text("# Title\nbody content here")
        plan = plan_import(ws, [], None)
        assert len(plan.planned) == 1


# ---------------------------------------------------------------------------
# Export-file round-trip
# ---------------------------------------------------------------------------

class TestExportFileRoundTrip:
    def test_export_file_detected_by_header(self, tmp_path):
        ws = _workspace(tmp_path)
        # Render a real export through the existing render path, then
        # verify the import-side detector picks it up.
        store = _store(tmp_path)
        store.remember(str(ws), "first body", priority="high", kind="finding",
                       provenance="agent", title="first title")
        store.remember(str(ws), "second body", priority="low", kind="task",
                       provenance="human", title="second title")
        notes = store.notes_for_export(str(ws))
        states = store.note_event_states(str(ws), notes)
        rendered = render_memory_markdown(notes, states)
        assert _is_export_file(rendered[:2048]) is True

    def test_handwritten_memory_md_not_misclassified_as_export(self, tmp_path):
        ws = _workspace(tmp_path)
        text = "# My notes\nThese are my personal notes.\nNothing like an export."
        # The detector only matches when the exact header comment is
        # present in the first 2KB.
        assert _is_export_file(text[:2048]) is False

    def test_export_round_trip_recreates_each_block_as_a_note(self, tmp_path):
        ws = _workspace(tmp_path)
        store = _store(tmp_path)
        store.remember(str(ws), "first body", priority="high", kind="finding",
                       provenance="agent", title="first title")
        store.remember(str(ws), "second body", priority="low", kind="task",
                       provenance="human", title="second title")
        notes = store.notes_for_export(str(ws))
        states = store.note_event_states(str(ws), notes)
        rendered = render_memory_markdown(notes, states)
        # Now import that rendered file. Use a fresh workspace so the
        # re-import doesn't have to fight idempotency with the source
        # notes themselves.
        target_ws = tmp_path / "target"
        target_ws.mkdir()
        (target_ws / "MEMORY.md").write_text(rendered)
        plan = plan_import(target_ws, [], None)
        kinds = [n.kind for n in plan.planned]
        titles = [n.title for n in plan.planned]
        assert "finding" in kinds
        assert "task" in kinds
        assert "first title" in titles
        assert "second title" in titles

    def test_export_round_trip_preserves_provenance(self, tmp_path):
        """A round-tripped human-provenance note must come back as
        human-provenance, NOT as auto. The export block carries the
        `provenance:` metadata bullet and `parse_edit_buffer()` extracts
        it; the import path re-emits it verbatim."""
        ws = _workspace(tmp_path)
        store = _store(tmp_path)
        store.remember(str(ws), "a human note", priority="medium", kind="finding",
                       provenance="human", title="Human Title")
        notes = store.notes_for_export(str(ws))
        states = store.note_event_states(str(ws), notes)
        rendered = render_memory_markdown(notes, states)
        target_ws = tmp_path / "target"
        target_ws.mkdir()
        (target_ws / "MEMORY.md").write_text(rendered)
        plan = plan_import(target_ws, [], None)
        human_notes = [n for n in plan.planned if n.title == "Human Title"]
        assert len(human_notes) == 1
        assert human_notes[0].provenance == "human"

    def test_export_round_trip_idempotent_on_repeat(self, tmp_path):
        """Re-importing the SAME export file is a no-op: the export
        carries the original body's bytes (after the export's own
        strip()), and the import tags every block with `src-sha:H(body)`.
        A second plan with the store passed in must skip every block."""
        ws = _workspace(tmp_path)
        store = _store(tmp_path)
        store.remember(str(ws), "first body", priority="high", kind="finding",
                       provenance="agent", title="first title")
        store.remember(str(ws), "second body", priority="low", kind="task",
                       provenance="human", title="second title")
        notes = store.notes_for_export(str(ws))
        states = store.note_event_states(str(ws), notes)
        rendered = render_memory_markdown(notes, states)
        target_ws = tmp_path / "target"
        target_ws.mkdir()
        (target_ws / "MEMORY.md").write_text(rendered)

        first_plan = plan_import(target_ws, [], None)
        first_result = execute_import(store, str(target_ws), first_plan)
        assert len(first_result.written) == 2

        # Second pass: same export file, same bodies -> idempotency
        # must short-circuit every block.
        second_plan = plan_import(target_ws, [], store)
        second_result = execute_import(store, str(target_ws), second_plan)
        assert second_result.written == []
        assert second_result.skipped_existing == 2
        # And the store has exactly the 2 imported notes, not 4.
        assert len(store.notes_for_export(str(target_ws))) == 2


# ---------------------------------------------------------------------------
# execute_import: writes happen through the real store
# ---------------------------------------------------------------------------

class TestExecuteImport:
    def test_execute_writes_each_planned_note(self, tmp_path):
        ws = _workspace(tmp_path)
        (ws / "MEMORY.md").write_text("# Title\nbody content here")
        store = _store(tmp_path)
        plan = plan_import(ws, [], None)
        result = execute_import(store, str(ws), plan)
        assert len(result.written) == 1
        new_id = result.written[0][1]
        note = store.get_note(str(ws), new_id)
        assert note is not None
        assert note.content == "body content here"
        assert "imported" in note.tags

    def test_execute_does_not_write_repeated_runs(self, tmp_path):
        ws = _workspace(tmp_path)
        (ws / "MEMORY.md").write_text("# Title\nbody content here")
        store = _store(tmp_path)
        first_plan = plan_import(ws, [], None)
        first_result = execute_import(store, str(ws), first_plan)
        assert len(first_result.written) == 1
        # Second plan WITH the store -> idempotency skips everything.
        second_plan = plan_import(ws, [], store)
        second_result = execute_import(store, str(ws), second_plan)
        assert second_result.written == []
        assert second_result.skipped_existing == 1
        # The store has exactly one note (not two).
        assert len(store.notes_for_export(str(ws))) == 1

    def test_dry_run_via_plan_then_no_execute_changes_nothing(self, tmp_path):
        """A dry-run preview is `plan_import(ws, [], None)` followed by
        formatting; execute_import is never called. The store stays
        untouched even when the plan has 5 entries."""
        ws = _workspace(tmp_path)
        (ws / "MEMORY.md").write_text(
            "# A\nbody one\n# B\nbody two\n# C\nbody three\n# D\nbody four\n# E\nbody five"
        )
        store = _store(tmp_path)
        plan = plan_import(ws, [], None)
        # Do NOT call execute_import. The store should still be empty.
        assert store.notes_for_export(str(ws)) == []
        assert len(plan.planned) == 5


# ---------------------------------------------------------------------------
# CLI: cmd_memory dispatches `import` to the right place
# ---------------------------------------------------------------------------

class TestCmdMemoryImport:
    def _args(self, workspace: str, paths: list[str] | None = None,
              dry_run: bool = False):
        return argparse.Namespace(
            memory_command="import", workspace=workspace,
            paths=paths or [], dry_run=dry_run,
        )

    def test_dry_run_prints_plan_and_writes_nothing(self, tmp_path, monkeypatch, capsys):
        import main as m

        db_dir = tmp_path / "cache_db"
        db_dir.mkdir()
        WorkingContextStore(str(db_dir))
        monkeypatch.setenv("VECTR_DB_DIR", str(db_dir))

        ws = _workspace(tmp_path)
        (ws / "MEMORY.md").write_text("# Hello\nbody content here")
        # Tell cmd_memory to use this workspace as --workspace. The
        # candidates look at the workspace root, so we point --workspace
        # at the dir we just populated.
        m.cmd_memory(self._args(str(ws), dry_run=True))

        out = capsys.readouterr().out
        assert "DRY RUN" in out
        assert "Hello" in out
        # Nothing was written to the store.
        store = WorkingContextStore(str(db_dir))
        assert store.notes_for_export(str(ws)) == []

    def test_real_run_writes_notes(self, tmp_path, monkeypatch, capsys):
        import main as m

        db_dir = tmp_path / "cache_db"
        db_dir.mkdir()
        store = WorkingContextStore(str(db_dir))
        monkeypatch.setenv("VECTR_DB_DIR", str(db_dir))

        ws = _workspace(tmp_path)
        (ws / "MEMORY.md").write_text("# Hello\nbody content here")

        m.cmd_memory(self._args(str(ws), dry_run=False))

        out = capsys.readouterr().out
        assert "Imported 1 note" in out
        notes = store.notes_for_export(str(ws))
        assert len(notes) == 1
        assert notes[0].content == "body content here"
        assert "imported" in notes[0].tags

    def test_idempotent_real_run_writes_nothing_on_second_call(self, tmp_path, monkeypatch, capsys):
        import main as m

        db_dir = tmp_path / "cache_db"
        db_dir.mkdir()
        store = WorkingContextStore(str(db_dir))
        monkeypatch.setenv("VECTR_DB_DIR", str(db_dir))

        ws = _workspace(tmp_path)
        (ws / "MEMORY.md").write_text("# Hello\nbody content here")

        m.cmd_memory(self._args(str(ws), dry_run=False))
        first_out = capsys.readouterr().out
        assert "Imported 1 note" in first_out

        m.cmd_memory(self._args(str(ws), dry_run=False))
        second_out = capsys.readouterr().out
        # No new imports.
        assert "Imported 0 note" in second_out
        # Still exactly one note in the store.
        assert len(store.notes_for_export(str(ws))) == 1

    def test_explicit_path_argument(self, tmp_path, monkeypatch, capsys):
        import main as m

        db_dir = tmp_path / "cache_db"
        db_dir.mkdir()
        WorkingContextStore(str(db_dir))
        monkeypatch.setenv("VECTR_DB_DIR", str(db_dir))

        ws = _workspace(tmp_path)
        extra = tmp_path / "extra.md"
        extra.write_text("# From extra\nbody content here")

        m.cmd_memory(self._args(str(ws), paths=[str(extra)], dry_run=True))
        out = capsys.readouterr().out
        assert "DRY RUN" in out
        assert "From extra" in out

    def test_unknown_subcommand_includes_import_in_usage(self, capsys):
        import main as m

        m.cmd_memory(argparse.Namespace(memory_command=None, workspace="."))
        err = capsys.readouterr().err
        assert "vectr memory import" in err

    def test_imported_note_provenance_is_auto(self, tmp_path, monkeypatch, capsys):
        """The provenance call: a non-directive imported note carries
        provenance=auto (the at-read-time 'this was not observed by
        the agent' signal), distinguishable from a vectr_remember'd
        note (default provenance=agent). This is the load-bearing
        correctness property the lane brief calls out."""
        import main as m

        db_dir = tmp_path / "cache_db"
        db_dir.mkdir()
        store = WorkingContextStore(str(db_dir))
        monkeypatch.setenv("VECTR_DB_DIR", str(db_dir))

        ws = _workspace(tmp_path)
        (ws / "MEMORY.md").write_text("# Plain notes\nThis is just prose.")

        m.cmd_memory(self._args(str(ws), dry_run=False))
        notes = store.notes_for_export(str(ws))
        assert len(notes) == 1
        assert notes[0].provenance == "auto"
        assert "imported" in notes[0].tags

    def test_directive_classification_promotes_to_agent_provenance(self, tmp_path, monkeypatch):
        """A short block under an imperative heading becomes a
        directive note with provenance=agent, not auto. This is the
        smallest possible deviation from auto needed to satisfy the
        auto+directive write-time guard."""
        import main as m

        db_dir = tmp_path / "cache_db"
        db_dir.mkdir()
        store = WorkingContextStore(str(db_dir))
        monkeypatch.setenv("VECTR_DB_DIR", str(db_dir))

        ws = _workspace(tmp_path)
        (ws / "MEMORY.md").write_text("# MUST run tests\nAlways run pytest.")

        m.cmd_memory(self._args(str(ws), dry_run=False))
        notes = store.notes_for_export(str(ws))
        assert len(notes) == 1
        assert notes[0].kind == "directive"
        assert notes[0].provenance == "agent"
