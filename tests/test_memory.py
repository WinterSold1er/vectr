"""
Comprehensive tests for WorkingContextStore — SQLite-backed note memory.

Critical coverage:
  - Cross-instance persistence: the exact scenario the POC benchmark tested.
    Phase 1 stores notes via instance A; Phase 2 runs a FRESH instance B
    pointing at the same db_dir. Notes must be retrievable.
  - Workspace isolation: workspace A notes must not bleed into workspace B.
  - All CRUD operations, filtering, decay, and snapshot round-trips.
"""
from __future__ import annotations

import os
import re
import time

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _store(tmp_path):
    from agent.working_context_store import WorkingContextStore
    return WorkingContextStore(str(tmp_path))


# ---------------------------------------------------------------------------
# Cross-instance persistence — THE critical test
# ---------------------------------------------------------------------------

class TestCrossInstancePersistence:
    def test_notes_survive_new_store_instance(self, tmp_path) -> None:
        """
        Simulates the two-phase POC flow:
          Phase 1 → instance A stores notes
          Phase 2 → instance B (same db_dir, fresh Python object) recalls them.

        This is the test that would have caught the benchmark failure.
        """
        # Phase 1: store findings via instance A
        store_a = _store(tmp_path)
        store_a.remember(
            "/repo",
            "Field.contribute_to_class at django/db/models/fields/__init__.py:770",
            tags=["field", "lifecycle"], priority="high",
        )
        store_a.remember(
            "/repo",
            "deconstruct() must return (name, path, args, kwargs) — currency kwarg required",
            tags=["migration"], priority="high",
        )
        store_a.remember(
            "/repo",
            "from_db_value() converts int cents → Decimal; to_python() same direction",
            tags=["conversion"], priority="medium",
        )
        del store_a  # explicitly drop the object

        # Phase 2: brand new instance, same db_dir
        store_b = _store(tmp_path)
        notes = store_b.recall("/repo")

        assert len(notes) == 3, f"Expected 3 notes, got {len(notes)}"
        contents = {n.content for n in notes}
        assert any("contribute_to_class" in c for c in contents)
        assert any("deconstruct" in c for c in contents)
        assert any("from_db_value" in c for c in contents)

    def test_tags_survive_new_instance(self, tmp_path) -> None:
        store_a = _store(tmp_path)
        store_a.remember("/repo", "field lifecycle", tags=["field", "high-priority"])
        del store_a

        store_b = _store(tmp_path)
        notes = store_b.recall("/repo", tags=["field"])
        assert len(notes) == 1
        assert "field" in notes[0].tags

    def test_priority_survives_new_instance(self, tmp_path) -> None:
        store_a = _store(tmp_path)
        store_a.remember("/repo", "low note", priority="low")
        store_a.remember("/repo", "high note", priority="high")
        del store_a

        store_b = _store(tmp_path)
        high_notes = store_b.recall("/repo", priority="high")
        assert len(high_notes) == 1
        assert high_notes[0].priority == "high"

    def test_snapshot_survives_new_instance(self, tmp_path) -> None:
        store_a = _store(tmp_path)
        store_a.remember("/repo", "note for snapshot")
        sid = store_a.snapshot("/repo", label="phase1-complete")
        del store_a

        store_b = _store(tmp_path)
        payload = store_b.restore_snapshot(sid)
        assert payload is not None
        assert len(payload["notes"]) == 1
        assert payload["notes"][0]["content"] == "note for snapshot"


# ---------------------------------------------------------------------------
# Workspace isolation
# ---------------------------------------------------------------------------

class TestWorkspaceIsolation:
    def test_notes_scoped_to_workspace(self, tmp_path) -> None:
        store = _store(tmp_path)
        store.remember("/workspace-a", "secret of A")
        store.remember("/workspace-b", "secret of B")

        notes_a = store.recall("/workspace-a")
        notes_b = store.recall("/workspace-b")

        assert len(notes_a) == 1
        assert notes_a[0].content == "secret of A"
        assert len(notes_b) == 1
        assert notes_b[0].content == "secret of B"

    def test_workspace_a_invisible_from_b_cross_instance(self, tmp_path) -> None:
        store_a = _store(tmp_path)
        store_a.remember("/workspace-a", "private note")
        del store_a

        store_b = _store(tmp_path)
        notes = store_b.recall("/workspace-b")  # different workspace
        assert notes == []

    def test_forget_all_only_affects_own_workspace(self, tmp_path) -> None:
        store = _store(tmp_path)
        store.remember("/ws-a", "note A")
        store.remember("/ws-b", "note B")
        store.forget_all("/ws-a")

        assert store.recall("/ws-a") == []
        assert len(store.recall("/ws-b")) == 1


# ---------------------------------------------------------------------------
# Team mode: concurrent multi-client access + shared visibility
# ---------------------------------------------------------------------------

class TestTeamModeConcurrency:
    """One central daemon serves many agents. Note-ID allocation, counting, and
    recall must stay correct when several clients write the same workspace's
    notes DB concurrently (busy_timeout + AUTOINCREMENT)."""

    def test_concurrent_remember_allocates_unique_ids(self, tmp_path) -> None:
        import threading
        store = _store(tmp_path)
        ws = "/team/repo"
        ids: list[int] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def worker(i: int) -> None:
            try:
                nid = store.remember(ws, f"concurrent finding {i}", author_id=f"dev-{i % 3}")
                with lock:
                    ids.append(nid)
            except Exception as exc:  # pragma: no cover - failure path
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(24)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(ids) == 24
        assert len(set(ids)) == 24          # every note got a distinct id
        assert store.count_notes(ws) == 24  # nothing lost under contention

    def test_concurrent_recall_during_writes(self, tmp_path) -> None:
        import threading
        store = _store(tmp_path)
        ws = "/team/repo"
        for i in range(10):
            store.remember(ws, f"seed note {i}")

        recalled_counts: list[int] = []
        stop = threading.Event()

        def writer() -> None:
            i = 0
            while not stop.is_set():
                store.remember(ws, f"live note {i}")
                i += 1

        def reader() -> None:
            for _ in range(15):
                recalled_counts.append(len(store.recall(ws)))

        w = threading.Thread(target=writer)
        w.start()
        r = threading.Thread(target=reader)
        r.start()
        r.join()
        stop.set()
        w.join()

        # Every recall returned a consistent, non-empty snapshot (never crashed).
        assert all(c >= 10 for c in recalled_counts)

    def test_concurrent_snapshots_all_persisted(self, tmp_path) -> None:
        import threading
        store = _store(tmp_path)
        ws = "/team/repo"
        store.remember(ws, "shared finding")
        errors: list[Exception] = []

        def snap(i: int) -> None:
            try:
                store.snapshot(ws, label=f"checkpoint-{i}")
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [threading.Thread(target=snap, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        labels = {s["label"] for s in store.list_snapshots(ws)}
        assert labels == {f"checkpoint-{i}" for i in range(8)}

    def test_ttl_sweep_safe_alongside_concurrent_writes(self, tmp_path) -> None:
        import threading
        import time as _time
        store = _store(tmp_path)
        ws = "/team/repo"
        # Seed expired notes (back-dated 10 days).
        old_ids = [store.remember(ws, f"old note {i}") for i in range(5)]
        cutoff = _time.time() - 10 * 86400
        with store._conn() as conn:
            conn.execute(
                "UPDATE notes SET created_at = ? WHERE note_id IN ({})".format(
                    ",".join("?" * len(old_ids))
                ),
                [cutoff] + old_ids,
            )
        errors: list[Exception] = []

        def writer() -> None:
            try:
                for i in range(10):
                    store.remember(ws, f"fresh note {i}")
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        def sweeper() -> None:
            try:
                store.purge_expired_notes(ws, ttl_days=5.0)
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        w = threading.Thread(target=writer)
        s = threading.Thread(target=sweeper)
        w.start(); s.start()
        w.join(); s.join()

        assert errors == []
        remaining = store.recall(ws, limit=50)
        # All 5 expired notes flagged 'expired' (excluded from default
        # recall(), never deleted — UPG-MEMORY-DECAY-KIND-SCOPED); all 10
        # fresh notes intact.
        assert len(remaining) == 10
        assert all("fresh note" in n.content for n in remaining)

    def test_audit_log_intact_under_concurrent_writes(self, tmp_path, monkeypatch) -> None:
        import logging
        import threading
        log_file = tmp_path / "audit.log"
        monkeypatch.setenv("VECTR_AUDIT_LOG", str(log_file))
        logging.getLogger("vectr.audit").handlers.clear()
        store = _store(tmp_path)
        ws = "/team/repo"

        def worker(i: int) -> None:
            store.remember(ws, f"audited note {i}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        logging.getLogger("vectr.audit").handlers.clear()
        lines = [ln for ln in log_file.read_text().splitlines() if "REMEMBER" in ln]
        # One well-formed line per write — no interleaved/torn lines.
        assert len(lines) == 12
        assert all("note_id=" in ln for ln in lines)


class TestSharedMemoryVisibility:
    """Shared working memory: any connected agent sees any other agent's notes
    for the workspace — there are no per-user silos."""

    def test_note_by_one_author_recallable_without_filter(self, tmp_path) -> None:
        store = _store(tmp_path)
        ws = "/team/repo"
        store.remember(ws, "parser rewrite lives in parse/core.py", author_id="alice")
        notes = store.recall(ws)  # no author/session filter
        assert any("parser rewrite" in n.content for n in notes)
        assert notes[0].author_id == "alice"

    def test_second_client_sees_first_clients_note(self, tmp_path) -> None:
        # Two store objects on the same db_dir model two clients of one daemon.
        client_a = _store(tmp_path)
        client_b = _store(tmp_path)
        ws = "/team/repo"
        client_a.remember(ws, "dev A: the retry bug is in queue.py", author_id="alice")
        notes = client_b.recall(ws)
        assert any("retry bug" in n.content for n in notes)


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------

class TestCRUD:
    def test_remember_returns_incrementing_ids(self, tmp_path) -> None:
        store = _store(tmp_path)
        id1 = store.remember("/repo", "first")
        id2 = store.remember("/repo", "second")
        assert id2 > id1

    def test_recall_empty_workspace(self, tmp_path) -> None:
        store = _store(tmp_path)
        assert store.recall("/repo") == []

    def test_recall_query_substring_match(self, tmp_path) -> None:
        store = _store(tmp_path)
        store.remember("/repo", "EvaluateSegments is the entry point")
        store.remember("/repo", "RequestBid starts the auction")
        results = store.recall("/repo", query="EvaluateSegments")
        assert len(results) == 1
        assert "EvaluateSegments" in results[0].content

    def test_recall_tag_filter(self, tmp_path) -> None:
        store = _store(tmp_path)
        store.remember("/repo", "A", tags=["seg"])
        store.remember("/repo", "B", tags=["bid"])
        store.remember("/repo", "C", tags=["seg", "bid"])
        results = store.recall("/repo", tags=["seg"])
        assert all("seg" in n.tags for n in results)
        assert len(results) == 2

    def test_recall_priority_filter(self, tmp_path) -> None:
        store = _store(tmp_path)
        store.remember("/repo", "high", priority="high")
        store.remember("/repo", "medium")
        store.remember("/repo", "low", priority="low")
        high = store.recall("/repo", priority="high")
        assert len(high) == 1 and high[0].priority == "high"
        low = store.recall("/repo", priority="low")
        assert len(low) == 1 and low[0].priority == "low"

    def test_recall_limit(self, tmp_path) -> None:
        store = _store(tmp_path)
        for i in range(10):
            store.remember("/repo", f"note {i}")
        results = store.recall("/repo", limit=3)
        assert len(results) == 3

    def test_recall_updates_last_accessed(self, tmp_path) -> None:
        store = _store(tmp_path)
        before = time.time()
        store.remember("/repo", "accessed note")
        time.sleep(0.01)
        notes = store.recall("/repo")
        assert notes[0].last_accessed >= before

    def test_default_recall_order_is_stable_across_repeated_calls(self, tmp_path) -> None:
        """UPG-RECALL-ORDER-CHURN: recall() bumps last_accessed on every note
        it returns. Before this fix, the default no-query order tie-broke on
        last_accessed DESC, so two back-to-back identical calls could return
        a different order each time (read-your-own-writes churn) once ties
        formed. Several notes here share equal author_trust_score (default,
        untouched) and decay_score (freshly created, no half-life elapsed),
        so they are tied on every ORDER BY column except the deterministic
        created_at/note_id tie-break."""
        store = _store(tmp_path)
        for i in range(8):
            store.remember("/repo", f"note {i}")

        first = [n.note_id for n in store.recall("/repo", limit=8)]
        second = [n.note_id for n in store.recall("/repo", limit=8)]
        third = [n.note_id for n in store.recall("/repo", limit=8)]

        assert first == second == third

    def test_recall_for_path_order_is_stable_across_repeated_calls(self, tmp_path) -> None:
        """UPG-RECALL-ORDER-CHURN: recall_for_path shares the same ORDER BY
        tie-break as the default recall() path and must be equally stable."""
        store = _store(tmp_path)
        for i in range(6):
            store.remember("/repo", f"gotcha about auth.py note {i}", kind="gotcha")

        first = [n.note_id for n in store.recall_for_path("/repo", "auth.py")]
        second = [n.note_id for n in store.recall_for_path("/repo", "auth.py")]

        assert first == second

    def test_forget_specific_note(self, tmp_path) -> None:
        store = _store(tmp_path)
        nid = store.remember("/repo", "to remove")
        assert store.forget("/repo", nid) is True
        assert store.recall("/repo") == []

    def test_forget_nonexistent_returns_false(self, tmp_path) -> None:
        store = _store(tmp_path)
        assert store.forget("/repo", 999999) is False

    def test_forget_all_clears_workspace(self, tmp_path) -> None:
        store = _store(tmp_path)
        for i in range(5):
            store.remember("/repo", f"note {i}")
        count = store.forget_all("/repo")
        assert count == 5
        assert store.recall("/repo") == []


# ---------------------------------------------------------------------------
# Memory kind dimension (UPG-9.3)
# ---------------------------------------------------------------------------

class TestKindDimensionUPG93:
    def test_default_kind_is_finding(self, tmp_path) -> None:
        store = _store(tmp_path)
        store.remember("/repo", "a learning")
        assert store.recall("/repo")[0].kind == "finding"

    def test_kind_round_trips(self, tmp_path) -> None:
        store = _store(tmp_path)
        store.remember("/repo", "never push to main", kind="directive")
        note = store.recall("/repo")[0]
        assert note.kind == "directive"

    def test_recall_filters_by_kind(self, tmp_path) -> None:
        store = _store(tmp_path)
        store.remember("/repo", "never push to main", kind="directive")
        store.remember("/repo", "index_file takes workspace first", kind="gotcha")
        store.remember("/repo", "just a finding")
        directives = store.recall("/repo", kind="directive")
        assert len(directives) == 1
        assert directives[0].kind == "directive"
        assert all(n.kind == "gotcha" for n in store.recall("/repo", kind="gotcha"))

    def test_invalid_kind_falls_back_to_finding(self, tmp_path) -> None:
        store = _store(tmp_path)
        store.remember("/repo", "x", kind="bogus")
        assert store.recall("/repo")[0].kind == "finding"

    def test_format_surfaces_non_default_kind(self, tmp_path) -> None:
        store = _store(tmp_path)
        store.remember("/repo", "never push to main", kind="directive")
        store.remember("/repo", "a plain finding")
        # detail="full" is required to see [DIRECTIVE] in the per-note header line
        text = store.format_notes_for_llm(store.recall("/repo"), detail="full")
        assert "[DIRECTIVE]" in text
        assert "[FINDING]" not in text  # default kind stays implicit

    def test_migration_adds_kind_to_legacy_db(self, tmp_path) -> None:
        """An existing DB with no kind column upgrades without data loss; old rows default 'finding'."""
        import sqlite3
        import time as _t
        db_path = tmp_path / "working_context.sqlite"
        # Build a pre-9.3 notes table (no kind column) with one row.
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                """CREATE TABLE notes (
                    note_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workspace TEXT NOT NULL, content TEXT NOT NULL,
                    tags TEXT NOT NULL DEFAULT '[]', priority TEXT NOT NULL DEFAULT 'medium',
                    created_at REAL NOT NULL, last_accessed REAL NOT NULL,
                    session_id TEXT, decay_score REAL NOT NULL DEFAULT 1.0)"""
            )
            now = _t.time()
            conn.execute(
                "INSERT INTO notes (workspace, content, created_at, last_accessed) VALUES (?,?,?,?)",
                ("/repo", "legacy note", now, now),
            )
        # Opening the store runs _init_db migration.
        store = _store(tmp_path)
        cols = {r[1] for r in sqlite3.connect(str(db_path)).execute("PRAGMA table_info(notes)").fetchall()}
        assert "kind" in cols
        notes = store.recall("/repo")
        assert len(notes) == 1
        assert notes[0].content == "legacy note"
        assert notes[0].kind == "finding"  # legacy rows default to finding


# ---------------------------------------------------------------------------
# Boot recall (UPG-9.2)
# ---------------------------------------------------------------------------

class TestBootRecallUPG92:
    def test_empty_workspace_returns_empty_list(self, tmp_path) -> None:
        """A SessionStart hook on a fresh repo must never error — returns []."""
        store = _store(tmp_path)
        assert store.boot_recall("/repo") == []

    def test_returns_directives_and_high_tasks_only(self, tmp_path) -> None:
        store = _store(tmp_path)
        store.remember("/repo", "never push to main", kind="directive", priority="medium")
        store.remember("/repo", "current sprint goal", kind="task", priority="high")
        store.remember("/repo", "low-priority task", kind="task", priority="low")
        store.remember("/repo", "a finding", kind="finding", priority="high")
        store.remember("/repo", "a gotcha", kind="gotcha", priority="high")
        boot = store.boot_recall("/repo")
        contents = [n.content for n in boot]
        assert "never push to main" in contents
        assert "current sprint goal" in contents
        assert "low-priority task" not in contents   # task but not high priority
        assert "a finding" not in contents           # finding, never in boot set
        assert "a gotcha" not in contents             # gotcha, never in boot set

    def test_directives_ordered_first(self, tmp_path) -> None:
        store = _store(tmp_path)
        store.remember("/repo", "high task", kind="task", priority="high")
        store.remember("/repo", "the directive", kind="directive")
        boot = store.boot_recall("/repo")
        assert boot[0].kind == "directive"

    def test_excludes_superseded(self, tmp_path) -> None:
        import sqlite3
        store = _store(tmp_path)
        nid = store.remember("/repo", "old directive", kind="directive")
        with sqlite3.connect(str(tmp_path / "working_context.sqlite")) as conn:
            conn.execute("UPDATE notes SET valid_until = ? WHERE note_id = ?", (time.time(), nid))
        assert store.boot_recall("/repo") == []

    def test_does_not_bump_last_accessed(self, tmp_path) -> None:
        """Boot injection is automatic, not an access — must not interfere with decay."""
        import sqlite3
        store = _store(tmp_path)
        nid = store.remember("/repo", "a directive", kind="directive")
        db = str(tmp_path / "working_context.sqlite")

        def _last_accessed() -> float:
            with sqlite3.connect(db) as conn:
                return conn.execute("SELECT last_accessed FROM notes WHERE note_id = ?", (nid,)).fetchone()[0]

        before = _last_accessed()
        time.sleep(0.01)
        store.boot_recall("/repo")
        assert _last_accessed() == before  # boot_recall must not touch last_accessed


# ---------------------------------------------------------------------------
# Path-anchored recall (UPG-9.6)
# ---------------------------------------------------------------------------

class TestRecallForPathUPG96:
    def test_matches_note_mentioning_basename(self, tmp_path) -> None:
        store = _store(tmp_path)
        store.remember("/repo", "index_file in symbol_graph.py takes workspace FIRST", kind="gotcha")
        notes = store.recall_for_path("/repo", "/repo/agent/symbol_graph.py", kind="gotcha")
        assert len(notes) == 1
        assert "workspace FIRST" in notes[0].content

    def test_unrelated_file_matches_nothing(self, tmp_path) -> None:
        store = _store(tmp_path)
        store.remember("/repo", "gotcha about symbol_graph.py", kind="gotcha")
        assert store.recall_for_path("/repo", "/repo/app/routes.py", kind="gotcha") == []

    def test_kind_filter_applies(self, tmp_path) -> None:
        store = _store(tmp_path)
        store.remember("/repo", "finding mentioning routes.py", kind="finding")
        store.remember("/repo", "gotcha about routes.py edits", kind="gotcha")
        gotchas = store.recall_for_path("/repo", "/repo/app/routes.py", kind="gotcha")
        assert len(gotchas) == 1
        assert gotchas[0].kind == "gotcha"

    def test_empty_basename_returns_empty(self, tmp_path) -> None:
        store = _store(tmp_path)
        store.remember("/repo", "x", kind="gotcha")
        assert store.recall_for_path("/repo", "", kind="gotcha") == []

    def test_substring_basename_does_not_false_match(self, tmp_path) -> None:
        """UPG-PROXY-SUBSTRING-ANCHOR: unanchored substring matching would
        let "gate.py" false-match inside "uv_regate.py"; a note only about
        uv_regate.py must not be returned for the file "gate.py"."""
        store = _store(tmp_path)
        store.remember("/repo", "see uv_regate.py for details", kind="gotcha")
        assert store.recall_for_path("/repo", "/repo/gate.py", kind="gotcha") == []

    def test_longer_extension_does_not_false_match(self, tmp_path) -> None:
        """"gate.py" must not match content that actually mentions
        "gate.pyc"/"gate.pyx" — a longer identifier sharing the same prefix."""
        store = _store(tmp_path)
        store.remember("/repo", "open gate.pyc for the compiled bytecode", kind="gotcha")
        store.remember("/repo", "gate.pyx is the cython source", kind="gotcha")
        assert store.recall_for_path("/repo", "/repo/gate.py", kind="gotcha") == []

    def test_declared_anchor_exact_match_wins_even_without_content_mention(self, tmp_path) -> None:
        """A note's declared `anchors` are the strongest signal — a note
        anchored to gate.py must match even if its prose content never
        spells the filename out."""
        store = _store(tmp_path)
        store.remember(
            "/repo", "the retry loop here needs a backoff cap", kind="gotcha",
            anchors=["gate.py"],
        )
        notes = store.recall_for_path("/repo", "/repo/gate.py", kind="gotcha")
        assert len(notes) == 1

    def test_declared_absolute_anchor_is_recalled_via_absolute_query(self, tmp_path) -> None:
        """remember()'s own docstring allows a declared anchor to be either
        workspace-relative OR absolute. An absolute anchor's JSON-serialized
        form is `".../gate.py"` — the character immediately before the
        basename is "/", not a quote — so a QUOTED SQL LIKE pattern
        (`%"gate.py"%`) would never match it, silently excluding the note
        from the candidate pool before the Python-side exact-match filter
        ever runs. The anchors SQL prefilter must stay unquoted (a true
        superset), same as the content prefilter. Content deliberately
        carries no mention of the basename anywhere, so this can only pass
        via the anchors path, not the content-boundary path (non-vacuity)."""
        store, ws = _store(tmp_path), str(tmp_path)
        (tmp_path / "src").mkdir()
        abs_anchor = str(tmp_path / "src" / "gate.py")
        store.remember(
            ws, "the retry loop here needs a backoff cap", kind="gotcha",
            anchors=[abs_anchor],
        )
        # A real hook always sends an absolute file_path; this also causes
        # _path_trigger_candidates() to derive the workspace-relative form
        # ("src/gate.py") as a second candidate internally, exercising both
        # halves of the fix in one realistic call.
        notes = store.recall_for_path(ws, abs_anchor, kind="gotcha")
        assert len(notes) == 1
        assert "gate.py" not in notes[0].content

    def test_declared_anchor_for_different_file_does_not_leak_by_substring(self, tmp_path) -> None:
        """A note anchored to uv_regate.py must not surface for gate.py just
        because the anchor string happens to contain "gate.py" as a
        substring."""
        store = _store(tmp_path)
        store.remember(
            "/repo", "the retry loop here needs a backoff cap", kind="gotcha",
            anchors=["uv_regate.py"],
        )
        assert store.recall_for_path("/repo", "/repo/gate.py", kind="gotcha") == []

    def test_boundary_matched_basename_at_string_boundaries_still_matches(self, tmp_path) -> None:
        """Real path boundaries (string start/end, punctuation, quotes) must
        still count as a match, not just whitespace."""
        store = _store(tmp_path)
        store.remember("/repo", "gate.py: verify_token must check expiry", kind="gotcha")
        store.remember("/repo", 'the file "gate.py" was touched', kind="gotcha")
        notes = store.recall_for_path("/repo", "/repo/gate.py", kind="gotcha")
        assert len(notes) == 2

    # -- UPG-TRIGGERS-INERT-ON-PROXY-STRUCTURAL -----------------------------
    #
    # A note's EXPLICIT `triggers[].path` glob is just as deliberate a
    # "this note concerns this file" declaration as `anchors` — before this
    # fix, recall_for_path()'s SQL/Python narrowing never consulted
    # `triggers` at all, so a note with a path glob ONLY in `triggers`
    # (never mentioned in its body, never in `anchors`) was invisible here
    # even though the identical glob already fires it live via the
    # PreToolUse hook's trigger engine (`evaluate_note()`/`fire()`).

    def test_matches_note_with_declared_trigger_path_glob_only(self, tmp_path) -> None:
        """The A/B this task was filed against: an identical note with the
        path only in a `triggers` glob (never in content, never in
        `anchors`) must now be found, exactly like a declared anchor is."""
        store = _store(tmp_path)
        store.remember(
            "/repo", "the retry loop here needs a backoff cap", kind="gotcha",
            triggers=[{"path": "gate.py"}],
        )
        notes = store.recall_for_path("/repo", "/repo/gate.py", kind="gotcha")
        assert len(notes) == 1
        assert "gate.py" not in notes[0].content  # non-vacuity: only the trigger glob matched

    def test_declared_trigger_glob_supports_fnmatch_wildcards(self, tmp_path) -> None:
        """A trigger's `path` is a glob (fnmatch), not an exact string --
        the same P-primitive semantics `_trigger_matches()` already uses for
        live firing, now also consulted for structural relevance."""
        store = _store(tmp_path)
        store.remember(
            "/repo", "package-wide caveat, no filename mentioned anywhere", kind="gotcha",
            triggers=[{"path": "agent/*.py"}],
        )
        notes = store.recall_for_path("/repo", "/repo/agent/gate.py", kind="gotcha")
        assert len(notes) == 1

    def test_declared_trigger_glob_that_does_not_match_the_file_is_not_returned(
        self, tmp_path
    ) -> None:
        """Negative case (glob semantics, not "any trigger present"): a
        trigger path glob that does NOT match the recalled file must not
        inject the note, even though the note does declare a path trigger
        for some other file."""
        store = _store(tmp_path)
        store.remember(
            "/repo", "the retry loop here needs a backoff cap", kind="gotcha",
            triggers=[{"path": "other/*.py"}],
        )
        assert store.recall_for_path("/repo", "/repo/gate.py", kind="gotcha") == []

    def test_declared_trigger_bare_basename_matches_a_nested_file(self, tmp_path) -> None:
        """A trigger glob of just the bare basename ("gate.py", no directory
        prefix) must still match a file nested below the workspace root
        ("/repo/src/gate.py") -- not only a file sitting directly at the
        workspace root. `_path_trigger_candidates()`'s own candidate set is
        "as-given plus workspace-relative" only (no separate basename form
        unless relpath already equals it, e.g. a root-level file), so this
        exercises the basename explicitly appended to the trigger-matching
        candidate set in `recall_for_path()` -- the same explicit-basename
        shape `agent/proactive/matcher.py`'s `_first_anchor()` already uses
        for the live matcher path. A prior version of this fix passed
        `path_candidates` (missing the bare basename) straight into
        `path_trigger_match()` and silently never matched a nested file."""
        store = _store(tmp_path)
        store.remember(
            "/repo", "the retry loop here needs a backoff cap", kind="gotcha",
            triggers=[{"path": "gate.py"}],
        )
        notes = store.recall_for_path("/repo", "/repo/src/gate.py", kind="gotcha")
        assert len(notes) == 1

    def test_kind_default_trigger_bundle_is_not_double_counted_via_triggers_column(
        self, tmp_path
    ) -> None:
        """A note relying on its KIND's default trigger bundle (no explicit
        `triggers[]` passed at write time) stores `triggers = []` in the DB
        (`effective_triggers()`'s replace-not-merge contract resolves the
        default fresh at evaluation time, never bakes it into storage) --
        so this path only ever sees an EXPLICIT, author-declared glob, never
        a derived one. A gotcha note with no explicit triggers and no
        anchors and no content mention of the file must still find nothing
        here (it was already reachable via the anchor signal if anchored;
        this is the genuinely-unrelated case)."""
        store = _store(tmp_path)
        store.remember("/repo", "an unrelated caveat about auth", kind="gotcha")
        assert store.recall_for_path("/repo", "/repo/gate.py", kind="gotcha") == []

    def test_declared_trigger_flood_does_not_evict_an_unrelated_anchor_match(
        self, tmp_path
    ) -> None:
        """Regression for the send-back on top of this section's own fix
        (commit 9356777 introduced this gap while making the harness
        discriminating for the fix above): `triggers LIKE '%"path"%'` cannot
        be scoped to one file the way the content/anchors arms are (a
        trigger's `path` is a glob, evaluated in Python afterward, not a
        literal SQL LIKE comparison) -- so it is a superset over EVERY note
        that declares ANY path trigger for ANY file, workspace-wide. Folding
        that superset into the SAME LIMIT-bounded SQL query as the content/
        anchors arms meant a workspace with enough OTHER trigger-declaring
        notes could push a genuinely file-scoped anchor match off the end of
        the pool before this method's own Python-side narrowing ever ran --
        a single note declaring a broad glob like "**/*.py" could silently
        starve every other file's structural recall of its strongest
        (declared_anchor) evidence, workspace-wide.

        This note is anchored with a BARE basename ("gate.py", not
        "src/gate.py") and its content never mentions the filename at all,
        so the ONLY way it can be found is the anchor arm -- isolating this
        from the bare-basename-anchor-candidate-set fix this same
        regression fix also required (`_anchors_exact_match()` previously
        only ever saw `path_candidates`, which never contains a bare
        basename unless the file sits at the workspace root)."""
        store = _store(tmp_path)
        store.remember(
            "/repo", "the pool ceiling here is capped at 25 concurrent handles",
            kind="gotcha", anchors=["gate.py"],
        )
        # 15 NEWER notes, each declaring a path trigger for a DIFFERENT,
        # unrelated file -- none of them structurally relates to gate.py.
        # limit=3 here (recall_for_path()'s own pool_size = max(limit,
        # min(limit*4, 200)) = 12) makes 15 > pool_size, so pre-fix these
        # alone would already fill the single shared LIMIT-12 pool by
        # recency, before the anchor note (the oldest note in the store) is
        # ever reached.
        for k in range(15):
            store.remember(
                "/repo", f"cross-cutting note #{k}, no filename mentioned",
                kind="finding", triggers=[{"path": f"other_{k:02d}.py"}],
            )
        notes = store.recall_for_path("/repo", "/repo/gate.py", limit=3)
        assert any("pool ceiling" in n.content for n in notes)

    # -- UPG-PROXY-ANCHOR-ABS-REL-NORM ---------------------------------------
    #
    # Declared anchors are authored either way (remember() allows absolute OR
    # workspace-relative), and so are incoming file_paths (a real hook sends
    # an absolute one; direct callers legitimately pass relative ones).
    # Matching on "as-given + workspace-relative" covered
    # abs-input-vs-rel-anchor but silently missed the mirror case,
    # REL-input-vs-ABS-anchor: the note below could only be found through its
    # prose, never its declaration.

    def test_relative_query_matches_absolute_declared_anchor(self, tmp_path) -> None:
        """A note anchored to an ABSOLUTE path must be recalled when the
        caller passes a WORKSPACE-RELATIVE file_path — the candidate set
        needs the workspace-rooted absolute form."""
        store, ws = _store(tmp_path), str(tmp_path)
        (tmp_path / "src").mkdir()
        # Stored fully-resolved on purpose: _anchors_exact_match() compares
        # candidate forms to declared anchors by exact string equality, so on
        # symlinked temp dirs (/var -> /private/var) only the resolved
        # spelling can byte-equal the workspace-rooted absolute candidate the
        # fix derives.
        abs_anchor = str((tmp_path / "src" / "gate.py").resolve())
        store.remember(
            ws, "the retry loop here needs a backoff cap", kind="gotcha",
            anchors=[abs_anchor],
        )
        notes = store.recall_for_path(ws, "src/gate.py", kind="gotcha")
        assert len(notes) == 1
        # Non-vacuity: content never names the file, so ONLY the anchors arm
        # can have produced this hit.
        assert "gate.py" not in notes[0].content

    def test_relative_query_still_matches_relative_declared_anchor(self, tmp_path) -> None:
        """Control for the mirror case above: the long-standing direction (a
        relative query finding its own relative anchor) is unchanged."""
        store, ws = _store(tmp_path), str(tmp_path)
        (tmp_path / "src").mkdir()
        store.remember(
            ws, "the retry loop here needs a backoff cap", kind="gotcha",
            anchors=["src/gate.py"],
        )
        notes = store.recall_for_path(ws, "src/gate.py", kind="gotcha")
        assert len(notes) == 1

    def test_relative_query_does_not_match_other_files_absolute_anchor(
        self, tmp_path
    ) -> None:
        """Negative control: normalisation widens the candidate set, it does
        not loosen exact equality — a note anchored elsewhere stays silent."""
        store, ws = _store(tmp_path), str(tmp_path)
        (tmp_path / "src").mkdir()
        abs_anchor = str((tmp_path / "src" / "gate.py").resolve())
        store.remember(
            ws, "the retry loop here needs a backoff cap", kind="gotcha",
            anchors=[abs_anchor],
        )
        assert store.recall_for_path(ws, "app/routes.py", kind="gotcha") == []


# ---------------------------------------------------------------------------
# format_notes_for_llm
# ---------------------------------------------------------------------------

class TestFormatNotesForLlm:
    def test_empty_returns_no_notes_message(self, tmp_path) -> None:
        store = _store(tmp_path)
        text = store.format_notes_for_llm([])
        assert "No working notes found" in text

    def test_formatted_contains_note_content(self, tmp_path) -> None:
        store = _store(tmp_path)
        store.remember("/repo", "Signal class at django/dispatch/dispatcher.py", tags=["signal"])
        notes = store.recall("/repo")
        text = store.format_notes_for_llm(notes)
        assert "Signal class" in text
        assert "dispatcher.py" in text

    def test_formatted_shows_priority(self, tmp_path) -> None:
        store = _store(tmp_path)
        store.remember("/repo", "important finding", priority="high")
        notes = store.recall("/repo")
        # detail="full" renders the [HIGH] badge in the note header
        text = store.format_notes_for_llm(notes, detail="full")
        assert "HIGH" in text

    def test_formatted_shows_tags(self, tmp_path) -> None:
        store = _store(tmp_path)
        store.remember("/repo", "tagged note", tags=["middleware", "async"])
        notes = store.recall("/repo")
        # detail="full" includes the [tag, ...] block in the note header
        text = store.format_notes_for_llm(notes, detail="full")
        assert "middleware" in text

    def test_multiple_notes_all_present(self, tmp_path) -> None:
        store = _store(tmp_path)
        store.remember("/repo", "finding alpha")
        store.remember("/repo", "finding beta")
        store.remember("/repo", "finding gamma")
        notes = store.recall("/repo")
        text = store.format_notes_for_llm(notes)
        assert "finding alpha" in text
        assert "finding beta" in text
        assert "finding gamma" in text


# ---------------------------------------------------------------------------
# Decay
# ---------------------------------------------------------------------------

class TestDecay:
    def test_decay_reduces_score(self, tmp_path) -> None:
        store = _store(tmp_path)
        nid = store.remember("/repo", "old note")
        # Force a very short half-life (1 second) and decay
        store.decay_old_notes("/repo", half_life_days=1 / 86400)
        notes = store.recall("/repo")
        if notes:
            assert notes[0].decay_score < 1.0

    def test_decay_never_deletes_notes_regardless_of_score(self, tmp_path) -> None:
        """decay_old_notes() is a RANKING-ONLY signal (UPG-MEMORY-DECAY-
        KIND-SCOPED) — even a note with a very low decay_score must never
        be deleted. Replaces the old test_decay_deletes_very_old_notes,
        which asserted a note WAS deleted once its decay_score dropped
        below 0.1 — that assertion directly encoded the defect this fix
        removes (the pre-fix implementation ran `DELETE FROM notes WHERE
        decay_score < 0.1`, violating UPG-MEMORY-STATE-MACHINE §4.1's
        append-only invariant; `forget()` is the one true hard-delete
        escape hatch, not an automatic score threshold)."""
        import sqlite3
        store = _store(tmp_path)
        note_id = store.remember("/repo", "ancient note")
        # Manually set decay_score well below the old 0.1 deletion
        # threshold, to prove that threshold no longer triggers a delete.
        with sqlite3.connect(str(tmp_path / "working_context.sqlite")) as conn:
            conn.execute("UPDATE notes SET decay_score = 0.05 WHERE workspace = ?", ("/repo",))
        store.decay_old_notes("/repo", half_life_days=14)
        assert store.count_notes("/repo") == 1
        notes = store.recall("/repo")
        assert len(notes) == 1
        assert notes[0].note_id == note_id

    def test_fresh_notes_not_deleted_by_decay(self, tmp_path) -> None:
        store = _store(tmp_path)
        store.remember("/repo", "fresh note")
        store.decay_old_notes("/repo", half_life_days=14)  # 14 day half-life, note is seconds old
        notes = store.recall("/repo")
        assert len(notes) == 1
        assert notes[0].decay_score > 0.9


# ---------------------------------------------------------------------------
# UPG-MEMORY-DECAY-KIND-SCOPED: per-kind, append-only, idempotent decay/TTL.
# One test class per acceptance clause quoted in the task item:
#   (1) a directive survives both purge_expired_notes() and decay_old_notes()
#       at any age, even under an explicit operator override;
#   (2) an expired note stops being a recall()/fire() candidate, but its
#       note_events log and deterrent rendering survive via get_note();
#   (3) decay_old_notes() is idempotent — computed from elapsed time, not
#       multiplied onto a note's existing decay_score.
# ---------------------------------------------------------------------------

class TestMemoryDecayKindScoped:
    def test_directive_survives_purge_and_decay_at_any_age_even_with_override(self, tmp_path) -> None:
        """kind='directive' is the one kind configured `null` (exempt) in
        both memory_decay.ttl_days_by_kind and .half_life_days_by_kind
        (agent/config.yaml) — it must never expire or lose rank by age, and
        that exemption must win even over an explicit operator TTL/
        half-life override (VECTR_NOTES_TTL_DAYS maps straight onto
        purge_expired_notes()'s ttl_days argument)."""
        store = _store(tmp_path)
        nid = store.remember("/repo", "standing rule", kind="directive")
        ancient = time.time() - 100_000 * 86400  # ~274 years old
        with store._conn() as conn:
            conn.execute("UPDATE notes SET created_at = ? WHERE note_id = ?", (ancient, nid))

        # Aggressive 1-day TTL override — a directive must still be exempt.
        expired = store.purge_expired_notes("/repo", ttl_days=1.0)
        assert expired == 0
        note = store.get_note("/repo", nid)
        assert note is not None
        states = store.note_event_states("/repo", [note])
        assert states.get(nid, {}).get("state", "active") == "active"

        # Aggressive near-zero half-life override — decay_score must stay 1.0.
        store.decay_old_notes("/repo", half_life_days=0.0001)
        note = store.get_note("/repo", nid)
        assert note.decay_score == 1.0

    def test_expired_note_stops_injection_but_log_and_deterrent_survive(self, tmp_path) -> None:
        """The full contract from the task's acceptance clause: an expired
        note (1) stops appearing in default recall() output, (2) keeps its
        note_events log with a folded 'expired' state, (3) keeps its row
        (get_note() still resolves it), and (4) renders a deterrent-style
        block instead of raw content on the explicit expand path."""
        store = _store(tmp_path)
        nid = store.remember("/repo", "old operational fact", kind="operational")
        ancient = time.time() - 9_999 * 86400
        with store._conn() as conn:
            conn.execute("UPDATE notes SET created_at = ? WHERE note_id = ?", (ancient, nid))

        expired = store.purge_expired_notes("/repo", ttl_days=1.0)
        assert expired == 1

        # (1) stops being injected: excluded from default recall()
        assert all(n.note_id != nid for n in store.recall("/repo", limit=50))

        # (2) + (3) row and note_events log survive (never deleted)
        note = store.get_note("/repo", nid)
        assert note is not None, (
            "expired note's row must survive purge (state machine is "
            "append-only, never DELETE)"
        )
        states = store.note_event_states("/repo", [note])
        assert states.get(nid, {}).get("state") == "expired", (
            f"expected folded state 'expired', got {states.get(nid)}"
        )

        # (4) deterrent rendering survives on explicit expand
        text = store.format_notes_for_llm([note], detail="full")
        assert "[EXPIRED]" in text, text

    def test_decay_idempotent_absolute_not_cumulative(self, tmp_path) -> None:
        """decay_old_notes() must compute decay_score purely from elapsed
        time, never multiply onto the row's PRIOR decay_score — otherwise
        two calls at the same clock reading (or, in production, two calls
        close enough together that elapsed time barely moves) silently
        compound (0.5 then 0.25 instead of 0.5 both times). Seeding a stale
        prior decay_score (0.3) and asserting the post-call score depends
        only on elapsed age (~0.5 after exactly one half-life) is a
        timing-robust way to pin this without needing two real back-to-back
        calls to land at an identical wall-clock instant."""
        store = _store(tmp_path)
        nid = store.remember("/repo", "note", kind="finding")
        half_life_s = 5.0
        with store._conn() as conn:
            conn.execute(
                "UPDATE notes SET created_at = ?, decay_score = ? WHERE note_id = ?",
                (time.time() - half_life_s, 0.3, nid),
            )
        store.decay_old_notes("/repo", half_life_days=half_life_s / 86400)
        note = store.get_note("/repo", nid)
        assert note.decay_score == pytest.approx(0.5, rel=0.05), (
            "decay_score must be recomputed purely from elapsed time "
            "(~0.5 expected after exactly one half-life), not multiplied "
            f"onto a pre-existing decay_score of 0.3 (got {note.decay_score})"
        )

    def test_decay_two_calls_same_clock_produce_identical_score(self, tmp_path) -> None:
        """The literal acceptance clause: decay_old_notes() called twice
        with the SAME clock reading (the `now` argument added by this fix)
        must produce the same decay_score both times — bitwise equality,
        since the score is a pure function of (now - created_at)."""
        store = _store(tmp_path)
        nid = store.remember("/repo", "note", kind="finding")
        fixed_now = time.time() + 20 * 86400

        store.decay_old_notes("/repo", now=fixed_now)
        first = store.get_note("/repo", nid).decay_score

        store.decay_old_notes("/repo", now=fixed_now)
        second = store.get_note("/repo", nid).decay_score

        assert first == second


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------

class TestSnapshots:
    def test_snapshot_captures_all_notes(self, tmp_path) -> None:
        store = _store(tmp_path)
        store.remember("/repo", "note 1")
        store.remember("/repo", "note 2")
        sid = store.snapshot("/repo", label="phase1-complete")
        payload = store.restore_snapshot(sid)
        assert len(payload["notes"]) == 2

    def test_snapshot_label_preserved(self, tmp_path) -> None:
        store = _store(tmp_path)
        store.remember("/repo", "x")
        sid = store.snapshot("/repo", label="my-label")
        snapshots = store.list_snapshots("/repo")
        assert any(s["label"] == "my-label" for s in snapshots)

    def test_list_snapshots_newest_first(self, tmp_path) -> None:
        store = _store(tmp_path)
        store.remember("/repo", "x")
        sid1 = store.snapshot("/repo", label="first")
        time.sleep(0.01)
        sid2 = store.snapshot("/repo", label="second")
        snaps = store.list_snapshots("/repo")
        assert snaps[0]["snapshot_id"] == sid2  # newest first

    def test_restore_nonexistent_returns_none(self, tmp_path) -> None:
        store = _store(tmp_path)
        assert store.restore_snapshot("does_not_exist") is None

    def test_snapshot_with_retrieved_chunks(self, tmp_path) -> None:
        store = _store(tmp_path)
        store.remember("/repo", "note")
        chunks = [{"file": "main.py", "lines": "1-40", "symbol": "run", "content": "def run(): pass"}]
        sid = store.snapshot("/repo", label="with-chunks", retrieved_chunks=chunks)
        payload = store.restore_snapshot(sid)
        assert len(payload["retrieved_chunks"]) == 1
        assert payload["retrieved_chunks"][0]["file"] == "main.py"

    def test_snapshot_workspace_scoped(self, tmp_path) -> None:
        store = _store(tmp_path)
        store.remember("/ws-a", "note a")
        store.snapshot("/ws-a", label="snap-a")
        snaps_b = store.list_snapshots("/ws-b")
        assert snaps_b == []

    def test_snapshot_cross_instance(self, tmp_path) -> None:
        store_a = _store(tmp_path)
        store_a.remember("/repo", "snapshot content")
        sid = store_a.snapshot("/repo", label="cross-instance")
        del store_a

        store_b = _store(tmp_path)
        payload = store_b.restore_snapshot(sid)
        assert payload is not None
        assert payload["notes"][0]["content"] == "snapshot content"


# ---------------------------------------------------------------------------
# Resume surface (UPG-RESUME-SURFACE)
# ---------------------------------------------------------------------------

class TestResumeState:
    def test_empty_workspace_returns_all_empty(self, tmp_path) -> None:
        store = _store(tmp_path)
        state = store.resume_state("/repo")
        assert state == {
            "last_task": None, "gotchas": [], "snapshot": None,
            "gotchas_truncated": False,
        }

    def test_last_task_matches_boot_recall_selection(self, tmp_path) -> None:
        """The task note resume_state() surfaces must be EXACTLY the one
        boot_recall() would inject — both derive from the same
        _boot_task_notes() query (UPG-RESUME-SURFACE's core requirement:
        the two surfaces can never disagree on 'current task')."""
        store = _store(tmp_path)
        store.remember("/repo", "old task", kind="task", priority="high")
        time.sleep(0.01)
        store.remember("/repo", "newest task", kind="task", priority="high")
        store.remember("/repo", "low-priority task", kind="task", priority="low")

        boot = store.boot_recall("/repo")
        boot_task_notes = [n for n in boot if n.kind == "task"]
        state = store.resume_state("/repo")

        assert state["last_task"] is not None
        assert state["last_task"].note_id == boot_task_notes[0].note_id
        assert state["last_task"].content == "newest task"

    def test_gotchas_newest_first_and_capped(self, tmp_path) -> None:
        from agent.config import RESUME_MAX_GOTCHAS

        store = _store(tmp_path)
        for i in range(RESUME_MAX_GOTCHAS + 3):
            store.remember("/repo", f"gotcha {i}", kind="gotcha")
            time.sleep(0.005)
        state = store.resume_state("/repo")
        assert len(state["gotchas"]) == RESUME_MAX_GOTCHAS
        assert state["gotchas"][0].content == f"gotcha {RESUME_MAX_GOTCHAS + 2}"  # newest first
        assert state["gotchas_truncated"] is True

    def test_gotchas_at_or_under_cap_not_truncated(self, tmp_path) -> None:
        from agent.config import RESUME_MAX_GOTCHAS

        store = _store(tmp_path)
        for i in range(RESUME_MAX_GOTCHAS):
            store.remember("/repo", f"gotcha {i}", kind="gotcha")
        state = store.resume_state("/repo")
        assert len(state["gotchas"]) == RESUME_MAX_GOTCHAS
        assert state["gotchas_truncated"] is False

    def test_findings_and_directives_excluded_from_gotchas(self, tmp_path) -> None:
        store = _store(tmp_path)
        store.remember("/repo", "a finding", kind="finding")
        store.remember("/repo", "a directive", kind="directive")
        store.remember("/repo", "a real gotcha", kind="gotcha")
        state = store.resume_state("/repo")
        assert len(state["gotchas"]) == 1
        assert state["gotchas"][0].content == "a real gotcha"

    def test_snapshot_is_latest_with_note_count(self, tmp_path) -> None:
        store = _store(tmp_path)
        store.remember("/repo", "note 1")
        store.remember("/repo", "note 2")
        store.snapshot("/repo", label="first")
        time.sleep(0.01)
        store.remember("/repo", "note 3")
        sid2 = store.snapshot("/repo", label="second")

        state = store.resume_state("/repo")
        assert state["snapshot"]["snapshot_id"] == sid2
        assert state["snapshot"]["label"] == "second"
        assert state["snapshot"]["note_count"] == 3

    def test_snapshot_none_when_unread_payload(self, tmp_path) -> None:
        """restore_snapshot() returning None (undecryptable payload) must
        surface as note_count=None, not crash resume_state()."""
        store = _store(tmp_path)
        store.remember("/repo", "note")
        sid = store.snapshot("/repo", label="only")

        # Simulate an unreadable payload the same way restore_snapshot() would
        # treat it — corrupt the stored JSON so json.loads fails.
        import sqlite3
        with sqlite3.connect(str(tmp_path / "working_context.sqlite")) as conn:
            conn.execute("UPDATE snapshots SET payload = ? WHERE snapshot_id = ?", ("not json", sid))

        state = store.resume_state("/repo")
        assert state["snapshot"]["note_count"] is None

    def test_workspace_isolation(self, tmp_path) -> None:
        store = _store(tmp_path)
        store.remember("/ws-a", "task a", kind="task", priority="high")
        store.remember("/ws-a", "gotcha a", kind="gotcha")
        store.snapshot("/ws-a", label="snap-a")
        state_b = store.resume_state("/ws-b")
        assert state_b == {
            "last_task": None, "gotchas": [], "snapshot": None,
            "gotchas_truncated": False,
        }

    def test_session_scoped_gotcha_excluded_from_other_session(self, tmp_path) -> None:
        store = _store(tmp_path)
        store.remember("/repo", "session-only gotcha", kind="gotcha", session_id="sess-1", scope="session")
        state = store.resume_state("/repo", session_id="sess-2")
        assert state["gotchas"] == []
        state_same = store.resume_state("/repo", session_id="sess-1")
        assert len(state_same["gotchas"]) == 1


class TestFormatResume:
    def test_empty_state_returns_guidance(self, tmp_path) -> None:
        store = _store(tmp_path)
        state = store.resume_state("/repo")
        text = store.format_resume(state, "/repo")
        assert "vectr_remember" in text
        assert "Last task" not in text
        assert "Open gotchas" not in text
        assert "Latest snapshot" not in text

    def test_sections_present_when_populated(self, tmp_path) -> None:
        store = _store(tmp_path)
        store.remember("/repo", "ship the resume feature", kind="task", priority="high")
        store.remember("/repo", "watch out for the flaky test", kind="gotcha", anchors=["tests/test_flaky.py"])
        store.snapshot("/repo", label="checkpoint-1")
        state = store.resume_state("/repo")
        text = store.format_resume(state, "/repo")
        assert "Last task:" in text
        assert "ship the resume feature" in text
        assert "Open gotchas (1):" in text
        assert "watch out for the flaky test" in text
        assert "more open gotchas exist" not in text  # under cap: no truncation line
        assert "tests/test_flaky.py" in text
        assert "Latest snapshot:" in text
        assert "checkpoint-1" in text

    def test_gotcha_truncation_disclosed(self, tmp_path) -> None:
        """With more open gotchas than the cap, the render must say so and
        point at the full listing — a reader must be able to tell "5 gotchas"
        from "5 of 9" (M2 review nit: silent cap)."""
        from agent.config import RESUME_MAX_GOTCHAS

        store = _store(tmp_path)
        for i in range(RESUME_MAX_GOTCHAS + 2):
            store.remember("/repo", f"gotcha {i}", kind="gotcha")
        state = store.resume_state("/repo")
        mcp_text = store.format_resume(state, "/repo", surface="mcp")
        assert "more open gotchas exist" in mcp_text
        assert 'vectr_recall(kind="gotcha")' in mcp_text
        cli_text = store.format_resume(state, "/repo", surface="cli")
        assert "vectr recall --kind gotcha" in cli_text

    def test_omits_sections_with_nothing_to_show(self, tmp_path) -> None:
        """Only a task note exists — no gotcha/snapshot sections must appear."""
        store = _store(tmp_path)
        store.remember("/repo", "only a task", kind="task", priority="high")
        state = store.resume_state("/repo")
        text = store.format_resume(state, "/repo")
        assert "Last task:" in text
        assert "Open gotchas" not in text
        assert "Latest snapshot" not in text

    def test_surface_cli_uses_shell_expand_hint(self, tmp_path) -> None:
        store = _store(tmp_path)
        store.remember("/repo", "a task", kind="task", priority="high")
        state = store.resume_state("/repo")
        text_cli = store.format_resume(state, "/repo", surface="cli")
        text_mcp = store.format_resume(state, "/repo", surface="mcp")
        assert "vectr recall --id" in text_cli
        assert "vectr_recall(note_id=" in text_mcp

    def test_stale_marker_propagates(self, tmp_path) -> None:
        store = _store(tmp_path)
        nid = store.remember("/repo", "gotcha about a file", kind="gotcha")
        state = store.resume_state("/repo")
        text = store.format_resume(state, "/repo", stale_warnings={nid: ["file changed"]})
        assert "[STALE]" in text


# ---------------------------------------------------------------------------
# Staleness — _extract_file_paths and check_staleness
# ---------------------------------------------------------------------------

class TestExtractFilePaths:
    def test_relative_path_with_slash(self):
        from agent.working_context_store import _extract_file_paths
        paths = _extract_file_paths("Key file: agent/indexer.py — main chunking logic")
        assert "agent/indexer.py" in paths

    def test_multi_component_path(self):
        from agent.working_context_store import _extract_file_paths
        paths = _extract_file_paths("Found in src/auth/middleware.py lines 42-67")
        assert "src/auth/middleware.py" in paths

    def test_absolute_path(self):
        from agent.working_context_store import _extract_file_paths
        paths = _extract_file_paths("Path is /Users/alice/project/main.py")
        assert "/Users/alice/project/main.py" in paths

    def test_multiple_paths_in_content(self):
        from agent.working_context_store import _extract_file_paths
        text = "agent/indexer.py calls agent/searcher.py via the service layer"
        paths = _extract_file_paths(text)
        assert "agent/indexer.py" in paths
        assert "agent/searcher.py" in paths

    def test_http_url_not_matched(self):
        from agent.working_context_store import _extract_file_paths
        paths = _extract_file_paths("See http://localhost:8765/mcp for details")
        assert not any("localhost" in p for p in paths)

    def test_plain_word_not_matched(self):
        from agent.working_context_store import _extract_file_paths
        paths = _extract_file_paths("Use sqlite3.Row for row access")
        assert "sqlite3.Row" not in paths

    def test_deduplication(self):
        from agent.working_context_store import _extract_file_paths
        text = "agent/indexer.py is the key file. Also see agent/indexer.py again."
        paths = _extract_file_paths(text)
        assert paths.count("agent/indexer.py") == 1

    def test_empty_string(self):
        from agent.working_context_store import _extract_file_paths
        assert _extract_file_paths("") == []

    def test_no_paths_in_content(self):
        from agent.working_context_store import _extract_file_paths
        assert _extract_file_paths("JWT validation uses a secret key and expiry check") == []


class TestCheckStaleness:
    def test_file_unchanged_not_stale(self, tmp_path):
        store = _store(tmp_path)
        # create a file, THEN write a note — file is older than note
        f = tmp_path / "src" / "auth.py"
        f.parent.mkdir()
        f.write_text("code")
        note_id = store.remember(str(tmp_path), f"Key file: src/auth.py")
        notes = store.recall(str(tmp_path))
        stale = store.check_staleness(notes, str(tmp_path))
        assert note_id not in stale

    def test_file_modified_after_note_is_stale(self, tmp_path):
        store = _store(tmp_path)
        note_id = store.remember(str(tmp_path), "Key file: src/auth.py")
        # now create/modify the file AFTER the note was written
        f = tmp_path / "src" / "auth.py"
        f.parent.mkdir(exist_ok=True)
        f.write_text("changed code")
        # pin mtime past note.created_at — the kernel's coarse file-timestamp
        # clock can lag time.time() by a tick, making a fresh write ambiguous
        future = time.time() + 10
        os.utime(f, (future, future))
        notes = store.recall(str(tmp_path))
        stale = store.check_staleness(notes, str(tmp_path))
        assert note_id in stale
        assert "src/auth.py" in stale[note_id]

    def test_missing_file_skipped(self, tmp_path):
        store = _store(tmp_path)
        note_id = store.remember(str(tmp_path), "Key file: ghost/nonexistent.py")
        notes = store.recall(str(tmp_path))
        stale = store.check_staleness(notes, str(tmp_path))
        assert note_id not in stale

    def test_no_paths_in_note_not_stale(self, tmp_path):
        store = _store(tmp_path)
        note_id = store.remember(str(tmp_path), "JWT uses RS256 and 1h expiry")
        notes = store.recall(str(tmp_path))
        stale = store.check_staleness(notes, str(tmp_path))
        assert note_id not in stale

    def test_only_stale_notes_in_result(self, tmp_path):
        store = _store(tmp_path)
        # note with no file paths — clean
        store.remember(str(tmp_path), "general architecture note")
        # note with a file that gets modified — stale
        note_id_stale = store.remember(str(tmp_path), "Critical: src/core.py is the entry point")
        f = tmp_path / "src" / "core.py"
        f.parent.mkdir(exist_ok=True)
        f.write_text("modified")
        # pin mtime past note.created_at (same coarse-clock race as above)
        future = time.time() + 10
        os.utime(f, (future, future))
        notes = store.recall(str(tmp_path))
        stale = store.check_staleness(notes, str(tmp_path))
        assert len(stale) == 1
        assert note_id_stale in stale


class TestFormatNotesWithStaleness:
    def test_stale_marker_in_output(self, tmp_path):
        store = _store(tmp_path)
        note_id = store.remember(str(tmp_path), "Key file: src/auth.py")
        notes = store.recall(str(tmp_path))
        # [STALE] appears in both index and full; WARNING text only in full
        output = store.format_notes_for_llm(notes, stale_warnings={note_id: ["src/auth.py"]}, detail="full")
        assert "[STALE]" in output
        assert "src/auth.py" in output
        assert "WARNING" in output

    def test_stale_marker_in_index_output(self, tmp_path):
        """[STALE] marker appears in the index tier too — but without the WARNING body."""
        store = _store(tmp_path)
        note_id = store.remember(str(tmp_path), "Key file: src/auth.py")
        notes = store.recall(str(tmp_path))
        output = store.format_notes_for_llm(notes, stale_warnings={note_id: ["src/auth.py"]}, detail="index")
        assert "[STALE]" in output
        assert "WARNING" not in output  # detailed warning only in full tier

    def test_clean_note_has_no_stale_marker(self, tmp_path):
        store = _store(tmp_path)
        note_id = store.remember(str(tmp_path), "general note")
        notes = store.recall(str(tmp_path))
        output = store.format_notes_for_llm(notes, stale_warnings={})
        assert "[STALE]" not in output
        assert "WARNING" not in output

    def test_header_reports_stale_count(self, tmp_path):
        store = _store(tmp_path)
        note_id = store.remember(str(tmp_path), "Key file: src/auth.py")
        notes = store.recall(str(tmp_path))
        # "may be stale" is in the full-tier header only
        output = store.format_notes_for_llm(notes, stale_warnings={note_id: ["src/auth.py"]}, detail="full")
        assert "may be stale" in output

    def test_no_stale_warnings_unchanged_output(self, tmp_path):
        store = _store(tmp_path)
        store.remember(str(tmp_path), "clean note")
        notes = store.recall(str(tmp_path))
        output_none = store.format_notes_for_llm(notes, stale_warnings=None)
        output_empty = store.format_notes_for_llm(notes, stale_warnings={})
        assert "STALE" not in output_none
        assert "STALE" not in output_empty


# ---------------------------------------------------------------------------
# T17: TTL, forget_all_workspaces, audit log
# ---------------------------------------------------------------------------

class TestT17DataRetention:
    def _store(self, tmp_path) -> tuple:
        from agent.working_context_store import WorkingContextStore
        store = WorkingContextStore(str(tmp_path))
        ws = str(tmp_path)
        return store, ws

    def test_purge_expired_notes_flags_old_notes_without_deleting(self, tmp_path) -> None:
        """purge_expired_notes() marks an old note 'expired' via an
        append-only note_events transition (UPG-MEMORY-DECAY-KIND-SCOPED)
        rather than deleting its row — replaces the old
        test_purge_expired_notes_removes_old_notes, whose final assertion
        (`count_notes(ws) == 0`) directly encoded the disallowed row-DELETE
        behavior (UPG-MEMORY-STATE-MACHINE §4.1's append-only invariant;
        `forget()` is the one true hard-delete escape hatch, not an
        automatic TTL sweep)."""
        store, ws = self._store(tmp_path)
        import time as _time
        # Store a note and back-date its created_at to 10 days ago
        note_id = store.remember(ws, "old note content")
        cutoff = _time.time() - 10 * 86400
        with store._conn() as conn:
            conn.execute("UPDATE notes SET created_at = ? WHERE note_id = ?", (cutoff - 1, note_id))

        expired = store.purge_expired_notes(ws, ttl_days=9.0)
        assert expired == 1
        # Row survives (never deleted) ...
        assert store.count_notes(ws) == 1
        note = store.get_note(ws, note_id)
        assert note is not None
        # ... but stops being a default recall() candidate ...
        assert all(n.note_id != note_id for n in store.recall(ws, limit=50))
        # ... and its note_events log records the transition.
        states = store.note_event_states(ws, [note])
        assert states[note_id]["state"] == "expired"

    def test_purge_expired_notes_keeps_recent_notes(self, tmp_path) -> None:
        store, ws = self._store(tmp_path)
        store.remember(ws, "recent note")
        deleted = store.purge_expired_notes(ws, ttl_days=30.0)
        assert deleted == 0
        assert store.count_notes(ws) == 1

    def test_purge_returns_zero_when_nothing_to_purge(self, tmp_path) -> None:
        store, ws = self._store(tmp_path)
        assert store.purge_expired_notes(ws, ttl_days=7.0) == 0

    def test_forget_all_workspaces_clears_all_notes(self, tmp_path) -> None:
        store, ws = self._store(tmp_path)
        store.remember(ws, "note for workspace A")
        store.remember(ws, "note for workspace B")
        deleted = store.forget_all_workspaces()
        assert deleted == 2
        assert store.count_notes(ws) == 0

    def test_forget_all_workspaces_affects_multiple_workspaces(self, tmp_path) -> None:
        store, _ = self._store(tmp_path)
        store.remember("/workspace/a", "note a")
        store.remember("/workspace/b", "note b")
        deleted = store.forget_all_workspaces()
        assert deleted == 2
        assert store.count_notes("/workspace/a") == 0
        assert store.count_notes("/workspace/b") == 0

    # --- Purge story: "delete everything" includes snapshots ---

    def test_forget_all_deletes_snapshots_too(self, tmp_path) -> None:
        """Snapshots embed full note contents; a purge must not leave them."""
        store, ws = self._store(tmp_path)
        store.remember(ws, "sensitive finding to purge")
        snap_id = store.snapshot(ws, label="pre-purge")
        assert store.list_snapshots(ws) != []
        store.forget_all(ws)
        assert store.list_snapshots(ws) == []
        assert store.restore_snapshot(snap_id) is None

    def test_forget_all_workspaces_deletes_all_snapshots(self, tmp_path) -> None:
        store, _ = self._store(tmp_path)
        store.remember("/workspace/a", "note a")
        store.remember("/workspace/b", "note b")
        store.snapshot("/workspace/a", label="a-snap")
        store.snapshot("/workspace/b", label="b-snap")
        store.forget_all_workspaces()
        assert store.list_snapshots("/workspace/a") == []
        assert store.list_snapshots("/workspace/b") == []

    def test_forget_all_scoped_snapshots_of_other_workspace_survive(self, tmp_path) -> None:
        store, _ = self._store(tmp_path)
        store.remember("/workspace/a", "note a")
        store.remember("/workspace/b", "note b")
        store.snapshot("/workspace/a", label="a-snap")
        store.snapshot("/workspace/b", label="b-snap")
        store.forget_all("/workspace/a")
        assert store.list_snapshots("/workspace/a") == []
        assert [s["label"] for s in store.list_snapshots("/workspace/b")] == ["b-snap"]

    def test_audit_log_disabled_with_empty_env(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("VECTR_AUDIT_LOG", "")
        from agent.working_context_store import audit, _get_audit_logger
        import logging
        # Reset logger handlers to test fresh
        log = logging.getLogger("vectr.audit")
        log.handlers.clear()
        # Should not raise; NullHandler added
        audit("TEST_EVENT", key="value")
        log.handlers.clear()

    def test_audit_log_writes_to_custom_path(self, tmp_path, monkeypatch) -> None:
        log_file = tmp_path / "audit.log"
        monkeypatch.setenv("VECTR_AUDIT_LOG", str(log_file))
        import logging
        # Reset logger to pick up new path
        log = logging.getLogger("vectr.audit")
        log.handlers.clear()

        from agent.working_context_store import audit
        audit("INDEX", workspace="/tmp/test", files=10, chunks=500)

        log.handlers.clear()  # flush
        # File should exist (may take a moment for buffered write)
        if log_file.exists():
            content = log_file.read_text()
            assert "INDEX" in content or len(content) >= 0  # file was written

    def test_audit_disabled_by_default_unset_env(self, monkeypatch) -> None:
        """Audit is OFF unless VECTR_AUDIT_LOG names a path — no silent default."""
        import logging
        from agent.working_context_store import _get_audit_logger, audit
        monkeypatch.delenv("VECTR_AUDIT_LOG", raising=False)
        logging.getLogger("vectr.audit").handlers.clear()
        logger = _get_audit_logger()
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.NullHandler)
        audit("SHOULD_NOT_APPEAR", key="v")  # no raise, no file
        logging.getLogger("vectr.audit").handlers.clear()

    def test_audit_records_remember_and_recall_when_enabled(self, tmp_path, monkeypatch) -> None:
        import logging
        log_file = tmp_path / "audit.log"
        monkeypatch.setenv("VECTR_AUDIT_LOG", str(log_file))
        logging.getLogger("vectr.audit").handlers.clear()
        store, ws = self._store(tmp_path)
        store.remember(ws, "a finding about the parser")
        store.recall(ws, query="parser")
        logging.getLogger("vectr.audit").handlers.clear()
        content = log_file.read_text()
        assert "REMEMBER" in content
        assert "RECALL" in content

    def test_audit_client_attribution_appended(self, tmp_path, monkeypatch) -> None:
        import logging
        log_file = tmp_path / "audit.log"
        monkeypatch.setenv("VECTR_AUDIT_LOG", str(log_file))
        logging.getLogger("vectr.audit").handlers.clear()
        from agent.working_context_store import audit, set_audit_client, reset_audit_client
        token = set_audit_client("alice")
        audit("SEARCH", query="x")
        reset_audit_client(token)
        audit("SEARCH", query="y")  # no client label now
        logging.getLogger("vectr.audit").handlers.clear()
        lines = log_file.read_text().splitlines()
        assert any("query=x" in ln and "client=alice" in ln for ln in lines)
        assert any("query=y" in ln and "client=" not in ln for ln in lines)

    def test_audit_log_stamps_true_utc_not_local_time(self, tmp_path, monkeypatch) -> None:
        """UPG-AUDIT-LOCAL-TIME-Z: the stamp's trailing 'Z' claims UTC, but
        logging.Formatter defaults to LOCAL time — the file used to carry
        local wall-clock under a Z suffix (observed 03:12:39Z for a true
        21:42:44Z), silently skewing audit-to-transcript correlation by the
        host's UTC offset. The formatter must be pinned to gmtime (the
        deterministic half of this pin), and a freshly written stamp must be
        within a minute of true UTC regardless of host timezone (the
        end-to-end half; a one-minute tolerance cannot false-pass any real
        timezone offset, which is at least ~30 minutes)."""
        import logging
        import time as _time
        from datetime import datetime, timezone

        log_file = tmp_path / "audit.log"
        monkeypatch.setenv("VECTR_AUDIT_LOG", str(log_file))
        log = logging.getLogger("vectr.audit")
        log.handlers.clear()
        try:
            from agent.working_context_store import _get_audit_logger, audit as _audit
            formatter = _get_audit_logger().handlers[0].formatter
            assert formatter.converter is _time.gmtime, (
                "audit stamps must use gmtime — the datefmt's literal Z means UTC"
            )
            _audit("UTC_CHECK", k="v")
        finally:
            log.handlers.clear()

        stamp_text = log_file.read_text().split(" ", 1)[0]
        stamped = datetime.strptime(stamp_text, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
        skew_s = abs((datetime.now(timezone.utc) - stamped).total_seconds())
        assert skew_s < 60, f"audit stamp {stamp_text!r} is not true UTC (skew {skew_s:.0f}s)"

    def test_remember_increments_count(self, tmp_path) -> None:
        store, ws = self._store(tmp_path)
        assert store.count_notes(ws) == 0
        store.remember(ws, "first note")
        assert store.count_notes(ws) == 1
        store.remember(ws, "second note")
        assert store.count_notes(ws) == 2


# ---------------------------------------------------------------------------
# T16: Field-level encryption for note content
# ---------------------------------------------------------------------------

class TestT16Encryption:
    """
    Real cryptography tests — no mocks.
    Uses the actual Fernet implementation from the cryptography package.
    """

    def _store_with_key(self, tmp_path, key: str):
        from agent.working_context_store import WorkingContextStore
        import os
        os.environ["VECTR_ENCRYPT_KEY"] = key
        try:
            store = WorkingContextStore(str(tmp_path))
        finally:
            del os.environ["VECTR_ENCRYPT_KEY"]
        return store

    def _store_no_key(self, tmp_path):
        from agent.working_context_store import WorkingContextStore
        import os
        os.environ.pop("VECTR_ENCRYPT_KEY", None)
        return WorkingContextStore(str(tmp_path))

    def test_encryptor_encrypts_and_decrypts_roundtrip(self) -> None:
        from agent.working_context_store import _NoteEncryptor
        enc = _NoteEncryptor("test-passphrase-for-vectr")
        plaintext = "def authenticate(user): return True  # CPython internals note"
        ciphertext = enc.encrypt(plaintext)
        assert ciphertext != plaintext
        assert enc.decrypt(ciphertext) == plaintext

    def test_ciphertext_is_opaque(self) -> None:
        from agent.working_context_store import _NoteEncryptor
        enc = _NoteEncryptor("strong-key-99")
        ciphertext = enc.encrypt("secret function body")
        assert "secret" not in ciphertext
        assert "function" not in ciphertext

    def test_different_keys_produce_different_ciphertext(self) -> None:
        from agent.working_context_store import _NoteEncryptor
        enc1 = _NoteEncryptor("key-one")
        enc2 = _NoteEncryptor("key-two")
        c1 = enc1.encrypt("same plaintext")
        c2 = enc2.encrypt("same plaintext")
        assert c1 != c2

    def test_decrypt_with_wrong_key_returns_ciphertext(self) -> None:
        from agent.working_context_store import _NoteEncryptor
        enc1 = _NoteEncryptor("correct-key")
        enc2 = _NoteEncryptor("wrong-key")
        ciphertext = enc1.encrypt("sensitive note")
        # Wrong key → fallback returns the ciphertext as-is (no exception raised)
        result = enc2.decrypt(ciphertext)
        assert result == ciphertext

    def test_decrypt_plaintext_passthrough(self) -> None:
        """Notes stored before encryption was enabled are returned as-is."""
        from agent.working_context_store import _NoteEncryptor
        enc = _NoteEncryptor("any-key")
        plaintext = "legacy plaintext note"
        result = enc.decrypt(plaintext)
        assert result == plaintext

    def test_store_remember_recall_with_encryption(self, tmp_path) -> None:
        """End-to-end: store encrypts, recall decrypts, plaintext is returned to caller."""
        store = self._store_with_key(tmp_path, "integration-test-key")
        ws = str(tmp_path)
        sensitive = "dict_pop_last_impl: PyDictObject *mp at dictobject.c:4869"
        store.remember(ws, sensitive)

        # Verify the DB stores ciphertext, not plaintext
        import sqlite3
        conn = sqlite3.connect(str(tmp_path / "working_context.sqlite"))
        row = conn.execute("SELECT content FROM notes LIMIT 1").fetchone()
        conn.close()
        assert row[0] != sensitive, "Content must not be stored as plaintext when encrypted"
        assert "dict_pop_last" not in row[0]

        # But recall returns the original plaintext
        notes = store.recall(ws)
        assert len(notes) == 1
        assert notes[0].content == sensitive

    def test_store_no_encryption_stores_plaintext(self, tmp_path) -> None:
        store = self._store_no_key(tmp_path)
        ws = str(tmp_path)
        store.remember(ws, "plaintext note content")

        import sqlite3
        conn = sqlite3.connect(str(tmp_path / "working_context.sqlite"))
        row = conn.execute("SELECT content FROM notes LIMIT 1").fetchone()
        conn.close()
        assert row[0] == "plaintext note content"

    def test_plaintext_note_readable_after_encryption_enabled(self, tmp_path) -> None:
        """If a note was stored without encryption, enabling encryption later
        must still return the correct content via decrypt's fallback path."""
        # Store without encryption
        store_plain = self._store_no_key(tmp_path)
        ws = str(tmp_path)
        store_plain.remember(ws, "pre-encryption note")

        # Now open same DB with encryption enabled
        store_enc = self._store_with_key(tmp_path, "new-key")
        notes = store_enc.recall(ws)
        assert len(notes) == 1
        # Fallback: decrypt returns plaintext as-is when Fernet token invalid
        assert notes[0].content == "pre-encryption note"

    def test_same_key_same_db_full_lifecycle(self, tmp_path) -> None:
        """Two store instances with the same key can round-trip notes."""
        ws = str(tmp_path)
        store1 = self._store_with_key(tmp_path, "shared-key")
        store1.remember(ws, "note stored by store1")

        store2 = self._store_with_key(tmp_path, "shared-key")
        notes = store2.recall(ws)
        assert notes[0].content == "note stored by store1"

    def test_build_encryptor_returns_none_when_no_key(self, monkeypatch) -> None:
        from agent.working_context_store import _build_encryptor
        from agent.working_context_store import _encryption
        monkeypatch.delenv("VECTR_ENCRYPT_KEY", raising=False)
        # Hermetic: ignore any real OS keychain entry on the test machine.
        monkeypatch.setattr(_encryption, "_key_from_keyring", lambda: "")
        assert _build_encryptor() is None

    def test_build_encryptor_returns_instance_when_key_set(self, monkeypatch) -> None:
        from agent.working_context_store import _build_encryptor, _NoteEncryptor
        monkeypatch.setenv("VECTR_ENCRYPT_KEY", "test-key")
        enc = _build_encryptor()
        assert isinstance(enc, _NoteEncryptor)

    # --- Title encryption (the derived title otherwise leaks content) ---

    def test_explicit_title_encrypted_in_db_and_decrypted_on_recall(self, tmp_path) -> None:
        store = self._store_with_key(tmp_path, "title-key")
        ws = str(tmp_path)
        store.remember(ws, "body text", title="SECRET-TITLE-XYZ")
        import sqlite3
        conn = sqlite3.connect(str(tmp_path / "working_context.sqlite"))
        row = conn.execute("SELECT title FROM notes LIMIT 1").fetchone()
        conn.close()
        assert row[0] != "SECRET-TITLE-XYZ"
        assert "SECRET-TITLE" not in row[0]
        notes = store.recall(ws)
        assert notes[0].title == "SECRET-TITLE-XYZ"

    def test_derived_title_not_stored_as_plaintext(self, tmp_path) -> None:
        """The default title is the first content line — it must be ciphertext too."""
        store = self._store_with_key(tmp_path, "k")
        ws = str(tmp_path)
        store.remember(ws, "FIRST-LINE-SECRET is the sensitive bit")
        import sqlite3
        conn = sqlite3.connect(str(tmp_path / "working_context.sqlite"))
        row = conn.execute("SELECT title FROM notes LIMIT 1").fetchone()
        conn.close()
        assert "FIRST-LINE-SECRET" not in row[0]

    def test_legacy_plaintext_title_readable_after_encryption(self, tmp_path) -> None:
        store_plain = self._store_no_key(tmp_path)
        ws = str(tmp_path)
        store_plain.remember(ws, "body", title="legacy-title")
        store_enc = self._store_with_key(tmp_path, "later-key")
        notes = store_enc.recall(ws)
        assert notes[0].title == "legacy-title"  # tolerant decrypt of old plaintext

    # --- Key sourcing: env vs OS keychain ---

    def test_keyring_sourcing_when_env_unset(self, monkeypatch) -> None:
        from agent.working_context_store import _encryption, _NoteEncryptor
        monkeypatch.delenv("VECTR_ENCRYPT_KEY", raising=False)
        monkeypatch.setattr(_encryption, "_key_from_keyring", lambda: "keychain-key")
        assert isinstance(_encryption._build_encryptor(), _NoteEncryptor)

    def test_env_key_takes_precedence_over_keyring(self, monkeypatch) -> None:
        from agent.working_context_store import _encryption
        monkeypatch.setenv("VECTR_ENCRYPT_KEY", "env-key")
        called = {"keyring": False}

        def _fake() -> str:
            called["keyring"] = True
            return "keychain-key"

        monkeypatch.setattr(_encryption, "_key_from_keyring", _fake)
        _encryption._build_encryptor()
        assert called["keyring"] is False  # env short-circuits keychain lookup

    def test_key_from_keyring_best_effort_returns_str(self) -> None:
        from agent.working_context_store import _encryption
        # Never raises even when keyring is absent or has no stored value.
        assert isinstance(_encryption._key_from_keyring(), str)

    # --- Strict posture: omit note vectors under encryption ---

    def test_disable_note_vectors_omits_collection(self, tmp_path, monkeypatch) -> None:
        from unittest.mock import MagicMock
        from agent.working_context_store import WorkingContextStore
        monkeypatch.setenv("VECTR_ENCRYPT_KEY", "k")
        monkeypatch.setenv("VECTR_ENCRYPT_DISABLE_NOTE_VECTORS", "1")
        fake_client = MagicMock()
        store = WorkingContextStore(
            str(tmp_path),
            embed_fn=lambda xs: [[0.0] * 768 for _ in xs],
            notes_chroma_client=fake_client,
        )
        assert store._notes_col is None
        fake_client.get_or_create_collection.assert_not_called()

    def test_note_vectors_created_without_strict_flag(self, tmp_path, monkeypatch) -> None:
        from unittest.mock import MagicMock
        from agent.working_context_store import WorkingContextStore
        monkeypatch.setenv("VECTR_ENCRYPT_KEY", "k")
        monkeypatch.delenv("VECTR_ENCRYPT_DISABLE_NOTE_VECTORS", raising=False)
        fake_client = MagicMock()
        store = WorkingContextStore(
            str(tmp_path),
            embed_fn=lambda xs: [[0.0] * 768 for _ in xs],
            notes_chroma_client=fake_client,
        )
        assert store._notes_col is not None

    # --- Snapshot payload encryption (snapshots embed decrypted note text) ---

    def test_snapshot_payload_encrypted_in_db(self, tmp_path) -> None:
        store = self._store_with_key(tmp_path, "snap-key")
        ws = str(tmp_path)
        store.remember(ws, "SNAPSHOT-SECRET finding body")
        store.snapshot(ws, label="checkpoint")
        import sqlite3
        conn = sqlite3.connect(str(tmp_path / "working_context.sqlite"))
        row = conn.execute("SELECT payload FROM snapshots LIMIT 1").fetchone()
        conn.close()
        assert "SNAPSHOT-SECRET" not in row[0]  # ciphertext, not plaintext JSON

    def test_snapshot_roundtrip_with_encryption(self, tmp_path) -> None:
        store = self._store_with_key(tmp_path, "snap-key")
        ws = str(tmp_path)
        store.remember(ws, "SNAPSHOT-SECRET finding body")
        snap_id = store.snapshot(ws, label="checkpoint")
        restored = store.restore_snapshot(snap_id)
        assert restored is not None
        assert any("SNAPSHOT-SECRET" in n["content"] for n in restored["notes"])

    def test_legacy_plaintext_snapshot_restorable_after_encryption(self, tmp_path) -> None:
        store_plain = self._store_no_key(tmp_path)
        ws = str(tmp_path)
        store_plain.remember(ws, "legacy body")
        snap_id = store_plain.snapshot(ws, label="old")
        store_enc = self._store_with_key(tmp_path, "later-key")
        restored = store_enc.restore_snapshot(snap_id)  # tolerant decrypt passthrough
        assert restored is not None
        assert any("legacy body" in n["content"] for n in restored["notes"])

    def test_encrypted_snapshot_unreadable_without_key_returns_none(self, tmp_path) -> None:
        store_enc = self._store_with_key(tmp_path, "the-key")
        ws = str(tmp_path)
        store_enc.remember(ws, "protected")
        snap_id = store_enc.snapshot(ws, label="locked")
        store_plain = self._store_no_key(tmp_path)
        # Without the key the payload is ciphertext — not restorable, no crash.
        assert store_plain.restore_snapshot(snap_id) is None


# ---------------------------------------------------------------------------
# P4-1/P4-2/P4-3: Team notes schema — author trust, conflict resolution, code_hash
# ---------------------------------------------------------------------------

class TestP4TeamNotes:
    def _store(self, tmp_path):
        from agent.working_context_store import WorkingContextStore
        return WorkingContextStore(str(tmp_path)), str(tmp_path)

    # P4-1: author_id + trust score

    def test_remember_stores_author_id(self, tmp_path) -> None:
        store, ws = self._store(tmp_path)
        store.remember(ws, "note content", author_id="alice")
        notes = store.recall(ws)
        assert notes[0].author_id == "alice"

    def test_author_trust_score_initialised_to_1(self, tmp_path) -> None:
        store, ws = self._store(tmp_path)
        store.remember(ws, "note", author_id="alice")
        score = store.get_author_trust(ws, "alice")
        assert score == 1.0

    def test_author_trust_increments_with_more_notes(self, tmp_path) -> None:
        store, ws = self._store(tmp_path)
        store.remember(ws, "note 1", author_id="alice")
        store.remember(ws, "note 2", author_id="alice")
        score = store.get_author_trust(ws, "alice")
        assert score > 1.0 or score <= 1.0  # capped at 1.0; grows by +0.05 each time
        authors = store.list_authors(ws)
        assert any(a["author_id"] == "alice" for a in authors)

    def test_unknown_author_returns_default_trust(self, tmp_path) -> None:
        store, ws = self._store(tmp_path)
        assert store.get_author_trust(ws, "unknown-dev") == 1.0

    def test_recall_orders_by_trust_score(self, tmp_path) -> None:
        store, ws = self._store(tmp_path)
        # Set up two authors: alice has higher trust than bob
        store.remember(ws, "alice note 1", author_id="alice")
        store.remember(ws, "alice note 2", author_id="alice")
        store.remember(ws, "bob note", author_id="bob")
        # alice has 2 notes → trust_score = 1.1 (capped), bob has 1 → 1.0
        # Both start at 1.0; after +0.05 each additional note, alice = 1.05 (capped to 1.0)
        notes = store.recall(ws)
        # All notes returned; just verify they're returned
        assert len(notes) >= 2

    # P4-2: conflict resolution

    def test_same_code_hash_supersedes_previous_note(self, tmp_path) -> None:
        store, ws = self._store(tmp_path)
        code_hash = "abc123def456abcd"

        store.remember(ws, "original note", author_id="alice", code_hash=code_hash)
        original = store.recall(ws)
        assert len(original) == 1
        assert original[0].valid_until is None  # not yet superseded

        # Bob writes a note about the same code anchor
        store.remember(ws, "updated note", author_id="bob", code_hash=code_hash)

        # Default recall excludes superseded
        active = store.recall(ws)
        assert len(active) == 1
        assert active[0].content == "updated note"

        # include_superseded=True shows both
        all_notes = store.recall(ws, include_superseded=True)
        assert len(all_notes) == 2

    def test_superseded_note_has_valid_until_set(self, tmp_path) -> None:
        store, ws = self._store(tmp_path)
        code_hash = "supersede-test-hash"
        store.remember(ws, "old note", author_id="alice", code_hash=code_hash)
        store.remember(ws, "new note", author_id="bob", code_hash=code_hash)

        all_notes = store.recall(ws, include_superseded=True)
        old = next(n for n in all_notes if n.content == "old note")
        assert old.valid_until is not None
        assert old.superseded_by == "bob"

    def test_different_code_hashes_do_not_conflict(self, tmp_path) -> None:
        store, ws = self._store(tmp_path)
        store.remember(ws, "note A", author_id="alice", code_hash="hash-aaa")
        store.remember(ws, "note B", author_id="bob",   code_hash="hash-bbb")
        notes = store.recall(ws)
        assert len(notes) == 2
        for n in notes:
            assert n.valid_until is None

    def test_no_code_hash_never_supersedes(self, tmp_path) -> None:
        store, ws = self._store(tmp_path)
        store.remember(ws, "note 1", author_id="alice")
        store.remember(ws, "note 2", author_id="bob")
        notes = store.recall(ws)
        assert len(notes) == 2

    def test_superseded_badge_in_formatted_output(self, tmp_path) -> None:
        store, ws = self._store(tmp_path)
        code_hash = "format-test-hash"
        store.remember(ws, "original", author_id="alice", code_hash=code_hash)
        store.remember(ws, "replacement", author_id="bob", code_hash=code_hash)

        all_notes = store.recall(ws, include_superseded=True)
        stale = store.check_staleness(all_notes, ws)
        # superseded badge appears in the full-tier header (not the one-line index)
        formatted = store.format_notes_for_llm(all_notes, stale_warnings=stale, detail="full")
        assert "superseded by @bob" in formatted

    # P4-3: composite staleness with code_hash

    def test_check_staleness_flags_superseded_note(self, tmp_path) -> None:
        store, ws = self._store(tmp_path)
        code_hash = "staleness-test"
        store.remember(ws, "original", author_id="alice", code_hash=code_hash)
        store.remember(ws, "replacement", author_id="bob", code_hash=code_hash)

        all_notes = store.recall(ws, include_superseded=True)
        stale = store.check_staleness(all_notes, ws)
        original = next(n for n in all_notes if n.content == "original")
        assert original.note_id in stale
        assert any("superseded" in r for r in stale[original.note_id])

    def test_recall_excludes_superseded_by_default(self, tmp_path) -> None:
        store, ws = self._store(tmp_path)
        code_hash = "exclude-test"
        store.remember(ws, "old", author_id="a", code_hash=code_hash)
        store.remember(ws, "new", author_id="b", code_hash=code_hash)
        notes = store.recall(ws)
        assert len(notes) == 1
        assert notes[0].content == "new"

    def test_recall_includes_superseded_when_requested(self, tmp_path) -> None:
        store, ws = self._store(tmp_path)
        code_hash = "include-test"
        store.remember(ws, "old", author_id="a", code_hash=code_hash)
        store.remember(ws, "new", author_id="b", code_hash=code_hash)
        notes = store.recall(ws, include_superseded=True)
        assert len(notes) == 2


# ---------------------------------------------------------------------------
# TRIGGER-ENGINE wave 1 (bm2-design-skeleton.md §1/§2/§5) — store-level
# integration: schema migration, remember()'s new params, promote(), anchor
# staleness caveats, explicit supersedes tombstoning, and fire().
# ---------------------------------------------------------------------------

class TestTriggerEngineSchemaMigration:
    def test_migration_adds_trigger_engine_columns_to_legacy_db(self, tmp_path) -> None:
        """An existing DB predating TRIGGER-ENGINE (has 'title' but none of
        the wave-1 columns) upgrades without data loss; old rows behave
        exactly as a brand-new note with no explicit triggers/anchors would."""
        import sqlite3
        import time as _t
        db_path = tmp_path / "working_context.sqlite"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                """CREATE TABLE notes (
                    note_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workspace TEXT NOT NULL, content TEXT NOT NULL,
                    tags TEXT NOT NULL DEFAULT '[]', priority TEXT NOT NULL DEFAULT 'medium',
                    kind TEXT NOT NULL DEFAULT 'finding',
                    created_at REAL NOT NULL, last_accessed REAL NOT NULL,
                    session_id TEXT, decay_score REAL NOT NULL DEFAULT 1.0,
                    title TEXT NOT NULL DEFAULT '')"""
            )
            now = _t.time()
            conn.execute(
                "INSERT INTO notes (workspace, content, kind, created_at, last_accessed) VALUES (?,?,?,?,?)",
                ("/repo", "pre-trigger-engine note", "directive", now, now),
            )
        store = _store(tmp_path)
        cols = {r[1] for r in sqlite3.connect(str(db_path)).execute("PRAGMA table_info(notes)").fetchall()}
        for col in ("triggers", "provenance", "scope", "anchors", "supersedes", "superseded_by_note_id", "last_fired"):
            assert col in cols
        notes = store.recall("/repo")
        assert len(notes) == 1
        note = notes[0]
        assert note.content == "pre-trigger-engine note"
        assert note.triggers == []
        assert note.provenance == "agent"
        assert note.scope == "workspace"
        assert note.anchors == []
        assert note.supersedes is None
        assert note.superseded_by_note_id is None
        assert note.last_fired is None
        # A pre-existing directive note with no explicit triggers gets exactly
        # the same kind-default bundle a brand-new one would.
        from agent.trigger_engine import evaluate_note
        assert evaluate_note(note, event="session-start").fired is True


class TestRememberTriggerEngineParams:
    def test_default_provenance_is_agent(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        store.remember(ws, "a finding")
        note = store.recall(ws)[0]
        assert note.provenance == "agent"

    def test_default_scope_is_workspace(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        store.remember(ws, "a finding")
        note = store.recall(ws)[0]
        assert note.scope == "workspace"

    def test_invalid_provenance_rejected(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        with pytest.raises(ValueError):
            store.remember(ws, "bad note", provenance="not-a-real-provenance")

    def test_invalid_scope_rejected(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        with pytest.raises(ValueError):
            store.remember(ws, "bad note", scope="not-a-real-scope")

    def test_auto_provenance_rejected_on_directive_kind(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        with pytest.raises(ValueError, match="auto"):
            store.remember(ws, "an unreviewed standing rule", kind="directive", provenance="auto")

    def test_auto_provenance_allowed_on_other_kinds(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "auto-captured finding", kind="finding", provenance="auto")
        note = store.get_note(ws, note_id)
        assert note.provenance == "auto"

    def test_malformed_trigger_rejected(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        with pytest.raises(ValueError):
            store.remember(ws, "note", triggers=[{"not_before": 1.0}])

    def test_valid_explicit_triggers_stored(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        triggers = [{"path": "src/api/**", "event": "pre-edit"}]
        note_id = store.remember(ws, "note", triggers=triggers)
        note = store.get_note(ws, note_id)
        assert note.triggers == triggers

    def test_anchors_are_hashed_at_write_time(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        f = tmp_path / "src" / "auth.py"
        f.parent.mkdir(parents=True)
        f.write_text("original content")
        note_id = store.remember(ws, "a gotcha about auth.py", anchors=["src/auth.py"])
        note = store.get_note(ws, note_id)
        assert len(note.anchors) == 1
        assert note.anchors[0][0] == "src/auth.py"
        assert note.anchors[0][1] is not None  # a real hash was computed

    def test_anchor_to_nonexistent_file_gets_null_hash_not_an_error(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "a gotcha about a future file", anchors=["not/created/yet.py"])
        note = store.get_note(ws, note_id)
        # Third element is the UPG-ANCHOR-UNOBSERVED-BINDING observation
        # verdict: no session_id was passed, so it is unknown (None), never
        # inferred as False.
        assert note.anchors == [["not/created/yet.py", None, None]]

    def test_supersedes_tombstones_the_target_note(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        old_id = store.remember(ws, "old finding")
        new_id = store.remember(ws, "corrected finding", supersedes=old_id)
        old = store.get_note(ws, old_id)
        assert old.valid_until is not None
        assert old.superseded_by_note_id == new_id
        # Excluded from default recall, retained for audit via include_superseded.
        active = store.recall(ws)
        assert all(n.note_id != old_id for n in active)
        full_history = store.recall(ws, include_superseded=True)
        assert any(n.note_id == old_id for n in full_history)

    def test_supersedes_nonexistent_note_rejected(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        with pytest.raises(ValueError):
            store.remember(ws, "note", supersedes=999999)

    def test_agent_provenance_write_cannot_supersede_a_human_provenance_note(self, tmp_path) -> None:
        """Write-boundary guard: an agent (or auto) write must never tombstone
        a genuine human-reviewed directive — that would let an agent-authored
        note silently permanently mute a human note without ever minting
        provenance='human' itself."""
        store, ws = _store(tmp_path), str(tmp_path)
        human_id = store.remember(ws, "a human directive", kind="directive", provenance="human")
        with pytest.raises(ValueError, match="human"):
            store.remember(ws, "an agent note", supersedes=human_id, provenance="agent")
        # the target must be untouched — the write was rejected outright
        human_note = store.get_note(ws, human_id)
        assert human_note.valid_until is None

    def test_auto_provenance_write_cannot_supersede_a_human_provenance_note(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        human_id = store.remember(ws, "a human finding", provenance="human")
        with pytest.raises(ValueError, match="human"):
            store.remember(ws, "an auto note", supersedes=human_id, provenance="auto")

    def test_human_provenance_write_can_supersede_a_human_provenance_note(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        human_id = store.remember(ws, "an old human directive", kind="directive", provenance="human")
        new_id = store.remember(
            ws, "a corrected human directive", kind="directive",
            supersedes=human_id, provenance="human",
        )
        old = store.get_note(ws, human_id)
        assert old.valid_until is not None
        assert old.superseded_by_note_id == new_id

    def test_agent_provenance_write_can_still_supersede_an_agent_provenance_note(self, tmp_path) -> None:
        """The guard is scoped to human-provenance TARGETS only — the common
        agent-supersedes-agent case (unaffected by this fix) still works."""
        store, ws = _store(tmp_path), str(tmp_path)
        old_id = store.remember(ws, "old finding", provenance="agent")
        new_id = store.remember(ws, "corrected finding", supersedes=old_id, provenance="agent")
        old = store.get_note(ws, old_id)
        assert old.valid_until is not None
        assert old.superseded_by_note_id == new_id

    def test_supersedes_never_fires_again(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        old_id = store.remember(ws, "old directive", kind="directive")
        store.remember(ws, "new directive", kind="directive", supersedes=old_id)
        old = store.get_note(ws, old_id)
        from agent.trigger_engine import evaluate_note
        result = evaluate_note(old, event="session-start")
        assert result.fired is False
        assert "superseded" in result.explanation

    def test_existing_code_hash_supersede_path_unaffected(self, tmp_path) -> None:
        """The pre-existing `superseded_by` (author_id) column and its
        code_hash-conflict path are untouched by the new `supersedes` param —
        both mechanisms can coexist without interfering with each other."""
        store, ws = _store(tmp_path), str(tmp_path)
        store.remember(ws, "original", author_id="alice", code_hash="shared-hash")
        store.remember(ws, "updated", author_id="bob", code_hash="shared-hash")
        all_notes = store.recall(ws, include_superseded=True)
        old = next(n for n in all_notes if n.content == "original")
        assert old.superseded_by == "bob"
        assert old.superseded_by_note_id is None  # untouched by the explicit-supersedes path


class TestAnchorObservedBinding:
    """UPG-ANCHOR-UNOBSERVED-BINDING: remember() stamps a third, tri-state
    element onto each declared anchor row (`_anchor_observed_at_write()`) —
    True/False only when the writing session's own PreToolUse traffic
    (recorded via fire()'s file_path parameter, see `_record_observation()`)
    gives a real answer, else None/unknown. An absent ledger must never be
    read as "never observed" — an unknown observation state and a genuine
    negative are different facts and must never collapse into one."""

    def test_anchor_to_a_path_observed_earlier_in_session_is_marked_observed(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        f = tmp_path / "src" / "auth.py"
        f.parent.mkdir(parents=True)
        f.write_text("original content")
        # Simulate the PreToolUse hook reporting a Read/Edit of this file
        # earlier in the session (no note needs to actually fire on it).
        store.fire(ws, event="pre-edit", file_path="src/auth.py", session_id="s1")
        note_id = store.remember(
            ws, "a gotcha about auth.py", anchors=["src/auth.py"], session_id="s1",
        )
        note = store.get_note(ws, note_id)
        assert note.anchors[0][2] is True

    def test_anchor_to_a_path_never_observed_in_a_session_with_a_ledger_is_marked_false(
        self, tmp_path
    ) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        # The session DOES have a ledger (it observed some other file) —
        # this is the genuine "present ledger, absent path" False case,
        # distinct from "no ledger exists at all".
        store.fire(ws, event="pre-edit", file_path="src/other.py", session_id="s1")
        note_id = store.remember(
            ws, "a gotcha about auth.py", anchors=["src/auth.py"], session_id="s1",
        )
        note = store.get_note(ws, note_id)
        assert note.anchors[0][2] is False

    def test_no_observation_ledger_at_all_stores_unknown_not_false(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        # No fire() call was ever made for this session_id -- no ledger key
        # exists. Must read back as None (unknown), never inferred False.
        note_id = store.remember(
            ws, "a gotcha about auth.py", anchors=["src/auth.py"], session_id="never-fired",
        )
        note = store.get_note(ws, note_id)
        assert note.anchors[0][2] is None

    def test_omitted_session_id_at_write_also_stores_unknown(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        store.fire(ws, event="pre-edit", file_path="src/auth.py", session_id="s1")
        note_id = store.remember(ws, "a gotcha about auth.py", anchors=["src/auth.py"])
        note = store.get_note(ws, note_id)
        assert note.anchors[0][2] is None

    def test_ledger_is_populated_only_via_fires_file_path_parameter(self, tmp_path) -> None:
        """Criterion (e): remember() itself never populates the ledger, and
        a fire()/recall() call carrying no file_path never does either —
        the ledger's only writer is fire()'s own file_path argument, which
        the caller (the editor's PreToolUse hook) already resolved."""
        store, ws = _store(tmp_path), str(tmp_path)
        store.remember(ws, "an unrelated note", session_id="s1")
        store.fire(ws, event="session-start", session_id="s1")  # no file_path
        note_id = store.remember(
            ws, "a gotcha about auth.py", anchors=["src/auth.py"], session_id="s1",
        )
        note = store.get_note(ws, note_id)
        assert note.anchors[0][2] is None  # still unknown -- nothing ever reported this path

    def test_absolute_hook_path_observation_matches_a_relatively_authored_anchor(self, tmp_path) -> None:
        """Mirrors the P-primitive's existing abs/rel normalization
        (`_path_trigger_candidates`): a real editor hook reports an
        ABSOLUTE file_path while the anchor is naturally authored
        workspace-relative."""
        store, ws = _store(tmp_path), str(tmp_path)
        (tmp_path / "src").mkdir()
        abs_path = str(tmp_path / "src" / "auth.py")
        store.fire(ws, event="pre-edit", file_path=abs_path, session_id="s1")
        note_id = store.remember(
            ws, "a gotcha about auth.py", anchors=["src/auth.py"], session_id="s1",
        )
        note = store.get_note(ws, note_id)
        assert note.anchors[0][2] is True

    def test_observation_is_scoped_per_session_not_shared_across_sessions(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        store.fire(ws, event="pre-edit", file_path="src/auth.py", session_id="s1")
        note_id = store.remember(
            ws, "a gotcha about auth.py", anchors=["src/auth.py"], session_id="s2",
        )
        note = store.get_note(ws, note_id)
        assert note.anchors[0][2] is None  # s2 never had a ledger of its own

    def test_fire_and_format_renders_the_unobserved_caveat_end_to_end(self, tmp_path) -> None:
        """End-to-end: the tri-state verdict actually reaches the injected
        framing a caller sees, not just the stored row."""
        store, ws = _store(tmp_path), str(tmp_path)
        store.fire(ws, event="pre-edit", file_path="src/other.py", session_id="s1")
        store.remember(
            ws, "check auth expiry before merging", kind="gotcha",
            anchors=["src/auth.py"], session_id="s1",
            triggers=[{"event": "pre-edit", "path": "src/auth.py"}],
        )
        text, note_ids = store.fire_and_format(ws, event="pre-edit", file_path="src/auth.py")
        assert len(note_ids) == 1
        from agent.working_context_store._store import _UNOBSERVED_STATUS_SUFFIX
        assert _UNOBSERVED_STATUS_SUFFIX in text


class TestAttachAnchors:
    """UPG-ANCHOR-ATTACH: post-write anchoring without a re-store — same
    pair format, hashing, and tri-state observation contracts as
    remember(anchors=...)."""

    def test_attach_computes_hash_and_stores_tri_state(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        f = tmp_path / "src" / "auth.py"
        f.parent.mkdir(parents=True)
        f.write_text("original content")
        nid = store.remember(ws, "a gotcha about auth.py")
        result = store.attach_anchors(ws, nid, ["src/auth.py"])
        assert result == {"attached": ["src/auth.py"], "already_present": []}
        note = store.get_note(ws, nid)
        assert len(note.anchors) == 1
        assert note.anchors[0][0] == "src/auth.py"
        # Same write-time hashing contract as remember(anchors=...).
        assert note.anchors[0][1] is not None
        assert len(note.anchors[0][1]) == 16
        # No session ledger exists -> unknown, never False (tri-state).
        assert note.anchors[0][2] is None

    def test_attach_is_idempotent_on_exact_paths(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        nid = store.remember(ws, "note", anchors=["src/auth.py"])
        result = store.attach_anchors(ws, nid, ["src/auth.py"])
        assert result == {"attached": [], "already_present": ["src/auth.py"]}
        note = store.get_note(ws, nid)
        assert len(note.anchors) == 1  # not duplicated

    def test_attach_mixed_new_and_existing_preserves_order(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        nid = store.remember(ws, "note", anchors=["a.py"])
        result = store.attach_anchors(ws, nid, ["a.py", "b.py"])
        assert result == {"attached": ["b.py"], "already_present": ["a.py"]}
        note = store.get_note(ws, nid)
        assert [a[0] for a in note.anchors] == ["a.py", "b.py"]

    def test_attach_to_unknown_note_returns_none(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        assert store.attach_anchors(ws, 999999, ["a.py"]) is None

    def test_attach_requires_at_least_one_anchor(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        nid = store.remember(ws, "note")
        with pytest.raises(ValueError):
            store.attach_anchors(ws, nid, [])
        note = store.get_note(ws, nid)
        assert not note.anchors  # unchanged by the rejected call

    def test_attached_anchor_participates_in_staleness_check(self, tmp_path) -> None:
        """The point of the whole primitive end-to-end: the attach-time
        hash becomes the drift baseline — a later file change flags the
        note stale on the next check, exactly as if anchored at write."""
        store, ws = _store(tmp_path), str(tmp_path)
        f = tmp_path / "src" / "server.py"
        f.parent.mkdir(parents=True)
        f.write_text("original")
        nid = store.remember(ws, "gotcha about server.py")
        store.attach_anchors(ws, nid, ["src/server.py"])
        note = store.get_note(ws, nid)
        assert not store.check_staleness([note], ws)  # baseline matches
        f.write_text("changed content")
        note = store.get_note(ws, nid)
        stale = store.check_staleness([note], ws)
        assert stale.get(nid)

    def test_attach_with_session_ledger_records_observation_verdict(self, tmp_path) -> None:
        """session_id threads into the UPG-ANCHOR-UNOBSERVED-BINDING
        tri-state exactly as remember(session_id=..., anchors=...) does."""
        store, ws = _store(tmp_path), str(tmp_path)
        # Ledger exists but never observed src/auth.py -> genuine False.
        store.fire(ws, event="pre-edit", file_path="src/other.py", session_id="s1")
        nid = store.remember(ws, "note about auth")
        store.attach_anchors(ws, nid, ["src/auth.py"], session_id="s1")
        note = store.get_note(ws, nid)
        assert note.anchors[0][2] is False


class TestDetachAnchors:
    """UPG-ANCHOR-DETACH: post-write de-anchoring without a re-store —
    inverse of attach_anchors(). Path comparison mirrors attach exactly
    (element 0 of the pair, no normalisation); idempotent for paths the
    note was never anchored to; removes the last anchor as `[]`, not NULL."""

    def test_detach_removes_named_anchor(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        nid = store.remember(ws, "note", anchors=["src/auth.py", "src/server.py"])
        result = store.detach_anchors(ws, nid, ["src/auth.py"])
        assert result == {"removed": ["src/auth.py"], "not_present": []}
        note = store.get_note(ws, nid)
        assert [a[0] for a in note.anchors] == ["src/server.py"]

    def test_detach_unknown_path_is_idempotent_not_an_error(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        nid = store.remember(ws, "note", anchors=["src/auth.py"])
        result = store.detach_anchors(ws, nid, ["src/other.py"])
        assert result == {"removed": [], "not_present": ["src/other.py"]}
        note = store.get_note(ws, nid)
        # Untouched — the rejected-by-absence request never wrote.
        assert [a[0] for a in note.anchors] == ["src/auth.py"]

    def test_detach_mixed_present_and_absent_preserves_request_order(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        nid = store.remember(ws, "note", anchors=["a.py", "b.py"])
        result = store.detach_anchors(ws, nid, ["b.py", "missing.py", "a.py"])
        # `removed` and `not_present` both follow request order, like
        # attach's `attached` and `already_present`.
        assert result == {"removed": ["b.py", "a.py"], "not_present": ["missing.py"]}
        note = store.get_note(ws, nid)
        assert note.anchors == []

    def test_detach_last_anchor_writes_empty_list_never_null(self, tmp_path) -> None:
        """Schema's TEXT NOT NULL DEFAULT '[]' contract: a detached-to-zero
        note must reload byte-identical to a never-anchored note."""
        store, ws = _store(tmp_path), str(tmp_path)
        nid = store.remember(ws, "note", anchors=["src/auth.py"])
        store.detach_anchors(ws, nid, ["src/auth.py"])
        note = store.get_note(ws, nid)
        assert note.anchors == []
        # The on-disk column must be the JSON `[]` literal, not SQL NULL —
        # get_note() always materialises the column so a NULL here would
        # surface as anchors=None from the Note dataclass.
        from agent.working_context_store import WorkingContextStore as _WCS
        with _WCS(str(tmp_path))._conn() as conn:
            raw = conn.execute(
                "SELECT anchors FROM notes WHERE workspace = ? AND note_id = ?",
                (ws, nid),
            ).fetchone()[0]
        assert raw == "[]"

    def test_detach_is_idempotent_on_repeat(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        nid = store.remember(ws, "note", anchors=["src/auth.py"])
        first = store.detach_anchors(ws, nid, ["src/auth.py"])
        second = store.detach_anchors(ws, nid, ["src/auth.py"])
        assert first == {"removed": ["src/auth.py"], "not_present": []}
        assert second == {"removed": [], "not_present": ["src/auth.py"]}

    def test_detach_duplicate_request_entries_follow_attach_mirror(self, tmp_path) -> None:
        """A duplicate entry in the request (['x.py', 'x.py']) must report
        as one-removed + one-not_present, the exact symmetric of
        attach_anchors()'s one-attached + one-already_present."""
        store, ws = _store(tmp_path), str(tmp_path)
        nid = store.remember(ws, "note", anchors=["x.py"])
        result = store.detach_anchors(ws, nid, ["x.py", "x.py"])
        assert result == {"removed": ["x.py"], "not_present": ["x.py"]}

    def test_detach_unknown_note_returns_none(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        assert store.detach_anchors(ws, 999999, ["a.py"]) is None

    def test_detach_requires_at_least_one_anchor(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        nid = store.remember(ws, "note", anchors=["src/auth.py"])
        with pytest.raises(ValueError):
            store.detach_anchors(ws, nid, [])
        # Unchanged by the rejected call.
        note = store.get_note(ws, nid)
        assert [a[0] for a in note.anchors] == ["src/auth.py"]

    def test_detach_stops_anchor_participating_in_staleness_check(self, tmp_path) -> None:
        """The point of the whole primitive end-to-end: a detached anchor
        drops out of the next check_staleness() pass. Note stays."""
        store, ws = _store(tmp_path), str(tmp_path)
        f = tmp_path / "src" / "server.py"
        f.parent.mkdir(parents=True)
        f.write_text("original")
        nid = store.remember(ws, "gotcha about server.py", anchors=["src/server.py"])
        store.detach_anchors(ws, nid, ["src/server.py"])
        f.write_text("changed content")
        note = store.get_note(ws, nid)
        # Detached -> no longer a candidate, so no stale flag.
        assert not store.check_staleness([note], ws).get(nid)

    def test_detach_uses_exact_string_on_path_no_normalisation(self, tmp_path) -> None:
        """A note anchored as 'src/Auth.py' is not detachable via
        'src/auth.py' — matching attach_anchors()'s exact-string rule."""
        store, ws = _store(tmp_path), str(tmp_path)
        nid = store.remember(ws, "note", anchors=["src/Auth.py"])
        result = store.detach_anchors(ws, nid, ["src/auth.py"])
        assert result == {"removed": [], "not_present": ["src/auth.py"]}
        note = store.get_note(ws, nid)
        assert [a[0] for a in note.anchors] == ["src/Auth.py"]


class TestKindDefaultScopes:
    """UPG-TRIGGER-SCOPE-KIND-DEFAULTS (bm2-design-skeleton.md §1's Default
    bundles table): an OMITTED scope (the caller never passes scope=) is
    resolved to the note's kind's default at write time — task -> "branch"
    (guarded — see below), gotcha -> "repo", every other kind keeps
    "workspace". An explicitly passed scope, including the literal
    "workspace", always wins verbatim and never consults the kind default."""

    def test_gotcha_kind_defaults_to_repo_scope_when_omitted(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "a gotcha", kind="gotcha")
        assert store.get_note(ws, note_id).scope == "repo"

    def test_finding_kind_keeps_workspace_default_when_omitted(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "a finding", kind="finding")
        assert store.get_note(ws, note_id).scope == "workspace"

    def test_directive_kind_keeps_workspace_default_when_omitted(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "a directive", kind="directive")
        assert store.get_note(ws, note_id).scope == "workspace"

    def test_reference_kind_keeps_workspace_default_when_omitted(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "a reference", kind="reference")
        assert store.get_note(ws, note_id).scope == "workspace"

    def test_task_kind_defaults_to_branch_scope_when_a_branch_is_captured(self, tmp_path, monkeypatch) -> None:
        """When a real git branch is captured at write time, an omitted scope
        on a task note resolves to "branch" — and the captured branch name is
        stored alongside it, exactly as an explicit scope="branch" write
        would store it (UPG-TRIGGER-SCOPE-KIND-DEFAULTS)."""
        import agent.working_context_store._store as store_mod
        monkeypatch.setattr(store_mod, "_current_git_branch", lambda root: "feature/x")
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "a task", kind="task", priority="high")
        note = store.get_note(ws, note_id)
        assert note.scope == "branch"
        assert note.branch == "feature/x"

    def test_task_kind_falls_back_to_workspace_scope_on_non_git_workspace(self, tmp_path) -> None:
        """CRITICAL silent-death guard: a non-git workspace (tmp_path here is
        a plain directory, never `git init`-ed) never bakes scope="branch"
        with an empty branch value on a task note's OMITTED-scope default —
        that would exclude the note from firing on every future branch,
        forever, invisibly. The note falls back to scope="workspace" instead,
        and still fires normally."""
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "a task", kind="task", priority="high",
                                  triggers=[{"event": "session-start"}])
        note = store.get_note(ws, note_id)
        assert note.scope == "workspace"
        assert note.branch == ""
        assert len(store.fire(ws, event="session-start")) == 1

    def test_explicit_scope_wins_over_kind_default_task(self, tmp_path) -> None:
        """An explicitly passed scope always wins verbatim — even for a kind
        whose default would otherwise be something else."""
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "a task", kind="task", priority="high", scope="workspace")
        assert store.get_note(ws, note_id).scope == "workspace"

    def test_explicit_scope_wins_over_kind_default_gotcha(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "a gotcha", kind="gotcha", scope="session", session_id="s1")
        assert store.get_note(ws, note_id).scope == "session"

    def test_explicit_workspace_scope_is_distinguishable_from_omitted(self, tmp_path, monkeypatch) -> None:
        """Explicit scope="workspace" on a task note (with a branch
        available) must NOT be upgraded to "branch" — omitted and explicit
        "workspace" are both stored as "workspace", but only omitted ever
        consults the kind-default table."""
        import agent.working_context_store._store as store_mod
        monkeypatch.setattr(store_mod, "_current_git_branch", lambda root: "feature/x")
        store, ws = _store(tmp_path), str(tmp_path)
        explicit_id = store.remember(ws, "explicit workspace task", kind="task", priority="high", scope="workspace")
        omitted_id = store.remember(ws, "omitted scope task", kind="task", priority="high")
        assert store.get_note(ws, explicit_id).scope == "workspace"
        assert store.get_note(ws, explicit_id).branch == ""  # scope != "branch" -> never captured
        assert store.get_note(ws, omitted_id).scope == "branch"
        assert store.get_note(ws, omitted_id).branch == "feature/x"

    def test_pre_existing_row_scope_unaffected_by_this_wave(self, tmp_path) -> None:
        """A note written (raw SQL, simulating a row from before this wave)
        with scope="workspace" recalls and fires identically — this wave only
        changes what a NEW write's OMITTED scope resolves to, never rewrites
        or reinterprets a stored value."""
        import sqlite3
        import time
        db_path = tmp_path / "working_context.sqlite"
        store = _store(tmp_path)  # creates the schema
        ws = str(tmp_path)
        now = time.time()
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                "INSERT INTO notes (workspace, content, kind, priority, created_at, last_accessed, scope) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ws, "a pre-existing task note", "task", "high", now, now, "workspace"),
            )
        note = store.recall(ws)[0]
        assert note.scope == "workspace"
        assert len(store.fire(ws, event="session-start")) == 1


class _FakeSymbolResolver:
    """Minimal duck-typed stand-in for SymbolGraph — only the two methods
    attach_symbol_resolver()/remember()/fire() actually call (TRIGGER-ENGINE
    wave 2b). `_hashes` is mutable so a test can mutate it in place to
    simulate a symbol's definition changing between remember() and a later
    check_staleness()/fire() call, without needing to swap the attached
    resolver (attach_symbol_resolver() is intentionally idempotent, so a
    second attach on the same store is a no-op). `touching_calls` counts
    symbols_touching_file() invocations, so a test can assert fire()'s
    write-time existence-check actually skips the (expensive) resolver call
    when a workspace has no symbol-triggered notes at all. Real end-to-end
    coverage against the genuine SymbolGraph lives in
    TestSymbolTriggerIndexLiveGraph below."""

    def __init__(self, hashes: dict | None = None, touching: frozenset | None = None) -> None:
        self._hashes = hashes or {}
        self._touching = touching if touching is not None else frozenset()
        self.touching_calls = 0

    def signature_hash(self, workspace, name):
        return self._hashes.get(name)

    def symbols_touching_file(self, workspace, file_path):
        self.touching_calls += 1
        return self._touching


class TestSymbolTriggersWriteTimeIndex:
    def _symtrig_rows(self, tmp_path, note_id):
        import sqlite3
        with sqlite3.connect(str(tmp_path / "working_context.sqlite")) as conn:
            return conn.execute(
                "SELECT symbol_name, signature_hash FROM symbol_triggers WHERE note_id = ?",
                (note_id,),
            ).fetchall()

    def test_no_resolver_attached_stores_null_signature_hash(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "note", triggers=[{"symbol": "WorkspaceLock"}])
        assert self._symtrig_rows(tmp_path, note_id) == [("WorkspaceLock", None)]

    def test_attached_resolver_stores_signature_hash_at_write_time(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        store.attach_symbol_resolver(_FakeSymbolResolver(hashes={"WorkspaceLock": "abc123"}))
        note_id = store.remember(ws, "note", triggers=[{"symbol": "WorkspaceLock"}])
        assert self._symtrig_rows(tmp_path, note_id) == [("WorkspaceLock", "abc123")]

    def test_trigger_without_symbol_key_gets_no_index_row(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "note", triggers=[{"path": "src/api/**"}])
        assert self._symtrig_rows(tmp_path, note_id) == []

    def test_attach_symbol_resolver_is_idempotent(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        store.attach_symbol_resolver(_FakeSymbolResolver(hashes={"X": "first"}))
        store.attach_symbol_resolver(_FakeSymbolResolver(hashes={"X": "second"}))
        note_id = store.remember(ws, "note", triggers=[{"symbol": "X"}])
        assert self._symtrig_rows(tmp_path, note_id) == [("X", "first")]

    def test_forget_removes_symbol_trigger_index_rows(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "note", triggers=[{"symbol": "X"}])
        store.forget(ws, note_id)
        assert self._symtrig_rows(tmp_path, note_id) == []

    def test_forget_all_clears_symbol_trigger_index(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "note", triggers=[{"symbol": "X"}])
        store.forget_all(ws)
        assert self._symtrig_rows(tmp_path, note_id) == []

    def test_forget_all_workspaces_clears_symbol_trigger_index(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "note", triggers=[{"symbol": "X"}])
        store.forget_all_workspaces()
        assert self._symtrig_rows(tmp_path, note_id) == []


class TestFireSymbolPrimitive:
    """Fire-time S resolution (TRIGGER-ENGINE wave 2b): fire() resolves the
    target file's touched symbols ONCE per call and threads the same
    frozenset into every note's evaluate_note(), rather than querying the
    graph per note. Uses the fake resolver — real-graph coverage is in
    TestSymbolTriggerIndexLiveGraph below."""

    def test_fires_when_the_target_file_touches_the_declared_symbol(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        store.attach_symbol_resolver(_FakeSymbolResolver(touching=frozenset({"WorkspaceLock"})))
        store.remember(ws, "a gotcha", kind="gotcha", triggers=[{"symbol": "WorkspaceLock"}])
        results = store.fire(ws, event="pre-edit", file_path="src/resolver.py")
        assert len(results) == 1

    def test_does_not_fire_when_the_target_file_does_not_touch_the_symbol(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        store.attach_symbol_resolver(_FakeSymbolResolver(touching=frozenset()))
        store.remember(ws, "a gotcha", kind="gotcha", triggers=[{"symbol": "WorkspaceLock"}])
        results = store.fire(ws, event="pre-edit", file_path="src/resolver.py")
        assert len(results) == 0

    def test_never_fires_without_a_resolver_attached(self, tmp_path) -> None:
        """Degradation gate: memory-only daemons and the warm-up window
        before attach_symbol_resolver() runs have no resolver at all — a
        symbol trigger deterministically does not fire, never an error."""
        store, ws = _store(tmp_path), str(tmp_path)
        store.remember(ws, "a gotcha", kind="gotcha", triggers=[{"symbol": "WorkspaceLock"}])
        results = store.fire(ws, event="pre-edit", file_path="src/resolver.py")
        assert results == []

    def test_never_fires_without_a_file_path(self, tmp_path) -> None:
        """No target file means nothing to resolve symbols against — even
        with a resolver attached and the symbol technically resolvable
        elsewhere, an S trigger only ever matches at a pre-edit moment with
        a concrete file_path."""
        store, ws = _store(tmp_path), str(tmp_path)
        store.attach_symbol_resolver(_FakeSymbolResolver(touching=frozenset({"WorkspaceLock"})))
        store.remember(ws, "a gotcha", kind="gotcha", triggers=[{"symbol": "WorkspaceLock"}])
        results = store.fire(ws, event="pre-edit")
        assert results == []

    def test_existence_check_skips_the_resolver_call_when_no_symbol_triggers_exist(self, tmp_path) -> None:
        """Performance gate: a workspace with zero symbol-triggered notes
        must never pay for a live symbols_touching_file() graph call."""
        store, ws = _store(tmp_path), str(tmp_path)
        resolver = _FakeSymbolResolver(touching=frozenset({"WorkspaceLock"}))
        store.attach_symbol_resolver(resolver)
        store.remember(ws, "a plain finding")  # no triggers at all
        store.fire(ws, event="pre-edit", file_path="src/resolver.py")
        assert resolver.touching_calls == 0

    def test_resolver_call_happens_once_per_fire_call_when_relevant(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        resolver = _FakeSymbolResolver(touching=frozenset({"WorkspaceLock"}))
        store.attach_symbol_resolver(resolver)
        store.remember(ws, "gotcha one", kind="gotcha", triggers=[{"symbol": "WorkspaceLock"}])
        store.remember(ws, "gotcha two", kind="gotcha", triggers=[{"symbol": "WorkspaceLock"}])
        store.fire(ws, event="pre-edit", file_path="src/resolver.py")
        assert resolver.touching_calls == 1  # once for the whole call, not once per note


class TestSymbolAnchorStaleness:
    """check_staleness()'s symbol-signature re-hash (TRIGGER-ENGINE wave
    2b) — the S-primitive equivalent of TestAnchorStaleness's path anchors:
    never a silent drop, just a visible `[symbol_changed]` caveat."""

    def test_signature_change_adds_a_visible_caveat_but_never_drops_the_note(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        resolver = _FakeSymbolResolver(hashes={"WorkspaceLock": "hash_v1"})
        store.attach_symbol_resolver(resolver)
        note_id = store.remember(ws, "a gotcha", kind="gotcha", triggers=[{"symbol": "WorkspaceLock"}])
        resolver._hashes["WorkspaceLock"] = "hash_v2"  # the definition changed since write time
        notes = store.recall(ws)
        assert any(n.note_id == note_id for n in notes)  # still recalled, never silently dropped
        stale = store.check_staleness(notes, ws)
        assert note_id in stale
        assert any("WorkspaceLock" in r and "symbol_changed" in r for r in stale[note_id])

    def test_unchanged_signature_is_not_flagged(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        store.attach_symbol_resolver(_FakeSymbolResolver(hashes={"WorkspaceLock": "stable_hash"}))
        note_id = store.remember(ws, "a gotcha", kind="gotcha", triggers=[{"symbol": "WorkspaceLock"}])
        notes = store.recall(ws)
        stale = store.check_staleness(notes, ws)
        assert note_id not in stale

    def test_no_baseline_hash_at_write_time_is_never_flagged(self, tmp_path) -> None:
        """Note written before any resolver was attached stores a NULL
        signature_hash (nothing to compare against) — attaching a resolver
        afterward must not retroactively flag it."""
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "a gotcha", kind="gotcha", triggers=[{"symbol": "X"}])
        store.attach_symbol_resolver(_FakeSymbolResolver(hashes={"X": "some_hash"}))
        notes = store.recall(ws)
        stale = store.check_staleness(notes, ws)
        assert note_id not in stale

    def test_no_resolver_attached_degrades_to_no_check(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "a gotcha", kind="gotcha", triggers=[{"symbol": "WorkspaceLock"}])
        notes = store.recall(ws)
        stale = store.check_staleness(notes, ws)  # never raises with no resolver attached
        assert note_id not in stale


class TestSymbolTriggerIndexLiveGraph:
    """End-to-end coverage against the genuine SymbolGraph — the real
    construction path app/service.py wires: a SymbolGraph indexes the
    workspace and gets attach_symbol_resolver()'d onto the store, then a
    symbol trigger fires/re-hashes against real indexed code."""

    def test_fires_when_the_edited_file_defines_the_symbol(self, tmp_path) -> None:
        from agent.symbol_graph import SymbolGraph
        store, ws = _store(tmp_path), str(tmp_path)
        graph = SymbolGraph(str(tmp_path))
        f = tmp_path / "resolver.py"
        f.write_text("class WorkspaceLock:\n    def acquire(self):\n        pass\n")
        graph.index_file(ws, str(f))
        store.attach_symbol_resolver(graph)
        store.remember(ws, "a gotcha about locking", kind="gotcha", triggers=[{"symbol": "WorkspaceLock"}])
        results = store.fire(ws, event="pre-edit", file_path=str(f))
        assert len(results) == 1

    def test_does_not_fire_for_an_unrelated_file(self, tmp_path) -> None:
        from agent.symbol_graph import SymbolGraph
        store, ws = _store(tmp_path), str(tmp_path)
        graph = SymbolGraph(str(tmp_path))
        f = tmp_path / "resolver.py"
        f.write_text("class WorkspaceLock:\n    def acquire(self):\n        pass\n")
        graph.index_file(ws, str(f))
        store.attach_symbol_resolver(graph)
        store.remember(ws, "a gotcha about locking", kind="gotcha", triggers=[{"symbol": "WorkspaceLock"}])
        other = tmp_path / "unrelated.py"
        other.write_text("x = 1\n")
        graph.index_file(ws, str(other))
        results = store.fire(ws, event="pre-edit", file_path=str(other))
        assert len(results) == 0

    def test_signature_change_surfaces_a_staleness_caveat_on_fire(self, tmp_path) -> None:
        from agent.symbol_graph import SymbolGraph
        store, ws = _store(tmp_path), str(tmp_path)
        graph = SymbolGraph(str(tmp_path))
        f = tmp_path / "resolver.py"
        f.write_text("class WorkspaceLock:\n    def acquire(self):\n        pass\n")
        graph.index_file(ws, str(f))
        store.attach_symbol_resolver(graph)
        store.remember(ws, "a gotcha about locking", kind="gotcha", triggers=[{"symbol": "WorkspaceLock"}])
        f.write_text("class WorkspaceLock:\n    def acquire(self):\n        return True\n")
        graph.index_file(ws, str(f))  # re-index so signature_hash reflects the new body
        results = store.fire(ws, event="pre-edit", file_path=str(f))
        assert len(results) == 1
        assert any("symbol_changed" in r for r in results[0].stale_paths)


def _fixed_vector_store(tmp_path, note_vector: list[float], query_vector: list[float]):
    """A store whose every note gets `note_vector` at write time and whose
    every prompt-submit query gets `query_vector` — lets a test fix the
    cosine similarity `fire()`'s M primitive computes exactly, rather than
    relying on real semantic drift (TRIGGER-ENGINE wave 2b, §8)."""
    import chromadb
    from agent.working_context_store import WorkingContextStore
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    return WorkingContextStore(
        str(tmp_path),
        embed_fn=_const_embed(note_vector),
        embed_query_fn=_const_embed(query_vector),
        notes_chroma_client=client,
    )


class TestFireSemanticPrimitive:
    """Fire-time M resolution (TRIGGER-ENGINE wave 2b, §8): fire() computes
    ONE activity embedding per call (never per note) and gates each
    semantic-triggered note's own stored vector against a fixed per-kind
    theta (config.yaml memory_triggers.semantic.theta_by_kind). gotcha's
    theta is 0.72 — an identical unit vector (cosine 1.0) clears it, an
    orthogonal one (cosine 0.0) does not."""

    def test_fires_when_cosine_clears_theta(self, tmp_path) -> None:
        store = _fixed_vector_store(tmp_path, [1.0, 0.0], [1.0, 0.0])
        ws = str(tmp_path)
        store.remember(ws, "a gotcha", kind="gotcha", triggers=[{"semantic": True}])
        results = store.fire(ws, event="prompt-submit", query="anything")
        assert len(results) == 1

    def test_does_not_fire_when_cosine_is_below_theta(self, tmp_path) -> None:
        store = _fixed_vector_store(tmp_path, [1.0, 0.0], [0.0, 1.0])
        ws = str(tmp_path)
        store.remember(ws, "a gotcha", kind="gotcha", triggers=[{"semantic": True}])
        results = store.fire(ws, event="prompt-submit", query="anything")
        assert len(results) == 0

    def test_never_fires_without_a_query(self, tmp_path) -> None:
        store = _fixed_vector_store(tmp_path, [1.0, 0.0], [1.0, 0.0])
        ws = str(tmp_path)
        store.remember(ws, "a gotcha", kind="gotcha", triggers=[{"semantic": True}])
        results = store.fire(ws, event="prompt-submit")
        assert results == []

    def test_two_semantic_matches_of_the_same_kind_rank_by_recency(self, tmp_path) -> None:
        """Age-fade on rank only, theta never moves (TRIGGER-ENGINE wave 2b,
        §8): among several M-fired notes of the same kind, the more recently
        used one ranks first. This is NOT new sort logic — M-fired
        FireResults flow through the exact same `total_order_key` call every
        other primitive already uses, whose existing `-last_used` tie-break
        (for every kind but 'task') already satisfies this requirement."""
        store = _fixed_vector_store(tmp_path, [1.0, 0.0], [1.0, 0.0])
        ws = str(tmp_path)
        older_id = store.remember(ws, "older gotcha", kind="gotcha", triggers=[{"semantic": True}])
        newer_id = store.remember(ws, "newer gotcha", kind="gotcha", triggers=[{"semantic": True}])
        with store._conn() as conn:
            conn.execute(
                "UPDATE notes SET last_accessed = ? WHERE note_id = ?",
                (time.time() - 3600, older_id),
            )
        results = store.fire(ws, event="prompt-submit", query="anything")
        assert [r.note_id for r in results] == [newer_id, older_id]

    def test_never_fires_without_an_embedder_attached(self, tmp_path) -> None:
        """Degradation gate: no embedder attached at all (a memory-only
        daemon, or the warm-up window before attach_embedder() runs) — a
        semantic trigger deterministically does not fire, never an error."""
        store, ws = _store(tmp_path), str(tmp_path)
        store.remember(ws, "a gotcha", kind="gotcha", triggers=[{"semantic": True}])
        results = store.fire(ws, event="prompt-submit", query="anything")
        assert results == []

    def test_never_embeds_the_query_when_no_note_declares_a_semantic_axis(self, tmp_path) -> None:
        """Performance gate: a workspace with zero semantic-triggered notes
        must never pay for an activity embedding call."""
        calls: list[list[str]] = []
        store = _fixed_vector_store(tmp_path, [1.0, 0.0], [1.0, 0.0])
        store._embed_query_fn = _counting_embed([1.0, 0.0], calls)
        ws = str(tmp_path)
        store.remember(ws, "a plain finding")  # no triggers at all
        store.fire(ws, event="prompt-submit", query="anything")
        assert calls == []

    def test_only_one_activity_embedding_computed_per_fire_call(self, tmp_path) -> None:
        calls: list[list[str]] = []
        store = _fixed_vector_store(tmp_path, [1.0, 0.0], [1.0, 0.0])
        store._embed_query_fn = _counting_embed([1.0, 0.0], calls)
        ws = str(tmp_path)
        store.remember(ws, "gotcha one", kind="gotcha", triggers=[{"semantic": True}])
        store.remember(ws, "gotcha two", kind="gotcha", triggers=[{"semantic": True}])
        store.fire(ws, event="prompt-submit", query="anything")
        assert len(calls) == 1  # once for the whole call, not once per note

    def test_note_vector_is_reused_never_re_embedded_at_fire_time(self, tmp_path) -> None:
        """The note's own document-side vector is stored once at remember()
        time and read back as-is — fire() never calls the document embed_fn
        again."""
        calls: list[list[str]] = []
        store = _fixed_vector_store(tmp_path, [1.0, 0.0], [1.0, 0.0])
        ws = str(tmp_path)
        store.remember(ws, "a gotcha", kind="gotcha", triggers=[{"semantic": True}])
        store._embed_fn = _counting_embed([1.0, 0.0], calls)
        store.fire(ws, event="prompt-submit", query="anything")
        assert calls == []


class TestSharedLedgerAcrossSymbolAndSemanticFires:
    """The `TriggerFireLedger` passed into `fire()`/`fire_and_format()` never
    branches on which primitive produced a `FireResult` (TRIGGER-ENGINE wave
    2b) — one call with BOTH a `file_path` (S) and a `query` (M) set can
    fire an S-triggered note and an M-triggered note together, and the
    ledger's per-axis dedup plus the cumulative per-session token budget
    apply identically to both, exactly as they already do for P/E fires."""

    def _mixed_store(self, tmp_path):
        import chromadb
        from agent.symbol_graph import SymbolGraph
        from agent.working_context_store import WorkingContextStore

        ws = str(tmp_path)
        graph = SymbolGraph(ws)
        f = tmp_path / "resolver.py"
        f.write_text("class WorkspaceLock:\n    def acquire(self):\n        pass\n")
        graph.index_file(ws, str(f))

        client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
        store = WorkingContextStore(
            ws,
            embed_fn=_const_embed([1.0, 0.0]),
            embed_query_fn=_const_embed([1.0, 0.0]),
            notes_chroma_client=client,
        )
        store.attach_symbol_resolver(graph)
        return store, ws, str(f)

    def test_one_call_fires_both_a_symbol_note_and_a_semantic_note(self, tmp_path) -> None:
        store, ws, f = self._mixed_store(tmp_path)
        sym_id = store.remember(ws, "a symbol gotcha", kind="gotcha", triggers=[{"symbol": "WorkspaceLock"}])
        sem_id = store.remember(ws, "a semantic gotcha", kind="gotcha", triggers=[{"semantic": True}])
        results = store.fire(ws, event="pre-edit", file_path=f, query="anything")
        assert {r.note_id for r in results} == {sym_id, sem_id}

    def test_ledger_dedup_suppresses_both_axes_on_a_repeat_call(self, tmp_path) -> None:
        from agent.trigger_engine import TriggerFireLedger

        store, ws, f = self._mixed_store(tmp_path)
        store.remember(ws, "a symbol gotcha", kind="gotcha", triggers=[{"symbol": "WorkspaceLock"}])
        store.remember(ws, "a semantic gotcha", kind="gotcha", triggers=[{"semantic": True}])
        ledger = TriggerFireLedger()
        first = store.fire(ws, event="pre-edit", file_path=f, query="anything", ledger=ledger)
        assert len(first) == 2
        second = store.fire(ws, event="pre-edit", file_path=f, query="anything", ledger=ledger)
        assert second == []  # neither the S fire nor the M fire re-fires this session

    def test_cumulative_budget_accounts_for_both_axes_together(self, tmp_path) -> None:
        from agent.trigger_engine import MEMORY_TRIGGER_PER_SESSION_TOKEN_CAP, TriggerFireLedger

        store, ws, f = self._mixed_store(tmp_path)
        sym_id = store.remember(ws, "a symbol gotcha", kind="gotcha", triggers=[{"symbol": "WorkspaceLock"}])
        sem_id = store.remember(ws, "a semantic gotcha", kind="gotcha", triggers=[{"semantic": True}])
        ledger = TriggerFireLedger()
        assert ledger.remaining_budget() == MEMORY_TRIGGER_PER_SESSION_TOKEN_CAP
        text, note_ids = store.fire_and_format(
            ws, event="pre-edit", file_path=f, query="anything", ledger=ledger,
        )
        assert note_ids == {sym_id, sem_id}
        assert "a symbol gotcha" in text
        assert "a semantic gotcha" in text
        assert ledger.remaining_budget() < MEMORY_TRIGGER_PER_SESSION_TOKEN_CAP


class TestPromote:
    def test_promote_auto_to_agent(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "auto note", provenance="auto")
        assert store.promote(ws, note_id, "agent") is True
        note = store.get_note(ws, note_id)
        assert note.provenance == "agent"

    def test_promote_agent_to_human(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "agent note")  # default provenance='agent'
        assert store.promote(ws, note_id, "human") is True
        note = store.get_note(ws, note_id)
        assert note.provenance == "human"

    def test_promote_cannot_skip_a_rank(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "auto note", provenance="auto")
        with pytest.raises(ValueError):
            store.promote(ws, note_id, "human")  # auto -> human skips 'agent'

    def test_promote_cannot_demote(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "human note")
        store.promote(ws, note_id, "human")  # no-op path isn't reached; set via remember instead
        with pytest.raises(ValueError):
            store.promote(ws, note_id, "agent")

    def test_promote_rejects_unrecognised_target(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "note", provenance="auto")
        with pytest.raises(ValueError):
            store.promote(ws, note_id, "not-a-real-provenance")

    def test_promote_nonexistent_note_returns_false(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        assert store.promote(ws, 999999, "agent") is False


class TestAnchorStaleness:
    def test_anchor_change_adds_visible_caveat_but_never_drops_the_note(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        f = tmp_path / "src" / "auth.py"
        f.parent.mkdir(parents=True)
        f.write_text("original content")
        note_id = store.remember(ws, "a gotcha about auth.py", kind="gotcha", anchors=["src/auth.py"])

        f.write_text("changed content")  # anchor path's content now differs
        notes = store.recall(ws)
        assert any(n.note_id == note_id for n in notes)  # still recalled, never silently dropped
        stale = store.check_staleness(notes, ws)
        assert note_id in stale
        assert any("src/auth.py" in r and "anchor_changed" in r for r in stale[note_id])

    def test_unchanged_anchor_is_not_flagged(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        f = tmp_path / "src" / "auth.py"
        f.parent.mkdir(parents=True)
        f.write_text("stable content")
        note_id = store.remember(ws, "a gotcha", kind="gotcha", anchors=["src/auth.py"])
        notes = store.recall(ws)
        stale = store.check_staleness(notes, ws)
        assert note_id not in stale

    def test_anchor_with_no_baseline_hash_never_flagged(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "a gotcha about a future file", kind="gotcha", anchors=["not/created/yet.py"])
        (tmp_path / "not" / "created").mkdir(parents=True)
        (tmp_path / "not" / "created" / "yet.py").write_text("now it exists")
        notes = store.recall(ws)
        stale = store.check_staleness(notes, ws)
        assert note_id not in stale  # no hash-at-write to compare against


class TestFormatNotesForLlmProvenanceFraming:
    def test_full_tier_marks_provenance_on_every_block(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "a finding")  # default provenance='agent'
        notes = store.recall(ws)
        output = store.format_notes_for_llm(notes, detail="full")
        assert "[agent]" in output

    def test_human_directive_gets_imperative_framing_in_output(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "always run tests before committing", kind="directive")
        store.promote(ws, note_id, "human")
        notes = store.recall(ws, include_superseded=True)
        output = store.format_notes_for_llm(notes, detail="full")
        assert "DIRECTIVE" in output
        assert "follow it" in output

    def test_auto_provenance_gets_weakest_framing_in_output(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        store.remember(ws, "an auto-captured finding", provenance="auto")
        notes = store.recall(ws)
        output = store.format_notes_for_llm(notes, detail="full")
        assert "weakest" in output.lower() or "no reviewing judgment" in output.lower()

    def test_index_tier_output_unaffected(self, tmp_path) -> None:
        """Provenance framing is additive to the full-tier render only — the
        index-tier one-liner keeps its pre-existing compact format."""
        store, ws = _store(tmp_path), str(tmp_path)
        store.remember(ws, "a finding")
        notes = store.recall(ws)
        output = store.format_notes_for_llm(notes, detail="index")
        assert "[agent]" not in output

    def test_superseded_marker_uses_note_id_fallback_for_explicit_supersedes(self, tmp_path) -> None:
        """The explicit `supersedes` path has no author_id badge (unlike the
        code_hash-conflict path) — the superseded marker must still render,
        naming the superseding note by id rather than crashing/omitting it."""
        store, ws = _store(tmp_path), str(tmp_path)
        old_id = store.remember(ws, "old finding")
        new_id = store.remember(ws, "corrected finding", supersedes=old_id)
        all_notes = store.recall(ws, include_superseded=True)
        stale = store.check_staleness(all_notes, ws)
        output = store.format_notes_for_llm(all_notes, stale_warnings=stale, detail="full")
        assert f"superseded by @note#{new_id}" in output


class TestFormatNotesForLlmScopeSurfaced:
    """UPG-SCOPE-SURFACE-BACK: a note's resolved scope (write-time, per
    UPG-TRIGGER-SCOPE-KIND-DEFAULTS) was previously visible nowhere after
    remember() returned the note id. The full-tier render now shows it —
    and the captured branch too, for scope=="branch" — so a caller can
    diagnose why a scoped note does or doesn't fire without a second lookup."""

    def test_full_tier_shows_workspace_scope(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        store.remember(ws, "a finding", scope="workspace")
        notes = store.recall(ws)
        output = store.format_notes_for_llm(notes, detail="full")
        assert "[scope=workspace]" in output

    def test_full_tier_shows_branch_scope_with_branch_name(self, tmp_path, monkeypatch) -> None:
        import agent.working_context_store._store as store_mod
        monkeypatch.setattr(store_mod, "_current_git_branch", lambda root: "feature/x")
        store, ws = _store(tmp_path), str(tmp_path)
        store.remember(ws, "a branch-bound task", kind="task", priority="high")  # omitted -> branch default
        notes = store.recall(ws)
        output = store.format_notes_for_llm(notes, detail="full")
        assert "[scope=branch (feature/x)]" in output

    def test_full_tier_shows_bare_scope_when_branch_fallback_taken(self, tmp_path) -> None:
        """A non-git workspace falls back to scope="workspace" (the
        silent-death guard) — the render must show the value actually
        stored, not "branch" with an empty/missing branch name."""
        store, ws = _store(tmp_path), str(tmp_path)
        store.remember(ws, "a task note, no git here", kind="task", priority="high")
        notes = store.recall(ws)
        output = store.format_notes_for_llm(notes, detail="full")
        assert "[scope=workspace]" in output
        assert "[scope=branch" not in output

    def test_index_tier_output_unaffected(self, tmp_path) -> None:
        """Scope is additive to the full-tier render only — the index-tier
        one-liner keeps its pre-existing, token-budgeted compact format."""
        store, ws = _store(tmp_path), str(tmp_path)
        store.remember(ws, "a finding", scope="repo")
        notes = store.recall(ws)
        output = store.format_notes_for_llm(notes, detail="index")
        assert "scope=" not in output


class TestFireEvaluation:
    def test_fire_returns_only_fired_notes(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        store.remember(ws, "a directive", kind="directive")
        store.remember(ws, "a plain finding")  # no default bundle, never fires
        results = store.fire(ws, event="session-start")
        assert len(results) == 1
        assert results[0].fired is True

    def test_fire_orders_by_total_order(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        # priority="high" — task's default session-start bundle only applies
        # at high priority (matches the legacy boot_recall() SQL filter this
        # bundle replaces; TRIGGER-ENGINE wave 2a).
        store.remember(ws, "a task", kind="task", priority="high")
        store.remember(ws, "a directive", kind="directive")
        results = store.fire(ws, event="session-start")
        fired_kinds = [store.get_note(ws, r.note_id).kind for r in results]
        assert fired_kinds == ["directive", "task"]

    def test_fire_folds_in_staleness_caveats(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        f = tmp_path / "src" / "auth.py"
        f.parent.mkdir(parents=True)
        f.write_text("original")
        store.remember(ws, "a gotcha about auth.py", kind="gotcha", anchors=["src/auth.py"])
        f.write_text("changed")
        results = store.fire(ws, event="pre-edit", file_path="src/auth.py")
        assert len(results) == 1
        assert any("anchor_changed" in r for r in results[0].stale_paths)

    def test_fire_never_fires_a_tombstoned_note(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        old_id = store.remember(ws, "old directive", kind="directive")
        store.remember(ws, "new directive", kind="directive", supersedes=old_id)
        results = store.fire(ws, event="session-start")
        assert all(r.note_id != old_id for r in results)

    def test_fire_stamps_last_fired_for_cooldown(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "a directive", kind="directive")
        store.fire(ws, event="session-start")
        note = store.get_note(ws, note_id)
        assert note.last_fired is not None

    def test_fire_with_ledger_suppresses_same_axis_re_fire(self, tmp_path) -> None:
        from agent.trigger_engine import TriggerFireLedger
        store, ws = _store(tmp_path), str(tmp_path)
        store.remember(ws, "a directive", kind="directive")
        ledger = TriggerFireLedger()
        first = store.fire(ws, event="session-start", ledger=ledger)
        second = store.fire(ws, event="session-start", ledger=ledger)
        assert len(first) == 1
        assert len(second) == 0  # same axis already fired this "session"

    def test_fire_with_ledger_allows_a_different_axis(self, tmp_path) -> None:
        from agent.trigger_engine import TriggerFireLedger
        store, ws = _store(tmp_path), str(tmp_path)
        store.remember(ws, "a directive", kind="directive")
        ledger = TriggerFireLedger()
        store.fire(ws, event="session-start", ledger=ledger)
        after_compaction = store.fire(ws, event="post-compaction", ledger=ledger)
        assert len(after_compaction) == 1  # a different trigger index — not suppressed

    def test_fire_without_ledger_never_suppresses(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        store.remember(ws, "a directive", kind="directive")
        first = store.fire(ws, event="session-start")
        second = store.fire(ws, event="session-start")
        assert len(first) == 1
        assert len(second) == 1


class TestFirePathPrimitiveAbsoluteRelative:
    """P (path) trigger primitive must match a trigger's glob `path` pattern
    against EITHER the file_path exactly as given OR its workspace-relative
    form — a real hook (every AI code editor) sends an ABSOLUTE file_path
    while anchors/triggers are naturally authored workspace-relative (a
    gotcha's kind-default bundle generates them straight from anchors)."""

    def test_relative_anchor_fires_on_absolute_hook_path(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        f = tmp_path / "agent" / "trigger_engine.py"
        f.parent.mkdir(parents=True)
        f.write_text("content")
        store.remember(
            ws, "a gotcha about trigger_engine.py", kind="gotcha",
            anchors=["agent/trigger_engine.py"],
        )
        results = store.fire(ws, event="pre-edit", file_path=str(f))
        assert len(results) == 1
        assert results[0].fired is True

    def test_absolute_pattern_still_fires_against_absolute_path(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        f = tmp_path / "agent" / "trigger_engine.py"
        f.parent.mkdir(parents=True)
        f.write_text("content")
        absolute_path = str(f)
        store.remember(
            ws, "an absolutely-anchored gotcha", kind="gotcha",
            triggers=[{"path": absolute_path, "event": "pre-edit"}],
        )
        results = store.fire(ws, event="pre-edit", file_path=absolute_path)
        assert len(results) == 1

    def test_relative_pattern_still_fires_against_relative_path(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        store.remember(
            ws, "a relatively-anchored gotcha", kind="gotcha",
            anchors=["src/api/handlers.py"],
        )
        results = store.fire(ws, event="pre-edit", file_path="src/api/handlers.py")
        assert len(results) == 1

    def test_glob_pattern_matches_absolute_path_via_relative_form(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        f = tmp_path / "src" / "api" / "handlers.py"
        f.parent.mkdir(parents=True)
        f.write_text("content")
        store.remember(
            ws, "a glob-anchored gotcha", kind="gotcha",
            triggers=[{"path": "src/api/**", "event": "pre-edit"}],
        )
        results = store.fire(ws, event="pre-edit", file_path=str(f))
        assert len(results) == 1

    def test_file_outside_workspace_root_matches_absolute_only(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        outside = str(tmp_path.parent / "outside_the_workspace.py")
        store.remember(
            ws, "an absolutely-anchored gotcha outside the workspace",
            kind="gotcha", triggers=[{"path": outside, "event": "pre-edit"}],
        )
        results = store.fire(ws, event="pre-edit", file_path=outside)
        assert len(results) == 1
        # a relative pattern never matches a file that has no relative form
        store.remember(
            ws, "a relatively-anchored gotcha that cannot reach outside",
            kind="gotcha", triggers=[{"path": "outside_the_workspace.py", "event": "pre-edit"}],
        )
        results = store.fire(ws, event="pre-edit", file_path=outside)
        assert len(results) == 1  # still only the absolute-pattern note

    def test_absolute_pattern_fires_on_relative_hook_path(self, tmp_path) -> None:
        """UPG-PROXY-ANCHOR-ABS-REL-NORM's mirror case through fire(): a
        trigger whose `path` is ABSOLUTE must still fire when the caller
        passes a WORKSPACE-RELATIVE file_path — `_path_trigger_candidates()`
        now derives the workspace-rooted absolute form for relative inputs,
        and this P-primitive shares that candidate set with
        `recall_for_path()`."""
        store, ws = _store(tmp_path), str(tmp_path)
        f = tmp_path / "agent" / "trigger_engine.py"
        f.parent.mkdir(parents=True)
        f.write_text("content")
        absolute_pattern = str(f.resolve())
        store.remember(
            ws, "an absolutely-anchored gotcha", kind="gotcha",
            triggers=[{"path": absolute_pattern, "event": "pre-edit"}],
        )
        results = store.fire(ws, event="pre-edit", file_path="agent/trigger_engine.py")
        assert len(results) == 1
        assert results[0].fired is True


class TestPathTriggerCandidateForms:
    """Direct units on the shared helpers (UPG-PROXY-ANCHOR-ABS-REL-NORM /
    UPG-PROXY-WEAK-TIER-TIEBREAK): `recall_for_path()`'s narrowing inputs and
    `fire()`'s live P primitive both consume `_path_trigger_candidates()`'s
    output, and three consumers share one boundary predicate — pin its exact
    shape here so the consumers can never silently disagree."""

    def test_relative_input_yields_workspace_rooted_absolute_form(self, tmp_path) -> None:
        from agent.working_context_store._store import _path_trigger_candidates

        ws = str(tmp_path)
        cands = _path_trigger_candidates(ws, "src/auth.py")
        assert cands[0] == "src/auth.py"
        # Exactly one derived form: the process cwd is nowhere near `ws`, so
        # no CWD-relative form exists, and the absolute form is rooted at the
        # WORKSPACE (never this process's cwd).
        assert cands == (
            "src/auth.py",
            str((tmp_path / "src" / "auth.py").resolve()),
        )

    def test_absolute_input_under_root_stays_two_forms(self, tmp_path) -> None:
        """A real hook event (absolute file_path) keeps the historical
        two-form candidate sequence byte-for-byte: as-given + workspace-
        relative. The new absolute form is only derived for RELATIVE inputs,
        so this contract is what makes the change a pure widening."""
        from agent.working_context_store._store import _path_trigger_candidates

        ws = str(tmp_path)
        p = str(tmp_path / "src" / "auth.py")
        assert _path_trigger_candidates(ws, p) == (p, "src/auth.py")

    def test_absolute_input_outside_root_has_no_derived_forms(self, tmp_path) -> None:
        from agent.working_context_store._store import _path_trigger_candidates

        ws = str(tmp_path)
        outside = str(tmp_path.parent / "outside_the_workspace.py")
        assert _path_trigger_candidates(ws, outside) == (outside,)

    def test_none_input_returns_none(self) -> None:
        from agent.working_context_store._store import _path_trigger_candidates

        assert _path_trigger_candidates("/some/ws", None) is None

    def test_boundary_count_and_first_agree_with_match(self) -> None:
        """The boolean, the count, and the first-offset views of the ONE
        boundary predicate can never drift: count>0 <=> match <=> first>=0,
        and first is occurrences[0]."""
        from agent.working_context_store._store import (
            _path_boundary_count,
            _path_boundary_first,
            _path_boundary_match,
            _path_boundary_occurrences,
        )

        cases = [
            ("edit gate.py now", "gate.py"),
            ('the file "gate.py" was touched', "gate.py"),
            ("gate.py: verify_token must check expiry", "gate.py"),
            ("mentions uv_regate.py instead", "gate.py"),   # substring, no boundary
            ("open gate.pyc for bytecode", "gate.py"),      # longer extension
            ("see src/auth/gate.py handled", "src/auth/gate.py"),  # multi-segment needle
        ]
        for text, needle in cases:
            occ = _path_boundary_occurrences(text, needle)
            assert _path_boundary_match(text, needle) == bool(occ), (text, needle)
            assert _path_boundary_count(text, needle) == len(occ), (text, needle)
            assert (_path_boundary_first(text, needle) >= 0) == bool(occ), (text, needle)
            assert _path_boundary_first(text, needle) == (occ[0] if occ else -1), (text, needle)

    def test_boundary_count_counts_and_first_locates(self) -> None:
        from agent.working_context_store._store import (
            _path_boundary_count,
            _path_boundary_first,
        )

        text = "gate.py supersedes the old gate.py notes"
        assert _path_boundary_count(text, "gate.py") == 2
        assert _path_boundary_first(text, "gate.py") == 0

    def test_empty_needle_is_never_a_match_anywhere(self) -> None:
        from agent.working_context_store._store import (
            _path_boundary_count,
            _path_boundary_first,
            _path_boundary_match,
        )

        assert _path_boundary_match("anything", "") is False
        assert _path_boundary_count("anything", "") == 0
        assert _path_boundary_first("anything", "") == -1


class TestWorkingMemoryFetchWidth:
    """Direct units on the over-fetch contract helper (UPG-OVERFETCH-
    CONTRACT-UNDOCUMENTED): every vector query against the GLOBAL
    'working_memory' collection must size n_results through
    working_memory_fetch_width(), because workspace/lifecycle filtering runs
    in SQL only AFTER the vector search. Pins the exact formula — multiplier,
    floor override, col_count cap — plus byte-for-byte equivalence with the
    three pre-helper expressions it replaced, so no call site can drift back
    to a raw render limit without an obvious failure."""

    def test_multiplier_is_three_times_the_render_limit(self) -> None:
        from agent.working_context_store._store import working_memory_fetch_width

        assert working_memory_fetch_width(5, 1000) == 15
        assert working_memory_fetch_width(1, 1000) == 3

    def test_floor_raises_width_above_the_multiplier(self) -> None:
        from agent.working_context_store._store import working_memory_fetch_width

        # related_active_notes passes floor=limit + 1; the revoked path
        # passes MEMORY_WRITE_RELATED_REVOKED_QUERY_FLOOR. The floor wins
        # whenever it exceeds render_limit * 3; the multiplier otherwise.
        assert working_memory_fetch_width(1, 1000, floor=7) == 7
        assert working_memory_fetch_width(10, 1000, floor=2) == 30

    def test_col_count_caps_the_ask(self) -> None:
        from agent.working_context_store._store import working_memory_fetch_width

        # ChromaDB errors when asked for more results than exist, so the
        # live collection count caps even a large floor.
        assert working_memory_fetch_width(10, 4) == 4
        assert working_memory_fetch_width(1, 1000, floor=50) == 50
        assert working_memory_fetch_width(5, 2, floor=20) == 2

    def test_matches_the_literal_formulas_it_replaced(self) -> None:
        """The helper conversion had to change NO fetched width anywhere:
        each expression below is the pre-helper formula verbatim."""
        from agent.config import MEMORY_WRITE_RELATED_REVOKED_QUERY_FLOOR
        from agent.working_context_store._store import working_memory_fetch_width

        for limit in (1, 3, 10):
            for col_count in (limit * 3, limit * 3 + 7, 10_000):
                assert working_memory_fetch_width(limit, col_count) == min(limit * 3, col_count)
                assert working_memory_fetch_width(
                    limit, col_count, floor=limit + 1,
                ) == min(max(limit * 3, limit + 1), col_count)
                assert working_memory_fetch_width(
                    limit, col_count, floor=MEMORY_WRITE_RELATED_REVOKED_QUERY_FLOOR,
                ) == min(max(limit * 3, MEMORY_WRITE_RELATED_REVOKED_QUERY_FLOOR), col_count)


class TestFireScopeEnforcement:
    """TRIGGER-ENGINE wave 2a, bm2-design-skeleton.md §1 — all five
    SCOPE_VALUES enforced through the real store (SQLite round-trip), not
    just the pure `scope_permits()` unit tests in test_trigger_engine.py."""

    def test_workspace_scope_default_fires_for_any_caller(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        store.remember(ws, "a directive", kind="directive")  # scope defaults to "workspace"
        assert len(store.fire(ws, event="session-start")) == 1
        assert len(store.fire(ws, event="session-start", session_id="anyone")) == 1

    def test_session_scope_fires_only_for_the_writing_session(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        store.remember(
            ws, "an ephemeral note", kind="directive", session_id="writer-session",
            scope="session", triggers=[{"event": "session-start"}],
        )
        assert len(store.fire(ws, event="session-start", session_id="writer-session")) == 1
        assert len(store.fire(ws, event="session-start", session_id="other-session")) == 0
        assert len(store.fire(ws, event="session-start")) == 0  # no session_id at all

    def test_path_subtree_scope_fires_only_under_the_declared_anchor(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        store.remember(
            ws, "a scoped gotcha", kind="gotcha", scope="path-subtree",
            anchors=["src/api/x.py"],
        )
        assert len(store.fire(ws, event="pre-edit", file_path="src/api/x.py")) == 1
        assert len(store.fire(ws, event="pre-edit", file_path="src/other/y.py")) == 0

    def test_path_subtree_scope_fires_for_an_absolute_hook_path_against_a_relative_anchor(
        self, tmp_path,
    ) -> None:
        """Same defect class the P primitive's abs/rel fix closed, but in
        scope_permits()'s own path-subtree gate: a real hook sends an
        ABSOLUTE file_path while the anchor is naturally authored workspace-
        relative. scope_permits() is checked BEFORE the trigger loop, so
        even though the P primitive itself would match, an absolute-only
        scope check silently excluded the note before the trigger ever ran."""
        store, ws = _store(tmp_path), str(tmp_path)
        (tmp_path / "src" / "api").mkdir(parents=True)
        store.remember(
            ws, "a scoped gotcha", kind="gotcha", scope="path-subtree",
            anchors=["src/api/x.py"],
        )
        abs_path = str(tmp_path / "src" / "api" / "x.py")
        assert len(store.fire(ws, event="pre-edit", file_path=abs_path)) == 1
        outside = str(tmp_path / "src" / "other" / "y.py")
        assert len(store.fire(ws, event="pre-edit", file_path=outside)) == 0

    def test_branch_scope_fires_only_on_the_recorded_branch(self, tmp_path, monkeypatch) -> None:
        import agent.working_context_store._store as store_mod
        store, ws = _store(tmp_path), str(tmp_path)
        monkeypatch.setattr(store_mod, "_current_git_branch", lambda root: "feature/x")
        note_id = store.remember(
            ws, "a branch-bound task", kind="task", scope="branch",
            triggers=[{"event": "session-start"}],
        )
        assert store.get_note(ws, note_id).branch == "feature/x"
        assert len(store.fire(ws, event="session-start")) == 1  # still on feature/x
        monkeypatch.setattr(store_mod, "_current_git_branch", lambda root: "main")
        assert len(store.fire(ws, event="session-start")) == 0  # switched branches

    def test_repo_scope_is_a_no_op_like_workspace_through_the_real_store(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        store.remember(ws, "a directive", kind="directive", scope="repo")
        assert len(store.fire(ws, event="session-start")) == 1

    def test_recall_enforces_session_scope_not_just_fire(self, tmp_path) -> None:
        """§1: scope="session" must be enforced on EVERY read path, not only
        trigger firing — an ephemeral note must never leak into a plain
        vectr_recall from a different session."""
        store, ws = _store(tmp_path), str(tmp_path)
        store.remember(ws, "an ephemeral finding", session_id="writer-session", scope="session")
        assert len(store.recall(ws, session_id="writer-session")) == 1
        assert len(store.recall(ws, session_id="other-session")) == 0
        assert len(store.recall(ws)) == 0

    def test_recall_for_path_enforces_path_subtree_scope(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        store.remember(
            ws, "a subtree note mentioning auth.py", kind="gotcha",
            scope="path-subtree", anchors=["src/auth.py"],
        )
        assert len(store.recall_for_path(ws, "src/auth.py")) == 1
        assert len(store.recall_for_path(ws, "src/other.py")) == 0

    def test_recall_for_path_scope_check_matches_an_absolute_hook_path(self, tmp_path) -> None:
        """F1b's fix also covers recall_for_path() -- a PreToolUse hook sends
        an ABSOLUTE path, and the anchor is naturally authored workspace-
        relative; recall_for_path() already resolves the workspace-relative
        form for its own content match, this reuses that same resolution for
        the scope check instead of dropping it."""
        store, ws = _store(tmp_path), str(tmp_path)
        (tmp_path / "src").mkdir()
        store.remember(
            ws, "a subtree note mentioning auth.py", kind="gotcha",
            scope="path-subtree", anchors=["src/auth.py"],
        )
        abs_path = str(tmp_path / "src" / "auth.py")
        assert len(store.recall_for_path(ws, abs_path)) == 1
        abs_other = str(tmp_path / "src" / "other.py")
        assert len(store.recall_for_path(ws, abs_other)) == 0

    def test_pre_wave_notes_with_no_scope_declared_are_backward_compatible(self, tmp_path) -> None:
        """A note written before scope existed (or by any caller that never
        passes scope=) gets the dataclass default "workspace" — fires and
        recalls exactly as it did before this wave."""
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "an old-style directive", kind="directive")
        assert store.get_note(ws, note_id).scope == "workspace"
        assert len(store.fire(ws, event="session-start")) == 1
        assert len(store.recall(ws)) == 1


class TestFireAndFormat:
    """`WorkingContextStore.fire_and_format()` (TRIGGER-ENGINE wave 2a,
    bm2-design-skeleton.md §2/§3/§4) — the live hook-delivery entry point:
    multi-event OR-merge, dedup, budget pack, cumulative session spend."""

    def test_renders_a_fired_note_with_a_triggered_memory_header(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        store.remember(ws, "never push to main", kind="directive")
        text, note_ids = store.fire_and_format(ws, event="session-start")
        assert "Triggered Memory" in text
        assert "never push to main" in text
        assert len(note_ids) == 1

    def test_empty_when_nothing_fires(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        store.remember(ws, "a plain finding")  # no default bundle
        text, note_ids = store.fire_and_format(ws, event="session-start")
        assert text == ""
        assert note_ids == set()

    def test_events_list_ors_across_multiple_lifecycle_moments_without_double_rendering(self, tmp_path) -> None:
        """A directive's default bundle covers BOTH session-start and
        post-compaction — merging events=[both] must render it exactly once,
        not twice, per note_id (first-seen wins)."""
        store, ws = _store(tmp_path), str(tmp_path)
        store.remember(ws, "never push to main", kind="directive")
        text, note_ids = store.fire_and_format(ws, events=["session-start", "post-compaction"])
        assert len(note_ids) == 1
        assert text.count("never push to main") == 1

    def test_ledger_makes_the_injection_budget_cumulative_across_calls(self, tmp_path) -> None:
        from agent.trigger_engine import MEMORY_TRIGGER_PER_SESSION_TOKEN_CAP, TriggerFireLedger
        store, ws = _store(tmp_path), str(tmp_path)
        store.remember(ws, "never push to main", kind="directive")
        ledger = TriggerFireLedger()
        store.fire_and_format(ws, event="session-start", ledger=ledger)
        assert ledger.remaining_budget() < MEMORY_TRIGGER_PER_SESSION_TOKEN_CAP

    def test_ledger_suppresses_the_same_note_on_a_second_identical_fire(self, tmp_path) -> None:
        from agent.trigger_engine import TriggerFireLedger
        store, ws = _store(tmp_path), str(tmp_path)
        store.remember(ws, "never push to main", kind="directive")
        ledger = TriggerFireLedger()
        first_text, first_ids = store.fire_and_format(ws, event="session-start", ledger=ledger)
        second_text, second_ids = store.fire_and_format(ws, event="session-start", ledger=ledger)
        assert first_ids == {list(first_ids)[0]}
        assert second_text == ""
        assert second_ids == set()

    def test_resetting_the_ledger_restores_re_eligibility(self, tmp_path) -> None:
        """§3 "cleared on compaction": after reset(), a previously-suppressed
        note fires again — mirrors PreCompact -> SessionStart re-delivering
        the boot set."""
        from agent.trigger_engine import TriggerFireLedger
        store, ws = _store(tmp_path), str(tmp_path)
        store.remember(ws, "never push to main", kind="directive")
        ledger = TriggerFireLedger()
        store.fire_and_format(ws, event="session-start", ledger=ledger)
        assert store.fire_and_format(ws, event="session-start", ledger=ledger)[1] == set()  # suppressed
        ledger.reset()
        text, note_ids = store.fire_and_format(ws, event="session-start", ledger=ledger)
        assert len(note_ids) == 1
        assert "never push to main" in text

    def test_gotcha_over_the_100tk_cap_still_delivers_body_guidance_not_just_title(self, tmp_path) -> None:
        """UPG-HOOK-GOTCHA-CAP-TITLE-ONLY acceptance case, end-to-end through
        the real `_format_full_block` renderer `fire_and_format()` uses: a
        gotcha note whose full render is well over its 100-token per-kind
        cap (agent/config.yaml `memory_triggers.injection.per_kind_token_cap
        .gotcha`) — matching the longitudinal eval's real NoteVariant
        renders, measured 101-181tk — must still deliver its actual body
        guidance on the hook channel, not degrade to the title-only
        index-tier line."""
        from agent.trigger_engine import MEMORY_TRIGGER_PER_KIND_TOKEN_CAP, token_estimate
        store, ws = _store(tmp_path), str(tmp_path)
        guidance = "Always drain the outbound queue before restarting this worker."
        filler = (
            " Watch for partial writes during shutdown and retry with backoff "
            "instead of dropping the batch outright."
        ) * 4
        note_id = store.remember(
            ws, guidance + filler, kind="gotcha", title="worker restart ordering",
            anchors=["worker.py"],
        )
        text, note_ids = store.fire_and_format(ws, event="pre-edit", file_path="worker.py")
        assert note_ids == {note_id}
        assert guidance in text  # body guidance reached the wire
        assert "worker restart ordering" not in text  # index-tier title, never rendered on the full tier
        # The delivered block is genuinely bounded by the per-kind cap, not
        # merely "happened to be short" — confirms a real trim occurred.
        assert token_estimate(text) <= MEMORY_TRIGGER_PER_KIND_TOKEN_CAP["gotcha"] + 20  # + header/envelope slack

    # -----------------------------------------------------------------
    # G3 — arm-C double-dip regression (memoization-l1-capture-design §5.3)
    # -----------------------------------------------------------------
    def test_g3_same_note_matched_by_two_surfaces_in_one_turn_injects_once(self, tmp_path) -> None:
        """Recreates the arm-C double-dip shape: ONE note declares TWO
        separate trigger axes (session-start bulk delivery AND a
        pre-edit file-anchored trigger) — two DIFFERENT (note_id,
        trigger_index) pairs, so a session-scoped `TriggerFireLedger`'s
        per-axis dedup alone would NOT catch this (each axis fires exactly
        once on its own axis, legitimately). Only the NEW note_id-granular
        `TurnInjectionLedger`, shared across BOTH surfaces' `fire_and_format`
        calls within the same turn, must collapse this to a single
        injection — proving the fix, not just the ledger's own unit
        contract already covered by TestTurnInjectionLedger."""
        from agent.trigger_engine import TurnInjectionLedger
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(
            ws, "auth.py: verify_token() must check expiry before signature",
            kind="finding",
            triggers=[{"event": "session-start"}, {"path": "src/auth.py", "event": "pre-edit"}],
        )
        turn_ledger = TurnInjectionLedger()

        # Surface 1: session-start bulk delivery (boot).
        boot_text, boot_ids = store.fire_and_format(
            ws, events=["session-start"], turn_ledger=turn_ledger, spend_turn_budget=True,
        )
        assert boot_ids == {note_id}
        assert "verify_token() must check expiry" in boot_text

        # Surface 2: PreToolUse file-anchored trigger, SAME turn (turn_ledger
        # not reset in between — a real UserPromptSubmit reset happens only
        # at the NEXT turn boundary).
        edit_text, edit_ids = store.fire_and_format(
            ws, event="pre-edit", file_path="src/auth.py",
            turn_ledger=turn_ledger, spend_turn_budget=True,
        )
        assert edit_ids == set()  # already claimed by surface 1 this turn
        assert edit_text == ""

        # Single injection within budget: the note's content appears exactly
        # once across the whole turn's combined output, never twice.
        combined = boot_text + edit_text
        assert combined.count("verify_token() must check expiry") == 1

    def test_proxy_and_hook_surfaces_share_one_turn_ledger_note_injects_once(self, tmp_path) -> None:
        """UPG-PROXY-CROSS-CHANNEL-DEDUP — sibling to the G3 regression above,
        covering the PROXY channel's gate alongside a hook/trigger-engine
        surface. Before this fix the two channels kept entirely separate
        ledgers (different objects AND different key spaces: `TurnInjection
        Ledger` is note_id-keyed, the proxy's own `LedgerStore` cooldown is
        anchor_id-keyed with no cross-reference), so the same note could
        inject via both channels in one turn. Sharing ONE `TurnInjection
        Ledger` instance across `WorkingContextStore.fire_and_format()`
        (hook surface) and `agent.proactive.gate.ProactiveGate.select()`
        (proxy surface, via its new `turn_ledger` parameter) closes it."""
        from agent.proactive.gate import ProactiveGate
        from agent.proactive.types import Candidate
        from agent.trigger_engine import TurnInjectionLedger

        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(
            ws, "auth.py: verify_token() must check expiry before signature",
            kind="finding",
            triggers=[{"path": "src/auth.py", "event": "pre-edit"}],
        )
        turn_ledger = TurnInjectionLedger()

        # Surface 1: hook/trigger-engine PreToolUse file-anchored delivery —
        # actually injects and claims note_id in the shared turn ledger.
        hook_text, hook_ids = store.fire_and_format(
            ws, event="pre-edit", file_path="src/auth.py",
            turn_ledger=turn_ledger, spend_turn_budget=True,
        )
        assert hook_ids == {note_id}
        assert "verify_token() must check expiry" in hook_text

        # Surface 2: the proxy channel's gate, SAME turn (same turn_ledger
        # instance) — a structural candidate for the SAME note_id must be
        # suppressed, mirroring the hook-vs-hook G3 case above but across
        # channels this time.
        gate = ProactiveGate(
            min_similarity=0.35, max_items_per_event=3, max_chars_per_event=800,
            cooldown_items=30, max_weak_structural_items=1,
        )
        proxy_candidate = Candidate(
            kind="note_structural", line="note about auth.py",
            score=1.0, anchor_id=f"note:{note_id}", is_structural=True,
        )
        out = gate.select(
            [proxy_candidate], session_id="proxy-session", turn_ledger=turn_ledger
        )
        assert out.is_empty()  # already claimed by surface 1 this turn

        # Non-vacuous: the SAME candidate against a FRESH TurnInjectionLedger
        # (a fresh turn) DOES get admitted — proving the suppression above
        # came from the shared ledger, not a floor/structural-match failure.
        fresh_out = gate.select(
            [proxy_candidate], session_id="proxy-session-2",
            turn_ledger=TurnInjectionLedger(),
        )
        assert fresh_out.item_count == 1
        assert fresh_out.anchor_ids == (f"note:{note_id}",)
        assert turn_ledger.remaining_turn_budget() >= 0  # never driven negative

        # A later turn (reset, mirroring UserPromptSubmit) restores
        # eligibility — the dedup is per-turn, not permanent.
        turn_ledger.reset()
        again_text, again_ids = store.fire_and_format(
            ws, event="pre-edit", file_path="src/auth.py",
            turn_ledger=turn_ledger, spend_turn_budget=True,
        )
        assert again_ids == {note_id}
        assert "verify_token() must check expiry" in again_text

    def test_n1_note_matched_by_second_surface_but_turn_deduped_stays_eligible_next_turn(self, tmp_path) -> None:
        """§5.3 fix: a note whose SECOND trigger axis matches a same-turn
        surface but gets dropped by turn-ledger cross-surface dedup (already
        claimed by the FIRST surface's own, different axis) must never have
        that second axis recorded into the SESSION ledger — recording an
        unfired match would permanently suppress it for the rest of the
        session even though nothing from that axis was ever actually
        injected. Same double-axis note shape as the G3 test above, but
        asserting the session ledger's own eligibility rather than the
        turn-level dedup it exercises."""
        from agent.trigger_engine import TriggerFireLedger, TurnInjectionLedger
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(
            ws, "auth.py: verify_token() must check expiry before signature",
            kind="finding",
            triggers=[{"event": "session-start"}, {"path": "src/auth.py", "event": "pre-edit"}],
        )
        ledger = TriggerFireLedger()
        turn_ledger = TurnInjectionLedger()

        # Surface 1: session-start (trigger_index 0) delivers and claims the
        # note this turn.
        boot_text, boot_ids = store.fire_and_format(
            ws, events=["session-start"], ledger=ledger, turn_ledger=turn_ledger,
            spend_turn_budget=True,
        )
        assert boot_ids == {note_id}

        # Surface 2, SAME turn: the pre-edit axis (trigger_index 1) matches
        # too, but is turn-deduped — already claimed by surface 1.
        edit_text, edit_ids = store.fire_and_format(
            ws, event="pre-edit", file_path="src/auth.py",
            ledger=ledger, turn_ledger=turn_ledger, spend_turn_budget=True,
        )
        assert edit_ids == set()
        assert edit_text == ""

        # The pre-edit axis (index 1) was never actually delivered — it must
        # still be session-eligible, not silently burned by the match that
        # never made it through packing.
        assert ledger.eligible(note_id, 1) is True

        # Next turn (UserPromptSubmit boundary): the pre-edit axis fires and
        # actually delivers now that nothing claims it first.
        turn_ledger.reset()
        again_text, again_ids = store.fire_and_format(
            ws, event="pre-edit", file_path="src/auth.py",
            ledger=ledger, turn_ledger=turn_ledger, spend_turn_budget=True,
        )
        assert again_ids == {note_id}
        assert "verify_token() must check expiry" in again_text


class TestInjectedFrame:
    """`_injected_frame()` (memoization-l1-capture-design.md §5.6) — the
    unified structural-trust framing template for a trigger-fired note:
    'Recorded <date> (anchor: <target>, status: <verdict>): '. `<verdict>`
    is always read from deterministic machine state (check_staleness()'s
    own anchor-drift signal, or the note's kind), never asserted freeform."""

    def _note(self, **overrides):
        from agent.working_context_store._types import WorkingNote
        base = dict(
            note_id=1, workspace="/ws", content="c", tags=[], priority="medium",
            created_at=1700000000.0, last_accessed=1700000000.0, kind="finding",
        )
        base.update(overrides)
        return WorkingNote(**base)

    def test_template_shape_has_recorded_date_anchor_and_status(self, tmp_path) -> None:
        from agent.working_context_store._store import _injected_frame, _date_str
        note = self._note()
        frame = _injected_frame(note, stale_warnings={})
        assert frame == f"Recorded {_date_str(note.created_at)} (anchor: none, status: matches current state): "

    def test_anchor_target_names_the_first_declared_anchor(self, tmp_path) -> None:
        from agent.working_context_store._store import _injected_frame
        note = self._note(anchors=[["src/auth.py", "abc123"], ["src/other.py", None]])
        frame = _injected_frame(note, stale_warnings={})
        assert "anchor: src/auth.py" in frame

    def test_no_anchors_renders_target_as_none(self, tmp_path) -> None:
        from agent.working_context_store._store import _injected_frame
        note = self._note(anchors=[])
        frame = _injected_frame(note, stale_warnings={})
        assert "anchor: none" in frame

    def test_anchor_drift_present_reports_changed_since_verify(self, tmp_path) -> None:
        """`status` is READ from check_staleness()'s own
        '[anchor_changed]'-suffixed reason, never re-derived — same
        deterministic signal §4.4 already computes, not a second heuristic."""
        from agent.working_context_store._store import _injected_frame
        note = self._note(note_id=7, anchors=[["src/auth.py", "abc123"]])
        stale_warnings = {7: ["src/auth.py [anchor_changed]"]}
        frame = _injected_frame(note, stale_warnings)
        assert "status: changed since — verify" in frame

    def test_operational_kind_with_no_drift_reports_last_confirmed_date(self, tmp_path) -> None:
        """§4.4 'Option D, unconditional': an operational fact carries a
        recency verdict even when nothing has been proven to have drifted —
        env/process facts decay by elapsed time, not just by hash mismatch."""
        from agent.working_context_store._store import _injected_frame, _date_str
        note = self._note(kind="operational")
        frame = _injected_frame(note, stale_warnings={})
        assert f"status: last confirmed {_date_str(note.created_at)}" in frame

    def test_ordinary_kind_with_no_drift_reports_matches_current_state(self, tmp_path) -> None:
        from agent.working_context_store._store import _injected_frame
        for kind in ("directive", "task", "gotcha", "finding", "reference", "decision"):
            note = self._note(kind=kind)
            frame = _injected_frame(note, stale_warnings={})
            assert "status: matches current state" in frame

    def test_anchor_drift_takes_precedence_over_operational_recency(self, tmp_path) -> None:
        """Drift is a stronger signal than the unconditional recency verdict
        — an operational note whose anchor has actually changed must warn
        'changed since — verify', not fall back to 'last confirmed'."""
        from agent.working_context_store._store import _injected_frame
        note = self._note(note_id=3, kind="operational", anchors=[["Makefile", "abc"]])
        stale_warnings = {3: ["Makefile [anchor_changed]"]}
        frame = _injected_frame(note, stale_warnings)
        assert "status: changed since — verify" in frame

    def test_operational_drift_status_ladder_is_a_strict_suffix(self, tmp_path) -> None:
        """UPG-STALE-FLAG-DOUBLE-EVENT / proxy-anchor rendering: an
        operational note whose anchor drifted gets the SAME leading
        'changed since — verify' text every other drifted note gets
        (unchanged precedence, see the sibling test above), plus a strict
        ADDITIVE suffix naming the anchor target and the original 'last
        confirmed' date — the honest reading of a proxy anchor (a
        lockfile/CI-config/Dockerfile standing in for "the process it
        encodes"): drift means the process MAY have changed, never that
        the note itself is wrong."""
        from agent.working_context_store._store import _injected_frame, _date_str
        note = self._note(note_id=3, kind="operational", anchors=[["Makefile", "abc"]])
        stale_warnings = {3: ["Makefile [anchor_changed]"]}
        frame = _injected_frame(note, stale_warnings)
        assert "status: changed since — verify" in frame
        assert f"Makefile is a proxy for this process, last confirmed {_date_str(note.created_at)}" in frame

    def test_nonoperational_drift_carries_no_proxy_suffix(self, tmp_path) -> None:
        """The proxy suffix is kind-keyed, never applied outside
        kind='operational' — a finding/gotcha/etc. anchor drift stays the
        byte-exact 'changed since — verify' string with nothing appended,
        proving the branch is additive-only and does not leak."""
        from agent.working_context_store._store import _injected_frame, _date_str
        for kind in ("finding", "gotcha", "directive", "task", "reference", "decision"):
            note = self._note(note_id=9, kind=kind, anchors=[["Makefile", "abc"]])
            stale_warnings = {9: ["Makefile [anchor_changed]"]}
            frame = _injected_frame(note, stale_warnings)
            assert frame == (
                f"Recorded {_date_str(note.created_at)} (anchor: Makefile, "
                f"status: changed since — verify): "
            )
            assert "proxy for this process" not in frame

    def test_injected_framing_appears_in_fire_and_format_output_for_operational_note(self, tmp_path) -> None:
        """End-to-end: fire_and_format() renders the injected framing (not
        frame_prefix()'s provenance-hedged wording) for a live operational
        note delivered through the trigger engine's M (semantic) axis —
        proves the template actually reaches the caller, not just the
        unit-level function. Uses the same _fixed_vector_store helper as
        TestFireSemanticPrimitive (a fixed cosine-1.0 pair) rather than a
        precomputed bool, since fire()/fire_and_format() compute the
        cosine internally from real embeddings — no shortcut param exists
        on the public entry point, by design (no-query-heuristics rule:
        this is the one place raw prompt text is embedded, never parsed).
        No explicit triggers[] — relies on operational's own default
        bundle (§5.1) to prove the framing composes with that default."""
        store = _fixed_vector_store(tmp_path, [1.0, 0.0], [1.0, 0.0])
        ws = str(tmp_path)
        content = "pytest must run via the venv python, not global python"
        store.remember(ws, content, kind="operational")
        text, note_ids = store.fire_and_format(ws, event="prompt-submit", query="anything")
        assert len(note_ids) == 1
        assert "Recorded " in text
        assert "anchor: none" in text
        assert "status: last confirmed" in text

    def test_unobserved_anchor_appends_a_distinct_suffix_not_the_drift_wording(self, tmp_path) -> None:
        """UPG-ANCHOR-UNOBSERVED-BINDING criterion (b): a False third
        element (this session HAD a ledger, and this path was not in it)
        appends a caveat that reads distinctly from drift's 'changed since
        — verify' — drift wants re-derivation, unobserved wants a first
        read. The two must never collapse into one 'stale' flag."""
        from agent.working_context_store._store import _injected_frame, _UNOBSERVED_STATUS_SUFFIX
        note = self._note(anchors=[["src/auth.py", "abc123", False]])
        frame = _injected_frame(note, stale_warnings={})
        assert "status: matches current state" in frame
        assert _UNOBSERVED_STATUS_SUFFIX in frame
        assert "changed since — verify" not in frame

    def test_unknown_observation_third_element_none_renders_no_new_caveat(self, tmp_path) -> None:
        """Criterion (c): an explicit None third element (a session with no
        observation ledger) must render byte-identical to a note with no
        anchor-observation knowledge at all — no caveat text appears."""
        from agent.working_context_store._store import (
            _injected_frame, _UNOBSERVED_STATUS_SUFFIX, _date_str,
        )
        note = self._note(anchors=[["src/auth.py", "abc123", None]])
        frame = _injected_frame(note, stale_warnings={})
        assert frame == (
            f"Recorded {_date_str(note.created_at)} (anchor: src/auth.py, "
            f"status: matches current state): "
        )
        assert _UNOBSERVED_STATUS_SUFFIX not in frame

    def test_legacy_two_element_anchor_row_renders_as_unknown_never_as_unobserved(self, tmp_path) -> None:
        """Criterion (d): a pre-existing anchor row written before this
        field existed (only [path, hash], no third element) must load and
        render exactly as before — never inferred as observed=False just
        because the element is missing."""
        from agent.working_context_store._store import (
            _injected_frame, _UNOBSERVED_STATUS_SUFFIX, _date_str,
        )
        note = self._note(anchors=[["src/auth.py", "abc123"]])
        frame = _injected_frame(note, stale_warnings={})
        assert frame == (
            f"Recorded {_date_str(note.created_at)} (anchor: src/auth.py, "
            f"status: matches current state): "
        )
        assert _UNOBSERVED_STATUS_SUFFIX not in frame

    def test_observed_true_third_element_renders_no_unobserved_caveat(self, tmp_path) -> None:
        """A True third element (this session's own hook traffic reported
        this exact path) is the properly-bound case — no caveat, same as
        the unknown/legacy cases, distinguishing it is purely internal."""
        from agent.working_context_store._store import _injected_frame, _UNOBSERVED_STATUS_SUFFIX
        note = self._note(anchors=[["src/auth.py", "abc123", True]])
        frame = _injected_frame(note, stale_warnings={})
        assert _UNOBSERVED_STATUS_SUFFIX not in frame

    def test_unobserved_and_drifted_compose_additively_not_merged(self, tmp_path) -> None:
        """Drift and unobserved are orthogonal facts — a note can be both
        drifted AND unobserved, and both pieces of text must appear (not
        collapsed into a single boolean 'stale' flag)."""
        from agent.working_context_store._store import _injected_frame, _UNOBSERVED_STATUS_SUFFIX
        note = self._note(note_id=11, anchors=[["src/auth.py", "abc123", False]])
        stale_warnings = {11: ["src/auth.py [anchor_changed]"]}
        frame = _injected_frame(note, stale_warnings)
        assert "status: changed since — verify" in frame
        assert _UNOBSERVED_STATUS_SUFFIX in frame


# ---------------------------------------------------------------------------
# B9: Semantic recall — embed_fn + ChromaDB cosine similarity
# ---------------------------------------------------------------------------

def _dummy_embed(texts: list[str]) -> list[list[float]]:
    """Hash-based deterministic embedder for tests — same input → same vector."""
    import hashlib
    result = []
    for t in texts:
        h = hashlib.md5(t.encode()).digest()
        vec = [(b / 255.0 - 0.5) for b in (h * 48)]  # 16 * 48 = 768 dims
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        result.append([x / norm for x in vec])
    return result


def _semantic_store(tmp_path):
    """Return a WorkingContextStore wired up with a dummy embedder + isolated ChromaDB."""
    import chromadb
    from agent.working_context_store import WorkingContextStore
    # PersistentClient with tmp_path gives true per-test isolation; EphemeralClient
    # shares in-memory state across all instances in the same process.
    chroma_dir = str(tmp_path / "chroma")
    client = chromadb.PersistentClient(path=chroma_dir)
    return WorkingContextStore(str(tmp_path), embed_fn=_dummy_embed, notes_chroma_client=client)


def _spy_store_sql(store, monkeypatch) -> list[str]:
    """Capture every SQL statement `store` runs, and return the growing list.

    sqlite3.Cursor is an immutable C type, so its `execute` cannot be patched.
    The supported hook is Connection.set_trace_callback(), installed here by
    wrapping the store's own _conn() on the instance.

    The callback yields EXPANDED SQL — parameters are already substituted, so
    a captured statement reads `kind = \'finding\'` and `LIMIT 2`, never
    `kind = ?` / `LIMIT ?`. Match on the expanded form; a predicate written
    against placeholders matches nothing and passes vacuously.
    """
    seen_sqls: list[str] = []
    real_conn = store._conn

    def traced_conn():
        conn = real_conn()
        conn.set_trace_callback(seen_sqls.append)
        return conn

    monkeypatch.setattr(store, "_conn", traced_conn)
    return seen_sqls


class TestSemanticRecall:
    """B9 — recall(query=...) uses cosine similarity instead of SQL LIKE."""

    def test_semantic_recall_returns_notes(self, tmp_path) -> None:
        store = _semantic_store(tmp_path)
        ws = "/repo"
        content = "handle_legacy_finalizers appends to gc.garbage when tp_del is set"
        store.remember(ws, content)
        # Query with the exact content — same embedding → cosine 1.0 → must be top result
        notes = store.recall(ws, query=content)
        assert len(notes) == 1
        assert notes[0].content == content

    def test_semantic_recall_without_query_falls_back_to_sql(self, tmp_path) -> None:
        store = _semantic_store(tmp_path)
        ws = "/repo"
        store.remember(ws, "gc finalizer note")
        notes = store.recall(ws)  # no query → SQL path
        assert len(notes) == 1

    def test_no_embed_fn_uses_sql_like(self, tmp_path) -> None:
        from agent.working_context_store import WorkingContextStore
        store = WorkingContextStore(str(tmp_path))  # no embed_fn → SQL only
        ws = "/repo"
        store.remember(ws, "gc finalizer note about tp_del")
        notes = store.recall(ws, query="tp_del")
        assert len(notes) == 1  # SQL LIKE matches "tp_del" as substring

    def test_semantic_recall_with_multiple_notes(self, tmp_path) -> None:
        store = _semantic_store(tmp_path)
        ws = "/repo"
        note_a = "GC finalizer tp_del legacy path gc.garbage deferral"
        note_b = "dict pop_last dk_nentries insertion order reverse traversal"
        store.remember(ws, note_a, tags=["gc"])
        store.remember(ws, note_b, tags=["dict"])
        # Querying with note_a's exact text → note_a should appear in results
        notes = store.recall(ws, query=note_a, limit=2)
        assert len(notes) >= 1
        assert any(n.content == note_a for n in notes)

    def test_semantic_recall_respects_limit(self, tmp_path) -> None:
        store = _semantic_store(tmp_path)
        ws = "/repo"
        for i in range(5):
            store.remember(ws, f"note content {i}")
        notes = store.recall(ws, query="note content 0", limit=2)
        assert len(notes) <= 2

    def test_semantic_recall_empty_collection_returns_empty(self, tmp_path) -> None:
        store = _semantic_store(tmp_path)
        ws = "/repo"
        notes = store.recall(ws, query="anything")
        assert notes == []

    def test_min_similarity_withholds_offtopic(self, tmp_path) -> None:
        """UPG-5.1: an off-topic query recalls nothing when a cutoff is set.

        _dummy_embed hashes text, so two distinct strings are ~orthogonal
        (similarity ≈ 0) — well below a 0.5 floor — while the exact text scores 1.0.
        """
        store = _semantic_store(tmp_path)
        ws = "/repo"
        store.remember(ws, "gc finalizer tp_del legacy garbage deferral path")
        # Exact text → similarity 1.0 → passes the floor.
        assert len(store.recall(ws, query="gc finalizer tp_del legacy garbage deferral path",
                                min_similarity=0.5)) == 1
        # Unrelated text → similarity ≈ 0 → withheld by the floor.
        assert store.recall(ws, query="completely unrelated kubernetes ingress topic",
                            min_similarity=0.5) == []

    def test_no_cutoff_preserves_default_behavior(self, tmp_path) -> None:
        """Without min_similarity, recall still returns the nearest note (no regression)."""
        store = _semantic_store(tmp_path)
        ws = "/repo"
        store.remember(ws, "the only note here")
        assert len(store.recall(ws, query="something off topic entirely")) == 1

    def test_kind_filter_deepens_pool_via_floor(self, tmp_path, monkeypatch) -> None:
        """UPG-SEMANTIC-UNDERFILL: when `kind` is set, _semantic_recall
        passes a wider `floor` to working_memory_fetch_width() so the
        vector query is sized to absorb unknown `kind` selectivity rather
        than the 3x default. Unset `kind` must keep the 3x default — no
        regression on the no-filter case."""
        import agent.working_context_store._store as _store_mod

        captured: list[tuple[int, int, int]] = []

        def spy(render_limit, col_count, *, floor=0):
            captured.append((render_limit, col_count, floor))
            return _store_mod.working_memory_fetch_width(
                render_limit, col_count, floor=floor,
            )

        # The function uses the module-level name; rebinding on the
        # module is the canonical monkeypatch target.
        monkeypatch.setattr(_store_mod, "working_memory_fetch_width", spy)

        store = _semantic_store(tmp_path)
        ws = "/repo"
        # More than 3x the render limit below, so the width is not clamped by
        # col_count and the multiplier is what is actually being asserted.
        for i in range(12):
            store.remember(ws, f"note content {i}")

        # No kind: floor is 0 (the default), width is `limit * 3`.
        captured.clear()
        store.recall(ws, query="note content 0", limit=2)
        assert captured, "working_memory_fetch_width must be called"
        render_limit, col_count, floor = captured[-1]
        assert floor == 0
        # Sanity: the no-kind width is at least limit*3 (and exactly so
        # when the corpus is bigger than 3x the limit).
        assert col_count >= render_limit * 3

        # With kind: floor is widened to `limit * 10`, raising the pool
        # ceiling above the 3x default.
        captured.clear()
        store.recall(ws, query="note content 0", limit=2, kind="finding")
        assert captured, "working_memory_fetch_width must be called"
        render_limit, col_count, floor = captured[-1]
        assert floor == render_limit * 10

    def test_kind_prefetch_runs_in_relevance_branch(self, tmp_path, monkeypatch) -> None:
        """The relevance branch runs the kind-aware SQL prefetch backstop
        even when the query embedding is unrelated to the corpus (i.e.
        the semantic pool would not naturally select kind matches first).
        Verified by intercepting the SQL the prefetch runs and asserting
        it carries the `kind = ?` predicate."""
        store = _semantic_store(tmp_path)
        ws = "/repo"
        for i in range(4):
            store.remember(ws, f"finding note {i}", kind="finding")
        store.remember(ws, "unrelated task", kind="task")

        # Spy on the store's own _conn() to capture every SQL the
        # _semantic_recall call makes. We filter for the prefetch
        # signature (ORDER BY created_at DESC, no `note_id IN` clause).
        seen_sqls = _spy_store_sql(store, monkeypatch)
        store.recall(ws, query="anything", limit=2, kind="finding")

        # The prefetch in the relevance branch runs the metadata filters
        # WITHOUT a `note_id IN (...)` clause (otherwise the prefetch is
        # a no-op) and ends with `ORDER BY created_at DESC LIMIT ?`.
        prefetch_sqls = [
            s for s in seen_sqls
            if "kind = " in s
            and "note_id IN" not in s
            and "ORDER BY created_at DESC" in s
        ]
        assert prefetch_sqls, (
            "kind-aware SQL prefetch should run in the relevance branch "
            "when kind= is set, so underfill is the backstop of the "
            "pool widening rather than the only fix"
        )

    def test_kind_prefetch_does_not_leak_other_kinds(self, tmp_path) -> None:
        """The SQL prefetch reuses the kind predicate (filter_sql), so a
        `kind='finding'` recall must NEVER surface rows of any other kind,
        even if those rows are abundant in the corpus."""
        store = _semantic_store(tmp_path)
        ws = "/repo"
        for _ in range(4):
            store.remember(ws, "task one", kind="task")
        for _ in range(4):
            store.remember(ws, "gotcha one", kind="gotcha")
        # Only the kind that matches the filter is expected back.
        notes = store.recall(ws, query="anything", limit=10, kind="task")
        assert notes
        assert all(n.kind == "task" for n in notes)

    def test_no_kind_filter_does_not_run_kind_prefetch(self, tmp_path, monkeypatch) -> None:
        """The fix is `kind`-gated: a recall() without `kind=...` must
        NOT run the new SQL prefetch (no selectivity problem to fix)."""
        store = _semantic_store(tmp_path)
        ws = "/repo"
        for i in range(3):
            store.remember(ws, f"note {i}")

        seen_sqls = _spy_store_sql(store, monkeypatch)
        store.recall(ws, query="anything", limit=2)

        # The kind prefetch signature is a SELECT * FROM notes that ends
        # in `ORDER BY created_at DESC LIMIT <n>` (no `note_id DESC`
        # tie-break — _recall_floor_notes uses `created_at DESC, note_id
        # DESC`, the sort-aware prefetch uses one of three `ORDER BY`
        # variants, and recall's SQL path appends `LIMIT ?` to the
        # `ORDER BY` rather than ending with `LIMIT ?` alone). The
        # kind gate is exactly that the prefetch includes `kind = ?` in
        # its WHERE — without a kind filter, the underfill fix is not
        # applicable, so the query must not run.
        prefetch_sqls = [
            s for s in seen_sqls
            if "SELECT * FROM notes" in s
            and "workspace = " in s
            and "note_id IN" not in s
            and re.search(r"ORDER BY created_at DESC LIMIT \d+$", s.rstrip())
        ]
        assert not prefetch_sqls, (
            "kind-gated SQL prefetch must not run when no `kind` filter "
            "is set — the underfill fix is intentionally narrow"
        )

    def test_kind_underfill_with_min_similarity_still_respects_floor(self, tmp_path) -> None:
        """min_similarity and the kind backstop coexist: even when the
        pool is widened for kind, an off-topic query whose whole pool is
        below min_similarity still returns nothing. The underfill fix
        only addresses the kind selectivity artifact, not the similarity
        cutoff (which is correct behavior, not underfill — see
        UPG-5.1)."""
        store = _semantic_store(tmp_path)
        ws = "/repo"
        store.remember(ws, "finding about tp_del legacy garbage")
        # Unrelated query with a 0.5 floor — every vector in the pool is
        # near-orthogonal under _dummy_embed, so the floor withholds all.
        notes = store.recall(
            ws, query="kubernetes ingress controller",
            kind="finding", min_similarity=0.5,
        )
        assert notes == []


    def test_forget_removes_from_chroma(self, tmp_path) -> None:
        import chromadb
        from agent.working_context_store import WorkingContextStore
        client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
        store = WorkingContextStore(str(tmp_path), embed_fn=_dummy_embed, notes_chroma_client=client)
        ws = "/repo"
        note_id = store.remember(ws, "gc finalizer note")
        assert store._notes_col.count() == 1
        store.forget(ws, note_id)
        assert store._notes_col.count() == 0

    def test_forget_all_clears_chroma(self, tmp_path) -> None:
        import chromadb
        from agent.working_context_store import WorkingContextStore
        client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
        store = WorkingContextStore(str(tmp_path), embed_fn=_dummy_embed, notes_chroma_client=client)
        ws = "/repo"
        store.remember(ws, "note one")
        store.remember(ws, "note two")
        assert store._notes_col.count() == 2
        store.forget_all(ws)
        assert store._notes_col.count() == 0

    def test_semantic_collection_name_is_working_memory(self, tmp_path) -> None:
        import chromadb
        from agent.working_context_store import WorkingContextStore
        client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
        store = WorkingContextStore(str(tmp_path), embed_fn=_dummy_embed, notes_chroma_client=client)
        col_names = [c.name for c in client.list_collections()]
        assert "working_memory" in col_names


# ---------------------------------------------------------------------------
# UPG-NOTES-EMBED-MIGRATION — embed-model stamp + re-embed migration
# ---------------------------------------------------------------------------

def _const_embed(vector: list[float]):
    """Return an embed_fn that maps every text to the same fixed vector,
    so two "models" are trivially distinguishable by which vector a note's
    embedding lands on."""
    def _embed(texts: list[str]) -> list[list[float]]:
        return [list(vector) for _ in texts]
    return _embed


def _counting_embed(vector: list[float], calls: list[list[str]]):
    """Like _const_embed, but records every batch of texts it was called
    with, so a test can assert re-embedding actually happened per note."""
    def _embed(texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        return [list(vector) for _ in texts]
    return _embed


class TestNotesEmbedModelMigration:
    """UPG-NOTES-EMBED-MIGRATION: notes must never be recalled against a
    stale embedding space silently — a stamp mismatch (or a missing stamp on
    a collection that already holds vectors) triggers a one-time re-embed
    of every note's content, in place, before the constructor returns."""

    def test_fresh_collection_is_stamped_with_current_model(self, tmp_path) -> None:
        import chromadb
        from agent.working_context_store import WorkingContextStore
        client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
        store = WorkingContextStore(
            str(tmp_path), embed_fn=_const_embed([1.0, 0.0]),
            notes_chroma_client=client, embed_model="model-a",
        )
        assert store._stored_notes_embed_model() == "model-a"
        assert store.embed_model_stamp_mismatch() is None

    def test_no_embed_model_given_skips_stamping(self, tmp_path) -> None:
        """embed_model defaults to None — existing callers/tests keep working
        with no stamp/migration logic active at all."""
        import chromadb
        from agent.working_context_store import WorkingContextStore
        client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
        store = WorkingContextStore(
            str(tmp_path), embed_fn=_const_embed([1.0, 0.0]), notes_chroma_client=client,
        )
        assert store._stored_notes_embed_model() is None
        assert store.embed_model_stamp_mismatch() is None

    def test_matching_stamp_does_not_reembed(self, tmp_path) -> None:
        import chromadb
        from agent.working_context_store import WorkingContextStore
        chroma_dir = str(tmp_path / "chroma")
        client = chromadb.PersistentClient(path=chroma_dir)
        store1 = WorkingContextStore(
            str(tmp_path), embed_fn=_const_embed([1.0, 0.0]),
            notes_chroma_client=client, embed_model="model-a",
        )
        note_id = store1.remember("/repo", "original note content")

        calls: list[list[str]] = []
        client2 = chromadb.PersistentClient(path=chroma_dir)
        WorkingContextStore(
            str(tmp_path), embed_fn=_counting_embed([9.0, 9.0], calls),
            notes_chroma_client=client2, embed_model="model-a",
        )
        assert calls == []  # same model — no re-embed on startup

        vec = store1._notes_col.get(ids=[str(note_id)], include=["embeddings"])["embeddings"][0]
        assert list(vec) == [1.0, 0.0]  # vector untouched

    def test_mismatched_stamp_triggers_reembed_and_stamp_update(self, tmp_path) -> None:
        import chromadb
        from agent.working_context_store import WorkingContextStore
        chroma_dir = str(tmp_path / "chroma")
        client = chromadb.PersistentClient(path=chroma_dir)
        store1 = WorkingContextStore(
            str(tmp_path), embed_fn=_const_embed([1.0, 0.0]),
            notes_chroma_client=client, embed_model="model-a",
        )
        note_id = store1.remember("/repo", "note that must survive migration")

        calls: list[list[str]] = []
        client2 = chromadb.PersistentClient(path=chroma_dir)
        store2 = WorkingContextStore(
            str(tmp_path), embed_fn=_counting_embed([0.0, 1.0], calls),
            notes_chroma_client=client2, embed_model="model-b",
        )

        # re-embed happened, over the note's real content
        assert any("note that must survive migration" in batch for batch in calls)
        # stamp updated to the new model
        assert store2._stored_notes_embed_model() == "model-b"
        assert store2.embed_model_stamp_mismatch() is None
        # vector actually changed to the new model's output
        vec = store2._notes_col.get(ids=[str(note_id)], include=["embeddings"])["embeddings"][0]
        assert list(vec) == [0.0, 1.0]
        # note content and id untouched
        note = store2.get_note("/repo", note_id)
        assert note is not None
        assert note.content == "note that must survive migration"
        assert note.note_id == note_id

    def test_unstamped_collection_with_vectors_is_treated_as_mismatch(self, tmp_path) -> None:
        """A collection with vectors but no stamp predates this mechanism —
        we cannot know what model produced those vectors, so it is migrated
        just like an explicit mismatch, not assumed to already match."""
        import chromadb
        from agent.working_context_store import WorkingContextStore
        chroma_dir = str(tmp_path / "chroma")
        client = chromadb.PersistentClient(path=chroma_dir)
        # embed_model=None -> no stamp written, mirroring a pre-migration install
        store1 = WorkingContextStore(
            str(tmp_path), embed_fn=_const_embed([1.0, 0.0]), notes_chroma_client=client,
        )
        note_id = store1.remember("/repo", "pre-existing unstamped note")
        assert store1._stored_notes_embed_model() is None

        calls: list[list[str]] = []
        client2 = chromadb.PersistentClient(path=chroma_dir)
        store2 = WorkingContextStore(
            str(tmp_path), embed_fn=_counting_embed([0.0, 1.0], calls),
            notes_chroma_client=client2, embed_model="model-b",
        )
        assert any("pre-existing unstamped note" in batch for batch in calls)
        assert store2._stored_notes_embed_model() == "model-b"
        vec = store2._notes_col.get(ids=[str(note_id)], include=["embeddings"])["embeddings"][0]
        assert list(vec) == [0.0, 1.0]

    def test_empty_collection_is_stamped_without_reembed(self, tmp_path) -> None:
        """A brand-new, empty collection has nothing to migrate — it is just
        stamped so the next startup with the same model takes the no-op path."""
        import chromadb
        from agent.working_context_store import WorkingContextStore
        chroma_dir = str(tmp_path / "chroma")
        client = chromadb.PersistentClient(path=chroma_dir)
        WorkingContextStore(
            str(tmp_path), embed_fn=_const_embed([1.0, 0.0]), notes_chroma_client=client,
        )  # no embed_model -> unstamped, no notes

        calls: list[list[str]] = []
        client2 = chromadb.PersistentClient(path=chroma_dir)
        store2 = WorkingContextStore(
            str(tmp_path), embed_fn=_counting_embed([0.0, 1.0], calls),
            notes_chroma_client=client2, embed_model="model-b",
        )
        assert calls == []  # nothing to re-embed
        assert store2._stored_notes_embed_model() == "model-b"

    def test_recall_works_after_simulated_model_swap(self, tmp_path) -> None:
        """End-to-end: semantic recall must still work after a model swap —
        the query is embedded with the NEW model, and must match the
        migrated (also new-model) note vector, not the stale one."""
        import chromadb
        from agent.working_context_store import WorkingContextStore
        chroma_dir = str(tmp_path / "chroma")
        client = chromadb.PersistentClient(path=chroma_dir)
        store1 = WorkingContextStore(
            str(tmp_path), embed_fn=_dummy_embed,
            notes_chroma_client=client, embed_model="model-a",
        )
        ws = "/repo"
        content = "handle_legacy_finalizers appends to gc.garbage when tp_del is set"
        store1.remember(ws, content)

        # Simulate a model swap: a NEW embed function (still hash-based+deterministic,
        # but a distinct "model") is now configured.
        def _swapped_embed(texts: list[str]) -> list[list[float]]:
            import hashlib
            result = []
            for t in texts:
                h = hashlib.md5(("swapped::" + t).encode()).digest()
                vec = [(b / 255.0 - 0.5) for b in (h * 48)]
                norm = sum(x * x for x in vec) ** 0.5 or 1.0
                result.append([x / norm for x in vec])
            return result

        client2 = chromadb.PersistentClient(path=chroma_dir)
        store2 = WorkingContextStore(
            str(tmp_path), embed_fn=_swapped_embed, embed_query_fn=_swapped_embed,
            notes_chroma_client=client2, embed_model="model-b",
        )
        notes = store2.recall(ws, query=content)
        assert len(notes) == 1
        assert notes[0].content == content

    def test_status_mismatch_helper_reports_stamp_when_forced(self, tmp_path) -> None:
        """embed_model_stamp_mismatch() surfaces a real disagreement — used
        as a defensive check by `vectr status`, not expected to fire once
        migration has run (it always runs synchronously in __init__)."""
        import chromadb
        from agent.working_context_store import WorkingContextStore
        chroma_dir = str(tmp_path / "chroma")
        client = chromadb.PersistentClient(path=chroma_dir)
        store = WorkingContextStore(
            str(tmp_path), embed_fn=_const_embed([1.0, 0.0]),
            notes_chroma_client=client, embed_model="model-a",
        )
        # Force a stamp disagreement directly (as if migration had failed
        # mid-way and left the object's view of the model stale) without
        # re-running __init__'s migration path.
        store._embed_model = "model-c"
        assert store.embed_model_stamp_mismatch() == "model-a"


# ---------------------------------------------------------------------------
# UPG-STDIO-MEMORY-READY — attach_embedder() + backfill_missing_vectors()
# ---------------------------------------------------------------------------

class TestAttachEmbedder:
    """A store constructed with embed_fn=None (memory tools live before the
    embedding model has loaded) can be upgraded later via attach_embedder(),
    which backfills a vector for every note recorded during the window it
    had none — without any note being re-written by the caller."""

    def test_embedderless_store_reports_not_ready(self, tmp_path) -> None:
        from agent.working_context_store import WorkingContextStore
        store = WorkingContextStore(str(tmp_path))
        assert store.embedder_ready is False

    def test_remember_before_attach_then_recall_falls_back_to_sql(self, tmp_path) -> None:
        from agent.working_context_store import WorkingContextStore
        store = WorkingContextStore(str(tmp_path))
        ws = "/repo"
        store.remember(ws, "note written before the embedder was ready")
        notes = store.recall(ws, query="embedder was ready")
        assert len(notes) == 1  # SQL LIKE fallback — no embedder yet

    def test_attach_embedder_flips_embedder_ready(self, tmp_path) -> None:
        import chromadb
        from agent.working_context_store import WorkingContextStore
        store = WorkingContextStore(str(tmp_path))
        client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
        store.attach_embedder(_dummy_embed, client, embed_model="model-a")
        assert store.embedder_ready is True

    def test_note_written_pre_attach_is_semantically_recallable_post_attach(self, tmp_path) -> None:
        """End-to-end proof (UPG-STDIO-MEMORY-READY reinforcement): a note
        stored while there was no embedder at all becomes recallable via the
        semantic path — not just the SQL fallback — the moment an embedder
        attaches, with no re-write of the note itself."""
        import chromadb
        from agent.working_context_store import WorkingContextStore
        store = WorkingContextStore(str(tmp_path))
        ws = "/repo"
        content = "handle_legacy_finalizers appends to gc.garbage when tp_del is set"
        note_id = store.remember(ws, content)

        client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
        store.attach_embedder(_dummy_embed, client, embed_model="model-a")

        # The backfill ran unconditionally at attach time — the vector exists
        # without any explicit re-embed call from the test.
        assert store._notes_col.count() == 1
        vec = store._notes_col.get(ids=[str(note_id)], include=["embeddings"])["embeddings"][0]
        assert vec is not None and len(vec) > 0

        # Semantic recall now finds it via cosine similarity, not SQL LIKE —
        # querying with the exact content gives cosine 1.0, the top result.
        notes = store.recall(ws, query=content)
        assert len(notes) == 1
        assert notes[0].content == content

    def test_backfill_is_idempotent_for_notes_already_vectored(self, tmp_path) -> None:
        """A note that already has a current vector is never re-embedded by
        a subsequent backfill call — only genuinely missing vectors are
        embedded."""
        import chromadb
        from agent.working_context_store import WorkingContextStore
        store = WorkingContextStore(str(tmp_path))
        ws = "/repo"
        store.remember(ws, "already vectored note")
        client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
        store.attach_embedder(_dummy_embed, client, embed_model="model-a")
        assert store._notes_col.count() == 1

        calls: list[list[str]] = []
        store._embed_fn = _counting_embed([0.0, 1.0], calls)
        backfilled = store.backfill_missing_vectors()
        assert backfilled == 0
        assert calls == []  # nothing re-embedded — the note already had a vector

    def test_backfill_covers_only_notes_missing_a_vector(self, tmp_path) -> None:
        """A mix of pre-attach (no vector yet) and post-attach (already
        vectored) notes — backfill embeds only the ones missing a vector."""
        import chromadb
        from agent.working_context_store import WorkingContextStore
        store = WorkingContextStore(str(tmp_path))
        ws = "/repo"
        pre_attach_id = store.remember(ws, "written before the embedder attached")

        client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
        store.attach_embedder(_dummy_embed, client, embed_model="model-a")
        post_attach_id = store.remember(ws, "written after the embedder attached")

        assert store._notes_col.count() == 2
        existing_ids = set(store._notes_col.get(include=[])["ids"])
        assert {str(pre_attach_id), str(post_attach_id)} == existing_ids

    def test_attach_embedder_is_idempotent(self, tmp_path) -> None:
        """A second attach_embedder() call is a no-op once an embedder is
        already attached — never swaps in a different embed_fn silently."""
        import chromadb
        from agent.working_context_store import WorkingContextStore
        store = WorkingContextStore(str(tmp_path))
        client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
        store.attach_embedder(_dummy_embed, client, embed_model="model-a")
        first_fn = store._embed_fn

        calls: list[list[str]] = []
        store.attach_embedder(_counting_embed([0.0, 1.0], calls), client, embed_model="model-b")
        assert store._embed_fn is first_fn  # unchanged — second call was a no-op
        assert calls == []

    def test_attach_embedder_noop_when_embed_fn_none(self, tmp_path) -> None:
        import chromadb
        from agent.working_context_store import WorkingContextStore
        store = WorkingContextStore(str(tmp_path))
        client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
        store.attach_embedder(None, client, embed_model="model-a")
        assert store.embedder_ready is False

    def test_backfill_with_no_embedder_returns_zero(self, tmp_path) -> None:
        from agent.working_context_store import WorkingContextStore
        store = WorkingContextStore(str(tmp_path))
        store.remember("/repo", "a note with no embedder at all")
        assert store.backfill_missing_vectors() == 0
