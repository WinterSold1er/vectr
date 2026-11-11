"""
Tests for UPG-DECISION-TIMELINE:
  (1) kind="decision" accepted throughout the note-kind surface (store, MCP
      schema, REST, CLI); unknown kinds are still rejected/fall back exactly
      as before this kind was added.
  (2) sort_by="chronological" — oldest-first ordering. This is a general
      lever on the existing sort machinery, not a decision-only special
      case: it composes with any kind filter (or none at all).
  (3) Chronological index-tier lines render the note's creation date instead
      of a relative age; every other sort_by renders exactly as before.
  (4) kind="decision" is NOT boot-privileged — absent from boot_recall() and
      from fire_and_format()'s session-start delivery.

All tests are deterministic; no network/model required (the one semantic-
path test uses a hash-based dummy embedder + a local ChromaDB
PersistentClient, mirroring TestSemanticRecall in test_memory.py).
"""
from __future__ import annotations

import datetime as dt
import sqlite3
import time

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _store(tmp_path):
    from agent.working_context_store import WorkingContextStore
    return WorkingContextStore(str(tmp_path))


def _dummy_embed(texts: list[str]) -> list[list[float]]:
    """Hash-based deterministic embedder for tests — same input -> same vector."""
    import hashlib
    result = []
    for t in texts:
        h = hashlib.md5(t.encode()).digest()
        vec = [(b / 255.0 - 0.5) for b in (h * 48)]  # 16 * 48 = 768 dims
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        result.append([x / norm for x in vec])
    return result


def _semantic_store(tmp_path):
    """A WorkingContextStore wired up with a dummy embedder + isolated ChromaDB."""
    import chromadb
    from agent.working_context_store import WorkingContextStore
    chroma_dir = str(tmp_path / "chroma")
    client = chromadb.PersistentClient(path=chroma_dir)
    return WorkingContextStore(str(tmp_path), embed_fn=_dummy_embed, notes_chroma_client=client)


def _set_created_at(store, note_id: int, created_at: float) -> None:
    with sqlite3.connect(str(store._db_path)) as conn:
        conn.execute("UPDATE notes SET created_at = ? WHERE note_id = ?", (created_at, note_id))


# ---------------------------------------------------------------------------
# (1a) kind="decision" — store level
# ---------------------------------------------------------------------------

class TestDecisionKindStore:
    def test_decision_kind_accepted_and_stored(self, tmp_path) -> None:
        store = _store(tmp_path)
        nid = store.remember(
            "/ws",
            "Chose SQLite over Postgres for the working-context store: "
            "zero-ops, embeds cleanly in a single-process laptop daemon.",
            kind="decision",
        )
        note = store.get_note("/ws", nid)
        assert note is not None
        assert note.kind == "decision"

    def test_recall_filters_by_kind_decision(self, tmp_path) -> None:
        store = _store(tmp_path)
        store.remember("/ws", "decision: use SQLite for the note store", kind="decision")
        store.remember("/ws", "just a plain finding")
        decisions = store.recall("/ws", kind="decision")
        assert len(decisions) == 1
        assert decisions[0].kind == "decision"

    def test_unknown_kind_still_falls_back_to_finding(self, tmp_path) -> None:
        """Regression: adding 'decision' to VALID_KINDS must not loosen the
        unrecognised-kind fallback for every OTHER unknown value."""
        store = _store(tmp_path)
        store.remember("/ws", "x", kind="not-a-real-kind")
        assert store.recall("/ws")[0].kind == "finding"


# ---------------------------------------------------------------------------
# (1b) kind="decision" / sort_by="chronological" — MCP schema level
# ---------------------------------------------------------------------------

class TestDecisionKindMCPSchema:
    def _tool(self, name: str) -> dict:
        from integrations.mcp_server._schemas import MCP_TOOLS
        matches = [t for t in MCP_TOOLS if t["name"] == name]
        assert len(matches) == 1
        return matches[0]

    def test_remember_kind_enum_includes_decision(self) -> None:
        props = self._tool("vectr_remember")["inputSchema"]["properties"]
        assert "decision" in props["kind"]["enum"]

    def test_recall_kind_enum_includes_decision(self) -> None:
        props = self._tool("vectr_recall")["inputSchema"]["properties"]
        assert "decision" in props["kind"]["enum"]

    def test_recall_sort_by_enum_includes_chronological(self) -> None:
        props = self._tool("vectr_recall")["inputSchema"]["properties"]
        assert "chronological" in props["sort_by"]["enum"]
        # The other three existing values must survive the addition untouched.
        assert set(props["sort_by"]["enum"]) == {
            "relevance", "recency", "priority", "chronological",
        }


# ---------------------------------------------------------------------------
# (1c) kind="decision" — REST level (client_real_memory: REAL store, mocked search)
# ---------------------------------------------------------------------------

class TestDecisionKindREST:
    def test_remember_accepts_kind_decision(self, client_real_memory) -> None:
        resp = client_real_memory.post(
            "/v1/remember",
            json={"content": "Chose X over Y because of Z", "kind": "decision"},
        )
        assert resp.status_code == 200

    def test_recall_kind_filter_decision_via_rest(self, client_real_memory) -> None:
        client = client_real_memory
        client.post("/v1/remember", json={"content": "decided to use SQLite", "kind": "decision"})
        client.post("/v1/remember", json={"content": "just a plain finding"})

        decisions = client.post("/v1/recall", json={"kind": "decision", "detail": "full"}).json()["notes"]
        assert "decided to use SQLite" in decisions
        assert "just a plain finding" not in decisions

    def test_invalid_kind_still_rejected_via_rest(self, client_real_memory) -> None:
        """Regression: adding 'decision' must not widen kind validation generally."""
        resp = client_real_memory.post("/v1/remember", json={"content": "x", "kind": "bogus"})
        assert resp.status_code == 422

    def test_recall_chronological_sort_via_rest(self, client_real_memory) -> None:
        client = client_real_memory
        r1 = client.post("/v1/remember", json={"content": "decided to use SQLite", "kind": "decision"})
        r2 = client.post("/v1/remember", json={"content": "decided to add a trigger engine", "kind": "decision"})
        id_first, id_second = r1.json()["note_id"], r2.json()["note_id"]

        text = client.post(
            "/v1/recall", json={"kind": "decision", "sort_by": "chronological"}
        ).json()["notes"]
        assert text.index(f"[#{id_first}]") < text.index(f"[#{id_second}]")


# ---------------------------------------------------------------------------
# (1d) kind="decision" / sort_by="chronological" — CLI level
#
# The choices=[...] lists are only enforced by real argparse parsing
# (parser.parse_args() inside main.main()); constructing an argparse.Namespace
# directly and calling cmd_remember/cmd_recall (the style used elsewhere in
# test_main.py) bypasses that enforcement entirely, so these tests drive the
# CLI through main.main() with sys.argv patched, matching
# TestStartCommand.test_start_help_discloses_ide_config_writes's pattern.
# ---------------------------------------------------------------------------

class TestDecisionKindCLI:
    def test_remember_accepts_kind_decision_choice(self, tmp_path) -> None:
        import main as m
        from unittest.mock import patch, MagicMock

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"note_id": 1, "message": "Stored note #1.", "processing_ms": 1}
        mock_resp.raise_for_status.return_value = None

        with patch("main.InstanceRegistry") as MockReg, \
             patch("main._check_version_skew"), \
             patch("httpx.post", return_value=mock_resp) as mock_post, \
             patch("sys.argv", [
                 "vectr", "remember", "chose SQLite over Postgres",
                 "--kind", "decision", "--path", str(tmp_path), "--port", "8765",
             ]):
            MockReg.return_value.get.return_value = None
            m.main()

        payload = mock_post.call_args[1]["json"]
        assert payload["kind"] == "decision"

    def test_remember_rejects_unknown_kind_choice(self, tmp_path) -> None:
        import main as m
        from unittest.mock import patch

        with patch("sys.argv", [
            "vectr", "remember", "x", "--kind", "bogus",
            "--path", str(tmp_path), "--port", "8765",
        ]):
            with pytest.raises(SystemExit) as exc:
                m.main()
        assert exc.value.code == 2

    def test_recall_accepts_kind_decision_and_sort_by_chronological_choices(self, tmp_path) -> None:
        import main as m
        from unittest.mock import patch, MagicMock

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"notes": "", "processing_ms": 1}
        mock_resp.raise_for_status.return_value = None

        with patch("main.InstanceRegistry") as MockReg, \
             patch("main._check_version_skew"), \
             patch("httpx.post", return_value=mock_resp) as mock_post, \
             patch("sys.argv", [
                 "vectr", "recall", "--kind", "decision", "--sort-by", "chronological",
                 "--path", str(tmp_path), "--port", "8765",
             ]):
            MockReg.return_value.get.return_value = None
            m.main()

        payload = mock_post.call_args[1]["json"]
        assert payload["kind"] == "decision"
        assert payload["sort_by"] == "chronological"

    def test_recall_rejects_unknown_sort_by_choice(self, tmp_path) -> None:
        import main as m
        from unittest.mock import patch

        with patch("sys.argv", [
            "vectr", "recall", "--sort-by", "bogus",
            "--path", str(tmp_path), "--port", "8765",
        ]):
            with pytest.raises(SystemExit) as exc:
                m.main()
        assert exc.value.code == 2


# ---------------------------------------------------------------------------
# (2) sort_by="chronological" — a general lever, not decision-only
# ---------------------------------------------------------------------------

class TestChronologicalSort:
    def _store_three_decisions(self, store):
        ws = "/ws"
        id_first = store.remember(ws, "decided to use SQLite for the note store", kind="decision")
        id_second = store.remember(ws, "decided to add the trigger engine", kind="decision")
        id_third = store.remember(ws, "decided to add the decision timeline", kind="decision")
        now = time.time()
        _set_created_at(store, id_first, now - 300)
        _set_created_at(store, id_second, now - 200)
        _set_created_at(store, id_third, now - 100)
        return ws, id_first, id_second, id_third

    def test_chronological_ascending_oldest_first(self, tmp_path) -> None:
        store = _store(tmp_path)
        ws, id_first, id_second, id_third = self._store_three_decisions(store)
        notes = store.recall(ws, kind="decision", sort_by="chronological")
        assert [n.note_id for n in notes] == [id_first, id_second, id_third]

    def test_chronological_composes_with_a_different_kind_filter(self, tmp_path) -> None:
        """Not a decision-only special case — the same sort_by value orders
        kind='finding' (or any other kind) just as well."""
        store = _store(tmp_path)
        ws = "/ws"
        id_old = store.remember(ws, "old finding", kind="finding")
        id_new = store.remember(ws, "new finding", kind="finding")
        now = time.time()
        _set_created_at(store, id_old, now - 500)
        _set_created_at(store, id_new, now - 100)
        notes = store.recall(ws, kind="finding", sort_by="chronological")
        assert [n.note_id for n in notes] == [id_old, id_new]

    def test_chronological_composes_with_no_kind_filter(self, tmp_path) -> None:
        """sort_by='chronological' with no kind filter at all — a general
        sort lever, never gated on kind content."""
        store = _store(tmp_path)
        ws, id_first, id_second, id_third = self._store_three_decisions(store)
        notes = store.recall(ws, sort_by="chronological")
        ids = [n.note_id for n in notes]
        assert ids.index(id_first) < ids.index(id_second) < ids.index(id_third)

    def test_chronological_tie_break_by_note_id(self, tmp_path) -> None:
        store = _store(tmp_path)
        ws = "/ws"
        id_a = store.remember(ws, "decision A", kind="decision")
        id_b = store.remember(ws, "decision B", kind="decision")
        now = time.time()
        _set_created_at(store, id_a, now)
        _set_created_at(store, id_b, now)
        notes = store.recall(ws, kind="decision", sort_by="chronological")
        assert [n.note_id for n in notes] == [id_a, id_b]

    def test_semantic_path_also_resorts_chronological(self, tmp_path) -> None:
        """The re-sort must apply on BOTH the SQL-only recall path and the
        query-driven semantic-recall path (_semantic_recall)."""
        store = _semantic_store(tmp_path)
        ws = "/ws"
        id_first = store.remember(ws, "decided to use SQLite for working memory", kind="decision")
        id_second = store.remember(ws, "decided to add a trigger engine for memory", kind="decision")
        now = time.time()
        _set_created_at(store, id_first, now - 300)
        _set_created_at(store, id_second, now - 100)
        notes = store.recall(ws, query="memory decision", kind="decision", sort_by="chronological")
        assert [n.note_id for n in notes] == [id_first, id_second]

    def test_other_sorts_unaffected_by_chronological_addition(self, tmp_path) -> None:
        """Regression: recency/priority/relevance keep their exact pre-existing
        behaviour after chronological is added as a fourth, independent value."""
        store = _store(tmp_path)
        ws = "/ws"
        id_low = store.remember(ws, "low", priority="low")
        id_high = store.remember(ws, "high", priority="high")

        recency_notes = store.recall(ws, sort_by="recency")
        assert recency_notes[0].note_id == id_high  # id_high stored most recently

        priority_notes = store.recall(ws, sort_by="priority")
        assert priority_notes[0].priority == "high"

        relevance_notes = store.recall(ws)  # default sort_by="relevance"
        assert len(relevance_notes) == 2


# ---------------------------------------------------------------------------
# (2b) UPG-RECALL-SEMANTIC-POOL-SORT-TRUNCATION: an explicit non-relevance
# sort must not be silently bounded by the semantic candidate pool
# ---------------------------------------------------------------------------

_POOL_MARKER = "zzq-off-topic-marker-banana-stand"


def _structured_embed(texts: list[str]) -> list[list[float]]:
    """Two-direction embedder for pool-truncation tests: text containing
    _POOL_MARKER maps to vector A, everything else to ORTHOGONAL vector B.
    Unlike a hash embedder this GUARANTEES the marked note ranks strictly
    last for an unmarked query, so its exclusion from the limit*3 pool is
    deterministic rather than luck-of-the-hash."""
    vecs = []
    for t in texts:
        if _POOL_MARKER in t:
            v = [1.0] + [0.0] * 15
        else:
            v = [0.0, 1.0] + [0.0] * 14
        vecs.append(v)
    return vecs


class TestSemanticPoolSortTruncation:
    """With more matching notes than limit*3, re-sorting only the semantic
    candidate pool silently dropped the true oldest/highest-priority note
    (it never made the pool). An explicit non-relevance sort now also
    prefetches the true top-limit rows by the requested key via the same
    WHERE clause."""

    def _store(self, tmp_path):
        import chromadb
        from agent.working_context_store import WorkingContextStore
        client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
        return WorkingContextStore(
            str(tmp_path), embed_fn=_structured_embed, notes_chroma_client=client,
        )

    def _seed(self, store):
        """7 decisions with identical on-topic embeddings plus the TRUE
        OLDEST decision whose marker content embeds orthogonally — it can
        only miss the semantic pool."""
        ws = "/ws"
        ids = [
            store.remember(ws, f"decision {i}: use blue-green deploys", kind="decision")
            for i in range(7)
        ]
        target = store.remember(
            ws, f"decision about {_POOL_MARKER} and unrelated subsystems", kind="decision",
        )
        now = time.time()
        for i, nid in enumerate(ids):
            _set_created_at(store, nid, now - 100 + i)
        _set_created_at(store, target, now - 10_000)  # far oldest overall
        return ws, target

    def test_chronological_surfaces_the_true_oldest_despite_pool_miss(self, tmp_path) -> None:
        store = self._store(tmp_path)
        ws, target = self._seed(store)
        # limit=2 -> pool holds min(6, 8) = 6 of the 7 on-topic notes; the
        # marked note (orthogonal to the query) never ranks into it. The
        # prefetch must still deliver the true oldest note FIRST.
        notes = store.recall(
            ws, query="deployment decisions", kind="decision",
            sort_by="chronological", limit=2,
        )
        assert len(notes) == 2
        assert notes[0].note_id == target

    def test_relevance_default_still_excludes_the_off_pool_note(self, tmp_path) -> None:
        """Control pinning the default path UNCHANGED: relevance keeps its
        exact pre-existing semantics — the fix must not leak prefetched rows
        into it."""
        store = self._store(tmp_path)
        ws, target = self._seed(store)
        notes = store.recall(
            ws, query="deployment decisions", kind="decision",
            sort_by="relevance", limit=2,
        )
        assert all(n.note_id != target for n in notes)

    def test_prefetch_reuses_the_metadata_filters(self, tmp_path) -> None:
        """The prefetch runs the same METADATA filters as the candidate fetch
        (it drops only the semantic `note_id IN (...)` restriction), so a
        non-matching kind can enter neither via the semantic pool nor via the
        sort-aware prefetch.

        Both conditions here are load-bearing. The stray is seeded OLDER than
        the target, so a kind filter that failed to reach the prefetch would
        put it first rather than merely somewhere in the tail. And limit=2
        keeps the semantic pool (6) smaller than the filtered corpus (9), so
        the prefetch is the only path that can deliver the target at all — at
        a limit whose pool covers every note, this assertion would hold with
        the prefetch deleted entirely."""
        store = self._store(tmp_path)
        ws, target = self._seed(store)
        stray = store.remember(ws, f"finding about {_POOL_MARKER}", kind="finding")
        _set_created_at(store, stray, time.time() - 20_000)  # older than target
        notes = store.recall(
            ws, query="deployment decisions", kind="decision",
            sort_by="chronological", limit=2,
        )
        assert len(notes) == 2
        assert notes[0].note_id == target
        assert all(n.note_id != stray for n in notes)
        assert all(n.kind == "decision" for n in notes)
        assert all(n.note_id != stray for n in notes)


# ---------------------------------------------------------------------------
# (3) Chronological rendering: date instead of relative age
# ---------------------------------------------------------------------------

class TestChronologicalRendering:
    def test_chronological_index_line_shows_date_not_age(self, tmp_path) -> None:
        store = _store(tmp_path)
        ws = "/ws"
        nid = store.remember(ws, "decided to use SQLite", kind="decision", title="use sqlite")
        backdated = time.time() - 10 * 86400
        _set_created_at(store, nid, backdated)

        notes = store.recall(ws, kind="decision", sort_by="chronological")
        text = store.format_notes_for_llm(notes, detail="index", sort_by="chronological")

        expected_date = dt.datetime.fromtimestamp(backdated).strftime("%Y-%m-%d")
        assert expected_date in text
        assert f"[#{nid}] {expected_date} decision/medium" in text
        assert "(10d)" not in text  # no relative-age suffix when chronological

    def test_non_chronological_sorts_keep_relative_age_rendering(self, tmp_path) -> None:
        """Regression: the default (and every other) sort_by's index rendering
        is unchanged by this feature — only sort_by='chronological' changes it."""
        store = _store(tmp_path)
        ws = "/ws"
        store.remember(ws, "a plain finding", title="plain")
        notes = store.recall(ws)

        text_default = store.format_notes_for_llm(notes, detail="index")
        text_explicit_relevance = store.format_notes_for_llm(notes, detail="index", sort_by="relevance")
        assert text_default == text_explicit_relevance
        assert "(0s)" in text_default  # brand-new note still renders relative age

    def test_chronological_multiple_decisions_render_in_order_with_dates(self, tmp_path) -> None:
        store = _store(tmp_path)
        ws = "/ws"
        id_a = store.remember(ws, "decision A", kind="decision", title="decision A")
        id_b = store.remember(ws, "decision B", kind="decision", title="decision B")
        now = time.time()
        _set_created_at(store, id_a, now - 20 * 86400)
        _set_created_at(store, id_b, now - 1 * 86400)

        notes = store.recall(ws, kind="decision", sort_by="chronological")
        text = store.format_notes_for_llm(notes, detail="index", sort_by="chronological")
        assert text.index(f"[#{id_a}]") < text.index(f"[#{id_b}]")  # oldest first — reads as a timeline

    def test_full_tier_rendering_unaffected_by_sort_by(self, tmp_path) -> None:
        """detail='full' bodies do not carry any sort_by-dependent rendering."""
        store = _store(tmp_path)
        ws = "/ws"
        body = "decided to use SQLite for the working-context store"
        store.remember(ws, body, kind="decision")
        notes = store.recall(ws, kind="decision", sort_by="chronological")
        text_chrono = store.format_notes_for_llm(notes, detail="full", sort_by="chronological")
        text_default = store.format_notes_for_llm(notes, detail="full")
        assert text_chrono == text_default
        assert body in text_chrono


# ---------------------------------------------------------------------------
# (4) kind="decision" is NOT boot-privileged
# ---------------------------------------------------------------------------

class TestDecisionNotBootPrivileged:
    def test_boot_recall_excludes_decision_notes(self, tmp_path) -> None:
        store = _store(tmp_path)
        ws = "/ws"
        store.remember(ws, "never push to main", kind="directive")
        store.remember(ws, "decided to use SQLite", kind="decision", priority="high")
        boot = store.boot_recall(ws)
        kinds = {n.kind for n in boot}
        assert "decision" not in kinds
        assert "directive" in kinds

    def test_default_trigger_bundle_for_decision_is_empty(self) -> None:
        """decision gets no wave-1 default bundle — the same evaluation-time
        bucket as finding/reference (UPG-DECISION-TIMELINE) — so it never
        auto-fires at session-start/post-compaction the way directive/
        high-priority task notes do."""
        from agent.trigger_engine import default_bundle_for_kind
        assert default_bundle_for_kind("decision", anchors=None, priority="high") == []

    def test_fire_and_format_does_not_inject_decision_at_session_start(self, tmp_path) -> None:
        store = _store(tmp_path)
        ws = "/ws"
        unique = "UNIQUE_DECISION_BODY_SQLITE_OVER_POSTGRES"
        store.remember(ws, unique, kind="decision", priority="high")
        text, fired_ids = store.fire_and_format(ws, event="session-start")
        assert fired_ids == set()
        assert unique not in text

    def test_rest_boot_recall_excludes_decision_notes(self, client_real_memory) -> None:
        client = client_real_memory
        client.post("/v1/remember", json={"content": "never push to main", "kind": "directive"})
        client.post("/v1/remember", json={"content": "decided to use SQLite", "kind": "decision", "priority": "high"})
        notes = client.post("/v1/recall", json={"boot": True}).json()["notes"]
        assert "never push to main" in notes
        assert "decided to use SQLite" not in notes
