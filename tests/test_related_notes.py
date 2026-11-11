"""Tests for agent/working_context_store/_related.py — write-time related-
notes lookup (WorkingContextStore.related_active_notes) and the sibling
write-time deterrent lookup (WorkingContextStore.revoked_related_notes,
UPG-RELATED-REVOKED-DETERRENT).

Every test builds a REAL WorkingContextStore backed by a real (temp-path)
ChromaDB collection and a deterministic controlled embed_fn that maps exact
note content to a hand-picked unit vector, so cosine similarity between any
two notes is known exactly rather than incidentally derived from a hash.
This is what makes the near-duplicate/unrelated assertions real rather than
tautological.
"""
from __future__ import annotations

import logging
import math

import chromadb
import pytest

from agent.working_context_store import WorkingContextStore
from agent.working_context_store._related import RelatedNote, RevokedRelatedNote


def _unit_vector_at_cosine(sim: float) -> list[float]:
    """A 4-d unit vector whose cosine similarity to (1, 0, 0, 0) is exactly
    `sim` (up to floating point)."""
    return [sim, math.sqrt(max(0.0, 1.0 - sim * sim)), 0.0, 0.0]


BASE_VEC = _unit_vector_at_cosine(1.0)          # (1, 0, 0, 0)
UNRELATED_VEC = [0.0, 1.0, 0.0, 0.0]             # orthogonal -> cosine sim 0.0


class _ControlledEmbedder:
    """Deterministic embed_fn: exact content string -> exact pre-registered
    unit vector. Raises KeyError for anything not explicitly registered, so
    a test can never accidentally rely on an un-controlled vector. Counts
    calls (not texts) so tests can assert zero added inference."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = dict(vectors)
        self.call_count = 0

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        return [self.vectors[t] for t in texts]


def _store(tmp_path, embedder: _ControlledEmbedder) -> WorkingContextStore:
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    return WorkingContextStore(str(tmp_path), embed_fn=embedder, notes_chroma_client=client)


WS = "/repo"


class TestRelatedActiveNotesRelevance:
    def test_returns_near_duplicate_not_unrelated(self, tmp_path) -> None:
        base = "WorkspaceLock.acquire takes a PID-scoped lock"
        near = "WorkspaceLock.acquire grabs a PID-scoped lock"
        unrelated = "the deploy pipeline retries on a 503"
        embedder = _ControlledEmbedder({
            base: BASE_VEC,
            near: _unit_vector_at_cosine(0.97),
            unrelated: UNRELATED_VEC,
        })
        store = _store(tmp_path, embedder)
        store.remember(WS, unrelated)
        near_id = store.remember(WS, near)
        base_id = store.remember(WS, base)

        related = store.related_active_notes(WS, base_id, limit=5, min_similarity=0.75)

        ids = [r["note_id"] for r in related]
        assert near_id in ids
        assert all(r["note_id"] != base_id for r in related)  # T7: self excluded
        # The unrelated note (cosine sim 0.0) must never appear — this is
        # the whole point of a controlled, non-hash-derived vector: the
        # assertion is checking a REAL exclusion, not a coincidence.
        assert len(related) == 1
        assert related[0]["note_id"] == near_id
        assert related[0]["similarity"] == pytest.approx(0.97, abs=1e-3)

    def test_related_note_fields(self, tmp_path) -> None:
        base = "content A"
        near = "content B"
        embedder = _ControlledEmbedder({base: BASE_VEC, near: _unit_vector_at_cosine(0.9)})
        store = _store(tmp_path, embedder)
        near_id = store.remember(WS, near, kind="gotcha", priority="high", title="B title")
        base_id = store.remember(WS, base)

        related = store.related_active_notes(WS, base_id, limit=5, min_similarity=0.5)

        assert len(related) == 1
        r: RelatedNote = related[0]
        assert r["note_id"] == near_id
        assert r["title"] == "B title"
        assert r["kind"] == "gotcha"
        assert r["priority"] == "high"
        assert isinstance(r["created_at"], float)


class TestRelatedActiveNotesLifecycle:
    def test_excludes_revoked_note(self, tmp_path) -> None:
        base = "base content"
        revoked = "near content that gets revoked"
        embedder = _ControlledEmbedder({base: BASE_VEC, revoked: _unit_vector_at_cosine(0.95)})
        store = _store(tmp_path, embedder)
        revoked_id = store.remember(WS, revoked)
        base_id = store.remember(WS, base)

        # Sanity: before revocation, the candidate is offered.
        before = store.related_active_notes(WS, base_id, limit=5, min_similarity=0.5)
        assert any(r["note_id"] == revoked_id for r in before)

        assert store.revoke_note(WS, revoked_id, reason="wrong") is True

        after = store.related_active_notes(WS, base_id, limit=5, min_similarity=0.5)
        assert all(r["note_id"] != revoked_id for r in after)

    def test_excludes_superseded_note(self, tmp_path) -> None:
        base = "base content 2"
        superseded = "near content that gets superseded"
        successor = "totally unrelated successor content"
        embedder = _ControlledEmbedder({
            base: BASE_VEC,
            superseded: _unit_vector_at_cosine(0.95),
            successor: UNRELATED_VEC,
        })
        store = _store(tmp_path, embedder)
        superseded_id = store.remember(WS, superseded)
        base_id = store.remember(WS, base)

        before = store.related_active_notes(WS, base_id, limit=5, min_similarity=0.5)
        assert any(r["note_id"] == superseded_id for r in before)

        store.remember(WS, successor, supersedes=superseded_id)

        after = store.related_active_notes(WS, base_id, limit=5, min_similarity=0.5)
        assert all(r["note_id"] != superseded_id for r in after)


class TestRelatedActiveNotesFloorAndLimit:
    def _three_candidate_store(self, tmp_path):
        base = "base content 3"
        near_a = "near a"   # sim 0.90
        near_b = "near b"   # sim 0.85
        near_c = "near c"   # sim 0.80
        embedder = _ControlledEmbedder({
            base: BASE_VEC,
            near_a: _unit_vector_at_cosine(0.90),
            near_b: _unit_vector_at_cosine(0.85),
            near_c: _unit_vector_at_cosine(0.80),
        })
        store = _store(tmp_path, embedder)
        id_a = store.remember(WS, near_a)
        id_b = store.remember(WS, near_b)
        id_c = store.remember(WS, near_c)
        base_id = store.remember(WS, base)
        return store, base_id, id_a, id_b, id_c

    def test_raising_the_floor_excludes_everything(self, tmp_path) -> None:
        store, base_id, id_a, id_b, id_c = self._three_candidate_store(tmp_path)

        # 0.75 floor: all three (0.90/0.85/0.80) qualify.
        loose = store.related_active_notes(WS, base_id, limit=10, min_similarity=0.75)
        assert {r["note_id"] for r in loose} == {id_a, id_b, id_c}

        # 0.95 floor: the closest candidate is 0.90 — none qualify.
        strict = store.related_active_notes(WS, base_id, limit=10, min_similarity=0.95)
        assert strict == []

    def test_limit_caps_the_returned_count_highest_similarity_first(self, tmp_path) -> None:
        store, base_id, id_a, id_b, id_c = self._three_candidate_store(tmp_path)

        capped = store.related_active_notes(WS, base_id, limit=2, min_similarity=0.75)
        assert len(capped) == 2
        assert [r["note_id"] for r in capped] == [id_a, id_b]  # 0.90 then 0.85

    def test_limit_zero_or_negative_returns_empty(self, tmp_path) -> None:
        store, base_id, *_ = self._three_candidate_store(tmp_path)
        assert store.related_active_notes(WS, base_id, limit=0, min_similarity=0.5) == []
        assert store.related_active_notes(WS, base_id, limit=-1, min_similarity=0.5) == []


class TestRelatedActiveNotesFailSafe:
    def test_no_embedder_returns_empty_list(self, tmp_path) -> None:
        store = WorkingContextStore(str(tmp_path))  # embed_fn=None, no chroma client
        note_id = store.remember(WS, "a plain note, no embedder attached")
        assert store.related_active_notes(WS, note_id, limit=5, min_similarity=0.5) == []

    def test_collection_query_raising_returns_empty_not_exception(self, tmp_path) -> None:
        embedder = _ControlledEmbedder({"solo note": BASE_VEC})
        store = _store(tmp_path, embedder)
        note_id = store.remember(WS, "solo note")

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated chroma failure")

        store._notes_col.query = _boom  # simulate a real backend failure, not a return-shape mock

        assert store.related_active_notes(WS, note_id, limit=5, min_similarity=0.5) == []

    def test_swallowed_failure_is_logged_at_debug(self, tmp_path, caplog) -> None:
        """The fail-safe returns [] on any error, which is ALSO the correct
        answer for most writes — so a permanently broken lookup would stay
        invisible and every other test would still pass. The debug log is
        the only thing that distinguishes 'nothing was near' from 'this is
        broken', so assert it actually fires."""
        embedder = _ControlledEmbedder({"solo note 2": BASE_VEC})
        store = _store(tmp_path, embedder)
        note_id = store.remember(WS, "solo note 2")

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated chroma failure")

        store._notes_col.query = _boom

        with caplog.at_level(logging.DEBUG, logger="agent.working_context_store._related"):
            assert store.related_active_notes(WS, note_id, limit=5, min_similarity=0.5) == []

        assert any(
            "related_active_notes failed" in r.message and r.exc_info is not None
            for r in caplog.records
        )


class TestRelatedActiveNotesZeroInference:
    def test_no_embed_calls_when_vector_already_in_collection(self, tmp_path) -> None:
        base = "zero inference base"
        near = "zero inference near"
        embedder = _ControlledEmbedder({base: BASE_VEC, near: _unit_vector_at_cosine(0.9)})
        store = _store(tmp_path, embedder)
        store.remember(WS, near)
        base_id = store.remember(WS, base)

        embedder.call_count = 0  # both writes above are done; reset before the call under test
        store.related_active_notes(WS, base_id, limit=5, min_similarity=0.5)

        assert embedder.call_count == 0

    def test_fallback_embeds_exactly_once_when_vector_missing_from_collection(self, tmp_path) -> None:
        """Documented fallback path (not the normal case): if a note exists
        in SQLite but its vector isn't in the collection, the function
        embeds its content ONCE rather than returning nothing."""
        base = "fallback base content"
        near = "fallback near content"
        embedder = _ControlledEmbedder({base: BASE_VEC, near: _unit_vector_at_cosine(0.9)})
        store = _store(tmp_path, embedder)
        near_id = store.remember(WS, near)
        base_id = store.remember(WS, base)

        # Simulate a note whose vector never made it into the collection.
        store._notes_col.delete(ids=[str(base_id)])

        embedder.call_count = 0
        related = store.related_active_notes(WS, base_id, limit=5, min_similarity=0.5)

        assert embedder.call_count == 1
        assert any(r["note_id"] == near_id for r in related)


class TestRelatedActiveNotesOverFetch:
    """The "working_memory" ChromaDB collection is GLOBAL — one collection
    shared by every workspace a daemon serves — and the vector query carries
    no per-workspace `where` clause, so workspace / expiry / lifecycle
    filtering all happens in SQL AFTER the search. The lookup must therefore
    over-fetch candidates rather than requesting exactly `limit + 1`, or
    nearer-but-filtered-out neighbours silently starve the result to [].

    Both tests below FAIL against an `n_results=limit + 1` implementation
    and pass with the `limit * 3` over-fetch, so they pin the behaviour
    rather than restating it.
    """

    def test_foreign_workspace_neighbours_do_not_starve_the_result(self, tmp_path) -> None:
        base = "base content ofw"
        other_a = "other workspace note a"
        other_b = "other workspace note b"
        near_a = "same workspace near note a"
        near_b = "same workspace near note b"
        embedder = _ControlledEmbedder({
            base: BASE_VEC,
            # The two NEAREST neighbours belong to a different workspace and
            # are dropped by the SQL workspace filter after the search.
            other_a: _unit_vector_at_cosine(0.99),
            other_b: _unit_vector_at_cosine(0.98),
            near_a: _unit_vector_at_cosine(0.95),
            near_b: _unit_vector_at_cosine(0.94),
        })
        store = _store(tmp_path, embedder)
        store.remember("/some/other/workspace", other_a)
        store.remember("/some/other/workspace", other_b)
        near_a_id = store.remember(WS, near_a)
        near_b_id = store.remember(WS, near_b)
        base_id = store.remember(WS, base)

        related = store.related_active_notes(WS, base_id, limit=2, min_similarity=0.5)

        assert [r["note_id"] for r in related] == [near_a_id, near_b_id]

    def test_revoked_neighbours_do_not_starve_the_result(self, tmp_path) -> None:
        base = "base content rev"
        revoked_a = "revoked near note a"
        revoked_b = "revoked near note b"
        active_a = "active near note a"
        active_b = "active near note b"
        embedder = _ControlledEmbedder({
            base: BASE_VEC,
            # The two NEAREST neighbours are revoked and are dropped by the
            # folded-lifecycle-state filter after the search.
            revoked_a: _unit_vector_at_cosine(0.99),
            revoked_b: _unit_vector_at_cosine(0.98),
            active_a: _unit_vector_at_cosine(0.95),
            active_b: _unit_vector_at_cosine(0.94),
        })
        store = _store(tmp_path, embedder)
        revoked_a_id = store.remember(WS, revoked_a)
        revoked_b_id = store.remember(WS, revoked_b)
        active_a_id = store.remember(WS, active_a)
        active_b_id = store.remember(WS, active_b)
        base_id = store.remember(WS, base)

        assert store.revoke_note(WS, revoked_a_id, reason="wrong") is True
        assert store.revoke_note(WS, revoked_b_id, reason="wrong") is True

        related = store.related_active_notes(WS, base_id, limit=2, min_similarity=0.5)

        assert [r["note_id"] for r in related] == [active_a_id, active_b_id]


class TestRevokedRelatedNotesRelevance:
    def test_returns_revoked_near_duplicate_not_unrelated(self, tmp_path) -> None:
        base = "WorkspaceLock.acquire takes a PID-scoped lock"
        near = "WorkspaceLock.acquire grabs a PID-scoped lock"
        unrelated = "the deploy pipeline retries on a 503"
        embedder = _ControlledEmbedder({
            base: BASE_VEC,
            near: _unit_vector_at_cosine(0.97),
            unrelated: UNRELATED_VEC,
        })
        store = _store(tmp_path, embedder)
        store.remember(WS, unrelated)
        near_id = store.remember(WS, near)
        base_id = store.remember(WS, base)
        assert store.revoke_note(WS, near_id, reason="was wrong") is True

        revoked = store.revoked_related_notes(WS, base_id, limit=5, min_similarity=0.75)

        assert len(revoked) == 1
        assert revoked[0]["note_id"] == near_id
        assert revoked[0]["similarity"] == pytest.approx(0.97, abs=1e-3)

    def test_revoked_related_note_fields(self, tmp_path) -> None:
        base = "content A rev"
        near = "content B rev"
        embedder = _ControlledEmbedder({base: BASE_VEC, near: _unit_vector_at_cosine(0.9)})
        store = _store(tmp_path, embedder)
        near_id = store.remember(WS, near, title="B title rev")
        base_id = store.remember(WS, base)
        assert store.revoke_note(WS, near_id, reason="superseded by a corrected finding") is True

        revoked = store.revoked_related_notes(WS, base_id, limit=5, min_similarity=0.5)

        assert len(revoked) == 1
        r: RevokedRelatedNote = revoked[0]
        assert r["note_id"] == near_id
        assert r["title"] == "B title rev"
        assert r["reason"] == "superseded by a corrected finding"
        assert isinstance(r["revoked_date"], str) and r["revoked_date"]

    def test_reason_defaults_to_a_readable_string_when_none_given(self, tmp_path) -> None:
        """`revoke_note()` itself requires a `reason` string, so this only
        guards the display fallback if a future caller (or a pre-migration
        row) ever leaves it empty — the deterrent must never render an
        empty/None reason to the caller LLM."""
        base = "content A rev empty reason"
        near = "content B rev empty reason"
        embedder = _ControlledEmbedder({base: BASE_VEC, near: _unit_vector_at_cosine(0.9)})
        store = _store(tmp_path, embedder)
        near_id = store.remember(WS, near)
        base_id = store.remember(WS, base)
        assert store.revoke_note(WS, near_id, reason="") is True

        revoked = store.revoked_related_notes(WS, base_id, limit=5, min_similarity=0.5)

        assert revoked[0]["reason"] == "no reason given"


class TestRevokedRelatedNotesLifecycle:
    def test_excludes_active_note(self, tmp_path) -> None:
        """The mirror image of `related_active_notes`'s own contract: a
        near-duplicate that is still ACTIVE must never appear here — that
        is exactly what `related_active_notes` is for. The two lookups
        answer different questions and must never overlap on the same
        candidate."""
        base = "base content rev active"
        active = "near content that stays active"
        embedder = _ControlledEmbedder({base: BASE_VEC, active: _unit_vector_at_cosine(0.95)})
        store = _store(tmp_path, embedder)
        active_id = store.remember(WS, active)
        base_id = store.remember(WS, base)

        revoked = store.revoked_related_notes(WS, base_id, limit=5, min_similarity=0.5)

        assert all(r["note_id"] != active_id for r in revoked)
        assert revoked == []

    def test_excludes_superseded_note(self, tmp_path) -> None:
        """A superseded note is a distinct lifecycle state from revoked
        (UPG-MEMORY-STATE-MACHINE §4.1) — it must not be picked up by the
        revoked-only filter either."""
        base = "base content rev superseded"
        superseded = "near content that gets superseded not revoked"
        successor = "totally unrelated successor content rev"
        embedder = _ControlledEmbedder({
            base: BASE_VEC,
            superseded: _unit_vector_at_cosine(0.95),
            successor: UNRELATED_VEC,
        })
        store = _store(tmp_path, embedder)
        superseded_id = store.remember(WS, superseded)
        base_id = store.remember(WS, base)
        store.remember(WS, successor, supersedes=superseded_id)

        revoked = store.revoked_related_notes(WS, base_id, limit=5, min_similarity=0.5)

        assert all(r["note_id"] != superseded_id for r in revoked)

    def test_reinstated_note_no_longer_appears(self, tmp_path) -> None:
        """Revocation is reversible (`reinstate_note`) — once reinstated a
        note's folded state is 'active' again, so it must drop back out of
        this deterrent list, matching the same fold every other lifecycle-
        aware surface in this codebase reads."""
        base = "base content rev reinstate"
        near = "near content that gets revoked then reinstated"
        embedder = _ControlledEmbedder({base: BASE_VEC, near: _unit_vector_at_cosine(0.95)})
        store = _store(tmp_path, embedder)
        near_id = store.remember(WS, near)
        base_id = store.remember(WS, base)
        assert store.revoke_note(WS, near_id, reason="wrong") is True

        before = store.revoked_related_notes(WS, base_id, limit=5, min_similarity=0.5)
        assert any(r["note_id"] == near_id for r in before)

        assert store.reinstate_note(WS, near_id) is True

        after = store.revoked_related_notes(WS, base_id, limit=5, min_similarity=0.5)
        assert all(r["note_id"] != near_id for r in after)


class TestRevokedRelatedNotesFloorAndLimit:
    def _three_revoked_candidate_store(self, tmp_path):
        base = "base content rev 3"
        near_a = "near a rev"   # sim 0.90
        near_b = "near b rev"   # sim 0.85
        near_c = "near c rev"   # sim 0.80
        embedder = _ControlledEmbedder({
            base: BASE_VEC,
            near_a: _unit_vector_at_cosine(0.90),
            near_b: _unit_vector_at_cosine(0.85),
            near_c: _unit_vector_at_cosine(0.80),
        })
        store = _store(tmp_path, embedder)
        id_a = store.remember(WS, near_a)
        id_b = store.remember(WS, near_b)
        id_c = store.remember(WS, near_c)
        base_id = store.remember(WS, base)
        for nid in (id_a, id_b, id_c):
            assert store.revoke_note(WS, nid, reason="wrong") is True
        return store, base_id, id_a, id_b, id_c

    def test_raising_the_floor_excludes_everything(self, tmp_path) -> None:
        store, base_id, id_a, id_b, id_c = self._three_revoked_candidate_store(tmp_path)

        loose = store.revoked_related_notes(WS, base_id, limit=10, min_similarity=0.75)
        assert {r["note_id"] for r in loose} == {id_a, id_b, id_c}

        strict = store.revoked_related_notes(WS, base_id, limit=10, min_similarity=0.95)
        assert strict == []

    def test_limit_caps_the_returned_count_highest_similarity_first(self, tmp_path) -> None:
        store, base_id, id_a, id_b, id_c = self._three_revoked_candidate_store(tmp_path)

        capped = store.revoked_related_notes(WS, base_id, limit=1, min_similarity=0.75)
        assert len(capped) == 1
        assert capped[0]["note_id"] == id_a  # 0.90, the closest

        capped2 = store.revoked_related_notes(WS, base_id, limit=2, min_similarity=0.75)
        assert [r["note_id"] for r in capped2] == [id_a, id_b]

    def test_limit_zero_or_negative_returns_empty(self, tmp_path) -> None:
        store, base_id, *_ = self._three_revoked_candidate_store(tmp_path)
        assert store.revoked_related_notes(WS, base_id, limit=0, min_similarity=0.5) == []
        assert store.revoked_related_notes(WS, base_id, limit=-1, min_similarity=0.5) == []


class TestRevokedRelatedNotesFailSafe:
    def test_no_embedder_returns_empty_list(self, tmp_path) -> None:
        store = WorkingContextStore(str(tmp_path))  # embed_fn=None, no chroma client
        note_id = store.remember(WS, "a plain note, no embedder attached, rev")
        assert store.revoked_related_notes(WS, note_id, limit=5, min_similarity=0.5) == []

    def test_collection_query_raising_returns_empty_not_exception(self, tmp_path) -> None:
        embedder = _ControlledEmbedder({"solo note rev": BASE_VEC})
        store = _store(tmp_path, embedder)
        note_id = store.remember(WS, "solo note rev")

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated chroma failure")

        store._notes_col.query = _boom

        assert store.revoked_related_notes(WS, note_id, limit=5, min_similarity=0.5) == []

    def test_swallowed_failure_is_logged_at_debug(self, tmp_path, caplog) -> None:
        embedder = _ControlledEmbedder({"solo note 2 rev": BASE_VEC})
        store = _store(tmp_path, embedder)
        note_id = store.remember(WS, "solo note 2 rev")

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated chroma failure")

        store._notes_col.query = _boom

        with caplog.at_level(logging.DEBUG, logger="agent.working_context_store._related"):
            assert store.revoked_related_notes(WS, note_id, limit=5, min_similarity=0.5) == []

        assert any(
            "revoked_related_notes failed" in r.message and r.exc_info is not None
            for r in caplog.records
        )


class TestRevokedRelatedNotesZeroInference:
    def test_no_embed_calls_when_vector_already_in_collection(self, tmp_path) -> None:
        base = "zero inference base rev"
        near = "zero inference near rev"
        embedder = _ControlledEmbedder({base: BASE_VEC, near: _unit_vector_at_cosine(0.9)})
        store = _store(tmp_path, embedder)
        near_id = store.remember(WS, near)
        base_id = store.remember(WS, base)
        store.revoke_note(WS, near_id, reason="wrong")

        embedder.call_count = 0
        store.revoked_related_notes(WS, base_id, limit=5, min_similarity=0.5)

        assert embedder.call_count == 0

    def test_fallback_embeds_exactly_once_when_vector_missing_from_collection(self, tmp_path) -> None:
        base = "fallback base content rev"
        near = "fallback near content rev"
        embedder = _ControlledEmbedder({base: BASE_VEC, near: _unit_vector_at_cosine(0.9)})
        store = _store(tmp_path, embedder)
        near_id = store.remember(WS, near)
        base_id = store.remember(WS, base)
        store.revoke_note(WS, near_id, reason="wrong")

        store._notes_col.delete(ids=[str(base_id)])

        embedder.call_count = 0
        revoked = store.revoked_related_notes(WS, base_id, limit=5, min_similarity=0.5)

        assert embedder.call_count == 1
        assert any(r["note_id"] == near_id for r in revoked)


class TestRevokedRelatedNotesOverFetch:
    """Same GLOBAL-collection over-fetch requirement as
    `TestRelatedActiveNotesOverFetch` above — the mirror image here is
    nearer-but-ACTIVE neighbours: if the lookup requested only `limit + 1`
    candidates, closer active notes would starve out a real (further)
    revoked near-duplicate."""

    def test_active_neighbours_do_not_starve_the_revoked_result(self, tmp_path) -> None:
        base = "base content rev ofw"
        active_a = "closer active neighbour a"
        active_b = "closer active neighbour b"
        revoked_a = "further revoked neighbour a"
        revoked_b = "further revoked neighbour b"
        embedder = _ControlledEmbedder({
            base: BASE_VEC,
            # The two NEAREST neighbours are active and are dropped by the
            # folded-lifecycle-state filter after the search.
            active_a: _unit_vector_at_cosine(0.99),
            active_b: _unit_vector_at_cosine(0.98),
            revoked_a: _unit_vector_at_cosine(0.95),
            revoked_b: _unit_vector_at_cosine(0.94),
        })
        store = _store(tmp_path, embedder)
        store.remember(WS, active_a)
        store.remember(WS, active_b)
        revoked_a_id = store.remember(WS, revoked_a)
        revoked_b_id = store.remember(WS, revoked_b)
        base_id = store.remember(WS, base)

        assert store.revoke_note(WS, revoked_a_id, reason="wrong") is True
        assert store.revoke_note(WS, revoked_b_id, reason="wrong") is True

        revoked = store.revoked_related_notes(WS, base_id, limit=2, min_similarity=0.5)

        assert [r["note_id"] for r in revoked] == [revoked_a_id, revoked_b_id]


class TestRevokedRelatedNotesOverFetchDepth:
    """UPG-RELATED-REVOKED-OVERFETCH-DEPTH: the revoked-path candidate pool
    (`n_query`) must be sized off `revoked_query_floor`, NOT off
    `limit * 3` — with the shipped `revoked_limit: 1`, `limit * 3` is only
    3, so as few as three closer ACTIVE near-duplicates on the same
    now-corrected topic can crowd a real revoked near-duplicate out of the
    pool before the lifecycle-state filter even runs. This is a materially
    different scenario from `TestRevokedRelatedNotesOverFetch` above (which
    uses only two active neighbours and a `limit=2` pool of 5 — already
    wide enough to hold the whole 5-note collection under the OLD formula
    too, so it does not exercise this defect).
    """

    def test_several_closer_active_near_duplicates_do_not_hide_a_real_revoked_one(
        self, tmp_path
    ) -> None:
        """Mirrors the shipped defaults: `revoked_limit=1`,
        `min_similarity=0.75` (see agent/config.yaml
        memory_write.related_notes). Three active near-duplicates all rank
        ABOVE the revoked one and all clear the similarity floor. Against
        the pre-fix `n_query = min(max(limit * 3, limit + 1), col_count)`
        formula (limit=1 -> n_query=3), the vector query only ever returns
        the three active notes and the revoked one is never fetched at all
        — this assertion fails on that code and passes once `n_query` is
        floored by `revoked_query_floor` instead.
        """
        base = "WorkspaceLock.acquire takes a PID-scoped lock depth"
        revoked = "WorkspaceLock.acquire grabs a PID-scoped lock depth (revoked)"
        active_a = "WorkspaceLock.acquire holds a PID-scoped lock depth a"
        active_b = "WorkspaceLock.acquire keeps a PID-scoped lock depth b"
        active_c = "WorkspaceLock.acquire retains a PID-scoped lock depth c"
        embedder = _ControlledEmbedder({
            base: BASE_VEC,
            # Three ACTIVE neighbours all rank closer than the revoked one.
            active_a: _unit_vector_at_cosine(0.99),
            active_b: _unit_vector_at_cosine(0.98),
            active_c: _unit_vector_at_cosine(0.97),
            # The revoked near-duplicate is real (clears min_similarity
            # 0.75) but ranks 4th — outside a pool of 3.
            revoked: _unit_vector_at_cosine(0.90),
        })
        store = _store(tmp_path, embedder)
        store.remember(WS, active_a)
        store.remember(WS, active_b)
        store.remember(WS, active_c)
        revoked_id = store.remember(WS, revoked)
        base_id = store.remember(WS, base)

        assert store.revoke_note(WS, revoked_id, reason="was wrong") is True

        found = store.revoked_related_notes(WS, base_id, limit=1, min_similarity=0.75)

        assert [r["note_id"] for r in found] == [revoked_id]


class TestRevokedRelatedNotesQueryDepthFormula:
    """Pins the exact `n_query` formula at the smallest possible level —
    the literal `n_results` value passed to the ChromaDB query — rather
    than only inferring it indirectly through which notes come back."""

    @staticmethod
    def _padding_content_and_vectors(n: int) -> dict[str, list[float]]:
        """`n` distinct note contents, each mapped to one of a small set of
        one-hot 4-d unit vectors (repeats across distinct note_ids are fine
        — ChromaDB dedups by id, not by vector). Only used to inflate
        `col_count`; these notes' actual similarity to anything else is
        irrelevant to the tests that use this helper."""
        one_hots = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        return {f"padding note {i}": one_hots[i % len(one_hots)] for i in range(n)}

    def test_n_query_uses_the_configured_floor_not_limit_times_three(self, tmp_path) -> None:
        from agent.config import MEMORY_WRITE_RELATED_REVOKED_QUERY_FLOOR

        base = "n_query formula base"
        # Pad the collection well past the floor so `col_count` never caps
        # `n_query` below it — that would make this test pass by accident.
        pad_n = MEMORY_WRITE_RELATED_REVOKED_QUERY_FLOOR + 50
        vectors = self._padding_content_and_vectors(pad_n)
        vectors[base] = BASE_VEC
        embedder = _ControlledEmbedder(vectors)
        store = _store(tmp_path, embedder)
        for i in range(pad_n):
            store.remember(WS, f"padding note {i}")
        base_id = store.remember(WS, base)

        captured: dict[str, int] = {}
        real_query = store._notes_col.query

        def _spy_query(*args, **kwargs):
            captured["n_results"] = kwargs.get("n_results")
            return real_query(*args, **kwargs)

        store._notes_col.query = _spy_query

        # limit=1 -> limit * 3 == 3, far below the floor: the floor must win.
        store.revoked_related_notes(WS, base_id, limit=1, min_similarity=0.5)
        assert captured["n_results"] == MEMORY_WRITE_RELATED_REVOKED_QUERY_FLOOR

        # A large enough limit makes `limit * 3` exceed the floor: the
        # formula's `max()` must still pick it up rather than clamping to
        # the floor unconditionally.
        big_limit = (MEMORY_WRITE_RELATED_REVOKED_QUERY_FLOOR // 3) + 5
        store.revoked_related_notes(WS, base_id, limit=big_limit, min_similarity=0.5)
        assert captured["n_results"] == big_limit * 3

    def test_active_path_formula_is_unchanged(self, tmp_path) -> None:
        """UPG-RELATED-REVOKED-OVERFETCH-DEPTH deliberately touches only the
        revoked path — `related_active_notes`'s pool still tracks the
        active target class (~99% of a typical corpus), which the pre-fix
        `limit * 3` over-fetch already served correctly. This pins that the
        active path's formula was NOT changed alongside the revoked path's."""
        base = "active n_query formula base"
        vectors = self._padding_content_and_vectors(200)
        vectors = {k.replace("padding note", "active padding note"): v for k, v in vectors.items()}
        vectors[base] = BASE_VEC
        embedder = _ControlledEmbedder(vectors)
        store = _store(tmp_path, embedder)
        for i in range(200):
            store.remember(WS, f"active padding note {i}")
        base_id = store.remember(WS, base)

        captured: dict[str, int] = {}
        real_query = store._notes_col.query

        def _spy_query(*args, **kwargs):
            captured["n_results"] = kwargs.get("n_results")
            return real_query(*args, **kwargs)

        store._notes_col.query = _spy_query

        store.related_active_notes(WS, base_id, limit=3, min_similarity=0.5)
        assert captured["n_results"] == 9  # unchanged limit * 3 formula
