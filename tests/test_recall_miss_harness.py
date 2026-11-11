"""Regression tests for the recall-miss measurement harness
(UPG-RECALL-MISS-FLOOR, part (a) only — see benchmarks/recall_miss/harness.py).

Two layers, both required to pin "the harness's own correctness":
  1. TestEvaluateRecallLogic — pure arithmetic/grouping of evaluate_recall()
     against a scripted store (no embedder, no I/O, no real recall() call).
  2. TestFixtureCorpusEndToEnd — the harness wired to a real
     WorkingContextStore + the committed fixture corpus, using the same
     deterministic hash embedder tests/test_memory.py's B9 semantic-recall
     suite already uses, so this needs no model download and stays fast.

Lives under tests/, not benchmarks/, specifically so it is collected by the
default `pytest -q` run and by CI's `pytest tests/` step — pytest.ini's
testpaths and .github/workflows/ci.yml both scope routine runs to
`tests agent integrations app` / `tests/`, so a test physically under
benchmarks/recall_miss/ would silently never run in CI.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "recall_miss"))

from harness import LabeledQuery, NoteInfo, evaluate_recall  # noqa: E402


class _StubNote:
    """Minimal stand-in for WorkingNote — evaluate_recall() only ever reads
    `.note_id` off whatever store.recall() returns."""

    def __init__(self, note_id: int) -> None:
        self.note_id = note_id


class _ScriptedStore:
    """Fake store: canned recall() results per query string, no embedder, no
    I/O. Records every call's (workspace, query, limit) so tests can assert
    on exactly how evaluate_recall() drives the real recall() signature."""

    def __init__(self, results: dict[str, list[int]]) -> None:
        self._results = results
        self.calls: list[tuple[str, str | None, int]] = []

    def recall(self, workspace: str, query: str | None = None, limit: int = 10, **_kwargs):
        self.calls.append((workspace, query, limit))
        return [_StubNote(nid) for nid in self._results.get(query, [])]


def _key_to_note() -> dict[str, NoteInfo]:
    return {
        "a": NoteInfo(note_id=1, kind="gotcha", content="note A content"),
        "b": NoteInfo(note_id=2, kind="finding", content="note B content"),
        "c": NoteInfo(note_id=3, kind="finding", content="note C content"),
        "d": NoteInfo(note_id=4, kind="directive", content="note D content"),
    }


class TestEvaluateRecallLogic:
    """Pure-logic pinning of evaluate_recall()'s arithmetic and grouping —
    no embedder, no I/O, no real recall() call."""

    def test_full_hit_full_miss_and_partial(self) -> None:
        key_to_note = _key_to_note()
        queries = [
            LabeledQuery(query="q-full-hit", relevant_keys=("a",), basis="b1"),
            LabeledQuery(query="q-full-miss", relevant_keys=("d",), basis="b2"),
            LabeledQuery(query="q-partial", relevant_keys=("b", "c"), basis="b3"),
        ]
        store = _ScriptedStore({
            "q-full-hit": [1],       # returns the one relevant note -> hit
            "q-full-miss": [99],     # returns something else entirely -> miss
            "q-partial": [2, 100],   # hits b, misses c
        })

        report = evaluate_recall(store, "/ws", queries, key_to_note, k=10)

        assert report.n_queries == 3
        assert report.n_queries_scored == 3
        assert report.total_relevant == 4  # a, d, b, c
        assert report.total_hits == 2      # a, b
        assert report.macro_recall_at_k == pytest.approx((1.0 + 0.0 + 0.5) / 3)
        assert report.micro_recall_at_k == pytest.approx(2 / 4)
        assert report.hits_by_kind == {"gotcha": 1, "finding": 1}
        assert report.misses_by_kind == {"directive": 1, "finding": 1}

        miss_note_ids = {m.note_id for m in report.misses}
        assert miss_note_ids == {4, 3}
        for m in report.misses:
            assert m.content_snippet.startswith("note")
            assert m.basis in ("b2", "b3")

        # every store.recall() call carries the harness's own k as `limit`
        assert all(c[2] == 10 for c in store.calls)
        assert all(c[0] == "/ws" for c in store.calls)

    def test_empty_relevant_keys_excluded_from_scoring_but_counted(self) -> None:
        key_to_note = _key_to_note()
        queries = [
            LabeledQuery(query="q-no-labels", relevant_keys=(), basis="nothing to recall"),
            LabeledQuery(query="q-hit", relevant_keys=("a",), basis="b1"),
        ]
        store = _ScriptedStore({"q-hit": [1]})

        report = evaluate_recall(store, "/ws", queries, key_to_note, k=5)

        assert report.n_queries == 2
        assert report.n_queries_scored == 1
        assert report.total_relevant == 1
        assert report.total_hits == 1
        assert report.macro_recall_at_k == pytest.approx(1.0)
        assert report.micro_recall_at_k == pytest.approx(1.0)
        # the unlabeled query must never reach store.recall() at all
        assert [c[1] for c in store.calls] == ["q-hit"]

    def test_unknown_key_raises_keyerror(self) -> None:
        key_to_note = _key_to_note()
        queries = [LabeledQuery(query="q", relevant_keys=("nope",), basis="stale key")]
        store = _ScriptedStore({})

        with pytest.raises(KeyError):
            evaluate_recall(store, "/ws", queries, key_to_note, k=10)

    def test_no_scored_queries_yields_zero_recall_not_zerodivision(self) -> None:
        key_to_note = _key_to_note()
        queries = [LabeledQuery(query="q", relevant_keys=(), basis="nothing")]
        store = _ScriptedStore({})

        report = evaluate_recall(store, "/ws", queries, key_to_note, k=10)

        assert report.n_queries == 1
        assert report.n_queries_scored == 0
        assert report.macro_recall_at_k == 0.0
        assert report.micro_recall_at_k == 0.0
        assert report.misses == []

    def test_format_text_reports_per_kind_miss_rate(self) -> None:
        key_to_note = _key_to_note()
        queries = [LabeledQuery(query="q", relevant_keys=("a", "d"), basis="b")]
        store = _ScriptedStore({"q": [1]})  # hits a (gotcha), misses d (directive)

        report = evaluate_recall(store, "/ws", queries, key_to_note, k=10)
        text = report.format_text(title="unit-test report")

        assert "unit-test report" in text
        assert "k=10" in text
        assert "gotcha" in text and "missed 0/1" in text
        assert "directive" in text and "missed 1/1" in text
        assert "MISSED note_id=4" in text


def _build_dummy_fixture_store(db_dir: Path):
    """A real WorkingContextStore wired with the same deterministic
    hash-based embedder tests/test_memory.py's B9 semantic-recall suite
    uses (`_dummy_embed` / `_semantic_store` there) — same input always
    produces the same vector, near-orthogonal for distinct text, so this
    needs no model download and stays fully reproducible."""
    import chromadb

    from agent.working_context_store import WorkingContextStore

    def _dummy_embed(texts: list[str]) -> list[list[float]]:
        import hashlib

        out = []
        for t in texts:
            h = hashlib.md5(t.encode()).digest()
            vec = [(b / 255.0 - 0.5) for b in (h * 48)]
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            out.append([x / norm for x in vec])
        return out

    client = chromadb.PersistentClient(path=str(db_dir / "chroma"))
    return WorkingContextStore(str(db_dir), embed_fn=_dummy_embed, notes_chroma_client=client)


class TestFixtureCorpusEndToEnd:
    """Smoke-tests the harness's wiring against a real WorkingContextStore
    and the committed fixture corpus (benchmarks/recall_miss/fixture_corpus.py).
    Pins the exact hit/miss split as a regression baseline — verified stable
    across repeated runs against this corpus (deterministic embedder, fixed
    note-insertion order, small enough corpus for ChromaDB's nearest-neighbor
    search to be exact rather than approximate)."""

    def test_exact_match_synthetic_queries_always_hit(self, tmp_path) -> None:
        from fixture_corpus import FIXTURE_QUERIES, FIXTURE_WORKSPACE, build_fixture_corpus

        store = _build_dummy_fixture_store(tmp_path)
        key_to_note = build_fixture_corpus(store, FIXTURE_WORKSPACE)

        exact_match_queries = [
            lq for lq in FIXTURE_QUERIES
            if lq.source == "synthetic"
            and len(lq.relevant_keys) == 1
            and lq.query == key_to_note[lq.relevant_keys[0]].content
        ]
        assert len(exact_match_queries) >= 4, (
            "fixture corpus must keep at least 4 exact-content-match synthetic "
            "queries as a sanity floor for the harness wiring"
        )

        report = evaluate_recall(store, FIXTURE_WORKSPACE, exact_match_queries, key_to_note, k=10)

        assert report.macro_recall_at_k == pytest.approx(1.0)
        assert report.misses == []

    def test_full_fixture_hit_miss_baseline(self, tmp_path) -> None:
        """Regression baseline for the full labeled query set: pins recall@10
        and the kind split so a future embedder/config/recall() change that
        moves this number shows up as a diff here, not silently. This is the
        deliverable itself in miniature — a fixture-scale recall-miss number
        with a per-kind miss breakdown, not just a pass/fail gate."""
        from fixture_corpus import FIXTURE_QUERIES, FIXTURE_WORKSPACE, build_fixture_corpus

        store = _build_dummy_fixture_store(tmp_path)
        key_to_note = build_fixture_corpus(store, FIXTURE_WORKSPACE)

        report = evaluate_recall(store, FIXTURE_WORKSPACE, FIXTURE_QUERIES, key_to_note, k=10)

        assert report.n_queries == len(FIXTURE_QUERIES)
        assert report.n_queries_scored == len(FIXTURE_QUERIES)
        assert report.total_relevant == 18
        assert report.total_hits == 6
        assert report.macro_recall_at_k == pytest.approx(0.353, abs=5e-4)
        assert report.micro_recall_at_k == pytest.approx(0.333, abs=5e-4)
        assert report.hits_by_kind == {
            "task": 1, "directive": 2, "reference": 1, "decision": 1, "operational": 1,
        }
        assert report.misses_by_kind == {
            "gotcha": 3, "task": 3, "decision": 1, "reference": 1, "finding": 4,
        }
        # every gotcha- and finding-kind label in this fixture misses under
        # the dummy embedder (paraphrase-worded queries against near-
        # orthogonal hash vectors) — exactly the failure mode
        # UPG-RECALL-MISS-FLOOR exists to measure and report by kind.
        assert "gotcha" not in report.hits_by_kind
        assert "finding" not in report.hits_by_kind


def _build_notes_snapshot(db_path, rows):
    """Minimal sqlite `notes` table — only the columns
    run_live.load_labels() actually reads — for testing the
    superseded-label guard without a full WorkingContextStore.

    Each row is (note_id, kind, content, valid_until, superseded_by,
    superseded_by_note_id) — same column order/names as the real store
    (agent/working_context_store/_store.py), so `valid_until` and the
    `superseded_by*` columns can be set independently in a test."""
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE notes (note_id INTEGER PRIMARY KEY, kind TEXT, content TEXT, "
        "valid_until REAL, superseded_by TEXT, superseded_by_note_id INTEGER)"
    )
    conn.executemany(
        "INSERT INTO notes (note_id, kind, content, valid_until, superseded_by, superseded_by_note_id) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


class TestLoadLabelsSupersededGuard:
    """run_live.py's load_labels() must refuse a label that points at a
    superseded note: recall() excludes superseded notes by default via
    `valid_until IS NULL` (see WorkingContextStore.recall() /
    agent/working_context_store/_store.py), so such a label can never be
    hit and would silently inflate the measured miss rate rather than
    reflect a real retrieval failure. Guards
    benchmarks/recall_miss/run_live.py:load_labels().

    `valid_until` is the AUTHORITATIVE column (the one recall() actually
    filters on) — superseded_by / superseded_by_note_id are carried only
    as diagnostics in the error message. A legacy code_hash auto-supersede
    path has historically set valid_until without always going through the
    same event-log write that sets superseded_by, so the two columns can
    diverge; the guard must key off valid_until, not superseded_by."""

    def test_superseded_note_id_raises(self, tmp_path) -> None:
        import json
        import sys as _sys

        _sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "recall_miss"))
        from run_live import load_labels  # noqa: PLC0415

        snapshot_dir = tmp_path / "snapshot"
        snapshot_dir.mkdir()
        _build_notes_snapshot(
            snapshot_dir / "working_context.sqlite",
            rows=[(1, "task", "old content", 1700000000.0, None, 2)],  # superseded by note 2
        )
        labels_path = tmp_path / "labels.json"
        labels_path.write_text(json.dumps([
            {"query": "q", "relevant_note_ids": [1], "basis": "b", "source": "live-session"},
        ]))

        with pytest.raises(SystemExit, match="SUPERSEDED"):
            load_labels(labels_path, snapshot_dir)

    def test_non_superseded_note_id_resolves(self, tmp_path) -> None:
        import json
        import sys as _sys

        _sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "recall_miss"))
        from run_live import load_labels  # noqa: PLC0415

        snapshot_dir = tmp_path / "snapshot"
        snapshot_dir.mkdir()
        _build_notes_snapshot(
            snapshot_dir / "working_context.sqlite",
            rows=[(1, "task", "current content", None, None, None)],
        )
        labels_path = tmp_path / "labels.json"
        labels_path.write_text(json.dumps([
            {"query": "q", "relevant_note_ids": [1], "basis": "b", "source": "live-session"},
        ]))

        queries, key_to_note = load_labels(labels_path, snapshot_dir)

        assert len(queries) == 1
        assert queries[0].relevant_keys == ("1",)
        assert key_to_note["1"] == NoteInfo(note_id=1, kind="task", content="current content")

    def test_valid_until_is_authoritative_not_superseded_by(self, tmp_path) -> None:
        """The exact gap sentinel flagged: a row where superseded_by* and
        valid_until DISAGREE must be judged by valid_until, since that is
        the column recall() actually filters on. Two sub-cases in one row
        each: (a) valid_until NULL but superseded_by set (legacy/stale
        superseded_by write with no accompanying valid_until) -> recall()
        would still surface this note, so the label must resolve, NOT
        raise; (b) valid_until set but superseded_by* both NULL (the
        legacy code_hash auto-supersede path) -> recall() would still
        exclude it, so the label must raise."""
        import json
        import sys as _sys

        _sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "recall_miss"))
        from run_live import load_labels  # noqa: PLC0415

        # (a) valid_until NULL, superseded_by set -> must resolve (not raise).
        snapshot_dir = tmp_path / "snapshot_a"
        snapshot_dir.mkdir()
        _build_notes_snapshot(
            snapshot_dir / "working_context.sqlite",
            rows=[(1, "task", "still live per recall()", None, "some-legacy-value", None)],
        )
        labels_path = tmp_path / "labels_a.json"
        labels_path.write_text(json.dumps([
            {"query": "q", "relevant_note_ids": [1], "basis": "b", "source": "live-session"},
        ]))
        queries, key_to_note = load_labels(labels_path, snapshot_dir)
        assert key_to_note["1"] == NoteInfo(note_id=1, kind="task", content="still live per recall()")

        # (b) valid_until set, superseded_by* both NULL -> must raise.
        snapshot_dir_b = tmp_path / "snapshot_b"
        snapshot_dir_b.mkdir()
        _build_notes_snapshot(
            snapshot_dir_b / "working_context.sqlite",
            rows=[(2, "task", "actually excluded by recall()", 1700000000.0, None, None)],
        )
        labels_path_b = tmp_path / "labels_b.json"
        labels_path_b.write_text(json.dumps([
            {"query": "q", "relevant_note_ids": [2], "basis": "b", "source": "live-session"},
        ]))
        with pytest.raises(SystemExit, match="SUPERSEDED"):
            load_labels(labels_path_b, snapshot_dir_b)
