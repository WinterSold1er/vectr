"""Notes-safety tests for the embed-model change path (UPG-NOTES-VECTOR-REBUILD).

A model change must invalidate and rebuild VECTOR indexes only. Working-memory
notes are the one thing in vectr a user cannot regenerate, so every test here
proves, against a real ChromaDB (mocking it would not prove the dimension
pinning that makes this a hazard) that after a model switch — including a switch
to a model of a DIFFERENT vector dimensionality — every note still exists with
its original content, tags and kind, and its vector has been rebuilt in the new
space.

Everything is tmp_path-based: no real ~/.vectr, no real cache root, no real
embedding model, no network.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import chromadb
import pytest

from agent.indexer import CodeIndexer
from agent.working_context_store import WorkingContextStore

OLD_MODEL = "local-4d-model"
NEW_MODEL = "google/gemini-embedding-2"


def _fixed_embed(dim: int):
    """Embed_fn stand-in returning a fixed `dim`-wide vector per text, and
    recording the texts it was asked to embed."""
    calls: list[list[str]] = []

    def _embed(texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        vector = [0.0] * dim
        vector[0] = 1.0
        vector[-1] = float(dim)
        return [list(vector) for _ in texts]

    _embed.calls = calls  # type: ignore[attr-defined]
    return _embed


def _note_rows(db_dir: Path) -> list[tuple]:
    conn = sqlite3.connect(str(db_dir / "working_context.sqlite"))
    try:
        return conn.execute(
            "SELECT note_id, content, tags, kind FROM notes ORDER BY note_id"
        ).fetchall()
    finally:
        conn.close()


@pytest.fixture
def db_dir(tmp_path: Path) -> Path:
    """The store's db directory: production creates + secures it in
    VectrService.__init__ before constructing the store, so a test has to."""
    directory = tmp_path / "db"
    directory.mkdir()
    return directory


def _seed_notes(store: WorkingContextStore, count: int = 3) -> list[int]:
    return [
        store.remember("/ws", f"note {i} about workspace locks", tags=["keep"])
        for i in range(count)
    ]


class TestNotesSurviveModelSwitch:
    def test_cross_dimension_switch_rebuilds_vectors_and_keeps_every_note(
        self, tmp_path, db_dir
    ) -> None:
        chroma_path = tmp_path / "chroma"

        client_a = chromadb.PersistentClient(path=str(chroma_path))
        store_a = WorkingContextStore(
            str(db_dir),
            embed_fn=_fixed_embed(4),
            notes_chroma_client=client_a,
            embed_model=OLD_MODEL,
        )
        note_ids = _seed_notes(store_a)
        assert store_a._stored_notes_embed_model() == OLD_MODEL
        before = _note_rows(db_dir)
        assert len(before) == 3

        # The daemon restarts on the new model: fresh store instance, fresh
        # ChromaDB client over the same files, 4-dim collection already pinned.
        embed_new = _fixed_embed(6)
        client_b = chromadb.PersistentClient(path=str(chroma_path))
        store_b = WorkingContextStore(
            str(db_dir),
            embed_fn=embed_new,
            notes_chroma_client=client_b,
            embed_model=NEW_MODEL,
        )

        assert _note_rows(db_dir) == before, "note rows must be byte-identical"
        assert store_b._stored_notes_embed_model() == NEW_MODEL
        notes_col = client_b.get_collection("working_memory")
        assert notes_col.count() == 3, "every note's vector must exist in the new space"
        assert sorted(notes_col.get(include=[])["ids"]) == sorted(str(i) for i in note_ids)
        embedded = [text for batch in embed_new.calls for text in batch]
        assert all(row[1] in embedded for row in before), (
            "each note's content must have been re-embedded into the new space"
        )
        # Semantic recall works in the rebuilt space, and what comes back is the
        # original note — same id, content and tags, not a re-created one.
        recalled = store_b.recall("/ws", "workspace locks")
        assert {n.note_id for n in recalled} == set(note_ids)
        assert sorted(n.content for n in recalled) == sorted(row[1] for row in before)
        assert all(n.tags == ["keep"] for n in recalled)

    def test_attach_embedder_path_is_equally_safe(self, tmp_path, db_dir) -> None:
        """The daemon's real upgrade path (memory tools live before the model
        loads, then attach_embedder runs) must not lose notes either."""
        chroma_path = tmp_path / "chroma"

        client_a = chromadb.PersistentClient(path=str(chroma_path))
        store_a = WorkingContextStore(
            str(db_dir),
            embed_fn=_fixed_embed(4),
            notes_chroma_client=client_a,
            embed_model=OLD_MODEL,
        )
        note_ids = _seed_notes(store_a)
        before = _note_rows(db_dir)

        embed_new = _fixed_embed(6)
        client_b = chromadb.PersistentClient(path=str(chroma_path))
        store_b = WorkingContextStore(str(db_dir))  # embedder-less start
        store_b.attach_embedder(
            embed_fn=embed_new, notes_chroma_client=client_b, embed_model=NEW_MODEL
        )

        assert _note_rows(db_dir) == before
        assert store_b._stored_notes_embed_model() == NEW_MODEL
        assert client_b.get_collection("working_memory").count() == len(note_ids)

    def test_same_dimension_switch_needs_no_rebuild(self, tmp_path, db_dir) -> None:
        """The rebuild is the fallback for a dimension change, not the normal
        path: a same-dimension switch re-embeds in place."""
        chroma_path = tmp_path / "chroma"

        client_a = chromadb.PersistentClient(path=str(chroma_path))
        WorkingContextStore(
            str(db_dir),
            embed_fn=_fixed_embed(4),
            notes_chroma_client=client_a,
            embed_model=OLD_MODEL,
        ).remember("/ws", "a note")

        embed_new = _fixed_embed(4)
        client_b = chromadb.PersistentClient(path=str(chroma_path))
        store_b = WorkingContextStore(
            str(db_dir),
            embed_fn=embed_new,
            notes_chroma_client=client_b,
            embed_model=NEW_MODEL,
        )
        assert store_b._stored_notes_embed_model() == NEW_MODEL
        assert client_b.get_collection("working_memory").count() == 1
        assert len(embed_new.calls) == 1, "one migration pass, no rebuild retry"
        assert len(_note_rows(db_dir)) == 1


class TestCodeIndexRebuildDoesNotReachNotes:
    def test_model_stamp_rebuild_recreates_code_collections_only(
        self, tmp_path, monkeypatch
    ) -> None:
        """The code index and the notes share one db_dir; the embed-model stamp
        mismatch path drops and recreates the code collections by name, so the
        working_memory collection sitting in the same database is untouched."""

        class _DummyProvider:
            def embed(self, texts):
                return [[1.0, 0.0] for _ in texts]

            def embed_query(self, texts):
                return self.embed(texts)

        monkeypatch.setattr("agent.indexer.get_embed_provider", lambda _spec: _DummyProvider())

        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "locks.py").write_text(
            "def acquire_lock(workspace):\n    return workspace\n", encoding="utf-8"
        )

        indexer = CodeIndexer(
            str(workspace), embed_model=OLD_MODEL, db_path=str(tmp_path / "db")
        )
        notes_col = indexer._client.get_or_create_collection(
            name="working_memory", metadata={"hnsw:space": "cosine"}
        )
        notes_col.upsert(ids=["11"], embeddings=[[1.0, 0.0]])
        assert notes_col.count() == 1

        # A model change is what index_workspace() detects via the stamp; the
        # rebuild it triggers is the same call the daemon makes.
        indexer.embed_model = NEW_MODEL
        indexer._recreate_collections()

        assert indexer._client.get_collection("code_chunks").count() == 0
        rebuilt_notes = indexer._client.get_collection("working_memory")
        assert rebuilt_notes.count() == 1
        assert rebuilt_notes.get(ids=["11"])["ids"] == ["11"]

    def test_a_note_vector_of_the_old_dimension_does_not_break_the_code_rebuild(
        self, tmp_path, monkeypatch
    ) -> None:
        class _DummyProvider:
            def embed(self, texts):
                return [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in texts]

            def embed_query(self, texts):
                return self.embed(texts)

        monkeypatch.setattr("agent.indexer.get_embed_provider", lambda _spec: _DummyProvider())

        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "locks.py").write_text(
            "def acquire_lock(workspace):\n    return workspace\n", encoding="utf-8"
        )
        indexer = CodeIndexer(
            str(workspace), embed_model=OLD_MODEL, db_path=str(tmp_path / "db")
        )
        notes_col = indexer._client.get_or_create_collection(
            name="working_memory", metadata={"hnsw:space": "cosine"}
        )
        notes_col.upsert(ids=["11"], embeddings=[[1.0, 0.0, 0.0, 0.0]])

        indexer.embed_model = NEW_MODEL
        indexer._recreate_collections()
        # Code vectors now live in the new dimension; the note vector is intact.
        indexer._collection.upsert(ids=["chunk-1"], embeddings=[[1.0] * 6])
        assert indexer._collection.count() == 1
        assert notes_col.count() == 1
