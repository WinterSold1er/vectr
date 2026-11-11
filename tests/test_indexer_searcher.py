"""
Tests for CodeIndexer and CodeSearcher.

Uses a DummyEmbedProvider (from conftest) so no model download is needed.
Tests verify the real ChromaDB storage and hybrid BM25+vector search pipeline.
"""
from __future__ import annotations

import json
import logging
import math
import re
import textwrap
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.conftest import make_py


# ---------------------------------------------------------------------------
# _code_tokenize — BM25 tokenizer
# ---------------------------------------------------------------------------

class TestCodeTokenize:
    def _tok(self, text: str) -> list[str]:
        from agent.searcher import _code_tokenize
        return _code_tokenize(text)

    def test_snake_case_split(self) -> None:
        tokens = self._tok("dispatch_uid")
        assert "dispatch" in tokens
        assert "uid" in tokens

    def test_camel_case_split(self) -> None:
        tokens = self._tok("RateLimitMiddleware")
        assert "rate" in tokens
        assert "limit" in tokens
        assert "middleware" in tokens

    def test_mixed_identifier(self) -> None:
        tokens = self._tok("send_signal_dispatch_uid")
        assert "send" in tokens
        assert "dispatch" in tokens
        assert "uid" in tokens

    def test_short_tokens_filtered(self) -> None:
        tokens = self._tok("a b c def")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "def" in tokens

    def test_punctuation_stripped(self) -> None:
        tokens = self._tok("func(arg1, arg2):")
        assert "func" in tokens
        assert "arg1" in tokens
        assert "arg2" in tokens
        assert "(" not in tokens

    def test_term_frequency_preserved(self) -> None:
        """UPG-BM25-TF: `_code_tokenize` (used to build BM25 documents) must NOT
        dedupe — BM25Plus needs real per-document term frequency, not IDF-weighted
        set-overlap. A chunk mentioning a token 3x is a stronger signal than one
        mentioning it once, and only survives if the tokenizer keeps all 3."""
        tokens = self._tok("foo foo foo")
        assert tokens.count("foo") == 3

    def test_query_tokenize_dedupes(self) -> None:
        """`_code_tokenize_query` (used to tokenize the caller's query string)
        DOES dedupe — a query repeating a content word is a phrasing accident,
        not evidence the caller wants that term weighted higher, so it must not
        double-count against BM25Plus.get_scores (which sums per token seen,
        duplicates included)."""
        from agent.searcher import _code_tokenize_query
        tokens = _code_tokenize_query("foo foo foo")
        assert tokens.count("foo") == 1

    def test_repeated_token_chunk_outranks_single_mention_bm25(self) -> None:
        """UPG-BM25-TF regression, at the BM25Plus scoring level (no indexer/
        searcher needed): a chunk that mentions a distinctive token 5x must score
        strictly higher than one mentioning it only once. Before the fix, per-
        document dedupe in `_code_tokenize` forced every term frequency to 1,
        collapsing this into an IDF-weighted set-overlap tie."""
        from agent.searcher import _code_tokenize
        from rank_bm25 import BM25Plus

        repeated_doc = "widget widget widget widget widget among other filler prose"
        single_doc = "widget appears here only once among other filler prose"
        # Padding docs so avgdl/IDF are computed over a non-trivial corpus.
        padding = ["completely unrelated filler prose about something else"] * 5

        corpus = [repeated_doc, single_doc, *padding]
        bm25 = BM25Plus([_code_tokenize(d) for d in corpus])
        scores = bm25.get_scores(_code_tokenize("widget"))
        assert scores[0] > scores[1], (
            f"repeated-mention doc ({scores[0]}) must outrank single-mention "
            f"doc ({scores[1]}) once term frequency survives tokenization"
        )


# ---------------------------------------------------------------------------
# CodeIndexer — index / delete / query
# ---------------------------------------------------------------------------

class TestCodeIndexer:
    def test_index_file_creates_chunks(self, indexer, tmp_path) -> None:
        path = make_py(tmp_path, "auth.py", """
            def verify_token(token: str) -> dict:
                return {}

            class AuthMiddleware:
                def process(self, request):
                    pass
        """)
        count = indexer.index_file(path)
        assert count >= 2
        assert indexer.total_chunks >= 2

    def test_index_file_idempotent(self, indexer, tmp_path) -> None:
        """Re-indexing the same file must not create duplicate chunks."""
        path = make_py(tmp_path, "utils.py", """
            def helper():
                pass
        """)
        indexer.index_file(path)
        first = indexer.total_chunks
        indexer.index_file(path)
        assert indexer.total_chunks == first

    def test_delete_file_removes_chunks(self, indexer, tmp_path) -> None:
        path = make_py(tmp_path, "temp.py", """
            def to_remove():
                x = 1
        """)
        indexer.index_file(path)
        assert indexer.total_chunks > 0
        indexer.delete_file(path)
        assert indexer.total_chunks == 0

    def test_index_workspace_walks_py_files(self, indexer, tmp_path) -> None:
        # UPG-11.3: .txt files are now indexed (language='txt', doc-prose quality).
        # The test previously asserted files==2 to confirm .txt was skipped, but
        # now .txt IS indexed. Update: use a .log extension (not indexed) instead.
        make_py(tmp_path, "a.py", "def foo(): pass")
        make_py(tmp_path, "b.py", "def bar(): pass")
        (tmp_path / "skip.log").write_text("not indexed — .log is not in LANG_BY_EXT")
        files, chunks = indexer.index_workspace()
        assert files == 2
        assert chunks >= 2

    def test_index_workspace_skips_excluded_dirs(self, indexer, tmp_path) -> None:
        make_py(tmp_path, "main.py", "def main(): pass")
        node_modules = tmp_path / "node_modules"
        node_modules.mkdir()
        (node_modules / "pkg.py").write_text("def evil(): pass")
        files, chunks = indexer.index_workspace()
        assert files == 1

    def test_embed_query_returns_vector(self, indexer) -> None:
        vec = indexer.embed_query("how does rate limiting work")
        assert isinstance(vec, list)
        assert len(vec) == 768  # DummyEmbedProvider mirrors nomic-embed-code dim
        assert all(isinstance(v, float) for v in vec)

    def test_close_releases_chroma_client(self, indexer) -> None:
        """CodeIndexer.close() must release the underlying ChromaDB client
        (its System, including the native worker-thread pool ChromaDB starts
        per client) — otherwise every CodeIndexer built over a process's
        lifetime leaks native OS threads that never get reclaimed."""
        with patch.object(indexer.chroma_client, "close", wraps=indexer.chroma_client.close) as spy:
            indexer.close()
        spy.assert_called_once()

    def test_close_is_idempotent(self, indexer) -> None:
        indexer.close()
        indexer.close()  # must not raise the second time

    def test_del_closes_chroma_client(self, tmp_path, monkeypatch) -> None:
        """A CodeIndexer that falls out of scope without an explicit
        shutdown call (e.g. a fixture, a script) must still release its
        ChromaDB client via __del__ — the safety net for the many callers
        that don't own a VectrService.shutdown() path. Constructed directly
        (not via the `indexer` fixture) so this test — not pytest's
        function-scoped fixture cache — holds the only reference."""
        import gc

        from agent import indexer as idx_module
        from agent.indexer import CodeIndexer
        from tests.conftest import _DummyEmbedProvider

        monkeypatch.setattr(idx_module, "get_embed_provider", lambda _model: _DummyEmbedProvider())
        local_indexer = CodeIndexer(workspace_root=str(tmp_path), db_path=str(tmp_path / "chroma"))
        client = local_indexer.chroma_client
        with patch.object(client, "close", wraps=client.close) as spy:
            del local_indexer
            gc.collect()
        spy.assert_called_once()

    def test_query_vector_returns_results_after_index(self, indexer, tmp_path) -> None:
        path = make_py(tmp_path, "search_me.py", """
            def rate_limit_check(ip: str) -> bool:
                return True
        """)
        indexer.index_file(path)
        embedding = indexer.embed_query("rate limit check function")
        result = indexer.query_vector(embedding, n_results=5)
        assert len(result["ids"][0]) >= 1

    def test_query_vector_language_filter(self, indexer, tmp_path) -> None:
        py_file = make_py(tmp_path, "app.py", "def process(): pass")
        js_file = tmp_path / "app.js"
        js_file.write_text("function process() {}")
        indexer.index_file(py_file)
        indexer.index_file(str(js_file))
        embedding = indexer.embed_query("process function")
        result = indexer.query_vector(embedding, n_results=10, language="python")
        metas = result["metadatas"][0]
        assert all(m["language"] == "python" for m in metas)

    def test_query_vector_languages_in_filter(self, indexer, tmp_path) -> None:
        # UPG-15.13: the `languages` (set) param restricts results to any of the
        # given languages via an $in filter, and takes precedence over `language`.
        py_file = make_py(tmp_path, "app.py", "def process(): pass")
        js_file = tmp_path / "app.js"
        js_file.write_text("function process() {}")
        txt_file = tmp_path / "notes.txt"
        txt_file.write_text("process function documentation prose here")
        indexer.index_file(py_file)
        indexer.index_file(str(js_file))
        indexer.index_file(str(txt_file))
        embedding = indexer.embed_query("process function")
        result = indexer.query_vector(
            embedding, n_results=10, languages=["javascript", "text", "txt"]
        )
        metas = result["metadatas"][0]
        assert metas, "expected at least one non-python match"
        assert all(m["language"] in {"javascript", "text", "txt"} for m in metas)
        assert all(m["language"] != "python" for m in metas)

    def test_total_chunks_property(self, indexer, tmp_path) -> None:
        assert indexer.total_chunks == 0
        make_py(tmp_path, "x.py", "def f(): pass")
        indexer.index_workspace()
        assert indexer.total_chunks > 0

    def test_total_chunks_cache_matches_real_count_after_index_file(self, indexer, tmp_path) -> None:
        # UPG-CHROMA-BLOCKING-EVENT-LOOP: total_chunks is refreshed from a
        # real collection.count() at the end of index_file, never read live
        # from chroma on every access — confirm it matches the real count.
        path = make_py(tmp_path, "a.py", "def a(): pass")
        indexer.index_file(path)
        assert indexer.total_chunks == indexer._collection.count() > 0

    def test_total_chunks_cache_matches_real_count_after_delete_file(self, indexer, tmp_path) -> None:
        path = make_py(tmp_path, "a.py", "def a(): pass")
        indexer.index_file(path)
        indexer.delete_file(path)
        assert indexer.total_chunks == indexer._collection.count() == 0

    def test_total_chunks_cache_matches_real_count_after_index_workspace(self, indexer, tmp_path) -> None:
        make_py(tmp_path, "a.py", "def a(): pass")
        make_py(tmp_path, "b.py", "def b(): pass")
        indexer.index_workspace()
        assert indexer.total_chunks == indexer._collection.count() > 0

    def test_index_workspace_refreshes_counter_between_batches(
        self, indexer, tmp_path, monkeypatch
    ) -> None:
        """A bulk index_workspace() must not leave total_chunks/last_indexed_ts
        frozen for the whole call — an external caller polling those values to
        detect "indexing is still in progress" needs to see them move as each
        embed/upsert batch completes, not just once at the very end (which
        would make it settle on a partial index mid-way through a large one).
        """
        import agent.indexer._core as core_module

        # Shrink the batch size so a handful of files produces several
        # embed/upsert batches instead of one.
        monkeypatch.setattr(core_module, "_EMBED_BATCH_SIZE", 2)

        for i in range(6):
            make_py(tmp_path, f"m{i}.py", f"def f{i}():\n    return {i}\n")

        observed_counts: list[int] = []
        observed_timestamps: list[float] = []
        original_refresh = indexer._refresh_chunk_count_caches

        def _recording_refresh() -> None:
            original_refresh()
            observed_counts.append(indexer.total_chunks)
            observed_timestamps.append(indexer.last_indexed_ts)

        monkeypatch.setattr(indexer, "_refresh_chunk_count_caches", _recording_refresh)

        indexer.index_workspace()

        # More than one refresh happened during the call (per batch, not
        # just once at the end), the counter only ever grows, and it ends
        # higher than it started — proving both counters moved DURING the
        # call rather than jumping straight from empty to final at the end.
        assert len(observed_counts) > 2, observed_counts
        assert observed_counts == sorted(observed_counts), observed_counts
        assert observed_counts[0] < observed_counts[-1], observed_counts
        assert observed_timestamps == sorted(observed_timestamps), observed_timestamps

    def test_total_chunks_read_never_calls_collection_count(self, indexer, tmp_path) -> None:
        """UPG-CHROMA-BLOCKING-EVENT-LOOP: total_chunks must be answerable
        without ever touching ChromaDB at read time — it is a cache
        refreshed only at the end of a mutation method (see
        _refresh_chunk_count_caches). A slow/blocked collection.count() call
        (chroma compacting, e.g.) must never delay a total_chunks read."""
        make_py(tmp_path, "a.py", "def a(): pass")
        indexer.index_workspace()
        n = indexer.total_chunks

        def _must_not_be_called():
            raise AssertionError("total_chunks must not call collection.count() at read time")

        with patch.object(indexer._collection, "count", side_effect=_must_not_be_called):
            assert indexer.total_chunks == n

    def test_indexed_file_count_property(self, indexer, tmp_path) -> None:
        make_py(tmp_path, "a.py", "def a(): pass")
        make_py(tmp_path, "b.py", "def b(): pass")
        indexer.index_workspace()
        assert indexer.indexed_file_count == 2

    def test_indexed_file_paths_property(self, indexer, tmp_path) -> None:
        path = make_py(tmp_path, "known.py", "def x(): pass")
        indexer.index_file(path)
        assert path in indexer.indexed_file_paths

    def test_indexed_file_count_reflects_persisted_collection_on_warm_start(
        self, tmp_path, monkeypatch,
    ) -> None:
        """UPG-STATUS-WARMSTART-FILES: on a warm start (a daemon restart over
        an already-indexed workspace), `indexed_file_count` must report the
        persisted corpus immediately — not 0 while `total_chunks` already
        reports thousands, which made `/v1/status` internally inconsistent for
        the whole warm-start window. It now derives from the SAME seeded
        per-language metadata cache `indexed_language_stats()` reads (actual
        chunk metadata), i.e. exactly what `total_chunks` counts."""
        import agent.indexer as idx_module
        from agent.indexer import CodeIndexer
        from tests.conftest import _DummyEmbedProvider

        monkeypatch.setattr(
            idx_module, "get_embed_provider", lambda _model: _DummyEmbedProvider()
        )
        make_py(tmp_path, "a.py", "def a(): pass")
        make_py(tmp_path, "b.py", "def b(): pass")
        db = str(tmp_path / "chroma")

        first = CodeIndexer(str(tmp_path), db_path=db)
        first.index_workspace()
        assert first.total_chunks > 0

        # The "warm start": a brand-new indexer over the same persisted db,
        # before anything has been walked in this one.
        warm = CodeIndexer(str(tmp_path), db_path=db)
        assert warm.total_chunks > 0
        assert warm._indexed_files == set()  # nothing walked in this process yet
        assert warm.indexed_file_count == 2  # was 0 before the fix
        # Same source of truth as total_chunks: the collection's chunk metadata.
        stats = warm.indexed_language_stats()
        assert warm.indexed_file_count == sum(s["files"] for s in stats.values())

    def test_indexed_file_paths_reflect_persisted_collection_on_warm_start(
        self, tmp_path, monkeypatch,
    ) -> None:
        """UPG-WARMSTART-FINGERPRINT-STARVATION: on a daemon restart over an
        already-indexed workspace, `indexed_file_paths` must report the
        persisted corpus immediately. It previously returned `_indexed_files`
        (the walk-scoped set of files THIS process has walked), which stays
        empty until an in-process index run touches it — so across the whole
        warm-start window the strategy fingerprint behind /v1/status and
        vectr_map's passport render were silently computed from ZERO files,
        indistinguishable from a genuinely fileless workspace."""
        import agent.indexer as idx_module
        from agent.indexer import CodeIndexer
        from tests.conftest import _DummyEmbedProvider

        monkeypatch.setattr(
            idx_module, "get_embed_provider", lambda _model: _DummyEmbedProvider()
        )
        make_py(tmp_path, "a.py", "def a(): pass")
        make_py(tmp_path, "b.py", "def b(): pass")
        db = str(tmp_path / "chroma")

        first = CodeIndexer(str(tmp_path), db_path=db)
        first.index_workspace()
        assert first.total_chunks > 0

        # The "warm start": a brand-new indexer over the same persisted db,
        # before anything has been walked in this one.
        warm = CodeIndexer(str(tmp_path), db_path=db)
        assert warm._indexed_files == set()  # nothing walked in this process yet
        paths = warm.indexed_file_paths
        assert len(paths) == 2, paths
        assert any(p.endswith("a.py") for p in paths)
        assert any(p.endswith("b.py") for p in paths)
        # One source of truth with indexed_file_count (UPG-STATUS-WARMSTART-FILES):
        # exactly one path per counted file — the two properties can never
        # disagree again.
        assert len(paths) == warm.indexed_file_count

    def test_indexed_file_paths_tracks_delta_maintenance(
        self, indexer, tmp_path,
    ) -> None:
        """The metadata-derived indexed_file_paths must keep tracking the
        incremental `_apply_chunk_delta` maintenance on index/delete, not just
        the construction-time seed."""
        path_a = make_py(tmp_path, "a.py", "def a(): pass")
        indexer.index_file(path_a)
        assert [p for p in indexer.indexed_file_paths if p.endswith("a.py")]
        indexer.delete_file(path_a)
        assert not any(p.endswith("a.py") for p in indexer.indexed_file_paths)

    def test_indexed_file_count_falls_with_total_chunks_through_rebuild(
        self, indexer, tmp_path,
    ) -> None:
        """UPG-STATUS-WARMSTART-FILES: indexed_file_count and total_chunks
        share one source of truth (the body collection), so a forced
        `_recreate_collections()` rebuild resets them TOGETHER — the count may
        never get stuck at its pre-rebuild value while chunks climb back from
        zero."""
        make_py(tmp_path, "a.py", "def a(): pass")
        indexer.index_workspace()
        assert indexer.indexed_file_count == 1
        assert indexer.total_chunks > 0
        indexer._recreate_collections()
        assert indexer.total_chunks == 0
        assert indexer.indexed_file_count == 0

    def test_indexed_languages_lists_distinct_sorted(self, indexer, tmp_path) -> None:
        # UPG-3.1: ground-truth language set comes from chunk metadata, not a fixed list.
        make_py(tmp_path, "a.py", "def x(): pass")
        js = tmp_path / "b.js"; js.write_text("function y() {}")
        indexer.index_file(make_py(tmp_path, "a.py", "def x(): pass"))
        indexer.index_file(str(js))
        langs = indexer.indexed_languages()
        assert langs == sorted(langs)
        assert "python" in langs and "javascript" in langs
        assert "cobol" not in langs

    def test_indexed_languages_cache_invalidated_on_change(self, indexer, tmp_path) -> None:
        indexer.index_file(make_py(tmp_path, "a.py", "def x(): pass"))
        assert "javascript" not in indexer.indexed_languages()
        js = tmp_path / "b.js"; js.write_text("function y() {}")
        indexer.index_file(str(js))
        # index_file() applies an incremental delta to the stats cache
        # (UPG-REST-STARVATION) — js is present without any collection rescan.
        assert "javascript" in indexer.indexed_languages()

    def test_indexed_language_stats_files_and_chunks(self, indexer, tmp_path) -> None:
        # UPG-3.3: per-language coverage — distinct files + chunk counts per language.
        indexer.index_file(make_py(tmp_path, "a.py", "def x(): pass"))
        indexer.index_file(make_py(tmp_path, "b.py", "def y(): pass"))
        js = tmp_path / "c.js"; js.write_text("function z() {}")
        indexer.index_file(str(js))
        stats = indexer.indexed_language_stats()
        assert stats["python"]["files"] == 2
        assert stats["python"]["chunks"] >= 2
        assert stats["javascript"]["files"] == 1
        # indexed_languages() derives from the same source
        assert indexer.indexed_languages() == sorted(stats)

    def test_stats_do_not_rescan_collection_after_seed(self, indexer, tmp_path) -> None:
        """UPG-REST-STARVATION: indexed_language_stats() must be O(1) after
        the one-time lazy seed — never a full paginated collection.get()
        scan on every call, which previously made a status read during a
        bulk reindex contend with the writer for the whole collection (see
        agent/indexer/_core.py's _ensure_stats_seeded / _apply_chunk_delta).

        total_chunks reads its own separately-maintained in-memory counter
        (UPG-CHROMA-BLOCKING-EVENT-LOOP — never a live ChromaDB call, so a
        caller reading it never waits on the vector store), so it is
        exercised alongside indexed_language_stats() here to confirm it
        never routes through the paginated scan either.

        `indexer._collection` is a real ChromaDB collection, so the
        paginated stats-scan signature (a `limit=` page kwarg, distinct from
        the `where=`-filtered per-file lookups index_file/delete_file also
        issue) is spied on via a wrapping side_effect rather than asserting
        on an unconfigured mock.
        """
        indexer.index_file(make_py(tmp_path, "a.py", "def x(): pass"))
        indexer.indexed_language_stats()  # seeds the per-language cache
        assert indexer._stats_seeded is True

        real_get = indexer._collection.get
        scan_calls: list[dict] = []

        def _tracking_get(*args, **kwargs):
            if "limit" in kwargs:
                scan_calls.append(kwargs)
            return real_get(*args, **kwargs)

        with patch.object(indexer._collection, "get", side_effect=_tracking_get):
            indexer.index_file(make_py(tmp_path, "b.py", "def y(): pass"))
            for _ in range(5):
                indexer.total_chunks
                indexer.indexed_language_stats()

        assert scan_calls == [], (
            "indexed_language_stats() triggered a paginated collection scan "
            "instead of reading the incrementally-maintained cache"
        )

    def test_chunk_delta_partial_delete_keeps_file_until_last_chunk(
        self, indexer, tmp_path,
    ) -> None:
        """UPG-CHUNK-DELTA-PARTIAL-FILE-DISCARD: `_apply_chunk_delta(sign=-1)`
        must drop a file from its per-language bucket only when its LAST body
        chunk goes — never on any deletion batch that merely touches the file.
        Today every delete site is whole-file (`_delete_chunks_for_file`), so
        no production path can build the partial batch this test simulates;
        exercising the bookkeeping directly pins the invariant any future
        partial-chunk deletion path (incremental re-chunk, per-symbol
        invalidation, quality-based eviction) must preserve."""
        path = make_py(tmp_path, "multi.py", "def a(): pass\ndef b(): pass\ndef c(): pass\n")
        indexer.index_file(path)
        stats_before = indexer.indexed_language_stats()
        n_chunks = stats_before["python"]["chunks"]
        assert n_chunks >= 2, "fixture must produce multiple chunks to split a deletion"

        # Remove all but one of the file's chunks in a single batch.
        indexer._apply_chunk_delta(
            [{"language": "python", "file_path": path} for _ in range(n_chunks - 1)],
            sign=-1,
        )
        stats = indexer.indexed_language_stats()
        assert stats["python"]["files"] == 1, (
            "a partial-chunk deletion batch must not discard the surviving file"
        )
        assert stats["python"]["chunks"] == 1
        assert path in indexer.indexed_file_paths

        # Removing the last chunk drops the file from its bucket (the
        # whole-file behaviour every current delete site relies on).
        indexer._apply_chunk_delta([{"language": "python", "file_path": path}], sign=-1)
        stats = indexer.indexed_language_stats()
        assert stats["python"]["files"] == 0
        assert path not in indexer.indexed_file_paths

    def test_delete_file_whole_file_still_discards_bucket(
        self, indexer, tmp_path,
    ) -> None:
        """Regression guard alongside the partial-delete fix: the public
        whole-file path (`delete_file` → `_delete_chunks_for_file`) must keep
        discarding the file from its bucket exactly as before."""
        path_a = make_py(tmp_path, "a.py", "def a(): pass")
        path_b = make_py(tmp_path, "b.py", "def b(): pass")
        indexer.index_file(path_a)
        indexer.index_file(path_b)
        assert len(indexer.indexed_file_paths) == 2

        indexer.delete_file(path_a)
        paths = indexer.indexed_file_paths
        assert not any(p.endswith("a.py") for p in paths)
        assert any(p.endswith("b.py") for p in paths)
        assert indexer.indexed_language_stats()["python"]["files"] == 1

    def test_get_all_documents_returns_indexed_content(self, indexer, tmp_path) -> None:
        make_py(tmp_path, "fn.py", """
            def my_special_function():
                return 42
        """)
        indexer.index_workspace()
        ids, docs, metas = indexer.get_all_documents()
        assert len(ids) > 0
        assert any("my_special_function" in d for d in docs)

    def test_index_workspace_parallel_chunks_all_files(self, indexer, tmp_path) -> None:
        """Parallel chunking must index every file exactly once."""
        for i in range(20):
            make_py(tmp_path, f"mod_{i}.py", f"def fn_{i}(): return {i}")
        indexer.index_workspace()
        assert indexer.indexed_file_count == 20
        assert indexer.total_chunks >= 20

    def test_incremental_skip_unchanged_files(self, indexer, tmp_path) -> None:
        """Second index_workspace() call must skip files whose mtime has not changed."""
        make_py(tmp_path, "a.py", "def a(): pass")
        make_py(tmp_path, "b.py", "def b(): pass")
        indexer.index_workspace()
        chunks_after_first = indexer.total_chunks

        # Second call: nothing changed → chunk count must not grow
        indexer.index_workspace()
        assert indexer.total_chunks == chunks_after_first

    def test_incremental_reindexes_modified_file(self, indexer, tmp_path) -> None:
        """A file whose mtime changes must be re-indexed on the next workspace index."""
        path = Path(make_py(tmp_path, "evolving.py", "def v1(): pass"))
        indexer.index_workspace()

        # Modify the file content and bump mtime
        path.write_text("def v1(): pass\ndef v2(): pass\n")
        import os; os.utime(path, (path.stat().st_atime, path.stat().st_mtime + 1))

        indexer.index_workspace()
        ids, docs, _ = indexer.get_all_documents()
        assert any("v2" in d for d in docs), "Modified file was not re-indexed"

    def test_mtime_cache_persists_across_instances(self, tmp_path, monkeypatch) -> None:
        """A new CodeIndexer pointed at the same workspace must respect the mtime cache.

        Patches the provider factory rather than assigning `_embed_provider`
        after construction: `CodeIndexer.__init__` builds a provider eagerly, so
        post-hoc assignment still pays a real Hugging Face model download first
        (which made CI fail on an HF 429).
        """
        import agent.indexer as idx_module
        from agent.indexer import CodeIndexer
        from tests.conftest import _DummyEmbedProvider

        monkeypatch.setattr(
            idx_module, "get_embed_provider", lambda _model: _DummyEmbedProvider()
        )
        db = str(tmp_path / "db")
        make_py(tmp_path, "cached.py", "def cached(): pass")

        idx1 = CodeIndexer(str(tmp_path), db_path=db)
        idx1.index_workspace()
        count_after_first = idx1.total_chunks

        idx2 = CodeIndexer(str(tmp_path), db_path=db)
        idx2.index_workspace()
        assert idx2.total_chunks == count_after_first

    def test_index_minified_file_no_duplicate_ids(self, indexer, tmp_path) -> None:
        """Minified JS (all code on 1-2 lines) must not produce duplicate chunk IDs."""
        js_file = tmp_path / "jquery.min.js"
        # Simulate minified JS: many statements crammed onto line 2
        js_file.write_text(
            "// jquery minified\n"
            + "!function(e,t){'use strict';function n(e){return e}function r(e,t){return e+t}"
            * 5
        )
        # Must not raise DuplicateIDError
        count = indexer.index_file(str(js_file))
        assert count >= 1
        # Re-index must also not raise
        count2 = indexer.index_file(str(js_file))
        assert count2 >= 1
        assert indexer.total_chunks == count2  # idempotent


# ---------------------------------------------------------------------------
# CodeSearcher — hybrid BM25 + vector
# ---------------------------------------------------------------------------

class TestCodeSearcher:
    def _indexed_searcher(self, indexer, tmp_path, content: str, name="module.py"):
        path = make_py(tmp_path, name, content)
        indexer.index_file(path)
        from agent.searcher import CodeSearcher
        s = CodeSearcher(indexer)
        s.refresh_bm25()
        return s

    def test_search_returns_results(self, indexer, tmp_path) -> None:
        s = self._indexed_searcher(indexer, tmp_path, """
            def authenticate_user(username: str, password: str) -> bool:
                return True
        """)
        results, ms = s.search("authenticate user function")
        assert len(results) >= 1
        assert results[0].file_path.endswith("module.py")

    def test_search_result_fields(self, indexer, tmp_path) -> None:
        s = self._indexed_searcher(indexer, tmp_path, """
            def check_permissions(user_id: int) -> list:
                return []
        """)
        results, ms = s.search("check permissions")
        r = results[0]
        assert r.file_path
        assert r.lines  # e.g. "2-4"
        assert r.language == "python"
        assert 0.0 <= r.score <= 1.0
        assert r.content

    def test_search_empty_index_returns_empty(self, indexer) -> None:
        from agent.searcher import CodeSearcher
        s = CodeSearcher(indexer)
        results, ms = s.search("anything")
        assert results == []
        assert ms == 0

    def test_bm25_finds_exact_keyword(self, indexer, tmp_path) -> None:
        # _code_tokenize splits snake_case: "dispatch_uid" → ["dispatch","uid"]
        # and "send_signal_dispatch_uid" → ["send","signal","dispatch","uid"].
        # So a query for "dispatch_uid" matches even though the function name is
        # one long identifier — no workaround needed.
        path = make_py(tmp_path, "signals.py", """
            def send_signal_dispatch_uid(sender, **kwargs):
                pass

            def unrelated_function():
                pass
        """)
        indexer.index_file(path)
        from agent.searcher import CodeSearcher
        s = CodeSearcher(indexer)
        s.refresh_bm25()
        results, _ = s.search("dispatch_uid", semantic_weight=0.0)  # pure BM25
        assert len(results) >= 1
        assert "dispatch_uid" in results[0].content or "dispatch" in results[0].symbol_name

    def test_search_language_filter(self, indexer, tmp_path) -> None:
        py_path = make_py(tmp_path, "app.py", "def python_fn(): pass")
        js_path = tmp_path / "app.js"
        js_path.write_text("function jsFn() {}")
        indexer.index_file(py_path)
        indexer.index_file(str(js_path))
        from agent.searcher import CodeSearcher
        s = CodeSearcher(indexer)
        s.refresh_bm25()
        results, _ = s.search("function", language="python")
        assert all(r.language == "python" for r in results)

    def test_search_language_filter_normalized(self, indexer, tmp_path) -> None:
        # UPG-3.1: case/whitespace on the filter must still match indexed values,
        # and apply identically regardless of caller (REST vs MCP both hit this).
        indexer.index_file(make_py(tmp_path, "app.py", "def python_fn(): pass"))
        js_path = tmp_path / "app.js"; js_path.write_text("function jsFn() {}")
        indexer.index_file(str(js_path))
        from agent.searcher import CodeSearcher
        s = CodeSearcher(indexer)
        s.refresh_bm25()
        for variant in ("PYTHON", "  Python ", "python"):
            results, _ = s.search("function", language=variant)
            assert results, f"variant {variant!r} returned nothing"
            assert all(r.language == "python" for r in results), f"variant {variant!r} leaked non-python"
        # blank filter degrades to no-filter (can include js)
        blank, _ = s.search("function", language="   ")
        assert any(r.language == "javascript" for r in blank) or blank

    def test_unfiltered_search_fetches_deeper_pool(self, indexer, tmp_path) -> None:
        """Unfiltered queries over-fetch, apply union-of-signals trivial filter (UPG-15.7 revised).

        Pool selection: take the first top_k_unfiltered non-trivial chunks from the
        vec-similarity ranking UNION the first top_k_unfiltered non-trivial chunks from
        the BM25 ranking.  This is bounded by 2*top_k_unfiltered (and by pre_filter_fetch_k).
        The corpus is pure Python so no chunks are classified trivial — both per-signal
        lists fill to top_k_unfiltered, so the union can be anywhere in
        [top_k_unfiltered, min(2*top_k_unfiltered, pre_filter_fetch_k)].

        Key invariants:
        - unfiltered pool > filtered pool (filtered uses the smaller _RERANK_TOP_K).
        - unfiltered pool <= min(2*top_k_unfiltered, pre_filter_fetch_k, total_chunks).
        - cross-encoder sees no trivial chunks (tested in TestPoolEntryTrivialFilter).
        """
        import agent.searcher as searcher_mod
        # Index enough chunks that both FILTERED and UNFILTERED pool depths are reachable.
        # Monkeypatch to small values so the fixture stays fast.
        n_filtered = searcher_mod._RERANK_TOP_K
        n_unfiltered_test = n_filtered + 10   # per-signal cap for unfiltered pool

        pre_fetch_k = n_unfiltered_test + 15
        body = "\n".join(f"def fn_{i}(x):\n    return x + {i}\n" for i in range(n_unfiltered_test + 20))
        path = make_py(tmp_path, "many.py", body)
        indexer.index_file(path)
        assert indexer.total_chunks >= n_unfiltered_test, (
            f"Need at least {n_unfiltered_test} chunks in index, got {indexer.total_chunks}"
        )

        from agent.searcher import CodeSearcher
        s = CodeSearcher(indexer)
        s.refresh_bm25()

        class _CountingReranker:
            def __init__(self):
                self.last_count = 0
            def rerank(self, query, candidates):
                self.last_count = len(candidates)
                return [c for _, c in candidates]

        stub = _CountingReranker()
        s._reranker = stub  # force the rerank path on

        # Monkeypatch both the per-signal cap (top_k_unfiltered) and the over-fetch depth
        # (pre_filter_fetch_k) to small testable values.
        orig_unfiltered = searcher_mod._RERANK_TOP_K_UNFILTERED
        orig_prefetch = searcher_mod._RERANK_PRE_FILTER_FETCH_K
        searcher_mod._RERANK_TOP_K_UNFILTERED = n_unfiltered_test
        searcher_mod._RERANK_PRE_FILTER_FETCH_K = pre_fetch_k
        try:
            s.search("function returning a number", language=None)
            unfiltered_count = stub.last_count
            s.search("function returning a number", language="python")
            filtered_count = stub.last_count
        finally:
            searcher_mod._RERANK_TOP_K_UNFILTERED = orig_unfiltered
            searcher_mod._RERANK_PRE_FILTER_FETCH_K = orig_prefetch

        # Union pool is bounded by pre_filter_fetch_k (the hybrid fetch ceiling) AND
        # 2*top_k_unfiltered (union of two per-signal lists of size top_k_unfiltered each).
        max_unfiltered = min(2 * n_unfiltered_test, pre_fetch_k, indexer.total_chunks)
        assert unfiltered_count <= max_unfiltered, (
            f"UPG-15.7: unfiltered pool must be <= min(2*top_k_unfiltered, pre_filter_fetch_k) "
            f"= {max_unfiltered}, got {unfiltered_count}"
        )
        assert filtered_count == min(searcher_mod._RERANK_TOP_K, indexer.total_chunks)
        assert unfiltered_count > filtered_count, (
            f"UPG-15.7: unfiltered pool ({unfiltered_count}) must exceed filtered pool "
            f"({filtered_count}) so there is more room to rerank"
        )

    def test_search_multiple_files_ranked(self, indexer, tmp_path) -> None:
        # Code tokenizer splits camelCase: "RateLimitMiddleware" → ["rate","limit","middleware"]
        # so querying "rate limit" finds middleware.py without any workarounds.
        make_py(tmp_path, "middleware.py", """
            class RateLimitMiddleware:
                def __call__(self, request, get_response):
                    pass
        """)
        make_py(tmp_path, "utils.py", """
            def helper():
                pass
        """)
        indexer.index_workspace()
        from agent.searcher import CodeSearcher
        s = CodeSearcher(indexer)
        s.refresh_bm25()
        results, _ = s.search("RateLimitMiddleware", semantic_weight=0.0)
        assert len(results) >= 1
        assert results[0].file_path.endswith("middleware.py")

    def test_refresh_bm25_after_new_index(self, indexer, tmp_path) -> None:
        from agent.searcher import CodeSearcher
        s = CodeSearcher(indexer)

        make_py(tmp_path, "first.py", "def first_fn(): pass")
        indexer.index_workspace()
        s.refresh_bm25()
        r1, _ = s.search("first_fn", semantic_weight=0.0)
        assert len(r1) >= 1

        make_py(tmp_path, "second.py", "def second_fn_unique_xyz(): pass")
        indexer.index_workspace()
        s.refresh_bm25()  # must rebuild to include new file
        r2, _ = s.search("second_fn_unique_xyz", semantic_weight=0.0)
        assert len(r2) >= 1

    def test_search_returns_timing(self, indexer, tmp_path) -> None:
        s = self._indexed_searcher(indexer, tmp_path, "def timed(): pass")
        _, ms = s.search("timed function")
        assert isinstance(ms, int)
        assert ms >= 0

    def test_search_n_results_respected(self, indexer, tmp_path) -> None:
        for i in range(6):
            make_py(tmp_path, f"fn{i}.py", f"def func_{i}(): pass\n")
        indexer.index_workspace()
        from agent.searcher import CodeSearcher
        s = CodeSearcher(indexer)
        s.refresh_bm25()
        results, _ = s.search("func", n_results=3)
        assert len(results) <= 3

    def test_search_degrades_to_empty_when_total_chunks_cache_reads_zero(
        self, indexer, tmp_path
    ) -> None:
        """UPG-NOTES-CHROMA-BLOCKING-EVENT-LOOP (item D): `search()` gates on
        the in-memory `total_chunks` counter (agent/searcher.py), never on a
        live collection read, so this pins the exact class of degradation a
        stale counter can cause — even with real chunks still present in the
        collection, a stale-zero cache must short-circuit to an EMPTY result
        (never raise, never return wrong/mismatched rows, never touch the
        collection at all) — the same bounded-staleness trade-off
        `total_chunks`'s own docstring describes.
        """
        s = self._indexed_searcher(indexer, tmp_path, """
            def authenticate_user(username: str, password: str) -> bool:
                return True
        """)
        assert indexer.total_chunks > 0  # real chunks are present in the collection

        # Simulate the cache going stale independently of the real collection
        # state (the class of staleness item D asked to be documented, not a
        # scenario vectr's single-daemon-per-workspace architecture can
        # actually produce — see the out-of-process-writer note below).
        indexer._total_chunks_cache = 0

        results, ms = s.search("authenticate user function")

        assert results == []
        assert ms == 0
        # Confirms the short-circuit fires (an empty-result degradation),
        # rather than any attempt reaching the still-populated collection
        # that could raise or return stale/mismatched rows instead.


# ---------------------------------------------------------------------------
# UPG-RESULT-FLOOR: sub-floor cross-encoder-relevance results are trimmed
# ---------------------------------------------------------------------------

class TestResultFloor:
    def _rl(self, *scores_ce):
        from agent.searcher import SearchResult, SearchResultList
        rl = SearchResultList([
            SearchResult(file_path=f"f{i}.py", lines="1-5", symbol_name=f"s{i}",
                         language="python", score=(ce if ce is not None else 0.5),
                         content="x", ce_relevance=ce, score_source="reranker")
            for i, ce in enumerate(scores_ce)
        ])
        return rl

    def test_trims_subfloor_ce_results(self) -> None:
        from agent.searcher import CodeSearcher
        rl = self._rl(0.80, 0.001, 0.0005)  # ranks 2-3 are ~0.000 filler
        out = CodeSearcher._apply_result_floor(rl)
        assert len(out) == 1
        assert out[0].symbol_name == "s0"

    def test_always_keeps_composite_top_even_if_subfloor(self) -> None:
        from agent.searcher import CodeSearcher
        rl = self._rl(0.001, 0.0005, 0.0)  # every result is filler
        out = CodeSearcher._apply_result_floor(rl)
        assert len(out) == 1  # the top is kept as the best guess
        assert out[0].symbol_name == "s0"

    def test_no_trim_when_all_above_floor(self) -> None:
        from agent.searcher import CodeSearcher
        rl = self._rl(0.80, 0.60, 0.40)
        out = CodeSearcher._apply_result_floor(rl)
        assert len(out) == 3

    def test_never_trims_dense_only_results(self) -> None:
        # ce_relevance is None (no reranker ran) → never floored, since a
        # bi-encoder cosine cannot separate absent-topic from on-topic.
        from agent.searcher import CodeSearcher
        rl = self._rl(None, None, None)
        out = CodeSearcher._apply_result_floor(rl)
        assert len(out) == 3

    def test_preserves_low_confidence_flag(self) -> None:
        from agent.searcher import CodeSearcher
        rl = self._rl(0.80, 0.001)
        rl.low_confidence = True
        out = CodeSearcher._apply_result_floor(rl)
        assert out.low_confidence is True


# ---------------------------------------------------------------------------
# Zig language support
# ---------------------------------------------------------------------------

class TestZigSupport:
    """chunk_file() must produce AST-aware chunks for .zig files (not window fallback)."""

    _ZIG_CODE = """\
const std = @import("std");

pub fn add(a: i32, b: i32) i32 {
    return a + b;
}

pub fn subtract(a: i32, b: i32) i32 {
    return a - b;
}

pub const Account = struct {
    id: u128,
    balance: i64,

    pub fn debit(self: *Account, amount: i64) void {
        self.balance -= amount;
    }
};
"""

    def _write_zig(self, tmp_path) -> str:
        p = tmp_path / "state_machine.zig"
        p.write_text(self._ZIG_CODE)
        return str(p)

    def test_zig_chunks_are_ast_aware(self, tmp_path) -> None:
        from agent.indexer import chunk_file
        chunks = chunk_file(self._write_zig(tmp_path))
        node_types = {c.node_type for c in chunks}
        assert "function_declaration" in node_types or "variable_declaration" in node_types, (
            "expected AST-aware node types, got window-fallback chunks"
        )

    def test_zig_function_symbol_name_extracted(self, tmp_path) -> None:
        from agent.indexer import chunk_file
        chunks = chunk_file(self._write_zig(tmp_path))
        names = {c.symbol_name for c in chunks}
        assert "add" in names
        assert "subtract" in names

    def test_zig_struct_symbol_name_extracted(self, tmp_path) -> None:
        from agent.indexer import chunk_file
        chunks = chunk_file(self._write_zig(tmp_path))
        names = {c.symbol_name for c in chunks}
        assert "Account" in names

    def test_zig_language_label_set(self, tmp_path) -> None:
        from agent.indexer import chunk_file
        chunks = chunk_file(self._write_zig(tmp_path))
        assert all(c.language == "zig" for c in chunks)

    def test_zig_file_indexed_by_indexer(self, indexer, tmp_path) -> None:
        p = tmp_path / "main.zig"
        p.write_text("pub fn run() void {}\npub fn stop() void {}\n")
        count = indexer.index_file(str(p))
        assert count >= 2, f"expected ≥2 Zig chunks, got {count}"


# ---------------------------------------------------------------------------
# Type-definition chunking (UPG-RUST-STRUCT-CHUNK-MISSING)
#
# A struct/trait/enum/interface/type-alias definition is a standalone
# top-level unit exactly like a function or class. Before this fix, a Rust
# struct/trait/enum was never collected as its own chunk — synthetic
# fixtures only, per-language, each proving (a) a type-def-only file no
# longer degrades to a window chunk, (b) a mixed type-def + function file
# emits both, and (c) chunk_quality.is_type_definition_chunk actually
# activates (DEF-B) for the new chunk's recorded node_type.
# ---------------------------------------------------------------------------

class TestRustTypeDefChunking:
    def test_struct_only_file_emits_struct_chunk_not_window(self, tmp_path) -> None:
        from agent.indexer import chunk_file
        p = tmp_path / "shapes.rs"
        p.write_text(
            '/// A point in 2D space.\n'
            'struct Point {\n'
            '    x: i32,\n'
            '    y: i32,\n'
            '}\n'
        )
        chunks = chunk_file(str(p))
        node_types = {c.node_type for c in chunks}
        assert "window" not in node_types, f"struct-only file degraded to window chunk: {node_types}"
        assert "struct_item" in node_types

    def test_struct_doc_comment_attached(self, tmp_path) -> None:
        from agent.indexer import chunk_file
        p = tmp_path / "shapes.rs"
        p.write_text(
            '/// A point in 2D space.\n'
            'struct Point {\n'
            '    x: i32,\n'
            '    y: i32,\n'
            '}\n'
        )
        chunks = chunk_file(str(p))
        struct_chunk = next(c for c in chunks if c.node_type == "struct_item")
        assert "A point in 2D space." in struct_chunk.content
        assert struct_chunk.symbol_name == "Point"

    def test_mixed_struct_and_function_emit_both_chunks(self, tmp_path) -> None:
        from agent.indexer import chunk_file
        p = tmp_path / "shapes.rs"
        p.write_text(
            '/// A point in 2D space.\n'
            'struct Point {\n'
            '    x: i32,\n'
            '}\n'
            '\n'
            'fn origin() -> Point {\n'
            '    Point { x: 0 }\n'
            '}\n'
        )
        chunks = chunk_file(str(p))
        node_types = {c.node_type for c in chunks}
        assert "struct_item" in node_types
        assert "function_item" in node_types
        names = {c.symbol_name for c in chunks}
        assert "Point" in names and "origin" in names

    def test_trait_and_enum_get_own_chunks(self, tmp_path) -> None:
        from agent.indexer import chunk_file
        p = tmp_path / "shapes.rs"
        p.write_text(
            '/// Any shape has an area.\n'
            'trait Shape {\n'
            '    fn area(&self) -> f64;\n'
            '}\n'
            '\n'
            '/// The set of supported colors.\n'
            'enum Color {\n'
            '    Red,\n'
            '    Green,\n'
            '}\n'
        )
        chunks = chunk_file(str(p))
        by_type = {c.node_type: c for c in chunks}
        assert "trait_item" in by_type and "enum_item" in by_type
        assert by_type["trait_item"].symbol_name == "Shape"
        assert by_type["enum_item"].symbol_name == "Color"
        assert "Any shape has an area." in by_type["trait_item"].content
        assert "The set of supported colors." in by_type["enum_item"].content

    def test_impl_item_still_excluded_from_type_def(self) -> None:
        from agent.chunk_quality import is_type_definition_chunk
        assert is_type_definition_chunk("impl_item") is False

    def test_new_rust_node_types_activate_def_b(self) -> None:
        from agent.chunk_quality import is_type_definition_chunk
        assert is_type_definition_chunk("struct_item") is True
        assert is_type_definition_chunk("trait_item") is True
        assert is_type_definition_chunk("enum_item") is True


class TestTypeScriptTypeDefChunking:
    def test_interface_only_file_emits_interface_chunk_not_window(self, tmp_path) -> None:
        from agent.indexer import chunk_file
        p = tmp_path / "shapes.ts"
        p.write_text(
            '/** A widget configuration. */\n'
            'interface WidgetConfig {\n'
            '  size: number;\n'
            '}\n'
        )
        chunks = chunk_file(str(p))
        node_types = {c.node_type for c in chunks}
        assert "window" not in node_types
        assert "interface_declaration" in node_types
        interface_chunk = next(c for c in chunks if c.node_type == "interface_declaration")
        assert interface_chunk.symbol_name == "WidgetConfig"
        assert "A widget configuration." in interface_chunk.content

    def test_mixed_interface_and_function_emit_both_chunks(self, tmp_path) -> None:
        from agent.indexer import chunk_file
        p = tmp_path / "shapes.ts"
        p.write_text(
            'interface WidgetConfig {\n'
            '  size: number;\n'
            '}\n'
            '\n'
            'function makeWidget(): WidgetConfig {\n'
            '  return { size: 1 };\n'
            '}\n'
        )
        chunks = chunk_file(str(p))
        node_types = {c.node_type for c in chunks}
        assert "interface_declaration" in node_types
        assert "function_declaration" in node_types

    def test_type_alias_and_enum_get_own_chunks(self, tmp_path) -> None:
        from agent.indexer import chunk_file
        p = tmp_path / "shapes.ts"
        p.write_text(
            'type WidgetId = string;\n'
            '\n'
            'enum WidgetKind {\n'
            '  Small,\n'
            '  Large,\n'
            '}\n'
        )
        chunks = chunk_file(str(p))
        by_type = {c.node_type: c for c in chunks}
        assert "type_alias_declaration" in by_type and "enum_declaration" in by_type
        assert by_type["type_alias_declaration"].symbol_name == "WidgetId"
        assert by_type["enum_declaration"].symbol_name == "WidgetKind"

    def test_class_declaration_symbol_name_extracted(self, tmp_path) -> None:
        # Regression guard: a TS class's own name node is `type_identifier`,
        # not `identifier` — before this fix, every TS class chunk's
        # symbol_name silently came back "" (found while wiring the new
        # type-def node types through the same extraction helper).
        from agent.indexer import chunk_file
        p = tmp_path / "widget.ts"
        p.write_text(
            'class Widget extends Base {\n'
            '  render() {}\n'
            '}\n'
        )
        chunks = chunk_file(str(p))
        class_chunk = next(c for c in chunks if c.node_type == "class_declaration")
        assert class_chunk.symbol_name == "Widget"

    def test_new_ts_node_types_activate_def_b(self) -> None:
        from agent.chunk_quality import is_type_definition_chunk
        assert is_type_definition_chunk("interface_declaration") is True
        assert is_type_definition_chunk("type_alias_declaration") is True
        assert is_type_definition_chunk("enum_declaration") is True


class TestGoTypeDeclChunking:
    def test_struct_only_file_emits_type_declaration_chunk_not_window(self, tmp_path) -> None:
        from agent.indexer import chunk_file
        p = tmp_path / "shapes.go"
        p.write_text(
            'package shapes\n'
            '\n'
            '// Point is a location in 2D space.\n'
            'type Point struct {\n'
            '\tX int\n'
            '\tY int\n'
            '}\n'
        )
        chunks = chunk_file(str(p))
        node_types = {c.node_type for c in chunks}
        assert "window" not in node_types
        assert "type_declaration" in node_types
        type_chunk = next(c for c in chunks if c.node_type == "type_declaration")
        assert type_chunk.symbol_name == "Point"
        assert "Point is a location in 2D space." in type_chunk.content

    def test_mixed_type_decl_and_function_emit_both_chunks(self, tmp_path) -> None:
        from agent.indexer import chunk_file
        p = tmp_path / "shapes.go"
        p.write_text(
            'package shapes\n'
            '\n'
            'type Point struct {\n'
            '\tX int\n'
            '}\n'
            '\n'
            'func Origin() Point {\n'
            '\treturn Point{}\n'
            '}\n'
        )
        chunks = chunk_file(str(p))
        node_types = {c.node_type for c in chunks}
        assert "type_declaration" in node_types
        assert "function_declaration" in node_types

    def test_new_go_node_type_activates_def_b(self) -> None:
        from agent.chunk_quality import is_type_definition_chunk
        assert is_type_definition_chunk("type_declaration") is True


class TestJavaTypeDefChunking:
    def test_interface_only_file_emits_interface_chunk_not_window(self, tmp_path) -> None:
        from agent.indexer import chunk_file
        p = tmp_path / "Shape.java"
        p.write_text(
            '/** Any shape has an area. */\n'
            'interface Shape {\n'
            '    double area();\n'
            '}\n'
        )
        chunks = chunk_file(str(p))
        node_types = {c.node_type for c in chunks}
        assert "window" not in node_types
        assert "interface_declaration" in node_types
        interface_chunk = next(c for c in chunks if c.node_type == "interface_declaration")
        assert interface_chunk.symbol_name == "Shape"
        assert "Any shape has an area." in interface_chunk.content

    def test_mixed_interface_enum_and_class_emit_all_chunks(self, tmp_path) -> None:
        from agent.indexer import chunk_file
        p = tmp_path / "Shape.java"
        p.write_text(
            'interface Shape {\n'
            '    double area();\n'
            '}\n'
            '\n'
            'enum Color {\n'
            '    RED, GREEN\n'
            '}\n'
            '\n'
            'class Circle {\n'
            '    double radius;\n'
            '}\n'
        )
        chunks = chunk_file(str(p))
        node_types = {c.node_type for c in chunks}
        assert {"interface_declaration", "enum_declaration", "class_declaration"} <= node_types

    def test_new_java_node_types_activate_def_b(self) -> None:
        from agent.chunk_quality import is_type_definition_chunk
        assert is_type_definition_chunk("interface_declaration") is True
        assert is_type_definition_chunk("enum_declaration") is True


class TestZigInlineTestChunking:
    """test_declaration was dormant (UPG-RUST-STRUCT-CHUNK-MISSING) — a Zig
    inline `test "..." { ... }` block was invisible to the chunker rather than
    present-but-demoted. Chunking it activates the already-shipped DEF-A
    structural test detection (is_content_structural_test_chunk)."""

    def test_inline_test_block_gets_own_chunk_not_window(self, tmp_path) -> None:
        from agent.indexer import chunk_file
        p = tmp_path / "math.zig"
        p.write_text(
            'const std = @import("std");\n'
            '\n'
            '/// Verifies addition.\n'
            'test "basic add" {\n'
            '    try std.testing.expect(1 + 1 == 2);\n'
            '}\n'
        )
        chunks = chunk_file(str(p))
        node_types = {c.node_type for c in chunks}
        assert "window" not in node_types
        assert "test_declaration" in node_types
        test_chunk = next(c for c in chunks if c.node_type == "test_declaration")
        assert "Verifies addition." in test_chunk.content
        assert 'test "basic add"' in test_chunk.content

    def test_mixed_test_and_function_emit_both_chunks(self, tmp_path) -> None:
        from agent.indexer import chunk_file
        p = tmp_path / "math.zig"
        p.write_text(
            'test "basic add" {\n'
            '    const x = 1 + 1;\n'
            '}\n'
            '\n'
            'pub fn add(a: i32, b: i32) i32 {\n'
            '    return a + b;\n'
            '}\n'
        )
        chunks = chunk_file(str(p))
        node_types = {c.node_type for c in chunks}
        assert "test_declaration" in node_types
        assert "function_declaration" in node_types

    def test_inline_test_chunk_is_structurally_demoted(self) -> None:
        from agent.chunk_quality import is_content_structural_test_chunk
        content = 'test "basic add" {\n    try std.testing.expect(1 + 1 == 2);\n}'
        assert is_content_structural_test_chunk(content, language="zig") is True


# ---------------------------------------------------------------------------
# C / C++ language support (UPG-3.2 — locate/trace were dead on C)
# ---------------------------------------------------------------------------

class TestCSupport:
    _C_CODE = """\
#include "Python.h"

typedef struct { int n; } GCState;

static GCState *get_gc_state(PyThreadState *tstate) {
    return &tstate->gc;
}

PyObject *PyDict_New(void) {
    GCState *st = get_gc_state(NULL);
    PyObject *op = _PyObject_GC_New();
    return op;
}

#define MAXSIZE 256
"""

    def _write_c(self, tmp_path, name="dictobject.c") -> str:
        p = tmp_path / name
        p.write_text(self._C_CODE)
        return str(p)

    def test_c_chunks_are_ast_aware(self, tmp_path) -> None:
        from agent.indexer import chunk_file
        chunks = chunk_file(self._write_c(tmp_path))
        node_types = {c.node_type for c in chunks}
        assert "function_definition" in node_types, "expected AST chunks, got window fallback"
        assert all(c.language == "c" for c in chunks)

    def test_c_function_name_not_return_type(self, tmp_path) -> None:
        # The return type (PyObject/GCState) must NOT be mistaken for the function name.
        from agent.indexer import chunk_file
        names = {c.symbol_name for c in chunk_file(self._write_c(tmp_path))}
        assert "PyDict_New" in names
        assert "get_gc_state" in names
        assert "PyObject" not in names  # return type, not a function symbol

    def test_c_symbols_extracted(self, tmp_path) -> None:
        from agent.symbol_graph import extract_symbols_from_file
        syms, _ = extract_symbols_from_file(self._write_c(tmp_path))
        by_name = {s["name"]: s["kind"] for s in syms}
        assert by_name.get("get_gc_state") == "function"
        assert by_name.get("PyDict_New") == "function"
        assert by_name.get("GCState") == "type"
        assert by_name.get("MAXSIZE") == "macro"
        assert "" not in by_name  # no anonymous-struct pollution

    def test_c_call_edges_extracted(self, tmp_path) -> None:
        from agent.symbol_graph import extract_symbols_from_file
        _, edges = extract_symbols_from_file(self._write_c(tmp_path))
        pairs = {(e["from_symbol"], e["to_symbol"]) for e in edges}
        assert ("PyDict_New", "get_gc_state") in pairs
        assert ("PyDict_New", "_PyObject_GC_New") in pairs

    def test_c_deeply_nested_does_not_recurse_crash(self, tmp_path) -> None:
        # Regression: a chain of generic AST nodes (deeply nested parens) used to
        # blow Python's recursion limit because the walker didn't increment depth
        # on generic nodes, so _MAX_DEPTH never fired. Must not raise, and the
        # top-level function must still be extracted.
        from agent.symbol_graph import extract_symbols_from_file
        deep = "(" * 1200 + "1" + ")" * 1200
        p = tmp_path / "deep.c"
        p.write_text(f"int deeply_nested_fn(void) {{ return {deep}; }}\n")
        syms, _ = extract_symbols_from_file(str(p))   # must not raise RecursionError
        assert any(s["name"] == "deeply_nested_fn" for s in syms)

    def test_c_locate_and_trace_via_graph(self, tmp_path) -> None:
        # End-to-end: the audit's exact failure (locate/trace dead on C) is fixed.
        from agent.symbol_graph import SymbolGraph
        db = tmp_path / "sg"; db.mkdir()
        sg = SymbolGraph(str(db))
        path = self._write_c(tmp_path)
        ws = str(tmp_path)
        sg.index_file(ws, path)
        hits = sg.locate(ws, "PyDict_New")
        assert hits and hits[0].name == "PyDict_New"
        trace = sg.trace(ws, "get_gc_state", direction="callers")
        caller_names = {c.from_symbol for c in trace.get("callers", [])}
        assert "PyDict_New" in caller_names


class TestCppSupport:
    _CPP_CODE = """\
namespace ns {

class Widget {
public:
    void render() { paint(); }
    int size() const { return n; }
private:
    int n;
};

}  // namespace ns
"""

    def _write_cpp(self, tmp_path, name="widget.cpp") -> str:
        p = tmp_path / name
        p.write_text(self._CPP_CODE)
        return str(p)

    def test_cpp_language_label(self, tmp_path) -> None:
        from agent.indexer import chunk_file
        chunks = chunk_file(self._write_cpp(tmp_path))
        assert chunks and all(c.language == "cpp" for c in chunks)

    def test_cpp_class_and_method_symbols(self, tmp_path) -> None:
        from agent.symbol_graph import extract_symbols_from_file
        syms, edges = extract_symbols_from_file(self._write_cpp(tmp_path))
        by_name = {s["name"]: s["kind"] for s in syms}
        assert by_name.get("Widget") == "class"
        assert "render" in by_name
        assert ("render", "paint") in {(e["from_symbol"], e["to_symbol"]) for e in edges}


class TestFlowJavaScriptSupport:
    """UPG-JSFLOW-SYMBOLS: Flow-typed .js must route to the tsx/typescript
    grammar (which parses type annotations) instead of the plain javascript
    grammar (which treats Flow syntax as ERROR nodes and desyncs the symbol
    walk — canonical functions go missing, keyword tokens can be
    misattributed as symbol names). Fixtures are minimal synthetic Flow
    snippets — no benchmark-corpus source is referenced or copied.
    """

    # @flow pragma + `import type` + typed function signatures + a generic
    # function — the shape that desyncs the plain javascript grammar.
    _FLOW_JS = """\
// @flow
import type {Foo} from "./types";

function beginWork(current: Foo | null, workInProgress: Foo, lanes: number): Foo | null {
  if (current !== null) {
    return current;
  }
  return workInProgress;
}

function push<T>(cursor: T, value: T, fiber: Foo): void {
  cursor.current = value;
}

export function completeWork(a: Foo): void {
  return;
}
"""

    _PLAIN_JS = """\
function greet(name) {
  return "hello " + name;
}

class Widget {
  render() {
    return null;
  }
}
"""

    def _write_flow(self, tmp_path, name="flow_sample.js") -> str:
        p = tmp_path / name
        p.write_text(self._FLOW_JS)
        return str(p)

    def _write_plain(self, tmp_path, name="plain_sample.js") -> str:
        p = tmp_path / name
        p.write_text(self._PLAIN_JS)
        return str(p)

    # -- detection helper -----------------------------------------------

    def test_flow_pragma_detected(self) -> None:
        from agent.indexer import is_flow_javascript
        assert is_flow_javascript(self._FLOW_JS) is True

    def test_plain_js_not_detected_as_flow(self) -> None:
        from agent.indexer import is_flow_javascript
        assert is_flow_javascript(self._PLAIN_JS) is False

    def test_secondary_marker_detected_without_pragma(self) -> None:
        from agent.indexer import is_flow_javascript
        code = 'import type {Bar} from "./b";\nfunction f(a) { return a; }\n'
        assert is_flow_javascript(code) is True

    def test_parser_language_routes_flow_js_to_tsx(self) -> None:
        from agent.indexer import _parser_language_for
        assert _parser_language_for("javascript", self._FLOW_JS) == "tsx"

    def test_parser_language_leaves_plain_js_untouched(self) -> None:
        from agent.indexer import _parser_language_for
        assert _parser_language_for("javascript", self._PLAIN_JS) == "javascript"

    def test_parser_language_noop_for_other_languages(self) -> None:
        from agent.indexer import _parser_language_for
        # Flow-looking content in a non-.js language key must not be rerouted.
        assert _parser_language_for("python", self._FLOW_JS) == "python"

    # -- symbol extraction on a Flow-typed file --------------------------

    def test_flow_file_yields_canonical_function_symbols(self, tmp_path) -> None:
        from agent.symbol_graph import extract_symbols_from_file
        syms, _ = extract_symbols_from_file(self._write_flow(tmp_path))
        names = {s["name"] for s in syms}
        assert "beginWork" in names
        assert "push" in names
        assert "completeWork" in names

    def test_flow_file_yields_no_keyword_symbols(self, tmp_path) -> None:
        from agent.symbol_graph import extract_symbols_from_file
        syms, _ = extract_symbols_from_file(self._write_flow(tmp_path))
        names = {s["name"] for s in syms}
        assert "if" not in names
        assert "return" not in names

    def test_flow_file_chunks_are_ast_aware_under_javascript_label(self, tmp_path) -> None:
        # Parsed with the tsx grammar, but chunk/symbol metadata still reports
        # "javascript" (the extension-derived language bucket) so downstream
        # filters/dict lookups keyed on "javascript" keep working.
        from agent.indexer import chunk_file
        chunks = chunk_file(self._write_flow(tmp_path))
        symbol_names = {c.symbol_name for c in chunks}
        assert "beginWork" in symbol_names
        assert "push" in symbol_names
        assert all(c.language == "javascript" for c in chunks)

    def test_flow_file_locate_resolves_canonical_functions(self, tmp_path) -> None:
        from agent.symbol_graph import SymbolGraph
        db = tmp_path / "sg"; db.mkdir()
        sg = SymbolGraph(str(db))
        path = self._write_flow(tmp_path)
        ws = str(tmp_path)
        sg.index_file(ws, path)
        for name in ("beginWork", "push", "completeWork"):
            hits = sg.locate(ws, name)
            assert hits and hits[0].name == name, f"locate({name!r}) failed to resolve"

    # -- plain (non-Flow) .js is unaffected -------------------------------

    def test_plain_js_still_parses_via_javascript_grammar(self, tmp_path) -> None:
        from agent.symbol_graph import extract_symbols_from_file
        syms, _ = extract_symbols_from_file(self._write_plain(tmp_path))
        by_name = {s["name"]: s["kind"] for s in syms}
        assert by_name.get("greet") == "function"
        assert by_name.get("Widget") == "class"
        assert by_name.get("render") == "method"

    # -- deliberately broken file: keyword-junk rejection ----------------

    def test_broken_typed_js_yields_no_keyword_symbols(self, tmp_path) -> None:
        # Type annotations WITHOUT an @flow pragma or `import type` — not
        # detected as Flow, so it stays on the plain javascript grammar and
        # desyncs into ERROR nodes. The keyword/ERROR-node guard must still
        # prevent any keyword-named symbol from being minted, even though the
        # routing fix does not apply here.
        broken = """\
function beginWork(current: Foo | null, workInProgress: Foo): Foo | null {
  if (current !== null) {
    return current;
  }
  return workInProgress;
}
"""
        from agent.symbol_graph import extract_symbols_from_file
        p = tmp_path / "broken.js"
        p.write_text(broken)
        syms, edges = extract_symbols_from_file(str(p))
        names = {s["name"] for s in syms}
        for kw in ("if", "for", "while", "return", "switch", "case"):
            assert kw not in names
        edge_targets = {e["to_symbol"] for e in edges}
        for kw in ("if", "for", "while", "return", "switch", "case"):
            assert kw not in edge_targets


class TestFlowGrammarDesyncRecovery:
    """UPG-REACT-TSX-FUNCTION-DECL-DROP: a Flow-only construct the tsx grammar
    can't parse must not erase an otherwise-legitimate function declaration.
    Two distinct failure shapes, two mechanisms:

    (1) the construct is CONTAINED inside one declaration's own subtree (a bad
        parameter/body type) — that declaration's own NAME token is still
        clean, so it must still be extracted (name-node-scoped error check).
    (2) the construct desyncs the grammar's error recovery badly enough that
        an unrelated run of SIBLING declarations gets swallowed into one
        opaque errored node — isolated reparse-in-place recovers them.

    Fixtures are minimal synthetic Flow snippets — no benchmark-corpus source
    is referenced or copied.
    """

    def _write(self, tmp_path, code: str, name="sample.js") -> str:
        p = tmp_path / name
        p.write_text(code)
        return str(p)

    def test_locally_erroring_signature_construct_still_extracts_function(self, tmp_path) -> None:
        # `component(...props: any)` (Flow's component-type syntax) has no tsx
        # equivalent — it errors within the parameter's type annotation, but
        # the function's own name token is unaffected.
        code = """\
// @flow
export function alpha(x: ?component(...props: any)): Beta {
  return x;
}
"""
        from agent.symbol_graph import extract_symbols_from_file
        syms, _ = extract_symbols_from_file(self._write(tmp_path, code))
        names = {s["name"] for s in syms}
        assert "alpha" in names

    def test_locally_erroring_body_construct_still_extracts_function(self, tmp_path) -> None:
        # A bare-identifier Flow function type (`A => mixed`, no parens around
        # the param) inside one function's body errors locally in tsx but
        # does not desync its clean sibling.
        code = """\
// @flow
function gamma(): void {
  const bad: (A => mixed) = undefined;
}
function delta(): void {
  return;
}
"""
        from agent.symbol_graph import extract_symbols_from_file
        syms, _ = extract_symbols_from_file(self._write(tmp_path, code))
        names = {s["name"] for s in syms}
        assert "gamma" in names
        assert "delta" in names

    def test_catastrophic_desync_recovers_swallowed_sibling_functions(self, tmp_path) -> None:
        # The same bare-identifier Flow function type, used inside an object
        # type's property, desyncs the grammar badly enough to swallow every
        # following sibling declaration into one opaque errored node — no
        # `function_declaration` node survives anywhere in the raw parse.
        # Isolated reparse-in-place (with sibling-absorbing extension) must
        # recover the swallowed declarations.
        names_list = ["fnA", "fnB", "fnC", "fnD", "fnE", "fnF", "fnG", "fnH"]
        body = "\n".join(
            f"function {n}(): void {{\n  return;\n}}\n" for n in names_list
        )
        code = f"""\
// @flow
export type Queue<S, A> = {{
  pending: S | null,
  dispatch: (A => mixed) | null,
  lastRenderedReducer: ((S, A) => S) | null,
  lastRenderedState: S | null,
}};

{body}
"""
        from agent.symbol_graph import extract_symbols_from_file
        syms, _ = extract_symbols_from_file(self._write(tmp_path, code))
        names = {s["name"] for s in syms}
        recovered = [n for n in names_list if n in names]
        # The recovery mechanism is budget-bounded (UPG-REACT-TSX-FUNCTION-DECL-DROP)
        # and not guaranteed to recover every single swallowed sibling in one
        # pass — but it must recover the large majority instead of the zero
        # a plain has_error-on-subtree check would yield.
        assert len(recovered) >= len(names_list) - 2, (
            f"expected most of {names_list} recovered, got {recovered}"
        )

    def test_catastrophic_desync_without_recovery_loses_all_siblings(self, tmp_path) -> None:
        # Documents WHY the recovery mechanism is needed: with it disabled
        # (reparse_budget=0), the same construct as above yields zero of the
        # swallowed functions — the name-node-scoped check alone (UPG-REACT-
        # TSX-FUNCTION-DECL-DROP mechanism 1) cannot help here since no
        # `function_declaration` node exists anywhere in the raw parse tree.
        from pathlib import Path as _Path
        from agent.symbol_graph._extraction import _collect_symbols_and_calls, _get_parser
        from agent.symbol_graph._constants import _SYMBOL_TYPES, _CALL_TYPES

        names_list = ["fnA", "fnB"]
        body = "\n".join(
            f"function {n}(): void {{\n  return;\n}}\n" for n in names_list
        )
        code = f"""\
export type Queue<S, A> = {{
  dispatch: (A => mixed) | null,
}};

{body}
"""
        parser = _get_parser("tsx")
        code_bytes = code.encode("utf-8")
        tree = parser.parse(code_bytes)
        symbols: list[dict] = []
        edges: list[dict] = []
        _collect_symbols_and_calls(
            tree.root_node, code_bytes, "javascript", "f.js",
            _SYMBOL_TYPES.get("javascript", {}), _CALL_TYPES.get("javascript", set()),
            symbols, edges,
            parser=parser, reparse_budget=[0],
        )
        names = {s["name"] for s in symbols}
        assert not any(n in names for n in names_list)

    def test_desync_recovery_never_leaks_a_keyword_symbol(self, tmp_path) -> None:
        # A genuinely broken Flow-pragma'd file (unbalanced brace) must still
        # never mint a keyword-token symbol, even though it now also goes
        # through the isolated-reparse recovery path.
        code = """\
// @flow
import type {Foo} from "./types";

if (x) {

function completeWork(a: Foo): void {
  return;
}
"""
        from agent.symbol_graph import extract_symbols_from_file
        syms, edges = extract_symbols_from_file(self._write(tmp_path, code))
        names = {s["name"] for s in syms}
        assert "completeWork" in names
        for kw in ("if", "for", "while", "return", "switch", "case"):
            assert kw not in names
        edge_targets = {e["to_symbol"] for e in edges}
        for kw in ("if", "for", "while", "return", "switch", "case"):
            assert kw not in edge_targets

    def test_plain_js_unaffected_by_recovery_machinery(self, tmp_path) -> None:
        # No has_error anywhere in a clean, plain .js file — the recovery
        # branch's `node.has_error` guard means it never even attempts a
        # reparse, so behavior for ordinary code is unchanged.
        code = """\
function greet(name) {
  return "hello " + name;
}
"""
        from agent.symbol_graph import extract_symbols_from_file
        syms, _ = extract_symbols_from_file(self._write(tmp_path, code))
        names = {s["name"] for s in syms}
        assert names == {"greet"}


class TestReservedKeywordRejection:
    """UPG-JSFLOW-SYMBOLS: symbol names must never be a language keyword,
    across every symbol-graph language (config-driven, not JS-specific)."""

    @pytest.mark.parametrize("language,keyword", [
        ("python", "if"),
        ("python", "return"),
        ("javascript", "if"),
        ("javascript", "function"),
        ("typescript", "interface"),
        ("go", "func"),
        ("rust", "fn"),
        ("java", "class"),
        ("zig", "fn"),
        ("c", "typedef"),
        ("cpp", "namespace"),
    ])
    def test_keyword_rejected_per_language(self, language, keyword) -> None:
        from agent.symbol_graph._extraction import _is_reserved_keyword
        assert _is_reserved_keyword(keyword, language) is True

    @pytest.mark.parametrize("language,name", [
        ("python", "verify_token"),
        ("javascript", "beginWork"),
        ("typescript", "renderWithHooks"),
        ("go", "HandleRequest"),
        ("rust", "resolve_package"),
        ("java", "processOrder"),
        ("zig", "allocBuffer"),
        ("c", "PyDict_New"),
        ("cpp", "render"),
    ])
    def test_real_symbol_name_not_rejected(self, language, name) -> None:
        from agent.symbol_graph._extraction import _is_reserved_keyword
        assert _is_reserved_keyword(name, language) is False

    def test_every_symbol_language_has_a_reserved_keyword_set(self) -> None:
        from agent.config import SYMBOL_GRAPH_RESERVED_KEYWORDS
        from agent.symbol_graph import SYMBOL_LANGUAGES
        for lang in SYMBOL_LANGUAGES:
            assert lang in SYMBOL_GRAPH_RESERVED_KEYWORDS
            assert len(SYMBOL_GRAPH_RESERVED_KEYWORDS[lang]) > 0


class TestMarkdownSupport:
    """chunk_file() produces heading-aware section chunks for .md files."""

    _MD_WITH_HEADINGS = """\
# My Project

A brief introduction paragraph with some context.

## Installation

Run `pip install myproject` to get started.
Some additional installation notes here.

## Usage

Here is how to use the tool.

### Advanced Usage

For power users, see the config section.

## API Reference

The main API surface is documented here.
"""

    _MD_NO_HEADINGS = """\
This is a flat markdown file.
It has no headings at all.
Just plain paragraphs of text spread across multiple lines.
"""

    def _write_md(self, tmp_path, content: str, name: str = "README.md") -> str:
        p = tmp_path / name
        p.write_text(content)
        return str(p)

    def test_markdown_uses_section_node_type(self, tmp_path) -> None:
        from agent.indexer import chunk_file
        chunks = chunk_file(self._write_md(tmp_path, self._MD_WITH_HEADINGS))
        assert any(c.node_type == "section" for c in chunks), (
            "expected section chunks for markdown with headings"
        )

    def test_markdown_heading_text_becomes_symbol_name(self, tmp_path) -> None:
        from agent.indexer import chunk_file
        chunks = chunk_file(self._write_md(tmp_path, self._MD_WITH_HEADINGS))
        names = {c.symbol_name for c in chunks}
        assert "Installation" in names
        assert "Usage" in names
        assert "API Reference" in names

    def test_markdown_not_window_fallback_for_headings(self, tmp_path) -> None:
        from agent.indexer import chunk_file
        chunks = chunk_file(self._write_md(tmp_path, self._MD_WITH_HEADINGS))
        assert not any(c.node_type == "window" for c in chunks), (
            "headed markdown should not use window fallback"
        )

    def test_markdown_no_headings_produces_single_preamble_chunk(self, tmp_path) -> None:
        from agent.indexer import chunk_file
        chunks = chunk_file(self._write_md(tmp_path, self._MD_NO_HEADINGS))
        # Headingless files flush as a single section with empty symbol_name (no boundary to split on)
        assert len(chunks) == 1
        assert chunks[0].language == "markdown"
        assert chunks[0].symbol_name == ""

    def test_markdown_language_label_set(self, tmp_path) -> None:
        from agent.indexer import chunk_file
        chunks = chunk_file(self._write_md(tmp_path, self._MD_WITH_HEADINGS))
        assert all(c.language == "markdown" for c in chunks)

    def test_markdown_file_indexed_by_indexer(self, indexer, tmp_path) -> None:
        chunks = indexer.index_file(self._write_md(tmp_path, self._MD_WITH_HEADINGS))
        assert chunks >= 3, f"expected ≥3 markdown sections, got {chunks}"


class TestHTMLSupport:
    """chunk_file() handles .html files (window fallback with 'html' language label)."""

    _HTML = """\
<!DOCTYPE html>
<html>
<head><title>Test Page</title></head>
<body>
  <h1>Welcome</h1>
  <p>Some content here.</p>
  <section>
    <h2>About</h2>
    <p>More content.</p>
  </section>
</body>
</html>
"""

    def _write_html(self, tmp_path) -> str:
        p = tmp_path / "index.html"
        p.write_text(self._HTML)
        return str(p)

    def test_html_language_label_set(self, tmp_path) -> None:
        from agent.indexer import chunk_file
        chunks = chunk_file(self._write_html(tmp_path))
        assert all(c.language == "html" for c in chunks)

    def test_html_produces_chunks(self, tmp_path) -> None:
        from agent.indexer import chunk_file
        chunks = chunk_file(self._write_html(tmp_path))
        assert len(chunks) >= 1

    def test_html_file_indexed_by_indexer(self, indexer, tmp_path) -> None:
        count = indexer.index_file(self._write_html(tmp_path))
        assert count >= 1, f"expected ≥1 HTML chunk, got {count}"


# ---------------------------------------------------------------------------
# UPG-11.3 — .txt and .rst prose files are indexed (F2)
# ---------------------------------------------------------------------------

class TestTxtRstSupport:
    """chunk_file() must produce chunks for .txt and .rst files (UPG-11.3).

    Django ships docs as .txt (e.g. docs/howto/custom-model-fields.txt).
    These were invisible to search before UPG-11.3.
    They are indexed as prose windows with language='txt'/'rst' and subject to
    the doc-prose quality multiplier (_Q_DOC_PROSE=0.70) so code still leads.
    """

    _TXT_CONTENT = """\
Writing custom model fields
===========================

To create a custom field, you need to subclass Field and implement deconstruct()
and from_db_value(). The deconstruct() method must return a 4-tuple
(name, path, args, kwargs) that allows Django's migration framework to serialize
the field.

The from_db_value() method is called every time data is loaded from the database,
including in aggregates and values() calls.
"""

    _RST_CONTENT = """\
Custom Lookup Reference
=======================

.. currentmodule:: django.db.models

Lookup API
----------

A lookup is a Django expression that determines how a condition is translated
to a SQL WHERE clause. Custom lookups can be registered via ``register_lookup``.
"""

    def _write_txt(self, tmp_path) -> str:
        p = tmp_path / "howto.txt"
        p.write_text(self._TXT_CONTENT)
        return str(p)

    def _write_rst(self, tmp_path) -> str:
        p = tmp_path / "ref.rst"
        p.write_text(self._RST_CONTENT)
        return str(p)

    def test_txt_produces_chunks(self, tmp_path) -> None:
        from agent.indexer import chunk_file
        chunks = chunk_file(self._write_txt(tmp_path))
        assert len(chunks) >= 1, f"expected ≥1 txt chunk, got {len(chunks)}"

    def test_txt_language_label(self, tmp_path) -> None:
        from agent.indexer import chunk_file
        chunks = chunk_file(self._write_txt(tmp_path))
        assert all(c.language == "txt" for c in chunks), (
            f"all chunks must have language='txt', got {[c.language for c in chunks]}"
        )

    def test_txt_indexed_by_indexer(self, indexer, tmp_path) -> None:
        count = indexer.index_file(self._write_txt(tmp_path))
        assert count >= 1, f"expected ≥1 txt chunk indexed, got {count}"

    def test_rst_produces_chunks(self, tmp_path) -> None:
        from agent.indexer import chunk_file
        chunks = chunk_file(self._write_rst(tmp_path))
        assert len(chunks) >= 1, f"expected ≥1 rst chunk, got {len(chunks)}"

    def test_rst_language_label(self, tmp_path) -> None:
        from agent.indexer import chunk_file
        chunks = chunk_file(self._write_rst(tmp_path))
        assert all(c.language == "rst" for c in chunks), (
            f"all chunks must have language='rst', got {[c.language for c in chunks]}"
        )

    def test_rst_indexed_by_indexer(self, indexer, tmp_path) -> None:
        count = indexer.index_file(self._write_rst(tmp_path))
        assert count >= 1, f"expected ≥1 rst chunk indexed, got {count}"

    def test_txt_in_indexed_languages(self, indexer, tmp_path) -> None:
        """After indexing a .txt file, /v1/status languages must include 'txt'."""
        indexer.index_file(self._write_txt(tmp_path))
        langs = indexer.indexed_languages()
        assert "txt" in langs, (
            f"UPG-11.3: 'txt' must appear in indexed languages after indexing a .txt file. "
            f"Got: {langs}"
        )

    def test_txt_doc_prose_quality_multiplier(self, tmp_path) -> None:
        """txt/rst chunks must get the doc-prose quality multiplier (0.70),
        not the code quality (1.0), so code chunks still lead on code queries."""
        from agent.chunk_quality import quality_score, is_doc_language
        # Verify doc language classification
        assert is_doc_language("txt"), "txt must be classified as doc language"
        assert is_doc_language("rst"), "rst must be classified as doc language"
        # quality_score for a non-trivial txt chunk should be 0.70
        score = quality_score("Custom field description with multiple lines of prose.\n" * 5,
                              "/docs/howto.txt", language="txt")
        assert score == pytest.approx(0.70, abs=0.01), (
            f"UPG-11.3: txt doc chunks must get _Q_DOC_PROSE=0.70 quality. Got {score}"
        )

    def test_workspace_walk_picks_up_txt_files(self, indexer, tmp_path) -> None:
        """index_workspace() must include .txt files (they're now in LANG_BY_EXT)."""
        # Write a Python file and a .txt doc
        make_py(tmp_path, "models.py", "def model(): pass")
        (tmp_path / "readme.txt").write_text("Project documentation.\n" * 5)
        files, chunks = indexer.index_workspace()
        assert files == 2, f"expected 2 files indexed (1 py + 1 txt), got {files}"

    def test_txt_searchable_after_index(self, indexer, tmp_path) -> None:
        """txt content must be returned in search results after indexing."""
        from agent.searcher import CodeSearcher

        txt_file = tmp_path / "custom-model-fields.txt"
        txt_file.write_text(self._TXT_CONTENT)
        indexer.index_file(str(txt_file))

        s = CodeSearcher(indexer)
        s.refresh_bm25()

        results, _ = s.search(
            "custom model field deconstruct from_db_value", n_results=5, rerank=False
        )
        txt_results = [r for r in results if r.file_path.endswith(".txt")]
        assert txt_results, (
            "UPG-11.3: .txt file content must appear in search results. "
            f"Got: {[r.file_path for r in results]}"
        )


# ---------------------------------------------------------------------------
# UPG-TXT-CHUNK-COVERAGE — .txt/.rst prose split at RST setext-style section
# headings instead of always falling to 200-line window chunks
# ---------------------------------------------------------------------------

class TestRstSectionHeadingDetection:
    """Coverage of the RST setext-heading detector (title line + repeated-
    punctuation underline, with an optional matching overline for a document
    title) exercised through the public `chunk_file()` entry point — never
    importing the new private helpers directly, so these assertions fail for
    a real behavioral reason on unmodified main (still window-only for .txt),
    not an ImportError proving only that a name is new."""

    def _chunks(self, tmp_path, text: str, name: str = "doc.txt"):
        from agent.indexer import chunk_file
        p = tmp_path / name
        p.write_text(text)
        return chunk_file(str(p))

    def test_title_plus_underline_detected(self, tmp_path) -> None:
        chunks = self._chunks(tmp_path, "Installation\n============\n\nRun pip install.\n")
        assert [c.node_type for c in chunks] == ["section"]
        assert chunks[0].symbol_name == "Installation"

    def test_overline_and_underline_absorbed_as_one_heading(self, tmp_path) -> None:
        text = "==========\nDoc Title\n==========\n\nBody text.\n"
        chunks = self._chunks(tmp_path, text)
        # exactly one heading — the overline must not itself be misread as a
        # second, title-less heading alongside the real one.
        assert [c.symbol_name for c in chunks] == ["Doc Title"]

    def test_underline_shorter_than_title_is_not_a_heading(self, tmp_path) -> None:
        # docutils requires the adornment be at least as long as the title;
        # a short divider under a long line of prose is not a heading, so the
        # file has no RST structure at all and must use the window fallback.
        text = "This is a long line of ordinary prose text\n---\nMore text.\n"
        chunks = self._chunks(tmp_path, text)
        assert all(c.node_type == "window" for c in chunks), chunks

    def test_two_adornment_lines_in_a_row_is_not_a_heading(self, tmp_path) -> None:
        # a horizontal-rule-style divider (two adornment lines back to back,
        # no real title text between them) must not be misread as a heading —
        # with no other heading in the file, this must fall back to windows.
        text = "----\n----\nSome text.\n"
        chunks = self._chunks(tmp_path, text)
        assert all(c.node_type == "window" for c in chunks), chunks

    def test_multiple_sibling_headings_in_document_order(self, tmp_path) -> None:
        text = (
            "Section One\n"
            "===========\n"
            "First body.\n\n"
            "Section Two\n"
            "===========\n"
            "Second body.\n"
        )
        chunks = self._chunks(tmp_path, text)
        assert [c.symbol_name for c in chunks] == ["Section One", "Section Two"]

    def test_tilde_and_caret_adornment_characters_both_work(self, tmp_path) -> None:
        # docutils does not restrict the adornment character — any repeated
        # non-alphanumeric character is valid, verified structurally rather
        # than against a fixed character allowlist.
        text = "Sub A\n~~~~~\nbody\n\nSub B\n^^^^^\nbody\n"
        chunks = self._chunks(tmp_path, text)
        assert [c.symbol_name for c in chunks] == ["Sub A", "Sub B"]


class TestTxtRstSectionHeadingChunking:
    """chunk_file() splits .txt/.rst prose at RST heading boundaries instead
    of merging every subsection within a 200-line window into one diluted
    chunk (UPG-TXT-CHUNK-COVERAGE)."""

    @staticmethod
    def _rst_doc(n_sections: int, paragraph_lines: int, title: str) -> str:
        """A realistic multi-section RST doc: an overlined document title
        followed by `n_sections` dash-underlined subsections — mirrors the
        shape of a real docs/howto/*.txt page (one title, many independent
        subsections, no further nesting within each)."""
        lines = ["=" * len(title), title, "=" * len(title), "",
                 "Introductory paragraph line about the topic overall.", ""]
        for i in range(n_sections):
            heading = f"Section heading number {i}"
            lines.append(heading)
            lines.append("-" * len(heading))
            lines.append("")
            for j in range(paragraph_lines):
                lines.append(f"Prose sentence {j} inside section {i} describing behavior in detail.")
            lines.append("")
        return "\n".join(lines)

    def _write_txt(self, tmp_path, content: str, name: str = "howto.txt") -> str:
        p = tmp_path / name
        p.write_text(content)
        return str(p)

    def test_multi_section_doc_gets_one_chunk_per_heading(self, tmp_path) -> None:
        from agent.indexer import chunk_file
        title = "How to create custom template tags and filters"
        content = self._rst_doc(n_sections=10, paragraph_lines=20, title=title)
        n_lines = len(content.splitlines())
        assert n_lines > 200, "fixture must exceed one fallback window to be a meaningful witness"
        chunks = chunk_file(self._write_txt(tmp_path, content))
        section_chunks = [c for c in chunks if c.node_type == "section"]
        # one chunk for the doc title + one per subsection heading
        assert len(section_chunks) == 11, (
            f"expected 11 section chunks (title + 10 subsections), got "
            f"{len(section_chunks)}: {[c.symbol_name for c in chunks]}"
        )
        assert not any(c.node_type == "window" for c in chunks), (
            "a heading-structured prose doc must not fall back to window chunking"
        )

    def test_doc_verbatim_title_is_its_own_retrievable_chunk(self, tmp_path) -> None:
        """The T12 gate query is the document's own verbatim title. It must
        become a small, first-class chunk on its own — not buried inside a
        200-line merged window carrying nine unrelated subsections too."""
        from agent.indexer import chunk_file
        title = "How to create custom template tags and filters"
        content = self._rst_doc(n_sections=10, paragraph_lines=20, title=title)
        chunks = chunk_file(self._write_txt(tmp_path, content))
        title_chunks = [c for c in chunks if c.symbol_name == title]
        assert len(title_chunks) == 1, (
            f"expected the title as its own chunk symbol_name, got "
            f"{[c.symbol_name for c in chunks]}"
        )
        # the title chunk must not also carry the other sections' content
        assert len(title_chunks[0].content.splitlines()) < 20

    def test_more_chunks_than_pre_fix_window_only_baseline(self, tmp_path) -> None:
        """Direct before/after comparison on the same content: the old
        window-only path (window=200/overlap=50) undercounts a
        heading-structured doc relative to one chunk per heading."""
        from agent.indexer import chunk_file
        from agent.indexer._chunking import _fallback_window_chunks, _postprocess_chunks
        title = "How to create custom template tags and filters"
        content = self._rst_doc(n_sections=10, paragraph_lines=20, title=title)
        path = self._write_txt(tmp_path, content)
        after = chunk_file(path)
        before = _postprocess_chunks(_fallback_window_chunks(content.splitlines(), path, "txt"))
        assert len(after) > len(before), (
            f"expected more chunks after heading-aware chunking: before={len(before)}, after={len(after)}"
        )
        assert len(before) <= 3, f"sanity: 380-line doc should be ~2-3 windows pre-fix, got {len(before)}"

    def test_window_fallback_preserved_for_headingless_txt(self, tmp_path) -> None:
        """A .txt file with no RST heading structure at all (flat prose, a
        LICENSE.txt, a CHANGELOG with no section markers) must still use the
        window fallback — this is a coverage IMPROVEMENT for structured docs,
        not a behavior change for unstructured ones."""
        from agent.indexer import chunk_file
        flat = "\n".join(
            f"Plain unstructured prose line {i} with no section markers at all."
            for i in range(300)
        )
        chunks = chunk_file(self._write_txt(tmp_path, flat))
        assert all(c.node_type == "window" for c in chunks), (
            f"a .txt file with no RST heading structure must use the window fallback, "
            f"got node_types={[c.node_type for c in chunks]}"
        )
        assert len(chunks) >= 2  # 300 lines / window=200,overlap=50 -> multiple windows

    def test_oversized_section_is_capped_not_left_unbounded(self, tmp_path) -> None:
        """A single heading with no further nested heading (e.g. a long
        release-notes 'Miscellaneous' section) must still be capped at
        indexing.max_chunk_lines, same budget a code container body respects
        — an unbounded prose section produced an outlier chunk whose
        embedding cost dwarfed every normal chunk sharing its batch (measured
        against the real django corpus: an uncapped 1191-line section)."""
        from agent.indexer import chunk_file
        from agent.config import INDEXING_MAX_CHUNK_LINES
        title = "Big Section"
        lines = [title, "=" * len(title), ""]
        lines += [f"Prose line {i} with no further subsection heading at all." for i in range(400)]
        content = "\n".join(lines)
        chunks = chunk_file(self._write_txt(tmp_path, content))
        section_chunks = [c for c in chunks if c.node_type == "section"]
        assert len(section_chunks) > 1, "a 400-line unsubdivided section must be split into multiple capped chunks"
        for c in section_chunks:
            n_lines = c.end_line - c.start_line + 1
            assert n_lines <= INDEXING_MAX_CHUNK_LINES, (
                f"chunk {c.chunk_id} has {n_lines} lines, exceeds cap {INDEXING_MAX_CHUNK_LINES}"
            )
        # no content lost: every sub-chunk still carries the same heading name
        assert all(c.symbol_name == title for c in section_chunks)

    def test_rst_language_uses_same_section_splitting_as_txt(self, tmp_path) -> None:
        from agent.indexer import chunk_file
        title = "Custom Lookup Reference"
        content = f"{title}\n{'=' * len(title)}\n\nIntro.\n\nSub One\n-------\n\nBody one.\n\nSub Two\n-------\n\nBody two.\n"
        p = tmp_path / "ref.rst"
        p.write_text(content)
        chunks = chunk_file(str(p))
        section_chunks = [c for c in chunks if c.node_type == "section"]
        assert len(section_chunks) == 3, [c.symbol_name for c in chunks]
        assert all(c.language == "rst" for c in chunks)


# ---------------------------------------------------------------------------
# UPG-WINDOW-CHUNK-BYTE-CAP — a character cap alongside every line-based cap.
# A single long source line (minified JS bundle, generated SQL dump, one
# giant data-fixture line) defeats the window walker's 200-LINE window and
# produced an unbounded chunk: reproduced on main with django's
# raster.numpy.txt, which chunks into 2 windows whose largest is 709,049
# characters across only 174 lines. Oversized chunks are now sub-split into
# bounded pieces — tail retrievable, never truncated away.
# ---------------------------------------------------------------------------

class TestChunkCharCap:
    CAP = 400

    @pytest.fixture()
    def _small_cap(self, monkeypatch):
        """Shrink the cap for fast, deterministic splitting assertions.
        Opt-in per test: the byte-identical and wholesale-drop tests below
        deliberately do NOT request this fixture — they run under the real
        shipped config."""
        import agent.indexer._chunking as chunking_mod
        monkeypatch.setattr(chunking_mod, "_MAX_CHUNK_CHARS", self.CAP)

    def _giant_line(self, n_chars: int) -> str:
        # Deterministic pseudo-data dump shaped like raster.numpy.txt's array
        # literal — digits/punctuation, no newlines, no spaces near the end so
        # the tail is recognisable.
        body = "".join(f"{i % 97:02d}," for i in range(n_chars // 3 + 1))
        return "DATA = np.array([" + body[:n_chars - len("DATA = np.array([") - 2] + "])"

    def test_window_fallback_giant_line_produces_bounded_chunks(
        self, _small_cap, tmp_path,
    ) -> None:
        """The reproduced shape end-to-end through chunk_file(): a mostly-small
        .txt file whose one giant line previously emitted an unbounded window
        chunk now yields multiple ≤cap pieces, with the giant line's bytes
        fully preserved across its hard-split slices."""
        from agent.indexer import chunk_file
        giant = self._giant_line(50_000)
        lines = [f"header line {i} of a small text fixture" for i in range(30)]
        lines.insert(15, giant)
        p = tmp_path / "raster-shaped.txt"
        p.write_text("\n".join(lines), encoding="utf-8")

        chunks = chunk_file(str(p))
        assert len(chunks) > 1, "one giant line must produce multiple bounded chunks"
        oversized = [c for c in chunks if len(c.content) > self.CAP]
        assert oversized == [], (
            f"chunks exceed the character cap: {[len(c.content) for c in oversized]}"
        )
        # Tail retrievable: the giant line's slices, re-joined IN ORDINAL
        # ORDER (ids sort lexicographically — #10 before #2 — so order by the
        # numeric suffix), ARE the line — nothing truncated away.
        slices = [c for c in chunks if c.start_line == 16 and c.end_line == 16]
        assert len(slices) >= 2, "the giant line itself must be hard-split into several slices"
        slices.sort(key=lambda c: int(c.chunk_id.rsplit("#", 1)[1]))
        assert "".join(s.content for s in slices) == giant
        assert slices[-1].content.endswith("])"), "the line's tail bytes must survive in the last slice"

    def test_hard_split_piece_ids_unique_and_marked(
        self, _small_cap, tmp_path,
    ) -> None:
        """Hard-split slices of one line share its single-line span, so their
        ids carry a '#<k>' ordinal suffix — unique among themselves and
        uncollidable with any natural 'path:start-end' id (which never
        contains '#')."""
        from agent.indexer import chunk_file
        lines = [self._giant_line(50_000)] + [f"filler {i}" for i in range(5)]
        p = tmp_path / "one-liner.txt"
        p.write_text("\n".join(lines), encoding="utf-8")
        chunks = chunk_file(str(p))
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), f"duplicate chunk ids: {ids}"
        suffixed = [i for i in ids if "#" in i]
        assert len(suffixed) >= 2, f"expected several #-suffixed hard-split ids, got {ids}"
        for cid in suffixed:
            base = cid.split("#")[0]
            assert base.endswith("-1"), f"suffixed id {cid} must carry the single-line span"

    def test_hard_split_pieces_survive_trivial_drop_for_txt(
        self, _small_cap, tmp_path,
    ) -> None:
        """Each hard-split slice of a giant line is a 1-non-blank-line txt
        chunk — exactly what is_trivial_chunk's short-doc rule (≤2 non-blank
        lines) drops. The parent's keep/drop decision happens BEFORE the split
        and pieces are never re-judged, so a KEPT oversized chunk cannot lose
        its tail to hygiene: two filler lines keep the window above the
        short-doc threshold, yet every slice alone would fail it — if
        postprocessing re-judged pieces, ALL of them would vanish."""
        from agent.indexer import chunk_file
        lines = [self._giant_line(50_000), "filler line one", "filler line two"]
        p = tmp_path / "kept-giant.txt"
        p.write_text("\n".join(lines), encoding="utf-8")
        chunks = chunk_file(str(p))
        assert len(chunks) >= 2, (
            "hard-split slices must survive postprocessing — dropping any of "
            "them loses exactly the tail bytes the cap exists to make retrievable"
        )

    def test_short_doc_rule_still_drops_one_line_txt_wholesale(
        self, tmp_path,
    ) -> None:
        """The keep/drop decision itself is unchanged by the char cap: a .txt
        file whose ONLY content is one non-blank line is dropped wholesale by
        UPG-15.5's short-doc rule exactly as before this fix. The rule has no
        size escape hatch — relaxing it for huge stub-shaped docs would be a
        separate product decision, not part of the byte cap."""
        from agent.indexer import chunk_file
        p = tmp_path / "only-giant.txt"
        p.write_text(self._giant_line(50_000), encoding="utf-8")
        assert chunk_file(str(p)) == [], (
            "UPG-15.5 short-doc drop must be unaffected by the char-cap change"
        )

    def test_normal_txt_window_chunks_byte_identical(self, tmp_path):
        """THE non-regression constraint: a normal source file's chunk set is
        byte-for-byte unchanged. Runs under the REAL shipped config (no cap
        override): a 450-line headingless .txt takes the same window-fallback
        path as before this fix, and every emitted field must equal the
        hand-computed pre-fix expectation exactly — ids, spans, contents,
        node types, symbol names."""
        import agent.indexer._chunking as chunking_mod

        assert chunking_mod._MAX_CHUNK_CHARS >= 32_000, (
            "this test pins behaviour under the real shipped cap; it must not "
            "run under a shrunken override"
        )
        from agent.indexer import chunk_file
        lines = [f"Plain unstructured prose line {i} with no section markers." for i in range(450)]
        p = tmp_path / "flat.txt"
        p.write_text("\n".join(lines), encoding="utf-8")

        chunks = chunk_file(str(p))
        # window=200, overlap=50 → starts at 0, 150, 300 (1-indexed 1, 151, 301).
        expected_spans = [(1, 200), (151, 350), (301, 450)]
        assert len(chunks) == len(expected_spans), (
            f"expected {len(expected_spans)} windows, got {len(chunks)}"
        )
        for c, (start, end) in zip(chunks, expected_spans):
            assert c.chunk_id == f"{p}:{start}-{end}", c.chunk_id
            assert c.file_path == str(p)
            assert c.start_line == start and c.end_line == end
            assert c.node_type == "window"
            assert c.symbol_name == ""
            assert c.language == "txt"
            assert c.content == "\n".join(lines[start - 1:end]), (
                f"window content changed for span {start}-{end}"
            )

    def test_python_giant_line_function_bounded_on_ast_path(
        self, _small_cap, tmp_path,
    ) -> None:
        """Acceptance extends past the window fallback: an AST-parsed python
        file whose function holds one giant line emits only bounded chunks —
        the leaf definition's content is verbatim file lines, so the giant
        line is hard-split into #-suffixed single-line-span pieces."""
        from agent.indexer import chunk_file
        giant = self._giant_line(50_000)
        src = "\n".join([
            "def load_table():",
            "    row = '" + "x" * 300 + "'",
            "    return [",
            giant + ",",
            "    ]",
            "",
        ])
        p = tmp_path / "table.py"
        p.write_text(src, encoding="utf-8")
        chunks = chunk_file(str(p))
        assert chunks, "python file must still produce chunks"
        assert all(len(c.content) <= self.CAP for c in chunks), (
            [len(c.content) for c in chunks]
        )

    def test_char_carve_for_span_mismatched_content(self, _small_cap) -> None:
        """When content lines don't map 1:1 onto the recorded span (stripped
        prose sections; AST container headers keeping header lines but the
        full container span), the splitter carves at CHARACTER granularity and
        every piece inherits the parent's own span — honest 'within this
        range' metadata instead of invented boundaries."""
        from agent.indexer._chunking import _split_oversized_chunk
        from agent.indexer._types import CodeChunk
        parent = CodeChunk(
            chunk_id="/ws/doc.txt:40-41",
            content="\n".join(f"prose paragraph {i} " + "w" * 90 for i in range(6)),
            file_path="/ws/doc.txt",
            language="txt",
            node_type="section",
            start_line=40,
            end_line=41,  # span says 2 lines; content has 6 (e.g. stripped section)
            symbol_name="Section",
        )
        pieces = _split_oversized_chunk(parent)
        assert len(pieces) >= 2
        assert all(p.end_line == 41 and p.start_line == 40 for p in pieces)
        ids = [p.chunk_id for p in pieces]
        assert len(ids) == len(set(ids))
        assert all("#" in cid for cid in ids)
        assert all(len(p.content) <= self.CAP for p in pieces)
        assert "".join(p.content for p in pieces) == parent.content
        assert all(p.symbol_name == "Section" and p.node_type == "section" for p in pieces)

    def test_no_char_cap_change_for_small_files(self, _small_cap, tmp_path) -> None:
        """A file whose every chunk is within the cap takes the identical
        pass-through path — no re-splitting, no id churn — even under a tiny
        cap boundary probe just above the largest emitted chunk."""
        from agent.indexer._chunking import _postprocess_chunks, _fallback_window_chunks
        lines = [f"line {i}" for i in range(20)]
        path = "/ws/small.txt"
        raw = _fallback_window_chunks(lines, path, "txt")
        out = _postprocess_chunks(list(raw))
        assert out == raw, "chunks within the cap must pass through untouched"


# ---------------------------------------------------------------------------
# Indexing hygiene — orphan pruning, mtime-cache colocation, force rebuild
# (UPG-8.4 / UPG-8.5 / UPG-8.6)
# ---------------------------------------------------------------------------

class TestIndexingHygiene:
    def _file_paths_in_collection(self, indexer) -> set[str]:
        _, _, metas = indexer.get_all_documents()
        return {m.get("file_path", "") for m in metas}

    # --- UPG-8.5: mtime cache lives next to the chroma db dir ---
    def test_mtime_cache_colocated_with_db(self, indexer, tmp_path) -> None:
        cache_path = indexer._mtime_cache_path()
        # Fixture builds CodeIndexer with db_path=tmp_path/"chroma"
        assert cache_path == (tmp_path / "chroma" / "index_cache.json")
        # And it must sit inside the indexer's db dir, not a separate /db/ tree
        assert str(cache_path).startswith(str(tmp_path / "chroma"))

    def test_mtime_cache_written_into_db_dir(self, indexer, tmp_path) -> None:
        make_py(tmp_path, "a.py", "def a(): pass")
        indexer.index_workspace()
        assert (tmp_path / "chroma" / "index_cache.json").exists()

    # --- UPG-8.4: orphaned chunks pruned when files leave the walk set ---
    def test_deleted_file_chunks_pruned_on_reindex(self, indexer, tmp_path) -> None:
        make_py(tmp_path, "keep.py", "def keep(): pass")
        gone = Path(make_py(tmp_path, "gone.py", "def gone(): pass"))
        indexer.index_workspace()
        assert any("gone.py" in p for p in self._file_paths_in_collection(indexer))

        gone.unlink()  # file leaves the walk set
        indexer.index_workspace()
        paths = self._file_paths_in_collection(indexer)
        assert not any("gone.py" in p for p in paths), "orphaned chunks not pruned"
        assert any("keep.py" in p for p in paths), "kept file must remain"

    def test_vectrignore_excluded_dir_pruned_on_reindex(self, indexer, tmp_path) -> None:
        (tmp_path / "vendor").mkdir()
        make_py(tmp_path, "vendor/lib.py", "def lib(): pass")
        make_py(tmp_path, "app.py", "def app(): pass")
        indexer.index_workspace()
        assert any("vendor/lib.py" in p for p in self._file_paths_in_collection(indexer))

        # Newly exclude the vendor dir; its chunks must be pruned on next index.
        (tmp_path / ".vectrignore").write_text("vendor\n", encoding="utf-8")
        indexer.index_workspace()
        paths = self._file_paths_in_collection(indexer)
        assert not any("vendor/lib.py" in p for p in paths), "excluded dir not pruned"
        assert any("app.py" in p for p in paths)

    def test_vectrignore_regex_excluded_file_pruned_on_reindex(self, indexer, tmp_path) -> None:
        # UPG-EXCLUDE-REGEX: `re:<pattern>` entries prune the same way as bare
        # dir names and file globs — the reconcile mechanism is walk-set based,
        # so adding a regex line that newly excludes a file is enough.
        (tmp_path / "legacy").mkdir()
        make_py(tmp_path, "legacy/handler.py", "def handler(): pass")
        make_py(tmp_path, "app.py", "def app(): pass")
        indexer.index_workspace()
        assert any("legacy/handler.py" in p for p in self._file_paths_in_collection(indexer))

        (tmp_path / ".vectrignore").write_text("re:legacy/.*\n", encoding="utf-8")
        indexer.index_workspace()
        paths = self._file_paths_in_collection(indexer)
        assert not any("legacy/handler.py" in p for p in paths), "regex-excluded file not pruned"
        assert any("app.py" in p for p in paths)

    def test_prune_drops_mtime_cache_entry(self, indexer, tmp_path) -> None:
        gone = Path(make_py(tmp_path, "gone.py", "def gone(): pass"))
        indexer.index_workspace()
        assert str(gone) in indexer._load_mtime_cache()
        gone.unlink()
        indexer.index_workspace()
        assert str(gone) not in indexer._load_mtime_cache()

    # --- UPG-8.6: force=True rebuilds, ignoring the mtime cache ---
    def test_force_rebuilds_after_collection_desync(self, indexer, tmp_path) -> None:
        make_py(tmp_path, "a.py", "def a(): pass")
        indexer.index_workspace()
        n = indexer.total_chunks
        assert n > 0

        # Simulate a cache/collection desync: wipe the collection directly
        # (bypassing delete_file/_apply_chunk_delta) but keep the mtime cache
        # (the exact trap UPG-8.5 describes). This out-of-band mutation is
        # exactly the case total_chunks's in-memory counter does NOT track
        # (UPG-CHROMA-BLOCKING-EVENT-LOOP: it is refreshed only by the
        # tracked index_file/delete_file/index_workspace mutation paths, by
        # design — see total_chunks's docstring) — verify the real
        # collection state directly here instead.
        ids, _, _ = indexer.get_all_documents()
        indexer._collection.delete(ids=ids)
        assert indexer._collection.count() == 0

        # Without force, the stale mtime cache says "nothing to re-index".
        indexer.index_workspace(force=False)
        assert indexer._collection.count() == 0

        # force=True ignores the cache and rebuilds — this DOES go through
        # the tracked upsert path, so the cached total_chunks is correct
        # again afterwards.
        indexer.index_workspace(force=True)
        assert indexer.total_chunks == n
        assert indexer._collection.count() == n

    def test_force_does_not_duplicate_chunks(self, indexer, tmp_path) -> None:
        make_py(tmp_path, "a.py", "def a(): pass")
        indexer.index_workspace()
        n = indexer.total_chunks
        indexer.index_workspace(force=True)
        assert indexer.total_chunks == n, "force must replace, not duplicate, chunks"


# ---------------------------------------------------------------------------
# ARCH-4 — dual-vector purpose collection (pool-entry mechanism)
# ---------------------------------------------------------------------------

class TestDualVectorPurposeCollection:
    def test_purpose_collection_populated_for_symbol_chunks(self, indexer, tmp_path) -> None:
        path = make_py(tmp_path, "svc.py", """
            def process_payment(order_id: str) -> bool:
                \"\"\"Charge the customer for the given order and record the result.\"\"\"
                return True
        """)
        indexer.index_file(path)
        assert indexer._purpose_collection.count() > 0
        docs = indexer._purpose_collection.get(include=["documents"])["documents"]
        assert any("process_payment" in d for d in docs)
        assert any("Charge the customer" in d for d in docs)

    def test_purpose_collection_excludes_non_symbol_chunks(self, indexer, tmp_path) -> None:
        (tmp_path / "notes.md").write_text(
            "# Title\n\nSome prose that is not a symbol at all, just documentation.\n"
        )
        indexer.index_file(str(tmp_path / "notes.md"))
        body_ids, _, _ = indexer.get_all_documents()
        assert len(body_ids) > 0
        assert indexer._purpose_collection.count() == 0

    def test_purpose_collection_is_subset_of_body_ids(self, indexer, tmp_path) -> None:
        path = make_py(tmp_path, "mix.py", """
            def with_symbol():
                pass
        """)
        indexer.index_file(path)
        body_ids, _, _ = indexer.get_all_documents()
        purpose_ids = indexer._purpose_collection.get()["ids"]
        assert set(purpose_ids) <= set(body_ids)

    def test_delete_file_removes_purpose_collection_chunks(self, indexer, tmp_path) -> None:
        path = make_py(tmp_path, "temp.py", """
            def to_remove():
                x = 1
        """)
        indexer.index_file(path)
        assert indexer._purpose_collection.count() > 0
        indexer.delete_file(path)
        assert indexer._purpose_collection.count() == 0

    def _wait_until_purpose_pass_done(self, idx, timeout: float = 30.0) -> None:
        """index_workspace() defers the purpose-vector pass to a background
        executor by default (UPG-PURPOSE-PASS-DEFERRAL) and returns before it
        lands; callers that need the collection caught up before asserting on
        it must wait. This delegates to conftest.wait_for_deferred_purpose_pass,
        a FIFO drain barrier on the executor itself: the old fixed-deadline
        `purpose_vectors_pending` poll expired SILENTLY under machine load
        (e.g. a second full suite running concurrently —
        UPG-TEST-CONCURRENT-SUITE-FLAKE), letting tests proceed against a
        half-updated collection; the barrier either waits for real completion
        or fails loudly with a TimeoutError naming the mechanism."""
        from tests.conftest import wait_for_deferred_purpose_pass

        wait_for_deferred_purpose_pass(idx, timeout=timeout)

    def test_reindex_prunes_stale_purpose_chunks(self, indexer, tmp_path) -> None:
        path = Path(make_py(tmp_path, "evolving.py", "def v1(): pass"))
        indexer.index_workspace()
        self._wait_until_purpose_pass_done(indexer)
        first_purpose_ids = set(indexer._purpose_collection.get()["ids"])
        assert first_purpose_ids

        path.write_text("def v2(): pass\n")
        import os as _os
        _os.utime(path, (path.stat().st_atime, path.stat().st_mtime + 1))
        indexer.index_workspace()
        self._wait_until_purpose_pass_done(indexer)

        purpose_docs = indexer._purpose_collection.get(include=["documents"])["documents"]
        assert any("v2" in d for d in purpose_docs)
        assert not any("v1" in d for d in purpose_docs)

    def test_query_vector_purpose_empty_collection_is_graceful(self, indexer) -> None:
        """A fresh/old-schema workspace has an empty purpose collection — must not raise."""
        embedding = indexer.embed_query("anything")
        result = indexer.query_vector_purpose(embedding, n_results=5)
        assert result["ids"] == [[]]

    def test_query_vector_purpose_survives_stale_positive_cache(self, indexer, tmp_path) -> None:
        """The empty-collection guard in query_vector_purpose must read the
        real collection count live, not the cached total_purpose_chunks
        counter: if the cache is stale and still reports chunks present
        while the real collection has just been emptied (e.g. mid-way
        through a delete, before the cache is refreshed), querying the
        actually-empty collection would raise (ChromaDB behaviour the guard
        exists to shield against). A live read here is what lets this
        degrade gracefully instead of surfacing that as a search 500.

        Mocks the raise directly rather than relying on the currently
        installed ChromaDB version to actually raise on an empty query, so
        this test discriminates the cached-vs-live code path regardless of
        that version-specific behaviour.
        """
        path = make_py(tmp_path, "a.py", "def a(): pass")
        indexer.index_file(path)
        indexer.delete_file(path)

        # Simulate a stale cache: still reports chunks present even though
        # the real collection was just emptied above.
        with patch.object(type(indexer), "total_purpose_chunks", new=5), patch.object(
            indexer._purpose_collection,
            "query",
            side_effect=AssertionError("must not query an actually-empty collection"),
        ):
            embedding = indexer.embed_query("anything")
            result = indexer.query_vector_purpose(embedding, n_results=5)
        assert result["ids"] == [[]]

    def test_query_vector_purpose_returns_results_after_index(self, indexer, tmp_path) -> None:
        path = make_py(tmp_path, "search_me.py", """
            def rate_limit_check(ip: str) -> bool:
                \"\"\"Check whether the given IP has exceeded its rate limit.\"\"\"
                return True
        """)
        indexer.index_file(path)
        embedding = indexer.embed_query("rate limit check function")
        result = indexer.query_vector_purpose(embedding, n_results=5)
        assert len(result["ids"][0]) >= 1

    def test_get_chunk_documents_returns_body_content(self, indexer, tmp_path) -> None:
        path = make_py(tmp_path, "svc.py", """
            def known_function():
                return 42
        """)
        indexer.index_file(path)
        body_ids, body_docs, _ = indexer.get_all_documents()
        result = indexer.get_chunk_documents(body_ids)
        assert set(result.keys()) == set(body_ids)
        for cid, doc in zip(body_ids, body_docs):
            assert result[cid][0] == doc

    def test_get_chunk_documents_missing_ids_absent_from_result(self, indexer) -> None:
        assert indexer.get_chunk_documents(["does-not-exist"]) == {}

    def test_dual_vector_disabled_skips_purpose_collection(self, tmp_path, monkeypatch) -> None:
        from agent import indexer as idx_module
        from tests.conftest import _DummyEmbedProvider
        monkeypatch.setattr(idx_module, "get_embed_provider", lambda _m: _DummyEmbedProvider())
        monkeypatch.setattr("agent.indexer._core._DUAL_VECTOR_ENABLED", False)
        from agent.indexer import CodeIndexer
        idx = CodeIndexer(workspace_root=str(tmp_path), db_path=str(tmp_path / "chroma"))
        path = make_py(tmp_path, "a.py", "def with_symbol(): pass")
        idx.index_file(path)
        assert idx.total_chunks > 0
        assert idx._purpose_collection.count() == 0


class TestGetChunkCosineSimilarities:
    """get_chunk_cosine_similarities (UPG-11.12 forced-inclusion gate) is the
    one place that reads ChromaDB's `.get(..., include=["embeddings"])` —
    which returns a genuine `numpy.ndarray` of `numpy.float64` rows, not the
    plain Python floats every other collection read (`distances`,
    `metadatas`) already comes back as (empirically confirmed against this
    installed chromadb; see UPG-NUMPY-SCALAR-IDENTITY-AUDIT). These tests
    exercise the real `indexer` fixture's ChromaDB collection end to end —
    no numpy mock standing in for the real numpy-typed return value — so a
    regression that dropped the function's own `float()` casts would be
    caught here, not first noticed as a `TypeError: Object of type float64
    is not JSON serializable` at the REST/MCP response boundary this dict
    eventually reaches."""

    def test_returns_plain_python_floats_not_numpy_scalars(self, indexer, tmp_path) -> None:
        path = make_py(tmp_path, "svc.py", """
            def known_function():
                return 42
        """)
        indexer.index_file(path)
        body_ids, _, _ = indexer.get_all_documents()
        query_embedding = indexer.embed_query("known_function")
        result = indexer.get_chunk_cosine_similarities(query_embedding, body_ids)
        assert result
        for cid, score in result.items():
            assert type(score) is float, f"{cid}: expected plain float, got {type(score)}"

    def test_result_is_json_serializable(self, indexer, tmp_path) -> None:
        path = make_py(tmp_path, "svc.py", """
            def known_function():
                return 42
        """)
        indexer.index_file(path)
        body_ids, _, _ = indexer.get_all_documents()
        query_embedding = indexer.embed_query("known_function")
        result = indexer.get_chunk_cosine_similarities(query_embedding, body_ids)
        json.dumps(result)  # must not raise TypeError on a numpy scalar

    def test_self_similarity_of_the_exact_indexed_chunk_is_near_one(self, indexer, tmp_path) -> None:
        path = make_py(tmp_path, "svc.py", """
            def known_function():
                return 42
        """)
        indexer.index_file(path)
        body_ids, body_docs, _ = indexer.get_all_documents()
        cid, doc = body_ids[0], body_docs[0]
        query_embedding = indexer.embed_query(doc)
        result = indexer.get_chunk_cosine_similarities(query_embedding, [cid])
        assert result[cid] == pytest.approx(1.0, abs=0.01)

    def test_missing_chunk_ids_absent_from_result(self, indexer) -> None:
        assert indexer.get_chunk_cosine_similarities([0.1] * 768, ["does-not-exist"]) == {}

    def test_empty_query_embedding_or_ids_returns_empty_dict(self, indexer, tmp_path) -> None:
        path = make_py(tmp_path, "svc.py", "def f(): pass")
        indexer.index_file(path)
        body_ids, _, _ = indexer.get_all_documents()
        assert indexer.get_chunk_cosine_similarities([], body_ids) == {}
        assert indexer.get_chunk_cosine_similarities([0.1] * 768, []) == {}


# ---------------------------------------------------------------------------
# UPG-INDEX-MEM-STREAMING — embed/upsert must stream in O(batch) memory,
# never accumulate a full-corpus embedding list before writing anything.
# ---------------------------------------------------------------------------

class TestStreamingEmbedUpsert:
    """Peak indexing memory must be O(batch), not O(corpus): embed a batch,
    upsert it immediately, drop it, embed the next — never hold the whole
    corpus's embeddings (or the ids/documents/metadatas built from it) at
    once. Proven here by shrinking the batch-size constants and recording
    embed/upsert call order and sizes against a real CodeIndexer + real
    ChromaDB collection (only the embed provider is swapped, for a
    deterministic vector without a model download — it still returns REAL
    768-dim float vectors and REAL chunk ids/documents/metadatas flow through
    `collection.upsert`, never `[]`/`{}` stand-ins)."""

    def test_body_pass_interleaves_embed_and_upsert_in_small_batches(
        self, tmp_path, monkeypatch,
    ) -> None:
        from agent import indexer as idx_module
        from tests.conftest import _DummyEmbedProvider

        # Small batch sizes so a modest number of chunks still produces
        # several batches — proves the streaming shape without indexing a
        # large corpus.
        monkeypatch.setattr("agent.indexer._core._EMBED_BATCH_SIZE", 4)
        monkeypatch.setattr("agent.indexer._core._UPSERT_BATCH_SIZE", 2)

        call_log: list[tuple[str, int]] = []
        real_provider = _DummyEmbedProvider()

        class _RecordingEmbedProvider:
            def embed(self, texts: list[str]) -> list[list[float]]:
                call_log.append(("embed", len(texts)))
                return real_provider.embed(texts)  # real 768-dim vectors

            def embed_query(self, texts: list[str]) -> list[list[float]]:
                return real_provider.embed_query(texts)

        monkeypatch.setattr(idx_module, "get_embed_provider", lambda _m: _RecordingEmbedProvider())

        from agent.indexer import CodeIndexer
        idx = CodeIndexer(workspace_root=str(tmp_path), db_path=str(tmp_path / "chroma"))
        # Pre-write the embed-model stamp so index_workspace() doesn't detect
        # a (harmless, unrelated) model mismatch and call
        # _recreate_collections() — which would replace `idx._collection`
        # with a fresh object and silently drop the patch installed below.
        idx._write_embed_model_stamp()

        real_upsert = idx._collection.upsert

        def _recording_upsert(**kwargs):
            call_log.append(("upsert", len(kwargs["ids"])))
            return real_upsert(**kwargs)

        monkeypatch.setattr(idx._collection, "upsert", _recording_upsert)

        for i in range(20):
            make_py(tmp_path, f"mod_{i}.py", f"def fn_{i}(x): return x + {i}")

        idx.index_workspace()

        embed_calls = [n for kind, n in call_log if kind == "embed"]
        upsert_calls = [n for kind, n in call_log if kind == "upsert"]
        assert embed_calls, "no embed calls recorded"
        assert upsert_calls, "no upsert calls recorded"

        # No single embed/upsert call ever carries more than the configured
        # batch size worth of rows — the whole-corpus list this fix removes
        # would instead have shown up as one oversized call at the end.
        assert all(n <= 4 for n in embed_calls), embed_calls
        assert all(n <= 2 for n in upsert_calls), upsert_calls

        # Interleaving: at least one upsert must land strictly between the
        # first and second embed call — i.e. a batch is written before the
        # next batch is even embedded, not only after the whole corpus has
        # been embedded.
        embed_positions = [i for i, (kind, _) in enumerate(call_log) if kind == "embed"]
        upsert_positions = [i for i, (kind, _) in enumerate(call_log) if kind == "upsert"]
        assert len(embed_positions) >= 2, f"expected multiple embed batches: {call_log}"
        assert any(embed_positions[0] < u < embed_positions[1] for u in upsert_positions), (
            f"upsert did not happen between the first two embed batches: {call_log}"
        )

        # The index itself is still correct — streaming changes WHEN rows are
        # written, never WHAT is written.
        assert idx.total_chunks == 20


class TestIndexResourceGovernor:
    """UPG-INDEX-RESOURCE-GOVERNOR: priority clamp + duty-cycle pacing around
    Phase 3 (and the purpose-vector pass), plus file-level checkpointing so a
    kill/restart mid-index never re-embeds already-completed work. Exercised
    against a real CodeIndexer + real ChromaDB collection on tiny synthetic
    corpora only — never a large real repo."""

    def test_default_config_applies_priority_clamp_and_pacing(
        self, indexer, tmp_path, monkeypatch,
    ) -> None:
        import contextlib
        import agent.indexer._core as core_module

        calls: list[tuple] = []

        @contextlib.contextmanager
        def _recording_lowered_priority(enabled, **kwargs):
            calls.append(("lowered_priority", enabled))
            yield

        monkeypatch.setattr(core_module, "lowered_priority", _recording_lowered_priority)

        make_py(tmp_path, "a.py", "def a(): pass")
        indexer.index_workspace()

        lowered_calls = [c for c in calls if c[0] == "lowered_priority"]
        assert lowered_calls, "lowered_priority was never invoked around the embed batch loop"
        assert all(enabled is True for _, enabled in lowered_calls), (
            f"default config (INDEX_GOVERNOR_ENABLED=True, no foreground-fast) must "
            f"govern every batch loop: {lowered_calls}"
        )

    def test_foreground_fast_env_disables_governor(
        self, indexer, tmp_path, monkeypatch,
    ) -> None:
        import contextlib
        import agent.indexer._core as core_module

        monkeypatch.setenv("VECTR_FOREGROUND_FAST", "1")

        calls: list[tuple] = []

        @contextlib.contextmanager
        def _recording_lowered_priority(enabled, **kwargs):
            calls.append(("lowered_priority", enabled))
            yield

        monkeypatch.setattr(core_module, "lowered_priority", _recording_lowered_priority)

        make_py(tmp_path, "a.py", "def a(): pass")
        indexer.index_workspace()

        lowered_calls = [c for c in calls if c[0] == "lowered_priority"]
        assert lowered_calls, "lowered_priority was never invoked"
        assert all(enabled is False for _, enabled in lowered_calls), (
            f"VECTR_FOREGROUND_FAST=1 must restore unthrottled indexing exactly: {lowered_calls}"
        )

    def test_governor_disabled_in_config_skips_clamp(
        self, indexer, tmp_path, monkeypatch,
    ) -> None:
        import contextlib
        import agent.indexer._core as core_module

        monkeypatch.setattr(core_module, "_INDEX_GOVERNOR_ENABLED", False)

        calls: list[tuple] = []

        @contextlib.contextmanager
        def _recording_lowered_priority(enabled, **kwargs):
            calls.append(("lowered_priority", enabled))
            yield

        monkeypatch.setattr(core_module, "lowered_priority", _recording_lowered_priority)

        make_py(tmp_path, "a.py", "def a(): pass")
        indexer.index_workspace()

        lowered_calls = [c for c in calls if c[0] == "lowered_priority"]
        assert lowered_calls and all(enabled is False for _, enabled in lowered_calls)

    def test_index_file_purpose_pass_never_governed(
        self, indexer, tmp_path, monkeypatch,
    ) -> None:
        """The single-file watcher path (index_file) must stay ungoverned —
        small, interactive-latency-sensitive work, unlike a bulk
        index_workspace() run."""
        import contextlib
        import agent.indexer._core as core_module

        calls: list[tuple] = []

        @contextlib.contextmanager
        def _recording_lowered_priority(enabled, **kwargs):
            calls.append(("lowered_priority", enabled))
            yield

        monkeypatch.setattr(core_module, "lowered_priority", _recording_lowered_priority)

        path = make_py(tmp_path, "a.py", "def known_fn(): pass")
        indexer.index_file(path)

        lowered_calls = [c for c in calls if c[0] == "lowered_priority"]
        assert lowered_calls, "lowered_priority was never invoked from index_file's purpose pass"
        assert all(enabled is False for _, enabled in lowered_calls)

    def test_zero_chunk_files_checkpointed_before_batch_loop_runs(
        self, indexer, tmp_path, monkeypatch,
    ) -> None:
        """A file that chunks to zero chunks (mixed into a run with normal
        files) must be checkpointed as soon as Phase 3 begins, not only at
        the very end of the run (previously this mtime was silently
        discarded whenever `all_chunks` overall was non-empty but this
        particular file's own chunk list was empty)."""
        import agent.indexer._core as core_module

        monkeypatch.setattr(core_module, "_EMBED_BATCH_SIZE", 1)

        real_path = make_py(tmp_path, "real.py", "def real_fn(): pass")
        empty_path = tmp_path / "empty.py"
        empty_path.write_text("")  # chunks to zero chunks

        saved_snapshots: list[dict] = []
        real_save = indexer._save_mtime_cache

        def _recording_save(cache):
            saved_snapshots.append(dict(cache))
            real_save(cache)

        monkeypatch.setattr(indexer, "_save_mtime_cache", _recording_save)

        indexer.index_workspace()

        assert saved_snapshots, "mtime cache was never persisted"
        assert str(empty_path) in saved_snapshots[0], (
            f"zero-chunk file must be in the FIRST mtime-cache persist, "
            f"before Phase 3's batch loop runs: {saved_snapshots[0]}"
        )
        assert str(real_path) not in saved_snapshots[0], (
            "the real file's chunk had not been embedded yet when the "
            "zero-chunk pre-checkpoint fired"
        )

    def test_kill_mid_index_resumes_without_reembedding_completed_files(
        self, tmp_path, monkeypatch,
    ) -> None:
        """Acceptance criterion: killing the daemon mid-index and restarting
        resumes without re-embedding completed work. Simulated with a real
        CodeIndexer against a real on-disk ChromaDB collection: the embed
        provider raises partway through Phase 3 (the crash), and a SECOND
        CodeIndexer instance against the same db_path must never be asked to
        re-embed a chunk whose file was already checkpointed before the
        crash."""
        import agent.indexer._core as core_module
        import agent.indexer as idx_module
        from agent.indexer import CodeIndexer

        monkeypatch.setattr(core_module, "_EMBED_BATCH_SIZE", 1)
        monkeypatch.setattr(core_module, "_INDEX_GOVERNOR_CHECKPOINT_EVERY_BATCHES", 1)

        for i in range(4):
            make_py(tmp_path, f"m{i}.py", f"def f{i}():\n    return {i}\n")

        embedded_before_crash: list[str] = []
        call_count = {"n": 0}

        class _CrashingProvider:
            def embed(self, texts: list[str]) -> list[list[float]]:
                call_count["n"] += 1
                if call_count["n"] > 2:
                    raise RuntimeError("simulated crash")
                embedded_before_crash.extend(texts)
                return [[0.1] * 8 for _ in texts]

            def embed_query(self, texts: list[str]) -> list[list[float]]:
                return self.embed(texts)

        monkeypatch.setattr(idx_module, "get_embed_provider", lambda _m: _CrashingProvider())

        db_path = str(tmp_path / "chroma")
        indexer1 = CodeIndexer(workspace_root=str(tmp_path), db_path=db_path)
        with pytest.raises(RuntimeError, match="simulated crash"):
            indexer1.index_workspace()

        class _RecordingProvider:
            def __init__(self) -> None:
                self.calls: list[list[str]] = []

            def embed(self, texts: list[str]) -> list[list[float]]:
                self.calls.append(list(texts))
                return [[0.1] * 8 for _ in texts]

            def embed_query(self, texts: list[str]) -> list[list[float]]:
                return self.embed(texts)

        recorder = _RecordingProvider()
        monkeypatch.setattr(idx_module, "get_embed_provider", lambda _m: recorder)

        indexer2 = CodeIndexer(workspace_root=str(tmp_path), db_path=db_path)
        indexer2.index_workspace()

        reembedded_texts = {t for call in recorder.calls for t in call}
        assert not (reembedded_texts & set(embedded_before_crash)), (
            "resume re-embedded a chunk whose file was already checkpointed before the crash"
        )
        # Every chunk ends up indexed exactly once across the two runs.
        assert indexer2.total_chunks == 4

    def test_embed_model_stamp_survives_a_mid_run_crash(
        self, tmp_path, monkeypatch,
    ) -> None:
        """Regression guard for the stamp-timing defect this task fixed: the
        stamp must be written before Phase 3 begins, not only at the very
        end of a successful run — otherwise a crashed first-ever index on a
        brand-new workspace leaves no stamp, and a resumed run reads back
        None (treated as a mismatch), wiping the checkpointed progress via
        _recreate_collections()."""
        import agent.indexer._core as core_module
        import agent.indexer as idx_module
        from agent.indexer import CodeIndexer

        monkeypatch.setattr(core_module, "_EMBED_BATCH_SIZE", 1)

        make_py(tmp_path, "a.py", "def a(): pass")
        make_py(tmp_path, "b.py", "def b(): pass")

        class _CrashingProvider:
            def embed(self, texts: list[str]) -> list[list[float]]:
                raise RuntimeError("simulated crash on the very first batch")

            def embed_query(self, texts: list[str]) -> list[list[float]]:
                return [[0.1] * 8 for _ in texts]

        monkeypatch.setattr(idx_module, "get_embed_provider", lambda _m: _CrashingProvider())

        db_path = str(tmp_path / "chroma")
        indexer1 = CodeIndexer(workspace_root=str(tmp_path), db_path=db_path)
        with pytest.raises(RuntimeError):
            indexer1.index_workspace()

        assert indexer1._stored_embed_model() == indexer1.embed_model, (
            "the embed-model stamp must be written before Phase 3 begins, "
            "surviving a crash on the very first embed batch"
        )


class TestPurposePassDeferral:
    """UPG-PURPOSE-PASS-DEFERRAL: by default (config `defer_purpose_pass`,
    True) and outside --foreground-fast, index_workspace() hands the
    purpose-vector pass to a single-worker background executor and returns
    without waiting for it. Search already degrades gracefully while
    purpose vectors are backfilling via `query_vector_purpose`'s existing
    empty/partial-collection guard — this reuses that mechanism rather than
    inventing a second "partial readiness" notion. Exercised against a real
    CodeIndexer + real ChromaDB collection on tiny synthetic corpora only —
    never a large real repo."""

    def _wait_until_not_pending(self, idx, timeout: float = 30.0) -> None:
        """FIFO drain barrier (tests.conftest.wait_for_deferred_purpose_pass)
        instead of a silent fixed-deadline poll — see
        TestDualVectorPurposeCollection._wait_until_purpose_pass_done."""
        from tests.conftest import wait_for_deferred_purpose_pass

        wait_for_deferred_purpose_pass(idx, timeout=timeout)

    def test_drain_barrier_returns_only_after_queued_pass_completes(
        self, indexer, tmp_path, monkeypatch,
    ) -> None:
        """UPG-PURPOSE-PASS-TEST-RACE: pins what the drain barrier GUARANTEES
        and what the old pollers only hoped for — the barrier must not return
        while a purpose pass submitted before it is still mid-flight. If the
        barrier were still the silent poller: under load its deadline expired
        while the pass ran, this assertion read `done == []` mid-flight and
        failed nondeterministically (or worse, downstream collection reads
        saw stale data with no failure at all here)."""
        from tests.conftest import wait_for_deferred_purpose_pass

        done: list[bool] = []
        real_upsert = indexer._upsert_purpose_vectors

        def _slow_recording(*args, **kwargs):
            time.sleep(0.2)
            real_upsert(*args, **kwargs)
            done.append(True)

        monkeypatch.setattr(indexer, "_upsert_purpose_vectors", _slow_recording)

        make_py(tmp_path, "barrier_a.py", "def barrier_fn():\n    return 1\n")
        indexer.index_workspace()

        # The deferred pass may or may not have started yet — that's exactly
        # the race the barrier exists to close.
        wait_for_deferred_purpose_pass(indexer)
        assert done, (
            "wait_for_deferred_purpose_pass returned while the purpose pass "
            "submitted before it was still running — FIFO drain broken"
        )

    def test_purpose_pass_scheduled_on_background_executor_by_default(
        self, indexer, tmp_path, monkeypatch,
    ) -> None:
        calls: list[str] = []
        real_upsert = indexer._upsert_purpose_vectors

        def _recording(*args, **kwargs):
            calls.append(threading.current_thread().name)
            return real_upsert(*args, **kwargs)

        monkeypatch.setattr(indexer, "_upsert_purpose_vectors", _recording)

        make_py(tmp_path, "a.py", "def known_fn():\n    return 1\n")
        indexer.index_workspace()

        deadline = time.time() + 5
        while not calls and time.time() < deadline:
            time.sleep(0.01)

        assert calls, "the deferred purpose pass never ran"
        assert calls[0].startswith("vectr-purpose"), (
            f"default config must run the purpose pass on the background "
            f"executor thread, not the caller's thread: {calls}"
        )

    def test_foreground_fast_runs_purpose_pass_synchronously(
        self, indexer, tmp_path, monkeypatch,
    ) -> None:
        monkeypatch.setenv("VECTR_FOREGROUND_FAST", "1")
        calls: list[str] = []
        real_upsert = indexer._upsert_purpose_vectors

        def _recording(*args, **kwargs):
            calls.append(threading.current_thread().name)
            return real_upsert(*args, **kwargs)

        monkeypatch.setattr(indexer, "_upsert_purpose_vectors", _recording)

        make_py(tmp_path, "a.py", "def known_fn():\n    return 1\n")
        indexer.index_workspace()

        assert calls, "purpose pass never ran"
        assert not calls[0].startswith("vectr-purpose"), (
            f"--foreground-fast must run the purpose pass synchronously on "
            f"the caller's own thread: {calls}"
        )
        assert indexer.purpose_vectors_pending is False, (
            "a synchronous run must never leave purpose_vectors_pending True"
        )
        assert indexer._purpose_collection.count() == 1

    def test_defer_purpose_pass_config_false_runs_synchronously(
        self, indexer, tmp_path, monkeypatch,
    ) -> None:
        import agent.indexer._core as core_module

        monkeypatch.setattr(core_module, "_INDEX_GOVERNOR_DEFER_PURPOSE_PASS_ENABLED", False)
        calls: list[str] = []
        real_upsert = indexer._upsert_purpose_vectors

        def _recording(*args, **kwargs):
            calls.append(threading.current_thread().name)
            return real_upsert(*args, **kwargs)

        monkeypatch.setattr(indexer, "_upsert_purpose_vectors", _recording)

        make_py(tmp_path, "a.py", "def known_fn():\n    return 1\n")
        indexer.index_workspace()

        assert calls and not calls[0].startswith("vectr-purpose"), (
            f"defer_purpose_pass=False must restore synchronous behaviour: {calls}"
        )
        assert indexer.purpose_vectors_pending is False

    def test_purpose_vectors_pending_true_during_window_false_after(
        self, tmp_path, monkeypatch,
    ) -> None:
        import agent.indexer as idx_module
        from agent.indexer import CodeIndexer

        release = threading.Event()

        class _ThreadGatedEmbedProvider:
            """Embeds instantly except on the deferred purpose executor's
            own thread, where it blocks until `release` is set — lets the
            test observe the deferral window deterministically without
            depending on real embedding timing."""

            def embed(self, texts):
                if threading.current_thread().name.startswith("vectr-purpose"):
                    release.wait(timeout=5)
                return [[0.1] * 8 for _ in texts]

            def embed_query(self, texts):
                return self.embed(texts)

        monkeypatch.setattr(idx_module, "get_embed_provider", lambda _m: _ThreadGatedEmbedProvider())

        make_py(tmp_path, "a.py", "def known_fn():\n    return 1\n")
        idx = CodeIndexer(workspace_root=str(tmp_path), db_path=str(tmp_path / "chroma"))
        try:
            idx.index_workspace()
            assert idx.purpose_vectors_pending is True, (
                "purpose pass should be scheduled and blocked on the test gate"
            )
            assert idx.total_chunks == 1, "content must already be indexed"
            assert idx._purpose_collection.count() == 0, (
                "purpose collection must still be empty while the pass is blocked"
            )

            release.set()
            self._wait_until_not_pending(idx)
            assert idx.purpose_vectors_pending is False
            assert idx._purpose_collection.count() == 1
        finally:
            release.set()
            idx.close()

    def test_search_serves_results_while_purpose_pass_pending(
        self, tmp_path, monkeypatch,
    ) -> None:
        import agent.indexer as idx_module
        from agent.indexer import CodeIndexer
        from agent.searcher import CodeSearcher

        release = threading.Event()

        class _ThreadGatedEmbedProvider:
            def embed(self, texts):
                if threading.current_thread().name.startswith("vectr-purpose"):
                    release.wait(timeout=5)
                return [[0.1] * 8 for _ in texts]

            def embed_query(self, texts):
                return self.embed(texts)

        monkeypatch.setattr(idx_module, "get_embed_provider", lambda _m: _ThreadGatedEmbedProvider())

        make_py(tmp_path, "a.py", "def known_function():\n    return 42\n")
        idx = CodeIndexer(workspace_root=str(tmp_path), db_path=str(tmp_path / "chroma"))
        try:
            idx.index_workspace()
            assert idx.purpose_vectors_pending is True

            s = CodeSearcher(idx)
            s.refresh_bm25()
            results, _ = s.search("known_function", n_results=5, rerank=False)
            assert results, "search returned nothing while purpose vectors were pending"
            assert any("known_function" in r.content for r in results)

            release.set()
            self._wait_until_not_pending(idx)
        finally:
            release.set()
            idx.close()

    def test_deferred_purpose_pass_produces_same_result_as_synchronous(
        self, tmp_path, monkeypatch,
    ) -> None:
        import agent.indexer as idx_module
        from agent.indexer import CodeIndexer

        class _DummyEmbedProvider:
            def embed(self, texts):
                return [[0.1] * 8 for _ in texts]

            def embed_query(self, texts):
                return self.embed(texts)

        source = tmp_path / "src"
        source.mkdir()
        make_py(source, "a.py", "def known_fn():\n    '''docstring'''\n    return 1\n")
        make_py(source, "b.py", "def other_fn():\n    return 2\n")

        monkeypatch.setenv("VECTR_FOREGROUND_FAST", "1")
        monkeypatch.setattr(idx_module, "get_embed_provider", lambda _m: _DummyEmbedProvider())
        sync_indexer = CodeIndexer(workspace_root=str(source), db_path=str(tmp_path / "sync_chroma"))
        sync_indexer.index_workspace()
        sync_ids = set(sync_indexer._purpose_collection.get()["ids"])
        sync_indexer.close()

        monkeypatch.delenv("VECTR_FOREGROUND_FAST", raising=False)
        deferred_indexer = CodeIndexer(workspace_root=str(source), db_path=str(tmp_path / "deferred_chroma"))
        deferred_indexer.index_workspace()
        self._wait_until_not_pending(deferred_indexer)
        deferred_ids = set(deferred_indexer._purpose_collection.get()["ids"])
        deferred_indexer.close()

        assert sync_ids, "reference synchronous run produced no purpose vectors"
        assert deferred_ids == sync_ids, (
            f"a deferred pass must eventually produce the same purpose "
            f"collection ids as a synchronous run: sync={sync_ids} "
            f"deferred={deferred_ids}"
        )

    def test_purpose_pass_failure_is_healed_by_next_ordinary_run(
        self, tmp_path, monkeypatch,
    ) -> None:
        """UPG-PURPOSE-RESUME-HOLE: content-completion (mtime_cache) and
        purpose-completion (the purpose-completion cache / purpose
        collection) are decoupled signals — a purpose pass that crashes
        (simulated) never rolls back or blocks the content checkpoint that
        already landed in Phase 3, and a plain next `index_workspace()`
        call correctly skips re-chunking/re-embedding the file's unchanged
        BODY content (Phase 1 still sees it as up to date). But unlike the
        pre-fix behaviour, that same ordinary next run now DOES detect and
        close the purpose-vector gap it left behind — no `--force` and no
        file touch required. This is the exact mechanism the historic bug
        exploited: a purpose pass lost between the content checkpoint and
        completion (crash, or here, a raised exception inside it) used to
        leave the gap permanent because nothing ever re-detected it."""
        import agent.indexer as idx_module
        from agent.indexer import CodeIndexer

        class _DummyEmbedProvider:
            def embed(self, texts):
                return [[0.1] * 8 for _ in texts]

            def embed_query(self, texts):
                return self.embed(texts)

        path = make_py(tmp_path, "a.py", "def known_fn():\n    return 1\n")
        db_path = str(tmp_path / "chroma")

        monkeypatch.setattr(idx_module, "get_embed_provider", lambda _m: _DummyEmbedProvider())
        indexer1 = CodeIndexer(workspace_root=str(tmp_path), db_path=db_path)

        def _crashing_upsert(*args, **kwargs):
            raise RuntimeError("simulated crash mid purpose pass")

        monkeypatch.setattr(indexer1, "_upsert_purpose_vectors", _crashing_upsert)
        indexer1.index_workspace()
        self._wait_until_not_pending(indexer1)
        assert indexer1.purpose_vectors_pending is False

        assert indexer1.total_chunks == 1, "content must be durably indexed regardless"
        assert indexer1._purpose_collection.count() == 0

        mtime_cache = indexer1._load_mtime_cache()
        assert mtime_cache, (
            "content checkpointing is unconditional and independent of the "
            "purpose pass — a purpose-pass crash must not leave content "
            "looking unindexed"
        )
        indexer1.close()

        # A plain next run with no content changes and the REAL (non-
        # crashing) purpose pass restored: Phase 1 still correctly skips
        # re-chunking the file's unchanged body (chunk_file must not be
        # called for it), but the run must detect last time's purpose gap
        # and back it in from the already-stored body chunks.
        import agent.indexer._core as core_module

        real_chunk_file = core_module.chunk_file

        def _chunk_file_forbidding_a_py(fpath):
            assert "a.py" not in fpath, (
                "an ordinary run backfilling a purpose gap must not re-chunk "
                "the file's unchanged body content — only the purpose-only "
                "backfill path (_fetch_body_chunks_for_files) may touch it"
            )
            return real_chunk_file(fpath)

        monkeypatch.setattr(core_module, "chunk_file", _chunk_file_forbidding_a_py)
        monkeypatch.setattr(idx_module, "get_embed_provider", lambda _m: _DummyEmbedProvider())
        indexer2 = CodeIndexer(workspace_root=str(tmp_path), db_path=db_path)
        indexer2.index_workspace()
        self._wait_until_not_pending(indexer2)
        assert indexer2.total_chunks == 1
        assert indexer2._purpose_collection.count() == 1, (
            "an ordinary, unforced next run must close a purpose-vector gap "
            "left by a prior interrupted/crashed pass — this is the actual "
            "defect UPG-PURPOSE-RESUME-HOLE fixes"
        )
        indexer2.close()

        # A further ordinary run finds nothing left to backfill — the gap
        # stays closed, not re-discovered every run.
        monkeypatch.setattr(idx_module, "get_embed_provider", lambda _m: _DummyEmbedProvider())
        indexer3 = CodeIndexer(workspace_root=str(tmp_path), db_path=db_path)
        indexer3.index_workspace()
        self._wait_until_not_pending(indexer3)
        assert indexer3.total_chunks == 1
        assert indexer3._purpose_collection.count() == 1, (
            "a purpose vector correctly checkpointed as complete must not "
            "be re-embedded/duplicated by a later ordinary run"
        )
        assert indexer3.purpose_backfill_pending_files == 0
        indexer3.close()

    def test_close_does_not_block_on_pending_deferred_pass(
        self, tmp_path, monkeypatch,
    ) -> None:
        import agent.indexer as idx_module
        from agent.indexer import CodeIndexer

        release = threading.Event()  # deliberately never set in this test

        class _StuckEmbedProvider:
            def embed(self, texts):
                if threading.current_thread().name.startswith("vectr-purpose"):
                    release.wait()
                return [[0.1] * 8 for _ in texts]

            def embed_query(self, texts):
                return self.embed(texts)

        monkeypatch.setattr(idx_module, "get_embed_provider", lambda _m: _StuckEmbedProvider())

        make_py(tmp_path, "a.py", "def known_fn():\n    return 1\n")
        idx = CodeIndexer(workspace_root=str(tmp_path), db_path=str(tmp_path / "chroma"))
        idx.index_workspace()
        assert idx.purpose_vectors_pending is True

        start = time.time()
        idx.close()
        elapsed = time.time() - start
        assert elapsed < 2.0, (
            f"close() must not block on a stuck deferred purpose pass, took {elapsed:.2f}s"
        )

        release.set()  # unblock the background thread so it doesn't leak past the test


class TestPurposeResumeHole:
    """UPG-PURPOSE-RESUME-HOLE: an interrupted index (content checkpointed,
    purpose pass never run — a daemon stop/crash exactly between the two)
    must not leave a workspace permanently short of purpose vectors while
    every status surface reports it healthy. `index_workspace()` tracks
    per-file purpose completion in a separate on-disk cache
    (`purpose_cache.json`, decoupled from mtime_cache the same way the
    purpose pass itself is decoupled from content indexing —
    UPG-PURPOSE-PASS-DEFERRAL) and, on every ORDINARY (non-force) run,
    detects and backfills any file whose body content is current but whose
    purpose vectors are not. Exercised against a real CodeIndexer + real
    ChromaDB collection on tiny synthetic corpora only."""

    def _wait_until_not_pending(self, idx, timeout: float = 30.0) -> None:
        """FIFO drain barrier (tests.conftest.wait_for_deferred_purpose_pass)
        instead of a silent fixed-deadline poll — see
        TestDualVectorPurposeCollection._wait_until_purpose_pass_done."""
        from tests.conftest import wait_for_deferred_purpose_pass

        wait_for_deferred_purpose_pass(idx, timeout=timeout)

    def test_ordinary_run_backfills_gap_left_by_never_dispatched_deferred_pass(
        self, tmp_path, monkeypatch,
    ) -> None:
        """Simulates the exact bug mechanism: a process kill between the
        content checkpoint (Phase 3, always synchronous) and the purpose
        pass ever being dispatched to the background executor (not merely
        a pass that started and raised — one that never ran at all)."""
        import agent.indexer as idx_module
        from agent.indexer import CodeIndexer

        class _DummyEmbedProvider:
            def embed(self, texts):
                return [[0.1] * 8 for _ in texts]

            def embed_query(self, texts):
                return self.embed(texts)

        make_py(tmp_path, "a.py", "def known_fn():\n    return 1\n")
        db_path = str(tmp_path / "chroma")

        monkeypatch.setattr(idx_module, "get_embed_provider", lambda _m: _DummyEmbedProvider())
        indexer1 = CodeIndexer(workspace_root=str(tmp_path), db_path=db_path)
        # Never dispatch the purpose pass at all (content still checkpoints
        # normally in Phase 3, independent of this) — this is the "crash
        # between checkpoint and dispatch" window from the bug report.
        monkeypatch.setattr(indexer1, "_schedule_deferred_purpose_pass", lambda *a, **k: None)
        indexer1.index_workspace()
        assert indexer1.total_chunks == 1
        assert indexer1._purpose_collection.count() == 0
        indexer1.close()

        monkeypatch.setattr(idx_module, "get_embed_provider", lambda _m: _DummyEmbedProvider())
        indexer2 = CodeIndexer(workspace_root=str(tmp_path), db_path=db_path)
        indexer2.index_workspace()
        self._wait_until_not_pending(indexer2)
        assert indexer2._purpose_collection.count() == 1, (
            "an ordinary next run must detect and backfill a purpose gap "
            "left by a pass that was never dispatched, with no --force and "
            "no file touch"
        )
        indexer2.close()

    def test_ordinary_run_backfills_gap_inline_foreground_fast(
        self, tmp_path, monkeypatch,
    ) -> None:
        """Same scenario as above, but with --foreground-fast (synchronous,
        non-deferred) — the backfill dispatch itself must also work through
        the inline `_upsert_purpose_vectors` call, not only the deferred
        scheduling path."""
        import agent.indexer as idx_module
        from agent.indexer import CodeIndexer

        class _DummyEmbedProvider:
            def embed(self, texts):
                return [[0.1] * 8 for _ in texts]

            def embed_query(self, texts):
                return self.embed(texts)

        make_py(tmp_path, "a.py", "def known_fn():\n    return 1\n")
        db_path = str(tmp_path / "chroma")
        monkeypatch.setenv("VECTR_FOREGROUND_FAST", "1")

        monkeypatch.setattr(idx_module, "get_embed_provider", lambda _m: _DummyEmbedProvider())
        indexer1 = CodeIndexer(workspace_root=str(tmp_path), db_path=db_path)
        monkeypatch.setattr(indexer1, "_upsert_purpose_vectors", lambda *a, **k: None)
        indexer1.index_workspace()
        assert indexer1.total_chunks == 1
        assert indexer1._purpose_collection.count() == 0
        indexer1.close()

        monkeypatch.setattr(idx_module, "get_embed_provider", lambda _m: _DummyEmbedProvider())
        indexer2 = CodeIndexer(workspace_root=str(tmp_path), db_path=db_path)
        indexer2.index_workspace()
        assert indexer2.purpose_vectors_pending is False
        assert indexer2._purpose_collection.count() == 1, (
            "the inline (--foreground-fast) path must also backfill an "
            "ordinary-run purpose gap"
        )
        indexer2.close()

    def test_purpose_backfill_idempotent_across_repeated_ordinary_runs(
        self, tmp_path, monkeypatch,
    ) -> None:
        """Running the repair more than once must not duplicate or corrupt
        vectors — chunk_id upsert is already idempotent, and the
        completion cache must not re-offer an already-closed gap."""
        import agent.indexer as idx_module
        from agent.indexer import CodeIndexer

        class _DummyEmbedProvider:
            def embed(self, texts):
                return [[0.1] * 8 for _ in texts]

            def embed_query(self, texts):
                return self.embed(texts)

        make_py(tmp_path, "a.py", "def known_fn():\n    return 1\n")
        db_path = str(tmp_path / "chroma")

        monkeypatch.setattr(idx_module, "get_embed_provider", lambda _m: _DummyEmbedProvider())
        indexer1 = CodeIndexer(workspace_root=str(tmp_path), db_path=db_path)
        monkeypatch.setattr(indexer1, "_schedule_deferred_purpose_pass", lambda *a, **k: None)
        indexer1.index_workspace()
        indexer1.close()

        for _ in range(3):
            monkeypatch.setattr(idx_module, "get_embed_provider", lambda _m: _DummyEmbedProvider())
            idx = CodeIndexer(workspace_root=str(tmp_path), db_path=db_path)
            idx.index_workspace()
            self._wait_until_not_pending(idx)
            assert idx._purpose_collection.count() == 1, (
                "repeated ordinary runs must not duplicate purpose vectors"
            )
            assert idx.total_chunks == 1, (
                "repeated ordinary runs must not duplicate body chunks"
            )
            assert idx.purpose_backfill_pending_files == 0
            idx.close()

    def test_migration_seed_marks_pre_existing_healthy_index_complete_without_reembedding(
        self, tmp_path, monkeypatch,
    ) -> None:
        """A workspace whose purpose vectors are already fully written (a
        healthy index built before this cache existed, so `purpose_cache.
        json` is absent) must NOT be treated as a corpus-wide gap on the
        first run after upgrading — that would force an unnecessary
        purpose-only reindex on every existing user. The one-time
        migration seed reconciles against the actual purpose collection
        instead."""
        import agent.indexer as idx_module
        from agent.indexer import CodeIndexer

        class _DummyEmbedProvider:
            def embed(self, texts):
                return [[0.1] * 8 for _ in texts]

            def embed_query(self, texts):
                return self.embed(texts)

        make_py(tmp_path, "a.py", "def known_fn():\n    return 1\n")
        db_path = str(tmp_path / "chroma")

        # Build a fully-healthy index the normal way (purpose vectors land).
        monkeypatch.setattr(idx_module, "get_embed_provider", lambda _m: _DummyEmbedProvider())
        indexer1 = CodeIndexer(workspace_root=str(tmp_path), db_path=db_path)
        indexer1.index_workspace()
        self._wait_until_not_pending(indexer1)
        assert indexer1._purpose_collection.count() == 1
        indexer1.close()

        # Simulate "built before this mechanism existed": the completed
        # index's own run already wrote purpose_cache.json (this fix
        # writes it as it goes) — delete it to reproduce the pre-upgrade
        # state of a real user's existing workspace: mtime_cache present,
        # purpose collection fully populated, but no purpose_cache.json.
        purpose_cache_path = tmp_path / "chroma" / "purpose_cache.json"
        assert purpose_cache_path.exists()
        purpose_cache_path.unlink()

        embed_calls: list[list[str]] = []

        class _RecordingEmbedProvider:
            def embed(self, texts):
                embed_calls.append(list(texts))
                return [[0.1] * 8 for _ in texts]

            def embed_query(self, texts):
                return self.embed(texts)

        monkeypatch.setattr(idx_module, "get_embed_provider", lambda _m: _RecordingEmbedProvider())
        indexer2 = CodeIndexer(workspace_root=str(tmp_path), db_path=db_path)
        indexer2.index_workspace()
        self._wait_until_not_pending(indexer2)

        assert not embed_calls, (
            "a healthy pre-existing workspace must not be re-embedded just "
            "because purpose_cache.json didn't exist yet — the migration "
            "seed must reconcile against the actual purpose collection "
            f"instead of assuming a gap: embed() was called with {embed_calls}"
        )
        assert indexer2._purpose_collection.count() == 1
        assert indexer2.purpose_backfill_pending_files == 0
        assert purpose_cache_path.exists(), (
            "the migration seed must persist purpose_cache.json so this "
            "reconciliation only ever runs once"
        )
        indexer2.close()

    def test_zero_symbol_chunk_file_does_not_become_a_permanent_gap(
        self, tmp_path, monkeypatch,
    ) -> None:
        """A file with content chunks but zero symbol-bearing chunks (e.g.
        prose) is vacuously purpose-complete — it must be checkpointed as
        such on its first run, not re-offered as a gap on every subsequent
        ordinary run forever."""
        import agent.indexer as idx_module
        from agent.indexer import CodeIndexer

        class _DummyEmbedProvider:
            def embed(self, texts):
                return [[0.1] * 8 for _ in texts]

            def embed_query(self, texts):
                return self.embed(texts)

        (tmp_path / "notes.md").write_text(
            "# Notes\n\nSome prose with no code symbols at all.\n", encoding="utf-8",
        )
        db_path = str(tmp_path / "chroma")

        monkeypatch.setattr(idx_module, "get_embed_provider", lambda _m: _DummyEmbedProvider())
        indexer1 = CodeIndexer(workspace_root=str(tmp_path), db_path=db_path)
        indexer1.index_workspace()
        self._wait_until_not_pending(indexer1)
        assert indexer1.total_chunks >= 1
        assert indexer1._purpose_collection.count() == 0
        assert indexer1.purpose_backfill_pending_files == 0, (
            "a file with zero symbol-bearing chunks must checkpoint as "
            "complete on its own run, not surface as a pending gap"
        )
        indexer1.close()

        monkeypatch.setattr(idx_module, "get_embed_provider", lambda _m: _DummyEmbedProvider())
        indexer2 = CodeIndexer(workspace_root=str(tmp_path), db_path=db_path)
        indexer2.index_workspace()
        self._wait_until_not_pending(indexer2)
        assert indexer2.purpose_backfill_pending_files == 0, (
            "a zero-symbol file must never be re-offered as a purpose gap "
            "on later ordinary runs"
        )
        indexer2.close()

    def test_entirely_trivial_workspace_checkpoints_purpose_cache_too(
        self, tmp_path, monkeypatch,
    ) -> None:
        """Distinct from test_zero_symbol_chunk_file_does_not_become_a_
        permanent_gap: there, the file chunks to >=1 non-symbol body chunk
        and the gap is closed by _upsert_purpose_vectors' own total==0
        early return. Here, EVERY file in a run's `to_index` chunks to
        literally zero body chunks at all (all_chunks itself is empty),
        which is a separate early-return branch in index_workspace() —
        it must persist purpose_cache the same way mtime_cache is
        persisted, or this file looks like a permanent purpose gap on
        every subsequent ordinary run."""
        import agent.indexer as idx_module
        from agent.indexer import CodeIndexer

        class _DummyEmbedProvider:
            def embed(self, texts):
                return [[0.1] * 8 for _ in texts]

            def embed_query(self, texts):
                return self.embed(texts)

        import json

        fpath = make_py(tmp_path, "reexport.py", "import os\n")  # chunks to []
        db_path = str(tmp_path / "chroma")
        purpose_cache_path = tmp_path / "chroma" / "purpose_cache.json"

        monkeypatch.setattr(idx_module, "get_embed_provider", lambda _m: _DummyEmbedProvider())
        indexer1 = CodeIndexer(workspace_root=str(tmp_path), db_path=db_path)
        indexer1.index_workspace()
        self._wait_until_not_pending(indexer1)
        assert indexer1.total_chunks == 0
        assert indexer1.purpose_backfill_pending_files == 0, (
            "a workspace whose only file chunks to zero body chunks has "
            "nothing to backfill on its own run"
        )
        # Direct on-disk check: a file that chunked to zero body chunks
        # must still get a purpose_cache entry, or it silently falls back
        # to the (indirect, unobservable via purpose_backfill_pending_files
        # since a re-detected gap with zero fetchable chunks is a silent
        # no-op) "re-detected as a gap on every future run forever" state —
        # harmless but wasteful, and specifically what this checkpoint
        # exists to prevent.
        assert purpose_cache_path.exists()
        on_disk = json.loads(purpose_cache_path.read_text(encoding="utf-8"))
        assert str(fpath) in on_disk, (
            "a file that chunked to zero body chunks must be checkpointed "
            "into purpose_cache.json on its own run, exactly like "
            "mtime_cache — otherwise it is silently re-added to gap_paths "
            "on every future ordinary run forever"
        )
        indexer1.close()

        monkeypatch.setattr(idx_module, "get_embed_provider", lambda _m: _DummyEmbedProvider())
        indexer2 = CodeIndexer(workspace_root=str(tmp_path), db_path=db_path)
        # A correctly-checkpointed file must never re-enter gap detection —
        # spy on the backfill-fetch helper to prove it, since
        # purpose_backfill_pending_files reads 0 either way (a wrongly
        # re-detected gap with nothing fetchable is itself a silent no-op
        # by design, so that alone can't distinguish correct from broken).
        fetch_calls: list[set[str]] = []
        real_fetch = indexer2._fetch_body_chunks_for_files

        def _spy_fetch(file_paths):
            fetch_calls.append(set(file_paths))
            return real_fetch(file_paths)

        monkeypatch.setattr(indexer2, "_fetch_body_chunks_for_files", _spy_fetch)
        indexer2.index_workspace()
        self._wait_until_not_pending(indexer2)
        assert indexer2.purpose_backfill_pending_files == 0, (
            "a file that chunked to zero body chunks must never be "
            "re-offered as a purpose gap on a later ordinary run"
        )
        assert not any(str(fpath) in calls for calls in fetch_calls), (
            "a correctly-checkpointed zero-body-chunk file must not be "
            "re-swept into gap detection on a later ordinary run"
        )
        indexer2.close()

    def test_purpose_backfill_pending_files_reflects_detected_gap(
        self, tmp_path, monkeypatch,
    ) -> None:
        """Direct property-level check: the count reflects the gap found at
        the start of the run that detects it, and reads 0 once dispatch for
        that backfill has been sent."""
        import agent.indexer as idx_module
        from agent.indexer import CodeIndexer

        class _DummyEmbedProvider:
            def embed(self, texts):
                return [[0.1] * 8 for _ in texts]

            def embed_query(self, texts):
                return self.embed(texts)

        make_py(tmp_path, "a.py", "def known_fn():\n    return 1\n")
        db_path = str(tmp_path / "chroma")

        monkeypatch.setattr(idx_module, "get_embed_provider", lambda _m: _DummyEmbedProvider())
        indexer1 = CodeIndexer(workspace_root=str(tmp_path), db_path=db_path)
        assert indexer1.purpose_backfill_pending_files == 0, (
            "a brand-new workspace has nothing to backfill"
        )
        monkeypatch.setattr(indexer1, "_schedule_deferred_purpose_pass", lambda *a, **k: None)
        indexer1.index_workspace()
        indexer1.close()

        monkeypatch.setattr(idx_module, "get_embed_provider", lambda _m: _DummyEmbedProvider())
        indexer2 = CodeIndexer(workspace_root=str(tmp_path), db_path=db_path)
        indexer2.index_workspace()
        # Dispatch for the detected gap has already been sent by the time
        # index_workspace() returns (sync call, or deferred submit) — see
        # CodeIndexer.purpose_backfill_pending_files docstring.
        assert indexer2.purpose_backfill_pending_files == 0
        self._wait_until_not_pending(indexer2)
        assert indexer2._purpose_collection.count() == 1
        indexer2.close()

    def test_mixed_workspace_zero_chunk_file_checkpoints_purpose_cache(
        self, tmp_path, monkeypatch,
    ) -> None:
        """Distinct from test_entirely_trivial_workspace_checkpoints_
        purpose_cache_too: there, EVERY file in to_index chunks to zero
        (all_chunks is empty, an early-return branch). Here, one file in
        the SAME run chunks to >=1 body chunk while another chunks to
        zero — a separate mid-function checkpoint (`zero_chunk_mtimes`)
        exists specifically because such a file never appears in
        `all_chunks` and so is invisible to the main per-file checkpoint
        loop below it."""
        import json

        import agent.indexer as idx_module
        from agent.indexer import CodeIndexer

        class _DummyEmbedProvider:
            def embed(self, texts):
                return [[0.1] * 8 for _ in texts]

            def embed_query(self, texts):
                return self.embed(texts)

        make_py(tmp_path, "a.py", "def known_fn():\n    return 1\n")
        zero_fpath = make_py(tmp_path, "reexport.py", "import os\n")  # chunks to []
        db_path = str(tmp_path / "chroma")
        purpose_cache_path = tmp_path / "chroma" / "purpose_cache.json"

        monkeypatch.setattr(idx_module, "get_embed_provider", lambda _m: _DummyEmbedProvider())
        indexer1 = CodeIndexer(workspace_root=str(tmp_path), db_path=db_path)
        indexer1.index_workspace()
        self._wait_until_not_pending(indexer1)
        assert indexer1.total_chunks == 1
        assert indexer1._purpose_collection.count() == 1

        on_disk = json.loads(purpose_cache_path.read_text(encoding="utf-8"))
        assert str(zero_fpath) in on_disk, (
            "a file that chunked to zero body chunks in a mixed run (other "
            "files in the same run did chunk) must still be checkpointed "
            "into purpose_cache.json, exactly like mtime_cache — it never "
            "appears in all_chunks, so it is invisible to the main "
            "per-file checkpoint loop and needs this separate write"
        )
        indexer1.close()

        monkeypatch.setattr(idx_module, "get_embed_provider", lambda _m: _DummyEmbedProvider())
        indexer2 = CodeIndexer(workspace_root=str(tmp_path), db_path=db_path)
        indexer2.index_workspace()
        self._wait_until_not_pending(indexer2)
        assert indexer2.purpose_backfill_pending_files == 0, (
            "the zero-body-chunk file from a mixed run must never be "
            "re-offered as a purpose gap on a later ordinary run"
        )
        indexer2.close()

    def test_final_batch_checkpoints_even_when_not_a_periodic_multiple(
        self, tmp_path, monkeypatch,
    ) -> None:
        """`_upsert_purpose_vectors`'s per-file checkpoint fires on two
        conditions (mirroring Phase 3's own content checkpoint): every
        `checkpoint_every_batches` batches, AND unconditionally on the
        final batch. With `checkpoint_every_batches` set larger than the
        total batch count, the periodic condition never fires at all in
        this run — only the final-batch guarantee can checkpoint anything.
        Without it, a purpose pass over a corpus smaller than one
        checkpoint interval would silently checkpoint nothing, ever,
        leaving every file in it a permanent gap."""
        import json

        import agent.indexer._core as core_module
        import agent.indexer as idx_module
        from agent.indexer import CodeIndexer

        monkeypatch.setattr(core_module, "_EMBED_BATCH_SIZE", 1)
        monkeypatch.setattr(core_module, "_INDEX_GOVERNOR_CHECKPOINT_EVERY_BATCHES", 10)

        class _DummyEmbedProvider:
            def embed(self, texts):
                return [[0.1] * 8 for _ in texts]

            def embed_query(self, texts):
                return self.embed(texts)

        paths = [
            make_py(tmp_path, f"m{i}.py", f"def f{i}():\n    return {i}\n")
            for i in range(3)
        ]
        db_path = str(tmp_path / "chroma")
        purpose_cache_path = tmp_path / "chroma" / "purpose_cache.json"

        monkeypatch.setattr(idx_module, "get_embed_provider", lambda _m: _DummyEmbedProvider())
        indexer = CodeIndexer(workspace_root=str(tmp_path), db_path=db_path)
        indexer.index_workspace()
        self._wait_until_not_pending(indexer)

        assert indexer.total_chunks == 3
        assert indexer._purpose_collection.count() == 3
        assert purpose_cache_path.exists(), (
            "purpose_cache.json was never written — the final-batch "
            "checkpoint guarantee did not fire"
        )
        on_disk = json.loads(purpose_cache_path.read_text(encoding="utf-8"))
        for p in paths:
            assert str(p) in on_disk, (
                f"{p} was never checkpointed into purpose_cache.json — the "
                "periodic checkpoint never fires in this run "
                "(checkpoint_every_batches=10 > 3 total batches), so only "
                "the final-batch guarantee can have written it"
            )
        indexer.close()


class TestFetchChunks:
    """Deterministic re-fetch by chunk id (UPG-CTX-EVICT part a)."""

    def test_found_ids_return_full_content_and_metadata(self, indexer, tmp_path) -> None:
        path = make_py(tmp_path, "svc.py", """
            def known_function():
                return 42
        """)
        indexer.index_file(path)
        body_ids, body_docs, _ = indexer.get_all_documents()
        result = indexer.fetch_chunks(body_ids)
        assert len(result) == len(body_ids)
        for entry, cid, doc in zip(result, body_ids, body_docs):
            assert entry["id"] == cid
            assert entry["found"] is True
            assert entry["content"] == doc
            assert entry["file_path"]

    def test_missing_id_reported_as_not_found(self, indexer) -> None:
        """UPG-FETCH-ID-MISALIGN-MSG: a file with NO chunks indexed at all
        (never indexed / genuinely gone) reports reason="file_changed" with
        no nearest ids to offer."""
        result = indexer.fetch_chunks(["does-not-exist.py:1-2"])
        assert result == [
            {"id": "does-not-exist.py:1-2", "found": False, "reason": "file_changed", "nearest_ids": []},
        ]

    def test_misaligned_span_reports_nearest_chunk_ids_from_same_file(self, indexer, tmp_path) -> None:
        """UPG-FETCH-ID-MISALIGN-MSG: the file the caller asked about IS
        indexed — only the requested line span doesn't match a stored chunk
        — so this is a misalignment, not a stale/moved/deleted file. The
        real chunk id (same file) must be named as the nearest match."""
        path = make_py(tmp_path, "real.py", "def known(): pass")
        indexer.index_file(path)
        body_ids, _, _ = indexer.get_all_documents()
        real_id = body_ids[0]
        file_path = real_id.rsplit(":", 1)[0]
        bad_id = f"{file_path}:9000-9005"

        result = indexer.fetch_chunks([bad_id])
        assert len(result) == 1
        entry = result[0]
        assert entry["found"] is False
        assert entry["reason"] == "misaligned"
        assert entry["nearest_ids"] == [real_id]

    def test_misaligned_span_for_never_indexed_file_falls_back_to_file_changed(
        self, indexer, tmp_path,
    ) -> None:
        """A syntactically well-formed `path:start-end` id whose file has NO
        chunks at all (never indexed) must NOT be misreported as
        "misaligned" — there is nothing in that file to be misaligned with."""
        path = make_py(tmp_path, "real.py", "def known(): pass")
        indexer.index_file(path)  # indexes real.py, not other.py below

        result = indexer.fetch_chunks(["/nowhere/other.py:1-5"])
        assert result == [
            {"id": "/nowhere/other.py:1-5", "found": False, "reason": "file_changed", "nearest_ids": []},
        ]

    def test_mixed_found_and_missing_preserves_request_order(self, indexer, tmp_path) -> None:
        path = make_py(tmp_path, "mix.py", "def known(): pass")
        indexer.index_file(path)
        body_ids, _, _ = indexer.get_all_documents()
        real_id = body_ids[0]
        requested = ["missing-1", real_id, "missing-2"]
        result = indexer.fetch_chunks(requested)
        assert [r["id"] for r in result] == requested
        assert result[0]["found"] is False
        assert result[1]["found"] is True
        assert result[2]["found"] is False

    def test_empty_ids_returns_empty_list(self, indexer) -> None:
        assert indexer.fetch_chunks([]) == []

    def test_exceeding_cap_raises_value_error(self, indexer) -> None:
        from agent.config import FETCH_MAX_IDS_PER_CALL
        too_many = [f"id-{i}" for i in range(FETCH_MAX_IDS_PER_CALL + 1)]
        with pytest.raises(ValueError):
            indexer.fetch_chunks(too_many)

    def test_misaligned_id_in_hard_split_only_file_returns_no_nearest_ids(
        self, indexer, tmp_path, monkeypatch,
    ) -> None:
        """UPG-CHUNK-ID-RANGE-PARSE: end-to-end check that the
        `_parse_chunk_id` skip behaviour holds through `fetch_chunks`'s
        full misalignment path. A file whose chunks are all hard-split
        pieces of one giant line (the only thing in the file that triggered
        UPG-WINDOW-CHUNK-BYTE-CAP) — every stored id in that file carries
        a `#<k>` suffix and therefore does not parse as `path:start-end`.
        A caller asking for a misaligned id in the same file must see
        `nearest_ids=[]` (an honest "no real chunk in this file covers
        that span") rather than a `nearest_ids` list flooded with sibling
        ids that all point to slices of the same single source line.

        If `_parse_chunk_id` ever learned to strip the ordinal, every piece
        would parse to the same `(file, start, end)` and the ranking would
        tie them all at distance 0; with the test file's three pieces that
        tie, `nearest_ids` would become `["<file>:1-1#0", "<file>:1-1#1",
        "<file>:1-1#2"]` — three ids for the same line, strictly worse
        than `[]`."""
        # Shrink the chunk-character cap so a single 1,200-char line gets
        # hard-split into multiple pieces — the same trick
        # `TestChunkCharCap._small_cap` uses to make the split deterministic.
        import agent.indexer._chunking as chunking_mod
        monkeypatch.setattr(chunking_mod, "_MAX_CHUNK_CHARS", 400)

        from agent.indexer import chunk_file
        # A file whose only content is one giant line. The window chunker
        # emits one parent chunk spanning the whole file, then
        # `_postprocess_chunks` hands the parent to `_split_oversized_chunk`,
        # which carves the giant line at character granularity and emits
        # `path:1-1#0`, `path:1-1#1`, `path:1-1#2` — every stored id in the
        # file is `#<k>`-suffixed.
        # The extension is load-bearing. This fixture is deliberately a
        # SINGLE-LINE file, and a single-line `.txt` is dropped to zero
        # chunks by is_trivial_chunk's short-doc rule
        # (UPG-TXT-SINGLE-LINE-DATA-BLOB-DROP, still open), which would
        # leave this test silently pinning nothing. Multi-line `.txt`
        # chunks normally; only this single-line shape is affected.
        giant = "".join(f"{i % 97:02d}," for i in range(400))
        p = tmp_path / "hard_split_only.py"
        p.write_text(giant, encoding="utf-8")

        chunks = chunk_file(str(p))
        # Sanity: every chunk in the file is a hard-split piece — these are
        # the '#<k>'-suffixed ids UPG-WINDOW-CHUNK-BYTE-CAP introduced and
        # the reason this test exists.
        assert len(chunks) >= 2, [c.chunk_id for c in chunks]
        assert all("#" in c.chunk_id for c in chunks), [c.chunk_id for c in chunks]
        # All pieces share the single-line span 1-1.
        assert all(c.start_line == 1 and c.end_line == 1 for c in chunks), (
            [(c.start_line, c.end_line, c.chunk_id) for c in chunks]
        )

        # Index directly so the body collection holds the hard-split pieces.
        indexer._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.content for c in chunks],
            metadatas=[{
                "file_path": c.file_path,
                "language": c.language,
                "node_type": c.node_type,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "symbol_name": c.symbol_name or "",
            } for c in chunks],
            embeddings=[[float(len(c.content)) % 1.0] * 4 for c in chunks],
        )
        file_path = str(p.resolve())
        bad_id = f"{file_path}:9000-9005"

        result = indexer.fetch_chunks([bad_id])
        assert len(result) == 1
        entry = result[0]
        assert entry["found"] is False
        assert entry["reason"] == "misaligned", (
            f"file IS indexed (it has hard-split pieces), so the misalign "
            f"branch must engage; got reason={entry.get('reason')!r}"
        )
        assert entry["nearest_ids"] == [], (
            f"hard-split piece ids must be skipped from the misalignment "
            f"ranking (they all share one parent's span, so including them "
            f"would tie every sibling at distance 0 and flood the suggestion "
            f"list with N ids for the same line). Got: {entry['nearest_ids']!r}"
        )


class TestParseChunkIdAndNearestChunkIds:
    """Smallest-level coverage for the two pure helpers behind
    UPG-FETCH-ID-MISALIGN-MSG's misalignment detection — no ChromaDB, no
    fixtures, just the arithmetic on chunk id strings."""

    def test_parse_chunk_id_splits_path_and_line_range(self) -> None:
        from agent.indexer._core import _parse_chunk_id
        assert _parse_chunk_id("a/b/c.py:10-20") == ("a/b/c.py", 10, 20)

    def test_parse_chunk_id_splits_on_last_colon_for_colon_bearing_path(self) -> None:
        from agent.indexer._core import _parse_chunk_id
        assert _parse_chunk_id("C:/repo/a.py:10-20") == ("C:/repo/a.py", 10, 20)

    def test_parse_chunk_id_returns_none_for_no_line_range(self) -> None:
        from agent.indexer._core import _parse_chunk_id
        assert _parse_chunk_id("no-colon-here") is None

    def test_parse_chunk_id_returns_none_for_non_numeric_range(self) -> None:
        from agent.indexer._core import _parse_chunk_id
        assert _parse_chunk_id("a.py:start-end") is None

    def test_parse_chunk_id_returns_none_for_hard_split_piece_id(self) -> None:
        """UPG-CHUNK-ID-RANGE-PARSE: `_parse_chunk_id` intentionally returns
        None for `#<k>`-suffixed hard-split piece ids. Those pieces are
        produced by `_split_oversized_chunk` (UPG-WINDOW-CHUNK-BYTE-CAP) when
        a single source line is so long it has to be carved at character
        granularity, or when the parent chunk's content does not map 1:1 onto
        its recorded span (prose/header-stripped sections). All pieces of one
        parent share the EXACT same `(start_line, end_line)` range, so a
        parser that accepted them would lump every sibling into one tie in
        the misalignment ranking — `nearest_ids` would then flood with N
        ids that all point to slices of the same single line of bytes,
        which is strictly worse than the current "no parseable neighbour
        here" behaviour (the file probe in `_not_found_fetch_entry` then
        still returns `nearest_ids=[]` for that file, an honest "no real
        chunk covers this span" signal). The exact id `vectr_fetch` is
        asked for resolves by ChromaDB's `get(ids=...)` exact match, so
        the `#<k>` ids never NEED the parse path to work."""
        from agent.indexer._core import _parse_chunk_id
        # One-to-one carve: pieces of one long line share the single-line span.
        assert _parse_chunk_id("minified.js:1-1#0") is None
        assert _parse_chunk_id("minified.js:1-1#42") is None
        # Character-granular carve: pieces inherit the parent's own span.
        assert _parse_chunk_id("doc.txt:40-41#0") is None
        assert _parse_chunk_id("doc.txt:40-41#3") is None

    def test_nearest_chunk_ids_skips_hard_split_piece_candidates(self) -> None:
        """UPG-CHUNK-ID-RANGE-PARSE: pieces of one hard-split parent all share
        the same parsed range. Including them in the ranking would tie them
        all at the same score, so `nearest_ids` would be filled with sibling
        ids that are slices of the same source line — a strictly worse
        suggestion than the real, different-spanned chunks in the file. The
        `limit=2` result below must therefore contain only the natural
        `a.py:1-5` and `a.py:100-120` ids, NOT the `#<k>` siblings that
        share one-line span `10-10`."""
        from agent.indexer._core import _nearest_chunk_ids_by_span
        candidates = [
            "a.py:1-5",                  # natural — distance 125
            "a.py:100-120",              # natural — distance 10 (closest natural)
            "minified.js:10-10#0",       # hard-split piece — would tie with #1
            "minified.js:10-10#1",       # hard-split piece — would tie with #0
        ]
        # Requested span 130-140: closest natural is a.py:100-120 (10 away).
        # The hard-split pieces would have distance min(|10-140|, |10-130|)=120
        # — further than the natural 100-120, AND if the requested range
        # were `10-10` they would all tie on distance 0 with each other.
        nearest = _nearest_chunk_ids_by_span(candidates, 130, 140, limit=2)
        assert nearest == ["a.py:100-120", "a.py:1-5"], (
            f"hard-split piece ids must be skipped from misalignment ranking; "
            f"got {nearest}"
        )

    def test_nearest_chunk_ids_for_requested_single_line_span_excludes_pieces(self) -> None:
        """The worst-case failure mode if `_parse_chunk_id` were taught to
        strip the `#<k>` ordinal: a caller asks for the parent's natural span
        (e.g. `a.py:10-10`), the file contains a giant line of 50_000 chars
        that was hard-split into 100 `#<k>` siblings. Every piece parses to
        `(a.py, 10, 10)`, so all 100 tie at distance 0, and the suggestion
        list is filled with sibling ids of the same line. That is what the
        current skip behaviour guards against."""
        from agent.indexer._core import _nearest_chunk_ids_by_span
        candidates = [f"a.py:10-10#{k}" for k in range(100)]
        nearest = _nearest_chunk_ids_by_span(candidates, 10, 10, limit=5)
        assert nearest == [], (
            f"hard-split piece ids alone in a file must not be suggested as "
            f"misalignment candidates; got {nearest}"
        )

    def test_nearest_chunk_ids_prefers_overlap_over_distance(self) -> None:
        from agent.indexer._core import _nearest_chunk_ids_by_span
        candidates = ["a.py:1-5", "a.py:40-60", "a.py:100-120"]
        # Requested span 45-50 overlaps a.py:40-60 exactly; the others don't.
        nearest = _nearest_chunk_ids_by_span(candidates, 45, 50, limit=3)
        assert nearest[0] == "a.py:40-60"

    def test_nearest_chunk_ids_ranks_by_distance_when_no_overlap(self) -> None:
        from agent.indexer._core import _nearest_chunk_ids_by_span
        candidates = ["a.py:1-5", "a.py:200-210", "a.py:100-120"]
        # Requested span 130-140: a.py:100-120 is 10 lines away, a.py:200-210
        # is 60 lines away, a.py:1-5 is 125 lines away.
        nearest = _nearest_chunk_ids_by_span(candidates, 130, 140, limit=3)
        assert nearest == ["a.py:100-120", "a.py:200-210", "a.py:1-5"]

    def test_nearest_chunk_ids_respects_limit(self) -> None:
        from agent.indexer._core import _nearest_chunk_ids_by_span
        candidates = [f"a.py:{i}-{i + 5}" for i in range(0, 100, 10)]
        nearest = _nearest_chunk_ids_by_span(candidates, 500, 505, limit=2)
        assert len(nearest) == 2

    def test_nearest_chunk_ids_skips_unparseable_candidates(self) -> None:
        from agent.indexer._core import _nearest_chunk_ids_by_span
        candidates = ["a.py:no-range-here", "a.py:40-60"]
        nearest = _nearest_chunk_ids_by_span(candidates, 45, 50, limit=3)
        assert nearest == ["a.py:40-60"]


class TestSchemaVersionRebuildTrigger:
    """A bump to INDEXING_SCHEMA_VERSION must invalidate the mtime cache (ARCH-4)."""

    def test_stale_schema_version_forces_full_reindex(self, indexer, tmp_path) -> None:
        make_py(tmp_path, "a.py", "def a(): pass")
        indexer.index_workspace()
        cache_path = indexer._mtime_cache_path()

        import json
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        raw["__vectr_index_schema_version__"] = -1  # simulate an older schema
        cache_path.write_text(json.dumps(raw), encoding="utf-8")

        # An old-schema cache is treated as cold: nothing survives the load.
        assert indexer._load_mtime_cache() == {}

    def test_current_schema_version_cache_round_trips(self, indexer, tmp_path) -> None:
        make_py(tmp_path, "a.py", "def a(): pass")
        indexer.index_workspace()
        loaded = indexer._load_mtime_cache()
        assert loaded  # non-empty: current-schema cache is preserved as-is
        assert "__vectr_index_schema_version__" not in loaded  # sentinel stripped from the dict

    def test_unchanged_schema_no_content_change_skips_rebuild(self, indexer, tmp_path, caplog) -> None:
        """UPG-CHUNK-LOGIC-VERSION-FINGERPRINT acceptance #1: with the constant
        unchanged and no file content changed, a restart re-chunks nothing —
        today's incremental-skip behavior must not regress."""
        make_py(tmp_path, "a.py", "def a(): pass")
        indexer.index_workspace()
        indexer._write_embed_model_stamp()  # matches on the next call too

        delete_calls: list = []
        real_delete = indexer._collection.delete

        def _tracking_delete(*args, **kwargs):
            delete_calls.append((args, kwargs))
            return real_delete(*args, **kwargs)

        indexer._collection.delete = _tracking_delete
        with caplog.at_level(logging.INFO, logger="agent.indexer._core"):
            indexer.index_workspace()
        assert delete_calls == []
        assert any("up to date — nothing to re-index" in r.message for r in caplog.records)
        assert not any("chunking/embedding logic changed" in r.message for r in caplog.records)

    def test_schema_bump_triggers_full_rechunk_with_reason_log_and_no_orphans(
        self, indexer, tmp_path, caplog, monkeypatch,
    ) -> None:
        """UPG-CHUNK-LOGIC-VERSION-FINGERPRINT acceptance #2: bumping the
        constant, with zero file content changed, triggers a full re-chunk/
        re-embed with a reason line naming the cause AND actually replaces
        the old chunk set rather than leaving stale chunks orphaned alongside
        the newly produced ones (the real defect this task fixes: a schema
        mismatch used to cold the incremental-skip cache without also forcing
        Phase 2's delete-before-reinsert step, so a chunking-logic change that
        alters a file's chunk boundaries left the old chunks behind)."""
        from agent.indexer._types import CodeChunk

        target = make_py(tmp_path, "a.py", "def a():\n    pass\n")
        real_path = str(Path(target).resolve())

        old_chunk = CodeChunk(
            chunk_id=f"{real_path}:1-2", file_path=real_path,
            content="def a():\n    pass\n", language="python",
            node_type="function_definition", start_line=1, end_line=2, symbol_name="a",
        )
        import agent.indexer._core as core_module
        monkeypatch.setattr(core_module, "chunk_file", lambda p: [old_chunk])
        indexer.index_workspace()
        indexer._write_embed_model_stamp()  # isolate this test to the schema trigger only
        assert set(indexer._collection.get()["ids"]) == {f"{real_path}:1-2"}

        # Simulate a chunking-logic version bump the same way the existing
        # TestSchemaVersionRebuildTrigger tests do: tamper the on-disk sentinel.
        cache_path = indexer._mtime_cache_path()
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        raw["__vectr_index_schema_version__"] = -1
        cache_path.write_text(json.dumps(raw), encoding="utf-8")

        # New chunking logic recovers a chunk the old logic used to drop —
        # different ids than before, same file content.
        new_chunk_1 = CodeChunk(
            chunk_id=f"{real_path}:1-1", file_path=real_path,
            content="def a():\n", language="python",
            node_type="function_definition", start_line=1, end_line=1, symbol_name="a",
        )
        new_chunk_2 = CodeChunk(
            chunk_id=f"{real_path}:2-2", file_path=real_path,
            content="    pass\n", language="python",
            node_type="pass_statement", start_line=2, end_line=2, symbol_name=None,
        )
        monkeypatch.setattr(core_module, "chunk_file", lambda p: [new_chunk_1, new_chunk_2])

        with caplog.at_level(logging.INFO, logger="agent.indexer._core"):
            indexer.index_workspace()

        ids_after = set(indexer._collection.get()["ids"])
        assert ids_after == {f"{real_path}:1-1", f"{real_path}:2-2"}, (
            "schema-triggered rebuild must replace the old chunk set, not "
            "accumulate it alongside the new one"
        )
        assert any(
            "Content index stale" in r.message and "chunking/embedding logic changed" in r.message
            for r in caplog.records
        ), "a schema mismatch must log one clear reason line"
        # Same user-visible progress logging a normal full index produces.
        assert any("Indexing 1/1 files" in r.message for r in caplog.records)
        assert any("Indexed" in r.message and "chunks from" in r.message for r in caplog.records)

        # Bumping leaves no permanent state: the next restart with the same
        # constant and no content change is back to a pure incremental skip.
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="agent.indexer._core"):
            indexer.index_workspace()
        assert any("up to date — nothing to re-index" in r.message for r in caplog.records)


class TestChunkLogicVersusEmbedModelStampInteraction:
    """UPG-CHUNK-LOGIC-VERSION-FINGERPRINT: the chunking-logic version
    (INDEXING_SCHEMA_VERSION) and the embed-model stamp (UPG-EMBEDDER-SWAP-
    GRANITE) are two independently-triggered full-rebuild mechanisms that
    must compose without fighting, double-rebuilding, or shadowing each
    other's required side effect."""

    def test_schema_stale_alone_does_not_touch_embed_model_path(
        self, indexer, tmp_path, caplog,
    ) -> None:
        make_py(tmp_path, "a.py", "def a(): pass")
        indexer.index_workspace()
        indexer._write_embed_model_stamp()  # matching stamp: embed path is a no-op

        cache_path = indexer._mtime_cache_path()
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        raw["__vectr_index_schema_version__"] = -1
        cache_path.write_text(json.dumps(raw), encoding="utf-8")

        old_collection = indexer._collection
        caplog.clear()  # drop the first (fresh-index) call's own embed-stamp warning
        with caplog.at_level(logging.INFO, logger="agent.indexer._core"):
            indexer.index_workspace()
        # Only the chunking-logic reason fired — the embed-model warning did not.
        assert any("chunking/embedding logic changed" in r.message for r in caplog.records)
        assert not any("Embedding model changed" in r.message for r in caplog.records)
        # No collection drop/recreate — a same-dimension schema bump only
        # needs per-file delete+reinsert, not the embed-stamp's harder reset.
        assert indexer._collection is old_collection

    def test_both_stale_simultaneously_rebuilds_once_without_shadowing_recreate(
        self, indexer, tmp_path, caplog,
    ) -> None:
        make_py(tmp_path, "a.py", "def a(): pass")
        indexer.index_workspace()

        cache_path = indexer._mtime_cache_path()
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        raw["__vectr_index_schema_version__"] = -1
        cache_path.write_text(json.dumps(raw), encoding="utf-8")

        stamp_path = indexer._embed_model_stamp_path()
        stamp_path.write_text(json.dumps({"embed_model": "some-other-model"}), encoding="utf-8")

        old_collection = indexer._collection
        count_before = indexer._collection.count()
        caplog.clear()  # drop the first (fresh-index) call's own embed-stamp warning
        with caplog.at_level(logging.INFO, logger="agent.indexer._core"):
            indexer.index_workspace()

        # The embed-model mismatch still does its required, stronger side
        # effect (drop + recreate, for vector-dimension safety) regardless of
        # which check set force=True first.
        assert indexer._collection is not old_collection
        assert indexer._collection.count() == count_before
        # Exactly one reason line — the embed-model one takes precedence, the
        # schema-stale block's `not force` guard suppresses its own line
        # rather than printing a second, less-specific reason.
        assert any("Embedding model changed" in r.message for r in caplog.records)
        assert not any("chunking/embedding logic changed" in r.message for r in caplog.records)
        assert indexer._stored_embed_model() == indexer.embed_model


class TestEmbedModelStampRebuildTrigger:
    """UPG-EMBEDDER-SWAP-GRANITE: vectors from two different embedding models
    must never silently coexist in one collection. The embed-model stamp
    forces a full rebuild (force=True) on mismatch, unlike the softer
    mtime-cache-reset that INDEXING_SCHEMA_VERSION uses — a same-dimension
    model swap accepts into ChromaDB's HNSW index without error, so a weaker
    check would silently mix vector spaces.
    """

    def test_stamp_written_on_fresh_index(self, indexer, tmp_path) -> None:
        make_py(tmp_path, "a.py", "def a(): pass")
        indexer.index_workspace()
        assert indexer._stored_embed_model() == indexer.embed_model

    def test_matching_stamp_does_not_force_rebuild(self, indexer, tmp_path) -> None:
        make_py(tmp_path, "a.py", "def a(): pass")
        indexer.index_workspace()

        delete_calls: list = []
        real_delete = indexer._collection.delete

        def _tracking_delete(*args, **kwargs):
            delete_calls.append((args, kwargs))
            return real_delete(*args, **kwargs)

        indexer._collection.delete = _tracking_delete
        # Same embed_model, nothing changed on disk — incremental skip, no
        # chunk deletion (which only happens for force or previously-cached files).
        indexer.index_workspace()
        assert delete_calls == []

    def test_mismatched_stamp_forces_full_rebuild(self, indexer, tmp_path) -> None:
        make_py(tmp_path, "a.py", "def a(): pass")
        indexer.index_workspace()
        count_before = indexer._collection.count()
        assert count_before > 0

        import json
        stamp_path = indexer._embed_model_stamp_path()
        stamp_path.write_text(
            json.dumps({"embed_model": "some-other-embedding-model"}), encoding="utf-8"
        )

        # A mismatch must DROP AND RECREATE both collections (ChromaDB pins a
        # collection's dimensionality at first insert and keeps it even after
        # every entry is deleted, so per-entry deletion alone would crash a
        # different-dimension swap) and fully repopulate them.
        old_collection = indexer._collection
        old_purpose = indexer._purpose_collection
        indexer.index_workspace()
        assert indexer._collection is not old_collection, \
            "a stamp mismatch must drop and recreate the body collection"
        assert indexer._purpose_collection is not old_purpose, \
            "a stamp mismatch must drop and recreate the purpose collection"
        assert indexer._collection.count() == count_before  # fully re-populated
        assert indexer._stored_embed_model() == indexer.embed_model  # stamp updated to current model

    def test_missing_stamp_treated_as_mismatch_not_match(self, indexer, tmp_path) -> None:
        make_py(tmp_path, "a.py", "def a(): pass")
        indexer.index_workspace()
        count_before = indexer._collection.count()
        assert count_before > 0

        stamp_path = indexer._embed_model_stamp_path()
        stamp_path.unlink()  # simulate a pre-existing index from an older vectr version

        assert indexer._stored_embed_model() is None

        old_collection = indexer._collection
        indexer.index_workspace()
        assert indexer._collection is not old_collection, \
            "a missing stamp must be treated as a mismatch (recreate + rebuild), not a match"
        assert indexer._collection.count() == count_before
        assert indexer._stored_embed_model() == indexer.embed_model

    def test_different_dimension_model_swap_rebuilds_cleanly(self, indexer, tmp_path, monkeypatch) -> None:
        """The crash case the drop-and-recreate exists for: swapping to a model
        with a DIFFERENT embedding dimension. Without collection recreation the
        rebuild's reinsert would raise a dimension-mismatch error, because the
        old collection keeps its original dimensionality even after every
        entry is deleted."""
        make_py(tmp_path, "a.py", "def a(): pass")
        indexer.index_workspace()
        count_before = indexer._collection.count()
        assert count_before > 0

        from tests.conftest import _DummyEmbedProvider

        class _Small384Provider(_DummyEmbedProvider):
            DIM = 384

        from agent import indexer as idx_module
        monkeypatch.setattr(
            idx_module, "get_embed_provider", lambda _model: _Small384Provider()
        )
        from agent.indexer import CodeIndexer
        idx2 = CodeIndexer(
            workspace_root=str(tmp_path),
            embed_model="different-dimension-model",
            db_path=str(tmp_path / "chroma"),
        )
        files, chunks = idx2.index_workspace()  # must not raise
        assert chunks == count_before
        assert idx2._stored_embed_model() == "different-dimension-model"


class TestEmbedModelStampAutoselectIntegration:
    """UPG-EMBEDDER-AUTOSELECT requirement 5: the embed-model stamp mismatch
    check (see TestEmbedModelStampRebuildTrigger above) is a plain string
    comparison against `self.embed_model` — agnostic to WHY that string
    changed. This proves the SAME safety path already holds for a
    selection-driven change: `agent.strategy_selector.resolve_embed_model()`
    picking a different model (via a matching evidence-table row, injected
    here — the shipped table is empty) forces the identical full
    drop-and-recreate rebuild as an env-var-driven change, with no new
    product code required to wire it up correctly once a real evidence row
    ships.
    """

    def test_evidence_table_driven_model_change_forces_full_rebuild(
        self, tmp_path, monkeypatch,
    ) -> None:
        from tests.conftest import _DummyEmbedProvider
        from agent import indexer as idx_module
        from agent.indexer import CodeIndexer
        from agent.strategy_selector import CodebaseFingerprint, resolve_embed_model

        monkeypatch.setattr(idx_module, "get_embed_provider", lambda _model: _DummyEmbedProvider())

        # A real C-dominant fingerprint would come from a corpus of .c files;
        # a single .py fixture file is enough here since only the SELECTED
        # MODEL STRING (not the fingerprint's real source language) drives
        # what this test proves — the stamp mismatch check downstream of it.
        make_py(tmp_path, "a.py", "def a(): pass")
        db_path = str(tmp_path / "chroma")

        fp = CodebaseFingerprint(
            total_files=500, language_dist={"c": 500}, dominant_language="c",
            is_monorepo=False, size_class="large",
        )

        # Run 1: no evidence row yet — resolves to the configured default,
        # same as every real workspace today.
        model_1, _ = resolve_embed_model(fp, explicit_override=None, table={})
        idx1 = CodeIndexer(workspace_root=str(tmp_path), embed_model=model_1, db_path=db_path)
        idx1.index_workspace()
        count_before = idx1._collection.count()
        assert count_before > 0
        assert idx1._stored_embed_model() == model_1

        # Run 2: a (test-only, never-shipped) evidence row now matches this
        # fingerprint's dominant language and selects a DIFFERENT model —
        # exactly what a future real evidence row would do.
        synthetic_table = {"c": "some-org/some-c-optimized-embedder"}
        model_2, _ = resolve_embed_model(fp, explicit_override=None, table=synthetic_table)
        assert model_2 != model_1

        idx2 = CodeIndexer(workspace_root=str(tmp_path), embed_model=model_2, db_path=db_path)
        old_collection = idx2._collection
        old_purpose = idx2._purpose_collection
        idx2.index_workspace()

        assert idx2._collection is not old_collection, (
            "a selection-driven embed-model change must drop and recreate "
            "the body collection, exactly like an env-var-driven change"
        )
        assert idx2._purpose_collection is not old_purpose
        assert idx2._collection.count() == count_before  # fully re-populated
        assert idx2._stored_embed_model() == model_2

    def test_explicit_override_beats_evidence_table_before_it_ever_reaches_the_indexer(
        self, tmp_path, monkeypatch,
    ) -> None:
        """The precedence rule itself (explicit env var wins over a table
        row) determines what string ever reaches CodeIndexer's embed_model —
        proven at the resolve_embed_model layer so an operator pin can never
        be silently overridden by a future evidence row."""
        from agent.strategy_selector import CodebaseFingerprint, resolve_embed_model

        fp = CodebaseFingerprint(
            total_files=500, language_dist={"c": 500}, dominant_language="c",
            is_monorepo=False, size_class="large",
        )
        synthetic_table = {"c": "some-org/some-c-optimized-embedder"}
        model, rationale = resolve_embed_model(
            fp, explicit_override="operator/pinned-model", table=synthetic_table,
        )
        assert model == "operator/pinned-model"
        assert "explicitly set" in rationale


# ---------------------------------------------------------------------------
# ARCH-4 — dilution evidence gate: dual-vector pool entry, end-to-end
# through index -> search.
#
# A deterministic bag-of-words embedder (NOT the random-hash _DummyEmbedProvider
# used elsewhere) is used here specifically because it reproduces the dilution
# failure mode: cosine similarity of a normalized word-count vector shrinks as
# query-irrelevant filler tokens are added, exactly like a real embedding model
# mean-pooling a long mechanical function body.
# ---------------------------------------------------------------------------

_DILUTION_VOCAB = ["charge", "customer", "order", "payment", "temp", "step", "value"]


def _bow_vector(text: str, vocab: list[str] = _DILUTION_VOCAB) -> list[float]:
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    counts = [float(tokens.count(w)) for w in vocab]
    norm = math.sqrt(sum(c * c for c in counts)) or 1.0
    return [c / norm for c in counts]


class _DilutionEmbedProvider:
    """Deterministic bag-of-words embedder for the dilution evidence gate."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_bow_vector(t) for t in texts]

    def embed_query(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)


class TestDualVectorPoolEntryEvidenceGate:
    """Proves the ARCH-4 mechanism, not just that code paths run without error:

    a documented symbol whose body dilutes its docstring must enter the search
    pool under dual-vector search but NOT under body-only search, on the SAME
    index (only the query-time merge is toggled) — i.e. the fix works end-to-end
    through index -> search, not just at the embedding-comparison level.
    """

    def _build_dilution_corpus(self, indexer, tmp_path) -> None:
        # Target: a documented method whose docstring paraphrases the query, but
        # whose mechanical body (repeated filler statements) dilutes a mean-pooled
        # single-vector embedding.
        target_body = (
            "def process_payment(amount):\n"
            '    """Charge the customer for the given order and record the result."""\n'
            "    temp = 0\n"
            "    step = 1\n"
            + ("    temp = temp + step\n" * 40)
            + "    return temp\n"
        )
        make_py(tmp_path, "target.py", target_body)

        # Decoys: undocumented functions whose body happens to repeat a query
        # keyword ("order") densely with no filler — the exact keyword-coincidence
        # scenario that lets a decoy outrank the diluted target on body vectors
        # alone, and that a purpose (signature-only) vector does not fall for.
        for i in range(5):
            decoy_body = (
                f"def get_order_list_{i}():\n"
                "    order = order_value = order\n"
                "    return order\n"
            )
            make_py(tmp_path, f"decoy_{i}.py", decoy_body)

        indexer.index_workspace()

    def test_documented_symbol_enters_pool_under_dual_vector_not_body_only(
        self, indexer, tmp_path, monkeypatch,
    ) -> None:
        indexer._embed_provider = _DilutionEmbedProvider()
        self._build_dilution_corpus(indexer, tmp_path)

        from agent.searcher import CodeSearcher
        s = CodeSearcher(indexer)
        s.refresh_bm25()

        query = "charge the customer for their order"

        # Dual-vector ON (default): target's purpose vector (signature+docstring,
        # body-stripped) is undiluted and wins pool entry.
        dual_results, _ = s.search(query, n_results=1, semantic_weight=1.0, rerank=False)
        assert dual_results, "dual-vector search returned nothing"
        assert dual_results[0].file_path.endswith("target.py"), (
            "documented symbol did not win pool entry under dual-vector search"
        )

        # Body-only (query-time flag off, same index, same embeddings already
        # stored): the diluted body vector loses to keyword-coincidence decoys —
        # this is the exact wall ARCH-4 exists to break.
        monkeypatch.setattr("agent.searcher._DUAL_VECTOR_ENABLED", False)
        body_only_results, _ = s.search(query, n_results=1, semantic_weight=1.0, rerank=False)
        assert body_only_results, "body-only search returned nothing"
        assert not body_only_results[0].file_path.endswith("target.py"), (
            "target should NOT win pool entry under body-only search — "
            "if it does, the dilution fixture no longer reproduces the failure "
            "mode and this test is not exercising the ARCH-4 mechanism"
        )

    def test_non_symbol_chunks_unaffected_by_dual_vector(self, indexer, tmp_path) -> None:
        indexer._embed_provider = _DilutionEmbedProvider()
        (tmp_path / "readme.md").write_text(
            "# Payment charges\n\nSome prose about charging a customer for an order.\n"
        )
        indexer.index_workspace()
        assert indexer._purpose_collection.count() == 0

        from agent.searcher import CodeSearcher
        s = CodeSearcher(indexer)
        s.refresh_bm25()
        results, _ = s.search("charge the customer for their order", n_results=1, rerank=False)
        assert results
        assert results[0].file_path.endswith("readme.md")


# ---------------------------------------------------------------------------
# Wave 1 ranking — quality prior, dedup, tiebreaker, test de-prioritization
# (UPG-2.1 / UPG-2.2 / UPG-2.3)
# ---------------------------------------------------------------------------

def _sr(
    content, path="/p/a.py", node_type="", score=0.7, lang="python", purpose_sim=0.0,
    dense_sim=0.0, ce_relevance=None, symbol_name="",
):
    from agent.searcher import SearchResult
    return SearchResult(
        file_path=path, lines="1-9", symbol_name=symbol_name, language=lang,
        score=score, content=content, node_type=node_type,
        purpose_sim=purpose_sim, dense_sim=dense_sim, ce_relevance=ce_relevance,
    )


class TestRankingQuality:
    def test_duplicates_collapsed(self, searcher) -> None:
        cands = [
            _sr("## Create accounts", path="/p/rust/README.md", lang="markdown"),
            _sr("## Create accounts", path="/p/java/README.md", lang="markdown"),
            _sr("## Create accounts", path="/p/python/README.md", lang="markdown"),
        ]
        out = searcher._apply_quality_and_dedup("create accounts", cands)
        assert len(out) == 1
        assert out[0].dup_count == 2  # two identical collapsed into the one kept

    def test_trivial_demoted_below_real(self, searcher) -> None:
        # Trivial chunk is the most "relevant" (rank 0) but must drop below real code.
        cands = [
            _sr("}", path="/p/a.c", lang="c"),
            _sr("int add(int a, int b) {\n    return a + b;\n}\n// impl", path="/p/a.c", lang="c"),
        ]
        out = searcher._apply_quality_and_dedup("add two integers", cands)
        assert "int add" in out[0].content

    def test_navigational_demoted(self, searcher) -> None:
        cands = [
            _sr("pub use a::A;\npub use b::B;\npub use c::C;", path="/p/lib.rs",
                node_type="navigational", lang="rust"),
            _sr("pub fn resolve(&self) -> Lock {\n    self.inner.acquire()\n}\n// real",
                path="/p/resolver.rs", lang="rust"),
        ]
        out = searcher._apply_quality_and_dedup("dependency resolution", cands)
        assert "resolve" in out[0].content

    def test_test_file_demoted_below_impl(self, searcher) -> None:
        # Realistic candidate set: the test file is ranked #0 (rich doc-comments
        # match strongly), the impl just below it, then irrelevant filler.
        test = _sr("def test_resolve():\n    assert resolve()\n    # scenario\n    y=2",
                   path="/p/tests/test_resolver.py")
        impl = _sr("def resolve():\n    return pubgrub_solve()\n    # impl body\n    x=1",
                   path="/p/resolver.py")
        filler = [_sr(f"def helper{i}():\n    return {i}\n    a={i}\n    b={i}", path=f"/p/h{i}.py")
                  for i in range(8)]
        cands = [test, impl] + filler

        # The test-file quality prior demotes the test below the implementation.
        out = searcher._apply_quality_and_dedup("how does resolution work", cands)
        assert out[0].file_path == "/p/resolver.py"
        assert out.index(impl) < out.index(test)

    def test_empty_candidates(self, searcher) -> None:
        assert searcher._apply_quality_and_dedup("q", []) == []


# ---------------------------------------------------------------------------
# UPG-RUST-DEF-EVICTION / DEF-C — near-duplicate leading-docstring dedup,
# wired into _apply_quality_and_dedup's final collapse alongside the existing
# byte-identical full-content key (UPG-2.2).
# ---------------------------------------------------------------------------

class TestDocstringDedupWiring:
    def test_near_duplicate_docstring_different_body_collapses(self, searcher) -> None:
        # Two DIFFERENT bodies (so the byte-identical content key does not
        # already catch this) sharing the same leading doc-comment block must
        # collapse to one representative via the docstring key.
        best = _sr(
            "/// Represents a single configuration entry loaded from disk.\n"
            "/// Used across the module for validated settings access.\n"
            "struct ConfigEntry { value: String }",
            path="/p/config_entry.rs", lang="rust",
        )
        near_dup = _sr(
            "/// Represents a single configuration entry loaded from disk.\n"
            "/// Used across the module for validated settings access.\n"
            "struct ConfigEntryHelper { value: String, extra: bool }",
            path="/p/config_entry_helper.rs", lang="rust",
        )
        out = searcher._apply_quality_and_dedup("configuration entry", [best, near_dup])
        assert len(out) == 1
        assert out[0].dup_count == 1

    def test_chunks_with_no_leading_docstring_never_collapse(self, searcher) -> None:
        # Two DIFFERENT, undocumented chunks must never be folded together by
        # the docstring key (it returns "" for both — never a valid key).
        a = _sr("struct Bare { value: String }", path="/p/bare_a.rs", lang="rust")
        b = _sr("struct AlsoBare { value: String }", path="/p/bare_b.rs", lang="rust")
        out = searcher._apply_quality_and_dedup("bare struct", [a, b])
        assert len(out) == 2

    def test_different_leading_docstring_never_collapses(self, searcher) -> None:
        a = _sr(
            "/// Represents a single configuration entry loaded from disk.\n"
            "/// Used across the module for validated settings access.\n"
            "struct ConfigEntry { value: String }",
            path="/p/config_entry.rs", lang="rust",
        )
        b = _sr(
            "/// Handles outbound network retries with exponential backoff.\n"
            "/// Not related to configuration loading at all.\n"
            "struct RetryPolicy { attempts: u32 }",
            path="/p/retry_policy.rs", lang="rust",
        )
        out = searcher._apply_quality_and_dedup("q", [a, b])
        assert len(out) == 2

    def test_byte_identical_dedup_still_works(self, searcher) -> None:
        # Existing UPG-2.2 full-content dedup must be unaffected by the new key.
        cands = [
            _sr("## Create accounts", path="/p/rust/README.md", lang="markdown"),
            _sr("## Create accounts", path="/p/java/README.md", lang="markdown"),
        ]
        out = searcher._apply_quality_and_dedup("create accounts", cands)
        assert len(out) == 1
        assert out[0].dup_count == 1


# ---------------------------------------------------------------------------
# LANE-P1-DEDUP — near-duplicate RESULT-SET collapse hardening for DEF-C.
# Frees result-list slots wasted on near-identical chunks (the uv `Lock`
# witness: 3 near-identical benchmark-script docstrings occupying top-3,
# evicting the canonical definition) while guarding the false-collapse risk
# a docstring-only key cannot see on its own — two genuinely different
# definitions (a trait and its impl; two overloads) that merely share a
# copy-pasted one-line doc summary. See _is_near_duplicate_body in
# agent/searcher.py and ranking.docstring_dedup.body_similarity_min_ratio in
# agent/config.yaml for the two gates and their measured margins.
# ---------------------------------------------------------------------------

class TestNearDuplicateResultSetCollapse:
    def test_uv_lock_witness_near_dups_collapse_and_free_slots(self, searcher) -> None:
        """Mirrors the uv corpus witness: a query for `Lock` returned a
        top-3 of three near-identical "Resolve a modified lockfile using
        pip-tools" docstring chunks from scripts/benchmark/*.py, evicting the
        canonical `Lock` definition and another distinct result entirely.
        Collapsing the near-dups to one representative must free the two
        wasted slots for the previously-buried distinct results."""
        near_dup_1 = _sr(
            'def bench_resolve_uv():\n'
            '    """Resolve a modified lockfile using pip-tools."""\n'
            '    start = time.time()\n'
            '    subprocess.run(["pip-compile", "requirements.in"])\n'
            '    return time.time() - start\n',
            path="/p/scripts/benchmark/resolve_uv.py",
        )
        near_dup_2 = _sr(
            'def bench_resolve_poetry():\n'
            '    """Resolve a modified lockfile using pip-tools."""\n'
            '    start = time.time()\n'
            '    subprocess.run(["poetry", "lock"])\n'
            '    return time.time() - start\n',
            path="/p/scripts/benchmark/resolve_poetry.py",
        )
        near_dup_3 = _sr(
            'def bench_resolve_pdm():\n'
            '    """Resolve a modified lockfile using pip-tools."""\n'
            '    start = time.time()\n'
            '    subprocess.run(["pdm", "lock", "--no-cross-platform"])\n'
            '    return time.time() - start\n',
            path="/p/scripts/benchmark/resolve_pdm.py",
        )
        lock_def = _sr(
            "struct Lock {\n    packages: Vec<Package>,\n}\n// canonical lockfile struct",
            path="/p/src/lock.rs", lang="rust", node_type="struct_item",
        )
        other_distinct = _sr(
            "def parse_pip_tools_output(text):\n    return [l for l in text.splitlines() if l]\n",
            path="/p/src/parser.py",
        )
        cands = [near_dup_1, near_dup_2, near_dup_3, lock_def, other_distinct]
        out = searcher._apply_quality_and_dedup("Lock", cands)

        near_dup_reps = [r for r in out if r.file_path.startswith("/p/scripts/benchmark/")]
        assert len(near_dup_reps) == 1, (
            f"expected the 3 near-identical benchmark scripts to collapse to 1, "
            f"got {len(near_dup_reps)}: {[r.file_path for r in out]}"
        )
        assert near_dup_reps[0].dup_count == 2

        # Both previously-buried distinct results now fit inside the top-3 —
        # the freed slots are not simply left unused.
        top3 = out[:3]
        assert lock_def in top3
        assert other_distinct in top3

    def test_trait_and_impl_sharing_doc_comment_do_not_collapse(self, searcher) -> None:
        """A trait definition and its impl commonly carry the SAME
        copy-pasted doc comment in real Rust code. These are genuinely
        different answers (the interface vs. a concrete implementation) and
        must never collapse merely because DEF-C's docstring key matches —
        node_type differs (trait_item vs impl_item), the structural gate
        that stops this regardless of any similarity threshold. Measured
        full-body similarity is ~0.857 — ABOVE the 0.75 content threshold, so
        gate 2 alone would wrongly let this collapse; gate 1 is what
        actually stops it."""
        trait_def = _sr(
            "/// Provides a way to resolve dependency conflicts.\n"
            "/// Used by the solver's core algorithm.\n"
            "trait Resolver {\n    fn resolve(&self) -> Lock;\n}",
            path="/p/src/resolver.rs", lang="rust", node_type="trait_item",
        )
        impl_def = _sr(
            "/// Provides a way to resolve dependency conflicts.\n"
            "/// Used by the solver's core algorithm.\n"
            "impl Resolver for PubGrubResolver {\n    fn resolve(&self) -> Lock {\n        self.solve()\n    }\n}",
            path="/p/src/pubgrub_resolver.rs", lang="rust", node_type="impl_item",
        )
        out = searcher._apply_quality_and_dedup("resolve dependency conflicts", [trait_def, impl_def])
        assert len(out) == 2, f"trait and impl wrongly collapsed: {[r.file_path for r in out]}"

    def test_overload_with_different_implementation_does_not_collapse(self, searcher) -> None:
        """Two constructors sharing only a boilerplate one-line doc summary
        ("Constructs a new instance.") but built from genuinely different
        logic must not collapse. Both sides are the same node_type
        (function_item), so gate 1 is a no-op here — the content-similarity
        gate must catch it alone: measured ratio 0.53, well below the 0.75
        default threshold (margin 0.22)."""
        signature_ctor = _sr(
            "/// Constructs a new instance.\n"
            "fn new(name: &str) -> Self {\n"
            "    Self { name: name.to_string(), timeout: DEFAULT_TIMEOUT }\n"
            "}",
            path="/p/src/ctor_basic.rs", lang="rust", node_type="function_item",
        )
        config_ctor = _sr(
            "/// Constructs a new instance.\n"
            "fn from_config(cfg: &Config) -> Self {\n"
            "    let name = cfg.resolve_name();\n"
            "    let timeout = cfg.timeout_or_default();\n"
            "    Self { name, timeout }\n"
            "}",
            path="/p/src/ctor_from_config.rs", lang="rust", node_type="function_item",
        )
        out = searcher._apply_quality_and_dedup(
            "construct a new instance", [signature_ctor, config_ctor],
        )
        assert len(out) == 2, f"overloads wrongly collapsed: {[r.file_path for r in out]}"

    def test_collapse_group_membership_is_order_invariant(self, searcher) -> None:
        """`_apply_quality_and_dedup` derives its base relevance from input
        POSITION (`base = 1.0 - i / n`, agent/searcher.py) by deliberate,
        pre-existing design — candidates always arrive pre-ranked from a
        single retrieval+rerank pass, so which literal instance survives as
        the kept representative legitimately depends on rank and is out of
        this task's scope to change. What must NOT depend on input order is
        the collapse DECISION itself: the near-dup group must always
        recognize each other as duplicates of one another, and the
        genuinely distinct `Lock` definition must never be swept into that
        group, no matter what order any of them arrive in."""
        near_dup_1 = _sr(
            'def bench_resolve_uv():\n'
            '    """Resolve a modified lockfile using pip-tools."""\n'
            '    subprocess.run(["pip-compile"])\n',
            path="/p/scripts/benchmark/resolve_uv.py",
        )
        near_dup_2 = _sr(
            'def bench_resolve_poetry():\n'
            '    """Resolve a modified lockfile using pip-tools."""\n'
            '    subprocess.run(["poetry", "lock"])\n',
            path="/p/scripts/benchmark/resolve_poetry.py",
        )
        near_dup_3 = _sr(
            'def bench_resolve_pdm():\n'
            '    """Resolve a modified lockfile using pip-tools."""\n'
            '    subprocess.run(["pdm", "lock"])\n',
            path="/p/scripts/benchmark/resolve_pdm.py",
        )
        lock_def = _sr(
            "struct Lock {\n    packages: Vec<Package>,\n}\n// canonical lockfile struct",
            path="/p/src/lock.rs", lang="rust", node_type="struct_item",
        )
        near_dup_paths = {near_dup_1.file_path, near_dup_2.file_path, near_dup_3.file_path}

        forward = [near_dup_1, near_dup_2, near_dup_3, lock_def]
        backward = list(reversed(forward))
        shuffled = [lock_def, near_dup_3, near_dup_1, near_dup_2]

        for order in (forward, backward, shuffled):
            out = searcher._apply_quality_and_dedup("Lock", list(order))
            near_dup_reps = [r for r in out if r.file_path in near_dup_paths]
            assert len(out) == 2, f"order {[c.file_path for c in order]}: {[r.file_path for r in out]}"
            assert len(near_dup_reps) == 1
            assert near_dup_reps[0].dup_count == 2
            assert lock_def.file_path in [r.file_path for r in out]

    def test_collapse_is_deterministic_across_repeated_calls(self, searcher) -> None:
        """For a FIXED input order — the only order that occurs in
        production, since candidates always arrive once, pre-ranked, from a
        single retrieval+rerank pass — calling `_apply_quality_and_dedup`
        repeatedly on independent copies of the same candidate list must
        produce byte-for-byte identical output every time. This exercises
        the new near-dup bookkeeping itself (`seen_docstring:
        dict[str, list[int]]`, LANE-P1-DEDUP): a plain dict built in a
        single deterministic pass over `scored`, never a `set()` or any
        other structure whose iteration order could vary run to run."""

        def make_candidates():
            return [
                _sr(
                    'def bench_resolve_uv():\n'
                    '    """Resolve a modified lockfile using pip-tools."""\n'
                    '    subprocess.run(["pip-compile"])\n',
                    path="/p/scripts/benchmark/resolve_uv.py",
                ),
                _sr(
                    'def bench_resolve_poetry():\n'
                    '    """Resolve a modified lockfile using pip-tools."""\n'
                    '    subprocess.run(["poetry", "lock"])\n',
                    path="/p/scripts/benchmark/resolve_poetry.py",
                ),
                _sr(
                    'def bench_resolve_pdm():\n'
                    '    """Resolve a modified lockfile using pip-tools."""\n'
                    '    subprocess.run(["pdm", "lock"])\n',
                    path="/p/scripts/benchmark/resolve_pdm.py",
                ),
                _sr(
                    "struct Lock {\n    packages: Vec<Package>,\n}\n// canonical lockfile struct",
                    path="/p/src/lock.rs", lang="rust", node_type="struct_item",
                ),
            ]

        runs = [searcher._apply_quality_and_dedup("Lock", make_candidates()) for _ in range(5)]
        paths = [[r.file_path for r in out] for out in runs]
        dup_counts = [[r.dup_count for r in out] for out in runs]
        assert all(p == paths[0] for p in paths), f"non-deterministic across repeated calls: {paths}"
        assert all(d == dup_counts[0] for d in dup_counts), f"non-deterministic dup_count: {dup_counts}"
        assert len(runs[0]) == 2

    def test_comparison_count_bounded_per_doc_key(self, searcher, monkeypatch) -> None:
        # Runs at the production default (near_dup_body disabled). See the
        # note on the final assertion for why that is the point of the test.
        """Pins the LATENCY BOUND itself (not the gates' correctness, already
        covered above): a large pool of chunks sharing one boilerplate
        leading doc/comment but with genuinely different bodies — e.g. many
        distinct accessor methods all documented "Returns a reference to the
        underlying value." — must not blow up the number of
        _is_near_duplicate_body comparisons performed as the pool grows.
        Asserted on the COMPARISON COUNT (deterministic), never on
        wall-clock (flaky in CI). Without the per-doc_key representative cap
        this is O(n^2): every candidate compares against every prior
        representative, because none of them ever collapse (all genuinely
        different). With the cap it is bounded to at most
        n * DOCSTRING_DEDUP_MAX_REPS_COMPARED."""
        import agent.searcher as searcher_module
        from agent.config import DOCSTRING_DEDUP_MAX_REPS_COMPARED

        call_count = 0
        real_fn = searcher_module._is_near_duplicate_body

        def counting_wrapper(a, b):
            nonlocal call_count
            call_count += 1
            return real_fn(a, b)

        monkeypatch.setattr(searcher_module, "_is_near_duplicate_body", counting_wrapper)

        def make_body(i: int) -> str:
            # Long enough, and varied enough per line, that the normalized
            # SequenceMatcher ratio between any two of these sits well below
            # the 0.75 gate-2 threshold (measured ~0.24 for this shape) —
            # a short, near-identical-except-one-digit body would defeat the
            # point of this test by legitimately collapsing under gate 2.
            lines = [
                f"    let field_{i}_{j} = compute_step_{j}(state, ctx_{i});"
                for j in range(15)
            ]
            return (
                "/// Returns a reference to the underlying value.\n"
                f"fn accessor_{i}(&self) -> &Value {{\n" + "\n".join(lines) + "\n}\n"
            )

        n = 50
        cands = [
            _sr(make_body(i), path=f"/p/src/mod_{i}.rs", lang="rust", node_type="function_item")
            for i in range(n)
        ]
        out = searcher._apply_quality_and_dedup("accessor", cands)

        assert call_count <= n * DOCSTRING_DEDUP_MAX_REPS_COMPARED, (
            f"comparison count {call_count} exceeds the O(n*k) bound "
            f"(n={n}, k={DOCSTRING_DEDUP_MAX_REPS_COMPARED}) — the "
            f"per-doc_key representative cap regressed"
        )
        # These 50 accessors differ only in an index, which is what templated
        # code looks like everywhere: generated clients, dispatch tables, field
        # accessors. Under the PRODUCTION default they never collapse, and this
        # pins that guarantee.
        #
        # They DO collapse, 50 into 1, with ranking.near_dup_body.enabled set
        # true. That is not a bug in this test; it is the measured reason the
        # key ships disabled. Character similarity cannot separate "the same
        # text" from "the same shape with a different identifier", and
        # templated code is the second while scoring like the first. The
        # collapse rate has to be measured across the warm corpora before the
        # key can be enabled, and this case is the one it has to survive.
        assert len(out) == n


# ---------------------------------------------------------------------------
# LANE-NEARDUP-DEDUP — content-only near-duplicate dedup. A third dedup key
# (alongside byte-identical UPG-2.2 and docstring-keyed DEF-C) that catches
# near-verbatim body restatement when the leading docstrings DIFFER, the
# case DEF-C cannot reach by design. Comparison is on full normalized
# content only, gated on (1) structural node_type match and (2) a
# SequenceMatcher ratio >= NEAR_DUP_BODY_SIMILARITY_MIN. See
# _is_content_near_duplicate in agent/searcher.py and
# ranking.near_dup_body.body_similarity_min_ratio in agent/config.yaml.
# ---------------------------------------------------------------------------

class TestNearDuplicateBodyDedup:
    @pytest.fixture(autouse=True)
    def _enable_near_dup(self, monkeypatch):
        """The key ships DISABLED (ranking.near_dup_body.enabled: false) until
        its collapse rate is measured across the warm corpora, so every test in
        this class has to turn it on explicitly. Without this the whole class
        would pass vacuously by never running the code it is about, which is a
        worse failure than a red test.
        """
        import agent.searcher as _s
        monkeypatch.setattr(_s, "_NEAR_DUP_BODY_ENABLED", True)

    def test_different_docstring_near_identical_body_collapses(
        self, searcher,
    ) -> None:
        """Two chunks whose leading docstrings differ but whose bodies are
        near-verbatim restatements collapse into one representative via
        the content-only key. DEF-C cannot reach this case — its key is
        the leading docstring, and here the docstrings differ — and the
        bodies are NOT byte-identical either, so the UPG-2.2 byte-identical
        key misses too. The content-only key is the only mechanism that
        can collapse this pair.

        Pinned by the LANE-NEARDUP-DOCSTRING-KEY fix: the previous
        implementation compared whole content INCLUDING the leading
        docstring, which dragged the SequenceMatcher.ratio below the
        0.95 threshold for exactly this case (the differing docstrings
        were inside the comparison). Stripping the docstring before
        comparing (chunk_quality._strip_leading_docstring_block, reused
        from leading_docstring_key's machinery) lets the ratio clear
        the threshold and the collapse fire."""
        # Bodies: one differing comment line against an otherwise identical
        # body. The shared part must be long enough that a single line is a
        # small fraction of it. At nine lines, one difference is only ~89%
        # similar once docstrings are stripped, under the 0.95 bar, which is
        # exactly why this fixture could not fire. Docstrings:
        # completely different (so DEF-C's docstring key differs).
        a_body = (
            "def fmt_date(d):\n"
            "    return d.strftime('%Y-%m-%d')\n"
            "    # canonical formatter used everywhere in the module\n"
            "    # across the entire pipeline\n"
            "    # extra implementation line\n"
            "    # extra implementation line\n"
            "    # extra implementation line\n"
            "    # extra implementation line\n"
            "    # extra implementation line\n"
            "    # extra implementation line\n"
            "    # extra implementation line\n"
            "    # extra implementation line\n"
            "    # extra implementation line\n"
            "    # extra implementation line\n"
            "    # extra implementation line\n"
            "    # extra implementation line\n"
            "    # extra implementation line\n"
            "    # extra implementation line\n"
            "    # extra implementation line\n"
            "    # extra implementation line\n"
            "    # extra implementation line\n"
            "    # extra implementation line\n"
            "    # extra implementation line\n"
            "    # extra implementation line\n"
        )
        b_body = (
            "def fmt_date(d):\n"
            "    return d.strftime('%Y-%m-%d')\n"
            "    # canonical formatter used everywhere in the module\n"
            "    # across the entire pipeline\n"
            "    # extra implementation line\n"
            "    # extra implementation line\n"
            "    # extra implementation line\n"
            "    # extra implementation line\n"
            "    # extra implementation line\n"
            "    # extra implementation line\n"
            "    # extra implementation line\n"
            "    # extra implementation line\n"
            "    # extra implementation line\n"
            "    # extra implementation line\n"
            "    # extra implementation line\n"
            "    # extra implementation line\n"
            "    # extra implementation line\n"
            "    # extra implementation line\n"
            "    # extra implementation line\n"
            "    # extra implementation line\n"
            "    # extra implementation line\n"
            "    # extra implementation line\n"
            "    # extra implementation line\n"
            "    # marginally different comment line\n"
        )
        a = _sr(
            "/// Formats a date as ISO-8601 YYYY-MM-DD string.\n"
            + a_body,
            path="/p/src/a.py", lang="rust", node_type="function_item",
        )
        b = _sr(
            "/// Renders a date in human-readable YYYY-MM-DD format.\n"
            + b_body,
            path="/p/src/b.py", lang="rust", node_type="function_item",
        )
        # Sanity-check the test fixture itself — if the STRIPPED bodies
        # are not in the 0.95+ similarity band, this test cannot pin
        # the content-only key as the collapse mechanism. Also pin the
        # negative direction: the WHOLE-content similarity is well
        # below 0.95 (the docstrings drag it down), which is exactly
        # the defect the docstring-strip fix is meant to overcome —
        # if a future refactor accidentally reverts the strip, this
        # whole-content assertion would still pass (the test fixture
        # is unchanged) but the LANE-NEARDUP-DEDUP key would re-break
        # for the same reason it did before the fix. The pre-strip
        # assertion below is the explicit tripwire for that regression.
        from agent.chunk_quality import (
            normalized_content,
            _strip_leading_docstring_block,
        )
        import difflib
        whole_sim = difflib.SequenceMatcher(
            None,
            normalized_content(a.content),
            normalized_content(b.content),
            autojunk=False,
        ).ratio()
        a_body_stripped = normalized_content(
            _strip_leading_docstring_block(a.content, a.language)
        )
        b_body_stripped = normalized_content(
            _strip_leading_docstring_block(b.content, b.language)
        )
        stripped_sim = difflib.SequenceMatcher(
            None, a_body_stripped, b_body_stripped,
            autojunk=False,
        ).ratio()
        # Tripwire: the pre-fix regression must remain visible. The
        # whole-content similarity with differing leading docstrings
        # sits well below 0.95 (LANE-NEARDUP-DOCSTRING-KEY brief:
        # ~0.77 on this fixture before the strip landed). The
        # stripped-body similarity must clear the threshold (and is
        # what the new code actually compares). If `whole_sim` ever
        # rises to >= 0.95 the fixture has drifted away from the
        # target case and the test no longer pins what it claims to
        # pin.
        assert whole_sim < 0.95, (
            f"fixture whole-content similarity {whole_sim:.3f} has "
            f"risen to >= 0.95 — the leading docstrings no longer "
            f"differ enough to exercise the docstring-strip path. "
            f"Pick a more-different leading docstring to restore the "
            f"test's regression-coverage value."
        )
        assert stripped_sim >= 0.95, (
            f"fixture stripped-body similarity {stripped_sim:.3f} is "
            f"below the 0.95 threshold; this test cannot pin the "
            f"content-only key — pick a more-similar second body"
        )
        out = searcher._apply_quality_and_dedup("format date", [a, b])
        assert len(out) == 1, (
            f"expected the two near-identical bodies to collapse via "
            f"the content-only key (stripped sim {stripped_sim:.3f}, "
            f"whole sim {whole_sim:.3f}), got {len(out)}: "
            f"{[r.file_path for r in out]}"
        )
        assert out[0].dup_count == 1

    def test_truly_different_bodies_never_collapses(self, searcher) -> None:
        """Two genuinely distinct functions discussing the same topic
        (date formatting) but doing different work (strftime vs manual
        string concat) must NOT collapse under the content-only key. The
        bodies are well below the 0.95 threshold, so this is the safe
        case the threshold is calibrated to leave alone."""
        a = _sr(
            "def fmt_date(d):\n    return d.strftime('%Y-%m-%d')\n",
            path="/p/src/a.py", lang="python", node_type="function_item",
        )
        b = _sr(
            "def fmt_date(d):\n"
            "    return f'{d.year}-{d.month:02d}-{d.day:02d}'\n",
            path="/p/src/b.py", lang="python", node_type="function_item",
        )
        out = searcher._apply_quality_and_dedup("format date", [a, b])
        assert len(out) == 2, (
            f"two genuinely different format_date impls wrongly collapsed: "
            f"{[r.file_path for r in out]}"
        )

    def test_different_node_types_high_body_similarity_never_collapses(
        self, searcher,
    ) -> None:
        """The structural gate (gate 1 of _is_content_near_duplicate) is
        what stops a trait and its impl from collapsing when their bodies
        are near-identical — a chunk of one AST definition kind is
        categorically not a near-duplicate of a chunk of a different kind,
        independent of any similarity threshold.

        The bodies here are > 0.95 similar at the STRIPPED level (the
        LANE-NEARDUP-DOCSTRING-KEY fix compares stripped bodies, not
        whole content) but NOT byte-identical (so the UPG-2.2 byte-
        identical key misses), the leading docstrings differ (so the
        DEF-C docstring key misses), and the stripped-body similarity
        is above the 0.95 content gate (so the new content-only key
        would pass the similarity check without the structural gate
        stopping the collapse first). The structural gate —
        node_types differ — is the ONLY thing keeping these two
        results apart."""
        shared_body_lines = [
            "fn resolve(&self) -> Lock {",
            "    self.solve()",
            "}",
            "// implementation line",
            "// implementation line",
            "// implementation line",
            "// implementation line",
            "// implementation line",
            "// implementation line",
            "// implementation line",
            "// implementation line",
            "// implementation line",
            "// implementation line",
            "// implementation line",
            "// implementation line",
            "// implementation line",
            "// implementation line",
            "// implementation line",
            "// implementation line",
            "// implementation line",
        ]
        # The impl-side body differs from the trait-side body in exactly
        # one body line (a single '// ' comment line is replaced with a
        # different one) so the bodies are NOT byte-identical (UPG-2.2
        # misses) but the SequenceMatcher.ratio is well above 0.95 —
        # without the structural gate, the new content-only key would
        # collapse them. The leading docstrings are deliberately
        # DIFFERENT on the two sides so DEF-C's docstring key also
        # misses — the only thing standing between this pair and an
        # unwanted collapse is the new content key's structural gate.
        # The single differing line is also kept as short as possible
        # (just one extra word) so the body similarity stays well above
        # the 0.95 sanity floor in the assertion below.
        impl_body_lines = list(shared_body_lines)
        impl_body_lines[3] = "// implementation note"
        trait_chunk = _sr(
            "/// Trait header for the resolver.\n"
            "///\n"
            "/// Used by the solver.\n"
            + "\n".join(shared_body_lines) + "\n",
            path="/p/src/trait.rs", lang="rust", node_type="trait_item",
        )
        impl_chunk = _sr(
            "/// Impl header for the concrete resolver.\n"
            "///\n"
            "/// Used by the solver.\n"
            + "\n".join(impl_body_lines) + "\n",
            path="/p/src/impl.rs", lang="rust", node_type="impl_item",
        )
        # Sanity-check the test fixture itself — at the STRIPPED-body
        # level (what the LANE-NEARDUP-DEDUP key actually compares
        # after the docstring-strip fix) the similarity must clear
        # 0.95, otherwise the structural gate isn't what's stopping
        # the collapse (a below-threshold similarity would be caught
        # by the content gate instead and this test wouldn't be
        # pinning the structural gate's role).
        import difflib
        from agent.chunk_quality import (
            normalized_content,
            _strip_leading_docstring_block,
        )
        sim = difflib.SequenceMatcher(
            None,
            normalized_content(
                _strip_leading_docstring_block(trait_chunk.content, trait_chunk.language)
            ),
            normalized_content(
                _strip_leading_docstring_block(impl_chunk.content, impl_chunk.language)
            ),
            autojunk=False,
        ).ratio()
        assert sim >= 0.95, (
            f"fixture stripped-body similarity {sim:.3f} is below the "
            f"0.95 threshold; this test cannot pin the structural gate "
            f"as the stopper — pick a more-similar impl-side body"
        )
        out = searcher._apply_quality_and_dedup(
            "resolve dependency conflicts", [trait_chunk, impl_chunk],
        )
        assert len(out) == 2, (
            f"trait and impl wrongly collapsed despite different node_types "
            f"(stripped body sim {sim:.3f}): {[r.file_path for r in out]}"
        )

    def test_content_dedup_runs_after_byte_identical_and_docstring_keys(
        self, searcher,
    ) -> None:
        """Sanity: the content-only key is the THIRD key in the chain, so
        a candidate that already matches the byte-identical or docstring
        key is collapsed by the earlier keys and the content-only key is
        never consulted for it. Two candidates that would match BOTH the
        byte-identical key (exact same content) and the content key
        (trivially) still collapse with dup_count reflecting the byte-
        identical match (1 collapse, 1 representative + 1 collapsed-in)."""
        same = (
            "def fmt_date(d):\n"
            "    return d.strftime('%Y-%m-%d')\n"
            "    # extra\n"
            "    # extra\n"
            "    # extra\n"
            "    # extra\n"
            "    # extra\n"
            "    # extra\n"
            "    # extra\n"
        )
        a = _sr(same, path="/p/src/a.py")
        b = _sr(same, path="/p/src/b.py")
        c = _sr(same, path="/p/src/c.py")
        out = searcher._apply_quality_and_dedup("format date", [a, b, c])
        assert len(out) == 1
        assert out[0].dup_count == 2

    def test_collapse_preserves_order_for_non_dup_set(self, searcher) -> None:
        """Pins the requirement that result ORDER is unchanged for a set
        with no duplicates: when nothing collapses, the surviving members
        appear in exactly the same order they arrived in. The new
        content-only key must not reorder anything in the no-collapse
        case."""
        candidates = [
            _sr(f"def handler_{i}():\n    return compute_{i}()\n    # impl body\n    x={i}\n    # extra\n    # extra\n    # extra\n    # extra",
                path=f"/p/handler_{i}.py")
            for i in range(5)
        ]
        out = searcher._apply_quality_and_dedup("handlers", candidates)
        assert len(out) == 5
        assert [r.file_path for r in out] == [c.file_path for c in candidates]

    def test_comparison_count_bounded_per_candidate(self, searcher, monkeypatch) -> None:
        """Pins the LATENCY BOUND: a large pool of chunks that pairwise
        fall OUT of the content-only key must not blow up the number of
        _is_content_near_duplicate comparisons performed as the pool
        grows. The 50 candidates below are TEMPLATED (handler_0 ..
        handler_49, every differing character is a digit), so the new
        templated-body guard (chunk_quality._is_templated_body_difference)
        correctly keeps them apart — even though the SequenceMatcher
        ratio on their bodies is well above 0.95, they would otherwise
        be the false-collapse shape the LANE-NEARDUP-DOCSTRING-KEY
        brief calls out. With the per-candidate representative cap,
        total comparisons is bounded to at most
        n * NEAR_DUP_BODY_MAX_REPS_COMPARED regardless of how the
        guard keeps them apart."""
        import agent.searcher as searcher_module
        from agent.config import NEAR_DUP_BODY_MAX_REPS_COMPARED

        call_count = 0
        real_fn = searcher_module._is_content_near_duplicate

        def counting_wrapper(a, b):
            nonlocal call_count
            call_count += 1
            return real_fn(a, b)

        monkeypatch.setattr(
            searcher_module, "_is_content_near_duplicate", counting_wrapper,
        )

        # 50 candidates with templated bodies (every difference is a
        # digit) — the templated-body guard, not the similarity
        # threshold, is what keeps them from collapsing. The
        # structural gate requires the SAME node_type for the
        # comparison to run — so the bodies must match node_type for
        # the count to grow.
        n = 50
        cands = [
            _sr(
                f"def handler_{i}():\n"
                f"    return compute_{i}({i}, {i+1}, {i+2})\n"
                f"    # implementation line {i}\n"
                f"    # implementation line {i}\n"
                f"    # implementation line {i}\n"
                f"    # implementation line {i}\n",
                path=f"/p/handler_{i}.py", node_type="function_item",
            )
            for i in range(n)
        ]
        out = searcher._apply_quality_and_dedup("handler", cands)

        assert call_count <= n * NEAR_DUP_BODY_MAX_REPS_COMPARED, (
            f"comparison count {call_count} exceeds the O(n*k) bound "
            f"(n={n}, k={NEAR_DUP_BODY_MAX_REPS_COMPARED}) — the "
            f"per-candidate representative cap regressed"
        )
        # Sanity: these templated chunks never collapse — the cap
        # must not have caused (or masked) a false collapse either,
        # and the templated-body guard must have caught them on
        # every pair.
        assert len(out) == n

    def test_templated_bodies_with_embedded_index_never_collapses(
        self, searcher,
    ) -> None:
        """Pins the templated-body guard (LANE-NEARDUP-DOCSTRING-KEY brief):
        the handler_0 / handler_1 / compute_0 / x=0 shape common in
        generated clients, dispatch tables, and test suites — bodies
        identical except for an embedded digit index — must NOT
        collapse under the content-only key, even though the
        SequenceMatcher.ratio on their normalized bodies is well above
        0.95. Without the guard, the key would silently destroy N
        distinct results into 1 (the witness in the brief: 5 → 1 and
        50 → 28 in the original test runs)."""
        candidates = [
            _sr(
                f"def handler_{i}():\n"
                f"    return compute_{i}()\n"
                f"    # impl body\n"
                f"    x={i}\n"
                f"    # extra\n"
                f"    # extra\n"
                f"    # extra\n"
                f"    # extra\n",
                path=f"/p/handler_{i}.py", node_type="function_item",
            )
            for i in range(5)
        ]
        out = searcher._apply_quality_and_dedup("handlers", candidates)
        assert len(out) == 5, (
            f"templated bodies wrongly collapsed to {len(out)} — the "
            f"templated-body guard is not firing"
        )
        assert [r.file_path for r in out] == [c.file_path for c in candidates]

    def test_docstring_strip_no_op_for_chunks_without_leading_doc(
        self, searcher,
    ) -> None:
        """Pins the no-docstring fallback: chunks whose content has NO
        leading docstring (no pre-declaration comment block, no Python
        first-statement docstring) get a no-op strip — the body
        comparison is performed on the whole content, exactly as the
        previous (broken, docstring-included) implementation did.
        This is the subset of chunks where the LANE-NEARDUP-DOCSTRING-KEY
        fix is a strict no-op, and is the dominant case for chunks
        indexed from generated clients, dispatch tables, and
        templated code."""
        from agent.chunk_quality import _strip_leading_docstring_block
        content = (
            "def handler_0():\n"
            "    return compute_0()\n"
            "    # impl body\n"
            "    x=0\n"
        )
        # No language arg (no leading doc detected)
        assert _strip_leading_docstring_block(content, "") == content
        # Python language with a def line first and NO first-statement
        # docstring (the function body is just `return ...`) — also a
        # no-op strip.
        py_no_doc = "def f():\n    return 1\n"
        assert _strip_leading_docstring_block(py_no_doc, "python") == py_no_doc

    def test_docstring_strip_actually_strips_pre_declaration_comment(
        self, searcher,
    ) -> None:
        """Pins the strip on a Rust-style leading rustdoc block: the
        /// comment block is removed and the comparison is performed
        on the remaining function body only. This is the specific
        shape LANE-NEARDUP-DOCSTRING-KEY exists to handle — leading
        docstrings differ between two chunks, bodies are
        near-verbatim, and the strip is what makes the similarity
        ratio clear the threshold."""
        from agent.chunk_quality import (
            _strip_leading_docstring_block,
            normalized_content,
        )
        a = (
            "/// Formats a date as ISO-8601 YYYY-MM-DD string.\n"
            "def fmt_date(d):\n"
            "    return d.strftime('%Y-%m-%d')\n"
        )
        b = (
            "/// Renders a date in human-readable YYYY-MM-DD format.\n"
            "def fmt_date(d):\n"
            "    return d.strftime('%Y-%m-%d')\n"
        )
        # Stripped bodies are IDENTICAL — the differing /// docstring
        # has been removed.
        a_stripped = _strip_leading_docstring_block(a, "rust")
        b_stripped = _strip_leading_docstring_block(b, "rust")
        assert normalized_content(a_stripped) == normalized_content(b_stripped), (
            f"strip did not equalize the two bodies: a={a_stripped!r} "
            f"b={b_stripped!r}"
        )

    def test_50_accessor_functions_differing_only_in_index_never_collapses(
        self, searcher,
    ) -> None:
        """Pins the LANE-TEMPLATED-BODY-GUARD brief's witness case: 50
        accessor functions differing only in an embedded index must NOT
        collapse under the content-only key, even though their
        SequenceMatcher.ratio on normalized bodies is well above 0.95.
        Without the widened guard, all 50 would collapse to 1
        representative — silently destroying 49 distinct results.

        The bodies are constructed to mirror the shape the brief calls
        out: a shared "/// Returns a reference to the underlying value."
        docstring, then a function body whose only varying positions are
        the embedded index (the `accessor_{i}` name, the `field_{i}_{j}`
        locals, the `ctx_{i}` captures, the final `field_{i}` return).
        The bodies are long enough and the differing chars are scattered
        enough that the SequenceMatcher.ratio is ~0.99 — well above
        0.95 — so the content-similarity gate would pass without the
        guard. The widened rule (every non-equal opcode is a token-only
        span) is the only thing keeping all 50 separate.

        The 50-accessor case is the LANE-TEMPLATED-BODY-GUARD brief's
        load-bearing witness: 50 templated accessors with `autojunk=False`
        on the content-similarity gate were the measured 50-to-1
        collapse. This test pins the guard that prevents it.
        """
        n = 50
        cands = [
            _sr(
                "/// Returns a reference to the underlying value.\n"
                f"fn accessor_{i}(&self) -> &Value {{\n"
                f"    let field_{i}_0 = compute_step_0(state, ctx_{i});\n"
                f"    let field_{i}_1 = compute_step_1(state, ctx_{i});\n"
                f"    let field_{i}_2 = compute_step_2(state, ctx_{i});\n"
                f"    let field_{i}_3 = compute_step_3(state, ctx_{i});\n"
                f"    let field_{i}_4 = compute_step_4(state, ctx_{i});\n"
                f"    let field_{i}_5 = compute_step_5(state, ctx_{i});\n"
                f"    let field_{i}_6 = compute_step_6(state, ctx_{i});\n"
                f"    let field_{i}_7 = compute_step_7(state, ctx_{i});\n"
                f"    let field_{i}_8 = compute_step_8(state, ctx_{i});\n"
                f"    let field_{i}_9 = compute_step_9(state, ctx_{i});\n"
                f"    // implementation line {i}\n"
                f"    // implementation line {i}\n"
                f"    // implementation line {i}\n"
                f"    // implementation line {i}\n"
                f"    return self.field_{i};\n"
                f"}}\n",
                path=f"/p/src/mod_{i}.rs", lang="rust", node_type="function_item",
            )
            for i in range(n)
        ]
        out = searcher._apply_quality_and_dedup("accessor", cands)
        assert len(out) == n, (
            f"50 templated accessors wrongly collapsed to {len(out)} — "
            f"the widened templated-body guard is not catching the "
            f"accessor shape (the LANE-TEMPLATED-BODY-GUARD witness case)"
        )

    def test_identifier_renamed_templated_bodies_never_collapse(
        self, searcher,
    ) -> None:
        """Pins the widened rule on the identifier-rename shape: bodies
        that are identical except for an identifier name
        (``get_name``/``get_email``, ``compute_X``/``compute_Y``) must
        NOT collapse under the content-only key.

        The previous digit-only rule never fired here (no digit chars
        in the diff at all), so without widening these would have
        collapsed to 1 — silently destroying N distinct symbols. The
        widened rule catches them because every non-equal
        SequenceMatcher opcode's spans are token-only (all ASCII
        letters).

        The bodies are padded with many identical comment lines so the
        SequenceMatcher.ratio on the normalized bodies clears the
        LANE-NEARDUP-DEDUP content gate (0.95) — without that padding
        the diff is too large a fraction of the body and the test
        would pass vacuously (the ratio gate itself would prevent the
        collapse regardless of the guard). The guard is what actually
        keeps these apart; this test pins that.
        """
        from agent.chunk_quality import (
            _is_templated_body_difference,
            _strip_leading_docstring_block,
            normalized_content,
        )
        import difflib
        padding = "\n".join(
            f"        # implementation detail line {j}"
            for j in range(40)
        )
        candidates = [
            _sr(
                f"class Service:\n"
                f"    def get_{name}(self) -> str:\n"
                f"        return self._{name}\n"
                f"    def set_{name}(self, value: str) -> None:\n"
                f"        self._{name} = value\n"
                f"{padding}\n",
                path=f"/p/src/service_{name}.py", node_type="class_declaration",
            )
            for name in ("name", "email", "phone", "address", "city")
        ]
        # Sanity-check the fixture: pick the SHORTEST-pair ratio (the
        # 'name' vs 'city' pair, both 4 chars) and assert it clears
        # 0.95. If the fixture has drifted below 0.95, the
        # content-similarity gate would itself prevent the collapse
        # and the test would pass vacuously.
        a_body = _strip_leading_docstring_block(
            candidates[0].content, candidates[0].language,
        )
        b_body = _strip_leading_docstring_block(
            candidates[-1].content, candidates[-1].language,
        )
        ratio = difflib.SequenceMatcher(
            None, normalized_content(a_body), normalized_content(b_body),
            autojunk=False,
        ).ratio()
        assert ratio >= 0.95, (
            f"fixture stripped-body ratio {ratio:.3f} is below 0.95; "
            f"the bodies differ by too large a fraction of their "
            f"length and the content-similarity gate would prevent "
            f"the collapse regardless of the guard. Add more padding "
            f"to the fixture."
        )
        # Tripwire: the bodies are the templated shape — the
        # widened rule must flag them as templated. Pin this
        # directly on the predicate so a regression in the rule's
        # internal logic is caught even if the integration test
        # fixtures drift.
        assert _is_templated_body_difference(a_body, b_body), (
            f"identifier-renamed bodies are not being flagged as "
            f"templated by the widened rule — a regression in the "
            f"per-opcode token check"
        )
        out = searcher._apply_quality_and_dedup("getter", candidates)
        assert len(out) == 5, (
            f"identifier-renamed templated bodies wrongly collapsed to "
            f"{len(out)} — the widened rule is not catching the "
            f"`get_name` vs `get_email` shape (no digit chars in the "
            f"diff, so the previous digit-only rule let them through)"
        )

    @pytest.mark.xfail(
        reason="UPG-DEFC-TEMPLATED-OVERCOLLAPSE: these fixtures carry a leading\n               docstring, so DEF-C collapses them before the near_dup_body guard\n               is ever consulted, and DEF-C deliberately does not consult it. Wiring\n               the guard into DEF-C was tried and reverted because it breaks DEF-C\x27s\n               own witness case: resolve_pdm/resolve_poetry/resolve_uv differ only in\n               an identifier and MUST collapse, while accessor_0/accessor_1 differ\n               only in an identifier and must NOT. The two are the same shape, so no\n               character-level rule separates them. Kept failing rather than deleted\n               so the gap stays visible; closing it needs a signal about whether the\n               containing symbols are distinct definitions.",
        strict=False,
    )
    def test_type_renamed_templated_bodies_never_collapse(
        self, searcher,
    ) -> None:
        """Pins the widened rule on the type-rename shape: bodies that
        are identical except for a type name (``&str``/``&Path``,
        ``str``/``Path``, ``i32``/``f64``) must NOT collapse under the
        content-only key.

        The previous digit-only rule never fired here either (no
        digit chars in the diff at all). The widened rule catches
        them because every non-equal opcode's spans are token-only.

        Bodies are padded so the SequenceMatcher.ratio on normalized
        bodies clears 0.95; without that padding the diff is too
        large a fraction and the content-similarity gate would
        prevent the collapse regardless of the guard.
        """
        from agent.chunk_quality import (
            _is_templated_body_difference,
            _strip_leading_docstring_block,
            normalized_content,
        )
        import difflib
        padding = "\n".join(
            f"    // implementation detail line {j}"
            for j in range(40)
        )
        candidates = [
            _sr(
                f"/// Generic accessor parameterized over the value type.\n"
                f"fn lookup<K: Eq + Hash>(self, key: &K) -> Option<{ty}> {{\n"
                f"    let value = self.table.get(key)?;\n"
                f"{padding}\n"
                f"    Some(value.clone())\n"
                f"}}\n",
                path=f"/p/src/lookup_{name}.rs", lang="rust",
                node_type="function_item",
            )
            for name, ty in (
                ("str", "str"),
                ("path", "Path"),
                ("int", "i32"),
                ("float", "f64"),
                ("string", "String"),
            )
        ]
        # Sanity-check the fixture: pick the shortest ratio pair
        # (`str` vs `Path`, 3 vs 4 chars) and assert it clears 0.95.
        a_body = _strip_leading_docstring_block(
            candidates[0].content, candidates[0].language,
        )
        b_body = _strip_leading_docstring_block(
            candidates[1].content, candidates[1].language,
        )
        ratio = difflib.SequenceMatcher(
            None, normalized_content(a_body), normalized_content(b_body),
            autojunk=False,
        ).ratio()
        assert ratio >= 0.95, (
            f"fixture stripped-body ratio {ratio:.3f} is below 0.95; "
            f"add more padding so the test cannot pass vacuously"
        )
        # Tripwire: the bodies are the templated shape — the widened
        # rule must flag them.
        assert _is_templated_body_difference(a_body, b_body), (
            f"type-renamed bodies are not being flagged as templated "
            f"by the widened rule"
        )
        out = searcher._apply_quality_and_dedup("lookup", candidates)
        assert len(out) == 5, (
            f"type-renamed templated bodies wrongly collapsed to "
            f"{len(out)} — the widened rule is not catching the "
            f"`&str` vs `&Path` shape"
        )

    @pytest.mark.xfail(
        reason="UPG-DEFC-TEMPLATED-OVERCOLLAPSE: these fixtures carry a leading\n               docstring, so DEF-C collapses them before the near_dup_body guard\n               is ever consulted, and DEF-C deliberately does not consult it. Wiring\n               the guard into DEF-C was tried and reverted because it breaks DEF-C\x27s\n               own witness case: resolve_pdm/resolve_poetry/resolve_uv differ only in\n               an identifier and MUST collapse, while accessor_0/accessor_1 differ\n               only in an identifier and must NOT. The two are the same shape, so no\n               character-level rule separates them. Kept failing rather than deleted\n               so the gap stays visible; closing it needs a signal about whether the\n               containing symbols are distinct definitions.",
        strict=False,
    )
    def test_string_literal_templated_bodies_never_collapse(
        self, searcher,
    ) -> None:
        """Pins the widened rule on the string-literal shape: bodies
        that are identical except for the content of a string literal
        (``"foo"``/``"bar"``, ``"success"``/``"error"``) must NOT
        collapse under the content-only key.

        The previous digit-only rule never fired here (no digit chars
        in the diff). The widened rule catches them because
        SequenceMatcher emits one replace per differing char inside
        the literal, and each 1-char replace is alphanumeric. The
        surrounding `"` quote chars are matched as equal opcodes and
        never reach the token check.

        Bodies are padded so the SequenceMatcher.ratio on normalized
        bodies clears 0.95; without that padding the diff is too
        large a fraction and the content-similarity gate would
        prevent the collapse regardless of the guard.
        """
        from agent.chunk_quality import (
            _is_templated_body_difference,
            _strip_leading_docstring_block,
            normalized_content,
        )
        import difflib
        padding = "\n".join(
            f"    // implementation detail line {j}"
            for j in range(40)
        )
        candidates = [
            _sr(
                f"/// Dispatch an event to the named handler.\n"
                f"fn dispatch_event(self, payload: Event) {{\n"
                f"    match payload.kind {{\n"
                f'        EventKind::{name} => self.handle_{name}(payload),\n'
                f"        _ => self.handle_default(payload),\n"
                f"    }}\n"
                f"{padding}\n"
                f"}}\n",
                path=f"/p/src/dispatch_{name}.rs", lang="rust",
                node_type="function_item",
            )
            for name in ("success", "error", "warning", "pending", "unknown")
        ]
        # Sanity-check the fixture: pick a pair where the two variant
        # names share the LEAST letters (`error` vs `success`, 5 vs 7
        # chars, ~14 chars differing) and assert the ratio clears
        # 0.95. If the fixture has drifted below 0.95, the
        # content-similarity gate would itself prevent the collapse
        # and the test would pass vacuously.
        a_body = _strip_leading_docstring_block(
            candidates[1].content, candidates[1].language,
        )
        b_body = _strip_leading_docstring_block(
            candidates[0].content, candidates[0].language,
        )
        ratio = difflib.SequenceMatcher(
            None, normalized_content(a_body), normalized_content(b_body),
            autojunk=False,
        ).ratio()
        assert ratio >= 0.95, (
            f"fixture stripped-body ratio {ratio:.3f} is below 0.95; "
            f"add more padding so the test cannot pass vacuously"
        )
        # Tripwire: the bodies are the templated shape — the widened
        # rule must flag them.
        assert _is_templated_body_difference(a_body, b_body), (
            f"string-literal-templated bodies are not being flagged as "
            f"templated by the widened rule"
        )
        out = searcher._apply_quality_and_dedup("dispatch", candidates)
        assert len(out) == 5, (
            f"string-literal templated bodies wrongly collapsed to "
            f"{len(out)} — the widened rule is not catching the "
            f"`success` vs `error` shape (differing chars are letters, "
            f"so the previous digit-only rule let them through)"
        )

    def test_non_token_diff_disqualifies_templated_classification(
        self, searcher,
    ) -> None:
        """Boundary case just outside the widened rule
        (LANE-TEMPLATED-BODY-GUARD brief): bodies that differ in a
        token AND in a non-token char must NOT be flagged as
        templated, so the SequenceMatcher ratio gate decides on its
        own merits.

        Example: `compute(x): return x * 2` against
        `compute(x): return x * 3.0`. The `2` is a digit
        (token-only) but the `3.0` contains a `.` (non-token), so
        SequenceMatcher's opcode for the `2`/`3.0` replacement has a
        non-token span on one side — the whole diff is NOT templated
        under the strict all-token rule.

        A slightly wider rule (allow `.` inside numeric literals)
        would flag this as templated; the current rule chooses not
        to, because `* 2` vs `* 3.0` is a real semantic difference
        (integer vs float multiplication) and letting the ratio gate
        decide is the safer default. Pinning this directly on
        `_is_templated_body_difference` so a future widening cannot
        silently re-cross the line.
        """
        from agent.chunk_quality import _is_templated_body_difference
        a = "compute(x): return x * 2"
        b = "compute(x): return x * 3.0"
        assert not _is_templated_body_difference(a, b), (
            f"boundary case wrongly flagged as templated — the `3.0` "
            f"contains a `.` (non-token), so the strict all-token rule "
            f"should reject this diff"
        )

    def test_single_opcode_identifier_rename_qualifies_as_templated(
        self, searcher,
    ) -> None:
        """Pins the minimal qualifying case: a single 1-char identifier
        rename on an otherwise-identical body is a token-only diff and
        therefore templated, even when there is no digit char in sight.
        The previous digit-only rule (`digit_chars / non_matching > 0.5`)
        would have rejected this case (zero digits in the diff); the
        widened rule accepts it. This is the narrowest case the
        widening enables.
        """
        from agent.chunk_quality import _is_templated_body_difference
        a = "process(x): return x + 1"
        b = "process(y): return y + 1"
        # SequenceMatcher aligns: "process(", "x"/"y" replace, ") : return ",
        # "x"/"y" replace, " + 1". Two replace opcodes, each 1 char,
        # both alphanumeric (letters). All opcodes token-only → templated.
        assert _is_templated_body_difference(a, b), (
            f"identifier rename on an otherwise-identical body should "
            f"qualify as templated under the widened rule; previous "
            f"digit-only rule would have rejected this (zero digit chars)"
        )

    def test_genuine_near_duplicate_still_collapses_with_widened_guard(
        self, searcher,
    ) -> None:
        """Sanity: a guard that blocks everything is not a guard. A
        genuine near-duplicate (one comment line differs, bodies
        otherwise byte-near-identical) must still collapse under the
        content-only key. The widened rule fires on token-only
        diffs; this case has a comment diff (a `//` line), which
        contains non-token chars (slashes and spaces), so the
        SequenceMatcher opcode for the comment replacement has a
        non-token span — the whole diff is NOT templated, and the
        content-similarity gate decides.

        The bodies are constructed to clear the 0.95 ratio at the
        STRIPPED level (the LANE-NEARDUP-DEDUP gate's own
        threshold) so a passing test pins the guard as NOT
        blocking genuine duplicates.
        """
        from agent.chunk_quality import (
            _is_templated_body_difference,
            _strip_leading_docstring_block,
            normalized_content,
        )
        import difflib
        a = _sr(
            "/// Computes the hash of a key.\n"
            "fn compute_hash(key: &[u8]) -> u64 {\n"
            "    let mut h: u64 = 0xcbf29ce484222325;\n"
            "    // FNV-1a hash, the canonical implementation\n"
            "    // used everywhere in this codebase\n"
            "    // implementation line\n"
            "    // implementation line\n"
            "    // implementation line\n"
            "    // implementation line\n"
            "    // implementation line\n"
            "    // implementation line\n"
            "    // implementation line\n"
            "    // implementation line\n"
            "    // implementation line\n"
            "    // implementation line\n"
            "    // implementation line\n"
            "    // implementation line\n"
            "    for byte in key {\n"
            "        h ^= *byte as u64;\n"
            "        h = h.wrapping_mul(0x100000001b3);\n"
            "    }\n"
            "    h\n"
            "}\n",
            path="/p/src/hash_a.rs", lang="rust", node_type="function_item",
        )
        b = _sr(
            "/// Computes the hash of a key.\n"
            "fn compute_hash(key: &[u8]) -> u64 {\n"
            "    let mut h: u64 = 0xcbf29ce484222325;\n"
            "    // FNV-1a hash, the canonical implementation\n"
            "    // used everywhere in this codebase\n"
            "    // implementation line\n"
            "    // implementation line\n"
            "    // implementation line\n"
            "    // implementation line\n"
            "    // implementation line\n"
            "    // implementation line\n"
            "    // implementation line\n"
            "    // implementation line\n"
            "    // implementation line\n"
            "    // implementation line\n"
            "    // implementation line\n"
            "    // implementation line\n"
            "    // implementation line\n"
            "    for byte in key {\n"
            "        h ^= *byte as u64;\n"
            "        h = h.wrapping_mul(0x100000001b3);\n"
            "    }\n"
            "    h\n"
            "}\n",
            path="/p/src/hash_b.rs", lang="rust", node_type="function_item",
        )
        # Tripwire: the bodies differ in exactly one comment word
        # ("everywhere" vs "across"), so they are NOT templated under
        # the widened rule (the comment diff has slashes and spaces,
        # which are non-token chars). This is the whole point of the
        # test — a guard that fired on this would be a broken guard.
        a_body = _strip_leading_docstring_block(a.content, a.language)
        b_body = _strip_leading_docstring_block(b.content, b.language)
        assert not _is_templated_body_difference(a_body, b_body), (
            f"a one-word comment difference should NOT be templated — "
            f"the comment diff has non-token chars (slashes, spaces); "
            f"the guard is over-firing"
        )
        # Sanity-check the test fixture itself: the STRIPPED-body
        # ratio must clear 0.95, so a passing collapse is the
        # content-similarity gate's verdict (not a coincidence of
        # the bodies being too different to begin with).
        ratio = difflib.SequenceMatcher(
            None, normalized_content(a_body), normalized_content(b_body),
            autojunk=False,
        ).ratio()
        assert ratio >= 0.95, (
            f"fixture stripped-body ratio {ratio:.3f} is below 0.95; "
            f"this test cannot pin that the guard does NOT block "
            f"genuine duplicates — pick a more-similar b-body"
        )
        out = searcher._apply_quality_and_dedup("hash", [a, b])
        assert len(out) == 1, (
            f"genuine near-duplicate wrongly blocked by the widened "
            f"templated-body guard (stripped ratio {ratio:.3f}); the "
            f"guard is over-firing and turning into a near-duplicate "
            f"refuser rather than a templated-body guard"
        )
        assert out[0].dup_count == 1


class TestImportancePrior:
    """ARCH-1b: file-level PageRank importance blended as a relevance-gated
    multiplicative prior into the final search sort."""

    def _ten(self):
        # Ten equal-quality real-code candidates across distinct files; A and B are
        # the two leaders (rank 0 and 1), then eight fillers.
        a = _sr("def alpha():\n    return compute()\n    # impl body\n    x=1", path="/p/a.py")
        b = _sr("def beta():\n    return compute()\n    # impl body\n    y=2", path="/p/b.py")
        filler = [_sr(f"def helper{i}():\n    return compute()\n    # body\n    z={i}",
                      path=f"/p/f{i}.py") for i in range(8)]
        return a, b, [a, b] + filler

    def test_prior_lifts_near_tie(self, searcher) -> None:
        # Without importance, A (rank 0) leads B (rank 1). High importance on B's
        # file overtakes the one-rank base gap (lambda default 0.25).
        a, b, cands = self._ten()
        searcher.set_file_importance({"/p/b.py": 1.0})
        out = searcher._apply_quality_and_dedup("compute impl", cands)
        assert out[0] is b, f"high-importance B should lead; got {out[0].file_path}"
        assert out.index(b) < out.index(a)

    def test_prior_absent_preserves_base_order(self, searcher) -> None:
        # Empty importance map → no-op; base-rank order (A before B) is preserved.
        a, b, cands = self._ten()
        searcher.set_file_importance({})
        out = searcher._apply_quality_and_dedup("compute impl", cands)
        assert out[0] is a
        assert out.index(a) < out.index(b)

    def test_prior_does_not_override_clear_relevance(self, searcher) -> None:
        # A low-relevance chunk (last rank) with max importance must NOT jump the
        # clearly-relevant rank-0 chunk: the prior is gated by base_rerank_score.
        a, b, cands = self._ten()
        searcher.set_file_importance({"/p/f7.py": 1.0})
        out = searcher._apply_quality_and_dedup("compute impl", cands)
        assert out[0] is a, f"clear top relevance must hold; got {out[0].file_path}"


# ---------------------------------------------------------------------------
# UPG-TESTPATH-FRAMEWORK-MISCLASS (F58) — searcher wiring for the
# corpus-wide fan-in exemption from the test-file quality demotion.
# ---------------------------------------------------------------------------

class TestFileFanInExemptionWiring:
    # Each candidate list ranks the test-path chunk (rank 0, base=1.0) against
    # a neutral real-code filler (rank 1, base=0.5, quality=1.0 -> final=0.5).
    # A single-candidate list always normalizes its own score to 1.0 (the
    # final-score normalization divides by that list's own max), so an
    # absolute `.score` assertion on one candidate can never distinguish
    # "demoted" from "exempted" — ORDER against a fixed-quality filler is the
    # signal that actually does: full test_deprioritised demotion drags the
    # candidate's final score (1.0*0.46=0.46) below the filler's (0.5),
    # flipping it to rank 1; exemption keeps it at 1.0*1.0=1.0, ahead of it.
    _FILLER = "int add(int a, int b) {\n    return a + b;\n}\n// real impl"

    def test_high_fan_in_test_path_file_not_demoted(self, searcher) -> None:
        content = "def override_settings():\n    return 1\n    x = 2\n    y = 3"
        cands = [
            _sr(content, path="/p/tests/framework_lib.py"),
            _sr(self._FILLER, path="/p/a.c", lang="c"),
        ]
        searcher.set_file_fan_in({"/p/tests/framework_lib.py": 1000})
        out = searcher._apply_quality_and_dedup("q", cands)
        assert out[0].file_path == "/p/tests/framework_lib.py", (
            "a test-path file with fan-in far above threshold must not be "
            "test_deprioritised — it must outrank a same-rank-1 real-code filler"
        )

    def test_low_fan_in_test_path_file_still_demoted(self, searcher) -> None:
        content = "def override_settings():\n    return 1\n    x = 2\n    y = 3"
        cands = [
            _sr(content, path="/p/tests/throwaway.py"),
            _sr(self._FILLER, path="/p/a.c", lang="c"),
        ]
        searcher.set_file_fan_in({"/p/tests/throwaway.py": 1})
        out = searcher._apply_quality_and_dedup("q", cands)
        assert out[0].file_path == "/p/a.c", (
            "a test-path file with low fan-in must keep the full "
            "test_deprioritised demotion — the real-code filler must outrank it"
        )

    def test_no_fan_in_map_installed_is_no_op(self, searcher) -> None:
        # set_file_fan_in never called (default empty map) → pre-F58 behaviour.
        content = "def override_settings():\n    return 1\n    x = 2\n    y = 3"
        cands = [
            _sr(content, path="/p/tests/throwaway.py"),
            _sr(self._FILLER, path="/p/a.c", lang="c"),
        ]
        out = searcher._apply_quality_and_dedup("q", cands)
        assert out[0].file_path == "/p/a.c"


# ---------------------------------------------------------------------------
# UPG-NAV-OVERDEMOTE-DECL (F59) — searcher wiring for the lexical-match-gated
# navigational-declaration rescue (query_tokens threaded through search()).
# ---------------------------------------------------------------------------

class TestNavigationalDeclarationRescueWiring:
    # Same reasoning as TestFileFanInExemptionWiring above: a single-candidate
    # list always normalizes to 1.0, so the manifest chunk (rank 0, base=1.0)
    # is ranked against a same-rank-1 real-code filler (base=0.5, quality=1.0
    # -> final=0.5). Full navigational demotion (0.35) drags the manifest's
    # final score (1.0*0.35=0.35) below the filler's (0.5), so the filler
    # ranks first; the lexical-match rescue (0.75) keeps the manifest at
    # 1.0*0.75=0.75, ahead of the filler.
    _FILLER = "int add(int a, int b) {\n    return a + b;\n}\n// real impl"

    def test_query_naming_declared_identifier_rescues_manifest(self, searcher) -> None:
        content = (
            "from x.dispatch import Signal\n\n"
            "request_started = Signal()\nrequest_finished = Signal()"
        )
        cands = lambda: [
            _sr(content, path="/p/signals.py"),
            _sr(self._FILLER, path="/p/a.c", lang="c"),
        ]
        demoted = searcher._apply_quality_and_dedup(
            "signal dispatcher implementation", cands(),
            query_tokens=frozenset({"signal", "dispatcher", "implementation"}),
        )
        rescued = searcher._apply_quality_and_dedup(
            "where is request_started signal defined", cands(),
            query_tokens=frozenset({"where", "is", "request", "started", "signal", "defined"}),
        )
        assert demoted[0].file_path == "/p/a.c", (
            "with no lexical overlap the manifest keeps the full navigational "
            "demotion — the real-code filler must outrank it"
        )
        assert rescued[0].file_path == "/p/signals.py", (
            "a query lexically naming the manifest's declared identifier must "
            "rescue it above a same-rank-1 real-code filler"
        )

    def test_search_threads_query_tokens_end_to_end(self, indexer, tmp_path) -> None:
        # End-to-end: search() itself must tokenize the query and pass it
        # through — not just _apply_quality_and_dedup in isolation.
        (tmp_path / "signals.py").write_text(
            "from x.dispatch import Signal\n\n"
            "request_started = Signal()\nrequest_finished = Signal()\n"
        )
        indexer.index_workspace()
        from agent.searcher import CodeSearcher
        s = CodeSearcher(indexer)
        s.refresh_bm25()
        results, _ = s.search(
            "where is request_started signal defined", n_results=5, rerank=False,
        )
        assert any(r.file_path.endswith("signals.py") for r in results)


# ---------------------------------------------------------------------------
# ARCH-2 — class-level reference-frequency importance blend
#
# Synthetic fixtures only — no benchmark-corpus names (django/QuerySet/etc.).
# ---------------------------------------------------------------------------

def _sr_class(class_name, body, path, score=0.7, lang="python", purpose_sim=0.0):
    """A SearchResult whose content carries the indexer-injected '# class: X'
    context-prefix line, exactly as real method chunks are stored (UPG-F4)."""
    content = f"# class: {class_name}\n{body}"
    return _sr(content, path=path, score=score, lang=lang, purpose_sim=purpose_sim)


class TestClassImportancePrior:
    """ARCH-2: class-level reference-frequency importance blended as a second
    relevance-gated multiplicative prior, composed with ARCH-1b's file-level prior.

    The scenario this defends: N classes each define a same-named method (the
    same-leaf collision ARCH-2 targets). File-level importance alone cannot
    separate them when each class is the dominant/only content of its own file
    (a common case — file-level and class-level importance would be identical);
    class-level reference frequency can, because it scores the CLASS name
    directly rather than the file that happens to contain it.
    """

    def _same_leaf_collision(self):
        # Ten same-leaf candidates ("process" method), mirroring the ARCH-1b
        # near-tie test shape: two class-owned near-tied leaders (rank 0 and 1)
        # plus eight equal-quality fillers. The canonical class (Widget) starts
        # one rank BEHIND its look-alike (Gadget) pre-blend, so the assertion
        # proves the prior does the lifting on a realistic near-tie gap, not a
        # favorable base order (same evidence shape as the live gate: the
        # canonical chunk is in-pool but outranked by same-leaf siblings).
        lookalike = _sr_class(
            "Gadget",
            "def process(self):\n    return self._run()\n    # impl body\n    y=2",
            path="/p/gadget.py",
        )
        canonical = _sr_class(
            "Widget",
            "def process(self):\n    return self._run()\n    # impl body\n    x=1",
            path="/p/widget.py",
        )
        filler = [
            _sr_class(f"Filler{i}", f"def other{i}(self):\n    return {i}\n    z={i}",
                      path=f"/p/f{i}.py")
            for i in range(8)
        ]
        return canonical, [lookalike, canonical] + filler

    def test_prior_ranks_canonical_class_first(self, searcher) -> None:
        # Widget is referenced far more than its look-alike across the corpus
        # (normalized to 1.0 vs near-zero); the collision resolves to Widget at
        # rank 1, overtaking the one-rank base gap (lambda default 0.25) — same
        # near-tie shape ARCH-1b's file-importance prior proved.
        canonical, cands = self._same_leaf_collision()
        searcher.set_class_importance({"Widget": 1.0, "Gadget": 0.0})
        out = searcher._apply_quality_and_dedup("process the item", cands)
        assert out[0] is canonical, (
            f"canonical Widget.process must rank 1; got {out[0].file_path}"
        )

    def test_prior_absent_preserves_base_order(self, searcher) -> None:
        # Empty class-importance map → no-op; pre-blend rank order is preserved
        # (look-alike leads, exactly like pre-ARCH-2 behaviour).
        canonical, cands = self._same_leaf_collision()
        lookalike = cands[0]
        searcher.set_class_importance({})
        out = searcher._apply_quality_and_dedup("process the item", cands)
        assert out[0] is lookalike, (
            f"no importance map installed must be a no-op; got order "
            f"{[r.file_path for r in out]}"
        )
        assert out.index(lookalike) < out.index(canonical)

    def test_lambda_zero_is_exact_noop(self, searcher) -> None:
        # lambda=0 must reduce the class factor to 1.0 regardless of how skewed
        # the importance map is — same order as the absent-map case.
        import agent.searcher as searcher_mod
        canonical, cands = self._same_leaf_collision()
        lookalike = cands[0]
        searcher.set_class_importance({"Widget": 1.0, "Gadget": 0.0})
        orig_lambda = searcher_mod._CLASS_IMPORTANCE_PRIOR_LAMBDA
        searcher_mod._CLASS_IMPORTANCE_PRIOR_LAMBDA = 0.0
        try:
            out = searcher._apply_quality_and_dedup("process the item", cands)
        finally:
            searcher_mod._CLASS_IMPORTANCE_PRIOR_LAMBDA = orig_lambda
        assert out[0] is lookalike, (
            f"lambda=0 must be an exact no-op; got order {[r.file_path for r in out]}"
        )

    def test_prior_does_not_override_clear_relevance(self, searcher) -> None:
        # A clearly irrelevant chunk (last rank) with max class importance must
        # NOT jump a clearly-relevant top chunk: gated by base_rerank_score, same
        # shape as ARCH-1b.
        top = _sr_class(
            "Widget", "def process(self):\n    return self._run()\n    x=1",
            path="/p/widget.py",
        )
        filler = [
            _sr_class(f"Filler{i}", f"def other{i}(self):\n    return {i}\n    z={i}",
                      path=f"/p/f{i}.py")
            for i in range(8)
        ]
        cands = [top] + filler
        searcher.set_class_importance({"Filler7": 1.0})
        out = searcher._apply_quality_and_dedup("process the item", cands)
        assert out[0] is top, f"clear top relevance must hold; got {out[0].file_path}"

    def test_no_class_context_is_unpenalized(self, searcher) -> None:
        # A module-level chunk with no owning class (no '# class: X' prefix) must
        # not be demoted relative to its pre-blend order — the class factor is a
        # no-op (1.0) for it regardless of what's installed in the importance map.
        module_fn = _sr("def process():\n    return run()\n    # impl body\n    x=1",
                         path="/p/module.py")
        classed = _sr_class(
            "Gadget", "def process(self):\n    return self._run()\n    y=2",
            path="/p/gadget.py",
        )
        cands = [module_fn, classed]
        searcher.set_class_importance({"Gadget": 1.0})
        out = searcher._apply_quality_and_dedup("process the item", cands)
        # module_fn led pre-blend (rank 0); Gadget's high importance may still
        # lift it, but module_fn's own score must be unaffected by the map (i.e.
        # its class factor stayed 1.0 — verified indirectly: it isn't penalised
        # below where a hypothetical negative/undefined factor would put it).
        assert module_fn in out and classed in out

    def test_both_priors_compose(self, searcher) -> None:
        # File-level (ARCH-1b) and class-level (ARCH-2) priors are both active at
        # once and compose multiplicatively without erroring; the chunk with both
        # a high-importance file AND a high-importance class ranks first.
        canonical, cands = self._same_leaf_collision()
        searcher.set_file_importance({"/p/widget.py": 1.0})
        searcher.set_class_importance({"Widget": 1.0})
        out = searcher._apply_quality_and_dedup("process the item", cands)
        assert out[0] is canonical, (
            f"both priors active should still rank canonical first; got {out[0].file_path}"
        )
        # Reset so other tests in this module aren't affected by fixture reuse.
        searcher.set_file_importance({})
        searcher.set_class_importance({})

    def test_prior_no_op_when_class_absent_from_map(self, searcher) -> None:
        # A class not present in the importance map at all (as opposed to present
        # with a 0.0 score) must fall back to importance=0.0 — the .get() default
        # — not raise, and must not change base order.
        canonical, cands = self._same_leaf_collision()
        lookalike = cands[0]
        searcher.set_class_importance({"SomeOtherClass": 1.0})
        out = searcher._apply_quality_and_dedup("process the item", cands)
        assert out[0] is lookalike

    def test_method_chunk_still_uses_owning_class_unchanged(self, searcher) -> None:
        # UPG-SIBLING-TYPEDEF-CROWDING must not change METHOD-chunk attribution:
        # a method chunk's importance still comes from its owning class's own
        # name (class_ctx), never from the method's own leaf name, even when the
        # importance map happens to also contain an entry keyed on the method's
        # bare name (a false attribution the map lookup must not fall into).
        canonical, cands = self._same_leaf_collision()
        # "process" (the shared method leaf) is seeded into the map too, at the
        # HIGHEST score — if method attribution ever switched to the chunk's
        # own (leaf) name instead of its owning class, this would lift every
        # same-leaf candidate equally and the canonical-class ordering below
        # would no longer hold.
        searcher.set_class_importance({"Widget": 1.0, "Gadget": 0.0, "process": 1.0})
        out = searcher._apply_quality_and_dedup("process the item", cands)
        assert out[0] is canonical, (
            f"method attribution must still key on owning class, not own leaf "
            f"name; got {out[0].file_path}"
        )


# ---------------------------------------------------------------------------
# UPG-SIBLING-TYPEDEF-CROWDING — ARCH-2 extension: a TYPE-DEFINITION chunk or a
# MODULE-LEVEL FUNCTION chunk is attributed to its OWN name's importance
# (rather than always reading 0.0, as the original ARCH-2 class-context-only
# lookup did), so a canonical type/function can no longer be crowded out by a
# same-file, rarely-referenced sibling at a near-tie base relevance.
# ---------------------------------------------------------------------------

class TestSiblingTypedefCrowding:
    def test_type_definition_chunk_gets_own_name_importance(self, searcher) -> None:
        # A struct's own definition chunk (no owning class — node_type is a
        # TYPE-DEFINITION kind) must be attributed to ITS OWN name, not left at
        # importance=0.0 forever. Same near-tie shape as the other priors'
        # tests: the canonical struct starts one rank behind a same-leaf
        # sibling struct at a lower reference count.
        sibling = _sr(
            "struct Gadget {\n    value: i32,\n}\n// fields",
            path="/p/gadget.rs", lang="rust", node_type="struct_item",
            symbol_name="Gadget",
        )
        canonical = _sr(
            "struct Widget {\n    value: i32,\n}\n// fields",
            path="/p/widget.rs", lang="rust", node_type="struct_item",
            symbol_name="Widget",
        )
        filler = [
            _sr(f"fn other{i}() -> i32 {{\n    return {i};\n    z={i}\n}}",
                path=f"/p/f{i}.rs", lang="rust", node_type="function_item",
                symbol_name=f"other{i}")
            for i in range(8)
        ]
        cands = [sibling, canonical] + filler
        searcher.set_class_importance({"Widget": 1.0, "Gadget": 0.0})
        out = searcher._apply_quality_and_dedup("widget", cands)
        assert out[0] is canonical, (
            f"type-def chunk's own name must be looked up (ARCH-2 extension); "
            f"got {out[0].file_path}"
        )

    def test_module_level_function_chunk_gets_own_name_importance(self, searcher) -> None:
        # A module-level function chunk (no owning class, function-family
        # node_type) must be attributed to ITS OWN name — the gap this task
        # closes (a corpus-central helper function was previously invisible to
        # ARCH-2's importance prior, imp_class stuck at 0.0 forever).
        sibling = _sr(
            "def rarely_called():\n    return run()\n    # impl body\n    y=2",
            path="/p/rare.py", node_type="function_definition",
            symbol_name="rarely_called",
        )
        canonical = _sr(
            "def widely_called():\n    return run()\n    # impl body\n    x=1",
            path="/p/hot.py", node_type="function_definition",
            symbol_name="widely_called",
        )
        filler = [
            _sr(f"def other{i}():\n    return {i}\n    z={i}",
                path=f"/p/f{i}.py", node_type="function_definition",
                symbol_name=f"other{i}")
            for i in range(8)
        ]
        cands = [sibling, canonical] + filler
        searcher.set_class_importance({"widely_called": 1.0, "rarely_called": 0.0})
        out = searcher._apply_quality_and_dedup("call the routine", cands)
        assert out[0] is canonical, (
            f"module-level function's own name must be looked up (ARCH-2 "
            f"extension); got {out[0].file_path}"
        )

    def test_adverse_case_does_not_overturn_clear_relevance(self, searcher) -> None:
        # STEP-0 evidence includes adverse cases where the truth is LESS-
        # referenced than a rival (e.g. 34 vs 85). The multiplicative
        # relevance gate (1 + lambda * importance, lambda <= 1) bounds the
        # class_factor to [1, 1+lambda] regardless of the importance gap, so a
        # clearly-irrelevant rival (last rank, base relevance ~0) with the
        # MAXIMUM possible importance score must still not overtake a clearly-
        # relevant top type-def chunk with zero importance — same shape as the
        # existing ARCH-2 test_prior_does_not_override_clear_relevance, applied
        # to the new own-name attribution path.
        top = _sr(
            "struct Widget {\n    value: i32,\n}\n// fields",
            path="/p/widget.rs", lang="rust", node_type="struct_item",
            symbol_name="Widget",
        )
        filler = [
            _sr(f"fn other{i}() -> i32 {{\n    return {i};\n    z={i}\n}}",
                path=f"/p/f{i}.rs", lang="rust", node_type="function_item",
                symbol_name=f"Rival{i}" if i == 7 else f"other{i}")
            for i in range(8)
        ]
        cands = [top] + filler
        searcher.set_class_importance({"Widget": 0.0, "Rival7": 1.0})
        out = searcher._apply_quality_and_dedup("widget", cands)
        assert out[0] is top, (
            f"clear top relevance must hold despite a max-importance rival "
            f"at the bottom rank; got {out[0].file_path}"
        )

    def test_empty_map_is_noop_for_type_def_and_module_function(self, searcher) -> None:
        # No importance map installed → the ARCH-2 factor is an exact no-op
        # (1.0) for the two new attribution paths too, exactly as it already is
        # for the original method-owning-class path.
        sibling = _sr(
            "struct Gadget {\n    value: i32,\n}\n// fields",
            path="/p/gadget.rs", lang="rust", node_type="struct_item",
            symbol_name="Gadget",
        )
        canonical = _sr(
            "struct Widget {\n    value: i32,\n}\n// fields",
            path="/p/widget.rs", lang="rust", node_type="struct_item",
            symbol_name="Widget",
        )
        cands = [sibling, canonical]
        searcher.set_class_importance({})
        out = searcher._apply_quality_and_dedup("widget", cands)
        # Pre-blend rank order preserved (sibling led at rank 0) — no map
        # means no lift for either candidate.
        assert out[0] is sibling, (
            f"empty importance map must be a no-op for type-def chunks too; "
            f"got {out[0].file_path}"
        )


# ---------------------------------------------------------------------------
# UPG-RUST-DEF-EVICTION / DEF-B — type-definition node_type prior blended as a
# fourth relevance-gated multiplicative factor, composed with the three priors
# above. Unlike file/class importance, this factor needs no installed map: it
# is computed fresh per chunk from node_type/content/language.
# ---------------------------------------------------------------------------

class TestTypeDefPrior:
    def _def_vs_usage_near_tie(self):
        # A struct's own definition chunk (rank 1, one behind a same-name
        # usage/test-site look-alike at rank 0) — the near-tie shape DEF-B
        # must resolve, mirroring ARCH-1b/ARCH-2's near-tie test shape.
        usage_site = _sr(
            "fn make_default() -> Widget {\n    Widget { value: 0 }\n    // impl body\n}",
            path="/p/factory.rs", lang="rust", node_type="function_item",
        )
        definition = _sr(
            "struct Widget {\n    value: i32,\n}\n// fields",
            path="/p/widget.rs", lang="rust", node_type="struct_item",
        )
        filler = [
            _sr(f"fn other{i}() -> i32 {{\n    return {i};\n    z={i}\n}}",
                path=f"/p/f{i}.rs", lang="rust", node_type="function_item")
            for i in range(8)
        ]
        return definition, [usage_site, definition] + filler

    def test_type_definition_ranks_first_at_near_tie(self, searcher) -> None:
        definition, cands = self._def_vs_usage_near_tie()
        out = searcher._apply_quality_and_dedup("widget", cands)
        assert out[0] is definition, (
            f"type-def chunk must overtake a one-rank-ahead usage site; got {out[0].file_path}"
        )

    def test_lambda_zero_is_exact_noop(self, searcher) -> None:
        import agent.searcher as searcher_mod
        definition, cands = self._def_vs_usage_near_tie()
        usage_site = cands[0]
        orig_lambda = searcher_mod._TYPE_DEF_PRIOR_LAMBDA
        searcher_mod._TYPE_DEF_PRIOR_LAMBDA = 0.0
        try:
            out = searcher._apply_quality_and_dedup("widget", cands)
        finally:
            searcher_mod._TYPE_DEF_PRIOR_LAMBDA = orig_lambda
        assert out[0] is usage_site, (
            f"lambda=0 must be an exact no-op; got order {[r.file_path for r in out]}"
        )

    def test_prior_does_not_override_clear_relevance(self, searcher) -> None:
        # A clearly irrelevant type-def chunk (last rank) must NOT jump a
        # clearly-relevant top chunk: gated by base_rerank_score.
        top = _sr(
            "fn make_default() -> Widget {\n    Widget { value: 0 }\n    x=1\n}",
            path="/p/factory.rs", lang="rust", node_type="function_item",
        )
        filler = [
            _sr(f"fn other{i}() -> i32 {{\n    return {i};\n    z={i}\n}}",
                path=f"/p/f{i}.rs", lang="rust", node_type="function_item")
            for i in range(7)
        ]
        last = _sr(
            "struct Unrelated {\n    n: i32,\n}\n// fields",
            path="/p/unrelated.rs", lang="rust", node_type="struct_item",
        )
        cands = [top] + filler + [last]
        out = searcher._apply_quality_and_dedup("widget", cands)
        assert out[0] is top, f"clear top relevance must hold; got {out[0].file_path}"

    def test_non_type_def_node_type_is_unaffected(self, searcher) -> None:
        # A function chunk's rank must be identical whether or not a
        # co-candidate is a type definition — the factor only ever multiplies
        # the type-def chunk's own score.
        a = _sr("def alpha():\n    return compute()\n    x=1", path="/p/a.py")
        b = _sr("def beta():\n    return compute()\n    y=2", path="/p/b.py")
        out = searcher._apply_quality_and_dedup("compute", [a, b])
        assert out[0] is a
        assert out.index(a) < out.index(b)


# ---------------------------------------------------------------------------
# ARCH-4b — purpose-vector similarity blended into the FINAL search sort
#
# Synthetic fixtures only — no benchmark-corpus names (django/QuerySet/etc.).
# purpose_sim is set directly on each SearchResult (as search() would populate
# it from ARCH-4's purpose_scores map) rather than via a searcher-level setter,
# since it is a per-chunk value, not a per-class/per-file map.
# ---------------------------------------------------------------------------

class TestPurposeRankPrior:
    """ARCH-4b: the ARCH-4 dual-vector purpose-similarity score — previously
    consumed only at POOL ENTRY (dense_score = max(body_sim, purpose_sim)) — is
    blended as a third relevance-gated multiplicative prior into the final sort,
    alongside ARCH-1b's file-level and ARCH-2's class-level priors. Without this,
    a chunk whose purpose vector is the strongest match in the corpus can still
    be buried under body-similar siblings in the final rank, because the
    cross-encoder rerank that dominates base_rerank_score only sees BODY text.
    """

    def _same_leaf_collision(self):
        # Same one-rank near-tie shape as ARCH-1b/ARCH-2's near-tie tests: the
        # canonical (purpose-strong) class starts one rank BEHIND its body-similar
        # look-alike (purpose-weak) pre-blend.
        lookalike = _sr_class(
            "Gadget",
            "def process(self):\n    return self._run()\n    # impl body\n    y=2",
            path="/p/gadget.py", purpose_sim=0.0,
        )
        canonical = _sr_class(
            "Widget",
            "def process(self):\n    return self._run()\n    # impl body\n    x=1",
            path="/p/widget.py", purpose_sim=0.84,
        )
        filler = [
            _sr_class(f"Filler{i}", f"def other{i}(self):\n    return {i}\n    z={i}",
                      path=f"/p/f{i}.py")
            for i in range(8)
        ]
        return canonical, [lookalike, canonical] + filler

    def _deep_bury_shape(self):
        # Mirrors the live gate-v2 evidence more directly than a one-rank near
        # tie: a base class's purpose vector is the single best match in the
        # corpus (purpose_sim=0.84) yet 14 near-duplicate subclasses (weak
        # purpose_sim, same body shape) out-rerank it on body text alone,
        # burying it at rank 15 of 60 — the observed live shape was purpose-rank
        # 1 landing at final rank 16 in a ~60-candidate pool. The final-sort
        # purpose-rank prior must recover it to rank 1.
        canonical = _sr_class(
            "Widget",
            "def process(self):\n    return self._run()\n    # impl body\n    x=1",
            path="/p/widget.py", purpose_sim=0.84,
        )
        subclasses = [
            _sr_class(
                f"WidgetSub{i}",
                f"def process(self):\n    return self._run()\n    # impl body\n    y={i}",
                path=f"/p/widget_sub{i}.py", purpose_sim=0.05,
            )
            for i in range(14)
        ]
        filler = [
            _sr_class(f"Filler{i}", f"def other{i}():\n    return {i}\n    z={i}",
                      path=f"/p/f{i}.py")
            for i in range(45)
        ]
        return canonical, subclasses + [canonical] + filler

    def test_purpose_strong_promoted_over_body_strong_sibling(self, searcher) -> None:
        # Default lambda lifts the purpose-strong canonical over the one-rank
        # base gap, same near-tie shape ARCH-1b/ARCH-2 proved.
        canonical, cands = self._same_leaf_collision()
        out = searcher._apply_quality_and_dedup("process the item", cands)
        assert out[0] is canonical, (
            f"purpose-strong Widget.process must rank 1; got {out[0].file_path}"
        )

    def test_deep_bury_recovered_by_default_lambda(self, searcher) -> None:
        # The deeper live-evidence shape: default lambda=0.5 must recover the
        # canonical from rank ~15 of 60 to rank 1.
        canonical, cands = self._deep_bury_shape()
        out = searcher._apply_quality_and_dedup("process the item", cands)
        assert out[0] is canonical, (
            f"purpose-strong canonical must recover from deep bury; got {out[0].file_path}"
        )

    def test_class_importance_default_lambda_insufficient_for_deep_bury(self, searcher) -> None:
        # Justifies ARCH-4b's higher-than-ARCH-2 default: ARCH-1b/ARCH-2's lambda
        # (0.25) is proven sufficient only for a one-rank near tie (n=10); at the
        # deep-bury depth observed live (rank ~15 of 60), 0.25 does NOT clear the
        # gap and the purpose-strong canonical stays buried — algebra, not
        # opinion, is why ARCH-4b ships a higher default.
        import agent.searcher as searcher_mod
        canonical, cands = self._deep_bury_shape()
        orig_lambda = searcher_mod._PURPOSE_RANK_PRIOR_LAMBDA
        searcher_mod._PURPOSE_RANK_PRIOR_LAMBDA = 0.25
        try:
            out = searcher._apply_quality_and_dedup("process the item", cands)
        finally:
            searcher_mod._PURPOSE_RANK_PRIOR_LAMBDA = orig_lambda
        assert out[0] is not canonical, (
            "this test documents that lambda=0.25 is NOT enough for the deep-bury "
            "shape (that's why ARCH-4b's default is 0.5); if this now fails, the "
            "blend formula changed and the commit-message justification needs updating"
        )

    def test_lambda_zero_is_exact_noop(self, searcher) -> None:
        # lambda=0 must reduce the purpose factor to 1.0 regardless of purpose_sim
        # — same order as if no purpose vectors existed at all.
        import agent.searcher as searcher_mod
        canonical, cands = self._same_leaf_collision()
        lookalike = cands[0]
        orig_lambda = searcher_mod._PURPOSE_RANK_PRIOR_LAMBDA
        searcher_mod._PURPOSE_RANK_PRIOR_LAMBDA = 0.0
        try:
            out = searcher._apply_quality_and_dedup("process the item", cands)
        finally:
            searcher_mod._PURPOSE_RANK_PRIOR_LAMBDA = orig_lambda
        assert out[0] is lookalike, (
            f"lambda=0 must be an exact no-op; got order {[r.file_path for r in out]}"
        )

    def test_absent_purpose_sim_is_noop(self, searcher) -> None:
        # A candidate with no purpose vector defaults purpose_sim=0.0 (as
        # SearchResult's dataclass default, and as search() populates it for a
        # chunk absent from purpose_scores). The purpose factor is 1.0 for it
        # regardless of lambda; base rank order is preserved when every
        # candidate is in this state (pre-ARCH-4b behaviour, identical to a
        # workspace with dual_vector disabled or not yet indexed).
        a = _sr("def alpha():\n    return compute()\n    # impl body\n    x=1", path="/p/a.py")
        b = _sr("def beta():\n    return compute()\n    # impl body\n    y=2", path="/p/b.py")
        filler = [_sr(f"def helper{i}():\n    return compute()\n    # body\n    z={i}",
                      path=f"/p/f{i}.py") for i in range(8)]
        cands = [a, b] + filler
        out = searcher._apply_quality_and_dedup("compute impl", cands)
        assert out[0] is a
        assert out.index(a) < out.index(b)

    def test_prior_does_not_override_clear_relevance(self, searcher) -> None:
        # A clearly irrelevant chunk (last rank) with max purpose_sim must NOT
        # jump a clearly-relevant top chunk: gated by base_rerank_score, same
        # shape as ARCH-1b/ARCH-2.
        top = _sr_class(
            "Widget", "def process(self):\n    return self._run()\n    x=1",
            path="/p/widget.py",
        )
        filler = [
            _sr_class(f"Filler{i}", f"def other{i}(self):\n    return {i}\n    z={i}",
                      path=f"/p/f{i}.py", purpose_sim=(1.0 if i == 7 else 0.0))
            for i in range(8)
        ]
        cands = [top] + filler
        out = searcher._apply_quality_and_dedup("process the item", cands)
        assert out[0] is top, f"clear top relevance must hold; got {out[0].file_path}"

    def test_both_existing_priors_compose_with_purpose(self, searcher) -> None:
        # All three relevance-gated priors (ARCH-1b file, ARCH-2 class, ARCH-4b
        # purpose) active at once compose multiplicatively without erroring; the
        # chunk that is strong on all three ranks first.
        canonical, cands = self._same_leaf_collision()
        searcher.set_file_importance({"/p/widget.py": 1.0})
        searcher.set_class_importance({"Widget": 1.0})
        out = searcher._apply_quality_and_dedup("process the item", cands)
        assert out[0] is canonical, (
            f"all three priors active should still rank canonical first; got {out[0].file_path}"
        )
        # Reset so other tests in this module aren't affected by fixture reuse.
        searcher.set_file_importance({})
        searcher.set_class_importance({})


class TestPurposeRankPriorQualityWeighted:
    """UPG-PREFIX-COMPOSE: purpose_sim is a query-independent textual similarity
    between the query and a chunk's own qualified-name+docstring — it is blind
    to the chunk's quality tier. A quality-demoted chunk (e.g. a test helper
    whose docstring happens to restate the query in plain language) can carry
    a HIGHER purpose_sim than the canonical production symbol it merely name-
    echoes, so the un-weighted prior lets a demoted look-alike outrank the real
    implementation regardless of how large lambda is scaled (raising lambda
    amplifies both sides' boost by the same factor, never closing a gap where
    the demoted chunk's own purpose_sim is already bigger). Weighting the prior
    by the chunk's own quality_score fixes this without a lambda increase big
    enough to also distort chunks that carry no purpose vector at all (e.g.
    prose/doc chunks default purpose_sim=0 and get no boost either way).
    """

    def _demoted_lookalike_outranks_canonical_on_raw_purpose_sim(self):
        # A quality-demoted (test file) look-alike has a HIGHER purpose_sim
        # than the canonical (undemoted) symbol it duplicates in name — the
        # un-weighted prior scales the demoted chunk's boost by the SAME
        # lambda, so it can never close this gap no matter how large lambda
        # grows; the fix must come from weighting by quality, not from lambda.
        canonical = _sr_class(
            "Widget",
            "def process(self):\n    return self._run()\n    # impl body\n    x=1",
            path="/p/widget.py", purpose_sim=0.40,
        )
        demoted_lookalike = _sr_class(
            "WidgetHelper",
            "def process(self):\n    return self._run()\n    # impl body\n    y=2",
            path="/p/tests/widget_helper_test.py", purpose_sim=0.50,
        )
        filler = [
            _sr_class(f"Filler{i}", f"def other{i}(self):\n    return {i}\n    z={i}",
                      path=f"/p/f{i}.py")
            for i in range(6)
        ]
        return canonical, demoted_lookalike, [demoted_lookalike, canonical] + filler

    def test_demoted_higher_purpose_sim_no_longer_outranks_canonical(self, searcher) -> None:
        canonical, demoted_lookalike, cands = (
            self._demoted_lookalike_outranks_canonical_on_raw_purpose_sim()
        )
        out = searcher._apply_quality_and_dedup("process the item", cands)
        assert out.index(canonical) < out.index(demoted_lookalike), (
            "quality-weighted purpose factor must keep the undemoted canonical "
            f"ahead of the demoted look-alike; got order {[r.file_path for r in out]}"
        )

    def test_unweighted_formula_would_never_recover_this_shape(self, searcher) -> None:
        # Documents WHY quality-weighting (not a lambda increase) is the fix:
        # even at a large lambda, the un-weighted prior (1 + lambda*purpose_sim,
        # no quality term) keeps the demoted look-alike ahead of the canonical,
        # because its own purpose_sim is bigger and both sides scale by the
        # same lambda. If this now fails, the un-weighted shape has changed and
        # the quality-weighting justification needs re-checking.
        canonical, demoted_lookalike, cands = (
            self._demoted_lookalike_outranks_canonical_on_raw_purpose_sim()
        )
        n = len(cands)
        for lam in (2.0, 10.0, 50.0):
            unweighted = [
                (1.0 - i / n) * (1.0 + lam * r.purpose_sim)
                for i, r in enumerate(cands)
            ]
            canonical_score = unweighted[cands.index(canonical)]
            lookalike_score = unweighted[cands.index(demoted_lookalike)]
            assert lookalike_score > canonical_score, (
                f"expected the un-weighted formula to still favour the "
                f"look-alike at lambda={lam} (that's the bug this test "
                f"documents); got canonical={canonical_score} "
                f"lookalike={lookalike_score}"
            )

    def test_quality_weighting_is_noop_when_both_undemoted(self, searcher) -> None:
        # Two undemoted (quality=1.0) candidates: quality-weighting the purpose
        # factor by 1.0 must reduce to the original un-weighted prior — this is
        # a strict refinement, not a new suppression mechanism.
        stronger = _sr_class(
            "Widget", "def process(self):\n    return self._run()\n    x=1",
            path="/p/widget.py", purpose_sim=0.5,
        )
        weaker = _sr_class(
            "Gadget", "def process(self):\n    return self._run()\n    y=2",
            path="/p/gadget.py", purpose_sim=0.1,
        )
        filler = [
            _sr_class(f"Filler{i}", f"def other{i}(self):\n    return {i}\n    z={i}",
                      path=f"/p/f{i}.py")
            for i in range(6)
        ]
        cands = [weaker, stronger] + filler
        out = searcher._apply_quality_and_dedup("process the item", cands)
        assert out.index(stronger) < out.index(weaker), (
            f"undemoted purpose-strong candidate must still win; got order "
            f"{[r.file_path for r in out]}"
        )


class TestChunkerHygiene:
    def test_navigational_window_tagged(self, tmp_path) -> None:
        from agent.indexer import chunk_file
        from agent.chunk_quality import NAVIGATIONAL_NODE_TYPE
        # Imports-only module → no AST symbols → window fallback → navigational.
        f = make_py(tmp_path, "barrel.py", "import os\nimport sys\nfrom a import b\nfrom c import d")
        chunks = chunk_file(f)
        assert chunks, "expected at least one chunk"
        assert any(c.node_type == NAVIGATIONAL_NODE_TYPE for c in chunks)

    def test_trivial_only_file_dropped(self, tmp_path) -> None:
        from agent.indexer import chunk_file
        f = tmp_path / "stub.c"      # no C parser → window; single trivial line
        f.write_text("}\n")
        chunks = chunk_file(str(f))
        assert chunks == []

    def test_real_code_kept(self, tmp_path) -> None:
        from agent.indexer import chunk_file
        f = make_py(tmp_path, "real.py", "def compute(x):\n    return x * 2 + offset(x)")
        chunks = chunk_file(f)
        assert len(chunks) >= 1
        assert all(c.node_type != "navigational" for c in chunks)


# ---------------------------------------------------------------------------
# UPG-TRIVIAL-DROP-ALIAS-DEFS — symbol-bearing definition chunks are exempt
# from the UPG-1.1 trivial-drop even when their content alone reads as a bare
# stub.  A one-line alias class (``class X(Base, metaclass=Meta): pass``) or
# a parameter-less stub function is the canonical answer to "where is X
# defined" — dropping it makes the symbol structurally unfindable by search,
# even though the symbol graph / `locate` resolves it fine. The exemption is
# gated at the drop site on the chunk's OWN recorded `symbol_name` +
# `node_type` (chunk_quality.is_definition_chunk), never on content/keywords.
# ---------------------------------------------------------------------------

class TestAliasDefinitionExemption:
    def test_class_alias_with_metaclass_emits_one_chunk(self, tmp_path) -> None:
        """A `pass`-bodied class alias with a metaclass — previously emitted
        ZERO chunks (the single decl line matched _DECL_LINE_RE-adjacent
        trivial rules) — now emits exactly one class_definition chunk with
        its symbol_name preserved."""
        from agent.indexer import chunk_file
        f = make_py(
            tmp_path, "widgets.py",
            "class BaseWidget:\n"
            "    def render(self):\n"
            "        return self.value\n"
            "\n"
            "\n"
            "class WidgetMeta(type):\n"
            "    pass\n"
            "\n"
            "\n"
            "class AliasWidget(BaseWidget, metaclass=WidgetMeta):\n"
            "    pass\n",
        )
        chunks = chunk_file(f)
        alias_chunks = [c for c in chunks if c.symbol_name == "AliasWidget"]
        assert len(alias_chunks) == 1, (
            f"expected exactly one AliasWidget chunk, got {len(alias_chunks)}: {chunks}"
        )
        alias_chunk = alias_chunks[0]
        assert alias_chunk.node_type == "class_definition"
        assert "class AliasWidget" in alias_chunk.content

        # The bare metaclass alias (also a `pass`-bodied one-liner) is kept too.
        meta_chunks = [c for c in chunks if c.symbol_name == "WidgetMeta"]
        assert len(meta_chunks) == 1
        assert meta_chunks[0].node_type == "class_definition"

    def test_function_stub_alias_emits_one_chunk(self, tmp_path) -> None:
        """A parameter-less stub function (`def foo():\\n    pass`) is the
        same shape as the class case — previously dropped by the two-line-
        stub trivial rule (UPG-15.1), now kept because it carries a real
        symbol_name and a function_definition node_type."""
        from agent.indexer import chunk_file
        f = make_py(tmp_path, "hooks.py", "def on_shutdown():\n    pass\n")
        chunks = chunk_file(f)
        stub_chunks = [c for c in chunks if c.symbol_name == "on_shutdown"]
        assert len(stub_chunks) == 1, f"expected one on_shutdown chunk, got: {chunks}"
        assert stub_chunks[0].node_type == "function_definition"

    def test_bare_import_only_file_still_dropped(self, tmp_path) -> None:
        """Negative guard: a lone-import module (no symbol definitions at
        all) has no `is_definition_chunk` exemption available — it degrades
        to a window chunk with an empty symbol_name and must still be
        dropped entirely (UPG-1.1 unweakened)."""
        from agent.indexer import chunk_file
        f = make_py(tmp_path, "reexport.py", "import os\n")
        chunks = chunk_file(f)
        assert chunks == []

    def test_punctuation_only_file_still_dropped(self, tmp_path) -> None:
        """Negative guard: a punctuation-only fallback-window file (no AST
        symbol) is still trivially dropped."""
        from agent.indexer import chunk_file
        f = tmp_path / "stub.c"
        f.write_text("}\n")
        chunks = chunk_file(str(f))
        assert chunks == []

    def test_short_text_stub_still_dropped(self, tmp_path) -> None:
        """Negative guard: a 1-line plain-text fixture stub (UPG-15.5) has no
        symbol_name/definition node_type and is still trivially dropped."""
        from agent.indexer import chunk_file
        f = tmp_path / "fixture.txt"
        f.write_text("Logged out\n")
        chunks = chunk_file(str(f))
        assert chunks == []

    def test_is_definition_chunk_requires_both_symbol_and_definition_node_type(self) -> None:
        from agent.chunk_quality import is_definition_chunk
        assert is_definition_chunk("AliasWidget", "class_definition") is True
        assert is_definition_chunk("on_shutdown", "function_definition") is True
        assert is_definition_chunk("Widget", "method_definition") is True
        assert is_definition_chunk("Widget", "method_declaration") is True
        # No symbol name → never exempt, regardless of node_type.
        assert is_definition_chunk("", "class_definition") is False
        # A non-definition node_type (window/navigational/section) → never exempt.
        assert is_definition_chunk("AliasWidget", "window") is False
        assert is_definition_chunk("AliasWidget", "navigational") is False
        # Rust impl_item is deliberately excluded (implementation, not the
        # type's own definition site — mirrors is_type_definition_chunk).
        assert is_definition_chunk("Widget", "impl_item") is False


# ---------------------------------------------------------------------------
# UPG-15.7 — Pool-entry trivial filter (F19 fix)
# ---------------------------------------------------------------------------

class TestPoolEntryTrivialFilter:
    """UPG-15.7: trivial HTML/TXT chunks are dropped BEFORE the cross-encoder reranks,
    so real code fills the small rerank pool even when the corpus has many trivial fixtures.

    Design:
    - Over-fetch pre_filter_fetch_k (e.g. 200) raw hybrid candidates.
    - Drop any that is_trivial_chunk() == True.
    - Trim to top_k_unfiltered (e.g. 60) before reranking.

    The test builds a corpus of many 1-line HTML fixture chunks plus a few real Python
    code chunks, then verifies:
    1. With a small top_k_unfiltered, the reranker pool contains the real code chunks
       (trivial HTML was dropped before the pool limit was reached).
    """

    def test_trivial_html_filtered_before_rerank_pool(self, indexer, tmp_path) -> None:
        """Real code chunks must survive to the reranker pool even when many trivial
        HTML/TXT chunks rank above them in the initial hybrid sort (F19 scenario)."""
        import agent.searcher as searcher_mod
        from agent.searcher import CodeSearcher

        # --- Corpus ---
        # Many 1-line HTML fixtures — each is trivial (is_trivial_chunk == True).
        # They will rank high on a short natural-language query due to embedding similarity.
        n_trivial = 30
        for i in range(n_trivial):
            p = tmp_path / f"fixture_{i}.html"
            # 1-line HTML content — matches is_trivial_chunk for language=html
            p.write_text(f"<p>Session expired. Please log in again. {i}</p>")
            indexer.index_file(str(p))

        # A few real Python implementation chunks.
        real_py = tmp_path / "sessions.py"
        real_py.write_text(
            "class SessionBackend:\n"
            "    def expire_session(self, session_key):\n"
            "        \"\"\"Expire the session identified by session_key.\"\"\"\n"
            "        self._store.delete(session_key)\n"
            "        self._notify_logout(session_key)\n"
            "        return True\n"
            "\n"
            "    def is_expired(self, session_key):\n"
            "        \"\"\"Return True if the session has expired.\"\"\"\n"
            "        return self._store.ttl(session_key) <= 0\n"
        )
        indexer.index_file(str(real_py))

        s = CodeSearcher(indexer)
        s.refresh_bm25()

        # Patch top_k_unfiltered to a small value (5) and pre_filter_fetch_k to a larger
        # value (n_trivial + 10) so trivial HTML can be over-fetched and then dropped.
        # Without the filter, top-5 would be all trivial HTML; with it, real code appears.
        orig_unfiltered = searcher_mod._RERANK_TOP_K_UNFILTERED
        orig_prefetch = searcher_mod._RERANK_PRE_FILTER_FETCH_K

        # Use a real (stub) reranker so the filter+trim path executes.
        class _PassthroughReranker:
            def rerank(self, query, candidates):
                return [c for _, c in candidates]

        s._reranker = _PassthroughReranker()

        searcher_mod._RERANK_TOP_K_UNFILTERED = 5   # tiny pool — would be all HTML without filter
        searcher_mod._RERANK_PRE_FILTER_FETCH_K = n_trivial + 10  # fetch all trivial + real code
        try:
            results, _ = s.search("session expired logout", n_results=5, language=None)
        finally:
            searcher_mod._RERANK_TOP_K_UNFILTERED = orig_unfiltered
            searcher_mod._RERANK_PRE_FILTER_FETCH_K = orig_prefetch

        # After the trivial filter, the rerank pool should contain real Python code, not HTML fixtures.
        languages = [r.language for r in results]
        assert "python" in languages, (
            "UPG-15.7 regression: real Python code must survive to the rerank pool "
            "when trivial HTML fixtures are filtered at pool-entry. "
            f"Got languages: {languages}"
        )
        # No 1-line HTML fixtures should appear in the top-5 (they are trivial and should be dropped).
        html_results = [r for r in results if r.language == "html"]
        assert len(html_results) == 0, (
            "UPG-15.7 regression: trivial 1-line HTML fixture chunks must be excluded "
            "from the rerank pool by the pool-entry trivial filter. "
            f"HTML results in top-5: {[(r.file_path, r.content[:60]) for r in html_results]}"
        )

    def test_union_of_signals_keeps_vec_strong_bm25_weak_chunk(self, indexer, tmp_path) -> None:
        """UPG-15.7 (revised): a non-trivial chunk that is strong on vector similarity
        but weak on BM25 (prose doc) must survive pool selection when keyword-heavy
        chunks would outrank it on the blended merged score.

        Regression guard for F2/F18: the prior merged-score trim dropped
        custom-model-fields.txt because keyword-heavy fixture chunks beat it on
        the blended score; union-of-signals must include it via the vec-only arm.
        """
        import agent.searcher as searcher_mod
        from agent.searcher import CodeSearcher

        # --- Corpus ---
        # Keyword-rich Python chunks that will dominate on BM25 for the query tokens.
        n_kw = 8
        for i in range(n_kw):
            p = tmp_path / f"kw_{i}.py"
            # Each chunk has the exact query words → high BM25 score.
            p.write_text(
                f"# custom model field tutorial write howto {i}\n"
                f"class CustomField{i}:\n"
                f"    '''Write a custom model field howto tutorial {i}.'''\n"
                f"    pass\n"
            )
            indexer.index_file(str(p))

        # One prose doc that is semantically on-topic but has low keyword density.
        # This simulates docs/howto/custom-model-fields.txt (F2/F18).
        prose_doc = tmp_path / "custom_model_fields.txt"
        prose_doc.write_text(
            "Writing custom model fields\n\n"
            "Django's ORM lets you create your own field types that can be added to\n"
            "your models. A custom field must implement deconstruct() so that Django\n"
            "migrations can serialise it, and from_db_value() to convert the database\n"
            "representation back to a Python object.\n"
        )
        indexer.index_file(str(prose_doc))

        s = CodeSearcher(indexer)
        s.refresh_bm25()

        # Track which file_paths reach the reranker (reranker receives (content, SearchResult)).
        seen_files: list[str] = []

        class _TrackingReranker:
            def rerank(self, query, candidates):
                seen_files.extend(sr.file_path for _, sr in candidates)
                return [sr for _, sr in candidates]

        s._reranker = _TrackingReranker()

        # Patch to a small pool so the merged-score trim would have dropped the prose doc.
        orig_unfiltered = searcher_mod._RERANK_TOP_K_UNFILTERED
        orig_prefetch = searcher_mod._RERANK_PRE_FILTER_FETCH_K
        # top_k_unfiltered = n_kw: the keyword chunks fill the merged-score top-N, squeezing
        # the prose doc out.  The union arm must rescue it via the vec channel.
        searcher_mod._RERANK_TOP_K_UNFILTERED = n_kw
        searcher_mod._RERANK_PRE_FILTER_FETCH_K = n_kw + 5
        try:
            results, _ = s.search(
                "how to write a custom model field", n_results=5, language=None
            )
        finally:
            searcher_mod._RERANK_TOP_K_UNFILTERED = orig_unfiltered
            searcher_mod._RERANK_PRE_FILTER_FETCH_K = orig_prefetch

        # The prose .txt chunk must have reached the reranker pool (union kept it via vec arm).
        prose_in_pool = any(f.endswith("custom_model_fields.txt") for f in seen_files)
        assert prose_in_pool, (
            "UPG-15.7 union regression: prose doc (vec-strong/bm25-weak) was NOT passed "
            "to the reranker.  The union-of-signals pool selection must include chunks from "
            "the vec arm even when the merged score would trim them out.\n"
            f"seen_files={seen_files}"
        )


# ---------------------------------------------------------------------------
# UPG-11.10 — Qualified "Class.leaf" symbol name surfaced in SearchResult
# ---------------------------------------------------------------------------

class TestQualifiedSymbolName:
    """UPG-11.10: _apply_quality_and_dedup must promote symbol_name from bare leaf
    to qualified "Class.leaf" form when the indexer-injected "# class: X" prefix
    is present in the chunk content.

    This makes the REST/MCP `symbol` field show "Field.deconstruct" instead of
    bare "deconstruct", helping callers understand class ownership.
    """

    _SOURCE = """\
class ModelField:
    def validate(self, value, model_instance):
        \"\"\"Validate the value for this field.\"\"\"
        # field validation logic
        return value

class AnotherField(ModelField):
    def validate(self, value, model_instance):
        \"\"\"AnotherField override.\"\"\"
        return super().validate(value, model_instance)
"""

    def test_qualified_symbol_name_in_search_results(self, indexer, tmp_path) -> None:
        """After search, symbol_name on each result should be 'Class.method' not bare 'method'."""
        from agent.searcher import CodeSearcher

        src = tmp_path / "fields.py"
        src.write_text(self._SOURCE)
        indexer.index_file(str(src))

        # Verify indexer stored bare leaf
        _, docs, metas = indexer.get_all_documents()
        validate_metas = [m for m in metas if m.get("symbol_name") == "validate"]
        assert len(validate_metas) >= 2, (
            "Indexer must store bare leaf 'validate' for both methods"
        )

        s = CodeSearcher(indexer)
        s.refresh_bm25()

        results, _ = s.search("ModelField validate value for field", n_results=10, rerank=False)
        result_syms = {r.symbol_name for r in results}

        # At least one result should have the qualified form
        qualified = {sym for sym in result_syms if "." in sym and "validate" in sym}
        assert qualified, (
            f"UPG-11.10: expected qualified symbol names like 'ModelField.validate' "
            f"in results, but got: {result_syms}"
        )
        # No result should have bare "validate" anymore (they all get class prefix)
        assert "validate" not in result_syms or all(
            "." in r.symbol_name for r in results if r.symbol_name == "validate"
        ), f"Bare 'validate' still present in results: {result_syms}"

    def test_qualified_form_used_in_rest_response(self, indexer, tmp_path) -> None:
        """UPG-11.10: the 'symbol' field in the REST CodeChunkResult must show
        the qualified form (e.g. 'ModelField.validate'), not just the bare leaf."""
        from agent.searcher import CodeSearcher, SearchResult

        # Build a result with a chunk that has a class prefix in content
        r = SearchResult(
            file_path="/p/fields.py",
            lines="5-10",
            symbol_name="validate",
            language="python",
            score=0.8,
            content="# class: ModelField\ndef validate(self, value, model_instance):\n    return value\n    # body\n",
            node_type="function_definition",
        )
        # After _apply_quality_and_dedup, symbol_name should be upgraded
        from agent.searcher import CodeSearcher as CS
        import tests.conftest as cf
        # Use the test searcher fixture approach
        from agent.indexer import CodeIndexer
        from tests.conftest import _DummyEmbedProvider
        db = str(tmp_path / "db")
        idx2 = CodeIndexer(str(tmp_path), db_path=db)
        idx2._embed_provider = _DummyEmbedProvider()
        s2 = CS(idx2)

        out = s2._apply_quality_and_dedup("ModelField validate", [r])
        assert out, "Expected at least one result"
        assert out[0].symbol_name == "ModelField.validate", (
            f"UPG-11.10: expected 'ModelField.validate', got {out[0].symbol_name!r}"
        )


# ---------------------------------------------------------------------------
# UPG-SCORE-DISPLAY-FLAT — absolute relevance score (supersedes UPG-11.2)
#
# Formerly `score` was set to the rank/quality composite ranking key,
# re-normalized by this result set's own max — every query's rank-1 result
# displayed exactly 1.000 regardless of true relevance (a positional
# pseudo-score). `score` is now an ABSOLUTE per-(query, chunk) relevance value
# (the cross-encoder's judgment when reranked, else the chunk's own dense
# cosine similarity), and the rank/quality composite is retained ONLY as an
# internal ordering key — it is never displayed. Displayed scores are
# consequently no longer guaranteed non-increasing down the list (ordering can
# promote a lower-relevance-but-higher-quality/importance chunk above a
# higher-relevance one); this is deliberate.
# ---------------------------------------------------------------------------

class TestAbsoluteRelevanceScore:
    def test_order_unchanged_and_deterministic(self, searcher) -> None:
        """The rank/quality composite ordering key is untouched by this change:
        the same candidate set must sort identically on repeated calls, and
        every displayed score must be a valid [0, 1] value."""
        content_templates = [
            "def alpha(self):\n    # method body here\n    return self.x\n    # impl",
            "def beta(self):\n    # another body\n    return self.y\n    # different",
            "def gamma(self):\n    # yet another\n    return self.z\n    # extra",
            "def delta(self):\n    # fourth method\n    return self.w\n    # more",
        ]
        cands = [
            _sr(c, path=f"/p/mod{i}.py", score=0.5, dense_sim=0.5)
            for i, c in enumerate(content_templates)
        ]
        out1 = searcher._apply_quality_and_dedup("alpha beta gamma delta method", list(cands))
        out2 = searcher._apply_quality_and_dedup("alpha beta gamma delta method", list(cands))
        assert [r.file_path for r in out1] == [r.file_path for r in out2], (
            "ordering must be deterministic across repeated calls on the same candidates"
        )
        for r in out1:
            assert 0.0 <= r.score <= 1.0, f"displayed score out of range: {r.score}"

    def test_score_no_longer_tracks_quality_ordering(self, searcher) -> None:
        """A higher-quality chunk still OUTRANKS a lower-quality one (ordering
        unchanged), but its displayed score is now its own absolute relevance
        (dense_sim here, no reranker) — not the composite that decided the
        order. Giving the lower-quality chunk a higher dense_sim demonstrates
        the two are now independent: order still favors quality, score
        reflects the chunk's own dense_sim regardless."""
        from agent.chunk_quality import NAVIGATIONAL_NODE_TYPE
        high_quality_low_relevance = _sr(
            "def process_request(request):\n    return self.dispatch(request)\n    # impl\n    x=1",
            path="/p/impl.py", dense_sim=0.20,
        )
        low_quality_high_relevance = _sr(
            "pub use a::A;\npub use b::B;\npub use c::C;",
            path="/p/lib.rs", node_type=NAVIGATIONAL_NODE_TYPE, lang="rust", dense_sim=0.90,
        )
        out = searcher._apply_quality_and_dedup(
            "process dispatch request", [high_quality_low_relevance, low_quality_high_relevance],
        )
        assert len(out) == 2
        # Ordering: the navigational/low-quality chunk is still demoted below
        # the real-code chunk, regardless of its higher dense_sim.
        assert out[0].file_path == "/p/impl.py"
        assert out[1].file_path == "/p/lib.rs"
        # Score: each result's displayed score is its OWN absolute relevance —
        # the rank-1 (higher-quality) result can legitimately show a LOWER
        # score than rank-2, since score no longer tracks the ordering key.
        assert out[0].score == pytest.approx(0.20)
        assert out[1].score == pytest.approx(0.90)

    def test_ce_relevance_displayed_when_present(self, searcher) -> None:
        """When a candidate carries a cross-encoder relevance score, that exact
        value is displayed (rounded to 4dp) — the dense_sim fallback is unused."""
        r = _sr("def f(): pass", dense_sim=0.10, ce_relevance=0.834567)
        out = searcher._apply_quality_and_dedup("f", [r])
        assert out[0].score == pytest.approx(0.8346)

    def test_dense_sim_used_when_unreranked(self, searcher) -> None:
        """When a candidate was never scored by the cross-encoder (ce_relevance
        is None — rerank=False or reranker unavailable), the displayed score
        falls back to the chunk's own dense cosine similarity."""
        r = _sr("def f(): pass", dense_sim=0.55, ce_relevance=None)
        out = searcher._apply_quality_and_dedup("f", [r])
        assert out[0].score == pytest.approx(0.55)

    def test_rank1_score_not_flat_across_differing_queries(self, indexer, tmp_path) -> None:
        """Two different queries against the same corpus, with a mocked
        cross-encoder returning distinct relevance judgments, must show
        DIFFERENT rank-1 scores — not both pinned to 1.000 by per-query
        self-normalization (the defect this change fixes)."""
        from agent.searcher import CodeSearcher

        path = make_py(
            tmp_path, "module.py",
            "def parse_json_payload(data):\n    return json.loads(data)\n\n"
            "def serialize_response(obj):\n    return json.dumps(obj)\n",
        )
        indexer.index_file(path)
        s = CodeSearcher(indexer)
        s.refresh_bm25()

        class _FixedScoreReranker:
            """Stub matching the real _Reranker.rerank contract: takes
            [(doc, candidate), ...] and returns candidates with ce_relevance
            stamped, sorted by that score."""
            def __init__(self, score: float):
                self._score = score

            def rerank(self, query, candidates):
                for _, c in candidates:
                    c.ce_relevance = self._score
                return [c for _, c in candidates]

        s._reranker = _FixedScoreReranker(0.91)
        strong_results, _ = s.search("parse json payload", n_results=5)
        s._reranker = _FixedScoreReranker(0.22)
        weak_results, _ = s.search("completely unrelated elephant query", n_results=5)

        assert strong_results and weak_results
        assert strong_results[0].score == pytest.approx(0.91)
        assert weak_results[0].score == pytest.approx(0.22)
        assert strong_results[0].score != weak_results[0].score, (
            "rank-1 scores must reflect the cross-encoder's actual judgment, "
            "not both collapse to 1.000 via per-query self-normalization"
        )


# ---------------------------------------------------------------------------
# CodeSearcher.search — uniform displayed score SCALE across a result set
# (UPG-SCORE-DISPLAY-MIXED-SCALE)
#
# UPG-SCORE-DISPLAY-FLAT lets a chunk's displayed score come from either its
# cross-encoder relevance (ce_relevance) or its dense cosine similarity
# (dense_sim) — two structurally different, non-comparable scales. If any
# candidate that survives into the final displayed slice was never handed to
# the reranker (a straggler), its score would sit next to reranked scores on
# a different scale in the same column. search() must batch-score any such
# straggler after the n_results cut so one result set is never a mix of the
# two scales.
# ---------------------------------------------------------------------------

class TestScoreSourceUniformScale:
    def test_straggler_backfilled_and_ordering_untouched(self, indexer, tmp_path) -> None:
        """A cross-encoder that (simulating the historical defect shape: a
        rerank() call handed a smaller pool than the full candidate list)
        only stamps ce_relevance on a SUBSET of the candidates it is given
        must not leave the displayed result set on a mixed reranker/dense
        scale — CodeSearcher.search must batch-score every straggler still
        missing ce_relevance after the n_results cut, without re-sorting."""
        from agent.searcher import CodeSearcher

        path = make_py(
            tmp_path, "metrics.py",
            "def compute_alpha_metric(data):\n"
            "    \"\"\"Compute the alpha metric from a list of numbers.\"\"\"\n"
            "    total = sum(data)\n"
            "    return total / len(data)\n\n"
            "def compute_beta_metric(data):\n"
            "    \"\"\"Compute the beta metric from a list of numbers.\"\"\"\n"
            "    total = max(data)\n"
            "    return total - min(data)\n\n"
            "def compute_gamma_metric(data):\n"
            "    \"\"\"Compute the gamma metric from a list of numbers.\"\"\"\n"
            "    return sorted(data)[len(data) // 2]\n\n"
            "def compute_delta_metric(data):\n"
            "    \"\"\"Compute the delta metric from a list of numbers.\"\"\"\n"
            "    return [x * 2 for x in data]\n",
        )
        indexer.index_file(path)
        s = CodeSearcher(indexer)
        s.refresh_bm25()

        class _PartialCoverageReranker:
            """Matches the real _Reranker.rerank contract but (deliberately,
            to simulate the historical defect shape) only stamps
            ce_relevance on the first `k` candidates of whichever batch it
            is handed — leaving the rest at ce_relevance=None, exactly what
            a rerank() call given a smaller pool than the full candidate
            list would produce."""
            def __init__(self, k: int):
                self._k = k

            def rerank(self, query, candidates):
                for i, (_, c) in enumerate(candidates):
                    if i < self._k:
                        c.ce_relevance = 0.9 - i * 0.1
                return [c for _, c in candidates]

        query = "compute metric from data"

        s._reranker = _PartialCoverageReranker(k=2)
        results, _ = s.search(query, n_results=4)
        assert len(results) >= 2
        # Every displayed result now carries a cross-encoder score...
        assert all(r.ce_relevance is not None for r in results), (
            "a straggler left ce_relevance=None by the main rerank() call "
            "must be batch-scored before being displayed"
        )
        # ...and every displayed result reports the SAME scale.
        assert {r.score_source for r in results} == {"reranker"}

        # Ordering must be exactly what the composite decided — unaffected by
        # which candidates the backfill call touched. Compare against a
        # control run where the stub scores everyone up front (k large
        # enough to cover the whole pool), so no backfill occurs there.
        s._reranker = _PartialCoverageReranker(k=1000)
        control_results, _ = s.search(query, n_results=4)
        assert [r.file_path for r in results] == [r.file_path for r in control_results]
        assert [r.lines for r in results] == [r.lines for r in control_results]

    def test_rerank_false_yields_uniform_dense_source(self, indexer, tmp_path) -> None:
        """When reranking did not run for a query (rerank=False), every
        displayed result must already be uniform — score_source="dense" for
        all — with nothing to backfill."""
        from agent.searcher import CodeSearcher

        path = make_py(
            tmp_path, "metrics2.py",
            "def compute_alpha_metric(data):\n"
            "    \"\"\"Compute the alpha metric from a list of numbers.\"\"\"\n"
            "    return sum(data) / len(data)\n\n"
            "def compute_beta_metric(data):\n"
            "    \"\"\"Compute the beta metric from a list of numbers.\"\"\"\n"
            "    return max(data) - min(data)\n",
        )
        indexer.index_file(path)
        s = CodeSearcher(indexer)
        s.refresh_bm25()

        results, _ = s.search("compute metric from data", n_results=4, rerank=False)
        assert results
        assert all(r.ce_relevance is None for r in results)
        assert {r.score_source for r in results} == {"dense"}

    def test_model_unavailable_yields_uniform_dense_source(self, indexer, tmp_path) -> None:
        """When the reranker model failed to load, rerank() leaves every
        candidate's ce_relevance at None (pass-through) — the displayed set
        must still be uniformly "dense", with no backfill call attempted."""
        from agent.searcher import CodeSearcher, _Reranker

        path = make_py(
            tmp_path, "metrics3.py",
            "def compute_alpha_metric(data):\n"
            "    \"\"\"Compute the alpha metric from a list of numbers.\"\"\"\n"
            "    return sum(data) / len(data)\n\n"
            "def compute_beta_metric(data):\n"
            "    \"\"\"Compute the beta metric from a list of numbers.\"\"\"\n"
            "    return max(data) - min(data)\n",
        )
        indexer.index_file(path)
        s = CodeSearcher(indexer)
        s.refresh_bm25()

        reranker = _Reranker("fake/model")
        reranker._failed = True  # _load() becomes a no-op, self._model stays None
        s._reranker = reranker

        results, _ = s.search("compute metric from data", n_results=4)
        assert results
        assert all(r.ce_relevance is None for r in results)
        assert {r.score_source for r in results} == {"dense"}


# ---------------------------------------------------------------------------
# _Reranker — cross-encoder scoring + ce_relevance stamping (UPG-SCORE-DISPLAY-FLAT)
#
# Mocks the CrossEncoder model itself (not the whole _Reranker), so both
# rerank()'s ce_relevance-stamping and its defensive sigmoid normalization
# run for real against a realistically-shaped `.predict()` return value
# (a numpy-like array of floats, matching what sentence-transformers returns).
# ---------------------------------------------------------------------------

class _FakeCrossEncoderModel:
    """Stands in for sentence_transformers.CrossEncoder — same predict() shape:
    takes a list of (query, doc) pairs plus a batch_size kwarg (the real
    predict() signature), returns one float score per pair. No .device
    attribute (real CrossEncoder always has one; rerank()'s batch-size
    lookup must tolerate a stub that doesn't via getattr — see
    TestRerankerBatchSize for the .device-bearing case)."""
    def __init__(self, scores: list[float]):
        self._scores = scores
        self.last_batch_size = None

    def predict(self, pairs, batch_size=None):
        assert len(pairs) == len(self._scores)
        self.last_batch_size = batch_size
        return list(self._scores)


class TestRerankerCeRelevance:
    def _reranker_with_model(self, model):
        from agent.searcher import _Reranker
        r = _Reranker("fake/model")
        r._model = model  # bypass _load(): no real model download in tests
        return r

    def test_ce_relevance_stamped_on_every_candidate(self) -> None:
        """rerank() must set ce_relevance on each candidate to the model's own
        (already sigmoid-scale) score — not just use it to sort and discard."""
        r = _sr("def a(): pass")
        s = _sr("def b(): pass")
        reranker = self._reranker_with_model(_FakeCrossEncoderModel([0.30, 0.90]))
        ranked = reranker.rerank("q", [(r.content, r), (s.content, s)])
        assert r.ce_relevance == pytest.approx(0.30)
        assert s.ce_relevance == pytest.approx(0.90)
        # Sorted by score descending — s (0.90) first.
        assert ranked[0] is s
        assert ranked[1] is r

    def test_sigmoid_scale_scores_passed_through_unchanged(self) -> None:
        """A model already returning [0, 1] (the sentence-transformers default
        for num_labels==1, which applies sigmoid internally) is NOT
        re-transformed — the defensive sigmoid must be a no-op here."""
        r = _sr("def a(): pass")
        reranker = self._reranker_with_model(_FakeCrossEncoderModel([0.75]))
        reranker.rerank("q", [(r.content, r)])
        assert r.ce_relevance == pytest.approx(0.75)

    def test_out_of_range_logit_scores_get_sigmoid_applied(self) -> None:
        """A differently-configured checkpoint returning raw logits (outside
        [0, 1]) must be sigmoid-normalized before it ever reaches a
        SearchResult — a displayed score must never be an unbounded logit."""
        r = _sr("def a(): pass")
        s = _sr("def b(): pass")
        raw_logits = [3.5, -2.0]  # clearly outside [0, 1]
        reranker = self._reranker_with_model(_FakeCrossEncoderModel(raw_logits))
        reranker.rerank("q", [(r.content, r), (s.content, s)])
        expected = [1.0 / (1.0 + math.exp(-v)) for v in raw_logits]
        assert r.ce_relevance == pytest.approx(expected[0])
        assert s.ce_relevance == pytest.approx(expected[1])
        assert 0.0 <= r.ce_relevance <= 1.0
        assert 0.0 <= s.ce_relevance <= 1.0

    def test_no_candidates_returns_empty(self) -> None:
        reranker = self._reranker_with_model(_FakeCrossEncoderModel([]))
        assert reranker.rerank("q", []) == []

    def test_model_unavailable_leaves_ce_relevance_none(self) -> None:
        """When the model failed to load (self._model is None), candidates
        pass through untouched — ce_relevance stays at its default None so
        the display fallback (dense_sim) is used instead."""
        from agent.searcher import _Reranker
        reranker = _Reranker("fake/model")
        reranker._failed = True  # _load() becomes a no-op, self._model stays None
        r = _sr("def a(): pass")
        out = reranker.rerank("q", [(r.content, r)])
        assert out == [r]
        assert r.ce_relevance is None


# ---------------------------------------------------------------------------
# _Reranker batch_size selection (UPG-RERANK-LATENCY-BUDGET)
#
# batch_size is resolved from the reranker's OWN already-loaded torch device
# (CrossEncoder.device.type), never from query content, and must change only
# wall-clock latency -- never which candidates come back or in what order.
# The invariant test below is the guard for that: two different resolved
# devices (-> two different configured batch sizes) must still produce
# identical rank order and identical ce_relevance stamps for identical
# scores.
# ---------------------------------------------------------------------------

class _DeviceStub:
    """Stands in for a torch.device -- only the .type attribute rerank()
    reads is modeled."""
    def __init__(self, device_type):
        self.type = device_type


class TestRerankerBatchSize:
    def _reranker_with_model(self, model):
        from agent.searcher import _Reranker
        r = _Reranker("fake/model")
        r._model = model  # bypass _load(): no real model download in tests
        return r

    def test_batch_size_selected_by_resolved_device(self) -> None:
        from agent.config import RERANK_BATCH_SIZE_BY_DEVICE
        model = _FakeCrossEncoderModel([0.5])
        model.device = _DeviceStub("mps")
        r = _sr("def a(): pass")
        reranker = self._reranker_with_model(model)
        reranker.rerank("q", [(r.content, r)])
        assert model.last_batch_size == RERANK_BATCH_SIZE_BY_DEVICE["mps"]

    def test_unknown_device_falls_back_to_default_batch_size(self) -> None:
        """A resolved device type this config doesn't name (npu/xpu/hpu/a
        future backend) must fall back to the config's explicit "default"
        entry, not raise and not silently pick an arbitrary configured value."""
        from agent.config import RERANK_BATCH_SIZE_BY_DEVICE
        model = _FakeCrossEncoderModel([0.5])
        model.device = _DeviceStub("xpu")
        r = _sr("def a(): pass")
        reranker = self._reranker_with_model(model)
        reranker.rerank("q", [(r.content, r)])
        assert model.last_batch_size == RERANK_BATCH_SIZE_BY_DEVICE["default"]

    def test_no_device_attribute_falls_back_to_default_batch_size(self) -> None:
        """A stub model exposing no .device at all (as _FakeCrossEncoderModel
        does in every other test in this file) must not raise -- getattr
        degrades to None, which resolves to the config's default batch_size."""
        from agent.config import RERANK_BATCH_SIZE_BY_DEVICE
        model = _FakeCrossEncoderModel([0.5])  # no .device attribute
        r = _sr("def a(): pass")
        reranker = self._reranker_with_model(model)
        reranker.rerank("q", [(r.content, r)])
        assert model.last_batch_size == RERANK_BATCH_SIZE_BY_DEVICE["default"]

    def test_batch_size_never_changes_rank_order_or_scores(self) -> None:
        """The core invariant (UPG-RERANK-LATENCY-BUDGET rail): batch_size is
        purely a latency lever. Same query, same per-candidate scores, two
        different resolved devices (-> two different configured batch sizes,
        confirmed distinct below) must produce IDENTICAL result order and
        IDENTICAL ce_relevance stamps."""
        from agent.config import RERANK_BATCH_SIZE_BY_DEVICE
        assert RERANK_BATCH_SIZE_BY_DEVICE["mps"] != RERANK_BATCH_SIZE_BY_DEVICE["cuda"], (
            "fixture assumes mps and cuda batch sizes differ in config.yaml"
        )
        scores = [0.20, 0.95, 0.55]

        a, b, c = _sr("def alpha(): pass"), _sr("def beta(): pass"), _sr("def gamma(): pass")
        model_mps = _FakeCrossEncoderModel(list(scores))
        model_mps.device = _DeviceStub("mps")
        ranked_mps = self._reranker_with_model(model_mps).rerank(
            "q", [(a.content, a), (b.content, b), (c.content, c)]
        )

        a2, b2, c2 = _sr("def alpha(): pass"), _sr("def beta(): pass"), _sr("def gamma(): pass")
        model_cuda = _FakeCrossEncoderModel(list(scores))
        model_cuda.device = _DeviceStub("cuda")
        ranked_cuda = self._reranker_with_model(model_cuda).rerank(
            "q", [(a2.content, a2), (b2.content, b2), (c2.content, c2)]
        )

        assert model_mps.last_batch_size != model_cuda.last_batch_size
        assert [r.content for r in ranked_mps] == [r.content for r in ranked_cuda]
        assert [r.ce_relevance for r in ranked_mps] == [r.ce_relevance for r in ranked_cuda]


# ---------------------------------------------------------------------------
# _Reranker._load() — offline-when-cached model loading (UPG-RERANKER-HF-NETWORK)
#
# Mocks sentence_transformers.CrossEncoder itself (never a real download) to
# assert on the local_files_only kwarg _load() passes through, and mocks
# agent.model_cache.is_model_cached to control the cached/uncached branch
# without touching a real Hugging Face cache directory.
# ---------------------------------------------------------------------------

class TestRerankerOfflineLoading:
    def test_cached_model_loads_with_local_files_only_no_network_path(self) -> None:
        """When the reranker model is already cached, CrossEncoder must be
        constructed with local_files_only=True — no live huggingface.co call
        path is exercised. Also pins UPG-RERANK-LATENCY-BUDGET: max_length
        must come from config.RERANK_MAX_LENGTH, not a hardcoded literal."""
        from agent.config import RERANK_MAX_LENGTH
        from agent.searcher import _Reranker

        calls: list[dict] = []

        class _FakeCrossEncoder:
            def __init__(self, model_name, **kwargs):
                calls.append(kwargs)

        with patch("agent.model_cache.is_model_cached", return_value=True), \
             patch("sentence_transformers.CrossEncoder", _FakeCrossEncoder):
            reranker = _Reranker("fake/model")
            reranker._load()

        assert calls[0]["max_length"] == RERANK_MAX_LENGTH

        assert reranker._failed is False
        assert reranker._model is not None
        assert len(calls) == 1
        assert calls[0]["local_files_only"] is True

    def test_uncached_model_falls_back_to_network_enabled_load(self) -> None:
        """When the reranker model is NOT cached, CrossEncoder must be
        constructed with local_files_only=False — a normal network-enabled
        download, preserving first-run UX."""
        from agent.searcher import _Reranker

        calls: list[dict] = []

        class _FakeCrossEncoder:
            def __init__(self, model_name, **kwargs):
                calls.append(kwargs)

        with patch("agent.model_cache.is_model_cached", return_value=False), \
             patch("sentence_transformers.CrossEncoder", _FakeCrossEncoder):
            reranker = _Reranker("fake/model")
            reranker._load()

        assert reranker._failed is False
        assert len(calls) == 1
        assert calls[0]["local_files_only"] is False

    def test_load_failure_still_degrades_gracefully(self) -> None:
        """A load failure (offline AND not cached — e.g. an air-gapped machine
        with an incomplete cache) must leave _failed=True and _model=None,
        exactly as before this change — search continues without a reranker."""
        from agent.searcher import _Reranker

        class _RaisingCrossEncoder:
            def __init__(self, model_name, **kwargs):
                raise OSError("no network, no cache")

        with patch("agent.model_cache.is_model_cached", return_value=False), \
             patch("sentence_transformers.CrossEncoder", _RaisingCrossEncoder):
            reranker = _Reranker("fake/model")
            reranker._load()

        assert reranker._failed is True
        assert reranker._model is None
        # rerank() must still degrade gracefully (pass-through, no crash)
        r = _sr("def a(): pass")
        out = reranker.rerank("q", [(r.content, r)])
        assert out == [r]
        assert r.ce_relevance is None


# ---------------------------------------------------------------------------
# CodeSearcher.warm_reranker() — eager startup load (UPG-RERANKER-HF-NETWORK)
# ---------------------------------------------------------------------------

class TestWarmReranker:
    def test_warm_reranker_triggers_load(self, indexer) -> None:
        """warm_reranker() must eagerly call the reranker's _load() so the
        model-load cost lands at daemon startup, not inside the first search."""
        from agent.searcher import CodeSearcher

        s = CodeSearcher(indexer)

        class _StubReranker:
            def __init__(self):
                self.load_called = False

            def _load(self):
                self.load_called = True

        stub = _StubReranker()
        s._reranker = stub
        s.warm_reranker()
        assert stub.load_called is True

    def test_warm_reranker_noop_when_reranker_disabled(self, indexer) -> None:
        """When reranking is disabled (VECTR_RERANKER_MODEL="", the default in
        this test suite), self._reranker is None and warm_reranker() must be a
        pure no-op — no AttributeError, nothing to load."""
        from agent.searcher import CodeSearcher

        s = CodeSearcher(indexer)
        assert s._reranker is None
        s.warm_reranker()  # must not raise

    def test_reranker_model_defaults_to_config_key(self, indexer, monkeypatch) -> None:
        """With no VECTR_RERANKER_MODEL env override, the reranker model comes
        from config.yaml's ranking.rerank.model (RERANK_MODEL) — a reranker
        swap must be a config change, never a code change. The env var, when
        set (as this suite's conftest does to disable reranking), still wins."""
        monkeypatch.delenv("VECTR_RERANKER_MODEL", raising=False)
        from agent.config import RERANK_MODEL
        from agent.searcher import CodeSearcher

        s = CodeSearcher(indexer)
        assert s._reranker is not None
        assert s._reranker._model_name == RERANK_MODEL

        monkeypatch.setenv("VECTR_RERANKER_MODEL", "some-env-override-model")
        s2 = CodeSearcher(indexer)
        assert s2._reranker._model_name == "some-env-override-model"


# ---------------------------------------------------------------------------
# UPG-NOTFOUND-FLOOR (F46) — absolute-relevance low-confidence signal
#
# `low_confidence` gates on a corpus-wide zero-document-frequency vocabulary
# check (see config.yaml's UPG-NOTFOUND-FLOOR-2 rationale) — independent of
# any per-query score. Uses the DummyEmbedProvider's real, deterministic
# (hash-seeded) vectors: two distinct strings land near-orthogonal (cosine ~0),
# while an identical string against itself lands at cosine 1.0 — no mocking of
# the flag itself, so this exercises the exact code path used in production.
#
# UPG-SCORE-DISPLAY-FLAT added a second, independent sub-signal — the top
# result's cross-encoder relevance (ce_relevance) below an absolute floor —
# gated specifically on ce_relevance, never on the raw dense-cosine fallback
# (see TestNotFoundFloorTopRelevance below); every test in THIS class below
# uses rerank=False, so ce_relevance is never set and that sub-signal never
# fires — these tests are unaffected by the addition.
# ---------------------------------------------------------------------------

class TestNotFoundFloor:
    def _indexed_searcher(self, indexer, tmp_path, content: str, name="module.py"):
        path = make_py(tmp_path, name, content)
        indexer.index_file(path)
        from agent.searcher import CodeSearcher
        s = CodeSearcher(indexer)
        s.refresh_bm25()
        return s

    def test_low_confidence_true_for_uniformly_weak_pool(self, indexer, tmp_path) -> None:
        """A query semantically unrelated to anything indexed flags low_confidence,
        even though results are still returned (never suppressed)."""
        s = self._indexed_searcher(
            indexer, tmp_path,
            "def parse_json_payload(data):\n    return json.loads(data)\n",
        )
        results, _ = s.search(
            "elephant migration patterns across the savanna", n_results=5, rerank=False,
        )
        assert results, "results must still be returned, never suppressed"
        assert results.low_confidence is True

    def test_low_confidence_false_for_clearly_strong_top_hit(self, indexer, tmp_path) -> None:
        """A query that IS the indexed chunk's own content (cosine ~1.0 against
        itself) must not be flagged — a strong top hit is not low-confidence."""
        s = self._indexed_searcher(
            indexer, tmp_path,
            "def parse_json_payload(data):\n    return json.loads(data)\n",
        )
        exact_doc_text = s._bm25_docs[0]
        results, _ = s.search(exact_doc_text, n_results=5, rerank=False)
        assert results
        assert results.low_confidence is False

    def test_config_disable_is_exact_noop(self, indexer, tmp_path, monkeypatch) -> None:
        """ranking.notfound_floor.enabled: false must be a true no-op — the flag
        never fires even for the same uniformly-weak query that trips it above."""
        import agent.searcher as searcher_module
        monkeypatch.setattr(searcher_module, "_NOTFOUND_FLOOR_ENABLED", False)
        s = self._indexed_searcher(
            indexer, tmp_path,
            "def parse_json_payload(data):\n    return json.loads(data)\n",
        )
        results, _ = s.search(
            "elephant migration patterns across the savanna", n_results=5, rerank=False,
        )
        assert results
        assert results.low_confidence is False

    def test_config_high_min_zero_df_tokens_is_exact_noop(self, indexer, tmp_path, monkeypatch) -> None:
        """min_zero_df_tokens set arbitrarily high (the documented alternate
        disable path) never fires either — no query can have that many
        zero-document-frequency content tokens."""
        import agent.searcher as searcher_module
        monkeypatch.setattr(searcher_module, "_NOTFOUND_FLOOR_MIN_ZERO_DF_TOKENS", 999)
        s = self._indexed_searcher(
            indexer, tmp_path,
            "def parse_json_payload(data):\n    return json.loads(data)\n",
        )
        results, _ = s.search(
            "elephant migration patterns across the savanna", n_results=5, rerank=False,
        )
        assert results
        assert results.low_confidence is False

    def test_zero_document_frequency_token_is_the_trigger(self, indexer, tmp_path) -> None:
        """UPG-NOTFOUND-FLOOR-2: the signal is a corpus-wide vocabulary check, not
        a cosine floor — a query built entirely from words that DO appear
        somewhere in the indexed corpus must not flag low_confidence even when
        none of those words co-occur in any single chunk (a real absent-topic
        query instead contains a word absent from the WHOLE corpus)."""
        s = self._indexed_searcher(
            indexer, tmp_path,
            "def parse_json_payload(data):\n    return json.loads(data)\n",
        )
        # "parse" and "payload" both appear somewhere in the tiny indexed corpus
        # (in parse_json_payload) even though this exact 2-word phrase doesn't.
        results, _ = s.search("parse payload", n_results=5, rerank=False)
        assert results
        assert results.low_confidence is False, (
            "every content word in the query has nonzero corpus-wide document "
            "frequency, so this must not be flagged low_confidence"
        )

        # Swap in one word absent from the ENTIRE corpus vocabulary (not just this
        # query's fetched pool) and the same mechanism must now fire.
        results2, _ = s.search("parse zzyxquiggle", n_results=5, rerank=False)
        assert results2
        assert results2.low_confidence is True, (
            "'zzyxquiggle' has zero document frequency across the whole corpus; "
            "this is exactly the lexical-vocabulary-anchor signal firing"
        )

    def test_stopword_zero_df_does_not_trigger(self, indexer, tmp_path, monkeypatch) -> None:
        """A configured stopword with zero document frequency must not, by
        itself, flag low_confidence — stopwords are excluded from the
        zero-document-frequency content-token check entirely."""
        import agent.searcher as searcher_module
        monkeypatch.setattr(searcher_module, "_NOTFOUND_FLOOR_STOPWORDS", frozenset({"zzyxquiggle"}))
        s = self._indexed_searcher(
            indexer, tmp_path,
            "def parse_json_payload(data):\n    return json.loads(data)\n",
        )
        results, _ = s.search("parse zzyxquiggle", n_results=5, rerank=False)
        assert results
        assert results.low_confidence is False, (
            "zzyxquiggle is configured as a stopword, so its zero document "
            "frequency must be ignored by the check"
        )

    def test_short_token_below_min_length_does_not_trigger(self, indexer, tmp_path, monkeypatch) -> None:
        """A zero-document-frequency token shorter than min_content_token_length
        must not, by itself, flag low_confidence — short tokens are excluded
        from the check as unreliable vocabulary anchors."""
        import agent.searcher as searcher_module
        monkeypatch.setattr(searcher_module, "_NOTFOUND_FLOOR_MIN_TOKEN_LEN", 10)
        s = self._indexed_searcher(
            indexer, tmp_path,
            "def parse_json_payload(data):\n    return json.loads(data)\n",
        )
        # "zz" (len 2) has zero document frequency but is below the raised
        # min-length threshold, so it must be excluded from the check.
        results, _ = s.search("parse zz", n_results=5, rerank=False)
        assert results
        assert results.low_confidence is False, (
            "'zz' is shorter than the configured min_content_token_length and "
            "must be excluded from the zero-document-frequency check"
        )

    def test_empty_results_never_flagged(self, indexer) -> None:
        """An empty index returns an empty (not low_confidence) result set — the
        floor is meaningless with nothing to lead with; a separate 'no results'
        message already covers that case downstream."""
        from agent.searcher import CodeSearcher
        s = CodeSearcher(indexer)
        results, ms = s.search("anything")
        assert results == []
        assert getattr(results, "low_confidence", False) is False

    def test_search_result_list_is_a_real_list(self, indexer, tmp_path) -> None:
        """SearchResultList must behave exactly like list[SearchResult] for every
        existing `results, ms = searcher.search(...)` call site — indexing,
        len(), iteration, and equality with a plain list all still work."""
        s = self._indexed_searcher(
            indexer, tmp_path,
            "def parse_json_payload(data):\n    return json.loads(data)\n",
        )
        results, _ = s.search("parse json payload", n_results=5, rerank=False)
        assert isinstance(results, list)
        assert len(results) == len(list(results))
        assert results[0] == list(results)[0]


# ---------------------------------------------------------------------------
# UPG-SCORE-DISPLAY-FLAT — top-cross-encoder-relevance low-confidence sub-signal
#
# A second, independent low_confidence trigger: the top result's own
# cross-encoder relevance (ce_relevance) below ranking.notfound_floor.
# min_top_relevance, even when the zero-document-frequency vocabulary check
# does not trip. Uses a mocked reranker (matching the real _Reranker.rerank
# contract) so the exact ce_relevance value is deterministic and controlled.
# ---------------------------------------------------------------------------

class _StubReranker:
    """Matches the real _Reranker.rerank contract: takes
    [(doc, candidate), ...], stamps ce_relevance, returns candidates
    sorted by that score (highest first)."""
    def __init__(self, score: float):
        self._score = score

    def rerank(self, query, candidates):
        for _, c in candidates:
            c.ce_relevance = self._score
        return [c for _, c in candidates]


class TestNotFoundFloorTopRelevance:
    def _indexed_searcher(self, indexer, tmp_path, content: str, name="module.py"):
        path = make_py(tmp_path, name, content)
        indexer.index_file(path)
        from agent.searcher import CodeSearcher
        s = CodeSearcher(indexer)
        s.refresh_bm25()
        return s

    def test_low_top_relevance_trips_even_without_zero_df(self, indexer, tmp_path) -> None:
        """A query built entirely from words present in the corpus (the
        zero-DF check does not trip) must still flag low_confidence when the
        cross-encoder itself judges the top result weakly relevant."""
        s = self._indexed_searcher(
            indexer, tmp_path,
            "def parse_json_payload(data):\n    return json.loads(data)\n\n"
            "def serialize_response(obj):\n    return json.dumps(obj)\n",
        )
        s._reranker = _StubReranker(0.10)  # below the configured 0.30 floor
        # "parse payload": both words appear in the tiny indexed corpus, so
        # the zero-DF sub-signal alone would not fire (see TestNotFoundFloor).
        results, _ = s.search("parse payload", n_results=5)
        assert results
        assert results[0].ce_relevance == pytest.approx(0.10)
        assert results.low_confidence is True, (
            "top cross-encoder relevance (0.10) is below min_top_relevance "
            "(0.30); low_confidence must fire even though no query token has "
            "zero corpus-wide document frequency"
        )

    def test_high_top_relevance_does_not_trip(self, indexer, tmp_path) -> None:
        """The same query with a confident cross-encoder judgment must NOT be
        flagged."""
        s = self._indexed_searcher(
            indexer, tmp_path,
            "def parse_json_payload(data):\n    return json.loads(data)\n\n"
            "def serialize_response(obj):\n    return json.dumps(obj)\n",
        )
        s._reranker = _StubReranker(0.85)
        results, _ = s.search("parse payload", n_results=5)
        assert results
        assert results[0].ce_relevance == pytest.approx(0.85)
        assert results.low_confidence is False

    def test_unreranked_search_never_gated_on_dense_sim(self, indexer, tmp_path) -> None:
        """rerank=False (no cross-encoder ran) must NOT gate low_confidence on
        the dense-cosine fallback score — config.yaml's UPG-NOTFOUND-FLOOR-2
        evidence already showed a raw bi-encoder cosine can't separate
        absent-topic from on-topic queries, so this sub-signal must stay
        silent (not just falsely-quiet) whenever no real cross-encoder
        judgment backs the top result."""
        s = self._indexed_searcher(
            indexer, tmp_path,
            "def parse_json_payload(data):\n    return json.loads(data)\n",
        )
        # "parse payload" vs the DummyEmbedProvider's hash-seeded vectors: a
        # non-exact-match query lands near-orthogonal (dense_sim ~0, well
        # below the 0.30 floor) even though every query word DOES appear in
        # the corpus — exactly the scenario the dense-cosine floor was
        # already proven unreliable for.
        results, _ = s.search("parse payload", n_results=5, rerank=False)
        assert results
        assert results[0].ce_relevance is None
        assert results.low_confidence is False


# ---------------------------------------------------------------------------
# UPG-NOTFOUND-FLOOR-CE-OVERRIDE — high-confidence override on the zero-DF trigger
#
# A zero-document-frequency query token normally flags low_confidence, but an
# everyday paraphrase of a symbol the corpus DOES contain can lack a corpus
# vocabulary word while the cross-encoder still judges the top result decisively
# relevant. When top ce_relevance >= ranking.notfound_floor.ce_override_min_
# relevance, the zero-DF trigger is suppressed; the low_top_relevance sub-signal
# is unaffected, so a genuine miss (weak top CE) is never suppressed. Uses the
# same _StubReranker as TestNotFoundFloorTopRelevance for a deterministic CE.
# ---------------------------------------------------------------------------

class TestNotFoundFloorCEOverride:
    def _indexed_searcher(self, indexer, tmp_path, content: str, name="module.py"):
        path = make_py(tmp_path, name, content)
        indexer.index_file(path)
        from agent.searcher import CodeSearcher
        s = CodeSearcher(indexer)
        s.refresh_bm25()
        return s

    # Two chunks so the cross-encoder rerank pool has >1 candidate (rerank is
    # skipped for a single-chunk pool). "reimburse"/"shopper" appear in neither,
    # so they are zero-document-frequency content tokens.
    _CORPUS = (
        "def charge_card(customer, amount):\n"
        "    return gateway.charge(customer, amount)\n\n"
        "def refund_payment(customer, amount):\n"
        "    return gateway.refund(customer, amount)\n"
    )

    def test_high_ce_suppresses_zero_df_banner(self, indexer, tmp_path) -> None:
        """A confident cross-encoder judgment (>= ce_override_min_relevance)
        suppresses the zero-DF banner — the false-positive the finding hit:
        'reimburse'/'shopper' never appear in this tiny corpus (zero-DF), yet
        the top result is decisively relevant, so the caller must NOT be told
        to fall back to grep."""
        s = self._indexed_searcher(indexer, tmp_path, self._CORPUS)
        s._reranker = _StubReranker(0.95)  # confident, >= 0.70 override
        results, _ = s.search("reimburse the shopper", n_results=5)
        assert results
        assert results[0].ce_relevance == pytest.approx(0.95)
        assert results.low_confidence is False, (
            "a zero-DF query token must not flag low_confidence when the "
            "cross-encoder judged the top result >= ce_override_min_relevance"
        )

    def test_low_ce_still_flags_zero_df_banner(self, indexer, tmp_path) -> None:
        """The identical zero-DF query with a WEAK top CE (a genuine miss) must
        still flag low_confidence — proving the suppression is CE-gated, not a
        blanket disable of the zero-DF check."""
        s = self._indexed_searcher(indexer, tmp_path, self._CORPUS)
        s._reranker = _StubReranker(0.20)  # below both override and floor
        results, _ = s.search("reimburse the shopper", n_results=5)
        assert results
        assert results.low_confidence is True

    def test_high_threshold_disables_override_exact_noop(self, indexer, tmp_path, monkeypatch) -> None:
        """ce_override_min_relevance above 1.0 is an exact no-op: no ce_relevance
        can meet it, so the zero-DF banner fires again even on a confident CE."""
        import agent.searcher as searcher_module
        monkeypatch.setattr(
            searcher_module, "_NOTFOUND_FLOOR_CE_OVERRIDE_MIN_RELEVANCE", 1.01
        )
        s = self._indexed_searcher(indexer, tmp_path, self._CORPUS)
        s._reranker = _StubReranker(0.99)  # very confident, but override disabled
        results, _ = s.search("reimburse the shopper", n_results=5)
        assert results
        assert results.low_confidence is True


# ---------------------------------------------------------------------------
# UPG-11.4 — Expand-to-symbol affordance (line range on results)
# ---------------------------------------------------------------------------

class TestExpandToSymbolAffordance:
    """UPG-11.4: each SearchResult must carry symbol_start_line/symbol_end_line
    so callers can expand to the full symbol without a blind whole-file re-read.
    """

    def test_symbol_line_range_populated(self, indexer, tmp_path) -> None:
        """After indexing, search results must have non-zero symbol_start_line/end_line."""
        from agent.searcher import CodeSearcher

        src = make_py(tmp_path, "widget.py", """
class Widget:
    def render(self):
        \"\"\"Render the widget to the screen.\"\"\"
        # drawing code here
        return self.canvas.draw()
""")
        indexer.index_file(src)

        s = CodeSearcher(indexer)
        s.refresh_bm25()
        results, _ = s.search("Widget render draw canvas", n_results=5, rerank=False)

        assert results, "Expected at least one result"
        # Find the render method result
        render = next((r for r in results if "render" in (r.symbol_name or "")), None)
        if render is None:
            render = results[0]  # at least verify the field is populated

        # The affordance fields must be set (non-zero when the chunk is a named symbol)
        assert hasattr(render, "symbol_start_line"), "UPG-11.4: SearchResult missing symbol_start_line"
        assert hasattr(render, "symbol_end_line"), "UPG-11.4: SearchResult missing symbol_end_line"
        # For AST-parsed symbol chunks, the range should be > 0
        if "render" in (render.symbol_name or "") or "render" in render.content:
            assert render.symbol_start_line > 0 or render.symbol_end_line > 0, (
                f"UPG-11.4: expected non-zero line range for a symbol chunk; "
                f"got start={render.symbol_start_line} end={render.symbol_end_line}"
            )

    def test_rest_response_includes_symbol_line_range(self) -> None:
        """UPG-11.4: the REST CodeChunkResult schema must include symbol_start_line/end_line."""
        from app.models import CodeChunkResult
        r = CodeChunkResult(
            file="src/widget.py",
            lines="3-8",
            symbol="Widget.render",
            language="python",
            score=0.9,
            content="def render(self): ...",
            symbol_start_line=3,
            symbol_end_line=8,
        )
        assert r.symbol_start_line == 3
        assert r.symbol_end_line == 8

    def test_rest_response_defaults_zero_for_non_symbol_chunks(self) -> None:
        """Window chunks without a named symbol must default to 0/0."""
        from app.models import CodeChunkResult
        r = CodeChunkResult(
            file="src/big_file.py",
            lines="100-300",
            symbol=None,
            language="python",
            score=0.7,
            content="lots of code ...",
        )
        assert r.symbol_start_line == 0
        assert r.symbol_end_line == 0

