"""CodeIndexer: ChromaDB-backed index orchestration (chunking + embed + upsert)."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import chromadb
import numpy as np

from agent.chroma_dispatch import timed_chroma_call as _timed_chroma_call
from agent.chunk_quality import build_purpose_text, is_symbol_bearing_chunk
from agent.config import DUAL_VECTOR_ENABLED as _DUAL_VECTOR_ENABLED
from agent.config import EMBEDDING_DEFAULT_MODEL as _EMBEDDING_DEFAULT_MODEL
from agent.config import FETCH_MISALIGN_NEAREST_MAX
from agent.config import INDEX_GOVERNOR_ENABLED as _INDEX_GOVERNOR_ENABLED
from agent.config import INDEX_GOVERNOR_DUTY_CYCLE as _INDEX_GOVERNOR_DUTY_CYCLE
from agent.config import INDEX_GOVERNOR_MACOS_QOS_CLASS as _INDEX_GOVERNOR_MACOS_QOS_CLASS
from agent.config import INDEX_GOVERNOR_LINUX_NICE_INCREMENT as _INDEX_GOVERNOR_LINUX_NICE_INCREMENT
from agent.config import (
    INDEX_GOVERNOR_MIN_BATCH_SECONDS_FOR_PACING as _INDEX_GOVERNOR_MIN_BATCH_SECONDS_FOR_PACING,
)
from agent.config import (
    INDEX_GOVERNOR_CHECKPOINT_EVERY_BATCHES as _INDEX_GOVERNOR_CHECKPOINT_EVERY_BATCHES,
)
from agent.config import (
    INDEX_GOVERNOR_DEFER_PURPOSE_PASS_ENABLED as _INDEX_GOVERNOR_DEFER_PURPOSE_PASS_ENABLED,
)
from agent.config import vectr_cache_root as _vectr_cache_root
from agent.indexer._constants import (
    EXCLUDED_DIRS,
    _FILE_BATCH_SIZE,
    _EMBED_BATCH_SIZE,
    _UPSERT_BATCH_SIZE,
    _CHUNK_WORKERS,
    _MTIME_CACHE_SCHEMA_KEY,
    INDEXING_SCHEMA_VERSION,
    _EMBED_MODEL_STAMP_FILE,
    _PURPOSE_CACHE_FILE,
    _PURPOSE_CACHE_SCHEMA_KEY,
)
from agent.indexer._chunking import chunk_file
from agent.indexer._priority import lowered_priority, pace
from agent.indexer._types import CodeChunk

logger = logging.getLogger(__name__)


def _foreground_fast_enabled() -> bool:
    """VECTR_FOREGROUND_FAST=1 restores index_workspace()'s pre-governor,
    unthrottled behaviour exactly — no priority clamp, no duty-cycle pacing
    (UPG-INDEX-RESOURCE-GOVERNOR). Set by `vectr start --foreground-fast` /
    `vectr restart --foreground-fast` in the daemon subprocess's env, same
    mechanism as VECTR_MEMORY_ONLY / VECTR_SEARCH_ONLY (main.py `_do_start`).
    Checkpointing (mtime-cache persisted incrementally as Phase 3 progresses)
    is a correctness fix, not a throttle, and is unaffected by this flag.
    """
    return os.getenv("VECTR_FOREGROUND_FAST", "") == "1"


def _chunk_metadata(c: CodeChunk) -> dict:
    """The metadata dict stored alongside a chunk's vector (body or purpose
    collection — same shape for both). Factored out so the streaming embed/
    upsert loops (UPG-INDEX-MEM-STREAMING) build it per-batch instead of once
    per corpus."""
    return {
        "file_path": c.file_path,
        "language": c.language,
        "node_type": c.node_type,
        "start_line": c.start_line,
        "end_line": c.end_line,
        "symbol_name": c.symbol_name,
    }


def _upsert_in_batches(
    collection,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict],
    embeddings: list[list[float]],
    batch_size: int,
    op_label: str = "upsert",
) -> None:
    """Upsert one embed-batch's rows into `collection` in `batch_size` slices
    (SQLite's 999-variable limit: 6 metadata fields x 100 rows = 600 <= 999).
    Bounded to a single embed-batch's worth of rows by every caller, so this
    never itself becomes an O(corpus) structure."""
    for j in range(0, len(ids), batch_size):
        with _timed_chroma_call(op_label):
            collection.upsert(
                ids=ids[j: j + batch_size],
                documents=documents[j: j + batch_size],
                metadatas=metadatas[j: j + batch_size],
                embeddings=embeddings[j: j + batch_size],
            )


def _parse_chunk_id(chunk_id: str) -> tuple[str, int, int] | None:
    """Split a stored `path:start_line-end_line` chunk id into its parts, or
    None when it doesn't parse as that shape.

    Mirrors `agent/render_paths.py::resolve_chunk_id`'s own convention: the
    line-range suffix is split on the LAST colon so a path containing a
    colon is never mangled. Used by `CodeIndexer.fetch_chunks`'s
    misalignment detection (UPG-FETCH-ID-MISALIGN-MSG), where a candidate
    chunk's own id is the only source of its line range — no extra
    metadata fetch needed.
    """
    path, sep, line_range = chunk_id.rpartition(":")
    if not sep:
        return None
    start_str, dash, end_str = line_range.rpartition("-")
    if not dash:
        return None
    try:
        return path, int(start_str), int(end_str)
    except ValueError:
        return None


def _nearest_chunk_ids_by_span(
    candidate_ids: list[str], start_line: int, end_line: int, limit: int,
) -> list[str]:
    """Rank same-file `candidate_ids` by proximity to (`start_line`,
    `end_line`) and return the closest `limit` (UPG-FETCH-ID-MISALIGN-MSG).

    Deterministic arithmetic on the ids' own encoded spans — no embedding or
    ranking involved: an overlapping chunk sorts first (distance 0), then
    non-overlapping chunks by distance from the nearer edge, with the id
    string itself as the final tiebreak. Ids that don't parse as
    `path:start-end` are skipped.
    """
    scored: list[tuple[int, int, str]] = []
    for cid in candidate_ids:
        parsed = _parse_chunk_id(cid)
        if parsed is None:
            continue
        _, c_start, c_end = parsed
        overlaps = c_start <= end_line and c_end >= start_line
        distance = 0 if overlaps else min(abs(c_start - end_line), abs(c_end - start_line))
        scored.append((0 if overlaps else 1, distance, cid))
    scored.sort()
    return [cid for _, _, cid in scored[:limit]]


class CodeIndexer:
    def __init__(
        self,
        workspace_root: str,
        embed_model: str = _EMBEDDING_DEFAULT_MODEL,
        db_path: str | None = None,
        extra_roots: list[str] | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self._extra_roots: list[Path] = [Path(r).resolve() for r in (extra_roots or [])]
        self.embed_model = embed_model

        db_dir = Path(db_path) if db_path else _vectr_cache_root() / "db" / self._workspace_hash()
        db_dir.mkdir(parents=True, exist_ok=True)
        self._db_dir = db_dir

        self._client = chromadb.PersistentClient(path=str(db_dir))
        self._create_collections()
        self._last_indexed: float = 0.0
        self._indexed_files: set[str] = set()
        # Incrementally-maintained per-language stats (UPG-3.1/3.3,
        # UPG-REST-STARVATION). Seeded once by a full metadata scan (now done
        # eagerly below, during construction — see the call to
        # `_ensure_stats_seeded()` at the end of this method), then updated by
        # `_apply_chunk_delta` on every subsequent insert/delete instead of
        # ever re-scanning the collection — see `_ensure_stats_seeded`'s
        # docstring for why this replaced a count-keyed rescan-on-change cache
        # that re-scanned the whole collection on every call while the chunk
        # count was still changing (i.e. throughout a bulk reindex), making
        # `indexed_language_stats()` an O(corpus) operation contending with
        # the writer.
        self._stats_lock = threading.Lock()
        self._stats_seeded = False
        self._lang_chunk_counts: dict[str, int] = {}
        self._lang_files: dict[str, set[str]] = {}
        # Live body-collection chunk count per file path (UPG-CHUNK-DELTA-
        # PARTIAL-FILE-DISCARD). A file's chunks all carry one language (it is
        # derived from the extension), so this is keyed by path alone. It is
        # what lets `_apply_chunk_delta(sign=-1)` tell "some chunks of this
        # file were deleted" apart from "the file's LAST chunk was deleted" —
        # only the latter may drop the file from its `_lang_files` bucket.
        self._file_chunk_counts: dict[str, int] = {}
        # Total chunk counts (body + purpose collections), read by
        # `total_chunks`/`total_purpose_chunks`. Seeded here from a real
        # (but construction-time, always off-the-event-loop) count;
        # refreshed the same way again by
        # `_refresh_chunk_count_caches` at the end of every mutation entry
        # point (index_file/delete_file/index_workspace) — never re-read
        # from the vector store on a later request.
        self._total_chunks_cache: int = 0
        self._purpose_chunks_cache: int = 0
        self._refresh_chunk_count_caches()

        # UPG-PURPOSE-PASS-DEFERRAL: a dedicated single-worker executor for
        # the background purpose-vector pass. max_workers=1 (not the chunk
        # pool's worker count) is load-bearing, not a tuning choice — it is
        # what serializes successive deferred passes (one per
        # index_workspace() call) into the purpose collection instead of
        # letting two overlapping passes race on the same writes; each
        # index_workspace() call submits at most one closure and returns
        # without waiting for it. `_purpose_pending_lock` guards the count
        # `purpose_vectors_pending` reads, incremented when a pass is
        # submitted and decremented when it finishes (success or failure).
        self._purpose_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="vectr-purpose",
        )
        self._purpose_pending_lock = threading.Lock()
        self._purpose_pending_count = 0
        # Guards `_save_mtime_cache`'s file write. Before UPG-PURPOSE-PASS-
        # DEFERRAL, every mtime-cache write happened on the single thread
        # running `index_workspace()`/`index_file()` — never concurrently.
        # `_upsert_purpose_vectors()` itself never touches the mtime cache —
        # all six `_save_mtime_cache()` call sites are on the caller's thread,
        # by design (content-completion and purpose-completion are decoupled
        # signals). What deferral changes is that `index_workspace()` now
        # RETURNS EARLY, while a purpose pass is still running, so a caller —
        # or the watcher's `index_file()` — is free to start another indexing
        # pass much sooner than before, and two such passes' writes can
        # genuinely overlap. This lock only prevents two threads'
        # `write_text()` calls from interleaving into corrupt JSON; it does
        # not resolve the logical lost-update case (one writer's snapshot
        # overwrites the other's) — that case is self-healing (the
        # overwritten file simply gets checkpointed again, or its content
        # safely re-embedded, on a later call), matching this whole
        # mechanism's at-least-once checkpoint philosophy.
        self._mtime_cache_lock = threading.Lock()
        # Same file-corruption guard as `_mtime_cache_lock`, for
        # `purpose_cache.json` (UPG-PURPOSE-RESUME-HOLE) — written from both
        # the caller's own thread (the synchronous/inline path, and
        # `index_workspace`'s one-time migration seed) and the deferred
        # purpose executor's background thread.
        self._purpose_cache_lock = threading.Lock()
        # Number of files `index_workspace()` most recently detected as
        # having current body content but incomplete purpose vectors, before
        # dispatching their backfill (UPG-PURPOSE-RESUME-HOLE) — surfaced via
        # `purpose_backfill_pending_files` for `/v1/status` so a gap this
        # large cannot hide behind an otherwise-green status. 0 whenever dual
        # vectors are disabled, the workspace has nothing indexed yet, or the
        # most recent run found (and dispatched backfill for) no gap.
        self._purpose_backfill_pending_count = 0

        # Deferred: look up get_embed_provider through the package namespace so that
        # test-time monkeypatching of agent.indexer.get_embed_provider is honoured
        # (identical to the original flat-module behaviour where the function lived
        # in the same module namespace that patches target).
        import agent.indexer as _idx
        self._embed_provider = _idx.get_embed_provider(embed_model)

        # Seed the per-language stats cache now, during construction — which
        # always runs off the event loop serving requests, either
        # synchronously before the daemon starts accepting them or on its own
        # background thread (this service's deferred-init path) — so
        # `indexed_language_stats()` (read by both `/v1/status` and
        # `/v1/map`) never performs its first-ever collection scan inside a
        # live request.
        self._ensure_stats_seeded()

    def _workspace_hash(self) -> str:
        return hashlib.md5(str(self.workspace_root).encode()).hexdigest()[:12]

    def _create_collections(self) -> None:
        self._collection = self._client.get_or_create_collection(
            name="code_chunks",
            metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 200,  # default 100 — denser graph, better recall
                "hnsw:search_ef": 100,         # default 10 — wider beam search at query time
                "hnsw:M": 32,                  # default 16 — more neighbours per node
            },
        )
        # ARCH-4: second collection holding the body-stripped "purpose" vector
        # (qualified signature + docstring) for symbol-bearing chunks only —
        # keyed by the SAME chunk_id as `self._collection`. get_or_create_collection
        # means an existing (pre-ARCH-4) workspace transparently gets an empty
        # purpose collection rather than an error; queries against it then return
        # no candidates until the workspace is (re)indexed, which is exactly the
        # graceful body-only fallback the spec calls for.
        self._purpose_collection = self._client.get_or_create_collection(
            name="code_chunks_purpose",
            metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 200,
                "hnsw:search_ef": 100,
                "hnsw:M": 32,
            },
        )

    def _recreate_collections(self) -> None:
        """Drop and recreate both vector collections.

        ChromaDB pins a collection's embedding dimensionality at first insert
        and keeps it even after every entry is deleted, so an embedding-model
        swap to a DIFFERENT-dimension model would crash the per-file
        delete-then-reinsert rebuild (`force=True`) against the old
        collection. On an embed-model stamp mismatch the collections
        themselves must be dropped, not just their contents — this also
        guarantees no old-model vector can survive regardless of any
        per-file bookkeeping.
        """
        for name in ("code_chunks", "code_chunks_purpose"):
            try:
                self._client.delete_collection(name)
            except Exception:
                pass  # collection may not exist yet — nothing to drop
        self._create_collections()
        # Both collections are freshly (re)created and empty — no scan needed
        # to know that; mark stats seeded directly (UPG-REST-STARVATION).
        with self._stats_lock:
            self._lang_chunk_counts = {}
            self._lang_files = {}
            self._file_chunk_counts = {}
            self._stats_seeded = True
            self._total_chunks_cache = 0
            self._purpose_chunks_cache = 0

    @property
    def all_roots(self) -> list[Path]:
        """All workspace roots: primary first, then extra roots in order."""
        return [self.workspace_root] + self._extra_roots

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_workspace(
        self, gitignore_patterns: list[str] | None = None, force: bool = False,
    ) -> tuple[int, int]:
        """Walk workspace, index all supported files. Returns (files_indexed, chunks_total).

        Three-phase pipeline:
          1. Parallel chunking   — ThreadPoolExecutor, tree-sitter releases GIL
          2. Global batch embed  — 256-chunk batches across all files (vs 64 per-file)
          3. Incremental skip    — files unchanged since last index are skipped via mtime cache

        Before indexing, chunks for files no longer in the walk set (excluded via
        .vectrignore/.gitignore, deleted, or moved out of all roots) are pruned so
        the collection never carries orphaned chunks (UPG-8.4).

        force=True ignores the mtime cache and re-chunks/re-embeds every file,
        replacing its chunks — a clean rebuild that recovers from any cache/
        collection desync without a manual cache wipe (UPG-8.6).
        """
        from integrations.workspace_detect import (
            should_index_file, get_gitignore_patterns, get_vectrignore_dirs,
            get_vectrignore_file_globs, get_vectrignore_regexes,
        )

        # Collect candidate files across all roots; each root gets its own
        # gitignore/vectrignore patterns so per-project exclusions are respected.
        all_files: list[Path] = []
        for root in self.all_roots:
            root_patterns = gitignore_patterns or get_gitignore_patterns(str(root))
            # UPG-13.3: .vectrignore file-glob entries (e.g. "*.generated.py") are
            # matched the same way as gitignore patterns — additive, on top of the
            # existing bare directory-name exclusions below.
            root_patterns = [*root_patterns, *get_vectrignore_file_globs(str(root))]
            vectrignore_dirs = get_vectrignore_dirs(str(root))
            # UPG-EXCLUDE-REGEX: `re:<pattern>` .vectrignore entries, matched
            # against each file's workspace-relative path — additive, on top
            # of the dir-name and glob exclusions above.
            vectrignore_regexes = get_vectrignore_regexes(str(root))
            all_excluded = EXCLUDED_DIRS | vectrignore_dirs
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in all_excluded and not d.startswith(".")]
                for fname in filenames:
                    fpath = Path(dirpath) / fname
                    if should_index_file(str(fpath), root_patterns, extra_excluded_dirs=vectrignore_dirs,
                                         workspace_root=str(root),
                                         extra_excluded_regexes=vectrignore_regexes):
                        all_files.append(fpath)

        should_index_paths = {str(f) for f in all_files}

        # Reconcile: drop chunks for files no longer indexable (excluded via
        # .vectrignore/.gitignore, deleted, or moved out of all roots). Without
        # this, editing .vectrignore stops *new* indexing but leaves the old
        # chunks in the collection forever. (UPG-8.4)
        mtime_cache, schema_stale = self._load_mtime_cache_with_reason()

        # UPG-PURPOSE-RESUME-HOLE: load the purpose-completion cache before
        # pruning so an orphaned/excluded file is dropped from BOTH caches in
        # the same pass its chunks are deleted from both collections —
        # mirrors _prune_orphaned_chunks' existing mtime_cache mutate-in-place.
        purpose_cache, purpose_cache_existed = self._load_purpose_cache()
        pruned = self._prune_orphaned_chunks(should_index_paths, mtime_cache, purpose_cache)

        # UPG-EMBEDDER-SWAP-GRANITE: vectors from two different embedding
        # models must never silently coexist in one collection (a same-
        # dimension model swap is exactly the silent-corruption case — cosine
        # distances between the two spaces are meaningless even though
        # ChromaDB's HNSW index accepts them without error). A stamp mismatch
        # — including a MISSING stamp, treated as a mismatch rather than a
        # match since a pre-existing index from an older vectr version
        # predates this mechanism — drops and recreates both collections
        # (ChromaDB pins dimensionality at first insert, so a different-
        # dimension model would otherwise crash the reinsert) and forces the
        # full-rebuild path: every file is re-chunked/re-embedded into the
        # fresh collections, so no old-model vector can survive.
        stored_embed_model = self._stored_embed_model()
        if stored_embed_model != self.embed_model:
            if not force:
                logger.warning(
                    "Embedding model changed (index built with %r, now configured "
                    "with %r) — forcing a full vector index rebuild so vectors "
                    "from the two models never mix in one collection",
                    stored_embed_model, self.embed_model,
                )
                force = True
            self._recreate_collections()

        # UPG-CHUNK-LOGIC-VERSION-FINGERPRINT: a cache written by an older
        # INDEXING_SCHEMA_VERSION means the chunking/embedding pipeline changed
        # since the last index — already-stored chunks no longer reflect what a
        # fresh index would produce (mirrors the symbol graph's own
        # `is_stale()` → "toolchain changed — full rebuild" pattern, see
        # agent/symbol_graph/_constants.py:graph_toolchain_fingerprint).
        # `not force` avoids a redundant second reason line when the embed-model
        # stamp above already decided this is a full rebuild.
        if schema_stale and not force:
            logger.info(
                "Content index stale — chunking/embedding logic changed since "
                "the last index — full re-chunk/re-embed"
            )
            force = True

        # Stamp the CURRENT embed model now — right after the collections are
        # known to be consistent with it — rather than deferring the write to
        # the very end of a successful run (UPG-INDEX-RESOURCE-GOVERNOR).
        # Deferring it meant a crash partway through Phase 3 on a brand-new
        # workspace left no stamp on disk; on restart `_stored_embed_model()`
        # read back None, which this same check treats as a mismatch, forcing
        # another `_recreate_collections()` that wiped the checkpointed
        # progress from the crashed run. Writing it here makes every
        # subsequent early return and mid-run crash see a correct, matching
        # stamp instead.
        self._write_embed_model_stamp()

        if force:
            # Clean rebuild: ignore the mtime cache so every file is re-indexed,
            # and (in Phase 2) its existing chunks are deleted first. (UPG-8.6)
            mtime_cache = {}

        # Incremental: split into files to index vs unchanged files to skip
        to_index: list[tuple[Path, float]] = []
        for f in all_files:
            try:
                mtime = f.stat().st_mtime
            except OSError:
                continue
            cached = mtime_cache.get(str(f))
            if cached is None or cached != mtime:
                to_index.append((f, mtime))
            else:
                self._indexed_files.add(str(f))  # already indexed — track in memory

        # UPG-PURPOSE-RESUME-HOLE: files whose body content is CURRENT (not
        # in `to_index` above — nothing below re-chunks/re-embeds them) but
        # whose purpose vectors are not confirmed complete: an interrupted
        # deferred pass (UPG-PURPOSE-PASS-DEFERRAL lost between a content
        # checkpoint and the purpose pass meant to follow it), or a first
        # run after upgrading to this cache. Detected on EVERY run,
        # including one where `to_index` ends up empty — that is the actual
        # hole this closes: previously, a workspace with nothing new to
        # content-index returned immediately below without ever revisiting
        # an old purpose gap, so it could persist forever absent --force or
        # a file touch.
        to_index_paths = {str(f) for f, _ in to_index}
        gap_paths: set[str] = set()
        if _DUAL_VECTOR_ENABLED:
            if not purpose_cache_existed and mtime_cache:
                # One-time migration seed: reconcile against the actual
                # purpose collection instead of assuming every pre-existing
                # file is a gap, which would force a corpus-wide purpose-
                # only backfill on every user's first run after upgrading.
                purpose_cache = self._seed_purpose_cache_from_collection(mtime_cache)
                self._save_purpose_cache(purpose_cache)
            gap_paths = {
                fpath for fpath, mtime in mtime_cache.items()
                if fpath not in to_index_paths
                and fpath in should_index_paths
                and purpose_cache.get(fpath) != mtime
            }
        gap_chunks = self._fetch_body_chunks_for_files(gap_paths)
        gap_mtimes = {p: mtime_cache[p] for p in gap_paths}
        self._purpose_backfill_pending_count = len(gap_paths)
        governed = _INDEX_GOVERNOR_ENABLED and not _foreground_fast_enabled()

        def _dispatch_purpose_pass(chunks: list[CodeChunk], file_mtimes: dict[str, float]) -> None:
            """Run/schedule the purpose pass (UPG-PURPOSE-PASS-DEFERRAL's
            existing sync/deferred choice) for `chunks`, then clear the
            backfill-gap counter — see `purpose_backfill_pending_files`'s
            docstring for exactly what "clear" means here (dispatch SENT,
            not necessarily landed, for the deferred case)."""
            if not chunks:
                self._purpose_backfill_pending_count = 0
                return
            defer_purpose = (
                _INDEX_GOVERNOR_DEFER_PURPOSE_PASS_ENABLED and not _foreground_fast_enabled()
            )
            if defer_purpose:
                self._schedule_deferred_purpose_pass(chunks, governed=governed, file_mtimes=file_mtimes)
            else:
                self._upsert_purpose_vectors(chunks, governed=governed, file_mtimes=file_mtimes)
            self._purpose_backfill_pending_count = 0

        if not to_index:
            if pruned:
                self._save_mtime_cache(mtime_cache)  # persist orphan removals
            # Embed-model stamp is already written above, right after the
            # mismatch check — no need to repeat it here.
            logger.info("All %d files up to date — nothing to re-index", len(all_files))
            _dispatch_purpose_pass(gap_chunks, gap_mtimes)
            self._last_indexed = time.time()
            self._refresh_chunk_count_caches()
            return len(self._indexed_files), self.total_chunks

        logger.info(
            "Indexing %d/%d files (%d unchanged, skipped)...",
            len(to_index), len(all_files), len(all_files) - len(to_index),
        )

        # Phase 1: parallel chunking
        all_chunks: list[CodeChunk] = []
        new_mtimes: dict[str, float] = {}

        def _safe_chunk(item: tuple[Path, float]) -> tuple[list[CodeChunk], str, float]:
            fpath, mtime = item
            try:
                return chunk_file(str(fpath)), str(fpath), mtime
            except Exception:
                return [], str(fpath), mtime

        with ThreadPoolExecutor(max_workers=_CHUNK_WORKERS) as pool:
            futures = {pool.submit(_safe_chunk, item): item for item in to_index}
            done = 0
            for fut in as_completed(futures):
                chunks, fpath_str, mtime = fut.result()
                seen: set[str] = set()
                for c in chunks:
                    if c.chunk_id not in seen:
                        seen.add(c.chunk_id)
                        all_chunks.append(c)
                self._indexed_files.add(fpath_str)
                new_mtimes[fpath_str] = mtime
                done += 1
                if done % 50 == 0 or done == len(to_index):
                    logger.info("  chunked %d/%d files (%d chunks so far)...",
                                done, len(to_index), len(all_chunks))

        if not all_chunks:
            # Every file in `to_index` chunked to zero chunks (e.g. all
            # whitespace/binary) — still persist their mtimes, otherwise
            # `new_mtimes` built during Phase 1 is silently discarded and
            # these files are needlessly re-chunked on every future run
            # (UPG-INDEX-RESOURCE-GOVERNOR). Zero chunks also means zero
            # purpose-vector candidates — checkpoint purpose_cache the same
            # way, otherwise these files would look like a permanent
            # purpose gap on every future run (UPG-PURPOSE-RESUME-HOLE).
            mtime_cache.update(new_mtimes)
            self._save_mtime_cache(mtime_cache)
            if _DUAL_VECTOR_ENABLED:
                purpose_cache.update(new_mtimes)
                self._save_purpose_cache(purpose_cache)
            _dispatch_purpose_pass(gap_chunks, gap_mtimes)
            self._last_indexed = time.time()
            self._refresh_chunk_count_caches()
            return len(self._indexed_files), self.total_chunks

        # Phase 2: delete stale chunks for re-indexed files (no-op for brand-new files).
        # Under force, mtime_cache is empty, so delete by file_path unconditionally
        # to avoid leaving stale chunks whose ids no longer match (UPG-8.6).
        # `_delete_chunks_for_file` handles both collections (the purpose
        # collection, ARCH-4, is keyed by the same file_path metadata) and
        # applies the incremental stats delta — same helper `delete_file()`
        # and the watcher's batched deletes use (UPG-REST-STARVATION).
        phase_start = time.time()
        for fpath_str in new_mtimes:
            if force or fpath_str in mtime_cache:  # previously indexed → delete old chunks
                self._delete_chunks_for_file(fpath_str)
        logger.info("  stale-chunk sweep done: %d files in %.0fs",
                    len(new_mtimes), time.time() - phase_start)

        # Phase 3: streaming batched embed + upsert (UPG-INDEX-MEM-STREAMING).
        # Embed in large batches (256) for BLAS efficiency, upsert each embed
        # batch immediately in smaller sub-batches (100, SQLite's 999-variable
        # limit: 6 metadata fields x 100 rows = 600) before embedding the next
        # batch. Peak memory is O(_EMBED_BATCH_SIZE) embeddings, never
        # O(corpus) — a full-corpus `all_embeddings` list was the dominant
        # memory cost on large workspaces (~4 GB of Python floats at 170k
        # chunks), and doubled while the purpose pass built its own full-
        # corpus list concurrently. ids/documents/metadatas are built per
        # batch from the `all_chunks` slice rather than once for the whole
        # corpus, for the same reason.
        total = len(all_chunks)
        phase_start = time.time()

        # UPG-INDEX-RESOURCE-GOVERNOR: checkpoint granularity is per FILE,
        # not per batch. Chunks for one file are always contiguous in
        # `all_chunks` (each Phase 1 thread-pool result above appends one
        # file's chunks as a single block), so the index of a file's LAST
        # chunk tells us exactly when every one of its chunks has been
        # embedded + upserted. A file's mtime is only persisted to the
        # on-disk cache once that point is reached — a crash before then
        # re-processes the whole file on the next run (safe: chunk ids are
        # deterministic, so re-upserting is an idempotent superset write,
        # never a duplicate or an orphan), but a partially-embedded file is
        # never left recorded as up to date.
        file_last_chunk_idx: dict[str, int] = {}
        for idx, c in enumerate(all_chunks):
            file_last_chunk_idx[c.file_path] = idx

        # Files that chunked to zero chunks have no embed work to wait for —
        # checkpoint them immediately rather than only at the very end.
        zero_chunk_mtimes = {
            fpath: mtime for fpath, mtime in new_mtimes.items()
            if fpath not in file_last_chunk_idx
        }
        if zero_chunk_mtimes:
            mtime_cache.update(zero_chunk_mtimes)
            self._save_mtime_cache(mtime_cache)
            if _DUAL_VECTOR_ENABLED:
                # Zero chunks for these files means zero purpose-vector
                # candidates — checkpoint them the same way, otherwise
                # they'd look like a permanent purpose gap on every future
                # run (UPG-PURPOSE-RESUME-HOLE). `governed` was already
                # computed above (gap-detection block), reused here.
                purpose_cache.update(zero_chunk_mtimes)
                self._save_purpose_cache(purpose_cache)

        with lowered_priority(
            governed,
            macos_qos_class=_INDEX_GOVERNOR_MACOS_QOS_CLASS,
            linux_nice_increment=_INDEX_GOVERNOR_LINUX_NICE_INCREMENT,
        ):
            for batch_num, i in enumerate(range(0, total, _EMBED_BATCH_SIZE), start=1):
                batch_start = time.time()
                batch = all_chunks[i: i + _EMBED_BATCH_SIZE]
                batch_ids = [c.chunk_id for c in batch]
                batch_docs = [c.content for c in batch]
                batch_metas = [_chunk_metadata(c) for c in batch]
                batch_embeddings = self._embed_provider.embed(batch_docs)
                _upsert_in_batches(
                    self._collection, batch_ids, batch_docs, batch_metas,
                    batch_embeddings, _UPSERT_BATCH_SIZE, op_label="upsert (body)",
                )
                self._apply_chunk_delta(batch_metas, sign=1)
                # Refresh after every batch, not just once the whole call
                # finishes: a bulk/forced reindex can run for minutes, and an
                # external caller polling `total_chunks`/`last_indexed_ts` to
                # detect "indexing is still in progress" must see them keep
                # moving throughout, not sit frozen at their pre-index values
                # until the very end and then jump straight to a value that
                # looks settled while the index is actually still incomplete.
                self._last_indexed = time.time()
                self._refresh_chunk_count_caches()

                # Checkpoint every `checkpoint_every_batches` batches:
                # persist the mtime of every file whose last chunk index is
                # now <= the highest index processed so far. Always runs —
                # a correctness guarantee, independent of `governed` below.
                # This tracks CONTENT completion only; it is intentionally
                # decoupled from purpose-vector completion (see
                # UPG-PURPOSE-PASS-DEFERRAL below) — content chunk ids and
                # purpose chunk ids are separate collections, upserted
                # idempotently by chunk_id, so a content checkpoint here
                # never claims more than "this file's body chunks are safely
                # persisted", which is true regardless of whether its
                # purpose vectors have been written yet.
                if batch_num % _INDEX_GOVERNOR_CHECKPOINT_EVERY_BATCHES == 0:
                    processed_upto = i + len(batch) - 1
                    newly_done = {
                        fpath: new_mtimes[fpath]
                        for fpath, last_idx in file_last_chunk_idx.items()
                        if last_idx <= processed_upto and fpath not in mtime_cache
                    }
                    if newly_done:
                        mtime_cache.update(newly_done)
                        self._save_mtime_cache(mtime_cache)

                if governed:
                    pace(
                        time.time() - batch_start,
                        _INDEX_GOVERNOR_DUTY_CYCLE,
                        _INDEX_GOVERNOR_MIN_BATCH_SECONDS_FOR_PACING,
                    )

                if i % (10 * _EMBED_BATCH_SIZE) == 0 and i > 0:
                    logger.info("  embedded %d/%d chunks...", i, total)
        logger.info("  content embed+upsert done: %d chunks in %.0fs",
                    total, time.time() - phase_start)

        # UPG-PURPOSE-PASS-DEFERRAL: when enabled (default) and not running
        # under --foreground-fast, the purpose pass is handed to a
        # single-worker background executor and this call returns without
        # waiting for it — search serves body-only results in the meantime
        # via `query_vector_purpose`'s existing empty-collection guard, and
        # `purpose_vectors_pending` (below) reports the window is open.
        # Disabled (or --foreground-fast), it runs synchronously exactly as
        # before this change. Either way, content completion (checkpointed
        # above) is independent of this — a crash during or after this call
        # never re-embeds already-persisted content; at worst it re-runs the
        # (idempotent, upsert-by-chunk_id) purpose pass for files whose
        # purpose vectors hadn't landed yet.
        #
        # UPG-PURPOSE-RESUME-HOLE: `gap_chunks`/`gap_mtimes` (computed above,
        # before Phase 1) fold any BACKLOG of already-content-indexed files
        # whose purpose vectors never landed into this SAME dispatch, so one
        # run closes both this run's own fresh files and any inherited gap
        # in a single pass/background job rather than two.
        _dispatch_purpose_pass(all_chunks + gap_chunks, {**new_mtimes, **gap_mtimes})

        logger.info("Indexed %d chunks from %d files", total, len(to_index))

        # Final persist: covers any file whose checkpoint above didn't land
        # on a `checkpoint_every_batches` boundary. The embed-model stamp
        # itself was already written earlier, right after the mismatch check.
        mtime_cache.update(new_mtimes)
        self._save_mtime_cache(mtime_cache)

        self._last_indexed = time.time()
        self._refresh_chunk_count_caches()
        return len(self._indexed_files), self.total_chunks

    def index_file(self, file_path: str) -> int:
        """Chunk and embed a single file. Returns number of chunks indexed."""
        chunks = chunk_file(file_path)
        if not chunks:
            return 0

        # Deduplicate by chunk_id (AST nodes on the same line range can collide in
        # minified files like jquery.min.js where many nodes share line 2-2)
        seen_ids: set[str] = set()
        deduped: list = []
        for c in chunks:
            if c.chunk_id not in seen_ids:
                seen_ids.add(c.chunk_id)
                deduped.append(c)
        chunks = deduped

        # Remove old chunks for this file before re-indexing. Calls the
        # internal helper directly rather than `delete_file()` — `index_file`
        # always re-adds `file_path` to `_indexed_files` and refreshes the
        # chunk-count caches itself a few lines below regardless of outcome,
        # so going through the public method here would only do that same
        # bookkeeping twice (and cost four `.count()` calls per file instead
        # of two).
        self._delete_chunks_for_file(file_path)

        # Streaming embed + upsert (same helper as index_workspace's Phase 3,
        # UPG-INDEX-MEM-STREAMING) — a single file's chunk count is already
        # small, but reusing the shared helper keeps the two paths' upsert
        # batching identical rather than a second hand-maintained copy.
        for i in range(0, len(chunks), _FILE_BATCH_SIZE):
            batch = chunks[i: i + _FILE_BATCH_SIZE]
            batch_ids = [c.chunk_id for c in batch]
            batch_docs = [c.content for c in batch]
            batch_metas = [_chunk_metadata(c) for c in batch]
            batch_embeddings = self._embed_provider.embed(batch_docs)
            assert len(batch_embeddings) == len(batch_ids), (
                f"Embed provider returned {len(batch_embeddings)} embeddings "
                f"for {len(batch_ids)} chunks"
            )
            _upsert_in_batches(
                self._collection, batch_ids, batch_docs, batch_metas,
                batch_embeddings, _FILE_BATCH_SIZE, op_label="upsert (body)",
            )
            self._apply_chunk_delta(batch_metas, sign=1)

        self._upsert_purpose_vectors(chunks)

        self._indexed_files.add(file_path)
        self._refresh_chunk_count_caches()
        return len(chunks)

    def delete_file(self, file_path: str) -> int:
        """Remove all chunks belonging to a file (body + purpose collections).

        Returns the number of body-collection chunks removed — used by the
        watcher's batch worker for its per-batch churn diagnostic
        (UPG-WATCH-REVERT-CHURN)."""
        removed = self._delete_chunks_for_file(file_path)
        self._indexed_files.discard(file_path)
        self._refresh_chunk_count_caches()
        return removed

    def _delete_chunks_for_file(self, file_path: str) -> int:
        """Remove chunks for `file_path` from both collections and apply the
        incremental stats delta (UPG-REST-STARVATION). Internal: unlike
        `delete_file()`, does not touch `_indexed_files` — used by
        `index_workspace`'s stale-chunk sweep, where the file is about to be
        immediately re-indexed within the same call, not removed from the
        index. Returns the number of body-collection chunks removed.
        """
        removed = 0
        try:
            with _timed_chroma_call("get"):
                existing = self._collection.get(where={"file_path": file_path})
            ids = existing["ids"]
            if ids:
                with _timed_chroma_call("delete"):
                    self._collection.delete(ids=ids)
                self._apply_chunk_delta(existing.get("metadatas") or [], sign=-1)
                removed = len(ids)
        except Exception:
            pass
        try:
            with _timed_chroma_call("get"):
                existing_p = self._purpose_collection.get(where={"file_path": file_path})
            if existing_p["ids"]:
                with _timed_chroma_call("delete"):
                    self._purpose_collection.delete(ids=existing_p["ids"])
        except Exception:
            pass
        return removed

    # ------------------------------------------------------------------
    # Purpose vectors (ARCH-4 dual-vector pool entry)
    # ------------------------------------------------------------------

    def _upsert_purpose_vectors(
        self,
        chunks: list[CodeChunk],
        governed: bool = False,
        file_mtimes: dict[str, float] | None = None,
    ) -> None:
        """Embed + upsert the body-stripped purpose text for symbol chunks.

        Skipped entirely when `DUAL_VECTOR_ENABLED` is False (config, default
        on) — reduces to the pre-ARCH-4 body-only index. Non-symbol chunks
        (`build_purpose_text` returns None) are not written — the purpose
        collection stays a strict subset of the body collection's ids.

        Streaming, same shape as index_workspace's Phase 3 (UPG-INDEX-MEM-
        STREAMING): `build_purpose_text` is called per _EMBED_BATCH_SIZE
        batch of `chunks` (not once up front for the whole corpus), and each
        batch's embeddings are upserted immediately, keeping peak memory
        O(batch). Previously this ran while the body pass's own full-corpus
        `all_embeddings`/`documents`/`metadatas` were still live in the
        caller's frame — the two full-corpus embedding lists held concurrently
        were the dominant swap driver on large workspaces.

        `governed=True` applies the same priority clamp + duty-cycle pacing
        as Phase 3 (UPG-INDEX-RESOURCE-GOVERNOR) — only `index_workspace`'s
        bulk call (sync or deferred, see `_schedule_deferred_purpose_pass`)
        opts in. `index_file`'s single-file call leaves this False: that
        path is the interactive file-save watcher, already small (well
        under one embed batch), and latency-sensitive — a saved file
        becoming searchable should never wait on a governor sleep.

        Content (mtime_cache) and purpose-vector completion are deliberately
        DECOUPLED signals (UPG-PURPOSE-PASS-DEFERRAL) — this method does not
        touch mtime_cache at all. A file's chunk ids are upserted here
        idempotently keyed by chunk_id, so a crash before this call runs (or
        mid-way through it, whether inline or on the deferred background
        thread) simply leaves that file's purpose vectors missing/partial.

        `file_mtimes` (UPG-PURPOSE-RESUME-HOLE), when given, is a
        file_path -> content-mtime map covering every file contributing
        chunks to this call. As each file's LAST chunk in `chunks` is
        reached — the same "last occurrence index" trick index_workspace's
        Phase 3 uses for mtime_cache; correct regardless of whether a
        file's chunks are contiguous, which they are NOT for the
        purpose-only backfill path, since `_fetch_body_chunks_for_files`
        batches its `$in` query across several files at once — that file's
        entry is written to the on-disk purpose-completion cache
        (`purpose_cache.json`) and persisted immediately, not only at the
        very end: a crash mid-pass loses at most the batch in flight, never
        the whole call. This is what makes a retry genuinely SELF-HEALING,
        not merely safe: a plain idempotent retry only stops corruption on
        replay, but without a persisted per-file completion signal a gap
        could never be *detected* on a later run at all (mtime_cache alone
        cannot distinguish "content indexed, purpose written" from "content
        indexed, purpose pass never got that far"), so nothing would ever
        prompt the retry to happen in the first place. With this cache, the
        NEXT ordinary `index_workspace()` call — no `--force` and no file
        touch required — finds a strictly smaller remaining gap, never the
        same one repeated; see `index_workspace`'s own gap-detection block
        for how that next call rediscovers it.
        """
        if file_mtimes and not _DUAL_VECTOR_ENABLED:
            # No purpose collection is being maintained at all — leave
            # purpose_cache untouched (not "complete") so a later run with
            # dual-vector re-enabled treats these as unseen rather than
            # wrongly skipping them; see _seed_purpose_cache_from_collection.
            return
        if not chunks:
            if file_mtimes:
                purpose_cache, _ = self._load_purpose_cache()
                purpose_cache.update(file_mtimes)
                self._save_purpose_cache(purpose_cache)
            return
        if not _DUAL_VECTOR_ENABLED:
            return
        total = sum(1 for c in chunks if is_symbol_bearing_chunk(c.symbol_name, c.node_type))
        if total == 0:
            # Zero symbol-bearing chunks in this call is still a fully
            # "considered" outcome for every file contributing to `chunks`
            # — checkpoint them rather than leaving them a permanent gap
            # (e.g. a file with no functions/classes/etc, UPG-PURPOSE-
            # RESUME-HOLE).
            if file_mtimes:
                purpose_cache, _ = self._load_purpose_cache()
                purpose_cache.update(file_mtimes)
                self._save_purpose_cache(purpose_cache)
            return

        # UPG-PURPOSE-RESUME-HOLE: last-occurrence index per file, mirroring
        # index_workspace Phase 3's file_last_chunk_idx checkpoint trick —
        # tells us exactly when every chunk belonging to a file has been
        # considered by the loop below (embedded + upserted, or skipped for
        # not being symbol-bearing).
        file_last_chunk_idx: dict[str, int] = {}
        purpose_cache: dict[str, float] = {}
        if file_mtimes:
            for idx, c in enumerate(chunks):
                file_last_chunk_idx[c.file_path] = idx
            purpose_cache, _ = self._load_purpose_cache()

        phase_start = time.time()
        done = 0
        with lowered_priority(
            governed,
            macos_qos_class=_INDEX_GOVERNOR_MACOS_QOS_CLASS,
            linux_nice_increment=_INDEX_GOVERNOR_LINUX_NICE_INCREMENT,
        ):
            for i in range(0, len(chunks), _EMBED_BATCH_SIZE):
                batch_start = time.time()
                batch = chunks[i: i + _EMBED_BATCH_SIZE]
                batch_ids: list[str] = []
                batch_docs: list[str] = []
                batch_metas: list[dict] = []
                for c in batch:
                    purpose_text = build_purpose_text(c.content, c.symbol_name, c.node_type, c.language)
                    if purpose_text is None:
                        continue
                    batch_ids.append(c.chunk_id)
                    batch_docs.append(purpose_text)
                    batch_metas.append(_chunk_metadata(c))
                if batch_ids:
                    batch_embeddings = self._embed_provider.embed(batch_docs)
                    _upsert_in_batches(
                        self._purpose_collection, batch_ids, batch_docs, batch_metas,
                        batch_embeddings, _UPSERT_BATCH_SIZE, op_label="upsert (purpose)",
                    )
                    done += len(batch_ids)

                # Checkpoint purpose completion the same cadence Phase 3
                # checkpoints content: every `checkpoint_every_batches`
                # batches, and unconditionally on the last one — so a crash
                # mid-pass never loses more than that many batches' worth of
                # already-completed files (UPG-PURPOSE-RESUME-HOLE).
                batch_num = i // _EMBED_BATCH_SIZE + 1
                is_last_batch = i + len(batch) >= len(chunks)
                if file_mtimes and (
                    batch_num % _INDEX_GOVERNOR_CHECKPOINT_EVERY_BATCHES == 0 or is_last_batch
                ):
                    processed_upto = i + len(batch) - 1
                    newly_done = {
                        fpath: file_mtimes[fpath]
                        for fpath, last_idx in file_last_chunk_idx.items()
                        if fpath in file_mtimes
                        and last_idx <= processed_upto
                        and purpose_cache.get(fpath) != file_mtimes[fpath]
                    }
                    if newly_done:
                        purpose_cache.update(newly_done)
                        self._save_purpose_cache(purpose_cache)

                if governed:
                    pace(
                        time.time() - batch_start,
                        _INDEX_GOVERNOR_DUTY_CYCLE,
                        _INDEX_GOVERNOR_MIN_BATCH_SECONDS_FOR_PACING,
                    )
                if i % (10 * _EMBED_BATCH_SIZE) == 0 and i > 0:
                    logger.info("  purpose-embedded %d/%d symbol chunks...", done, total)
        logger.info("  purpose embed+upsert done: %d symbol chunks in %.0fs",
                    total, time.time() - phase_start)

    def _schedule_deferred_purpose_pass(
        self,
        chunks: list[CodeChunk],
        governed: bool,
        file_mtimes: dict[str, float] | None = None,
    ) -> None:
        """UPG-PURPOSE-PASS-DEFERRAL: hand the purpose-vector pass to
        `_purpose_executor` (max_workers=1, see __init__) and return
        immediately — `index_workspace()` does not wait for it. `governed`
        is threaded through from the caller's own computed value (same
        priority clamp + duty-cycle pacing decision `index_workspace`'s
        Phase 3 made, not hardcoded) so `--foreground-fast`/the governor
        config toggle apply uniformly whether the pass ends up running
        inline or on this background thread. `purpose_vectors_pending`
        reads True from the moment this is submitted until the closure's
        `finally` clears it, whether the pass succeeds, raises, or the
        process is killed mid-pass.

        `file_mtimes` (UPG-PURPOSE-RESUME-HOLE) is threaded straight through
        to `_upsert_purpose_vectors`, unchanged — its own per-batch
        checkpointing runs exactly the same way on this background thread
        as it would inline, so a process kill mid-deferred-pass loses at
        most the batch in flight, not the whole call: the next ordinary
        `index_workspace()` re-detects and re-dispatches only the files
        that never got checkpointed, never the same gap it already closed
        (this is the crash this mechanism specifically targets — see
        UPG-PURPOSE-RESUME-HOLE's investigation, where a deferred pass lost
        between a content checkpoint and its own end-of-run purpose pass
        left the gap permanent under the pre-fix code)."""
        with self._purpose_pending_lock:
            self._purpose_pending_count += 1

        def _run() -> None:
            try:
                self._upsert_purpose_vectors(chunks, governed=governed, file_mtimes=file_mtimes)
            except Exception:
                # Best-effort background pass: never let a failure here
                # surface anywhere but the log. Purpose vectors for the
                # affected files simply stay unwritten; content is already
                # safely checkpointed (decoupled, see _upsert_purpose_vectors)
                # so nothing here is silently lost or double-counted.
                logger.exception("Deferred purpose-vector pass failed")
            finally:
                with self._purpose_pending_lock:
                    self._purpose_pending_count -= 1

        self._purpose_executor.submit(_run)

    # ------------------------------------------------------------------
    # mtime cache — tracks file modification times for incremental indexing
    # ------------------------------------------------------------------

    def _mtime_cache_path(self) -> Path:
        # Co-locate the mtime cache with the ChromaDB dir so the two always clear
        # together. Previously this hardcoded ~/.cache/vectr/db/<hash> regardless
        # of the db_path override, so when the collection lived elsewhere (e.g. the
        # service's ~/.cache/vectr/<hash>/chroma) the cache could desync — a stale
        # cache then reported "nothing to re-index" against an empty collection,
        # leaving 0 chunks. (UPG-8.5)
        return self._db_dir / "index_cache.json"

    def _load_mtime_cache(self) -> dict[str, float]:
        cache, _schema_stale = self._load_mtime_cache_with_reason()
        return cache

    def _load_mtime_cache_with_reason(self) -> tuple[dict[str, float], bool]:
        """Load the mtime cache, plus whether it was invalidated specifically
        because INDEXING_SCHEMA_VERSION changed (as opposed to no cache file
        existing at all — a genuinely fresh workspace, where there is nothing
        to compare against and no "changed since last index" reason to report).

        `index_workspace()` uses the second element to (a) force=True the same
        way an embed-model-stamp mismatch does, so Phase 2's stale-chunk sweep
        actually deletes each file's previously-stored chunks instead of
        silently leaving them alongside freshly re-chunked ones (a schema
        mismatch that only cleared the incremental-skip dict, without also
        forcing the delete-before-reinsert step, could accumulate orphaned
        chunks whose ids no longer match anything a fresh chunker would
        produce), and (b) log one clear reason line, matching the symbol
        graph's is_stale() → "toolchain changed" message style.
        """
        path = self._mtime_cache_path()
        try:
            cache = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}, False
        # Schema-version gate: a cache written by an older INDEXING_SCHEMA_VERSION
        # (e.g. before ARCH-4's purpose vector existed) is treated as a cold
        # cache — every file falls into `to_index` on the next index_workspace()
        # and is fully re-chunked/re-embedded, the same recovery path force=True
        # uses. This is what makes a pipeline change (new derived vector, new
        # chunk content) reach an already-indexed workspace without a manual
        # cache wipe. A cache with no version key at all (pre-ARCH-4, unversioned)
        # is also treated as stale.
        stored_version = cache.pop(_MTIME_CACHE_SCHEMA_KEY, None)
        if stored_version != INDEXING_SCHEMA_VERSION:
            return {}, True
        return cache, False

    def _save_mtime_cache(self, cache: dict[str, float]) -> None:
        with self._mtime_cache_lock:
            path = self._mtime_cache_path()
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = dict(cache)
                payload[_MTIME_CACHE_SCHEMA_KEY] = INDEXING_SCHEMA_VERSION
                path.write_text(json.dumps(payload), encoding="utf-8")
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Purpose-vector completion cache (UPG-PURPOSE-RESUME-HOLE) — see
    # _PURPOSE_CACHE_FILE's docstring for why this is a separate file/cache
    # from the mtime cache rather than another key inside it.
    # ------------------------------------------------------------------

    def _purpose_cache_path(self) -> Path:
        return self._db_dir / _PURPOSE_CACHE_FILE

    def _load_purpose_cache(self) -> tuple[dict[str, float], bool]:
        """Load the purpose-completion cache: file_path -> the content mtime
        as of which that file's purpose vectors are confirmed fully written.

        Returns `(cache, existed)`. `existed=False` covers both a missing
        file and a schema-version mismatch (same treatment as
        `_load_mtime_cache_with_reason`) — `index_workspace()` uses it to
        tell "this workspace has never run the purpose-completion tracker
        before" (triggering a one-time reconciliation seed against the
        actual purpose collection, `_seed_purpose_cache_from_collection`)
        apart from "this file just doesn't have an entry yet" (an ordinary,
        cheap incremental gap once the cache exists) — without that
        distinction, upgrading to this mechanism would make every
        already-healthy file in an existing large workspace look like a
        purpose gap and force a needless corpus-wide purpose-only reindex.
        """
        path = self._purpose_cache_path()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}, False
        stored_version = raw.pop(_PURPOSE_CACHE_SCHEMA_KEY, None)
        if stored_version != INDEXING_SCHEMA_VERSION:
            return {}, False
        return raw, True

    def _save_purpose_cache(self, cache: dict[str, float]) -> None:
        with self._purpose_cache_lock:
            path = self._purpose_cache_path()
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = dict(cache)
                payload[_PURPOSE_CACHE_SCHEMA_KEY] = INDEXING_SCHEMA_VERSION
                path.write_text(json.dumps(payload), encoding="utf-8")
            except OSError:
                pass

    def _seed_purpose_cache_from_collection(
        self, mtime_cache: dict[str, float],
    ) -> dict[str, float]:
        """One-time migration seed (UPG-PURPOSE-RESUME-HOLE), called only
        when `_load_purpose_cache()` reports no usable cache exists yet — a
        pre-this-mechanism workspace, or one whose cache went stale via an
        INDEXING_SCHEMA_VERSION bump. Marks every file that already has at
        least one chunk in the purpose collection as complete, from a single
        paginated metadata-only scan of `self._purpose_collection` (the same
        shape/cost as `_collection_file_paths()`'s existing per-run scan of
        the body collection for orphan pruning — paid once here, not on
        every subsequent run, since after this call the cache file exists).

        Files with NO purpose-collection presence are deliberately left
        unseeded — both real cases (a file affected by the historic
        deferred-pass-lost-on-crash gap, and a file that legitimately has
        zero symbol-bearing chunks, e.g. pure markdown) fall through to
        `index_workspace()`'s normal gap detection on this same run, and
        `_upsert_purpose_vectors`'s own per-file checkpoint resolves both
        correctly the first time it processes them — a file-presence check
        alone cannot tell those two cases apart, but processing them once
        settles it either way, permanently.
        """
        if not mtime_cache:
            return {}
        files_with_purpose = self._collection_file_paths(self._purpose_collection)
        return {
            fpath: mtime for fpath, mtime in mtime_cache.items()
            if fpath in files_with_purpose
        }

    def _fetch_body_chunks_for_files(self, file_paths: set[str]) -> list[CodeChunk]:
        """Reconstruct `CodeChunk`s for `file_paths` from their already-
        indexed body-collection rows (UPG-PURPOSE-RESUME-HOLE), for the
        purpose-only backfill path: these files' body content is current
        (they are staying out of this run's `to_index`), so no re-chunking
        (tree-sitter) is needed — the body collection already holds exactly
        the chunk id/content/metadata `_upsert_purpose_vectors` needs to
        build purpose text. Batches the `$in` where-clause over
        `_UPSERT_BATCH_SIZE`-sized groups of paths (SQLite's 999-variable
        limit) and paginates each group's own result set the same way
        `_collection_file_paths` does. Rows across different files' batches
        are not guaranteed contiguous in the returned list, but
        `_upsert_purpose_vectors`'s checkpoint only needs each file's LAST
        occurrence index, which is correct regardless of interleaving — see
        that method's checkpoint comment.
        """
        if not file_paths:
            return []
        chunks: list[CodeChunk] = []
        paths = list(file_paths)
        for i in range(0, len(paths), _UPSERT_BATCH_SIZE):
            batch_paths = paths[i:i + _UPSERT_BATCH_SIZE]
            offset = 0
            while True:
                with _timed_chroma_call("get"):
                    page = self._collection.get(
                        where={"file_path": {"$in": batch_paths}},
                        include=["documents", "metadatas"],
                        limit=1000,
                        offset=offset,
                    )
                ids = page["ids"]
                if not ids:
                    break
                for cid, doc, meta in zip(ids, page["documents"], page["metadatas"]):
                    chunks.append(CodeChunk(
                        chunk_id=cid,
                        content=doc,
                        file_path=meta.get("file_path", ""),
                        language=meta.get("language", ""),
                        node_type=meta.get("node_type", ""),
                        start_line=meta.get("start_line", 0),
                        end_line=meta.get("end_line", 0),
                        symbol_name=meta.get("symbol_name", ""),
                    ))
                offset += len(ids)
                if len(ids) < 1000:
                    break
        return chunks

    # ------------------------------------------------------------------
    # Embedding-model version stamp (UPG-EMBEDDER-SWAP-GRANITE) — see
    # _EMBED_MODEL_STAMP_FILE's docstring for why this is a separate file
    # from the mtime cache rather than another key inside it.
    # ------------------------------------------------------------------

    def _embed_model_stamp_path(self) -> Path:
        return self._db_dir / _EMBED_MODEL_STAMP_FILE

    def _stored_embed_model(self) -> str | None:
        """The embed_model that built the collection's CURRENT vectors, or
        None if no stamp exists yet (fresh index, or a pre-existing index
        from a vectr version that predates this mechanism — callers must
        treat None as a mismatch, not a match)."""
        path = self._embed_model_stamp_path()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        model = data.get("embed_model")
        return str(model) if model else None

    def _write_embed_model_stamp(self) -> None:
        path = self._embed_model_stamp_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"embed_model": self.embed_model}), encoding="utf-8")
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Orphan pruning — reconcile the collection against the current walk
    # ------------------------------------------------------------------

    def _collection_file_paths(self, collection=None) -> set[str]:
        """Distinct file_path values currently stored in `collection`
        (default: the body collection — every pre-existing call site relies
        on this default). Metadata-only paginated scan (no documents/
        embeddings loaded). Also used against `self._purpose_collection`
        by `_seed_purpose_cache_from_collection` (UPG-PURPOSE-RESUME-HOLE)
        — same shape, so one paginated-scan implementation serves both
        collections rather than a second hand-maintained copy."""
        if collection is None:
            collection = self._collection
        _PAGE = 1000
        paths: set[str] = set()
        offset = 0
        while True:
            with _timed_chroma_call("get"):
                page = collection.get(include=["metadatas"], limit=_PAGE, offset=offset)
            ids = page["ids"]
            if not ids:
                break
            for meta in page["metadatas"]:
                fp = meta.get("file_path")
                if fp:
                    paths.add(fp)
            offset += len(ids)
            if len(ids) < _PAGE:
                break
        return paths

    def _prune_orphaned_chunks(
        self,
        should_index_paths: set[str],
        mtime_cache: dict[str, float],
        purpose_cache: dict[str, float] | None = None,
    ) -> int:
        """Delete chunks for indexed files no longer in the walk set. Returns count.

        Files become orphaned when they're newly excluded (.vectrignore/.gitignore),
        deleted, or moved out of all roots. `mtime_cache` is mutated in place so the
        pruned entries don't linger and re-trigger a phantom skip. (UPG-8.4)

        `purpose_cache` (UPG-PURPOSE-RESUME-HOLE), when given, is pruned the
        same way — an orphaned file's stale purpose-completion entry is
        harmless either way (gap detection already excludes anything not in
        `should_index_paths`), but leaving it would let the cache grow
        unboundedly for a workspace whose files churn over time.
        """
        try:
            indexed_paths = self._collection_file_paths()
        except Exception:
            return 0
        orphaned = indexed_paths - should_index_paths
        for path in orphaned:
            self.delete_file(path)
            mtime_cache.pop(path, None)
            if purpose_cache is not None:
                purpose_cache.pop(path, None)
        if orphaned:
            logger.info("Pruned chunks for %d orphaned/excluded file(s)", len(orphaned))
        return len(orphaned)

    # ------------------------------------------------------------------
    # Stats — incrementally maintained, never a full-collection rescan on
    # the hot path (UPG-REST-STARVATION). See `_apply_chunk_delta` and
    # `_ensure_stats_seeded` for the mechanism.
    # ------------------------------------------------------------------

    def _ensure_stats_seeded(self) -> None:
        """One-time full metadata scan seeding `_lang_chunk_counts`/
        `_lang_files`, run at most once per collection generation
        (`_recreate_collections` re-arms it without a rescan, since a
        freshly (re)created collection is known-empty). Every insert/delete
        after this updates the counters directly via `_apply_chunk_delta`,
        so `indexed_language_stats()` never re-scans the collection again on
        a later call — this is what previously made a `/v1/status` call
        during a bulk reindex pay for a full paginated metadata scan of
        `self._collection` on every single call (the old cache was keyed on
        chunk count, which was still changing on every watcher write),
        directly contending with the collection the watcher was writing to.

        Now also run eagerly once at the end of `CodeIndexer.__init__`
        (always off the request event loop — construction either happens
        synchronously before the daemon starts serving requests, or on its
        own background thread), so this scan is never triggered by the
        first live request to reach `indexed_language_stats()`. This method
        staying idempotent and safe to call redundantly (the `_stats_seeded`
        guard below) is what makes that eager call harmless.

        `total_chunks`/`total_purpose_chunks` use a SEPARATE in-memory
        counter (`_total_chunks_cache`/`_purpose_chunks_cache`), not this
        cache — see `total_chunks`'s docstring.
        """
        if self._stats_seeded:
            return
        with self._stats_lock:
            if self._stats_seeded:  # re-check inside the lock
                return
            _PAGE = 1000
            chunks: dict[str, int] = {}
            files: dict[str, set[str]] = {}
            file_counts: dict[str, int] = {}
            offset = 0
            while True:
                with _timed_chroma_call("get"):
                    page = self._collection.get(include=["metadatas"], limit=_PAGE, offset=offset)
                ids = page["ids"]
                if not ids:
                    break
                for meta in page["metadatas"]:
                    lang = meta.get("language")
                    if not lang:
                        continue
                    chunks[lang] = chunks.get(lang, 0) + 1
                    fp = meta.get("file_path")
                    if fp:
                        files.setdefault(lang, set()).add(fp)
                        file_counts[fp] = file_counts.get(fp, 0) + 1
                offset += len(ids)
                if len(ids) < _PAGE:
                    break
            self._lang_chunk_counts = chunks
            self._lang_files = files
            self._file_chunk_counts = file_counts
            self._stats_seeded = True

    def _apply_chunk_delta(self, metadatas: list[dict], sign: int) -> None:
        """Incrementally update the in-memory per-language stats cache from a
        batch of chunk metadata just inserted (`sign=1`) or removed
        (`sign=-1`) from the body collection. Held only around O(1) dict
        arithmetic — never around a ChromaDB or embedding call — so this
        lock is never the thing a request handler waits on
        (UPG-REST-STARVATION).

        On removal, a file path is dropped from its `_lang_files` bucket only
        when its LAST body chunk goes (tracked via `_file_chunk_counts`),
        not on any deletion batch that merely touches the file
        (UPG-CHUNK-DELTA-PARTIAL-FILE-DISCARD). Today every delete site is
        whole-file (`_delete_chunks_for_file`, the only caller with
        `sign=-1`, deletes exactly the `where={"file_path": ...}` result set),
        so the two behaviours coincide — but the moment any partial-chunk
        deletion path exists (incremental re-chunk, per-symbol invalidation,
        quality-based eviction), discarding on first touch would silently
        undercount `indexed_file_paths`/`indexed_language_stats()["files"]`
        while `total_chunks` stayed right.
        """
        if not metadatas:
            return
        with self._stats_lock:
            if sign > 0:
                for meta in metadatas:
                    lang = meta.get("language")
                    if not lang:
                        continue
                    self._lang_chunk_counts[lang] = (
                        self._lang_chunk_counts.get(lang, 0) + 1
                    )
                    fp = meta.get("file_path")
                    if fp:
                        self._lang_files.setdefault(lang, set()).add(fp)
                        self._file_chunk_counts[fp] = (
                            self._file_chunk_counts.get(fp, 0) + 1
                        )
            else:
                removed_per_file: dict[str, int] = {}
                for meta in metadatas:
                    lang = meta.get("language")
                    if not lang:
                        continue
                    self._lang_chunk_counts[lang] = max(
                        0, self._lang_chunk_counts.get(lang, 0) - 1
                    )
                    fp = meta.get("file_path")
                    if fp:
                        removed_per_file[fp] = removed_per_file.get(fp, 0) + 1
                # A file's chunks all share one language, so scanning every
                # bucket's set here costs O(languages) — trivial next to the
                # per-chunk loop above, and immune to any historical metadata
                # having filed the path under an unexpected bucket.
                for fp, n_removed in removed_per_file.items():
                    remaining = self._file_chunk_counts.get(fp, 0) - n_removed
                    if remaining > 0:
                        self._file_chunk_counts[fp] = remaining
                    else:
                        self._file_chunk_counts.pop(fp, None)
                        for files in self._lang_files.values():
                            files.discard(fp)

    def _refresh_chunk_count_caches(self) -> None:
        """Refresh `total_chunks`/`total_purpose_chunks` from a real, one-off
        ChromaDB `.count()` call on each collection. Called at the end of
        every mutation entry point — `index_file`, `delete_file`,
        `index_workspace` — and, inside `index_workspace`'s own bulk embed/
        upsert loop, once per batch as well, so an external caller watching
        these counters for indexing progress never sees them frozen for the
        full duration of a large reindex. Always runs on whatever thread
        performed the mutation (the indexing/watcher thread, or a request's
        off-loop dispatch-executor thread — see agent/chroma_dispatch.py) —
        never on a request's own event-loop thread, since it is only ever
        reached from inside one of those methods.

        A real re-count (rather than incremental delta bookkeeping) is what
        makes this self-correcting: it is exactly as accurate as a live call
        would have been for anything that mutated the collection through one
        of these three methods, and it does not need special-case handling
        for a collection changed some other way (a test simulating a desync,
        an external tool) — the next call through any of the three methods
        re-syncs it regardless of how it went stale.
        """
        with _timed_chroma_call("count"):
            total = self._collection.count()
        with _timed_chroma_call("count"):
            purpose_total = self._purpose_collection.count()
        with self._stats_lock:
            self._total_chunks_cache = total
            self._purpose_chunks_cache = purpose_total

    @property
    def total_chunks(self) -> int:
        """Total chunk count in the body collection.

        Returns an in-memory counter, refreshed from a real ChromaDB count at
        construction and again at the end of every mutation entry point
        (`index_file`/`delete_file`/`index_workspace`, via
        `_refresh_chunk_count_caches`) — never a live call into the vector
        store made from this property itself. This trades bounded staleness
        (at most one in-flight mutation call, applied the moment that call
        finishes) for availability: a caller reading this must never wait on
        the vector store's own internal state, including while it is busy
        compacting a large collection — a single status read blocking behind
        exactly that once made every other request the daemon was serving
        unreachable.
        """
        with self._stats_lock:
            return self._total_chunks_cache

    @property
    def total_purpose_chunks(self) -> int:
        """Purpose-collection counterpart of `total_chunks` (the dual-vector
        purpose collection's own pool-entry point) — same in-memory-counter
        treatment, for reporting only. `query_vector_purpose`'s own
        empty-collection guard deliberately reads a live count instead of
        this cache — see that method's comment for why."""
        with self._stats_lock:
            return self._purpose_chunks_cache

    @property
    def purpose_vectors_pending(self) -> bool:
        """True while a deferred purpose-vector pass (UPG-PURPOSE-PASS-
        DEFERRAL) has been scheduled and has not yet finished — surfaced via
        `/v1/status` so a search-quality dip during the window (results
        ranked on body similarity alone until purpose vectors backfill) is
        explainable rather than silent. False whenever nothing is pending,
        including when deferral is disabled (the pass already completed
        synchronously before index_workspace() returned)."""
        with self._purpose_pending_lock:
            return self._purpose_pending_count > 0

    @property
    def purpose_backfill_pending_files(self) -> int:
        """Count of files `index_workspace()` most recently found to have
        current body content but incomplete purpose vectors, as of the start
        of that run (UPG-PURPOSE-RESUME-HOLE) — an interrupted deferred
        pass, or a first run after upgrading to this mechanism. Distinct
        from `purpose_vectors_pending` (which reports whether a pass is
        *currently running*, of any kind): this reports the *size of the gap
        last detected*, so `/v1/status` can distinguish "nothing is wrong"
        from "N files are still waiting on a backfill that hasn't started
        yet" instead of only "something is in flight right now". 0 once a
        run's own dispatch has been sent for backfill (whether or not that
        backfill has actually landed for the deferred/background case —
        `purpose_vectors_pending` covers that window)."""
        return self._purpose_backfill_pending_count

    @property
    def last_indexed_ts(self) -> float:
        return self._last_indexed

    @property
    def indexed_file_count(self) -> int:
        """Distinct files present in the body collection (UPG-STATUS-WARMSTART-FILES).

        Derived from the same seeded / incrementally-maintained per-language
        metadata cache `indexed_language_stats()` reads — actual chunk
        metadata, i.e. exactly what `total_chunks` counts — so a daemon
        restart over an already-indexed workspace reports the persisted file
        count immediately instead of `indexed_files: 0` sitting next to an
        already-populated `total_chunks` on `/v1/status`. That mismatch was
        the old behaviour: this used to read `_indexed_files`, the set of
        files THIS PROCESS has walked, which stays empty until some in-process
        index run touches it (the wart once forced the benchmark runners to
        refuse scoring on `indexed_files==0`). `_recreate_collections()`
        resets the cache together with `_total_chunks_cache`, so both counters
        also fall to 0 and climb together through a forced rebuild.
        `_indexed_files` itself keeps its separate, narrower meaning — files
        walked this process — feeding only `index_workspace()`'s return value;
        `indexed_file_paths` below reads this same metadata cache, so the two
        properties can no longer disagree across a warm-start window
        (UPG-WARMSTART-FINGERPRINT-STARVATION).
        """
        self._ensure_stats_seeded()
        with self._stats_lock:
            return sum(len(files) for files in self._lang_files.values())

    @property
    def indexed_file_paths(self) -> list[str]:
        """All file paths currently present in the body collection.

        Derived from the SAME seeded / incrementally-maintained per-language
        metadata cache `indexed_file_count` reads — actual chunk metadata, i.e.
        exactly the corpus that property and `total_chunks` describe. It
        previously returned `_indexed_files`, the set of files THIS PROCESS has
        walked, which stays empty across a daemon restart's warm-start window
        until some in-process index run touches it (UPG-WARMSTART-FINGERPRINT-
        STARVATION). Every consumer silently degraded over that window: the
        workspace fingerprint behind `/v1/status`'s strategy rationale was
        computed from zero files, and vectr_map rendered its passport import
        graph over an empty file list — silently, because a fingerprint over an
        empty path set still returns a well-formed answer indistinguishable
        from a genuinely fileless workspace. Metadata-derived, consumers see
        the persisted corpus immediately at construction (`_ensure_stats_seeded`
        runs eagerly in `__init__`), and the property now honours its own
        documented "currently in the index" contract.
        """
        self._ensure_stats_seeded()
        with self._stats_lock:
            return [fp for files in self._lang_files.values() for fp in files]

    def indexed_language_stats(self) -> dict[str, dict[str, int]]:
        """Per-language coverage in the collection: `{lang: {"files", "chunks"}}`.

        The ground truth for what the index actually contains — derived from
        chunk metadata, not a fixed allow-list. Reads the incrementally-
        maintained cache (`_ensure_stats_seeded` / `_apply_chunk_delta`) —
        O(languages present), never a re-scan of the collection, so this stays
        cheap to call even while a bulk reindex is concurrently writing to the
        same collection (UPG-REST-STARVATION). UPG-3.1 (which languages exist)
        and UPG-3.3 (per-language coverage + symbol availability) both read
        from this single source.
        """
        self._ensure_stats_seeded()
        with self._stats_lock:
            return {
                lang: {"files": len(self._lang_files.get(lang, ())), "chunks": n}
                for lang, n in self._lang_chunk_counts.items()
            }

    def indexed_languages(self) -> list[str]:
        """Distinct, sorted `language` values currently in the collection (UPG-3.1)."""
        return sorted(self.indexed_language_stats())

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search-query string. Public method for external callers.

        Uses the embed provider's query mode (`embed_query`), not `embed` — for
        asymmetric models (the default arctic-embed) this applies the model's
        registered query prompt, which is required for correct dense retrieval and
        must never be applied on the document/indexing side.
        """
        return self._embed_provider.embed_query([text])[0]

    def embed_query_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of search-query strings using the provider's query mode.

        For working-memory recall, where a query is matched against previously
        stored note embeddings — same query/document asymmetry as code search.
        """
        return self._embed_provider.embed_query(texts)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts (document/indexing side) using the configured
        embed provider. Never applies a query prompt — for chunk/note content being
        stored, not for search queries."""
        return self._embed_provider.embed(texts)

    @property
    def chroma_client(self):
        """The underlying ChromaDB client — shared with the working memory collection."""
        return self._client

    def close(self) -> None:
        """Release the underlying ChromaDB client's resources (its System —
        connections, and the native worker-thread pool ChromaDB's Rust
        bindings start per client). `chromadb`'s own `.close()` is already
        idempotent, so calling this more than once (or after `__del__` has
        already run it) is safe.

        Every `CodeIndexer` — hence every real `VectrService` — otherwise
        leaks a fixed-size native thread pool for the life of the process:
        confirmed empirically, a fresh `chromadb.PersistentClient()` left
        unclosed adds ~13 OS threads that `threading.active_count()` never
        sees (native, not Python-managed) and that never get reclaimed.
        Harmless for a single short-lived process, but a real leak for any
        long-running process — or test suite — that constructs many
        `CodeIndexer` instances over its lifetime. `VectrService.shutdown()`
        calls this; `__del__` below is a safety net for callers (tests,
        scripts) that construct a `CodeIndexer` directly and let it fall out
        of scope without an explicit shutdown path."""
        # UPG-PURPOSE-PASS-DEFERRAL: drop any queued-but-not-started deferred
        # purpose pass and don't wait for one already running — shutdown
        # must never block on background embedding work. A dropped pass
        # loses nothing durable: the files it would have covered never had
        # their mtimes checkpointed (see index_workspace), so the next
        # index_workspace() call picks them back up.
        executor = getattr(self, "_purpose_executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        client = getattr(self, "_client", None)
        if client is not None:
            client.close()

    def __del__(self) -> None:
        # Best-effort only — must never raise, especially during interpreter
        # shutdown when module globals this depends on may already be torn
        # down (attribute lookups can fail in that window).
        try:
            self.close()
        except Exception:
            pass

    def get_all_documents(self) -> tuple[list[str], list[str], list[dict]]:
        """Return (ids, documents, metadatas) for all stored chunks — used by BM25 index.

        Paginates in batches of 500 to avoid ChromaDB's SQLite variable limit (~999).
        """
        _PAGE = 500
        all_ids: list[str] = []
        all_docs: list[str] = []
        all_meta: list[dict] = []
        offset = 0
        while True:
            with _timed_chroma_call("get"):
                page = self._collection.get(
                    include=["documents", "metadatas"],
                    limit=_PAGE,
                    offset=offset,
                )
            batch_ids = page["ids"]
            if not batch_ids:
                break
            all_ids.extend(batch_ids)
            all_docs.extend(page["documents"])
            all_meta.extend(page["metadatas"])
            offset += len(batch_ids)
            if len(batch_ids) < _PAGE:
                break
        return all_ids, all_docs, all_meta

    def query_vector(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        language: str | None = None,
        languages: list[str] | None = None,
    ) -> dict:
        # `languages` (a set/list) takes precedence over a single `language`:
        # restricts the vector search to any of the given languages via an $in
        # filter. Used by the doc-intent pool reservation (UPG-15.13) to fetch
        # documentation-prose chunks that embed below the unfiltered fetch depth.
        if languages:
            where = {"language": {"$in": list(languages)}}
        elif language:
            where = {"language": language}
        else:
            where = None
        with _timed_chroma_call("query"):
            return self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, max(1, self.total_chunks)),
                where=where,
                include=["documents", "metadatas", "distances"],
            )

    def get_chunk_cosine_similarities(
        self, query_embedding: list[float], chunk_ids: list[str]
    ) -> dict[str, float]:
        """Return cosine similarity scores for a specific set of chunk IDs.

        Used by the forced-inclusion relevance gate (UPG-11.12) to check vector
        similarity for non-compound candidates that are not in the natural pool.
        ChromaDB embeddings are L2-normalised by default, so cosine = dot product.

        Returns a dict {chunk_id: cosine_similarity}.  Chunk IDs not found in the
        collection are absent from the result (similarity treated as 0.0 by caller).
        """
        if not chunk_ids or not query_embedding:
            return {}
        try:
            with _timed_chroma_call("get"):
                batch = self._collection.get(ids=chunk_ids, include=["embeddings"])
        except Exception:
            return {}
        result: dict[str, float] = {}
        import numpy as np
        q = np.asarray(query_embedding, dtype=np.float32)
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0.0:
            return result
        # batch["embeddings"] may be a numpy array — avoid truthiness check on it.
        embeddings_raw = batch.get("embeddings")
        ids_raw = batch.get("ids")
        if embeddings_raw is None or ids_raw is None:
            return result
        embeddings_list = list(embeddings_raw) if not isinstance(embeddings_raw, list) else embeddings_raw
        ids_list = list(ids_raw) if not isinstance(ids_raw, list) else ids_raw
        for cid, emb_raw in zip(ids_list, embeddings_list):
            if emb_raw is None:
                continue
            emb = np.asarray(emb_raw, dtype=np.float32)
            if emb.size == 0:
                continue
            dot = float(np.dot(q, emb))
            e_norm = float(np.linalg.norm(emb))
            if e_norm == 0.0:
                continue
            result[cid] = dot / (q_norm * e_norm)
        return result

    def query_vector_purpose(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        language: str | None = None,
        languages: list[str] | None = None,
    ) -> dict:
        """Query the purpose-vector collection (ARCH-4 dual-vector pool entry).

        Mirrors `query_vector` but against `self._purpose_collection`, which only
        holds symbol-bearing chunks (see `_upsert_purpose_vectors`). A workspace
        indexed before ARCH-4 shipped has an empty purpose collection — `count()`
        guards against ChromaDB raising on a query against zero rows, so old
        indexes degrade gracefully to body-only results until reindexed.
        """
        # This count must be read live rather than from the cached
        # `total_purpose_chunks` counter: the guard below exists specifically
        # because a query against an ACTUALLY-empty collection raises, and a
        # stale cached count that still shows chunks present (e.g. mid-way
        # through a delete that has already emptied the real collection but
        # not yet refreshed the cache) would let that exception through
        # uncaught into the caller. This one read stays live for
        # correctness; it is only reached from this method, which — like
        # every vector-store-reaching call made from a request handler —
        # already runs off the request event loop (the dedicated executor,
        # or the watcher/indexing thread), so a live read here does not
        # reintroduce loop-blocking risk.
        with _timed_chroma_call("count"):
            count = self._purpose_collection.count()
        if count == 0:
            return {"ids": [[]], "distances": [[]], "documents": [[]], "metadatas": [[]]}
        if languages:
            where = {"language": {"$in": list(languages)}}
        elif language:
            where = {"language": language}
        else:
            where = None
        with _timed_chroma_call("query"):
            return self._purpose_collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, max(1, count)),
                where=where,
                include=["documents", "metadatas", "distances"],
            )

    def fetch_chunks(self, ids: list[str]) -> list[dict]:
        """Deterministic re-fetch by chunk id (UPG-CTX-EVICT part a).

        A direct ``self._collection.get(ids=...)`` — no embedding, no rerank,
        no ranking of any kind. Every rendered search result carries its
        chunk's exact ChromaDB id (``file_path:start_line-end_line``); this is
        the counterpart lookup that restores the full chunk from that id
        alone, so a chunk cleared from the caller's context (by harness tool-
        result eviction, ``/compact``, or an API context-editing tombstone)
        never requires a re-search or a blind file re-read.

        Returns one dict per requested id, in REQUEST order (Chroma's own
        ``get(ids=...)`` does not preserve order and silently omits ids it
        doesn't have — both are corrected here):
          {"id": ..., "found": True,  "file_path": ..., "start_line": ...,
           "end_line": ..., "symbol_name": ..., "language": ..., "content": ...}
          {"id": ..., "found": False, "reason": "misaligned" | "file_changed",
           "nearest_ids": [...]}

        UPG-FETCH-ID-MISALIGN-MSG: a miss comes in two genuinely different
        shapes and callers act on them differently. If the requested id's
        line range simply doesn't line up with a stored chunk span but the
        SAME file still has other chunks indexed, the file is fine — the
        caller just guessed a span (e.g. from a stale rendering, or a
        hand-typed range) — so `reason="misaligned"` and `nearest_ids` names
        the closest real chunk ids in that file, letting the caller's very
        next `vectr_fetch` call succeed. If the file has no chunks indexed
        at all, it genuinely may have changed since indexing (edited, moved,
        or deleted) — `reason="file_changed"`, `nearest_ids` empty. An id
        that doesn't parse as `path:start-end` also falls back to
        `"file_changed"` (nothing to compare against).

        Raises ValueError if `ids` exceeds FETCH_MAX_IDS_PER_CALL.
        """
        from agent.config import FETCH_MAX_IDS_PER_CALL
        if len(ids) > FETCH_MAX_IDS_PER_CALL:
            raise ValueError(
                f"Too many ids requested ({len(ids)}) — vectr_fetch accepts at "
                f"most {FETCH_MAX_IDS_PER_CALL} per call."
            )
        if not ids:
            return []
        with _timed_chroma_call("get"):
            batch = self._collection.get(ids=ids, include=["documents", "metadatas"])
        found_ids = batch.get("ids") or []
        found_docs = batch.get("documents") or []
        found_metas = batch.get("metadatas") or []
        by_id: dict[str, tuple[str, dict]] = {
            cid: (doc, meta) for cid, doc, meta in zip(found_ids, found_docs, found_metas)
        }
        results: list[dict] = []
        for requested_id in ids:
            hit = by_id.get(requested_id)
            if hit is None:
                results.append(self._not_found_fetch_entry(requested_id))
                continue
            doc, meta = hit
            results.append({
                "id": requested_id,
                "found": True,
                "file_path": meta.get("file_path", ""),
                "start_line": int(meta.get("start_line", 0)),
                "end_line": int(meta.get("end_line", 0)),
                "symbol_name": meta.get("symbol_name", ""),
                "language": meta.get("language", ""),
                "content": doc,
            })
        return results

    def chunk_span_for_lines(
        self, locations: list[tuple[str, int]],
    ) -> dict[tuple[str, int], tuple[int, int]]:
        """Real, stored chunk boundaries covering each ``(file_path, line)``
        in *locations* — the fetch-id support for trace output
        (UPG-TRACE-FETCH-IDS). ``vectr_trace`` lines name a call site or a
        definition by file+line, not by chunk id; this resolves each such
        site to the ACTUAL indexed chunk span containing it, so the trace
        renderer can hand back a ``relpath:start-end`` id ``vectr_fetch``
        accepts, in the same response — no follow-up ``vectr_locate`` round
        trip per name.

        One metadata-only ``get(where={"file_path": ...})`` per distinct
        file (the same query shape ``_delete_chunks_for_file`` already uses),
        never a query-time embedding or rerank. A location with no covering
        chunk in the index is simply absent from the returned mapping — the
        caller must never invent a boundary for a site the index doesn't
        actually have a chunk for.
        """
        result: dict[tuple[str, int], tuple[int, int]] = {}
        if not locations:
            return result
        by_file: dict[str, list[int]] = {}
        for file_path, line in locations:
            by_file.setdefault(file_path, []).append(line)
        for file_path, lines in by_file.items():
            try:
                with _timed_chroma_call("get"):
                    existing = self._collection.get(
                        where={"file_path": file_path}, include=["metadatas"],
                    )
            except Exception:
                continue
            metas = existing.get("metadatas") or []
            spans: list[tuple[int, int]] = []
            for meta in metas:
                try:
                    spans.append((int(meta["start_line"]), int(meta["end_line"])))
                except (KeyError, TypeError, ValueError):
                    continue
            for line in lines:
                best: tuple[int, int] | None = None
                for s, e in spans:
                    if s <= line <= e and (best is None or (e - s) < (best[1] - best[0])):
                        best = (s, e)
                if best is not None:
                    result[(file_path, line)] = best
        return result

    def _not_found_fetch_entry(self, requested_id: str) -> dict:
        """Build the `found=False` entry for one missed `fetch_chunks` id,
        distinguishing a line-range MISALIGNMENT from a genuinely absent file
        (UPG-FETCH-ID-MISALIGN-MSG).

        A cheap ids-only existence probe (`include=[]`, no metadata/document
        payload) against the SAME `file_path` tells the two cases apart: if
        the file has any other chunk indexed at all, the file itself is
        fine and the miss is purely a span mismatch — `nearest_ids` names the
        closest real chunk ids in that file (ranked by span proximity, an
        arithmetic comparison of the ids' own encoded line ranges — no
        embedding/ranking involved) so the caller's next `vectr_fetch` call
        can succeed without a round trip. If the file has no chunks indexed
        at all, or `requested_id` doesn't even parse as `path:start-end`,
        this falls back to the original "may have changed since indexing"
        signal.
        """
        parsed = _parse_chunk_id(requested_id)
        if parsed is None:
            return {"id": requested_id, "found": False, "reason": "file_changed", "nearest_ids": []}
        file_path, start_line, end_line = parsed
        try:
            with _timed_chroma_call("get"):
                existing = self._collection.get(where={"file_path": file_path}, include=[])
        except Exception:
            existing = {}
        file_chunk_ids = list(existing.get("ids") or [])
        if not file_chunk_ids:
            return {"id": requested_id, "found": False, "reason": "file_changed", "nearest_ids": []}
        nearest = _nearest_chunk_ids_by_span(
            file_chunk_ids, start_line, end_line, FETCH_MISALIGN_NEAREST_MAX,
        )
        return {"id": requested_id, "found": False, "reason": "misaligned", "nearest_ids": nearest}

    def get_chunk_documents(self, chunk_ids: list[str]) -> dict[str, tuple[str, dict]]:
        """Batch-fetch (document, metadata) from the body collection for given ids.

        Used to backfill body content/metadata for chunk ids that were only
        discovered via the purpose-vector query (ARCH-4) and therefore did not
        come back from the body `query_vector` call in the same search pass.
        Ids not found in the body collection are absent from the result.
        """
        if not chunk_ids:
            return {}
        try:
            with _timed_chroma_call("get"):
                batch = self._collection.get(ids=chunk_ids, include=["documents", "metadatas"])
        except Exception:
            return {}
        ids_raw = batch.get("ids") or []
        docs_raw = batch.get("documents") or []
        metas_raw = batch.get("metadatas") or []
        return {
            cid: (doc, meta)
            for cid, doc, meta in zip(ids_raw, docs_raw, metas_raw)
        }
