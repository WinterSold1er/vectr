"""Language/extension mappings, exclusion sets, and batch-size constants.

Intentionally NOT in config.yaml (Tier-3): batch/worker counts are perf/
throughput constants, not user-facing tunables.  Changing them via config could
silently corrupt batch inserts (_UPSERT_BATCH_SIZE is bounded by the SQLite
999-variable limit: 6 fields × 100 = 600 ≤ 999).
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Language extension mapping
# ---------------------------------------------------------------------------

LANG_BY_EXT: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".hxx": "cpp",
    ".zig": "zig",
    ".md": "markdown",
    ".html": "html",
    # UPG-11.3: prose documentation formats — indexed with the doc-prose quality
    # multiplier (_Q_DOC_PROSE = 0.70) so code chunks still lead on code queries
    # while docs surface for prose/howto queries. Uses window-based chunking since
    # there is no AST grammar for plain text or reStructuredText.
    ".txt": "txt",
    ".rst": "rst",
}

EXCLUDED_DIRS = {
    ".git", ".svn", "node_modules", "__pycache__", ".venv", "venv", "env",
    ".env", "dist", "build", ".build", ".next", ".nuxt", "target", "out",
    "coverage", ".coverage", ".mypy_cache", ".pytest_cache", ".ruff_cache",
}

# Intentionally NOT in config.yaml (Tier-3): perf/throughput constants.
# _UPSERT_BATCH_SIZE=100 is bounded by SQLite's 999-variable limit (a code
# invariant: 6 fields × 100 = 600 ≤ 999); changing it via config could
# silently corrupt batch inserts.  _FILE_BATCH_SIZE/_EMBED_BATCH_SIZE are
# pure throughput levers with no behavioural effect on ranking or output.
_FILE_BATCH_SIZE = 64     # used by index_file() — single-file watcher path
_EMBED_BATCH_SIZE = 256   # texts per model.encode() call — larger = better BLAS utilisation
_UPSERT_BATCH_SIZE = 100  # rows per ChromaDB upsert — SQLite variable limit is 999; 6 fields×100=600
_CHUNK_WORKERS = min(8, os.cpu_count() or 4)  # parallel chunking workers

# ---------------------------------------------------------------------------
# L3 content-index chunking/embedding-logic version — mtime-cache rebuild trigger
# ---------------------------------------------------------------------------

# THE CHUNKING-LOGIC VERSION. Bump this — and only this — whenever a change to
# chunking or embedding behavior would make chunks already sitting in an
# existing user's index stale, incomplete, or different from what a fresh
# index would now produce (a new/changed chunk boundary, a chunk that used to
# be dropped and now isn't, a new derived vector, changed purpose-text
# distillation, etc). A binary upgrade with no version bump silently keeps
# serving the OLD chunk set forever — restarting on the new binary re-chunks
# only files whose content itself changed, not the ones affected by the logic
# change (UPG-CHUNK-LOGIC-VERSION-FINGERPRINT).
#
# Intentionally NOT in config.yaml (Tier-3): INDEXING_SCHEMA_VERSION is a
# schema-migration trigger, same category as symbol_graph.SYMBOL_SCHEMA_VERSION
# (agent/symbol_graph/_constants.py) — mirror that file's is_stale()/
# graph_toolchain_fingerprint() pattern when reasoning about this one; the two
# are deliberately the same idea for the two different indexes (L2 symbol
# graph vs L3 content index), not two different designs. They stay separate
# constants/mechanisms (not one merged fingerprint) because they recover
# differently: the symbol graph has no incremental path and always fully
# rebuilds, so its is_stale() only decides whether to log why; this constant's
# mismatch instead colds the incremental mtime cache so index_workspace()'s
# normal per-file diff (chunk vs skip) naturally re-chunks every file, the
# same recovery path force=True already uses (UPG-8.6). Changing it via
# config could silently corrupt or force a rebuild without the usual
# version-bump safeguard.
#
# Stored as a sentinel entry in the per-workspace mtime cache (index_cache.json,
# see CodeIndexer._load_mtime_cache_with_reason/_save_mtime_cache, both
# co-located with the ChromaDB collection under the same per-workspace cache
# dir so they always clear together). On load, a version mismatch is treated
# as a cold cache — every file re-enters `to_index` on the next
# `index_workspace()`, `force` is set True (mirroring the embed-model-stamp
# mismatch a few lines below it) so Phase 2 actually deletes each file's
# previously-stored chunks before Phase 3 re-inserts the fresh ones, and one
# INFO line names the reason, matching the symbol graph's own
# "stale or toolchain changed — full rebuild" message style. A cache with no
# version key at all (pre-ARCH-4, unversioned) is also treated as stale.
INDEXING_SCHEMA_VERSION = 7  # 1: pre-ARCH-4 baseline (unversioned cache, implicit) · 2: per-symbol purpose vector added (ARCH-4) · 3: purpose-text docstring distillation changed to first-paragraph-only + capped non-Python leading-doc block (ARCH-4-DEBUG) — old purpose vectors are stale relative to a fresh index and must rebuild · 4: symbol-bearing definition chunks (class/struct/enum/interface/type-alias/function/method) exempted from UPG-1.1 trivial-drop (UPG-TRIVIAL-DROP-ALIAS-DEFS) — previously-dropped one-line alias defs now emit a chunk, changing the chunk set for affected files · 5: .txt/.rst prose split at reStructuredText setext-style section headings instead of always falling to 200-line window chunks (UPG-TXT-CHUNK-COVERAGE) — a doc with RST section structure now emits one "section" chunk per heading instead of a handful of merged windows · 6: chunks exceeding indexing.max_chunk_chars are sub-split into bounded pieces (UPG-WINDOW-CHUNK-BYTE-CAP) — oversized chunks (one giant line defeats every line-based cap) now emit several ≤cap pieces with #<k>-suffixed ids where a single line is hard-split, changing the chunk set for affected files · 7: header-cap truncation no longer silently drops content — a capped node's post-cap remainder that no member chunk carries is re-emitted as bounded "overflow" continuation chunks with honest sub-spans (fields/static-initializers/enum constants past the container cap; giant leaves past max_chunk_lines), and java constructor_declaration joins the chunk targets (UPG-ENUM-CONTAINER-NONMETHOD-COVERAGE); rust impl_item chunk symbol_name resolved via the grammar `type` field, so `impl Trait for Type` members stop carrying `# class: <trait>` context (UPG-RUST-IMPL-SYMBOL-NAME-TRAIT-VS-TYPE) — all three change stored chunk sets/text

# Sentinel key inside the mtime-cache JSON that carries INDEXING_SCHEMA_VERSION.
# Chosen to never collide with a real filesystem path (mtime-cache keys are
# always absolute file paths).
_MTIME_CACHE_SCHEMA_KEY = "__vectr_index_schema_version__"

# ---------------------------------------------------------------------------
# Embedding-model version stamp — vector-space safety (UPG-EMBEDDER-SWAP-GRANITE)
# ---------------------------------------------------------------------------

# A separate small JSON file (co-located with the mtime cache under the same
# per-workspace `~/.cache/vectr/db/<hash>` directory) recording the
# `CodeIndexer.embed_model` identifier that built the CURRENT contents of the
# ChromaDB collection. Kept out of the mtime-cache JSON deliberately: that
# file's mismatch handling only resets the incremental-skip state (an
# optimization), whereas an embed-model mismatch is a correctness issue —
# vectors from two different models must never coexist in one collection,
# so it must unconditionally force CodeIndexer.index_workspace()'s
# `force=True` full-rebuild path (unconditional per-file delete-then-
# reinsert), not just the softer "treat cache as cold" behaviour. A missing
# stamp file (a pre-existing index built by a vectr version that predates
# this mechanism) is treated as a mismatch too, since we cannot know what
# model produced those vectors.
_EMBED_MODEL_STAMP_FILE = "embed_model_stamp.json"

# ---------------------------------------------------------------------------
# Purpose-vector completion cache (UPG-PURPOSE-RESUME-HOLE)
# ---------------------------------------------------------------------------

# A separate small JSON file (same per-workspace cache dir as the mtime cache
# and the embed-model stamp) recording, per file, the content mtime as of
# which that file's purpose vectors (ARCH-4's dual-vector pool-entry signal)
# are known fully written to `code_chunks_purpose` — idempotently upserted by
# chunk_id, so "fully written" here means every one of that file's candidate
# chunks has actually been considered (embedded+upserted, or determined not
# symbol-bearing) at least once.
#
# Deliberately a SEPARATE file/cache from `index_cache.json` (mtime_cache),
# not another key inside it: content-completion and purpose-completion are
# decoupled signals by design (UPG-PURPOSE-PASS-DEFERRAL) — a file can be
# content-complete (in mtime_cache) while its purpose vectors are still
# missing (an interrupted deferred pass, or a first run after upgrading to
# this mechanism). CodeIndexer.index_workspace() diffs mtime_cache against
# this cache on every ordinary (non-force) run to find files needing a
# purpose-only backfill, closing the hole where such a gap could previously
# persist forever until --force or a file touch. See
# `_load_purpose_cache`/`_save_purpose_cache`/`_seed_purpose_cache_from_collection`.
_PURPOSE_CACHE_FILE = "purpose_cache.json"

# Sentinel key inside the purpose-cache JSON, same convention and purpose as
# `_MTIME_CACHE_SCHEMA_KEY` above — a purpose cache written by an older
# INDEXING_SCHEMA_VERSION (e.g. before purpose-text distillation changed) is
# treated as cold, so every file's completion is re-derived rather than
# trusted stale.
_PURPOSE_CACHE_SCHEMA_KEY = "__vectr_purpose_schema_version__"
