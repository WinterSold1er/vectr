"""Business logic: owns the indexer, searcher, watcher, and memory layer lifecycle."""
from __future__ import annotations

import logging
import os
import secrets
import threading
import time
import weakref
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from agent.config import (
    DEFAULT_PORT,
    INDEXING_VECTOR_STORE_DISPATCH_MAX_WORKERS,
    STRATEGY_DEFAULT_BM25_WEIGHT,
    STRATEGY_DEFAULT_SEMANTIC_WEIGHT,
    STRATEGY_KNOWN_FRAMEWORKS,
    SEARCH_IDENTIFIER_HINT_ENABLED,
    SEARCH_IDENTIFIER_HINT_MAX_IDENTIFIERS,
    SEARCH_IDENTIFIER_HINT_MAX_LOCATIONS,
    SEARCH_IDENTIFIER_HINT_NEARMISS_ENABLED,
    SEARCH_IDENTIFIER_HINT_NEARMISS_MAX,
    ARC_DETECTION_ENABLED,
    BOUNDARY_PRECOMPACT_ENABLED,
    BOUNDARY_PRECOMPACT_TOKEN_CAP,
    embed_model_spec,
    EPISODES_DIGEST_HEAD_LINES,
    EPISODES_DIGEST_MAX_CHARS,
    EPISODES_DIGEST_TAIL_LINES,
    EPISODES_MAX_ROWS,
    EPISODES_TTL_DAYS,
    EVICTION_MAX_TRACKED_SESSIONS,
    HOOKS_COMMIT_NOTE_MAX_FILES,
    HOOKS_COMMIT_NOTE_MAX_SUBJECT_CHARS,
    HOOKS_LOG_INJECTIONS,
    HOOKS_LOG_CHARS_PER_TOKEN,
    HOOKS_MIN_SIMILARITY,
    MEMORY_HYGIENE_STALE_TASK_WARN_AGE_DAYS,
    MEMORY_HYGIENE_STALE_TASK_WARN_COUNT,
    MEMORY_WRITE_RELATED_ENABLED,
    MEMORY_WRITE_RELATED_LIMIT,
    MEMORY_WRITE_RELATED_MIN_SIMILARITY,
    MEMORY_WRITE_PROXY_SUGGEST_ENABLED,
    MEMORY_WRITE_PROXY_SUGGEST_LIMIT,
    vectr_cache_root,
)
from agent.eviction_advisor import EvictionAdvisor
from agent.proxy_anchors import suggest_proxy_anchors
from agent.trigger_engine import TriggerFireLedger, TurnInjectionLedger, token_estimate
from agent.version_stamp import compute_version_stamp
from agent.working_context_store._related import RelatedNote

logger = logging.getLogger(__name__)

_DB_DIR_ENV = "VECTR_DB_DIR"

# PreCompact boundary-preservation surface (GET /v1/boundary/precompact):
# arc-independent base text, always rendered when the surface is enabled.
# Static product copy, not a tunable — see `agent/config.yaml`'s
# `episodes.boundary_precompact_*` keys for the enable switch and token cap.
_BOUNDARY_PRECOMPACT_BASE_TEXT = (
    "Before this context is replaced: preserve, in the summary, the concrete things this session LEARNED "
    "that are not already written down — commands that failed and what actually fixed them, environment and "
    "build facts specific to this machine or repo, and conventions discovered by reading the code. Keep them "
    "as specific statements (the command, the flag, the path), not as a topic list."
)


def _cache_dir_slug(workspace_root: str) -> str:
    """The per-workspace cache-dir name: md5(abs_workspace_path)[:12]."""
    import hashlib
    return hashlib.md5(workspace_root.encode()).hexdigest()[:12]


def _default_db_dir(workspace_root: str) -> str:
    """Resolve the DB dir ~/.cache/vectr/<workspace-hash>/ for this workspace.

    The cache holds the plaintext code index and (unless encrypted) working-
    memory notes, so the shared parent is restricted to the owner on POSIX
    hosts (see agent/fs_permissions.py); because the parent is owner-only, any
    child it later gains is unreachable by other users regardless of its own
    mode.

    UPG-CACHE-LITTER: this resolver only secures the shared parent and returns
    the path — it does NOT create the per-workspace subdir. Eagerly creating it
    on every path resolution (service init / status probe, before any indexing
    or note write) left hundreds of empty <hash>/ dirs under the cache root. The
    dir is created + secured by the constructing service (VectrService.__init__)
    only when a real service is built, which then immediately writes its notes
    DB — so an empty dir never corresponds to a live workspace."""
    from agent.fs_permissions import secure_dir
    cache_root = vectr_cache_root()
    secure_dir(cache_root)
    return str(cache_root / _cache_dir_slug(workspace_root))


_MEMORY_ONLY_MSG = (
    "vectr is in memory-only mode for this workspace — semantic search and the "
    "symbol graph are disabled; memory tools (remember/recall/snapshot) and hooks "
    "are active."
)

_SEARCH_ONLY_MSG = (
    "vectr is in search-only mode for this workspace — working-memory tools "
    "(remember/recall/forget/snapshot) are disabled; semantic search, locate, "
    "trace and map are active."
)

# UPG-STDIO-MEMORY-READY: shown by the REST/MCP-HTTP transport (see routes.py)
# for search-touching endpoints called before phase 2 (embedder/indexer/
# searcher/watcher/symbol-graph construction) finishes. Memory tools
# (remember/recall/forget/status/snapshot) never show this — they work from
# process start, on every transport, because the working-memory store is
# constructed without an embedder in phase 1. Mirrors stdio transport's own
# "_STILL_STARTING_MSG" (integrations/mcp_server/_stdio.py) for the same
# state, kept as a separate constant since the two transports' callers never
# share a code path.
_STILL_INITIALIZING_MSG = (
    "vectr is still starting up (loading the embedding model / building the "
    "workspace index) — search, locate, trace, and map are not yet available. "
    "Memory tools (remember/recall/forget/snapshot/status) are ready now. Try "
    "again in a few moments."
)

# UPG-STDIO-MEMORY-READY: appended to a vectr_recall(query=...) result when
# the call landed during the phase-2 warm-up window — the embedding model is
# still loading/downloading in the background, so this result used the
# lexical (SQL LIKE) fallback rather than semantic ranking. State-based only
# (gated on VectrService.embedder_ready, never on the query's content):
# recall() already chooses lexical vs. semantic purely from whether an
# embedder is attached, so this notice reports that same state truthfully
# rather than re-deriving it. Never appears once the embedder has attached —
# `recall(query=None)` (index/list mode) never appends it either, since
# there is no ranking to degrade.
_EMBEDDER_LOADING_RECALL_NOTICE = (
    "\n\n[semantic ranking unavailable: the embedding model is still loading "
    "or downloading in the background — these are lexical (keyword) matches "
    "only. Re-run this same vectr_recall query later for semantically-ranked "
    "results.]"
)

# UPG-CTX-EVICT: shared note appended to a vectr_fetch/`/v1/fetch` response
# whenever at least one requested id came back missing — the file most
# likely changed since indexing (edited, moved, or deleted), shifting or
# removing that chunk's id. Re-running vectr_search recovers the current
# content under a fresh id.
_FETCH_NOT_FOUND_NOTE = (
    "One or more requested ids were not found — the file likely changed "
    "since indexing (edited, moved, or deleted), which shifts or removes "
    "chunk ids. Re-run vectr_search to get current ids."
)

# UPG-COMMIT-MEMORY-HOOK: markers on every note `record_commit_note` writes.
# `_COMMIT_NOTE_TAG` makes auto-provenance notes filterable/identifiable via
# the existing tags mechanism (no schema change); `_COMMIT_NOTE_AGENT_ID` is
# a complementary attribution marker on the note's existing `author_id`
# column (renders as "@git-post-commit-hook" in recall index lines,
# mirroring how vectr_remember(agent=...) already attributes multi-agent
# notes). Boot-injection exclusion itself does not depend on either of
# these — it falls structurally out of kind="finding" (see
# `record_commit_note`'s docstring).
_COMMIT_NOTE_TAG = "auto-provenance"
_COMMIT_NOTE_AGENT_ID = "git-post-commit-hook"

# Ceiling on deferred-charge proactive retrievals held awaiting delivery
# confirmation (UPG-PROXY-APPEND-BURNS-COOLDOWN). Only entries between a
# caller's retrieval and its confirmation live here — a handful even under
# heavy concurrent proxy traffic — so this is a runaway guard, not a working
# size. Hardcoded rather than configurable for the same reason
# `agent.proactive.gate.LedgerStore` hardcodes its own `max_sessions` bound:
# it is a memory-safety invariant, not a tuning knob.
PROACTIVE_MAX_PENDING_DELIVERIES = 256


def _format_commit_note_content(
    sha: str, subject: str, branch: str, files: list[str], task_note,
) -> str:
    """Deterministic, zero-inference content for a commit-provenance note
    (UPG-COMMIT-MEMORY-HOOK) — every line is either a raw git fact or an
    existing note's own id/title, never an inference. `files` beyond
    HOOKS_COMMIT_NOTE_MAX_FILES collapse into "+N more" rather than growing
    the note unbounded on a large mechanical commit."""
    subject = subject[:HOOKS_COMMIT_NOTE_MAX_SUBJECT_CHARS]
    lines = [f"Commit {sha} on {branch or '(detached HEAD)'}: {subject}"]
    if files:
        shown = files[:HOOKS_COMMIT_NOTE_MAX_FILES]
        remainder = len(files) - len(shown)
        file_list = ", ".join(shown)
        if remainder > 0:
            file_list = f"{file_list}, +{remainder} more"
        lines.append(f"Files ({len(files)}): {file_list}")
    if task_note is not None:
        task_title = task_note.title or (
            task_note.content.strip().splitlines()[0][:80] if task_note.content.strip() else ""
        )
        lines.append(f"Active task: [#{task_note.note_id}] {task_title}")
    return "\n".join(lines)


@dataclass(frozen=True)
class RememberOutcome:
    """Return shape of `VectrService.remember_with_extras()` — the write
    itself (`note_id`) plus two strictly-additive write-time offers computed
    alongside it (see that method's docstring for the exact gating). Never
    replaces the plain `remember()` -> int contract: this is a NEW method,
    kept separate so existing callers (and the 495+ test call sites against
    `remember()`) are untouched."""
    note_id: int
    related: list[RelatedNote]
    proxy_anchor_suggestions: list[str]


class VectrService:
    """Singleton-style service. Create once at startup; shared via FastAPI app state."""

    def __init__(
        self,
        workspace_root: str,
        port: int = DEFAULT_PORT,
        extra_roots: list[str] | None = None,
        memory_only: bool = False,
        search_only: bool = False,
        workspace_explicit: bool = False,
        configure_ide: bool = True,
        defer_search_init: bool = False,
    ) -> None:
        """`defer_search_init` (UPG-STDIO-MEMORY-READY): when True, this
        constructor runs ONLY the fast, synchronous phase below — no
        embedding model load, no code indexer/searcher/watcher/symbol-graph
        construction, no IDE-config writes. Working-memory tools (remember/
        recall/forget/status/snapshot/snapshot_list) are fully usable the
        moment the constructor returns. The caller must then call
        `complete_search_init()` (typically on a background thread) to run
        the deferred phase 2. Defaults to False so every existing caller —
        including the 17+ test modules that construct this class directly —
        gets today's fully-synchronous, single-phase behavior unchanged;
        only the stdio transport (main.py) and the HTTP daemon (api.py)
        opt in, since only they need memory tools to answer before the
        embedding model finishes loading/downloading.
        """
        from agent.working_context_store import WorkingContextStore
        from integrations.workspace_detect import find_workspace_root

        # UPG-WS-ROOT-MISDETECT: an explicitly-given workspace path (CLI
        # positional arg or --path flag) always wins verbatim — it must never
        # be silently replaced by the enclosing git repo's root. The
        # git-toplevel walk-up in find_workspace_root only applies when the
        # caller gave no path at all (workspace_explicit=False, e.g. a bare
        # `vectr start` defaulting to cwd).
        if workspace_explicit:
            self._workspace_root = str(Path(workspace_root).resolve())
        else:
            self._workspace_root = find_workspace_root(workspace_root)
        self._extra_roots: list[str] = list(extra_roots or [])
        self._port = port
        self._configure_ide = configure_ide
        # Some embedding models are asymmetric — search queries must be embedded
        # via the provider's embed_query (registered "query" prompt), never embed().
        # See agent/indexer/_types.py:LocalEmbedProvider for the query/document split,
        # detected from the loaded model itself rather than hardcoded per model.
        self._embed_model = embed_model_spec()
        # Memory-only mode: code indexing + file watcher are disabled.
        # Reads from env (propagated by _do_start) or from the constructor arg.
        self._memory_only: bool = memory_only or (os.getenv("VECTR_MEMORY_ONLY", "") == "1")
        # Search-only mode: the dual of memory-only — indexing/watcher run normally,
        # but the working-memory layer is disabled (no notes DB is created).
        # Reads from env (propagated by _do_start) or from the constructor arg.
        self._search_only: bool = search_only or (os.getenv("VECTR_SEARCH_ONLY", "") == "1")
        if self._memory_only and self._search_only:
            raise ValueError(
                "VectrService cannot run in both memory_only and search_only mode simultaneously"
            )

        db_dir = os.getenv(_DB_DIR_ENV) or _default_db_dir(self._workspace_root)
        # UPG-STDIO-MEMORY-READY: the working-memory store (below) is now the
        # FIRST thing constructed against db_dir — previously CodeIndexer ran
        # first and its ChromaDB PersistentClient happened to create the
        # directory as a side effect. This is where the per-workspace dir is
        # created + secured: `_default_db_dir` deliberately no longer does so
        # (UPG-CACHE-LITTER — a bare path resolution must not litter the cache
        # root), and a VECTR_DB_DIR override (e.g. a hosted deployment's mounted
        # volume) may point at a path that doesn't exist yet. Constructing a real
        # service is exactly the point at which a durable dir is warranted — the
        # store's _init_db() writes into it immediately below, so it never stays
        # empty.
        from agent.fs_permissions import secure_dir
        secure_dir(db_dir)
        self._db_dir = db_dir

        logger.info("Initialising Vectr for workspace: %s (db: %s)", self._workspace_root, db_dir)

        # Version stamp (UPG-CLI-DAEMON-VERSION-SKEW): computed once at daemon
        # startup and never refreshed for the lifetime of this process — the
        # whole point is to detect that *this running process* predates a
        # source upgrade the CLI now sees on disk.
        self._version_stamp = compute_version_stamp()

        # Phase-2 objects (UPG-STDIO-MEMORY-READY): None until
        # _init_search_layer() runs — inline below by default, or later via
        # complete_search_init() when defer_search_init=True. Every reader
        # of these (status(), last_indexed, search/locate/trace/fetch/map)
        # either tolerates None (memory-facing code never touches them) or
        # is itself gated on `fully_ready` by the caller (MCP dispatch / REST
        # routes), so a phase-1-only service never raises — it just reports
        # "not built yet" truthfully.
        self._indexer = None
        self._searcher = None
        self._watcher = None
        self._passport_store = None
        self._symbol_graph = None

        # Memory layer — constructed WITHOUT an embedder (UPG-STDIO-MEMORY-READY):
        # remember/recall/forget/status/snapshot work immediately, on every
        # transport, before the embedding model has loaded. Semantic ranking
        # (and the note-vector store, sharing the code index's ChromaDB
        # client) attaches once phase 2 completes — see _init_search_layer()
        # and WorkingContextStore.attach_embedder(). Until then, recall()
        # transparently uses its existing lexical SQL LIKE fallback. In
        # search-only mode the store is never constructed at all
        # (UPG-SEARCH-ONLY-MODE): no notes DB file and no 'working_memory'
        # Chroma collection are created for a workspace that never writes a note.
        self._context_store: WorkingContextStore | None = None
        if not self._search_only:
            self._context_store = WorkingContextStore(db_dir)

        # Episode capture: its own store/connection to the SAME db file,
        # deliberately never imported by `agent/searcher.py` or
        # `agent/working_context_store` — episode rows cannot reach a
        # vectr_search/vectr_recall result by construction. Same
        # search-only gating as the context store above: no episodes table
        # writer exists for a workspace that never records one.
        from agent.episode_store import EpisodeStore
        self._episode_store: EpisodeStore | None = None
        if not self._search_only:
            self._episode_store = EpisodeStore(db_dir)

        # Arc detector: one instance for this VectrService's lifetime,
        # mirroring the episode store's own search-only gating immediately
        # above — a workspace that never records an episode never needs a
        # detector either. Its
        # internal per-session state (app/arcs.py's `_SessionState`) is
        # in-memory only and reset on daemon restart; that is acceptable
        # because the design's pending-failure window is itself bounded
        # (ARC_WINDOW_TTL_SECONDS / ARC_WINDOW_MAX_COMMANDS), not meant to
        # survive process restarts.
        from app.arcs import ArcDetector
        self._arc_detector: ArcDetector | None = None
        if not self._search_only:
            self._arc_detector = ArcDetector()

        # Eviction advisor (UPG-EVICT-SESSION-SCOPE): one advisor per calling
        # MCP session, so one session never sees chunks retrieved by another.
        # `self._eviction_advisor` remains the shared advisor used when no
        # session_id is known (REST callers, backwards-compat callers) — every
        # session-scoped lookup below falls back to it when session_id is None.
        self._eviction_threshold_tokens = int(os.getenv("VECTR_EVICT_THRESHOLD", "4000"))
        self._eviction_advisor = EvictionAdvisor(
            eviction_threshold_tokens=self._eviction_threshold_tokens
        )
        self._session_advisors: dict[str, EvictionAdvisor] = {}

        # Per-session trigger fire ledger (TRIGGER-ENGINE wave 1,
        # bm2-design-skeleton.md §3) — mirrors the EvictionAdvisor registry
        # immediately above. Unlike that registry there is no daemon-global
        # fallback ledger: fire-dedup is only meaningful within one session's
        # identity, so a caller with no session_id (see `_ledger_for`) simply
        # gets no suppression rather than sharing state across callers.
        self._trigger_ledgers: dict[str, TriggerFireLedger] = {}

        # Per-turn cross-surface injection ledger (TRIGGER-ENGINE wave 3,
        # UPG-MEMORY-STATE-MACHINE §5.3/§5.4) — a SEPARATE registry from
        # `self._trigger_ledgers` immediately above: that ledger is
        # session-scoped and only resets at compaction; this one is
        # note_id-granular (not axis-granular) and resets every
        # UserPromptSubmit (see `recall()`), closing the arm-C double-dip
        # finding across session-start bulk / PreToolUse anchored-or-
        # command-family / prompt-time semantic. Same LRU-bounded shape as
        # the advisor/ledger registries above.
        self._turn_ledgers: dict[str, TurnInjectionLedger] = {}

        self._indexing = False
        self._index_thread: threading.Thread | None = None
        self._index_lock = threading.Lock()

        # Per-tool call counters — tracked across all callers (parent + sub-agents)
        # so benchmark tooling can read accurate counts via GET /v1/call_counts.
        self._call_counts: dict[str, int] = {}
        self._call_counts_lock = threading.Lock()

        # Per-hook-kind injection counters (UPG-HOOK-INJECT-OBSERVABILITY): how
        # many times each hook's recall actually returned notes to inject,
        # since this process started. Recorded from `recall(hook_event=...)`,
        # surfaced in `status()` / `vectr status`.
        self._hook_injection_counts: dict[str, int] = {}
        self._hook_injection_lock = threading.Lock()

        # Proactive context (UPG-PRO) — per-channel injection counters (like the
        # hook counters above) + a per-session dedup ledger shared across the
        # matcher/gate for the lifetime of this process. Lazily built.
        self._proactive_injection_counts: dict[str, int] = {}
        self._proactive_injection_lock = threading.Lock()
        self._proactive_ledger = None  # agent.proactive.gate.LedgerStore

        # Deferred-charge retrievals awaiting delivery confirmation
        # (UPG-PROXY-APPEND-BURNS-COOLDOWN): delivery_token -> the retrieval
        # metadata needed to charge the cooldown ledger and write a truthful
        # PROACTIVE_INJECT line once the caller confirms the block was actually
        # appended. Bounded LRU with its own lock, same discipline as
        # `agent.proactive.gate.LedgerStore`: a long-lived daemon serving a
        # chatty proxy can never grow unbounded state here.
        #
        # CONCURRENCY RESIDUAL (accepted, see `proactive_context`): the charge
        # now lands one round-trip after selection, so two proxy requests in
        # flight AT THE SAME TIME can both select the same note before either
        # confirms, and it is injected twice. Requests through one proxy are
        # serialised by the model turn between them, so this needs genuinely
        # concurrent traffic (e.g. parallel subagents sharing a proxy) to
        # occur. A duplicated note line is noise; the alternative — charging at
        # retrieval — silently loses the note entirely, which is the P0 this
        # replaces. Overflow eviction fails the same safe way: an evicted entry
        # is simply never charged, so its note may be injected again later.
        self._proactive_pending: "OrderedDict[str, dict]" = OrderedDict()
        self._proactive_pending_lock = threading.Lock()

        # Org-wide vectr-artifact cache (UPG-PRO caching) — off unless enabled.
        # Bumped monotonically on every note mutation so recall artifacts keyed
        # by the notes epoch invalidate the moment a note changes.
        self._notes_mutation_seq = 0
        self._notes_mutation_lock = threading.Lock()
        self._artifact_cache = None  # agent.proactive.cache.ArtifactCache
        try:
            from agent.proactive.settings import ProactiveSettings
            _pro = ProactiveSettings.from_env()
            if _pro.cache_enabled:
                from agent.proactive.cache import ArtifactCache
                self._artifact_cache = ArtifactCache(
                    max_entries=_pro.cache_max_entries,
                    ttl_seconds=_pro.cache_ttl_seconds,
                    similarity_threshold=_pro.cache_similarity_threshold,
                )
        except Exception:
            self._artifact_cache = None

        # Adaptive strategy — computed after first index, defaults until then
        from agent.strategy_selector import RetrievalStrategy
        self._strategy: RetrievalStrategy | None = None

        # Phase-2 readiness signals (UPG-STDIO-MEMORY-READY): `embedder_ready`
        # flips the moment CodeIndexer's embed provider finishes loading
        # (inside _init_search_layer, before the slower searcher/watcher/
        # symbol-graph work) — used to gate the vectr_recall lexical-fallback
        # notice. `fully_ready` flips once ALL of phase 2 has completed — used
        # to gate search/locate/trace/map/fetch across every transport. Both
        # are set before this constructor returns in the default
        # (defer_search_init=False) path, so existing callers never observe
        # a not-ready state.
        self._embedder_ready = threading.Event()
        self._fully_ready = threading.Event()
        # UPG-SHUTDOWN-INIT-RACE: set by shutdown(). Phase 2 runs on a
        # background thread for deferred callers, so shutdown can arrive
        # before or DURING _init_search_layer — this event makes a
        # not-yet-started phase 2 a no-op and a mid-flight one tear down
        # whatever it just constructed instead of orphaning it.
        self._shutdown_requested = threading.Event()
        # UPG-STATUS-STALE-GRAPH: set while _build_symbol_graph() is running so
        # /v1/status can flag that the symbol_count / symbol_graph_complete it is
        # serving comes from the previous (possibly stale-toolchain) persisted
        # graph and a rebuild is overwriting it right now.
        self._symbol_graph_rebuilding = threading.Event()

        # A search/fetch call made from a REST route or the equivalent MCP
        # tool dispatch — both served by the same request event loop —
        # reaches the vector store (embed + query/get). Such a call is
        # routed through this single dedicated executor via
        # agent.chroma_dispatch instead of running in place, so a slow
        # vector-store call (e.g. one blocked behind the store's own
        # internal compaction) cannot hold the loop and freeze every other
        # in-flight request. A single worker preserves the request-at-a-time
        # ordering these specific calls already had while they ran directly
        # on the one event-loop thread — see config.yaml's
        # `indexing.vector_store_bridge.dispatch_max_workers` comment for why
        # this must stay 1. A full-workspace index is a separate, much
        # longer-running kind of vector-store work and is deliberately NOT
        # routed through this executor (see the /v1/index route) — sharing
        # it would serialize every search/fetch call behind however long
        # whatever index call happened to be in flight took, which is a new
        # form of starvation this fix must not introduce. Created in phase 1
        # (unconditionally, before `defer_search_init` is even checked) so it
        # exists for the lifetime of the service regardless of when/whether
        # phase 2 completes.
        self._chroma_executor = ThreadPoolExecutor(
            max_workers=INDEXING_VECTOR_STORE_DISPATCH_MAX_WORKERS,
            thread_name_prefix="vectr-chroma",
        )
        # Safety net: a ThreadPoolExecutor's worker thread blocks indefinitely
        # waiting for work and is only stopped by an explicit shutdown() (or
        # interpreter exit) — a VectrService instance that is dropped without
        # going through its own shutdown() (a short-lived script, a failed
        # startup elsewhere, a test fixture with no teardown) would otherwise
        # leak that thread for the rest of the process's life. weakref.finalize
        # guarantees the executor is shut down when this service is garbage-
        # collected even if shutdown() was never called; whichever of the two
        # runs first wins and the other becomes a safe no-op (finalize runs
        # its callback at most once).
        self._chroma_executor_finalizer = weakref.finalize(
            self, self._chroma_executor.shutdown, wait=False, cancel_futures=True,
        )
        # UPG-NOTES-CHROMA-BLOCKING-EVENT-LOOP: this `wait=False` only governs
        # what THIS finalizer/shutdown() call blocks on — it does not opt this
        # executor's worker thread out of `concurrent.futures.thread._python_exit`,
        # a module-level `atexit` hook every `ThreadPoolExecutor` registers into
        # unconditionally (verified against the running interpreter's stdlib:
        # `inspect.getsource(concurrent.futures.thread._python_exit)`). At real
        # interpreter exit that hook wakes every still-tracked worker thread and
        # `.join()`s each one with no timeout, regardless of any executor's own
        # `wait=` argument. `concurrent.futures` exposes no supported way to opt
        # an executor's threads out of it. In practice this only extends process
        # exit by however long whatever chroma call is in flight at that exact
        # moment takes (the same bounded, `slow_call_warn_seconds`-watched
        # operations already logged elsewhere) — it cannot hang indefinitely
        # under normal operation, and calling `shutdown()` here (already done,
        # see above) during a graceful daemon stop lets the worker exit on its
        # own well before this hook ever runs. Documented rather than worked
        # around: building a private thread pool to sidestep a stdlib atexit
        # hook is out of proportion to this exposure. See UPG-CHROMA-EXECUTOR-
        # ATEXIT-JOIN for the proposed follow-up if this ever needs revisiting.

        if not defer_search_init:
            self._init_search_layer()

    def _init_search_layer(self) -> None:
        """Phase 2 (UPG-STDIO-MEMORY-READY): the slow, model-loading half of
        construction — CodeIndexer (embedding model), CodeSearcher,
        CodeWatcher, reranker warm-up, the L1 passport store, the L2 symbol
        graph, and (unless search-only) attaching the now-loaded embedder to
        the already-live working-memory store. Runs inline from __init__ by
        default; deferred callers (stdio, HTTP daemon) run it explicitly via
        `complete_search_init()`, typically on a background thread, after
        phase 1 has already made memory tools available.
        """
        from agent.indexer import CodeIndexer
        from agent.searcher import CodeSearcher
        from agent.watcher import CodeWatcher
        from agent.cartographer import PassportStore
        from agent.symbol_graph import SymbolGraph
        from integrations.vscode_bridge import configure_all

        # L3 — content retrieval (existing)
        # db_path scopes ChromaDB under the same configured db_dir as all other stores
        self._indexer = CodeIndexer(
            self._workspace_root,
            embed_model=self._embed_model,
            db_path=str(Path(self._db_dir) / "chroma"),
            extra_roots=self._extra_roots,
        )
        # The embedding model has now loaded. Attach it to the memory layer
        # immediately — before the slower searcher/watcher/symbol-graph work
        # below — so notes written during the warm-up window get semantic
        # recall, and any note missing a vector gets backfilled, as soon as
        # possible rather than only once the rest of phase 2 finishes.
        if self._context_store is not None:
            self._context_store.attach_embedder(
                embed_fn=self._indexer.embed_texts,
                embed_query_fn=self._indexer.embed_query_batch,
                notes_chroma_client=self._indexer.chroma_client,
                embed_model=self._indexer.embed_model,
            )
        self._embedder_ready.set()

        self._searcher = CodeSearcher(self._indexer)
        self._watcher = CodeWatcher(self._indexer, searcher_refresh_fn=self._searcher.refresh_bm25)
        # UPG-RERANKER-HF-NETWORK: warm the reranker at startup, alongside the
        # embedder already loaded synchronously above (CodeIndexer's constructor
        # instantiates the embed provider). This moves the cross-encoder's
        # model-load cost out of the first vectr_search call. Skipped in
        # memory-only mode: there is no code index, search is disabled, and
        # there is nothing to rerank.
        if not self._memory_only:
            self._searcher.warm_reranker()

        # L1 — codebase passport (AI-written, stored by vectr_map_save)
        self._passport_store = PassportStore(self._db_dir)

        # L2 — symbol graph
        self._symbol_graph = SymbolGraph(self._db_dir)
        # TRIGGER-ENGINE wave 2b: upgrade the memory layer to the S (symbol)
        # trigger primitive now that the symbol graph exists — mirrors the
        # attach_embedder() upgrade-after-construction above. Memory-only
        # daemons (self._context_store is not None but this phase never
        # runs) and the warm-up window before this line executes both leave
        # the store's resolver unattached, so a symbol trigger deterministically
        # does not fire until this call lands — never an error.
        if self._context_store is not None:
            self._context_store.attach_symbol_resolver(self._symbol_graph)
        # ARCH-1b: seed the searcher's importance prior from the persisted
        # symbol_importance table so a restart over an already-indexed workspace
        # ranks with importance immediately (empty until ARCH-1a has run once).
        self._searcher.set_file_importance(
            self._symbol_graph.file_importance(self._workspace_root)
        )
        # ARCH-2: seed the searcher's class-importance prior from the persisted
        # class_importance table the same way (empty until ARCH-2 has run once).
        self._searcher.set_class_importance(
            self._symbol_graph.class_importance(self._workspace_root)
        )
        # UPG-TESTPATH-FRAMEWORK-MISCLASS (F58): seed the searcher's test-framework
        # fan-in exemption map from the persisted file_fan_in table the same way
        # (empty until it has run once).
        self._searcher.set_file_fan_in(
            self._symbol_graph.file_fan_in(self._workspace_root)
        )

        # IDE config files point at this process's HTTP port — meaningless for a
        # caller that has no HTTP port (e.g. the stdio transport) and potentially
        # disruptive for a workspace it doesn't own (e.g. a hosting platform's
        # mounted container filesystem). Callers without a real port opt out.
        if self._configure_ide:
            for root in self._indexer.all_roots:
                configure_all(str(root), self._port)

        # UPG-SHUTDOWN-INIT-RACE: shutdown() raced this phase-2 run (e.g. an
        # instant client disconnect during warm-up). Tear down what was just
        # constructed rather than orphaning it — watcher.stop() and
        # indexer.close() are both idempotent, so overlapping with
        # shutdown()'s own stop/close pass is safe — and never report
        # fully_ready on a shut-down service.
        if self._shutdown_requested.is_set():
            if self._watcher is not None:
                self._watcher.stop()
            if self._indexer is not None:
                self._indexer.close()
            return

        self._fully_ready.set()

    def complete_search_init(self) -> None:
        """Run phase 2 for a service constructed with `defer_search_init=True`
        (UPG-STDIO-MEMORY-READY). Callers (stdio transport, HTTP daemon
        lifespan) invoke this once, typically on a background thread, right
        after phase 1 has already made memory tools available.

        Idempotent: a no-op if phase 2 has already completed — either because
        this was already called once, or because the service was constructed
        with `defer_search_init=False` (the default) and phase 2 already ran
        inline in `__init__`. Without this guard, a caller that (accidentally,
        or via test/dependency-injection wiring) invokes this on an
        already-`fully_ready` service would reconstruct the indexer/searcher/
        watcher a second time and leak the discarded watcher's background
        threads.

        Also a no-op after shutdown() (UPG-SHUTDOWN-INIT-RACE): a service
        that was shut down during the warm-up window must not go on to
        construct the search layer it can no longer release.
        """
        if self._shutdown_requested.is_set():
            return
        if self._fully_ready.is_set():
            return
        self._init_search_layer()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def memory_only(self) -> bool:
        """True when this daemon runs in memory-only mode (no indexing/watcher)."""
        return self._memory_only

    @property
    def search_only(self) -> bool:
        """True when this daemon runs in search-only mode (no working-memory layer)."""
        return self._search_only

    @property
    def fully_ready(self) -> bool:
        """True once phase 2 (embedder/indexer/searcher/watcher/symbol-graph
        construction) has completed (UPG-STDIO-MEMORY-READY). Search-touching
        tools/endpoints (search/locate/trace/map/fetch/evict-hint) are gated
        on this across every transport. Memory tools are never gated on it —
        they work from phase 1. Always True immediately once a
        `defer_search_init=False` (the default) construction returns."""
        return self._fully_ready.is_set()

    @property
    def embedder_ready(self) -> bool:
        """True once the embedding model has loaded and been attached to the
        working-memory store (UPG-STDIO-MEMORY-READY) — flips before the rest
        of phase 2 (searcher/watcher/symbol-graph) finishes. Used to gate the
        vectr_recall lexical-fallback notice: state-based only, never
        query-content-based. Always True immediately once a
        `defer_search_init=False` (the default) construction returns."""
        return self._embedder_ready.is_set()

    def start_background_index(self) -> None:
        """Kick off workspace indexing in a background thread.

        In memory-only mode, the index thread and file watcher are skipped.
        In search-only mode, indexing and the watcher run normally, but the
        two startup memory-maintenance passes below are skipped — there is
        no context store to run them against. Otherwise both run:
        decay_old_notes() (UPG-MEMORY-DECAY-KIND-SCOPED) unconditionally —
        it never deletes or hides anything, only recomputes the ranking-only
        decay_score kind-scoped from each note's age — and
        purge_expired_notes() only when VECTR_NOTES_TTL_DAYS is set, same as
        before this fix (it is the one of the two with an observable effect
        on default recall()/fire() listings, so it stays opt-in).
        """
        if self._watcher is None:
            raise RuntimeError(
                "start_background_index() called before phase 2 completed — "
                "call complete_search_init() first (see defer_search_init)"
            )
        if self._indexing:
            return
        self._indexing = True

        if self._context_store is not None:
            # decay_old_notes() is non-destructive (ranking signal only —
            # see UPG-MEMORY-DECAY-KIND-SCOPED) so it runs unconditionally,
            # unlike the TTL purge below which stays behind an explicit
            # opt-in env var since it changes what's visible by default.
            try:
                decayed = self._context_store.decay_old_notes(self._workspace_root)
                if decayed:
                    logger.info("recomputed decay_score for %d notes at startup", decayed)
            except Exception:
                logger.exception("decay_old_notes failed at startup")

            # apply TTL to working notes at startup if VECTR_NOTES_TTL_DAYS is set.
            ttl_days_str = os.getenv("VECTR_NOTES_TTL_DAYS", "")
            if ttl_days_str:
                try:
                    ttl = float(ttl_days_str)
                except ValueError:
                    logger.warning("VECTR_NOTES_TTL_DAYS is not a valid float: %r", ttl_days_str)
                else:
                    try:
                        expired = self._context_store.purge_expired_notes(self._workspace_root, ttl)
                        if expired:
                            logger.info("flagged %d notes expired (TTL=%.1f days)", expired, ttl)
                    except Exception:
                        logger.exception("purge_expired_notes failed at startup")

        if self._memory_only:
            logger.info(
                "memory-only mode: code indexing + file watcher disabled; "
                "working-memory tools + hooks active"
            )
            self._indexing = False
            return

        self._index_thread = threading.Thread(target=self._do_index, daemon=True)
        self._index_thread.start()
        self._watcher.start()

    def _do_index(self) -> None:
        with self._index_lock:
            try:
                logger.info("Starting workspace index...")
                files, chunks = self._indexer.index_workspace()
                self._searcher.refresh_bm25()
                logger.info("Indexed %d files → %d chunks", files, chunks)

                # audit index event
                from agent.working_context_store import audit as _audit
                _audit("INDEX", workspace=self._workspace_root, files=files, chunks=chunks)

                self._build_symbol_graph()
                self._refresh_strategy()

            except Exception:
                logger.exception("Indexing failed")
            finally:
                self._indexing = False

    def _refresh_strategy(self) -> None:
        try:
            from agent.strategy_selector import fingerprint, select_strategy
            fp = fingerprint(self._workspace_root, self._indexer.indexed_file_paths)
            self._strategy = select_strategy(fp)
            logger.info(
                "Retrieval strategy: sem=%.2f bm25=%.2f graph_first=%s — %s",
                self._strategy.semantic_weight,
                self._strategy.bm25_weight,
                self._strategy.graph_first,
                self._strategy.rationale,
            )
        except Exception:
            logger.exception("Strategy selection failed (non-fatal)")

    def _build_symbol_graph(self) -> None:
        """Rebuild the symbol graph from the files the indexer already walked."""
        self._symbol_graph_rebuilding.set()
        try:
            from agent.symbol_graph import SYMBOL_LANGUAGES, available_symbol_languages
            # Warn loudly when a declared grammar is not importable in this
            # environment so the degraded state is never silent.
            missing = sorted(SYMBOL_LANGUAGES - available_symbol_languages())
            if missing:
                logger.warning(
                    "tree-sitter grammar(s) not importable: %s — locate/trace disabled for "
                    "these; reinstall vectr deps (pip install -e .) to enable",
                    ", ".join(missing),
                )

            # Reuse the indexer's file list — avoids a second expensive walk and
            # guarantees the symbol graph covers exactly the same files as the
            # vector index (same filters, same exclusions).
            file_paths = self._indexer.indexed_file_paths
            # UPG-8.7: a toolchain change (vectr upgrade / new parser / model)
            # means the persisted graph is stale; we always full-rebuild here, but
            # surface *why* so a stale graph is never silently served.
            if self._symbol_graph.is_stale(self._workspace_root, self._embed_model):
                logger.info("Symbol graph stale or toolchain changed — full rebuild")
            stats = self._symbol_graph.build_for_workspace(
                self._workspace_root, file_paths, embed_model=self._embed_model,
            )
            logger.info(
                "Symbol graph: %d symbols, %d edges across %d files (%d failed, complete=%s)",
                stats["symbols"], stats["edges"], stats["files"],
                stats["failed"], stats["complete"],
            )
            # ARCH-1b: hand the freshly-computed file-level importance (ARCH-1a) to
            # the searcher so the ranking prior reflects the current graph.
            self._searcher.set_file_importance(
                self._symbol_graph.file_importance(self._workspace_root)
            )
            # ARCH-2: same for the freshly-computed class-level importance.
            self._searcher.set_class_importance(
                self._symbol_graph.class_importance(self._workspace_root)
            )
            # UPG-TESTPATH-FRAMEWORK-MISCLASS (F58): same for the freshly-computed
            # test-framework fan-in exemption map.
            self._searcher.set_file_fan_in(
                self._symbol_graph.file_fan_in(self._workspace_root)
            )
        except Exception:
            logger.exception("Symbol graph build failed (non-fatal)")
        finally:
            self._symbol_graph_rebuilding.clear()

    def save_map(self, summary: str, overwrite: bool = False) -> dict:
        """
        Persist an AI-written codebase passport.
        Called via vectr_map_save — the AI editor has synthesised the summary
        after reading the raw metadata returned by vectr_map on first call.

        Does NOT silently overwrite an existing passport (UPG-6.2): if one is
        already saved and `overwrite` is not set, the write is a no-op and the
        existing summary is returned so the caller can decide whether to
        retry with `overwrite=True`.

        Returns {"saved": bool, "existing_summary": str | None}.
        """
        existing = self._passport_store.load()
        if existing and existing.get("summary") and not overwrite:
            logger.info("Passport save skipped — one already exists and overwrite was not set")
            return {"saved": False, "existing_summary": existing["summary"]}
        self._passport_store.save_summary(summary, self._workspace_root)
        logger.info("Passport saved by AI editor (%d chars)", len(summary))
        return {"saved": True, "existing_summary": None}

    def shutdown(self) -> None:
        # UPG-STDIO-MEMORY-READY: a service constructed with
        # defer_search_init=True can be asked to shut down (e.g. stdin EOF)
        # while still in the phase-2 background-construction window, before
        # the watcher/indexer exist — nothing to stop/close yet, not an error.
        # UPG-SHUTDOWN-INIT-RACE: the event must be set BEFORE the
        # None-checks below — a phase 2 that assigns watcher/indexer after
        # this read then sees the event at its own end and tears them down
        # itself; whichever side observes the other's write cleans up.
        self._shutdown_requested.set()
        if self._watcher is not None:
            self._watcher.stop()
        # Release the indexer's ChromaDB client (and the notes store's, which
        # shares the same underlying client — see attach_embedder above).
        # Without this, every VectrService that ever reaches phase 2 leaks a
        # native worker-thread pool for the rest of the process's life.
        if self._indexer is not None:
            self._indexer.close()
        # Release the dedicated search/fetch vector-store-dispatch executor
        # — created unconditionally in phase 1, so always present regardless
        # of how far phase 2 got. Invoked through its finalizer (see
        # __init__) rather than calling shutdown() directly so an explicit
        # call here and the garbage-collection safety net can never both
        # run: whichever fires first wins, the other is then a no-op.
        # cancel_futures=True: any call still queued (not yet started) is
        # dropped rather than waited out during shutdown; wait=False so
        # shutdown itself never blocks on a call already in flight.
        self._chroma_executor_finalizer()

    # ------------------------------------------------------------------
    # Call counters — all callers (parent + sub-agents) hit the same server,
    # so these are accurate totals regardless of how many agents spawned.
    # ------------------------------------------------------------------

    def increment_call_count(self, tool_name: str) -> None:
        with self._call_counts_lock:
            self._call_counts[tool_name] = self._call_counts.get(tool_name, 0) + 1

    def get_call_counts(self) -> dict[str, int]:
        with self._call_counts_lock:
            return dict(self._call_counts)

    def reset_call_counts(self) -> dict[str, int]:
        with self._call_counts_lock:
            old = dict(self._call_counts)
            self._call_counts.clear()
            return old

    # ------------------------------------------------------------------
    # Hook injection counters (UPG-HOOK-INJECT-OBSERVABILITY): recorded from
    # `recall()` below, only when a hook-declared call actually returned
    # notes. A working memory system and a dead one look identical to the
    # human until this makes injection visible — read via `status()`.
    # ------------------------------------------------------------------

    def _record_hook_injection(self, hook_event: str, notes: str) -> None:
        """Count one injection for `hook_event`, and optionally log it.

        No-op when `notes` is empty — an empty-result hook call (e.g. a
        SessionStart on a fresh workspace with no directives yet) injected
        nothing, so it must not count as an injection.
        """
        if not notes:
            return
        with self._hook_injection_lock:
            self._hook_injection_counts[hook_event] = (
                self._hook_injection_counts.get(hook_event, 0) + 1
            )
        if HOOKS_LOG_INJECTIONS:
            self._append_hook_injection_log(hook_event, notes)

    def _append_hook_injection_log(self, hook_event: str, notes: str) -> None:
        """Append one line to ~/.vectr/logs/<workspace-hash>.hooks.log.

        Failure here must never break recall — any error is logged and
        swallowed, never raised.
        """
        try:
            from agent.instance_registry import workspace_hash
            log_dir = Path.home() / ".vectr" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{workspace_hash(self._workspace_root)}.hooks.log"
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            tokens = len(notes) // HOOKS_LOG_CHARS_PER_TOKEN
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{ts}\t{hook_event}\ttokens={tokens}\n")
        except Exception:
            logger.exception("failed to write hook injection log")

    def get_hook_injection_counts(self) -> dict[str, int]:
        with self._hook_injection_lock:
            return dict(self._hook_injection_counts)

    # ------------------------------------------------------------------
    # L3 — search and index operations
    # ------------------------------------------------------------------

    def index(self, path: str, force: bool = False) -> tuple[int, int, int]:
        """Index a path. Returns (files_indexed, total_chunks, elapsed_ms)."""
        with self._index_lock:
            t0 = time.monotonic()
            target = Path(path).resolve()
            if target.is_file():
                self._indexer.index_file(str(target))
                self._searcher.refresh_bm25()
                self._symbol_graph.index_file(self._workspace_root, str(target))
                files, chunks = 1, self._indexer.total_chunks
            else:
                files, chunks = self._indexer.index_workspace(force=force)
                self._searcher.refresh_bm25()
                self._build_symbol_graph()
                self._refresh_strategy()
            elapsed = int((time.monotonic() - t0) * 1000)
            from agent.working_context_store import audit as _audit
            _audit("INDEX", workspace=self._workspace_root, files=files, chunks=chunks)
            return files, chunks, elapsed

    def _advisor_for(self, session_id: str | None) -> EvictionAdvisor:
        """Look up (or create) the EvictionAdvisor for the calling session
        (UPG-EVICT-SESSION-SCOPE). `session_id=None` (REST callers, or an MCP
        transport that never sent one) shares one daemon-global advisor —
        unchanged backwards-compat behaviour. A known session_id gets its own
        advisor so it never sees chunks another session retrieved. The
        registry is LRU-bounded to EVICTION_MAX_TRACKED_SESSIONS so a
        long-running daemon serving many short-lived sessions can't grow it
        without bound."""
        if not session_id:
            return self._eviction_advisor
        advisor = self._session_advisors.get(session_id)
        if advisor is None:
            if len(self._session_advisors) >= EVICTION_MAX_TRACKED_SESSIONS:
                oldest_id = next(iter(self._session_advisors))
                del self._session_advisors[oldest_id]
            advisor = EvictionAdvisor(eviction_threshold_tokens=self._eviction_threshold_tokens)
            self._session_advisors[session_id] = advisor
        else:
            # Refresh recency for the LRU bound above.
            del self._session_advisors[session_id]
            self._session_advisors[session_id] = advisor
        return advisor

    def _ledger_for(self, session_id: str | None) -> TriggerFireLedger | None:
        """Look up (or create) the per-session `TriggerFireLedger` (TRIGGER-
        ENGINE wave 1, bm2-design-skeleton.md §3), mirroring `_advisor_for`
        immediately above. `session_id=None` returns None rather than a
        shared fallback ledger — fire-dedup only makes sense within one
        session's identity; a caller with no session_id gets no suppression
        at all, never cross-session suppression. LRU-bounded to the same
        EVICTION_MAX_TRACKED_SESSIONS cap for the same reason as the advisor
        registry (a long-running daemon serving many short-lived sessions)."""
        if not session_id:
            return None
        ledger = self._trigger_ledgers.get(session_id)
        if ledger is None:
            if len(self._trigger_ledgers) >= EVICTION_MAX_TRACKED_SESSIONS:
                oldest_id = next(iter(self._trigger_ledgers))
                del self._trigger_ledgers[oldest_id]
            ledger = TriggerFireLedger()
            self._trigger_ledgers[session_id] = ledger
        else:
            del self._trigger_ledgers[session_id]
            self._trigger_ledgers[session_id] = ledger
        return ledger

    def reset_trigger_ledger(self, session_id: str | None) -> None:
        """Reset one session's fire-dedup ledger (TRIGGER-ENGINE
        bm2-design-skeleton.md §3: "cleared on compaction"). A caller with no
        session_id, or a session that has never fired a trigger yet, is a
        no-op."""
        if session_id and session_id in self._trigger_ledgers:
            self._trigger_ledgers[session_id].reset()

    def _turn_ledger_for(self, session_id: str | None) -> TurnInjectionLedger | None:
        """Look up (or create) the per-session `TurnInjectionLedger`
        (TRIGGER-ENGINE wave 3, §5.3/§5.4), mirroring `_ledger_for` above.
        `session_id=None` returns None rather than a shared fallback ledger
        — same rationale as `_ledger_for`: cross-surface dedup only makes
        sense within one session's identity."""
        if not session_id:
            return None
        ledger = self._turn_ledgers.get(session_id)
        if ledger is None:
            if len(self._turn_ledgers) >= EVICTION_MAX_TRACKED_SESSIONS:
                oldest_id = next(iter(self._turn_ledgers))
                del self._turn_ledgers[oldest_id]
            ledger = TurnInjectionLedger()
            self._turn_ledgers[session_id] = ledger
        else:
            del self._turn_ledgers[session_id]
            self._turn_ledgers[session_id] = ledger
        return ledger

    def _existing_turn_ledger(self, session_id: str | None) -> TurnInjectionLedger | None:
        """Look up a per-session `TurnInjectionLedger` WITHOUT ever
        allocating one (UPG-PROXY-CROSS-CHANNEL-DEDUP).

        `proactive_context` (the proxy channel) uses this instead of
        `_turn_ledger_for` above. The proxy channel's session_id is a
        per-process instance id minted once by `VectrProxy.__init__`
        (`agent/proactive/proxy.py`'s `self._instance_id`, prefixed
        `proxy-` — see UPG-PROXY-COOLDOWN-SESSION-IDENTITY) and is, by
        construction, disjoint from the editor-supplied session id the
        hook/trigger-engine surfaces use: the two namespaces are never
        meant to collide, so this stays a guaranteed lookup miss on the
        proxy path. If this allocated a ledger on miss instead, one entry
        per proxy PROCESS (not per conversation, so far less churn than the
        old per-first-message identity, but still a distinct process-scoped
        id) would consume a slot in the `EVICTION_MAX_TRACKED_SESSIONS`-
        bounded `_turn_ledgers` LRU and could evict a live editor session's
        real `TurnInjectionLedger` mid-turn — the exact double-dip
        `TurnInjectionLedger` exists to prevent (see the G3 regression
        test). A lookup-only accessor costs nothing when the namespaces
        don't collide, and starts deduping for free the moment session
        identity is unified, with no further code change here."""
        if not session_id:
            return None
        return self._turn_ledgers.get(session_id)

    def reset_turn_ledger(self, session_id: str | None) -> None:
        """Reset one session's per-turn dedup/budget ledger — called from
        `recall()` at every UserPromptSubmit (§5.3: "reset at each
        UserPromptSubmit"), never at PreCompact (that resets
        `self._trigger_ledgers` above instead, a different lifecycle). A
        caller with no session_id, or a session with no turn ledger yet, is
        a no-op."""
        if session_id and session_id in self._turn_ledgers:
            self._turn_ledgers[session_id].reset()

    def fire_triggers(
        self,
        event: str | None = None,
        file_path: str | None = None,
        session_id: str | None = None,
    ):
        """Live per-memory trigger evaluation (TRIGGER-ENGINE) for one
        lifecycle moment. Returns the ordered list of `agent.trigger_engine
        .FireResult` for every note that fires — see
        `WorkingContextStore.fire()` for the full contract (evaluation,
        staleness caveats folded in, the one shared total order, per-session
        dedup via `_ledger_for`, scope enforcement, and the `last_fired`
        cooldown stamp).

        Raw evaluation entry point — returns `FireResult`s, not rendered
        text; `fire_and_recall()` below is the rendering wrapper the live
        hook pipeline (main.py's `cmd_hook`) actually calls. Kept as a
        separate, still-callable capability for any caller that wants the
        unrendered results directly."""
        self._require_memory_layer()
        return self._context_store.fire(
            self._workspace_root,
            event=event,
            file_path=file_path,
            ledger=self._ledger_for(session_id),
            session_id=session_id,
        )

    def fire_and_recall(
        self,
        event: str | None = None,
        events: list[str] | None = None,
        file_path: str | None = None,
        session_id: str | None = None,
        surface: str = "mcp",
        hook_event: str | None = None,
    ) -> str:
        """Rendering wrapper around `WorkingContextStore.fire_and_format()`
        (TRIGGER-ENGINE wave 2a) — evaluate one or more lifecycle events,
        pack the fired notes against the per-session cumulative injection
        budget, and return the rendered text ready for hook injection.

        `hook_event`, if given, records the delivery via `_record_hook_injection`
        (UPG-HOOK-INJECT-OBSERVABILITY) — the same counter `recall()`'s own
        `hook_event` parameter feeds — so a trigger-fired delivery shows up
        in `status()`'s hook injection counts exactly like a semantic-recall
        hook delivery does."""
        self._require_memory_layer()
        text, _ = self._context_store.fire_and_format(
            self._workspace_root,
            event=event,
            events=events,
            file_path=file_path,
            session_id=session_id,
            ledger=self._ledger_for(session_id),
            surface=surface,
        )
        if hook_event is not None:
            self._record_hook_injection(hook_event, text)
        return text

    def search(
        self, query: str, n_results: int = 10, language: str | None = None
    ) -> tuple[list, int]:
        """Returns (SearchResult list, query_time_ms).

        Does NOT record into any eviction advisor (UPG-EVICT-SESSION-SCOPE) —
        recording happens exactly once, at render time, against the calling
        session's advisor (see MCP dispatch's vectr_search handler /
        `record_results`). A REST `/v1/search` caller has no session_id and
        gets pure retrieval with no eviction-tracking side effect.
        """
        from agent.working_context_store import audit as _audit

        # Org-wide artifact cache (UPG-PRO): identical query against the same
        # code index -> identical results. Keyed by the code index epoch, so a
        # re-index invalidates. Off unless proactive.cache is enabled. The
        # "what was queried" audit still fires on a hit (the query was issued),
        # but query_ms is reported 0 to reflect that no search actually ran.
        cache = self._artifact_cache
        key = ""
        if cache is not None:
            from agent.proactive.cache import canonical_key
            key = canonical_key(
                "search",
                {"query": query, "n_results": n_results, "language": language},
                self._index_epoch("code"),
            )
            found, cached = cache.get(key)
            if found:
                results, _ = cached
                _audit("SEARCH", workspace=self._workspace_root,
                       query=query[:200], results=len(results))
                return results, 0

        sem_w = self._strategy.semantic_weight if self._strategy else STRATEGY_DEFAULT_SEMANTIC_WEIGHT
        results, query_ms = self._searcher.search(
            query, n_results=n_results, language=language, semantic_weight=sem_w
        )
        # Audit "what was queried" (opt-in; the query text is the whole point of
        # an audit log, so it is only recorded when the operator enables one).
        _audit(
            "SEARCH", workspace=self._workspace_root,
            query=query[:200], results=len(results),
        )
        if cache is not None:
            cache.put(key, (results, query_ms))
        return results, query_ms

    def record_results(self, results: list, session_id: str | None = None) -> None:
        """Record rendered SearchResult chunks into the calling session's
        eviction advisor (UPG-EVICT-SESSION-SCOPE). Call with the exact list
        that was serialized into the tool response — not the pre-truncation
        candidate pool."""
        self._advisor_for(session_id).record_results(results)

    def record_chunk(
        self, *, file_path: str, lines: str, symbol_name: str, content: str,
        chunk_id: str = "", session_id: str | None = None,
    ) -> None:
        """Record one rendered chunk (vectr_fetch, vectr_locate snippets) into
        the calling session's eviction advisor (UPG-EVICT-SESSION-SCOPE)."""
        self._advisor_for(session_id).record(
            file_path=file_path, lines=lines, symbol_name=symbol_name,
            content=content, chunk_id=chunk_id,
        )

    def note_remembered(self, session_id: str | None = None) -> None:
        """Tell the calling session's eviction advisor that vectr_remember
        just succeeded (UPG-REMEMBER-BANNER-FATIGUE) — resets its
        chunks-since-last-remember counter so the escalated ACTION REQUIRED
        directive doesn't immediately re-fire on the next retrieval."""
        self._advisor_for(session_id).note_remembered()

    def indexed_languages(self) -> list[str]:
        """Distinct languages actually present in the index (UPG-3.1)."""
        return self._indexer.indexed_languages()

    def fetch(self, ids: list[str]) -> list[dict]:
        """Deterministic re-fetch by chunk id (UPG-CTX-EVICT part a).

        Disabled in memory-only mode — there is no code index to fetch from,
        the same guard as search/locate/trace.

        UPG-RELATIVE-PATH-RENDER: search/evict now render workspace-RELATIVE
        chunk ids, but the index stores ABSOLUTE ids. Resolve each incoming id
        against the workspace root before lookup — a relative id is joined onto
        the root, an already-absolute id passes through unchanged — so both the
        new relative ids and the absolute ids existing sessions still hold
        fetch correctly. The returned entry's file_path/lines come from the
        stored metadata (absolute), which the renderer re-relativizes.
        """
        if self._memory_only:
            raise RuntimeError(_MEMORY_ONLY_MSG)
        from agent.render_paths import resolve_chunk_id
        resolved = [resolve_chunk_id(i, self._workspace_root) for i in ids]
        return self._indexer.fetch_chunks(resolved)

    def identifier_hint_symbols(self, query: str) -> list:
        """Additive, high-precision symbol-graph hint (UPG-QUERYTYPE-REROUTE).

        Scans the RAW query for identifier-SHAPED tokens (CamelCase,
        snake_case, dotted `Class.method` — a structural transform, never a
        keyword/intent classification) and attempts EXACT symbol-graph
        resolution for each, capped at `search.identifier_hint.max_identifiers`
        candidate tokens. No fuzzy/edit-distance/prefix matching — only an
        exact-name (or exact class-qualified) hit counts, the same exactness
        `locate` uses for a full-name hit. Returns a flat, deduplicated list of
        resolved `Symbol` objects in query order, each capped to
        `search.identifier_hint.max_locations` — empty when the query has no
        identifier-shaped token or none resolves exactly. This never adjusts
        search weights or reorders/prepends anything; it is a pure addition
        appended below the L3 results by the MCP dispatch layer.
        """
        if not SEARCH_IDENTIFIER_HINT_ENABLED:
            return []
        from agent.identifier_hint import extract_identifier_tokens

        tokens = extract_identifier_tokens(query)[:SEARCH_IDENTIFIER_HINT_MAX_IDENTIFIERS]
        resolved: list = []
        seen_ids: set[int] = set()
        for token in tokens:
            result = self._symbol_graph.locate_l2(
                self._workspace_root, token, limit=SEARCH_IDENTIFIER_HINT_MAX_LOCATIONS
            )
            if result.resolution_strategy != "exact":
                continue
            for sym in result.symbols[:SEARCH_IDENTIFIER_HINT_MAX_LOCATIONS]:
                if sym.symbol_id in seen_ids:
                    continue
                seen_ids.add(sym.symbol_id)
                resolved.append(sym)
        return resolved

    def identifier_hint_nearmiss(self, query: str) -> list[tuple[str, list]]:
        """Additive, honestly-labeled near-miss hint (UPG-NEARMISS-SYMBOL-NAMES).

        For each identifier-shaped token attempted by `identifier_hint_symbols`
        (same tokenizer, same `max_identifiers` cap) that does NOT resolve to
        an exact symbol-graph hit, looks up the nearest existing symbol NAMES
        via `SymbolGraph.nearest_symbol_names` — the symbol graph's own cheap,
        deterministic partial-match machinery, never a semantic guess. A token
        that resolves exactly, or that has no near-miss candidate at all, is
        simply absent from the returned list (matching the "nothing appears"
        behaviour of `identifier_hint_symbols`). Capped at
        `search.identifier_hint.nearmiss_max` distinct names TOTAL across the
        whole response, not per token, so the addition stays small. Returns a
        list of `(token, [Symbol, ...])` pairs; every symbol here is inexact
        by construction — the caller (MCP dispatch) must label it as a
        near-miss, never present it as a match.
        """
        if not SEARCH_IDENTIFIER_HINT_ENABLED or not SEARCH_IDENTIFIER_HINT_NEARMISS_ENABLED:
            return []
        from agent.identifier_hint import extract_identifier_tokens

        tokens = extract_identifier_tokens(query)[:SEARCH_IDENTIFIER_HINT_MAX_IDENTIFIERS]
        pairs: list[tuple[str, list]] = []
        budget = SEARCH_IDENTIFIER_HINT_NEARMISS_MAX
        for token in tokens:
            if budget <= 0:
                break
            exact = self._symbol_graph.locate_l2(
                self._workspace_root, token, limit=SEARCH_IDENTIFIER_HINT_MAX_LOCATIONS
            )
            if exact.resolution_strategy == "exact":
                continue
            near = self._symbol_graph.nearest_symbol_names(
                self._workspace_root, token, limit=budget
            )
            if not near:
                continue
            pairs.append((token, near))
            budget -= len(near)
        return pairs

    @property
    def last_indexed(self) -> str:
        """Single source of truth for the last-indexed timestamp string.

        Used by both `status()` and the `/v1/health` route so the two
        endpoints never disagree on freshness (UPG-8.2).

        "never" before phase 2 has built the indexer (UPG-STDIO-MEMORY-READY)
        — truthful, not an error: nothing has been indexed yet.
        """
        last_ts = self._indexer.last_indexed_ts if self._indexer is not None else None
        return (
            datetime.fromtimestamp(last_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            if last_ts
            else "never"
        )

    def status(self) -> dict:
        from agent.symbol_graph import SYMBOL_LANGUAGES, available_symbol_languages
        # Retrieval weights + strategy fields are always present (UPG-8.2) —
        # before the first index-time fingerprint (`self._strategy`) has run,
        # fall back to the config-declared defaults rather than omitting the
        # fields. This keeps `status` output deterministic in shape from the
        # very first call, across both REST and MCP.
        if self._strategy:
            strategy_info = {
                "semantic_weight": self._strategy.semantic_weight,
                "bm25_weight": self._strategy.bm25_weight,
                "graph_first": self._strategy.graph_first,
                "recommended_embed_model": self._strategy.recommended_embed_model,
                "strategy_rationale": self._strategy.rationale,
            }
        else:
            strategy_info = {
                "semantic_weight": STRATEGY_DEFAULT_SEMANTIC_WEIGHT,
                "bm25_weight": STRATEGY_DEFAULT_BM25_WEIGHT,
                "graph_first": False,
                "recommended_embed_model": self._embed_model,
                "strategy_rationale": "default weights — no workspace fingerprint yet, index the workspace to compute one",
            }
        missing = sorted(SYMBOL_LANGUAGES - available_symbol_languages())
        if self._memory_only:
            mode = "memory-only"
        elif self._search_only:
            mode = "search-only"
        else:
            mode = "full"
        # UPG-REST-STARVATION requirement #2: a truthful, lock-free signal
        # that bulk index work is running right now — `Lock.locked()` is a
        # non-blocking read (never acquires), and `watcher_status()` reads
        # only in-memory counters/flags, so neither call can itself wait on
        # the bulk work it's reporting about. True while either an explicit
        # `index()`/startup index holds `_index_lock`, or the watcher's
        # coalesced batch worker is actively re-indexing/de-indexing.
        # UPG-STDIO-MEMORY-READY: before phase 2 has built the watcher, there
        # is truthfully nothing pending/running yet — the same all-False/zero
        # shape `CodeWatcher.watcher_status()` itself returns for a quiet
        # workspace, so `status()`'s output shape never changes across the
        # phase-1/phase-2 boundary.
        watcher_status = (
            self._watcher.watcher_status() if self._watcher is not None else {
                "watcher_burst_mode": False,
                "watcher_pending_files": 0,
                "watcher_batch_running": False,
                "watcher_last_batch_duration_ms": 0,
            }
        )
        reindex_in_progress = (
            self._index_lock.locked() or watcher_status.get("watcher_batch_running", False)
        )
        # UPG-TASK-SUPERSEDES-HYGIENE: a stale-task nudge, not a lifecycle
        # change — surfaced unconditionally so the caller (MCP handler / REST
        # consumer) decides whether the count clears the warn threshold.
        stale_task_count, stale_task_oldest_id = self.stale_task_summary()
        return {
            "indexed_files": self._indexer.indexed_file_count if self._indexer else 0,
            "total_chunks": self._indexer.total_chunks if self._indexer else 0,
            "last_indexed": self.last_indexed,
            "embed_model": self._embed_model,
            "workspace_root": self._workspace_root,
            "symbol_count": (
                self._symbol_graph.symbol_count(self._workspace_root)
                if self._symbol_graph is not None else 0
            ),
            "languages": self._language_coverage(),
            "notes_count": self.count_notes(),
            # UPG-TASK-SUPERSEDES-HYGIENE: count of live kind="task" notes
            # older than MEMORY_HYGIENE_STALE_TASK_WARN_AGE_DAYS, plus the
            # oldest such note's id (None if there are none). The MCP/REST
            # surfaces append a one-line nudge once count clears
            # MEMORY_HYGIENE_STALE_TASK_WARN_COUNT.
            "stale_task_count": stale_task_count,
            "stale_task_oldest_id": stale_task_oldest_id,
            "grammars_unavailable": missing,
            "mode": mode,
            "version_stamp": self._version_stamp,
            "reindex_in_progress": reindex_in_progress,
            # UPG-PURPOSE-PASS-DEFERRAL: True while a background purpose-
            # vector pass is scheduled/running (config `defer_purpose_pass`,
            # default on) — search still returns results during this window
            # (body-similarity only, `query_vector_purpose`'s empty/partial-
            # collection guard), so this makes a possible quality dip
            # explainable instead of silent. Always False when the indexer
            # hasn't been constructed yet, or when deferral never ran
            # (disabled, or `--foreground-fast`).
            "purpose_vectors_pending": (
                self._indexer.purpose_vectors_pending if self._indexer else False
            ),
            # UPG-NOTES-EMBED-MIGRATION: normally None — migration runs
            # synchronously at startup, so this only surfaces a mid-failure
            # state (e.g. the embedder was unavailable during migration).
            "notes_embed_model_mismatch": (
                self._context_store.embed_model_stamp_mismatch()
                if self._context_store is not None else None
            ),
            **self._symbol_graph_status(),
            **strategy_info,
            **watcher_status,
            "hook_injection_counts": self.get_hook_injection_counts(),
            "proactive_injection_counts": self.get_proactive_injection_counts(),
            # Effective ambient (hook-channel) master opt-in, visible BEFORE any
            # injection has happened — the counts above only appear after the
            # fact. The proxy channel injects by launch consent regardless.
            "proactive_enabled": self._proactive_master_enabled(),
            "artifact_cache": self.cache_metrics(),
            # UPG-STDIO-MEMORY-READY: additive warm-up signals. Both True
            # immediately for every existing (non-deferred) caller — a
            # deferred caller (stdio/HTTP daemon) sees `fully_ready=False`
            # (and possibly `embedder_ready=False`) only during the phase-2
            # background-construction window.
            "fully_ready": self.fully_ready,
            "embedder_ready": self.embedder_ready,
            # Episode capture (spec §7 gate G2): a plain count, never the
            # rows themselves — status is not a reader of episode content,
            # only GET /v1/episodes is.
            "episodes_count": self.count_episodes(),
            # Best-effort: 0 until the arc detector's own table exists (see
            # `EpisodeStore.count_arcs_pending_distill`).
            "arcs_pending_distill": self.count_arcs_pending_distill(),
        }

    @staticmethod
    def _proactive_master_enabled() -> bool:
        """Effective ambient master opt-in as `/v1/status` should report it
        (UPG-PROXY-STATUS-TRUE-STATE): the config flag ANDed with the
        unconditional bind gate. On a non-loopback bind `proactive_context()`
        refuses every channel regardless of config, so reporting the bare
        config value there would claim injection is on where it is in fact
        refused."""
        from agent.proactive.settings import ProactiveSettings, proactive_bind_is_loopback
        return ProactiveSettings.from_env().enabled and proactive_bind_is_loopback()

    def _symbol_graph_status(self) -> dict:
        """Symbol-graph build trust signals for `status` (UPG-8.7): whether the
        persisted graph is complete (no files failed extraction) and built by the
        current toolchain. Lets a benchmark/user confirm locate/trace coverage is
        trustworthy rather than a silently-partial graph.

        Returns the same "nothing built yet" shape before phase 2 has
        constructed the symbol graph (UPG-STDIO-MEMORY-READY) as it does for
        a workspace that has never been indexed.

        UPG-STATUS-STALE-GRAPH: also reports whether the graph being served is
        stale and whether a rebuild is overwriting it right now, so a caller (or
        a gate preflight) never mistakes a persisted stale-toolchain graph for
        current state. `symbol_graph_stale` is true when the persisted graph was
        built by a different toolchain (vectr/parser/model/schema change) or was
        left incomplete, or has not been built yet. `symbol_graph_rebuild_in_progress`
        is true while _build_symbol_graph() is running — during that window the
        symbol_count / symbol_graph_complete above still reflect the OLD persisted
        graph until the rebuild's write lands."""
        rebuilding = self._symbol_graph_rebuilding.is_set()
        if self._symbol_graph is None:
            return {
                "symbol_graph_complete": False,
                "symbol_graph_failed_files": 0,
                "symbol_graph_stale": True,
                "symbol_graph_rebuild_in_progress": rebuilding,
            }
        stale = self._symbol_graph.is_stale(self._workspace_root, self._embed_model)
        meta = self._symbol_graph.graph_meta(self._workspace_root)
        if not meta:
            return {
                "symbol_graph_complete": False,
                "symbol_graph_failed_files": 0,
                "symbol_graph_stale": stale,
                "symbol_graph_rebuild_in_progress": rebuilding,
            }
        return {
            "symbol_graph_complete": meta.get("complete") == "1",
            "symbol_graph_failed_files": int(meta.get("failed", "0") or "0"),
            "symbol_graph_stale": stale,
            "symbol_graph_rebuild_in_progress": rebuilding,
        }

    def _language_coverage(self) -> list[dict]:
        """Per-language coverage + symbol availability for `status` (UPG-3.3).

        Lets the caller LLM route: `symbols=True` languages support locate/trace;
        the rest are search-only. A language declared in SYMBOL_LANGUAGES is only
        marked symbols=True when its tree-sitter grammar actually loads in this
        environment (grammar_available check) — prevents advertising locate/trace
        for a language whose grammar package is missing.
        Ordered by file count (dominant language first).

        Empty before phase 2 has built the indexer (UPG-STDIO-MEMORY-READY) —
        nothing has been walked/indexed yet, same as an empty workspace.
        """
        if self._indexer is None:
            return []
        from agent.symbol_graph import supports_symbols, grammar_available
        stats = self._indexer.indexed_language_stats()
        return [
            {
                "language": lang,
                "files": s["files"],
                "chunks": s["chunks"],
                "symbols": supports_symbols(lang) and grammar_available(lang),
            }
            for lang, s in sorted(
                stats.items(), key=lambda kv: (-kv[1]["files"], kv[0])
            )
        ]

    @property
    def total_chunks(self) -> int:
        return self._indexer.total_chunks

    # ------------------------------------------------------------------
    # L1 — codebase passport
    # ------------------------------------------------------------------

    def get_map(self) -> str:
        """
        Return codebase passport for AI consumption.
        If cached: instant ~300-token summary.
        If not: raw structural metadata + instruction to call vectr_map_save.

        Raw-metadata language detection uses the indexer's real per-language
        coverage (UPG-6.1) rather than a directory-walk extension guess.

        Raw-metadata import-graph/module-community detection is built only
        over the indexer's real file list (`indexed_file_paths`), forwarded
        here the same way — never an unbounded directory walk that could
        descend into an in-repo virtualenv (UPG-MAP-VENV-WALK).

        Guarded for the phase-2 warm-up window: every transport gates
        non-memory tools on `fully_ready` before reaching this method
        (routes.py's `_require_fully_ready`, mcp_server/_stdio.py's
        `MEMORY_READY_TOOLS` gate), so `self._indexer` is never None on a
        supported transport — this early return is a defensive fallback,
        not a reachable path.
        """
        if self._indexer is None:
            return _STILL_INITIALIZING_MSG
        return self._passport_store.format_for_llm(
            self._workspace_root,
            language_stats=self._indexer.indexed_language_stats(),
            indexed_files=self._indexer.indexed_file_paths,
        )

    # ------------------------------------------------------------------
    # L2 — symbol graph
    # ------------------------------------------------------------------

    def locate(self, name: str, limit: int = 10) -> list:
        return self._symbol_graph.locate(self._workspace_root, name, limit)

    def locate_with_snippets(self, name: str, limit: int = 10, caller_file: str | None = None):
        """Locate symbols via L2 multi-strategy resolution. Returns LocateResult. No LLM call."""
        return self._symbol_graph.locate_l2(self._workspace_root, name, limit=limit, caller_file=caller_file)

    def trace(self, name: str, direction: str = "both", limit: int = 20,
              include_builtins: bool = False) -> dict:
        return self._symbol_graph.trace(self._workspace_root, name, direction, limit, include_builtins)  # type: ignore[arg-type]

    def trace_with_snippets(self, name: str, direction: str = "both", limit: int = 20,
                            include_builtins: bool = False) -> dict:
        """Trace call graph. Caller/callee names are returned as-is; AI can locate() them for snippets.
        Builtin/stdlib callees hidden unless include_builtins (UPG-4.3)."""
        return self._symbol_graph.trace(self._workspace_root, name, direction, limit, include_builtins)  # type: ignore[arg-type]

    def ingest_traces(self, trace_events: list[dict]) -> dict:
        """Ingest runtime trace events into the symbol graph."""
        return self._symbol_graph.ingest_trace_data(self._workspace_root, trace_events)

    def format_locate(self, result, name: str = "") -> str:
        from agent.symbol_graph import LocateResult
        if isinstance(result, LocateResult):
            return self._symbol_graph.format_locate_l2_for_llm(result)
        return self._symbol_graph.format_locate_for_llm(result, name)

    def format_trace(self, trace_result: dict, name: str) -> str:
        # UPG-TRACE-ABS-PATH / UPG-TRACE-FETCH-IDS: workspace root doubles as
        # both the display-relpath root and the chunk-span lookup's own
        # symbol-graph query key (same value `trace()` already uses for
        # DB scoping).
        chunk_span_lookup = getattr(self._indexer, "chunk_span_for_lines", None)
        return self._symbol_graph.format_trace_for_llm(
            trace_result, name,
            workspace=self._workspace_root,
            chunk_span_lookup=chunk_span_lookup,
        )

    # ------------------------------------------------------------------
    # Memory — working context store
    # ------------------------------------------------------------------

    def _require_memory_layer(self) -> None:
        """Guard for every memory-facing method (UPG-SEARCH-ONLY-MODE): in
        search-only mode there is no context store — raise rather than hit an
        AttributeError on `self._context_store is None`. The MCP dispatch and
        REST layers intercept these calls earlier with a friendlier message;
        this is the root-cause guard for any other caller."""
        if self._search_only:
            raise RuntimeError(_SEARCH_ONLY_MSG)

    def remember(
        self,
        content: str,
        tags: list[str] | None = None,
        priority: str = "medium",
        session_id: str | None = None,
        kind: str = "finding",
        title: str = "",
        agent: str = "",
        triggers: list[dict] | None = None,
        provenance: str = "agent",
        scope: str | None = None,
        anchors: list[str] | None = None,
        supersedes: int | None = None,
        contradicts: int | None = None,
        user_quote: str | None = None,
    ) -> int:
        """`agent` (UPG-SUBAGENT-MEMORY): optional caller-declared identifier
        for the agent/subagent authoring this note (e.g. "coder-2") — never
        inferred. Stored on the note's existing `author_id` column (already
        documented there as a "developer/agent identifier"; no schema change
        needed) and surfaced as an attribution tag in recall index lines when
        present. Absent (default "") renders exactly as before this feature.

        `triggers`/`provenance`/`scope`/`anchors`/`supersedes` (TRIGGER-ENGINE
        wave 1, bm2-design-skeleton.md §1/§2/§5): additive, all optional,
        passed straight through to `WorkingContextStore.remember()` — see its
        docstring for exact validation (raises ValueError on malformed
        triggers, an unrecognised provenance/scope, provenance="auto" on
        kind="directive", or a `supersedes` target that does not exist in
        this workspace). This method does no validation of its own so the
        store stays the single source of truth for these rules.

        `contradicts` (UPG-MEMORY-STATE-MACHINE §4.2): note_id this new note
        asserts was WRONG — passed straight through unchanged; see
        `WorkingContextStore.remember()`'s docstring for the exact
        `revoked` event this appends to the target.

        `user_quote` (UPG-MEM-PROVENANCE-USER-STATED): a verbatim excerpt of
        the user turn this note transcribes, passed straight through. The
        store binds it with a deterministic substring check and, only on a
        successful bind, stores the note at provenance="user-stated"; a
        failed bind never fails the write. No validation here — the store
        stays the single source of truth for that rule, like every parameter
        above.

        `scope`: None (the default) means OMITTED — the store resolves it to
        this note's kind's default scope at write time
        (UPG-TRIGGER-SCOPE-KIND-DEFAULTS). An explicitly passed scope,
        including the literal string "workspace", always wins verbatim."""
        self._require_memory_layer()
        note_id = self._context_store.remember(
            workspace=self._workspace_root,
            content=content,
            tags=tags,
            priority=priority,
            session_id=session_id,
            kind=kind,
            title=title,
            author_id=agent,
            triggers=triggers,
            provenance=provenance,
            scope=scope,
            anchors=anchors,
            supersedes=supersedes,
            contradicts=contradicts,
            user_quote=user_quote,
        )
        self._bump_notes_epoch()
        return note_id

    def remember_with_extras(
        self,
        content: str,
        tags: list[str] | None = None,
        priority: str = "medium",
        session_id: str | None = None,
        kind: str = "finding",
        title: str = "",
        agent: str = "",
        triggers: list[dict] | None = None,
        provenance: str = "agent",
        scope: str | None = None,
        anchors: list[str] | None = None,
        supersedes: int | None = None,
        contradicts: int | None = None,
        user_quote: str | None = None,
    ) -> RememberOutcome:
        """Same parameters and same write as `remember()`, plus two
        strictly-additive write-time offers for the caller LLM — never a
        second write, never a gate on the first one.

        The write always succeeds or raises exactly as `remember()` does
        (this method calls it directly, so every existing validation error
        propagates unchanged). Once `note_id` exists, computing the two
        offers below is wrapped in one `try/except Exception`: any failure
        there — including one offer succeeding and the other raising — logs
        at debug and degrades the WHOLE extras computation to empty lists.
        A related-notes/proxy-anchor lookup must never turn a successful
        write into a failed response.

        `related`: the nearest existing ACTIVE notes to this one, computed
        only when `MEMORY_WRITE_RELATED_ENABLED` — see
        `agent/working_context_store/_related.py` for the exact contract
        (nearest-by-similarity only, never a contradiction judgment).

        `proxy_anchor_suggestions`: workspace-relative paths of proxy-anchor
        files present in this workspace (see `agent/proxy_anchors.py`),
        computed only when ALL of: `MEMORY_WRITE_PROXY_SUGGEST_ENABLED`,
        `kind == "operational"`, and the caller passed no `anchors` on this
        call. That triple gate is deliberate: offer an anchor exactly when
        an operational note arrives with none, never when the caller
        already made that choice.
        """
        note_id = self.remember(
            content=content,
            tags=tags,
            priority=priority,
            session_id=session_id,
            kind=kind,
            title=title,
            agent=agent,
            triggers=triggers,
            provenance=provenance,
            scope=scope,
            anchors=anchors,
            supersedes=supersedes,
            contradicts=contradicts,
            user_quote=user_quote,
        )

        related: list[RelatedNote] = []
        proxy_anchor_suggestions: list[str] = []
        try:
            if MEMORY_WRITE_RELATED_ENABLED and self._context_store is not None:
                related = self._context_store.related_active_notes(
                    self._workspace_root,
                    note_id,
                    limit=MEMORY_WRITE_RELATED_LIMIT,
                    min_similarity=MEMORY_WRITE_RELATED_MIN_SIMILARITY,
                )
            if (
                MEMORY_WRITE_PROXY_SUGGEST_ENABLED
                and kind == "operational"
                and not anchors
            ):
                proxy_anchor_suggestions = suggest_proxy_anchors(
                    self._workspace_root, MEMORY_WRITE_PROXY_SUGGEST_LIMIT,
                )
        except Exception:
            logger.debug(
                "remember_with_extras: write-time offer computation failed for note #%s",
                note_id, exc_info=True,
            )
            related = []
            proxy_anchor_suggestions = []

        return RememberOutcome(
            note_id=note_id,
            related=related,
            proxy_anchor_suggestions=proxy_anchor_suggestions,
        )

    def _current_task_note(self):
        """The most recent kind='task' note, if any — SAME selection boot
        injection uses (`WorkingContextStore.boot_recall`'s task-row query:
        priority='high', newest-first), reused rather than duplicated. Uses
        `boot_recall` (not `fire()`/`fire_and_format()`) deliberately: it is
        legacy-but-correct and side-effect-free (does not bump
        `last_accessed`/`last_fired`), matching this passive "peek at
        current task" lookup rather than a genuine SessionStart firing
        event. Returns None in search-only mode or when there is no such
        note."""
        if self._context_store is None:
            return None
        for note in self._context_store.boot_recall(self._workspace_root):
            if note.kind == "task":
                return note
        return None

    def record_commit_note(
        self, sha: str, subject: str, branch: str, files: list[str],
    ) -> int:
        """Write ONE deterministic, zero-inference working-memory note per
        git commit (UPG-COMMIT-MEMORY-HOOK) — git already records WHAT
        changed; this records the commit's identity, its touched files, and
        the current task context, so a future session (or this one, past
        /compact) can recall WHY a change happened without re-deriving it
        from `git log`/`git show`.

        Called only by the git post-commit hook `vectr init --hooks`
        installs (see main.cmd_hook's "post-commit" branch /
        `_write_git_post_commit_hook`) — never by the editor's LLM.
        `sha`/`subject`/`branch`/`files` are raw git facts gathered
        client-side by that hook's own `git` subprocess calls; ALL
        interpretation (file-list capping, active-task lookup, content
        formatting) happens HERE, once, server-side — the single source of
        truth, same rationale as `recall()`'s own HOOKS_MIN_SIMILARITY
        default.

        kind="finding" (never "task" or "directive"): `default_bundle_for_
        kind` (agent/trigger_engine.py) gives finding/reference an EMPTY
        SessionStart trigger bundle at every priority, so this note can
        never boot-inject or crowd the directive/task notes a human or an
        agent wrote deliberately — structurally, not by an extra filter —
        it is recall-on-demand only (semantic recall can still find it).
        provenance="auto": captured by a mechanism with no reviewing
        judgment (a git hook, not an agent's own assessment) — the existing
        PROVENANCE_VALUES framing built for exactly this case, and rejected
        outright by the store for kind="directive" (moot here, since this
        method always writes kind="finding"). tags=[_COMMIT_NOTE_TAG] and
        agent=_COMMIT_NOTE_AGENT_ID are two complementary, pre-existing
        markers (no schema change) that make these notes filterable/
        attributable without affecting boot-injection eligibility, which
        kind alone already settles.
        """
        self._require_memory_layer()
        task_note = self._current_task_note()
        content = _format_commit_note_content(sha, subject, branch, files, task_note)
        return self.remember(
            content=content,
            tags=[_COMMIT_NOTE_TAG],
            priority="low",
            kind="finding",
            title=f"Commit {sha}: {subject[:HOOKS_COMMIT_NOTE_MAX_SUBJECT_CHARS]}",
            agent=_COMMIT_NOTE_AGENT_ID,
            provenance="auto",
        )

    # ------------------------------------------------------------------
    # Episode capture
    # ------------------------------------------------------------------

    def record_episode(
        self,
        *,
        session_id: str | None,
        ts: float | None,
        cwd: str,
        tool: str,
        command: str | None,
        description: str | None,
        file_path: str | None,
        rc: int | None,
        is_error: bool,
        interrupted: bool,
        stdout_tail: str,
        stderr_tail: str,
        harness_success: bool | None = None,
    ) -> int:
        """Write ONE deterministic, zero-inference episode row per
        Bash/Edit/Write/MultiEdit tool call (called only by `POST /v1/episode`,
        which is called only by `vectr hook post-tool-use`'s detached
        worker). ALL interpretation — argv normalization, digest
        canonicalization, outcome derivation — happens HERE, once,
        server-side, mirroring `record_commit_note`'s own "raw facts
        client-side, interpretation server-side" split; no embedding ever
        happens on this path.

        `harness_success` (tri-state, default None) is forwarded straight
        into `agent.outcome.derive_outcome` — see that module's docstring
        for what it is and why it exists; it is never persisted as its own
        column, only consumed to help derive the `outcome` column, exactly
        like `is_error`/`interrupted`.

        `description` (Bash tool_input.description, an LLM-authored one-line
        summary) is accepted for forward compatibility but not persisted —
        the episode row schema has no column for it, and no logic here
        ever branches on its content (R5: no query/prompt-content
        classification)."""
        self._require_memory_layer()
        from agent.episode_canon import canonicalize_digest
        from agent.outcome import derive_outcome
        from app.cmdnorm import normalize_command

        cmd_raw = command or ""
        if tool == "bash" and cmd_raw:
            # Single canonical normalizer: the duplicate
            # `agent/episode_normalize.py` diverged from this one and is
            # now deleted. `app/arcs.py`'s ArcDetector consumes the
            # exact same `NormalizedCommand` shape, so a persisted episode's
            # (verb, flags, args) always matches what the detector sees.
            parsed = normalize_command(cmd_raw)
            verb = parsed.verb
            flags = list(parsed.flags)
            args = list(parsed.args)
            env_delta_names = list(parsed.env_prefix_names)
        else:
            verb, flags, args, env_delta_names = "", [], [], []

        stdout_digest = canonicalize_digest(
            stdout_tail or "",
            EPISODES_DIGEST_MAX_CHARS,
            EPISODES_DIGEST_HEAD_LINES,
            EPISODES_DIGEST_TAIL_LINES,
        )
        stderr_digest = canonicalize_digest(
            stderr_tail or "",
            EPISODES_DIGEST_MAX_CHARS,
            EPISODES_DIGEST_HEAD_LINES,
            EPISODES_DIGEST_TAIL_LINES,
        )
        derived = derive_outcome(
            rc=rc,
            is_error=is_error,
            interrupted=interrupted,
            stdout_digest=stdout_digest,
            stderr_digest=stderr_digest,
            harness_success=harness_success,
        )
        episode_ts = ts if ts is not None else time.time()
        episode_id = self._episode_store.insert(
            self._workspace_root,
            session_id=session_id,
            ts=episode_ts,
            cwd=cwd or "",
            tool=tool,
            cmd_raw=cmd_raw,
            verb=verb,
            flags=flags,
            args=args,
            rc=rc,
            termination=derived["termination"],
            outcome=derived["outcome"],
            stdout_digest=stdout_digest,
            stderr_digest=stderr_digest,
            markers_matched=derived["markers_matched"],
            env_delta_names=env_delta_names,
            file_path=file_path,
            max_rows=EPISODES_MAX_ROWS,
            ttl_days=EPISODES_TTL_DAYS,
        )

        # Feed the arc detector: config-gated so the write path can be
        # disabled without touching the detector's own always-loaded
        # config. The dict shape
        # matches app/arcs.py's documented `episode` contract exactly
        # (`markers`, not `markers_matched` — a deliberate naming
        # difference between the DB column and the detector's in-memory
        # field); `episode_id` is an extra key the detector never reads but
        # carries through by reference into any `Arc` it emits, letting
        # `_persist_arc` correlate an arc back to the episode rows it
        # resolved.
        if ARC_DETECTION_ENABLED and self._arc_detector is not None:
            detector_episode = {
                "episode_id": episode_id,
                "session_id": session_id,
                "ts": episode_ts,
                "cwd": cwd or "",
                "tool": tool,
                "cmd_raw": cmd_raw,
                "outcome": derived["outcome"],
                "termination": derived["termination"],
                "markers": derived["markers_matched"],
                "env_delta_names": env_delta_names,
                "file_path": file_path,
            }
            for arc in self._arc_detector.observe(detector_episode):
                self._persist_arc(arc)

        return episode_id

    def _persist_arc(self, arc) -> None:
        """Persist one `ArcDetector`-emitted arc into the quarantined
        `arcs` table — episode capture itself never writes notes, and an
        arc row is no exception — and stamp `arc_id` back onto every
        episode row the arc resolved. The
        `episode_id` key read here was injected into each detector-episode
        dict in `record_episode`, above — `arc.failures_chain`/`arc.success`
        are those SAME dicts by reference (app/arcs.py stores episodes by
        reference, never copies), so it survives unchanged."""
        failure_ids = [
            e["episode_id"] for e in arc.failures_chain if e.get("episode_id") is not None
        ]
        success_id = arc.success.get("episode_id")
        arc_row_id = self._episode_store.insert_arc(
            self._workspace_root,
            session_id=arc.session_id,
            cwd=arc.success.get("cwd") or "",
            ts=arc.success.get("ts") or time.time(),
            confidence=arc.confidence,
            mutation_diff=arc.mutation_diff,
            failure_episode_ids=failure_ids,
            success_episode_id=success_id,
        )
        resolved_ids = list(failure_ids)
        if success_id is not None:
            resolved_ids.append(success_id)
        for episode_id in resolved_ids:
            self._episode_store.mark_episode_arc(episode_id, arc_row_id)

    def list_episodes(
        self,
        *,
        session_id: str | None = None,
        arc_id: int | None = None,
        since_ts: float | None = None,
        until_ts: float | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """The only bulk reader of the `episodes` table besides the
        aggregate counts in `status()` — deliberately absent from
        `agent/searcher.py` and `agent/working_context_store`, so nothing on
        the search/recall path can ever import it."""
        self._require_memory_layer()
        return self._episode_store.list_episodes(
            self._workspace_root,
            session_id=session_id,
            arc_id=arc_id,
            since_ts=since_ts,
            until_ts=until_ts,
            limit=limit,
        )

    def count_episodes(self) -> int:
        """Search-only mode (or no episode store yet): always 0, never an
        error — mirrors `count_notes()`'s own contract."""
        if self._episode_store is None:
            return 0
        return self._episode_store.count_episodes(self._workspace_root)

    def count_arcs_pending_distill(self) -> int:
        """Best-effort — see `EpisodeStore.count_arcs_pending_distill`."""
        if self._episode_store is None:
            return 0
        return self._episode_store.count_arcs_pending_distill(self._workspace_root)

    def list_arcs(self, *, status: str = "pending", limit: int = 100) -> list[dict]:
        """Read surface for `GET /v1/arcs` and `vectr_distill()` — see
        `EpisodeStore.list_arcs` for the returned shape."""
        self._require_memory_layer()
        return self._episode_store.list_arcs(
            self._workspace_root, status=status, limit=limit,
        )

    def resolve_arcs_distilled(self, arc_ids: list[int], note_id: int) -> dict:
        """Write-back for `vectr_remember(..., distilled_from=[...])`:
        marks `arc_ids` distilled
        into `note_id`. Returns `{"resolved": [...], "unresolved": [...]}`
        — idempotent, never raises on unknown/already-resolved ids."""
        self._require_memory_layer()
        return self._episode_store.resolve_arcs_distilled(
            self._workspace_root, arc_ids, note_id,
        )

    def resolve_arcs_dismissed(self, arc_ids: list[int], reason: str) -> dict:
        """Write-back for `vectr_distill(dismiss=[...], reason=...)` /
        `POST /v1/arcs/dismiss`.
        Returns `{"resolved": [...], "unresolved": [...]}` — idempotent,
        never raises on unknown/already-resolved ids."""
        self._require_memory_layer()
        return self._episode_store.resolve_arcs_dismissed(
            self._workspace_root, arc_ids, reason,
        )

    def promote_note(self, note_id: int, to: str) -> bool:
        """Explicit provenance promotion (TRIGGER-ENGINE wave 1,
        bm2-design-skeleton.md §5): auto -> agent -> human, one step at a
        time. See `WorkingContextStore.promote()` for the exact contract
        (raises ValueError on an invalid/out-of-order transition; returns
        False if the note does not exist)."""
        self._require_memory_layer()
        promoted = self._context_store.promote(self._workspace_root, note_id, to)
        if promoted:
            self._bump_notes_epoch()
        return promoted

    def revoke_note(self, note_id: int, reason: str, actor: str = "agent") -> bool:
        """Explicit revocation (UPG-MEMORY-STATE-MACHINE §4.2): caller
        (agent or human, never "system" — that actor is reserved for the
        deterministic anchor-drift transition) asserts this note is wrong.
        See `WorkingContextStore.revoke_note()` for the exact contract
        (raises ValueError on an invalid actor; returns False if the note
        does not exist). The note stays fire/recall-eligible afterward in
        anti-memory (deterrent) form — this is not a delete."""
        self._require_memory_layer()
        revoked = self._context_store.revoke_note(self._workspace_root, note_id, reason, actor=actor)
        if revoked:
            self._bump_notes_epoch()
        return revoked

    def reinstate_note(self, note_id: int, actor: str = "agent", reason: str | None = None) -> bool:
        """Revert-of-revert (UPG-MEMORY-STATE-MACHINE §4.2): always legal.
        See `WorkingContextStore.reinstate_note()` for the exact contract
        (raises ValueError on an invalid actor; returns False if the note
        does not exist)."""
        self._require_memory_layer()
        reinstated = self._context_store.reinstate_note(self._workspace_root, note_id, actor=actor, reason=reason)
        if reinstated:
            self._bump_notes_epoch()
        return reinstated

    def get_note(self, note_id: int):
        """Fetch a single note by ID (UPG-RECALL-HIERARCHY expand path)."""
        self._require_memory_layer()
        return self._context_store.get_note(self._workspace_root, note_id)

    def recall(
        self,
        query: str | None = None,
        tags: list[str] | None = None,
        priority: str | None = None,
        limit: int = 10,
        kind: str | None = None,
        boot: bool = False,
        min_similarity: float | None = None,
        file_path: str | None = None,
        command: str | None = None,
        max_age_days: float | None = None,
        sort_by: str = "relevance",
        detail: str = "index",
        note_id: int | None = None,
        surface: str = "mcp",
        hook_event: str | None = None,
        session_id: str | None = None,
        events: list[str] | None = None,
    ) -> str:
        """`surface` selects the expand-hint phrasing rendered by
        `format_notes_for_llm` — 'mcp' (default: the MCP dispatch path, and
        hook-injected recall, both leave this unset since their reader is the
        editor's LLM) or 'cli' (only `cmd_recall`'s own REST request sets this
        explicitly, since its reader is a human terminal). See its docstring
        (UPG-CLI-RECALL-HINT).

        `hook_event` (UPG-HOOK-INJECT-OBSERVABILITY): set only by `vectr
        hook`'s own request — 'SessionStart' | 'UserPromptSubmit' |
        'PreToolUse'. When set and this call actually returns notes, counts
        one injection under that hook kind (see `status()`). None (the
        default — direct vectr_recall/`vectr recall` calls) records nothing.

        `min_similarity` defaults to `HOOKS_MIN_SIMILARITY`
        (agent/config.yaml `hooks.min_similarity`) whenever `hook_event` is
        set and the caller omitted an explicit value (UPG-HOOK-SUBPROCESS-
        IMPORT-TAX). The per-turn relevance floor lives in exactly one place
        — here, where `agent.config` is already loaded once at daemon
        startup — rather than being re-derived by the short-lived `vectr
        hook` subprocess on every single turn, which would force it to pay
        `agent.config`'s full import cost on its own hot path for a value it
        can otherwise ask the daemon to apply by default. A direct
        vectr_recall/`vectr recall` call (hook_event=None) is unaffected:
        `min_similarity=None` there still means "no floor," as before.

        `session_id`/`events` (TRIGGER-ENGINE wave 2a, bm2-design-skeleton.md
        §3): the calling session's identity and the lifecycle event(s) this
        recall is standing in for. Threaded into the `boot`/`file_path`
        branches below so per-memory triggers (declared via
        `vectr_remember(triggers=..., scope=...)`) fire through the live
        engine — ledgered per session, budget-packed, scope-enforced —
        alongside (boot) or merged with (file_path) the legacy delivery
        paths those branches already provide. Both default to None, which
        reproduces today's ledger-less, budget-less, scope-unenforced
        behaviour exactly for any caller that predates this wave.

        `command` (wave 3, UPG-MEMORY-STATE-MACHINE §5.2): the raw Bash
        command about to run — `PreToolUse`'s new `Bash` matcher lane
        (main.py's `cmd_hook`). Mutually exclusive with `file_path` at the
        hook-call level (one PreToolUse event is either a file tool or a
        Bash tool); threaded into `_recall_impl()`'s own `command` branch.

        Serving-policy hardening (wave 3, §5.3/§5.5): a `hook_event`==
        "UserPromptSubmit" call resets this session's per-turn injection
        ledger (`reset_turn_ledger`) and advances its semantic-cooldown
        turn counter BEFORE evaluating anything this turn — the one place
        "reset at each UserPromptSubmit" (§5.3) is implemented, reusing the
        hook signal `main.py`'s `user-prompt-submit` branch already sends
        rather than adding a new endpoint."""
        if hook_event == "UserPromptSubmit" and session_id:
            self.reset_turn_ledger(session_id)
            turn_counter_ledger = self._ledger_for(session_id)
            if turn_counter_ledger is not None:
                turn_counter_ledger.advance_turn()
        if hook_event is not None and min_similarity is None:
            min_similarity = HOOKS_MIN_SIMILARITY
        notes = self._recall_impl(
            query=query, tags=tags, priority=priority, limit=limit, kind=kind,
            boot=boot, min_similarity=min_similarity, file_path=file_path, command=command,
            max_age_days=max_age_days, sort_by=sort_by, detail=detail,
            note_id=note_id, surface=surface, session_id=session_id, events=events,
        )
        if hook_event is not None:
            self._record_hook_injection(hook_event, notes)
        return notes

    def _recall_impl(
        self,
        query: str | None = None,
        tags: list[str] | None = None,
        priority: str | None = None,
        limit: int = 10,
        kind: str | None = None,
        boot: bool = False,
        min_similarity: float | None = None,
        file_path: str | None = None,
        command: str | None = None,
        max_age_days: float | None = None,
        sort_by: str = "relevance",
        detail: str = "index",
        note_id: int | None = None,
        surface: str = "mcp",
        session_id: str | None = None,
        events: list[str] | None = None,
    ) -> str:
        self._require_memory_layer()
        # Single-note expand: note_id overrides everything else (UPG-RECALL-HIERARCHY).
        if note_id is not None:
            note = self._context_store.get_note(self._workspace_root, note_id)
            if note is None:
                return f"Note #{note_id} not found."
            stale = self._context_store.check_staleness([note], self._workspace_root)
            return self._context_store.format_notes_for_llm([note], stale_warnings=stale, detail="full", surface=surface)

        # Boot mode (UPG-9.2 contract; TRIGGER-ENGINE wave 2a rewires the
        # mechanism onto the live engine). Ignores query/tags/priority/kind/
        # limit and returns "" (never the "no notes" placeholder) so a
        # SessionStart hook injects nothing on a fresh workspace rather than
        # noise. The directive/task kind-default trigger bundles (see
        # agent/trigger_engine.py's kind-default table) reproduce the old
        # `boot_recall()`'s unconditional-directive + high-task coverage
        # exactly, so `fire_and_format()` fully replaces that dump — with a
        # genuine improvement: an explicit session-start trigger declared on
        # any OTHER kind (finding/reference/gotcha) now also fires here,
        # where `boot_recall()` silently ignored it (kind-gated, not
        # trigger-gated). `events` lets a caller merge post-compaction
        # eligibility into the same boot fire (see main.py's `cmd_hook`);
        # absent that, the implicit event is `["session-start"]`.
        if boot:
            events_to_fire = events if events else ["session-start"]
            ledger = self._ledger_for(session_id)
            # `spend_turn_budget` deliberately omitted (defaults False):
            # session-start bulk keeps its existing separate per-SESSION
            # cap (`ledger`) rather than the smaller ordinary-turn
            # allowance (§5.4) — the turn ledger's dedup CLAIM still runs
            # via `turn_ledger` below, so a note delivered at boot is still
            # excluded from a same-turn PreToolUse/prompt-submit re-delivery.
            fire_text, _ = self._context_store.fire_and_format(
                self._workspace_root,
                events=events_to_fire,
                session_id=session_id,
                ledger=ledger,
                turn_ledger=self._turn_ledger_for(session_id),
                surface=surface,
            )
            # UPG-NUDGE-HOOK-PATH-UNREACHABLE (B9): the stale-task hygiene nudge
            # renders in vectr_status, but the shipped guidance steers agents
            # away from calling status at session start (notes auto-inject
            # instead) — so the state-based backstop was unreachable exactly
            # where the task-note pile-up lives. Surface the same one-line nudge
            # on the SessionStart injection path agents actually hit, budget-aware
            # (skipped when the note pack already spent the session's token cap).
            nudge = self._stale_task_nudge_line()
            if nudge and (ledger is None or ledger.remaining_budget() > 0):
                fire_text = f"{fire_text}\n\n{nudge}" if fire_text else nudge
            # Pending-arc nudge: same session-start/
            # post-compaction payload, one more additive line, budget-aware
            # the same way the stale-task nudge above is.
            arc_nudge = self._arc_distill_nudge_line()
            if arc_nudge and (ledger is None or ledger.remaining_budget() > 0):
                fire_text = f"{fire_text}\n\n{arc_nudge}" if fire_text else arc_nudge
            return fire_text

        # Path-anchored mode (UPG-9.6 contract; TRIGGER-ENGINE wave 2a adds
        # the live engine alongside it). A gotcha with a structured trigger
        # (anchors[]/triggers[]) now fires through the engine — §3 block
        # format, provenance frame, staleness caveat, budget pack — and is
        # excluded from the legacy `recall_for_path()` pass below by
        # note_id, so the same note is never rendered twice. A gotcha with
        # no anchors (matched only by `recall_for_path()`'s path/content
        # heuristics) keeps the legacy path unchanged — the engine does not
        # cover it this wave.
        if file_path:
            turn_ledger = self._turn_ledger_for(session_id)
            fire_text, fired_ids = self._context_store.fire_and_format(
                self._workspace_root,
                event="pre-edit",
                file_path=file_path,
                session_id=session_id,
                ledger=self._ledger_for(session_id),
                turn_ledger=turn_ledger,
                spend_turn_budget=True,
                surface=surface,
            )
            legacy_notes = self._context_store.recall_for_path(
                self._workspace_root, file_path, kind=kind, limit=limit, session_id=session_id)
            # `fired_ids` alone only excludes what THIS call's own engine
            # fire delivered; a note claimed by an EARLIER surface this
            # same turn (turn-deduped out of `fired_ids` here on purpose)
            # must also be excluded from the legacy content-match fallback
            # below, or the cross-surface dedup the turn ledger exists for
            # (§5.3/§5.4) leaks back in through this pre-engine path — the
            # same double-dip class `fire_and_format()`'s own turn-dedup
            # prevents, just reached via a different code path.
            legacy_notes = [
                n for n in legacy_notes
                if n.note_id not in fired_ids
                and (turn_ledger is None or turn_ledger.eligible(n.note_id))
            ]
            legacy_text = ""
            if legacy_notes:
                stale = self._context_store.check_staleness(legacy_notes, self._workspace_root)
                legacy_text = self._context_store.format_notes_for_llm(
                    legacy_notes, stale_warnings=stale, detail=detail, surface=surface)
            if fire_text and legacy_text:
                return fire_text + "\n\n" + legacy_text
            return fire_text or legacy_text

        # Command-anchored mode (wave 3, UPG-MEMORY-STATE-MACHINE §5.2):
        # PreToolUse's new `Bash` matcher lane. No legacy-recall merge
        # here — unlike the file_path branch above, there is no pre-
        # existing content-substring path for command-anchored notes to
        # fall back to; the 'command' trigger primitive is the entire
        # surface. `spend_turn_budget=True`: this is an ordinary per-turn
        # surface, metered against the shared ≤500-token turn allowance
        # (§5.4), unlike session-start's boot delivery above.
        if command:
            fire_text, _ = self._context_store.fire_and_format(
                self._workspace_root,
                event="pre-run",
                command=command,
                session_id=session_id,
                ledger=self._ledger_for(session_id),
                turn_ledger=self._turn_ledger_for(session_id),
                spend_turn_budget=True,
                surface=surface,
            )
            return fire_text

        # Generic query mode (TRIGGER-ENGINE wave 2a): `events`, when given,
        # is the same opt-in merge the file_path branch above always applies
        # — e.g. main.py's UserPromptSubmit hook passes events=["prompt-
        # submit"] alongside its query. Only an EXPLICIT `triggers[]`
        # override ever matches "prompt-submit" (no kind's default bundle
        # does), so this is a no-op for every note that predates an
        # explicit override; absent `events` entirely (the default — every
        # other caller: direct vectr_recall, `vectr recall`, etc.) this
        # branch behaves exactly as before the engine existed.
        #
        # `query` is forwarded here too (wave 2b, §8): a prompt-submit event
        # is the M (semantic) primitive's one fire point — this is the only
        # `fire_and_format()` call site in this file that ever has both an
        # `events` list AND the actual prompt text available together.
        fire_text, fired_ids = "", set()
        turn_ledger = self._turn_ledger_for(session_id)
        if events:
            fire_text, fired_ids = self._context_store.fire_and_format(
                self._workspace_root,
                events=events,
                query=query,
                session_id=session_id,
                ledger=self._ledger_for(session_id),
                turn_ledger=turn_ledger,
                spend_turn_budget=True,
                surface=surface,
            )

        notes = self._context_store.recall(
            workspace=self._workspace_root,
            query=query,
            tags=tags,
            priority=priority,
            limit=limit,
            kind=kind,
            min_similarity=min_similarity,
            max_age_days=max_age_days,
            sort_by=sort_by,
            session_id=session_id,
        )
        # Same cross-surface leak the file_path branch above guards
        # against, but ONLY when `events` is given — i.e. this call is
        # itself standing in for an injection surface (main.py's
        # UserPromptSubmit hook passing events=["prompt-submit"]
        # alongside `query`), not a caller's own deliberate
        # `vectr_recall(query=...)` lookup. A plain direct recall has no
        # `events` and must never be turn-deduped: it is an explicit
        # request for a note's content, not a passive injection surface —
        # suppressing it because some OTHER surface already injected the
        # note earlier this turn would silently hide the note (worst case,
        # only its budget-capped index line was ever injected, and the
        # caller can no longer expand the full body this turn). `fired_ids`
        # is always empty here when `events` is falsy (the `if events:`
        # block above never ran), so this condition is equivalent to
        # "only when the injection-surface fire_and_format call above
        # actually ran."
        if events:
            notes = [
                n for n in notes
                if n.note_id not in fired_ids
                and (turn_ledger is None or turn_ledger.eligible(n.note_id))
            ]
        stale = self._context_store.check_staleness(notes, self._workspace_root)
        formatted = self._context_store.format_notes_for_llm(
            notes, stale_warnings=stale, detail=detail, surface=surface, sort_by=sort_by
        )
        # UPG-STDIO-MEMORY-READY: a query-bearing recall that lands before the
        # embedder has attached used the lexical SQL LIKE fallback — tell the
        # caller LLM plainly, so it knows to retry later for semantic ranking
        # rather than assuming these are the best-ranked matches available.
        # Gated purely on `embedder_ready` state (never on `query`'s content)
        # and only when a query was given at all — index/list-mode recall
        # (query=None) has no ranking to degrade, so it is never appended
        # there, and it never appears once the embedder has attached.
        if query and not self.embedder_ready:
            formatted += _EMBEDDER_LOADING_RECALL_NOTICE
        if fire_text and formatted:
            return fire_text + "\n\n" + formatted
        return fire_text or formatted

    def forget_note(self, note_id: int) -> bool:
        self._require_memory_layer()
        ok = self._context_store.forget(self._workspace_root, note_id)
        if ok:
            self._bump_notes_epoch()
        return ok

    def forget_all(self) -> int:
        self._require_memory_layer()
        n = self._context_store.forget_all(self._workspace_root)
        self._bump_notes_epoch()
        return n

    # ------------------------------------------------------------------
    # Proactive context (UPG-PRO) + org-wide artifact cache
    # ------------------------------------------------------------------

    def _bump_notes_epoch(self) -> None:
        """Advance the notes-mutation sequence so any recall artifact cached
        under the previous notes epoch can never be served after a change."""
        with self._notes_mutation_lock:
            self._notes_mutation_seq += 1

    def _index_epoch(self, scope: str) -> str:
        """Identity of the current index state for a cache scope. A change here
        invalidates every artifact keyed under it. `scope='code'` ties to the
        code index (chunks + last-indexed + embed model + version); `scope='notes'`
        ties to the notes-mutation sequence."""
        if scope == "notes":
            with self._notes_mutation_lock:
                return f"notes:{self._notes_mutation_seq}"
        return (
            f"code:{self._indexer.total_chunks}:{self.last_indexed}:"
            f"{self._embed_model}:{self._version_stamp}"
        )

    def recall_scored(
        self,
        query: str | None = None,
        tags: list[str] | None = None,
        priority: str | None = None,
        limit: int = 10,
        kind: str | None = None,
        min_similarity: float | None = None,
        max_age_days: float | None = None,
        sort_by: str = "relevance",
    ) -> list:
        """Structured scored recall (UPG-PRO-1): list[(WorkingNote, score|None)].

        Consults the org-wide artifact cache when enabled (keyed by args + the
        current notes epoch, so a note change invalidates it). The SQL fallback
        yields None scores — never a fabricated number.
        """
        self._require_memory_layer()

        def _compute():
            return self._context_store.recall_scored(
                workspace=self._workspace_root, query=query, tags=tags,
                priority=priority, limit=limit, kind=kind,
                min_similarity=min_similarity, max_age_days=max_age_days, sort_by=sort_by,
            )

        cache = self._artifact_cache
        if cache is None:
            return _compute()
        from agent.proactive.cache import canonical_key
        args = {
            "query": query, "tags": tags, "priority": priority, "limit": limit,
            "kind": kind, "min_similarity": min_similarity,
            "max_age_days": max_age_days, "sort_by": sort_by,
        }
        key = canonical_key("recall_scored", args, self._index_epoch("notes"))
        found, cached = cache.get(key)
        if found:
            return cached
        result = _compute()
        cache.put(key, result)
        return result

    def _proactive_gate(self, settings):
        from agent.proactive.gate import LedgerStore, ProactiveGate
        if self._proactive_ledger is None:
            self._proactive_ledger = LedgerStore(settings.cooldown_items)
        return ProactiveGate(
            min_similarity=settings.min_similarity,
            max_items_per_event=settings.max_items_per_event,
            max_chars_per_event=settings.max_chars_per_event,
            cooldown_items=settings.cooldown_items,
            max_weak_structural_items=settings.max_weak_structural_items,
            ledger_store=self._proactive_ledger,
        )

    def proactive_context(
        self,
        *,
        text: str = "",
        file_paths: list[str] | None = None,
        symbols: list[str] | None = None,
        edited_file_paths: list[str] | None = None,
        session_id: str = "",
        channel: str = "proxy",
        structural_only: bool = False,
        defer_charge: bool = False,
        confirm_token: str = "",
    ) -> dict:
        """Run the matcher + gate over an already-assembled window and return
        packed proactive context (UPG-PRO-7 subset serving the proxy).

        Honors the master opt-in and the memory layer being present. Returns an
        empty result (never an error) when disabled or when nothing clears the
        floor + budget, so a caller can always forward unmodified.

        CHARGE TIMING (UPG-PROXY-APPEND-BURNS-COOLDOWN). By default this method
        charges at RETRIEVAL: the gate records the selected anchors in the
        cooldown ledger and a PROACTIVE_INJECT audit event is written here and
        now. That is correct for any caller where returning the block IS
        delivering it, and it is this method's unchanged historical behaviour.

        A caller that cannot know at this point whether the block will reach
        the model — the proxy, which only afterwards discovers whether the
        request body can carry it — passes `defer_charge=True`. Then nothing is
        charged and no PROACTIVE_INJECT is written; a `delivery_token` is
        returned instead, and the caller hands it back as `confirm_token` on a
        later call once the block is confirmed appended. Only then is the
        cooldown charged and PROACTIVE_INJECT emitted.

        `confirm_token` is consumed FIRST, before this request's own matching
        and selection. That ordering is the invariant that keeps deferred
        charging safe: by the time the gate consults the cooldown ledger, every
        previously-delivered anchor is already recorded in it, so a note cannot
        be selected twice in a sequence of requests.
        """
        # Deliberately the historical FOUR keys. The deferred-charge fields are
        # added only on the path that actually defers, so the shape of "injected
        # nothing" is unchanged for every existing caller and assertion; there
        # is by definition no delivery to confirm on this path. `routes.py`
        # reads both fields with `.get`, so their absence is not a KeyError.
        empty = {"context": "", "item_count": 0, "anchor_ids": [], "scores": []}
        # Confirmation is charged even when the checks below decide to inject
        # nothing this time round — it settles a PREVIOUS delivery and must not
        # be dropped because, say, this window happens to be empty.
        if confirm_token:
            self._confirm_proactive_delivery(confirm_token)
        if self._search_only:
            # No working-memory layer in search-only mode; nothing to inject.
            return empty
        # UPG-PROXY-LOOPBACK-BYPASS: unconditional, channel-independent bind
        # refusal — sits BEFORE and INDEPENDENT of the master-switch/channel
        # policy below. A client-supplied `channel` label (its default value
        # is "proxy") must never be able to route around this: a non-loopback
        # bind fails closed to the empty result for every channel, including
        # the proxy channel's own launch-is-consent exemption from the
        # master switch just below.
        from agent.proactive.settings import proactive_bind_is_loopback
        if not proactive_bind_is_loopback():
            return empty
        from agent.proactive.settings import ProactiveSettings
        settings = ProactiveSettings.from_env()
        # The master opt-in gates AMBIENT surfaces (hooks read the transcript
        # without any per-session user action). The proxy channel is different:
        # the user explicitly launched `vectr proxy` with injection enabled and
        # pointed their client at it — that launch IS the consent for this
        # channel. The daemon is localhost-only and already serves notes to
        # local callers ungated (recall/search), so honoring the proxy channel
        # here adds no exposure beyond existing endpoints. (The bind check
        # above already ensures "localhost-only" is actually true at runtime.)
        if not settings.enabled and channel != "proxy":
            return empty
        from agent.proactive.matcher import ProactiveMatcher
        from agent.proactive.types import ProactiveWindow

        window = ProactiveWindow(
            text=text or "",
            file_paths=list(file_paths or []),
            symbols=list(symbols or []),
            edited_file_paths=list(edited_file_paths or []),
        )
        if window.is_empty():
            return empty

        service = self

        class _ServiceMatchSource:
            def structural_notes(self, paths):
                # UPG-PROXY-INJECT-PRECISION lever 1 + 1b: the structural
                # channel only ever admits notes whose `kind` is on the
                # declared eligibility allowlist (a static NOTE PROPERTY,
                # never a read of query/window content) — `task`/`directive`
                # notes are excluded by default since they describe
                # in-progress work or standing rules, not durable facts about
                # this file. `recall_for_path` itself is untouched; this is a
                # filter-after-return. Because the filter can only shrink the
                # pool, the requested pool is over-fetched by a configured
                # multiplier (capped by an absolute ceiling) so the filter
                # cannot starve the channel down to fewer than
                # `max_items_per_event` eligible notes when the path has a
                # healthy mix of kinds.
                pool_size = min(
                    settings.max_items_per_event * settings.structural_overfetch_multiplier,
                    settings.structural_overfetch_ceiling,
                )
                seen: dict[int, object] = {}
                for p in paths:
                    try:
                        for note in service._context_store.recall_for_path(
                            service._workspace_root, p, limit=pool_size
                        ):
                            if note.kind in settings.structural_kinds:
                                seen.setdefault(note.note_id, note)
                    except Exception:
                        continue
                return list(seen.values())

            def semantic_notes(self, wtext, min_similarity, limit):
                scored = service.recall_scored(
                    query=wtext, limit=limit, min_similarity=min_similarity,
                )
                return [(n, s) for (n, s) in scored if s is not None]

            def code_search(self, wtext, n_results):
                if service._memory_only:
                    return []
                try:
                    results, _ms = service.search(wtext, n_results=n_results)
                    return list(results)
                except Exception:
                    return []

            def note_states(self, notes):
                # UPG-PROXY-REVOKED-LEAK: batched fold, called once by
                # ProactiveMatcher.match() for the union of matched notes —
                # never per-note.
                #
                # Deliberately NOT wrapped in a try/except, unlike the
                # sibling match methods above. Those degrade to an empty
                # RESULT (fewer candidates — safe). This one degrades to
                # empty STATE, which the renderer would read as "every note
                # is active" and would put a revoked note's raw content back
                # into injected context as apparent fact. A failure here must
                # propagate so ProactiveMatcher.match() can fail closed and
                # drop the note candidates entirely.
                return service._context_store.note_event_states(
                    service._workspace_root, notes
                )

        matcher = ProactiveMatcher(
            _ServiceMatchSource(),
            min_similarity=settings.min_similarity,
            max_chars_per_event=settings.max_chars_per_event,
            structural_note=settings.matcher_structural_note,
            semantic_note=settings.matcher_semantic_note,
            code_search=settings.matcher_code_search,
            note_limit=max(settings.max_items_per_event * 2, settings.max_items_per_event),
            # UPG-PROXY-INJECT-ROLE-PROVENANCE: directive notes are excluded
            # CHANNEL-SCOPED, not globally — only the "proxy" channel appends
            # injected context into the newest USER-authored message on the
            # wire (agent/proactive/request_window.py's append_context_block,
            # the only production caller of which is agent/proactive/proxy.py,
            # which always passes channel="proxy"). A hypothetical future
            # channel that injects somewhere other than a user turn (e.g. a
            # session-start system message, matching how the hook/trigger-
            # engine surface already delivers directives correctly via a
            # wholly separate path — WorkingContextStore.fire_and_format(),
            # never this matcher/gate at all) should keep seeing directives.
            # Same precedent as `channel != "proxy"` at this method's master-
            # switch check above.
            exclude_directive_notes=(
                settings.proxy_exclude_directive_notes and channel == "proxy"
            ),
            structural_score_declared_anchor=settings.structural_score_declared_anchor,
            structural_score_declared_trigger=settings.structural_score_declared_trigger,
            structural_score_gotcha_mention=settings.structural_score_gotcha_mention,
            structural_score_mention=settings.structural_score_mention,
        )
        candidates = matcher.match(window)
        # UPG-PROXY-CROSS-CHANNEL-DEDUP: consult the SAME per-session
        # TurnInjectionLedger the hook/trigger-engine surfaces share (see
        # `_turn_ledger_for`) so a note already claimed this turn by that
        # surface is not re-injected here, and vice versa. Additive to the
        # gate's own anchor-id cooldown ledger, never a replacement.
        #
        # LOOKUP-ONLY, never allocate: use `_existing_turn_ledger`, not
        # `_turn_ledger_for`. KNOWN LIMITATION (out of scope here, filed as
        # UPG-PROXY-COOLDOWN-SESSION-IDENTITY): the proxy channel's
        # `session_id` is a per-proxy-PROCESS instance id
        # (agent/proactive/proxy.py's `VectrProxy._instance_id`, prefixed
        # `proxy-`), by construction disjoint from the editor-supplied
        # session id hook/trigger-engine surfaces use — so this namespace
        # mismatch means a proxy call would otherwise allocate a ledger no
        # hook surface will ever consult, for zero dedup benefit, while
        # consuming eviction-LRU slots and risking evicting a live editor
        # session's real ledger under heavy multi-subagent session churn.
        # Looking up only: zero cost when the namespaces don't collide, and
        # starts deduping for free the moment session identity is unified,
        # no further code change needed here.
        result = self._proactive_gate(settings).select(
            candidates, session_id=session_id, structural_only=structural_only,
            turn_ledger=self._existing_turn_ledger(session_id),
            # Deferred callers charge on confirmed delivery instead; the gate's
            # cooldown/turn-ledger READS are unaffected either way.
            record=not defer_charge,
            edited_file_paths=frozenset(window.edited_file_paths),
        )
        out = {
            "context": result.context,
            "item_count": result.item_count,
            "anchor_ids": list(result.anchor_ids),
            "scores": list(result.scores),
        }
        if result.is_empty():
            # Nothing selected: no delivery exists to confirm later, so this
            # keeps the historical four-key shape whether or not the caller
            # asked to defer.
            return out
        if not defer_charge:
            self._record_proactive_injection(channel, result)
            return out
        # Deferred: the two extra keys appear on exactly the path that produces
        # something to confirm. `charge_deferred` is the feature-detect signal a
        # caller uses to tell "this daemon deferred" from "this daemon predates
        # deferred charging and has already charged at retrieval".
        out["delivery_token"] = self._defer_proactive_charge(
            channel, session_id, result
        )
        out["charge_deferred"] = True
        return out

    def _defer_proactive_charge(self, channel: str, session_id: str, result) -> str:
        """Park a selected-but-unconfirmed result and return its opaque token.

        Writes a PROACTIVE_RETRIEVE audit event carrying exactly the metadata
        PROACTIVE_INJECT will later carry, so the durable log distinguishes
        "these items were selected and handed to a caller" from "these items
        reached the model" — see `_record_proactive_injection` for the full
        event contract.

        The parked entry holds the retrieval-time metadata (`chars`, `states`)
        that only this side knows: `/v1/proactive`'s response deliberately does
        not transport per-item lifecycle state, so replaying it from here at
        confirm time is what keeps the PROACTIVE_INJECT line's shape — and its
        anchors/states positional alignment — byte-identical to the
        charge-at-retrieval path.
        """
        token = secrets.token_hex(8)
        entry = {
            "channel": channel,
            "session_id": session_id,
            "anchor_ids": tuple(result.anchor_ids),
            "item_count": result.item_count,
            "chars": len(result.context),
            "states": tuple(result.states),
            # UPG-PROXY-INJECT-SINGLE-TURN: carried through so confirm-time
            # charging (below) can honor the SAME event-anchored exemption
            # the gate already decided at selection, instead of charging
            # every anchor in full once delivery is confirmed.
            "unretired_anchor_ids": tuple(result.unretired_anchor_ids),
        }
        with self._proactive_pending_lock:
            self._proactive_pending[token] = entry
            while len(self._proactive_pending) > PROACTIVE_MAX_PENDING_DELIVERIES:
                # Oldest-first eviction, bounding daemon state exactly like the
                # gate's own session ledger. An evicted entry is simply never
                # charged: its note may be injected again later (noise), which
                # is the safe direction — the failure this replaces was losing
                # the note permanently.
                self._proactive_pending.popitem(last=False)
        from agent.working_context_store import audit as _audit
        _audit(
            "PROACTIVE_RETRIEVE", workspace=self._workspace_root, channel=channel,
            items=result.item_count, anchors=",".join(result.anchor_ids),
            chars=len(result.context), states=",".join(result.states), token=token,
        )
        return token

    def _confirm_proactive_delivery(self, token: str) -> bool:
        """Charge a deferred retrieval whose block the caller confirms reached
        the model. Returns True when a pending entry was settled.

        Idempotent by construction: the entry is POPPED, so replaying the same
        token charges nothing the second time. An unknown or already-evicted
        token is a silent no-op — never an error, since this runs inside a
        request the caller must be able to complete regardless.
        """
        if not token:
            return False
        with self._proactive_pending_lock:
            entry = self._proactive_pending.pop(token, None)
        if entry is None:
            return False

        session_id = entry["session_id"]
        anchor_ids = entry["anchor_ids"]
        # UPG-PROXY-INJECT-SINGLE-TURN: exclude anchors the gate already
        # decided, at selection time, must stay eligible rather than retire
        # (a declared-anchor note whose file was matched but not yet edited
        # THAT request) — this confirms delivery of the whole block without
        # spending those anchors' cooldown slots. A later request whose
        # window shows the same file actually edited charges them normally.
        unretired = set(entry.get("unretired_anchor_ids") or ())
        chargeable_ids = tuple(aid for aid in anchor_ids if aid not in unretired)
        if session_id and self._proactive_ledger is not None and chargeable_ids:
            self._proactive_ledger.record(session_id, chargeable_ids)
        # Cross-channel turn dedup: claim here rather than at selection, so the
        # claim marks a real delivery. LOOKUP-ONLY, exactly as on the selection
        # path — `_existing_turn_ledger`, never `_turn_ledger_for` — so the
        # proxy channel's process-scoped session id still cannot allocate a
        # ledger entry and evict a live editor session's real one. Today this
        # is a guaranteed miss on the proxy path; it starts deduping for free
        # if session identity is ever unified.
        # UPG-PROXY-INJECT-SINGLE-TURN: unlike `_proactive_ledger` above, the
        # turn ledger claims EVERY delivered anchor (not just `chargeable_ids`)
        # — it is a same-turn, cross-surface dedup that resets every turn
        # regardless, so an unretired item still needs claiming to suppress a
        # hook surface delivering the same note again this same turn.
        turn_ledger = self._existing_turn_ledger(session_id)
        if turn_ledger is not None:
            from agent.proactive.gate import _note_id_from_anchor
            for aid in anchor_ids:
                note_id = _note_id_from_anchor(aid)
                if note_id is not None:
                    turn_ledger.claim(note_id)

        self._emit_proactive_injection(
            entry["channel"],
            items=entry["item_count"],
            anchors=anchor_ids,
            chars=entry["chars"],
            states=entry["states"],
            token=token,
        )
        return True

    def _record_proactive_injection(self, channel: str, result) -> None:
        self._emit_proactive_injection(
            channel,
            items=result.item_count,
            anchors=tuple(result.anchor_ids),
            chars=len(result.context),
            states=tuple(result.states),
        )

    def _emit_proactive_injection(
        self, channel: str, *, items: int, anchors, chars: int, states, token: str = "",
    ) -> None:
        """Count and durably record ONE injection that reached the model.

        AUDIT EVENT CONTRACT (UPG-PROXY-AUDIT-CLAIMS-UNDELIVERED-INJECTION).
        `PROACTIVE_INJECT` means DELIVERED — the block entered the model's
        context. It is written here and only here, from two places:

          * the charge-at-retrieval path, where returning the block IS
            delivering it (`proactive_context` with `defer_charge=False`); and
          * `_confirm_proactive_delivery`, when a deferred caller confirms the
            block was appended. Deferred retrievals additionally emit a
            `PROACTIVE_RETRIEVE` event at selection time (see
            `_defer_proactive_charge`), so a log consumer reads the pair as:
            RETRIEVE = selected and handed to a caller; INJECT = confirmed
            delivered; RETRIEVE with no matching INJECT (join on `token=`) =
            selected but NOT delivered. Legacy callers emit no RETRIEVE line at
            all, which is itself the signal that the log is recording the
            retrieval-time contract.

        UNDER-REPORT RESIDUAL, deliberate: on the deferred path a caller that
        delivers the block and then dies before its next call never returns the
        token, so no PROACTIVE_INJECT is written and the cooldown is never
        charged for an injection that DID reach the model. `PROACTIVE_INJECT`
        and `proactive_injection_counts` are therefore a strict LOWER BOUND on
        delivery across caller crashes. That is the correct direction to fail —
        the defect this replaces over-reported, asserting injections that never
        happened — but a consumer must not read the count as exact.

        Metadata-only (design §9): ids + counts + size + per-item lifecycle
        state, never the conversation text or note bodies. `chars` is the exact
        length of the injected `context` string (what actually enters the
        prompt) so budget adherence (`max_chars_per_event`) is observable from
        the durable log without reconstruction. `states` is positionally
        aligned with `anchors` — joining them tells an operator which selected
        items were injected as raw assertions ("active") vs. deterrents
        ("revoked"), which `items` alone cannot. `token` appears only on the
        deferred path, where it joins the line to its `PROACTIVE_RETRIEVE`.
        """
        with self._proactive_injection_lock:
            self._proactive_injection_counts[channel] = (
                self._proactive_injection_counts.get(channel, 0) + 1
            )
        from agent.working_context_store import audit as _audit
        fields = dict(
            workspace=self._workspace_root, channel=channel,
            items=items, anchors=",".join(anchors),
            chars=chars, states=",".join(states),
        )
        if token:
            fields["token"] = token
        _audit("PROACTIVE_INJECT", **fields)

    def get_proactive_injection_counts(self) -> dict:
        with self._proactive_injection_lock:
            return dict(self._proactive_injection_counts)

    def cache_metrics(self) -> dict | None:
        """Org-wide artifact-cache metrics for `status` (None when off)."""
        if self._artifact_cache is None:
            return None
        return self._artifact_cache.metrics()

    def snapshot_session(self, label: str, session_id: str | None = None) -> str:
        self._require_memory_layer()
        return self._context_store.snapshot(
            workspace=self._workspace_root,
            label=label,
            retrieved_chunks=self._advisor_for(session_id).as_chunk_dicts(),
            session_id=session_id,
        )

    def list_snapshots(self) -> list[dict]:
        self._require_memory_layer()
        return self._context_store.list_snapshots(self._workspace_root)

    def restore_snapshot(self, snapshot_id: str) -> dict | None:
        self._require_memory_layer()
        return self._context_store.restore_snapshot(snapshot_id)

    def resume(self, session_id: str | None = None, surface: str = "mcp") -> dict:
        """UPG-RESUME-SURFACE: 'pick up where you left off' as one named,
        discoverable surface (CLI `vectr resume` / REST GET /v1/resume / MCP
        `vectr_resume`) over internals SessionStart boot injection already
        uses.

        Selection is computed exactly once, entirely in
        `WorkingContextStore.resume_state()` — shared verbatim with
        `boot_recall()`'s own task-note query (`_boot_task_notes()`), so this
        surface and SessionStart injection can never disagree on which task
        note is "current". This method only adds staleness flags and two
        renderings on top of that one selection: a structured, JSON-safe
        summary per note (for REST/programmatic callers) and one formatted
        text block (for CLI/MCP, via `format_resume()`).

        Works identically in memory-only mode (the memory layer is fully
        active there); raises in search-only mode via `_require_memory_layer`
        — callers (REST/MCP) intercept that earlier with the shared
        `_SEARCH_ONLY_MSG`, same as every other memory method.
        """
        from agent.working_context_store import _note_title

        self._require_memory_layer()
        state = self._context_store.resume_state(self._workspace_root, session_id=session_id)
        last_task = state["last_task"]
        gotchas = state["gotchas"]
        snapshot = state["snapshot"]

        notes_for_staleness = ([last_task] if last_task else []) + gotchas
        stale = (
            self._context_store.check_staleness(notes_for_staleness, self._workspace_root)
            if notes_for_staleness else {}
        )

        def _summary(note) -> dict:
            # Index-tier only (no body) — matches vectr_recall's own
            # index-tier contract, so a JSON/REST consumer gets the same
            # bounded fields the CLI/MCP text render shows, and expands via
            # note_id (vectr_recall(note_id=N) / vectr recall --id N).
            return {
                "note_id": note.note_id,
                "kind": note.kind,
                "priority": note.priority,
                "title": _note_title(note),
                "created_at": note.created_at,
                "anchors": [a[0] for a in (note.anchors or []) if a],
                "stale": note.note_id in stale,
            }

        formatted = self._context_store.format_resume(
            state, self._workspace_root, stale_warnings=stale, surface=surface,
        )

        return {
            "last_task": _summary(last_task) if last_task else None,
            "gotchas": [_summary(g) for g in gotchas],
            "snapshot": snapshot,
            "gotchas_truncated": state["gotchas_truncated"],
            "formatted": formatted,
        }

    # ------------------------------------------------------------------
    # Eviction advisor
    # ------------------------------------------------------------------

    def eviction_hint(self, session_id: str | None = None, on_demand: bool = False) -> str:
        # UPG-7.2: the explicit vectr_evict_hint tool / /v1/evict pass
        # on_demand=True for an eviction-focused informational framing distinct
        # from the gated auto-footer's remember alarm.
        return self._advisor_for(session_id).eviction_hint(
            on_demand=on_demand, workspace_root=self._workspace_root,
        )

    def auto_eviction_hint(self, session_id: str | None = None) -> str:
        """Gated per-response footer (UPG-7.1) — fires only on fresh context-
        pressure escalation, not every response. Used by the MCP search/locate/
        trace auto-append; the explicit vectr_evict_hint tool uses eviction_hint().
        Reads the calling session's advisor (UPG-EVICT-SESSION-SCOPE)."""
        return self._advisor_for(session_id).auto_eviction_hint(
            workspace_root=self._workspace_root,
        )

    def should_evict(self, session_id: str | None = None) -> bool:
        return self._advisor_for(session_id).should_evict()

    def count_notes(self) -> int:
        """Return number of active working-memory notes for this workspace.

        Search-only mode has no context store — always 0, never an error, so
        `status()` and `suggest_instruction_style()` stay usable unconditionally.
        """
        if self._context_store is None:
            return 0
        return self._context_store.count_notes(self._workspace_root)

    def stale_task_summary(self) -> tuple[int, int | None]:
        """Count of live kind="task" notes older than
        MEMORY_HYGIENE_STALE_TASK_WARN_AGE_DAYS, plus the oldest such note's
        id (UPG-TASK-SUPERSEDES-HYGIENE). Feeds the `vectr_status`
        memory-hygiene nudge. Search-only mode has no context store — always
        (0, None), never an error.
        """
        if self._context_store is None:
            return 0, None
        return self._context_store.stale_task_summary(
            self._workspace_root, MEMORY_HYGIENE_STALE_TASK_WARN_AGE_DAYS
        )

    def _stale_task_nudge_line(self) -> str:
        """One-line stale-task hygiene nudge, or "" when below the warn count
        (UPG-NUDGE-HOOK-PATH-UNREACHABLE). The same signal vectr_status renders,
        surfaced on the SessionStart injection path so it reaches agents that
        never call status."""
        count, oldest_id = self.stale_task_summary()
        if count < MEMORY_HYGIENE_STALE_TASK_WARN_COUNT:
            return ""
        return (
            f"Memory hygiene: {count} task note(s) older than "
            f"{MEMORY_HYGIENE_STALE_TASK_WARN_AGE_DAYS} days are still active "
            f"(oldest: #{oldest_id}) — vectr_remember(kind=\"task\", supersedes=<old id>) "
            "if the work moved on, or vectr_forget(note_id=...) if it's done."
        )

    def _arc_distill_nudge_line(self) -> str:
        """One-line pending-arc distillation nudge, or "" when there are
        none — appended to the same
        session-start/post-compaction injection payload the stale-task
        nudge above rides. Deterministic integer-count comparison on
        vectr's own store — no content classification, nothing gated or
        rerouted. Suppressed at count 0; suppressed in search-only mode by
        construction, since the boot branch that calls this never runs
        there (`_recall_impl` raises via `_require_memory_layer()` first)."""
        count = self.count_arcs_pending_distill()
        if count <= 0:
            return ""
        return (
            f"{count} command-discovery arcs are pending distillation — call "
            "vectr_distill() to review them and persist the lessons as notes "
            "(or dismiss them)."
        )

    def _boundary_precompact_text(self, arcs_pending: int | None = None) -> str:
        """Render the PreCompact boundary-preservation text for
        GET /v1/boundary/precompact.

        `arcs_pending` lets the caller pass a count it has ALREADY read, so
        one request renders from a single store snapshot; omitted, the count
        is read here. `boundary_precompact()` below passes it — otherwise the
        rendered sentence and the response's own `arcs_pending` field would
        come from two separate reads and could disagree if an arc landed
        between them.

        Server-rendered so the product copy has one home — the pre-compact
        hook branch (main.py / agent/hook_cli.py) only fetches and prints
        this verbatim as plain text, never composes it itself.

        Returns "" when the surface is disabled by config. Otherwise
        returns the arc-independent base text (BOUNDARY_PRECOMPACT_ENABLED
        guards the whole call, not just the arc sentence), with ONE
        pending-arc count sentence appended when
        `count_arcs_pending_distill()` is positive AND the combined text
        still fits BOUNDARY_PRECOMPACT_TOKEN_CAP (agent/trigger_engine.py's
        token_estimate — the same estimator every other injection surface
        uses). The base text alone is small enough to always fit; only the
        arc sentence is ever dropped for budget.

        Deterministic count comparison on vectr's own store, same as
        `_arc_distill_nudge_line` above — no content classification,
        nothing gated or rerouted by query/session content. Contains no
        note bodies or ids: notes are already durable and already
        re-injected after compaction via the SessionStart `compact`
        matcher (UPG-9.4) — restating them here would double the same
        information rather than adding to it.
        """
        if not BOUNDARY_PRECOMPACT_ENABLED:
            return ""
        text = _BOUNDARY_PRECOMPACT_BASE_TEXT
        count = (
            self.count_arcs_pending_distill() if arcs_pending is None else arcs_pending
        )
        if count > 0:
            arc_sentence = (
                f"{count} command-discovery arc(s) recorded this session are still "
                "pending distillation; keep the reasoning about what they mean."
            )
            candidate = f"{text}\n\n{arc_sentence}"
            if token_estimate(candidate) <= BOUNDARY_PRECOMPACT_TOKEN_CAP:
                text = candidate
        return text

    def boundary_precompact(self) -> dict:
        """Read surface for `GET /v1/boundary/precompact` — the rendered
        text (`_boundary_precompact_text` above) alongside the current
        pending-arc count. Memory-layer gated by the caller: the route
        checks `search_only` before calling here (mirroring `GET
        /v1/arcs`), since `count_arcs_pending_distill()` is already
        zero-safe in search-only mode and there is nothing else here to
        guard.

        The pending count is read ONCE and shared with the renderer, so the
        count named in `text` and the `arcs_pending` field are always the
        same number — and one request costs one store read, not two, on a
        hook path that runs while compaction waits."""
        arcs_pending = self.count_arcs_pending_distill()
        return {
            "text": self._boundary_precompact_text(arcs_pending=arcs_pending),
            "arcs_pending": arcs_pending,
        }

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Adaptive prompt intelligence
    # ------------------------------------------------------------------

    def suggest_instruction_style(self) -> str:
        """Return the recommended CLAUDE.md instruction variant for this workspace.

        Returns one of: "additive" | "directed" | "memory-first".

        Decision logic (priority order):
        1. File override (.vectr/style) — always wins.
        2. "memory-first" — prior notes exist AND codebase is well-known (small
                            or familiar framework) → recall-forward session.
        3. "directed"     — large or complex unfamiliar codebase → explicit tool
                            guidance reduces wasted exploration turns.
        4. "additive"     — default; model decides based on when-to-use hints.

        Research basis: additive outperforms forced/memory-first in A/B tests
        (spec §CLAUDE.md framing choices). directed is warranted only when
        codebase is genuinely unfamiliar at implementation depth.

        Note: this is a CLAUDE.md *authoring-style* label, unrelated to
        `memory_only` (the operating mode that hard-disables search — see
        `status()`'s "mode" field). UPG-TOOLSTYLE-LABEL-COLLISION renamed
        this label from the identical-looking "memory-only" string so the
        two concepts never render side by side looking contradictory (e.g.
        `vectr_status`'s "Mode: full" next to "Tool style: [memory-only]").
        """
        override = self._read_style_override()
        if override in ("additive", "directed", "memory-first"):
            return override

        notes_count = self.count_notes()
        fp = None
        if self._strategy is not None:
            try:
                from agent.strategy_selector import fingerprint as _fingerprint
                fp = _fingerprint(self._workspace_root, self._indexer.indexed_file_paths)
            except Exception:
                pass

        # Well-known frameworks: model knows these at implementation depth from
        # training. The set is config-declared (STRATEGY_KNOWN_FRAMEWORKS,
        # UPG-KNOWN-FRAMEWORKS-CONFIG); match case-insensitively.
        known_codebase = (
            fp is not None
            and bool(
                STRATEGY_KNOWN_FRAMEWORKS.intersection(
                    f.lower() for f in fp.detected_frameworks
                )
            )
        )

        if notes_count > 0 and (known_codebase or (fp and fp.size_class == "small")):
            return "memory-first"

        if fp is not None:
            is_large_unfamiliar = (
                fp.size_class == "large"
                and not known_codebase
            )
            is_complex = fp.complexity_class == "complex" and not known_codebase
            if is_large_unfamiliar or is_complex:
                return "directed"

        return "additive"

    def _read_style_override(self) -> str:
        """Read .vectr/style file if present.

        UPG-TOOLSTYLE-LABEL-COLLISION renamed the "memory-only" style label
        to "memory-first". A workspace whose .vectr/style file was written
        by a pre-rename `vectr init --style memory-only` still has that
        literal string on disk — accept it transparently on read (the file
        is never rewritten here; only a fresh `vectr init --style ...` run
        writes the current label).
        """
        style_file = Path(self._workspace_root) / ".vectr" / "style"
        try:
            raw = style_file.read_text(encoding="utf-8").strip()
        except OSError:
            return ""
        if raw == "memory-only":
            return "memory-first"
        return raw
