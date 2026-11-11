"""Pydantic v2 request and response models."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from agent.working_context_store import SORT_BY_VALUES


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------

class IndexRequest(BaseModel):
    path: str = Field(default=".", description="Absolute or relative path to index")
    force: bool = Field(default=False, description="Force full re-index even if already indexed")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language or code query")
    n_results: int = Field(default=10, ge=1, le=50, description="Number of results to return")
    # UPG-3.1: any indexed language is accepted (no fixed allow-list, no 422).
    # Normalisation (lower/strip, blank→None) lives in CodeSearcher.search — the
    # shared path for both REST and MCP — so behaviour can't diverge by entrypoint.
    language: str | None = Field(
        default=None,
        description="Filter to a specific indexed language (e.g. python, rust, c, zig). "
                    "Any language the index actually contains is accepted; unindexed "
                    "languages return no matches rather than an error.",
    )


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

class CodeChunkResult(BaseModel):
    file: str
    lines: str
    symbol: str | None   # None when chunk falls outside any named symbol
    language: str
    score: float
    content: str
    # UPG-11.4: expand-to-symbol affordance — the exact line range of the indexed
    # symbol so callers can read the full definition without a blind whole-file re-read.
    # 0/0 means the chunk was not associated with a named symbol (e.g. a window chunk).
    symbol_start_line: int = 0
    symbol_end_line: int = 0
    # UPG-CTX-EVICT: the exact chunk id — pass verbatim to vectr_fetch /
    # POST /v1/fetch to restore this chunk deterministically, no re-search
    # or file re-read needed.
    id: str = ""
    # UPG-SCORE-DISPLAY-MIXED-SCALE: which scale `score` was drawn from for
    # THIS chunk — "reranker" (a calibrated cross-encoder judgment) or
    # "dense" (a raw bi-encoder cosine similarity). The two scales are not
    # comparable to each other; every result in one response is guaranteed to
    # share the same value (CodeSearcher.search backfills any straggler so a
    # response is never a mix of the two once reranking has run for it).
    score_source: str = "dense"


class SearchResponse(BaseModel):
    results: list[CodeChunkResult]
    query_time_ms: int
    chunks_searched: int
    processing_ms: int
    # UPG-NOTFOUND-FLOOR (F46/F52): true when the query names a concept that
    # has no lexical anchor anywhere in the indexed corpus — at least
    # `ranking.notfound_floor.min_zero_df_tokens` of its content words have
    # zero document frequency across the whole corpus, not just the query's
    # candidate pool. The displayed per-result `score` is a per-query
    # rank-derived value that always looks confident near the top, so this
    # field is the caller's only signal that the whole result set may be a
    # weak/unrelated guess. Results are still returned in full; this never
    # suppresses them.
    low_confidence: bool = False


class IndexResponse(BaseModel):
    indexed_files: int
    total_chunks: int
    processing_ms: int


class FetchRequest(BaseModel):
    ids: list[str] = Field(
        ..., min_length=1,
        description="Chunk ids to re-fetch verbatim — the exact `file:start-end` "
                    "id shown in a search/locate/trace result.",
    )


class FetchEntry(BaseModel):
    id: str
    found: bool
    file_path: str = ""
    lines: str = ""
    symbol: str | None = None
    language: str = ""
    content: str = ""
    # UPG-FETCH-ID-MISALIGN-MSG: only set when found=False. "misaligned"
    # means the same file has other chunks indexed — the requested line
    # range just doesn't match a stored span; `nearest_ids` names the
    # closest real chunk ids so the caller's next fetch can succeed without
    # a round trip. "file_changed" means the file has no chunks indexed at
    # all — the file most likely changed since indexing.
    reason: str | None = None
    nearest_ids: list[str] = []


class FetchResponse(BaseModel):
    results: list[FetchEntry]
    # Shared note, present only when at least one requested id was NOT found
    # for the "file_changed" reason above — genuinely absent/moved/deleted
    # files, not misaligned-but-present ones (those carry their own
    # per-entry reason + nearest_ids instead).
    note: str | None = None
    processing_ms: int


class LanguageStat(BaseModel):
    """Per-language index coverage + symbol availability (UPG-3.3).

    `symbols=True` means locate/trace work for this language; otherwise it is
    search-only. Lets a REST consumer route the same way the MCP agent does.
    """
    language: str
    files: int
    chunks: int
    symbols: bool


class StatusResponse(BaseModel):
    indexed_files: int
    total_chunks: int
    last_indexed: str
    embed_model: str
    workspace_root: str
    symbol_count: int = 0
    languages: list[LanguageStat] = []
    notes_count: int = 0
    # Episode-capture aggregates: plain counts, never rows (status is not a
    # reader of episode content — only GET /v1/episodes is). Always an int —
    # 0 in search-only mode or before the first episode write, mirroring
    # notes_count's zero-safe contract. Declared here because the service
    # dict is filtered through this model: an undeclared key is silently
    # dropped from the REST response even when status() provides it.
    episodes_count: int = 0
    arcs_pending_distill: int = 0
    # Symbol-graph build trust signals (UPG-8.7): complete = no files failed
    # extraction; failed_files counts those skipped, so a partial graph is visible.
    symbol_graph_complete: bool = False
    symbol_graph_failed_files: int = 0
    # UPG-STATUS-STALE-GRAPH: staleness signals for the symbol-graph fields above.
    # `stale` = the persisted graph was built by a different toolchain (vectr/
    # parser/model/schema change), left incomplete, or never built — so the
    # symbol_count / symbol_graph_complete served may not reflect current state.
    # `rebuild_in_progress` = a full rebuild is overwriting the graph right now,
    # so those fields still reflect the OLD graph until it lands.
    symbol_graph_stale: bool = True
    symbol_graph_rebuild_in_progress: bool = False
    processing_ms: int
    # Adaptive retrieval strategy. Always populated by VectrService.status()
    # (UPG-8.2): the config-declared defaults before the first index-time
    # fingerprint, the fingerprint-derived values after. Optional here only
    # so a stale/partial mock in a test doesn't fail response validation.
    semantic_weight: float | None = None
    bm25_weight: float | None = None
    graph_first: bool | None = None
    recommended_embed_model: str | None = None
    strategy_rationale: str | None = None
    # Daemon mode: "full" (default), "memory-only" (no indexing/watcher), or
    # "search-only" (no working-memory layer — see UPG-SEARCH-ONLY-MODE)
    mode: str = "full"
    # Set only when the working-memory note vectors are stamped with a
    # different embed model than the one currently configured (should not
    # persist past startup — migration runs synchronously — see
    # UPG-NOTES-EMBED-MIGRATION). None means the two agree.
    notes_embed_model_mismatch: str | None = None
    # UPG-CLI-DAEMON-VERSION-SKEW: package version (+ short git SHA when the
    # daemon runs from a git checkout), stamped once at process startup. The
    # CLI recomputes the same stamp per invocation and warns on mismatch.
    # Optional so a stale/partial mock in a test doesn't fail validation.
    version_stamp: str | None = None
    # Watcher backlog observability (UPG-WATCHER-PRESSURE-GOVERNOR): whether
    # a sustained multi-file edit stream has coalesced into burst mode,
    # outstanding paths not yet re-indexed (per-file debounce + burst
    # collection + queued batch), whether a batch worker is currently
    # running, and the last batch's wall-clock duration — so runaway churn
    # is visible instead of silent.
    watcher_burst_mode: bool = False
    watcher_pending_files: int = 0
    watcher_batch_running: bool = False
    watcher_last_batch_duration_ms: int = 0
    # Hook-driven injection counters (UPG-HOOK-INJECT-OBSERVABILITY): how many
    # times each hook kind's recall actually returned notes to inject, since
    # this daemon process started. Only counts hook-declared calls (see
    # RecallRequest.hook_event); direct vectr_recall/`vectr recall` calls are
    # never counted here. Empty dict means no hook has injected notes yet.
    hook_injection_counts: dict[str, int] = {}
    # Proactive-context injection counters (UPG-PRO): how many times each
    # channel (e.g. "proxy") actually injected packed context since startup.
    # Empty until proactive context has injected anything.
    proactive_injection_counts: dict[str, int] = {}
    # Effective ambient (hook-channel) proactive master opt-in, visible before
    # any injection has happened. The proxy channel injects by launch consent
    # regardless of this flag (UPG-PROXY-HIDDEN-MASTER-SWITCH).
    proactive_enabled: bool = False
    # Org-wide artifact-cache metrics (UPG-PRO caching): hits / misses /
    # hit_rate / entries / est_tokens_saved, or None when the cache is off.
    artifact_cache: dict | None = None
    # UPG-REST-STARVATION: true while bulk index work (an explicit
    # index()/startup index, or the watcher's coalesced batch worker) is
    # running. Backed by a non-blocking lock read + in-memory watcher flags —
    # this field is always cheap to compute, even mid-reindex.
    reindex_in_progress: bool = False
    # UPG-PURPOSE-PASS-DEFERRAL: true while a background purpose-vector pass
    # is scheduled/running (config `defer_purpose_pass`, default on) — search
    # still serves results during this window (body-similarity only), so
    # this makes a possible ranking-quality dip explainable instead of
    # silent. False once the pass completes, or if deferral never ran.
    purpose_vectors_pending: bool = False
    # UPG-PURPOSE-RESUME-HOLE: purpose (ARCH-4 dual-vector) chunk count,
    # surfaced alongside total_chunks so a caller can see the two diverge
    # instead of a green status hiding a gap (e.g. an interrupted deferred
    # pass leaving most of the corpus without purpose vectors). Was already
    # computed (CodeIndexer.total_purpose_chunks) but never rendered on any
    # status surface before this field existed.
    total_purpose_chunks: int = 0
    # Count of files the most recent index_workspace() run found with
    # current body content but incomplete purpose vectors, as of the START
    # of that run — 0 once dispatch for that backfill has been sent (not
    # necessarily landed for the deferred/background case; see
    # purpose_vectors_pending above for "still in flight").
    purpose_backfill_pending_files: int = 0
    # UPG-STDIO-MEMORY-READY: additive warm-up signals. `fully_ready` is True
    # once phase 2 (embedder/indexer/searcher/watcher/symbol-graph) has
    # completed — search/locate/trace/map/fetch are gated on it across every
    # transport. `embedder_ready` is True once the embedding model has
    # loaded/attached — vectr_recall's lexical-fallback notice is gated on
    # it. Both default True so an older/partial mock in a test still
    # validates, and both are True immediately for any daemon that wasn't
    # constructed with defer_search_init=True.
    fully_ready: bool = True
    embedder_ready: bool = True
    # UPG-TASK-SUPERSEDES-HYGIENE: count of live (non-superseded) kind="task"
    # notes older than MEMORY_HYGIENE_STALE_TASK_WARN_AGE_DAYS, plus the
    # oldest such note's id (None if there are none or the count is 0). A
    # nudge only — task notes are never auto-superseded or expired.
    stale_task_count: int = 0
    stale_task_oldest_id: int | None = None


class HealthResponse(BaseModel):
    status: str
    embed_model: str
    # Sourced from the same VectrService.last_indexed property as
    # /v1/status so the two endpoints never disagree on freshness (UPG-8.2).
    last_indexed: str


# ---------------------------------------------------------------------------
# Codebase passport
# ---------------------------------------------------------------------------

class MapSaveRequest(BaseModel):
    summary: str = Field(..., min_length=1, description="AI-written plain-English codebase summary")
    # UPG-6.2: vectr_map_save must not silently clobber an existing passport —
    # the caller must explicitly opt in to replace it.
    overwrite: bool = False


class MapSaveResponse(BaseModel):
    message: str
    processing_ms: int
    # False when a passport already existed and overwrite was not set — the
    # request was a no-op and `message` carries the existing summary (UPG-6.2).
    saved: bool = True


# ---------------------------------------------------------------------------
# Memory / working context
# ---------------------------------------------------------------------------

_VALID_KINDS = ("directive", "task", "gotcha", "finding", "reference", "decision", "operational")

# TRIGGER-ENGINE wave 1 (bm2-design-skeleton.md §1/§2/§5) — mirrors
# agent.working_context_store._types.SCOPE_VALUES/PROVENANCE_VALUES. Kept as
# a local closed-vocabulary tuple here rather than imported, matching this
# module's existing convention for _VALID_KINDS above (the REST/MCP request
# schema validates independently of the store's own internal validation).
_SCOPE_VALUES = ("workspace", "repo", "path-subtree", "branch", "session")
# The promotion ladder (mirrors _types.PROMOTION_LADDER, strongest last),
# used by PromoteRequest's `to` below. Deliberately does NOT include
# "user-stated": that class is derived at write time from a bound verbatim
# user excerpt (UPG-MEM-PROVENANCE-USER-STATED), never a promotion target.
_PROVENANCE_VALUES = ("human", "agent", "auto")
# TRIGGER-ENGINE wave 2a — mirrors
# agent.working_context_store._types.EVENT_VALUES, same local-tuple
# convention as _SCOPE_VALUES/_PROVENANCE_VALUES above.
_EVENT_VALUES = ("session-start", "prompt-submit", "pre-edit", "pre-run", "pre-commit", "post-compaction")
# 'human' is not settable via vectr_remember/RememberRequest — only 'agent'
# (default) or 'auto' may be declared at write time; promoting to 'human' is
# a separate, explicit call (PromoteRequest below) once a person has endorsed
# the note.
_REST_PROVENANCE_VALUES = ("agent", "auto")
# UPG-REST-SORTBY-UNVALIDATED — mirrors recall()'s sort_by contract
# (agent/working_context_store/_store.py::recall). Same local-tuple
# convention as _VALID_KINDS above; MCP (tool schema enum) and CLI
# (argparse choices) already enforce this vocabulary — without this tuple
# the REST surface accepted any string and the store silently treated an
# unknown value as relevance.
#
# UPG-SORTBY-SHARED-VOCAB: this is no longer a local definition — it is the
# same SORT_BY_VALUES constant the MCP dispatch, MCP tool schema, and CLI
# also import. Re-aliased under the historical private name (a handful of
# existing in-module call sites read `_SORT_BY_VALUES`) so that adding a
# fifth sort mode at the source automatically tightens the REST validator
# here, alongside the other three surfaces. Kept the private alias purely
# to minimise the diff at the two in-module consumers below; the
# authoritative declaration lives in agent.working_context_store._types.
_SORT_BY_VALUES = SORT_BY_VALUES


class RememberRequest(BaseModel):
    content: str | None = Field(
        default=None,
        description=(
            "Working note to store. Mutually exclusive with content_file — "
            "pass exactly one, not both."
        ),
    )
    content_file: str | None = Field(
        default=None,
        description=(
            "UPG-REMEMBER-MCP-LONG-PAYLOAD-PARSE-LOSS: path to a UTF-8 file "
            "containing the note body, as an alternative to inline content. "
            "Absolute, or relative to the workspace root; rejected if it "
            "resolves outside the workspace, is missing/unreadable, exceeds "
            "the configured size cap, or is not valid UTF-8. Mutually "
            "exclusive with content — pass exactly one, not both."
        ),
    )
    tags: list[str] | None = Field(default=None, description="Topic tags")
    priority: str = Field(
        default="medium",
        description=(
            "high | medium | low. Session-start/resume surfaces show only "
            "priority='high' kind='task' notes — a checkpoint meant for "
            "resume needs priority='high' (UPG-RESUME-TASK-PRIORITY-VISIBILITY)."
        ),
    )
    kind: str = Field(default="finding", description="directive | task | gotcha | finding | reference | decision | operational")
    session_id: str | None = Field(default=None)
    title: str = Field(default="", description="Short label for index-tier display (optional; derived from first content line if empty)")
    agent: str = Field(
        default="",
        description=(
            "Optional caller-declared identifier for the agent/subagent authoring this "
            "note (e.g. 'coder-2'), for multi-agent shared-memory attribution. Never "
            "inferred. Shown in recall index lines when present (e.g. "
            "'[#12] task/high (coder-2) · title'); absent renders exactly as before."
        ),
    )
    triggers: list[dict] | None = Field(
        default=None,
        description=(
            "TRIGGER-ENGINE wave 1: explicit P/E/T trigger overrides. Each "
            "entry declares 'path' (glob) and/or 'event' (session-start | "
            "prompt-submit | pre-edit | pre-run | pre-commit | "
            "post-compaction) plus optional 'not_before' / "
            "'expires_visibility' / 'cooldown' (T modifiers, never fire "
            "alone). Entries are OR'd together across the list; omitted or "
            "empty uses this note's kind's default trigger bundle, computed "
            "at evaluation time."
        ),
    )
    provenance: str = Field(
        default="agent",
        description=(
            "Trust/endorsement class (TRIGGER-ENGINE wave 1): 'agent' "
            "(default — an AI session recorded this, framed as memory to "
            "verify) or 'auto' (captured with no reviewing judgment, weakest "
            "framing; not allowed on kind='directive'). 'human' is not "
            "settable from this tool — see the separate promote call."
        ),
    )
    scope: str | None = Field(
        default=None,
        description=(
            "workspace | repo | path-subtree | branch | session. Omit "
            "(None) to get this note's kind's own default scope, resolved "
            "at write time (UPG-TRIGGER-SCOPE-KIND-DEFAULTS): kind='task' "
            "-> 'branch' (falls back to 'workspace' when no git branch was "
            "actually captured), kind='gotcha' -> 'repo', every other kind "
            "-> 'workspace'. Pass a value explicitly, including 'workspace', "
            "to override the kind default."
        ),
    )
    anchors: list[str] | None = Field(
        default=None,
        description=(
            "Workspace-relative (or absolute) file paths this note is "
            "anchored to. Each path's current content hash is computed at "
            "write time and re-checked at recall/fire time — a mismatch adds "
            "a visible staleness caveat, the note still fires/recalls "
            "(never silently dropped)."
        ),
    )
    supersedes: int | None = Field(
        default=None,
        description=(
            "note_id this new note explicitly tombstones. The target note "
            "is excluded from recall/fire from now on but retained, "
            "unchanged, for audit. Rejected if the target does not exist in "
            "this workspace."
        ),
    )
    contradicts: int | None = Field(
        default=None,
        description=(
            "note_id this new note asserts was WRONG (UPG-MEMORY-STATE-"
            "MACHINE) — distinct from `supersedes` ('replaced by something "
            "better'): the target note stays a live recall/fire candidate "
            "(never excluded), but every surface that renders it "
            "afterward substitutes an anti-memory deterrent block for its "
            "raw content until reinstated. Appends a `revoked` event to "
            "the target, always actor='agent', reason=f'contradicted by "
            "#<this note id>'. Rejected if the target does not exist in "
            "this workspace."
        ),
    )
    user_quote: str = Field(
        default="",
        description=(
            "Optional verbatim excerpt of the user turn this note "
            "transcribes (UPG-MEM-PROVENANCE-USER-STATED). Bound only if it "
            "is genuinely quoted inside `content` (whitespace-normalized "
            "substring — a deterministic string check, never semantic); a "
            "bound excerpt stores the note as provenance='user-stated', a "
            "class between 'human' and 'agent'. An excerpt that does not "
            "bind is discarded, the note is stored at its ordinary "
            "provenance, and the reason is appended to the response message "
            "— never an error."
        ),
    )
    distilled_from: list[int] | None = Field(
        default=None,
        description=(
            "Arc ids (from GET /v1/arcs or vectr_distill()) this note "
            "distills. After the note "
            "write succeeds, each named arc is marked distilled into this "
            "note's id. Unknown/already-resolved ids are reported back in "
            "the response, never an error (idempotent)."
        ),
    )
    pin: bool = Field(
        default=False,
        description=(
            "UPG-RECALL-MISS-FLOOR part (b): pin this note into Tier 0 at "
            "write time — equivalent to a separate POST /v1/pin call right "
            "after this write. Pinned notes are injected unconditionally "
            "on every vectr_recall(query=...) call, alongside "
            "kind='directive' notes, up to the configured Tier 0 bound. "
            "Default False changes nothing about existing callers."
        ),
    )

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v: str) -> str:
        if v not in ("high", "medium", "low"):
            raise ValueError("priority must be high, medium, or low")
        return v

    @field_validator("kind")
    @classmethod
    def validate_kind(cls, v: str) -> str:
        if v not in _VALID_KINDS:
            raise ValueError(f"kind must be one of: {', '.join(_VALID_KINDS)}")
        return v

    @field_validator("provenance")
    @classmethod
    def validate_provenance(cls, v: str) -> str:
        if v not in _REST_PROVENANCE_VALUES:
            raise ValueError(
                f"provenance must be one of: {', '.join(_REST_PROVENANCE_VALUES)} "
                "('human' is not settable here — promote a note to 'human' separately)"
            )
        return v

    @field_validator("scope")
    @classmethod
    def validate_scope(cls, v: str | None) -> str | None:
        # None means "omitted" (UPG-TRIGGER-SCOPE-KIND-DEFAULTS) — passed
        # straight through so the store resolves this note's kind's default
        # scope at write time; only a NON-None value is checked against
        # SCOPE_VALUES here.
        if v is not None and v not in _SCOPE_VALUES:
            raise ValueError(f"scope must be one of: {', '.join(_SCOPE_VALUES)}")
        return v


class RelatedNoteModel(BaseModel):
    """One nearest EXISTING, currently-active note surfaced alongside a
    fresh write (agent/working_context_store/_related.py) — a similarity
    ranking only, never a contradiction judgment. The caller decides
    whether this note is still correct, superseded, or unrelated."""

    note_id: int
    title: str
    kind: str
    # Carried through from the RelatedNote TypedDict, which has always
    # computed it. Without this field pydantic's default extra='ignore'
    # silently dropped it, so a REST caller could not tell a high-priority
    # related note from a low-priority one — exactly the signal that
    # decides whether a near-duplicate is worth acting on.
    priority: str
    similarity: float
    created_at: float


class RevokedRelatedNoteModel(BaseModel):
    """The nearest EXISTING note to a fresh write whose folded lifecycle
    state is currently `revoked` (agent/working_context_store/_related.py
    `revoked_related_notes()`, UPG-RELATED-REVOKED-DETERRENT) — a deterrent,
    not a similarity nudge like `RelatedNoteModel` above: it means an agent
    is about to write something close to a note the user or a prior agent
    turn already marked wrong. `revoked_date`/`reason` come from the
    revocation event itself, not the note's original metadata."""

    note_id: int
    title: str
    similarity: float
    revoked_date: str
    reason: str


class RememberResponse(BaseModel):
    note_id: int
    message: str
    processing_ms: int
    distilled: ArcResolveResult | None = Field(
        default=None,
        description="Present only when `distilled_from` was passed — the arc ids resolved into this note vs. any that were unknown/already-resolved.",
    )
    related: list[RelatedNoteModel] = Field(
        default_factory=list,
        description=(
            "Nearest EXISTING active notes to the one just written, nearest-first "
            "(agent/working_context_store/_related.py) — empty unless "
            "memory_write.related_notes.enabled and at least one candidate clears "
            "the configured similarity floor. Similarity is closeness, never a "
            "contradiction verdict; the caller judges whether one is now stale."
        ),
    )
    revoked_related: list[RevokedRelatedNoteModel] = Field(
        default_factory=list,
        description=(
            "The nearest EXISTING note to the one just written whose folded "
            "lifecycle state is currently 'revoked' (agent/working_context_store/"
            "_related.py revoked_related_notes(), UPG-RELATED-REVOKED-DETERRENT) — "
            "empty unless memory_write.related_notes.revoked_enabled and a "
            "candidate clears the configured similarity floor. A deterrent, not a "
            "similarity nudge: it means this write is close to something already "
            "marked wrong."
        ),
    )
    proxy_anchor_suggestions: list[str] = Field(
        default_factory=list,
        description=(
            "Workspace-relative proxy-anchor file paths present in this workspace "
            "(agent/proxy_anchors.py) — empty unless memory_write."
            "proxy_anchor_suggestions.enabled, kind='operational', and this call "
            "passed no anchors. Presence is glob-detected only, never content-read; "
            "attach one with vectr_anchor(note_id=<this note>, anchors=[<path>])."
        ),
    )


class RecallRequest(BaseModel):
    query: str | None = Field(default=None)
    tags: list[str] | None = Field(default=None)
    priority: str | None = Field(default=None)
    kind: str | None = Field(default=None, description="Filter by kind: directive | task | gotcha | finding | reference | decision | operational")
    limit: int = Field(default=10, ge=1, le=100)
    boot: bool = Field(default=False, description="Boot mode (UPG-9.2): unconditional directives + high-priority tasks; ignores query/tags/priority/kind/limit")
    min_similarity: float | None = Field(default=None, ge=0.0, le=1.0, description="Relevance cutoff (UPG-5.1): drop semantic matches below this cosine similarity; only applies with a query")
    file_path: str | None = Field(default=None, description="Path-anchored recall (UPG-9.6): notes recorded against this file (basename/relpath match); for the PreToolUse gotcha hook")
    command: str | None = Field(default=None, description="Command-anchored recall (UPG-MEMORY-STATE-MACHINE §5): raw about-to-run shell command (Bash tool_input.command) for the PreToolUse command-family trigger; normalized via app.cmdnorm before matching")
    max_age_days: float | None = Field(default=None, gt=0.0, description="Time filter (UPG-RECALL-HIERARCHY): only return notes created within this many days")
    sort_by: str = Field(default="relevance", description="Sort order (UPG-RECALL-HIERARCHY, UPG-DECISION-TIMELINE): relevance | recency | priority | chronological (oldest-first; index lines render the creation date)")
    detail: str = Field(default="index", description="Detail level (UPG-RECALL-HIERARCHY): 'index' = one-line summary per note (default, token-bounded); 'full' = full bodies")
    note_id: int | None = Field(default=None, description="Expand a single note by ID (UPG-RECALL-HIERARCHY): returns full body, ignores query")
    surface: str = Field(
        default="mcp",
        description=(
            "Caller surface (UPG-CLI-RECALL-HINT): 'mcp' (default) — the "
            "response's expand hint uses the MCP tool-call form, correct for "
            "the MCP dispatch path and for hook-injected context (both are "
            "ultimately read by the editor's LLM). 'cli' — the hint uses the "
            "real shell form; set by `vectr recall`'s own request, whose "
            "reader is a human terminal."
        ),
    )
    hook_event: str | None = Field(
        default=None,
        description=(
            "Caller-declared hook kind (UPG-HOOK-INJECT-OBSERVABILITY): "
            "'SessionStart' | 'UserPromptSubmit' | 'PreToolUse', set only by "
            "`vectr hook`'s own request. When set and this call actually "
            "returns notes, the daemon counts one injection under this hook "
            "kind (surfaced in `vectr status`). None (the default — used by "
            "direct vectr_recall/`vectr recall` calls) records nothing; only "
            "harness-injected recall is counted."
        ),
    )
    session_id: str | None = Field(
        default=None,
        description=(
            "TRIGGER-ENGINE wave 2a: the calling session's identity — for a "
            "hook-injected call this is the harness's own hook-JSON session "
            "id (the same id across a /compact). Enforces scope='session' "
            "notes, threads into the per-session cumulative injection "
            "budget and fire-dedup ledger, and (combined with the current "
            "git branch) enforces scope='branch' notes. None (the default) "
            "reproduces today's ledger-less, budget-less, scope-unenforced "
            "behaviour exactly."
        ),
    )
    events: list[str] | None = Field(
        default=None,
        description=(
            "TRIGGER-ENGINE wave 2a: lifecycle event(s) this recall stands "
            "in for, OR'd together — e.g. ['session-start', "
            "'post-compaction'] for the first SessionStart call after a "
            "/compact, or ['prompt-submit'] for a per-turn recall. Only "
            "notes with an EXPLICIT triggers[] override on one of these "
            "events (or, in boot mode, a kind-default bundle) fire; absent "
            "(the default), boot mode implies ['session-start'] and every "
            "other mode fires nothing extra beyond its own query/file_path "
            "match."
        ),
    )

    @field_validator("kind")
    @classmethod
    def validate_kind(cls, v: str | None) -> str | None:
        if v is not None and v not in _VALID_KINDS:
            raise ValueError(f"kind must be one of: {', '.join(_VALID_KINDS)}")
        return v

    @field_validator("sort_by")
    @classmethod
    def validate_sort_by(cls, v: str) -> str:
        # UPG-REST-SORTBY-UNVALIDATED: reject unknown values with a 422 at
        # the REST boundary instead of letting the store silently treat
        # them as relevance (same precedent as validate_kind above).
        if v not in _SORT_BY_VALUES:
            raise ValueError(f"sort_by must be one of: {', '.join(_SORT_BY_VALUES)}")
        return v

    @field_validator("surface")
    @classmethod
    def validate_surface(cls, v: str) -> str:
        if v not in ("mcp", "cli"):
            raise ValueError("surface must be one of: mcp, cli")
        return v

    @field_validator("hook_event")
    @classmethod
    def validate_hook_event(cls, v: str | None) -> str | None:
        if v is not None and v not in ("SessionStart", "UserPromptSubmit", "PreToolUse"):
            raise ValueError("hook_event must be one of: SessionStart, UserPromptSubmit, PreToolUse")
        return v

    @field_validator("events")
    @classmethod
    def validate_events(cls, v: list[str] | None) -> list[str] | None:
        if v is not None:
            for ev in v:
                if ev not in _EVENT_VALUES:
                    raise ValueError(f"each event must be one of: {', '.join(_EVENT_VALUES)}")
        return v


class RecallResponse(BaseModel):
    notes: str
    processing_ms: int


class ProactiveRequest(BaseModel):
    """An already-assembled proactive window (UPG-PRO-7). The caller (the proxy,
    or a future hook adapter) extracts the window; the daemon does the matching
    and gating. Kept structured — no raw request body is sent here."""

    text: str = Field(default="", description="Assembled recent-conversation text (the semantic query)")
    file_paths: list[str] = Field(default_factory=list, description="Deterministic file-path anchors from tool traffic")
    symbols: list[str] = Field(default_factory=list, description="Deterministic symbol anchors from tool traffic")
    edited_file_paths: list[str] = Field(default_factory=list, description="Subset of file_paths a genuine edit-type tool call (Edit/Write/MultiEdit/apply_patch) touched this request, not merely read or mentioned (UPG-PROXY-INJECT-SINGLE-TURN) — drives event-anchored retirement of declared-anchor structural notes")
    session_id: str = Field(default="", description="Per-conversation id for dedup/cooldown")
    channel: str = Field(default="proxy", description="Delivery channel label (e.g. 'proxy'); used for per-channel policy + metrics")
    structural_only: bool = Field(default=False, description="Emit only exact structural matches (a static per-channel policy, not a content decision)")
    # UPG-PROXY-APPEND-BURNS-COOLDOWN. Both default to today's behaviour, so a
    # caller that predates deferred charging (an older `vectr proxy`) gets
    # retrieval-time charging exactly as before — no field, no change.
    defer_charge: bool = Field(default=False, description="Select WITHOUT charging the cooldown ledger; the caller confirms delivery later via confirm_token. Callers that cannot observe delivery (the proxy) set this")
    confirm_token: str = Field(default="", description="A delivery_token from an earlier deferred response whose block was confirmed delivered; charged before this request's own selection. Opaque, never parsed")


class ProactiveResponse(BaseModel):
    context: str = Field(default="", description="Packed context to inject ('' = inject nothing)")
    item_count: int = 0
    anchor_ids: list[str] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)
    processing_ms: int = 0
    # UPG-PROXY-APPEND-BURNS-COOLDOWN. `charge_deferred` is the FEATURE-DETECT
    # signal: a daemon that predates deferred charging simply has no such field
    # in its response model, so a new proxy reading it back as False knows the
    # daemon already charged at retrieval and that there is nothing to confirm.
    delivery_token: str = Field(default="", description="Opaque handle to confirm delivery of this result (empty when the ledger was charged at selection)")
    charge_deferred: bool = Field(default=False, description="True when this result's cooldown slots are NOT yet charged and await confirmation")


class ForgetRequest(BaseModel):
    note_id: int | None = Field(default=None, description="Delete this one note (the [#N] id from recall)")
    all: bool = Field(default=False, description="Delete ALL notes for this workspace. Irreversible.")


class PromoteRequest(BaseModel):
    """Explicit provenance promotion (TRIGGER-ENGINE wave 1,
    bm2-design-skeleton.md §5): auto -> agent -> human, one step at a time.
    Provenance is immutable at write; this is the one sanctioned way to raise
    it afterward (e.g. a human reviews and endorses an agent-authored note).
    Demotion is impossible."""

    note_id: int = Field(..., description="Note to promote")
    to: str = Field(..., description="Target provenance — must be exactly one step above the note's current provenance (auto -> agent -> human)")

    @field_validator("to")
    @classmethod
    def validate_to(cls, v: str) -> str:
        if v not in _PROVENANCE_VALUES:
            raise ValueError(f"to must be one of: {', '.join(_PROVENANCE_VALUES)}")
        return v


class PromoteResponse(BaseModel):
    note_id: int
    provenance: str
    processing_ms: int


class PinRequest(BaseModel):
    """Explicit Tier 0 membership toggle (UPG-RECALL-MISS-FLOOR part (b)):
    set or clear a note's `pinned` flag. Pinned notes are injected
    unconditionally alongside kind='directive' notes on every
    vectr_recall(query=...) call, regardless of query content, up to the
    configured Tier 0 bound (recall_floor.tier0_max_notes). Never inferred
    from content, usage, or query history — always an explicit caller
    decision."""

    note_id: int = Field(..., description="Note to pin/unpin")
    pinned: bool = Field(default=True, description="True to pin (join Tier 0), False to unpin")


class PinResponse(BaseModel):
    note_id: int
    pinned: bool
    message: str
    processing_ms: int


class AnchorRequest(BaseModel):
    """Attach anchor paths to an EXISTING note post-write
    (UPG-ANCHOR-ATTACH) — the first-class replacement for the
    re-store-with-supersedes workaround. Anchors are delivery-shaping
    staleness probes, never a claim about note correctness: check_staleness()
    compares their write-time hashes against the live files on every check.
    See WorkingContextStore.attach_anchors() for the pair format and
    idempotency contract."""

    note_id: int = Field(..., description="Note to attach anchors to")
    anchors: list[str] = Field(
        ...,
        min_length=1,
        description="Workspace-relative file paths to anchor this note to (at least one)",
    )
    session_id: str | None = Field(
        default=None,
        description=(
            "Calling session's identity — threads into the "
            "UPG-ANCHOR-UNOBSERVED-BINDING tri-state observation verdict "
            "recorded per attached path (True/False when this session has "
            "an observation ledger, None when none exists)"
        ),
    )


class AnchorResponse(BaseModel):
    note_id: int
    attached: list[str] = Field(default_factory=list, description="Paths actually added, in request order")
    already_present: list[str] = Field(default_factory=list, description="Requested paths the note was already anchored to (idempotent no-ops)")
    message: str
    processing_ms: int


class UnanchorRequest(BaseModel):
    """Remove anchor paths from an EXISTING note post-write
    (UPG-ANCHOR-DETACH) — the inverse of `AnchorRequest` and the
    single-call replacement for the re-store-with-supersedes workaround's
    mirror case. Anchors are delivery-shaping staleness probes, never a
    claim about note correctness: a removed anchor simply stops being a
    candidate in the next check_staleness() check. See
    `WorkingContextStore.detach_anchors()` for the idempotency contract
    and the path-equality rule (exact-string on element 0, matching
    `attach_anchors()` so a note is detachable by the same spelling that
    anchored it)."""

    note_id: int = Field(..., description="Note to remove anchors from")
    anchors: list[str] = Field(
        ...,
        min_length=1,
        description="Workspace-relative file paths to detach this note from (at least one)",
    )


class UnanchorResponse(BaseModel):
    note_id: int
    removed: list[str] = Field(default_factory=list, description="Paths actually removed, in request order")
    not_present: list[str] = Field(default_factory=list, description="Requested paths the note was not anchored to (idempotent no-ops)")
    message: str
    processing_ms: int


# 'system' is not settable via these REST requests — reserved for the one
# deterministic, non-judgment transition (stale_flagged, anchor drift);
# every REVOKE/REINSTATE call here is, definitionally, a judgment call, so
# only 'agent'/'human' are ever accepted (same "REST is a person-operated
# surface, so 'human' is reachable here but not from MCP" split as
# PromoteRequest above / the vectr_revoke MCP tool, which hardcodes
# actor='agent' and exposes no actor field at all).
_NOTE_EVENT_ACTOR_VALUES = ("agent", "human")


class RevokeRequest(BaseModel):
    """Explicit revocation (UPG-MEMORY-STATE-MACHINE §4.2) — 'this note was
    WRONG', appends a `revoked` event. The note stays a live recall/fire
    candidate; every surface that renders it afterward substitutes an
    anti-memory deterrent block for its raw content until reinstated."""

    note_id: int = Field(..., description="Note to revoke")
    reason: str = Field(..., min_length=1, description="Why this note is being revoked")
    actor: str = Field(default="agent", description="agent | human")

    @field_validator("actor")
    @classmethod
    def validate_actor(cls, v: str) -> str:
        if v not in _NOTE_EVENT_ACTOR_VALUES:
            raise ValueError(f"actor must be one of: {', '.join(_NOTE_EVENT_ACTOR_VALUES)}")
        return v


class RevokeResponse(BaseModel):
    note_id: int
    message: str
    processing_ms: int


class ReinstateRequest(BaseModel):
    """Revert a revocation (UPG-MEMORY-STATE-MACHINE §4.2) — appends a
    `reinstated` event. Always legal (revert-of-revert), one more event,
    never a rewrite of the revocation it reverses."""

    note_id: int = Field(..., description="Note to reinstate")
    reason: str | None = Field(default=None, description="Optional reason for the reinstatement")
    actor: str = Field(default="agent", description="agent | human")

    @field_validator("actor")
    @classmethod
    def validate_actor(cls, v: str) -> str:
        if v not in _NOTE_EVENT_ACTOR_VALUES:
            raise ValueError(f"actor must be one of: {', '.join(_NOTE_EVENT_ACTOR_VALUES)}")
        return v


class ReinstateResponse(BaseModel):
    note_id: int
    message: str
    processing_ms: int


class SupersedeRequest(BaseModel):
    """Post-hoc supersession (UPG-NOTE-RETIRE-POST-HOC) — retire an
    already-stored note as superseded, for a replacement that was written
    without `supersedes=`. Appends a `superseded_post_hoc` event and sets
    the same tombstone columns as the write-time path, so the note stops
    being a default recall/fire candidate and renders with the factual
    "[superseded ...]" badge — never the revoked deterrent. Reversible via
    /v1/reinstate."""

    note_id: int = Field(..., description="Note to retire as superseded")
    superseded_by: int | None = Field(
        default=None,
        description="Optional ID of the replacement note; must already exist in this workspace",
    )
    reason: str | None = Field(
        default=None, description="Optional: why this note is being retired as superseded"
    )
    actor: str = Field(default="agent", description="agent | human")

    @field_validator("actor")
    @classmethod
    def validate_actor(cls, v: str) -> str:
        if v not in _NOTE_EVENT_ACTOR_VALUES:
            raise ValueError(f"actor must be one of: {', '.join(_NOTE_EVENT_ACTOR_VALUES)}")
        return v


class SupersedeResponse(BaseModel):
    note_id: int
    message: str
    processing_ms: int


class CommitNoteRequest(BaseModel):
    """UPG-COMMIT-MEMORY-HOOK: the git post-commit hook's own write path —
    called only by `vectr hook post-commit` (main.cmd_hook), never by the
    editor's LLM. Every field is a raw git fact gathered client-side by that
    hook's own `git` subprocess calls; all interpretation (file-list
    capping, active-task lookup, note content/kind/provenance) happens
    server-side in `VectrService.record_commit_note`, the single source of
    truth for the resulting note's shape."""

    sha: str = Field(..., min_length=1, description="Short commit sha (git rev-parse --short HEAD)")
    subject: str = Field(default="", description="Commit subject line (git log -1 --format=%s)")
    branch: str = Field(default="", description="Current branch name (git rev-parse --abbrev-ref HEAD); empty on a detached HEAD")
    files: list[str] = Field(
        default_factory=list,
        description=(
            "Files touched by this commit (git diff-tree --no-commit-id "
            "--name-only -r <sha>), sent uncapped — the server applies the "
            "display cap (agent/config.yaml hooks.commit_note_max_files)."
        ),
    )


class CommitNoteResponse(BaseModel):
    note_id: int
    processing_ms: int


class SnapshotRequest(BaseModel):
    label: str = Field(..., min_length=1)
    session_id: str | None = Field(default=None)


class SnapshotResponse(BaseModel):
    snapshot_id: str
    label: str
    processing_ms: int


class ResumeNoteSummary(BaseModel):
    """One resume-surface note, index-tier only (UPG-RESUME-SURFACE) — same
    bounded fields (id, kind/priority, title, age, staleness) the CLI/MCP
    text render shows; expand the full body with
    vectr_recall(note_id=...)/`vectr recall --id <note_id>`."""
    note_id: int
    kind: str
    priority: str
    title: str
    created_at: float
    anchors: list[str] = []
    stale: bool = False


class ResumeSnapshotSummary(BaseModel):
    snapshot_id: str
    label: str
    created_at: float
    # None when the snapshot payload could not be decrypted/read (see
    # WorkingContextStore.restore_snapshot) — distinct from an empty snapshot.
    note_count: int | None = None


class ResumeResponse(BaseModel):
    """GET /v1/resume (UPG-RESUME-SURFACE): deterministic 'pick up where you
    left off' — the most recent high-priority task note (the SAME selection
    SessionStart boot injection uses), the latest saved snapshot, and open
    gotchas. Any of the three is None/empty when there is nothing to show —
    never an error."""
    last_task: ResumeNoteSummary | None = None
    snapshot: ResumeSnapshotSummary | None = None
    gotchas: list[ResumeNoteSummary] = []
    # True when more open gotchas exist beyond the configured cap
    # (behavior.resume.max_gotchas) — the formatted text discloses this too.
    gotchas_truncated: bool = False
    # Human/LLM-readable sectioned rendering of the fields above — the same
    # text the CLI prints and the MCP tool returns.
    formatted: str
    processing_ms: int


class TriggerResetRequest(BaseModel):
    """Reset one session's trigger-engine state (TRIGGER-ENGINE wave 2a,
    bm2-design-skeleton.md §3: "cleared on compaction"): the per-session fire
    ledger (so a previously-fired memory becomes re-eligible) and the
    cumulative injection token budget both zero out. Called by the
    PreCompact hook, right before the SessionStart hook that follows a
    /compact re-delivers the boot set. A session_id with no tracked ledger
    yet (or None) is a no-op, never an error — mirrors
    `VectrService.reset_trigger_ledger()`'s own contract."""

    session_id: str | None = Field(default=None)


class TriggerResetResponse(BaseModel):
    reset: bool
    processing_ms: int


# ---------------------------------------------------------------------------
# Episode capture
# ---------------------------------------------------------------------------

class EpisodeRequest(BaseModel):
    """PostToolUse hook's own write path — called only by `vectr hook
    post-tool-use` (main.cmd_hook / agent.hook_cli.run_hook)'s detached
    worker, never by the editor's LLM directly. Every field optional except
    `tool`: the hook forwards whatever the editor's PostToolUse payload
    actually contains, defensively (payload shape across editor versions is
    not fully known — see the G0 live-capture gate). ALL
    interpretation (normalization, outcome derivation, digest
    canonicalization) happens server-side in `VectrService.record_episode`,
    the single source of truth for the resulting row's shape — mirroring
    `CommitNoteRequest`'s "gather raw facts client-side, interpret
    server-side" split."""

    session_id: str | None = Field(default=None)
    ts: float | None = Field(default=None, description="Client-observed event time; server stamps its own if omitted")
    cwd: str = Field(default="")
    tool: str = Field(..., min_length=1, description="'bash' or 'edit'")
    command: str | None = Field(default=None, description="tool_input.command (Bash only)")
    description: str | None = Field(default=None, description="tool_input.description (Bash only)")
    file_path: str | None = Field(default=None, description="tool_input.file_path (Edit/Write/MultiEdit only) — path only, never content")
    rc: int | None = Field(default=None, description="Exit code, if the editor's tool_response ever includes one")
    is_error: bool = Field(default=False)
    interrupted: bool = Field(default=False)
    harness_success: bool | None = Field(
        default=None,
        description=(
            "Tri-state: True/False only when the hook layer is certain which "
            "discrete post-tool-call event the editor routed this call through "
            "(e.g. Claude Code's PostToolUse vs PostToolUseFailure); None when "
            "this integration doesn't expose that signal. See agent/outcome.py."
        ),
    )
    stdout_tail: str = Field(default="", description="Client-truncated stdout text")
    stderr_tail: str = Field(default="", description="Client-truncated stderr text")


class EpisodeResponse(BaseModel):
    episode_id: int
    processing_ms: int


class EpisodeRecord(BaseModel):
    """One row as returned by GET /v1/episodes — the only bulk reader of the
    `episodes` table besides the aggregate counts in `vectr_status`."""

    id: int
    session_id: str | None = None
    ts: float
    cwd: str
    tool: str
    cmd_raw: str
    verb: str
    flags: list[str]
    args: list[str]
    rc: int | None = None
    termination: str
    outcome: str
    stdout_digest: str
    stderr_digest: str
    markers_matched: list[str]
    env_delta_names: list[str]
    file_path: str | None = None
    arc_id: int | None = None


class EpisodesResponse(BaseModel):
    episodes: list[EpisodeRecord]
    processing_ms: int


# ---------------------------------------------------------------------------
# Arc distillation
# ---------------------------------------------------------------------------

class ArcFailureRecord(BaseModel):
    """One failed attempt in an arc's failure chain."""

    episode_id: int
    verb: str
    outcome: str
    markers_matched: list[str]


class ArcSuccessRecord(BaseModel):
    """The episode that resolved an arc's failure chain."""

    episode_id: int
    verb: str
    cmd_raw: str


class ArcRecord(BaseModel):
    """One row as returned by GET /v1/arcs / vectr_distill() — a discovered
    failure->success discovery moment, joined with its episodes' summary
    fields. Rendered facts only — no
    advice, no suggested kinds; that judgment lives in static tool-surface
    text, never in generated data."""

    id: int
    ts: float
    cwd: str
    confidence: str
    mutation_diff: dict
    failures: list[ArcFailureRecord]
    success: ArcSuccessRecord | None = None
    distilled_at: float | None = None
    distilled_note_id: int | None = None
    dismissed_reason: str | None = None


class ArcsResponse(BaseModel):
    arcs: list[ArcRecord]
    total_pending: int
    processing_ms: int


class ArcResolveResult(BaseModel):
    """Idempotent bulk-resolve outcome: ids that were pending and are
    now resolved vs. ids that were
    unknown or already resolved — never an error either way."""

    resolved: list[int]
    unresolved: list[int]


class BoundaryPrecompactResponse(BaseModel):
    """`GET /v1/boundary/precompact` — the rendered PreCompact
    boundary-preservation text (see `VectrService._boundary_precompact_text`)
    plus the pending-arc count it was rendered from. `text` is "" (never
    absent) when the surface is disabled by config
    (`episodes.boundary_precompact_enabled`) — the pre-compact hook branch
    treats an empty string as "nothing to print", not an error. Declared
    here (pydantic filters undeclared keys on a `response_model`) so both
    fields survive the REST boundary even if the service dict ever grows
    another key."""

    text: str
    arcs_pending: int


class ArcsDismissRequest(BaseModel):
    arc_ids: list[int] = Field(..., min_length=1, description="Arc ids to dismiss")
    reason: str = Field(..., min_length=1, description="Why these arcs are not worth distilling")


class ArcsDismissResponse(BaseModel):
    resolved: list[int]
    unresolved: list[int]
    processing_ms: int


# ---------------------------------------------------------------------------
# Symbol graph
# ---------------------------------------------------------------------------

class LocateRequest(BaseModel):
    name: str = Field(..., min_length=1)
    limit: int = Field(default=10, ge=1, le=50)


class SymbolResult(BaseModel):
    name: str
    kind: str
    file_path: str
    start_line: int
    end_line: int
    snippet: str = ""   # first ~12 lines of symbol body — AI reads this directly


class LocateResponse(BaseModel):
    results: list[SymbolResult]
    formatted: str
    processing_ms: int


class TraceRequest(BaseModel):
    name: str = Field(..., min_length=1)
    direction: str = Field(default="both")
    limit: int = Field(default=20, ge=1, le=100)
    include_builtins: bool = Field(default=False)

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, v: str) -> str:
        if v not in ("callers", "callees", "both"):
            raise ValueError("direction must be callers, callees, or both")
        return v


class TraceResponse(BaseModel):
    formatted: str
    processing_ms: int
