"""vectr bundled-config loader.

Reads ``agent/config.yaml`` at import time and caches the results.  All paths
are resolved via ``importlib.resources`` so this works both when running from
the repository root *and* when vectr is installed as the global binary
(``/opt/homebrew/bin/vectr`` or similar pip install), where a ``cwd``-relative
``open()`` would fail.

Public API
----------
QUALITY_TRIVIAL : float
    Quality prior for bare stub chunks (UPG-12.1).

QUALITY_NAVIGATIONAL : float
    Quality prior for re-export / import-only navigational chunks (UPG-12.1).

QUALITY_NAV_DECLARATION_RESCUE : float
    Softened quality prior for a bare-constructor-manifest navigational chunk
    when the query lexically names one of the identifiers it declares
    (UPG-NAV-OVERDEMOTE-DECL / F59). Applied instead of QUALITY_NAVIGATIONAL
    only for that lexically-gated case; every other query still gets the full
    navigational demotion.

QUALITY_HEADING_ONLY : float
    Quality prior for markdown heading-only chunks (UPG-12.1).

QUALITY_GENERATED : float
    Quality prior for machine-generated files (UPG-12.1).

QUALITY_VECTR_CONFIG : float
    Quality prior for vectr's own config files (UPG-12.1).

QUALITY_TEST_DEPRIORITISED : float
    Quality prior for test files (UPG-12.1).

TEST_FRAMEWORK_FAN_IN_THRESHOLD : int
    Minimum unambiguous corpus-wide caller-file count (symbol_graph.file_fan_in())
    above which a test-path-classified file is exempted from
    QUALITY_TEST_DEPRIORITISED (UPG-TESTPATH-FRAMEWORK-MISCLASS / F58) — it is a
    shipped testing-framework subpackage, not disposable test code.

QUALITY_DOC_PROSE : float
    Quality prior for documentation prose chunks (UPG-12.1).

QUALITY_SHORT_PENALTY : float
    Quality prior for chunks with very few meaningful lines (UPG-12.1).

QUALITY_PRIVATE_SYMBOL : float
    Quality prior for a private/internal symbol (single leading underscore,
    not a dunder) — a language-general naming convention, not corpus- or
    query-specific (UPG-16.1 / F30).

TRIVIAL_DOC_MAX_LINES : int
    Maximum non-blank lines for an HTML/markup or plain-text chunk to be
    classified as trivial by is_trivial_chunk() (UPG-15.5). 1–2-line test
    fixture templates and egg-info TXT files are trivial; multi-line .rst/.txt
    docs are not affected.

RERANK_TOP_K : int
    Number of hybrid candidates to rerank before trimming to n_results (UPG-12.1).

RERANK_TOP_K_UNFILTERED : int
    Deeper candidate pool for unfiltered (no language filter) queries (UPG-12.1).

RERANK_PRE_FILTER_FETCH_K : int
    Over-fetch depth for the pool-entry trivial filter (UPG-15.7). The hybrid
    retrieval fetches this many raw candidates, drops trivial (non-forced) chunks
    via is_trivial_chunk(), then trims to top_k_unfiltered before the cross-encoder
    runs. Ensures the rerank pool is filled with real code on fixture-heavy corpora.

RERANK_MAX_LENGTH : int
    Cross-encoder max sequence length in tokens (UPG-RERANK-LATENCY-BUDGET).
    Was hardcoded in agent/searcher.py; now a config sibling of model/top_k.

RERANK_BATCH_SIZE_BY_DEVICE : dict[str, int]
    predict() batch_size for the reranker cross-encoder, keyed by the
    resolved torch device type (CrossEncoder.device.type — the same
    attribute sentence_transformers itself populates when no device is
    given). See agent/config.yaml's ranking.rerank.batch_size comment for
    the measured per-device evidence (UPG-RERANK-LATENCY-BUDGET); batch_size
    only changes wall-clock latency, never rank order. The "default" key is
    the fallback for any device this hasn't been measured on.

INDEXING_MAX_CHUNK_LINES : int
    Hard cap on lines per chunk — prevents single huge chunks diluting embeddings (UPG-12.1).

INDEXING_MAX_CHUNK_CHARS : int
    Hard cap on characters per chunk (UPG-WINDOW-CHUNK-BYTE-CAP). A single long
    source line defeats every line-based cap; oversized chunks are sub-split
    into bounded pieces so the tail stays retrievable instead of being embedded
    (paid for in memory/latency) and then truncated at the embedder's token cap.

INDEXING_CLASS_HEADER_LINES : int
    Lines kept for class-level chunk (sig + docstring + attrs) (UPG-12.1).

OUTPUT_SNIPPET_LINES : int
    Lines returned as a snippet with each symbol location (UPG-12.1).

BEHAVIOR_REMEMBER_NUDGE_THRESHOLD : int
    Tool calls without vectr_remember before a nudge fires (UPG-12.1).

BEHAVIOR_REMEMBER_NUDGE_COOLDOWN : int
    Calls between repeated nudges after the threshold first fires (UPG-12.1).

EVICTION_RETRIEVED_TOKEN_GATE : int
    Minimum accumulated retrieved-token estimate since the last auto-eviction hint
    before auto_eviction_hint() will emit. Suppresses the hint on bursts of tiny
    searches that contribute negligible context pressure (UPG-11.15).

EVICTION_HINT_MAX_IDS : int
    Maximum chunk ids listed as re-fetch keys in eviction_hint()'s render
    (UPG-EVICT-SESSION-SCOPE).

EVICTION_MAX_TRACKED_SESSIONS : int
    Maximum concurrent per-session EvictionAdvisor instances tracked by a
    daemon before the oldest is dropped (LRU) (UPG-EVICT-SESSION-SCOPE).
    Also reused, same cap and same LRU shape, by WorkingContextStore's
    per-session anchor-observation ledger (UPG-ANCHOR-UNOBSERVED-BINDING)
    — one more per-session in-memory dict with the identical unbounded-
    session-growth risk, not a reason for a second tunable.

EVICTION_REMEMBER_ESCALATION_CHUNKS : int
    Chunks retrieved since the caller's last vectr_remember before
    auto_eviction_hint()'s escalated ACTION REQUIRED directive fires again
    (UPG-REMEMBER-BANNER-FATIGUE).

EVICTION_REMEMBER_ESCALATION_TOKENS : int
    Tokens retrieved since the caller's last vectr_remember required IN ADDITION
    to EVICTION_REMEMBER_ESCALATION_CHUNKS before the escalated directive fires
    again — a companion gate so one large single search cannot trip both the
    chunk-count and token gates in a single burst (UPG-EVICT-ESCALATION-GATE-TOO-LOW).

BOOT_MAX_DIRECTIVE_NOTES : int
    Maximum directive notes returned by boot_recall() (UPG-9.2). Directives
    are ordered oldest-first (standing rules stay in a stable order).

BOOT_MAX_TASK_NOTES : int
    Maximum high-priority task notes returned by boot_recall() (UPG-9.2),
    ordered newest-first (UPG-TASK-NOTE-INJECTION-RECENCY): task notes are
    current-work state, so the boot set surfaces only the freshest
    checkpoints rather than the full task history.

RESUME_MAX_GOTCHAS : int
    Maximum open (non-superseded) kind="gotcha" notes returned by
    resume_state() (UPG-RESUME-SURFACE), newest first. The task-note and
    snapshot sections of `resume` reuse BOOT_MAX_TASK_NOTES and the
    snapshots table's own "latest" selection respectively — this is the one
    new bound resume introduces.

INGEST_TRACES_MAX_UNRESOLVED_EXAMPLES : int
    Maximum unresolved caller/callee example strings surfaced in the
    vectr_ingest_traces response (UPG-7.3). Edges are ingested regardless;
    this only bounds the warning text length.

SYMBOL_NAME_PARAM_ALIASES : tuple[str, ...]
    Alternate argument keys accepted in place of "name" on vectr_locate and
    vectr_trace (F40-class ergonomics). A tool-description example that reads
    positional trains callers to guess a wrong key; the dispatch layer accepts
    these as drop-in aliases so the call succeeds instead of erroring.

MEMORY_HYGIENE_STALE_TASK_WARN_COUNT : int
    Minimum number of live kind="task" notes older than
    MEMORY_HYGIENE_STALE_TASK_WARN_AGE_DAYS before vectr_status appends a
    stale-task nudge (UPG-TASK-SUPERSEDES-HYGIENE). Additive/state-based only
    — never mutates or expires notes.

MEMORY_HYGIENE_STALE_TASK_WARN_AGE_DAYS : int
    Age in days after which a kind="task" note counts toward the stale count
    above (UPG-TASK-SUPERSEDES-HYGIENE).

MEMORY_EXPORT_DEBOUNCE_SECONDS : float
    Seconds of write-quiet, per workspace, before a scheduled `vectr memory
    export` re-render actually runs (UPG-MEMORY-LEGIBLE-FILE-PROJECTION part
    (a)). Coalesces a same-turn burst of writes into a single re-render.

LOCATE_LARGE_SPAN_THRESHOLD : int
    Line-span (end_line - start_line) at or above which a located symbol is
    considered "large" — typically a canonical library class or function (UPG-15.10).
    Symbols with span >= this value get the best (lowest) span bucket in locate
    ranking, so canonical 1000+ line base classes rank before tiny test stubs.

LOCATE_SMALL_SPAN_THRESHOLD : int
    Line-span below which a located symbol is considered "tiny" — a stub or inner
    test class (UPG-15.10). Symbols with span < this value get the worst (highest)
    span bucket in locate ranking, penalising 2–5-line test-inner stub classes.

IMPORTANCE_PRIOR_LAMBDA : float
    Blend weight for the file-level PageRank importance prior in the final search
    sort (ARCH-1b): final_score = base_rerank_score * quality_score * (1 + lambda *
    importance). 0 disables (pre-ARCH-1b behaviour). Relevance-gated by the multiply
    against base_rerank_score.

CLASS_IMPORTANCE_PRIOR_LAMBDA : float
    Blend weight for the class-level reference-frequency importance prior in the
    final search sort (ARCH-2), composed with IMPORTANCE_PRIOR_LAMBDA:
    final_score = base_rerank_score * quality_score * (1 + lambda_file * file_imp)
    * (1 + lambda_class * class_imp). 0 disables (pre-ARCH-2 behaviour). The lever
    that discriminates same-leaf method collisions file-level importance cannot.

PURPOSE_RANK_PRIOR_LAMBDA : float
    Blend weight for the ARCH-4 dual-vector purpose-similarity prior in the final
    search sort (ARCH-4b), composed with the two priors above: final_score =
    base_rerank_score * quality_score * (1 + lambda_file * file_imp) * (1 +
    lambda_class * class_imp) * (1 + lambda_purpose * purpose_sim). purpose_sim is
    the chunk's own body-vs-purpose cosine similarity (0 when no purpose vector
    exists for the chunk). 0 disables (pre-ARCH-4b behaviour). Carries the ARCH-4
    pool-entry signal into the final rank, which the body-only cross-encoder
    rerank cannot see.

TYPE_DEF_PRIOR_LAMBDA : float
    Blend weight for the type-definition node_type prior in the final search
    sort (UPG-RUST-DEF-EVICTION / DEF-B), composed with the three priors
    above: final_score = ... * (1 + lambda_purpose * purpose_sim *
    quality_score) * (1 + lambda_def * is_type_definition). is_type_definition
    is 1 when the chunk's own node_type is a struct/enum/trait/class/interface
    definition (chunk_quality.is_type_definition_chunk), else 0. 0 disables
    (pre-DEF-B behaviour). The lever that keeps a canonical type definition
    from losing to a same-name test/usage site at comparable rerank score.

DOCSTRING_DEDUP_LINES : int
    Leading docstring/comment lines compared when computing the near-duplicate
    docstring dedup key (UPG-RUST-DEF-EVICTION / DEF-C).

DOCSTRING_DEDUP_MIN_CHARS : int
    Minimum normalized leading-docstring length (chars) below which a chunk is
    never folded by the docstring dedup key — chunks with a trivial or absent
    leading header keep every occurrence (UPG-RUST-DEF-EVICTION / DEF-C).

DOCSTRING_DEDUP_BODY_SIMILARITY_MIN : float
    Minimum difflib.SequenceMatcher ratio between two chunks' full normalized
    bodies required before a matching docstring-dedup key is allowed to
    actually collapse them (LANE-P1-DEDUP false-collapse guard on DEF-C). A
    shared leading doc/comment alone is not sufficient — a trait/interface and
    its impl, or two different overloads, often carry the identical
    copy-pasted doc summary over genuinely different implementations.

DOCSTRING_DEDUP_MAX_REPS_COMPARED : int
    Max already-kept representatives compared per doc_key before a new
    candidate stops being compared and becomes its own representative
    (LANE-P1-DEDUP latency bound on the guard above). Bounds the collapse
    loop to O(n * k) instead of O(n^2) SequenceMatcher calls when many
    chunks share one boilerplate leading doc but are genuinely different.
    Fails in the safe direction only — a candidate past this bound is never
    falsely collapsed, only left as an extra separate result.

NEAR_DUP_BODY_ENABLED : bool
NEAR_DUP_BODY_SIMILARITY_MIN : float
    Minimum difflib.SequenceMatcher ratio between two chunks' full normalized
    bodies required to collapse a candidate into an already-kept
    representative under the LANE-NEARDUP-DEDUP content-only key. Higher than
    DOCSTRING_DEDUP_BODY_SIMILARITY_MIN (0.75) — this lane is for the
    truly-near-identical-content case (verbatim or nearly-verbatim
    restatement), not the boilerplate-header case DEF-C catches. Two
    genuinely different functions discussing the same subject sit well below
    this threshold; only near-verbatim copy-pasted bodies clear it.

NEAR_DUP_BODY_MAX_REPS_COMPARED : int
    Max already-kept representatives compared per new candidate under the
    LANE-NEARDUP-DEDUP content-only key before the candidate stops being
    compared and becomes its own representative. Same shape and
    safe-failure direction as DOCSTRING_DEDUP_MAX_REPS_COMPARED above;
    separate knob so the two mechanisms can be tuned independently.

INDEXING_FLOW_SCAN_HEAD_BYTES : int
    Bytes scanned from the start of a `.js` file when detecting Flow type syntax
    (UPG-JSFLOW-SYMBOLS). A header scan, not a full-file walk.

INDEXING_FLOW_PRAGMA : str
    Primary Flow-detection signal — the `@flow` pragma (UPG-JSFLOW-SYMBOLS).

INDEXING_FLOW_SECONDARY_MARKERS : tuple[str, ...]
    Secondary Flow-detection signals — Flow-only import syntax (UPG-JSFLOW-SYMBOLS).

INDEXING_VECTOR_STORE_DISPATCH_MAX_WORKERS : int
    Worker count of the dedicated executor a search/fetch request's
    vector-store call is dispatched to, off the event loop serving that
    request. Must stay 1 — see config.yaml comment.

INDEXING_VECTOR_STORE_SLOW_CALL_WARN_SECONDS : float
    Seconds a single vector-store call may run before a WARNING is logged
    naming the call and its elapsed time.

INDEX_GOVERNOR_ENABLED : bool
    Master switch for the bulk-indexing priority clamp + duty-cycle pacing
    (UPG-INDEX-RESOURCE-GOVERNOR). False (or --foreground-fast /
    VECTR_FOREGROUND_FAST=1 at the call site) restores unthrottled indexing.

INDEX_GOVERNOR_DUTY_CYCLE : float
    Fraction of wall-clock time the indexing thread spends working vs
    sleeping between embed batches (UPG-INDEX-RESOURCE-GOVERNOR). 1.0
    disables pacing.

INDEX_GOVERNOR_MACOS_QOS_CLASS : str
    macOS QoS class name the indexing thread is set to for each governed
    batch loop (UPG-INDEX-RESOURCE-GOVERNOR) — one of user_interactive,
    user_initiated, default, utility, background.

INDEX_GOVERNOR_LINUX_NICE_INCREMENT : int
    Linux per-thread niceness increment applied for each governed batch
    loop (UPG-INDEX-RESOURCE-GOVERNOR).

INDEX_GOVERNOR_MIN_BATCH_SECONDS_FOR_PACING : float
    A batch whose own work took less than this many seconds is never paced
    (UPG-INDEX-RESOURCE-GOVERNOR) — avoids a pacing sleep dwarfing a tiny
    batch's actual work.

INDEX_GOVERNOR_CHECKPOINT_EVERY_BATCHES : int
    How often (in embed batches) newly-completed files' mtimes are
    persisted to the incremental-index cache during Phase 3 / the
    purpose-vector pass (UPG-INDEX-RESOURCE-GOVERNOR) — always applies,
    independent of INDEX_GOVERNOR_ENABLED.

INDEX_GOVERNOR_DEFER_PURPOSE_PASS_ENABLED : bool
    Run the purpose-vector pass as a background phase after content
    embedding completes rather than blocking index_workspace()'s return on
    it (UPG-PURPOSE-PASS-DEFERRAL). False (or --foreground-fast /
    VECTR_FOREGROUND_FAST=1) restores the pre-deferral synchronous
    behaviour.

SYMBOL_GRAPH_RESERVED_KEYWORDS : dict[str, frozenset[str]]
    Per-language keyword sets that must never be minted as a symbol name or
    call-edge target — guards against a desynced/ERROR-node parse misattributing
    a keyword token as an identifier (UPG-JSFLOW-SYMBOLS).

SYMBOL_GRAPH_ERROR_RECOVERY_MIN_SPAN_LINES : int
    Minimum line-span an opaque errored (non-symbol-type) node must cover before
    an isolated reparse-recovery attempt is made on its byte range
    (UPG-REACT-TSX-FUNCTION-DECL-DROP).

SYMBOL_GRAPH_ERROR_RECOVERY_MAX_REPARSE_ATTEMPTS : int
    Reparse-recovery attempts budget per file, bounding worst-case parse cost
    (UPG-REACT-TSX-FUNCTION-DECL-DROP).

SYMBOL_GRAPH_ERROR_RECOVERY_MAX_EXTEND_STEPS_PER_ATTEMPT : int
    Per-attempt cap on sibling-absorption steps while growing a reparse past
    a mid-declaration cut, so one badly-cut region can't burn the whole
    per-file reparse budget (UPG-REACT-TSX-FUNCTION-DECL-DROP).

SYMBOL_GRAPH_TRACE_QUALIFIER_NEARMISS_MAX : int
    Maximum nearest-real-qualified-name suggestions surfaced by `trace()` when
    a "Class.method" qualifier fails to resolve to a real definition
    (UPG-TRACE-CALLERS-QUALIFIER-VALIDATION).

SYMBOL_GRAPH_QUALIFIED_NEARMISS_CANDIDATE_CAP : int
    Cap on exact-leaf-name candidates pulled before `nearest_symbol_names`
    ranks them by qualifier/enclosing-type similarity, when a "Class.method"
    qualifier fails to resolve under every `locate_l2` strategy
    (UPG-LOCATE-FALLBACK-NO-SIMILARITY).

WORKSPACE_DEFAULT_VECTRIGNORE_DIRS : tuple[str, ...]
    Directory names seeded into a fresh .vectrignore on first `vectr start`/`vectr
    init` when the workspace has none yet (UPG-13.2). Never overwrites an existing
    .vectrignore.

WATCHER_TOP_LEVEL_RESCAN_INTERVAL_S : float
    Seconds between CodeWatcher's shallow top-level-only rescans, used to pick up
    new top-level directories/files since the watcher never watches the workspace
    root itself (UPG-13.1/13.3).

WATCHER_BURST_FILES_THRESHOLD : int
    Distinct paths pending per-file debounce simultaneously above which
    CodeWatcher cancels every per-file timer and collapses into one deferred
    batch re-index (UPG-WATCHER-PRESSURE-GOVERNOR) — bounds embed-pipeline
    load under a sustained multi-file edit stream.

WATCHER_BURST_QUIET_SECONDS : float
    Seconds of repo-wide silence required, once burst coalescing has started,
    before the collapsed batch actually runs (UPG-WATCHER-PRESSURE-GOVERNOR).

WATCHER_MAX_RSS_MB : float
    Self-limit (MB) on this process's own peak resident set size above which
    a watcher-triggered batch re-index is deferred to the next quiet window
    instead of run (UPG-WATCHER-PRESSURE-GOVERNOR). 0 disables the check.

HOOKS_LOG_INJECTIONS : bool
    When true, every hook-driven recall that actually injects notes appends
    one line to ~/.vectr/logs/<workspace-hash>.hooks.log (UPG-HOOK-INJECT-
    OBSERVABILITY). Off by default — the per-hook-kind counters in `vectr
    status` cover the common case without writing to disk.

HOOKS_LOG_CHARS_PER_TOKEN : int
    Divisor for the approximate token count written to the optional hook
    injection log above (UPG-HOOK-INJECT-OBSERVABILITY).

HOOKS_MIN_SIMILARITY : float
    Per-turn UserPromptSubmit recall relevance floor (UPG-9.5). Raised from
    0.35 to 0.72 (measured 2026-07-15): at 0.35 the
    hook injected 3 irrelevant notes on 8/8 deliberately off-topic prompts;
    the M trigger primitive's own per-kind thetas (0.72-0.80) had 0/8 false
    fires on the identical prompts.

HOOKS_COMMIT_NOTE_MAX_FILES : int
    Display cap on the file list a commit-provenance note shows verbatim
    (UPG-COMMIT-MEMORY-HOOK) — files beyond this count collapse into "+N
    more".

HOOKS_COMMIT_NOTE_MAX_SUBJECT_CHARS : int
    Hard cap on a commit-provenance note's subject-line length
    (UPG-COMMIT-MEMORY-HOOK).

HOOKS_POST_COMMIT_TIMEOUT_S : float
    Hard timeout (seconds) the post-commit git hook's own daemon HTTP call
    (POST /v1/commit-note) is held to (UPG-COMMIT-MEMORY-HOOK) — defense in
    depth behind the hook's own shell-level backgrounding, which is what
    actually keeps `git commit` from ever waiting on it.

EPISODES_MAX_ROWS : int
    Per-workspace ring-buffer size for the `episodes` table (episode
    capture) — rows beyond this count (newest kept) are pruned on every
    write.

EPISODES_TTL_DAYS : float
    TTL (days) beyond which an episode row is pruned on the next write,
    independent of the ring buffer above.

EPISODES_CLIENT_TRUNCATE_CHARS : int
    Hard cap (characters) on the raw stdout/stderr text the PostToolUse
    hook's detached worker forwards to the daemon, applied client-side
    before the HTTP POST.

EPISODES_DIGEST_MAX_CHARS : int
    Server-side digest cap (characters) for stdout_digest/stderr_digest,
    applied at insert time (agent/episode_canon.py) after the client-side
    truncation above.

EPISODES_DIGEST_HEAD_LINES : int
EPISODES_DIGEST_TAIL_LINES : int
    Lines kept from the start/end of a digest once it must be elided
    (agent/episode_canon.py).

EPISODES_POST_TIMEOUT_S : float
    Hard timeout (seconds) the PostToolUse hook's detached worker holds its
    own POST /v1/episode call to, client-side.

EPISODES_STALE_TEMP_FILE_SWEEP_AGE_S : float
    Age (seconds) beyond which an orphaned `vectr-episode-*.json` payload
    temp file is swept by agent/episode_worker.py — a
    `_spawn_episode_worker` spawn failure can leave one behind with no
    process left to clean it up.

EPISODES_DISTILL_MAX_ARCS_RENDERED : int
    Max arcs rendered per `vectr_distill()` no-args call,
    confidence-first then oldest-first — the rest stay pending for a
    later call.

EPISODES_DISTILL_RENDER_TOKEN_CAP : int
    Hard cap (estimated tokens, agent/trigger_engine.py's token_estimate)
    on the same `vectr_distill()` render, alongside
    EPISODES_DISTILL_MAX_ARCS_RENDERED above — whichever limit is hit
    first stops the render.

BOUNDARY_PRECOMPACT_ENABLED : bool
    Master switch for the PreCompact boundary-preservation surface
    (GET /v1/boundary/precompact). False makes the route return an empty
    text with HTTP 200 rather than the rendered nudge.

BOUNDARY_PRECOMPACT_TOKEN_CAP : int
    Hard cap (estimated tokens, agent/trigger_engine.py's token_estimate)
    on the rendered boundary-preservation text. The arc-independent base
    text always fits; a pending-arc count sentence is appended only when
    it still fits under this cap.

STRATEGY_DEFAULT_SEMANTIC_WEIGHT : float
STRATEGY_DEFAULT_BM25_WEIGHT : float
    Fallback hybrid-search weights used before the first index-time codebase
    fingerprint has run (UPG-8.2). Keeps `search` and `status` deterministic
    from the first call instead of the weight fields being silently absent.
DUAL_VECTOR_ENABLED : bool
    Master switch for the ARCH-4 per-symbol purpose-vector pool-entry mechanism.
    True stores + queries a second body-stripped "purpose" embedding (qualified
    signature + docstring) per symbol chunk. False reduces to pre-ARCH-4
    body-only behaviour.

DUAL_VECTOR_BLEND_MODE : str
    How a chunk's body and purpose similarity scores combine into one dense
    score: "max" (default, non-averaging pool-entry rescue) or "weighted".

DUAL_VECTOR_BLEND_WEIGHT : float
    Purpose-vector weight when DUAL_VECTOR_BLEND_MODE == "weighted".

DUAL_VECTOR_MAX_SIGNATURE_LINES : int
    Maximum declaration lines captured when distilling a chunk's purpose text.

DUAL_VECTOR_MAX_DOCSTRING_LINES : int
    Maximum docstring/leading-comment lines captured when distilling a chunk's
    purpose text.

DUAL_VECTOR_MAX_DOCSTRING_CHARS : int
    Maximum characters kept from the captured docstring/leading-comment text.

NOTFOUND_FLOOR_ENABLED : bool
    Master switch for the UPG-NOTFOUND-FLOOR low-confidence signal (F46).
    False is an exact no-op: low_confidence is always False, restoring
    pre-UPG-NOTFOUND-FLOOR behaviour.

NOTFOUND_FLOOR_MIN_TOKEN_LEN : int
    Minimum length (chars) for a query token to be considered a content word
    in the zero-document-frequency check (UPG-NOTFOUND-FLOOR-2).

NOTFOUND_FLOOR_STOPWORDS : frozenset[str]
    Generic English/query-scaffolding words excluded from the zero-document-
    frequency check.

NOTFOUND_FLOOR_MIN_ZERO_DF_TOKENS : int
    Minimum number of query content tokens with zero corpus-wide document
    frequency (i.e. the token never appears in ANY indexed chunk, not just
    the query's fetched candidate pool) required to flag a search's result
    set low_confidence. Replaces the absolute-cosine floor (dense_score_floor)
    of the first UPG-NOTFOUND-FLOOR iteration, which measurement showed
    cannot separate absent-topic from on-topic queries against the
    production embedder — see config.yaml for the evidence.

NOTFOUND_FLOOR_MIN_TOP_RELEVANCE : float
    Absolute floor (UPG-SCORE-DISPLAY-FLAT) on the top result's cross-encoder
    relevance score (ce_relevance): below this, low_confidence is flagged
    even if the zero-document-frequency check above doesn't trip. Never
    applied to the raw dense-cosine fallback score — see config.yaml for why.

NOTFOUND_FLOOR_CE_OVERRIDE_MIN_RELEVANCE : float
    UPG-NOTFOUND-FLOOR-CE-OVERRIDE. High-confidence override on the
    zero-document-frequency trigger: when the top result's cross-encoder
    relevance (ce_relevance) is >= this value, the zero-DF trigger is
    suppressed (an everyday paraphrase of a symbol the corpus contains can
    lack a corpus vocabulary word yet still match decisively — 0.988
    observed). A threshold comparison on an already-computed calibrated
    score, never the raw dense-cosine fallback. The low_top_relevance
    sub-signal is unaffected. Set above 1.0 to disable (no ce_relevance can
    meet it) — an exact no-op restoring the pre-override OR behaviour.

NOTFOUND_FLOOR_BANNER : str
    Low-confidence banner text prepended to the MCP vectr_search response
    when the floor fires.

NOTFOUND_FLOOR_BANNER_CLI : str
    Low-confidence banner text printed by `vectr search` (CLI surface) when
    the floor fires (UPG-CLI-SEARCH-FLOOR). Separate from the MCP banner
    above because that text names `vectr_locate`, meaningless at a shell
    prompt — no `vectr locate` subcommand exists.

RELATIVE_CLIFF_ENABLED : bool
    UPG-BANNER-CALIBRATION Phase 3: master switch for the rank-1-to-rank-2
    ce_relevance cliff sub-signal (ranking.relative_cliff.enabled in
    config.yaml). False is an exact no-op: the cliff sub-signal never
    enters the low_confidence OR, restoring pre-this-feature behaviour.
    The Phase 1 measurement could not validate either a recalibrated
    absolute floor or a rank-1-vs-rank-2 cliff on the two corpora it
    reached (django + tigerbeetle); the default stays OFF until Phase 3
    measurement on the missing three corpora (react, cpython, uv)
    shows a separation. A change to the default is a product decision
    (Rail 7) and not made in this lane.

RELATIVE_CLIFF_MIN_TOP : float
    UPG-BANNER-CALIBRATION Phase 3: absolute ce_relevance floor below
    which a rank-1 cannot credibly lead a set, in the same scale and
    gate (ce_relevance-only, never the dense-cosine fallback) as
    NOTFOUND_FLOOR_MIN_TOP_RELEVANCE above. The cliff sub-signal only
    fires when rank-1 is at least this high; a weak top result with a
    wide-looking drop to rank-2 is not a confident result regardless
    of the drop. No-op whenever RELATIVE_CLIFF_ENABLED is False.

RELATIVE_CLIFF_MIN_DROP : float
    UPG-BANNER-CALIBRATION Phase 3: the required rank-1-to-rank-2 gap
    expressed as a share of remaining headroom above rank-1, i.e.
        (score[0] - score[1]) >= min_drop * (1 - score[0])
    not a fixed score difference — see the score_order_explain entry
    for why a fixed margin on bounded scores is unsatisfiable on a
    confident query. The exact value is the Phase 3 question the
    distribution study answers; the shipped default 0.50 is a
    placeholder the enabled=false default never exercises. No-op
    whenever RELATIVE_CLIFF_ENABLED is False.

EMBEDDING_DEFAULT_MODEL : str
    Default local (sentence-transformers) embedding model for the L3 content
    index (UPG-EMBEDDER-SWAP-GRANITE). A workspace's ChromaDB collection is
    stamped with whichever model built it (CodeIndexer's embed-model stamp);
    changing this value forces a full vector index rebuild on next index
    rather than mixing vectors from two models in one collection.

EMBEDDING_THREAD_CAP : int
    Resolved CPU thread cap applied to the local (torch-backed) embedding
    model at construction (UPG-EMBED-THREAD-CONTENTION) — `embedding.
    thread_cap` verbatim if positive, else `embedding.thread_cap_auto_fraction`
    of `os.cpu_count()` (minimum 1) when that key is 0. Governs the
    process-wide torch thread pool, so it also caps the cross-encoder
    reranker.

FETCH_MAX_IDS_PER_CALL : int
    Maximum chunk ids accepted per vectr_fetch / POST /v1/fetch / `vectr
    fetch` call (UPG-CTX-EVICT). Bounds the deterministic re-fetch-by-id
    surface so it can't be used as an unbounded bulk export of the index.

FETCH_MISALIGN_NEAREST_MAX : int
    Maximum nearest stored chunk ids reported per requested id whose line
    range merely misaligns with chunk boundaries in an otherwise-indexed
    file (UPG-FETCH-ID-MISALIGN-MSG) — distinguishes this case from a
    genuinely absent/changed file so vectr_fetch's own "not found" response
    tells the caller what to ask for next instead of implying the index is
    stale.

ARC_DETECTION_ENABLED : bool
    Master switch for arc detection: whether `VectrService.record_episode`
    feeds persisted bash/edit episodes into `ArcDetector.observe()` at all.
    False disables the write-path call only — the detector's own config
    below stays loaded either way.

ARC_NORM_UUID_REGEX, ARC_NORM_VERSION_REGEX, ARC_NORM_NUM_REGEX,
ARC_NORM_PATH_EXTENSION_REGEX : str
    Positional-argument abstraction-class regexes used by app/cmdnorm.py
    for command normalization. Comparison-only classification of argv
    structure — concrete values are always preserved alongside.

ARC_NORM_ENV_ASSIGNMENT_REGEX : str
    Matches a leading `NAME=value` env-var-assignment prefix token on a
    Bash command — its name is captured, the token is stripped.

ARC_NORM_STDERR_MERGE_TOKEN : str
    Trailing redirect token stripped from the first pipeline stage
    (`cmd 2>&1 | tail -30` -> `cmd`) — semantics-neutral decoration.

ARC_NORM_MAX_VERB_TOKENS : int
    Maximum leading tokens folded into a normalized command's `verb`
    (binary + immediate subcommand chain, e.g. `npm run build`) — bounds
    runaway absorption of positional arguments into the verb for commands
    with no flags (e.g. `cp src dest`).

ARC_NORM_WRAPPER_PREFIXES : dict[str, str]
    Transparent wrapper-prefix tokens (`timeout`, `env`, `nice`, `nohup`,
    `stdbuf`) stripped iteratively before verb extraction so the wrapped
    command — not the wrapper — becomes the verb. Value names the
    wrapper's own token-consumption shape (`bare`, `fixed_arg`,
    `nice_niceness`, `dash_flags`, `env_assignments`); see app/cmdnorm.py.
    `xargs` is deliberately never in this map — its argument is a command
    template, not the command that ran.

ARC_NORM_NICE_NICENESS_FLAG : str
    The niceness-value flag (`-n`) recognized by the `nice_niceness`
    wrapper-consumption rule above.

ARC_NORM_PIPELINE_DISPLAY_ONLY_VERBS : frozenset[str]
    Verbs (`cat`, `tail`, `head`) whose pipeline stage is dropped only when
    it is part of a TRAILING run of such stages — a non-trailing
    multi-stage pipeline always keeps every stage's tokens in the
    comparison set.

ARC_SIMILARITY_VERB_WEIGHT, ARC_SIMILARITY_FLAG_WEIGHT,
ARC_SIMILARITY_ARG_WEIGHT : float
    Composite mutation-similarity weights: score = verb_weight*verb +
    flag_weight*jaccard(flags) + arg_weight*jaccard(args). Sum to 1.0.

ARC_SIMILARITY_VERB_SOFT_MATCH_MIN_RATIO : float
    Levenshtein ratio above which two non-identical verbs count as a soft
    match (typo-fix verbs) rather than unrelated.

ARC_SIMILARITY_VERB_SOFT_MATCH_SCORE : float
    Fixed verb-component score assigned to a soft verb match — not the
    raw Levenshtein ratio.

ARC_MUTATION_BAND_MIN, ARC_MUTATION_BAND_MAX : float
    Composite-similarity band a candidate failure->success pair must fall
    in to count as a mutation arc. A score of 1.0 (identical normalized
    command) is never a mutation — it routes to the edit-mediated/flaky
    check instead.

ARC_WINDOW_MAX_COMMANDS : int
    Sliding-window bound: a pending failure or edit record ages out once
    more than this many Bash/Edit episodes have since been observed in
    the session.

ARC_WINDOW_TTL_SECONDS : float
    Sliding-window bound: a pending failure or edit record ages out once
    this many seconds have elapsed (by episode timestamp, never wall
    clock) since it was observed.

ARC_WINDOW_MAX_PENDING_PER_VERB_FAMILY : int
    Cap on concurrently pending failures tracked per (session, verb-family,
    cwd) bucket — the oldest is evicted once a new failure exceeds it.

ARC_TS_MONOTONIC_FALLBACK_SECONDS : float
    Step added to a session's last-seen `ts` to synthesize a
    strictly-increasing timestamp when an episode's own `ts` is
    missing/None/unparseable (`observe()` must never raise on this).

ARC_FLAKE_SUPPRESS_MIN_COUNT : int
    Number of proven flaky-retry flips (identical command, no intervening
    edit, no env delta — cwd is a bucket key) for the same normalized
    command within a session before near-threshold mutation-band matches
    for that command are suppressed too (Travis-CI base-rate defense).

ARC_FLAKE_NEAR_THRESHOLD_MIN : float
    Lower bound of the "near-identical" similarity sub-band subject to the
    flaky-suppression rule above — a genuinely different mutation at a
    lower score is never suppressed by this rule.

ARC_TRANSIENT_MARKER_IDS : frozenset[str]
    Marker ids (tool-output classification — see agent/markers.yaml) that
    mark a captured arc `low_confidence`: the failure's stderr matched a
    transient-error signature, so the "fix" may just be an environment
    retry rather than a real one.

CACHE_DIR_ENV : str
    Name of the environment variable (``VECTR_CACHE_DIR``) that overrides
    the cache root every on-disk cache path resolves under (UPG-TEST-CACHE-
    ISOLATION). Defined once here so every callsite imports the same name
    instead of re-typing the literal string.

vectr_cache_root() -> Path
    The single resolution point for vectr's cache root — every module that
    used to compute ``Path.home() / ".cache" / "vectr"`` on its own calls
    this instead. Reads ``VECTR_CACHE_DIR`` at call time (not import time),
    so a caller (e.g. the test suite, via a session-scoped fixture) can set
    it before constructing any service and every subsequent cache path
    (workspace DB dirs, the Hugging Face model cache, cache-maintenance
    sweeps) is redirected there.
"""
from __future__ import annotations

import importlib.resources as _ilr
import os as _os
import re as _re
from pathlib import Path
from typing import Any

import yaml as _yaml


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_bundled_text(resource_path: str) -> str:
    """Read a text file bundled inside the ``agent`` package.

    ``resource_path`` is relative to the ``agent`` package root, using forward
    slashes (e.g. ``"config.yaml"``).  We use
    ``importlib.resources.files()`` (Python 3.9+) so the file is found whether
    the package is on-disk or inside a zip/wheel.
    """
    parts = resource_path.split("/")
    # Start from the agent package anchor.
    pkg = _ilr.files(__name__.rsplit(".", 1)[0] if "." in __name__ else __name__)
    resource = pkg
    for part in parts:
        resource = resource.joinpath(part)  # type: ignore[arg-type]
    return resource.read_text(encoding="utf-8")


class _StrictBoolLoader(_yaml.SafeLoader):
    """SafeLoader that does NOT apply YAML-1.1's implicit bool resolver to the
    bare tokens ``on/off/yes/no`` (UPG-YAML-BOOL).

    PyYAML's default resolver parses ``on/off/yes/no`` (any case) as booleans,
    so a string list containing one of them silently gets a ``True``/``False``
    element instead of the string — a stopword list with a bare ``on`` entry
    became ``[..., True, ...]`` until quoted. Only ``true``/``false`` remain
    genuine booleans (the single style config.yaml uses for real bools), so a
    string-typed list is never corrupted by an unquoted English word.
    """


# Strip the inherited YAML-1.1 bool resolver, then re-register one that matches
# only true/false. Build fresh per-first-character lists so the parent class's
# resolver table is untouched.
_StrictBoolLoader.yaml_implicit_resolvers = {
    ch: [(tag, regexp) for tag, regexp in resolvers
         if tag != "tag:yaml.org,2002:bool"]
    for ch, resolvers in _yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_StrictBoolLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    _re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
    list("tTfF"),
)


def _load_config() -> dict[str, Any]:
    """Load and return the parsed agent/config.yaml as a dict."""
    raw = _read_bundled_text("config.yaml")
    return _yaml.load(raw, Loader=_StrictBoolLoader)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Eagerly load once at import; cache in module-level constants.
# ---------------------------------------------------------------------------

_cfg = _load_config()

# ---------------------------------------------------------------------------
# Quality priors — ranking multipliers (UPG-12.1)
# ---------------------------------------------------------------------------

_qp_cfg: dict[str, Any] = _cfg["ranking"]["quality_priors"]

QUALITY_TRIVIAL: float = float(_qp_cfg["trivial"])
QUALITY_NAVIGATIONAL: float = float(_qp_cfg["navigational"])
QUALITY_NAV_DECLARATION_RESCUE: float = float(_qp_cfg["navigational_declaration_rescue"])
QUALITY_HEADING_ONLY: float = float(_qp_cfg["heading_only"])
QUALITY_GENERATED: float = float(_qp_cfg["generated"])
QUALITY_VECTR_CONFIG: float = float(_qp_cfg["vectr_config"])
QUALITY_TEST_DEPRIORITISED: float = float(_qp_cfg["test_deprioritised"])
TEST_FRAMEWORK_FAN_IN_THRESHOLD: int = int(_qp_cfg["test_framework_fan_in_threshold"])
QUALITY_DOC_PROSE: float = float(_qp_cfg["doc_prose"])
QUALITY_SHORT_PENALTY: float = float(_qp_cfg["short_penalty"])
QUALITY_PRIVATE_SYMBOL: float = float(_qp_cfg["private_symbol_deprioritised"])
TRIVIAL_DOC_MAX_LINES: int = int(_qp_cfg["trivial_doc_max_lines"])
TRIVIAL_ATTR_CLASS_MAX_ATTRS: int = int(_qp_cfg["trivial_attr_class_max_attrs"])

# ---------------------------------------------------------------------------
# Locate ranking tunables (UPG-15.10)
# ---------------------------------------------------------------------------

_lr_cfg: dict[str, Any] = _cfg["ranking"]["locate_ranking"]

LOCATE_LARGE_SPAN_THRESHOLD: int = int(_lr_cfg["large_span_threshold"])
LOCATE_SMALL_SPAN_THRESHOLD: int = int(_lr_cfg["small_span_threshold"])

# ---------------------------------------------------------------------------
# Importance prior — file-level PageRank blend (ARCH-1b)
# ---------------------------------------------------------------------------

_imp_cfg: dict[str, Any] = _cfg["ranking"]["importance_prior"]

IMPORTANCE_PRIOR_LAMBDA: float = float(_imp_cfg["lambda"])

# ---------------------------------------------------------------------------
# Class importance prior — class-level reference-frequency blend (ARCH-2)
# ---------------------------------------------------------------------------

_cimp_cfg: dict[str, Any] = _cfg["ranking"]["class_importance"]

CLASS_IMPORTANCE_PRIOR_LAMBDA: float = float(_cimp_cfg["lambda"])

# ---------------------------------------------------------------------------
# Purpose-rank prior — dual-vector purpose-similarity blend into final sort (ARCH-4b)
# ---------------------------------------------------------------------------

_prp_cfg: dict[str, Any] = _cfg["ranking"]["purpose_rank"]

PURPOSE_RANK_PRIOR_LAMBDA: float = float(_prp_cfg["lambda"])

# ---------------------------------------------------------------------------
# Type-definition node_type prior (UPG-RUST-DEF-EVICTION / DEF-B)
# ---------------------------------------------------------------------------

_tdp_cfg: dict[str, Any] = _cfg["ranking"]["type_def_prior"]

TYPE_DEF_PRIOR_LAMBDA: float = float(_tdp_cfg["lambda"])

# ---------------------------------------------------------------------------
# Docstring near-duplicate dedup (UPG-RUST-DEF-EVICTION / DEF-C)
# ---------------------------------------------------------------------------

_ddd_cfg: dict[str, Any] = _cfg["ranking"]["docstring_dedup"]

DOCSTRING_DEDUP_LINES: int = int(_ddd_cfg["lines"])
DOCSTRING_DEDUP_MIN_CHARS: int = int(_ddd_cfg["min_chars"])
DOCSTRING_DEDUP_BODY_SIMILARITY_MIN: float = float(_ddd_cfg["body_similarity_min_ratio"])
DOCSTRING_DEDUP_MAX_REPS_COMPARED: int = int(_ddd_cfg["max_reps_compared"])

# ---------------------------------------------------------------------------
# Content-based near-duplicate dedup (LANE-NEARDUP-DEDUP)
# ---------------------------------------------------------------------------

_ndb_cfg: dict[str, Any] = _cfg["ranking"]["near_dup_body"]

# Minimum difflib.SequenceMatcher ratio between two chunks' full normalized
# bodies required to collapse a candidate into an already-kept representative
# under the near-dup-body key. Much higher than DOCSTRING_DEDUP_BODY_SIMILARITY_MIN
# (0.75) — this lane is for the truly-near-identical-content case, not the
# boilerplate-header case. Read via
# config.NEAR_DUP_BODY_SIMILARITY_MIN.
NEAR_DUP_BODY_ENABLED: bool = bool(_ndb_cfg.get("enabled", False))
NEAR_DUP_BODY_SIMILARITY_MIN: float = float(_ndb_cfg["body_similarity_min_ratio"])
# Max already-kept representatives compared per new candidate before the
# candidate stops being compared and becomes its own representative — same
# latency bound as DOCSTRING_DEDUP_MAX_REPS_COMPARED, separate config knob so
# the two mechanisms can be tuned independently. Fails in the safe direction
# only (a candidate past the cap is never falsely collapsed, only left as an
# extra separate result). Read via config.NEAR_DUP_BODY_MAX_REPS_COMPARED.
NEAR_DUP_BODY_MAX_REPS_COMPARED: int = int(_ndb_cfg["max_reps_compared"])

# ---------------------------------------------------------------------------
# Rerank pool sizes (UPG-12.1) + reranker model
# ---------------------------------------------------------------------------

_rr_cfg: dict[str, Any] = _cfg["ranking"]["rerank"]

# Cross-encoder reranker (HuggingFace id). VECTR_RERANKER_MODEL overrides at
# the searcher; empty disables reranking. Query-time only — no reindex on swap.
RERANK_MODEL: str = str(_rr_cfg["model"])

RERANK_TOP_K: int = int(_rr_cfg["top_k"])
RERANK_TOP_K_UNFILTERED: int = int(_rr_cfg["top_k_unfiltered"])
RERANK_PRE_FILTER_FETCH_K: int = int(_rr_cfg["pre_filter_fetch_k"])
RERANK_MAX_LENGTH: int = int(_rr_cfg["max_length"])
# Per-device predict() batch_size (UPG-RERANK-LATENCY-BUDGET). Every entry is
# read directly (no .get() default) so a missing device key fails loudly at
# import time; the "default" key itself is the intentional runtime fallback
# for a resolved torch device the config doesn't name (see agent/searcher.py).
RERANK_BATCH_SIZE_BY_DEVICE: dict[str, int] = {
    str(_device): int(_bs) for _device, _bs in _rr_cfg["batch_size"].items()
}
if "default" not in RERANK_BATCH_SIZE_BY_DEVICE:
    raise KeyError("ranking.rerank.batch_size must define a 'default' entry")

# ---------------------------------------------------------------------------
# Indexing tunables (UPG-12.1)
# ---------------------------------------------------------------------------

_idx_cfg: dict[str, Any] = _cfg["indexing"]

INDEXING_MAX_CHUNK_LINES: int = int(_idx_cfg["max_chunk_lines"])
INDEXING_MAX_CHUNK_CHARS: int = int(_idx_cfg["max_chunk_chars"])
INDEXING_CLASS_HEADER_LINES: int = int(_idx_cfg["class_header_lines"])
INDEXING_BUILD_ARTIFACT_DIR_SUFFIXES: tuple[str, ...] = tuple(
    str(s).lower() for s in _idx_cfg["build_artifact_dir_suffixes"]
)

# UPG-JSFLOW-SYMBOLS: Flow-typed .js detection (routes to the typescript/tsx grammar).
_flow_cfg: dict[str, Any] = _idx_cfg["flow_detection"]

INDEXING_FLOW_SCAN_HEAD_BYTES: int = int(_flow_cfg["scan_head_bytes"])
INDEXING_FLOW_PRAGMA: str = str(_flow_cfg["pragma"])
INDEXING_FLOW_SECONDARY_MARKERS: tuple[str, ...] = tuple(
    str(m) for m in _flow_cfg["secondary_markers"]
)

# Off-event-loop dispatch for request-handling vector-store calls.
_vsb_cfg: dict[str, Any] = _idx_cfg["vector_store_bridge"]

INDEXING_VECTOR_STORE_DISPATCH_MAX_WORKERS: int = int(_vsb_cfg["dispatch_max_workers"])
INDEXING_VECTOR_STORE_SLOW_CALL_WARN_SECONDS: float = float(_vsb_cfg["slow_call_warn_seconds"])

# Bulk-indexing resource governor: priority clamp + duty-cycle pacing +
# checkpoint cadence (UPG-INDEX-RESOURCE-GOVERNOR).
_ig_cfg: dict[str, Any] = _idx_cfg["index_governor"]

INDEX_GOVERNOR_ENABLED: bool = bool(_ig_cfg["enabled"])
INDEX_GOVERNOR_DUTY_CYCLE: float = float(_ig_cfg["duty_cycle"])
INDEX_GOVERNOR_MACOS_QOS_CLASS: str = str(_ig_cfg["macos_qos_class"])
INDEX_GOVERNOR_LINUX_NICE_INCREMENT: int = int(_ig_cfg["linux_nice_increment"])
INDEX_GOVERNOR_MIN_BATCH_SECONDS_FOR_PACING: float = float(_ig_cfg["min_batch_seconds_for_pacing"])
INDEX_GOVERNOR_CHECKPOINT_EVERY_BATCHES: int = int(_ig_cfg["checkpoint_every_batches"])
INDEX_GOVERNOR_DEFER_PURPOSE_PASS_ENABLED: bool = bool(_ig_cfg["defer_purpose_pass"])

# ---------------------------------------------------------------------------
# Output tunables (UPG-12.1)
# ---------------------------------------------------------------------------

_out_cfg: dict[str, Any] = _cfg["output"]

OUTPUT_SNIPPET_LINES: int = int(_out_cfg["snippet_lines"])

# ---------------------------------------------------------------------------
# Behaviour tunables (UPG-12.1)
# ---------------------------------------------------------------------------

_beh_cfg: dict[str, Any] = _cfg["behavior"]["remember_nudge"]

BEHAVIOR_REMEMBER_NUDGE_THRESHOLD: int = int(_beh_cfg["threshold"])
BEHAVIOR_REMEMBER_NUDGE_COOLDOWN: int = int(_beh_cfg["cooldown"])

# ---------------------------------------------------------------------------
# Eviction auto-hint token gate (UPG-11.15)
# ---------------------------------------------------------------------------

_evict_cfg: dict[str, Any] = _cfg["behavior"]["eviction"]

EVICTION_RETRIEVED_TOKEN_GATE: int = int(_evict_cfg["retrieved_token_gate"])
EVICTION_HINT_MAX_IDS: int = int(_evict_cfg["hint_max_ids"])
EVICTION_MAX_TRACKED_SESSIONS: int = int(_evict_cfg["max_tracked_sessions"])
EVICTION_REMEMBER_ESCALATION_CHUNKS: int = int(_evict_cfg["remember_escalation_chunks"])
EVICTION_REMEMBER_ESCALATION_TOKENS: int = int(_evict_cfg["remember_escalation_tokens"])

# ---------------------------------------------------------------------------
# Boot recall bounds (UPG-9.2 / UPG-TASK-NOTE-INJECTION-RECENCY)
# ---------------------------------------------------------------------------

_boot_cfg: dict[str, Any] = _cfg["behavior"]["boot"]

BOOT_MAX_DIRECTIVE_NOTES: int = int(_boot_cfg["max_directive_notes"])
BOOT_MAX_TASK_NOTES: int = int(_boot_cfg["max_task_notes"])

# ---------------------------------------------------------------------------
# Resume surface (UPG-RESUME-SURFACE)
# ---------------------------------------------------------------------------

_resume_cfg: dict[str, Any] = _cfg["behavior"]["resume"]

RESUME_MAX_GOTCHAS: int = int(_resume_cfg["max_gotchas"])

# ---------------------------------------------------------------------------
# Symbol graph — reserved keywords (UPG-JSFLOW-SYMBOLS)
# ---------------------------------------------------------------------------

_sg_cfg: dict[str, Any] = _cfg["symbol_graph"]

SYMBOL_GRAPH_RESERVED_KEYWORDS: dict[str, frozenset[str]] = {
    str(lang): frozenset(str(kw) for kw in kws)
    for lang, kws in _sg_cfg["reserved_keywords"].items()
}

_sg_error_recovery_cfg: dict[str, Any] = _sg_cfg["error_recovery"]

SYMBOL_GRAPH_ERROR_RECOVERY_MIN_SPAN_LINES: int = int(_sg_error_recovery_cfg["min_span_lines"])
SYMBOL_GRAPH_ERROR_RECOVERY_MAX_REPARSE_ATTEMPTS: int = int(
    _sg_error_recovery_cfg["max_reparse_attempts_per_file"]
)
SYMBOL_GRAPH_ERROR_RECOVERY_MAX_EXTEND_STEPS_PER_ATTEMPT: int = int(
    _sg_error_recovery_cfg["max_extend_steps_per_attempt"]
)

# UPG-TRACE-CALLERS-QUALIFIER-VALIDATION: near-miss suggestion cap when a
# trace() "Class.method" qualifier fails to resolve to a real definition.
_sg_trace_qualifier_cfg: dict[str, Any] = _sg_cfg["trace_qualifier"]
SYMBOL_GRAPH_TRACE_QUALIFIER_NEARMISS_MAX: int = int(_sg_trace_qualifier_cfg["nearmiss_max"])

# UPG-LOCATE-FALLBACK-NO-SIMILARITY: candidate pool cap for the qualifier-
# similarity near-miss widening in SymbolGraph.nearest_symbol_names.
_sg_qualified_nearmiss_cfg: dict[str, Any] = _sg_cfg["qualified_nearmiss"]
SYMBOL_GRAPH_QUALIFIED_NEARMISS_CANDIDATE_CAP: int = int(
    _sg_qualified_nearmiss_cfg["candidate_pool_cap"]
)

# ---------------------------------------------------------------------------
# Instance registry — daemon port allocation (UPG-RESTART-PORT-WALK-BREAKS-MCP)
# ---------------------------------------------------------------------------

_instance_registry_cfg: dict[str, Any] = _cfg["instance_registry"]

INSTANCE_REGISTRY_PORT_REUSE_RETRY_ATTEMPTS: int = int(
    _instance_registry_cfg["port_reuse_retry_attempts"]
)
INSTANCE_REGISTRY_PORT_REUSE_RETRY_DELAY_S: float = float(
    _instance_registry_cfg["port_reuse_retry_delay_s"]
)

# ---------------------------------------------------------------------------
# CLI daemon-readiness poll (UPG-CLI-START-READY-RACE)
# ---------------------------------------------------------------------------

_cli_cfg: dict[str, Any] = _cfg["cli"]

CLI_START_READY_POLL_TIMEOUT_S: float = float(_cli_cfg["start_ready_poll_timeout_s"])
CLI_START_READY_POLL_INTERVAL_S: float = float(_cli_cfg["start_ready_poll_interval_s"])
CLI_START_READY_PROBE_TIMEOUT_S: float = float(_cli_cfg["start_ready_probe_timeout_s"])
CLI_VERSION_SKEW_PROBE_TIMEOUT_S: float = float(_cli_cfg["version_skew_probe_timeout_s"])

# ---------------------------------------------------------------------------
# Workspace / watcher tunables (UPG-13.1/13.2/13.3)
# ---------------------------------------------------------------------------

_ws_cfg: dict[str, Any] = _cfg["workspace"]

WORKSPACE_DEFAULT_VECTRIGNORE_DIRS: tuple[str, ...] = tuple(
    str(d) for d in _ws_cfg["default_vectrignore_dirs"]
)

_watcher_cfg: dict[str, Any] = _cfg["watcher"]

WATCHER_TOP_LEVEL_RESCAN_INTERVAL_S: float = float(_watcher_cfg["top_level_rescan_interval_s"])

# UPG-WATCHER-PRESSURE-GOVERNOR: burst coalescing + self-limit tunables.
WATCHER_BURST_FILES_THRESHOLD: int = int(_watcher_cfg["burst_files_threshold"])
WATCHER_BURST_QUIET_SECONDS: float = float(_watcher_cfg["burst_quiet_seconds"])
WATCHER_MAX_RSS_MB: float = float(_watcher_cfg["max_rss_mb"])

# ---------------------------------------------------------------------------
# Hook injection observability (UPG-HOOK-INJECT-OBSERVABILITY)
# ---------------------------------------------------------------------------

_hooks_cfg: dict[str, Any] = _cfg["hooks"]

HOOKS_LOG_INJECTIONS: bool = bool(_hooks_cfg["log_injections"])
HOOKS_LOG_CHARS_PER_TOKEN: int = int(_hooks_cfg["log_chars_per_token"])
HOOKS_MIN_SIMILARITY: float = float(_hooks_cfg["min_similarity"])
HOOKS_COMMIT_NOTE_MAX_FILES: int = int(_hooks_cfg["commit_note_max_files"])
HOOKS_COMMIT_NOTE_MAX_SUBJECT_CHARS: int = int(_hooks_cfg["commit_note_max_subject_chars"])
HOOKS_POST_COMMIT_TIMEOUT_S: float = float(_hooks_cfg["post_commit_timeout_s"])

# ---------------------------------------------------------------------------
# Episode capture
# ---------------------------------------------------------------------------

_episodes_cfg: dict[str, Any] = _cfg["episodes"]

EPISODES_MAX_ROWS: int = int(_episodes_cfg["max_rows"])
EPISODES_TTL_DAYS: float = float(_episodes_cfg["ttl_days"])
EPISODES_CLIENT_TRUNCATE_CHARS: int = int(_episodes_cfg["client_truncate_chars"])
EPISODES_DIGEST_MAX_CHARS: int = int(_episodes_cfg["digest_max_chars"])
EPISODES_DIGEST_HEAD_LINES: int = int(_episodes_cfg["digest_head_lines"])
EPISODES_DIGEST_TAIL_LINES: int = int(_episodes_cfg["digest_tail_lines"])
EPISODES_POST_TIMEOUT_S: float = float(_episodes_cfg["post_timeout_s"])
EPISODES_STALE_TEMP_FILE_SWEEP_AGE_S: float = float(
    _episodes_cfg["stale_temp_file_sweep_age_s"]
)
EPISODES_DISTILL_MAX_ARCS_RENDERED: int = int(_episodes_cfg["distill_max_arcs_rendered"])
EPISODES_DISTILL_RENDER_TOKEN_CAP: int = int(_episodes_cfg["distill_render_token_cap"])
BOUNDARY_PRECOMPACT_ENABLED: bool = bool(_episodes_cfg["boundary_precompact_enabled"])
BOUNDARY_PRECOMPACT_TOKEN_CAP: int = int(_episodes_cfg["boundary_precompact_token_cap"])

# ---------------------------------------------------------------------------
# ingest_traces unresolved-caller/callee warning cap (UPG-7.3)
# ---------------------------------------------------------------------------

_ingest_cfg: dict[str, Any] = _cfg["behavior"]["ingest_traces"]

INGEST_TRACES_MAX_UNRESOLVED_EXAMPLES: int = int(_ingest_cfg["max_unresolved_examples"])

# ---------------------------------------------------------------------------
# MCP param-name aliases (F40-class ergonomics, UPG-TRACE-GRAPH-INCOMPLETE)
# ---------------------------------------------------------------------------

SYMBOL_NAME_PARAM_ALIASES: tuple[str, ...] = tuple(
    str(a) for a in _cfg["behavior"]["symbol_name_param_aliases"]
)

# ---------------------------------------------------------------------------
# Retrieval strategy fallback (UPG-8.2)
# ---------------------------------------------------------------------------

_strat_cfg: dict[str, Any] = _cfg["strategy"]

STRATEGY_DEFAULT_SEMANTIC_WEIGHT: float = float(_strat_cfg["default_semantic_weight"])
STRATEGY_DEFAULT_BM25_WEIGHT: float = float(_strat_cfg["default_bm25_weight"])
# Frameworks the caller model already knows at implementation depth from
# training — config-declared so the set is tunable without a code change
# (UPG-KNOWN-FRAMEWORKS-CONFIG). Matched case-insensitively against
# fingerprint.detected_frameworks by suggest_instruction_style().
STRATEGY_KNOWN_FRAMEWORKS: frozenset[str] = frozenset(
    str(f).lower() for f in _strat_cfg["known_frameworks"]
)
# Dual-vector pool entry — purpose (signature+docstring) vector (ARCH-4)
# ---------------------------------------------------------------------------

_dv_cfg: dict[str, Any] = _cfg["retrieval"]["dual_vector"]

DUAL_VECTOR_ENABLED: bool = bool(_dv_cfg["enabled"])
DUAL_VECTOR_BLEND_MODE: str = str(_dv_cfg["blend_mode"])
DUAL_VECTOR_BLEND_WEIGHT: float = float(_dv_cfg["blend_weight"])
DUAL_VECTOR_MAX_SIGNATURE_LINES: int = int(_dv_cfg["max_signature_lines"])
DUAL_VECTOR_MAX_DOCSTRING_LINES: int = int(_dv_cfg["max_docstring_lines"])
DUAL_VECTOR_MAX_DOCSTRING_CHARS: int = int(_dv_cfg["max_docstring_chars"])

# ---------------------------------------------------------------------------
# Not-found floor — lexical-vocabulary-anchor low-confidence signal
# (UPG-NOTFOUND-FLOOR, UPG-NOTFOUND-FLOOR-2)
# ---------------------------------------------------------------------------

_nff_cfg: dict[str, Any] = _cfg["ranking"]["notfound_floor"]

NOTFOUND_FLOOR_ENABLED: bool = bool(_nff_cfg["enabled"])
NOTFOUND_FLOOR_MIN_TOKEN_LEN: int = int(_nff_cfg["min_content_token_length"])
NOTFOUND_FLOOR_STOPWORDS: frozenset[str] = frozenset(_nff_cfg["stopwords"])
NOTFOUND_FLOOR_MIN_ZERO_DF_TOKENS: int = int(_nff_cfg["min_zero_df_tokens"])
NOTFOUND_FLOOR_MIN_TOP_RELEVANCE: float = float(_nff_cfg["min_top_relevance"])
NOTFOUND_FLOOR_CE_OVERRIDE_MIN_RELEVANCE: float = float(_nff_cfg["ce_override_min_relevance"])
NOTFOUND_FLOOR_BANNER: str = str(_nff_cfg["banner"])
NOTFOUND_FLOOR_BANNER_CLI: str = str(_nff_cfg["banner_cli"])

# UPG-SCORE-ORDER-EXPLAIN: annotate a large displayed-relevance-vs-order
# divergence with the demoting prior's reason (render-only, additive).
# UPG-GATE-V4-MINORS: the margin is a share of the score headroom above
# rank-1 — gap >= MIN_HEADROOM_RATIO * (1 - top_score) — replacing the fixed
# ratio r >= MARGIN_RATIO * top_score, which scores bounded in [0, 1] made
# unsatisfiable on any confident query (top_score >= 2/3).
_soe_cfg: dict[str, Any] = _cfg["ranking"]["score_order_explain"]
SCORE_ORDER_EXPLAIN_ENABLED: bool = bool(_soe_cfg["enabled"])
SCORE_ORDER_EXPLAIN_MIN_HEADROOM_RATIO: float = float(_soe_cfg["min_headroom_ratio"])

# UPG-RESULT-FLOOR: drop sub-floor cross-encoder-relevance results instead of
# returning a full n_results block of ~0.000 ANN neighbours (keeps >=1).
_rf_cfg: dict[str, Any] = _cfg["ranking"]["result_floor"]
RESULT_FLOOR_ENABLED: bool = bool(_rf_cfg["enabled"])
RESULT_FLOOR_MIN_RELEVANCE: float = float(_rf_cfg["min_relevance"])

# UPG-POINTER-MODE-UNIFORM-STRIP: per-result override on low-confidence
# pointer mode — a result whose own ce_relevance clears this floor keeps a
# bounded excerpt of its body instead of a bare pointer, even while the
# surrounding result set is still flagged low confidence.
# UPG-BANNER-CALIBRATION: min_relevance is now an independently-calibrated
# constant, deliberately lower than notfound_floor.min_top_relevance (see
# config.yaml comment) — retain_rank1_always/RANK1_LABEL add a second,
# rank-based (not score-based) exemption for the set's top result alone.
_pmr_cfg: dict[str, Any] = _cfg["ranking"]["pointer_mode_retain"]
POINTER_MODE_RETAIN_ENABLED: bool = bool(_pmr_cfg["enabled"])
POINTER_MODE_RETAIN_MIN_RELEVANCE: float = float(_pmr_cfg["min_relevance"])
POINTER_MODE_RETAIN_EXCERPT_LINES: int = int(_pmr_cfg["excerpt_lines"])
POINTER_MODE_RETAIN_LABEL: str = str(_pmr_cfg["label"])
POINTER_MODE_RETAIN_RANK1_ALWAYS: bool = bool(_pmr_cfg["retain_rank1_always"])
POINTER_MODE_RETAIN_RANK1_LABEL: str = str(_pmr_cfg["rank1_label"])

# UPG-BANNER-CALIBRATION Phase 3: an alternative low-confidence sub-signal
# (ranking.relative_cliff in config.yaml) — a rank-1-to-rank-2 ce_relevance
# drop test, an alternative to the absolute min_top_relevance floor. SHIPPED
# DEFAULT OFF; the Phase 3 distribution study on the three unmeasured corpora
# is what would justify turning it on, and the value of min_drop. See the
# config.yaml comment for the headroom-relative gap formula; the cliff
# function itself is a pure-Python comparison of two already-computed scores
# in agent.ranking_cliff.relative_cliff_fires, with no query-side logic.
_rc_cfg: dict[str, Any] = _cfg["ranking"]["relative_cliff"]
RELATIVE_CLIFF_ENABLED: bool = bool(_rc_cfg["enabled"])
RELATIVE_CLIFF_MIN_TOP: float = float(_rc_cfg["min_top"])
RELATIVE_CLIFF_MIN_DROP: float = float(_rc_cfg["min_drop"])

# ---------------------------------------------------------------------------
# Search — additive identifier-shape symbol-graph hint (UPG-QUERYTYPE-REROUTE)
# ---------------------------------------------------------------------------

_id_hint_cfg: dict[str, Any] = _cfg["search"]["identifier_hint"]

SEARCH_IDENTIFIER_HINT_ENABLED: bool = bool(_id_hint_cfg["enabled"])
SEARCH_IDENTIFIER_HINT_MAX_IDENTIFIERS: int = int(_id_hint_cfg["max_identifiers"])
SEARCH_IDENTIFIER_HINT_MAX_LOCATIONS: int = int(_id_hint_cfg["max_locations"])

# UPG-NEARMISS-SYMBOL-NAMES: additive, honestly-labeled near-miss names for an
# identifier-shaped token that fails EXACT symbol-graph resolution.
SEARCH_IDENTIFIER_HINT_NEARMISS_ENABLED: bool = bool(_id_hint_cfg["nearmiss_enabled"])
SEARCH_IDENTIFIER_HINT_NEARMISS_MAX: int = int(_id_hint_cfg["nearmiss_max"])
SEARCH_IDENTIFIER_HINT_NEARMISS_MIN_PREFIX_LEN: int = int(_id_hint_cfg["nearmiss_min_prefix_len"])

# ---------------------------------------------------------------------------
# Server bind defaults (T26): one source of truth for the default port/host,
# de-hardcoding the 8765 / 127.0.0.1 literals formerly repeated across
# main.py / app / api / integrations. The VECTR_PORT env var still overrides.
# ---------------------------------------------------------------------------

DEFAULT_PORT: int = 8765
DEFAULT_HOST: str = "127.0.0.1"

# ---------------------------------------------------------------------------
# Embedding — default local model (UPG-EMBEDDER-SWAP-GRANITE)
# ---------------------------------------------------------------------------

EMBEDDING_DEFAULT_MODEL: str = str(_cfg["embedding"]["default_model"])
EMBEDDING_MAX_SEQ_LENGTH: int = int(_cfg["embedding"]["max_seq_length"])

# UPG-EMBED-THREAD-CONTENTION: 0 means "auto" — derive from
# thread_cap_auto_fraction * os.cpu_count(), minimum 1 thread. A positive
# configured value is an explicit operator override, used verbatim.
EMBEDDING_THREAD_CAP_CONFIGURED: int = int(_cfg["embedding"]["thread_cap"])
EMBEDDING_THREAD_CAP_AUTO_FRACTION: float = float(_cfg["embedding"]["thread_cap_auto_fraction"])


def _resolve_embedding_thread_cap() -> int:
    if EMBEDDING_THREAD_CAP_CONFIGURED > 0:
        return EMBEDDING_THREAD_CAP_CONFIGURED
    cores = _os.cpu_count() or 2
    return max(1, int(cores * EMBEDDING_THREAD_CAP_AUTO_FRACTION))


EMBEDDING_THREAD_CAP: int = _resolve_embedding_thread_cap()

# ---------------------------------------------------------------------------
# Deterministic re-fetch-by-chunk-id contract (UPG-CTX-EVICT)
# ---------------------------------------------------------------------------

FETCH_MAX_IDS_PER_CALL: int = int(_cfg["fetch"]["max_ids_per_call"])
FETCH_MISALIGN_NEAREST_MAX: int = int(_cfg["fetch"]["misalign_nearest_max"])

# ---------------------------------------------------------------------------
# Arc detection: command normalization + mutation-similarity + streaming
# detector state machine.
# ---------------------------------------------------------------------------

_arc_cfg: dict[str, Any] = _cfg["arc_detection"]

# Master switch — whether the episode write path feeds bash/edit episodes
# into ArcDetector.observe() at all.
ARC_DETECTION_ENABLED: bool = bool(_arc_cfg["enabled"])

_arc_norm_cfg: dict[str, Any] = _arc_cfg["normalization"]

ARC_NORM_UUID_REGEX: str = str(_arc_norm_cfg["uuid_regex"])
ARC_NORM_VERSION_REGEX: str = str(_arc_norm_cfg["version_regex"])
ARC_NORM_NUM_REGEX: str = str(_arc_norm_cfg["num_regex"])
ARC_NORM_PATH_EXTENSION_REGEX: str = str(_arc_norm_cfg["path_extension_regex"])
ARC_NORM_ENV_ASSIGNMENT_REGEX: str = str(_arc_norm_cfg["env_assignment_regex"])
ARC_NORM_STDERR_MERGE_TOKEN: str = str(_arc_norm_cfg["stderr_merge_token"])
ARC_NORM_MAX_VERB_TOKENS: int = int(_arc_norm_cfg["max_verb_tokens"])
ARC_NORM_WRAPPER_PREFIXES: dict[str, str] = {
    str(k): str(v) for k, v in _arc_norm_cfg["wrapper_prefixes"].items()
}
ARC_NORM_NICE_NICENESS_FLAG: str = str(_arc_norm_cfg["nice_niceness_flag"])
ARC_NORM_PIPELINE_DISPLAY_ONLY_VERBS: frozenset[str] = frozenset(
    str(v) for v in _arc_norm_cfg["pipeline_display_only_verbs"]
)

_arc_sim_cfg: dict[str, Any] = _arc_cfg["similarity"]

ARC_SIMILARITY_VERB_WEIGHT: float = float(_arc_sim_cfg["verb_weight"])
ARC_SIMILARITY_FLAG_WEIGHT: float = float(_arc_sim_cfg["flag_weight"])
ARC_SIMILARITY_ARG_WEIGHT: float = float(_arc_sim_cfg["arg_weight"])
ARC_SIMILARITY_VERB_SOFT_MATCH_MIN_RATIO: float = float(_arc_sim_cfg["verb_soft_match_min_ratio"])
ARC_SIMILARITY_VERB_SOFT_MATCH_SCORE: float = float(_arc_sim_cfg["verb_soft_match_score"])
ARC_MUTATION_BAND_MIN: float = float(_arc_sim_cfg["mutation_band_min"])
ARC_MUTATION_BAND_MAX: float = float(_arc_sim_cfg["mutation_band_max"])

_arc_window_cfg: dict[str, Any] = _arc_cfg["window"]

ARC_WINDOW_MAX_COMMANDS: int = int(_arc_window_cfg["max_commands"])
ARC_WINDOW_TTL_SECONDS: float = float(_arc_window_cfg["ttl_seconds"])
ARC_WINDOW_MAX_PENDING_PER_VERB_FAMILY: int = int(_arc_window_cfg["max_pending_per_verb_family"])
ARC_TS_MONOTONIC_FALLBACK_SECONDS: float = float(_arc_window_cfg["ts_monotonic_fallback_seconds"])

_arc_flake_cfg: dict[str, Any] = _arc_cfg["flake"]

ARC_FLAKE_SUPPRESS_MIN_COUNT: int = int(_arc_flake_cfg["suppress_min_count"])
ARC_FLAKE_NEAR_THRESHOLD_MIN: float = float(_arc_flake_cfg["near_threshold_min"])

ARC_TRANSIENT_MARKER_IDS: frozenset[str] = frozenset(
    str(m) for m in _arc_cfg["transient_marker_ids"]
)

# ---------------------------------------------------------------------------
# Proactive context — bundled defaults (UPG-PRO). Runtime/deployment toggles
# come from env vars (VECTR_PROACTIVE*), applied over these defaults by
# agent/proactive/settings.py. These are the product-behaviour defaults only.
# ---------------------------------------------------------------------------

_pro_cfg: dict[str, Any] = _cfg["proactive"]

PROACTIVE_ENABLED: bool = bool(_pro_cfg["enabled"])
PROACTIVE_MIN_SIMILARITY: float = float(_pro_cfg["min_similarity"])
PROACTIVE_MAX_ITEMS_PER_EVENT: int = int(_pro_cfg["max_items_per_event"])
PROACTIVE_MAX_CHARS_PER_EVENT: int = int(_pro_cfg["max_chars_per_event"])
PROACTIVE_BODY_TRUNCATION_MIN_BOUNDARY_FRACTION: float = float(
    _pro_cfg["body_truncation_min_boundary_fraction"]
)
_pro_envelope_cfg: dict[str, Any] = _pro_cfg["envelope"]
PROACTIVE_ENVELOPE_OPEN_AUTO: str = str(_pro_envelope_cfg["open_auto"])
PROACTIVE_ENVELOPE_OPEN_AGENT: str = str(_pro_envelope_cfg["open_agent"])
PROACTIVE_ENVELOPE_OPEN_HUMAN: str = str(_pro_envelope_cfg["open_human"])
PROACTIVE_ENVELOPE_CLOSE: str = str(_pro_envelope_cfg["close"])

PROACTIVE_COOLDOWN_ITEMS: int = int(_pro_cfg["cooldown_items"])
# UPG-PROXY-COOLDOWN-NO-TIME-DECAY: wall-clock suppression window (seconds) on
# top of the count ring. 0 disables the decay (pure count ring, the historical
# behaviour); agent/proactive/settings.py normalises non-positive to None.
PROACTIVE_COOLDOWN_TTL_SECONDS: float = float(_pro_cfg["cooldown_ttl_seconds"])

# UPG-PROXY-INJECT-PRECISION: structural (path-anchored) channel relevance
# gate. Every value below is a STATIC config threshold or a note.kind
# allowlist — never derived from query/window content (no-query-heuristics
# rule). See the matching config.yaml comment block for the full rationale.
PROACTIVE_STRUCTURAL_KINDS: tuple[str, ...] = tuple(str(k) for k in _pro_cfg["structural_kinds"])
PROACTIVE_STRUCTURAL_OVERFETCH_MULTIPLIER: int = int(_pro_cfg["structural_overfetch_multiplier"])
PROACTIVE_STRUCTURAL_OVERFETCH_CEILING: int = int(_pro_cfg["structural_overfetch_ceiling"])

_pro_structural_scores_cfg: dict[str, Any] = _pro_cfg["structural_scores"]
PROACTIVE_STRUCTURAL_SCORE_DECLARED_ANCHOR: float = float(
    _pro_structural_scores_cfg["declared_anchor"]
)
PROACTIVE_STRUCTURAL_SCORE_DECLARED_TRIGGER: float = float(
    _pro_structural_scores_cfg["declared_trigger"]
)
PROACTIVE_STRUCTURAL_SCORE_GOTCHA_MENTION: float = float(
    _pro_structural_scores_cfg["gotcha_mention"]
)
PROACTIVE_STRUCTURAL_SCORE_MENTION: float = float(_pro_structural_scores_cfg["mention"])

PROACTIVE_MAX_WEAK_STRUCTURAL_ITEMS: int = int(_pro_cfg["max_weak_structural_items"])

_pro_matchers_cfg: dict[str, Any] = _pro_cfg["matchers"]
PROACTIVE_MATCHER_STRUCTURAL_NOTE: bool = bool(_pro_matchers_cfg["structural_note"])
PROACTIVE_MATCHER_SEMANTIC_NOTE: bool = bool(_pro_matchers_cfg["semantic_note"])
PROACTIVE_MATCHER_CODE_SEARCH: bool = bool(_pro_matchers_cfg["code_search"])

_proxy_cfg: dict[str, Any] = _pro_cfg["proxy"]
# (UPG-PROACTIVE-DEAD-GATES: the old `proxy.enabled` constant is gone — the
# proxy listener's injection behaviour is governed by `proxy.inject`, and the
# localhost boundary by settings.py's unconditional `proactive_bind_is_loopback`.)
PROACTIVE_PROXY_HOST: str = str(_proxy_cfg["host"])
PROACTIVE_PROXY_PORT: int = int(_proxy_cfg["port"])
PROACTIVE_PROXY_UPSTREAM_BASE_URL: str = str(_proxy_cfg["upstream_base_url"])
PROACTIVE_PROXY_CONNECT_TIMEOUT_S: float = float(_proxy_cfg["connect_timeout_s"])
PROACTIVE_PROXY_READ_TIMEOUT_S: float = float(_proxy_cfg["read_timeout_s"])
PROACTIVE_PROXY_INJECT: bool = bool(_proxy_cfg["inject"])
PROACTIVE_PROXY_INJECT_BUDGET_MS: int = int(_proxy_cfg["inject_budget_ms"])
PROACTIVE_PROXY_INJECT_PROVIDER_TIMEOUT_FRACTION: float = float(
    _proxy_cfg["inject_provider_timeout_fraction"]
)
PROACTIVE_PROXY_INJECT_PROVIDER_TIMEOUT_MAX_S: float = float(
    _proxy_cfg["inject_provider_timeout_max_s"]
)
PROACTIVE_PROXY_EXCLUDE_DIRECTIVE_NOTES: bool = bool(_proxy_cfg["exclude_directive_notes"])

_cache_cfg: dict[str, Any] = _pro_cfg["cache"]
PROACTIVE_CACHE_ENABLED: bool = bool(_cache_cfg["enabled"])
PROACTIVE_CACHE_MAX_ENTRIES: int = int(_cache_cfg["max_entries"])
PROACTIVE_CACHE_TTL_SECONDS: float = float(_cache_cfg["ttl_seconds"])
PROACTIVE_CACHE_SIMILARITY_THRESHOLD: float = float(_cache_cfg["similarity_threshold"])

_resp_cache_cfg: dict[str, Any] = _cache_cfg["response_cache"]
PROACTIVE_RESPONSE_CACHE_ENABLED: bool = bool(_resp_cache_cfg["enabled"])
PROACTIVE_RESPONSE_CACHE_TTL_SECONDS: float = float(_resp_cache_cfg["ttl_seconds"])
PROACTIVE_RESPONSE_CACHE_MAX_ENTRIES: int = int(_resp_cache_cfg["max_entries"])

# ---------------------------------------------------------------------------
# Trigger engine (TRIGGER-ENGINE) — total order + injection budgets.
# See agent/trigger_engine.py.
# ---------------------------------------------------------------------------

_trig_cfg: dict[str, Any] = _cfg["memory_triggers"]
_trig_order_cfg: dict[str, Any] = _trig_cfg["total_order"]

MEMORY_TRIGGER_KIND_PRIORITY: tuple[str, ...] = tuple(
    str(k) for k in _trig_order_cfg["kind_priority"]
)
MEMORY_TRIGGER_PRIORITY_RANK: tuple[str, ...] = tuple(
    str(p) for p in _trig_order_cfg["priority_rank"]
)

_trig_inject_cfg: dict[str, Any] = _trig_cfg["injection"]
MEMORY_TRIGGER_PER_INJECTION_TOKEN_CAP: int = int(_trig_inject_cfg["per_injection_token_cap"])
MEMORY_TRIGGER_PER_SESSION_TOKEN_CAP: int = int(_trig_inject_cfg["per_session_token_cap"])
MEMORY_TRIGGER_CHARS_PER_TOKEN: int = int(_trig_inject_cfg["chars_per_token"])

# Serving-policy hardening (UPG-MEMORY-STATE-MACHINE §5.4) — the ordinary-
# turn allowance shared by every per-turn injection surface combined
# (PreToolUse anchored/command-family + prompt-time semantic), and each
# kind's override of the flat per-injection cap above. `per_kind_token_cap`
# itself is a required key (direct subscript — a config.yaml missing the key
# entirely raises KeyError at import); individual kind entries within it are
# legitimately a partial mapping (a kind absent from it falls back to
# MEMORY_TRIGGER_PER_INJECTION_TOKEN_CAP, consulted by trigger_engine.py's
# pack_injection()), since not every kind needs a tighter override.
MEMORY_TRIGGER_PER_TURN_TOKEN_CAP: int = int(_trig_inject_cfg["per_turn_token_cap"])
MEMORY_TRIGGER_PER_KIND_TOKEN_CAP: dict[str, int] = {
    str(kind): int(cap) for kind, cap in _trig_inject_cfg["per_kind_token_cap"].items()
}
# UPG-HOOK-GOTCHA-CAP-TITLE-ONLY: min fraction of a trim's char budget a
# sentence/word boundary must still preserve to be used, else `pack_injection`
# falls back to a hard character cut — see agent/trigger_engine.py's
# `_trim_full_text_to_cap()`. Same shape as the proxy channel's
# PROACTIVE_BODY_TRUNCATION_MIN_BOUNDARY_FRACTION, kept as a distinct key.
MEMORY_TRIGGER_BODY_TRUNCATION_MIN_BOUNDARY_FRACTION: float = float(
    _trig_inject_cfg["body_truncation_min_boundary_fraction"]
)

# The M (semantic) primitive's fixed per-kind cosine thresholds. Built by direct
# subscript against every kind already enumerated in kind_priority above, so
# a config.yaml missing an entry for any kind raises KeyError here at import
# rather than silently defaulting a kind to "never matches" or "always
# matches" at runtime.
_trig_semantic_cfg: dict[str, Any] = _trig_cfg["semantic"]
_trig_theta_cfg: dict[str, Any] = _trig_semantic_cfg["theta_by_kind"]
MEMORY_TRIGGER_SEMANTIC_THETA_BY_KIND: dict[str, float] = {
    kind: float(_trig_theta_cfg[kind]) for kind in MEMORY_TRIGGER_KIND_PRIORITY
}
# Serving-policy hardening (UPG-MEMORY-STATE-MACHINE §5.5) — see the cooldown_turns key's own
# config.yaml comment for the turns-vs-wall-clock-seconds distinction.
MEMORY_TRIGGER_SEMANTIC_COOLDOWN_TURNS: int = int(_trig_semantic_cfg["cooldown_turns"])

# ---------------------------------------------------------------------------
# Write-time offers computed alongside remember() — additive only, never a
# gate on the write. See agent/working_context_store/_related.py and
# agent/proxy_anchors.py.
# ---------------------------------------------------------------------------

_mem_write_cfg: dict[str, Any] = _cfg["memory_write"]

_related_notes_cfg: dict[str, Any] = _mem_write_cfg["related_notes"]
MEMORY_WRITE_RELATED_ENABLED: bool = bool(_related_notes_cfg["enabled"])
MEMORY_WRITE_RELATED_LIMIT: int = int(_related_notes_cfg["limit"])
MEMORY_WRITE_RELATED_MIN_SIMILARITY: float = float(_related_notes_cfg["min_similarity"])
MEMORY_WRITE_RELATED_REVOKED_ENABLED: bool = bool(_related_notes_cfg["revoked_enabled"])
MEMORY_WRITE_RELATED_REVOKED_LIMIT: int = int(_related_notes_cfg["revoked_limit"])
MEMORY_WRITE_RELATED_REVOKED_QUERY_FLOOR: int = int(_related_notes_cfg["revoked_query_floor"])

_user_quote_cfg: dict[str, Any] = _mem_write_cfg["user_quote"]
MEMORY_WRITE_USER_QUOTE_MIN_CHARS: int = int(_user_quote_cfg["min_chars"])
MEMORY_WRITE_USER_QUOTE_AUTO_BIND_MAX_AGE_SECONDS: float = float(
    _user_quote_cfg["auto_bind_max_age_seconds"]
)
MEMORY_WRITE_USER_QUOTE_SPAN_BIND_MIN_CHARS: int = int(_user_quote_cfg["span_bind_min_chars"])
MEMORY_WRITE_USER_QUOTE_SPAN_BIND_MAX_PRODUCT_CHARS: int = int(
    _user_quote_cfg["span_bind_max_product_chars"]
)

_proxy_suggest_cfg: dict[str, Any] = _mem_write_cfg["proxy_anchor_suggestions"]
MEMORY_WRITE_PROXY_SUGGEST_ENABLED: bool = bool(_proxy_suggest_cfg["enabled"])
MEMORY_WRITE_PROXY_SUGGEST_LIMIT: int = int(_proxy_suggest_cfg["limit"])

_content_file_cfg: dict[str, Any] = _mem_write_cfg["content_file"]
MEMORY_WRITE_CONTENT_FILE_MAX_BYTES: int = int(_content_file_cfg["max_bytes"])
# UPG-REMEMBER-CONTENT-FILE-PATH-REFUSAL: operator-configured additional
# absolute roots content_file reads are allowed to resolve under, beyond the
# primary workspace root and the multi-root `extra_roots` already accepted by
# the resolver. Operator-only — the trust decision is made at daemon startup
# (config), not per call, so a calling agent cannot widen containment.
# Empty tuple by default: the boundary is unchanged until an operator opts in.
MEMORY_WRITE_CONTENT_FILE_ADDITIONAL_READABLE_ROOTS: tuple[str, ...] = tuple(
    str(r) for r in _content_file_cfg["additional_readable_roots"]
)

# ---------------------------------------------------------------------------
# UPG-TASK-SUPERSEDES-HYGIENE: vectr_status stale-task nudge thresholds.
# Nudge only — never decay, auto-supersede, or auto-expire a note.
# ---------------------------------------------------------------------------

_hygiene_cfg: dict[str, Any] = _cfg["memory_hygiene"]
MEMORY_HYGIENE_STALE_TASK_WARN_COUNT: int = int(_hygiene_cfg["stale_task_warn_count"])
MEMORY_HYGIENE_STALE_TASK_WARN_AGE_DAYS: int = int(_hygiene_cfg["stale_task_warn_age_days"])

_memory_export_cfg: dict[str, Any] = _cfg["memory_export"]
MEMORY_EXPORT_DEBOUNCE_SECONDS: float = float(_memory_export_cfg["debounce_seconds"])

# UPG-MEMORY-IMPORT: closed list of workspace-relative paths the import
# command's default-discovery path walks when --path is not given. Closed
# (same reasoning as VALID_KINDS / SCOPE_VALUES / etc. — a fixed protocol
# vocabulary rather than a tunable threshold), lives here as a plain
# constant rather than in config.yaml's operator-tunable section. Mirrors
# `memory_import.candidate_files` in config.yaml, but reading the
# constant form from the same dict makes the value the importer sees
# match the documented default if a test does not override it.
_memory_import_cfg: dict[str, Any] = _cfg["memory_import"]
MEMORY_IMPORT_CANDIDATE_FILES: tuple[str, ...] = tuple(_memory_import_cfg["candidate_files"])

# ---------------------------------------------------------------------------
# UPG-MEMORY-DECAY-KIND-SCOPED: per-kind decay half-life (decay_old_notes(),
# ranking-only) and TTL (purge_expired_notes(), visibility-only — a
# note_events 'expired' transition, never a row DELETE) — see the
# memory_decay config.yaml comment for the full contract. `null` in
# config.yaml maps to `None` here, meaning "exempt from this path at any
# age"; the only kind exempt from both today is "directive".
#
# The seven kind names are duplicated here as a literal tuple rather than
# imported from agent.working_context_store._types.VALID_KINDS, because
# that package's __init__ (via _store.py -> _user_quote.py/_content_file.py)
# already imports names FROM agent.config at module level — importing
# VALID_KINDS the other way round here creates a genuine circular import
# (verified: `agent.config` would be "partially initialized" when
# working_context_store tries to read MEMORY_WRITE_USER_QUOTE_MIN_CHARS off
# it). Same closed-vocabulary-duplication reasoning _events.py's
# NOTE_EVENT_KINDS docstring already gives for not reusing EVENT_VALUES.
# ---------------------------------------------------------------------------

_MEMORY_DECAY_KINDS: tuple[str, ...] = (
    "directive", "task", "gotcha", "finding", "reference", "decision", "operational",
)

_decay_cfg: dict[str, Any] = _cfg["memory_decay"]
_decay_half_life_cfg: dict[str, Any] = _decay_cfg["half_life_days_by_kind"]
_decay_ttl_cfg: dict[str, Any] = _decay_cfg["ttl_days_by_kind"]

MEMORY_DECAY_HALF_LIFE_DAYS_BY_KIND: dict[str, float | None] = {
    kind: (None if _decay_half_life_cfg[kind] is None else float(_decay_half_life_cfg[kind]))
    for kind in _MEMORY_DECAY_KINDS
}
MEMORY_DECAY_TTL_DAYS_BY_KIND: dict[str, float | None] = {
    kind: (None if _decay_ttl_cfg[kind] is None else float(_decay_ttl_cfg[kind]))
    for kind in _MEMORY_DECAY_KINDS
}

# ---------------------------------------------------------------------------
# UPG-PROXY-SUBSTRING-ANCHOR: recall_for_path() SQL over-fetch pool sizing.
# See agent/working_context_store/_store.py:recall_for_path() and the
# memory_recall_for_path config.yaml comment for why the SQL prefilter must
# over-fetch before the Python-side path-boundary filter narrows it down.
# ---------------------------------------------------------------------------

_recall_for_path_cfg: dict[str, Any] = _cfg["memory_recall_for_path"]
MEMORY_RECALL_FOR_PATH_OVERFETCH_MULTIPLIER: int = int(_recall_for_path_cfg["overfetch_multiplier"])
MEMORY_RECALL_FOR_PATH_OVERFETCH_CEILING: int = int(_recall_for_path_cfg["overfetch_ceiling"])

# ---------------------------------------------------------------------------
# UPG-RECALL-MISS-FLOOR parts (b)/(c): Tier 0 (directive + pinned) unconditional
# injection and the deterministic tag/anchor/symbol/title channels composed
# with the ranked (semantic/SQL) result in WorkingContextStore.recall().
# See the recall_floor config.yaml comment for the full rationale.
# ---------------------------------------------------------------------------

_recall_floor_cfg: dict[str, Any] = _cfg["recall_floor"]
RECALL_FLOOR_TIER0_MAX_NOTES: int = int(_recall_floor_cfg["tier0_max_notes"])
RECALL_FLOOR_DETERMINISTIC_MAX_NOTES: int = int(_recall_floor_cfg["deterministic_max_notes"])
RECALL_FLOOR_METADATA_SCAN_CEILING: int = int(_recall_floor_cfg["metadata_scan_ceiling"])
RECALL_FLOOR_TITLE_MIN_TOKEN_LEN: int = int(_recall_floor_cfg["title_min_token_len"])
RECALL_FLOOR_TITLE_MIN_OVERLAP_TOKENS: int = int(_recall_floor_cfg["title_min_overlap_tokens"])
RECALL_FLOOR_TIER0_MIN_NOTES: int = int(_recall_floor_cfg["tier0_min_notes"])
RECALL_FLOOR_TIER0_MAX_SHARE_OF_LIMIT: float = float(_recall_floor_cfg["tier0_max_share_of_limit"])
RECALL_FLOOR_DETERMINISTIC_MAX_SHARE_OF_REMAINING: float = float(
    _recall_floor_cfg["deterministic_max_share_of_remaining"]
)
RECALL_FLOOR_DETERMINISTIC_ENABLED: bool = bool(_recall_floor_cfg["deterministic_enabled"])

# ---------------------------------------------------------------------------
# UPG-TEST-CACHE-ISOLATION: single override point for vectr's on-disk cache
# root. Every module that resolves a cache path (workspace DB dirs, the
# Hugging Face model cache, cache-maintenance sweeps) calls vectr_cache_root()
# instead of independently hardcoding Path.home() / ".cache" / "vectr" — the
# most direct consequence being that the test suite can redirect the WHOLE
# cache root to a tmp_path via one env var (see tests/conftest.py's
# `_isolated_cache_root` fixture) rather than each callsite needing its own
# override plumbed through. Not a config.yaml tunable: this is a runtime
# location override (same category as VECTR_DB_DIR/VECTR_PORT), not a magic
# number/threshold/weight.
# ---------------------------------------------------------------------------

CACHE_DIR_ENV: str = "VECTR_CACHE_DIR"


def vectr_cache_root() -> Path:
    """Root directory for vectr's on-disk cache (workspace DB dirs, HF model
    cache). Reads CACHE_DIR_ENV (VECTR_CACHE_DIR) at call time — never cached
    at import — so a caller that sets the env var after this module has
    already been imported (e.g. a pytest fixture) is still honored on every
    subsequent call. Falls back to the standard ~/.cache/vectr location when
    unset."""
    override = _os.environ.get(CACHE_DIR_ENV)
    return Path(override) if override else Path.home() / ".cache" / "vectr"
