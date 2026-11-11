"""FastAPI route handlers: REST API + MCP protocol endpoints."""
from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Body, HTTPException, Query, Request, Response
from starlette.concurrency import run_in_threadpool

import agent.config as config
from agent.chroma_dispatch import dispatch_chroma_async
from agent.working_context_store import USER_STATED_PROVENANCE, bind_user_quote, resolve_remember_content

from app.models import (
    AnchorRequest,
    AnchorResponse,
    UnanchorRequest,
    UnanchorResponse,
    ArcRecord,
    ArcResolveResult,
    ArcsDismissRequest,
    ArcsDismissResponse,
    ArcsResponse,
    BoundaryPrecompactResponse,
    CommitNoteRequest,
    CommitNoteResponse,
    EpisodeRecord,
    EpisodeRequest,
    EpisodeResponse,
    EpisodesResponse,
    FetchEntry,
    FetchRequest,
    FetchResponse,
    ForgetRequest,
    HealthResponse,
    IndexRequest,
    IndexResponse,
    LocateRequest,
    LocateResponse,
    MapSaveRequest,
    MapSaveResponse,
    PinRequest,
    PinResponse,
    ProactiveRequest,
    ProactiveResponse,
    PromoteRequest,
    PromoteResponse,
    RecallRequest,
    RecallResponse,
    ReinstateRequest,
    ReinstateResponse,
    RelatedNoteModel,
    RememberRequest,
    RememberResponse,
    ResumeResponse,
    RevokedRelatedNoteModel,
    RevokeRequest,
    RevokeResponse,
    SearchRequest,
    SearchResponse,
    SupersedeRequest,
    SupersedeResponse,
    SnapshotRequest,
    SnapshotResponse,
    StatusResponse,
    TraceRequest,
    TraceResponse,
    TriggerResetRequest,
    TriggerResetResponse,
    CodeChunkResult,
    SymbolResult,
)
from integrations.mcp_server import (
    MCP_SERVER_INFO,
    handle_tools_call,
    handle_tools_list,
)
from integrations.mcp_server._schemas import MEMORY_READY_TOOLS

router = APIRouter()


def _service(request: Request):
    return request.app.state.service


def _require_fully_ready(svc) -> None:
    """503 gate for search-touching REST routes (UPG-STDIO-MEMORY-READY):
    index/search/fetch/locate/trace/map/evict-hint all read the indexer,
    searcher, watcher, or symbol graph, which only exist once phase 2 of
    `VectrService` construction has completed. Working-memory routes
    (remember/recall/forget/snapshot/status) never call this — they are
    servable from process start. Keyed purely on service state, mirroring
    the existing memory_only/search_only 503 pattern below."""
    if not getattr(svc, "fully_ready", True):
        from app.service import _STILL_INITIALIZING_MSG
        raise HTTPException(
            status_code=503,
            detail={"error": "still_initialising", "detail": _STILL_INITIALIZING_MSG},
        )


def _mcp_tool_still_initialising(svc, tool_name: str) -> dict | None:
    """Per-tool readiness gate for the MCP transport (UPG-STDIO-MEMORY-READY),
    keyed on TOOL NAME and service STATE only, never on call arguments.
    Working-memory tools (`MEMORY_READY_TOOLS`) are servable the moment the
    service object exists (phase 1); every other tool additionally needs
    `svc.fully_ready` (phase 2 — embedder/indexer/searcher/watcher/symbol
    graph). Returns the graceful tool-call response to send verbatim when
    still gated, or None when the call may proceed."""
    if tool_name in MEMORY_READY_TOOLS or getattr(svc, "fully_ready", True):
        return None
    from app.service import _STILL_INITIALIZING_MSG
    return {"content": [{"type": "text", "text": _STILL_INITIALIZING_MSG}], "isError": False}


# ---------------------------------------------------------------------------
# REST API — L3 (existing)
# ---------------------------------------------------------------------------

@router.get("/v1/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    svc = _service(request)
    return HealthResponse(
        status="ok",
        embed_model=svc._embed_model,
        last_indexed=svc.last_indexed,
    )


@router.post("/v1/index", response_model=IndexResponse)
async def index(body: IndexRequest, request: Request) -> IndexResponse:
    svc = _service(request)
    _require_fully_ready(svc)
    if getattr(svc, "memory_only", False):
        from app.service import _MEMORY_ONLY_MSG
        raise HTTPException(status_code=503, detail={"error": "memory_only_mode", "detail": _MEMORY_ONLY_MSG})
    try:
        # UPG-REST-STARVATION requirement #1: a full-workspace index is
        # bulk chroma/embedding work — run it off the event-loop thread so a
        # large `/v1/index` call never blocks every other request for its
        # full duration (previously `svc.index()` ran directly on the async
        # handler, holding the event loop for as long as indexing took).
        # Runs on Starlette's own general-purpose threadpool (many workers),
        # not the single-worker executor search/fetch use below — sharing
        # that executor would serialize every /v1/search, /v1/fetch, and MCP
        # search/fetch call behind the full duration of whatever index call
        # happened to be in flight, a new form of starvation this route
        # never had before.
        files, chunks, elapsed = await run_in_threadpool(svc.index, body.path, force=body.force)
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"error": "index_failed", "detail": str(exc)})
    return IndexResponse(
        indexed_files=files,
        total_chunks=chunks,
        processing_ms=elapsed,
    )


@router.post("/v1/search", response_model=SearchResponse)
async def search(body: SearchRequest, request: Request) -> SearchResponse:
    t0 = time.monotonic()
    svc = _service(request)
    _require_fully_ready(svc)
    if getattr(svc, "memory_only", False):
        from app.service import _MEMORY_ONLY_MSG
        raise HTTPException(status_code=503, detail={"error": "memory_only_mode", "detail": _MEMORY_ONLY_MSG})
    try:
        # Search reaches the vector store (embed + query) — dispatched off
        # this route's event loop through the service's dedicated executor
        # rather than run in place.
        results, query_ms = await dispatch_chroma_async(
            svc, svc.search, body.query, n_results=body.n_results, language=body.language
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"error": "search_failed", "detail": str(exc)})

    return SearchResponse(
        results=[
            CodeChunkResult(
                file=r.file_path,
                lines=r.lines,
                symbol=r.symbol_name or None,
                language=r.language,
                score=r.score,
                content=r.content,
                symbol_start_line=getattr(r, "symbol_start_line", 0),
                symbol_end_line=getattr(r, "symbol_end_line", 0),
                id=getattr(r, "chunk_id", ""),
                score_source=getattr(r, "score_source", "dense"),
            )
            for r in results
        ],
        query_time_ms=query_ms,
        chunks_searched=svc.total_chunks,
        processing_ms=int((time.monotonic() - t0) * 1000),
        low_confidence=getattr(results, "low_confidence", False),
    )


@router.post("/v1/fetch", response_model=FetchResponse)
async def fetch(body: FetchRequest, request: Request) -> FetchResponse:
    """Deterministic re-fetch by chunk id (UPG-CTX-EVICT part a) — no
    embedding, no rerank, just the exact chunk(s) named by id."""
    t0 = time.monotonic()
    svc = _service(request)
    _require_fully_ready(svc)
    if getattr(svc, "memory_only", False):
        from app.service import _MEMORY_ONLY_MSG
        raise HTTPException(status_code=503, detail={"error": "memory_only_mode", "detail": _MEMORY_ONLY_MSG})
    try:
        # Fetch is a direct vector-store get by id — same off-loop dispatch
        # as search above.
        entries = await dispatch_chroma_async(svc, svc.fetch, body.ids)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"error": "too_many_ids", "detail": str(exc)})
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"error": "fetch_failed", "detail": str(exc)})

    from app.service import _FETCH_NOT_FOUND_NOTE
    # UPG-FETCH-ID-MISALIGN-MSG: the shared note is the genuine
    # "file likely changed" signal only — a missing entry whose id defaults
    # to that reason (or predates this field, e.g. an older indexer/mock)
    # counts toward it; a "misaligned" entry carries its own reason +
    # nearest_ids per-entry instead, so it must not also trigger the
    # generic note.
    any_file_changed = any(
        not e["found"] and e.get("reason", "file_changed") != "misaligned"
        for e in entries
    )
    return FetchResponse(
        results=[
            FetchEntry(
                id=e["id"],
                found=e["found"],
                file_path=e.get("file_path", ""),
                lines=f"{e['start_line']}-{e['end_line']}" if e["found"] else "",
                symbol=e.get("symbol_name") or None if e["found"] else None,
                language=e.get("language", ""),
                content=e.get("content", ""),
                reason=e.get("reason") if not e["found"] else None,
                nearest_ids=e.get("nearest_ids") or [],
            )
            for e in entries
        ],
        note=_FETCH_NOT_FOUND_NOTE if any_file_changed else None,
        processing_ms=int((time.monotonic() - t0) * 1000),
    )


@router.get("/v1/status", response_model=StatusResponse)
async def status(request: Request) -> StatusResponse:
    t0 = time.monotonic()
    svc = _service(request)
    data = svc.status()
    return StatusResponse(
        **data,
        processing_ms=int((time.monotonic() - t0) * 1000),
    )


@router.get("/v1/call_counts")
async def get_call_counts(request: Request) -> dict:
    return _service(request).get_call_counts()


@router.delete("/v1/call_counts")
async def reset_call_counts(request: Request) -> dict:
    return {"reset": True, "previous_counts": _service(request).reset_call_counts()}


# ---------------------------------------------------------------------------
# REST API — L1 codebase map
# ---------------------------------------------------------------------------

@router.get("/v1/map")
async def map_workspace(request: Request) -> dict:
    svc = _service(request)
    _require_fully_ready(svc)
    return {"map": svc.get_map()}


@router.post("/v1/map", response_model=MapSaveResponse)
async def map_save(body: MapSaveRequest, request: Request) -> MapSaveResponse:
    t0 = time.monotonic()
    svc = _service(request)
    _require_fully_ready(svc)
    result = svc.save_map(body.summary, overwrite=body.overwrite)
    if not result["saved"]:
        return MapSaveResponse(
            message=(
                "A passport already exists for this workspace — not overwritten. "
                "Pass overwrite=true to replace it. Existing summary:\n\n"
                f"{result['existing_summary']}"
            ),
            processing_ms=int((time.monotonic() - t0) * 1000),
            saved=False,
        )
    return MapSaveResponse(
        message=f"Passport saved ({len(body.summary)} chars). Future vectr_map calls return this instantly.",
        processing_ms=int((time.monotonic() - t0) * 1000),
        saved=True,
    )


# ---------------------------------------------------------------------------
# REST API — L2 symbol graph
# ---------------------------------------------------------------------------

@router.post("/v1/locate", response_model=LocateResponse)
async def locate(body: LocateRequest, request: Request) -> LocateResponse:
    t0 = time.monotonic()
    svc = _service(request)
    _require_fully_ready(svc)
    if getattr(svc, "memory_only", False):
        from app.service import _MEMORY_ONLY_MSG
        raise HTTPException(status_code=503, detail={"error": "memory_only_mode", "detail": _MEMORY_ONLY_MSG})
    result = svc.locate_with_snippets(body.name, limit=body.limit)
    return LocateResponse(
        results=[
            SymbolResult(
                name=s.name,
                kind=s.kind,
                file_path=s.file_path,
                start_line=s.start_line,
                end_line=s.end_line,
                snippet=s.snippet,
            )
            for s in result.symbols
        ],
        formatted=svc.format_locate(result, body.name),
        processing_ms=int((time.monotonic() - t0) * 1000),
    )


@router.post("/v1/trace", response_model=TraceResponse)
async def trace(body: TraceRequest, request: Request) -> TraceResponse:
    t0 = time.monotonic()
    svc = _service(request)
    _require_fully_ready(svc)
    if getattr(svc, "memory_only", False):
        from app.service import _MEMORY_ONLY_MSG
        raise HTTPException(status_code=503, detail={"error": "memory_only_mode", "detail": _MEMORY_ONLY_MSG})
    trace_result = svc.trace_with_snippets(
        body.name, direction=body.direction, limit=body.limit,
        include_builtins=body.include_builtins,
    )
    return TraceResponse(
        formatted=svc.format_trace(trace_result, body.name),
        processing_ms=int((time.monotonic() - t0) * 1000),
    )


# ---------------------------------------------------------------------------
# REST API — Memory layer
# ---------------------------------------------------------------------------

@router.post("/v1/remember", response_model=RememberResponse)
async def remember(body: RememberRequest, request: Request) -> RememberResponse:
    t0 = time.monotonic()
    svc = _service(request)
    if getattr(svc, "search_only", False):
        from app.service import _SEARCH_ONLY_MSG
        raise HTTPException(status_code=503, detail={"error": "search_only_mode", "detail": _SEARCH_ONLY_MSG})
    try:
        # UPG-REMEMBER-MCP-LONG-PAYLOAD-PARSE-LOSS: resolve content_file
        # (mutually exclusive with inline content) before the write, using
        # the exact same shared resolver the MCP vectr_remember tool calls
        # — both surfaces enforce the identical rule. No stripping applied
        # here, matching this route's existing pass-through-unmodified
        # handling of an inline `content` body. extra_roots is passed
        # through so a multi-root instance accepts a content_file path
        # under ANY served root, not only the primary.
        ws_root = getattr(svc, "_workspace_root", "")
        if not isinstance(ws_root, str):
            ws_root = ""
        extra_roots = getattr(svc, "_extra_roots", [])
        if not isinstance(extra_roots, list):
            extra_roots = []
        extra_roots = [r for r in extra_roots if isinstance(r, str)]
        content = resolve_remember_content(
            ws_root, body.content, body.content_file, extra_roots=extra_roots,
        )
        outcome = await dispatch_chroma_async(
            svc,
            svc.remember_with_extras,
            content=content,
            tags=body.tags,
            priority=body.priority,
            session_id=body.session_id,
            kind=body.kind,
            title=body.title,
            agent=body.agent,
            triggers=body.triggers,
            provenance=body.provenance,
            scope=body.scope,
            anchors=body.anchors,
            supersedes=body.supersedes,
            contradicts=body.contradicts,
            user_quote=body.user_quote,
            pin=body.pin,
        )
    except ValueError as exc:
        # TRIGGER-ENGINE wave 1: malformed triggers, an unrecognised
        # provenance/scope combination (e.g. provenance='auto' +
        # kind='directive'), a `supersedes` target that does not exist, or
        # (UPG-MEMORY-STATE-MACHINE §4.2) a `contradicts` target that does
        # not exist — all caller input errors, never a server fault.
        raise HTTPException(status_code=422, detail={"error": "invalid_memory_object", "detail": str(exc)})
    note_id = outcome.note_id
    # UPG-MEM-PROVENANCE-USER-STATED / UPG-PROVENANCE-NEVER-RISES: a
    # user_quote that did not bind never fails the write — it is discarded
    # and the caller is told why, in the same response, recomputed with the
    # same pure function the store used (no second write path, no stored
    # rejection state). That recompute alone is blind to the harness-driven
    # auto-bind fallback (tried server-side whenever the explicit quote did
    # not bind), so the actual stored provenance is checked too — a note can
    # land at provenance='user-stated' with no body.user_quote at all, or
    # with one that itself failed to bind, when it was auto-bound instead.
    quote_suffix = ""
    _bound, reason = bind_user_quote(body.content, body.user_quote) if body.user_quote else ("", "")
    if _bound:
        quote_suffix = " Bound the user_quote: provenance='user-stated'."
    else:
        note_is_user_stated = False
        try:
            note_is_user_stated = svc.get_note(note_id).provenance == USER_STATED_PROVENANCE
        except Exception:
            note_is_user_stated = False
        if note_is_user_stated:
            quote_suffix = (
                " No user_quote bound explicitly, but this note auto-bound to the "
                "most recently captured user turn: provenance='user-stated'."
            )
        elif body.user_quote and reason:
            quote_suffix = f" {reason}"
    distilled = None
    if body.distilled_from:
        # Distillation write-back: the note write
        # already succeeded (note_id above); resolve the named arcs as a
        # second step, never blocking the note write on it.
        result = svc.resolve_arcs_distilled(body.distilled_from, note_id)
        distilled = ArcResolveResult(resolved=result["resolved"], unresolved=result["unresolved"])
    return RememberResponse(
        note_id=note_id,
        # CLI-form hint (UPG-CLI-RECALL-HINT): this route is the CLI's `vectr
        # remember`, not the MCP tool — the MCP dispatch builds its own
        # separate confirmation text in `vectr_remember(note_id=N)` form.
        message=f"Stored note #{note_id}. Recall with: vectr recall --id {note_id}{quote_suffix}",
        processing_ms=int((time.monotonic() - t0) * 1000),
        distilled=distilled,
        related=[RelatedNoteModel(**r) for r in outcome.related],
        revoked_related=[RevokedRelatedNoteModel(**r) for r in outcome.revoked_related],
        proxy_anchor_suggestions=outcome.proxy_anchor_suggestions,
    )


@router.post("/v1/recall", response_model=RecallResponse)
async def recall(body: RecallRequest, request: Request) -> RecallResponse:
    t0 = time.monotonic()
    svc = _service(request)
    if getattr(svc, "search_only", False):
        from app.service import _SEARCH_ONLY_MSG
        raise HTTPException(status_code=503, detail={"error": "search_only_mode", "detail": _SEARCH_ONLY_MSG})
    notes_text = await dispatch_chroma_async(
        svc,
        svc.recall,
        query=body.query,
        tags=body.tags,
        priority=body.priority,
        limit=body.limit,
        kind=body.kind,
        boot=body.boot,
        min_similarity=body.min_similarity,
        file_path=body.file_path,
        command=body.command,
        max_age_days=body.max_age_days,
        sort_by=body.sort_by,
        detail=body.detail,
        note_id=body.note_id,
        surface=body.surface,
        hook_event=body.hook_event,
        session_id=body.session_id,
        events=body.events,
    )
    return RecallResponse(
        notes=notes_text,
        processing_ms=int((time.monotonic() - t0) * 1000),
    )


@router.get("/v1/resume", response_model=ResumeResponse)
async def resume(
    request: Request,
    session_id: str | None = Query(default=None),
) -> ResumeResponse:
    """UPG-RESUME-SURFACE: deterministic 'pick up where you left off' — the
    CLI/MCP-equivalent view of the same selection SessionStart boot injection
    already replays (see VectrService.resume / WorkingContextStore.resume_state).
    GET, no body: this reads existing state, it stores nothing."""
    t0 = time.monotonic()
    svc = _service(request)
    if getattr(svc, "search_only", False):
        from app.service import _SEARCH_ONLY_MSG
        raise HTTPException(status_code=503, detail={"error": "search_only_mode", "detail": _SEARCH_ONLY_MSG})
    data = svc.resume(session_id=session_id, surface="cli")
    return ResumeResponse(
        **data,
        processing_ms=int((time.monotonic() - t0) * 1000),
    )


@router.post("/v1/promote", response_model=PromoteResponse)
async def promote(body: PromoteRequest, request: Request) -> PromoteResponse:
    """Explicit provenance promotion (TRIGGER-ENGINE wave 1,
    bm2-design-skeleton.md §5): auto -> agent -> human, one step at a time.
    Provenance is immutable at write; this is the one sanctioned way to raise
    it afterward.

    This is the user-side promotion surface — a CLI/UI a person operates
    calls this route directly, so it supports the full one-step chain
    including the final agent -> human step. The MCP tool (`vectr_promote`,
    the AI's own surface) deliberately does NOT expose that last step: an
    agent deciding on its own that a person has endorsed a note would reopen
    the trust-inversion hole §5 closes structurally, so it only allows
    auto -> agent and returns a tool error for 'human'."""
    t0 = time.monotonic()
    svc = _service(request)
    if getattr(svc, "search_only", False):
        from app.service import _SEARCH_ONLY_MSG
        raise HTTPException(status_code=503, detail={"error": "search_only_mode", "detail": _SEARCH_ONLY_MSG})
    try:
        promoted = svc.promote_note(body.note_id, body.to)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"error": "invalid_promotion", "detail": str(exc)})
    if not promoted:
        raise HTTPException(status_code=404, detail={"error": "note_not_found", "detail": f"Note #{body.note_id} not found."})
    return PromoteResponse(
        note_id=body.note_id,
        provenance=body.to,
        processing_ms=int((time.monotonic() - t0) * 1000),
    )


@router.post("/v1/pin", response_model=PinResponse)
async def pin(body: PinRequest, request: Request) -> PinResponse:
    """Explicit Tier 0 membership toggle (UPG-RECALL-MISS-FLOOR part (b)):
    set or clear a note's `pinned` flag. Pinned notes join kind='directive'
    notes in the unconditional-injection tier of every
    vectr_recall(query=...) call, up to the configured Tier 0 bound
    (recall_floor.tier0_max_notes) — never scored against the query, never
    inferred. Idempotent: pinning an already-pinned note (or unpinning an
    already-unpinned one) still returns 200."""
    t0 = time.monotonic()
    svc = _service(request)
    if getattr(svc, "search_only", False):
        from app.service import _SEARCH_ONLY_MSG
        raise HTTPException(status_code=503, detail={"error": "search_only_mode", "detail": _SEARCH_ONLY_MSG})
    changed = svc.pin_note(body.note_id, body.pinned)
    if not changed:
        raise HTTPException(status_code=404, detail={"error": "note_not_found", "detail": f"Note #{body.note_id} not found."})
    verb = "pinned" if body.pinned else "unpinned"
    return PinResponse(
        note_id=body.note_id,
        pinned=body.pinned,
        message=f"Note #{body.note_id} {verb}.",
        processing_ms=int((time.monotonic() - t0) * 1000),
    )


@router.post("/v1/anchor", response_model=AnchorResponse)
async def anchor(body: AnchorRequest, request: Request) -> AnchorResponse:
    """Attach anchor paths to an existing note post-write
    (UPG-ANCHOR-ATTACH, `vectr_anchor`): the single-call replacement for the
    re-store-with-supersedes workaround. Anchors are delivery-shaping
    metadata — check_staleness() compares their write-time hashes against
    the live files on every check, so drift surfaces as a stale flag, never
    as a judgment that the note is wrong. Idempotent: paths the note is
    already anchored to are reported in `already_present`, not duplicated."""
    t0 = time.monotonic()
    svc = _service(request)
    if getattr(svc, "search_only", False):
        from app.service import _SEARCH_ONLY_MSG
        raise HTTPException(status_code=503, detail={"error": "search_only_mode", "detail": _SEARCH_ONLY_MSG})
    result = svc.attach_anchors(body.note_id, body.anchors, session_id=body.session_id)
    if result is None:
        raise HTTPException(status_code=404, detail={"error": "note_not_found", "detail": f"Note #{body.note_id} not found."})
    return AnchorResponse(
        note_id=body.note_id,
        attached=result["attached"],
        already_present=result["already_present"],
        message=f"Attached {len(result['attached'])} anchor(s) to note #{body.note_id}.",
        processing_ms=int((time.monotonic() - t0) * 1000),
    )


@router.post("/v1/unanchor", response_model=UnanchorResponse)
async def unanchor(body: UnanchorRequest, request: Request) -> UnanchorResponse:
    """Remove anchor paths from an existing note post-write
    (UPG-ANCHOR-DETACH, `vectr_unanchor`): the inverse of POST /v1/anchor.
    Anchors are delivery-shaping metadata — a removed anchor simply stops
    being a candidate in the next check_staleness() check, never a claim
    that the note is wrong. Idempotent: paths the note was never anchored
    to are reported in `not_present`, not treated as failures."""
    t0 = time.monotonic()
    svc = _service(request)
    if getattr(svc, "search_only", False):
        from app.service import _SEARCH_ONLY_MSG
        raise HTTPException(status_code=503, detail={"error": "search_only_mode", "detail": _SEARCH_ONLY_MSG})
    result = svc.detach_anchors(body.note_id, body.anchors)
    if result is None:
        raise HTTPException(status_code=404, detail={"error": "note_not_found", "detail": f"Note #{body.note_id} not found."})
    return UnanchorResponse(
        note_id=body.note_id,
        removed=result["removed"],
        not_present=result["not_present"],
        message=f"Removed {len(result['removed'])} anchor(s) from note #{body.note_id}.",
        processing_ms=int((time.monotonic() - t0) * 1000),
    )


@router.post("/v1/revoke", response_model=RevokeResponse)
async def revoke(body: RevokeRequest, request: Request) -> RevokeResponse:
    """Explicit revocation (UPG-MEMORY-STATE-MACHINE §4.2): caller asserts
    the target note is wrong. Zero-inference — vectr never decides a
    revocation itself; this route only appends the event and returns. The
    note is not deleted: it stays fire/recall-eligible afterward, rendered
    in anti-memory (deterrent) form on every surface until `reinstated`
    (see `/v1/reinstate`) or hard-deleted via `/v1/forget`.

    Person-operated surface: `actor` defaults to "agent" but accepts
    "human" explicitly (never "system" — that actor is reserved for the
    deterministic anchor-drift transition in `check_staleness()`)."""
    t0 = time.monotonic()
    svc = _service(request)
    if getattr(svc, "search_only", False):
        from app.service import _SEARCH_ONLY_MSG
        raise HTTPException(status_code=503, detail={"error": "search_only_mode", "detail": _SEARCH_ONLY_MSG})
    try:
        revoked = svc.revoke_note(body.note_id, body.reason, actor=body.actor)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"error": "invalid_revocation", "detail": str(exc)})
    if not revoked:
        raise HTTPException(status_code=404, detail={"error": "note_not_found", "detail": f"Note #{body.note_id} not found."})
    return RevokeResponse(
        note_id=body.note_id,
        message=f"Note #{body.note_id} revoked.",
        processing_ms=int((time.monotonic() - t0) * 1000),
    )


@router.post("/v1/reinstate", response_model=ReinstateResponse)
async def reinstate(body: ReinstateRequest, request: Request) -> ReinstateResponse:
    """Revert-of-revert (UPG-MEMORY-STATE-MACHINE §4.2): always legal.
    Reverses a prior `/v1/revoke` (or a `contradicts=` write), a TTL
    expiry, or a supersession (`supersedes=` at write time or
    `/v1/supersede` post-hoc — the tombstone columns are cleared so the
    note returns to default recall/fire): the note returns to active state
    and its raw content, not the anti-memory deterrent block, is rendered
    again on every surface."""
    t0 = time.monotonic()
    svc = _service(request)
    if getattr(svc, "search_only", False):
        from app.service import _SEARCH_ONLY_MSG
        raise HTTPException(status_code=503, detail={"error": "search_only_mode", "detail": _SEARCH_ONLY_MSG})
    try:
        reinstated = svc.reinstate_note(body.note_id, actor=body.actor, reason=body.reason)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"error": "invalid_reinstatement", "detail": str(exc)})
    if not reinstated:
        raise HTTPException(status_code=404, detail={"error": "note_not_found", "detail": f"Note #{body.note_id} not found."})
    return ReinstateResponse(
        note_id=body.note_id,
        message=f"Note #{body.note_id} reinstated.",
        processing_ms=int((time.monotonic() - t0) * 1000),
    )


@router.post("/v1/supersede", response_model=SupersedeResponse)
async def supersede(body: SupersedeRequest, request: Request) -> SupersedeResponse:
    """Post-hoc supersession (UPG-NOTE-RETIRE-POST-HOC): retire an
    already-stored note as superseded — the transition write-time
    `supersedes=` cannot reach, for a replacement that was written without
    it. Zero-inference — vectr never decides a retirement itself; this
    route only appends the event and returns. The note is not deleted and
    not revoked: it stops being a default recall/fire candidate and renders
    with the factual "[superseded ...]" badge (never the anti-memory
    deterrent — that framing is reserved for /v1/revoke's "this note was
    WRONG"). Reversible via `/v1/reinstate`.

    Person-operated surface: `actor` defaults to "agent" but accepts
    "human" explicitly (never "system"). An agent-actor call may not retire
    a human-provenance note; only actor="human" may."""
    t0 = time.monotonic()
    svc = _service(request)
    if getattr(svc, "search_only", False):
        from app.service import _SEARCH_ONLY_MSG
        raise HTTPException(status_code=503, detail={"error": "search_only_mode", "detail": _SEARCH_ONLY_MSG})
    try:
        retired = svc.supersede_note(
            body.note_id, superseded_by=body.superseded_by, reason=body.reason, actor=body.actor,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"error": "invalid_supersedence", "detail": str(exc)})
    if not retired:
        raise HTTPException(status_code=404, detail={"error": "note_not_found", "detail": f"Note #{body.note_id} not found."})
    return SupersedeResponse(
        note_id=body.note_id,
        message=f"Note #{body.note_id} retired as superseded.",
        processing_ms=int((time.monotonic() - t0) * 1000),
    )


@router.post("/v1/commit-note", response_model=CommitNoteResponse)
async def commit_note(body: CommitNoteRequest, request: Request) -> CommitNoteResponse:
    """Git post-commit hook write path (UPG-COMMIT-MEMORY-HOOK): one
    deterministic, zero-inference working-memory note per commit, capturing
    the commit's identity, touched files, and active task context — git
    records WHAT changed, this records the context for WHY. Called only by
    `vectr hook post-commit` (main.cmd_hook), never by the editor's LLM."""
    t0 = time.monotonic()
    svc = _service(request)
    if getattr(svc, "search_only", False):
        from app.service import _SEARCH_ONLY_MSG
        raise HTTPException(status_code=503, detail={"error": "search_only_mode", "detail": _SEARCH_ONLY_MSG})
    note_id = await dispatch_chroma_async(
        svc,
        svc.record_commit_note,
        sha=body.sha, subject=body.subject, branch=body.branch, files=body.files,
    )
    return CommitNoteResponse(
        note_id=note_id,
        processing_ms=int((time.monotonic() - t0) * 1000),
    )


@router.post("/v1/proactive", response_model=ProactiveResponse)
async def proactive(body: ProactiveRequest, request: Request) -> ProactiveResponse:
    """Packed proactive context for an assembled window (UPG-PRO-7 subset).

    Hook-facing / proxy-facing: never errors the caller. When proactive is
    disabled, the memory layer is absent, or nothing clears the floor + budget,
    it returns an empty result so the caller forwards the request unmodified.
    """
    t0 = time.monotonic()
    svc = _service(request)
    if getattr(svc, "search_only", False):
        # No working-memory layer to draw from; inject nothing (never 503 here).
        return ProactiveResponse(processing_ms=int((time.monotonic() - t0) * 1000))
    try:
        result = await dispatch_chroma_async(
            svc,
            svc.proactive_context,
            text=body.text,
            file_paths=body.file_paths,
            symbols=body.symbols,
            edited_file_paths=body.edited_file_paths,
            session_id=body.session_id,
            channel=body.channel,
            structural_only=body.structural_only,
            defer_charge=body.defer_charge,
            confirm_token=body.confirm_token,
        )
    except Exception:
        result = {"context": "", "item_count": 0, "anchor_ids": [], "scores": []}
    return ProactiveResponse(
        context=result["context"],
        item_count=result["item_count"],
        anchor_ids=result["anchor_ids"],
        scores=result["scores"],
        # `.get` because the deferred-charge fields are additive: a service
        # that predates them (or a test double standing in for one) returns the
        # older 4-key result, and must keep working rather than KeyError into
        # the caller's request path.
        delivery_token=result.get("delivery_token", ""),
        charge_deferred=result.get("charge_deferred", False),
        processing_ms=int((time.monotonic() - t0) * 1000),
    )


@router.post("/v1/snapshot", response_model=SnapshotResponse)
async def snapshot(body: SnapshotRequest, request: Request) -> SnapshotResponse:
    t0 = time.monotonic()
    svc = _service(request)
    if getattr(svc, "search_only", False):
        from app.service import _SEARCH_ONLY_MSG
        raise HTTPException(status_code=503, detail={"error": "search_only_mode", "detail": _SEARCH_ONLY_MSG})
    snapshot_id = svc.snapshot_session(label=body.label, session_id=body.session_id)
    return SnapshotResponse(
        snapshot_id=snapshot_id,
        label=body.label,
        processing_ms=int((time.monotonic() - t0) * 1000),
    )


@router.post("/v1/trigger/reset", response_model=TriggerResetResponse)
async def trigger_reset(body: TriggerResetRequest, request: Request) -> TriggerResetResponse:
    """Reset one session's trigger-engine state (TRIGGER-ENGINE wave 2a) —
    called by the PreCompact hook right before the SessionStart hook that
    follows a /compact re-delivers the boot set. Never 503s even in
    search-only mode: the per-session ledger lives in `VectrService` itself,
    not the working-memory store, so there is nothing to gate on."""
    t0 = time.monotonic()
    svc = _service(request)
    svc.reset_trigger_ledger(body.session_id)
    return TriggerResetResponse(
        reset=True,
        processing_ms=int((time.monotonic() - t0) * 1000),
    )


@router.post("/v1/memory/clear")
async def memory_clear(request: Request) -> dict:
    """Delete all working-memory notes and snapshots for the current workspace."""
    svc = _service(request)
    if getattr(svc, "search_only", False):
        from app.service import _SEARCH_ONLY_MSG
        raise HTTPException(status_code=503, detail={"error": "search_only_mode", "detail": _SEARCH_ONLY_MSG})
    deleted = await dispatch_chroma_async(svc, svc.forget_all)
    return {"deleted": deleted}


@router.post("/v1/forget")
async def forget(body: ForgetRequest, request: Request) -> dict:
    """Delete one note by id, or all notes when all=true. No arguments deletes nothing."""
    svc = _service(request)
    if getattr(svc, "search_only", False):
        from app.service import _SEARCH_ONLY_MSG
        raise HTTPException(status_code=503, detail={"error": "search_only_mode", "detail": _SEARCH_ONLY_MSG})
    if body.note_id is not None:
        deleted = await dispatch_chroma_async(svc, svc.forget_note, body.note_id)
        return {"deleted": 1 if deleted else 0, "note_id": body.note_id, "found": deleted}
    if body.all:
        return {"deleted": await dispatch_chroma_async(svc, svc.forget_all)}
    raise HTTPException(
        status_code=422,
        detail="Pass note_id to delete one note, or all=true to clear every note.",
    )


@router.post("/v1/episode", response_model=EpisodeResponse)
async def episode(body: EpisodeRequest, request: Request) -> EpisodeResponse:
    """PostToolUse hook write path (episode capture): one deterministic,
    zero-inference row per Bash/Edit/Write/MultiEdit tool call. Called only
    by `vectr hook
    post-tool-use`'s detached worker, never by the editor's LLM directly.

    Quarantined by construction: this route is the ONLY write path into the
    `episodes` table (via `EpisodeStore`, a module neither `agent/searcher.py`
    nor `agent/working_context_store` ever imports) — episode rows can never
    surface in `vectr_search`/`vectr_recall` results or any hook-injected
    context. No embedding happens here or anywhere on this path."""
    t0 = time.monotonic()
    svc = _service(request)
    if getattr(svc, "search_only", False):
        from app.service import _SEARCH_ONLY_MSG
        raise HTTPException(status_code=503, detail={"error": "search_only_mode", "detail": _SEARCH_ONLY_MSG})
    episode_id = svc.record_episode(
        session_id=body.session_id,
        ts=body.ts,
        cwd=body.cwd,
        tool=body.tool,
        command=body.command,
        description=body.description,
        file_path=body.file_path,
        rc=body.rc,
        is_error=body.is_error,
        interrupted=body.interrupted,
        harness_success=body.harness_success,
        stdout_tail=body.stdout_tail,
        stderr_tail=body.stderr_tail,
    )
    return EpisodeResponse(
        episode_id=episode_id,
        processing_ms=int((time.monotonic() - t0) * 1000),
    )


@router.get("/v1/episodes", response_model=EpisodesResponse)
async def episodes(
    request: Request,
    session_id: str | None = Query(default=None),
    arc_id: int | None = Query(default=None),
    since_ts: float | None = Query(default=None),
    until_ts: float | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
) -> EpisodesResponse:
    """The only bulk reader of the `episodes` table besides the aggregate
    counts folded into `vectr_status` — deliberately not part of the
    vectr_search/vectr_recall surface."""
    t0 = time.monotonic()
    svc = _service(request)
    if getattr(svc, "search_only", False):
        from app.service import _SEARCH_ONLY_MSG
        raise HTTPException(status_code=503, detail={"error": "search_only_mode", "detail": _SEARCH_ONLY_MSG})
    rows = svc.list_episodes(
        session_id=session_id, arc_id=arc_id, since_ts=since_ts, until_ts=until_ts, limit=limit,
    )
    return EpisodesResponse(
        episodes=[EpisodeRecord(**row) for row in rows],
        processing_ms=int((time.monotonic() - t0) * 1000),
    )


@router.get("/v1/arcs", response_model=ArcsResponse)
async def arcs(
    request: Request,
    status: str = Query(default="pending", description="pending | resolved | all"),
    limit: int | None = Query(
        default=None, ge=1, le=1000,
        description="defaults to episodes.distill_max_arcs_rendered (config.yaml) when omitted",
    ),
) -> ArcsResponse:
    """Read surface for arc distillation: arc rows joined with their
    episodes' summary fields — id/ts/cwd/confidence, the failure chain,
    the mutation diff, the resolving success episode — plus
    `total_pending`. Rendered facts only: no advice, no suggested kinds
    (that judgment lives in static tool-surface text, never in generated
    data). An omitted `limit` resolves to `episodes.distill_max_arcs_rendered`
    — the same cap the MCP `vectr_distill`
    render uses — read from the config module at request time so a config
    reload or test override is honored rather than baked in at import time."""
    t0 = time.monotonic()
    svc = _service(request)
    if getattr(svc, "search_only", False):
        from app.service import _SEARCH_ONLY_MSG
        raise HTTPException(status_code=503, detail={"error": "search_only_mode", "detail": _SEARCH_ONLY_MSG})
    if status not in ("pending", "resolved", "all"):
        raise HTTPException(
            status_code=422,
            detail={"error": "invalid_status", "detail": "status must be pending, resolved, or all"},
        )
    resolved_limit = limit if limit is not None else config.EPISODES_DISTILL_MAX_ARCS_RENDERED
    rows = svc.list_arcs(status=status, limit=resolved_limit)
    return ArcsResponse(
        arcs=[ArcRecord(**row) for row in rows],
        total_pending=svc.count_arcs_pending_distill(),
        processing_ms=int((time.monotonic() - t0) * 1000),
    )


@router.post("/v1/arcs/dismiss", response_model=ArcsDismissResponse)
async def arcs_dismiss(body: ArcsDismissRequest, request: Request) -> ArcsDismissResponse:
    """Write-back verdict: dismiss —
    sets `distilled_at`/`dismissed_reason` on each named arc. Idempotent:
    unknown/already-resolved ids come back in `unresolved`, never an
    error."""
    t0 = time.monotonic()
    svc = _service(request)
    if getattr(svc, "search_only", False):
        from app.service import _SEARCH_ONLY_MSG
        raise HTTPException(status_code=503, detail={"error": "search_only_mode", "detail": _SEARCH_ONLY_MSG})
    result = svc.resolve_arcs_dismissed(body.arc_ids, body.reason)
    return ArcsDismissResponse(
        resolved=result["resolved"],
        unresolved=result["unresolved"],
        processing_ms=int((time.monotonic() - t0) * 1000),
    )


@router.get("/v1/boundary/precompact", response_model=BoundaryPrecompactResponse)
async def boundary_precompact(request: Request) -> BoundaryPrecompactResponse:
    """PreCompact boundary-preservation surface: the text the pre-compact
    hook branch (main.py / agent/hook_cli.py) prints verbatim as PLAIN TEXT
    for the summarizer to read — never as a `hookSpecificOutput` JSON
    envelope (the PreCompact hook's stdout is not parsed as one). Server-
    rendered (`VectrService._boundary_precompact_text`) so the product copy
    has one home; `text` is "" (still HTTP 200) when
    `episodes.boundary_precompact_enabled` is false."""
    svc = _service(request)
    if getattr(svc, "search_only", False):
        from app.service import _SEARCH_ONLY_MSG
        raise HTTPException(status_code=503, detail={"error": "search_only_mode", "detail": _SEARCH_ONLY_MSG})
    result = svc.boundary_precompact()
    return BoundaryPrecompactResponse(
        text=result["text"],
        arcs_pending=result["arcs_pending"],
    )


@router.get("/v1/evict-hint")
async def evict_hint(request: Request) -> dict:
    svc = _service(request)
    _require_fully_ready(svc)
    # UPG-7.2: an explicit GET is a deliberate ask — on-demand, eviction-focused
    # framing, distinct from the gated auto-footer's remember alarm.
    hint = svc.eviction_hint(on_demand=True)
    return {"hint": hint or "No retrieved chunks to evict.", "should_evict": svc.should_evict()}


# ---------------------------------------------------------------------------
# MCP protocol
# ---------------------------------------------------------------------------

@router.get("/mcp")
async def mcp_info() -> dict:
    return MCP_SERVER_INFO


@router.post("/mcp")
async def mcp_jsonrpc(request: Request, response: Response, body: dict = Body(...)) -> dict:
    """Standard MCP JSON-RPC transport — handles initialize / tools/list / tools/call / ping.

    Session identity (UPG-MCP-SESSION-ID-HANDSHAKE): per the MCP streamable-HTTP
    transport, the server assigns a session id by returning it in the
    `Mcp-Session-Id` response header on `initialize`; a compliant client echoes
    that same header on every subsequent request. We honor that header first
    (case-insensitive per HTTP, handled by Starlette), then fall back to
    vectr-invented mechanisms (`_meta.sessionId`, `X-Session-ID`) for callers —
    tests, curl, older shims — that predate the handshake.
    """
    jsonrpc_id = body.get("id")
    method = body.get("method", "")
    params = body.get("params") or {}

    def _ok(result: dict) -> dict:
        return {"jsonrpc": "2.0", "id": jsonrpc_id, "result": result}

    def _err(code: int, message: str) -> dict:
        return {"jsonrpc": "2.0", "id": jsonrpc_id, "error": {"code": code, "message": message}}

    if method == "initialize":
        response.headers["Mcp-Session-Id"] = uuid.uuid4().hex
        return _ok({
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": MCP_SERVER_INFO["name"], "version": MCP_SERVER_INFO["version"]},
        })

    if method in ("notifications/initialized", "initialized"):
        return {}  # notification — no response body

    if method == "ping":
        return _ok({})

    # Resolve session id: server-assigned handshake header first, then the
    # vectr-invented fallbacks kept for backwards compatibility.
    meta = params.get("_meta") or body.get("_meta") or {}
    session_id = (
        request.headers.get("Mcp-Session-Id")
        or meta.get("sessionId")
        or request.headers.get("X-Session-ID")
        or None
    )

    if method == "tools/list":
        svc = _service(request)
        return _ok(handle_tools_list(session_id=session_id, service=svc))

    if method == "tools/call":
        tool_name = params.get("name")
        arguments  = params.get("arguments") or {}
        if not tool_name:
            return _err(-32602, "Missing required param: name")
        svc = _service(request)
        still_initialising = _mcp_tool_still_initialising(svc, tool_name)
        if still_initialising is not None:
            return _ok(still_initialising)
        client_label = request.headers.get("X-Vectr-Client", "") or ""
        # UPG-EMBED-THREAD-CONTENTION: a tool call may embed a query
        # (search/recall) — CPU-bound work that must not run on the event
        # loop thread, the same reasoning /v1/index already applies to a
        # full-workspace index. Off the event loop, a CPU-saturated moment
        # degrades this call to slow instead of black-holing every other
        # request the daemon serves concurrently.
        result = await run_in_threadpool(
            handle_tools_call, tool_name, arguments, svc,
            session_id=session_id, client_label=client_label,
        )
        return _ok(result)

    return _err(-32601, f"Method not found: {method}")


@router.post("/mcp/tools/list")
async def mcp_tools_list(request: Request, body: dict = Body(default={})) -> dict:
    svc = _service(request)
    session_id = request.headers.get("X-Session-ID") or None
    return handle_tools_list(session_id=session_id, service=svc)


@router.post("/mcp/tools/call")
async def mcp_tools_call(request: Request, body: dict = Body(...)) -> dict:
    tool_name = body.get("name") or body.get("tool")
    arguments = body.get("arguments") or body.get("input") or {}
    if not tool_name:
        raise HTTPException(
            status_code=400,
            detail={"error": "missing_tool_name", "detail": "Request body must include 'name'"},
        )
    svc = _service(request)
    still_initialising = _mcp_tool_still_initialising(svc, tool_name)
    if still_initialising is not None:
        return still_initialising
    session_id = request.headers.get("X-Session-ID") or None
    client_label = request.headers.get("X-Vectr-Client", "") or ""
    # UPG-EMBED-THREAD-CONTENTION: see the /mcp tools/call branch above —
    # same legacy-route dispatch, same off-event-loop requirement.
    return await run_in_threadpool(
        handle_tools_call, tool_name, arguments, svc,
        session_id=session_id, client_label=client_label,
    )
