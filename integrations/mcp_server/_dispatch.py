"""MCP tool dispatch — handle_tools_call and handle_tools_list."""
from __future__ import annotations

import os
from typing import Any

from agent.config import (
    SYMBOL_NAME_PARAM_ALIASES,
    MEMORY_HYGIENE_STALE_TASK_WARN_COUNT,
    MEMORY_HYGIENE_STALE_TASK_WARN_AGE_DAYS,
    SCORE_ORDER_EXPLAIN_ENABLED,
    SCORE_ORDER_EXPLAIN_MIN_HEADROOM_RATIO,
    POINTER_MODE_RETAIN_ENABLED,
    POINTER_MODE_RETAIN_MIN_RELEVANCE,
    POINTER_MODE_RETAIN_EXCERPT_LINES,
    POINTER_MODE_RETAIN_LABEL,
    POINTER_MODE_RETAIN_RANK1_ALWAYS,
    POINTER_MODE_RETAIN_RANK1_LABEL,
    EPISODES_DISTILL_MAX_ARCS_RENDERED,
    EPISODES_DISTILL_RENDER_TOKEN_CAP,
)
from agent.chroma_dispatch import dispatch_chroma_sync
from agent.render_paths import workspace_relpath
from agent.working_context_store import SORT_BY_VALUES, USER_STATED_PROVENANCE, bind_user_quote, resolve_remember_content


def _service_ws_root(service) -> str:
    """The daemon's workspace root as a plain string (UPG-RELATIVE-PATH-RENDER),
    or "" when unavailable/non-str — so a test MagicMock's auto-attribute never
    leaks into a rendered path or header."""
    root = getattr(service, "_workspace_root", "")
    return root if isinstance(root, str) else ""


def _service_extra_roots(service) -> list[str]:
    """The daemon's additional served workspace roots (multi-root instance)
    as a plain list of strings, or [] when unavailable/malformed — same
    mock-safety rationale as `_service_ws_root`: a test MagicMock's
    auto-attribute must never leak into a containment check."""
    roots = getattr(service, "_extra_roots", [])
    if not isinstance(roots, list):
        return []
    return [r for r in roots if isinstance(r, str)]
from integrations.mcp_server._schemas import (
    _EXPLORATION_TOOLS,
    _MEMORY_WRITE_TOOLS,
    _MEMORY_TOOLS,
    _UTILITY_TOOLS,
    _DISTILL_RULES_TEXT,
    MCP_TOOLS,
)
from integrations.mcp_server._session import (
    _memory_enabled_sessions,
    _session_calls_since_save,
    _increment_calls_since_save,
    _reset_calls_since_save,
    _should_nudge_remember,
    _remember_nudge_text,
    enable_memory_for_session,
    is_memory_enabled,
)


def _symbol_name_arg(arguments: dict) -> str:
    """Read the symbol-name argument for vectr_locate/vectr_trace, accepting
    SYMBOL_NAME_PARAM_ALIASES as drop-in aliases for the schema's "name" key
    (F40-class param ergonomics, UPG-TRACE-GRAPH-INCOMPLETE): a tool
    description that reads as a positional example trains a caller to guess
    a different key, and a required-arg error that never says which key is
    correct trains them to abandon the tool rather than retry. "name" wins
    when present alongside an alias."""
    name = arguments.get("name")
    if name:
        return str(name).strip()
    for alias in SYMBOL_NAME_PARAM_ALIASES:
        value = arguments.get(alias)
        if value:
            return str(value).strip()
    return ""


def handle_tools_list(session_id: str | None = None, service: Any = None) -> dict:
    """Return tools appropriate for this session.

    Always shown: the exploration tools (vectr_search, vectr_fetch,
    vectr_status, vectr_map, vectr_map_save, vectr_locate, vectr_trace),
    the memory-write tools (vectr_remember, vectr_evict_hint,
    vectr_distill), and the utility tool (vectr_ingest_traces). Gated on
    notes existing: every _MEMORY_TOOLS member (vectr_recall,
    vectr_snapshot, vectr_snapshot_list, vectr_resume, vectr_forget,
    vectr_promote, vectr_revoke, vectr_reinstate, vectr_supersede,
    vectr_pin, vectr_anchor, vectr_unanchor). The grouped lists above
    are kept in sync with _EXPLORATION_TOOLS / _MEMORY_WRITE_TOOLS /
    _UTILITY_TOOLS / _MEMORY_TOOLS in integrations/mcp_server/_schemas.py
    — re-derive them from those constants when changing tool membership,
    rather than hand-editing.
    """
    # Hosted/registry deployments (e.g. a catalog's containerised inspector)
    # start with an empty note store but must still advertise the complete
    # tool surface — the memory read tools would otherwise stay hidden until
    # the first note exists. Default-off; editor sessions are unaffected.
    if os.getenv("VECTR_MCP_ALL_TOOLS", "") == "1":
        return {
            "tools": _EXPLORATION_TOOLS + _MEMORY_WRITE_TOOLS + _UTILITY_TOOLS + _MEMORY_TOOLS
        }

    # Pre-enable memory read tools if notes already exist
    if session_id and service and not is_memory_enabled(session_id):
        try:
            notes_count = service.count_notes() if hasattr(service, "count_notes") else 0
            if notes_count > 0:
                enable_memory_for_session(session_id)
        except Exception:
            pass

    base = _EXPLORATION_TOOLS + _MEMORY_WRITE_TOOLS + _UTILITY_TOOLS
    if is_memory_enabled(session_id):
        return {"tools": base + _MEMORY_TOOLS}
    return {"tools": base}


def handle_tools_call(
    tool_name: str,
    arguments: dict,
    service: Any,
    session_id: str | None = None,
    client_label: str = "",
) -> dict:
    """Dispatch an MCP tool call. `service` is the VectrService instance.

    `client_label` (team mode) is the connecting client's attribution label,
    read from the `X-Vectr-Client` request header. It becomes the default note
    author when a `vectr_remember` call does not declare its own `agent`, and
    attributes every audit event this call triggers (search/index/recall too).
    """
    # Attribute audit events (opt-in audit log) to the connecting client. Set
    # unconditionally — including to "" — so a subsequent call in the same task
    # never inherits a previous call's label. Task-local via ContextVar, so
    # concurrent clients never cross-attribute.
    from agent.working_context_store import set_audit_client
    set_audit_client(client_label)

    # Count every tool call — used by the tool-call-count eviction trigger.
    # Reads the calling session's own advisor (UPG-EVICT-SESSION-SCOPE).
    try:
        service._advisor_for(session_id).increment_tool_call()
    except Exception:
        pass

    # Count per-tool calls for benchmark metrics (accurate across parent + sub-agents)
    try:
        service.increment_call_count(tool_name)
    except Exception:
        pass

    # Increment turn-count nudge counter for all tool calls
    _increment_calls_since_save(session_id)

    # Count retrieval-specific calls (search/locate/trace) for the retrieval-count trigger
    if tool_name in ("vectr_search", "vectr_locate", "vectr_trace"):
        try:
            service._advisor_for(session_id).increment_retrieval_call()
        except Exception:
            pass

    # ---- vectr_search ----
    if tool_name == "vectr_search":
        query = arguments.get("query", "")
        n_results = min(int(arguments.get("n_results", 5)), 50)
        language = arguments.get("language") or None

        if not query:
            return _mcp_error("query is required")

        # Memory-only mode: search is disabled for this workspace
        if getattr(service, "memory_only", False):
            from app.service import _MEMORY_ONLY_MSG
            return {"content": [{"type": "text", "text": _MEMORY_ONLY_MSG}], "isError": False}

        if service.total_chunks == 0:
            return {
                "content": [{"type": "text", "text": "Vectr is still indexing the codebase. Try again in a few moments."}],
                "isError": False,
            }

        # Dispatched off the request event loop through the service's
        # dedicated executor — `handle_tools_call` itself already runs
        # off-loop (its own callers hop it onto a thread pool), but as a
        # plain synchronous function it must block-wait for the result
        # rather than `await` it.
        results, query_ms = dispatch_chroma_sync(
            service, service.search, query, n_results=n_results, language=language
        )

        # Record the rendered chunks into the calling session's eviction
        # advisor (UPG-EVICT-SESSION-SCOPE) — the single recording site for
        # vectr_search, at render time, against the calling session only.
        try:
            service.record_results(results, session_id=session_id)
        except Exception:
            pass

        sections: list[str] = []

        # UPG-NOTFOUND-FLOOR (F46/F52), extended by UPG-SCORE-DISPLAY-FLAT: lead
        # with a low-confidence banner when either (a) the query names a
        # concept with no lexical anchor anywhere in the indexed corpus, or
        # (b) the top result's own absolute relevance score (below) is itself
        # below the configured floor. The per-result score is now an absolute
        # query-doc relevance value rather than a per-query rank-derived one,
        # so a caller can also read it directly — this banner remains the
        # explicit, hard-to-miss signal that the whole result set may be a
        # weak/unrelated guess. Every result is still LISTED below it (never
        # dropped) but its body is not, by default: pointer mode
        # (UPG-LOWCONF-OUTPUT-SLIM / UPG-FLOOR-SLIM-PAYLOAD) renders each
        # result as a file:line pointer rather than full content, expandable
        # on demand with vectr_fetch(ids=[...]); a result whose own
        # ce_relevance individually clears the retention floor keeps a
        # bounded excerpt instead of a bare pointer (UPG-POINTER-MODE-
        # UNIFORM-STRIP).
        low_conf = getattr(results, "low_confidence", False)
        if low_conf:
            from agent.config import NOTFOUND_FLOOR_BANNER
            sections.append(f"─── Low confidence ───\n{NOTFOUND_FLOOR_BANNER}")

        # UPG-3.1: a language filter that matched nothing because that language
        # isn't indexed (not merely a query miss) should tell the caller what IS
        # indexable, rather than silently returning empty.
        if language and not results:
            indexed = service.indexed_languages()
            if language.lower() not in {l.lower() for l in indexed}:
                avail = ", ".join(indexed) if indexed else "(none yet)"
                sections.append(
                    f"─── No results: language={language!r} is not indexed ───\n"
                    f"Indexed languages: {avail}.\n"
                    f"Re-run without the language filter, or use one of the above."
                )

        # UPG-RELATIVE-PATH-RENDER: the daemon's own workspace root — rendered
        # once in the search header, then stripped from every path/id below.
        ws_root = _service_ws_root(service)

        # L3 chunks
        sections.append(_format_search_results(results, query, query_ms, service.total_chunks, ws_root))

        # UPG-QUERYTYPE-REROUTE: additive, high-precision symbol-graph hint —
        # exact identifier-shaped-token matches only (never a keyword/intent
        # match), appended BELOW the L3 results. Nothing is prepended,
        # reordered, or reweighted; empty when no identifier-shaped token in
        # the query resolves exactly.
        hint_symbols = service.identifier_hint_symbols(query)
        # UPG-LOWCONF-SLIM-DEDUPE: in pointer mode the L3 pointer list already
        # shows file:line for every result — an exact-match hint for the same
        # (file_path, start_line) would just repeat it verbatim. Deterministic
        # set intersection on the already-computed result set, not a
        # query-content heuristic. A hint symbol not already shown still gets
        # its own line.
        if low_conf and hint_symbols:
            shown = {(r.file_path, r.symbol_start_line) for r in results}
            hint_symbols = [s for s in hint_symbols if (s.file_path, s.start_line) not in shown]
        if hint_symbols:
            hint_lines = ["─── Symbol graph (exact matches for query identifiers) ───"]
            for s in hint_symbols:
                hint_lines.append(
                    f"  [{s.kind}] {s.name}  {workspace_relpath(s.file_path, ws_root)}:{s.start_line}"
                )
            sections.append("\n".join(hint_lines))

        # UPG-NEARMISS-SYMBOL-NAMES: additive, honestly-labeled follow-on —
        # for an identifier-shaped token that did NOT resolve exactly, show
        # the nearest existing symbol names so the caller learns it
        # misremembered a name instead of seeing nothing. Explicitly labeled
        # inexact ("No exact match ... nearest symbol names"), appended below
        # the exact-match section above; never reorders/replaces L3 results.
        nearmiss_pairs = service.identifier_hint_nearmiss(query)
        if nearmiss_pairs:
            nm_lines = []
            for token, syms in nearmiss_pairs:
                names = ", ".join(
                    f"{s.name} ({workspace_relpath(s.file_path, ws_root)}:{s.start_line})" for s in syms
                )
                nm_lines.append(
                    f"─── No exact match for {token!r}; nearest symbol names: {names} ───"
                )
            sections.append("\n".join(nm_lines))

        content_text = "\n\n".join(sections)

        # auto-append eviction hint only on a FRESH context-pressure escalation
        # (UPG-7.1) — gated so it can't repeat on every response. Reads the
        # calling session's own advisor (UPG-EVICT-SESSION-SCOPE).
        # UPG-REMEMBER-BANNER-FATIGUE: at most one remember-related banner per
        # response — the eviction hint (when it fires) already tells the
        # caller to call vectr_remember, so the softer turn-count nudge only
        # runs when the eviction hint didn't.
        hint = service.auto_eviction_hint(session_id=session_id)
        if hint:
            content_text += f"\n\n─── Context management hint ───\n{hint}"
        elif _should_nudge_remember(session_id):
            content_text += _remember_nudge_text(session_id)

        return {"content": [{"type": "text", "text": content_text}], "isError": False}

    # ---- vectr_fetch ----
    if tool_name == "vectr_fetch":
        ids = arguments.get("ids")
        if not isinstance(ids, list) or not ids:
            return _mcp_error("ids is required (a non-empty list of chunk ids)")
        ids = [str(i) for i in ids]

        # Memory-only mode: no code index to fetch from — same guard as search/locate/trace.
        if getattr(service, "memory_only", False):
            from app.service import _MEMORY_ONLY_MSG
            return {"content": [{"type": "text", "text": _MEMORY_ONLY_MSG}], "isError": False}

        try:
            # Same off-loop dispatch as vectr_search above.
            entries = dispatch_chroma_sync(service, service.fetch, ids)
        except ValueError as exc:
            return _mcp_error(str(exc))

        # Record fetched chunks into the calling session's eviction advisor —
        # mirrors the recording behaviour of vectr_search above.
        try:
            for e in entries:
                if e["found"]:
                    service.record_chunk(
                        file_path=e["file_path"],
                        lines=f"{e['start_line']}-{e['end_line']}",
                        symbol_name=e.get("symbol_name", ""),
                        content=e["content"],
                        chunk_id=e["id"],
                        session_id=session_id,
                    )
        except Exception:
            pass

        text = _format_fetch_results(entries, _service_ws_root(service))
        return {"content": [{"type": "text", "text": text}], "isError": False}

    # ---- vectr_status ----
    if tool_name == "vectr_status":
        status = service.status()
        notes_count = status.get("notes_count", 0)

        # if this session has prior notes, enable memory tools immediately
        if notes_count > 0:
            enable_memory_for_session(session_id)

        recall_hint = (
            f"  → call vectr_recall(query=...) to retrieve them"
            if notes_count > 0
            else "  → no prior notes; skip vectr_recall"
        )
        # Episode-capture aggregates: deterministic integer counts from
        # vectr's own store, rendered on every call. The distill hint is
        # count-gated only — no content classification.
        episodes_count = status.get("episodes_count", 0)
        arcs_pending = status.get("arcs_pending_distill", 0)
        distill_hint = (
            "  → call vectr_distill() to review them" if arcs_pending > 0 else ""
        )
        mode = status.get("mode", "full")
        lines = [
            "Vectr status",
            f"  Mode           : {mode}",
            f"  Indexed files  : {status['indexed_files']}",
            f"  Total chunks   : {status['total_chunks']}",
            f"  Purpose vectors: {status.get('total_purpose_chunks', 0)}",
            f"  Symbols indexed: {status.get('symbol_count', 'n/a')}",
            f"  Prior notes    : {notes_count}{recall_hint}",
            f"  Episodes       : {episodes_count}",
            f"  Pending arcs   : {arcs_pending}{distill_hint}",
            f"  Last indexed   : {status['last_indexed']}",
            f"  Embed model    : {status['embed_model']}",
            f"  Workspace      : {status['workspace_root']}",
        ]
        if mode == "memory-only":
            lines.append(
                "  → memory-only mode: search/locate/trace are disabled; "
                "memory tools (remember/recall/snapshot) and hooks are active"
            )
        elif mode == "search-only":
            lines.append(
                "  → search-only mode: working-memory tools (remember/recall/forget/"
                "snapshot) are disabled; search, locate, trace and map are active"
            )

        # UPG-STDIO-MEMORY-READY: additive warm-up indicator — only shown
        # while phase 2 (embedder/indexer/searcher/watcher/symbol-graph) is
        # still building. Memory tools (this one included) are already fully
        # usable at this point; only search/locate/trace/map/fetch are not.
        if not status.get("fully_ready", True):
            if not status.get("embedder_ready", True):
                lines.append(
                    "  → still starting up: the embedding model is loading (or "
                    "downloading) in the background — search/locate/trace/map "
                    "are not yet available; memory tools work now, but "
                    "vectr_recall semantic ranking is temporarily lexical-only "
                    "until the model finishes loading"
                )
            else:
                lines.append(
                    "  → still starting up: indexing and the symbol graph are "
                    "still building — search/locate/trace/map are not yet "
                    "available; memory tools are fully ready"
                )

        # UPG-PURPOSE-PASS-DEFERRAL: purpose vectors (the dual-vector search
        # ranking signal) are still backfilling in the background — search
        # already returns results during this window (body-similarity only),
        # so this makes a possible temporary ranking-quality dip explainable
        # rather than silent. Clears on its own once the pass finishes; no
        # action needed from the caller.
        if status.get("purpose_vectors_pending", False):
            lines.append(
                "  → purpose vectors are backfilling in the background: "
                "search results are ranked on body similarity only until this "
                "finishes — no action needed, this clears automatically"
            )

        # UPG-PURPOSE-RESUME-HOLE: an explicit warning rather than leaving
        # the reader to divide "Total chunks" by "Purpose vectors" (a ratio
        # that is ALWAYS < 1 even when nothing is wrong — only symbol-
        # bearing chunks get a purpose vector at all). This count is instead
        # the exact number of files the most recent index run found with
        # current body content but a purpose-vector gap — e.g. a deferred
        # purpose pass interrupted by a daemon stop/crash. Fires only while
        # a gap was actually detected and its backfill dispatch is still
        # pending; a dispatched-but-still-running backfill is already
        # covered by the purpose_vectors_pending line above.
        purpose_gap_files = status.get("purpose_backfill_pending_files", 0)
        if purpose_gap_files > 0:
            lines.append(
                f"  WARNING        : {purpose_gap_files} file(s) have body "
                "content indexed but incomplete purpose vectors — search "
                "ranking for their symbols may be body-similarity-only until "
                "this backfills; the next index run closes this "
                "automatically, no action needed"
            )

        # UPG-NOTES-EMBED-MIGRATION: surfaces a mid-failure state only — the
        # migration that keeps note vectors and the configured embed model in
        # sync runs synchronously at startup, so this is normally absent.
        notes_mismatch = status.get("notes_embed_model_mismatch")
        if notes_mismatch:
            lines.append(
                f"  WARNING        : working-memory notes are stamped with embed "
                f"model {notes_mismatch!r} but {status['embed_model']!r} is "
                "configured — semantic recall ranking may be degraded until "
                "migration completes on the next restart"
            )

        # UPG-TASK-SUPERSEDES-HYGIENE / UPG-STATUS-AGE-ONLY-FORGET-NUDGE: a
        # neutral inventory line, not a WARNING. Age alone is not a
        # deterministic staleness signal — a 10-day-old checkpoint may be
        # exactly current work — so this never alarms and never nominates
        # vectr_forget (the one destructive verb, whose product thesis
        # forbids removal on an age threshold). The only remediation it
        # names is supersession: recording what happened to the work via a
        # successor checkpoint is a caller judgment about STATE, which is
        # precisely the remediation kind="task" notes admit.
        stale_task_count = status.get("stale_task_count", 0)
        if stale_task_count >= MEMORY_HYGIENE_STALE_TASK_WARN_COUNT:
            oldest_id = status.get("stale_task_oldest_id")
            lines.append(
                f"  Task notes     : {stale_task_count} checkpoint(s) older than "
                f"{MEMORY_HYGIENE_STALE_TASK_WARN_AGE_DAYS} days are still active "
                f"(oldest: #{oldest_id}) — age alone never retires a note. When one "
                "describes finished or moved-on work, record that with "
                "vectr_remember(kind=\"task\", priority=\"high\", supersedes=<old id>)"
            )

        # Per-language coverage + symbol availability (UPG-3.3). Tells the agent
        # where locate/trace will work (symbol graph) vs. where to use search only.
        langs = status.get("languages") or []
        if langs:
            lines.append("  Languages      : (✓ = locate/trace available; others are search-only)")
            for L in langs[:8]:
                mark = "✓ locate/trace" if L["symbols"] else "search-only"
                lines.append(f"      {L['language']:<12} {L['files']:>5} files   {mark}")
            if len(langs) > 8:
                lines.append(f"      … +{len(langs) - 8} more")
            top = langs[0]
            if not top["symbols"]:
                lines.append(
                    f"  → Primary language ({top['language']}) has no symbol graph — "
                    "prefer vectr_search; locate/trace will be empty here."
                )
        # Warn when declared grammars are not importable in this environment.
        grammars_unavailable = status.get("grammars_unavailable") or []
        if grammars_unavailable:
            lines.append(
                f"  WARNING        : tree-sitter grammar(s) not importable: "
                f"{', '.join(grammars_unavailable)} — locate/trace disabled for these; "
                "reinstall vectr deps (pip install -e .) to enable"
            )
        if status.get("semantic_weight") is not None:
            lines.append(
                f"  Retrieval      : semantic={status['semantic_weight']:.0%}  "
                f"bm25={status['bm25_weight']:.0%}  "
                f"graph_first={status['graph_first']}"
            )
            if status.get("strategy_rationale"):
                lines.append(f"  Strategy why   : {status['strategy_rationale']}")

        # Watcher backlog (UPG-WATCHER-PRESSURE-GOVERNOR): only surfaced when
        # there is something to report, so a quiet workspace's status stays terse.
        pending_files = status.get("watcher_pending_files", 0)
        burst_mode = status.get("watcher_burst_mode", False)
        batch_running = status.get("watcher_batch_running", False)
        if pending_files or burst_mode or batch_running:
            last_ms = status.get("watcher_last_batch_duration_ms", 0)
            lines.append(
                f"  Watcher        : {pending_files} file(s) pending"
                f"{' (burst mode)' if burst_mode else ''}"
                f"{' — batch running' if batch_running else ''}"
                f"{f', last batch {last_ms}ms' if last_ms else ''}"
            )

        # Hook injection counters (UPG-HOOK-INJECT-OBSERVABILITY): only shown
        # when at least one hook has actually injected notes, so a workspace
        # with no Claude Code hooks installed (or whose hooks haven't fired
        # yet) stays terse — same quiet-when-zero pattern as the watcher line
        # above. Without this line, hook injection is invisible: notes land
        # silently in the model's context and there's no way to tell a
        # working memory system from a dead one.
        hook_counts = status.get("hook_injection_counts") or {}
        if hook_counts:
            hook_parts = ", ".join(f"{kind} {n}" for kind, n in hook_counts.items())
            lines.append(f"  Hook injections: {hook_parts}")

        # inject adaptive instruction style hint at session start
        try:
            style = service.suggest_instruction_style()
            # UPG-TOOLSTYLE-LABEL-COLLISION: this is a CLAUDE.md authoring-style
            # label, unrelated to the "Mode" line above (full/memory-only/
            # search-only, from service.status()) — "memory-first" (not
            # "memory-only") so the two never render side by side looking
            # contradictory, e.g. "Mode: full" next to "Tool style: [memory-only]".
            style_hints = {
                "additive":     "Use vectr tools when they'd be faster than reading files — see CLAUDE.md.",
                "directed":     "This is a large/unfamiliar codebase. Use vectr_map → vectr_search → vectr_locate before reading files.",
                "memory-first": "Prior notes exist. Call vectr_recall(query=...) first; use search only to fill gaps.",
            }
            hint = style_hints.get(style, "")
            if hint:
                lines.append(f"  Tool style     : [{style}] {hint}")
        except Exception:
            pass

        text = "\n".join(lines)
        return {"content": [{"type": "text", "text": text}], "isError": False}

    # ---- vectr_map ----
    if tool_name == "vectr_map":
        # Memory-only mode: no code index to map — same guard every sibling
        # code-index tool carries (UPG-MAP-MEMORY-ONLY-GUARD). Without it a
        # memory-only daemon returns an empty-ish passport instead of the
        # mode-contract message search/locate/trace/fetch all give.
        if getattr(service, "memory_only", False):
            from app.service import _MEMORY_ONLY_MSG
            return {"content": [{"type": "text", "text": _MEMORY_ONLY_MSG}], "isError": False}
        text = service.get_map()
        return {"content": [{"type": "text", "text": text}], "isError": False}

    # ---- vectr_map_save ----
    if tool_name == "vectr_map_save":
        if getattr(service, "memory_only", False):
            from app.service import _MEMORY_ONLY_MSG
            return {"content": [{"type": "text", "text": _MEMORY_ONLY_MSG}], "isError": False}
        summary = arguments.get("summary", "").strip()
        if not summary:
            return _mcp_error("summary is required")
        overwrite = bool(arguments.get("overwrite", False))
        result = service.save_map(summary, overwrite=overwrite)
        if not result["saved"]:
            return {
                "content": [{"type": "text", "text": (
                    "A passport already exists for this workspace — not overwritten. "
                    "Call vectr_map_save again with overwrite=true to replace it. "
                    "Existing summary:\n\n"
                    f"{result['existing_summary']}"
                )}],
                "isError": False,
            }
        return {
            "content": [{"type": "text", "text": (
                f"Passport saved ({len(summary)} chars). "
                "Future vectr_map calls will return this summary instantly."
            )}],
            "isError": False,
        }

    # ---- vectr_locate ----
    if tool_name == "vectr_locate":
        name = _symbol_name_arg(arguments)
        if not name:
            return _mcp_error("name is required")

        # Memory-only mode: locate is disabled for this workspace
        if getattr(service, "memory_only", False):
            from app.service import _MEMORY_ONLY_MSG
            return {"content": [{"type": "text", "text": _MEMORY_ONLY_MSG}], "isError": False}

        limit = int(arguments.get("limit", 10))
        caller_file = arguments.get("caller_file", "").strip() or None
        symbols = service.locate_with_snippets(name, limit=limit, caller_file=caller_file)
        text = service.format_locate(symbols, name)

        # Record the rendered snippets into the calling session's eviction
        # advisor (UPG-EVICT-SESSION-SCOPE) — same render-time recording
        # contract as vectr_search. No chunk_id: a symbol's line range is not
        # guaranteed to match a stored chunk boundary, so vectr_fetch is never
        # advertised for these (would risk a re-fetch key that doesn't work).
        try:
            rendered_symbols = getattr(symbols, "symbols", None)
            if rendered_symbols is None and isinstance(symbols, list):
                rendered_symbols = symbols
            for s in rendered_symbols or []:
                if getattr(s, "snippet", ""):
                    service.record_chunk(
                        file_path=s.file_path,
                        lines=f"{s.start_line}-{s.end_line}",
                        symbol_name=s.name,
                        content=s.snippet,
                        session_id=session_id,
                    )
        except Exception:
            pass

        # UPG-REMEMBER-BANNER-FATIGUE: at most one remember-related banner.
        hint = service.auto_eviction_hint(session_id=session_id)  # UPG-7.1: gated, not every response
        if hint:
            text += f"\n\n─── Context management hint ───\n{hint}"
        elif _should_nudge_remember(session_id):
            text += _remember_nudge_text(session_id)
        return {"content": [{"type": "text", "text": text}], "isError": False}

    # ---- vectr_trace ----
    if tool_name == "vectr_trace":
        name = _symbol_name_arg(arguments)
        if not name:
            return _mcp_error("name is required")

        # Memory-only mode: trace is disabled for this workspace
        if getattr(service, "memory_only", False):
            from app.service import _MEMORY_ONLY_MSG
            return {"content": [{"type": "text", "text": _MEMORY_ONLY_MSG}], "isError": False}

        direction = arguments.get("direction", "both")
        if direction not in ("callers", "callees", "both"):
            direction = "both"
        limit = int(arguments.get("limit", 20))
        include_builtins = bool(arguments.get("include_builtins", False))
        trace_result = service.trace_with_snippets(
            name, direction=direction, limit=limit, include_builtins=include_builtins
        )
        text = service.format_trace(trace_result, name)
        # UPG-REMEMBER-BANNER-FATIGUE: at most one remember-related banner.
        hint = service.auto_eviction_hint(session_id=session_id)  # UPG-7.1: gated, not every response
        if hint:
            text += f"\n\n─── Context management hint ───\n{hint}"
        elif _should_nudge_remember(session_id):
            text += _remember_nudge_text(session_id)
        return {"content": [{"type": "text", "text": text}], "isError": False}

    # ---- vectr_remember ----
    if tool_name == "vectr_remember":
        # UPG-REMEMBER-MCP-LONG-PAYLOAD-PARSE-LOSS: content_file is an
        # alternative body source read from disk, mutually exclusive with
        # inline content — see resolve_remember_content's docstring. The
        # resolved body is then stripped exactly as inline `content` always
        # was here, so content_file behaves identically to content at this
        # surface regardless of which one supplied the body. extra_roots is
        # passed through so a multi-root instance (a primary root plus one
        # or more extra_roots) accepts a content_file path under ANY served
        # root, not only the primary — see _service_extra_roots.
        try:
            content = resolve_remember_content(
                _service_ws_root(service),
                arguments.get("content"),
                arguments.get("content_file"),
                extra_roots=_service_extra_roots(service),
            ).strip()
        except ValueError as exc:
            return _mcp_error(str(exc))
        if not content:
            return _mcp_error("content is required")

        # Search-only mode: the working-memory layer is disabled for this workspace
        if getattr(service, "search_only", False):
            from app.service import _SEARCH_ONLY_MSG
            return {"content": [{"type": "text", "text": _SEARCH_ONLY_MSG}], "isError": False}

        tags = arguments.get("tags") or None
        priority = arguments.get("priority", "medium")
        if priority not in ("high", "medium", "low"):
            priority = "medium"
        kind = arguments.get("kind", "finding")
        title = arguments.get("title", "") or ""
        # UPG-SUBAGENT-MEMORY: optional caller-declared agent/subagent
        # identifier for multi-agent shared-memory attribution — never inferred.
        # In team mode, fall back to the connecting client's label (X-Vectr-Client
        # header) when the call itself does not declare an agent.
        agent = (arguments.get("agent", "") or "") or client_label
        # TRIGGER-ENGINE wave 1 (bm2-design-skeleton.md §1/§2/§5) — all
        # optional, all additive; omitting every one of these reproduces
        # exactly the pre-wave-1 vectr_remember call.
        triggers = arguments.get("triggers") or None
        provenance = arguments.get("provenance", "agent") or "agent"
        if provenance == "human":
            # The MCP surface is the AGENT's own surface (bm2-design-skeleton
            # .md §5: "promotion is an explicit user act") -- the same
            # boundary vectr_promote's own to="human" rejection enforces
            # below. Minting a note straight to provenance='human' here would
            # let an agent decide, on its own, that a person endorsed
            # something: a one-call trust forgery, since only human-
            # provenance directives render with the unhedged imperative
            # framing (format_notes_for_llm()) that gets auto-injected at
            # every future session start. Human endorsement happens on a
            # user-side surface instead (the REST /v1/remember or
            # /v1/promote routes a person's own CLI/UI calls), never here.
            return _mcp_error(
                "provenance='human' is not available via this tool -- "
                "human endorsement happens on a user-side surface (e.g. a "
                "CLI/UI a person operates), not through the AI's own MCP "
                "tools. Omit provenance (defaults to 'agent') or pass "
                "'agent'/'auto'; ask the person to record it themselves, or "
                "promote it later, if it warrants human endorsement."
            )
        # UPG-TRIGGER-SCOPE-KIND-DEFAULTS: None (omitted, or an empty string)
        # means "let the store resolve this kind's default scope at write
        # time"; an explicit non-empty scope (including "workspace") is
        # passed through verbatim.
        scope = arguments.get("scope") or None
        anchors = arguments.get("anchors") or None
        # UPG-MEM-PROVENANCE-USER-STATED: a verbatim excerpt of the user turn
        # this note transcribes. Unlike `provenance` just above, this is NOT a
        # trust claim the caller can assert — the store binds it only when the
        # excerpt is verifiably quoted in `content`, so what an agent supplies
        # here is evidence, checkable by machine, rather than self-attestation.
        # A non-string (or an unbindable) value never fails the write.
        user_quote = arguments.get("user_quote") or None
        if user_quote is not None and not isinstance(user_quote, str):
            return _mcp_error("user_quote must be a string")
        supersedes = arguments.get("supersedes")
        if supersedes is not None:
            try:
                supersedes = int(supersedes)
            except (TypeError, ValueError):
                return _mcp_error("supersedes must be an integer note_id")
        # UPG-MEMORY-STATE-MACHINE §4.2: contradicts is a peer of supersedes,
        # not a replacement for it -- see remember()'s docstring for the
        # exact revoked-event semantics.
        contradicts = arguments.get("contradicts")
        if contradicts is not None:
            try:
                contradicts = int(contradicts)
            except (TypeError, ValueError):
                return _mcp_error("contradicts must be an integer note_id")
        # UPG-RECALL-MISS-FLOOR part (b): pin this note into Tier 0 at write
        # time — equivalent to a separate vectr_pin call right after this
        # write. Default false changes nothing about existing callers.
        pin = bool(arguments.get("pin", False))
        try:
            outcome = dispatch_chroma_sync(
                service,
                service.remember_with_extras,
                content=content, tags=tags, priority=priority, kind=kind, title=title, agent=agent,
                triggers=triggers, provenance=provenance, scope=scope, anchors=anchors,
                supersedes=supersedes, contradicts=contradicts, session_id=session_id,
                user_quote=user_quote, pin=pin,
            )
        except ValueError as exc:
            # Malformed triggers, an unrecognised provenance/scope, a
            # supersedes target, or a contradicts target that does not
            # exist — a caller input error, surfaced plainly rather than
            # raised as an unhandled exception.
            return _mcp_error(str(exc))
        note_id = outcome.note_id
        # reset the turn-count nudge, the eviction advisor's remember-fatigue
        # counter (UPG-REMEMBER-BANNER-FATIGUE), and enable memory tools
        _reset_calls_since_save(session_id)
        service.note_remembered(session_id=session_id)
        enable_memory_for_session(session_id)
        # This note distills one or more
        # pending arcs — resolve them as a second step, after the note write
        # already succeeded, never blocking it. The schema declares
        # distilled_from as a list of integers; an element that is not
        # genuinely an int (bool excluded — JSON true/false deserializes to
        # a Python bool, a subclass of int, but is never a valid arc id) is
        # reported back to the caller rather than silently dropped, and
        # never causes a genuinely-int sibling in the same list to be lost.
        distill_suffix = ""
        distilled_from = arguments.get("distilled_from") or None
        if distilled_from:
            arc_ids: list[int] = []
            invalid_arc_ids: list[Any] = []
            for a in distilled_from:
                if isinstance(a, int) and not isinstance(a, bool):
                    arc_ids.append(a)
                else:
                    invalid_arc_ids.append(a)
            if arc_ids:
                result = service.resolve_arcs_distilled(arc_ids, note_id)
                resolved = result.get("resolved", [])
                unresolved = result.get("unresolved", [])
                distill_suffix = f"\n  Distilled arcs {resolved}"
                if unresolved:
                    distill_suffix += f" (unresolvable: {unresolved})"
            if invalid_arc_ids:
                distill_suffix += f"\n  Ignored non-integer distilled_from entries: {invalid_arc_ids!r}"
        # UPG-SCOPE-SURFACE-BACK: an omitted scope is resolved from `kind`'s
        # default at write time (UPG-TRIGGER-SCOPE-KIND-DEFAULTS), but that
        # resolution was write-only — the caller had no way to learn where
        # the note actually landed without a separate vectr_recall(detail=
        # "full") round trip. One cheap primary-key getter (the same one the
        # vectr_recall expand path already uses) surfaces it in this same
        # confirmation instead. Best-effort: a lookup failure just omits the
        # scope suffix rather than failing the whole write confirmation.
        scope_suffix = ""
        try:
            stored_note = service.get_note(note_id)
        except Exception:
            stored_note = None
        if stored_note is not None:
            if stored_note.scope == "branch" and stored_note.branch:
                scope_suffix = f" (scope=branch ({stored_note.branch}))"
            else:
                scope_suffix = f" (scope={stored_note.scope})"
        # UPG-ADOPTION-V2-MINOR (b): echo the stored title + first content line so
        # the caller can confirm the write landed correctly without a verify
        # round-trip via vectr_recall. Uses the note already fetched above — no
        # extra lookup. Bounded so a long note can't bloat the confirmation.
        echo = ""
        if stored_note is not None:
            title = (getattr(stored_note, "title", "") or "").strip()
            content_lines = (getattr(stored_note, "content", "") or "").strip().splitlines()
            first_line = content_lines[0].strip() if content_lines else ""
            if len(first_line) > 120:
                first_line = first_line[:117] + "..."
            parts = []
            if title:
                parts.append(f"title: {title}")
            # Only show the first line when it adds information — when no title
            # was supplied it is derived from the first content line (app/models.py),
            # so title == first_line and echoing both just prints the same text twice.
            if first_line and first_line != title:
                parts.append(f"first line: {first_line}")
            if parts:
                echo = "\n  Stored — " + " · ".join(parts)
        # UPG-MEM-PROVENANCE-USER-STATED / UPG-PROVENANCE-NEVER-RISES: report
        # what the binding check decided, in the same response as the write.
        # A failed bind is not a failed write (the note is stored at its
        # ordinary provenance), so the reason has to be surfaced here or the
        # caller would silently believe it recorded a user statement.
        # `_bound_quote`/`quote_reason` are recomputed with the same pure
        # function the store used for the EXPLICIT `user_quote` argument (no
        # rejection state is persisted anywhere) — but that recompute alone
        # is blind to the harness-driven auto-bind fallback
        # (`bind_user_quote_auto`, tried server-side whenever the explicit
        # quote did not bind), so the actual STORED provenance is checked too:
        # a note may land at provenance='user-stated' with no `user_quote`
        # argument at all, or with one that itself failed to bind, when it
        # was auto-bound instead — this must still be surfaced, or the
        # calling LLM would have no way to learn its note's real trust class
        # without a separate vectr_recall round trip.
        quote_suffix = ""
        _bound_quote, quote_reason = bind_user_quote(content, user_quote) if user_quote else ("", "")
        note_is_user_stated = stored_note is not None and stored_note.provenance == USER_STATED_PROVENANCE
        if _bound_quote:
            quote_suffix = f"\n  user_quote bound — stored as provenance='{USER_STATED_PROVENANCE}'."
        elif note_is_user_stated:
            quote_suffix = (
                "\n  No user_quote argument bound, but this note auto-bound to the "
                f"most recently captured user turn — stored as provenance="
                f"'{USER_STATED_PROVENANCE}'."
            )
        elif user_quote and quote_reason:
            quote_suffix = f"\n  {quote_reason}"
        # Write-time related-notes offer (agent/working_context_store/_related.py):
        # nearest-by-similarity only, never a contradiction verdict — the caller
        # LLM judges whether one is now stale. Empty unless `outcome.related` is
        # non-empty (config-gated in remember_with_extras).
        related_suffix = ""
        if outcome.related:
            rendered = []
            for r in outcome.related:
                related_title = r["title"]
                if len(related_title) > 60:
                    related_title = related_title[:57] + "..."
                rendered.append(f"#{r['note_id']} \"{related_title}\" ({r['similarity']:.2f})")
            related_suffix = (
                "\nRelated existing notes (nearest by similarity — not a contradiction "
                f"judgment): {', '.join(rendered)}. If one is now wrong, pass "
                "supersedes=<id> on a corrected note or call vectr_revoke."
            )
        # Write-time REVOKED-related offer (UPG-RELATED-REVOKED-DETERRENT,
        # agent/working_context_store/_related.py::revoked_related_notes) —
        # deliberately distinct from `related_suffix` above: this is a
        # deterrent, not a similarity nudge, and the caller LLM is writing
        # something similar to a note the user already marked wrong, right
        # now. Never repeats the revoked note's raw content, only its
        # bounded title plus the revocation's own date/reason. Empty unless
        # `outcome.revoked_related` is non-empty (config-gated in
        # remember_with_extras).
        revoked_suffix = ""
        if outcome.revoked_related:
            rendered = []
            for r in outcome.revoked_related:
                revoked_title = r["title"]
                if len(revoked_title) > 60:
                    revoked_title = revoked_title[:57] + "..."
                rendered.append(
                    f"#{r['note_id']} \"{revoked_title}\" (similarity {r['similarity']:.2f}, "
                    f"revoked {r['revoked_date']}, reason: {r['reason']})"
                )
            revoked_suffix = (
                "\nDETERRENT — an existing note like this one was already revoked: "
                f"{'; '.join(rendered)}. Do not re-derive or restate it without "
                "verification; if this write repeats what was revoked, reconsider "
                "before storing."
            )
        # Write-time proxy-anchor offer (agent/proxy_anchors.py): glob presence
        # only, never a claim that the note is stale. Empty unless
        # `outcome.proxy_anchor_suggestions` is non-empty (config- and
        # kind/anchors-gated in remember_with_extras).
        proxy_suffix = ""
        if outcome.proxy_anchor_suggestions:
            # UPG-ANCHOR-ATTACH: advertise the single-call vectr_anchor
            # primitive instead of the old re-store-with-supersedes workaround.
            proxy_suffix = (
                f"\nProcess files present here: {', '.join(outcome.proxy_anchor_suggestions)}. "
                "To anchor this note to one, call vectr_anchor(note_id=<this note>, "
                "anchors=[<path>]) — no re-store needed. A changed anchor means the "
                "process MAY have changed, never that the note is wrong."
            )
        # UPG-RESUME-TASK-PRIORITY-VISIBILITY: session-start boot injection and
        # the resume surface show only priority='high' task notes (the shared
        # _boot_task_notes() query), while 'medium' is the default — so a
        # checkpoint saved without an explicit priority would never appear at
        # resume, silently. Surface that fact in the SAME confirmation as the
        # write, and only when the caller let the default stand (an explicitly
        # chosen lower priority is an intentional ranking decision, not an
        # oversight to nag about).
        priority_hint = ""
        if kind == "task" and arguments.get("priority") is None:
            priority_hint = (
                "\n  Note: session-start/resume surfaces show only priority='high' task "
                "checkpoints — this one stored as 'medium'. If it should be picked up at "
                "resume, re-store it with priority=\"high\" (and supersedes=<this id> when "
                "it replaces an older checkpoint)."
            )
        return {
            "content": [{"type": "text", "text": f"Stored note #{note_id}{scope_suffix}. Recall with vectr_recall — <50ms, verbatim, any time.{echo}{quote_suffix}{distill_suffix}{related_suffix}{revoked_suffix}{proxy_suffix}{priority_hint}"}],
            "isError": False,
        }

    # ---- vectr_recall ----
    if tool_name == "vectr_recall":
        # Search-only mode: the working-memory layer is disabled for this workspace
        if getattr(service, "search_only", False):
            from app.service import _SEARCH_ONLY_MSG
            return {"content": [{"type": "text", "text": _SEARCH_ONLY_MSG}], "isError": False}

        query = arguments.get("query") or None
        tags = arguments.get("tags") or None
        priority = arguments.get("priority") or None
        kind = arguments.get("kind") or None
        boot = bool(arguments.get("boot", False))
        limit = int(arguments.get("limit", 10))
        detail = arguments.get("detail", "index") or "index"
        sort_by = arguments.get("sort_by", "relevance") or "relevance"
        # UPG-MCP-SORTBY-PASSTHROUGH: the store used to silently treat any
        # unknown sort_by as 'relevance', which masks a caller typo as
        # "results just look different". Validate at the dispatch boundary
        # with the same vocabulary and the same "must be one of" error
        # shape as REST (app/models.py::RecallRequest.validate_sort_by,
        # UPG-REST-SORTBY-UNVALIDATED) — the schema enum constrains
        # well-behaved clients, this guards everything else.
        # UPG-SORTBY-SHARED-VOCAB: the vocabulary itself is the shared
        # SORT_BY_VALUES constant from agent.working_context_store so a
        # fifth sort mode added at the source automatically tightens this
        # guard, the REST validator, the MCP schema enum, and the CLI
        # argparse choices — adding it at one place and forgetting another
        # is the drift this change exists to eliminate.
        if sort_by not in SORT_BY_VALUES:
            return _mcp_error(
                f"sort_by must be one of: {', '.join(SORT_BY_VALUES)}"
            )
        max_age_days = arguments.get("max_age_days") or None
        if max_age_days is not None:
            try:
                max_age_days = float(max_age_days)
            except (TypeError, ValueError):
                max_age_days = None
        # UPG-RECALL-NOTE-ID-NO-EXPAND: note_id= is a LOOKUP of one stored
        # note, not a retrieval — a malformed id must say so explicitly
        # instead of silently degrading to the generic ranked/index path.
        # The reported live symptom: a caller passing "#581" (copied from
        # the [#N] prefix in index output) failed int(), became None here,
        # and got the whole index listing back instead of note #581's body.
        # Bool is rejected first because JSON true/false are int subclasses
        # (int(True) == 1 would silently expand an unrelated note), matching
        # the distilled_from handler's bool exclusion above.
        note_id_arg = arguments.get("note_id")
        if isinstance(note_id_arg, bool):
            return _mcp_error(
                "note_id must be an integer note id (the [#N] shown by "
                f"vectr_recall); got {note_id_arg!r}. Nothing was recalled."
            )
        if note_id_arg is not None:
            try:
                note_id_arg = int(note_id_arg)
            except (TypeError, ValueError):
                return _mcp_error(
                    "note_id must be an integer note id (the [#N] shown by "
                    f"vectr_recall); got {note_id_arg!r}. Nothing was recalled."
                )
        text = dispatch_chroma_sync(
            service,
            service.recall,
            query=query, tags=tags, priority=priority, limit=limit, kind=kind, boot=boot,
            detail=detail, sort_by=sort_by, max_age_days=max_age_days, note_id=note_id_arg,
            session_id=session_id,
        )
        hint = service.auto_eviction_hint(session_id=session_id)  # UPG-7.1: gated, not every response
        if hint:
            text += f"\n\n─── Context management hint ───\n{hint}"
        return {"content": [{"type": "text", "text": text}], "isError": False}

    # ---- vectr_evict_hint ----
    if tool_name == "vectr_evict_hint":
        # UPG-7.2: an explicit ask gets the on-demand, eviction-focused framing —
        # distinct from the gated auto-footer's "ACTION REQUIRED" remember alarm.
        hint = service.eviction_hint(session_id=session_id, on_demand=True)
        if not hint:
            hint = "No retrieved chunks to evict. Context window is clean."
        return {"content": [{"type": "text", "text": hint}], "isError": False}

    # ---- vectr_distill ----
    if tool_name == "vectr_distill":
        # Search-only mode: the working-memory layer is disabled for this workspace
        if getattr(service, "search_only", False):
            from app.service import _SEARCH_ONLY_MSG
            return {"content": [{"type": "text", "text": _SEARCH_ONLY_MSG}], "isError": False}

        dismiss_arg = arguments.get("dismiss") or None
        if dismiss_arg:
            reason = (arguments.get("reason") or "").strip()
            if not reason:
                return _mcp_error("reason is required together with dismiss=")
            try:
                arc_ids = [int(a) for a in dismiss_arg]
            except (TypeError, ValueError):
                return _mcp_error("dismiss must be a list of integer arc ids")
            result = service.resolve_arcs_dismissed(arc_ids, reason)
            resolved = result.get("resolved", [])
            unresolved = result.get("unresolved", [])
            text = f"Dismissed arcs {resolved}."
            if unresolved:
                text += f" Unresolvable (unknown or already resolved): {unresolved}."
            return {"content": [{"type": "text", "text": text}], "isError": False}

        # No args: render pending arcs for review — confidence-first then
        # oldest-first, already the order service.list_arcs returns.
        rows = service.list_arcs(status="pending", limit=EPISODES_DISTILL_MAX_ARCS_RENDERED)
        total_pending = service.count_arcs_pending_distill()
        return {"content": [{"type": "text", "text": _format_pending_arcs(rows, total_pending)}], "isError": False}

    # ---- vectr_snapshot ----
    if tool_name == "vectr_snapshot":
        label = arguments.get("label", "").strip()
        if not label:
            return _mcp_error("label is required")

        # Search-only mode: the working-memory layer is disabled for this workspace
        if getattr(service, "search_only", False):
            from app.service import _SEARCH_ONLY_MSG
            return {"content": [{"type": "text", "text": _SEARCH_ONLY_MSG}], "isError": False}

        # UPG-EVICT-SESSION-SCOPE: an explicit "session_id" argument (e.g. a
        # multi-agent caller labeling a shared snapshot) overrides the calling
        # MCP session; otherwise default to the transport-level session_id so
        # the snapshot captures THIS session's own retrieved chunks rather than
        # the anonymous shared advisor's (near-always empty for a real session).
        snapshot_session_id = arguments.get("session_id") or session_id
        snapshot_id = service.snapshot_session(label=label, session_id=snapshot_session_id)
        return {
            "content": [{"type": "text", "text": f"Snapshot saved: {snapshot_id}\nLabel: {label}\nNotes are available via vectr_recall any time — later in this session or in future sessions."}],
            "isError": False,
        }

    # ---- vectr_snapshot_list ----
    if tool_name == "vectr_snapshot_list":
        # Search-only mode: the working-memory layer is disabled for this workspace
        if getattr(service, "search_only", False):
            from app.service import _SEARCH_ONLY_MSG
            return {"content": [{"type": "text", "text": _SEARCH_ONLY_MSG}], "isError": False}

        snapshots = service.list_snapshots()
        if not snapshots:
            text = "No snapshots saved for this workspace yet. Use vectr_snapshot to save one."
        else:
            import datetime
            lines = [f"Saved snapshots ({len(snapshots)}):\n"]
            for s in snapshots:
                ts = datetime.datetime.fromtimestamp(s["created_at"]).strftime("%Y-%m-%d %H:%M")
                lines.append(f"  {s['snapshot_id']}  [{ts}]  {s['label']}")
            lines.append("\nUse vectr_recall to retrieve the notes from any session.")
            text = "\n".join(lines)
        return {"content": [{"type": "text", "text": text}], "isError": False}

    # ---- vectr_resume ----
    if tool_name == "vectr_resume":
        # Search-only mode: the working-memory layer is disabled for this workspace
        if getattr(service, "search_only", False):
            from app.service import _SEARCH_ONLY_MSG
            return {"content": [{"type": "text", "text": _SEARCH_ONLY_MSG}], "isError": False}

        result = service.resume(session_id=session_id, surface="mcp")
        return {"content": [{"type": "text", "text": result["formatted"]}], "isError": False}

    # ---- vectr_forget ----
    if tool_name == "vectr_forget":
        # Search-only mode: the working-memory layer is disabled for this workspace
        if getattr(service, "search_only", False):
            from app.service import _SEARCH_ONLY_MSG
            return {"content": [{"type": "text", "text": _SEARCH_ONLY_MSG}], "isError": False}

        note_id = arguments.get("note_id")
        if note_id is not None:
            # UPG-MCP-NOTEID-BOOL-GUARDS: JSON true/false are int subclasses
            # and int(True) == 1 would silently delete note #1 — same guard
            # vectr_recall/vectr_anchor already carry. The destructive failure
            # mode is the reason this item leads the lane.
            if isinstance(note_id, bool):
                return _mcp_error(
                    "note_id must be an integer (the [#N] id shown by "
                    f"vectr_recall); got {note_id!r}. Nothing was deleted."
                )
            try:
                nid = int(note_id)
            except (TypeError, ValueError):
                return _mcp_error("note_id must be an integer (the [#N] id shown by vectr_recall)")
            if dispatch_chroma_sync(service, service.forget_note, nid):
                return {
                    "content": [{"type": "text", "text": f"Deleted note #{nid}."}],
                    "isError": False,
                }
            return {
                "content": [{"type": "text", "text": f"Note #{nid} not found — nothing deleted."}],
                "isError": False,
            }
        if arguments.get("all") is True:
            deleted = dispatch_chroma_sync(service, service.forget_all)
            return {
                "content": [{"type": "text", "text": f"Deleted {deleted} working-memory notes. Starting fresh."}],
                "isError": False,
            }
        # No arguments must never destroy data: require an explicit target.
        return _mcp_error(
            "Pass note_id=<N> to delete one note, or all=true to clear every note for this workspace."
        )

    # ---- vectr_promote ----
    if tool_name == "vectr_promote":
        # Search-only mode: the working-memory layer is disabled for this workspace
        if getattr(service, "search_only", False):
            from app.service import _SEARCH_ONLY_MSG
            return {"content": [{"type": "text", "text": _SEARCH_ONLY_MSG}], "isError": False}

        note_id = arguments.get("note_id")
        to = arguments.get("to", "")
        if note_id is None:
            return _mcp_error("note_id is required (the [#N] id shown by vectr_recall)")
        # UPG-MCP-NOTEID-BOOL-GUARDS: JSON true/false are int subclasses and
        # int(True) == 1 would promote an unrelated note — same guard
        # vectr_recall/vectr_anchor already carry.
        if isinstance(note_id, bool):
            return _mcp_error(
                "note_id must be an integer (the [#N] id shown by "
                f"vectr_recall); got {note_id!r}. Nothing was promoted."
            )
        try:
            nid = int(note_id)
        except (TypeError, ValueError):
            return _mcp_error("note_id must be an integer (the [#N] id shown by vectr_recall)")
        if not to:
            return _mcp_error("to is required (only 'agent' is available via this tool)")
        if to == "human":
            # The MCP surface is the AGENT's surface (bm2-design-skeleton.md §5:
            # "promotion is an explicit user act"). Letting an agent raise its own
            # note straight to provenance='human' would let it decide, on its own,
            # that a person endorsed something -- reopening the trust-inversion
            # hole §5 closes structurally (only human-authored/endorsed notes get
            # the unhedged imperative directive framing in format_notes_for_llm()).
            # Human endorsement happens on a user-side surface instead (the REST
            # POST /v1/promote route a person's own CLI/UI calls), never here.
            return _mcp_error(
                "Promotion to provenance='human' is not available via this tool -- "
                "human endorsement happens on a user-side surface (e.g. a CLI/UI a "
                "person operates), not through the AI's own MCP tools. This tool "
                "only supports the auto -> agent step; ask the person to promote "
                "the note to 'human' themselves if it warrants that trust level."
            )
        try:
            promoted = service.promote_note(nid, to)
        except ValueError as exc:
            return _mcp_error(str(exc))
        if not promoted:
            return _mcp_error(f"Note #{nid} not found.")
        return {
            "content": [{"type": "text", "text": f"Promoted note #{nid} to provenance='{to}'."}],
            "isError": False,
        }

    # ---- vectr_revoke ----
    if tool_name == "vectr_revoke":
        # Search-only mode: the working-memory layer is disabled for this workspace
        if getattr(service, "search_only", False):
            from app.service import _SEARCH_ONLY_MSG
            return {"content": [{"type": "text", "text": _SEARCH_ONLY_MSG}], "isError": False}

        note_id = arguments.get("note_id")
        reason = (arguments.get("reason") or "").strip()
        if note_id is None:
            return _mcp_error("note_id is required (the [#N] id shown by vectr_recall)")
        # UPG-MCP-NOTEID-BOOL-GUARDS: JSON true/false are int subclasses and
        # int(True) == 1 would revoke an unrelated note — same guard
        # vectr_recall/vectr_anchor already carry.
        if isinstance(note_id, bool):
            return _mcp_error(
                "note_id must be an integer (the [#N] id shown by "
                f"vectr_recall); got {note_id!r}. Nothing was revoked."
            )
        try:
            nid = int(note_id)
        except (TypeError, ValueError):
            return _mcp_error("note_id must be an integer (the [#N] id shown by vectr_recall)")
        if not reason:
            return _mcp_error("reason is required — shown verbatim in the deterrent framing")
        # UPG-MEMORY-STATE-MACHINE §4.2: MCP is the AI's own surface, so
        # actor is hardcoded here (never accepted as a caller argument) --
        # only the REST /v1/revoke route a person's own CLI/UI calls can
        # attribute a revocation to actor="human".
        try:
            revoked = service.revoke_note(nid, reason, actor="agent")
        except ValueError as exc:
            return _mcp_error(str(exc))
        if not revoked:
            return _mcp_error(f"Note #{nid} not found.")
        return {
            "content": [{"type": "text", "text": f"Revoked note #{nid}. It will now surface as a deterrent instead of its original content until reinstated."}],
            "isError": False,
        }

    # ---- vectr_reinstate ----
    if tool_name == "vectr_reinstate":
        # Search-only mode: the working-memory layer is disabled for this workspace
        if getattr(service, "search_only", False):
            from app.service import _SEARCH_ONLY_MSG
            return {"content": [{"type": "text", "text": _SEARCH_ONLY_MSG}], "isError": False}

        note_id = arguments.get("note_id")
        reason = arguments.get("reason") or None
        if note_id is None:
            return _mcp_error("note_id is required (the [#N] id shown by vectr_recall)")
        # UPG-MCP-NOTEID-BOOL-GUARDS: JSON true/false are int subclasses and
        # int(True) == 1 would reinstate an unrelated note — same guard
        # vectr_recall/vectr_anchor already carry.
        if isinstance(note_id, bool):
            return _mcp_error(
                "note_id must be an integer (the [#N] id shown by "
                f"vectr_recall); got {note_id!r}. Nothing was reinstated."
            )
        try:
            nid = int(note_id)
        except (TypeError, ValueError):
            return _mcp_error("note_id must be an integer (the [#N] id shown by vectr_recall)")
        try:
            reinstated = service.reinstate_note(nid, actor="agent", reason=reason)
        except ValueError as exc:
            return _mcp_error(str(exc))
        if not reinstated:
            return _mcp_error(f"Note #{nid} not found.")
        return {
            "content": [{"type": "text", "text": f"Reinstated note #{nid}. Its original content will surface again."}],
            "isError": False,
        }

    # ---- vectr_supersede ----
    if tool_name == "vectr_supersede":
        # Search-only mode: the working-memory layer is disabled for this workspace
        if getattr(service, "search_only", False):
            from app.service import _SEARCH_ONLY_MSG
            return {"content": [{"type": "text", "text": _SEARCH_ONLY_MSG}], "isError": False}

        note_id = arguments.get("note_id")
        # Bool rejected up front: JSON true/false are int subclasses and
        # int(True) == 1 would retire an unrelated note.
        if note_id is None or isinstance(note_id, bool):
            return _mcp_error("note_id is required as an integer (the [#N] id shown by vectr_recall)")
        try:
            nid = int(note_id)
        except (TypeError, ValueError):
            return _mcp_error("note_id must be an integer (the [#N] id shown by vectr_recall)")
        superseded_by = arguments.get("superseded_by")
        if superseded_by is not None:
            if isinstance(superseded_by, bool):
                return _mcp_error("superseded_by must be an integer note id")
            try:
                superseded_by = int(superseded_by)
            except (TypeError, ValueError):
                return _mcp_error("superseded_by must be an integer note id")
        reason = arguments.get("reason") or None
        # UPG-MEMORY-STATE-MACHINE §4.2 convention (same as vectr_revoke):
        # MCP is the AI's own surface, so actor is hardcoded here (never a
        # caller argument) — only the REST /v1/supersede route a person's
        # own CLI/UI calls can attribute the retirement to actor="human".
        try:
            retired = service.supersede_note(nid, superseded_by=superseded_by, reason=reason, actor="agent")
        except ValueError as exc:
            return _mcp_error(str(exc))
        if not retired:
            return _mcp_error(f"Note #{nid} not found.")
        successor_str = (
            f" Its replacement is #{superseded_by}." if superseded_by is not None else ""
        )
        return {
            "content": [{"type": "text", "text": (
                f"Retired note #{nid} as superseded. It will no longer surface in "
                f"default recall/session-start and renders with the factual "
                f"'[superseded ...]' badge — not the revoked deterrent.{successor_str} "
                "Reverse with vectr_reinstate if this was wrong."
            )}],
            "isError": False,
        }

    # ---- vectr_pin ----
    if tool_name == "vectr_pin":
        # Search-only mode: the working-memory layer is disabled for this workspace
        if getattr(service, "search_only", False):
            from app.service import _SEARCH_ONLY_MSG
            return {"content": [{"type": "text", "text": _SEARCH_ONLY_MSG}], "isError": False}

        note_id = arguments.get("note_id")
        pinned = arguments.get("pinned", True)
        if note_id is None:
            return _mcp_error("note_id is required (the [#N] id shown by vectr_recall)")
        # UPG-MCP-NOTEID-BOOL-GUARDS: JSON true/false are int subclasses and
        # int(True) == 1 would pin an unrelated note — same guard
        # vectr_recall/vectr_anchor already carry.
        if isinstance(note_id, bool):
            return _mcp_error(
                "note_id must be an integer (the [#N] id shown by "
                f"vectr_recall); got {note_id!r}. Nothing was pinned."
            )
        try:
            nid = int(note_id)
        except (TypeError, ValueError):
            return _mcp_error("note_id must be an integer (the [#N] id shown by vectr_recall)")
        changed = service.pin_note(nid, bool(pinned))
        if not changed:
            return _mcp_error(f"Note #{nid} not found.")
        verb = "Pinned" if pinned else "Unpinned"
        return {
            "content": [{"type": "text", "text": f"{verb} note #{nid}. " + (
                "It will now be injected on every vectr_recall(query=...) call, regardless of the query."
                if pinned else
                "It will no longer be injected unconditionally — it now recalls the same as any other note."
            )}],
            "isError": False,
        }

    # ---- vectr_anchor ----
    if tool_name == "vectr_anchor":
        # Search-only mode: the working-memory layer is disabled for this workspace
        if getattr(service, "search_only", False):
            from app.service import _SEARCH_ONLY_MSG
            return {"content": [{"type": "text", "text": _SEARCH_ONLY_MSG}], "isError": False}

        # UPG-ANCHOR-ATTACH: post-write anchoring without a re-store.
        note_id = arguments.get("note_id")
        # Bool rejected up front: JSON true/false are int subclasses and
        # int(True) == 1 would silently mutate an unrelated note.
        if note_id is None or isinstance(note_id, bool):
            return _mcp_error("note_id is required as an integer (the [#N] id shown by vectr_recall)")
        try:
            nid = int(note_id)
        except (TypeError, ValueError):
            return _mcp_error("note_id must be an integer (the [#N] id shown by vectr_recall)")
        anchors_arg = arguments.get("anchors")
        # `not anchors_arg` is load-bearing: all() over an empty list is
        # True, so an explicit anchors=[] would otherwise pass a check
        # whose own message promises a non-empty list. REST enforces the
        # same floor with min_length=1 and the CLI with nargs="+".
        if not isinstance(anchors_arg, list) or not anchors_arg or not all(
            isinstance(a, str) and a.strip() for a in anchors_arg
        ):
            return _mcp_error("anchors must be a non-empty list of file paths")
        try:
            # Transport-level session identity only (same convention as
            # vectr_remember): the UPG-ANCHOR-UNOBSERVED-BINDING tri-state
            # verdict describes THIS harness session's own observation
            # ledger, so the calling LLM never gets to declare one.
            result = service.attach_anchors(
                note_id=nid, anchors=anchors_arg, session_id=session_id,
            )
        except ValueError as exc:
            return _mcp_error(str(exc))
        if result is None:
            return _mcp_error(f"Note #{nid} not found.")
        parts = [f"Attached {len(result['attached'])} anchor(s) to note #{nid}."]
        if result["already_present"]:
            parts.append(f"Already anchored (no-op): {', '.join(result['already_present'])}.")
        parts.append("A changed anchor means the process MAY have changed, never that the note is wrong.")
        return {
            "content": [{"type": "text", "text": " ".join(parts)}],
            "isError": False,
        }

    # ---- vectr_unanchor ----
    if tool_name == "vectr_unanchor":
        # UPG-ANCHOR-DETACH: post-write de-anchoring without a re-store.
        if getattr(service, "search_only", False):
            from app.service import _SEARCH_ONLY_MSG
            return {"content": [{"type": "text", "text": _SEARCH_ONLY_MSG}], "isError": False}

        note_id = arguments.get("note_id")
        # Bool rejected up front: same guard as vectr_anchor (JSON true/false
        # are int subclasses, so int(True) == 1 would silently mutate note #1).
        if note_id is None or isinstance(note_id, bool):
            return _mcp_error("note_id is required as an integer (the [#N] id shown by vectr_recall)")
        try:
            nid = int(note_id)
        except (TypeError, ValueError):
            return _mcp_error("note_id must be an integer (the [#N] id shown by vectr_recall)")
        anchors_arg = arguments.get("anchors")
        # Same non-empty-list guard as vectr_anchor (see comment there on
        # why `not anchors_arg` is load-bearing, not `not all(...)`).
        if not isinstance(anchors_arg, list) or not anchors_arg or not all(
            isinstance(a, str) and a.strip() for a in anchors_arg
        ):
            return _mcp_error("anchors must be a non-empty list of file paths")
        try:
            result = service.detach_anchors(note_id=nid, anchors=anchors_arg)
        except ValueError as exc:
            return _mcp_error(str(exc))
        if result is None:
            return _mcp_error(f"Note #{nid} not found.")
        parts = [f"Removed {len(result['removed'])} anchor(s) from note #{nid}."]
        if result["not_present"]:
            parts.append(f"Not anchored (no-op): {', '.join(result['not_present'])}.")
        return {
            "content": [{"type": "text", "text": " ".join(parts)}],
            "isError": False,
        }

    # ---- vectr_ingest_traces ----
    if tool_name == "vectr_ingest_traces":
        events = arguments.get("events")
        if not isinstance(events, list):
            return _mcp_error("events must be a list of trace event dicts")
        result = service.ingest_traces(events)
        text = (
            f"Ingested {result['ingested']} dynamic call edges into the symbol graph. "
            f"({result['skipped_invalid']} events skipped — missing caller or callee field.) "
            "Dynamic edges now appear in vectr_trace results (marked \"(dynamic)\") "
            "alongside static edges."
        )
        # UPG-7.3: an unresolved caller/callee is not an error (the target may be
        # external, runtime-only, or a symbol kind the static extractor doesn't
        # capture) but is worth surfacing — it may also be a typo'd trace event.
        unresolved_callers = result.get("unresolved_callers", 0)
        unresolved_callees = result.get("unresolved_callees", 0)
        if unresolved_callers or unresolved_callees:
            text += (
                f"\n\nWarning: {unresolved_callers} caller name(s) and "
                f"{unresolved_callees} callee name(s) did not match any symbol in "
                "the indexed graph — edges were still ingested, but double-check "
                "these aren't typos:"
            )
            for example in result.get("unresolved_examples", []):
                text += f"\n  {example}"
        return {
            "content": [{"type": "text", "text": text}],
            "isError": False,
        }

    return _mcp_error(f"Unknown tool: {tool_name}")


def _storage_cap_truncation_warning(
    content: str, file_path: str, symbol_start_line: int, symbol_end_line: int,
) -> str | None:
    """Detect index-time storage-cap truncation and return the Read() fallback
    warning, or None when the content is complete.

    A symbol chunk that exceeds the indexer's per-chunk cap (large classes/
    methods — see agent/indexer/_chunking.py) is stored capped: fewer lines
    than the symbol's own recorded `symbol_start_line`/`symbol_end_line` span.
    Detected by comparing the rendered content's line count against that span
    (UPG-11.4-b / UPG-FETCH-TRUNCATION-SILENT). No vectr_fetch(...) can
    recover the missing tail — the index itself holds only the capped
    content — so the fallback points at a direct file read reaching it.
    Shared by every surface that renders a chunk's content as prose
    (_format_search_results, _format_fetch_results) so a truncated chunk
    never appears complete on one surface and flagged on another.
    """
    content_lines = content.splitlines()
    symbol_range_lines = (
        symbol_end_line - symbol_start_line + 1
        if (symbol_start_line and symbol_end_line and symbol_end_line > symbol_start_line)
        else 0
    )
    if symbol_range_lines > 0 and len(content_lines) < symbol_range_lines - 5:
        missing = symbol_range_lines - len(content_lines)
        return (
            f"... {missing} more lines (content capped at ~2000 chars) — "
            f"Read({file_path!r}, offset={symbol_start_line - 1}, limit={symbol_range_lines}) for full definition"
        )
    return None


def _format_search_results(
    results, query: str, query_ms: int, chunks_searched: int, workspace_root: str = "",
) -> str:
    if not results:
        return f"No results found for: {query}"
    # UPG-LOWCONF-OUTPUT-SLIM / UPG-FLOOR-SLIM-PAYLOAD: when the low-confidence
    # banner fires, the caller is told these results may be unrelated — so ship
    # pointer-mode entries (file:line + symbol + score, no bodies) instead of
    # serializing ~2k tokens of chunks the caller was just told not to trust.
    # The chunk id is shown so a promising pointer can be expanded with
    # vectr_fetch. Deterministic, response-shaping only.
    low_conf = getattr(results, "low_confidence", False)
    # UPG-SCORE-ORDER-EXPLAIN: displayed relevance may disagree with display
    # order (the composite priors decide order, not the score). When a result
    # below rank 1 shows a MUCH higher relevance, annotate the demoting prior so
    # the divergence is readable. Additive; ordering untouched.
    top_score = results[0].score if results else 0.0
    # UPG-POINTER-MODE-UNIFORM-STRIP: pointer mode strips bodies uniformly per
    # SET, but a result whose own ce_relevance independently clears the
    # retention floor keeps a bounded excerpt instead — computed once here
    # (per result) so both the header wording and the per-result render below
    # agree on which results are retained. See agent/config.py
    # POINTER_MODE_RETAIN_* / config.yaml ranking.pointer_mode_retain.
    #
    # UPG-BANNER-CALIBRATION: two independent exemptions, tracked separately
    # so the render below can label each correctly. (1) rank1_retained — the
    # set's composite-top result, exempted purely by RANK, never by score
    # (mirrors result_floor's "always keep the top result" precedent). This
    # is the fix for the case an absolute floor structurally cannot reach:
    # when low_confidence fired because rank-1's own ce_relevance is below
    # notfound_floor.min_top_relevance, no retain floor >= that same bar can
    # ever pass for rank-1 (F47: rank-1 correct, ce_relevance 0.024, banner
    # fired, zero code delivered). (2) floor_retained — any result (including
    # rank-1) whose own ce_relevance clears the independently-calibrated
    # POINTER_MODE_RETAIN_MIN_RELEVANCE bar, unchanged in mechanism from
    # before this fix, just decoupled in value.
    rank1_retained = [
        bool(low_conf and i == 0 and POINTER_MODE_RETAIN_ENABLED and POINTER_MODE_RETAIN_RANK1_ALWAYS)
        for i in range(len(results))
    ]
    floor_retained = [
        bool(
            low_conf
            and POINTER_MODE_RETAIN_ENABLED
            and r.ce_relevance is not None
            and r.ce_relevance >= POINTER_MODE_RETAIN_MIN_RELEVANCE
        )
        for r in results
    ]
    retain_body = [a or b for a, b in zip(rank1_retained, floor_retained)]
    header = f"Found {len(results)} results for '{query}' ({query_ms}ms, {chunks_searched} chunks searched)"
    if low_conf:
        header += " — low confidence: pointers only (vectr_fetch(ids=[...]) to expand)"
        if any(retain_body):
            if any(rank1_retained):
                extra = sum(floor_retained[1:])
                header += "; the top-ranked result below keeps an excerpt as a best guess"
                if extra:
                    header += f", plus {extra} more that individually clear the confidence floor"
            else:
                header += f"; {sum(retain_body)} result(s) individually clear the confidence floor and keep an excerpt below"
    # UPG-RELATIVE-PATH-RENDER: print the absolute workspace root ONCE here; every
    # path/chunk-id below is rendered relative to it, so the ~21-token absolute
    # prefix stops riding every result line. vectr_fetch accepts the relative ids.
    lines = [header]
    if workspace_root:
        lines.append(f"workspace: {workspace_root}")
    lines.append("")
    for i, r in enumerate(results, 1):
        lines.append(f"{'─' * 60}")
        dup = f"  (+{r.dup_count} more identical)" if getattr(r, "dup_count", 0) else ""
        chunk_id = f"{workspace_relpath(r.file_path, workspace_root)}:{r.lines}"
        # UPG-MCP-SCORE-SOURCE-RENDER: surface which scale the displayed score is
        # on — "reranker" (cross-encoder sigmoid) vs "dense" (bi-encoder cosine).
        # REST already carries score_source; the caller LLM reads this score to
        # plan, so the render must say what it means. The mixed-scale fix keeps
        # score_source uniform within a response.
        src = getattr(r, "score_source", "") or ""
        src_label = f" ({src})" if src else ""
        # UPG-SCORE-ORDER-EXPLAIN, reformulated by UPG-GATE-V4-MINORS: a
        # below-rank-1 result whose displayed relevance exceeds rank 1's by a
        # large share of the score range still available ABOVE rank 1 gets a
        # one-phrase reason for its lower rank. Headroom share, not a fixed
        # multiple: displayed scores are bounded in [0, 1], so the old ratio
        # rule (r.score >= 1.5 * top_score) was structurally unsatisfiable on
        # confident queries — once top_score reached 2/3, no inversion however
        # large could ever be annotated (measured gate witness: rank-1 0.864
        # vs a rank-5 0.969). The headroom form shrinks the required gap as
        # rank-1 approaches the ceiling: gap >= MIN_HEADROOM_RATIO * (1 -
        # top_score), i.e. the inversion must consume that share of what's
        # left of the range (calibration in config.yaml; no division, so
        # top_score == 1.0 needs no guard — r.score > top_score cannot hold).
        # UPG-GATE-V4-MINORS (a): an EMPTY quality_reason means this chunk
        # carries no quality demotion at all, so its lower rank came from
        # rank-1-side priors or pool order — nothing observable from this
        # chunk alone. The old "composite ranking prior" fallback named a
        # mechanism without explaining anything; silence is the honest state.
        rank_note = ""
        reason = getattr(r, "quality_reason", "")
        if (
            SCORE_ORDER_EXPLAIN_ENABLED
            and i > 1
            and reason
            and r.score > top_score
            and (r.score - top_score) >= SCORE_ORDER_EXPLAIN_MIN_HEADROOM_RATIO * (1.0 - top_score)
        ):
            rank_note = f"  (ranked lower: {reason})"
        lines.append(f"[{i}] {chunk_id}  score {r.score:.3f}{src_label}{dup}{rank_note}")
        if r.symbol_name:
            # UPG-11.4: include symbol line-range so the caller knows the full
            # definition's extent even when the displayed content is capped —
            # vectr_fetch(ids=[chunk_id]) below restores it in full if needed.
            sym_range = ""
            s_start = getattr(r, "symbol_start_line", 0)
            s_end = getattr(r, "symbol_end_line", 0)
            if s_start and s_end:
                sym_range = f"  [lines {s_start}–{s_end}]"
            lines.append(f"    symbol: {r.symbol_name}{sym_range}  language: {r.language}")
        if low_conf and not retain_body[i - 1]:
            # Pointer mode: no body. A caller that finds a promising pointer
            # expands it deterministically with vectr_fetch.
            lines.append("")
            continue
        if low_conf:
            # UPG-POINTER-MODE-UNIFORM-STRIP / UPG-BANNER-CALIBRATION: this
            # result kept a bounded excerpt (never the full body) for one of
            # two independent reasons — say which, so the caller can weigh
            # trust correctly. floor_retained: its own ce_relevance clears
            # the (now decoupled) retention floor. rank1_retained-only: it is
            # simply the set's top-ranked guess, exempted by rank alone, and
            # may have no ce_relevance at all (reranking may not have run).
            if floor_retained[i - 1]:
                lines.append(f"    ({POINTER_MODE_RETAIN_LABEL}: ce_relevance {r.ce_relevance:.3f})")
            elif r.ce_relevance is not None:
                lines.append(f"    ({POINTER_MODE_RETAIN_RANK1_LABEL}; ce_relevance {r.ce_relevance:.3f})")
            else:
                lines.append(f"    ({POINTER_MODE_RETAIN_RANK1_LABEL})")
            lines.append("")
            # UPG-POINTER-RETAIN-STORAGE-CAP-HONESTY: same storage-cap check
            # the full-body path below applies — a retained excerpt can ALSO
            # be a chunk the indexer itself truncated at index time, in which
            # case vectr_fetch would restore the same capped content, not the
            # missing tail. Computed here (not reusing the honest-path
            # locals, which aren't in scope for a `continue`d low_conf
            # branch) so this excerpt never implies a full-symbol restore
            # that vectr_fetch can't deliver.
            s_start = getattr(r, "symbol_start_line", 0)
            s_end = getattr(r, "symbol_end_line", 0)
            truncation_warning = _storage_cap_truncation_warning(
                r.content, workspace_relpath(r.file_path, workspace_root), s_start, s_end,
            )
            content_lines = r.content.splitlines()
            if len(content_lines) > POINTER_MODE_RETAIN_EXCERPT_LINES:
                lines.append("\n".join(content_lines[:POINTER_MODE_RETAIN_EXCERPT_LINES]))
                lines.append(
                    f"... {len(content_lines) - POINTER_MODE_RETAIN_EXCERPT_LINES} more lines — "
                    f"vectr_fetch(ids=[{chunk_id!r}]) restores the full chunk"
                )
                if truncation_warning:
                    lines.append(truncation_warning)
            else:
                lines.append(r.content)
                if truncation_warning:
                    lines.append(truncation_warning)
            lines.append("")
            continue
        lines.append("")
        content_lines = r.content.splitlines()
        # UPG-11.4-b: emit an expand hint when the stored content was truncated
        # by the 2000-char cap (searcher stores content[:2000], ~48 lines for
        # dense methods) — see _storage_cap_truncation_warning().
        # UPG-CTX-EVICT: the display-cap expand hint points at
        # vectr_fetch(ids=[...]) — vectr_fetch returns the chunk's FULL stored
        # content (no 80-line display cap) and keeps the caller in the vectr
        # tool family. The STORAGE-cap branch below must NOT use it: a chunk
        # capped at ~2000 chars at index time is stored capped, so vectr_fetch
        # would return the same truncated content — only a file read reaches
        # the missing tail.
        s_start = getattr(r, "symbol_start_line", 0)
        s_end = getattr(r, "symbol_end_line", 0)
        truncation_warning = _storage_cap_truncation_warning(
            r.content, workspace_relpath(r.file_path, workspace_root), s_start, s_end,
        )
        if len(content_lines) > 80:
            # Hard cap: the content itself is long but we also cap the display.
            lines.append("\n".join(content_lines[:80]))
            lines.append(
                f"... {len(content_lines) - 80} more lines — "
                f"vectr_fetch(ids=[{chunk_id!r}]) restores the full chunk"
            )
        elif truncation_warning:
            # Content was silently capped by the 2000-char storage limit before
            # it reached the full symbol body.  Show what we have, then prompt.
            lines.append(r.content)
            lines.append(truncation_warning)
        else:
            lines.append(r.content)
        lines.append("")
    # UPG-LOWCONF-SLIM-DEDUPE: in pointer mode no body was ever shown, so
    # nothing "left your context" — the footer would be actively misleading.
    # The pointers above are already the fetch keys. UPG-POINTER-MODE-
    # UNIFORM-STRIP: a retained excerpt IS shown content, so the footer
    # applies again whenever at least one result kept a body.
    if not low_conf or any(retain_body):
        lines.append(
            'Results are re-fetchable anytime: vectr_fetch(ids=["<id>"]) restores '
            "a chunk after it leaves your context."
        )
    return "\n".join(lines)


def _relativize_chunk_id(chunk_id: str, workspace_root: str) -> str:
    """Render a `path:start-end` chunk id workspace-relative for display
    (same convention as UPG-RELATIVE-PATH-RENDER's other id renderers). An
    id that doesn't parse that way is returned unchanged."""
    path, sep, line_range = chunk_id.rpartition(":")
    if not sep:
        return chunk_id
    return f"{workspace_relpath(path, workspace_root)}:{line_range}"


def _format_fetch_results(entries: list[dict], workspace_root: str = "") -> str:
    """Render vectr_fetch results using the same id + symbol + content
    conventions as _format_search_results (UPG-CTX-EVICT), so a restored
    chunk reads identically to how it first appeared in a search response.

    UPG-FETCH-TRUNCATION-SILENT: a chunk stored incomplete because of the
    indexer's per-chunk storage cap (see _storage_cap_truncation_warning) is
    STILL stored incomplete when re-fetched by id — vectr_fetch restores the
    exact stored bytes, not the original file. Apply the same truncation
    check search already applies so a re-fetch never silently looks complete
    on exactly the large chunks a caller is most tempted to evict.

    UPG-RELATIVE-PATH-RENDER: a found entry's id is re-rendered workspace-
    relative (from its own file_path, so the display matches search's relative
    ids regardless of whether the caller fetched by a relative or absolute id);
    the absolute root is printed once in the header. A not-found entry echoes
    the exact id the caller passed so a bad id is recognizable.

    UPG-FETCH-ID-MISALIGN-MSG: a not-found entry with `reason="misaligned"`
    (the file has other chunks indexed; the requested span just doesn't
    match one of them) renders its own message naming the nearest stored
    chunk ids — response-packing, so the caller's very next vectr_fetch call
    can succeed without a round trip — and is excluded from the generic
    "file likely changed" footer below, which now fires only for entries
    that are genuinely absent (`reason="file_changed"`, or an older/mocked
    entry with no `reason` at all, defaulting to the same genuine-miss
    handling as before this change).
    """
    lines: list[str] = []
    if workspace_root:
        lines.append(f"workspace: {workspace_root}")
    missing_file_changed = [
        e["id"] for e in entries
        if not e["found"] and e.get("reason", "file_changed") != "misaligned"
    ]
    for e in entries:
        lines.append(f"{'─' * 60}")
        if not e["found"]:
            if e.get("reason") == "misaligned":
                nearest = [_relativize_chunk_id(n, workspace_root) for n in (e.get("nearest_ids") or [])]
                nearest_str = ", ".join(nearest) if nearest else "(none in this file)"
                lines.append(
                    f"[{e['id']}] no chunk covers this exact span — "
                    f"nearest stored chunks are [{nearest_str}]"
                )
            else:
                lines.append(f"[{e['id']}] not found")
            lines.append("")
            continue
        rel_path = workspace_relpath(e.get("file_path", ""), workspace_root)
        rel_id = f"{rel_path}:{e.get('start_line', 0)}-{e.get('end_line', 0)}"
        lines.append(f"[{rel_id}]")
        if e.get("symbol_name"):
            lines.append(f"    symbol: {e['symbol_name']}  language: {e.get('language', '')}")
        lines.append("")
        content = e.get("content", "")
        lines.append(content)
        truncation_warning = _storage_cap_truncation_warning(
            content, rel_path, e.get("start_line", 0), e.get("end_line", 0),
        )
        if truncation_warning:
            lines.append(truncation_warning)
        lines.append("")
    if missing_file_changed:
        from app.service import _FETCH_NOT_FOUND_NOTE
        lines.append(_FETCH_NOT_FOUND_NOTE)
    return "\n".join(lines)


def _arc_age_str(ts: float | None) -> str:
    """Coarse human age string for one arc's timestamp — same rounding
    granularity as the working-memory note age renderer, kept independent
    (arcs are a different, quarantined table) rather than importing that
    module's private helper."""
    if not ts:
        return "unknown age"
    import time as _time
    delta = max(0.0, _time.time() - ts)
    if delta < 3600:
        return f"{int(delta // 60)}m ago"
    if delta < 86400:
        return f"{int(delta // 3600)}h ago"
    return f"{int(delta // 86400)}d ago"


def _format_arc_block(arc: dict) -> str:
    """One arc's render block for vectr_distill()/GET /v1/arcs — rendered
    facts only: the failure chain,
    the resolving success, and the mutation diff. No advice, no suggested
    kind — that judgment is the static rules text surrounding this block,
    never generated per-arc."""
    lines = [
        f"[arc #{arc['id']}] confidence={arc.get('confidence', 'normal')} "
        f"· {_arc_age_str(arc.get('ts'))} · cwd={arc.get('cwd', '')}"
    ]
    for f in arc.get("failures") or []:
        markers = ", ".join(f.get("markers_matched") or []) or "none"
        lines.append(
            f"  failed: {f.get('verb', '')} (outcome={f.get('outcome', '')}, markers={markers})"
        )
    success = arc.get("success")
    if success:
        lines.append(f"  succeeded: {success.get('verb', '')} — {success.get('cmd_raw', '')}")
    mutation_diff = arc.get("mutation_diff") or {}
    if mutation_diff:
        lines.append(f"  mutation diff: {mutation_diff}")
    return "\n".join(lines)


def _format_pending_arcs(rows: list[dict], total_pending: int) -> str:
    """Render vectr_distill()'s no-args response: the fixed distiller-rules
    header (verbatim static text) plus
    as many arc blocks from `rows` as fit under
    EPISODES_DISTILL_RENDER_TOKEN_CAP — the first block always renders even
    if it alone exceeds the cap, so a render never comes back empty. `rows`
    is already ordered confidence-first then oldest-first and capped at
    EPISODES_DISTILL_MAX_ARCS_RENDERED by the caller; `total_pending` (the
    workspace's full pending count) drives the trailing "N more" note when
    either cap held some arcs back."""
    from agent.trigger_engine import token_estimate

    header = _DISTILL_RULES_TEXT
    if not rows:
        return f"{header}\n\nNo arcs pending distillation."
    blocks: list[str] = []
    used_tokens = token_estimate(header)
    for arc in rows:
        block = _format_arc_block(arc)
        block_tokens = token_estimate(block)
        if blocks and used_tokens + block_tokens > EPISODES_DISTILL_RENDER_TOKEN_CAP:
            break
        blocks.append(block)
        used_tokens += block_tokens
    text = header + "\n\n" + "\n\n".join(blocks)
    remaining = total_pending - len(blocks)
    if remaining > 0:
        text += (
            f"\n\n({remaining} more pending arc(s) not shown — call vectr_distill() "
            "again after distilling or dismissing some.)"
        )
    return text


def _mcp_error(message: str) -> dict:
    return {
        "content": [{"type": "text", "text": f"Error: {message}"}],
        "isError": True,
    }
