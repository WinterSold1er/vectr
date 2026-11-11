"""MCP tool schema definitions — exploration, memory write, memory read, utility."""
from __future__ import annotations

from importlib import metadata as _importlib_metadata

from agent.working_context_store import SORT_BY_VALUES


def _package_version() -> str:
    try:
        return _importlib_metadata.version("vectr")
    except _importlib_metadata.PackageNotFoundError:
        return "0.0.0"


MCP_SERVER_INFO = {
    "name": "vectr",
    "version": _package_version(),
    "description": "Zero-config semantic code search + persistent working memory for AI agents",
    "capabilities": {"tools": {}},
}

# Exploration tools: always shown
_EXPLORATION_TOOLS = [
    # ---- L3: content retrieval ----
    {
        "name": "vectr_search",
        "annotations": {
            "title": "Semantic code search",
            "readOnlyHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "description": (
            "Use when you know WHAT you're looking for but not WHERE it is or WHAT it's called. "
            "Hybrid semantic + BM25 search — finds code by concept, behaviour, or description. "
            "Returns function/class bodies with file paths and line numbers. "
            "NOT when you already know the symbol name — use vectr_locate instead. "
            "NOT when you want call relationships — use vectr_trace instead."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language or code description of what you're looking for",
                },
                "n_results": {
                    "type": "integer",
                    "description": (
                        "Number of results to return (default: 5, max: 50). "
                        "Prefer 1–2 for a specific symbol or single-answer lookup "
                        "(\"where is X defined\", \"the function that does Y\") — the top "
                        "hit is usually the answer and fewer results cost far fewer tokens. "
                        "Widen to 5+ only for exploratory or survey queries where several "
                        "distinct implementations are genuinely useful."
                    ),
                    "default": 5,
                },
                "language": {
                    "type": "string",
                    "description": "Filter to a specific indexed language (e.g. python, rust, c, zig). "
                                   "Any language the index contains is accepted; an unindexed "
                                   "language returns no results plus the list of indexed languages.",
                    "nullable": True,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "vectr_fetch",
        "annotations": {
            "title": "Restore code chunks by id",
            "readOnlyHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "description": (
            "Deterministic re-fetch of a code chunk by its exact id — no embedding, "
            "no rerank, just the chunk. Every vectr_search/vectr_locate/vectr_trace "
            "result carries its chunk's id (the `file:start-end` shown in the result "
            "header). Use this to restore a chunk that was cleared from your context "
            "(by tool-result eviction, context compaction, or a context-editing tombstone) "
            "instead of re-running vectr_search or re-reading the whole file. "
            "NOT for finding NEW content — use vectr_search for that."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Chunk ids to restore, exactly as shown in a prior "
                                    "search/locate/trace result (e.g. 'src/auth.py:10-20').",
                },
            },
            "required": ["ids"],
        },
    },
    {
        "name": "vectr_status",
        "annotations": {
            "title": "Index and memory status",
            "readOnlyHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "description": (
            "Returns index health (files, chunks, embed model) AND notes_count (number of notes "
            "stored — earlier in this session or in prior sessions). "
            "Call once at the start of any session to decide whether vectr_recall is worth calling: "
            "if notes_count > 0, call vectr_recall(query=...) to retrieve relevant notes. "
            "If notes_count == 0, skip recall entirely. "
            "If your session already shows auto-injected Working Notes (vectr hooks), those ARE "
            "the recall output — do not re-call vectr_recall for them. "
            "Also useful when vectr_search returns nothing and you suspect indexing is still running."
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    # ---- L1: codebase map ----
    {
        "name": "vectr_map",
        "annotations": {
            "title": "Codebase overview",
            "readOnlyHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "description": (
            "Use at the start of a session on an UNFAMILIAR codebase to get a structural overview "
            "without reading any files. "
            "If a passport has been saved: returns a compact (~300 token) plain-English summary instantly. "
            "If not yet saved: returns raw structural metadata (dir tree, languages, frameworks) "
            "and instructs you to call vectr_map_save with your synthesised summary. "
            "NOT needed if you already know the codebase structure. "
            "NOT a substitute for vectr_recall — call vectr_status first to check for prior notes."
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "vectr_map_save",
        "annotations": {
            "title": "Save codebase passport",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "description": (
            "Save your synthesised codebase summary as the permanent passport. "
            "Call this ONLY after vectr_map returned raw metadata — i.e. on your first visit to a codebase. "
            "NOT when vectr_map already returned a saved summary (passport already exists). "
            "Write a concise plain-English summary: what the codebase does, tech stack, "
            "key modules, entry points, domain terms. Aim for ~200-350 tokens. "
            "Does NOT overwrite an existing passport by default — if one is already saved, the call "
            "is a no-op and returns the existing summary; pass overwrite=true to replace it."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Your plain-English codebase summary (~200-350 tokens)",
                },
                "overwrite": {
                    "type": "boolean",
                    "description": "Set true to replace an already-saved passport. Default false.",
                    "default": False,
                },
            },
            "required": ["summary"],
        },
    },
    # ---- L2: symbol graph ----
    {
        "name": "vectr_locate",
        "annotations": {
            "title": "Locate symbol definition",
            "readOnlyHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "description": (
            "Use when you know the SYMBOL NAME but not which file it's in. "
            "Returns file path + line number + kind for every matching definition. "
            "NOT when you're searching by concept or behaviour — use vectr_search instead. "
            "NOT when you want call relationships — use vectr_trace instead. "
            "Example: vectr_locate(name='EvaluateSegments') → 'targeting/evaluator.go:45'"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Symbol name or partial name to locate (case-sensitive partial match)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default: 10)",
                    "default": 10,
                },
                "caller_file": {
                    "type": "string",
                    "description": "Absolute path of the file containing the call site. "
                                   "Enables same-module and import-chain fallback strategies.",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "vectr_trace",
        "annotations": {
            "title": "Symbol call graph",
            "readOnlyHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "description": (
            "Use when you know the SYMBOL NAME and need to understand its callers or callees "
            "before modifying it. Traverses the call graph in both directions. "
            "NOT when you don't know the symbol name yet — use vectr_search or vectr_locate first. "
            "NOT when you just want the definition location — use vectr_locate instead. "
            "Example: vectr_trace(name='EvaluateSegments') → 'Called by: RequestBid() in bidder/auction.go'"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Symbol name to trace",
                },
                "direction": {
                    "type": "string",
                    "description": "'callers' (who calls this), 'callees' (what it calls), or 'both' (default)",
                    "default": "both",
                    "enum": ["callers", "callees", "both"],
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results per direction (default: 20)",
                    "default": 20,
                },
                "include_builtins": {
                    "type": "boolean",
                    "description": (
                        "Include language builtins/stdlib in the 'calls' list (len, assert, "
                        "Ok, Some, malloc, …). Default false — only repo-internal calls are "
                        "shown, with a count of how many builtins were hidden."
                    ),
                    "default": False,
                },
            },
            "required": ["name"],
        },
    },
]  # end _EXPLORATION_TOOLS

# Distiller rules — the static prompt
# surface for arc distillation. Defined once here and reused verbatim in
# both the vectr_distill tool description below and the fixed header line
# _dispatch.py prepends to every vectr_distill() render — never generated
# from arc content, never gated on it.
_DISTILL_RULES_TEXT = (
    "Distiller rules: (1) recall-first dedupe — vectr_recall(query=<the lesson>) before "
    "writing; if an existing note already covers it, dismiss the arc (reason: covered by "
    "note #N) — if the arc proves an existing note wrong or outdated, use contradicts=/"
    "supersedes= on that note instead of writing a duplicate. "
    "(2) Generalize — store the lesson (what class of command fails, why, what fixed it), "
    "not the transcript; keep concrete commands/paths only when they ARE the lesson. "
    "(3) Kind mapping — an env/process/build fact -> kind='operational' (add triggers=[{'event': "
    "'prompt-submit', 'semantic': True}, {'command': '<verb glob>'}] when the lesson is tied "
    "to a command family); tied to specific files -> kind='gotcha' with those files as "
    "anchors; a standing user rule -> kind='directive' (rare from arcs). "
    "(4) Low-confidence arcs are dismissed unless the same lesson recurs across >= 2 arcs. "
    "(5) Priority defaults to medium; high only when acting on the stale belief is costly "
    "(e.g. false-pass verification traps). "
    "(6) Batch cap — distill at most ~5 notes per sitting; leave the rest pending."
)

# Always-on memory write tools — visible from turn 1, no notes required.
# vectr_remember: only way to create notes; hiding it is a catch-22.
# vectr_evict_hint: fires on retrieval pressure, not note count.
_MEMORY_WRITE_TOOLS = [
    {
        "name": "vectr_remember",
        "annotations": {
            "title": "Store working-memory note",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False,
        },
        "description": (
            "Save a working note and recall it on demand in <50ms — "
            "whether later this session, through context compaction, or in a future session. "
            "Use the moment you discover something non-obvious: a key file path, a call pattern, a gotcha, "
            "a partial stub, task progress. "
            "Store the actual code or finding — vectr returns it in <50ms; "
            "re-reading the file costs tokens and turns. "
            "Do NOT store obvious or easily re-derivable facts (e.g. 'the main file is main.py'). "
            "For a body over ~2KB, especially code-heavy with quotes/escapes, write it to a file and pass "
            "content_file instead of content — long escape-dense strings can be corrupted mid-stream as a "
            "tool-call argument. "
            "Retrieve with vectr_recall(query='what you need') — any time, same session or later."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": (
                        "The note to store. Store whatever you would need to avoid re-reading the file later. "
                        "If you found a function you'll call or modify — paste its signature and body. "
                        "If you found a pattern you'll need to replicate — paste the pattern. "
                        "If you found a location — include the file:line AND the relevant excerpt, not just the pointer. "
                        "Prose descriptions send the next conversation back to the file; actual code does not. "
                        "Mutually exclusive with content_file (see below); pass exactly one."
                    ),
                },
                "content_file": {
                    "type": "string",
                    "description": (
                        "Path to a UTF-8 file containing the note body — use instead of content for a body over "
                        "~2KB, especially code-heavy with quotes/escapes, so it never has to stream as a long JSON "
                        "string argument. Absolute, or relative to the workspace root; rejected if it resolves "
                        "outside the workspace. Mutually exclusive with content; pass exactly one."
                    ),
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Topic tags for later recall (e.g. ['segment-targeting', 'wip'])",
                },
                "priority": {
                    "type": "string",
                    "description": (
                        "Note priority: 'high' | 'medium' (default) | 'low'. "
                        "Session-start boot injection and the resume surface show only "
                        "priority='high' kind='task' notes, so a checkpoint you want "
                        "picked up at resume needs priority='high'."
                    ),
                    "default": "medium",
                    "enum": ["high", "medium", "low"],
                },
                "kind": {
                    "type": "string",
                    "description": (
                        "Memory kind, controlling how the note is injected (default 'finding'): "
                        "'directive' = a must-never-miss rule, injected unconditionally every session; "
                        "'task' = current-work context (save checkpoints with priority=\"high\" — "
                        "session-start/resume surfaces show only high-priority task notes); "
                        "'gotcha' = a file/path-anchored caveat; "
                        "'finding' = a relevance-ranked learning; 'reference' = a pointer (URL/ticket); "
                        "'decision' = an architectural/design decision plus its why — not auto-injected, "
                        "recall the group chronologically with vectr_recall(kind=\"decision\", "
                        "sort_by=\"chronological\") for an ADR-style decision history; "
                        "'operational' = a build/env/process fact (a build quirk, a CI gotcha, "
                        "feedback-loop knowledge — e.g. 'tests must run via ./.venv/bin/python'), not "
                        "anchored to a single code file the way 'gotcha' is. By default surfaces via "
                        "prompt-time semantic recall (equivalent to declaring triggers=[{'event': "
                        "'prompt-submit', 'semantic': True}]). An explicit triggers=[...] REPLACES that "
                        "default rather than adding to it — declaring triggers=[{'command': "
                        "'<verb-glob>'}] alone means the note fires ONLY right before a matching shell "
                        "command and stops firing on prompt-submit. To keep both surfaces, declare both "
                        "entries: triggers=[{'event': 'prompt-submit', 'semantic': True}, {'command': "
                        "'<verb-glob>'}]. "
                        "When a new kind='task' note replaces an earlier checkpoint (the work moved on, "
                        "the old note is no longer the current state), pass supersedes=<old note_id> so "
                        "the stale checkpoint stops firing at every future session-start instead of "
                        "piling up alongside the new one."
                    ),
                    "default": "finding",
                    "enum": ["directive", "task", "gotcha", "finding", "reference", "decision", "operational"],
                },
                "title": {
                    "type": "string",
                    "description": (
                        "Short label for index-tier display (optional, max ~80 chars). "
                        "If omitted, the first non-empty line of content is used as the title. "
                        "Shown in vectr_recall() index output so you can identify notes without reading their bodies."
                    ),
                    "default": "",
                },
                "agent": {
                    "type": "string",
                    "description": (
                        "Optional: your identifier when called by a subagent or orchestrator in a "
                        "multi-agent workflow (e.g. 'coder-2'). Never inferred — set it explicitly if "
                        "you want attribution. Shown in vectr_recall() index output as a tag, e.g. "
                        "'[#12] task/high (coder-2) · title'. A subagent should call vectr_remember "
                        "with its findings BEFORE finishing so the orchestrator can recall them "
                        "instead of re-reading the subagent's full transcript."
                    ),
                    "default": "",
                },
                "triggers": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": (
                        "Advanced: explicit overrides for WHEN this note resurfaces (path glob, "
                        "event, symbol, semantic-similarity, timing/cooldown fields — AND within "
                        "an entry, OR across entries; see docs for the full field DSL). Omit this "
                        "entirely (recommended) — each kind already gets a sensible default, e.g. "
                        "'directive' fires at session-start and after compaction, 'gotcha' fires "
                        "when its anchored file is about to be edited."
                    ),
                },
                "provenance": {
                    "type": "string",
                    "description": (
                        "How much to trust this note when it resurfaces (default 'agent'): "
                        "'agent' = self-recorded, framed as memory to verify; 'auto' = no "
                        "reviewing judgment, weakest framing, incompatible with kind='directive'. "
                        "'human' is only reached via explicit promotion, not settable here."
                    ),
                    "default": "agent",
                    "enum": ["agent", "auto"],
                },
                "user_quote": {
                    "type": "string",
                    "description": (
                        "Optional: when this note transcribes something the USER said, the "
                        "user's own words, verbatim. Bound only if that exact text also "
                        "appears inside `content` (whitespace-insensitive substring check); "
                        "a bound quote stores the note as provenance='user-stated' so it "
                        "resurfaces as the user's statement rather than your own recollection. "
                        "Paraphrase, or words the user did not actually write, will not bind "
                        "— the note is stored as an ordinary 'agent' note and the reason is "
                        "returned. Never fails the write."
                    ),
                },
                "scope": {
                    "type": "string",
                    "description": (
                        "Advanced: visibility scope, enforced at recall/trigger time. Omit for "
                        "the kind's own default ('task'→'branch', 'gotcha'→'repo', else "
                        "'workspace'); pass explicitly to override — 'branch' (git branch at "
                        "write time), 'path-subtree' (paths under an anchored dir), 'session' "
                        "(writing session only), 'repo' (same as 'workspace' today)."
                    ),
                    "enum": ["workspace", "repo", "path-subtree", "branch", "session"],
                },
                "anchors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional: file paths this note is about, content-hashed at write time so "
                        "a later change surfaces as a staleness caveat instead of silently going stale."
                    ),
                },
                "supersedes": {
                    "type": "integer",
                    "description": (
                        "Optional: the note_id this new note replaces — the old note is retired "
                        "(excluded from recall/firing) but kept for audit, unlike vectr_forget. "
                        "Especially important for kind='task': pass the prior checkpoint's id so "
                        "it stops firing at every session-start once superseded."
                    ),
                },
                "contradicts": {
                    "type": "integer",
                    "description": (
                        "Optional: the note_id this new note proves WRONG — distinct from "
                        "supersedes (a normal replacement). The target note is revoked: it "
                        "stays visible on every future recall/fire but rendered as a deterrent "
                        "('previously believed... do not re-derive without verification') "
                        "instead of its raw content, until vectr_reinstate reverses it."
                    ),
                },
                "distilled_from": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": (
                        "Optional: arc ids (from vectr_distill() or GET /v1/arcs) this note "
                        "distills. After this note is stored, each named arc is marked "
                        "distilled into it. Unknown/already-resolved ids, and any entry that "
                        "is not itself an integer, are reported back in the confirmation, "
                        "never an error and never silently dropped."
                    ),
                },
                "pin": {
                    "type": "boolean",
                    "description": (
                        "Optional: pin this note into Tier 0 at write time — injected on "
                        "EVERY future vectr_recall(query=...) call regardless of the query, "
                        "same effect as a separate vectr_pin call right after this write. "
                        "Bounded, so pin sparingly. Default false."
                    ),
                    "default": False,
                },
            },
            # UPG-REMEMBER-MCP-LONG-PAYLOAD-PARSE-LOSS: content is no longer
            # unconditionally required at the schema level — content_file is
            # a valid alternative body source. Exactly one of content /
            # content_file is enforced server-side (resolve_remember_content),
            # with an actionable error naming which rule was broken.
            "required": [],
        },
    },
    {
        "name": "vectr_evict_hint",
        "annotations": {
            "title": "List re-retrievable chunks",
            "readOnlyHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "description": (
            "Lists the code chunks retrieved by THIS session that are safe to drop from "
            "context — each is re-fetchable verbatim in one deterministic call, and the "
            "response includes the exact vectr_fetch(ids=[...]) re-fetch keys. "
            "Use at the exploration → implementation transition, or when context pressure builds. "
            "This is the reverse signal in the vectr protocol: "
            "the AI saves findings (vectr_remember), "
            "vectr signals what it can restore instantly (vectr_evict_hint). "
            "NOT needed on short sessions — most useful after many vectr_search/vectr_locate calls."
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "vectr_distill",
        "annotations": {
            "title": "Review and distill pending arcs",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "description": (
            "Review pending arcs — discovered failure->success moments captured automatically "
            "from your own tool calls (e.g. a command that failed twice, then succeeded after an "
            "edit) — and turn the ones worth keeping into working-memory notes. "
            "Call with no arguments to render pending arcs for review, oldest and most-confident "
            "first, token-bounded. For each arc worth keeping, call "
            "vectr_remember(..., distilled_from=[arc_id]) to persist the lesson as a note (see "
            "vectr_remember's own parameters for kind/priority/triggers). For arcs not worth "
            "keeping, call vectr_distill(dismiss=[arc_id, ...], reason='...'). "
            + _DISTILL_RULES_TEXT
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "dismiss": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": (
                        "Arc ids (from this tool's own render or GET /v1/arcs) to dismiss "
                        "without distilling into a note. Pass together with reason."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": "Required together with dismiss= — why these arcs are not worth distilling (e.g. 'covered by note #12', 'transient network flake').",
                },
            },
            "required": [],
        },
    },
]  # end _MEMORY_WRITE_TOOLS

# Memory read/manage tools: only useful when notes already exist.
# Exposed once notes_count > 0 or vectr_remember has been called this session.
_MEMORY_TOOLS = [
    {
        "name": "vectr_recall",
        "annotations": {
            "title": "Recall working-memory notes",
            "readOnlyHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "description": (
            "Retrieve notes stored earlier in this session or in prior sessions. "
            "TWO-TIER RECALL (UPG-RECALL-HIERARCHY): "
            "By default returns a crisp one-line index per note (id + kind/priority + title + age) — "
            "token-bounded, safe to call broadly. "
            "To read a note body: pass note_id=N (expand one note, full body) or detail='full' (all bodies). "
            "Use when vectr_status() confirmed notes_count > 0 — notes may have been stored this session "
            "or in a previous one; either way they are immediately useful. "
            "Pass a targeted query to retrieve only the notes relevant to your current task — "
            "do NOT call with no query unless you need everything."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural language query to retrieve only relevant notes "
                        "(e.g. 'set cartesian product frozenset' returns notes about that task only). "
                        "Omit only when you need all stored notes."
                    ),
                    "nullable": True,
                },
                "note_id": {
                    "type": "integer",
                    "description": (
                        "Expand a single note by ID — returns the full body of that note, ignoring query. "
                        "Use after seeing the index output to read the note you care about. "
                        "Get IDs from the [#N] prefix in index output."
                    ),
                    "nullable": True,
                },
                "detail": {
                    "type": "string",
                    "description": (
                        "Detail level: "
                        "'index' (default) = one-line summary per note (id, kind/priority, title, age) — token-bounded; "
                        "'full' = full note bodies (use when you need to read all matching notes)."
                    ),
                    "default": "index",
                    "enum": ["index", "full"],
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by tags",
                },
                "priority": {
                    "type": "string",
                    "description": "Filter by priority: 'high' | 'medium' | 'low'",
                    "nullable": True,
                },
                "kind": {
                    "type": "string",
                    "description": "Filter to one memory kind: 'directive' | 'task' | 'gotcha' | 'finding' | 'reference' | 'decision' | 'operational'",
                    "nullable": True,
                    "enum": ["directive", "task", "gotcha", "finding", "reference", "decision", "operational"],
                },
                "sort_by": {
                    "type": "string",
                    "description": (
                        "Sort order: 'relevance' (semantic/trust order, default), "
                        "'recency' (newest first), 'priority' (high→medium→low then newest), "
                        "'chronological' (oldest first — index lines show the creation date instead "
                        "of a relative age; combine with kind=\"decision\" for an ADR-style decision "
                        "timeline, or with any other kind/tag filter for the same time-ordered view)."
                    ),
                    "default": "relevance",
                    # UPG-SORTBY-SHARED-VOCAB: same SORT_BY_VALUES constant the
                    # REST validator, MCP dispatch guard, and CLI argparse
                    # choices consume. JSON Schema needs a JSON-native list,
                    # not a tuple, so we list()-cast at the boundary; the
                    # vocabulary itself is still the shared constant.
                    "enum": list(SORT_BY_VALUES),
                },
                "max_age_days": {
                    "type": "number",
                    "description": "Time filter: only return notes created within this many days.",
                    "nullable": True,
                },
                "boot": {
                    "type": "boolean",
                    "description": (
                        "Boot mode: return ALL directives + high-priority task notes unconditionally "
                        "(no semantic filter, safe on a fresh workspace). Ignores query/tags/priority/kind/limit."
                    ),
                    "default": False,
                },
                "limit": {
                    "type": "integer",
                    "description": "Max notes to return (default: 10)",
                    "default": 10,
                },
            },
            "required": [],
        },
    },
    {
        "name": "vectr_snapshot",
        "annotations": {
            "title": "Checkpoint working memory",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False,
        },
        "description": (
            "Seal all current vectr_remember() notes as a named checkpoint. "
            "Use when you've stored multiple notes and want to mark a milestone you can return to "
            "(e.g. 'auth-refactor-wip', 'segment-targeting-done'). "
            "The next time you work on this, vectr_recall will return these notes. "
            "NOT required if you only stored 1-2 notes — vectr_recall retrieves all notes regardless."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "description": "Human-readable label for this snapshot (e.g. 'segment-targeting-wip')",
                },
                "session_id": {
                    "type": "string",
                    "description": "Optional session identifier",
                    "nullable": True,
                },
            },
            "required": ["label"],
        },
    },
    {
        "name": "vectr_snapshot_list",
        "annotations": {
            "title": "List memory checkpoints",
            "readOnlyHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "description": (
            "List all saved session snapshots for this workspace, newest first. "
            "Use at session start to find an existing checkpoint if vectr_recall returned nothing "
            "or if you want to resume a specific named session."
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "vectr_resume",
        "annotations": {
            "title": "Pick up where you left off",
            "readOnlyHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "description": (
            "One-call 'pick up where you left off': the most recent current-task note, the "
            "latest saved snapshot, and any open gotchas with their file anchors. Call at the "
            "start of a session or after a gap to reorient without re-reading files or "
            "re-deriving state vectr_recall would otherwise need several calls to assemble."
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "vectr_forget",
        "annotations": {
            "title": "Delete working-memory notes",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "description": (
            "Delete working-memory notes. Pass note_id to delete ONE note — the usual case, "
            "when a note is stale or superseded (ids are the [#N] shown by vectr_recall). "
            "Pass all=true to irreversibly clear EVERY note for this workspace (e.g. after a "
            "large refactor). Calling with no arguments deletes nothing. "
            "Snapshots are preserved — only active notes are removed."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "note_id": {
                    "type": "integer",
                    "description": "ID of the single note to delete (the [#N] id from vectr_recall)",
                },
                "all": {
                    "type": "boolean",
                    "description": "Set true to delete ALL notes for this workspace. Irreversible.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "vectr_promote",
        "annotations": {
            "title": "Promote a note's provenance",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "description": (
            "Raise an auto-captured note's trust class to 'agent' — e.g. after this session "
            "has reviewed an auto-captured note and confirmed it still holds. This tool only "
            "takes that one step (auto -> agent); it never promotes a note to 'human', because "
            "deciding that a person has endorsed something is not the agent's call to make. "
            "Human endorsement happens on a user-side surface instead (a CLI/UI a person "
            "operates), not through this tool."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "note_id": {
                    "type": "integer",
                    "description": "ID of the note to promote (the [#N] id from vectr_recall)",
                },
                "to": {
                    "type": "string",
                    "description": "Target provenance. Only 'agent' is available via this tool (auto -> agent); promoting to 'human' is a user-side action, not available here.",
                    "enum": ["agent"],
                },
            },
            "required": ["note_id", "to"],
        },
    },
    {
        "name": "vectr_revoke",
        "annotations": {
            "title": "Revoke a note",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "description": (
            "Flag a stored note as WRONG without deleting it — use when you've confirmed a "
            "prior finding no longer holds (contradicted by newer evidence, or you got it "
            "wrong the first time). Unlike vectr_forget, the note is not erased: it stays "
            "visible on future vectr_recall/session-start as a deterrent — 'previously "
            "believed..., revoked..., do not re-derive this without verification' — instead "
            "of its original content, so nothing silently repeats the mistake. Reversible "
            "with vectr_reinstate. Prefer passing contradicts=<note_id> directly to "
            "vectr_remember when you're recording the correction anyway; use this tool when "
            "you need to revoke without writing a replacement note."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "note_id": {
                    "type": "integer",
                    "description": "ID of the note to revoke (the [#N] id from vectr_recall)",
                },
                "reason": {
                    "type": "string",
                    "description": "Why this note is being revoked (shown verbatim in the deterrent framing).",
                },
            },
            "required": ["note_id", "reason"],
        },
    },
    {
        "name": "vectr_reinstate",
        "annotations": {
            "title": "Reinstate a revoked note",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "description": (
            "Reverse a prior vectr_revoke (or a contradicts= write) — the note returns to "
            "active state and its original content, not the deterrent block, is shown again."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "note_id": {
                    "type": "integer",
                    "description": "ID of the revoked note to reinstate (the [#N] id from vectr_recall)",
                },
                "reason": {
                    "type": "string",
                    "description": "Optional: why this note is being reinstated.",
                },
            },
            "required": ["note_id"],
        },
    },
    {
        "name": "vectr_supersede",
        "annotations": {
            "title": "Retire a note as superseded (post-hoc)",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "description": (
            "Retire an ALREADY-STORED note as superseded, when you discover only AFTER "
            "writing its replacement that supersedes=<old id> was not passed on that "
            "write. The retired note stops appearing in default recall/session-start "
            "and renders with a factual '[superseded ...]' badge — it is NOT marked "
            "wrong and never shows the revoked deterrent (it was accurate when "
            "written; the world moved on). Use vectr_revoke instead when the note is "
            "actually FALSE. Pass superseded_by=<note_id> to link the replacement if "
            "one exists; omit it to retire with no successor. Reversible with "
            "vectr_reinstate."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "note_id": {
                    "type": "integer",
                    "description": "ID of the note to retire (the [#N] id from vectr_recall)",
                },
                "superseded_by": {
                    "type": "integer",
                    "description": (
                        "Optional: ID of the replacement note that already exists "
                        "(the write that should have carried supersedes=<this note>)."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": "Optional: why this note is being retired as superseded.",
                },
            },
            "required": ["note_id"],
        },
    },
    {
        "name": "vectr_pin",
        "annotations": {
            "title": "Pin/unpin a note into Tier 0",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "description": (
            "Pin a note so it is injected on EVERY future vectr_recall(query=...) call, "
            "regardless of the query — never relevance-scored, never dropped for being "
            "off-topic. Use for a note that must never be missed by an unlucky query "
            "(the same standing-rule role kind='directive' already plays, but for a note "
            "you don't want to reclassify as a directive). Pass pinned=false to unpin. "
            "Bounded: only a small configured number of directive+pinned notes ever "
            "occupy this unconditional tier, so pin sparingly. Equivalent to passing "
            "pin=true directly to vectr_remember at write time."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "note_id": {
                    "type": "integer",
                    "description": "ID of the note to pin/unpin (the [#N] id from vectr_recall)",
                },
                "pinned": {
                    "type": "boolean",
                    "description": "true to pin (default), false to unpin",
                },
            },
            "required": ["note_id"],
        },
    },
    {
        "name": "vectr_anchor",
        "annotations": {
            "title": "Attach anchor paths to a stored note",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "description": (
            "Attach file paths to an EXISTING note so future changes to those "
            "files can flag the note as possibly-stale (UPG-ANCHOR-ATTACH). "
            "The single-call replacement for re-storing with anchors=[...] and "
            "supersedes=<note_id> when you notice afterwards that a note is "
            "about a specific process/config file. Anchors are staleness "
            "probes, never a claim that the note is wrong: on the next "
            "staleness check, a changed anchor means the process MAY have "
            "changed. Idempotent — paths already attached are reported back, "
            "not duplicated."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "note_id": {
                    "type": "integer",
                    "description": "ID of the note to attach anchors to (the [#N] id from vectr_recall)",
                },
                "anchors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Workspace-relative file paths to anchor this note to (at least one)",
                },
            },
            "required": ["note_id", "anchors"],
        },
    },
    {
        "name": "vectr_unanchor",
        "annotations": {
            "title": "Remove anchor paths from a stored note",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "description": (
            "Remove file paths from an EXISTING note's anchor set "
            "(UPG-ANCHOR-DETACH). The inverse of `vectr_anchor`: use it when "
            "you discover a note was anchored to the wrong file, or when the "
            "anchored file's relevance has gone away. A removed anchor simply "
            "stops being a candidate in the next staleness check — never a "
            "claim that the note is wrong. Idempotent — paths the note was "
            "never anchored to are reported back, not treated as failures. "
            "Path comparison is exact-string on the spelling the note was "
            "anchored with, matching `vectr_anchor`'s contract."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "note_id": {
                    "type": "integer",
                    "description": "ID of the note to remove anchors from (the [#N] id from vectr_recall)",
                },
                "anchors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Workspace-relative file paths to detach this note from (at least one)",
                },
            },
            "required": ["note_id", "anchors"],
        },
    },
]  # end _MEMORY_TOOLS

# ingest_traces — not gated by session memory (always available)
_UTILITY_TOOLS = [
    {
        "name": "vectr_ingest_traces",
        "annotations": {
            "title": "Ingest runtime call edges",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False,
        },
        "description": (
            "Import runtime trace events into the symbol graph to enrich static call analysis. "
            "Use when you have runtime profiling data (Python sys.settrace output, JSON trace logs) "
            "that reveals dynamic dispatch patterns the static analyser cannot see: decorators, "
            "__getattr__, dependency injection, monkey-patching, etc. "
            "Pass a list of trace events: [{caller, callee, caller_file?, caller_line?}, ...]. "
            "Dynamic edges are stored with edge_type='dynamic' and appear in vectr_trace results "
            "marked \"(dynamic)\" so you can tell them apart from statically-discovered calls. "
            "A caller/callee name that matches no indexed symbol is still ingested (it may be "
            "external or runtime-only) but is reported back as a warning — check for typos. "
            "NOT needed if static analysis (vectr_trace) already shows the call relationships."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "events": {
                    "type": "array",
                    "description": "List of trace events. Each event: {caller, callee, caller_file?, caller_line?}",
                    "items": {
                        "type": "object",
                        "properties": {
                            "caller":      {"type": "string", "description": "Calling function/symbol name"},
                            "callee":      {"type": "string", "description": "Called function/symbol name"},
                            "caller_file": {"type": "string", "description": "Source file of the caller"},
                            "caller_line": {"type": "integer", "description": "Line number of the call site"},
                        },
                        "required": ["caller", "callee"],
                    },
                },
            },
            "required": ["events"],
        },
    },
]

# Full list for serialization / backwards compat
MCP_TOOLS = _EXPLORATION_TOOLS + _MEMORY_WRITE_TOOLS + _MEMORY_TOOLS + _UTILITY_TOOLS

# UPG-STDIO-MEMORY-READY: tools backed entirely by WorkingContextStore (SQLite
# write path + best-effort embedding). These never need the embedder,
# indexer, searcher, watcher, or symbol graph, so they are servable as soon
# as phase 1 of VectrService construction completes -- long before phase 2
# (model load, indexing) finishes. Transports key dispatch on the TOOL NAME
# below and on service readiness state only, never on query text content.
MEMORY_READY_TOOLS = frozenset(
    {
        "vectr_remember",
        "vectr_recall",
        "vectr_forget",
        "vectr_promote",
        "vectr_revoke",
        "vectr_reinstate",
        "vectr_supersede",
        "vectr_pin",
        "vectr_anchor",
        "vectr_unanchor",
        "vectr_status",
        "vectr_snapshot",
        "vectr_snapshot_list",
        "vectr_resume",
        "vectr_distill",
    }
)
