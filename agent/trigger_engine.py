"""
Trigger engine (TRIGGER-ENGINE).

vectr's working memory is not just a note store — each memory declares WHEN it
is relevant to resurface, so the caller LLM does not have to guess or ask for
it. This module is the pure, deterministic core of that mechanism: P (path),
E (event), S (symbol), and M (semantic) trigger primitives, T (temporal)
modifiers, composition (conjunction within one trigger, disjunction across a
note's `triggers[]`), kind-default bundles, one shared total order (fire
precedence, injection ordering, and budget eviction all reuse the SAME
function), a per-session fire ledger, and the two-tier injection budget/pack.

Hard invariant (no-query-heuristics rule): every function here operates ONLY
on a memory's own declared trigger data plus tool/lifecycle state that the
CALLER already resolved (event name, a workspace-relative file path, a
caller-resolved symbol-graph set, a precomputed semantic-match boolean, a
clock reading). Nothing in this module ever reads a user prompt or query
string, parses text, or touches a vector — there is no such parameter
anywhere below; S is exact set-membership against a symbol the caller already
resolved, and M is a single precomputed boolean the caller already derived
from a cosine-vs-threshold check (agent/working_context_store/_store.py's
`fire()` — the only place a prompt or a vector is ever touched).

Write-time contradiction detection remains out of scope here.

Live delivery surface: `evaluate_note()`/`fire()` below and
`WorkingContextStore.fire()`/`fire_and_format()`
(agent/working_context_store/_store.py), driven by the hook branches of
`VectrService.recall` (app/service.py), are wired into the live hook
pipeline. (`VectrService.fire_triggers()`/`fire_and_recall()` exist but have
no live caller — not part of this surface.)
`main.py`'s `cmd_hook` maps its own hook-name vocabulary (`session-start`,
`user-prompt-submit`, `pre-tool-use`, `pre-compact`) onto this module's
EVENT_VALUES (`session-start`, `prompt-submit`, `pre-edit`, `pre-run`,
`pre-commit`, `post-compaction`) and calls `/v1/recall` (boot mode, now
engine-driven) and `/v1/trigger/reset` (PreCompact) — see `cmd_hook`'s
docstring for the exact mapping table and rationale. `pre-run` is live via
the pre-tool-use command lane (cmd_hook posts a Bash call and the service's
command branch fires event="pre-run" with the normalized verb); `pre-commit`
alone remains declared-but-inert (no lifecycle moment maps to it — never an
error): a note with an explicit `triggers=[{"event": "pre-commit"}]`
override simply never fires today, exactly like an unrecognised kind falls
back to its default.
"""
from __future__ import annotations

import fnmatch
import time
from dataclasses import dataclass, field

from agent.config import (
    MEMORY_TRIGGER_BODY_TRUNCATION_MIN_BOUNDARY_FRACTION,
    MEMORY_TRIGGER_CHARS_PER_TOKEN,
    MEMORY_TRIGGER_KIND_PRIORITY,
    MEMORY_TRIGGER_PER_INJECTION_TOKEN_CAP,
    MEMORY_TRIGGER_PER_KIND_TOKEN_CAP,
    MEMORY_TRIGGER_PER_SESSION_TOKEN_CAP,
    MEMORY_TRIGGER_PER_TURN_TOKEN_CAP,
    MEMORY_TRIGGER_PRIORITY_RANK,
)
from agent.working_context_store._types import (
    DEFAULT_SCOPE,
    EVENT_VALUES,
    USER_STATED_PROVENANCE,
    WorkingNote,
)

# Kinds that inject full-text under the two-tier injection budget; every other
# kind injects its index-tier one-liner. Not config.yaml: this is the kind
# taxonomy's own injection policy, the same closed-vocabulary category as
# VALID_KINDS/EVENT_VALUES, not a tunable weight.
FULL_TEXT_KINDS: tuple[str, ...] = ("directive", "gotcha", "operational")


# ---------------------------------------------------------------------------
# Trigger validation (write-time)
# ---------------------------------------------------------------------------

def validate_trigger(trigger: dict) -> None:
    """Raise ValueError if `trigger` is not a well-formed P/S/M/E/T/command primitive.

    A trigger must declare at least one of 'path' (P), 'event' (E), 'symbol'
    (S), 'semantic' (M), or 'command' (UPG-MEMORY-STATE-MACHINE §5.2) —
    T (not_before/expires_visibility/cooldown) is a modifier only and can
    never fire a trigger by itself.

    'symbol' names a code symbol resolved at fire time against the pre-built
    code symbol graph (the same store `vectr_locate`/`vectr_trace` use) — it
    matches when that symbol is defined in, or referenced by, the file
    targeted by the current lifecycle moment.

    'semantic', when true, declares the M axis: at prompt-submit, the note
    matches when cosine(activity embedding, note's own stored embedding) is
    at or above a fixed per-kind threshold (config.yaml
    `memory_triggers.semantic.theta_by_kind`) — the caller (which has the
    embedder) computes that boolean; this module never touches the vector or
    the prompt text itself (no-query-heuristics rule).

    'command' is a glob string matched against the NORMALIZED VERB of a
    Bash tool call about to run (e.g. "pytest*", "./mvnw*") — the same
    fnmatch mechanism 'path' already uses, against tool-call structure the
    caller already resolved via `app.cmdnorm.normalize_command()` (the one
    shared normalizer; this module never parses a raw command string
    itself). Tool-call argv classification is sanctioned by the
    no-query-heuristics rule (only prompt/query CONTENT classification is
    forbidden) — see agent/trigger_engine.py's module docstring.

    All four (path/symbol/semantic/command) compose with 'event' under the
    SAME conjunction rule (every declared axis in one trigger dict must ALL
    match); a trigger may declare at most one value per axis by
    construction (each axis is a single dict key)."""
    if not isinstance(trigger, dict):
        raise ValueError(
            "each trigger must be an object with 'path', 'event', 'symbol', "
            "'semantic', and/or 'command' keys"
        )
    path = trigger.get("path")
    event = trigger.get("event")
    symbol = trigger.get("symbol")
    semantic = trigger.get("semantic")
    command = trigger.get("command")
    if path is None and event is None and symbol is None and semantic is None and command is None:
        raise ValueError(
            "a trigger must declare at least one of 'path', 'event', 'symbol', "
            "'semantic', or 'command' — T (not_before/expires_visibility/cooldown) "
            "is a modifier only and never fires alone"
        )
    if path is not None and not isinstance(path, str):
        raise ValueError("trigger 'path' must be a glob string")
    if event is not None and event not in EVENT_VALUES:
        raise ValueError(f"trigger 'event' must be one of: {', '.join(EVENT_VALUES)}")
    if symbol is not None and (not isinstance(symbol, str) or not symbol):
        raise ValueError("trigger 'symbol' must be a non-empty string naming a code symbol")
    if semantic is not None and not isinstance(semantic, bool):
        raise ValueError("trigger 'semantic' must be a boolean")
    if command is not None and (not isinstance(command, str) or not command):
        raise ValueError("trigger 'command' must be a non-empty glob string naming a command verb")
    for key in ("not_before", "expires_visibility", "cooldown"):
        value = trigger.get(key)
        if value is not None and not isinstance(value, (int, float)):
            raise ValueError(f"trigger '{key}' must be a number (epoch seconds, or seconds for cooldown)")


def validate_triggers(triggers: list[dict] | None) -> list[dict]:
    """Validate every trigger in a note's declared `triggers[]`. Returns a
    plain list (empty when `triggers` is None/empty — that note falls back to
    its kind's default bundle at evaluation time; explicit triggers[] fully
    REPLACE the default bundle, they never merge with it)."""
    if not triggers:
        return []
    for trig in triggers:
        validate_trigger(trig)
    return list(triggers)


# ---------------------------------------------------------------------------
# Kind-default bundles
# ---------------------------------------------------------------------------

def default_bundle_for_kind(
    kind: str,
    anchors: list[list[str]] | None,
    priority: str | None = None,
) -> list[dict]:
    """The trigger bundle a note gets when it declares no explicit `triggers[]`
    override, computed fresh at evaluation time (never baked into storage, so
    a future default-bundle change applies retroactively).

    - directive: fires at session-start AND post-compaction (must-never-miss),
                 at ANY priority — matches the legacy `boot_recall()`'s own
                 unfiltered directive query.
    - task:      fires at session-start (current-work state), until closed —
                 but ONLY at priority=='high':
                 the legacy `boot_recall()` this bundle replaces filters task
                 notes with `AND priority = 'high'` in SQL; a medium/low
                 priority task gets NO default trigger here (an explicit
                 `triggers[]` override can still make it fire — this only
                 narrows the IMPLICIT default to match what boot_recall()
                 has always surfaced, not a new restriction on the note
                 itself). `priority` is a caller-resolved note property
                 (never a query string) — same category as `anchors` below.
    - gotcha:    one path trigger per declared anchor, at pre-edit — a
                 symbol-ref default is deliberately absent (the S primitive
                 exists but no kind's default bundle uses it). A gotcha
                 note with NO structured anchors gets no default bundle here
                 (empty list) — it continues to be served exactly as today by
                 WorkingContextStore.recall_for_path()'s unrelated content-
                 substring match, left untouched by this engine.
    - finding/reference/decision: M-primitive territory (θ-gated semantic
                 trigger) — no default bundle. Their current
                 relevance-rank recall() injection is unchanged and
                 unaffected by this engine. `decision` (UPG-DECISION-
                 TIMELINE) is deliberately NOT boot-privileged — an
                 architectural decision is meant to be pulled on demand as a
                 group (vectr_recall(kind="decision",
                 sort_by="chronological")), not pushed into every session.
    - operational: env/process/build facts (UPG-MEMORY-STATE-MACHINE §5.1).
                 Primary surface is prompt-time semantic matching — the
                 SAME θ-gated M primitive as finding/reference, but
                 (unlike them) given a real default bundle since
                 operational facts are meant to resurface proactively, not
                 only on an explicit `triggers[]` override. The secondary
                 PreToolUse command-family surface (UPG-MEMORY-STATE-MACHINE
                 §5.2, the 'command' axis) is opt-in only — a caller that
                 wants a note to also fire beside a specific command family
                 declares an explicit
                 `triggers=[{"command": "..."}]` override, which fully
                 replaces this default bundle rather than adding to it (the
                 same replace-not-merge rule every kind's explicit
                 `triggers[]` already follows).
    """
    if kind == "directive":
        return [{"event": "session-start"}, {"event": "post-compaction"}]
    if kind == "task":
        return [{"event": "session-start"}] if priority == "high" else []
    if kind == "gotcha":
        return [
            {"path": anchor[0], "event": "pre-edit"}
            for anchor in (anchors or [])
            if anchor and anchor[0]
        ]
    if kind == "operational":
        return [{"event": "prompt-submit", "semantic": True}]
    return []


def effective_triggers(note: WorkingNote) -> list[dict]:
    """The trigger bundle `evaluate_note()` actually walks for `note`:
    its own explicit `triggers[]` when non-empty, else its kind's default
    bundle (`default_bundle_for_kind()`) resolved fresh from the note's
    current kind/anchors/priority. Single source of truth for "replace-
    not-merge" resolution — `evaluate_note()` uses it directly, and a
    caller needing to know WHICH axes a note would be evaluated against
    without re-running the whole evaluation (e.g.
    `WorkingContextStore.fire()`'s semantic-candidate prefilter, which
    must not miss a note relying on an implicit default bundle) calls this
    instead of re-deriving the same replace-not-merge rule a second time."""
    return note.triggers if note.triggers else default_bundle_for_kind(note.kind, note.anchors, note.priority)


def path_trigger_match(
    triggers: list[dict] | None, path_candidates: tuple[str, ...] | None
) -> str | None:
    """The declared-glob half of the P (path) trigger primitive, read as a
    STRUCTURAL RELEVANCE signal rather than a live-firing conjunction.

    `_trigger_matches()` below answers "does trigger T fire right now?" — a
    conjunction over every axis T declares (path AND event AND symbol...).
    This answers a narrower question the structural recall channel
    (`WorkingContextStore.recall_for_path()`, `agent.proactive.matcher.
    _first_anchor()`) needs: "did the note's author deliberately say this
    note concerns this file?" — checking ONLY the 'path' glob of each
    trigger in `triggers` against `path_candidates` (every equivalent form
    of the file being recalled, the same shape `_trigger_matches()` takes),
    ignoring every other axis a trigger might also declare (event/symbol/
    semantic/command). `triggers` must be the note's own EXPLICIT
    `triggers[]` — never a kind-default bundle resolved via
    `default_bundle_for_kind()`/`effective_triggers()`: a default bundle's
    path glob is already derived straight from `anchors` (see
    `default_bundle_for_kind()`'s gotcha branch), so that case is already
    covered by the existing anchor signal, and re-deriving it here would
    just double-count the same declaration under a different tier.

    Returns the first matching glob pattern (for an explanatory "triggers
    on <pattern>" render), or None when no declared trigger's path glob
    matches any candidate, `triggers` is empty/None, or `path_candidates`
    is empty/None.

    UPG-TRIGGERS-INERT-ON-PROXY-STRUCTURAL: this is a declared-glob-vs-
    file-path comparison — both sides are structural properties (the
    note's own declared trigger, the file being recalled), never a read of
    prompt/query/conversation content, so it stays inside the no-query-
    heuristics rule's explicit carve-out for structural path matching (the
    same carve-out `_trigger_matches()`'s own P primitive already relies
    on)."""
    if not triggers or not path_candidates:
        return None
    for trigger in triggers:
        path_pattern = trigger.get("path")
        if path_pattern is None:
            continue
        if any(fnmatch.fnmatch(candidate, path_pattern) for candidate in path_candidates):
            return path_pattern
    return None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class FireResult:
    note_id: int
    fired: bool
    explanation: str
    trigger_index: int | None = None
    # T:expires_visibility passed — the note still fired (T never blocks P/E),
    # but the caller should rank/pack it as faded.
    faded: bool = False
    # Filled in by the caller (WorkingContextStore.fire()) from the existing
    # staleness machinery — never computed here, this module never touches
    # the filesystem (evaluate_note is pure).
    stale_paths: list[str] = field(default_factory=list)
    # Whether the WINNING trigger declared the M (semantic) axis (UPG-
    # MEMORY-STATE-MACHINE §5.5) — lets the caller apply the semantic-only
    # session cooldown
    # (TriggerFireLedger.eligible(..., semantic=True)) without re-deriving
    # which trigger index matched or re-walking default_bundle_for_kind()
    # itself; this module already knows it at match time.
    semantic: bool = False


def _trigger_matches(
    trigger: dict,
    event: str | None,
    path_candidates: tuple[str, ...] | None,
    *,
    resolved_symbols: frozenset[str] | None = None,
    semantic_matched: bool | None = None,
    command_verb: str | None = None,
) -> tuple[bool, str]:
    """Conjunction check for ONE trigger's declared P/E/S/M primitives
    against the current lifecycle state. Returns (matched, human-readable
    description) — the description feeds the one-line fire explanation.

    `path_candidates` is every equivalent form of the current lifecycle's
    target file the CALLER already resolved — e.g. the path exactly as given
    plus its workspace-relative form (`WorkingContextStore.fire()` computes
    both; this module never touches a filesystem or a workspace root, so it
    never resolves paths itself). The P primitive matches the trigger's glob
    `path` pattern against ANY candidate — a real hook sends an ABSOLUTE
    file_path while triggers are naturally authored workspace-relative (a
    gotcha's default bundle generates them straight from anchors), so
    matching only one form would silently never fire a relatively-anchored
    trigger against a real hook event. `None`/empty means no file path this
    call — a trigger declaring 'path' then deterministically does not match.

    `resolved_symbols` is the set of symbol names defined in or referenced by
    the current lifecycle's target file — resolved ONCE per `fire()` call by
    the caller (a single symbol-graph lookup),
    never here: this function only ever does an O(1) set-membership check, no
    graph access. `resolved_symbols=None` means the symbol graph is
    unavailable this call (a memory-only daemon, a warm-up window before the
    graph finishes building, or no `file_path` was given) — a trigger
    declaring 'symbol' then deterministically does not match; never an
    error, never a fuzzy/near-miss guess (exact-resolution only, per the
    no-query-heuristics rule's carve-out for symbol-graph lookups).

    `semantic_matched` is the precomputed cosine(activity, note_vector) >=
    theta[kind] boolean for THIS note — computed once by the caller from ONE
    per-call activity embedding and the note's own already-stored vector;
    this function never sees a vector or prompt text, only the boolean
    outcome. `None` means "not evaluated this call"
    (no embedder attached, embedder still warming up, or the note declares
    no semantic axis) — a trigger declaring 'semantic' then deterministically
    does not match; never an error.

    `command_verb` (UPG-MEMORY-STATE-MACHINE §5.2) is the NORMALIZED
    verb of the Bash command about to run, resolved ONCE per `fire()` call
    by the caller via `app.cmdnorm.normalize_command()` (this module never
    parses a raw command string itself — purity invariant, same category as
    `path_candidates`). The 'command' primitive fnmatch-globs the trigger's
    declared pattern against it, exactly like 'path' globs against a
    resolved file path. `None` means no Bash command this call (every other
    lifecycle moment) — a trigger declaring 'command' then deterministically
    does not match; never an error, never a fuzzy guess."""
    path_pattern = trigger.get("path")
    want_event = trigger.get("event")
    want_symbol = trigger.get("symbol")
    want_semantic = trigger.get("semantic")
    want_command = trigger.get("command")
    if (
        path_pattern is None and want_event is None and want_symbol is None
        and want_semantic is None and want_command is None
    ):
        return False, ""  # malformed (should have been rejected by validate_trigger)

    if path_pattern is not None:
        if not path_candidates or not any(
            fnmatch.fnmatch(candidate, path_pattern) for candidate in path_candidates
        ):
            return False, ""
    if want_event is not None and want_event != event:
        return False, ""
    if want_symbol is not None:
        if resolved_symbols is None or want_symbol not in resolved_symbols:
            return False, ""
    if want_semantic:
        if semantic_matched is not True:
            return False, ""
    if want_command is not None:
        if command_verb is None or not fnmatch.fnmatch(command_verb, want_command):
            return False, ""

    parts = []
    if path_pattern is not None:
        parts.append(f"path {path_pattern}")
    if want_symbol is not None:
        parts.append(f"symbol {want_symbol}")
    if want_semantic:
        parts.append("semantic")
    if want_command is not None:
        parts.append(f"command {want_command}")
    desc = " + ".join(parts) if parts else ""
    if want_event is not None:
        desc = f"{desc} at {want_event}" if desc else f"event {want_event}"
    return True, desc


def scope_permits(
    note: WorkingNote,
    *,
    session_id: str | None = None,
    branch: str | None = None,
    file_path: str | tuple[str, ...] | None = None,
) -> tuple[bool, str]:
    """Whether `note`'s declared `scope` permits it
    to be considered at all for the given lifecycle context. Operates ONLY on
    the note's own stored scope/session_id/branch/anchors plus caller-resolved
    lifecycle state (a session id, a branch name, a file path) — never a query
    string (no-query-heuristics rule).

    "workspace" (default) and "repo" are true no-ops — see
    SCOPE_VALUES in _types.py for why "repo" isn't yet a real cross-store
    scope. "session" and "branch" need the caller to supply the matching
    lifecycle value; if the caller doesn't supply one (e.g. plain recall()
    with no branch context), the note is excluded rather than guessed open.

    "path-subtree" checks `file_path` against the directory of each declared
    anchor path. `file_path` accepts either a single path string or a tuple
    of equivalent candidate forms for the SAME file (mirroring
    `evaluate_note`'s `file_path`/`_trigger_matches`'s P primitive, e.g. the
    real hook path as given plus its workspace-relative form, both computed
    by the caller — this module never resolves or normalizes a path itself)
    — the subtree matches if ANY candidate falls under ANY declared anchor's
    directory. A note with no anchors, or given no candidate at all, is
    excluded (there is no declared subtree to match against, or nothing to
    match it against)."""
    scope = note.scope or DEFAULT_SCOPE
    if scope in ("workspace", "repo"):
        return True, ""
    if scope == "session":
        if session_id and note.session_id and session_id == note.session_id:
            return True, ""
        return False, "scope 'session' — not the writing session"
    if scope == "branch":
        if branch and note.branch and branch == note.branch:
            return True, ""
        return False, f"scope 'branch' — recorded on {note.branch or 'unknown'!r}, current is {branch or 'unknown'!r}"
    if scope == "path-subtree":
        if isinstance(file_path, str):
            candidates: tuple[str, ...] = (file_path,)
        elif file_path:
            candidates = tuple(file_path)
        else:
            candidates = ()
        if not candidates or not note.anchors:
            return False, "scope 'path-subtree' — no file path or no declared anchor subtree"
        for raw_candidate in candidates:
            candidate = raw_candidate.replace("\\", "/").lstrip("./")
            for anchor in note.anchors:
                if not anchor or not anchor[0]:
                    continue
                anchor_path = str(anchor[0]).replace("\\", "/").lstrip("./")
                anchor_dir = anchor_path.rsplit("/", 1)[0] if "/" in anchor_path else ""
                if candidate == anchor_path or (anchor_dir and (candidate == anchor_dir or candidate.startswith(anchor_dir + "/"))):
                    return True, ""
        return False, "scope 'path-subtree' — file not under the note's declared subtree"
    return True, ""  # unrecognised scope value never blocks


def evaluate_note(
    note: WorkingNote,
    *,
    event: str | None = None,
    file_path: str | tuple[str, ...] | None = None,
    now: float | None = None,
    session_id: str | None = None,
    branch: str | None = None,
    resolved_symbols: frozenset[str] | None = None,
    semantic_matched: bool | None = None,
    command_verb: str | None = None,
) -> FireResult:
    """Deterministic, total (never raises for well-formed input), linear-in-
    triggers evaluation of whether `note` fires for the given lifecycle state.

    A tombstoned note (`valid_until` set — explicitly superseded) never
    fires. A note whose declared `scope` does not permit this lifecycle
    context (`scope_permits()`) never fires either — checked before the
    trigger loop, since an out-of-scope note has nothing to evaluate.
    Otherwise: explicit `triggers[]` fully replace the kind's default
    bundle; an empty/absent `triggers[]` falls back to
    `default_bundle_for_kind()`. Each trigger in the (possibly default)
    bundle is tried in order; the FIRST one whose P/E/S conjunction matches
    AND whose T modifiers (not_before, cooldown) do not withhold it wins — OR
    composition across the list. `not_before` and `cooldown` withhold a fire
    outright; `expires_visibility` never withholds — it only marks the result
    `faded` for ranking (T is a modifier, and a modifier only gates through
    not_before/cooldown; visibility fade is a ranking signal, not a gate).

    `file_path` accepts either a single path string (the common/legacy case —
    a caller with only one path form) or a tuple of equivalent candidate
    forms for the SAME file (e.g. `WorkingContextStore.fire()` passes
    `(as_given, workspace_relative)` — see its docstring). The FULL candidate
    tuple is forwarded to both path-matching consumers: the P primitive
    (`_trigger_matches` docstring) and `scope_permits()`'s `path-subtree`
    check (its own docstring) each match against ANY candidate — an absolute
    hook path and a workspace-relative anchor/pattern are both given a
    chance, never just the first form. The one-line fire explanation never
    touches the resolved path text at all (it renders the trigger's own
    declared glob pattern, e.g. "path src/api/**"), so no convention is
    needed there. This module still never resolves or normalizes a path
    itself, that is entirely the caller's job (purity invariant).

    `resolved_symbols` is passed straight through to `_trigger_matches()`
    for the S primitive — see its docstring; this function never touches
    the symbol graph itself, only a caller-resolved set of names.
    `semantic_matched` is likewise passed straight through for the M
    primitive — a single precomputed boolean for this note, never a vector
    or prompt text. `command_verb` (UPG-MEMORY-STATE-MACHINE §5.2) is
    likewise passed straight through for the 'command' primitive — the
    caller-normalized Bash verb, never a raw command string."""
    if now is None:
        now = time.time()

    if isinstance(file_path, str) or file_path is None:
        path_candidates: tuple[str, ...] | None = (file_path,) if file_path is not None else None
    else:
        path_candidates = tuple(file_path) if file_path else None

    if note.valid_until is not None:
        return FireResult(note.note_id, False, "superseded — a tombstoned memory never fires")

    permitted, scope_reason = scope_permits(note, session_id=session_id, branch=branch, file_path=path_candidates)
    if not permitted:
        return FireResult(note.note_id, False, scope_reason)

    triggers = effective_triggers(note)
    if not triggers:
        return FireResult(note.note_id, False, "no triggers declared for this note/kind")

    for idx, trig in enumerate(triggers):
        matched, desc = _trigger_matches(
            trig, event, path_candidates,
            resolved_symbols=resolved_symbols, semantic_matched=semantic_matched,
            command_verb=command_verb,
        )
        if not matched:
            continue
        not_before = trig.get("not_before")
        if not_before is not None and now < not_before:
            continue
        cooldown = trig.get("cooldown")
        if cooldown is not None and note.last_fired is not None and (now - note.last_fired) < cooldown:
            continue
        expires_visibility = trig.get("expires_visibility")
        faded = expires_visibility is not None and now >= expires_visibility
        return FireResult(
            note_id=note.note_id,
            fired=True,
            explanation=f"fired: trigger {idx + 1} — {desc}",
            trigger_index=idx,
            faded=faded,
            semantic=bool(trig.get("semantic")),
        )
    return FireResult(note.note_id, False, "no declared trigger matched this event/path")


# ---------------------------------------------------------------------------
# One total order — reused for fire precedence, injection ordering, AND
# budget eviction (one shared implementation by design).
# ---------------------------------------------------------------------------

def total_order_key(note: WorkingNote) -> tuple[int, int, float, int]:
    """kind priority -> note priority -> last_used recency -> note_id.

    The final tie-break is note_id — immutable and monotonic — deliberately,
    following the same lesson as UPG-RECALL-ORDER-CHURN (recall()'s own
    tie-break excludes last_accessed because recall() bumps it on every read,
    which would reorder ties on the very next identical call). `last_fired`
    (set only by this engine's own fires, never by a plain vectr_recall) is
    preferred over `last_accessed` for "last_used" so a direct recall doesn't
    quietly reorder trigger-fire precedence.

    kind='task' exception (UPG-TASK-NOTE-INJECTION-RECENCY, matching
    recall()/recall_for_path()'s own SQL ordering for this kind): every note
    that fires together in one `fire()` call is stamped with the SAME
    `last_fired` reading (one shared clock per evaluation, by design — see
    `WorkingContextStore.fire()`), which silently ties `last_used` between
    two task notes the moment BOTH have ever fired once — collapsing to the
    ascending note_id tie-break below, which is backwards for "current
    checkpoint" state (a task note is current-work state, not a
    relevance-ranked learning; an older task note must never outrank a newer
    one). So kind='task' orders on note_id DESCENDING directly — an
    immutable, monotonic recency proxy immune to the shared-clock tie —
    instead of last_used. Every other kind is unaffected."""
    try:
        kind_rank = MEMORY_TRIGGER_KIND_PRIORITY.index(note.kind)
    except ValueError:
        kind_rank = len(MEMORY_TRIGGER_KIND_PRIORITY)
    try:
        priority_rank = MEMORY_TRIGGER_PRIORITY_RANK.index(note.priority)
    except ValueError:
        priority_rank = len(MEMORY_TRIGGER_PRIORITY_RANK)
    if note.kind == "task":
        return (kind_rank, priority_rank, 0.0, -note.note_id)
    last_used = note.last_fired if note.last_fired is not None else note.last_accessed
    return (kind_rank, priority_rank, -last_used, note.note_id)


# ---------------------------------------------------------------------------
# Per-session fire ledger (per-session dedup window)
# ---------------------------------------------------------------------------

class TriggerFireLedger:
    """Per-session dedup for the trigger engine.

    Once a note fires via trigger index K in this session, further
    evaluations that ALSO match trigger index K are suppressed for the rest
    of the session — "axis" is each entry in a note's own (possibly
    kind-default) `triggers[]` list, since each entry is literally one axis
    of that note's OR-composition. A DIFFERENT trigger index on the SAME
    note firing (e.g. a second gotcha anchor on a different file, or a task
    note's session-start trigger after its own compaction-reset) is a
    genuinely new reason to resurface it and is never suppressed here.

    `reset()` clears all suppression state for this session — call it on a
    compaction event (pre-compact/post-compaction resets eligibility).
    A brand-new session simply gets a brand-new ledger instance, so session
    end needs no explicit handling here.

    Also tracks the per-session CUMULATIVE injection spend: the
    per-session token cap bounds the total tokens injected across every
    `fire()`/`fire_and_format()` call in one session, not each call in
    isolation. `record_spend()` is called once per delivery with however many
    tokens that delivery actually packed; `remaining_budget()` is what the
    next delivery has left to spend. `reset()` also zeroes the spend —
    compaction makes the whole budget available again, consistent with
    previously-fired memories becoming re-eligible.

    Serving-policy hardening (UPG-MEMORY-STATE-MACHINE §5.5): the M
    (semantic) axis gets a DIFFERENT re-eligibility rule than every other
    axis — "probabilistic
    (semantic) hits get a per-note session cooldown (~10 turns)" rather than
    "fires once per session, ever". `eligible()`/`record_fire()` accept an
    optional `semantic=True` + `cooldown_turns=N`: when both are given, the
    (note_id, trigger_index) pair becomes re-eligible again once at least
    `cooldown_turns` UserPromptSubmit turns (`advance_turn()`) have passed
    since it last fired, instead of never. Deterministic (path/command-
    anchored) axes are unaffected — they keep the plain forever-this-session
    suppression above, which already serves as their debounce (every
    lifecycle moment a session-scoped ledger instance covers is bounded by
    the next compaction reset, unchanged)."""

    def __init__(self) -> None:
        self._fired: dict[int, set[int]] = {}
        self._spent_tokens: int = 0
        self._semantic_last_fired_turn: dict[tuple[int, int], int] = {}
        self._turn: int = 0

    def advance_turn(self) -> None:
        """Call once per UserPromptSubmit — advances the turn counter the
        semantic-axis cooldown above is measured in. Deliberately NOT called
        by `reset()`: the turn count is wall-session progress, not
        fire/spend eligibility state, so a compaction event resetting
        eligibility does not also rewind how many turns have elapsed."""
        self._turn += 1

    def eligible(
        self,
        note_id: int,
        trigger_index: int,
        *,
        semantic: bool = False,
        cooldown_turns: int | None = None,
    ) -> bool:
        if semantic and cooldown_turns is not None:
            last_turn = self._semantic_last_fired_turn.get((note_id, trigger_index))
            return last_turn is None or (self._turn - last_turn) >= cooldown_turns
        return trigger_index not in self._fired.get(note_id, set())

    def record_fire(self, note_id: int, trigger_index: int, *, semantic: bool = False) -> None:
        self._fired.setdefault(note_id, set()).add(trigger_index)
        if semantic:
            self._semantic_last_fired_turn[(note_id, trigger_index)] = self._turn

    def remaining_budget(self) -> int:
        return max(0, MEMORY_TRIGGER_PER_SESSION_TOKEN_CAP - self._spent_tokens)

    def record_spend(self, tokens: int) -> None:
        self._spent_tokens += max(0, tokens)

    def reset(self) -> None:
        self._fired.clear()
        self._spent_tokens = 0
        # Restores full re-eligibility for the semantic axis too (§3 "cleared
        # on compaction" is a per-session-ledger-wide contract — every other
        # axis's suppression is cleared by `_fired.clear()` above; the
        # semantic axis's suppression lives in this separate dict instead,
        # so it needs its own clear to avoid a note that fired via the
        # semantic axis just before compaction staying wrongly suppressed
        # afterward). `_turn` itself is deliberately NOT reset here — see
        # `advance_turn()`'s docstring: it is wall-session progress, not
        # eligibility state, and clearing the dict above already makes
        # `last_turn is None` for every note regardless of `_turn`'s value.
        self._semantic_last_fired_turn.clear()


# ---------------------------------------------------------------------------
# Per-turn cross-surface dedup + budget ledger (UPG-MEMORY-STATE-MACHINE §5.3
# /§5.4) — closes the arm-C double-dip finding.
# ---------------------------------------------------------------------------

class TurnInjectionLedger:
    """Per-turn, cross-SURFACE dedup + budget, daemon-side, reset at every
    UserPromptSubmit — distinct from `TriggerFireLedger` above (session-
    scoped, per-(note_id, trigger_index) axis, persists across turns and
    only resets at compaction).

    vectr has THREE independent injection surfaces that each call
    `fire()`/`fire_and_format()` on their own schedule: session-start bulk,
    PreToolUse (file-anchored or command-family), and prompt-time semantic.
    Without a shared ledger, the SAME note can match more than one surface
    in the SAME turn and get injected twice — the documented arm-C
    double-dip. This ledger is note_id-granular (not axis-granular — which
    SURFACE claimed it doesn't matter, only that exactly one did) and is
    shared by every surface's `fire_and_format()` call this turn via the
    SAME instance (`VectrService`'s per-session registry): the first
    surface to match a note claims it; a later surface's `fire_and_format()`
    call filters that note_id out before it ever reaches the injection
    budget pack, so it is never even considered for injection twice — never
    silently dropped from the FIRST surface's delivery, only suppressed
    everywhere else this turn.

    `remaining_turn_budget()` bounds the ORDINARY-turn allowance (§5.4,
    default 500 tokens) shared by every per-turn surface combined.
    Session-start's own once-per-session bulk delivery is deliberately NOT
    metered here — see `WorkingContextStore.fire_and_format()`'s
    `spend_turn_budget` flag — it keeps its existing separate
    per-session cap (`TriggerFireLedger.remaining_budget()`); the dedup
    claim above still applies to it, only the budget accounting does not."""

    def __init__(self) -> None:
        self._claimed: set[int] = set()
        self._spent_tokens: int = 0

    def eligible(self, note_id: int) -> bool:
        return note_id not in self._claimed

    def claim(self, note_id: int) -> None:
        self._claimed.add(note_id)

    def remaining_turn_budget(self) -> int:
        return max(0, MEMORY_TRIGGER_PER_TURN_TOKEN_CAP - self._spent_tokens)

    def record_turn_spend(self, tokens: int) -> None:
        self._spent_tokens += max(0, tokens)

    def reset(self) -> None:
        self._claimed.clear()
        self._spent_tokens = 0


# ---------------------------------------------------------------------------
# Provenance framing
# ---------------------------------------------------------------------------

# Fixed protocol strings, not tunables (same category as the pre-existing
# hardcoded "[STALE]"/"WARNING: ..." strings in format_notes_for_llm) — every
# injected block's framing is deterministic on provenance+kind alone.
_HUMAN_DIRECTIVE_FRAME = "DIRECTIVE (standing rule from the user — follow it): "
_HUMAN_FRAME = "Recorded by the user: "
# UPG-MEM-PROVENANCE-USER-STATED: one frame for every kind, deliberately. The
# unhedged imperative stays reserved for human provenance — this class says
# what is machine-checkable and nothing more (a verbatim excerpt of the user's
# words is bound to this note's content) and lets the reader draw the
# conclusion, rather than asserting endorsement nobody verified.
_USER_STATED_FRAME = "User-stated (verbatim excerpt bound, transcribed by an AI session): "
# UPG-PROVENANCE-AUTOBIND-SPAN: the counterpart frame for a note whose bound
# `user_quote` is a SUB-SPAN of a larger, agent-authored body (e.g. `USER
# DIRECTIVE 2026-07-12: "<the user's actual words>"`, where the label and
# quotation marks are the writing agent's, only the inner clause the user's)
# rather than the note's whole content. `_USER_STATED_FRAME` above applies
# that same unqualified "user-stated" reading to the ENTIRE line that
# follows it; doing that here would launder the agent-authored framing
# around the quote into user authority — exactly the trust-laundering this
# frame exists to prevent. Names the bound span explicitly and states in the
# frame itself, not just in the note body, that the remainder is agent
# context.
_USER_STATED_SPAN_FRAME_TEMPLATE = (
    "User-stated in part — only this exact excerpt is confirmed the user's "
    'own words, bound verbatim by an AI session: "{quote}". The rest of '
    "this note is agent-added context around that excerpt, not itself "
    "user-endorsed: "
)
_AGENT_FRAME = "Memory to verify (recorded by an AI session, not human-endorsed): "
_AUTO_FRAME = "Auto-captured (weakest confidence, no reviewing judgment applied — verify before relying on this): "


def _is_whole_note_quote(user_quote: str, content: str) -> bool:
    """True when the bound `user_quote` covers the note's ENTIRE content
    (whitespace-normalized) rather than a genuine sub-span of it — the
    boundary between `_USER_STATED_FRAME` (unqualified — the whole line
    really is the user's transcribed words) and
    `_USER_STATED_SPAN_FRAME_TEMPLATE` (qualified — only the named excerpt
    is). Deferred import: agent/working_context_store/_user_quote.py is
    imported here rather than at module load time for the same reason
    `_format_full_block()` already imports `frame_prefix` lazily —
    agent/working_context_store/_store.py imports this module at load time,
    so a top-level import back would cycle."""
    from agent.working_context_store._user_quote import normalize_for_binding

    return normalize_for_binding(user_quote) == normalize_for_binding(content)


def frame_prefix(provenance: str, kind: str, *, user_quote: str = "", content: str = "") -> str:
    """The imperative/hedged framing prefix for one injected memory block.
    Only human-provenance directives ever render as an unhedged imperative;
    user-stated provenance names its bound verbatim excerpt; agent-provenance
    is framed as memory to verify; auto-provenance carries the weakest framing.
    Immutable per note — this is a pure function
    of the note's own stored (provenance, kind), plus the two new optional
    keyword arguments below.

    `user_quote`/`content` (UPG-PROVENANCE-AUTOBIND-SPAN, both optional,
    default "" — every existing 2-argument call keeps today's framing
    unchanged): disambiguate the two shapes a `user-stated` note can take.
    When the bound quote covers the note's entire (whitespace-normalized)
    content, the whole line genuinely is the user's transcribed words and
    keeps the unqualified `_USER_STATED_FRAME`. When the quote is a strict
    sub-span of a larger, agent-authored body (the harness's span auto-bind
    — see `bind_user_quote_auto()`), `_USER_STATED_SPAN_FRAME_TEMPLATE`
    names the bound span explicitly and states the remainder is agent
    context, not user-endorsed."""
    if provenance == "human":
        return _HUMAN_DIRECTIVE_FRAME if kind == "directive" else _HUMAN_FRAME
    if provenance == USER_STATED_PROVENANCE:
        if user_quote and content and not _is_whole_note_quote(user_quote, content):
            return _USER_STATED_SPAN_FRAME_TEMPLATE.format(quote=user_quote)
        return _USER_STATED_FRAME
    if provenance == "auto":
        return _AUTO_FRAME
    return _AGENT_FRAME  # "agent" (default) and any unrecognised value


# ---------------------------------------------------------------------------
# Two-tier injection budget + pack
# ---------------------------------------------------------------------------

def token_estimate(text: str) -> int:
    return max(1, len(text) // MEMORY_TRIGGER_CHARS_PER_TOKEN)


def _trim_full_text_to_cap(text: str, max_tokens: int) -> tuple[str, int]:
    """Trim TEXT to fit within MAX_TOKENS (converted to chars via
    `MEMORY_TRIGGER_CHARS_PER_TOKEN`), backing off to a SENTENCE boundary
    (". ") within the trim budget when one exists there, else a WORD
    boundary (" "), else a hard character cut — never a bare mid-word/mid-
    sentence slice for its own sake, but never refusing to trim either.

    UPG-HOOK-GOTCHA-CAP-TITLE-ONLY: mirrors `agent/proactive/matcher.py`'s
    `_cap()` (the proxy channel's own UPG-PROXY-INJECT-TITLE-ONLY fix),
    applied to `pack_injection()`'s per-kind/per-injection cap instead of the
    proxy's single-line candidate cap — same ladder, same
    min-boundary-fraction guard, different caller. Unlike `_cap()`, this does
    NOT collapse TEXT to one line first: a full-tier block
    (`_format_full_block()`) is deliberately multi-line (header, content,
    staleness warnings), and only the trailing slice is affected by
    truncation, so internal newlines before the cut point are preserved.

    Returns `(trimmed_text, token_estimate(trimmed_text))`. The returned
    text is ALWAYS at most `max_tokens` tokens by construction (the trimmed
    char length never exceeds `max_tokens * MEMORY_TRIGGER_CHARS_PER_TOKEN`),
    except when `max_tokens` is too small to hold even the trailing ellipsis
    character — the caller falls back to the index-tier line in that
    pathological case (see `pack_injection`)."""
    if max_tokens <= 0:
        return "…", token_estimate("…")
    max_chars = max_tokens * MEMORY_TRIGGER_CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text, token_estimate(text)
    if max_chars <= 1:
        return "…", token_estimate("…")

    budget = max_chars - 1
    window = text[:budget]
    min_boundary = max(1, int(budget * MEMORY_TRIGGER_BODY_TRUNCATION_MIN_BOUNDARY_FRACTION))

    sentence_boundary = window.rfind(". ")
    if sentence_boundary >= min_boundary:
        trimmed = window[: sentence_boundary + 1] + "…"
        return trimmed, token_estimate(trimmed)

    word_boundary = window.rfind(" ")
    if word_boundary >= min_boundary:
        trimmed = window[:word_boundary].rstrip() + "…"
        return trimmed, token_estimate(trimmed)

    trimmed = window.rstrip() + "…"
    return trimmed, token_estimate(trimmed)


@dataclass
class PackedItem:
    note_id: int
    text: str
    tier: str  # "full" | "index"


def pack_injection(
    items: list[tuple[WorkingNote, str, str]],
    *,
    budget: int | None = None,
) -> list[PackedItem]:
    """Two-tier budget pack: directive/gotcha prefer full text, every other
    kind injects its index-tier one-liner. `items` is
    [(note, full_text, index_text), ...] in any order; this function sorts
    by the shared `total_order_key` and packs greedily, spending the given
    `budget` (defaults to the full per-session cap when omitted/None — a
    single call in isolation, e.g. a direct unit-test call or a fresh
    session's first delivery).

    A full-tier memory whose OWN per-kind/per-injection cap it exceeds is
    TRIMMED into that cap (`_trim_full_text_to_cap()`, sentence/word-
    boundary backoff — UPG-HOOK-GOTCHA-CAP-TITLE-ONLY, parity with the
    proxy channel's UPG-PROXY-INJECT-TITLE-ONLY fix) rather than dropped
    wholesale to its index-tier title line: the whole point of a full-tier
    kind (directive/gotcha/operational) is that its BODY guidance reaches
    the wire, and a bare character cut there would have read as corrupted
    output rather than an intentionally shortened block, so it still backs
    off to a clean boundary first. It only falls back to the index-tier
    line when the cap is too small to hold even the trimmed text's trailing
    ellipsis (a pathological config, not a real per-kind cap). Separately,
    a memory is NEVER partially truncated against the shared multi-note
    BUDGET below — it packs whole (at whichever tier/trim it already
    reached above), or is evicted entirely if even its index-tier line does
    not fit in what's left of `budget`. Eviction is always from the BOTTOM
    of the shared total order — the lowest-precedence notes are the ones
    dropped first. The moment any item is evicted for not fitting even at
    the index tier, packing STOPS entirely: nothing lower-precedence is
    ever allowed to ship while something higher-precedence was dropped,
    even if it would have fit in the leftover budget.

    Passing the session ledger's `remaining_budget()` here is what makes the
    per-session cap CUMULATIVE across every `fire()`/`fire_and_format()`
    call in one session rather than a fresh allowance each call."""
    ordered = sorted(items, key=lambda triple: total_order_key(triple[0]))
    budget = MEMORY_TRIGGER_PER_SESSION_TOKEN_CAP if budget is None else max(0, budget)
    packed: list[PackedItem] = []
    for note, full_text, index_text in ordered:
        prefer_full = note.kind in FULL_TEXT_KINDS
        text, tier = (full_text, "full") if prefer_full else (index_text, "index")
        tokens = token_estimate(text)

        # Per-kind cap (directive<=400, gotcha<=100, operational<=250)
        # overrides the flat per-injection cap for kinds that declare one;
        # a kind absent from the override falls back to the single global
        # cap.
        per_injection_cap = MEMORY_TRIGGER_PER_KIND_TOKEN_CAP.get(
            note.kind, MEMORY_TRIGGER_PER_INJECTION_TOKEN_CAP
        )
        if tier == "full" and tokens > per_injection_cap:
            trimmed_text, trimmed_tokens = _trim_full_text_to_cap(text, per_injection_cap)
            if trimmed_tokens <= per_injection_cap:
                text, tokens = trimmed_text, trimmed_tokens
            else:
                text, tier, tokens = index_text, "index", token_estimate(index_text)

        if tokens > budget:
            if tier == "full":
                text, tier, tokens = index_text, "index", token_estimate(index_text)
            if tokens > budget:
                break  # evicted — stop packing so nothing lower-precedence backfills

        packed.append(PackedItem(note_id=note.note_id, text=text, tier=tier))
        budget -= tokens
    return packed
