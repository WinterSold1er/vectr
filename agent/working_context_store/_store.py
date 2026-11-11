"""
WorkingContextStore — SQLite-backed store for LLM working notes and session snapshots.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

from agent.chroma_dispatch import timed_chroma_call
from agent.working_context_store._audit import audit
from agent.working_context_store._encryption import _build_encryptor, _extract_file_paths, _NoteEncryptor
from agent.working_context_store._events import NOTE_EVENT_ACTORS, NOTE_EVENT_KINDS, fold as _fold_note_events
from agent.working_context_store._related import (
    RelatedNote,
    RevokedRelatedNote,
    related_active_notes as _related_active_notes,
    revoked_related_notes as _revoked_related_notes,
)
from agent.working_context_store._types import (
    DEFAULT_KIND,
    DEFAULT_PROVENANCE,
    DEFAULT_SCOPE,
    KIND_DEFAULT_SCOPES,
    PROMOTION_LADDER,
    PROMOTION_RANK,
    PROVENANCE_VALUES,
    SCOPE_VALUES,
    USER_STATED_PROVENANCE,
    VALID_KINDS,
    SnapshotEntry,
    WorkingNote,
)
from agent.working_context_store._user_quote import bind_user_quote, bind_user_quote_auto

logger = logging.getLogger(__name__)


def _hash_path_content(root: Path, raw_path: str) -> str | None:
    """sha256[:16] of a workspace-relative (or absolute) path's current file
    content, or None if it cannot be read (missing, directory, permission
    error). Shared by `remember()`'s anchor-hash-at-write and
    `check_staleness()`'s anchor re-hash-at-fire-time (TRIGGER-ENGINE wave 1,
    bm2-design-skeleton.md §5) so both apply the exact same rule — this is
    the same sha256[:16] shape as the pre-existing `code_hash` staleness
    check just below, generalised to any anchor path rather than one
    extension allowlist."""
    path = Path(raw_path)
    resolved = path if path.is_absolute() else root / path
    try:
        return hashlib.sha256(resolved.read_bytes()).hexdigest()[:16]
    except OSError:
        return None


def _append_event(
    conn: sqlite3.Connection,
    workspace: str,
    note_id: int,
    event: str,
    actor: str,
    reason: str | None = None,
    payload: str | None = None,
    ts: float | None = None,
) -> None:
    """Append one row to `note_events` (UPG-MEMORY-STATE-MACHINE §4.1) on an
    ALREADY-OPEN connection/transaction, so an event is always written in
    the same transaction as the note mutation it documents (e.g. `remember
    ()`'s `created` insert, or the `superseded` UPDATE it pairs with) —
    never a separate commit that could observably split from its cause.
    `event`/`actor` are validated against the closed protocol vocabularies
    in _events.py; an invalid value is a programming error in THIS module,
    not caller input (every public entry point below validates its own
    caller-facing values before ever reaching here), so this asserts rather
    than raising a caller-facing ValueError."""
    assert event in NOTE_EVENT_KINDS, f"unknown note event kind: {event!r}"
    assert actor in NOTE_EVENT_ACTORS, f"unknown note event actor: {actor!r}"
    conn.execute(
        """INSERT INTO note_events (workspace, note_id, event, actor, reason, payload, ts)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (workspace, note_id, event, actor, reason, payload, ts if ts is not None else time.time()),
    )


def _actor_for_provenance(provenance: str) -> str:
    """The `created` event's actor, derived from the note's own provenance
    (bm2-design-skeleton.md §5's existing human/agent/auto trust classes) —
    not a separate caller-supplied value, since a note's provenance already
    says who stands behind it. provenance='auto' ("captured by a mechanism
    with no reviewing judgment") maps to actor='system', the same reserved
    class `check_staleness()`'s `stale_flagged` events use for the one
    other non-judgment transition; 'human'/'agent' map onto themselves.
    provenance='user-stated' maps to actor='agent': the excerpt bound to the
    note is the user's, but the act of writing it was an AI session's, and
    NOTE_EVENT_ACTORS records who performed the transition, never whose words
    it carries."""
    if provenance == "auto":
        return "system"
    if provenance == USER_STATED_PROVENANCE:
        return "agent"
    return provenance


def _actor_for_promotion(to: str) -> str:
    """The `promoted` event's actor: promoting TO provenance='human' is,
    definitionally, a human endorsing the note (mirrors the existing MCP
    vectr_promote trust boundary — only a human-operated surface may ever
    promote to 'human'); promoting to 'agent' (from 'auto') is an agent
    decision either way."""
    return "human" if to == "human" else "agent"


def _path_trigger_candidates(workspace_root: str, file_path: str | None) -> tuple[str, ...] | None:
    """The P (path) trigger primitive's candidate forms for one lifecycle
    file_path: the path exactly as given, plus — each when computable — its
    workspace-relative form AND its workspace-rooted ABSOLUTE form (the SAME
    resolve()/relative_to() normalization `recall_for_path()` already uses
    just above; the absolute form of a relative input is derived from
    workspace_root, not this process's cwd, so it means the same file
    regardless of where the daemon happens to have been launched).

    UPG-PROXY-ANCHOR-ABS-REL-NORM: declared anchors/triggers are authored
    either way (remember()'s own docstring allows both), and so are incoming
    file_paths — a real hook sends an absolute one, while direct callers of
    recall_for_path()/fire() legitimately pass workspace-relative ones.
    Matching only "as-given + relative" covered abs-input-vs-rel-anchor but
    silently missed the mirror case, rel-input-vs-ABS-anchor: a note whose
    anchors column holds "/ws/src/auth.py" was invisible to a caller asking
    for "src/auth.py", falling back to (at best) content-boundary matching.
    Deriving the third form closes that direction at this single boundary,
    which both `recall_for_path()` and `fire()`'s live P-primitive share.

    A file outside `workspace_root` simply has no relative form — the
    as-given form is still returned, just without extra candidates, never
    an error. Returns None only when `file_path` itself is None (no path
    this call)."""
    if file_path is None:
        return None
    candidates = [file_path]
    resolved_root = Path(workspace_root).resolve()
    try:
        relpath = str(Path(file_path).resolve().relative_to(resolved_root))
    except (ValueError, OSError):
        relpath = None
    if relpath and relpath not in candidates:
        candidates.append(relpath)
    # The workspace-rooted absolute form only carries information the
    # as-given form does not when the input was relative. For an absolute
    # input this join reproduces the input itself (already present), so the
    # candidate sequence for a real hook event stays exactly two forms.
    if not Path(file_path).is_absolute():
        abspath = str((resolved_root / file_path).resolve())
        if abspath not in candidates:
            candidates.append(abspath)
    return tuple(candidates)


def _is_path_identifier_char(ch: str) -> bool:
    """A char that extends a filename/identifier token: alphanumeric, "_",
    or "-". Used by `_path_boundary_match()` to tell a genuine path
    boundary ("/", whitespace, quotes, punctuation, string start/end) apart
    from a longer identifier that merely contains the needle as a
    substring (e.g. "uv_regate.py" contains "gate.py")."""
    return ch.isalnum() or ch in ("_", "-")


def _path_boundary_occurrences(text: str, needle: str) -> list[int]:
    """Every offset in `text` where `needle` (a file basename or relative
    path) occurs at a genuine path boundary, ascending — the single
    implementation behind `_path_boundary_match`, `_path_boundary_count`,
    and `_path_boundary_first` (UPG-PROXY-WEAK-TIER-TIEBREAK), so the
    boolean, the count, and the position of a mention can never disagree
    about what counts as one.

    A match only counts when the character immediately before the
    occurrence is not identifier-class (or the occurrence starts the
    string) AND the character immediately after is not identifier-class
    (or the occurrence ends the string) — see `_path_boundary_match`."""
    if not needle:
        return []
    found: list[int] = []
    start = 0
    while True:
        idx = text.find(needle, start)
        if idx == -1:
            return found
        before_ok = idx == 0 or not _is_path_identifier_char(text[idx - 1])
        after_idx = idx + len(needle)
        after_ok = after_idx >= len(text) or not _is_path_identifier_char(text[after_idx])
        if before_ok and after_ok:
            found.append(idx)
        start = idx + 1


def _path_boundary_match(text: str, needle: str) -> bool:
    """True if `needle` (a file basename or relative path) occurs in `text`
    at a genuine path boundary, not merely as a substring.

    UPG-PROXY-SUBSTRING-ANCHOR: unanchored substring matching lets
    "gate.py" false-match inside "uv_regate.py", and "gate.py" false-match
    a longer extension like "gate.pyc"/"gate.pyx". A match only counts
    when the character immediately before the occurrence is not
    identifier-class (or the occurrence starts the string) AND the
    character immediately after is not identifier-class (or the
    occurrence ends the string). Callers that need a cheap prefilter
    (e.g. a SQL LIKE clause) should treat this function's result as the
    precise filter applied afterward, not a replacement for the prefilter
    — see `recall_for_path()`.

    Delegates to `_path_boundary_occurrences()` (same scan, early exit on
    the first hit) rather than keeping its own copy of the boundary rule —
    three consumers now share that predicate, and they must never drift."""
    return bool(_path_boundary_occurrences(text, needle))


def _path_boundary_count(text: str, needle: str) -> int:
    """How many separate path-boundary occurrences `needle` has in `text`
    (UPG-PROXY-WEAK-TIER-TIEBREAK): a note whose subject is a file tends to
    name it repeatedly; a note that merely name-drops it in passing names
    it once. Same boundary predicate as `_path_boundary_match`, which is
    exactly `count > 0`."""
    return len(_path_boundary_occurrences(text, needle))


def _path_boundary_first(text: str, needle: str) -> int:
    """The character offset of the FIRST path-boundary occurrence of
    `needle` in `text`, or -1 when there is none (UPG-PROXY-WEAK-TIER-
    TIEBREAK): among notes that mention a file equally often, one whose
    subject is the file tends to name it early (title / opening sentence),
    where a passing mention lands mid-list. Same boundary predicate as
    `_path_boundary_match`."""
    hits = _path_boundary_occurrences(text, needle)
    return hits[0] if hits else -1


def _anchors_exact_match(
    anchors: "list[list[str]] | None", path_candidates: tuple[str, ...] | None
) -> bool:
    """True when a note's declared `anchors` (already-parsed
    `[[path, hash_or_None], ...]`, see `WorkingNote.anchors`) contains one
    of a file's candidate path forms EXACTLY — the strongest possible
    signal a note is about that file, independent of whatever its prose
    content happens to mention. `path_candidates` is
    `_path_trigger_candidates()`'s output — as-given + workspace-relative
    + (for a relative input) workspace-rooted absolute forms
    (UPG-PROXY-ANCHOR-ABS-REL-NORM) — for the file being recalled."""
    if not anchors or not path_candidates:
        return False
    declared = {a[0] for a in anchors if a}
    return any(c in declared for c in path_candidates)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Plain dot-product cosine similarity between two equal-length vectors.
    Used only by the M (semantic) trigger primitive (TRIGGER-ENGINE wave 2b,
    §8) to compare one prompt-submit activity embedding against a note's own
    already-stored vector — no external numeric dependency needed for the
    small, session-scoped candidate counts involved. Returns 0.0 for a
    degenerate (zero-length) vector rather than raising ZeroDivisionError.

    Always returns a plain Python float, even when `b` is a numpy array (as
    Chroma's `.get(..., include=["embeddings"])` returns) — otherwise the
    result silently becomes a numpy.float64/numpy.bool_ once compared, and
    `numpy.True_ is True` is False (distinct objects), which would make the M
    primitive's `semantic_matched is True` gate in trigger_engine.py never
    fire for a real, Chroma-backed note even on an exact vector match."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def _current_git_branch(
    root: Path,
    *,
    _run_git: Callable[..., "subprocess.CompletedProcess"] | None = None,
) -> str | None:
    """Current git branch name for `root`, or None when `root` isn't inside a
    git work tree, git is unavailable, or HEAD is detached (`rev-parse` then
    returns the literal string "HEAD", which is never treated as a branch
    name). Mirrors `agent.version_stamp._git_short_sha`'s exact subprocess
    pattern (injectable `_run_git` for testability, 2s timeout, never raises)
    — all failure is swallowed here, the caller degrades to no branch
    context rather than raising.

    Used by `remember()` (write-time capture for scope="branch") and by
    `fire()` (current-branch check at evaluation time, TRIGGER-ENGINE
    wave 2a, bm2-design-skeleton.md §1)."""
    run = _run_git or subprocess.run
    try:
        result = run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    branch = result.stdout.strip()
    if not branch or branch == "HEAD":
        return None
    return branch


def _scope_filter(
    notes: list[WorkingNote],
    *,
    session_id: str | None = None,
    file_path: str | tuple[str, ...] | None = None,
) -> list[WorkingNote]:
    """Recall-side scope enforcement (TRIGGER-ENGINE wave 2a,
    bm2-design-skeleton.md §1) — a pure post-filter over an already-fetched
    note list, reusing `scope_permits()`.

    Only "session" scope is enforced unconditionally here: an ephemeral note
    must never surface outside its writing session via ANY read path, not
    just trigger firing. "branch" is deliberately never enforced at the
    recall side (see the SCOPE_VALUES comment in _types.py) — only fire()
    enforces it, since branch scope is about bounding ambient trigger noise,
    not about hiding a note from a deliberate recall query, and this also
    avoids a git subprocess call on the hot, frequently-called plain
    recall() path. "path-subtree" is enforced ONLY when the caller has a
    `file_path` to filter against (recall_for_path()); plain query-based
    recall() has no file context, so a path-subtree-scoped note is left
    unfiltered there. `file_path` accepts either a single string or a tuple
    of candidate forms for the same file (as-given plus workspace-relative,
    plus the workspace-rooted absolute form for a relative input — the same
    shape `fire()`'s `_path_trigger_candidates()` produces) — a real
    hook/tool call sends an ABSOLUTE path while anchors are naturally
    authored workspace-relative, so `recall_for_path()` passes both forms
    to give `scope_permits()`'s path-subtree check a relative form to
    match against.

    Accepted trade-off: called AFTER a SQL LIMIT at every call site, so
    excluding a scoped note here may return fewer than `limit` results —
    the same class of trade-off the pre-existing min_similarity cutoff
    already accepts, not a new one."""
    from agent.trigger_engine import scope_permits

    out = []
    for note in notes:
        if note.scope == "session":
            permitted, _ = scope_permits(note, session_id=session_id)
            if not permitted:
                continue
        elif note.scope == "path-subtree" and file_path is not None:
            permitted, _ = scope_permits(note, file_path=file_path)
            if not permitted:
                continue
        out.append(note)
    return out


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _title_token_set(text: str, min_len: int) -> set[str]:
    """Lowercased alphanumeric tokens of `text`, dropping anything shorter
    than `min_len` — the same uniform tokenization applied to a note's own
    title and to the incoming query text (UPG-RECALL-MISS-FLOOR part (c)),
    never a fixed vocabulary of interesting/uninteresting words."""
    return {t.lower() for t in _TOKEN_RE.findall(text) if len(t) >= min_len}


def _declared_metadata_match_specificity(note: WorkingNote, query: str) -> tuple[int, int]:
    """(channels_matched, total_matched_chars) for how strongly `note`'s OWN
    declared tag/anchor/trigger-symbol/title metadata is found in `query`'s
    text — deterministic containment tested against this note's stored
    metadata, never a classification of what kind of query `query` is
    (UPG-RECALL-MISS-FLOOR part (c); see the no-query-heuristics rail in
    the module's callers). Every note in a workspace is checked the same
    uniform way regardless of the query's content — the branches below are
    channels over one note's declared fields, not a lookup table of query
    patterns. (0, 0) means no channel matched at all.

    This is used two ways by callers: membership (`channels_matched > 0`,
    via `_note_matches_declared_metadata()` below) and, once a note has
    already independently matched, ORDER among matches
    (`_recall_floor_notes()`'s deterministic channel — UPG-RECALL-MISS-
    FLOOR F2 fix). channels_matched counts how many of the four distinct
    declared-metadata FIELDS (tag/anchor/symbol/title) matched at all;
    total_matched_chars is the combined length of the specific text that
    matched within those fields — a longer, more specific matched string is
    less likely to be a coincidental overlap than a short generic one.
    Both are properties of the NOTE's own metadata and how much of it
    overlaps `query`, never of the query's content/intent/type on its own.
    """
    from agent.config import (
        RECALL_FLOOR_TITLE_MIN_OVERLAP_TOKENS,
        RECALL_FLOOR_TITLE_MIN_TOKEN_LEN,
    )

    q_lower = query.lower()
    channels_matched = 0
    total_chars = 0

    # Tag channel: one or more of THIS note's own declared tags appear
    # verbatim in the query text.
    tag_hits = []
    for t in note.tags:
        norm = (t or "").strip().lower()
        if norm and norm in q_lower:
            tag_hits.append(norm)
    if tag_hits:
        channels_matched += 1
        total_chars += sum(len(t) for t in tag_hits)

    # Anchor channel: the basename (no extension) of a file THIS note is
    # anchored to appears verbatim in the query text.
    anchor_hits = []
    for anchor in note.anchors:
        path = anchor[0] if anchor else ""
        stem = Path(path).stem.lower() if path else ""
        if stem and stem in q_lower:
            anchor_hits.append(stem)
    if anchor_hits:
        channels_matched += 1
        total_chars += sum(len(s) for s in anchor_hits)

    # Symbol channel: a code symbol name THIS note declared via an
    # explicit trigger's 'symbol' key appears verbatim in the query text.
    # Case-sensitive: symbol names are code identifiers, not prose.
    symbol_hits = []
    for trig in note.triggers or []:
        sym = trig.get("symbol") if isinstance(trig, dict) else None
        if sym and sym in query:
            symbol_hits.append(sym)
    if symbol_hits:
        channels_matched += 1
        total_chars += sum(len(s) for s in symbol_hits)

    # Title channel: THIS note's own title, tokenized the same uniform way
    # as the query, overlaps the query's tokens by at least the configured
    # floor.
    if note.title:
        title_tokens = _title_token_set(note.title, RECALL_FLOOR_TITLE_MIN_TOKEN_LEN)
        if title_tokens:
            query_tokens = _title_token_set(query, RECALL_FLOOR_TITLE_MIN_TOKEN_LEN)
            overlap = title_tokens & query_tokens
            if len(overlap) >= RECALL_FLOOR_TITLE_MIN_OVERLAP_TOKENS:
                channels_matched += 1
                total_chars += sum(len(t) for t in overlap)

    return channels_matched, total_chars


def _note_matches_declared_metadata(note: WorkingNote, query: str) -> bool:
    """True if `note`'s OWN declared tag, anchor, trigger-symbol, or title
    metadata is found in `query`'s text at all — see
    `_declared_metadata_match_specificity()` above for the full channel
    definitions; this is just its membership test (channels_matched > 0)."""
    return _declared_metadata_match_specificity(note, query)[0] > 0


def _merge_recall_floor(
    tier0: list[WorkingNote],
    deterministic: list[WorkingNote],
    ranked: list[WorkingNote],
    limit: int,
) -> list[WorkingNote]:
    """Compose Tier 0, the deterministic channels, and the ranked
    (semantic/SQL) result into one de-duplicated list, in that fixed
    precedence, truncated to `limit` (UPG-RECALL-MISS-FLOOR parts (b)/(c)).
    Never returns more than `limit` notes — the STEP 0 guard in
    benchmarks/recall_miss/harness.py depends on that being true by
    construction here, not by convention at each call site. When neither
    `tier0` nor `deterministic` contributed anything (the common case —
    apply_floor=False, a workspace with no pinned notes and the
    deterministic channel at its shipped default-off, or simply no
    matching notes), this returns `ranked` unchanged, truncated to
    `limit` exactly like the pre-existing behavior it replaces."""
    if not tier0 and not deterministic:
        return ranked[:limit]
    combined: list[WorkingNote] = []
    seen: set[int] = set()
    for group in (tier0, deterministic, ranked):
        for note in group:
            if len(combined) >= limit:
                return combined
            if note.note_id not in seen:
                seen.add(note.note_id)
                combined.append(note)
    return combined


def _age_str(created_at: float) -> str:
    """`created_at`'s age as a compact human string ("45s", "12m", "3h",
    "2d") — hoisted to module level (was a closure inside
    `format_notes_for_llm`) so `_format_index_line`/`_format_full_block`
    (and `fire_and_format()`, TRIGGER-ENGINE wave 2a) can share it."""
    age_s = max(time.time() - created_at, 0.0)
    if age_s < 60:
        return f"{age_s:.0f}s"
    if age_s < 3600:
        return f"{age_s / 60:.0f}m"
    age_h = age_s / 3600
    return f"{age_h:.0f}h" if age_h < 48 else f"{age_h / 24:.0f}d"


def _date_str(created_at: float) -> str:
    """`created_at`'s calendar date as `YYYY-MM-DD` (UPG-DECISION-TIMELINE) —
    used by `_format_index_line` in place of `_age_str`'s relative age when
    `sort_by='chronological'`, so a time-ordered listing reads as a dated
    timeline rather than a set of ages that all drift as the caller keeps
    reading. Deterministic, calendar-only (no time-of-day) so two notes
    written the same day render identically regardless of when the caller
    happens to recall them."""
    import datetime as _dt
    return _dt.datetime.fromtimestamp(created_at).strftime("%Y-%m-%d")


def _note_title(note: WorkingNote) -> str:
    """The display title for one note: the caller-declared `title`, or the
    first content line (truncated) when none was given. Hoisted out of
    `_format_index_line` so `resume_state()`'s JSON note summaries (UPG-
    RESUME-SURFACE) derive a title the identical way the index-tier render
    does, rather than re-deriving the same fallback separately."""
    return note.title or (
        note.content.strip().splitlines()[0][:80] if note.content.strip() else "(no title)"
    )


def _format_index_line(
    note: WorkingNote,
    stale_warnings: dict[int, list[str]],
    *,
    surface: str = "mcp",
    sort_by: str = "relevance",
    note_states: dict[int, dict] | None = None,
) -> str:
    """One note's single index-tier line — hoisted from
    `format_notes_for_llm()`'s detail='index' loop body (unchanged rendering)
    so `fire_and_format()` (TRIGGER-ENGINE wave 2a) can render the same
    index-tier line for a trigger-fired note without duplicating the
    format.

    `sort_by='chronological'` (UPG-DECISION-TIMELINE) swaps the trailing
    relative age for the note's creation date, so a chronological listing —
    typically `vectr_recall(kind="decision", sort_by="chronological")`, an
    ADR-style decision timeline — reads as a dated sequence instead of a set
    of ages. Every other `sort_by` value renders exactly as before this
    branch existed.

    `note_states` (UPG-MEMORY-STATE-MACHINE §4.1), when given, is the
    `note_event_states()` fold result — an entry with state=='revoked' adds
    a `[REVOKED]` marker, same shape as the pre-existing `[STALE]` marker
    just below (an index line never shows content, so the full anti-memory
    paragraph belongs on `_format_full_block`'s revoked branch instead; this
    is just the signal that expanding this note will show one). Omitted
    (None, the default) renders exactly as before this parameter existed."""
    n = note
    kind_label = n.kind if n.kind else DEFAULT_KIND
    title = _note_title(n)
    stale_marker = " [STALE]" if n.note_id in stale_warnings else ""
    revoked_marker = (
        " [REVOKED]" if note_states and note_states.get(n.note_id, {}).get("state") == "revoked" else ""
    )
    # UPG-MEMORY-DECAY-KIND-SCOPED: same signal shape as revoked_marker
    # above, for a note whose kind's TTL elapsed (`purge_expired_notes()`).
    # An expired note is excluded from default recall()/fire() listings
    # (`_exclude_expired()`), so this marker only ever renders on an
    # explicit `include_superseded`-style or single-note-expand path that
    # bypasses that filter — never on a fresh default listing.
    expired_marker = (
        " [EXPIRED]" if note_states and note_states.get(n.note_id, {}).get("state") == "expired" else ""
    )
    id_str = f"#{n.note_id}" if surface == "mcp" else f"{n.note_id}"
    # UPG-SUBAGENT-MEMORY: caller-declared agent/subagent attribution
    # (author_id) — never inferred. Absent renders exactly as before.
    agent_marker = f" ({n.author_id})" if n.author_id else ""
    if sort_by == "chronological":
        return (
            f"[{id_str}] {_date_str(n.created_at)} {kind_label}/{n.priority}{agent_marker}"
            f" · {title}{stale_marker}{revoked_marker}{expired_marker}"
        )
    return (
        f"[{id_str}] {kind_label}/{n.priority}{agent_marker} · {title}"
        f"  ({_age_str(n.created_at)}){stale_marker}{revoked_marker}{expired_marker}"
    )


# Anti-memory injection framing (UPG-MEMORY-STATE-MACHINE §4.3) — an output
# template, not a query-classification list, so it lives here as a plain
# constant rather than in config.yaml (same precedent as trigger_engine.py's
# _HUMAN_FRAME/_AGENT_FRAME/_AUTO_FRAME provenance-framing templates: fixed
# text shaping how a fact is PRESENTED, never text that inspects or reroutes
# on prompt/query content). Wording is the design doc's literal template
# (memoization-l1-capture-design.md §4.3); field ORDER is not — the design
# doc writes the warning last, but every surface that renders this template
# truncates from the right under a budget (agent/proactive/matcher.py's
# `_cap()` for the proxy channel; agent/trigger_engine.py's
# `_trim_full_text_to_cap()` for the hook full-tier channel), so a
# right-truncation can only ever remove trailing text.
#
# UPG-DETERRENT-TITLE-ONLY: the warning clause is placed FIRST so it can
# never be the thing a tight budget cuts — before this, only the proxy
# channel protected it, via a runtime `rpartition(". ")` reorder local to
# `_revoked_summary()`; the hook channel's `_trim_full_text_to_cap()` ran
# the RAW (unreordered) template through a generic right-truncation with no
# template awareness at all, so a revoked note routed through a tight
# per-kind cap (e.g. `gotcha`'s 100-token/400-char cap) could lose the
# warning clause ENTIRELY while still rendering "Previously believed: ..."
# — the exact silent-caveat-loss the design doc's poisoning-mitigation row
# rules out ("visible caveat, never silent suppression"). Putting the
# warning first in the template itself (rather than reordering it per
# caller) protects it on every surface uniformly, and lets
# `_revoked_summary()`'s runtime reorder be deleted as redundant.
#
# `reason` is placed second (ahead of the quoted summary) for the same
# right-truncation reason: `summary` is already a bounded one-line title-or-
# first-line excerpt (`_note_title()`, <=80 chars, itself the design doc's
# literal "<one-line content summary>" field — full multi-line content is
# deliberately never rendered here, see tests/test_note_events.py::
# TestAntiMemoryRendering::test_revoked_note_renders_deterrent_not_raw_content),
# while `reason` is free text with no cap of its own — ordering it before
# the already-small summary means a right-truncation sacrifices the summary
# first, matching benchmarks/anti_memory/DESIGN.md §13's T0-4 probe (reason
# must survive verbatim on the proxy surface) by construction instead of by
# accident of today's generous default budgets.
_ANTI_MEMORY_TEMPLATE = (
    'Do not re-derive this from other sources without verification. '
    'Previously believed (recorded {created_date}, revoked {revoked_date}, '
    'reason: {reason}): "{summary}".'
)


# Expiry deterrent framing (UPG-MEMORY-DECAY-KIND-SCOPED) — same category as
# _ANTI_MEMORY_TEMPLATE above (an output template, not a query-
# classification list), but a distinct wording: expiry is an age-based
# noise-reduction signal, not a correctness verdict the way revocation is,
# so this deliberately does NOT say "do not re-derive" — the content may
# still be true, it has simply aged out of default injection and is worth a
# fresh look rather than a warning against re-deriving it.
_EXPIRED_TEMPLATE = (
    'Expired by age (recorded {created_date}, expired {expired_date}, '
    'reason: {reason}): "{summary}". Row and event history retained — '
    'use vectr_reinstate to bring it back into active recall.'
)


# Serving-policy hardening (UPG-MEMORY-STATE-MACHINE §5.6) — the unified
# framing template for TRIGGER-fired (injected) notes: "structural trust,
# never imperative, no hedge adjectives". Replaces trigger_engine.py's
# provenance-hedged `frame_prefix()` (DIRECTIVE/"memory to verify"/"auto-
# captured") ONLY on this delivery path (`fire_and_format()` via
# `_format_full_block(..., injected=True)`) — a note pulled through a direct
# `vectr_recall`/manual-expand call keeps the existing provenance framing
# unchanged, since that path is a deliberate caller query, not an unsolicited
# injection the caller needs to weigh trust on sight for. A fixed protocol
# string shaping how a fact is PRESENTED (same category as _ANTI_MEMORY_
# TEMPLATE and trigger_engine.py's own frame constants), not query-
# classification, so it lives here as a plain constant rather than in
# config.yaml.
_INJECTED_FRAME_TEMPLATE = "Recorded {date} (anchor: {target}, status: {status}): "

# UPG-ANCHOR-UNOBSERVED-BINDING: the unobserved-binding caveat, appended to
# `_injected_frame()`'s `status` field only when the note's first anchor was
# declared with observed=False (never on None/unknown — see
# `_anchor_observed_at_write()`). Same category as _INJECTED_FRAME_TEMPLATE/
# _ANTI_MEMORY_TEMPLATE/_EXPIRED_TEMPLATE just above: a fixed protocol string
# shaping how a fact is PRESENTED, not query-classification, so it lives here
# as a plain constant rather than in config.yaml. Deliberately worded
# DIFFERENTLY from drift's "changed since — verify": drift means the FILE
# changed after the anchor hash was recorded and the remedy is re-deriving
# the note from the current source; this means the WRITING agent never
# actually opened the file it anchored to and the remedy is a first read —
# conflating the two into one "stale" phrase would erase that distinction
# (the acceptance bar this task was written against explicitly rules out
# merging them).
_UNOBSERVED_STATUS_SUFFIX = "not verified as read this session — recommend a first read, not re-derivation"


def _injected_frame(note: WorkingNote, stale_warnings: dict[int, list[str]]) -> str:
    """The structural-trust framing prefix for one ACTIVE (non-revoked)
    injected note (§5.6/§4.4). `anchor: <target>` names the note's first
    declared anchor path, or "none" when it has no declared anchors —
    proxy anchors (§4.4: lockfiles/CI configs/etc standing in for "the
    process they encode") are anchors like any other, so no special-casing
    is needed here beyond reading `note.anchors[0]`.

    `status` is derived from the SAME deterministic anchor-drift signal
    `check_staleness()` already writes into `stale_warnings` (the
    `[anchor_changed]`-suffixed reason, §4.4) — never re-derived here, just
    read: drift present -> "changed since — verify". No drift and the note
    is kind="operational" -> "last confirmed <date>" (§4.4 "Option D,
    unconditional": operational facts carry a recency verdict even when
    nothing has been proven to have drifted, since env/process facts decay
    by elapsed time, not just by a hash mismatch). Otherwise -> "matches
    current state".

    Drift on a kind="operational" note gets one further, PURELY ADDITIVE
    refinement: the leading "changed since — verify" text is unchanged
    (never reordered, never dropped — an operational note's anchor having
    drifted is still exactly as urgent as any other note's), but a strict
    suffix is appended naming the anchor target and the note's original
    "last confirmed" date. This is the honest reading of a §4.4 proxy
    anchor: a lockfile/CI-config/Dockerfile/etc. anchored to an operational
    note stands in for the PROCESS it encodes, not for the note's content
    directly, so its drift means "the process this note describes may have
    changed" — a signal to re-verify, never a claim that the note itself is
    now wrong. Naming that plainly (rather than folding it into the same
    unqualified "changed since — verify" a code-anchored note gets) keeps
    the caller from over-trusting a stale environment fact while also not
    under-selling it as a flat contradiction. Non-operational drift and the
    no-drift branches are completely untouched by this refinement.

    UPG-ANCHOR-UNOBSERVED-BINDING adds one further, independently-gated,
    PURELY ADDITIVE suffix: when the note's first anchor was written with
    observed=False (`note.anchors[0][2]` — the third element `remember()`
    now stores; see `_anchor_observed_at_write()`), `_UNOBSERVED_STATUS_
    SUFFIX` is appended to whatever `status` the drift logic above already
    produced. This is deliberately independent of, and can compose with,
    the drift branches above (a note can be both drifted AND unobserved —
    two orthogonal true facts, both worth saying) rather than being folded
    into the SAME `anchor_drifted` boolean, per this task's own acceptance
    bar: drift and non-observation are different failure modes with
    different remedies and must never collapse into one merged "stale"
    flag. The observed value is read via `len(anchor) >= 3` — an anchor
    pair written before this wave (or one written by a session with no
    observation ledger, see the None branch below) has no third element or
    stores `None`, and BOTH render with no suffix at all: only a literal
    `False` (never `None`, never a missing element) ever adds the caveat,
    so this is byte-identical to pre-this-task output for every anchor this
    task doesn't have positive evidence about."""
    date = _date_str(note.created_at)
    first_anchor = note.anchors[0] if note.anchors and note.anchors[0] else None
    target = first_anchor[0] if first_anchor and first_anchor[0] else "none"
    stale_files = stale_warnings.get(note.note_id, [])
    anchor_drifted = any(f.endswith("[anchor_changed]") for f in stale_files)
    if anchor_drifted and note.kind == "operational":
        status = f"changed since — verify; {target} is a proxy for this process, last confirmed {date}"
    elif anchor_drifted:
        status = "changed since — verify"
    elif note.kind == "operational":
        status = f"last confirmed {date}"
    else:
        status = "matches current state"
    observed = first_anchor[2] if first_anchor and len(first_anchor) >= 3 else None
    if observed is False:
        status = f"{status}; {_UNOBSERVED_STATUS_SUFFIX}"
    return _INJECTED_FRAME_TEMPLATE.format(date=date, target=target, status=status)


def _scope_label(note: WorkingNote) -> str:
    """Human-readable scope label for the 'full' detail render
    (UPG-SCOPE-SURFACE-BACK): the bare scope value, or 'branch (<name>)' when
    scope=='branch' and a branch was actually captured at write time. Lets a
    caller diagnose why a scoped note does or doesn't fire (e.g. a note
    scoped to a stale "branch (old-feature)" no longer matches the current
    branch) without a separate lookup — a resolved scope was previously
    write-only, visible nowhere after remember() returned the note id."""
    if note.scope == "branch" and note.branch:
        return f"branch ({note.branch})"
    return note.scope


def _format_full_block(
    note: WorkingNote,
    stale_warnings: dict[int, list[str]],
    note_states: dict[int, dict] | None = None,
    injected: bool = False,
) -> str:
    """One note's multi-line 'full' detail block (age, tags, author,
    kind/provenance markers, provenance-framed content, staleness warnings)
    — hoisted from `format_notes_for_llm()`'s detail='full' loop body
    (unchanged rendering) so `fire_and_format()` (TRIGGER-ENGINE wave 2a) can
    render the same block for a trigger-fired note. Ends with a trailing
    blank line (its own line list's last element is "") so that joining
    several blocks with "\\n" reproduces the exact spacing
    `format_notes_for_llm`'s original flat per-line loop produced.

    `note_states` (UPG-MEMORY-STATE-MACHINE §4.1), when given, is the
    `note_event_states()` fold result. Omitted (None, the default) renders
    exactly as before this parameter existed.

    `injected` (§5.6, wave 3), when true, renders the ACTIVE-note content
    line through the unified structural-trust framing template
    (`_injected_frame()`) instead of `frame_prefix()`'s provenance-hedged
    imperative wording — `fire_and_format()` is the only caller that ever
    passes True (an unsolicited trigger delivery); a direct
    `vectr_recall`/manual-expand call via `format_notes_for_llm()` never
    does, and keeps today's framing unchanged. The revoked/anti-memory
    branch immediately below is COMPLETELY UNAFFECTED by this flag — §4.3's
    "Anti-memory shares the per-turn ledger and budget like any injection"
    means anti-memory composes WITH the injected surface, not that it also
    adopts this content-line template; its own deterrent wording already
    is the injected framing for a revoked note."""
    from agent.trigger_engine import frame_prefix

    n = note
    age_str = _age_str(n.created_at) + " ago"
    tag_str = f"  [{', '.join(n.tags)}]" if n.tags else ""
    author_str = f"  @{n.author_id}" if n.author_id else ""
    stale_files = stale_warnings.get(n.note_id, [])
    stale_marker = " [STALE]" if stale_files else ""

    # Anti-memory (§4.3): a revoked note substitutes this deterrent block for
    # its raw content ENTIRELY — additive to the note's continued presence
    # in the result set (never dropped, per the design doc), never a silent
    # exclusion. Takes precedence over every other marker below (kind/
    # provenance/scope/superseded) when note_states reports 'revoked' —
    # revoked is definitionally the note's most recent lifecycle transition
    # whenever the fold reports it, so it is the one signal that matters;
    # rendering it alongside a possibly-stale superseded badge from an
    # earlier, now-superseded-by-events transition would only muddy the one
    # thing this block exists to communicate.
    state_info = (note_states or {}).get(n.note_id)
    if state_info is not None and state_info["state"] == "revoked":
        revoked_date = _date_str(state_info["ts"]) if state_info["ts"] else _date_str(n.created_at)
        reason = state_info["reason"] or "no reason given"
        anti_memory = _ANTI_MEMORY_TEMPLATE.format(
            created_date=_date_str(n.created_at),
            revoked_date=revoked_date,
            reason=reason,
            summary=_note_title(n),
        )
        return "\n".join([
            f"[{n.note_id}] [REVOKED]{tag_str}{author_str}",
            f"  {anti_memory}",
            "",
        ])

    # Expiry deterrent (UPG-MEMORY-DECAY-KIND-SCOPED) — same shape as the
    # revoked branch above, checked next so a note whose LATEST folded
    # transition is 'expired' (not 'revoked') renders this block instead.
    # Never reached via default recall()/fire() (those exclude expired
    # notes entirely via _exclude_expired()) — only via get_note()'s
    # unfiltered single-note expand path, which is exactly what "deterrent
    # rendering survives" requires.
    if state_info is not None and state_info["state"] == "expired":
        expired_date = _date_str(state_info["ts"]) if state_info["ts"] else _date_str(n.created_at)
        reason = state_info["reason"] or "ttl exceeded"
        deterrent = _EXPIRED_TEMPLATE.format(
            created_date=_date_str(n.created_at),
            expired_date=expired_date,
            reason=reason,
            summary=_note_title(n),
        )
        return "\n".join([
            f"[{n.note_id}] [EXPIRED]{tag_str}{author_str}",
            f"  {deterrent}",
            "",
        ])

    # superseded badge
    superseded_marker = ""
    if n.valid_until is not None:
        sup_by = n.superseded_by or (
            f"note#{n.superseded_by_note_id}" if n.superseded_by_note_id else None
        )
        import datetime as _dt
        sup_date = _dt.datetime.fromtimestamp(n.superseded_at or n.valid_until).strftime("%Y-%m-%d")
        if sup_by:
            superseded_marker = f" [superseded by @{sup_by}, {sup_date}]"
        else:
            # UPG-NOTE-RETIRE-POST-HOC: a note retired with no successor
            # link (`supersede_note(superseded_by=None)` — the only path
            # that can leave valid_until set with neither identifier).
            # Factual framing, never blank: an excluded-from-recall note
            # that renders as if active on the include-superseded surfaces
            # would mislead exactly the reader who dug it up.
            superseded_marker = f" [superseded, {sup_date}]"

    # Surface the kind when it carries injection semantics (UPG-9.3) —
    # 'finding' is the default and adds no signal, so it's left implicit.
    kind_marker = f" [{n.kind.upper()}]" if n.kind and n.kind != DEFAULT_KIND else ""
    # Provenance class (TRIGGER-ENGINE, bm2-design-skeleton.md §5) — marked
    # on every full-tier block, unlike kind_marker above, since the caller's
    # trust posture depends on it regardless of whether provenance is the
    # default ("agent").
    provenance_marker = f" [{n.provenance}]"
    # UPG-SCOPE-SURFACE-BACK: surface the RESOLVED scope (and, for
    # scope=="branch", the captured branch) on every full-tier block —
    # additive only, index-tier lines are untouched (token-budgeted).
    scope_marker = f" [scope={_scope_label(n)}]"

    if injected:
        # Serving-policy hardening (§5.6): structural trust, never
        # imperative — the verdict comes from deterministic machine state
        # (anchor hash match/drift, or an operational note's recency),
        # never a hedge adjective.
        content_line = f"  {_injected_frame(n, stale_warnings)}{n.content}"
    else:
        # Provenance framing (§5): only a human-provenance directive ever
        # renders as an unhedged imperative; agent-provenance is framed as
        # memory to verify; auto-provenance carries the weakest framing.
        # UPG-PROVENANCE-AUTOBIND-SPAN: passing user_quote/content lets
        # frame_prefix distinguish a user-stated note whose bound quote IS
        # its whole body (unqualified framing, unchanged) from one where the
        # quote is only a sub-span of a larger agent-authored body (the
        # qualified span framing) — a no-op for every other provenance.
        content_line = f"  {frame_prefix(n.provenance, n.kind, user_quote=n.user_quote, content=n.content)}{n.content}"

    lines = [
        f"[{n.note_id}] [{n.priority.upper()}]{kind_marker}{provenance_marker}{scope_marker}{tag_str}{author_str}  ({age_str})"
        f"{stale_marker}{superseded_marker}",
        content_line,
    ]
    if stale_files:
        changed = ", ".join(stale_files)
        lines.append(f"  WARNING: These files changed after this note was written: {changed}")
        # UPG-MEMORY-STATE-MACHINE §4.4: a declared-anchor drift (including a
        # proxy anchor — a lockfile/CI-config/etc anchored the same way as
        # any code file, standing in for "the process it encodes") gets its
        # own structural verdict line, additive to the combined WARNING
        # above (unchanged — still covers mtime/code_hash/symbol drift the
        # same way it always has). "structural trust, not adjectives": the
        # verdict names what changed and says verify, nothing more.
        anchor_drift = [f for f in stale_files if f.endswith("[anchor_changed]")]
        if anchor_drift:
            lines.append(f"  VERDICT: anchor changed since — verify: {', '.join(anchor_drift)}")
        lines.append(f"  WARNING: Verify this note is still accurate before relying on it.")
    lines.append("")
    return "\n".join(lines)


# Metadata key stamped on the 'working_memory' ChromaDB collection recording
# the embed model that produced its CURRENT vectors (UPG-NOTES-EMBED-MIGRATION).
# Mirrors CodeIndexer's embed-model stamp for the code index (see
# agent/indexer/_constants.py's _EMBED_MODEL_STAMP_FILE docstring), but notes
# are irreplaceable user memory rather than a rebuildable derived index, so a
# stamp mismatch here triggers an in-place re-embed + vector update instead of
# a drop-and-rebuild.
_NOTES_EMBED_MODEL_KEY = "embed_model"

# Intentionally NOT in config.yaml (Tier-3, same category as the indexer's
# _EMBED_BATCH_SIZE): a pure throughput lever with no effect on ranking,
# recall behavior, or note content — only how many texts are handed to
# embed_fn per call during a one-time startup migration.
_NOTES_REEMBED_BATCH_SIZE = 256


def _is_embedding_dimension_conflict(exc: BaseException) -> bool:
    """True when chromadb rejects a write because the vectors' dimensionality
    does not match what the collection was created with.

    The wording differs by chromadb backend — the local/rust backend says
    "Collection expecting embedding with dimension of 4, got 6", older Python
    backends say "embedding dimension mismatch" — and the error arrives as
    chromadb.errors.InvalidArgumentError. Both wordings put the two words
    `embedding` and `dimension` in the message, so that is what is matched;
    the caller only ever applies this to a chromadb error raised by a note
    vector upsert, where no other failure mode talks about both.
    """
    message = str(exc).lower()
    return "embedding" in message and "dimension" in message

# SQLite busy-wait for a contended write lock (team mode: concurrent clients +
# the CLI can share one workspace's notes DB). Intentionally NOT in config.yaml
# — a robustness/timeout knob, same category as the throughput constants above.
_SQLITE_BUSY_TIMEOUT_S = 5.0

# Over-fetch multiplier behind working_memory_fetch_width(): how many
# candidates a 'working_memory' vector query fetches per note it intends to
# render, so that hits consumed by other workspaces' notes don't starve the
# result. Kept as a named constant so the helper below is the single place
# the number lives.
_WORKING_MEMORY_OVERFETCH_MULTIPLIER = 3


def working_memory_fetch_width(render_limit: int, col_count: int, *, floor: int = 0) -> int:
    """How many candidates a 'working_memory' vector query must fetch to be able
    to render up to `render_limit` notes.

    INVARIANT for every vector-query path against this collection: size
    n_results through this helper, never with a raw render limit. The
    'working_memory' collection is GLOBAL — one ChromaDB collection shared by
    every workspace a daemon serves, queried with no per-workspace `where=`
    clause — so every filter (other workspaces, valid_until expiry,
    superseded/folded lifecycle state, similarity cutoffs) runs in SQLite or
    Python only AFTER the vector search has already chosen its hits. A query
    sized exactly to its render budget can have every slot taken by other
    workspaces' notes and legitimately render nothing. Fetching a multiple of
    the render limit keeps same-workspace survivors likely; `floor` lets a
    caller demand a deeper pool than the multiplier implies when its contract
    needs one; capping at `col_count` avoids ChromaDB errors on collections
    smaller than the ask.
    """
    return min(max(render_limit * _WORKING_MEMORY_OVERFETCH_MULTIPLIER, floor), col_count)


class WorkingContextStore:
    """
    SQLite-backed store for LLM working notes and session snapshots.

    Design principle: the LLM should never be afraid to forget something
    if Vectr has it. This store is the guarantee.

    When embed_fn and notes_chroma_client are provided, note content is embedded
    at remember() time and stored in a ChromaDB 'working_memory' collection.
    recall(query=...) then uses cosine similarity to find relevant notes instead
    of SQL LIKE substring matching. SQL LIKE is retained as a fallback.

    embed_fn embeds document-side text (note content being stored); embed_query_fn
    embeds the recall query. These are kept distinct because asymmetric embedding
    models require a different prompt for queries than for the passages they're
    matched against — reusing embed_fn for both would silently skip that prompt on
    every recall. Callers that don't care about the distinction (e.g. tests with a
    plain symmetric stand-in) may omit embed_query_fn; it then defaults to embed_fn.

    embed_model, when given, identifies the model embed_fn/embed_query_fn currently
    embed with (e.g. "ibm-granite/granite-embedding-english-r2"). It is stamped onto
    the 'working_memory' collection's metadata (UPG-NOTES-EMBED-MIGRATION). If the
    stamp on an existing collection differs from embed_model — including a MISSING
    stamp on a collection that already holds vectors, since we cannot know what
    model produced those — every active note's content is re-embedded with the
    current embed_fn and the collection's vectors are updated in place before the
    constructor returns, so note vectors and query vectors are never silently drawn
    from two different embedding spaces. This mirrors CodeIndexer's embed-model
    stamp for the code index, but migrates rather than drops: notes are
    irreplaceable user memory, and re-embedding a few hundred short texts is cheap.
    embed_model is optional and defaults to None, in which case no stamping or
    migration happens at all (existing constructions and tests are unaffected).
    """

    def __init__(
        self,
        db_dir: str,
        embed_fn=None,
        notes_chroma_client=None,
        embed_query_fn=None,
        embed_model: str | None = None,
    ) -> None:
        self._db_path = Path(db_dir) / "working_context.sqlite"
        self._encryptor: _NoteEncryptor | None = _build_encryptor()
        # Semantic recall: embed notes at write time, cosine search at recall time
        self._embed_fn = embed_fn   # Callable[[list[str]], list[list[float]]] | None
        self._embed_query_fn = embed_query_fn or embed_fn  # query-mode embed, defaults to embed_fn
        self._embed_model = embed_model  # current model name, for the embed-model stamp/migration
        self._notes_col = None
        # Kept so the embed-model migration can rebuild the collection when a
        # model change changes vector dimensionality (see
        # `_recreate_notes_collection`).
        self._notes_chroma_client = notes_chroma_client
        # Set by `_reembed_all_notes` when it had to rebuild `working_memory`.
        self._notes_collection_recreated = False
        # TRIGGER-ENGINE wave 2b — the S (symbol) primitive's resolver. None
        # until attach_symbol_resolver() runs (phase-2 service init, once the
        # code symbol graph is built); a memory-only daemon or a warm-up
        # window before then simply never has one, and symbol triggers
        # deterministically never fire — see attach_symbol_resolver()'s
        # docstring, the same "attach after construction" shape as the
        # embedder below.
        self._symbol_resolver = None
        # UPG-ANCHOR-UNOBSERVED-BINDING: per-session, in-memory ledger of file
        # paths this session's PreToolUse hook calls have already reported
        # (`fire()` records into it; `remember()` reads it to mark each
        # declared anchor observed/unobserved/unknown). Keyed by
        # (workspace, session_id) -> set of path-candidate strings (both the
        # as-given and workspace-relative forms `_path_trigger_candidates()`
        # already computes, so it matches an anchor declared in either form).
        # Deliberately NEVER persisted to the sqlite DB or any other durable
        # store: a daemon restart, a stdio transport with no session_id, or a
        # team-mode bind that never routes hook events through this process
        # all mean "no record", and the whole point of the tri-state design
        # (see `_anchor_observed_at_write()`) is that "no record" must read
        # as unknown, never as a false "never observed" accusation — an
        # ephemeral, LRU-bounded dict is what makes that distinction free:
        # a key that was never populated this process's lifetime is simply
        # absent, the same shape as one deliberately never touched.
        # LRU-bounded by EVICTION_MAX_TRACKED_SESSIONS, same cap and same
        # "delete-then-reinsert to touch" eviction shape VectrService already
        # applies to its own per-session ledgers (_trigger_ledgers/
        # _turn_ledgers/_session_advisors in app/service.py) — kept here
        # instead of there because populating it requires the workspace-root
        # path normalization `_path_trigger_candidates()` performs, and that
        # normalization is deliberately confined to the fire()/remember()
        # boundary in this module (trigger_engine.py's filesystem-purity
        # invariant), not duplicated into the service layer.
        self._observed_paths: dict[tuple[str, str], set[str]] = {}
        # Guards attach_embedder() (UPG-STDIO-MEMORY-READY): the store can be
        # constructed with embed_fn=None (memory tools live before the
        # embedding model has loaded) and upgraded to a real embedder later,
        # from a background thread, once phase-2 service init completes. The
        # lock makes that upgrade idempotent and keeps a concurrent
        # remember()/recall() from ever observing self._notes_col set while
        # self._embed_fn is still the old value (or vice versa). Also guards
        # attach_symbol_resolver() (same idempotent-upgrade shape).
        self._attach_lock = threading.Lock()
        if embed_fn is not None and notes_chroma_client is not None and not self._vectors_disabled():
            try:
                # GLOBAL collection: shared by every workspace this daemon
                # serves, with no per-workspace `where=` scoping — any query
                # path added against it must over-fetch before filtering (see
                # working_memory_fetch_width()). Two more contract facts:
                # `hnsw:space=cosine` is what makes this repo's
                # `similarity = 1 - distance` arithmetic valid (Chroma's l2
                # default would make it meaningless), and chroma ids come
                # from SQLite's GLOBAL INTEGER PRIMARY KEY AUTOINCREMENT, so
                # ids never collide across workspaces.
                self._notes_col = notes_chroma_client.get_or_create_collection(
                    name="working_memory",
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception:
                pass  # embedding unavailable — fall back to SQL LIKE silently
        self._init_db()
        if self._notes_col is not None and self._embed_model:
            try:
                self._reconcile_embed_model_stamp()
            except Exception:
                logger.warning(
                    "vectr: working-memory embed-model migration failed — "
                    "note vectors may still be in a stale embedding space; "
                    "will retry on next startup",
                    exc_info=True,
                )

    def _vectors_disabled(self) -> bool:
        """Strict encryption posture (VECTR_ENCRYPT_DISABLE_NOTE_VECTORS): when
        encryption is on, the note embedding vectors are a lossy plaintext
        projection of note content living in the Chroma store. Setting this
        omits them entirely — recall falls back to lexical SQL LIKE — so no
        representation of note content leaves the encrypted SQLite column.
        Shared by __init__ and attach_embedder() so both paths honor it the
        same way regardless of when the embedder becomes available."""
        return (
            self._encryptor is not None
            and os.getenv("VECTR_ENCRYPT_DISABLE_NOTE_VECTORS", "") == "1"
        )

    @property
    def embedder_ready(self) -> bool:
        """True once an embedder is attached — either passed to __init__ or
        via a later attach_embedder() call (UPG-STDIO-MEMORY-READY).

        False means remember()/recall() are lexical/SQL-only for now: notes
        still write and read correctly, just without semantic ranking or
        vectors, until an embedder attaches."""
        return self._embed_fn is not None

    def attach_embedder(
        self,
        embed_fn,
        notes_chroma_client,
        embed_query_fn=None,
        embed_model: str | None = None,
    ) -> None:
        """Upgrade an embedder-less store (constructed with embed_fn=None) to
        semantic recall, once an embedding model becomes available
        (UPG-STDIO-MEMORY-READY).

        Lets memory tools (remember/recall/forget/status/snapshot) work from
        process start, before the embedding model has finished loading or
        downloading, on every transport — the store itself never needs an
        embedder to read or write a note. This method is how the store is
        upgraded once phase-2 service init (CodeIndexer's embed provider)
        completes in the background, without re-writing or losing any note
        recorded during the window it was missing.

        Idempotent: a second call is a no-op once an embedder is already
        attached. Thread-safe: `self._notes_col` and `self._embed_fn` are
        only ever set together inside `_attach_lock`, and `self._embed_fn`
        (the field every reader gates on) is set last — a concurrent
        remember()/recall() either sees the fully-attached state or the
        original embedder-less state, never a half-attached mix.

        After attaching, runs the existing embed-model stamp reconcile (in
        case the configured model differs from what a previous run stamped)
        and then backfills a vector for every note that doesn't have one yet
        — the notes written during the window this store had no embedder at
        all. Both steps are best-effort: a failure here never raises, since
        the store is already fully usable via the SQL fallback.
        """
        with self._attach_lock:
            if self._embed_fn is not None:
                return  # already attached
            if embed_fn is None or notes_chroma_client is None or self._vectors_disabled():
                return
            try:
                # GLOBAL collection — same invariant as the __init__ creation
                # site above: query paths must over-fetch before filtering
                # (working_memory_fetch_width()).
                notes_col = notes_chroma_client.get_or_create_collection(
                    name="working_memory",
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception:
                return  # embedding still unavailable — stay on SQL LIKE fallback
            self._embed_query_fn = embed_query_fn or embed_fn
            self._embed_model = embed_model
            self._notes_col = notes_col
            self._notes_chroma_client = notes_chroma_client
            self._embed_fn = embed_fn

        if self._embed_model:
            try:
                self._reconcile_embed_model_stamp()
            except Exception:
                logger.warning(
                    "vectr: working-memory embed-model migration failed after "
                    "attach — note vectors may still be in a stale embedding "
                    "space; will retry on next startup",
                    exc_info=True,
                )
        try:
            self.backfill_missing_vectors()
        except Exception:
            logger.warning(
                "vectr: working-memory vector backfill failed after attach — "
                "notes written before the embedder was ready may not be "
                "semantically recallable until the next restart",
                exc_info=True,
            )

    def attach_symbol_resolver(self, symbol_graph) -> None:
        """Upgrade the store to the S (symbol) trigger primitive, once the
        code symbol graph is built (TRIGGER-ENGINE wave 2b,
        bm2-design-skeleton.md §2) — the same "attach after construction"
        shape as `attach_embedder()` above, since `SymbolGraph` is built in
        `VectrService`'s phase-2 search-layer init, after this store already
        exists and is already serving remember/recall.

        `symbol_graph` is duck-typed to `SymbolGraph`'s own
        `symbols_touching_file(workspace, file_path)` / `signature_hash
        (workspace, name)` methods — this store never imports SymbolGraph
        itself, keeping the dependency direction the caller's choice (same
        reasoning as accepting a bare `embed_fn` callable rather than an
        indexer instance).

        Idempotent: a second call is a no-op once a resolver is already
        attached. Thread-safe: shares `_attach_lock` with `attach_embedder()`
        since both are one-shot upgrades performed once from the same
        phase-2 init path.

        Before this is called (memory-only daemon, or the warm-up window
        before phase-2 completes), `self._symbol_resolver` stays None — every
        S trigger deterministically does not fire (see `fire()`), never an
        error."""
        with self._attach_lock:
            if self._symbol_resolver is not None or symbol_graph is None:
                return
            self._symbol_resolver = symbol_graph

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=_SQLITE_BUSY_TIMEOUT_S)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        # Team mode: several clients (and the CLI) can hit one workspace's notes
        # DB concurrently. WAL allows concurrent readers + one writer; busy_timeout
        # makes a would-be second writer wait for the lock instead of immediately
        # raising "database is locked". note_id is AUTOINCREMENT, so IDs stay
        # unique under concurrent inserts once writes are serialized by the lock.
        conn.execute(f"PRAGMA busy_timeout={int(_SQLITE_BUSY_TIMEOUT_S * 1000)}")
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS notes (
                    note_id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    workspace           TEXT NOT NULL,
                    content             TEXT NOT NULL,
                    tags                TEXT NOT NULL DEFAULT '[]',
                    priority            TEXT NOT NULL DEFAULT 'medium',
                    kind                TEXT NOT NULL DEFAULT 'finding',
                    created_at          REAL NOT NULL,
                    last_accessed       REAL NOT NULL,
                    session_id          TEXT,
                    decay_score         REAL NOT NULL DEFAULT 1.0,
                    author_id           TEXT NOT NULL DEFAULT '',
                    author_trust_score  REAL NOT NULL DEFAULT 1.0,
                    valid_from          REAL NOT NULL DEFAULT 0.0,
                    valid_until         REAL,
                    code_hash           TEXT NOT NULL DEFAULT '',
                    superseded_by       TEXT,
                    superseded_at       REAL,
                    pinned              INTEGER NOT NULL DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_notes_workspace ON notes(workspace);
                CREATE INDEX IF NOT EXISTS idx_notes_tags ON notes(tags);

                CREATE TABLE IF NOT EXISTS author_trust (
                    workspace           TEXT NOT NULL,
                    author_id           TEXT NOT NULL,
                    trust_score         REAL NOT NULL DEFAULT 1.0,
                    note_count          INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (workspace, author_id)
                );

                CREATE TABLE IF NOT EXISTS snapshots (
                    snapshot_id  TEXT PRIMARY KEY,
                    workspace    TEXT NOT NULL,
                    label        TEXT NOT NULL,
                    payload      TEXT NOT NULL,
                    created_at   REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_snap_workspace ON snapshots(workspace);

                -- TRIGGER-ENGINE wave 2b (bm2-design-skeleton.md §2/§5) — the
                -- S (symbol) primitive's write-time index: one row per
                -- (note, trigger-with-a-'symbol'-key). Lets fire() skip the
                -- symbol-graph lookup entirely when a workspace has no
                -- symbol-triggered notes at all, and stores the signature
                -- hash captured at write time for staleness re-checking
                -- (check_staleness()) — mirroring path anchors' content hash,
                -- generalised to code symbols instead of file paths.
                CREATE TABLE IF NOT EXISTS symbol_triggers (
                    workspace       TEXT NOT NULL,
                    note_id         INTEGER NOT NULL,
                    trigger_index   INTEGER NOT NULL,
                    symbol_name     TEXT NOT NULL,
                    signature_hash  TEXT,
                    PRIMARY KEY (workspace, note_id, trigger_index)
                );

                CREATE INDEX IF NOT EXISTS idx_symtrig_workspace ON symbol_triggers(workspace);
                CREATE INDEX IF NOT EXISTS idx_symtrig_name ON symbol_triggers(workspace, symbol_name);
                CREATE INDEX IF NOT EXISTS idx_symtrig_note ON symbol_triggers(workspace, note_id);

                -- UPG-MEMORY-STATE-MACHINE §4.1 — append-only note lifecycle
                -- log. Nothing here is ever UPDATEd or DELETEd (except the
                -- cascade in forget(), which removes a note's rows entirely
                -- along with the note itself); "current state" is always a
                -- fold over this table (see _events.py's fold()), never a
                -- column read directly. `id` (not `ts`) is the fold's
                -- ordering key — see fold()'s docstring for why.
                CREATE TABLE IF NOT EXISTS note_events (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    workspace   TEXT NOT NULL,
                    note_id     INTEGER NOT NULL,
                    event       TEXT NOT NULL,
                    actor       TEXT NOT NULL,
                    reason      TEXT,
                    payload     TEXT,
                    ts          REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_note_events_note ON note_events(workspace, note_id, id);
            """)
            # P4: migrate existing databases that predate P4 columns
            existing_cols = {r[1] for r in conn.execute("PRAGMA table_info(notes)").fetchall()}
            p4_cols = {
                "author_id":          "TEXT NOT NULL DEFAULT ''",
                "author_trust_score": "REAL NOT NULL DEFAULT 1.0",
                "valid_from":         "REAL NOT NULL DEFAULT 0.0",
                "valid_until":        "REAL",
                "code_hash":          "TEXT NOT NULL DEFAULT ''",
                "superseded_by":      "TEXT",
                "superseded_at":      "REAL",
                # UPG-9.3: memory kind dimension — existing rows default to 'finding'.
                "kind":               "TEXT NOT NULL DEFAULT 'finding'",
                # UPG-RECALL-HIERARCHY: per-note title for index-tier display.
                "title":              "TEXT NOT NULL DEFAULT ''",
                # TRIGGER-ENGINE wave 1 (bm2-design-skeleton.md §1) — additive,
                # fully backward-compatible columns. Existing rows default to
                # DEFAULT_PROVENANCE/DEFAULT_SCOPE/no triggers/no anchors, so an
                # existing note's evaluation-time behaviour is unchanged: empty
                # triggers falls back to its kind's default bundle exactly as a
                # brand-new note with no explicit triggers would.
                "triggers":              "TEXT NOT NULL DEFAULT '[]'",
                "provenance":            "TEXT NOT NULL DEFAULT 'agent'",
                "scope":                 "TEXT NOT NULL DEFAULT 'workspace'",
                "anchors":               "TEXT NOT NULL DEFAULT '[]'",
                "supersedes":            "INTEGER",
                # Distinct from the existing `superseded_by` (TEXT author_id,
                # set by the code_hash-conflict path above) — this records the
                # note_id of the memory that explicitly superseded this one via
                # the new `supersedes` write-time parameter.
                "superseded_by_note_id": "INTEGER",
                "last_fired":            "REAL",
                # TRIGGER-ENGINE wave 2a: git branch recorded at write time
                # when scope=="branch"; "" for every other scope, for notes
                # written before this wave, or when git is unavailable.
                "branch":                "TEXT NOT NULL DEFAULT ''",
                # UPG-MEM-PROVENANCE-USER-STATED: the verbatim user-turn
                # excerpt bound to this note's content at write time. Non-empty
                # ONLY on a note whose provenance is 'user-stated' — an excerpt
                # that failed the binding check is never stored, so this column
                # can never hold an unverified "the user said this". Encrypted
                # alongside content/title when encryption is on (it is by
                # construction a substring of content).
                "user_quote":            "TEXT NOT NULL DEFAULT ''",
                # UPG-RECALL-MISS-FLOOR part (b), round 4: explicit pin — the
                # SOLE Tier 0 unconditional-injection membership test in
                # recall() (directive notes were dropped from Tier 0 in
                # round 4; they still get their own unconditional delivery
                # via boot_recall(), which is unrelated to this column).
                # Existing rows default to unpinned (0/False), unchanged behavior.
                "pinned":                "INTEGER NOT NULL DEFAULT 0",
            }
            for col, typedef in p4_cols.items():
                if col not in existing_cols:
                    conn.execute(f"ALTER TABLE notes ADD COLUMN {col} {typedef}")

            # Create indexes that depend on migrated columns — must run AFTER migration
            conn.execute("CREATE INDEX IF NOT EXISTS idx_notes_code_hash ON notes(code_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_notes_valid ON notes(valid_until)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_notes_kind ON notes(kind)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_notes_pinned ON notes(workspace, pinned)")

    # ------------------------------------------------------------------
    # Notes — vectr_remember / vectr_recall / vectr_forget
    # ------------------------------------------------------------------

    def remember(
        self,
        workspace: str,
        content: str,
        tags: list[str] | None = None,
        priority: str = "medium",
        session_id: str | None = None,
        author_id: str = "",
        code_hash: str = "",
        kind: str = DEFAULT_KIND,
        title: str = "",
        triggers: list[dict] | None = None,
        provenance: str = DEFAULT_PROVENANCE,
        scope: str | None = None,
        anchors: list[str] | None = None,
        supersedes: int | None = None,
        contradicts: int | None = None,
        user_quote: str | None = None,
        recent_user_message: str | None = None,
        pin: bool = False,
    ) -> int:
        """Store a working note. Returns the note_id.

        `pin` (UPG-RECALL-MISS-FLOOR part (b)): explicit, caller-declared
        Tier 0 membership at write time — the unconditional-injection
        guarantee in `recall()`, orthogonal to `kind` (a pinned note keeps
        its own kind; a task note can be pinned without becoming a
        directive). As of round 4, pinning is the ONLY way into Tier 0 —
        directive notes no longer enter it (they already have their own
        unconditional, uncapped-per-query delivery via `boot_recall()`).
        Never inferred from content or usage. Use `set_pinned()` to
        pin/unpin an EXISTING note instead of at write time.

        If code_hash is provided and another non-superseded note exists for the
        same workspace + code_hash (same code anchor), the older note is marked
        superseded before the new note is inserted.

        `kind` is one of VALID_KINDS (directive|task|gotcha|finding|reference|
        decision|operational); an unrecognised value falls back to DEFAULT_KIND.

        `title` is a short label for index-tier display (UPG-RECALL-HIERARCHY).
        When empty, a fallback is derived from the first non-empty line of content,
        stripped and truncated to 80 characters.

        Trigger engine wave 1 (TRIGGER-ENGINE, bm2-design-skeleton.md §1/§2/§5)
        — all additive, all optional, one call, zero trigger literacy required:

        `triggers`: explicit P/E/T trigger overrides (see agent/trigger_engine
        .validate_trigger for the shape). Raises ValueError if malformed.
        Omitted/empty means "use this kind's default bundle at evaluation
        time" — evaluation-time, never baked into storage.

        `provenance`: "human", "agent" or "auto". Raises ValueError if not one
        of those, OR if the EFFECTIVE provenance (see `user_quote` below) is
        "auto" and kind="directive" — an unreviewed standing rule is a
        contradiction in terms and is rejected outright rather than silently
        downgraded. "user-stated" is also in PROVENANCE_VALUES but is rejected
        here when passed directly: it is derived from a bound `user_quote`,
        never declared, so that the class always means a machine-verified
        binding rather than a writer's own claim about itself.

        `user_quote` (UPG-MEM-PROVENANCE-USER-STATED): a verbatim excerpt of
        the user turn this note transcribes. Bound by
        `_user_quote.bind_user_quote()` — a whitespace-normalized substring
        check against `content`, deterministic and never semantic. When it
        binds, the note's provenance becomes "user-stated" (a class between
        "human" and "agent": the words are the user's, the transcription is an
        AI session's) and the excerpt is stored on the note. When it does NOT
        bind — too short, or not actually quoted in `content` — the write
        still succeeds, unchanged, at its ordinary provenance, and the excerpt
        is DISCARDED rather than persisted, so the store never holds an
        unverified "the user said this". The rejection reason is not raised
        here (a failed binding is not a failed write); a surface recomputes it
        with the same pure function to tell the caller why. A binding never
        overrides provenance="human" — a person's own endorsement already
        outranks a transcription.

        `recent_user_message` (UPG-PROVENANCE-NEVER-RISES): the harness-layer
        fallback for the exact same "user-stated" upgrade above, tried ONLY
        when `user_quote` did not already bind — `user_quote` is an explicit
        caller claim and always takes priority. `bind_user_quote` requires the
        WRITING AGENT to remember to pass `user_quote=` on every call; on a
        live production corpus this voluntary step was essentially never
        taken, leaving "user-stated" unreachable in practice — the identical
        adoption gap hook-injected recall was built to close on the read
        side. This is the write-side counterpart: the caller (`VectrService.
        remember()`) passes the raw text of the most recently captured
        `UserPromptSubmit` prompt for this workspace, ALREADY recency-checked
        against `memory_write.user_quote.auto_bind_max_age_seconds` before it
        ever reaches here (this method does no clock work of its own — a
        `None` here just means "no recent-enough turn to compare against,"
        indistinguishable from one that was never captured). Bound by
        `_user_quote.bind_user_quote_auto()` — the MIRROR containment check
        (`content` ⊆ `recent_user_message`, not `user_quote` ⊆ `content`),
        same whitespace-normalized deterministic substring comparison, same
        "user-stated" provenance outcome, same never-overrides-"human"
        ceiling, same discard-on-no-bind behaviour.

        `scope`: one of SCOPE_VALUES, or None (the default) to mean OMITTED.
        An explicitly passed scope — including the literal string
        "workspace" — always wins verbatim and is stored as given (raises
        ValueError if not one of SCOPE_VALUES). Omitted (None) resolves HERE,
        at write time, to the note's `kind`'s default scope per
        bm2-design-skeleton.md §1's Default bundles table
        (KIND_DEFAULT_SCOPES in _types.py: task -> "branch", gotcha ->
        "repo"; every other kind keeps "workspace") and the RESOLVED value —
        never a sentinel — is what gets stored on the row (write-time baked;
        a note written before this wave, or under an explicit scope, is
        untouched by any later change to the default table).

        CRITICAL guard (UPG-TRIGGER-SCOPE-KIND-DEFAULTS): a `kind` whose
        default resolves to "branch" only actually gets "branch" when a real
        git branch is captured at write time. A non-git workspace or a
        detached HEAD falls back to "workspace" instead — baking scope=
        "branch" with an empty branch value would exclude the note from
        EVERY future branch forever (`fire()` compares stored branch to the
        branch current at evaluation time; an empty recorded branch never
        equals a real branch name), a silent, permanent, undiscoverable
        scope-exclusion. This guard applies only to the OMITTED-scope
        default-resolution path — an explicitly passed scope="branch" keeps
        its pre-existing documented behaviour (stores an empty branch value
        on a non-git workspace; the caller opted into "branch" scope
        themselves, so the empty-branch outcome is on them, not silent).
        When scope=="branch", the current git branch (`_current_git_branch()`)
        is captured HERE, at write time, and stored on the note — `fire()`
        compares it against the branch current at evaluation time (empty
        string when git is unavailable or `root` isn't a git checkout; such a
        note then never matches, since an empty recorded branch never equals
        a real current branch name).

        `anchors`: a list of workspace-relative (or absolute) file paths this
        note is anchored to. Each path's current content hash is computed
        HERE, at write time (never supplied by the caller) and stored
        alongside the path — `check_staleness()` re-hashes at fire/recall
        time and raises a visible (never silent) staleness caveat on
        mismatch. A path that cannot be read yet (e.g. a file not created
        yet) stores a null hash and is simply never flagged stale until it
        exists.

        UPG-ANCHOR-UNOBSERVED-BINDING: each stored anchor pair also carries a
        third element, the tri-state OBSERVATION verdict — whether THIS
        session ever reported reading/touching this exact path before this
        write (via a real PreToolUse hook call reaching `fire()`; see
        `_anchor_observed_at_write()`). This is evidence the anchor's file
        was actually opened by the writing agent, distinct from the content
        hash above (which only proves what the FILE currently contains, not
        that anyone read it). True/False only when this session's own
        observation ledger exists and does/doesn't contain the path; None
        when no ledger exists for this session at all (no session_id, hooks
        not wired, stdio transport, team-mode bind, daemon restarted
        mid-session) — an absent ledger is never treated as proof of
        non-observation, so it is stored (and later rendered) as unknown,
        never as a false unobserved accusation.

        `supersedes`: the note_id this note explicitly tombstones. The
        target note (looked up in this workspace) has `valid_until`/
        `superseded_at` set (excluding it from recall() by default and from
        ever firing again, per evaluate_note) and `superseded_by_note_id` set
        to this new note's id — kept for audit, never deleted. Raises
        ValueError if the target note does not exist in this workspace, OR if
        the target's provenance is "human" while THIS write's own provenance
        is not — a write-boundary guard, surface-agnostic and provenance-
        based: only a provenance="human" write may tombstone a genuine
        human-reviewed directive; an agent/auto write may not silently
        permanently-mute one this way. A provenance="human" write may still
        supersede anything, including another human note.
        Distinct from the pre-existing `superseded_by` (author_id) column,
        which is set only by the unrelated code_hash-conflict path above.

        `contradicts` (UPG-MEMORY-STATE-MACHINE §4.2): the note_id this note
        asserts was WRONG — semantically distinct from `supersedes`
        ("replaced by something better/newer"): the target note stays
        exactly as recall()/fire() would otherwise show it (no
        `valid_until`, still a live candidate on every surface) but with a
        `revoked` event appended to its `note_events` log, `actor="agent"`,
        `reason=f"contradicted by #{new_note_id}"` — always these exact
        values; there is no separate actor/reason parameter here; a caller
        wanting a custom reason or a human-attributed revocation uses
        `revoke_note()`/`vectr_revoke` directly instead. Every surface that
        renders the target note afterward (recall, fire, resume) substitutes
        an anti-memory deterrent block for its raw content (see
        `_format_full_block`) until a `reinstated` event is appended
        (`reinstate_note()`) — always legal, revert-of-revert. Raises
        ValueError if the target note does not exist in this workspace. No
        human-provenance write-boundary guard (unlike `supersedes`): a
        revoked note is never silently excluded from any surface — it stays
        visible in deterrent framing, which is the opposite of the silent
        muting `supersedes`' guard defends against, so vectr does not gate
        who may raise the flag; judgment about whether the contradiction is
        valid stays entirely caller-side (a human can always `reinstate_note
        ()` a revocation it disagrees with).
        """
        from agent.trigger_engine import validate_triggers

        now = time.time()
        tags_json = json.dumps(tags or [])
        if kind not in VALID_KINDS:
            kind = DEFAULT_KIND

        # UPG-MEM-PROVENANCE-USER-STATED: the derived class is not declarable.
        # Checked BEFORE the membership test below (it is a member — it is a
        # legal STORED value, just never a legal INPUT one) so the message
        # names the actual rule instead of listing the vocabulary.
        if provenance == USER_STATED_PROVENANCE:
            raise ValueError(
                f"provenance='{USER_STATED_PROVENANCE}' cannot be set directly — it is "
                "derived at write time from a user_quote that is verifiably quoted in "
                "this note's content. Pass user_quote='<the user's own words>' instead."
            )
        if provenance not in PROVENANCE_VALUES:
            raise ValueError(f"provenance must be one of: {', '.join(PROVENANCE_VALUES)}")
        # Resolve the binding BEFORE the auto/directive guard below, so both
        # that guard and every write below see the note's EFFECTIVE class: a
        # standing rule carrying the user's own bound words is exactly the
        # reviewed directive the guard exists to require, not the unreviewed
        # one it exists to reject.
        bound_quote, _ = bind_user_quote(content, user_quote)
        # UPG-PROVENANCE-NEVER-RISES: the harness-driven fallback, tried only
        # when the caller made no explicit (and successful) `user_quote`
        # claim of its own — an explicit claim always wins over the
        # automatic one, same precedence a caller-declared `title` has over
        # the content-derived fallback below.
        if not bound_quote:
            bound_quote, _ = bind_user_quote_auto(content, recent_user_message)
        if bound_quote and provenance != "human":
            provenance = USER_STATED_PROVENANCE
        if provenance == "auto" and kind == "directive":
            raise ValueError(
                "provenance='auto' is not allowed on kind='directive' — an "
                "unreviewed standing rule is a contradiction in terms; use "
                "provenance='agent' (or have a human endorse it) instead"
            )
        if scope is not None and scope not in SCOPE_VALUES:
            raise ValueError(f"scope must be one of: {', '.join(SCOPE_VALUES)}")

        triggers_list = validate_triggers(triggers)
        triggers_json = json.dumps(triggers_list)

        root = Path(workspace)
        # UPG-ANCHOR-UNOBSERVED-BINDING: third element is the tri-state
        # observation verdict for this SESSION's writing of THIS note — True
        # ("this session's own PreToolUse traffic reported this exact
        # path"), False ("this session has an observation ledger and this
        # path is not in it"), or None ("no ledger for this session exists
        # at all — hooks not wired, stdio transport, team-mode bind, or a
        # daemon restart mid-session — an absent ledger is never treated as
        # proof of non-observation"). See `_anchor_observed_at_write()`.
        anchor_pairs = [
            [p, _hash_path_content(root, p), self._anchor_observed_at_write(workspace, session_id, p)]
            for p in (anchors or [])
        ]
        anchors_json = json.dumps(anchor_pairs)

        # UPG-TRIGGER-SCOPE-KIND-DEFAULTS: an omitted scope (None) resolves to
        # this kind's default bundle at WRITE TIME (see the docstring above
        # and KIND_DEFAULT_SCOPES in _types.py); an explicitly passed scope
        # (including explicit "workspace") is never touched here.
        branch_for_default: str | None = None
        if scope is None:
            scope = KIND_DEFAULT_SCOPES.get(kind, DEFAULT_SCOPE)
            if scope == "branch":
                # Silent-death guard: only actually adopt "branch" when a
                # real branch was captured now — a non-git workspace or a
                # detached HEAD must never bake scope="branch" with an empty
                # branch value (see docstring: that would exclude the note
                # from firing on every future branch, forever, invisibly).
                branch_for_default = _current_git_branch(root) or ""
                if not branch_for_default:
                    scope = DEFAULT_SCOPE

        # branch_value is captured whenever the FINAL resolved scope is
        # "branch" — reusing branch_for_default (already computed just above)
        # when the default-resolution path took it, to avoid a second git
        # subprocess call; falling back to a fresh lookup for an explicitly
        # passed scope="branch" (pre-existing documented behaviour, empty
        # string on a non-git workspace, unchanged by this wave).
        if scope == "branch":
            branch_value = branch_for_default if branch_for_default is not None else (_current_git_branch(root) or "")
        else:
            branch_value = ""

        # Derive title fallback from first non-empty content line (80-char cap).
        if not title:
            for line in content.splitlines():
                stripped = line.strip()
                if stripped:
                    title = stripped[:80]
                    break
        # Encrypt content, the (possibly content-derived) title, and any bound
        # user_quote: the default title is the first content line and a bound
        # quote is by construction a substring of content, so a plaintext
        # column for either would leak the very text encryption is meant to
        # protect.
        if self._encryptor:
            stored_content = self._encryptor.encrypt(content)
            stored_title = self._encryptor.encrypt(title)
            stored_user_quote = self._encryptor.encrypt(bound_quote) if bound_quote else ""
        else:
            stored_content = content
            stored_title = title
            stored_user_quote = bound_quote

        with self._conn() as conn:
            # supersedes (TRIGGER-ENGINE): validate the target BEFORE insert so
            # a bad note_id never leaves a half-applied write.
            if supersedes is not None:
                target = conn.execute(
                    "SELECT note_id, provenance FROM notes WHERE workspace = ? AND note_id = ?",
                    (workspace, supersedes),
                ).fetchone()
                if target is None:
                    raise ValueError(f"supersedes references note #{supersedes}, which does not exist in this workspace")
                # Write-boundary guard (companion to the MCP vectr_remember
                # provenance='human' rejection): a write whose OWN provenance
                # is not "human" may never tombstone a note whose provenance
                # IS "human" -- otherwise an agent-authored write could
                # silently supersede (and so permanently stop firing) a
                # genuine human-reviewed directive, without ever minting
                # provenance='human' itself. Surface-agnostic (applies to
                # REST and any future caller identically, not just MCP) and
                # provenance-based, not surface-based -- a human-provenance
                # write may still supersede anything, including another
                # human note.
                target_provenance = target[1]
                if target_provenance == "human" and provenance != "human":
                    raise ValueError(
                        f"supersedes references note #{supersedes}, which is "
                        "provenance='human' -- a write whose own provenance "
                        "is not 'human' may not supersede a human-provenance "
                        "note (only a human-provenance write may)"
                    )

            # contradicts (UPG-MEMORY-STATE-MACHINE §4.2): validate the
            # target BEFORE insert, same half-applied-write guard as
            # supersedes just above. No provenance write-boundary guard here
            # — see the `contradicts` docstring for why.
            if contradicts is not None:
                contradicts_target = conn.execute(
                    "SELECT note_id FROM notes WHERE workspace = ? AND note_id = ?",
                    (workspace, contradicts),
                ).fetchone()
                if contradicts_target is None:
                    raise ValueError(f"contradicts references note #{contradicts}, which does not exist in this workspace")

            # conflict resolution: if another note anchors the same code block, supersede it
            if code_hash:
                conflicting = conn.execute(
                    """SELECT note_id FROM notes
                       WHERE workspace = ? AND code_hash = ?
                       AND valid_until IS NULL""",
                    (workspace, code_hash),
                ).fetchall()
                if conflicting:
                    conn.execute(
                        """UPDATE notes SET valid_until = ?, superseded_by = ?, superseded_at = ?
                           WHERE note_id IN ({})""".format(",".join("?" * len(conflicting))),
                        [now, author_id or "unknown", now] + [r[0] for r in conflicting],
                    )

            cur = conn.execute(
                """
                INSERT INTO notes (workspace, content, tags, priority, kind, created_at,
                                   last_accessed, session_id, decay_score,
                                   author_id, author_trust_score, valid_from,
                                   valid_until, code_hash, title,
                                   triggers, provenance, scope, anchors, supersedes, branch,
                                   user_quote, pinned)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1.0, ?, 1.0, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (workspace, stored_content, tags_json, priority, kind, now, now, session_id,
                 author_id, now, code_hash, stored_title,
                 triggers_json, provenance, scope, anchors_json, supersedes, branch_value,
                 stored_user_quote, 1 if pin else 0),
            )
            note_id = cur.lastrowid

            # UPG-MEMORY-STATE-MACHINE §4.1: every note's log starts with a
            # `created` event — the fold's own "active" default already
            # means this, but the log stays a complete history with no
            # implicit first row (see NOTE_EVENT_KINDS's docstring).
            _append_event(
                conn, workspace, note_id, "created",
                actor=_actor_for_provenance(provenance), ts=now,
            )

            # TRIGGER-ENGINE wave 2b (bm2-design-skeleton.md §2/§5) — write-
            # time symbol->note index + signature-hash anchor for every
            # explicit trigger declaring a 'symbol' key (validate_triggers()
            # above already confirmed each is a non-empty string). fire()
            # uses this table to skip the symbol-graph lookup entirely when a
            # workspace has no symbol-triggered notes at all; check_staleness
            # ()re-hashes against it to raise a visible (never silent)
            # staleness caveat, mirroring path anchors' content hash. A null
            # hash (resolver not attached yet — memory-only daemon or warm-up
            # window) is never a staleness caveat later, since there is
            # nothing recorded to compare against.
            for trig_idx, trig in enumerate(triggers_list):
                symbol_name = trig.get("symbol")
                if not symbol_name:
                    continue
                sig_hash = (
                    self._symbol_resolver.signature_hash(workspace, symbol_name)
                    if self._symbol_resolver is not None else None
                )
                conn.execute(
                    """INSERT INTO symbol_triggers
                       (workspace, note_id, trigger_index, symbol_name, signature_hash)
                       VALUES (?, ?, ?, ?, ?)""",
                    (workspace, note_id, trig_idx, symbol_name, sig_hash),
                )

            # Explicit tombstone (TRIGGER-ENGINE §1): the superseded note is
            # excluded from recall() by default (valid_until IS NULL filter,
            # pre-existing) and never fires again (evaluate_note checks
            # valid_until), but is retained with its full provenance for audit.
            if supersedes is not None:
                conn.execute(
                    """UPDATE notes SET valid_until = ?, superseded_at = ?, superseded_by_note_id = ?
                       WHERE workspace = ? AND note_id = ?""",
                    (now, now, note_id, workspace, supersedes),
                )
                # UPG-MEMORY-STATE-MACHINE §4.1 migration: `supersedes`
                # becomes sugar over the event log — the pre-existing
                # column write above is unchanged (existing behaviour is
                # never touched), this just also appends the generalized
                # event, payload = the superseding (this) note's id.
                _append_event(
                    conn, workspace, supersedes, "superseded",
                    actor=_actor_for_provenance(provenance), payload=str(note_id), ts=now,
                )

            # contradicts (UPG-MEMORY-STATE-MACHINE §4.2): "that note was
            # WRONG" — appends `revoked` to the TARGET note's log, always
            # actor="agent" and this exact reason text (see the
            # `contradicts` docstring for why no other actor/reason is
            # accepted here). Does NOT set valid_until — a revoked note
            # stays a live recall()/fire() candidate so the anti-memory
            # deterrent (`_format_full_block`) can surface in its place.
            if contradicts is not None:
                _append_event(
                    conn, workspace, contradicts, "revoked",
                    actor="agent", reason=f"contradicted by #{note_id}", payload=str(note_id), ts=now,
                )

            # update author trust score registry (Bayesian: count-weighted)
            if author_id:
                conn.execute(
                    """INSERT INTO author_trust (workspace, author_id, trust_score, note_count)
                       VALUES (?, ?, 1.0, 1)
                       ON CONFLICT(workspace, author_id) DO UPDATE SET
                           note_count = note_count + 1,
                           trust_score = MIN(1.0, trust_score + 0.05)""",
                    (workspace, author_id),
                )

        # Embed and store in the vector index so recall(query=...) can use cosine similarity.
        # content is the plaintext (before encryption) — embeddings are over raw text.
        if self._notes_col is not None and self._embed_fn is not None:
            try:
                vec = self._embed_fn([content])[0]
                with timed_chroma_call("upsert"):
                    self._notes_col.upsert(ids=[str(note_id)], embeddings=[vec])
            except Exception:
                pass  # embedding failure never blocks the write path

        audit("REMEMBER", workspace=workspace, note_id=note_id, priority=priority,
              kind=kind, author_id=author_id, code_hash=code_hash[:8] if code_hash else "",
              tags=",".join(tags or []), chars=len(content), provenance=provenance, scope=scope)
        return note_id  # type: ignore[return-value]

    def promote(self, workspace: str, note_id: int, to: str) -> bool:
        """Explicit provenance promotion (bm2-design-skeleton.md §5):
        auto -> agent -> human only, one step at a time. Provenance is
        immutable at write; this is the one sanctioned, explicit way to
        raise it after the fact (e.g. a human reviews and endorses an
        agent-authored note). Demotion is impossible — `to` must be exactly
        one rank above the note's current provenance.

        Returns True if promoted, False if the note does not exist. Raises
        ValueError if `to` is not a valid single-step promotion from the
        note's current provenance.

        UPG-MEM-PROVENANCE-USER-STATED: "user-stated" is not a promotion
        TARGET (it is derived from a bound verbatim excerpt at write time —
        promoting into it would restore by self-attestation the class the
        binding check exists to make machine-checkable), but it is a valid
        SOURCE ranked with "agent" (PROMOTION_RANK), so a user-stated note's
        one legal step stays "human" — a person endorsing a transcription.
        """
        note = self.get_note(workspace, note_id)
        if note is None:
            return False
        order = PROMOTION_LADDER  # ("auto", "agent", "human") — promotion direction
        current_rank = PROMOTION_RANK.get(note.provenance, 0)
        if to not in order:
            raise ValueError(f"'to' must be one of: {', '.join(order)}")
        to_rank = order.index(to)
        if to_rank != current_rank + 1:
            raise ValueError(
                f"promote() only allows a single step up from '{note.provenance}' "
                f"(auto -> agent -> human); '{to}' is not that step"
            )
        with self._conn() as conn:
            conn.execute(
                "UPDATE notes SET provenance = ? WHERE workspace = ? AND note_id = ?",
                (to, workspace, note_id),
            )
            # UPG-MEMORY-STATE-MACHINE §4.1 migration: existing promotion
            # writes a `promoted` event — the pre-existing column write
            # above is unchanged, this is purely additive audit history.
            # payload records the new provenance rank reached.
            _append_event(
                conn, workspace, note_id, "promoted",
                actor=_actor_for_promotion(to), payload=to,
            )
        audit("PROMOTE", workspace=workspace, note_id=note_id, provenance=to)
        return True

    def set_pinned(self, workspace: str, note_id: int, pinned: bool) -> bool:
        """Pin or unpin an existing note (UPG-RECALL-MISS-FLOOR part (b),
        `vectr_pin`): explicit Tier 0 membership, set/cleared entirely by
        caller decision — never inferred from content, usage, or query
        history. As of round 4, a pinned note is the ONLY way into
        `recall()`'s unconditional-injection Tier 0 (bounded by
        `recall_floor.tier0_max_notes`, see agent/config.yaml) — directive
        notes no longer enter this tier; they have their own unconditional,
        uncapped-per-query delivery via `boot_recall()` instead. Idempotent:
        pinning an already-pinned note (or unpinning an already-unpinned
        one) is a no-op write that still returns True. Returns False if the
        note does not exist in this workspace."""
        note = self.get_note(workspace, note_id)
        if note is None:
            return False
        with self._conn() as conn:
            conn.execute(
                "UPDATE notes SET pinned = ? WHERE workspace = ? AND note_id = ?",
                (1 if pinned else 0, workspace, note_id),
            )
        audit("PIN", workspace=workspace, note_id=note_id, pinned=pinned)
        return True

    def attach_anchors(
        self,
        workspace: str,
        note_id: int,
        anchors: list[str],
        session_id: str | None = None,
    ) -> dict[str, list[str]] | None:
        """Attach anchor paths to an EXISTING note (UPG-ANCHOR-ATTACH,
        `vectr_anchor`): first-class post-write anchoring, replacing the
        re-store-with-supersedes workaround. Like set_pinned()/
        update_note_fields(), anchors are mutable delivery-shaping metadata,
        so this is a plain UPDATE plus one audit() call — deliberately NOT a
        note_events lifecycle event: NOTE_EVENT_KINDS covers the note's
        content lifecycle, and staleness is re-derived from live anchor
        hashes on every check_staleness() anyway, so a newly attached anchor
        participates in the next check with no event required.

        Pair construction mirrors remember(): [path, content_hash_or_None,
        observed_tri_state] — hash computed NOW via _hash_path_content()
        (it becomes the baseline the next check_staleness() compares
        against), observation verdict via _anchor_observed_at_write().

        Idempotent: a requested path already present on the note (exact
        string match on element 0, same value remember() stored verbatim)
        is skipped and reported in `already_present`, not duplicated.
        A fully-idempotent request (nothing new) writes nothing and audits
        nothing.

        Returns {"attached": [...], "already_present": [...]} — attached
        lists paths actually added, in request order — or None if the note
        does not exist in this workspace. Raises ValueError when `anchors`
        is empty, mirroring update_note_fields()'s
        require-at-least-one-field contract."""
        if not anchors:
            raise ValueError("attach_anchors requires at least one anchor path")
        note = self.get_note(workspace, note_id)
        if note is None:
            return None
        root = Path(workspace)
        existing_paths = {a[0] for a in (note.anchors or [])}
        attached: list[str] = []
        already_present: list[str] = []
        new_pairs: list[list] = []
        for p in anchors:
            if p in existing_paths:
                already_present.append(p)
                continue
            # Third element is the UPG-ANCHOR-UNOBSERVED-BINDING tri-state —
            # same semantics as in remember().
            new_pairs.append(
                [p, _hash_path_content(root, p), self._anchor_observed_at_write(workspace, session_id, p)]
            )
            existing_paths.add(p)
            attached.append(p)
        if new_pairs:
            merged = [list(a) for a in (note.anchors or [])] + new_pairs
            with self._conn() as conn:
                conn.execute(
                    "UPDATE notes SET anchors = ? WHERE workspace = ? AND note_id = ?",
                    (json.dumps(merged), workspace, note_id),
                )
            audit("ANCHOR_ATTACH", workspace=workspace, note_id=note_id, attached=attached)
        return {"attached": attached, "already_present": already_present}

    def detach_anchors(
        self,
        workspace: str,
        note_id: int,
        anchors: list[str],
    ) -> dict[str, list[str]] | None:
        """Remove anchor paths from an EXISTING note (UPG-ANCHOR-DETACH,
        `vectr_unanchor`): the inverse of `attach_anchors()`. Like attach,
        anchors are delivery-shaping metadata, so this is a plain UPDATE
        plus one `audit()` call — deliberately NOT a note_events lifecycle
        event: NOTE_EVENT_KINDS covers the note's content lifecycle, and
        staleness is re-derived from live anchor hashes on every
        check_staleness() anyway, so a removed anchor drops out of the
        next check with no event required.

        Path comparison mirrors `attach_anchors()` exactly: exact string
        equality on element 0 of the pair (the same value `remember()`
        stored verbatim). A note anchored via one spelling of a path is
        detachable by that same spelling only — no normalisation, by
        design, so a deliberate spelling difference stays a deliberate
        spelling difference.

        Idempotent: a requested path not present on the note is reported
        in `not_present` (the attach-mirror's name `already_present`
        would be a lie — the path is not "already" removed, it was never
        there). A fully-idempotent request (nothing to remove) writes
        nothing and audits nothing.

        Returns {"removed": [...], "not_present": [...]} — removed lists
        paths actually dropped, in request order — or None if the note
        does not exist in this workspace. Raises ValueError when
        `anchors` is empty, mirroring `attach_anchors()`'s
        require-at-least-one-field contract.

        When the last anchor is removed, the column is written as `[]` —
        the same default the schema migration writes for notes that have
        never had an anchor (TEXT NOT NULL DEFAULT '[]'), so a detached-
        to-zero note is byte-identical to a never-anchored note on
        reload. No NULL is ever written.
        """
        if not anchors:
            raise ValueError("detach_anchors requires at least one anchor path")
        note = self.get_note(workspace, note_id)
        if note is None:
            return None
        existing_pairs = [list(a) for a in (note.anchors or [])]
        existing_paths = {a[0]: a for a in existing_pairs}
        removed: list[str] = []
        not_present: list[str] = []
        keep_pairs: list[list] = []
        # Walk the REQUEST list (not the existing pairs) so a duplicate
        # entry in the request (`['x.py', 'x.py']`) reports as
        # one-removed + one-dup, mirroring attach_anchors()'s
        # one-attached + one-already_present. Per-anchor dedup is
        # responsibility of the caller; the contract is that we report
        # what each entry of the request resolved to.
        requested_seen: set[str] = set()
        for p in anchors:
            if p in requested_seen:
                # Duplicate of an earlier request entry — exact
                # symmetric of attach_anchors()'s `if p in existing_paths`
                # branch: the path was already removed by the first
                # occurrence, so report it as a no-op against the now-
                # empty post-removal state.
                not_present.append(p)
                continue
            requested_seen.add(p)
            if p in existing_paths:
                # Drop the existing pair, record the removal.
                keep_pairs = [a for a in existing_pairs if a[0] != p]
                existing_pairs = keep_pairs
                existing_paths.pop(p, None)
                removed.append(p)
            else:
                not_present.append(p)
        if removed:
            with self._conn() as conn:
                conn.execute(
                    "UPDATE notes SET anchors = ? WHERE workspace = ? AND note_id = ?",
                    (json.dumps(keep_pairs), workspace, note_id),
                )
            audit("ANCHOR_DETACH", workspace=workspace, note_id=note_id, removed=removed)
        return {"removed": removed, "not_present": not_present}

    def revoke_note(
        self,
        workspace: str,
        note_id: int,
        reason: str,
        actor: str = "agent",
    ) -> bool:
        """Explicit revocation (UPG-MEMORY-STATE-MACHINE §4.2,
        `vectr_revoke`): "this note was WRONG", proposed BY THE CALLER — the
        one dedicated tool for a revocation with a custom reason/actor,
        distinct from `remember(contradicts=...)`'s fixed
        actor='agent'/auto-derived-reason shorthand for "the note I'm
        writing right now contradicts an older one".

        Appends a `revoked` event; does NOT set `valid_until` — the note
        stays a live recall()/fire() candidate so every rendering surface
        substitutes the anti-memory deterrent block for its raw content
        (see `_format_full_block`) instead of silently excluding it.
        Reversible at any time via `reinstate_note()` — always legal,
        revert-of-revert, one more event, never a rewrite of this one.

        `actor` must be one of NOTE_EVENT_ACTORS other than 'system'
        (reserved for the one deterministic, non-judgment transition,
        `stale_flagged` — a human/agent making an explicit revoke call is
        always a judgment call, never automatic). Returns True if the note
        exists and was revoked, False if it does not exist in this
        workspace. Raises ValueError if `actor` is invalid.
        """
        if actor not in NOTE_EVENT_ACTORS or actor == "system":
            raise ValueError(
                f"actor must be one of: {', '.join(a for a in NOTE_EVENT_ACTORS if a != 'system')}"
            )
        note = self.get_note(workspace, note_id)
        if note is None:
            return False
        with self._conn() as conn:
            _append_event(conn, workspace, note_id, "revoked", actor=actor, reason=reason)
        audit("REVOKE", workspace=workspace, note_id=note_id, actor=actor)
        return True

    def reinstate_note(
        self,
        workspace: str,
        note_id: int,
        actor: str = "agent",
        reason: str | None = None,
    ) -> bool:
        """Revert a revocation OR an expiry OR a supersession
        (UPG-MEMORY-STATE-MACHINE §4.2, UPG-MEMORY-DECAY-KIND-SCOPED,
        UPG-NOTE-RETIRE-POST-HOC; `vectr_reinstate`) — appends a
        `reinstated` event, always legal regardless of the note's current
        folded state (a no-op-in-spirit reinstate on an already-active note
        is harmless, same "just append, let the fold decide" philosophy as
        a repeat revoke). The SAME event kind reverses every terminal state
        — `reinstated` means "return to active" regardless of whether the
        note got there via `revoke_note()`, `purge_expired_notes()`'s
        automatic TTL expiry, or a supersession (`supersedes=` at write
        time, `supersede_note()` post-hoc), so no separate un-expire /
        un-supersede call is needed. Returns True if the note exists, False
        if it does not exist in this workspace. Raises ValueError if
        `actor` is invalid (same restriction as `revoke_note` —
        reinstatement is always a judgment call, never `actor='system'`).

        Supersession additionally tombstones COLUMNS on the note row
        (`valid_until`/`superseded_at`/`superseded_by_note_id` — recall()'s
        default exclusion and evaluate_note's never-fire check both read
        those columns, not the fold), so reversing it also clears them here
        when the CURRENT folded state is 'superseded'. Revocation and
        expiry are event-only transitions and need no column reversal; an
        active-state note gets no column write at all, so reinstating an
        already-active note stays a pure no-op exactly as before."""
        if actor not in NOTE_EVENT_ACTORS or actor == "system":
            raise ValueError(
                f"actor must be one of: {', '.join(a for a in NOTE_EVENT_ACTORS if a != 'system')}"
            )
        note = self.get_note(workspace, note_id)
        if note is None:
            return False
        was_superseded = (
            self.note_event_states(workspace, [note]).get(note_id, {}).get("state")
            == "superseded"
        )
        with self._conn() as conn:
            if was_superseded:
                # UPG-NOTE-RETIRE-POST-HOC: clear the tombstone columns in
                # the SAME transaction as the event, so a caller can never
                # observe "folded back to active" while the row is still
                # excluded from default recall()/fire().
                conn.execute(
                    """UPDATE notes SET valid_until = NULL, superseded_at = NULL,
                       superseded_by_note_id = NULL
                       WHERE workspace = ? AND note_id = ?""",
                    (workspace, note_id),
                )
            _append_event(conn, workspace, note_id, "reinstated", actor=actor, reason=reason)
        audit("REINSTATE", workspace=workspace, note_id=note_id, actor=actor)
        return True

    def supersede_note(
        self,
        workspace: str,
        note_id: int,
        superseded_by: int | None = None,
        reason: str | None = None,
        actor: str = "agent",
    ) -> bool:
        """Post-hoc supersession (UPG-NOTE-RETIRE-POST-HOC, `vectr_supersede`)
        — retire an ALREADY-STORED note as superseded, for the case the
        write-time `supersedes=` parameter cannot reach: the replacement was
        written without it, and the old note only later turned out to be
        already-superseded-in-substance. Appends a `superseded_post_hoc`
        event — the SAME lifecycle state as write-time `superseded`, but a
        distinct KIND, so the trail honestly records this link was asserted
        after the fact by a caller judgment rather than born of a write.

        Sets the exact columns the write-time path sets (`valid_until`,
        `superseded_at`, `superseded_by_note_id`) so exclusion behaves
        identically: dropped from default recall()/fire(), retained with its
        history, rendered with the factual "[superseded ...]" badge — never
        the revoked deterrent ("this note was WRONG" is `vectr_revoke`'s
        semantics; a retired note was true when written and the world moved
        on). Reversible via `reinstate_note()`.

        `superseded_by`: optional note_id of the replacement, which must
        already exist in this workspace (validated BEFORE any mutation —
        same half-applied-write guard as remember()'s supersedes path).
        Omitted, the note retires with no successor link and renders
        "[superseded, <date>]".

        `reason`: free text recorded on the event (the caller's evidence
        that the old note's substance lives on elsewhere).

        `actor` must be one of NOTE_EVENT_ACTORS other than 'system'
        (reserved for the deterministic anchor-drift transition) — deciding
        a stored note is already-superseded is a judgment call. Same
        provenance write-boundary guard as remember()'s supersedes path,
        keyed on the CALLER instead of a new write's provenance (there is
        no new write here): an agent-actor call may not retire a
        human-provenance note; only actor='human' may.

        Returns True if the note exists and was retired, False if it does
        not exist in this workspace. Raises ValueError on an invalid actor,
        a `superseded_by` equal to `note_id` itself, or a `superseded_by`
        that does not exist in this workspace.
        """
        if actor not in NOTE_EVENT_ACTORS or actor == "system":
            raise ValueError(
                f"actor must be one of: {', '.join(a for a in NOTE_EVENT_ACTORS if a != 'system')}"
            )
        note = self.get_note(workspace, note_id)
        if note is None:
            return False
        if superseded_by == note_id:
            raise ValueError("superseded_by must reference a different note than the one being retired")
        now = time.time()
        with self._conn() as conn:
            # Validate the successor BEFORE mutating anything, mirroring
            # remember()'s supersedes pre-insert validation.
            if superseded_by is not None:
                target = conn.execute(
                    "SELECT note_id FROM notes WHERE workspace = ? AND note_id = ?",
                    (workspace, superseded_by),
                ).fetchone()
                if target is None:
                    raise ValueError(
                        f"superseded_by references note #{superseded_by}, "
                        "which does not exist in this workspace"
                    )
            # Provenance write-boundary guard (companion to remember()'s
            # supersedes guard): an agent-actor call may not retire a
            # human-provenance note — only a human-actor call may.
            if note.provenance == "human" and actor != "human":
                raise ValueError(
                    f"note #{note_id} is provenance='human' -- an agent-actor "
                    "call may not supersede a human-provenance note (only a "
                    "human-actor call may)"
                )
            conn.execute(
                """UPDATE notes SET valid_until = ?, superseded_at = ?, superseded_by_note_id = ?
                   WHERE workspace = ? AND note_id = ?""",
                (now, now, superseded_by, workspace, note_id),
            )
            _append_event(
                conn, workspace, note_id, "superseded_post_hoc",
                actor=actor, reason=reason,
                payload=str(superseded_by) if superseded_by is not None else None,
                ts=now,
            )
        audit("SUPERSEDE", workspace=workspace, note_id=note_id,
              superseded_by=superseded_by, actor=actor)
        return True

    def note_event_states(
        self, workspace: str, notes: list[WorkingNote],
    ) -> dict[int, dict]:
        """Batched fold of `note_events` for the given notes
        (UPG-MEMORY-STATE-MACHINE §4.1) — one query, one fold pass per note,
        for every surface that renders notes (`format_notes_for_llm()`,
        `fire_and_format()`, `format_resume()`) to look up folded state
        without a per-note round trip.

        Returns {note_id: {"state": "active"|"superseded"|"revoked",
        "reason": str|None, "actor": str|None, "ts": float|None}} — see
        `_events.fold()` for exactly what reason/actor/ts mean. A note_id
        with NO rows in `note_events` at all (pre-migration notes, or an
        empty `notes` list) is simply absent from the returned dict; every
        caller here treats a missing key as "active" (the same state an
        intact created-only log would fold to), so old notes render exactly
        as before this feature shipped.
        """
        if not notes:
            return {}
        return self._note_event_states_by_ids(workspace, [n.note_id for n in notes])

    def _note_event_states_by_ids(self, workspace: str, note_ids: list[int]) -> dict[int, dict]:
        """Batched fold of `note_events` for a list of note IDs directly
        (UPG-MEMORY-DECAY-KIND-SCOPED) — the same query/fold `note_event_
        states()` performs, factored out so a caller with only IDs in hand
        (no `WorkingNote` objects yet — `purge_expired_notes()`'s
        idempotency check, `_exclude_expired()`'s post-fetch filter) doesn't
        need to construct them first. `note_event_states()` is now a thin
        wrapper over this for its existing `list[WorkingNote]` callers.
        """
        if not note_ids:
            return {}
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT note_id, event, actor, reason, payload, ts, id FROM note_events "
                "WHERE workspace = ? AND note_id IN ({}) ORDER BY note_id, id ASC".format(
                    ",".join("?" * len(note_ids))
                ),
                [workspace] + list(note_ids),
            ).fetchall()
        by_note: dict[int, list[dict]] = {}
        for r in rows:
            by_note.setdefault(r["note_id"], []).append(dict(r))
        return {nid: _fold_note_events(evs) for nid, evs in by_note.items()}

    def _exclude_expired(self, workspace: str, notes: list[WorkingNote]) -> list[WorkingNote]:
        """Drop notes whose current folded lifecycle state (UPG-MEMORY-
        STATE-MACHINE §4.1) is 'expired' from a candidate pool (UPG-MEMORY-
        DECAY-KIND-SCOPED) — the "stops being injected" half of expiry's
        contract. The row itself, its note_events log, and its deterrent
        rendering all survive untouched (`get_note()` and
        `format_notes_for_llm()` never call this — only the default/
        proactive listing surfaces below do). Applied as a post-fetch
        filter, the same "may return fewer than `limit` results" trade-off
        `_scope_filter()` already accepts on every one of these same call
        sites."""
        if not notes:
            return notes
        states = self._note_event_states_by_ids(workspace, [n.note_id for n in notes])
        return [n for n in notes if states.get(n.note_id, {}).get("state") != "expired"]

    def _floor_candidate_pool(
        self,
        workspace: str,
        tags: list[str] | None,
        priority: str | None,
        max_age_days: float | None,
        session_id: str | None,
    ) -> list[WorkingNote]:
        """Every non-superseded, non-expired, scope-eligible note in
        `workspace` (respecting the same tags/priority/max_age_days filters
        an ordinary `recall()` call would apply), most-recent-first,
        capped at `recall_floor.metadata_scan_ceiling` (UPG-RECALL-MISS-
        FLOOR parts (b)/(c)). This is the candidate pool Tier 0 membership
        and the deterministic tag/anchor/symbol/title channels are checked
        against — unlike the ranked result, there is no relevance-ranked
        shortcut for "does this note's own metadata mention the query", so
        the whole eligible set (bounded by the ceiling) has to be scanned."""
        from agent.config import RECALL_FLOOR_METADATA_SCAN_CEILING

        sql = "SELECT * FROM notes WHERE workspace = ? AND valid_until IS NULL"
        params: list = [workspace]
        if priority:
            sql += " AND priority = ?"
            params.append(priority)
        if tags:
            tag_clauses = " OR ".join(["tags LIKE ?" for _ in tags])
            sql += f" AND ({tag_clauses})"
            params.extend([f'%"{t}"%' for t in tags])
        if max_age_days is not None:
            cutoff = time.time() - max_age_days * 86400
            sql += " AND created_at >= ?"
            params.append(cutoff)
        sql += " ORDER BY created_at DESC, note_id DESC LIMIT ?"
        params.append(RECALL_FLOOR_METADATA_SCAN_CEILING)

        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        notes = [self._row_to_note(r) for r in rows]
        notes = _scope_filter(notes, session_id=session_id)
        notes = self._exclude_expired(workspace, notes)
        return notes

    def _recall_floor_notes(
        self,
        workspace: str,
        query: str | None,
        tags: list[str] | None,
        priority: str | None,
        max_age_days: float | None,
        session_id: str | None,
        limit: int,
    ) -> tuple[list[WorkingNote], list[WorkingNote]]:
        """Tier 0 (pinned notes, unconditional) and deterministic
        tag/anchor/symbol/title channel hits for `query` (UPG-RECALL-MISS-
        FLOOR parts (b)/(c), revised round 4). Returns (tier0,
        deterministic) — `recall()` merges both ahead of the ranked
        result via `_merge_recall_floor()`. Tier 0 needs no `query` at
        all (a pin is unconditional by definition); the deterministic
        channel returns [] when `query` is empty since there is nothing
        to check containment against, and also returns [] whenever
        `RECALL_FLOOR_DETERMINISTIC_ENABLED` is false (see below).

        Tier 0 is PINNED-ONLY as of round 4 (it used to also include
        every `kind == 'directive'` note). Directive notes were dropped
        from its candidate predicate because `boot_recall()` already
        returns ALL directive notes unconditionally every session
        (capped by config.BOOT_MAX_DIRECTIVE_NOTES, default 100) —
        including them here a second time only charged query budget for
        a guarantee that already exists on a stronger, dedicated
        channel. The pin is the only note kind with NO other guaranteed-
        delivery path, so it is the only kind Tier 0 still carries; the
        pin-vs-directive priority sort this method used to need is gone
        along with the directive branch — pinned is now the sole
        candidate kind, so there is nothing left to choose between.

        Tier 0's cap (UPG-RECALL-MISS-FLOOR F1 fix, revised per review) is
        `min(tier0_max_notes, max(tier0_guaranteed, floor(limit *
        tier0_max_share_of_limit)))`, where `tier0_guaranteed = max(1,
        min(tier0_min_notes, floor(limit * tier0_max_share_of_limit)))` —
        never 0 for any `limit >= 1` (F1's requirement), but scaled by the
        same share as the ceiling rather than flat-taxing a tight budget
        at the full `tier0_min_notes` count. At the production default
        (limit=10) this yields 2 (room for more than one pinned note to
        survive at once); at limit=3 or limit=5 it yields 1 (a single
        guaranteed slot — the most recently pinned note). "A pinned note
        is returned regardless of query" holds at every limit >= 1; room
        for a SECOND simultaneous pin only opens up at limit>=10; see
        agent/config.yaml's recall_floor comments for the exact crossover
        reasoning.

        The deterministic channel is OFF BY DEFAULT
        (`RECALL_FLOOR_DETERMINISTIC_ENABLED`, config key
        `recall_floor.deterministic_enabled`). Measured against the live
        38-query labeled corpus, its declared-metadata containment
        predicate is too loose to act as a guarantee (mean ~50 candidate
        notes matched per query against ~1 labeled-relevant note; a
        floor-cap sweep showed its effect on measured recall is NOT
        monotonic in budget — see the config.yaml comment on
        `deterministic_enabled` for the full numbers). This is evidence-
        gating, not deletion: the code below, its cap formula (F3 fix),
        and its match-specificity ordering (F2 fix) are unchanged and
        still fully tested — a future task can tighten the predicate
        (e.g. require 2+ matched channels for pool membership, not just
        to win a tie) and flip the flag back on with fresh evidence.

        The deterministic channel's cap (UPG-RECALL-MISS-FLOOR F3 fix),
        when enabled, is `min(deterministic_max_notes, floor((limit -
        len(tier0)) * deterministic_max_share_of_remaining))` — bounded
        by a share of what Tier 0 did NOT use, not a share of `limit`
        directly, so it is the direct statement of "the deterministic
        channel never takes more than the ranked group": at most an
        equal split of whatever Tier 0 left behind, guaranteeing ranked
        fill at least as many slots as the deterministic channel claims.

        When more notes match the deterministic channel than fit its cap,
        the matched set is ranked by its own match specificity (UPG-
        RECALL-MISS-FLOOR F2 fix — `_declared_metadata_match_specificity()`
        above) before truncating, so a note matching MORE of its own
        declared fields, or matching them with a longer/more-specific
        string, wins a scarce slot over one that merely happens to be the
        most recently written among dozens of equally-valid matches.
        """
        from agent.config import (
            RECALL_FLOOR_DETERMINISTIC_ENABLED,
            RECALL_FLOOR_DETERMINISTIC_MAX_NOTES,
            RECALL_FLOOR_DETERMINISTIC_MAX_SHARE_OF_REMAINING,
            RECALL_FLOOR_TIER0_MAX_NOTES,
            RECALL_FLOOR_TIER0_MAX_SHARE_OF_LIMIT,
            RECALL_FLOOR_TIER0_MIN_NOTES,
        )

        pool = self._floor_candidate_pool(workspace, tags, priority, max_age_days, session_id)

        tier0_share_cap = int(limit * RECALL_FLOOR_TIER0_MAX_SHARE_OF_LIMIT)
        # max(1, ...) is F1's guarantee: never truncate to 0 for any
        # limit >= 1. min(TIER0_MIN_NOTES, share) rather than flat
        # TIER0_MIN_NOTES: at a tight `limit` the guarantee scales down
        # to a single slot instead of unconditionally spending
        # TIER0_MIN_NOTES worth of budget.
        tier0_guaranteed = max(1, min(RECALL_FLOOR_TIER0_MIN_NOTES, tier0_share_cap))
        tier0_cap = min(RECALL_FLOOR_TIER0_MAX_NOTES, max(tier0_guaranteed, tier0_share_cap))
        # `pool` is ordered created_at DESC, note_id DESC (see
        # _floor_candidate_pool) — slicing to `tier0_cap` here means Tier 0
        # is, in practice, "the tier0_cap MOST RECENTLY PINNED notes" in
        # the workspace. A workspace holding more pinned notes than the
        # cap will never surface an older one via this tier; that is a
        # deliberate, stated bound of the design (recency is a reasonable
        # deterministic tie-break for an otherwise-unconditional
        # guarantee), not an accident of pool order. Pinned-only as of
        # round 4 (see the docstring above for why directives were
        # dropped) — no ordering is needed among candidates beyond the
        # pool's own recency order, since there is only one candidate
        # kind left.
        tier0_candidates = [n for n in pool if n.pinned]
        tier0 = tier0_candidates[:tier0_cap]
        tier0_ids = {n.note_id for n in tier0}

        deterministic: list[WorkingNote] = []
        if not RECALL_FLOOR_DETERMINISTIC_ENABLED:
            return tier0, deterministic

        remaining_after_tier0 = max(limit - len(tier0), 0)
        det_cap = min(
            RECALL_FLOOR_DETERMINISTIC_MAX_NOTES,
            int(remaining_after_tier0 * RECALL_FLOOR_DETERMINISTIC_MAX_SHARE_OF_REMAINING),
        )
        if query and det_cap > 0:
            scored: list[tuple[tuple[int, int], WorkingNote]] = []
            for n in pool:
                if n.note_id in tier0_ids:
                    continue
                specificity = _declared_metadata_match_specificity(n, query)
                if specificity[0] > 0:
                    scored.append((specificity, n))
            # Order the already-matched set by its own match specificity,
            # strongest first; Python's sort is stable, so notes tied on
            # specificity keep the pool's created_at DESC order as the
            # final, deterministic tie-break. This orders WHICH matches
            # win a scarce slot — it never decides membership, which
            # already happened above via containment against each note's
            # own declared metadata (UPG-RECALL-MISS-FLOOR F2 fix).
            scored.sort(key=lambda pair: pair[0], reverse=True)
            deterministic = [n for _, n in scored[:det_cap]]
        return tier0, deterministic

    def recall(
        self,
        workspace: str,
        query: str | None = None,
        tags: list[str] | None = None,
        priority: str | None = None,
        limit: int = 10,
        include_superseded: bool = False,
        kind: str | None = None,
        min_similarity: float | None = None,
        max_age_days: float | None = None,
        sort_by: str = "relevance",
        session_id: str | None = None,
        apply_floor: bool = False,
    ) -> list[WorkingNote]:
        """Retrieve working notes.

        When query is provided and semantic search is available (embed_fn + ChromaDB
        collection configured), uses cosine similarity to rank notes by relevance.
        Falls back to SQL LIKE substring match when semantic search is unavailable.

        Superseded notes are excluded by default. Pass include_superseded=True
        to see the full history including notes marked as superseded.
        Without a query, results are ordered by author_trust_score DESC, decay_score DESC,
        created_at DESC, note_id DESC so the highest-trust contributor's notes surface
        first, with a fully deterministic tie-break (UPG-RECALL-ORDER-CHURN: last_accessed
        is intentionally excluded from this ordering — recall itself updates
        last_accessed on every note it returns, so using it as a sort key made
        two back-to-back identical calls read-your-own-writes into a different
        order each time). kind='task' notes are exempted from the trust/decay
        part of that ordering (UPG-TASK-NOTE-INJECTION-RECENCY): a task note is
        current-work state, not a relevance-ranked learning, so an older task
        note with a higher author_trust_score/decay_score must never outrank a
        newer one — for kind='task' rows the trust/decay columns are treated
        as equal and created_at DESC (then note_id DESC) decides the order
        directly. Every other kind is unaffected.

        max_age_days: when set, only notes created within the last max_age_days days
        are returned (UPG-RECALL-HIERARCHY time filter).

        sort_by: one of 'relevance' (semantic/default SQL order), 'recency'
        (created_at DESC), 'priority' (high>medium>low, then created_at DESC),
        or 'chronological' (created_at ASC — oldest first; UPG-DECISION-
        TIMELINE). 'chronological' composes with any `kind`/`tags`/`priority`
        filter the same way 'recency'/'priority' already do — it is not
        specific to kind='decision', just the lever that turns a filtered
        recall into a time-ordered listing (e.g. an ADR-style decision
        timeline via kind='decision'). In the semantic path, recency/
        priority/chronological are applied as a re-sort after candidate fetch
        (relevance = semantic order is unchanged).

        session_id: enforces scope="session" notes (TRIGGER-ENGINE wave 2a,
        §1) — a note scoped to a different session (or to none, when
        session_id is omitted here) is excluded from the result. Applied as
        a post-filter after the SQL LIMIT, so it may return fewer than
        `limit` results — the same trade-off min_similarity already accepts.

        apply_floor (UPG-RECALL-MISS-FLOOR parts (b)/(c), revised round
        4): compose the ordinary ranked result (semantic or SQL, above/
        below unchanged) with two never-scored guarantees, in this fixed
        precedence — Tier 0 (every explicitly PINNED note, unconditional;
        directive notes were dropped from this tier in round 4 — see
        `_recall_floor_notes()`'s docstring for why: they already have
        their own unconditional, uncapped-per-query channel in
        `boot_recall()`) first, then the deterministic tag/anchor/symbol/
        title channels (`_note_matches_declared_metadata()` — matched
        against each candidate note's OWN stored metadata, never a
        classification of `query`, ranked among themselves by match
        specificity when more match than fit — UPG-RECALL-MISS-FLOOR F2
        fix; OFF BY DEFAULT as of round 4, see `recall_floor.
        deterministic_enabled` in agent/config.yaml for the measured
        evidence gate), then the ranked result fills whatever budget
        remains. Tier 0 is bounded by a guaranteed MINIMUM (recall_floor.
        tier0_min_notes — room for more than one simultaneous pin
        whenever `limit` leaves enough room) with an absolute ceiling
        (tier0_max_notes) and a share-of-`limit` ceiling
        (tier0_max_share_of_limit) on top of that minimum; the
        deterministic channel, when enabled, is bounded by its own
        absolute ceiling (deterministic_max_notes) and a share of
        whatever Tier 0 did NOT use (deterministic_max_share_of_remaining
        — never more than the ranked group gets from what's left). See
        `_recall_floor_notes()` for the exact formulas. Together these
        mean the two floor channels combined can never claim the whole
        budget and displace ranked fill entirely, at any `limit` a caller
        passes, while Tier 0's own guarantee cannot silently shrink to
        zero at a small `limit` either. With zero pinned notes in a
        workspace and the deterministic channel at its shipped default of
        off, apply_floor=True and apply_floor=False produce byte-
        identical output — nothing is consumed unconditionally unless a
        caller actually opted in via a pin. The combined, de-duplicated
        result is always truncated to `limit` — this method never returns
        more than `limit` notes regardless of how many candidates the
        floor and the ranked result each contribute.
        Ignored (no floor notes computed) when `kind` narrows the query to
        one specific kind — a caller that explicitly asked for one kind
        gets exactly that kind, unmodified, same as apply_floor=False.
        Defaults False, preserving this method's exact pre-existing
        behavior for every caller that predates this parameter (including
        every other call site in this codebase); `VectrService._recall_impl
        ()`'s ordinary query branch — the code path `vectr_recall`/`vectr
        recall` actually hits for a plain query — passes True.
        """
        floor_tier0: list[WorkingNote] = []
        floor_det: list[WorkingNote] = []
        if apply_floor and kind is None:
            floor_tier0, floor_det = self._recall_floor_notes(
                workspace, query, tags, priority, max_age_days, session_id, limit,
            )

        # Semantic path: embed the query, find cosine-nearest notes, then fetch from SQLite.
        if query and self._notes_col is not None and self._embed_fn is not None:
            try:
                notes = self._semantic_recall(
                    workspace, query, tags, priority, limit, include_superseded, kind,
                    min_similarity, max_age_days, sort_by, session_id=session_id,
                )
                audit("RECALL", workspace=workspace, query=query, notes_returned=len(notes),
                      method="semantic")
                return _merge_recall_floor(floor_tier0, floor_det, notes, limit)
            except Exception:
                pass  # fall through to SQL LIKE

        # SQL path: used when no query, or when semantic search is unavailable/errored.
        sql = "SELECT * FROM notes WHERE workspace = ?"
        params: list = [workspace]

        if not include_superseded:
            sql += " AND valid_until IS NULL"

        if priority:
            sql += " AND priority = ?"
            params.append(priority)

        if kind:
            sql += " AND kind = ?"
            params.append(kind)

        if tags:
            tag_clauses = " OR ".join(["tags LIKE ?" for _ in tags])
            sql += f" AND ({tag_clauses})"
            params.extend([f'%"{t}"%' for t in tags])

        if query:
            sql += " AND content LIKE ?"
            params.append(f"%{query}%")

        if max_age_days is not None:
            cutoff = time.time() - max_age_days * 86400
            sql += " AND created_at >= ?"
            params.append(cutoff)

        # sort_by applies to the SQL path's ORDER BY clause.
        if sort_by == "recency":
            sql += " ORDER BY created_at DESC LIMIT ?"
        elif sort_by == "priority":
            sql += (
                " ORDER BY CASE priority WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,"
                " created_at DESC LIMIT ?"
            )
        elif sort_by == "chronological":
            # Oldest-first (UPG-DECISION-TIMELINE) — the inverse of 'recency',
            # so a filtered recall (e.g. kind='decision') reads as a timeline
            # rather than a most-recent-first feed. note_id ASC breaks a
            # created_at tie deterministically (two notes written in the same
            # wall-clock second still get a stable relative order).
            sql += " ORDER BY created_at ASC, note_id ASC LIMIT ?"
        else:
            # relevance (default): trust/decay ordering, then a deterministic
            # tie-break (UPG-RECALL-ORDER-CHURN — last_accessed excluded, see
            # the recall() docstring above). kind='task' notes are current-work
            # state, not relevance-ranked learnings — author_trust_score/
            # decay_score are neutralised to a constant for them so recency
            # dominates regardless of trust (UPG-TASK-NOTE-INJECTION-RECENCY);
            # every other kind keeps the unmodified trust/decay ordering.
            sql += (
                " ORDER BY"
                " (CASE WHEN kind = 'task' THEN 1.0 ELSE author_trust_score END) DESC,"
                " (CASE WHEN kind = 'task' THEN 1.0 ELSE decay_score END) DESC,"
                " created_at DESC, note_id DESC LIMIT ?"
            )
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()

        notes = [self._row_to_note(r) for r in rows]
        notes = _scope_filter(notes, session_id=session_id)
        # UPG-MEMORY-DECAY-KIND-SCOPED: an expired note stops being a default
        # recall() candidate (its row/log/deterrent rendering survive via
        # get_note()/format_notes_for_llm() — see _exclude_expired()).
        notes = self._exclude_expired(workspace, notes)

        if notes:
            ids = [n.note_id for n in notes]
            with self._conn() as conn:
                conn.execute(
                    f"UPDATE notes SET last_accessed = ? WHERE note_id IN ({','.join('?' * len(ids))})",
                    [time.time(), *ids],
                )

        audit("RECALL", workspace=workspace, query=query or "", notes_returned=len(notes),
              method="sql")
        return _merge_recall_floor(floor_tier0, floor_det, notes, limit)

    def recall_scored(
        self,
        workspace: str,
        query: str | None = None,
        tags: list[str] | None = None,
        priority: str | None = None,
        limit: int = 10,
        include_superseded: bool = False,
        kind: str | None = None,
        min_similarity: float | None = None,
        max_age_days: float | None = None,
        sort_by: str = "relevance",
        session_id: str | None = None,
    ) -> list[tuple[WorkingNote, float | None]]:
        """Like recall(), but each note is paired with its cosine similarity
        (UPG-PRO-1) so a gating layer can budget/threshold on it.

        Semantic path returns real similarities (1 - cosine distance). The SQL
        LIKE fallback has no cosine to report and returns None per note — never
        a fabricated number. Ordering matches the scoreless recall() exactly.
        """
        if query and self._notes_col is not None and self._embed_fn is not None:
            try:
                return self._semantic_recall(
                    workspace, query, tags, priority, limit, include_superseded, kind,
                    min_similarity, max_age_days, sort_by, session_id=session_id, return_scores=True,
                )
            except Exception:
                pass  # fall through to SQL LIKE (scoreless)
        notes = self.recall(
            workspace, query=query, tags=tags, priority=priority, limit=limit,
            include_superseded=include_superseded, kind=kind,
            min_similarity=min_similarity, max_age_days=max_age_days, sort_by=sort_by,
            session_id=session_id,
        )
        return [(n, None) for n in notes]

    def _semantic_recall(
        self,
        workspace: str,
        query: str,
        tags: list[str] | None,
        priority: str | None,
        limit: int,
        include_superseded: bool,
        kind: str | None = None,
        min_similarity: float | None = None,
        max_age_days: float | None = None,
        sort_by: str = "relevance",
        session_id: str | None = None,
        return_scores: bool = False,
    ) -> list:
        """Find the most relevant notes by cosine similarity, then fetch from SQLite.

        When min_similarity is set (UPG-5.1), candidates whose cosine similarity
        falls below the floor are dropped, so an off-topic query recalls nothing
        instead of the nearest-but-irrelevant note. ChromaDB cosine distance is
        `1 - cosine_similarity`, so similarity = `1 - distance`.

        max_age_days: applied as a SQL filter after candidate fetch (UPG-RECALL-HIERARCHY).
        sort_by: 'relevance' preserves semantic order; 'recency'/'priority'/
        'chronological' each re-sort the candidate set after fetch, with a
        sort-aware SQL prefetch guaranteeing the true top-`limit` rows by
        that key are present even when they missed the semantic pool
        (UPG-RECALL-HIERARCHY, UPG-DECISION-TIMELINE,
        UPG-RECALL-SEMANTIC-POOL-SORT-TRUNCATION).
        session_id: scope="session" enforcement (TRIGGER-ENGINE wave 2a) —
        see recall()'s docstring; the same post-LIMIT trade-off applies.
        """
        with timed_chroma_call("count"):
            col_count = self._notes_col.count()
        if col_count == 0:
            return []
        # GLOBAL collection, SQL-filtered only after the vector search:
        # over-fetch through the canonical width helper (docstring carries
        # the full invariant; the col_count cap avoids ChromaDB errors on
        # collections smaller than the ask).
        #
        # UPG-SEMANTIC-UNDERFILL: a `kind` filter is a SQL-side predicate
        # whose selectivity against the same-workspace corpus is unknowable
        # at fetch time. The base width is `limit * 3`; if 90% of the pool
        # is the wrong kind, the render still leaves the caller short of
        # `limit` with no signal that the shortfall is a pool artifact
        # rather than the store's true answer. `kind` is the one filter
        # whose selectivity has a rough prior (kind takes one of 7 values,
        # so even a perfectly-distributed corpus has ~14% of rows passing),
        # so deepen the pool with `floor=limit*10` only when `kind` is set.
        # This is a guess, not a measurement: the cost is a single ChromaDB
        # query at a wider `n_results` (cosine over pre-computed vectors,
        # the actual embeddings are not recomputed), which the
        # `working_memory_fetch_width` helper's col_count cap still bounds
        # for small corpora. min_similarity is intentionally NOT widened
        # here: it's a cutoff on the score itself, so a query whose whole
        # pool sits below the floor SHOULD return nothing — that's correct
        # behavior, not underfill.
        kind_floor = limit * 10 if kind else 0
        n_query = working_memory_fetch_width(limit, col_count, floor=kind_floor)

        q_vec = self._embed_query_fn([query])[0]
        with timed_chroma_call("query"):
            results = self._notes_col.query(query_embeddings=[q_vec], n_results=n_query)

        if not results or not results["ids"] or not results["ids"][0]:
            return []

        raw_ids = results["ids"][0]
        distances = (results.get("distances") or [[None] * len(raw_ids)])[0]
        candidate_ids = [int(id_) for id_ in raw_ids]
        # Per-note cosine similarity (1 - distance), kept for the scored recall
        # path (UPG-PRO-1). Computed here where the distances are still in hand;
        # None when ChromaDB returned no distance for a candidate.
        id_to_sim: dict[int, float | None] = {
            int(i): (1.0 - d) if d is not None else None
            for i, d in zip(raw_ids, distances)
        }

        # Relevance cutoff (UPG-5.1) — withhold candidates below the similarity floor.
        if min_similarity is not None:
            id_dist = {int(i): d for i, d in zip(raw_ids, distances)}
            candidate_ids = [
                nid for nid in candidate_ids
                if id_dist.get(nid) is None or (1.0 - id_dist[nid]) >= min_similarity
            ]
            if not candidate_ids:
                return []

        # Metadata filters, built ONCE and shared by two queries: the
        # semantic-candidate fetch immediately below, and the sort-aware
        # prefetch in the non-relevance branch further down
        # (UPG-RECALL-SEMANTIC-POOL-SORT-TRUNCATION). They deliberately
        # EXCLUDE the `note_id IN (...)` semantic restriction, which is
        # appended only to the candidate fetch: the prefetch's entire job is
        # to reach rows that MISSED the semantic pool, so reusing a WHERE
        # clause that pins membership to that same pool would make it a
        # no-op.
        filter_sql = ""
        filter_params: list = []

        if not include_superseded:
            filter_sql += " AND valid_until IS NULL"

        if priority:
            filter_sql += " AND priority = ?"
            filter_params.append(priority)

        if kind:
            filter_sql += " AND kind = ?"
            filter_params.append(kind)

        if tags:
            tag_clauses = " OR ".join(["tags LIKE ?" for _ in tags])
            filter_sql += f" AND ({tag_clauses})"
            filter_params.extend([f'%"{t}"%' for t in tags])

        if max_age_days is not None:
            cutoff = time.time() - max_age_days * 86400
            filter_sql += " AND created_at >= ?"
            filter_params.append(cutoff)

        # Fetch from SQLite by semantic candidate IDs, applying metadata filters
        placeholders = ",".join("?" * len(candidate_ids))
        sql = (
            f"SELECT * FROM notes WHERE workspace = ? AND note_id IN ({placeholders})"
            + filter_sql
        )
        params: list = [workspace, *candidate_ids, *filter_params]

        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()

        # Preserve semantic rank order by default (ChromaDB returns by ascending distance)
        id_to_row = {r["note_id"]: r for r in rows}
        if sort_by == "relevance":
            # UPG-SEMANTIC-UNDERFILL: a `kind` filter can leave the
            # relevance branch under-filled even after the fetch width
            # deepening above. The pool-deepening is a guess; this prefetch
            # is the deterministic backstop — it re-runs the same
            # `kind`/`valid_until`/etc. metadata filters (NO semantic
            # `note_id IN (...)` clause, otherwise the prefetch would be a
            # no-op) and unions the survivors into the rank-ordered pool.
            # Survivors arrive appended to the end of candidate_ids so the
            # `[:limit]` below keeps the existing rank order for notes that
            # were in the original pool and only fills the tail with these
            # additional same-workspace, kind-matching rows. min_similarity
            # is NOT applied here: it's a vector-score cutoff, not a
            # selectivity filter, so prefetched rows have no measured score
            # to compare against. (This mirrors the existing
            # sort-aware-prefetch in the non-relevance branch below, which
            # also exempts prefetched rows from min_similarity — see the
            # UPG-RECALL-SEMANTIC-POOL-SORT-TRUNCATION comment.)
            if kind:
                prefetch_sql_kind = (
                    "SELECT * FROM notes WHERE workspace = ?"
                    + filter_sql
                    + " ORDER BY created_at DESC LIMIT ?"
                )
                with self._conn() as conn:
                    rows_kind = conn.execute(
                        prefetch_sql_kind, [workspace, *filter_params, limit],
                    ).fetchall()
                existing_ids = set(id_to_row.keys())
                for r in rows_kind:
                    if r["note_id"] not in existing_ids:
                        id_to_row[r["note_id"]] = r
                        # Append to candidate_ids so the [:limit] below
                        # keeps the semantic rank order for pool survivors
                        # and only fills the tail with the prefetched rows.
                        candidate_ids.append(r["note_id"])
            ordered = [id_to_row[nid] for nid in candidate_ids if nid in id_to_row][:limit]
            notes = [self._row_to_note(r) for r in ordered]
        else:
            # UPG-RECALL-SEMANTIC-POOL-SORT-TRUNCATION: the pool holds only
            # the top `limit * 3` SEMANTIC matches, so when more notes than
            # that pass the metadata filters above, re-sorting just the pool
            # can silently drop the true oldest/highest-priority/newest note
            # (it never made the semantic pool). For an explicit non-relevance
            # sort, re-run the same METADATA filters (without the semantic
            # `note_id IN (...)` restriction) with that sort's ORDER BY so the
            # true top-`limit` rows by the requested key are guaranteed
            # present, and union them into the pool (setdefault keeps every
            # semantic-pool row already fetched). Membership deliberately may
            # ignore query similarity here: a caller naming a sort invokes an
            # ordering contract over the filtered corpus, not "the nearest few,
            # reordered". min_similarity does NOT gate these prefetched rows —
            # cosine similarity is only defined within the measured pool.
            # Unknown sort_by values get no prefetch, matching recall()'s SQL
            # path's fall-through-to-default leniency.
            _SORT_ORDER_SQL = {
                # ORDER BY fragments mirror recall()'s canonical copies above.
                "recency": " ORDER BY created_at DESC LIMIT ?",
                "priority": (
                    " ORDER BY CASE priority WHEN 'high' THEN 0"
                    " WHEN 'medium' THEN 1 ELSE 2 END, created_at DESC LIMIT ?"
                ),
                "chronological": " ORDER BY created_at ASC, note_id ASC LIMIT ?",
            }
            order_sql = _SORT_ORDER_SQL.get(sort_by)
            if order_sql is not None:
                prefetch_sql = (
                    "SELECT * FROM notes WHERE workspace = ?" + filter_sql + order_sql
                )
                with self._conn() as conn:
                    rows_pre = conn.execute(
                        prefetch_sql, [workspace, *filter_params, limit]
                    ).fetchall()
                for r in rows_pre:
                    id_to_row.setdefault(r["note_id"], r)
            # Convert all candidates to notes, then re-sort.
            all_notes = [self._row_to_note(r) for r in id_to_row.values()]
            if sort_by == "recency":
                all_notes.sort(key=lambda n: n.created_at, reverse=True)
            elif sort_by == "priority":
                _prio_rank = {"high": 0, "medium": 1, "low": 2}
                all_notes.sort(key=lambda n: (_prio_rank.get(n.priority, 1), -n.created_at))
            elif sort_by == "chronological":
                # Oldest-first (UPG-DECISION-TIMELINE) — mirrors the SQL
                # path's ORDER BY created_at ASC, note_id ASC.
                all_notes.sort(key=lambda n: (n.created_at, n.note_id))
            notes = all_notes[:limit]

        notes = _scope_filter(notes, session_id=session_id)
        # UPG-MEMORY-DECAY-KIND-SCOPED: same expiry exclusion as recall()'s
        # SQL path — see that method's comment and _exclude_expired().
        notes = self._exclude_expired(workspace, notes)

        if notes:
            ids = [n.note_id for n in notes]
            with self._conn() as conn:
                conn.execute(
                    f"UPDATE notes SET last_accessed = ? WHERE note_id IN ({','.join('?' * len(ids))})",
                    [time.time(), *ids],
                )

        if return_scores:
            return [(n, id_to_sim.get(n.note_id)) for n in notes]
        return notes

    # ------------------------------------------------------------------
    # Embed-model stamp + migration (UPG-NOTES-EMBED-MIGRATION)
    # ------------------------------------------------------------------

    def _stored_notes_embed_model(self) -> str | None:
        """The embed model stamped on the 'working_memory' collection's
        metadata, or None if no stamp exists yet (fresh collection, or one
        created before this mechanism existed)."""
        if self._notes_col is None:
            return None
        metadata = self._notes_col.metadata or {}
        model = metadata.get(_NOTES_EMBED_MODEL_KEY)
        return str(model) if model else None

    def _write_notes_embed_model_stamp(self) -> None:
        """Stamp the 'working_memory' collection with the embed model that
        produced its CURRENT vectors.

        Two ChromaDB gotchas apply to `collection.modify(metadata=...)`, both
        confirmed against the installed client version:
          - It REPLACES the collection's metadata wholesale rather than
            merging into it, so passing only the changed key would drop
            every other existing metadata entry. The existing metadata is
            always merged with the new stamp first.
          - It unconditionally rejects an `hnsw:space` key in the passed
            metadata with a ValueError — "changing the distance function...
            is not supported" — even when the value is unchanged, so
            `hnsw:space` must be stripped from the merged dict before the
            call. This is safe: the collection's actual distance function is
            pinned at creation independent of the metadata dict shown
            afterward, so dropping the key from displayed metadata does not
            change query behavior (verified: cosine ordering is identical
            before and after a modify() call that omits `hnsw:space`).
        """
        if self._notes_col is None or not self._embed_model:
            return
        merged = {
            k: v for k, v in (self._notes_col.metadata or {}).items()
            if k != "hnsw:space"
        }
        merged[_NOTES_EMBED_MODEL_KEY] = self._embed_model
        try:
            with timed_chroma_call("modify"):
                self._notes_col.modify(metadata=merged)
        except Exception:
            pass

    def _reconcile_embed_model_stamp(self) -> None:
        """Migrate note vectors when the configured embed model differs from
        the stamp recorded on the 'working_memory' collection.

        A missing stamp on a collection that already holds vectors is treated
        as a mismatch, not a match: we cannot know what model produced those
        vectors, so re-embedding with the current model is the only way to
        make the space self-consistent either way. An empty/new collection
        has nothing to migrate and is simply stamped.
        """
        stamped = self._stored_notes_embed_model()
        if stamped == self._embed_model:
            return
        with timed_chroma_call("count"):
            count = self._notes_col.count()
        if count == 0:
            self._write_notes_embed_model_stamp()
            return
        start = time.time()
        migrated = self._reembed_all_notes()
        self._write_notes_embed_model_stamp()
        if self._notes_collection_recreated:
            logger.warning(
                "vectr: working_memory collection rebuilt for embed model "
                "switch %r -> %r — note vectors were rebuilt from SQLite "
                "content; note rows and every other collection are untouched",
                stamped, self._embed_model,
            )
        logger.info(
            "vectr: migrated %d working-memory note vector(s) from embed model "
            "%r to %r in %.2fs",
            migrated, stamped, self._embed_model, time.time() - start,
        )

    def _reembed_all_notes(self) -> int:
        """Re-embed every note row's content with the current embed_fn and
        overwrite its vector in the 'working_memory' collection (same id).

        Operates across ALL workspaces in this SQLite file — the collection
        keys vectors by note_id alone, with no workspace scoping — and
        includes superseded notes: they remain queryable via
        `recall(include_superseded=True)`, so their vectors must stay in the
        current embedding space too. Note content, ids, and every other SQL
        column are untouched; only the vector side is rewritten.
        """
        with self._conn() as conn:
            rows = conn.execute("SELECT note_id, content FROM notes").fetchall()
        if not rows:
            return 0

        ids: list[str] = []
        contents: list[str] = []
        for row in rows:
            content = row["content"]
            if self._encryptor:
                try:
                    content = self._encryptor.decrypt(content)
                except Exception:
                    continue  # skip a row that can't be decrypted rather than abort the migration
            ids.append(str(row["note_id"]))
            contents.append(content)
        if not contents:
            return 0

        # chromadb 1.x keeps `errors` off the top-level import (lazy), so this
        # import sits here rather than at the top of the module.
        from chromadb.errors import ChromaError

        self._notes_collection_recreated = False
        for start in range(0, len(contents), _NOTES_REEMBED_BATCH_SIZE):
            batch_ids = ids[start:start + _NOTES_REEMBED_BATCH_SIZE]
            batch_contents = contents[start:start + _NOTES_REEMBED_BATCH_SIZE]
            vectors = self._embed_fn(batch_contents)
            try:
                with timed_chroma_call("upsert"):
                    self._notes_col.upsert(ids=batch_ids, embeddings=vectors)
            except ChromaError as exc:
                if not _is_embedding_dimension_conflict(exc):
                    raise
                # The collection pinned its vector dimensionality on the first
                # vector written to it (an older model's output size), so a
                # model change that lands on a different size can never be
                # migrated by upsert alone — the vectors are rejected before a
                # single batch is written. The vectors are derived data: the
                # SQLite rows are the notes, and this migration rewrites
                # vectors only. So rebuild the `working_memory` collection only
                # — never any other chroma collection, never SQLite — and retry
                # the migration once. A failure after the retry propagates to
                # the caller's warning path, which leaves the SQLite notes
                # exactly as they were.
                logger.warning(
                    "vectr: note re-embed upsert failed (%s) — rebuilding the "
                    "working_memory collection and retrying once", exc,
                )
                self._recreate_notes_collection()
                self._notes_collection_recreated = True
                with timed_chroma_call("upsert"):
                    self._notes_col.upsert(ids=batch_ids, embeddings=vectors)
        return len(contents)

    def _recreate_notes_collection(self) -> None:
        """Drop and recreate the `working_memory` chroma collection ONLY.

        Scoped to that one collection: the index/code-symbol collections live
        in a different chroma client (app/service.py's) and are not touched,
        and no SQLite statement runs here — note rows, ids and every other
        column survive a rebuild. `get_or_create_collection` re-applies the
        same creation metadata (`hnsw:space=cosine`) the startup paths use, so
        the rebuilt collection has the same contract as a fresh one, and the
        next vector written re-pins the dimensionality to the CURRENT embed
        model.
        """
        self._notes_collection_recreated = True
        if self._notes_chroma_client is None:
            self._notes_col = None
            return
        try:
            with timed_chroma_call("delete"):
                self._notes_chroma_client.delete_collection("working_memory")
        except Exception:
            pass  # nothing to delete — create below
        self._notes_col = self._notes_chroma_client.get_or_create_collection(
            name="working_memory",
            metadata={"hnsw:space": "cosine"},
        )

    def backfill_missing_vectors(self) -> int:
        """Embed and store a vector for every note that doesn't have one yet
        (UPG-STDIO-MEMORY-READY).

        A note can be missing a vector entirely — as opposed to having a
        stale one (`_reembed_all_notes` above) — when it was written while
        `self._embed_fn` was still None, i.e. during the window between
        process start and the embedding model finishing load/download. Those
        notes are never lost (they wrote to SQLite immediately, per
        remember()'s "embedding failure never blocks the write path"
        contract) but they need this pass once an embedder becomes available
        so they become semantically recallable without any re-write.

        Idempotent: a note already present in the 'working_memory'
        collection's id set is left untouched and never re-embedded — safe
        to call after every attach_embedder(), and more than once. Same
        cross-workspace scope as `_reembed_all_notes`: the collection keys
        vectors by note_id alone, with no workspace scoping, and includes
        superseded notes for the same reason (see that method's docstring).
        """
        if self._notes_col is None or self._embed_fn is None:
            return 0
        with self._conn() as conn:
            rows = conn.execute("SELECT note_id, content FROM notes").fetchall()
        if not rows:
            return 0

        with timed_chroma_call("get"):
            existing_ids: set[str] = set(self._notes_col.get(include=[])["ids"])

        ids: list[str] = []
        contents: list[str] = []
        for row in rows:
            note_id = str(row["note_id"])
            if note_id in existing_ids:
                continue  # already has a current vector — never re-embedded
            content = row["content"]
            if self._encryptor:
                try:
                    content = self._encryptor.decrypt(content)
                except Exception:
                    continue  # skip a row that can't be decrypted rather than abort the backfill
            ids.append(note_id)
            contents.append(content)
        if not contents:
            return 0

        for start in range(0, len(contents), _NOTES_REEMBED_BATCH_SIZE):
            batch_ids = ids[start:start + _NOTES_REEMBED_BATCH_SIZE]
            batch_contents = contents[start:start + _NOTES_REEMBED_BATCH_SIZE]
            vectors = self._embed_fn(batch_contents)
            with timed_chroma_call("upsert"):
                self._notes_col.upsert(ids=batch_ids, embeddings=vectors)
        logger.info("vectr: backfilled %d working-memory note vector(s)", len(contents))
        return len(contents)

    def embed_model_stamp_mismatch(self) -> str | None:
        """Return the stamped embed model if it still differs from the
        configured one, else None.

        Migration runs synchronously in the constructor, so under normal
        operation this returns None by the time __init__ has returned. It
        exists so `vectr status` can surface a mid-failure state (e.g. the
        embedder was unavailable during the migration attempt) instead of
        silently masking a stale note-vector space.
        """
        if self._notes_col is None or not self._embed_model:
            return None
        stamped = self._stored_notes_embed_model()
        if stamped != self._embed_model:
            return stamped or "unknown (unstamped)"
        return None

    def _boot_task_notes(self, workspace: str) -> list[WorkingNote]:
        """The 'current task' selection SessionStart boot injection uses
        (UPG-9.2 / UPG-TASK-NOTE-INJECTION-RECENCY): kind='task', priority=
        'high' only, newest-first (created_at DESC, note_id DESC tie-break),
        capped at config.BOOT_MAX_TASK_NOTES.

        Extracted out of `boot_recall()` so `resume_state()` (UPG-RESUME-
        SURFACE) reuses this EXACT query for its own 'last task note' rather
        than re-deriving a possibly different notion of "current task" — the
        SessionStart injection and the `vectr resume` surface can never
        disagree on which task note is current, by construction.
        """
        from agent.config import BOOT_MAX_TASK_NOTES

        with self._conn() as conn:
            task_rows = conn.execute(
                "SELECT * FROM notes WHERE workspace = ? AND valid_until IS NULL "
                "AND kind = 'task' AND priority = 'high' "
                "ORDER BY created_at DESC, note_id DESC LIMIT ?",
                (workspace, BOOT_MAX_TASK_NOTES),
            ).fetchall()
        return [self._row_to_note(r) for r in task_rows]

    def boot_recall(self, workspace: str) -> list[WorkingNote]:
        """Unconditional 'boot set' for harness-injected recall (UPG-9.2).

        Returns ALL directive notes plus high-priority task notes — the
        must-never-miss memory that should reach the model every session
        regardless of the prompt. Deliberately NOT semantic and NOT gated on
        notes_count: a similarity miss on "never push to main" is unacceptable,
        and a SessionStart hook must work on a fresh (0-note) workspace without
        erroring. Returns [] when there is nothing to inject.

        Directives are returned first (they are imperatives), oldest-first so
        standing rules stay in a stable order, capped at
        config.BOOT_MAX_DIRECTIVE_NOTES. High-priority task notes follow, via
        `_boot_task_notes()` (UPG-TASK-NOTE-INJECTION-RECENCY): a task note is
        current-work state, so the boot set must surface the latest checkpoint
        first rather than whichever task note happens to be oldest, and must
        not grow unbounded as task notes accumulate over a long-running
        workspace. Does NOT bump last_accessed — boot injection is automatic,
        not an agency-driven access, so it must not interfere with decay.
        """
        from agent.config import BOOT_MAX_DIRECTIVE_NOTES

        with self._conn() as conn:
            directive_rows = conn.execute(
                "SELECT * FROM notes WHERE workspace = ? AND valid_until IS NULL "
                "AND kind = 'directive' ORDER BY created_at ASC LIMIT ?",
                (workspace, BOOT_MAX_DIRECTIVE_NOTES),
            ).fetchall()
        notes = [self._row_to_note(r) for r in directive_rows] + self._boot_task_notes(workspace)
        audit("RECALL", workspace=workspace, query="", notes_returned=len(notes), method="boot")
        return notes

    def resume_state(self, workspace: str, session_id: str | None = None) -> dict:
        """Deterministic 'pick up where you left off' selection (UPG-RESUME-
        SURFACE) — the shared internals behind `vectr resume` / GET /v1/resume
        / `vectr_resume`, and the ONLY place this selection is computed.

        `session_id`, when given, enforces scope="session" notes the same way
        a direct `recall()` call would (a gotcha explicitly scoped to a
        different/no session is excluded) — the task-note query
        (`_boot_task_notes()`) is never scope-filtered, matching
        `boot_recall()`'s own unfiltered behaviour exactly.

        Returns:
          {
            "last_task": WorkingNote | None,   # most recent high-priority task
                                                # note, via `_boot_task_notes()`
                                                # — the SAME query SessionStart
                                                # boot injection uses, so this
                                                # can never show a different
                                                # 'current task' than what was
                                                # already injected.
            "gotchas": list[WorkingNote],       # open (non-superseded) kind=
                                                 # 'gotcha' notes, newest first,
                                                 # capped at config.RESUME_MAX_GOTCHAS.
            "snapshot": dict | None,            # latest saved snapshot —
                                                 # {'snapshot_id', 'label',
                                                 # 'created_at', 'note_count'};
                                                 # note_count is None when the
                                                 # payload could not be read
                                                 # (see restore_snapshot()).
            "gotchas_truncated": bool,          # True when more open gotchas
                                                 # exist beyond the cap — the
                                                 # render must disclose it.
          }

        Not semantic, not gated on notes_count — this is deterministic state
        (existing notes/snapshots), not a search; a fresh, empty workspace
        returns every field empty rather than erroring.
        """
        from agent.config import RESUME_MAX_GOTCHAS

        task_notes = self._boot_task_notes(workspace)
        last_task = task_notes[0] if task_notes else None

        # kind='gotcha' newest-first, reusing the same recall() every other
        # kind-filtered query goes through (sort_by='recency' is a plain SQL
        # ORDER BY — no new ranking invented for this surface). Over-fetch by
        # one so the render can disclose truncation instead of silently
        # capping — a reader must be able to tell "5 gotchas" from "5 of 9".
        gotchas = self.recall(
            workspace, kind="gotcha", limit=RESUME_MAX_GOTCHAS + 1, sort_by="recency",
            session_id=session_id,
        )
        gotchas_truncated = len(gotchas) > RESUME_MAX_GOTCHAS
        gotchas = gotchas[:RESUME_MAX_GOTCHAS]

        snapshot: dict | None = None
        snaps = self.list_snapshots(workspace)
        if snaps:
            latest = snaps[0]
            payload = self.restore_snapshot(latest["snapshot_id"])
            note_count = len(payload["notes"]) if payload else None
            snapshot = {**latest, "note_count": note_count}

        return {
            "last_task": last_task,
            "gotchas": gotchas,
            "snapshot": snapshot,
            "gotchas_truncated": gotchas_truncated,
        }

    def format_resume(
        self,
        state: dict,
        workspace: str,
        *,
        stale_warnings: dict[int, list[str]] | None = None,
        surface: str = "mcp",
    ) -> str:
        """Render a `resume_state()` result as one sectioned, token-bounded
        block (UPG-RESUME-SURFACE) — shared by the CLI/REST/MCP `resume`
        surfaces so none of them re-derive their own rendering.

        Sections that have nothing to show are OMITTED entirely (no snapshot
        -> no snapshot section; no gotchas -> no gotchas section), per the
        same "never inject noise" convention `boot_recall()`/
        `format_notes_for_llm()` already follow. An entirely empty state
        returns friendly guidance naming `vectr_remember` rather than an
        error or a bare "nothing found".

        `surface` controls the note-id form and expand hint exactly as
        `format_notes_for_llm()` does — 'mcp' -> `vectr_recall(note_id=N)`,
        'cli' -> `vectr recall --id N`.

        UPG-MEMORY-STATE-MACHINE §4.1: internally folds `note_events` for
        the task/gotchas being rendered so a revoked one carries the same
        `[REVOKED]` marker `format_notes_for_llm()`/`fire_and_format()`
        already show — no new parameter for callers, computed here from the
        `workspace` this method already takes.
        """
        stale_warnings = stale_warnings or {}
        last_task = state["last_task"]
        gotchas = state["gotchas"]
        snapshot = state["snapshot"]

        note_states = self.note_event_states(
            workspace, ([last_task] if last_task is not None else []) + list(gotchas)
        )

        sections: list[str] = []

        if last_task is not None:
            sections.append(
                "Last task:\n  "
                + _format_index_line(last_task, stale_warnings, surface=surface, note_states=note_states)
            )

        if snapshot is not None:
            import datetime
            ts = datetime.datetime.fromtimestamp(snapshot["created_at"]).strftime("%Y-%m-%d %H:%M")
            count = snapshot["note_count"]
            count_str = f"{count} notes" if count is not None else "note count unavailable"
            sections.append(
                f"Latest snapshot: {snapshot['snapshot_id']}  [{ts}]  "
                f"{snapshot['label']}  ({count_str})"
            )

        if gotchas:
            lines = [f"Open gotchas ({len(gotchas)}):"]
            for g in gotchas:
                anchor_str = ""
                paths = [a[0] for a in (g.anchors or []) if a]
                if paths:
                    anchor_str = "  [" + ", ".join(paths) + "]"
                lines.append(
                    "  " + _format_index_line(g, stale_warnings, surface=surface, note_states=note_states)
                    + anchor_str
                )
            if state.get("gotchas_truncated"):
                gotcha_hint = (
                    'vectr_recall(kind="gotcha")' if surface == "mcp"
                    else "vectr recall --kind gotcha"
                )
                lines.append(f"  …more open gotchas exist — {gotcha_hint} lists all")
            sections.append("\n".join(lines))

        if not sections:
            return (
                "Nothing to resume yet — no task notes, snapshots, or gotchas "
                "recorded for this workspace. Use vectr_remember(kind='task', ...) "
                "to start one."
            )

        expand_hint = "vectr_recall(note_id=N)" if surface == "mcp" else "vectr recall --id N"
        header = f"# Resume — pick up where you left off (use {expand_hint} to expand a note)\n"
        return header + "\n\n".join(sections)

    def get_note(self, workspace: str, note_id: int) -> "WorkingNote | None":
        """Fetch a single note by ID for the expand path (UPG-RECALL-HIERARCHY).

        Returns None if the note does not exist or belongs to a different workspace.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM notes WHERE workspace = ? AND note_id = ?",
                (workspace, note_id),
            ).fetchone()
        return self._row_to_note(row) if row is not None else None

    def notes_for_export(self, workspace: str) -> list[WorkingNote]:
        """Every note still in this workspace's table (not hard-deleted via
        forget), in a fixed note_id-ascending order — the deterministic
        source list for `vectr memory export`
        (UPG-MEMORY-LEGIBLE-FILE-PROJECTION part (a)).

        Unlike recall()/format_notes_for_llm(), this is deliberately
        unfiltered: it includes superseded and revoked notes (the export
        mirror is meant to read as a full audit log of what memory learned,
        not a ranked or narrowed view) and is never relevance-ranked, so
        calling this twice with no intervening write returns the identical
        list in the identical order every time — the property the export
        file's byte-identical-re-export acceptance bar depends on.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM notes WHERE workspace = ? ORDER BY note_id ASC",
                (workspace,),
            ).fetchall()
        return [self._row_to_note(r) for r in rows]

    def notes_for_edit(self, workspace: str) -> list[WorkingNote]:
        """Every note in `workspace` whose folded lifecycle state (UPG-
        MEMORY-STATE-MACHINE §4.1) is `active` — the candidate set for
        `vectr memory edit` (UPG-MEMORY-LEGIBLE-FILE-PROJECTION part (b)),
        in the same note_id-ascending order `notes_for_export()` uses.

        Deliberately NARROWER than `notes_for_export()`, which renders
        superseded and revoked notes too because the export mirror is a
        full audit log — an edit buffer must offer only notes a caller
        could sensibly change: editing a revoked note's anti-memory
        deterrent text is meaningless, and presenting a superseded note's
        stale content as "deletable" invites deleting the wrong half of a
        supersession pair (the supersession itself already IS that note's
        retirement).

        `valid_until IS NULL` alone is not a sufficient filter here (unlike
        `_floor_candidate_pool`'s use of it, where a revoked note is fine to
        include): it excludes superseded notes, but a revoked note keeps
        `valid_until` NULL too — only the `note_events` fold distinguishes
        "revoked" from "active". This method's own filtering step below is
        the same fold `_exclude_expired()` already performs, just inverted
        to KEEP `active` instead of dropping `expired`.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM notes WHERE workspace = ? AND valid_until IS NULL ORDER BY note_id ASC",
                (workspace,),
            ).fetchall()
        notes = [self._row_to_note(r) for r in rows]
        if not notes:
            return notes
        states = self._note_event_states_by_ids(workspace, [n.note_id for n in notes])
        return [n for n in notes if states.get(n.note_id, {}).get("state", "active") == "active"]

    def update_note_fields(
        self,
        workspace: str,
        note_id: int,
        kind: str | None = None,
        priority: str | None = None,
        tags: list[str] | None = None,
    ) -> bool:
        """In-place metadata update for `kind`/`priority`/`tags` — mutable
        columns, unlike `content` (protected by the append-only lifecycle
        invariant; a content change is always a `remember(supersedes=...)`
        write, never a mutation of an existing row — see `remember()`'s
        docstring). Powers `vectr memory edit`'s field-update mapping
        (UPG-MEMORY-LEGIBLE-FILE-PROJECTION part (b)): body unchanged,
        kind/priority/tags changed. Same "plain UPDATE + one `audit()` call,
        no `note_events` row" shape `set_pinned()` already uses for its own
        directly-mutable column — kind/priority/tags carry no lifecycle-
        state meaning of their own, so there is nothing here for the
        append-only fold to need to reverse.

        Only the fields actually passed (not None) are written; at least
        one must be given. An unrecognised `kind` falls back to
        `DEFAULT_KIND`, matching `remember()`'s own silent-fallback
        behaviour for the same input. Returns False if the note does not
        exist in this workspace.
        """
        if kind is None and priority is None and tags is None:
            raise ValueError("update_note_fields requires at least one of kind/priority/tags")
        if kind is not None and kind not in VALID_KINDS:
            kind = DEFAULT_KIND
        note = self.get_note(workspace, note_id)
        if note is None:
            return False
        set_clauses: list[str] = []
        params: list = []
        if kind is not None:
            set_clauses.append("kind = ?")
            params.append(kind)
        if priority is not None:
            set_clauses.append("priority = ?")
            params.append(priority)
        if tags is not None:
            set_clauses.append("tags = ?")
            params.append(json.dumps(tags))
        params.extend([workspace, note_id])
        with self._conn() as conn:
            conn.execute(
                f"UPDATE notes SET {', '.join(set_clauses)} WHERE workspace = ? AND note_id = ?",
                params,
            )
        audit(
            "MEMORY_EDIT_FIELD_UPDATE", workspace=workspace, note_id=note_id,
            kind=kind or "", priority=priority or "",
            tags=",".join(tags) if tags is not None else "",
        )
        return True

    def recall_for_path(
        self,
        workspace: str,
        file_path: str,
        kind: str | None = None,
        limit: int = 10,
        session_id: str | None = None,
    ) -> list[WorkingNote]:
        """Recall notes anchored to a specific file (UPG-9.6).

        Matches notes whose content mentions the file's basename or its
        workspace-relative path at a genuine path boundary, whose declared
        `anchors` list the file exactly — the anchor a typed gotcha
        actually carries ("index_file in symbol_graph.py takes workspace
        first") — or whose EXPLICIT `triggers[]` declares a 'path' glob
        that matches the file (UPG-TRIGGERS-INERT-ON-PROXY-STRUCTURAL:
        `triggers` is just as deliberate a "this note concerns this file"
        declaration as `anchors` — before this fix, a note with a path
        glob ONLY in `triggers`, never mentioned in its body and never in
        `anchors`, was invisible to this method even though the same glob
        already fires it live via the PreToolUse hook's trigger engine).
        Combined with `kind="gotcha"` this powers the PreToolUse hook:
        editing a file surfaces the caveat recorded against it, and an
        unrelated file matches nothing.

        UPG-PROXY-SUBSTRING-ANCHOR: the SQL LIKE clause is a cheap SUPERSET
        prefilter only (it would match "gate.py" inside "uv_regate.py");
        `_path_boundary_match()`, `_anchors_exact_match()`, and (new)
        `path_trigger_match()` narrow it down to genuine matches afterward.
        Because that narrowing happens after the SQL fetch, the fetch
        pulls a bounded over-fetched pool (see `memory_recall_for_path` in
        config.yaml) rather than exactly `limit` rows — otherwise a true
        match could be discarded before the boundary filter ever saw it,
        purely because it didn't fit in an under-sized SQL LIMIT. Not
        semantic — a substring anchor avoids false "nearby file" hits. The
        triggers SQL arm is a superset too: a trigger's `path` value is a
        glob (e.g. "agent/*.py"), which a SQL LIKE cannot itself evaluate,
        so the prefilter only checks that the note declares SOME 'path'
        trigger at all (`triggers LIKE '%"path"%'`) — `path_trigger_match
        ()` does the real glob evaluation afterward, exactly like the
        other two arms narrow their own SQL supersets.

        session_id: scope="session" enforcement, same as recall(). `file_path`
        (already a parameter here) also enforces scope="path-subtree" — the
        one recall path where that's free, since the file context already
        exists (TRIGGER-ENGINE wave 2a, §1). A real caller (e.g. the
        PreToolUse hook) sends an ABSOLUTE `file_path`; `_path_trigger_
        candidates()` (shared with `fire()`) resolves its workspace-relative
        form too, so a path-subtree-scoped note anchored the natural,
        workspace-relative way still matches (F1b — the same abs/rel
        candidate-matching fix as the P trigger primitive, applied here to
        the scope check).
        """
        from agent.config import (
            MEMORY_RECALL_FOR_PATH_OVERFETCH_CEILING,
            MEMORY_RECALL_FOR_PATH_OVERFETCH_MULTIPLIER,
        )
        from agent.trigger_engine import path_trigger_match

        basename = Path(file_path).name
        if not basename:
            return []
        path_candidates = _path_trigger_candidates(workspace, file_path)
        relpath = next((c for c in (path_candidates or ()) if c != file_path), "")
        relpath_or_basename = relpath if relpath else basename
        # `path_candidates` is "as-given plus workspace-relative, plus the
        # workspace-rooted absolute form for a relative input"
        # (`_path_trigger_candidates()`'s own contract, shared with `fire()`'s
        # live P-primitive evaluation) — it has no bare-basename form of its
        # own unless the file happens to sit directly at the workspace root
        # (where relpath == basename already). The content-mention arm below
        # sidesteps this by checking `basename` and `relpath` as two SEPARATE
        # substring probes; `path_trigger_match()`'s glob evaluation instead
        # needs every candidate in one sequence, so basename is appended
        # explicitly here — the same explicit-basename-alongside-full-path
        # shape `agent/proactive/matcher.py`'s `_first_anchor()` already uses
        # for its own `path_trigger_match()` call over window file paths.
        trigger_path_candidates = tuple(path_candidates or ()) + (basename,)

        pool_size = max(
            limit,
            min(limit * MEMORY_RECALL_FOR_PATH_OVERFETCH_MULTIPLIER, MEMORY_RECALL_FOR_PATH_OVERFETCH_CEILING),
        )

        # The anchors LIKE arms are UNQUOTED — same superset treatment as the
        # content arms above, not the tighter quoted form. A declared anchor
        # is legitimately either workspace-relative OR absolute (remember()'s
        # own `anchors` docstring), and an absolute anchor's JSON-serialized
        # form is `".../<basename>"` — the character right before the
        # basename is "/", not a quote, so a quoted `%"<basename>"%` pattern
        # would silently never match an absolute anchor. `_anchors_exact_match()`
        # below is what does the precise narrowing; the SQL step here only
        # needs to guarantee no true match is excluded from the pool.
        #
        # UPG-TRIGGERS-INERT-ON-PROXY-STRUCTURAL regression fix: the triggers
        # arm runs as its OWN query with its OWN `LIMIT pool_size`, never
        # OR-ed into the content/anchors query. `triggers LIKE '%"path"%'`
        # cannot be scoped to this file the way the content/anchors arms are
        # (a declared trigger is a glob, e.g. "**/*.py", evaluated below via
        # `path_trigger_match()` — no LIKE pattern expresses glob semantics),
        # so it is a superset over EVERY note that declares ANY path trigger
        # for ANY file, workspace-wide. Folding that superset into one shared
        # `LIMIT`-bounded query meant a workspace with enough trigger-
        # declaring notes could push genuinely file-scoped content/anchor
        # matches off the end of the pool before this method's own Python-
        # side narrowing ever ran — a single note declaring a broad glob
        # like "**/*.py" could silently starve every other file's structural
        # recall. Two independently-bounded queries hold the pool size
        # invariant regardless of how many trigger-declaring notes exist.
        content_anchors_sql = (
            "SELECT * FROM notes WHERE workspace = ? AND valid_until IS NULL "
            "AND (content LIKE ? OR content LIKE ? OR anchors LIKE ? OR anchors LIKE ?)"
        )
        content_anchors_params: list = [
            workspace,
            f"%{basename}%",
            f"%{relpath_or_basename}%",
            f"%{basename}%",
            f"%{relpath_or_basename}%",
        ]
        triggers_sql = (
            "SELECT * FROM notes WHERE workspace = ? AND valid_until IS NULL "
            "AND triggers LIKE ?"
        )
        triggers_params: list = [workspace, '%"path"%']
        if kind:
            content_anchors_sql += " AND kind = ?"
            content_anchors_params.append(kind)
            triggers_sql += " AND kind = ?"
            triggers_params.append(kind)
        # UPG-RECALL-ORDER-CHURN: same deterministic tie-break as recall()'s
        # default SQL path — last_accessed excluded (recall() bumps it on
        # every note it returns, which would otherwise reorder these ties on
        # the very next call). UPG-TASK-NOTE-INJECTION-RECENCY: same kind='task'
        # trust/decay exemption as recall() — see that method's docstring.
        order_by = (
            " ORDER BY"
            " (CASE WHEN kind = 'task' THEN 1.0 ELSE author_trust_score END) DESC,"
            " (CASE WHEN kind = 'task' THEN 1.0 ELSE decay_score END) DESC,"
            " created_at DESC, note_id DESC LIMIT ?"
        )
        content_anchors_sql += order_by
        content_anchors_params.append(pool_size)
        triggers_sql += order_by
        triggers_params.append(pool_size)

        with self._conn() as conn:
            content_anchors_rows = conn.execute(content_anchors_sql, content_anchors_params).fetchall()
            triggers_rows = conn.execute(triggers_sql, triggers_params).fetchall()

        # Merge the two independently-bounded pools, deduping by note_id (a
        # note can legitimately surface from both queries — e.g. it mentions
        # this file in its content AND declares an unrelated path trigger —
        # and must only occupy one slot), then re-sort the union with the
        # same ordering the two SQL queries already used, so a note from
        # either arm competes on equal footing rather than the content/
        # anchors arm's rows always sorting ahead by construction.
        def _sort_key(note: WorkingNote) -> tuple:
            trust = 1.0 if note.kind == "task" else note.author_trust_score
            decay = 1.0 if note.kind == "task" else note.decay_score
            return (trust, decay, note.created_at, note.note_id)

        merged_by_id: dict[int, WorkingNote] = {}
        for row in (*content_anchors_rows, *triggers_rows):
            note = self._row_to_note(row)
            merged_by_id.setdefault(note.note_id, note)
        pool = sorted(merged_by_id.values(), key=_sort_key, reverse=True)

        # Each match is tagged with how strongly it relates to this file —
        # 0 = declared anchor (strongest), 1 = declared triggers[].path glob,
        # 2 = content-boundary mention (weakest) — matching the same
        # relation hierarchy `agent/proactive/matcher.py`'s tier scoring
        # already encodes downstream (declared_anchor > declared_trigger >
        # mention). `pool` overflowing this method's own `limit` is now
        # common once a workspace has enough trigger-declaring notes (a
        # single broad glob like "**/*.py" genuinely matches every file, so
        # every such note is a real match for every path) — a plain
        # recency-ordered truncation would let a flood of newer, weaker
        # trigger/content matches silently displace an older but stronger
        # anchor match. Sorting by relation strength before truncating (a
        # structural note property, not a query-content check) guarantees
        # that never happens; `pool`'s existing recency order is preserved
        # as the tiebreak *within* each relation tier via Python's stable
        # sort.
        matched: list[tuple[int, WorkingNote]] = []
        for note in pool:
            # `trigger_path_candidates` (not the bare `path_candidates`) —
            # a note's `anchors` entry is legitimately either workspace-
            # relative/absolute OR a bare basename (`remember(anchors=[...])`
            # accepts either; `agent/proactive/matcher.py`'s own
            # `_first_anchor()` checks both `base in declared` and `path in
            # declared`). Checking only `path_candidates` here previously
            # meant a bare-basename anchor could never satisfy this exact-
            # match arm unless the file sat at the workspace root, silently
            # demoting a genuine declared-anchor note to a weaker
            # content-boundary classification (or, when its content never
            # happens to mention the basename, dropping it from `matched`
            # entirely).
            if _anchors_exact_match(note.anchors, trigger_path_candidates):
                matched.append((0, note))
                continue
            if path_trigger_match(note.triggers, trigger_path_candidates) is not None:
                matched.append((1, note))
                continue
            content = note.content or ""
            if _path_boundary_match(content, basename) or (
                relpath and _path_boundary_match(content, relpath)
            ):
                matched.append((2, note))
        matched.sort(key=lambda pair: pair[0])
        notes = [note for _, note in matched[:limit]]

        notes = _scope_filter(notes, session_id=session_id, file_path=path_candidates)
        # UPG-MEMORY-DECAY-KIND-SCOPED: same expiry exclusion as recall()'s
        # SQL path — see that method's comment and _exclude_expired().
        notes = self._exclude_expired(workspace, notes)
        audit("RECALL", workspace=workspace, query=basename, notes_returned=len(notes), method="path")
        return notes

    def forget(self, workspace: str, note_id: int) -> bool:
        """Explicitly delete a note (LLM decided it's no longer relevant)."""
        with self._conn() as conn:
            count = conn.execute(
                "DELETE FROM notes WHERE workspace = ? AND note_id = ?",
                (workspace, note_id),
            ).rowcount
            # TRIGGER-ENGINE wave 2b: a deleted note's symbol-trigger index
            # rows must go with it, or fire()'s "does this workspace have any
            # symbol-triggered notes" existence check would keep resolving
            # the symbol graph for a note that no longer exists.
            conn.execute(
                "DELETE FROM symbol_triggers WHERE workspace = ? AND note_id = ?",
                (workspace, note_id),
            )
            # UPG-MEMORY-STATE-MACHINE §4.1: forget() stays the one true
            # hard-delete escape hatch (the design doc's own framing) — a
            # forgotten note's event log goes with it rather than becoming a
            # dangling history for a note_id that no longer exists.
            conn.execute(
                "DELETE FROM note_events WHERE workspace = ? AND note_id = ?",
                (workspace, note_id),
            )
        if count > 0 and self._notes_col is not None:
            try:
                with timed_chroma_call("delete"):
                    self._notes_col.delete(ids=[str(note_id)])
            except Exception:
                pass
        return count > 0

    def count_notes(self, workspace: str) -> int:
        """Return the number of notes stored for this workspace."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM notes WHERE workspace = ?", (workspace,)
            ).fetchone()
        return row[0] if row else 0

    def stale_task_summary(self, workspace: str, min_age_days: float) -> tuple[int, int | None]:
        """Count live (non-superseded, non-tombstoned) kind="task" notes older
        than min_age_days, plus the oldest such note's id.

        Feeds the `vectr_status` memory-hygiene nudge (UPG-TASK-SUPERSEDES-
        HYGIENE): a task note left un-superseded keeps firing at every
        session-start forever, so the count/oldest-id pair lets the caller
        surface a one-line "consider supersedes/forget" warning. State-based
        only (kind + age + tombstone status) — never inspects note content,
        never mutates or expires anything itself.

        A note counts as "live" here the same way recall() treats it by
        default: a `valid_until IS NULL` row AND a folded event state of
        `active`. `valid_until IS NULL` alone is the original (pre-UPG-
        STALE-TASK-SUMMARY-COUNTS-DEAD-STATES) predicate — it caught the
        supersede tombstone (an explicit `valid_until` write) but missed
        two EVENT-fold terminal states the same store can leave a row in
        with `valid_until` still NULL:
          - REVOKED (`revoke_note()` never sets `valid_until` by the
            anti-memory contract — a revoked note stays a default recall()/
            fire() candidate so its deterrent rendering can substitute for
            the raw content; tombstoning it would silently hide exactly
            the wrongness the caller flagged).
          - EXPIRED (`purge_expired_notes()` appends the `expired` event
            and deliberately leaves the columns NULL, per
            UPG-MEMORY-DECAY-KIND-SCOPED's append-only invariant; the row
            stays visible via get_note() so a caller can still inspect
            what expired, only the default listing surfaces exclude it).
        Consequence before this fix: a stale-task nudge would recommend
        "consider supersedes=<old id>" for notes the system itself had
        already retired (expired) or the user had already judged wrong
        (revoked) — wrong-footed rather than harmful, but exactly the
        noise UPG-HYGIENE-THRESHOLD-RETUNE is trying to measure. The fix
        consults the event fold (UPG-MEMORY-STATE-MACHINE §4.1) for the
        candidate task-note set and drops every note whose folded state
        is not `active`. Notes with no events at all (pre-migration
        rows) fold to `active` by `fold()`'s own default — the same
        state a brand-new `created` event would produce, so old notes
        keep counting exactly as before.

        Cost profile change: the old shape was one SQL round trip
        returning `(count, oldest_id)`; the new shape is one SQL round
        trip for the candidate set, one batched `note_events` fetch +
        Python fold for those candidate IDs, and a final min() over the
        filtered list. The `note_events` fetch is bounded by the size of
        the candidate task-note set (not the whole workspace), so the
        added work is O(N) over the at-risk note population, not O(N)
        over the workspace. The derived-column alternative (cache the
        folded state on the row) would buy the same correctness at a
        schema-migration cost; rejected for the same reason
        `_exclude_expired()` chose the fold — a column duplicating
        something the append-only log already authoritatively says would
        be a permanent two-writer-update consistency hazard (every
        `revoke_note` / `reinstate_note` / `purge_expired_notes` call
        would have to remember to update the cache alongside the event
        write, or the cache would silently disagree with the fold).

        Oldest is by created_at ASC, note_id ASC tie-break, matching
        the ordering convention used elsewhere in this store.
        """
        cutoff = time.time() - min_age_days * 86400
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT note_id, created_at FROM notes
                WHERE workspace = ? AND kind = 'task' AND valid_until IS NULL
                      AND created_at < ?
                """,
                (workspace, cutoff),
            ).fetchall()
        if not rows:
            return 0, None
        # Fold the candidate set: `valid_until IS NULL` already excludes
        # the supersede tombstone (which DOES set `valid_until`), so the
        # remaining candidates can only be in EVENT-fold terminal states
        # `revoked` or `expired` (both leave `valid_until` NULL by their
        # contracts above). A note with no events folds to `active` —
        # the pre-migration default, preserved here.
        candidate_ids = [r["note_id"] for r in rows]
        states = self._note_event_states_by_ids(workspace, candidate_ids)
        live = [
            r for r in rows
            if states.get(r["note_id"], {}).get("state", "active") == "active"
        ]
        if not live:
            return 0, None
        # Oldest is the same ORDER BY rule the old SQL used: created_at
        # ASC, note_id ASC tie-break. min() with that key tuple gives the
        # same row the previous LIMIT 1 subquery did.
        oldest = min(live, key=lambda r: (r["created_at"], r["note_id"]))
        return len(live), oldest["note_id"]

    def forget_all(self, workspace: str) -> int:
        """Clear all notes AND snapshots for a workspace.

        Snapshots embed full note contents in their payload, so a purge that
        deleted only the notes table would silently keep every note's text
        alive in `snapshots` — "delete everything" must mean everything,
        including the note embedding vectors in the Chroma collection."""
        with self._conn() as conn:
            deleted = conn.execute(
                "DELETE FROM notes WHERE workspace = ?", (workspace,)
            ).rowcount
            conn.execute("DELETE FROM snapshots WHERE workspace = ?", (workspace,))
            conn.execute("DELETE FROM symbol_triggers WHERE workspace = ?", (workspace,))
        if deleted > 0 and self._notes_col is not None:
            try:
                with timed_chroma_call("get"):
                    existing_ids = self._notes_col.get(include=[])["ids"]
                if existing_ids:
                    with timed_chroma_call("delete"):
                        self._notes_col.delete(ids=existing_ids)
            except Exception:
                pass
        audit("FORGET_ALL", workspace=workspace, deleted=deleted)
        return deleted

    def forget_all_workspaces(self) -> int:
        """Delete ALL notes, snapshots, and note vectors across ALL workspaces
        in this SQLite file.

        Used by `vectr forget --all` to give a global clean slate — the same
        "everything means everything" contract as forget_all above.
        Audit entry logged per deletion.
        """
        with self._conn() as conn:
            deleted = conn.execute("DELETE FROM notes").rowcount
            conn.execute("DELETE FROM snapshots")
            conn.execute("DELETE FROM symbol_triggers")
        if self._notes_col is not None:
            try:
                with timed_chroma_call("get"):
                    existing_ids = self._notes_col.get(include=[])["ids"]
                if existing_ids:
                    with timed_chroma_call("delete"):
                        self._notes_col.delete(ids=existing_ids)
            except Exception:
                pass
        audit("FORGET_ALL_WORKSPACES", deleted=deleted)
        return deleted

    def purge_expired_notes(
        self, workspace: str, ttl_days: float | None = None, now: float | None = None,
    ) -> int:
        """Flag notes older than their kind's TTL as expired (UPG-MEMORY-
        DECAY-KIND-SCOPED) — an append-only `expired` note_events
        transition (actor='system', a deterministic non-judgment call, same
        class as `stale_flagged`), never a row DELETE (UPG-MEMORY-STATE-
        MACHINE §4.1's append-only invariant — the pre-fix implementation's
        `DELETE FROM notes WHERE created_at < cutoff` directly violated it,
        and was also kind-blind, deleting directives on the same schedule
        as scratch findings).

        Called at startup when VECTR_NOTES_TTL_DAYS is set (the live path,
        wired from `app/service.py`'s `start_background_index()`). `ttl_days`,
        when given, is that operator override — applied to every kind's TTL
        EXCEPT a kind whose baseline TTL is `null` in config.yaml's
        `memory_decay.ttl_days_by_kind` (`MEMORY_DECAY_TTL_DAYS_BY_KIND`) —
        that exemption ALWAYS wins over the override, so kind='directive'
        (and any future kind marked null) never expires regardless of what
        an operator sets VECTR_NOTES_TTL_DAYS to. `ttl_days=None` (the
        default) uses each kind's own configured TTL with no override.

        Idempotent: skips any note whose CURRENT folded lifecycle state
        (`note_event_states()`) is already 'expired' (a repeat call is a
        no-op for it, never a duplicate event) or 'revoked' (a revoked
        note's anti-memory deterrent must stay visible in default recall()/
        fire() forever, per its own contract — expiry must never
        additionally hide it).

        Returns the number of notes newly transitioned to 'expired' this
        call — NOT a delete count; nothing is ever deleted here (`forget()`
        remains the one true hard-delete escape hatch).
        """
        from agent.config import MEMORY_DECAY_TTL_DAYS_BY_KIND

        if now is None:
            now = time.time()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT note_id, kind, created_at FROM notes WHERE workspace = ?",
                (workspace,),
            ).fetchall()
        if not rows:
            return 0

        candidate_ids: list[int] = []
        for r in rows:
            kind = r["kind"] or DEFAULT_KIND
            baseline = MEMORY_DECAY_TTL_DAYS_BY_KIND[kind]
            if baseline is None:
                continue  # kind exempt from expiry at any age (e.g. directive)
            effective_ttl = ttl_days if ttl_days is not None else baseline
            cutoff = now - effective_ttl * 86400
            if r["created_at"] < cutoff:
                candidate_ids.append(r["note_id"])
        if not candidate_ids:
            return 0

        states = self._note_event_states_by_ids(workspace, candidate_ids)
        expired_count = 0
        with self._conn() as conn:
            for note_id in candidate_ids:
                state = states.get(note_id, {}).get("state", "active")
                if state in ("expired", "revoked"):
                    continue
                _append_event(
                    conn, workspace, note_id, "expired", actor="system",
                    reason="ttl exceeded", ts=now,
                )
                expired_count += 1
        if expired_count:
            audit("PURGE_EXPIRED", workspace=workspace, ttl_days=ttl_days, expired=expired_count)
        return expired_count

    def decay_old_notes(
        self, workspace: str, half_life_days: float | None = None, now: float | None = None,
    ) -> int:
        """Recompute every note's `decay_score` from its age (UPG-MEMORY-
        DECAY-KIND-SCOPED) — a pure, idempotent function of elapsed time
        (`pow(0.5, elapsed / half_life)`), ASSIGNED fresh on every call
        rather than multiplied onto the row's existing value. The pre-fix
        implementation did `decay_score = decay_score * pow(...)`, so
        calling it twice at the same clock reading silently double-decayed
        a note (0.5 then 0.25) instead of leaving it at 0.5 both times —
        this version depends only on `now` and the note's unchanging
        `created_at`, so two calls at an identical `now` always produce an
        identical decay_score.

        `half_life_days`, when given, overrides every kind's configured
        half-life EXCEPT a kind whose baseline half-life is `null` in
        config.yaml's `memory_decay.half_life_days_by_kind`
        (`MEMORY_DECAY_HALF_LIFE_DAYS_BY_KIND`) — kind='directive' today —
        mirroring `purge_expired_notes()`'s exemption contract exactly; an
        exempt note's decay_score is left untouched (stays at its existing
        value, 1.0 for a note that has never been touched by this method)
        regardless of age or override. `half_life_days=None` (the default —
        what the unconditional startup caller in `app/service.py` uses)
        applies each kind's own configured half-life with no override.

        decay_score is a ranking signal ONLY — the tie-break in recall()'s
        default `ORDER BY`. This method NEVER deletes a note regardless of
        how low its decay_score falls; the pre-fix implementation's
        `DELETE FROM notes WHERE decay_score < 0.1` violated UPG-MEMORY-
        STATE-MACHINE §4.1's append-only invariant exactly the same way
        `purge_expired_notes()`'s old DELETE did — `forget()` remains the
        one true hard-delete escape hatch.

        Returns the number of notes whose decay_score was recomputed
        (exempt-kind notes are skipped without a write and don't count).
        """
        from agent.config import MEMORY_DECAY_HALF_LIFE_DAYS_BY_KIND

        if now is None:
            now = time.time()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT note_id, kind, created_at FROM notes WHERE workspace = ?",
                (workspace,),
            ).fetchall()
        if not rows:
            return 0

        updated = 0
        with self._conn() as conn:
            for r in rows:
                kind = r["kind"] or DEFAULT_KIND
                baseline = MEMORY_DECAY_HALF_LIFE_DAYS_BY_KIND[kind]
                if baseline is None:
                    continue  # kind exempt from decay at any age (e.g. directive)
                effective_half_life = half_life_days if half_life_days is not None else baseline
                half_life_s = effective_half_life * 86400
                elapsed_s = now - r["created_at"]
                # Clamp to 1.0 for elapsed_s <= 0 (a note created at/after
                # `now`, e.g. clock skew) — pow(0.5, negative) is > 1.0,
                # which would rank a skewed note ABOVE a genuinely fresh one.
                score = pow(0.5, elapsed_s / half_life_s) if elapsed_s > 0 else 1.0
                conn.execute(
                    "UPDATE notes SET decay_score = ? WHERE workspace = ? AND note_id = ?",
                    (score, workspace, r["note_id"]),
                )
                updated += 1
        if updated:
            audit("DECAY_NOTES", workspace=workspace, half_life_days=half_life_days, updated=updated)
        return updated

    # ------------------------------------------------------------------
    # Snapshots — vectr_snapshot / vectr_restore
    # ------------------------------------------------------------------

    def snapshot(
        self,
        workspace: str,
        label: str,
        retrieved_chunks: list[dict] | None = None,
        session_id: str | None = None,
    ) -> str:
        """
        Save a session snapshot: all current notes + what was in context.
        Returns snapshot_id.
        """
        import hashlib
        snapshot_id = hashlib.md5(f"{workspace}{label}{time.time()}".encode()).hexdigest()[:12]
        notes = self.recall(workspace, limit=100)
        payload = json.dumps({
            "notes": [
                {
                    "note_id": n.note_id,
                    "content": n.content,
                    "tags": n.tags,
                    "priority": n.priority,
                }
                for n in notes
            ],
            "retrieved_chunks": retrieved_chunks or [],
            "session_id": session_id,
        })
        # The payload embeds decrypted note contents (recall() decrypts), so a
        # plaintext snapshots table would bypass note encryption entirely.
        # Encrypt the whole payload under the same key as note content.
        if self._encryptor:
            payload = self._encryptor.encrypt(payload)
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO snapshots (snapshot_id, workspace, label, payload, created_at) VALUES (?, ?, ?, ?, ?)",
                (snapshot_id, workspace, label, payload, time.time()),
            )
        return snapshot_id

    def list_snapshots(self, workspace: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT snapshot_id, label, created_at FROM snapshots WHERE workspace = ? ORDER BY created_at DESC",
                (workspace,),
            ).fetchall()
        return [{"snapshot_id": r["snapshot_id"], "label": r["label"], "created_at": r["created_at"]} for r in rows]

    def restore_snapshot(self, snapshot_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM snapshots WHERE snapshot_id = ?", (snapshot_id,)
            ).fetchone()
        if row is None:
            return None
        payload = row["payload"]
        if self._encryptor:
            # Tolerant decrypt: snapshots written before payload encryption (or
            # before a key was configured) pass through unchanged.
            payload = self._encryptor.decrypt(payload)
        try:
            return json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            # Ciphertext without the (correct) key configured — unreadable by
            # design; treat as not restorable rather than crashing the caller.
            logger.warning("snapshot %s payload is not readable (encrypted with a different key?)", snapshot_id)
            return None

    # ------------------------------------------------------------------
    # Eviction hints — which chunks can vectr re-retrieve in <50ms?
    # ------------------------------------------------------------------

    def build_eviction_hint(
        self,
        workspace: str,
        session_retrieved_chunks: list[dict],
    ) -> str:
        """
        Given a list of chunks the LLM has retrieved this session,
        return a message listing which chunks vectr can re-retrieve in <50ms.

        The guarantee: anything listed here is fully indexed, re-retrievable in <50ms.
        """
        if not session_retrieved_chunks:
            return "No retrieved chunks to evict."

        # estimate token cost (rough: 1 token ≈ 4 chars)
        total_chars = sum(len(c.get("content", "")) for c in session_retrieved_chunks)
        est_tokens = total_chars // 4

        by_file: dict[str, list[dict]] = {}
        for chunk in session_retrieved_chunks:
            f = chunk.get("file", "unknown")
            by_file.setdefault(f, []).append(chunk)

        lines = [
            f"Vectr has {len(session_retrieved_chunks)} chunks (~{est_tokens} tokens) indexed and instantly retrievable.",
            "Vectr can re-retrieve these in <50ms — no need to re-read them:",
            "",
        ]
        for fpath, chunks in by_file.items():
            line_ranges = ", ".join(f"lines {c.get('lines', '?')}" for c in chunks)
            lines.append(f"  {fpath}  [{line_ranges}]")

        lines += [
            "",
            "To retrieve any of them: vectr_search('<symbol name or description>')",
            "Recall latency: <50ms. Nothing will be lost.",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Staleness detection
    # ------------------------------------------------------------------

    def check_staleness(
        self,
        notes: list[WorkingNote],
        workspace_root: str,
    ) -> dict[int, list[str]]:
        """Identify notes whose referenced files have changed since the note was written.

        Composite staleness fires the stale flag when ANY of:
          - A referenced file's mtime > note.created_at (original mtime check)
          - note.code_hash != sha256[:16] of the current file content (code moved/changed)
          - Note is marked superseded (valid_until is set)
          - A declared anchor's (TRIGGER-ENGINE wave 1, bm2-design-skeleton.md
            §5) content hash no longer matches its hash-at-write — this is a
            VISIBLE caveat only, never a silent drop: the memory still fires/
            recalls, this dict just flags it. Reuses `_hash_path_content()`,
            the same sha256[:16] helper `remember()` uses to compute the
            anchor's original hash, so both sides apply the identical rule.

        Returns {note_id: [stale_path/reason, ...]} — only stale notes included.

        As a write side-effect (UPG-MEMORY-STATE-MACHINE §4.1/§4.4), every
        note whose DECLARED anchor (not the mtime/code_hash content-prose
        checks above — those are unrelated pre-existing signals) is
        currently drifted gets an idempotent `stale_flagged` event appended
        to its log, keyed on a SIGNATURE of that note's drifted anchors
        (sorted `path:current_hash` pairs, sha256[:16] — order-independent
        and deterministic), not merely on "is it drifted" — see
        `_flag_stale_anchors()`. This is the same "a read method also has a
        write side-effect" shape `fire()` (stamps `last_fired`) and
        `recall()` (bumps `last_accessed`) already have in this store.
        """
        root = Path(workspace_root)
        stale: dict[int, list[str]] = {}
        anchor_drift_sigs: dict[int, str] = {}

        for note in notes:
            reasons: list[str] = []
            drift_pairs: list[tuple[str, str]] = []

            # superseded notes are always stale
            if note.valid_until is not None:
                sup_by = note.superseded_by or (
                    f"note#{note.superseded_by_note_id}" if note.superseded_by_note_id else "unknown"
                )
                reasons.append(f"[superseded by @{sup_by}]")

            for raw_path in _extract_file_paths(note.content):
                path = Path(raw_path)
                resolved = path if path.is_absolute() else root / path
                try:
                    stat = resolved.stat()
                except OSError:
                    continue

                # mtime staleness (original signal)
                if stat.st_mtime > note.created_at:
                    reasons.append(raw_path)

                # code_hash staleness — detect if the anchored code changed
                if note.code_hash and resolved.suffix.lower() in {".py", ".c", ".h", ".go", ".rs"}:
                    try:
                        current_hash = hashlib.sha256(
                            resolved.read_bytes()
                        ).hexdigest()[:16]
                        if current_hash != note.code_hash and raw_path not in reasons:
                            reasons.append(f"{raw_path}[code_hash_changed]")
                    except OSError:
                        pass

            # Declarative anchor re-hash (TRIGGER-ENGINE §5) — independent of
            # the content-prose path extraction above; anchors are structured
            # (path, hash) pairs set explicitly at write time.
            for anchor in note.anchors:
                if not anchor or len(anchor) < 2:
                    continue
                anchor_path, anchor_hash = anchor[0], anchor[1]
                if not anchor_path or not anchor_hash:
                    continue  # no baseline recorded at write time — nothing to compare
                current_hash = _hash_path_content(root, anchor_path)
                if current_hash is not None and current_hash != anchor_hash:
                    reasons.append(f"{anchor_path}[anchor_changed]")
                    drift_pairs.append((anchor_path, current_hash))

            if drift_pairs:
                anchor_drift_sigs[note.note_id] = hashlib.sha256(
                    ";".join(f"{p}:{h}" for p, h in sorted(drift_pairs)).encode()
                ).hexdigest()[:16]

            if reasons:
                stale[note.note_id] = reasons

        # Declarative symbol-anchor re-hash (TRIGGER-ENGINE wave 2b, S
        # primitive) — independent of the content-prose/anchor checks above.
        # A note whose trigger declared `symbol` records that symbol's
        # signature hash at write time (remember() populates
        # `symbol_triggers`); here we re-hash the CURRENT canonical
        # definition and surface a caveat on drift, same shape as the
        # path-anchor check above: never a silent drop, just a visible
        # `f"{symbol_name}[symbol_changed]"` string appended to `stale`.
        # Degrades to a no-op when no symbol resolver is attached
        # (memory-only daemons, warm-up windows) — never an error.
        if self._symbol_resolver is not None and notes:
            note_ids = [n.note_id for n in notes]
            with self._conn() as conn:
                sym_rows = conn.execute(
                    "SELECT note_id, symbol_name, signature_hash FROM symbol_triggers "
                    "WHERE workspace = ? AND note_id IN ({})".format(
                        ",".join("?" * len(note_ids))
                    ),
                    [workspace_root] + note_ids,
                ).fetchall()
            hash_cache: dict[str, str | None] = {}
            for note_id, symbol_name, written_hash in sym_rows:
                if not written_hash:
                    continue  # no baseline recorded at write time — nothing to compare
                if symbol_name not in hash_cache:
                    hash_cache[symbol_name] = self._symbol_resolver.signature_hash(
                        workspace_root, symbol_name
                    )
                current_hash = hash_cache[symbol_name]
                if current_hash is not None and current_hash != written_hash:
                    caveat = f"{symbol_name}[symbol_changed]"
                    reasons_for_note = stale.setdefault(note_id, [])
                    if caveat not in reasons_for_note:
                        reasons_for_note.append(caveat)

        self._flag_stale_anchors(workspace_root, anchor_drift_sigs)
        return stale

    def _flag_stale_anchors(self, workspace: str, drift_sigs: dict[int, str]) -> None:
        """Idempotent write side-effect of `check_staleness()`
        (UPG-MEMORY-STATE-MACHINE §4.1/§4.4): append one `stale_flagged`
        event (`actor='system'` — the one deterministic, non-judgment
        transition) for each note_id in `drift_sigs` whose declared anchor
        is currently drifted from its content hash at write time, keyed by
        `drift_sigs[note_id]` — a sha256[:16] signature of that note's
        drifted anchors (sorted `path:current_hash` pairs; see
        `check_staleness()`). The anchors mechanism is already generic (any
        workspace-relative file path); a §4.4 proxy anchor (a lockfile, CI
        config, Dockerfile, etc. anchored the same way as any code file to
        stand in for "the process it encodes") drifts through this exact
        same path — no separate mechanism.

        Signature-keyed, not merely edge-triggered: a note is skipped only
        when its MOST RECENT `stale_flagged` event's `payload` already
        equals its current drift signature — not merely when the most
        recent event of ANY kind is `stale_flagged`. This distinction
        matters in two directions:
          - An intervening lifecycle event (revoke/reinstate/promote) while
            the anchor stays drifted in the exact same way never admits a
            duplicate audit row — the comparison looks past it to the last
            `stale_flagged` row's payload, which is unchanged.
          - A genuinely NEW drift of the same anchor (the file changed
            again, to different content) always writes a fresh event,
            because its signature differs from whatever the last
            `stale_flagged` row recorded — it is never permanently
            suppressed by an old, now-stale signature.
        `check_staleness()` runs on every recall()/fire() call while a
        drift persists, and writing a fresh event on every one of those
        calls (or on every unrelated lifecycle event) would flood the log
        with duplicate history for a fact that hasn't actually changed
        since the last flag; comparing signatures rather than "is the
        latest event stale_flagged" is what keeps that idempotent without
        also silently swallowing a real re-drift.

        This is what "automatic, reversible" still means for staleness (as
        distinct from revoked/reinstated, whose reversal is always one
        explicit event): a fixed anchor needs no "un-stale" event at all —
        once it's fixed, `check_staleness()` simply stops passing that
        note_id in `drift_sigs` on the next call, so nothing is written and
        nothing here needs to notice the recovery.

        Pre-migration compatibility: `stale_flagged` rows written before
        this signature scheme existed carry `payload = NULL`. `None` never
        equals a signature string, so the first check after upgrade always
        appends one fresh event carrying the current signature — a single,
        correct one-time "resync" row, not a bug — and steady state (no
        further rows while the drift is unchanged) resumes immediately
        after.
        """
        if not drift_sigs:
            return
        note_ids = list(drift_sigs.keys())
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT ne.note_id, ne.payload FROM note_events ne
                   INNER JOIN (
                       SELECT note_id, MAX(id) AS max_id FROM note_events
                       WHERE workspace = ? AND note_id IN ({}) AND event = 'stale_flagged'
                       GROUP BY note_id
                   ) latest ON ne.note_id = latest.note_id AND ne.id = latest.max_id""".format(
                    ",".join("?" * len(note_ids))
                ),
                [workspace] + note_ids,
            ).fetchall()
            latest_sig_by_id = {r["note_id"]: r["payload"] for r in rows}
            now = time.time()
            for nid, sig in drift_sigs.items():
                if latest_sig_by_id.get(nid) == sig:
                    continue
                _append_event(conn, workspace, nid, "stale_flagged", actor="system", payload=sig, ts=now)

    def _record_observation(
        self, workspace: str, session_id: str | None, path_candidates: "tuple[str, ...] | None"
    ) -> None:
        """UPG-ANCHOR-UNOBSERVED-BINDING: record that this session's own
        PreToolUse hook call reported `path_candidates` (the SAME
        as-given-plus-workspace-relative tuple `fire()` just computed via
        `_path_trigger_candidates()` for its own P-primitive matching, never
        recomputed or re-derived here) — the caller (the editor's PreToolUse
        hook, forwarding whatever `tool_input.file_path` it already resolved
        for a Read/Edit/Write/MultiEdit/NotebookEdit call) is the entire
        source of this fact; nothing here parses prompt or note content.

        A no-op when `session_id` or `path_candidates` is None/empty — a
        stdio transport or a caller that never supplies a session_id simply
        never gets a ledger entry, which is exactly the "no record exists"
        state `_anchor_observed_at_write()` reads back as unknown, not as a
        false negative.

        Same LRU-bounded, delete-then-reinsert-to-touch eviction shape as
        `VectrService._ledger_for`/`_turn_ledger_for` (app/service.py) — see
        `self._observed_paths`'s own comment in `__init__` for why the bound
        lives on this dict rather than being unbounded."""
        if session_id is None or not path_candidates:
            return
        from agent.config import EVICTION_MAX_TRACKED_SESSIONS

        key = (workspace, session_id)
        observed = self._observed_paths.get(key)
        if observed is None:
            if len(self._observed_paths) >= EVICTION_MAX_TRACKED_SESSIONS:
                oldest_key = next(iter(self._observed_paths))
                del self._observed_paths[oldest_key]
            observed = set()
        else:
            del self._observed_paths[key]
        observed.update(path_candidates)
        self._observed_paths[key] = observed

    def _anchor_observed_at_write(
        self, workspace: str, session_id: str | None, anchor_path: str
    ) -> bool | None:
        """UPG-ANCHOR-UNOBSERVED-BINDING: three-valued observation verdict
        for one declared anchor path, read by `remember()` at write time —
        True (this session's own PreToolUse traffic reported this exact
        path, in either candidate form), False (this session HAS a ledger
        entry — at least one path was observed — but this particular path is
        not in it), or None/unknown (no ledger entry exists for this
        session at all: no session_id, hooks never wired, stdio transport,
        team-mode bind, or a daemon restart lost the in-memory ledger since
        this session started).

        The None branch is the load-bearing one (THE CRITICAL CONSTRAINT):
        an absent ledger is NOT evidence of non-observation, so this
        deliberately returns None rather than False whenever the ledger key
        itself is missing — only a PRESENT ledger with a PRESENT-but-
        non-matching entry ever earns a real `False`. Never widen the first
        branch below into `if session_id is None or key not in ...: return
        False` — that would collapse the required three states into two and
        manufacture a false "never observed" accusation on every session
        that simply has no ledger, which is worse than saying nothing."""
        if session_id is None:
            return None
        observed = self._observed_paths.get((workspace, session_id))
        if observed is None:
            return None
        candidates = _path_trigger_candidates(workspace, anchor_path)
        if not candidates:
            return False
        return any(c in observed for c in candidates)

    def fire(
        self,
        workspace: str,
        *,
        event: str | None = None,
        file_path: str | None = None,
        command: str | None = None,
        query: str | None = None,
        ledger: "TriggerFireLedger | None" = None,
        now: float | None = None,
        session_id: str | None = None,
        record_fires: bool = True,
    ) -> list["FireResult"]:
        """Live evaluation entry point (TRIGGER-ENGINE, bm2-design-skeleton.md
        §2/§4): evaluate every non-tombstoned note in `workspace` against one
        lifecycle moment (`event` and/or `file_path` at time `now`), fold in
        `check_staleness()`'s anchor/file caveats, and return only the notes
        that fired — ordered by the single shared `total_order_key` (fire
        precedence == injection ordering == budget eviction order, per the
        design doc's "one total order").

        `ledger`, if given (a per-session `TriggerFireLedger` — see
        `VectrService`'s per-session registry, mirroring its existing
        per-session `EvictionAdvisor` pattern), suppresses a note whose
        matched trigger index already fired this session on that SAME axis
        (the `eligible()` check below — this ALWAYS runs when `ledger` is
        given, regardless of `record_fires`). Passing no ledger evaluates
        statelessly with no suppression (used by tests and any one-shot
        caller that manages its own dedup).

        `record_fires` (wave 3, §5.3 — closes the "session ledger burns a
        trigger on a non-delivery" finding), when True (the default),
        records a fresh match into `ledger` before returning — the original,
        still-correct contract for any caller whose every returned
        `FireResult` IS the delivery (no further
        packing/dedup happens downstream). `fire_and_format()` below passes
        `record_fires=False` instead: it calls `fire()` once per event in an
        OR list and then still has to survive turn-ledger cross-surface
        dedup AND budget packing before a note is actually injected — a note
        that matches here but gets turn-deduped or budget-evicted there must
        NOT be recorded as session-fired, or that axis is wrongly suppressed
        for the rest of the session even though nothing was ever delivered.
        `fire_and_format()` performs the real `record_fire()` call itself,
        once per PACKED (i.e. actually-injected) note, after packing
        decides survivors — see its own docstring.

        `session_id` enforces scope="session" notes and, combined with a
        current-branch lookup computed ONCE per call, scope="branch" notes
        (TRIGGER-ENGINE wave 2a, §1) — full scope enforcement, unlike
        recall()/recall_for_path()'s partial enforcement (see the
        SCOPE_VALUES comment in _types.py for the split and why).

        `query`, when given (the prompt-submit text — the same string
        already threaded as `recall(query=...)`'s semantic search input),
        drives the M (semantic) primitive: ONE activity embedding is
        computed for it here — never per note, and never inside
        trigger_engine.py itself (no-query-heuristics rule: this is the
        only place raw prompt text is ever touched by the trigger engine's
        wiring) — then compared by cosine against each candidate note's own
        already-stored vector, never re-embedding the note.

        `file_path` is normalized into its P-primitive candidate forms
        (`_path_trigger_candidates()`: as-given plus workspace-relative)
        exactly once here — the only place that knows `workspace` is a
        filesystem root — and that tuple, not the raw string, is what gets
        passed into `evaluate_note()`; `trigger_engine.py` never resolves a
        path itself (purity invariant).

        `command` (wave 3, UPG-MEMORY-STATE-MACHINE §5.2), when given, is a
        raw Bash command about to run (`PreToolUse`'s `Bash` matcher lane) —
        normalized into its VERB exactly once here via
        `app.cmdnorm.normalize_command()` (the single shared normalizer, also
        used by `app/arcs.py`'s arc detection; never re-implemented) and
        passed through as `command_verb` for the 'command' trigger primitive.
        Classifying the argv of a tool call the caller is ABOUT to run is
        tool-call structure, not prompt/query content (no-query-heuristics
        rule's sanctioned carve-out — see agent/trigger_engine.py's module
        docstring and the design doc's §6 compliance ledger).

        Every note that actually fires has its `last_fired` column stamped to
        `now` — this is what makes a trigger's `cooldown` T-modifier
        meaningful on the NEXT evaluation (evaluate_note() reads `last_fired`
        directly off the note). A note whose winning trigger declared the M
        (semantic) axis is additionally gated by `ledger`'s per-note
        session-turn cooldown (§5.5) rather than the plain forever-this-
        session suppression every other axis gets — see
        `TriggerFireLedger.eligible()`'s docstring."""
        from agent.config import (
            MEMORY_TRIGGER_SEMANTIC_COOLDOWN_TURNS,
            MEMORY_TRIGGER_SEMANTIC_THETA_BY_KIND,
        )
        from agent.trigger_engine import effective_triggers, evaluate_note, total_order_key

        if now is None:
            now = time.time()

        command_verb: str | None = None
        if command:
            from app.cmdnorm import normalize_command
            command_verb = normalize_command(command).verb

        branch = _current_git_branch(Path(workspace))
        path_candidates = _path_trigger_candidates(workspace, file_path)
        # UPG-ANCHOR-UNOBSERVED-BINDING: this IS the read signal — every
        # PreToolUse call with a file_path (Read/Edit/Write/MultiEdit/
        # NotebookEdit; app/service.py's `_recall_impl` file_path branch
        # always reaches here via fire_and_format()) reports the caller's
        # own already-resolved path, unconditionally, regardless of whether
        # any note ends up matching it below — a later remember() in this
        # same session can then mark an anchor to this path observed=true.
        self._record_observation(workspace, session_id, path_candidates)

        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM notes WHERE workspace = ? AND valid_until IS NULL",
                (workspace,),
            ).fetchall()
        notes = [self._row_to_note(row) for row in rows]
        # UPG-MEMORY-DECAY-KIND-SCOPED: an expired note never fires (same
        # exclusion recall() applies) — filtered before check_staleness() so
        # no wasted anchor-hash work happens on a note that can't fire
        # anyway.
        notes = self._exclude_expired(workspace, notes)
        notes_by_id = {n.note_id: n for n in notes}

        stale = self.check_staleness(notes, workspace)

        # S (symbol) primitive (TRIGGER-ENGINE wave 2b): resolve the target
        # file's defined/referenced symbols ONCE per call, never per note —
        # and only when there is a `file_path` to resolve against, a symbol
        # resolver is attached, AND this workspace has at least one
        # symbol-triggered note (a cheap existence check against the
        # write-time `symbol_triggers` index, avoiding a live graph query
        # for the common case of no symbol triggers at all). Degrades to
        # `None` — meaning "no symbol matches this call" — whenever any of
        # those conditions is unmet (memory-only daemons and warm-up windows
        # have no resolver attached at all): a symbol trigger then
        # deterministically does not fire, never an error.
        resolved_symbols: frozenset[str] | None = None
        if file_path is not None and self._symbol_resolver is not None:
            with self._conn() as conn:
                has_symbol_trigger = conn.execute(
                    "SELECT 1 FROM symbol_triggers WHERE workspace = ? LIMIT 1",
                    (workspace,),
                ).fetchone()
            if has_symbol_trigger is not None:
                resolved_symbols = self._symbol_resolver.symbols_touching_file(
                    workspace, file_path
                )

        # M (semantic) primitive (TRIGGER-ENGINE wave 2b, §8): embed the
        # prompt ONCE per call (never per note), only when at least one note
        # actually declares a semantic axis and the embedder has attached
        # (degrades to "no match" during warm-up or in a memory-only
        # daemon — never an error). Per-note cosine against each note's own
        # already-stored vector (reused, never re-embedded) is gated by a
        # fixed per-kind theta — no runtime adaptation, no query parsing.
        semantic_matched_by_id: dict[int, bool] = {}
        # `effective_triggers()` resolves each note's ACTUAL evaluated bundle
        # (explicit triggers[], or its kind's default bundle when none are
        # declared — same replace-not-merge rule evaluate_note() itself
        # applies) rather than reading `n.triggers` directly: a note relying
        # on an implicit default bundle with a semantic axis (e.g.
        # kind="operational", §5.1) would otherwise never have its vector
        # fetched here, so `evaluate_note()` would always see
        # `semantic_matched=None` for it and the trigger could never fire —
        # a real bug caught while wiring operational's default bundle.
        notes_wanting_semantic = [
            n for n in notes if any(t.get("semantic") for t in effective_triggers(n))
        ]
        if (
            query
            and notes_wanting_semantic
            and self._notes_col is not None
            and self._embed_query_fn is not None
        ):
            try:
                activity_vector = self._embed_query_fn([query])[0]
                with timed_chroma_call("get"):
                    fetched = self._notes_col.get(
                        ids=[str(n.note_id) for n in notes_wanting_semantic],
                        include=["embeddings"],
                    )
                vector_by_id = dict(zip(fetched["ids"], fetched["embeddings"]))
            except Exception:
                activity_vector, vector_by_id = None, {}
            if activity_vector is not None:
                for n in notes_wanting_semantic:
                    vec = vector_by_id.get(str(n.note_id))
                    theta = MEMORY_TRIGGER_SEMANTIC_THETA_BY_KIND.get(n.kind)
                    if vec is None or theta is None:
                        continue
                    similarity = _cosine_similarity(activity_vector, vec)
                    # bool(...): a note's stored vector comes back from Chroma
                    # as a numpy array, so `similarity` is a numpy.bool_, not a
                    # plain Python bool — and `numpy.True_ is True` is False
                    # (numpy scalars are never the same object as the builtin
                    # singleton). _trigger_matches()'s gate is a strict `is
                    # True` check (by design — it must tell "matched" apart
                    # from "not evaluated" (`None`) using identity, not
                    # truthiness), so an un-coerced numpy bool would silently
                    # never fire a real semantic trigger.
                    semantic_matched_by_id[n.note_id] = bool(similarity >= theta)

        fired_ids: list[int] = []
        results = []
        for note in notes:
            result = evaluate_note(
                note, event=event, file_path=path_candidates, now=now,
                session_id=session_id, branch=branch,
                resolved_symbols=resolved_symbols,
                semantic_matched=semantic_matched_by_id.get(note.note_id),
                command_verb=command_verb,
            )
            if not result.fired:
                continue
            if ledger is not None and result.trigger_index is not None:
                cooldown_turns = (
                    MEMORY_TRIGGER_SEMANTIC_COOLDOWN_TURNS if result.semantic else None
                )
                if not ledger.eligible(
                    note.note_id, result.trigger_index,
                    semantic=result.semantic, cooldown_turns=cooldown_turns,
                ):
                    continue
            result.stale_paths = stale.get(note.note_id, [])
            results.append(result)
            fired_ids.append(note.note_id)
            if record_fires and ledger is not None and result.trigger_index is not None:
                ledger.record_fire(note.note_id, result.trigger_index, semantic=result.semantic)

        results.sort(key=lambda r: total_order_key(notes_by_id[r.note_id]))

        if fired_ids:
            with self._conn() as conn:
                conn.execute(
                    "UPDATE notes SET last_fired = ? WHERE note_id IN ({})".format(
                        ",".join("?" * len(fired_ids))
                    ),
                    [now] + fired_ids,
                )

        audit(
            "TRIGGER_FIRE", workspace=workspace, trigger_event=event or "",
            file_path=file_path or "", command=command_verb or "", fired=len(results),
        )
        return results

    def fire_and_format(
        self,
        workspace: str,
        *,
        events: list[str] | None = None,
        event: str | None = None,
        file_path: str | None = None,
        command: str | None = None,
        query: str | None = None,
        session_id: str | None = None,
        ledger: "TriggerFireLedger | None" = None,
        turn_ledger: "TurnInjectionLedger | None" = None,
        spend_turn_budget: bool = False,
        surface: str = "mcp",
        now: float | None = None,
    ) -> tuple[str, set[int]]:
        """Live hook-delivery entry point (TRIGGER-ENGINE wave 2a,
        bm2-design-skeleton.md §2/§3/§4): call `fire()` for one or more
        lifecycle events, merge + dedup the results, render them through the
        §3 two-tier budget pack, and return `(rendered_text, note_ids)`.

        `events`, if given, is evaluated as an OR across the whole list —
        e.g. `["session-start", "post-compaction"]` right after a `/compact`,
        so a directive whose default bundle fires on EITHER axis is rendered
        exactly once (first-seen wins across the merged event list) rather
        than twice. `event` (singular) is a convenience for the common
        one-event case; when both are omitted, `fire()` is called once with
        `event=None` (matches its own default, e.g. a plain PreToolUse
        file_path-only check with no event name).

        `command` (wave 3, §5.2) is forwarded to every `fire()` call
        unchanged — see `fire()`'s own docstring for the normalization
        contract.

        `query` (TRIGGER-ENGINE wave 2b, §8) is forwarded to every `fire()`
        call unchanged — the M primitive's ONE activity embedding is
        computed inside `fire()` itself, not duplicated per event in this
        loop's OR list (a caller with more than one event in `events` still
        only pays for one embed call per `fire()` invocation, one per event
        as today, never one per note).

        The returned `note_ids` set lets a caller merging this with a
        legacy/unrelated recall path (e.g. `recall_for_path`'s content-
        substring match) exclude notes already delivered here, preventing
        the SAME note from being injected twice through two different
        mechanisms (double-injection prevention, wave 2a deliverable).

        `ledger`, if given, also makes the per-session injection budget
        CUMULATIVE across every call in one session (§3): the budget spent
        by earlier deliveries this session is subtracted before packing this
        one, and this delivery's own spend is recorded back into the ledger
        before returning. Every `fire()` call below passes
        `record_fires=False` — this method, not `fire()` itself, is
        responsible for calling `ledger.record_fire()`, and does so ONLY for
        a note that survives all the way to `packed` (§5.3: "session ledger
        burns a trigger on a non-delivery" fix). A note that matches here
        but is then dropped by the turn-ledger dedup below or evicted by
        `pack_injection()`'s budget must NOT be recorded as session-fired —
        recording an unfired match would permanently suppress that axis for
        the rest of the session even though nothing was ever actually
        injected. `fire()`'s own `ledger.eligible()` gate (unaffected by
        this) still runs on every call, so a note already recorded as fired
        by an EARLIER, successfully-packed `fire_and_format()` call this
        session is correctly excluded here too.

        `turn_ledger` (wave 3, §5.3/§5.4 — closes the arm-C double-dip
        finding), if given, is consulted TWICE: (1) after merging `seen`
        across the OR event list, any note_id another surface already
        claimed THIS TURN (`turn_ledger.eligible()` is False) is dropped
        before it is even considered for packing — never delivered twice in
        one turn, regardless of how many separate surfaces matched it; (2)
        after `pack_injection()` decides which notes actually survive the
        budget, every SURVIVING note_id is claimed
        (`turn_ledger.claim()`) — a note that gets evicted for budget
        reasons is deliberately NOT claimed, so it remains eligible for a
        later surface this same turn to still deliver it (never silently
        dropped for the whole turn just because an earlier, budget-
        exhausted surface merely matched it).

        `spend_turn_budget`, when True, additionally caps this call's
        packing budget by `turn_ledger.remaining_turn_budget()` (the ≤500
        ordinary-turn allowance shared by every per-turn surface combined,
        §5.4) and records this delivery's spend back into it. Session-
        start's own bulk boot delivery passes False (the default) — it
        keeps its existing separate `ledger`-cumulative per-SESSION cap
        only; the turn-ledger CLAIM above still runs for it regardless, so
        a note delivered at boot is still correctly excluded from a
        same-turn PreToolUse/prompt-submit re-delivery, it is just never
        counted against the smaller per-turn allowance."""
        from agent.trigger_engine import pack_injection, token_estimate, total_order_key

        event_list = events if events else ([event] if event is not None else [None])
        if now is None:
            now = time.time()

        seen: dict[int, "FireResult"] = {}
        for ev in event_list:
            for r in self.fire(
                workspace, event=ev, file_path=file_path, command=command, query=query,
                session_id=session_id, ledger=ledger, now=now, record_fires=False,
            ):
                if r.note_id not in seen:
                    seen[r.note_id] = r

        if turn_ledger is not None:
            seen = {nid: r for nid, r in seen.items() if turn_ledger.eligible(nid)}

        if not seen:
            return "", set()

        notes_by_id: dict[int, WorkingNote] = {}
        for note_id in seen:
            note = self.get_note(workspace, note_id)
            if note is not None:
                notes_by_id[note_id] = note

        ordered_ids = sorted(notes_by_id, key=lambda nid: total_order_key(notes_by_id[nid]))
        stale_by_id = {nid: r.stale_paths for nid, r in seen.items() if r.stale_paths}
        # UPG-MEMORY-STATE-MACHINE §4.1/§4.3: fold note_events for every note
        # about to be rendered so a revoked one gets the anti-memory
        # deterrent block below instead of its raw content — computed here
        # (not threaded in from a caller) since fire_and_format() already
        # has both `workspace` and the exact note set that's about to render.
        note_states = self.note_event_states(workspace, list(notes_by_id.values()))

        items = []
        for note_id in ordered_ids:
            note = notes_by_id[note_id]
            full_text = _format_full_block(note, stale_by_id, note_states, injected=True)
            index_text = _format_index_line(note, stale_by_id, surface=surface, note_states=note_states)
            items.append((note, full_text, index_text))

        budget = ledger.remaining_budget() if ledger is not None else None
        if spend_turn_budget and turn_ledger is not None:
            turn_budget = turn_ledger.remaining_turn_budget()
            budget = turn_budget if budget is None else min(budget, turn_budget)
        packed = pack_injection(items, budget=budget)
        if not packed:
            return "", set()
        # §5.3 fix: record each PACKED note's fire into the session ledger
        # here — the only point in this method that knows which notes
        # actually survived turn-dedup AND budget packing to reach real
        # delivery. `fire()` itself was called with `record_fires=False`
        # above specifically so a note that matched but was then dropped
        # (turn-deduped or budget-evicted) never burns its axis's
        # session-wide eligibility for nothing.
        if ledger is not None:
            for p in packed:
                result = seen.get(p.note_id)
                if result is not None and result.trigger_index is not None:
                    ledger.record_fire(p.note_id, result.trigger_index, semantic=result.semantic)
            spent = sum(token_estimate(p.text) for p in packed)
            ledger.record_spend(spent)
        if spend_turn_budget and turn_ledger is not None:
            turn_ledger.record_turn_spend(sum(token_estimate(p.text) for p in packed))
        if turn_ledger is not None:
            for p in packed:
                turn_ledger.claim(p.note_id)

        header = f"# Triggered Memory ({len(packed)} fired)\n"
        text = header + "\n\n".join(p.text for p in packed)
        return text, {p.note_id for p in packed}

    def get_author_trust(self, workspace: str, author_id: str) -> float:
        """Return the Bayesian trust score for an author in this workspace."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT trust_score FROM author_trust WHERE workspace = ? AND author_id = ?",
                (workspace, author_id),
            ).fetchone()
        return row[0] if row else 1.0

    def list_authors(self, workspace: str) -> list[dict]:
        """Return all authors with their trust scores and note counts."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT author_id, trust_score, note_count FROM author_trust WHERE workspace = ? ORDER BY trust_score DESC",
                (workspace,),
            ).fetchall()
        return [{"author_id": r[0], "trust_score": r[1], "note_count": r[2]} for r in rows]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _row_to_note(self, row: sqlite3.Row) -> WorkingNote:
        content = row["content"]
        keys = row.keys()
        title = row["title"] if "title" in keys else ""
        user_quote = row["user_quote"] if "user_quote" in keys else ""
        if self._encryptor:
            content = self._encryptor.decrypt(content)
            # Tolerant decrypt: titles written before title-encryption (or before
            # encryption was enabled at all) are returned unchanged.
            title = self._encryptor.decrypt(title)
            if user_quote:
                user_quote = self._encryptor.decrypt(user_quote)
        return WorkingNote(
            note_id=row["note_id"],
            workspace=row["workspace"],
            content=content,
            tags=json.loads(row["tags"]),
            priority=row["priority"],
            created_at=row["created_at"],
            last_accessed=row["last_accessed"],
            session_id=row["session_id"],
            decay_score=row["decay_score"],
            kind=row["kind"] if "kind" in keys else DEFAULT_KIND,
            # team notes fields (present in all new DBs; guarded for old DBs without migration)
            author_id=row["author_id"] if "author_id" in keys else "",
            author_trust_score=row["author_trust_score"] if "author_trust_score" in keys else 1.0,
            valid_from=row["valid_from"] if "valid_from" in keys else 0.0,
            valid_until=row["valid_until"] if "valid_until" in keys else None,
            code_hash=row["code_hash"] if "code_hash" in keys else "",
            superseded_by=row["superseded_by"] if "superseded_by" in keys else None,
            superseded_at=row["superseded_at"] if "superseded_at" in keys else None,
            title=title,
            # TRIGGER-ENGINE wave 1 fields — guarded for pre-migration DBs.
            triggers=json.loads(row["triggers"]) if "triggers" in keys and row["triggers"] else [],
            provenance=row["provenance"] if "provenance" in keys and row["provenance"] else DEFAULT_PROVENANCE,
            scope=row["scope"] if "scope" in keys and row["scope"] else DEFAULT_SCOPE,
            anchors=json.loads(row["anchors"]) if "anchors" in keys and row["anchors"] else [],
            supersedes=row["supersedes"] if "supersedes" in keys else None,
            superseded_by_note_id=row["superseded_by_note_id"] if "superseded_by_note_id" in keys else None,
            last_fired=row["last_fired"] if "last_fired" in keys else None,
            branch=row["branch"] if "branch" in keys and row["branch"] else "",
            # UPG-MEM-PROVENANCE-USER-STATED — guarded for pre-migration DBs.
            user_quote=user_quote or "",
            # UPG-RECALL-MISS-FLOOR part (b) — guarded for pre-migration DBs.
            pinned=bool(row["pinned"]) if "pinned" in keys and row["pinned"] is not None else False,
        )

    def format_notes_for_llm(
        self,
        notes: list[WorkingNote],
        stale_warnings: dict[int, list[str]] | None = None,
        detail: str = "index",
        surface: str = "mcp",
        sort_by: str = "relevance",
    ) -> str:
        """Format recalled notes into a clean LLM-readable string.

        detail='index' (default, UPG-RECALL-HIERARCHY): renders ONE crisp line per note:
            [#<note_id>] <kind>/<priority> · <title>  (<relative age>)
          No body is included. Token-bounded for hook injection and default recall.
          When the note carries a caller-declared agent/subagent identifier
          (`author_id`, set via vectr_remember's optional `agent` argument —
          UPG-SUBAGENT-MEMORY), it renders as an attribution tag right after
          priority: `[#12] task/high (coder-2) · title  (2h)`. Never inferred;
          a note with no `agent` renders exactly as before this feature shipped.

          sort_by='chronological' (UPG-DECISION-TIMELINE) — passed through
          unchanged from the same-named `recall()` argument that produced
          `notes` — replaces the trailing relative age with the note's
          creation date:
            [#<note_id>] <YYYY-MM-DD> <kind>/<priority> · <title>
          so the index reads as a dated timeline (e.g.
          `vectr_recall(kind="decision", sort_by="chronological")` for an
          ADR-style decision history). Every other `sort_by` value (the
          default 'relevance', plus 'recency'/'priority') renders exactly as
          before this parameter existed — this only changes the rendering,
          the caller is still responsible for having asked `recall()` to
          actually sort chronologically.

        detail='full': renders the full body format (pre-existing behaviour).
          Use for explicit vectr_recall(detail='full') or single-note expand (note_id path).

        If stale_warnings is provided (full detail only), notes whose referenced files
        have changed are flagged with a [STALE] marker and a warning.

        surface='mcp' (default): the expand hint uses the MCP tool-call form
        (`vectr_recall(note_id=N)`) — correct for the MCP dispatch path, whose
        caller is an editor's LLM. surface='cli': the expand hint uses the
        actual shell form (`vectr recall --id N`) — used by the REST route
        `vectr recall`/`vectr remember` go through, whose caller is a human
        terminal (UPG-CLI-RECALL-HINT: MCP tool syntax is meaningless there).

        surface also controls how each note's id is rendered in the index
        listing (UPG-CLI-RECALL-ID-FOOTGUN): 'mcp' keeps `[#N]` — the `#` is
        harmless there since the editor's LLM never pastes it into a shell.
        'cli' renders the bare `[N]` instead — a terminal user who copies
        `[#125]` into `vectr recall #125` hits zsh's `interactive_comments`,
        which silently strips `#125` as a comment and leaves a bare
        `vectr recall` (a semantic-query no-op) with no error. The 'cli'
        header hint also names a real id from the current results
        (`vectr recall --id <that id>`) rather than a generic placeholder,
        so it is directly copy-pasteable.
        """
        if not notes:
            return "No working notes found."

        stale_warnings = stale_warnings or {}
        # UPG-MEMORY-STATE-MACHINE §4.1/§4.3: fold note_events for every note
        # about to render, so a revoked one carries `[REVOKED]`/the
        # anti-memory deterrent block on every surface that recalls notes —
        # computed here rather than threaded in from a caller (`notes[0]
        # .workspace` — every note passed into one format call already
        # shares a workspace, same assumption `check_staleness()`'s callers
        # already make).
        note_states = self.note_event_states(notes[0].workspace, notes)

        if detail == "index":
            if surface == "mcp":
                expand_hint = "use vectr_recall(note_id=N) to expand"
            else:
                expand_hint = f"run `vectr recall --id {notes[0].note_id}` to expand"
            header = f"# Working Notes — index ({len(notes)} entries; {expand_hint})\n"
            lines = [header]
            for n in notes:
                lines.append(_format_index_line(n, stale_warnings, surface=surface, sort_by=sort_by, note_states=note_states))
            return "\n".join(lines)

        # detail == "full" (original behaviour, with stale warnings)
        stale_count = len(stale_warnings)
        header = f"# Working Notes ({len(notes)} entries"
        if stale_count:
            header += f", {stale_count} may be stale"
        header += ")\n"

        lines = [header]
        for n in notes:
            lines.append(_format_full_block(n, stale_warnings, note_states))
        return "\n".join(lines)

    def related_active_notes(
        self,
        workspace: str,
        note_id: int,
        *,
        limit: int,
        min_similarity: float,
    ) -> list[RelatedNote]:
        """Write-time related-notes lookup — the `limit` nearest EXISTING,
        currently-active notes to `note_id`, above `min_similarity`.
        Thin delegate to `agent.working_context_store._related` (see that
        module's docstring for the full contract: additive-only, never
        asserts a contradiction, never mutates anything, and never raises).
        """
        return _related_active_notes(
            self, workspace, note_id, limit=limit, min_similarity=min_similarity,
        )

    def revoked_related_notes(
        self,
        workspace: str,
        note_id: int,
        *,
        limit: int,
        min_similarity: float,
    ) -> list[RevokedRelatedNote]:
        """Write-time deterrent lookup (UPG-RELATED-REVOKED-DETERRENT) — the
        `limit` nearest EXISTING notes to `note_id` whose folded lifecycle
        state is `revoked`, above `min_similarity`. Thin delegate to
        `agent.working_context_store._related` (see that module's docstring
        for the full contract and why this is a separate function from
        `related_active_notes` rather than a filter change to it).
        """
        return _revoked_related_notes(
            self, workspace, note_id, limit=limit, min_similarity=min_similarity,
        )
