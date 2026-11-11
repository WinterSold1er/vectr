"""Matching engine for proactive context (UPG-PRO-4).

Given a `ProactiveWindow`, run every enabled matcher UNCONDITIONALLY and return
the union of scored `Candidate`s. No matcher branches on window *content* to
decide whether to run — which matchers are active is a static config toggle,
never a runtime read of the conversation (the no-query-heuristics hard rule).

The matchers do not implement retrieval themselves; they call a `MatchSource`,
which the daemon backs with the existing recall / path-anchor / search paths.
The honest shipped subset is:
  M1 structural_note  — exact file-path -> anchored/mentioning note (tiered score)
  M3 semantic_note    — cosine note recall above the similarity floor
  M4 code_semantic    — hybrid code search (opt-in; needs a built index)
M2 (symbol-definition via locate) is defined by the design but deferred here;
adding it is another matcher + threshold, not a content classifier.

Every rendered note line also carries a provenance/trust marker (UPG-PROXY-
INJECT-ROLE-PROVENANCE, `_provenance_label()` below) so a note that escapes
the outer envelope (agent/proactive/gate.py) — truncation, a model that skims
— still self-describes as recalled memory of a given trust class rather than
reading as an instruction. `ProactiveMatcher` can additionally be told to
drop `kind == "directive"` notes entirely before they ever become a
`Candidate` (`exclude_directive_notes`) — see that flag's own docstring for
why this belongs here rather than in the gate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from agent.config import PROACTIVE_BODY_TRUNCATION_MIN_BOUNDARY_FRACTION
from agent.proactive.types import (
    CANDIDATE_STATE_NOT_APPLICABLE,
    STRUCTURAL_TIER_DECLARED_ANCHOR,
    STRUCTURAL_TIER_DECLARED_TRIGGER,
    STRUCTURAL_TIER_GOTCHA_MENTION,
    STRUCTURAL_TIER_MENTION,
    Candidate,
    ProactiveWindow,
)
from agent.trigger_engine import path_trigger_match
from agent.working_context_store._store import (
    _ANTI_MEMORY_TEMPLATE,
    _date_str,
    _note_title,
    _path_boundary_count,
    _path_boundary_first,
    _path_boundary_match,
)
from agent.working_context_store._types import PROVENANCE_VALUES, WorkingNote


@runtime_checkable
class MatchSource(Protocol):
    """Retrieval surface the matchers consume. The daemon implements this over
    its in-process store/searcher; tests implement it with real return types."""

    def structural_notes(self, file_paths: list[str]) -> list[WorkingNote]:
        ...

    def semantic_notes(
        self, text: str, min_similarity: float, limit: int
    ) -> list[tuple[WorkingNote, float]]:
        ...

    def code_search(self, text: str, n_results: int) -> list:
        ...

    def note_states(self, notes: list[WorkingNote]) -> dict[int, dict]:
        """Batched fold of note-lifecycle state for the given notes (UPG-
        MEMORY-STATE-MACHINE), keyed by note_id -> {"state": "active"|
        "superseded"|"revoked", "reason": str|None, "actor": str|None,
        "ts": float|None} — see `WorkingContextStore.note_event_states()`.
        A note_id absent from the returned dict means active. Called ONCE
        per `match()` invocation for the union of every note the other
        matchers found, never per-note."""
        ...


def _one_line(text: str) -> str:
    """Collapse to a single whitespace-normalised line (injected context is
    line-oriented, and a note body may span many lines)."""
    return " ".join(text.split())


def _cap(text: str, max_chars: int) -> str:
    """Collapse TEXT to one line and cap it to at most MAX_CHARS.

    UPG-PROXY-INJECT-TITLE-ONLY: once a candidate line carries a note's BODY
    rather than just its short title (`_raw_summary()` below), hitting this
    cap is the routine case, not the rare one — a bare character cut there
    regularly lands mid-word or mid-sentence ("...without verificat…"),
    which reads as corrupted output rather than an intentionally shortened
    summary. The cut backs off to the nearest SENTENCE boundary (". ")
    within the allowed budget when one exists there; failing that, to the
    nearest WORD boundary (" "); failing that (a single token longer than
    the whole budget), a hard character cut. A boundary is only used when
    it still keeps at least
    `PROACTIVE_BODY_TRUNCATION_MIN_BOUNDARY_FRACTION` (config.yaml
    `proactive.body_truncation_min_boundary_fraction`) of the budget —
    backing off further than that would throw away most of the content
    chasing a nicer cut point, which is worse than a hard cut. Every
    truncated result ends in an ellipsis so the caller can tell content was
    cut, exactly as before this change."""
    text = _one_line(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars == 1:
        return "…"
    budget = max_chars - 1
    window = text[:budget]
    min_boundary = max(1, int(budget * PROACTIVE_BODY_TRUNCATION_MIN_BOUNDARY_FRACTION))

    sentence_boundary = window.rfind(". ")
    if sentence_boundary >= min_boundary:
        return window[: sentence_boundary + 1] + "…"

    word_boundary = window.rfind(" ")
    if word_boundary >= min_boundary:
        return window[:word_boundary].rstrip() + "…"

    return window.rstrip() + "…"


def _raw_summary(note: WorkingNote) -> str:
    """The candidate line's raw summary text (UPG-PROXY-INJECT-TITLE-ONLY):
    the note's BODY, not just its title. Before this, a note with a title
    injected ONLY that title — its actual guidance (a caveat, a fix, a
    "do not do X" reason) lived in `content` and never crossed the wire,
    which measurement (LANE-UTILITY-2, injection_utility benchmark) showed
    produced a title-shaped, actionless injected line for every real note
    that had one.

    `title` and `content` are distinct stored fields (a short index-tier
    LABEL vs. the BODY — see `WorkingNote`'s own field comments), so this
    joins both rather than picking one:

      - no title: the body alone (unchanged from before for the common
        vectr_remember() call that never passes a title).
      - a title that is already a prefix of the (whitespace-normalised)
        body: the body alone — `remember()`'s own title FALLBACK derives a
        title from the first content line when the caller omits one, so
        this is the common case where prepending the title back would just
        duplicate the opening words verbatim.
      - otherwise: "{title}: {body}" — the label orients the reader before
        the detail.

    The combined text is NOT truncated here — `_cap()` truncates the fully
    assembled candidate line (kind/provenance/anchor prefix + this summary)
    to the caller's budget in one place, once, at a sentence/word boundary
    rather than mid-word."""
    body = _one_line(note.content or "")
    title = note.title.strip() if note.title else ""
    if not title:
        return body
    if not body or body.startswith(title):
        return body or title
    return f"{title}: {body}"


def _revoked_summary(note: WorkingNote, state: dict) -> str:
    """Deterrent rendering for a revoked note (UPG-PROXY-REVOKED-LEAK):
    reuses the shared `_ANTI_MEMORY_TEMPLATE` (agent/working_context_store/
    _store.py) rather than inventing separate deterrent copy — the same
    wording a revoked note gets on the multi-line `/v1/recall` surface
    (`_format_full_block()`), so this single-line proxy injection agrees
    with it.

    UPG-DETERRENT-TITLE-ONLY: `_ANTI_MEMORY_TEMPLATE` itself now puts the
    "Do not re-derive..." warning FIRST and the free-text `reason` ahead of
    the (already short, `_note_title()`-bounded) quoted summary, so
    `_cap()`'s right-truncation naturally sacrifices the quote before the
    reason, and the reason before the warning, with no reordering needed
    here — this function used to relocate the warning to the front at
    render time via `rpartition(". ")`; that is now redundant since the
    template's own field order already does it, uniformly across every
    surface that renders this template (not just this one)."""
    revoked_date = _date_str(state["ts"]) if state.get("ts") else _date_str(note.created_at)
    reason = state.get("reason") or "no reason given"
    return _ANTI_MEMORY_TEMPLATE.format(
        created_date=_date_str(note.created_at),
        revoked_date=revoked_date,
        reason=reason,
        summary=_note_title(note),
    )


def _note_summary(note: WorkingNote, state: dict | None = None) -> str:
    """The candidate line's summary text. A note whose folded lifecycle
    state (UPG-MEMORY-STATE-MACHINE) is "revoked" renders through the
    anti-memory deterrent instead of its raw content — a revoked note
    still injects (catching an agent about to re-assert a wrong belief is
    the highest-value injection), it just never injects as raw fact."""
    if state and state.get("state") == "revoked":
        return _revoked_summary(note, state)
    return _raw_summary(note)


def _kind_label(note: WorkingNote, state: dict | None) -> str:
    """The kind marker for the candidate line: "REVOKED" takes precedence
    over the bare kind name so the injected line clearly flags a revoked
    note rather than reading like an ordinary kind label."""
    if state and state.get("state") == "revoked":
        return "REVOKED"
    return note.kind


def _state_label(state: dict | None) -> str:
    """The folded lifecycle state to carry onto the `Candidate` for the audit
    line (UPG-PROXY-AUDIT-DURABLE). Mirrors `note_event_states()`'s own
    documented contract: a note absent from the fold (or with no fold result
    at all) is "active" — the same default every other renderer here uses."""
    if state and state.get("state"):
        return state["state"]
    return "active"


def _provenance_label(note: WorkingNote) -> str:
    """The trust/provenance marker for the injected line (UPG-PROXY-INJECT-
    ROLE-PROVENANCE): `WorkingNote.provenance` is a structural note property
    (human | agent | auto — bm2-design-skeleton.md §5), never conversation
    content. Surfacing it lets an unreviewed auto-captured note read as
    visibly weaker than an agent-recorded or human-endorsed one, even after
    the note has left the outer envelope framing (agent/proactive/gate.py).

    A missing or unrecognised value defaults to the WEAKEST label ("auto")
    rather than the strongest — this must never fabricate a "human"-endorsed
    claim for a note whose provenance field is absent or malformed."""
    value = getattr(note, "provenance", None)
    if value in PROVENANCE_VALUES:
        return value
    return "auto"


def _structural_note_candidate(
    note: WorkingNote,
    anchor: str,
    relation: str,
    max_chars: int,
    score_declared_anchor: float,
    score_declared_trigger: float,
    score_gotcha_mention: float,
    score_mention: float,
    state: dict | None = None,
    anchor_path: str | None = None,
    path_mention_count: int = 0,
    path_mention_first_offset: int = -1,
) -> Candidate:
    """`relation` is one of the STRUCTURAL_TIER_* constants `_first_anchor()`
    resolved: DECLARED_ANCHOR, DECLARED_TRIGGER, or MENTION (raw — kind
    branching below turns a mention into GOTCHA_MENTION or plain MENTION,
    exactly as before this note's kind was already the only thing that
    branch depended on).

    `path_mention_count` / `path_mention_first_offset`
    (UPG-PROXY-WEAK-TIER-TIEBREAK): how many times the matched file's
    basename occurs in this note's content at a genuine path boundary, and
    where the first one lands — carried on the Candidate so the gate can
    break ties among EQUAL-score candidates (every Tier-D item scores the
    same `structural_scores.mention`) by relevance instead of insertion
    order. Both are structural note-vs-path properties computed by the
    same predicate `_path_boundary_match()` applies; they never touch
    `score`, which would change tier competition against the semantic
    band."""
    summary = _note_summary(note, state)
    kind_label = _kind_label(note, state)
    provenance_label = _provenance_label(note)
    if relation == STRUCTURAL_TIER_DECLARED_ANCHOR:
        relation_text = "anchored to"
    elif relation == STRUCTURAL_TIER_DECLARED_TRIGGER:
        relation_text = "triggers on"
    else:
        relation_text = "mentions"
    line = _cap(
        f"note #{note.note_id} ({kind_label}, {provenance_label}, {relation_text} {anchor}): "
        f"{summary}",
        max_chars,
    )
    # UPG-PROXY-INJECT-PRECISION lever 2: a tiered score, replacing a flat
    # hardcoded 1.0. Each branch reads only a STRUCTURAL property already in
    # hand — `relation` (computed above by `_first_anchor` from `note.
    # anchors`/`note.triggers`) or `note.kind` — never window/query content.
    if relation == STRUCTURAL_TIER_DECLARED_ANCHOR:
        score = score_declared_anchor
        tier = STRUCTURAL_TIER_DECLARED_ANCHOR
    elif relation == STRUCTURAL_TIER_DECLARED_TRIGGER:
        score = score_declared_trigger
        tier = STRUCTURAL_TIER_DECLARED_TRIGGER
    elif note.kind == "gotcha":
        score = score_gotcha_mention
        tier = STRUCTURAL_TIER_GOTCHA_MENTION
    else:
        score = score_mention
        tier = STRUCTURAL_TIER_MENTION
    return Candidate(
        kind="note_structural",
        line=line,
        score=score,
        anchor_id=f"note:{note.note_id}",
        is_structural=True,
        state=_state_label(state),
        structural_tier=tier,
        # UPG-PROXY-INJECT-SINGLE-TURN: only carried for the strongest tier —
        # the gate's event-anchored retirement (agent/proactive/gate.py's
        # `select()`) only ever reads this for a
        # STRUCTURAL_TIER_DECLARED_ANCHOR candidate; populating it for a
        # weaker tier would be inert (the gate never checks it there) but
        # confusing to read, so it is left None there deliberately.
        anchor_path=anchor_path if tier == STRUCTURAL_TIER_DECLARED_ANCHOR else None,
        # UPG-PROXY-INJECT-ROLE-PROVENANCE: the same normalised label already
        # baked into `line` above, also carried as a structured field so the
        # gate can pick the whole block's envelope wording without parsing
        # rendered text.
        note_provenance=provenance_label,
        # UPG-PROXY-WEAK-TIER-TIEBREAK: the equal-score relevance tie-break
        # signals (see this function's docstring).
        path_mention_count=path_mention_count,
        path_mention_first_offset=path_mention_first_offset,
    )


def _semantic_note_candidate(
    note: WorkingNote, score: float, max_chars: int, state: dict | None = None
) -> Candidate:
    summary = _note_summary(note, state)
    kind_label = _kind_label(note, state)
    provenance_label = _provenance_label(note)
    line = _cap(
        f"note #{note.note_id} ({kind_label}, {provenance_label}): {summary}", max_chars
    )
    return Candidate(
        kind="note_semantic",
        line=line,
        score=float(score),
        anchor_id=f"note:{note.note_id}",
        is_structural=False,
        state=_state_label(state),
        note_provenance=provenance_label,  # see _structural_note_candidate above
    )


def _code_candidate(result, max_chars: int) -> Candidate:
    file_path = getattr(result, "file_path", "") or ""
    lines = getattr(result, "lines", "") or ""
    symbol = getattr(result, "symbol_name", "") or ""
    score = float(getattr(result, "score", 0.0) or 0.0)
    chunk_id = getattr(result, "chunk_id", "") or f"{file_path}:{lines}"
    first = ""
    for ln in (getattr(result, "content", "") or "").splitlines():
        if ln.strip():
            first = ln.strip()
            break
    where = f"{file_path}:{lines}" if lines else file_path
    label = f" ({symbol})" if symbol else ""
    line = _cap(f"search hit {where}{label}: {first}", max_chars)
    return Candidate(
        kind="code_semantic",
        line=line,
        score=score,
        anchor_id=f"chunk:{chunk_id}",
        is_structural=False,
        state=CANDIDATE_STATE_NOT_APPLICABLE,
    )


class ProactiveMatcher:
    """Runs the enabled matchers and returns the union of scored candidates."""

    def __init__(
        self,
        source: MatchSource,
        *,
        min_similarity: float,
        max_chars_per_event: int,
        structural_score_declared_anchor: float,
        structural_score_declared_trigger: float,
        structural_score_gotcha_mention: float,
        structural_score_mention: float,
        structural_note: bool = True,
        semantic_note: bool = True,
        code_search: bool = False,
        note_limit: int = 10,
        code_limit: int = 5,
        exclude_directive_notes: bool = False,
    ) -> None:
        self._source = source
        self._min_similarity = min_similarity
        self._max_chars = max_chars_per_event
        self._structural_score_declared_anchor = structural_score_declared_anchor
        self._structural_score_declared_trigger = structural_score_declared_trigger
        self._structural_score_gotcha_mention = structural_score_gotcha_mention
        self._structural_score_mention = structural_score_mention
        self._structural_note = structural_note
        self._semantic_note = semantic_note
        self._code_search = code_search
        self._note_limit = note_limit
        self._code_limit = code_limit
        # UPG-PROXY-INJECT-ROLE-PROVENANCE: a static per-channel policy flag
        # (set by the caller from config, e.g. `proactive.proxy.
        # exclude_directive_notes` — see app/service.py's ProactiveMatcher
        # construction), never a runtime read of window/conversation content.
        # When True, `match()` drops every kind=="directive" note BEFORE it
        # is folded into a Candidate at all — filtering on `WorkingNote.kind`,
        # a structural note property, is the allowed "config-declared
        # quality prior on a PROPERTY" category of the no-query-heuristics
        # rule, not a query-content classifier.
        self._exclude_directive_notes = exclude_directive_notes

    def match(self, window: ProactiveWindow) -> list[Candidate]:
        candidates: list[Candidate] = []

        # M1 — structural file match: exact file path -> anchored notes (1.0).
        structural_notes: list[WorkingNote] = []
        if self._structural_note and window.file_paths:
            try:
                structural_notes = self._source.structural_notes(window.file_paths)
            except Exception:
                structural_notes = []
            if self._exclude_directive_notes:
                structural_notes = [n for n in structural_notes if n.kind != "directive"]

        # M3 — semantic note match: cosine recall above the similarity floor.
        semantic_scored: list[tuple[WorkingNote, float]] = []
        if self._semantic_note and window.text.strip():
            try:
                semantic_scored = self._source.semantic_notes(
                    window.text, self._min_similarity, self._note_limit
                )
            except Exception:
                semantic_scored = []
            if self._exclude_directive_notes:
                semantic_scored = [(n, s) for (n, s) in semantic_scored if n.kind != "directive"]

        # UPG-PROXY-REVOKED-LEAK: fold lifecycle state for the UNION of every
        # note either matcher found, ONCE — never per-note, never per-matcher
        # — before rendering a single candidate line.
        #
        # Two distinct failure modes, deliberately handled differently,
        # because conflating them reintroduces the very defect this fixes:
        #
        #   (a) The source does not implement note_states() at all (an older
        #       fake or a backend predating this protocol method). That is a
        #       STATIC property of the source, checked with getattr() and not
        #       by catching AttributeError — every note is active, which is
        #       exactly note_event_states()'s documented contract for an
        #       absent note_id. Backward compatible, renders raw content.
        #
        #   (b) The source implements it but the CALL FAILS (e.g. the
        #       note_events table is unreadable mid-migration). Lifecycle
        #       state is then UNKNOWN, and "unknown" must never be rendered
        #       as "active" — that would put a revoked note's raw content
        #       back into injected context as apparent fact, which is the
        #       defect. Fail CLOSED: drop every note candidate for this
        #       window. M4 code search is unaffected (it carries no note
        #       lifecycle), so the caller degrades to code context only
        #       rather than to wrong memory.
        matched_notes: dict[int, WorkingNote] = {}
        for note in structural_notes:
            matched_notes.setdefault(note.note_id, note)
        for note, _score in semantic_scored:
            matched_notes.setdefault(note.note_id, note)
        note_states: dict[int, dict] = {}
        _states_fn = getattr(self._source, "note_states", None)
        if _states_fn is not None:
            try:
                note_states = _states_fn(list(matched_notes.values()))
            except Exception:
                structural_notes = []
                semantic_scored = []
                note_states = {}

        if self._structural_note and window.file_paths:
            anchors = {p: Path(p).name or p for p in window.file_paths}
            for note in structural_notes:
                state = note_states.get(note.note_id)
                if state and state.get("state") == "superseded":
                    # Already excluded via valid_until in the normal path;
                    # defensive skip if one still reaches the renderer —
                    # never render a superseded note's raw content as an
                    # active fact.
                    continue
                found = _first_anchor(note, anchors)
                if found is not None:
                    anchor_path, anchor, relation = found
                    # UPG-PROXY-WEAK-TIER-TIEBREAK: count how hard THIS note
                    # talks about the matched file (basename occurrences at
                    # genuine path boundaries, and where the first lands) —
                    # the gate's equal-score tie-break signals.
                    # UPG-MATCHER-RELATIVE-ANCHOR-LABEL: `anchor` may now be
                    # the declared form that actually matched (a full path or
                    # a glob), so the signals stay keyed to the BASENAME they
                    # have always counted — the label change alters nothing
                    # about equal-score ordering.
                    base = anchors[anchor_path]
                    content = note.content or ""
                    mention_count = _path_boundary_count(content, base)
                    mention_first_offset = _path_boundary_first(content, base)
                else:
                    # Defensive fallback: the source returned a note this
                    # window's anchors/triggers/content narrowing can't
                    # explain (e.g. a MatchSource test double that ignores
                    # `structural_notes(paths)`'s own path argument). Treated
                    # as the strongest tier, matching this branch's pre-
                    # existing behaviour before the anchor/trigger/mention
                    # 3-way split.
                    anchor_path, anchor, relation = None, "file", STRUCTURAL_TIER_DECLARED_ANCHOR
                    # Leave the tie-break signals at their inert defaults:
                    # `anchor` above is a display placeholder ("file"), not a
                    # filename, and boundary-counting a common English word
                    # would be prose statistics, not structure.
                    mention_count = 0
                    mention_first_offset = -1
                candidates.append(
                    _structural_note_candidate(
                        note, anchor, relation, self._max_chars,
                        self._structural_score_declared_anchor,
                        self._structural_score_declared_trigger,
                        self._structural_score_gotcha_mention,
                        self._structural_score_mention,
                        state,
                        anchor_path,
                        mention_count,
                        mention_first_offset,
                    )
                )

        if self._semantic_note and window.text.strip():
            for note, score in semantic_scored:
                state = note_states.get(note.note_id)
                if state and state.get("state") == "superseded":
                    continue
                candidates.append(
                    _semantic_note_candidate(note, score, self._max_chars, state)
                )

        # M4 — semantic code match: hybrid search hits (opt-in; needs an index).
        if self._code_search and window.text.strip():
            try:
                results = self._source.code_search(window.text, self._code_limit)
            except Exception:
                results = []
            for result in results:
                candidates.append(_code_candidate(result, self._max_chars))

        return candidates


def _first_anchor(
    note: WorkingNote, anchors: dict[str, str]
) -> tuple[str, str, str] | None:
    """Return (window_file_path, display_label, relation) for the first
    window file this note is actually about. `relation` is one of the
    STRUCTURAL_TIER_* constants: DECLARED_ANCHOR, DECLARED_TRIGGER, or
    MENTION (the caller further splits MENTION into GOTCHA_MENTION/MENTION
    by `note.kind` — see `_structural_note_candidate`).

    UPG-PROXY-SUBSTRING-ANCHOR/2b: the note's DECLARED `anchors` column is
    checked first — an exact match there is the strongest signal, and the
    caller may honestly say "anchored to X".

    UPG-TRIGGERS-INERT-ON-PROXY-STRUCTURAL: when no declared anchor
    matches, the note's own EXPLICIT `triggers[]` is checked next for a
    'path' glob matching this window file, via `path_trigger_match()`
    (agent/trigger_engine.py) — the same declared-glob evaluation the live
    hook's trigger engine already uses to FIRE such a note, now also
    consulted for structural RELEVANCE. A glob is deliberate authorial
    intent ("this note concerns this file"), same evidentiary class as an
    anchor, so the caller renders it as "triggers on X" — weaker than
    "anchored to X" (a glob can match many files; an anchor names exactly
    one) but strictly stronger than an incidental content mention.

    Only when neither matches does this fall back to a boundary-matched
    content mention, using the SAME `_path_boundary_match()` rule
    `recall_for_path()` uses (shared, not forked) — the caller must render
    that case as "mentions X", the weakest provenance claim. Returns None
    when the note is about none of the window's files.

    The display label is the form that ACTUALLY matched
    (UPG-MATCHER-RELATIVE-ANCHOR-LABEL): the declared anchor string (basename
    or exact path — whichever the note's `anchors` column contains; basename
    wins when both are declared), or the declared glob pattern for a trigger
    match. Never a derived form the author didn't write.

    The window file path (`anchors` dict's key, not its basename value) is
    returned alongside the display label so `match()` can carry it onto the
    STRUCTURAL_TIER_DECLARED_ANCHOR `Candidate.anchor_path` (UPG-PROXY-
    INJECT-SINGLE-TURN) — the gate needs the exact path to test membership
    against a request's `edited_file_paths`; the basename alone is not
    enough to do that safely."""
    declared = {a[0] for a in (note.anchors or []) if a}
    for path, base in anchors.items():
        if base and (base in declared or path in declared):
            # Report the declared string that actually matched, not
            # unconditionally the basename (UPG-MATCHER-RELATIVE-ANCHOR-LABEL);
            # when BOTH forms are declared, keep the historical basename.
            return path, base if base in declared else path, STRUCTURAL_TIER_DECLARED_ANCHOR
    for path, base in anchors.items():
        matched_glob = path_trigger_match(note.triggers, (path, base))
        if matched_glob is not None:
            # Render the glob that actually matched — path_trigger_match()
            # returns it explicitly for this explanatory purpose, and a
            # wildcard glob reported as one concrete basename overstates a
            # package-wide declaration as a single-file one.
            return path, matched_glob, STRUCTURAL_TIER_DECLARED_TRIGGER
    content = note.content or ""
    for path, base in anchors.items():
        if base and _path_boundary_match(content, base):
            return path, base, STRUCTURAL_TIER_MENTION
    return None
