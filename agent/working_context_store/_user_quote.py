"""Verbatim user-excerpt binding (UPG-MEM-PROVENANCE-USER-STATED).

The trust ladder (bm2-design-skeleton.md §5) has exactly two classes an AI
session can reach on its own: `agent` and `auto`. That leaves a note which
merely TRANSCRIBES something the user said rendering as "Memory to verify
(recorded by an AI session, not human-endorsed)" — understating what it is —
while the mirror risk is the opposite: an agent misremembering "the user said
X" and that claim resurfacing with no frame around it at all.

Neither problem is fixable by letting the writer declare its own authority:
an agent asserting "this came from the user" is self-attestation, exactly the
one-call trust forgery the `provenance="human"` rejection already refuses.
What CAN be checked by machine is whether the note actually quotes the user.
So a write may attach a verbatim excerpt of the user turn it transcribes, and
this module answers one deterministic question about it:

    is that excerpt genuinely present, verbatim, inside the note's content?

Exact characters, compared after whitespace normalization (a transcribed
excerpt is routinely re-wrapped; a re-wrapped quote is still verbatim). Case
is significant, and nothing here is semantic: no embedding, no similarity, no
paraphrase allowance, no query-content inspection. The check either succeeds
on the strings themselves or it does not, and only a success earns the
`user-stated` class (`_store.remember()`). A failure stores the note in the
ordinary `agent` class and reports why — the excerpt is discarded rather than
persisted, so the database never holds an unverified "the user said this".
"""
from __future__ import annotations

import difflib

from agent.config import (
    MEMORY_WRITE_USER_QUOTE_MIN_CHARS,
    MEMORY_WRITE_USER_QUOTE_SPAN_BIND_MAX_PRODUCT_CHARS,
    MEMORY_WRITE_USER_QUOTE_SPAN_BIND_MIN_CHARS,
)

# Rejection reasons surfaced back to the caller at write time. Fixed protocol
# strings, same category as the provenance framing constants in
# agent/trigger_engine.py: the wording IS the contract a caller reads, not an
# operator-tunable value (the one tunable here — how much of the user's actual
# wording an excerpt must carry — is memory_write.user_quote.min_chars).
USER_QUOTE_TOO_SHORT = (
    "user_quote not bound: the excerpt is shorter than "
    "{min_chars} characters, too little of the user's own wording to be "
    "evidence of anything — this note was stored with 'agent' provenance."
)
USER_QUOTE_NOT_CONTAINED = (
    "user_quote not bound: the excerpt does not appear verbatim in this "
    "note's content (whitespace-insensitive, case-sensitive), so nothing "
    "mechanically ties the note to what the user said — this note was stored "
    "with 'agent' provenance. Quote the user's words inside `content` to bind "
    "it."
)
# Auto-bind (UPG-PROVENANCE-NEVER-RISES / UPG-PROVENANCE-AUTOBIND-SPAN)
# rejection reasons. Not caller-facing in the way the two above are — there
# is no explicit `user_quote` argument for a caller to have gotten wrong —
# but returned for the same symmetry and testability `bind_user_quote()`
# offers, and to keep every "why didn't this bind" answer computed by one
# pure function rather than re-derived ad hoc at a call site.
AUTO_QUOTE_TOO_SHORT = (
    "auto-bind not applied: this note's content is shorter than "
    "{min_chars} whitespace-normalized characters, too little of it to be "
    "evidence of anything even if it does match the captured user turn — "
    "this note keeps its ordinary provenance."
)
AUTO_QUOTE_NOT_CONTAINED = (
    "auto-bind not applied: this note's content does not appear verbatim, "
    "as a run of at least {min_chars} characters, in the most recently "
    "captured user turn (whitespace-insensitive, case-sensitive) — a "
    "shorter incidental overlap is not evidence the note transcribes what "
    "the user said, so this note keeps its ordinary provenance."
)
AUTO_QUOTE_NO_RECENT_TURN = (
    "auto-bind not applied: no user turn was captured recently enough (see "
    "memory_write.user_quote.auto_bind_max_age_seconds) to compare against — "
    "this note keeps its ordinary provenance."
)
AUTO_QUOTE_TOO_LARGE_TO_SEARCH = (
    "auto-bind not applied: this note's content and the captured user turn "
    "together exceed the {max_product}-character-product bound for the "
    "verbatim-span search (memory_write.user_quote.span_bind_max_product_"
    "chars) — the search is skipped rather than run unbounded; this note "
    "keeps its ordinary provenance."
)


def normalize_for_binding(text: str) -> str:
    """Collapse every run of whitespace to a single space and strip the ends.

    The ONLY transform applied before comparison. Line wrapping, indentation
    and trailing whitespace differ freely between a user turn and its
    transcription into a note; the words themselves must not.
    """
    return " ".join(text.split())


def bind_user_quote(content: str, user_quote: str | None) -> tuple[str, str]:
    """Return `(bound_quote, rejection_reason)` — exactly one is non-empty.

    `bound_quote` is the caller's excerpt with surrounding whitespace
    stripped, otherwise character-for-character as given (the stored evidence
    must be the user's text, not a normalized rewrite of it). An omitted or
    blank `user_quote` returns `("", "")`: no claim was made, so there is
    nothing to reject.

    Pure and total — no I/O, no clock, no store access — so every surface can
    call it to compute the same reason the store computed, without a second
    write path.
    """
    quote = (user_quote or "").strip()
    if not quote:
        return "", ""
    normalized_quote = normalize_for_binding(quote)
    if len(normalized_quote) < MEMORY_WRITE_USER_QUOTE_MIN_CHARS:
        return "", USER_QUOTE_TOO_SHORT.format(min_chars=MEMORY_WRITE_USER_QUOTE_MIN_CHARS)
    if normalized_quote not in normalize_for_binding(content):
        return "", USER_QUOTE_NOT_CONTAINED
    return quote, ""


def _normalize_with_offsets(text: str) -> tuple[str, list[int]]:
    """Whitespace-collapsing normalization identical to `normalize_for_binding`
    above, plus a parallel index that maps each character of the normalized
    output back to its raw index in `text`.

    Returns `(normalized, offsets)` where `offsets[i]` is the raw index in
    `text` at which normalized character `i` starts, and the trailing
    sentinel `offsets[len(normalized)]` is the raw index one past the last
    character the normalized text represents. For any `0 <= i <= j <=
    len(normalized)`, `text[offsets[i]:offsets[j]]` is the literal raw
    substring that reproduces `normalized[i:j]`'s exact original formatting
    (internal whitespace runs included in full, not collapsed) — this is how
    a span found on the normalized text is turned back into the note's own
    raw characters, never a normalized rewrite of them, for storage: the
    same "stored evidence must be the user's text" principle
    `bind_user_quote()`'s own docstring states for the explicit-quote path
    above. One edge the raw slice does NOT reproduce: if `i` or `j` lands
    exactly on a normalized space character that stands in for a raw
    whitespace RUN, the raw slice includes that entire run rather than a
    single space — re-normalizing `text[offsets[i]:offsets[j]]` in isolation
    then differs from `normalized[i:j]` at that one boundary, since a raw
    run sitting at the very edge of an isolated slice normalizes away
    (`normalize_for_binding` strips ends). Callers that extract a span for
    external use handle this with a plain `.strip()` on the result rather
    than by complicating this function further.
    """
    chars: list[str] = []
    offsets: list[int] = []
    n = len(text)
    i = 0
    seen_content = False
    content_end = 0
    while i < n:
        if text[i].isspace():
            start = i
            while i < n and text[i].isspace():
                i += 1
            if i >= n:
                # Trailing whitespace run (including a run that reaches the
                # end of the string before any real character has been
                # seen at all): contributes nothing to the normalized text
                # (normalize_for_binding strips the ends), and content_end
                # must stay at whatever it already was — the raw index just
                # past the last real character appended so far, or its
                # initial value of 0 if none has been appended yet — so a
                # span ending at the last real character never absorbs
                # this run when sliced back out of the raw text.
                break
            if seen_content:
                # An INTERNAL run: normalize_for_binding's
                # " ".join(text.split()) collapses it to exactly one space.
                chars.append(" ")
                offsets.append(start)
            # else: a LEADING run. normalize_for_binding's .split() strips
            # it entirely (there is no character before the first real one
            # for a leading run to sit "between"), so nothing is appended
            # and no offset recorded for it — matching normalize_for_binding
            # exactly rather than collapsing it to a stray leading space.
        else:
            chars.append(text[i])
            offsets.append(i)
            content_end = i + 1
            seen_content = True
            i += 1
    offsets.append(content_end)
    return "".join(chars), offsets


def _longest_common_span(a_norm: str, b_norm: str) -> tuple[int, int]:
    """Return the `[start, end)` span in `a_norm` of the longest substring
    common to both `a_norm` and `b_norm`, or `(0, 0)` when there is none (or
    either string is empty).

    `difflib.SequenceMatcher` with `autojunk=False`: the default autojunk
    heuristic silently treats an element as noise once it recurs "too often"
    (over 1% of a sequence longer than 200 elements) on the theory that
    heavy repetition is formatting filler, which would make the match found
    depend on how often a character happens to repeat rather than being a
    true longest-common-substring search — the same determinism bar
    `bind_user_quote()` already holds itself to (no fuzzy/semantic/
    frequency-based matching). Ties resolve deterministically per the
    stdlib's own documented contract: the earliest-starting match in
    `a_norm`, and among those, the earliest-starting match in `b_norm`.
    """
    if not a_norm or not b_norm:
        return 0, 0
    match = difflib.SequenceMatcher(None, a_norm, b_norm, autojunk=False).find_longest_match(
        0, len(a_norm), 0, len(b_norm)
    )
    if match.size <= 0:
        return 0, 0
    return match.a, match.a + match.size


def bind_user_quote_auto(content: str, recent_user_message: str | None) -> tuple[str, str]:
    """Harness-layer counterpart to `bind_user_quote()` — the actual fix for
    UPG-PROVENANCE-NEVER-RISES: on a live production corpus, `user_quote=`
    binding is a VOLUNTARY parameter the writing agent must remember to pass
    on every `remember()` call, and in practice it is essentially never
    passed — exactly the adoption gap hook-injected recall was already built
    to close for the READ side of memory (a caller choosing to call
    `vectr_recall` vs. notes simply arriving via the hook). This is the WRITE
    side of the same fix: a note earns `user-stated` provenance because the
    HARNESS mechanically checked it against what the user actually typed,
    never because the writer remembered to self-declare it.

    `bind_user_quote()` verifies a caller-declared EXCERPT is genuinely
    inside the note (excerpt ⊆ content) — the excerpt is presumed correct and
    the check confirms the note actually contains it. There is no
    caller-declared excerpt here, only the note's own CONTENT and
    RECENT_USER_MESSAGE (the raw, unmodified text of the most recently
    captured user turn in this workspace — see
    `VectrService._last_user_prompt`, populated only from the verbatim
    `UserPromptSubmit` hook payload, never from any interpretation of it).

    UPG-PROVENANCE-AUTOBIND-SPAN: an earlier version of this function
    required CONTENT's entire body to appear verbatim inside
    RECENT_USER_MESSAGE — measured reach on a live 742-note corpus was "at
    most 1 of 45" directives, because a real directive is routinely wrapped
    in agent-authored framing the user never typed (e.g. `USER DIRECTIVE
    2026-07-12: "<the user's actual words>"` — the surrounding label and
    quotation marks are the writing agent's, only the inner clause is the
    user's). Whole-body containment can never match that. This function
    instead searches for the LONGEST contiguous run of CONTENT that appears
    verbatim (whitespace-insensitive, case-sensitive — the same
    normalization `bind_user_quote()` uses) inside RECENT_USER_MESSAGE, and
    binds that run alone when it reaches `span_bind_min_chars` — deliberately
    a HIGHER floor than `bind_user_quote()`'s `min_chars`, because a run
    found by search is a weaker signal than one a caller explicitly declared
    at the same length (see memory_write.user_quote.span_bind_min_chars in
    config.yaml for the concrete false-bind examples that floor was set
    against). A note whose entire body happens to be the matching span (the
    old whole-body case) still binds exactly as before — that is simply the
    span covering all of CONTENT. This is still deliberately NOT a fuzzy or
    semantic match: no embedding, no similarity, no paraphrase allowance —
    only a longest-run substring search over two already-normalized strings,
    exact characters or nothing. Search cost is bounded (see
    span_bind_max_product_chars) rather than letting a pathological pair of
    inputs run unbounded.

    Returns `(bound_quote, rejection_reason)` exactly like
    `bind_user_quote()`. `bound_quote` is the matched span sliced out of
    CONTENT's own raw characters (via `_normalize_with_offsets`), never a
    normalized rewrite of it — same "stored evidence is the user's text"
    contract `bind_user_quote()` holds for its own excerpt. Pure and total —
    no I/O, no clock; the caller is responsible for the recency check
    (comparing the captured turn's own timestamp against
    `auto_bind_max_age_seconds`) before ever passing RECENT_USER_MESSAGE in
    here, same separation of concerns as `bind_user_quote()` never touching
    the clock either."""
    recent = (recent_user_message or "").strip()
    stripped_content = (content or "").strip()
    if not stripped_content:
        return "", ""
    if not recent:
        return "", AUTO_QUOTE_NO_RECENT_TURN
    normalized_content, content_offsets = _normalize_with_offsets(stripped_content)
    if len(normalized_content) < MEMORY_WRITE_USER_QUOTE_SPAN_BIND_MIN_CHARS:
        return "", AUTO_QUOTE_TOO_SHORT.format(
            min_chars=MEMORY_WRITE_USER_QUOTE_SPAN_BIND_MIN_CHARS
        )
    normalized_recent = normalize_for_binding(recent)
    product = len(normalized_content) * len(normalized_recent)
    if product > MEMORY_WRITE_USER_QUOTE_SPAN_BIND_MAX_PRODUCT_CHARS:
        return "", AUTO_QUOTE_TOO_LARGE_TO_SEARCH.format(
            max_product=MEMORY_WRITE_USER_QUOTE_SPAN_BIND_MAX_PRODUCT_CHARS
        )
    start, end = _longest_common_span(normalized_content, normalized_recent)
    if end - start < MEMORY_WRITE_USER_QUOTE_SPAN_BIND_MIN_CHARS:
        return "", AUTO_QUOTE_NOT_CONTAINED.format(
            min_chars=MEMORY_WRITE_USER_QUOTE_SPAN_BIND_MIN_CHARS
        )
    span = stripped_content[content_offsets[start]:content_offsets[end]]
    # `.strip()`: only bites when `start`/`end` land exactly on a normalized
    # space character standing in for a raw whitespace RUN (see
    # `_normalize_with_offsets`'s docstring) — the raw slice then carries
    # that whole run at its edge rather than the single collapsed space,
    # which `.strip()` removes. Losing at most one collapsed-space's worth
    # of normalized length off either edge never threatens the floor
    # already enforced above at a much larger threshold.
    return span.strip(), ""
