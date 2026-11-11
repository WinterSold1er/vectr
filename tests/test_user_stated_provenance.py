"""UPG-MEM-PROVENANCE-USER-STATED — the machine-checkable middle trust class.

A note that merely transcribes what the user said used to render with the
agent frame ("Memory to verify ... not human-endorsed"), understating it,
while an agent misremembering "the user said X" carried no frame at all. The
fix is deliberately NOT a new self-declared provenance value: an agent
asserting its own authority is the forgery `provenance="human"` is already
rejected for. It is a deterministic string check — a verbatim user excerpt
must actually be quoted inside the note's own content — and only that check
can mint the class.

Covered here: the binding function itself, the store's write-time behaviour
(bound, unbound/adversarial, human-wins, directive interaction, persistence),
the promotion ladder's treatment of a derived class, the rendered frame, and
both caller surfaces (MCP dispatch + REST route).
"""
from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock

import pytest

from agent.config import (
    MEMORY_WRITE_USER_QUOTE_MIN_CHARS,
    MEMORY_WRITE_USER_QUOTE_SPAN_BIND_MAX_PRODUCT_CHARS,
    MEMORY_WRITE_USER_QUOTE_SPAN_BIND_MIN_CHARS,
)
from agent.proactive.types import NOTE_PROVENANCE_TRUST_RANK
from agent.trigger_engine import frame_prefix
from agent.working_context_store import (
    AUTO_QUOTE_TOO_LARGE_TO_SEARCH,
    PROMOTION_LADDER,
    PROMOTION_RANK,
    PROVENANCE_VALUES,
    USER_STATED_PROVENANCE,
    WorkingContextStore,
    _longest_common_span,
    _normalize_with_offsets,
    bind_user_quote,
    bind_user_quote_auto,
    normalize_for_binding,
)

# A user turn long enough to clear the minimum-excerpt floor.
QUOTE = "always run the tests through the venv interpreter"
CONTENT = f'The user said: "{QUOTE}" — global python lacks the C grammars.'


def _store(tmp_path) -> WorkingContextStore:
    return WorkingContextStore(str(tmp_path))


# ---------------------------------------------------------------------------
# bind_user_quote — the deterministic check itself
# ---------------------------------------------------------------------------

class TestBindUserQuote:
    def test_exact_substring_binds(self) -> None:
        bound, reason = bind_user_quote(CONTENT, QUOTE)
        assert bound == QUOTE
        assert reason == ""

    def test_rewrapped_quote_still_binds(self) -> None:
        """A transcribed excerpt is routinely re-wrapped; the words are what
        must match, not the line breaks."""
        bound, reason = bind_user_quote(
            "The user said:\n  always run the tests\n  through the venv interpreter\nso do that.",
            "always run the tests through the venv interpreter",
        )
        assert bound == "always run the tests through the venv interpreter"
        assert reason == ""

    def test_bound_quote_keeps_the_users_own_characters(self) -> None:
        """The stored evidence is the excerpt as given (whitespace-stripped at
        the ends only), never a normalized rewrite of it."""
        bound, _ = bind_user_quote(CONTENT, f"  {QUOTE}  ")
        assert bound == QUOTE

    def test_paraphrase_does_not_bind(self) -> None:
        bound, reason = bind_user_quote(
            "The user wants the venv interpreter used for tests.",
            "always run the tests through the venv interpreter",
        )
        assert bound == ""
        assert "does not appear verbatim" in reason

    def test_check_is_case_sensitive(self) -> None:
        bound, reason = bind_user_quote(CONTENT, QUOTE.upper())
        assert bound == ""
        assert reason != ""

    def test_trivial_excerpt_is_rejected_before_containment(self) -> None:
        """Without a floor, a 1-character 'excerpt' would be a substring of
        almost any content — confirming nothing while upgrading trust."""
        bound, reason = bind_user_quote("a note that certainly contains an a", "a")
        assert bound == ""
        assert str(MEMORY_WRITE_USER_QUOTE_MIN_CHARS) in reason

    @pytest.mark.parametrize("quote", [None, "", "   ", "\n\t "])
    def test_absent_quote_is_no_claim_and_no_rejection(self, quote) -> None:
        assert bind_user_quote(CONTENT, quote) == ("", "")

    def test_normalize_collapses_whitespace_only(self) -> None:
        assert normalize_for_binding("  a\n\tb   c  ") == "a b c"

    def test_is_pure_and_repeatable(self) -> None:
        assert bind_user_quote(CONTENT, QUOTE) == bind_user_quote(CONTENT, QUOTE)


# ---------------------------------------------------------------------------
# bind_user_quote_auto — the harness-driven counterpart
# (UPG-PROVENANCE-NEVER-RISES). `bind_user_quote` checks excerpt ⊆ content;
# this checks the MIRROR direction, content ⊆ recent_user_message — the
# note's whole content must appear verbatim inside the raw captured user
# turn, not the other way around.
# ---------------------------------------------------------------------------

class TestBindUserQuoteAuto:
    def test_content_contained_in_recent_message_binds(self) -> None:
        bound, reason = bind_user_quote_auto(
            QUOTE, f"the user said: {QUOTE} and nothing else",
        )
        assert bound == QUOTE
        assert reason == ""

    def test_bound_value_is_the_content_not_the_recent_message(self) -> None:
        """The evidence returned is what gets stored as `user_quote` on the
        note — it must be the note's OWN content (stripped), not the longer
        captured turn it was found inside, mirroring `bind_user_quote`
        returning the excerpt rather than the whole content."""
        recent = f"lots of preamble here. {QUOTE} and a trailing remark too."
        bound, _ = bind_user_quote_auto(QUOTE, recent)
        assert bound == QUOTE

    def test_rewrapped_content_still_binds(self) -> None:
        bound, reason = bind_user_quote_auto(
            "always run the tests\nthrough the venv interpreter",
            "The user said: always run the tests through the venv interpreter, ok.",
        )
        assert bound == "always run the tests\nthrough the venv interpreter"
        assert reason == ""

    def test_paraphrased_recent_message_does_not_bind(self) -> None:
        bound, reason = bind_user_quote_auto(
            QUOTE, "the user wants the venv interpreter used for tests",
        )
        assert bound == ""
        assert "does not appear verbatim" in reason

    def test_direction_is_genuinely_mirrored_not_symmetric(self) -> None:
        """`bind_user_quote(content, excerpt)` requires excerpt ⊆ content.
        `bind_user_quote_auto(content, recent_message)` requires the OPPOSITE
        containment, content ⊆ recent_message. A short note whose content is
        itself only a fragment of a much longer captured turn must still
        bind here even though the equivalent `bind_user_quote` call (with
        the arguments in the non-mirrored order) would not."""
        long_turn = f"context before. {QUOTE}. context after, unrelated words."
        bound, _ = bind_user_quote_auto(QUOTE, long_turn)
        assert bound == QUOTE
        # The non-mirrored direction (QUOTE as content, long_turn as the
        # thing that must be contained in it) correctly fails to bind.
        non_mirrored_bound, _ = bind_user_quote(QUOTE, long_turn)
        assert non_mirrored_bound == ""

    def test_check_is_case_sensitive(self) -> None:
        bound, reason = bind_user_quote_auto(QUOTE, QUOTE.upper())
        assert bound == ""
        assert reason != ""

    def test_trivial_content_is_rejected_before_containment(self) -> None:
        """The auto-bind path's floor is `span_bind_min_chars`, deliberately
        HIGHER than `bind_user_quote()`'s own `min_chars` -- see
        UPG-PROVENANCE-AUTOBIND-SPAN in tasks.md and the config.yaml
        comments on `memory_write.user_quote.span_bind_min_chars` for the
        concrete false-bind examples that floor was set against (a span
        found by search is a weaker signal than one a caller explicitly
        declared at the same length)."""
        bound, reason = bind_user_quote_auto("a", "a note that certainly contains an a")
        assert bound == ""
        assert str(MEMORY_WRITE_USER_QUOTE_SPAN_BIND_MIN_CHARS) in reason

    @pytest.mark.parametrize("recent_message", [None, "", "   ", "\n\t "])
    def test_no_recent_message_is_no_bind_and_a_distinct_reason(self, recent_message) -> None:
        bound, reason = bind_user_quote_auto(QUOTE, recent_message)
        assert bound == ""
        assert reason != ""

    def test_empty_content_is_no_claim_and_no_rejection(self) -> None:
        assert bind_user_quote_auto("", f"the user said: {QUOTE}") == ("", "")

    def test_is_pure_and_repeatable(self) -> None:
        recent = f"the user said: {QUOTE}"
        assert bind_user_quote_auto(QUOTE, recent) == bind_user_quote_auto(QUOTE, recent)


# ---------------------------------------------------------------------------
# Longest-verbatim-span auto-bind (UPG-PROVENANCE-AUTOBIND-SPAN) — the
# measured reach on a live 742-note corpus for whole-body containment was
# "at most 1 of 45" directives, because a real directive is routinely
# wrapped in agent-authored framing the user never typed. These tests pin
# the LCS-span search itself, the higher span-only floor, the cost cap, and
# the offset-mapping helper it depends on.
# ---------------------------------------------------------------------------

class TestNormalizeWithOffsets:
    """`_normalize_with_offsets` must reproduce `normalize_for_binding`'s
    output exactly (same collapsing rule) while also handing back a way to
    slice the ORIGINAL raw characters back out for any span found on the
    normalized text."""

    def test_matches_normalize_for_binding_on_leading_internal_trailing_whitespace(
        self,
    ) -> None:
        text = "  a   b  "
        normalized, _offsets = _normalize_with_offsets(text)
        assert normalized == normalize_for_binding(text) == "a b"

    def test_offsets_length_is_normalized_length_plus_one(self) -> None:
        normalized, offsets = _normalize_with_offsets("  a   b  ")
        assert len(offsets) == len(normalized) + 1

    def test_leading_whitespace_is_stripped_not_collapsed_to_a_stray_space(self) -> None:
        """Regression guard: an earlier version of this function treated
        EVERY non-trailing whitespace run identically (collapse to one
        space), which wrongly emitted a leading space for text starting
        with whitespace — `normalize_for_binding`'s `" ".join(text.split())`
        strips a leading run entirely instead."""
        normalized, _offsets = _normalize_with_offsets("   a")
        assert normalized == "a"
        assert not normalized.startswith(" ")

    def test_all_whitespace_text_normalizes_to_empty(self) -> None:
        normalized, offsets = _normalize_with_offsets("   ")
        assert normalized == ""
        assert offsets == [0]

    def test_no_whitespace_text_offsets_are_identity(self) -> None:
        normalized, offsets = _normalize_with_offsets("abc")
        assert normalized == "abc"
        assert offsets == [0, 1, 2, 3]

    def test_empty_text(self) -> None:
        assert _normalize_with_offsets("") == ("", [0])

    def test_internal_run_offset_points_at_the_start_of_the_raw_run(self) -> None:
        """A span whose right edge lands just past a collapsed internal
        run must, when sliced back out of the raw text, include the WHOLE
        raw run (not just one space) — this is what lets a span ending
        right before the next real character reproduce the original
        formatting exactly, rather than truncating it mid-run."""
        text = "a   b"
        normalized, offsets = _normalize_with_offsets(text)
        assert normalized == "a b"
        # normalized[0:2] == "a " -- the raw slice for that span is the
        # 'a' plus the ENTIRE 3-space run, not just one space.
        assert text[offsets[0]:offsets[2]] == "a   "


class TestLongestCommonSpan:
    def test_finds_the_full_shared_substring(self) -> None:
        start, end = _longest_common_span("hello world", "say hello world now")
        assert "hello world"[start:end] == "hello world"

    def test_no_overlap_returns_zero_width(self) -> None:
        assert _longest_common_span("abcdef", "ghijkl") == (0, 0)

    def test_empty_inputs_return_zero_width(self) -> None:
        assert _longest_common_span("", "anything") == (0, 0)
        assert _longest_common_span("anything", "") == (0, 0)

    def test_autojunk_is_disabled_so_a_repeated_character_still_matches(self) -> None:
        """`difflib.SequenceMatcher`'s default autojunk heuristic silently
        drops an element that recurs "too often" as presumed formatting
        filler once a sequence exceeds 200 elements — with autojunk left
        on, a real long repeated run would be classified as junk and the
        true longest match would be missed non-deterministically w.r.t.
        input length. Build a >200-char string whose only long common run
        is a heavily-repeated character to catch a regression to the
        default (autojunk=True)."""
        a = "x" * 250 + "UNIQUETAIL"
        b = "y" * 10 + "x" * 250 + "UNIQUETAIL" + "z" * 10
        start, end = _longest_common_span(a, b)
        assert end - start == len(a)  # the entire run + tail matches


class TestBindUserQuoteAutoSpanBinding:
    def test_span_binds_inside_agent_authored_framing(self) -> None:
        """The motivating UPG-PROVENANCE-AUTOBIND-SPAN example: a real
        directive note wraps the user's actual words in a label and quote
        marks the user never typed. Whole-body containment could never
        match this; the span search binds the inner clause alone."""
        inner = (
            "once camel completes embedding, start the eval, "
            "dont wait for my go ahead."
        )
        content = f'USER DIRECTIVE 2026-07-12: "{inner}"'
        bound, reason = bind_user_quote_auto(content, inner)
        assert bound == inner
        assert reason == ""

    def test_whole_body_case_still_binds_exactly_as_before(self) -> None:
        bound, reason = bind_user_quote_auto(QUOTE, f"the user said: {QUOTE}")
        assert bound == QUOTE
        assert reason == ""

    def test_short_coincidental_overlap_below_the_span_floor_does_not_bind(self) -> None:
        """False-bind guard: 'the venv interpreter' occurs verbatim inside
        both CONTENT and an UNRELATED recent turn about linting, but at 21
        normalized characters it is well under the 30-character span floor
        -- a coincidental phrase overlap this short is not evidence the
        note transcribes what the user actually said."""
        content = "always run the tests through the venv interpreter"
        recent = "please make sure you run the linter before committing, thanks"
        bound, reason = bind_user_quote_auto(content, recent)
        assert bound == ""
        assert "does not appear verbatim" in reason
        assert "30" in reason

    def test_second_short_coincidental_overlap_below_the_span_floor_does_not_bind(
        self,
    ) -> None:
        content = "please run the linter before you push this branch"
        recent = "always run the tests through the venv interpreter, thanks"
        bound, reason = bind_user_quote_auto(content, recent)
        assert bound == ""
        assert "does not appear verbatim" in reason

    def test_span_just_at_the_floor_binds(self) -> None:
        # Boundary characters (AAA/BBB vs CCC/DDD) deliberately differ on
        # each side of the shared run so the longest common span is EXACTLY
        # the run itself -- no incidental shared whitespace at either edge
        # to (dis)qualify the match by accident.
        thirty_chars = "x" * 30
        content = f"AAA{thirty_chars}BBB"
        recent = f"CCC{thirty_chars}DDD"
        bound, reason = bind_user_quote_auto(content, recent)
        assert bound == thirty_chars
        assert reason == ""

    def test_span_one_under_the_floor_does_not_bind(self) -> None:
        twenty_nine_chars = "x" * 29
        content = f"AAA{twenty_nine_chars}BBB"
        recent = f"CCC{twenty_nine_chars}DDD"
        bound, reason = bind_user_quote_auto(content, recent)
        assert bound == ""
        assert "does not appear verbatim" in reason

    def test_oversized_input_pair_is_rejected_without_running_the_search(self) -> None:
        """Cost cap: `span_bind_max_product_chars` bounds the O(n*m)
        search's worst case. A pair whose normalized-length product
        exceeds it is rejected outright rather than run unbounded."""
        big_content = "a" * 3000
        big_recent = "b" * 3000  # product = 9,000,000 > 4,000,000 default
        bound, reason = bind_user_quote_auto(big_content, big_recent)
        assert bound == ""
        assert reason == AUTO_QUOTE_TOO_LARGE_TO_SEARCH.format(
            max_product=4_000_000,
        )

    def test_oversized_input_pair_returns_promptly(self) -> None:
        """The size check happens BEFORE `_longest_common_span` is ever
        called -- confirmed by wall-clock: a real O(n*m) search over two
        3000-character strings would not return in well under a second on
        any reasonable machine, so a fast return here is direct evidence
        the search was skipped, not merely won by an efficient
        implementation."""
        import time

        big_content = "a" * 3000
        big_recent = "b" * 3000
        t0 = time.monotonic()
        bind_user_quote_auto(big_content, big_recent)
        assert time.monotonic() - t0 < 1.0

    def test_span_extraction_strips_stray_whitespace_from_a_collapsed_boundary_edge(
        self,
    ) -> None:
        """Regression guard: when the matched span's boundary lands exactly
        on a normalized space character that stands in for a raw
        whitespace RUN, the raw slice (before `.strip()`) carries the
        WHOLE run at its edge rather than a single space. Construct
        content whose only long common run with `recent` starts right at
        such a boundary (a 5-raw-space run collapsing to the one leading
        space of the match) and confirm the bound quote has no stray
        leading whitespace."""
        content = "X" * 10 + "     " + "Y" * 30
        recent = "prefix " + "Y" * 30 + " suffix"
        bound, reason = bind_user_quote_auto(content, recent)
        assert reason == ""
        assert bound == "Y" * 30
        assert not bound.startswith(" ")
        assert not bound.endswith(" ")


# ---------------------------------------------------------------------------
# WorkingContextStore.remember — write-time class derivation
# ---------------------------------------------------------------------------

class TestStoreUserQuoteBinding:
    def test_bound_quote_stores_user_stated_provenance(self, tmp_path) -> None:
        store = _store(tmp_path)
        ws = str(tmp_path)
        note_id = store.remember(ws, CONTENT, user_quote=QUOTE)
        note = store.get_note(ws, note_id)
        assert note.provenance == USER_STATED_PROVENANCE
        assert note.user_quote == QUOTE

    def test_unbound_quote_stores_agent_provenance_and_discards_the_quote(self, tmp_path) -> None:
        """THE adversarial case: an agent claims the user said something the
        note does not actually quote. The write succeeds, at the ordinary
        class, and the unverified claim is never persisted."""
        store = _store(tmp_path)
        ws = str(tmp_path)
        note_id = store.remember(
            ws, "The user wants the venv used.", user_quote=QUOTE,
        )
        note = store.get_note(ws, note_id)
        assert note.provenance == "agent"
        assert note.user_quote == ""

    def test_unbound_quote_is_not_in_the_database_at_all(self, tmp_path) -> None:
        store = _store(tmp_path)
        store.remember(str(tmp_path), "unrelated content", user_quote=QUOTE)
        conn = sqlite3.connect(str(tmp_path / "working_context.sqlite"))
        row = conn.execute("SELECT user_quote FROM notes LIMIT 1").fetchone()
        conn.close()
        assert row[0] == ""

    def test_omitted_quote_keeps_the_default_class(self, tmp_path) -> None:
        store = _store(tmp_path)
        ws = str(tmp_path)
        note = store.get_note(ws, store.remember(ws, CONTENT))
        assert note.provenance == "agent"
        assert note.user_quote == ""

    def test_provenance_user_stated_cannot_be_declared_directly(self, tmp_path) -> None:
        store = _store(tmp_path)
        with pytest.raises(ValueError) as exc:
            store.remember(str(tmp_path), CONTENT, provenance=USER_STATED_PROVENANCE)
        assert "cannot be set directly" in str(exc.value)
        assert "user_quote" in str(exc.value)

    def test_declared_user_stated_is_rejected_even_with_a_bindable_quote(self, tmp_path) -> None:
        """The rejection is about the DECLARATION, not the evidence — the
        same write succeeds when the caller lets the check do the work."""
        store = _store(tmp_path)
        with pytest.raises(ValueError):
            store.remember(
                str(tmp_path), CONTENT,
                provenance=USER_STATED_PROVENANCE, user_quote=QUOTE,
            )

    def test_human_provenance_outranks_a_binding(self, tmp_path) -> None:
        store = _store(tmp_path)
        ws = str(tmp_path)
        note = store.get_note(ws, store.remember(ws, CONTENT, provenance="human", user_quote=QUOTE))
        assert note.provenance == "human"
        assert note.user_quote == QUOTE

    def test_auto_with_a_bound_quote_becomes_user_stated(self, tmp_path) -> None:
        store = _store(tmp_path)
        ws = str(tmp_path)
        note = store.get_note(ws, store.remember(ws, CONTENT, provenance="auto", user_quote=QUOTE))
        assert note.provenance == USER_STATED_PROVENANCE

    def test_bound_quote_makes_an_auto_directive_writable(self, tmp_path) -> None:
        """The auto/directive guard rejects an UNREVIEWED standing rule. A
        directive carrying the user's own bound words is the opposite of
        that, so the guard must see the effective class."""
        store = _store(tmp_path)
        ws = str(tmp_path)
        note_id = store.remember(
            ws, CONTENT, kind="directive", provenance="auto", user_quote=QUOTE,
        )
        assert store.get_note(ws, note_id).provenance == USER_STATED_PROVENANCE

    def test_auto_directive_without_a_binding_is_still_rejected(self, tmp_path) -> None:
        store = _store(tmp_path)
        with pytest.raises(ValueError) as exc:
            store.remember(str(tmp_path), CONTENT, kind="directive", provenance="auto")
        assert "directive" in str(exc.value)

    def test_unbindable_quote_does_not_rescue_an_auto_directive(self, tmp_path) -> None:
        store = _store(tmp_path)
        with pytest.raises(ValueError):
            store.remember(
                str(tmp_path), "content that never quotes the user",
                kind="directive", provenance="auto", user_quote=QUOTE,
            )

    def test_binding_survives_a_fresh_store_instance(self, tmp_path) -> None:
        ws = str(tmp_path)
        note_id = _store(tmp_path).remember(ws, CONTENT, user_quote=QUOTE)
        note = _store(tmp_path).get_note(ws, note_id)
        assert note.provenance == USER_STATED_PROVENANCE
        assert note.user_quote == QUOTE

    def test_bound_quote_is_encrypted_alongside_content(self, tmp_path, monkeypatch) -> None:
        """A bound quote is by construction a substring of content — storing
        it in plaintext would leak exactly what encryption protects."""
        monkeypatch.setenv("VECTR_ENCRYPT_KEY", "user-stated-provenance-test-key")
        ws = str(tmp_path)
        store = WorkingContextStore(ws)
        note_id = store.remember(ws, CONTENT, user_quote=QUOTE)
        conn = sqlite3.connect(str(tmp_path / "working_context.sqlite"))
        row = conn.execute("SELECT user_quote FROM notes WHERE note_id = ?", (note_id,)).fetchone()
        conn.close()
        assert QUOTE not in row[0]
        assert store.get_note(ws, note_id).user_quote == QUOTE

    def test_recall_returns_the_bound_class(self, tmp_path) -> None:
        store = _store(tmp_path)
        ws = str(tmp_path)
        store.remember(ws, CONTENT, user_quote=QUOTE)
        notes = store.recall(ws)
        assert [n.provenance for n in notes] == [USER_STATED_PROVENANCE]


# ---------------------------------------------------------------------------
# WorkingContextStore.remember — the harness-driven auto-bind fallback
# (UPG-PROVENANCE-NEVER-RISES). Same write-time upgrade as
# TestStoreUserQuoteBinding above, reached without any explicit `user_quote`
# argument — the caller instead supplies `recent_user_message`, the raw text
# of the most recently captured user turn.
# ---------------------------------------------------------------------------

class TestAutoBindStoreBinding:
    def test_content_contained_in_recent_message_binds_to_user_stated(self, tmp_path) -> None:
        store = _store(tmp_path)
        ws = str(tmp_path)
        note_id = store.remember(ws, QUOTE, recent_user_message=f"the user said: {QUOTE}")
        note = store.get_note(ws, note_id)
        assert note.provenance == USER_STATED_PROVENANCE
        assert note.user_quote == QUOTE

    def test_paraphrased_recent_message_keeps_agent_provenance(self, tmp_path) -> None:
        store = _store(tmp_path)
        ws = str(tmp_path)
        note_id = store.remember(
            ws, QUOTE, recent_user_message="the user wants the venv interpreter used",
        )
        note = store.get_note(ws, note_id)
        assert note.provenance == "agent"
        assert note.user_quote == ""

    def test_no_recent_message_keeps_agent_provenance(self, tmp_path) -> None:
        store = _store(tmp_path)
        ws = str(tmp_path)
        note = store.get_note(ws, store.remember(ws, QUOTE))
        assert note.provenance == "agent"
        assert note.user_quote == ""

    def test_explicit_user_quote_wins_over_auto_bind_when_both_would_succeed(self, tmp_path) -> None:
        """An explicit claim is a deliberate act by the caller; the harness
        cache is a fallback for callers that made no claim at all. When both
        would independently bind, the explicit one is what gets stored —
        confirmed by using a DIFFERENT (still-valid) excerpt for each, so a
        wrong precedence is observable in which text ends up as user_quote."""
        other_excerpt = "global python lacks the C grammars."
        store = _store(tmp_path)
        ws = str(tmp_path)
        note_id = store.remember(
            ws, CONTENT,
            user_quote=other_excerpt,
            recent_user_message=f"the user said: {CONTENT}",
        )
        note = store.get_note(ws, note_id)
        assert note.provenance == USER_STATED_PROVENANCE
        assert note.user_quote == other_excerpt

    def test_failed_explicit_quote_still_falls_back_to_auto_bind(self, tmp_path) -> None:
        """A `user_quote` that does not bind is not a hard failure of the
        upgrade path as a whole — it only means the EXPLICIT claim did not
        check out; the harness-cache fallback still gets its own chance."""
        store = _store(tmp_path)
        ws = str(tmp_path)
        note_id = store.remember(
            ws, QUOTE,
            user_quote="the user definitely said something else entirely",
            recent_user_message=f"the user said: {QUOTE}",
        )
        note = store.get_note(ws, note_id)
        assert note.provenance == USER_STATED_PROVENANCE
        assert note.user_quote == QUOTE

    def test_human_provenance_outranks_auto_bind(self, tmp_path) -> None:
        store = _store(tmp_path)
        ws = str(tmp_path)
        note = store.get_note(ws, store.remember(
            ws, QUOTE, provenance="human", recent_user_message=f"the user said: {QUOTE}",
        ))
        assert note.provenance == "human"
        assert note.user_quote == QUOTE

    def test_auto_bind_makes_an_auto_directive_writable(self, tmp_path) -> None:
        store = _store(tmp_path)
        ws = str(tmp_path)
        note_id = store.remember(
            ws, QUOTE, kind="directive", provenance="auto",
            recent_user_message=f"the user said: {QUOTE}",
        )
        assert store.get_note(ws, note_id).provenance == USER_STATED_PROVENANCE

    def test_short_content_is_rejected_before_containment(self, tmp_path) -> None:
        store = _store(tmp_path)
        ws = str(tmp_path)
        note = store.get_note(ws, store.remember(
            ws, "hi", recent_user_message="hi there, the user said hi",
        ))
        assert note.provenance == "agent"


# ---------------------------------------------------------------------------
# VectrService — recall(hook_event="UserPromptSubmit") populates the cache
# remember()'s auto-bind fallback reads (UPG-PROVENANCE-NEVER-RISES). Uses
# the real (non-mocked) VectrService construction pattern from
# tests/test_memory_only_mode.py (_make_service), memory_only=True and a
# dummy embed provider so no real model loads.
# ---------------------------------------------------------------------------

class TestAutoBindServiceIntegration:
    def test_hook_recall_then_remember_auto_binds(self, tmp_path, monkeypatch) -> None:
        from tests.test_memory_only_mode import _make_service

        svc = _make_service(tmp_path, monkeypatch, memory_only=True)
        svc.recall(query=f"the user said: {QUOTE}", hook_event="UserPromptSubmit")
        note = svc.get_note(svc.remember(QUOTE))
        assert note.provenance == USER_STATED_PROVENANCE
        assert note.user_quote == QUOTE

    def test_direct_recall_call_does_not_populate_the_cache(self, tmp_path, monkeypatch) -> None:
        """A plain vectr_recall call (hook_event=None) is not harness-captured
        user input — it must never seed the auto-bind cache."""
        from tests.test_memory_only_mode import _make_service

        svc = _make_service(tmp_path, monkeypatch, memory_only=True)
        svc.recall(query=f"the user said: {QUOTE}")
        note = svc.get_note(svc.remember(QUOTE))
        assert note.provenance == "agent"

    def test_stale_cached_prompt_does_not_auto_bind(self, tmp_path, monkeypatch) -> None:
        from tests.test_memory_only_mode import _make_service
        from agent.config import MEMORY_WRITE_USER_QUOTE_AUTO_BIND_MAX_AGE_SECONDS

        svc = _make_service(tmp_path, monkeypatch, memory_only=True)
        svc.recall(query=f"the user said: {QUOTE}", hook_event="UserPromptSubmit")
        # Backdate the cache past the recency gate without a real sleep.
        svc._last_user_prompt_ts -= MEMORY_WRITE_USER_QUOTE_AUTO_BIND_MAX_AGE_SECONDS + 1
        note = svc.get_note(svc.remember(QUOTE))
        assert note.provenance == "agent"

    def test_explicit_user_quote_still_wins_at_the_service_layer(self, tmp_path, monkeypatch) -> None:
        other_excerpt = "global python lacks the C grammars."
        from tests.test_memory_only_mode import _make_service

        svc = _make_service(tmp_path, monkeypatch, memory_only=True)
        svc.recall(query=f"the user said: {CONTENT}", hook_event="UserPromptSubmit")
        note = svc.get_note(svc.remember(CONTENT, user_quote=other_excerpt))
        assert note.provenance == USER_STATED_PROVENANCE
        assert note.user_quote == other_excerpt


# ---------------------------------------------------------------------------
# Promotion ladder — a derived class is a source, never a target
# ---------------------------------------------------------------------------

class TestPromotionInteraction:
    def test_user_stated_is_not_a_promotion_target(self, tmp_path) -> None:
        store = _store(tmp_path)
        ws = str(tmp_path)
        note_id = store.remember(ws, "an ordinary agent note")
        with pytest.raises(ValueError):
            store.promote(ws, note_id, USER_STATED_PROVENANCE)

    def test_user_stated_note_promotes_to_human(self, tmp_path) -> None:
        store = _store(tmp_path)
        ws = str(tmp_path)
        note_id = store.remember(ws, CONTENT, user_quote=QUOTE)
        assert store.promote(ws, note_id, "human") is True
        assert store.get_note(ws, note_id).provenance == "human"

    def test_user_stated_note_cannot_be_demoted_via_promote(self, tmp_path) -> None:
        """An off-ladder value defaulting to rank 0 would have let an
        'auto -> agent' promotion quietly weaken this note."""
        store = _store(tmp_path)
        ws = str(tmp_path)
        note_id = store.remember(ws, CONTENT, user_quote=QUOTE)
        with pytest.raises(ValueError):
            store.promote(ws, note_id, "agent")
        assert store.get_note(ws, note_id).provenance == USER_STATED_PROVENANCE

    def test_existing_ladder_steps_are_unchanged(self, tmp_path) -> None:
        store = _store(tmp_path)
        ws = str(tmp_path)
        note_id = store.remember(ws, "auto-captured", provenance="auto")
        assert store.promote(ws, note_id, "agent") is True
        assert store.promote(ws, note_id, "human") is True


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

class TestUserStatedFraming:
    def test_frame_is_distinct_from_agent_and_human(self) -> None:
        user_stated = frame_prefix(USER_STATED_PROVENANCE, "finding")
        assert user_stated != frame_prefix("agent", "finding")
        assert user_stated != frame_prefix("human", "finding")

    def test_frame_names_the_binding_and_the_transcriber(self) -> None:
        text = frame_prefix(USER_STATED_PROVENANCE, "finding")
        assert "User-stated" in text
        assert "verbatim excerpt bound" in text
        assert "AI session" in text

    @pytest.mark.parametrize("kind", ["directive", "task", "gotcha", "finding", "reference"])
    def test_same_frame_for_every_kind(self, kind: str) -> None:
        """The unhedged imperative stays reserved for human provenance — a
        bound excerpt states what was checked, it does not command."""
        assert frame_prefix(USER_STATED_PROVENANCE, kind) == frame_prefix(
            USER_STATED_PROVENANCE, "finding"
        )
        assert "DIRECTIVE" not in frame_prefix(USER_STATED_PROVENANCE, kind)

    def test_full_block_marks_the_class_and_frames_the_content(self, tmp_path) -> None:
        """QUOTE is a strict excerpt of CONTENT (wrapped in agent-authored
        framing: 'The user said: "<QUOTE>" -- global python lacks the C
        grammars.'). UPG-PROVENANCE-AUTOBIND-SPAN's attribution fix (the
        span frame) applies to ANY bound excerpt shorter than the whole
        note, not only auto-bound ones -- a stored note carries no record
        of which path (`bind_user_quote` vs `bind_user_quote_auto`) bound
        it, only `user_quote` and `content` themselves, so rendering must
        key on the same whole-vs-partial relationship regardless of how the
        excerpt got bound. Before this fix the explicit path had the exact
        same misattribution bug UPG-PROVENANCE-AUTOBIND-SPAN was written to
        fix for auto-bind: the unqualified frame let "-- global python
        lacks the C grammars." read as user-endorsed when only the quoted
        clause actually was."""
        store = _store(tmp_path)
        ws = str(tmp_path)
        store.remember(ws, CONTENT, user_quote=QUOTE)
        rendered = store.format_notes_for_llm(store.recall(ws), detail="full")
        assert f"[{USER_STATED_PROVENANCE}]" in rendered
        assert "User-stated in part" in rendered
        assert f'"{QUOTE}"' in rendered
        assert "agent-added context" in rendered

    def test_full_block_uses_the_unqualified_frame_when_quote_is_the_whole_body(
        self, tmp_path
    ) -> None:
        """The whole-body case (the pre-UPG-PROVENANCE-AUTOBIND-SPAN
        behavior) is unaffected: when the bound quote IS the entire note
        content, there is no "rest of the note" to disclaim, so the
        original unqualified frame still renders."""
        store = _store(tmp_path)
        ws = str(tmp_path)
        store.remember(ws, QUOTE, user_quote=QUOTE)
        rendered = store.format_notes_for_llm(store.recall(ws), detail="full")
        assert f"[{USER_STATED_PROVENANCE}]" in rendered
        assert "User-stated (verbatim excerpt bound" in rendered
        assert "User-stated in part" not in rendered

    def test_unbound_claim_renders_with_the_agent_frame(self, tmp_path) -> None:
        store = _store(tmp_path)
        ws = str(tmp_path)
        store.remember(ws, "The user definitely told me to do this.", user_quote=QUOTE)
        rendered = store.format_notes_for_llm(store.recall(ws), detail="full")
        assert "User-stated" not in rendered
        assert "[agent]" in rendered


# ---------------------------------------------------------------------------
# Vocabulary + rank drift guards
# ---------------------------------------------------------------------------

class TestVocabularyInvariants:
    def test_user_stated_is_a_valid_stored_value(self) -> None:
        assert USER_STATED_PROVENANCE in PROVENANCE_VALUES

    def test_promotion_ladder_excludes_the_derived_class(self) -> None:
        assert PROMOTION_LADDER == ("auto", "agent", "human")
        assert USER_STATED_PROVENANCE not in PROMOTION_LADDER

    def test_user_stated_ranks_with_agent_on_the_ladder(self) -> None:
        assert PROMOTION_RANK[USER_STATED_PROVENANCE] == PROMOTION_RANK["agent"]
        assert PROMOTION_RANK[USER_STATED_PROVENANCE] < PROMOTION_RANK["human"]

    def test_injection_envelope_rank_is_agent_tier_never_human(self) -> None:
        """The proxy envelope describes who authored the text on the wire —
        still an AI session. Listing it explicitly also keeps a mixed block
        from falling to the weakest tier via the unknown-value default."""
        assert NOTE_PROVENANCE_TRUST_RANK[USER_STATED_PROVENANCE] == (
            NOTE_PROVENANCE_TRUST_RANK["agent"]
        )
        assert NOTE_PROVENANCE_TRUST_RANK[USER_STATED_PROVENANCE] < (
            NOTE_PROVENANCE_TRUST_RANK["human"]
        )


# ---------------------------------------------------------------------------
# MCP surface
# ---------------------------------------------------------------------------

def _mcp_service(*, recent_user_message: str | None = None):
    """A MagicMock VectrService for the MCP dispatch tests below.

    `get_note()` is NOT a fixed stub: `integrations/mcp_server/_dispatch.py`'s
    vectr_remember handler now reads the stored note's OWN provenance back
    (via `service.get_note(note_id)`) to decide whether to render the
    "auto-bound" confirmation message (UPG-PROVENANCE-NEVER-RISES), on top of
    the pre-existing explicit-`user_quote` recompute it already did. A fixed
    `provenance=USER_STATED_PROVENANCE` return, regardless of what a given
    test actually passed to `remember_with_extras`, would make every test
    below spuriously see "auto-bound" -- so `get_note()` instead replays the
    same two-path binding logic (`bind_user_quote` then, on failure,
    `bind_user_quote_auto`) that `WorkingContextStore.remember()` itself runs,
    against the LAST call actually made to `svc.remember_with_extras` -- the
    same derivation, not a second guess at it. `recent_user_message` defaults
    to None (no recently captured user turn), matching a freshly constructed
    mock service with nothing populating its prompt cache; pass it explicitly
    to simulate a harness-captured turn for the auto-bind test below.
    """
    from app.service import RememberOutcome
    from agent.working_context_store import WorkingNote, bind_user_quote_auto

    svc = MagicMock()
    svc.remember_with_extras.return_value = RememberOutcome(
        note_id=7, related=[], revoked_related=[], proxy_anchor_suggestions=[],
    )

    def _get_note(_note_id):
        call = svc.remember_with_extras.call_args
        kwargs = call.kwargs if call is not None else {}
        content = kwargs.get("content", CONTENT)
        note_provenance = kwargs.get("provenance") or "agent"
        explicit_quote = kwargs.get("user_quote")
        bound, _ = bind_user_quote(content, explicit_quote) if explicit_quote else ("", "")
        if not bound:
            bound, _ = bind_user_quote_auto(content, recent_user_message)
        if bound and note_provenance != "human":
            note_provenance = USER_STATED_PROVENANCE
        return WorkingNote(
            note_id=7, workspace="/repo", content=content, tags=[], priority="medium",
            created_at=0.0, last_accessed=0.0, kind="finding", scope="workspace",
            provenance=note_provenance, user_quote=bound,
        )

    svc.get_note.side_effect = _get_note
    svc.search_only = False
    return svc


class TestMcpRememberUserQuote:
    def test_user_quote_is_passed_through_to_the_service(self) -> None:
        from integrations.mcp_server import handle_tools_call

        svc = _mcp_service()
        handle_tools_call(
            "vectr_remember", {"content": CONTENT, "user_quote": QUOTE}, svc,
        )
        assert svc.remember_with_extras.call_args.kwargs["user_quote"] == QUOTE

    def test_bound_quote_is_reported_in_the_confirmation(self) -> None:
        from integrations.mcp_server import handle_tools_call

        result = handle_tools_call(
            "vectr_remember", {"content": CONTENT, "user_quote": QUOTE}, _mcp_service(),
        )
        text = result["content"][0]["text"]
        assert result["isError"] is False
        assert USER_STATED_PROVENANCE in text

    def test_unbound_quote_reports_why_without_failing_the_write(self) -> None:
        from integrations.mcp_server import handle_tools_call

        result = handle_tools_call(
            "vectr_remember",
            {"content": "content that never quotes the user", "user_quote": QUOTE},
            _mcp_service(),
        )
        text = result["content"][0]["text"]
        assert result["isError"] is False
        assert "Stored note #7" in text
        assert "does not appear verbatim" in text

    def test_omitted_quote_leaves_the_confirmation_unchanged(self) -> None:
        from integrations.mcp_server import handle_tools_call

        result = handle_tools_call("vectr_remember", {"content": CONTENT}, _mcp_service())
        text = result["content"][0]["text"]
        assert "user_quote" not in text
        assert USER_STATED_PROVENANCE not in text

    def test_auto_bound_note_reports_auto_bind_in_the_confirmation(self) -> None:
        """UPG-PROVENANCE-NEVER-RISES: no explicit user_quote argument, but
        the harness recently captured a user turn containing this note's
        content verbatim -- the confirmation must say so, distinctly from
        the explicit-user_quote wording asserted in
        `test_bound_quote_is_reported_in_the_confirmation` above."""
        from integrations.mcp_server import handle_tools_call

        recent_turn = f"earlier in this turn the user said: {CONTENT} -- noted."
        result = handle_tools_call(
            "vectr_remember", {"content": CONTENT}, _mcp_service(recent_user_message=recent_turn),
        )
        text = result["content"][0]["text"]
        assert result["isError"] is False
        assert USER_STATED_PROVENANCE in text
        assert "auto-bound" in text

    def test_non_string_user_quote_is_a_caller_error(self) -> None:
        from integrations.mcp_server import handle_tools_call

        svc = _mcp_service()
        result = handle_tools_call(
            "vectr_remember", {"content": CONTENT, "user_quote": 42}, svc,
        )
        assert result["isError"] is True
        svc.remember_with_extras.assert_not_called()

    def test_tool_schema_declares_user_quote(self) -> None:
        from integrations.mcp_server import _MEMORY_WRITE_TOOLS

        remember = next(t for t in _MEMORY_WRITE_TOOLS if t["name"] == "vectr_remember")
        prop = remember["inputSchema"]["properties"]["user_quote"]
        assert prop["type"] == "string"
        assert "user_quote" not in remember["inputSchema"]["required"]

    def test_tool_schema_never_offers_user_stated_as_a_provenance(self) -> None:
        from integrations.mcp_server import _MEMORY_WRITE_TOOLS

        remember = next(t for t in _MEMORY_WRITE_TOOLS if t["name"] == "vectr_remember")
        assert USER_STATED_PROVENANCE not in (
            remember["inputSchema"]["properties"]["provenance"]["enum"]
        )


# ---------------------------------------------------------------------------
# REST surface
# ---------------------------------------------------------------------------

class TestRestRememberUserQuote:
    def test_bound_quote_round_trips_to_user_stated(self, client_real_memory) -> None:
        resp = client_real_memory.post(
            "/v1/remember", json={"content": CONTENT, "user_quote": QUOTE},
        )
        assert resp.status_code == 200
        assert USER_STATED_PROVENANCE in resp.json()["message"]
        recalled = client_real_memory.post(
            "/v1/recall", json={"note_id": resp.json()["note_id"]},
        ).json()["notes"]
        assert f"[{USER_STATED_PROVENANCE}]" in recalled
        # QUOTE is a strict excerpt of CONTENT -- see
        # TestUserStatedFraming.test_full_block_marks_the_class_and_frames_the_content
        # for why the span frame (not the unqualified whole-body frame) is
        # correct here.
        assert "User-stated in part" in recalled
        assert f'"{QUOTE}"' in recalled

    def test_unbound_quote_returns_200_and_the_reason(self, client_real_memory) -> None:
        resp = client_real_memory.post("/v1/remember", json={
            "content": "content that never quotes the user", "user_quote": QUOTE,
        })
        assert resp.status_code == 200
        assert "does not appear verbatim" in resp.json()["message"]
        recalled = client_real_memory.post(
            "/v1/recall", json={"note_id": resp.json()["note_id"]},
        ).json()["notes"]
        assert "[agent]" in recalled
        assert "User-stated" not in recalled

    def test_omitted_quote_leaves_the_message_unchanged(self, client_real_memory) -> None:
        resp = client_real_memory.post("/v1/remember", json={"content": CONTENT})
        assert "user_quote" not in resp.json()["message"]

    def test_auto_bound_note_reports_auto_bind_in_the_message(self, client) -> None:
        """UPG-PROVENANCE-NEVER-RISES at the REST route: no `user_quote` in
        the request body at all, but the stored note (as the mocked
        `get_note` reports it — `client_real_memory`'s closures do not
        thread a harness prompt cache, so this route-level behaviour is
        exercised with a plain mocked service instead, same as the MCP
        surface's `_mcp_service()`) came back at provenance='user-stated' —
        the message must say auto-bound, distinctly from the explicit-bind
        wording covered by `client_real_memory` above."""
        from agent.working_context_store import WorkingNote

        client.app.state.service.get_note.return_value = WorkingNote(
            note_id=1, workspace="/repo", content=CONTENT, tags=[], priority="medium",
            created_at=0.0, last_accessed=0.0, kind="finding", scope="workspace",
            provenance=USER_STATED_PROVENANCE, user_quote=CONTENT,
        )
        resp = client.post("/v1/remember", json={"content": CONTENT})
        assert resp.status_code == 200
        assert USER_STATED_PROVENANCE in resp.json()["message"]
        assert "auto-bound" in resp.json()["message"]

    def test_declared_user_stated_provenance_is_rejected_at_the_schema(self, client) -> None:
        resp = client.post("/v1/remember", json={
            "content": CONTENT, "provenance": USER_STATED_PROVENANCE,
        })
        assert resp.status_code == 422

    def test_promote_route_never_accepts_the_derived_class(self, client) -> None:
        resp = client.post("/v1/promote", json={"note_id": 1, "to": USER_STATED_PROVENANCE})
        assert resp.status_code == 422
