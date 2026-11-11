"""Matching-engine tests (UPG-PRO-4). Fakes return the REAL WorkingNote /
SearchResult types the daemon would return."""
from __future__ import annotations

import time

from agent.config import (
    PROACTIVE_STRUCTURAL_SCORE_DECLARED_ANCHOR,
    PROACTIVE_STRUCTURAL_SCORE_DECLARED_TRIGGER,
    PROACTIVE_STRUCTURAL_SCORE_GOTCHA_MENTION,
    PROACTIVE_STRUCTURAL_SCORE_MENTION,
)
from agent.proactive.matcher import ProactiveMatcher
from agent.proactive.types import (
    STRUCTURAL_TIER_DECLARED_ANCHOR,
    STRUCTURAL_TIER_DECLARED_TRIGGER,
    STRUCTURAL_TIER_GOTCHA_MENTION,
    STRUCTURAL_TIER_MENTION,
    ProactiveWindow,
)
from agent.searcher import SearchResult
from agent.working_context_store._types import WorkingNote


def _note(note_id, content, kind="finding", title="", triggers=None):
    return WorkingNote(
        note_id=note_id, workspace="/ws", content=content, tags=[], priority="medium",
        created_at=time.time(), last_accessed=time.time(), kind=kind, title=title,
        triggers=triggers,
    )


class _Source:
    """Deliberately does NOT implement note_states() -- the pre-UPG-PROXY-
    REVOKED-LEAK MatchSource shape. Every test using this fake exercises
    the backward-compat degrade-to-{} path (a missing note_states() call
    means every note is treated as active)."""

    def __init__(self, structural=None, semantic=None, code=None):
        self._structural = structural or []
        self._semantic = semantic or []
        self._code = code or []
        self.calls = []

    def structural_notes(self, file_paths):
        self.calls.append(("structural", tuple(file_paths)))
        return list(self._structural)

    def semantic_notes(self, text, min_similarity, limit):
        self.calls.append(("semantic", text))
        return [(n, s) for (n, s) in self._semantic if s >= min_similarity]

    def code_search(self, text, n_results):
        self.calls.append(("code", text))
        return list(self._code)


class _StatefulSource(_Source):
    """Same as _Source but implements note_states() -- used by revoked/
    superseded lifecycle tests."""

    def __init__(self, structural=None, semantic=None, code=None, states=None):
        super().__init__(structural=structural, semantic=semantic, code=code)
        self._states = states or {}

    def note_states(self, notes):
        self.calls.append(("note_states", tuple(n.note_id for n in notes)))
        return dict(self._states)


def _matcher(source, **kw):
    defaults = dict(
        min_similarity=0.35, max_chars_per_event=800,
        structural_note=True, semantic_note=True, code_search=True,
        structural_score_declared_anchor=PROACTIVE_STRUCTURAL_SCORE_DECLARED_ANCHOR,
        structural_score_declared_trigger=PROACTIVE_STRUCTURAL_SCORE_DECLARED_TRIGGER,
        structural_score_gotcha_mention=PROACTIVE_STRUCTURAL_SCORE_GOTCHA_MENTION,
        structural_score_mention=PROACTIVE_STRUCTURAL_SCORE_MENTION,
    )
    defaults.update(kw)
    return ProactiveMatcher(source, **defaults)


def test_structural_note_scores_one_and_anchors():
    """UPG-PROXY-SUBSTRING-ANCHOR/2b: this note has no DECLARED anchor --
    it only matched because window file_paths' basename ("resolver.py")
    appears in its content. That is a "mentions" claim, not "anchored
    to" -- see test_structural_note_declared_anchor_says_anchored_to for
    the genuine declared-anchor case.

    UPG-PROXY-INJECT-PRECISION lever 2: a plain content-mention match on a
    non-gotcha kind ("finding" here) is the weakest evidence tier (Tier C
    / STRUCTURAL_TIER_MENTION) and scores PROACTIVE_STRUCTURAL_SCORE_MENTION
    -- no longer the flat 1.0 every structural match used to get regardless
    of how it was found."""
    n = _note(1, "gotcha: resolver.py lock drops on scope exit")
    src = _Source(structural=[n])
    w = ProactiveWindow(text="", file_paths=["/abs/resolver.py"], symbols=[])
    cands = _matcher(src, semantic_note=False, code_search=False).match(w)
    assert len(cands) == 1
    c = cands[0]
    assert c.kind == "note_structural" and c.is_structural
    assert c.score == PROACTIVE_STRUCTURAL_SCORE_MENTION
    assert c.structural_tier == STRUCTURAL_TIER_MENTION
    assert "mentions resolver.py" in c.line
    assert c.anchor_id == "note:1"


def test_structural_note_declared_anchor_says_anchored_to():
    """A note whose declared `anchors` column actually contains the
    window file gets the stronger "anchored to X" wording -- even when
    its prose content never spells the filename out. UPG-PROXY-INJECT-
    PRECISION lever 2: a declared anchor is the strongest evidence tier
    (Tier A) and scores PROACTIVE_STRUCTURAL_SCORE_DECLARED_ANCHOR."""
    n = _note(12, "the backoff cap here needs tuning")
    n.anchors = [["resolver.py", None]]
    src = _Source(structural=[n])
    w = ProactiveWindow(text="", file_paths=["/abs/resolver.py"], symbols=[])
    cands = _matcher(src, semantic_note=False, code_search=False).match(w)
    assert len(cands) == 1
    c = cands[0]
    assert "anchored to resolver.py" in c.line
    assert "mentions" not in c.line
    assert c.score == PROACTIVE_STRUCTURAL_SCORE_DECLARED_ANCHOR
    assert c.structural_tier == STRUCTURAL_TIER_DECLARED_ANCHOR


def test_declared_anchor_candidate_carries_the_window_file_path():
    """UPG-PROXY-INJECT-SINGLE-TURN: a STRUCTURAL_TIER_DECLARED_ANCHOR
    candidate carries the exact `window.file_paths` entry it matched on
    `Candidate.anchor_path` -- the FULL path, not the basename `_first_anchor`
    uses for display -- so the gate can test it against a later request's
    `edited_file_paths` for event-anchored retirement."""
    n = _note(13, "the backoff cap here needs tuning")
    n.anchors = [["resolver.py", None]]
    src = _Source(structural=[n])
    w = ProactiveWindow(text="", file_paths=["/abs/resolver.py"], symbols=[])
    cands = _matcher(src, semantic_note=False, code_search=False).match(w)
    assert len(cands) == 1
    assert cands[0].anchor_path == "/abs/resolver.py"


def test_weaker_structural_tiers_never_carry_anchor_path():
    """A content-mention match (no declared anchor), on either a gotcha or
    a plain note, leaves `anchor_path` unset -- event-anchored retirement
    is scoped to declared-anchor evidence only (see gate.py's `select()`)."""
    plain = _note(14, "gotcha: resolver.py lock drops on scope exit")
    gotcha = _note(15, "resolver.py: the retry timeout is 30 seconds", kind="gotcha")
    src = _Source(structural=[plain, gotcha])
    w = ProactiveWindow(text="", file_paths=["/abs/resolver.py"], symbols=[])
    cands = _matcher(src, semantic_note=False, code_search=False).match(w)
    assert len(cands) == 2
    assert all(c.anchor_path is None for c in cands)


def test_structural_note_gotcha_mention_is_middle_tier():
    """UPG-PROXY-INJECT-PRECISION lever 2: a content-mention match (no
    declared anchor) on a kind="gotcha" note is Tier B -- stronger than a
    plain mention on any other kind, weaker than a declared anchor,
    scoring strictly between PROACTIVE_STRUCTURAL_SCORE_MENTION and
    PROACTIVE_STRUCTURAL_SCORE_DECLARED_ANCHOR."""
    n = _note(7, "resolver.py: the retry timeout is 30 seconds", kind="gotcha")
    src = _Source(structural=[n])
    w = ProactiveWindow(text="", file_paths=["/abs/resolver.py"], symbols=[])
    cands = _matcher(src, semantic_note=False, code_search=False).match(w)
    assert len(cands) == 1
    c = cands[0]
    assert c.score == PROACTIVE_STRUCTURAL_SCORE_GOTCHA_MENTION
    assert c.structural_tier == STRUCTURAL_TIER_GOTCHA_MENTION
    assert (
        PROACTIVE_STRUCTURAL_SCORE_MENTION
        < c.score
        < PROACTIVE_STRUCTURAL_SCORE_DECLARED_ANCHOR
    )


# -- UPG-TRIGGERS-INERT-ON-PROXY-STRUCTURAL: declared trigger-glob tier -----

def test_structural_note_declared_trigger_says_triggers_on():
    """A note whose EXPLICIT `triggers[]` declares a 'path' glob matching
    the window file gets the "triggers on X" wording, scores
    PROACTIVE_STRUCTURAL_SCORE_DECLARED_TRIGGER, and is tagged
    STRUCTURAL_TIER_DECLARED_TRIGGER -- even though it has no declared
    `anchors` entry and its content never mentions the filename (non-
    vacuity: this can only be the trigger-glob path, not the anchor or
    mention path)."""
    n = _note(20, "the backoff cap here needs tuning", triggers=[{"path": "resolver.py"}])
    src = _Source(structural=[n])
    w = ProactiveWindow(text="", file_paths=["/abs/resolver.py"], symbols=[])
    cands = _matcher(src, semantic_note=False, code_search=False).match(w)
    assert len(cands) == 1
    c = cands[0]
    assert "triggers on resolver.py" in c.line
    assert "anchored to" not in c.line
    assert "mentions" not in c.line
    assert c.score == PROACTIVE_STRUCTURAL_SCORE_DECLARED_TRIGGER
    assert c.structural_tier == STRUCTURAL_TIER_DECLARED_TRIGGER


def test_declared_trigger_ranks_between_anchor_and_gotcha_mention():
    """UPG-PROXY-WEAK-TIER-TIEBREAK: a declared trigger glob is an explicit
    author declaration, the same evidentiary class as an anchor -- it must
    not be laundered through the weak `mention` score band (16/20 admitted
    weak-tier items there were off-topic). Ranked below a plain anchor
    (a glob can match many files; an anchor names exactly one) but strictly
    above gotcha_mention/mention."""
    n = _note(21, "the backoff cap here needs tuning", triggers=[{"path": "resolver.py"}])
    src = _Source(structural=[n])
    w = ProactiveWindow(text="", file_paths=["/abs/resolver.py"], symbols=[])
    c = _matcher(src, semantic_note=False, code_search=False).match(w)[0]
    assert (
        PROACTIVE_STRUCTURAL_SCORE_GOTCHA_MENTION
        < c.score
        < PROACTIVE_STRUCTURAL_SCORE_DECLARED_ANCHOR
    )


def test_declared_trigger_glob_wildcard_matches_via_matcher():
    n = _note(22, "package-wide caveat", triggers=[{"path": "*.py"}])
    src = _Source(structural=[n])
    w = ProactiveWindow(text="", file_paths=["/abs/resolver.py"], symbols=[])
    cands = _matcher(src, semantic_note=False, code_search=False).match(w)
    assert len(cands) == 1
    assert cands[0].structural_tier == STRUCTURAL_TIER_DECLARED_TRIGGER


def test_declared_trigger_wildcard_renders_the_matching_glob():
    """UPG-MATCHER-RELATIVE-ANCHOR-LABEL: the provenance label names the
    form that ACTUALLY matched. A '*.py' glob must render as
    "triggers on *.py", not as the concrete basename it happened to hit —
    before this fix the line claimed "triggers on resolver.py", overstating
    a package-wide declaration as a single-file one."""
    n = _note(41, "package-wide caveat", triggers=[{"path": "*.py"}])
    src = _Source(structural=[n])
    w = ProactiveWindow(text="", file_paths=["/abs/resolver.py"], symbols=[])
    cands = _matcher(src, semantic_note=False, code_search=False).match(w)
    assert len(cands) == 1
    assert "triggers on *.py" in cands[0].line


def test_full_path_declared_anchor_renders_the_declared_string():
    """UPG-MATCHER-RELATIVE-ANCHOR-LABEL: an anchor authored as the exact
    window path renders as itself ("anchored to /abs/resolver.py"), not as
    the basename the match never used."""
    n = _note(42, "the backoff cap here needs tuning")
    n.anchors = [["/abs/resolver.py", None]]
    src = _Source(structural=[n])
    w = ProactiveWindow(text="", file_paths=["/abs/resolver.py"], symbols=[])
    cands = _matcher(src, semantic_note=False, code_search=False).match(w)
    assert len(cands) == 1
    assert "anchored to /abs/resolver.py" in cands[0].line
    assert cands[0].anchor_path == "/abs/resolver.py"


def test_basename_and_full_path_both_declared_still_render_basename():
    """UPG-MATCHER-RELATIVE-ANCHOR-LABEL preserves the pre-change label when
    BOTH forms are declared: only the case where the basename was never
    declared switched to reporting the matched string."""
    n = _note(43, "the backoff cap here needs tuning")
    n.anchors = [["resolver.py", None], ["/abs/resolver.py", None]]
    src = _Source(structural=[n])
    w = ProactiveWindow(text="", file_paths=["/abs/resolver.py"], symbols=[])
    cands = _matcher(src, semantic_note=False, code_search=False).match(w)
    assert len(cands) == 1
    assert "anchored to resolver.py" in cands[0].line


def test_tiebreak_signals_stay_keyed_to_the_basename():
    """UPG-MATCHER-RELATIVE-ANCHOR-LABEL must be label-only: a note whose
    prose names resolver.py twice keeps counting 2 mentions even though its
    label now renders a wildcard glob, so gate.py's equal-score ordering
    inputs are byte-for-byte what they were before the label change."""
    n = _note(44, "resolver.py lock drops on scope exit; see resolver.py docs",
              triggers=[{"path": "*.py"}])
    src = _Source(structural=[n])
    w = ProactiveWindow(text="", file_paths=["/abs/resolver.py"], symbols=[])
    c = _matcher(src, semantic_note=False, code_search=False).match(w)[0]
    assert "triggers on *.py" in c.line
    assert c.path_mention_count == 2
    assert c.path_mention_first_offset == 0


def test_declared_anchor_still_wins_over_a_declared_trigger_on_the_same_note():
    """When a note declares BOTH an exact anchor and a (broader) trigger
    glob for the same file, the stronger anchor relation wins -- matching
    `_first_anchor()`'s declared-order precedence (anchor checked first)."""
    n = _note(23, "the backoff cap here needs tuning", triggers=[{"path": "*.py"}])
    n.anchors = [["resolver.py", None]]
    src = _Source(structural=[n])
    w = ProactiveWindow(text="", file_paths=["/abs/resolver.py"], symbols=[])
    c = _matcher(src, semantic_note=False, code_search=False).match(w)[0]
    assert c.structural_tier == STRUCTURAL_TIER_DECLARED_ANCHOR
    assert c.score == PROACTIVE_STRUCTURAL_SCORE_DECLARED_ANCHOR


def test_declared_trigger_candidate_does_not_carry_anchor_path():
    """Event-anchored single-turn retirement (UPG-PROXY-INJECT-SINGLE-TURN)
    stays scoped to declared-anchor evidence only -- a declared_trigger
    candidate is a weaker tier and must not participate."""
    n = _note(24, "the backoff cap here needs tuning", triggers=[{"path": "resolver.py"}])
    src = _Source(structural=[n])
    w = ProactiveWindow(text="", file_paths=["/abs/resolver.py"], symbols=[])
    c = _matcher(src, semantic_note=False, code_search=False).match(w)[0]
    assert c.anchor_path is None


# -- UPG-PROXY-WEAK-TIER-TIEBREAK: relevance-bearing equal-score signals -----
#
# Every Tier-C candidate carries the SAME flat score (structural_scores.
# mention), so the gate's per-event weak-item cap was admitting whichever
# candidate an irrelevant input-order tie-break happened to surface first
# (16/20 admitted items measured off-topic). The matcher now stamps each
# structural candidate with HOW HARD its note talks about the matched file —
# path-boundary basename mentions, and where the first one lands — as
# tie-break-ONLY signals (they never touch score or the floor; see
# agent/proactive/gate.py's step-5 sort).

def test_mention_candidate_carries_mention_count_and_first_offset():
    n = _note(30, "resolver.py lock drops on scope exit; see resolver.py docs too")
    src = _Source(structural=[n])
    w = ProactiveWindow(text="", file_paths=["/abs/resolver.py"], symbols=[])
    c = _matcher(src, semantic_note=False, code_search=False).match(w)[0]
    assert c.structural_tier == STRUCTURAL_TIER_MENTION
    # The subject names its file twice, starting at character 0.
    assert c.path_mention_count == 2
    assert c.path_mention_first_offset == 0


def test_gotcha_mention_candidate_carries_the_same_signals_midprose():
    n = _note(31, "watch out near resolver.py: the retry timeout is 30s", kind="gotcha")
    src = _Source(structural=[n])
    w = ProactiveWindow(text="", file_paths=["/abs/resolver.py"], symbols=[])
    c = _matcher(src, semantic_note=False, code_search=False).match(w)[0]
    assert c.structural_tier == STRUCTURAL_TIER_GOTCHA_MENTION
    # "watch out near " is 15 characters.
    assert c.path_mention_count == 1
    assert c.path_mention_first_offset == 15


def test_silent_anchored_note_keeps_the_inert_defaults():
    """A declared anchor whose prose never spells the filename out has no
    honest basename occurrence to count — the signals stay at their inert
    defaults (count 0, offset -1) rather than being fabricated."""
    n = _note(32, "the backoff cap here needs tuning")
    n.anchors = [["resolver.py", None]]
    src = _Source(structural=[n])
    w = ProactiveWindow(text="", file_paths=["/abs/resolver.py"], symbols=[])
    c = _matcher(src, semantic_note=False, code_search=False).match(w)[0]
    assert c.structural_tier == STRUCTURAL_TIER_DECLARED_ANCHOR
    assert c.path_mention_count == 0
    assert c.path_mention_first_offset == -1


def test_non_structural_candidates_never_carry_tiebreak_signals():
    r = SearchResult(file_path="a.py", lines="1-2", symbol_name="", language="python",
                     score=0.6, content="code")
    src = _Source(semantic=[(_note(33, "workspace lock flow"), 0.8)], code=[r])
    w = ProactiveWindow(text="how does the workspace lock work")
    cands = _matcher(src, structural_note=False).match(w)
    assert {c.kind for c in cands} == {"note_semantic", "code_semantic"}
    for c in cands:
        assert c.kind != "note_structural"
        assert c.path_mention_count == 0
        assert c.path_mention_first_offset == -1


def test_semantic_note_respects_floor():
    src = _Source(semantic=[(_note(2, "workspace lock flow"), 0.8), (_note(3, "off topic"), 0.10)])
    w = ProactiveWindow(text="how does the workspace lock work")
    cands = _matcher(src, structural_note=False, code_search=False).match(w)
    ids = {c.anchor_id for c in cands}
    assert "note:2" in ids
    assert "note:3" not in ids  # below floor, dropped by the source's threshold


def test_code_search_candidates():
    r = SearchResult(file_path="resolver.py", lines="10-20", symbol_name="lock",
                     language="python", score=0.72, content="def lock():\n    ...")
    src = _Source(code=[r])
    w = ProactiveWindow(text="lock acquisition")
    cands = _matcher(src, structural_note=False, semantic_note=False).match(w)
    assert len(cands) == 1
    assert cands[0].kind == "code_semantic"
    assert cands[0].anchor_id == "chunk:resolver.py:10-20"
    assert cands[0].score == 0.72


def test_all_matchers_run_unconditionally():
    n = _note(1, "resolver.py note")
    r = SearchResult(file_path="a.py", lines="1-2", symbol_name="", language="python",
                     score=0.6, content="code")
    src = _Source(structural=[n], semantic=[(_note(2, "sem"), 0.9)], code=[r])
    w = ProactiveWindow(text="lock", file_paths=["/x/resolver.py"], symbols=[])
    cands = _matcher(src).match(w)
    kinds = {c.kind for c in cands}
    assert kinds == {"note_structural", "note_semantic", "code_semantic"}
    # All three matchers were consulted regardless of window content.
    call_kinds = {c[0] for c in src.calls}
    assert call_kinds == {"structural", "semantic", "code"}


def test_disabled_matcher_not_called():
    src = _Source(semantic=[(_note(2, "sem"), 0.9)])
    w = ProactiveWindow(text="lock", file_paths=["/x/resolver.py"])
    _matcher(src, structural_note=False, code_search=False).match(w)
    # Only the semantic matcher ran; disabling is a static toggle, not a content read.
    assert {c[0] for c in src.calls} == {"semantic"}


def test_source_error_is_swallowed():
    class _Boom:
        def structural_notes(self, fp):
            raise RuntimeError("db down")

        def semantic_notes(self, t, m, l):
            raise RuntimeError("embed down")

        def code_search(self, t, n):
            raise RuntimeError("index down")

    w = ProactiveWindow(text="lock", file_paths=["/x/a.py"])
    cands = _matcher(_Boom()).match(w)
    assert cands == []  # a failing source degrades to no candidates, never raises


def test_source_without_note_states_degrades_gracefully_to_active():
    """A MatchSource that predates note_states() (an older fake or backend)
    must not break -- a missing note_states() call is treated as every
    note being active, matching note_event_states()'s own contract for a
    note_id absent from the fold."""
    n = _note(10, "some finding")
    src = _Source(structural=[n])  # _Source deliberately has no note_states
    w = ProactiveWindow(text="", file_paths=["/abs/resolver.py"], symbols=[])
    cands = _matcher(src, semantic_note=False, code_search=False).match(w)
    assert len(cands) == 1
    assert "REVOKED" not in cands[0].line


def test_note_states_called_once_for_the_union_of_matched_notes():
    """note_states() must be called exactly once per match(), for the
    UNION of every note either the structural or semantic matcher found
    -- never once per note, never once per matcher."""
    shared = _note(20, "shared note matched by both matchers")
    structural_only = _note(21, "structural only")
    src = _StatefulSource(
        structural=[shared, structural_only],
        semantic=[(shared, 0.9)],
    )
    w = ProactiveWindow(text="shared note", file_paths=["/abs/resolver.py"], symbols=[])
    _matcher(src).match(w)
    note_states_calls = [c for c in src.calls if c[0] == "note_states"]
    assert len(note_states_calls) == 1
    assert set(note_states_calls[0][1]) == {20, 21}


class TestRevokedNoteDeterrentRendering:
    """UPG-PROXY-REVOKED-LEAK: a revoked note must never inject its raw
    content as active fact -- it still injects (catching an agent about
    to re-assert a wrong belief is the highest-value injection), just
    rendered through the anti-memory deterrent instead."""

    def test_structural_revoked_note_renders_deterrent_first(self):
        n = _note(30, "the retry timeout is 30 seconds", kind="gotcha")
        w = ProactiveWindow(text="", file_paths=["/abs/resolver.py"], symbols=[])

        # Non-vacuity: the SAME note, while still active, surfaces its raw
        # content verbatim -- proving the match itself is real, not just
        # an artifact of the revoked-state rendering path.
        active_cands = _matcher(
            _Source(structural=[n]), semantic_note=False, code_search=False
        ).match(w)
        assert len(active_cands) == 1
        assert "the retry timeout is 30 seconds" in active_cands[0].line

        states = {30: {"state": "revoked", "reason": "wrong assumption",
                       "actor": "agent", "ts": time.time()}}
        revoked_cands = _matcher(
            _StatefulSource(structural=[n], states=states),
            semantic_note=False, code_search=False,
        ).match(w)
        assert len(revoked_cands) == 1
        line = revoked_cands[0].line
        assert "REVOKED" in line
        assert "Do not re-derive" in line
        # The deterrent clause is placed FIRST so _cap() truncation can
        # never remove it while leaving the raw claim still legible.
        assert line.index("Do not re-derive") < line.index("Previously believed")

    def test_semantic_revoked_note_renders_deterrent_first(self):
        n = _note(31, "cache eviction runs every 5 minutes", kind="gotcha")
        w = ProactiveWindow(text="how does cache eviction work")

        active_cands = _matcher(
            _Source(semantic=[(n, 0.9)]), structural_note=False, code_search=False
        ).match(w)
        assert len(active_cands) == 1
        assert "cache eviction runs every 5 minutes" in active_cands[0].line

        states = {31: {"state": "revoked", "reason": "outdated",
                       "actor": "agent", "ts": time.time()}}
        revoked_cands = _matcher(
            _StatefulSource(semantic=[(n, 0.9)], states=states),
            structural_note=False, code_search=False,
        ).match(w)
        assert len(revoked_cands) == 1
        line = revoked_cands[0].line
        assert "REVOKED" in line
        assert line.index("Do not re-derive") < line.index("Previously believed")

    def test_truncation_preserves_deterrent_over_raw_quote(self):
        """A tight per-event char budget must cut the tail-positioned raw
        quoted claim before it ever cuts the front-positioned deterrent
        warning."""
        n = _note(32, "a very specific raw claim about default timeout values", kind="gotcha")
        w = ProactiveWindow(text="", file_paths=["/abs/resolver.py"], symbols=[])
        states = {32: {"state": "revoked", "reason": "wrong",
                       "actor": "agent", "ts": time.time()}}

        # Uncapped line, to locate exactly where the deterrent clause ends.
        uncapped = _matcher(
            _StatefulSource(structural=[n], states=states),
            semantic_note=False, code_search=False, max_chars_per_event=10_000,
        ).match(w)
        full_line = uncapped[0].line
        deterrent_end = full_line.index("without verification.") + len("without verification.")
        assert full_line.index("Do not re-derive") < full_line.index(
            '"a very specific raw claim'
        )

        # A budget that fits the deterrent clause but not the raw quote.
        tight = _matcher(
            _StatefulSource(structural=[n], states=states),
            semantic_note=False, code_search=False, max_chars_per_event=deterrent_end + 5,
        ).match(w)
        tight_line = tight[0].line
        assert "Do not re-derive this from other sources without verification." in tight_line
        assert "a very specific raw claim about default timeout values" not in tight_line

    def test_superseded_note_reaching_renderer_is_skipped(self):
        """Superseded notes are normally excluded via valid_until before
        ever reaching the matcher; this is a defensive check that if one
        still reaches here, it is skipped rather than rendered as an
        active fact."""
        n = _note(33, "old cache size was 128mb")
        w = ProactiveWindow(text="", file_paths=["/abs/resolver.py"], symbols=[])
        states = {33: {"state": "superseded", "reason": None,
                       "actor": "agent", "ts": time.time()}}
        cands = _matcher(
            _StatefulSource(structural=[n], states=states),
            semantic_note=False, code_search=False,
        ).match(w)
        assert cands == []


# -- UPG-PROXY-REVOKED-LEAK: unknown lifecycle state must fail CLOSED -------

class _FailingStateSource(_Source):
    """Implements note_states() but the call FAILS -- e.g. the note_events
    table is unreadable mid-migration. Distinct from _Source, which does
    not implement the method at all."""

    def note_states(self, notes):
        raise RuntimeError("note_events unreadable")


def test_note_states_failure_drops_note_candidates_instead_of_rendering_raw():
    """A note_states() that RAISES leaves lifecycle state unknown. Unknown
    must never be rendered as active: a revoked note would otherwise have
    its raw content injected as apparent fact, which is exactly the defect
    UPG-PROXY-REVOKED-LEAK fixes. Fail closed -- drop the note candidates.

    Non-vacuity: the identical source shape whose note_states() SUCCEEDS
    (returning no state, i.e. all-active) does render the note, proving the
    match itself is real and the drop is caused by the failure alone."""
    n = _note(1, "resolver.py holds the workspace lock")
    window = ProactiveWindow(text="workspace lock", file_paths=["/x/resolver.py"])

    failing = _matcher(_FailingStateSource(structural=[n], semantic=[(n, 0.9)]))
    assert [c for c in failing.match(window) if c.kind.startswith("note_")] == []

    working = _matcher(_StatefulSource(structural=[n], semantic=[(n, 0.9)], states={}))
    assert [c for c in working.match(window) if c.kind.startswith("note_")] != []


def test_note_states_failure_still_yields_code_candidates():
    """Failing closed drops NOTE candidates only. M4 code-search hits carry
    no note lifecycle, so they must survive -- the caller degrades to code
    context rather than losing the whole injection."""
    n = _note(1, "resolver.py holds the workspace lock")
    hit = SearchResult(
        file_path="/x/resolver.py", lines="1-4", symbol_name="lock",
        language="python", score=0.9, content="def lock():\n    ...",
    )
    src = _FailingStateSource(structural=[n], semantic=[(n, 0.9)], code=[hit])
    kinds = {c.kind for c in _matcher(src).match(
        ProactiveWindow(text="workspace lock", file_paths=["/x/resolver.py"])
    )}
    assert "code_semantic" in kinds
    assert not any(k.startswith("note_") for k in kinds)


# -- UPG-PROXY-AUDIT-DURABLE: Candidate.state carries the folded lifecycle --
# state through to the gate/audit line (previously computed here and thrown
# away after rendering).

class TestCandidateStateCarriage:
    def test_note_with_no_state_entry_is_active(self):
        """note_event_states()'s documented contract: a note_id absent from
        the fold is active. The Candidate.state carried for it must say so
        explicitly, not just render raw content and leave state implicit."""
        n = _note(40, "some finding")
        src = _StatefulSource(structural=[n], states={})
        w = ProactiveWindow(text="", file_paths=["/abs/resolver.py"], symbols=[])
        cands = _matcher(src, semantic_note=False, code_search=False).match(w)
        assert len(cands) == 1
        assert cands[0].state == "active"

    def test_source_without_note_states_defaults_candidate_state_to_active(self):
        """Same default for the backward-compat path: a MatchSource that
        predates note_states() entirely still carries an honest "active"
        state, never a blank or missing value."""
        n = _note(41, "some finding")
        src = _Source(structural=[n])  # no note_states() at all
        w = ProactiveWindow(text="", file_paths=["/abs/resolver.py"], symbols=[])
        cands = _matcher(src, semantic_note=False, code_search=False).match(w)
        assert len(cands) == 1
        assert cands[0].state == "active"

    def test_revoked_note_candidate_carries_revoked_state(self):
        n = _note(42, "the retry timeout is 30 seconds", kind="gotcha")
        states = {42: {"state": "revoked", "reason": "wrong assumption",
                       "actor": "agent", "ts": time.time()}}
        src = _StatefulSource(structural=[n], states=states)
        w = ProactiveWindow(text="", file_paths=["/abs/resolver.py"], symbols=[])
        cands = _matcher(src, semantic_note=False, code_search=False).match(w)
        assert len(cands) == 1
        assert cands[0].state == "revoked"
        assert "REVOKED" in cands[0].line  # rendering and state agree

    def test_semantic_note_candidate_carries_state_too(self):
        n = _note(43, "cache eviction runs every 5 minutes", kind="gotcha")
        states = {43: {"state": "revoked", "reason": "outdated",
                       "actor": "agent", "ts": time.time()}}
        src = _StatefulSource(semantic=[(n, 0.9)], states=states)
        w = ProactiveWindow(text="how does cache eviction work")
        cands = _matcher(src, structural_note=False, code_search=False).match(w)
        assert len(cands) == 1
        assert cands[0].state == "revoked"

    def test_code_search_candidate_state_is_not_applicable_not_active(self):
        """A code-search hit has no note lifecycle. Labeling it "active"
        would dishonestly imply lifecycle tracking that does not exist for
        it -- it must carry the distinct, explicit not-applicable value."""
        from agent.proactive.types import CANDIDATE_STATE_NOT_APPLICABLE

        r = SearchResult(file_path="resolver.py", lines="10-20", symbol_name="lock",
                         language="python", score=0.72, content="def lock():\n    ...")
        src = _Source(code=[r])
        w = ProactiveWindow(text="lock acquisition")
        cands = _matcher(src, structural_note=False, semantic_note=False).match(w)
        assert len(cands) == 1
        assert cands[0].state == CANDIDATE_STATE_NOT_APPLICABLE
        assert cands[0].state != "active"

    def test_fail_closed_drop_leaves_surviving_code_candidate_state_intact(self):
        """When note_states() raises, note candidates are dropped entirely
        (see test_note_states_failure_still_yields_code_candidates); the
        surviving code candidate's state must still be the honest
        not-applicable value, unaffected by the note-lifecycle failure."""
        from agent.proactive.types import CANDIDATE_STATE_NOT_APPLICABLE

        n = _note(1, "resolver.py holds the workspace lock")
        hit = SearchResult(
            file_path="/x/resolver.py", lines="1-4", symbol_name="lock",
            language="python", score=0.9, content="def lock():\n    ...",
        )
        src = _FailingStateSource(structural=[n], semantic=[(n, 0.9)], code=[hit])
        cands = _matcher(src).match(
            ProactiveWindow(text="workspace lock", file_paths=["/x/resolver.py"])
        )
        code_cands = [c for c in cands if c.kind == "code_semantic"]
        assert len(code_cands) == 1
        assert code_cands[0].state == CANDIDATE_STATE_NOT_APPLICABLE


# -- UPG-PROXY-INJECT-TITLE-ONLY: the injected line must carry the note's
# BODY, not just its title -- the root cause LANE-UTILITY-2 measured behind
# the 0/7 proxy-injection utility null (a titled note injected only its
# title; the actual guidance in `content` never crossed the wire).

class TestBodyInclusion:
    def test_titled_note_injects_body_not_just_title(self):
        n = _note(
            60,
            "must not be used by new code, whatever the README says; "
            "ship events one at a time instead",
            title="Acme send_batch drops events behind our gzipping proxy (ACME-8891)",
        )
        src = _Source(structural=[n])
        w = ProactiveWindow(text="", file_paths=["/abs/resolver.py"], symbols=[])
        cands = _matcher(src, semantic_note=False, code_search=False, max_chars_per_event=800).match(w)
        assert len(cands) == 1
        line = cands[0].line
        # The title still orients the reader...
        assert "Acme send_batch drops events behind our gzipping proxy" in line
        # ...but the actual guidance (previously dropped entirely) is present.
        assert "ship events one at a time instead" in line

    def test_untitled_note_still_injects_its_body_unchanged(self):
        """Non-vacuity / regression guard: a note with no title (the common
        vectr_remember() call) behaved correctly before this fix and must
        keep doing so -- this path is untouched."""
        n = _note(61, "resolver.py holds the workspace lock; drops on scope exit")
        src = _Source(structural=[n])
        w = ProactiveWindow(text="", file_paths=["/abs/resolver.py"], symbols=[])
        cands = _matcher(src, semantic_note=False, code_search=False).match(w)
        assert len(cands) == 1
        assert "resolver.py holds the workspace lock; drops on scope exit" in cands[0].line

    def test_title_that_duplicates_the_content_opening_is_not_repeated(self):
        """`remember()`'s own title FALLBACK derives an empty title from the
        first content line -- so a note whose title is a verbatim prefix of
        its body (the common shape once callers start passing an explicit
        title) must not render as "title: title body...", just the body."""
        n = _note(
            62, "the retry limit in widget.py is 3, not 5 as the README claims",
            title="the retry limit in widget.py is 3, not 5 as the README claims",
        )
        src = _Source(structural=[n])
        w = ProactiveWindow(text="", file_paths=["/abs/widget.py"], symbols=[])
        cands = _matcher(src, semantic_note=False, code_search=False).match(w)
        assert len(cands) == 1
        line = cands[0].line
        assert line.count("retry limit in widget.py is 3") == 1

    def test_semantic_channel_also_carries_body(self):
        n = _note(
            63, "verify the ticket before trusting this — measured wrong twice before",
            title="rate limit is per-key not per-ip",
        )
        src = _Source(semantic=[(n, 0.9)])
        w = ProactiveWindow(text="what is the rate limit keyed on")
        cands = _matcher(src, structural_note=False, code_search=False).match(w)
        assert len(cands) == 1
        assert "verify the ticket before trusting this" in cands[0].line

    def test_long_body_truncates_at_a_word_boundary_not_mid_word(self):
        """A budget too small for the full body must not chop mid-word --
        UPG-PROXY-INJECT-TITLE-ONLY's truncate-whole-note requirement."""
        long_body = (
            "the connection pool must be sized to the upstream service's own "
            "concurrency limit or requests silently queue forever without any "
            "visible error in the logs"
        )
        n = _note(64, long_body)
        src = _Source(structural=[n])
        w = ProactiveWindow(text="", file_paths=["/abs/pool.py"], symbols=[])
        cands = _matcher(
            src, semantic_note=False, code_search=False, max_chars_per_event=70,
        ).match(w)
        assert len(cands) == 1
        line = cands[0].line
        assert line.endswith("…")
        before_ellipsis = line[:-1].rstrip()
        # The character immediately before the ellipsis ends a whole word
        # (or sentence) -- never a bare fragment like "concurrenc".
        assert before_ellipsis[-1] not in (" ",)
        last_word = before_ellipsis.rsplit(" ", 1)[-1].rstrip(".")
        assert long_body.split() and any(
            w.startswith(last_word) or last_word == w for w in long_body.split()
        ), f"truncation produced a mid-word fragment: {last_word!r}"

    def test_revoked_deterrent_rendering_is_unaffected_by_body_inclusion(self):
        """W1 scope guard: the revoked-note deterrent path (_revoked_summary)
        must render exactly as before -- it never calls _raw_summary at all,
        so title/body joining must not leak into it."""
        n = _note(65, "the retry timeout is 30 seconds", kind="gotcha",
                   title="retry timeout gotcha")
        states = {65: {"state": "revoked", "reason": "wrong", "actor": "agent",
                       "ts": time.time()}}
        src = _StatefulSource(structural=[n], states=states)
        w = ProactiveWindow(text="", file_paths=["/abs/resolver.py"], symbols=[])
        cands = _matcher(src, semantic_note=False, code_search=False).match(w)
        assert len(cands) == 1
        line = cands[0].line
        assert "REVOKED" in line
        assert "Do not re-derive" in line
        assert line.index("Do not re-derive") < line.index("Previously believed")

    def test_provenance_label_still_present_alongside_body(self):
        """Regression guard: provenance labels (auto/agent/human) must
        survive body inclusion unchanged."""
        n = _note(66, "the cache ttl is 300 seconds not 60", title="cache ttl")
        n.provenance = "human"
        n.anchors = [["cache.py", None]]
        src = _Source(structural=[n])
        w = ProactiveWindow(text="", file_paths=["/abs/cache.py"], symbols=[])
        cands = _matcher(src, semantic_note=False, code_search=False).match(w)
        assert len(cands) == 1
        assert "(finding, human, anchored to cache.py)" in cands[0].line
        assert "the cache ttl is 300 seconds not 60" in cands[0].line
