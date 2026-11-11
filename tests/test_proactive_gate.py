"""Gating policy tests (UPG-PRO-5): floor, budget, dedup/cooldown, determinism."""
from __future__ import annotations

from agent.config import PROACTIVE_MAX_WEAK_STRUCTURAL_ITEMS
from agent.proactive.gate import LedgerStore, ProactiveGate, SessionLedger
from agent.proactive.types import (
    STRUCTURAL_TIER_DECLARED_ANCHOR,
    STRUCTURAL_TIER_GOTCHA_MENTION,
    STRUCTURAL_TIER_MENTION,
    Candidate,
)


def _cand(kind, line, score, anchor, structural, state="active", structural_tier=None,
          anchor_path=None, path_mention_count=0, path_mention_first_offset=-1):
    return Candidate(
        kind=kind, line=line, score=score, anchor_id=anchor, is_structural=structural,
        state=state, structural_tier=structural_tier, anchor_path=anchor_path,
        path_mention_count=path_mention_count,
        path_mention_first_offset=path_mention_first_offset,
    )


def _gate(**kw):
    defaults = dict(
        min_similarity=0.35, max_items_per_event=3, max_chars_per_event=800, cooldown_items=30,
        max_weak_structural_items=PROACTIVE_MAX_WEAK_STRUCTURAL_ITEMS,
    )
    defaults.update(kw)
    return ProactiveGate(**defaults)


def test_budget_takes_top_k_deterministically():
    cands = [
        _cand("code_semantic", f"hit {i}", 0.9 - i * 0.01, f"chunk:{i}", False)
        for i in range(10)
    ]
    gate = _gate(max_items_per_event=3)
    out = gate.select(cands, session_id="s1")
    assert out.item_count == 3
    # Top-3 by score desc, deterministic.
    assert out.anchor_ids == ("chunk:0", "chunk:1", "chunk:2")


def test_identical_input_identical_output():
    cands = [
        _cand("note_semantic", "note a", 0.8, "note:1", False),
        _cand("code_semantic", "hit b", 0.7, "chunk:x", False),
    ]
    a = _gate().select(list(cands), session_id="")
    b = _gate().select(list(cands), session_id="")
    assert a == b


def test_floor_applies_to_structural_candidates_too():
    # UPG-PROXY-LOOPBACK-BYPASS / similarity-floor bypass: structural exact
    # matches no longer skip the floor. A structural candidate BELOW the
    # floor is dropped exactly like a semantic one; a structural candidate
    # AT the floor (1.0, the score every exact match carries) still passes —
    # this is behavior-preserving for genuine structural matches.
    cands = [
        _cand("note_semantic", "weak", 0.20, "note:1", False),          # below 0.35 floor
        _cand("note_structural", "below-floor", 0.20, "note:2", True),  # structural below floor: dropped
        _cand("note_structural", "at-floor", 1.0, "note:3", True),      # structural at 1.0: clears floor
    ]
    out = _gate().select(cands, session_id="")
    assert out.item_count == 1
    assert out.anchor_ids == ("note:3",)


def test_floor_above_one_admits_nothing_even_structural():
    # Fail-closed: a floor configured above the maximum possible score
    # admits nothing, including an exact structural match.
    cands = [_cand("note_structural", "struct", 1.0, "note:1", True)]
    out = _gate(min_similarity=1.01).select(cands, session_id="")
    assert out.is_empty()


def test_structural_outranks_semantic_on_tie():
    cands = [
        _cand("note_semantic", "sem", 1.0, "note:9", False),
        _cand("note_structural", "struct", 1.0, "note:1", True),
    ]
    out = _gate(max_items_per_event=2).select(cands, session_id="")
    # Same score => provenance rank breaks tie: structural (0) before note (1).
    assert out.anchor_ids[0] == "note:1"


def test_dedup_cooldown_suppresses_repeat():
    gate = _gate()
    c = [_cand("note_structural", "x", 1.0, "note:5", True)]
    first = gate.select(list(c), session_id="sess")
    second = gate.select(list(c), session_id="sess")
    assert first.item_count == 1
    assert second.item_count == 0  # already emitted within cooldown


def test_dedup_across_matchers_keeps_highest_score():
    cands = [
        _cand("note_semantic", "low", 0.5, "note:1", False),
        _cand("note_structural", "high", 1.0, "note:1", True),  # same anchor
    ]
    out = _gate().select(cands, session_id="")
    assert out.item_count == 1
    assert out.scores == (1.0,)


def test_structural_only_drops_semantic():
    cands = [
        _cand("note_semantic", "sem", 0.99, "note:1", False),
        _cand("note_structural", "struct", 1.0, "note:2", True),
    ]
    out = _gate().select(cands, session_id="", structural_only=True)
    assert out.anchor_ids == ("note:2",)


def test_empty_selection_returns_empty_string():
    out = _gate().select([], session_id="")
    assert out.is_empty()
    assert out.context == ""


def test_char_budget_stops_overflow():
    long_line = "x" * 100
    cands = [
        _cand("note_semantic", long_line, 0.9, "note:1", False),
        _cand("note_semantic", long_line, 0.8, "note:2", False),
        _cand("note_semantic", long_line, 0.7, "note:3", False),
    ]
    # header + 100 + \n + 100 = fits 2 but not 3 within 260.
    out = _gate(max_chars_per_event=260, max_items_per_event=5).select(cands, session_id="")
    assert out.item_count == 2


def test_session_ledger_bounded_ring():
    led = SessionLedger(capacity=2)
    led.add("a")
    led.add("b")
    led.add("c")  # evicts "a"
    assert not led.seen("a")
    assert led.seen("b")
    assert led.seen("c")


def test_ledger_store_lru_bounded_sessions():
    store = LedgerStore(cooldown_items=5, max_sessions=2)
    store.record("s1", ["a"])
    store.record("s2", ["b"])
    store.record("s3", ["c"])  # evicts s1 (LRU)
    assert not store.seen("s1", "a")
    assert store.seen("s3", "c")


# -- cross-channel turn-ledger dedup (UPG-PROXY-CROSS-CHANNEL-DEDUP) --------
# Uses the REAL `TurnInjectionLedger` (not a stand-in) — the same object the
# hook/trigger-engine surfaces share via `VectrService._turn_ledger_for`.

def test_note_already_claimed_this_turn_is_dropped():
    from agent.trigger_engine import TurnInjectionLedger

    turn_ledger = TurnInjectionLedger()
    turn_ledger.claim(1)  # another surface already injected note #1 this turn
    cands = [
        _cand("note_structural", "struct", 1.0, "note:1", True),
        _cand("note_semantic", "sem", 0.9, "note:2", False),
    ]
    out = _gate().select(cands, session_id="", turn_ledger=turn_ledger)
    assert out.anchor_ids == ("note:2",)


def test_turn_ledger_absent_is_a_no_op():
    cands = [_cand("note_structural", "struct", 1.0, "note:1", True)]
    out = _gate().select(cands, session_id="")  # turn_ledger omitted entirely
    assert out.item_count == 1


def test_selected_notes_are_claimed_in_turn_ledger():
    from agent.trigger_engine import TurnInjectionLedger

    turn_ledger = TurnInjectionLedger()
    cands = [_cand("note_structural", "struct", 1.0, "note:1", True)]
    out = _gate().select(cands, session_id="", turn_ledger=turn_ledger)
    assert out.item_count == 1
    assert turn_ledger.eligible(1) is False  # claimed on the way out


def test_budget_evicted_note_is_not_claimed_in_turn_ledger():
    from agent.trigger_engine import TurnInjectionLedger

    turn_ledger = TurnInjectionLedger()
    cands = [
        _cand("note_structural", "kept", 1.0, "note:1", True),
        _cand("note_structural", "evicted", 0.9, "note:2", True),
    ]
    gate = _gate(max_items_per_event=1)  # only room for one item
    out = gate.select(cands, session_id="", turn_ledger=turn_ledger)
    assert out.anchor_ids == ("note:1",)
    assert turn_ledger.eligible(1) is False  # actually delivered: claimed
    assert turn_ledger.eligible(2) is True   # matched but budget-evicted: NOT claimed


def test_chunk_candidates_unaffected_by_turn_ledger():
    from agent.trigger_engine import TurnInjectionLedger

    turn_ledger = TurnInjectionLedger()
    cands = [_cand("code_semantic", "hit", 0.9, "chunk:foo.py:1-9", False)]
    out = _gate().select(cands, session_id="", turn_ledger=turn_ledger)
    assert out.item_count == 1  # non-note anchors have no note_id to check


# -- UPG-PROXY-AUDIT-DURABLE: states carried through to InjectionResult -----

def test_selected_states_positionally_aligned_with_anchor_ids():
    cands = [
        _cand("note_structural", "kept", 1.0, "note:1", True, state="revoked"),
        _cand("note_semantic", "also kept", 0.9, "note:2", False, state="active"),
    ]
    out = _gate(max_items_per_event=2).select(cands, session_id="")
    assert out.anchor_ids == ("note:1", "note:2")
    assert out.states == ("revoked", "active")


def test_empty_result_has_empty_states_tuple():
    out = _gate().select([], session_id="")
    assert out.states == ()


def test_budget_evicted_candidate_state_not_carried_into_result():
    cands = [
        _cand("note_structural", "kept", 1.0, "note:1", True, state="active"),
        _cand("note_structural", "evicted", 0.9, "note:2", True, state="revoked"),
    ]
    out = _gate(max_items_per_event=1).select(cands, session_id="")
    assert out.anchor_ids == ("note:1",)
    assert out.states == ("active",)


# -- lever 3: max_weak_structural_items cap (UPG-PROXY-INJECT-PRECISION) ----
# `structural_tier` is a property of the CANDIDATE (computed by the matcher
# from note.kind / is_declared_anchor), never a read of query content — the
# gate only counts it.



def test_weak_structural_cap_limits_tier_c_items():
    """With a budget of 5 items but a weak-item cap of 1, only the first
    (highest-scoring) Tier-C ("mention") candidate is admitted -- the rest
    are skipped, leaving room for whatever else clears the floor."""
    cands = [
        _cand("note_structural", f"weak {i}", 0.60 - i * 0.001, f"note:{i}", True,
              structural_tier=STRUCTURAL_TIER_MENTION)
        for i in range(5)
    ]
    out = _gate(max_items_per_event=5, max_weak_structural_items=1).select(
        cands, session_id=""
    )
    assert out.item_count == 1
    assert out.anchor_ids == ("note:0",)  # the single highest-scoring weak item


def test_weak_structural_cap_does_not_limit_stronger_tiers():
    """The cap only counts STRUCTURAL_TIER_MENTION candidates -- declared
    anchors and gotcha mentions are unaffected regardless of count."""
    cands = [
        _cand("note_structural", "anchor a", 1.0, "note:1", True,
              structural_tier=STRUCTURAL_TIER_DECLARED_ANCHOR),
        _cand("note_structural", "anchor b", 1.0, "note:2", True,
              structural_tier=STRUCTURAL_TIER_DECLARED_ANCHOR),
        _cand("note_structural", "gotcha a", 0.9, "note:3", True,
              structural_tier=STRUCTURAL_TIER_GOTCHA_MENTION),
    ]
    out = _gate(max_items_per_event=5, max_weak_structural_items=0).select(
        cands, session_id=""
    )
    assert out.item_count == 3


def test_weak_structural_cap_leaves_room_for_a_later_stronger_candidate():
    """A capped Tier-C item's slot is not wasted: the budget loop continues
    past it (rather than stopping), so a later, stronger candidate in
    score-descending order still gets admitted."""
    cands = [
        _cand("note_structural", "weak 1", 0.60, "note:1", True,
              structural_tier=STRUCTURAL_TIER_MENTION),
        _cand("note_structural", "weak 2", 0.59, "note:2", True,
              structural_tier=STRUCTURAL_TIER_MENTION),
        _cand("note_semantic", "sem", 0.50, "note:3", False),
    ]
    out = _gate(max_items_per_event=5, max_weak_structural_items=1).select(
        cands, session_id=""
    )
    assert out.anchor_ids == ("note:1", "note:3")  # note:2 skipped by the cap


def test_weak_structural_cap_and_kind_filter_never_claim_in_turn_ledger():
    """Ledger rule pin (UPG-PROXY-INJECT-PRECISION): an item dropped by
    EITHER lever 1 (kind-eligibility -- simulated here as never becoming a
    Candidate at all, matching _ServiceMatchSource.structural_notes()'s
    filter-after-return contract) or lever 3 (the weak-item cap) must never
    be claimed in the cross-channel TurnInjectionLedger, using the REAL
    ledger from agent/trigger_engine.py -- not a stand-in."""
    from agent.trigger_engine import TurnInjectionLedger

    turn_ledger = TurnInjectionLedger()
    # note:1 and note:2 are BOTH Tier-C ("mention"); the cap admits only one.
    # note:99, standing in for a note lever 1 would have filtered out before
    # a Candidate ever existed, is deliberately absent from `cands` — there
    # is nothing lever 1 could claim, by construction (see
    # app/service.py::_ServiceMatchSource.structural_notes()).
    cands = [
        _cand("note_structural", "weak 1", 0.60, "note:1", True,
              structural_tier=STRUCTURAL_TIER_MENTION),
        _cand("note_structural", "weak 2", 0.59, "note:2", True,
              structural_tier=STRUCTURAL_TIER_MENTION),
    ]
    out = _gate(max_items_per_event=5, max_weak_structural_items=1).select(
        cands, session_id="", turn_ledger=turn_ledger,
    )
    assert out.anchor_ids == ("note:1",)
    assert turn_ledger.eligible(1) is False  # delivered: claimed
    assert turn_ledger.eligible(2) is True   # capped by lever 3: NEVER claimed


# -- UPG-PROXY-INJECT-SINGLE-TURN: event-anchored retirement ----------------
#
# A declared-anchor structural candidate (the strongest tier) is delivered
# every request its anchor path keeps matching, but only RETIRES from the
# cross-turn cooldown ledger once a request's `edited_file_paths` shows its
# own anchor path was genuinely edited — not merely read/mentioned. This is
# the mechanism behind the fix: a note anchored to a file the agent has only
# read so far stays eligible to resurface right up to the edit decision,
# instead of retiring after its first (possibly premature) delivery.

def test_declared_anchor_note_stays_eligible_across_calls_until_its_file_is_edited():
    gate = _gate()
    cand = [_cand("note_structural", "gotcha for resolver.py", 1.0, "note:7", True,
                   structural_tier=STRUCTURAL_TIER_DECLARED_ANCHOR,
                   anchor_path="resolver.py")]

    # Delivered on the first call; the file was only matched, not edited.
    first = gate.select(list(cand), session_id="sess")
    assert first.item_count == 1
    assert first.unretired_anchor_ids == ("note:7",)

    # A SECOND request that still only reads resolver.py: delivered AGAIN —
    # the plain SessionLedger cooldown that would normally suppress a repeat
    # (see test_dedup_cooldown_suppresses_repeat above) does not apply here.
    second = gate.select(list(cand), session_id="sess")
    assert second.item_count == 1
    assert second.unretired_anchor_ids == ("note:7",)

    # A THIRD request whose window shows resolver.py actually being edited:
    # delivered one final time, and now retires.
    third = gate.select(
        list(cand), session_id="sess", edited_file_paths=frozenset({"resolver.py"}),
    )
    assert third.item_count == 1
    assert third.unretired_anchor_ids == ()  # charged this time

    # A FOURTH request, identical to the others: now suppressed by the
    # ordinary cooldown ledger, exactly like any other retired item.
    fourth = gate.select(list(cand), session_id="sess")
    assert fourth.item_count == 0


def test_declared_anchor_note_retires_immediately_when_its_own_file_is_already_edited():
    gate = _gate()
    cand = [_cand("note_structural", "gotcha", 1.0, "note:8", True,
                   structural_tier=STRUCTURAL_TIER_DECLARED_ANCHOR,
                   anchor_path="resolver.py")]
    first = gate.select(
        list(cand), session_id="sess", edited_file_paths=frozenset({"resolver.py"}),
    )
    assert first.item_count == 1
    assert first.unretired_anchor_ids == ()
    second = gate.select(list(cand), session_id="sess")
    assert second.item_count == 0  # already retired on the first (edit) call


def test_event_anchored_retirement_is_scoped_to_declared_anchor_tier_only():
    """A weaker structural tier (gotcha-mention / mention) is unaffected even
    when it happens to carry an `anchor_path` -- the gate only exempts
    STRUCTURAL_TIER_DECLARED_ANCHOR from charging, matching the same
    evidence tier the hook channel's own pre-edit trigger bundle is built
    from (agent.trigger_engine.default_bundle_for_kind)."""
    gate = _gate()
    cand = [_cand("note_structural", "weak mention", 0.6, "note:9", True,
                   structural_tier=STRUCTURAL_TIER_MENTION, anchor_path="resolver.py")]
    first = gate.select(list(cand), session_id="sess")
    assert first.item_count == 1
    assert first.unretired_anchor_ids == ()  # charged normally, tier-scoped exemption skipped
    second = gate.select(list(cand), session_id="sess")
    assert second.item_count == 0  # ordinary cooldown suppresses the repeat


def test_declared_anchor_note_without_anchor_path_charges_normally():
    """Backward-compat guard: a hand-built declared-anchor candidate that
    never sets `anchor_path` (every pre-existing test/caller in this file)
    keeps the plain charge-on-delivery behaviour -- event-anchored
    retirement only activates when `anchor_path` is populated."""
    gate = _gate()
    cand = [_cand("note_structural", "no anchor_path set", 1.0, "note:10", True,
                   structural_tier=STRUCTURAL_TIER_DECLARED_ANCHOR)]
    first = gate.select(list(cand), session_id="sess")
    assert first.unretired_anchor_ids == ()
    second = gate.select(list(cand), session_id="sess")
    assert second.item_count == 0


def test_event_anchored_exempt_item_still_claimed_in_turn_ledger_same_turn():
    """The per-turn cross-channel dedup ledger is unaffected by event-anchored
    retirement: an unretired item WAS delivered this turn, so it must still
    be claimed there to suppress a hook surface delivering the same note
    again in the same turn (UPG-PROXY-CROSS-CHANNEL-DEDUP stays additive)."""
    from agent.trigger_engine import TurnInjectionLedger

    turn_ledger = TurnInjectionLedger()
    cand = [_cand("note_structural", "gotcha", 1.0, "note:11", True,
                   structural_tier=STRUCTURAL_TIER_DECLARED_ANCHOR,
                   anchor_path="resolver.py")]
    out = _gate().select(list(cand), session_id="sess", turn_ledger=turn_ledger)
    assert out.unretired_anchor_ids == ("note:11",)
    assert turn_ledger.eligible(11) is False  # still claimed this turn
    assert turn_ledger.eligible(99) is True  # kind-filtered by lever 1 (never
                                              # even a Candidate): NEVER claimed


# -- UPG-PROXY-WEAK-TIER-TIEBREAK: relevance-bearing equal-score tie-break ---
#
# Every Tier-C candidate carries the SAME score (structural_scores.mention),
# so the per-event weak-item cap admitted whichever equal-score candidate the
# previous tie-break chain (provenance rank, then anchor_id — both irrelevant
# to aboutness) happened to surface; measured 16/20 off-topic, with two input
# orderings producing 71.9% vs 100% strict precision. The gate's step-5 sort
# now breaks EQUAL scores by how hard each note talks about its file (basename
# path-boundary mentions), then by where the first mention lands — subjects
# name their file repeatedly and early; passing references don't.

def test_weak_cap_tiebreak_prefers_the_candidate_that_mentions_its_file_more():
    cands = [
        _cand("note_structural", "passing mention", 0.60, "note:1", True,
              structural_tier=STRUCTURAL_TIER_MENTION,
              path_mention_count=1, path_mention_first_offset=40),
        _cand("note_structural", "subject note", 0.60, "note:2", True,
              structural_tier=STRUCTURAL_TIER_MENTION,
              path_mention_count=3, path_mention_first_offset=0),
    ]
    out = _gate(max_items_per_event=5, max_weak_structural_items=1).select(
        list(cands), session_id=""
    )
    # Equal scores: the count signal decides, NOT anchor_id order.
    assert out.anchor_ids == ("note:2",)


def test_weak_cap_tiebreak_at_equal_count_prefers_the_earlier_first_mention():
    cands = [
        _cand("note_structural", "mid-list mention", 0.60, "note:1", True,
              structural_tier=STRUCTURAL_TIER_MENTION,
              path_mention_count=1, path_mention_first_offset=31),
        _cand("note_structural", "opens with the file", 0.60, "note:2", True,
              structural_tier=STRUCTURAL_TIER_MENTION,
              path_mention_count=1, path_mention_first_offset=13),
    ]
    out = _gate(max_items_per_event=5, max_weak_structural_items=1).select(
        list(cands), session_id=""
    )
    assert out.anchor_ids == ("note:2",)


def test_weak_cap_tiebreak_is_input_order_independent():
    """The measured failure mode itself: two ORDERINGS of the same candidates
    used to select different items (71.9% vs 100% strict precision). The
    selection must now be a pure function of the candidate SET."""
    def cand(note_id, count, offset):
        return _cand("note_structural", f"weak {note_id}", 0.60, f"note:{note_id}",
                     True, structural_tier=STRUCTURAL_TIER_MENTION,
                     path_mention_count=count, path_mention_first_offset=offset)

    one_way = [cand(1, 1, 31), cand(2, 1, 13), cand(3, 2, 8)]
    other_way = list(reversed(one_way))
    kw = dict(max_items_per_event=5, max_weak_structural_items=1)
    a = _gate(**kw).select(list(one_way), session_id="")
    b = _gate(**kw).select(list(other_way), session_id="")
    assert a.anchor_ids == b.anchor_ids == ("note:3",)


def test_tiebreak_signals_are_inert_for_handbuilt_candidates():
    """Backward-compat guard: every pre-existing caller/test builds Candidates
    without the new fields (defaults count=0 / offset=-1). Those constants
    fall every equal-score comparison through to provenance rank then
    anchor_id — byte-identical to the pre-change ordering."""
    cands = [
        _cand("note_structural", "b", 0.60, "note:2", True,
              structural_tier=STRUCTURAL_TIER_MENTION),
        _cand("note_structural", "a", 0.60, "note:1", True,
              structural_tier=STRUCTURAL_TIER_MENTION),
    ]
    out = _gate(max_items_per_event=5, max_weak_structural_items=2).select(
        list(cands), session_id=""
    )
    assert out.anchor_ids == ("note:1", "note:2")  # anchor_id tie-break, as before


def test_tiebreak_never_promotes_a_lower_scored_candidate():
    """The signals are tie-breaks ONLY: they never compare against the floor
    and never fold into score, so a higher-scoring candidate from ANY tier
    still outranks a more-mentioning weaker one."""
    cands = [
        _cand("note_structural", "weak but chatty", 0.60, "note:1", True,
              structural_tier=STRUCTURAL_TIER_MENTION,
              path_mention_count=9, path_mention_first_offset=0),
        _cand("note_semantic", "semantic", 0.61, "note:2", False),
    ]
    out = _gate(max_items_per_event=5, max_weak_structural_items=1).select(
        list(cands), session_id=""
    )
    assert out.anchor_ids == ("note:2", "note:1")


# -- UPG-PROXY-COOLDOWN-NO-TIME-DECAY: TTL on the cooldown ring --------------
#
# The proxy channel's session key is process-scoped (UPG-PROXY-COOLDOWN-
# SESSION-IDENTITY), so "last 30" spans the whole process lifetime and a note
# that becomes genuinely relevant again hours later stays suppressed until 30
# OTHER anchors cycle through. Entries can now expire after a wall-clock TTL.
# All clocks here are injected fakes — no test sleeps.


class _FakeClock:
    def __init__(self, start=1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_session_ledger_ttl_expires_entry_after_window():
    clock = _FakeClock()
    led = SessionLedger(capacity=30, ttl_seconds=60.0, now_fn=clock)
    led.add("a")
    assert led.seen("a")
    clock.advance(30)
    assert led.seen("a")   # inside the window: still suppressed
    clock.advance(31)      # past the window boundary: expired
    assert not led.seen("a")


def test_session_ledger_without_ttl_is_pure_count_ring():
    """ttl_seconds=None (the historical behaviour, and what every pre-existing
    direct construction gets) never time-expires, however long the clock
    runs — suppression ends only when capacity evicts."""
    clock = _FakeClock()
    led = SessionLedger(capacity=2, ttl_seconds=None, now_fn=clock)
    led.add("a")
    led.add("b")
    clock.advance(10**9)
    assert led.seen("a") is True and led.seen("b") is True
    led.add("c")           # capacity eviction, not time, removes "a"
    assert not led.seen("a")
    assert led.seen("b") and led.seen("c")


def test_session_ledger_refresh_extends_suppression_from_most_recent_delivery():
    clock = _FakeClock()
    led = SessionLedger(capacity=30, ttl_seconds=60.0, now_fn=clock)
    led.add("a")
    clock.advance(50)
    led.add("a")           # re-delivered: suppression restarts from NOW
    clock.advance(50)
    assert led.seen("a")   # only 50s since the most recent delivery
    clock.advance(10)
    assert not led.seen("a")


def test_capacity_bound_still_applies_on_top_of_the_ttl():
    """The TTL only ever EXPIRES entries early; it never extends retention
    past the ring."""
    clock = _FakeClock()
    led = SessionLedger(capacity=2, ttl_seconds=3600.0, now_fn=clock)
    led.add("a")
    led.add("b")
    led.add("c")           # evicts "a" by capacity even though its TTL runs
    assert not led.seen("a")
    assert led.seen("b") and led.seen("c")


def test_ttl_expires_entries_regardless_of_ring_position():
    clock = _FakeClock()
    led = SessionLedger(capacity=3, ttl_seconds=60.0, now_fn=clock)
    led.add("old")
    clock.advance(30)
    led.add("new")
    clock.advance(35)      # "old" is 65s past delivery, "new" only 35s
    assert not led.seen("old")
    assert led.seen("new")
    # The ring was rebuilt around the survivor: capacity accounting continues
    # from the LIVE entries only.
    led.add("x")
    led.add("y")           # ring now exactly full: new,x,y (capacity 3)
    assert led.seen("new") and led.seen("x") and led.seen("y")
    led.add("z")           # evicts "new", the oldest-inserted LIVE entry
    assert not led.seen("new")
    assert led.seen("x") and led.seen("y") and led.seen("z")


def test_ledger_store_threads_ttl_and_clock_to_every_session():
    clock = _FakeClock()
    store = LedgerStore(cooldown_items=5, ttl_seconds=60.0, now_fn=clock)
    store.record("s1", ["a"])
    store.record("s2", ["b"])
    assert store.seen("s1", "a") and store.seen("s2", "b")
    clock.advance(61)
    assert not store.seen("s1", "a")
    assert not store.seen("s2", "b")


def test_proactive_gate_threads_cooldown_ttl_into_its_own_ledger(monkeypatch):
    """Wiring pin: ProactiveGate(cooldown_ttl_seconds=...) must reach the
    LedgerStore it constructs internally (app/service.py passes the settings
    knob straight through)."""
    import agent.proactive.gate as gate_mod

    captured = {}
    real_store = gate_mod.LedgerStore

    def spy(cooldown_items, **kw):
        captured.update(kw)
        return real_store(cooldown_items, **kw)

    monkeypatch.setattr(gate_mod, "LedgerStore", spy)
    _gate(cooldown_ttl_seconds=90)
    assert captured.get("ttl_seconds") == 90

    captured.clear()
    _gate()  # default: no TTL (historical behaviour for direct constructions)
    assert captured.get("ttl_seconds") is None
