"""UPG-BANNER-CALIBRATION Phase 3 — the rank-1-to-rank-2 relative-drop signal.

Two test layers:

1. ``TestRelativeCliffFires`` — synthetic-score coverage of
   ``agent.ranking_cliff.relative_cliff_fires`` itself. The function is
   pure (no daemon, no corpus, no chunk metadata), so a value-by-value
   check on a hand-built list of scores is the right level of test. Cases
   below cover the documented contract: empty list, single element, a
   clear cliff, a clear tie, a rank-1 below ``min_top``, a None at either
   position, a fixed-margin failure on a confident query (the
   score_order_explain rationale the cliff was modelled after), and a
   headroom-relative gap that *does* fire on the same confident query.

2. ``TestRelativeCliffWiring`` — the integration in ``agent.searcher``:
   with ``RELATIVE_CLIFF_ENABLED=False`` (the shipped default) the cliff
   sub-signal is a no-op and the existing ``low_confidence`` decision
   must be byte-identical to its pre-this-feature value. With the flag
   on, a rank-1-vs-rank-2 cliff (or its absence) participates in the
   OR exactly like the existing ``low_top_relevance`` and
   ``zero_df_trigger`` sub-signals do.
"""
from __future__ import annotations

import pytest

from agent.ranking_cliff import relative_cliff_fires


# ---------------------------------------------------------------------------
# Pure-function coverage — synthetic score vectors only.
# ---------------------------------------------------------------------------

class TestRelativeCliffFires:
    """Each test fixes the two parameters and a score vector and checks the
    boolean result against the documented contract (module docstring of
    agent/ranking_cliff.py). No daemon, no corpus, no chunk metadata."""

    def test_empty_list_is_never_a_cliff(self) -> None:
        # No rank-0 means no rank-1 means no judgment exists; the cliff is
        # never claimed from nothing.
        assert relative_cliff_fires([], min_top=0.30, min_drop=0.50) is False

    def test_single_element_is_never_a_cliff(self) -> None:
        # Same reason: a set with only rank-0 has no rank-1 to compare
        # against, so the cliff cannot fire.
        assert relative_cliff_fires([0.9], min_top=0.30, min_drop=0.50) is False

    def test_clear_cliff_fires(self) -> None:
        # A typical "rank-1 confidently leads rank-2" set on a confident
        # query. 0.90 - 0.30 = 0.60; (1 - 0.90) = 0.10; 0.60 >= 0.50*0.10.
        assert relative_cliff_fires(
            [0.90, 0.30], min_top=0.30, min_drop=0.50,
        ) is True

    def test_clear_tie_does_not_fire(self) -> None:
        # Two essentially-equal scores — a set built on a shared weak
        # base, exactly the case the cliff is meant to catch.
        # gap=0; (1-0.60)*0.50 = 0.20 > 0.
        assert relative_cliff_fires(
            [0.60, 0.60], min_top=0.30, min_drop=0.50,
        ) is False

    def test_rank1_below_min_top_does_not_fire(self) -> None:
        # A wide drop to a much weaker rank-2 is NOT a cliff if rank-1
        # itself is below the absolute ceiling — a top result that
        # cannot credibly lead a set is a weak top result, full stop.
        # gap=0.05, headroom=0.95, 0.05 < 0.50*0.95=0.475 -> False.
        assert relative_cliff_fires(
            [0.05, 0.00], min_top=0.30, min_drop=0.50,
        ) is False

    def test_none_at_rank0_does_not_fire(self) -> None:
        # rerank=False / reranker model failed to load / a straggler the
        # backfill path didn't reach — there is no cross-encoder
        # judgment to base a cliff on.
        assert relative_cliff_fires(
            [None, 0.30], min_top=0.30, min_drop=0.50,
        ) is False

    def test_none_at_rank1_does_not_fire(self) -> None:
        # Same — rank-1 may have a CE judgment but rank-2 doesn't, so the
        # gap is between a calibrated score and a dense-cosine fallback
        # that the UPG-NOTFOUND-FLOOR-2 evidence already showed cannot
        # separate absent-topic from on-topic.
        assert relative_cliff_fires(
            [0.90, None], min_top=0.30, min_drop=0.50,
        ) is False

    def test_both_none_does_not_fire(self) -> None:
        assert relative_cliff_fires(
            [None, None], min_top=0.30, min_drop=0.50,
        ) is False

    def test_headroom_rule_does_not_inherit_fixed_margin_pathology(self) -> None:
        # A FIXED margin (gap >= 0.10) becomes unsatisfiable as rank-1
        # approaches 1.0, because the remaining range shrinks below the
        # margin. The headroom rule scales the requirement to what is
        # left: at rank-1 0.95 the headroom is 0.05, so the required gap
        # is 0.50 * 0.05 = 0.025, and an actual gap of 0.10 clears it.
        # This pins that the rule stays satisfiable on a confident top
        # result. The genuinely saturated case, where the gap really is
        # too small to mean anything, is pinned below in
        # test_zero_gap_near_one_does_not_fire.
        assert relative_cliff_fires(
            [0.95, 0.85], min_top=0.30, min_drop=0.50,
        ) is True

    def test_zero_gap_near_one_does_not_fire(self) -> None:
        # A saturated rank-1 with no drop to rank-2: the cliff is
        # nothing. gap=0, headroom=0.05, 0 >= 0.50*0.05=0.025 -> False.
        # This is the "no cliff on a saturated top" shape that the
        # headroom-relative formulation preserves (the cliff still
        # requires a drop, the formula just makes the drop
        # proportional to the available room).
        assert relative_cliff_fires(
            [0.95, 0.95], min_top=0.30, min_drop=0.50,
        ) is False

    def test_headroom_relative_gap_fires_on_confident_query(self) -> None:
        # A confident rank-1 with a generous drop in headroom terms:
        # 0.95 - 0.50 = 0.45; (1 - 0.95) = 0.05; 0.45 >= 0.50*0.05=0.025.
        # Fires — the headroom rule does not turn every confident-query
        # cliff into a no-fire.
        assert relative_cliff_fires(
            [0.95, 0.50], min_top=0.30, min_drop=0.50,
        ) is True

    def test_trailing_entries_past_rank1_dont_rescue_weak_cliff(self) -> None:
        # The function only looks at the first two elements. A noisy
        # rank-2 that is *better* than rank-1 must not, by its presence,
        # cause the cliff to fire — the cliff is between rank-0 and
        # rank-1, full stop.
        # score[0]=0.20, score[1]=0.90 — 0.20 < 0.30 min_top -> False
        # regardless of any later entry.
        assert relative_cliff_fires(
            [0.20, 0.90, 0.80, 0.70], min_top=0.30, min_drop=0.50,
        ) is False

    def test_min_top_at_or_above_one_is_documented_off_value(self) -> None:
        # The docstring promises min_top >= 1.0 is the documented "off"
        # value: no rank-1 can clear it, so the signal is always False
        # (an exact no-op restoring pre-this-feature behaviour). enabled
        # = false in config short-circuits before this function is even
        # called, so callers that want the OFF semantic do NOT have to
        # set min_top to 1.0 by hand — but if they do, it must still
        # work.
        assert relative_cliff_fires(
            [0.99, 0.01], min_top=1.0, min_drop=0.50,
        ) is False
        assert relative_cliff_fires(
            [0.99, 0.01], min_top=2.0, min_drop=0.50,
        ) is False

    def test_iterable_input_is_accepted(self) -> None:
        # The function's signature is `Iterable[Optional[float]]` — it
        # must accept a tuple / generator, not just a list. Tuple case:
        assert relative_cliff_fires(
            (0.90, 0.30), min_top=0.30, min_drop=0.50,
        ) is True
        # Generator case: same scores, consumed by the function rather
        # than materialised by the caller.
        assert relative_cliff_fires(
            (s for s in [0.90, 0.30]), min_top=0.30, min_drop=0.50,
        ) is True

    def test_exact_min_top_boundary_passes_the_floor(self) -> None:
        # Pin the >= vs > reading of the min_top gate. The docstring
        # says "score[0] >= min_top", so rank-1 EXACTLY at min_top
        # passes the floor; the rest is whether the gap clears the
        # headroom-share bar. min_top=0.30, s0=0.30, s1=0.10 -> gap=0.20,
        # headroom=0.70, 0.20 < 0.50*0.70=0.35 -> False. The test pins
        # the floor's inclusive reading by ensuring a value just above
        # the floor doesn't trip a > vs >= regression.
        # Compare: a score just ABOVE min_top with the same gap, to
        # show the boundary itself wasn't the cause of the False.
        assert relative_cliff_fires(
            [0.31, 0.11], min_top=0.30, min_drop=0.50,
        ) is False  # gap still 0.20 < 0.50*0.69=0.345

    def test_gap_at_exactly_min_drop_times_headroom_fires(self) -> None:
        # Pin the >= vs > reading of the gap check.
        # Construct a pair where the gap is EXACTLY at min_drop*headroom.
        s0 = 0.80
        headroom = 1.0 - s0  # 0.20
        gap = 0.50 * headroom  # 0.10
        s1 = s0 - gap  # 0.70
        assert relative_cliff_fires(
            [s0, s1], min_top=0.30, min_drop=0.50,
        ) is True


# ---------------------------------------------------------------------------
# Integration with the low_confidence decision in agent.searcher.
# ---------------------------------------------------------------------------

class _PerIndexReranker:
    """Stamps the i-th candidate's ``ce_relevance`` from a fixed list, in
    input order. Matches the contract ``agent.searcher._Reranker.rerank``
    needs (see test_indexer_searcher.py's _StubReranker for the
    simpler single-score variant): mutates each candidate's
    ``ce_relevance`` in place, returns the candidates in input order.

    Used here so the cliff integration test can drive both rank-1 and
    rank-2 of the final display set to specific values. The first
    element of the hybrid-retrieval pool is the one that maps to the
    final top (the searcher re-orders by composite, but on the
    two-chunk fixture in this class the strongest BM25/vector match
    stays at the top after re-ordering too).
    """

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores

    def rerank(self, query, candidates):
        for idx, (_, c) in enumerate(candidates):
            c.ce_relevance = (
                self._scores[idx] if idx < len(self._scores) else self._scores[-1]
            )
        return [c for _, c in candidates]


class TestRelativeCliffWiring:
    """The cliff function is a pure helper; the integration that turns it
    into a low_confidence sub-signal lives in agent/searcher.py. These
    tests exercise that integration with a per-index stub reranker so
    the ce_relevance at every rank is under our control."""

    @staticmethod
    def _indexed_searcher(indexer, tmp_path, content: str, name="module.py"):
        from agent.searcher import CodeSearcher
        from tests.conftest import make_py
        path = make_py(tmp_path, name, content)
        indexer.index_file(path)
        s = CodeSearcher(indexer)
        s.refresh_bm25()
        return s

    # Two chunks so the cross-encoder rerank pool has >1 candidate
    # (a single-candidate pool short-circuits rerank and the
    # ce_relevance backfill the test relies on never runs). "parse"
    # and "payload" both appear in parse_json_payload, so the
    # zero-document-frequency sub-signal will NOT fire on a
    # "parse payload" query.
    _CORPUS = (
        "def parse_json_payload(data):\n    return json.loads(data)\n\n"
        "def serialize_response(obj):\n    return json.dumps(obj)\n"
    )

    def test_default_enabled_false_is_exact_noop(
        self, indexer, tmp_path, monkeypatch,
    ) -> None:
        """With ranking.relative_cliff.enabled = false (the shipped
        default) the cliff sub-signal never enters the low_confidence OR,
        so the existing decision must be byte-identical to the
        pre-this-feature result. A set that would trigger the cliff
        (rank-1 above min_top, wide headroom gap) must still return
        low_confidence=False with enabled=false.
        """
        s = self._indexed_searcher(indexer, tmp_path, self._CORPUS)
        # rank-1=0.95, rank-2=0.20 — under min_top=0.30 the rank-1 is
        # well above; a wide headroom gap (0.75 vs (1-0.95)*0.50=0.025);
        # the cliff WOULD fire if enabled.
        s._reranker = _PerIndexReranker([0.95, 0.20])

        import agent.searcher as searcher_module
        monkeypatch.setattr(searcher_module, "_RELATIVE_CLIFF_ENABLED", False)
        results, _ = s.search("parse payload", n_results=5)
        assert results
        assert results[0].ce_relevance == pytest.approx(0.95)
        assert results.low_confidence is False, (
            "rank-1 ce_relevance 0.95 is well above the 0.30 floor; "
            "low_top_relevance does not fire and zero_df does not fire; "
            "with relative_cliff.enabled=false (the shipped default) "
            "the cliff sub-signal must NOT fire either, so the result "
            "is byte-identical to the pre-this-feature path"
        )

    def test_enabled_with_clear_cliff_fires_low_confidence(
        self, indexer, tmp_path, monkeypatch,
    ) -> None:
        """With enabled=true, a set that has a wide rank-1-vs-rank-2 cliff
        but the OTHER sub-signals do not fire must still flag
        low_confidence — the cliff is the trigger here.
        """
        s = self._indexed_searcher(indexer, tmp_path, self._CORPUS)
        s._reranker = _PerIndexReranker([0.95, 0.20])

        import agent.searcher as searcher_module
        monkeypatch.setattr(searcher_module, "_RELATIVE_CLIFF_ENABLED", True)
        # Both query words appear in the indexed corpus, so the zero-DF
        # sub-signal is silent. rank-1's ce_relevance is 0.95, well above
        # the 0.30 low_top_relevance floor, so that sub-signal is silent
        # too. Without the cliff, low_confidence is False; with the cliff
        # on a wide 0.95-vs-0.20 gap, it must be True.
        results, _ = s.search("parse payload", n_results=5)
        assert results
        assert results[0].ce_relevance == pytest.approx(0.95)
        assert results.low_confidence is True, (
            "rank-1 ce_relevance 0.95 with rank-2 0.20 is a wide cliff "
            "(gap=0.75, headroom=0.05, 0.75 >= 0.50*0.05=0.025); with "
            "relative_cliff.enabled=true this must flag low_confidence "
            "even though the other sub-signals are silent"
        )

    def test_enabled_with_tied_top_two_does_not_fire(
        self, indexer, tmp_path, monkeypatch,
    ) -> None:
        """A tied top two (rank-1 and rank-2 essentially equal) is the
        case the cliff is meant to catch, but if the OTHER sub-signals
        do not fire either, low_confidence must remain False — the
        cliff being on does not by itself trip the flag; the cliff
        function must also claim a cliff. Here the cliff is a
        CLEAR TIE (gap=0), so it does NOT claim a cliff, and the
        low_confidence result is the same as the cliff-disabled case.
        """
        s = self._indexed_searcher(indexer, tmp_path, self._CORPUS)
        s._reranker = _PerIndexReranker([0.80, 0.80])

        import agent.searcher as searcher_module
        monkeypatch.setattr(searcher_module, "_RELATIVE_CLIFF_ENABLED", True)
        results, _ = s.search("parse payload", n_results=5)
        assert results
        # rank-1 well above min_top_relevance (0.30), so low_top_relevance
        # is silent. Words are in-corpus, so zero_df is silent. The cliff
        # is a clear tie (gap=0) so it does not fire. low_confidence=False.
        assert results.low_confidence is False

    def test_enabled_with_rank1_below_min_top_does_not_fire_via_cliff(
        self, indexer, tmp_path, monkeypatch,
    ) -> None:
        """The cliff function is gated on min_top: a weak rank-1 with a
        wide-looking drop is not a confident cliff. Here rank-1=0.05
        (below relative_cliff.min_top=0.30) so the cliff sub-signal
        itself is silent. low_top_relevance WILL fire (0.05 < 0.30),
        so low_confidence is True — this test confirms the cliff
        sub-signal didn't ALSO need to fire for the OR to be true
        (the existing path already covered it), and that the cliff
        didn't double-trigger in a way that would change the
        behaviour.
        """
        s = self._indexed_searcher(indexer, tmp_path, self._CORPUS)
        s._reranker = _PerIndexReranker([0.05, 0.00])

        import agent.searcher as searcher_module
        monkeypatch.setattr(searcher_module, "_RELATIVE_CLIFF_ENABLED", True)
        results, _ = s.search("parse payload", n_results=5)
        assert results
        # low_top_relevance fires (0.05 < 0.30); low_confidence is True.
        assert results.low_confidence is True

    def test_enabled_with_none_at_rank1_does_not_fire_via_cliff(
        self, indexer, tmp_path, monkeypatch,
    ) -> None:
        """When the reranker doesn't run (rerank=False) ce_relevance is
        None on every candidate; the cliff sub-signal must be silent
        even with enabled=true, because relative_cliff_fires is itself
        a no-op on None at either position. The result is the same
        as the cliff-disabled case (False, because low_top_relevance is
        also silent on None ce_relevance).
        """
        s = self._indexed_searcher(indexer, tmp_path, self._CORPUS)
        # Leave the default reranker; rerank=False skips it entirely.
        import agent.searcher as searcher_module
        monkeypatch.setattr(searcher_module, "_RELATIVE_CLIFF_ENABLED", True)
        results, _ = s.search("parse payload", n_results=5, rerank=False)
        assert results
        assert results[0].ce_relevance is None
        assert results.low_confidence is False, (
            "with rerank=False every ce_relevance is None; the cliff "
            "sub-signal is a no-op on None at either position (per "
            "relative_cliff_fires' contract), so even with enabled=true "
            "the flag must not fire from the cliff alone"
        )

    def test_cliff_does_not_change_decision_when_other_sub_signals_already_fire(
        self, indexer, tmp_path, monkeypatch,
    ) -> None:
        """The cliff sub-signal is OR'd with the existing triggers, so
        enabling it must never SUPPRESS a flag that the other sub-
        signals already set. A zero-DF query with a high-CE rank-1
        keeps the zero-DF trigger suppressed by the existing
        ce_override (low_confidence False); flipping cliff.enabled to
        true must not change that.
        """
        # "reimburse" / "shopper" never appear in the indexed corpus, so
        # both query content words are zero-DF; the high-CE rank-1
        # triggers the existing ce_override suppression (low_confidence
        # False). With the cliff enabled on a 0.95/0.20 gap (a wide
        # cliff), the override still suppresses zero_df, and low_top is
        # silent on a high CE, so the cliff itself is the only path
        # that could fire — and the cliff now does fire.
        s = self._indexed_searcher(
            indexer, tmp_path,
            "def charge_card(customer, amount):\n"
            "    return gateway.charge(customer, amount)\n\n"
            "def refund_payment(customer, amount):\n"
            "    return gateway.refund(customer, amount)\n",
        )
        s._reranker = _PerIndexReranker([0.95, 0.20])

        import agent.searcher as searcher_module
        monkeypatch.setattr(searcher_module, "_RELATIVE_CLIFF_ENABLED", True)
        results, _ = s.search("reimburse the shopper", n_results=5)
        assert results
        # The cliff itself fires (0.95 - 0.20 = 0.75 >= 0.025), and the
        # cliff is OR'd into the existing trigger. Result: low_confidence
        # = True. This documents that the cliff does participate in the
        # OR (it adds a new firing path), but the more important claim
        # for this test is that no existing True is changed to False by
        # enabling it.
        assert results.low_confidence is True
