"""UPG-BANNER-CALIBRATION Phase 3 — the rank-1-to-rank-2 relative-drop signal.

A pure-Python function that tests whether the top result's cross-encoder
relevance (ce_relevance) is a meaningful cliff above the next result, or
whether the two are essentially tied — the latter being a signal the
caller's whole set is built on a weak shared base rather than on a real
best guess.

This is the "relative cliff" hypothesis the Phase 1 measurement opened but
could not validate on two corpora (no clean rank-1-vs-rank-2 separation on
django, gte's compressed range lifted the whole distribution). Phase 3
is the instrument, not a number: this module ships the MECHANISM behind
a config switch default OFF, so the reviewer's distribution study can
measure whether the cliff separates the known-good from the known-absent
class on the corpora Phase 1 didn't reach. A change to the default is
explicitly NOT made here (Rail 7; the evidence does not exist yet).

Design contract (the function this file exports):

  relative_cliff_fires(scores, *, min_top, min_drop) -> bool

  - scores: a finite, non-decreasing-IN-RANK list of ce_relevance
    values, in display order. May be any length (including 0 or 1);
    no daemon, no corpus, no chunk metadata needed.
  - min_top:  the absolute floor below which a "high" rank-1 score
    cannot credibly lead a set. A rank-1 already well below this bar
    is not a confident result regardless of how big the drop to rank-2
    looks; the cliff only matters on a top result that has any business
    being a top result.
  - min_drop: the relative gap required between rank-1 and rank-2 to
    count as a true cliff — expressed as a SHARE OF REMAINING HEADROOM
    above rank-1, i.e.
        (score[0] - score[1]) >= min_drop * (1 - score[0])
    rather than a fixed score difference, because the score scale is
    bounded in [0, 1] and a constant margin is unsatisfiable on a
    confident query (UPG-GATE-V4-MINORS, the score_order_explain
    rationale, restated for this signal: once score[0] reaches 2/3, no
    gap however large can clear a fixed margin).
  - Returns True iff:
        len(scores) >= 2
        AND score[0] is not None
        AND score[1] is not None
        AND score[0] >= min_top
        AND (score[0] - score[1]) >= min_drop * (1 - score[0])
    Otherwise False. min_top >= 1.0 is the documented "off" value: no
    rank-1 can clear it, so the signal is always False (an exact
    no-op restoring pre-this-feature behaviour). enabled=false in
    config short-circuits before this function is even called, so
    callers that want the OFF semantic do NOT have to set min_top
    to 1.0 by hand.

Score interpretation is the same as searcher._apply_quality_and_dedup's
displayed-score contract: ce_relevance is the calibrated cross-encoder
sigmoid, bounded in [0, 1]. A None at rank 0 or rank 1 means reranking
didn't run on that candidate (rerank=False, or the reranker model
failed to load, or the candidate fell outside the rerank pool and
wasn't backfilled); in any of those cases the cliff is a no-op — the
displayed score would be a dense-cosine fallback, and the UPG-
NOTFOUND-FLOOR-2 evidence already showed a bi-encoder cosine cannot
separate absent-topic from on-topic, so reusing it here would
re-introduce the same defect. Gating on ce_relevance-only is
deliberate, matching min_top_relevance's own gate above.

No query-side heuristics: this function is a deterministic comparison
of two SCORES, both already-computed, both structural. It does not
look at the query text, the query tokens, the corpus, or any chunk
metadata. The relative-drop test on SCORES is the allowed category
under the query-side-heuristics rail; a test on query TEXT would not
be, and is not implemented here.
"""
from __future__ import annotations

from typing import Iterable, Optional


def relative_cliff_fires(
    scores: Iterable[Optional[float]],
    *,
    min_top: float,
    min_drop: float,
) -> bool:
    """Return True iff the rank-1-to-rank-2 cliff is decisive.

    See the module docstring for the full contract. The function is
    deliberately defined on an Iterable so a test can pass
    `scores=[0.9, 0.3, 0.2]` or any other sequence-shaped input; the
    first two elements are the only ones read. None on either is a
    "no judgment exists" — same semantics as a reranker that didn't
    run. The result is not affected by trailing entries past rank 1
    (e.g. a noisy rank-2 cannot rescue a weak cliff: a real cliff is
    between rank-1 and the next candidate, full stop).
    """
    seq = list(scores)
    if len(seq) < 2:
        return False
    s0 = seq[0]
    s1 = seq[1]
    # None on either side means no cross-encoder judgment exists for that
    # rank — the dense-cosine fallback is the only score available, and
    # UPG-NOTFOUND-FLOOR-2's evidence rules it out as a separation signal.
    if s0 is None or s1 is None:
        return False
    if s0 < min_top:
        return False
    # Headroom-relative gap: a cliff measured as a share of the score
    # range still available above rank-1. A 0.10 fixed-difference rule
    # would be unsatisfiable on any confident query where score[0] is
    # already close to 1.0; this formulation degrades gracefully with
    # rank-1 confidence (see module docstring).
    headroom = 1.0 - s0
    gap = s0 - s1
    return gap >= min_drop * headroom
