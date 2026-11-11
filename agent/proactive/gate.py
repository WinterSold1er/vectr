"""Gating policy for proactive context (UPG-PRO-5).

Turns a set of scored `Candidate`s into 0..K emitted items packed into one
deterministic block. Every rule is numeric and deterministic — a similarity
floor (unbypassable, structural exact matches included), a per-event budget,
a per-session dedup/cooldown ledger, an optional ADDITIVE cross-channel
per-turn dedup (UPG-PROXY-CROSS-CHANNEL-DEDUP), and a stable sort. There is
no branch that reads conversation content to decide *what kind* of help to
give; the gate only orders, thresholds, and concatenates.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from typing import Callable, Protocol, runtime_checkable

from agent.config import (
    PROACTIVE_ENVELOPE_CLOSE,
    PROACTIVE_ENVELOPE_OPEN_AGENT,
    PROACTIVE_ENVELOPE_OPEN_AUTO,
    PROACTIVE_ENVELOPE_OPEN_HUMAN,
)
from agent.proactive.types import (
    NOTE_PROVENANCE_TRUST_RANK,
    STRUCTURAL_TIER_DECLARED_ANCHOR,
    STRUCTURAL_TIER_MENTION,
    Candidate,
    InjectionResult,
)

# Envelope wrapped around every packed injection block (UPG-PROXY-INJECT-
# ROLE-PROVENANCE). The proxy channel appends this block onto the newest
# USER-authored message on the wire (agent/proactive/request_window.py's
# append_context_block), so without an explicit, unambiguous boundary a
# receiving model can read the injected notes as carrying user authority —
# a note with kind="directive" in particular is WRITTEN as an imperative and
# was observed being followed as an in-turn instruction. The open/close pair
# marks the whole block, unambiguously, as machine-retrieved reference
# material: not a request, not an instruction, no authority.
#
# Three OPEN variants, wired from agent/config.yaml's `proactive.envelope`
# block, chosen at pack time by `select()` from the WEAKEST
# `Candidate.note_provenance` among the items actually selected (see
# `_envelope_open_for` below) — a note PROPERTY, never query/window content.
# `_ENVELOPE_OPEN`/`_ENVELOPE_CLOSE` keep these exact names, assigned as
# plain module-level globals read live (not captured into a frozen dict at
# import time) rather than inlined at each call site, because
# benchmarks/injection_utility/analysis/envpatch/sitecustomize.py
# monkeypatches them directly for an envelope-wording A/B override; adding
# the two new tiers as ADDITIONAL separate globals preserves that mechanism
# unchanged for the "auto" tier.
#
# Every marker is a compile-time-shaped constant — fixed overhead per tier,
# NOT counted against `self._max_chars` below (which governs only the packed
# item lines joined between them). Total envelope overhead is therefore the
# exact same fixed number of characters (for a given tier) on every injected
# request regardless of how many items are packed inside; only the item-line
# portion between the markers varies, and that portion stays bounded by
# `max_chars_per_event` as before. `InjectionResult.context` is
# `len(open) + 1 + len(body) + 1 + len(close)` — a deterministic function of
# the selected tier's envelope plus the selected lines, never unbounded.
_ENVELOPE_OPEN = PROACTIVE_ENVELOPE_OPEN_AUTO
_ENVELOPE_OPEN_AGENT = PROACTIVE_ENVELOPE_OPEN_AGENT
_ENVELOPE_OPEN_HUMAN = PROACTIVE_ENVELOPE_OPEN_HUMAN
_ENVELOPE_CLOSE = PROACTIVE_ENVELOPE_CLOSE

# Sort-key stand-in for a candidate whose first path-mention offset was never
# computed (or has no mention at all): it must order AFTER every candidate
# that does carry a mention, never before offset 0 (UPG-PROXY-WEAK-TIER-
# TIEBREAK). Only ever compared against ints, never returned.
_UNMENTIONED_SORT_KEY = float("inf")


def _envelope_open_for(selected: list[Candidate]) -> str:
    """Pick the envelope-open variant for one packed block (UPG-PROXY-INJECT-
    ROLE-PROVENANCE): the WEAKEST `note_provenance` among every candidate
    actually selected. A candidate with no recorded provenance — unset, or a
    non-note candidate such as code_semantic — looks up as rank 0 ("auto"),
    the same "default to weakest" rule `matcher.py`'s `_provenance_label()`
    already applies per-note; this applies it again, block-wide. One weak
    item in an otherwise strong block is enough to fall back to the
    maximum-skepticism wording, since the envelope wraps the WHOLE block —
    each line already carries its own per-item provenance marker regardless
    of which envelope wraps it."""
    weakest = min(
        (NOTE_PROVENANCE_TRUST_RANK.get(c.note_provenance, 0) for c in selected),
        default=0,
    )
    if weakest >= NOTE_PROVENANCE_TRUST_RANK["human"]:
        return _ENVELOPE_OPEN_HUMAN
    if weakest >= NOTE_PROVENANCE_TRUST_RANK["agent"]:
        return _ENVELOPE_OPEN_AGENT
    return _ENVELOPE_OPEN


class SessionLedger:
    """Bounded ring of recently-emitted anchor ids for one session.

    Suppresses re-injecting the same item within a cooldown window (last N
    emitted anchor ids). Insertion-ordered; the oldest id is evicted once the
    capacity is exceeded.

    UPG-PROXY-COOLDOWN-NO-TIME-DECAY: an entry may also carry a time-to-
    live. `ttl_seconds=None` (the historical behaviour, and what every
    pre-existing direct construction gets) is a pure count ring — an id
    stays suppressed until `capacity` OTHER distinct ids cycle through,
    however long that takes. With a TTL, an entry stops suppressing once it
    is older than the TTL regardless of ring position: under the proxy
    channel's process-scoped session key (UPG-PROXY-COOLDOWN-SESSION-
    IDENTITY) "last 30" spans the whole proxy process lifetime, so without
    a time bound a note that becomes genuinely relevant again hours later
    stays suppressed until 30 other anchors have cycled. The capacity bound
    is unchanged and still applies on top of the TTL — the TTL only ever
    EXPIRES entries early, never extends retention past the ring.

    The clock is injected (`now_fn`) so a test can advance time arbitrarily
    without sleeping; production uses `time.monotonic()`, which cannot go
    backwards or jump with wall-clock adjustments — elapsed-time policy
    must not be sensitive to either.
    """

    def __init__(
        self,
        capacity: int,
        ttl_seconds: float | None = None,
        now_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._capacity = max(1, capacity)
        self._ttl_seconds = ttl_seconds
        self._now_fn = now_fn
        self._ring: deque[str] = deque()
        # UPG-PROXY-COOLDOWN-NO-TIME-DECAY: set -> dict (anchor_id -> last-
        # delivery monotonic ts). Membership semantics are unchanged; the
        # timestamp exists only when a TTL is configured.
        self._seen: dict[str, float] = {}

    def _expire(self) -> None:
        """Drop entries older than the TTL. No-op when no TTL is configured.
        A linear sweep over at most `capacity` entries; the ring is rebuilt
        only when something actually expired."""
        if self._ttl_seconds is None or not self._seen:
            return
        now = self._now_fn()
        expired = [aid for aid, ts in self._seen.items() if now - ts >= self._ttl_seconds]
        if not expired:
            return
        for aid in expired:
            del self._seen[aid]
        self._ring = deque(aid for aid in self._ring if aid in self._seen)

    def seen(self, anchor_id: str) -> bool:
        self._expire()
        return anchor_id in self._seen

    def add(self, anchor_id: str) -> None:
        self._expire()
        if anchor_id in self._seen:
            # Refresh: suppression runs from the MOST RECENT delivery of
            # this anchor, not its first. Also moves the id to the ring's
            # tail so capacity eviction (oldest-inserted first) evicts the
            # least-recently-delivered id, matching the refresh semantics.
            self._seen[anchor_id] = self._now_fn()
            self._ring.remove(anchor_id)
            self._ring.append(anchor_id)
            return
        self._ring.append(anchor_id)
        self._seen[anchor_id] = self._now_fn()
        while len(self._ring) > self._capacity:
            old = self._ring.popleft()
            self._seen.pop(old, None)


class LedgerStore:
    """Thread-safe, LRU-bounded map of session_id -> SessionLedger.

    Proxied requests can arrive concurrently, so ledger access is locked. The
    number of tracked sessions is bounded so a long-lived proxy cannot grow
    unbounded state; the least-recently-used session's ledger is dropped first.
    """

    def __init__(
        self,
        cooldown_items: int,
        max_sessions: int = 512,
        ttl_seconds: float | None = None,
        now_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cooldown_items = cooldown_items
        self._max_sessions = max(1, max_sessions)
        self._ttl_seconds = ttl_seconds
        self._now_fn = now_fn
        self._ledgers: "OrderedDict[str, SessionLedger]" = OrderedDict()
        self._lock = threading.Lock()

    def _ledger_for(self, session_id: str) -> SessionLedger:
        led = self._ledgers.get(session_id)
        if led is None:
            led = SessionLedger(self._cooldown_items, self._ttl_seconds, self._now_fn)
            self._ledgers[session_id] = led
            while len(self._ledgers) > self._max_sessions:
                self._ledgers.popitem(last=False)
        else:
            self._ledgers.move_to_end(session_id)
        return led

    def seen(self, session_id: str, anchor_id: str) -> bool:
        with self._lock:
            return self._ledger_for(session_id).seen(anchor_id)

    def record(self, session_id: str, anchor_ids) -> None:
        with self._lock:
            led = self._ledger_for(session_id)
            for aid in anchor_ids:
                led.add(aid)


@runtime_checkable
class TurnLedger(Protocol):
    """Structural type for `agent.trigger_engine.TurnInjectionLedger`
    (UPG-PROXY-CROSS-CHANNEL-DEDUP). Declared here by shape rather than
    imported, so this low-level gating module stays decoupled from the
    trigger-engine package; the service layer passes its real
    `TurnInjectionLedger` instance straight through `select()`'s
    `turn_ledger` parameter."""

    def eligible(self, note_id: int) -> bool: ...

    def claim(self, note_id: int) -> None: ...


def _note_id_from_anchor(anchor_id: str) -> int | None:
    """Parse the note id out of a `note:<id>` anchor_id, or None for a
    non-note anchor (e.g. `chunk:<...>` from the code-search matcher) —
    the cross-channel turn ledger is note-granular only, matching the
    hook/trigger-engine side's own `TurnInjectionLedger` (keyed on
    `WorkingNote.note_id`, which has no equivalent for a code chunk)."""
    if not anchor_id.startswith("note:"):
        return None
    try:
        return int(anchor_id[len("note:"):])
    except ValueError:
        return None


class ProactiveGate:
    """Applies floor -> dedup/cooldown -> budget -> deterministic pack."""

    def __init__(
        self,
        *,
        min_similarity: float,
        max_items_per_event: int,
        max_chars_per_event: int,
        cooldown_items: int,
        max_weak_structural_items: int,
        cooldown_ttl_seconds: float | None = None,
        ledger_store: LedgerStore | None = None,
    ) -> None:
        self._min_similarity = min_similarity
        self._max_items = max(0, max_items_per_event)
        self._max_chars = max(0, max_chars_per_event)
        self._max_weak_structural_items = max(0, max_weak_structural_items)
        # UPG-PROXY-COOLDOWN-NO-TIME-DECAY: threaded into the default ledger
        # store; an explicitly passed `ledger_store` configures its own TTL.
        self._ledger = ledger_store or LedgerStore(
            cooldown_items, ttl_seconds=cooldown_ttl_seconds
        )

    def select(
        self,
        candidates: list[Candidate],
        *,
        session_id: str = "",
        structural_only: bool = False,
        turn_ledger: TurnLedger | None = None,
        record: bool = True,
        edited_file_paths: frozenset[str] = frozenset(),
    ) -> InjectionResult:
        """Deterministically pick and pack the items to inject.

        `structural_only` (a static per-channel policy, never a content read)
        drops every semantic candidate — used by the high-frequency channels
        where only exact matches are cheap enough to be worth the budget.

        `record` (UPG-PROXY-APPEND-BURNS-COOLDOWN) controls whether SELECTION
        also CHARGES — i.e. writes the selected anchors into the cooldown
        `_ledger` and claims their note ids in `turn_ledger`. It defaults to
        True, which is retrieval-time charging: correct, and byte-identical to
        this method's historical behaviour, for any channel where retrieval IS
        delivery (the hook/trigger-engine surfaces, which see the packed result
        directly). The PROXY channel is the one place where it is not: the
        proxy retrieves over HTTP and only afterwards discovers whether the
        request body can actually carry the block. Charging there at selection
        spends a note's one cooldown slot on a request that may never receive
        it. Those callers pass `record=False` and charge later, on confirmed
        delivery (see `VectrService.proactive_context`).

        `record=False` changes ONLY the two write-backs. Every READ stays
        exactly as it is — the cooldown `seen()` filter in step 4 and the
        `turn_ledger.eligible()` filter in step 3 both still apply, so an
        uncharged selection is still suppressed by whatever WAS charged before
        it.

        `turn_ledger` (UPG-PROXY-CROSS-CHANNEL-DEDUP), if given, is the SAME
        per-session `TurnInjectionLedger` the hook/trigger-engine surfaces
        share (via the service's `_turn_ledger_for`) — ADDITIVE to, never a
        replacement for, this gate's own anchor-id cooldown `_ledger` below
        (which covers code-chunk candidates too, and provides cross-TURN
        cooldown this ledger does not). A note_id another surface already
        claimed this turn is dropped before packing; every note_id that
        SURVIVES the budget below is claimed here so a later surface this
        same turn will not re-inject it either — a note matched here but
        evicted for budget is deliberately left unclaimed, mirroring
        `WorkingContextStore.fire_and_format()`'s own turn-ledger contract.

        `edited_file_paths` (UPG-PROXY-INJECT-SINGLE-TURN) is this request's
        `ProactiveWindow.edited_file_paths` — the subset of the window's file
        paths a genuine edit-type tool call touched, as opposed to merely
        being read or mentioned. It governs EVENT-ANCHORED RETIREMENT, an
        exception to `record`'s plain "charge every selected item" rule that
        applies to exactly one case: a selected candidate whose
        `structural_tier` is `STRUCTURAL_TIER_DECLARED_ANCHOR` (a note's own
        declared anchor, the strongest structural evidence — the same
        evidence the hook channel's default pre-edit trigger bundle is built
        from, see `agent.trigger_engine.default_bundle_for_kind`) and whose
        `anchor_path` is NOT among `edited_file_paths` is delivered this
        turn same as any other selected item, but is EXCLUDED from the
        cooldown/turn-ledger write-back below — it stays eligible and may be
        selected again on a later request that still matches it, instead of
        retiring after one delivery (UPG-PROXY-INJECT-SINGLE-TURN's "too
        early, then evicted" failure: a note anchored to a file the agent has
        only read so far is not yet done being useful). The moment a request
        DOES show that file being edited, the same candidate is charged
        (retired) exactly like any other selected item. Every other
        candidate — every non-anchor structural tier, every semantic hit,
        every candidate whose `anchor_path` is unset — is charged on
        selection precisely as before this parameter existed; this is a pure
        narrowing of `record`'s scope, never a new class of unbounded
        delivery. `InjectionResult.unretired_anchor_ids` names exactly the
        anchor ids this call exempted, so a caller charging on a LATER
        confirmed-delivery (the proxy's deferred-charge path) can honor the
        same exemption instead of re-charging in full at confirm time.

        The envelope wrapping the packed block (UPG-PROXY-INJECT-ROLE-
        PROVENANCE) is chosen by `_envelope_open_for`, from the WEAKEST
        `Candidate.note_provenance` among `selected` — a mixed-provenance
        block always gets the most skeptical wording that applies to any
        item inside it, never the strongest.
        """
        if self._max_items == 0 or self._max_chars == 0:
            return InjectionResult.empty()

        # 1. Dedup across matchers by anchor_id: keep the strongest occurrence
        #    (highest score, then best provenance rank). Deterministic.
        best: dict[str, Candidate] = {}
        for c in candidates:
            cur = best.get(c.anchor_id)
            if cur is None or (c.score, -c.provenance_rank) > (cur.score, -cur.provenance_rank):
                best[c.anchor_id] = c

        # 2. Floor + per-channel policy: `structural_only` drops every
        #    non-structural candidate (a static per-channel policy); every
        #    remaining candidate — structural or semantic — must then clear
        #    the similarity floor. No exemption: a floor configured above
        #    1.0 admits nothing, including exact structural matches (which
        #    score exactly 1.0).
        eligible: list[Candidate] = []
        for c in best.values():
            if structural_only and not c.is_structural:
                continue
            if c.score >= self._min_similarity:
                eligible.append(c)

        # 3. Cross-channel per-turn dedup (additive — see docstring above).
        if turn_ledger is not None:
            eligible = [
                c for c in eligible
                if _note_id_from_anchor(c.anchor_id) is None
                or turn_ledger.eligible(_note_id_from_anchor(c.anchor_id))
            ]

        # 4. Cooldown: drop anything already emitted for this session recently.
        if session_id:
            eligible = [c for c in eligible if not self._ledger.seen(session_id, c.anchor_id)]

        if not eligible:
            return InjectionResult.empty()

        # 5. Deterministic order: score desc, then — UPG-PROXY-WEAK-TIER-
        #    TIEBREAK — relevance among EQUAL-score candidates, desc/asc
        #    respectively, then provenance rank asc, anchor_id asc.
        #
        #    Every Tier-D ("weak mention") candidate scores exactly
        #    `structural_scores.mention`, so before this tie-break the single
        #    weak item the per-event cap admits was chosen by provenance rank
        #    + anchor_id — insertion-order luck carrying no relevance signal
        #    at all (measured: two runs of identical product code over two
        #    orderings of the same store landed 71.9% vs 100% strict
        #    precision, with 16 of 20 weak slots going to notes whose subject
        #    was a DIFFERENT file). The two added terms are structural
        #    properties of the note-vs-path match computed by the matcher:
        #    how many times the matched file's basename occurs in the note
        #    body at a path boundary (more occurrences => more likely the
        #    note's SUBJECT is this file, not a drive-by name-drop), and how
        #    early the first occurrence lands (a subject names its file in
        #    the title/opening; a passing mention sits mid-list). Both
        #    default inert (0 / unmentioned) for hand-built and non-note
        #    candidates, so every pre-existing equal-score ordering —
        #    including the structural-vs-semantic provenance tie-break
        #    pinned by test_structural_outranks_semantic_on_tie — is
        #    unchanged. Tuning `structural_scores.mention` itself is NOT
        #    what happens here: that would change which TIER wins against
        #    the semantic band, not which Tier-D item wins against another.
        eligible.sort(
            key=lambda c: (
                -c.score,
                -c.path_mention_count,
                c.path_mention_first_offset if c.path_mention_first_offset >= 0 else _UNMENTIONED_SORT_KEY,
                c.provenance_rank,
                c.anchor_id,
            )
        )

        # 6. Budget: at most K items and T chars. Each candidate `line` is capped
        #    to T by the matcher, so a single item always fits; stop at the first
        #    item that would overflow the running character total.
        #
        #    UPG-PROXY-INJECT-PRECISION lever 3: at most
        #    `max_weak_structural_items` Tier-D ("weak mention") structural
        #    candidates are selected per delivery moment — a check against
        #    `Candidate.structural_tier`, a STRUCTURAL PROPERTY of the match
        #    computed by the matcher, never a read of window/query content. A
        #    capped item is skipped (never appended to `selected`, never
        #    counted against the char budget), leaving its slot for a
        #    stronger candidate later in `eligible`'s score-descending order.
        selected: list[Candidate] = []
        used_chars = 0
        weak_structural_selected = 0
        for c in eligible:
            if len(selected) >= self._max_items:
                break
            if (
                c.structural_tier == STRUCTURAL_TIER_MENTION
                and weak_structural_selected >= self._max_weak_structural_items
            ):
                continue
            line = c.line
            add_chars = len(line) + (1 if selected else 0)  # newline separator
            if used_chars + add_chars > self._max_chars:
                break
            selected.append(c)
            used_chars += add_chars
            if c.structural_tier == STRUCTURAL_TIER_MENTION:
                weak_structural_selected += 1

        if not selected:
            return InjectionResult.empty()

        # UPG-PROXY-INJECT-SINGLE-TURN: event-anchored retirement (see
        # `edited_file_paths` in the docstring) — a declared-anchor
        # structural candidate whose own anchor path was not just edited
        # this request is exempted from the write-backs below, however
        # `record` reads, so it stays eligible for a later request instead
        # of retiring after this one delivery.
        unretired = [
            c for c in selected
            if c.structural_tier == STRUCTURAL_TIER_DECLARED_ANCHOR
            and c.anchor_path is not None
            and c.anchor_path not in edited_file_paths
        ]
        unretired_ids = {c.anchor_id for c in unretired}
        chargeable = [c for c in selected if c.anchor_id not in unretired_ids]

        # Charge (see `record` in the docstring): skipped wholesale when the
        # caller will charge later on confirmed delivery. Both write-backs move
        # together — charging one ledger but not the other would leave the two
        # dedup surfaces disagreeing about what has been emitted. The
        # event-anchored exemption above applies ONLY to this cross-TURN
        # cooldown `_ledger` — `turn_ledger` is a same-turn, cross-SURFACE
        # dedup that resets every turn regardless (see its own docstring),
        # so an unretired item is still claimed there: it WAS delivered this
        # turn, and a hook surface seeing the same note in the same turn
        # must still be suppressed, exactly as for any other delivered item.
        if record:
            if session_id and chargeable:
                self._ledger.record(session_id, [c.anchor_id for c in chargeable])
            if turn_ledger is not None:
                for c in selected:
                    note_id = _note_id_from_anchor(c.anchor_id)
                    if note_id is not None:
                        turn_ledger.claim(note_id)

        body = "\n".join(c.line for c in selected)
        envelope_open = _envelope_open_for(selected)
        context = f"{envelope_open}\n{body}\n{_ENVELOPE_CLOSE}"
        return InjectionResult(
            context=context,
            item_count=len(selected),
            anchor_ids=tuple(c.anchor_id for c in selected),
            scores=tuple(round(c.score, 4) for c in selected),
            states=tuple(c.state for c in selected),
            unretired_anchor_ids=tuple(c.anchor_id for c in unretired),
        )
