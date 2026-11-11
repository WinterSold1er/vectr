"""Core value types for the proactive-context engine (UPG-PRO).

All types are plain data. The matcher produces `Candidate`s; the gate turns a
set of candidates into one `InjectionResult`. `ProactiveWindow` is the
normalised, source-agnostic input both delivery seams (hooks, proxy) build.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# Provenance rank breaks score ties deterministically: structural exact matches
# outrank semantic notes, which outrank code-search hits. Lower is stronger.
PROVENANCE_RANK: dict[str, int] = {
    "note_structural": 0,
    "symbol_def": 1,
    "note_semantic": 1,
    "code_semantic": 2,
}

# Trust rank among `WorkingNote.provenance` values (UPG-PROXY-INJECT-ROLE-
# PROVENANCE) — NOT the candidate-KIND tie-break above; this ranks the note
# PROPERTY carried in `Candidate.note_provenance`. Lower is weaker. Matches
# working_context_store's own promotion-direction ordering (`_store.py`'s
# `promote()`: "auto" -> "agent" -> "human"). The gate uses this to pick the
# weakest tier among every candidate actually selected for one delivery, so
# a mixed-provenance block never overstates trust; a candidate with no
# recorded provenance (unset, or a non-note candidate such as code_semantic)
# looks itself up here via `.get(..., 0)` and lands at the weakest rank.
#
# "user-stated" (UPG-MEM-PROVENANCE-USER-STATED) ranks WITH "agent", not above
# it, matching `_types.PROMOTION_RANK`: the envelope describes who authored
# the text arriving on the wire, and that is still an AI session transcribing
# the user, so the agent-tier envelope is the honest one. Ranking it at
# "human" would let a bound excerpt select the strongest envelope for the
# whole block; leaving it unlisted would drag a mixed block down to the
# weakest one. It is listed explicitly rather than left to `.get(..., 0)` for
# exactly that reason.
NOTE_PROVENANCE_TRUST_RANK: dict[str, int] = {
    "auto": 0, "agent": 1, "user-stated": 1, "human": 2,
}

# The folded note-lifecycle state (UPG-MEMORY-STATE-MACHINE) a non-note
# candidate carries — a code-search hit has no note to fold state for, and
# rendering it as "active" would honestly mislabel it as an assertion under
# lifecycle tracking. Kept as an explicit, named value (not blank/None) so an
# audit reader always sees an honest label positionally aligned with every
# other candidate's state, never a silent gap.
CANDIDATE_STATE_NOT_APPLICABLE = "n/a"


# Structural-match evidence tiers (UPG-PROXY-INJECT-PRECISION). Populated only
# on `kind="note_structural"` candidates (see matcher.py's
# `_structural_note_candidate`); every other candidate leaves `Candidate.
# structural_tier` at its default `None`. Each label names HOW a structural
# match was found — a declared anchor, a declared trigger path glob, a
# content mention in a kind=gotcha note, or a content mention in any other
# allowed kind — a property of the match itself, never a read of query/
# window content. `STRUCTURAL_TIER_MENTION` (Tier D, the weakest) is what
# the gate's per-event weak-item cap (`max_weak_structural_items`) counts
# against.
#
# UPG-TRIGGERS-INERT-ON-PROXY-STRUCTURAL: `STRUCTURAL_TIER_DECLARED_TRIGGER`
# ranks directly below `DECLARED_ANCHOR` and above both mention tiers — a
# `triggers[].path` glob is, like `anchors`, an explicit author declaration
# ("this note concerns this file"), not an incidental content mention, so it
# must not collapse into the weak `mention` tier (UPG-PROXY-WEAK-TIER-
# TIEBREAK found 16/20 admitted weak-tier items were off-topic; reusing that
# tier for a deliberate declaration would launder a strong signal through a
# score band designed for a weak one). It ranks below a plain anchor because
# a glob is inherently less precise per-file than one exact path — many
# files can satisfy "agent/*.py", only one can satisfy an exact anchor.
STRUCTURAL_TIER_DECLARED_ANCHOR = "declared_anchor"
STRUCTURAL_TIER_DECLARED_TRIGGER = "declared_trigger"
STRUCTURAL_TIER_GOTCHA_MENTION = "gotcha_mention"
STRUCTURAL_TIER_MENTION = "mention"


@dataclass(frozen=True)
class Candidate:
    """One concrete thing that could be injected, already rendered.

    `line` is the self-describing, provenance-tagged text the model sees (e.g.
    "note #12 (gotcha, anchored to resolver.py): drops on scope exit"). The gate
    only orders/dedups/budgets/joins these; it never rewrites or reasons about
    `line`, so packing stays additive and deterministic.
    """

    kind: str            # note_structural | note_semantic | symbol_def | code_semantic
    line: str            # rendered, self-describing injected text
    score: float         # tiered for structural (see structural_tier); cosine/hybrid otherwise
    anchor_id: str       # stable dedup id, e.g. "note:12" / "chunk:foo.py:1-9"
    is_structural: bool  # True => a structural (path-anchored) match; still
                          # subject to the gate's similarity floor like any
                          # other candidate — see agent/proactive/gate.py
    state: str = "active"  # folded note-lifecycle state ("active" |
                          # "superseded" | "revoked") for a note candidate, or
                          # CANDIDATE_STATE_NOT_APPLICABLE for a non-note
                          # candidate (UPG-PROXY-AUDIT-DURABLE). Metadata
                          # carried through to the audit line only — never
                          # used to reorder or reweight the candidate itself.
    structural_tier: str | None = None  # one of the STRUCTURAL_TIER_* labels
                          # above for a structural candidate, None otherwise —
                          # see agent/proactive/gate.py's weak-item cap.
                          # Independent of `state`: the tier describes HOW the
                          # match was found, the state describes what the note
                          # currently IS — a revoked note keeps its tier and so
                          # still competes for a slot as a deterrent.
    anchor_path: str | None = None  # the exact `ProactiveWindow.file_paths`
                          # entry a STRUCTURAL_TIER_DECLARED_ANCHOR candidate
                          # matched (UPG-PROXY-INJECT-SINGLE-TURN); None for
                          # every other candidate, including weaker structural
                          # tiers. Used ONLY by the gate's event-anchored
                          # retirement (agent/proactive/gate.py's `select()`)
                          # to test membership against a request's
                          # `edited_file_paths` — never for display; the
                          # rendered `line`'s anchor label comes from
                          # matcher.py independently of this field (since
                          # UPG-MATCHER-RELATIVE-ANCHOR-LABEL it names the
                          # form that actually matched — declared anchor
                          # string or glob — not necessarily a basename).
    note_provenance: str | None = None  # `WorkingNote.provenance` (UPG-PROXY-
                          # INJECT-ROLE-PROVENANCE), one of PROVENANCE_VALUES
                          # ("human" | "agent" | "auto") for a note-backed
                          # candidate; None for a non-note candidate
                          # (code_semantic). Populated via matcher.py's
                          # `_provenance_label()`, which already normalises a
                          # missing/invalid value on the note itself down to
                          # "auto" — so this field is never an unrecognised
                          # string, only ever one of the three tiers or None.
                          # Used ONLY by the gate to pick the envelope
                          # wording for the whole packed block (the weakest
                          # tier among every SELECTED candidate); never for
                          # display — the per-line provenance marker in
                          # `line` is independent of this field and always
                          # present regardless of whether the gate reads it.
    path_mention_count: int = 0  # UPG-PROXY-WEAK-TIER-TIEBREAK: how many times
                          # the matched file's basename occurs in the note's
                          # content at a genuine path boundary (the SAME
                          # predicate `_path_boundary_match` applies). Populated
                          # only by matcher.py for `note_structural` candidates;
                          # every other candidate (and any hand-built one) keeps
                          # the inert default. Read ONLY by the gate's step-5
                          # sort as a tie-break among EQUAL-score candidates —
                          # never compared against the similarity floor and
                          # never added into `score` (moving a value there would
                          # change which TIER wins against the semantic band,
                          # which is explicitly out of scope).
    path_mention_first_offset: int = -1  # UPG-PROXY-WEAK-TIER-TIEBREAK:
                          # character offset of the FIRST such boundary
                          # occurrence in the note's content, -1 when never
                          # computed / no mention (the default every non-
                          # structural candidate keeps; the gate sorts -1 after
                          # every real offset). Same population scope and same
                          # tie-break-only reader as `path_mention_count`
                          # above: among notes mentioning the file equally
                          # often, one whose SUBJECT is the file tends to name
                          # it early rather than mid-list.

    @property
    def provenance_rank(self) -> int:
        return PROVENANCE_RANK.get(self.kind, 9)


@dataclass
class ProactiveWindow:
    """Normalised, in-memory view of the recent conversation.

    Built from a proxied request body (proxy seam) or a transcript tail (hook
    seam, future). Never persisted. `text` is the assembled query; `file_paths`
    and `symbols` are deterministic structural anchors extracted from tool
    traffic — no free-text path/identifier guessing.
    """

    text: str = ""
    file_paths: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    # UPG-PROXY-INJECT-SINGLE-TURN: the subset of `file_paths` this request's
    # tool traffic actually EDITED (Edit/Write/MultiEdit/apply_patch — the
    # same tool-name set `agent/hook_cli.py`'s `_build_episode_payload` and
    # `main.py`'s PreToolUse hook group already use to distinguish an edit
    # call from a read/search one), never merely mentioned or read. A subset
    # of `file_paths` by construction — every edit-tool call also carries a
    # `file_path` extracted into `file_paths` by the same pass. Drives the
    # gate's event-anchored retirement for declared-anchor structural
    # candidates (agent/proactive/gate.py's `select()`); it plays no role in
    # matching itself, only in whether an already-matched anchored note
    # retires afterward.
    edited_file_paths: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.text.strip() or self.file_paths or self.symbols)


@dataclass(frozen=True)
class InjectionResult:
    """The gate's decision for one delivery moment.

    `context` is the packed block to inject ("" means inject nothing). The
    metadata fields are audit-safe: ids, scores, and lifecycle states — never
    conversation text or note bodies.
    """

    context: str
    item_count: int
    anchor_ids: tuple[str, ...]
    scores: tuple[float, ...]
    # Positionally aligned with anchor_ids/scores WHEN POPULATED (the gate
    # builds all three from the same `selected` list, so they cannot drift).
    # Empty means "lifecycle state not transported for this result" — never
    # "no revoked items"; see agent/proactive/provider.py, which reconstructs
    # a result from the daemon's `/v1/proactive` response and cannot know it.
    states: tuple[str, ...] = ()
    # Opaque confirm handle for a DEFERRED-CHARGE retrieval (UPG-PROXY-APPEND-
    # BURNS-COOLDOWN). Non-empty only when the daemon selected these items
    # WITHOUT charging their cooldown slots, and is waiting to be told whether
    # they were actually delivered. The proxy hands it back verbatim once the
    # block is confirmed appended; it is never parsed, and it carries no
    # content. Empty means "nothing to confirm" — either the gate charged at
    # selection (the default, and what every non-proxy channel does) or the
    # daemon predates deferred charging.
    delivery_token: str = ""
    # UPG-PROXY-INJECT-SINGLE-TURN: the subset of `anchor_ids` selected THIS
    # delivery that must NOT be written into the cross-turn cooldown ledger
    # even once delivery is confirmed — a declared-anchor structural
    # candidate whose anchored file was matched but not yet EDITED this
    # request (`Candidate.anchor_path` absent from the window's
    # `edited_file_paths`). Such a note stays eligible for the very same
    # anchor_id on a later request instead of retiring after one delivery.
    # Always a subset of `anchor_ids`; empty for every ordinary
    # (non-event-anchored) result, which is charged in full exactly as
    # before this field existed.
    unretired_anchor_ids: tuple[str, ...] = ()

    @staticmethod
    def empty() -> "InjectionResult":
        return InjectionResult(context="", item_count=0, anchor_ids=(), scores=(), states=())

    def is_empty(self) -> bool:
        return self.item_count == 0 or not self.context
