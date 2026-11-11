"""
Dataclasses and constants for the working-context store.
"""
from __future__ import annotations

from dataclasses import dataclass


# Memory kinds (UPG-9.3). Mirrors the disk-memory split that makes CLAUDE.md
# (unconditional directives) distinct from MEMORY.md (relevance-ranked learnings):
#   directive — must-never-miss rules; injected unconditionally at SessionStart.
#   task      — current-work context; injected in the SessionStart boot set.
#   gotcha    — file/path-anchored caveats; injected at PreToolUse + semantic recall.
#   finding   — relevance-ranked learnings; injected per-prompt at UserPromptSubmit.
#   reference — pointers (URLs/tickets); surfaced on demand only.
#   decision  — an architectural/design decision plus its why (UPG-DECISION-
#               TIMELINE). NOT boot-privileged (no default trigger bundle, see
#               default_bundle_for_kind() in trigger_engine.py — same
#               evaluation-time bucket as finding/reference); accrues over a
#               project's life and is meant to be recalled as a group with
#               `vectr_recall(kind="decision", sort_by="chronological")`,
#               yielding an ADR-like decision timeline for free.
#   operational — an env/process/build fact (build quirks, CI gotchas,
#               feedback-loop knowledge) that is not anchored to a single code
#               file the way `gotcha` is (UPG-MEMORY-STATE-MACHINE §5.1).
#               Primary surface is prompt-time semantic matching
#               (UserPromptSubmit, embeddings only); a secondary PreToolUse
#               command-family surface is opt-in via an explicit `triggers[]`
#               override (§5.2). Always carries a last-confirmed/drift verdict
#               (§4.4) rather than the plain provenance framing other kinds get.
VALID_KINDS: tuple[str, ...] = (
    "directive", "task", "gotcha", "finding", "reference", "decision", "operational",
)
DEFAULT_KIND = "finding"

# Trigger engine wave 1 (TRIGGER-ENGINE, bm2-design-skeleton.md §2/§5) — closed
# protocol vocabularies, not tunable thresholds, so these live beside VALID_KINDS
# as plain constants rather than in config.yaml (same reasoning as VALID_KINDS
# itself: a caller either uses one of these values or the write is rejected;
# there is nothing here for an operator to retune).
#
# EVENT_VALUES: the E trigger primitive's enum. Every value is a real lifecycle
# moment; none has a live hook caller into fire()/evaluate_note() yet this wave
# (see agent/trigger_engine.py's module docstring for the exact wiring status)
# — an event with no live caller is simply never evaluated, never an error
# ("declared but inert" per the design doc, not a defect).
EVENT_VALUES: tuple[str, ...] = (
    "session-start", "prompt-submit", "pre-edit", "pre-run", "pre-commit", "post-compaction",
)

# SCOPE_VALUES: how broadly a memory's declared triggers apply.
# "workspace" (default) is a true no-op — today's unfiltered behaviour.
# "repo" is currently a no-op identical to "workspace": true cross-worktree/
# cross-clone sharing needs the store keyed by git-common-dir instead of by
# workspace path, which is a larger storage change (see
# UPG-TRIGGER-SCOPE-REPO-CROSSSTORE); recorded faithfully at write time so the
# value round-trips once that lands. "session", "branch", and "path-subtree"
# are enforced (TRIGGER-ENGINE wave 2a, `scope_permits()` in
# agent/trigger_engine.py):
#   - "session": enforced in BOTH fire() and recall()/recall_for_path() —
#     an ephemeral note must never surface outside its writing session via
#     any path, not just trigger firing.
#   - "branch": enforced ONLY in fire() — bounds ambient trigger noise (a
#     branch-bound task stops nagging on the wrong branch); a deliberate
#     vectr_recall query is never blocked by branch, and this avoids a git
#     subprocess call on the hot, frequently-called plain recall() path.
#   - "path-subtree": enforced in fire() (against the note's declared
#     anchors[]) and in recall_for_path() (file_path is already a parameter
#     there — free); not enforced in the general query-based recall(), which
#     has no file context to filter against.
# Notes written before this wave carry scope="workspace" (the dataclass
# default) and are therefore unaffected — this is backward compatible.
SCOPE_VALUES: tuple[str, ...] = ("workspace", "repo", "path-subtree", "branch", "session")
DEFAULT_SCOPE = "workspace"

# KIND_DEFAULT_SCOPES (UPG-TRIGGER-SCOPE-KIND-DEFAULTS, bm2-design-skeleton.md
# §1's Default bundles table): the scope a note gets WHEN ITS WRITE OMITS
# `scope` entirely (None) -- an explicitly passed scope (including explicit
# "workspace") always wins verbatim and never consults this table. Resolved
# and BAKED INTO THE ROW at write time by `WorkingContextStore.remember()`
# (unlike `default_bundle_for_kind()` in trigger_engine.py, which is
# deliberately evaluation-time so trigger-bundle changes apply
# retroactively -- scope is different: a note's scope is part of its
# identity/visibility contract, not a re-computable ranking input, so
# retroactivity is deliberately avoided here; a note written before this
# table existed keeps whatever scope it was actually written with). Only the
# two kinds the design skeleton's table assigns a non-default scope appear
# here; every other kind (including any future kind) keeps DEFAULT_SCOPE.
# Same rationale as VALID_KINDS/SCOPE_VALUES just above for living beside
# them as a plain constant rather than in config.yaml: this is a closed
# protocol mapping from the design doc, not an operator-tunable weight --
# there is nothing here for an operator to retune.
KIND_DEFAULT_SCOPES: dict[str, str] = {
    "task": "branch",
    "gotcha": "repo",
    "operational": "repo",
}

# PROVENANCE_VALUES: trust/endorsement class (bm2-design-skeleton.md §5),
# strongest first. "human" = a person recorded or endorsed this; only
# imperative directive framing is ever allowed for it. "user-stated" =
# transcribed by an AI session from a direct user statement, with a verbatim
# excerpt of that statement MECHANICALLY bound to the note's content (see
# _user_quote.py) — never settable by a caller, only derived at write time by
# that deterministic check, so it is evidence rather than self-attestation.
# "agent" (default for vectr_remember) = an AI session recorded this; rendered
# as memory-to-verify. "auto" = captured by a mechanism with no reviewing
# judgment; weakest framing, and never allowed on kind="directive" (rejected
# at write time — an unreviewed standing rule is a contradiction in terms).
USER_STATED_PROVENANCE = "user-stated"
PROVENANCE_VALUES: tuple[str, ...] = ("human", USER_STATED_PROVENANCE, "agent", "auto")
DEFAULT_PROVENANCE = "agent"

# SORT_BY_VALUES (UPG-SORTBY-SHARED-VOCAB): the closed recall() sort-mode
# vocabulary. Lives in agent/working_context_store (the layer that owns
# recall semantics and the SQL ORDER BY map) and is re-imported by every
# other surface that declares a sort_by contract — REST (app/models.py),
# MCP dispatch + tool schema (integrations/mcp_server/_dispatch.py +
# _schemas.py), and the CLI (main.py). The brief explicitly considered the
# alternative of keeping it in app/models.py (the REST layer) and rejected
# that as inverted layering: every other surface would then have to import
# from app/ to declare its own contract, making the REST layer the source of
# truth for vocabulary the store actually implements. Putting the constant
# here flows downhill — REST, MCP, CLI, and the store itself all import
# from this one place, and adding a fifth sort mode becomes one edit, not
# five. Public (no leading underscore) because the whole point is
# cross-module import. The store's internal _SORT_ORDER_SQL map and the
# branch conditions in recall()/_semantic_recall()/_format_index_line are
# intentionally NOT rewritten to reference SORT_BY_VALUES: they are
# IMPLEMENTATION sites (per-mode ORDER BY fragments, per-mode sort keys),
# and the constant is the VOCABULARY site — replacing each `if sort_by ==
# "recency":` branch with an enum lookup would buy nothing and obscure the
# per-mode SQL. Same local-tuple convention as VALID_KINDS/EVENT_VALUES/
# SCOPE_VALUES/PROVENANCE_VALUES above — closed protocol vocabulary, not an
# operator-tunable threshold, so it lives here rather than in config.yaml.
SORT_BY_VALUES: tuple[str, ...] = (
    "relevance", "recency", "priority", "chronological",
)

# The ladder `promote()` walks, weakest first — deliberately NOT
# PROVENANCE_VALUES reversed any more (UPG-MEM-PROVENANCE-USER-STATED):
# "user-stated" is DERIVED from a bound verbatim excerpt at write time, so it
# is not a promotion target. Allowing promote(to="user-stated") would hand
# back by self-attestation exactly the class the binding check exists to make
# machine-checkable. The ladder's steps are unchanged from before that class
# existed.
PROMOTION_LADDER: tuple[str, ...] = ("auto", "agent", "human")
# Rank of each stored provenance ON that ladder. "user-stated" ranks with the
# class it replaces at write time ("agent"), which leaves "human" as its one
# legal single step — a person endorsing a transcribed statement — and keeps
# promote() from treating an off-ladder value as rank 0 (which would let an
# "auto -> agent" promotion silently DEMOTE a user-stated note).
PROMOTION_RANK: dict[str, int] = {
    "auto": 0,
    "agent": 1,
    USER_STATED_PROVENANCE: 1,
    "human": 2,
}


@dataclass
class WorkingNote:
    note_id: int
    workspace: str
    content: str
    tags: list[str]
    priority: str          # "high" | "medium" | "low"
    created_at: float
    last_accessed: float
    session_id: str | None = None
    decay_score: float = 1.0
    kind: str = DEFAULT_KIND  # directive | task | gotcha | finding | reference | decision (UPG-9.3, UPG-DECISION-TIMELINE)
    # team/shared notes tri-key model
    author_id: str = ""              # developer/agent identifier
    author_trust_score: float = 1.0  # Bayesian weight per contributor (0.0–1.0)
    valid_from: float = 0.0          # bi-temporal: when the note became valid
    valid_until: float | None = None # bi-temporal: None = still valid; float = superseded
    code_hash: str = ""              # sha256[:16] of the anchored code block at write time
    superseded_by: str | None = None  # author_id that superseded this note (code_hash conflict path)
    superseded_at: float | None = None
    title: str = ""                    # short label for index-tier display (UPG-RECALL-HIERARCHY)
    # Trigger engine wave 1 (TRIGGER-ENGINE) — additive memory-object fields.
    triggers: list[dict] = None        # type: ignore[assignment]  # explicit P/E/T trigger overrides; [] = use the kind's default bundle
    provenance: str = DEFAULT_PROVENANCE  # human | user-stated | agent | auto (bm2-design-skeleton.md §5)
    user_quote: str = ""               # verbatim user-turn excerpt bound to `content` at write time (UPG-MEM-PROVENANCE-USER-STATED); "" unless provenance == USER_STATED_PROVENANCE
    scope: str = DEFAULT_SCOPE            # workspace | repo | path-subtree | branch | session
    anchors: list[list[str]] = None    # type: ignore[assignment]  # [[path, content_hash_at_write_or_None, observed_bool_or_None], ...] — 3rd element (UPG-ANCHOR-UNOBSERVED-BINDING) is True/False only when this session had an observation ledger, else None (unknown); a 2-element row written before this field existed loads with no 3rd element and reads back as None via `len(anchor) >= 3` checks, never inferred as False
    supersedes: int | None = None      # note_id THIS note explicitly tombstones at write time
    superseded_by_note_id: int | None = None  # reciprocal: set on the OLD note by the explicit-supersedes path (distinct from `superseded_by`'s code_hash-conflict/author_id semantics)
    last_fired: float | None = None    # last time the trigger engine actually fired this note (T:cooldown + total-order tie-break)
    branch: str = ""                   # git branch recorded at write time when scope=="branch" (TRIGGER-ENGINE wave 2a); "" for every other scope or when git is unavailable
    pinned: bool = False               # UPG-RECALL-MISS-FLOOR part (b): explicit user/agent pin — Tier 0 unconditional-injection membership alongside kind='directive', set/cleared via vectr_pin (or vectr_remember(pin=True) at write time), never inferred

    def __post_init__(self) -> None:
        if self.triggers is None:
            self.triggers = []
        if self.anchors is None:
            self.anchors = []


@dataclass
class SnapshotEntry:
    snapshot_id: str
    workspace: str
    label: str
    notes: list[WorkingNote]
    retrieved_chunks: list[dict]   # {file, lines, symbol, content} of what was in context
    created_at: float
