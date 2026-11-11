"""
Note lifecycle event log (UPG-MEMORY-STATE-MACHINE).

A note's lifecycle is append-only: every transition — write, tombstone,
retraction, drift flag, revert, provenance bump — is recorded as one row in
`note_events` rather than mutated in place on the note itself. "Current
state" is never stored directly; it is always a fold over that log, so a
transition can be undone (`revoked` -> `reinstated`) by appending one more
event, never by rewriting history. `vectr_forget` remains the one true
hard-delete escape hatch — this log is for judgment calls that might be
wrong, not for a caller who genuinely wants the row gone.

NOTE_EVENT_KINDS is a closed protocol vocabulary — a caller either uses one
of these values or the write is rejected — not an operator-tunable
threshold, so it lives here as a plain constant rather than in
config.yaml (same reasoning already applied to VALID_KINDS/SCOPE_VALUES/
PROVENANCE_VALUES in _types.py).

Deliberately named NOTE_EVENT_KINDS, not EVENT_VALUES: `_types.py` already
exports an unrelated `EVENT_VALUES` (the trigger engine's E primitive —
session-start/prompt-submit/etc., "what lifecycle moment fired a trigger").
Reusing that name here for "what happened to a note" would be a genuine
collision between two different concepts that happen to share the word
"event" — this module picks a distinct name specifically to avoid it.
"""
from __future__ import annotations

from typing import Iterable, TypedDict


# Every lifecycle moment a note can pass through. `created` seeds the log at
# write time (redundant with the fold's own "active" default, kept so the
# log is a complete history with no implicit first row). `promoted` mirrors
# WorkingContextStore.promote()'s existing provenance step-up; it changes
# provenance, not lifecycle state, so the fold does not treat it as a
# state-changing event. `stale_flagged` and `expired` are the two
# system-actor transitions (deterministic, never a judgment call) and are
# each handled differently by the fold: staleness is re-derived fresh
# against live anchor hashes on every check_staleness() call (see that
# method), so a fixed anchor simply stops being flagged on the next check
# with no corresponding "un-stale" event needed — this is what makes it
# "automatic, reversible" per the design doc. `expired` (UPG-MEMORY-DECAY-
# KIND-SCOPED) is a genuine state-changing transition instead — a note
# whose kind's configured TTL has elapsed (`purge_expired_notes()`) is
# appended one `expired` event rather than deleted (UPG-MEMORY-STATE-
# MACHINE's append-only invariant: the row, its note_events history, and
# its deterrent rendering all survive; only proactive/default recall()/
# fire() listing excludes it) — reversed the same way `revoked` is, by
# appending a `reinstated` event (no separate "un-expire" event kind
# needed; `reinstated` already means "return to active" regardless of
# which terminal state preceded it).
# `superseded_post_hoc` (UPG-NOTE-RETIRE-POST-HOC): the same lifecycle
# transition as `superseded` — a note retires because its substance was
# replaced — but recorded when the link is asserted AFTER both notes
# already existed (the replacement was written without supersedes=, and a
# caller later judges the old note already-superseded-in-substance), via
# `WorkingContextStore.supersede_note()` / `vectr_supersede`. A distinct
# KIND, not a payload convention on `superseded`, so the audit trail stays
# machine-queryable about which retirements were post-hoc judgments; it
# folds to the SAME "superseded" state (no fourth lifecycle state is added),
# so every rendering surface treats the two identically.
NOTE_EVENT_KINDS: tuple[str, ...] = (
    "created", "superseded", "revoked", "stale_flagged", "reinstated", "promoted", "expired",
    "superseded_post_hoc",
)

# Who proposed a transition. Vectr never decides a revocation/reinstatement
# itself (judgment stays caller-side, per the design doc) — "system" is
# reserved for the deterministic, non-judgment transitions: `stale_flagged`
# (a content-hash mismatch either happened or it didn't) and `expired` (a
# note's age either crossed its kind's configured TTL or it didn't).
NOTE_EVENT_ACTORS: tuple[str, ...] = ("agent", "human", "system")

# The subset of NOTE_EVENT_KINDS that changes fold()'s lifecycle state, and
# the state each one sets. stale_flagged/promoted are intentionally absent —
# see the NOTE_EVENT_KINDS docstring above. `superseded_post_hoc` lands on
# the same "superseded" state as write-time `superseded` (UPG-NOTE-RETIRE-
# POST-HOC) — only the audit kind differs, never the resulting state.
_LIFECYCLE_STATE_AFTER: dict[str, str] = {
    "created": "active",
    "superseded": "superseded",
    "revoked": "revoked",
    "reinstated": "active",
    "expired": "expired",
    "superseded_post_hoc": "superseded",
}

# The fold's possible lifecycle states — every note is exactly one of these
# at any point in its history.
NOTE_LIFECYCLE_STATES: tuple[str, ...] = ("active", "superseded", "revoked", "expired")


class NoteEventRow(TypedDict, total=False):
    event: str
    actor: str
    reason: str | None
    payload: str | None
    ts: float
    id: int


class FoldedState(TypedDict):
    state: str
    reason: str | None
    actor: str | None
    ts: float | None


def fold(events: Iterable[NoteEventRow]) -> FoldedState:
    """Fold an append-only note_events log into the note's current
    lifecycle state (UPG-MEMORY-STATE-MACHINE §4.1).

    `events` must already be ordered oldest-first by the log's monotonic
    `id` (SQLite AUTOINCREMENT) — NOT by `ts` (wall-clock `time.time()`),
    which can collide when two events land in the same low-resolution tick
    (e.g. a `contradicts=` write appending `created` on the new note and
    `revoked` on the target in the same call). `id` is strictly increasing
    and never collides, so it is the only reliable ordering for the fold's
    tie-break.

    A note with no events at all (pre-migration rows, or a caller that
    queries before any event was ever written) folds to "active" — the
    same default a brand-new `created` event would produce, so a missing
    log is never distinguishable from an intact one that just started
    "active" and hasn't moved.

    Returns the folded state plus the reason/actor/ts of the last
    state-changing event (None for all three when nothing has changed the
    note away from its initial "active" state) — the context anti-memory
    rendering needs (revoked date + reason) without a second query.
    """
    result: FoldedState = {"state": "active", "reason": None, "actor": None, "ts": None}
    for ev in events:
        new_state = _LIFECYCLE_STATE_AFTER.get(ev["event"])
        if new_state is None:
            continue  # stale_flagged / promoted: audit-only, no state change
        result = {
            "state": new_state,
            "reason": ev.get("reason"),
            "actor": ev.get("actor"),
            "ts": ev.get("ts"),
        }
    return result
