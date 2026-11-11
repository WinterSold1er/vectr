"""
Transactional check-out / check-in editing of vectr working memory
(UPG-MEMORY-LEGIBLE-FILE-PROJECTION part (b)).

Reuses part (a)'s rendering path (`agent/working_context_store/_export.py`:
`render_memory_markdown()`, `_render_note_block()`, `_group_by_kind()`) for
the SAME buffer grammar (`### [#N] Title` heading + metadata bullets + body
per note, grouped into `##` kind sections) rather than forking a second
renderer — see `render_memory_markdown()`'s `header_comment` parameter,
added specifically to let this module reuse it with its own header.

Shape (settled design, task item's own wording):
  1. Render live notes to a temp file carrying stable `[#N]` id anchors.
  2. Open `$EDITOR`.
  3. On save, diff the buffer against EXACTLY the snapshot it rendered and
     translate the diff into note operations.

Mapping:
  - body changed                        -> SUPERSEDE (`remember(supersedes=
    ...)`; NEVER an in-place content mutation — the append-only lifecycle
    invariant, UPG-MEMORY-STATE-MACHINE §4.1 "Option E", is the single most
    important correctness property in this module).
  - kind/priority/tags changed, body unchanged -> FIELD_UPDATE (`update_
    note_fields()` — a plain column UPDATE, since those columns carry no
    lifecycle-state meaning of their own).
  - new block, no `[#N]`                -> REMEMBER (a fresh note).
  - block deleted from the buffer       -> REVOKE (default, reversible via
    `vectr_reinstate`) or FORGET (only via the explicit marker below).
  - buffer byte-identical to the snapshot it was rendered from -> zero
    writes of any kind.

Two-writer-race handling (never a timestamp heuristic): `capture_edit_
snapshot()` fingerprints every rendered note (content, kind, priority,
tags) at render time. `apply_edit_plan()` re-reads and re-fingerprints
every note the plan TOUCHES — never an untouched one — immediately before
issuing a single write; any mismatch raises `MemoryEditConflict` naming
every drifted id, and nothing is written (no partial apply, no silent pick
of either side).

Testable seam: `plan_edit_operations()` is pure — no store access, no clock
reads, no randomness — so every acceptance test drives it directly.
`edit_memory_interactive()` is the thin interactive wrapper around it, and
its own `$EDITOR`-launching step is swapped out in tests via the injectable
`launch_editor` parameter.

Known limitation (inherited from part (a)'s markdown round-trip design,
not introduced here): a note whose CONTENT itself contains a line starting
with `##` or `###` (e.g. an embedded markdown snippet) will confuse the
block boundary parser, the same class of edge case a git commit message
has with lines starting with `#`. Not hardened against here — flagged for
anyone extending this module.
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from agent.working_context_store._export import render_memory_markdown
from agent.working_context_store._types import DEFAULT_KIND

if TYPE_CHECKING:
    from agent.working_context_store._store import WorkingContextStore
    from agent.working_context_store._types import WorkingNote

# ---------------------------------------------------------------------------
# Closed protocol vocabulary — fixed sentinel strings, not operator-tunable
# weights/thresholds, so these live here as plain constants rather than in
# config.yaml (same precedent as VALID_KINDS/_HEADER_COMMENT/
# _EXPORT_MARKER_REL: there is nothing here for an operator to retune, only
# a fixed protocol to document and test).
# ---------------------------------------------------------------------------

# The one deliberate, hard-to-type-by-accident escape hatch to a real
# `forget()` hard delete (UPG-MEMORY-LEGIBLE-FILE-PROJECTION part (b),
# design call 1). Must appear as a metadata bullet on a block whose `[#N]`
# heading is still present — mutually exclusive BY CONSTRUCTION with the
# "block absent from the buffer" deletion path (a fully deleted block has
# nowhere left to carry this line), which is what makes ordinary deletion
# (-> revoke, reversible) and this marker (-> forget, irreversible) two
# distinct, deliberately-different-effort actions.
FORGET_META_KEY = "forget"
FORGET_MARKER_VALUE = "PERMANENTLY-DELETE-CONFIRMED"
FORGET_MARKER_LINE = f"- {FORGET_META_KEY}: {FORGET_MARKER_VALUE}"

# Default revocation reason for a block that is simply absent from the
# saved buffer (the ordinary, reversible deletion path).
DELETE_REVOKE_REASON = "removed in memory edit session"

# Metadata bullet keys `_render_note_block()` (agent/working_context_store/
# _export.py) can emit, plus the forget marker above — the parser only
# treats a leading run of `- key: value` lines as this block's metadata
# section when every key is one of these; an unrecognized bullet ends the
# metadata section and folds back into the body instead (never silently
# swallowed), so a note whose body legitimately starts with a bullet list
# is not misparsed.
_KNOWN_META_KEYS = frozenset(
    {"kind", "priority", "created", "state", "tags", "author", "provenance", FORGET_META_KEY}
)

EDIT_HEADER_COMMENT = (
    "<!-- vectr memory edit -- check-out/check-in buffer, ACTIVE notes only "
    "(superseded and revoked notes are omitted here; see `vectr memory "
    "export` for the full audit mirror). Keep a block's `[#N]` heading "
    "intact to edit that note: changing its body supersedes it (the "
    "original row is kept, never overwritten, per vectr's append-only "
    "memory invariant); changing its `kind`/`priority`/`tags` bullet "
    "updates that field in place. Delete an entire block to revoke that "
    "note (reversible with vectr_reinstate). Add a new `### Title` block "
    "with no `[#N]` to remember it as a new note. To PERMANENTLY delete a "
    "note instead of revoking it (irreversible, bypasses vectr_reinstate), "
    "add the bullet `" + FORGET_MARKER_LINE + "` to its metadata before "
    "saving -- exact text, case-sensitive, and the block's heading must "
    "still be present. Saving with no changes writes nothing. -->"
)


class MemoryEditError(Exception):
    """Base class for every `vectr memory edit` failure."""


class MemoryEditParseError(MemoryEditError):
    """The saved buffer does not parse as a valid memory-edit buffer (a
    duplicate `[#N]`, or an `[#N]` that names no note in the rendered
    snapshot)."""


class MemoryEditConflict(MemoryEditError):
    """A note the buffer touches drifted in the store since the buffer was
    rendered (design call 3). Carries every conflicting note id; raising
    this means NOTHING was written — apply_edit_plan() checks every touched
    note's fingerprint before issuing its first write, not after."""

    def __init__(self, note_ids: list[int]):
        self.note_ids = sorted(set(note_ids))
        ids_text = ", ".join(f"#{i}" for i in self.note_ids)
        super().__init__(
            f"memory edit conflict: note(s) {ids_text} changed since this buffer "
            "was rendered -- nothing was written; re-run `vectr memory edit` to "
            "see current state and redo your changes"
        )


def _note_fingerprint(note: "WorkingNote") -> tuple:
    """Everything an edit session can change about a note, as a comparable
    tuple — the conflict check's unit of "did this note actually change".
    Deliberately excludes last_accessed/decay_score/etc, which drift on
    ordinary reads and firings and would falsely flag every session as
    conflicting with itself."""
    return (note.content, note.kind, note.priority, tuple(note.tags or []))


def _parse_tags_value(raw: str) -> list[str]:
    """Inverse of `_render_note_block`'s `", ".join(note.tags)` — comma-
    separated, each entry stripped, empty entries dropped."""
    return [t.strip() for t in raw.split(",") if t.strip()]


@dataclass
class EditSnapshot:
    """Exactly what a rendered edit buffer showed, captured at render time
    — the only thing `plan_edit_operations()`/`apply_edit_plan()` ever diff
    against (design call 3: real state, never a timestamp heuristic)."""

    workspace: str
    notes: list["WorkingNote"]  # notes_for_edit() result, note_id ascending
    buffer_text: str  # exactly what was written to the temp file

    def by_id(self) -> dict[int, "WorkingNote"]:
        return {n.note_id: n for n in self.notes}

    def fingerprint(self, note_id: int) -> tuple | None:
        note = self.by_id().get(note_id)
        return None if note is None else _note_fingerprint(note)


def capture_edit_snapshot(store: "WorkingContextStore", workspace: str) -> EditSnapshot:
    """Render `workspace`'s ACTIVE-only notes (design call 2: `notes_for_
    edit()`, deliberately narrower than the export path's `notes_for_
    export()`) into an `EditSnapshot` — the single render call both the
    interactive wrapper and every acceptance test share, so a test's
    `plan_edit_operations()` call sees literally the same snapshot shape a
    real editing session would."""
    notes = store.notes_for_edit(workspace)
    note_states = store.note_event_states(workspace, notes)
    buffer_text = render_memory_markdown(notes, note_states, header_comment=EDIT_HEADER_COMMENT)
    return EditSnapshot(workspace=workspace, notes=notes, buffer_text=buffer_text)


@dataclass
class ParsedBlock:
    """One `### [#N] Title` block extracted from a saved buffer.
    `note_id` is None for a block with no `[#N]` anchor (a new note)."""

    note_id: int | None
    title: str
    meta: dict[str, str]
    body: str


# Any `##` or `###` heading line — the two boundary markers that can end a
# note block's span (a new note heading, or the next `## <Kind>` section
# heading). Deliberately excludes single-`#` (`# Working Memory`, the
# document title) so that never terminates a block early.
_BOUNDARY_RE = re.compile(r"^(#{2,3}) (.*)$", re.MULTILINE)
_NOTE_HEADING_RE = re.compile(r"^(?:\[#(\d+)\]\s*)?(.*)$")


def _split_meta_and_body(block_text: str) -> tuple[dict[str, str], str]:
    """Split a block's content (everything after its `### ` heading line)
    into its metadata-bullet dict and its body. A leading contiguous run of
    `- key: value` lines, where every key is in `_KNOWN_META_KEYS`, is the
    metadata section (mirroring exactly what `_render_note_block()` emits,
    plus the forget marker a caller may add by hand); the first line that
    is blank or does not match ends it. A block with no recognizable
    metadata bullets at all (the common case for a brand-new block the
    editor just typed a title and body for) yields an empty meta dict and
    the entire text as body — never misparsed as headerless metadata."""
    lines = block_text.split("\n")
    meta: dict[str, str] = {}
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            break
        if not line.startswith("-"):
            break
        key, sep, value = line[1:].partition(":")
        key = key.strip().lower()
        if not sep or key not in _KNOWN_META_KEYS:
            break
        meta[key] = value.strip()
        idx += 1
    body = "\n".join(lines[idx:]).strip()
    return meta, body


def parse_edit_buffer(text: str) -> list[ParsedBlock]:
    """Pure parser over the buffer's own serialization format (the shape
    `render_memory_markdown()`/`_render_note_block()` produce) — this is
    parsing vectr's own rendered grammar, not a query-content heuristic."""
    boundaries = list(_BOUNDARY_RE.finditer(text))
    blocks: list[ParsedBlock] = []
    for i, m in enumerate(boundaries):
        marker = m.group(1)
        if marker != "###":
            continue
        heading_match = _NOTE_HEADING_RE.match(m.group(2))
        raw_id = heading_match.group(1) if heading_match else None
        note_id = int(raw_id) if raw_id is not None else None
        title = (heading_match.group(2) if heading_match else m.group(2)).strip()
        content_start = m.end()
        content_end = boundaries[i + 1].start() if i + 1 < len(boundaries) else len(text)
        block_text = text[content_start:content_end].strip("\n")
        meta, body = _split_meta_and_body(block_text)
        blocks.append(ParsedBlock(note_id=note_id, title=title, meta=meta, body=body))
    return blocks


@dataclass
class Operation:
    """One planned note write. `kind` is one of "supersede", "field_update",
    "remember", "revoke", "forget"."""

    kind: str
    note_id: int | None = None
    content: str | None = None
    title: str = ""
    new_kind: str | None = None
    new_priority: str | None = None
    new_tags: list[str] | None = None
    reason: str | None = None


@dataclass
class EditPlan:
    """The pure translation output: what to write, plus which existing note
    ids the plan TOUCHES (referenced by any operation) — `apply_edit_plan()`
    re-checks exactly this set for concurrent drift before writing anything.
    A note never mentioned here is never even read by apply — it drifting
    concurrently is not a conflict (design call 3)."""

    operations: list[Operation] = field(default_factory=list)
    touched_note_ids: list[int] = field(default_factory=list)


def plan_edit_operations(snapshot: EditSnapshot, buffer_text: str) -> EditPlan:
    """Pure translation: `snapshot` (exactly what was rendered) + `buffer_
    text` (exactly what came back from the editor) -> `EditPlan`. No store
    access, no clock reads, no randomness (design call 4) — every
    acceptance test drives this function directly, never `$EDITOR`.

    Raises `MemoryEditParseError` for a buffer that cannot map back onto
    the snapshot it claims to be an edit of: the same `[#N]` appearing on
    two blocks, or an `[#N]` naming a note absent from `snapshot` entirely.
    """
    if buffer_text == snapshot.buffer_text:
        return EditPlan()

    blocks = parse_edit_buffer(buffer_text)
    by_id = snapshot.by_id()

    seen_ids: set[int] = set()
    for b in blocks:
        if b.note_id is None:
            continue
        if b.note_id in seen_ids:
            raise MemoryEditParseError(f"note #{b.note_id} appears more than once in the buffer")
        seen_ids.add(b.note_id)
        if b.note_id not in by_id:
            raise MemoryEditParseError(
                f"note #{b.note_id} does not match any note in the rendered snapshot "
                "-- re-run `vectr memory edit` and edit only the blocks it rendered"
            )

    deleted_ids = set(by_id.keys()) - seen_ids

    operations: list[Operation] = []
    touched: list[int] = []

    for b in blocks:
        if b.note_id is None:
            # New block -> fresh remember. The buffer format doesn't ask a
            # new block to declare kind/priority/tags; it gets remember()'s
            # own defaults, same as an MCP vectr_remember call with none
            # of those passed.
            operations.append(Operation(kind="remember", content=b.body, title=b.title))
            continue

        original = by_id[b.note_id]

        if b.meta.get(FORGET_META_KEY) == FORGET_MARKER_VALUE:
            operations.append(Operation(kind="forget", note_id=b.note_id))
            touched.append(b.note_id)
            continue

        body_changed = b.body != original.content.strip()
        new_kind = b.meta.get("kind", original.kind)
        new_priority = b.meta.get("priority", original.priority)
        if "tags" in b.meta:
            new_tags = _parse_tags_value(b.meta["tags"])
        elif original.tags:
            # The tags bullet WAS rendered (tags were non-empty) and is now
            # gone from the buffer -> its absence is the deletion, not "no
            # opinion". A note with no tags never rendered a bullet at all,
            # so its continued absence here is simply "unchanged".
            new_tags = []
        else:
            new_tags = list(original.tags or [])

        kind_changed = new_kind != original.kind
        priority_changed = new_priority != original.priority
        tags_changed = list(new_tags) != list(original.tags or [])

        if body_changed:
            # `title` is deliberately NOT taken from the parsed heading
            # here (unlike the new-block "remember" path below): title
            # isn't in the editable-field list this module supports (only
            # body/kind/priority/tags are), so a heading-text edit on an
            # EXISTING block is ignored and the original note's own title
            # is carried forward verbatim onto the superseding note —
            # exact and deterministic, unlike letting remember() re-derive
            # a fresh title from the new content (which could silently
            # diverge from a caller's explicitly-set custom title).
            operations.append(Operation(
                kind="supersede", note_id=b.note_id, content=b.body, title=original.title,
                new_kind=new_kind, new_priority=new_priority, new_tags=new_tags,
            ))
            touched.append(b.note_id)
        elif kind_changed or priority_changed or tags_changed:
            operations.append(Operation(
                kind="field_update", note_id=b.note_id,
                new_kind=new_kind if kind_changed else None,
                new_priority=new_priority if priority_changed else None,
                new_tags=new_tags if tags_changed else None,
            ))
            touched.append(b.note_id)
        # else: this block round-tripped unchanged -> no operation.

    for note_id in sorted(deleted_ids):
        operations.append(Operation(kind="revoke", note_id=note_id, reason=DELETE_REVOKE_REASON))
        touched.append(note_id)

    return EditPlan(operations=operations, touched_note_ids=touched)


@dataclass
class ApplyResult:
    """What `apply_edit_plan()` actually wrote, one list per operation
    kind — the CLI's post-apply summary reads straight off this."""

    supersedes: list[tuple[int, int]] = field(default_factory=list)  # (old_id, new_id)
    field_updates: list[int] = field(default_factory=list)
    remembers: list[int] = field(default_factory=list)  # new ids
    revokes: list[int] = field(default_factory=list)
    forgets: list[int] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (
            self.supersedes or self.field_updates or self.remembers
            or self.revokes or self.forgets
        )


def apply_edit_plan(
    store: "WorkingContextStore",
    workspace: str,
    snapshot: EditSnapshot,
    plan: EditPlan,
    provenance: str = "human",
    actor: str = "human",
    author_id: str = "",
) -> ApplyResult:
    """The only function in this module that writes. Re-reads and
    re-fingerprints every note `plan.touched_note_ids` names BEFORE issuing
    a single write (design call 3): any drift from `snapshot`'s own
    fingerprint of that note raises `MemoryEditConflict` naming every
    conflicting id, and applies NOTHING — no partial apply, no silent pick
    of either side. A note the buffer never touched is never read here, so
    it drifting concurrently (someone else's unrelated write, or this
    note's own ordinary decay/last_accessed churn) is never a conflict.

    `provenance`/`actor` default to "human": `vectr memory edit` is a
    person editing memory directly through `$EDITOR`, which also satisfies
    `remember(supersedes=...)`'s existing write-boundary guard (a write
    whose own provenance is not "human" may not supersede a provenance=
    "human" target note) — a human editing session must default to human
    provenance or it would be blocked from editing its own prior
    human-authored notes.
    """
    result = ApplyResult()
    if not plan.operations:
        return result

    conflicting: list[int] = []
    for note_id in plan.touched_note_ids:
        fresh = store.get_note(workspace, note_id)
        fresh_fp = _note_fingerprint(fresh) if fresh is not None else None
        if fresh_fp != snapshot.fingerprint(note_id):
            conflicting.append(note_id)
    if conflicting:
        raise MemoryEditConflict(conflicting)

    for op in plan.operations:
        if op.kind == "supersede":
            new_id = store.remember(
                workspace, op.content or "", tags=op.new_tags,
                priority=op.new_priority or "medium", kind=op.new_kind or DEFAULT_KIND,
                title=op.title, provenance=provenance, author_id=author_id,
                supersedes=op.note_id,
            )
            result.supersedes.append((op.note_id, new_id))
        elif op.kind == "field_update":
            store.update_note_fields(
                workspace, op.note_id, kind=op.new_kind, priority=op.new_priority,
                tags=op.new_tags,
            )
            result.field_updates.append(op.note_id)
        elif op.kind == "remember":
            new_id = store.remember(
                workspace, op.content or "", title=op.title,
                provenance=provenance, author_id=author_id,
            )
            result.remembers.append(new_id)
        elif op.kind == "revoke":
            store.revoke_note(workspace, op.note_id, reason=op.reason or DELETE_REVOKE_REASON, actor=actor)
            result.revokes.append(op.note_id)
        elif op.kind == "forget":
            store.forget(workspace, op.note_id)
            result.forgets.append(op.note_id)
        else:
            raise MemoryEditError(f"unknown operation kind: {op.kind!r}")
    return result


def _default_launch_editor(path: Path) -> None:
    """Launch `$EDITOR` (or `$VISUAL`, or `vi`) on `path` and block until it
    exits. Not unit-tested directly — `edit_memory_interactive()`'s
    `launch_editor` parameter exists precisely so tests never have to call
    this function (design call 4)."""
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "vi"
    cmd = shlex.split(editor) + [str(path)]
    subprocess.run(cmd, check=True)


def edit_memory_interactive(
    store: "WorkingContextStore",
    workspace: str,
    launch_editor: Callable[[Path], None] = _default_launch_editor,
    provenance: str = "human",
    actor: str = "human",
    author_id: str = "",
) -> ApplyResult:
    """Thin interactive wrapper: render -> tempfile -> launch_editor -> read
    back -> plan -> apply. All the actual logic lives in the pure functions
    above; this function's only job is temp-file/editor-process plumbing.
    `launch_editor` is injectable so a test can drive the whole round trip
    (parse real buffer text, apply real writes) without spawning a real
    editor process — it just needs to mutate the file at the given path."""
    snapshot = capture_edit_snapshot(store, workspace)
    fd, tmp_path_str = tempfile.mkstemp(prefix="vectr-memory-edit-", suffix=".md")
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(snapshot.buffer_text)
        launch_editor(tmp_path)
        buffer_text = tmp_path.read_text(encoding="utf-8")
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass
    plan = plan_edit_operations(snapshot, buffer_text)
    return apply_edit_plan(
        store, workspace, snapshot, plan,
        provenance=provenance, actor=actor, author_id=author_id,
    )
