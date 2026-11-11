"""
WorkingContextStore — persists working notes the LLM saves to Vectr.

This is the core of the bidirectional protocol. The LLM calls vectr_remember()
to store what it has learned. Vectr stores it persistently for fast recall.
vectr_recall() brings it back on demand — later in this session or in a future
one. Recall is instant (<50ms) and lossless.

Storage: SQLite in the vectr DB dir — available immediately within the same
session and persists across IDE restarts and reboots.

Package layout:
  _audit.py      — rotating audit logger (_get_audit_logger, audit)
  _types.py      — dataclasses (WorkingNote, SnapshotEntry), constants (VALID_KINDS, DEFAULT_KIND)
  _encryption.py — field-level encryption (_NoteEncryptor, _build_encryptor,
                   _FILE_PATH_RE, _extract_file_paths)
  _events.py     — note lifecycle event log (NOTE_EVENT_KINDS, NOTE_EVENT_ACTORS,
                   NOTE_LIFECYCLE_STATES, fold()) — UPG-MEMORY-STATE-MACHINE
  _content_file.py — content_file path resolution + read (resolve_remember_content,
                   read_content_file) — UPG-REMEMBER-MCP-LONG-PAYLOAD-PARSE-LOSS
  _store.py      — WorkingContextStore class (full store API)

All names that existed on the flat agent/working_context_store.py module are
re-exported here so every existing import site keeps working unchanged:
  from agent.working_context_store import WorkingContextStore
  from agent.working_context_store import WorkingNote
  from agent.working_context_store import SnapshotEntry
  from agent.working_context_store import VALID_KINDS
  from agent.working_context_store import DEFAULT_KIND
  from agent.working_context_store import audit
  from agent.working_context_store import _get_audit_logger
  from agent.working_context_store import _NoteEncryptor
  from agent.working_context_store import _build_encryptor
  from agent.working_context_store import _extract_file_paths
  from agent.working_context_store import _FILE_PATH_RE
  from agent.working_context_store import _note_title
  from agent.working_context_store import _path_boundary_match
  from agent.working_context_store import _anchors_exact_match
"""
from __future__ import annotations

# Audit logger
from agent.working_context_store._audit import (
    _get_audit_logger,
    audit,
    reset_audit_client,
    set_audit_client,
)

# Types and constants
from agent.working_context_store._types import (
    DEFAULT_KIND,
    DEFAULT_PROVENANCE,
    DEFAULT_SCOPE,
    EVENT_VALUES,
    PROMOTION_LADDER,
    PROMOTION_RANK,
    PROVENANCE_VALUES,
    SCOPE_VALUES,
    SORT_BY_VALUES,
    USER_STATED_PROVENANCE,
    VALID_KINDS,
    SnapshotEntry,
    WorkingNote,
)

# Verbatim user-excerpt binding (UPG-MEM-PROVENANCE-USER-STATED) and its
# harness-driven auto-bind counterpart (UPG-PROVENANCE-NEVER-RISES)
from agent.working_context_store._user_quote import (
    AUTO_QUOTE_NOT_CONTAINED,
    AUTO_QUOTE_NO_RECENT_TURN,
    AUTO_QUOTE_TOO_LARGE_TO_SEARCH,
    AUTO_QUOTE_TOO_SHORT,
    USER_QUOTE_NOT_CONTAINED,
    USER_QUOTE_TOO_SHORT,
    _longest_common_span,
    _normalize_with_offsets,
    bind_user_quote,
    bind_user_quote_auto,
    normalize_for_binding,
)

# content_file resolution (UPG-REMEMBER-MCP-LONG-PAYLOAD-PARSE-LOSS)
from agent.working_context_store._content_file import (
    read_content_file,
    resolve_content_file_path,
    resolve_remember_content,
)

# Encryption helpers
from agent.working_context_store._encryption import (
    _FILE_PATH_RE,
    _NoteEncryptor,
    _build_encryptor,
    _extract_file_paths,
)

# Note lifecycle event log (UPG-MEMORY-STATE-MACHINE)
from agent.working_context_store._events import (
    NOTE_EVENT_ACTORS,
    NOTE_EVENT_KINDS,
    NOTE_LIFECYCLE_STATES,
    fold as fold_note_events,
)

# Store class
from agent.working_context_store._store import (
    WorkingContextStore,
    _anchors_exact_match,
    _note_title,
    _path_boundary_match,
)

__all__ = [
    # Audit
    "_get_audit_logger",
    "audit",
    "set_audit_client",
    "reset_audit_client",
    # Types and constants
    "DEFAULT_KIND",
    "VALID_KINDS",
    # TRIGGER-ENGINE wave 1 (bm2-design-skeleton.md §1/§2/§5)
    "EVENT_VALUES",
    "SCOPE_VALUES",
    "DEFAULT_SCOPE",
    "PROVENANCE_VALUES",
    "DEFAULT_PROVENANCE",
    # Shared recall sort-mode vocabulary (UPG-SORTBY-SHARED-VOCAB) —
    # imported by REST, MCP dispatch + tool schema, and the CLI so a fifth
    # sort mode can be added at one place, not five.
    "SORT_BY_VALUES",
    "SnapshotEntry",
    "WorkingNote",
    # Verbatim user-excerpt binding (UPG-MEM-PROVENANCE-USER-STATED)
    "USER_STATED_PROVENANCE",
    "PROMOTION_LADDER",
    "PROMOTION_RANK",
    "USER_QUOTE_NOT_CONTAINED",
    "USER_QUOTE_TOO_SHORT",
    "bind_user_quote",
    "normalize_for_binding",
    # Harness-driven auto-bind counterpart (UPG-PROVENANCE-NEVER-RISES)
    "AUTO_QUOTE_NOT_CONTAINED",
    "AUTO_QUOTE_NO_RECENT_TURN",
    "AUTO_QUOTE_TOO_SHORT",
    "bind_user_quote_auto",
    # Longest-verbatim-span auto-bind (UPG-PROVENANCE-AUTOBIND-SPAN)
    "AUTO_QUOTE_TOO_LARGE_TO_SEARCH",
    "_normalize_with_offsets",
    "_longest_common_span",
    # content_file resolution (UPG-REMEMBER-MCP-LONG-PAYLOAD-PARSE-LOSS)
    "read_content_file",
    "resolve_content_file_path",
    "resolve_remember_content",
    # Encryption
    "_FILE_PATH_RE",
    "_NoteEncryptor",
    "_build_encryptor",
    "_extract_file_paths",
    # Note lifecycle event log (UPG-MEMORY-STATE-MACHINE)
    "NOTE_EVENT_KINDS",
    "NOTE_EVENT_ACTORS",
    "NOTE_LIFECYCLE_STATES",
    "fold_note_events",
    # Store
    "WorkingContextStore",
    "_note_title",
    "_path_boundary_match",
    "_anchors_exact_match",
]
