"""Opt-in rotating audit logger for vectr working-context and index events.

OFF BY DEFAULT. Enable by setting VECTR_AUDIT_LOG to a file path; when unset,
nothing is recorded (a NullHandler). When enabled, logs remember/recall/forget/
purge/index/search events to that local file — never transmitted anywhere.
Rotates at 10 MB; keeps 3 backups.

In team mode, each line optionally carries a `client=<label>` attribution taken
from the connecting client's X-Vectr-Client header (see set_audit_client), so
the operator can see what each client indexed/recalled — still a local file on
the server host only.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import threading
import time
from contextvars import ContextVar
from pathlib import Path

# Guards lazy handler initialization: without it, two threads auditing
# concurrently can both see an empty handler list and both attach a file
# handler, duplicating every subsequent line.
_init_lock = threading.Lock()


# Per-request client attribution label (team mode). Set at the MCP dispatch
# boundary from the X-Vectr-Client header; read by audit() so every event a
# request triggers is attributed. A ContextVar (not a global) so concurrent
# requests never bleed each other's label.
_audit_client: ContextVar[str] = ContextVar("vectr_audit_client", default="")


def set_audit_client(label: str):
    """Set the client-attribution label for audit lines emitted on this task.
    Returns the ContextVar token so the caller can reset() it afterwards."""
    return _audit_client.set(label or "")


def reset_audit_client(token) -> None:
    """Restore the previous client-attribution label."""
    try:
        _audit_client.reset(token)
    except Exception:
        pass


def _get_audit_logger() -> logging.Logger:
    """Return the vectr audit logger (lazy-initialised, singleton per process).

    Disabled unless VECTR_AUDIT_LOG names a file path (opt-in)."""
    name = "vectr.audit"
    log = logging.getLogger(name)
    if log.handlers:
        return log

    with _init_lock:
        if log.handlers:  # another thread initialized while we waited
            return log
        return _init_audit_logger(log)


def _init_audit_logger(log: logging.Logger) -> logging.Logger:
    """Attach the configured handler to `log` (caller holds _init_lock)."""
    log_path_str = os.getenv("VECTR_AUDIT_LOG", "")  # opt-in: unset ⇒ disabled
    if not log_path_str:
        log.addHandler(logging.NullHandler())
        return log

    log_path = Path(log_path_str)
    # The audit file records query text — keep its directory owner-only (0700)
    # on POSIX hosts, same as the rest of vectr's state.
    try:
        from agent.fs_permissions import secure_dir
        secure_dir(log_path.parent)
    except Exception:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.handlers.RotatingFileHandler(
        str(log_path), maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8",
    )
    # UPG-AUDIT-LOCAL-TIME-Z: the trailing "Z" in the datefmt means UTC, but
    # logging.Formatter stamps LOCAL time unless told otherwise — so the
    # file carried local wall-clock under a Z that claimed otherwise
    # (observed 03:12:39Z for a true 21:42:44Z), silently skewing any
    # audit-to-transcript correlation by the host's UTC offset. Pin the
    # formatter's converter to gmtime so the stamp is true UTC and the
    # suffix is honest. Lines written before this fix in a still-rotating
    # pre-existing audit file are local time despite their Z; rotation
    # (10 MB x 3 backups) ages them out, and per-run harness audit logs are
    # always fresh artifacts, so no format change or marker scheme is
    # warranted.
    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ")
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False
    return log


def audit(event: str, **kwargs) -> None:
    """Write one audit log entry: `event key=val key=val …` (no-op when disabled).

    A non-empty client-attribution label (set_audit_client) is appended as
    `client=<label>`."""
    try:
        parts = [event] + [f"{k}={v}" for k, v in kwargs.items()]
        client = _audit_client.get()
        if client:
            parts.append(f"client={client}")
        _get_audit_logger().info(" ".join(parts))
    except Exception:
        pass  # audit failures must never crash the main path
