"""content_file — an on-disk alternative to vectr_remember's `content` arg.

UPG-REMEMBER-MCP-LONG-PAYLOAD-PARSE-LOSS: an AI code editor emits an MCP
tool-call argument by streaming model-generated JSON. A large, escape-dense
note body (nested quoted code, backslash escapes, non-ASCII punctuation) can
be corrupted mid-stream before it ever reaches vectr — the call fails with an
`InputValidationError` at the editor's own tool-input layer, and vectr never
even sees it (nothing to fix server-side; the input never arrives). This is
exactly the payload shape vectr's own prompts prescribe ("store the actual
code block, not a file pointer"), so the flagship use case is the one most
exposed.

`content_file` sidesteps the failure mode entirely: the caller writes the
note body to a file first (an ordinary tool call, not a long streamed JSON
string argument) and passes the PATH instead of the body. The path itself is
a short, escape-free string, so it survives the same JSON-streaming step that
corrupts a long body. This module resolves that path and reads the body —
used identically by the MCP `vectr_remember` tool and the REST
`POST /v1/remember` route, so the two surfaces enforce exactly the same
rules.

A vectr instance can serve several workspace roots at once (a primary root
plus `extra_roots`, e.g. a multi-root IDE workspace) — a file the caller
wants to read may live under any of them, not only the primary, so
containment is checked against every served root (see
`resolve_content_file_path`'s docstring for the exact relative-path rule).

UPG-REMEMBER-CONTENT-FILE-PATH-REFUSAL: an operator may also opt in to a
configured set of ADDITIONAL roots via
`MEMORY_WRITE_CONTENT_FILE_ADDITIONAL_READABLE_ROOTS` (config-only, set at
daemon startup, not per call). When that tuple is non-empty, paths resolving
under any of those roots are accepted in addition to the served workspace
roots — same symlink-following, same containment check, same refusal
wording. The trust decision is the operator's, made once; a calling agent
cannot widen containment per call.
"""
from __future__ import annotations

from pathlib import Path

from agent.config import (
    MEMORY_WRITE_CONTENT_FILE_ADDITIONAL_READABLE_ROOTS,
    MEMORY_WRITE_CONTENT_FILE_MAX_BYTES,
)


def resolve_content_file_path(
    workspace_root: str,
    raw_path: str,
    extra_roots: list[str] | None = None,
    additional_readable_roots: list[str] | None = None,
) -> Path:
    """Resolve `raw_path` to an absolute `Path` inside a served workspace
    root — `workspace_root` (the PRIMARY root), any of `extra_roots` (the
    additional roots a multi-root vectr instance serves alongside it, e.g.
    the other folders of a multi-root IDE workspace), or any of
    `additional_readable_roots` (UPG-REMEMBER-CONTENT-FILE-PATH-REFUSAL —
    the operator-configured extra readable roots).

    Relative-path rule: a relative `raw_path` is always resolved relative to
    `workspace_root` (the primary root) only — it is never trial-resolved
    against each of `extra_roots` or `additional_readable_roots` in turn,
    so a relative path never becomes an implicit search across every served
    root. `extra_roots` and `additional_readable_roots` exist so an
    ABSOLUTE path that already lives inside another accepted root is
    accepted, not so a relative path can name a file by searching for it.

    Either way, the RESOLVED path must land inside `workspace_root`,
    `extra_roots`, or `additional_readable_roots`; a `..` escape, or a
    symlink that resolves outside every accepted root, raises `ValueError`
    naming the original argument, the path it resolved to, and every root
    that was checked — never silently reading a file outside the accepted
    set (e.g. `/etc/passwd`, another project's tree). Resolution (which
    follows symlinks) always happens before this containment check, so a
    symlink inside an accepted root that points outside every accepted
    root is still rejected.

    `additional_readable_roots=None` (the default) means "use the
    operator-configured default"
    (`MEMORY_WRITE_CONTENT_FILE_ADDITIONAL_READABLE_ROOTS` from
    agent.config). Pass an explicit list to override the configured default
    for one call (e.g. a test pinning the off-by-default behaviour without
    mutating the config singleton).
    """
    root = Path(workspace_root).resolve()
    served_roots = [root] + [Path(r).resolve() for r in (extra_roots or [])]
    if additional_readable_roots is None:
        additional_readable_roots = list(
            MEMORY_WRITE_CONTENT_FILE_ADDITIONAL_READABLE_ROOTS
        )
    for r in additional_readable_roots:
        served_roots.append(Path(r).resolve())
    candidate = Path(raw_path)
    unresolved = candidate if candidate.is_absolute() else (root / candidate)
    try:
        resolved = unresolved.resolve()
    except OSError:
        # A component that cannot be stat'd (e.g. a broken symlink) still
        # yields a resolved Path via strict=False (the default) on modern
        # Python; this except only guards older/edge platform behavior. Fall
        # back to the unresolved candidate so the containment check below
        # still runs and reports a sensible path.
        resolved = unresolved
    for served_root in served_roots:
        try:
            resolved.relative_to(served_root)
            return resolved
        except ValueError:
            continue
    roots_desc = ", ".join(f"'{r}'" for r in served_roots)
    raise ValueError(
        f"content_file '{raw_path}' resolves to '{resolved}', which is "
        f"outside every workspace root this vectr instance serves "
        f"({roots_desc}) -- refusing to read a file outside the workspace. "
        "Pass a path inside a served workspace root, absolute or relative "
        "to the primary root."
    )


def read_content_file(
    workspace_root: str,
    raw_path: str,
    extra_roots: list[str] | None = None,
    additional_readable_roots: list[str] | None = None,
) -> str:
    """Read and UTF-8-decode the note body at `raw_path`, resolved against
    `workspace_root`, `extra_roots`, or `additional_readable_roots` (see
    `resolve_content_file_path` for the exact multi-root resolution rule).

    Raises `ValueError` — never a raw `OSError`/`UnicodeDecodeError` — on
    each distinct failure, always naming the resolved path:
      - the path escapes every accepted root
      - the resolved path does not exist, or is not a regular file
      - the file is larger than `MEMORY_WRITE_CONTENT_FILE_MAX_BYTES`
      - the file's bytes are not valid UTF-8
      - the file is empty (zero bytes)

    The returned text is the file's exact decoded contents — no stripping,
    no normalization. Callers apply whatever transform they already apply to
    an inline `content` argument (MCP's `vectr_remember` strips leading/
    trailing whitespace off `content` today; the REST `/v1/remember` route
    does not), so `content_file` behaves identically to `content` at each
    surface rather than introducing a third, file-specific convention.
    """
    resolved = resolve_content_file_path(
        workspace_root, raw_path, extra_roots, additional_readable_roots,
    )
    if not resolved.exists():
        raise ValueError(
            f"content_file '{raw_path}' does not exist (resolved to "
            f"'{resolved}')."
        )
    if not resolved.is_file():
        raise ValueError(
            f"content_file '{raw_path}' (resolved to '{resolved}') is not a "
            "regular file."
        )
    try:
        size = resolved.stat().st_size
    except OSError as exc:
        raise ValueError(
            f"content_file '{raw_path}' (resolved to '{resolved}') could "
            f"not be read: {exc.strerror or exc}."
        ) from exc
    if size > MEMORY_WRITE_CONTENT_FILE_MAX_BYTES:
        raise ValueError(
            f"content_file '{raw_path}' (resolved to '{resolved}') is "
            f"{size} bytes, over the {MEMORY_WRITE_CONTENT_FILE_MAX_BYTES}-"
            f"byte limit ({MEMORY_WRITE_CONTENT_FILE_MAX_BYTES // 1024} "
            "KiB) -- split the note across multiple content_file calls or "
            "trim the file."
        )
    try:
        raw_bytes = resolved.read_bytes()
    except OSError as exc:
        raise ValueError(
            f"content_file '{raw_path}' (resolved to '{resolved}') could "
            f"not be read: {exc.strerror or exc}."
        ) from exc
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"content_file '{raw_path}' (resolved to '{resolved}') is not "
            f"valid UTF-8 (decode failed at byte offset {exc.start}: "
            f"{exc.reason}). Save the file as UTF-8 and retry."
        ) from exc
    if not text:
        raise ValueError(
            f"content_file '{raw_path}' (resolved to '{resolved}') is "
            "empty -- a note body must not be blank."
        )
    return text


def resolve_remember_content(
    workspace_root: str,
    content: str | None,
    content_file: str | None,
    extra_roots: list[str] | None = None,
    additional_readable_roots: list[str] | None = None,
) -> str:
    """Resolve the note body for one `vectr_remember` call from exactly one
    of `content` (the literal body) or `content_file` (a path to read it
    from, resolved against `workspace_root`, `extra_roots`, or
    `additional_readable_roots` — see `resolve_content_file_path` for the
    multi-root rule). Mutually exclusive, exactly one required — both
    present or both absent raise `ValueError` with an actionable message.
    Shared by the MCP `vectr_remember` tool and the REST `/v1/remember`
    route so both enforce the identical rule.

    Returns the raw resolved body unmodified (no stripping) either way — see
    `read_content_file`'s docstring for why that is the caller's job, not
    this function's.
    """
    has_content = bool(content and content.strip())
    has_content_file = bool(content_file and content_file.strip())
    if has_content and has_content_file:
        raise ValueError(
            "content and content_file are mutually exclusive -- pass "
            "exactly one, not both."
        )
    if has_content_file:
        return read_content_file(
            workspace_root,
            content_file.strip(),
            extra_roots,
            additional_readable_roots,
        )
    if has_content:
        return content
    raise ValueError(
        "content or content_file is required -- pass the note body "
        "directly via content, or a path to a file containing it via "
        "content_file (use content_file for long, escape-dense, code-heavy "
        "bodies over ~2KB)."
    )
