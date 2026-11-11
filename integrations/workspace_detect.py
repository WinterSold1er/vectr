"""Workspace root detection, .gitignore / .vectrignore pattern parsing."""
from __future__ import annotations

import fnmatch
import logging
import re
from pathlib import Path

from agent.chunk_quality import is_generated_file, is_vectr_config_file, is_build_artifact_file
from agent.indexer import LANG_BY_EXT

logger = logging.getLogger(__name__)

_SUPPORTED_EXTS = set(LANG_BY_EXT.keys())

_ALWAYS_SKIP = {
    ".git", ".svn", "__pycache__", "node_modules", ".venv", "venv",
    "dist", "build", ".next", ".nuxt", "target", "coverage",
}

# UPG-EXCLUDE-REGEX: a .vectrignore line prefixed with this is a regex matched
# against the workspace-relative POSIX path of each candidate file, rather
# than a bare directory name or a filename glob. e.g. "re:legacy/.*" or
# "re:.*_(backup|old|copy)\\.\\w+$".
VECTRIGNORE_REGEX_PREFIX = "re:"


def find_workspace_root(start_path: str) -> str:
    """Walk up from start_path looking for a .git directory. Falls back to start_path."""
    p = Path(start_path).resolve()
    if p.is_file():
        p = p.parent
    for candidate in [p, *p.parents]:
        if (candidate / ".git").exists():
            return str(candidate)
    return str(p)


class WorkspaceEnvError(RuntimeError):
    """A workspace path (VECTR_WORKSPACE env var, or an equivalent explicit
    CLI argument) names a path that doesn't exist or isn't a directory
    (UPG-WORKSPACE-ENV-VALIDATE).

    Raised at daemon/service startup, before find_workspace_root's git
    walk-up ever runs, so a typo'd harness env fails loudly instead of
    silently falling back to cwd-based detection — silently indexing the
    wrong tree is worse than a startup crash naming the bad path.
    """


def validate_workspace_env(raw_path: str, *, env_var: str = "VECTR_WORKSPACE") -> None:
    """Raise WorkspaceEnvError if raw_path is set but isn't an existing directory.

    Called wherever a workspace path string sourced from an environment
    variable is about to be trusted (currently: VECTR_WORKSPACE, read by both
    api.py's daemon startup and main.py's CLI). An empty/falsy raw_path is a
    no-op — callers pass the already-defaulted value (e.g. "." for "not
    set"), and "." always resolves to the existing current directory, so the
    unset case is unaffected by construction.
    """
    if not raw_path:
        return
    p = Path(raw_path)
    if not p.exists():
        raise WorkspaceEnvError(
            f"{env_var}={raw_path!r} does not exist. Refusing to silently "
            f"fall back to the current directory — fix or unset {env_var}."
        )
    if not p.is_dir():
        raise WorkspaceEnvError(
            f"{env_var}={raw_path!r} is not a directory. Refusing to silently "
            f"fall back to the current directory — fix or unset {env_var}."
        )


def _read_ignore_lines(path: Path) -> list[str]:
    """Read non-blank, non-comment lines from an ignore file."""
    if not path.exists():
        return []
    lines = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


def get_gitignore_patterns(workspace_root: str) -> list[str]:
    """Read .gitignore and return a list of glob patterns."""
    return _read_ignore_lines(Path(workspace_root) / ".gitignore")


def _is_regex_pattern(entry: str) -> bool:
    """True if a .vectrignore entry is a `re:<pattern>` path regex (UPG-EXCLUDE-REGEX)."""
    return entry.startswith(VECTRIGNORE_REGEX_PREFIX)


def _is_glob_pattern(entry: str) -> bool:
    """True if a .vectrignore entry is a file glob rather than a bare dir name.

    Bare directory names (e.g. "vendor") are handled by get_vectrignore_dirs.
    Entries containing glob metacharacters (e.g. "*.generated.py") are handled
    by get_vectrignore_file_globs (UPG-13.3) — additive, so existing dir-name
    behaviour is unchanged for every entry that isn't a glob. A `re:` regex
    entry is never treated as a glob even if its pattern text contains glob
    metacharacters (UPG-EXCLUDE-REGEX) — it's routed to get_vectrignore_regexes
    instead.
    """
    if _is_regex_pattern(entry):
        return False
    return any(ch in entry for ch in "*?[")


def get_vectrignore_dirs(workspace_root: str) -> set[str]:
    """Read .vectrignore and return a set of directory names to exclude.

    .vectrignore format: one directory name, file glob (see
    get_vectrignore_file_globs), or `re:<pattern>` path regex (see
    get_vectrignore_regexes) per line, # comments supported.
    Example:
        # exclude vendor and generated code
        vendor
        generated
        proto-gen
        re:legacy/.*
    """
    return {
        line for line in _read_ignore_lines(Path(workspace_root) / ".vectrignore")
        if not _is_glob_pattern(line) and not _is_regex_pattern(line)
    }


def get_vectrignore_file_globs(workspace_root: str) -> list[str]:
    """Read .vectrignore and return file glob patterns (UPG-13.3).

    An entry is treated as a glob when it contains a glob metacharacter
    (*, ?, or [) — e.g. "*.generated.py" — rather than a bare directory name
    or a `re:` path regex. Additive layer on top of get_vectrignore_dirs:
    dir-name entries keep their existing directory-exclusion semantics
    unchanged; globs are matched against individual file names by callers
    (agent.watcher.CodeWatcher, should_index_file).
    """
    return [
        line for line in _read_ignore_lines(Path(workspace_root) / ".vectrignore")
        if _is_glob_pattern(line)
    ]


def get_vectrignore_regexes(workspace_root: str) -> list[re.Pattern]:
    """Read .vectrignore and compile `re:<pattern>` entries (UPG-EXCLUDE-REGEX).

    Each pattern is matched (via `.search`) against the workspace-relative
    POSIX path of a candidate file — e.g. "re:legacy/.*" excludes every file
    under a top-level legacy/ dir, "re:.*_(backup|old|copy)\\.\\w+$" excludes
    stale renamed files anywhere in the tree. Additive alongside bare
    directory names (get_vectrignore_dirs) and file globs
    (get_vectrignore_file_globs).

    A line whose pattern text fails to compile is logged as a single warning
    naming the bad line and skipped — a malformed regex in a hand-edited
    .vectrignore must never keep the daemon from starting.
    """
    regexes: list[re.Pattern] = []
    for line in _read_ignore_lines(Path(workspace_root) / ".vectrignore"):
        if not _is_regex_pattern(line):
            continue
        pattern_text = line[len(VECTRIGNORE_REGEX_PREFIX):]
        try:
            regexes.append(re.compile(pattern_text))
        except re.error as exc:
            logger.warning(
                "Skipping invalid regex in .vectrignore (%r): %s", line, exc
            )
    return regexes


def write_default_vectrignore(workspace_root: str) -> bool:
    """Seed a fresh .vectrignore with the standard non-indexable dirs (UPG-13.2).

    Only writes when no .vectrignore exists yet for this workspace_root — an
    existing file (even an empty one, or one with unrelated content) is never
    touched, so a user's hand-authored exclusions are never clobbered.
    Returns True if a file was written, False if one already existed.
    """
    from agent.config import WORKSPACE_DEFAULT_VECTRIGNORE_DIRS

    vectrignore = Path(workspace_root) / ".vectrignore"
    if vectrignore.exists():
        return False

    lines = [
        "# vectr default excludes — generated on first `vectr start`/`vectr init`.",
        "# vectr never overwrites this file once it exists; edit freely.",
        "# One entry per line, in any of three forms:",
        "#   a directory name           (e.g. vendor)",
        "#   a file glob                (e.g. *.generated.py)",
        "#   a `re:` path regex         (e.g. re:legacy/.*  or  re:.*_(backup|old|copy)\\.\\w+$)",
        "# `re:` patterns match against the workspace-relative path of each file.",
        "",
        *WORKSPACE_DEFAULT_VECTRIGNORE_DIRS,
        "",
    ]
    vectrignore.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def write_vectrignore(workspace_root: str, dirs: list[str]) -> None:
    """Append entries to .vectrignore, skipping duplicates.

    Entries are written verbatim, so this also accepts file globs and
    `re:<pattern>` path regexes (UPG-EXCLUDE-REGEX) alongside bare directory
    names — callers that accept `re:` entries from a user (e.g. the `--exclude`
    CLI flag) must validate them with re.compile before calling this, since
    this function performs no validation itself.
    """
    vectrignore = Path(workspace_root) / ".vectrignore"
    existing: set[str] = set()
    lines: list[str] = []

    if vectrignore.exists():
        raw = vectrignore.read_text(encoding="utf-8", errors="ignore")
        lines = raw.splitlines()
        existing = {l.strip() for l in lines if l.strip() and not l.strip().startswith("#")}

    new_dirs = [d for d in dirs if d not in existing]
    if not new_dirs:
        return

    if lines and lines[-1].strip():
        lines.append("")  # blank separator before new entries
    lines.extend(new_dirs)
    vectrignore.write_text("\n".join(lines) + "\n", encoding="utf-8")


def matches_gitignore_pattern(path: Path, gitignore_patterns: list[str]) -> bool:
    """True if `path` matches any of the given raw .gitignore/.vectrignore-glob
    lines — matched against the bare filename, the path string as given, and
    (for a directory-only entry ending in "/") any path component.

    Extracted as a standalone predicate so every caller that needs to honor
    ignore-file patterns (the bulk workspace walk in `should_index_file`
    below, and the live file-watcher's per-event exclusion check) shares
    exactly one matching implementation — a pattern added to a project's
    .gitignore is honored identically whether a file appears via a full
    reindex or a live create/modify/delete/move event.
    """
    name = path.name
    rel = str(path)
    for pattern in gitignore_patterns:
        if fnmatch.fnmatch(name, pattern):
            return True
        if fnmatch.fnmatch(rel, pattern):
            return True
        if pattern.endswith("/"):
            if pattern.rstrip("/") in path.parts:
                return True
    return False


def should_index_file(
    file_path: str,
    gitignore_patterns: list[str],
    extra_excluded_dirs: set[str] | None = None,
    workspace_root: str | None = None,
    extra_excluded_regexes: list[re.Pattern] | None = None,
) -> bool:
    """Return True if the file should be indexed.

    Excluded directory NAMES are matched only against path components below
    workspace_root (when given) — a workspace that itself lives under an
    excluded-sounding prefix (e.g. /tmp/myproject or repo/tmp/fixture) must
    not have every one of its files excluded by its own absolute path.
    extra_excluded_regexes (UPG-EXCLUDE-REGEX, see get_vectrignore_regexes)
    are matched the same way — against the workspace-relative path — for the
    same reason.
    """
    path = Path(file_path)

    if path.suffix.lower() not in _SUPPORTED_EXTS:
        return False

    excluded = _ALWAYS_SKIP | (extra_excluded_dirs or set())

    rel_path = path
    if workspace_root:
        try:
            rel_path = path.resolve().relative_to(Path(workspace_root).resolve())
        except ValueError:
            pass  # file outside the root — fall back to the full path

    # UPG-1.3/UPG-15.9: never index vectr's own injected IDE-config files,
    # machine-generated/vendored files (lookup tables, protobuf, minified), or
    # files inside build-artifact directories (*.egg-info, *.dist-info).
    # UPG-RAGAS-WORKTREE-GUARD: the matchers scan every component OF THE PATH
    # THEY ARE GIVEN, so handing them the absolute path let an ANCESTOR of the
    # workspace decide exclusion — a checkout living under a directory named
    # `generated`/`node_modules`, `mypkg.egg-info`, `.vscode`, etc. silently
    # lost its entire tree even though nothing below the workspace root
    # matches. Feed them the workspace-relative path for exactly the reason
    # the ignore-dir loop right below is relative-only. When relativization
    # fails (file outside the root) we still fall back to the full path,
    # matching the loop's behavior.
    if is_vectr_config_file(str(rel_path)) or is_generated_file(str(rel_path)) or is_build_artifact_file(str(rel_path)):
        return False

    for part in rel_path.parts:
        if part in excluded:
            return False

    if extra_excluded_regexes:
        rel_posix = rel_path.as_posix()
        for regex in extra_excluded_regexes:
            if regex.search(rel_posix):
                return False

    if matches_gitignore_pattern(rel_path, gitignore_patterns):
        return False

    return True
